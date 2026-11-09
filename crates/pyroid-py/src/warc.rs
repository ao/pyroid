//! `pyroid._pyroid.warc`: the native side of `pyroid.warc` (warcio 1.8.2's reader, ported in
//! `pyroid_core::warc`).
//!
//! `python/pyroid/warc/` owns the Python API: warcio's classes and signatures, and the Python
//! steps of datatrove's `process_record`. This module runs the core reader and builds the
//! Python objects.
//!
//! Sources: a Python file object is read with the interpreter attached, one `fh.read(n)` per
//! core read and `fh.tell()` where warcio calls it; a path is opened here and read detached.
//! Each reader's state sits behind a `Mutex` locked with `lock_py_attached` (detached while it
//! waits), so a thread waiting for a reader never blocks the interpreter. A file object (or a
//! record or headers class) that calls back into its own reader while the reader is in a call
//! gets a `RuntimeError` instead of a deadlock. Lock order: a reader's state, then a content
//! stream's decoder, then (inside the core) a record's raw stream.
//!
//! The messages the core queues (warcio's `sys.stderr` writes and `warcio.recordloader` log
//! warnings) are taken after every call and emitted in order once the lock is released, before
//! the call returns or raises.

use std::any::Any;
use std::ffi::CString;
use std::fs::File;
use std::io;
use std::ops::{Deref, DerefMut};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use pyo3::exceptions::{
    PyAttributeError, PyEOFError, PyIndexError, PyKeyError, PyOSError, PyOverflowError,
    PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyCapsule, PyDict, PyInt, PyList, PyString, PyTuple};
use pyo3::{PyTraverseError, PyVisit, intern};
use pyroid_core::columnar::{StringBuffers, bitmap};
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::warc::gzipfile::{GzipFile, GzipParams};
use pyroid_core::warc::{
    self as core, ArchiveReader, ContentStream as CoreStream, DigestKind, Doc,
    DocReader as CoreDocs, EofPayload, Error, Format, Message, PyExc, PyIntValue, RawHandle,
    ReadSource, ReaderOptions, Record, Rows, Source, StatusAndHeaders, VarColumn,
};

use crate::arrow::{
    Chunk, Column, ColumnType, Field, RecordBatch, array_capsules, schema_capsule, stream_capsule,
};
use crate::errors::type_name;

// ---- a mutex that refuses re-entry -------------------------------------------------------

/// A `Mutex` locked without blocking the interpreter, which raises `RuntimeError` when the
/// thread that holds it tries to lock it again (a file object calling back into its reader).
#[derive(Debug)]
struct Locked<T> {
    inner: Mutex<T>,
    /// [`thread_key`] of the holder, 0 when free.
    owner: AtomicUsize,
}

#[derive(Debug)]
struct Guard<'a, T> {
    guard: MutexGuard<'a, T>,
    owner: &'a AtomicUsize,
}

impl<T> Deref for Guard<'_, T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.guard
    }
}

impl<T> DerefMut for Guard<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        &mut self.guard
    }
}

impl<T> Drop for Guard<'_, T> {
    fn drop(&mut self) {
        // Before the mutex unlocks (fields drop after this).
        self.owner.store(0, Ordering::Release);
    }
}

/// A per-thread nonzero value: the address of a thread-local.
fn thread_key() -> usize {
    thread_local! {
        static KEY: u8 = const { 0 };
    }
    KEY.with(|k| std::ptr::from_ref(k).addr())
}

impl<T> Locked<T> {
    fn new(value: T) -> Self {
        Self {
            inner: Mutex::new(value),
            owner: AtomicUsize::new(0),
        }
    }

    fn lock(&self, py: Python<'_>) -> PyResult<Guard<'_, T>> {
        let me = thread_key();
        if self.owner.load(Ordering::Acquire) == me {
            return Err(PyRuntimeError::new_err(
                "a pyroid.warc reader was called from inside one of its own calls \
                 (its file object's read() or tell(), or a record or headers class)",
            ));
        }
        let guard = self
            .inner
            .lock_py_attached(py)
            .unwrap_or_else(PoisonError::into_inner);
        self.owner.store(me, Ordering::Release);
        Ok(Guard {
            guard,
            owner: &self.owner,
        })
    }

    /// For the garbage collector, which must not block.
    fn try_lock(&self) -> Option<MutexGuard<'_, T>> {
        match self.inner.try_lock() {
            Ok(g) => Some(g),
            Err(std::sync::TryLockError::Poisoned(p)) => Some(p.into_inner()),
            Err(std::sync::TryLockError::WouldBlock) => None,
        }
    }
}

fn lock_plain<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---- sources -----------------------------------------------------------------------------

/// A Python binary file object.
#[derive(Debug)]
struct PyFile {
    fh: Py<PyAny>,
    /// `fh.tell()` worked in the constructor (else warcio's `UnseekableYetTellable`).
    tellable: bool,
    /// A `block_size` outside `i64`, passed to `fh.read` as given.
    block: Option<Py<PyAny>>,
}

/// What a reader reads.
#[derive(Debug)]
enum Src {
    /// A path, opened here (read detached).
    File(ReadSource<File>),
    /// A path read through the `gzip.GzipFile` emulation (datatrove's `.gz` files).
    FileGz(ReadSource<File>, Box<GzipFile>),
    /// A Python binary file object.
    Py(PyFile),
    /// A Python file object read through the `gzip.GzipFile` emulation.
    PyGz(PyFile, Box<GzipFile>),
    /// Closed by `raw_stream.close()`: reads raise `ValueError(msg)`.
    Closed(&'static str),
}

/// `fh.read(n)` / `fh.tell()` on a Python file object.
struct PySource<'a, 'py> {
    fh: &'a Bound<'py, PyAny>,
    tellable: bool,
    block: Option<&'a Bound<'py, PyAny>>,
}

/// The `n` the core passes for a `block_size` past `i64::MAX`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // i64::MAX, as the core converts it
const HUGE_BLOCK: usize = i64::MAX as usize;

fn source_error(py: Python<'_>, e: PyErr) -> Error {
    if e.is_instance_of::<PyEOFError>(py) {
        Error::Source(Box::new(EofPayload(Box::new(e))))
    } else {
        Error::Source(Box::new(e))
    }
}

/// Append what `fh.read()` returned (bytes, any buffer, or None for nothing) to `out`.
fn append_read(data: &Bound<'_, PyAny>, out: &mut Vec<u8>) -> PyResult<()> {
    if let Ok(b) = data.cast::<PyBytes>() {
        out.extend_from_slice(b.as_bytes());
        return Ok(());
    }
    if data.is_none() {
        return Ok(());
    }
    if data.is_instance_of::<PyString>() || data.is_instance_of::<PyInt>() {
        return Err(PyTypeError::new_err(format!(
            "a bytes-like object is required, not '{}'",
            data.get_type().name()?
        )));
    }
    let b = data.py().get_type::<PyBytes>().call1((data,))?;
    out.extend_from_slice(b.cast::<PyBytes>()?.as_bytes());
    Ok(())
}

impl Source for PySource<'_, '_> {
    fn read(&mut self, n: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        let py = self.fh.py();
        let read = intern!(py, "read");
        let r = match self.block {
            Some(b) if n == HUGE_BLOCK || n == usize::MAX => self.fh.call_method1(read, (b,)),
            _ if n == usize::MAX => self.fh.call_method1(read, (-1i64,)),
            _ => self.fh.call_method1(read, (n,)),
        };
        r.and_then(|data| append_read(&data, out))
            .map_err(|e| source_error(py, e))
    }

    fn tell(&mut self) -> Result<Option<i64>, Error> {
        if !self.tellable {
            return Ok(None);
        }
        let py = self.fh.py();
        self.fh
            .call_method0(intern!(py, "tell"))
            .and_then(|t| t.extract::<i64>())
            .map(Some)
            .map_err(|e| source_error(py, e))
    }
}

/// Reads after `raw_stream.close()` on a file pyroid opened.
struct ClosedSource(&'static str);

impl Source for ClosedSource {
    fn read(&mut self, _n: usize, _out: &mut Vec<u8>) -> Result<(), Error> {
        Err(Error::Source(Box::new(PyValueError::new_err(self.0))))
    }

    fn tell(&mut self) -> Result<Option<i64>, Error> {
        Err(Error::Source(Box::new(PyValueError::new_err(
            "I/O operation on closed file.",
        ))))
    }
}

/// Run `f` on the source: detached for a file pyroid opened, attached for a Python one.
fn with_src<R: Send>(
    py: Python<'_>,
    src: &mut Src,
    f: impl FnOnce(&mut dyn Source) -> R + Send,
) -> R {
    match src {
        Src::File(file) => py.detach(move || f(file)),
        Src::FileGz(file, gz) => py.detach(move || f(&mut gz.source(file))),
        Src::Py(p) => {
            let fh = p.fh.bind(py);
            let block = p.block.as_ref().map(|b| b.bind(py));
            f(&mut PySource {
                fh,
                tellable: p.tellable,
                block,
            })
        }
        Src::PyGz(p, gz) => {
            let fh = p.fh.bind(py);
            let mut s = PySource {
                fh,
                tellable: false,
                block: None,
            };
            f(&mut gz.source(&mut s))
        }
        Src::Closed(msg) => {
            let msg: &'static str = msg;
            f(&mut ClosedSource(msg))
        }
    }
}

/// `raw_stream.close()`'s `self.stream.close()`: the Python file object's `close()`, or mark
/// a file pyroid opened as closed.
fn close_src(py: Python<'_>, src: &mut Src) -> PyResult<()> {
    match src {
        Src::Py(p) => {
            p.fh.bind(py).call_method0(intern!(py, "close"))?;
        }
        // `GzipFile(fileobj=f).close()` leaves `f` open, and reads raise.
        Src::FileGz(..) | Src::PyGz(..) => *src = Src::Closed("I/O operation on closed file."),
        Src::File(_) => *src = Src::Closed("read of closed file"),
        Src::Closed(_) => {}
    }
    Ok(())
}

impl Src {
    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Src::Py(p) | Src::PyGz(p, _) = self {
            visit.call(&p.fh)?;
            visit.call(p.block.as_ref())?;
        }
        Ok(())
    }

    fn fh(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        match self {
            Src::Py(p) | Src::PyGz(p, _) => Some(p.fh.clone_ref(py)),
            _ => None,
        }
    }
}

/// `OSError(errno, os.strerror(errno), path)` for a failed open (Python picks the subclass).
fn open_error(py: Python<'_>, e: &io::Error, path: &Bound<'_, PyAny>) -> PyErr {
    let Some(code) = e.raw_os_error() else {
        return PyOSError::new_err(e.to_string());
    };
    let strerror = py
        .import("os")
        .and_then(|os| os.call_method1("strerror", (code,)))
        .and_then(|s| s.extract::<String>())
        .unwrap_or_else(|_| e.to_string());
    PyOSError::new_err((code, strerror, path.clone().unbind()))
}

/// Open a path (detached).
fn open_path(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<File> {
    let path: PathBuf = source.extract()?;
    py.detach(|| File::open(&path))
        .map_err(|e| open_error(py, &e, source))
}

// ---- options and interpreter facts --------------------------------------------------------

/// What the Python layer measured about the running interpreter: brotlicffi usable (as
/// warcio's `try_brotli_init`), `unicodedata.unidata_version`, `base64.b64decode('AA==AA') ==
/// b'\x00'`, whether `zlib.decompressobj().flush()` raises on a broken stream, and
/// `sys.version_info.minor` (for the `gzip.GzipFile` emulation).
#[derive(Debug, FromPyObject)]
struct Facts(bool, String, bool, bool, u32);

impl Facts {
    fn options(&self, int_max_str_digits: usize) -> ReaderOptions {
        ReaderOptions {
            brotli: self
                .0
                .then_some(core::brotli::brotli_decompressor as core::DecompressorFactory),
            unicode: UnicodeVersion::from_unidata_version(&self.1)
                .unwrap_or(UnicodeVersion::LATEST),
            int_max_str_digits,
            base64_legacy_padding: self.2,
            zlib_flush_raises: self.3,
            ..ReaderOptions::default()
        }
    }

    fn gzip(&self) -> PyResult<GzipParams> {
        GzipParams::for_python(self.4).ok_or_else(|| {
            PyValueError::new_err(format!(
                "the gzip.GzipFile emulation needs Python 3.11 or later, not 3.{}",
                self.4
            ))
        })
    }
}

/// The classes errors and messages go to, and the running interpreter's `base64.b16decode`
/// messages.
#[derive(Debug)]
struct ErrClasses {
    archive_load_failed: Py<PyAny>,
    parser_exception: Py<PyAny>,
    logger: Py<PyAny>,
    /// `b16decode`'s "Non-base16 digit found" and "Odd-length string" as this Python words
    /// them (3.15 says "Non-hexadecimal digit found", "Odd number of hexadecimal digits").
    b16_messages: (String, String),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ErrClasses {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (archive_load_failed, parser_exception, logger, b16_messages) = ob.extract()?;
        Ok(Self {
            archive_load_failed,
            parser_exception,
            logger,
            b16_messages,
        })
    }
}

impl ErrClasses {
    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.archive_load_failed)?;
        visit.call(&self.parser_exception)?;
        visit.call(&self.logger)
    }

    /// The Python exception for a core error.
    fn to_pyerr(&self, py: Python<'_>, e: Error, statusline: Option<&str>) -> PyErr {
        match e {
            Error::Source(payload) => source_payload(payload),
            Error::Py(kind, msg) => self.py_exc(py, kind, msg, statusline).unwrap_or_else(|e| e),
        }
    }

    fn py_exc(
        &self,
        py: Python<'_>,
        kind: PyExc,
        msg: String,
        statusline: Option<&str>,
    ) -> PyResult<PyErr> {
        Ok(match kind {
            PyExc::ArchiveLoadFailed => {
                PyErr::from_value(self.archive_load_failed.bind(py).call1((msg,))?)
            }
            PyExc::EofError if msg.is_empty() => PyEOFError::new_err(()),
            PyExc::EofError => PyEOFError::new_err(msg),
            PyExc::BadGzipFile => module_exc(py, "gzip", "BadGzipFile", msg)?,
            PyExc::ZlibError => module_exc(py, "zlib", "error", msg)?,
            PyExc::BinasciiError => {
                // The core words them as CPython 3.11 to 3.14 do.
                let msg = match msg.as_str() {
                    "Non-base16 digit found" => self.b16_messages.0.clone(),
                    "Odd-length string" => self.b16_messages.1.clone(),
                    _ => msg,
                };
                module_exc(py, "binascii", "Error", msg)?
            }
            PyExc::BrotliError => {
                // brotlicffi raises decoding errors with a bytes argument: `str(e)` is its repr.
                let cls = py.import("brotlicffi")?.getattr("error")?;
                let value = match msg.strip_prefix("b'").and_then(|m| m.strip_suffix('\'')) {
                    Some(inner) => cls.call1((PyBytes::new(py, inner.as_bytes()),))?,
                    None => cls.call1((msg,))?,
                };
                PyErr::from_value(value)
            }
            PyExc::AttributeError => PyAttributeError::new_err(msg),
            PyExc::KeyError => PyKeyError::new_err(key_of(&msg).to_owned()),
            PyExc::ValueError => PyValueError::new_err(msg),
            PyExc::OsError => PyOSError::new_err(msg),
            PyExc::TypeError => PyTypeError::new_err(msg),
            PyExc::OverflowError => PyOverflowError::new_err(msg),
            PyExc::StatusAndHeadersParserException => PyErr::from_value(
                self.parser_exception
                    .bind(py)
                    .call1((msg, statusline.unwrap_or_default()))?,
            ),
        })
    }
}

/// `module.name(msg)` as an exception.
fn module_exc(py: Python<'_>, module: &str, name: &str, msg: String) -> PyResult<PyErr> {
    let cls = py.import(module)?.getattr(name)?;
    Ok(PyErr::from_value(cls.call1((msg,))?))
}

/// The key of a `KeyError` whose `str()` is `msg` (`'uri'` -> `uri`).
fn key_of(msg: &str) -> &str {
    msg.strip_prefix('\'')
        .and_then(|m| m.strip_suffix('\''))
        .unwrap_or(msg)
}

/// A source's own exception, unchanged.
fn source_payload(payload: Box<dyn Any + Send>) -> PyErr {
    let payload = match payload.downcast::<EofPayload>() {
        Ok(eof) => eof.0,
        Err(p) => p,
    };
    match payload.downcast::<PyErr>() {
        Ok(e) => *e,
        Err(p) => match p.downcast::<io::Error>() {
            Ok(e) => PyErr::from(*e),
            Err(_) => PyRuntimeError::new_err("pyroid.warc: a source failed without an exception"),
        },
    }
}

/// Emit the core's messages: `sys.stderr.write(text)` / `logger.warning(text)`.
fn emit(py: Python<'_>, logger: &Py<PyAny>, msgs: Vec<Message>) -> PyResult<()> {
    for m in msgs {
        match m {
            Message::Stderr(text) => {
                py.import("sys")?
                    .getattr(intern!(py, "stderr"))?
                    .call_method1(intern!(py, "write"), (text,))?;
            }
            Message::LogWarning(text) => {
                logger
                    .bind(py)
                    .call_method1(intern!(py, "warning"), (text,))?;
            }
        }
    }
    Ok(())
}

/// Emit `msgs`, then return `r` (or the error emitting raised).
fn finish<T>(
    py: Python<'_>,
    logger: &Py<PyAny>,
    msgs: Vec<Message>,
    r: PyResult<T>,
) -> PyResult<T> {
    emit(py, logger, msgs)?;
    r
}

/// Messages as `[(is_log, text), ...]` for the Python layer to emit later.
fn messages_list(py: Python<'_>, msgs: Vec<Message>) -> PyResult<Bound<'_, PyList>> {
    PyList::new(
        py,
        msgs.into_iter().map(|m| match m {
            Message::Stderr(t) => (false, t),
            Message::LogWarning(t) => (true, t),
        }),
    )
}

/// A Python `int`.
fn py_int<'py>(py: Python<'py>, v: &PyIntValue) -> PyResult<Bound<'py, PyAny>> {
    match v {
        PyIntValue::Int(i) => Ok(i.into_pyobject(py)?.into_any()),
        PyIntValue::Big(s) => py.get_type::<PyInt>().call1((s.as_str(),)),
    }
}

/// A `read(length)` argument: `None`, or an int (saturated past `i128`).
fn read_len(length: Option<&Bound<'_, PyAny>>) -> PyResult<Option<i128>> {
    let Some(l) = length.filter(|l| !l.is_none()) else {
        return Ok(None);
    };
    if let Ok(v) = l.extract::<i64>() {
        return Ok(Some(i128::from(v)));
    }
    if let Ok(v) = l.extract::<i128>() {
        return Ok(Some(v));
    }
    if l.is_instance_of::<PyInt>() {
        return Ok(Some(if l.lt(0)? { i128::MIN } else { i128::MAX }));
    }
    Err(PyTypeError::new_err(format!(
        "argument should be integer or None, not '{}'",
        l.get_type().name()?
    )))
}

/// The record and headers classes `ArchiveIterator` takes from its class attributes.
#[derive(Debug)]
struct Classes {
    record: Py<PyAny>,
    /// `record` is `ArcWarcRecord._from_native` (positional, no `__init__`).
    record_fast: bool,
    headers: Py<PyAny>,
    /// `headers` is `StatusAndHeaders._from_native`.
    headers_fast: bool,
    digest_checker: Py<PyAny>,
    /// The `check_digests` argument as given (`DigestChecker(kind)`).
    check_digests: Py<PyAny>,
    err: ErrClasses,
}

impl<'a, 'py> FromPyObject<'a, 'py> for Classes {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let (record, record_fast, headers, headers_fast, digest_checker, check_digests, err) =
            ob.extract()?;
        Ok(Self {
            record,
            record_fast,
            headers,
            headers_fast,
            digest_checker,
            check_digests,
            err,
        })
    }
}

impl Classes {
    fn visit(&self, visit: &PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.record)?;
        visit.call(&self.headers)?;
        visit.call(&self.digest_checker)?;
        visit.call(&self.check_digests)?;
        self.err.visit(visit)
    }

    /// `StatusAndHeaders(statusline=..., headers=[...], protocol=..., total_len=...)`.
    fn make_headers<'py>(
        &self,
        py: Python<'py>,
        h: &StatusAndHeaders,
    ) -> PyResult<Bound<'py, PyAny>> {
        let items = h
            .headers()
            .map(|(n, v)| PyTuple::new(py, [n, v]))
            .collect::<PyResult<Vec<_>>>()?;
        let list = PyList::new(py, items)?;
        let f = self.headers.bind(py);
        if self.headers_fast {
            return f.call1((
                h.statusline.as_str(),
                list,
                h.protocol.as_str(),
                h.total_len,
            ));
        }
        let kw = PyDict::new(py);
        kw.set_item(intern!(py, "statusline"), h.statusline.as_str())?;
        kw.set_item(intern!(py, "headers"), list)?;
        kw.set_item(intern!(py, "protocol"), h.protocol.as_str())?;
        kw.set_item(intern!(py, "total_len"), h.total_len)?;
        f.call((), Some(&kw))
    }
}

/// `DigestChecker` state into the Python checker: append the problems it lacks, set `_passed`.
fn sync_checker(py: Python<'_>, dc: &Py<PyAny>, raw: &RawHandle) -> PyResult<()> {
    let snap = lock_plain(raw).digest_checker().clone();
    let dc = dc.bind(py);
    let problems = dc.getattr(intern!(py, "_problem"))?;
    let have = problems.len()?;
    for p in snap.problems.iter().skip(have) {
        problems.call_method1(intern!(py, "append"), (p.as_str(),))?;
    }
    dc.setattr(intern!(py, "_passed"), snap.passed)
}

// ---- the iterator ------------------------------------------------------------------------

#[derive(Debug)]
struct ReaderState {
    reader: ArchiveReader,
    src: Src,
    /// `it.record` (while the core has one).
    record: Option<Py<PyAny>>,
    /// The current record's raw stream and Python digest checker (digests on only).
    current: Option<(RawHandle, Py<PyAny>)>,
}

impl ReaderState {
    fn sync_current(&self, py: Python<'_>) -> PyResult<()> {
        match &self.current {
            Some((raw, dc)) => sync_checker(py, dc, raw),
            None => Ok(()),
        }
    }
}

/// Which Python digest checkers a call may have changed.
enum Refresh<'a> {
    /// The iterator's current record (`next`, `read_to_end`).
    Current,
    /// The record whose stream was read.
    This(&'a RawHandle, &'a Py<PyAny>),
    Nothing,
}

/// `ArchiveIterator`'s state: `iter(pyroid.warc.ArchiveIterator(...))`.
#[pyclass(frozen, module = "pyroid.warc", name = "_Reader")]
#[derive(Debug)]
struct Reader {
    state: Locked<ReaderState>,
    classes: Classes,
    /// `check_digests` was true.
    digests: bool,
    opts: ReaderOptions,
}

impl Reader {
    /// Run a core call on the reader and its source; emit the messages it queued.
    fn run<R: Send>(
        &self,
        py: Python<'_>,
        sync: Refresh<'_>,
        f: impl FnOnce(&mut ArchiveReader, &mut dyn Source) -> Result<R, Error> + Send,
    ) -> PyResult<R> {
        let mut st = self.state.lock(py)?;
        let s = &mut *st;
        let r = with_src(py, &mut s.src, |src| f(&mut s.reader, src));
        let msgs = s.reader.take_messages();
        let mut r = r.map_err(|e| {
            self.classes
                .err
                .to_pyerr(py, e, s.reader.parser_exception_statusline())
        });
        if self.digests {
            let synced = match sync {
                Refresh::Current => s.sync_current(py),
                Refresh::This(raw, dc) => sync_checker(py, dc, raw),
                Refresh::Nothing => Ok(()),
            };
            if let (Ok(_), Err(e)) = (&r, synced) {
                r = Err(e);
            }
        }
        drop(st);
        finish(py, &self.classes.err.logger, msgs, r)
    }

    /// The Python record for a core record, and its digest checker.
    fn build_record<'py>(
        &self,
        owner: &Bound<'py, Self>,
        rec: &Record,
    ) -> PyResult<(Bound<'py, PyAny>, Py<PyAny>)> {
        let py = owner.py();
        let c = &self.classes;
        let rec_headers = c.make_headers(py, &rec.rec_headers)?;
        let http_headers = match &rec.http_headers {
            Some(h) => c.make_headers(py, h)?,
            None => py.None().into_bound(py),
        };
        let dc = c
            .digest_checker
            .bind(py)
            .call1((c.check_digests.bind(py),))?;
        if self.digests {
            sync_checker(py, &dc.clone().unbind(), &rec.raw)?;
        }
        let raw = Bound::new(
            py,
            RawStream {
                reader: owner.clone().unbind(),
                handle: Arc::clone(&rec.raw),
                dc: self.digests.then(|| dc.clone().unbind()),
            },
        )?;
        let format = match rec.format {
            Format::Warc => intern!(py, "warc"),
            Format::Arc => intern!(py, "arc"),
        };
        let length = match &rec.length {
            Some(l) => py_int(py, l)?,
            None => py.None().into_bound(py),
        };
        let payload_length = py_int(py, &rec.payload_length)?;
        let rec_type = rec.rec_type.as_deref();
        let content_type = rec.content_type.as_deref();
        let f = c.record.bind(py);
        let obj = if c.record_fast {
            f.call1((
                format,
                rec_type,
                rec_headers,
                raw,
                http_headers,
                content_type,
                length,
                payload_length,
                &dc,
            ))?
        } else {
            let kw = PyDict::new(py);
            kw.set_item(intern!(py, "payload_length"), payload_length)?;
            kw.set_item(intern!(py, "digest_checker"), &dc)?;
            f.call(
                (
                    format,
                    rec_type,
                    rec_headers,
                    raw,
                    http_headers,
                    content_type,
                    length,
                ),
                Some(&kw),
            )?
        };
        Ok((obj, dc.unbind()))
    }
}

/// `block_size`: an i64, or an int outside it (kept to pass to `fh.read` as is).
fn block_size(b: &Bound<'_, PyAny>) -> PyResult<(i64, Option<Py<PyAny>>)> {
    if let Ok(v) = b.extract::<i64>() {
        return Ok((v, None));
    }
    if b.is_instance_of::<PyInt>() {
        let v = if b.gt(0)? { i64::MAX } else { -1 };
        return Ok((v, Some(b.clone().unbind())));
    }
    Err(PyTypeError::new_err(format!(
        "'{}' object cannot be interpreted as an integer",
        b.get_type().name()?
    )))
}

/// `fh.tell()` as warcio's constructor calls it: `(tell, tellable)`.
fn initial_tell(fh: &Bound<'_, PyAny>) -> (i64, bool) {
    match fh
        .call_method0(intern!(fh.py(), "tell"))
        .and_then(|t| t.extract::<i64>())
    {
        Ok(t) => (t, true),
        Err(_) => (0, false),
    }
}

#[pymethods]
impl Reader {
    #[new]
    #[pyo3(signature = (source, is_path, flags, block_size, check_digests, known_format, classes, facts, int_max_str_digits))]
    #[allow(clippy::too_many_arguments)] // one per ArchiveIterator argument group
    fn new(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        is_path: bool,
        flags: (bool, bool, bool, bool),
        block_size: &Bound<'_, PyAny>,
        check_digests: u8,
        known_format: Option<&str>,
        classes: Classes,
        facts: Facts,
        int_max_str_digits: usize,
    ) -> PyResult<Self> {
        let (no_record_parse, verify_http, arc2warc, ensure_http_headers) = flags;
        let (block, huge) = self::block_size(block_size)?;
        let (src, tell) = if is_path {
            if huge.is_some() {
                return Err(PyOverflowError::new_err(
                    "Python int too large to convert to C ssize_t",
                ));
            }
            (Src::File(ReadSource(open_path(py, source)?)), 0)
        } else {
            let (tell, tellable) = initial_tell(source);
            let file = PyFile {
                fh: source.clone().unbind(),
                tellable,
                block: huge,
            };
            (Src::Py(file), tell)
        };
        let check = match check_digests {
            0 => DigestKind::Off,
            2 => DigestKind::Log,
            3 => DigestKind::Raise,
            _ => DigestKind::Record,
        };
        let known_format = match known_format {
            Some("warc") => Some(Format::Warc),
            Some("arc") => Some(Format::Arc),
            _ => None,
        };
        let opts = ReaderOptions {
            no_record_parse,
            verify_http,
            arc2warc,
            ensure_http_headers,
            block_size: block,
            check_digests: check,
            known_format,
            ..facts.options(int_max_str_digits)
        };
        Ok(Self {
            state: Locked::new(ReaderState {
                reader: ArchiveReader::new(opts.clone(), tell),
                src,
                record: None,
                current: None,
            }),
            classes,
            digests: check != DigestKind::Off,
            opts,
        })
    }

    fn __iter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __next__(slf: &Bound<'_, Self>) -> PyResult<Option<Py<PyAny>>> {
        let py = slf.py();
        let this = slf.get();
        let mut st = this.state.lock(py)?;
        let s = &mut *st;
        let r = with_src(py, &mut s.src, |src| s.reader.next_record(src));
        let msgs = s.reader.take_messages();
        // `next` ran `read_to_end` on the previous record first.
        let synced = if this.digests {
            s.sync_current(py)
        } else {
            Ok(())
        };
        let out = match (r, synced) {
            (Ok(Some(rec)), Ok(())) => this.build_record(slf, &rec).map(|(obj, dc)| {
                s.record = Some(obj.clone().unbind());
                if this.digests {
                    s.current = Some((Arc::clone(&rec.raw), dc));
                }
                Some(obj.unbind())
            }),
            (Ok(None), Ok(())) => Ok(None),
            (Err(e), _) => {
                Err(this
                    .classes
                    .err
                    .to_pyerr(py, e, s.reader.parser_exception_statusline()))
            }
            (Ok(_), Err(e)) => Err(e),
        };
        drop(st);
        finish(py, &this.classes.err.logger, msgs, out)
    }

    /// `ArchiveIterator.close()`.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        self.run(py, Refresh::Nothing, |rd, _| rd.close())
    }

    /// `ArchiveIterator.read_to_end()`.
    fn read_to_end(&self, py: Python<'_>) -> PyResult<()> {
        self.run(py, Refresh::Current, ArchiveReader::read_to_end)
    }

    /// `ArchiveIterator.get_record_offset()`.
    fn get_record_offset(&self, py: Python<'_>) -> PyResult<i64> {
        self.run(py, Refresh::Current, ArchiveReader::get_record_offset)
    }

    /// `ArchiveIterator.get_record_length()`.
    fn get_record_length(&self, py: Python<'_>) -> PyResult<i64> {
        self.run(py, Refresh::Current, ArchiveReader::get_record_length)
    }

    #[getter]
    fn offset(&self, py: Python<'_>) -> PyResult<i64> {
        Ok(self.state.lock(py)?.reader.offset())
    }

    #[getter]
    fn err_count(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(self.state.lock(py)?.reader.err_count())
    }

    #[getter]
    fn member_info(&self, py: Python<'_>) -> PyResult<Option<(i64, i64)>> {
        Ok(self.state.lock(py)?.reader.member_info())
    }

    #[getter]
    fn known_format(&self, py: Python<'_>) -> PyResult<Option<&'static str>> {
        Ok(self
            .state
            .lock(py)?
            .reader
            .known_format()
            .map(Format::as_str))
    }

    #[getter]
    fn record(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let st = self.state.lock(py)?;
        Ok(if st.reader.has_record() {
            st.record.as_ref().map(|r| r.clone_ref(py))
        } else {
            None
        })
    }

    /// `it.reader is None` (after `close()`).
    #[getter]
    fn closed(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(self.state.lock(py)?.reader.is_closed())
    }

    #[getter]
    fn fh(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        Ok(self.state.lock(py)?.src.fh(py))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.classes.visit(&visit)?;
        if let Some(st) = self.state.try_lock() {
            st.src.visit(&visit)?;
            visit.call(st.record.as_ref())?;
            if let Some((_, dc)) = &st.current {
                visit.call(dc)?;
            }
        }
        Ok(())
    }

    fn __clear__(&self) {
        if let Some(mut st) = self.state.try_lock() {
            st.record = None;
            st.current = None;
            if matches!(st.src, Src::Py(_) | Src::PyGz(..)) {
                st.src = Src::Closed("read of closed file");
            }
        }
    }
}

// ---- record streams -----------------------------------------------------------------------

/// `record.raw_stream`: warcio's `LimitReader` (or `DigestVerifyingReader`, or the iterator's
/// own reader when the record has no length) over the iterator's file.
#[pyclass(frozen, module = "pyroid.warc", name = "RawStream")]
#[derive(Debug)]
struct RawStream {
    reader: Py<Reader>,
    handle: RawHandle,
    /// The record's digest checker (digests on only).
    dc: Option<Py<PyAny>>,
}

impl RawStream {
    fn sync(&self) -> Refresh<'_> {
        match &self.dc {
            Some(dc) => Refresh::This(&self.handle, dc),
            None => Refresh::Nothing,
        }
    }
}

#[pymethods]
impl RawStream {
    /// Read up to `length` bytes of the record (all of it when `None`).
    #[pyo3(signature = (length = None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        length: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let n = read_len(length)?;
        let mut out = Vec::new();
        let h = &self.handle;
        self.reader
            .get()
            .run(py, self.sync(), |rd, src| rd.raw_read(h, src, n, &mut out))?;
        Ok(PyBytes::new(py, &out))
    }

    /// Read a line (at most `length` bytes).
    #[pyo3(signature = (length = None))]
    fn readline<'py>(
        &self,
        py: Python<'py>,
        length: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let n = read_len(length)?;
        let mut out = Vec::new();
        let h = &self.handle;
        self.reader.get().run(py, self.sync(), |rd, src| {
            rd.raw_readline(h, src, n, &mut out)
        })?;
        Ok(PyBytes::new(py, &out))
    }

    /// The bytes of the record read so far (`LimitReader.tell()`).
    fn tell(&self, py: Python<'_>) -> PyResult<i64> {
        Ok(self
            .reader
            .get()
            .state
            .lock(py)?
            .reader
            .raw_tell(&self.handle))
    }

    /// Close the file object, then the decompressor (warcio's `close()`).
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let reader = self.reader.get();
        let mut st = reader.state.lock(py)?;
        let s = &mut *st;
        let (close_file, r) = s.reader.raw_close(&self.handle);
        let msgs = s.reader.take_messages();
        let closed = if close_file {
            close_src(py, &mut s.src)
        } else {
            Ok(())
        };
        let r = closed.and_then(|()| {
            r.map_err(|e| {
                reader
                    .classes
                    .err
                    .to_pyerr(py, e, s.reader.parser_exception_statusline())
            })
        });
        drop(st);
        finish(py, &reader.classes.err.logger, msgs, r)
    }

    /// `LimitReader.limit`: the bytes left (`None` when the record has no length).
    #[getter]
    fn limit(&self) -> Option<i64> {
        lock_plain(&self.handle).limit()
    }

    /// A decoder over this stream (`content_stream()`), or `None` where warcio returns the
    /// raw stream itself. `encoding` is already lower-cased and supported.
    #[pyo3(name = "_decoder", signature = (chunked, encoding))]
    fn decoder(
        &self,
        py: Python<'_>,
        chunked: bool,
        encoding: Option<&str>,
    ) -> Option<ContentStream> {
        let mut h = StatusAndHeaders::new("HTTP/1.1", "200 OK", 0);
        if let Some(e) = encoding {
            h.add_header("Content-Encoding", e);
        }
        if chunked {
            h.add_header("Transfer-Encoding", "chunked");
        }
        let rec = Record {
            format: Format::Warc,
            rec_type: None,
            rec_headers: StatusAndHeaders::default(),
            http_headers: Some(h),
            content_type: None,
            length: None,
            payload_length: PyIntValue::Int(-1),
            raw: Arc::clone(&self.handle),
        };
        match rec.content_stream(&self.reader.get().opts) {
            CoreStream::Raw(_) => None,
            cs @ CoreStream::Decoder(_) => Some(ContentStream {
                reader: self.reader.clone_ref(py),
                cs: Mutex::new(cs),
                handle: Arc::clone(&self.handle),
                dc: self.dc.as_ref().map(|d| d.clone_ref(py)),
            }),
        }
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reader)?;
        visit.call(self.dc.as_ref())
    }
}

/// What `record.content_stream()` returns when it decodes: warcio's `ChunkedDataReader` or
/// `BufferedReader` with a decompressor, over the record's raw stream.
#[pyclass(frozen, module = "pyroid.warc", name = "ContentStream")]
#[derive(Debug)]
struct ContentStream {
    reader: Py<Reader>,
    cs: Mutex<CoreStream>,
    handle: RawHandle,
    dc: Option<Py<PyAny>>,
}

impl ContentStream {
    fn sync(&self) -> Refresh<'_> {
        match &self.dc {
            Some(dc) => Refresh::This(&self.handle, dc),
            None => Refresh::Nothing,
        }
    }
}

#[pymethods]
impl ContentStream {
    /// Read up to `length` decoded bytes (all of them when `None`).
    #[pyo3(signature = (length = None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        length: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let n = read_len(length)?;
        let mut out = Vec::new();
        let cs = &self.cs;
        self.reader.get().run(py, self.sync(), |rd, src| {
            rd.stream_read(&mut lock_plain(cs), src, n, &mut out)
        })?;
        Ok(PyBytes::new(py, &out))
    }

    /// Read a decoded line (at most `length` bytes).
    #[pyo3(signature = (length = None))]
    fn readline<'py>(
        &self,
        py: Python<'py>,
        length: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let n = read_len(length)?;
        let mut out = Vec::new();
        let cs = &self.cs;
        self.reader.get().run(py, self.sync(), |rd, src| {
            rd.stream_readline(&mut lock_plain(cs), src, n, &mut out)
        })?;
        Ok(PyBytes::new(py, &out))
    }

    /// The decoded bytes buffered so far (`BufferedReader.tell()`).
    fn tell(&self, py: Python<'_>) -> PyResult<i64> {
        let st = self.reader.get().state.lock(py)?;
        Ok(st.reader.stream_tell(&lock_plain(&self.cs)))
    }

    /// Close the underlying stream (and the file object), then the decompressor.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let reader = self.reader.get();
        let mut st = reader.state.lock(py)?;
        let s = &mut *st;
        let (close_file, r) = s.reader.stream_close(&mut lock_plain(&self.cs));
        let msgs = s.reader.take_messages();
        let closed = if close_file {
            close_src(py, &mut s.src)
        } else {
            Ok(())
        };
        let r = closed.and_then(|()| {
            r.map_err(|e| {
                reader
                    .classes
                    .err
                    .to_pyerr(py, e, s.reader.parser_exception_statusline())
            })
        });
        drop(st);
        finish(py, &reader.classes.err.logger, msgs, r)
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(&self.reader)?;
        visit.call(self.dc.as_ref())
    }
}

// ---- the datatrove path ------------------------------------------------------------------

#[derive(Debug)]
struct DocState {
    docs: CoreDocs,
    src: Src,
    /// An error to raise at the next call (after the documents read before it).
    pending: Option<PyErr>,
    done: bool,
}

/// `for ri, record in enumerate(ArchiveIterator(f))` with `process_record`'s record-type and
/// payload-type checks: the documents `iter_documents` / `iter_batches` finish in Python.
#[pyclass(frozen, module = "pyroid.warc", name = "_DocReader")]
#[derive(Debug)]
struct DocReader {
    state: Locked<DocState>,
    err: ErrClasses,
}

/// A ready document `(ri, {"text", "id", "url", "date"})`, or one the Python layer finishes:
/// `(ri, conversion, mime_type, content, utf8, id, url, date, key_error)`.
fn doc_item<'py>(py: Python<'py>, doc: &Doc<'_>) -> PyResult<Bound<'py, PyAny>> {
    let key_error = doc.url.err().or(doc.date.err()).map(key_of);
    if doc.mime_type.is_some() && doc.utf8 && key_error.is_none() {
        let text = simdutf8::basic::from_utf8(doc.content)
            .map_err(|_| PyRuntimeError::new_err("pyroid.warc: content is not UTF-8"))?;
        let d = PyDict::new(py);
        d.set_item(intern!(py, "text"), PyString::new(py, text))?;
        d.set_item(intern!(py, "id"), doc.id)?;
        d.set_item(intern!(py, "url"), doc.url.ok())?;
        d.set_item(intern!(py, "date"), doc.date.ok())?;
        return Ok((doc.index, d).into_pyobject(py)?.into_any());
    }
    Ok((
        doc.index,
        doc.conversion,
        doc.mime_type,
        PyBytes::new(py, doc.content),
        doc.utf8,
        doc.id,
        doc.url.ok(),
        doc.date.ok(),
        key_error,
    )
        .into_pyobject(py)?
        .into_any())
}

#[pymethods]
impl DocReader {
    #[new]
    #[pyo3(signature = (source, is_path, gzip, err, facts, int_max_str_digits))]
    fn new(
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        is_path: bool,
        gzip: bool,
        err: ErrClasses,
        facts: Facts,
        int_max_str_digits: usize,
    ) -> PyResult<Self> {
        let (src, tell) = match (is_path, gzip) {
            (true, false) => (Src::File(ReadSource(open_path(py, source)?)), 0),
            (true, true) => {
                let gz = Box::new(GzipFile::new(facts.gzip()?));
                (Src::FileGz(ReadSource(open_path(py, source)?), gz), 0)
            }
            (false, true) => {
                let gz = Box::new(GzipFile::new(facts.gzip()?));
                let file = PyFile {
                    fh: source.clone().unbind(),
                    tellable: false,
                    block: None,
                };
                (Src::PyGz(file, gz), 0)
            }
            (false, false) => {
                let (tell, tellable) = initial_tell(source);
                let file = PyFile {
                    fh: source.clone().unbind(),
                    tellable,
                    block: None,
                };
                (Src::Py(file), tell)
            }
        };
        Ok(Self {
            state: Locked::new(DocState {
                docs: CoreDocs::new(facts.options(int_max_str_digits), tell),
                src,
                pending: None,
                done: false,
            }),
            err,
        })
    }

    /// Up to `max_docs` documents (or `max_bytes` of content): a list of ready documents,
    /// documents to finish, and message lists (each before the document it belongs to);
    /// `None` at the end. An error comes at the call after the documents before it.
    fn next_batch<'py>(
        &self,
        py: Python<'py>,
        max_docs: usize,
        max_bytes: usize,
    ) -> PyResult<Option<Bound<'py, PyList>>> {
        let mut st = self.state.lock(py)?;
        if let Some(e) = st.pending.take() {
            return Err(e);
        }
        if st.done {
            return Ok(None);
        }
        let out = PyList::empty(py);
        let (mut docs, mut bytes) = (0usize, 0usize);
        loop {
            let s = &mut *st;
            let r = with_src(py, &mut s.src, |src| s.docs.next_doc(src));
            let item = match r {
                Ok(Some(doc)) => {
                    bytes += doc.content.len();
                    Some(doc_item(py, &doc)?)
                }
                Ok(None) => {
                    s.done = true;
                    None
                }
                Err(e) => {
                    let statusline = s.docs.reader().parser_exception_statusline();
                    s.pending = Some(self.err.to_pyerr(py, e, statusline));
                    s.done = true;
                    None
                }
            };
            let msgs = s.docs.reader().take_messages();
            if !msgs.is_empty() {
                out.append(messages_list(py, msgs)?)?;
            }
            match item {
                Some(item) => out.append(item)?,
                None => break,
            }
            docs += 1;
            if docs >= max_docs || bytes >= max_bytes {
                break;
            }
        }
        Ok(Some(out))
    }

    /// Up to `max_rows` documents as columns, read in one call (detached for a path); `None`
    /// at the end. An error comes at the call after the rows before it.
    fn next_rows(&self, py: Python<'_>, max_rows: usize) -> PyResult<Option<RowBatch>> {
        let mut st = self.state.lock(py)?;
        if let Some(e) = st.pending.take() {
            return Err(e);
        }
        if st.done {
            return Ok(None);
        }
        let s = &mut *st;
        let docs = &mut s.docs;
        let (rows, msgs, r) = with_src(py, &mut s.src, move |src| {
            let mut rows = Rows::default();
            let mut msgs = Vec::new();
            loop {
                let r = match docs.next_doc(src) {
                    Ok(Some(doc)) => {
                        rows.push(&doc);
                        Ok(true)
                    }
                    Ok(None) => Ok(false),
                    Err(e) => Err(e),
                };
                msgs.extend(docs.reader().take_messages());
                match r {
                    Ok(true) if rows.len() < max_rows => {}
                    Ok(true) => return (rows, msgs, Ok(true)),
                    other => return (rows, msgs, other),
                }
            }
        });
        match r {
            Ok(true) => {}
            Ok(false) => s.done = true,
            Err(e) => {
                let statusline = s.docs.reader().parser_exception_statusline();
                s.pending = Some(self.err.to_pyerr(py, e, statusline));
                s.done = true;
            }
        }
        if rows.is_empty() && msgs.is_empty() && s.pending.is_none() {
            return Ok(None);
        }
        Ok(Some(RowBatch {
            rows: Mutex::new(Some(rows)),
            messages: messages_list(py, msgs)?.unbind(),
        }))
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.err.visit(&visit)?;
        if let Some(st) = self.state.try_lock() {
            st.src.visit(&visit)?;
        }
        Ok(())
    }

    fn __clear__(&self) {
        if let Some(mut st) = self.state.try_lock() {
            st.src = Src::Closed("read of closed file");
        }
    }
}

/// Rows read by `next_rows`, before the Python layer fills the rows that need `magic` or
/// `cchardet`.
#[pyclass(frozen, module = "pyroid.warc", name = "_RowBatch")]
#[derive(Debug)]
struct RowBatch {
    rows: Mutex<Option<Rows>>,
    messages: Py<PyList>,
}

fn var_buffers(c: VarColumn) -> StringBuffers<i64> {
    let len = c.len();
    let null_count = c.valid.iter().filter(|v| !**v).count();
    let validity = (null_count > 0).then(|| bitmap(c.valid.iter().copied(), len));
    StringBuffers {
        len,
        offsets: c.offsets,
        data: c.data,
        validity,
        null_count,
    }
}

/// Keep the first `n` rows of a column.
fn truncate_var(c: &mut VarColumn, n: usize) {
    if n < c.len() {
        c.valid.truncate(n);
        c.offsets.truncate(n + 1);
        c.data
            .truncate(usize::try_from(c.offsets[n]).expect("offsets are non-negative"));
    }
}

/// Keep the first `n` rows.
fn truncate_rows(r: &mut Rows, n: usize) {
    r.index.truncate(n);
    for c in [
        &mut r.id,
        &mut r.url,
        &mut r.date,
        &mut r.mime_type,
        &mut r.content_type,
        &mut r.html,
    ] {
        truncate_var(c, n);
    }
    r.utf8.truncate(n);
    r.conversion.truncate(n);
    r.key_error.truncate(n);
}

/// `pyroid.warc.Batch`'s column names, in order.
const COLUMNS: [&str; 7] = [
    "id",
    "url",
    "date",
    "mime_type",
    "content_type",
    "html",
    "text",
];

#[pymethods]
impl RowBatch {
    /// The messages queued while these rows were read: `[(is_log, text), ...]`.
    #[getter]
    fn messages(&self, py: Python<'_>) -> Py<PyList> {
        self.messages.clone_ref(py)
    }

    /// The rows whose text needs Python: `[(row, conversion, mime_type, content, utf8), ...]`
    /// for rows without a payload type (`magic`) or whose content is not UTF-8 (`cchardet`).
    fn pending<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let guard = lock_plain(&self.rows);
        let out = PyList::empty(py);
        let Some(rows) = guard.as_ref() else {
            return Ok(out);
        };
        for i in 0..rows.len() {
            let mime = rows.mime_type.get(i);
            if mime.is_some() && rows.utf8[i] {
                continue;
            }
            let mime = mime.map(|m| String::from_utf8_lossy(m).into_owned());
            let content = PyBytes::new(py, rows.html.get(i).unwrap_or_default());
            out.append((i, rows.conversion[i], mime, content, rows.utf8[i]))?;
        }
        Ok(out)
    }

    /// The `Batch` of the first `limit` rows (all when `None`), with `fills[row]` as the text
    /// of each pending row: `None` (datatrove skips it), `True` (the content, which is UTF-8)
    /// or a `str`.
    #[pyo3(signature = (fills, limit = None))]
    fn finish(&self, fills: &Bound<'_, PyDict>, limit: Option<usize>) -> PyResult<Batch> {
        let mut rows = lock_plain(&self.rows)
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("pyroid.warc: batch already finished"))?;
        if let Some(n) = limit {
            truncate_rows(&mut rows, n);
        }
        let mut text = VarColumn::default();
        for i in 0..rows.len() {
            if rows.mime_type.valid[i] && rows.utf8[i] {
                text.push(rows.html.get(i));
                continue;
            }
            match fills.get_item(i)? {
                Some(v) if v.is_instance_of::<PyString>() => {
                    text.push(Some(v.cast::<PyString>()?.to_str()?.as_bytes()));
                }
                Some(v) if v.is_truthy()? => text.push(rows.html.get(i)),
                _ => text.push(None),
            }
        }
        let Rows {
            id,
            url,
            date,
            mime_type,
            content_type,
            html,
            ..
        } = rows;
        let cols = [
            (ColumnType::LargeUtf8, Chunk::LargeUtf8(var_buffers(id))),
            (ColumnType::LargeUtf8, Chunk::LargeUtf8(var_buffers(url))),
            (ColumnType::LargeUtf8, Chunk::LargeUtf8(var_buffers(date))),
            (
                ColumnType::LargeUtf8,
                Chunk::LargeUtf8(var_buffers(mime_type)),
            ),
            (
                ColumnType::LargeUtf8,
                Chunk::LargeUtf8(var_buffers(content_type)),
            ),
            (
                ColumnType::LargeBinary,
                Chunk::LargeBinary(var_buffers(html)),
            ),
            (ColumnType::LargeUtf8, Chunk::LargeUtf8(var_buffers(text))),
        ];
        let fields = COLUMNS
            .iter()
            .zip(cols)
            .map(|(name, (ty, chunk))| Field {
                name: CString::new(*name).expect("no NUL"),
                ty,
                chunk: Arc::new(chunk),
            })
            .collect();
        Ok(Batch {
            inner: Arc::new(RecordBatch::new(fields)),
        })
    }
}

// ---- Arrow batches -----------------------------------------------------------------------

/// A batch of WARC documents as Arrow columns.
#[pyclass(frozen, module = "pyroid.warc", name = "Batch")]
#[derive(Debug)]
struct Batch {
    inner: Arc<RecordBatch>,
}

fn field_name(f: &Field) -> &str {
    f.name.to_str().unwrap_or_default()
}

/// The values of one chunk as Python objects (`str`, `bytes` or `None`).
fn chunk_values<'py>(py: Python<'py>, chunk: &Chunk) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    match chunk {
        Chunk::LargeUtf8(b) => {
            for i in 0..b.len {
                match b.get(i) {
                    Some(v) => out.append(PyString::new(
                        py,
                        std::str::from_utf8(v)
                            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?,
                    ))?,
                    None => out.append(py.None())?,
                }
            }
        }
        Chunk::LargeBinary(b) => {
            for i in 0..b.len {
                match b.get(i) {
                    Some(v) => out.append(PyBytes::new(py, v))?,
                    None => out.append(py.None())?,
                }
            }
        }
        Chunk::Utf8(_) | Chunk::Bool(_) => {
            return Err(PyRuntimeError::new_err(
                "pyroid.warc: unexpected column type",
            ));
        }
    }
    Ok(out)
}

#[pymethods]
impl Batch {
    /// Rows in the batch.
    #[getter]
    fn num_rows(&self) -> usize {
        self.inner.num_rows
    }

    /// Columns in the batch.
    #[getter]
    fn num_columns(&self) -> usize {
        self.inner.fields.len()
    }

    /// The column names, in order.
    #[getter]
    fn column_names(&self) -> Vec<&str> {
        self.inner.fields.iter().map(field_name).collect()
    }

    /// A column by name or position, as a `pyroid.Column` (no copy).
    fn column(&self, key: &Bound<'_, PyAny>) -> PyResult<Column> {
        let fields = &self.inner.fields;
        let field = if let Ok(name) = key.extract::<&str>() {
            fields
                .iter()
                .find(|f| field_name(f) == name)
                .ok_or_else(|| PyKeyError::new_err(name.to_owned()))?
        } else if let Ok(i) = key.extract::<isize>() {
            let n = isize::try_from(fields.len()).unwrap_or(isize::MAX);
            let j = if i < 0 { i + n } else { i };
            usize::try_from(j)
                .ok()
                .and_then(|j| fields.get(j))
                .ok_or_else(|| PyIndexError::new_err(format!("column index {i} out of range")))?
        } else {
            return Err(PyTypeError::new_err(format!(
                "column() takes a name or an index, not {}",
                type_name(key)
            )));
        };
        Ok(Column::from_shared(
            field.ty,
            field_name(field),
            vec![Arc::clone(&field.chunk)],
        ))
    }

    /// The batch as `{name: [values]}`.
    fn to_pydict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for f in &self.inner.fields {
            d.set_item(field_name(f), chunk_values(py, &f.chunk)?)?;
        }
        Ok(d)
    }

    fn __len__(&self) -> usize {
        self.inner.num_rows
    }

    /// Export the schema (a struct of the columns; Arrow PyCapsule interface).
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        schema_capsule(py, &self.inner)
    }

    /// Export as one Arrow struct array (a record batch). ``requested_schema`` is ignored.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_array__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = requested_schema;
        array_capsules(py, &self.inner)
    }

    /// Export as an Arrow C stream of this one batch. ``requested_schema`` is ignored.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = requested_schema;
        stream_capsule(py, &self.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "pyroid.warc.Batch(num_rows={}, columns={:?})",
            self.inner.num_rows,
            self.column_names()
        )
    }
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new(parent.py(), "pyroid.warc")?;
    m.gil_used(false)?;
    m.add_class::<Reader>()?;
    m.add_class::<RawStream>()?;
    m.add_class::<ContentStream>()?;
    m.add_class::<DocReader>()?;
    m.add_class::<RowBatch>()?;
    m.add_class::<Batch>()?;
    m.add("BUFF_SIZE", core::BUFF_SIZE)?;
    parent.add_submodule(&m)
}
