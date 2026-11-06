// `pyroid.fasttext` bindings. The inference code is pyroid's port of fastText
// (https://github.com/facebookresearch/fastText), Copyright (c) 2016-present,
// Facebook, Inc., MIT License; see crates/pyroid-core/src/fasttext/.

//! `pyroid.fasttext`: a drop-in for the inference API of the `fasttext`
//! package. `load_model(path)` returns a [`Model`] (frozen, `Sync`): share it
//! across threads. `predict` and `get_sentence_vector` also take batches
//! (list/tuple/Arrow) and run them detached on the shared pool.
//!
//! Return types follow fastText's Python layer: `predict(str)` gives
//! `(tuple[str, ...], ndarray[float64])`, `predict(list)` gives
//! `(list[list[str]], list[ndarray[float32]])`; vectors are float32 arrays.

use std::path::PathBuf;

use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::{
    PyIndexError, PyMemoryError, PyOSError, PyRuntimeError, PyTypeError, PyUnicodeEncodeError,
    PyValueError,
};
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple, PyType};
use pyroid_core::fasttext::{self as ft, LoadError, PredictError, Prediction};

use crate::batch::{DocCost, map_values, map_values_cost};
use crate::errors::{PyroidError, with_note};
use crate::input::{TextInput, Threads};

/// `get_line` result for one document: its word tokens and known labels.
type LineTokens = (Vec<String>, Vec<String>);

static MODEL_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
static NUMPY: PyOnceLock<()> = PyOnceLock::new();

/// `pyroid.fasttext.ModelError(PyroidError, ValueError)`: a ValueError, like
/// fastText's own load errors.
fn model_error_type(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    MODEL_ERROR
        .get_or_try_init(py, || {
            let ns = PyDict::new(py);
            ns.set_item("__module__", "pyroid.fasttext")?;
            ns.set_item(
                "__doc__",
                "A fastText model file is malformed, truncated, or not a fastText model.",
            )?;
            let bases = PyTuple::new(
                py,
                [py.get_type::<PyroidError>(), py.get_type::<PyValueError>()],
            )?;
            let ty = py.get_type::<PyType>().call1(("ModelError", bases, ns))?;
            Ok::<_, PyErr>(ty.cast_into::<PyType>()?.unbind())
        })
        .map(|t| t.bind(py))
}

/// rust-numpy needs numpy at the first array; fail with a clear ImportError
/// instead of its panic.
fn require_numpy(py: Python<'_>) -> PyResult<()> {
    NUMPY
        .get_or_try_init(py, || {
            py.import("numpy").map(|_| ()).map_err(|e| {
                let err = pyo3::exceptions::PyImportError::new_err(
                    "pyroid.fasttext needs numpy: pip install 'pyroid[fasttext]' (or numpy)",
                );
                err.set_cause(py, Some(e));
                err
            })
        })
        .copied()
}

fn load_error(py: Python<'_>, path: &std::path::Path, err: LoadError) -> PyErr {
    let shown = path.display();
    match err {
        LoadError::Io(e) => match e.raw_os_error() {
            // OSError(errno, strerror, filename) becomes FileNotFoundError etc.
            Some(code) => {
                let msg = e.to_string();
                // io::Error's Display appends " (os error N)": Python shows the errno itself.
                let strerror = msg
                    .strip_suffix(&format!(" (os error {code})"))
                    .unwrap_or(&msg)
                    .to_owned();
                PyOSError::new_err((code, strerror, shown.to_string()))
            }
            None => PyOSError::new_err(format!("{shown}: {e}")),
        },
        // MemoryError, as fastText's std::bad_alloc becomes in Python (callers fall back on it).
        e @ LoadError::OutOfMemory { .. } => PyMemoryError::new_err(format!("{shown}: {e}")),
        other => match model_error_type(py) {
            Ok(ty) => PyErr::from_type(ty.clone(), format!("{shown} {other}")),
            Err(e) => e,
        },
    }
}

fn predict_error(e: PredictError) -> PyErr {
    match e {
        PredictError::NaN => PyRuntimeError::new_err(e.message()),
        _ => PyValueError::new_err(e.message()),
    }
}

/// The first failure of a batch, in fastText's order: every line is checked
/// for `'\n'` before any is predicted.
fn first_error<T>(results: &[Option<Result<T, PredictError>>]) -> Option<(usize, PredictError)> {
    let mut first = None;
    for (i, r) in results.iter().enumerate() {
        if let Some(Err(e)) = r {
            if *e == PredictError::Newline {
                return Some((i, *e));
            }
            first.get_or_insert((i, *e));
        }
    }
    first
}

fn check<T>(
    py: Python<'_>,
    input: &TextInput,
    results: &[Option<Result<T, PredictError>>],
) -> PyResult<()> {
    match first_error(results) {
        None => Ok(()),
        Some((_, e)) if matches!(input, TextInput::Str(_)) => Err(predict_error(e)),
        Some((i, e)) => Err(with_note(
            py,
            predict_error(e),
            format!("pyroid: raised for the document at index {i}"),
        )),
    }
}

/// fastText's C++ takes `std::string`: a str with a lone surrogate can't be
/// converted, and pybind11 raises TypeError. Keep that type (with the
/// UnicodeEncodeError as the cause).
fn surrogate_as_type_error(py: Python<'_>, err: PyErr) -> PyErr {
    if !err.is_instance_of::<PyUnicodeEncodeError>(py) {
        return err;
    }
    let te = PyTypeError::new_err(
        "the text contains a lone surrogate (U+D800..U+DFFF), which has no UTF-8 encoding \
         (fastText raises TypeError for it too); remove or replace it first, e.g. \
         s.encode('utf-8', 'replace').decode('utf-8')",
    );
    // Keep the note that names the offending item of a batch.
    if let Ok(notes) = err.value(py).getattr(pyo3::intern!(py, "__notes__")) {
        if let Ok(notes) = notes.cast::<PyList>() {
            for note in notes.iter() {
                if let Ok(note) = note.extract::<String>() {
                    let _ = te.add_note(py, note);
                }
            }
        }
    }
    te.set_cause(py, Some(err));
    te
}

/// fastText's messages for a multi-line input.
const PREDICT_NEWLINE: &str = "predict processes one line at a time (remove '\\n')";
const GET_LINE_NEWLINE: &str = "get_line processes one line at a time (remove '\\n')";

/// The `ValueError` fastText's Python layer raises for a `"\n"` in `obj` (a
/// str, or the first str item of a list/tuple holding one), if any.
fn newline_error(obj: &Bound<'_, PyAny>, msg: &'static str) -> Option<PyErr> {
    let has_newline = |s: &Bound<'_, PyAny>| {
        s.cast::<PyString>()
            .is_ok_and(|s| s.contains("\n").unwrap_or(false))
    };
    if has_newline(obj) {
        return Some(PyValueError::new_err(msg));
    }
    let items: Vec<Bound<'_, PyAny>> = if let Ok(list) = obj.cast::<PyList>() {
        list.iter().collect()
    } else if let Ok(tuple) = obj.cast::<PyTuple>() {
        tuple.iter().collect()
    } else {
        return None;
    };
    let i = items.iter().position(has_newline)?;
    Some(with_note(
        obj.py(),
        PyValueError::new_err(msg),
        format!("pyroid: raised for the document at index {i}"),
    ))
}

/// The documents of a `predict`-like call. fastText's Python layer looks for
/// `"\n"` in every line before the C++ conversion rejects a lone surrogate,
/// so a newline anywhere wins over a surrogate anywhere.
fn text_input(
    obj: &Bound<'_, PyAny>,
    newline_msg: &'static str,
    threads: Option<Threads>,
) -> PyResult<TextInput> {
    TextInput::extract_with(obj, threads).map_err(|e| {
        let py = obj.py();
        if e.is_instance_of::<PyUnicodeEncodeError>(py) {
            if let Some(nl) = newline_error(obj, newline_msg) {
                return nl;
            }
        }
        surrogate_as_type_error(py, e)
    })
}

fn one_str(obj: &Bound<'_, PyAny>) -> PyResult<PyBackedStr> {
    obj.extract::<PyBackedStr>()
        .map_err(|e| surrogate_as_type_error(obj.py(), e))
}

/// `castToPythonString`: decode with the caller's error handler.
fn decode<'py>(py: Python<'py>, bytes: &[u8], errors: &str) -> PyResult<Bound<'py, PyString>> {
    match std::str::from_utf8(bytes) {
        Ok(s) => Ok(PyString::new(py, s)),
        Err(_) => Ok(PyBytes::new(py, bytes)
            .call_method1(pyo3::intern!(py, "decode"), ("utf-8", errors))?
            .cast_into::<PyString>()?),
    }
}

fn f32_vector(py: Python<'_>, v: Vec<f32>) -> Bound<'_, PyArray1<f32>> {
    PyArray1::from_vec(py, v)
}

fn f32_matrix(
    py: Python<'_>,
    rows: usize,
    cols: usize,
    data: Vec<f32>,
) -> PyResult<Bound<'_, PyAny>> {
    Ok(PyArray1::from_vec(py, data)
        .reshape([rows, cols])?
        .into_any())
}

/// A loaded fastText model (``.bin`` or quantized ``.ftz``). Immutable and
/// thread-safe: share one model across threads.
#[pyclass(frozen, module = "pyroid.fasttext", name = "Model")]
pub struct Model {
    inner: ft::Model,
    path: PathBuf,
    /// Label strs, decoded once; `None` if not valid UTF-8 (decoded per call
    /// with the caller's `on_unicode_error`).
    labels: Vec<Option<Py<PyString>>>,
    words_cache: PyOnceLock<Py<PyList>>,
    labels_cache: PyOnceLock<Py<PyList>>,
}

impl std::fmt::Debug for Model {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model")
            .field("path", &self.path)
            .field("inner", &self.inner)
            .finish_non_exhaustive()
    }
}

impl Model {
    fn new(py: Python<'_>, inner: ft::Model, path: PathBuf) -> Self {
        let labels = (0..inner.nlabels())
            .map(|i| {
                std::str::from_utf8(inner.label(i))
                    .ok()
                    .map(|s| PyString::new(py, s).unbind())
            })
            .collect();
        Self {
            inner,
            path,
            labels,
            words_cache: PyOnceLock::new(),
            labels_cache: PyOnceLock::new(),
        }
    }

    fn label_str<'py>(
        &self,
        py: Python<'py>,
        i: usize,
        errors: &str,
    ) -> PyResult<Bound<'py, PyString>> {
        match &self.labels[i] {
            Some(s) => Ok(s.bind(py).clone()),
            None => decode(py, self.inner.label(i), errors),
        }
    }

    fn label_list<'py>(
        &self,
        py: Python<'py>,
        preds: &[Prediction],
        errors: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let items = preds
            .iter()
            .map(|p| self.label_str(py, p.label, errors))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    fn words_list<'py>(&self, py: Python<'py>, errors: &str) -> PyResult<Bound<'py, PyList>> {
        let items = (0..self.inner.nwords())
            .map(|i| decode(py, self.inner.word(i), errors))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    fn labels_list<'py>(&self, py: Python<'py>, errors: &str) -> PyResult<Bound<'py, PyList>> {
        let items = (0..self.inner.nlabels())
            .map(|i| self.label_str(py, i, errors))
            .collect::<PyResult<Vec<_>>>()?;
        PyList::new(py, items)
    }

    fn get_words_impl<'py>(
        &self,
        py: Python<'py>,
        include_freq: bool,
        errors: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        let words = self.words_list(py, errors)?;
        if !include_freq {
            return Ok(words.into_any());
        }
        require_numpy(py)?;
        let counts: Vec<i64> = (0..self.inner.nwords())
            .map(|i| self.inner.word_count(i))
            .collect();
        Ok(PyTuple::new(
            py,
            [words.into_any(), PyArray1::from_vec(py, counts).into_any()],
        )?
        .into_any())
    }
}

fn quantized_matrix_error() -> PyErr {
    PyValueError::new_err("Can't get quantized Matrix")
}

#[pymethods]
impl Model {
    /// Predict the most likely labels of a line of text.
    ///
    /// Same results as fastText's ``predict``; see ``pyroid/fasttext.pyi``.
    #[pyo3(signature = (text, k = 1, threshold = 0.0, on_unicode_error = "strict", *, threads = None))]
    fn predict<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyAny>,
        k: i64,
        threshold: f32,
        on_unicode_error: &str,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyTuple>> {
        require_numpy(py)?;
        let input = text_input(text, PREDICT_NEWLINE, threads)?;
        // fastText's k is a C int: pybind11 rejects anything else with TypeError.
        let k = i32::try_from(k).map_err(|_| {
            // The Python layer's newline check runs before pybind11 converts k.
            newline_error(text, PREDICT_NEWLINE).unwrap_or_else(|| {
                PyTypeError::new_err(format!(
                    "k must fit in a 32-bit int (fastText's int32_t), got {k}"
                ))
            })
        })?;
        let model = &self.inner;
        let results = map_values_cost(py, &input, threads, DocCost::Heavy, |s| {
            model.predict(s, i64::from(k), threshold)
        })?;
        check(py, &input, &results)?;
        if matches!(input, TextInput::Str(_)) {
            let preds = match results.into_iter().next() {
                Some(Some(Ok(p))) => p,
                _ => Vec::new(),
            };
            let labels = preds
                .iter()
                .map(|p| self.label_str(py, p.label, on_unicode_error))
                .collect::<PyResult<Vec<_>>>()?;
            let probs = PyArray1::from_iter(py, preds.iter().map(|p| f64::from(p.prob)));
            return PyTuple::new(py, [PyTuple::new(py, labels)?.into_any(), probs.into_any()]);
        }
        let mut all_labels = Vec::with_capacity(results.len());
        let mut all_probs = Vec::with_capacity(results.len());
        for r in &results {
            if let Some(Ok(preds)) = r {
                all_labels.push(self.label_list(py, preds, on_unicode_error)?.into_any());
                all_probs.push(PyArray1::from_iter(py, preds.iter().map(|p| p.prob)).into_any());
            } else {
                // Arrow nulls stay null.
                all_labels.push(py.None().into_bound(py));
                all_probs.push(py.None().into_bound(py));
            }
        }
        PyTuple::new(
            py,
            [
                PyList::new(py, all_labels)?.into_any(),
                PyList::new(py, all_probs)?.into_any(),
            ],
        )
    }

    /// The sentence vector of a line (float32), or a 2-D array for a batch.
    #[pyo3(signature = (text, *, threads = None))]
    fn get_sentence_vector<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyAny>,
        threads: Option<Threads>,
    ) -> PyResult<Bound<'py, PyAny>> {
        require_numpy(py)?;
        let input = text_input(text, PREDICT_NEWLINE, threads)?;
        let model = &self.inner;
        let results = map_values_cost(py, &input, threads, DocCost::Heavy, |s| {
            model.sentence_vector(s)
        })?;
        check(py, &input, &results)?;
        let dim = model.dim();
        if matches!(input, TextInput::Str(_)) {
            let v = match results.into_iter().next() {
                Some(Some(Ok(v))) => v,
                _ => vec![0.0; dim],
            };
            return Ok(f32_vector(py, v).into_any());
        }
        let n = results.len();
        let mut flat = Vec::with_capacity(n * dim);
        for r in results {
            match r {
                Some(Ok(v)) => flat.extend_from_slice(&v),
                _ => flat.extend(std::iter::repeat_n(f32::NAN, dim)),
            }
        }
        f32_matrix(py, n, dim, flat)
    }

    /// The vector of a word: the mean of its subword rows (float32).
    fn get_word_vector<'py>(
        &self,
        py: Python<'py>,
        word: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        require_numpy(py)?;
        let word = one_str(word)?;
        let v = self.inner.word_vector(&word).ok_or_else(|| {
            PyValueError::new_err("a subword of this word is outside the input matrix")
        })?;
        Ok(f32_vector(py, v))
    }

    /// Row ``ind`` of the input matrix (float32).
    fn get_input_vector<'py>(
        &self,
        py: Python<'py>,
        ind: i64,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        require_numpy(py)?;
        let v = usize::try_from(ind)
            .ok()
            .and_then(|i| self.inner.input_vector(i))
            .ok_or_else(|| {
                PyIndexError::new_err(format!("input vector index {ind} out of range"))
            })?;
        Ok(f32_vector(py, v))
    }

    /// The words of the dictionary (no labels, no subwords).
    #[pyo3(signature = (include_freq = false, on_unicode_error = "strict"))]
    fn get_words<'py>(
        &self,
        py: Python<'py>,
        include_freq: bool,
        on_unicode_error: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.get_words_impl(py, include_freq, on_unicode_error)
    }

    /// The labels (prefix included). Unsupervised models return their words.
    #[pyo3(signature = (include_freq = false, on_unicode_error = "strict"))]
    fn get_labels<'py>(
        &self,
        py: Python<'py>,
        include_freq: bool,
        on_unicode_error: &str,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !self.inner.is_supervised() {
            // fastText passes include_freq only (strict decoding).
            return self.get_words_impl(py, include_freq, "strict");
        }
        let labels = self.labels_list(py, on_unicode_error)?;
        if !include_freq {
            return Ok(labels.into_any());
        }
        require_numpy(py)?;
        let counts: Vec<i64> = (0..self.inner.nlabels())
            .map(|i| self.inner.label_count(i))
            .collect();
        Ok(PyTuple::new(
            py,
            [labels.into_any(), PyArray1::from_vec(py, counts).into_any()],
        )?
        .into_any())
    }

    /// The dictionary's words (cached ``get_words()``).
    #[getter(words)]
    fn words_property<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.words_cache
            .get_or_try_init(py, || self.words_list(py, "strict").map(Bound::unbind))
            .map(|l| l.bind(py).clone())
    }

    /// The model's labels (cached ``get_labels()``).
    #[getter(labels)]
    fn labels_property<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        self.labels_cache
            .get_or_try_init(py, || {
                let l = if self.inner.is_supervised() {
                    self.labels_list(py, "strict")?
                } else {
                    self.words_list(py, "strict")?
                };
                Ok::<_, PyErr>(l.unbind())
            })
            .map(|l| l.bind(py).clone())
    }

    /// Words and labels of a line (or of each line of a list).
    #[pyo3(signature = (text, on_unicode_error = "strict"))]
    fn get_line<'py>(
        &self,
        py: Python<'py>,
        text: &Bound<'py, PyAny>,
        on_unicode_error: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = on_unicode_error; // tokens of a str are valid UTF-8
        let input = text_input(text, GET_LINE_NEWLINE, None)?;
        let model = &self.inner;
        let results = map_values(
            py,
            &input,
            Some(Threads(pyroid_core::par::Threads::ONE)),
            |s| {
                if s.contains('\n') {
                    return Err(());
                }
                let (w, l) = model.line(s);
                Ok((
                    w.into_iter().map(str::to_owned).collect::<Vec<_>>(),
                    l.into_iter().map(str::to_owned).collect::<Vec<_>>(),
                ))
            },
        )?;
        if results.iter().any(|r| matches!(r, Some(Err(())))) {
            return Err(PyValueError::new_err(GET_LINE_NEWLINE));
        }
        let pair = |r: Option<Result<LineTokens, ()>>| -> PyResult<[Bound<'py, PyAny>; 2]> {
            match r {
                Some(Ok((w, l))) => Ok([
                    PyList::new(py, w)?.into_any(),
                    PyList::new(py, l)?.into_any(),
                ]),
                _ => Ok([py.None().into_bound(py), py.None().into_bound(py)]),
            }
        };
        if matches!(input, TextInput::Str(_)) {
            return PyTuple::new(py, pair(results.into_iter().next().flatten())?);
        }
        let mut words = Vec::with_capacity(results.len());
        let mut labels = Vec::with_capacity(results.len());
        for r in results {
            let [w, l] = pair(r)?;
            words.push(w);
            labels.push(l);
        }
        PyTuple::new(
            py,
            [
                PyList::new(py, words)?.into_any(),
                PyList::new(py, labels)?.into_any(),
            ],
        )
    }

    /// The word and its character n-grams, with their input-matrix rows.
    #[pyo3(signature = (word, on_unicode_error = "strict"))]
    fn get_subwords<'py>(
        &self,
        py: Python<'py>,
        word: &Bound<'py, PyAny>,
        on_unicode_error: &str,
    ) -> PyResult<Bound<'py, PyTuple>> {
        let _ = on_unicode_error; // substrings of a str are valid UTF-8
        require_numpy(py)?;
        let word = one_str(word)?;
        let (strings, ids) = self.inner.subwords(&word);
        let ids: Vec<i64> = ids.into_iter().map(i64::from).collect();
        PyTuple::new(
            py,
            [
                PyList::new(py, strings)?.into_any(),
                PyArray1::from_vec(py, ids).into_any(),
            ],
        )
    }

    /// Dictionary id of ``word`` (labels included), or -1.
    fn get_word_id(&self, word: &Bound<'_, PyAny>) -> PyResult<i32> {
        Ok(self.inner.word_id(&one_str(word)?))
    }

    /// Label id of ``label``, or -1 (fastText's arithmetic: a word gives a
    /// negative id other than -1).
    fn get_label_id(&self, label: &Bound<'_, PyAny>) -> PyResult<i64> {
        Ok(self.inner.label_id(&one_str(label)?))
    }

    /// The input-matrix row a character n-gram hashes to.
    fn get_subword_id(&self, subword: &Bound<'_, PyAny>) -> PyResult<i64> {
        self.inner
            .subword_id(&one_str(subword)?)
            .ok_or_else(|| PyValueError::new_err("this model has no subword buckets (bucket = 0)"))
    }

    /// Dimension of the vectors.
    fn get_dimension(&self) -> usize {
        self.inner.dim()
    }

    /// True for a quantized (``.ftz``) model.
    fn is_quantized(&self) -> bool {
        self.inner.is_quantized()
    }

    /// A copy of the input matrix (float32, rows x dim).
    fn get_input_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        require_numpy(py)?;
        match (self.inner.is_quantized(), self.inner.input_matrix()) {
            (false, Some((r, c, data))) => f32_matrix(py, r, c, data.to_vec()),
            _ => Err(quantized_matrix_error()),
        }
    }

    /// A copy of the output matrix (float32, rows x dim).
    fn get_output_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        require_numpy(py)?;
        match (self.inner.is_quantized(), self.inner.output_matrix()) {
            (false, Some((r, c, data))) => f32_matrix(py, r, c, data.to_vec()),
            _ => Err(quantized_matrix_error()),
        }
    }

    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        word: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray1<f32>>> {
        self.get_word_vector(py, word)
    }

    fn __contains__(&self, word: &Bound<'_, PyAny>) -> bool {
        let Ok(word) = word.extract::<PyBackedStr>() else {
            return false;
        };
        let id = self.inner.word_id(&word);
        usize::try_from(id).is_ok_and(|i| i < self.inner.nwords())
    }

    fn __repr__(&self) -> String {
        format!(
            "<pyroid.fasttext.Model {:?}: dim {}, {} words, {} labels, {}>",
            self.path.display().to_string(),
            self.inner.dim(),
            self.inner.nwords(),
            self.inner.nlabels(),
            if self.inner.is_quantized() {
                "quantized"
            } else {
                "dense"
            }
        )
    }
}

/// Split ``text`` into fastText tokens (``fasttext.tokenize``): ``"</s>"``
/// for each newline.
#[pyfunction]
fn tokenize(text: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let text = one_str(text)?;
    Ok(ft::tokenize(&text).into_iter().map(str::to_owned).collect())
}

/// Load a fastText model (``.bin``, or quantized ``.ftz``).
#[pyfunction]
fn load_model(py: Python<'_>, path: PathBuf) -> PyResult<Model> {
    let inner = py
        .detach(|| ft::Model::load(&path))
        .map_err(|e| load_error(py, &path, e))?;
    Ok(Model::new(py, inner, path))
}

pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "pyroid.fasttext")?;
    m.gil_used(false)?;
    m.add_class::<Model>()?;
    m.add("ModelError", model_error_type(py)?)?;
    m.add_function(wrap_pyfunction!(load_model, &m)?)?;
    m.add_function(wrap_pyfunction!(tokenize, &m)?)?;
    parent.add_submodule(&m)
}
