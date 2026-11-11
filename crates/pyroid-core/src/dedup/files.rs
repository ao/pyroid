// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! datatrove's MinHash stages 1-3 on disk, with its intermediate files byte
//! for byte (all little-endian):
//!
//! - Stage 1 (`MinhashDedupSignature`), [`SigFileWriter`]: for every bucket
//!   `b`, `bucket_{b:03d}/{rank:05d}.minhash.sig`, one record per document
//!   with a signature: the bucket's `hashes_per_bucket` values (`u64`, or
//!   `u32` for 32-bit hashes) then the `u32` document index, sorted by
//!   (values, document). Document indices count every document added.
//! - Stage 2 (`MinhashDedupBuckets`), [`minhash_buckets`]: one bucket
//!   worker's `{b:05d}_{w:02d}.dups`, `<4I` records `(rank, doc, rank, doc)`
//!   for every two consecutive signatures with equal bands, in datatrove's
//!   heap order (bands, then the file's position in the reader list, then
//!   the document). [`worker_hash_range`] is `get_worker_hash_range`.
//! - Stage 3 (`MinhashDedupCluster`), [`minhash_cluster`]: the union-find
//!   over every pair, then `{file:06d}.remove` (`<I` doc) and optionally
//!   `{file:06d}.clusters` / `{file:06d}.sizes` (`<2I` doc, value).
//!
//! Memory is bounded. Stage 1 keeps signatures up to a byte budget, spills
//! each bucket as a sorted run past it, and merges the runs at the end.
//! Stage 2 is a streaming k-way merge with bounded reader buffers; with more
//! files than may be open at once it merges groups into runs first. Stage 3
//! holds one entry per document that appears in a pair and streams the
//! pairs. Every output is written to a temporary file, then renamed into
//! place; temporary files are removed on success, on error and on abort.

use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::time::{SystemTime, UNIX_EPOCH};

use rustc_hash::FxHashMap;

use super::config::{MinhashConfig, Precision};
use super::rng::MERSENNE_PRIME;
use crate::par::{self, PoolError, Threads};

/// datatrove's `SENTINEL`: the file and document of a match with an index.
pub const SENTINEL: u32 = u32::MAX;
/// The `(SENTINEL, SENTINEL)` node as a stage-3 key.
const SENTINEL_KEY: u64 = u64::MAX;

const KIB: usize = 1 << 10;
const MIB: usize = 1 << 20;
/// Smallest and largest buffer of one merge reader.
const MIN_READ_BUFFER: usize = 8 * KIB;
const MAX_READ_BUFFER: usize = MIB;
/// Buffer of every output file.
const WRITE_BUFFER: usize = 256 * KIB;
/// Buffer of a stage-3 `.dups` reader.
const DUPS_READ_BUFFER: usize = 256 * KIB;
/// Runs one stage-1 merge reads at once; more are merged in passes.
const SPILL_FANIN: usize = 30;
/// Buckets a stage-1 writer sorts or merges at once.
const BUCKET_TASKS: usize = 8;
/// The name of stage 1's temporary folder, inside the output folder.
pub const SIG_TMP_DIR: &str = ".pyroid-tmp";

/// Why a MinHash file stage failed.
#[derive(Debug)]
pub enum MinhashFileError {
    /// Reading or writing `path` failed.
    Io {
        path: PathBuf,
        err: io::Error,
    },
    /// `path` is not a valid input file (size, order or name).
    Malformed {
        path: PathBuf,
        msg: String,
    },
    /// An argument or a limit (datatrove's assertions).
    Invalid(String),
    Pool(PoolError),
}

impl fmt::Display for MinhashFileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, err } => write!(f, "{}: {err}", path.display()),
            Self::Malformed { path, msg } => write!(f, "{}: {msg}", path.display()),
            Self::Invalid(msg) => f.write_str(msg),
            Self::Pool(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for MinhashFileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { err, .. } => Some(err),
            Self::Pool(e) => Some(e),
            Self::Malformed { .. } | Self::Invalid(_) => None,
        }
    }
}

impl From<PoolError> for MinhashFileError {
    fn from(e: PoolError) -> Self {
        Self::Pool(e)
    }
}

type Result<T, E = MinhashFileError> = std::result::Result<T, E>;

/// `map_err` adapter: an IO error on `path`.
fn io_at(path: &Path) -> impl FnOnce(io::Error) -> MinhashFileError + '_ {
    move |err| MinhashFileError::Io {
        path: path.to_owned(),
        err,
    }
}

fn malformed(path: &Path, msg: impl Into<String>) -> MinhashFileError {
    MinhashFileError::Malformed {
        path: path.to_owned(),
        msg: msg.into(),
    }
}

fn not_whole(path: &Path, size: u64, record: usize) -> MinhashFileError {
    malformed(
        path,
        format!("{size} bytes is not a whole number of {record}-byte records"),
    )
}

fn read_u32(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

/// The record layout of one bucket's signature files.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SigLayout {
    pub hashes_per_bucket: usize,
    pub precision: Precision,
}

impl SigLayout {
    pub fn new(config: &MinhashConfig) -> Self {
        Self {
            hashes_per_bucket: config.hashes_per_bucket,
            precision: config.hash_config.precision,
        }
    }

    /// Bytes per hash value (4 or 8).
    pub fn width(self) -> usize {
        match self.precision {
            Precision::P32 => 4,
            Precision::P64 => 8,
        }
    }

    /// Bytes of one band (`hashes_per_bucket` values).
    pub fn band_bytes(self) -> usize {
        self.hashes_per_bucket * self.width()
    }

    /// Bytes of one `.minhash.sig` record (band, then `u32` document).
    pub fn record_size(self) -> usize {
        self.band_bytes() + 4
    }

    /// datatrove's default (exclusive) `hash_max`: `2**61 - 1` for 64-bit
    /// hashes, `2**32 - 1` for 32-bit ones.
    pub fn default_hash_max(self) -> u64 {
        match self.precision {
            Precision::P32 => u64::from(u32::MAX),
            Precision::P64 => MERSENNE_PRIME,
        }
    }

    /// Append the band's values to `out` (32-bit values are truncated).
    pub fn encode_band(self, band: &[u64], out: &mut Vec<u8>) {
        match self.precision {
            Precision::P64 => {
                for &v in band {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            Precision::P32 => {
                for &v in band {
                    // Signatures of 32-bit configs are masked to 32 bits.
                    #[allow(clippy::cast_possible_truncation)]
                    out.extend_from_slice(&(v as u32).to_le_bytes());
                }
            }
        }
    }

    /// Append one `.minhash.sig` record to `out`.
    pub fn encode(self, band: &[u64], doc: u32, out: &mut Vec<u8>) {
        self.encode_band(band, out);
        out.extend_from_slice(&doc.to_le_bytes());
    }

    /// Decode a band (`band_bytes` bytes) into `band`.
    pub fn decode_band(self, bytes: &[u8], band: &mut [u64]) {
        match self.precision {
            Precision::P64 => {
                for (v, c) in band.iter_mut().zip(bytes.chunks_exact(8)) {
                    *v = u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]);
                }
            }
            Precision::P32 => {
                for (v, c) in band.iter_mut().zip(bytes.chunks_exact(4)) {
                    *v = u64::from(read_u32(c));
                }
            }
        }
    }

    /// Decode one `.minhash.sig` record: the band into `band`, returns the
    /// document.
    pub fn decode(self, record: &[u8], band: &mut [u64]) -> u32 {
        let bb = self.band_bytes();
        self.decode_band(&record[..bb], band);
        read_u32(&record[bb..])
    }

    fn first_value(self, bytes: &[u8]) -> u64 {
        let mut v = [0u64; 1];
        self.decode_band(&bytes[..self.width()], &mut v);
        v[0]
    }
}

/// `folder/bucket_{bucket:03d}/{rank:05d}.minhash.sig`.
pub fn sig_file_path(folder: &Path, bucket: usize, rank: u32) -> PathBuf {
    folder
        .join(format!("bucket_{bucket:03}"))
        .join(format!("{rank:05}.minhash.sig"))
}

/// A name part no other writer uses at the same time.
fn unique_tag() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, AtomicOrdering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.subsec_nanos());
    format!("{}-{nanos:08x}-{n}", std::process::id())
}

/// A new, empty directory `parent/{prefix}-{unique}`.
///
/// Other writers remove `parent` when they leave it empty, at any time. So
/// either step can fail with `AlreadyExists` or `NotFound`, `create_dir_all`
/// included: it reports `AlreadyExists` when its `mkdir` fails because the
/// directory exists and the directory is gone again by the time it checks.
/// Both are retried.
fn make_temp_dir(parent: &Path, prefix: &str) -> Result<PathBuf> {
    let mut tries = 0;
    loop {
        let dir = parent.join(format!("{prefix}-{}", unique_tag()));
        let made = fs::create_dir_all(parent)
            .map_err(|e| (parent.to_path_buf(), e))
            .and_then(|()| fs::create_dir(&dir).map_err(|e| (dir.clone(), e)));
        match made {
            Ok(()) => return Ok(dir),
            Err((_, e))
                if tries < 64
                    && matches!(
                        e.kind(),
                        io::ErrorKind::AlreadyExists | io::ErrorKind::NotFound
                    ) =>
            {
                tries += 1;
            }
            Err((path, err)) => return Err(MinhashFileError::Io { path, err }),
        }
    }
}

/// Files removed when dropped, unless forgotten (temporary outputs, runs).
#[derive(Debug, Default)]
struct TempFiles(Vec<PathBuf>);

impl TempFiles {
    fn add(&mut self, path: PathBuf) -> PathBuf {
        self.0.push(path.clone());
        path
    }

    /// Delete `path` now.
    fn remove(&mut self, path: &Path) {
        let _ = fs::remove_file(path);
        self.forget(path);
    }

    /// Stop tracking `path` (it was renamed into place).
    fn forget(&mut self, path: &Path) {
        self.0.retain(|p| p != path);
    }
}

impl Drop for TempFiles {
    fn drop(&mut self) {
        for p in &self.0 {
            let _ = fs::remove_file(p);
        }
    }
}

fn create(path: &Path, buffer: usize) -> Result<BufWriter<File>> {
    File::create(path)
        .map(|f| BufWriter::with_capacity(buffer, f))
        .map_err(io_at(path))
}

/// Flush and close `w`, reporting any error.
fn close(w: BufWriter<File>, path: &Path) -> Result<()> {
    let f = w.into_inner().map_err(|e| io_at(path)(e.into_error()))?;
    drop(f);
    Ok(())
}

fn rename(from: &Path, to: &Path) -> Result<()> {
    fs::rename(from, to).map_err(io_at(to))
}

fn file_size(path: &Path) -> Result<u64> {
    Ok(fs::metadata(path).map_err(io_at(path))?.len())
}

/// clamp(budget / readers, 8 KiB, 1 MiB).
fn read_buffer(budget: usize, readers: usize) -> usize {
    (budget / readers.max(1)).clamp(MIN_READ_BUFFER, MAX_READ_BUFFER)
}

fn as_u64(n: usize) -> u64 {
    u64::try_from(n).unwrap_or(u64::MAX)
}

// ---------------------------------------------------------------------------
// k-way merge
// ---------------------------------------------------------------------------

/// The current record of a merge input: band values, and for stage 2 the
/// file's reader position and rank.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Rec {
    band: Vec<u64>,
    pos: u32,
    rank: u32,
    doc: u32,
}

impl Rec {
    fn new(layout: SigLayout) -> Self {
        Self {
            band: vec![0; layout.hashes_per_bucket],
            ..Self::default()
        }
    }
}

/// Stage 1 order: (band, document).
fn by_doc(a: &Rec, b: &Rec) -> Ordering {
    a.band.cmp(&b.band).then(a.doc.cmp(&b.doc))
}

/// Stage 2 order: datatrove's `HashSig` heap order with one entry per
/// reader, (band, reader position); a reader's own records stay in file
/// order.
fn by_pos(a: &Rec, b: &Rec) -> Ordering {
    a.band.cmp(&b.band).then(a.pos.cmp(&b.pos))
}

/// A merge input, positioned on a record.
trait Cursor {
    fn rec(&self) -> &Rec;
    /// Move to the next record; `false` at the end.
    fn advance(&mut self) -> Result<bool>;
}

fn sift_down(heap: &mut [usize], mut i: usize, less: impl Fn(usize, usize) -> bool) {
    loop {
        let l = 2 * i + 1;
        if l >= heap.len() {
            return;
        }
        let c = if l + 1 < heap.len() && less(heap[l + 1], heap[l]) {
            l + 1
        } else {
            l
        };
        if !less(heap[c], heap[i]) {
            return;
        }
        heap.swap(i, c);
        i = c;
    }
}

/// `emit` every record of `cursors` (each on its first record) in `cmp`
/// order. `cmp` must not tie records of different cursors.
fn merge<C: Cursor>(
    mut cursors: Vec<C>,
    cmp: fn(&Rec, &Rec) -> Ordering,
    mut emit: impl FnMut(&Rec) -> Result<()>,
) -> Result<()> {
    let mut heap: Vec<usize> = (0..cursors.len()).collect();
    for i in (0..heap.len() / 2).rev() {
        sift_down(&mut heap, i, |a, b| {
            cmp(cursors[a].rec(), cursors[b].rec()) == Ordering::Less
        });
    }
    while let Some(&top) = heap.first() {
        emit(cursors[top].rec())?;
        if !cursors[top].advance()? {
            heap.swap_remove(0);
        }
        sift_down(&mut heap, 0, |a, b| {
            cmp(cursors[a].rec(), cursors[b].rec()) == Ordering::Less
        });
    }
    Ok(())
}

/// Keep `c` if it has a first record.
fn started<C: Cursor>(mut c: C) -> Result<Option<C>> {
    Ok(c.advance()?.then_some(c))
}

/// The records of the run files of one stage, and their order: stage-1
/// records (band, doc) or, with `ext`, stage-2 records (band, reader
/// position, rank, doc).
#[derive(Clone, Copy, Debug)]
struct RunKind {
    layout: SigLayout,
    ext: bool,
    cmp: fn(&Rec, &Rec) -> Ordering,
}

impl RunKind {
    fn stage1(layout: SigLayout) -> Self {
        Self {
            layout,
            ext: false,
            cmp: by_doc,
        }
    }

    fn stage2(layout: SigLayout) -> Self {
        Self {
            layout,
            ext: true,
            cmp: by_pos,
        }
    }

    fn record_size(self) -> usize {
        self.layout.band_bytes() + if self.ext { 12 } else { 4 }
    }

    fn encode(self, rec: &Rec, line: &mut Vec<u8>) {
        line.clear();
        self.layout.encode_band(&rec.band, line);
        if self.ext {
            line.extend_from_slice(&rec.pos.to_le_bytes());
            line.extend_from_slice(&rec.rank.to_le_bytes());
        }
        line.extend_from_slice(&rec.doc.to_le_bytes());
    }
}

/// A run file written by a merge pass.
#[derive(Debug)]
struct RunFile {
    r: BufReader<File>,
    path: PathBuf,
    kind: RunKind,
    left: u64,
    raw: Vec<u8>,
    rec: Rec,
}

impl RunFile {
    fn open(path: &Path, kind: RunKind, buffer: usize) -> Result<Option<Self>> {
        let size = kind.record_size();
        let bytes = file_size(path)?;
        if bytes % as_u64(size) != 0 {
            return Err(not_whole(path, bytes, size));
        }
        let f = File::open(path).map_err(io_at(path))?;
        started(Self {
            r: BufReader::with_capacity(buffer, f),
            path: path.to_owned(),
            kind,
            left: bytes / as_u64(size),
            raw: vec![0; size],
            rec: Rec::new(kind.layout),
        })
    }
}

impl Cursor for RunFile {
    fn rec(&self) -> &Rec {
        &self.rec
    }

    fn advance(&mut self) -> Result<bool> {
        if self.left == 0 {
            return Ok(false);
        }
        self.r
            .read_exact(&mut self.raw)
            .map_err(io_at(&self.path))?;
        self.left -= 1;
        let bb = self.kind.layout.band_bytes();
        self.kind
            .layout
            .decode_band(&self.raw[..bb], &mut self.rec.band);
        let tail = &self.raw[bb..];
        if self.kind.ext {
            self.rec.pos = read_u32(tail);
            self.rec.rank = read_u32(&tail[4..]);
            self.rec.doc = read_u32(&tail[8..]);
        } else {
            self.rec.doc = read_u32(tail);
        }
        Ok(true)
    }
}

/// Merge `cursors` into the run file `path`.
fn write_run<C: Cursor>(cursors: Vec<C>, kind: RunKind, path: &Path, buffer: usize) -> Result<()> {
    let mut w = create(path, buffer)?;
    let mut line = Vec::with_capacity(kind.record_size());
    merge(cursors, kind.cmp, |rec| {
        kind.encode(rec, &mut line);
        w.write_all(&line).map_err(io_at(path))
    })?;
    close(w, path)
}

/// Merge run files in groups of `fanin` until at most `fanin` are left.
/// `new_run` names each new run (and tracks it); merged inputs are deleted.
fn reduce_runs(
    mut runs: Vec<PathBuf>,
    fanin: usize,
    kind: RunKind,
    budget: usize,
    temps: &mut TempFiles,
    mut new_run: impl FnMut(&mut TempFiles) -> PathBuf,
) -> Result<Vec<PathBuf>> {
    while runs.len() > fanin {
        let mut next = Vec::with_capacity(runs.len().div_ceil(fanin));
        for group in runs.chunks(fanin) {
            if let [only] = group {
                next.push(only.clone());
                continue;
            }
            let buffer = read_buffer(budget, group.len() + 1);
            let cursors = open_runs(group, kind, buffer)?;
            let out = new_run(temps);
            write_run(cursors, kind, &out, buffer)?;
            for p in group {
                temps.remove(p);
            }
            next.push(out);
        }
        runs = next;
    }
    Ok(runs)
}

fn open_runs(paths: &[PathBuf], kind: RunKind, buffer: usize) -> Result<Vec<RunFile>> {
    let mut out = Vec::with_capacity(paths.len());
    for p in paths {
        out.extend(RunFile::open(p, kind, buffer)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Stage 1
// ---------------------------------------------------------------------------

/// The buffered records of one bucket, sorted: `(first value, record)`.
fn sorted_keys(sigs: &[u64], nh: usize, h: usize, bucket: usize) -> Vec<(u64, u32)> {
    let off = bucket * h;
    let band = |i: u32| {
        let s = i as usize * nh + off;
        &sigs[s..s + h]
    };
    // The buffer holds fewer than 2**32 records (documents are u32).
    #[allow(clippy::cast_possible_truncation)]
    let mut keys: Vec<(u64, u32)> = (0..(sigs.len() / nh) as u32)
        .map(|i| (band(i)[0], i))
        .collect();
    keys.sort_unstable();
    if h > 1 {
        // Records are buffered in document order, so (band, record) order is
        // (band, document) order. Finish each run of equal first values.
        let mut start = 0;
        while start < keys.len() {
            let first = keys[start].0;
            let end = start + keys[start..].partition_point(|k| k.0 == first);
            if end - start > 1 {
                keys[start..end].sort_unstable_by(|x, y| {
                    band(x.1)[1..].cmp(&band(y.1)[1..]).then(x.1.cmp(&y.1))
                });
            }
            start = end;
        }
    }
    keys
}

/// The buffered records of one bucket as a merge input.
#[derive(Debug)]
struct BufferRecs<'a> {
    sigs: &'a [u64],
    docs: &'a [u32],
    keys: Vec<(u64, u32)>,
    next: usize,
    nh: usize,
    off: usize,
    rec: Rec,
}

impl Cursor for BufferRecs<'_> {
    fn rec(&self) -> &Rec {
        &self.rec
    }

    fn advance(&mut self) -> Result<bool> {
        let Some(&(_, i)) = self.keys.get(self.next) else {
            return Ok(false);
        };
        self.next += 1;
        let i = i as usize;
        let h = self.rec.band.len();
        let s = i * self.nh + self.off;
        self.rec.band.copy_from_slice(&self.sigs[s..s + h]);
        self.rec.doc = self.docs[i];
        Ok(true)
    }
}

#[derive(Debug)]
enum Stage1<'a> {
    Run(RunFile),
    Buffer(BufferRecs<'a>),
}

impl Cursor for Stage1<'_> {
    fn rec(&self) -> &Rec {
        match self {
            Self::Run(c) => c.rec(),
            Self::Buffer(c) => c.rec(),
        }
    }

    fn advance(&mut self) -> Result<bool> {
        match self {
            Self::Run(c) => c.advance(),
            Self::Buffer(c) => c.advance(),
        }
    }
}

/// How many buckets to process at once.
fn bucket_tasks(num_buckets: usize, records: usize, threads: Threads) -> Result<usize> {
    if threads == Threads::ONE || records <= par::MIN_DOCS_PER_TASK {
        return Ok(1);
    }
    Ok(par::max_tasks(threads)?
        .min(BUCKET_TASKS)
        .min(num_buckets)
        .max(1))
}

/// `f(b)` for every bucket, `tasks` contiguous groups at once.
fn for_buckets<F>(num_buckets: usize, tasks: usize, threads: Threads, f: F) -> Result<()>
where
    F: Fn(usize) -> Result<()> + Sync + Send,
{
    if tasks <= 1 {
        return (0..num_buckets).try_for_each(f);
    }
    let groups: Vec<Range<usize>> = (0..tasks)
        .map(|t| t * num_buckets / tasks..(t + 1) * num_buckets / tasks)
        .collect();
    par::map_tasks(&groups, threads, |g| g.clone().try_for_each(&f))?
        .into_iter()
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WriterState {
    Open,
    Failed,
    Finished,
    Aborted,
}

/// Stage 1 for one rank (datatrove's `MinhashDedupSignature.run`): add the
/// signatures of the rank's documents in order, then [`Self::finish`]
/// writes every bucket's `.minhash.sig` file.
///
/// Signatures are buffered up to the memory budget; past it each bucket's
/// buffer is sorted and spilled as a run under `folder/.pyroid-tmp/`, and
/// `finish` merges the runs with the rest of the buffer. Bucket files are
/// renamed into place only by `finish`; [`Self::abort`] (or dropping the
/// writer unfinished) removes the temporary files and writes nothing.
#[derive(Debug)]
pub struct SigFileWriter {
    folder: PathBuf,
    rank: u32,
    layout: SigLayout,
    num_buckets: usize,
    budget: usize,
    /// Records buffered before a spill.
    capacity: usize,
    tmp: Option<PathBuf>,
    /// `num_hashes` values per buffered record.
    sigs: Vec<u64>,
    docs: Vec<u32>,
    spills: usize,
    spilled: usize,
    documents: u64,
    state: WriterState,
}

impl SigFileWriter {
    /// A writer for `folder/bucket_XXX/{rank:05d}.minhash.sig`. Creates its
    /// temporary folder `folder/.pyroid-tmp/sig-{rank}-...` (and `folder`).
    pub fn new(
        folder: impl Into<PathBuf>,
        rank: u32,
        config: &MinhashConfig,
        memory_budget: usize,
    ) -> Result<Self> {
        config
            .validate()
            .map_err(|e| MinhashFileError::Invalid(e.to_string()))?;
        let folder = folder.into();
        let nh = config.num_hashes();
        // Per buffered record: its signature, its document, and the sort key
        // of each bucket sorted at the same time.
        let per_record = 8 * nh + 4 + 16 * config.num_buckets.min(BUCKET_TASKS);
        let tmp = make_temp_dir(&folder.join(SIG_TMP_DIR), &format!("sig-{rank:05}"))?;
        Ok(Self {
            folder,
            rank,
            layout: SigLayout::new(config),
            num_buckets: config.num_buckets,
            budget: memory_budget,
            capacity: (memory_budget / per_record).max(1),
            tmp: Some(tmp),
            sigs: Vec::new(),
            docs: Vec::new(),
            spills: 0,
            spilled: 0,
            documents: 0,
            state: WriterState::Open,
        })
    }

    /// Documents added so far (with or without a signature).
    pub fn documents(&self) -> u64 {
        self.documents
    }

    /// Runs spilled so far (each is one file per bucket).
    pub fn spills(&self) -> usize {
        self.spills
    }

    fn num_hashes(&self) -> usize {
        self.num_buckets * self.layout.hashes_per_bucket
    }

    fn check_open(&self) -> Result<()> {
        let why = match self.state {
            WriterState::Open => return Ok(()),
            WriterState::Failed => "the writer failed earlier; call abort()",
            WriterState::Finished => "finish() was already called",
            WriterState::Aborted => "the writer was aborted",
        };
        Err(MinhashFileError::Invalid(why.to_owned()))
    }

    /// Add `present.len()` documents, numbered from [`Self::documents`]:
    /// `rows` holds `num_hashes` values per document (bucket by bucket),
    /// read only where `present` is true (documents without a signature
    /// get no record). Returns the range of indices given. After an IO
    /// error the writer only accepts [`Self::abort`].
    pub fn append(
        &mut self,
        rows: &[u64],
        present: &[bool],
        threads: Threads,
    ) -> Result<Range<u64>> {
        self.check_open()?;
        let nh = self.num_hashes();
        assert_eq!(rows.len(), present.len() * nh, "one row per document");
        let start = self.documents;
        let end = start + as_u64(present.len());
        if end > u64::from(u32::MAX) {
            return Err(MinhashFileError::Invalid(
                "a signature file holds at most 2**32 - 1 documents".to_owned(),
            ));
        }
        for (i, (row, &has)) in rows.chunks_exact(nh).zip(present).enumerate() {
            if !has {
                continue;
            }
            self.reserve();
            self.sigs.extend_from_slice(row);
            // `end` fits u32, checked above.
            #[allow(clippy::cast_possible_truncation)]
            self.docs.push((start + as_u64(i)) as u32);
            if self.docs.len() >= self.capacity {
                if let Err(e) = self.spill(threads) {
                    self.state = WriterState::Failed;
                    return Err(e);
                }
            }
        }
        self.documents = end;
        Ok(start..end)
    }

    /// Grow the buffer without allocating past the capacity.
    fn reserve(&mut self) {
        // Called with fewer than `capacity` records buffered (a full buffer
        // is spilled at once).
        let nh = self.num_hashes();
        if self.docs.len() == self.docs.capacity() {
            let want = (self.docs.capacity() * 2).max(1024).min(self.capacity);
            self.docs.reserve_exact(want - self.docs.len());
        }
        let need = self.docs.capacity().min(self.capacity) * nh;
        if self.sigs.capacity() < need {
            self.sigs.reserve_exact(need - self.sigs.len());
        }
    }

    fn run_path(tmp: &Path, bucket: usize, spill: usize) -> PathBuf {
        tmp.join(format!("{bucket:03}-{spill:06}.run"))
    }

    /// Sort each bucket's buffer and write it as a run.
    fn spill(&mut self, threads: Threads) -> Result<()> {
        let tmp = self
            .tmp
            .as_deref()
            .expect("an open writer has a temp folder");
        let (sigs, docs, layout) = (&self.sigs, &self.docs, self.layout);
        let (nh, h, k) = (self.num_hashes(), layout.hashes_per_bucket, self.spills);
        let tasks = bucket_tasks(self.num_buckets, docs.len(), threads)?;
        for_buckets(self.num_buckets, tasks, threads, |b| {
            let keys = sorted_keys(sigs, nh, h, b);
            let path = Self::run_path(tmp, b, k);
            let mut w = create(&path, WRITE_BUFFER)?;
            let mut line = Vec::with_capacity(layout.record_size());
            for &(_, i) in &keys {
                let s = i as usize * nh + b * h;
                line.clear();
                layout.encode(&sigs[s..s + h], docs[i as usize], &mut line);
                w.write_all(&line).map_err(io_at(&path))?;
            }
            close(w, &path)
        })?;
        self.spills += 1;
        self.spilled += self.docs.len();
        self.sigs.clear();
        self.docs.clear();
        Ok(())
    }

    /// Merge bucket `b`'s runs and buffer into `out`.
    fn write_bucket(&self, b: usize, out: &Path, buffer_budget: usize) -> Result<()> {
        let tmp = self
            .tmp
            .as_deref()
            .expect("an open writer has a temp folder");
        let layout = self.layout;
        let mut temps = TempFiles::default();
        let runs: Vec<PathBuf> = (0..self.spills)
            .map(|k| temps.add(Self::run_path(tmp, b, k)))
            .collect();
        let kind = RunKind::stage1(layout);
        let mut pass = 0usize;
        let runs = reduce_runs(runs, SPILL_FANIN, kind, buffer_budget, &mut temps, |t| {
            pass += 1;
            t.add(tmp.join(format!("{b:03}-m{pass:06}.run")))
        })?;
        let buffer = read_buffer(buffer_budget, runs.len() + 1);
        let mut cursors: Vec<Stage1<'_>> = open_runs(&runs, kind, buffer)?
            .into_iter()
            .map(Stage1::Run)
            .collect();
        let nh = self.num_hashes();
        let h = layout.hashes_per_bucket;
        cursors.extend(
            started(BufferRecs {
                sigs: &self.sigs,
                docs: &self.docs,
                keys: sorted_keys(&self.sigs, nh, h, b),
                next: 0,
                nh,
                off: b * h,
                rec: Rec::new(layout),
            })?
            .map(Stage1::Buffer),
        );
        write_run(cursors, kind, out, WRITE_BUFFER)
        // `temps` removes the merged runs.
    }

    /// Write every bucket's `.minhash.sig` file (empty when no document has
    /// a signature) and remove the temporary files. Returns the number of
    /// documents added. Fails if called twice.
    pub fn finish(&mut self, threads: Threads) -> Result<u64> {
        self.check_open()?;
        let res = self.write_all(threads);
        self.state = if res.is_ok() {
            WriterState::Finished
        } else {
            WriterState::Failed
        };
        self.cleanup();
        res.map(|()| self.documents)
    }

    fn write_all(&mut self, threads: Threads) -> Result<()> {
        let tmp = self.tmp.clone().expect("an open writer has a temp folder");
        let outputs: Vec<PathBuf> = (0..self.num_buckets)
            .map(|b| tmp.join(format!("{b:03}.sig")))
            .collect();
        let tasks = bucket_tasks(self.num_buckets, self.spilled + self.docs.len(), threads)?;
        // Readers of all buckets merged at once share a quarter of the budget.
        let per_task = self.budget / 4 / tasks;
        let this = &*self;
        for_buckets(self.num_buckets, tasks, threads, |b| {
            this.write_bucket(b, &outputs[b], per_task)
        })?;
        for (b, out) in outputs.iter().enumerate() {
            let dest = sig_file_path(&self.folder, b, self.rank);
            if let Some(dir) = dest.parent() {
                fs::create_dir_all(dir).map_err(io_at(dir))?;
            }
            rename(out, &dest)?;
        }
        Ok(())
    }

    /// Remove the temporary files and write nothing. Does nothing after
    /// [`Self::finish`] or a previous abort.
    pub fn abort(&mut self) {
        if matches!(self.state, WriterState::Open | WriterState::Failed) {
            self.state = WriterState::Aborted;
        }
        self.cleanup();
    }

    fn cleanup(&mut self) {
        self.sigs = Vec::new();
        self.docs = Vec::new();
        if let Some(tmp) = self.tmp.take() {
            let _ = fs::remove_dir_all(&tmp);
            if let Some(parent) = tmp.parent() {
                // Only if no other writer still uses it.
                let _ = fs::remove_dir(parent);
            }
        }
    }
}

impl Drop for SigFileWriter {
    fn drop(&mut self) {
        self.cleanup();
    }
}

// ---------------------------------------------------------------------------
// Stage 2
// ---------------------------------------------------------------------------

/// The first value of record `line`.
fn first_value_at(f: &mut File, path: &Path, layout: SigLayout, line: u64) -> Result<u64> {
    let mut b = [0u8; 8];
    let w = layout.width();
    f.seek(SeekFrom::Start(line * as_u64(layout.record_size())))
        .map_err(io_at(path))?;
    f.read_exact(&mut b[..w]).map_err(io_at(path))?;
    Ok(layout.first_value(&b[..w]))
}

/// datatrove's `seek_to_start`: the first record whose first value is at
/// least `hash_min` (binary search; `records` if there is none).
fn seek_start(
    f: &mut File,
    path: &Path,
    layout: SigLayout,
    records: u64,
    hash_min: u64,
) -> Result<u64> {
    if hash_min == 0 || records == 0 {
        return Ok(0);
    }
    if first_value_at(f, path, layout, 0)? >= hash_min {
        return Ok(0);
    }
    if first_value_at(f, path, layout, records - 1)? < hash_min {
        return Ok(records);
    }
    let (mut lo, mut hi) = (0, records);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if first_value_at(f, path, layout, mid)? < hash_min {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    Ok(lo)
}

/// The rank of a signature file: its name without `.minhash.sig`.
fn rank_of(path: &Path) -> Result<u32> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let stem = name.strip_suffix(".minhash.sig").unwrap_or(name);
    stem.parse::<u32>().map_err(|_| {
        malformed(
            path,
            "the file name is not {rank}.minhash.sig with an integer rank below 2**32",
        )
    })
}

/// One non-empty signature file of a bucket.
#[derive(Debug)]
struct SigInput<'a> {
    path: &'a Path,
    pos: u32,
    rank: u32,
    records: u64,
}

/// datatrove's `read_sigs`: a signature file's records in
/// `[hash_min, hash_max)`, checking their order.
#[derive(Debug)]
struct SigReader {
    r: BufReader<File>,
    path: PathBuf,
    layout: SigLayout,
    left: u64,
    index: u64,
    hash_min: u64,
    hash_max: u64,
    raw: Vec<u8>,
    rec: Rec,
    next: Rec,
    started: bool,
}

impl SigReader {
    fn open(
        input: &SigInput<'_>,
        layout: SigLayout,
        hash_min: u64,
        hash_max: u64,
        buffer: usize,
    ) -> Result<Option<Self>> {
        let path = input.path;
        let mut f = File::open(path).map_err(io_at(path))?;
        let start = seek_start(&mut f, path, layout, input.records, hash_min)?;
        f.seek(SeekFrom::Start(start * as_u64(layout.record_size())))
            .map_err(io_at(path))?;
        let rec = Rec {
            pos: input.pos,
            rank: input.rank,
            ..Rec::new(layout)
        };
        started(Self {
            r: BufReader::with_capacity(buffer, f),
            path: path.to_owned(),
            layout,
            left: input.records - start,
            index: start,
            hash_min,
            hash_max,
            raw: vec![0; layout.record_size()],
            next: rec.clone(),
            rec,
            started: false,
        })
    }
}

impl Cursor for SigReader {
    fn rec(&self) -> &Rec {
        &self.rec
    }

    fn advance(&mut self) -> Result<bool> {
        if self.left == 0 {
            return Ok(false);
        }
        self.r
            .read_exact(&mut self.raw)
            .map_err(io_at(&self.path))?;
        self.left -= 1;
        let i = self.index;
        self.index += 1;
        self.next.doc = self.layout.decode(&self.raw, &mut self.next.band);
        let first = self.next.band[0];
        let before =
            self.started && (&self.next.band, self.next.doc) < (&self.rec.band, self.rec.doc);
        if first < self.hash_min || before {
            return Err(malformed(
                &self.path,
                format!(
                    "record {i} is out of order (signature files are sorted by hashes, then \
                     document)"
                ),
            ));
        }
        if first >= self.hash_max {
            self.left = 0;
            return Ok(false);
        }
        std::mem::swap(&mut self.rec, &mut self.next);
        self.started = true;
        Ok(true)
    }
}

#[derive(Debug)]
enum Stage2 {
    Sig(SigReader),
    Run(RunFile),
}

impl Cursor for Stage2 {
    fn rec(&self) -> &Rec {
        match self {
            Self::Sig(c) => c.rec(),
            Self::Run(c) => c.rec(),
        }
    }

    fn advance(&mut self) -> Result<bool> {
        match self {
            Self::Sig(c) => c.advance(),
            Self::Run(c) => c.advance(),
        }
    }
}

/// Stage-2 settings: the hash range of this bucket worker, the memory for
/// reader buffers, and how many files may be open at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketOptions {
    pub hash_min: u64,
    /// Exclusive.
    pub hash_max: u64,
    pub memory_budget: usize,
    pub max_open: usize,
}

impl BucketOptions {
    /// datatrove's full hash range, 64 MiB of buffers, 512 open files.
    pub fn new(config: &MinhashConfig) -> Self {
        Self {
            hash_min: 0,
            hash_max: SigLayout::new(config).default_hash_max(),
            memory_budget: 64 * MIB,
            max_open: 512,
        }
    }
}

/// datatrove's `get_worker_hash_range`: the `[hash_min, hash_max)` range of
/// bucket worker `bucket_worker` of `workers_per_bucket`, split at the
/// first values of evenly spaced records of the bucket's first signature
/// file (`None` when the bucket has no files).
pub fn worker_hash_range(
    first_file: Option<&Path>,
    config: &MinhashConfig,
    bucket_worker: usize,
    workers_per_bucket: usize,
) -> Result<(u64, u64)> {
    if workers_per_bucket == 0 {
        return Err(MinhashFileError::Invalid(
            "workers_per_bucket must be at least 1".to_owned(),
        ));
    }
    if bucket_worker >= workers_per_bucket {
        return Err(MinhashFileError::Invalid(format!(
            "bucket_worker {bucket_worker} is out of range for {workers_per_bucket} workers per \
             bucket"
        )));
    }
    let layout = SigLayout::new(config);
    let (mut lo, mut hi) = (0, layout.default_hash_max());
    let Some(path) = first_file.filter(|_| workers_per_bucket > 1) else {
        return Ok((lo, hi));
    };
    let size = file_size(path)?;
    let record = layout.record_size();
    if size % as_u64(record) != 0 {
        return Err(not_whole(path, size, record));
    }
    let lines = size / as_u64(record);
    let workers = as_u64(workers_per_bucket);
    if lines < workers {
        return Err(malformed(
            path,
            format!(
                "tried to use workers_per_bucket={workers_per_bucket} but there are only {lines} \
                 lines"
            ),
        ));
    }
    let step = lines / workers;
    let w = as_u64(bucket_worker);
    let mut f = File::open(path).map_err(io_at(path))?;
    if w > 0 {
        lo = first_value_at(&mut f, path, layout, step * w)?;
    }
    if w + 1 < workers {
        hi = first_value_at(&mut f, path, layout, step * (w + 1))?;
    }
    Ok((lo, hi))
}

/// The non-empty files of `files`, checked.
fn sig_inputs(files: &[PathBuf], layout: SigLayout) -> Result<Vec<SigInput<'_>>> {
    let record = layout.record_size();
    let mut inputs = Vec::with_capacity(files.len());
    for (pos, path) in files.iter().enumerate() {
        let size = file_size(path)?;
        if size == 0 {
            continue;
        }
        if size % as_u64(record) != 0 {
            return Err(not_whole(path, size, record));
        }
        inputs.push(SigInput {
            path,
            pos: u32::try_from(pos).map_err(|_| {
                MinhashFileError::Invalid("at most 2**32 signature files per bucket".to_owned())
            })?,
            rank: rank_of(path)?,
            records: size / as_u64(record),
        });
    }
    Ok(inputs)
}

fn open_sigs(
    inputs: &[SigInput<'_>],
    layout: SigLayout,
    opts: &BucketOptions,
    buffer: usize,
) -> Result<Vec<Stage2>> {
    let mut out = Vec::with_capacity(inputs.len());
    for input in inputs {
        out.extend(
            SigReader::open(input, layout, opts.hash_min, opts.hash_max, buffer)?.map(Stage2::Sig),
        );
    }
    Ok(out)
}

/// Stage 2 for one bucket worker (datatrove's `MinhashDedupBuckets.run`
/// without index files): merge the bucket's signature files (`files`, in
/// datatrove's reader order: sorted names) over `[hash_min, hash_max)` and
/// write the duplicate pairs to `output` (a `.dups` file). Returns the
/// number of pairs.
///
/// With more non-empty files than `max_open`, groups of `max_open` are
/// first merged into temporary runs next to `output`; the pairs are the
/// same as from one flat merge.
pub fn minhash_buckets(
    files: &[PathBuf],
    output: &Path,
    config: &MinhashConfig,
    opts: &BucketOptions,
) -> Result<u64> {
    if opts.max_open < 2 {
        return Err(MinhashFileError::Invalid(
            "max_open must be at least 2".to_owned(),
        ));
    }
    let layout = SigLayout::new(config);
    let Some(name) = output.file_name().and_then(|n| n.to_str()) else {
        return Err(MinhashFileError::Invalid(format!(
            "{}: the output must be a file name",
            output.display()
        )));
    };
    let parent = match output.parent() {
        Some(p) if !p.as_os_str().is_empty() => p,
        _ => Path::new("."),
    };
    let inputs = sig_inputs(files, layout)?;
    fs::create_dir_all(parent).map_err(io_at(parent))?;
    let tag = unique_tag();
    let mut temps = TempFiles::default();
    let mut runs_made = 0usize;
    let mut new_run = |t: &mut TempFiles| {
        runs_made += 1;
        t.add(parent.join(format!(".{name}.{tag}.run{runs_made}")))
    };
    let budget = opts.memory_budget;
    let kind = RunKind::stage2(layout);
    let cursors = if inputs.len() > opts.max_open {
        let mut runs = Vec::new();
        for group in inputs.chunks(opts.max_open) {
            let buffer = read_buffer(budget, group.len() + 1);
            let cursors = open_sigs(group, layout, opts, buffer)?;
            let run = new_run(&mut temps);
            write_run(cursors, kind, &run, buffer)?;
            runs.push(run);
        }
        let runs = reduce_runs(runs, opts.max_open, kind, budget, &mut temps, &mut new_run)?;
        let buffer = read_buffer(budget, runs.len() + 1);
        open_runs(&runs, kind, buffer)?
            .into_iter()
            .map(Stage2::Run)
            .collect()
    } else {
        open_sigs(&inputs, layout, opts, read_buffer(budget, inputs.len() + 1))?
    };
    let tmp = temps.add(parent.join(format!(".{name}.{tag}.tmp")));
    let mut w = create(&tmp, read_buffer(budget, cursors.len() + 1))?;
    let mut last = Rec::new(layout);
    let (mut have_last, mut pairs) = (false, 0u64);
    merge(cursors, by_pos, |rec| {
        if have_last && last.band == rec.band {
            let mut pair = [0u8; 16];
            for (dst, value) in pair
                .chunks_exact_mut(4)
                .zip([last.rank, last.doc, rec.rank, rec.doc])
            {
                dst.copy_from_slice(&value.to_le_bytes());
            }
            w.write_all(&pair).map_err(io_at(&tmp))?;
            pairs += 1;
        }
        last.band.clone_from(&rec.band);
        (last.rank, last.doc, have_last) = (rec.rank, rec.doc, true);
        Ok(())
    })?;
    close(w, &tmp)?;
    rename(&tmp, output)?;
    temps.forget(&tmp);
    Ok(pairs)
}

// ---------------------------------------------------------------------------
// Stage 3
// ---------------------------------------------------------------------------

/// Stage-3 settings (datatrove's `MinhashDedupCluster` arguments).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterOptions {
    pub save_cluster_id: bool,
    pub save_cluster_size: bool,
    /// Skip pairs whose first document is `(SENTINEL, SENTINEL)` (an index
    /// match).
    pub ignore_index_matches: bool,
}

/// What [`minhash_cluster`] found and wrote.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClusterSummary {
    /// Documents that appear in a pair.
    pub nodes: u64,
    /// Documents written to `.remove` files.
    pub removed: u64,
    /// Sets in the union-find (kept documents among the nodes).
    pub clusters: u64,
    /// `.remove` files written.
    pub files: u64,
}

/// datatrove's `union_set` / `set_size` over the documents that appear in
/// pairs: a dense id per `(file << 32) | doc`.
#[derive(Debug, Default)]
struct NodeSets {
    ids: FxHashMap<u64, u32>,
    parent: Vec<u32>,
    size: Vec<u32>,
    sentinel: Option<u32>,
}

impl NodeSets {
    /// datatrove's `parent(x)` adding `x`: its dense id.
    fn id(&mut self, key: u64) -> Result<u32> {
        match self.ids.entry(key) {
            Entry::Occupied(e) => Ok(*e.get()),
            Entry::Vacant(e) => {
                let id = u32::try_from(self.parent.len())
                    .ok()
                    .filter(|&i| i < u32::MAX)
                    .ok_or_else(|| {
                        MinhashFileError::Invalid(
                            "more than 2**32 - 2 documents appear in duplicate pairs".to_owned(),
                        )
                    })?;
                e.insert(id);
                self.parent.push(id);
                self.size.push(1);
                if key == SENTINEL_KEY {
                    self.sentinel = Some(id);
                }
                Ok(id)
            }
        }
    }

    /// The root of `x` (path halving; roots are as with datatrove's path
    /// compression).
    fn find(&mut self, mut x: u32) -> u32 {
        loop {
            let p = self.parent[x as usize];
            if p == x {
                return x;
            }
            let gp = self.parent[p as usize];
            self.parent[x as usize] = gp;
            x = gp;
        }
    }

    /// datatrove's `union(a, b)`: union by size, ties to `a`'s root, and
    /// the `(SENTINEL, SENTINEL)` root always wins.
    fn union(&mut self, a: u64, b: u64) -> Result<()> {
        let a = self.id(a)?;
        let b = self.id(b)?;
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return Ok(());
        }
        let (sa, sb) = (self.size[ra as usize], self.size[rb as usize]);
        if Some(rb) == self.sentinel || (Some(ra) != self.sentinel && sa < sb) {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] = sa + sb;
        Ok(())
    }
}

fn node_key(file: u32, doc: u32) -> u64 {
    (u64::from(file) << 32) | u64::from(doc)
}

/// `(file, doc)` of a node key.
fn split_key(key: u64) -> (u32, u32) {
    // The high and low halves.
    #[allow(clippy::cast_possible_truncation)]
    ((key >> 32) as u32, key as u32)
}

/// Union every pair of one `.dups` file into `sets`.
fn read_dups(path: &Path, ignore_index_matches: bool, sets: &mut NodeSets) -> Result<()> {
    let size = file_size(path)?;
    if size % 16 != 0 {
        return Err(not_whole(path, size, 16));
    }
    let f = File::open(path).map_err(io_at(path))?;
    let mut r = BufReader::with_capacity(DUPS_READ_BUFFER, f);
    let mut rec = [0u8; 16];
    for _ in 0..size / 16 {
        r.read_exact(&mut rec).map_err(io_at(path))?;
        let a = node_key(read_u32(&rec), read_u32(&rec[4..]));
        let b = node_key(read_u32(&rec[8..]), read_u32(&rec[12..]));
        if ignore_index_matches && a == SENTINEL_KEY {
            continue;
        }
        sets.union(a, b)?;
    }
    Ok(())
}

const OUTPUT_KINDS: [&str; 3] = ["remove", "clusters", "sizes"];

/// The stage-3 outputs of one file: `.remove`, `.clusters`, `.sizes`,
/// each opened (as a temporary file) on its first record.
#[derive(Debug)]
struct FileOutputs<'a> {
    folder: &'a Path,
    tag: &'a str,
    file: u32,
    open: [Option<(PathBuf, BufWriter<File>)>; 3],
    /// Closed temporary files and their final names.
    done: Vec<(PathBuf, PathBuf)>,
    removes: u64,
}

impl FileOutputs<'_> {
    fn write(&mut self, kind: usize, bytes: &[u8], temps: &mut TempFiles) -> Result<()> {
        if self.open[kind].is_none() {
            let name = format!(".{:06}.{}.{}.tmp", self.file, OUTPUT_KINDS[kind], self.tag);
            let tmp = temps.add(self.folder.join(name));
            let w = create(&tmp, 64 * KIB)?;
            self.open[kind] = Some((tmp, w));
        }
        let (tmp, w) = self.open[kind].as_mut().expect("opened above");
        w.write_all(bytes).map_err(io_at(tmp))
    }

    /// Close this file's outputs and move on to `file`.
    fn next_file(&mut self, file: u32) -> Result<()> {
        for (kind, slot) in self.open.iter_mut().enumerate() {
            if let Some((tmp, w)) = slot.take() {
                close(w, &tmp)?;
                let dest = self
                    .folder
                    .join(format!("{:06}.{}", self.file, OUTPUT_KINDS[kind]));
                self.done.push((tmp, dest));
                if kind == 0 {
                    self.removes += 1;
                }
            }
        }
        self.file = file;
        Ok(())
    }
}

fn pair_bytes(a: u32, b: u32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&a.to_le_bytes());
    out[4..].copy_from_slice(&b.to_le_bytes());
    out
}

/// Stage 3 (datatrove's `MinhashDedupCluster.run`): union the pairs of the
/// `.dups` files (in the order given; datatrove's is sorted names), then,
/// over the documents in (file, doc) order, write `{file:06d}.remove` for
/// every document that is not its set's root, and with the options
/// `{file:06d}.clusters` (cluster ids in order of first appearance) and
/// `{file:06d}.sizes` into `output_folder`. A file gets outputs only if it
/// has records for them.
pub fn minhash_cluster(
    dups: &[PathBuf],
    output_folder: &Path,
    opts: ClusterOptions,
) -> Result<ClusterSummary> {
    let mut sets = NodeSets::default();
    for path in dups {
        read_dups(path, opts.ignore_index_matches, &mut sets)?;
    }
    let mut nodes: Vec<(u64, u32)> = std::mem::take(&mut sets.ids).into_iter().collect();
    nodes.sort_unstable_by_key(|n| n.0);
    fs::create_dir_all(output_folder).map_err(io_at(output_folder))?;
    let tag = unique_tag();
    let mut temps = TempFiles::default();
    let mut cluster_of: Vec<u32> = if opts.save_cluster_id {
        vec![u32::MAX; sets.parent.len()]
    } else {
        Vec::new()
    };
    let mut next_cluster = 0u32;
    let mut summary = ClusterSummary {
        nodes: as_u64(nodes.len()),
        ..ClusterSummary::default()
    };
    let mut outputs = FileOutputs {
        folder: output_folder,
        tag: &tag,
        file: nodes.first().map_or(0, |n| split_key(n.0).0),
        open: [None, None, None],
        done: Vec::new(),
        removes: 0,
    };
    for &(key, id) in &nodes {
        let (file, doc) = split_key(key);
        if file != outputs.file {
            outputs.next_file(file)?;
        }
        let root = sets.find(id);
        if root == id {
            summary.clusters += 1;
        } else {
            outputs.write(0, &doc.to_le_bytes(), &mut temps)?;
            summary.removed += 1;
        }
        if opts.save_cluster_id {
            let c = &mut cluster_of[root as usize];
            if *c == u32::MAX {
                *c = next_cluster;
                next_cluster += 1;
            }
            outputs.write(1, &pair_bytes(doc, *c), &mut temps)?;
        }
        if opts.save_cluster_size {
            outputs.write(2, &pair_bytes(doc, sets.size[root as usize]), &mut temps)?;
        }
    }
    outputs.next_file(0)?;
    for (tmp, dest) in &outputs.done {
        rename(tmp, dest)?;
        temps.forget(tmp);
    }
    summary.files = outputs.removes;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, HashMap};

    use super::*;
    use crate::dedup::config::HashConfig;

    /// A fresh directory under the system temp dir, removed on drop.
    struct TestDir(PathBuf);

    impl TestDir {
        fn new(name: &str) -> Self {
            let p = std::env::temp_dir().join(format!("pyroid-files-{name}-{}", unique_tag()));
            fs::create_dir_all(&p).unwrap();
            Self(p)
        }

        fn path(&self, rel: &str) -> PathBuf {
            self.0.join(rel)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn temp_dirs_survive_writers_removing_the_parent() {
        // Each thread makes a temporary folder and removes it and, as a finished
        // writer does, the shared parent when that is empty. Before the retry
        // covered create_dir_all, a thread could get AlreadyExists for the parent.
        let dir = TestDir::new("racing-tmp");
        let parent = dir.path(SIG_TMP_DIR);
        std::thread::scope(|s| {
            for t in 0..8 {
                let parent = &parent;
                s.spawn(move || {
                    for _ in 0..2_000 {
                        let tmp = make_temp_dir(parent, &format!("t{t}")).unwrap();
                        fs::remove_dir(&tmp).unwrap();
                        let _ = fs::remove_dir(parent);
                    }
                });
            }
        });
    }

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn config(num_buckets: usize, per_bucket: usize, precision: Precision) -> MinhashConfig {
        MinhashConfig {
            num_buckets,
            hashes_per_bucket: per_bucket,
            hash_config: HashConfig {
                precision,
                ..HashConfig::default()
            },
            ..MinhashConfig::default()
        }
    }

    /// Random signatures for `n` documents, with values from a small range
    /// so bands collide; `None` for documents without one.
    fn random_docs(rng: &mut Rng, n: usize, nh: usize, p: Precision) -> Vec<Option<Vec<u64>>> {
        let top = match p {
            Precision::P32 => u64::from(u32::MAX),
            Precision::P64 => MERSENNE_PRIME - 1,
        };
        (0..n)
            .map(|_| {
                (rng.below(7) != 0).then(|| {
                    (0..nh)
                        .map(|_| match rng.below(20) {
                            0 => top,
                            1 => 0,
                            _ => rng.below(3) * 1_000 + 7,
                        })
                        .collect()
                })
            })
            .collect()
    }

    fn rows_of(docs: &[Option<Vec<u64>>], nh: usize) -> (Vec<u64>, Vec<bool>) {
        let mut rows = Vec::new();
        let mut present = Vec::new();
        for d in docs {
            match d {
                Some(s) => rows.extend_from_slice(s),
                None => rows.extend(std::iter::repeat_n(99, nh)),
            }
            present.push(d.is_some());
        }
        (rows, present)
    }

    /// Write one rank's signature files with `budget`, in batches of `batch`.
    fn write_rank(
        folder: &Path,
        rank: u32,
        cfg: &MinhashConfig,
        docs: &[Option<Vec<u64>>],
        budget: usize,
        batch: usize,
    ) -> usize {
        let nh = cfg.num_hashes();
        let mut w = SigFileWriter::new(folder, rank, cfg, budget).unwrap();
        for chunk in docs.chunks(batch.max(1)) {
            let (rows, present) = rows_of(chunk, nh);
            w.append(&rows, &present, Threads::Auto).unwrap();
        }
        let spills = w.spills();
        assert_eq!(w.finish(Threads::Auto).unwrap(), as_u64(docs.len()));
        spills
    }

    /// Every file under `dir` (relative path -> bytes).
    fn tree(dir: &Path) -> BTreeMap<String, Vec<u8>> {
        let mut out = BTreeMap::new();
        let mut stack = vec![dir.to_owned()];
        while let Some(d) = stack.pop() {
            for e in fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    stack.push(p);
                } else {
                    let rel = p.strip_prefix(dir).unwrap().to_string_lossy().into_owned();
                    out.insert(rel, fs::read(&p).unwrap());
                }
            }
        }
        out
    }

    /// datatrove's stage-1 bytes of one bucket: (band, doc) sorted.
    fn reference_sig_file(docs: &[Option<Vec<u64>>], cfg: &MinhashConfig, b: usize) -> Vec<u8> {
        let layout = SigLayout::new(cfg);
        let h = cfg.hashes_per_bucket;
        let mut recs: Vec<(Vec<u64>, u32)> = docs
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                d.as_ref()
                    .map(|s| (s[b * h..(b + 1) * h].to_vec(), u32::try_from(i).unwrap()))
            })
            .collect();
        recs.sort();
        let mut out = Vec::new();
        for (band, doc) in recs {
            layout.encode(&band, doc, &mut out);
        }
        out
    }

    #[test]
    fn records_round_trip_both_precisions() {
        let l64 = SigLayout::new(&config(1, 2, Precision::P64));
        let mut out = Vec::new();
        l64.encode(&[0x0102_0304_0506_0708, 9], 0xaabb_ccdd, &mut out);
        // struct.pack("<2QI", 0x0102030405060708, 9, 0xaabbccdd)
        assert_eq!(
            out,
            [
                8, 7, 6, 5, 4, 3, 2, 1, 9, 0, 0, 0, 0, 0, 0, 0, 0xdd, 0xcc, 0xbb, 0xaa
            ]
        );
        assert_eq!(l64.record_size(), 20);
        let mut band = [0; 2];
        assert_eq!(l64.decode(&out, &mut band), 0xaabb_ccdd);
        assert_eq!(band, [0x0102_0304_0506_0708, 9]);

        let l32 = SigLayout::new(&config(1, 3, Precision::P32));
        let mut out = Vec::new();
        l32.encode(&[u64::from(u32::MAX), 1, 0x0102_0304], 7, &mut out);
        // struct.pack("<3II", 2**32 - 1, 1, 0x01020304, 7)
        assert_eq!(
            out,
            [255, 255, 255, 255, 1, 0, 0, 0, 4, 3, 2, 1, 7, 0, 0, 0]
        );
        let mut band = [0; 3];
        assert_eq!(l32.decode(&out, &mut band), 7);
        assert_eq!(band, [u64::from(u32::MAX), 1, 0x0102_0304]);
        assert_eq!(l32.first_value(&out), u64::from(u32::MAX));
        assert_eq!(l32.default_hash_max(), u64::from(u32::MAX));
        assert_eq!(l64.default_hash_max(), MERSENNE_PRIME);
    }

    #[test]
    fn writer_sorts_by_band_then_doc() {
        for (p, h) in [
            (Precision::P64, 3),
            (Precision::P32, 2),
            (Precision::P64, 1),
        ] {
            let cfg = config(4, h, p);
            let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ as_u64(h));
            let docs = random_docs(&mut rng, 300, cfg.num_hashes(), p);
            let dir = TestDir::new("sort");
            assert_eq!(write_rank(&dir.0, 3, &cfg, &docs, 1 << 30, 37), 0);
            let files = tree(&dir.0);
            assert_eq!(files.len(), 4, "{:?}", files.keys());
            for b in 0..4 {
                let got = &files[&format!("bucket_{b:03}/00003.minhash.sig")];
                assert_eq!(got, &reference_sig_file(&docs, &cfg, b), "bucket {b}");
            }
        }
    }

    #[test]
    fn spill_and_merge_equal_no_spill() {
        let cfg = config(3, 2, Precision::P64);
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        let docs = random_docs(&mut rng, 200, cfg.num_hashes(), Precision::P64);
        let flat = TestDir::new("flat");
        assert_eq!(write_rank(&flat.0, 0, &cfg, &docs, 1 << 30, 1000), 0);
        // One record per run: far more runs than SPILL_FANIN, so the final
        // merge runs in passes.
        let tiny = TestDir::new("tiny");
        assert!(write_rank(&tiny.0, 0, &cfg, &docs, 1, 13) > SPILL_FANIN * SPILL_FANIN / 8);
        // A few records per run.
        let some = TestDir::new("some");
        let spills = write_rank(&some.0, 0, &cfg, &docs, 2_000, 7);
        assert!(spills > 3 && spills < SPILL_FANIN, "{spills}");
        let want = tree(&flat.0);
        assert_eq!(tree(&tiny.0), want);
        assert_eq!(tree(&some.0), want);
        assert!(!flat.path(SIG_TMP_DIR).exists());
        assert!(!tiny.path(SIG_TMP_DIR).exists());
    }

    #[test]
    fn empty_rank_writes_empty_files() {
        let cfg = config(3, 2, Precision::P32);
        let dir = TestDir::new("empty");
        let mut w = SigFileWriter::new(&dir.0, 12, &cfg, 1 << 20).unwrap();
        // Documents without signatures still count.
        let range = w.append(&[0; 12], &[false, false], Threads::ONE).unwrap();
        assert_eq!(range, 0..2);
        assert_eq!(w.finish(Threads::ONE).unwrap(), 2);
        let files = tree(&dir.0);
        assert_eq!(files.len(), 3);
        assert!(files.values().all(Vec::is_empty));
        assert!(files.contains_key("bucket_002/00012.minhash.sig"));
        let err = w.finish(Threads::ONE).unwrap_err();
        assert_eq!(err.to_string(), "finish() was already called");
        w.abort(); // no-op after finish
        assert_eq!(tree(&dir.0).len(), 3);
    }

    #[test]
    fn abort_and_drop_write_nothing() {
        let cfg = config(2, 1, Precision::P64);
        let dir = TestDir::new("abort");
        let mut rng = Rng(77);
        let docs = random_docs(&mut rng, 50, 2, Precision::P64);
        let (rows, present) = rows_of(&docs, 2);
        let mut w = SigFileWriter::new(&dir.0, 0, &cfg, 1).unwrap();
        w.append(&rows, &present, Threads::Auto).unwrap();
        assert!(w.spills() > 0);
        assert!(dir.path(SIG_TMP_DIR).exists());
        w.abort();
        w.abort();
        assert!(tree(&dir.0).is_empty());
        assert!(!dir.path(SIG_TMP_DIR).exists());
        assert_eq!(
            w.append(&rows, &present, Threads::Auto)
                .unwrap_err()
                .to_string(),
            "the writer was aborted"
        );
        {
            let mut w = SigFileWriter::new(&dir.0, 1, &cfg, 1).unwrap();
            w.append(&rows, &present, Threads::Auto).unwrap();
        }
        assert!(tree(&dir.0).is_empty());
        assert!(!dir.path(SIG_TMP_DIR).exists());
    }

    /// Write raw records of one file.
    fn write_raw(path: &Path, layout: SigLayout, recs: &[(Vec<u64>, u32)]) {
        let mut out = Vec::new();
        for (band, doc) in recs {
            layout.encode(band, *doc, &mut out);
        }
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, out).unwrap();
    }

    #[test]
    fn seek_matches_partition_point() {
        let dir = TestDir::new("seek");
        for p in [Precision::P64, Precision::P32] {
            let layout = SigLayout::new(&config(1, 2, p));
            let firsts = [3u64, 3, 3, 5, 8, 8, 9, 9, 9, 9, 12, 20, 20];
            let recs: Vec<(Vec<u64>, u32)> = firsts
                .iter()
                .zip(0..)
                .map(|(&v, d)| (vec![v, 1], d))
                .collect();
            let path = dir.path("00000.minhash.sig");
            write_raw(&path, layout, &recs);
            let mut f = File::open(&path).unwrap();
            for hash_min in 0..25 {
                let got = seek_start(&mut f, &path, layout, as_u64(recs.len()), hash_min).unwrap();
                let want = firsts.partition_point(|&v| v < hash_min);
                assert_eq!(got, as_u64(want), "{hash_min}");
            }
        }
    }

    /// One file's sorted `(band, doc)` records.
    type Records = Vec<(Vec<u64>, u32)>;
    /// A signature file in memory: its rank and records.
    type RankFile = (u32, Records);

    /// datatrove's stage 2 over in-memory files: records in range, stable
    /// sorted by (band, reader position), pairs of consecutive equal bands.
    fn reference_pairs(files: &[RankFile], lo: u64, hi: u64) -> Vec<u8> {
        let mut all: Vec<(&Vec<u64>, usize, u32, u32)> = Vec::new();
        for (pos, (rank, recs)) in files.iter().enumerate() {
            let start = recs.partition_point(|r| r.0[0] < lo);
            for (band, doc) in &recs[start..] {
                if band[0] >= hi {
                    break;
                }
                all.push((band, pos, *rank, *doc));
            }
        }
        all.sort_by(|a, b| a.0.cmp(b.0).then(a.1.cmp(&b.1)));
        let mut out = Vec::new();
        for w in all.windows(2) {
            if w[0].0 == w[1].0 {
                for v in [w[0].2, w[0].3, w[1].2, w[1].3] {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        out
    }

    /// Several ranks of one bucket (rank -> sorted records), one of them
    /// empty, written as `bucket_000/{rank}.minhash.sig`.
    fn bucket_files(
        dir: &Path,
        cfg: &MinhashConfig,
        seed: u64,
        ranks: &[u32],
    ) -> (Vec<PathBuf>, Vec<RankFile>) {
        let layout = SigLayout::new(cfg);
        let mut rng = Rng(seed);
        let mut paths = Vec::new();
        let mut files = Vec::new();
        for &rank in ranks {
            let n = if rank == 2 { 0 } else { 20 + rng.below(40) };
            let docs = random_docs(
                &mut rng,
                usize::try_from(n).unwrap(),
                cfg.hashes_per_bucket,
                cfg.hash_config.precision,
            );
            let mut recs: Vec<(Vec<u64>, u32)> = docs
                .iter()
                .zip(0..)
                .filter_map(|(d, i)| d.clone().map(|s| (s, i)))
                .collect();
            recs.sort();
            let path = sig_file_path(dir, 0, rank);
            write_raw(&path, layout, &recs);
            paths.push(path);
            files.push((rank, recs));
        }
        (paths, files)
    }

    fn run_buckets(
        files: &[PathBuf],
        out: &Path,
        cfg: &MinhashConfig,
        opts: &BucketOptions,
    ) -> (u64, Vec<u8>) {
        let n = minhash_buckets(files, out, cfg, opts).unwrap();
        (n, fs::read(out).unwrap())
    }

    #[test]
    fn buckets_match_reference_and_hierarchical_merge() {
        for (p, h) in [
            (Precision::P64, 2),
            (Precision::P32, 1),
            (Precision::P64, 3),
        ] {
            let cfg = config(1, h, p);
            let dir = TestDir::new("buckets");
            let ranks = [0, 1, 2, 3, 4, 5, 6, 7, 9, 12, 30];
            let (paths, files) = bucket_files(&dir.0, &cfg, 0x1234 + as_u64(h), &ranks);
            let full = BucketOptions::new(&cfg);
            let want = reference_pairs(&files, 0, full.hash_max);
            assert!(want.len() > 16 * 20, "few pairs: {}", want.len() / 16);
            let out = dir.path("dups/00000_00.dups");
            let (n, flat) = run_buckets(&paths, &out, &cfg, &full);
            assert_eq!(flat, want);
            assert_eq!(n, as_u64(want.len() / 16));
            for max_open in [2, 3, 4, 10] {
                for memory_budget in [0, 1 << 20] {
                    let opts = BucketOptions {
                        memory_budget,
                        max_open,
                        ..full
                    };
                    assert_eq!(run_buckets(&paths, &out, &cfg, &opts).1, want, "{max_open}");
                }
            }
            // Only the output is left.
            assert_eq!(tree(&dir.path("dups")).len(), 1);
        }
    }

    #[test]
    fn precision_32_max_first_value_never_matches() {
        let cfg = config(1, 2, Precision::P32);
        let layout = SigLayout::new(&cfg);
        let dir = TestDir::new("p32max");
        let m = u64::from(u32::MAX);
        let a = dir.path("00000.minhash.sig");
        let b = dir.path("00001.minhash.sig");
        write_raw(
            &a,
            layout,
            &[(vec![7, 1], 0), (vec![m, 1], 1), (vec![m, 1], 2)],
        );
        write_raw(&b, layout, &[(vec![7, 1], 4), (vec![m, 1], 5)]);
        let out = dir.path("o.dups");
        let opts = BucketOptions::new(&cfg);
        let (n, bytes) = run_buckets(&[a.clone(), b.clone()], &out, &cfg, &opts);
        assert_eq!(n, 1);
        assert_eq!(bytes, [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0, 0, 0]);
        // An explicit full range includes them: (0:1, 0:2) and (0:2, 1:5).
        let all = BucketOptions {
            hash_max: u64::MAX,
            ..opts
        };
        assert_eq!(run_buckets(&[a, b], &out, &cfg, &all).0, 3);
    }

    #[test]
    fn hash_ranges_split_the_pairs() {
        for p in [Precision::P64, Precision::P32] {
            let cfg = config(1, 2, p);
            let dir = TestDir::new("ranges");
            let (paths, files) = bucket_files(&dir.0, &cfg, 99, &[0, 1, 2, 3]);
            let first = files[0].1.iter().map(|r| r.0[0]).collect::<Vec<_>>();
            let full = BucketOptions::new(&cfg);
            let want = reference_pairs(&files, 0, full.hash_max);
            for workers in 1..=5 {
                let mut joined = Vec::new();
                let mut prev_hi = 0;
                for w in 0..workers {
                    let (lo, hi) =
                        worker_hash_range(Some(paths[0].as_path()), &cfg, w, workers).unwrap();
                    let step = first.len() / workers;
                    assert_eq!(lo, if w == 0 { 0 } else { first[step * w] });
                    let want_hi = if w + 1 == workers {
                        full.hash_max
                    } else {
                        first[step * (w + 1)]
                    };
                    assert_eq!(hi, want_hi);
                    // Contiguous ranges.
                    assert_eq!(lo, prev_hi);
                    prev_hi = hi;
                    let opts = BucketOptions {
                        hash_min: lo,
                        hash_max: hi,
                        max_open: 2,
                        ..full
                    };
                    let out = dir.path(&format!("d/{w}.dups"));
                    let part = run_buckets(&paths, &out, &cfg, &opts).1;
                    assert_eq!(part, reference_pairs(&files, lo, hi));
                    joined.extend(part);
                }
                assert_eq!(joined, want, "{workers} workers");
            }
            // The split points repeat first values: the boundary record and
            // every record with the same first value go to the next worker.
            assert!(first.windows(2).any(|w| w[0] == w[1]));
            assert_eq!(
                worker_hash_range(None, &cfg, 1, 3).unwrap(),
                (0, full.hash_max)
            );
            assert_eq!(
                worker_hash_range(Some(paths[0].as_path()), &cfg, 0, 1).unwrap(),
                (0, full.hash_max)
            );
        }
    }

    #[test]
    fn hash_range_errors() {
        let cfg = config(1, 1, Precision::P64);
        let dir = TestDir::new("rangeerr");
        let path = dir.path("00000.minhash.sig");
        write_raw(&path, SigLayout::new(&cfg), &[(vec![1], 0), (vec![2], 1)]);
        let path = path.as_path();
        let e = worker_hash_range(Some(path), &cfg, 0, 3).unwrap_err();
        assert!(
            e.to_string()
                .ends_with("tried to use workers_per_bucket=3 but there are only 2 lines"),
            "{e}"
        );
        assert!(worker_hash_range(Some(path), &cfg, 2, 2).is_err());
        assert!(worker_hash_range(Some(path), &cfg, 0, 0).is_err());
        assert_eq!(
            worker_hash_range(Some(path), &cfg, 1, 2).unwrap(),
            (2, MERSENNE_PRIME)
        );
        let missing = dir.path("nope.sig");
        match worker_hash_range(Some(missing.as_path()), &cfg, 0, 2).unwrap_err() {
            MinhashFileError::Io { path, err } => {
                assert_eq!(path, missing);
                assert_eq!(err.kind(), io::ErrorKind::NotFound);
            }
            e => panic!("{e}"),
        }
    }

    #[test]
    fn malformed_signature_files() {
        let cfg = config(1, 2, Precision::P64);
        let layout = SigLayout::new(&cfg);
        let dir = TestDir::new("bad");
        let out = dir.path("o.dups");
        let opts = BucketOptions::new(&cfg);
        let bad = dir.path("00001.minhash.sig");
        fs::write(&bad, [0u8; 41]).unwrap();
        let e = minhash_buckets(std::slice::from_ref(&bad), &out, &cfg, &opts).unwrap_err();
        assert_eq!(
            e.to_string(),
            format!(
                "{}: 41 bytes is not a whole number of 20-byte records",
                bad.display()
            )
        );
        assert!(matches!(e, MinhashFileError::Malformed { .. }));
        // Out of order: by band, and by document within a band.
        for recs in [
            vec![(vec![5, 1], 0), (vec![4, 9], 1)],
            vec![(vec![5, 1], 3), (vec![5, 1], 2)],
        ] {
            write_raw(&bad, layout, &recs);
            let e = minhash_buckets(std::slice::from_ref(&bad), &out, &cfg, &opts).unwrap_err();
            assert!(e.to_string().contains("record 1 is out of order"), "{e}");
        }
        // Equal (band, doc) records are allowed, as in datatrove.
        write_raw(&bad, layout, &[(vec![5, 1], 3), (vec![5, 1], 3)]);
        assert_eq!(
            minhash_buckets(std::slice::from_ref(&bad), &out, &cfg, &opts).unwrap(),
            1
        );
        let named = dir.path("x.minhash.sig");
        write_raw(&named, layout, &[(vec![5, 1], 3)]);
        let e = minhash_buckets(&[named], &out, &cfg, &opts).unwrap_err();
        assert!(e.to_string().contains("the file name is not"), "{e}");
        // An empty file is skipped whatever its name.
        let empty = dir.path("empty");
        fs::write(&empty, b"").unwrap();
        assert_eq!(minhash_buckets(&[empty], &out, &cfg, &opts).unwrap(), 0);
        assert_eq!(fs::read(&out).unwrap(), b"");
        let missing = dir.path("00009.minhash.sig");
        let e = minhash_buckets(&[missing], &out, &cfg, &opts).unwrap_err();
        assert!(matches!(e, MinhashFileError::Io { .. }), "{e}");
        let one = BucketOptions {
            max_open: 1,
            ..opts
        };
        assert!(minhash_buckets(&[], &out, &cfg, &one).is_err());
        // Nothing but the last good output is left behind.
        let left: Vec<String> = tree(&dir.0)
            .into_keys()
            .filter(|k| k.starts_with('.'))
            .collect();
        assert!(left.is_empty(), "{left:?}");
    }

    type Node = (u32, u32);
    const S: Node = (SENTINEL, SENTINEL);

    /// Direct transcription of datatrove's `MinhashDedupCluster.run`
    /// (`parent`, `union` and the output loop): file name -> bytes.
    fn reference_cluster(
        pairs: &[(Node, Node)],
        opts: ClusterOptions,
    ) -> BTreeMap<String, Vec<u8>> {
        fn parent(u: &mut HashMap<Node, Node>, x: Node) -> Node {
            match u.get(&x).copied() {
                Some(p) if p != x => {
                    let r = parent(u, p);
                    u.insert(x, r);
                    r
                }
                _ => {
                    u.insert(x, x);
                    x
                }
            }
        }
        let mut union_set: HashMap<Node, Node> = HashMap::new();
        let mut set_size: HashMap<Node, u32> = HashMap::new();
        for &(a, b) in pairs {
            if opts.ignore_index_matches && a == S {
                continue;
            }
            let mut root_a = parent(&mut union_set, a);
            let mut root_b = parent(&mut union_set, b);
            if root_a != root_b {
                let size_a = set_size.get(&root_a).copied().unwrap_or(1);
                let size_b = set_size.get(&root_b).copied().unwrap_or(1);
                if root_b == S || (root_a != S && size_a < size_b) {
                    std::mem::swap(&mut root_a, &mut root_b);
                }
                union_set.insert(root_b, root_a);
                set_size.insert(root_a, size_a + size_b);
                set_size.remove(&root_b);
            }
        }
        let mut nodes: Vec<Node> = union_set.keys().copied().collect();
        nodes.sort_unstable();
        let mut out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
        let mut cluster_ids: HashMap<Node, u32> = HashMap::new();
        for node in nodes {
            let (file, doc) = node;
            let p = parent(&mut union_set, node);
            if node != p {
                out.entry(format!("{file:06}.remove"))
                    .or_default()
                    .extend(doc.to_le_bytes());
            }
            if opts.save_cluster_id {
                let next = u32::try_from(cluster_ids.len()).unwrap();
                let c = *cluster_ids.entry(p).or_insert(next);
                out.entry(format!("{file:06}.clusters"))
                    .or_default()
                    .extend(pair_bytes(doc, c));
            }
            if opts.save_cluster_size {
                let s = set_size[&p];
                out.entry(format!("{file:06}.sizes"))
                    .or_default()
                    .extend(pair_bytes(doc, s));
            }
        }
        out
    }

    fn write_dups(path: &Path, pairs: &[(Node, Node)]) {
        let mut out = Vec::new();
        for &((f1, d1), (f2, d2)) in pairs {
            for v in [f1, d1, f2, d2] {
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        fs::write(path, out).unwrap();
    }

    fn all_options() -> Vec<ClusterOptions> {
        let mut v = Vec::new();
        for i in 0..8u8 {
            v.push(ClusterOptions {
                save_cluster_id: i & 1 != 0,
                save_cluster_size: i & 2 != 0,
                ignore_index_matches: i & 4 != 0,
            });
        }
        v
    }

    #[test]
    fn cluster_matches_reference_union_find() {
        let mut rng = Rng(0xdead_beef_cafe_f00d);
        for round in 0..150u64 {
            let files = 1 + rng.below(5);
            let docs = 2 + round % 30;
            let node = |rng: &mut Rng| -> Node {
                if rng.below(25) == 0 {
                    S
                } else {
                    (
                        u32::try_from(rng.below(files)).unwrap() * 3,
                        u32::try_from(rng.below(docs)).unwrap(),
                    )
                }
            };
            let mut pairs = Vec::new();
            for _ in 0..rng.below(80) {
                let (a, b) = (node(&mut rng), node(&mut rng));
                if a != b {
                    pairs.push((a, b));
                }
            }
            let dir = TestDir::new("cluster");
            // Split the pairs over three .dups files, read in order.
            let cut1 = pairs.len() / 3;
            let cut2 = 2 * pairs.len() / 3;
            let dups: Vec<PathBuf> = [&pairs[..cut1], &pairs[cut1..cut2], &pairs[cut2..]]
                .iter()
                .enumerate()
                .map(|(i, part)| {
                    let p = dir.path(&format!("{i:05}_00.dups"));
                    write_dups(&p, part);
                    p
                })
                .collect();
            for (k, opts) in all_options().into_iter().enumerate() {
                let out = dir.path(&format!("out-{k}"));
                let summary = minhash_cluster(&dups, &out, opts).unwrap();
                let want = reference_cluster(&pairs, opts);
                let got = tree(&out);
                assert_eq!(got, want, "{pairs:?} {opts:?}");
                let removed: usize = want
                    .iter()
                    .filter(|(k, _)| k.ends_with(".remove"))
                    .map(|(_, v)| v.len() / 4)
                    .sum();
                assert_eq!(summary.removed, as_u64(removed));
                assert_eq!(summary.clusters, summary.nodes - summary.removed);
                let remove_files = want.keys().filter(|k| k.ends_with(".remove")).count();
                assert_eq!(summary.files, as_u64(remove_files));
            }
        }
    }

    #[test]
    fn sentinel_is_always_root() {
        let dir = TestDir::new("sentinel");
        // {(0,1), (0,2), (0,3)} is larger, but joins the index match's set.
        let pairs = [
            ((0, 1), (0, 2)),
            ((0, 2), (0, 3)),
            (S, (1, 5)),
            ((1, 5), (0, 3)),
        ];
        let dups = dir.path("00000_00.dups");
        write_dups(&dups, &pairs);
        let opts = ClusterOptions {
            save_cluster_id: true,
            save_cluster_size: true,
            ignore_index_matches: false,
        };
        let out = dir.path("out");
        let s = minhash_cluster(std::slice::from_ref(&dups), &out, opts).unwrap();
        assert_eq!(
            s,
            ClusterSummary {
                nodes: 5,
                removed: 4,
                clusters: 1,
                files: 2
            }
        );
        let got = tree(&out);
        assert_eq!(got, reference_cluster(&pairs, opts));
        assert_eq!(got["000000.remove"], [1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0]);
        assert_eq!(got["4294967295.sizes"], pair_bytes(SENTINEL, 5));
        // Ignoring index matches drops the pair that starts with it.
        let ignore = ClusterOptions {
            ignore_index_matches: true,
            ..opts
        };
        let out = dir.path("out2");
        let s = minhash_cluster(&[dups], &out, ignore).unwrap();
        assert_eq!(s.nodes, 4);
        assert_eq!(tree(&out), reference_cluster(&pairs, ignore));
    }

    #[test]
    fn malformed_dups_files() {
        let dir = TestDir::new("baddups");
        let bad = dir.path("00000_00.dups");
        fs::write(&bad, [0u8; 20]).unwrap();
        let out = dir.path("out");
        let e = minhash_cluster(std::slice::from_ref(&bad), &out, ClusterOptions::default())
            .unwrap_err();
        assert_eq!(
            e.to_string(),
            format!(
                "{}: 20 bytes is not a whole number of 16-byte records",
                bad.display()
            )
        );
        let missing = dir.path("00001_00.dups");
        let e = minhash_cluster(&[missing], &out, ClusterOptions::default()).unwrap_err();
        assert!(matches!(e, MinhashFileError::Io { .. }));
        // A self pair is one set of one document (datatrove raises KeyError).
        write_dups(&bad, &[((0, 4), (0, 4))]);
        let opts = ClusterOptions {
            save_cluster_size: true,
            ..ClusterOptions::default()
        };
        let s = minhash_cluster(&[bad], &out, opts).unwrap();
        assert_eq!(
            s,
            ClusterSummary {
                nodes: 1,
                removed: 0,
                clusters: 1,
                files: 0
            }
        );
        assert_eq!(tree(&out)["000000.sizes"], pair_bytes(4, 1));
    }
}
