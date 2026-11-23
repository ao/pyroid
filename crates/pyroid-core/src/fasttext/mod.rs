// pyroid's port of fastText inference (https://github.com/facebookresearch/fastText).
//
// fastText is Copyright (c) 2016-present, Facebook, Inc., and released under
// the MIT License; this module translates its model format and inference code
// (args.cc, dictionary.cc, densematrix.cc, quantmatrix.cc, productquantizer.cc,
// loss.cc, model.cc, fasttext.cc) to Rust. Pyroid is also MIT-licensed.

//! fastText inference: load `.bin` / `.ftz` models and predict labels, with
//! results identical to fastText's own C++.
//!
//! - [`Model::load`] reads the header, dictionary and matrices (dense or
//!   product-quantized) in one buffered pass and validates every size and id
//!   range, so a malformed or truncated file is a [`LoadError`], never a
//!   panic or an out-of-bounds read.
//! - [`Model::predict`] reproduces `FastText::predictLine` for one line:
//!   tokenization, the `</s>` token, character n-gram hashing (with the
//!   signed-char FNV quirk), word n-gram buckets, the averaged hidden vector,
//!   the softmax / hierarchical softmax / one-vs-all / negative-sampling
//!   heads, `k` and `threshold`, and libstdc++'s tie order.
//!
//! A [`Model`] is immutable and `Sync`: share one across threads.
//!
//! ```
//! use pyroid_core::fasttext::Model;
//! # fn demo(path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
//! let model = Model::load(path)?;
//! for p in model.predict("Bonjour tout le monde", 2, 0.0)? {
//!     println!("{} {}", String::from_utf8_lossy(model.label(p.label)), p.prob);
//! }
//! # Ok(()) }
//! ```

// Integer casts mirror fastText's C++ conversions on purpose (int32 ids and
// sizes, uint32 hashes, int64 counts); each one is either range-checked at
// load time or wraps exactly as the C++ does.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless
)]

mod args;
mod dict;
mod error;
mod io;
mod loss;
mod matrix;

use std::cell::RefCell;
use std::fs::File;
use std::path::Path;

pub use args::{Args, LossKind, ModelKind};
pub use dict::{EOS, LABEL_PREFIX, hash};
pub use error::{LoadError, PredictError};

use dict::{Dictionary, LineScratch};
use io::Reader;
use loss::{Head, Scored};
use matrix::Matrix;

/// `fasttext.tokenize(text)`: every token `readWord` yields for the whole
/// text, with `</s>` for each `'\n'` (no end-of-line token is added and a
/// literal `</s>` does not stop it). Tokens are split on fastText's
/// separators: space, `\t`, `\v`, `\f`, `\r`, `\n` and NUL.
pub fn tokenize(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    for line in text.split_inclusive('\n') {
        let newline = line.ends_with('\n');
        let body = if newline {
            &line[..line.len() - 1]
        } else {
            line
        };
        out.extend(
            body.split(|c: char| c.is_ascii() && dict::is_separator(c as u8))
                .filter(|t| !t.is_empty()),
        );
        if newline {
            out.push("</s>");
        }
    }
    out
}

/// `FASTTEXT_FILEFORMAT_MAGIC_INT32`.
pub const MAGIC: i32 = 793_712_314;
/// `FASTTEXT_VERSION`: the newest format version fastText (and pyroid) reads.
pub const VERSION: i32 = 12;

/// The largest `dim` of a model with no weights at all (both matrices dense
/// with 0 rows; fastText could only train one from a file of labels without
/// a newline, and its vectors are all zeros). Real models store `dim` floats
/// per matrix row, so their `dim` is bounded by the file size.
const MAX_UNBACKED_DIM: usize = 1 << 16;

/// One predicted label.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Prediction {
    /// fastText's probability: `exp(log(p + 1e-5))` in float, so it can
    /// slightly exceed 1.
    pub prob: f32,
    /// Label index: [`Model::label`] gives its text.
    pub label: usize,
}

/// Per-thread buffers, so a batch allocates nothing per document.
#[derive(Debug, Default)]
struct Scratch {
    line: LineScratch,
    hidden: Vec<f32>,
    output: Vec<f32>,
    heap: Vec<Scored>,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// A loaded fastText model.
#[derive(Debug)]
pub struct Model {
    version: i32,
    args: Args,
    dict: Dictionary,
    input: Matrix,
    output: Matrix,
    qout: bool,
    /// The prediction head (supervised models only).
    head: Option<Head>,
}

impl Model {
    /// Load a `.bin` or `.ftz` model from a file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let mut file = File::open(path).map_err(LoadError::Io)?;
        let meta = file.metadata().map_err(LoadError::Io)?;
        if meta.is_dir() {
            // Let the OS say why (EISDIR), whatever size it reports for a directory.
            let mut probe = [0u8; 1];
            std::io::Read::read(&mut file, &mut probe).map_err(LoadError::Io)?;
        }
        Self::read(file, meta.len())
    }

    /// Load a model from bytes in memory.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, LoadError> {
        Self::read(bytes, bytes.len() as u64)
    }

    /// Load a model from a reader of `len` bytes (`FastText::loadModel`).
    pub fn read(source: impl std::io::Read, len: u64) -> Result<Self, LoadError> {
        let mut r = Reader::new(source, len);
        // checkModel
        let magic = r.i32("magic number")?;
        if magic != MAGIC {
            return Err(LoadError::WrongFormat {
                magic,
                version: None,
            });
        }
        let version = r.i32("format version")?;
        if version > VERSION {
            return Err(LoadError::WrongFormat {
                magic,
                version: Some(version),
            });
        }
        let mut args = Args::load(&mut r)?;
        if version == 11 && args.model == ModelKind::Supervised {
            // Backward compatibility: old supervised models do not use char n-grams.
            args.maxn = 0;
        }
        let dict = Dictionary::load(&mut r, &args)?;

        let quant_input = r.bool("quant_input flag")?;
        let input = Matrix::load(&mut r, quant_input, "input matrix")?;
        if !quant_input && dict.is_pruned() {
            return Err(LoadError::Invalid(
                "Invalid model file.\nPlease download the updated model from www.fasttext.cc.\n\
                 See issue #332 on Github for more information.\n"
                    .to_owned(),
            ));
        }
        let qout = r.bool("qout flag")?;
        let output = Matrix::load(&mut r, quant_input && qout, "output matrix")?;

        let model = Self {
            version,
            head: None,
            args,
            dict,
            input,
            output,
            qout,
        };
        model.validate()?.with_head()
    }

    /// Reject files whose sizes or ids would make fastText read out of bounds.
    fn validate(self) -> Result<Self, LoadError> {
        let dim = usize::try_from(self.args.dim)
            .map_err(|_| LoadError::Invalid(format!("negative dim {}", self.args.dim)))?;
        for (name, m) in [("input", &self.input), ("output", &self.output)] {
            if m.cols() != dim {
                return Err(LoadError::Invalid(format!(
                    "{name} matrix has {} columns, but dim is {dim}",
                    m.cols()
                )));
            }
        }
        // Every vector has `dim` floats, and a matrix row (or a quantizer's
        // centroids) stores at least that many in the file, except when both
        // matrices are dense and empty: then a few header bytes could make
        // every word or sentence vector allocate up to 8 GiB of zeros.
        let unbacked = [&self.input, &self.output]
            .iter()
            .all(|m| m.rows() == 0 && m.as_dense().is_some());
        if unbacked && dim > MAX_UNBACKED_DIM {
            return Err(LoadError::Invalid(format!(
                "dim is {dim} but both matrices are empty (at most {MAX_UNBACKED_DIM} without \
                 any weights)"
            )));
        }
        if self.dict.has_negative_pruned_ids() {
            return Err(LoadError::Invalid(
                "the pruning index maps n-grams to negative rows".to_owned(),
            ));
        }
        let need = self.dict.rows_needed();
        if (self.input.rows() as u64) < need {
            return Err(LoadError::Invalid(format!(
                "input matrix has {} rows, but the dictionary uses {need} (nwords {} + n-gram \
                 buckets)",
                self.input.rows(),
                self.dict.nwords()
            )));
        }
        if self.is_supervised() {
            if self.dict.nlabels() == 0 {
                return Err(LoadError::Invalid(
                    "supervised model without labels".to_owned(),
                ));
            }
            if self.output.rows() < self.dict.nlabels() {
                return Err(LoadError::Invalid(format!(
                    "output matrix has {} rows for {} labels",
                    self.output.rows(),
                    self.dict.nlabels()
                )));
            }
        }
        Ok(self)
    }

    /// `FastText::buildModel` / `createLoss`, for prediction.
    fn with_head(mut self) -> Result<Self, LoadError> {
        if self.is_supervised() {
            self.head = Some(match self.args.loss {
                LossKind::Softmax => Head::softmax(self.dict.nlabels()),
                LossKind::OneVsAll | LossKind::NegativeSampling => {
                    Head::sigmoid(self.dict.nlabels())
                }
                LossKind::HierarchicalSoftmax => {
                    Head::hierarchical(self.dict.label_counts()).map_err(LoadError::Invalid)?
                }
            });
        }
        Ok(self)
    }

    // ------------------------------------------------------------ introspection

    /// The training arguments stored in the header.
    pub fn args(&self) -> &Args {
        &self.args
    }

    /// The file's format version (11 or 12 for released models).
    pub fn version(&self) -> i32 {
        self.version
    }

    /// Dimension of the word / sentence vectors.
    pub fn dim(&self) -> usize {
        self.input.cols()
    }

    pub fn is_supervised(&self) -> bool {
        self.args.model == ModelKind::Supervised
    }

    /// The input matrix is product-quantized (a `.ftz` model).
    pub fn is_quantized(&self) -> bool {
        matches!(self.input, Matrix::Quant(_))
    }

    /// The output matrix is product-quantized too (`-qout`).
    pub fn is_output_quantized(&self) -> bool {
        self.qout && self.is_quantized()
    }

    pub fn nwords(&self) -> usize {
        self.dict.nwords()
    }

    pub fn nlabels(&self) -> usize {
        self.dict.nlabels()
    }

    /// Word `i`'s bytes (`0 <= i < nwords`).
    pub fn word(&self, i: usize) -> &[u8] {
        self.dict.entry(i)
    }

    /// Word `i`'s training count.
    pub fn word_count(&self, i: usize) -> i64 {
        self.dict.count(i)
    }

    /// Label `i`'s bytes (`0 <= i < nlabels`), prefix included.
    pub fn label(&self, i: usize) -> &[u8] {
        self.dict.label(i)
    }

    /// Label `i`'s training count.
    pub fn label_count(&self, i: usize) -> i64 {
        self.dict.count(self.dict.nwords() + i)
    }

    /// Number of tokens in the training data.
    pub fn ntokens(&self) -> i64 {
        self.dict.ntokens()
    }

    /// Dense input matrix as `(rows, cols, row-major data)`; `None` if quantized.
    pub fn input_matrix(&self) -> Option<(usize, usize, &[f32])> {
        self.input
            .as_dense()
            .map(|d| (self.input.rows(), self.input.cols(), d.data()))
    }

    /// Dense output matrix as `(rows, cols, row-major data)`; `None` if quantized.
    pub fn output_matrix(&self) -> Option<(usize, usize, &[f32])> {
        self.output
            .as_dense()
            .map(|d| (self.output.rows(), self.output.cols(), d.data()))
    }

    // ------------------------------------------------------------ lookups

    /// `getWordId`: the entry id of `word` (labels included), or -1.
    pub fn word_id(&self, word: &str) -> i32 {
        self.dict.id(word.as_bytes())
    }

    /// `getLabelId`: `getId(label) - nwords`, or -1. As in fastText, a
    /// *word* gives a negative id other than -1.
    pub fn label_id(&self, label: &str) -> i64 {
        let id = self.dict.id(label.as_bytes());
        if id == -1 {
            -1
        } else {
            i64::from(id) - self.dict.nwords() as i64
        }
    }

    /// `getSubwordId`: the input row a character n-gram hashes to (no
    /// pruning); `None` if the model has no n-gram buckets.
    pub fn subword_id(&self, subword: &str) -> Option<i64> {
        self.dict.subword_id(subword.as_bytes())
    }

    /// `getSubwords`: the word (if known) and its character n-grams, with
    /// their input rows (see [`Dictionary::subwords`] for pruned models).
    pub fn subwords(&self, word: &str) -> (Vec<String>, Vec<i32>) {
        let (strings, ids) = self.dict.subwords(word.as_bytes());
        // N-grams start and end on UTF-8 boundaries of a valid str, and an
        // entry equal to `word` is `word`: both are valid UTF-8.
        let strings = strings
            .into_iter()
            .map(|b| String::from_utf8_lossy(&b).into_owned())
            .collect();
        (strings, ids)
    }

    /// `getLine`: the line's word tokens (with `</s>`) and known labels.
    pub fn line<'t>(&self, text: &'t str) -> (Vec<&'t str>, Vec<&'t str>) {
        let (words, labels) = self.dict.line_tokens(text.as_bytes());
        // Tokens are split on ASCII separators, so they are valid UTF-8.
        let cvt = |v: Vec<&'t [u8]>| -> Vec<&'t str> {
            v.into_iter()
                .map(|b| std::str::from_utf8(b).unwrap_or_default())
                .collect()
        };
        (cvt(words), cvt(labels))
    }

    // ------------------------------------------------------------ vectors

    /// `x += row i` for each row id; `false` (and `x` unspecified) if an id
    /// is out of range (possible only for a word vector of a label in a
    /// pruned model).
    fn add_rows(&self, x: &mut [f32], ids: &[i32]) -> bool {
        // One vectorized pass: as unsigned, a negative id is 2^31 or more.
        let max = ids.iter().fold(0u32, |m, &id| m.max(id as u32));
        if !ids.is_empty() && (max > i32::MAX as u32 || max as usize >= self.input.rows()) {
            return false;
        }
        self.input.add_rows(x, ids);
        true
    }

    /// `hidden = mean of the rows` (`computeHidden` / `getSentenceVector`).
    fn average(&self, ids: &[i32], hidden: &mut Vec<f32>) -> bool {
        hidden.clear();
        hidden.resize(self.dim(), 0.0);
        if !self.add_rows(hidden, ids) {
            return false;
        }
        if !ids.is_empty() {
            // Vector::mul(1.0 / input.size()): the double quotient rounded to float.
            let a = (1.0f64 / ids.len() as f64) as f32;
            for h in hidden.iter_mut() {
                *h *= a;
            }
        }
        true
    }

    /// `getInputVector`: row `i` of the input matrix.
    pub fn input_vector(&self, i: usize) -> Option<Vec<f32>> {
        if i >= self.input.rows() {
            return None;
        }
        let mut x = vec![0.0; self.dim()];
        self.input.add_row(&mut x, i);
        Some(x)
    }

    /// `getWordVector`: the mean of the word's subword rows (zeros for a
    /// word without any). `None` only if a row is out of range.
    pub fn word_vector(&self, word: &str) -> Option<Vec<f32>> {
        let mut ids = Vec::new();
        self.dict.word_ids(word.as_bytes(), &mut ids);
        let mut v = Vec::new();
        self.average(&ids, &mut v).then_some(v)
    }

    /// `getSentenceVector` for one line of text.
    ///
    /// Supervised models average the rows `predict` uses (so this is the
    /// hidden layer). Unsupervised models average the L2-normalized word
    /// vectors of the whitespace-separated words.
    pub fn sentence_vector(&self, text: &str) -> Result<Vec<f32>, PredictError> {
        if text.contains('\n') {
            return Err(PredictError::Newline);
        }
        if self.is_supervised() {
            return Ok(SCRATCH.with_borrow_mut(|s| {
                self.dict.line_ids(text.as_bytes(), &mut s.line);
                let mut v = Vec::new();
                // Line rows are always in range (checked at load).
                self.average(&s.line.ids, &mut v);
                v
            }));
        }
        let dim = self.dim();
        let mut svec = vec![0.0f32; dim];
        let mut count = 0usize;
        // `iss >> word`: C-locale isspace (NUL is not a separator here).
        for word in text
            .split([' ', '\t', '\n', '\x0b', '\x0c', '\r'])
            .filter(|w| !w.is_empty())
        {
            let Some(mut vec) = self.word_vector(word) else {
                continue;
            };
            // Vector::norm: sqrt of the sum of squares, accumulated in float
            // (fused on aarch64: bit-identical to upstream fastText 0.9.2 there).
            let mut sum = 0.0f32;
            for &x in &vec {
                sum = matrix::fused(sum, x, x);
            }
            let norm = sum.sqrt();
            if norm > 0.0 {
                let a = (1.0f64 / f64::from(norm)) as f32;
                for x in &mut vec {
                    *x *= a;
                }
                for (s, x) in svec.iter_mut().zip(&vec) {
                    *s += *x;
                }
                count += 1;
            }
        }
        if count > 0 {
            let a = (1.0f64 / count as f64) as f32;
            for s in &mut svec {
                *s *= a;
            }
        }
        Ok(svec)
    }

    // ------------------------------------------------------------ prediction

    /// The rows `predict` averages for `text` (`Dictionary::getLine`).
    pub fn line_ids(&self, text: &str) -> Vec<i32> {
        let mut s = LineScratch::new();
        self.dict.line_ids(text.as_bytes(), &mut s);
        s.ids
    }

    /// `predict` for one line, as fastText's Python `predict(text, k,
    /// threshold)`: up to `k` labels (`k = -1`: all) with probability at
    /// least `threshold`, best first.
    ///
    /// Errors in fastText's order: a `'\n'` in `text`; then, if the line
    /// has any tokens, an unsupervised model or `k` of 0 or below -1.
    pub fn predict(
        &self,
        text: &str,
        k: i64,
        threshold: f32,
    ) -> Result<Vec<Prediction>, PredictError> {
        let mut out = Vec::new();
        self.predict_into(text, k, threshold, &mut out)?;
        Ok(out)
    }

    /// [`Model::predict`] into a reusable vector.
    pub fn predict_into(
        &self,
        text: &str,
        k: i64,
        threshold: f32,
        out: &mut Vec<Prediction>,
    ) -> Result<(), PredictError> {
        out.clear();
        if text.contains('\n') {
            return Err(PredictError::Newline);
        }
        SCRATCH.with_borrow_mut(|s| {
            let Scratch {
                line,
                hidden,
                output,
                heap,
            } = s;
            self.dict.line_ids(text.as_bytes(), line);
            // FastText::predict: no tokens, no predictions (and no checks).
            if line.ids.is_empty() {
                return Ok(());
            }
            let Some(head) = &self.head else {
                return Err(PredictError::NotSupervised);
            };
            // Model::predict: k == -1 means every label.
            let k = match k {
                -1 => self.output.rows(),
                k if k <= 0 => return Err(PredictError::BadK),
                k => usize::try_from(k).unwrap_or(usize::MAX),
            };
            self.average(&line.ids, hidden);
            head.predict(&self.output, hidden, k, threshold, output, heap)
                .map_err(|_| PredictError::NaN)?;
            // predictLine: std::exp of the float log-probability.
            out.extend(heap.iter().map(|&(score, label)| Prediction {
                prob: score.exp(),
                label: label as usize,
            }));
            Ok(())
        })
    }
}
