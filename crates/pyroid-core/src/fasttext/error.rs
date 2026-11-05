// Part of pyroid's port of fastText (https://github.com/facebookresearch/fastText).
// fastText: Copyright (c) 2016-present, Facebook, Inc. MIT License.

//! Errors from loading a model and from inference.

use std::fmt;
use std::io;

/// Why a model file could not be loaded.
#[derive(Debug)]
pub enum LoadError {
    /// The file could not be opened or read.
    Io(io::Error),
    /// Not a fastText model: wrong magic number, or a format version newer
    /// than this reader (fastText: "has wrong file format!").
    WrongFormat { magic: i32, version: Option<i32> },
    /// The file ends inside a section.
    Truncated {
        what: &'static str,
        offset: u64,
        need: u64,
        have: u64,
    },
    /// The file is structurally invalid: inconsistent sizes, unknown enum
    /// values, ids outside the matrices. fastText would read out of bounds
    /// or crash on these; pyroid refuses them.
    Invalid(String),
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::WrongFormat {
                magic,
                version: None,
            } => write!(
                f,
                "has wrong file format! (not a fastText model: magic number {magic:#010x}, \
                 expected {:#010x})",
                super::MAGIC
            ),
            Self::WrongFormat {
                version: Some(v), ..
            } => write!(
                f,
                "has wrong file format! (fastText format version {v} is newer than the \
                 newest supported, {})",
                super::VERSION
            ),
            Self::Truncated {
                what,
                offset,
                need,
                have,
            } => write!(
                f,
                "is truncated: {what} at byte {offset} needs {need} more bytes, the file has {have}"
            ),
            Self::Invalid(msg) => write!(f, "is not a valid fastText model: {msg}"),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

/// Why a prediction (or sentence vector) failed. The messages are
/// fastText's own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictError {
    /// The text contains `'\n'` (`predict` handles one line at a time).
    Newline,
    /// `k` is 0 or below -1.
    BadK,
    /// The model is unsupervised (cbow / skipgram): it has no labels.
    NotSupervised,
    /// A dense dot product produced NaN (the model holds NaN or infinite weights).
    NaN,
}

impl PredictError {
    /// fastText's message for this error.
    pub fn message(self) -> &'static str {
        match self {
            Self::Newline => "predict processes one line at a time (remove '\\n')",
            Self::BadK => "k needs to be 1 or higher!",
            Self::NotSupervised => "Model needs to be supervised for prediction!",
            Self::NaN => "Encountered NaN.",
        }
    }
}

impl fmt::Display for PredictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for PredictError {}
