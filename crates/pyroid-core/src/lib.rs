//! Pure-Rust kernels behind pyroid.
//!
//! This crate has no Python dependency: everything here is plain, safe Rust
//! that is unit-tested with `cargo test`. The `pyroid-py` crate wraps it for
//! Python (argument conversion, detaching from the interpreter, Arrow FFI).
//!
//! - [`par`]: the one shared, lazily built, fork-aware thread pool and the
//!   order-preserving batch maps every kernel runs on.
//! - [`columnar`]: builders for Arrow-layout buffers (utf8, large utf8,
//!   boolean) so the bindings can export results without an Arrow dependency.
//! - [`text`]: text kernels (`pyroid.text`, the demonstration module).
//! - [`dedup`]: MinHash and exact deduplication (`pyroid.dedup`).
//! - [`html`]: an HTML parser with lxml 6.1.3's trees (a port of libxml2 2.14.6's), for
//!   `pyroid.trafilatura`.
//! - [`metrics`]: ROUGE, BLEU and chrF, exact ports of rouge-score and sacrebleu (`pyroid.metrics`).
//! - [`nltk`]: NLTK 3.10.4's Punkt sentence splitter and Treebank word tokenizers (`pyroid.nltk`).
//! - [`trafilatura`]: Trafilatura 2.3.1's and 1.11.0's text extraction, ported exactly
//!   (`pyroid.trafilatura`).
//! - [`sudachi`]: Japanese morphemes as SudachiPy 0.7.0 gives them, from an
//!   installed SudachiDict (datatrove's Japanese tokenizer in the filters).
//! - [`version`]: the PEP 440 form of the Cargo version (`pyroid.__version__`).
//!
//! Each module adds a `src/<mod>/` directory here; see `docs/dev/MODULES.md`.
#![forbid(unsafe_code)]

pub mod columnar;
pub mod datasketch;
pub mod dedup;
pub mod fasttext;
pub mod filters;
pub mod ftfy;
pub mod html;
pub mod jieba;
pub mod json_repair;
pub mod metrics;
pub mod nltk;
pub mod par;
pub mod sudachi;
pub mod text;
pub mod trafilatura;
pub mod translit;
pub mod version;
