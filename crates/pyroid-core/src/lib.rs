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
//! - [`dateutil`]: python-dateutil's date parser, ported exactly (`pyroid.dateutil`).
//! - [`difflib`]: CPython's difflib, ported exactly (`pyroid.difflib`).
//! - [`html`]: an HTML parser with lxml 6.1.3's trees (a port of libxml2 2.14.6's), for
//!   `pyroid.trafilatura`.
//! - [`metrics`]: ROUGE, BLEU and chrF, exact ports of rouge-score and sacrebleu (`pyroid.metrics`).
//! - [`nemo`]: NeMo Curator 1.4.0's CPU text heuristics and string modifiers (`pyroid.nemo`).
//! - [`nltk`]: NLTK 3.10.4's Punkt sentence splitter and Treebank word tokenizers (`pyroid.nltk`).
//! - [`trafilatura`]: Trafilatura 2.3.1's and 1.11.0's text extraction, ported exactly
//!   (`pyroid.trafilatura`).
//! - [`sqlparse`]: sqlparse 0.6.1's lexer, grouping and formatter, ported exactly
//!   (`pyroid.sqlparse`).
//! - [`sudachi`]: Japanese morphemes as SudachiPy 0.7.0 gives them, from an
//!   installed SudachiDict (datatrove's Japanese tokenizer in the filters).
//! - [`warc`]: warcio 1.8.2's WARC/ARC reader, ported exactly (`pyroid.warc`).
//! - [`tabulate`]: tabulate 0.10.0's table rendering (`pyroid.tabulate`).
//! - [`pyvi`]: Vietnamese words as spaCy's tokenizer gets them from pyvi 0.1.1
//!   (datatrove's Vietnamese tokenizer in the filters).
//! - [`pythainlp`]: Thai words and sentences as PyThaiNLP 5.4.0 gives them,
//!   from an installed PyThaiNLP's word list and sentence model (datatrove's
//!   Thai tokenizer in the filters).
//! - [`crfsuite`]: CRFsuite model files and Viterbi tagging, for pyvi and
//!   PyThaiNLP's sentence model.
//! - [`yaml`]: PyYAML 6.0.3's safe loader and dumper, ported exactly (`pyroid.yaml`).
//! - [`jmespath`]: jmespath 1.1.0's lexer and parser (`pyroid.jmespath`).
//! - [`packaging`]: packaging 26.3's versions, specifiers, markers and requirements
//!   (`pyroid.packaging`).
//! - [`langdetect`]: langdetect 1.0.9's language detection, ported exactly
//!   (`pyroid.langdetect`).
//! - [`version`]: the PEP 440 form of the Cargo version (`pyroid.__version__`).
//!
//! Each module adds a `src/<mod>/` directory here; see `docs/dev/MODULES.md`.
#![forbid(unsafe_code)]

pub mod columnar;
pub mod crfsuite;
pub mod datasketch;
pub mod dateutil;
pub mod dedup;
pub mod difflib;
pub mod fasttext;
pub mod filters;
pub mod ftfy;
pub mod html;
pub mod jieba;
pub mod jmespath;
pub mod json_repair;
pub mod langdetect;
pub mod metrics;
pub mod nemo;
pub mod nltk;
pub mod packaging;
pub mod par;
pub mod pythainlp;
pub mod pyvi;
pub mod scratch;
pub mod sqlparse;
pub mod sudachi;
pub mod tabulate;
pub mod text;
pub mod trafilatura;
pub mod translit;
pub mod version;
pub mod warc;
pub mod yaml;
