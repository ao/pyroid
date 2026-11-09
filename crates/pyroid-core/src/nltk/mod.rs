//! NLTK 3.10.4's sentence and word tokenizers, ported exactly.
//!
//! - [`punkt`]: `PunktSentenceTokenizer` with given parameters (the
//!   `punkt_tab` models are the user's data, loaded by the bindings).
//! - [`treebank`]: `TreebankWordTokenizer`, `NLTKWordTokenizer` (what
//!   `word_tokenize` runs on each sentence), `TreebankWordDetokenizer` and
//!   `align_tokens`.
//!
//! NLTK runs its patterns with the third-party `regex` module and, around
//! them, CPython's `str` methods; [`chars`] has both sets of character
//! classes. Results that depend on CPython's Unicode version take a
//! [`UnicodeVersion`].

mod chars;

pub use crate::ftfy::UnicodeVersion;
pub use chars::lower;
