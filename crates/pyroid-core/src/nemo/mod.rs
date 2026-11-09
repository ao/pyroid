// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! NeMo Curator 1.4.0's CPU text stages (`pyroid.nemo`): the heuristic document filters' scores and the
//! string modifiers, with the same results as NeMo's Python code (`nemo_curator.stages.text`).
//!
//! - [`filters`]: `string.py`'s filters (ratios over characters, words and lines).
//! - [`repetition`]: the repeated lines / paragraphs and n-gram filters.
//! - [`fasttext`]: the lines NeMo's fastText filters predict on.
//! - [`text`]: the word splitters, sentences, paragraphs and Python `str` semantics they share;
//!   [`tables`] holds `[[:alpha:]]` as read from the `regex` module (`generate_tables.py`).

pub mod fasttext;
pub mod filters;
pub mod modifiers;
pub mod repetition;
pub mod tables;
pub mod text;
