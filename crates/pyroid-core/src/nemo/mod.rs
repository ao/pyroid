// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! NeMo Curator 1.4.0's CPU text stages (`pyroid.nemo`): the heuristic document filters' scores and the
//! string modifiers, with the same results as NeMo's Python code (`nemo_curator.stages.text`).
//!
//! - [`filters`]: `string.py`'s filters (ratios over characters, words and lines).
//! - [`text`]: the word splitters, sentences, paragraphs and Python `str` semantics they share;
//!   [`tables`] holds `[[:alpha:]]` as read from the `regex` module (`generate_tables.py`).

pub mod filters;
pub mod modifiers;
pub mod tables;
pub mod text;
