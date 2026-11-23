// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! sqlparse 0.6.1, ported exactly: the lexer ([`lexer`], with the `re`-compatible matcher in
//! [`regex`]), the statement splitter ([`splitter`]), the token tree and its grouping
//! ([`tree`], [`grouping`]) and the formatting filters ([`filters`], [`reindent`]), driven by
//! [`engine`]. `pyroid.sqlparse` builds sqlparse's own Python token objects from [`tree`].

// A line-by-line port keeps sqlparse's names and shapes: its short loop variables, its
// boolean state fields and options, and its long methods.
#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::fn_params_excessive_bools
)]

pub mod engine;
pub mod filters;
pub mod grouping;
pub mod lexer;
pub mod pystr;
pub mod regex;
pub mod reindent;
pub mod splitter;
mod tables;
pub mod tree;
pub mod ttype;
