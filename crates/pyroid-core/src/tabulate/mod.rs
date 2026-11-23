// Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
//! tabulate 0.10.0's table rendering, for `pyroid.tabulate`.
//!
//! The Python side (`python/pyroid/tabulate/`) keeps tabulate's input normalisation, option
//! handling and text wrapping; the bindings (`pyroid-py/src/tabulate.rs`) read the cells. This
//! crate holds the parts that run per cell:
//!
//! - [`number`]: CPython's `int(str)`/`float(str)` grammars, tabulate's `_type` for strings,
//!   the thousands-separator regex and `_afterpoint`;
//! - [`ansi`]: tabulate's ANSI escape regex (`_strip_ansi`, the invisible-codes search);
//! - [`pyfmt`]: `format(x, spec)` for floats and ints, `repr(float)`;
//! - [`pystr`]: `str.strip`, `str.splitlines`, `len` and the other `str` semantics used;
//! - [`table`]: `_align_column`, `_align_header`, `_format_table` and the built-in formats.
//!
//! Text the native code cannot decide exactly (non-ASCII text that may hold Unicode digits,
//! widths of text that is not printable ASCII, ANSI links with non-ASCII parameters) goes to
//! the interpreter through [`table::Host`] or a `None`/[`number::Tri::Ask`] answer.

pub mod ansi;
pub mod number;
pub mod pyfmt;
pub mod pystr;
pub mod table;
