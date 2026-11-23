// Derived from python-dateutil 4.0.0 (BSD-3-Clause), © Gustavo Niemeyer, Paul Ganssle and
// dateutil contributors; modified.
//! A port of python-dateutil's date parser (`pyroid.dateutil.parser`): `parse` as in dateutil
//! 2.9.0.post0, 3.0.0 and 4.0.0 (the three are the same), and `isoparse` in its 4.0.0 and
//! 2.9.0.post0 forms.
//!
//! - [`timelex`]: the tokenizer (`_timelex`).
//! - [`info`]: a compiled `parserinfo`.
//! - [`parser`]: `parser._parse` and its helpers.
//! - [`isoparser`]: dateutil's `isoparser.isoparse`.
//! - [`cal`]: calendar arithmetic as `datetime.date` does it.
//! - [`pystr`]: the Python `str`, `int()` and `Decimal` semantics they need.
//!
//! The kernels never call Python. What needs Python objects (`datetime` construction, the tz
//! classes, `tzinfos`, error messages that come from `datetime`) is done by the caller.

pub mod cal;
pub mod info;
pub mod isoparser;
pub mod parser;
pub mod pystr;
pub mod timelex;

pub use info::{ParserInfo, Settings};
pub use parser::{ParseOptions, Parsed, Res, parse};
pub use pystr::{Fail, Int, Props, Reason};
