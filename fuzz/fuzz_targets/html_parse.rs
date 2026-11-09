//! The HTML parser (`pyroid_core::html`): `lxml::fromstring_bytes`, `fromstring_wtf8` and
//! `fromstring_str` on the raw input bytes.
//!
//! Input: the document bytes, as they are (any bytes; `fromstring_wtf8` and `fromstring_str`
//! run when the input is surrogate-tolerant UTF-8 or UTF-8).
//!
//! Checked (the list and the code are in `crates/pyroid-core/tests/html_data/fuzz_check.rs`,
//! which `tests/html_robust.rs` also runs under a seeded random loop):
//! - no panic on any input, and the same result twice;
//! - `fromstring_str` equals `fromstring_wtf8` on UTF-8, and `fromstring_wtf8` reports the
//!   offset `ftfy::validate_wtf8` reports for other bytes;
//! - every tree is well formed: links agree, `nchild` counts children, depth at most 256,
//!   lowercase names, no empty text nodes, and at most three bytes of text per input byte.
//!
//! Exactness against lxml is checked by the differential runs in `crates/pyroid-core/tests/`
//! (`html_lxml.rs`, `html_large.rs`, `html_fromstring.rs`), not here.

#![no_main]

use libfuzzer_sys::fuzz_target;

include!("../../crates/pyroid-core/tests/html_data/fuzz_check.rs");

fuzz_target!(|data: &[u8]| check(data));
