// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! packaging 26.3's versions, specifiers, markers and requirements (`pyroid.packaging`).
//!
//! - [`version`]: PEP 440 parsing, ordering, hashing and printing.
//! - [`specifiers`]: specifier parsing and the range engine behind `contains` and `filter`.
//! - [`markers`]: the PEP 508 tokenizer and parser (markers and requirements).
//! - [`num`]: the arbitrary-size integers PEP 440 numbers are.

pub mod markers;
pub mod num;
pub mod specifiers;
pub mod version;

#[cfg(test)]
mod tests;
