// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! datatrove's `simplify_text` (`datatrove/utils/text.py`), as a given CPython
//! version computes it.
//!
//! PLACEHOLDER: being ported; returns the text unchanged.

use std::borrow::Cow;

use super::config::{TextNormConfig, UnicodeVersion};

/// datatrove's `simplify_text(text, config)`.
///
/// Returns `Cow::Borrowed(text)` (the whole input) when nothing changes.
pub fn simplify_text<'a>(
    text: &'a str,
    config: &TextNormConfig,
    unicode: UnicodeVersion,
) -> Cow<'a, str> {
    let _ = (config, unicode);
    Cow::Borrowed(text)
}
