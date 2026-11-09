// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! The text NeMo's fastText filters give the model
//! (`nemo_curator.stages.text.filters.fasttext`); prediction is `crate::fasttext`.

use std::borrow::Cow;

use crate::filters::text::py_strip;

/// `FastTextLangId.score_document`'s line: `text.strip().replace("\n", " ")`.
pub fn lang_id_line(text: &str) -> Cow<'_, str> {
    let s = py_strip(text);
    if s.contains('\n') {
        Cow::Owned(s.replace('\n', " "))
    } else {
        Cow::Borrowed(s)
    }
}

/// `FastTextQualityFilter.score_document`'s line:
/// `text.replace("\n", " ").replace("__label__", " ")`.
pub fn quality_line(text: &str) -> Cow<'_, str> {
    let mut out = Cow::Borrowed(text);
    if out.contains('\n') {
        out = Cow::Owned(out.replace('\n', " "));
    }
    if out.contains("__label__") {
        out = Cow::Owned(out.replace("__label__", " "));
    }
    out
}

/// `label.removeprefix("__label__")`.
pub fn lang_code(label: &str) -> &str {
    label.strip_prefix("__label__").unwrap_or(label)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lines() {
        assert_eq!(lang_id_line("\u{3000} a\nb \n"), "a b");
        assert!(matches!(lang_id_line("ab"), Cow::Borrowed("ab")));
        assert_eq!(quality_line("a\n__label__x__label___"), "a  x _");
        // The replacements run in NeMo's order: a newline never forms "__label__".
        assert_eq!(quality_line("__lab\nel__"), "__lab el__");
        assert_eq!(quality_line("____label____"), "__ __");
        assert_eq!(lang_code("__label__en"), "en");
        assert_eq!(lang_code("__label__"), "");
        assert_eq!(lang_code("en"), "en");
    }
}
