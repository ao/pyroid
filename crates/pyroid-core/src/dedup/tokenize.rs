//! The word tokenizer that splits simplified text into shingle words.
//!
//! datatrove tokenizes with spaCy's English tokenizer (`spacy.blank("en")`),
//! keeping each token's text stripped and dropping empty ones
//! (`SpaCyTokenizer.word_tokenize`). The spaCy-exact Rust tokenizer is built
//! by the `filters` track (`crate::tokenize`); until it is wired in, the
//! pipeline runs behind the [`WordTokenizer`] trait with the
//! [`WhitespaceTokenizer`] stand-in, which is **not** datatrove-identical.

/// Splits text into words, the way datatrove's `word_tokenize` does.
pub trait WordTokenizer: Sync {
    /// Append the tokens of `text` to `out`, in order. Each token is a
    /// non-empty substring of `text` with no leading or trailing whitespace.
    fn word_tokenize<'a>(&self, text: &'a str, out: &mut Vec<&'a str>);
}

/// Splits on Python whitespace (`str.split()`).
///
/// A stand-in for spaCy: on the FineWeb dev slice plus the adversarial set it
/// gives spaCy's tokens for 9,163 of 13,232 documents after `simplify_text`.
/// It never splits contractions (`dont` -> `do`, `nt`) or symbols (`£0`).
#[derive(Clone, Copy, Debug, Default)]
pub struct WhitespaceTokenizer;

impl WordTokenizer for WhitespaceTokenizer {
    fn word_tokenize<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) {
        out.extend(
            text.split(crate::text::is_py_whitespace)
                .filter(|t| !t.is_empty()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn whitespace_split() {
        let mut out = Vec::new();
        WhitespaceTokenizer.word_tokenize("  a bb\u{a0}c\n\nd ", &mut out);
        assert_eq!(out, ["a", "bb", "c", "d"]);
    }
}
