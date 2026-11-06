//! The word tokenizer that splits simplified text into shingle words.
//!
//! datatrove tokenizes with spaCy's English tokenizer (`spacy.blank("en")`),
//! keeping each token's text stripped and dropping empty ones
//! (`SpaCyTokenizer.word_tokenize`). [`SpacyTokenizer`] is pyroid's port of that
//! tokenizer (`crate::filters::tokenizer`, shared with the quality filters);
//! its non-whitespace tokens are exactly datatrove's words.

pub use crate::filters::FilterError as TokenizeError;
pub use crate::filters::tokenizer::SpacyTokenizer;

/// Splits text into words, the way datatrove's `word_tokenize` does.
pub trait WordTokenizer: Sync {
    /// Append the tokens of `text` to `out`, in order. Each token is a
    /// non-empty substring of `text` with no leading or trailing whitespace.
    ///
    /// Fails only where datatrove's tokenizer raises (spaCy's `E025` for a
    /// text of 2**30 characters or more).
    fn word_tokenize<'a>(&self, text: &'a str, out: &mut Vec<&'a str>)
    -> Result<(), TokenizeError>;
}

impl WordTokenizer for SpacyTokenizer {
    fn word_tokenize<'a>(
        &self,
        text: &'a str,
        out: &mut Vec<&'a str>,
    ) -> Result<(), TokenizeError> {
        crate::filters::WordTokenizer::words(self, text, out)
    }
}

/// Any of datatrove's word tokenizers pyroid ports
/// ([`crate::filters::tokenizer_for`]).
impl WordTokenizer for dyn crate::filters::WordTokenizer {
    fn word_tokenize<'a>(
        &self,
        text: &'a str,
        out: &mut Vec<&'a str>,
    ) -> Result<(), TokenizeError> {
        self.words(text, out)
    }
}

/// Splits on Python whitespace (`str.split()`). **Not** datatrove's tokenizer.
///
/// After `simplify_text` it gives spaCy's tokens for 9,163 of the 13,232
/// FineWeb dev + adversarial documents: it never splits contractions
/// (`dont` -> `do`, `nt`) or symbols (`£0` -> `£`, `0`).
#[derive(Clone, Copy, Debug, Default)]
pub struct WhitespaceTokenizer;

impl WordTokenizer for WhitespaceTokenizer {
    fn word_tokenize<'a>(
        &self,
        text: &'a str,
        out: &mut Vec<&'a str>,
    ) -> Result<(), TokenizeError> {
        out.extend(
            text.split(crate::text::is_py_whitespace)
                .filter(|t| !t.is_empty()),
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dedup::UnicodeVersion;

    #[test]
    fn whitespace_split() {
        let mut out = Vec::new();
        WhitespaceTokenizer
            .word_tokenize("  a bb\u{a0}c\n\nd ", &mut out)
            .unwrap();
        assert_eq!(out, ["a", "bb", "c", "d"]);
    }

    #[test]
    fn spacy_splits_what_whitespace_does_not() {
        let spacy = SpacyTokenizer::english(UnicodeVersion::V15_1);
        let mut out = Vec::new();
        spacy
            .word_tokenize("i dont pay £0 for it  ok", &mut out)
            .unwrap();
        assert_eq!(out, ["i", "do", "nt", "pay", "£", "0", "for", "it", "ok"]);
    }
}
