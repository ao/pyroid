// Derived from spaCy 3.8.18 (MIT), © ExplosionAI GmbH, datatrove 0.10.1
// (Apache-2.0), © HuggingFace, and pyvi 0.1.1 (MIT), © 2016 Viet-Trung Tran;
// modified.
//! datatrove's word tokenizer for Vietnamese: `SpaCyTokenizer("vi")`, which
//! is `spacy.blank("vi")` (`VietnameseTokenizer` with pyvi) plus the
//! `sentencizer` pipe.
//!
//! spaCy's tokenizer takes `pyvi_tokenize(text)` ([`Segmenter::tokenize`])
//! and aligns the words with the text (`util.get_words_and_spaces`: a single
//! `' '` after a word becomes its trailing space, any other whitespace a
//! token of its own; a whitespace-only text is one token). The sentencizer
//! splits that `Doc`, and datatrove's `strip_strings` drops the whitespace
//! tokens, so the words are pyvi's words.

use std::ops::Range;

use super::Segmenter;
use crate::filters::{FilterError, WordTokenizer, tokenizer};
use crate::ftfy::UnicodeVersion;
use crate::text::is_py_whitespace;

/// The Vietnamese tokenizer for one Unicode version.
#[derive(Debug)]
pub struct ViTokenizer {
    version: UnicodeVersion,
}

static VI: [ViTokenizer; 5] = [
    ViTokenizer {
        version: UnicodeVersion::V14_0,
    },
    ViTokenizer {
        version: UnicodeVersion::V15_0,
    },
    ViTokenizer {
        version: UnicodeVersion::V15_1,
    },
    ViTokenizer {
        version: UnicodeVersion::V16_0,
    },
    ViTokenizer {
        version: UnicodeVersion::V17_0,
    },
];

thread_local! {
    static WORDS: std::cell::RefCell<Vec<Range<usize>>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl ViTokenizer {
    /// The shared instance for `version`.
    pub fn get(version: UnicodeVersion) -> &'static ViTokenizer {
        VI.iter()
            .find(|t| t.version == version)
            .expect("one instance per version")
    }

    /// `pyvi_tokenize(text)` as byte ranges, appended to `out`.
    pub fn pyvi_words(&self, text: &str, out: &mut Vec<Range<usize>>) {
        Segmenter::get().tokenize(text, self.version, out);
    }

    /// The `Doc`'s tokens (byte ranges), whitespace tokens included, as
    /// `get_words_and_spaces` builds them from pyvi's words.
    pub fn tokens(&self, text: &str, out: &mut Vec<(usize, usize)>) {
        if text.is_empty() {
            return;
        }
        if text.chars().all(is_py_whitespace) {
            out.push((0, text.len()));
            return;
        }
        WORDS.with_borrow_mut(|words| {
            words.clear();
            self.pyvi_words(text, words);
            // The words are slices of `text` in order, separated by
            // whitespace, and start with a non-space character, so
            // `text[text_pos:].index(word)` finds each where it is.
            let mut pos = 0usize;
            for w in words.iter() {
                if w.start > pos {
                    out.push((pos, w.start));
                }
                out.push((w.start, w.end));
                pos = w.end;
                if text.as_bytes().get(pos) == Some(&b' ') {
                    pos += 1;
                }
            }
            if pos < text.len() {
                out.push((pos, text.len()));
            }
        });
    }
}

impl WordTokenizer for ViTokenizer {
    fn unicode_version(&self) -> UnicodeVersion {
        self.version
    }

    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        // pyvi's words start and end with a non-space character, so
        // `strip_strings` keeps them as they are and drops only the
        // whitespace tokens.
        if text.chars().all(is_py_whitespace) {
            return Ok(());
        }
        WORDS.with_borrow_mut(|words| {
            words.clear();
            self.pyvi_words(text, words);
            out.extend(words.iter().map(|r| &text[r.clone()]));
        });
        Ok(())
    }

    fn sentence_ends(&self, text: &str, out: &mut Vec<usize>) -> Result<(), FilterError> {
        let mut toks = Vec::new();
        self.tokens(text, &mut toks);
        let Some(&(_, last)) = toks.last() else {
            return Ok(());
        };
        // `Sentencizer.predict`, as `SpacyTokenizer::sentence_ends_in`.
        let mut seen_period = false;
        for (i, &(s, e)) in toks.iter().enumerate() {
            let t = &text[s..e];
            let in_punct_chars = tokenizer::is_sent_punct(t);
            if seen_period
                && !in_punct_chars
                && !t
                    .chars()
                    .all(|c| crate::filters::unicode::is_punct(c, self.version))
            {
                out.push(toks[i - 1].1);
                seen_period = false;
            } else if in_punct_chars {
                seen_period = true;
            }
        }
        out.push(last);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok() -> &'static ViTokenizer {
        ViTokenizer::get(UnicodeVersion::V15_1)
    }

    fn words(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        tok().words(s, &mut out).unwrap();
        out
    }

    fn spans(s: &str) -> Vec<(usize, usize)> {
        // span_tokenize in characters: each sentence runs from its first
        // token's start to its last token's end.
        let mut toks = Vec::new();
        tok().tokens(s, &mut toks);
        let mut ends = Vec::new();
        tok().sentence_ends(s, &mut ends).unwrap();
        let c = |b: usize| s[..b].chars().count();
        let mut out = Vec::new();
        let mut k = 0;
        for e in ends {
            let start = toks[k].0;
            while toks[k].1 != e {
                k += 1;
            }
            k += 1;
            out.push((c(start), c(e)));
        }
        out
    }

    #[test]
    fn datatrove_examples() {
        // Recorded from datatrove 0.10.1 + spaCy 3.8.18 + pyvi 0.1.1.
        let t = "Hôm nay, Việt Nam  có 12.500 người. TP. Hồ Chí Minh đẹp lắm!\nMr. Smith đến từ Hà Nội.";
        assert_eq!(words(t).len(), 18);
        assert_eq!(spans(t), [(0, 35), (36, 60), (60, 85)]);
        let mut s = Vec::new();
        tok().sentences(t, &mut s).unwrap();
        assert_eq!(
            s,
            [
                "Hôm nay, Việt Nam  có 12.500 người.",
                "TP. Hồ Chí Minh đẹp lắm!",
                "Mr. Smith đến từ Hà Nội."
            ]
        );
        assert_eq!(words(""), Vec::<&str>::new());
        assert_eq!(words(" \n"), Vec::<&str>::new());
        assert_eq!(spans(" \n"), [(0, 2)]);
        assert_eq!(spans(""), []);
    }
}
