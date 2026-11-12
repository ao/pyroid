// Derived from spaCy 3.8.18 (MIT), © ExplosionAI GmbH, datatrove 0.10.1
// (Apache-2.0), © HuggingFace, and jieba 0.42.1 (MIT), © 2013 Sun Junyi;
// modified.
//! datatrove's word tokenizer for Chinese: `SpaCyTokenizer("zh")`, which is
//! `spacy.blank("zh", config={"nlp": {"tokenizer": {"segmenter": "jieba"}}})`
//! plus the `sentencizer` pipe.
//!
//! spaCy's `ChineseTokenizer` takes `jieba.cut(text, cut_all=False)` (with
//! jieba's default dictionary and HMM), aligns the words with the text
//! (`util.get_words_and_spaces`: whitespace words are dropped, a single
//! `' '` after a word becomes its trailing space, and any other whitespace
//! becomes a token of its own), and the sentencizer splits that `Doc`.
//! datatrove's `strip_strings` then drops the whitespace tokens, so the
//! words are jieba's words without the whitespace ones.
//!
//! This always uses jieba's default dictionary, unmodified: words added with
//! `pyroid.jieba` do not change it.

use super::{Dictionary, ForceSplit};
use crate::filters::{FilterError, WordTokenizer, tokenizer};
use crate::ftfy::UnicodeVersion;
use crate::text::is_py_whitespace;

/// The Chinese tokenizer for one Unicode version (only spaCy's `is_punct`,
/// in the sentencizer, depends on it).
#[derive(Debug)]
pub struct ZhTokenizer {
    version: UnicodeVersion,
}

static ZH: [ZhTokenizer; 5] = [
    ZhTokenizer {
        version: UnicodeVersion::V14_0,
    },
    ZhTokenizer {
        version: UnicodeVersion::V15_0,
    },
    ZhTokenizer {
        version: UnicodeVersion::V15_1,
    },
    ZhTokenizer {
        version: UnicodeVersion::V16_0,
    },
    ZhTokenizer {
        version: UnicodeVersion::V17_0,
    },
];

static NO_FORCE_SPLIT: std::sync::LazyLock<ForceSplit> =
    std::sync::LazyLock::new(ForceSplit::default);

/// `word.isspace()`.
fn is_space_word(w: &str) -> bool {
    !w.is_empty() && w.chars().all(is_py_whitespace)
}

impl ZhTokenizer {
    /// The shared instance for `version`.
    pub fn get(version: UnicodeVersion) -> &'static ZhTokenizer {
        ZH.iter()
            .find(|t| t.version == version)
            .expect("one instance per version")
    }

    /// `jieba.cut(text)` with the default dictionary.
    fn jieba<'a>(text: &'a str, out: &mut Vec<&'a str>) {
        Dictionary::default_ref()
            .cut(text, true, &NO_FORCE_SPLIT, out)
            .expect("the default dictionary has a positive total and frequencies");
    }

    /// The `Doc`'s tokens (byte ranges), whitespace tokens included, as
    /// `get_words_and_spaces` builds them from jieba's words.
    pub fn tokens(&self, text: &str, out: &mut Vec<(usize, usize)>) {
        let mut words = Vec::new();
        Self::jieba(text, &mut words);
        let (mut pos, mut off) = (0usize, 0usize);
        for w in words {
            let (ws, we) = (off, off + w.len());
            off = we;
            if is_space_word(w) {
                continue;
            }
            if ws > pos {
                out.push((pos, ws));
            }
            out.push((ws, we));
            pos = we;
            if text.as_bytes().get(pos) == Some(&b' ') {
                pos += 1;
            }
        }
        if pos < text.len() {
            out.push((pos, text.len()));
        }
    }
}

impl WordTokenizer for ZhTokenizer {
    fn unicode_version(&self) -> UnicodeVersion {
        self.version
    }

    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        let start = out.len();
        Self::jieba(text, out);
        let mut keep = start;
        for i in start..out.len() {
            if !is_space_word(out[i]) {
                out[keep] = out[i];
                keep += 1;
            }
        }
        out.truncate(keep);
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

    fn words(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        ZhTokenizer::get(UnicodeVersion::V15_1)
            .words(s, &mut out)
            .unwrap();
        out
    }

    fn ends(s: &str) -> Vec<usize> {
        let mut out = Vec::new();
        ZhTokenizer::get(UnicodeVersion::V15_1)
            .sentence_ends(s, &mut out)
            .unwrap();
        out
    }

    /// Byte offset of the character offset `c` in `s`.
    fn b(s: &str, c: usize) -> usize {
        s.char_indices().nth(c).map_or(s.len(), |(i, _)| i)
    }

    #[test]
    fn datatrove_examples() {
        // Recorded from datatrove 0.10.1 + spaCy 3.8.18 + jieba 0.42.1:
        // word_tokenize / span_tokenize (character offsets).
        assert_eq!(words("IS_ALPHA 中文"), ["IS", "_", "ALPHA", "中文"]);
        let s = "你好。\n再见！  好的";
        assert_eq!(words(s), ["你好", "。", "再见", "！", "好", "的"]);
        assert_eq!(ends(s), [b(s, 3), b(s, 7), b(s, 11)]);
        let s = "  前导空格\r\n后面";
        assert_eq!(words(s), ["前导", "空格", "后面"]);
        assert_eq!(ends(s), [b(s, 10)]);
        assert_eq!(words("中  文 abc  def。"), ["中", "文", "abc", "def", "。"]);
        assert_eq!(words("").len(), 0);
        assert_eq!(ends("").len(), 0);
    }
}
