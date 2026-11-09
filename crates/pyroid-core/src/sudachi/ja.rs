// Derived from spaCy 3.8.18 (MIT), © ExplosionAI GmbH, and datatrove 0.10.1
// (Apache-2.0), © HuggingFace; modified.
//! datatrove's word tokenizer for Japanese: `SpaCyTokenizer("ja")`.
//!
//! That is `spacy.blank("ja")` with datatrove's copy of spaCy's
//! `JapaneseTokenizer` (`datatrove.ja.JapaneseTokenizer`: SudachiPy in split
//! mode A with `sudachidict_core`) plus the `sentencizer` pipe:
//!
//! - `SpaCyTokenizer._do_tokenize` cuts the text with
//!   `chunk_text_on_bytes(text, 40000)` and tokenizes each chunk on its own.
//! - `_get_dtokens` drops the morphemes with an empty surface;
//!   `get_dtokens_and_spaces` aligns the rest with the chunk: whitespace
//!   morphemes become gap tokens, except a lone `" "` morpheme after a word,
//!   which is that word's trailing space.
//! - The sentencizer splits each chunk's `Doc`.
//! - `word_tokenize` is every token's text, stripped, the empty ones dropped
//!   (datatrove's `strip_strings`).
//! - `span_tokenize` offsets the sentences of each chunk after the first by
//!   the end of the previous sentence, not by the chunk's start (datatrove
//!   does this, so a chunk whose text ends in a trailing space shifts the
//!   later spans by one character).
//!
//! The dictionary is the one [`install`]ed in this process.

use std::ops::Range;
use std::sync::{Mutex, OnceLock};

use super::{DicError, Dictionary, Morpheme};
use crate::filters::{FilterError, WordTokenizer, tokenizer};
use crate::ftfy::UnicodeVersion;
use crate::text::is_py_whitespace;

static DICT: OnceLock<Dictionary> = OnceLock::new();
static LOADING: Mutex<()> = Mutex::new(());

/// Load the system dictionary at `path` for the Japanese tokenizer, once per
/// process. Later calls return the dictionary already loaded, whatever their
/// path.
pub fn install(path: impl AsRef<std::path::Path>) -> Result<&'static Dictionary, DicError> {
    if let Some(d) = DICT.get() {
        return Ok(d);
    }
    let _guard = LOADING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(d) = DICT.get() {
        return Ok(d);
    }
    let d = Dictionary::open(path)?;
    Ok(DICT.get_or_init(|| d))
}

/// The dictionary [`install`] loaded, if any.
pub fn installed() -> Option<&'static Dictionary> {
    DICT.get()
}

/// The Japanese tokenizer for one Unicode version (only spaCy's `is_punct`,
/// in the sentencizer, depends on it).
#[derive(Debug)]
pub struct JaTokenizer {
    version: UnicodeVersion,
}

static JA: [JaTokenizer; 5] = [
    JaTokenizer {
        version: UnicodeVersion::V14_0,
    },
    JaTokenizer {
        version: UnicodeVersion::V15_0,
    },
    JaTokenizer {
        version: UnicodeVersion::V15_1,
    },
    JaTokenizer {
        version: UnicodeVersion::V16_0,
    },
    JaTokenizer {
        version: UnicodeVersion::V17_0,
    },
];

/// `chunk_text_on_bytes`'s `max_chunk_size` in `SpaCyTokenizer._do_tokenize`.
const CHUNK: usize = 40_000;

/// `str.isspace()` of a non-empty string.
fn is_space_word(w: &str) -> bool {
    !w.is_empty() && w.chars().all(is_py_whitespace)
}

/// Python's `str.strip()`.
fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_whitespace)
}

/// datatrove's `chunk_text_on_bytes(text, max_chunk_size)`, as byte ranges.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::single_range_in_vec_init
)]
pub fn chunk_text_on_bytes(text: &str, max_chunk_size: usize) -> Vec<Range<usize>> {
    if text.is_empty() {
        return Vec::new();
    }
    if text.len() <= max_chunk_size {
        // `part` only grows until it is the whole text.
        return vec![0..text.len()];
    }
    // Byte offset of every character boundary.
    let bounds: Vec<usize> = text
        .char_indices()
        .map(|(i, _)| i)
        .chain(std::iter::once(text.len()))
        .collect();
    let n_chars = bounds.len() - 1;
    let factor = n_chars as f64 / text.len() as f64;
    let max = max_chunk_size as f64;
    let increase_by = (max * 0.1).clamp(1.0, 10.0) as usize;
    let shrink_by = (max * 0.001).min(10.0) as usize;
    let guess = max * factor - 10.0;
    let mut initial = if guess < 1.0 { 1 } else { guess as usize };
    let mut out = Vec::new();
    let mut pos = 0usize; // char index of `remaining`
    while pos < n_chars {
        let left = n_chars - pos;
        let bytes = |k: usize| bounds[pos + k.min(left)] - bounds[pos];
        if bytes(initial) > max_chunk_size {
            initial = initial.saturating_sub(shrink_by).max(1);
            continue;
        }
        let mut cut_after = initial;
        let mut part = initial.min(left);
        while bytes(part) < max_chunk_size && part != left {
            cut_after = left.min(cut_after + increase_by);
            part = cut_after;
        }
        if bytes(part) > max_chunk_size {
            cut_after -= increase_by;
        }
        let take = cut_after.min(left);
        out.push(bounds[pos]..bounds[pos + take]);
        pos += take;
    }
    out
}

/// One token of a chunk's `Doc`: a byte range of the chunk.
#[derive(Clone, Copy, Debug)]
struct Tok {
    start: usize,
    end: usize,
}

impl JaTokenizer {
    /// The shared instance for `version`, if a dictionary is [`install`]ed.
    pub fn get(version: UnicodeVersion) -> Option<&'static JaTokenizer> {
        installed()?;
        JA.iter().find(|t| t.version == version)
    }

    fn dict() -> Result<&'static Dictionary, FilterError> {
        installed().ok_or(FilterError::Tokenizer(super::TokenizeError::Internal))
    }

    /// The non-empty morpheme surfaces of `chunk`, as byte ranges.
    fn surfaces(chunk: &str, out: &mut Vec<Range<usize>>) -> Result<(), FilterError> {
        let mut ms: Vec<Morpheme> = Vec::new();
        Self::dict()?
            .tokenize(chunk, &mut ms)
            .map_err(FilterError::Tokenizer)?;
        out.extend(
            ms.iter()
                .filter(|m| m.end > m.begin)
                .map(|m| m.begin..m.end),
        );
        Ok(())
    }

    /// `get_dtokens_and_spaces`: the chunk's `Doc` tokens.
    fn doc_tokens(chunk: &str, surfaces: &[Range<usize>], out: &mut Vec<Tok>) {
        if surfaces.is_empty() {
            return;
        }
        let space = |r: &Range<usize>| is_space_word(&chunk[r.clone()]);
        if surfaces.iter().all(space) {
            out.push(Tok {
                start: 0,
                end: chunk.len(),
            });
            return;
        }
        let mut pos = 0usize;
        for (i, r) in surfaces.iter().enumerate() {
            if space(r) {
                continue;
            }
            // Surfaces partition the chunk and the skipped ones are
            // whitespace, so `text[text_pos:].index(word)` finds the word
            // where it is.
            debug_assert_eq!(chunk[pos..].find(&chunk[r.clone()]), Some(r.start - pos));
            if r.start > pos {
                out.push(Tok {
                    start: pos,
                    end: r.start,
                });
            }
            out.push(Tok {
                start: r.start,
                end: r.end,
            });
            pos = r.end;
            if surfaces
                .get(i + 1)
                .is_some_and(|n| &chunk[n.clone()] == " ")
            {
                pos += 1;
            }
        }
        if pos < chunk.len() {
            out.push(Tok {
                start: pos,
                end: chunk.len(),
            });
        }
    }

    /// `Sentencizer.predict` over `toks`: each sentence's `(start, end)`
    /// byte range in the chunk.
    fn sentences_in(&self, chunk: &str, toks: &[Tok], out: &mut Vec<(usize, usize)>) {
        let Some(last) = toks.last() else {
            return;
        };
        let mut start = 0usize;
        let mut seen_period = false;
        for (i, t) in toks.iter().enumerate() {
            let s = &chunk[t.start..t.end];
            let in_punct_chars = tokenizer::is_sent_punct(s);
            if seen_period
                && !in_punct_chars
                && !s
                    .chars()
                    .all(|c| crate::filters::unicode::is_punct(c, self.version))
            {
                out.push((toks[start].start, toks[i - 1].end));
                start = i;
                seen_period = false;
            } else if in_punct_chars {
                seen_period = true;
            }
        }
        out.push((toks[start].start, last.end));
    }

    /// Each chunk's sentences (byte ranges in the chunk), in order.
    fn chunk_sentences(
        &self,
        text: &str,
        mut f: impl FnMut(Range<usize>, &[(usize, usize)]),
    ) -> Result<(), FilterError> {
        let (mut surfaces, mut toks, mut sents) = (Vec::new(), Vec::new(), Vec::new());
        for chunk in chunk_text_on_bytes(text, CHUNK) {
            let c = &text[chunk.clone()];
            surfaces.clear();
            toks.clear();
            sents.clear();
            Self::surfaces(c, &mut surfaces)?;
            Self::doc_tokens(c, &surfaces, &mut toks);
            self.sentences_in(c, &toks, &mut sents);
            f(chunk, &sents);
        }
        Ok(())
    }

    /// datatrove's `span_tokenize(text)`: `(start, end)` character offsets,
    /// appended to `out`. A chunk's spans are offset by the previous span's
    /// end, not by the chunk's start, as datatrove does.
    pub fn spans(&self, text: &str, out: &mut Vec<(usize, usize)>) -> Result<(), FilterError> {
        let first = out.len();
        self.chunk_sentences(text, |chunk, sents| {
            let start = if out.len() > first {
                out[out.len() - 1].1
            } else {
                0
            };
            let c = &text[chunk];
            // Byte offsets of the chunk to its characters, in one pass.
            let (mut at_b, mut at_c) = (0usize, 0usize);
            let mut chars = |b: usize| {
                at_c += n_chars(&c[at_b..b]);
                at_b = b;
                at_c
            };
            for &(s, e) in sents {
                let s = chars(s);
                let e = chars(e);
                out.push((start + s, start + e));
            }
        })
    }
}

/// Characters in `s`.
fn n_chars(s: &str) -> usize {
    s.chars().count()
}

impl WordTokenizer for JaTokenizer {
    fn unicode_version(&self) -> UnicodeVersion {
        self.version
    }

    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        let mut surfaces = Vec::new();
        for chunk in chunk_text_on_bytes(text, CHUNK) {
            surfaces.clear();
            Self::surfaces(&text[chunk.clone()], &mut surfaces)?;
            for r in &surfaces {
                let w = &text[chunk.start + r.start..chunk.start + r.end];
                if !is_space_word(w) {
                    out.push(py_strip(w));
                }
            }
        }
        Ok(())
    }

    fn sentence_ends(&self, text: &str, out: &mut Vec<usize>) -> Result<(), FilterError> {
        let mut chunks = 0usize;
        let mut ends = Vec::new();
        self.chunk_sentences(text, |chunk, sents| {
            chunks += 1;
            ends.extend(sents.iter().map(|s| chunk.start + s.1));
        })?;
        if chunks <= 1 {
            // One chunk: the spans are the chunk's own.
            out.extend(ends);
            return Ok(());
        }
        let mut spans = Vec::new();
        self.spans(text, &mut spans)?;
        // `span_tokenize`'s ends are characters of `text`: back to bytes.
        let mut bounds = text
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(text.len()));
        let mut at = 0usize; // characters consumed from `bounds`
        let mut byte = 0usize;
        for (_, e) in spans {
            for b in bounds.by_ref().take(e + 1 - at) {
                byte = b;
            }
            at = e + 1;
            out.push(byte);
        }
        Ok(())
    }

    fn sentences<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        // `sent_tokenize`: each sentence's own text (chunk-relative), stripped.
        self.chunk_sentences(text, |chunk, sents| {
            for &(s, e) in sents {
                let t = py_strip(&text[chunk.start + s..chunk.start + e]);
                if !t.is_empty() {
                    out.push(t);
                }
            }
        })
    }
}

#[cfg(test)]
#[allow(clippy::single_range_in_vec_init)]
mod tests {
    use super::*;

    #[test]
    fn chunking_matches_datatrove_on_simple_inputs() {
        assert_eq!(chunk_text_on_bytes("", 40_000).len(), 0);
        assert_eq!(chunk_text_on_bytes("abc", 40_000), [0..3]);
        // Recorded from datatrove 0.10.1: lengths in characters.
        let lens = |t: &str, n: usize| -> Vec<usize> {
            chunk_text_on_bytes(t, n)
                .into_iter()
                .map(|r| n_chars(&t[r]))
                .collect()
        };
        assert_eq!(lens(&"a".repeat(100_000), 40_000), [40_000, 40_000, 20_000]);
        assert_eq!(lens(&"あ".repeat(30_000), 40_000), [13_333, 13_333, 3_334]);
    }
}
