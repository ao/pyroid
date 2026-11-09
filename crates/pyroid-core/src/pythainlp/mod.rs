// Derived from PyThaiNLP 5.4.0 (Apache-2.0), © 2016-2026 PyThaiNLP Project,
// and datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! The Thai word and sentence tokenizers of PyThaiNLP 5.4.0 that datatrove
//! 0.10.1's `ThaiTokenizer` uses, with the same results:
//!
//! - [`tcc()`]: `tokenize.tcc_p` (the Thai Character Clusters newmm uses).
//! - [`Thai::newmm`]: `tokenize.newmm.segment`, with and without safe mode.
//! - [`Thai::word_tokenize`]: `word_tokenize(text)` with its defaults
//!   (newmm, `rejoin_formatted_num`, whitespace kept): the tokens crfcut
//!   reads.
//! - [`Thai::crfcut_labels`] and [`Thai::crfcut`]: `tokenize.crfcut` (its
//!   features, the CRF tagger of [`crate::crfsuite`], and `segment`).
//! - [`th::ThTokenizer`]: datatrove's `ThaiTokenizer` (`word_tokenize`,
//!   `sent_tokenize`, `span_tokenize`) as a [`crate::filters::WordTokenizer`].
//!
//! The word list (`corpus/words_th.txt`) and the sentence model
//! (`corpus/sentenceseg_crfcut.model`) are not bundled: [`Thai::open`] reads
//! them at run time from the corpus directory of the user's installed
//! PyThaiNLP, and [`install`] does so once per process for the tokenizer.
//!
//! Positions and lengths are code points, as in Python. Only `re`'s `\d`
//! (in newmm's non-Thai pattern and in `rejoin_formatted_num`) depends on
//! the Unicode version; `str.strip()` and `str.splitlines()` are the same in
//! every supported version.

use std::cell::RefCell;
use std::fmt;
use std::path::{Path, PathBuf};

use crate::crfsuite::{Model, ModelError};
use crate::ftfy::UnicodeVersion;

mod crfcut;
mod newmm;
mod tcc;
pub mod th;
mod trie;

#[cfg(test)]
mod replay;

pub use th::{install, installed};

/// `get_corpus`'s file name of the word list.
pub const WORDS_FILE: &str = "words_th.txt";
/// `crfcut`'s model file name.
pub const MODEL_FILE: &str = "sentenceseg_crfcut.model";

/// Why the corpus files could not be loaded.
#[derive(Debug)]
pub enum ThaiError {
    /// A file could not be read.
    Io(PathBuf, std::io::Error),
    /// The word list is not UTF-8, or the model is not a CRFsuite model.
    Format(String),
}

impl fmt::Display for ThaiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ThaiError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            ThaiError::Format(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ThaiError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ThaiError::Io(_, e) => Some(e),
            ThaiError::Format(_) => None,
        }
    }
}

/// PyThaiNLP's default word trie (`word_dict_trie()`) and crfcut's model.
#[derive(Debug)]
pub struct Thai {
    trie: trie::Trie,
    model: Model,
    attrs: crfcut::Attrs,
}

/// Python's `str.splitlines()` boundaries: `\n`, `\r`, `\r\n`, `\v`, `\f`,
/// `\x1c`-`\x1e`, `\x85`, U+2028, U+2029.
fn py_splitlines(s: &str) -> impl Iterator<Item = &str> {
    let mut rest = s;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        let Some((i, c)) = rest.char_indices().find(|&(_, c)| {
            matches!(
                c,
                '\n' | '\r'
                    | '\x0b'
                    | '\x0c'
                    | '\x1c'
                    | '\x1d'
                    | '\x1e'
                    | '\u{85}'
                    | '\u{2028}'
                    | '\u{2029}'
            )
        }) else {
            let line = rest;
            rest = "";
            return Some(line);
        };
        let line = &rest[..i];
        let mut next = i + c.len_utf8();
        if c == '\r' && rest[next..].starts_with('\n') {
            next += 1;
        }
        rest = &rest[next..];
        Some(line)
    })
}

/// Reused per-thread buffers.
#[derive(Debug, Default)]
struct Scratch {
    /// The text's characters, and the byte offset of each (plus the end).
    chars: Vec<char>,
    offs: Vec<usize>,
    /// Segment ends, in characters.
    ends: Vec<u32>,
    joined: Vec<u32>,
    work: newmm::Work,
}

impl Scratch {
    fn load(&mut self, text: &str) {
        self.chars.clear();
        self.offs.clear();
        for (i, c) in text.char_indices() {
            self.chars.push(c);
            self.offs.push(i);
        }
        self.offs.push(text.len());
    }

    /// `newmm.segment(text)` into `self.ends`; with `rejoin`, then
    /// `rejoin_formatted_num`, into `self.joined`. Returns the segments' byte
    /// ranges.
    fn segment(
        &mut self,
        trie: &trie::Trie,
        text: &str,
        safe: bool,
        rejoin: bool,
        v: UnicodeVersion,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.load(text);
        self.ends.clear();
        self.work
            .segment(&self.chars, trie, safe, v, &mut self.ends);
        self.joined.clear();
        if rejoin {
            newmm::rejoin_formatted_num(&self.chars, &self.ends, v, &mut self.joined);
        }
        let ends = if rejoin { &self.joined } else { &self.ends };
        ranges(&self.offs, ends)
    }
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Run `f` with this thread's buffers (fresh ones if they are in use).
fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    let mut f = Some(f);
    let r = SCRATCH.with(|s| {
        s.try_borrow_mut()
            .ok()
            .map(|mut s| (f.take().expect("called once"))(&mut s))
    });
    match r {
        Some(r) => r,
        None => (f.take().expect("called once"))(&mut Scratch::default()),
    }
}

/// The byte ranges of consecutive segments ending at `ends` (characters).
fn ranges<'e>(offs: &'e [usize], ends: &'e [u32]) -> impl Iterator<Item = (usize, usize)> + 'e {
    let mut prev = 0usize;
    ends.iter().map(move |&e| {
        let r = (offs[prev], offs[e as usize]);
        prev = e as usize;
        r
    })
}

/// `tcc_p.segment(text)`: the Thai Character Clusters of `text`, appended to
/// `out`.
pub fn tcc<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let s: Vec<char> = chars.iter().map(|&(_, c)| c).collect();
    let at = |i: usize| chars.get(i).map_or(text.len(), |&(b, _)| b);
    let mut p = 0;
    while p < s.len() {
        let n = tcc::cluster_len(&s, p);
        out.push(&text[at(p)..at(p + n)]);
        p += n;
    }
}

/// `newmm.segment(text, trie, safe_mode=safe)`, appended to `out`.
fn newmm_with<'a>(
    trie: &trie::Trie,
    text: &'a str,
    safe: bool,
    version: UnicodeVersion,
    out: &mut Vec<&'a str>,
) {
    with_scratch(|w| {
        let segs = w.segment(trie, text, safe, false, version);
        out.extend(segs.map(|(s, e)| &text[s..e]));
    });
}

impl Thai {
    /// Read `words_th.txt` and `sentenceseg_crfcut.model` from `corpus_dir`
    /// (PyThaiNLP's `corpus` directory, `pythainlp.corpus.corpus_path()`).
    pub fn open(corpus_dir: impl AsRef<Path>) -> Result<Self, ThaiError> {
        let dir = corpus_dir.as_ref();
        let read = |name: &str| {
            let p = dir.join(name);
            std::fs::read(&p).map_err(|e| ThaiError::Io(p, e))
        };
        let words = read(WORDS_FILE)?;
        let model = read(MODEL_FILE)?;
        let words = std::str::from_utf8(&words)
            .map_err(|e| ThaiError::Format(format!("{WORDS_FILE} is not UTF-8: {e}")))?;
        Self::from_parts(words, &model)
    }

    /// The word list's text and the model file's bytes.
    ///
    /// The words are `get_corpus`'s: a leading BOM is dropped (`utf-8-sig`),
    /// the text is split with `str.splitlines()`, empty lines are skipped and
    /// each word is stripped (`Trie.add`).
    pub fn from_parts(words_text: &str, model_bytes: &[u8]) -> Result<Self, ThaiError> {
        let words_text = words_text.strip_prefix('\u{feff}').unwrap_or(words_text);
        let trie = trie::Trie::new(
            py_splitlines(words_text)
                .filter(|l| !l.is_empty())
                .map(crfcut::py_strip),
        );
        let model = Model::from_bytes(model_bytes).map_err(|e| match e {
            ModelError::Io(e) => ThaiError::Format(e.to_string()),
            ModelError::Format(m) => ThaiError::Format(format!("{MODEL_FILE}: {m}")),
        })?;
        let attrs = crfcut::Attrs::new(&model);
        Ok(Thai { trie, model, attrs })
    }

    /// The number of distinct words in the trie.
    pub fn num_words(&self) -> usize {
        self.trie.len()
    }

    /// The sentence model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// `newmm.segment(text, safe_mode=safe)` with the default dictionary,
    /// appended to `out`.
    pub fn newmm<'a>(
        &self,
        text: &'a str,
        safe: bool,
        version: UnicodeVersion,
        out: &mut Vec<&'a str>,
    ) {
        newmm_with(&self.trie, text, safe, version, out);
    }

    /// `pythainlp.tokenize.word_tokenize(text)` with its defaults (newmm,
    /// `keep_whitespace=True`, `join_broken_num=True`), appended to `out`.
    pub fn word_tokenize<'a>(
        &self,
        text: &'a str,
        version: UnicodeVersion,
        out: &mut Vec<&'a str>,
    ) {
        with_scratch(|w| {
            let segs = w.segment(&self.trie, text, false, true, version);
            out.extend(segs.map(|(s, e)| &text[s..e]));
        });
    }

    /// datatrove's `ThaiTokenizer.word_tokenize(text)`: `word_tokenize(text,
    /// keep_whitespace=False, engine="newmm-safe")`, each word stripped and
    /// the empty ones dropped (`strip_strings`), appended to `out`.
    pub fn datatrove_words<'a>(
        &self,
        text: &'a str,
        version: UnicodeVersion,
        out: &mut Vec<&'a str>,
    ) {
        with_scratch(|w| {
            let segs = w.segment(&self.trie, text, true, true, version);
            out.extend(
                segs.map(|(s, e)| crfcut::py_strip(&text[s..e]))
                    .filter(|w| !w.is_empty()),
            );
        });
    }

    /// The raw labels of crfcut's tagger for `toks`
    /// (`_tagger.tag(_extract_features(toks))`): `true` for `"E"`, written to
    /// `out` (cleared first).
    pub fn crfcut_labels(&self, toks: &[&str], out: &mut Vec<bool>) {
        crfcut::labels(&self.model, &self.attrs, toks, out);
    }

    /// `crfcut.segment(text)`: the sentences' byte ranges, appended to `out`.
    fn crfcut_ranges(&self, text: &str, version: UnicodeVersion, out: &mut Vec<(usize, usize)>) {
        if text.is_empty() {
            return;
        }
        let mut toks = Vec::new();
        with_scratch(|w| {
            toks.extend(w.segment(&self.trie, text, false, true, version));
        });
        let words: Vec<&str> = toks.iter().map(|&(s, e)| &text[s..e]).collect();
        let mut is_e = Vec::new();
        self.crfcut_labels(&words, &mut is_e);
        crfcut::join(text, &toks, &mut is_e, out);
    }

    /// `crfcut.segment(text)` (`sent_tokenize(text)` with its defaults):
    /// the sentences, whitespace kept, appended to `out`.
    pub fn crfcut<'a>(&self, text: &'a str, version: UnicodeVersion, out: &mut Vec<&'a str>) {
        let mut r = Vec::new();
        self.crfcut_ranges(text, version, &mut r);
        out.extend(r.into_iter().map(|(s, e)| &text[s..e]));
    }

    /// datatrove's `ThaiTokenizer.sent_tokenize(text)`: the crfcut sentences
    /// stripped, the empty ones dropped, as byte ranges appended to `out`.
    pub fn datatrove_sentence_ranges(
        &self,
        text: &str,
        version: UnicodeVersion,
        out: &mut Vec<(usize, usize)>,
    ) {
        let first = out.len();
        self.crfcut_ranges(text, version, out);
        let mut keep = first;
        for i in first..out.len() {
            let (s, e) = out[i];
            let raw = &text[s..e];
            let t = crfcut::py_strip(raw);
            if !t.is_empty() {
                let start =
                    s + (raw.len() - raw.trim_start_matches(crate::text::is_py_whitespace).len());
                out[keep] = (start, start + t.len());
                keep += 1;
            }
        }
        out.truncate(keep);
    }

    /// datatrove's `ThaiTokenizer.sent_tokenize(text)`, appended to `out`.
    pub fn datatrove_sentences<'a>(
        &self,
        text: &'a str,
        version: UnicodeVersion,
        out: &mut Vec<&'a str>,
    ) {
        let mut r = Vec::new();
        self.datatrove_sentence_ranges(text, version, &mut r);
        out.extend(r.into_iter().map(|(s, e)| &text[s..e]));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitlines_like_python() {
        let lines: Vec<&str> =
            py_splitlines("a\nb\r\nc\rd\x0be\x1cf\u{2028}g\u{85}\n\nh").collect();
        assert_eq!(lines, ["a", "b", "c", "d", "e", "f", "g", "", "", "h"]);
        assert_eq!(py_splitlines("a\n").collect::<Vec<_>>(), ["a"]);
        assert_eq!(py_splitlines("").count(), 0);
    }

    #[test]
    fn tcc_slices() {
        let mut out = Vec::new();
        tcc("แก้ไข a", &mut out);
        assert_eq!(out, ["แก้", "ไข", " ", "a"]);
    }
}
