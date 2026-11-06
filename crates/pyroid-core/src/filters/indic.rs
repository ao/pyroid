// Derived from indic-nlp-library 0.92 (MIT), © Anoop Kunchukuttan, urduhack
// 1.0.3 (MIT), © Urdu Hack, and datatrove 0.10.1 (Apache-2.0), © HuggingFace;
// modified.
//! datatrove's `IndicNLPTokenizer`: indic-nlp-library 0.92's
//! `trivial_tokenize` (words) and `sentence_split` (sentences), wrapped the
//! way datatrove wraps them:
//!
//! - `word_tokenize(text) = strip_strings(trivial_tokenize(text, lang))`
//! - `sent_tokenize(text) = strip_strings(sentence_split(text, lang))`
//! - `span_tokenize(text) = simple_span_tokenize(text, sent_tokenize(text))`
//!
//! Every quirk is kept: the number-merging pass skips a number sequence at
//! the very start of the text, merged numbers lose the spaces around their
//! separators (so such a word is *not* a slice of the input), the abbreviation
//! pass of `sentence_split` drops a buffered sentence in its "bad state" and
//! joins sentences with `' '` (so `text.index` can fail: datatrove raises
//! `ValueError: substring not found`), and `sentence_split` for `ur` is
//! urduhack 1.0.3's rule-based splitter (the module indic-nlp imports; it is
//! importable when urduhack and TensorFlow are installed, as datatrove's
//! `multilingual` extra does).
//!
//! Only `str.isnumeric()` depends on the Unicode version (the character before
//! a sentence delimiter); everything else is ASCII, fixed character sets, or
//! `str.isspace()`, which is the same in every supported version.

use std::borrow::Cow;
use std::sync::LazyLock;

use memchr::memmem;

use super::text::{CharSet, py_split, py_strip};
use super::{FilterError, WordTokenizer};
use crate::ftfy::UnicodeVersion;

#[rustfmt::skip]
#[path = "indic_tables.rs"]
pub(super) mod tables;

/// How `UnicodeIndicTransliterator.transliterate(text, lang, "hi")` maps a
/// language's characters to Devanagari (only `is_acronym_abbvr` uses it).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Script {
    /// `SCRIPT_RANGES[lang][0]`: offsets 0..=0x6F from it move to U+0900.
    Offset(u32),
    /// Sinhala: `sinhala_to_devanagari` first, then the Devanagari rule.
    Sinhala,
    /// Not in `SCRIPT_RANGES`: text unchanged. Also selects the Urdu
    /// tokenizer and urduhack's sentence splitter.
    Urdu,
}

/// datatrove's `IndicNLPTokenizer(lang)` for one indic-nlp language code and
/// one Unicode version. Get one with [`IndicNlp::get`].
#[derive(Debug)]
pub struct IndicNlp {
    lang: &'static str,
    script: Script,
    /// `langinfo.is_danda_delim(lang)`.
    danda: bool,
    version: UnicodeVersion,
}

const VERSIONS: [UnicodeVersion; 5] = [
    UnicodeVersion::V14_0,
    UnicodeVersion::V15_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V16_0,
    UnicodeVersion::V17_0,
];

static INSTANCES: LazyLock<Vec<IndicNlp>> = LazyLock::new(|| {
    tables::LANGS
        .iter()
        .flat_map(|&(lang, script, danda)| {
            VERSIONS.map(|version| IndicNlp {
                lang,
                script,
                danda,
                version,
            })
        })
        .collect()
});

static INDIC_PUNCT: LazyLock<CharSet> =
    LazyLock::new(|| CharSet::new(tables::INDIC_PUNCT.iter().copied()));
static URDU_PUNCT: LazyLock<CharSet> =
    LazyLock::new(|| CharSet::new(tables::URDU_PUNCT.iter().copied()));
static DELIM_DANDA: LazyLock<CharSet> =
    LazyLock::new(|| CharSet::new(tables::DELIM_DANDA.iter().copied()));
static DELIM_NO_DANDA: LazyLock<CharSet> =
    LazyLock::new(|| CharSet::new(tables::DELIM_NO_DANDA.iter().copied()));

/// The longest entry of `tables::ACRONYMS`, in bytes.
static MAX_ACRONYM: LazyLock<usize> =
    LazyLock::new(|| tables::ACRONYMS.iter().map(|a| a.len()).max().unwrap_or(0));

impl IndicNlp {
    /// The tokenizer for an indic-nlp language code as datatrove passes it
    /// (`tok_code` in `tokenizer_assignment.csv`: `as`, `bn`, `gu`, `hi`,
    /// `kK`, `kn`, `ml`, `mr`, `ne`, `or`, `pa`, `sa`, `si`, `ta`, `te`,
    /// `ur`); `None` for any other code.
    pub fn get(tok_code: &str, version: UnicodeVersion) -> Option<&'static IndicNlp> {
        let i = tables::LANGS
            .binary_search_by(|&(code, _, _)| code.cmp(tok_code))
            .ok()?;
        let v = VERSIONS.iter().position(|&x| x == version)?;
        Some(&INSTANCES[i * VERSIONS.len() + v])
    }

    /// The indic-nlp language codes [`IndicNlp::get`] accepts, sorted.
    pub fn tok_codes() -> impl Iterator<Item = &'static str> {
        tables::LANGS.iter().map(|&(code, _, _)| code)
    }

    /// The indic-nlp language code.
    pub fn language(&self) -> &'static str {
        self.lang
    }

    /// `IndicNLPTokenizer.word_tokenize(text)` (= `split_into_words`),
    /// appended to `out`. A word is borrowed from `text` unless the
    /// number-merging pass removed spaces from it (`"1 , 000"` -> `"1,000"`).
    pub fn word_tokenize<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) {
        let urdu = self.script == Script::Urdu;
        let mut toks = Vec::new();
        split_tokens(
            text,
            if urdu { &URDU_PUNCT } else { &INDIC_PUNCT },
            &mut toks,
        );
        if urdu {
            out.extend(
                toks.iter()
                    .map(|&(a, b)| py_strip(&text[a..b]))
                    .filter(|w| !w.is_empty())
                    .map(Cow::Borrowed),
            );
            return;
        }
        let mut glue = vec![false; toks.len()];
        glue_numbers(text, &toks, &mut glue);
        let mut i = 0;
        while i < toks.len() {
            let mut j = i;
            while glue[j] {
                j += 1;
            }
            let word = if (i..j).all(|k| toks[k].1 == toks[k + 1].0) {
                Cow::Borrowed(py_strip(&text[toks[i].0..toks[j].1]))
            } else {
                let joined: String = toks[i..=j].iter().map(|&(a, b)| &text[a..b]).collect();
                Cow::Owned(py_strip(&joined).to_owned())
            };
            if !word.is_empty() {
                out.push(word);
            }
            i = j + 1;
        }
    }

    /// `IndicNLPTokenizer.sent_tokenize(text)`, appended to `out`. Sentences
    /// that `sentence_split` joined are owned (they may not occur in `text`).
    ///
    /// Fails only for `ur` on a text containing U+10FFFF, where urduhack's
    /// `chr(ord(max(text)) + 1)` raises `ValueError` (reported as
    /// [`FilterError::SubstringNotFound`], the other `ValueError`).
    pub fn sent_tokenize<'a>(
        &self,
        text: &'a str,
        out: &mut Vec<Cow<'a, str>>,
    ) -> Result<(), FilterError> {
        if self.script == Script::Urdu {
            return urdu_sentences(text, out);
        }
        self.sentence_split(text, out);
        Ok(())
    }

    /// `sentence_split(text, lang)` (non-Urdu) followed by `strip_strings`.
    fn sentence_split<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) {
        // Danda languages split on dandas only if the text has one.
        let danda = self.danda && text.contains(['\u{964}', '\u{965}']);
        let delims: &CharSet = if danda { &DELIM_DANDA } else { &DELIM_NO_DANDA };
        // Phase 1: break after each delimiter not preceded by a numeric char.
        let t = py_strip(text);
        let mut cand: Vec<&'a str> = Vec::new();
        let mut begin = 0;
        let mut prev: Option<char> = None;
        for (i, c) in t.char_indices() {
            if delims.contains(c) && !prev.is_some_and(|p| is_numeric(p, self.version)) {
                let end = i + c.len_utf8();
                push_stripped(&mut cand, &t[begin..end]);
                begin = end;
            }
            prev = Some(c);
        }
        push_stripped(&mut cand, &t[begin..]);
        if danda {
            // DELIM_PAT_DANDA has no '.': phase 2 does not run.
            out.extend(cand.into_iter().map(Cow::Borrowed));
            return;
        }
        // Phase 2: '.' may not end a sentence (single words, abbreviations).
        let start = out.len();
        let mut buf: Cow<'a, str> = Cow::Borrowed("");
        let mut bad = false;
        for sentence in cand {
            let ends_with_period = sentence.ends_with('.');
            let last_word = sentence.rsplit(' ').next().unwrap_or(sentence);
            if ends_with_period && !sentence.contains(' ') {
                bad = true;
                buf = join(buf, sentence);
            } else if ends_with_period && self.is_acronym(&last_word[..last_word.len() - 1]) {
                if !buf.is_empty() && !bad {
                    out.push(buf);
                }
                bad = true;
                buf = Cow::Borrowed(sentence);
            } else if bad {
                out.push(join(buf, sentence));
                buf = Cow::Borrowed("");
                bad = false;
            } else {
                if !buf.is_empty() {
                    out.push(buf);
                }
                buf = Cow::Borrowed(sentence);
            }
        }
        if !buf.is_empty() {
            out.push(buf);
        }
        // strip_strings (only buffers that began with ' ' change).
        let mut k = start;
        while k < out.len() {
            let stripped = py_strip(&out[k]);
            if stripped.is_empty() {
                out.remove(k);
                continue;
            }
            if stripped.len() != out[k].len() {
                out[k] = Cow::Owned(stripped.to_owned());
            }
            k += 1;
        }
    }

    /// `is_acronym_abbvr(word, lang)`: the word, transliterated to
    /// Devanagari, is one of the non-breaking phrases.
    fn is_acronym(&self, word: &str) -> bool {
        // Every phrase is Devanagari (3 bytes a character) and every character
        // that maps into Devanagari is 3 bytes too, so lengths must agree.
        if word.is_empty() || word.len() > *MAX_ACRONYM {
            return false;
        }
        let mut buf = String::with_capacity(word.len());
        buf.extend(word.chars().map(|c| to_devanagari(c, self.script)));
        tables::ACRONYMS.binary_search(&buf.as_str()).is_ok()
    }
}

/// `buf + ' ' + sentence`.
fn join<'a>(buf: Cow<'a, str>, sentence: &str) -> Cow<'a, str> {
    let mut s = buf.into_owned();
    s.reserve(1 + sentence.len());
    s.push(' ');
    s.push_str(sentence);
    Cow::Owned(s)
}

fn push_stripped<'a>(out: &mut Vec<&'a str>, s: &'a str) {
    let s = py_strip(s);
    if !s.is_empty() {
        out.push(s);
    }
}

/// `str.isnumeric()` for one character.
fn is_numeric(c: char, v: UnicodeVersion) -> bool {
    if c.is_ascii() {
        return c.is_ascii_digit();
    }
    let cp = u32::from(c);
    let ranges = tables::numeric(v);
    let after = ranges.partition_point(|&(lo, _)| lo <= cp);
    after > 0 && cp <= ranges[after - 1].1
}

/// One character of `UnicodeIndicTransliterator.transliterate(c, lang, "hi")`.
fn to_devanagari(c: char, script: Script) -> char {
    let (c, base) = match script {
        Script::Offset(base) => (c, base),
        Script::Sinhala => {
            let mapped = tables::SINHALA_TO_DEVANAGARI
                .binary_search_by_key(&c, |&(s, _)| s)
                .map_or(c, |i| tables::SINHALA_TO_DEVANAGARI[i].1);
            (mapped, 0x900)
        }
        Script::Urdu => return c,
    };
    let offset = u32::from(c).wrapping_sub(base);
    if offset <= 0x6F && c != '\u{964}' && c != '\u{965}' {
        char::from_u32(0x900 + offset).unwrap_or(c)
    } else {
        c
    }
}

/// The tokens of `s.split(' ')` in `trivial_tokenize`, as byte ranges of
/// `text`: tabs become spaces, each character of `punct` is padded with
/// spaces, runs of spaces collapse and the ends are stripped of spaces, so
/// the tokens are the maximal runs of characters other than `' '`, `'\t'`
/// and `punct`, plus every `punct` character on its own.
fn split_tokens(text: &str, punct: &CharSet, out: &mut Vec<(usize, usize)>) {
    let mut start = None;
    for (i, c) in text.char_indices() {
        if c == ' ' || c == '\t' {
            if let Some(s) = start.take() {
                out.push((s, i));
            }
        } else if punct.contains(c) {
            if let Some(s) = start.take() {
                out.push((s, i));
            }
            out.push((i, i + c.len_utf8()));
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        out.push((s, text.len()));
    }
}

/// `trivial_tokenize_indic`'s "do not tokenize numbers and dates" pass over
/// the space-joined tokens: for each `pat_num_seq` match
/// (`([0-9]+ [,.:/] )+[0-9]+`) the spaces inside it are removed, gluing its
/// tokens (and the rest of the first and last token) into one word. A match
/// at position 0 is left alone (`if start>prev` with `prev=0`). Sets
/// `glue[i]` when token `i` joins token `i + 1`.
fn glue_numbers(text: &str, toks: &[(usize, usize)], glue: &mut [bool]) {
    let count = toks.len();
    let tok = |k: usize| &text.as_bytes()[toks[k].0..toks[k].1];
    // A token that is one of `[,.:/]` followed by a space (another token).
    let sep_then_space = |k: usize| k + 1 < count && matches!(tok(k), [b',' | b'.' | b':' | b'/']);
    let all_digits = |k: usize| tok(k).iter().all(u8::is_ascii_digit);
    // Search position: token `i`, byte `off` within it.
    let (mut i, mut off) = (0, 0);
    while i < count {
        let cur = tok(i);
        // A match starts at a digit whose run reaches the token's end (then
        // a space): the token's all-digit suffix, from `off` on.
        let from =
            (cur.len() - cur.iter().rev().take_while(|b| b.is_ascii_digit()).count()).max(off);
        if from == cur.len() || !sep_then_space(i + 1) {
            (i, off) = (i + 1, 0);
            continue;
        }
        // Greedy `(...)+`: more "digits-only token, separator, space".
        let mut iterations = 1;
        let mut j = i + 2;
        while all_digits(j) && sep_then_space(j + 1) {
            iterations += 1;
            j += 2;
        }
        // The final `[0-9]+`: the next token's leading digits, or (one
        // iteration given back) the whole digits-only token before.
        let lead = tok(j).iter().take_while(|b| b.is_ascii_digit()).count();
        let (last, end) = if lead > 0 {
            (j, lead)
        } else if iterations >= 2 {
            (j - 2, tok(j - 2).len())
        } else {
            (i, off) = (i + 1, 0);
            continue;
        };
        if i > 0 || from > 0 {
            glue[i..last].fill(true);
        }
        (i, off) = if end == tok(last).len() {
            (last + 1, 0)
        } else {
            (last, end)
        };
    }
}

/// urduhack 1.0.3 `tokenization.eos._generate_sentences(text)` (what
/// `sentence_split(text, "ur")` returns), then `strip_strings`.
fn urdu_sentences<'a>(text: &'a str, out: &mut Vec<Cow<'a, str>>) -> Result<(), FilterError> {
    // `_split_and_keep` computes `chr(ord(max(text)) + 1)`.
    if text.contains('\u{10FFFF}') {
        return Err(FilterError::SubstringNotFound);
    }
    // `_split_and_keep(text, '۔')`; a trailing empty piece never counts.
    for sentence in text.split_inclusive('\u{6d4}') {
        if py_split(sentence).nth(1).is_none() {
            continue;
        }
        if sentence.contains('\u{61f}') {
            for piece in sentence.split_inclusive('\u{61f}') {
                urdu_lines(piece, out);
            }
        } else {
            urdu_lines(sentence, out);
        }
    }
    Ok(())
}

/// One piece of `_generate_sentences`: its words, cut after a "newline word"
/// that is not followed by a conjunction (taking a following `۔` or `،`
/// along); each cut of two words or more becomes a sentence joined by `' '`.
fn urdu_lines(piece: &str, out: &mut Vec<Cow<'_, str>>) {
    let words: Vec<&str> = py_split(piece).collect();
    let mut emit = |ws: &[&str]| {
        if ws.len() >= 2 {
            out.push(Cow::Owned(ws.join(" ")));
        }
    };
    let (mut start, mut k) = (0, 0);
    while k < words.len() {
        let breaks = k + 1 < words.len()
            && tables::URDU_NEWLINE_WORDS.binary_search(&words[k]).is_ok()
            && tables::URDU_CONJUNCTIONS
                .binary_search(&words[k + 1])
                .is_err();
        if !breaks {
            k += 1;
        } else if matches!(words[k + 1], "\u{6d4}" | "\u{60c}") {
            emit(&words[start..k + 2]);
            (start, k) = (k + 2, k + 2);
        } else {
            emit(&words[start..=k]);
            (start, k) = (k + 1, k + 1);
        }
    }
    emit(&words[start..]);
}

/// datatrove's `simple_span_tokenize(text, sents)`: the end byte offset of
/// each sentence, found with `text.index(sent, start)` from the previous end
/// (`(0, len(text))` for a single sentence).
pub(super) fn simple_span_ends<S: AsRef<str>>(
    text: &str,
    sents: &[S],
    out: &mut Vec<usize>,
) -> Result<(), FilterError> {
    if sents.len() == 1 {
        out.push(text.len());
        return Ok(());
    }
    let first = out.len();
    let mut start = 0;
    for sent in sents {
        let sent = sent.as_ref().as_bytes();
        let Some(at) = memmem::find(&text.as_bytes()[start..], sent) else {
            out.truncate(first);
            return Err(FilterError::SubstringNotFound);
        };
        start += at + sent.len();
        out.push(start);
    }
    Ok(())
}

/// The datatrove language codes that use `IndicNLPTokenizer`, with the
/// indic-nlp language code each passes (for [`IndicNlp::get`]), sorted.
pub fn datatrove_languages() -> &'static [(&'static str, &'static str)] {
    tables::INDIC_LANGUAGES
}

impl WordTokenizer for IndicNlp {
    fn unicode_version(&self) -> UnicodeVersion {
        self.version
    }

    /// The words of [`IndicNlp::word_tokenize`] while they are all slices of
    /// `text`. A merged number whose spaces were removed is not one: such a
    /// text fails with [`FilterError::SubstringNotFound`] rather than giving
    /// wrong words. The filters and MinHash use [`WordTokenizer::words_cow`].
    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        let mut words = Vec::new();
        self.word_tokenize(text, &mut words);
        let first = out.len();
        for w in words {
            let Cow::Borrowed(w) = w else {
                out.truncate(first);
                return Err(FilterError::SubstringNotFound);
            };
            out.push(w);
        }
        Ok(())
    }

    fn words_cow<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) -> Result<(), FilterError> {
        self.word_tokenize(text, out);
        Ok(())
    }

    fn sentence_ends(&self, text: &str, out: &mut Vec<usize>) -> Result<(), FilterError> {
        let mut sents = Vec::new();
        self.sent_tokenize(text, &mut sents)?;
        simple_span_ends(text, &sents, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hi() -> &'static IndicNlp {
        IndicNlp::get("hi", UnicodeVersion::V15_1).unwrap()
    }

    fn words(tok: &IndicNlp, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        tok.word_tokenize(text, &mut out);
        out.into_iter().map(Cow::into_owned).collect()
    }

    fn sents(tok: &IndicNlp, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        tok.sent_tokenize(text, &mut out).unwrap();
        out.into_iter().map(Cow::into_owned).collect()
    }

    #[test]
    fn codes() {
        let codes: Vec<&str> = IndicNlp::tok_codes().collect();
        assert_eq!(
            codes,
            [
                "as", "bn", "gu", "hi", "kK", "kn", "ml", "mr", "ne", "or", "pa", "sa", "si", "ta",
                "te", "ur"
            ]
        );
        for code in codes {
            for v in VERSIONS {
                let tok = IndicNlp::get(code, v).unwrap();
                assert_eq!((tok.language(), tok.unicode_version()), (code, v));
            }
        }
        assert!(IndicNlp::get("en", UnicodeVersion::V15_1).is_none());
        assert!(datatrove_languages().contains(&("hin_Deva", "hi")));
        assert!(datatrove_languages().contains(&("urd_Arab", "ur")));
    }

    #[test]
    fn number_merging() {
        let tok = hi();
        assert_eq!(
            words(tok, "a 1 , 000 b 1,000 c. d"),
            ["a", "1,000", "b", "1,000", "c", ".", "d"]
        );
        // A sequence at the very start is not merged; later ones are.
        assert_eq!(words(tok, "1,000 x 2,5"), ["1", ",", "000", "x", "2,5"]);
        assert_eq!(words(tok, "\n1,000"), ["1,000"]);
        // The match covers digit runs at the token edges only.
        assert_eq!(words(tok, "x ab12,34cd"), ["x", "ab12,34cd"]);
        assert_eq!(words(tok, "x 1,2,y"), ["x", "1,2", ",", "y"]);
        assert_eq!(words(tok, "x 1,2,3x,4"), ["x", "1,2,3x", ",", "4"]);
        assert_eq!(words(tok, "x 1,"), ["x", "1", ","]);
        assert_eq!(
            words(tok, "x 12:30 1/2 3.14 1-2"),
            ["x", "12:30", "1/2", "3.14", "1", "-", "2"]
        );
        let mut out = Vec::new();
        tok.word_tokenize("x 1, 2", &mut out);
        assert!(matches!(&out[1], Cow::Owned(w) if w == "1,2"));
        tok.word_tokenize("x 1,2", &mut out);
        assert!(matches!(out[3], Cow::Borrowed("1,2")));
        // Urdu: no merging, its own punctuation.
        let ur = IndicNlp::get("ur", UnicodeVersion::V15_1).unwrap();
        assert_eq!(
            words(ur, "a 1,000\tb\u{6d4}"),
            ["a", "1", ",", "000", "b", "\u{6d4}"]
        );
    }

    #[test]
    fn words_trait_refuses_merged_numbers() {
        let mut out = vec!["keep"];
        assert_eq!(
            hi().words("x 1 , 2", &mut out),
            Err(FilterError::SubstringNotFound)
        );
        assert_eq!(out, ["keep"]);
        hi().words("x 1,2 \\ y", &mut out).unwrap();
        assert_eq!(out, ["keep", "x", "1,2", "\\", "y"]);
    }

    #[test]
    fn sentences() {
        let tok = hi();
        // No danda: '.' splits; a single word ending in '.' and an
        // abbreviation start a "bad state" that swallows a buffered sentence.
        assert_eq!(sents(tok, "क. ख ग. घ."), ["ख ग. घ."]);
        assert_eq!(sents(tok, "a b. c d. e f"), ["a b.", "c d.", "e f"]);
        assert_eq!(sents(tok, "x.  y z"), ["x. y z"]);
        // With a danda only ? ! । ॥ split, and never after a numeric char.
        assert_eq!(sents(tok, "a. b। c२। d"), ["a. b।", "c२। d"]);
        let mut ends = Vec::new();
        assert_eq!(
            tok.sentence_ends("a b. c d. x.\ny z", &mut ends),
            Err(FilterError::SubstringNotFound)
        );
        tok.sentence_ends(" a b. c d ", &mut ends).unwrap();
        assert_eq!(ends, [5, 9]);
        ends.clear();
        tok.sentence_ends(" a b ", &mut ends).unwrap();
        assert_eq!(ends, [5]);
    }

    #[test]
    fn transliteration() {
        assert_eq!(to_devanagari('\u{995}', Script::Offset(0x980)), '\u{915}');
        assert_eq!(to_devanagari('\u{9f0}', Script::Offset(0x980)), '\u{9f0}');
        assert_eq!(to_devanagari('\u{964}', Script::Offset(0x900)), '\u{964}');
        assert_eq!(to_devanagari('\u{d9a}', Script::Sinhala), '\u{915}');
        assert_eq!(to_devanagari('k', Script::Offset(0x900)), 'k');
        let bn = IndicNlp::get("bn", UnicodeVersion::V15_1).unwrap();
        assert!(bn.is_acronym("\u{995}") && !bn.is_acronym("\u{915}\u{915}"));
    }

    #[test]
    fn urdu_splitter() {
        let ur = IndicNlp::get("ur", UnicodeVersion::V15_1).unwrap();
        // "... ہے" ends a sentence unless a conjunction follows.
        assert_eq!(sents(ur, "a b ہے c d\u{6d4} e"), ["a b ہے", "c d\u{6d4}"]);
        assert_eq!(sents(ur, "a ہے اور b"), ["a ہے اور b"]);
        assert_eq!(sents(ur, "a b ہے \u{6d4} c d"), ["a b ہے \u{6d4}", "c d"]);
        assert_eq!(sents(ur, "x"), Vec::<String>::new());
        let mut out = Vec::new();
        assert_eq!(
            ur.sent_tokenize("a b \u{10ffff}", &mut out),
            Err(FilterError::SubstringNotFound)
        );
    }

    #[test]
    fn numeric_by_version() {
        // U+3405 (CJK, numeric 5) is numeric and alpha in every version.
        for v in VERSIONS {
            assert!(is_numeric('\u{3405}', v) && is_numeric('7', v) && !is_numeric('a', v));
        }
        // KAWI DIGIT ZERO is new in Unicode 15.0.
        assert!(!is_numeric('\u{11f50}', UnicodeVersion::V14_0));
        assert!(is_numeric('\u{11f50}', UnicodeVersion::V15_0));
        // U+4E24 got a numeric value in Unicode 15.1.
        assert!(!is_numeric('\u{4e24}', UnicodeVersion::V15_0));
        assert!(is_numeric('\u{4e24}', UnicodeVersion::V15_1));
    }

    #[test]
    fn tables_are_sorted() {
        let ranges: [&[(u32, u32)]; 6] = [
            tables::numeric(UnicodeVersion::V14_0),
            tables::numeric(UnicodeVersion::V15_0),
            tables::numeric(UnicodeVersion::V15_1),
            tables::numeric(UnicodeVersion::V16_0),
            tables::numeric(UnicodeVersion::V17_0),
            &tables::LU,
        ];
        for t in ranges {
            assert!(t.iter().all(|&(lo, hi)| lo <= hi && hi <= 0x10_FFFF));
            assert!(t.windows(2).all(|w| w[0].1 + 1 < w[1].0));
        }
        assert!(tables::ACRONYMS.windows(2).all(|w| w[0] < w[1]));
        assert!(tables::URDU_NEWLINE_WORDS.windows(2).all(|w| w[0] < w[1]));
        assert!(tables::URDU_CONJUNCTIONS.windows(2).all(|w| w[0] < w[1]));
        assert!(tables::LANGS.windows(2).all(|w| w[0].0 < w[1].0));
        assert!(
            tables::SINHALA_TO_DEVANAGARI
                .windows(2)
                .all(|w| w[0].0 < w[1].0)
        );
        assert!(tables::ACRONYMS.iter().all(|a| {
            a.chars()
                .all(|c| ('\u{900}'..='\u{96f}').contains(&c) && c.len_utf8() == 3)
        }));
    }

    /// Every code point in every version against the generator's dumps:
    /// `PYROID_INDIC_DUMPS=/target/gen-indic cargo test -p pyroid-core
    /// filters::indic -- --ignored`.
    #[test]
    #[ignore = "needs the generate_indic_tables.py dumps in $PYROID_INDIC_DUMPS"]
    fn numeric_matches_cpython_dumps() {
        let dir = std::env::var("PYROID_INDIC_DUMPS").expect("set PYROID_INDIC_DUMPS");
        let mut seen = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let dump: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            let v =
                UnicodeVersion::from_unidata_version(dump["unicode"].as_str().unwrap()).unwrap();
            let mut want = vec![false; 0x11_0000];
            for r in dump["numeric"].as_array().unwrap() {
                let lo = usize::try_from(r[0].as_u64().unwrap()).unwrap();
                let hi = usize::try_from(r[1].as_u64().unwrap()).unwrap();
                want[lo..=hi].fill(true);
            }
            for c in (0..=0x10_FFFF).filter_map(char::from_u32) {
                let i = usize::try_from(u32::from(c)).unwrap();
                assert_eq!(is_numeric(c, v), want[i], "{c:?} {v:?}");
            }
            seen.push(v);
        }
        seen.sort();
        assert_eq!(seen, VERSIONS);
    }
}
