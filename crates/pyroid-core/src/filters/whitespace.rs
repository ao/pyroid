// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! datatrove's `WhitespaceTokenizer`, the fallback for languages without a
//! dedicated tokenizer: words from spaCy `xx` (another [`WordTokenizer`]),
//! sentences from one regex (the `regex` module, 2026.9.29):
//!
//! ```text
//! (?:(?:\p{Lu}\.)|.)+?[<TERMINAL_PUNCTUATION>\n]+["'”]?
//! ```
//!
//! `sent_tokenize(text) = strip_strings(findall(text))` and `span_tokenize`
//! is `simple_span_tokenize(text, sent_tokenize(text))`.
//!
//! [`sentence_ends`] reproduces the backtracking matcher's result in one
//! linear pass (no recursion, no quadratic rescans; the regex module itself
//! is quadratic on a long text without terminals): see `match_end`.

use std::borrow::Cow;
use std::sync::LazyLock;

use super::indic::tables;
use super::text::{CharSet, py_rstrip, py_strip};
use super::{FilterError, WordTokenizer};
use crate::ftfy::UnicodeVersion;

/// datatrove `WhitespaceTokenizer`: words from `words` (spaCy `xx`), sentences
/// from the regex. The Unicode version is the words tokenizer's: the regex
/// uses the `regex` module's own Unicode data, whatever the CPython.
#[derive(Debug, Clone, Copy)]
pub struct Whitespace {
    words: &'static dyn WordTokenizer,
}

impl Whitespace {
    /// The tokenizer whose words come from `words`.
    pub const fn new(words: &'static dyn WordTokenizer) -> Self {
        Self { words }
    }
}

impl WordTokenizer for Whitespace {
    fn unicode_version(&self) -> UnicodeVersion {
        self.words.unicode_version()
    }

    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        self.words.words(text, out)
    }

    fn words_cow<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) -> Result<(), FilterError> {
        self.words.words_cow(text, out)
    }

    fn sentence_ends(&self, text: &str, out: &mut Vec<usize>) -> Result<(), FilterError> {
        sentence_ends(text, out)
    }
}

/// The datatrove language codes whose tokenizer is `WhitespaceTokenizer()`,
/// sorted.
pub fn datatrove_languages() -> &'static [&'static str] {
    tables::WHITESPACE_LANGUAGES
}

/// The pattern's `[...\n]` class: `TERMINAL_PUNCTUATION` and `\n`.
static TERMINAL: LazyLock<CharSet> =
    LazyLock::new(|| CharSet::new(tables::WS_TERMINAL.iter().copied()));

#[inline]
fn is_terminal(c: char) -> bool {
    TERMINAL.contains(c)
}

/// `regex`'s `\p{Lu}`.
#[inline]
fn is_upper(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_uppercase();
    }
    let cp = u32::from(c);
    let after = tables::LU.partition_point(|&(lo, _)| lo <= cp);
    after > 0 && cp <= tables::LU[after - 1].1
}

#[inline]
fn char_at(text: &str, i: usize) -> char {
    text[i..].chars().next().unwrap_or('\0')
}

/// Where the match starting at byte `s` ends, given that a terminal
/// character starts at or after the first character (`last_terminal`, the
/// byte offset of the last one in `text`, is at least the end of that
/// character, and `text[s]` is not `\n`).
///
/// The lazy `(?:A|B)+?` (A = `\p{Lu}\.`, B = `.`) first takes one iteration,
/// then at each position tries the rest of the pattern before another
/// iteration, and A before B. The rest, `T+["'”]?` with nothing after it,
/// succeeds exactly at a terminal character `T`. Neither A nor B consumes a
/// `\n`, which is terminal, so from position `p` the search succeeds iff a
/// terminal character starts at or after `p`. Depth-first search therefore
/// takes A at `p` whenever A applies and a terminal follows the `.` it eats;
/// if none follows, B leaves it at that `.`, which is terminal. Otherwise it
/// steps over one character. The first terminal reached is where `T+` starts
/// (greedily), then `["'”]?`.
fn match_end(text: &str, start: usize, last_terminal: usize) -> usize {
    let bytes = text.as_bytes();
    let mut p = start;
    let mut first = true;
    let run = loop {
        let c = char_at(text, p);
        if !first && is_terminal(c) {
            break p;
        }
        first = false;
        let next = p + c.len_utf8();
        if bytes.get(next) == Some(&b'.') && is_upper(c) {
            if last_terminal > next {
                p = next + 1;
                continue;
            }
            break next;
        }
        p = next;
    };
    let mut end = run;
    while end < text.len() {
        let c = char_at(text, end);
        if !is_terminal(c) {
            if matches!(c, '"' | '\'' | '\u{201d}') {
                end += c.len_utf8();
            }
            break;
        }
        end += c.len_utf8();
    }
    end
}

/// The byte ranges of `_sent_regex.findall(text)`, in order.
fn find_all(text: &str, mut f: impl FnMut(usize, usize)) {
    let Some(last_terminal) = text
        .char_indices()
        .rev()
        .find(|&(_, c)| is_terminal(c))
        .map(|(i, _)| i)
    else {
        return;
    };
    let mut s = 0;
    while s < text.len() {
        let c = char_at(text, s);
        // No match starts at a '\n' (neither A nor B takes it); the search
        // moves on by one character.
        if c == '\n' {
            s += 1;
            continue;
        }
        // Without a terminal after the first character nothing matches here,
        // nor anywhere later.
        if last_terminal < s + c.len_utf8() {
            return;
        }
        let e = match_end(text, s, last_terminal);
        f(s, e);
        s = e;
    }
}

/// `WhitespaceTokenizer.sent_tokenize(text)`, appended to `out`: the regex
/// matches, stripped, empty ones dropped. Every sentence is a slice of
/// `text`.
pub fn sent_tokenize<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    find_all(text, |s, e| {
        let sent = py_strip(&text[s..e]);
        if !sent.is_empty() {
            out.push(sent);
        }
    });
}

/// The end byte offsets of `WhitespaceTokenizer.span_tokenize(text)`,
/// appended to `out`. Never fails.
///
/// `simple_span_tokenize` looks each sentence up with `text.index(sent,
/// start)` from the previous end; what lies between two stripped matches is
/// whitespace (stripped ends and skipped `\n`s), so that finds each sentence
/// where it was matched, and its end is the stripped match's end. A single
/// sentence spans the whole text.
pub fn sentence_ends(text: &str, out: &mut Vec<usize>) -> Result<(), FilterError> {
    let first = out.len();
    find_all(text, |s, e| {
        let sent = &text[s..e];
        if !py_strip(sent).is_empty() {
            out.push(s + py_rstrip(sent).len());
        }
    });
    if out.len() - first == 1 {
        out[first] = text.len();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sents(text: &str) -> Vec<&str> {
        let mut out = Vec::new();
        sent_tokenize(text, &mut out);
        out
    }

    fn ends(text: &str) -> Vec<usize> {
        let mut out = Vec::new();
        sentence_ends(text, &mut out).unwrap();
        out
    }

    #[test]
    fn splits() {
        assert_eq!(
            sents("Hello there. How are you?"),
            ["Hello there.", "How are you?"]
        );
        assert_eq!(sents("abc.\n\ndef.\n"), ["abc.", "def."]);
        assert_eq!(sents("no end"), Vec::<&str>::new());
        assert_eq!(sents("a\nb"), ["a"]);
        assert_eq!(sents("\n\n x!?\" y"), ["x!?\""]);
        assert_eq!(sents("x.A.b"), ["x.", "A."]);
        assert_eq!(sents(" \n.\n"), ["."]);
        assert_eq!(sents(".."), [".."]);
    }

    #[test]
    fn acronyms() {
        // `\p{Lu}\.` steps over the period when a terminal follows.
        assert_eq!(sents("U.S. is big. Ok."), ["U.S. is big.", "Ok."]);
        assert_eq!(sents("U.S."), ["U.S."]);
        assert_eq!(sents("A. b"), ["A."]);
        assert_eq!(sents("Д.Ж. и т.д. Э."), ["Д.Ж. и т.", "д.", "Э."]);
        assert_eq!(sents("a A.\nb"), ["a A."]);
    }

    #[test]
    fn spans() {
        assert_eq!(ends("Hello there. How are you?  "), [12, 25]);
        assert_eq!(ends("  one sentence. rest"), [20]);
        assert_eq!(ends("nothing"), Vec::<usize>::new());
        assert_eq!(ends("a।b॥c"), [4, 8]);
    }

    #[test]
    fn long_text_without_terminals_is_linear() {
        let text = "Ab ".repeat(400_000);
        assert_eq!(ends(&text), Vec::<usize>::new());
        let text = format!("{}.", "A.b".repeat(300_000));
        assert_eq!(ends(&text), [text.len()]);
        let text = "x\n".repeat(300_000);
        assert_eq!(ends(&text).len(), 300_000);
    }
}
