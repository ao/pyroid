// Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
//! NLTK's regular-expression and string tokenizers: `WordPunctTokenizer`
//! (`wordpunct_tokenize`), `WhitespaceTokenizer`, `BlanklineTokenizer` and
//! a few common `RegexpTokenizer` patterns (`nltk.tokenize.regexp`), and
//! `SpaceTokenizer`, `TabTokenizer` and `LineTokenizer`
//! (`nltk.tokenize.simple`), each with its `span_tokenize`.
//!
//! Tokens are slices of the text; spans are `(start, end)` in code points,
//! as Python indexes the text. NLTK runs the patterns with the `regex`
//! module, so `\w`, `\d` and `\s` are `chars::rx_word`, `rx_digit` and
//! `rx_space` (U+001C..U+001F are not `\s`). Every native pattern matches a
//! non-empty run, found leftmost and without overlap as `findall`,
//! `finditer` and `split` find it.

use super::chars::{py_space, rx_digit, rx_space, rx_word};

/// A `RegexpTokenizer` pattern that runs natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Pattern {
    /// `\w+`
    Word,
    /// `\w+|[^\w\s]+` (`WordPunctTokenizer`).
    WordPunct,
    /// `\w+|\$[\d\.]+|\S+` (the `RegexpTokenizer` docstring's example).
    WordMoneyNonSpace,
    /// `[\w']+`
    WordApostrophe,
    /// `\S+`
    NonSpace,
    /// `\s+` (`WhitespaceTokenizer`, with `gaps=True`).
    Space,
    /// `[a-zA-Z]+`
    AsciiLetters,
    /// `(?:(?<!\s)\s*)?\n\s*\n\s*` (`BlanklineTokenizer`, with `gaps=True`).
    Blankline,
}

/// The `re` flags that change nothing for the native patterns:
/// `re.UNICODE | re.MULTILINE | re.DOTALL`, `RegexpTokenizer`'s default.
pub const DEFAULT_FLAGS: u32 = 32 | 8 | 16;

impl Pattern {
    /// The native pattern for `RegexpTokenizer(pattern, flags=flags)`, when
    /// `pattern` is spelled exactly as one of them and `flags` holds no
    /// flag other than [`DEFAULT_FLAGS`]' (`IGNORECASE`, `ASCII`, `VERBOSE`
    /// and the others change what the patterns match).
    ///
    /// ```
    /// use pyroid_core::nltk::regexp::{DEFAULT_FLAGS, Pattern};
    /// assert_eq!(Pattern::from_regex(r"\w+", DEFAULT_FLAGS), Some(Pattern::Word));
    /// assert_eq!(Pattern::from_regex(r"\w+", 2), None); // re.IGNORECASE
    /// assert_eq!(Pattern::from_regex(r"[\w]+", DEFAULT_FLAGS), None);
    /// ```
    pub fn from_regex(pattern: &str, flags: u32) -> Option<Self> {
        if flags & !DEFAULT_FLAGS != 0 {
            return None;
        }
        Some(match pattern {
            r"\w+" => Self::Word,
            r"\w+|[^\w\s]+" => Self::WordPunct,
            r"\w+|\$[\d\.]+|\S+" => Self::WordMoneyNonSpace,
            r"[\w']+" => Self::WordApostrophe,
            r"\S+" => Self::NonSpace,
            r"\s+" => Self::Space,
            r"[a-zA-Z]+" => Self::AsciiLetters,
            r"(?:(?<!\s)\s*)?\n\s*\n\s*" => Self::Blankline,
            _ => return None,
        })
    }

    /// The leftmost match at or after byte `from`, as byte offsets.
    fn find(self, s: &str, from: usize) -> Option<(usize, usize)> {
        let not_space = |c: char| !rx_space(c);
        match self {
            Self::Word => run(s, from, rx_word),
            Self::WordPunct => {
                let (i, c) = first(s, from, not_space)?;
                if rx_word(c) {
                    Some((i, run_end(s, i, rx_word)))
                } else {
                    Some((i, run_end(s, i, |c| !rx_word(c) && !rx_space(c))))
                }
            }
            Self::WordMoneyNonSpace => {
                let (i, c) = first(s, from, not_space)?;
                let money = |c: char| rx_digit(c) || c == '.';
                let end = if rx_word(c) {
                    run_end(s, i, rx_word)
                } else if c == '$' && s[i + 1..].chars().next().is_some_and(money) {
                    run_end(s, i + 1, money)
                } else {
                    run_end(s, i, not_space)
                };
                Some((i, end))
            }
            Self::WordApostrophe => run(s, from, |c| rx_word(c) || c == '\''),
            Self::NonSpace => run(s, from, not_space),
            Self::Space => run(s, from, rx_space),
            Self::AsciiLetters => run(s, from, |c| c.is_ascii_alphabetic()),
            Self::Blankline => blank_lines(s, from),
        }
    }
}

/// The first character at or after byte `from` that `f` accepts.
#[inline]
fn first(s: &str, from: usize, f: impl Fn(char) -> bool) -> Option<(usize, char)> {
    s[from..]
        .char_indices()
        .find(|&(_, c)| f(c))
        .map(|(k, c)| (from + k, c))
}

/// The end of the run of characters `f` accepts from byte `i`.
#[inline]
fn run_end(s: &str, i: usize, f: impl Fn(char) -> bool) -> usize {
    s[i..]
        .char_indices()
        .find(|&(_, c)| !f(c))
        .map_or(s.len(), |(k, _)| i + k)
}

/// The first non-empty run of characters `f` accepts, at or after `from`.
#[inline]
fn run(s: &str, from: usize, f: impl Fn(char) -> bool + Copy) -> Option<(usize, usize)> {
    let (i, _) = first(s, from, f)?;
    Some((i, run_end(s, i, f)))
}

/// `(?:(?<!\s)\s*)?\n\s*\n\s*`: a whole run of `\s` with two newlines in it.
///
/// With the optional group, a match can only start where a `\s` run starts
/// (its lookbehind) and its last `\s*` takes the rest of the run; without
/// it, it starts at a newline inside a run that already failed from its
/// start, with fewer newlines, so it fails too.
fn blank_lines(s: &str, from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    loop {
        let (start, _) = first(s, i, rx_space)?;
        let end = run_end(s, start, rx_space);
        if s[start..end].bytes().filter(|&b| b == b'\n').count() >= 2 {
            return Some((start, end));
        }
        i = end;
    }
}

/// Converts increasing byte offsets of one text into code point offsets.
struct CodePoints<'a> {
    text: &'a str,
    byte: usize,
    cp: usize,
}

impl<'a> CodePoints<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            byte: 0,
            cp: 0,
        }
    }

    /// The code point offset of byte offset `to` (not before the last one).
    fn at(&mut self, to: usize) -> usize {
        let skipped = &self.text.as_bytes()[self.byte..to];
        self.cp += skipped.iter().filter(|&&b| b & 0xc0 != 0x80).count();
        self.byte = to;
        self.cp
    }

    fn span(&mut self, (a, e): (usize, usize)) -> (usize, usize) {
        (self.at(a), self.at(e))
    }
}

/// `RegexpTokenizer(pattern, gaps, discard_empty)` for a native [`Pattern`].
///
/// ```
/// use pyroid_core::nltk::regexp::{Pattern, RegexpTokenizer, WORD_PUNCT};
/// let s = "Good muffins cost $3.88\nin New York.";
/// assert_eq!(
///     WORD_PUNCT.tokenize(s),
///     ["Good", "muffins", "cost", "$", "3", ".", "88", "in", "New", "York", "."]
/// );
/// let money = RegexpTokenizer::new(Pattern::WordMoneyNonSpace, false, true);
/// assert_eq!(money.tokenize(s), ["Good", "muffins", "cost", "$3.88", "in", "New", "York", "."]);
/// assert_eq!(money.span_tokenize("$3.88 é."), [(0, 5), (6, 7), (7, 8)]);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegexpTokenizer {
    pattern: Pattern,
    gaps: bool,
    discard_empty: bool,
}

/// `WordPunctTokenizer()`; its `tokenize` is `wordpunct_tokenize`.
pub const WORD_PUNCT: RegexpTokenizer = RegexpTokenizer::new(Pattern::WordPunct, false, true);
/// `WhitespaceTokenizer()`.
pub const WHITESPACE: RegexpTokenizer = RegexpTokenizer::new(Pattern::Space, true, true);
/// `BlanklineTokenizer()`; its `tokenize` is `blankline_tokenize`.
pub const BLANKLINE: RegexpTokenizer = RegexpTokenizer::new(Pattern::Blankline, true, true);

impl RegexpTokenizer {
    /// `RegexpTokenizer(pattern, gaps=gaps, discard_empty=discard_empty)`.
    pub const fn new(pattern: Pattern, gaps: bool, discard_empty: bool) -> Self {
        Self {
            pattern,
            gaps,
            discard_empty,
        }
    }

    /// `RegexpTokenizer(pattern, gaps, discard_empty, flags)` when the
    /// pattern and flags are native ([`Pattern::from_regex`]).
    pub fn from_regex(pattern: &str, gaps: bool, discard_empty: bool, flags: u32) -> Option<Self> {
        Some(Self::new(
            Pattern::from_regex(pattern, flags)?,
            gaps,
            discard_empty,
        ))
    }

    fn matches<'s>(&self, text: &'s str) -> impl Iterator<Item = (usize, usize)> + 's {
        let pattern = self.pattern;
        let mut from = 0;
        std::iter::from_fn(move || {
            let m = pattern.find(text, from)?;
            from = m.1;
            Some(m)
        })
    }

    /// `tokenize(text)`: the matches (`findall`), or with `gaps` the text
    /// between them (`split`, without the empty strings when
    /// `discard_empty`).
    pub fn tokenize<'s>(&self, text: &'s str) -> Vec<&'s str> {
        if !self.gaps {
            return self.matches(text).map(|(a, e)| &text[a..e]).collect();
        }
        let mut out = Vec::new();
        let mut left = 0;
        for (a, e) in self.matches(text) {
            out.push(&text[left..a]);
            left = e;
        }
        out.push(&text[left..]);
        if self.discard_empty {
            out.retain(|t| !t.is_empty());
        }
        out
    }

    /// `span_tokenize(text)`, in code points. With `gaps` this is
    /// `regexp_span_tokenize`, which drops the empty spans before matches
    /// but not the last one, then the `discard_empty` filter.
    pub fn span_tokenize(&self, text: &str) -> Vec<(usize, usize)> {
        let mut cps = CodePoints::new(text);
        if !self.gaps {
            return self.matches(text).map(|m| cps.span(m)).collect();
        }
        let mut out: Vec<(usize, usize)> = gap_spans(self.matches(text), text.len())
            .map(|m| cps.span(m))
            .collect();
        if self.discard_empty {
            out.retain(|&(a, e)| a != e);
        }
        out
    }
}

/// `regexp_span_tokenize`'s byte spans between the matches.
fn gap_spans(
    matches: impl Iterator<Item = (usize, usize)>,
    len: usize,
) -> impl Iterator<Item = (usize, usize)> {
    let mut left = 0;
    let mut matches = matches.fuse();
    let mut done = false;
    std::iter::from_fn(move || {
        for (right, next) in matches.by_ref() {
            let span = (left, right);
            left = next;
            if right != span.0 {
                return Some(span);
            }
        }
        if done {
            return None;
        }
        done = true;
        Some((left, len))
    })
}

/// `wordpunct_tokenize(text)`: `\w+|[^\w\s]+`.
///
/// ```
/// use pyroid_core::nltk::regexp::wordpunct_tokenize;
/// assert_eq!(wordpunct_tokenize("can't (2.5x)"), ["can", "'", "t", "(", "2", ".", "5x", ")"]);
/// ```
pub fn wordpunct_tokenize(text: &str) -> Vec<&str> {
    WORD_PUNCT.tokenize(text)
}

/// `blankline_tokenize(text)`: the text between blank lines.
///
/// ```
/// use pyroid_core::nltk::regexp::blankline_tokenize;
/// assert_eq!(blankline_tokenize("a\nb\n \t\nc\n"), ["a\nb", "c\n"]);
/// ```
pub fn blankline_tokenize(text: &str) -> Vec<&str> {
    BLANKLINE.tokenize(text)
}

/// `string_span_tokenize(s, sep)`, in code points, or `None` for an empty
/// `sep` (NLTK raises `ValueError("Token delimiter must not be empty")`).
///
/// Like NLTK it skips the span before a separator at index 0 only, so
/// other empty spans stay.
///
/// ```
/// use pyroid_core::nltk::regexp::string_span_tokenize;
/// assert_eq!(string_span_tokenize(" a  b ", " "), Some(vec![(1, 2), (3, 3), (4, 5)]));
/// ```
pub fn string_span_tokenize(s: &str, sep: &str) -> Option<Vec<(usize, usize)>> {
    if sep.is_empty() {
        return None;
    }
    let mut cps = CodePoints::new(s);
    let mut out = Vec::new();
    let mut left = 0;
    loop {
        let Some(k) = s[left..].find(sep) else {
            if left != s.len() {
                out.push(cps.span((left, s.len())));
            }
            return Some(out);
        };
        let right = left + k;
        if right != 0 {
            out.push(cps.span((left, right)));
        }
        left = right + sep.len();
    }
}

/// `SpaceTokenizer().tokenize(s)`: `s.split(" ")`.
pub fn space_tokenize(s: &str) -> Vec<&str> {
    s.split(' ').collect()
}

/// `SpaceTokenizer().span_tokenize(s)`.
pub fn space_span_tokenize(s: &str) -> Vec<(usize, usize)> {
    string_span_tokenize(s, " ").unwrap_or_default()
}

/// `TabTokenizer().tokenize(s)`: `s.split("\t")`.
pub fn tab_tokenize(s: &str) -> Vec<&str> {
    s.split('\t').collect()
}

/// `TabTokenizer().span_tokenize(s)`.
pub fn tab_span_tokenize(s: &str) -> Vec<(usize, usize)> {
    string_span_tokenize(s, "\t").unwrap_or_default()
}

/// `LineTokenizer(blanklines)`'s handling of blank lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Blanklines {
    /// `"discard"` (the default): drop lines that are only whitespace.
    Discard,
    /// `"keep"`.
    Keep,
    /// `"discard-eof"`: drop the last line when it is only whitespace.
    DiscardEof,
}

impl Blanklines {
    /// The mode NLTK calls `name` (it raises `ValueError` for others).
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "discard" => Self::Discard,
            "keep" => Self::Keep,
            "discard-eof" => Self::DiscardEof,
            _ => return None,
        })
    }
}

/// `s.splitlines()` in Python.
fn splitlines(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        let end = match c {
            '\r' if chars.peek().is_some_and(|&(_, n)| n == '\n') => {
                chars.next();
                i + 2
            }
            '\n' | '\r' | '\x0b' | '\x0c' | '\x1c' | '\x1d' | '\x1e' | '\u{85}' | '\u{2028}'
            | '\u{2029}' => i + c.len_utf8(),
            _ => continue,
        };
        out.push(&s[start..i]);
        start = end;
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
    out
}

/// `LineTokenizer(blanklines).tokenize(s)`: `s.splitlines()`, then the
/// blank-line rule.
///
/// ```
/// use pyroid_core::nltk::regexp::{Blanklines, line_tokenize};
/// let s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks.";
/// assert_eq!(line_tokenize(s, Blanklines::Keep).len(), 5);
/// assert_eq!(line_tokenize(s, Blanklines::Discard)[3], "Thanks.");
/// ```
pub fn line_tokenize(s: &str, blanklines: Blanklines) -> Vec<&str> {
    let mut lines = splitlines(s);
    match blanklines {
        Blanklines::Discard => lines.retain(|l| !l.trim_end_matches(py_space).is_empty()),
        Blanklines::DiscardEof => {
            if lines
                .last()
                .is_some_and(|l| l.trim_matches(py_space).is_empty())
            {
                lines.pop();
            }
        }
        Blanklines::Keep => {}
    }
    lines
}

/// `LineTokenizer(blanklines).span_tokenize(s)`, in code points.
///
/// NLTK's `"keep"` splits at the two characters `\n` (backslash, `n`):
/// it passes `r"\n"` to `string_span_tokenize`. The other modes split at
/// `regexp_span_tokenize(s, r"\n(\s+\n)*")` and keep its empty last span.
pub fn line_span_tokenize(s: &str, blanklines: Blanklines) -> Vec<(usize, usize)> {
    if blanklines == Blanklines::Keep {
        return string_span_tokenize(s, "\\n").unwrap_or_default();
    }
    let mut from = 0;
    let newlines = std::iter::from_fn(|| {
        let m = newline_run(s, from)?;
        from = m.1;
        Some(m)
    });
    let mut cps = CodePoints::new(s);
    gap_spans(newlines, s.len()).map(|m| cps.span(m)).collect()
}

/// `\n(\s+\n)*` at or after byte `from`: a newline, then, when the `\s` run
/// after it has a newline after its first character, up to the last such
/// newline (the greedy `\s+` takes it in one repetition).
fn newline_run(s: &str, from: usize) -> Option<(usize, usize)> {
    let i = from + s.as_bytes()[from..].iter().position(|&b| b == b'\n')?;
    let run = run_end(s, i + 1, rx_space);
    let last = s.as_bytes()[i + 1..run].iter().rposition(|&b| b == b'\n');
    Some(match last {
        Some(k) if k >= 1 => (i, i + 1 + k + 1),
        _ => (i, i + 1),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A text, its tokens and its spans.
    type Case = (
        &'static str,
        &'static [&'static str],
        &'static [(usize, usize)],
    );

    const MUFFINS: &str =
        "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks.";

    #[test]
    fn doctests() {
        assert_eq!(
            WHITESPACE.tokenize(MUFFINS),
            [
                "Good", "muffins", "cost", "$3.88", "in", "New", "York.", "Please", "buy", "me",
                "two", "of", "them.", "Thanks."
            ]
        );
        assert_eq!(
            wordpunct_tokenize(MUFFINS),
            [
                "Good", "muffins", "cost", "$", "3", ".", "88", "in", "New", "York", ".", "Please",
                "buy", "me", "two", "of", "them", ".", "Thanks", "."
            ]
        );
        assert_eq!(
            space_tokenize(MUFFINS),
            [
                "Good",
                "muffins",
                "cost",
                "$3.88\nin",
                "New",
                "York.",
                "",
                "Please",
                "buy",
                "me\ntwo",
                "of",
                "them.\n\nThanks."
            ]
        );
        assert_eq!(tab_tokenize("a\tb c\n\t d"), ["a", "b c\n", " d"]);
        assert_eq!(
            line_tokenize(MUFFINS, Blanklines::Keep),
            [
                "Good muffins cost $3.88",
                "in New York.  Please buy me",
                "two of them.",
                "",
                "Thanks."
            ]
        );
        assert_eq!(
            line_tokenize(MUFFINS, Blanklines::Discard),
            [
                "Good muffins cost $3.88",
                "in New York.  Please buy me",
                "two of them.",
                "Thanks."
            ]
        );
        // util.py doctests (the text has one newline fewer than MUFFINS).
        let s = "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\n\nThanks.";
        assert_eq!(
            string_span_tokenize(s, " ").unwrap(),
            [
                (0, 4),
                (5, 12),
                (13, 17),
                (18, 26),
                (27, 30),
                (31, 36),
                (37, 37),
                (38, 44),
                (45, 48),
                (49, 55),
                (56, 58),
                (59, 73)
            ]
        );
        assert_eq!(string_span_tokenize(s, ""), None);
    }

    #[test]
    fn regexp_doctests() {
        let s2 = "Alas, it has not rained today. When, do you think, will it rain again?";
        let words = RegexpTokenizer::new(Pattern::Word, false, true);
        assert_eq!(words.tokenize(s2)[..3], ["Alas", "it", "has"]);
        let s3 =
            "<p>Although this is <b>not</b> the case here, we must not relax our vigilance!</p>";
        assert_eq!(
            RegexpTokenizer::new(Pattern::AsciiLetters, true, true).tokenize("ab1cd"),
            ["1"]
        );
        assert_eq!(
            RegexpTokenizer::new(Pattern::NonSpace, false, true).tokenize(s3)[..2],
            ["<p>Although", "this"]
        );
        assert_eq!(
            RegexpTokenizer::new(Pattern::WordApostrophe, false, true).tokenize("can't 'x'"),
            ["can't", "'x'"]
        );
        let gaps = RegexpTokenizer::new(Pattern::Space, true, false);
        assert_eq!(gaps.tokenize(" a b "), ["", "a", "b", ""]);
        assert_eq!(gaps.span_tokenize(" a b "), [(1, 2), (3, 4), (5, 5)]);
        assert_eq!(
            RegexpTokenizer::new(Pattern::WordMoneyNonSpace, false, true)
                .tokenize("$abc a$1 $3.88, ,abc"),
            ["$abc", "a", "$1", "$3.88", ",", ",abc"]
        );
        assert_eq!(
            Pattern::from_regex(r"(?:(?<!\s)\s*)?\n\s*\n\s*", DEFAULT_FLAGS),
            Some(Pattern::Blankline)
        );
        assert_eq!(Pattern::from_regex(r"\s+", 0), Some(Pattern::Space));
        assert_eq!(Pattern::from_regex(r"\s+", 256), None);
    }

    #[test]
    fn blank_lines() {
        let cases: [Case; 8] = [
            ("a\n\nb", &["a", "b"], &[(0, 1), (3, 4)]),
            ("a \n \n b", &["a", "b"], &[(0, 1), (6, 7)]),
            ("\n\na", &["a"], &[(2, 3)]),
            ("a\n\x1c\nb", &["a\n\x1c\nb"], &[(0, 5)]),
            ("a\n\u{85}\n", &["a"], &[(0, 1)]),
            (
                "a  \n\nb\n\n\nc",
                &["a", "b", "c"],
                &[(0, 1), (5, 6), (9, 10)],
            ),
            ("\n\n", &[], &[]),
            ("a\nb", &["a\nb"], &[(0, 3)]),
        ];
        for (s, toks, spans) in cases {
            assert_eq!(BLANKLINE.tokenize(s), toks, "{s:?}");
            assert_eq!(BLANKLINE.span_tokenize(s), spans, "{s:?}");
        }
    }

    #[test]
    fn lines() {
        use Blanklines::{Discard, DiscardEof, Keep};
        type LineCase<'a> = (&'a str, Blanklines, &'a [&'a str], &'a [(usize, usize)]);
        let cases: [LineCase; 12] = [
            ("a\n\nb", Keep, &["a", "", "b"], &[(0, 4)]),
            ("a\n\nb", Discard, &["a", "b"], &[(0, 1), (3, 4)]),
            ("a\n\nb", DiscardEof, &["a", "", "b"], &[(0, 1), (3, 4)]),
            ("a\n  \n\nb", Discard, &["a", "b"], &[(0, 1), (6, 7)]),
            ("a\n \n", Discard, &["a"], &[(0, 1), (4, 4)]),
            ("\n\n", Keep, &["", ""], &[(0, 2)]),
            ("\n\n", Discard, &[], &[(2, 2)]),
            ("\n\n", DiscardEof, &[""], &[(2, 2)]),
            ("a\\nb", Keep, &["a\\nb"], &[(0, 1), (3, 4)]),
            ("\n a \n", DiscardEof, &["", " a "], &[(1, 4), (5, 5)]),
            ("", Keep, &[], &[]),
            ("", Discard, &[], &[(0, 0)]),
        ];
        for (s, mode, toks, spans) in cases {
            assert_eq!(line_tokenize(s, mode), toks, "{s:?} {mode:?}");
            assert_eq!(line_span_tokenize(s, mode), spans, "{s:?} {mode:?}");
        }
        assert_eq!(
            splitlines("a\r\nb\rc\u{2029}d\x1fe\n"),
            ["a", "b", "c", "d\x1fe"]
        );
        assert_eq!(Blanklines::from_name("discard-eof"), Some(DiscardEof));
        assert_eq!(Blanklines::from_name("x"), None);
    }

    #[test]
    fn space_tokenizer() {
        let cases: [Case; 6] = [
            ("", &[""], &[]),
            (" ", &["", ""], &[]),
            ("a  b", &["a", "", "b"], &[(0, 1), (2, 2), (3, 4)]),
            (" a", &["", "a"], &[(1, 2)]),
            ("a ", &["a", ""], &[(0, 1)]),
            ("  ", &["", "", ""], &[(1, 1)]),
        ];
        for (s, toks, spans) in cases {
            assert_eq!(space_tokenize(s), toks, "{s:?}");
            assert_eq!(space_span_tokenize(s), spans, "{s:?}");
        }
    }
}
