// Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
//! NLTK's Treebank word tokenizers and detokenizer (`nltk.tokenize.treebank`,
//! `nltk.tokenize.destructive`, `nltk.tokenize.util.align_tokens`).
//!
//! [`tokenize_text`] is `TreebankWordTokenizer.tokenize` or
//! `NLTKWordTokenizer.tokenize` (what `word_tokenize` runs on each sentence)
//! up to their final `text.split()`, which is [`tokens`]. [`span_tokenize`]
//! and [`align_tokens`] give NLTK's spans in code points; [`detokenize`] is
//! `TreebankWordDetokenizer.detokenize`.
//!
//! Each `re.sub` of the Python classes is a forward scan written for its
//! pattern that finds the same matches: leftmost, non-overlapping, the next
//! search starting where the previous match ended, lookarounds and the
//! character before a match read from the text the pass started with.
//! Passes are merged into one scan only where the result is the same: they
//! touch disjoint characters, and a pass that inserts spaces is merged only
//! with passes that ignore spaces (the `\s+` collapse of `NLTKWordTokenizer`
//! is merged by emitting each space through a collapsing writer).
//!
//! Character classes: NLTK compiles the patterns with the `regex` module
//! (`\w`, `\d` and `\s` are `chars::rx_word`, `rx_digit` and `rx_space`,
//! `(?i)` is `rx_ci`), except the tokenizers' `CONTRACTIONS2` and
//! `CONTRACTIONS3`, which use `re`: there `\b` follows `re_word` in the
//! given Unicode version, `(?i)` is `re_ci` (`i` also matches U+0130 and
//! U+0131) and `\s` is `str.isspace()`. `text.split()` and `text.strip()`
//! are Python's.

use std::fmt;

use memchr::{memchr, memchr2};

use super::UnicodeVersion;
use super::chars::{py_space, re_ci, re_word, rx_ci, rx_digit, rx_space, rx_word};
use crate::metrics::sacrebleu::pystr;

/// Which NLTK word tokenizer to reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Flavor {
    /// `nltk.tokenize.TreebankWordTokenizer`.
    Treebank,
    /// `nltk.tokenize.NLTKWordTokenizer`, the tokenizer `word_tokenize` runs
    /// on each sentence.
    Nltk,
}

/// Why [`span_tokenize`] or [`align_tokens`] failed, as NLTK fails.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AlignError {
    /// `align_tokens` raised `ValueError`: `token` is not in `sentence` after
    /// the previous token. The message (`Display`) is NLTK's.
    NotFound {
        /// The token that was searched for.
        token: String,
        /// The whole sentence.
        sentence: String,
    },
    /// `span_tokenize` met more quote tokens than quotes in the text: its
    /// `next()` raised `StopIteration` inside the generator, which Python
    /// reports as `RuntimeError("generator raised StopIteration")`.
    QuotesExhausted,
}

impl fmt::Display for AlignError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound { token, sentence } => {
                write!(f, "substring \"{token}\" not found in \"{sentence}\"")
            }
            Self::QuotesExhausted => f.write_str("generator raised StopIteration"),
        }
    }
}

impl std::error::Error for AlignError {}

/// The text after every substitution of `tokenize(text, convert_parentheses)`
/// (before its final `text.split()`); its tokens are [`tokens`]`(&result)`.
///
/// `v` is the CPython Unicode version, which decides `re`'s `\b` and so the
/// contractions (`cannot` -> `can not`) next to letters and digits that are
/// new in a version.
///
/// ```
/// use pyroid_core::nltk::UnicodeVersion;
/// use pyroid_core::nltk::treebank::{Flavor, tokenize_text, tokens};
/// let text = tokenize_text("They'll save $3.88 (I can't).", Flavor::Nltk, false, UnicodeVersion::V15_1);
/// let words: Vec<&str> = tokens(&text).collect();
/// assert_eq!(words, ["They", "'ll", "save", "$", "3.88", "(", "I", "ca", "n't", ")", "."]);
/// ```
pub fn tokenize_text(
    text: &str,
    flavor: Flavor,
    convert_parentheses: bool,
    v: UnicodeVersion,
) -> String {
    let mut ch = Chain::new(text);
    match flavor {
        Flavor::Treebank => {
            // STARTING_QUOTES
            ch.run(leading_double_quote);
            ch.run(|s, o| pad_literal(s, o, b"``"));
            ch.run(opening_quote);
            // PUNCTUATION
            ch.run(colon_comma);
            ch.run(colon_comma_end);
            ch.run(treebank_ellipsis_symbols);
            ch.run(|s, o| final_period(s, o, false));
            ch.run(|s, o| pad_either(s, o, b'?', b'!'));
            ch.run(quote_space);
            // PARENS_BRACKETS, CONVERT_PARENTHESES, DOUBLE_DASHES, the padding,
            // and ENDING_QUOTES[:2]
            ch.run(|s, o| treebank_late(s, o, convert_parentheses));
        }
        Flavor::Nltk => {
            // STARTING_QUOTES
            ch.run(nltk_opening_quotes);
            ch.run(opening_quote);
            ch.run(opening_single_quote);
            // PUNCTUATION
            ch.run(|s, o| final_period(s, o, true));
            ch.run(colon_comma);
            ch.run(colon_comma_end);
            ch.run(nltk_dots_symbols_dashes);
            ch.run(|s, o| final_period(s, o, false));
            ch.run(|s, o| pad_either(s, o, b'?', b'!'));
            ch.run(quote_space);
            // [*], PARENS_BRACKETS, CONVERT_PARENTHESES, DOUBLE_DASHES, the
            // padding, and ENDING_QUOTES[:4]
            ch.run(|s, o| nltk_late(s, o, convert_parentheses));
        }
    }
    // The last two ENDING_QUOTES, the same in both classes.
    ch.run(ending_s_m_d);
    ch.run(ending_ll_re_ve_nt);
    // CONTRACTIONS2: when none matches, every pass leaves the text unchanged.
    if CONTRACTIONS2
        .iter()
        .any(|&c| any_contraction(&ch.cur, c, v))
    {
        for c in CONTRACTIONS2 {
            ch.run(|s, o| contraction(s, o, c, v));
        }
    }
    // CONTRACTIONS3
    for word in [&b"is"[..], b"was"] {
        ch.run(|s, o| t_contraction(s, o, word, v));
    }
    ch.cur
}

/// `s.split()` in Python: the tokens of a [`tokenize_text`] result.
pub fn tokens(s: &str) -> impl Iterator<Item = &str> {
    pystr::split(s)
}

/// `span_tokenize(text)`: NLTK's `(start, end)` spans of the tokens, in code
/// points (as Python indexes `text`).
///
/// Like NLTK it tokenizes, puts the text's own `"`, ``` `` ``` and `''` back
/// in place of the quote tokens when the text has `"` or `''`, then aligns
/// the tokens with [`align_tokens`].
///
/// ```
/// use pyroid_core::nltk::UnicodeVersion;
/// use pyroid_core::nltk::treebank::{Flavor, span_tokenize};
/// let spans = span_tokenize("He said \"no\".", Flavor::Nltk, UnicodeVersion::V15_1).unwrap();
/// assert_eq!(spans, [(0, 2), (3, 7), (8, 9), (9, 11), (11, 12), (12, 13)]);
/// ```
pub fn span_tokenize(
    text: &str,
    flavor: Flavor,
    v: UnicodeVersion,
) -> Result<Vec<(usize, usize)>, AlignError> {
    let tokenized = tokenize_text(text, flavor, false, v);
    let raw = tokens(&tokenized);
    if memchr(b'"', text.as_bytes()).is_none() && !text.contains("''") {
        return align(raw.map(Ok), text);
    }
    let mut quotes = QuoteMatches { text, at: 0 };
    // NLTK builds the whole token list first, so running out of quotes wins
    // over a token that is not found.
    let toks = raw
        .map(|t| match t {
            "\"" | "``" | "''" => quotes.next().ok_or(AlignError::QuotesExhausted),
            _ => Ok(t),
        })
        .collect::<Result<Vec<&str>, AlignError>>()?;
    align(toks.into_iter().map(Ok), text)
}

/// `align_tokens(tokens, sentence)`: the code point span of each token, found
/// with `sentence.index(token, end of the previous span)`.
///
/// ```
/// use pyroid_core::nltk::treebank::align_tokens;
/// assert_eq!(align_tokens(&["ab", "cd"], "ab cd ef").unwrap(), [(0, 2), (3, 5)]);
/// let err = align_tokens(&["ab", "cd", "efg"], "ab cd ef").unwrap_err();
/// assert_eq!(err.to_string(), "substring \"efg\" not found in \"ab cd ef\"");
/// ```
pub fn align_tokens(tokens: &[&str], sentence: &str) -> Result<Vec<(usize, usize)>, AlignError> {
    align(tokens.iter().map(|&t| Ok(t)), sentence)
}

/// `TreebankWordDetokenizer().detokenize(tokens, convert_parentheses)`.
///
/// ```
/// use pyroid_core::nltk::treebank::detokenize;
/// let toks = ["hello", ",", "i", "ca", "n't", "feel", "my", "feet", "!", "Help", "!", "!"];
/// assert_eq!(detokenize(&toks, false), "hello, i can't feel my feet! Help!!");
/// assert_eq!(detokenize(&["-LRB-", "hello", "-RRB-"], true), "(hello)");
/// ```
pub fn detokenize(tokens: &[&str], convert_parentheses: bool) -> String {
    let len = tokens.iter().map(|t| t.len() + 1).sum::<usize>() + 1;
    let mut joined = String::with_capacity(len);
    // " " + " ".join(tokens) + " "
    joined.push(' ');
    for (i, t) in tokens.iter().enumerate() {
        if i > 0 {
            joined.push(' ');
        }
        joined.push_str(t);
    }
    joined.push(' ');
    let mut ch = Chain::from_string(joined);

    // CONTRACTIONS3, then CONTRACTIONS2 (`(?#X)` became `\s`).
    for word in [&b"is"[..], b"was"] {
        ch.run(|s, o| detok_t_contraction(s, o, word));
    }
    if CONTRACTIONS2
        .iter()
        .any(|&c| any_detok_contraction(&ch.cur, c))
    {
        for c in CONTRACTIONS2 {
            ch.run(|s, o| detok_contraction(s, o, c));
        }
    }
    // ENDING_QUOTES
    ch.run(detok_ending_ll_re_ve_nt);
    ch.run(detok_ending_s_m_d);
    ch.run(detok_space_double_quote);
    ch.run(detok_punct_spaces_quote);
    ch.run(detok_double_quote_space_punct);
    ch.run(|s, o| replace_literal(s, o, "''", "\""));
    ch.run(detok_swap_quotes);
    ch.strip();
    // DOUBLE_DASHES
    ch.run(|s, o| replace_literal(s, o, " -- ", "--"));
    if convert_parentheses {
        for (ptb, bracket) in PTB_BRACKETS {
            ch.run(|s, o| replace_literal(s, o, ptb, bracket));
        }
    }
    // PARENS_BRACKETS
    ch.run(|s, o| drop_space_after(s, o, &OPEN_BRACKETS));
    ch.run(|s, o| drop_space_before(s, o, &CLOSE_BRACKETS));
    ch.run(detok_close_space_punct);
    // PUNCTUATION
    ch.run(detok_single_quote);
    ch.run(|s, o| drop_space_before(s, o, &QUESTION_EXCLAMATION));
    ch.run(detok_space_period);
    ch.run(|s, o| drop_space_after(s, o, &HASH_DOLLAR));
    ch.run(|s, o| drop_space_before(s, o, &SEMICOLON_PERCENT));
    ch.run(detok_ellipsis);
    ch.run(|s, o| drop_space_before(s, o, &COLON_COMMA));
    // STARTING_QUOTES
    ch.run(detok_open_space_backticks);
    ch.run(detok_backticks_space);
    ch.run(|s, o| replace_literal(s, o, "``", "\""));
    ch.strip();
    ch.cur
}

// ---------------------------------------------------------------------------
// The machinery: a text passed from substitution to substitution, `re.sub`.

/// The text between substitutions, and a second buffer to write the next
/// one into (the two are swapped after a pass that changed something).
struct Chain {
    cur: String,
    tmp: String,
}

impl Chain {
    fn new(text: &str) -> Self {
        let mut cur = String::with_capacity(text.len() * 2 + 8);
        cur.push_str(text);
        Self::from_string(cur)
    }

    fn from_string(cur: String) -> Self {
        let tmp = String::with_capacity(cur.capacity());
        Self { cur, tmp }
    }

    /// Runs one pass: `pass` writes the new text into its (empty) second
    /// argument and returns `true`, or returns `false` when it would leave
    /// the text unchanged.
    #[inline]
    fn run(&mut self, pass: impl FnOnce(&str, &mut String) -> bool) {
        self.tmp.clear();
        if pass(&self.cur, &mut self.tmp) {
            std::mem::swap(&mut self.cur, &mut self.tmp);
        }
    }

    /// `text = text.strip()`.
    fn strip(&mut self) {
        let t = pystr::strip(&self.cur);
        if t.len() != self.cur.len() {
            self.tmp.clear();
            self.tmp.push_str(t);
            std::mem::swap(&mut self.cur, &mut self.tmp);
        }
    }
}

/// `pattern.sub(repl, src)` into `out`, or `false` when nothing matches.
///
/// `find(src, from)` returns the leftmost match starting at or after `from`
/// as `(start, end, groups)`; `emit` writes its replacement. No pattern here
/// matches the empty string, so every match ends after `from`.
#[inline]
fn sub<G>(
    src: &str,
    out: &mut String,
    find: impl Fn(&str, usize) -> Option<(usize, usize, G)>,
    emit: impl Fn(&mut String, &str, usize, usize, G),
) -> bool {
    let Some(mut m) = find(src, 0) else {
        return false;
    };
    let mut last = 0;
    loop {
        let (start, end, groups) = m;
        out.push_str(&src[last..start]);
        emit(out, src, start, end, groups);
        last = end;
        match find(src, end) {
            Some(next) => m = next,
            None => break,
        }
    }
    out.push_str(&src[last..]);
    true
}

/// A set of bytes, for ASCII character classes and scan triggers.
struct ByteSet([bool; 256]);

impl ByteSet {
    const fn new(bytes: &[u8]) -> Self {
        let mut set = [false; 256];
        let mut i = 0;
        while i < bytes.len() {
            set[bytes[i] as usize] = true;
            i += 1;
        }
        Self(set)
    }

    #[inline]
    fn has(&self, b: u8) -> bool {
        self.0[usize::from(b)]
    }

    /// The first index at or after `from` whose byte is in the set.
    #[inline]
    fn find(&self, b: &[u8], from: usize) -> Option<usize> {
        b.get(from..)?
            .iter()
            .position(|&c| self.has(c))
            .map(|k| from + k)
    }

    /// Whether the byte at `i` exists and is in the set.
    #[inline]
    fn at(&self, b: &[u8], i: usize) -> bool {
        b.get(i).is_some_and(|&c| self.has(c))
    }
}

const QUESTION_EXCLAMATION: ByteSet = ByteSet::new(b"?!");
const OPEN_BRACKETS: ByteSet = ByteSet::new(b"[({<");
const CLOSE_BRACKETS: ByteSet = ByteSet::new(b"])}>");
const HASH_DOLLAR: ByteSet = ByteSet::new(b"#$");
const SEMICOLON_PERCENT: ByteSet = ByteSet::new(b";%");
const COLON_COMMA: ByteSet = ByteSet::new(b":,");
/// `[ \(\[{<]` before an opening `"` or `''`.
const BEFORE_OPENING_QUOTE: ByteSet = ByteSet::new(b" ([{<");
/// `[\]\)}>"\']`, what may follow a final period (and the detokenizer's `.`).
const AFTER_PERIOD: ByteSet = ByteSet::new(b"])}>\"'");
/// `[,.;:!?']` before closing quotes in the detokenizer.
const BEFORE_CLOSING_QUOTE: ByteSet = ByteSet::new(b",.;:!?'");
/// `[.,:)\]>};%]` after `''` in the detokenizer.
const AFTER_CLOSING_QUOTE: ByteSet = ByteSet::new(b".,:)]>};%");
/// `[,.;:!?]` before `"'` in the detokenizer.
const BEFORE_SWAPPED_QUOTES: ByteSet = ByteSet::new(b",.;:!?");
/// `[ (\[{<]` before `` in the detokenizer.
const BEFORE_BACKTICKS: ByteSet = ByteSet::new(b" ([{<");
/// `[:;,.]` after a closing bracket in the detokenizer.
const AFTER_CLOSE_BRACKET: ByteSet = ByteSet::new(b":;,.");

/// CONVERT_PARENTHESES, in NLTK's order.
const PTB_BRACKETS: [(&str, &str); 6] = [
    ("-LRB-", "("),
    ("-RRB-", ")"),
    ("-LSB-", "["),
    ("-RSB-", "]"),
    ("-LCB-", "{"),
    ("-RCB-", "}"),
];

/// The PTB symbol of a bracket that CONVERT_PARENTHESES converts.
#[inline]
fn ptb_symbol(b: u8) -> Option<&'static str> {
    Some(match b {
        b'(' => "-LRB-",
        b')' => "-RRB-",
        b'[' => "-LSB-",
        b']' => "-RSB-",
        b'{' => "-LCB-",
        b'}' => "-RCB-",
        _ => return None,
    })
}

/// The character starting at byte `i`.
#[inline]
fn char_at(s: &str, i: usize) -> Option<char> {
    let &b = s.as_bytes().get(i)?;
    if b < 0x80 {
        Some(char::from(b))
    } else {
        s[i..].chars().next()
    }
}

/// The character ending at byte `i`.
#[inline]
fn char_before(s: &str, i: usize) -> Option<char> {
    let b = *s.as_bytes().get(i.checked_sub(1)?)?;
    if b < 0x80 {
        Some(char::from(b))
    } else {
        s[..i].chars().next_back()
    }
}

/// Byte length of the `regex` `\s` character starting at `i`, if it is one.
#[inline]
fn space_at(s: &str, i: usize) -> Option<usize> {
    char_at(s, i).filter(|&c| rx_space(c)).map(char::len_utf8)
}

/// Start of the `regex` `\s` character ending at `i`, if it is one.
#[inline]
fn space_before(s: &str, i: usize) -> Option<usize> {
    char_before(s, i)
        .filter(|&c| rx_space(c))
        .map(|c| i - c.len_utf8())
}

/// `regex`'s `\b` at byte `i`.
#[inline]
fn rx_boundary(s: &str, i: usize) -> bool {
    char_before(s, i).is_some_and(rx_word) != char_at(s, i).is_some_and(rx_word)
}

/// `re`'s `\b` at byte `i` in Unicode version `v`.
#[inline]
fn re_boundary(s: &str, i: usize, v: UnicodeVersion) -> bool {
    let word = |c: char| re_word(c, v);
    char_before(s, i).is_some_and(word) != char_at(s, i).is_some_and(word)
}

/// Matches `pat` (lowercase ASCII letters and `'`) at byte `i` under `(?i)`,
/// where `ci(c, letter)` says whether `c` matches `letter`; returns the end.
#[inline]
fn match_ci(s: &str, i: usize, pat: &[u8], ci: impl Fn(char, u8) -> bool) -> Option<usize> {
    let mut end = i;
    for &letter in pat {
        let &byte = s.as_bytes().get(end)?;
        if byte < 0x80 {
            let ok = if letter == b'\'' {
                byte == b'\''
            } else {
                byte.to_ascii_lowercase() == letter
            };
            if !ok {
                return None;
            }
            end += 1;
        } else {
            let c = s[end..].chars().next()?;
            if letter == b'\'' || !ci(c, letter) {
                return None;
            }
            end += c.len_utf8();
        }
    }
    Some(end)
}

/// The first occurrence of `lit` at or after `from`.
#[inline]
fn find_literal(b: &[u8], from: usize, lit: &[u8]) -> Option<usize> {
    let mut i = from;
    loop {
        i += memchr(lit[0], b.get(i..)?)?;
        if b[i..].starts_with(lit) {
            return Some(i);
        }
        i += 1;
    }
}

/// `re.sub(lit, " " + lit + " ", src)`.
fn pad_literal(src: &str, out: &mut String, lit: &[u8]) -> bool {
    sub(
        src,
        out,
        |s, from| find_literal(s.as_bytes(), from, lit).map(|i| (i, i + lit.len(), ())),
        |o, s, a, e, ()| {
            o.push(' ');
            o.push_str(&s[a..e]);
            o.push(' ');
        },
    )
}

/// `re.sub(lit, to, src)`.
fn replace_literal(src: &str, out: &mut String, lit: &str, to: &str) -> bool {
    sub(
        src,
        out,
        |s, from| find_literal(s.as_bytes(), from, lit.as_bytes()).map(|i| (i, i + lit.len(), ())),
        |o, _, _, _, ()| o.push_str(to),
    )
}

/// `re.sub(r"[xy]", r" \g<0> ", src)` for two ASCII characters.
fn pad_either(src: &str, out: &mut String, x: u8, y: u8) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let i = from + memchr2(x, y, s.as_bytes().get(from..)?)?;
            Some((i, i + 1, ()))
        },
        |o, s, a, e, ()| {
            o.push(' ');
            o.push_str(&s[a..e]);
            o.push(' ');
        },
    )
}

// ---------------------------------------------------------------------------
// The tokenizers' passes.

/// `^\"` -> ``` `` ```.
fn leading_double_quote(src: &str, out: &mut String) -> bool {
    let Some(rest) = src.strip_prefix('"') else {
        return false;
    };
    out.push_str("``");
    out.push_str(rest);
    true
}

/// `([ \(\[{<])(\"|\'{2})` -> ``\1 `` ``.
fn opening_quote(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from + 1;
            loop {
                q += memchr2(b'"', b'\'', b.get(q..)?)?;
                if BEFORE_OPENING_QUOTE.has(b[q - 1]) {
                    if b[q] == b'"' {
                        return Some((q - 1, q + 1, ()));
                    }
                    if b.get(q + 1) == Some(&b'\'') {
                        return Some((q - 1, q + 2, ()));
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, ()| {
            o.push_str(&s[a..=a]);
            o.push_str(" `` ");
        },
    )
}

/// `NLTKWordTokenizer.STARTING_QUOTES[:3]`: `([«“‘„]|[`]+)` -> ` \1 `, then
/// `^\"` -> ``` `` ```, then `(``)` -> ` \1 `, in one scan. The first pass
/// pads every backtick run, so the third pairs backticks inside a run only,
/// and the `` from `^\"` alone.
fn nltk_opening_quotes(src: &str, out: &mut String) -> bool {
    let b = src.as_bytes();
    let leading = b.first() == Some(&b'"');
    let trigger = |c: u8| c == b'`' || c == 0xc2 || c == 0xe2;
    if !leading && !b.iter().any(|&c| trigger(c)) {
        return false;
    }
    let mut last = 0;
    let mut i = 0;
    if leading {
        out.push_str(" `` ");
        last = 1;
        i = 1;
    }
    let mut changed = leading;
    while i < b.len() {
        let c = b[i];
        if c == b'`' {
            let run = b[i..].iter().take_while(|&&x| x == b'`').count();
            out.push_str(&src[last..i]);
            out.push(' ');
            for _ in 0..run / 2 {
                out.push_str(" `` ");
            }
            if run % 2 == 1 {
                out.push('`');
            }
            out.push(' ');
            i += run;
            last = i;
            changed = true;
        } else if c >= 0x80 {
            let ch = src[i..].chars().next().unwrap_or_default();
            let n = ch.len_utf8();
            if matches!(ch, '«' | '“' | '‘' | '„') {
                out.push_str(&src[last..i]);
                out.push(' ');
                out.push(ch);
                out.push(' ');
                last = i + n;
                changed = true;
            }
            i += n;
        } else {
            i += 1;
        }
    }
    out.push_str(&src[last..]);
    changed
}

/// `(?i)(?<!\w)(\')(?!(?:re|ve|ll|m|t|s|d|n)\b)(?=\w)` -> `\1 `: an opening
/// single quote before a word that is not a clitic.
fn opening_single_quote(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if !char_before(s, q).is_some_and(rx_word)
                    && char_at(s, q + 1).is_some_and(rx_word)
                    && !clitic_follows(s, q + 1)
                {
                    return Some((q, q + 1, ()));
                }
                q += 1;
            }
        },
        |o, _, _, _, ()| o.push_str("' "),
    )
}

/// `(?:re|ve|ll|m|t|s|d|n)\b` at byte `i`, with `regex`'s `(?i)` and `\b`.
fn clitic_follows(s: &str, i: usize) -> bool {
    [&b"re"[..], b"ve", b"ll", b"m", b"t", b"s", b"d", b"n"]
        .iter()
        .any(|alt| match_ci(s, i, alt, rx_ci).is_some_and(|e| rx_boundary(s, e)))
}

/// `([:,])([^\d])` -> ` \1 \2`.
fn colon_comma(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i += memchr2(b':', b',', b.get(i..)?)?;
                match char_at(s, i + 1) {
                    Some(c) if !rx_digit(c) => return Some((i, i + 1 + c.len_utf8(), ())),
                    Some(_) => i += 1,
                    None => return None,
                }
            }
        },
        |o, s, a, e, ()| {
            o.push(' ');
            o.push_str(&s[a..=a]);
            o.push(' ');
            o.push_str(&s[a + 1..e]);
        },
    )
}

/// `([:,])$` -> ` \1 `, where `$` is the end or before a final newline.
fn colon_comma_end(src: &str, out: &mut String) -> bool {
    let b = src.as_bytes();
    let n = b.len();
    let i = if n >= 1 && COLON_COMMA.has(b[n - 1]) {
        n - 1
    } else if n >= 2 && b[n - 1] == b'\n' && COLON_COMMA.has(b[n - 2]) {
        n - 2
    } else {
        return false;
    };
    out.push_str(&src[..i]);
    out.push(' ');
    out.push_str(&src[i..=i]);
    out.push(' ');
    out.push_str(&src[i + 1..]);
    true
}

/// `TreebankWordTokenizer`'s `\.\.\.` -> ` ... ` and `[;@#$%&]` -> ` \g<0> `
/// in one scan (they pad disjoint characters).
fn treebank_ellipsis_symbols(src: &str, out: &mut String) -> bool {
    const TRIGGERS: ByteSet = ByteSet::new(b".;@#$%&");
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i = TRIGGERS.find(b, i)?;
                if b[i] != b'.' {
                    return Some((i, i + 1, ()));
                }
                if b[i..].starts_with(b"...") {
                    return Some((i, i + 3, ()));
                }
                i += 1;
            }
        },
        |o, s, a, e, ()| {
            o.push(' ');
            o.push_str(&s[a..e]);
            o.push(' ');
        },
    )
}

/// `NLTKWordTokenizer`'s `\.{2,}`, `[;@#$%&]` and `[‒-―]`, each
/// -> ` \g<0> `, in one scan (they pad disjoint characters).
fn nltk_dots_symbols_dashes(src: &str, out: &mut String) -> bool {
    const TRIGGERS: ByteSet = ByteSet::new(b".;@#$%&\xe2");
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i = TRIGGERS.find(b, i)?;
                match b[i] {
                    b'.' => {
                        let run = b[i..].iter().take_while(|&&x| x == b'.').count();
                        if run >= 2 {
                            return Some((i, i + run, ()));
                        }
                        i += 1;
                    }
                    // U+2012..U+2015 are E2 80 92..95.
                    0xe2 => {
                        if b.get(i + 1) == Some(&0x80) && matches!(b.get(i + 2), Some(0x92..=0x95))
                        {
                            return Some((i, i + 3, ()));
                        }
                        i += 1;
                    }
                    _ => return Some((i, i + 1, ())),
                }
            }
        },
        |o, s, a, e, ()| {
            o.push(' ');
            o.push_str(&s[a..e]);
            o.push(' ');
        },
    )
}

/// The final period.
///
/// `possessive = false`: `([^\.])(\.)([\]\)}>"\']*)\s*$` -> `\1 \2\3 `.
/// `possessive = true` (`NLTKWordTokenizer.PUNCTUATION[0]`):
/// `([^\.])(\.)([\]\)}>"\'»”’ ]*+)\s*$` -> `\1 \2 \3 `.
///
/// A match runs to the end of the text (`\s*` takes the whole trailing run,
/// `$` included before a final newline), so after its `.` there are only
/// class characters and then `\s` characters: there is at most one match,
/// found from the end. The class run of the replacement is the longest one
/// after the `.` (for the possessive class it may take spaces of the
/// trailing run).
fn final_period(src: &str, out: &mut String, possessive: bool) -> bool {
    let class = |c: char| {
        matches!(c, ']' | ')' | '}' | '>' | '"' | '\'')
            || (possessive && matches!(c, '»' | '”' | '’' | ' '))
    };
    let b = src.as_bytes();
    let mut q = src.len();
    while let Some(c) = char_before(src, q).filter(|&c| rx_space(c)) {
        q -= c.len_utf8();
    }
    let mut p = q;
    while let Some(c) = char_before(src, p).filter(|&c| class(c)) {
        p -= c.len_utf8();
    }
    if p < 2 || b[p - 1] != b'.' || b[p - 2] == b'.' {
        return false;
    }
    let mut k = q;
    if possessive {
        while b.get(k) == Some(&b' ') {
            k += 1;
        }
    }
    out.push_str(&src[..p - 1]);
    out.push_str(if possessive { " . " } else { " ." });
    out.push_str(&src[p..k]);
    out.push(' ');
    true
}

/// `([^'])' ` -> `\1 ' `.
fn quote_space(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if b.get(q + 1) == Some(&b' ') {
                    if let Some(x) = char_before(s, q).filter(|&x| x != '\'') {
                        let xs = q - x.len_utf8();
                        if xs >= from {
                            return Some((xs, q + 2, q));
                        }
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, q| {
            o.push_str(&s[a..q]);
            o.push_str(" ' ");
        },
    )
}

/// `TreebankWordTokenizer` from PARENS_BRACKETS to `ENDING_QUOTES[:2]` in
/// one scan: `[\]\[\(\)\{\}\<\>]` -> ` \g<0> ` (then CONVERT_PARENTHESES),
/// `--` -> ` -- `, `" " + text + " "`, `''` -> ` '' `, `"` -> ` '' `. They
/// touch disjoint characters and only insert spaces (and PTB symbols, which
/// stay between spaces), so the passes see the same matches in one scan.
fn treebank_late(src: &str, out: &mut String, convert: bool) -> bool {
    const SPECIAL: ByteSet = ByteSet::new(b"[](){}<>-'\"");
    let b = src.as_bytes();
    out.push(' ');
    let mut last = 0;
    let mut i = 0;
    while let Some(k) = SPECIAL.find(b, i) {
        let c = b[k];
        let (n, repl): (usize, &str) = match c {
            b'-' if b.get(k + 1) == Some(&b'-') => (2, "--"),
            b'\'' if b.get(k + 1) == Some(&b'\'') => (2, "''"),
            b'"' => (1, "''"),
            b'-' | b'\'' => {
                i = k + 1;
                continue;
            }
            _ => (1, ptb_symbol(c).filter(|_| convert).unwrap_or(&src[k..=k])),
        };
        out.push_str(&src[last..k]);
        out.push(' ');
        out.push_str(repl);
        out.push(' ');
        i = k + n;
        last = i;
    }
    out.push_str(&src[last..]);
    out.push(' ');
    true
}

/// Appends one space unless `out` already ends with one: `\s+` -> `" "` for
/// text written through it.
#[inline]
fn collapsed_space(out: &mut String) {
    if !out.ends_with(' ') {
        out.push(' ');
    }
}

/// `NLTKWordTokenizer` from `PUNCTUATION[-1]` to `ENDING_QUOTES[:4]` in one
/// scan: `[*]` and `[\]\[\(\)\{\}\<\>]` -> ` \g<0> ` (then
/// CONVERT_PARENTHESES), `--` -> ` -- `, `" " + text + " "`, `([»”’])` ->
/// ` \1 `, `''` -> ` '' `, `"` -> ` '' `, and `\s+` -> `" "`. The first
/// seven touch disjoint characters and only insert spaces (and PTB
/// symbols, which stay between spaces); the collapse of each run of `\s`
/// into one space is done by writing every space, inserted or not, through
/// [`collapsed_space`] (the only `\s` left in the text is then `" "`).
fn nltk_late(src: &str, out: &mut String, convert: bool) -> bool {
    // ASCII \s, the specials, and the lead bytes of non-ASCII \s and »”’.
    const SPECIAL: ByteSet = ByteSet::new(b"*[](){}<>-'\" \t\n\x0b\x0c\r\xc2\xe1\xe2\xe3");
    let b = src.as_bytes();
    out.push(' ');
    let mut last = 0;
    let mut i = 0;
    while let Some(k) = SPECIAL.find(b, i) {
        let c = b[k];
        let (n, repl): (usize, Option<&str>) = match c {
            b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r' => (1, None),
            b'-' if b.get(k + 1) == Some(&b'-') => (2, Some("--")),
            b'\'' if b.get(k + 1) == Some(&b'\'') => (2, Some("''")),
            b'"' => (1, Some("''")),
            b'-' | b'\'' => {
                i = k + 1;
                continue;
            }
            0x80.. => {
                let ch = src[k..].chars().next().unwrap_or_default();
                let n = ch.len_utf8();
                if rx_space(ch) {
                    (n, None)
                } else if matches!(ch, '»' | '”' | '’') {
                    (n, Some(&src[k..k + n]))
                } else {
                    i = k + n;
                    continue;
                }
            }
            _ => (
                1,
                Some(ptb_symbol(c).filter(|_| convert).unwrap_or(&src[k..=k])),
            ),
        };
        out.push_str(&src[last..k]);
        collapsed_space(out);
        if let Some(r) = repl {
            out.push_str(r);
            out.push(' ');
        }
        i = k + n;
        last = i;
    }
    out.push_str(&src[last..]);
    collapsed_space(out);
    true
}

/// The end of the group `('[sS]|'[mM]|'[dD]|')` at the `'` at byte `q`, when
/// a space follows it.
#[inline]
fn s_m_d_group(b: &[u8], q: usize) -> Option<usize> {
    match b.get(q + 1)? {
        b's' | b'S' | b'm' | b'M' | b'd' | b'D' if b.get(q + 2) == Some(&b' ') => Some(q + 2),
        b' ' => Some(q + 1),
        _ => None,
    }
}

/// The group `('ll|'LL|'re|'RE|'ve|'VE|n't|N'T)` around the `'` at byte `q`,
/// as `(start, end)`, when a space follows it.
#[inline]
fn ll_re_ve_nt_group(b: &[u8], q: usize) -> Option<(usize, usize)> {
    let next = b.get(q + 1..q + 3)?;
    if matches!(next, b"ll" | b"LL" | b"re" | b"RE" | b"ve" | b"VE") && b.get(q + 3) == Some(&b' ')
    {
        return Some((q, q + 3));
    }
    let n = *b.get(q.checked_sub(1)?)?;
    if ((n == b'n' && next[0] == b't') || (n == b'N' && next[0] == b'T')) && next[1] == b' ' {
        return Some((q - 1, q + 2));
    }
    None
}

/// Start of the character ending at `i` when it is neither `'` nor `" "`
/// (`[^' ]`) and starts at or after `from`.
#[inline]
fn not_quote_space_before(s: &str, i: usize, from: usize) -> Option<usize> {
    let x = char_before(s, i).filter(|&x| x != '\'' && x != ' ')?;
    Some(i - x.len_utf8()).filter(|&xs| xs >= from)
}

/// `([^' ])('[sS]|'[mM]|'[dD]|') ` -> `\1 \2 `.
fn ending_s_m_d(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if let Some(ge) = s_m_d_group(b, q) {
                    if let Some(xs) = not_quote_space_before(s, q, from) {
                        return Some((xs, ge + 1, (q, ge)));
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, (gs, ge)| {
            o.push_str(&s[a..gs]);
            o.push(' ');
            o.push_str(&s[gs..ge]);
            o.push(' ');
        },
    )
}

/// `([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ` -> `\1 \2 `.
fn ending_ll_re_ve_nt(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if let Some((gs, ge)) = ll_re_ve_nt_group(b, q) {
                    if let Some(xs) = not_quote_space_before(s, gs, from) {
                        return Some((xs, ge + 1, (gs, ge)));
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, (gs, ge)| {
            o.push_str(&s[a..gs]);
            o.push(' ');
            o.push_str(&s[gs..ge]);
            o.push(' ');
        },
    )
}

/// `MacIntyreContractions.CONTRACTIONS2` as `(group 1, group 2, ends with
/// (?=\s) rather than \b)`: `(?i)\b(can)(?#X)(not)\b` and so on.
const CONTRACTIONS2: [(&[u8], &[u8], bool); 8] = [
    (b"can", b"not", false),
    (b"d", b"'ye", false),
    (b"gim", b"me", false),
    (b"gon", b"na", false),
    (b"got", b"ta", false),
    (b"lem", b"me", false),
    (b"more", b"'n", false),
    (b"wan", b"na", true),
];

type Contraction = (&'static [u8], &'static [u8], bool);

/// A tokenizer CONTRACTIONS2 match (compiled with `re`) at byte `i`: the end
/// of group 1 and of the match.
#[inline]
fn contraction_at(
    s: &str,
    i: usize,
    (one, two, ahead): Contraction,
    v: UnicodeVersion,
) -> Option<(usize, usize)> {
    // The letters first: they rule out most candidates.
    let mid = match_ci(s, i, one, re_ci)?;
    let end = match_ci(s, mid, two, re_ci)?;
    let ok = re_boundary(s, i, v)
        && if ahead {
            char_at(s, end).is_some_and(py_space)
        } else {
            re_boundary(s, end, v)
        };
    ok.then_some((mid, end))
}

/// The positions at or after `from` where a contraction may start: its
/// first letter in either case (only ASCII letters match them under `(?i)`).
#[inline]
fn first_letter(b: &[u8], from: usize, letter: u8) -> Option<usize> {
    Some(from + memchr2(letter, letter.to_ascii_uppercase(), b.get(from..)?)?)
}

/// Whether the CONTRACTIONS2 pattern `c` matches anywhere in `s`.
fn any_contraction(s: &str, c: Contraction, v: UnicodeVersion) -> bool {
    let mut from = 0;
    while let Some(k) = first_letter(s.as_bytes(), from, c.0[0]) {
        if contraction_at(s, k, c, v).is_some() {
            return true;
        }
        from = k + 1;
    }
    false
}

/// One CONTRACTIONS2 pattern, `-> " \1 \2 "`.
fn contraction(src: &str, out: &mut String, c: Contraction, v: UnicodeVersion) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i = first_letter(b, i, c.0[0])?;
                if let Some((mid, end)) = contraction_at(s, i, c, v) {
                    return Some((i, end, mid));
                }
                i += 1;
            }
        },
        |o, s, a, e, mid| {
            o.push(' ');
            o.push_str(&s[a..mid]);
            o.push(' ');
            o.push_str(&s[mid..e]);
            o.push(' ');
        },
    )
}

/// CONTRACTIONS3 (`re`): `(?i) ('t)(?#X)(is)\b` or `... (was)\b` ->
/// ` \1 \2 `.
fn t_contraction(src: &str, out: &mut String, word: &[u8], v: UnicodeVersion) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from + 1;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if b[q - 1] == b' ' {
                    if let Some(t) = match_ci(s, q, b"'t", re_ci) {
                        if let Some(end) = match_ci(s, t, word, re_ci) {
                            if re_boundary(s, end, v) {
                                return Some((q - 1, end, t));
                            }
                        }
                    }
                }
                q += 1;
            }
        },
        |o, s, a, e, t| {
            o.push(' ');
            o.push_str(&s[a + 1..t]);
            o.push(' ');
            o.push_str(&s[t..e]);
            o.push(' ');
        },
    )
}

// ---------------------------------------------------------------------------
// span_tokenize

/// `regex.finditer(r"``|'{2}|\"", text)`, the matched strings.
struct QuoteMatches<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> Iterator for QuoteMatches<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<&'a str> {
        let b = self.text.as_bytes();
        let mut i = self.at;
        loop {
            const QUOTES: ByteSet = ByteSet::new(b"`'\"");
            i = QUOTES.find(b, i)?;
            let n = match b[i] {
                b'"' => 1,
                c if b.get(i + 1) == Some(&c) => 2,
                _ => {
                    i += 1;
                    continue;
                }
            };
            self.at = i + n;
            return Some(&self.text[i..i + n]);
        }
    }
}

/// Number of code points in `s`.
#[inline]
fn code_points(s: &str) -> usize {
    s.bytes().filter(|&b| b & 0xc0 != 0x80).count()
}

/// `align_tokens` over tokens that may already have failed.
fn align<'t>(
    tokens: impl Iterator<Item = Result<&'t str, AlignError>>,
    sentence: &str,
) -> Result<Vec<(usize, usize)>, AlignError> {
    let mut spans = Vec::new();
    // `point` as a byte offset and as a code point offset.
    let (mut point, mut cp) = (0, 0);
    for token in tokens {
        let token = token?;
        let Some(k) = sentence[point..].find(token) else {
            return Err(AlignError::NotFound {
                token: token.to_owned(),
                sentence: sentence.to_owned(),
            });
        };
        let start = cp + code_points(&sentence[point..point + k]);
        cp = start + code_points(token);
        point += k + token.len();
        spans.push((start, cp));
    }
    Ok(spans)
}

// ---------------------------------------------------------------------------
// The detokenizer's passes (all compiled with `regex`).

/// A detokenizer CONTRACTIONS2 match at byte `i`: `(?i)\b(one)\s(two)\b`
/// (or `(?=\s)`), as the ends of group 1, of the `\s` and of the match.
#[inline]
fn detok_contraction_at(
    s: &str,
    i: usize,
    (one, two, ahead): Contraction,
) -> Option<(usize, usize, usize)> {
    let mid = match_ci(s, i, one, rx_ci)?;
    let two_start = mid + space_at(s, mid)?;
    let end = match_ci(s, two_start, two, rx_ci)?;
    let ok = rx_boundary(s, i)
        && if ahead {
            space_at(s, end).is_some()
        } else {
            rx_boundary(s, end)
        };
    ok.then_some((mid, two_start, end))
}

/// Whether the detokenizer's CONTRACTIONS2 pattern `c` matches anywhere.
fn any_detok_contraction(s: &str, c: Contraction) -> bool {
    let b = s.as_bytes();
    let mut i = 0;
    while let Some(k) = first_letter(b, i, c.0[0]) {
        if detok_contraction_at(s, k, c).is_some() {
            return true;
        }
        i = k + 1;
    }
    false
}

/// One detokenizer CONTRACTIONS2 pattern, `-> \1\2`.
fn detok_contraction(src: &str, out: &mut String, c: Contraction) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i = first_letter(b, i, c.0[0])?;
                if let Some((mid, two, end)) = detok_contraction_at(s, i, c) {
                    return Some((i, end, (mid, two)));
                }
                i += 1;
            }
        },
        |o, s, a, e, (mid, two)| {
            o.push_str(&s[a..mid]);
            o.push_str(&s[two..e]);
        },
    )
}

/// Detokenizer CONTRACTIONS3: `(?i) ('t)\s(is)\b` (or `was`) -> `\1\2`.
fn detok_t_contraction(src: &str, out: &mut String, word: &[u8]) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from + 1;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if b[q - 1] == b' ' {
                    if let Some(t) = match_ci(s, q, b"'t", rx_ci) {
                        if let Some(w) = space_at(s, t) {
                            if let Some(end) = match_ci(s, t + w, word, rx_ci) {
                                if rx_boundary(s, end) {
                                    return Some((q - 1, end, (t, t + w)));
                                }
                            }
                        }
                    }
                }
                q += 1;
            }
        },
        |o, s, a, e, (t, w)| {
            o.push_str(&s[a + 1..t]);
            o.push_str(&s[w..e]);
        },
    )
}

/// `([^' ])\s` before byte `i`: the start of the `[^' ]` character (at or
/// after `from`) and of the `\s` character.
#[inline]
fn x_space_before(s: &str, i: usize, from: usize) -> Option<(usize, usize)> {
    let ws = space_before(s, i)?;
    Some((not_quote_space_before(s, ws, from)?, ws))
}

/// `([^' ])\s('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) ` -> `\1\2 `.
fn detok_ending_ll_re_ve_nt(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if let Some((gs, ge)) = ll_re_ve_nt_group(b, q) {
                    if let Some((xs, ws)) = x_space_before(s, gs, from) {
                        return Some((xs, ge + 1, (ws, gs, ge)));
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, (ws, gs, ge)| {
            o.push_str(&s[a..ws]);
            o.push_str(&s[gs..ge]);
            o.push(' ');
        },
    )
}

/// `([^' ])\s('[sS]|'[mM]|'[dD]|') ` -> `\1\2 `.
fn detok_ending_s_m_d(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if let Some(ge) = s_m_d_group(b, q) {
                    if let Some((xs, ws)) = x_space_before(s, q, from) {
                        return Some((xs, ge + 1, (ws, q, ge)));
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, (ws, gs, ge)| {
            o.push_str(&s[a..ws]);
            o.push_str(&s[gs..ge]);
            o.push(' ');
        },
    )
}

/// The positions of `''` at or after `from`, overlapping ones included.
#[inline]
fn find_two_quotes(b: &[u8], from: usize) -> Option<usize> {
    let mut q = from;
    loop {
        q += memchr(b'\'', b.get(q..)?)?;
        if b.get(q + 1) == Some(&b'\'') {
            return Some(q);
        }
        q += 1;
    }
}

/// `([^'\s])\s(\'\')` -> `\1\2`.
fn detok_space_double_quote(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q = find_two_quotes(b, q)?;
                if let Some(ws) = space_before(s, q) {
                    if let Some(x) = char_before(s, ws).filter(|&x| x != '\'' && !rx_space(x)) {
                        let xs = ws - x.len_utf8();
                        if xs >= from {
                            return Some((xs, q + 2, ws));
                        }
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, ws| {
            o.push_str(&s[a..ws]);
            o.push_str("''");
        },
    )
}

/// `([,.;:!?'])\s+(\"|\'\')` -> `\1\2`. `\s+` can only take the whole run
/// of spaces after the punctuation (a quote must follow it).
fn detok_punct_spaces_quote(src: &str, out: &mut String) -> bool {
    if memchr2(b'"', b'\'', src.as_bytes()).is_none() {
        return false;
    }
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i = BEFORE_CLOSING_QUOTE.find(b, i)?;
                let mut k = i + 1;
                while let Some(n) = space_at(s, k) {
                    k += n;
                }
                if k > i + 1 {
                    if b.get(k) == Some(&b'"') {
                        return Some((i, k + 1, k));
                    }
                    if b.get(k..k + 2) == Some(b"''") {
                        return Some((i, k + 2, k));
                    }
                }
                i += 1;
            }
        },
        |o, s, a, e, k| {
            o.push_str(&s[a..=a]);
            o.push_str(&s[k..e]);
        },
    )
}

/// `(\'\')\s([.,:)\]>};%])` -> `\1\2`.
fn detok_double_quote_space_punct(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q = find_two_quotes(b, q)?;
                if let Some(w) = space_at(s, q + 2) {
                    if AFTER_CLOSING_QUOTE.at(b, q + 2 + w) {
                        return Some((q, q + 3 + w, ()));
                    }
                }
                q += 1;
            }
        },
        |o, s, _, e, ()| {
            o.push_str("''");
            o.push_str(&s[e - 1..e]);
        },
    )
}

/// `([,.;:!?])"(\')` -> `\1\2"`.
fn detok_swap_quotes(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from + 1;
            loop {
                q += memchr(b'"', b.get(q..)?)?;
                if BEFORE_SWAPPED_QUOTES.has(b[q - 1]) && b.get(q + 1) == Some(&b'\'') {
                    return Some((q - 1, q + 2, ()));
                }
                q += 1;
            }
        },
        |o, s, a, _, ()| {
            o.push_str(&s[a..=a]);
            o.push_str("'\"");
        },
    )
}

/// `([set])\s` -> `\1`.
fn drop_space_after(src: &str, out: &mut String, set: &ByteSet) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i = set.find(b, i)?;
                if let Some(w) = space_at(s, i + 1) {
                    return Some((i, i + 1 + w, ()));
                }
                i += 1;
            }
        },
        |o, s, a, _, ()| o.push_str(&s[a..=a]),
    )
}

/// `\s([set])` -> `\1`.
fn drop_space_before(src: &str, out: &mut String, set: &ByteSet) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i = set.find(b, i)?;
                if let Some(ws) = space_before(s, i).filter(|&ws| ws >= from) {
                    return Some((ws, i + 1, ()));
                }
                i += 1;
            }
        },
        |o, s, _, e, ()| o.push_str(&s[e - 1..e]),
    )
}

/// `([\]\)\}\>])\s([:;,.])` -> `\1\2`.
fn detok_close_space_punct(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut i = from;
            loop {
                i = CLOSE_BRACKETS.find(b, i)?;
                if let Some(w) = space_at(s, i + 1) {
                    if AFTER_CLOSE_BRACKET.at(b, i + 1 + w) {
                        return Some((i, i + 2 + w, ()));
                    }
                }
                i += 1;
            }
        },
        |o, s, a, e, ()| {
            o.push_str(&s[a..=a]);
            o.push_str(&s[e - 1..e]);
        },
    )
}

/// `([^'])\s'\s` -> `\1' `.
fn detok_single_quote(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q += memchr(b'\'', b.get(q..)?)?;
                if let (Some(ws), Some(w)) = (space_before(s, q), space_at(s, q + 1)) {
                    if let Some(x) = char_before(s, ws).filter(|&x| x != '\'') {
                        let xs = ws - x.len_utf8();
                        if xs >= from {
                            return Some((xs, q + 1 + w, ws));
                        }
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, ws| {
            o.push_str(&s[a..ws]);
            o.push_str("' ");
        },
    )
}

/// `([^\.])\s(\.)(?!\.)([\]\)}>"\']*)` -> `\1\2\3`.
fn detok_space_period(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q += memchr(b'.', b.get(q..)?)?;
                if b.get(q + 1) != Some(&b'.') {
                    if let Some(ws) = space_before(s, q) {
                        if let Some(x) = char_before(s, ws).filter(|&x| x != '.') {
                            let xs = ws - x.len_utf8();
                            if xs >= from {
                                let run = b[q + 1..]
                                    .iter()
                                    .take_while(|&&c| AFTER_PERIOD.has(c))
                                    .count();
                                return Some((xs, q + 1 + run, (ws, q)));
                            }
                        }
                    }
                }
                q += 1;
            }
        },
        |o, s, a, e, (ws, q)| {
            o.push_str(&s[a..ws]);
            o.push_str(&s[q..e]);
        },
    )
}

/// `\s\.\.\.\s` -> `...`.
fn detok_ellipsis(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q = find_literal(b, q, b"...")?;
                if let Some(ws) = space_before(s, q).filter(|&ws| ws >= from) {
                    if let Some(w) = space_at(s, q + 3) {
                        return Some((ws, q + 3 + w, ()));
                    }
                }
                q += 1;
            }
        },
        |o, _, _, _, ()| o.push_str("..."),
    )
}

/// `([ (\[{<])\s``` -> ``\1`` ``.
fn detok_open_space_backticks(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q = find_literal(b, q, b"``")?;
                if let Some(ws) = space_before(s, q) {
                    if ws > from && BEFORE_BACKTICKS.has(b[ws - 1]) {
                        return Some((ws - 1, q + 2, ()));
                    }
                }
                q += 1;
            }
        },
        |o, s, a, _, ()| {
            o.push_str(&s[a..=a]);
            o.push_str("``");
        },
    )
}

/// `(``)\s` -> `\1`.
fn detok_backticks_space(src: &str, out: &mut String) -> bool {
    sub(
        src,
        out,
        |s, from| {
            let b = s.as_bytes();
            let mut q = from;
            loop {
                q = find_literal(b, q, b"``")?;
                if let Some(w) = space_at(s, q + 2) {
                    return Some((q, q + 2 + w, ()));
                }
                q += 1;
            }
        },
        |o, _, _, _, ()| o.push_str("``"),
    )
}

#[cfg(test)]
#[path = "testdata/treebank/replay.rs"]
mod replay;

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    fn tb(s: &str) -> Vec<String> {
        toks(s, Flavor::Treebank, false)
    }

    fn nl(s: &str) -> Vec<String> {
        toks(s, Flavor::Nltk, false)
    }

    fn toks(s: &str, flavor: Flavor, convert: bool) -> Vec<String> {
        tokens(&tokenize_text(s, flavor, convert, V))
            .map(str::to_owned)
            .collect()
    }

    fn detok(t: &[&str]) -> String {
        detokenize(t, false)
    }

    fn spans(s: &str, flavor: Flavor) -> Vec<(usize, usize)> {
        span_tokenize(s, flavor, V).unwrap()
    }

    /// The code point slices of `s` at `spans`.
    fn slices(s: &str, spans: &[(usize, usize)]) -> Vec<String> {
        let c: Vec<char> = s.chars().collect();
        spans
            .iter()
            .map(|&(a, e)| c[a..e].iter().collect())
            .collect()
    }

    const MUFFINS: &str =
        "Good muffins cost $3.88\nin New York.  Please buy me\ntwo of them.\nThanks.";
    const MUFFINS_PARENS: &str =
        "Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).";
    const MUFFINS_SPANS: [(usize, usize); 23] = [
        (0, 4),
        (5, 12),
        (13, 17),
        (18, 19),
        (19, 23),
        (24, 26),
        (27, 30),
        (31, 32),
        (32, 36),
        (36, 37),
        (37, 38),
        (40, 46),
        (47, 48),
        (48, 51),
        (51, 52),
        (53, 55),
        (56, 59),
        (60, 62),
        (63, 68),
        (69, 70),
        (70, 76),
        (76, 77),
        (77, 78),
    ];
    const MUFFINS_PARENS_TOKENS: [&str; 23] = [
        "Good", "muffins", "cost", "$", "3.88", "in", "New", "(", "York", ")", ".", "Please", "(",
        "buy", ")", "me", "two", "of", "them.", "(", "Thanks", ")", ".",
    ];

    #[test]
    fn treebank_doctests() {
        assert_eq!(
            tb(MUFFINS),
            [
                "Good", "muffins", "cost", "$", "3.88", "in", "New", "York.", "Please", "buy",
                "me", "two", "of", "them.", "Thanks", "."
            ]
        );
        assert_eq!(
            tb("They'll save and invest more."),
            ["They", "'ll", "save", "and", "invest", "more", "."]
        );
        assert_eq!(
            tb("hi, my name can't hello,"),
            ["hi", ",", "my", "name", "ca", "n't", "hello", ","]
        );
        let s = "Good muffins cost $3.88 (roughly 3,36 euros)\nin New York.  Please buy me\ntwo of them.\nThanks.";
        let want = [
            "Good", "muffins", "cost", "$", "3.88", "(", "roughly", "3,36", "euros", ")", "in",
            "New", "York.", "Please", "buy", "me", "two", "of", "them.", "Thanks", ".",
        ];
        for flavor in [Flavor::Treebank, Flavor::Nltk] {
            assert_eq!(toks(s, flavor, false), want);
            let mut ptb = want;
            ptb[5] = "-LRB-";
            ptb[9] = "-RRB-";
            assert_eq!(toks(s, flavor, true), ptb);
            assert_eq!(spans(MUFFINS_PARENS, flavor), MUFFINS_SPANS);
            assert_eq!(
                slices(MUFFINS_PARENS, &spans(MUFFINS_PARENS, flavor)),
                MUFFINS_PARENS_TOKENS
            );
        }
    }

    #[test]
    fn treebank_span_tokenizer() {
        // test_tokenize.py: test_treebank_span_tokenizer
        let s = "The DUP is similar to the \"religious right\" in the United States and takes a hardline stance on social issues";
        let want = [
            (0, 3),
            (4, 7),
            (8, 10),
            (11, 18),
            (19, 21),
            (22, 25),
            (26, 27),
            (27, 36),
            (37, 42),
            (42, 43),
            (44, 46),
            (47, 50),
            (51, 57),
            (58, 64),
            (65, 68),
            (69, 74),
            (75, 76),
            (77, 85),
            (86, 92),
            (93, 95),
            (96, 102),
            (103, 109),
        ];
        assert_eq!(spans(s, Flavor::Treebank), want);
        let s = "The DUP is similar to the \"religious right\" in the United States and takes a ``hardline'' stance on social issues";
        let want = [
            (0, 3),
            (4, 7),
            (8, 10),
            (11, 18),
            (19, 21),
            (22, 25),
            (26, 27),
            (27, 36),
            (37, 42),
            (42, 43),
            (44, 46),
            (47, 50),
            (51, 57),
            (58, 64),
            (65, 68),
            (69, 74),
            (75, 76),
            (77, 79),
            (79, 87),
            (87, 89),
            (90, 96),
            (97, 99),
            (100, 106),
            (107, 113),
        ];
        assert_eq!(spans(s, Flavor::Treebank), want);
    }

    #[test]
    fn nltk_spans_with_quotes() {
        // tokenize.doctest
        let s = "I said, \"I'd like to buy some ''good muffins\" which cost $3.88\n each in New (York).\"";
        let want = [
            (0, 1),
            (2, 6),
            (6, 7),
            (8, 9),
            (9, 10),
            (10, 12),
            (13, 17),
            (18, 20),
            (21, 24),
            (25, 29),
            (30, 32),
            (32, 36),
            (37, 44),
            (44, 45),
            (46, 51),
            (52, 56),
            (57, 58),
            (58, 62),
            (64, 68),
            (69, 71),
            (72, 75),
            (76, 77),
            (77, 81),
            (81, 82),
            (82, 83),
            (83, 84),
        ];
        let got = spans(s, Flavor::Nltk);
        assert_eq!(got, want);
        assert_eq!(
            slices(s, &got),
            [
                "I", "said", ",", "\"", "I", "'d", "like", "to", "buy", "some", "''", "good",
                "muffins", "\"", "which", "cost", "$", "3.88", "each", "in", "New", "(", "York",
                ")", ".", "\""
            ]
        );
    }

    /// tokenize.doctest: `word_tokenize` of one-sentence texts.
    const WORD_TOKENIZE_DOCTESTS: [(&str, &[&str]); 13] = [
        (
            "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.",
            &[
                "On", "a", "$", "50,000", "mortgage", "of", "30", "years", "at", "8", "percent",
                ",", "the", "monthly", "payment", "would", "be", "$", "366.88", ".",
            ],
        ),
        (
            "\"We beat some pretty good teams to get here,\" Slocum said.",
            &[
                "``", "We", "beat", "some", "pretty", "good", "teams", "to", "get", "here", ",",
                "''", "Slocum", "said", ".",
            ],
        ),
        (
            "Well, we couldn't have this predictable, cliche-ridden, \"Touched by an Angel\" (a show creator John Masius worked on) wanna-be if she didn't.",
            &[
                "Well",
                ",",
                "we",
                "could",
                "n't",
                "have",
                "this",
                "predictable",
                ",",
                "cliche-ridden",
                ",",
                "``",
                "Touched",
                "by",
                "an",
                "Angel",
                "''",
                "(",
                "a",
                "show",
                "creator",
                "John",
                "Masius",
                "worked",
                "on",
                ")",
                "wanna-be",
                "if",
                "she",
                "did",
                "n't",
                ".",
            ],
        ),
        (
            "I cannot cannot work under these conditions!",
            &[
                "I",
                "can",
                "not",
                "can",
                "not",
                "work",
                "under",
                "these",
                "conditions",
                "!",
            ],
        ),
        (
            "The company spent $30,000,000 last year.",
            &[
                "The",
                "company",
                "spent",
                "$",
                "30,000,000",
                "last",
                "year",
                ".",
            ],
        ),
        (
            "The company spent 40.75% of its income last year.",
            &[
                "The", "company", "spent", "40.75", "%", "of", "its", "income", "last", "year", ".",
            ],
        ),
        (
            "He arrived at 3:00 pm.",
            &["He", "arrived", "at", "3:00", "pm", "."],
        ),
        (
            "I bought these items: books, pencils, and pens.",
            &[
                "I", "bought", "these", "items", ":", "books", ",", "pencils", ",", "and", "pens",
                ".",
            ],
        ),
        (
            "Though there were 150, 100 of them were old.",
            &[
                "Though", "there", "were", "150", ",", "100", "of", "them", "were", "old", ".",
            ],
        ),
        (
            "There were 300,000, but that wasn't enough.",
            &[
                "There", "were", "300,000", ",", "but", "that", "was", "n't", "enough", ".",
            ],
        ),
        (
            "It's more'n enough.",
            &["It", "'s", "more", "'n", "enough", "."],
        ),
        (
            "\u{ab}Now that I can do.\u{bb}",
            &["\u{ab}", "Now", "that", "I", "can", "do", ".", "\u{bb}"],
        ),
        (
            "The unicode 201C and 201D \u{201c}LEFT(RIGHT) DOUBLE QUOTATION MARK\u{201d} is also OPEN_PUNCT and CLOSE_PUNCT.",
            &[
                "The",
                "unicode",
                "201C",
                "and",
                "201D",
                "\u{201c}",
                "LEFT",
                "(",
                "RIGHT",
                ")",
                "DOUBLE",
                "QUOTATION",
                "MARK",
                "\u{201d}",
                "is",
                "also",
                "OPEN_PUNCT",
                "and",
                "CLOSE_PUNCT",
                ".",
            ],
        ),
    ];

    #[test]
    fn word_tokenize_doctests() {
        for (s, want) in WORD_TOKENIZE_DOCTESTS {
            assert_eq!(nl(s), want, "{s:?}");
        }
    }

    #[test]
    fn word_tokenize_unit_tests() {
        let cases: [(&str, &[&str]); 12] = [
            (
                "This is a, *weird sentence with *asterisks in it.",
                &[
                    "This",
                    "is",
                    "a",
                    ",",
                    "*",
                    "weird",
                    "sentence",
                    "with",
                    "*",
                    "asterisks",
                    "in",
                    "it",
                    ".",
                ],
            ),
            (
                "The food\u{2014}which was delicious\u{2014}reminded me of home.",
                &[
                    "The",
                    "food",
                    "\u{2014}",
                    "which",
                    "was",
                    "delicious",
                    "\u{2014}",
                    "reminded",
                    "me",
                    "of",
                    "home",
                    ".",
                ],
            ),
            (
                "The score was 3\u{2013}1.",
                &["The", "score", "was", "3", "\u{2013}", "1", "."],
            ),
            (
                "He said \u{2015} wait for it \u{2015} a new thing.",
                &[
                    "He", "said", "\u{2015}", "wait", "for", "it", "\u{2015}", "a", "new", "thing",
                    ".",
                ],
            ),
            (
                "The number is 12\u{2012}34.",
                &["The", "number", "is", "12", "\u{2012}", "34", "."],
            ),
            ("state-of-the-art", &["state-of-the-art"]),
            (
                "The 'v', I've been fooled but I'll seek revenge.",
                &[
                    "The", "'", "v", "'", ",", "I", "'ve", "been", "fooled", "but", "I", "'ll",
                    "seek", "revenge", ".",
                ],
            ),
            ("'v' 're'", &["'", "v", "'", "'re", "'"]),
            ("'Hard' to tell", &["'", "Hard", "'", "to", "tell"]),
            (
                "He said 'hello' to me",
                &["He", "said", "'", "hello", "'", "to", "me"],
            ),
            ("It's o'clock already", &["It", "'s", "o'clock", "already"]),
            ("O'Connor went home", &["O'Connor", "went", "home"]),
        ];
        for (s, want) in cases {
            assert_eq!(nl(s), want, "{s:?}");
        }
        // test_pad_dotdot, one sentence at a time (word_tokenize splits it).
        assert_eq!(
            nl("Why did dotdot.. not get tokenized but dotdotdot... did?"),
            [
                "Why",
                "did",
                "dotdot",
                "..",
                "not",
                "get",
                "tokenized",
                "but",
                "dotdotdot",
                "...",
                "did",
                "?"
            ]
        );
        assert_eq!(
            nl("How about manydots....."),
            ["How", "about", "manydots", "....."]
        );
        // word_tokenize(s11), one sentence at a time.
        assert_eq!(
            nl("I called Dr. Jones."),
            ["I", "called", "Dr.", "Jones", "."]
        );
    }

    #[test]
    fn detokenizer_doctests() {
        let toks = tb(MUFFINS);
        let toks: Vec<&str> = toks.iter().map(String::as_str).collect();
        assert_eq!(
            detok(&toks),
            "Good muffins cost $3.88 in New York. Please buy me two of them. Thanks."
        );
        let ptb = toks_owned(MUFFINS_PARENS, Flavor::Treebank, true);
        let ptb: Vec<&str> = ptb.iter().map(String::as_str).collect();
        let mut want = MUFFINS_PARENS_TOKENS;
        for t in &mut want {
            match *t {
                "(" => *t = "-LRB-",
                ")" => *t = "-RRB-",
                _ => {}
            }
        }
        assert_eq!(ptb, want);
        assert_eq!(
            detokenize(&ptb, true),
            "Good muffins cost $3.88 in New (York). Please (buy) me two of them. (Thanks)."
        );
        assert_eq!(
            detok(&[
                "hello", ",", "i", "ca", "n't", "feel", "my", "feet", "!", "Help", "!", "!"
            ]),
            "hello, i can't feel my feet! Help!!"
        );
        assert_eq!(
            detok(&[
                "hello", ",", "i", "can't", "feel", ";", "my", "feet", "!", "Help", "!", "!", "He",
                "said", ":", "Help", ",", "help", "?", "!"
            ]),
            "hello, i can't feel; my feet! Help!! He said: Help, help?!"
        );
        // tokenize.doctest: detokenize(word_tokenize(s)) == s
        for s in [
            "On a $50,000 mortgage of 30 years at 8 percent, the monthly payment would be $366.88.",
            "\"We beat some pretty good teams to get here,\" Slocum said.",
            "Well, we couldn't have this predictable, cliche-ridden, \"Touched by an Angel\" (a show creator John Masius worked on) wanna-be if she didn't.",
            "I cannot cannot work under these conditions!",
            "The company spent $30,000,000 last year.",
            "The company spent 40.75% of its income last year.",
            "He arrived at 3:00 pm.",
            "I bought these items: books, pencils, and pens.",
            "Though there were 150, 100 of them were old.",
            "There were 300,000, but that wasn't enough.",
            "How \"are\" you?",
            "Hello (world)",
            "<A sentence> with (many) [kinds] of {parentheses}. \"Sometimes it's inside (quotes)\". (\"Sometimes the otherway around\").",
            "Sentence ending with (parentheses)",
            "(Sentence) starting with parentheses.",
            "I've",
            "Don't",
            "I'd",
        ] {
            let t = nl(s);
            let t: Vec<&str> = t.iter().map(String::as_str).collect();
            assert_eq!(detok(&t), s);
        }
    }

    fn toks_owned(s: &str, flavor: Flavor, convert: bool) -> Vec<String> {
        toks(s, flavor, convert)
    }

    #[test]
    fn detokenizer_unit_tests() {
        let cases: [(&[&str], &str); 31] = [
            (&["Hello", ",", "world", "."], "Hello, world."),
            (&["I", "'m", "sure", "."], "I'm sure."),
            (&["You", "'ll", "see", "."], "You'll see."),
            (&["I", "do", "n't", "know", "."], "I don't know."),
            (&["``", "Hello", "''"], "\"Hello\""),
            (&["He", "said", "``", "hi", "''", "."], "He said \"hi\"."),
            (
                &["``", "Yes", ",", "''", "he", "said", "."],
                "\"Yes,\" he said.",
            ),
            (&["The", "dog", "'s", "bone", "."], "The dog's bone."),
            (&["foo", "--", "bar"], "foo--bar"),
            (&["word", ",", "'", "''"], "word,'\""),
            (&["Hello", ",", "world", "."], "Hello, world."),
            (&["a", ";", "b"], "a; b"),
            (&["a", ":", "b"], "a: b"),
            (&["we", "'ve", "seen", "."], "we've seen."),
            (&["who", "'d", "know", "."], "who'd know."),
            (&["word", ".", "''"], "word.\""),
            (&["word", ",", "''"], "word,\""),
            (&["word", "!", "''"], "word!\""),
            (&["word", "?", "''"], "word?\""),
            (&["``", "A", "B", "C", "''"], "\"A B C\""),
            (
                &["``", "Hi", ",", "''", "said", "Jo", "."],
                "\"Hi,\" said Jo.",
            ),
            (&["a", ",", "b"], "a, b"),
            (&["a", "."], "a."),
            (&["``", "a", "''"], "\"a\""),
            (
                &[
                    "Lorem",
                    "ipsum",
                    "dolor",
                    "sit",
                    "amet",
                    ".",
                    "consectetur",
                    "adipiscing",
                    "elit",
                    ".",
                ],
                "Lorem ipsum dolor sit amet. consectetur adipiscing elit.",
            ),
            (
                &[
                    "I", "called", "Dr.", "Jones", ".", "I", "called", "Dr.", "Jones", ".",
                ],
                "I called Dr. Jones. I called Dr. Jones.",
            ),
            (&["wait", "...", "what"], "wait...what"),
            (&["`", "Hello", "'"], "` Hello'"),
            (&[], ""),
            (&[""], ""),
            (&["d", "'ye", "gim", "me", "'t", "is"], "d'ye gimme'tis"),
        ];
        for (t, want) in cases {
            assert_eq!(detok(t), want, "{t:?}");
        }
        assert_eq!(detokenize(&["-LRB-", "hello", "-RRB-"], true), "(hello)");
        // test_issue_3260_nested_quotes
        let t = [
            "``", "Shippers", "are", "saying", "`", "the", "party", "'s", "over", ",", "'", "''",
            "said", "Mr.", "LaLonde", ".",
        ];
        assert!(detok(&t).contains(",'\""));
        assert_eq!(tb("Hello, world."), ["Hello", ",", "world", "."]);
    }

    #[test]
    fn align_tokens_doctests() {
        assert_eq!(align_tokens(&[""], ""), Ok(vec![(0, 0)]));
        assert_eq!(align_tokens(&[""], " "), Ok(vec![(0, 0)]));
        assert_eq!(align_tokens(&[], ""), Ok(vec![]));
        assert_eq!(align_tokens(&["a"], "a"), Ok(vec![(0, 1)]));
        assert_eq!(
            align_tokens(&["abc", "def"], "abcdef"),
            Ok(vec![(0, 3), (3, 6)])
        );
        assert_eq!(
            align_tokens(&["abc", "def"], "abc def"),
            Ok(vec![(0, 3), (4, 7)])
        );
        assert_eq!(
            align_tokens(&["ab", "cd", "ef"], "ab cd ef"),
            Ok(vec![(0, 2), (3, 5), (6, 8)])
        );
        let err = align_tokens(&["ab", "cd", "ef", "gh"], "ab cd ef").unwrap_err();
        assert_eq!(
            err.to_string(),
            "substring \"gh\" not found in \"ab cd ef\""
        );
        let s = "The plane, bound for St Petersburg, crashed in Egypt's Sinai desert just 23 minutes after take-off from Sharm el-Sheikh on Saturday.";
        let t = tb(s);
        let t: Vec<&str> = t.iter().map(String::as_str).collect();
        let got = align_tokens(&t, s).unwrap();
        assert_eq!(got.len(), 25);
        assert_eq!(got[..4], [(0, 3), (4, 9), (9, 10), (11, 16)]);
        assert_eq!(got[24], (131, 132));
        assert_eq!(slices(s, &got), t);
        // Code point offsets.
        assert_eq!(
            align_tokens(&["\u{e9}", "b"], "\u{e9} b"),
            Ok(vec![(0, 1), (2, 3)])
        );
        assert_eq!(
            AlignError::QuotesExhausted.to_string(),
            "generator raised StopIteration"
        );
    }

    #[test]
    fn regex_details() {
        // `$` before a final newline; `[^\d]` takes the newline.
        assert_eq!(tokenize_text("a,", Flavor::Treebank, false, V), " a ,  ");
        assert_eq!(
            tokenize_text("a,\n", Flavor::Treebank, false, V),
            " a , \n "
        );
        // Non-overlapping: the second comma is the first one's `[^\d]`.
        assert_eq!(tb("a,,1"), ["a", ",", ",1"]);
        assert_eq!(nl("a.  "), ["a", "."]);
        assert_eq!(tokenize_text("a.  ", Flavor::Nltk, false, V), " a . ");
        // Backtick runs.
        assert_eq!(nl("```a"), ["``", "`", "a"]);
        assert_eq!(tb("```a"), ["``", "`a"]);
        assert_eq!(nl("\"`a"), ["``", "`", "a"]);
        // Clitics under regex's (?i): U+017F matches `s`, U+0131 does not match `i`.
        assert_eq!(nl("'\u{17f} x"), ["'\u{17f}", "x"]);
        assert_eq!(nl("'abc x"), ["'", "abc", "x"]);
        // re's (?i) in the contractions: U+0130 and U+0131 match `i`.
        assert_eq!(nl("g\u{130}mme"), ["g\u{130}m", "me"]);
        assert_eq!(nl("g\u{131}mme"), ["g\u{131}m", "me"]);
        assert_eq!(tb("'T\u{130}s"), ["'T", "\u{130}s"]);
        assert_eq!(nl("'T\u{130}s"), ["'", "T\u{130}s"]);
        // wanna needs a following (str.isspace) space; U+001C is one for re.
        assert_eq!(nl("wanna\u{1c}x"), ["wan", "na", "x"]);
        assert_eq!(nl("wanna"), ["wan", "na"]);
        assert_eq!(nl("wannax"), ["wannax"]);
        // re's \b: U+0301 is not a word character there, U+00B2 is.
        assert_eq!(nl("\u{301}cannot"), ["\u{301}", "can", "not"]);
        assert_eq!(nl("\u{b2}cannot"), ["\u{b2}cannot"]);
        assert_eq!(nl(""), Vec::<String>::new());
        assert_eq!(nl(" \t\n"), Vec::<String>::new());
        assert_eq!(span_tokenize("", Flavor::Nltk, V), Ok(vec![]));
    }
}
