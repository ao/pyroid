// Derived from sacrebleu 2.6.0 (Apache-2.0), © Amazon.com, Inc. or its affiliates; modified.
//! sacrebleu's native BLEU tokenizers: `13a` (the default), `intl`, `zh`,
//! `char` and `none`.
//!
//! Each is a port of the Python class's `__call__` (`tokenizer_13a.py`,
//! `tokenizer_re.py`, `tokenizer_intl.py`, `tokenizer_zh.py`,
//! `tokenizer_char.py`, `tokenizer_none.py`). The `re.sub` passes are
//! reproduced as left-to-right scans: every pattern is one or two
//! single-character classes, so at each position it either matches (and the
//! scan resumes after the match) or the character is copied, which is exactly
//! what `re.sub` does for such patterns. In `[^0-9]` the class is the ASCII
//! digits (no flags); negated classes and `\P{N}` also match `\n`.
//!
//! The zh tokenizer's `_UCODE_RANGES` (with its five-hex-digit bounds such
//! as `' 0'`, i.e. `' ' + '0'`) and intl's `regex` classes are
//! generated tables (`tables.rs`).

use std::borrow::Cow;

use super::pystr;
use super::tables;

/// sacrebleu's `BLEU.TOKENIZERS` that pyroid runs natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Tokenizer {
    /// `none`: the line as is.
    None,
    /// `13a`: mteval-v13a.pl, sacrebleu's default.
    Thirteen,
    /// `intl`: mteval-v14.pl's international tokenization.
    Intl,
    /// `zh`: one token per Chinese character, `13a`-style elsewhere.
    Zh,
    /// `char`: one token per (non-whitespace) code point.
    Char,
}

impl Tokenizer {
    /// The tokenizer called `name` in `BLEU.TOKENIZERS`, if it is native.
    ///
    /// ```
    /// use pyroid_core::metrics::sacrebleu::Tokenizer;
    /// assert_eq!(Tokenizer::from_name("13a"), Some(Tokenizer::Thirteen));
    /// assert_eq!(Tokenizer::from_name("ja-mecab"), None);
    /// ```
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "none" => Self::None,
            "13a" => Self::Thirteen,
            "intl" => Self::Intl,
            "zh" => Self::Zh,
            "char" => Self::Char,
            _ => return None,
        })
    }

    /// `tokenizer.signature()`, the `tok:` field of BLEU signatures.
    pub fn signature(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Thirteen => "13a",
            Self::Intl => "intl",
            Self::Zh => "zh",
            Self::Char => "char",
        }
    }

    /// `tokenizer(line)` in sacrebleu: the tokenized line, tokens separated
    /// by single spaces (`none` returns `line` unchanged).
    ///
    /// ```
    /// use pyroid_core::metrics::sacrebleu::Tokenizer;
    /// assert_eq!(Tokenizer::Thirteen.tokenize("Hello, world!"), "Hello , world !");
    /// assert_eq!(Tokenizer::Intl.tokenize("1,000.5 $"), "1,000.5 $");
    /// assert_eq!(Tokenizer::Zh.tokenize("我爱 NLP."), "我 爱 NLP .");
    /// assert_eq!(Tokenizer::Char.tokenize("ab c"), "a b   c");
    /// assert_eq!(Tokenizer::None.tokenize(" a  b "), " a  b ");
    /// ```
    pub fn tokenize(self, line: &str) -> Cow<'_, str> {
        match self {
            Self::None => Cow::Borrowed(line),
            // " ".join(char for char in line)
            Self::Char => {
                let mut out = String::with_capacity(line.len() * 2);
                for (i, c) in line.chars().enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push(c);
                }
                Cow::Owned(out)
            }
            _ => {
                let text = self.pre_split(line);
                let mut out = String::with_capacity(text.len());
                for (i, tok) in pystr::split(&text).enumerate() {
                    if i > 0 {
                        out.push(' ');
                    }
                    out.push_str(tok);
                }
                Cow::Owned(out)
            }
        }
    }

    /// The text before the tokenizer's final `' '.join(line.split())` (for
    /// `char` and `none`, the line itself).
    fn pre_split(self, line: &str) -> Cow<'_, str> {
        match self {
            Self::None | Self::Char => Cow::Borrowed(line),
            Self::Thirteen => Cow::Owned(thirteen_a(line)),
            Self::Intl => Cow::Owned(intl(line)),
            Self::Zh => Cow::Owned(zh(line)),
        }
    }

    /// The tokens BLEU counts for `line`: `self(line).split()` in sacrebleu
    /// (BLEU's `_preprocess_segment` then `extract_all_word_ngrams`, after the
    /// caller's `rstrip`). The tokens borrow from the returned text.
    pub fn tokens(self, line: &str) -> Tokens<'_> {
        Tokens {
            text: self.pre_split(line),
            per_char: self == Self::Char,
        }
    }
}

/// A tokenized line (see [`Tokenizer::tokens`]).
#[derive(Debug)]
pub struct Tokens<'a> {
    text: Cow<'a, str>,
    per_char: bool,
}

impl Tokens<'_> {
    /// Appends the tokens to `out`.
    pub fn push_to<'t>(&'t self, out: &mut Vec<&'t str>) {
        if self.per_char {
            // `" ".join(chars).split()`: every non-whitespace code point.
            out.extend(
                self.text
                    .char_indices()
                    .filter(|&(_, c)| !pystr::is_space(c))
                    .map(|(i, c)| &self.text[i..i + c.len_utf8()]),
            );
        } else {
            out.extend(pystr::split(&self.text));
        }
    }
}

/// `str.replace(old, new)`, without a copy when `old` does not occur.
fn replace<'a>(s: Cow<'a, str>, old: &str, new: &str) -> Cow<'a, str> {
    if s.contains(old) {
        Cow::Owned(s.replace(old, new))
    } else {
        s
    }
}

/// `Tokenizer13a.__call__` without the final split/join.
fn thirteen_a(line: &str) -> String {
    let mut s = Cow::Borrowed(line);
    s = replace(s, "<skipped>", "");
    s = replace(s, "-\n", "");
    s = replace(s, "\n", " ");
    if s.contains('&') {
        s = replace(s, "&quot;", "\"");
        s = replace(s, "&amp;", "&");
        s = replace(s, "&lt;", "<");
        s = replace(s, "&gt;", ">");
    }
    let mut padded = String::with_capacity(s.len() + 2);
    padded.push(' ');
    padded.push_str(&s);
    padded.push(' ');
    regexp(&padded)
}

/// `[\{-\~\[-\` -\&\(-\+\:-\@\/]`: the characters `13a` surrounds with spaces.
const fn is_13a_punct(b: u8) -> bool {
    matches!(b, b'{'..=b'~' | b'['..=b'`' | b' '..=b'&' | b'('..=b'+' | b':'..=b'@' | b'/')
}

const fn is_period_comma(b: u8) -> bool {
    matches!(b, b'.' | b',')
}

const fn is_ascii_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

const fn not_ascii_digit(b: u8) -> bool {
    !b.is_ascii_digit()
}

/// Where a two-group replacement puts its spaces.
#[derive(Clone, Copy)]
enum Pair {
    /// `r'\1 \2 '`
    AfterBoth,
    /// `r' \1 \2'`
    BeforeBoth,
}

/// `re.sub(r'(X)', r' \1 ', s)` for an ASCII class `X`, over UTF-8 bytes
/// (a byte of a multi-byte code point is never ASCII).
fn sub_one_bytes(s: &[u8], out: &mut Vec<u8>, x: impl Fn(u8) -> bool) {
    out.clear();
    let mut run = 0;
    for (i, &b) in s.iter().enumerate() {
        if x(b) {
            out.extend_from_slice(&s[run..i]);
            out.extend_from_slice(&[b' ', b, b' ']);
            run = i + 1;
        }
    }
    out.extend_from_slice(&s[run..]);
}

/// `re.sub(r'(A)(B)', repl, s)` over UTF-8 bytes, left to right and
/// non-overlapping, for the `13a` pairs: one class is ASCII (`[\.,]`,
/// `[0-9]`, `-`), the other ASCII or its complement (`[^0-9]`). That is the
/// same as over code points: a byte pair inside one code point never matches
/// (an ASCII class never holds a lead or continuation byte); a match whose
/// complement side is a multi-byte code point matches on its last byte (`A`)
/// or first byte (`B`), and the bytes left over can't start a match. The
/// spaces go in at code point boundaries.
fn sub_pair_bytes(
    s: &[u8],
    out: &mut Vec<u8>,
    a: impl Fn(u8) -> bool,
    b: impl Fn(u8) -> bool,
    repl: Pair,
) {
    out.clear();
    let (mut i, mut run) = (0, 0);
    while i + 1 < s.len() {
        if a(s[i]) && b(s[i + 1]) {
            out.extend_from_slice(&s[run..i]);
            match repl {
                Pair::AfterBoth => out.extend_from_slice(&[s[i], b' ', s[i + 1], b' ']),
                Pair::BeforeBoth => out.extend_from_slice(&[b' ', s[i], b' ', s[i + 1]]),
            }
            i += 2;
            run = i;
        } else {
            i += 1;
        }
    }
    out.extend_from_slice(&s[run..]);
}

/// `TokenizerRegexp.__call__` without the final split/join.
fn regexp(s: &str) -> String {
    let mut a = Vec::with_capacity(s.len() + s.len() / 2 + 8);
    let mut b = Vec::with_capacity(a.capacity());
    // language-dependent part (assuming Western languages)
    sub_one_bytes(s.as_bytes(), &mut a, is_13a_punct);
    if memchr::memchr2(b'.', b',', &a).is_some() {
        // tokenize period and comma unless preceded by a digit
        sub_pair_bytes(
            &a,
            &mut b,
            not_ascii_digit,
            is_period_comma,
            Pair::AfterBoth,
        );
        // tokenize period and comma unless followed by a digit
        sub_pair_bytes(
            &b,
            &mut a,
            is_period_comma,
            not_ascii_digit,
            Pair::BeforeBoth,
        );
    }
    if memchr::memchr(b'-', &a).is_some() {
        // tokenize dash when preceded by a digit
        sub_pair_bytes(&a, &mut b, is_ascii_digit, |c| c == b'-', Pair::AfterBoth);
        std::mem::swap(&mut a, &mut b);
    }
    String::from_utf8(a).expect("spaces go in at code point boundaries")
}

/// `re.sub(r'(X)', r' \1 ', s)` for a one-code-point class `X`.
fn sub_one(s: &str, out: &mut String, x: impl Fn(char) -> bool) {
    out.clear();
    let mut run = 0;
    for (i, c) in s.char_indices() {
        if x(c) {
            out.push_str(&s[run..i]);
            out.push(' ');
            out.push(c);
            out.push(' ');
            run = i + c.len_utf8();
        }
    }
    out.push_str(&s[run..]);
}

/// `re.sub(r'(A)(B)', repl, s)` for one-code-point classes `A` and `B`:
/// left to right, non-overlapping.
fn sub_pair(
    s: &str,
    out: &mut String,
    a: impl Fn(char) -> bool,
    b: impl Fn(char) -> bool,
    repl: Pair,
) {
    out.clear();
    let mut run = 0;
    let mut chars = s.char_indices().peekable();
    while let Some((i, c)) = chars.next() {
        if let Some(&(j, d)) = chars.peek() {
            if a(c) && b(d) {
                chars.next();
                out.push_str(&s[run..i]);
                let (first, second) = match repl {
                    Pair::AfterBoth => ("", " "),
                    Pair::BeforeBoth => (" ", ""),
                };
                out.push_str(first);
                out.push(c);
                out.push(' ');
                out.push(d);
                out.push_str(second);
                run = j + d.len_utf8();
            }
        }
    }
    out.push_str(&s[run..]);
}

/// A code point's class for the `intl` tokenizer's `regex` patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntlClass {
    /// None of the below.
    Other,
    /// `\p{P}`: punctuation.
    Punct,
    /// `\p{S}`: symbols.
    Symbol,
    /// `\p{N}`: numbers.
    Number,
}

/// The class of `c` in the `regex` release sacrebleu was generated against.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::{IntlClass, intl_class};
/// assert_eq!(intl_class('.'), IntlClass::Punct);
/// assert_eq!(intl_class('€'), IntlClass::Symbol);
/// assert_eq!(intl_class('٣'), IntlClass::Number);
/// assert_eq!(intl_class('a'), IntlClass::Other);
/// ```
pub fn intl_class(c: char) -> IntlClass {
    let cp = u32::from(c);
    if let Some(&class) = tables::INTL_ASCII.get(cp as usize) {
        return class;
    }
    let table = tables::INTL;
    match table.binary_search_by(|&(first, last, _)| {
        if last < cp {
            std::cmp::Ordering::Less
        } else if first > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => table[i].2,
        Err(_) => IntlClass::Other,
    }
}

fn is_punct(c: char) -> bool {
    intl_class(c) == IntlClass::Punct
}

fn is_symbol(c: char) -> bool {
    intl_class(c) == IntlClass::Symbol
}

fn not_number(c: char) -> bool {
    intl_class(c) != IntlClass::Number
}

/// `TokenizerV14International.__call__` without the final split/join.
fn intl(line: &str) -> String {
    let mut a = String::with_capacity(line.len() + line.len() / 2 + 8);
    let mut b = String::with_capacity(a.capacity());
    // Separate out punctuations preceded by a non-digit
    sub_pair(line, &mut a, not_number, is_punct, Pair::AfterBoth);
    // Separate out punctuations followed by a non-digit
    sub_pair(&a, &mut b, is_punct, not_number, Pair::BeforeBoth);
    // Separate out symbols
    sub_one(&b, &mut a, is_symbol);
    a
}

/// `TokenizerZh._is_chinese_char(c)`.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::is_chinese_char;
/// assert!(is_chinese_char('中') && is_chinese_char('，') && is_chinese_char('\u{2001}'));
/// // ' 0' in _UCODE_RANGES is ' ' + '0': U+2000 itself is out, U+20000 too.
/// assert!(!is_chinese_char('\u{2000}') && !is_chinese_char('\u{20000}'));
/// ```
pub fn is_chinese_char(c: char) -> bool {
    let cp = u32::from(c);
    if cp < 0x2001 {
        return false;
    }
    let table = tables::ZH;
    let i = table.partition_point(|&(first, _)| first <= cp);
    i > 0 && cp <= table[i - 1].1
}

/// `TokenizerZh.__call__` without the final split/join.
fn zh(line: &str) -> String {
    let line = pystr::strip(line);
    let mut chars = String::with_capacity(line.len() + line.len() / 2);
    for c in line.chars() {
        if is_chinese_char(c) {
            chars.push(' ');
            chars.push(c);
            chars.push(' ');
        } else {
            chars.push(c);
        }
    }
    regexp(&chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(t: Tokenizer, s: &str) -> String {
        t.tokenize(s).into_owned()
    }

    fn tokens(t: Tokenizer, s: &str) -> Vec<String> {
        let tk = t.tokens(s);
        let mut out = Vec::new();
        tk.push_to(&mut out);
        out.into_iter().map(str::to_owned).collect()
    }

    #[test]
    fn thirteen_a_matches_sacrebleu() {
        // Expected values from sacrebleu 2.6.0's Tokenizer13a.
        let cases = [
            ("Hello, world!", "Hello , world !"),
            (
                "It costs $3.50, or 3,50 EUR.",
                "It costs $ 3.50 , or 3,50 EUR .",
            ),
            ("a-b 1-2 x.y", "a-b 1 - 2 x . y"),
            (
                "&quot;hi&quot; &amp;amp; &lt;b&gt;",
                "\" hi \" & amp ; < b >",
            ),
            ("line<skipped>one-\ntwo\nthree", "lineonetwo three"),
            (
                "e.g., (see) [x] {y} ~z",
                "e . g . , ( see ) [ x ] { y } ~ z",
            ),
            ("5.", "5 ."),
            (".5 ,5 a,", ". 5 , 5 a ,"),
            ("é. ü,ß", "é . ü , ß"),
            ("", ""),
        ];
        for (s, want) in cases {
            assert_eq!(tok(Tokenizer::Thirteen, s), want, "{s:?}");
        }
    }

    #[test]
    fn regexp_passes_do_not_overlap() {
        // ([^0-9])([\.,]) consumes both characters; the second period is
        // tokenized by the next pass. Expected values from sacrebleu's passes.
        assert_eq!(regexp("a.."), "a  .  .");
        assert_eq!(regexp("\n."), "\n  .  ");
        assert_eq!(regexp("1-2-3"), "1 - 2 - 3");
    }

    #[test]
    fn intl_matches_sacrebleu() {
        let cases = [
            ("Hello, world!", "Hello , world !"),
            ("1,000.5 $ 2.", "1,000.5 $ 2."),
            ("«quoted» — dash", "« quoted » — dash"),
            ("a+b=c", "a + b = c"),
            ("x.\ny", "x . y"),
        ];
        for (s, want) in cases {
            assert_eq!(tok(Tokenizer::Intl, s), want, "{s:?}");
        }
    }

    #[test]
    fn zh_and_char() {
        assert_eq!(
            tok(Tokenizer::Zh, " 我爱北京天安门。Hi, 2.5! "),
            "我 爱 北 京 天 安 门 。 Hi , 2.5 !"
        );
        assert_eq!(tokens(Tokenizer::Char, "a b\u{a0}c"), ["a", "b", "c"]);
        assert_eq!(tokens(Tokenizer::None, " a  b "), ["a", "b"]);
        assert_eq!(tokens(Tokenizer::Thirteen, "a, b"), ["a", ",", "b"]);
    }

    #[test]
    fn names_round_trip() {
        for t in [
            Tokenizer::None,
            Tokenizer::Thirteen,
            Tokenizer::Intl,
            Tokenizer::Zh,
            Tokenizer::Char,
        ] {
            assert_eq!(Tokenizer::from_name(t.signature()), Some(t));
        }
    }

    #[test]
    fn tables_are_sorted_and_disjoint() {
        assert!(tables::INTL.windows(2).all(|w| w[0].1 < w[1].0));
        assert!(
            tables::INTL
                .iter()
                .all(|r| r.0 <= r.1 && r.2 != IntlClass::Other)
        );
        assert!(tables::ZH.windows(2).all(|w| w[0].1 < w[1].0));
        assert_eq!(tables::SACREBLEU_VERSION, "2.6.0");
    }
}
