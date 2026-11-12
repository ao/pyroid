// Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
//! rouge-score's default tokenizer (`rouge_score.tokenize.tokenize`):
//!
//! ```python
//! text = text.lower()
//! text = re.sub(r"[^a-z0-9]+", " ", text)
//! tokens = re.split(r"\s+", text)
//! tokens = [stemmer.stem(x) if len(x) > 3 else x for x in tokens]   # use_stemmer=True
//! tokens = [x for x in tokens if re.match(r"^[a-z0-9]+$", x)]
//! ```
//!
//! Only ASCII letters and digits survive, so of Python's full Unicode
//! lowercasing only what produces them matters. Over all of Unicode
//! (checked exhaustively on CPython 3.12 to 3.15, Unicode 15.0 to 17.0)
//! that is `A-Z`, plus two code points:
//! - U+0130 LATIN CAPITAL LETTER I WITH DOT ABOVE lowercases to `"i\u{307}"`:
//!   an `i` that ends the token (U+0307 is a separator);
//! - U+212A KELVIN SIGN lowercases to `"k"`, which joins its neighbours.
//!
//! Every other character is a separator. Lowercasing is per character
//! (final sigma is the one context rule, and it never yields ASCII), so a
//! text's tokens are the concatenation of its lines' tokens: rougeLsum's
//! `text.split("\n")` sentences are runs of the whole text's tokens.

use super::porter;

/// ASCII byte -> token byte (`0`: separator), after `str.lower()`.
const FOLD: [u8; 128] = ascii_map(true);
/// ASCII byte -> token byte for text that is already lowercased: `A-Z` are separators.
const AS_IS: [u8; 128] = ascii_map(false);

const fn ascii_map(fold: bool) -> [u8; 128] {
    let mut map = [0; 128];
    let mut b: u8 = 0;
    while b < 128 {
        map[b as usize] = if b.is_ascii_lowercase() || b.is_ascii_digit() {
            b
        } else if fold && b.is_ascii_uppercase() {
            b.to_ascii_lowercase()
        } else {
            0
        };
        b += 1;
    }
    map
}

/// The tokens of one or more texts, packed into one buffer (unstemmed).
#[derive(Debug, Default, Clone)]
pub struct Tokens {
    bytes: Vec<u8>,
    /// Token `i` is `bytes[ends[i - 1]..ends[i]]` (`ends[-1]` = 0).
    ends: Vec<usize>,
    /// Token indices at which a line (`"\n"`) starts, recorded by [`Tokens::push_text`].
    lines: Vec<usize>,
}

impl Tokens {
    /// Number of tokens.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// No tokens?
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Token `i`.
    pub fn get(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        &self.bytes[start..self.ends[i]]
    }

    /// The tokens, in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &[u8]> + '_ {
        (0..self.len()).map(|i| self.get(i))
    }

    /// Token indices where a `"\n"` line started, in order and without
    /// repeats (a line with no tokens leaves no trace).
    pub fn line_starts(&self) -> &[usize] {
        &self.lines
    }

    /// Append the tokens of `text`. `fold`: apply `str.lower()` first (as
    /// rouge-score does); `false` for text the caller already lowercased (so
    /// `A-Z`, U+0130 and U+212A are separators like any other character).
    pub fn push_text(&mut self, text: &str, fold: bool) {
        let map = if fold { &FOLD } else { &AS_IS };
        let b = text.as_bytes();
        // Every input byte yields at most one token byte.
        self.bytes.reserve(b.len());
        let mut start = self.bytes.len();
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c < 0x80 {
                let m = map[usize::from(c)];
                if m == 0 {
                    self.end_token(&mut start);
                    if c == b'\n' {
                        self.line_break();
                    }
                } else {
                    self.bytes.push(m);
                }
                i += 1;
            } else if fold && c == 0xC4 && b.get(i + 1) == Some(&0xB0) {
                // U+0130 -> "i\u{307}": the combining dot ends the token.
                self.bytes.push(b'i');
                self.end_token(&mut start);
                i += 2;
            } else if fold
                && c == 0xE2
                && b.get(i + 1) == Some(&0x84)
                && b.get(i + 2) == Some(&0xAA)
            {
                // U+212A KELVIN SIGN -> "k".
                self.bytes.push(b'k');
                i += 3;
            } else {
                // Any other non-ASCII character (continuation bytes are never
                // 0xC4 or 0xE2, so stepping one byte at a time is safe).
                self.end_token(&mut start);
                i += 1;
            }
        }
        self.end_token(&mut start);
    }

    fn end_token(&mut self, start: &mut usize) {
        if self.bytes.len() > *start {
            self.ends.push(self.bytes.len());
            *start = self.bytes.len();
        }
    }

    fn line_break(&mut self) {
        let n = self.ends.len();
        if self.lines.last() != Some(&n) {
            self.lines.push(n);
        }
    }
}

/// `rouge_score.tokenize.tokenize(text, stemmer)` with `stemmer` either
/// `None` or nltk's `PorterStemmer()` (`stem = true`).
///
/// `fold = false` skips the lowercasing, for a caller that already applied
/// Python's `lower()` itself (rouge-score lowercases `bytes` ASCII-only).
///
/// ```
/// use pyroid_core::metrics::rouge::tokenize::tokenize;
/// assert_eq!(tokenize("One Two, three!", false, true), ["one", "two", "three"]);
/// assert_eq!(tokenize("The friends had a meeting", true, true), ["the", "friend", "had", "a", "meet"]);
/// assert_eq!(tokenize("İstanbul \u{212A}elvin", false, true), ["i", "stanbul", "kelvin"]);
/// ```
pub fn tokenize(text: &str, stem: bool, fold: bool) -> Vec<String> {
    let mut tokens = Tokens::default();
    tokens.push_text(text, fold);
    tokens
        .iter()
        .map(|t| {
            let t = if stem && t.len() > 3 {
                porter::stem(t)
            } else {
                t.into()
            };
            // Token bytes are ASCII letters and digits (the stemmer keeps them so).
            String::from_utf8(t.into_owned()).expect("tokens are ASCII")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toks(text: &str) -> Vec<String> {
        tokenize(text, false, true)
    }

    #[test]
    fn rouge_score_cases() {
        assert_eq!(toks("one Two three"), ["one", "two", "three"]);
        assert_eq!(toks("one\n Two \nthree"), ["one", "two", "three"]);
        assert_eq!(toks("this is a test"), ["this", "is", "a", "test"]);
        assert_eq!(
            tokenize("the friends had a meeting", true, true),
            ["the", "friend", "had", "a", "meet"]
        );
        assert_eq!(toks(""), Vec::<String>::new());
        assert_eq!(toks(" \t\n/!"), Vec::<String>::new());
    }

    #[test]
    fn unicode_lowercasing() {
        // U+0130 -> "i" + U+0307 (a separator); U+212A -> "k" (joins).
        assert_eq!(toks("İSTANBUL"), ["i", "stanbul"]);
        assert_eq!(toks("aİb"), ["ai", "b"]);
        assert_eq!(toks("\u{212A}g 5\u{212A}"), ["kg", "5k"]);
        // Everything else non-ASCII separates, including other letters that
        // lowercase to non-ASCII and ASCII-looking fullwidth forms.
        assert_eq!(
            toks("straße ÀB ＡＢ x\u{301}y"),
            ["stra", "e", "b", "x", "y"]
        );
        // Already lowercased text: uppercase ASCII and the specials separate.
        assert_eq!(tokenize("aBc İx \u{212A}", false, false), ["a", "c", "x"]);
    }

    #[test]
    fn stemming_only_long_tokens() {
        assert_eq!(
            tokenize("cats dogs running", true, true),
            ["cat", "dog", "run"]
        );
        assert_eq!(tokenize("ies sky was", true, true), ["ies", "sky", "was"]);
    }

    #[test]
    fn line_starts() {
        let mut t = Tokens::default();
        t.push_text("a b\nc\n\n!!\nd e\n", true);
        assert_eq!(t.len(), 5);
        assert_eq!(t.line_starts(), [2, 3, 5]);
        assert_eq!(t.iter().collect::<Vec<_>>(), [b"a", b"b", b"c", b"d", b"e"]);
        let mut t = Tokens::default();
        t.push_text("\nx", true);
        assert_eq!(t.line_starts(), [0]);
        assert!(!t.is_empty());
    }
}
