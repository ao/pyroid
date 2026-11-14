// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! The reader (`reader.py`): the decoded stream, `peek`/`prefix`/`forward` and marks.
//!
//! PyYAML's reader holds a buffer of code points ending in `'\0'`. This one walks the UTF-8
//! text with a byte cursor and counts code points, lines and columns exactly as
//! `Reader.forward` does (`'\r'` not followed by `'\n'`, `'\n'`, `'\x85'`, U+2028 and U+2029
//! end a line; U+FEFF does not move the column). Past the end it reads `'\0'`.
//!
//! For file objects PyYAML reads 4096 units at a time and only decodes as far as the scanner
//! looks, so a decoding error (or a character it rejects) further on is raised only when the
//! scanner gets there. The bindings give such a stream as its valid prefix plus a `limit`:
//! looking past it sets a flag that the scanner turns into [`YamlError::NeedMore`], and the
//! bindings raise the pending `ReaderError`.

use std::cell::Cell;

use super::error::{Mark, ReaderError};

/// The first code point of the stream that PyYAML's `Reader.NON_PRINTABLE` rejects.
///
/// PyYAML accepts `\t \n \r`, `\x20-\x7E`, `\x85`, `\xA0-\uD7FF`, `\uE000-\uFFFD` and
/// `\U00010000-\U0010FFFF`. Returns `(code point index, character)`.
pub fn find_non_printable(text: &str) -> Option<(usize, char)> {
    let bytes = text.as_bytes();
    let mut index = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            if b < 0x20 && b != b'\t' && b != b'\n' && b != b'\r' || b == 0x7F {
                return Some((index, char::from(b)));
            }
            i += 1;
        } else {
            let ch = text[i..].chars().next().unwrap_or('\0');
            if !is_printable(ch) {
                return Some((index, ch));
            }
            i += ch.len_utf8();
        }
        index += 1;
    }
    None
}

/// PyYAML's printable set (a Rust `char` is never a surrogate).
pub fn is_printable(ch: char) -> bool {
    matches!(ch, '\t' | '\n' | '\r' | '\x20'..='\x7E' | '\u{85}' | '\u{A0}'..='\u{D7FF}' | '\u{E000}'..='\u{FFFD}' | '\u{10000}'..)
}

/// Whether `text` holds a character in U+10F800..=U+10FFFF, the range the scanner uses to
/// carry lone surrogates from `\uD800`-style escapes.
pub fn has_reserved(text: &str) -> bool {
    // U+10F800.. is F4 8F A0 80 ..= F4 8F BF BF in UTF-8.
    memchr::memmem::find_iter(text.as_bytes(), b"\xF4\x8F")
        .any(|i| text.as_bytes().get(i + 2).is_some_and(|&b| b >= 0xA0))
}

/// First code point of the reserved range (see [`has_reserved`]).
pub const SURROGATE_BASE: u32 = 0x10_F800;

/// The `ReaderError` PyYAML raises for invalid UTF-8 `bytes` (`codecs.utf_8_decode(..., final=True)`).
pub fn utf8_error(bytes: &[u8], err: std::str::Utf8Error) -> ReaderError {
    let start = err.valid_up_to();
    let byte = bytes[start];
    let reason = match err.error_len() {
        None => "unexpected end of data",
        Some(_) if (0xC2..=0xF4).contains(&byte) => "invalid continuation byte",
        Some(_) => "invalid start byte",
    };
    ReaderError {
        position: start,
        character: u32::from(byte),
        encoding: "utf-8",
        reason,
    }
}

/// The `ReaderError` for a code point PyYAML rejects (`check_printable`).
pub fn printable_error(position: usize, ch: u32) -> ReaderError {
    ReaderError {
        position,
        character: ch,
        encoding: "unicode",
        reason: "special characters are not allowed",
    }
}

/// The decoded stream and the current position.
#[derive(Debug)]
pub struct Reader {
    text: String,
    /// Byte offset of the current character.
    pos: usize,
    /// Code points before the current one.
    pub index: usize,
    pub line: usize,
    pub column: usize,
    /// Code points PyYAML has decoded before a pending reader error (file objects).
    limit: Option<usize>,
    hit_limit: Cell<bool>,
}

impl Reader {
    /// A reader over `text` (already decoded and checked). `limit`: see the module docs.
    pub fn new(text: String, limit: Option<usize>) -> Self {
        let reader = Self {
            text,
            pos: 0,
            index: 0,
            line: 0,
            column: 0,
            limit,
            hit_limit: Cell::new(false),
        };
        // PyYAML's Reader.__init__ ends with `update(1)` (determine_encoding).
        reader.need(1);
        reader
    }

    /// The whole text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Byte offset of the current character.
    pub fn pos(&self) -> usize {
        self.pos
    }

    /// Whether the scanner looked past the pending reader error.
    pub fn hit_limit(&self) -> bool {
        self.hit_limit.get()
    }

    /// PyYAML's `update(n)`: the next `n` code points must be decoded.
    #[inline]
    fn need(&self, n: usize) {
        if self.limit.is_some_and(|limit| self.index + n > limit) {
            self.hit_limit.set(true);
        }
    }

    /// The character at byte offset `at` (`'\0'` past the end) and its UTF-8 length.
    #[inline]
    pub fn char_at(&self, at: usize) -> (char, usize) {
        match self.text.as_bytes().get(at) {
            None => ('\0', 0),
            Some(&b) if b < 0x80 => (char::from(b), 1),
            Some(_) => {
                let ch = self.text[at..].chars().next().unwrap_or('\0');
                (ch, ch.len_utf8())
            }
        }
    }

    /// `peek()`: the current character.
    #[inline]
    pub fn peek0(&self) -> char {
        self.need(1);
        match self.text.as_bytes().get(self.pos) {
            None => '\0',
            Some(&b) if b < 0x80 => char::from(b),
            Some(_) => self.text[self.pos..].chars().next().unwrap_or('\0'),
        }
    }

    /// `peek(k)`: the character `k` code points ahead.
    #[inline]
    pub fn peek(&self, k: usize) -> char {
        if k == 0 {
            return self.peek0();
        }
        self.need(k + 1);
        let mut at = self.pos;
        for _ in 0..k {
            let (_, len) = self.char_at(at);
            if len == 0 {
                return '\0';
            }
            at += len;
        }
        self.char_at(at).0
    }

    /// The byte offset `k` code points ahead (the end of the text at most).
    pub fn offset(&self, k: usize) -> usize {
        let mut at = self.pos;
        for _ in 0..k {
            let (_, len) = self.char_at(at);
            if len == 0 {
                break;
            }
            at += len;
        }
        at
    }

    /// `prefix(n)`: the next `n` code points (fewer at the end).
    pub fn prefix(&self, n: usize) -> &str {
        self.need(n);
        let end = self.offset(n);
        &self.text[self.pos..end]
    }

    /// `prefix(n) == s` for an ASCII `s` of `n` bytes.
    #[inline]
    pub fn prefix_is(&self, s: &str) -> bool {
        self.need(s.len());
        self.text.as_bytes()[self.pos..].starts_with(s.as_bytes())
    }

    /// `forward(n)`: move `n` code points on.
    pub fn forward(&mut self, n: usize) {
        self.need(n + 1);
        for _ in 0..n {
            let (ch, len) = self.char_at(self.pos);
            if len == 0 {
                // PyYAML would hit the '\0' sentinel; nothing to move past.
                self.index += 1;
                self.column += 1;
                continue;
            }
            self.pos += len;
            self.index += 1;
            match ch {
                '\n' | '\u{85}' | '\u{2028}' | '\u{2029}' => {
                    self.line += 1;
                    self.column = 0;
                }
                '\r' if self.text.as_bytes().get(self.pos) != Some(&b'\n') => {
                    self.line += 1;
                    self.column = 0;
                }
                '\u{FEFF}' => {}
                _ => self.column += 1,
            }
        }
    }

    /// `forward(1)` over a character known not to be a line break or U+FEFF.
    #[inline]
    pub fn forward_plain1(&mut self, len: usize) {
        self.need(2);
        self.pos += len;
        self.index += 1;
        self.column += 1;
    }

    /// `forward(chars)` over the `bytes` bytes after the cursor, known to hold no line break.
    /// `boms` of them are U+FEFF.
    #[inline]
    pub fn forward_span(&mut self, bytes: usize, chars: usize, boms: usize) {
        self.need(chars + 1);
        self.pos += bytes;
        self.index += chars;
        self.column += chars - boms;
    }

    /// `get_mark()`.
    #[inline]
    pub fn mark(&self) -> Mark {
        Mark {
            index: self.index,
            line: self.line,
            column: self.column,
        }
    }

    /// Note that the scanner looked at `k` code points ahead (for scans that walk the text
    /// themselves with [`Reader::char_at`]).
    #[inline]
    pub fn need_ahead(&self, k: usize) {
        self.need(k + 1);
    }

    /// Give the text back.
    pub fn into_text(self) -> String {
        self.text
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_printable() {
        assert_eq!(find_non_printable("ab\tc\n"), None);
        assert_eq!(find_non_printable("abé\u{7}"), Some((3, '\u{7}')));
        assert_eq!(find_non_printable("a\u{9F}"), Some((1, '\u{9F}')));
        assert_eq!(find_non_printable("a\u{85}\u{FFFE}"), Some((2, '\u{FFFE}')));
        assert_eq!(find_non_printable("\u{10FFFF}\u{7F}"), Some((1, '\u{7F}')));
    }

    #[test]
    fn reserved() {
        assert!(!has_reserved("abc\u{10F7FF}"));
        assert!(has_reserved("a\u{10F800}"));
        assert!(has_reserved("\u{10FFFF}"));
    }

    #[test]
    fn utf8_errors() {
        let check = |b: &[u8]| {
            let e = std::str::from_utf8(b).unwrap_err();
            let r = utf8_error(b, e);
            (r.position, r.character, r.reason)
        };
        assert_eq!(check(b"ab\xff"), (2, 0xff, "invalid start byte"));
        assert_eq!(check(b"a\xe4\x41"), (1, 0xe4, "invalid continuation byte"));
        assert_eq!(check(b"a\xe4\xb8"), (1, 0xe4, "unexpected end of data"));
        assert_eq!(
            check(b"\xed\xa0\x80"),
            (0, 0xed, "invalid continuation byte")
        );
        assert_eq!(check(b"\xc0\x80"), (0, 0xc0, "invalid start byte"));
    }

    #[test]
    fn forward_counts_like_pyyaml() {
        let mut r = Reader::new("a\r\nb\rc\u{FEFF}d\u{2028}e".to_owned(), None);
        r.forward(1);
        assert_eq!(
            r.mark(),
            Mark {
                index: 1,
                line: 0,
                column: 1
            }
        );
        r.forward(1); // '\r' before '\n': column moves
        assert_eq!(
            r.mark(),
            Mark {
                index: 2,
                line: 0,
                column: 2
            }
        );
        r.forward(1);
        assert_eq!(
            r.mark(),
            Mark {
                index: 3,
                line: 1,
                column: 0
            }
        );
        r.forward(2); // b, lone \r
        assert_eq!(
            r.mark(),
            Mark {
                index: 5,
                line: 2,
                column: 0
            }
        );
        r.forward(2); // c, BOM
        assert_eq!(
            r.mark(),
            Mark {
                index: 7,
                line: 2,
                column: 1
            }
        );
        assert_eq!(r.peek0(), 'd');
        assert_eq!(r.peek(1), '\u{2028}');
        assert_eq!(r.peek(3), '\0');
        r.forward(2);
        assert_eq!(
            r.mark(),
            Mark {
                index: 9,
                line: 3,
                column: 0
            }
        );
        assert_eq!(r.prefix(5), "e");
    }

    #[test]
    fn limit_flags_reads_past_it() {
        let r = Reader::new("abc".to_owned(), Some(3));
        assert_eq!(r.peek(2), 'c');
        assert!(!r.hit_limit());
        let _ = r.peek(3);
        assert!(r.hit_limit());
    }
}
