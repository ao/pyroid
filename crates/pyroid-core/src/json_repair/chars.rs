// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! Python's `str` predicates (`isspace`, `isalpha`, `isdigit`, `isalnum`) per
//! Unicode version, and the lone-surrogate remapping.
//!
//! The tables are generated from CPython by `generate_tables.py` (see
//! [`super::tables`]); every predicate has an inline ASCII fast path.
//!
//! Lone surrogates: Python strings may hold U+D800..U+DFFF, `char` may not. The
//! kernel stores surrogate `s` as `char::from_u32(SURROGATE_BASE + (s - 0xD800))`
//! (plane-16 private use, U+10F800..U+10FFFF). For those code points and for real
//! surrogates every predicate here is false, so the parser behaves the same.
//! Inputs that already contain a char in that range are refused
//! ([`InputError::ReservedCodePoint`]): the caller delegates them.
//!
//! In UTF-8 a remapped surrogate is `F4 8F xx yy` and the surrogate itself, as
//! Python's `"surrogatepass"` handler writes it, is `ED xx yy` with the same two
//! continuation bytes (both ranges start on a multiple of 0x800), so converting
//! between the two forms rewrites the lead bytes only.

use std::fmt;

use super::tables::{self, VersionTables};

/// First char of the remapped-surrogate range (U+D800 maps here).
pub const SURROGATE_BASE: u32 = 0x10_F800;

/// The Unicode versions the tables cover (`unicodedata.unidata_version` of CPython 3.11..3.15).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnicodeVersion {
    V14_0,
    V15_0,
    V15_1,
    V16_0,
    V17_0,
}

impl UnicodeVersion {
    /// Every supported version, oldest first.
    pub const ALL: [Self; 5] = [
        Self::V14_0,
        Self::V15_0,
        Self::V15_1,
        Self::V16_0,
        Self::V17_0,
    ];

    /// From `unicodedata.unidata_version` (`"15.1.0"`).
    pub fn from_unidata_version(v: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|x| x.as_str() == v)
    }

    /// `unicodedata.unidata_version` spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::V14_0 => "14.0.0",
            Self::V15_0 => "15.0.0",
            Self::V15_1 => "15.1.0",
            Self::V16_0 => "16.0.0",
            Self::V17_0 => "17.0.0",
        }
    }

    fn tables(self) -> &'static VersionTables {
        match self {
            Self::V14_0 => &tables::V14_0,
            Self::V15_0 => &tables::V15_0,
            Self::V15_1 => &tables::V15_1,
            Self::V16_0 => &tables::V16_0,
            Self::V17_0 => &tables::V17_0,
        }
    }
}

/// Character predicates for one Unicode version. `Copy`; build once per call.
#[derive(Clone, Copy)]
pub struct CharProps {
    version: UnicodeVersion,
    tables: &'static VersionTables,
}

impl fmt::Debug for CharProps {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CharProps")
            .field("version", &self.version)
            .finish_non_exhaustive()
    }
}

impl CharProps {
    pub fn new(version: UnicodeVersion) -> Self {
        Self {
            version,
            tables: version.tables(),
        }
    }

    pub fn version(self) -> UnicodeVersion {
        self.version
    }

    /// `c.isspace()` (the same set in every supported version).
    #[inline]
    #[allow(clippy::unused_self)]
    pub fn is_space(self, c: char) -> bool {
        matches!(
            c,
            '\t'..='\r'
                | '\x1c'..=' '
                | '\u{85}'
                | '\u{a0}'
                | '\u{1680}'
                | '\u{2000}'..='\u{200a}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202f}'
                | '\u{205f}'
                | '\u{3000}'
        )
    }

    /// `c.isalpha()`.
    #[inline]
    pub fn is_alpha(self, c: char) -> bool {
        if c.is_ascii() {
            return c.is_ascii_alphabetic();
        }
        in_ranges(self.tables.alpha, u32::from(c))
    }

    /// `c.isdigit()` (Unicode `Numeric_Type` Digit or Decimal, e.g. `'²'`).
    #[inline]
    pub fn is_digit(self, c: char) -> bool {
        if c.is_ascii() {
            return c.is_ascii_digit();
        }
        in_ranges(self.tables.digit, u32::from(c))
    }

    /// `c.isalnum()` (`isalpha` or `isnumeric`, e.g. `'½'`).
    #[inline]
    pub fn is_alnum(self, c: char) -> bool {
        if c.is_ascii() {
            return c.is_ascii_alphanumeric();
        }
        in_ranges(self.tables.alnum, u32::from(c))
    }
}

/// Whether `cp` lies in one of the sorted, disjoint, inclusive `table` ranges.
fn in_ranges(table: &[(u32, u32)], cp: u32) -> bool {
    match table.binary_search_by(|&(lo, _)| lo.cmp(&cp)) {
        Ok(_) => true,
        Err(0) => false,
        Err(k) => cp <= table[k - 1].1,
    }
}

/// Why an input cannot be handled by the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputError {
    /// The input holds a char in U+10F800..U+10FFFF (reserved for remapped surrogates).
    ReservedCodePoint,
}

/// The remapped char for surrogate code point `cp` (U+D800..=U+DFFF).
#[inline]
pub fn surrogate_char(cp: u32) -> char {
    debug_assert!((0xD800..=0xDFFF).contains(&cp));
    char::from_u32(SURROGATE_BASE + (cp - 0xD800)).expect("in range")
}

/// The Python code point a kernel char stands for (undoes the remapping).
#[inline]
pub fn code_point(c: char) -> u32 {
    let u = c as u32;
    if u >= SURROGATE_BASE {
        0xD800 + (u - SURROGATE_BASE)
    } else {
        u
    }
}

/// Whether `c` is a remapped lone surrogate.
#[inline]
pub fn is_surrogate(c: char) -> bool {
    c as u32 >= SURROGATE_BASE
}

/// Whether the UTF-8 at `b[i..]` starts with a char in U+10F800..=U+10FFFF
/// (`F4 8F A0..BF xx`), given `b[i] == 0xF4`.
#[inline]
fn reserved_at(b: &[u8], i: usize) -> bool {
    b.get(i + 1) == Some(&0x8F) && b.get(i + 2).is_some_and(|&x| x >= 0xA0)
}

/// Whether `s` holds a char in U+10F800..=U+10FFFF: a remapped surrogate, or
/// (in a `str` straight from Python) a reserved code point.
pub fn contains_reserved(s: &str) -> bool {
    let b = s.as_bytes();
    memchr::memchr_iter(0xF4, b).any(|i| reserved_at(b, i))
}

/// The chars of a `str` (no surrogates possible), refusing reserved code points.
pub fn chars_from_str(s: &str) -> Result<Vec<char>, InputError> {
    if contains_reserved(s) {
        return Err(InputError::ReservedCodePoint);
    }
    Ok(s.chars().collect())
}

/// Append the chars of `s` to `out`, refusing reserved code points.
fn push_checked(out: &mut Vec<char>, s: &str) -> Result<(), InputError> {
    if contains_reserved(s) {
        return Err(InputError::ReservedCodePoint);
    }
    out.extend(s.chars());
    Ok(())
}

/// The chars of Python's `str.encode("utf-8", "surrogatepass")` bytes, with lone
/// surrogates remapped; refuses reserved code points. `None` if `b` is not
/// generalized UTF-8 (UTF-8 that may also encode U+D800..U+DFFF, one surrogate
/// per 3-byte sequence: an encoded surrogate *pair* stays two chars, as in Python).
pub fn chars_from_wtf8(b: &[u8]) -> Option<Result<Vec<char>, InputError>> {
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    loop {
        let (valid, stop) = match std::str::from_utf8(&b[i..]) {
            Ok(s) => (s, None),
            Err(e) => {
                let v = e.valid_up_to();
                (std::str::from_utf8(&b[i..i + v]).ok()?, Some(i + v))
            }
        };
        if let Err(e) = push_checked(&mut out, valid) {
            return Some(Err(e));
        }
        let Some(j) = stop else {
            return Some(Ok(out));
        };
        // Not UTF-8 at `j`: only an encoded surrogate `ED A0..BF 80..BF` may follow.
        match b.get(j..j + 3) {
            Some(&[0xED, b1 @ 0xA0..=0xBF, b2 @ 0x80..=0xBF]) => {
                let cp = 0xD000 | (u32::from(b1 & 0x3F) << 6) | u32::from(b2 & 0x3F);
                out.push(surrogate_char(cp));
                i = j + 3;
            }
            _ => return None,
        }
    }
}

/// `s` as UTF-8 with remapped surrogates written back as surrogatepass bytes
/// (ED A0..BF xx); decode with Python's `"surrogatepass"` handler.
pub fn to_wtf8(s: &str) -> Vec<u8> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut run = 0;
    for i in memchr::memchr_iter(0xF4, b) {
        if reserved_at(b, i) {
            // `F4 8F xx yy` (U+10F800 + k) -> `ED xx yy` (U+D800 + k).
            out.extend_from_slice(&b[run..i]);
            out.push(0xED);
            out.extend_from_slice(&b[i + 2..i + 4]);
            run = i + 4;
        }
    }
    out.extend_from_slice(&b[run..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_versions() -> impl Iterator<Item = CharProps> {
        UnicodeVersion::ALL.into_iter().map(CharProps::new)
    }

    #[test]
    fn versions_round_trip() {
        for v in UnicodeVersion::ALL {
            assert_eq!(UnicodeVersion::from_unidata_version(v.as_str()), Some(v));
            assert_eq!(CharProps::new(v).version(), v);
        }
        assert_eq!(UnicodeVersion::from_unidata_version("13.0.0"), None);
    }

    #[test]
    fn spot_checks() {
        for p in all_versions() {
            // isspace: the C0 separators count, zero-width space does not.
            for c in [
                '\x1c', '\x1f', '\x0b', '\x0c', '\u{85}', '\u{a0}', '\u{3000}', '\u{2028}',
            ] {
                assert!(p.is_space(c), "{c:?}");
            }
            for c in ['\u{200b}', '\u{feff}', '\u{180e}', 'a', '\0'] {
                assert!(!p.is_space(c), "{c:?}");
            }
            // isdigit is wider than isdecimal; isalnum also takes isnumeric.
            assert!(p.is_digit('²') && p.is_alnum('²') && !p.is_alpha('²'));
            assert!(p.is_digit('٣') && p.is_alnum('٣'));
            assert!(!p.is_digit('½') && p.is_alnum('½') && !p.is_alpha('½'));
            assert!(!p.is_digit('Ⅻ') && p.is_alnum('Ⅻ'));
            assert!(p.is_alpha('ǅ') && p.is_alpha('é') && p.is_alpha('中') && p.is_alnum('ß'));
            assert!(!p.is_alpha('_') && !p.is_alnum('_') && !p.is_alpha('\u{301}'));
            assert!(!p.is_alpha('😀') && !p.is_alnum('😀') && p.is_alpha('𝐀'));
            for c in ['a', 'Z'] {
                assert!(p.is_alpha(c) && p.is_alnum(c) && !p.is_digit(c));
            }
            assert!(p.is_digit('7') && p.is_alnum('7') && !p.is_alpha('7'));
        }
    }

    #[test]
    fn surrogates_and_reserved_range_are_nothing() {
        for p in all_versions() {
            for cp in [0xD800, 0xDBFF, 0xDC00, 0xDFFF] {
                let c = surrogate_char(cp);
                assert!(is_surrogate(c) && code_point(c) == cp);
                assert!(!p.is_space(c) && !p.is_alpha(c) && !p.is_digit(c) && !p.is_alnum(c));
            }
        }
        assert_eq!(code_point('a'), 0x61);
        assert!(!is_surrogate('\u{10f7ff}'));
    }

    #[test]
    fn is_space_matches_the_table() {
        for cp in 0..0x11_0000 {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            let p = CharProps::new(UnicodeVersion::V17_0);
            assert_eq!(p.is_space(c), in_ranges(&tables::SPACE, cp), "{cp:#x}");
        }
    }

    #[test]
    fn version_differences() {
        // U+1E4F0 NAG MUNDARI DIGIT ZERO: new in Unicode 15.0.
        let c = '\u{1e4f0}';
        assert!(!CharProps::new(UnicodeVersion::V14_0).is_digit(c));
        assert!(CharProps::new(UnicodeVersion::V15_0).is_digit(c));
        // U+31350 CJK EXTENSION H: new in Unicode 15.0.
        assert!(!CharProps::new(UnicodeVersion::V14_0).is_alpha('\u{31350}'));
        assert!(CharProps::new(UnicodeVersion::V15_0).is_alpha('\u{31350}'));
    }

    /// Every code point against a `generate_tables.py oracle` dump for one interpreter.
    #[test]
    fn full_range_oracle() {
        let Ok(path) = std::env::var("PYROID_JSONREPAIR_CHARS_ORACLE") else {
            return;
        };
        let text = std::fs::read_to_string(&path).expect("read oracle");
        let (version, digits) = text.split_once('\n').expect("header line");
        let p = CharProps::new(UnicodeVersion::from_unidata_version(version).expect("version"));
        let digits = digits.trim_end().as_bytes();
        assert_eq!(digits.len(), 0x11_0000);
        let mut checked = 0;
        for (cp, &d) in (0u32..).zip(digits) {
            let bits = char::from(d).to_digit(16).expect("hex digit");
            // Python's answer for a real surrogate is what the kernel must give
            // for its remapped char; the chars that stand for them are skipped.
            let c = match cp {
                0xD800..=0xDFFF => surrogate_char(cp),
                SURROGATE_BASE..=0x10_FFFF => continue,
                _ => char::from_u32(cp).expect("scalar value"),
            };
            let got = u32::from(p.is_space(c))
                | (u32::from(p.is_alpha(c)) << 1)
                | (u32::from(p.is_digit(c)) << 2)
                | (u32::from(p.is_alnum(c)) << 3);
            assert_eq!(got, bits, "U+{cp:04X} on Unicode {version}");
            checked += 1;
        }
        // The plane-16 code points standing for surrogates are private use: all false.
        assert!(digits[SURROGATE_BASE as usize..].iter().all(|&d| d == b'0'));
        assert_eq!(checked, 0x11_0000 - 0x800);
    }

    #[test]
    fn str_input() {
        assert_eq!(chars_from_str("aé😀"), Ok(vec!['a', 'é', '😀']));
        assert_eq!(chars_from_str("\u{10f7ff}"), Ok(vec!['\u{10f7ff}']));
        assert_eq!(
            chars_from_str("x\u{10f800}"),
            Err(InputError::ReservedCodePoint)
        );
        assert_eq!(
            chars_from_str("\u{10ffff}"),
            Err(InputError::ReservedCodePoint)
        );
        assert!(contains_reserved("\u{10fabc}") && !contains_reserved("\u{10000}\u{fffff}"));
    }

    #[test]
    fn wtf8_round_trip() {
        // "a😀b".encode("utf-8", "surrogatepass"): a pair encoded as two
        // surrogates stays two code points.
        let b = b"a\xed\xa0\xbd\xed\xb8\x80b";
        let want = vec!['a', surrogate_char(0xD83D), surrogate_char(0xDE00), 'b'];
        assert_eq!(chars_from_wtf8(b), Some(Ok(want.clone())));
        let s: String = want.iter().collect();
        assert_eq!(to_wtf8(&s), b);
        for cp in [0xD800, 0xDBFF, 0xDC00, 0xDFFF] {
            let s = format!("é{}😀{}", surrogate_char(cp), surrogate_char(cp));
            let w = to_wtf8(&s);
            assert_eq!(w.len(), s.len() - 2);
            let back: String = chars_from_wtf8(&w).unwrap().unwrap().into_iter().collect();
            assert_eq!(back, s);
        }
        assert_eq!(chars_from_wtf8(b""), Some(Ok(vec![])));
        assert_eq!(to_wtf8("plain é"), "plain é".as_bytes());
        assert_eq!(to_wtf8("\u{10f7ff}"), "\u{10f7ff}".as_bytes());
    }

    #[test]
    fn wtf8_rejects() {
        for bad in [
            &b"\xff"[..],
            b"\xc0\x80",         // overlong
            b"\xe0\x80\x80",     // overlong
            b"\xed\xa0",         // truncated surrogate
            b"\xed\x9f",         // truncated
            b"\xed\xc0\x80",     // bad continuation
            b"\xf4\x90\x80\x80", // > U+10FFFF
            b"a\x80",
            b"\xed\xa0\x80\xe2", // surrogate then truncated
        ] {
            assert_eq!(chars_from_wtf8(bad), None, "{bad:x?}");
        }
        assert_eq!(
            chars_from_wtf8("\u{10f800}".as_bytes()),
            Some(Err(InputError::ReservedCodePoint))
        );
        assert_eq!(
            chars_from_wtf8(b"\xed\xa0\x80\xf4\x8f\xbf\xbf"),
            Some(Err(InputError::ReservedCodePoint))
        );
        assert_eq!(chars_from_wtf8(b"\xed\x9f\xbf"), Some(Ok(vec!['\u{d7ff}'])));
    }
}
