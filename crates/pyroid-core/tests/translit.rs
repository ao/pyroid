//! Integration tests of the public `pyroid_core::translit` API.
//!
//! Every code point is checked in each [`Missing`] mode against the
//! `any_ascii` crate and [`has_transliteration`]. Random strings (an inline
//! xorshift generator with fixed seeds, so failures reproduce) check that the
//! kernels work character by character, report the first missing character's
//! code-point index, and that the WTF-8 entry points (for Python strs holding
//! lone surrogates) agree with the str ones and treat a lone surrogate as a
//! character without a transliteration. Every result must be `Cow::Borrowed`
//! of the whole input exactly when the input is pure ASCII (as Unidecode
//! returns its input object only for ASCII input).

use std::borrow::Cow;
use std::fmt::Debug;

use pyroid_core::translit::{
    Missing, MissingAt, MissingWtf8, Unit, anyascii, anyascii_char, anyascii_wtf8,
    ascii_prefix_len, first_missing, first_missing_wtf8, has_transliteration, is_ascii,
    surrogates_wtf8, unidecode, unidecode_wtf8,
};

/// A replacement that no transliteration contains (and that the random inputs
/// never hold), so replacements can be counted.
const MARK: &str = "\u{1}";

/// Every Unicode scalar value.
fn all_chars() -> impl Iterator<Item = char> {
    (0..=0x10_FFFF).filter_map(char::from_u32)
}

/// `str` and `[u8]` inputs, for [`assert_cow`].
trait Input: ToOwned + PartialEq + Debug {
    fn is_all_ascii(&self) -> bool;
}

impl Input for str {
    fn is_all_ascii(&self) -> bool {
        self.is_ascii()
    }
}

impl Input for [u8] {
    fn is_all_ascii(&self) -> bool {
        self.is_ascii()
    }
}

/// `out` is `Cow::Borrowed` of the whole `input` exactly when `input` is pure
/// ASCII (and then equals it).
#[allow(clippy::ptr_arg, reason = "the Cow variant is what is checked")]
fn assert_cow<B>(input: &B, out: &Cow<'_, B>, what: &str)
where
    B: ?Sized + Input,
{
    match out {
        Cow::Borrowed(x) => assert!(
            std::ptr::eq(*x, input) && input.is_all_ascii(),
            "{what}: borrowed, but the input {input:?} is not pure ASCII"
        ),
        Cow::Owned(_) => assert!(
            !input.is_all_ascii(),
            "{what}: an owned copy of the pure-ASCII input {input:?}"
        ),
    }
}

/// anyascii's transliteration of one character, from the crate itself.
fn crate_ascii(c: char) -> String {
    any_ascii::any_ascii(c.encode_utf8(&mut [0; 4]))
}

fn missing_char(c: char) -> MissingAt {
    MissingAt {
        index: 0,
        unit: Unit::Char(c),
    }
}

#[test]
fn every_char_in_every_mode() {
    let (mut mapped, mut deliberately_empty) = (0_usize, 0_usize);
    for c in all_chars() {
        let s = c.to_string();
        let ascii = crate_ascii(c);
        let has = has_transliteration(c);
        let id = u32::from(c);
        assert_eq!(anyascii_char(c), ascii, "U+{id:04X}");
        assert_eq!(anyascii(&s), ascii, "U+{id:04X}");
        assert_eq!(unidecode(&s, Missing::Drop), ascii, "U+{id:04X}");
        let replaced = if has { ascii.as_str() } else { MARK };
        assert_eq!(
            unidecode(&s, Missing::Replace(MARK)),
            replaced,
            "U+{id:04X}"
        );
        assert_eq!(unidecode(&s, Missing::Replace("")), ascii, "U+{id:04X}");
        let kept = if has { ascii.as_str() } else { s.as_str() };
        assert_eq!(unidecode(&s, Missing::Keep), kept, "U+{id:04X}");
        assert_eq!(
            first_missing(&s),
            (!has).then(|| missing_char(c)),
            "U+{id:04X}"
        );
        // A character has a transliteration when it is ASCII or maps to a
        // non-empty string; otherwise only if anyascii lists it as "".
        if c.is_ascii() || !ascii.is_empty() {
            assert!(has, "U+{id:04X}");
        } else if has {
            deliberately_empty += 1;
        }
        if has && !c.is_ascii() {
            mapped += 1;
        }
        assert!(c.is_ascii() || !ascii.contains(MARK), "U+{id:04X}");
        assert_eq!(is_ascii(&s), c.is_ascii());
        assert_eq!(ascii_prefix_len(s.as_bytes()), usize::from(c.is_ascii()));
        check_str_kernels(&s);
    }
    // The counts the module docs give (python/pyroid/translit.pyi).
    assert_eq!(mapped, 123_799);
    assert_eq!(deliberately_empty, 1_192);
}

#[test]
fn documented_examples() {
    assert_eq!(anyascii("\u{5317}\u{4eac}"), "BeiJing");
    assert_eq!(unidecode("\u{1f600}", Missing::Drop), ":grinning:");
    for c in ['\u{301}', '\u{200d}', '\u{fe0f}', '\u{640}', '\u{1f3fb}'] {
        assert!(has_transliteration(c), "U+{:04X}", u32::from(c));
        assert_eq!(unidecode(&format!("a{c}b"), Missing::Replace("?")), "ab");
    }
    for c in ['\u{e000}', '\u{f8ff}', '\u{378}', '\u{f0000}', '\u{10fffd}'] {
        assert!(!has_transliteration(c), "U+{:04X}", u32::from(c));
    }
    // Positions are in code points, not bytes, after ASCII runs of any length.
    assert_eq!(
        first_missing("\u{5317}\u{4eac}\u{e000}").map(|m| m.index),
        Some(2)
    );
    let long = format!("{}\u{e9}\u{e000}", "a".repeat(37));
    assert_eq!(first_missing(&long).map(|m| m.index), Some(38));
}

#[test]
fn replacing_a_missing_char_by_itself_gives_an_equal_owned_copy() {
    // The output equals the input but is a new string, as Unidecode returns a
    // new str for any non-ASCII input.
    let s = "a\u{e000}";
    let wtf8 = unidecode_wtf8(s.as_bytes(), MissingWtf8::Replace("\u{e000}".as_bytes()));
    assert_cow(s.as_bytes(), &wtf8, "unidecode_wtf8 Replace(itself)");
    let out = unidecode(s, Missing::Replace("\u{e000}"));
    assert_eq!(out, s);
    assert_cow(s, &out, "unidecode Replace(itself)");
}

// ===================================================================== random strings

/// xorshift64* (Vigna): small, fast, and deterministic for a given seed.
#[derive(Debug)]
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// A number in `0..n` (modulo bias does not matter here).
    fn below(&mut self, n: usize) -> usize {
        let n = u64::try_from(n).expect("usize fits in u64");
        usize::try_from(self.next_u64() % n).expect("below n")
    }

    fn pick<T: Copy>(&mut self, items: &[T]) -> T {
        items[self.below(items.len())]
    }
}

/// Mapped characters of one to four UTF-8 bytes, some mapping to several.
const MAPPED: &[char] = &[
    '\u{e9}',
    '\u{df}',
    '\u{416}',
    '\u{5317}',
    '\u{4eac}',
    '\u{1f600}',
    '\u{fb01}',
    '\u{2460}',
    '\u{a0}',
    '\u{3a9}',
    '\u{ac00}',
    '\u{1d400}',
];
/// Characters anyascii maps to "" on purpose: never missing.
const EMPTY: &[char] = &['\u{301}', '\u{200d}', '\u{fe0f}', '\u{640}', '\u{1f3fb}'];
/// Characters without a transliteration.
const MISSING: &[char] = &[
    '\u{e000}',
    '\u{f8ff}',
    '\u{378}',
    '\u{ffff}',
    '\u{f0000}',
    '\u{10fffd}',
];

/// One code unit of a Python str: a character or a lone surrogate.
#[derive(Clone, Copy, Debug)]
enum Item {
    Char(char),
    Surrogate(u16),
}

fn random_char(rng: &mut Rng) -> char {
    match rng.below(6) {
        0 | 1 => char::from(b' ' + u8::try_from(rng.below(95)).expect("< 95")),
        2 => rng.pick(MAPPED),
        3 => rng.pick(EMPTY),
        4 => rng.pick(MISSING),
        _ => loop {
            let cp = u32::try_from(rng.below(0x11_0000)).expect("< 0x110000");
            match char::from_u32(cp) {
                Some(c) if c != '\u{1}' => break c,
                _ => {}
            }
        },
    }
}

/// Up to `max` items, sometimes with ASCII runs long enough for the kernels'
/// word-at-a-time path, and lone surrogates if `surrogates`.
fn random_items(rng: &mut Rng, max: usize, surrogates: bool) -> Vec<Item> {
    let mut items = Vec::new();
    for _ in 0..rng.below(max + 1) {
        match rng.below(10) {
            0 => {
                for _ in 0..8 + rng.below(33) {
                    items.push(Item::Char(char::from(
                        b'a' + u8::try_from(rng.below(26)).expect("< 26"),
                    )));
                }
            }
            1 if surrogates => {
                let unit = 0xD800 + u16::try_from(rng.below(0x800)).expect("< 0x800");
                items.push(Item::Surrogate(unit));
            }
            _ => items.push(Item::Char(random_char(rng))),
        }
    }
    items
}

/// A lone surrogate as Python's `"surrogatepass"` UTF-8 encodes it.
fn surrogate_bytes(unit: u16) -> [u8; 3] {
    let [hi, lo] = unit.to_be_bytes();
    [
        0xE0 | (hi >> 4),
        0x80 | ((hi & 0x0F) << 2) | (lo >> 6),
        0x80 | (lo & 0x3F),
    ]
}

fn wtf8(items: &[Item]) -> Vec<u8> {
    let mut out = Vec::new();
    for item in items {
        match *item {
            Item::Char(c) => out.extend_from_slice(c.encode_utf8(&mut [0; 4]).as_bytes()),
            Item::Surrogate(unit) => out.extend_from_slice(&surrogate_bytes(unit)),
        }
    }
    out
}

/// What [`unidecode_wtf8`] must give, built one item at a time from the
/// `any_ascii` crate and [`has_transliteration`].
fn reference(items: &[Item], missing: MissingWtf8<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    for item in items {
        let raw = match *item {
            Item::Char(c) if has_transliteration(c) => {
                out.extend_from_slice(crate_ascii(c).as_bytes());
                continue;
            }
            Item::Char(c) => c.to_string().into_bytes(),
            Item::Surrogate(unit) => surrogate_bytes(unit).to_vec(),
        };
        match missing {
            MissingWtf8::Drop => {}
            MissingWtf8::Replace(r) => out.extend_from_slice(r),
            MissingWtf8::Keep => out.extend_from_slice(&raw),
        }
    }
    out
}

/// The first item without a transliteration (a lone surrogate never has one).
fn reference_first_missing(items: &[Item]) -> Option<MissingAt> {
    items
        .iter()
        .enumerate()
        .find_map(|(index, item)| match *item {
            Item::Char(c) if has_transliteration(c) => None,
            Item::Char(c) => Some(MissingAt {
                index,
                unit: Unit::Char(c),
            }),
            Item::Surrogate(unit) => Some(MissingAt {
                index,
                unit: Unit::Surrogate(unit),
            }),
        })
}

/// How many times [`MARK`] was put in.
fn count_marks(out: &[u8]) -> usize {
    memchr::memchr_iter(MARK.as_bytes()[0], out).count()
}

fn count_missing(items: &[Item]) -> usize {
    items
        .iter()
        .filter(|item| match **item {
            Item::Char(c) => !has_transliteration(c),
            Item::Surrogate(_) => true,
        })
        .count()
}

/// The str kernels on `s` equal the WTF-8 kernels on its bytes, and keep the
/// `Cow` promise.
fn check_str_kernels(s: &str) {
    let b = s.as_bytes();
    let out = anyascii(s);
    assert_cow(s, &out, "anyascii");
    let out_wtf8 = anyascii_wtf8(b);
    assert_cow(b, &out_wtf8, "anyascii_wtf8");
    assert_eq!(out_wtf8.as_ref(), out.as_bytes());
    for (mode, mode_wtf8) in [
        (Missing::Drop, MissingWtf8::Drop),
        (
            Missing::Replace(MARK),
            MissingWtf8::Replace(MARK.as_bytes()),
        ),
        (Missing::Keep, MissingWtf8::Keep),
    ] {
        let out = unidecode(s, mode);
        let out_wtf8 = unidecode_wtf8(b, mode_wtf8);
        assert_eq!(out_wtf8.as_ref(), out.as_bytes(), "{mode:?} on {s:?}");
        assert_cow(b, &out_wtf8, "unidecode_wtf8");
        assert_cow(s, &out, "unidecode");
    }
    assert_eq!(first_missing_wtf8(b), first_missing(s));
    assert_eq!(surrogates_wtf8(b), Vec::new());
}

#[test]
fn random_strs() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let replacements = ["", MARK, "<\u{e9}>", "\u{e000}", "\u{1f600}"];
    for _ in 0..20_000 {
        let items = random_items(&mut rng, 24, false);
        let s = String::from_utf8(wtf8(&items)).expect("no surrogates");
        assert_eq!(anyascii(&s), any_ascii::any_ascii(&s));
        for r in replacements {
            let out = unidecode(&s, Missing::Replace(r));
            let want = reference(&items, MissingWtf8::Replace(r.as_bytes()));
            assert_eq!(out.as_bytes(), want, "Replace({r:?}) on {s:?}");
        }
        for (mode, mode_wtf8) in [
            (Missing::Drop, MissingWtf8::Drop),
            (Missing::Keep, MissingWtf8::Keep),
        ] {
            let want = reference(&items, mode_wtf8);
            assert_eq!(unidecode(&s, mode).as_bytes(), want, "{mode:?} on {s:?}");
        }
        let marks = unidecode(&s, Missing::Replace(MARK)).matches(MARK).count();
        assert_eq!(marks, count_missing(&items), "{s:?}");
        assert_eq!(first_missing(&s), reference_first_missing(&items), "{s:?}");
        assert_eq!(is_ascii(&s), s.bytes().all(|x| x < 0x80));
        let naive = s.bytes().position(|x| x >= 0x80).unwrap_or(s.len());
        assert_eq!(ascii_prefix_len(s.as_bytes()), naive);
        check_str_kernels(&s);
    }
}

#[test]
fn random_strs_with_lone_surrogates() {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let surrogate = surrogate_bytes(0xD83D);
    let replacements: [&[u8]; 5] = [
        b"",
        MARK.as_bytes(),
        "<\u{e9}>".as_bytes(),
        &surrogate,
        b"??",
    ];
    for _ in 0..20_000 {
        let items = random_items(&mut rng, 24, true);
        let b = wtf8(&items);
        let out = anyascii_wtf8(&b);
        assert_cow(b.as_slice(), &out, "anyascii_wtf8");
        assert_eq!(
            out.as_ref(),
            reference(&items, MissingWtf8::Drop),
            "{items:?}"
        );
        let modes = replacements
            .iter()
            .map(|r| MissingWtf8::Replace(r))
            .chain([MissingWtf8::Drop, MissingWtf8::Keep]);
        for mode in modes {
            let out = unidecode_wtf8(&b, mode);
            assert_cow(b.as_slice(), &out, "unidecode_wtf8");
            assert_eq!(
                out.as_ref(),
                reference(&items, mode),
                "{mode:?} on {items:?}"
            );
        }
        let replaced = unidecode_wtf8(&b, MissingWtf8::Replace(MARK.as_bytes()));
        let marks = count_marks(&replaced);
        assert_eq!(marks, count_missing(&items), "{items:?}");
        assert_eq!(
            first_missing_wtf8(&b),
            reference_first_missing(&items),
            "{items:?}"
        );
        let surrogates: Vec<(usize, u16)> = items
            .iter()
            .enumerate()
            .filter_map(|(i, item)| match *item {
                Item::Surrogate(unit) => Some((i, unit)),
                Item::Char(_) => None,
            })
            .collect();
        assert_eq!(surrogates_wtf8(&b), surrogates, "{items:?}");
    }
}

#[test]
fn a_surrogate_pair_in_a_str_is_two_lone_surrogates() {
    // A Python str holding the code units U+D83D U+DE00 (two lone surrogates,
    // not the character U+1F600), surrogatepass-encoded.
    let b = [surrogate_bytes(0xD83D), surrogate_bytes(0xDE00)].concat();
    assert_eq!(anyascii_wtf8(&b).as_ref(), b"");
    assert_eq!(
        unidecode_wtf8(&b, MissingWtf8::Replace(b"?")).as_ref(),
        b"??"
    );
    let kept = unidecode_wtf8(&b, MissingWtf8::Keep);
    assert_cow(b.as_slice(), &kept, "Keep");
    assert_eq!(surrogates_wtf8(&b), vec![(0, 0xD83D), (1, 0xDE00)]);
    assert_eq!(
        first_missing_wtf8(&b),
        Some(MissingAt {
            index: 0,
            unit: Unit::Surrogate(0xD83D)
        })
    );
}

#[test]
fn arbitrary_bytes_never_panic() {
    // Not WTF-8 in general: a byte that is not part of a character or a
    // surrogate counts as one missing unit.
    let mut rng = Rng(0x2545_F491_4F6C_DD1D);
    let fragments: [&[u8]; 8] = [
        b"a",
        b"abcdefghij",
        "\u{e9}".as_bytes(),
        "\u{e000}".as_bytes(),
        &[0xED, 0xA0, 0x80],
        &[0xF0, 0x9F],
        &[0xC3],
        &[0xFF],
    ];
    for _ in 0..20_000 {
        let mut b = Vec::new();
        for _ in 0..rng.below(12) {
            if rng.below(3) == 0 {
                let byte = u8::try_from(rng.below(256)).expect("< 256");
                b.push(if byte == 1 { 0 } else { byte });
            } else {
                b.extend_from_slice(rng.pick(&fragments));
            }
        }
        let dropped = anyascii_wtf8(&b);
        assert!(dropped.is_ascii(), "{b:?}");
        assert_eq!(unidecode_wtf8(&b, MissingWtf8::Drop), dropped, "{b:?}");
        let replaced = unidecode_wtf8(&b, MissingWtf8::Replace(MARK.as_bytes()));
        assert!(replaced.is_ascii(), "{b:?}");
        assert_eq!(
            first_missing_wtf8(&b).is_some(),
            replaced.contains(&1),
            "{b:?}"
        );
        assert!(surrogates_wtf8(&b).len() <= count_marks(&replaced));
        let kept = unidecode_wtf8(&b, MissingWtf8::Keep);
        assert_cow(b.as_slice(), &kept, "Keep");
    }
}
