//! CPython's character predicates, per Unicode version.
//!
//! The filters must make the same decisions as the Python reference on the
//! caller's CPython, whose `unicodedata` version decides what counts as a
//! letter, a digit or punctuation. Each predicate reproduces one CPython test
//! exactly, from tables dumped from each interpreter by
//! `generate_unicode_tables.py`:
//!
//! | function     | CPython                                       |
//! |--------------|-----------------------------------------------|
//! | [`is_alpha`] | `str.isalpha()`                               |
//! | [`is_punct`] | `unicodedata.category(c).startswith("P")`     |
//! | [`is_word`]  | `re`'s `\w` (str pattern, default flags)      |
//! | [`is_digit`] | `re`'s `\d` (str pattern, default flags)      |
//! | [`is_space`] | `str.isspace()` (the same in every version)   |
//!
//! ASCII is answered from a 128-bit mask; other code points by a binary
//! search over sorted inclusive ranges.

use crate::ftfy::UnicodeVersion;

#[rustfmt::skip]
#[path = "unicode_tables.rs"]
mod tables;

/// Whether `cp` lies in one of the sorted, disjoint inclusive `ranges`.
#[inline]
fn in_ranges(ranges: &[(u32, u32)], cp: u32) -> bool {
    let after = ranges.partition_point(|&(lo, _)| lo <= cp);
    ranges
        .get(after.wrapping_sub(1))
        .is_some_and(|&(_, hi)| cp <= hi)
}

/// One property: bit `cp` of `ascii` below U+0080, else the version's ranges.
#[inline]
fn has(
    c: char,
    v: UnicodeVersion,
    ascii: u128,
    ranges: impl FnOnce(UnicodeVersion) -> &'static [(u32, u32)],
) -> bool {
    let cp = u32::from(c);
    if cp < 0x80 {
        (ascii >> cp) & 1 != 0
    } else {
        in_ranges(ranges(v), cp)
    }
}

/// `c.isalpha()`: general category Lu, Ll, Lt, Lm or Lo.
#[inline]
pub(crate) fn is_alpha(c: char, v: UnicodeVersion) -> bool {
    has(c, v, tables::ALPHA_ASCII, tables::alpha)
}

/// `unicodedata.category(c).startswith("P")`: categories Pc, Pd, Ps, Pe, Pi,
/// Pf and Po. In ASCII that is ``!"#%&'()*,-./:;?@[\]_{}``; ``$+<=>^`|~`` are
/// symbols (S*), not punctuation.
#[inline]
pub(crate) fn is_punct(c: char, v: UnicodeVersion) -> bool {
    has(c, v, tables::PUNCT_ASCII, tables::punct)
}

/// `re.match(r"\w", c)`: `c.isalnum()` or `_` (so letters, every numeric
/// character including `²` and `Ⅰ`, and `_`).
#[inline]
pub(crate) fn is_word(c: char, v: UnicodeVersion) -> bool {
    has(c, v, tables::WORD_ASCII, tables::word)
}

/// `re.match(r"\d", c)`: `c.isdecimal()` (decimal digits only, so not `²`).
#[inline]
pub(crate) fn is_digit(c: char, v: UnicodeVersion) -> bool {
    has(c, v, tables::DIGIT_ASCII, tables::digit)
}

/// `c.isspace()`, which is the same set in every supported version and the
/// same as [`crate::text::is_py_whitespace`] (the tests check both).
#[inline]
pub(crate) fn is_space(c: char) -> bool {
    crate::text::is_py_whitespace(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [UnicodeVersion; 5] = [
        UnicodeVersion::V14_0,
        UnicodeVersion::V15_0,
        UnicodeVersion::V15_1,
        UnicodeVersion::V16_0,
        UnicodeVersion::V17_0,
    ];

    type Pred = fn(char, UnicodeVersion) -> bool;
    const PREDS: [(&str, Pred); 4] = [
        ("alpha", is_alpha),
        ("punct", is_punct),
        ("word", is_word),
        ("digit", is_digit),
    ];

    fn all_chars() -> impl Iterator<Item = char> {
        (0..=0x10_FFFF).filter_map(char::from_u32)
    }

    fn idx(c: char) -> usize {
        usize::try_from(u32::from(c)).unwrap()
    }

    /// `(char, alpha, punct, word, digit)`, the same in every version
    /// (checked against CPython 3.11-3.15).
    #[rustfmt::skip]
    const STABLE: &[(char, bool, bool, bool, bool)] = &[
        ('A', true, false, true, false),
        ('z', true, false, true, false),
        ('7', false, false, true, true),
        ('_', false, true, true, false),        // Pc: punctuation and \w
        ('!', false, true, false, false),
        ('@', false, true, false, false),
        ('$', false, false, false, false),      // Sc
        ('+', false, false, false, false),      // Sm
        ('`', false, false, false, false),      // Sk
        (' ', false, false, false, false),
        ('\0', false, false, false, false),
        ('\u{7f}', false, false, false, false),
        ('é', true, false, true, false),
        ('ß', true, false, true, false),
        ('一', true, false, true, false),       // U+4E00
        ('ª', true, false, true, false),        // feminine ordinal, Lo
        ('º', true, false, true, false),        // masculine ordinal, Lo
        ('\u{2160}', false, false, true, false), // Ⅰ, Nl: \w but not alpha
        ('\u{b2}', false, false, true, false),  // ², No: \w but not \d
        ('\u{660}', false, false, true, true),  // Arabic-Indic digit zero
        ('\u{6f9}', false, false, true, true),  // extended Arabic-Indic digit nine
        ('\u{a7}', false, true, false, false),  // §, Po
        ('\u{2019}', false, true, false, false), // ’, Pf
        ('\u{a0}', false, false, false, false), // no-break space
        ('\u{1f600}', false, false, false, false), // 😀, So
    ];

    #[test]
    fn stable_spot_checks() {
        for &(c, alpha, punct, word, digit) in STABLE {
            for v in ALL {
                let got = (
                    is_alpha(c, v),
                    is_punct(c, v),
                    is_word(c, v),
                    is_digit(c, v),
                );
                assert_eq!(
                    got,
                    (alpha, punct, word, digit),
                    "{c:?} (U+{:04X}) {v:?}",
                    u32::from(c)
                );
            }
        }
    }

    /// `(char, property, first version that has it)`: unassigned before that.
    #[rustfmt::skip]
    const NEW: &[(char, &str, UnicodeVersion)] = &[
        ('\u{11f50}', "digit", UnicodeVersion::V15_0), // KAWI DIGIT ZERO
        ('\u{11f50}', "word", UnicodeVersion::V15_0),
        ('\u{2ebf0}', "alpha", UnicodeVersion::V15_1), // CJK UNIFIED IDEOGRAPH-2EBF0 (Ext. I)
        ('\u{1c89}', "alpha", UnicodeVersion::V16_0),  // a Cyrillic capital letter
        ('\u{1c89}', "word", UnicodeVersion::V16_0),
        ('\u{10d40}', "digit", UnicodeVersion::V16_0), // GARAY DIGIT ZERO
        ('\u{1b4e}', "punct", UnicodeVersion::V16_0),  // BALINESE INVERTED CARIK SIKI
        ('\u{a7ce}', "alpha", UnicodeVersion::V17_0),  // LATIN CAPITAL LETTER PHARYNGEAL VOICED FRICATIVE
        ('\u{a7ce}', "word", UnicodeVersion::V17_0),
        ('\u{11de0}', "digit", UnicodeVersion::V17_0), // TOLONG SIKI DIGIT ZERO
        ('\u{10ed0}', "punct", UnicodeVersion::V17_0), // ARABIC BIBLICAL END OF VERSE
    ];

    #[test]
    fn characters_new_in_a_version_flip_there() {
        for &(c, prop, since) in NEW {
            let (_, pred) = PREDS.iter().find(|(name, _)| *name == prop).unwrap();
            for v in ALL {
                assert_eq!(
                    pred(c, v),
                    v >= since,
                    "{prop} {c:?} (U+{:04X}) {v:?}",
                    u32::from(c)
                );
                if v < since {
                    // Unassigned: no property at all.
                    assert!(PREDS.iter().all(|(_, p)| !p(c, v)), "{c:?} {v:?}");
                }
            }
        }
    }

    #[test]
    fn space_is_cpython_isspace_and_py_whitespace() {
        for c in all_chars() {
            assert_eq!(
                is_space(c),
                in_ranges(&tables::SPACE, u32::from(c)),
                "{c:?}"
            );
        }
        assert!(is_space('\u{1c}') && is_space('\u{3000}') && !is_space('\u{200b}'));
    }

    #[test]
    fn tables_are_sorted_disjoint_and_above_ascii() {
        for (prop, ranges) in [
            (
                "alpha",
                tables::alpha as fn(UnicodeVersion) -> &'static [(u32, u32)],
            ),
            ("punct", tables::punct),
            ("word", tables::word),
            ("digit", tables::digit),
        ] {
            for v in ALL {
                let t = ranges(v);
                assert!(t.first().is_some_and(|&(lo, _)| lo >= 0x80), "{prop} {v:?}");
                assert!(
                    t.iter().all(|&(lo, hi)| lo <= hi && hi <= 0x10_FFFF),
                    "{prop} {v:?}"
                );
                // Strictly increasing and never adjacent (the generator merges runs).
                assert!(t.windows(2).all(|w| w[0].1 + 1 < w[1].0), "{prop} {v:?}");
            }
        }
    }

    #[test]
    fn in_ranges_edges() {
        let t = [(0x80, 0x80), (0x100, 0x1FF)];
        let hits: Vec<u32> = [
            0,
            0x7F,
            0x80,
            0x81,
            0xFF,
            0x100,
            0x150,
            0x1FF,
            0x200,
            u32::MAX,
        ]
        .into_iter()
        .filter(|&cp| in_ranges(&t, cp))
        .collect();
        assert_eq!(hits, [0x80, 0x100, 0x150, 0x1FF]);
        assert!(!in_ranges(&[], 0x80));
    }

    /// Every code point in every version against the generator's JSON dumps:
    /// `PYROID_UNICODE_DUMPS=/target/gen-filters cargo test -p pyroid-core
    /// filters::unicode -- --ignored`.
    #[test]
    #[ignore = "needs the generate_unicode_tables.py dumps in $PYROID_UNICODE_DUMPS"]
    fn matches_cpython_dumps() {
        let dir = std::env::var("PYROID_UNICODE_DUMPS").expect("set PYROID_UNICODE_DUMPS");
        let mut seen = Vec::new();
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_none_or(|e| e != "json") {
                continue;
            }
            let text = std::fs::read_to_string(&path).unwrap();
            let dump: serde_json::Value = serde_json::from_str(&text).unwrap();
            let v =
                UnicodeVersion::from_unidata_version(dump["unicode"].as_str().unwrap()).unwrap();
            let expand = |name: &str| {
                let mut hit = vec![false; 0x11_0000];
                for r in dump["props"][name].as_array().unwrap() {
                    let lo = usize::try_from(r[0].as_u64().unwrap()).unwrap();
                    let hi = usize::try_from(r[1].as_u64().unwrap()).unwrap();
                    hit[lo..=hi].fill(true);
                }
                hit
            };
            for (name, pred) in PREDS {
                let want = expand(name);
                for c in all_chars() {
                    assert_eq!(pred(c, v), want[idx(c)], "{name} {c:?} {v:?}");
                }
            }
            let want = expand("space");
            assert!(
                all_chars().all(|c| is_space(c) == want[idx(c)]),
                "space {v:?}"
            );
            seen.push(v);
        }
        seen.sort();
        assert_eq!(seen, ALL);
    }
}
