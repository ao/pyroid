use super::*;

fn chars() -> impl Iterator<Item = char> {
    (0..=0x10_FFFF).filter_map(char::from_u32)
}

#[test]
fn table_agrees_with_the_crate_data() {
    // Every deliberately empty code point maps to "" in the crate's data, and
    // the code points with a transliteration are exactly the ones the table
    // lists (generate_tables.py checks the table against the PyPI package).
    for &(lo, hi) in tables::EMPTY {
        assert!(lo <= hi);
        for cp in lo..=hi {
            let c = char::from_u32(cp).expect("EMPTY holds scalar values");
            assert_eq!(anyascii_char(c), "", "U+{cp:04X}");
        }
    }
    assert!(tables::EMPTY.windows(2).all(|w| w[0].1 + 1 < w[1].0));
    let listed = chars()
        .filter(|c| !c.is_ascii() && has_transliteration(*c))
        .count();
    assert_eq!(listed, tables::LISTED_COUNT);
}

#[test]
fn ascii_prefix_len_matches_naive() {
    for len in 0..40 {
        for pos in 0..=len {
            let mut v = vec![b'a'; len];
            if pos < len {
                v[pos] = 0xC3;
            }
            assert_eq!(ascii_prefix_len(&v), pos, "len {len} pos {pos}");
        }
    }
}

#[test]
fn anyascii_matches_the_crate_per_char_and_concatenated() {
    let mut all = String::new();
    let mut expected = String::new();
    for c in chars() {
        let s = c.to_string();
        let want = any_ascii::any_ascii(&s);
        assert_eq!(anyascii(&s), want, "U+{:04X}", u32::from(c));
        all.push(c);
        expected.push_str(&want);
    }
    assert_eq!(anyascii(&all), expected);
}

#[test]
fn only_ascii_input_is_borrowed() {
    for s in ["", "plain ascii\n\0\x7f"] {
        assert!(matches!(anyascii(s), Cow::Borrowed(x) if x == s));
        assert!(matches!(unidecode(s, Missing::Drop), Cow::Borrowed(x) if x == s));
    }
    let kept = "a\u{e000}b";
    // Equal to the input, but owned: Unidecode returns a new str here too.
    assert!(matches!(unidecode(kept, Missing::Keep), Cow::Owned(x) if x == kept));
    assert!(
        matches!(unidecode_wtf8(kept.as_bytes(), MissingWtf8::Keep), Cow::Owned(x) if x == kept.as_bytes())
    );
    assert!(matches!(anyascii("é"), Cow::Owned(x) if x == "e"));
}

#[test]
fn has_transliteration_cases() {
    for c in [
        'a',
        '\0',
        'é',
        '\u{301}',
        '\u{200d}',
        '\u{640}',
        '\u{627}',
        '\u{fe0f}',
        '\u{1f3fb}',
        '😀',
        '北',
    ] {
        assert!(has_transliteration(c), "U+{:04X}", u32::from(c));
    }
    for c in ['\u{e000}', '\u{378}', '\u{10ffff}', '\u{f0000}', '\u{ffff}'] {
        assert!(!has_transliteration(c), "U+{:04X}", u32::from(c));
    }
}

#[test]
fn missing_modes() {
    let s = "x\u{e000}é\u{301}\u{378}y";
    assert_eq!(unidecode(s, Missing::Drop), "xey");
    assert_eq!(unidecode(s, Missing::Replace("?")), "x?e?y");
    assert_eq!(unidecode(s, Missing::Replace("")), "xey");
    assert_eq!(unidecode(s, Missing::Keep), "x\u{e000}e\u{378}y");
    assert_eq!(
        first_missing(s),
        Some(MissingAt {
            index: 1,
            unit: Unit::Char('\u{e000}')
        })
    );
    assert_eq!(first_missing("北京\u{378}").map(|m| m.index), Some(2));
    assert_eq!(first_missing("é\u{301}北 ok"), None);
}

/// `s` with lone surrogates spliced in at the given code point positions, as
/// Python's `"surrogatepass"` encodes them.
fn wtf8(parts: &[Result<&str, u16>]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in parts {
        match p {
            Ok(s) => out.extend_from_slice(s.as_bytes()),
            Err(u) => {
                let u = *u;
                out.extend_from_slice(&[
                    0xE0 | u8::try_from(u >> 12).unwrap(),
                    0x80 | u8::try_from((u >> 6) & 0x3F).unwrap(),
                    0x80 | u8::try_from(u & 0x3F).unwrap(),
                ]);
            }
        }
    }
    out
}

#[test]
fn wtf8_paths() {
    let b = wtf8(&[
        Ok("ab"),
        Err(0xD800),
        Ok("é"),
        Err(0xDFFF),
        Err(0xDC00),
        Ok("\u{e000}"),
    ]);
    assert_eq!(anyascii_wtf8(&b).as_ref(), b"abe");
    assert_eq!(unidecode_wtf8(&b, MissingWtf8::Drop).as_ref(), b"abe");
    assert_eq!(
        unidecode_wtf8(&b, MissingWtf8::Replace(b"<>")).as_ref(),
        b"ab<>e<><><>"
    );
    let kept = unidecode_wtf8(&b, MissingWtf8::Keep);
    let mut want = wtf8(&[Ok("ab"), Err(0xD800), Ok("e"), Err(0xDFFF), Err(0xDC00)]);
    want.extend_from_slice("\u{e000}".as_bytes());
    assert_eq!(kept.as_ref(), want.as_slice());
    assert_eq!(
        first_missing_wtf8(&b),
        Some(MissingAt {
            index: 2,
            unit: Unit::Surrogate(0xD800)
        })
    );
    assert_eq!(
        surrogates_wtf8(&b),
        vec![(2, 0xD800), (4, 0xDFFF), (5, 0xDC00)]
    );
    // Valid UTF-8 gives the same results as the str functions.
    for s in ["", "abc", "x\u{e000}é\u{301}\u{378}y", "北京😀"] {
        assert_eq!(anyascii_wtf8(s.as_bytes()).as_ref(), anyascii(s).as_bytes());
        assert_eq!(
            unidecode_wtf8(s.as_bytes(), MissingWtf8::Replace(b"?")).as_ref(),
            unidecode(s, Missing::Replace("?")).as_bytes()
        );
        assert_eq!(first_missing_wtf8(s.as_bytes()), first_missing(s));
        assert_eq!(surrogates_wtf8(s.as_bytes()), Vec::new());
    }
    // A replacement holding a surrogate is spliced in as bytes.
    let rep = wtf8(&[Err(0xD83D)]);
    assert_eq!(
        unidecode_wtf8("a\u{e000}".as_bytes(), MissingWtf8::Replace(&rep)).as_ref(),
        [b"a".as_slice(), &rep].concat().as_slice()
    );
}
