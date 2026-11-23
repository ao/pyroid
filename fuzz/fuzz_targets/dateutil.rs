//! python-dateutil's parser (`pyroid_core::dateutil`): `parse` under options from the first two
//! input bytes, and `isoparse` in both modes, on the rest of the input as text.
//!
//! Bits of the first byte: 0 `dayfirst`, 1 `yearfirst` (each `None` when bit 2 is clear), 3
//! `fuzzy`, 4 `fuzzy_with_tokens`, 5-7 the Unicode tables (newest first). The second byte: 0-1
//! `sys.get_int_max_str_digits()` (0, 640, 4300, 100000), 2 a custom parserinfo (extra jump
//! words, a two-digit year base of 1950), 3 an isoparse separator of `T`. The text is the rest,
//! decoded lossily (the binding passes UTF-8).
//!
//! Checked, besides "no panic":
//! - deterministic: the same result twice;
//! - with `fuzzy_with_tokens`, the skipped tokens are non-empty and hold no more characters than
//!   the text;
//! - `ampm` is 0 or 1 (the tables' values);
//! - `isoparse` is deterministic in both modes and gives microseconds below 1,000,000. (Range
//!   checks of the fields are the caller's, `datetime`'s. The modes can disagree on inputs both
//!   accept: 2.9.0.post0's `int()` takes `"3\r"` as a day, so `"2002003\r20"` is a common date
//!   with month 0 there, and an ordinal date, January 3, in 4.0.0.)

#![no_main]

use libfuzzer_sys::fuzz_target;
use pyroid_core::dateutil::info::{ParserInfo, Tables};
use pyroid_core::dateutil::isoparser::{self, Mode};
use pyroid_core::dateutil::{self as du, ParseOptions, Props, Settings};
use pyroid_core::ftfy::UnicodeVersion;

const VERSIONS: [UnicodeVersion; 5] = [
    UnicodeVersion::V17_0,
    UnicodeVersion::V16_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V15_0,
    UnicodeVersion::V14_0,
];
const DIGITS: [usize; 4] = [0, 640, 4300, 100_000];

fn custom() -> ParserInfo {
    let mut t = Tables::default();
    for (i, w) in [" ", ".", ",", "-", "/", "at", "um", "uhr", "den"].iter().enumerate() {
        t.jump.insert((*w).into(), i64::try_from(i).unwrap());
    }
    for (i, m) in ["jan", "feb", "mär", "märz", "apr", "mai"].iter().enumerate() {
        t.months.insert((*m).into(), i64::try_from(i.min(4)).unwrap());
    }
    t.hms.insert("h".into(), 0);
    t.ampm.insert("am".into(), 0);
    t.ampm.insert("pm".into(), 1);
    t.utczone.insert("utc".into(), 0);
    ParserInfo::new(t, Default::default(), vec!["UTC".into()])
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let (a, b) = (data[0], data[1]);
    let text = String::from_utf8_lossy(&data[2..]);
    let p = Props::new(VERSIONS[usize::from(a >> 5) % VERSIONS.len()]);
    let tri = |bit: u8| (a & 4 != 0).then_some(a & bit != 0);
    let opts = ParseOptions {
        dayfirst: tri(1),
        yearfirst: tri(2),
        fuzzy: a & 8 != 0,
        fuzzy_with_tokens: a & 16 != 0,
        max_str_digits: DIGITS[usize::from(b & 3)],
    };
    let (info, settings) = if b & 4 != 0 {
        (custom(), Settings::default_for(1950))
    } else {
        (ParserInfo::default_tables(), Settings::default_for(2026))
    };
    let r1 = du::parse(&text, &info, settings, p, opts);
    let r2 = du::parse(&text, &info, settings, p, opts);
    assert_eq!(r1, r2, "deterministic");
    if let Ok(Some(parsed)) = &r1 {
        let res = &parsed.res;
        if let Some(ap) = res.ampm {
            assert!(ap == 0 || ap == 1, "ampm {ap}");
        }
        if let Some(tokens) = &parsed.skipped {
            assert!(opts.fuzzy_with_tokens);
            // Pieces of the text: non-empty, and no more characters than it has (a number's
            // "," can become "." and a sign after a zone name is flipped, so they are not
            // always verbatim; the parity suite compares them with dateutil's).
            assert!(tokens.iter().all(|t| !t.is_empty()));
            let total: usize = tokens.iter().map(|t| t.chars().count()).sum();
            assert!(total <= text.chars().count());
        } else {
            assert!(!opts.fuzzy_with_tokens);
        }
    }
    let sep = (b & 8 != 0).then_some(b'T');
    let strict = isoparser::isoparse(text.as_bytes(), sep, Mode::Strict);
    let lenient = isoparser::isoparse(text.as_bytes(), sep, Mode::Lenient);
    assert_eq!(strict, isoparser::isoparse(text.as_bytes(), sep, Mode::Strict));
    assert_eq!(lenient, isoparser::isoparse(text.as_bytes(), sep, Mode::Lenient));
    for r in [&strict, &lenient] {
        if let Ok(iso) = r {
            assert!((0..1_000_000).contains(&iso.fields[6]));
        }
    }
});
