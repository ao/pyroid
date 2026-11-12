//! Transliteration: `anyascii`, `unidecode` in each `Missing` mode,
//! `first_missing`, `is_ascii`, `ascii_prefix_len` and the `*_wtf8` variants.
//!
//! Input: `[replacement length] [replacement bytes] text...`. The str
//! functions get the text through `from_utf8_lossy`; the `*_wtf8` functions
//! get the raw bytes (and the replacement's raw bytes when they are WTF-8).
//!
//! Checked against a per-character model (each code point maps to itself if
//! ASCII, to `anyascii_char` if it has a transliteration, else to what the
//! `Missing` mode says; a lone surrogate never has one):
//! - every function's output equals the model; `anyascii` and
//!   `Missing::Drop` output is ASCII, and so is `Missing::Keep` output
//!   exactly when `first_missing` finds nothing;
//! - `Cow::Borrowed` exactly when the input is ASCII (then it is the input);
//! - `unidecode(_, Drop)` is `anyascii`; without a missing character every
//!   mode gives the same result (Unidecode's strict mode then succeeds);
//! - `first_missing` is the first code point without a transliteration;
//! - the `*_wtf8` variants equal the str ones on UTF-8 input;
//!   `surrogates_wtf8` lists the lone surrogates;
//! - on bytes that are not WTF-8 at all: no panic, ASCII where promised.
//!
//! `PYROID_FUZZ_DUMP=<file>` also writes every input's `anyascii` output, for a
//! differential run of a corpus against the `anyascii` package (see `dump`).

#![no_main]

use std::borrow::Cow;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

use libfuzzer_sys::fuzz_target;
use pyroid_core::ftfy::validate_wtf8;
use pyroid_core::translit::{
    self, Missing, MissingAt, MissingWtf8, Unit, anyascii_char, has_transliteration,
};

/// Code points of valid WTF-8, with their bytes.
fn code_points(b: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let lead = b[i];
        let n = match lead {
            0..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            _ => 4,
        };
        let mut cp = u32::from(lead) & (0x7F >> (n - 1 + usize::from(n > 1)));
        for &x in &b[i + 1..i + n] {
            cp = (cp << 6) | u32::from(x & 0x3F);
        }
        out.push((cp, &b[i..i + n]));
        i += n;
    }
    out
}

#[derive(Clone, Copy)]
enum Mode<'r> {
    Drop,
    Replace(&'r [u8]),
    Keep,
}

/// The per-character model over valid WTF-8.
fn model(b: &[u8], mode: Mode<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    for (cp, raw) in code_points(b) {
        match char::from_u32(cp) {
            Some(c) if c.is_ascii() => out.push(raw[0]),
            Some(c) if has_transliteration(c) => {
                out.extend_from_slice(anyascii_char(c).as_bytes());
            }
            _ => match mode {
                Mode::Drop => {}
                Mode::Replace(r) => out.extend_from_slice(r),
                Mode::Keep => out.extend_from_slice(raw),
            },
        }
    }
    out
}

fn model_first_missing(b: &[u8]) -> Option<MissingAt> {
    code_points(b)
        .into_iter()
        .enumerate()
        .find_map(|(index, (cp, _))| match char::from_u32(cp) {
            Some(c) if has_transliteration(c) => None,
            Some(c) => Some(MissingAt {
                index,
                unit: Unit::Char(c),
            }),
            None => Some(MissingAt {
                index,
                unit: Unit::Surrogate(u16::try_from(cp).expect("surrogate")),
            }),
        })
}

fn check_borrow<T: ?Sized + ToOwned + AsRef<[u8]>>(what: &str, input: &T, out: &Cow<'_, T>) {
    let ascii = input.as_ref().is_ascii();
    match out {
        Cow::Borrowed(b) => {
            assert!(ascii, "{what}: Borrowed for non-ASCII input");
            assert!(
                std::ptr::eq(b.as_ref().as_ptr(), input.as_ref().as_ptr())
                    && b.as_ref().len() == input.as_ref().len(),
                "{what}: Borrowed is not the input"
            );
        }
        Cow::Owned(_) => assert!(!ascii, "{what}: Owned for ASCII input"),
    }
}

fn check_str(s: &str, rep: &str) {
    let b = s.as_bytes();
    assert_eq!(translit::is_ascii(s), s.is_ascii());
    let a = translit::anyascii(s);
    check_borrow("anyascii", s, &a);
    assert!(a.is_ascii(), "anyascii output is not ASCII");
    assert_eq!(a.as_bytes(), model(b, Mode::Drop), "anyascii vs model");

    let drop = translit::unidecode(s, Missing::Drop);
    check_borrow("unidecode drop", s, &drop);
    assert_eq!(drop, a, "unidecode(Drop) vs anyascii");
    let keep = translit::unidecode(s, Missing::Keep);
    check_borrow("unidecode keep", s, &keep);
    assert_eq!(
        keep.as_bytes(),
        model(b, Mode::Keep),
        "unidecode(Keep) vs model"
    );
    let replace = translit::unidecode(s, Missing::Replace(rep));
    check_borrow("unidecode replace", s, &replace);
    assert_eq!(
        replace.as_bytes(),
        model(b, Mode::Replace(rep.as_bytes())),
        "unidecode(Replace) vs model"
    );

    let first = translit::first_missing(s);
    assert_eq!(first, model_first_missing(b), "first_missing vs model");
    assert_eq!(
        keep.is_ascii(),
        first.is_none(),
        "Keep output ASCII iff nothing missing"
    );
    if first.is_none() {
        assert_eq!(keep, a, "no missing: Keep vs Drop");
        assert_eq!(replace, a, "no missing: Replace vs Drop");
    }
}

fn check_wtf8(b: &[u8], rep: &[u8], str_rep: Option<&str>) {
    assert_eq!(
        translit::ascii_prefix_len(b),
        b.iter().position(|&x| x >= 0x80).unwrap_or(b.len()),
        "ascii_prefix_len"
    );
    let a = translit::anyascii_wtf8(b);
    check_borrow("anyascii_wtf8", b, &a);
    assert!(a.is_ascii(), "anyascii_wtf8 output is not ASCII");
    let drop = translit::unidecode_wtf8(b, MissingWtf8::Drop);
    assert!(drop.is_ascii(), "unidecode_wtf8(Drop) output is not ASCII");
    let keep = translit::unidecode_wtf8(b, MissingWtf8::Keep);
    let replace = translit::unidecode_wtf8(b, MissingWtf8::Replace(rep));
    let first = translit::first_missing_wtf8(b);
    let surrogates = translit::surrogates_wtf8(b);
    if validate_wtf8(b).is_err() {
        // Not something the bindings ever pass; only "no panic" is promised.
        return;
    }
    check_borrow("unidecode_wtf8 drop", b, &drop);
    check_borrow("unidecode_wtf8 keep", b, &keep);
    check_borrow("unidecode_wtf8 replace", b, &replace);
    assert_eq!(&a[..], model(b, Mode::Drop), "anyascii_wtf8 vs model");
    assert_eq!(drop, a, "unidecode_wtf8(Drop) vs anyascii_wtf8");
    assert_eq!(
        &keep[..],
        model(b, Mode::Keep),
        "unidecode_wtf8(Keep) vs model"
    );
    assert_eq!(
        &replace[..],
        model(b, Mode::Replace(rep)),
        "unidecode_wtf8(Replace) vs model"
    );
    assert_eq!(first, model_first_missing(b), "first_missing_wtf8 vs model");
    let model_surrogates: Vec<(usize, u16)> = code_points(b)
        .into_iter()
        .enumerate()
        .filter_map(|(i, (cp, _))| {
            (0xD800..0xE000)
                .contains(&cp)
                .then(|| (i, u16::try_from(cp).expect("surrogate")))
        })
        .collect();
    assert_eq!(surrogates, model_surrogates, "surrogates_wtf8 vs model");

    if let Ok(s) = std::str::from_utf8(b) {
        assert!(surrogates.is_empty());
        assert_eq!(
            &a[..],
            translit::anyascii(s).as_bytes(),
            "anyascii: wtf8 vs str"
        );
        assert_eq!(
            first,
            translit::first_missing(s),
            "first_missing: wtf8 vs str"
        );
        assert_eq!(
            &keep[..],
            translit::unidecode(s, Missing::Keep).as_bytes(),
            "unidecode Keep: wtf8 vs str"
        );
        if let Some(r) = str_rep {
            assert_eq!(
                &replace[..],
                translit::unidecode(s, Missing::Replace(r)).as_bytes(),
                "unidecode Replace: wtf8 vs str"
            );
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&n, rest)) = data.split_first() else {
        return;
    };
    let n = usize::from(n % 8).min(rest.len());
    let (rep, body) = rest.split_at(n);
    let str_rep = String::from_utf8_lossy(rep);
    let text = String::from_utf8_lossy(body);
    check_str(&text, &str_rep);
    // The bindings pass a WTF-8 replacement only when it is one.
    let (wrep, wrep_str): (&[u8], Option<&str>) = if validate_wtf8(rep).is_ok() {
        (rep, std::str::from_utf8(rep).ok())
    } else {
        (str_rep.as_bytes(), Some(&str_rep))
    };
    check_wtf8(body, wrep, wrep_str);
    if !matches!(text, Cow::Borrowed(_)) {
        check_wtf8(text.as_bytes(), wrep, wrep_str);
    }
    dump(text.as_bytes());
    if validate_wtf8(body).is_ok() && std::str::from_utf8(body).is_err() {
        dump(body);
    }
});

/// `PYROID_FUZZ_DUMP=<file>`: every input's `anyascii_wtf8`, one JSON object
/// per line, for a differential run of a corpus against the `anyascii` PyPI
/// package (`cargo fuzz run translit <corpus> -- -runs=0`).
fn dump(b: &[u8]) {
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let Some(f) = FILE
        .get_or_init(|| {
            let path = std::env::var_os("PYROID_FUZZ_DUMP")?;
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("open PYROID_FUZZ_DUMP");
            Some(Mutex::new(f))
        })
        .as_ref()
    else {
        return;
    };
    let hex = |b: &[u8]| -> String { b.iter().map(|x| format!("{x:02x}")).collect() };
    let line = format!(
        "{{\"in\":\"{}\",\"anyascii\":\"{}\"}}\n",
        hex(b),
        hex(&translit::anyascii_wtf8(b))
    );
    let mut f = f.lock().expect("dump lock");
    f.write_all(line.as_bytes()).expect("write dump");
}
