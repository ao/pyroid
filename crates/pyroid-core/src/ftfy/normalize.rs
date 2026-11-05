//! `unicodedata.normalize`, as a given CPython version computes it.
//!
//! The `unicode-normalization` crate implements a newer Unicode version than
//! any supported CPython. Normalization is stable for assigned characters, so
//! the only difference between versions is that characters *unassigned* in
//! the interpreter's version are inert there: they have no decomposition,
//! combining class 0 and never compose. Lone surrogates are inert too. We
//! reproduce that by normalizing the runs between inert characters and copying
//! the inert characters through unchanged.

use unicode_normalization::{
    IsNormalized, UnicodeNormalization, is_nfc_quick, is_nfd_quick, is_nfkc_quick, is_nfkd_quick,
};

use super::chardata::is_assigned;
use super::text::{decode_at, has_surrogates};
use super::{Normalization, UnicodeVersion};

fn quick(s: &str, form: Normalization) -> IsNormalized {
    match form {
        Normalization::Nfc => is_nfc_quick(s.chars()),
        Normalization::Nfd => is_nfd_quick(s.chars()),
        Normalization::Nfkc => is_nfkc_quick(s.chars()),
        Normalization::Nfkd => is_nfkd_quick(s.chars()),
    }
}

/// Normalize a run of text that contains no inert characters.
fn normalize_run(s: &str, form: Normalization, out: &mut Vec<u8>) {
    if quick(s, form) == IsNormalized::Yes {
        out.extend_from_slice(s.as_bytes());
        return;
    }
    let mut tmp = String::with_capacity(s.len() + 8);
    match form {
        Normalization::Nfc => tmp.extend(s.chars().nfc()),
        Normalization::Nfd => tmp.extend(s.chars().nfd()),
        Normalization::Nfkc => tmp.extend(s.chars().nfkc()),
        Normalization::Nfkd => tmp.extend(s.chars().nfkd()),
    }
    out.extend_from_slice(tmp.as_bytes());
}

/// `unicodedata.normalize(form, text)` for Unicode version `v`; `None` if unchanged.
pub(crate) fn normalize(text: &[u8], form: Normalization, v: UnicodeVersion) -> Option<Vec<u8>> {
    if text.is_ascii() {
        return None;
    }
    // Every code point below U+0300 is NFC-stable with combining class 0.
    if form == Normalization::Nfc && text.iter().all(|&b| b < 0xCC) {
        return None;
    }
    if !has_surrogates(text) {
        let s = std::str::from_utf8(text).expect("text without surrogates is UTF-8");
        // A string that is normalized in the crate's (newer) version is also
        // normalized in `v`: unassigned characters can only block changes.
        if quick(s, form) == IsNormalized::Yes {
            return None;
        }
    }
    let mut out = Vec::with_capacity(text.len() + 8);
    let mut run_start = 0;
    let mut i = 0;
    while i < text.len() {
        let (cp, n) = decode_at(text, i);
        let inert = (0xD800..=0xDFFF).contains(&cp) || (cp >= 0x0300 && !is_assigned(cp, v));
        if inert {
            let run = std::str::from_utf8(&text[run_start..i]).expect("runs are UTF-8");
            normalize_run(run, form, &mut out);
            out.extend_from_slice(&text[i..i + n]);
            run_start = i + n;
        }
        i += n;
    }
    let run = std::str::from_utf8(&text[run_start..]).expect("runs are UTF-8");
    normalize_run(run, form, &mut out);
    (out != text).then_some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ftfy::text::push_cp;

    const V: UnicodeVersion = UnicodeVersion::V15_0;

    fn nfc(s: &str) -> String {
        String::from_utf8(
            normalize(s.as_bytes(), Normalization::Nfc, V).unwrap_or(s.as_bytes().to_vec()),
        )
        .unwrap()
    }

    #[test]
    fn composes() {
        assert_eq!(nfc("e\u{301}"), "é");
        assert_eq!(nfc("café"), "café");
        assert!(normalize(b"plain", Normalization::Nfkd, V).is_none());
        let nfkc = normalize("ﬁ".as_bytes(), Normalization::Nfkc, V).unwrap();
        assert_eq!(nfkc, b"fi");
    }

    #[test]
    fn surrogates_are_inert_starters() {
        let mut t = Vec::new();
        for cp in [0x65, 0xD800, 0x301, 0x65, 0x301] {
            push_cp(&mut t, cp);
        }
        let mut want = Vec::new();
        for cp in [0x65, 0xD800, 0x301, 0xE9] {
            push_cp(&mut want, cp);
        }
        assert_eq!(normalize(&t, Normalization::Nfc, V), Some(want));
    }

    #[test]
    fn unassigned_chars_are_inert() {
        // U+1E5D0..U+1E5FF (Ol Onal, Unicode 16.0) includes combining marks.
        let s = "a\u{1e5ee}\u{301}";
        let new = normalize(s.as_bytes(), Normalization::Nfc, UnicodeVersion::V17_0);
        let old = normalize(s.as_bytes(), Normalization::Nfc, UnicodeVersion::V15_0);
        assert!(old.is_none() || old != new);
    }
}
