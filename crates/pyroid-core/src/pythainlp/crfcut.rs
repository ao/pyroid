// Derived from PyThaiNLP 5.4.0 (Apache-2.0), © 2016-2026 PyThaiNLP Project,
// and datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! `pythainlp.tokenize.crfcut`: `_extract_features` (window 2, n-grams up
//! to 3, `"xxpad"` padding, the ender and starter word sets), the CRF
//! tagger, and `segment`'s post-processing and joining.
//!
//! Each token gets 37 string attributes, in `_extract_features`' order:
//! `bias`, then for each n-gram size and offset `word_{n}_{a}_{b}=`,
//! `ender_{n}_{a}_{b}=` and `starter_{n}_{a}_{b}=` with the tokens (or their
//! classes) joined by `|`. Tokens are not escaped; CRFsuite reads an
//! attribute up to its first NUL.

use std::io::Write as _;
use std::sync::LazyLock;

use rustc_hash::FxHashSet;

use crate::crfsuite::{Model, Tagger};
use crate::text::is_py_whitespace;

/// `_ENDERS`.
const ENDERS: [&str; 47] = [
    "ครับ",
    "ค่ะ",
    "คะ",
    "นะคะ",
    "นะ",
    "จ้ะ",
    "จ้า",
    "จ๋า",
    "ฮะ",
    "ๆ",
    "ได้",
    "แล้ว",
    "ด้วย",
    "เลย",
    "มาก",
    "น้อย",
    "กัน",
    "เช่นกัน",
    "เท่านั้น",
    "อยู่",
    "ลง",
    "ขึ้น",
    "มา",
    "ไป",
    "ไว้",
    "เอง",
    "อีก",
    "ใหม่",
    "จริงๆ",
    "บ้าง",
    "หมด",
    "ทีเดียว",
    "เดียว",
    "นั้น",
    "นี้",
    "เหล่านี้",
    "เหล่านั้น",
    "อย่างไร",
    "ยังไง",
    "หรือไม่",
    "มั้ย",
    "ไหน",
    "ไหม",
    "อะไร",
    "ทำไม",
    "เมื่อไหร่",
    "เมื่อไร",
];

/// `_STARTERS`.
const STARTERS: [&str; 42] = [
    "ผม",
    "ฉัน",
    "ดิฉัน",
    "ชั้น",
    "คุณ",
    "มัน",
    "เขา",
    "เค้า",
    "เธอ",
    "เรา",
    "พวกเรา",
    "พวกเขา",
    "กู",
    "มึง",
    "แก",
    "ข้าพเจ้า",
    "และ",
    "หรือ",
    "แต่",
    "เมื่อ",
    "ถ้า",
    "ใน",
    "ด้วย",
    "เพราะ",
    "เนื่องจาก",
    "ซึ่ง",
    "ไม่",
    "ตอนนี้",
    "ทีนี้",
    "ดังนั้น",
    "เพราะฉะนั้น",
    "ฉะนั้น",
    "ตั้งแต่",
    "ในที่สุด",
    "ก็",
    "กับ",
    "แก่",
    "ต่อ",
    "นั้น",
    "นี้",
    "เหล่านี้",
    "เหล่านั้น",
];

static ENDER_SET: LazyLock<FxHashSet<&'static str>> =
    LazyLock::new(|| ENDERS.into_iter().collect());
static STARTER_SET: LazyLock<FxHashSet<&'static str>> =
    LazyLock::new(|| STARTERS.into_iter().collect());

/// The n-gram size, first offset and `feature_position` of each feature
/// group, in `_extract_features`' order.
const GROUPS: [(usize, isize, &str); 12] = [
    (1, -2, "1_-2_-1"),
    (1, -1, "1_-1_0"),
    (1, 0, "1_0_1"),
    (1, 1, "1_1_2"),
    (1, 2, "1_2_3"),
    (2, -2, "2_-2_0"),
    (2, -1, "2_-1_1"),
    (2, 0, "2_0_2"),
    (2, 1, "2_1_3"),
    (3, -2, "3_-2_1"),
    (3, -1, "3_-1_2"),
    (3, 0, "3_0_3"),
];

const PAD: &str = "xxpad";
/// `_extract_features`' `window`.
const WINDOW: usize = 2;

/// The attribute ids of a model that do not depend on the words: `bias`,
/// and every `ender_*` / `starter_*` value, by group and by the classes of
/// its n tokens (bit `k` set: token `k` is an ender / starter).
#[derive(Debug, Clone)]
pub(super) struct Attrs {
    bias: Option<u32>,
    ender: [[Option<u32>; 8]; 12],
    starter: [[Option<u32>; 8]; 12],
}

impl Attrs {
    pub(super) fn new(model: &Model) -> Self {
        let mut a = Attrs {
            bias: model.attr_id(b"bias"),
            ender: [[None; 8]; 12],
            starter: [[None; 8]; 12],
        };
        let mut buf = Vec::new();
        for (g, &(n, _, pos)) in GROUPS.iter().enumerate() {
            for mask in 0..1usize << n {
                for (kind, yes, ids) in [
                    ("ender", "ender", &mut a.ender),
                    ("starter", "starter", &mut a.starter),
                ] {
                    buf.clear();
                    write!(buf, "{kind}_{pos}=").expect("Vec write");
                    for k in 0..n {
                        if k > 0 {
                            buf.push(b'|');
                        }
                        let class = if mask >> k & 1 == 1 { yes } else { "normal" };
                        buf.extend_from_slice(class.as_bytes());
                    }
                    ids[g][mask] = model.attr_id(&buf);
                }
            }
        }
        a
    }
}

/// `_tagger.tag(_extract_features(toks))`: whether each token's label is
/// `"E"`, written to `out` (cleared first).
pub(super) fn labels(model: &Model, attrs: &Attrs, toks: &[&str], out: &mut Vec<bool>) {
    out.clear();
    if toks.is_empty() {
        return;
    }
    // The padded document, and each word's ender and starter bit.
    let doc: Vec<&str> = [PAD; WINDOW]
        .into_iter()
        .chain(toks.iter().copied())
        .chain([PAD; WINDOW])
        .collect();
    let classes: Vec<(usize, usize)> = doc
        .iter()
        .map(|w| {
            (
                usize::from(ENDER_SET.contains(w)),
                usize::from(STARTER_SET.contains(w)),
            )
        })
        .collect();
    let mut tagger = Tagger::new(model);
    tagger.clear();
    let mut buf: Vec<u8> = Vec::with_capacity(128);
    for at in WINDOW..doc.len() - WINDOW {
        tagger.push_item();
        if let Some(bias) = attrs.bias {
            tagger.add_one(bias);
        }
        for (group, &(size, off, pos)) in GROUPS.iter().enumerate() {
            let first = at.checked_add_signed(off).expect("inside the padding");
            buf.clear();
            buf.extend_from_slice(b"word_");
            buf.extend_from_slice(pos.as_bytes());
            buf.push(b'=');
            let (mut enders, mut starters) = (0usize, 0usize);
            for (bit, (word, &(ender, starter))) in doc[first..first + size]
                .iter()
                .zip(&classes[first..first + size])
                .enumerate()
            {
                if bit > 0 {
                    buf.push(b'|');
                }
                buf.extend_from_slice(word.as_bytes());
                enders |= ender << bit;
                starters |= starter << bit;
            }
            tagger.add_str(&buf);
            if let Some(id) = attrs.ender[group][enders] {
                tagger.add_one(id);
            }
            if let Some(id) = attrs.starter[group][starters] {
                tagger.add_one(id);
            }
        }
    }
    let mut ids = Vec::with_capacity(toks.len());
    tagger.viterbi(&mut ids);
    let end = model.label_id("E");
    out.extend(ids.iter().map(|&l| Some(l as usize) == end));
}

/// Python's `str.strip()`.
pub(super) fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_whitespace)
}

/// `segment`'s post-processing of the labels and its joining: `toks` are
/// consecutive byte ranges, `is_e` the tagger's labels (changed in place as
/// `segment` changes `labs`). Each sentence's byte range is appended to
/// `out`; tokens after the last `"E"` are dropped.
pub(super) fn join(
    text: &str,
    toks: &[(usize, usize)],
    is_e: &mut [bool],
    out: &mut Vec<(usize, usize)>,
) {
    let Some(last) = is_e.last_mut() else {
        return;
    };
    *last = true;
    for idx in 0..toks.len() {
        let t = py_strip(&text[toks[idx].0..toks[idx].1]);
        if t.ends_with(['!', '.', '?']) {
            is_e[idx] = true;
        } else if (idx == 0 || is_e[idx - 1]) && t.is_empty() {
            is_e[idx] = false;
        }
    }
    // Tokens are never empty, so a sentence is never empty at an "E".
    let mut start = None;
    for (&(s, e), &cut) in toks.iter().zip(is_e.iter()) {
        let from = *start.get_or_insert(s);
        if cut {
            out.push((from, e));
            start = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_sets() {
        assert_eq!(ENDER_SET.len(), 47);
        assert_eq!(STARTER_SET.len(), 42);
        assert!(ENDER_SET.contains("ด้วย") && STARTER_SET.contains("ด้วย"));
        assert!(!ENDER_SET.contains(PAD) && !STARTER_SET.contains(PAD));
    }

    fn joined(text: &str, toks: &[&str], labs: &str) -> Vec<String> {
        let mut ranges = Vec::new();
        let mut at = 0;
        for t in toks {
            ranges.push((at, at + t.len()));
            at += t.len();
        }
        assert_eq!(at, text.len());
        let mut is_e: Vec<bool> = labs.chars().map(|c| c == 'E').collect();
        let mut out = Vec::new();
        join(text, &ranges, &mut is_e, &mut out);
        out.into_iter()
            .map(|(s, e)| text[s..e].to_owned())
            .collect()
    }

    #[test]
    fn post_processing() {
        // Terminal punctuation cuts; the last token always cuts.
        assert_eq!(
            joined("a. b c", &["a.", " ", "b", " ", "c"], "IIIII"),
            ["a.", " b c"]
        );
        // Whitespace right after a cut is not a cut, and a trailing one is
        // dropped.
        assert_eq!(joined("ab  ", &["a", "b", "  "], "IEE"), ["ab"]);
        assert_eq!(joined("a b", &["a", " ", "b"], "EEI"), ["a", " b"]);
        assert_eq!(joined("   ", &["   "], "E"), [""; 0]);
    }
}
