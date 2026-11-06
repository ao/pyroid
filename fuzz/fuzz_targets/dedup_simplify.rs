//! dedup: `simplify_text` / `simplify_text_into`, then the MinHash signature
//! of the result with a small config.
//!
//! Input: `[norm bits] [version] [n_grams] [buckets] [hashes per bucket]
//! [seed: 4 bytes LE] [hash bits] text...`. Zero bits keep datatrove's
//! `TextNormConfig` defaults; the MinHash sizes are small (1..=8 n-grams,
//! 1..=4 buckets of 1..=4 hashes) but every seed and hash function is used.
//!
//! Checked:
//! - no panic; `simplify_text_into` (into a dirty buffer) equals
//!   `simplify_text`, which is `Borrowed` exactly when unchanged and is
//!   stripped; with `norm_whitespace` and without diacritics removal the
//!   only whitespace left is single spaces;
//! - idempotence only where datatrove's `simplify_text` has it: not with
//!   `norm_numbers` (`"1.2.3"` -> `"0.0"` -> `"0"`), not with
//!   `norm_unicode_diacritics` (removing a mark can join digits or spaces),
//!   and not with `lowercase` plus weekday/month names (`"WEEKDAY"` is
//!   lowercased by the second pass);
//! - `MinHasher::signature` is deterministic, has `num_buckets *
//!   hashes_per_bucket` values, exists iff the simplified text has at least
//!   `n_grams` spaCy words, equals the tokens/shingles entry points, and
//!   equals a naive `min((s * a + b) mod 2**64 mod (2**61 - 1)) & mask`.
//!
//! `PYROID_FUZZ_DUMP=<file>` also writes every input's results, for a
//! differential run of a corpus against datatrove (see `dump`).

#![no_main]

use std::borrow::Cow;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

use libfuzzer_sys::fuzz_target;
use pyroid_core::dedup::{
    HashConfig, HashFn, MERSENNE_PRIME, MinHasher, MinhashConfig, Precision, TextNormConfig,
    WordTokenizer as _, simplify::simplify_text_into, simplify_text,
};
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::text::is_py_whitespace;

const VERSIONS: [UnicodeVersion; 5] = [
    UnicodeVersion::V17_0,
    UnicodeVersion::V16_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V15_0,
    UnicodeVersion::V14_0,
];
const HEADER: usize = 10;

fn norm_config(bits: u8) -> TextNormConfig {
    let flip = |i: u8| bits >> i & 1 != 0;
    let d = TextNormConfig::default();
    TextNormConfig {
        lowercase: d.lowercase ^ flip(0),
        norm_whitespace: d.norm_whitespace ^ flip(1),
        remove_punctuation: d.remove_punctuation ^ flip(2),
        norm_unicode_diacritics: d.norm_unicode_diacritics ^ flip(3),
        norm_numbers: d.norm_numbers ^ flip(4),
        norm_weekdays: d.norm_weekdays ^ flip(5),
        norm_monthnames: d.norm_monthnames ^ flip(6),
    }
}

fn check_simplify(text: &str, cfg: &TextNormConfig, v: UnicodeVersion) -> String {
    let s = simplify_text(text, cfg, v);
    match &s {
        Cow::Borrowed(b) => assert!(
            b.as_ptr() == text.as_ptr() && b.len() == text.len(),
            "Borrowed is not the input"
        ),
        Cow::Owned(o) => assert!(o != text, "Owned result equals the input"),
    }
    let mut into = String::from("dirty buffer \u{e9}");
    simplify_text_into(text, cfg, v, &mut into);
    assert_eq!(into, s, "simplify_text_into vs simplify_text");
    assert_eq!(
        s.trim_matches(is_py_whitespace),
        &*s,
        "result is not stripped"
    );
    if cfg.norm_whitespace && !cfg.norm_unicode_diacritics {
        assert!(
            !s.contains("  ") && s.chars().all(|c| c == ' ' || !is_py_whitespace(c)),
            "whitespace not normalized: {s:?}"
        );
    }
    let idempotent = !cfg.norm_numbers
        && !cfg.norm_unicode_diacritics
        && !(cfg.lowercase && (cfg.norm_weekdays || cfg.norm_monthnames));
    if idempotent {
        assert_eq!(
            simplify_text(&s, cfg, v),
            s,
            "simplify_text is not idempotent"
        );
    }
    s.into_owned()
}

fn naive_signature(h: &MinHasher, shingles: &[u64]) -> Vec<u64> {
    let (a, b) = h.parameters();
    let mask = match h.config().hash_config.precision {
        Precision::P32 => u64::from(u32::MAX),
        Precision::P64 => u64::MAX,
    };
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            shingles
                .iter()
                .map(|&s| {
                    let x = u128::from(s.wrapping_mul(a).wrapping_add(b));
                    u64::try_from(x % u128::from(MERSENNE_PRIME)).expect("< p") & mask
                })
                .min()
                .expect("shingles")
        })
        .collect()
}

fn minhash_config(header: &[u8]) -> MinhashConfig {
    let n_grams = 1 + usize::from(header[2] % 8);
    let num_buckets = 1 + usize::from(header[3] % 4);
    let hashes_per_bucket = 1 + usize::from(header[4] % 4);
    let seed = u32::from_le_bytes([header[5], header[6], header[7], header[8]]);
    let hash_config = HashConfig {
        precision: if header[9] & 1 != 0 {
            Precision::P32
        } else {
            Precision::P64
        },
        hash_fc: if header[9] & 2 != 0 {
            HashFn::Sha1
        } else {
            HashFn::Xxhash
        },
    };
    MinhashConfig {
        n_grams,
        num_buckets,
        hashes_per_bucket,
        seed,
        norm_config: norm_config(header[0]),
        hash_config,
    }
}

fn check_minhash(header: &[u8], text: &str, simplified: &str, v: UnicodeVersion) {
    let config = minhash_config(header);
    let (n_grams, num_buckets, hashes_per_bucket) =
        (config.n_grams, config.num_buckets, config.hashes_per_bucket);
    let h = MinHasher::new(config, v).expect("valid config");
    let n = num_buckets * hashes_per_bucket;
    assert_eq!(h.num_hashes(), n, "num_hashes");
    assert_eq!(config.num_hashes(), n, "config num_hashes");
    let (a, b) = h.parameters();
    assert_eq!((a.len(), b.len()), (n, n), "parameter count");

    let mut sig = vec![0u64; n];
    let has = h.signature(text, &mut sig).expect("short text");
    let mut sig2 = vec![1u64; n];
    let has2 = h.signature(text, &mut sig2).expect("short text");
    assert_eq!(has, has2, "signature presence is not deterministic");

    let words = h.spacy().words(simplified);
    let mut dwords = Vec::new();
    h.spacy()
        .word_tokenize(simplified, &mut dwords)
        .expect("short text");
    assert_eq!(dwords, words, "word_tokenize vs words");
    assert_eq!(
        has,
        words.len() >= n_grams,
        "signature iff >= n_grams words"
    );

    let mut shingles = Vec::new();
    h.shingles_from_tokens(&words, &mut shingles);
    assert_eq!(
        shingles.len(),
        (words.len() + 1).saturating_sub(n_grams),
        "shingle count"
    );
    let mut sig3 = vec![2u64; n];
    assert_eq!(h.signature_from_tokens(&words, &mut sig3), has);
    let mut sig4 = vec![3u64; n];
    assert_eq!(h.signature_from_shingles(&shingles, &mut sig4), has);
    if has {
        assert_eq!(sig, sig2, "signature is not deterministic");
        assert_eq!(sig, sig3, "signature vs signature_from_tokens");
        assert_eq!(sig, sig4, "signature vs signature_from_shingles");
        assert_eq!(sig, naive_signature(&h, &shingles), "signature vs naive");
        assert!(sig.iter().all(|&x| x < MERSENNE_PRIME), "value >= p");
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER {
        return;
    }
    let (header, body) = data.split_at(HEADER);
    let cfg = norm_config(header[0]);
    let v = VERSIONS[usize::from(header[1]) % VERSIONS.len()];
    let text = String::from_utf8_lossy(body);
    let simplified = check_simplify(&text, &cfg, v);
    check_minhash(header, &text, &simplified, v);
    dump(header, &text);
});

/// `PYROID_FUZZ_DUMP=<file>`: every input's Unicode 15.1 `simplify_text`,
/// words and signature, one JSON object per line, for a differential run of
/// a corpus against datatrove (`cargo fuzz run dedup_simplify <corpus> --
/// -runs=0`).
fn dump(header: &[u8], text: &str) {
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
    let config = minhash_config(header);
    let n = config.norm_config;
    let h = MinHasher::new(config, UnicodeVersion::V15_1).expect("valid config");
    let simplified = simplify_text(text, &n, UnicodeVersion::V15_1);
    let words: Vec<String> = h
        .spacy()
        .words(&simplified)
        .iter()
        .map(|w| format!("\"{}\"", hex(w.as_bytes())))
        .collect();
    let mut sig = vec![0u64; h.num_hashes()];
    let sig = if h.signature(text, &mut sig).expect("short text") {
        let v: Vec<String> = sig.iter().map(u64::to_string).collect();
        format!("[{}]", v.join(","))
    } else {
        "null".to_owned()
    };
    let line = format!(
        "{{\"norm\":{{\"lowercase\":{},\"norm_whitespace\":{},\"remove_punctuation\":{},\
         \"norm_unicode_diacritics\":{},\"norm_numbers\":{},\"norm_weekdays\":{},\
         \"norm_monthnames\":{}}},\"n_grams\":{},\"num_buckets\":{},\"hashes_per_bucket\":{},\
         \"seed\":{},\"precision\":{},\"hash_fc\":\"{}\",\"in\":\"{}\",\"simplified\":\"{}\",\
         \"words\":[{}],\"signature\":{sig}}}\n",
        n.lowercase,
        n.norm_whitespace,
        n.remove_punctuation,
        n.norm_unicode_diacritics,
        n.norm_numbers,
        n.norm_weekdays,
        n.norm_monthnames,
        config.n_grams,
        config.num_buckets,
        config.hashes_per_bucket,
        config.seed,
        config.hash_config.precision.bits(),
        config.hash_config.hash_fc.as_str(),
        hex(text.as_bytes()),
        hex(simplified.as_bytes()),
        words.join(","),
    );
    let mut f = f.lock().expect("dump lock");
    f.write_all(line.as_bytes()).expect("write dump");
}
