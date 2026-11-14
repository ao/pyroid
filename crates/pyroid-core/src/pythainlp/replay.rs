//! Replays of the PyThaiNLP 5.4.0 and datatrove 0.10.1 recordings in
//! `testdata/` (written by `testdata/oracle.py`).
//!
//! The replays need the corpus the recordings were made with:
//! `PYROID_PYTHAINLP_CORPUS` is PyThaiNLP's `corpus` directory (the
//! header's sha256 of `words_th.txt` and `sentenceseg_crfcut.model` are
//! checked). Without it the tests print why and pass.
//! `PYROID_PYTHAINLP_GOLDEN_DIR` replays the files of another directory
//! instead of `testdata/`, and `PYROID_PYTHAINLP_REPLAY` adds files
//! (comma-separated paths).
//!
//! Each row has the text `"t"` and any of: `"tcc"` (cluster lengths), `"nm"`
//! and `"nms"` (newmm segment lengths, without and with safe mode),
//! `"toks"` (crfcut's token lengths), `"lab"` (the tagger's raw labels),
//! `"w"`, `"s"` and `"sp"` (datatrove's words, sentences and spans).
//! `"x_<key>"` records that Python raised for that key. A header with
//! `"words"` (a list) was recorded with `newmm.segment(t,
//! custom_dict=Trie(words))`: its rows have only `"tcc"`, `"nm"` and `"nms"`
//! and need no corpus.
#![allow(clippy::print_stderr)] // reports what was replayed

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::Value;

use super::th::ThTokenizer;
use super::trie::Trie;
use super::{MODEL_FILE, Thai, WORDS_FILE};
use crate::filters::WordTokenizer;
use crate::ftfy::UnicodeVersion;

fn corpus_dir() -> Option<PathBuf> {
    std::env::var_os("PYROID_PYTHAINLP_CORPUS").map(PathBuf::from)
}

fn thai() -> Option<&'static Thai> {
    static T: OnceLock<Option<&'static Thai>> = OnceLock::new();
    *T.get_or_init(|| {
        let dir = corpus_dir()?;
        Some(super::install(dir).expect("PYROID_PYTHAINLP_CORPUS is PyThaiNLP's corpus directory"))
    })
}

/// The sha256 of the installed corpus files, computed once.
fn corpus_hashes() -> &'static (String, String) {
    static H: OnceLock<(String, String)> = OnceLock::new();
    H.get_or_init(|| {
        let dir = corpus_dir().expect("a corpus directory");
        let h = |name: &str| sha256_hex(&std::fs::read(dir.join(name)).expect("readable"));
        (h(WORDS_FILE), h(MODEL_FILE))
    })
}

/// SHA-256 (FIPS 180-4), for the header check only.
#[allow(clippy::many_single_char_names, clippy::too_many_lines)] // the standard's names
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    let mut msg = data.to_vec();
    let bits = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bits.to_be_bytes());
    for block in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    h.iter().fold(String::with_capacity(64), |mut s, x| {
        use std::fmt::Write as _;
        write!(s, "{x:08x}").expect("String write");
        s
    })
}

/// A recorded text: a string, or a recipe (`[item, ...]`, each a string or
/// `[string, count]`).
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let mut out = String::new();
            for item in items {
                match item {
                    Value::String(s) => out.push_str(s),
                    Value::Array(pair) => {
                        let s = pair[0].as_str().expect("recipe string");
                        let n = usize::try_from(pair[1].as_u64().expect("recipe count")).unwrap();
                        out.push_str(&s.repeat(n));
                    }
                    _ => panic!("bad recipe item {item}"),
                }
            }
            out
        }
        _ => panic!("bad text {v}"),
    }
}

/// A recording's rows, read one line at a time.
struct Rows {
    lines: std::io::Lines<std::io::BufReader<std::fs::File>>,
    n: usize,
}

impl Iterator for Rows {
    type Item = Value;

    fn next(&mut self) -> Option<Value> {
        let line = self.lines.next()?.expect("readable");
        self.n += 1;
        Some(serde_json::from_str(&line).expect("JSON"))
    }
}

/// The header and rows of a JSONL recording.
fn rows(path: &Path) -> (Value, Rows) {
    use std::io::BufRead;
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut rows = Rows {
        lines: std::io::BufReader::new(file).lines(),
        n: 0,
    };
    let header = rows.next().expect("a header")["header"].clone();
    rows.n = 0;
    (header, rows)
}

fn short(t: &str) -> String {
    let s: String = t.chars().take(40).collect();
    if s.len() < t.len() {
        format!("{s:?}...")
    } else {
        format!("{s:?}")
    }
}

fn numbers(v: &Value) -> Vec<u64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|v| v.as_u64().expect("a number"))
        .collect()
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().expect("an array").iter().map(text).collect()
}

fn lengths(xs: &[&str]) -> Vec<u64> {
    xs.iter().map(|x| x.chars().count() as u64).collect()
}

/// Where two sequences first differ.
fn first_diff<A: PartialEq<B>, B>(a: &[A], b: &[B]) -> usize {
    a.iter()
        .zip(b)
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()))
}

fn diff<T: std::fmt::Debug + PartialEq>(have: &[T], want: &[T]) -> Option<String> {
    if have == want {
        return None;
    }
    let i = first_diff(have, want);
    Some(format!(
        "item {i}: have {:?}, want {:?} ({} vs {} items)",
        have.get(i),
        want.get(i),
        have.len(),
        want.len()
    ))
}

/// What a recording is replayed against.
struct Ctx<'a> {
    /// The installed corpus (`None` for a recording with its own `words`).
    thai: Option<(&'static Thai, &'static ThTokenizer)>,
    /// The trie newmm uses.
    trie: &'a Trie,
    v: UnicodeVersion,
}

/// One row's mismatches, as `"key: what"`, appended to `fails`; returns the
/// number of results checked.
fn check_row(cx: &Ctx<'_>, row: &Value, fails: &mut Vec<String>) -> usize {
    let t = text(&row["t"]);
    let obj = row.as_object().expect("a row object");
    let v = cx.v;
    let corpus = || {
        cx.thai
            .expect("this key needs the installed corpus, not custom words")
    };
    let mut items = 0;
    for (key, want) in obj {
        let mismatch = match key.as_str() {
            "t" => continue,
            "tcc" => {
                let mut out = Vec::new();
                super::tcc(&t, &mut out);
                diff(&lengths(&out), &numbers(want))
            }
            "nm" | "nms" => {
                let mut out = Vec::new();
                super::newmm_with(cx.trie, &t, key == "nms", v, &mut out);
                diff(&lengths(&out), &numbers(want))
            }
            "toks" => {
                let mut out = Vec::new();
                corpus().0.word_tokenize(&t, v, &mut out);
                diff(&lengths(&out), &numbers(want))
            }
            "lab" => {
                let thai = corpus().0;
                let mut toks = Vec::new();
                thai.word_tokenize(&t, v, &mut toks);
                let mut labs = Vec::new();
                thai.crfcut_labels(&toks, &mut labs);
                let have: String = labs.iter().map(|&e| if e { 'E' } else { 'I' }).collect();
                let want = want.as_str().expect("a label string");
                (have != want).then(|| format!("have {have}, want {want}"))
            }
            "w" => {
                let want = strings(want);
                let mut out = Vec::new();
                corpus().1.words(&t, &mut out).expect("no error");
                diff(&out, &want.iter().map(String::as_str).collect::<Vec<_>>())
            }
            "s" => {
                let want = strings(want);
                let mut out = Vec::new();
                corpus().1.sentences(&t, &mut out).expect("no error");
                let have: Vec<&str> = out.iter().map(|s| &**s).collect();
                diff(&have, &want.iter().map(String::as_str).collect::<Vec<_>>())
            }
            "sp" => {
                let tok = corpus().1;
                let want: Vec<(u64, u64)> = want
                    .as_array()
                    .expect("an array")
                    .iter()
                    .map(|p| (p[0].as_u64().unwrap(), p[1].as_u64().unwrap()))
                    .collect();
                let mut spans = Vec::new();
                tok.spans(&t, &mut spans).expect("a short text");
                let have: Vec<(u64, u64)> =
                    spans.iter().map(|&(s, e)| (s as u64, e as u64)).collect();
                // `split_into_sentences` uses the span ends: here as bytes.
                let want_ends: Vec<usize> = want
                    .iter()
                    .map(|&(_, e)| {
                        let e = usize::try_from(e).unwrap();
                        t.char_indices().nth(e).map_or(t.len(), |(b, _)| b)
                    })
                    .collect();
                let mut ends = Vec::new();
                tok.sentence_ends(&t, &mut ends).expect("no error");
                diff(&have, &want)
                    .or_else(|| diff(&ends, &want_ends).map(|d| format!("sentence ends: {d}")))
            }
            k if k.starts_with("x_") => Some(format!("Python raised {want}")),
            _ => continue,
        };
        items += 1;
        if let Some(m) = mismatch {
            fails.push(format!("{} {key}: {m}", short(&t)));
        }
    }
    items
}

/// Replay one recording. A header with `"words"` was recorded with
/// `newmm.segment(t, custom_dict=Trie(words))` and needs no corpus.
fn replay(path: &Path) {
    let name = path.display();
    let (header, mut rows) = rows(path);
    let unicode = header["unicode"]
        .as_str()
        .expect("the header's Unicode version");
    let v = UnicodeVersion::from_unidata_version(unicode).expect("a supported Unicode version");
    let custom = header.get("words").map(|w| {
        let words = strings(w);
        Trie::new(
            words
                .iter()
                .map(|w| super::crfcut::py_strip(w))
                .collect::<Vec<_>>(),
        )
    });
    let thai = if custom.is_some() {
        None
    } else {
        let Some(thai) = thai() else {
            eprintln!("{name}: skipped, set PYROID_PYTHAINLP_CORPUS");
            return;
        };
        // `data_sha256: {"corpus/<file>": hash}` (oracle.py), or
        // `words_sha256` / `model_sha256`.
        let (words, model) = corpus_hashes();
        let mut checked = 0;
        for (file, key, have) in [
            (WORDS_FILE, "words_sha256", words),
            (MODEL_FILE, "model_sha256", model),
        ] {
            let want = header["data_sha256"]
                .get(format!("corpus/{file}"))
                .or_else(|| header.get(key))
                .and_then(Value::as_str);
            if let Some(want) = want {
                assert_eq!(have, want, "{name} was recorded with another {file}");
                checked += 1;
            }
        }
        assert_eq!(
            checked, 2,
            "{name}: the header has no sha256 of the corpus files"
        );
        Some((thai, ThTokenizer::get(v).expect("the corpus is installed")))
    };
    let cx = Ctx {
        thai,
        trie: custom
            .as_ref()
            .unwrap_or_else(|| &thai.expect("the corpus").0.trie),
        v,
    };
    let mut fails = Vec::new();
    let mut items = 0;
    for (i, row) in rows.by_ref().enumerate() {
        let before = fails.len();
        items += check_row(&cx, &row, &mut fails);
        for f in &mut fails[before..] {
            *f = format!("row {i}: {f}");
        }
    }
    for f in fails.iter().take(25) {
        eprintln!("{name}: {f}");
    }
    assert!(
        fails.is_empty(),
        "{name}: {} mismatches in {} rows",
        fails.len(),
        rows.n
    );
    eprintln!("{name}: {} rows ({items} results) match", rows.n);
}

fn golden_dir() -> PathBuf {
    std::env::var_os("PYROID_PYTHAINLP_GOLDEN_DIR").map_or_else(
        || {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/src/pythainlp/testdata"
            ))
        },
        PathBuf::from,
    )
}

#[test]
fn golden_files() {
    let dir = golden_dir();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    if files.is_empty() {
        eprintln!("{}: no recordings; skipped", dir.display());
        return;
    }
    for f in files {
        replay(&f);
    }
}

/// `PYROID_PYTHAINLP_REPLAY`: comma-separated paths of further recordings.
#[test]
fn extra_recordings() {
    let Ok(paths) = std::env::var("PYROID_PYTHAINLP_REPLAY") else {
        return;
    };
    for path in paths.split(',').filter(|p| !p.is_empty()) {
        replay(Path::new(path));
    }
}

#[test]
fn sha256_known_values() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(&b"a".repeat(1000)),
        "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3"
    );
}
