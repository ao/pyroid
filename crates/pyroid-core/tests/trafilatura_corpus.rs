// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! Corpus parity with Trafilatura 2.3.1 or 1.11.0 (ignored by default; run in release mode).
//!
//! Inputs: `$PYROID_HTMLX_DATA/<split>/raw.bin` (the HTML strings datatrove passes, u32 LE count
//! then u32 LE length + surrogatepass UTF-8 per document) and references written by
//! `tests/trafilatura_data/ref.py` in `$PYROID_TRAF_REFS/<split><py>/<mode>.jsonl` (`extract`) and
//! `<mode>.bare.jsonl` (`bare_extraction`), or the `.sha1` digests of `ref.py --hash`.
//! `PYROID_TRAF_SPLIT` (default `dev`), `PYROID_TRAF_PY` (`311`..`315`, default `313`: picks the
//! Unicode version), `PYROID_TRAF_MODES` (comma-separated, default all) and
//! `PYROID_TRAF_VERSION` (`2.3.1`, the default, or `1.11.0`, whose references are in
//! `<split><py>-1.11.0/`) select the run. Exceptions are compared by type and message.
//!
//! ```text
//! PYROID_HTMLX_DATA=/target/cc PYROID_TRAF_REFS=/target/out-traf/refs \
//!   cargo test --release -p pyroid-core --test trafilatura_corpus -- --ignored --nocapture
//! ```

#![allow(clippy::print_stderr)] // reports counts and the first differences
#![allow(
    clippy::many_single_char_names,
    clippy::cast_precision_loss,
    clippy::too_many_lines
)] // a test harness

use std::fmt::Write as _;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;

use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::trafilatura::{self, Focus, LruCache, Options, PyError, Version};
use serde_json::Value;

/// Python's `repr()` of a str, near enough for the literals of int() errors (ASCII and
/// printable text; other characters escaped as CPython would for non-printables).
fn py_repr(s: &str) -> String {
    let q = if s.contains('\'') && !s.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut out = String::from(q);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            c if c == q => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || (0x7f..0xa1).contains(&(c as u32)) => {
                let _ = write!(out, "\\x{:02x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push(q);
    out
}

/// `f"{type(e).__name__}: {e}"` for an exception the port reports.
fn py_message(e: &PyError) -> String {
    match e {
        PyError::IntMaxStrDigits { digits } => format!(
            "ValueError: Exceeds the limit (4300 digits) for integer string conversion: value has \
             {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
        ),
        PyError::IntLiteral { literal } => {
            let r: String = py_repr(literal).chars().take(200).collect();
            format!("ValueError: invalid literal for int() with base 10: {r}")
        }
        PyError::Overflow { message } => format!("OverflowError: {message}"),
        PyError::Memory => "MemoryError: ".into(),
        PyError::NoneAttribute { attr } => {
            format!("AttributeError: 'NoneType' object has no attribute '{attr}'")
        }
    }
}

const MODES: &[&str] = &[
    "datatrove",
    "nodedup",
    "fast",
    "fastdedup",
    "default",
    "default_fast",
    "recall",
    "precision_comments",
    "default_dedup",
    "notables",
];

/// (options, deduplicate) for a reference mode of ref.py.
fn mode(name: &str, v: UnicodeVersion, version: Version) -> (Options, bool) {
    let d = Options {
        version,
        unicode: v,
        ..Options::default()
    };
    let p = Options {
        focus: Focus::Precision,
        comments: false,
        ..d.clone()
    };
    match name {
        "datatrove" => (p, true),
        "nodedup" => (p, false),
        "fast" => (Options { fast: true, ..p }, false),
        "fastdedup" => (Options { fast: true, ..p }, true),
        "default" => (d, false),
        "default_fast" => (Options { fast: true, ..d }, false),
        "recall" => (
            Options {
                focus: Focus::Recall,
                ..d
            },
            false,
        ),
        "precision_comments" => (
            Options {
                focus: Focus::Precision,
                ..d
            },
            false,
        ),
        "default_dedup" => (d, true),
        "notables" => (Options { tables: false, ..d }, false),
        _ => panic!("unknown mode {name}"),
    }
}

fn read_bin(path: &PathBuf) -> Vec<Vec<u8>> {
    let mut f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut b = Vec::new();
    f.read_to_end(&mut b).unwrap();
    let u = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap()) as usize;
    let n = u(0);
    let mut out = Vec::with_capacity(n);
    let mut i = 4;
    for _ in 0..n {
        let k = u(i);
        out.push(b[i + 4..i + 4 + k].to_vec());
        i += 4 + k;
    }
    out
}

fn read_jsonl(path: &PathBuf) -> Vec<Value> {
    let s = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    s.lines()
        .filter(|l| !l.is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn show(s: &str, k: usize) -> String {
    let start = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i + 80 <= k)
        .last()
        .unwrap_or(0);
    let end = s
        .char_indices()
        .map(|(i, _)| i)
        .find(|&i| i > k + 100)
        .unwrap_or(s.len());
    format!("{:?}", &s[start..end])
}

/// ref.py `digest`: sha1 over the fields, `N` for None or `S` and the UTF-8, each followed by
/// NUL; a null record is "None".
fn digest(rec: &Value) -> String {
    use sha1::{Digest, Sha1};
    let Some(o) = rec.as_object() else {
        return "None".into();
    };
    if let Some(Value::String(e)) = o.get("error") {
        return format!("!{e}");
    }
    let mut h = Sha1::new();
    for k in ["text", "raw_text", "comments", "body", "commentsbody"] {
        match o.get(k) {
            None => {}
            Some(Value::String(s)) => {
                h.update(b"S");
                h.update(s.as_bytes());
                h.update(b"\0");
            }
            Some(_) => h.update(b"N\0"),
        }
    }
    let mut out = String::new();
    for b in h.finalize() {
        out.push(char::from(b"0123456789abcdef"[usize::from(b >> 4)]));
        out.push(char::from(b"0123456789abcdef"[usize::from(b & 15)]));
    }
    out
}

fn first_diff(a: &str, b: &str) -> usize {
    a.bytes()
        .zip(b.bytes())
        .position(|(x, y)| x != y)
        .unwrap_or(a.len().min(b.len()))
}

#[test]
#[ignore = "needs the CC sample and Trafilatura references (PYROID_HTMLX_DATA, PYROID_TRAF_REFS)"]
fn corpus_parity() {
    let Ok(data) = std::env::var("PYROID_HTMLX_DATA") else {
        eprintln!("PYROID_HTMLX_DATA not set");
        return;
    };
    let refs = std::env::var("PYROID_TRAF_REFS").unwrap_or_else(|_| data.clone());
    let split = std::env::var("PYROID_TRAF_SPLIT").unwrap_or_else(|_| "dev".into());
    let py = std::env::var("PYROID_TRAF_PY").unwrap_or_else(|_| "313".into());
    let v = match py.as_str() {
        "311" => UnicodeVersion::V14_0,
        "312" => UnicodeVersion::V15_0,
        "313" => UnicodeVersion::V15_1,
        "314" => UnicodeVersion::V16_0,
        "315" => UnicodeVersion::V17_0,
        _ => panic!("PYROID_TRAF_PY {py}"),
    };
    let modes: Vec<String> = std::env::var("PYROID_TRAF_MODES").map_or_else(
        |_| MODES.iter().map(|s| (*s).to_owned()).collect(),
        |m| m.split(',').map(str::to_owned).collect(),
    );
    let (version, vdir) = match std::env::var("PYROID_TRAF_VERSION").as_deref() {
        Err(_) | Ok("2.3.1") => (Version::V2_3_1, String::new()),
        Ok("1.11.0") => (Version::V1_11_0, "-1.11.0".to_owned()),
        Ok(x) => panic!("PYROID_TRAF_VERSION {x}"),
    };
    let docs = read_bin(&PathBuf::from(&data).join(&split).join("raw.bin"));
    let refdir = PathBuf::from(&refs).join(format!("{split}{py}{vdir}"));
    let mut failed = 0usize;
    for m in &modes {
        let (bare, name) = match m.strip_suffix(".bare") {
            Some(n) => (true, n),
            None => (false, m.as_str()),
        };
        let (opts, dedup) = mode(name, v, version);
        let stem = if bare {
            format!("{name}.bare")
        } else {
            name.to_owned()
        };
        // full references, or the per-document digests `ref.py --hash` writes
        let path = refdir.join(format!("{stem}.jsonl"));
        let hashed = !path.exists();
        let want: Vec<Value> = if hashed {
            let p = refdir.join(format!("{stem}.sha1"));
            std::fs::read_to_string(&p)
                .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
                .lines()
                .map(|l| Value::String(l.to_owned()))
                .collect()
        } else {
            read_jsonl(&path)
        };
        assert_eq!(want.len(), docs.len(), "{}", path.display());
        let mut cache = LruCache::default();
        let mut diffs = Vec::new();
        let t0 = Instant::now();
        for (i, html) in docs.iter().enumerate() {
            let lru = if dedup { Some(&mut cache) } else { None };
            let got: Value = if bare {
                match trafilatura::bare_extraction(html, &opts, lru) {
                    Err(e) => serde_json::json!({ "error": py_message(&e) }),
                    Ok(None) => Value::Null,
                    Ok(Some(x)) => serde_json::json!({
                        "text": x.text, "raw_text": x.raw_text, "comments": x.comments,
                        "body": x.body,
                        // "" stands for None (1.11.0 without comments)
                        "commentsbody": if x.commentsbody.is_empty() { Value::Null } else { Value::String(x.commentsbody) },
                    }),
                }
            } else {
                match trafilatura::extract(html, &opts, lru) {
                    Ok(t) => serde_json::json!({ "text": t }),
                    Err(e) => serde_json::json!({ "error": py_message(&e) }),
                }
            };
            let got = if hashed {
                Value::String(digest(&got))
            } else {
                got
            };
            if got != want[i] {
                diffs.push((i, got));
            }
        }
        let ms = 1000.0 * t0.elapsed().as_secs_f64() / docs.len() as f64;
        eprintln!(
            "{split} py{py} {m}: {} / {} identical ({ms:.2} ms/doc)",
            docs.len() - diffs.len(),
            docs.len()
        );
        for (i, got) in diffs.iter().take(3) {
            let w = &want[*i];
            let field = ["text", "raw_text", "comments", "body", "commentsbody"]
                .into_iter()
                .find(|f| w.get(f) != got.get(f))
                .unwrap_or("text");
            let (ws, gs) = (
                w.get(field).and_then(Value::as_str),
                got.get(field).and_then(Value::as_str),
            );
            match (ws, gs) {
                (Some(a), Some(b)) => {
                    let k = first_diff(a, b);
                    eprintln!(
                        "  [{i}] {field}: first diff at byte {k} of {} / {}",
                        a.len(),
                        b.len()
                    );
                    eprintln!("    want: {}", show(a, k));
                    eprintln!("    got : {}", show(b, k));
                }
                _ => eprintln!(
                    "  [{i}] {field}: want {} got {}",
                    w.get(field).unwrap_or(w),
                    got.get(field).unwrap_or(got)
                ),
            }
        }
        if let Ok(dir) = std::env::var("PYROID_TRAF_DUMP") {
            let ids: Vec<String> = diffs.iter().map(|(i, _)| i.to_string()).collect();
            std::fs::write(
                PathBuf::from(dir).join(format!("{split}{py}-{m}.diffs")),
                ids.join("\n"),
            )
            .unwrap();
        }
        failed += diffs.len();
    }
    assert_eq!(failed, 0, "documents differ from Trafilatura");
}
