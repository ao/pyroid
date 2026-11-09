//! Replays of the pyvi 0.1.1 / spaCy 3.8.18 / datatrove 0.10.1 recordings in
//! `testdata/` (written by `testdata/oracle.py`).
//!
//! Each row gives a text and what the references returned for it: the
//! syllables (`seg`, lengths in code points), the CRF labels (`lab`),
//! `pyvi_tokenize` (`tok`) and datatrove's `word_tokenize` (`w`),
//! `span_tokenize` (`sp`) and `sent_tokenize` (`s`). The header's Unicode
//! version picks the tokenizer. `PYROID_PYVI_GOLDEN_DIR` replays the files of
//! another recording (an uncommitted `--full` run) instead of `testdata/`;
//! `PYROID_PYVI_REPLAY` names further files (comma-separated).
#![allow(clippy::print_stderr)] // reports what was replayed

use std::path::{Path, PathBuf};

use serde_json::Value;

use super::Segmenter;
use super::vi::ViTokenizer;
use crate::filters::WordTokenizer;
use crate::ftfy::UnicodeVersion;

fn golden_dir() -> PathBuf {
    std::env::var_os("PYROID_PYVI_GOLDEN_DIR").map_or_else(
        || PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/src/pyvi/testdata")),
        PathBuf::from,
    )
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

fn strings(v: &Value) -> Vec<String> {
    v.as_array().expect("a list").iter().map(text).collect()
}

fn numbers(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("a list")
        .iter()
        .map(|x| usize::try_from(x.as_u64().expect("a number")).unwrap())
        .collect()
}

fn n_chars(s: &str) -> usize {
    s.chars().count()
}

fn short(t: &str) -> String {
    let s: String = t.chars().take(60).collect();
    format!("{s:?}{}", if n_chars(t) > 60 { "..." } else { "" })
}

/// Replay one file; returns (rows, compared items, failures).
fn replay_file(path: &Path) -> (usize, usize, Vec<String>) {
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut lines = data.lines();
    let header: Value = serde_json::from_str(lines.next().expect("a header")).unwrap();
    let unicode = header["header"]["unicode"]
        .as_str()
        .expect("header.unicode");
    let version = UnicodeVersion::from_unidata_version(unicode).expect("a known Unicode version");
    let tok = ViTokenizer::get(version);
    let seg = Segmenter::get();
    let (mut rows, mut items, mut fails) = (0usize, 0usize, Vec::new());
    for (i, line) in lines.enumerate() {
        let row: Value = serde_json::from_str(line).unwrap();
        let t = text(&row["t"]);
        rows += 1;
        let mut fail = |what: &str, want: String, got: String| {
            if fails.len() < 30 {
                fails.push(format!(
                    "{} row {i} {what}: {}\n    want {want}\n    got  {got}",
                    path.file_name().unwrap().to_string_lossy(),
                    short(&t)
                ));
            } else {
                fails.push(String::new());
            }
        };
        if let Some(want) = row.get("seg") {
            let mut got = Vec::new();
            Segmenter::syllables(&t, version, &mut got);
            let got: Vec<usize> = got.into_iter().map(|r| n_chars(&t[r])).collect();
            let want = numbers(want);
            items += want.len();
            if got != want {
                fail("seg", format!("{want:?}"), format!("{got:?}"));
            }
        }
        if let Some(want) = row.get("lab") {
            let want = want.as_str().expect("labels");
            let got: String = seg
                .labels(&t, version)
                .into_iter()
                .map(|i| if i { 'I' } else { 'B' })
                .collect();
            items += want.len();
            if got != want {
                fail("lab", want.to_owned(), got);
            }
        }
        if let Some(want) = row.get("tok") {
            let want = strings(want);
            let mut got = Vec::new();
            tok.pyvi_words(&t, &mut got);
            let got: Vec<&str> = got.into_iter().map(|r| &t[r]).collect();
            items += want.len();
            if got != want {
                fail("tok", format!("{want:?}"), format!("{got:?}"));
            }
        }
        if let Some(want) = row.get("w") {
            let want = strings(want);
            let mut got = Vec::new();
            tok.words(&t, &mut got).unwrap();
            items += want.len();
            if got != want {
                fail("w", format!("{want:?}"), format!("{got:?}"));
            }
        }
        if let Some(want) = row.get("s") {
            let want = strings(want);
            let mut got = Vec::new();
            tok.sentences(&t, &mut got).unwrap();
            items += want.len();
            if got != want {
                fail("s", format!("{want:?}"), format!("{got:?}"));
            }
        }
        if let Some(want) = row.get("sp") {
            let want: Vec<(usize, usize)> = want
                .as_array()
                .expect("spans")
                .iter()
                .map(|p| {
                    let p = numbers(p);
                    (p[0], p[1])
                })
                .collect();
            let got = spans(tok, &t);
            items += want.len();
            if got != want {
                fail("sp", format!("{want:?}"), format!("{got:?}"));
            }
        }
    }
    (rows, items, fails)
}

/// `span_tokenize(text)` in code points: each sentence from its first
/// token's start to its last token's end.
fn spans(tok: &ViTokenizer, t: &str) -> Vec<(usize, usize)> {
    let mut toks = Vec::new();
    tok.tokens(t, &mut toks);
    let mut ends = Vec::new();
    tok.sentence_ends(t, &mut ends).unwrap();
    let mut out = Vec::new();
    let mut k = 0;
    let mut c = 0usize; // chars before toks[k].0, tracked incrementally
    let mut at = 0usize;
    for e in ends {
        let start = toks[k].0;
        c += n_chars(&t[at..start]);
        at = start;
        let cs = c;
        while toks[k].1 != e {
            k += 1;
        }
        k += 1;
        c += n_chars(&t[at..e]);
        at = e;
        out.push((cs, c));
    }
    out
}

fn report(name: &str, rows: usize, items: usize, fails: &[String]) {
    eprintln!(
        "{name}: {rows} rows, {items} items, {} differences",
        fails.len()
    );
    for f in fails.iter().filter(|f| !f.is_empty()) {
        eprintln!("  {f}");
    }
}

fn replay_named(name: &str) {
    let path = golden_dir().join(name);
    if !path.exists() {
        eprintln!("{name}: not recorded yet, skipped");
        return;
    }
    let (rows, items, fails) = replay_file(&path);
    report(name, rows, items, &fails);
    assert!(fails.is_empty(), "{name}: {} differences", fails.len());
}

#[test]
fn cases() {
    replay_named("cases.jsonl");
}

#[test]
fn fineweb() {
    replay_named("fineweb.jsonl");
}

#[test]
fn fuzz() {
    replay_named("fuzz.jsonl");
}

#[test]
fn sweep() {
    replay_named("sweep.jsonl");
}

#[test]
fn extra_recordings() {
    let Ok(paths) = std::env::var("PYROID_PYVI_REPLAY") else {
        return;
    };
    let mut all = Vec::new();
    for p in paths.split(',').filter(|p| !p.is_empty()) {
        let (rows, items, fails) = replay_file(Path::new(p));
        report(p, rows, items, &fails);
        all.extend(fails);
    }
    assert!(all.is_empty(), "{} differences", all.len());
}
