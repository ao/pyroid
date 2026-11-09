// Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
//! Differential tests: replay what NLTK 3.10.4 returned, as recorded by
//! `record.py` (next to this file), through `nltk::treebank` and
//! `nltk::regexp`, and require identical results.
//!
//! * `golden.jsonl` (committed): NLTK's test sentences, an adversarial set,
//!   FineWeb and FineWeb-2 sentences and seeded fuzz, recorded on CPython
//!   3.13 (Unicode 15.1).
//! * `golden-regexp.jsonl` (committed): the regexp and simple tokenizers on
//!   their doctest texts, fuzz and FineWeb snippets.
//! * `PYROID_TREEBANK_RECORDS=DIR`: every `*.jsonl` file in `DIR`, from
//!   `record.py large` (rows) or `record.py codepoints` (code point
//!   templates, replayed in the Unicode version of the recording).
//!
//! `record.py` describes the fields.

#![allow(clippy::print_stderr)] // reports what was replayed

use std::path::Path;

use rayon::prelude::*;
use serde_json::{Map, Value, json};

use super::{
    AlignError, Flavor, UnicodeVersion, align_tokens, detokenize, span_tokenize, tokenize_text,
    tokens,
};
use crate::nltk::regexp::{self, Blanklines, Pattern, RegexpTokenizer};

type Row = Map<String, Value>;

/// Mismatches shown before a test fails.
const MAX_SHOWN: usize = 12;

fn read(path: &Path) -> (Row, Vec<Row>) {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let mut rows = data
        .lines()
        .enumerate()
        .map(|(i, line)| match serde_json::from_str(line) {
            Ok(Value::Object(row)) => row,
            _ => panic!("{}:{}: not a JSON object", path.display(), i + 1),
        });
    let header = rows.next().expect("a header line");
    (header, rows.collect())
}

fn version(header: &Row) -> UnicodeVersion {
    let v = header["unidata"].as_str().expect("unidata");
    UnicodeVersion::from_unidata_version(v).unwrap_or_else(|| panic!("Unicode {v}"))
}

fn joined(text: &str) -> String {
    tokens(text).collect::<Vec<_>>().join(" ")
}

/// The tokens of a recorded `" ".join(tokens)`.
fn split_joined(s: &str) -> Vec<&str> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(' ').collect()
    }
}

fn outcome(r: &Result<Vec<(usize, usize)>, AlignError>) -> Value {
    match r {
        Ok(spans) => spans.iter().flat_map(|&(a, e)| [a, e]).collect(),
        Err(e) => {
            let kind = match e {
                AlignError::NotFound { .. } => "ValueError",
                AlignError::QuotesExhausted => "RuntimeError",
            };
            json!({"e": kind, "m": e.to_string()})
        }
    }
}

/// Collects the checks of one row.
#[derive(Default)]
struct Tally {
    checks: usize,
    failures: Vec<String>,
}

impl Tally {
    fn check(&mut self, what: &str, text: &str, got: &Value, want: &Value) {
        self.checks += 1;
        if got != want {
            self.failures.push(format!(
                "{what} of {text:?}:\n    got  {got}\n    want {want}"
            ));
        }
    }

    fn merge(mut self, other: Self) -> Self {
        self.checks += other.checks;
        self.failures.extend(other.failures);
        self
    }
}

fn str_list(v: &Value) -> Vec<&str> {
    v.as_array()
        .expect("a list")
        .iter()
        .map(|t| t.as_str().expect("a string"))
        .collect()
}

/// Checks one "rows" record.
fn check_row(row: &Row, v: UnicodeVersion) -> Tally {
    let mut t = Tally::default();
    let text = row["t"].as_str().expect("t");
    let field = |k: &str, default: &str| row.get(k).unwrap_or(&row[default]);
    for (key, default, flavor, convert) in [
        ("tb", "tb", Flavor::Treebank, false),
        ("tbp", "tb", Flavor::Treebank, true),
        ("nl", "nl", Flavor::Nltk, false),
        ("nlp", "nl", Flavor::Nltk, true),
    ] {
        let got = joined(&tokenize_text(text, flavor, convert, v));
        t.check(key, text, &got.into(), field(key, default));
    }
    for (key, flavor) in [("tbs", Flavor::Treebank), ("nls", Flavor::Nltk)] {
        t.check(
            key,
            text,
            &outcome(&span_tokenize(text, flavor, v)),
            &row[key],
        );
    }
    let rec = |k: &str, default: &str| split_joined(field(k, default).as_str().expect("tokens"));
    for (key, toks, convert) in [
        ("dnl", rec("nl", "nl"), false),
        ("dtb", rec("tb", "tb"), false),
        ("dnlp", rec("nlp", "nl"), true),
    ] {
        t.check(
            key,
            text,
            &detokenize(&toks, convert).into(),
            field(key, "dnl"),
        );
    }
    if let Some(edits) = row.get("x") {
        let edits = edits.as_array().expect("x");
        for (i, x) in edits.iter().enumerate() {
            let x = str_list(x);
            let what = format!("detokenize({x:?})");
            t.check(&what, text, &detokenize(&x, false).into(), &row["dx"][i]);
            let want = row.get("dxp").unwrap_or(&row["dx"]);
            t.check(&what, text, &detokenize(&x, true).into(), &want[i]);
            if let Some(ax) = row.get("ax") {
                let what = format!("align_tokens({x:?})");
                t.check(&what, text, &outcome(&align_tokens(&x, text)), &ax[i]);
            }
        }
    }
    t
}

/// Replaces U+0000 (the code point's placeholder) with `c` in every string.
fn decode(v: &Value, c: &str) -> Value {
    match v {
        Value::String(s) => s.replace('\0', c).into(),
        Value::Array(a) => a.iter().map(|x| decode(x, c)).collect(),
        Value::Object(o) => o.iter().map(|(k, x)| (k.clone(), decode(x, c))).collect(),
        other => other.clone(),
    }
}

/// Runs `check(text, expected)` for the template of one code point record
/// with every code point of its ranges.
fn for_each_codepoint(row: &Row, check: impl Fn(&str, &Value) -> Tally + Sync) -> Tally {
    let tpl = row["tpl"].as_str().expect("tpl");
    let ranges: Vec<(u32, u32)> = row["cps"]
        .as_array()
        .expect("cps")
        .iter()
        .map(|r| {
            let n = |i: usize| u32::try_from(r[i].as_u64().expect("code point")).unwrap();
            (n(0), n(1))
        })
        .collect();
    ranges
        .par_iter()
        .flat_map_iter(|&(lo, hi)| lo..=hi)
        .map(|cp| {
            let c = char::from_u32(cp).expect("not a surrogate").to_string();
            check(&tpl.replace("{c}", &c), &decode(&row["r"], &c))
        })
        .reduce(Tally::default, Tally::merge)
}

/// Checks one "codepoints" record.
fn check_codepoints(row: &Row, v: UnicodeVersion) -> Tally {
    for_each_codepoint(row, |text, r| {
        let mut t = Tally::default();
        for (key, flavor) in [("tb", Flavor::Treebank), ("nl", Flavor::Nltk)] {
            let got = joined(&tokenize_text(text, flavor, false, v));
            t.check(key, text, &got.into(), &r[key]);
        }
        for (key, flavor) in [("tbs", Flavor::Treebank), ("nls", Flavor::Nltk)] {
            if let Some(want) = r.get(key) {
                t.check(key, text, &outcome(&span_tokenize(text, flavor, v)), want);
            }
        }
        if let Some(want) = r.get("dnl") {
            let toks = split_joined(r["nl"].as_str().expect("nl"));
            t.check("dnl", text, &detokenize(&toks, false).into(), want);
        }
        if let Some(want) = r.get("d1") {
            t.check("d1", text, &detokenize(&[text], false).into(), want);
        }
        t
    })
}

// nltk::regexp

/// `record.py`'s REGEXP_PATTERNS.
const PATTERNS: [Pattern; 8] = [
    Pattern::Word,
    Pattern::WordPunct,
    Pattern::WordMoneyNonSpace,
    Pattern::WordApostrophe,
    Pattern::NonSpace,
    Pattern::Space,
    Pattern::AsciiLetters,
    Pattern::Blankline,
];

/// `record.py`'s LINE_MODES.
const LINE_MODES: [Blanklines; 3] = [
    Blanklines::Discard,
    Blanklines::Keep,
    Blanklines::DiscardEof,
];

fn flat(spans: &[(usize, usize)]) -> Value {
    spans.iter().flat_map(|&(a, e)| [a, e]).collect()
}

fn strs(toks: &[&str]) -> Value {
    toks.iter().copied().collect()
}

/// `record.py`'s `regexp_fields(text)`, computed by `nltk::regexp`.
fn regexp_fields(text: &str) -> Map<String, Value> {
    let mut r = Map::new();
    let f = PATTERNS.map(|p| {
        let t = RegexpTokenizer::new(p, false, true);
        json!([strs(&t.tokenize(text)), flat(&t.span_tokenize(text))])
    });
    r.insert("f".into(), f.into_iter().collect());
    let g = PATTERNS.map(|p| {
        let [d, keep] = [true, false].map(|d| RegexpTokenizer::new(p, true, d));
        json!([
            strs(&d.tokenize(text)),
            flat(&d.span_tokenize(text)),
            strs(&keep.tokenize(text)),
            flat(&keep.span_tokenize(text)),
        ])
    });
    r.insert("g".into(), g.into_iter().collect());
    let named = json!([
        strs(&regexp::wordpunct_tokenize(text)),
        flat(&regexp::WORD_PUNCT.span_tokenize(text)),
        strs(&regexp::WHITESPACE.tokenize(text)),
        flat(&regexp::WHITESPACE.span_tokenize(text)),
        strs(&regexp::blankline_tokenize(text)),
        flat(&regexp::BLANKLINE.span_tokenize(text)),
        strs(&regexp::space_tokenize(text)),
        flat(&regexp::space_span_tokenize(text)),
        strs(&regexp::tab_tokenize(text)),
        flat(&regexp::tab_span_tokenize(text)),
    ]);
    r.insert("named".into(), named);
    let lines = LINE_MODES.map(|m| {
        json!([
            strs(&regexp::line_tokenize(text, m)),
            flat(&regexp::line_span_tokenize(text, m))
        ])
    });
    r.insert("lines".into(), lines.into_iter().collect());
    r
}

fn compare_regexp(text: &str, want: &Value) -> Tally {
    let mut t = Tally::default();
    for (key, got) in regexp_fields(text) {
        t.check(&key, text, &got, &want[&key]);
    }
    t
}

/// Checks one "regexp" record.
fn check_regexp(row: &Row, _: UnicodeVersion) -> Tally {
    let text = row["t"].as_str().expect("t");
    compare_regexp(text, &Value::Object(row.clone()))
}

/// Checks one "regexp-codepoints" record.
fn check_regexp_codepoints(row: &Row, _: UnicodeVersion) -> Tally {
    for_each_codepoint(row, compare_regexp)
}

/// Replays one file; returns (records, checks, failures).
fn replay_file(path: &Path) -> (usize, Tally) {
    let (header, rows) = read(path);
    let v = version(&header);
    let check = match header["kind"].as_str() {
        Some("rows") => check_row,
        Some("codepoints") => check_codepoints,
        Some("regexp") => check_regexp,
        Some("regexp-codepoints") => check_regexp_codepoints,
        k => panic!("{}: unknown kind {k:?}", path.display()),
    };
    let tally = rows
        .par_iter()
        .map(|row| check(row, v))
        .reduce(Tally::default, Tally::merge);
    (rows.len(), tally)
}

fn report(name: &str, records: usize, tally: &Tally) {
    eprintln!(
        "{name}: {records} records, {} checks, {} differences",
        tally.checks,
        tally.failures.len()
    );
    for f in tally.failures.iter().take(MAX_SHOWN) {
        eprintln!("  {f}");
    }
}

fn replay_testdata(name: &str) -> (usize, Tally) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/nltk/testdata/treebank");
    let (records, tally) = replay_file(&dir.join(name));
    report(name, records, &tally);
    (records, tally)
}

#[test]
fn golden() {
    let (records, tally) = replay_testdata("golden.jsonl");
    assert!(records > 3000 && tally.checks > 35_000);
    assert_eq!(tally.failures.len(), 0);
}

#[test]
fn golden_regexp() {
    let (records, tally) = replay_testdata("golden-regexp.jsonl");
    assert!(records > 200);
    assert_eq!(tally.failures.len(), 0);
}

/// `PYROID_TREEBANK_RECORDS=DIR`: every recording in `DIR`.
#[test]
fn records() {
    let Some(dir) = std::env::var_os("PYROID_TREEBANK_RECORDS") else {
        return;
    };
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("PYROID_TREEBANK_RECORDS")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "jsonl"))
        .collect();
    paths.sort();
    assert_ne!(paths.len(), 0, "no .jsonl files");
    let mut failed = 0;
    let (mut all_records, mut all_checks) = (0, 0);
    for path in &paths {
        let (records, tally) = replay_file(path);
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        report(&name, records, &tally);
        all_records += records;
        all_checks += tally.checks;
        failed += tally.failures.len();
    }
    eprintln!("total: {all_records} records, {all_checks} checks, {failed} differences");
    assert_eq!(failed, 0);
}
