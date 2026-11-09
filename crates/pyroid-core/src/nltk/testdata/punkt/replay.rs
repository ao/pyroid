// Derived from NLTK 3.10.4 (Apache-2.0), © 2001-2026 NLTK Project; modified.
//! Differential tests: replay what NLTK 3.10.4's Punkt returned, as recorded
//! by `record.py` in this directory, and require identical results.
//!
//! * `golden-{nltk,adversarial,fuzz,fineweb,tokens}.jsonl`: recorded on
//!   CPython 3.13 with the parameter sets `params-*.json` (trained by
//!   `PunktTrainer` on FineWeb and FineWeb-2 text, or written by hand) and no
//!   parameters (`empty`).
//! * `golden-unicode-<version>.jsonl`: texts whose results depend on the
//!   CPython Unicode version, one file per interpreter, with
//!   `params-unicode.json`.
//! * The `punkt_tab` models (not committed): `pretrained_models` replays
//!   `$PYROID_PUNKT_RECORDS/<lang>.jsonl` with the model in
//!   `$PYROID_PUNKT_TAB/<lang>/`, and is skipped unless both are set.
//!
//! `PYROID_PUNKT_GOLDEN_DIR` replays the `golden-*.jsonl` and `params-*.json`
//! of another recording instead (`record.py fuzz DIR N`).
//!
//! `record.py`'s docstring describes the rows.

#![allow(clippy::print_stderr)] // reports what was replayed

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value, json};

use super::{Decision, LangVars, OrthoHeuristic, Params, Punkt, Raised, Reason, char_offsets};
use crate::ftfy::UnicodeVersion;
use crate::nltk::chars::py_space;

/// Mismatches shown in full before a test fails.
const MAX_EXAMPLES: usize = 12;
/// Characters of a value shown in a failure message.
const MAX_SHOWN: usize = 300;

/// `REASONS` in `record.py`.
const REASONS: [&str; 6] = [
    "default decision",
    "known collocation (both words)",
    "abbreviation + orthographic heuristic",
    "abbreviation + frequent sentence starter",
    "initial + orthographic heuristic",
    "initial + special orthographic heuristic",
];

type Row = Map<String, Value>;

/// `testdata/punkt/name`, or `$PYROID_PUNKT_GOLDEN_DIR/name`.
fn testdata(name: &str) -> PathBuf {
    std::env::var_os("PYROID_PUNKT_GOLDEN_DIR")
        .map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("src/nltk/testdata/punkt"),
            PathBuf::from,
        )
        .join(name)
}

/// The `.jsonl` files in the golden directory whose names start with `prefix`.
fn golden_files(prefix: &str) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(testdata(""))
        .expect("the golden directory")
        .map(|e| e.expect("an entry").path())
        .filter(|p| {
            p.extension().is_some_and(|x| x == "jsonl")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with(prefix))
        })
        .collect();
    files.sort();
    files
}

fn read_rows(path: &Path) -> Vec<Row> {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    data.lines()
        .enumerate()
        .map(|(i, line)| match serde_json::from_str(line) {
            Ok(Value::Object(row)) => row,
            _ => panic!("{}:{}: not a JSON object", path.display(), i + 1),
        })
        .collect()
}

fn strs(v: &Value) -> Vec<String> {
    v.as_array()
        .expect("a list")
        .iter()
        .map(|s| s.as_str().expect("a string").to_owned())
        .collect()
}

/// A `params-<name>.json` file (`record.py`'s `params_to_json`).
fn load_params(name: &str) -> Params {
    if name == "empty" {
        return Params::default();
    }
    let path = testdata(&format!("params-{name}.json"));
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let d: Value = serde_json::from_str(&data).expect("JSON");
    let pairs = d["collocations"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|p| {
            let p = strs(p);
            (p[0].clone(), p[1].clone())
        });
    let ortho = d["ortho_context"]
        .as_array()
        .expect("a list")
        .iter()
        .map(|p| {
            (
                p[0].as_str().expect("a type").to_owned(),
                p[1].as_i64().expect("an int"),
            )
        });
    Params::new(
        strs(&d["abbrev_types"]),
        pairs.collect::<Vec<_>>(),
        strs(&d["sent_starters"]),
        ortho.collect::<Vec<_>>(),
    )
}

// ---------------------------------------------------------------------------
// Results in record.py's format
// ---------------------------------------------------------------------------

fn spans_value(text: &str, spans: &[(usize, usize)]) -> Value {
    let mut flat: Vec<usize> = spans.iter().flat_map(|&(s, e)| [s, e]).collect();
    char_offsets(text, &mut flat);
    json!(flat)
}

fn generator_value<T>(r: &Result<Vec<T>, Raised<T>>, item: impl Fn(&T) -> Value) -> Value {
    match r {
        Ok(items) => Value::Array(items.iter().map(item).collect()),
        Err(raised) => json!({
            "e": raised.error.python_type(),
            "m": raised.error.message(),
            "y": raised.yielded.iter().map(item).collect::<Vec<_>>(),
        }),
    }
}

fn decision_value(text: &str, d: &Decision) -> Value {
    let mut at = [d.period_index];
    char_offsets(text, &mut at);
    let heuristic = match d.type2_ortho_heuristic {
        OrthoHeuristic::Yes => json!(true),
        OrthoHeuristic::No => json!(false),
        OrthoHeuristic::Unknown => json!("unknown"),
    };
    let reason = REASONS
        .iter()
        .position(|&r| r == d.reason.as_str())
        .expect("a known reason");
    json!([
        at[0],
        d.text,
        d.type1,
        d.type2,
        u8::from(d.type1_in_abbrs),
        u8::from(d.type1_is_initial),
        u8::from(d.type2_is_sent_starter),
        heuristic,
        d.type2_ortho_contexts,
        u8::from(d.collocation),
        reason,
        d.break_decision,
    ])
}

fn decisions_value(text: &str, r: &Result<Vec<Decision>, Raised<Decision>>) -> Value {
    match r {
        Ok(ds) => Value::Array(ds.iter().map(|d| decision_value(text, d)).collect()),
        Err(raised) => {
            let mut v: Vec<Value> = raised
                .yielded
                .iter()
                .map(|d| decision_value(text, d))
                .collect();
            v.push(json!({"e": raised.error.python_type(), "m": raised.error.message()}));
            Value::Array(v)
        }
    }
}

/// `str.split()`.
fn py_split(text: &str) -> Vec<&str> {
    text.split(py_space).filter(|s| !s.is_empty()).collect()
}

// ---------------------------------------------------------------------------
// Replaying rows
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Report {
    rows: usize,
    texts: usize,
    sentences: usize,
    decisions: usize,
    checks: usize,
    failures: usize,
    examples: String,
}

impl Report {
    fn check(&mut self, what: &str, input: &str, want: &Value, got: &Value) {
        self.checks += 1;
        if want == got {
            return;
        }
        self.failures += 1;
        if self.failures <= MAX_EXAMPLES {
            let _ = writeln!(
                self.examples,
                "{what}\n  input: {}\n  want:  {}\n  got:   {}",
                shown(&Value::String(input.to_owned())),
                shown(want),
                shown(got)
            );
        }
    }

    fn add(&mut self, other: &Self) {
        self.rows += other.rows;
        self.texts += other.texts;
        self.sentences += other.sentences;
        self.decisions += other.decisions;
        self.checks += other.checks;
        self.failures += other.failures;
        if self.examples.lines().count() < 4 * MAX_EXAMPLES {
            self.examples.push_str(&other.examples);
        }
    }
}

fn shown(v: &Value) -> String {
    let s = v.to_string();
    if s.chars().count() <= MAX_SHOWN {
        s
    } else {
        s.chars().take(MAX_SHOWN).collect::<String>() + "..."
    }
}

/// Replay one row with `punkt`; `at` names it in messages.
fn replay_row(punkt: &Punkt, row: &Row, at: &str, report: &mut Report) {
    report.rows += 1;
    if let Some(tokens) = row.get("k") {
        let tokens = strs(tokens);
        let got = generator_value(&punkt.sentences_from_tokens(&tokens), |&n| json!(n));
        report.check(
            &format!("{at}: sentences_from_tokens"),
            &tokens.join(" | "),
            &row["r"],
            &got,
        );
        return;
    }
    let text = row["t"].as_str().expect("a text");
    report.texts += 1;

    let realigned = punkt.span_tokenize(text, true);
    let raw = punkt.span_tokenize(text, false);
    report.sentences += realigned.len();
    let s = spans_value(text, &realigned);
    let n = spans_value(text, &raw);
    report.check(&format!("{at}: span_tokenize"), text, &row["s"], &s);
    let want_n = if row["n"] == "=s" {
        &row["s"]
    } else {
        &row["n"]
    };
    report.check(
        &format!("{at}: span_tokenize(realign_boundaries=False)"),
        text,
        want_n,
        &n,
    );

    report.check(
        &format!("{at}: text_contains_sentbreak"),
        text,
        &row["b"],
        &json!(punkt.text_contains_sentbreak(text)),
    );

    if let Some(want) = row.get("g") {
        let want = match want.as_str() {
            Some("=s") => json!(punkt.tokenize(text, true)),
            Some("=n") => json!(punkt.tokenize(text, false)),
            _ => want.clone(),
        };
        let got = generator_value(&punkt.sentences_from_text_legacy(text), |s| json!(s));
        report.check(
            &format!("{at}: sentences_from_text_legacy"),
            text,
            &want,
            &got,
        );
    }

    if let Some(want) = row.get("d") {
        let got = punkt.debug_decisions(text);
        report.decisions += got.as_ref().map_or_else(|r| r.yielded.len(), Vec::len);
        report.check(
            &format!("{at}: debug_decisions"),
            text,
            want,
            &decisions_value(text, &got),
        );
    }

    if let Some(want) = row.get("w") {
        let got: Vec<&str> = punkt.words(text).collect();
        report.check(&format!("{at}: _tokenize_words"), text, want, &json!(got));
    }

    if let Some(want) = row.get("ks") {
        let got = generator_value(&punkt.sentences_from_tokens(&py_split(text)), |&n| json!(n));
        report.check(
            &format!("{at}: sentences_from_tokens(text.split())"),
            text,
            want,
            &got,
        );
    }
}

fn replay_golden(path: &Path, report: &mut Report) {
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let mut punkts: HashMap<(String, String, UnicodeVersion), Punkt> = HashMap::new();
    for (i, row) in read_rows(path).iter().enumerate() {
        let params = row["p"].as_str().expect("a parameter set").to_owned();
        let ends = row
            .get("l")
            .and_then(Value::as_str)
            .unwrap_or(".?!")
            .to_owned();
        // Rows without "v" were recorded on CPython 3.13.
        let version = row
            .get("v")
            .and_then(Value::as_str)
            .map_or(UnicodeVersion::V15_1, |v| {
                UnicodeVersion::from_unidata_version(v).expect("a known version")
            });
        let punkt = punkts
            .entry((params.clone(), ends.clone(), version))
            .or_insert_with(|| {
                Punkt::new(
                    load_params(&params),
                    LangVars::new(ends.chars().collect()),
                    version,
                )
            });
        replay_row(punkt, row, &format!("{name}:{}", i + 1), report);
    }
}

fn finish(what: &str, report: &Report) {
    eprintln!(
        "{what}: {} rows ({} texts, {} sentences, {} decisions), {} checks, {} differences",
        report.rows,
        report.texts,
        report.sentences,
        report.decisions,
        report.checks,
        report.failures
    );
    assert!(
        report.failures == 0,
        "{what}: {} of {} checks differ from NLTK:\n{}",
        report.failures,
        report.checks,
        report.examples
    );
}

#[test]
fn goldens() {
    let mut total = Report::default();
    let files: Vec<PathBuf> = golden_files("golden-")
        .into_iter()
        .filter(|p| !p.to_string_lossy().contains("golden-unicode-"))
        .collect();
    assert!(!files.is_empty(), "no golden-*.jsonl files");
    for path in files {
        let mut report = Report::default();
        replay_golden(&path, &mut report);
        eprintln!(
            "{}: {} rows, {} differences",
            path.display(),
            report.rows,
            report.failures
        );
        total.add(&report);
    }
    finish("goldens", &total);
}

#[test]
fn unicode_versions() {
    let mut total = Report::default();
    let files = golden_files("golden-unicode-");
    if files.is_empty() && std::env::var_os("PYROID_PUNKT_GOLDEN_DIR").is_some() {
        return;
    }
    let params = load_params("unicode");
    assert!(
        files.len() >= 2,
        "expected recordings for several Unicode versions"
    );
    for path in files {
        let rows = read_rows(&path);
        let version = rows[0]["v"].as_str().expect("a version");
        let version = UnicodeVersion::from_unidata_version(version).expect("a known version");
        let punkt = Punkt::new(params.clone(), LangVars::default(), version);
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_owned();
        let mut report = Report::default();
        for (i, row) in rows.iter().enumerate() {
            replay_row(&punkt, row, &format!("{name}:{}", i + 1), &mut report);
        }
        eprintln!(
            "{name}: {} rows, {} differences",
            report.rows, report.failures
        );
        total.add(&report);
    }
    finish("unicode versions", &total);
}

/// `load_punkt_params` on `dir`.
fn load_punkt_tab(dir: &Path) -> Params {
    let read = |name: &str| {
        let path = dir.join(name);
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        String::from_utf8(bytes).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
    };
    Params::from_punkt_tab(
        &read("collocations.tab"),
        &read("sent_starters.txt"),
        &read("abbrev_types.txt"),
        &read("ortho_context.tab"),
    )
    .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
}

#[test]
fn pretrained_models() {
    let (Some(models), Some(records)) = (
        std::env::var_os("PYROID_PUNKT_TAB"),
        std::env::var_os("PYROID_PUNKT_RECORDS"),
    ) else {
        eprintln!(
            "skipped: set PYROID_PUNKT_TAB and PYROID_PUNKT_RECORDS to replay the punkt_tab models"
        );
        return;
    };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&records)
        .expect("PYROID_PUNKT_RECORDS")
        .map(|e| e.expect("an entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no <lang>.jsonl in PYROID_PUNKT_RECORDS");
    let mut total = Report::default();
    for path in files {
        let lang = path
            .file_stem()
            .and_then(|n| n.to_str())
            .expect("a name")
            .to_owned();
        let params = load_punkt_tab(&Path::new(&models).join(&lang));
        let punkt = Punkt::new(params, LangVars::default(), UnicodeVersion::V15_1);
        let mut report = Report::default();
        for (i, row) in read_rows(&path).iter().enumerate() {
            replay_row(&punkt, row, &format!("{lang}:{}", i + 1), &mut report);
        }
        eprintln!(
            "{lang}: {} docs, {} sentences, {} decisions, {} checks, {} differences",
            report.texts, report.sentences, report.decisions, report.checks, report.failures
        );
        total.add(&report);
    }
    finish("punkt_tab models", &total);
}

#[test]
fn reasons_cover_nltk() {
    // Every reason text is one of record.py's, and the number reason shares
    // the initial one's text, as in NLTK.
    for r in [
        Reason::DefaultDecision,
        Reason::KnownCollocation,
        Reason::AbbrWithOrthographicHeuristic,
        Reason::AbbrWithSentenceStarter,
        Reason::InitialWithOrthographicHeuristic,
        Reason::NumberWithOrthographicHeuristic,
        Reason::InitialWithSpecialOrthographicHeuristic,
    ] {
        assert!(REASONS.contains(&r.as_str()));
    }
    assert_eq!(
        Reason::NumberWithOrthographicHeuristic.as_str(),
        Reason::InitialWithOrthographicHeuristic.as_str()
    );
}
