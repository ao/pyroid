// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace, and
// indic-nlp-library 0.92 (MIT), © Anoop Kunchukuttan; modified.
//! Differential tests: replay what datatrove 0.10.1's `IndicNLPTokenizer`
//! (indic-nlp-library 0.92, urduhack 1.0.3 for `ur` sentences) really
//! returns, as recorded by `tests/filters_data/indic/record.py`, and require
//! identical results from `pyroid_core::filters::indic::IndicNlp`:
//!
//! * `"w"`: `split_into_words(text, lang)` vs `IndicNlp::word_tokenize`
//!   (and the `WordTokenizer::words` view of it),
//! * `"z"`: `sent_tokenize(text)` vs `IndicNlp::sent_tokenize`,
//! * `"e"`: the ends of `span_tokenize(text)` vs `sentence_ends`,
//! * `"s"`: `len(split_into_sentences(text, lang))` vs `count_sentences`.
//!
//! A recorded exception (`"w!"`, ...) must be reproduced: `ValueError`s
//! (`substring not found`, urduhack's `chr()` range error) are
//! `FilterError::SubstringNotFound`.
//!
//! `filters_data/indic/golden.jsonl` is committed; `PYROID_INDIC_ORACLE=a.jsonl:b.jsonl`
//! replays more recordings (corpus, large fuzz runs, other CPython versions).
//! Every row is replayed under the Unicode version it was recorded with.

#![allow(clippy::print_stderr)] // reports how many rows were replayed

use std::borrow::Cow;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use pyroid_core::filters::indic::IndicNlp;
use pyroid_core::filters::{FilterError, WordTokenizer};
use pyroid_core::ftfy::UnicodeVersion;
use rayon::prelude::*;
use serde_json::Value;

const MAX_EXAMPLES: usize = 12;
const MAX_SHOWN: usize = 300;
const FIELDS: [&str; 4] = ["w", "z", "e", "s"];

fn show(text: &str) -> String {
    let shown: String = text.chars().take(MAX_SHOWN).collect();
    if shown.len() == text.len() {
        format!("{shown:?}")
    } else {
        format!("{shown:?} (truncated; {} chars)", text.chars().count())
    }
}

/// What a recorded field says: a value, a Python exception, or nothing.
enum Want<'r> {
    Value(&'r Value),
    Raise(&'r str),
    Absent,
}

fn want<'r>(row: &'r Value, key: &str) -> Want<'r> {
    if let Some(v) = row.get(key) {
        Want::Value(v)
    } else if let Some(e) = row.get(format!("{key}!")).and_then(Value::as_str) {
        Want::Raise(e)
    } else {
        Want::Absent
    }
}

/// The FilterError a recorded Python exception corresponds to.
fn expected_error(exc: &str) -> Result<FilterError, String> {
    if exc == "ValueError: substring not found"
        || exc == "ValueError: chr() arg not in range(0x110000)"
    {
        Ok(FilterError::SubstringNotFound)
    } else {
        Err(format!("datatrove raised {exc}; not replayable"))
    }
}

/// Compare one field; `Ok(true)` if it matches.
fn check_field<T: PartialEq>(
    got: Result<T, FilterError>,
    want: &Want<'_>,
    parse: impl Fn(&Value) -> Option<T>,
) -> Result<bool, String> {
    match want {
        Want::Absent => Ok(true),
        Want::Value(v) => {
            let v = parse(v).ok_or("malformed recorded value")?;
            Ok(got.is_ok_and(|g| g == v))
        }
        Want::Raise(exc) => {
            let e = expected_error(exc)?;
            Ok(got.err() == Some(e))
        }
    }
}

fn strings(v: &Value) -> Option<Vec<String>> {
    v.as_array()?
        .iter()
        .map(|s| s.as_str().map(str::to_owned))
        .collect()
}

fn usizes(v: &Value) -> Option<Vec<usize>> {
    v.as_array()?
        .iter()
        .map(|n| n.as_u64().and_then(|n| usize::try_from(n).ok()))
        .collect()
}

/// Everything pyroid says about one text.
struct Got {
    words: Vec<String>,
    owned_words: bool,
    trait_words_ok: bool,
    sents: Result<Vec<String>, FilterError>,
    /// Sentence ends in code points, as Python reports them.
    ends: Result<Vec<usize>, FilterError>,
    count: Result<usize, FilterError>,
}

fn run(tok: &IndicNlp, text: &str) -> Got {
    let mut words = Vec::new();
    tok.word_tokenize(text, &mut words);
    let owned_words = words.iter().any(|w| matches!(w, Cow::Owned(_)));
    let mut slices = Vec::new();
    let trait_words = tok.words(text, &mut slices);
    let trait_words_ok = if owned_words {
        trait_words == Err(FilterError::SubstringNotFound) && slices.is_empty()
    } else {
        trait_words.is_ok() && slices.iter().copied().eq(words.iter().map(|w| &**w))
    };
    let mut sents = Vec::new();
    let sents = tok
        .sent_tokenize(text, &mut sents)
        .map(|()| sents.into_iter().map(Cow::into_owned).collect());
    let mut ends = Vec::new();
    let ends = tok.sentence_ends(text, &mut ends).map(|()| {
        ends.iter()
            .map(|&b| text[..b].chars().count())
            .collect::<Vec<_>>()
    });
    Got {
        words: words.into_iter().map(Cow::into_owned).collect(),
        owned_words,
        trait_words_ok,
        sents,
        ends,
        count: tok.count_sentences(text),
    }
}

/// Per-row result: which fields differ (in FIELDS order), whether a word was
/// owned, and whether the trait view was consistent.
struct RowResult {
    mismatched: [bool; 4],
    owned_words: bool,
    trait_ok: bool,
}

fn replay(line: &str) -> Result<RowResult, String> {
    let row: Value = serde_json::from_str(line).map_err(|e| format!("bad JSON: {e}"))?;
    let text = row["t"].as_str().ok_or("no \"t\"")?;
    let code = row["c"].as_str().ok_or("no \"c\"")?;
    let unidata = row["u"].as_str().ok_or("no \"u\"")?;
    let version = UnicodeVersion::from_unidata_version(unidata)
        .ok_or_else(|| format!("unsupported Unicode version {unidata}"))?;
    let tok = IndicNlp::get(code, version).ok_or_else(|| format!("no tokenizer for {code}"))?;
    let got = std::panic::catch_unwind(|| run(tok, text))
        .map_err(|_| format!("pyroid panicked on {}", show(text)))?;
    let ok = [
        check_field(Ok(got.words), &want(&row, "w"), strings)?,
        check_field(got.sents, &want(&row, "z"), strings)?,
        check_field(got.ends, &want(&row, "e"), usizes)?,
        check_field(got.count, &want(&row, "s"), |v| {
            v.as_u64().and_then(|n| usize::try_from(n).ok())
        })?,
    ];
    Ok(RowResult {
        mismatched: ok.map(|b| !b),
        owned_words: got.owned_words,
        trait_ok: got.trait_words_ok,
    })
}

fn describe(location: &str, line: &str) -> String {
    let row: Value = serde_json::from_str(line).unwrap();
    let text = row["t"].as_str().unwrap();
    let version = UnicodeVersion::from_unidata_version(row["u"].as_str().unwrap()).unwrap();
    let tok = IndicNlp::get(row["c"].as_str().unwrap(), version).unwrap();
    let got = run(tok, text);
    let mut msg = format!(
        "{location} [{} {}]\n  text: {}\n",
        row["c"],
        version.as_str(),
        show(text)
    );
    let fields: [(&str, String); 4] = [
        ("w", format!("{:?}", got.words)),
        ("z", format!("{:?}", got.sents)),
        ("e", format!("{:?}", got.ends)),
        ("s", format!("{:?}", got.count)),
    ];
    for (key, mine) in fields {
        let theirs = match want(&row, key) {
            Want::Value(v) => v.to_string(),
            Want::Raise(e) => format!("raise {e}"),
            Want::Absent => continue,
        };
        writeln!(msg, "  {key}: datatrove {theirs}\n     pyroid    {mine}").unwrap();
    }
    msg
}

#[derive(Default)]
struct Tally {
    rows: usize,
    mismatches: [usize; 4],
    owned_word_rows: usize,
    trait_inconsistent: usize,
    errors: Vec<String>,
    examples: Vec<String>,
}

fn replay_file(path: &Path, tally: &mut Tally) {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let lines: Vec<&str> = data.lines().collect();
    let results: Vec<_> = lines.par_iter().map(|line| replay(line)).collect();
    let mut file_mismatches = [0; 4];
    for (n, (line, result)) in lines.iter().zip(results).enumerate() {
        let location = format!("{}:{}", path.display(), n + 1);
        match result {
            Err(e) => tally.errors.push(format!("{location}: {e}")),
            Ok(r) => {
                tally.rows += 1;
                tally.owned_word_rows += usize::from(r.owned_words);
                tally.trait_inconsistent += usize::from(!r.trait_ok);
                for (n, bad) in file_mismatches.iter_mut().zip(r.mismatched) {
                    *n += usize::from(bad);
                }
                if (r.mismatched.contains(&true) || !r.trait_ok)
                    && tally.examples.len() < MAX_EXAMPLES
                {
                    tally.examples.push(describe(&location, line));
                }
            }
        }
    }
    for (total, n) in tally.mismatches.iter_mut().zip(file_mismatches) {
        *total += n;
    }
    eprintln!(
        "{}: {} rows; mismatches w/z/e/s = {:?}",
        path.display(),
        lines.len(),
        file_mismatches
    );
}

fn check(files: &[PathBuf]) {
    let mut tally = Tally::default();
    for path in files {
        replay_file(path, &mut tally);
    }
    let failed =
        tally.mismatches.iter().sum::<usize>() + tally.trait_inconsistent + tally.errors.len() > 0;
    let summary = format!(
        "{} rows; mismatches {}; {} rows with merged (owned) words; {} trait-view \
         inconsistencies; {} unreplayable rows",
        tally.rows,
        FIELDS
            .iter()
            .zip(tally.mismatches)
            .map(|(f, n)| format!("{f}={n}"))
            .collect::<Vec<_>>()
            .join(" "),
        tally.owned_word_rows,
        tally.trait_inconsistent,
        tally.errors.len()
    );
    if failed {
        let mut msg = String::new();
        for example in &tally.examples {
            writeln!(msg, "{example}").unwrap();
        }
        for error in tally.errors.iter().take(MAX_EXAMPLES) {
            writeln!(msg, "error: {error}").unwrap();
        }
        panic!("pyroid differs from datatrove:\n{msg}{summary}");
    }
    eprintln!("{summary}");
}

#[test]
fn golden_recording_matches_datatrove() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/filters_data/indic/golden.jsonl");
    check(&[path]);
}

#[test]
fn extra_recordings_match_datatrove() {
    let Ok(list) = std::env::var("PYROID_INDIC_ORACLE") else {
        eprintln!("PYROID_INDIC_ORACLE not set; skipping");
        return;
    };
    let files: Vec<PathBuf> = list
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    check(&files);
}
