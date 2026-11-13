// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace, and spaCy 3.8.18
// (MIT), © ExplosionAI GmbH; modified.
//! Differential tests: replay what datatrove 0.10.1 really returns for
//! `split_into_words(text, language)` / `split_into_sentences(text, language)`
//! (spaCy 3.8.18 `spacy.blank(code)` + `sentencizer` for the spaCy languages),
//! as recorded by `tests/filters_data/oracle.py`, and require identical
//! results, raised exceptions included.
//!
//! * `filters_data/golden-tokenizer.jsonl` (committed): English fixed-seed
//!   fuzz inputs, C4-style lines and short real documents, recorded on
//!   CPython 3.13 (Unicode 15.1).
//! * `filters_data/multilingual/*.jsonl` (committed): the same per language
//!   (`"l"`: the datatrove language code each row was recorded with).
//! * `PYROID_FILTERS_ORACLE=a.jsonl:b.jsonl` adds more recordings, such as
//!   corpus and large fuzz runs generated in the dev container.
//!
//! Every row is replayed under the Unicode version it was recorded with
//! (`"u"`). Rows whose text contains `IS_ALPHA` are skipped: what datatrove
//! returns for them depends on what the process tokenized before (a spaCy
//! symbol-table bug; `oracle.py isalpha` characterises it).

#![allow(clippy::print_stderr)] // reports how many rows were replayed

use std::fmt::Write as _;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use pyroid_core::filters::{FilterError, WordTokenizer, tokenizer_for};
use pyroid_core::ftfy::UnicodeVersion;
use rayon::prelude::*;
use serde_json::Value;

/// Mismatches shown in full before the test fails.
const MAX_EXAMPLES: usize = 20;
/// Characters of a text shown in a failure message.
const MAX_SHOWN: usize = 300;

/// One recorded row.
struct Row {
    text: String,
    /// The datatrove language code (`"l"`; `"eng"`, the default, when absent).
    language: String,
    version: UnicodeVersion,
    expect: Expect,
}

/// What datatrove returned.
enum Expect {
    Ok {
        words: Vec<String>,
        sentences: usize,
    },
    /// `split_into_words` (`at_words`) or `split_into_sentences` raised;
    /// `words` when only the sentences raised and the recorder kept them.
    Raise {
        exc: String,
        at_words: bool,
        words: Option<Vec<String>>,
    },
}

/// What replaying one row gave.
enum Outcome {
    Match,
    SkippedIsAlpha,
    Mismatch { words: bool, sentences: bool },
    Error(String),
}

/// Counts over every replayed file.
#[derive(Default)]
struct Tally {
    rows: usize,
    skipped: usize,
    word_mismatches: usize,
    sentence_mismatches: usize,
    errors: Vec<String>,
    examples: Vec<String>,
}

/// `text` Debug-escaped, cut to `MAX_SHOWN` characters.
fn show(text: &str) -> String {
    let shown: String = text.chars().take(MAX_SHOWN).collect();
    if shown.len() == text.len() {
        format!("{shown:?}")
    } else {
        format!("{shown:?} (truncated; {} chars)", text.chars().count())
    }
}

fn words_of(v: &Value) -> Result<Vec<String>, String> {
    v.as_array()
        .ok_or("\"w\" is not a list")?
        .iter()
        .map(|w| {
            w.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "a word is not a string".to_owned())
        })
        .collect()
}

/// Exceptions that say the recording environment lacked a package datatrove
/// imports for that language (indic-nlp's Urdu sentences need urduhack): not
/// datatrove's behaviour, so not compared.
const MISSING_BACKEND: &[&str] = &["ModuleNotFoundError", "ImportError"];

/// Parse one recorded line; `Ok(None)` for an `IS_ALPHA` row.
fn parse(line: &str) -> Result<Option<Row>, String> {
    let row: Value = serde_json::from_str(line).map_err(|e| format!("bad JSON: {e}"))?;
    let text = row.get("t").and_then(Value::as_str).ok_or("no \"t\"")?;
    if text.contains("IS_ALPHA") {
        return Ok(None);
    }
    let language = row.get("l").and_then(Value::as_str).unwrap_or("eng");
    let unidata = row.get("u").and_then(Value::as_str).ok_or("no \"u\"")?;
    let version = UnicodeVersion::from_unidata_version(unidata)
        .ok_or_else(|| format!("unsupported Unicode version {unidata}"))?;
    let expect = if let Some(exc) = row.get("raise") {
        let exc = exc.as_str().ok_or("\"raise\" is not a string")?;
        let at = row.get("at").and_then(Value::as_str).ok_or("no \"at\"")?;
        Expect::Raise {
            exc: exc.to_owned(),
            at_words: at == "w",
            words: row.get("w").map(words_of).transpose()?,
        }
    } else {
        let words = words_of(row.get("w").ok_or("no \"w\"")?)?;
        let sentences = row
            .get("s")
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .ok_or("no \"s\"")?;
        Expect::Ok { words, sentences }
    };
    Ok(Some(Row {
        text: text.to_owned(),
        language: language.to_owned(),
        version,
        expect,
    }))
}

/// The Python exception datatrove raises where pyroid returns `e`.
fn py_exception(e: FilterError) -> &'static str {
    match e {
        FilterError::ZeroDivision => "ZeroDivisionError",
        FilterError::TextTooLong(_)
        | FilterError::SubstringNotFound
        | FilterError::NeedsTokenizer
        | FilterError::Tokenizer(_)
        | FilterError::ThaiTextTooLong(_) => "ValueError",
    }
}

/// What pyroid gives for a row: words (or the exception's name) and the
/// sentence count (or the exception's name).
type Got = (Result<Vec<String>, String>, Result<usize, String>);

fn run(row: &Row) -> Result<Got, String> {
    let tok: &dyn WordTokenizer = tokenizer_for(&row.language, row.version)
        .ok_or_else(|| format!("pyroid has no tokenizer for {:?}", row.language))?;
    let mut words = Vec::new();
    let words = tok
        .words_cow(&row.text, &mut words)
        .map(|()| words.iter().map(ToString::to_string).collect())
        .map_err(|e| py_exception(e).to_owned());
    let sentences = tok
        .count_sentences(&row.text)
        .map_err(|e| py_exception(e).to_owned());
    Ok((words, sentences))
}

/// (words ok, sentences ok) for `got` against the recording.
fn compare(row: &Row, got: &Got) -> (bool, bool) {
    match &row.expect {
        Expect::Ok { words, sentences } => (
            got.0.as_ref().is_ok_and(|w| w == words),
            got.1.as_ref().is_ok_and(|s| s == sentences),
        ),
        Expect::Raise {
            exc,
            at_words: true,
            ..
        } => (got.0.as_ref().err() == Some(exc), true),
        Expect::Raise {
            exc,
            at_words: false,
            words,
        } => {
            let words_ok = match words {
                Some(w) => got.0.as_ref().is_ok_and(|g| g == w),
                None => got.0.is_ok(),
            };
            let sentences_ok =
                MISSING_BACKEND.contains(&exc.as_str()) || got.1.as_ref().err() == Some(exc);
            (words_ok, sentences_ok)
        }
    }
}

/// A row with nothing to compare: the recorder's environment lacked a
/// backend package and kept no words.
fn uncomparable(row: &Row) -> bool {
    matches!(&row.expect, Expect::Raise { exc, words: None, .. } if MISSING_BACKEND.contains(&exc.as_str()))
}

fn outcome(line: &str) -> Outcome {
    let row = match parse(line) {
        Ok(Some(row)) => row,
        Ok(None) => return Outcome::SkippedIsAlpha,
        Err(e) => return Outcome::Error(e),
    };
    if uncomparable(&row) {
        return Outcome::SkippedIsAlpha;
    }
    let replayed = panic::catch_unwind(AssertUnwindSafe(|| run(&row)));
    match replayed {
        Ok(Ok(got)) => match compare(&row, &got) {
            (true, true) => Outcome::Match,
            (words_ok, sentences_ok) => Outcome::Mismatch {
                words: !words_ok,
                sentences: !sentences_ok,
            },
        },
        Ok(Err(e)) => Outcome::Error(e),
        Err(_) => Outcome::Error(format!(
            "pyroid panicked on {} ({})",
            show(&row.text),
            row.language
        )),
    }
}

/// Where the word lists first differ, with a window of both around it.
fn word_diff(want: &[&str], got: &[&str]) -> String {
    let first = want
        .iter()
        .zip(got)
        .position(|(w, g)| w != g)
        .unwrap_or_else(|| want.len().min(got.len()));
    let lo = first.saturating_sub(3);
    let window = |words: &[&str]| -> Vec<String> {
        words[lo.min(words.len())..(first + 5).min(words.len())]
            .iter()
            .map(|w| (*w).to_owned())
            .collect()
    };
    format!(
        "first difference at word {first} (datatrove has {} words, pyroid {}):\n    \
         datatrove[{lo}..]: {:?}\n    pyroid   [{lo}..]: {:?}",
        want.len(),
        got.len(),
        window(want),
        window(got),
    )
}

/// The full story of one mismatching row (recomputed, so only for the shown ones).
fn describe(location: &str, line: &str) -> String {
    let Ok(Some(row)) = parse(line) else {
        return format!("{location}: unparsable row");
    };
    let mut msg = format!(
        "{location} [{}, Unicode {}]\n  text: {}\n",
        row.language,
        row.version.as_str(),
        show(&row.text)
    );
    let Ok((got_words, got_sentences)) = run(&row) else {
        return msg + "  (no tokenizer)";
    };
    match &row.expect {
        Expect::Ok { words, sentences } => {
            let want: Vec<&str> = words.iter().map(String::as_str).collect();
            match &got_words {
                Ok(got) => {
                    let got: Vec<&str> = got.iter().map(String::as_str).collect();
                    if got != want {
                        writeln!(msg, "  words: {}", word_diff(&want, &got)).unwrap();
                    }
                }
                Err(e) => writeln!(msg, "  words: pyroid raised {e}").unwrap(),
            }
            write!(
                msg,
                "  sentences: datatrove {sentences}, pyroid {got_sentences:?}"
            )
            .unwrap();
        }
        Expect::Raise { exc, at_words, .. } => {
            let at = if *at_words { "words" } else { "sentences" };
            write!(
                msg,
                "  datatrove raised {exc} in {at}; pyroid words {:?}, sentences {got_sentences:?}",
                got_words.as_ref().map(Vec::len)
            )
            .unwrap();
        }
    }
    msg
}

fn replay_file(path: &Path, tally: &mut Tally) {
    let data = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let lines: Vec<&str> = data.lines().collect();
    let outcomes: Vec<Outcome> = lines.par_iter().map(|line| outcome(line)).collect();
    let (mut replayed, mut skipped) = (0, 0);
    for (n, (line, result)) in lines.iter().zip(outcomes).enumerate() {
        let location = || format!("{}:{}", path.display(), n + 1);
        match result {
            Outcome::Match => replayed += 1,
            Outcome::SkippedIsAlpha => skipped += 1,
            Outcome::Error(e) => tally.errors.push(format!("{}: {e}", location())),
            Outcome::Mismatch { words, sentences } => {
                replayed += 1;
                tally.word_mismatches += usize::from(words);
                tally.sentence_mismatches += usize::from(sentences);
                if tally.examples.len() < MAX_EXAMPLES {
                    tally.examples.push(describe(&location(), line));
                }
            }
        }
    }
    tally.rows += replayed;
    tally.skipped += skipped;
    eprintln!(
        "{}: {replayed} rows replayed, {skipped} rows skipped (IS_ALPHA, or recorded without a backend package)",
        path.display()
    );
}

/// Replay every file and fail with examples and totals on any difference.
fn check(files: &[PathBuf]) {
    let mut tally = Tally::default();
    for path in files {
        replay_file(path, &mut tally);
    }
    let failed = tally.word_mismatches + tally.sentence_mismatches + tally.errors.len() > 0;
    if failed {
        let mut msg = String::new();
        for example in &tally.examples {
            writeln!(msg, "{example}\n").unwrap();
        }
        for error in tally.errors.iter().take(MAX_EXAMPLES) {
            writeln!(msg, "error: {error}").unwrap();
        }
        write!(
            msg,
            "{} rows, {} word mismatches, {} sentence mismatches, {} unreplayable rows \
             ({} rows skipped (IS_ALPHA, or recorded without a backend package))",
            tally.rows,
            tally.word_mismatches,
            tally.sentence_mismatches,
            tally.errors.len(),
            tally.skipped
        )
        .unwrap();
        panic!("pyroid differs from datatrove:\n{msg}");
    }
    eprintln!(
        "{} rows match datatrove ({} rows skipped (IS_ALPHA, or recorded without a backend package))",
        tally.rows, tally.skipped
    );
}

#[test]
fn golden_recording_matches_datatrove() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/filters_data/golden-tokenizer.jsonl");
    check(&[path]);
}

#[test]
fn extra_recordings_match_datatrove() {
    let Ok(list) = std::env::var("PYROID_FILTERS_ORACLE") else {
        eprintln!("PYROID_FILTERS_ORACLE not set; skipping");
        return;
    };
    let files: Vec<PathBuf> = list
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    check(&files);
}

/// The committed per-language recordings (`filters_data/multilingual/`, one
/// file per datatrove tokenizer), each row replayed with its own language.
#[test]
fn multilingual_recordings_match_datatrove() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/filters_data/multilingual");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot list {}: {e}", dir.display()))
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "jsonl"))
        .collect();
    files.sort();
    assert!(!files.is_empty(), "no recordings in {}", dir.display());
    check(&files);
}
