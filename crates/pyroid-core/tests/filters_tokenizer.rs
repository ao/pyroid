// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace, and spaCy 3.8.18
// (MIT), © ExplosionAI GmbH; modified.
//! Differential tests: replay what datatrove 0.10.1 really returns for English
//! `split_into_words` / `split_into_sentences` (spaCy 3.8.18
//! `spacy.blank("en")` + `sentencizer`), as recorded by
//! `tests/filters_data/oracle.py`, and require identical results.
//!
//! * `filters_data/golden-tokenizer.jsonl` (committed): fixed-seed fuzz
//!   inputs, C4-style lines and short real documents, recorded on CPython
//!   3.13 (Unicode 15.1).
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

use pyroid_core::filters::tokenizer::SpacyEnglish;
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
    version: UnicodeVersion,
    words: Vec<String>,
    sentences: usize,
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

/// Parse one recorded line; `Ok(None)` for an `IS_ALPHA` row.
fn parse(line: &str) -> Result<Option<Row>, String> {
    let row: Value = serde_json::from_str(line).map_err(|e| format!("bad JSON: {e}"))?;
    let text = row.get("t").and_then(Value::as_str).ok_or("no \"t\"")?;
    if text.contains("IS_ALPHA") {
        return Ok(None);
    }
    if let Some(exc) = row.get("raise") {
        let at = row.get("at").unwrap_or(&Value::Null);
        return Err(format!(
            "datatrove raised {exc} (in {at}) on {}; not replayable",
            show(text)
        ));
    }
    let unidata = row.get("u").and_then(Value::as_str).ok_or("no \"u\"")?;
    let version = UnicodeVersion::from_unidata_version(unidata)
        .ok_or_else(|| format!("unsupported Unicode version {unidata}"))?;
    let words = row
        .get("w")
        .and_then(Value::as_array)
        .ok_or("no \"w\"")?
        .iter()
        .map(|w| {
            w.as_str()
                .map(str::to_owned)
                .ok_or("a word is not a string")
        })
        .collect::<Result<Vec<_>, _>>()?;
    let sentences = row
        .get("s")
        .and_then(Value::as_u64)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or("no \"s\"")?;
    Ok(Some(Row {
        text: text.to_owned(),
        version,
        words,
        sentences,
    }))
}

fn outcome(line: &str) -> Outcome {
    let row = match parse(line) {
        Ok(Some(row)) => row,
        Ok(None) => return Outcome::SkippedIsAlpha,
        Err(e) => return Outcome::Error(e),
    };
    let replayed = panic::catch_unwind(AssertUnwindSafe(|| {
        let tok = SpacyEnglish::get(row.version);
        (
            tok.words(&row.text) == row.words,
            tok.count_sentences(&row.text) == row.sentences,
        )
    }));
    match replayed {
        Ok((true, true)) => Outcome::Match,
        Ok((words_ok, sentences_ok)) => Outcome::Mismatch {
            words: !words_ok,
            sentences: !sentences_ok,
        },
        Err(_) => Outcome::Error(format!("pyroid panicked on {}", show(&row.text))),
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
    let tok = SpacyEnglish::get(row.version);
    let got = tok.words(&row.text);
    let got_sentences = tok.count_sentences(&row.text);
    let want: Vec<&str> = row.words.iter().map(String::as_str).collect();
    let mut msg = format!(
        "{location} [Unicode {}]\n  text: {}\n",
        row.version.as_str(),
        show(&row.text)
    );
    if got != want {
        writeln!(msg, "  words: {}", word_diff(&want, &got)).unwrap();
    }
    let mark = if got_sentences == row.sentences {
        ""
    } else {
        "  <- differs"
    };
    write!(
        msg,
        "  sentences: datatrove {}, pyroid {got_sentences}{mark}",
        row.sentences
    )
    .unwrap();
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
        "{}: {replayed} rows replayed, {skipped} IS_ALPHA rows skipped",
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
             ({} IS_ALPHA rows skipped)",
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
        "{} rows match datatrove ({} IS_ALPHA rows skipped)",
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
