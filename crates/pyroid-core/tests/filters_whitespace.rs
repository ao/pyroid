// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! Differential tests: replay what datatrove 0.10.1's `WhitespaceTokenizer`
//! sentence splitting really returns, as recorded by
//! `tests/filters_data/whitespace/record.py`, and require identical results
//! from `pyroid_core::filters::whitespace`:
//!
//! * `"e"`: the ends of `WhitespaceTokenizer().span_tokenize(text)` vs
//!   `sentence_ends`,
//! * `"s"`: `len(split_into_sentences(text, "kat_Geor"))` vs
//!   `Whitespace::count_sentences`,
//! * `"z"`: `sent_tokenize(text)` vs `sent_tokenize`.
//!
//! `filters_data/whitespace/golden.jsonl` is committed;
//! `PYROID_WHITESPACE_ORACLE=a.jsonl:b.jsonl` replays more recordings.

#![allow(clippy::print_stderr)] // reports how many rows were replayed

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use pyroid_core::filters::whitespace::{self, Whitespace};
use pyroid_core::filters::{FilterError, WordTokenizer};
use pyroid_core::ftfy::UnicodeVersion;
use rayon::prelude::*;
use serde_json::Value;

const MAX_EXAMPLES: usize = 12;
const MAX_SHOWN: usize = 300;

fn show(text: &str) -> String {
    let shown: String = text.chars().take(MAX_SHOWN).collect();
    if shown.len() == text.len() {
        format!("{shown:?}")
    } else {
        format!("{shown:?} (truncated; {} chars)", text.chars().count())
    }
}

/// A stand-in for spaCy `xx`: only the sentences are under test here.
#[derive(Debug)]
struct NoWords;

impl WordTokenizer for NoWords {
    fn unicode_version(&self) -> UnicodeVersion {
        UnicodeVersion::LATEST
    }

    fn words<'a>(&self, _: &'a str, _: &mut Vec<&'a str>) -> Result<(), FilterError> {
        Ok(())
    }

    fn sentence_ends(&self, _: &str, _: &mut Vec<usize>) -> Result<(), FilterError> {
        unreachable!("Whitespace has its own sentences")
    }
}

static NO_WORDS: NoWords = NoWords;

/// pyroid's `(ends in code points, count, sentences)` for `text`. The
/// recorded Unicode version does not matter: the regex has its own data.
fn run(text: &str) -> (Vec<usize>, usize, Vec<String>) {
    let tok = Whitespace::new(&NO_WORDS);
    let mut ends = Vec::new();
    whitespace::sentence_ends(text, &mut ends).unwrap();
    let ends = ends.iter().map(|&b| text[..b].chars().count()).collect();
    let count = tok.count_sentences(text).unwrap();
    let mut sents = Vec::new();
    whitespace::sent_tokenize(text, &mut sents);
    (ends, count, sents.into_iter().map(str::to_owned).collect())
}

struct Row {
    text: String,
    ends: Vec<usize>,
    count: usize,
    sents: Option<Vec<String>>,
}

fn parse(line: &str) -> Result<Row, String> {
    let row: Value = serde_json::from_str(line).map_err(|e| format!("bad JSON: {e}"))?;
    let text = row["t"].as_str().ok_or("no \"t\"")?;
    for key in ["e!", "s!", "z!"] {
        if let Some(e) = row.get(key) {
            return Err(format!("datatrove raised {e} on {}", show(text)));
        }
    }
    let unidata = row["u"].as_str().ok_or("no \"u\"")?;
    let num = |v: &Value| v.as_u64().and_then(|n| usize::try_from(n).ok());
    UnicodeVersion::from_unidata_version(unidata)
        .ok_or_else(|| format!("unsupported Unicode version {unidata}"))?;
    Ok(Row {
        text: text.to_owned(),
        ends: row["e"]
            .as_array()
            .ok_or("no \"e\"")?
            .iter()
            .map(num)
            .collect::<Option<_>>()
            .ok_or("bad \"e\"")?,
        count: num(&row["s"]).ok_or("no \"s\"")?,
        sents: row.get("z").and_then(Value::as_array).map(|z| {
            z.iter()
                .filter_map(|s| s.as_str().map(str::to_owned))
                .collect()
        }),
    })
}

/// `Ok(mismatches in e, s, z)` for one row.
fn replay(line: &str) -> Result<[bool; 3], String> {
    let row = parse(line)?;
    let (ends, count, sents) = std::panic::catch_unwind(|| run(&row.text))
        .map_err(|_| format!("pyroid panicked on {}", show(&row.text)))?;
    Ok([
        ends != row.ends,
        count != row.count,
        row.sents.is_some_and(|z| z != sents),
    ])
}

fn describe(location: &str, line: &str) -> String {
    let row = parse(line).unwrap();
    let (ends, count, sents) = run(&row.text);
    format!(
        "{location}\n  text: {}\n  e: datatrove {:?}\n     pyroid    {ends:?}\n  \
         s: datatrove {}, pyroid {count}\n  z: datatrove {:?}\n     pyroid    {sents:?}\n",
        show(&row.text),
        row.ends,
        row.count,
        row.sents
    )
}

fn check(files: &[PathBuf]) {
    let (mut rows, mut mismatches, mut errors, mut examples) = (0, [0; 3], Vec::new(), Vec::new());
    for path in files {
        let data = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
        let lines: Vec<&str> = data.lines().collect();
        let results: Vec<_> = lines.par_iter().map(|line| replay(line)).collect();
        let mut file = [0; 3];
        for (n, (line, result)) in lines.iter().zip(results).enumerate() {
            let location = format!("{}:{}", path.display(), n + 1);
            match result {
                Err(e) => errors.push(format!("{location}: {e}")),
                Ok(bad) => {
                    rows += 1;
                    for (n, b) in file.iter_mut().zip(bad) {
                        *n += usize::from(b);
                    }
                    if bad.contains(&true) && examples.len() < MAX_EXAMPLES {
                        examples.push(describe(&location, line));
                    }
                }
            }
        }
        for (total, n) in mismatches.iter_mut().zip(file) {
            *total += n;
        }
        eprintln!(
            "{}: {} rows; mismatches e/s/z = {file:?}",
            path.display(),
            lines.len()
        );
    }
    let summary = format!(
        "{rows} rows; mismatches e={} s={} z={}; {} unreplayable rows",
        mismatches[0],
        mismatches[1],
        mismatches[2],
        errors.len()
    );
    if mismatches.iter().sum::<usize>() + errors.len() > 0 {
        let mut msg = String::new();
        for example in &examples {
            writeln!(msg, "{example}").unwrap();
        }
        for error in errors.iter().take(MAX_EXAMPLES) {
            writeln!(msg, "error: {error}").unwrap();
        }
        panic!("pyroid differs from datatrove:\n{msg}{summary}");
    }
    eprintln!("{summary}");
}

#[test]
fn golden_recording_matches_datatrove() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/filters_data/whitespace/golden.jsonl");
    check(&[path]);
}

#[test]
fn extra_recordings_match_datatrove() {
    let Ok(list) = std::env::var("PYROID_WHITESPACE_ORACLE") else {
        eprintln!("PYROID_WHITESPACE_ORACLE not set; skipping");
        return;
    };
    let files: Vec<PathBuf> = list
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    check(&files);
}

#[test]
fn megabyte_without_terminals_is_fast() {
    let text = "Lorem ipsum A dolor ".repeat(55_000);
    let start = std::time::Instant::now();
    let mut ends = Vec::new();
    whitespace::sentence_ends(&text, &mut ends).unwrap();
    assert_eq!(ends, Vec::<usize>::new());
    let text = format!("{text}.");
    whitespace::sentence_ends(&text, &mut ends).unwrap();
    assert_eq!(ends, [text.len()]);
    // Linear: well under a second even unoptimized (quadratic would be hours).
    assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
}
