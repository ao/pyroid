//! Replays of the SudachiPy 0.7.0 and datatrove 0.10.1 recordings in
//! `testdata/` (written by `testdata/oracle.py`).
//!
//! The morpheme and datatrove replays need the dictionary the recordings were
//! made with, sudachidict-core 20261015: `PYROID_SUDACHI_DICT` is the path of
//! its `system.dic` (the header's sha256 is checked). Without it those tests
//! print why and pass; the chunking replay always runs.
//! `PYROID_SUDACHI_GOLDEN_DIR` replays the files of another recording (an
//! uncommitted `oracle.py --full` or `large` run) instead of `testdata/`.
#![allow(clippy::print_stderr)] // reports what was replayed

use std::path::PathBuf;
use std::sync::OnceLock;

use serde_json::Value;

use super::ja::{self, JaTokenizer, chunk_text_on_bytes};
use super::{Dictionary, Morpheme};
use crate::filters::{FilterError, WordTokenizer};
use crate::ftfy::UnicodeVersion;

const DICT_SHA256: &str = "e815659f6ad6f6118725103d47d317d660276ff3a15f668f8fc123ec3c4e6646";

fn golden(name: &str) -> PathBuf {
    std::env::var_os("PYROID_SUDACHI_GOLDEN_DIR")
        .map_or_else(
            || PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/src/sudachi/testdata")),
            PathBuf::from,
        )
        .join(name)
}

fn dictionary() -> Option<&'static Dictionary> {
    static D: OnceLock<Option<&'static Dictionary>> = OnceLock::new();
    *D.get_or_init(|| {
        let path = std::env::var_os("PYROID_SUDACHI_DICT")?;
        Some(ja::install(path).expect("PYROID_SUDACHI_DICT is a system.dic"))
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

/// A recording's rows, read one line at a time (the large ones are
/// hundreds of MB).
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
fn rows(name: &str) -> Option<(Value, Rows)> {
    use std::io::BufRead;
    let path = golden(name);
    let file = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(e) if std::env::var_os("PYROID_SUDACHI_GOLDEN_DIR").is_some() => {
            eprintln!("{}: {e}; skipped", path.display());
            return None;
        }
        Err(e) => panic!("{}: {e}", path.display()),
    };
    let mut rows = Rows {
        lines: std::io::BufReader::new(file).lines(),
        n: 0,
    };
    let header = rows.next().expect("a header")["header"].clone();
    rows.n = 0;
    Some((header, rows))
}

fn check_dictionary(name: &str, header: &Value) {
    assert_eq!(
        header["system_dic_sha256"].as_str(),
        Some(DICT_SHA256),
        "{name} was recorded with another dictionary"
    );
}

/// Byte offset -> character offset in `text`.
fn char_offset(text: &str, byte: usize) -> usize {
    text[..byte].chars().count()
}

fn n_chars(s: &str) -> u64 {
    s.chars().count() as u64
}

fn short(t: &str) -> String {
    let s: String = t.chars().take(60).collect();
    if s.len() < t.len() {
        format!("{s:?}...")
    } else {
        format!("{s:?}")
    }
}

/// Whether `e` is the error a row recorded as `x`.
fn same_error(e: impl std::fmt::Display, x: &str) -> bool {
    let msg = e.to_string();
    x.ends_with(msg.trim_start_matches("Error during tokenization: "))
}

fn report(name: &str, n_rows: usize, n_items: usize, fails: &[String]) {
    for f in fails.iter().take(10) {
        eprintln!("{name}: {f}");
    }
    assert!(
        fails.is_empty(),
        "{name}: {} of {n_rows} rows differ",
        fails.len()
    );
    eprintln!("{name}: {n_rows} rows ({n_items} items) match");
}

fn replay_morphemes(dic: &Dictionary, name: &str) {
    let Some((header, mut rows)) = rows(name) else {
        return;
    };
    check_dictionary(name, &header);
    let mut fails = Vec::new();
    let mut ms: Vec<Morpheme> = Vec::new();
    let mut n_morphemes = 0usize;
    for row in rows.by_ref() {
        let row = &row;
        let t = text(&row["t"]);
        ms.clear();
        let got = dic.tokenize(&t, &mut ms);
        if let Some(x) = row.get("x") {
            let x = x.as_str().unwrap();
            match got {
                Ok(()) => fails.push(format!("{}: no error, want {x}", short(&t))),
                Err(e) if !same_error(e, x) => {
                    fails.push(format!("{}: raised {e}, want {x}", short(&t)));
                }
                Err(_) => {}
            }
            continue;
        }
        if let Err(e) = got {
            fails.push(format!("{}: raised {e}", short(&t)));
            continue;
        }
        let want: Vec<(usize, usize, u64, u64)> = row["m"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                let n = |i: usize| m[i].as_u64().unwrap();
                let at = |i: usize| usize::try_from(n(i)).unwrap();
                (at(0), at(1), n(2), n(3))
            })
            .collect();
        let have: Vec<(usize, usize, u64, u64)> = ms
            .iter()
            .map(|m| {
                (
                    char_offset(&t, m.begin),
                    char_offset(&t, m.end),
                    u64::from(m.pos_id),
                    u64::from(m.word_id),
                )
            })
            .collect();
        n_morphemes += want.len();
        if have != want {
            let i = have
                .iter()
                .zip(&want)
                .position(|(a, b)| a != b)
                .unwrap_or(have.len().min(want.len()));
            fails.push(format!(
                "{}: morpheme {i}: have {:?}, want {:?} ({} vs {} morphemes)",
                short(&t),
                have.get(i),
                want.get(i),
                have.len(),
                want.len()
            ));
        }
    }
    report(name, rows.n, n_morphemes, &fails);
}

fn strings(v: &Value) -> Vec<String> {
    v.as_array().unwrap().iter().map(text).collect()
}

fn chunk_lengths(t: &str, n: usize) -> Vec<u64> {
    chunk_text_on_bytes(t, n)
        .into_iter()
        .map(|r| n_chars(&t[r]))
        .collect()
}

fn numbers(v: &Value) -> Vec<u64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect()
}

fn replay_datatrove(name: &str) {
    let Some((header, mut rows)) = rows(name) else {
        return;
    };
    check_dictionary(name, &header);
    let version = UnicodeVersion::from_unidata_version(header["unicode"].as_str().unwrap())
        .expect("a supported Unicode version");
    let tok = JaTokenizer::get(version).expect("a dictionary is installed");
    let mut fails = Vec::new();
    let mut n_items = 0usize;
    for row in rows.by_ref() {
        let row = &row;
        let t = text(&row["t"]);
        if let Some(c) = row.get("c") {
            let (have, want) = (chunk_lengths(&t, 40_000), numbers(c));
            if have != want {
                fails.push(format!("{}: chunks {have:?}, want {want:?}", short(&t)));
                continue;
            }
        }
        let mut words = Vec::new();
        let words_r = tok.words(&t, &mut words);
        let mut spans = Vec::new();
        let spans_r = tok.spans(&t, &mut spans);
        let mut ends = Vec::new();
        let ends_r = tok.sentence_ends(&t, &mut ends);
        let mut sents = Vec::new();
        let sents_r = tok.sentences(&t, &mut sents);
        if let Some(x) = row.get("x") {
            let x = x.as_str().unwrap();
            for r in [&words_r, &spans_r, &ends_r, &sents_r] {
                match r {
                    Err(FilterError::Tokenizer(e)) if same_error(e, x) => {}
                    other => fails.push(format!("{}: {other:?}, want {x}", short(&t))),
                }
            }
            continue;
        }
        if let Some(e) = [words_r, spans_r, ends_r, sents_r]
            .into_iter()
            .find_map(Result::err)
        {
            fails.push(format!("{}: raised {e}", short(&t)));
            continue;
        }
        let want_w = strings(&row["w"]);
        let want_s = strings(&row["s"]);
        let want_sp: Vec<(usize, usize)> = row["sp"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let at = |i: usize| usize::try_from(p[i].as_u64().unwrap()).unwrap();
                (at(0), at(1))
            })
            .collect();
        // `split_into_sentences` uses the span ends: here as bytes of `t`.
        let want_ends: Vec<usize> = want_sp
            .iter()
            .map(|&(_, e)| t.char_indices().nth(e).map_or(t.len(), |(b, _)| b))
            .collect();
        n_items += want_w.len() + want_sp.len();
        if words != want_w {
            let i = words
                .iter()
                .zip(&want_w)
                .position(|(a, b)| a != b)
                .unwrap_or(words.len().min(want_w.len()));
            fails.push(format!(
                "{}: word {i}: have {:?}, want {:?}",
                short(&t),
                words.get(i),
                want_w.get(i)
            ));
        } else if spans != want_sp {
            fails.push(format!("{}: spans {spans:?}, want {want_sp:?}", short(&t)));
        } else if ends != want_ends {
            fails.push(format!(
                "{}: sentence ends {ends:?}, want {want_ends:?}",
                short(&t)
            ));
        } else if sents != want_s {
            fails.push(format!(
                "{}: sentences {sents:?}, want {want_s:?}",
                short(&t)
            ));
        }
    }
    report(name, rows.n, n_items, &fails);
}

macro_rules! morpheme_replays {
    ($($test:ident => $file:literal,)*) => {$(
        #[test]
        fn $test() {
            match dictionary() {
                Some(d) => replay_morphemes(d, $file),
                None => eprintln!("{}: skipped, set PYROID_SUDACHI_DICT", $file),
            }
        }
    )*};
}

macro_rules! datatrove_replays {
    ($($test:ident => $file:literal,)*) => {$(
        #[test]
        fn $test() {
            match dictionary() {
                Some(_) => replay_datatrove($file),
                None => eprintln!("{}: skipped, set PYROID_SUDACHI_DICT", $file),
            }
        }
    )*};
}

morpheme_replays! {
    morphemes_cases => "morphemes-cases.jsonl",
    morphemes_fineweb => "morphemes-fineweb.jsonl",
    morphemes_fuzz => "morphemes-fuzz.jsonl",
    morphemes_sweep => "morphemes-sweep.jsonl",
}

datatrove_replays! {
    datatrove_cases => "datatrove-cases.jsonl",
    datatrove_fineweb => "datatrove-fineweb.jsonl",
    datatrove_fuzz => "datatrove-fuzz.jsonl",
    datatrove_sweep => "datatrove-sweep.jsonl",
    datatrove_long => "datatrove-long.jsonl",
}

#[test]
fn chunking() {
    let Some((_, mut rows)) = rows("chunks.jsonl") else {
        return;
    };
    let (mut fails, mut raised) = (Vec::new(), 0);
    for row in rows.by_ref() {
        let row = &row;
        let t = text(&row["t"]);
        let n = usize::try_from(row["n"].as_u64().unwrap()).unwrap();
        if row.get("x").is_some() {
            // Python raises when a chunk size below 10,000 makes the shrink
            // step a float; datatrove uses 40,000, where it is an integer.
            assert_ne!(n, 40_000, "{}: {}", short(&t), row["x"]);
            raised += 1;
            continue;
        }
        let (have, want) = (chunk_lengths(&t, n), numbers(&row["c"]));
        if have != want {
            fails.push(format!("{} (n={n}): {have:?}, want {want:?}", short(&t)));
        }
    }
    eprintln!("chunks.jsonl: {raised} rows record datatrove's TypeError (not replayed)");
    report("chunks.jsonl", rows.n - raised, rows.n - raised, &fails);
}

#[test]
fn every_pos_has_a_spacy_tag() {
    // datatrove's `resolve_pos` raises KeyError for a tag outside spaCy's
    // TAG_MAP; with no such POS, the tags never change the words.
    let path = golden("tags.json");
    let Ok(data) = std::fs::read_to_string(&path) else {
        eprintln!("{}: missing; skipped", path.display());
        return;
    };
    let tags: Value = serde_json::from_str(&data).unwrap();
    assert_eq!(tags["missing"].as_array().map(Vec::len), Some(0));
    let Some(dic) = dictionary() else {
        eprintln!("tags.json: POS check skipped, set PYROID_SUDACHI_DICT");
        return;
    };
    check_dictionary("tags.json", &tags["header"]);
    let pos = tags["pos"].as_array().unwrap();
    assert_eq!(pos.len(), dic.pos_count());
    for p in pos {
        let id = u16::try_from(p[0].as_u64().unwrap()).unwrap();
        let want: Vec<&str> = p[1]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let have: Vec<&str> = dic.pos(id).unwrap().iter().map(|s| &**s).collect();
        assert_eq!(have, want, "POS {id}");
    }
}

/// `PYROID_SUDACHI_REPLAY`: comma-separated paths of further recordings
/// (`oracle.py large` or `--full` output), replayed by their header's level.
#[test]
fn extra_recordings() {
    let Ok(paths) = std::env::var("PYROID_SUDACHI_REPLAY") else {
        return;
    };
    let Some(dic) = dictionary() else {
        eprintln!("PYROID_SUDACHI_REPLAY: skipped, set PYROID_SUDACHI_DICT");
        return;
    };
    for path in paths.split(',').filter(|p| !p.is_empty()) {
        let first = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{path}: {e}"))
            .lines()
            .next()
            .map(|l| serde_json::from_str::<Value>(l).expect("JSON"))
            .expect("a header");
        match first["header"]["level"].as_str() {
            Some("morphemes") => replay_morphemes(dic, path),
            Some("datatrove") => replay_datatrove(path),
            other => panic!("{path}: level {other:?}"),
        }
    }
}
