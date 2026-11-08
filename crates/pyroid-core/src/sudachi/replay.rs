//! Replays of SudachiPy and datatrove recordings (`testdata/`).
//!
//! The tests need a SudachiDict system dictionary: `PYROID_SUDACHI_DICT` is
//! the path of `sudachidict_core`'s `system.dic`. Without it they print why
//! and pass. `PYROID_SUDACHI_REPLAY` adds one more JSONL file of morphemes to
//! replay (an uncommitted recording).
#![allow(clippy::print_stderr)] // reports what was replayed

use std::sync::OnceLock;

use serde_json::Value;

use super::{Dictionary, Morpheme};

fn dictionary() -> Option<&'static Dictionary> {
    static D: OnceLock<Option<Dictionary>> = OnceLock::new();
    D.get_or_init(|| {
        let path = std::env::var_os("PYROID_SUDACHI_DICT")?;
        Some(Dictionary::open(path).expect("PYROID_SUDACHI_DICT is a system.dic"))
    })
    .as_ref()
}

/// Byte offset -> character offset in `text`.
fn char_offset(text: &str, byte: usize) -> usize {
    text[..byte].chars().count()
}

/// Replay one file of `{"t", "m": [[begin, end, pos_id, word_id]]}` or
/// `{"t", "x"}` rows; the number of rows and failures.
fn replay_morphemes(dic: &Dictionary, path: &str) -> (usize, Vec<String>) {
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut rows = 0;
    let mut fails = Vec::new();
    let mut ms: Vec<Morpheme> = Vec::new();
    for line in data.lines() {
        let row: Value = serde_json::from_str(line).expect("JSON");
        let Some(text) = row.get("t").and_then(Value::as_str) else {
            continue; // the header
        };
        rows += 1;
        ms.clear();
        let got = dic.tokenize(text, &mut ms);
        match (row.get("m"), row.get("x")) {
            (Some(want), _) => {
                if let Err(e) = got {
                    fails.push(format!("{text:?}: raised {e}, want {want}"));
                    continue;
                }
                let want: Vec<(usize, usize, u16, u32)> = want
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|cols| {
                        let cols = cols.as_array().unwrap();
                        let n = |i: usize| cols[i].as_u64().unwrap();
                        #[allow(clippy::cast_possible_truncation)]
                        (n(0) as usize, n(1) as usize, n(2) as u16, n(3) as u32)
                    })
                    .collect();
                let have: Vec<(usize, usize, u16, u32)> = ms
                    .iter()
                    .map(|m| {
                        (
                            char_offset(text, m.begin),
                            char_offset(text, m.end),
                            m.pos_id,
                            m.word_id,
                        )
                    })
                    .collect();
                if have != want {
                    let i = have
                        .iter()
                        .zip(&want)
                        .position(|(a, b)| a != b)
                        .unwrap_or(have.len().min(want.len()));
                    fails.push(format!(
                        "{text:?}: first difference at morpheme {i}: have {:?}, want {:?} ({} vs {} morphemes)",
                        have.get(i),
                        want.get(i),
                        have.len(),
                        want.len()
                    ));
                }
            }
            (None, Some(x)) => match got {
                Ok(()) => fails.push(format!("{text:?}: no error, want {x}")),
                Err(e) => {
                    let x = x.as_str().unwrap();
                    if !x.contains(&e.to_string().replace("Tokenization error: ", "")) {
                        fails.push(format!("{text:?}: raised {e}, want {x}"));
                    }
                }
            },
            _ => panic!("row without m or x"),
        }
    }
    (rows, fails)
}

#[test]
fn replay_extra_recording() {
    let (Some(dic), Some(path)) = (dictionary(), std::env::var("PYROID_SUDACHI_REPLAY").ok())
    else {
        eprintln!("skipped: set PYROID_SUDACHI_DICT and PYROID_SUDACHI_REPLAY");
        return;
    };
    let (rows, fails) = replay_morphemes(dic, &path);
    for f in fails.iter().take(20) {
        eprintln!("{f}");
    }
    assert!(
        fails.is_empty(),
        "{path}: {} of {rows} rows differ",
        fails.len()
    );
    eprintln!("{path}: {rows} rows match SudachiPy");
}
