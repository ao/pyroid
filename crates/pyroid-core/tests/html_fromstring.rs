//! `lxml::fromstring_wtf8` and `lxml::fromstring_bytes` against `lxml.html.fromstring` with
//! Trafilatura's parser on a str and on its surrogatepass bytes: the hand-written cases and
//! small fuzzed documents in `html_data/fromstring.jsonl`, written by
//! `html_data/lxml_sets.py fixture` with lxml 6.1.3. Each line holds the input (`text`, or
//! `hex` when it has lone surrogates) and per input kind the tree in the `.tree` line format
//! or the exception class and message.

use pyroid_core::html::lxml::{LxmlError, fromstring_bytes, fromstring_str, fromstring_wtf8};
use pyroid_core::html::tree::Document;

fn input(v: &serde_json::Value) -> Vec<u8> {
    if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
        return t.as_bytes().to_vec();
    }
    let h = v["hex"].as_str().expect("hex");
    (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).expect("hex digit"))
        .collect()
}

fn result(r: Result<Document, LxmlError>) -> serde_json::Value {
    match r {
        Ok(d) => {
            let mut s = String::new();
            d.dump(d.root, true, &mut s);
            serde_json::json!({ "tree": s })
        }
        Err(LxmlError::ParserError(m)) => serde_json::json!({"error": "ParserError", "message": m}),
        Err(LxmlError::ValueError(m)) => serde_json::json!({"error": "ValueError", "message": m}),
        Err(LxmlError::XmlSyntaxError(m)) => {
            serde_json::json!({"error": "XMLSyntaxError", "message": m})
        }
        Err(e @ (LxmlError::InvalidInput(_) | LxmlError::InputTooLarge)) => {
            serde_json::json!({"error": "other", "message": e.to_string()})
        }
    }
}

#[test]
fn fromstring_matches_lxml() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/html_data/fromstring.jsonl"
    );
    let text = std::fs::read_to_string(path).expect("fixture");
    let mut failures = Vec::new();
    let mut n = 0;
    for (i, line) in text.lines().enumerate() {
        let rec: serde_json::Value = serde_json::from_str(line).expect("json line");
        let s = input(&rec["input"]);
        let on_str = result(fromstring_wtf8(&s));
        if let Ok(valid) = std::str::from_utf8(&s) {
            assert_eq!(result(fromstring_str(valid)), on_str, "line {}", i + 1);
        }
        for (key, got) in [("str", on_str), ("bytes", result(fromstring_bytes(&s)))] {
            n += 1;
            if rec[key] != got {
                failures.push(format!(
                    "line {} ({key}): lxml {} ours {got}",
                    i + 1,
                    rec[key]
                ));
            }
        }
    }
    assert!(n > 300, "{n} results");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
