//! `pyroid_core::html` against lxml 6.1.3 on documents over 10 MB, which reach libxml2's
//! limits: XML_MAX_LOOKUP_LIMIT in `xmlParserGrow` (one token spanning more than 10,000,000
//! bytes of the input buffer), the text node and attribute value lengths, around the
//! thresholds and through every input path (ASCII, Latin-1, UCS-2 and UCS-4 str buffers, and
//! bytes).
//!
//! The documents are described by segments (`text * count`), so nothing large is stored:
//!
//! ```sh
//! python tests/html_data/lxml_sets.py large /target/out-html/large.jsonl
//! PYROID_HTMLX_LARGE=/target/out-html/large.jsonl \
//!     cargo test -p pyroid-core --release --test html_large -- --ignored --nocapture
//! ```
//!
//! Each line holds the label, the segments, and for `lxml.html.fromstring` on the str and on
//! its surrogatepass bytes either the sha1 and length of the tree's dump or the exception.
#![allow(clippy::print_stderr)] // reports of the comparisons

use pyroid_core::html::lxml::{LxmlError, fromstring_bytes, fromstring_wtf8};
use sha1::{Digest, Sha1};

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

fn hex(b: &[u8]) -> String {
    use std::fmt::Write as _;
    b.iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

/// Our result in the fixture's shape.
fn ours(r: Result<pyroid_core::html::tree::Document, LxmlError>) -> serde_json::Value {
    match r {
        Ok(d) => {
            let mut s = String::new();
            d.dump(d.root, true, &mut s);
            serde_json::json!({"sha1": hex(&Sha1::digest(s.as_bytes())), "len": s.len()})
        }
        Err(e) => {
            let (class, message) = match &e {
                LxmlError::ParserError(m) => ("ParserError", m.clone()),
                LxmlError::ValueError(m) => ("ValueError", m.clone()),
                LxmlError::XmlSyntaxError(m) => ("XMLSyntaxError", m.clone()),
                LxmlError::InvalidInput(_) | LxmlError::InputTooLarge => ("other", e.to_string()),
            };
            serde_json::json!({"error": class, "message": message})
        }
    }
}

/// Exception classes must match; for XMLSyntaxError lxml appends ", line L, column C" to
/// libxml2's message, which the port does not track.
fn same(want: &serde_json::Value, got: &serde_json::Value) -> bool {
    if want.get("error").is_none() || want["error"] != "XMLSyntaxError" {
        return want == got;
    }
    let w = want["message"].as_str().unwrap_or("");
    let g = got["message"].as_str().unwrap_or("?");
    got["error"] == "XMLSyntaxError" && w.starts_with(g)
}

#[test]
#[ignore = "needs the large-document specs; set PYROID_HTMLX_LARGE"]
fn large_documents_match_lxml() {
    let path = std::env::var("PYROID_HTMLX_LARGE")
        .unwrap_or_else(|_| "/target/out-html/large.jsonl".to_owned());
    let text = std::fs::read_to_string(&path).expect("large.jsonl");
    let mut bad = 0;
    let mut n = 0;
    for line in text.lines().filter(|l| !l.is_empty()) {
        let rec: serde_json::Value = serde_json::from_str(line).expect("json line");
        let mut doc = Vec::new();
        for seg in rec["segments"].as_array().expect("segments") {
            let s = input(&seg[0]);
            for _ in 0..seg[1].as_u64().expect("count") {
                doc.extend_from_slice(&s);
            }
        }
        for (key, got) in [
            ("str", ours(fromstring_wtf8(&doc))),
            ("bytes", ours(fromstring_bytes(&doc))),
        ] {
            n += 1;
            if !same(&rec[key], &got) {
                bad += 1;
                eprintln!("{} ({key}): lxml {} ours {got}", rec["label"], rec[key]);
            }
        }
    }
    eprintln!("{path}: {} of {n} results match lxml", n - bad);
    assert_eq!(bad, 0);
}
