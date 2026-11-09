// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! Hand-written pages for Trafilatura's rare paths (exceptions lxml raises on assignments, the
//! JSON-LD and Discourse rescues, forums, comments, tables, readability), against trafilatura
//! 2.3.1's `extract` and `bare_extraction` on CPython 3.13 (`trafilatura_data/edge.json`,
//! written by `edge_cases.py`).

use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::trafilatura::{Focus, Options, bare_extraction, extract};
use serde_json::{Value, json};

fn options(set: &str) -> Options {
    let d = Options {
        unicode: UnicodeVersion::V15_1,
        ..Options::default()
    };
    match set {
        "default" => d,
        "precision" => Options {
            focus: Focus::Precision,
            comments: false,
            ..d
        },
        "recall" => Options {
            focus: Focus::Recall,
            ..d
        },
        "precision_comments" => Options {
            focus: Focus::Precision,
            ..d
        },
        "fast" => Options { fast: true, ..d },
        "notables" => Options { tables: false, ..d },
        _ => panic!("option set {set}"),
    }
}

fn unhex(h: &str) -> Vec<u8> {
    (0..h.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
        .collect()
}

#[test]
fn edge_cases_match_trafilatura() {
    let fixture: Value = serde_json::from_str(include_str!("trafilatura_data/edge.json")).unwrap();
    let mut checked = 0;
    for case in fixture.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let html = unhex(case["html"].as_str().unwrap());
        for (set, want) in case["extract"].as_object().unwrap() {
            let got = match extract(&html, &options(set), None) {
                Ok(t) => json!(t),
                Err(_) => json!({"raises": "ValueError"}),
            };
            assert_eq!(&got, want, "extract: {name}, {set}");
            checked += 1;
        }
        for (set, want) in case["bare"].as_object().unwrap() {
            let got = match bare_extraction(&html, &options(set), None) {
                Ok(None) => Value::Null,
                Ok(Some(x)) => json!({
                    "text": x.text, "raw_text": x.raw_text, "comments": x.comments,
                    "body": x.body, "commentsbody": x.commentsbody,
                }),
                Err(_) => json!({"raises": "ValueError"}),
            };
            assert_eq!(&got, want, "bare_extraction: {name}, {set}");
            checked += 1;
        }
    }
    assert!(checked > 100);
}
