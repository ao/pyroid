// Derived from trafilatura 2.3.1 and 1.11.0 (Apache-2.0), © Adrien Barbaresi; modified.
//! Hand-written pages for Trafilatura's rare paths (exceptions lxml raises on assignments, the
//! JSON-LD and Discourse rescues, forums, comments, tables, readability), against trafilatura
//! 2.3.1's and 1.11.0's `extract` and `bare_extraction` on CPython 3.13
//! (`trafilatura_data/edge.json` and `edge-1.11.0.json`, written by `edge_cases.py`).

use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::trafilatura::{Focus, Options, PyError, Version, bare_extraction, extract};
use serde_json::{Value, json};

fn options(set: &str, version: Version) -> Options {
    let d = Options {
        version,
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

/// What the fixture records for an exception: its type for 2.3.1, "Type: message" for 1.11.0
/// (the literals of these pages are plain ASCII, so their repr is the quoted string).
fn raises(e: &PyError, version: Version) -> Value {
    let msg = match e {
        PyError::IntMaxStrDigits { digits } => format!(
            "ValueError: Exceeds the limit (4300 digits) for integer string conversion: value has \
             {digits} digits; use sys.set_int_max_str_digits() to increase the limit"
        ),
        PyError::IntLiteral { literal } => {
            format!("ValueError: invalid literal for int() with base 10: '{literal}'")
        }
        PyError::Overflow { message } => format!("OverflowError: {message}"),
        PyError::Memory => "MemoryError: ".to_owned(),
        PyError::NoneAttribute { attr } => {
            format!("AttributeError: 'NoneType' object has no attribute '{attr}'")
        }
    };
    match version {
        Version::V2_3_1 => json!({"raises": msg.split(':').next().unwrap_or("")}),
        _ => json!({ "raises": msg }),
    }
}

fn check(fixture: &str, version: Version) -> usize {
    let fixture: Value = serde_json::from_str(fixture).unwrap();
    let mut checked = 0;
    for case in fixture.as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let html = unhex(case["html"].as_str().unwrap());
        for (set, want) in case["extract"].as_object().unwrap() {
            let got = match extract(&html, &options(set, version), None) {
                Ok(t) => json!(t),
                Err(e) => raises(&e, version),
            };
            assert_eq!(&got, want, "extract: {name}, {set}");
            checked += 1;
        }
        for (set, want) in case["bare"].as_object().unwrap() {
            let got = match bare_extraction(&html, &options(set, version), None) {
                Ok(None) => Value::Null,
                Ok(Some(x)) => json!({
                    "text": x.text, "raw_text": x.raw_text, "comments": x.comments, "body": x.body,
                    // "" stands for None (1.11.0 without comments)
                    "commentsbody": if x.commentsbody.is_empty() { Value::Null } else { json!(x.commentsbody) },
                }),
                Err(e) => raises(&e, version),
            };
            assert_eq!(&got, want, "bare_extraction: {name}, {set}");
            checked += 1;
        }
    }
    checked
}

#[test]
fn edge_cases_match_trafilatura() {
    assert!(check(include_str!("trafilatura_data/edge.json"), Version::V2_3_1) > 100);
}

#[test]
fn edge_cases_match_trafilatura_1_11() {
    assert!(
        check(
            include_str!("trafilatura_data/edge-1.11.0.json"),
            Version::V1_11_0
        ) > 100
    );
}
