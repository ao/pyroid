// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! Pages whose extracted text depends on the CPython version (new letters, digits and
//! unassigned characters, decomposed sequences): trafilatura 2.3.1's output on CPython
//! 3.11-3.15, from `trafilatura_data/unicode.json` (written by `unicode_cases.py`).

use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::trafilatura::{Focus, Options, extract};
use serde_json::Value;

#[test]
fn pages_per_unicode_version() {
    let fixture: Value =
        serde_json::from_str(include_str!("trafilatura_data/unicode.json")).unwrap();
    let pages = fixture["inputs"]["pages"].as_array().unwrap();
    let mut checked = 0;
    for (ver, out) in fixture["outputs"].as_object().unwrap() {
        let v = UnicodeVersion::from_unidata_version(ver).unwrap();
        for (set, want) in out["pages"].as_object().unwrap() {
            let d = Options {
                unicode: v,
                ..Options::default()
            };
            let opts = match set.as_str() {
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
                "fast" => Options { fast: true, ..d },
                _ => panic!("option set {set}"),
            };
            for (i, (page, w)) in pages.iter().zip(want.as_array().unwrap()).enumerate() {
                let got = extract(page.as_str().unwrap().as_bytes(), &opts, None).unwrap();
                assert_eq!(got.as_deref(), w.as_str(), "page {i}, {set}, Unicode {ver}");
                checked += 1;
            }
        }
    }
    assert_eq!(checked, 5 * 4 * pages.len());
}
