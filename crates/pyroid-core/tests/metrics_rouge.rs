// Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors, and NLTK 3.10.4
// (Apache-2.0), © NLTK Project; modified.
//! Differential tests: replay what rouge-score 0.1.2 and nltk 3.10.4 really
//! return, as recorded by `tests/metrics_data/rouge_oracle.py` in
//! `metrics_data/rouge_golden.jsonl`, and require identical results: Porter
//! stems, `DefaultTokenizer` tokens, and `RougeScorer` scores for rouge1..9,
//! rougeL and rougeLsum (float bits, and rouge-score's int zeros).

use std::path::Path;

use pyroid_core::metrics::rouge::{RougeType, Score, Scorer, porter, tokenize};
use serde_json::Value;

fn rows() -> Vec<Value> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/metrics_data/rouge_golden.jsonl");
    let text = std::fs::read_to_string(&path).expect("golden file");
    text.lines()
        .map(|l| serde_json::from_str(l).expect("valid JSON row"))
        .collect()
}

fn str_of(v: &Value) -> &str {
    v.as_str().expect("a string")
}

fn float(hex: &Value) -> f64 {
    f64::from_bits(u64::from_str_radix(str_of(hex), 16).expect("hex bits"))
}

fn expected(v: &Value) -> Score {
    if v.as_str() == Some("int0") {
        return Score::IntZero;
    }
    let f = v.as_array().expect("three floats");
    Score::Float {
        precision: float(&f[0]),
        recall: float(&f[1]),
        fmeasure: float(&f[2]),
    }
}

/// Floats compared by bits (`==` would accept -0.0 for 0.0).
fn same(a: Score, b: Score) -> bool {
    match (a, b) {
        (Score::IntZero, Score::IntZero) => true,
        (Score::Float { .. }, Score::Float { .. }) => {
            let (x, y) = (a.values(), b.values());
            x.0.to_bits() == y.0.to_bits()
                && x.1.to_bits() == y.1.to_bits()
                && x.2.to_bits() == y.2.to_bits()
        }
        _ => false,
    }
}

#[test]
fn porter_matches_nltk() {
    let mut n = 0;
    for row in rows().iter().filter(|r| r.get("stem").is_some()) {
        let word = str_of(&row["stem"]);
        let got = porter::stem(word.as_bytes());
        assert_eq!(
            std::str::from_utf8(&got).unwrap(),
            str_of(&row["out"]),
            "{word}"
        );
        n += 1;
    }
    assert!(n > 1_000, "{n} words");
}

#[test]
fn tokenizer_matches_rouge_score() {
    let mut n = 0;
    for row in rows().iter().filter(|r| r.get("tokens").is_some()) {
        let text = str_of(&row["text"]);
        let stem = row["stemmer"].as_bool().unwrap();
        let want: Vec<&str> = row["tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(str_of)
            .collect();
        assert_eq!(
            tokenize::tokenize(text, stem, true),
            want,
            "{text:?} stem={stem}"
        );
        n += 1;
    }
    assert!(n > 20, "{n} texts");
}

#[test]
fn scores_match_rouge_score() {
    let mut n = 0;
    for row in rows().iter().filter(|r| r.get("scores").is_some()) {
        let types: Vec<RougeType> = row["types"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| RougeType::parse(str_of(t)).unwrap())
            .collect();
        let scorer = Scorer::new(types, row["stemmer"].as_bool().unwrap());
        let (target, prediction) = (str_of(&row["target"]), str_of(&row["prediction"]));
        let got = scorer.score(target, prediction);
        let want: Vec<Score> = row["scores"]
            .as_array()
            .unwrap()
            .iter()
            .map(expected)
            .collect();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                same(*g, *w),
                "type {i}: got {g:?}, want {w:?}\ntarget {target:?}\nprediction {prediction:?}"
            );
        }
        n += 1;
    }
    assert!(n > 100, "{n} pairs");
}
