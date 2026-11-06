//! The datatrove quality filters and `Pipeline`, with configs from the
//! input's first bytes.
//!
//! Input: `[version] [n_steps] [kinds] [4 x 24-byte step configs] text...`.
//! All-zero header bytes select datatrove's defaults (a one-step
//! GopherQuality pipeline); every value stays inside what the Python API
//! accepts (any float, including NaN and infinities, for thresholds and
//! counts; falsy Gopher thresholds become `None`; n-gram sizes >= 1).
//!
//! Every input also runs the four default filters and C4 with FineWeb's
//! `filter_no_terminal_punct=False`. Checked:
//! - no panic; same result when run twice;
//! - `GopherRepetition`, `C4Quality` and `FineWebQuality` never fail, and
//!   `GopherQuality` fails (ZeroDivision, as datatrove does) only on a
//!   document without words;
//! - every drop reason is one the filter can give, for a check that is on;
//! - C4: a kept text is stripped, borrowed exactly when unchanged, and has no
//!   line or word the config removes (`{`, "javascript", and with
//!   `split_paragraph` "lorem ipsum", policy phrases and lines without
//!   terminal punctuation; without `remove_citations`, over-long words);
//! - `Pipeline::run` equals running its steps one by one, each on the text
//!   the previous C4 step left.
//!
//! `PYROID_FUZZ_DUMP=<file>` also writes every input's outcomes, for a
//! differential run of a corpus against datatrove (see `dump`).

#![no_main]

use std::borrow::Cow;
use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

use libfuzzer_sys::fuzz_target;
use pyroid_core::filters::{
    self, C4Outcome, C4Quality, Doc, FilterError, FineWebQuality, GopherQuality, GopherRepetition,
    Pipeline, Reason, Step, WordTokenizer,
};
use pyroid_core::ftfy::UnicodeVersion;

const VERSIONS: [UnicodeVersion; 5] = [
    UnicodeVersion::V17_0,
    UnicodeVersion::V16_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V15_0,
    UnicodeVersion::V14_0,
];
const SLOT: usize = 24;
const MAX_STEPS: usize = 4;
const HEADER: usize = 3 + MAX_STEPS * SLOT;

/// Reads a fixed slot; past its end every read is zero (the default).
struct Reader<'a> {
    b: &'a [u8],
    i: usize,
}

impl Reader<'_> {
    fn u8(&mut self) -> u8 {
        let x = self.b.get(self.i).copied().unwrap_or(0);
        self.i += 1;
        x
    }

    fn u64(&mut self) -> u64 {
        let mut w = [0u8; 8];
        for x in &mut w {
            *x = self.u8();
        }
        u64::from_le_bytes(w)
    }

    /// A float: the default for 0, else one from a palette of edge cases.
    fn float(&mut self, default: f64) -> f64 {
        match self.u8() % 16 {
            0 => default,
            1 => default * 0.5,
            2 => default * 2.0,
            3 => 0.0,
            4 => -1.0,
            5 => 1.0,
            6 => 0.5,
            7 => f64::NAN,
            8 => f64::INFINITY,
            9 => f64::NEG_INFINITY,
            10 => 1e-9,
            11 => 3.0,
            12 => 1e9,
            13 => -0.0,
            14 => f64::from(self.u8()),
            _ => f64::from_bits(self.u64()),
        }
    }

    /// A Gopher threshold: datatrove skips the check for falsy values.
    fn threshold(&mut self, default: f64) -> Option<f64> {
        let x = self.float(default);
        (x != 0.0).then_some(x)
    }

    /// `(n, fraction)` pairs; 0 entries gives the default list.
    fn ngrams(&mut self, default: &[(usize, f64)], max: usize) -> Vec<(usize, f64)> {
        let k = usize::from(self.u8()) % (max + 2);
        if k == 0 {
            return default.to_vec();
        }
        (1..k)
            .map(|_| {
                let n = match self.u8() {
                    255 => 1000,
                    b => 1 + usize::from(b) % 12,
                };
                (n, self.float(0.2))
            })
            .collect()
    }
}

/// Strings a bitmask picks from (for stop words and stop chars).
fn pick(mask: u8, from: &[&str]) -> Vec<String> {
    from.iter()
        .enumerate()
        .filter(|(i, _)| mask >> i & 1 != 0)
        .map(|(_, s)| (*s).to_owned())
        .collect()
}

fn gopher_quality(r: &mut Reader<'_>) -> GopherQuality {
    let d = GopherQuality::default();
    let mut g = GopherQuality {
        min_doc_words: r.threshold(50.0),
        max_doc_words: r.threshold(100_000.0),
        min_avg_word_length: r.threshold(3.0),
        max_avg_word_length: r.threshold(10.0),
        max_symbol_word_ratio: r.threshold(0.1),
        max_bullet_lines_ratio: r.threshold(0.9),
        max_ellipsis_lines_ratio: r.threshold(0.3),
        max_non_alpha_words_ratio: r.threshold(0.8),
        min_stop_words: r.threshold(2.0),
        stop_words: d.stop_words,
    };
    let mask = r.u8();
    if mask != 0 {
        g.stop_words = pick(
            mask,
            &["the", "of", "", "The", "\u{e9}", "a", "x y", "\u{fb01}"],
        );
    }
    g
}

fn gopher_repetition(r: &mut Reader<'_>) -> GopherRepetition {
    let d = GopherRepetition::default();
    GopherRepetition {
        dup_line_frac: r.threshold(0.3),
        dup_para_frac: r.threshold(0.3),
        dup_line_char_frac: r.threshold(0.2),
        dup_para_char_frac: r.threshold(0.2),
        top_n_grams: r.ngrams(&d.top_n_grams, 3),
        dup_n_grams: r.ngrams(&d.dup_n_grams, 4),
    }
}

fn c4_quality(r: &mut Reader<'_>) -> C4Quality {
    let bits = r.u8();
    let flip = |i: u8| bits >> i & 1 != 0;
    let d = C4Quality::default();
    C4Quality {
        split_paragraph: d.split_paragraph ^ flip(0),
        remove_citations: d.remove_citations ^ flip(1),
        filter_no_terminal_punct: d.filter_no_terminal_punct ^ flip(2),
        filter_lorem_ipsum: d.filter_lorem_ipsum ^ flip(3),
        filter_javascript: d.filter_javascript ^ flip(4),
        filter_curly_bracket: d.filter_curly_bracket ^ flip(5),
        filter_policy: d.filter_policy ^ flip(6),
        min_num_sentences: r.float(5.0),
        min_words_per_line: r.float(3.0),
        max_word_length: r.float(1000.0),
    }
}

fn fineweb_quality(r: &mut Reader<'_>) -> FineWebQuality {
    let flags = r.u8();
    let mask = r.u8();
    FineWebQuality {
        line_punct_thr: r.float(0.12),
        line_punct_exclude_zero: flags & 1 != 0,
        stop_chars: (mask != 0).then(|| {
            pick(
                mask,
                &[".", "!", "", "..", "\u{3002}", "a", "\"", "e\u{301}"],
            )
        }),
        short_line_thr: r.float(0.67),
        short_line_length: r.float(30.0),
        char_duplicates_ratio: r.float(0.01),
        new_line_ratio: r.float(0.3),
    }
}

/// `C4QualityFilter` with FineWeb's `filter_no_terminal_punct=False`.
fn c4_fineweb() -> C4Quality {
    C4Quality {
        filter_no_terminal_punct: false,
        ..C4Quality::default()
    }
}

/// ASCII-lowercased bytes, as the C4 substring checks see `line.lower()`.
fn fold(s: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        match c {
            _ if c.is_ascii() => out.push(c.to_ascii_lowercase() as u8),
            '\u{212a}' => out.push(b'k'),
            '\u{130}' => out.extend_from_slice(&[b'i', 0xFF]),
            _ => out.push(0xFF),
        }
    }
    out
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(pyroid_core::text::is_py_whitespace)
}

fn py_words(s: &str) -> impl Iterator<Item = &str> {
    s.split(pyroid_core::text::is_py_whitespace)
        .filter(|w| !w.is_empty())
}

fn check_c4(c: &C4Quality, text: &str, out: &C4Outcome<'_>) {
    match out {
        C4Outcome::Drop(r) => {
            let ok = match r.as_str().as_ref() {
                "lorem_ipsum" => c.filter_lorem_ipsum,
                "curly_bracket" => c.filter_curly_bracket,
                "too_few_sentences" => c.min_num_sentences > 0.0,
                _ => false,
            };
            assert!(ok, "C4 reason {r} with {c:?}");
        }
        C4Outcome::Keep(kept) => {
            match kept {
                Cow::Borrowed(b) => assert!(
                    b.as_ptr() == text.as_ptr() && b.len() == text.len(),
                    "C4 Borrowed is not the input"
                ),
                Cow::Owned(o) => assert!(o != text, "C4 Owned equals the input"),
            }
            assert_eq!(py_strip(kept), &**kept, "C4 text is not stripped");
            let folded = fold(kept);
            if c.filter_curly_bracket {
                assert!(!kept.contains('{'), "C4 kept a curly bracket");
            }
            if c.filter_javascript {
                assert!(!contains(&folded, b"javascript"), "C4 kept javascript");
            }
            if c.split_paragraph {
                if c.filter_lorem_ipsum {
                    assert!(!contains(&folded, b"lorem ipsum"), "C4 kept lorem ipsum");
                }
                if c.filter_policy {
                    for p in [
                        &b"terms of use"[..],
                        b"privacy policy",
                        b"cookie policy",
                        b"uses cookies",
                        b"use of cookies",
                        b"use cookies",
                    ] {
                        assert!(!contains(&folded, p), "C4 kept a policy line");
                    }
                }
                if c.filter_no_terminal_punct && !kept.is_empty() {
                    for line in kept.split('\n') {
                        assert!(
                            line.ends_with(['.', '?', '!', '"', '\'']) && !line.ends_with("..."),
                            "C4 kept a line without terminal punctuation: {line:?}"
                        );
                    }
                }
            }
            // Word lengths are measured before citations are removed, and
            // removing "[citation needed]" can join two words.
            #[allow(clippy::float_cmp)]
            if !c.remove_citations && c.max_word_length != -1.0 && !c.max_word_length.is_nan() {
                for w in py_words(kept) {
                    #[allow(clippy::cast_precision_loss)]
                    let n = w.chars().count() as f64;
                    assert!(n <= c.max_word_length, "C4 kept an over-long word {w:?}");
                }
            }
        }
    }
}

fn gopher_quality_reason_ok(g: &GopherQuality, r: &Reason) -> bool {
    match r.as_str().as_ref() {
        "gopher_short_doc" => g.min_doc_words.is_some(),
        "gopher_long_doc" => g.max_doc_words.is_some(),
        "gopher_below_avg_threshold" => g.min_avg_word_length.is_some(),
        "gopher_above_avg_threshold" => g.max_avg_word_length.is_some(),
        "gopher_too_many_hashes" | "gopher_too_many_ellipsis" => g.max_symbol_word_ratio.is_some(),
        "gopher_too_many_bullets" => g.max_bullet_lines_ratio.is_some(),
        "gopher_too_many_end_ellipsis" => g.max_ellipsis_lines_ratio.is_some(),
        "gopher_below_alpha_threshold" => g.max_non_alpha_words_ratio.is_some(),
        "gopher_enough_stop_words" => g.min_stop_words.is_some(),
        _ => false,
    }
}

fn gopher_repetition_reason_ok(g: &GopherRepetition, r: &Reason) -> bool {
    match *r {
        Reason::TopNGram(n) => g.top_n_grams.iter().any(|&(m, _)| m == n),
        Reason::DuplicatedNGrams(n) => g.dup_n_grams.iter().any(|&(m, _)| m == n),
        Reason::Fixed(s) => match s {
            "empty" => true,
            "dup_para_frac" => g.dup_para_frac.is_some(),
            "dup_para_char_frac" => g.dup_para_char_frac.is_some(),
            "dup_line_frac" => g.dup_line_frac.is_some(),
            "dup_line_char_frac" => g.dup_line_char_frac.is_some(),
            _ => false,
        },
    }
}

/// One step on its own, checked: `(drop reason, C4's new text if owned)`.
fn run_step(
    step: &Step,
    text: &str,
    tok: &'static dyn WordTokenizer,
) -> Result<(Option<Reason>, Option<String>), FilterError> {
    match step {
        Step::GopherQuality(g) => {
            let r = g.filter(&mut Doc::new(text, tok));
            match &r {
                Ok(Some(reason)) => {
                    assert!(
                        gopher_quality_reason_ok(g, reason),
                        "GopherQuality {reason}"
                    );
                }
                Ok(None) => {}
                Err(e) => {
                    assert_eq!(*e, FilterError::ZeroDivision, "GopherQuality error");
                    let mut words = Vec::new();
                    tok.words(text, &mut words).expect("short text");
                    assert!(words.is_empty(), "ZeroDivision on a document with words");
                }
            }
            Ok((r?, None))
        }
        Step::GopherRepetition(g) => {
            let r = g
                .filter(&mut Doc::new(text, tok))
                .expect("GopherRepetition never fails");
            if let Some(reason) = &r {
                assert!(
                    gopher_repetition_reason_ok(g, reason),
                    "GopherRepetition {reason}"
                );
                assert_eq!(*reason == Reason::Fixed("empty"), text.is_empty());
            }
            Ok((r, None))
        }
        Step::C4Quality(c) => {
            let out = c.filter(text, tok).expect("C4Quality never fails");
            check_c4(c, text, &out);
            Ok(match out {
                C4Outcome::Drop(r) => (Some(r), None),
                C4Outcome::Keep(Cow::Borrowed(_)) => (None, None),
                C4Outcome::Keep(Cow::Owned(s)) => (None, Some(s)),
            })
        }
        Step::FineWebQuality(f) => {
            let r = f
                .filter(&mut Doc::new(text, tok))
                .expect("FineWebQuality never fails");
            if let Some(reason) = &r {
                let ok = matches!(
                    reason.as_str().as_ref(),
                    "empty"
                        | "line_punct_ratio"
                        | "short_line_ratio"
                        | "char_dup_ratio"
                        | "list_ratio"
                );
                assert!(ok, "FineWebQuality {reason}");
                assert_eq!(
                    reason.as_str() == "empty",
                    py_strip(text).is_empty(),
                    "FineWeb empty"
                );
            }
            Ok((r, None))
        }
    }
}

/// `Pipeline::run`, step by step.
fn run_by_steps<'a>(p: &Pipeline, text: &'a str) -> Result<filters::Outcome<'a>, FilterError> {
    let mut cur: Cow<'a, str> = Cow::Borrowed(text);
    for (i, (step, tok)) in p.steps.iter().enumerate() {
        let (reason, new_text) = run_step(step, &cur, *tok)?;
        if let Some(reason) = reason {
            return Ok(filters::Outcome {
                reason: Some(reason),
                step: Some(i),
                text: cur,
            });
        }
        if let Some(s) = new_text {
            cur = Cow::Owned(s);
        }
    }
    Ok(filters::Outcome {
        reason: None,
        step: None,
        text: cur,
    })
}

fn pipeline(header: &[u8], tok: &'static dyn WordTokenizer) -> Pipeline {
    let n = 1 + usize::from(header[1]) % MAX_STEPS;
    let kinds = header[2];
    let steps = (0..n)
        .map(|i| {
            let slot = &header[3 + i * SLOT..3 + (i + 1) * SLOT];
            let mut r = Reader { b: slot, i: 0 };
            let step = match (kinds >> (2 * i)) & 3 {
                0 => Step::GopherQuality(gopher_quality(&mut r)),
                1 => Step::GopherRepetition(gopher_repetition(&mut r)),
                2 => Step::C4Quality(c4_quality(&mut r)),
                _ => Step::FineWebQuality(fineweb_quality(&mut r)),
            };
            (step, tok)
        })
        .collect();
    Pipeline { steps }
}

fn check_pipeline(p: &Pipeline, text: &str) {
    let a = p.run(text);
    let b = p.run(text);
    assert_eq!(a, b, "Pipeline::run is not deterministic");
    let by_steps = run_by_steps(p, text);
    assert_eq!(a, by_steps, "Pipeline::run differs from running its steps");
    if let Ok(o) = &a {
        assert_eq!(o.reason.is_some(), o.step.is_some(), "reason without step");
        if let Some(i) = o.step {
            assert!(i < p.steps.len(), "step index out of range");
        }
        if let Cow::Owned(s) = &o.text {
            assert!(s != text, "Pipeline Owned text equals the input");
        }
    }
}

fn default_steps(tok: &'static dyn WordTokenizer) -> Vec<(Step, &'static dyn WordTokenizer)> {
    vec![
        (Step::GopherRepetition(GopherRepetition::default()), tok),
        (Step::GopherQuality(GopherQuality::default()), tok),
        (Step::C4Quality(C4Quality::default()), tok),
        (Step::C4Quality(c4_fineweb()), tok),
        (Step::FineWebQuality(FineWebQuality::default()), tok),
    ]
}

// ------------------------------------------------------------------ dump mode

/// `PYROID_FUZZ_DUMP=<file>`: every input with the outcomes of the pipelines
/// it runs (rebuilt with Unicode 15.1, the version of the CPython 3.13 the
/// comparison runs on), one JSON object per line, for a differential run of
/// a corpus against datatrove (`cargo fuzz run filters <corpus> -- -runs=0`).
fn dump() -> Option<&'static Mutex<std::fs::File>> {
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    FILE.get_or_init(|| {
        let path = std::env::var_os("PYROID_FUZZ_DUMP")?;
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .expect("open PYROID_FUZZ_DUMP");
        Some(Mutex::new(f))
    })
    .as_ref()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A float as Python's `json` reads it.
fn fj(x: f64) -> String {
    if x.is_nan() {
        "NaN".to_owned()
    } else if x.is_infinite() {
        if x > 0.0 { "Infinity" } else { "-Infinity" }.to_owned()
    } else {
        format!("{x:?}")
    }
}

fn tj(x: Option<f64>) -> String {
    x.map_or("null".to_owned(), fj)
}

/// Strings as a JSON list of hex strings.
fn strings_json(v: &[String]) -> String {
    let items: Vec<String> = v
        .iter()
        .map(|s| format!("\"{}\"", hex(s.as_bytes())))
        .collect();
    format!("[{}]", items.join(","))
}

fn ngrams_json(v: &[(usize, f64)]) -> String {
    let items: Vec<String> = v.iter().map(|(n, f)| format!("[{n},{}]", fj(*f))).collect();
    format!("[{}]", items.join(","))
}

/// A step as `[kind, datatrove keyword arguments]` (strings hex-encoded).
fn step_json(step: &Step) -> String {
    match step {
        Step::GopherQuality(g) => format!(
            "[\"gopher_quality\",{{\"min_doc_words\":{},\"max_doc_words\":{},\
             \"min_avg_word_length\":{},\"max_avg_word_length\":{},\
             \"max_symbol_word_ratio\":{},\"max_bullet_lines_ratio\":{},\
             \"max_ellipsis_lines_ratio\":{},\"max_non_alpha_words_ratio\":{},\
             \"min_stop_words\":{},\"stop_words\":{}}}]",
            tj(g.min_doc_words),
            tj(g.max_doc_words),
            tj(g.min_avg_word_length),
            tj(g.max_avg_word_length),
            tj(g.max_symbol_word_ratio),
            tj(g.max_bullet_lines_ratio),
            tj(g.max_ellipsis_lines_ratio),
            tj(g.max_non_alpha_words_ratio),
            tj(g.min_stop_words),
            strings_json(&g.stop_words),
        ),
        Step::GopherRepetition(g) => format!(
            "[\"gopher_repetition\",{{\"dup_line_frac\":{},\"dup_para_frac\":{},\
             \"dup_line_char_frac\":{},\"dup_para_char_frac\":{},\
             \"top_n_grams\":{},\"dup_n_grams\":{}}}]",
            tj(g.dup_line_frac),
            tj(g.dup_para_frac),
            tj(g.dup_line_char_frac),
            tj(g.dup_para_char_frac),
            ngrams_json(&g.top_n_grams),
            ngrams_json(&g.dup_n_grams),
        ),
        Step::C4Quality(c) => format!(
            "[\"c4_quality\",{{\"split_paragraph\":{},\"remove_citations\":{},\
             \"filter_no_terminal_punct\":{},\"min_num_sentences\":{},\
             \"min_words_per_line\":{},\"max_word_length\":{},\
             \"filter_lorem_ipsum\":{},\"filter_javascript\":{},\
             \"filter_curly_bracket\":{},\"filter_policy\":{}}}]",
            c.split_paragraph,
            c.remove_citations,
            c.filter_no_terminal_punct,
            fj(c.min_num_sentences),
            fj(c.min_words_per_line),
            fj(c.max_word_length),
            c.filter_lorem_ipsum,
            c.filter_javascript,
            c.filter_curly_bracket,
            c.filter_policy,
        ),
        Step::FineWebQuality(f) => format!(
            "[\"fineweb_quality\",{{\"line_punct_thr\":{},\"line_punct_exclude_zero\":{},\
             \"stop_chars\":{},\"short_line_thr\":{},\"short_line_length\":{},\
             \"char_duplicates_ratio\":{},\"new_line_ratio\":{}}}]",
            fj(f.line_punct_thr),
            f.line_punct_exclude_zero,
            f.stop_chars
                .as_deref()
                .map_or("null".to_owned(), strings_json),
            fj(f.short_line_thr),
            fj(f.short_line_length),
            fj(f.char_duplicates_ratio),
            fj(f.new_line_ratio),
        ),
    }
}

fn dump_pipeline(p: &Pipeline, text: &str) -> String {
    let steps: Vec<String> = p.steps.iter().map(|(s, _)| step_json(s)).collect();
    let out = match p.run(text) {
        Ok(o) => format!(
            "{{\"reason\":{},\"step\":{},\"text\":\"{}\"}}",
            o.reason.map_or("null".to_owned(), |r| format!("\"{r}\"")),
            o.step.map_or("null".to_owned(), |i| i.to_string()),
            hex(o.text.as_bytes()),
        ),
        Err(e) => format!("{{\"error\":\"{e:?}\"}}"),
    };
    format!("{{\"steps\":[{}],\"out\":{out}}}", steps.join(","))
}

fn dump_all(header: &[u8], text: &str) {
    let Some(f) = dump() else { return };
    let tok = filters::tokenizer_for("eng", UnicodeVersion::V15_1).expect("English tokenizer");
    let defaults = default_steps(tok);
    let mut pipelines: Vec<String> = defaults
        .iter()
        .map(|s| {
            let single = Pipeline {
                steps: vec![s.clone()],
            };
            dump_pipeline(&single, text)
        })
        .collect();
    pipelines.push(dump_pipeline(&Pipeline { steps: defaults }, text));
    pipelines.push(dump_pipeline(&pipeline(header, tok), text));
    let line = format!(
        "{{\"in\":\"{}\",\"pipelines\":[{}]}}\n",
        hex(text.as_bytes()),
        pipelines.join(",")
    );
    let mut f = f.lock().expect("dump lock");
    f.write_all(line.as_bytes()).expect("write dump");
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER {
        return;
    }
    let (header, body) = data.split_at(HEADER);
    let text: Cow<'_, str> = String::from_utf8_lossy(body);
    let version = VERSIONS[usize::from(header[0]) % VERSIONS.len()];
    let tok = filters::tokenizer_for("eng", version).expect("English tokenizer");

    let defaults = Pipeline {
        steps: default_steps(tok),
    };
    // Each default filter alone (a pipeline stops at the first drop).
    for (step, tok) in &defaults.steps {
        let single = Pipeline {
            steps: vec![(step.clone(), *tok)],
        };
        check_pipeline(&single, &text);
    }
    check_pipeline(&defaults, &text);
    check_pipeline(&pipeline(header, tok), &text);
    dump_all(header, &text);
});
