//! ftfy: every public entry point, str and WTF-8, under a config taken from
//! the first five input bytes.
//!
//! Input: `[cfg0 cfg1 cfg2 cfg3 max_len] text...`. Bits of the little-endian
//! `u32` `cfg` flip ftfy's defaults (all zero = `TextFixerConfig(explain=False)`,
//! which is what `ftfy.fix_text(text)` uses); `max_len` picks
//! `max_decode_length` from the values the Python API accepts (>= 1).
//!
//! Checked, for valid UTF-8 (and for any input via `from_utf8_lossy`):
//! - no panic; `fix_and_explain` / `fix_encoding_and_explain` return the same
//!   text as `fix_text_segment` / `fix_encoding`, with an explanation exactly
//!   when ftfy gives one, and a non-empty one whenever the text changed;
//! - every `Cow` is `Borrowed` exactly when the result equals the input;
//! - `fix_text` equals ftfy's own loop (`fix_text_segment` per line or
//!   `max_decode_length` chunk, `unescape_html="auto"` switched off for good
//!   after the first segment holding `<`), re-implemented here;
//! - `fix_text_segment` and `fix_encoding` are idempotent (both iterate to a
//!   fixed point in ftfy; `fix_text` is not, see the README);
//! - `is_bad(t) == (badness(t) > 0)`.
//!
//! And for the raw bytes, through the `*_wtf8` entry points:
//! - invalid WTF-8 gives `InvalidWtf8` at the offset `validate_wtf8` reports;
//! - outputs are valid WTF-8 and agree with the str functions on UTF-8 input;
//! - the same segmentation, explanation and idempotence checks as above.
//!
//! Two environment variables turn on logging (one JSON object per line,
//! strings hex-encoded), for comparisons with ftfy itself over a corpus
//! (`cargo fuzz run ftfy <corpus> -- -runs=0`):
//! - `PYROID_FUZZ_REPORT=<file>`: the inputs on which
//!   `fix_text(fix_text(x)) != fix_text(x)`;
//! - `PYROID_FUZZ_DUMP=<file>`: every input with every result and
//!   explanation (computed with Unicode 15.1, CPython 3.13's version).

#![no_main]

use std::borrow::Cow;
use std::io::Write as _;
use std::num::NonZeroUsize;
use std::sync::{Mutex, OnceLock};

use libfuzzer_sys::fuzz_target;
use pyroid_core::ftfy::{
    self, ExplainedText, Normalization, TextFixerConfig, UnescapeHtml, UnicodeVersion,
};

const HEADER: usize = 5;

/// Newest first, so a zero field selects the default (`LATEST`).
const VERSIONS: [UnicodeVersion; 5] = [
    UnicodeVersion::V17_0,
    UnicodeVersion::V16_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V15_0,
    UnicodeVersion::V14_0,
];

fn config(h: &[u8]) -> TextFixerConfig {
    let bits = u32::from_le_bytes([h[0], h[1], h[2], h[3]]);
    let flip = |i: u32| (bits >> i) & 1 != 0;
    let d = TextFixerConfig::default();
    let max = match h[4] {
        0 => 1_000_000,
        n @ 1..=250 => usize::from(n),
        // The Python shim caps max_decode_length at 2**62.
        _ => 1 << 62,
    };
    TextFixerConfig {
        unescape_html: match bits & 3 {
            1 => UnescapeHtml::Always,
            2 => UnescapeHtml::Never,
            _ => UnescapeHtml::Auto,
        },
        remove_terminal_escapes: d.remove_terminal_escapes ^ flip(2),
        fix_encoding: d.fix_encoding ^ flip(3),
        restore_byte_a0: d.restore_byte_a0 ^ flip(4),
        replace_lossy_sequences: d.replace_lossy_sequences ^ flip(5),
        decode_inconsistent_utf8: d.decode_inconsistent_utf8 ^ flip(6),
        fix_c1_controls: d.fix_c1_controls ^ flip(7),
        fix_latin_ligatures: d.fix_latin_ligatures ^ flip(8),
        fix_character_width: d.fix_character_width ^ flip(9),
        uncurl_quotes: d.uncurl_quotes ^ flip(10),
        fix_line_breaks: d.fix_line_breaks ^ flip(11),
        fix_surrogates: d.fix_surrogates ^ flip(12),
        remove_control_chars: d.remove_control_chars ^ flip(13),
        // ftfy.fix_text's default is explain=False.
        explain: flip(14),
        normalization: match (bits >> 15) & 7 {
            1 => Some(Normalization::Nfd),
            2 => Some(Normalization::Nfkc),
            3 => Some(Normalization::Nfkd),
            4 => None,
            _ => Some(Normalization::Nfc),
        },
        unicode_version: VERSIONS[((bits >> 18) & 7) as usize % VERSIONS.len()],
        max_decode_length: NonZeroUsize::new(max).expect("nonzero"),
    }
}

/// `Borrowed` exactly when the result equals the input (and then it is the
/// whole input).
fn check_cow<T: ?Sized + PartialEq + ToOwned + AsRef<[u8]>>(
    what: &str,
    input: &T,
    out: &Cow<'_, T>,
) {
    match out {
        Cow::Borrowed(b) => {
            assert!(
                std::ptr::eq(b.as_ref().as_ptr(), input.as_ref().as_ptr())
                    && b.as_ref().len() == input.as_ref().len(),
                "{what}: Borrowed is not the whole input"
            );
        }
        Cow::Owned(_) => {
            assert!(&**out != input, "{what}: Owned result equals the input");
        }
    }
}

fn check_explained<T: AsRef<[u8]> + ?Sized>(
    what: &str,
    input: &T,
    r: &ExplainedText<impl AsRef<[u8]>>,
    plain: &[u8],
    expect_explanation: bool,
) {
    assert_eq!(
        r.text.as_ref(),
        plain,
        "{what}: text differs from the plain call"
    );
    assert_eq!(
        r.explanation.is_some(),
        expect_explanation,
        "{what}: explanation presence"
    );
    if let Some(steps) = &r.explanation {
        if plain != input.as_ref() {
            assert!(!steps.is_empty(), "{what}: changed text, empty explanation");
        }
    }
}

/// Code points in surrogate-tolerant UTF-8.
fn cp_len(b: &[u8]) -> usize {
    b.iter().filter(|&&x| x & 0xC0 != 0x80).count()
}

/// Byte offset after the first `n` code points.
fn cp_offset(b: &[u8], n: usize) -> usize {
    let mut seen = 0;
    for (i, &x) in b.iter().enumerate() {
        if x & 0xC0 != 0x80 {
            if seen == n {
                return i;
            }
            seen += 1;
        }
    }
    b.len()
}

/// ftfy's `fix_text` loop over `fix_text_segment` (as WTF-8).
fn fix_text_by_segments(text: &[u8], config: &TextFixerConfig) -> Vec<u8> {
    let mut cfg = config.clone();
    let max = cfg.max_decode_length.get();
    let mut out = Vec::with_capacity(text.len());
    let mut pos = 0;
    while pos < text.len() {
        let rest = &text[pos..];
        let mut end = rest
            .iter()
            .position(|&b| b == b'\n')
            .map_or(rest.len(), |k| k + 1);
        if cp_len(&rest[..end]) > max {
            end = cp_offset(rest, max);
        }
        let seg = &rest[..end];
        if cfg.unescape_html == UnescapeHtml::Auto && seg.contains(&b'<') {
            cfg.unescape_html = UnescapeHtml::Never;
        }
        out.extend_from_slice(&ftfy::fix_text_segment_wtf8(seg, &cfg).expect("valid segment"));
        pos += end;
    }
    out
}

/// The file named by the environment variable `var`, opened for appending.
fn log_file(var: &str) -> Option<Mutex<std::fs::File>> {
    let path = std::env::var_os(var)?;
    let f = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .expect("open log file");
    Some(Mutex::new(f))
}

fn report() -> Option<&'static Mutex<std::fs::File>> {
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    FILE.get_or_init(|| log_file("PYROID_FUZZ_REPORT")).as_ref()
}

/// `PYROID_FUZZ_DUMP=<file>`: every input with its results, one JSON object
/// per line, for a differential run against ftfy over a corpus
/// (`cargo fuzz run ftfy <corpus> -- -runs=0`).
fn dump() -> Option<&'static Mutex<std::fs::File>> {
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    FILE.get_or_init(|| log_file("PYROID_FUZZ_DUMP")).as_ref()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn write_line(f: &Mutex<std::fs::File>, line: &str) {
    let mut f = f.lock().expect("log lock");
    f.write_all(line.as_bytes()).expect("write log");
}

fn steps_json(steps: Option<&Vec<ftfy::ExplanationStep>>) -> String {
    steps.map_or("null".to_owned(), |s| {
        let items: Vec<String> = s
            .iter()
            .map(|x| format!("[\"{}\",\"{}\"]", x.action.as_str(), x.parameter))
            .collect();
        format!("[{}]", items.join(","))
    })
}

/// The config as `TextFixerConfig` keyword arguments, plus the Unicode version.
fn config_json(c: &TextFixerConfig) -> String {
    let unescape = match c.unescape_html {
        UnescapeHtml::Auto => "\"auto\"",
        UnescapeHtml::Always => "true",
        UnescapeHtml::Never => "false",
    };
    let norm = c
        .normalization
        .map_or("null".to_owned(), |n| format!("\"{}\"", n.as_str()));
    format!(
        "\"unicode\":\"{}\",\"config\":{{\"unescape_html\":{unescape},\
         \"remove_terminal_escapes\":{},\"fix_encoding\":{},\"restore_byte_a0\":{},\
         \"replace_lossy_sequences\":{},\"decode_inconsistent_utf8\":{},\
         \"fix_c1_controls\":{},\"fix_latin_ligatures\":{},\"fix_character_width\":{},\
         \"uncurl_quotes\":{},\"fix_line_breaks\":{},\"fix_surrogates\":{},\
         \"remove_control_chars\":{},\"normalization\":{norm},\"max_decode_length\":{},\
         \"explain\":{}}}",
        c.unicode_version.as_str(),
        c.remove_terminal_escapes,
        c.fix_encoding,
        c.restore_byte_a0,
        c.replace_lossy_sequences,
        c.decode_inconsistent_utf8,
        c.fix_c1_controls,
        c.fix_latin_ligatures,
        c.fix_character_width,
        c.uncurl_quotes,
        c.fix_line_breaks,
        c.fix_surrogates,
        c.remove_control_chars,
        c.max_decode_length,
        c.explain,
    )
}

/// One input and every result, for `PYROID_FUZZ_DUMP` (WTF-8, hex). Always
/// with Unicode 15.1, the version of the CPython 3.13 the comparison runs on.
fn dump_results(b: &[u8], config: &TextFixerConfig) {
    let Some(f) = dump() else { return };
    let c = &TextFixerConfig {
        unicode_version: UnicodeVersion::V15_1,
        ..config.clone()
    };
    let text = ftfy::fix_text_wtf8(b, c).expect("valid");
    let seg = ftfy::fix_and_explain_wtf8(b, c).expect("valid");
    let enc = ftfy::fix_encoding_and_explain_wtf8(b, c).expect("valid");
    let line = format!(
        "{{{},\"in\":\"{}\",\"fix_text\":\"{}\",\"fix_and_explain\":\"{}\",\
         \"explanation\":{},\"fix_encoding\":\"{}\",\"encoding_explanation\":{},\
         \"badness\":{}}}\n",
        config_json(c),
        hex(b),
        hex(&text),
        hex(&seg.text),
        steps_json(seg.explanation.as_ref()),
        hex(&enc.text),
        steps_json(enc.explanation.as_ref()),
        std::str::from_utf8(b).map_or(-1, |s| {
            i64::try_from(ftfy::badness(s, c.unicode_version)).unwrap_or(-1)
        }),
    );
    write_line(f, &line);
}

fn report_non_idempotent(text: &[u8], once: &[u8], twice: &[u8], c: &TextFixerConfig) {
    let Some(f) = report() else { return };
    let line = format!(
        "{{{},\"in\":\"{}\",\"once\":\"{}\",\"twice\":\"{}\"}}\n",
        config_json(c),
        hex(text),
        hex(once),
        hex(twice),
    );
    write_line(f, &line);
}

/// The str entry points; returns `fix_text`, `fix_text_segment`, `fix_encoding`.
fn check_str<'a>(s: &'a str, cfg: &TextFixerConfig) -> [Cow<'a, str>; 3] {
    let v = cfg.unicode_version;
    assert_eq!(
        ftfy::is_bad(s, v),
        ftfy::badness(s, v) > 0,
        "is_bad vs badness"
    );

    let text = ftfy::fix_text(s, cfg);
    check_cow("fix_text", s, &text);
    let seg = ftfy::fix_text_segment(s, cfg);
    check_cow("fix_text_segment", s, &seg);
    let enc = ftfy::fix_encoding(s, cfg);
    check_cow("fix_encoding", s, &enc);
    if !cfg.fix_encoding {
        assert!(
            matches!(enc, Cow::Borrowed(_)),
            "fix_encoding=False changed text"
        );
    }

    let fe = ftfy::fix_and_explain(s, cfg);
    check_explained("fix_and_explain", s, &fe, seg.as_bytes(), cfg.explain);
    let fee = ftfy::fix_encoding_and_explain(s, cfg);
    check_explained("fix_encoding_and_explain", s, &fee, enc.as_bytes(), true);

    assert_eq!(
        ftfy::fix_text_segment(&seg, cfg),
        seg,
        "fix_text_segment is not idempotent"
    );
    assert_eq!(
        ftfy::fix_encoding(&enc, cfg),
        enc,
        "fix_encoding is not idempotent"
    );
    [text, seg, enc]
}

/// The `*_wtf8` entry points on raw bytes; `str_results` are the str
/// functions' results when `b` is UTF-8.
fn check_wtf8(b: &[u8], cfg: &TextFixerConfig, str_results: Option<&[Cow<'_, str>; 3]>) {
    let valid = ftfy::validate_wtf8(b);
    let text = ftfy::fix_text_wtf8(b, cfg);
    let seg = ftfy::fix_text_segment_wtf8(b, cfg);
    let enc = ftfy::fix_encoding_wtf8(b, cfg);
    let fe = ftfy::fix_and_explain_wtf8(b, cfg);
    let fee = ftfy::fix_encoding_and_explain_wtf8(b, cfg);
    let (text, seg, enc, fe, fee) = match valid {
        Err(off) => {
            for (what, err) in [
                ("fix_text_wtf8", text.err()),
                ("fix_text_segment_wtf8", seg.err()),
                ("fix_encoding_wtf8", enc.err()),
                ("fix_and_explain_wtf8", fe.err()),
                ("fix_encoding_and_explain_wtf8", fee.err()),
            ] {
                assert_eq!(err.map(|e| e.0), Some(off), "{what}: error offset");
            }
            return;
        }
        Ok(()) => (
            text.expect("valid"),
            seg.expect("valid"),
            enc.expect("valid"),
            fe.expect("valid"),
            fee.expect("valid"),
        ),
    };
    for (what, out) in [
        ("fix_text_wtf8", &text),
        ("fix_text_segment_wtf8", &seg),
        ("fix_encoding_wtf8", &enc),
    ] {
        assert!(
            ftfy::validate_wtf8(out).is_ok(),
            "{what}: output is not WTF-8"
        );
        check_cow(what, b, out);
    }
    check_explained("fix_and_explain_wtf8", b, &fe, &seg, cfg.explain);
    check_cow("fix_and_explain_wtf8", b, &fe.text);
    check_explained("fix_encoding_and_explain_wtf8", b, &fee, &enc, true);
    check_cow("fix_encoding_and_explain_wtf8", b, &fee.text);

    if let Some([s_text, s_seg, s_enc]) = str_results {
        assert_eq!(&text[..], s_text.as_bytes(), "fix_text: wtf8 vs str");
        assert_eq!(&seg[..], s_seg.as_bytes(), "fix_text_segment: wtf8 vs str");
        assert_eq!(&enc[..], s_enc.as_bytes(), "fix_encoding: wtf8 vs str");
    } else {
        assert_eq!(
            ftfy::fix_text_segment_wtf8(&seg, cfg)
                .expect("valid")
                .as_ref(),
            &seg[..],
            "fix_text_segment_wtf8 is not idempotent"
        );
        assert_eq!(
            ftfy::fix_encoding_wtf8(&enc, cfg).expect("valid").as_ref(),
            &enc[..],
            "fix_encoding_wtf8 is not idempotent"
        );
    }

    assert_eq!(
        fix_text_by_segments(b, cfg),
        &text[..],
        "fix_text differs from ftfy's segment loop"
    );

    if report().is_some() {
        let twice = ftfy::fix_text_wtf8(&text, cfg).expect("valid");
        if twice != text {
            report_non_idempotent(b, &text, &twice, cfg);
        }
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < HEADER {
        return;
    }
    let cfg = config(&data[..HEADER]);
    let body = &data[HEADER..];
    match std::str::from_utf8(body) {
        Ok(s) => {
            let results = check_str(s, &cfg);
            check_wtf8(body, &cfg, Some(&results));
            dump_results(body, &cfg);
        }
        Err(_) => {
            check_wtf8(body, &cfg, None);
            if ftfy::validate_wtf8(body).is_ok() {
                dump_results(body, &cfg);
            }
            let lossy = String::from_utf8_lossy(body);
            let results = check_str(&lossy, &cfg);
            check_wtf8(lossy.as_bytes(), &cfg, Some(&results));
            dump_results(lossy.as_bytes(), &cfg);
        }
    }
});
