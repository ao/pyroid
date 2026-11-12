//! The spaCy-English tokenizer port (`filters::tokenizer::SpacyTokenizer`, English) and
//! the dedup word tokenizers built on it.
//!
//! Input: `[version] text...` (invalid UTF-8 goes through `from_utf8_lossy`).
//! Checked:
//! - tokens are non-empty, in order, non-overlapping, in bounds and on char
//!   boundaries, and cover the text the way spaCy's `Doc` does: the first
//!   starts at 0 and the gap after each token is "" or one `' '` (spaCy's
//!   `whitespace_`), so `doc.text` round-trips;
//! - whitespace tokens hold only whitespace; other tokens neither start nor
//!   end with it (datatrove strips tokens and drops empty ones, which the
//!   port implements by dropping the whitespace tokens);
//! - the same tokens on a second call (the per-thread span cache);
//! - `words`, `filters::WordTokenizer::words` and `dedup`'s `word_tokenize`
//!   equal the non-whitespace token texts;
//! - the sentence functions agree: `sentence_ends_in` is strictly increasing,
//!   ends at token ends and at the last token, its length is
//!   `count_sentences_in`, and `count_sentences` (both versions) is that,
//!   at least 1; `WordTokenizer::sentence_ends` gives the same ends;
//! - `WhitespaceTokenizer` equals `str.split()`.

#![no_main]

use std::io::Write as _;
use std::sync::{Mutex, OnceLock};

use libfuzzer_sys::fuzz_target;
use pyroid_core::dedup::{self, WhitespaceTokenizer, WordTokenizer as _};
use pyroid_core::filters::WordTokenizer as _;
use pyroid_core::filters::tokenizer::{SpacyTokenizer, Token};
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::text::is_py_whitespace;

const VERSIONS: [UnicodeVersion; 5] = [
    UnicodeVersion::V17_0,
    UnicodeVersion::V16_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V15_0,
    UnicodeVersion::V14_0,
];

fn check_tokens(text: &str, toks: &[Token]) {
    if text.is_empty() {
        assert!(toks.is_empty(), "tokens for an empty text");
        return;
    }
    assert!(!toks.is_empty(), "no tokens for a non-empty text");
    assert_eq!(toks[0].start, 0, "first token does not start at 0");
    let mut prev_end = 0usize;
    for (i, t) in toks.iter().enumerate() {
        let (s, e) = (t.start as usize, t.end as usize);
        assert!(s < e, "empty token {i}: {t:?}");
        assert!(e <= text.len(), "token {i} out of bounds");
        assert!(
            text.is_char_boundary(s) && text.is_char_boundary(e),
            "token {i} not on char boundaries"
        );
        if i > 0 {
            let gap = &text[prev_end..s];
            assert!(gap.is_empty() || gap == " ", "gap {gap:?} before token {i}");
        }
        let tt = t.text(text);
        if t.space {
            assert!(tt.chars().all(is_py_whitespace), "space token {tt:?}");
        } else {
            assert!(
                !tt.starts_with(is_py_whitespace) && !tt.ends_with(is_py_whitespace),
                "token {tt:?} has outer whitespace"
            );
        }
        prev_end = e;
    }
    let tail = &text[prev_end..];
    assert!(
        tail.is_empty() || tail == " ",
        "tail {tail:?} after the last token"
    );
}

fn check_sentences(sp: &SpacyTokenizer, text: &str, toks: &[Token]) {
    let mut ends = Vec::new();
    sp.sentence_ends_in(text, toks, &mut ends);
    assert_eq!(
        ends.len(),
        sp.count_sentences_in(text, toks),
        "count_sentences_in"
    );
    assert_eq!(
        sp.count_sentences(text),
        ends.len().max(1),
        "count_sentences"
    );
    assert_eq!(
        pyroid_core::filters::WordTokenizer::count_sentences(sp, text),
        Ok(ends.len().max(1)),
        "WordTokenizer::count_sentences"
    );
    let mut via_trait = Vec::new();
    sp.sentence_ends(text, &mut via_trait).expect("short text");
    assert_eq!(via_trait, ends, "WordTokenizer::sentence_ends");
    match toks.last() {
        None => assert!(ends.is_empty(), "sentence ends without tokens"),
        Some(last) => assert_eq!(ends.last().copied(), Some(last.end as usize), "last end"),
    }
    assert!(ends.windows(2).all(|w| w[0] < w[1]), "ends not increasing");
    for e in &ends {
        assert!(
            toks.iter().any(|t| t.end as usize == *e),
            "sentence end {e} is not a token end"
        );
    }
}

/// `PYROID_FUZZ_DUMP=<file>`: every input's Unicode 15.1 tokens, words and
/// sentence ends (in code points), one JSON object per line, for a
/// differential run of a corpus against spaCy through datatrove
/// (`cargo fuzz run tokenizer <corpus> -- -runs=0`).
fn dump(text: &str) {
    static FILE: OnceLock<Option<Mutex<std::fs::File>>> = OnceLock::new();
    let Some(f) = FILE
        .get_or_init(|| {
            let path = std::env::var_os("PYROID_FUZZ_DUMP")?;
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .expect("open PYROID_FUZZ_DUMP");
            Some(Mutex::new(f))
        })
        .as_ref()
    else {
        return;
    };
    let hex = |b: &[u8]| -> String { b.iter().map(|x| format!("{x:02x}")).collect() };
    let list = |v: &[&str]| -> String {
        let items: Vec<String> = v
            .iter()
            .map(|s| format!("\"{}\"", hex(s.as_bytes())))
            .collect();
        format!("[{}]", items.join(","))
    };
    let sp = SpacyTokenizer::english(UnicodeVersion::V15_1);
    let mut toks = Vec::new();
    sp.tokens(text, &mut toks);
    let texts: Vec<&str> = toks.iter().map(|t| t.text(text)).collect();
    let mut ends = Vec::new();
    sp.sentence_ends_in(text, &toks, &mut ends);
    let ends: Vec<String> = ends
        .iter()
        .map(|&e| text[..e].chars().count().to_string())
        .collect();
    let line = format!(
        "{{\"in\":\"{}\",\"tokens\":{},\"words\":{},\"ends\":[{}],\"sentences\":{}}}\n",
        hex(text.as_bytes()),
        list(&texts),
        list(&sp.words(text)),
        ends.join(","),
        sp.count_sentences(text),
    );
    let mut f = f.lock().expect("dump lock");
    f.write_all(line.as_bytes()).expect("write dump");
}

fuzz_target!(|data: &[u8]| {
    let Some((&v, body)) = data.split_first() else {
        return;
    };
    let version = VERSIONS[usize::from(v) % VERSIONS.len()];
    let text = String::from_utf8_lossy(body);
    let text: &str = &text;
    let sp = SpacyTokenizer::english(version);

    let mut toks = Vec::new();
    sp.tokens(text, &mut toks);
    check_tokens(text, &toks);

    let mut again = vec![Token {
        start: 7,
        end: 9,
        space: true,
    }];
    sp.tokens(text, &mut again);
    assert_eq!(
        &again[1..],
        &toks[..],
        "tokens differ on a second call (appended)"
    );

    let expected: Vec<&str> = toks
        .iter()
        .filter(|t| !t.space)
        .map(|t| t.text(text))
        .collect();
    assert_eq!(sp.words(text), expected, "words");
    let mut words = vec!["sentinel"];
    pyroid_core::filters::WordTokenizer::words(sp, text, &mut words).expect("short text");
    assert_eq!(&words[1..], &expected[..], "WordTokenizer::words");
    let mut dwords = Vec::new();
    sp.word_tokenize(text, &mut dwords).expect("short text");
    assert_eq!(dwords, expected, "dedup word_tokenize");

    check_sentences(sp, text, &toks);

    let mut ws = Vec::new();
    dedup::WordTokenizer::word_tokenize(&WhitespaceTokenizer, text, &mut ws).expect("infallible");
    let split: Vec<&str> = text
        .split(is_py_whitespace)
        .filter(|w| !w.is_empty())
        .collect();
    assert_eq!(ws, split, "WhitespaceTokenizer");
    dump(text);
});
