//! jieba 0.42.1 user dictionaries (`pyroid_core::jieba`): `Dictionary::from_entries` for a
//! custom main dictionary, `load_userdict`'s lines through `add_word` and `suggest_freq`,
//! then every segmentation entry point over a short text.
//!
//! Input: `[flags n] text dict...`: `text` is the next `n` bytes, the rest the dictionary
//! bytes (both decoded as below). Bits of `flags`: 0 `HMM=False`; 1 the dictionary bytes up
//! to the first NUL are a main dictionary (`jieba.Tokenizer(path)`, lines `word freq[ tag]`
//! with ASCII-digit frequencies, as pyroid's fast path reads them; anything else falls back
//! to the default dictionary) and the rest is the user dictionary; 2 load the user
//! dictionary twice. All zero: `jieba.load_userdict(f)`, then `jieba.cut(text)`.
//!
//! The user dictionary is read as `load_userdict` reads a binary file: lines split at `\n`,
//! `bytes.strip()`, UTF-8 (an invalid line stops the load, as jieba's ValueError does),
//! leading BOMs dropped, empty lines skipped, then `re_userdict`
//! (`^(.+?)( [0-9]+)?( [a-z]+)?$`) and `add_word(word, int(freq))`, or
//! `add_word(word, suggest_freq(word))` without a frequency; frequency 0 adds the word to
//! `Force_Split_Words`. A frequency over `i64` or a failing `suggest_freq` / `add_word` stops
//! the load (Python raises there too). The text is decoded lossily; the input is meant to stay
//! small (`-max_len=4096`): jieba keeps every prefix of every word, so a word costs memory
//! quadratic in its length, here as in jieba.
//!
//! Checked, besides "no panic":
//! - `FREQ` and `total`: every added word has its last frequency and all its prefixes are
//!   keys; `total` is the base total plus every added frequency; `len(FREQ)` is the base's
//!   plus the new keys; for a custom main dictionary, `FREQ.items()` has `len(FREQ)`
//!   distinct keys with the values `get` returns, and its words keep their last frequency;
//! - `cut` (HMM as flagged) fails exactly when `total` is 0 and the text has a word block
//!   (`log(0)` in `calc`); otherwise its words are non-empty and join to the text, and it is
//!   deterministic;
//! - `tokenize` gives `cut`'s words with contiguous character offsets from 0 to the text's
//!   length; `tokenize(mode="search")` gives `cut_for_search`'s words, each at the offsets
//!   where it occurs in the text;
//! - `get_DAG`: one non-empty, increasing list of ends in `k..len` per character `k`;
//! - `cut_all` runs and is deterministic.

#![no_main]

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use libfuzzer_sys::fuzz_target;
use pyroid_core::jieba::{CutError, Dictionary, ForceSplit};

/// `bytes.strip()`: b" \t\n\r\x0b\x0c".
fn py_strip(b: &[u8]) -> &[u8] {
    let ws = |c: &u8| matches!(c, b' ' | b'\t' | b'\n' | b'\r' | b'\x0b' | b'\x0c');
    let start = b.iter().position(|c| !ws(c)).unwrap_or(b.len());
    let end = b.iter().rposition(|c| !ws(c)).map_or(start, |i| i + 1);
    &b[start..end]
}

/// `re_han_default`: `[一-鿕a-zA-Z0-9+#&\._%\-]`.
fn is_han_default(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(c, '+' | '#' | '&' | '.' | '_' | '%' | '-')
        || ('\u{4e00}'..='\u{9fd5}').contains(&c)
}

/// A main dictionary as pyroid's fast path parses it (`gen_pfdict` with ASCII-digit
/// frequencies): the entries in file order and their total.
fn main_dict(data: &[u8]) -> Option<(Vec<(String, i64)>, i64)> {
    let body = data.strip_suffix(b"\n").unwrap_or(data);
    if data.is_empty() {
        return Some((Vec::new(), 0));
    }
    let mut entries = Vec::new();
    let mut total: i64 = 0;
    for raw in body.split(|&b| b == b'\n') {
        let line = std::str::from_utf8(py_strip(raw)).ok()?;
        let mut parts = line.split(' ');
        let word = parts.next()?;
        let freq = parts.next()?;
        if freq.is_empty() || !freq.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let freq: i64 = freq.parse().ok()?;
        total = total.checked_add(freq)?;
        entries.push((word.to_owned(), freq));
    }
    Some((entries, total))
}

/// Whether `s` matches `( [0-9]+)?( [a-z]+)?$`, with the digits if there are any.
fn rest_freq(s: &str) -> Option<Option<&str>> {
    if s.is_empty() {
        return Some(None);
    }
    let s = s.strip_prefix(' ')?;
    let digits = s.bytes().take_while(u8::is_ascii_digit).count();
    let (freq, tail) = s.split_at(digits);
    let tag = |t: &str| !t.is_empty() && t.bytes().all(|b| b.is_ascii_lowercase());
    if digits == 0 {
        return tag(s).then_some(None);
    }
    if tail.is_empty() {
        return Some(Some(freq));
    }
    tail.strip_prefix(' ')
        .filter(|t| tag(t))
        .map(|_| Some(freq))
}

/// `re_userdict.match(line).groups()[:2]`: the shortest non-empty word whose rest is
/// `( [0-9]+)?( [a-z]+)?`, and the frequency digits.
fn user_line(line: &str) -> (&str, Option<&str>) {
    for (i, _) in line.char_indices().skip(1) {
        if let Some(freq) = rest_freq(&line[i..]) {
            return (&line[..i], freq);
        }
    }
    (line, None)
}

struct Loaded {
    /// The last frequency `add_word` set per word (and the main dictionary's words).
    words: HashMap<String, i64>,
    total: i64,
}

/// `load_userdict` on `data`; stops where jieba would raise.
fn load_userdict(d: &mut Dictionary, fs: &mut ForceSplit, data: &[u8], m: &mut Loaded) {
    for raw in data.split_inclusive(|&b| b == b'\n') {
        let Ok(line) = std::str::from_utf8(py_strip(raw)) else {
            return;
        };
        let line = line.trim_start_matches('\u{feff}');
        if line.is_empty() {
            continue;
        }
        let (word, freq) = user_line(line);
        let freq = match freq {
            Some(digits) => match digits.parse::<i64>() {
                Ok(f) => f,
                Err(_) => return,
            },
            None => match d.suggest_freq(word) {
                Ok(f) => f,
                Err(_) => return,
            },
        };
        if d.add_word(word, freq).is_err() {
            assert!(m.total.checked_add(freq).is_none(), "add_word failed");
            return;
        }
        m.total += freq;
        m.words.insert(word.to_owned(), freq);
        if freq == 0 {
            fs.insert(word.into());
        }
    }
}

fn prefixes(w: &str) -> impl Iterator<Item = &str> {
    w.char_indices().map(move |(i, c)| &w[..i + c.len_utf8()])
}

fn check_dict(d: &Dictionary, base: &Dictionary, m: &Loaded, custom: bool) {
    assert_eq!(d.total(), m.total, "total");
    let mut new_keys: HashSet<&str> = HashSet::new();
    for (w, &f) in &m.words {
        assert_eq!(d.get(w), Some(f), "FREQ[{w:?}]");
        for p in prefixes(w) {
            assert!(d.get(p).is_some(), "prefix {p:?} of {w:?}");
            if base.get(p).is_none() {
                new_keys.insert(p);
            }
        }
    }
    assert_eq!(d.len(), base.len() + new_keys.len(), "len(FREQ)");
    if custom {
        let entries = d.entries();
        assert_eq!(entries.len(), d.len());
        let mut seen = HashSet::new();
        for (k, v) in &entries {
            assert!(seen.insert(k.as_str()), "duplicate key {k:?}");
            assert_eq!(d.get(k), Some(*v), "FREQ.items() and get disagree on {k:?}");
        }
    }
}

fn check_cut(d: &Dictionary, fs: &ForceSplit, text: &str, hmm: bool) {
    let mut words = Vec::new();
    let r = d.cut(text, hmm, fs, &mut words);
    let blocks = text.chars().any(is_han_default);
    assert_eq!(
        r.is_err(),
        d.total() == 0 && blocks,
        "cut {r:?} with total {}",
        d.total()
    );
    if let Err(e) = r {
        assert_eq!(e, CutError::MathDomain);
        return;
    }
    assert!(
        words.iter().all(|w| !w.is_empty()),
        "an empty word: {words:?}"
    );
    assert_eq!(words.concat(), text, "cut covers the text");
    let mut again = Vec::new();
    d.cut(text, hmm, fs, &mut again).expect("deterministic");
    assert_eq!(again, words, "cut is deterministic");

    let chars: Vec<char> = text.chars().collect();
    let mut toks = Vec::new();
    d.tokenize(text, false, hmm, fs, &mut toks).expect("as cut");
    let mut at = 0;
    for (&(w, s, e), &cw) in toks.iter().zip(&words) {
        assert_eq!(w, cw, "tokenize's words are cut's");
        assert_eq!(s, at, "contiguous offsets");
        assert_eq!(e - s, w.chars().count());
        at = e;
    }
    assert_eq!(toks.len(), words.len());
    assert_eq!(at, chars.len());

    let mut search = Vec::new();
    d.cut_for_search(text, hmm, fs, &mut search)
        .expect("as cut");
    let mut stoks = Vec::new();
    d.tokenize(text, true, hmm, fs, &mut stoks).expect("as cut");
    let swords: Vec<&str> = stoks.iter().map(|t| t.0).collect();
    assert_eq!(
        swords, search,
        "tokenize(search) words are cut_for_search's"
    );
    for &(w, s, e) in &stoks {
        let at: String = chars[s..e].iter().collect();
        assert_eq!(at, w, "search token at {s}..{e}");
    }
}

fn check_dag(d: &Dictionary, text: &str) {
    let n = text.chars().count();
    let dag = d.dag(text);
    assert_eq!(dag.len(), n, "one DAG entry per character");
    for (k, ends) in dag.iter().enumerate() {
        assert!(!ends.is_empty());
        assert!(ends.windows(2).all(|w| w[0] < w[1]), "DAG[{k}] = {ends:?}");
        assert!(
            ends[0] >= k && ends[ends.len() - 1] < n,
            "DAG[{k}] = {ends:?}"
        );
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&[flags, n], rest)) = data.split_first_chunk::<2>() else {
        return;
    };
    let (text, dict) = rest.split_at(usize::from(n).min(rest.len()));
    let text = String::from_utf8_lossy(text);
    let hmm = flags & 1 == 0;

    let mut m = Loaded {
        words: HashMap::new(),
        total: 0,
    };
    let (base, user, custom) = match (flags & 2 != 0)
        .then(|| dict.iter().position(|&b| b == 0))
        .flatten()
    {
        Some(nul) => match main_dict(&dict[..nul]) {
            Some((entries, total)) => {
                for (w, f) in &entries {
                    m.words.insert(w.clone(), *f);
                }
                m.total = total;
                let base = Dictionary::from_entries(entries, total).expect("total fits");
                (base, &dict[nul + 1..], true)
            }
            None => (Dictionary::default_ref().clone(), &dict[nul + 1..], false),
        },
        None => (Dictionary::default_ref().clone(), dict, false),
    };
    if !custom {
        m.total = base.total();
    }
    check_dict(&base, &base, &m, custom);
    let mut d = base.clone();
    let mut fs = ForceSplit::default();
    load_userdict(&mut d, &mut fs, user, &mut m);
    if flags & 4 != 0 {
        load_userdict(&mut d, &mut fs, user, &mut m);
    }
    check_dict(&d, &base, &m, custom);

    check_cut(&d, &fs, &text, hmm);
    check_dag(&d, &text);
    let mut all: Vec<Cow<'_, str>> = Vec::new();
    d.cut_all(&text, &mut all);
    let mut again = Vec::new();
    d.cut_all(&text, &mut again);
    assert_eq!(all, again, "cut_all is deterministic");
});
