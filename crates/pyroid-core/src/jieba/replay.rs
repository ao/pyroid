// Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi, and datatrove 0.10.1
// (Apache-2.0), © HuggingFace; modified.
//! Differential tests: replay what jieba 0.42.1 (and datatrove 0.10.1's
//! Chinese word tokenizer) returned, as recorded by `testdata/oracle.py`,
//! through this module's API, and require identical results.
//!
//! * `testdata/golden-{tests,fineweb-1,fineweb-2,mixed,fuzz}.jsonl`: texts
//!   cut with the default dictionary: `cut` with and without the HMM,
//!   `cut_all`, `cut_for_search` with and without the HMM, `tokenize` in both
//!   modes (the positions of the cuts), and for most rows datatrove's
//!   `word_tokenize` and `span_tokenize` ([`ZhTokenizer`]).
//! * `testdata/golden-scenarios.jsonl`: `add_word`, `del_word`,
//!   `suggest_freq`, `load_userdict` and custom dictionaries, step by step,
//!   with `finalseg.Force_Split_Words` modelled as jieba keeps it.
//! * `testdata/golden-dict-small.jsonl`: the same for jieba's
//!   `extra_dict/dict.txt.small`, replayed when `JIEBA_DICT_SMALL` names
//!   that file (it is not committed).
//!
//! `PYROID_JIEBA_GOLDEN_DIR` replays the files of another recording instead
//! (`oracle.py record DIR`). `oracle.py` describes the rows and the compact
//! word lists.

#![allow(clippy::print_stderr)] // reports what was replayed

use std::fmt::{Debug, Write as _};
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use super::zh::ZhTokenizer;
use super::{CutError, DictError, Dictionary, ForceSplit, SuggestError};
use crate::filters::WordTokenizer;
use crate::ftfy::UnicodeVersion;

/// Mismatches shown in full before a test fails.
const MAX_EXAMPLES: usize = 10;
/// Characters of a text shown in a failure message.
const MAX_SHOWN: usize = 160;
/// Words shown around the first difference of two word lists.
const WINDOW: usize = 4;

const CORPORA: [&str; 5] = [
    "golden-tests.jsonl",
    "golden-fineweb-1.jsonl",
    "golden-fineweb-2.jsonl",
    "golden-mixed.jsonl",
    "golden-fuzz.jsonl",
];

type Row = Map<String, Value>;

/// One expected word and its character offset (`None` for a literal word).
type Word = (String, Option<usize>);

/// What a call returned: a list, or the name of the Python exception.
type Outcome<T> = Result<Vec<T>, String>;

/// `testdata/name`, or `$PYROID_JIEBA_GOLDEN_DIR/name` (another recording).
fn testdata(name: &str) -> PathBuf {
    std::env::var_os("PYROID_JIEBA_GOLDEN_DIR")
        .map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("src/jieba/testdata"),
            PathBuf::from,
        )
        .join(name)
}

fn read_rows(name: &str) -> Vec<Row> {
    let path = testdata(name);
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    data.lines()
        .enumerate()
        .map(|(i, line)| match serde_json::from_str(line) {
            Ok(Value::Object(row)) => row,
            _ => panic!("{name}:{}: not a JSON object", i + 1),
        })
        .collect()
}

/// The Unicode version datatrove's fields were recorded under.
fn recorded_unicode() -> UnicodeVersion {
    let path = testdata("recorded.json");
    let data = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    let meta: Value = serde_json::from_str(&data).expect("recorded.json is JSON");
    let v = meta["recorded_with"]["unicode"]
        .as_str()
        .expect("recorded_with.unicode");
    UnicodeVersion::from_unidata_version(v).unwrap_or_else(|| panic!("Unicode {v}"))
}

/// A text and the byte offset of each of its characters.
struct Text<'a> {
    s: &'a str,
    /// Byte offset of each character, then `s.len()`.
    offs: Vec<usize>,
}

impl<'a> Text<'a> {
    fn new(s: &'a str) -> Self {
        let mut offs: Vec<usize> = s.char_indices().map(|(i, _)| i).collect();
        offs.push(s.len());
        Text { s, offs }
    }

    fn chars(&self) -> usize {
        self.offs.len() - 1
    }

    /// Characters `a..b`.
    fn slice(&self, a: usize, b: usize) -> &'a str {
        &self.s[self.offs[a]..self.offs[b]]
    }

    /// The character offset of byte offset `b`.
    fn char_at(&self, b: usize) -> usize {
        self.offs
            .binary_search(&b)
            .unwrap_or_else(|_| panic!("byte {b} is not a character boundary"))
    }
}

/// `text` Debug-escaped, cut to `MAX_SHOWN` characters.
fn show(text: &str) -> String {
    let shown: String = text.chars().take(MAX_SHOWN).collect();
    if shown.len() == text.len() {
        format!("{shown:?}")
    } else {
        format!("{shown:?}... ({} chars)", text.chars().count())
    }
}

// --------------------------------------------------------------------------
// The compact word lists

fn parse_num<T: std::str::FromStr>(s: &str, code: &str) -> Result<T, String> {
    s.parse()
        .map_err(|_| format!("bad number {s:?} in {code:?}"))
}

/// Decode a word list string (see `oracle.py`'s `code`).
fn decode_code(text: &Text, code: &str) -> Result<Vec<Word>, String> {
    if !code.is_ascii() {
        return Err(format!("bad code {code:?}"));
    }
    let bytes = code.as_bytes();
    let mut out = Vec::new();
    // The previous word's start and end (characters).
    let (mut prev_start, mut prev_end) = (0usize, 0usize);
    let mut pos = 0;
    while pos < bytes.len() {
        let (start, len) = match bytes[pos] {
            digit @ b'0'..=b'9' => {
                pos += 1;
                (Some(prev_end), usize::from(digit - b'0'))
            }
            open @ (b'(' | b'<') => {
                let close = if open == b'(' { ')' } else { '>' };
                let end = code[pos..]
                    .find(close)
                    .ok_or_else(|| format!("unclosed item in {code:?}"))?
                    + pos;
                let inner = &code[pos + 1..end];
                pos = end + 1;
                if open == b'(' {
                    (Some(prev_end), parse_num(inner, code)?)
                } else if let Some((delta, len)) = inner.split_once(',') {
                    (
                        prev_start.checked_add_signed(parse_num(delta, code)?),
                        parse_num(len, code)?,
                    )
                } else {
                    (Some(prev_start), parse_num(inner, code)?)
                }
            }
            _ => return Err(format!("bad item at {pos} in {code:?}")),
        };
        let start = start
            .filter(|&start| start + len <= text.chars())
            .ok_or_else(|| format!("{code:?} leaves the text"))?;
        out.push((text.slice(start, start + len).to_owned(), Some(start)));
        prev_start = start;
        prev_end = start + len;
    }
    Ok(out)
}

/// Field `key` of `row`: the words (and offsets) jieba returned, or the
/// exception it raised.
fn field(t: &Text, row: &Row, key: &str) -> Result<Outcome<Word>, String> {
    let v = row.get(key).ok_or_else(|| format!("no {key:?}"))?;
    let v = match v.as_str().and_then(|s| s.strip_prefix('=')) {
        Some(other) => {
            let r = row.get(other).ok_or_else(|| format!("no {other:?}"))?;
            if r.as_str().is_some_and(|s| s.starts_with('=')) {
                return Err(format!("{key:?} refers to a reference"));
            }
            r
        }
        None => v,
    };
    match v {
        Value::String(code) => decode_code(t, code).map(Ok),
        Value::Array(words) => words
            .iter()
            .map(|w| {
                w.as_str()
                    .map(|w| (w.to_owned(), None))
                    .ok_or_else(|| format!("{key:?}: a word is not a string"))
            })
            .collect::<Result<_, _>>()
            .map(Ok),
        Value::Object(o) => o
            .get("raise")
            .and_then(Value::as_str)
            .map(|e| Err(e.to_owned()))
            .ok_or_else(|| format!("{key:?}: bad object")),
        _ => Err(format!("{key:?}: bad value")),
    }
}

fn words_of(o: &Outcome<Word>) -> Outcome<String> {
    o.as_ref()
        .map(|ws| ws.iter().map(|(w, _)| w.clone()).collect())
        .map_err(Clone::clone)
}

/// `(word, start, end)` of each word, as `tokenize` yields them.
fn spans_of(o: &Outcome<Word>) -> Result<Outcome<(String, usize, usize)>, String> {
    match o {
        Err(e) => Ok(Err(e.clone())),
        Ok(ws) => ws
            .iter()
            .map(|(w, s)| {
                let s = s.ok_or("a word without its offset")?;
                Ok((w.clone(), s, s + w.chars().count()))
            })
            .collect::<Result<_, String>>()
            .map(Ok),
    }
}

fn ints(v: &Value, what: &str) -> Result<Vec<usize>, String> {
    v.as_array()
        .ok_or_else(|| format!("{what} is not a list"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .and_then(|x| usize::try_from(x).ok())
                .ok_or_else(|| format!("{what}: not an offset"))
        })
        .collect()
}

fn get_i64(row: &Row, key: &str) -> Result<i64, String> {
    row.get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("no integer {key:?}"))
}

fn get_str<'r>(row: &'r Row, key: &str) -> Result<&'r str, String> {
    row.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| format!("no string {key:?}"))
}

// --------------------------------------------------------------------------
// The Rust side, with Python's exception names

fn py_cut(e: CutError) -> String {
    match e {
        CutError::MathDomain => "ValueError".to_owned(),
    }
}

fn py_suggest(e: SuggestError) -> String {
    match e {
        SuggestError::Cut(_) => "ValueError",
        SuggestError::ZeroDivision => "ZeroDivisionError",
        SuggestError::Overflow => "OverflowError",
    }
    .to_owned()
}

fn py_dict(e: &DictError) -> String {
    match e {
        DictError::Overflow => "OverflowError".to_owned(),
    }
}

fn owned(words: Vec<&str>) -> Vec<String> {
    words.into_iter().map(str::to_owned).collect()
}

fn cut(d: &Dictionary, fs: &ForceSplit, t: &str, hmm: bool) -> Outcome<String> {
    let mut out = Vec::new();
    d.cut(t, hmm, fs, &mut out).map_err(py_cut)?;
    Ok(owned(out))
}

fn cut_for_search(d: &Dictionary, fs: &ForceSplit, t: &str, hmm: bool) -> Outcome<String> {
    let mut out = Vec::new();
    d.cut_for_search(t, hmm, fs, &mut out).map_err(py_cut)?;
    Ok(owned(out))
}

fn cut_all(d: &Dictionary, t: &str) -> Vec<String> {
    let mut out = Vec::new();
    d.cut_all(t, &mut out);
    out.into_iter().map(std::borrow::Cow::into_owned).collect()
}

fn tokenize(
    d: &Dictionary,
    fs: &ForceSplit,
    t: &str,
    search: bool,
    hmm: bool,
) -> Outcome<(String, usize, usize)> {
    let mut out = Vec::new();
    d.tokenize(t, search, hmm, fs, &mut out).map_err(py_cut)?;
    Ok(out
        .into_iter()
        .map(|(w, s, e)| (w.to_owned(), s, e))
        .collect())
}

// --------------------------------------------------------------------------
// Comparing

/// Counts for one file.
#[derive(Default)]
struct Tally {
    rows: usize,
    comparisons: usize,
    mismatches: usize,
}

/// Counts and the first mismatches, over every replayed file.
#[derive(Default)]
struct Replay {
    files: Vec<(String, Tally)>,
    examples: Vec<String>,
    errors: Vec<String>,
}

/// The window of two lists around their first difference.
fn list_diff<T: PartialEq + Debug>(want: &Outcome<T>, got: &Outcome<T>) -> String {
    let (Ok(w), Ok(g)) = (want, got) else {
        return format!("recorded: {}\n    rust:     {}", brief(want), brief(got));
    };
    let first = w
        .iter()
        .zip(g)
        .position(|(a, b)| a != b)
        .unwrap_or_else(|| w.len().min(g.len()));
    let lo = first.saturating_sub(WINDOW);
    let window = |v: &[T]| format!("{:?}", &v[lo.min(v.len())..(first + WINDOW).min(v.len())]);
    format!(
        "first difference at item {first} (recorded {}, rust {}):\n    recorded[{lo}..]: {}\n    rust    [{lo}..]: {}",
        w.len(),
        g.len(),
        window(w),
        window(g)
    )
}

fn brief<T: Debug>(o: &Outcome<T>) -> String {
    match o {
        Ok(v) => format!("{} items", v.len()),
        Err(e) => format!("raised {e}"),
    }
}

impl Replay {
    fn tally(&mut self) -> &mut Tally {
        &mut self.files.last_mut().expect("a file").1
    }

    fn start(&mut self, name: &str) {
        self.files.push((name.to_owned(), Tally::default()));
    }

    fn fail(&mut self, what: String) {
        self.tally().mismatches += 1;
        if self.examples.len() < MAX_EXAMPLES {
            self.examples.push(what);
        }
    }

    /// Compare two lists (or exceptions).
    fn list<T: PartialEq + Debug>(
        &mut self,
        at: &dyn Fn() -> String,
        what: &str,
        want: &Outcome<T>,
        got: &Outcome<T>,
    ) {
        self.tally().comparisons += 1;
        if want != got {
            let msg = format!("{}\n  {what}: {}", at(), list_diff(want, got));
            self.fail(msg);
        }
    }

    /// Compare two values.
    fn value<T: PartialEq + Debug>(
        &mut self,
        at: &dyn Fn() -> String,
        what: &str,
        want: &T,
        got: &T,
    ) {
        self.tally().comparisons += 1;
        if want != got {
            let msg = format!("{}\n  {what}: recorded {want:?}, rust {got:?}", at());
            self.fail(msg);
        }
    }

    fn error(&mut self, at: &str, e: &str) {
        self.errors.push(format!("{at}: {e}"));
    }

    /// The five cuts and four tokenize outputs of `row` against `d`.
    fn cuts(
        &mut self,
        at: &dyn Fn() -> String,
        t: &Text,
        row: &Row,
        d: &Dictionary,
        fs: &ForceSplit,
    ) -> Result<(), String> {
        for (key, search, hmm) in [
            ("c", false, true),
            ("c0", false, false),
            ("s", true, true),
            ("s0", true, false),
        ] {
            let want = field(t, row, key)?;
            let got = if search {
                cut_for_search(d, fs, t.s, hmm)
            } else {
                cut(d, fs, t.s, hmm)
            };
            self.list(at, key, &words_of(&want), &got);
            let got = tokenize(d, fs, t.s, search, hmm);
            self.list(at, &format!("{key} (tokenize)"), &spans_of(&want)?, &got);
        }
        let want = field(t, row, "a")?;
        self.list(at, "a", &words_of(&want), &Ok(cut_all(d, t.s)));
        Ok(())
    }

    /// datatrove's `word_tokenize` and `span_tokenize` of `row`, if recorded.
    fn zh(
        &mut self,
        at: &dyn Fn() -> String,
        t: &Text,
        row: &Row,
        zh: &ZhTokenizer,
    ) -> Result<bool, String> {
        let want: Outcome<String> = if let Some(drop) = row.get("zd") {
            let drop = ints(drop, "zd")?;
            words_of(&field(t, row, "c")?).map(|ws| {
                ws.into_iter()
                    .enumerate()
                    .filter(|(i, _)| !drop.contains(i))
                    .map(|(_, w)| w)
                    .collect()
            })
        } else if row.contains_key("zw") {
            words_of(&field(t, row, "zw")?)
        } else {
            return Ok(false);
        };
        let mut words = Vec::new();
        zh.words(t.s, &mut words).map_err(|e| format!("{e:?}"))?;
        self.list(at, "datatrove words", &want, &Ok(owned(words)));

        let spans = ints(row.get("zs").ok_or("no \"zs\"")?, "zs")?;
        let want_starts: Vec<usize> = spans.iter().step_by(2).copied().collect();
        let want_ends: Vec<usize> = spans.iter().skip(1).step_by(2).copied().collect();
        let mut ends = Vec::new();
        zh.sentence_ends(t.s, &mut ends)
            .map_err(|e| format!("{e:?}"))?;
        let got_ends: Vec<usize> = ends.iter().map(|&b| t.char_at(b)).collect();
        self.value(at, "datatrove sentence ends", &want_ends, &got_ends);
        // Each sentence starts at the token after the previous one's last.
        let mut toks = Vec::new();
        zh.tokens(t.s, &mut toks);
        let mut starts = Vec::new();
        if let Some(&(first, _)) = toks.first() {
            starts.push(t.char_at(first));
            for &end in &ends[..ends.len().saturating_sub(1)] {
                let j = toks
                    .iter()
                    .position(|&(_, e)| e == end)
                    .ok_or("a sentence end that ends no token")?;
                let next = toks.get(j + 1).ok_or("a sentence end at the last token")?;
                starts.push(t.char_at(next.0));
            }
        }
        self.value(at, "datatrove sentence starts", &want_starts, &starts);
        Ok(true)
    }

    fn report(&self, title: &str) {
        let (mut rows, mut comparisons, mut mismatches) = (0, 0, 0);
        for (name, t) in &self.files {
            eprintln!(
                "{name}: {} rows, {} comparisons, {} mismatches",
                t.rows, t.comparisons, t.mismatches
            );
            rows += t.rows;
            comparisons += t.comparisons;
            mismatches += t.mismatches;
        }
        if mismatches + self.errors.len() > 0 {
            let mut msg = String::new();
            for e in &self.examples {
                writeln!(msg, "{e}\n").unwrap();
            }
            for e in self.errors.iter().take(MAX_EXAMPLES) {
                writeln!(msg, "error: {e}").unwrap();
            }
            write!(
                msg,
                "{rows} rows, {comparisons} comparisons, {mismatches} mismatches, {} unreplayable rows",
                self.errors.len()
            )
            .unwrap();
            panic!("{title}: the Rust port differs from the recording:\n{msg}");
        }
        eprintln!("{title}: {rows} rows, {comparisons} comparisons, all match");
    }
}

// --------------------------------------------------------------------------
// The corpora

fn replay_corpus(r: &mut Replay, name: &str, zh: &ZhTokenizer) {
    r.start(name);
    let d = Dictionary::default_ref();
    let fs = ForceSplit::default();
    let mut with_zh = 0;
    for (n, row) in read_rows(name).iter().enumerate() {
        let Some(text) = row.get("t").and_then(Value::as_str) else {
            r.error(&format!("{name}:{}", n + 1), "no \"t\"");
            continue;
        };
        let t = Text::new(text);
        let at = || format!("{name}:{}: {}", n + 1, show(text));
        r.tally().rows += 1;
        let done = r
            .cuts(&at, &t, row, d, &fs)
            .and_then(|()| r.zh(&at, &t, row, zh));
        match done {
            Ok(z) => with_zh += usize::from(z),
            Err(e) => r.error(&format!("{name}:{}", n + 1), &e),
        }
    }
    eprintln!("{name}: {with_zh} rows with datatrove's fields");
}

#[test]
fn corpora_match_jieba() {
    let zh = ZhTokenizer::get(recorded_unicode());
    let mut r = Replay::default();
    for name in CORPORA {
        replay_corpus(&mut r, name, zh);
    }
    r.report("the corpora");
}

// --------------------------------------------------------------------------
// The scenarios

/// One `jieba.Tokenizer` and `finalseg.Force_Split_Words`.
struct State {
    scn: String,
    d: Dictionary,
    fs: ForceSplit,
}

impl State {
    /// `add_word(w, freq)` with a known `freq`, as the Python layer calls it.
    fn add(&mut self, w: &str, freq: i64) -> Result<(), String> {
        self.d.add_word(w, freq).map_err(|e| py_dict(&e))?;
        if freq == 0 {
            self.fs.insert(w.into());
        }
        Ok(())
    }
}

/// `len(FREQ)`, `total` and the `FREQ.get` queries recorded after a step.
fn check_state(
    r: &mut Replay,
    at: &dyn Fn() -> String,
    row: &Row,
    st: &State,
) -> Result<(), String> {
    if row.contains_key("len") {
        let len = usize::try_from(get_i64(row, "len")?).map_err(|e| e.to_string())?;
        r.value(at, "len(FREQ)", &len, &st.d.len());
        r.value(at, "total", &get_i64(row, "total")?, &st.d.total());
    }
    if let Some(gets) = row.get("get") {
        for g in gets.as_array().ok_or("\"get\" is not a list")? {
            let q = g[0].as_str().ok_or("a query is not a string")?;
            let want = g[1].as_i64();
            r.value(at, &format!("FREQ.get({q:?})"), &want, &st.d.get(q));
        }
    }
    Ok(())
}

/// The dictionary of a "new" row.
fn new_dict(row: &Row, dict_small: Option<&Path>) -> Result<Dictionary, String> {
    match get_str(row, "dict")? {
        "default" => Ok(Dictionary::default_ref().clone()),
        "dict.txt.small" => read_dict_file(dict_small.ok_or("no dict.txt.small")?),
        _ => {
            let entries = row
                .get("entries")
                .and_then(Value::as_array)
                .ok_or("no \"entries\"")?
                .iter()
                .map(|e| {
                    let w = e[0].as_str().ok_or("an entry's word")?;
                    let f = e[1].as_i64().ok_or("an entry's frequency")?;
                    Ok((w.to_owned(), f))
                })
                .collect::<Result<Vec<_>, String>>()?;
            Dictionary::from_entries(entries, get_i64(row, "total")?).map_err(|e| py_dict(&e))
        }
    }
}

/// A dictionary file in `dict.txt`'s format (`word freq [tag]` per line), as
/// `gen_pfdict` reads it.
fn read_dict_file(path: &Path) -> Result<Dictionary, String> {
    let data = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut entries = Vec::new();
    let mut total = 0i64;
    for (i, line) in data.split_terminator('\n').enumerate() {
        // bytes.strip(): ASCII whitespace.
        let line =
            line.trim_matches(|c: char| matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'));
        let mut parts = line.split(' ');
        let word = parts.next().unwrap_or_default();
        let freq: i64 = parts
            .next()
            .and_then(|f| f.parse().ok())
            .ok_or_else(|| format!("{}:{}: not `word freq`", path.display(), i + 1))?;
        total = total.checked_add(freq).ok_or("total overflows")?;
        entries.push((word.to_owned(), freq));
    }
    Dictionary::from_entries(entries, total).map_err(|e| py_dict(&e))
}

fn replay_step(
    r: &mut Replay,
    at: &dyn Fn() -> String,
    row: &Row,
    state: &mut Option<State>,
    dict_small: Option<&Path>,
) -> Result<(), String> {
    let op = get_str(row, "op")?;
    if op == "new" {
        let keep = row.get("keep_fs").and_then(Value::as_bool) == Some(true);
        let fs = match state.take() {
            Some(old) if keep => old.fs,
            _ => ForceSplit::default(),
        };
        let st = state.insert(State {
            scn: get_str(row, "scn")?.to_owned(),
            d: new_dict(row, dict_small)?,
            fs,
        });
        check_state(r, at, row, st)?;
        if let Some(items) = row.get("items").and_then(Value::as_array) {
            let mut want = items
                .iter()
                .map(|kv| Some((kv[0].as_str()?.to_owned(), kv[1].as_i64()?)))
                .collect::<Option<Vec<_>>>()
                .ok_or("bad \"items\"")?;
            want.sort();
            let mut got = st.d.entries();
            got.sort();
            r.list(at, "FREQ.items()", &Ok::<_, String>(want), &Ok(got));
        }
        return Ok(());
    }
    let st = state.as_mut().ok_or("a step before \"new\"")?;
    match op {
        "add" => {
            let w = get_str(row, "w")?;
            let mut freq = row.get("f").and_then(Value::as_i64);
            if freq.is_none() {
                // add_word(w): suggest_freq(w, tune=False) decides.
                let got = st.d.suggest_freq(w).map_err(py_suggest);
                let want = match row.get("raise").and_then(Value::as_str) {
                    Some(e) => Err(e.to_owned()),
                    None => Ok(get_i64(row, "freq")?),
                };
                r.value(at, "suggest_freq", &want, &got);
                freq = got.ok();
            }
            if let Some(f) = freq {
                st.add(w, f)?;
            }
        }
        "del" => st.add(get_str(row, "w")?, 0)?,
        "sf" => {
            let seg = row.get("seg").ok_or("no \"seg\"")?;
            let (word, got) = match seg {
                Value::String(w) => (w.clone(), st.d.suggest_freq(w)),
                Value::Array(parts) => {
                    let parts: Vec<&str> = parts
                        .iter()
                        .map(|p| p.as_str().ok_or("a segment is not a string"))
                        .collect::<Result<_, _>>()?;
                    (parts.concat(), st.d.suggest_freq_split(&parts))
                }
                _ => return Err("bad \"seg\"".into()),
            };
            let got = got.map_err(py_suggest);
            let want = match row.get("raise").and_then(Value::as_str) {
                Some(e) => Err(e.to_owned()),
                None => Ok(get_i64(row, "ret")?),
            };
            r.value(at, "suggest_freq", &want, &got);
            if row.get("tune").and_then(Value::as_bool) == Some(true) {
                if let Ok(f) = got {
                    st.add(&word, f)?;
                }
            }
        }
        "probe" => {
            let t = Text::new(get_str(row, "t")?);
            return r.cuts(at, &t, row, &st.d, &st.fs);
        }
        _ => return Err(format!("unknown op {op:?}")),
    }
    check_state(r, at, row, st)
}

fn replay_scenarios(r: &mut Replay, name: &str, dict_small: Option<&Path>) {
    r.start(name);
    let mut state: Option<State> = None;
    for (n, row) in read_rows(name).iter().enumerate() {
        r.tally().rows += 1;
        let scn = match (&state, row.get("scn").and_then(Value::as_str)) {
            (_, Some(s)) => s.to_owned(),
            (Some(st), None) => st.scn.clone(),
            (None, None) => String::new(),
        };
        let what = row
            .get("t")
            .or_else(|| row.get("w"))
            .or_else(|| row.get("seg"))
            .map(|v| match v.as_str() {
                Some(s) => show(s),
                None => v.to_string(),
            })
            .unwrap_or_default();
        let op = row.get("op").and_then(Value::as_str).unwrap_or("?");
        let at = || format!("{name}:{} [{scn}] {op} {what}", n + 1);
        if let Err(e) = replay_step(r, &at, row, &mut state, dict_small) {
            r.error(&at(), &e);
        }
    }
}

#[test]
fn scenarios_match_jieba() {
    let mut r = Replay::default();
    replay_scenarios(&mut r, "golden-scenarios.jsonl", None);
    r.report("the scenarios");
}

/// jieba's `extra_dict/dict.txt.small` (sha256 479cf8cc...ff664), from
/// https://raw.githubusercontent.com/fxsjy/jieba/v0.42.1/extra_dict/dict.txt.small
#[test]
fn dict_small_matches_jieba() {
    let Some(path) = std::env::var_os("JIEBA_DICT_SMALL").map(PathBuf::from) else {
        eprintln!("JIEBA_DICT_SMALL not set; skipping golden-dict-small.jsonl");
        return;
    };
    let mut r = Replay::default();
    replay_scenarios(&mut r, "golden-dict-small.jsonl", Some(&path));
    r.report("dict.txt.small");
}

#[test]
fn word_list_codes() {
    let t = Text::new("中国科学院abc");
    let words = |code: &str| -> Vec<Word> { decode_code(&t, code).unwrap() };
    let w = |s: &str, at: usize| (s.to_owned(), Some(at));
    assert_eq!(words(""), Vec::<Word>::new());
    assert_eq!(words("23"), [w("中国", 0), w("科学院", 2)]);
    assert_eq!(
        words("22<1,2><-1,3><-2,5>"),
        [
            w("中国", 0),
            w("科学", 2),
            w("学院", 3),
            w("科学院", 2),
            w("中国科学院", 0)
        ]
    );
    assert_eq!(words("<5>"), [w("中国科学院", 0)]);
    assert_eq!(words("(8)"), [w("中国科学院abc", 0)]);
    assert_eq!(
        words("50300"),
        [
            w("中国科学院", 0),
            w("", 5),
            w("abc", 5),
            w("", 8),
            w("", 8)
        ]
    );
    for bad in ["9", "(9)", "<-1,1>", "<3,9>", "x", "(2", "<1,>"] {
        assert!(decode_code(&t, bad).is_err(), "{bad}");
    }
}
