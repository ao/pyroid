// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! Sentence-level deduplication: datatrove 0.10.1's `SentenceDedupSignature`
//! -> `SentenceFindDedups` -> `SentenceDedupFilter` (C4: "we discarded all
//! but one of any three-sentence span occurring more than once").
//!
//! - Stage 1 ([`SentenceDeduper::hashes`]): the document's sentences
//!   (`sent_tokenize`: spaCy sentences, stripped, empty ones dropped; or
//!   `str.splitlines()`), each through `simplify_text`; every window of
//!   `n_sentences` of them, joined by `" "`, is hashed with `hash_fc`
//!   unless it is blank. The window's position counts the skipped ones.
//!   [`signature_files`] lays the sorted signatures out as datatrove's
//!   `.c4_sig` files.
//! - Stage 2 ([`find_duplicates`], [`Duplicates::within`]): a merge of the
//!   signature files in datatrove's heap order `(hash, doc, file, window)`;
//!   in each run of equal hashes every signature after the first is a
//!   duplicate (with index files: the ones matching the index).
//! - Stage 3 ([`SentenceDeduper::filter`]): spaCy's `span_tokenize` (every
//!   sentence, whitespace-only ones included: the indices are datatrove's,
//!   shifts and all), the duplicate windows dropped, then datatrove's keep
//!   conditions.

use std::borrow::Cow;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fmt;
use std::io::{self, Read};

use twox_hash::{XxHash32, XxHash64};

use super::config::{ConfigError, HashConfig, HashFn, Precision, TextNormConfig, UnicodeVersion};
use super::minhash::sha1_prefix64;
use super::simplify::simplify_text;
use crate::filters::{FilterError, WordTokenizer, tokenizer_for};
use crate::text::is_py_whitespace;

/// datatrove's `SentDedupConfig`, field for field. The three thresholds use
/// 0 for "disabled" (datatrove disables them for any value `<= 0`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SentDedupConfig {
    /// Sentences per window (at least 1).
    pub n_sentences: usize,
    /// spaCy sentences (`true`) or lines (`false`).
    pub split_sentences: bool,
    /// With index files: only remove windows that are in the index.
    pub only_dedup_in_index: bool,
    /// Drop a changed document with fewer words (0: no minimum).
    pub min_doc_words: usize,
    /// Drop a changed document with fewer sentences (0: no minimum).
    pub min_num_sentences: usize,
    /// Restore a removed run whose first span has fewer words (0: never).
    pub min_words_to_remove_span: usize,
    pub norm_config: TextNormConfig,
    pub hash_config: HashConfig,
}

impl Default for SentDedupConfig {
    fn default() -> Self {
        Self {
            n_sentences: 3,
            split_sentences: true,
            only_dedup_in_index: true,
            min_doc_words: 50,
            min_num_sentences: 3,
            min_words_to_remove_span: 0,
            norm_config: TextNormConfig::default(),
            hash_config: HashConfig::default(),
        }
    }
}

impl SentDedupConfig {
    /// Does stage 1 (`stage3 = false`) or stage 3 (`true`) need the word
    /// tokenizer?
    pub fn needs_tokenizer(&self, stage3: bool) -> bool {
        self.split_sentences
            || (stage3
                && (self.min_doc_words > 0
                    || self.min_num_sentences > 0
                    || self.min_words_to_remove_span > 0))
    }
}

/// Why sentence dedup failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SentenceError {
    /// spaCy's `[E025]` (a text of 2**30 characters or more).
    Tokenize(FilterError),
    /// The configuration needs a word tokenizer and the language has none.
    NoTokenizer,
    /// datatrove stores the window index as `<u2`: numpy raises
    /// `OverflowError: Python integer N out of bounds for uint16`.
    SentenceOverflow(usize),
    /// datatrove stores the document index as `<u4` (same `OverflowError`).
    DocOverflow(usize),
    /// datatrove's `ValueError("Error with duplicate line index")`: a
    /// duplicate listed twice for one document.
    DuplicateLineIndex,
}

impl From<FilterError> for SentenceError {
    fn from(e: FilterError) -> Self {
        Self::Tokenize(e)
    }
}

impl fmt::Display for SentenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokenize(e) => e.fmt(f),
            Self::NoTokenizer => f.write_str("no word tokenizer for this language"),
            Self::SentenceOverflow(n) => write!(f, "Python integer {n} out of bounds for uint16"),
            Self::DocOverflow(n) => write!(f, "Python integer {n} out of bounds for uint32"),
            Self::DuplicateLineIndex => f.write_str("Error with duplicate line index"),
        }
    }
}

impl std::error::Error for SentenceError {}

/// `hash_fc(s)` for one window string (datatrove's `create_hash_func`): the
/// same functions as the MinHash shingles.
#[inline]
pub fn hash_bytes(hc: HashConfig, bytes: &[u8]) -> u64 {
    match (hc.hash_fc, hc.precision) {
        (HashFn::Xxhash, Precision::P64) => XxHash64::oneshot(0, bytes),
        (HashFn::Xxhash, Precision::P32) => u64::from(XxHash32::oneshot(0, bytes)),
        (HashFn::Sha1, Precision::P64) => sha1_prefix64(bytes),
        (HashFn::Sha1, Precision::P32) => sha1_prefix64(bytes) & u64::from(u32::MAX),
    }
}

/// `s.strip()`.
#[inline]
fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_whitespace)
}

/// `str.splitlines()` boundaries (`\r\n` counts as one).
#[inline]
fn is_line_break(c: char) -> bool {
    matches!(
        c,
        '\n' | '\r'
            | '\x0b'
            | '\x0c'
            | '\x1c'
            | '\x1d'
            | '\x1e'
            | '\u{85}'
            | '\u{2028}'
            | '\u{2029}'
    )
}

/// `s.splitlines()`: lines without their terminators, no trailing empty line.
fn py_splitlines<'a>(s: &'a str, out: &mut Vec<&'a str>) {
    let mut start = 0;
    let mut it = s.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        if is_line_break(c) {
            out.push(&s[start..i]);
            let mut end = i + c.len_utf8();
            if c == '\r' && it.peek().is_some_and(|&(_, n)| n == '\n') {
                it.next();
                end += 1;
            }
            start = end;
        }
    }
    if start < s.len() {
        out.push(&s[start..]);
    }
}

/// One `.c4_sig` record: datatrove's `("hash", "<u8"|"<u4"), ("doc", "<u4"),
/// ("sent", "<u2")`. The derived order is numpy's sort order of those records.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Signature {
    pub hash: u64,
    pub doc: u32,
    pub sent: u16,
}

/// What stage 3 made of one document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Filtered<'a> {
    /// The text afterwards (`Borrowed` of the whole input when unchanged).
    pub text: Cow<'a, str>,
    /// datatrove keeps the document.
    pub keep: bool,
    /// The text with `>>>` / `<<<` around removed runs, as datatrove hands
    /// it to the exclusion writer; `None` for a document with no duplicates.
    pub formatted: Option<String>,
    /// `original_sentences`: spans of the document (0 without duplicates).
    pub spans: usize,
    /// Spans kept (`removed_sentences` is `spans - kept`).
    pub kept: usize,
}

/// One configuration of the three stages: the config, the interpreter's
/// Unicode version and the word tokenizer. Cheap to share between threads.
#[derive(Clone, Copy, Debug)]
pub struct SentenceDeduper {
    config: SentDedupConfig,
    unicode: UnicodeVersion,
    tokenizer: Option<&'static dyn WordTokenizer>,
}

impl SentenceDeduper {
    /// `language` is a datatrove language code; a code pyroid cannot
    /// tokenize is accepted, but every call that needs the tokenizer then
    /// fails with [`SentenceError::NoTokenizer`].
    pub fn new(
        config: SentDedupConfig,
        unicode: UnicodeVersion,
        language: &str,
    ) -> Result<Self, ConfigError> {
        if config.n_sentences == 0 {
            return Err(ConfigError("n_sentences must be at least 1".to_owned()));
        }
        Ok(Self {
            config,
            unicode,
            tokenizer: tokenizer_for(language, unicode),
        })
    }

    pub fn config(&self) -> &SentDedupConfig {
        &self.config
    }

    pub fn unicode(&self) -> UnicodeVersion {
        self.unicode
    }

    /// Is there a word tokenizer for the language?
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    fn tokenizer(&self) -> Result<&'static dyn WordTokenizer, SentenceError> {
        self.tokenizer.ok_or(SentenceError::NoTokenizer)
    }

    /// Stage 1's sentences: `tokenizer.sent_tokenize(text)` (spaCy sentences,
    /// stripped, the empty ones dropped) or `text.splitlines()`.
    pub fn sentences<'a>(
        &self,
        text: &'a str,
        out: &mut Vec<&'a str>,
    ) -> Result<(), SentenceError> {
        if self.config.split_sentences {
            // A sentence's text runs from its first token to its last; the
            // previous sentence ends at most one space (a token's
            // `whitespace_`) before that, which `strip()` removes anyway.
            let mut ends = Vec::new();
            self.tokenizer()?.sentence_ends(text, &mut ends)?;
            let mut start = 0;
            for end in ends {
                let s = py_strip(&text[start..end]);
                if !s.is_empty() {
                    out.push(s);
                }
                start = end;
            }
        } else {
            py_splitlines(text, out);
        }
        Ok(())
    }

    /// Stage 1 for one document (datatrove's `get_hashes`): `(hash, window
    /// index)` per non-blank window, appended to `out`. The index is not
    /// range-checked here (see [`records`]).
    pub fn hashes(&self, text: &str, out: &mut Vec<(u64, usize)>) -> Result<(), SentenceError> {
        let mut sents = Vec::new();
        self.sentences(text, &mut sents)?;
        let n = self.config.n_sentences;
        if sents.len() < n {
            return Ok(());
        }
        let simplified: Vec<Cow<'_, str>> = sents
            .iter()
            .map(|s| simplify_text(s, &self.config.norm_config, self.unicode))
            .collect();
        let hc = self.config.hash_config;
        let mut buf = String::with_capacity(256);
        out.reserve(simplified.len() + 1 - n);
        for (i, window) in simplified.windows(n).enumerate() {
            buf.clear();
            for (j, s) in window.iter().enumerate() {
                if j > 0 {
                    buf.push(' ');
                }
                buf.push_str(s);
            }
            // `if n_sent_gram.strip() != ""`
            if !py_strip(&buf).is_empty() {
                out.push((hash_bytes(hc, buf.as_bytes()), i));
            }
        }
        Ok(())
    }

    /// Stage 3's spans: `doc.text[last_end:end]` for every spaCy sentence
    /// (`span_tokenize`, whitespace-only sentences included), or the lines.
    fn spans<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), SentenceError> {
        if self.config.split_sentences {
            let mut ends = Vec::new();
            self.tokenizer()?.sentence_ends(text, &mut ends)?;
            let mut last = 0;
            for end in ends {
                out.push(&text[last..end]);
                last = end;
            }
        } else {
            py_splitlines(text, out);
        }
        Ok(())
    }

    fn word_count(&self, text: &str) -> Result<usize, SentenceError> {
        let mut words = Vec::new();
        self.tokenizer()?.words(text, &mut words)?;
        Ok(words.len())
    }

    /// datatrove's `min_words_to_remove_span` test on a removed run, of
    /// which datatrove only remembers (and restores) the first span.
    fn restore(&self, first: &str) -> Result<bool, SentenceError> {
        let min = self.config.min_words_to_remove_span;
        Ok(min > 0 && self.word_count(first)? < min)
    }

    /// datatrove's `remove_dup_sentences`: `(filtered_text, original_formatted,
    /// spans, kept)` for a document and its sorted duplicate window indices.
    pub fn remove_dup_sentences(
        &self,
        text: &str,
        dups: &[u16],
    ) -> Result<(String, String, usize, usize), SentenceError> {
        let mut spans = Vec::new();
        self.spans(text, &mut spans)?;
        let n = self.config.n_sentences;
        let mut kept: Vec<&str> = Vec::with_capacity(spans.len());
        let mut formatted: Vec<&str> = Vec::with_capacity(spans.len() + 2);
        // datatrove's `removed_span` list only ever holds a run's first span.
        let mut removed: Option<&str> = None;
        let mut du = 0;
        let mut drop_until = 0;
        for (idx, &line) in spans.iter().enumerate() {
            if let Some(&d) = dups.get(du) {
                let d = usize::from(d);
                if d < idx {
                    return Err(SentenceError::DuplicateLineIndex);
                } else if d == idx {
                    drop_until = idx + n;
                    du += 1;
                }
            }
            if idx >= drop_until {
                if let Some(first) = removed.take() {
                    formatted.push("<<<");
                    if self.restore(first)? {
                        kept.push(first);
                    }
                }
                kept.push(line);
            } else if removed.is_none() {
                removed = Some(line);
                formatted.push(">>>");
            }
            formatted.push(line);
        }
        if let Some(first) = removed {
            formatted.push("<<<");
            if self.restore(first)? {
                kept.push(first);
            }
        }
        let sep = if self.config.split_sentences {
            ""
        } else {
            "\n"
        };
        let joined = kept.join(sep);
        let filtered = match joined.trim_start_matches(is_py_whitespace) {
            s if s.len() == joined.len() => joined,
            s => s.to_owned(),
        };
        Ok((filtered, formatted.join(sep), spans.len(), kept.len()))
    }

    /// datatrove's keep condition for a document whose duplicates were
    /// removed (`original` before, `filtered` after).
    pub fn keeps(&self, original: &str, filtered: &str) -> Result<bool, SentenceError> {
        if filtered.is_empty() {
            return Ok(false);
        }
        if filtered == original {
            return Ok(true);
        }
        let c = &self.config;
        if c.min_doc_words > 0 && self.word_count(filtered)? < c.min_doc_words {
            return Ok(false);
        }
        if c.min_num_sentences > 0
            && self.tokenizer()?.count_sentences(filtered)? < c.min_num_sentences
        {
            return Ok(false);
        }
        Ok(true)
    }

    /// Stage 3 for one document and its sorted duplicate window indices.
    pub fn filter<'a>(&self, text: &'a str, dups: &[u16]) -> Result<Filtered<'a>, SentenceError> {
        if dups.is_empty() {
            return Ok(Filtered {
                text: Cow::Borrowed(text),
                keep: !text.is_empty(),
                formatted: None,
                spans: 0,
                kept: 0,
            });
        }
        let (filtered, formatted, spans, kept) = self.remove_dup_sentences(text, dups)?;
        let keep = self.keeps(text, &filtered)?;
        Ok(Filtered {
            text: if filtered == text {
                Cow::Borrowed(text)
            } else {
                Cow::Owned(filtered)
            },
            keep,
            formatted: Some(formatted),
            spans,
            kept,
        })
    }
}

/// One document's `(hash, window)` pairs as `.c4_sig` records (`doc`: its
/// index in the task), with the range checks of datatrove's
/// `np.array(signatures, dtype=...)`.
pub fn records(
    doc: usize,
    hashes: &[(u64, usize)],
    out: &mut Vec<Signature>,
) -> Result<(), SentenceError> {
    if hashes.is_empty() {
        return Ok(());
    }
    // numpy converts each tuple field by field: the document index first.
    let doc = u32::try_from(doc).map_err(|_| SentenceError::DocOverflow(doc))?;
    let mut recs = Vec::with_capacity(hashes.len());
    for &(hash, s) in hashes {
        let sent = u16::try_from(s).map_err(|_| SentenceError::SentenceOverflow(s))?;
        recs.push(Signature { hash, doc, sent });
    }
    out.append(&mut recs);
    Ok(())
}

/// Bytes per `.c4_sig` record.
fn record_len(precision: Precision) -> usize {
    match precision {
        Precision::P32 => 10,
        Precision::P64 => 14,
    }
}

fn hash_max(precision: Precision) -> u64 {
    match precision {
        Precision::P32 => u64::from(u32::MAX),
        Precision::P64 => u64::MAX,
    }
}

#[allow(clippy::cast_possible_truncation)] // 32-bit hashes are below 2**32
fn push_hash(out: &mut Vec<u8>, hash: u64, precision: Precision) {
    match precision {
        Precision::P32 => out.extend_from_slice(&(hash as u32).to_le_bytes()),
        Precision::P64 => out.extend_from_slice(&hash.to_le_bytes()),
    }
}

/// datatrove's `save_hashes`: `sigs` sorted (in place), split into the
/// `finder_workers` hash ranges. Returns the contents of the files datatrove
/// creates, bucket by bucket (`{bucket:04d}/{rank:05d}.c4_sig` for
/// `bucket in 0..len()`): it stops after the bucket that holds the last
/// signature, and creates one empty file when there are none.
pub fn signature_files(
    sigs: &mut [Signature],
    precision: Precision,
    finder_workers: usize,
) -> Vec<Vec<u8>> {
    let workers = finder_workers.max(1);
    sigs.sort_unstable();
    let max = hash_max(precision);
    let per_worker = max / workers as u64;
    let rec = record_len(precision);
    let mut files = Vec::new();
    let mut left = 0;
    for i in 0..workers {
        let right_hash = if i == workers - 1 {
            max
        } else {
            (i as u64 + 1) * per_worker
        };
        // `searchsorted(right_hash, side="right")`
        let right = left + sigs[left..].partition_point(|s| s.hash <= right_hash);
        let mut buf = Vec::with_capacity((right - left) * rec);
        for s in &sigs[left..right] {
            push_hash(&mut buf, s.hash, precision);
            buf.extend_from_slice(&s.doc.to_le_bytes());
            buf.extend_from_slice(&s.sent.to_le_bytes());
        }
        files.push(buf);
        left = right;
        if right >= sigs.len() {
            break;
        }
    }
    files
}

/// Stage 2's duplicates of one task: per document, the sorted window
/// indices to remove (datatrove's `all_dups` after `sort()`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Duplicates {
    /// Documents with duplicates, ascending.
    docs: Vec<u32>,
    /// `sents[starts[k]..starts[k + 1]]` belong to `docs[k]`.
    starts: Vec<usize>,
    sents: Vec<u16>,
}

/// A malformed signature, index or duplicates file.
#[derive(Debug)]
pub enum FileError {
    Io {
        file: usize,
        err: io::Error,
    },
    /// The size is not a multiple of the record size.
    Truncated {
        file: usize,
        record: usize,
    },
    /// datatrove's `AssertionError: Hash order error`.
    Unsorted {
        file: usize,
    },
}

impl fmt::Display for FileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { file, err } => write!(f, "file {file}: {err}"),
            Self::Truncated { file, record } => write!(
                f,
                "file {file}: the size is not a multiple of the {record}-byte record"
            ),
            Self::Unsorted { file } => write!(f, "file {file}: Hash order error"),
        }
    }
}

impl std::error::Error for FileError {}

impl Duplicates {
    /// From `(doc, window)` pairs in any order (a pair listed twice stays
    /// twice, as in datatrove).
    pub fn from_pairs(mut pairs: Vec<(u32, u16)>) -> Self {
        pairs.sort_unstable();
        let mut d = Self {
            docs: Vec::new(),
            starts: Vec::new(),
            sents: Vec::with_capacity(pairs.len()),
        };
        for (i, &(doc, sent)) in pairs.iter().enumerate() {
            if d.docs.last() != Some(&doc) {
                d.docs.push(doc);
                d.starts.push(i);
            }
            d.sents.push(sent);
        }
        d.starts.push(pairs.len());
        d
    }

    /// The contents of datatrove's `.c4_dup` files for one rank (`<IH`
    /// records), taken together as `SentenceDedupFilter` does.
    pub fn parse(files: &[&[u8]]) -> Result<Self, FileError> {
        let mut pairs = Vec::new();
        for (file, bytes) in files.iter().enumerate() {
            if bytes.len() % 6 != 0 {
                return Err(FileError::Truncated { file, record: 6 });
            }
            pairs.extend(bytes.chunks_exact(6).map(|r| {
                (
                    u32::from_le_bytes([r[0], r[1], r[2], r[3]]),
                    u16::from_le_bytes([r[4], r[5]]),
                )
            }));
        }
        Ok(Self::from_pairs(pairs))
    }

    /// Stage 2 over the signatures of one task, as `SentenceFindDedups`
    /// finds them (with any number of finder workers): in each run of equal
    /// hashes, every signature but the first (smallest document, then
    /// window) is a duplicate. Sorts `sigs`.
    pub fn within(sigs: &mut [Signature]) -> Self {
        sigs.sort_unstable();
        let pairs = sigs
            .windows(2)
            .filter(|w| w[0].hash == w[1].hash)
            .map(|w| (w[1].doc, w[1].sent))
            .collect();
        Self::from_pairs(pairs)
    }

    /// Number of `(doc, window)` pairs.
    pub fn len(&self) -> usize {
        self.sents.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sents.is_empty()
    }

    /// The documents with duplicates, ascending.
    pub fn docs(&self) -> &[u32] {
        &self.docs
    }

    /// The sorted duplicate window indices of document `doc`.
    pub fn of_doc(&self, doc: usize) -> &[u16] {
        let Ok(doc) = u32::try_from(doc) else {
            return &[];
        };
        match self.docs.binary_search(&doc) {
            Ok(k) => &self.sents[self.starts[k]..self.starts[k + 1]],
            Err(_) => &[],
        }
    }

    /// Every `(doc, window)` pair, sorted.
    pub fn pairs(&self) -> impl Iterator<Item = (u32, u16)> + '_ {
        self.docs.iter().enumerate().flat_map(move |(k, &doc)| {
            self.sents[self.starts[k]..self.starts[k + 1]]
                .iter()
                .map(move |&s| (doc, s))
        })
    }
}

/// Fill `buf` from `r`; the number of bytes read (short only at EOF).
fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> io::Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => break,
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(n)
}

/// A heap entry in datatrove's `HashSig` order `(hash_value, doc_id,
/// file_id, sent_id)`: index signatures (`doc_id = sent_id = -1`) have 0
/// there, data ones their value plus one.
type Key = (u64, u64, u32, u32);

/// One sorted signature (or index) file, read sequentially.
struct SigReader<R> {
    r: io::BufReader<R>,
    file: u32,
    index: bool,
    precision: Precision,
    last: Option<u64>,
    buf: [u8; 14],
}

impl<R: Read> SigReader<R> {
    fn new(r: R, file: usize, index: bool, precision: Precision) -> Self {
        Self {
            r: io::BufReader::with_capacity(1 << 16, r),
            file: u32::try_from(file).unwrap_or(u32::MAX),
            index,
            precision,
            last: None,
            buf: [0; 14],
        }
    }

    fn read_key(&mut self) -> Result<Option<Key>, FileError> {
        let hl = match self.precision {
            Precision::P32 => 4,
            Precision::P64 => 8,
        };
        let len = if self.index { hl } else { hl + 6 };
        let file = self.file as usize;
        let n = read_full(&mut self.r, &mut self.buf[..len])
            .map_err(|err| FileError::Io { file, err })?;
        if n == 0 {
            return Ok(None);
        }
        if n < len {
            return Err(FileError::Truncated { file, record: len });
        }
        let b = &self.buf;
        let hash = if hl == 4 {
            u64::from(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        } else {
            u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
        };
        if self.last.is_some_and(|l| hash < l) {
            return Err(FileError::Unsorted { file });
        }
        self.last = Some(hash);
        if self.index {
            return Ok(Some((hash, 0, self.file, 0)));
        }
        let doc = u32::from_le_bytes([b[hl], b[hl + 1], b[hl + 2], b[hl + 3]]);
        let sent = u16::from_le_bytes([b[hl + 4], b[hl + 5]]);
        Ok(Some((
            hash,
            u64::from(doc) + 1,
            self.file,
            u32::from(sent) + 1,
        )))
    }
}

/// datatrove's heap merge: every signature of `readers`, in `HashSig` order
/// (the smallest head first, as `heapq` pops them).
fn merge<R: Read>(mut readers: Vec<SigReader<R>>, mut f: impl FnMut(Key)) -> Result<(), FileError> {
    let mut heap = BinaryHeap::with_capacity(readers.len());
    for r in &mut readers {
        if let Some(k) = r.read_key()? {
            heap.push(Reverse(k));
        }
    }
    while let Some(Reverse(k)) = heap.pop() {
        f(k);
        if let Some(next) = readers[k.2 as usize].read_key()? {
            heap.push(Reverse(next));
        }
    }
    Ok(())
}

/// datatrove's `SentenceFindDedups.run` for one finder task: the `.c4_sig`
/// files of its bucket (in datatrove's file order) and the index files
/// (none: no index folder, or an empty one). Returns, per signature file,
/// the bytes of its `.c4_dup` file (`<IH` records in datatrove's order;
/// empty: datatrove creates no file).
pub fn find_duplicates<R: Read>(
    sigs: Vec<R>,
    index: Vec<R>,
    precision: Precision,
    only_dedup_in_index: bool,
) -> Result<Vec<Vec<u8>>, FileError> {
    let n_sigs = sigs.len();
    let has_index = !index.is_empty();
    let readers = sigs
        .into_iter()
        .map(|r| (r, false))
        .chain(index.into_iter().map(|r| (r, true)))
        .enumerate()
        .map(|(i, (r, idx))| SigReader::new(r, i, idx, precision))
        .collect();
    let mut out = vec![Vec::new(); n_sigs];
    let mut last: Option<Key> = None;
    let from_index = |k: &Key| k.1 == 0;
    merge(readers, |v| {
        if let Some(l) = last {
            // We never match signatures of the index itself; with an index,
            // `only_dedup_in_index` only matches against it.
            if l.0 == v.0
                && !from_index(&v)
                && (from_index(&l) || !has_index || !only_dedup_in_index)
            {
                let o = &mut out[v.2 as usize];
                let doc = u32::try_from(v.1 - 1).unwrap_or(u32::MAX);
                let sent = u16::try_from(v.3 - 1).unwrap_or(u16::MAX);
                o.extend_from_slice(&doc.to_le_bytes());
                o.extend_from_slice(&sent.to_le_bytes());
            }
        }
        if last.is_none_or(|l| l.0 != v.0 || !from_index(&l)) {
            last = Some(v);
        }
    })?;
    Ok(out)
}

/// datatrove's `SentenceDedupBuildIndex.run`: every distinct hash of the
/// `.c4_sig` files, ascending, as `<Q` / `<I` records.
pub fn build_index<R: Read>(sigs: Vec<R>, precision: Precision) -> Result<Vec<u8>, FileError> {
    let readers = sigs
        .into_iter()
        .enumerate()
        .map(|(i, r)| SigReader::new(r, i, false, precision))
        .collect();
    let mut out = Vec::new();
    let mut last = None;
    merge(readers, |v| {
        if last != Some(v.0) {
            push_hash(&mut out, v.0, precision);
        }
        last = Some(v.0);
    })?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deduper(config: SentDedupConfig) -> SentenceDeduper {
        SentenceDeduper::new(config, UnicodeVersion::LATEST, "eng").unwrap()
    }

    fn lines(n_sentences: usize) -> SentenceDeduper {
        deduper(SentDedupConfig {
            n_sentences,
            split_sentences: false,
            min_doc_words: 0,
            min_num_sentences: 0,
            ..SentDedupConfig::default()
        })
    }

    #[test]
    fn splitlines_matches_python() {
        let mut out = Vec::new();
        py_splitlines("a\nb\r\nc\rd\x0be\u{2028}f\n\n", &mut out);
        assert_eq!(out, ["a", "b", "c", "d", "e", "f", ""]);
        out.clear();
        py_splitlines("", &mut out);
        assert_eq!(out, [] as [&str; 0]);
        py_splitlines("\n", &mut out);
        assert_eq!(out, [""]);
    }

    #[test]
    fn sentences_are_stripped_and_spans_are_not() {
        let d = deduper(SentDedupConfig::default());
        let text = "Hello there. World is big.  ";
        let mut s = Vec::new();
        d.sentences(text, &mut s).unwrap();
        assert_eq!(s, ["Hello there.", "World is big."]);
        let mut spans = Vec::new();
        d.spans(text, &mut spans).unwrap();
        // spaCy's trailing whitespace-only sentence is a span, not a sentence.
        assert_eq!(spans, ["Hello there.", " World is big.", "  "]);
    }

    #[test]
    fn window_hashes_skip_blank_windows_but_count_them() {
        let d = lines(2);
        let mut h = Vec::new();
        d.hashes("a\n\n\n\nb", &mut h).unwrap();
        let hc = d.config().hash_config;
        assert_eq!(h, [(hash_bytes(hc, b"a "), 0), (hash_bytes(hc, b" b"), 3)]);
        h.clear();
        d.hashes("only one", &mut h).unwrap();
        assert_eq!(h, [] as [(u64, usize); 0]);
    }

    #[test]
    fn hash_functions_match_python() {
        // struct.unpack("<Q", hashlib.sha1(b"a b c d e").digest()[:8])[0]
        let sha64 = HashConfig {
            precision: Precision::P64,
            hash_fc: HashFn::Sha1,
        };
        assert_eq!(hash_bytes(sha64, b"a b c d e"), 0x10dc_3151_b6a3_9587);
        let xx32 = HashConfig {
            precision: Precision::P32,
            hash_fc: HashFn::Xxhash,
        };
        assert_eq!(hash_bytes(xx32, b""), 0x02cc_5d05);
    }

    #[test]
    fn removal_marks_runs() {
        let d = lines(2);
        let text = "one\ntwo\nthree\nfour";
        let (kept, formatted, spans, k) = d.remove_dup_sentences(text, &[1]).unwrap();
        assert_eq!(kept, "one\nfour");
        assert_eq!(formatted, "one\n>>>\ntwo\nthree\n<<<\nfour");
        assert_eq!((spans, k), (4, 2));
        // A run at the end; overlapping windows extend it.
        let (kept, formatted, _, _) = d.remove_dup_sentences(text, &[2, 3]).unwrap();
        assert_eq!(kept, "one\ntwo");
        assert_eq!(formatted, "one\ntwo\n>>>\nthree\nfour\n<<<");
        assert_eq!(
            d.remove_dup_sentences(text, &[1, 1]),
            Err(SentenceError::DuplicateLineIndex)
        );
    }

    #[test]
    fn min_words_to_remove_span_restores_only_the_first_span() {
        let d = deduper(SentDedupConfig {
            n_sentences: 2,
            split_sentences: false,
            min_words_to_remove_span: 3,
            min_doc_words: 0,
            min_num_sentences: 0,
            ..SentDedupConfig::default()
        });
        let (kept, _, _, _) = d.remove_dup_sentences("a\nb\nc d e f\ng", &[0, 1]).unwrap();
        // The run is "a", "b", "c d e f": only "a" (1 word < 3) comes back.
        assert_eq!(kept, "a\ng");
    }

    #[test]
    fn filter_keep_conditions() {
        let d = lines(1);
        let f = d.filter("", &[]).unwrap();
        assert!(!f.keep, "datatrove drops empty documents");
        let f = d.filter("x\ny", &[]).unwrap();
        assert!(f.keep && matches!(f.text, Cow::Borrowed(_)) && f.formatted.is_none());
        let f = d.filter("x\ny", &[0, 1]).unwrap();
        assert!(!f.keep);
        assert_eq!(f.text, "");
        assert_eq!(f.formatted.as_deref(), Some(">>>\nx\ny\n<<<"));
        let strict = deduper(SentDedupConfig {
            n_sentences: 1,
            split_sentences: false,
            min_doc_words: 3,
            min_num_sentences: 0,
            ..SentDedupConfig::default()
        });
        assert!(!strict.filter("a b\nc d", &[0]).unwrap().keep);
        assert!(strict.filter("a b\nc d e", &[0]).unwrap().keep);
    }

    fn sig(hash: u64, doc: u32, sent: u16) -> Signature {
        Signature { hash, doc, sent }
    }

    fn file(sigs: &[Signature], p: Precision) -> Vec<u8> {
        let mut v = sigs.to_vec();
        signature_files(&mut v, p, 1).remove(0)
    }

    #[test]
    fn signature_files_split_like_searchsorted() {
        let p = Precision::P32;
        let per = u64::from(u32::MAX) / 3;
        let mut sigs = vec![sig(per + 1, 0, 0), sig(per, 1, 2), sig(5, 0, 1)];
        let files = signature_files(&mut sigs, p, 3);
        // Bucket 0 holds hashes <= per; the last file is bucket 1's.
        assert_eq!(files.len(), 2);
        assert_eq!(files[0].len(), 20);
        assert_eq!(files[1].len(), 10);
        assert_eq!(&files[0][..4], &5u32.to_le_bytes());
        assert_eq!(signature_files(&mut [], p, 4), vec![Vec::<u8>::new()]);
    }

    #[test]
    fn find_duplicates_orders_doc_before_file() {
        let p = Precision::P64;
        let f0 = file(&[sig(7, 5, 0), sig(9, 1, 0)], p);
        let f1 = file(&[sig(7, 2, 3), sig(9, 1, 1)], p);
        let out = find_duplicates(vec![&f0[..], &f1[..]], vec![], p, true).unwrap();
        // Hash 7: (doc 2, file 1) comes first, so file 0's doc 5 is the
        // duplicate. Hash 9: doc 1 in both; file 1's is the duplicate.
        assert_eq!(out[0], [5, 0, 0, 0, 0, 0]);
        assert_eq!(out[1], [1, 0, 0, 0, 1, 0]);
        // With an index only its hashes are removed, every copy of them.
        let idx = 7u64.to_le_bytes();
        let out = find_duplicates(vec![&f0[..], &f1[..]], vec![&idx[..]], p, true).unwrap();
        assert_eq!(out[0], [5, 0, 0, 0, 0, 0]);
        assert_eq!(out[1], [2, 0, 0, 0, 3, 0]);
        let out = find_duplicates(vec![&f0[..], &f1[..]], vec![&idx[..]], p, false).unwrap();
        assert_eq!(out[1].len(), 12);
        assert_eq!(
            build_index(vec![&f0[..], &f1[..]], p).unwrap(),
            [7u64.to_le_bytes(), 9u64.to_le_bytes()].concat()
        );
    }

    #[test]
    fn malformed_files_are_errors() {
        let p = Precision::P64;
        let f = file(&[sig(1, 0, 0)], p);
        assert!(matches!(
            find_duplicates(vec![&f[..13]], vec![], p, true),
            Err(FileError::Truncated {
                file: 0,
                record: 14
            })
        ));
        let mut bad = file(&[sig(2, 0, 0)], p);
        bad.extend(file(&[sig(1, 0, 0)], p));
        assert!(matches!(
            find_duplicates(vec![&bad[..]], vec![], p, true),
            Err(FileError::Unsorted { file: 0 })
        ));
    }

    #[test]
    fn duplicates_within_one_task() {
        let mut sigs = vec![sig(3, 2, 0), sig(3, 0, 4), sig(1, 1, 0), sig(3, 0, 1)];
        let d = Duplicates::within(&mut sigs);
        assert_eq!(d.pairs().collect::<Vec<_>>(), [(0, 4), (2, 0)]);
        assert_eq!(d.of_doc(0), [4]);
        assert_eq!(d.of_doc(1), [] as [u16; 0]);
        let parsed = Duplicates::parse(&[&[2, 0, 0, 0, 0, 0], &[0, 0, 0, 0, 4, 0]]).unwrap();
        assert_eq!(parsed, d);
    }

    #[test]
    fn window_index_overflow_is_numpys_error() {
        let mut out = Vec::new();
        assert_eq!(
            records(0, &[(1, 65_535), (2, 65_536)], &mut out),
            Err(SentenceError::SentenceOverflow(65_536))
        );
        assert_eq!(out, [] as [Signature; 0]);
        records(4, &[(1, 65_535)], &mut out).unwrap();
        assert_eq!(out, [sig(1, 4, 65_535)]);
    }
}
