// Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi; modified.
//! Chinese word segmentation with the same output as jieba 0.42.1
//! (`jieba.cut`, `cut_for_search`, `tokenize`, user dictionaries).
//!
//! The pieces are jieba's: the text is split into blocks with
//! `re_han_default` and `re_skip_default`; each block's words come from the
//! prefix dictionary's DAG and its maximum-probability route (`calc`, with
//! the same floating-point operations and tie-breaking), and runs of single
//! characters left by the route go to the BMES HMM (`finalseg`), unless
//! `HMM=False`. Full mode (`cut_all=True`) lists every dictionary word.
//!
//! - [`Dictionary`]: `Tokenizer.FREQ` and `Tokenizer.total`; the default one
//!   is jieba's `dict.txt`, stored as a prebuilt character trie and read in
//!   about 2 ms.
//! - [`zh`]: datatrove's Chinese word tokenizer (spaCy's `ChineseTokenizer`
//!   with the jieba segmenter, plus the sentencizer), for the filters and
//!   MinHash.
//!
//! Words are slices of the input text, except in full mode, where jieba glues
//! some ASCII runs together.

use std::borrow::Cow;
use std::cell::RefCell;

use rustc_hash::FxHashSet;

mod dict;
mod hmm;
mod trie;
pub mod zh;

pub use dict::{DEFAULT_FREQ_LEN, DEFAULT_TOTAL, DEFAULT_WORDS, DictError, Dictionary};

/// `finalseg.Force_Split_Words`: words the HMM must not produce whole
/// (`add_word(word, 0)` and `del_word` add to it). jieba keeps one set for
/// the whole process, shared by every `Tokenizer`.
pub type ForceSplit = FxHashSet<Box<str>>;

/// Why a cut failed: what jieba raises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutError {
    /// Python `ValueError: math domain error`: `calc` took the log of a
    /// non-positive number (a total of 0, or a negative frequency on a
    /// route).
    MathDomain,
}

impl std::fmt::Display for CutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CutError::MathDomain => f.write_str("math domain error"),
        }
    }
}

impl std::error::Error for CutError {}

/// `re_han_default`'s class: `[一-鿕a-zA-Z0-9+#&\._%\-]`.
#[inline]
pub(crate) fn is_han_default(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(c, '+' | '#' | '&' | '.' | '_' | '%' | '-')
        || hmm::is_re_han(c)
}

/// `re_eng.match(s)`: `s` starts with an ASCII letter or digit.
#[inline]
fn starts_eng(s: &str) -> bool {
    s.as_bytes().first().is_some_and(u8::is_ascii_alphanumeric)
}

/// Reused per-thread buffers.
#[derive(Debug, Default)]
struct Scratch {
    /// The block's characters.
    chars: Vec<char>,
    /// Byte offset of each character of the block, and its length.
    offs: Vec<u32>,
    /// The DAG: `(end char index, frequency)` per start, flattened.
    dag: Vec<(u32, i64)>,
    /// Start of each character's entries in `dag` (one more than chars).
    dag_start: Vec<u32>,
    /// `route[idx] = (log probability, end)`.
    route: Vec<(f64, u32)>,
    hmm: hmm::Scratch,
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::default());
}

/// Run `f` with this thread's scratch buffers (fresh ones if re-entered).
fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    let mut f = Some(f);
    let r = SCRATCH.with(|s| {
        s.try_borrow_mut()
            .ok()
            .map(|mut s| (f.take().expect("once"))(&mut s))
    });
    match r {
        Some(r) => r,
        None => (f.take().expect("once"))(&mut Scratch::default()),
    }
}

/// `re_han_default.split(text)` without the empty strings: `(is_word_block,
/// block)`, where a word block is a run of `re_han_default` characters.
fn blocks(text: &str) -> impl Iterator<Item = (bool, &str)> {
    let mut rest = text;
    std::iter::from_fn(move || {
        let first = rest.chars().next()?;
        let is_han = is_han_default(first);
        let end = rest
            .char_indices()
            .find(|&(_, c)| is_han_default(c) != is_han)
            .map_or(rest.len(), |(i, _)| i);
        let (blk, tail) = rest.split_at(end);
        rest = tail;
        Some((is_han, blk))
    })
}

/// `re_skip_default.split(blk)` on a run of non-`re_han_default` characters:
/// calls `f(x, is_separator)` for each separator (`\r\n` or one whitespace
/// character) and for the (possibly empty) text around them, in order.
fn split_skip<'a>(blk: &'a str, mut f: impl FnMut(&'a str, bool)) {
    let mut last = 0;
    let mut it = blk.char_indices().peekable();
    while let Some((i, c)) = it.next() {
        let end = if c == '\r' && it.peek().is_some_and(|&(_, d)| d == '\n') {
            it.next();
            i + 2
        } else if crate::text::is_py_whitespace(c) {
            i + c.len_utf8()
        } else {
            continue;
        };
        f(&blk[last..i], false);
        f(&blk[i..end], true);
        last = end;
    }
    f(&blk[last..], false);
}

/// `x` one character at a time.
fn chars_of(x: &str) -> impl Iterator<Item = &str> {
    x.char_indices().map(move |(i, c)| &x[i..i + c.len_utf8()])
}

/// `log(n)` as `math.log` computes it for a Python int, or the
/// `ValueError` it raises for `n <= 0`.
#[inline]
#[allow(clippy::cast_precision_loss)] // `math.log` rounds the int to a double the same way
fn log_int(n: i64) -> Result<f64, CutError> {
    if n > 0 {
        Ok((n as f64).ln())
    } else {
        Err(CutError::MathDomain)
    }
}

impl Dictionary {
    /// Fill `s.chars`, `s.offs`, `s.dag` and `s.dag_start` for `blk` (`get_DAG`).
    fn build_dag(&self, blk: &str, s: &mut Scratch) {
        assert!(
            u32::try_from(blk.len()).is_ok(),
            "a run of 4 GiB without punctuation or spaces"
        );
        s.chars.clear();
        s.offs.clear();
        for (i, c) in blk.char_indices() {
            s.chars.push(c);
            #[allow(clippy::cast_possible_truncation)] // checked above
            s.offs.push(i as u32);
        }
        #[allow(clippy::cast_possible_truncation)]
        s.offs.push(blk.len() as u32);
        let n = s.chars.len();
        s.dag.clear();
        s.dag_start.clear();
        for k in 0..n {
            #[allow(clippy::cast_possible_truncation)]
            s.dag_start.push(s.dag.len() as u32);
            self.dag_from(blk, &s.chars, &s.offs, k, &mut s.dag);
        }
        #[allow(clippy::cast_possible_truncation)]
        s.dag_start.push(s.dag.len() as u32);
    }

    #[inline]
    fn dag_at(s: &Scratch, k: usize) -> &[(u32, i64)] {
        &s.dag[s.dag_start[k] as usize..s.dag_start[k + 1] as usize]
    }

    /// `calc(sentence, DAG, route)`: the maximum-probability route, from the
    /// end, as jieba computes it (`max` over `(logprob, end)` tuples: a tie
    /// goes to the longer word).
    fn calc(&self, s: &mut Scratch) -> Result<(), CutError> {
        let n = s.offs.len() - 1;
        s.route.clear();
        s.route.resize(n + 1, (0.0, 0));
        let logtotal = log_int(self.total())?;
        for idx in (0..n).rev() {
            let mut best: Option<(f64, u32)> = None;
            for &(x, f) in &s.dag[s.dag_start[idx] as usize..s.dag_start[idx + 1] as usize] {
                let p = log_int(f)? - logtotal + s.route[x as usize + 1].0;
                best = Some(match best {
                    #[allow(clippy::float_cmp)] // Python's tuple order
                    Some(b) if !(p > b.0 || (p == b.0 && x > b.1)) => b,
                    _ => (p, x),
                });
            }
            s.route[idx] = best.expect("a DAG entry per character");
        }
        Ok(())
    }

    /// `__cut_DAG(blk)` (`HMM=True`).
    fn cut_dag<'a>(
        &self,
        blk: &'a str,
        fs: &ForceSplit,
        s: &mut Scratch,
        out: &mut Vec<&'a str>,
    ) -> Result<(), CutError> {
        self.build_dag(blk, s);
        self.calc(s)?;
        let n = s.offs.len() - 1;
        let at = |s: &Scratch, i: usize| s.offs[i] as usize;
        // buf: the pending single-character words, chars [buf, x).
        let mut buf: Option<usize> = None;
        let mut x = 0;
        while x < n {
            let y = s.route[x].1 as usize + 1;
            if y - x == 1 {
                buf.get_or_insert(x);
            } else {
                if let Some(b) = buf.take() {
                    self.flush_buf(&blk[at(s, b)..at(s, x)], x - b, fs, s, out);
                }
                out.push(&blk[at(s, x)..at(s, y)]);
            }
            x = y;
        }
        if let Some(b) = buf {
            self.flush_buf(&blk[at(s, b)..], n - b, fs, s, out);
        }
        Ok(())
    }

    /// `__cut_DAG`'s handling of `buf` (`chars` characters): as is when it is
    /// one character, character by character when it is a dictionary word,
    /// else through the HMM.
    fn flush_buf<'a>(
        &self,
        buf: &'a str,
        chars: usize,
        fs: &ForceSplit,
        s: &mut Scratch,
        out: &mut Vec<&'a str>,
    ) {
        if chars == 1 {
            out.push(buf);
        } else if self.get(buf).is_none_or(|f| f == 0) {
            hmm::cut(buf, fs, &mut s.hmm, out);
        } else {
            out.extend(chars_of(buf));
        }
    }

    /// `__cut_DAG_NO_HMM(blk)`.
    fn cut_dag_no_hmm<'a>(
        &self,
        blk: &'a str,
        s: &mut Scratch,
        out: &mut Vec<&'a str>,
    ) -> Result<(), CutError> {
        self.build_dag(blk, s);
        self.calc(s)?;
        let n = s.offs.len() - 1;
        let at = |s: &Scratch, i: usize| s.offs[i] as usize;
        let mut buf: Option<usize> = None;
        let mut x = 0;
        while x < n {
            let y = s.route[x].1 as usize + 1;
            let word = &blk[at(s, x)..at(s, y)];
            if y - x == 1 && starts_eng(word) {
                buf.get_or_insert(x);
            } else {
                if let Some(b) = buf.take() {
                    out.push(&blk[at(s, b)..at(s, x)]);
                }
                out.push(word);
            }
            x = y;
        }
        if let Some(b) = buf {
            out.push(&blk[at(s, b)..]);
        }
        Ok(())
    }

    /// `__cut_all(blk)`.
    fn cut_all_block<'a>(&self, blk: &'a str, s: &mut Scratch, out: &mut Vec<Cow<'a, str>>) {
        self.build_dag(blk, s);
        let n = s.offs.len() - 1;
        let at = |i: usize| s.offs[i] as usize;
        let mut old_j: isize = -1;
        // eng_buf: pieces of `blk` (jieba concatenates them; they are
        // usually adjacent).
        let mut eng: Option<Vec<(usize, usize)>> = None;
        let flush = |eng: Vec<(usize, usize)>, out: &mut Vec<Cow<'a, str>>| {
            out.push(join_pieces(blk, &eng));
        };
        #[allow(clippy::cast_possible_wrap)] // char indices < 2**31
        for k in 0..n {
            let l = Self::dag_at(s, k);
            if eng.is_some() && !starts_eng(&blk[at(k)..]) {
                flush(eng.take().expect("eng"), out);
            }
            if l.len() == 1 && k as isize > old_j {
                let j = l[0].0 as usize;
                let (ws, we) = (at(k), at(j + 1));
                if starts_eng(&blk[ws..we]) {
                    eng.get_or_insert_with(Vec::new).push((ws, we));
                }
                if eng.is_none() {
                    out.push(Cow::Borrowed(&blk[ws..we]));
                }
                old_j = j as isize;
            } else {
                for &(j, _) in l {
                    let j = j as usize;
                    if j > k {
                        out.push(Cow::Borrowed(&blk[at(k)..at(j + 1)]));
                        old_j = j as isize;
                    }
                }
            }
        }
        if let Some(e) = eng {
            flush(e, out);
        }
    }

    /// `jieba.cut(text, cut_all=False, HMM=hmm)`, appended to `out`.
    ///
    /// `fs` is `finalseg.Force_Split_Words` (only the HMM reads it).
    pub fn cut<'a>(
        &self,
        text: &'a str,
        hmm: bool,
        fs: &ForceSplit,
        out: &mut Vec<&'a str>,
    ) -> Result<(), CutError> {
        with_scratch(|s| {
            for (is_han, blk) in blocks(text) {
                if !is_han {
                    split_skip(blk, |x, sep| {
                        if sep {
                            out.push(x);
                        } else {
                            out.extend(chars_of(x));
                        }
                    });
                } else if hmm {
                    self.cut_dag(blk, fs, s, out)?;
                } else {
                    self.cut_dag_no_hmm(blk, s, out)?;
                }
            }
            Ok(())
        })
    }

    /// `jieba.cut(text, cut_all=True)`, appended to `out`, empty strings
    /// included (jieba yields one on each side of every separator it meets
    /// outside the word blocks).
    pub fn cut_all<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) {
        with_scratch(|s| {
            for (is_han, blk) in blocks(text) {
                if is_han {
                    self.cut_all_block(blk, s, out);
                } else {
                    split_skip(blk, |x, _| out.push(Cow::Borrowed(x)));
                }
            }
        });
    }

    /// `jieba.cut(text, cut_all, HMM)` in any mode.
    pub fn cut_any<'a>(
        &self,
        text: &'a str,
        cut_all: bool,
        hmm: bool,
        fs: &ForceSplit,
        out: &mut Vec<Cow<'a, str>>,
    ) -> Result<(), CutError> {
        if cut_all {
            self.cut_all(text, out);
            return Ok(());
        }
        let mut words = Vec::new();
        self.cut(text, hmm, fs, &mut words)?;
        out.extend(words.into_iter().map(Cow::Borrowed));
        Ok(())
    }

    /// The 2- and 3-character dictionary words inside `w` that
    /// `cut_for_search` adds before it: `(char offset, gram)`.
    fn search_grams<'a>(&self, w: &'a str, mut f: impl FnMut(usize, &'a str)) {
        let starts: Vec<usize> = w.char_indices().map(|(i, _)| i).chain([w.len()]).collect();
        let n = starts.len() - 1;
        for (size, min_len) in [(2usize, 2usize), (3, 3)] {
            if n > min_len {
                for i in 0..=n - size {
                    let g = &w[starts[i]..starts[i + size]];
                    if self.get(g).is_some_and(|f| f != 0) {
                        f(i, g);
                    }
                }
            }
        }
    }

    /// `jieba.cut_for_search(text, HMM=hmm)`, appended to `out`.
    pub fn cut_for_search<'a>(
        &self,
        text: &'a str,
        hmm: bool,
        fs: &ForceSplit,
        out: &mut Vec<&'a str>,
    ) -> Result<(), CutError> {
        let mut words = Vec::new();
        self.cut(text, hmm, fs, &mut words)?;
        for w in words {
            self.search_grams(w, |_, g| out.push(g));
            out.push(w);
        }
        Ok(())
    }

    /// `jieba.tokenize(text, mode, HMM=hmm)`: `(word, start, end)` with
    /// character offsets, appended to `out`; `search` is `mode="search"`.
    pub fn tokenize<'a>(
        &self,
        text: &'a str,
        search: bool,
        hmm: bool,
        fs: &ForceSplit,
        out: &mut Vec<(&'a str, usize, usize)>,
    ) -> Result<(), CutError> {
        let mut words = Vec::new();
        self.cut(text, hmm, fs, &mut words)?;
        let mut start = 0;
        for w in words {
            let width = w.chars().count();
            if search {
                self.search_grams(w, |i, g| {
                    let len = if g.chars().nth(2).is_some() { 3 } else { 2 };
                    out.push((g, start + i, start + i + len));
                });
            }
            out.push((w, start, start + width));
            start += width;
        }
        Ok(())
    }

    /// `get_DAG(sentence)`: for each character index, the end indices of
    /// the dictionary words that start there (or the index itself).
    pub fn dag(&self, sentence: &str) -> Vec<Vec<usize>> {
        with_scratch(|s| {
            self.build_dag(sentence, s);
            (0..s.offs.len() - 1)
                .map(|k| {
                    Self::dag_at(s, k)
                        .iter()
                        .map(|&(j, _)| j as usize)
                        .collect()
                })
                .collect()
        })
    }

    /// `suggest_freq(word, tune=False)` for a `str` segment: the frequency
    /// that makes `word` cut whole (`HMM=False`).
    pub fn suggest_freq(&self, word: &str) -> Result<i64, SuggestError> {
        if self.total() == 0 {
            // jieba cuts lazily, so the first block decides which error comes
            // first: `calc`'s log(0) for a word block, else the division.
            if let Some(c) = word.chars().next() {
                return Err(if is_han_default(c) {
                    SuggestError::Cut(CutError::MathDomain)
                } else {
                    SuggestError::ZeroDivision
                });
            }
        }
        let mut segs = Vec::new();
        self.cut(word, false, &ForceSplit::default(), &mut segs)
            .map_err(SuggestError::Cut)?;
        let base = if segs.is_empty() {
            // `freq` stays the int 1: int(1 * total) + 1, exactly.
            self.total().checked_add(1).ok_or(SuggestError::Overflow)?
        } else {
            let ftotal = self.total_f64();
            let mut freq = 1.0f64;
            for seg in segs {
                freq *= self.get_or_f64(seg, 1) / ftotal;
            }
            py_int(freq * ftotal)?
                .checked_add(1)
                .ok_or(SuggestError::Overflow)?
        };
        Ok(base.max(self.get(word).unwrap_or(1)))
    }

    /// `suggest_freq(segments, tune=False)` for a tuple of segments: the
    /// frequency that makes `''.join(segments)` cut into them.
    pub fn suggest_freq_split(&self, segments: &[&str]) -> Result<i64, SuggestError> {
        let word: String = segments.concat();
        let base = if segments.is_empty() {
            self.total()
        } else {
            let ftotal = self.total_f64();
            let mut freq = 1.0f64;
            for seg in segments {
                if ftotal == 0.0 {
                    return Err(SuggestError::ZeroDivision);
                }
                freq *= self.get_or_f64(seg, 1) / ftotal;
            }
            py_int(freq * ftotal)?
        };
        Ok(base.min(self.get(&word).unwrap_or(0)))
    }

    #[allow(clippy::cast_precision_loss)] // float(self.total), as jieba
    fn total_f64(&self) -> f64 {
        self.total() as f64
    }

    #[allow(clippy::cast_precision_loss)]
    fn get_or_f64(&self, w: &str, default: i64) -> f64 {
        self.get(w).unwrap_or(default) as f64
    }
}

/// Concatenate byte ranges of `blk` (borrowed when they are adjacent).
fn join_pieces<'a>(blk: &'a str, pieces: &[(usize, usize)]) -> Cow<'a, str> {
    let adjacent = pieces.windows(2).all(|w| w[0].1 == w[1].0);
    match (pieces.first(), pieces.last()) {
        (Some(&(s, _)), Some(&(_, e))) if adjacent => Cow::Borrowed(&blk[s..e]),
        _ => Cow::Owned(pieces.iter().map(|&(s, e)| &blk[s..e]).collect()),
    }
}

/// Why `suggest_freq` failed: what jieba raises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SuggestError {
    /// From the cut (`ValueError: math domain error`).
    Cut(CutError),
    /// `ZeroDivisionError: float division by zero` (a total of 0).
    ZeroDivision,
    /// The result does not fit in an `i64` (Python would return a larger
    /// int; `OverflowError` for an infinite float).
    Overflow,
}

impl std::fmt::Display for SuggestError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SuggestError::Cut(e) => e.fmt(f),
            SuggestError::ZeroDivision => f.write_str("float division by zero"),
            SuggestError::Overflow => f.write_str("frequency out of range"),
        }
    }
}

impl std::error::Error for SuggestError {}

/// Python's `int(x)` for a float that fits in an `i64`.
#[allow(clippy::cast_possible_truncation)] // checked range
fn py_int(x: f64) -> Result<i64, SuggestError> {
    let t = x.trunc();
    // -2**63 <= t < 2**63
    if (-9_223_372_036_854_775_808.0..9_223_372_036_854_775_808.0).contains(&t) {
        Ok(t as i64)
    } else {
        Err(SuggestError::Overflow)
    }
}

#[cfg(test)]
mod replay;
#[cfg(test)]
mod tests;
