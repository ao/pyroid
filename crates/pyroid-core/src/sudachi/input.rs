// Derived from sudachi.rs 0.7.0 (Apache-2.0), © Works Applications Co., Ltd.;
// modified.
//! The input buffer (`InputBuffer`) and the three input-text plugins of
//! SudachiPy's default configuration: `DefaultInputTextPlugin`,
//! `ProlongedSoundMarkPlugin` and `IgnoreYomiganaPlugin`.
//!
//! The analysis runs on the rewritten ("modified") text; `m2o` maps each of
//! its bytes back to a byte of the original, so morphemes report original
//! offsets.

use std::borrow::Cow;
use std::ops::Range;

use rustc_hash::{FxHashMap, FxHashSet};
use unicode_normalization::{IsNormalized, UnicodeNormalization, is_nfkc_quick};

use super::chars::{CharCategory, REWRITE_DEF, cat};
use super::dic::DicError;
use crate::ftfy::UnicodeVersion;
use crate::ftfy::chardata::is_assigned;

/// `MAX_LENGTH`: the longest input sudachi.rs accepts, in bytes.
pub(crate) const MAX_LENGTH: usize = u16::MAX as usize / 4 * 3;
/// `REALLY_MAX_LENGTH`: the longest rewritten text, in bytes.
pub(crate) const REALLY_MAX_LENGTH: usize = u16::MAX as usize;
const REALLY_MAX_LENGTH_ISIZE: isize = 65_535;

/// What tokenizing a text can fail with (`SudachiError`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizeError {
    /// `InputTooLong(len, max)`: the text (or its rewritten form) is too long.
    InputTooLong(usize, usize),
    /// `EosBosDisconnect`: the lattice has no path (never with the default
    /// plugins).
    EosBosDisconnect,
    /// A state in which sudachi.rs panics or reads out of bounds (only seen
    /// with inconsistent dictionary data).
    Internal,
}

impl std::fmt::Display for TokenizeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TokenizeError::InputTooLong(n, max) => write!(
                f,
                "Error during tokenization: Input is too long, it can't be more than {max} bytes, was {n}"
            ),
            TokenizeError::EosBosDisconnect => {
                f.write_str("Error during tokenization: EOS isn't connected to BOS")
            }
            TokenizeError::Internal => {
                f.write_str("Error during tokenization: invalid dictionary data")
            }
        }
    }
}

impl std::error::Error for TokenizeError {}

/// What a byte range of the current text is replaced with.
pub(crate) enum Repl<'d> {
    Str(Cow<'d, str>),
    Char(char),
}

/// `InputBuffer`.
#[derive(Default)]
pub(crate) struct Input {
    pub original: String,
    pub modified: String,
    modified_2: String,
    /// Modified byte -> original byte (length `modified.len() + 1`).
    pub m2o: Vec<usize>,
    m2o_2: Vec<usize>,
    pub chars: Vec<char>,
    /// Modified char -> modified byte (with a sentinel).
    pub c2b: Vec<usize>,
    /// Modified byte -> modified char (with a sentinel).
    pub b2c: Vec<usize>,
    pub bow: Vec<bool>,
    pub cat: Vec<u32>,
    pub cont: Vec<usize>,
}

impl Input {
    /// `reset` + `push_str` + `start_build`.
    pub fn start(&mut self, text: &str) -> Result<(), TokenizeError> {
        self.original.clear();
        self.original.push_str(text);
        self.modified.clear();
        self.m2o.clear();
        self.chars.clear();
        self.c2b.clear();
        self.b2c.clear();
        self.bow.clear();
        self.cat.clear();
        self.cont.clear();
        if text.len() > MAX_LENGTH {
            return Err(TokenizeError::InputTooLong(text.len(), MAX_LENGTH));
        }
        self.modified.push_str(text);
        self.m2o.extend(0..=text.len());
        Ok(())
    }

    /// `with_editor` + `commit`: apply `edits` (sorted, non-overlapping).
    pub fn commit(
        &mut self,
        edits: &mut Vec<(Range<usize>, Repl<'_>)>,
    ) -> Result<(), TokenizeError> {
        if edits.is_empty() {
            return Ok(());
        }
        self.chars.clear();
        self.modified_2.clear();
        self.m2o_2.clear();
        let sz = resolve_edits(
            &self.modified,
            &self.m2o,
            &mut self.modified_2,
            &mut self.m2o_2,
            edits,
        );
        if sz > REALLY_MAX_LENGTH {
            return Err(TokenizeError::InputTooLong(sz, REALLY_MAX_LENGTH));
        }
        std::mem::swap(&mut self.modified, &mut self.modified_2);
        std::mem::swap(&mut self.m2o, &mut self.m2o_2);
        Ok(())
    }

    /// `build`: characters, categories and word-start flags of the modified
    /// text.
    pub fn build(&mut self, cats: &CharCategory) {
        self.chars.clear();
        let non_starting = cat::ALPHA | cat::GREEK | cat::CYRILLIC;
        let mut prev_cat = 0u32;
        self.bow.clear();
        self.bow.resize(self.modified.len(), false);
        let mut last_offset = 0;
        let mut last_chidx = 0;
        for (chidx, (bidx, ch)) in self.modified.char_indices().enumerate() {
            self.chars.push(ch);
            let c = cats.get(ch);
            self.cat.push(c);
            self.c2b.push(bidx);
            self.b2c
                .extend(std::iter::repeat_n(last_chidx, bidx - last_offset));
            last_offset = bidx;
            last_chidx = chidx;
            let can_bow = if c & non_starting != 0 {
                c & prev_cat == 0
            } else {
                true
            };
            self.bow[bidx] = can_bow;
            prev_cat = c;
        }
        self.b2c.extend(std::iter::repeat_n(
            last_chidx,
            self.modified.len() - last_offset,
        ));
        self.c2b.push(self.b2c.len());
        self.b2c.push(last_chidx + 1);
        self.fill_cat_continuity();
    }

    fn fill_cat_continuity(&mut self) {
        if self.chars.is_empty() {
            return;
        }
        let n = self.cat.len();
        let mut length = 1;
        for start in 0..n {
            // Inside a run, count down instead of rescanning (as sudachi.rs).
            if length > 1 {
                length -= 1;
                self.cont.push(length);
                continue;
            }
            let mut common = self.cat[start];
            length = 1;
            while start + length < n {
                common &= self.cat[start + length];
                if common == 0 {
                    break;
                }
                length += 1;
            }
            self.cont.push(length);
        }
    }

    /// `cat_of_range`: the categories every character of `range` has.
    #[inline]
    pub fn cat_of_range(&self, range: Range<usize>) -> u32 {
        if range.is_empty() {
            return 0;
        }
        self.cat[range].iter().fold(cat::ALL, |a, &b| a & b)
    }

    /// `can_oov_bow`.
    #[inline]
    pub fn can_oov_bow(&self, offset: usize) -> bool {
        self.cat[offset] & cat::NOOOVBOW == 0
            && (offset == 0 || self.cat[offset - 1] & cat::NOOOVEOW == 0)
    }

    /// `get_word_candidate_length` (the simple OOV plugin).
    pub fn word_candidate_length(&self, char_idx: usize) -> usize {
        let n = self.chars.len();
        for i in char_idx + 1..n {
            if self.bow[self.c2b[i]] {
                return i - char_idx;
            }
        }
        n - char_idx
    }
}

fn resolve_edits(
    source: &str,
    source_mapping: &[usize],
    target: &mut String,
    target_mapping: &mut Vec<usize>,
    edits: &mut Vec<(Range<usize>, Repl<'_>)>,
) -> usize {
    let mut start = 0usize;
    #[allow(clippy::cast_possible_wrap)] // lengths are below 2**16 here
    let mut cur_len = source.len() as isize;
    let mut buf = [0u8; 4];
    for (what, with) in edits.drain(..) {
        target.push_str(&source[start..what.start]);
        target_mapping.extend_from_slice(&source_mapping[start..what.start]);
        start = what.end;
        let with: &str = match &with {
            Repl::Str(s) => s,
            Repl::Char(c) => c.encode_utf8(&mut buf),
        };
        cur_len += add_replace(source_mapping, target, target_mapping, what, with);
        if cur_len > REALLY_MAX_LENGTH_ISIZE {
            return cur_len.unsigned_abs();
        }
    }
    target.push_str(&source[start..]);
    target_mapping.extend_from_slice(&source_mapping[start..]);
    if let Some(v) = target_mapping.first_mut() {
        *v = 0;
    }
    // Never negative: edits replace parts of the source.
    cur_len.unsigned_abs()
}

fn add_replace(
    source_mapping: &[usize],
    target: &mut String,
    target_mapping: &mut Vec<usize>,
    what: Range<usize>,
    with: &str,
) -> isize {
    #[allow(clippy::cast_possible_wrap)] // lengths are below 2**16 here
    let (w, n) = (what.len() as isize, with.len() as isize);
    if with.is_empty() {
        return -w;
    }
    target.push_str(with);
    target_mapping.push(source_mapping[what.start]);
    let pos = source_mapping[what.end];
    target_mapping.extend(std::iter::repeat_n(pos, with.len() - 1));
    n - w
}

/// Whether `c` is assigned in Unicode 17.0, the version of the
/// `unicode-normalization` tables (0.1.25) and of the Rust standard library
/// SudachiPy 0.7.0 is built with. Characters assigned later are left alone.
#[inline]
fn assigned17(c: char) -> bool {
    u32::from(c) < 0x80 || is_assigned(u32::from(c), UnicodeVersion::V17_0)
}

/// `char::is_uppercase`, as of Unicode 17.0.
#[inline]
fn is_upper(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_uppercase();
    }
    c.is_uppercase() && assigned17(c)
}

/// `is_nfkc_quick(once(c)) != Yes`, as of Unicode 17.0.
#[inline]
fn needs_nfkc(c: char) -> bool {
    !c.is_ascii() && assigned17(c) && is_nfkc_quick(std::iter::once(c)) != IsNormalized::Yes
}

/// `char::to_lowercase`, as of Unicode 17.0 (a mapping to a later character
/// is no mapping).
fn lowercase(c: char) -> impl Iterator<Item = char> + Clone {
    let lower = c.to_lowercase();
    let keep = lower.clone().all(assigned17);
    let (a, b) = if keep {
        (Some(lower), None)
    } else {
        (None, Some(c))
    };
    a.into_iter().flatten().chain(b)
}

/// Replacement keys by first character: `(key chars, replacement)`, shortest
/// key first.
type Keys = FxHashMap<char, Vec<(Box<[char]>, Box<str>)>>;

/// `DefaultInputTextPlugin` with SudachiPy's `rewrite.def`.
#[derive(Debug)]
pub(crate) struct DefaultInputText {
    ignore: FxHashSet<char>,
    keys: Keys,
}

impl DefaultInputText {
    pub fn new() -> Result<Self, DicError> {
        let mut ignore = FxHashSet::default();
        let mut keys: Keys = FxHashMap::default();
        let mut seen: FxHashSet<&str> = FxHashSet::default();
        for (i, line) in REWRITE_DEF.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            let err = || DicError::Format(format!("invalid rewrite.def at line {i}"));
            match cols.len() {
                1 => {
                    let mut it = cols[0].chars();
                    let (Some(c), None) = (it.next(), it.next()) else {
                        return Err(err());
                    };
                    ignore.insert(c);
                }
                2 => {
                    if !seen.insert(cols[0]) {
                        return Err(err());
                    }
                    let key: Box<[char]> = cols[0].chars().collect();
                    keys.entry(key[0]).or_default().push((key, cols[1].into()));
                }
                _ => return Err(err()),
            }
        }
        for v in keys.values_mut() {
            v.sort_by_key(|(k, _)| k.len());
        }
        Ok(Self { ignore, keys })
    }

    /// The replacement key matching at `chars[i..]`: the shortest (the
    /// anchored "earliest" search of the slow path) or the longest (the
    /// leftmost-longest search of the fast path).
    #[inline]
    fn key_at(&self, chars: &[char], i: usize, longest: bool) -> Option<(usize, &str)> {
        let cands = self.keys.get(&chars[i])?;
        let ok = |k: &[char]| chars.get(i..i + k.len()) == Some(k);
        let found = if longest {
            cands.iter().rev().find(|(k, _)| ok(k))
        } else {
            cands.iter().find(|(k, _)| ok(k))
        };
        found.map(|(k, r)| (k.len(), &**r))
    }

    /// `rewrite`.
    pub fn edits<'d>(&'d self, input: &Input, out: &mut Vec<(Range<usize>, Repl<'d>)>) {
        let chars: Vec<char> = input.modified.chars().collect();
        let need_nfkc = is_nfkc_quick(chars.iter().map(|&c| if assigned17(c) { c } else { 'a' }))
            != IsNormalized::Yes;
        let need_lowercase = chars.iter().any(|&c| is_upper(c));
        let slow = need_nfkc || need_lowercase;
        let mut offset = 0usize;
        let mut i = 0usize;
        while i < chars.len() {
            let ch = chars[i];
            if let Some((n, repl)) = self.key_at(&chars, i, !slow) {
                let len: usize = chars[i..i + n].iter().map(|c| c.len_utf8()).sum();
                out.push((offset..offset + len, Repl::Str(Cow::Borrowed(repl))));
                offset += len;
                i += n;
                continue;
            }
            if slow {
                let lower = is_upper(ch);
                let nfkc = !self.ignore.contains(&ch) && needs_nfkc(ch);
                match (lower, nfkc) {
                    (false, false) => {}
                    (true, false) => push_normalized(lowercase(ch), out, offset, ch),
                    (false, true) => push_normalized(std::iter::once(ch).nfkc(), out, offset, ch),
                    (true, true) => push_normalized(lowercase(ch).nfkc(), out, offset, ch),
                }
            }
            offset += ch.len_utf8();
            i += 1;
        }
    }
}

/// `handle_normalization_slow` + `replace_char_iter`.
fn push_normalized(
    mut data: impl Iterator<Item = char>,
    out: &mut Vec<(Range<usize>, Repl<'_>)>,
    start: usize,
    ch: char,
) {
    let Some(ch2) = data.next() else {
        return;
    };
    if ch2 == ch {
        return;
    }
    let range = start..start + ch.len_utf8();
    match data.next() {
        None => out.push((range, Repl::Char(ch2))),
        Some(ch3) => {
            let mut s = String::with_capacity(12);
            s.push(ch2);
            s.push(ch3);
            s.extend(data);
            out.push((range, Repl::Str(Cow::Owned(s))));
        }
    }
}

/// `ProlongedSoundMarkPlugin` with SudachiPy's settings: every run of two or
/// more of `ー - ⁓ 〜 〰` becomes one `ー`.
pub(crate) fn prolonged_sound_marks(input: &Input, out: &mut Vec<(Range<usize>, Repl<'_>)>) {
    let is_psm = |c: char| matches!(c, 'ー' | '-' | '⁓' | '〜' | '〰');
    let mut run: Option<(usize, usize, usize)> = None; // (start, end, count)
    for (b, c) in input.modified.char_indices() {
        if is_psm(c) {
            run = match run {
                Some((s, _, n)) => Some((s, b + c.len_utf8(), n + 1)),
                None => Some((b, b + c.len_utf8(), 1)),
            };
        } else if let Some((s, e, n)) = run.take() {
            if n >= 2 {
                out.push((s..e, Repl::Char('ー')));
            }
        }
    }
    if let Some((s, e, n)) = run {
        if n >= 2 {
            out.push((s..e, Repl::Char('ー')));
        }
    }
}

/// `IgnoreYomiganaPlugin` with SudachiPy's settings: a kanji followed by
/// 1-4 kana in `(` `（` ... `)` `）` loses the bracketed reading.
pub(crate) fn ignore_yomigana(
    input: &Input,
    cats: &CharCategory,
    out: &mut Vec<(Range<usize>, Repl<'_>)>,
) {
    const MAX: usize = 4;
    let opens = |c: char| matches!(c, '(' | '（');
    let closes = |c: char| matches!(c, ')' | '）');
    let is_kanji = |c: char| cats.get(c) & cat::KANJI != 0;
    let is_reading = |c: char| cats.get(c) & (cat::HIRAGANA | cat::KATAKANA) != 0;
    let chars: Vec<(usize, char)> = input.modified.char_indices().collect();
    let n = chars.len();
    let mut i = 0;
    while i < n {
        if is_kanji(chars[i].1) && i + 1 < n && opens(chars[i + 1].1) {
            let mut k = 0;
            while k < MAX && i + 2 + k < n && is_reading(chars[i + 2 + k].1) {
                k += 1;
            }
            if k >= 1 && i + 2 + k < n && closes(chars[i + 2 + k].1) {
                let start = chars[i + 1].0;
                let (rb, rc) = chars[i + 2 + k];
                out.push((start..rb + rc.len_utf8(), Repl::Str(Cow::Borrowed(""))));
                i += 3 + k;
                continue;
            }
        }
        i += 1;
    }
}
