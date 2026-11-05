// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! `GopherQualityFilter` and `GopherRepetitionFilter`
//! (`datatrove/pipeline/filters/gopher_*_filter.py`).

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::LazyLock;

use rustc_hash::FxBuildHasher;

use super::text::{self, CharSet, py_len};
use super::{Doc, FilterError, Reason, Threshold, Verdict, as_f64, ratio, tables, unicode};

/// datatrove's `STOP_WORDS` (the Gopher default).
pub const STOP_WORDS: [&str; 8] = ["the", "be", "to", "of", "and", "that", "have", "with"];

static PUNCTUATION: LazyLock<CharSet> =
    LazyLock::new(|| CharSet::new(tables::PUNCTUATION_SET.iter().copied()));

/// `GopherQualityFilter(...)`. `None` disables a check, like datatrove's
/// falsy values (`0`, `None`).
#[derive(Clone, Debug)]
pub struct GopherQuality {
    pub min_doc_words: Threshold,
    pub max_doc_words: Threshold,
    pub min_avg_word_length: Threshold,
    pub max_avg_word_length: Threshold,
    pub max_symbol_word_ratio: Threshold,
    pub max_bullet_lines_ratio: Threshold,
    pub max_ellipsis_lines_ratio: Threshold,
    pub max_non_alpha_words_ratio: Threshold,
    pub min_stop_words: Threshold,
    /// The stop words (a set, as in datatrove).
    pub stop_words: Vec<String>,
}

impl Default for GopherQuality {
    fn default() -> Self {
        Self {
            min_doc_words: Some(50.0),
            max_doc_words: Some(100_000.0),
            min_avg_word_length: Some(3.0),
            max_avg_word_length: Some(10.0),
            max_symbol_word_ratio: Some(0.1),
            max_bullet_lines_ratio: Some(0.9),
            max_ellipsis_lines_ratio: Some(0.3),
            max_non_alpha_words_ratio: Some(0.8),
            min_stop_words: Some(2.0),
            stop_words: STOP_WORDS.iter().map(|s| (*s).to_owned()).collect(),
        }
    }
}

#[allow(clippy::unnecessary_wraps)] // reads like datatrove's `return False, "reason"`
const fn drop(reason: &'static str) -> Result<Verdict, FilterError> {
    Ok(Some(Reason::Fixed(reason)))
}

impl GopherQuality {
    /// `GopherQualityFilter.filter(doc)`.
    pub fn filter(&self, doc: &mut Doc<'_, '_>) -> Result<Verdict, FilterError> {
        let text = doc.text();
        let version = doc.tokenizer().unicode_version();
        let words = doc.words()?;
        let n_words = words.len();

        let mut n_non_symbol = 0usize;
        let mut non_symbol_len = 0usize;
        for w in words {
            if w.chars().any(|c| !PUNCTUATION.contains(c)) {
                n_non_symbol += 1;
                non_symbol_len += py_len(w);
            }
        }
        if let Some(t) = self.min_doc_words {
            if as_f64(n_non_symbol) < t {
                return drop("gopher_short_doc");
            }
        }
        if let Some(t) = self.max_doc_words {
            if as_f64(n_non_symbol) > t {
                return drop("gopher_long_doc");
            }
        }
        // np.mean of the lengths: the integer sum is exact in f64; NaN when empty.
        let avg = if n_non_symbol == 0 {
            f64::NAN
        } else {
            as_f64(non_symbol_len) / as_f64(n_non_symbol)
        };
        if let Some(t) = self.min_avg_word_length {
            if avg < t {
                return drop("gopher_below_avg_threshold");
            }
        }
        if let Some(t) = self.max_avg_word_length {
            if avg > t {
                return drop("gopher_above_avg_threshold");
            }
        }
        if let Some(t) = self.max_symbol_word_ratio {
            let hashes = memchr::memchr_iter(b'#', text.as_bytes()).count();
            if ratio(hashes, n_words)? > t {
                return drop("gopher_too_many_hashes");
            }
            let ellipses = text.matches("...").count() + text.matches('\u{2026}').count();
            if ratio(ellipses, n_words)? > t {
                return drop("gopher_too_many_ellipsis");
            }
        }
        if self.max_bullet_lines_ratio.is_some() || self.max_ellipsis_lines_ratio.is_some() {
            let (mut n_lines, mut bullets, mut ellipsis_end) = (0usize, 0usize, 0usize);
            for line in text::py_splitlines(text) {
                n_lines += 1;
                let l = text::py_lstrip(line);
                bullets += usize::from(l.starts_with('\u{2022}') || l.starts_with('-'));
                let r = text::py_rstrip(line);
                ellipsis_end += usize::from(r.ends_with("...") || r.ends_with('\u{2026}'));
            }
            if let Some(t) = self.max_bullet_lines_ratio {
                if ratio(bullets, n_lines)? > t {
                    return drop("gopher_too_many_bullets");
                }
            }
            if let Some(t) = self.max_ellipsis_lines_ratio {
                if ratio(ellipsis_end, n_lines)? > t {
                    return drop("gopher_too_many_end_ellipsis");
                }
            }
        }
        if let Some(t) = self.max_non_alpha_words_ratio {
            let alpha = words
                .iter()
                .filter(|w| w.chars().any(|c| unicode::is_alpha(c, version)))
                .count();
            if ratio(alpha, n_words)? < t {
                return drop("gopher_below_alpha_threshold");
            }
        }
        if let Some(t) = self.min_stop_words {
            let stop: HashSet<&str, FxBuildHasher> =
                self.stop_words.iter().map(String::as_str).collect();
            let mut found: HashSet<&str, FxBuildHasher> = HashSet::default();
            for w in words {
                if let Some(s) = stop.get(w) {
                    found.insert(s);
                    if as_f64(found.len()) >= t {
                        break;
                    }
                }
            }
            if as_f64(found.len()) < t {
                return drop("gopher_enough_stop_words");
            }
        }
        Ok(None)
    }
}

/// `GopherRepetitionFilter(...)`. `None` disables a fraction check.
#[derive(Clone, Debug)]
pub struct GopherRepetition {
    pub dup_line_frac: Threshold,
    pub dup_para_frac: Threshold,
    pub dup_line_char_frac: Threshold,
    pub dup_para_char_frac: Threshold,
    /// `(n, max fraction)` for the most frequent n-gram (`n >= 1`).
    pub top_n_grams: Vec<(usize, f64)>,
    /// `(n, max fraction)` for duplicated n-grams (`n >= 1`).
    pub dup_n_grams: Vec<(usize, f64)>,
}

impl Default for GopherRepetition {
    fn default() -> Self {
        Self {
            dup_line_frac: Some(0.3),
            dup_para_frac: Some(0.3),
            dup_line_char_frac: Some(0.2),
            dup_para_char_frac: Some(0.2),
            top_n_grams: vec![(2, 0.2), (3, 0.18), (4, 0.16)],
            dup_n_grams: vec![
                (5, 0.15),
                (6, 0.14),
                (7, 0.13),
                (8, 0.12),
                (9, 0.11),
                (10, 0.10),
            ],
        }
    }
}

/// `find_duplicates(x)`: (repeated elements, their total length in chars).
fn find_duplicates<'a>(items: impl IntoIterator<Item = &'a str>) -> (usize, usize) {
    let mut seen: HashSet<&str, FxBuildHasher> = HashSet::default();
    let (mut elems, mut chars) = (0, 0);
    for s in items {
        if !seen.insert(s) {
            elems += 1;
            chars += py_len(s);
        }
    }
    (elems, chars)
}

/// Polynomial hash of bytes, composable over concatenation:
/// `H(a + b) = H(a) * P^len(b) + H(b)` (mod 2^64).
const P: u64 = 0x100_0000_01b3;

fn poly(s: &str) -> (u64, u64) {
    let (mut h, mut p) = (0u64, 1u64);
    for &b in s.as_bytes() {
        h = h.wrapping_mul(P).wrapping_add(u64::from(b) + 1);
        p = p.wrapping_mul(P);
    }
    (h, p)
}

/// An n-gram compared as a word sequence (`" ".join(words)`: words contain no
/// spaces, so the join is injective).
struct Seq<'w, 'a> {
    words: &'w [&'a str],
    hash: u64,
}

impl Hash for Seq<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl PartialEq for Seq<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash && self.words == other.words
    }
}

impl Eq for Seq<'_, '_> {}

/// An n-gram compared as the concatenation `"".join(words)`.
struct Concat<'w, 'a> {
    words: &'w [&'a str],
    hash: u64,
}

impl Hash for Concat<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.hash);
    }
}

impl PartialEq for Concat<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
            && self
                .words
                .iter()
                .flat_map(|w| w.bytes())
                .eq(other.words.iter().flat_map(|w| w.bytes()))
    }
}

impl Eq for Concat<'_, '_> {}

/// Per-word hashes and lengths, computed once for all n.
struct WordStats<'w, 'a> {
    words: &'w [&'a str],
    hash: Vec<u64>,
    pow: Vec<u64>,
    chars: Vec<usize>,
}

impl<'w, 'a> WordStats<'w, 'a> {
    fn new(words: &'w [&'a str]) -> Self {
        let mut hash = Vec::with_capacity(words.len());
        let mut pow = Vec::with_capacity(words.len());
        let mut chars = Vec::with_capacity(words.len());
        for w in words {
            let (h, p) = poly(w);
            hash.push(h);
            pow.push(p);
            chars.push(py_len(w));
        }
        Self {
            words,
            hash,
            pow,
            chars,
        }
    }

    fn concat_hash(&self, i: usize, n: usize) -> u64 {
        (i..i + n).fold(0u64, |h, j| {
            h.wrapping_mul(self.pow[j]).wrapping_add(self.hash[j])
        })
    }

    fn seq_hash(&self, i: usize, n: usize) -> u64 {
        (i..i + n).fold(0u64, |h, j| {
            h.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add(self.hash[j])
                ^ (h >> 29)
        })
    }

    /// `find_top_duplicate(get_n_grams(words, n))`: chars of the most common
    /// n-gram (first seen wins ties) times its count; `None` if no n-grams.
    fn top_duplicate(&self, n: usize) -> Option<usize> {
        let len = self.words.len();
        if n > len {
            return None;
        }
        let mut counts: HashMap<Seq<'w, 'a>, (u32, u32), FxBuildHasher> =
            HashMap::with_capacity_and_hasher(len - n + 1, FxBuildHasher);
        for i in 0..=len - n {
            let key = Seq {
                words: &self.words[i..i + n],
                hash: self.seq_hash(i, n),
            };
            let first = u32::try_from(i).unwrap_or(u32::MAX); // spaCy caps texts at 2**30 chars
            counts.entry(key).or_insert((0, first)).0 += 1;
        }
        let (count, first) = counts
            .values()
            .copied()
            .max_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)))?;
        let first = first as usize;
        let chars: usize = self.chars[first..first + n].iter().sum::<usize>() + (n - 1);
        Some(chars * count as usize)
    }

    /// `find_all_duplicate(words, n)`.
    fn all_duplicate(&self, n: usize) -> usize {
        let len = self.words.len();
        let mut unique: HashSet<Concat<'w, 'a>, FxBuildHasher> = HashSet::default();
        let (mut repeated, mut idx) = (0usize, 0usize);
        while idx + n <= len {
            let key = Concat {
                words: &self.words[idx..idx + n],
                hash: self.concat_hash(idx, n),
            };
            if unique.contains(&key) {
                repeated += self.chars[idx..idx + n].iter().sum::<usize>();
                idx += n;
            } else {
                unique.insert(key);
                idx += 1;
            }
        }
        repeated
    }
}

impl GopherRepetition {
    /// `GopherRepetitionFilter.filter(doc)`.
    pub fn filter(&self, doc: &mut Doc<'_, '_>) -> Result<Verdict, FilterError> {
        let text = doc.text();
        if text.is_empty() {
            return drop("empty");
        }
        let text_len = py_len(text);

        let paragraphs = text::split_newline_runs(text::py_strip(text), 2);
        let (dups, chars) = find_duplicates(paragraphs.iter().copied());
        if let Some(t) = self.dup_para_frac {
            if ratio(dups, paragraphs.len())? > t {
                return drop("dup_para_frac");
            }
        }
        if let Some(t) = self.dup_para_char_frac {
            if ratio(chars, text_len)? > t {
                return drop("dup_para_char_frac");
            }
        }

        let lines = text::split_newline_runs(text, 1);
        let (dups, chars) = find_duplicates(lines.iter().copied());
        if let Some(t) = self.dup_line_frac {
            if ratio(dups, lines.len())? > t {
                return drop("dup_line_frac");
            }
        }
        if let Some(t) = self.dup_line_char_frac {
            if ratio(chars, text_len)? > t {
                return drop("dup_line_char_frac");
            }
        }

        // Tokenized even with no n-gram checks: datatrove does (and can raise).
        let words = doc.words()?;
        let stats = WordStats::new(words);
        for &(n, frac) in &self.top_n_grams {
            let Some(top) = stats.top_duplicate(n) else {
                continue;
            };
            if ratio(top, text_len)? > frac {
                return Ok(Some(Reason::TopNGram(n)));
            }
        }
        for &(n, frac) in &self.dup_n_grams {
            if ratio(stats.all_duplicate(n), text_len)? > frac {
                return Ok(Some(Reason::DuplicatedNGrams(n)));
            }
        }
        Ok(None)
    }
}
