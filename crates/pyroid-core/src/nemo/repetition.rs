// Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
//! The repetition filters' `score_document`
//! (`nemo_curator.stages.text.filters.heuristic.repetition`).
//!
//! N-grams are windows over word ids: each distinct word gets a `u32`, and a window is
//! hashed as its slice of ids (FxHash), so a document's words are hashed once.

use memchr::memmem;
use rustc_hash::{FxHashMap, FxHashSet};

use super::pyhash::{self, HashKey, PySet};
use super::text::{ScoreError, Splitter, paragraphs, ratio, sentences};
use crate::filters::text::py_len;

/// Counts over `items`: how many, how many distinct, and their lengths (code points).
struct Repeats {
    total: usize,
    distinct: usize,
    chars: usize,
    distinct_chars: usize,
}

fn repeats<'a>(items: impl Iterator<Item = &'a str>) -> Repeats {
    let mut seen = FxHashSet::default();
    let mut r = Repeats {
        total: 0,
        distinct: 0,
        chars: 0,
        distinct_chars: 0,
    };
    for s in items {
        let len = py_len(s);
        r.total += 1;
        r.chars += len;
        if seen.insert(s) {
            r.distinct += 1;
            r.distinct_chars += len;
        }
    }
    r
}

/// `RepeatedLinesFilter`: `len(set(sentences)) / len(sentences)`.
pub fn repeated_lines(text: &str) -> Result<f64, ScoreError> {
    let r = repeats(sentences(text));
    if r.total == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(r.distinct, r.total))
}

/// `RepeatedParagraphsFilter`: `len(set(paragraphs)) / len(paragraphs)`.
pub fn repeated_paragraphs(text: &str) -> f64 {
    // `split` gives at least one paragraph.
    let r = repeats(paragraphs(text));
    ratio(r.distinct, r.total)
}

/// `RepeatedLinesByCharFilter`: `len("".join(set(sentences))) / len("".join(sentences))`.
pub fn repeated_lines_by_char(text: &str) -> Result<f64, ScoreError> {
    let r = repeats(sentences(text));
    if r.chars == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(r.distinct_chars, r.chars))
}

/// `RepeatedParagraphsByCharFilter`: the same over paragraphs; divides by zero when every
/// paragraph is empty (`""`, `"\n\n"`).
pub fn repeated_paragraphs_by_char(text: &str) -> Result<f64, ScoreError> {
    let r = repeats(paragraphs(text));
    if r.chars == 0 {
        return Err(ScoreError::ZeroDivision);
    }
    Ok(ratio(r.distinct_chars, r.chars))
}

/// A document's words and their ids (equal words, equal ids).
fn intern<'a>(words: impl Iterator<Item = &'a str>) -> (Vec<&'a str>, Vec<u32>) {
    let mut table: FxHashMap<&str, u32> = FxHashMap::default();
    let mut list = Vec::new();
    let mut ids = Vec::new();
    for w in words {
        let next = u32::try_from(table.len()).expect("fewer than 2**32 distinct words");
        ids.push(*table.entry(w).or_insert(next));
        list.push(w);
    }
    (list, ids)
}

/// `RepeatingTopNGramsFilter`'s result for one document.
#[derive(Clone, Debug, PartialEq)]
pub enum TopNGram {
    /// NeMo's score: the most frequent n-gram is unique, there are fewer than `n` words, or
    /// the tie was resolved with the interpreter's hash key.
    Score(f64),
    /// Several n-grams share the top count. NeMo takes the first of them in the iteration
    /// order of `set(ngrams)`, which depends on the process's string hash seed; without the
    /// key the caller picks it in Python.
    Tied(Tie),
}

/// The tied n-grams of a [`TopNGram::Tied`] document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tie {
    /// How often each of them occurs.
    pub count: usize,
    /// Word index of each tied n-gram's first occurrence, in increasing order; `None` when
    /// every distinct n-gram is tied (NeMo then takes the first element of the set).
    pub starts: Option<Vec<usize>>,
}

/// NeMo's score for the n-gram `words` (joined with single spaces): the characters
/// `text.replace(joined, "")` removes, over `len(text)`.
pub fn top_ngram_score(text: &str, words: &[&str]) -> f64 {
    let joined = words.join(" ");
    let nchar = py_len(text);
    if nchar == 0 {
        return 1.0;
    }
    if joined.is_empty() {
        return 0.0;
    }
    // `str.replace` removes non-overlapping occurrences, left to right.
    let k = memmem::find_iter(text.as_bytes(), joined.as_bytes()).count();
    ratio(k * py_len(&joined), nchar)
}

/// The first n-gram of `set(ngrams)`'s iteration order with `top` occurrences, as CPython
/// orders it with `key`. `uniq` holds `(count, first start)` per distinct n-gram, in order of
/// first occurrence (the order `set()` inserts them; repeats change nothing).
fn first_in_set_order(
    key: HashKey,
    words: &[&str],
    ids: &[u32],
    n: usize,
    uniq: &[(usize, usize)],
    top: usize,
) -> usize {
    // hash(word), once per distinct word.
    let mut word_hash: Vec<Option<i64>> = Vec::new();
    let mut buf = Vec::new();
    for (&w, &id) in words.iter().zip(ids) {
        let id = id as usize;
        if id >= word_hash.len() {
            word_hash.resize(id + 1, None);
        }
        if word_hash[id].is_none() {
            word_hash[id] = Some(pyhash::str_hash(key, w, &mut buf));
        }
    }
    let lane = |id: &u32| word_hash[*id as usize].expect("every word is hashed");
    let mut set = PySet::new();
    for (k, &(_, start)) in uniq.iter().enumerate() {
        let hash = pyhash::tuple_hash(ids[start..start + n].iter().map(lane));
        let item = u32::try_from(k).expect("fewer than 2**32 distinct n-grams");
        // The n-grams are distinct: an equal hash is a collision, never the same tuple.
        set.add(item, hash, |_, _| false);
    }
    set.iter()
        .map(|k| uniq[k as usize])
        .find(|&(count, _)| count == top)
        .map_or(0, |(_, start)| start)
}

/// `RepeatingTopNGramsFilter(n)`: the share of characters the most frequent word n-gram
/// covers (`" ".join(ngram)` removed from `text`); 1.0 below `n` words.
///
/// When several n-grams share the top count, NeMo takes the first in `set(ngrams)` order:
/// with the interpreter's hash `key` that is resolved here, else the document comes back
/// [`TopNGram::Tied`]. `n < 1` gives no n-grams, and NeMo's `max()` raises ValueError.
pub fn repeating_top_ngrams(
    text: &str,
    n: i64,
    splitter: Splitter,
    key: Option<HashKey>,
) -> Result<TopNGram, ScoreError> {
    if n < 1 {
        return Err(ScoreError::EmptyMax);
    }
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    let (words, ids) = intern(splitter.doc_words(text));
    if words.len() < n {
        return Ok(TopNGram::Score(1.0));
    }
    // (count, first start) per distinct window, in order of first occurrence.
    let mut index: FxHashMap<&[u32], usize> = FxHashMap::default();
    index.reserve(ids.len() - n + 1);
    let mut uniq: Vec<(usize, usize)> = Vec::new();
    for (i, w) in ids.windows(n).enumerate() {
        let next = uniq.len();
        let k = *index.entry(w).or_insert(next);
        if k == next {
            uniq.push((0, i));
        }
        uniq[k].0 += 1;
    }
    let top = uniq.iter().map(|&(c, _)| c).max().unwrap_or(0);
    let mut tied = uniq.iter().filter(|&&(c, _)| c == top);
    let first = tied.next().map_or(0, |&(_, s)| s);
    if tied.next().is_none() {
        return Ok(TopNGram::Score(top_ngram_score(
            text,
            &words[first..first + n],
        )));
    }
    if let Some(key) = key {
        let s = first_in_set_order(key, &words, &ids, n, &uniq, top);
        return Ok(TopNGram::Score(top_ngram_score(text, &words[s..s + n])));
    }
    let starts: Vec<usize> = uniq
        .iter()
        .filter(|&&(c, _)| c == top)
        .map(|&(_, s)| s)
        .collect();
    let all = starts.len() == uniq.len();
    Ok(TopNGram::Tied(Tie {
        count: top,
        starts: (!all).then_some(starts),
    }))
}

/// `RepeatingDuplicateNGramsFilter(n)`: the share of characters in n-grams that occurred
/// before (overlapping n-grams counted once, with the spaces between their words); 1.0
/// below `n` words. `n < 1` gives no n-grams: 0.0 (1.0 for `""`).
pub fn repeating_duplicate_ngrams(text: &str, n: i64, splitter: Splitter) -> f64 {
    let nchar = py_len(text);
    let score = |dup: usize| if nchar > 0 { ratio(dup, nchar) } else { 1.0 };
    if n < 1 {
        return score(0);
    }
    let n = usize::try_from(n).unwrap_or(usize::MAX);
    let (words, ids) = intern(splitter.doc_words(text));
    if words.len() < n {
        return 1.0;
    }
    let lens: Vec<usize> = words.iter().map(|w| py_len(w)).collect();
    let mut counts: FxHashMap<&[u32], u32> = FxHashMap::default();
    counts.reserve(ids.len() - n + 1);
    let (mut dup, mut overlap) = (0usize, 0usize);
    for (i, w) in ids.windows(n).enumerate() {
        let c = counts.entry(w).or_insert(0);
        *c += 1;
        if *c > 1 {
            // The words not counted yet, and the spaces before them.
            dup += lens[i + overlap..i + n].iter().sum::<usize>() + (n - overlap).min(n - 1);
            overlap = n;
        }
        overlap = overlap.saturating_sub(1);
    }
    score(dup)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WS: Splitter = Splitter::Whitespace;

    #[test]
    fn repeated() {
        assert_eq!(repeated_lines("a\nb\na\n \n"), Ok(2.0 / 3.0));
        assert_eq!(repeated_lines("a\n a"), Ok(1.0));
        assert_eq!(repeated_lines(" \n"), Err(ScoreError::ZeroDivision));
        assert_eq!(repeated_paragraphs("a\n\nb\n\na"), 2.0 / 3.0);
        assert_eq!(repeated_paragraphs(""), 1.0);
        assert_eq!(repeated_paragraphs("\n\n"), 0.5);
        assert_eq!(repeated_lines_by_char("ab\nc\nab"), Ok(3.0 / 5.0));
        assert_eq!(repeated_lines_by_char(""), Err(ScoreError::ZeroDivision));
        assert_eq!(repeated_paragraphs_by_char("é\n\nxy\n\né"), Ok(3.0 / 4.0));
        assert_eq!(
            repeated_paragraphs_by_char(""),
            Err(ScoreError::ZeroDivision)
        );
        assert_eq!(
            repeated_paragraphs_by_char("\n\n\n\n"),
            Err(ScoreError::ZeroDivision)
        );
        assert_eq!(repeated_paragraphs_by_char("\n\n\n"), Ok(1.0));
    }

    #[test]
    fn top_ngrams() {
        // "a b" twice, every other bigram once: unique top.
        let t = "a b a b c";
        assert_eq!(
            repeating_top_ngrams(t, 2, WS, None),
            Ok(TopNGram::Score(6.0 / 9.0))
        );
        // The joined n-gram is searched in the text: different spacing does not count,
        // occurrences inside other words do.
        assert_eq!(
            repeating_top_ngrams("x y\tx y xx yy", 2, WS, None),
            Ok(TopNGram::Score(9.0 / 13.0))
        );
        assert_eq!(
            repeating_top_ngrams("a b", 3, WS, None),
            Ok(TopNGram::Score(1.0))
        );
        assert_eq!(
            repeating_top_ngrams("", 1, WS, None),
            Ok(TopNGram::Score(1.0))
        );
        assert_eq!(
            repeating_top_ngrams("a", 0, WS, None),
            Err(ScoreError::EmptyMax)
        );
        assert_eq!(
            repeating_top_ngrams("", -3, WS, None),
            Err(ScoreError::EmptyMax)
        );
        assert_eq!(
            repeating_top_ngrams("a b c", 2, WS, None),
            Ok(TopNGram::Tied(Tie {
                count: 1,
                starts: None
            }))
        );
        assert_eq!(
            repeating_top_ngrams("a b a b c d c d e", 2, WS, None),
            Ok(TopNGram::Tied(Tie {
                count: 2,
                starts: Some(vec![0, 4])
            }))
        );
        assert_eq!(
            repeating_top_ngrams("x", 1, WS, None),
            Ok(TopNGram::Score(1.0))
        );
        assert_eq!(
            repeating_top_ngrams("x x y", 1, WS, None),
            Ok(TopNGram::Score(2.0 / 5.0))
        );
    }

    #[test]
    fn top_ngram_ties_with_seed_0() {
        // `PYTHONHASHSEED=0`: NeMo's RepeatingTopNGramsFilter(n).score_document(t) (the key is
        // then all zeros).
        let key = HashKey::from_secret(&[0; 16]);
        let w300: String = (0..300)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let cases: [(&str, [f64; 3]); 6] = [
            ("a b c", [0.2, 0.6, 1.0]),
            (
                "a b a b c d c d e",
                [
                    0.117_647_058_823_529_41,
                    0.352_941_176_470_588_26,
                    0.294_117_647_058_823_54,
                ],
            ),
            (
                "the cat the dog the cat the dog",
                [
                    0.387_096_774_193_548_4,
                    0.451_612_903_225_806_44,
                    0.709_677_419_354_838_7,
                ],
            ),
            (
                "x y z x y z q",
                [
                    0.153_846_153_846_153_85,
                    0.461_538_461_538_461_56,
                    0.769_230_769_230_769_3,
                ],
            ),
            (
                "中文 é 😀 中文 é 😀 a",
                [0.133_333_333_333_333_33, 0.4, 0.8],
            ),
            (
                &w300,
                [
                    0.002_879_769_618_430_525_4,
                    0.006_479_481_641_468_682_5,
                    0.007_919_366_450_683_946,
                ],
            ),
        ];
        for (text, want) in cases {
            for (n, &w) in (1..).zip(&want) {
                assert_eq!(
                    repeating_top_ngrams(text, n, WS, key),
                    Ok(TopNGram::Score(w)),
                    "{text:?} {n}"
                );
            }
        }
    }

    #[test]
    fn duplicate_ngrams() {
        // Python: RepeatingDuplicateNGramsFilter(n).score_document(t).
        assert_eq!(repeating_duplicate_ngrams("a b a b", 2, WS), 3.0 / 7.0);
        assert_eq!(repeating_duplicate_ngrams("a b a b a b", 2, WS), 7.0 / 11.0);
        assert_eq!(
            repeating_duplicate_ngrams("ab cd ab cd ab", 2, WS),
            8.0 / 14.0
        );
        assert_eq!(repeating_duplicate_ngrams("x x x", 1, WS), 2.0 / 5.0);
        assert_eq!(repeating_duplicate_ngrams("a b c", 2, WS), 0.0);
        assert_eq!(repeating_duplicate_ngrams("a b", 3, WS), 1.0);
        assert_eq!(repeating_duplicate_ngrams("a", 0, WS), 0.0);
        assert_eq!(repeating_duplicate_ngrams("", 0, WS), 1.0);
        assert_eq!(repeating_duplicate_ngrams("", -1, WS), 1.0);
        assert_eq!(repeating_duplicate_ngrams("", 1, WS), 1.0);
    }
}
