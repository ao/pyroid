//! Word n-gram shingles of a text, as `pyroid.datasketch.minhash_batch`
//! defines them for text input:
//!
//! ```python
//! words = text.split()
//! shingles = [" ".join(words[i : i + n]).encode("utf-8")
//!             for i in range(len(words) - n + 1)]
//! ```
//!
//! A text with fewer than `n` words has no shingles. Words are split on
//! exactly the characters Python's `str.split()` uses
//! ([`crate::text::is_py_whitespace`]).

use crate::text::is_py_whitespace;

/// Calls `f` with the UTF-8 bytes of every word `n`-gram of `text`, in order.
///
/// `scratch` holds word boundaries and joined shingles between calls, so a
/// caller that reuses it allocates nothing per document. An n-gram whose
/// words are separated by single ASCII spaces in `text` is passed as a slice
/// of `text` itself.
///
/// # Panics
/// If `n` is 0.
pub fn for_each_word_ngram(text: &str, n: usize, scratch: &mut Scratch, mut f: impl FnMut(&[u8])) {
    assert!(n > 0, "n-gram size must be positive");
    let words = &mut scratch.words;
    words.clear();
    let mut start = None;
    for (i, c) in text.char_indices() {
        if is_py_whitespace(c) {
            if let Some(s) = start.take() {
                words.push((s, i));
            }
        } else if start.is_none() {
            start = Some(i);
        }
    }
    if let Some(s) = start {
        words.push((s, text.len()));
    }
    if words.len() < n {
        return;
    }
    let bytes = text.as_bytes();
    // single_gap[i]: words i and i + 1 are separated by exactly one b' '.
    let gaps = &mut scratch.single_gap;
    gaps.clear();
    gaps.extend(
        words
            .windows(2)
            .map(|w| w[1].0 == w[0].1 + 1 && bytes[w[0].1] == b' '),
    );
    // Number of non-single gaps among the n - 1 gaps of the current window.
    let mut odd = gaps[..n - 1].iter().filter(|&&g| !g).count();
    let buf = &mut scratch.joined;
    for i in 0..=words.len() - n {
        if i > 0 {
            // Gap i + n - 2 enters the window, gap i - 1 leaves it (the same gap when n is 1).
            odd += usize::from(!gaps[i + n - 2]);
            odd -= usize::from(!gaps[i - 1]);
        }
        let (s, _) = words[i];
        let (_, e) = words[i + n - 1];
        if odd == 0 {
            f(&bytes[s..e]);
        } else {
            buf.clear();
            for (j, &(ws, we)) in words[i..i + n].iter().enumerate() {
                if j > 0 {
                    buf.push(b' ');
                }
                buf.extend_from_slice(&bytes[ws..we]);
            }
            f(buf);
        }
    }
}

/// Reusable buffers for [`for_each_word_ngram`].
#[derive(Debug, Default)]
pub struct Scratch {
    words: Vec<(usize, usize)>,
    single_gap: Vec<bool>,
    joined: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(text: &str, n: usize) -> Vec<Vec<u8>> {
        let words: Vec<&str> = text
            .split(is_py_whitespace)
            .filter(|w| !w.is_empty())
            .collect();
        if words.len() < n {
            return Vec::new();
        }
        (0..=words.len() - n)
            .map(|i| words[i..i + n].join(" ").into_bytes())
            .collect()
    }

    fn shingles(text: &str, n: usize) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        for_each_word_ngram(text, n, &mut Scratch::default(), |s| out.push(s.to_vec()));
        out
    }

    #[test]
    fn matches_split_and_join() {
        let texts = [
            "",
            " ",
            "a",
            "a b",
            "  a  b\tc\n d e\u{3000}f\u{1c}g ",
            "one two three four five six",
            "x  y z",
            "\u{e9}t\u{e9} na\u{ef}ve \u{1f600} ok",
            "a\u{85}b\u{a0}c\u{2028}d",
        ];
        for t in texts {
            for n in 1..=6 {
                assert_eq!(shingles(t, n), reference(t, n), "{t:?} n={n}");
            }
        }
    }

    #[test]
    fn fewer_words_than_n() {
        assert_eq!(shingles("a b", 3), Vec::<Vec<u8>>::new());
        assert_eq!(shingles("a b c", 3), [b"a b c".to_vec()]);
    }

    #[test]
    fn reuses_slices_when_single_spaced() {
        let text = "a b  c d";
        let mut from_text = 0;
        for_each_word_ngram(text, 2, &mut Scratch::default(), |s| {
            let p = s.as_ptr() as usize;
            let base = text.as_ptr() as usize;
            if p >= base && p < base + text.len() {
                from_text += 1;
            }
        });
        // "a b" and "c d" are slices of the text; "b c" is joined.
        assert_eq!(from_text, 2);
    }
}
