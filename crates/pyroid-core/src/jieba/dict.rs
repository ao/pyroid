// Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi; modified.
//! jieba's prefix dictionary (`Tokenizer.FREQ` and `Tokenizer.total`).
//!
//! jieba keeps every word with its frequency in a dict, plus every prefix of
//! every word with frequency 0, so "is `frag` a prefix of some word" is
//! `frag in FREQ`. Here the words and prefixes are the nodes of a character
//! trie ([`super::trie`]). The default dictionary's trie is built at
//! generation time and stored in the binary (2.5 MB); reading it takes about
//! 2 ms, on first use. Words added later (`add_word`, `del_word`,
//! `load_userdict`) go to a hash map overlay that takes precedence, with the
//! prefixes jieba would add.

use std::sync::{Arc, LazyLock};

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use super::trie::Trie;

/// The default dictionary (jieba 0.42.1 `dict.txt`) as a serialized trie of
/// its words and prefixes with their frequencies (the last line wins for a
/// duplicate word, as in `gen_pfdict`). Regenerate with the `regenerate` test.
static DEFAULT_TRIE: &[u8] = include_bytes!("data/dict.trie");
/// `Tokenizer.total` for the default dictionary: the sum of the frequency
/// column over every line (the duplicate line included).
pub const DEFAULT_TOTAL: i64 = 60_101_967;
/// `len(Tokenizer.FREQ)` for the default dictionary: its words and their
/// prefixes.
pub const DEFAULT_FREQ_LEN: usize = 498_113;
/// Words (distinct keys) in the default dictionary.
pub const DEFAULT_WORDS: usize = 349_045;

/// Why a dictionary could not be built or changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DictError {
    /// The sum of the frequencies does not fit in an `i64`.
    Overflow,
}

impl std::fmt::Display for DictError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DictError::Overflow => f.write_str("dictionary frequencies overflow a 64-bit integer"),
        }
    }
}

impl std::error::Error for DictError {}

/// `Tokenizer.FREQ` and `Tokenizer.total`. Cheap to clone (the trie is
/// shared); `Sync`, so one instance serves every thread.
#[derive(Clone)]
pub struct Dictionary {
    trie: Arc<Trie>,
    /// Entries that differ from the trie: words added or changed after
    /// loading, and the new prefixes `add_word` inserts with frequency 0.
    overlay: FxHashMap<Box<str>, i64>,
    /// First characters of the overlay's keys (the overlay is only consulted
    /// for fragments that start with one of them).
    overlay_first: FxHashSet<char>,
    /// Overlay keys that are not in the trie (each adds 1 to `len`).
    overlay_new: usize,
    total: i64,
}

impl std::fmt::Debug for Dictionary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Dictionary")
            .field("len", &self.len())
            .field("total", &self.total)
            .field("overlay", &self.overlay.len())
            .finish_non_exhaustive()
    }
}

static DEFAULT: LazyLock<Dictionary> = LazyLock::new(|| Dictionary {
    trie: Arc::new(Trie::from_blob(DEFAULT_TRIE).expect("the bundled jieba dictionary")),
    overlay: FxHashMap::default(),
    overlay_first: FxHashSet::default(),
    overlay_new: 0,
    total: DEFAULT_TOTAL,
});

impl Dictionary {
    /// jieba's default dictionary (`dict.txt` of jieba 0.42.1), unmodified.
    /// Read from the bundled trie on first use.
    pub fn default_ref() -> &'static Dictionary {
        &DEFAULT
    }

    /// The bundled trie of the default dictionary, serialized.
    pub fn default_trie_bytes() -> &'static [u8] {
        DEFAULT_TRIE
    }

    /// The serialized trie of `(word, freq)` words (as `dict.txt` lists
    /// them: the last frequency wins), as bundled for the default
    /// dictionary; `None` if it does not fit the format (a label outside the
    /// BMP, 65,536 distinct frequencies or more, a negative frequency or the
    /// empty word).
    pub fn serialize_words(words: &[(&str, i64)]) -> Option<Vec<u8>> {
        let mut last: FxHashMap<&str, i64> = FxHashMap::default();
        for &(w, f) in words {
            last.insert(w, f);
        }
        let mut sorted: Vec<(&str, i64)> = last.into_iter().collect();
        sorted.sort_unstable();
        if sorted.iter().any(|&(w, f)| f < 0 || w.is_empty()) {
            return None;
        }
        Trie::from_sorted(sorted.iter().map(|&(w, f)| (w, f.unsigned_abs())))
            .to_blob()
            .ok()
    }

    /// A dictionary from `(word, freq)` entries in file order and the file's
    /// `total` (the sum of every line's frequency), as `gen_pfdict` builds
    /// `FREQ`: for a repeated word the last frequency wins, and every prefix
    /// of a word is present (with frequency 0 unless it is a word itself).
    pub fn from_entries(entries: Vec<(String, i64)>, total: i64) -> Result<Self, DictError> {
        let mut last: FxHashMap<String, i64> =
            FxHashMap::with_capacity_and_hasher(entries.len(), FxBuildHasher);
        for (w, f) in entries {
            last.insert(w, f);
        }
        let mut words: Vec<(String, i64)> = last.into_iter().collect();
        words.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        // The trie stores non-negative frequencies; a negative one (only
        // possible through Python's `int()`) goes to the overlay, and so does
        // the empty word (a prefix of everything, so not a trie node).
        let (base, rest): (Vec<_>, Vec<_>) = words
            .into_iter()
            .partition(|(w, f)| *f >= 0 && !w.is_empty());
        let trie = Trie::from_sorted(base.iter().map(|(w, f)| (w.as_str(), f.unsigned_abs())));
        let mut me = Dictionary {
            trie: Arc::new(trie),
            overlay: FxHashMap::default(),
            overlay_first: FxHashSet::default(),
            overlay_new: 0,
            total,
        };
        for (w, f) in rest {
            me.set(&w, f);
            me.add_prefixes(&w);
        }
        Ok(me)
    }

    /// `Tokenizer.total`.
    pub fn total(&self) -> i64 {
        self.total
    }

    /// `len(Tokenizer.FREQ)`.
    pub fn len(&self) -> usize {
        self.trie.len() + self.overlay_new
    }

    /// `not Tokenizer.FREQ`.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The default dictionary, unmodified.
    pub fn is_pristine_default(&self) -> bool {
        self.overlay.is_empty()
            && self.total == DEFAULT_TOTAL
            && Arc::ptr_eq(&self.trie, &DEFAULT.trie)
    }

    /// `Tokenizer.FREQ.get(word)`.
    pub fn get(&self, word: &str) -> Option<i64> {
        if let Some(&f) = self.overlay.get(word) {
            return Some(f);
        }
        self.base_get(word)
    }

    /// `word in FREQ` and its value, from the trie alone: `Some(freq)` for a
    /// word, `Some(0)` for a prefix of one.
    #[allow(clippy::cast_possible_wrap)] // trie frequencies come from i64s >= 0
    fn base_get(&self, word: &str) -> Option<i64> {
        self.trie.find(word).map(|x| self.trie.freq(x) as i64)
    }

    /// `FREQ[word] = freq` (the overlay, keeping `len` in step).
    fn set(&mut self, word: &str, freq: i64) {
        if let Some(v) = self.overlay.get_mut(word) {
            *v = freq;
            return;
        }
        if self.base_get(word).is_none() {
            self.overlay_new += 1;
        }
        if let Some(c) = word.chars().next() {
            self.overlay_first.insert(c);
        }
        self.overlay.insert(word.into(), freq);
    }

    /// `add_word`'s prefix loop: every prefix of `word` not in `FREQ` gets 0.
    fn add_prefixes(&mut self, word: &str) {
        for (i, c) in word.char_indices() {
            let frag = &word[..i + c.len_utf8()];
            if self.get(frag).is_none() {
                self.set(frag, 0);
            }
        }
    }

    /// `add_word(word, freq)` once `freq` is known: `FREQ[word] = freq`,
    /// `total += freq`, and the missing prefixes with 0. (The caller handles
    /// the tag and `finalseg.add_force_split` for `freq == 0`.)
    pub fn add_word(&mut self, word: &str, freq: i64) -> Result<(), DictError> {
        let total = self.total.checked_add(freq).ok_or(DictError::Overflow)?;
        self.set(word, freq);
        self.total = total;
        self.add_prefixes(word);
        Ok(())
    }

    /// `get_DAG` for one fragment start `k`: every end `i` (an index into
    /// `chars`, inclusive) such that `chars[k..=i]` has a non-zero frequency,
    /// with that frequency, appended to `out`; or `(k, 1)` when there is none
    /// (`calc` then uses `FREQ.get(...) or 1`). The loop stops at the first
    /// fragment not in `FREQ`, as jieba's does.
    ///
    /// `text[offs[j]..offs[j + 1]]` is `chars[j]`.
    #[inline]
    pub(crate) fn dag_from(
        &self,
        text: &str,
        chars: &[char],
        offs: &[u32],
        k: usize,
        out: &mut Vec<(u32, i64)>,
    ) {
        let before = out.len();
        let overlay = !self.overlay.is_empty() && self.overlay_first.contains(&chars[k]);
        let mut node = self.trie.root(chars[k]);
        for i in k..chars.len() {
            if i > k {
                node = node.and_then(|x| self.trie.child(x, chars[i]));
            }
            #[allow(clippy::cast_possible_wrap)]
            let freq = if overlay {
                match self
                    .overlay
                    .get(&text[offs[k] as usize..offs[i + 1] as usize])
                {
                    Some(&f) => Some(f),
                    None => node.map(|x| self.trie.freq(x) as i64),
                }
            } else {
                match node {
                    Some(x) => Some(self.trie.freq(x) as i64),
                    None => break,
                }
            };
            match freq {
                None => break,
                Some(0) => {}
                #[allow(clippy::cast_possible_truncation)] // blocks < 4 GiB
                Some(f) => out.push((i as u32, f)),
            }
        }
        if out.len() == before {
            #[allow(clippy::cast_possible_truncation)]
            out.push((k as u32, 1));
        }
    }

    /// Every entry of `FREQ` (words and prefixes), the trie's in code point
    /// order and then the overlay's new keys, each with its current value.
    pub fn entries(&self) -> Vec<(String, i64)> {
        let mut out: Vec<(String, i64)> = self
            .trie
            .entries()
            .into_iter()
            .map(|(k, _)| {
                let v = self.get(&k).unwrap_or(0);
                (k, v)
            })
            .collect();
        let mut extra: Vec<(&str, i64)> = self
            .overlay
            .iter()
            .filter(|(k, _)| self.base_get(k).is_none())
            .map(|(k, &v)| (&**k, v))
            .collect();
        extra.sort_unstable();
        out.extend(extra.into_iter().map(|(k, v)| (k.to_owned(), v)));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_dictionary_matches_jieba() {
        let d = Dictionary::default_ref();
        assert_eq!(d.total(), DEFAULT_TOTAL);
        assert_eq!(d.len(), DEFAULT_FREQ_LEN);
        let words = d.trie.entries().iter().filter(|(_, f)| *f > 0).count();
        assert_eq!(words, DEFAULT_WORDS);
        // dict.txt: "了 883634 ul", "B超 3 n" (twice), "AT&T 3 nz".
        assert_eq!(d.get("了"), Some(883_634));
        assert_eq!(d.get("B超"), Some(3));
        assert_eq!(d.get("AT&T"), Some(3));
        assert_eq!(d.get("AT&"), Some(0));
        assert_eq!(d.get("AT&Tx"), None);
        assert_eq!(d.get(""), None);
        assert!(d.is_pristine_default());
    }

    #[test]
    fn add_word_adds_prefixes_and_total() {
        let mut d = Dictionary::default_ref().clone();
        let n = d.len();
        d.add_word("石墨烯烯烯", 7).unwrap();
        assert_eq!(d.get("石墨烯烯烯"), Some(7));
        assert_eq!(d.get("石墨烯烯"), Some(0));
        assert_eq!(d.total(), DEFAULT_TOTAL + 7);
        // "石墨" is in dict.txt; jieba 0.42.1 adds 3 entries.
        assert_eq!(d.len(), n + 3);
        d.add_word("了", 0).unwrap();
        assert_eq!(d.get("了"), Some(0));
        assert_eq!(d.len(), n + 3);
        assert!(!d.is_pristine_default());
        d.add_word("", 5).unwrap();
        assert_eq!(d.get(""), Some(5));
        assert_eq!(d.len(), n + 4);
    }

    #[test]
    fn from_entries_is_gen_pfdict() {
        let d = Dictionary::from_entries(
            vec![
                ("ab".into(), 3),
                ("a".into(), 2),
                ("ab".into(), 5),
                ("xyz".into(), -4),
            ],
            6,
        )
        .unwrap();
        assert_eq!(d.get("ab"), Some(5));
        assert_eq!(d.get("a"), Some(2));
        assert_eq!(d.get("xyz"), Some(-4));
        assert_eq!(d.get("xy"), Some(0));
        assert_eq!(d.get("x"), Some(0));
        assert_eq!(d.len(), 5);
        assert_eq!(d.total(), 6);
        assert_eq!(d.entries().len(), 5);
    }
}
