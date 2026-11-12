// Derived from PyThaiNLP 5.4.0 (Apache-2.0), © 2016-2026 PyThaiNLP Project,
// and datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! `pythainlp.util.Trie`, built once and read only: [`Trie::prefixes`] gives
//! the lengths of the words that start at a position, shortest first, as
//! `Trie.prefixes(text, start)` lists them.
//!
//! The nodes are laid out breadth first, each with its edges sorted by
//! character in one flat array.

/// A read-only trie over characters.
#[derive(Clone)]
pub(super) struct Trie {
    /// `first[n]..first[n + 1]`: the edges of node `n`.
    first: Vec<u32>,
    /// Whether a word ends at node `n`.
    end: Vec<bool>,
    /// Edge labels, sorted within a node.
    chars: Vec<char>,
    /// Edge targets.
    next: Vec<u32>,
    words: usize,
}

impl std::fmt::Debug for Trie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Trie")
            .field("words", &self.words)
            .field("nodes", &self.end.len())
            .finish_non_exhaustive()
    }
}

fn id(n: usize) -> u32 {
    u32::try_from(n).expect("fewer than 2**32 trie nodes")
}

impl Trie {
    /// A trie of `words` (`Trie(words)`; the caller strips them as
    /// `Trie.add` does). The empty word is stored but never a prefix.
    pub(super) fn new<'a>(words: impl IntoIterator<Item = &'a str>) -> Self {
        let mut words: Vec<&str> = words.into_iter().collect();
        // Byte order is code point order, and a word's extensions follow it.
        words.sort_unstable();
        words.dedup();
        let mut trie = Trie {
            first: Vec::new(),
            end: Vec::new(),
            chars: Vec::new(),
            next: Vec::new(),
            words: words.len(),
        };
        // Node `node` is `queue[node]`: the words `lo..hi` that share its
        // prefix of `depth` bytes.
        let mut queue: Vec<(usize, usize, usize)> = vec![(0, words.len(), 0)];
        let mut node = 0;
        while node < queue.len() {
            let (lo, hi, depth) = queue[node];
            trie.first.push(id(trie.chars.len()));
            let mut is_end = false;
            let mut from = lo;
            while from < hi {
                let rest = &words[from][depth..];
                let Some(ch) = rest.chars().next() else {
                    // Sorts first among the words with this prefix.
                    is_end = true;
                    from += 1;
                    continue;
                };
                let mut to = from + 1;
                while to < hi && words[to][depth..].starts_with(ch) {
                    to += 1;
                }
                trie.chars.push(ch);
                trie.next.push(id(queue.len()));
                queue.push((from, to, depth + ch.len_utf8()));
                from = to;
            }
            trie.end.push(is_end);
            node += 1;
        }
        trie.first.push(id(trie.chars.len()));
        trie
    }

    /// The number of distinct words.
    pub(super) fn len(&self) -> usize {
        self.words
    }

    /// The child of `node` along `c`.
    #[inline]
    fn child(&self, node: u32, c: char) -> Option<u32> {
        let n = node as usize;
        let (lo, hi) = (self.first[n] as usize, self.first[n + 1] as usize);
        let cs = &self.chars[lo..hi];
        let i = if cs.len() <= 8 {
            cs.iter().position(|&x| x == c)?
        } else {
            cs.binary_search(&c).ok()?
        };
        Some(self.next[lo + i])
    }

    /// The lengths of the words that start at `start` in `text`, shortest
    /// first (`Trie.prefixes(text, start)`).
    #[inline]
    pub(super) fn prefixes<'t>(&'t self, text: &'t [char], start: usize) -> Prefixes<'t> {
        Prefixes {
            trie: self,
            text,
            start,
            i: start,
            node: Some(0),
        }
    }

    /// Whether `word` is in the trie (`word in trie`).
    #[cfg(test)]
    pub(super) fn contains(&self, word: &str) -> bool {
        let mut node = 0;
        for c in word.chars() {
            match self.child(node, c) {
                Some(n) => node = n,
                None => return false,
            }
        }
        self.end[node as usize]
    }
}

/// See [`Trie::prefixes`].
#[derive(Debug)]
pub(super) struct Prefixes<'t> {
    trie: &'t Trie,
    text: &'t [char],
    start: usize,
    i: usize,
    node: Option<u32>,
}

impl Iterator for Prefixes<'_> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let mut node = self.node?;
        while let Some(&c) = self.text.get(self.i) {
            let Some(n) = self.trie.child(node, c) else {
                break;
            };
            node = n;
            self.i += 1;
            if self.trie.end[n as usize] {
                self.node = Some(n);
                return Some(self.i - self.start);
            }
        }
        self.node = None;
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prefixes(t: &Trie, text: &str, start: usize) -> Vec<usize> {
        let s: Vec<char> = text.chars().collect();
        t.prefixes(&s, start).collect()
    }

    #[test]
    fn prefixes_like_pythainlp() {
        let t = Trie::new(["สวัสดี", "สวัส", "ดี", "ครับ", "", "a", "ab", "abcd"]);
        assert_eq!(t.len(), 8);
        assert_eq!(prefixes(&t, "สวัสดีครับ", 0), [4, 6]);
        assert_eq!(prefixes(&t, "สวัสดีครับ", 4), [2]);
        assert_eq!(prefixes(&t, "สวัสดีครับ", 6), [4]);
        assert_eq!(prefixes(&t, "สวัสดีครับ", 10), [0usize; 0]);
        assert_eq!(prefixes(&t, "abcde", 0), [1, 2, 4]);
        assert_eq!(prefixes(&t, "abc", 0), [1, 2]);
        assert_eq!(prefixes(&t, "xabc", 1), [1, 2]);
        assert!(t.contains("ครับ") && t.contains("") && !t.contains("คร"));
    }
}
