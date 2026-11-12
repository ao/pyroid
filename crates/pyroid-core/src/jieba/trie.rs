//! The prefix dictionary as a character trie: `frag in FREQ` is "the trie has
//! a node for `frag`", and `FREQ[frag]` is that node's frequency (0 for a
//! prefix that is not itself a word).
//!
//! Nodes are stored level by level (all one-character prefixes in order, then
//! all two-character ones, ...), so the children of a node are contiguous
//! and sorted: `children(x)` is `first[x]..first[x + 1]`, searched by binary
//! search. The first character is looked up in a direct table for the Basic
//! Multilingual Plane.

/// No node.
const NONE: u32 = u32::MAX;

/// Magic and version of the serialized form ([`Trie::to_blob`]).
const MAGIC: &[u8; 4] = b"JBTR";
const VERSION: u32 = 2;

/// Why a serialized trie could not be read or written.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BlobError(pub(crate) &'static str);

/// A character trie over a sorted set of words.
#[derive(Debug)]
pub(crate) struct Trie {
    /// The last character of each node's prefix.
    label: Vec<char>,
    /// `first[x]..first[x + 1]` are the children of node `x` (one more entry
    /// than nodes).
    first: Vec<u32>,
    /// The frequency of each node's prefix as a word, 0 if it is only a prefix.
    freq: Vec<u64>,
    /// Number of one-character prefixes (nodes `0..roots`).
    roots: u32,
    /// For each BMP code point, its one-character node, or `NONE`.
    root_bmp: Box<[u32]>,
}

impl Trie {
    /// Build from `(word, freq)` in increasing byte (= code point) order,
    /// without duplicates.
    pub(crate) fn from_sorted<'a>(words: impl Iterator<Item = (&'a str, u64)>) -> Self {
        // levels[d]: the (d + 1)-character prefixes, in order, as
        // (label, index of the parent in levels[d - 1], freq).
        let mut levels: Vec<Vec<(char, u32, u64)>> = Vec::new();
        let mut prev: Vec<char> = Vec::new();
        let mut cur: Vec<char> = Vec::new();
        for (w, f) in words {
            cur.clear();
            cur.extend(w.chars());
            let lcp = prev.iter().zip(&cur).take_while(|(a, b)| a == b).count();
            if lcp == cur.len() {
                // `w` is a prefix of the previous word: impossible in sorted
                // order unless it repeats it.
                if let Some(last) = levels
                    .get_mut(cur.len().wrapping_sub(1))
                    .and_then(|l| l.last_mut())
                {
                    last.2 = f;
                }
                continue;
            }
            for d in lcp..cur.len() {
                if levels.len() <= d {
                    levels.push(Vec::new());
                }
                #[allow(clippy::cast_possible_truncation)] // < 2**32 nodes
                let parent = if d == 0 {
                    NONE
                } else {
                    (levels[d - 1].len() - 1) as u32
                };
                let freq = if d + 1 == cur.len() { f } else { 0 };
                levels[d].push((cur[d], parent, freq));
            }
            std::mem::swap(&mut prev, &mut cur);
        }
        let n: usize = levels.iter().map(Vec::len).sum();
        let mut label = Vec::with_capacity(n);
        let mut freq = Vec::with_capacity(n);
        let mut first = Vec::with_capacity(n + 1);
        let mut offset = 0usize;
        for (d, level) in levels.iter().enumerate() {
            let next_offset = offset + level.len();
            // Children of this level's node j start at next_offset + (number of
            // next-level entries whose parent is < j).
            let mut counts = vec![0u32; level.len() + 1];
            if let Some(next) = levels.get(d + 1) {
                for &(_, p, _) in next {
                    counts[p as usize + 1] += 1;
                }
            }
            let mut acc = next_offset;
            for &c in &counts[..level.len()] {
                acc += c as usize;
                #[allow(clippy::cast_possible_truncation)]
                first.push(acc as u32);
            }
            for &(c, _, f) in level {
                label.push(c);
                freq.push(f);
            }
            offset = next_offset;
        }
        #[allow(clippy::cast_possible_truncation)]
        first.push(n as u32);
        #[allow(clippy::cast_possible_truncation)]
        let roots = levels.first().map_or(0, Vec::len) as u32;
        Self::with_roots(label, first, freq, roots)
    }

    /// The serialized form, for a trie whose labels are in the BMP and with
    /// fewer than 65,536 distinct frequencies (jieba's default dictionary has
    /// 5,089), little-endian: `b"JBTR"`, u32 version, u32 nodes, u32 roots,
    /// u32 distinct frequencies and those values (u64, increasing, 0 first),
    /// u32 byte length of the child counts, then per node in order its number
    /// of children (unsigned LEB128), its label (u16) and the index of its
    /// frequency (u16). About 5 bytes per node.
    pub(crate) fn to_blob(&self) -> Result<Vec<u8>, BlobError> {
        let n = self.len();
        let mut values: Vec<u64> = self.freq.clone();
        values.push(0);
        values.sort_unstable();
        values.dedup();
        if values.len() > 1 << 16 {
            return Err(BlobError("65,536 distinct frequencies or more"));
        }
        let mut counts = Vec::with_capacity(n + n / 8);
        for x in 0..n {
            let mut k = self.first[x + 1] - self.first[x];
            loop {
                #[allow(clippy::cast_possible_truncation)] // the low 7 bits
                let byte = (k & 0x7f) as u8;
                k >>= 7;
                if k == 0 {
                    counts.push(byte);
                    break;
                }
                counts.push(byte | 0x80);
            }
        }
        let u32_of = |v: usize, what: &'static str| u32::try_from(v).map_err(|_| BlobError(what));
        let mut out = Vec::with_capacity(28 + 8 * values.len() + counts.len() + 4 * n);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&VERSION.to_le_bytes());
        out.extend_from_slice(&u32_of(n, "too many nodes")?.to_le_bytes());
        out.extend_from_slice(&self.roots.to_le_bytes());
        out.extend_from_slice(&u32_of(values.len(), "too many values")?.to_le_bytes());
        for v in &values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out.extend_from_slice(&u32_of(counts.len(), "counts too long")?.to_le_bytes());
        out.extend_from_slice(&counts);
        for &c in &self.label {
            let c =
                u16::try_from(u32::from(c)).map_err(|_| BlobError("a label outside the BMP"))?;
            out.extend_from_slice(&c.to_le_bytes());
        }
        for f in &self.freq {
            let i = values
                .binary_search(f)
                .expect("every frequency is in the table");
            #[allow(clippy::cast_possible_truncation)] // < 65,536
            out.extend_from_slice(&(i as u16).to_le_bytes());
        }
        Ok(out)
    }

    /// Read [`Self::to_blob`]'s output (about 2 ms for jieba's dictionary).
    pub(crate) fn from_blob(b: &[u8]) -> Result<Self, BlobError> {
        struct Reader<'a>(&'a [u8]);
        impl<'a> Reader<'a> {
            fn take(&mut self, k: usize) -> Result<&'a [u8], BlobError> {
                if self.0.len() < k {
                    return Err(BlobError("truncated"));
                }
                let (head, rest) = self.0.split_at(k);
                self.0 = rest;
                Ok(head)
            }
            fn u32(&mut self) -> Result<u32, BlobError> {
                Ok(u32::from_le_bytes(
                    self.take(4)?.try_into().expect("4 bytes"),
                ))
            }
        }
        let mut r = Reader(b);
        if r.take(4)? != MAGIC.as_slice() || r.u32()? != VERSION {
            return Err(BlobError("not a version 2 trie"));
        }
        let n = r.u32()? as usize;
        let roots = r.u32()?;
        let n_values = r.u32()? as usize;
        let values: Vec<u64> = r
            .take(8 * n_values)?
            .chunks_exact(8)
            .map(|w| u64::from_le_bytes(w.try_into().expect("8 bytes")))
            .collect();
        let counts_len = r.u32()? as usize;
        let counts = r.take(counts_len)?;
        let labels = r.take(2 * n)?;
        let freqs = r.take(2 * n)?;
        if !r.0.is_empty() || roots as usize > n {
            return Err(BlobError("wrong length"));
        }
        let mut first = Vec::with_capacity(n + 1);
        let mut acc = u64::from(roots);
        let (mut k, mut shift) = (0u64, 0u32);
        for &byte in counts {
            k |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                #[allow(clippy::cast_possible_truncation)] // checked below
                first.push(acc as u32);
                acc += k;
                (k, shift) = (0, 0);
            } else {
                shift += 7;
                if shift > 28 {
                    return Err(BlobError("a child count too long"));
                }
            }
        }
        if first.len() != n || acc != n as u64 {
            return Err(BlobError("children do not add up to the nodes"));
        }
        #[allow(clippy::cast_possible_truncation)]
        first.push(acc as u32);
        let label = labels
            .chunks_exact(2)
            .map(|w| char::from_u32(u32::from(u16::from_le_bytes([w[0], w[1]]))))
            .collect::<Option<Vec<char>>>()
            .ok_or(BlobError("a surrogate label"))?;
        let freq = freqs
            .chunks_exact(2)
            .map(|w| {
                values
                    .get(usize::from(u16::from_le_bytes([w[0], w[1]])))
                    .copied()
            })
            .collect::<Option<Vec<u64>>>()
            .ok_or(BlobError("a frequency index out of range"))?;
        Ok(Self::with_roots(label, first, freq, roots))
    }

    fn with_roots(label: Vec<char>, first: Vec<u32>, freq: Vec<u64>, roots: u32) -> Self {
        let mut root_bmp = vec![NONE; 0x1_0000].into_boxed_slice();
        for (i, &c) in label[..roots as usize].iter().enumerate() {
            if let Some(slot) = root_bmp.get_mut(c as usize) {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *slot = i as u32;
                }
            }
        }
        Trie {
            label,
            first,
            freq,
            roots,
            root_bmp,
        }
    }

    /// Number of nodes (`len(FREQ)` for this word set).
    pub(crate) fn len(&self) -> usize {
        self.label.len()
    }

    /// The one-character node for `c`.
    #[inline]
    pub(crate) fn root(&self, c: char) -> Option<u32> {
        if let Some(&x) = self.root_bmp.get(c as usize) {
            return (x != NONE).then_some(x);
        }
        let roots = &self.label[..self.roots as usize];
        #[allow(clippy::cast_possible_truncation)]
        roots.binary_search(&c).ok().map(|i| i as u32)
    }

    /// The child of `x` for `c`.
    #[inline]
    pub(crate) fn child(&self, x: u32, c: char) -> Option<u32> {
        let (a, b) = (self.first[x as usize], self.first[x as usize + 1]);
        let kids = &self.label[a as usize..b as usize];
        #[allow(clippy::cast_possible_truncation)]
        kids.binary_search(&c).ok().map(|i| a + i as u32)
    }

    /// The frequency of node `x`.
    #[inline]
    pub(crate) fn freq(&self, x: u32) -> u64 {
        self.freq[x as usize]
    }

    /// The node for `word` (non-empty).
    pub(crate) fn find(&self, word: &str) -> Option<u32> {
        let mut it = word.chars();
        let mut x = self.root(it.next()?)?;
        for c in it {
            x = self.child(x, c)?;
        }
        Some(x)
    }

    /// Every node's prefix and frequency, in code point order.
    pub(crate) fn entries(&self) -> Vec<(String, u64)> {
        let mut out = Vec::with_capacity(self.len());
        let mut prefix = String::new();
        // (node, depth) in reverse order, so the smallest child pops first.
        let mut stack: Vec<(u32, usize)> = (0..self.roots).rev().map(|x| (x, 0)).collect();
        let mut lens: Vec<usize> = Vec::new();
        while let Some((x, depth)) = stack.pop() {
            lens.truncate(depth);
            prefix.truncate(lens.last().copied().unwrap_or(0));
            prefix.push(self.label[x as usize]);
            lens.push(prefix.len());
            out.push((prefix.clone(), self.freq(x)));
            let (a, b) = (self.first[x as usize], self.first[x as usize + 1]);
            stack.extend((a..b).rev().map(|y| (y, depth + 1)));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_trie() {
        let words = [
            ("a", 1u64),
            ("ab", 0),
            ("abc", 5),
            ("b", 2),
            ("bd", 3),
            ("中", 4),
            ("𠀀x", 6),
        ];
        let t = Trie::from_sorted(words.iter().copied());
        // a ab abc b bd 中 𠀀 𠀀x
        assert_eq!(t.len(), 8);
        assert_eq!(t.find("a").map(|x| t.freq(x)), Some(1));
        assert_eq!(t.find("ab").map(|x| t.freq(x)), Some(0));
        assert_eq!(t.find("abc").map(|x| t.freq(x)), Some(5));
        assert_eq!(t.find("abx"), None);
        assert_eq!(t.find("bd").map(|x| t.freq(x)), Some(3));
        assert_eq!(t.find("中").map(|x| t.freq(x)), Some(4));
        assert_eq!(t.find("𠀀").map(|x| t.freq(x)), Some(0));
        assert_eq!(t.find("𠀀x").map(|x| t.freq(x)), Some(6));
        assert_eq!(t.find("c"), None);
        let blob = t.to_blob();
        assert!(blob.is_err(), "the astral label does not fit");
        let t2 = Trie::from_sorted(words[..6].iter().copied());
        let back = Trie::from_blob(&t2.to_blob().unwrap()).unwrap();
        assert_eq!(back.entries(), t2.entries());
        assert_eq!(back.first, t2.first);
        let e = t.entries();
        let keys: Vec<&str> = e.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["a", "ab", "abc", "b", "bd", "中", "𠀀", "𠀀x"]);
    }
}
