// Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
//! `DetectorFactory.word_lang_prob_map`: n-gram -> per-language probabilities.
//!
//! langdetect keeps one Python list per n-gram, created with the length of the profile set
//! being loaded (`[0.0] * langsize`) and indexed by the profile's position in that set. Rows
//! made by different `load_profile` calls can have different lengths, and detection raises
//! `IndexError` on a row shorter than the language list. [`Builder`] keeps each row's length
//! and its explicitly set entries (most rows have two or three), and [`Frozen`] is the
//! read-only form detection uses.

use rustc_hash::FxHashMap;

use super::chars::MAX_CP;

/// A 1-3 code point n-gram packed into one integer: `(c0 + 1) << 42 | (c1 + 1) << 21 | ...`.
#[inline]
pub fn pack(gram: &[u32]) -> Option<u64> {
    if gram.is_empty() || gram.len() > 3 {
        return None;
    }
    let mut key = 0u64;
    for &c in gram {
        debug_assert!(c < MAX_CP);
        key = (key << 21) | u64::from(c + 1);
    }
    Some(key)
}

/// The code points of a packed key.
pub fn unpack(key: u64) -> Vec<u32> {
    let mut out = Vec::with_capacity(3);
    let mut k = key;
    while k != 0 {
        // 21-bit fields.
        #[allow(clippy::cast_possible_truncation)]
        out.push((k & 0x1F_FFFF) as u32 - 1);
        k >>= 21;
    }
    out.reverse();
    out
}

/// An index outside a row (`list assignment index out of range`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutOfRange;

#[derive(Clone, Debug)]
struct Row {
    /// The list's length (`langsize` when the row was made).
    len: usize,
    /// Explicitly set `(index, value)`, sorted by index; others are `0.0`.
    entries: Vec<(u32, f64)>,
}

/// The mutable map, filled by `add_profile`.
#[derive(Clone, Debug, Default)]
pub struct Builder {
    /// Keys of 1-3 code points (the only ones detection can look up).
    index: FxHashMap<u64, u32>,
    /// Other keys (empty, or longer than 3): kept so the map's contents and order stay
    /// langdetect's.
    other: FxHashMap<Box<[u32]>, u32>,
    rows: Vec<Row>,
    /// The key of each row, in insertion order (`None` for a key the bindings keep).
    keys: Vec<Key>,
}

/// A row's key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Key {
    /// A str key of 1-3 code points.
    Packed(u64),
    /// A str key of another length.
    Long(Box<[u32]>),
    /// A key that is not a str; the bindings hold the object, this is its number.
    Foreign(usize),
}

impl Builder {
    /// Number of rows (`len(word_lang_prob_map)`).
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// No rows.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// The row of a str key, if there is one.
    pub fn find(&self, word: &[u32]) -> Option<usize> {
        match pack(word) {
            Some(k) => self.index.get(&k).map(|&r| r as usize),
            None => self.other.get(word).map(|&r| r as usize),
        }
    }

    /// The row of a str key, made with `len` zeros if there is none (`[0.0] * langsize`).
    pub fn ensure(&mut self, word: &[u32], len: usize) -> usize {
        if let Some(r) = self.find(word) {
            return r;
        }
        let r = self.push_row(len);
        // Fewer than 2^32 rows (one per distinct n-gram).
        #[allow(clippy::cast_possible_truncation)]
        let r32 = r as u32;
        if let Some(k) = pack(word) {
            self.index.insert(k, r32);
            self.keys.push(Key::Packed(k));
        } else {
            self.other.insert(word.into(), r32);
            self.keys.push(Key::Long(word.into()));
        }
        r
    }

    /// A new row for a non-str key the bindings track; returns the row.
    pub fn push_foreign(&mut self, len: usize, id: usize) -> usize {
        let r = self.push_row(len);
        self.keys.push(Key::Foreign(id));
        r
    }

    fn push_row(&mut self, len: usize) -> usize {
        self.rows.push(Row {
            len,
            entries: Vec::new(),
        });
        self.rows.len() - 1
    }

    /// The length of row `row`.
    pub fn row_len(&self, row: usize) -> usize {
        self.rows[row].len
    }

    /// `row[index] = value`, with Python's negative indices.
    pub fn set(&mut self, row: usize, index: i64, value: f64) -> Result<(), OutOfRange> {
        let r = &mut self.rows[row];
        let len = i64::try_from(r.len).map_err(|_| OutOfRange)?;
        let i = if index < 0 { index + len } else { index };
        if i < 0 || i >= len {
            return Err(OutOfRange);
        }
        // 0 <= i < len <= usize::MAX.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let i = i as u32;
        match r.entries.binary_search_by_key(&i, |e| e.0) {
            Ok(pos) => r.entries[pos].1 = value,
            Err(pos) => r.entries.insert(pos, (i, value)),
        }
        Ok(())
    }

    /// Row `row` as a dense list of its length.
    pub fn dense(&self, row: usize) -> Vec<f64> {
        let r = &self.rows[row];
        let mut out = vec![0.0; r.len];
        for &(i, v) in &r.entries {
            out[i as usize] = v;
        }
        out
    }

    /// The keys in insertion order.
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    /// The read-only form detection uses.
    pub fn freeze(&self) -> Frozen {
        let mut start = Vec::with_capacity(self.rows.len() + 1);
        let mut lang = Vec::new();
        let mut prob = Vec::new();
        let mut len = Vec::with_capacity(self.rows.len());
        start.push(0);
        for r in &self.rows {
            for &(i, v) in &r.entries {
                lang.push(i);
                prob.push(v);
            }
            start.push(lang.len());
            len.push(r.len);
        }
        let keys = self
            .keys
            .iter()
            .map(|k| match k {
                Key::Packed(p) => *p,
                _ => 0,
            })
            .collect();
        let mut uni = vec![0u32; 0x1_0000].into_boxed_slice();
        for (&key, &row) in &self.index {
            if key < 0x1_0001 {
                // A 1-gram of a BMP character: key = cp + 1.
                uni[(key - 1) as usize] = row + 1;
            }
        }
        let nonneg_finite = prob.iter().all(|p| p.is_finite() && *p >= 0.0);
        let max_prob = prob.iter().copied().fold(0.0f64, f64::max);
        Frozen {
            index: self.index.clone(),
            uni,
            len,
            start,
            lang,
            prob,
            keys,
            nonneg_finite,
            max_prob,
        }
    }
}

/// The map as detection reads it.
#[derive(Debug, Default)]
pub struct Frozen {
    index: FxHashMap<u64, u32>,
    /// Row + 1 of each BMP character's 1-gram (0: none).
    uni: Box<[u32]>,
    len: Vec<usize>,
    start: Vec<usize>,
    lang: Vec<u32>,
    prob: Vec<f64>,
    /// The packed key of each row (0 for keys detection never looks up).
    keys: Vec<u64>,
    /// Every set value is finite and not negative.
    nonneg_finite: bool,
    /// The largest set value (0.0 if none).
    max_prob: f64,
}

/// One row as detection reads it.
#[derive(Clone, Copy, Debug)]
pub struct RowRef<'a> {
    /// The row's length.
    pub len: usize,
    /// Indices of the explicitly set entries.
    pub lang: &'a [u32],
    /// Their values.
    pub prob: &'a [f64],
}

impl Frozen {
    /// The row of a packed 1-3-gram.
    #[inline]
    pub fn lookup(&self, key: u64) -> Option<u32> {
        self.index.get(&key).copied()
    }

    /// The row of an n-gram of 1-3 code points.
    #[inline]
    pub fn lookup_gram(&self, gram: &[u32]) -> Option<u32> {
        if let [c] = gram {
            if let Some(&r) = self.uni.get(*c as usize) {
                return r.checked_sub(1);
            }
        }
        pack(gram).and_then(|k| self.lookup(k))
    }

    /// Every set value is finite and not negative.
    pub fn probs_nonneg_finite(&self) -> bool {
        self.nonneg_finite
    }

    /// The largest set value.
    pub fn max_prob(&self) -> f64 {
        self.max_prob
    }

    /// Row `row`.
    #[inline]
    pub fn row(&self, row: u32) -> RowRef<'_> {
        let r = row as usize;
        let (a, b) = (self.start[r], self.start[r + 1]);
        RowRef {
            len: self.len[r],
            lang: &self.lang[a..b],
            prob: &self.prob[a..b],
        }
    }

    /// The code points of the n-gram of row `row`.
    pub fn word(&self, row: u32) -> Vec<u32> {
        unpack(self.keys[row as usize])
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.len.len()
    }

    /// No rows.
    pub fn is_empty(&self) -> bool {
        self.len.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trips() {
        for g in [&[0u32][..], &[0x20, 0x61], &[0x0010_FFFF, 0, 0xD800]] {
            assert_eq!(unpack(pack(g).unwrap()), g);
        }
        assert_ne!(pack(&[0]), pack(&[0, 0]));
        assert_eq!(pack(&[]), None);
        assert_eq!(pack(&[1, 2, 3, 4]), None);
    }

    #[test]
    fn rows_keep_their_length() {
        let mut b = Builder::default();
        let r = b.ensure(&[0x61], 3);
        assert_eq!(b.ensure(&[0x61], 5), r);
        assert_eq!(b.set(r, 2, 0.5), Ok(()));
        assert_eq!(b.set(r, -1, 0.25), Ok(()));
        assert_eq!(b.set(r, 3, 1.0), Err(OutOfRange));
        assert_eq!(b.set(r, -4, 1.0), Err(OutOfRange));
        assert_eq!(b.dense(r), vec![0.0, 0.0, 0.25]);
        let f = b.freeze();
        let row = f.row(f.lookup(pack(&[0x61]).unwrap()).unwrap());
        assert_eq!((row.len, row.lang, row.prob), (3, &[2u32][..], &[0.25][..]));
    }
}
