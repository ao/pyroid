//! N-gram counting over integer ids (interned words, or code points).
//!
//! sacrebleu counts n-grams as `Counter`s of tuples (BLEU), strings (chrF's
//! character n-grams) or `' '.join`ed words (chrF++'s word n-grams). Only the
//! counts matter, so here an n-gram is its sequence of ids: interned tokens
//! (one vocabulary per segment, built from the references, see [`Vocab`]) or
//! code points. When `n * bits <= 128` the sequence is packed into a `u128`
//! (each id + 1 in a `bits`-wide slot, so n-grams of different lengths never
//! collide and one map holds every order); otherwise it is boxed.
//!
//! Matches are counted the cheap way when the reference counts are owned
//! (one segment, no cache): every hypothesis n-gram takes one from the
//! reference count if any is left ([`Grams::take_matches`]), which adds up
//! to `sum(min(hyp[g], ref[g]))`, sacrebleu's clipped count.

use std::borrow::Borrow;
use std::hash::Hash;

use rustc_hash::FxHashMap;

/// N-gram counts with packed keys, or boxed keys when ids do not fit.
#[derive(Clone, Debug)]
pub enum Grams {
    /// Keys packed by [`pack`] with the given slot width.
    Packed(FxHashMap<u128, u32>, u32),
    /// Keys as id sequences.
    Long(FxHashMap<Box<[u32]>, u32>),
}

/// The slot width that packs n-grams of up to `max_order` ids, each at most
/// `max_id`, into a `u128`; `None` if they do not fit.
///
/// ```
/// use pyroid_core::metrics::sacrebleu::ngrams::packing_bits;
/// assert_eq!(packing_bits(4, 1_000), Some(32));
/// assert_eq!(packing_bits(6, 0x10_FFFF), Some(21)); // chrF's six-character n-grams
/// assert_eq!(packing_bits(7, 0x10_FFFF), None);
/// ```
pub fn packing_bits(max_order: usize, max_id: u32) -> Option<u32> {
    let bits = u32::try_from(128 / max_order.max(1)).map_or(32, |b| b.min(32));
    (bits > 0 && u64::from(max_id) + 1 < 1_u64 << bits).then_some(bits)
}

/// `ids` packed into `bits`-wide slots (id + 1 each, first id lowest).
#[inline]
fn pack(ids: &[u32], bits: u32) -> u128 {
    ids.iter()
        .rev()
        .fold(0, |key, &id| (key << bits) | (u128::from(id) + 1))
}

/// The length of the n-gram a packed key holds (its highest non-empty slot).
#[inline]
fn packed_order(key: u128, bits: u32) -> usize {
    (128 - key.leading_zeros()).div_ceil(bits) as usize
}

/// Total windows of orders `1..=max_order` in a sequence of `len`.
pub fn all_windows(len: usize, max_order: usize) -> usize {
    (1..=max_order).map(|n| (len + 1).saturating_sub(n)).sum()
}

impl Grams {
    /// An empty map for n-grams of up to `max_order` ids at most `max_id`,
    /// with room for `capacity` distinct n-grams.
    pub fn with_capacity(max_order: usize, max_id: u32, capacity: usize) -> Self {
        let hasher = rustc_hash::FxBuildHasher;
        match packing_bits(max_order, max_id) {
            Some(bits) => Self::Packed(FxHashMap::with_capacity_and_hasher(capacity, hasher), bits),
            None => Self::Long(FxHashMap::with_capacity_and_hasher(capacity, hasher)),
        }
    }

    /// An empty map for n-grams of up to `max_order` ids at most `max_id`.
    pub fn new(max_order: usize, max_id: u32) -> Self {
        Self::with_capacity(max_order, max_id, 0)
    }

    /// The n-gram counts of orders `1..=max_order` of `ids`.
    pub fn count(ids: &[u32], max_order: usize, max_id: u32) -> Self {
        let mut g = Self::with_capacity(max_order, max_id, all_windows(ids.len(), max_order));
        for n in 1..=max_order {
            g.add_windows(ids, n);
        }
        g
    }

    /// No n-grams at all?
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Packed(m, _) => m.is_empty(),
            Self::Long(m) => m.is_empty(),
        }
    }

    /// Counts every window of `n` ids of `ids`.
    pub fn add_windows(&mut self, ids: &[u32], n: usize) {
        if n == 0 || ids.len() < n {
            return;
        }
        match self {
            Self::Packed(m, bits) => {
                for w in ids.windows(n) {
                    *m.entry(pack(w, *bits)).or_insert(0) += 1;
                }
            }
            Self::Long(m) => {
                for w in ids.windows(n) {
                    if let Some(c) = m.get_mut(w) {
                        *c += 1;
                    } else {
                        m.insert(w.into(), 1);
                    }
                }
            }
        }
    }

    /// For every n-gram of `other`, keeps the larger count (sacrebleu's
    /// multi-reference merge, `ngrams[g] = max(ngrams[g], count)`).
    /// `other` must have the same key layout.
    pub fn merge_max(&mut self, other: Self) {
        match (self, other) {
            (Self::Packed(a, _), Self::Packed(b, _)) => {
                for (k, c) in b {
                    let e = a.entry(k).or_insert(0);
                    *e = (*e).max(c);
                }
            }
            (Self::Long(a), Self::Long(b)) => {
                for (k, c) in b {
                    let e = a.entry(k).or_insert(0);
                    *e = (*e).max(c);
                }
            }
            _ => unreachable!("n-gram maps of one segment share their key layout"),
        }
    }

    /// The clipped matches of the `n`-grams of `hyp` against these counts,
    /// `sum(min(hyp[g], self[g]))`, using the counts up: each hypothesis
    /// n-gram takes one if any is left.
    pub fn take_matches(&mut self, hyp: &[u32], n: usize) -> u64 {
        if n == 0 || hyp.len() < n {
            return 0;
        }
        let mut matches = 0;
        match self {
            Self::Packed(m, bits) => {
                for w in hyp.windows(n) {
                    if let Some(c) = m.get_mut(&pack(w, *bits)) {
                        if *c > 0 {
                            *c -= 1;
                            matches += 1;
                        }
                    }
                }
            }
            Self::Long(m) => {
                for w in hyp.windows(n) {
                    if let Some(c) = m.get_mut(w) {
                        if *c > 0 {
                            *c -= 1;
                            matches += 1;
                        }
                    }
                }
            }
        }
        matches
    }

    /// Per order, `sum(min(self[g], other[g]))` over the n-grams of that
    /// order: `out[n - 1]` gets the clipped matches of the `n`-grams.
    /// `other` must have the same key layout.
    pub fn overlap_by_order(&self, other: &Self, out: &mut [u64]) {
        out.fill(0);
        match (self, other) {
            (Self::Packed(a, bits), Self::Packed(b, _)) => {
                let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
                for (k, &c) in small {
                    if let Some(&d) = large.get(k) {
                        out[packed_order(*k, *bits) - 1] += u64::from(c.min(d));
                    }
                }
            }
            (Self::Long(a), Self::Long(b)) => {
                let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
                for (k, &c) in small {
                    if let Some(&d) = large.get(k) {
                        out[k.len() - 1] += u64::from(c.min(d));
                    }
                }
            }
            _ => unreachable!("n-gram maps of one segment share their key layout"),
        }
    }
}

/// Number of windows of `n` items in a sequence of `len` (`len - n + 1`,
/// or 0).
#[inline]
pub fn windows(len: usize, n: usize) -> u64 {
    (len + 1).saturating_sub(n) as u64
}

/// A segment's token vocabulary: the references' tokens get ids in order of
/// appearance; a hypothesis token that no reference has gets
/// [`Vocab::unknown`], which never matches.
#[derive(Clone, Debug, Default)]
pub struct Vocab<V: Hash + Eq> {
    ids: FxHashMap<V, u32>,
}

impl<V: Hash + Eq + Borrow<str>> Vocab<V> {
    /// An empty vocabulary with room for `capacity` tokens.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ids: FxHashMap::with_capacity_and_hasher(capacity, rustc_hash::FxBuildHasher),
        }
    }

    /// The id of `token`, adding it if new.
    pub fn intern(&mut self, token: V) -> u32 {
        let next = u32::try_from(self.ids.len()).expect("fewer than 2^32 distinct tokens");
        *self.ids.entry(token).or_insert(next)
    }

    /// The id of a hypothesis token.
    pub fn lookup(&self, token: &str) -> u32 {
        self.ids
            .get(token)
            .copied()
            .unwrap_or_else(|| self.unknown())
    }

    /// The id of tokens no reference has (the largest id in use).
    pub fn unknown(&self) -> u32 {
        u32::try_from(self.ids.len()).expect("fewer than 2^32 distinct tokens")
    }

    /// The same vocabulary with owned keys.
    pub fn to_owned(&self) -> Vocab<Box<str>> {
        Vocab {
            ids: self
                .ids
                .iter()
                .map(|(k, &v)| (Box::from(k.borrow()), v))
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn long(ids: &[u32], max_order: usize) -> Grams {
        let mut g = Grams::Long(FxHashMap::default());
        for n in 1..=max_order {
            g.add_windows(ids, n);
        }
        g
    }

    #[test]
    fn packing_distinguishes_lengths() {
        // (0) vs (0, 0): id + 1 in every slot.
        assert_ne!(pack(&[0], 32), pack(&[0, 0], 32));
        assert_ne!(pack(&[1, 0], 32), pack(&[0, 1], 32));
        assert_eq!(packed_order(pack(&[0, 0, 0], 21), 21), 3);
        assert_eq!(packed_order(pack(&[0x10_FFFF; 6], 21), 21), 6);
        assert_eq!(packed_order(pack(&[7], 32), 32), 1);
        assert_eq!(packing_bits(1, u32::MAX - 1), Some(32));
        assert_eq!(packing_bits(1, u32::MAX), None);
        assert_eq!(packing_bits(200, 0), None);
        assert_eq!(all_windows(3, 4), 3 + 2 + 1);
    }

    #[test]
    fn packed_and_long_agree() {
        let hyp = [0, 1, 2, 0, 1, 2, 3];
        let refs = [0, 1, 2, 3, 3, 0, 1];
        // 1-grams: 0,1,2 twice in hyp; refs have 0 twice, 1 twice, 2 once, 3 twice.
        // 2-grams: (0,1) x2, (1,2), (2,3); 3-grams: (0,1,2), (1,2,3); 4-grams: (0,1,2,3).
        let want = [2 + 2 + 1 + 1, 4, 2, 1];
        for (h, r) in [
            (Grams::count(&hyp, 4, 9), Grams::count(&refs, 4, 9)),
            (long(&hyp, 4), long(&refs, 4)),
        ] {
            let mut by_order = [0; 4];
            h.overlap_by_order(&r, &mut by_order);
            assert_eq!(by_order, want);
            r.overlap_by_order(&h, &mut by_order);
            assert_eq!(by_order, want);
            let mut taken = r.clone();
            let got: Vec<u64> = (1..=4).map(|n| taken.take_matches(&hyp, n)).collect();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn merge_max_keeps_the_larger_count() {
        let mut a = Grams::count(&[0, 0, 1], 1, 5);
        a.merge_max(Grams::count(&[0, 1, 1, 1], 1, 5));
        assert_eq!(a.take_matches(&[0, 0, 0, 1, 1, 1, 1], 1), 2 + 3);
        let mut b = long(&[0, 0, 1], 1);
        b.merge_max(long(&[0, 1, 1, 1], 1));
        assert_eq!(b.take_matches(&[0, 0, 0, 1, 1, 1, 1], 1), 2 + 3);
        assert!(!b.is_empty() && Grams::new(2, 3).is_empty());
    }

    #[test]
    fn vocab_unknown_never_matches() {
        let mut v: Vocab<&str> = Vocab::with_capacity(4);
        assert_eq!(v.intern("a"), 0);
        assert_eq!(v.intern("b"), 1);
        assert_eq!(v.intern("a"), 0);
        assert_eq!(v.lookup("zzz"), 2);
        assert_eq!(v.to_owned().lookup("b"), 1);
        assert_eq!(windows(3, 1), 3);
        assert_eq!(windows(3, 4), 0);
        assert_eq!(windows(0, 1), 0);
    }
}
