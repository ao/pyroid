//! CPython's `hash()` of str and of tuples of str, and the iteration order of a `set` built
//! from a list, so `RepeatingTopNGramsFilter`'s tie-break (the first tied n-gram in
//! `set(ngrams)` order) can be resolved without building the set in Python.
//!
//! This is CPython 3.11-3.15 on 64-bit builds with `sys.hash_info.algorithm == "siphash13"`
//! and no small-string cutoff (`Python/pyhash.c`, `Objects/tupleobject.c`,
//! `Objects/setobject.c`; the free-threaded builds only add locking). The key is the
//! process's `_Py_HashSecret`, which only Python can read; `pyroid.nemo` checks these hashes
//! and the set order against the running interpreter before it uses them.

// `as` between i64, u64 and usize is C's reinterpretation of `Py_hash_t` as `Py_uhash_t` and
// `size_t` (64-bit builds only), which these functions reproduce.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]

/// The siphash key of CPython's `_Py_HashSecret` (its first 16 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HashKey {
    k0: u64,
    k1: u64,
}

impl HashKey {
    /// The key from the first 16 bytes of `_Py_HashSecret` (`pysiphash` reads both halves
    /// little-endian). `None` if there are fewer than 16 bytes.
    pub fn from_secret(secret: &[u8]) -> Option<Self> {
        let k0 = u64::from_le_bytes(secret.get(..8)?.try_into().ok()?);
        let k1 = u64::from_le_bytes(secret.get(8..16)?.try_into().ok()?);
        Some(Self { k0, k1 })
    }
}

#[inline]
fn sip_round(v: &mut [u64; 4]) {
    v[0] = v[0].wrapping_add(v[1]);
    v[2] = v[2].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(13) ^ v[0];
    v[3] = v[3].rotate_left(16) ^ v[2];
    v[0] = v[0].rotate_left(32);
    v[2] = v[2].wrapping_add(v[1]);
    v[0] = v[0].wrapping_add(v[3]);
    v[1] = v[1].rotate_left(17) ^ v[2];
    v[3] = v[3].rotate_left(21) ^ v[0];
    v[2] = v[2].rotate_left(32);
}

/// `siphash13` (`Python/pyhash.c`): SipHash-1-3, 64-bit output.
fn siphash13(key: HashKey, data: &[u8]) -> u64 {
    let mut v = [
        key.k0 ^ 0x736f_6d65_7073_6575,
        key.k1 ^ 0x646f_7261_6e64_6f6d,
        key.k0 ^ 0x6c79_6765_6e65_7261,
        key.k1 ^ 0x7465_6462_7974_6573,
    ];
    let mut blocks = data.chunks_exact(8);
    for block in &mut blocks {
        let m = u64::from_le_bytes(block.try_into().expect("8 bytes"));
        v[3] ^= m;
        sip_round(&mut v);
        v[0] ^= m;
    }
    let mut tail = [0u8; 8];
    let rest = blocks.remainder();
    tail[..rest.len()].copy_from_slice(rest);
    let b = ((data.len() as u64) << 56) | u64::from_le_bytes(tail);
    v[3] ^= b;
    sip_round(&mut v);
    v[0] ^= b;
    v[2] ^= 0xff;
    sip_round(&mut v);
    sip_round(&mut v);
    sip_round(&mut v);
    (v[0] ^ v[1]) ^ (v[2] ^ v[3])
}

/// `_Py_HashBytes`: 0 for no bytes, -1 becomes -2.
fn hash_bytes(key: HashKey, data: &[u8]) -> i64 {
    if data.is_empty() {
        return 0;
    }
    match siphash13(key, data) as i64 {
        -1 => -2,
        x => x,
    }
}

/// `hash(s)`: siphash13 over the str's PEP 393 form (one byte per character if all are below
/// U+0100, two below U+10000, else four; in the machine's byte order). `buf` is scratch.
pub fn str_hash(key: HashKey, s: &str, buf: &mut Vec<u8>) -> i64 {
    if s.is_ascii() {
        return hash_bytes(key, s.as_bytes());
    }
    let max = s.chars().map(u32::from).max().unwrap_or(0);
    buf.clear();
    if max < 0x100 {
        // Every character fits a byte (`as u8` keeps it whole).
        buf.extend(s.chars().map(|c| u32::from(c).to_le_bytes()[0]));
    } else if max < 0x1_0000 {
        for c in s.chars() {
            let unit = u16::try_from(u32::from(c)).expect("below U+10000");
            buf.extend_from_slice(&unit.to_ne_bytes());
        }
    } else {
        for c in s.chars() {
            buf.extend_from_slice(&u32::from(c).to_ne_bytes());
        }
    }
    hash_bytes(key, buf)
}

const XXPRIME_1: u64 = 11_400_714_785_074_694_791;
const XXPRIME_2: u64 = 14_029_467_366_897_019_727;
const XXPRIME_5: u64 = 2_870_177_450_012_600_261;

/// `hash(t)` for a tuple whose items hash to `lanes` (`tuplehash`, CPython 3.8+).
pub fn tuple_hash(lanes: impl ExactSizeIterator<Item = i64>) -> i64 {
    let len = lanes.len() as u64;
    let mut acc = XXPRIME_5;
    for lane in lanes {
        acc = acc.wrapping_add((lane as u64).wrapping_mul(XXPRIME_2));
        acc = acc.rotate_left(31);
        acc = acc.wrapping_mul(XXPRIME_1);
    }
    // 0x0035_D373 is 3527539, the constant CPython keeps `hash(())` with.
    acc = acc.wrapping_add(len ^ (XXPRIME_5 ^ 0x0035_D373));
    if acc == u64::MAX {
        1_546_275_796
    } else {
        acc as i64
    }
}

const MINSIZE: usize = 8;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const EMPTY: u32 = u32::MAX;

/// A CPython `set` reduced to what decides its iteration order: each slot's hash and item
/// (an index chosen by the caller). Items are only ever added, as `set(list)` does.
#[derive(Clone, Debug)]
pub struct PySet {
    hashes: Vec<i64>,
    items: Vec<u32>,
    fill: usize,
}

impl Default for PySet {
    fn default() -> Self {
        Self::new()
    }
}

impl PySet {
    /// An empty set (`PySet_MINSIZE` slots).
    pub fn new() -> Self {
        Self {
            hashes: vec![0; MINSIZE],
            items: vec![EMPTY; MINSIZE],
            fill: 0,
        }
    }

    fn mask(&self) -> usize {
        self.items.len() - 1
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.fill
    }

    /// True if the set has no items.
    pub fn is_empty(&self) -> bool {
        self.fill == 0
    }

    /// `set_add_entry`: add `item` with `hash` unless an equal one (same hash and
    /// `same(existing, item)`) is there. True if it was added. `item` must be below
    /// `u32::MAX`.
    pub fn add(&mut self, item: u32, hash: i64, same: impl Fn(u32, u32) -> bool) -> bool {
        debug_assert!(item != EMPTY);
        let mask = self.mask();
        let mut perturb = hash as u64 as usize;
        let mut i = perturb & mask;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                let at = self.items[j];
                if at == EMPTY {
                    self.items[j] = item;
                    self.hashes[j] = hash;
                    self.fill += 1;
                    // `used` equals `fill`: nothing is ever removed.
                    if self.fill * 5 >= mask * 3 {
                        let used = self.fill;
                        self.resize(if used > 50_000 { used * 2 } else { used * 4 });
                    }
                    return true;
                }
                if self.hashes[j] == hash && same(at, item) {
                    return false;
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
        }
    }

    /// `set_table_resize`: the smallest power of two above `min_used` slots, refilled with
    /// `set_insert_clean` in the old table's slot order.
    fn resize(&mut self, min_used: usize) {
        let mut size = MINSIZE;
        while size <= min_used {
            size <<= 1;
        }
        let old_hashes = std::mem::replace(&mut self.hashes, vec![0; size]);
        let old_items = std::mem::replace(&mut self.items, vec![EMPTY; size]);
        for (&item, &hash) in old_items.iter().zip(&old_hashes) {
            if item != EMPTY {
                self.insert_clean(item, hash);
            }
        }
    }

    /// `set_insert_clean`: put an item known to be absent in the first empty slot of its
    /// probe sequence.
    fn insert_clean(&mut self, item: u32, hash: i64) {
        let mask = self.mask();
        let mut perturb = hash as u64 as usize;
        let mut i = perturb & mask;
        loop {
            let probes = if i + LINEAR_PROBES <= mask {
                LINEAR_PROBES
            } else {
                0
            };
            for j in i..=i + probes {
                if self.items[j] == EMPTY {
                    self.items[j] = item;
                    self.hashes[j] = hash;
                    return;
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
        }
    }

    /// The items in iteration order (slot order).
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.items.iter().copied().filter(|&x| x != EMPTY)
    }
}

/// The first empty slot of `hash`'s probe sequence (`set_add_entry` and `set_insert_clean`
/// probe the same slots).
#[inline]
fn place(slots: &mut [u32], item: u32, hash: i64) {
    let mask = slots.len() - 1;
    let mut perturb = hash as u64 as usize;
    let mut i = perturb & mask;
    loop {
        let probes = if i + LINEAR_PROBES <= mask {
            LINEAR_PROBES
        } else {
            0
        };
        if let Some(slot) = slots[i..=i + probes].iter_mut().find(|x| **x == EMPTY) {
            *slot = item;
            return;
        }
        perturb >>= PERTURB_SHIFT;
        i = i.wrapping_mul(5).wrapping_add(1).wrapping_add(perturb) & mask;
    }
}

/// [`PySet`] for items known to be pairwise distinct, as `set()` meets a document's distinct
/// n-grams in order of first occurrence. `set_add_entry` then stops at the first empty slot
/// of the probe sequence (an equal hash is a collision, never a match), so the slots hold
/// only items and each item's hash is kept once, by item. Items are numbered from 0 in the
/// order they are added.
#[derive(Clone, Debug)]
pub struct DistinctSet {
    slots: Vec<u32>,
    hashes: Vec<i64>,
}

impl Default for DistinctSet {
    fn default() -> Self {
        Self {
            slots: vec![EMPTY; MINSIZE],
            hashes: Vec::new(),
        }
    }
}

impl DistinctSet {
    /// Empty it, keeping its buffers.
    pub fn clear(&mut self) {
        self.slots.clear();
        self.slots.resize(MINSIZE, EMPTY);
        self.hashes.clear();
    }

    /// Add the next item, which has `hash` and equals none of the items before it.
    pub fn add(&mut self, hash: i64) {
        let item = u32::try_from(self.hashes.len()).expect("fewer than 2**32 items");
        self.hashes.push(hash);
        let mask = self.slots.len() - 1;
        place(&mut self.slots, item, hash);
        let used = self.hashes.len();
        if used * 5 >= mask * 3 {
            let mut size = MINSIZE;
            let min_used = if used > 50_000 { used * 2 } else { used * 4 };
            while size <= min_used {
                size <<= 1;
            }
            let old = std::mem::replace(&mut self.slots, vec![EMPTY; size]);
            for item in old.into_iter().filter(|&x| x != EMPTY) {
                place(&mut self.slots, item, self.hashes[item as usize]);
            }
        }
    }

    /// The items in iteration order (slot order).
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots.iter().copied().filter(|&x| x != EMPTY)
    }
}

/// The order `list(set(xs))` gives, for items with `hashes` (in list order) that are equal
/// when `same(a, b)`: the index of each distinct item's first occurrence.
pub fn set_order(hashes: &[i64], same: impl Fn(usize, usize) -> bool) -> Vec<usize> {
    let mut set = PySet::new();
    for (i, &h) in hashes.iter().enumerate() {
        let item = u32::try_from(i).expect("fewer than 2**32 items");
        set.add(item, h, |a, b| same(a as usize, b as usize));
    }
    set.iter().map(|x| x as usize).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ZERO: HashKey = HashKey { k0: 0, k1: 0 };

    #[test]
    fn str_and_tuple_hashes_with_seed_0() {
        // `PYTHONHASHSEED=0 python -c 'print(hash(...))'` (the key is then all zeros).
        let mut buf = Vec::new();
        let cases: [(&str, i64); 7] = [
            ("", 0),
            ("a", 4_644_417_185_603_328_019),
            ("abcdefgh", 4_574_395_652_268_504_554),
            ("abcdefghi", -532_774_252_720_507_163),
            ("\u{e9}", 6_047_309_291_227_476_195),
            ("\u{4e2d}\u{6587}", 501_833_946_989_431_548),
            ("\u{1f600}x", -8_926_728_262_118_538_918),
        ];
        for (s, want) in cases {
            assert_eq!(str_hash(ZERO, s, &mut buf), want, "{s:?}");
        }
        let a = str_hash(ZERO, "a", &mut buf);
        let b = str_hash(ZERO, "b", &mut buf);
        assert_eq!(tuple_hash([a, b].into_iter()), 7_372_137_184_884_179_083);
        assert_eq!(tuple_hash(std::iter::empty()), 5_740_354_900_026_072_187);
        assert_eq!(tuple_hash([0].into_iter()), -8_753_497_827_991_233_192);
        assert_eq!(HashKey::from_secret(&[0; 24]), Some(ZERO));
        assert_eq!(HashKey::from_secret(&[0; 15]), None);
    }

    #[test]
    fn small_ints() {
        // `list(set([...]))` for ints, which hash to themselves: (5, 13) share slot 5 of 8.
        let xs: [i64; 6] = [5, 13, 3, 5, 0, -2];
        let order = set_order(&xs, |a, b| xs[a] == xs[b]);
        let got: Vec<i64> = order.iter().map(|&i| xs[i]).collect();
        // Five items fill 8 slots past 3/5: the table grows to 32 (mask 31) on the fifth.
        assert_eq!(got, [0, 3, 5, 13, -2]);
    }

    #[test]
    fn distinct_set_is_a_pyset_of_distinct_items() {
        let mut state = 99u64;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            state
        };
        let mut set = DistinctSet::default();
        for size in [0usize, 1, 4, 5, 19, 20, 77, 78, 300, 4916, 78_643, 100_000] {
            // Random hashes, some equal (collisions of distinct items) and some sharing slots.
            let hashes: Vec<i64> = (0..size)
                .map(|i| match next() % 4 {
                    0 if i > 0 => (next() % 64) as i64,
                    1 => ((next() >> 40) << 20) as i64,
                    _ => next() as i64,
                })
                .collect();
            set.clear();
            let mut py = PySet::new();
            for (i, &h) in hashes.iter().enumerate() {
                set.add(h);
                py.add(u32::try_from(i).unwrap(), h, |_, _| false);
            }
            assert!(set.iter().eq(py.iter()), "{size}");
        }
    }

    #[test]
    fn collisions_keep_both() {
        let hashes = [7, 7, 7, 15];
        let order = set_order(&hashes, |a, b| a == b);
        assert_eq!(order.len(), 4);
        let order = set_order(&hashes, |_, _| true);
        // 15 probes past slot 7 (taken by item 0) to slot 4.
        assert_eq!(order, [3, 0]);
    }
}
