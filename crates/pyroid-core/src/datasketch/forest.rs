// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! The sorted-prefix index behind `MinHashLSHForest`.
//!
//! datasketch keeps `keys` (a dict from each key to its `l` band keys `H`),
//! `l` `defaultdict(list)` tables from `H` to the keys added with it (in
//! order), and `sorted_hashtables`: each table's band keys, sorted, as of the
//! last `index()`. A query walks the sorted keys whose first bytes match the
//! query's prefix and yields the bucket members, for prefix lengths `k` down
//! to 1, stopping as soon as it has `k` distinct keys; [`ForestIndex::query`]
//! visits buckets and members in exactly that order, so the early exit keeps
//! the same keys. Keys added after `index()` are found through a band key
//! that was indexed, as in datasketch, because buckets are live.
//!
//! Ids, their `keys` entries and reference counts work as in
//! [`super::lsh`]; forest buckets are lists, so a key added again after
//! `del forest.keys[key]` appears twice in a bucket, as in datasketch.

use std::sync::Arc;

use rustc_hash::{FxHashMap, FxHashSet};

use super::lsh::{BandKey, CapacityError, FormatError, Ids, KIND_FOREST, Reader, Slots, Writer};

type Table = FxHashMap<BandKey, Ids>;

/// The `MinHashLSHForest` index over ids.
#[derive(Clone, Debug)]
pub struct ForestIndex {
    slots: Slots,
    tables: Vec<Table>,
    sorted: Vec<Vec<BandKey>>,
}

/// `h[:size]` (a Python slice: shorter keys stay whole).
fn prefix(h: &[u8], size: usize) -> &[u8] {
    &h[..size.min(h.len())]
}

impl ForestIndex {
    /// An empty forest with `l` prefix trees.
    pub fn new(l: usize) -> Self {
        Self {
            slots: Slots::default(),
            tables: vec![Table::default(); l],
            sorted: vec![Vec::new(); l],
        }
    }

    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }

    /// `len(forest.keys)`.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// No keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// datasketch's `is_empty()`: some sorted table is empty (nothing indexed).
    pub fn has_empty_table(&self) -> bool {
        self.sorted.iter().any(Vec::is_empty)
    }

    /// A new id, alive and unused: [`Self::add`] it or [`Self::release`] it.
    pub fn alloc(&mut self) -> Result<u32, CapacityError> {
        self.slots.alloc()
    }

    /// Free `id` if it has no `keys` entry and is in no bucket; true if freed.
    pub fn release(&mut self, id: u32) -> bool {
        self.slots.release(id)
    }

    pub fn is_alive(&self, id: u32) -> bool {
        self.slots.is_alive(id)
    }

    /// `key in forest.keys`.
    pub fn contains(&self, id: u32) -> bool {
        self.slots.hs(id).is_some()
    }

    /// `forest.keys[key]`.
    pub fn keys_of(&self, id: u32) -> Option<&[BandKey]> {
        self.slots.hs(id)
    }

    /// The ids with a `keys` entry, in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots.ids()
    }

    /// `add` after its checks: `keys[key] = hs`, then `hashtable[H].append(key)`
    /// for `zip(hs, hashtables)`.
    ///
    /// # Panics
    /// If `id` is not allocated.
    pub fn add(&mut self, id: u32, hs: Vec<BandKey>) {
        for (h, table) in hs.iter().zip(&mut self.tables) {
            if let Some(ids) = table.get_mut(&h[..]) {
                ids.push(id);
            } else {
                table.insert(Arc::clone(h), Ids::One(id));
            }
            self.slots.add_ref(id);
        }
        self.slots.set_hs(id, hs);
    }

    /// `forest.keys[key] = hs` (the tables are not touched).
    ///
    /// # Panics
    /// If `id` is not allocated.
    pub fn set_keys(&mut self, id: u32, hs: Vec<BandKey>) {
        self.slots.set_hs(id, hs);
    }

    /// `del forest.keys[key]`: false if there was no entry; else whether the
    /// id was freed (it is in no bucket).
    pub fn del_keys(&mut self, id: u32) -> Option<bool> {
        self.slots.take_hs(id)?;
        Some(self.slots.release(id))
    }

    /// `index()`: sort every table's band keys.
    pub fn index(&mut self) {
        for (table, sorted) in self.tables.iter().zip(&mut self.sorted) {
            *sorted = table.keys().cloned().collect();
            sorted.sort_unstable();
        }
    }

    /// The members of bucket `h` of table `table`, in insertion order.
    pub fn bucket(&self, table: usize, h: &[u8]) -> Option<&[u32]> {
        self.tables[table].get(h).map(Ids::as_slice)
    }

    /// The number of buckets in table `table`.
    pub fn table_len(&self, table: usize) -> usize {
        self.tables[table].len()
    }

    /// The buckets of table `table`, in no particular order.
    pub fn buckets(&self, table: usize) -> impl Iterator<Item = (&BandKey, &[u32])> + '_ {
        self.tables[table]
            .iter()
            .map(|(h, ids)| (h, ids.as_slice()))
    }

    /// The number of allocated ids (after [`Self::from_bytes`]: they are `0..n`).
    pub fn id_count(&self) -> usize {
        self.slots.alive_count()
    }

    /// `sorted_hashtables[table]`.
    pub fn sorted(&self, table: usize) -> &[BandKey] {
        &self.sorted[table]
    }

    /// The members of every indexed bucket of `table` whose key starts with
    /// `hp` (`ht[j][:prefix_size] == hp`), in datasketch's order: sorted band
    /// keys from the first with `ht[j][:prefix_size] >= hp`.
    pub fn prefix_members<'a>(
        &'a self,
        table: usize,
        hp: &'a [u8],
        prefix_size: usize,
    ) -> impl Iterator<Item = u32> + 'a {
        let sorted = &self.sorted[table];
        let start = sorted.partition_point(|h| prefix(h, prefix_size) < hp);
        sorted[start..]
            .iter()
            .take_while(move |h| prefix(h, prefix_size) == hp)
            .flat_map(move |h| self.bucket(table, h).unwrap_or_default().iter().copied())
    }

    /// `_query(minhash, r, b)`: every member [`Self::prefix_members`] yields
    /// for the prefixes `hps` (one per table, `zip`), in order. `prefix_size`
    /// is `len(hps[0])`.
    pub fn query_prefixes<K: AsRef<[u8]>>(&self, hps: &[K]) -> Vec<u32> {
        let Some(first) = hps.first() else {
            return Vec::new();
        };
        let size = first.as_ref().len();
        (0..self.tables.len().min(hps.len()))
            .flat_map(|t| self.prefix_members(t, hps[t].as_ref(), size))
            .collect()
    }

    /// `query(minhash, k)`: the distinct ids found for prefix depths `depth`,
    /// `depth - 1`, ..., 1, stopping once there are `want`. `full[t]` is the
    /// query's band key for table `t` (`depth` values of `row_bytes` bytes,
    /// fewer if the signature is short); the prefix of depth `r` is its first
    /// `r * row_bytes` bytes, which is `_H(hashvalues[start:start + r])`.
    pub fn query<K: AsRef<[u8]>>(
        &self,
        full: &[K],
        depth: usize,
        row_bytes: usize,
        want: usize,
    ) -> Vec<u32> {
        let mut seen = FxHashSet::default();
        let mut out = Vec::new();
        if want == 0 {
            return out;
        }
        for r in (1..=depth).rev() {
            let size = r.saturating_mul(row_bytes);
            let hps: Vec<&[u8]> = full.iter().map(|h| prefix(h.as_ref(), size)).collect();
            let Some(first) = hps.first() else {
                return out;
            };
            let prefix_size = first.len();
            for (t, hp) in hps.iter().enumerate().take(self.tables.len()) {
                for id in self.prefix_members(t, hp, prefix_size) {
                    if seen.insert(id) {
                        out.push(id);
                        if out.len() >= want {
                            return out;
                        }
                    }
                }
            }
        }
        out
    }

    /// Serialize the forest (for pickling); see [`super::lsh::LshIndex::to_bytes`].
    pub fn to_bytes(&self) -> (Vec<u8>, Vec<u32>) {
        let (mut w, ids) = Writer::new(&self.slots, self.tables.len());
        w.tables(&self.tables);
        for sorted in &self.sorted {
            w.int(sorted.len());
            for h in sorted {
                w.key(h);
            }
        }
        (w.finish(KIND_FOREST), ids)
    }

    /// The forest [`Self::to_bytes`] wrote, with ids `0..n`.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FormatError> {
        let mut r = Reader::new(data, KIND_FOREST)?;
        let (mut slots, num_tables) = r.slots()?;
        let tables = r.tables(num_tables, &mut slots, false)?;
        let mut sorted = Vec::with_capacity(tables.len());
        for _ in 0..num_tables {
            let n = r.count(1)?;
            let keys = (0..n).map(|_| r.key()).collect::<Result<Vec<_>, _>>()?;
            if keys.windows(2).any(|w| w[0] >= w[1]) {
                return Err(FormatError::new("sorted table is not sorted"));
            }
            sorted.push(keys);
        }
        r.finish(&slots)?;
        Ok(Self {
            slots,
            tables,
            sorted,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::lsh::{Bands, band_keys};
    use super::*;
    use std::collections::{HashMap, HashSet};

    /// datasketch's MinHashLSHForest, transcribed (the reference).
    struct Reference {
        k: usize,
        tables: Vec<HashMap<Vec<u8>, Vec<u32>>>,
        sorted: Vec<Vec<Vec<u8>>>,
    }

    impl Reference {
        fn h(values: &[u64], start: usize, end: usize) -> Vec<u8> {
            super::super::lsh::swapped_rows(values, 1, start, end)
        }

        fn add(&mut self, id: u32, values: &[u64]) {
            for (i, table) in self.tables.iter_mut().enumerate() {
                let h = Self::h(values, i * self.k, (i + 1) * self.k);
                table.entry(h).or_default().push(id);
            }
        }

        fn index(&mut self) {
            for (table, sorted) in self.tables.iter().zip(&mut self.sorted) {
                *sorted = table.keys().cloned().collect();
                sorted.sort();
            }
        }

        fn binary_search(len: usize, pred: impl Fn(usize) -> bool) -> usize {
            let (mut lo, mut hi) = (0, len);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if pred(mid) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            lo
        }

        fn query_r(&self, values: &[u64], r: usize) -> Vec<u32> {
            let hps: Vec<Vec<u8>> = (0..self.tables.len())
                .map(|i| Self::h(values, i * self.k, i * self.k + r))
                .collect();
            let size = hps[0].len();
            let mut out = Vec::new();
            for ((ht, hp), table) in self.sorted.iter().zip(&hps).zip(&self.tables) {
                let pre = |x: &Vec<u8>| x[..size.min(x.len())].to_vec();
                let i = Self::binary_search(ht.len(), |x| pre(&ht[x]) >= *hp);
                let mut j = i;
                while j < ht.len() && pre(&ht[j]) == *hp {
                    out.extend(&table[&ht[j]]);
                    j += 1;
                }
            }
            out
        }

        fn query(&self, values: &[u64], want: usize) -> Vec<u32> {
            let mut seen = HashSet::new();
            let mut out = Vec::new();
            for r in (1..=self.k).rev() {
                for id in self.query_r(values, r) {
                    if seen.insert(id) {
                        out.push(id);
                        if out.len() >= want {
                            return out;
                        }
                    }
                }
            }
            out
        }
    }

    fn rng(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    #[test]
    fn matches_datasketch_query_order_and_early_exit() {
        let (l, k) = (4, 3);
        let bands = Bands { b: l, r: k };
        let mut state = 0x9e37_79b9_7f4a_7c15;
        // Small alphabet, so prefixes collide often.
        let sig = |state: &mut u64| -> Vec<u64> { (0..l * k).map(|_| rng(state) % 3).collect() };
        let mut forest = ForestIndex::new(l);
        let mut reference = Reference {
            k,
            tables: vec![HashMap::new(); l],
            sorted: vec![Vec::new(); l],
        };
        let mut sigs = Vec::new();
        for round in 0..3 {
            for _ in 0..60 {
                let values = sig(&mut state);
                let id = forest.alloc().unwrap();
                forest.add(id, band_keys(&values, 1, bands));
                reference.add(id, &values);
                sigs.push(values);
            }
            if round < 2 {
                forest.index();
                reference.index();
            }
            assert!(!forest.has_empty_table());
            for _ in 0..40 {
                let q = sig(&mut state);
                let full = band_keys(&q, 1, bands);
                for want in [1, 2, 5, 17, 1000] {
                    assert_eq!(forest.query(&full, k, 8, want), reference.query(&q, want));
                }
                for r in 1..=k {
                    let hps: Vec<Vec<u8>> =
                        (0..l).map(|i| Reference::h(&q, i * k, i * k + r)).collect();
                    assert_eq!(forest.query_prefixes(&hps), reference.query_r(&q, r));
                }
            }
        }
        // A short signature: the prefixes are clamped as Python slices are.
        let short: Vec<u64> = sigs[0][..5].to_vec();
        let full = band_keys(&short, 1, bands);
        assert_eq!(
            forest.query(&full, k, 8, 1000),
            reference.query(&short, 1000)
        );
    }

    #[test]
    fn keys_entries_and_relisting() {
        let mut forest = ForestIndex::new(2);
        assert!(forest.has_empty_table());
        let a = forest.alloc().unwrap();
        let hs = vec![BandKey::from(&b"ab"[..]), BandKey::from(&b"cd"[..])];
        forest.add(a, hs.clone());
        assert!(forest.has_empty_table(), "not searchable before index()");
        forest.index();
        assert_eq!(forest.sorted(0), [BandKey::from(&b"ab"[..])]);
        assert_eq!(forest.query(&hs, 2, 1, 10), [a]);
        // `del forest.keys[a]`: still in the tables, so not freed; adding the
        // key again lists it twice, and a query reports it once.
        assert_eq!(forest.del_keys(a), Some(false));
        assert_eq!(forest.del_keys(a), None);
        assert!(!forest.contains(a) && forest.is_alive(a));
        forest.add(a, hs.clone());
        assert_eq!(forest.bucket(0, b"ab"), Some(&[a, a][..]));
        assert_eq!(forest.query(&hs, 2, 1, 10), [a]);
        // `forest.keys[b] = [...]` without add: no buckets, freed on delete.
        let b = forest.alloc().unwrap();
        forest.set_keys(b, vec![BandKey::from(&b"zz"[..])]);
        assert_eq!(forest.ids().collect::<Vec<_>>(), [a, b]);
        assert_eq!(forest.del_keys(b), Some(true));
        assert!(!forest.is_alive(b));
    }

    #[test]
    fn bytes_round_trip() {
        let mut forest = ForestIndex::new(3);
        let mut state = 7;
        for _ in 0..50 {
            let values: Vec<u64> = (0..6).map(|_| rng(&mut state) % 4).collect();
            let id = forest.alloc().unwrap();
            forest.add(id, band_keys(&values, 1, Bands { b: 3, r: 2 }));
        }
        forest.index();
        let late = forest.alloc().unwrap();
        forest.add(late, band_keys(&[1_u64; 6], 1, Bands { b: 3, r: 2 }));
        forest.del_keys(3);
        let (bytes, ids) = forest.to_bytes();
        assert_eq!(ids.len(), 51);
        let back = ForestIndex::from_bytes(&bytes).unwrap();
        assert_eq!(
            back.to_bytes().0,
            ForestIndex::from_bytes(&bytes).unwrap().to_bytes().0
        );
        assert_eq!(
            back.ids().collect::<Vec<_>>(),
            forest.ids().collect::<Vec<_>>()
        );
        for t in 0..3 {
            assert_eq!(back.sorted(t), forest.sorted(t));
            for h in forest.sorted(t) {
                assert_eq!(back.bucket(t, h), forest.bucket(t, h));
            }
        }
        for cut in 0..bytes.len() {
            assert!(ForestIndex::from_bytes(&bytes[..cut]).is_err());
        }
        assert!(
            super::super::lsh::LshIndex::from_bytes(&bytes).is_err(),
            "kind is checked"
        );
    }
}
