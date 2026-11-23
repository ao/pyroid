// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! Stages 2–4 (`MinhashDedupBuckets`, `MinhashDedupCluster`,
//! `MinhashDedupFilter`) over signatures held in memory.
//!
//! **Stage 2.** In each bucket datatrove sorts the documents' band tuples
//! (`hashes_per_bucket` values) together with the document index, and writes
//! a duplicate pair `(previous, current)` for every two *consecutive* entries
//! with equal bands. A group of `k` documents sharing a band therefore gives
//! the chain `(d0, d1), (d1, d2), ...` in index order, and groups come in
//! ascending band order. With 32-bit hashes, datatrove's reader stops at the
//! first band whose first value is `u32::MAX` (`sigdata[0] >= max_hash`), so
//! such bands never match.
//!
//! **Stage 3.** The pairs of bucket 0, then bucket 1, ... are unioned in that
//! order into a union-find with *union by size*: the root of the larger set
//! wins; on a tie the root of the pair's first document wins. Every document
//! in a cluster except its root is removed, so **the kept document of a
//! cluster is its union-find root** — usually its first member, but a later
//! document wins when its side of a merge was larger. The order of unions
//! (buckets, then band order inside a bucket) decides ties, which is why the
//! exact sort order is reproduced here.
//!
//! **Stage 4** keeps every document that is not removed. Documents without a
//! signature (fewer than `n_grams` words) never match and are always kept.

use super::config::{MinhashConfig, Precision};
use crate::par::{self, PoolError, Threads};

/// More documents than a `u32` index (datatrove's doc ids are `u32`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityError;

impl std::fmt::Display for CapacityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a MinHash index holds at most 2**32 - 1 documents")
    }
}

impl std::error::Error for CapacityError {}

/// The signatures of every document added so far (the stage-1 output).
#[derive(Clone, Debug)]
pub struct SignatureStore {
    num_buckets: usize,
    per_bucket: usize,
    precision: Precision,
    /// `len * num_hashes` values; rows of documents without a signature are 0.
    data: Vec<u64>,
    present: Vec<bool>,
}

impl SignatureStore {
    pub fn new(config: &MinhashConfig) -> Self {
        Self {
            num_buckets: config.num_buckets,
            per_bucket: config.hashes_per_bucket,
            precision: config.hash_config.precision,
            data: Vec::new(),
            present: Vec::new(),
        }
    }

    pub fn num_hashes(&self) -> usize {
        self.num_buckets * self.per_bucket
    }

    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }

    /// Documents added (with or without a signature).
    pub fn len(&self) -> usize {
        self.present.len()
    }

    pub fn is_empty(&self) -> bool {
        self.present.is_empty()
    }

    pub fn clear(&mut self) {
        self.data.clear();
        self.present.clear();
    }

    /// Room for `additional` more documents, or an error past `u32::MAX`.
    pub fn reserve(&mut self, additional: usize) -> Result<(), CapacityError> {
        let total = self.len().checked_add(additional).ok_or(CapacityError)?;
        if total > u32::MAX as usize {
            return Err(CapacityError);
        }
        self.present.reserve(additional);
        self.data.reserve(additional * self.num_hashes());
        Ok(())
    }

    /// Add `n` documents whose signatures `fill` writes in place, with no
    /// per-document allocation or copy: it gets the `n * num_hashes` new
    /// values (zeroed, one row per document) and returns whether each
    /// document has a signature. Rows of documents without one are zeroed
    /// again afterwards. If `fill` fails or panics, the store is left as it
    /// was.
    pub fn append_with<E, F>(&mut self, n: usize, fill: F) -> Result<(), E>
    where
        E: From<CapacityError>,
        F: FnOnce(&mut [u64]) -> Result<Vec<bool>, E>,
    {
        /// Truncates the values back to `start` unless disarmed (on error or unwind).
        struct Rollback<'a> {
            data: &'a mut Vec<u64>,
            start: usize,
            armed: bool,
        }
        impl Drop for Rollback<'_> {
            fn drop(&mut self) {
                if self.armed {
                    self.data.truncate(self.start);
                }
            }
        }

        self.reserve(n)?;
        let h = self.num_hashes();
        let start = self.data.len();
        let mut guard = Rollback {
            data: &mut self.data,
            start,
            armed: true,
        };
        guard.data.resize(start + n * h, 0);
        let present = fill(&mut guard.data[start..])?;
        assert_eq!(present.len(), n, "one presence flag per document");
        for (row, &has) in guard.data[start..].chunks_mut(h).zip(&present) {
            if !has {
                row.fill(0);
            }
        }
        guard.armed = false;
        drop(guard);
        self.present.extend(present);
        Ok(())
    }

    /// Add one document: its signature (`num_hashes` values), or `None` if it
    /// had no shingles.
    pub fn push(&mut self, signature: Option<&[u64]>) -> Result<(), CapacityError> {
        self.reserve(1)?;
        let h = self.num_hashes();
        if let Some(s) = signature {
            assert_eq!(s.len(), h, "signature length");
            self.data.extend_from_slice(s);
            self.present.push(true);
        } else {
            self.data.resize(self.data.len() + h, 0);
            self.present.push(false);
        }
        Ok(())
    }

    /// The signature of document `i`, if it has one.
    pub fn get(&self, i: usize) -> Option<&[u64]> {
        let h = self.num_hashes();
        self.present
            .get(i)
            .copied()
            .unwrap_or(false)
            .then(|| &self.data[i * h..(i + 1) * h])
    }

    fn band(&self, i: u32, bucket: usize) -> &[u64] {
        let start = i as usize * self.num_hashes() + bucket * self.per_bucket;
        &self.data[start..start + self.per_bucket]
    }

    /// Stage 2 for one bucket: datatrove's duplicate pairs, in its order.
    pub fn bucket_pairs(&self, bucket: usize) -> Vec<(u32, u32)> {
        assert!(bucket < self.num_buckets, "bucket out of range");
        let skip_max = self.precision == Precision::P32;
        let mut keys: Vec<(u64, u32)> = Vec::with_capacity(self.len());
        for (i, &p) in self.present.iter().enumerate() {
            if !p {
                continue;
            }
            // `reserve` keeps the count within u32.
            #[allow(clippy::cast_possible_truncation)]
            let i = i as u32;
            let first = self.band(i, bucket)[0];
            if skip_max && first == u64::from(u32::MAX) {
                continue;
            }
            keys.push((first, i));
        }
        // Sort by (first value, index); then order each run that shares a
        // first value by the rest of the band, giving (band, index) order.
        keys.sort_unstable();
        let mut pairs = Vec::new();
        let mut start = 0;
        while start < keys.len() {
            let first = keys[start].0;
            let mut end = start + 1;
            while end < keys.len() && keys[end].0 == first {
                end += 1;
            }
            if end - start >= 2 {
                let run = &mut keys[start..end];
                if self.per_bucket > 1 {
                    run.sort_by(|x, y| {
                        self.band(x.1, bucket)[1..]
                            .cmp(&self.band(y.1, bucket)[1..])
                            .then(x.1.cmp(&y.1))
                    });
                }
                for w in run.windows(2) {
                    if self.band(w[0].1, bucket) == self.band(w[1].1, bucket) {
                        pairs.push((w[0].1, w[1].1));
                    }
                }
            }
            start = end;
        }
        pairs
    }

    /// Stages 2 and 3: every bucket's pairs (buckets in parallel), unioned in
    /// datatrove's order.
    pub fn cluster(&self, threads: Threads) -> Result<Clusters, PoolError> {
        let mut per_bucket: Vec<Vec<(u32, u32)>> = vec![Vec::new(); self.num_buckets];
        let items: Vec<(usize, &mut Vec<(u32, u32)>)> = per_bucket.iter_mut().enumerate().collect();
        par::for_each(items, self.len(), threads, |(b, slot)| {
            *slot = self.bucket_pairs(b);
        })?;
        Ok(Clusters::from_pairs(
            self.len(),
            per_bucket.iter().flatten().copied(),
        ))
    }
}

/// datatrove's `union_set` / `set_size`, as arrays over document indices.
struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
    in_set: Vec<bool>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        // `n <= u32::MAX` (enforced by `SignatureStore::reserve`).
        #[allow(clippy::cast_possible_truncation)]
        let parent = (0..n).map(|i| i as u32).collect();
        Self {
            parent,
            size: vec![1; n],
            in_set: vec![false; n],
        }
    }

    /// datatrove's `parent(x)`: adds `x` to the set, returns its root, and
    /// compresses the path.
    fn find(&mut self, x: u32) -> u32 {
        self.in_set[x as usize] = true;
        let mut root = x;
        while self.parent[root as usize] != root {
            root = self.parent[root as usize];
        }
        let mut cur = x;
        while self.parent[cur as usize] != root {
            let next = self.parent[cur as usize];
            self.parent[cur as usize] = root;
            cur = next;
        }
        root
    }

    /// datatrove's `union(a, b)`: union by size, ties to `a`'s root.
    fn union(&mut self, a: u32, b: u32) {
        let mut ra = self.find(a);
        let mut rb = self.find(b);
        if ra == rb {
            return;
        }
        let (sa, sb) = (self.size[ra as usize], self.size[rb as usize]);
        if sa < sb {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb as usize] = ra;
        self.size[ra as usize] = sa + sb;
    }
}

/// The stage-3 result: which documents are removed, and the clusters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Clusters {
    /// The kept document of each document's cluster (itself if it is kept).
    root: Vec<u32>,
    /// datatrove's `minhash_cluster_id`: clusters numbered in order of their
    /// smallest member; -1 for documents in no cluster.
    cluster_id: Vec<i64>,
    /// datatrove's `minhash_cluster_size` (1 for documents in no cluster).
    cluster_size: Vec<u32>,
    count: usize,
}

impl Clusters {
    /// Union `pairs` (in order) over documents `0..n`.
    pub fn from_pairs(n: usize, pairs: impl IntoIterator<Item = (u32, u32)>) -> Self {
        let mut uf = UnionFind::new(n);
        for (a, b) in pairs {
            uf.union(a, b);
        }
        let mut root = Vec::with_capacity(n);
        let mut cluster_id = vec![-1i64; n];
        let mut cluster_size = vec![1u32; n];
        let mut id_of_root: Vec<i64> = vec![-1; n];
        let mut next_id = 0i64;
        for i in 0..n {
            #[allow(clippy::cast_possible_truncation)]
            let x = i as u32;
            if !uf.in_set[i] {
                root.push(x);
                continue;
            }
            let r = uf.find(x);
            root.push(r);
            let slot = &mut id_of_root[r as usize];
            if *slot < 0 {
                *slot = next_id;
                next_id += 1;
            }
            cluster_id[i] = *slot;
            cluster_size[i] = uf.size[r as usize];
        }
        Self {
            root,
            cluster_id,
            cluster_size,
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            count: next_id as usize,
        }
    }

    pub fn len(&self) -> usize {
        self.root.len()
    }

    pub fn is_empty(&self) -> bool {
        self.root.is_empty()
    }

    /// Stage 4: `true` for documents to keep.
    pub fn keep_mask(&self) -> Vec<bool> {
        self.root
            .iter()
            .enumerate()
            .map(|(i, &r)| r as usize == i)
            .collect()
    }

    pub fn is_kept(&self, i: usize) -> bool {
        self.root[i] as usize == i
    }

    /// The kept document of `i`'s cluster (`i` itself when `i` is kept).
    pub fn representative(&self, i: usize) -> u32 {
        self.root[i]
    }

    /// Indices of removed documents, ascending (datatrove's `.remove` file).
    pub fn duplicates(&self) -> Vec<u32> {
        self.root
            .iter()
            .zip(0u32..)
            .filter(|&(&r, i)| r != i)
            .map(|(_, i)| i)
            .collect()
    }

    pub fn cluster_ids(&self) -> &[i64] {
        &self.cluster_id
    }

    pub fn cluster_sizes(&self) -> &[u32] {
        &self.cluster_size
    }

    pub fn num_clusters(&self) -> usize {
        self.count
    }

    /// Clusters of two or more documents, in cluster-id order. Each lists
    /// its kept document first, then the removed ones in ascending order.
    pub fn clusters(&self) -> Vec<Vec<u32>> {
        let mut out: Vec<Vec<u32>> = vec![Vec::new(); self.count];
        for (i, &id) in self.cluster_id.iter().enumerate() {
            if id < 0 {
                continue;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let c = &mut out[id as usize];
            if c.is_empty() {
                c.push(self.root[i]);
            }
            if self.root[i] as usize != i {
                #[allow(clippy::cast_possible_truncation)]
                c.push(i as u32);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dedup::config::HashConfig;

    fn config(num_buckets: usize, per_bucket: usize, precision: Precision) -> MinhashConfig {
        MinhashConfig {
            num_buckets,
            hashes_per_bucket: per_bucket,
            hash_config: HashConfig {
                precision,
                ..HashConfig::default()
            },
            ..MinhashConfig::default()
        }
    }

    /// Python transcription of datatrove's MinhashDedupCluster.union/parent.
    fn reference(n: usize, pairs: &[(u32, u32)]) -> Vec<bool> {
        use std::collections::HashMap;
        fn parent(u: &mut HashMap<u32, u32>, x: u32) -> u32 {
            match u.get(&x).copied() {
                None => {
                    u.insert(x, x);
                    x
                }
                Some(p) if p == x => x,
                Some(p) => {
                    let r = parent(u, p);
                    u.insert(x, r);
                    r
                }
            }
        }
        let mut u = HashMap::new();
        let mut size: HashMap<u32, u32> = HashMap::new();
        for &(a, b) in pairs {
            let (mut ra, mut rb) = (parent(&mut u, a), parent(&mut u, b));
            if ra != rb {
                let sa = *size.get(&ra).unwrap_or(&1);
                let sb = *size.get(&rb).unwrap_or(&1);
                if sa < sb {
                    std::mem::swap(&mut ra, &mut rb);
                }
                u.insert(rb, ra);
                size.insert(ra, sa + sb);
                size.remove(&rb);
            }
        }
        (0..u32::try_from(n).unwrap())
            .map(|i| !u.contains_key(&i) || parent(&mut u.clone(), i) == i)
            .collect()
    }

    #[test]
    fn union_by_size_picks_the_larger_root() {
        // {1,2} (root 1) and {3,4} (root 3); then 0-3 makes root 3's side
        // size 3, so 2-4 joins {1,2} under root 3.
        let c = Clusters::from_pairs(5, [(1, 2), (3, 4), (0, 3), (2, 4)]);
        assert_eq!(c.keep_mask(), [false, false, false, true, false]);
        assert_eq!(c.clusters(), [vec![3, 0, 1, 2, 4]]);
        assert_eq!(c.cluster_ids(), [0, 0, 0, 0, 0]);
        assert_eq!(c.cluster_sizes(), [5, 5, 5, 5, 5]);
        // Same pairs, different order: the tie now goes to root 1.
        let c = Clusters::from_pairs(5, [(1, 2), (3, 4), (2, 4), (0, 3)]);
        assert_eq!(c.keep_mask(), [false, true, false, false, false]);
        assert_eq!(c.duplicates(), [0, 2, 3, 4]);
    }

    #[test]
    fn matches_reference_union_find() {
        let mut x = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = |m: u32| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            #[allow(clippy::cast_possible_truncation)]
            let v = (x % u64::from(m)) as u32;
            v
        };
        for round in 0..300 {
            let n = 2 + (round % 40);
            let pairs: Vec<(u32, u32)> = (0..next(60))
                .map(|_| {
                    let a = next(n);
                    let b = next(n);
                    (a.min(b), a.max(b))
                })
                .filter(|(a, b)| a != b)
                .collect();
            let c = Clusters::from_pairs(n as usize, pairs.iter().copied());
            assert_eq!(c.keep_mask(), reference(n as usize, &pairs), "{pairs:?}");
        }
    }

    #[test]
    fn bucket_pairs_chain_in_band_order() {
        let cfg = config(1, 2, Precision::P64);
        let mut s = SignatureStore::new(&cfg);
        for sig in [[5, 1], [3, 9], [5, 1], [3, 2], [5, 1], [3, 9], [3, 2]] {
            s.push(Some(&sig)).unwrap();
        }
        s.push(None).unwrap();
        // Bands sorted: (3,2):{3,6} (3,9):{1,5} (5,1):{0,2,4}
        assert_eq!(s.bucket_pairs(0), [(3, 6), (1, 5), (0, 2), (2, 4)]);
        let c = s.cluster(Threads::ONE).unwrap();
        assert_eq!(
            c.keep_mask(),
            [true, true, false, true, false, false, false, true]
        );
    }

    #[test]
    fn precision_32_ignores_max_first_value() {
        let cfg = config(1, 2, Precision::P32);
        let m = u64::from(u32::MAX);
        let mut s = SignatureStore::new(&cfg);
        for sig in [[m, 1], [m, 1], [7, 1], [7, 1]] {
            s.push(Some(&sig)).unwrap();
        }
        assert_eq!(s.bucket_pairs(0), [(2, 3)]);
    }

    #[test]
    fn signatures_round_trip() {
        let cfg = config(2, 2, Precision::P64);
        let mut s = SignatureStore::new(&cfg);
        s.push(Some(&[1, 2, 3, 4])).unwrap();
        s.push(None).unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.get(0), Some(&[1, 2, 3, 4][..]));
        assert_eq!(s.get(1), None);
        assert_eq!(s.get(2), None);
        let c = s.cluster(Threads::Auto).unwrap();
        assert_eq!(c.keep_mask(), [true, true]);
        assert_eq!(c.cluster_ids(), [-1, -1]);
        assert_eq!(c.clusters(), Vec::<Vec<u32>>::new());
    }

    #[test]
    fn append_with_matches_push_and_rolls_back() {
        let cfg = config(2, 2, Precision::P64);
        let mut pushed = SignatureStore::new(&cfg);
        pushed.push(Some(&[9, 9, 9, 9])).unwrap();
        pushed.push(Some(&[1, 2, 3, 4])).unwrap();
        pushed.push(None).unwrap();
        let mut filled = SignatureStore::new(&cfg);
        filled.push(Some(&[9, 9, 9, 9])).unwrap();
        filled
            .append_with(2, |rows: &mut [u64]| {
                assert_eq!(rows, [0; 8]);
                rows.copy_from_slice(&[1, 2, 3, 4, 7, 7, 7, 7]); // row 1 written, then "no signature"
                Ok::<_, CapacityError>(vec![true, false])
            })
            .unwrap();
        assert_eq!(filled.len(), 3);
        assert_eq!(filled.data, pushed.data); // the absent row is zeroed again
        assert_eq!(filled.present, pushed.present);

        let err = filled.append_with(1, |_: &mut [u64]| Err(CapacityError));
        assert_eq!(err, Err(CapacityError));
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = filled.append_with(5, |_: &mut [u64]| -> Result<Vec<bool>, CapacityError> {
                panic!("kernel bug")
            });
        }));
        assert!(unwound.is_err());
        assert_eq!(filled.data, pushed.data);
        assert_eq!(filled.present, pushed.present);
    }
}
