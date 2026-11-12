// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! The in-memory band index behind `MinHashLSH` (datasketch's `"dict"` storage).
//!
//! datasketch keeps two structures: `keys`, a `defaultdict(list)` from each key
//! to the band keys `H` of its signature, and one `defaultdict(set)` per band
//! from `H` to the keys in that bucket. [`LshIndex`] holds the same two
//! structures over `u32` ids: the bindings map each Python key (by Python
//! equality, as a dict does) to one id and keep the key objects.
//!
//! An id stays alive while it has a `keys` entry or sits in any bucket: after
//! `insert(k, m1)`, `insert(k, m2, check_duplication=False)` and `remove(k)`,
//! datasketch leaves `k` in the buckets of `m2`, and a later `insert(k, ...)`
//! must find that same key in those sets. Freed ids are reused.
//!
//! The band key of rows `start..end` is `bytes(hashvalues[start:end].byteswap().data)`:
//! the rows' native bytes with every value's bytes reversed ([`swapped_rows`]).

use std::fmt;
use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::par;

/// One band key `H` (shared between an id's list and its bucket).
pub type BandKey = Arc<[u8]>;

/// One band table: `H` -> the ids in that bucket.
type Table = FxHashMap<BandKey, Ids>;

/// A value numpy can hold in `hashvalues`: its bytes can be reversed.
pub trait HashValue: Copy + Send + Sync {
    /// Append the bytes of each value, reversed (numpy's `byteswap()`).
    fn extend_swapped(values: &[Self], out: &mut Vec<u8>);
}

macro_rules! int_hash_value {
    ($($t:ty),*) => {$(
        impl HashValue for $t {
            fn extend_swapped(values: &[Self], out: &mut Vec<u8>) {
                out.reserve(std::mem::size_of_val(values));
                for v in values {
                    out.extend_from_slice(&v.swap_bytes().to_ne_bytes());
                }
            }
        }
    )*};
}

int_hash_value!(u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! float_hash_value {
    ($($t:ty),*) => {$(
        impl HashValue for $t {
            fn extend_swapped(values: &[Self], out: &mut Vec<u8>) {
                out.reserve(std::mem::size_of_val(values));
                for v in values {
                    out.extend_from_slice(&v.to_bits().swap_bytes().to_ne_bytes());
                }
            }
        }
    )*};
}

float_hash_value!(f32, f64);

/// The band layout of an index: `b` bands of `r` rows, band `i` covering rows
/// `i * r .. (i + 1) * r` (datasketch's `hashranges`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bands {
    pub b: usize,
    pub r: usize,
}

impl Bands {
    /// The rows of band `i`.
    pub fn range(self, i: usize) -> (usize, usize) {
        (
            i.saturating_mul(self.r),
            i.saturating_add(1).saturating_mul(self.r),
        )
    }
}

/// `bytes(hashvalues[start:end].byteswap().data)` for a C-contiguous array
/// of `row_len` values per row (`values.len() / row_len` rows): the rows are
/// clamped to those that exist, as a slice is.
pub fn swapped_rows<T: HashValue>(
    values: &[T],
    row_len: usize,
    start: usize,
    end: usize,
) -> Vec<u8> {
    let mut out = Vec::new();
    extend_swapped_rows(values, row_len, start, end, &mut out);
    out
}

/// [`swapped_rows`], appended to `out`.
pub fn extend_swapped_rows<T: HashValue>(
    values: &[T],
    row_len: usize,
    start: usize,
    end: usize,
    out: &mut Vec<u8>,
) {
    if row_len == 0 {
        return;
    }
    let rows = values.len() / row_len;
    let (start, end) = (start.min(rows), end.min(rows));
    if start < end {
        T::extend_swapped(&values[start * row_len..end * row_len], out);
    }
}

/// [`swapped_rows`] over the native bytes of the values: `itemsize` bytes per
/// value, `row_bytes` per row (a multiple of `itemsize`).
///
/// # Panics
/// If `itemsize` is 0 or does not divide `row_bytes`.
pub fn swapped_rows_ne(
    bytes: &[u8],
    itemsize: usize,
    row_bytes: usize,
    start: usize,
    end: usize,
) -> Vec<u8> {
    assert!(
        itemsize > 0 && row_bytes % itemsize == 0,
        "rows of {row_bytes} bytes are not whole {itemsize}-byte values"
    );
    let mut out = Vec::new();
    if row_bytes == 0 {
        return out;
    }
    let rows = bytes.len() / row_bytes;
    let (start, end) = (start.min(rows), end.min(rows));
    if start < end {
        out.reserve((end - start) * row_bytes);
        for value in bytes[start * row_bytes..end * row_bytes].chunks_exact(itemsize) {
            out.extend(value.iter().rev());
        }
    }
    out
}

/// The band keys of one signature (`self._H(hashvalues[s:e])` for every
/// `(s, e)` in `hashranges`).
pub fn band_keys<T: HashValue>(values: &[T], row_len: usize, bands: Bands) -> Vec<BandKey> {
    // One scratch buffer: each key is then a single allocation of its exact size.
    let mut buf = Vec::new();
    (0..bands.b)
        .map(|i| {
            let (start, end) = bands.range(i);
            buf.clear();
            extend_swapped_rows(values, row_len, start, end, &mut buf);
            BandKey::from(&buf[..])
        })
        .collect()
}

/// [`band_keys`] over native bytes (see [`swapped_rows_ne`]).
pub fn band_keys_ne(bytes: &[u8], itemsize: usize, row_bytes: usize, bands: Bands) -> Vec<BandKey> {
    (0..bands.b)
        .map(|i| {
            let (start, end) = bands.range(i);
            BandKey::from(&swapped_rows_ne(bytes, itemsize, row_bytes, start, end)[..])
        })
        .collect()
}

/// [`band_keys`] of `items` signatures stored back to back in `values`
/// (`item_len` values each, `row_len` per row), in parallel per `threads`.
///
/// # Panics
/// If `values` does not hold `items * item_len` values.
pub fn band_keys_many<T: HashValue>(
    values: &[T],
    items: usize,
    item_len: usize,
    row_len: usize,
    bands: Bands,
    threads: par::Threads,
) -> Result<Vec<Vec<BandKey>>, par::PoolError> {
    assert_eq!(
        Some(values.len()),
        items.checked_mul(item_len),
        "{items} signatures of {item_len} values"
    );
    par::map(items, threads, |j| {
        band_keys(&values[j * item_len..(j + 1) * item_len], row_len, bands)
    })
}

/// The ids in one bucket: a sorted set (`MinHashLSH`) or a list in insertion
/// order (`MinHashLSHForest`). Most buckets hold one id, kept inline.
#[derive(Clone, Debug)]
pub(crate) enum Ids {
    One(u32),
    Many(Vec<u32>),
}

impl Default for Ids {
    fn default() -> Self {
        Self::Many(Vec::new())
    }
}

impl Ids {
    pub(crate) fn as_slice(&self) -> &[u32] {
        match self {
            Self::One(id) => std::slice::from_ref(id),
            Self::Many(ids) => ids,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.as_slice().len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.as_slice().is_empty()
    }

    pub(crate) fn from_vec(mut ids: Vec<u32>) -> Self {
        if ids.len() == 1 {
            Self::One(ids[0])
        } else {
            ids.shrink_to_fit();
            Self::Many(ids)
        }
    }

    /// `set.add(id)`; true if it was not there.
    pub(crate) fn insert_sorted(&mut self, id: u32) -> bool {
        match self {
            Self::One(x) if *x == id => false,
            Self::Many(ids) if ids.is_empty() => {
                *self = Self::One(id);
                true
            }
            Self::Many(ids) => match ids.binary_search(&id) {
                Ok(_) => false,
                Err(at) => {
                    ids.insert(at, id);
                    true
                }
            },
            Self::One(x) => {
                let x = *x;
                *self = Self::Many(if x < id { vec![x, id] } else { vec![id, x] });
                true
            }
        }
    }

    /// `set.discard(id)`; true if it was there.
    pub(crate) fn remove_sorted(&mut self, id: u32) -> bool {
        match self {
            Self::One(x) if *x == id => {
                *self = Self::Many(Vec::new());
                true
            }
            Self::One(_) => false,
            Self::Many(ids) => match ids.binary_search(&id) {
                Ok(at) => {
                    ids.remove(at);
                    if ids.len() == 1 {
                        *self = Self::One(ids[0]);
                    }
                    true
                }
                Err(_) => false,
            },
        }
    }

    /// `list.append(id)`.
    pub(crate) fn push(&mut self, id: u32) {
        match self {
            Self::Many(ids) if ids.is_empty() => *self = Self::One(id),
            Self::Many(ids) => ids.push(id),
            Self::One(x) => *self = Self::Many(vec![*x, id]),
        }
    }
}

/// An index can hold at most `u32::MAX - 1` live ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CapacityError;

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "an index holds at most {} keys", HOLE - 1)
    }
}

impl std::error::Error for CapacityError {}

/// Marks a removed entry in [`Slots::order`].
const HOLE: u32 = u32::MAX;

#[derive(Clone, Debug, Default)]
struct Slot {
    /// The `keys` entry (`None`: not in `keys`).
    hs: Option<Vec<BandKey>>,
    /// Bucket memberships.
    refs: usize,
    /// Position in [`Slots::order`] while `hs` is `Some`.
    pos: usize,
    alive: bool,
}

/// The ids of an index: their `keys` entries in insertion order, bucket
/// reference counts, and a free list.
#[derive(Clone, Debug, Default)]
pub(crate) struct Slots {
    entries: Vec<Slot>,
    free: Vec<u32>,
    /// Ids with a `keys` entry, in the order the entries were created (a
    /// dict's order), with [`HOLE`]s for removed ones.
    order: Vec<u32>,
    holes: usize,
}

impl Slots {
    fn slot(&self, id: u32) -> Option<&Slot> {
        self.entries.get(id as usize).filter(|s| s.alive)
    }

    fn slot_mut(&mut self, id: u32) -> &mut Slot {
        let slot = &mut self.entries[id as usize];
        assert!(slot.alive, "id {id} is not allocated");
        slot
    }

    pub(crate) fn alloc(&mut self) -> Result<u32, CapacityError> {
        let id = if let Some(id) = self.free.pop() {
            id
        } else {
            let id = u32::try_from(self.entries.len())
                .ok()
                .filter(|&id| id < HOLE - 1)
                .ok_or(CapacityError)?;
            self.entries.push(Slot::default());
            id
        };
        self.entries[id as usize] = Slot {
            alive: true,
            ..Slot::default()
        };
        Ok(id)
    }

    /// Free `id` if nothing refers to it; true if it was freed.
    pub(crate) fn release(&mut self, id: u32) -> bool {
        let Some(slot) = self.entries.get_mut(id as usize) else {
            return false;
        };
        if !slot.alive || slot.hs.is_some() || slot.refs > 0 {
            return false;
        }
        *slot = Slot::default();
        self.free.push(id);
        true
    }

    pub(crate) fn is_alive(&self, id: u32) -> bool {
        self.slot(id).is_some()
    }

    pub(crate) fn hs(&self, id: u32) -> Option<&[BandKey]> {
        self.slot(id).and_then(|s| s.hs.as_deref())
    }

    /// The number of `keys` entries.
    pub(crate) fn len(&self) -> usize {
        self.order.len() - self.holes
    }

    pub(crate) fn add_ref(&mut self, id: u32) {
        self.slot_mut(id).refs += 1;
    }

    pub(crate) fn drop_ref(&mut self, id: u32) {
        let slot = self.slot_mut(id);
        slot.refs = slot.refs.checked_sub(1).expect("bucket reference count");
    }

    fn push_order(&mut self, id: u32) -> usize {
        self.order.push(id);
        self.order.len() - 1
    }

    /// `keys[id] = hs` (a new entry goes last, as in a dict).
    pub(crate) fn set_hs(&mut self, id: u32, hs: Vec<BandKey>) {
        let new = self.slot_mut(id).hs.is_none();
        let pos = if new {
            self.push_order(id)
        } else {
            self.slot_mut(id).pos
        };
        let slot = self.slot_mut(id);
        slot.hs = Some(hs);
        slot.pos = pos;
    }

    /// `keys[id].extend(hs)` on a `defaultdict(list)`.
    pub(crate) fn extend_hs(&mut self, id: u32, hs: &[BandKey]) {
        if self.slot_mut(id).hs.is_none() {
            self.set_hs(id, hs.to_vec());
        } else if let Some(list) = &mut self.slot_mut(id).hs {
            list.extend(hs.iter().cloned());
        }
    }

    /// `del keys[id]`: the entry, if there was one.
    pub(crate) fn take_hs(&mut self, id: u32) -> Option<Vec<BandKey>> {
        let slot = self.entries.get_mut(id as usize).filter(|s| s.alive)?;
        let hs = slot.hs.take()?;
        let pos = slot.pos;
        self.order[pos] = HOLE;
        self.holes += 1;
        if self.holes > 64 && self.holes * 2 > self.order.len() {
            self.compact_order();
        }
        Some(hs)
    }

    fn compact_order(&mut self) {
        self.order.retain(|&id| id != HOLE);
        self.holes = 0;
        for (pos, &id) in self.order.iter().enumerate() {
            self.entries[id as usize].pos = pos;
        }
    }

    /// Ids with a `keys` entry, in insertion order.
    pub(crate) fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.order.iter().copied().filter(|&id| id != HOLE)
    }

    pub(crate) fn alive_count(&self) -> usize {
        self.entries.iter().filter(|s| s.alive).count()
    }

    /// Every allocated id, ascending.
    fn alive_ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.entries.len())
            .filter(|&i| self.entries[i].alive)
            .map(|i| u32::try_from(i).expect("ids fit in u32"))
    }
}

/// Why [`LshIndex::remove`] failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RemoveError {
    /// The id has no `keys` entry (datasketch's "The given key does not
    /// exist"). Nothing changed.
    NotFound,
    /// `set.remove` raised `KeyError` in band `table`: the key was not in its
    /// bucket. As in datasketch, the bands before it were updated, an absent
    /// bucket was created empty (`defaultdict`), and the `keys` entry stays.
    Missing { table: usize },
}

impl fmt::Display for RemoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => f.write_str("The given key does not exist"),
            Self::Missing { table } => write!(f, "the key is not in its bucket of band {table}"),
        }
    }
}

impl std::error::Error for RemoveError {}

/// `set.add(id)` on `table[h]`; true if the bucket did not hold `id`.
fn add_to_bucket(table: &mut Table, h: &BandKey, id: u32) -> bool {
    if let Some(ids) = table.get_mut(&h[..]) {
        ids.insert_sorted(id)
    } else {
        table.insert(Arc::clone(h), Ids::One(id));
        true
    }
}

/// The in-memory `MinHashLSH` index: `keys` and `b` band tables over ids.
#[derive(Clone, Debug)]
pub struct LshIndex {
    slots: Slots,
    tables: Vec<Table>,
}

impl LshIndex {
    /// An empty index with `num_tables` band tables.
    pub fn new(num_tables: usize) -> Self {
        Self {
            slots: Slots::default(),
            tables: vec![Table::default(); num_tables],
        }
    }

    pub fn num_tables(&self) -> usize {
        self.tables.len()
    }

    /// The number of keys (`len(lsh.keys)`).
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// No keys.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// datasketch's `MinHashLSH.is_empty()`: some band table has no bucket.
    pub fn has_empty_table(&self) -> bool {
        self.tables.iter().any(FxHashMap::is_empty)
    }

    /// A new id, alive and unused: [`Self::insert`] it or [`Self::release`] it.
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

    /// `key in lsh.keys`.
    pub fn contains(&self, id: u32) -> bool {
        self.slots.hs(id).is_some()
    }

    /// `lsh.keys.get(key)`: the band keys of `id`, `None` without an entry.
    pub fn keys_of(&self, id: u32) -> Option<&[BandKey]> {
        self.slots.hs(id)
    }

    /// The ids with a `keys` entry, in insertion order.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.slots.ids()
    }

    /// `keys[id].extend(hs)`, then `table.add(id)` for `zip(hs, tables)`.
    ///
    /// # Panics
    /// If `id` is not allocated.
    pub fn insert(&mut self, id: u32, hs: &[BandKey]) {
        self.slots.extend_hs(id, hs);
        for (h, table) in hs.iter().zip(&mut self.tables) {
            if add_to_bucket(table, h, id) {
                self.slots.add_ref(id);
            }
        }
    }

    /// [`Self::insert`] of every item in order, the band tables filled in
    /// parallel (one task per table).
    ///
    /// # Panics
    /// If an id is not allocated.
    pub fn insert_many(
        &mut self,
        items: &[(u32, Vec<BandKey>)],
        threads: par::Threads,
    ) -> Result<(), par::PoolError> {
        if items.len() <= par::MIN_DOCS_PER_TASK || par::max_tasks(threads)? <= 1 {
            // Serially, item by item is faster: one pass over the band keys.
            for (id, hs) in items {
                self.insert(*id, hs);
            }
            return Ok(());
        }
        for (id, hs) in items {
            self.slots.extend_hs(*id, hs);
        }
        let mut added = vec![Vec::new(); self.tables.len()];
        let tasks: Vec<(usize, &mut Table, &mut Vec<bool>)> = self
            .tables
            .iter_mut()
            .zip(&mut added)
            .enumerate()
            .map(|(t, (table, added))| (t, table, added))
            .collect();
        par::for_each(tasks, items.len(), threads, |(t, table, added)| {
            *added = items
                .iter()
                .map(|(id, hs)| hs.get(t).is_some_and(|h| add_to_bucket(table, h, *id)))
                .collect();
        })?;
        for added in &added {
            for ((id, _), &new) in items.iter().zip(added) {
                if new {
                    self.slots.add_ref(*id);
                }
            }
        }
        Ok(())
    }

    /// datasketch's `_remove` after its `key in self.keys` check: for
    /// `zip(keys[key], tables)` remove `id` from the bucket (`KeyError` if it
    /// is not there) and drop the bucket once empty; then `del keys[key]`.
    /// `Ok(true)` if the id was freed (it is in no other bucket).
    pub fn remove(&mut self, id: u32) -> Result<bool, RemoveError> {
        let hs = self.slots.hs(id).ok_or(RemoveError::NotFound)?.to_vec();
        for (t, (h, table)) in hs.iter().zip(&mut self.tables).enumerate() {
            let Some(ids) = table.get_mut(&h[..]) else {
                table.insert(Arc::clone(h), Ids::default());
                return Err(RemoveError::Missing { table: t });
            };
            if !ids.remove_sorted(id) {
                return Err(RemoveError::Missing { table: t });
            }
            self.slots.drop_ref(id);
            if ids.is_empty() {
                table.remove(&h[..]);
            }
        }
        self.slots.take_hs(id);
        Ok(self.slots.release(id))
    }

    /// The ids in the buckets of `hs` (one band key per table, `zip`), each
    /// once, ascending (`MinHashLSH.query`, and `_query_b` with `b` keys).
    pub fn query<K: AsRef<[u8]>>(&self, hs: &[K]) -> Vec<u32> {
        let mut out = Vec::new();
        for (h, table) in hs.iter().zip(&self.tables) {
            if let Some(ids) = table.get(h.as_ref()) {
                out.extend_from_slice(ids.as_slice());
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// [`Self::query`] of every item, in parallel per `threads`.
    pub fn query_many<K: AsRef<[u8]> + Sync>(
        &self,
        queries: &[Vec<K>],
        threads: par::Threads,
    ) -> Result<Vec<Vec<u32>>, par::PoolError> {
        par::map(queries.len(), threads, |j| self.query(&queries[j]))
    }

    /// [`Self::query`] of the band keys of `values` (see [`band_keys`]), computed
    /// band by band into one scratch buffer: no allocation per band.
    pub fn query_values<T: HashValue>(
        &self,
        values: &[T],
        row_len: usize,
        bands: Bands,
    ) -> Vec<u32> {
        let mut buf = Vec::new();
        let mut out = Vec::new();
        for (i, table) in self.tables.iter().enumerate().take(bands.b) {
            let (start, end) = bands.range(i);
            buf.clear();
            extend_swapped_rows(values, row_len, start, end, &mut buf);
            if let Some(ids) = table.get(&buf[..]) {
                out.extend_from_slice(ids.as_slice());
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    /// [`Self::query_values`] of `items` signatures stored back to back in
    /// `values` (as [`band_keys_many`]), in parallel per `threads`.
    ///
    /// # Panics
    /// If `values` does not hold `items * item_len` values.
    pub fn query_values_many<T: HashValue>(
        &self,
        values: &[T],
        items: usize,
        item_len: usize,
        row_len: usize,
        bands: Bands,
        threads: par::Threads,
    ) -> Result<Vec<Vec<u32>>, par::PoolError> {
        assert_eq!(
            Some(values.len()),
            items.checked_mul(item_len),
            "{items} signatures of {item_len} values"
        );
        par::map(items, threads, |j| {
            self.query_values(&values[j * item_len..(j + 1) * item_len], row_len, bands)
        })
    }

    /// The number of buckets in band `table`.
    pub fn table_len(&self, table: usize) -> usize {
        self.tables[table].len()
    }

    /// The bucket `h` of band `table` (`None` if there is none; it can be
    /// empty after a failed remove).
    pub fn bucket(&self, table: usize, h: &[u8]) -> Option<&[u32]> {
        self.tables[table].get(h).map(Ids::as_slice)
    }

    /// The buckets of band `table`, in no particular order.
    pub fn buckets(&self, table: usize) -> impl Iterator<Item = (&BandKey, &[u32])> + '_ {
        self.tables[table]
            .iter()
            .map(|(h, ids)| (h, ids.as_slice()))
    }
}

/// A pickled index could not be read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatError(&'static str);

impl FormatError {
    pub(crate) const fn new(what: &'static str) -> Self {
        Self(what)
    }
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid pickled index state: {}", self.0)
    }
}

impl std::error::Error for FormatError {}

const MAGIC: &[u8; 8] = b"pyroidDS";
const VERSION: u8 = 1;
pub(crate) const KIND_LSH: u8 = b'L';
pub(crate) const KIND_FOREST: u8 = b'F';

fn put_int(out: &mut Vec<u8>, mut v: usize) {
    loop {
        let low = u8::try_from(v & 0x7f).expect("7 bits");
        v >>= 7;
        if v == 0 {
            out.push(low);
            return;
        }
        out.push(low | 0x80);
    }
}

/// Writes the pickle format: a header, the distinct band keys once (the
/// pool), then the ids, `keys` order and buckets as LEB128 integers (pool
/// indices, and ids renumbered `0..n` in ascending order).
pub(crate) struct Writer<'a> {
    body: Vec<u8>,
    pool: FxHashMap<&'a [u8], usize>,
    pool_keys: Vec<&'a [u8]>,
    /// Original id -> number (`usize::MAX` for free ids).
    numbers: Vec<usize>,
}

impl<'a> Writer<'a> {
    /// Start with the ids of `slots` and their `keys` entries; returns the
    /// writer and the original id of each number.
    pub(crate) fn new(slots: &'a Slots, num_tables: usize) -> (Self, Vec<u32>) {
        let ids: Vec<u32> = slots.alive_ids().collect();
        let mut numbers = vec![usize::MAX; slots.entries.len()];
        for (n, &id) in ids.iter().enumerate() {
            numbers[id as usize] = n;
        }
        let mut w = Self {
            body: Vec::new(),
            pool: FxHashMap::default(),
            pool_keys: Vec::new(),
            numbers,
        };
        w.int(num_tables);
        w.int(ids.len());
        for &id in &ids {
            match slots.hs(id) {
                None => w.int(0),
                Some(hs) => {
                    w.int(hs.len() + 1);
                    for h in hs {
                        w.key(h);
                    }
                }
            }
        }
        w.int(slots.len());
        for id in slots.ids() {
            w.id(id);
        }
        (w, ids)
    }

    pub(crate) fn int(&mut self, v: usize) {
        put_int(&mut self.body, v);
    }

    pub(crate) fn id(&mut self, id: u32) {
        let n = self.numbers[id as usize];
        self.int(n);
    }

    pub(crate) fn key(&mut self, h: &'a [u8]) {
        let next = self.pool_keys.len();
        let index = *self.pool.entry(h).or_insert(next);
        if index == next {
            self.pool_keys.push(h);
        }
        self.int(index);
    }

    pub(crate) fn tables(&mut self, tables: &'a [Table]) {
        for table in tables {
            self.int(table.len());
            for (h, ids) in table {
                self.key(h);
                self.int(ids.len());
                for &id in ids.as_slice() {
                    self.id(id);
                }
            }
        }
    }

    pub(crate) fn finish(self, kind: u8) -> Vec<u8> {
        let mut out = MAGIC.to_vec();
        out.extend([kind, VERSION]);
        put_int(&mut out, self.pool_keys.len());
        for h in &self.pool_keys {
            put_int(&mut out, h.len());
            out.extend_from_slice(h);
        }
        out.extend_from_slice(&self.body);
        out
    }
}

/// Reads what [`Writer`] wrote, checking every count and index.
pub(crate) struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
    pool: Vec<BandKey>,
}

impl<'a> Reader<'a> {
    pub(crate) fn new(data: &'a [u8], kind: u8) -> Result<Self, FormatError> {
        let head = data
            .get(..MAGIC.len() + 2)
            .ok_or(FormatError("truncated"))?;
        if &head[..MAGIC.len()] != MAGIC || head[MAGIC.len()] != kind {
            return Err(FormatError("not a pickled index of this kind"));
        }
        if head[MAGIC.len() + 1] != VERSION {
            return Err(FormatError("unsupported format version"));
        }
        let mut r = Self {
            data,
            pos: MAGIC.len() + 2,
            pool: Vec::new(),
        };
        let n = r.count(1)?;
        let mut pool = Vec::with_capacity(n);
        for _ in 0..n {
            let len = r.count(1)?;
            let bytes = &r.data[r.pos..r.pos + len];
            r.pos += len;
            pool.push(BandKey::from(bytes));
        }
        r.pool = pool;
        Ok(r)
    }

    pub(crate) fn int(&mut self) -> Result<usize, FormatError> {
        let mut v: usize = 0;
        for shift in (0..usize::BITS).step_by(7) {
            let byte = *self.data.get(self.pos).ok_or(FormatError("truncated"))?;
            self.pos += 1;
            let bits = usize::from(byte & 0x7f);
            if shift > 0 && bits >> (usize::BITS - shift) != 0 {
                return Err(FormatError("integer overflow"));
            }
            v |= bits << shift;
            if byte & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(FormatError("integer overflow"))
    }

    /// A count of items taking at least `min_bytes` each (so a corrupt count
    /// cannot ask for a huge allocation).
    pub(crate) fn count(&mut self, min_bytes: usize) -> Result<usize, FormatError> {
        let n = self.int()?;
        let left = self.data.len() - self.pos;
        if n.checked_mul(min_bytes).is_none_or(|need| need > left) {
            return Err(FormatError("truncated"));
        }
        Ok(n)
    }

    pub(crate) fn key(&mut self) -> Result<BandKey, FormatError> {
        let index = self.int()?;
        self.pool
            .get(index)
            .cloned()
            .ok_or(FormatError("band key index out of range"))
    }

    /// An id number below `n`.
    pub(crate) fn id(&mut self, n: usize) -> Result<u32, FormatError> {
        let id = self.int()?;
        if id >= n {
            return Err(FormatError("id out of range"));
        }
        u32::try_from(id).map_err(|_| FormatError("id out of range"))
    }

    /// The ids and their `keys` entries; returns them and the table count.
    pub(crate) fn slots(&mut self) -> Result<(Slots, usize), FormatError> {
        let num_tables = self.int()?;
        let n = self.count(1)?;
        if n >= HOLE as usize - 1 {
            return Err(FormatError("too many ids"));
        }
        let mut slots = Slots::default();
        slots.entries.reserve(n);
        for _ in 0..n {
            let hs = match self.int()? {
                0 => None,
                len => {
                    let len = len - 1;
                    if len > self.data.len() - self.pos {
                        return Err(FormatError("truncated"));
                    }
                    Some((0..len).map(|_| self.key()).collect::<Result<_, _>>()?)
                }
            };
            slots.entries.push(Slot {
                hs,
                refs: 0,
                pos: 0,
                alive: true,
            });
        }
        let entries = slots.entries.iter().filter(|s| s.hs.is_some()).count();
        if self.count(1)? != entries {
            return Err(FormatError("keys order does not list every entry"));
        }
        let mut seen = vec![false; n];
        for pos in 0..entries {
            let id = self.id(n)?;
            let slot = &mut slots.entries[id as usize];
            if slot.hs.is_none() || std::mem::replace(&mut seen[id as usize], true) {
                return Err(FormatError(
                    "keys order lists an id twice or without an entry",
                ));
            }
            slot.pos = pos;
            slots.order.push(id);
        }
        Ok((slots, num_tables))
    }

    /// `num_tables` tables of buckets; `sets`: members must be strictly
    /// ascending (`MinHashLSH`), else any list (the forest). Counts the
    /// references in `slots`.
    pub(crate) fn tables(
        &mut self,
        num_tables: usize,
        slots: &mut Slots,
        sets: bool,
    ) -> Result<Vec<Table>, FormatError> {
        let n = slots.entries.len();
        let mut tables = Vec::with_capacity(num_tables.min(self.data.len() - self.pos));
        for _ in 0..num_tables {
            let buckets = self.count(2)?;
            let mut table = Table::default();
            table.reserve(buckets);
            for _ in 0..buckets {
                let h = self.key()?;
                let len = self.count(1)?;
                let mut ids = Vec::with_capacity(len);
                for _ in 0..len {
                    let id = self.id(n)?;
                    if sets && ids.last().is_some_and(|&last| last >= id) {
                        return Err(FormatError("bucket ids are not ascending"));
                    }
                    slots.entries[id as usize].refs += 1;
                    ids.push(id);
                }
                if table.insert(h, Ids::from_vec(ids)).is_some() {
                    return Err(FormatError("a band key appears twice in one table"));
                }
            }
            tables.push(table);
        }
        Ok(tables)
    }

    /// Check that all input was read and every id is used.
    pub(crate) fn finish(&self, slots: &Slots) -> Result<(), FormatError> {
        if self.pos != self.data.len() {
            return Err(FormatError("trailing data"));
        }
        if slots.entries.iter().any(|s| s.hs.is_none() && s.refs == 0) {
            return Err(FormatError("an id is neither in keys nor in a bucket"));
        }
        Ok(())
    }
}

impl LshIndex {
    /// Serialize the index (for pickling). Ids are renumbered `0..n` in
    /// ascending order; the second value is the original id of each number.
    pub fn to_bytes(&self) -> (Vec<u8>, Vec<u32>) {
        let (mut w, ids) = Writer::new(&self.slots, self.tables.len());
        w.tables(&self.tables);
        (w.finish(KIND_LSH), ids)
    }

    /// The index [`Self::to_bytes`] wrote, with ids `0..n`.
    pub fn from_bytes(data: &[u8]) -> Result<Self, FormatError> {
        let mut r = Reader::new(data, KIND_LSH)?;
        let (mut slots, num_tables) = r.slots()?;
        let tables = r.tables(num_tables, &mut slots, true)?;
        r.finish(&slots)?;
        Ok(Self { slots, tables })
    }

    /// The number of allocated ids (after [`Self::from_bytes`]: they are `0..n`).
    pub fn id_count(&self) -> usize {
        self.slots.alive_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(bytes: &[u8]) -> BandKey {
        BandKey::from(bytes)
    }

    fn keys(hs: &[&[u8]]) -> Vec<BandKey> {
        hs.iter().map(|h| key(h)).collect()
    }

    /// Everything observable: keys in order, and each table's buckets sorted.
    type Dump = (Vec<(u32, Vec<Vec<u8>>)>, Vec<Vec<(Vec<u8>, Vec<u32>)>>);

    fn dump(index: &LshIndex) -> Dump {
        let entries = index
            .ids()
            .map(|id| {
                let hs = index.keys_of(id).unwrap().iter().map(|h| h.to_vec());
                (id, hs.collect())
            })
            .collect();
        let tables = (0..index.num_tables())
            .map(|t| {
                let mut b: Vec<_> = index
                    .buckets(t)
                    .map(|(h, ids)| (h.to_vec(), ids.to_vec()))
                    .collect();
                b.sort();
                b
            })
            .collect();
        (entries, tables)
    }

    #[test]
    fn swapped_rows_reverses_each_value() {
        let v: [u64; 3] = [0x0102_0304_0506_0708, 1, u64::MAX - 1];
        let mut want = Vec::new();
        for x in &v[..2] {
            want.extend_from_slice(&x.swap_bytes().to_ne_bytes());
        }
        assert_eq!(swapped_rows(&v, 1, 0, 2), want);
        if cfg!(target_endian = "little") {
            assert_eq!(swapped_rows(&v, 1, 0, 1), [1, 2, 3, 4, 5, 6, 7, 8]);
        }
        // Clamped like a slice; empty when start >= end or past the rows.
        assert_eq!(swapped_rows(&v, 1, 2, 10).len(), 8);
        assert_eq!(swapped_rows(&v, 1, 3, 9), [0_u8; 0]);
        assert_eq!(swapped_rows(&v, 1, 2, 1), [0_u8; 0]);
        assert_eq!(swapped_rows(&v, 0, 0, 1), [0_u8; 0]);
        let w: [u32; 2] = [0x0a0b_0c0d, 7];
        assert_eq!(swapped_rows(&w, 1, 0, 2).len(), 8);
        let f: [f64; 1] = [1.5];
        assert_eq!(
            swapped_rows(&f, 1, 0, 1),
            1.5_f64.to_bits().swap_bytes().to_ne_bytes()
        );
    }

    #[test]
    fn rows_of_two_values() {
        // A WeightedMinHash: (k, 2) int64, row-major.
        let v: Vec<i64> = (0..8).collect();
        let row1: Vec<u8> = [2_i64, 3]
            .iter()
            .flat_map(|x| x.swap_bytes().to_ne_bytes())
            .collect();
        assert_eq!(swapped_rows(&v, 2, 1, 2), row1);
        assert_eq!(swapped_rows(&v, 2, 3, 7).len(), 16);
    }

    #[test]
    fn byte_and_typed_forms_agree() {
        let v: Vec<u32> = (0..37_u32).map(|i| i.wrapping_mul(2_654_435_761)).collect();
        let ne: Vec<u8> = v.iter().flat_map(|x| x.to_ne_bytes()).collect();
        for bands in [
            Bands { b: 5, r: 7 },
            Bands { b: 9, r: 4 },
            Bands { b: 3, r: 0 },
        ] {
            assert_eq!(band_keys(&v, 1, bands), band_keys_ne(&ne, 4, 4, bands));
        }
        assert_eq!(band_keys(&v, 1, Bands { b: 2, r: 3 })[1].len(), 12);
        let many =
            band_keys_many(&v[..36], 4, 9, 1, Bands { b: 3, r: 3 }, par::Threads::Auto).unwrap();
        for (j, hs) in many.iter().enumerate() {
            assert_eq!(
                *hs,
                band_keys(&v[j * 9..j * 9 + 9], 1, Bands { b: 3, r: 3 })
            );
        }
        // Huge r saturates instead of overflowing.
        assert!(
            band_keys(
                &v,
                1,
                Bands {
                    b: 3,
                    r: usize::MAX
                }
            )[2]
            .is_empty()
        );
    }

    #[test]
    fn ids_set_and_list() {
        let mut s = Ids::default();
        assert!(s.insert_sorted(5) && !s.insert_sorted(5));
        assert!(s.insert_sorted(2) && s.insert_sorted(9));
        assert_eq!(s.as_slice(), [2, 5, 9]);
        assert!(s.remove_sorted(5) && !s.remove_sorted(5));
        assert!(s.remove_sorted(2) && s.remove_sorted(9));
        assert!(s.is_empty());
        let mut l = Ids::default();
        for id in [3, 1, 3] {
            l.push(id);
        }
        assert_eq!(l.as_slice(), [3, 1, 3]);
    }

    #[test]
    fn insert_query_remove() {
        let mut index = LshIndex::new(2);
        assert!(index.has_empty_table());
        let a = index.alloc().unwrap();
        let b = index.alloc().unwrap();
        index.insert(a, &keys(&[b"x", b"y"]));
        index.insert(b, &keys(&[b"x", b"z"]));
        assert!(!index.has_empty_table() && index.len() == 2);
        assert_eq!(index.query(&[b"x", b"q"]), [a, b]);
        assert_eq!(index.query(&[b"q", b"z"]), [b]);
        assert_eq!(index.query::<&[u8]>(&[]), Vec::<u32>::new());
        assert_eq!(index.bucket(0, b"x"), Some(&[a, b][..]));
        assert_eq!(index.remove(a), Ok(true));
        assert!(!index.is_alive(a) && !index.contains(a));
        assert_eq!(index.bucket(1, b"y"), None);
        assert_eq!(index.bucket(0, b"x"), Some(&[b][..]));
        assert_eq!(index.remove(a), Err(RemoveError::NotFound));
        // The freed id is reused.
        assert_eq!(index.alloc().unwrap(), a);
    }

    #[test]
    fn extended_keys_leave_dangling_members() {
        // insert(k, m1); insert(k, m2, check_duplication=False); remove(k).
        let mut index = LshIndex::new(2);
        let k = index.alloc().unwrap();
        index.insert(k, &keys(&[b"a1", b"a2"]));
        index.insert(k, &keys(&[b"b1", b"b2"]));
        assert_eq!(index.keys_of(k).unwrap().len(), 4);
        // Only zip(keys[k], tables), the first b band keys, are removed.
        assert_eq!(index.remove(k), Ok(false));
        assert!(index.is_alive(k) && !index.contains(k) && index.is_empty());
        assert_eq!(index.query(&[b"b1", b"zz"]), [k]);
        // A later insert of the same key finds the same members.
        index.insert(k, &keys(&[b"b1", b"c2"]));
        assert_eq!(index.bucket(0, b"b1"), Some(&[k][..]));
        assert_eq!(index.remove(k), Ok(false));
        assert_eq!(index.bucket(0, b"b1"), None);
        assert_eq!(index.bucket(1, b"b2"), Some(&[k][..]));
    }

    #[test]
    fn remove_missing_member_is_keyerror_with_datasketch_side_effects() {
        let mut index = LshIndex::new(3);
        let k = index.alloc().unwrap();
        index.insert(k, &keys(&[b"a", b"b", b"c"]));
        // Take k out of band 1's bucket behind the index's back.
        index.tables[1].get_mut(&b"b"[..]).unwrap().remove_sorted(k);
        assert_eq!(index.remove(k), Err(RemoveError::Missing { table: 1 }));
        assert_eq!(index.bucket(0, b"a"), None, "band 0 was updated");
        assert_eq!(index.bucket(2, b"c"), Some(&[k][..]), "band 2 was not");
        assert!(index.contains(k), "the keys entry stays");
        // Again: band 0's bucket is gone now, so defaultdict creates it empty.
        assert_eq!(index.remove(k), Err(RemoveError::Missing { table: 0 }));
        assert_eq!(index.bucket(0, b"a"), Some(&[][..]));
        assert_eq!(index.table_len(0), 1);
    }

    #[test]
    fn keys_keep_dict_order() {
        let mut index = LshIndex::new(1);
        let ids: Vec<u32> = (0..200).map(|_| index.alloc().unwrap()).collect();
        for (i, &id) in ids.iter().enumerate() {
            index.insert(id, &[key(&[u8::try_from(i % 256).unwrap()])]);
        }
        // Remove enough to compact the order, then re-insert one: it goes last.
        for &id in &ids[..150] {
            index.remove(id).unwrap();
        }
        let again = index.alloc().unwrap();
        index.insert(again, &[key(b"again")]);
        let mut want: Vec<u32> = ids[150..].to_vec();
        want.push(again);
        assert_eq!(index.ids().collect::<Vec<_>>(), want);
        assert_eq!(index.len(), 51);
    }

    #[test]
    fn insert_many_equals_sequential_inserts() {
        let mut items = Vec::new();
        let mut one = LshIndex::new(4);
        for j in 0..500_u32 {
            let id = if j % 7 == 3 { j - 1 } else { j }; // some repeated ids
            let hs: Vec<BandKey> = (0..4_u32)
                .map(|t| key(&((j / 3) * 4 + t % 2).to_le_bytes()))
                .collect();
            items.push((id, hs));
        }
        let mut many = one.clone();
        for index in [&mut one, &mut many] {
            for _ in 0..500 {
                index.alloc().unwrap();
            }
        }
        for (id, hs) in &items {
            one.insert(*id, hs);
        }
        for t in [par::Threads::ONE, par::Threads::Auto] {
            let mut m = many.clone();
            m.insert_many(&items, t).unwrap();
            assert_eq!(dump(&m), dump(&one));
            for (id, _) in &items {
                assert_eq!(
                    m.slots.slot(*id).unwrap().refs,
                    one.slots.slot(*id).unwrap().refs
                );
            }
        }
        let queries: Vec<Vec<BandKey>> = items.iter().map(|(_, hs)| hs.clone()).collect();
        // query_values computes the same band keys on the fly.
        let values: Vec<u64> = (0..500_u64 * 8).map(|v| (v / 3) % 17).collect();
        let mut by_values = LshIndex::new(4);
        for j in 0..500 {
            let id = by_values.alloc().unwrap();
            by_values.insert(
                id,
                &band_keys(&values[j * 8..j * 8 + 8], 1, Bands { b: 4, r: 2 }),
            );
        }
        for j in 0..500 {
            let row = &values[j * 8..j * 8 + 8];
            let bands = Bands { b: 4, r: 2 };
            assert_eq!(
                by_values.query_values(row, 1, bands),
                by_values.query(&band_keys(row, 1, bands))
            );
            let fewer = Bands { b: 2, r: 2 };
            assert_eq!(
                by_values.query_values(row, 1, fewer),
                by_values.query(&band_keys(row, 1, fewer))
            );
        }
        let many = by_values
            .query_values_many(&values, 500, 8, 1, Bands { b: 4, r: 2 }, par::Threads::Auto)
            .unwrap();
        for (j, ids) in many.iter().enumerate() {
            assert_eq!(
                *ids,
                by_values.query_values(&values[j * 8..j * 8 + 8], 1, Bands { b: 4, r: 2 })
            );
        }
        let got = one.query_many(&queries, par::Threads::Auto).unwrap();
        for (q, ids) in queries.iter().zip(got) {
            assert_eq!(ids, one.query(q));
        }
    }

    #[test]
    fn bytes_round_trip() {
        let mut index = LshIndex::new(3);
        let ids: Vec<u32> = (0..6).map(|_| index.alloc().unwrap()).collect();
        for (i, &id) in ids.iter().enumerate() {
            let i = u8::try_from(i).unwrap();
            index.insert(id, &keys(&[&[i % 2], &[i % 3, 9], b""]));
        }
        index.insert(ids[1], &keys(&[b"x", b"y", b"z"]));
        index.remove(ids[1]).unwrap(); // dangling in x/y/z
        index.remove(ids[2]).unwrap(); // freed
        let (bytes, numbered) = index.to_bytes();
        assert_eq!(numbered, [ids[0], ids[1], ids[3], ids[4], ids[5]]);
        let back = LshIndex::from_bytes(&bytes).unwrap();
        let renumber = |d: Dump| -> Dump {
            let n =
                |id: u32| u32::try_from(numbered.iter().position(|&x| x == id).unwrap()).unwrap();
            (
                d.0.into_iter().map(|(id, hs)| (n(id), hs)).collect(),
                d.1.into_iter()
                    .map(|t| {
                        t.into_iter()
                            .map(|(h, m)| (h, m.into_iter().map(n).collect()))
                            .collect()
                    })
                    .collect(),
            )
        };
        assert_eq!(dump(&back), renumber(dump(&index)));
        assert_eq!(back.id_count(), 5);
        assert_eq!(back.query(&[b"x"]), [1]);
        // An empty index, and one with an empty bucket.
        let empty = LshIndex::new(2);
        assert_eq!(
            dump(&LshIndex::from_bytes(&empty.to_bytes().0).unwrap()),
            dump(&empty)
        );
    }

    #[test]
    fn corrupt_bytes_are_errors_not_panics() {
        let mut index = LshIndex::new(2);
        for i in 0..20_u8 {
            let id = index.alloc().unwrap();
            index.insert(id, &keys(&[&[i % 4], &[i, i]]));
        }
        let (bytes, _) = index.to_bytes();
        for cut in 0..bytes.len() {
            assert!(LshIndex::from_bytes(&bytes[..cut]).is_err(), "cut at {cut}");
        }
        for at in 0..bytes.len() {
            for flip in [1_u8, 0x80, 0xff] {
                let mut bad = bytes.clone();
                bad[at] ^= flip;
                let _ = LshIndex::from_bytes(&bad);
            }
        }
        assert!(LshIndex::from_bytes(b"pyroidDSF\x01\x00").is_err());
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(LshIndex::from_bytes(&extra).is_err());
    }
}
