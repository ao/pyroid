// Derived from trafilatura 2.3.1 (Apache-2.0), © Adrien Barbaresi; modified.
//! deduplication.LRUCache: an OrderedDict of string -> count with a maximum size.

use std::collections::{BTreeMap, HashMap};
use std::num::NonZeroUsize;

/// `trafilatura.deduplication.LRUCache`. Trafilatura's module-level cache (`LRU_TEST`, used
/// by `deduplicate=True`) is `LRUCache(maxsize=4096)`, which is what [`Default`] gives.
#[derive(Clone, Debug)]
pub struct LruCache {
    maxsize: usize,
    seq: u64,
    /// key -> (position, value)
    map: HashMap<String, (u64, i64)>,
    /// position -> key, oldest first
    order: BTreeMap<u64, String>,
}

/// settings.LRU_SIZE.
pub const LRU_SIZE: usize = 4096;

impl Default for LruCache {
    fn default() -> Self {
        LruCache::new(NonZeroUsize::new(LRU_SIZE).unwrap_or(NonZeroUsize::MIN))
    }
}

impl LruCache {
    /// `LRUCache(maxsize)` (Trafilatura raises ValueError for `maxsize < 1`).
    pub fn new(maxsize: NonZeroUsize) -> Self {
        LruCache {
            maxsize: maxsize.get(),
            seq: 0,
            map: HashMap::new(),
            order: BTreeMap::new(),
        }
    }

    /// `maxsize`.
    pub fn maxsize(&self) -> usize {
        self.maxsize
    }

    /// `len(cache.cache)`.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache holds nothing.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn touch(&mut self, key: &str, value: i64) {
        if let Some((s, _)) = self.map.remove(key) {
            self.order.remove(&s);
        }
        self.seq += 1;
        self.map.insert(key.to_owned(), (self.seq, value));
        self.order.insert(self.seq, key.to_owned());
    }

    /// `get(key)`: the value (made most recent), or -1.
    pub fn get(&mut self, key: &str) -> i64 {
        match self.map.get(key) {
            Some(&(_, v)) => {
                self.touch(key, v);
                v
            }
            None => -1,
        }
    }

    /// `put(key, value)`: store as most recent, evicting the oldest entry past maxsize.
    pub fn put(&mut self, key: &str, value: i64) {
        self.touch(key, value);
        if self.map.len() > self.maxsize {
            if let Some((_, k)) = self.order.pop_first() {
                self.map.remove(&k);
            }
        }
    }

    /// `increment(key)`: store the previous count + 1 (at least 1) and return the previous
    /// count, or -1.
    pub fn increment(&mut self, key: &str) -> i64 {
        let previous = match self.map.remove(key) {
            Some((s, c)) => {
                self.order.remove(&s);
                c
            }
            None => -1,
        };
        self.put(key, previous.max(0) + 1);
        previous
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.map.clear();
        self.order.clear();
    }

    /// The entries, oldest first (`list(cache.cache.items())`).
    pub fn items(&self) -> impl Iterator<Item = (&str, i64)> + '_ {
        self.order
            .values()
            .map(|k| (k.as_str(), self.map.get(k).map_or(0, |&(_, v)| v)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru() {
        let mut c = LruCache::new(NonZeroUsize::new(2).unwrap());
        assert_eq!(c.increment("a"), -1);
        assert_eq!(c.increment("a"), 1);
        assert_eq!(c.increment("b"), -1);
        assert_eq!(c.get("a"), 2);
        assert_eq!(c.increment("c"), -1); // evicts b, the oldest
        assert_eq!(c.get("b"), -1);
        assert_eq!(c.items().collect::<Vec<_>>(), vec![("a", 2), ("c", 1)]);
        c.put("a", 7);
        assert_eq!(c.get("a"), 7);
        c.clear();
        assert!(c.is_empty());
    }
}
