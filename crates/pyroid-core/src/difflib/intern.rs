//! Element interners: `b`'s distinct elements get ids `0..n` in
//! first-occurrence order; `a`'s elements are looked up ([`NONE`] if absent).

use std::borrow::Borrow;
use std::hash::Hash;

use rustc_hash::FxHashMap;

use super::matcher::NONE;

/// Ids for hashable keys.
#[derive(Debug, Clone)]
pub struct Interner<K> {
    map: FxHashMap<K, u32>,
}

impl<K> Default for Interner<K> {
    fn default() -> Self {
        Self {
            map: FxHashMap::default(),
        }
    }
}

impl<K: Hash + Eq> Interner<K> {
    /// The id of `key`, numbering it if it is new. Returns `(id, new)`.
    #[allow(clippy::cast_possible_truncation)] // fewer than u32::MAX keys (BIndex checks the length)
    pub fn intern(&mut self, key: K) -> (u32, bool) {
        let next = self.map.len() as u32;
        let mut new = false;
        let id = *self.map.entry(key).or_insert_with(|| {
            new = true;
            next
        });
        (id, new)
    }

    /// The id of `key`, or [`NONE`].
    pub fn get<Q>(&self, key: &Q) -> u32
    where
        K: Borrow<Q>,
        Q: Hash + Eq + ?Sized,
    {
        self.map.get(key).copied().unwrap_or(NONE)
    }

    /// Number of keys.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// True if there are no keys.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// Ids for code points, with a table for ASCII.
#[derive(Debug, Clone)]
pub struct CharInterner {
    ascii: [u32; 128],
    other: FxHashMap<u32, u32>,
    chars: Vec<u32>,
}

impl Default for CharInterner {
    fn default() -> Self {
        Self {
            ascii: [NONE; 128],
            other: FxHashMap::default(),
            chars: Vec::new(),
        }
    }
}

impl CharInterner {
    /// Number `chars` (the elements of `b`) and return their ids.
    pub fn build(chars: impl IntoIterator<Item = u32>) -> (Self, Vec<u32>) {
        let mut this = Self::default();
        let ids = chars.into_iter().map(|c| this.intern(c)).collect();
        (this, ids)
    }

    #[allow(clippy::cast_possible_truncation)] // < u32::MAX distinct code points
    fn intern(&mut self, c: u32) -> u32 {
        let next = self.chars.len() as u32;
        let slot = if c < 128 {
            &mut self.ascii[c as usize]
        } else {
            self.other.entry(c).or_insert(NONE)
        };
        if *slot == NONE {
            *slot = next;
            self.chars.push(c);
        }
        *slot
    }

    /// Number `chars` afresh, reusing this interner's memory; the ids go to `ids`.
    pub fn rebuild(&mut self, chars: impl IntoIterator<Item = u32>, ids: &mut Vec<u32>) {
        for &c in &self.chars {
            if c < 128 {
                self.ascii[c as usize] = NONE;
            }
        }
        self.other.clear();
        self.chars.clear();
        ids.clear();
        ids.extend(chars.into_iter().map(|c| self.intern(c)));
    }

    /// The id of code point `c`, or [`NONE`].
    pub fn get(&self, c: u32) -> u32 {
        if c < 128 {
            self.ascii[c as usize]
        } else {
            self.other.get(&c).copied().unwrap_or(NONE)
        }
    }

    /// Encode `a` against these ids.
    pub fn encode(&self, a: impl IntoIterator<Item = u32>) -> Vec<u32> {
        a.into_iter().map(|c| self.get(c)).collect()
    }

    /// The code point of each id, in id order.
    pub fn chars(&self) -> &[u32] {
        &self.chars
    }
}

/// The code points of `s`.
pub fn code_points(s: &str) -> impl Iterator<Item = u32> + '_ {
    s.chars().map(u32::from)
}
