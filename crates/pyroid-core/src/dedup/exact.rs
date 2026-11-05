//! Exact deduplication: keep the first copy of each distinct text.
//!
//! Each document (optionally passed through `simplify_text` first) is reduced
//! to a 128-bit XXH3 hash, and a document is kept unless an earlier one had
//! the same hash. With 10⁹ documents the chance that any two different texts
//! share a hash is about 10⁻²¹, so this is exact for any practical corpus.
//! Unlike a set of strings, the state is 16 bytes per distinct document,
//! which is what lets [`ExactSeen`] stream across batches.

use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hasher};

use twox_hash::XxHash3_128;

use super::config::{TextNormConfig, UnicodeVersion};
use super::simplify::simplify_text;

/// The 128-bit content hash of one document.
pub fn content_hash(text: &str) -> u128 {
    XxHash3_128::oneshot(text.as_bytes())
}

/// The content hash after `simplify_text(text, norm)` (or of `text` itself).
pub fn normalized_hash(text: &str, norm: Option<&TextNormConfig>, unicode: UnicodeVersion) -> u128 {
    match norm {
        None => content_hash(text),
        Some(cfg) => content_hash(&simplify_text(text, cfg, unicode)),
    }
}

/// The keys are already uniform hashes: use their low 64 bits as is.
#[derive(Default)]
struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        // Only reached for non-u128 keys, which this map never uses.
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }

    fn write_u128(&mut self, i: u128) {
        // Truncation intended: the low half of a uniform hash.
        #[allow(clippy::cast_possible_truncation)]
        let low = i as u64;
        self.0 = low;
    }
}

/// The hashes seen so far, across any number of batches.
#[derive(Default)]
pub struct ExactSeen {
    seen: HashSet<u128, BuildHasherDefault<PassThrough>>,
}

impl std::fmt::Debug for ExactSeen {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExactSeen")
            .field("distinct", &self.seen.len())
            .finish()
    }
}

impl ExactSeen {
    pub fn new() -> Self {
        Self::default()
    }

    /// Distinct documents seen.
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }

    pub fn clear(&mut self) {
        self.seen.clear();
    }

    /// Record a batch in order; `true` for each first copy. `None` entries
    /// (Arrow nulls) stay `None` and are not recorded.
    pub fn keep_first(&mut self, hashes: &[Option<u128>]) -> Vec<Option<bool>> {
        self.seen.reserve(hashes.len());
        hashes
            .iter()
            .map(|h| h.map(|h| self.seen.insert(h)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_first_copies_across_batches() {
        let mut seen = ExactSeen::new();
        let h = |s: &str| Some(content_hash(s));
        assert_eq!(
            seen.keep_first(&[h("a"), h("b"), h("a"), None, h("")]),
            [Some(true), Some(true), Some(false), None, Some(true)]
        );
        assert_eq!(
            seen.keep_first(&[h("b"), h("c"), h("")]),
            [Some(false), Some(true), Some(false)]
        );
        assert_eq!(seen.len(), 4);
    }

    #[test]
    fn normalization_merges_variants() {
        let norm = TextNormConfig::default();
        let v = UnicodeVersion::LATEST;
        assert_ne!(content_hash("Hello, World"), content_hash("hello world"));
        assert_eq!(
            normalized_hash("Hello, World!", Some(&norm), v),
            normalized_hash("  hello   world ", Some(&norm), v)
        );
    }
}
