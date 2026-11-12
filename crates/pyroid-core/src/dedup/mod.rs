// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! Near-duplicate and exact deduplication behind `pyroid.dedup`.
//!
//! The MinHash pipeline makes the same decisions as datatrove 0.10.1's
//! `MinhashDedupSignature` -> `MinhashDedupBuckets` -> `MinhashDedupCluster`
//! -> `MinhashDedupFilter`, in memory:
//!
//! - [`simplify`]: `simplify_text`, the normalisation before shingling.
//! - [`tokenize`]: the word tokenizer: spaCy English, as in datatrove
//!   (pyroid's port in `crate::filters::tokenizer`).
//! - [`minhash`]: shingles, the permutation parameters and signatures.
//! - [`cluster`]: bucket matching and the union-find that picks which
//!   document of each cluster is kept.
//! - [`exact`]: hash-based exact deduplication (keep the first copy).
//! - [`files`]: stages 1-3 on disk, with datatrove's intermediate files
//!   byte for byte (`.minhash.sig`, `.dups`, `.remove` / `.clusters` /
//!   `.sizes`), in bounded memory.

pub mod cluster;
pub mod config;
pub mod exact;
pub mod files;
pub mod minhash;
mod rng;
pub mod simplify;
mod simplify_tables;
pub mod tokenize;

pub use cluster::{CapacityError, Clusters, SignatureStore};
pub use config::{
    ConfigError, HashConfig, HashFn, MinhashConfig, Precision, TextNormConfig, UnicodeVersion,
};
pub use exact::ExactSeen;
pub use files::{
    BucketOptions, ClusterOptions, ClusterSummary, MinhashFileError, SigFileWriter,
    minhash_buckets, minhash_cluster, worker_hash_range,
};
pub use minhash::MinHasher;
pub use rng::MERSENNE_PRIME;
pub use simplify::simplify_text;
pub use tokenize::{SpacyTokenizer, TokenizeError, WhitespaceTokenizer, WordTokenizer};

// More datatrove 0.10.1 steps.
pub mod sentence;
