// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! The kernels behind `pyroid.datasketch`, an exact drop-in for datasketch
//! 2.0.0's MinHash family.
//!
//! - [`rng`]: numpy's legacy `RandomState` (MT19937 + masked `randint`), which
//!   draws the permutation parameters.
//! - [`minhash`]: the SHA1 hash functions, the three permutation schemes
//!   (`legacy`, `affine32`, `affine64`) and signature updates.
//! - [`shingle`]: word n-gram shingles of a text (`str.split()` words).
//! - [`quad`]: QUADPACK's `dqagse`, as `scipy.integrate.quad` runs it.
//! - [`params`]: the LSH and LSH Ensemble `(b, r)` searches built on it.
//! - [`lsh`]: the in-memory band index behind `MinHashLSH`.
//! - [`forest`]: the sorted-prefix index behind `MinHashLSHForest`.

pub mod forest;
pub mod lsh;
pub mod minhash;
pub mod params;
pub mod quad;
pub mod rng;
pub mod shingle;
