//! Pure-Rust kernels behind pyroid.
//!
//! This crate has no Python dependency: everything here is plain, safe Rust
//! that is unit-tested with `cargo test`. The `pyroid-py` crate wraps it for
//! Python (argument conversion, detaching from the interpreter, Arrow FFI).
//!
//! - [`par`]: the one shared, lazily built, fork-aware thread pool and the
//!   order-preserving batch maps every kernel runs on.
//! - [`columnar`]: builders for Arrow-layout buffers (utf8, large utf8,
//!   boolean) so the bindings can export results without an Arrow dependency.
//! - [`text`]: text kernels (`pyroid.text`, the demonstration module).
//! - [`dedup`]: MinHash and exact deduplication (`pyroid.dedup`).
//!
//! Product tracks add a `src/<mod>/` directory here; see `docs/dev/MODULES.md`.
#![forbid(unsafe_code)]

pub mod columnar;
pub mod dedup;
pub mod fasttext;
pub mod filters;
pub mod ftfy;
pub mod par;
pub mod text;
