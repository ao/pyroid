//! A lean, hand-rolled Arrow C Data / C Stream importer and exporter for
//! string and boolean columns, used through the Arrow `PyCapsule` interface.
//!
//! It replaces pyo3-arrow + arrow-rs, which grow the extension from ~0.5 MB to
//! ~9 MB (measured when the Arrow interface was chosen). Scope is deliberately small: import
//! `utf8`/`large_utf8`/`utf8_view`; export `utf8`/`large_utf8`/`bool`/`large_binary`, and
//! record batches of those as struct arrays (`record_batch`).
//! All `unsafe` lives in this module; each block states why it is sound.

mod column;
mod ffi;
mod import;
mod record_batch;

pub use column::{Chunk, Column, ColumnType, build_error};
pub use import::{ArrowStrings, StrChunk, StrType, import};
pub use record_batch::{Field, RecordBatch, array_capsules, schema_capsule, stream_capsule};
