# fuzz/arrow: the Arrow C Data importer

A cargo-fuzz project of its own (own `[workspace]`) for the hand-rolled
importer in `crates/pyroid-py/src/arrow/{ffi,import}.rs`. pyroid-py is a PyO3
cdylib and can't be a dependency, so `fuzz_targets/arrow_import.rs` compiles
the real files verbatim through `#[path]` modules:

```rust
#[path = "../../../crates/pyroid-py/src/errors.rs"]
mod errors; // import.rs uses crate::errors::{input_error, type_name}

#[path = "../../../crates/pyroid-py/src/arrow"] // inline module: its `mod`
mod arrow {                                     // items load from that dir
    pub mod ffi;
    pub mod import; // finds super::ffi as in pyroid-py
}
```

Plain `mod` items (no per-item `#[path]`) keep `ffi.rs` / `import.rs`
ordinary module files, so their `#[cfg(test)] mod tests;` still means
`ffi/tests.rs` / `import/tests.rs` (not compiled here, but rustfmt follows
it; with a per-item `#[path]` it would look for `arrow/tests.rs` and fail).
Format with `rustfmt --edition 2024 fuzz/arrow/fuzz_targets/arrow_import.rs`.

The binary links libpython (pyo3 without `extension-module`) but never
initializes or calls Python. Dependencies: libfuzzer-sys, pyo3 =0.30.1 and
simdutf8 0.1.5, the versions pyroid-py uses.

## Running

From the repository root, on Linux, with a nightly toolchain, `cargo-fuzz` and a C++ compiler (see
[`../README.md`](../README.md)) and a CPython 3.12 that has its shared library:

```sh
export PYO3_PYTHON=/path/to/python3.12                 # pyo3's build script reads its config
export LD_LIBRARY_PATH=$(dirname "$PYO3_PYTHON")/../lib  # where libpython3.12.so is, at run time
cargo +nightly fuzz run --fuzz-dir fuzz/arrow arrow_import -- \
    -max_total_time=1200 -rss_limit_mb=1536 -max_len=16384 -timeout=10
```

`--fuzz-dir` is needed: run from `fuzz/arrow` without it, cargo-fuzz picks
the enclosing `fuzz/` project. The corpus is `corpus/arrow_import`
(hand-written seeds plus `cargo fuzz cmin` output).

## What it checks

The C Data Interface carries no buffer lengths, and the importer trusts the
producer on them (import.rs, "Soundness"). So the harness builds each
`ArrowArray` like a conformant producer, every buffer **exactly** as long as
the spec guarantees, each in its own `malloc` allocation (a 0-byte buffer is
still a distinct allocation), so ASan flags any read past one:

| buffer | size |
|---|---|
| `buffers` array | `n_buffers` pointers |
| validity | `ceil((offset + length) / 8)` bytes |
| utf8 / large_utf8 offsets | `offset + length + 1` entries (int32 / int64) |
| utf8 data | the last offset's bytes when the window's offsets are valid; **0 bytes** when they are not (the importer must reject before reading data) |
| utf8_view views | `offset + length` views of 16 bytes |
| variadic sizes | one int64 per variadic buffer |
| each variadic buffer | its declared size (0 bytes if negative) |

Offsets and views before `offset` hold poison (negative) values, so reading
outside the window shows up as a mismatch. Sizes are capped (1024 rows, offset
64, variadic buffers of 2 KiB, utf8 data of 1 MiB; inputs that would need more
are skipped), so one input allocates at most about 1.1 MiB.

Per input, the harness predicts from the spec and its own bytes whether the
importer must accept the array, and asserts:

- `StrChunk::new` accepts exactly the arrays the oracle accepts. Rejected:
  released arrays, negative or overflowing `length`/`offset`, children or a
  dictionary, a wrong `n_buffers` (utf8: 3; view: at least 3, the last being
  the sizes buffer), a null `buffers`, `null_count` other than 0, -1 or
  positive-with-bitmap; and, when `length > 0`, null offsets / views / sizes
  buffers, negative or decreasing offsets in the window, a null data buffer
  under non-empty rows, negative variadic sizes, null variadic buffers of
  non-zero size, and non-null rows whose view has a negative length or a long
  view outside its buffer. Inconsistent `null_count`s (`0` with a bitmap that
  has zeros, positive counts that disagree with the bitmap) are accepted:
  `0` means "ignore the bitmap", anything else uses it.
- The release callback (which frees every buffer and counts its calls) runs
  exactly once: inside `new` when it returns `Err`, at `drop` after `Ok`;
  never for a released array (the harness frees that one itself).
- On success, for every row: null exactly when the bitmap says so (and
  `null_count != 0`); `bytes(i)` equals the bytes the offsets / view describe;
  `get(i)` is `Ok` (with the same bytes) iff they are valid UTF-8.

## Input format

Little-endian; reading past the end gives zeros (bulk bytes give `'a'`).

```
u8  type          % 3: utf8, large_utf8, utf8_view
u8  flags         0x01 released (release = NULL)   0x02 buffers = NULL
                  0x04 dictionary                  0x08 n_children = next i8
                  0x10 n_buffers = next i8 (views: only values < 3 apply)
                  0x20 header only: length, offset = SPECIAL[next u8 % 10]
                       each (negative / overflowing only; others skipped)
                  0x40 validity bitmap             0x80 null mask = next u8
                       (bit k: buffers[k] = NULL)
[i8 n_children] [i8 n_buffers] [u8 null mask]      in this order, if flagged
u16 length % 1025, u8 offset % 65
[validity bytes]
u8  null_count    % 8: 0,1 -> 0; 2 -> -1; 3,4 -> the bitmap's zeros;
                  5 -> 1 + next u8; 6 -> negative; 7 -> i64::MAX
utf8 / large_utf8:
  u8 mode % 4     0 valid (first 0), 1 valid + one corruption, 2 raw, 3 valid from u16
  per entry (length + 1): mode 2: i32/i64 raw; else u8 step % 17 after it
  mode 1: u16 position, u8 kind (-1, MIN, decrease by 1 + next u8, MAX)
  data bytes
utf8_view:
  u8 variadic count % 4; per buffer: u16 size (0x8000: negative), its bytes
  per view, u8 kind % 6: 0 inline (u8 len % 13, bytes); 1-2 long in range
  (u8 buffer, u16 len, u16 offset, reduced into range); 3 long, likely out of
  range; 4 raw 16 bytes; 5 negative length
```

## Not covered

- `parse_schema` (and `describe`, `c_string` on schema strings) is private to
  `import.rs` and only reachable through `import()`, which needs Python
  objects (`__arrow_c_array__` / `__arrow_c_stream__`), so the harness can't
  reach it without calling Python. Same for `take`/`move_out` from capsules
  and the stream loop (`import_stream`).
- `column.rs` (export) is not compiled.
- Producers that mutate buffers while a chunk is live (the `Sync` note in
  import.rs), and huge-but-valid lengths no allocation could back.
