//! brotlicffi 1.2.1.0's `Decompressor`, as warcio 1.8.2 uses it for `Content-Encoding: br`.
//!
//! warcio calls `decompress(data)` once per block it reads, without `output_buffer_limit`, and
//! `flush()` when it closes the decompressor. This is that object over the pure-Rust
//! `brotli-decompressor` crate (a port of Google's C decoder; brotlicffi bundles the C library):
//! the same output for each call, and the same `brotlicffi.error` at the same call with the
//! same `str(e)`. A failing call returns no output (brotlicffi discards what it had decoded).
//! After the stream ends, any further input fails with `trailing data after stream.`; after a
//! decoding error, every later call fails again with the same error (or, if the failing call
//! left input unconsumed, with the `can_accept_more_data()` error when given more data).
//!
//! brotli-decompressor 6 reports the same error codes as libbrotli, including
//! `_ERROR_FORMAT_BLOCK_LENGTH_2` for a command that runs past the end of its metablock (6.2.0)
//! and `_ERROR_FORMAT_PADDING_1` for non-zero padding after a metadata or uncompressed
//! metablock header, so its codes are passed through unchanged.

use brotli_decompressor::{BrotliDecompressStream, BrotliResult, BrotliState, StandardAlloc};

use super::{Error, PyExc};

/// `BrotliDecoderErrorString` of brotli 1.2 (the library brotlicffi 1.2.1.0 bundles).
fn error_string(code: i32) -> &'static str {
    match code {
        0 => "_NO_ERROR",
        1 => "_SUCCESS",
        2 => "_NEEDS_MORE_INPUT",
        3 => "_NEEDS_MORE_OUTPUT",
        -1 => "_ERROR_FORMAT_EXUBERANT_NIBBLE",
        -2 => "_ERROR_FORMAT_RESERVED",
        -3 => "_ERROR_FORMAT_EXUBERANT_META_NIBBLE",
        -4 => "_ERROR_FORMAT_SIMPLE_HUFFMAN_ALPHABET",
        -5 => "_ERROR_FORMAT_SIMPLE_HUFFMAN_SAME",
        -6 => "_ERROR_FORMAT_CL_SPACE",
        -7 => "_ERROR_FORMAT_HUFFMAN_SPACE",
        -8 => "_ERROR_FORMAT_CONTEXT_MAP_REPEAT",
        -9 => "_ERROR_FORMAT_BLOCK_LENGTH_1",
        -10 => "_ERROR_FORMAT_BLOCK_LENGTH_2",
        -11 => "_ERROR_FORMAT_TRANSFORM",
        -12 => "_ERROR_FORMAT_DICTIONARY",
        -13 => "_ERROR_FORMAT_WINDOW_BITS",
        -14 => "_ERROR_FORMAT_PADDING_1",
        -15 => "_ERROR_FORMAT_PADDING_2",
        -16 => "_ERROR_FORMAT_DISTANCE",
        -17 => "_ERROR_FORMAT_BLOCK_SWITCH",
        -18 => "_ERROR_COMPOUND_DICTIONARY",
        -19 => "_ERROR_DICTIONARY_NOT_SET",
        -20 => "_ERROR_INVALID_ARGUMENTS",
        -21 => "_ERROR_ALLOC_CONTEXT_MODES",
        -22 => "_ERROR_ALLOC_TREE_GROUPS",
        -25 => "_ERROR_ALLOC_CONTEXT_MAP",
        -26 => "_ERROR_ALLOC_RING_BUFFER_1",
        -27 => "_ERROR_ALLOC_RING_BUFFER_2",
        -30 => "_ERROR_ALLOC_BLOCK_TYPE_TREES",
        -31 => "_ERROR_UNREACHABLE",
        _ => "INVALID",
    }
}

fn brotli_error(msg: String) -> Error {
    Error::Py(PyExc::BrotliError, msg)
}

/// `brotlicffi.Decompressor()`.
///
/// Errors are [`PyExc::BrotliError`] (`brotlicffi.error`) with `str(e)` as the message. For
/// decoding errors brotlicffi passes a `bytes` argument, so `str(e)` is its repr:
/// `b'Decompression error: _ERROR_FORMAT_PADDING_1'`.
pub struct Decompressor {
    state: BrotliState<StandardAlloc, StandardAlloc, StandardAlloc>,
    /// `_unconsumed_data`.
    unconsumed: Vec<u8>,
    /// The decoder's error code once it failed (libbrotli fails every later call with it).
    error: Option<i32>,
}

impl std::fmt::Debug for Decompressor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Decompressor")
            .field("error", &self.error)
            .field("unconsumed", &self.unconsumed.len())
            .finish_non_exhaustive()
    }
}

impl Default for Decompressor {
    fn default() -> Self {
        Self::new()
    }
}

impl Decompressor {
    /// A decoder with libbrotli's defaults (no large-window streams).
    pub fn new() -> Self {
        Self {
            state: BrotliState::new_strict(
                StandardAlloc::default(),
                StandardAlloc::default(),
                StandardAlloc::default(),
            ),
            unconsumed: Vec::new(),
            error: None,
        }
    }

    /// `Decompressor.decompress(data)`: append the output to `out` (nothing on an error).
    pub fn decompress(&mut self, data: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
        if !self.unconsumed.is_empty() && !data.is_empty() {
            return Err(brotli_error(
                "brotli: decoder process called with data when 'can_accept_more_data()' is False"
                    .to_owned(),
            ));
        }
        let taken;
        let input: &[u8] = if self.unconsumed.is_empty() {
            data
        } else {
            taken = std::mem::take(&mut self.unconsumed);
            &taken
        };
        if let Some(code) = self.error {
            return Err(brotli_error(format!(
                "b'Decompression error: {}'",
                error_string(code)
            )));
        }
        let start = out.len();
        let mut avail_in = input.len();
        let mut in_off = 0;
        let mut chunks = 0usize;
        let mut total = 0usize;
        loop {
            // `_calculate_buffer_size` without `output_buffer_limit`.
            let size = if input.is_empty() {
                1usize << (chunks + 15).min(24)
            } else {
                5 * input.len()
            };
            let w = out.len();
            out.resize(w + size, 0);
            let mut avail_out = size;
            let mut out_off = w;
            let rc = BrotliDecompressStream(
                &mut avail_in,
                &mut in_off,
                input,
                &mut avail_out,
                &mut out_off,
                out.as_mut_slice(),
                &mut total,
                &mut self.state,
            );
            match rc {
                BrotliResult::ResultFailure => {
                    out.truncate(start);
                    let code = self.state.error_code as i32;
                    self.error = Some(code);
                    return Err(brotli_error(format!(
                        "b'Decompression error: {}'",
                        error_string(code)
                    )));
                }
                BrotliResult::ResultSuccess if avail_in > 0 => {
                    out.truncate(start);
                    self.unconsumed.clear();
                    return Err(brotli_error(
                        "Decompression error: trailing data after stream.".to_owned(),
                    ));
                }
                _ => {}
            }
            out.truncate(w + (size - avail_out));
            self.unconsumed.clear();
            self.unconsumed
                .extend_from_slice(&input[in_off..in_off + avail_in]);
            match rc {
                BrotliResult::NeedsMoreOutput => chunks += 1,
                _ => return Ok(()),
            }
        }
    }

    /// `Decompressor.flush()`: always empty (brotlicffi decompresses eagerly).
    pub fn flush(&mut self) -> Vec<u8> {
        Vec::new()
    }

    /// `Decompressor.is_finished()`. Never after an error, as in libbrotli.
    pub fn is_finished(&self) -> bool {
        self.error.is_none() && brotli_decompressor::BrotliDecoderIsFinished(&self.state)
    }

    /// `Decompressor.can_accept_more_data()`.
    pub fn can_accept_more_data(&self) -> bool {
        self.unconsumed.is_empty() && !brotli_decompressor::BrotliDecoderHasMoreOutput(&self.state)
    }

    /// `Decompressor.finish()`: an error if the stream is incomplete.
    pub fn finish(&mut self) -> Result<(), Error> {
        if self.is_finished() {
            Ok(())
        } else {
            Err(brotli_error(
                "Decompression error: incomplete compressed stream.".to_owned(),
            ))
        }
    }
}

/// `brotlicffi.decompress(data)`: `Decompressor().decompress(data)` then `finish()`.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, Error> {
    let mut d = Decompressor::new();
    let mut out = Vec::new();
    d.decompress(data, &mut out)?;
    d.finish()?;
    Ok(out)
}

/// warcio's reader calls it as any other decompressor object; `unused_data` stays empty (warcio
/// sets brotlicffi's to `None`) and `flush()` returns nothing.
impl super::Decompressor for Decompressor {
    fn decompress(&mut self, data: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
        Decompressor::decompress(self, data, out)
    }
}

/// warcio's `brotli_decompressor` (for [`super::ReaderOptions`]'s `brotli`): a new
/// `brotlicffi.Decompressor()`.
pub fn brotli_decompressor() -> Box<dyn super::Decompressor> {
    Box::new(Decompressor::new())
}
