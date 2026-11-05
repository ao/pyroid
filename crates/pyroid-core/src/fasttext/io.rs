// Part of pyroid's port of fastText (https://github.com/facebookresearch/fastText).
// fastText: Copyright (c) 2016-present, Facebook, Inc. MIT License.

//! Bounds-checked little-endian reads for the model loader.
//!
//! Every read is checked against the bytes left in the source *before* it
//! happens, so a corrupt size field yields [`LoadError::Truncated`] instead
//! of a huge allocation (which would abort the process) or a short read.

use std::io::{self, BufRead, BufReader, Read};

use super::error::LoadError;

pub struct Reader<R> {
    inner: BufReader<R>,
    pos: u64,
    len: u64,
}

impl<R: Read> Reader<R> {
    /// `len` is the total size of the source in bytes (the file size).
    pub fn new(inner: R, len: u64) -> Self {
        Self {
            inner: BufReader::with_capacity(1 << 16, inner),
            pos: 0,
            len,
        }
    }

    pub fn remaining(&self) -> u64 {
        self.len.saturating_sub(self.pos)
    }

    fn truncated(&self, what: &'static str, need: u64) -> LoadError {
        LoadError::Truncated {
            what,
            offset: self.pos,
            need,
            have: self.remaining(),
        }
    }

    /// Fail with `Truncated` unless `need` more bytes are available.
    pub fn require(&self, what: &'static str, need: u64) -> Result<(), LoadError> {
        if need > self.remaining() {
            Err(self.truncated(what, need))
        } else {
            Ok(())
        }
    }

    fn fill(&mut self, buf: &mut [u8], what: &'static str) -> Result<(), LoadError> {
        let need = buf.len() as u64;
        self.require(what, need)?;
        self.inner.read_exact(buf).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                // The source is shorter than its reported length (it shrank).
                self.truncated(what, need)
            } else {
                LoadError::Io(e)
            }
        })?;
        self.pos += need;
        Ok(())
    }

    fn array<const N: usize>(&mut self, what: &'static str) -> Result<[u8; N], LoadError> {
        let mut b = [0u8; N];
        self.fill(&mut b, what)?;
        Ok(b)
    }

    pub fn i32(&mut self, what: &'static str) -> Result<i32, LoadError> {
        self.array(what).map(i32::from_le_bytes)
    }

    pub fn i64(&mut self, what: &'static str) -> Result<i64, LoadError> {
        self.array(what).map(i64::from_le_bytes)
    }

    pub fn f64(&mut self, what: &'static str) -> Result<f64, LoadError> {
        self.array(what).map(f64::from_le_bytes)
    }

    pub fn u8(&mut self, what: &'static str) -> Result<u8, LoadError> {
        self.array::<1>(what).map(|[b]| b)
    }

    /// A C++ `bool` written with `ostream::write`: one byte, non-zero is true.
    pub fn bool(&mut self, what: &'static str) -> Result<bool, LoadError> {
        self.u8(what).map(|b| b != 0)
    }

    /// Bytes up to (not including) the next NUL, which is consumed. Appends to `out`.
    pub fn cstring(&mut self, out: &mut Vec<u8>, what: &'static str) -> Result<(), LoadError> {
        let start = out.len();
        let limit = self.remaining();
        let n = (&mut self.inner)
            .take(limit)
            .read_until(0, out)
            .map_err(LoadError::Io)? as u64;
        if out.last() != Some(&0) || out.len() == start {
            // EOF before the terminator. (fastText's loader would loop forever here.)
            out.truncate(start);
            return Err(self.truncated(what, n + 1));
        }
        out.pop();
        self.pos += n;
        Ok(())
    }

    /// `count` raw bytes.
    pub fn bytes(&mut self, count: u64, what: &'static str) -> Result<Vec<u8>, LoadError> {
        self.require(what, count)?;
        let n = usize::try_from(count).map_err(|_| self.truncated(what, count))?;
        let mut out = vec![0u8; n];
        self.fill(&mut out, what)?;
        Ok(out)
    }

    /// `count` little-endian `f32`s, read in chunks straight into the result.
    pub fn f32s(&mut self, count: u64, what: &'static str) -> Result<Vec<f32>, LoadError> {
        let bytes = count
            .checked_mul(4)
            .ok_or_else(|| self.truncated(what, u64::MAX))?;
        self.require(what, bytes)?;
        let n = usize::try_from(count).map_err(|_| self.truncated(what, bytes))?;
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| LoadError::Invalid(format!("{what}: cannot allocate {bytes} bytes")))?;
        let mut buf = vec![0u8; (1 << 16).min(n * 4)];
        let mut left = n;
        while left > 0 {
            let take = left.min(buf.len() / 4);
            let chunk = &mut buf[..take * 4];
            self.fill(chunk, what)?;
            out.extend(
                chunk
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
            );
            left -= take;
        }
        Ok(out)
    }
}
