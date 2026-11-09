// Reproduces CPython's zlib module (PSF-2.0) over zlib-rs.
//! `zlib.decompressobj(wbits)` as warcio uses it: `decompress(data)` returns all the output
//! (no `max_length`) or raises `zlib.error` with CPython's message, losing that call's
//! output; at the end of the stream the rest of the input, and all input of later calls,
//! goes to `unused_data`. After an error the stream stays broken: every later call raises
//! the same error, as zlib's `BAD` mode does (zlib-rs would say "repeated call with bad
//! state" instead, so the error is remembered here).

use zlib_rs::{Inflate, InflateError, InflateFlush, Status};

use super::{Error, PyExc};

/// Window bits of `zlib.decompressobj(16 + zlib.MAX_WBITS)`: a gzip member.
pub const WBITS_GZIP: i32 = 31;
/// `zlib.decompressobj()`: a zlib stream.
pub const WBITS_ZLIB: i32 = 15;
/// `zlib.decompressobj(-zlib.MAX_WBITS)`: raw deflate.
pub const WBITS_RAW: i32 = -15;

/// Output grows by this much when the buffer is full (CPython's blocks start at 32 KiB).
const GROW: usize = 64 * 1024;

/// A `zlib.decompressobj`.
pub struct DecompressObj {
    inner: Inflate,
    eof: bool,
    unused_data: Vec<u8>,
    /// The sticky error: zlib's return code and message.
    error: Option<(i32, Option<&'static str>)>,
}

impl std::fmt::Debug for DecompressObj {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecompressObj")
            .field("eof", &self.eof)
            .field("unused_data", &self.unused_data.len())
            .field("error", &self.error)
            .finish_non_exhaustive()
    }
}

/// CPython's `zlib_error(state, zst, err, "while decompressing data")`.
fn zlib_message(code: i32, msg: Option<&str>) -> String {
    let msg = msg.or(match code {
        -5 => Some("incomplete or truncated stream"),
        -2 => Some("inconsistent stream state"),
        -3 => Some("invalid input data"),
        _ => None,
    });
    match msg {
        Some(m) => {
            // "%.200s"
            let m: String = m.chars().take(200).collect();
            format!("Error {code} while decompressing data: {m}")
        }
        None => format!("Error {code} while decompressing data"),
    }
}

impl DecompressObj {
    /// `zlib.decompressobj(wbits)` for `wbits` 31 (gzip), 15 (zlib) or -15 (raw deflate).
    pub fn new(wbits: i32) -> Self {
        let inner = match wbits {
            WBITS_GZIP => Inflate::new(true, 31),
            WBITS_ZLIB => Inflate::new(true, 15),
            _ => Inflate::new(false, 15),
        };
        Self {
            inner,
            eof: false,
            unused_data: Vec::new(),
            error: None,
        }
    }

    /// `self.eof`.
    pub fn eof(&self) -> bool {
        self.eof
    }

    /// `self.unused_data`.
    pub fn unused_data(&self) -> &[u8] {
        &self.unused_data
    }

    /// Take `unused_data` (warcio hands it to the next member's decompressor).
    pub fn take_unused_data(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.unused_data)
    }

    /// `self.decompress(data)`: the output goes to `buf[*end..]`, growing `buf` with zeros as
    /// needed (bytes past the new `*end` are left initialized for the next call). On error
    /// `*end` is unchanged and the error is `zlib.error`.
    pub fn decompress_into(
        &mut self,
        data: &[u8],
        buf: &mut Vec<u8>,
        end: &mut usize,
    ) -> Result<(), Error> {
        if let Some((code, msg)) = self.error {
            return Err(Error::Py(PyExc::ZlibError, zlib_message(code, msg)));
        }
        if self.eof {
            self.unused_data.extend_from_slice(data);
            return Ok(());
        }
        let mut pos = 0usize;
        let mut out = *end;
        loop {
            if buf.len() - out < 1024 {
                buf.resize(buf.len().max(out) + GROW, 0);
            }
            let in_before = self.inner.total_in();
            let out_before = self.inner.total_out();
            let res = self
                .inner
                .decompress(&data[pos..], &mut buf[out..], InflateFlush::SyncFlush);
            pos += usize::try_from(self.inner.total_in() - in_before).expect("input count");
            out += usize::try_from(self.inner.total_out() - out_before).expect("output count");
            match res {
                Ok(Status::StreamEnd) => {
                    self.eof = true;
                    self.unused_data.extend_from_slice(&data[pos..]);
                    break;
                }
                Ok(Status::Ok | Status::BufError) => {
                    // CPython's `while (avail_out == 0)`: only a full buffer is grown and
                    // tried again (zlib stops early only when the input is used up).
                    if out < buf.len() {
                        break;
                    }
                }
                Err(e) => {
                    let (code, msg) = match e {
                        InflateError::NeedDict { .. } => (2, None),
                        InflateError::StreamError => (-2, self.inner.error_message()),
                        InflateError::DataError => (-3, self.inner.error_message()),
                        InflateError::MemError => (-4, self.inner.error_message()),
                    };
                    self.error = Some((code, msg));
                    return Err(Error::Py(PyExc::ZlibError, zlib_message(code, msg)));
                }
            }
        }
        *end = out;
        Ok(())
    }

    /// `flush()`: no output is ever left (`decompress` returns all of it); with `raises`
    /// (CPython 3.13 and later) a broken stream raises its error again.
    pub fn flush(&mut self, raises: bool) -> Result<(), Error> {
        match self.error {
            Some((code, msg)) if raises => {
                Err(Error::Py(PyExc::ZlibError, zlib_message(code, msg)))
            }
            _ => Ok(()),
        }
    }

    /// `self.decompress(data)`, appending the output to `out`.
    pub fn decompress(&mut self, data: &[u8], out: &mut Vec<u8>) -> Result<(), Error> {
        let start = out.len();
        let mut end = start;
        let r = self.decompress_into(data, out, &mut end);
        out.truncate(if r.is_ok() { end } else { start });
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn gzip(data: &[u8]) -> Vec<u8> {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn member_then_unused_data() {
        let mut g = gzip(b"hello world");
        g.extend_from_slice(b"rest");
        let mut d = DecompressObj::new(WBITS_GZIP);
        let mut out = Vec::new();
        d.decompress(&g[..5], &mut out).unwrap();
        assert_eq!(out, b"");
        d.decompress(&g[5..], &mut out).unwrap();
        assert_eq!(out, b"hello world");
        assert!(d.eof());
        assert_eq!(d.unused_data(), b"rest");
        d.decompress(b"more", &mut out).unwrap();
        assert_eq!(d.unused_data(), b"restmore");
        assert_eq!(out, b"hello world");
    }

    #[test]
    fn errors_like_cpython() {
        // Messages from CPython 3.13 (zlib 1.3.2).
        let mut d = DecompressObj::new(WBITS_GZIP);
        let mut out = Vec::new();
        let e = d.decompress(b"WARC/1.0\r\n", &mut out).unwrap_err();
        assert_eq!(
            e.py(),
            Some((
                PyExc::ZlibError,
                "Error -3 while decompressing data: incorrect header check"
            ))
        );
        let e = d.decompress(b"x", &mut out).unwrap_err();
        assert_eq!(
            e.py().unwrap().1,
            "Error -3 while decompressing data: incorrect header check"
        );
        let mut d = DecompressObj::new(WBITS_ZLIB);
        let e = d
            .decompress(&[0x78, 0xbb, 0, 0, 0, 0], &mut out)
            .unwrap_err();
        assert_eq!(e.py().unwrap().1, "Error 2 while decompressing data");
        assert_eq!(out, b"");
    }

    #[test]
    fn output_lost_on_error() {
        let mut g = gzip(&vec![b'a'; 100_000]);
        let n = g.len();
        g[n - 8] ^= 1; // CRC
        let mut d = DecompressObj::new(WBITS_GZIP);
        let mut out = b"keep".to_vec();
        let e = d.decompress(&g, &mut out).unwrap_err();
        assert_eq!(
            e.py().unwrap().1,
            "Error -3 while decompressing data: incorrect data check"
        );
        assert_eq!(out, b"keep");
    }

    #[test]
    fn large_output() {
        let data: Vec<u8> = (0..1_000_000u32).map(|i| (i % 251) as u8).collect();
        let g = gzip(&data);
        let mut d = DecompressObj::new(WBITS_GZIP);
        let mut out = Vec::new();
        d.decompress(&g, &mut out).unwrap();
        assert_eq!(out, data);
    }
}
