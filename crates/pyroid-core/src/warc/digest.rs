// Ports warcio 1.8.2's digestverifyingreader.py and utils.Digester (Apache-2.0), and the
// parts of CPython's base64/binascii they use (PSF-2.0).
//! `check_digests`: `DigestChecker`, the digest half of `DigestVerifyingReader`, and
//! `_compare_digest_rfc_3548` (base32, base16 and base64 values, including the `-_`
//! alphabet), with CPython 3.13's `binascii` errors.
//!
//! Algorithms: md5, sha1, sha224, sha256, sha384 and sha512, under the names CPython's
//! `hashlib.new` accepts for them with OpenSSL 3 (case-insensitive: `sha1`, `SHA-1`,
//! `sha2-256`, `ssl3-md5`, the OIDs, ...). Any other name is an "unknown hash algorithm"
//! here, where hashlib may know it (sha3_*, blake2*, sm3, ripemd160, `sha512_224`, ...).

use md5::Md5;
use sha1::Sha1;
use sha2::{Digest as _, Sha224, Sha256, Sha384, Sha512};

use super::pystr::py_len;
use super::{Error, Message, PyExc};

/// A `hashlib` object of one of the supported algorithms.
#[derive(Clone, Debug)]
pub(crate) enum Hasher {
    Md5(Md5),
    Sha1(Sha1),
    Sha224(Sha224),
    Sha256(Sha256),
    Sha384(Sha384),
    Sha512(Sha512),
}

/// Names `hashlib.new` takes for each algorithm (lower-cased; OpenSSL 3.5, CPython 3.13).
const NAMES: [(&str, &[&str]); 6] = [
    ("md5", &["md5", "ssl3-md5", "1.2.840.113549.2.5"]),
    ("sha1", &["sha1", "sha-1", "ssl3-sha1", "1.3.14.3.2.26"]),
    (
        "sha224",
        &["sha224", "sha-224", "sha2-224", "2.16.840.1.101.3.4.2.4"],
    ),
    (
        "sha256",
        &["sha256", "sha-256", "sha2-256", "2.16.840.1.101.3.4.2.1"],
    ),
    (
        "sha384",
        &["sha384", "sha-384", "sha2-384", "2.16.840.1.101.3.4.2.2"],
    ),
    (
        "sha512",
        &["sha512", "sha-512", "sha2-512", "2.16.840.1.101.3.4.2.3"],
    ),
];

impl Hasher {
    /// `hashlib.new(name)`, for the supported algorithms.
    pub fn new(name: &str) -> Option<Self> {
        // OpenSSL ignores trailing spaces in a name ('sha1 ' works, ' sha1' does not).
        let name = name.trim_end_matches(' ');
        let (algo, _) = NAMES
            .iter()
            .find(|(_, names)| names.iter().any(|n| n.eq_ignore_ascii_case(name)))?;
        Some(match *algo {
            "md5" => Self::Md5(Md5::new()),
            "sha1" => Self::Sha1(Sha1::new()),
            "sha224" => Self::Sha224(Sha224::new()),
            "sha256" => Self::Sha256(Sha256::new()),
            "sha384" => Self::Sha384(Sha384::new()),
            _ => Self::Sha512(Sha512::new()),
        })
    }

    pub fn update(&mut self, b: &[u8]) {
        match self {
            Self::Md5(h) => h.update(b),
            Self::Sha1(h) => h.update(b),
            Self::Sha224(h) => h.update(b),
            Self::Sha256(h) => h.update(b),
            Self::Sha384(h) => h.update(b),
            Self::Sha512(h) => h.update(b),
        }
    }

    /// `digest()` (does not end the hash).
    pub fn digest(&self) -> Vec<u8> {
        match self.clone() {
            Self::Md5(h) => h.finalize().to_vec(),
            Self::Sha1(h) => h.finalize().to_vec(),
            Self::Sha224(h) => h.finalize().to_vec(),
            Self::Sha256(h) => h.finalize().to_vec(),
            Self::Sha384(h) => h.finalize().to_vec(),
            Self::Sha512(h) => h.finalize().to_vec(),
        }
    }
}

/// `base64.b32encode(b)` as str.
pub(crate) fn b32encode(b: &[u8]) -> String {
    const A: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";
    let mut out = String::with_capacity(b.len().div_ceil(5) * 8);
    for chunk in b.chunks(5) {
        let mut buf = [0u8; 5];
        buf[..chunk.len()].copy_from_slice(chunk);
        let v = buf.iter().fold(0u64, |acc, &x| (acc << 8) | u64::from(x));
        let n = match chunk.len() {
            1 => 2,
            2 => 4,
            3 => 5,
            4 => 7,
            _ => 8,
        };
        for i in 0..8 {
            if i < n {
                out.push(char::from(
                    A[usize::try_from((v >> (35 - 5 * i)) & 31).expect("5 bits")],
                ));
            } else {
                out.push('=');
            }
        }
    }
    out
}

fn binascii(msg: &str) -> Error {
    Error::Py(PyExc::BinasciiError, msg.into())
}

/// `base64._bytes_from_decode_data(s)` for a str.
fn ascii_bytes(s: &str) -> Result<&[u8], Error> {
    if s.is_ascii() {
        Ok(s.as_bytes())
    } else {
        Err(Error::Py(
            PyExc::ValueError,
            "string argument should contain only ASCII characters".into(),
        ))
    }
}

/// `base64.b16decode(s, casefold=True)`.
pub(crate) fn b16decode(s: &str) -> Result<Vec<u8>, Error> {
    let b = ascii_bytes(s)?;
    if !b.iter().all(u8::is_ascii_hexdigit) {
        return Err(binascii("Non-base16 digit found"));
    }
    if b.len() % 2 == 1 {
        return Err(binascii("Odd-length string"));
    }
    Ok(b.chunks(2)
        .map(|p| {
            let hi = char::from(p[0]).to_digit(16).expect("hex");
            let lo = char::from(p[1]).to_digit(16).expect("hex");
            u8::try_from(hi * 16 + lo).expect("byte")
        })
        .collect())
}

/// `binascii.a2b_base64(b, strict_mode=False)`. With `legacy` (CPython 3.12.16 and older
/// releases), a complete padding ends the data; in CPython 3.13.18 and 3.14.10 decoding
/// goes on after it and padding only counts at the end.
fn a2b_base64(b: &[u8], legacy: bool) -> Result<Vec<u8>, Error> {
    let mut out = Vec::with_capacity(b.len() / 4 * 3 + 3);
    let mut quad_pos = 0u8;
    let mut leftchar = 0u8;
    let mut pads = 0u8;
    for &c in b {
        if c == b'=' {
            if quad_pos >= 2 {
                pads = pads.saturating_add(1);
                if legacy && quad_pos + pads >= 4 {
                    return Ok(out);
                }
            }
            continue;
        }
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            _ => continue,
        };
        pads = 0;
        match quad_pos {
            0 => {
                quad_pos = 1;
                leftchar = v;
            }
            1 => {
                quad_pos = 2;
                out.push((leftchar << 2) | (v >> 4));
                leftchar = v & 0x0f;
            }
            2 => {
                quad_pos = 3;
                out.push((leftchar << 4) | (v >> 2));
                leftchar = v & 0x03;
            }
            _ => {
                quad_pos = 0;
                out.push((leftchar << 6) | v);
                leftchar = 0;
            }
        }
    }
    match quad_pos {
        0 => Ok(out),
        _ if !legacy && quad_pos >= 2 && quad_pos.saturating_add(pads) >= 4 => Ok(out),
        1 => Err(binascii(&format!(
            "Invalid base64-encoded string: number of data characters ({}) cannot be 1 more \
             than a multiple of 4",
            out.len() / 3 * 4 + 1
        ))),
        _ => Err(binascii("Incorrect padding")),
    }
}

/// `_b64_wrapper(value)`.
fn b64_wrapper(value: &str, legacy: bool) -> Result<Vec<u8>, Error> {
    let b = ascii_bytes(value)?;
    if value.contains(['-', '_']) {
        let t: Vec<u8> = b
            .iter()
            .map(|&c| match c {
                b'-' => b'+',
                b'_' => b'/',
                _ => c,
            })
            .collect();
        a2b_base64(&t, legacy)
    } else {
        a2b_base64(b, legacy)
    }
}

/// `_parse_digest(digest)`: `(algo, value)`.
pub(crate) fn parse_digest(digest: &str) -> Result<(&str, &str), Error> {
    digest.split_once(':').ok_or_else(|| {
        Error::Py(
            PyExc::ValueError,
            format!("could not parse digest algorithm out of {digest}"),
        )
    })
}

/// `_compare_digest_rfc_3548(digester, digest)`.
pub(crate) fn compare(
    digester: Option<&Hasher>,
    digest: Option<&str>,
    legacy_b64: bool,
) -> Result<Option<bool>, Error> {
    let (Some(digester), Some(digest)) = (digester, digest.filter(|d| !d.is_empty())) else {
        return Ok(None);
    };
    let our_value = b32encode(&digester.digest());
    let (_, warc_value) = parse_digest(digest)?;
    let len = py_len(warc_value);
    let warc_b32 = match len.cmp(&our_value.len()) {
        // casefold needed here? -- rfc recommends not allowing
        std::cmp::Ordering::Equal => return Ok(Some(our_value == warc_value)),
        std::cmp::Ordering::Greater => b32encode(&b16decode(warc_value)?),
        std::cmp::Ordering::Less => b32encode(&b64_wrapper(warc_value, legacy_b64)?),
    };
    Ok(Some(our_value == warc_b32))
}

/// What `check_digests` was: warcio's `DigestChecker(kind)`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DigestKind {
    /// `check_digests=False`: no checks.
    #[default]
    Off,
    /// `True` (or any other true value but 'log' and 'raise'): record problems.
    Record,
    /// `'log'`: also write each problem to stderr.
    Log,
    /// `'raise'`: raise `ArchiveLoadFailed` at the first problem.
    Raise,
}

/// `DigestChecker`: `passed` and `problems` of one record.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DigestChecker {
    /// `kind`.
    pub kind: DigestKind,
    /// `problems`.
    pub problems: Vec<String>,
    /// `passed`.
    pub passed: Option<bool>,
}

impl DigestChecker {
    /// `problem(value)`.
    pub(crate) fn problem(&mut self, value: String, msgs: &mut Vec<Message>) -> Result<(), Error> {
        self.problems.push(value.clone());
        match self.kind {
            DigestKind::Raise => return Err(Error::Py(PyExc::ArchiveLoadFailed, value)),
            DigestKind::Log => msgs.push(Message::Stderr(value + "\n")),
            _ => {}
        }
        self.passed = Some(false);
        Ok(())
    }

    fn pass(&mut self) {
        if self.passed != Some(false) {
            self.passed = Some(true);
        }
    }
}

/// The digest state of a `DigestVerifyingReader`.
#[derive(Clone, Debug)]
pub(crate) struct Verifier {
    payload_digest: Option<String>,
    block_digest: Option<String>,
    /// `payload_digester_obj`; `payload_digester` is it while `payload_on`.
    payload: Option<Hasher>,
    payload_on: bool,
    block: Option<Hasher>,
    legacy_b64: bool,
}

impl Verifier {
    /// `DigestVerifyingReader.__init__`'s digest part.
    pub fn new(
        record_type: Option<&str>,
        mut payload_digest: Option<String>,
        mut block_digest: Option<String>,
        segment_number: Option<&str>,
        legacy_b64: bool,
        checker: &mut DigestChecker,
        msgs: &mut Vec<Message>,
    ) -> Result<Self, Error> {
        if record_type == Some("revisit") {
            block_digest = None;
            payload_digest = None;
        }
        if segment_number.is_some() {
            payload_digest = None;
        }
        let mut v = Self {
            payload_digest,
            block_digest,
            payload: None,
            payload_on: false,
            block: None,
            legacy_b64,
        };
        if let Some(b) = v.block_digest.as_deref().filter(|b| !b.is_empty()) {
            v.block = parse_digest(b).ok().and_then(|(algo, _)| Hasher::new(algo));
            if v.block.is_none() {
                checker.problem("unknown hash algorithm name in block digest".into(), msgs)?;
            }
        }
        if let Some(p) = v.payload_digest.as_deref().filter(|p| !p.is_empty()) {
            v.payload = parse_digest(p).ok().and_then(|(algo, _)| Hasher::new(algo));
            if v.payload.is_none() {
                checker.problem("unknown hash algorithm name in payload digest".into(), msgs)?;
            }
        }
        Ok(v)
    }

    /// `begin_payload()`, with the reader's current `limit`.
    pub fn begin_payload(
        &mut self,
        limit: i128,
        checker: &mut DigestChecker,
        msgs: &mut Vec<Message>,
    ) -> Result<(), Error> {
        self.payload_on = self.payload.is_some();
        if limit == 0 {
            let check = compare(
                self.payload_hasher(),
                self.payload_digest.as_deref(),
                self.legacy_b64,
            )?;
            if check == Some(false) {
                let d = self.payload_digest.clone().unwrap_or_default();
                checker.problem(format!("payload digest failed: {d}"), msgs)?;
                self.payload_on = false; // prevent double-fire
            } else if check == Some(true) {
                checker.pass();
            }
        }
        Ok(())
    }

    fn payload_hasher(&self) -> Option<&Hasher> {
        self.payload.as_ref().filter(|_| self.payload_on)
    }

    /// The digest part of `_update(buff)`, after the limit went down to `limit`.
    pub fn update(
        &mut self,
        buff: &[u8],
        limit: i128,
        checker: &mut DigestChecker,
        msgs: &mut Vec<Message>,
    ) -> Result<(), Error> {
        if self.payload_on
            && let Some(h) = self.payload.as_mut()
        {
            h.update(buff);
        }
        if let Some(h) = self.block.as_mut() {
            h.update(buff);
        }
        if limit == 0 {
            let check = compare(
                self.block.as_ref(),
                self.block_digest.as_deref(),
                self.legacy_b64,
            )?;
            if check == Some(false) {
                let d = self.block_digest.clone().unwrap_or_default();
                checker.problem(format!("block digest failed: {d}"), msgs)?;
            } else if check == Some(true) {
                checker.pass();
            }
            let check = compare(
                self.payload_hasher(),
                self.payload_digest.as_deref(),
                self.legacy_b64,
            )?;
            if check == Some(false) {
                let d = self.payload_digest.clone().unwrap_or_default();
                checker.problem(format!("payload digest failed {d}"), msgs)?;
            } else if check == Some(true) {
                checker.pass();
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b64(s: &str) -> Result<Vec<u8>, String> {
        b64_wrapper(s, false).map_err(|e| e.py().unwrap().1.to_string())
    }

    #[test]
    fn base64_like_cpython() {
        // Expected values from CPython 3.13's base64.b64decode.
        assert_eq!(b64("AAAA=").unwrap(), [0, 0, 0]);
        assert_eq!(b64("AAA").unwrap_err(), "Incorrect padding");
        assert_eq!(
            b64("A").unwrap_err(),
            "Invalid base64-encoded string: number of data characters (1) cannot be 1 more \
             than a multiple of 4"
        );
        assert_eq!(
            b64("AAAAA").unwrap_err(),
            "Invalid base64-encoded string: number of data characters (5) cannot be 1 more \
             than a multiple of 4"
        );
        assert_eq!(b64("AA==AA").unwrap(), [0, 0, 0]);
        assert_eq!(a2b_base64(b"AA==AA", true).unwrap(), [0]);
        assert_eq!(b64("QQ==QUJD").unwrap_err(), "Incorrect padding");
        assert_eq!(a2b_base64(b"QQ==QUJD", true).unwrap(), b"A");
        assert_eq!(b64("QUI==").unwrap(), b"AB");
        assert_eq!(b64("QQ=AA").unwrap(), b"A\0\0");
        assert_eq!(b64("A=A=").unwrap_err(), "Incorrect padding");
        assert_eq!(b64("QUJD\n").unwrap(), b"ABC");
        assert_eq!(b64("!!!").unwrap(), b"");
        assert_eq!(b64("QUJD===").unwrap(), b"ABC");
        assert_eq!(b64("Q=UJD").unwrap(), b"ABC");
        assert_eq!(b64("-_8=").unwrap(), [0xfb, 0xff]);
        assert_eq!(
            b64("\u{e9}").unwrap_err(),
            "string argument should contain only ASCII characters"
        );
    }

    #[test]
    fn base16_like_cpython() {
        assert_eq!(b16decode("abcd").unwrap(), [0xab, 0xcd]);
        let err = |s| {
            b16decode(s)
                .unwrap_err()
                .py()
                .map(|(k, m)| (k, m.to_string()))
        };
        assert_eq!(
            err("abc"),
            Some((PyExc::BinasciiError, "Odd-length string".into()))
        );
        assert_eq!(
            err("zz"),
            Some((PyExc::BinasciiError, "Non-base16 digit found".into()))
        );
        assert_eq!(err("\u{e9}").unwrap().0, PyExc::ValueError);
    }

    #[test]
    fn base32() {
        assert_eq!(b32encode(b""), "");
        assert_eq!(b32encode(b"f"), "MY======");
        assert_eq!(b32encode(b"fooba"), "MZXW6YTB");
        assert_eq!(b32encode(b"foobar"), "MZXW6YTBOI======");
    }

    #[test]
    fn hash_names() {
        for n in [
            "sha1",
            "SHA-1",
            "Sha1",
            "ssl3-sha1",
            "sha1 ",
            "1.3.14.3.2.26",
            "sha2-256",
            "MD5",
        ] {
            assert!(Hasher::new(n).is_some(), "{n}");
        }
        for n in [" sha1", "md-5", "rsa-sha1", "sha3_256", "sha1\0"] {
            assert!(Hasher::new(n).is_none(), "{n}");
        }
    }

    #[test]
    fn compare_encodings() {
        let mut h = Hasher::new("sha1").unwrap();
        h.update(b"hello");
        // sha1("hello") = aaf4c61d...; base32 / base16 / base64 of it.
        let b32 = b32encode(&h.digest());
        assert_eq!(
            compare(Some(&h), Some(&format!("sha1:{b32}")), false).unwrap(),
            Some(true)
        );
        let hex = "aaf4c61ddcc5e8a2dabede0f3b482cd9aea9434d";
        assert_eq!(
            compare(Some(&h), Some(&format!("sha1:{hex}")), false).unwrap(),
            Some(true)
        );
        let b64 = "qvTGHdzF6KLavt4PO0gs2a6pQ00=";
        assert_eq!(
            compare(Some(&h), Some(&format!("sha1:{b64}")), false).unwrap(),
            Some(true)
        );
        assert_eq!(
            compare(
                Some(&h),
                Some(&format!("sha1:{}", b32.to_lowercase())),
                false
            )
            .unwrap(),
            Some(false)
        );
        assert_eq!(compare(None, Some("sha1:x"), false).unwrap(), None);
        assert_eq!(compare(Some(&h), Some(""), false).unwrap(), None);
    }
}
