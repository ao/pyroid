// Ports warcio 1.8.2's statusandheaders.py, the ARC parsers of recordloader.py and
// timeutils.timestamp_to_iso_date (Apache-2.0).
//! `StatusAndHeaders`, `StatusAndHeadersParser.parse`, `ARCHeadersParser` and
//! `ARC2WARCHeadersParser`, with Python's string semantics: each line decoded as UTF-8
//! else latin-1, `rstrip()`/`lstrip()`/`strip()` on Unicode whitespace, `split(':', 1)`,
//! continuation lines, `split_prefix` with `str.upper()`, and `total_len` counted in code
//! points.

use std::borrow::Cow;
use std::fmt::Write as _;

use super::pystr::{self, decode_header, lstrip, py_len, rstrip, strip};
use super::{Error, PyExc};
use crate::ftfy::UnicodeVersion;

/// warcio's `StatusAndHeaders`: a status line and `(name, value)` headers in order.
///
/// The headers live in one string; `fields` holds the byte ranges of each name and value.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StatusAndHeaders {
    /// `protocol` (e.g. `WARC/1.0`, `HTTP/1.1`, `GET`).
    pub protocol: String,
    /// `statusline` (e.g. `200 OK`; empty for WARC headers).
    pub statusline: String,
    /// `total_len`: the code points of every line read, line ends included.
    pub total_len: u64,
    text: String,
    fields: Vec<[u32; 4]>,
}

impl StatusAndHeaders {
    /// Empty headers.
    pub fn new(protocol: &str, statusline: &str, total_len: u64) -> Self {
        Self {
            protocol: protocol.into(),
            statusline: statusline.into(),
            total_len,
            ..Self::default()
        }
    }

    pub(crate) fn clear(&mut self) {
        self.protocol.clear();
        self.statusline.clear();
        self.total_len = 0;
        self.text.clear();
        self.fields.clear();
    }

    /// The number of headers.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// No headers.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    fn at(&self, f: [u32; 4]) -> (&str, &str) {
        let r = |a: u32, b: u32| &self.text[a as usize..b as usize];
        (r(f[0], f[1]), r(f[2], f[3]))
    }

    /// Header `i`: `headers[i]`.
    pub fn get(&self, i: usize) -> Option<(&str, &str)> {
        self.fields.get(i).map(|&f| self.at(f))
    }

    /// The headers in order.
    pub fn headers(&self) -> impl ExactSizeIterator<Item = (&str, &str)> + '_ {
        self.fields.iter().map(|&f| self.at(f))
    }

    /// `headers` as owned pairs.
    pub fn to_vec(&self) -> Vec<(String, String)> {
        self.headers()
            .map(|(n, v)| (n.to_owned(), v.to_owned()))
            .collect()
    }

    /// `add_header(name, value)`.
    pub fn add_header(&mut self, name: &str, value: &str) {
        let n0 = self.text.len();
        self.text.push_str(name);
        let n1 = self.text.len();
        self.text.push_str(value);
        let v1 = self.text.len();
        self.fields.push([pos(n0), pos(n1), pos(n1), pos(v1)]);
    }

    /// Start a header whose value may still grow ([`StatusAndHeaders::extend_last`]).
    fn push_name(&mut self, name: &str) {
        let n0 = self.text.len();
        self.text.push_str(name);
        let n1 = pos(self.text.len());
        self.fields.push([pos(n0), n1, n1, n1]);
    }

    fn extend_last(&mut self, s: &str) {
        self.text.push_str(s);
        let end = pos(self.text.len());
        if let Some(f) = self.fields.last_mut() {
            f[3] = end;
        }
    }

    /// `get_header(name)`: the first header whose `name.lower()` matches.
    pub fn get_header(&self, name: &str) -> Option<&str> {
        let target: Cow<'_, str> = if name.is_ascii() {
            Cow::Owned(name.to_ascii_lowercase())
        } else {
            Cow::Owned(name.to_lowercase())
        };
        self.headers()
            .find(|(n, _)| name_matches(n, &target))
            .map(|(_, v)| v)
    }

    /// `get_header(name)` for an ASCII lower-case `name` (no allocation).
    pub(crate) fn get_lower(&self, lower: &str) -> Option<&str> {
        debug_assert!(lower.is_ascii() && !lower.bytes().any(|b| b.is_ascii_uppercase()));
        self.headers()
            .find(|(n, _)| pystr::lower_eq(n, lower))
            .map(|(_, v)| v)
    }

    /// `replace_header(name, value)`: the last matching header gets `value`, else it is added.
    pub fn replace_header(&mut self, name: &str, value: &str) {
        let target = name.to_lowercase();
        let i = (0..self.fields.len())
            .rev()
            .find(|&i| name_matches(self.at(self.fields[i]).0, &target));
        match i {
            Some(i) => {
                let v0 = pos(self.text.len());
                self.text.push_str(value);
                let f = &mut self.fields[i];
                f[2] = v0;
                f[3] = pos(self.text.len());
            }
            None => self.add_header(name, value),
        }
    }

    /// `dict(self.headers)[key]`: the last header named exactly `key`.
    pub fn dict_get(&self, key: &str) -> Option<&str> {
        self.headers()
            .filter(|(n, _)| *n == key)
            .last()
            .map(|(_, v)| v)
    }

    /// `bool(self)`: a status line or any header.
    pub fn is_truthy(&self) -> bool {
        !self.statusline.is_empty() || !self.fields.is_empty()
    }
}

/// `n.lower() == target` (target already lower-cased).
fn name_matches(n: &str, target: &str) -> bool {
    if target.is_ascii() {
        pystr::lower_eq(n, target)
    } else {
        n.to_lowercase() == target
    }
}

fn pos(n: usize) -> u32 {
    u32::try_from(n).expect("headers under 4 GiB")
}

/// `stream.readline()` for the parsers.
pub(crate) trait LineSource {
    fn readline(&mut self, out: &mut Vec<u8>) -> Result<(), Error>;
}

/// Why `parse` did not return headers.
#[derive(Debug)]
pub(crate) enum ParseError {
    /// `EOFError()` (an empty first line).
    Eof,
    /// `StatusAndHeadersParserException(msg, statusline)`.
    Status { msg: String, statusline: String },
    /// The ARC parser's exception, whose `statusline` is the list of parts.
    Parts(Vec<String>),
    /// Anything the stream raised.
    Io(Error),
}

impl From<Error> for ParseError {
    fn from(e: Error) -> Self {
        Self::Io(e)
    }
}

/// The first line given to `parse`: the bytes warcio read ahead, or the str of a previous
/// parser's exception.
pub(crate) enum FirstLine<'a> {
    Read,
    Bytes(&'a [u8]),
    Str(&'a str),
}

/// `str(list)` of ASCII constants.
fn const_list(items: &[&str]) -> String {
    let inner: Vec<String> = items.iter().map(|s| format!("'{s}'")).collect();
    format!("[{}]", inner.join(", "))
}

/// Reused buffers for parsing lines.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    raw: Vec<u8>,
    line: String,
    next: String,
}

/// Read one line and decode it into `out` (`decode_header(stream.readline())`); its length
/// in code points.
fn read_decoded(
    src: &mut dyn LineSource,
    raw: &mut Vec<u8>,
    out: &mut String,
) -> Result<u64, Error> {
    raw.clear();
    src.readline(raw)?;
    out.clear();
    out.push_str(&decode_header(raw));
    Ok(py_len(out) as u64)
}

/// `StatusAndHeadersParser(statuslist, verify).parse(stream, full_statusline)` into `out`.
pub(crate) fn parse(
    out: &mut StatusAndHeaders,
    src: &mut dyn LineSource,
    first: FirstLine<'_>,
    statuslist: &[&'static str],
    verify: bool,
    scratch: &mut Scratch,
) -> Result<(), ParseError> {
    out.clear();
    let Scratch { raw, line, next } = scratch;
    // status line w newlines intact
    let full: Cow<'_, str> = match first {
        FirstLine::Read => {
            raw.clear();
            src.readline(raw)?;
            decode_header(raw)
        }
        FirstLine::Bytes(b) => decode_header(b),
        FirstLine::Str(s) => Cow::Borrowed(s),
    };
    let statusline = rstrip(&full);
    let mut total_read = py_len(&full) as u64;
    // at end of stream
    if total_read == 0 {
        return Err(ParseError::Eof);
    }
    if statusline.is_empty() {
        out.total_len = total_read;
        return Ok(());
    }
    // validate only if verify is set
    let (protocol, rest): (&str, Option<&str>) = if verify {
        match pystr::split_prefix(statusline, statuslist) {
            Some((p, r)) => (p, Some(r)),
            None => {
                return Err(ParseError::Status {
                    msg: format!(
                        "Expected Status Line starting with {} - Found: {}",
                        const_list(statuslist),
                        statusline
                    ),
                    statusline: full.into_owned(),
                });
            }
        }
    } else {
        match statusline.split_once(' ') {
            Some((p, r)) => (p, Some(r)),
            None => (statusline, None),
        }
    };
    out.protocol.push_str(protocol);
    if let Some(r) = rest {
        out.statusline.push_str(strip(r));
    }
    drop(full);

    total_read += read_decoded(src, raw, line)?;
    let len = rstrip(line).len();
    line.truncate(len);
    while !line.is_empty() {
        let has_value = match line.split_once(':') {
            Some((name, value)) => {
                out.push_name(name.trim_end_matches([' ', '\t']));
                out.extend_last(lstrip(value));
                true
            }
            None => false,
        };
        total_read += read_decoded(src, raw, next)?;
        let len = rstrip(next).len();
        next.truncate(len);
        // append continuation lines, if any
        while next.starts_with([' ', '\t']) {
            if has_value {
                out.extend_last(next);
            }
            total_read += read_decoded(src, raw, next)?;
            let len = rstrip(next).len();
            next.truncate(len);
        }
        std::mem::swap(line, next);
    }
    out.total_len = total_read;
    Ok(())
}

/// The ARC 1.0 header names, and the WARC names arc2warc gives them.
const ARC_HEADERS: [&str; 5] = [
    "uri",
    "ip-address",
    "archive-date",
    "content-type",
    "length",
];
const ARC_TO_WARC_HEADERS: [&str; 5] = [
    "WARC-Target-URI",
    "WARC-IP-Address",
    "WARC-Date",
    "Content-Type",
    "Content-Length",
];

/// `ARCHeadersParser.parse` / `ARC2WARCHeadersParser.parse` into `out`.
pub(crate) fn parse_arc(
    out: &mut StatusAndHeaders,
    src: &mut dyn LineSource,
    first: FirstLine<'_>,
    arc2warc: bool,
    make_id: &dyn Fn() -> String,
    unicode: UnicodeVersion,
    scratch: &mut Scratch,
) -> Result<(), ParseError> {
    out.clear();
    let Scratch { raw, line, next } = scratch;
    let mut total_read = 0u64;
    let headerline: Cow<'_, str> = match first {
        FirstLine::Read => {
            raw.clear();
            src.readline(raw)?;
            Cow::Owned(decode_header(raw).into_owned())
        }
        FirstLine::Bytes(b) => decode_header(b),
        FirstLine::Str(s) => Cow::Borrowed(s),
    };
    if headerline.is_empty() {
        return Err(ParseError::Eof);
    }
    let headerline = rstrip(&headerline);
    // if arc header, consume next two lines
    if headerline.starts_with("filedesc://") {
        total_read += read_decoded(src, raw, line)?; // skip version
        total_read += read_decoded(src, raw, next)?; // skip header spec, use preset one
    }
    let mut parts: Vec<&str> = headerline.rsplitn(5, ' ').collect();
    parts.reverse();
    if parts.len() != 5 {
        return Err(ParseError::Parts(
            parts.iter().map(|&s| s.to_owned()).collect(),
        ));
    }
    if arc2warc {
        let warcinfo = headerline.starts_with("filedesc://");
        let rec_type = if warcinfo { "warcinfo" } else { "response" };
        if !warcinfo {
            parts[3] = "application/http;msgtype=response";
        }
        out.add_header("WARC-Type", rec_type);
        out.add_header("WARC-Record-ID", &make_id());
        for (name, value) in ARC_TO_WARC_HEADERS.iter().zip(&parts) {
            if *name == "WARC-Date" {
                out.add_header(name, &timestamp_to_iso_date(value, unicode));
            } else if warcinfo && *name == "WARC-Target-URI" {
                out.add_header("WARC-Filename", &value["filedesc://".len()..]);
            } else {
                out.add_header(name, value);
            }
        }
    } else {
        for (name, value) in ARC_HEADERS.iter().zip(&parts) {
            out.add_header(name, value);
        }
    }
    out.protocol.push_str("WARC/1.0");
    out.total_len = total_read;
    Ok(())
}

/// `timeutils.timestamp_to_iso_date(string)`: pad to 6 digits with `299912`, clamp each
/// field (`int()` of the slice, else its maximum), format `%Y-%m-%dT%H:%M:%SZ`.
pub(crate) fn timestamp_to_iso_date(s: &str, unicode: UnicodeVersion) -> String {
    const PAD_6_UP: &str = "299912";
    let mut chars: Vec<char> = s.chars().collect();
    if chars.len() < 6 {
        chars.extend(PAD_6_UP.chars().skip(chars.len()));
    }
    let extract = |start: usize, end: usize, min: i128, max: i128| -> i128 {
        if chars.len() < end {
            return max;
        }
        let slice: String = chars[start..end].iter().collect();
        match pystr::py_int(&slice, 0, unicode) {
            Some(pystr::PyInt::Small(v)) => v.clamp(min, max),
            _ => max,
        }
    };
    let year = extract(0, 4, 1900, 2999);
    let month = extract(4, 6, 1, 12);
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let mdays = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    let day = extract(6, 8, 1, mdays);
    let hour = extract(8, 10, 0, 23);
    let minute = extract(10, 12, 0, 59);
    let second = extract(12, 14, 0, 59);
    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// `StatusAndHeadersParser.make_warc_id()`: `<urn:uuid:...>` with a random (version 4) UUID.
pub fn make_warc_id() -> String {
    use std::hash::{BuildHasher as _, Hasher as _};
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    // std's RandomState keys come from the OS; hashing a counter with fresh keys gives
    // unpredictable bits without another dependency.
    let word = || {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
        h.finish()
    };
    let mut b = [0u8; 16];
    b[..8].copy_from_slice(&word().to_le_bytes());
    b[8..].copy_from_slice(&word().to_le_bytes());
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    let h = b.iter().fold(String::with_capacity(32), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    });
    format!(
        "<urn:uuid:{}-{}-{}-{}-{}>",
        &h[..8],
        &h[8..12],
        &h[12..16],
        &h[16..20],
        &h[20..]
    )
}

/// The parser exception as warcio raises it out of `load_http_headers`.
pub(crate) fn status_error(msg: String) -> Error {
    Error::Py(PyExc::StatusAndHeadersParserException, msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lines(Vec<&'static [u8]>);

    impl LineSource for Lines {
        fn readline(&mut self, out: &mut Vec<u8>) -> Result<(), Error> {
            if !self.0.is_empty() {
                out.extend_from_slice(self.0.remove(0));
            }
            Ok(())
        }
    }

    fn parse_lines(
        lines: &[&'static [u8]],
        statuslist: &[&'static str],
        verify: bool,
    ) -> Result<StatusAndHeaders, ParseError> {
        let mut out = StatusAndHeaders::default();
        parse(
            &mut out,
            &mut Lines(lines.to_vec()),
            FirstLine::Read,
            statuslist,
            verify,
            &mut Scratch::default(),
        )?;
        Ok(out)
    }

    const HTTP: &[&str] = &["HTTP/1.0", "HTTP/1.1"];

    #[test]
    fn status_and_headers() {
        // From warcio's test_statusandheaders.py (status_headers_1).
        let h = parse_lines(
            &[
                b"HTTP/1.0 200 OK\r\n",
                b"Content-Type: ABC\r\n",
                b"Some: Value\r\n",
                b"Multi-Line: Value1\r\n",
                b"    Also This\r\n",
                b"\r\n",
                b"Body",
            ],
            HTTP,
            true,
        )
        .unwrap();
        assert_eq!(h.protocol, "HTTP/1.0");
        assert_eq!(h.statusline, "200 OK");
        assert_eq!(
            h.to_vec(),
            [
                ("Content-Type".into(), "ABC".into()),
                ("Some".into(), "Value".into()),
                ("Multi-Line".into(), "Value1    Also This".into()),
            ]
        );
        assert_eq!(h.total_len, 86);
        assert_eq!(h.get_header("content-type"), Some("ABC"));
        assert_eq!(h.get_lower("multi-line"), Some("Value1    Also This"));
        assert!(h.is_truthy());
    }

    #[test]
    fn no_colon_and_continuations() {
        let h = parse_lines(
            &[
                b"HTTP/1.0 204 Empty\r\n",
                b"Content-Type: Value\r\n",
                b"Content-Length: 0\r\n",
                b"Bad\r\n",
                b" Cont\r\n",
                b"\r\n",
            ],
            HTTP,
            true,
        )
        .unwrap();
        assert_eq!(h.len(), 2);
        assert_eq!(h.statusline, "204 Empty");
    }

    #[test]
    fn verify_and_errors() {
        match parse_lines(&[b"Foo 200\r\n"], HTTP, true) {
            Err(ParseError::Status { msg, statusline }) => {
                assert_eq!(
                    msg,
                    "Expected Status Line starting with ['HTTP/1.0', 'HTTP/1.1'] - Found: Foo 200"
                );
                assert_eq!(statusline, "Foo 200\r\n");
            }
            r => panic!("{r:?}"),
        }
        assert!(matches!(parse_lines(&[], HTTP, true), Err(ParseError::Eof)));
        let h = parse_lines(&[b"\r\n"], HTTP, true).unwrap();
        assert!(!h.is_truthy());
        assert_eq!(h.total_len, 2);
        // Without verify the status line is split at the first space.
        let h = parse_lines(&[b"Foo  200 OK \r\n", b"\r\n"], HTTP, false).unwrap();
        assert_eq!(
            (h.protocol.as_str(), h.statusline.as_str()),
            ("Foo", "200 OK")
        );
        let h = parse_lines(&[b"http/1.1\r\n", b"\r\n"], HTTP, true).unwrap();
        assert_eq!(
            (h.protocol.as_str(), h.statusline.as_str()),
            ("HTTP/1.1", "")
        );
    }

    #[test]
    fn latin1_and_lengths() {
        let h = parse_lines(
            &[b"HTTP/1.0 200 OK\r\n", b"X: caf\xe9\r\n", b"\r\n"],
            HTTP,
            true,
        )
        .unwrap();
        assert_eq!(h.get_lower("x"), Some("caf\u{e9}"));
        let h = parse_lines(
            &[b"HTTP/1.0 200 OK\r\n", b"X: caf\xc3\xa9\r\n", b"\r\n"],
            HTTP,
            true,
        )
        .unwrap();
        assert_eq!(h.total_len, 17 + 9 + 2);
    }

    #[test]
    fn replace_and_dict() {
        let mut h = StatusAndHeaders::new("WARC/1.0", "", 0);
        h.add_header("uri", "a");
        h.add_header("URI", "b");
        h.add_header("uri", "c");
        assert_eq!(h.get_header("Uri"), Some("a"));
        assert_eq!(h.dict_get("uri"), Some("c"));
        h.replace_header("uri", "z");
        assert_eq!(h.get(2), Some(("uri", "z")));
        h.replace_header("new", "v");
        assert_eq!(h.get(3), Some(("new", "v")));
    }

    #[test]
    fn arc_headers() {
        let mut out = StatusAndHeaders::default();
        let id = || "<urn:uuid:x>".to_string();
        parse_arc(
            &mut out,
            &mut Lines(vec![
                b"1 0 InternetArchive\r\n",
                b"URL IP-address Archive-date Content-type Archive-length\r\n",
            ]),
            FirstLine::Bytes(
                b"filedesc://live-web-example.arc.gz 127.0.0.1 20140216050221 text/plain 75\n",
            ),
            true,
            &id,
            UnicodeVersion::V15_1,
            &mut Scratch::default(),
        )
        .unwrap();
        assert_eq!(
            out.to_vec(),
            [
                ("WARC-Type".into(), "warcinfo".into()),
                ("WARC-Record-ID".into(), "<urn:uuid:x>".into()),
                ("WARC-Filename".into(), "live-web-example.arc.gz".into()),
                ("WARC-IP-Address".into(), "127.0.0.1".into()),
                ("WARC-Date".into(), "2014-02-16T05:02:21Z".into()),
                ("Content-Type".into(), "text/plain".into()),
                ("Content-Length".into(), "75".into()),
            ]
        );
        assert_eq!(out.total_len, 21 + 57);
        match parse_arc(
            &mut out,
            &mut Lines(vec![]),
            FirstLine::Str("a b c\r\n"),
            false,
            &id,
            UnicodeVersion::V15_1,
            &mut Scratch::default(),
        ) {
            Err(ParseError::Parts(p)) => assert_eq!(p, ["a", "b", "c"]),
            r => panic!("{r:?}"),
        }
    }

    #[test]
    fn timestamps() {
        // The doctests of warcio's timeutils.timestamp_to_datetime.
        let t = |s| timestamp_to_iso_date(s, UnicodeVersion::V15_1);
        assert_eq!(t("2014122609501011"), "2014-12-26T09:50:10Z");
        assert_eq!(t("20141226095010"), "2014-12-26T09:50:10Z");
        assert_eq!(t("2014122609501"), "2014-12-26T09:50:59Z");
        assert_eq!(t("201412260950"), "2014-12-26T09:50:59Z");
        assert_eq!(t("20141226095"), "2014-12-26T09:59:59Z");
        assert_eq!(t("201412260"), "2014-12-26T23:59:59Z");
        assert_eq!(t("2014122"), "2014-12-31T23:59:59Z");
        assert_eq!(t("201410"), "2014-10-31T23:59:59Z");
        assert_eq!(t("20141"), "2014-12-31T23:59:59Z");
        assert_eq!(t("201"), "2019-12-31T23:59:59Z");
        assert_eq!(t("20"), "2099-12-31T23:59:59Z");
        assert_eq!(t("2"), "2999-12-31T23:59:59Z");
        assert_eq!(t("3"), "2999-12-31T23:59:59Z");
        assert_eq!(t(""), "2999-12-31T23:59:59Z");
        assert_eq!(t("20131709005601"), "2013-12-09T00:56:01Z");
        assert_eq!(t("40001965252477"), "2999-12-31T23:24:59Z");
        assert_eq!(t("2010abc"), "2010-12-31T23:59:59Z");
        assert_eq!(t("20000229"), "2000-02-29T23:59:59Z");
        assert_eq!(t("19000229"), "1900-02-28T23:59:59Z");
    }

    #[test]
    fn warc_ids() {
        let a = make_warc_id();
        let b = make_warc_id();
        assert_ne!(a, b);
        assert_eq!(a.len(), "<urn:uuid:>".len() + 36);
        assert_eq!(&a[24..25], "4");
    }
}
