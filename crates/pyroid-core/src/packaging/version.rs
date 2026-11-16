// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! `packaging.version`: PEP 440 versions, parsed, ordered, hashed and printed as packaging 26.3
//! does.
//!
//! The parser follows `VERSION_PATTERN` matched with `re.VERBOSE | re.IGNORECASE` and
//! `fullmatch`, including the surrounding Python `\s*`, the optional `v`, the `[._-]`
//! separators, every pre-, post- and dev-release spelling and the ASCII-only local label. All of
//! the pattern's groups are possessive or need no backtracking, so one greedy pass decides.

use std::cmp::Ordering;
use std::fmt::Write as _;

use super::num::{Num, int_hash, tuple_hash};
use crate::text::is_py_whitespace;

/// A pre-release kind, in sort order.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum Pre {
    /// `a` (also `alpha`).
    A,
    /// `b` (also `beta`).
    B,
    /// `rc` (also `c`, `pre`, `preview`).
    Rc,
}

impl Pre {
    /// The normalised letter.
    #[must_use]
    pub fn letter(self) -> &'static str {
        match self {
            Pre::A => "a",
            Pre::B => "b",
            Pre::Rc => "rc",
        }
    }

    /// `normalize_pre(letter)` when it is one of the three kinds.
    #[must_use]
    pub fn from_letter(letter: &str) -> Option<Pre> {
        match letter.to_ascii_lowercase().as_str() {
            "a" | "alpha" => Some(Pre::A),
            "b" | "beta" => Some(Pre::B),
            "rc" | "c" | "pre" | "preview" => Some(Pre::Rc),
            _ => None,
        }
    }

    fn rank(self) -> i64 {
        match self {
            Pre::A => 0,
            Pre::B => 1,
            Pre::Rc => 2,
        }
    }
}

/// One segment of a local version label: lowercased letters, or a number.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Local {
    /// A segment of digits only (`int(part)`).
    Num(Num),
    /// Any other segment, lowercased.
    Str(Box<str>),
}

impl Ord for Local {
    fn cmp(&self, other: &Self) -> Ordering {
        // _cmpkey: ints are (n, ""), strings (-1, s): strings sort first.
        match (self, other) {
            (Local::Num(a), Local::Num(b)) => a.cmp(b),
            (Local::Str(a), Local::Str(b)) => a.cmp(b),
            (Local::Str(_), Local::Num(_)) => Ordering::Less,
            (Local::Num(_), Local::Str(_)) => Ordering::Greater,
        }
    }
}

impl PartialOrd for Local {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A parsed version: packaging's `_epoch`, `_release`, `_pre`, `_post`, `_dev` and `_local`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Version {
    /// The epoch (`0` when absent).
    pub epoch: Num,
    /// The release segment, as written (trailing zeros kept).
    pub release: Vec<Num>,
    /// The pre-release, if any.
    pub pre: Option<(Pre, Num)>,
    /// The post-release number, if any.
    pub post: Option<Num>,
    /// The dev-release number, if any.
    pub dev: Option<Num>,
    /// The local label segments, if any.
    pub local: Option<Vec<Local>>,
}

/// A digit run longer than 19 characters: packaging's `int()` of it may raise `ValueError`
/// (`sys.get_int_max_str_digits()`), so the bindings convert it with Python's `int`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct LongRun {
    /// Byte offset of the run in the parsed string.
    pub start: usize,
    /// Byte offset just past the run.
    pub end: usize,
}

/// The outcome of [`parse`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Parsed {
    /// The version.
    pub version: Version,
    /// Digit runs over 19 characters, in the order packaging converts them.
    pub long_runs: Vec<LongRun>,
}

struct Cursor<'a> {
    s: &'a [u8],
    i: usize,
    long_runs: Vec<LongRun>,
}

impl Cursor<'_> {
    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn digits(&mut self) -> Option<Num> {
        let start = self.i;
        while self.peek().is_some_and(|b| b.is_ascii_digit()) {
            self.i += 1;
        }
        if self.i == start {
            return None;
        }
        if self.i - start > 19 {
            self.long_runs.push(LongRun { start, end: self.i });
        }
        // The slice is ASCII digits.
        let run = std::str::from_utf8(&self.s[start..self.i]).unwrap_or("0");
        Some(Num::from_digits(run))
    }

    fn sep(&mut self) -> bool {
        if matches!(self.peek(), Some(b'.' | b'_' | b'-')) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    /// The first of `words` (in order) that the input continues with, case-insensitively.
    fn word(&mut self, words: &[&'static str]) -> Option<&'static str> {
        let rest = &self.s[self.i..];
        for w in words {
            if rest.len() >= w.len() && rest[..w.len()].eq_ignore_ascii_case(w.as_bytes()) {
                self.i += w.len();
                return Some(w);
            }
        }
        None
    }
}

const PRE_WORDS: &[&str] = &["alpha", "a", "beta", "b", "preview", "pre", "c", "rc"];
const POST_WORDS: &[&str] = &["post", "rev", "r"];

/// Strip Python's `\s` (`str.isspace`) from both ends.
#[must_use]
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_whitespace)
}

/// Parse `text` as `Version(text)` does; `None` means `InvalidVersion`.
#[must_use]
pub fn parse(text: &str) -> Option<Parsed> {
    let core = py_strip(text);
    // Everything inside the surrounding whitespace is ASCII.
    let bytes = core.as_bytes();
    let offset = core.as_ptr() as usize - text.as_ptr() as usize;
    let mut c = Cursor { s: bytes, i: 0, long_runs: Vec::new() };
    if matches!(c.peek(), Some(b'v' | b'V')) {
        c.i += 1;
    }
    // Epoch: digits followed by '!', else no epoch.
    let save = c.i;
    let saved_runs = c.long_runs.len();
    let mut epoch = Num::ZERO;
    if let Some(n) = c.digits() {
        if c.peek() == Some(b'!') {
            c.i += 1;
            epoch = n;
        } else {
            c.i = save;
            c.long_runs.truncate(saved_runs);
        }
    }
    // Release.
    let mut release = vec![c.digits()?];
    while c.peek() == Some(b'.') && c.s.get(c.i + 1).is_some_and(u8::is_ascii_digit) {
        c.i += 1;
        release.push(c.digits()?);
    }
    // Pre-release.
    let mut pre = None;
    {
        let save = c.i;
        c.sep();
        if let Some(w) = c.word(PRE_WORDS) {
            c.sep();
            let n = c.digits().unwrap_or(Num::ZERO);
            pre = Some((Pre::from_letter(w).unwrap_or(Pre::Rc), n));
        } else {
            c.i = save;
        }
    }
    // Post-release.
    let mut post = None;
    {
        let save = c.i;
        let saved_runs = c.long_runs.len();
        if c.peek() == Some(b'-') {
            c.i += 1;
            if let Some(n) = c.digits() {
                post = Some(n);
            } else {
                c.i = save;
                c.long_runs.truncate(saved_runs);
            }
        }
        if post.is_none() {
            c.sep();
            if c.word(POST_WORDS).is_some() {
                c.sep();
                post = Some(c.digits().unwrap_or(Num::ZERO));
            } else {
                c.i = save;
            }
        }
    }
    // Dev release.
    let mut dev = None;
    {
        let save = c.i;
        c.sep();
        if c.word(&["dev"]).is_some() {
            c.sep();
            dev = Some(c.digits().unwrap_or(Num::ZERO));
        } else {
            c.i = save;
        }
    }
    // Local label: '+' [a-z0-9]+ ([._-][a-z0-9]+)*
    let mut local = None;
    if c.peek() == Some(b'+') {
        let save = c.i;
        c.i += 1;
        let mut segs = Vec::new();
        loop {
            let start = c.i;
            while c.peek().is_some_and(|b| b.is_ascii_alphanumeric()) {
                c.i += 1;
            }
            if c.i == start {
                // A separator not followed by a segment ends the label before it.
                if segs.is_empty() {
                    c.i = save;
                } else {
                    c.i = start - 1;
                }
                break;
            }
            segs.push((start, c.i));
            if matches!(c.peek(), Some(b'.' | b'_' | b'-'))
                && c.s.get(c.i + 1).is_some_and(u8::is_ascii_alphanumeric)
            {
                c.i += 1;
            } else {
                break;
            }
        }
        if !segs.is_empty() {
            let mut parts = Vec::with_capacity(segs.len());
            for (a, b) in segs {
                let seg = &c.s[a..b];
                if seg.iter().all(u8::is_ascii_digit) {
                    if b - a > 19 {
                        c.long_runs.push(LongRun { start: a, end: b });
                    }
                    let run = std::str::from_utf8(seg).unwrap_or("0");
                    parts.push(Local::Num(Num::from_digits(run)));
                } else {
                    let low = std::str::from_utf8(seg).unwrap_or("").to_ascii_lowercase();
                    parts.push(Local::Str(low.into()));
                }
            }
            local = Some(parts);
        }
    }
    if c.i != c.s.len() {
        return None;
    }
    let mut long_runs = c.long_runs;
    for r in &mut long_runs {
        r.start += offset;
        r.end += offset;
    }
    Some(Parsed { version: Version { epoch, release, pre, post, dev, local }, long_runs })
}

/// The release without trailing zeros (packaging's `_cmpkey` and `_TrimmedRelease` keep one
/// element: `(0, 0)` trims to `(0,)` there and to `()` in `_cmpkey`).
#[must_use]
pub fn trim_zeros(release: &[Num], keep_one: bool) -> &[Num] {
    let mut n = release.len();
    let floor = usize::from(keep_one);
    while n > floor && release[n - 1].is_zero() {
        n -= 1;
    }
    &release[..n]
}

/// The comparison suffix of `_cmpkey`: (`pre_rank`, `pre_n`, `post_rank`, `post_n`, `dev_rank`,
/// `dev_n`).
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Suffix {
    /// -1 dev-only, 0 a, 1 b, 2 rc, 3 none.
    pub pre_rank: i64,
    /// The pre-release number (0 without one).
    pub pre_n: Num,
    /// 1 with a post-release, else 0.
    pub post_rank: i64,
    /// The post-release number (0 without one).
    pub post_n: Num,
    /// 0 with a dev release, else 1.
    pub dev_rank: i64,
    /// The dev-release number (0 without one).
    pub dev_n: Num,
}

impl Version {
    /// A plain release `release` (epoch 0, nothing else).
    #[must_use]
    pub fn release_only(release: Vec<Num>) -> Version {
        Version { epoch: Num::ZERO, release, pre: None, post: None, dev: None, local: None }
    }

    /// `_cmpkey`'s suffix.
    #[must_use]
    pub fn suffix(&self) -> Suffix {
        let (pre_rank, pre_n) = match (&self.pre, &self.post, &self.dev) {
            (None, None, Some(_)) => (-1, Num::ZERO),
            (None, _, _) => (3, Num::ZERO),
            (Some((k, n)), _, _) => (k.rank(), n.clone()),
        };
        Suffix {
            pre_rank,
            pre_n,
            post_rank: i64::from(self.post.is_some()),
            post_n: self.post.clone().unwrap_or(Num::ZERO),
            dev_rank: i64::from(self.dev.is_none()),
            dev_n: self.dev.clone().unwrap_or(Num::ZERO),
        }
    }

    /// Is this a pre-release (`dev` or `pre` set)?
    #[must_use]
    pub fn is_prerelease(&self) -> bool {
        self.dev.is_some() || self.pre.is_some()
    }

    /// Compare as packaging's `_cmpkey` tuples compare.
    #[must_use]
    pub fn cmp_key(&self, other: &Version) -> Ordering {
        self.epoch
            .cmp(&other.epoch)
            .then_with(|| {
                trim_zeros(&self.release, false).cmp(trim_zeros(&other.release, false))
            })
            .then_with(|| self.cmp_suffix(other))
            .then_with(|| match (&self.local, &other.local) {
                (None, None) => Ordering::Equal,
                (None, Some(_)) => Ordering::Less,
                (Some(_), None) => Ordering::Greater,
                (Some(a), Some(b)) => a.as_slice().cmp(b.as_slice()),
            })
    }

    fn cmp_suffix(&self, other: &Version) -> Ordering {
        // The common case, both plain releases, without building the suffixes.
        if self.pre.is_none()
            && self.post.is_none()
            && self.dev.is_none()
            && other.pre.is_none()
            && other.post.is_none()
            && other.dev.is_none()
        {
            return Ordering::Equal;
        }
        self.suffix().cmp(&other.suffix())
    }

    /// CPython's `hash(self._key)`, or `None` when the local label has letters (their `str`
    /// hashes are per-process; the bindings ask Python).
    #[must_use]
    pub fn py_hash(&self) -> Option<i64> {
        let release: Vec<i64> = trim_zeros(&self.release, false).iter().map(Num::py_hash).collect();
        let s = self.suffix();
        let suffix = [
            int_hash(s.pre_rank),
            s.pre_n.py_hash(),
            int_hash(s.post_rank),
            s.post_n.py_hash(),
            int_hash(s.dev_rank),
            s.dev_n.py_hash(),
        ];
        let mut lanes = vec![self.epoch.py_hash(), tuple_hash(&release), tuple_hash(&suffix)];
        if let Some(local) = &self.local {
            let mut segs = Vec::with_capacity(local.len());
            for seg in local {
                match seg {
                    // (n, ""): hash("") is 0 in CPython.
                    Local::Num(n) => segs.push(tuple_hash(&[n.py_hash(), 0])),
                    Local::Str(_) => return None,
                }
            }
            lanes.push(tuple_hash(&segs));
        }
        Some(tuple_hash(&lanes))
    }

    /// `str(Version)`.
    #[must_use]
    pub fn to_string_py(&self) -> String {
        let mut out = String::with_capacity(16);
        if !self.epoch.is_zero() {
            let _ = write!(out, "{}!", self.epoch);
        }
        self.write_release(&mut out, &self.release);
        if let Some((k, n)) = &self.pre {
            let _ = write!(out, "{}{n}", k.letter());
        }
        if let Some(n) = &self.post {
            let _ = write!(out, ".post{n}");
        }
        if let Some(n) = &self.dev {
            let _ = write!(out, ".dev{n}");
        }
        if let Some(local) = &self.local {
            out.push('+');
            self.write_local(&mut out, local);
        }
        out
    }

    /// `str(_TrimmedRelease(v))`: as [`Version::to_string_py`] with trailing release zeros
    /// stripped (one kept).
    #[must_use]
    pub fn to_string_trimmed(&self) -> String {
        let mut v = self.clone();
        v.release = trim_zeros(&self.release, true).to_vec();
        v.to_string_py()
    }

    fn write_release(&self, out: &mut String, release: &[Num]) {
        let _ = self;
        for (i, n) in release.iter().enumerate() {
            if i > 0 {
                out.push('.');
            }
            let _ = write!(out, "{n}");
        }
    }

    fn write_local(&self, out: &mut String, local: &[Local]) {
        let _ = self;
        for (i, seg) in local.iter().enumerate() {
            if i > 0 {
                out.push('.');
            }
            match seg {
                Local::Num(n) => {
                    let _ = write!(out, "{n}");
                }
                Local::Str(s) => out.push_str(s),
            }
        }
    }

    /// The `local` property: the label joined with `.`.
    #[must_use]
    pub fn local_str(&self) -> Option<String> {
        let local = self.local.as_ref()?;
        let mut out = String::new();
        self.write_local(&mut out, local);
        Some(out)
    }

    /// The `public` property.
    #[must_use]
    pub fn public(&self) -> String {
        let mut v = self.clone();
        v.local = None;
        v.to_string_py()
    }

    /// The `base_version` property.
    #[must_use]
    pub fn base_version(&self) -> String {
        let mut out = String::new();
        if !self.epoch.is_zero() {
            let _ = write!(out, "{}!", self.epoch);
        }
        self.write_release(&mut out, &self.release);
        out
    }
}

/// `normalize_pre(letter)`: lowercase, then the alternate spellings.
#[must_use]
pub fn normalize_letter(letter: &str) -> String {
    let low = letter.to_lowercase();
    match low.as_str() {
        "alpha" => "a".into(),
        "beta" => "b".into(),
        "c" | "pre" | "preview" => "rc".into(),
        "rev" | "r" => "post".into(),
        _ => low,
    }
}

/// `_LOCAL_PATTERN.fullmatch(value)` (`[a-z0-9]+(?:[._-][a-z0-9]+)*`, ASCII, ignoring case),
/// then `_parse_local_version`.
#[must_use]
pub fn parse_local(value: &str) -> Option<Vec<Local>> {
    if value.is_empty() || !value.is_ascii() {
        return None;
    }
    let mut out = Vec::new();
    for seg in value.split(['.', '_', '-']) {
        if seg.is_empty() || !seg.bytes().all(|b| b.is_ascii_alphanumeric()) {
            return None;
        }
        if seg.bytes().all(|b| b.is_ascii_digit()) {
            out.push(Local::Num(Num::from_digits(seg)));
        } else {
            out.push(Local::Str(seg.to_ascii_lowercase().into()));
        }
    }
    Some(out)
}
