// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc., from
// tldextract 5.4.0 (BSD-3-Clause), © John Kurkowski, and from CPython 3.13's
// `ipaddress` (PSF-2.0); modified.
//! datatrove 0.10.1's `URLFilter`: the same keep/drop decisions and drop
//! reasons, with tldextract 5.4.0's `TLDExtract.extract_str` underneath.
//!
//! datatrove checks, in this order (the first hit is the reason):
//!
//! 1. `"domain"`: `extract(url).registered_domain` (=
//!    `top_domain_under_public_suffix`) is a blocked domain;
//! 2. `"subdomain"`: `extract(url).fqdn` is a blocked domain;
//! 3. `"url"`: the URL itself is a blocked URL;
//! 4. `"hard_blacklisted"`: a word of `re.split(r"[^a-zA-Z0-9]+", url)`
//!    (case-sensitive; `""` when the URL is empty or starts or ends with a
//!    separator) is a banned word;
//! 5. `"soft_blacklisted"`: at least `soft_word_threshold` distinct words are
//!    soft-banned words;
//! 6. `"blacklisted_subword"`: a banned subword is a substring of the URL
//!    with every character but `[a-zA-Z0-9]` removed, lowercased (pyahocorasick
//!    ignores `""`; with no other subword its automaton raises `AttributeError`,
//!    [`UrlError::NoAutomaton`]).
//!
//! ## tldextract
//!
//! [`SuffixList::domain_parts`] is `extract_str` for the two values datatrove
//! reads: `lenient_netloc` (scheme, `/?#`, userinfo, `[IPv6]`, port, Python
//! `strip()`, trailing dots), the replacement of `。．｡` by `.`, the bracketed
//! IPv6 test (`ipaddress.IPv6Address`, ported, scope ids and embedded IPv4
//! included), and `suffix_index` over the Public Suffix List trie, wildcards
//! and `!` exceptions included, with each label as `_decode_punycode` sees
//! it: CPython's `str.lower()` for the interpreter's Unicode version, then,
//! for `xn--` labels, `idna.decode`. The IPv4 test of `extract_str` is not
//! needed: both of its outcomes give an empty suffix, so an empty registered
//! domain and FQDN.
//!
//! idna is not ported. Only whether the decoded label is a key of the trie
//! matters, so the trie comes with every lowered label that decodes to one of
//! its strings, computed by Python with the real `idna` when the list is
//! built: each string that decodes to itself, and the canonical A-label of
//! each string (`"xn--" + s.encode("punycode")`) that `idna.decode` turns into
//! it. idna 3.20 accepts an A-label only if it is the canonical encoding of
//! its U-label (RFC 5891 §5.3), so no other label can decode to a trie
//! string. With an idna without that check, [`SuffixList`] is built with
//! `defer_xn`, and any other ASCII `xn--` label it meets makes the lookup
//! [`Deferred`] (the caller asks tldextract).

use std::borrow::Cow;
use std::fmt;
use std::hash::BuildHasher;

use rustc_hash::{FxBuildHasher, FxHashMap};

use super::Reason;
use super::text::py_len;
use crate::dedup::{TextNormConfig, simplify_text};
use crate::ftfy::UnicodeVersion;
use crate::text::is_py_whitespace;

/// `"domain"`: the registered domain is blocked.
pub const DOMAIN: Reason = Reason::Fixed("domain");
/// `"subdomain"`: the FQDN is blocked.
pub const SUBDOMAIN: Reason = Reason::Fixed("subdomain");
/// `"url"`: the URL is blocked.
pub const URL: Reason = Reason::Fixed("url");
/// `"hard_blacklisted"`: a banned word.
pub const HARD_BLACKLISTED: Reason = Reason::Fixed("hard_blacklisted");
/// `"soft_blacklisted"`: enough soft-banned words.
pub const SOFT_BLACKLISTED: Reason = Reason::Fixed("soft_blacklisted");
/// `"blacklisted_subword"`: a banned subword.
pub const BLACKLISTED_SUBWORD: Reason = Reason::Fixed("blacklisted_subword");

const NONE: u32 = u32::MAX;

/// What [`UrlFilter::check`] cannot answer by itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UrlError {
    /// The domain lookup needs tldextract itself ([`Deferred`]): call
    /// [`UrlFilter::check_with_parts`] with its registered domain and FQDN.
    Deferred,
    /// pyahocorasick's `AttributeError`: the banned subwords are only `""`,
    /// so the automaton has no words, and datatrove's `iter()` raises.
    NoAutomaton,
}

impl fmt::Display for UrlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Deferred => "the domain lookup needs tldextract",
            Self::NoAutomaton => {
                "Not an Aho-Corasick automaton yet: call add_word to add some keys and call \
                 make_automaton to convert the trie to an automaton."
            }
        })
    }
}

impl std::error::Error for UrlError {}

/// A label [`SuffixList`] cannot decode without idna (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Deferred;

/// Why a list could not be built.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// More than 4 GiB of strings, or 2**32 - 2 of them, in one set.
    TooLarge,
    /// An inconsistent suffix trie.
    BadTrie(String),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooLarge => f.write_str("a URL block list holds more than 4 GiB of strings"),
            Self::BadTrie(m) => write!(f, "bad suffix trie: {m}"),
        }
    }
}

impl std::error::Error for BuildError {}

// ---------------------------------------------------------------------------
// A compact set of strings (the blocked domains: millions of them)
// ---------------------------------------------------------------------------

/// An exact set of strings in one byte arena, with an open-addressing table
/// of 64-bit slots (a 32-bit hash tag and the entry number): 8 bytes of table
/// per slot, 4 of offset per entry, plus the bytes themselves.
#[derive(Clone, Default)]
pub struct StrSet {
    bytes: Vec<u8>,
    ends: Vec<u32>,
    slots: Vec<u64>,
}

impl fmt::Debug for StrSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StrSet")
            .field("len", &self.len())
            .field("bytes", &self.bytes.len())
            .finish_non_exhaustive()
    }
}

const TAG: u64 = 0xFFFF_FFFF_0000_0000;

#[inline]
fn hash_bytes(s: &[u8]) -> u64 {
    // murmur3's finalizer over FxHash: every bit of the slot index and of the
    // tag depends on every input byte.
    let mut h = FxBuildHasher.hash_one(s);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

/// The home slot of hash `h` (its low bits; truncation intended).
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn home(h: u64, mask: usize) -> usize {
    (h as usize) & mask
}

/// The entry number in a slot (its low 32 bits, minus one).
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn slot_entry(slot: u64) -> usize {
    (slot as u32 - 1) as usize
}

impl StrSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct strings.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// No strings?
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// Bytes held (arena, offsets and table).
    pub fn heap_bytes(&self) -> usize {
        self.bytes.capacity() + 4 * self.ends.capacity() + 8 * self.slots.capacity()
    }

    fn entry(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] as usize };
        &self.bytes[start..self.ends[i] as usize]
    }

    /// Make room for `entries` more strings of `bytes` bytes in total.
    pub fn reserve(&mut self, entries: usize, bytes: usize) {
        self.bytes.reserve(bytes);
        self.ends.reserve(entries);
        let want = (self.ends.len() + entries).saturating_mul(4) / 3 + 1;
        if want > self.slots.len() {
            self.rehash(want.next_power_of_two().max(16));
        }
    }

    fn rehash(&mut self, cap: usize) {
        let mut slots = vec![0u64; cap];
        let mask = cap - 1;
        for (i, number) in (1..=self.ends.len() as u64).enumerate() {
            let h = hash_bytes(self.entry(i));
            let mut pos = home(h, mask);
            while slots[pos] != 0 {
                pos = (pos + 1) & mask;
            }
            slots[pos] = (h & TAG) | number;
        }
        self.slots = slots;
    }

    /// Probe for `s`: `Ok(())` if present, else `Err(free slot)`.
    fn find(&self, s: &[u8], h: u64) -> Result<(), usize> {
        let mask = self.slots.len() - 1;
        let mut pos = home(h, mask);
        loop {
            let slot = self.slots[pos];
            if slot == 0 {
                return Err(pos);
            }
            if slot & TAG == h & TAG && self.entry(slot_entry(slot)) == s {
                return Ok(());
            }
            pos = (pos + 1) & mask;
        }
    }

    /// Is `s` in the set?
    #[inline]
    pub fn contains(&self, s: &str) -> bool {
        !self.slots.is_empty() && self.find(s.as_bytes(), hash_bytes(s.as_bytes())).is_ok()
    }

    /// Add `s`; `Ok(false)` if it was already there.
    pub fn insert(&mut self, s: &str) -> Result<bool, BuildError> {
        if (self.ends.len() + 1) * 4 > self.slots.len() * 3 {
            self.rehash((self.slots.len() * 2).max(16));
        }
        let h = hash_bytes(s.as_bytes());
        let Err(pos) = self.find(s.as_bytes(), h) else {
            return Ok(false);
        };
        let (Ok(end), Ok(number)) = (
            u32::try_from(self.bytes.len() + s.len()),
            u32::try_from(self.ends.len() + 2),
        ) else {
            return Err(BuildError::TooLarge);
        };
        self.bytes.extend_from_slice(s.as_bytes());
        self.ends.push(end);
        self.slots[pos] = (h & TAG) | u64::from(number - 1);
        Ok(true)
    }

    /// datatrove's `parse_list(f, do_normalize=False)` over a text file: each
    /// line (`f`'s lines, newlines already translated to `"\n"` by Python's
    /// text mode) that does not start with `#`, `strip()`ped. A blank line
    /// adds `""`.
    pub fn extend_stripped_lines(&mut self, text: &str) -> Result<(), BuildError> {
        let lines = memchr::memchr_iter(b'\n', text.as_bytes()).count() + 1;
        self.reserve(lines, text.len());
        for line in text.split_inclusive('\n') {
            if !line.starts_with('#') {
                self.insert(line.trim_matches(is_py_whitespace))?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// tldextract: lenient_netloc
// ---------------------------------------------------------------------------

/// `urllib.parse.scheme_chars`.
fn is_scheme_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.')
}

/// tldextract's `_schemeless_url`.
fn schemeless(url: &str) -> &str {
    let b = url.as_bytes();
    let Some(dss) = memchr::memmem::find(b, b"//") else {
        return url;
    };
    if dss == 0 {
        return &url[2..];
    }
    if dss < 2 || b[dss - 1] != b':' || !b[..dss - 1].iter().all(|&c| is_scheme_char(c)) {
        return url;
    }
    &url[dss + 2..]
}

/// tldextract's `_host_from_authority`.
fn host_from_authority(authority: &str) -> &str {
    let after = match authority.rfind('@') {
        Some(i) => &authority[i + 1..],
        None => authority,
    };
    if after.starts_with('[') {
        if let Some(j) = after.find(']') {
            return &after[..=j];
        }
    }
    let host = after.find(':').map_or(after, |i| &after[..i]);
    host.trim_matches(is_py_whitespace)
        .trim_end_matches(['.', '\u{3002}', '\u{ff0e}', '\u{ff61}'])
}

/// tldextract's `lenient_netloc(url)`: the host, case preserved.
pub fn lenient_netloc(url: &str) -> &str {
    let rest = schemeless(url);
    let cut = memchr::memchr3(b'/', b'?', b'#', rest.as_bytes()).unwrap_or(rest.len());
    host_from_authority(&rest[..cut])
}

/// `_extract_netloc`'s `netloc.replace("\u3002", ".")` (and U+FF0E, U+FF61).
fn ascii_dots(netloc: &str) -> Cow<'_, str> {
    if netloc.is_ascii() || !netloc.contains(['\u{3002}', '\u{ff0e}', '\u{ff61}']) {
        return Cow::Borrowed(netloc);
    }
    Cow::Owned(netloc.replace(['\u{3002}', '\u{ff0e}', '\u{ff61}'], "."))
}

// ---------------------------------------------------------------------------
// ipaddress.IPv6Address (CPython 3.13)
// ---------------------------------------------------------------------------

/// `IPv4Address._parse_octet` succeeds.
fn ipv4_octet(o: &str) -> bool {
    !o.is_empty()
        && o.len() <= 3
        && o.bytes().all(|b| b.is_ascii_digit())
        && (o == "0" || !o.starts_with('0'))
        && o.parse::<u16>().is_ok_and(|v| v <= 255)
}

/// `IPv4Address(s)` succeeds (`s` has no `/`).
fn ipv4(s: &str) -> bool {
    let mut n = 0;
    for o in s.split('.') {
        n += 1;
        if n > 4 || !ipv4_octet(o) {
            return false;
        }
    }
    n == 4
}

/// `_BaseV6._parse_hextet` succeeds.
fn hextet(h: &str) -> bool {
    !h.is_empty() && h.len() <= 4 && h.bytes().all(|b| b.is_ascii_hexdigit())
}

/// `_BaseV6._ip_int_from_string(ip)` succeeds.
fn ipv6_int(ip: &str) -> bool {
    if ip.is_empty() || py_len(ip) > 45 {
        return false;
    }
    // ip_str.split(':', maxsplit=9)
    let mut parts: Vec<&str> = ip.splitn(10, ':').collect();
    if parts.len() < 3 {
        return false;
    }
    if parts.last().is_some_and(|p| p.contains('.')) {
        if !parts.pop().is_some_and(ipv4) {
            return false;
        }
        // Two hextets ('%x' of each half), always valid.
        parts.extend(["0", "0"]);
    }
    let n = parts.len();
    if n > 9 {
        return false;
    }
    let mut skip = None;
    for (i, p) in parts.iter().enumerate().take(n - 1).skip(1) {
        if p.is_empty() {
            if skip.is_some() {
                return false;
            }
            skip = Some(i);
        }
    }
    let (hi, lo) = if let Some(skip) = skip {
        let (mut hi, mut lo) = (skip, n - skip - 1);
        if parts[0].is_empty() {
            hi -= 1;
            if hi != 0 {
                return false;
            }
        }
        if parts[n - 1].is_empty() {
            lo -= 1;
            if lo != 0 {
                return false;
            }
        }
        if hi + lo >= 8 {
            return false;
        }
        (hi, lo)
    } else {
        if n != 8 || parts[0].is_empty() || parts[n - 1].is_empty() {
            return false;
        }
        (n, 0)
    };
    parts[..hi]
        .iter()
        .chain(&parts[n - lo..])
        .all(|p| hextet(p))
}

/// tldextract's `looks_like_ipv6`: `ipaddress.IPv6Address(s)` does not raise
/// (CPython 3.13: no `/`, at most one `%` followed by a non-empty scope id).
pub fn looks_like_ipv6(s: &str) -> bool {
    if s.contains('/') {
        return false;
    }
    let addr = match s.find('%') {
        None => s,
        Some(i) => {
            let scope = &s[i + 1..];
            if scope.is_empty() || scope.contains('%') {
                return false;
            }
            &s[..i]
        }
    };
    ipv6_int(addr)
}

// ---------------------------------------------------------------------------
// str.lower()
// ---------------------------------------------------------------------------

const LOWER: TextNormConfig = TextNormConfig {
    lowercase: true,
    norm_whitespace: false,
    remove_punctuation: false,
    norm_unicode_diacritics: false,
    norm_numbers: false,
    norm_weekdays: false,
    norm_monthnames: false,
};

/// CPython's `s.lower()` in Unicode version `v`.
///
/// `simplify_text` with only `lowercase` is `s.lower().strip()`. Lowercasing
/// maps no character to or from whitespace, and whitespace is neither cased
/// nor case-ignorable, so it ends a Final_Sigma context as the string's ends
/// do: `s.lower()` is `s`'s leading whitespace + that + its trailing one.
pub fn py_lower(s: &str, v: UnicodeVersion) -> Cow<'_, str> {
    if s.is_ascii() {
        return if s.bytes().any(|b| b.is_ascii_uppercase()) {
            Cow::Owned(s.to_ascii_lowercase())
        } else {
            Cow::Borrowed(s)
        };
    }
    let core = simplify_text(s, &LOWER, v);
    let lead = s.len() - s.trim_start_matches(is_py_whitespace).len();
    let trail = s.trim_end_matches(is_py_whitespace).len();
    if lead == 0 && trail == s.len() {
        return core;
    }
    if lead == s.len() {
        return Cow::Borrowed(s);
    }
    Cow::Owned([&s[..lead], &core, &s[trail..]].concat())
}

// ---------------------------------------------------------------------------
// The Public Suffix List trie
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
struct Node {
    end: bool,
    private: bool,
}

/// tldextract's suffix `Trie` (the one `suffix_index` walks), with the
/// lowered labels that decode to each of its strings.
#[derive(Debug)]
pub struct SuffixList {
    /// Lowered label -> id of `_decode_punycode(label)`, for every label that
    /// decodes to one of `strings`.
    labels: FxHashMap<Box<str>, u32>,
    /// id -> the string (edge keys, and edge keys without their leading `!`).
    strings: Vec<Box<str>>,
    /// id of `s` -> id of `"!" + s` (or `NONE`).
    bang: Vec<u32>,
    /// id of `"*"` (or `NONE`).
    star: u32,
    nodes: Vec<Node>,
    edges: FxHashMap<u64, u32>,
    defer_xn: bool,
}

/// The two strings of tldextract's `ExtractResult` that datatrove reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DomainParts<'a> {
    /// `top_domain_under_public_suffix` (`registered_domain`): `domain + "." +
    /// suffix` when both are set, else `""`.
    pub registered: Cow<'a, str>,
    /// `fqdn`: subdomain, domain and suffix joined (empty ones skipped) when
    /// there is a suffix and a domain (or a private suffix), else `""`.
    pub fqdn: Cow<'a, str>,
}

/// A part of the netloc: empty, a byte range, or a joined string.
enum Piece {
    Empty,
    Slice(usize, usize),
    Joined(String),
}

fn piece<'a>(netloc: &Cow<'a, str>, p: Piece) -> Cow<'a, str> {
    match (p, netloc) {
        (Piece::Empty, _) => Cow::Borrowed(""),
        (Piece::Slice(a, z), Cow::Borrowed(s)) => Cow::Borrowed(&s[a..z]),
        (Piece::Slice(a, z), Cow::Owned(s)) => Cow::Owned(s[a..z].to_owned()),
        (Piece::Joined(s), _) => Cow::Owned(s),
    }
}

#[inline]
fn edge_key(node: u32, key: u32) -> u64 {
    (u64::from(node) << 32) | u64::from(key)
}

fn id_of(i: usize) -> Result<u32, BuildError> {
    u32::try_from(i)
        .ok()
        .filter(|&i| i != NONE)
        .ok_or(BuildError::TooLarge)
}

impl SuffixList {
    /// The trie as tldextract built it, in the form Python dumps it:
    ///
    /// - `strings`: every edge key, plus each `!`-key without its `!`;
    /// - `self_decoding`: the ids of the strings that `_decode_punycode`
    ///   leaves as they are (all but the `xn--` strings idna decodes);
    /// - `aliases`: other lowered labels and the id they decode to;
    /// - `nodes`: `(end, is_private)` per node, the root first;
    /// - `edges`: `(parent, key id, child)`;
    /// - `defer_xn`: idna lacks the canonical A-label check (module docs).
    pub fn new(
        strings: Vec<String>,
        self_decoding: &[u32],
        aliases: Vec<(String, u32)>,
        nodes: &[(bool, bool)],
        edges: &[(u32, u32, u32)],
        defer_xn: bool,
    ) -> Result<Self, BuildError> {
        let bad = |m: String| Err(BuildError::BadTrie(m));
        let n_strings = strings.len();
        let id_ok = |id: u32| (id as usize) < n_strings;
        let strings: Vec<Box<str>> = strings.into_iter().map(String::into_boxed_str).collect();
        let mut ids: FxHashMap<&str, u32> = FxHashMap::default();
        for (i, s) in strings.iter().enumerate() {
            if ids.insert(s, id_of(i)?).is_some() {
                return bad(format!("duplicate string {s:?}"));
            }
        }
        let mut bang = vec![NONE; strings.len()];
        for (s, &i) in &ids {
            if let Some(rest) = s.strip_prefix('!') {
                let Some(&j) = ids.get(rest) else {
                    return bad(format!("{s:?} without {rest:?}"));
                };
                bang[j as usize] = i;
            }
        }
        let star = ids.get("*").copied().unwrap_or(NONE);
        let mut labels: FxHashMap<Box<str>, u32> = FxHashMap::default();
        for &id in self_decoding {
            if !id_ok(id) {
                return bad(format!("string id {id}"));
            }
            labels.insert(strings[id as usize].clone(), id);
        }
        for (a, id) in aliases {
            if !id_ok(id) {
                return bad(format!("string id {id}"));
            }
            if labels
                .insert(a.clone().into_boxed_str(), id)
                .is_some_and(|old| old != id)
            {
                return bad(format!("label {a:?} decodes two ways"));
            }
        }
        if nodes.is_empty() {
            return bad("no root".to_owned());
        }
        let mut edge_map = FxHashMap::default();
        for &(parent, key, child) in edges {
            if parent as usize >= nodes.len() || child as usize >= nodes.len() || !id_ok(key) {
                return bad(format!("edge {parent} -{key}-> {child}"));
            }
            if edge_map.insert(edge_key(parent, key), child).is_some() {
                return bad(format!("two edges {parent} -{key}->"));
            }
        }
        Ok(Self {
            labels,
            strings,
            bang,
            star,
            nodes: nodes
                .iter()
                .map(|&(end, private)| Node { end, private })
                .collect(),
            edges: edge_map,
            defer_xn,
        })
    }

    /// tldextract's `Trie.create(public, private)` from suffix rules, every
    /// label decoding to itself plus the given `(A-label, label)` aliases (for
    /// tests and pure-Rust callers; Python builds the real one with [`new`]).
    ///
    /// [`new`]: Self::new
    pub fn from_rules(
        public: &[&str],
        private: &[&str],
        aliases: &[(&str, &str)],
    ) -> Result<Self, BuildError> {
        fn intern(
            s: &str,
            strings: &mut Vec<String>,
            ids: &mut FxHashMap<String, u32>,
        ) -> Result<u32, BuildError> {
            if let Some(&id) = ids.get(s) {
                return Ok(id);
            }
            let id = id_of(strings.len())?;
            strings.push(s.to_owned());
            ids.insert(s.to_owned(), id);
            Ok(id)
        }
        let mut strings = Vec::new();
        let mut ids = FxHashMap::default();
        let mut nodes = vec![(false, false)];
        let mut edges: FxHashMap<(u32, u32), u32> = FxHashMap::default();
        let rules = public.iter().map(|r| (r, false));
        for (rule, private) in rules.chain(private.iter().map(|r| (r, true))) {
            let mut node = 0u32;
            for label in rule.rsplit('.') {
                let key = intern(label, &mut strings, &mut ids)?;
                if let Some(rest) = label.strip_prefix('!') {
                    intern(rest, &mut strings, &mut ids)?;
                }
                node = if let Some(&child) = edges.get(&(node, key)) {
                    child
                } else {
                    let child = id_of(nodes.len())?;
                    nodes.push((false, false));
                    edges.insert((node, key), child);
                    child
                };
            }
            nodes[node as usize] = (true, private);
        }
        let mut alias_ids = Vec::new();
        for &(a, s) in aliases {
            alias_ids.push((a.to_owned(), intern(s, &mut strings, &mut ids)?));
        }
        let self_ids: Vec<u32> = (0..id_of(strings.len())?).collect();
        let edges: Vec<(u32, u32, u32)> = edges.into_iter().map(|((p, k), c)| (p, k, c)).collect();
        Self::new(strings, &self_ids, alias_ids, &nodes, &edges, false)
    }

    /// Number of trie nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of distinct label strings.
    pub fn string_count(&self) -> usize {
        self.strings.len()
    }

    /// Number of lowered labels that decode to a string of the trie.
    pub fn label_count(&self) -> usize {
        self.labels.len()
    }

    #[inline]
    fn edge(&self, node: u32, key: u32) -> Option<u32> {
        if key == NONE {
            return None;
        }
        self.edges.get(&edge_key(node, key)).copied()
    }

    /// The id of `_decode_punycode(label)` if it is one of the trie's strings.
    fn decoded(&self, label: &str, v: UnicodeVersion) -> Result<Option<u32>, Deferred> {
        let lowered = py_lower(label, v);
        if let Some(&id) = self.labels.get(&*lowered) {
            return Ok(Some(id));
        }
        if self.defer_xn && lowered.is_ascii() && lowered.starts_with("xn--") {
            return Err(Deferred);
        }
        Ok(None)
    }

    /// `_extract_netloc(s)` on the dot-replaced netloc `s`: the registered
    /// domain and the FQDN.
    fn pieces(&self, s: &str, v: UnicodeVersion) -> Result<(Piece, Piece), Deferred> {
        let b = s.as_bytes();
        if b.first() == Some(&b'[')
            && b.last() == Some(&b']')
            && py_len(s) >= 4
            && looks_like_ipv6(&s[1..s.len() - 1])
        {
            return Ok((Piece::Empty, Piece::Empty));
        }
        // suffix_index(labels), walking the labels from the right.
        let n = memchr::memchr_iter(b'.', b).count() + 1;
        let (mut node, mut label_idx, mut end) = (0u32, n, s.len());
        let mut found = None; // (index, start byte, node) of the last suffix end
        let mut wildcard = None;
        for i in (0..n).rev() {
            let start = memchr::memrchr(b'.', &b[..end]).map_or(0, |p| p + 1);
            let id = self.decoded(&s[start..end], v)?;
            if let Some(child) = id.and_then(|k| self.edge(node, k)) {
                label_idx -= 1;
                node = child;
                if self.nodes[child as usize].end {
                    found = Some((label_idx, start, child));
                }
                end = start.saturating_sub(1);
                continue;
            }
            if let Some(star) = self.edge(node, self.star) {
                let exception =
                    id.is_some_and(|k| self.edge(node, self.bang[k as usize]).is_some());
                // The exception's suffix starts after this label (end + 1, past
                // the netloc when no label matched).
                wildcard = Some(if exception {
                    (label_idx, end + 1, star)
                } else {
                    (i, start, star)
                });
            }
            break;
        }
        let Some((psi, sstart, snode)) = wildcard.or(found) else {
            return Ok((Piece::Empty, Piece::Empty));
        };
        // ExtractResult(subdomain, domain, suffix, is_private)
        if psi == n || sstart >= s.len() {
            return Ok((Piece::Empty, Piece::Empty)); // suffix == ""
        }
        let private = self.nodes[snode as usize].private;
        let (ds, de) = if psi == 0 {
            (0, 0)
        } else {
            let de = sstart - 1;
            (memchr::memrchr(b'.', &b[..de]).map_or(0, |p| p + 1), de)
        };
        let domain = ds < de;
        let sub = if psi >= 2 { &s[..ds - 1] } else { "" };
        let registered = if domain {
            Piece::Slice(ds, s.len())
        } else {
            Piece::Empty
        };
        let fqdn = match (sub.is_empty(), domain) {
            _ if !domain && !private => Piece::Empty,
            (true, true) => Piece::Slice(ds, s.len()),
            (false, true) => Piece::Slice(0, s.len()),
            (true, false) => Piece::Slice(sstart, s.len()),
            (false, false) => Piece::Joined([sub, &s[sstart..]].join(".")),
        };
        Ok((registered, fqdn))
    }

    /// `TLDExtract.extract_str(url)`'s `top_domain_under_public_suffix` and
    /// `fqdn`, lowering labels as CPython with Unicode `v` does.
    pub fn domain_parts<'a>(
        &self,
        url: &'a str,
        v: UnicodeVersion,
    ) -> Result<DomainParts<'a>, Deferred> {
        let netloc = ascii_dots(lenient_netloc(url));
        let (registered, fqdn) = self.pieces(&netloc, v)?;
        Ok(DomainParts {
            registered: piece(&netloc, registered),
            fqdn: piece(&netloc, fqdn),
        })
    }
}

// ---------------------------------------------------------------------------
// Banned subwords: an Aho-Corasick DFA over [a-z0-9]
// ---------------------------------------------------------------------------

const SYMBOLS: usize = 36;

/// The symbol of a byte of `normalize(url)` (`[a-zA-Z0-9]`, lowercased), or
/// `None` for a byte `normalize` removes.
#[inline]
fn symbol(b: u8) -> Option<usize> {
    match b {
        b'0'..=b'9' => Some(usize::from(b - b'0')),
        b'a'..=b'z' => Some(10 + usize::from(b - b'a')),
        b'A'..=b'Z' => Some(10 + usize::from(b - b'A')),
        _ => None,
    }
}

/// "Is any word a substring of `normalize(text)`?", one pass, no allocation.
#[derive(Debug)]
struct Dfa {
    delta: Vec<u32>,
    accept: Vec<bool>,
}

impl Dfa {
    /// Words with a byte outside `[a-z0-9]` can never be a substring of
    /// `normalize(...)` and are left out.
    fn new<'w>(words: impl IntoIterator<Item = &'w str>) -> Self {
        let mut delta = vec![NONE; SYMBOLS];
        let mut accept = vec![false];
        for w in words {
            if w.is_empty() || !w.bytes().all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9')) {
                continue;
            }
            let mut s = 0usize;
            for b in w.bytes() {
                let k = s * SYMBOLS + symbol(b).unwrap_or(0);
                if delta[k] == NONE {
                    delta[k] = u32::try_from(accept.len()).unwrap_or(NONE);
                    delta.extend([NONE; SYMBOLS]);
                    accept.push(false);
                }
                s = delta[k] as usize;
            }
            accept[s] = true;
        }
        // Breadth-first: failure links, then complete transitions.
        let mut fail = vec![0u32; accept.len()];
        let mut queue = std::collections::VecDeque::new();
        for d in delta.iter_mut().take(SYMBOLS) {
            if *d == NONE {
                *d = 0;
            } else {
                queue.push_back(*d);
            }
        }
        while let Some(s) = queue.pop_front() {
            let s = s as usize;
            let f = fail[s] as usize;
            accept[s] |= accept[f];
            for c in 0..SYMBOLS {
                let t = delta[s * SYMBOLS + c];
                let via_fail = delta[f * SYMBOLS + c];
                if t == NONE {
                    delta[s * SYMBOLS + c] = via_fail;
                } else {
                    fail[t as usize] = via_fail;
                    queue.push_back(t);
                }
            }
        }
        Self { delta, accept }
    }

    #[inline]
    fn matches(&self, text: &str) -> bool {
        let mut s = 0usize;
        for b in text.bytes() {
            if let Some(c) = symbol(b) {
                s = self.delta[s * SYMBOLS + c] as usize;
                if self.accept[s] {
                    return true;
                }
            }
        }
        false
    }
}

#[derive(Debug)]
enum Subwords {
    /// `self.banned_subwords` is empty: the check is skipped.
    Off,
    /// Only `""`: pyahocorasick's automaton has no words and raises.
    NoAutomaton,
    Dfa(Dfa),
}

// ---------------------------------------------------------------------------
// The filter
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Word {
    hard: bool,
    soft: u32,
}

const NO_WORD: Word = Word {
    hard: false,
    soft: NONE,
};

/// What a [`UrlFilter`] blocks: datatrove's final sets (after `parse_list` /
/// `get_list`), and the soft-word threshold as a count.
#[derive(Debug)]
pub struct UrlLists {
    /// `block_listed_domains`.
    pub domains: StrSet,
    /// `block_listed_url`.
    pub urls: StrSet,
    /// `banned_words` (normalized).
    pub banned_words: Vec<String>,
    /// `soft_banned_words` (normalized).
    pub soft_banned_words: Vec<String>,
    /// `banned_subwords` (normalized; `""` entries kept: they decide
    /// [`UrlError::NoAutomaton`]).
    pub banned_subwords: Vec<String>,
    /// The smallest count `n` with `n >= soft_word_threshold` true
    /// (`u64::MAX`: never). Default 2, datatrove's default threshold.
    pub min_soft_words: u64,
}

impl Default for UrlLists {
    fn default() -> Self {
        Self {
            domains: StrSet::new(),
            urls: StrSet::new(),
            banned_words: Vec::new(),
            soft_banned_words: Vec::new(),
            banned_subwords: Vec::new(),
            min_soft_words: 2,
        }
    }
}

/// datatrove 0.10.1's `URLFilter`, built once.
#[derive(Debug)]
pub struct UrlFilter {
    suffixes: SuffixList,
    domains: StrSet,
    urls: StrSet,
    words: FxHashMap<Box<[u8]>, Word>,
    min_soft: u64,
    subwords: Subwords,
    unicode: UnicodeVersion,
}

impl UrlFilter {
    /// A filter over `lists`, extracting domains with `suffixes` and CPython's
    /// `str.lower()` of Unicode `unicode`.
    pub fn new(suffixes: SuffixList, lists: UrlLists, unicode: UnicodeVersion) -> Self {
        let mut words: FxHashMap<Box<[u8]>, Word> = FxHashMap::default();
        for w in &lists.banned_words {
            words.entry(w.as_bytes().into()).or_insert(NO_WORD).hard = true;
        }
        let mut n_soft = 0u32;
        for w in &lists.soft_banned_words {
            let e = words.entry(w.as_bytes().into()).or_insert(NO_WORD);
            if e.soft == NONE {
                e.soft = n_soft;
                n_soft += 1;
            }
        }
        let subwords = if lists.banned_subwords.is_empty() {
            Subwords::Off
        } else if lists.banned_subwords.iter().all(String::is_empty) {
            Subwords::NoAutomaton
        } else {
            Subwords::Dfa(Dfa::new(lists.banned_subwords.iter().map(String::as_str)))
        };
        Self {
            suffixes,
            domains: lists.domains,
            urls: lists.urls,
            words,
            min_soft: lists.min_soft_words,
            subwords,
            unicode,
        }
    }

    /// The suffix trie.
    pub fn suffixes(&self) -> &SuffixList {
        &self.suffixes
    }

    /// The blocked domains.
    pub fn domains(&self) -> &StrSet {
        &self.domains
    }

    /// The blocked URLs.
    pub fn urls(&self) -> &StrSet {
        &self.urls
    }

    /// The Unicode version labels are lowered with.
    pub fn unicode_version(&self) -> UnicodeVersion {
        self.unicode
    }

    /// tldextract's registered domain and FQDN of `url`.
    pub fn domain_parts<'a>(&self, url: &'a str) -> Result<DomainParts<'a>, Deferred> {
        self.suffixes.domain_parts(url, self.unicode)
    }

    /// datatrove's `filter()`: `None` to keep, else the drop reason.
    pub fn check(&self, url: &str) -> Result<Option<Reason>, UrlError> {
        if !self.domains.is_empty() {
            let parts = self
                .domain_parts(url)
                .map_err(|Deferred| UrlError::Deferred)?;
            if self.domains.contains(&parts.registered) {
                return Ok(Some(DOMAIN));
            }
            if self.domains.contains(&parts.fqdn) {
                return Ok(Some(SUBDOMAIN));
            }
        }
        self.check_rest(url)
    }

    /// [`check`](Self::check) with the registered domain and FQDN given (by
    /// tldextract, after [`UrlError::Deferred`]).
    pub fn check_with_parts(
        &self,
        url: &str,
        registered: &str,
        fqdn: &str,
    ) -> Result<Option<Reason>, UrlError> {
        if self.domains.contains(registered) {
            return Ok(Some(DOMAIN));
        }
        if self.domains.contains(fqdn) {
            return Ok(Some(SUBDOMAIN));
        }
        self.check_rest(url)
    }

    fn check_rest(&self, url: &str) -> Result<Option<Reason>, UrlError> {
        if self.urls.contains(url) {
            return Ok(Some(URL));
        }
        let b = url.as_bytes();
        let mut soft: Vec<u32> = Vec::new();
        if !self.words.is_empty() {
            // re.split(r"[^a-zA-Z0-9]+", url): the runs, plus "" when the URL
            // is empty or starts or ends with a separator.
            let edge = b.first().is_none_or(|c| !c.is_ascii_alphanumeric())
                || b.last().is_some_and(|c| !c.is_ascii_alphanumeric());
            let words = b
                .split(|c: &u8| !c.is_ascii_alphanumeric())
                .filter(|w| !w.is_empty())
                .chain(edge.then_some(&b[..0]));
            for w in words {
                if let Some(word) = self.words.get(w) {
                    if word.hard {
                        return Ok(Some(HARD_BLACKLISTED));
                    }
                    if word.soft != NONE && !soft.contains(&word.soft) {
                        soft.push(word.soft);
                    }
                }
            }
        }
        if soft.len() as u64 >= self.min_soft {
            return Ok(Some(SOFT_BLACKLISTED));
        }
        match &self.subwords {
            Subwords::Off => Ok(None),
            Subwords::NoAutomaton => Err(UrlError::NoAutomaton),
            Subwords::Dfa(d) => Ok(d.matches(url).then_some(BLACKLISTED_SUBWORD)),
        }
    }

    /// Bytes held by the domain and URL sets.
    pub fn heap_bytes(&self) -> usize {
        self.domains.heap_bytes() + self.urls.heap_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    /// tldextract's default trie: public suffixes only.
    fn psl() -> SuffixList {
        SuffixList::from_rules(
            &[
                "com",
                "uk",
                "co.uk",
                "*.ck",
                "!www.ck",
                "jp",
                "*.kawasaki.jp",
                "!city.kawasaki.jp",
                "рф",
                "公司.cn",
                "cn",
            ],
            &[],
            &[("xn--p1ai", "рф"), ("xn--55qx5d", "公司")],
        )
        .unwrap()
    }

    /// `include_psl_private_domains=True`.
    fn psl_private() -> SuffixList {
        SuffixList::from_rules(&["com", "uk", "co.uk"], &["blogspot.com", "s3.co.uk"], &[]).unwrap()
    }

    fn parts(p: &SuffixList, url: &str) -> (String, String) {
        let d = p.domain_parts(url, V).unwrap();
        (d.registered.into_owned(), d.fqdn.into_owned())
    }

    #[test]
    fn netloc() {
        let cases = [
            ("http://forums.news.cnn.com/", "forums.news.cnn.com"),
            ("https://user:pw@Example.COM:8080/a?b#c", "Example.COM"),
            ("//example.com/x", "example.com"),
            ("example.com/x", "example.com"),
            ("a+b-c.d://host/x", "host"),
            ("a_b://host/x", "a_b"),
            ("://host", ""),
            ("x:/host", "x"),
            ("http://[::1]:80/", "[::1]"),
            ("http://[::1/", "["),
            ("http://u@[fe80::1%eth0]x", "[fe80::1%eth0]"),
            ("http://a@b@c.com", "c.com"),
            ("http:// a.com ./", "a.com "),
            ("http://example.com.。．｡", "example.com"),
            ("http://\u{2003}example.com\u{85}", "example.com"),
            ("", ""),
            ("http://", ""),
        ];
        for (url, want) in cases {
            assert_eq!(lenient_netloc(url), want, "{url:?}");
        }
    }

    #[test]
    fn ipv6() {
        for ok in [
            "::",
            "::1",
            "1::",
            "1:2:3:4:5:6:7:8",
            "1:2:3:4:5:6:7::",
            "::2:3:4:5:6:7:8",
            "fe80::1%eth0",
            "fe80::1%é",
            "::ffff:1.2.3.4",
            "1:2:3:4:5:6:1.2.3.4",
            "aBcD:ef01:2345:6789:aBcD:ef01:127.0.0.1",
            "0:0:0:0:0:0:0:0",
        ] {
            assert!(looks_like_ipv6(ok), "{ok:?}");
        }
        for bad in [
            "",
            ":",
            "1:2",
            ":::",
            "1:2:3:4:5:6:7:8:9",
            "1:2:3:4:5:6:7",
            "1::2::3",
            ":1::",
            "::1:",
            "12345::",
            "g::",
            "::%",
            "::1%a%b",
            "::1/64",
            "::1.2.3",
            "::01.2.3.4",
            "::256.1.1.1",
            "1:2:3:4:5:6:7:1.2.3.4",
            "::１",
            "1:2:3:4:5:6:7:8::",
            "::1:2:3:4:5:6:7:8",
        ] {
            assert!(!looks_like_ipv6(bad), "{bad:?}");
        }
    }

    #[test]
    fn lower() {
        assert_eq!(py_lower("ABC", V), "abc");
        assert_eq!(py_lower("İ", V), "i\u{307}");
        assert_eq!(py_lower("ΑΣ", V), "ας");
        assert_eq!(py_lower(" ΑΣ ", V), " ας ");
        assert_eq!(py_lower("ΣΑ", V), "σα");
        assert_eq!(py_lower("\u{212a}Z", V), "kz");
        assert_eq!(py_lower("\u{3000}", V), "\u{3000}");
    }

    #[test]
    fn suffixes() {
        let p = psl();
        let cases = [
            ("http://forums.bbc.co.uk/x", "bbc.co.uk", "forums.bbc.co.uk"),
            ("http://Example.COM", "Example.COM", "Example.COM"),
            ("http://localhost:8080", "", ""),
            ("http://127.0.0.1/", "", ""),
            ("http://[::1]/", "", ""),
            ("http://com", "", ""),
            ("http://co.uk", "", ""),
            ("http://a.b.ck", "a.b.ck", "a.b.ck"),
            ("http://b.ck", "", ""),
            ("http://www.ck", "www.ck", "www.ck"),
            ("http://a.www.ck", "www.ck", "a.www.ck"),
            (
                "http://x.city.kawasaki.jp",
                "city.kawasaki.jp",
                "x.city.kawasaki.jp",
            ),
            (
                "http://x.foo.kawasaki.jp",
                "x.foo.kawasaki.jp",
                "x.foo.kawasaki.jp",
            ),
            ("http://*.ck", "", ""),
            ("http://a.!www.ck", "a.!www.ck", "a.!www.ck"),
            ("http://пример.рф", "пример.рф", "пример.рф"),
            ("http://пример.РФ", "пример.РФ", "пример.РФ"),
            (
                "http://xn--e1afmkfd.xn--p1ai",
                "xn--e1afmkfd.xn--p1ai",
                "xn--e1afmkfd.xn--p1ai",
            ),
            ("http://a.XN--P1AI", "a.XN--P1AI", "a.XN--P1AI"),
            ("http://a.xn--p1aj", "", ""),
            (
                "http://a.b.xn--55qx5d.cn",
                "b.xn--55qx5d.cn",
                "a.b.xn--55qx5d.cn",
            ),
            ("http://a.\u{212a}Z", "", ""),
            ("http://x.blogspot.com", "blogspot.com", "x.blogspot.com"),
            ("http://a。b．com", "b.com", "a.b.com"),
            ("http://.a.com", "a.com", "a.com"),
            ("http://..a.com", "a.com", "..a.com"),
            ("http://x..com", "", ""),
            ("http://a. com", "", ""),
        ];
        for (url, rd, fqdn) in cases {
            assert_eq!(parts(&p, url), (rd.to_owned(), fqdn.to_owned()), "{url:?}");
        }
    }

    #[test]
    fn private_suffixes() {
        let p = psl_private();
        let cases = [
            ("http://x.blogspot.com", "x.blogspot.com", "x.blogspot.com"),
            ("http://blogspot.com", "", "blogspot.com"),
            ("http://a..blogspot.com", "", "a.blogspot.com"),
            ("http://.blogspot.com", "", "blogspot.com"),
            ("http://b.s3.co.uk", "b.s3.co.uk", "b.s3.co.uk"),
        ];
        for (url, rd, fqdn) in cases {
            assert_eq!(parts(&p, url), (rd.to_owned(), fqdn.to_owned()), "{url:?}");
        }
    }

    #[test]
    fn deferral() {
        let deferring = SuffixList {
            defer_xn: true,
            ..psl()
        };
        assert!(deferring.domain_parts("http://a.xn--p1ai", V).is_ok());
        assert_eq!(
            deferring.domain_parts("http://a.xn--p1aj", V),
            Err(Deferred)
        );
        assert!(deferring.domain_parts("http://a.com", V).is_ok());
    }

    #[test]
    fn str_set() {
        let mut s = StrSet::new();
        assert!(!s.contains(""));
        for i in 0..10_000 {
            assert!(s.insert(&format!("d{i}.com")).unwrap());
        }
        assert!(!s.insert("d5.com").unwrap());
        assert!(s.insert("").unwrap());
        assert_eq!(s.len(), 10_001);
        assert!(s.contains("d9999.com") && s.contains("") && !s.contains("d10000.com"));
        let mut t = StrSet::new();
        t.extend_stripped_lines("a.com\n#b.com\n \n  c.com \t\n #d.com\ne.com")
            .unwrap();
        for (x, want) in [
            ("a.com", true),
            ("b.com", false),
            ("#b.com", false),
            ("", true),
            ("c.com", true),
            ("#d.com", true),
            ("e.com", true),
        ] {
            assert_eq!(t.contains(x), want, "{x:?}");
        }
    }

    fn filter(lists: UrlLists) -> UrlFilter {
        UrlFilter::new(psl(), lists, V)
    }

    fn set(xs: &[&str]) -> StrSet {
        let mut s = StrSet::new();
        for x in xs {
            s.insert(x).unwrap();
        }
        s
    }

    fn strings(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| (*s).to_owned()).collect()
    }

    fn reason(f: &UrlFilter, url: &str) -> Option<String> {
        f.check(url).unwrap().map(|r| r.as_str().into_owned())
    }

    #[test]
    fn checks_in_order() {
        let f = filter(UrlLists {
            domains: set(&["bad.com", "sub.ok.com"]),
            urls: set(&["http://ok.com/x"]),
            banned_words: strings(&["porn"]),
            soft_banned_words: strings(&["sex", "cam", "cam"]),
            banned_subwords: strings(&["girlsgone", ""]),
            min_soft_words: 2,
        });
        let cases = [
            ("http://bad.com/porn", Some("domain")),
            ("http://x.sub.ok.com/", None),
            ("http://sub.ok.com/", Some("subdomain")),
            ("http://ok.com/x", Some("url")),
            ("http://ok.com/porn-sex", Some("hard_blacklisted")),
            ("http://ok.com/PORN", None),
            ("http://ok.com/sex/sex", None),
            ("http://ok.com/sex/cam", Some("soft_blacklisted")),
            ("http://ok.com/Girls-Gone", Some("blacklisted_subword")),
            ("http://ok.com/girl", None),
        ];
        for (url, want) in cases {
            assert_eq!(reason(&f, url).as_deref(), want, "{url:?}");
        }
    }

    #[test]
    fn empty_words_and_thresholds() {
        let f = filter(UrlLists {
            banned_words: strings(&[""]),
            ..UrlLists::default()
        });
        assert_eq!(
            reason(&f, "http://a.com/").as_deref(),
            Some("hard_blacklisted")
        );
        assert_eq!(reason(&f, "a.com").as_deref(), None);
        assert_eq!(reason(&f, "").as_deref(), Some("hard_blacklisted"));
        let always = filter(UrlLists {
            min_soft_words: 0,
            ..UrlLists::default()
        });
        assert_eq!(reason(&always, "a").as_deref(), Some("soft_blacklisted"));
        let never = filter(UrlLists {
            min_soft_words: u64::MAX,
            banned_subwords: strings(&[""]),
            ..UrlLists::default()
        });
        assert_eq!(never.check("a"), Err(UrlError::NoAutomaton));
        let blank_domain = filter(UrlLists {
            domains: set(&[""]),
            ..UrlLists::default()
        });
        assert_eq!(
            reason(&blank_domain, "http://localhost/").as_deref(),
            Some("domain")
        );
    }

    #[test]
    fn dfa() {
        let d = Dfa::new(["he", "she", "his", "hers", "x9"]);
        for (t, want) in [
            ("ushers", true),
            ("S-H-E", true),
            ("hi", false),
            ("ahxis", false),
            ("h/i/s", true),
            ("x é 9", true),
            ("", false),
        ] {
            assert_eq!(d.matches(t), want, "{t:?}");
        }
    }
}
