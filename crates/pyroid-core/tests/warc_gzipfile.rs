//! Differential test of `warc::gzipfile` against CPython's `gzip.GzipFile` read path.
//!
//! The inputs are generated here from fixed seeds by a small deflate encoder of our own
//! (stored, fixed and dynamic blocks), so they never change with a library version. Each case
//! is an input, a source behaviour (whole reads, short reads, a source that raises) and a read
//! size; its trace is every `GzipFile.read(n)` result (length and bytes, or the exception type
//! and message), `tell()` after each call, and `mtime` at the end. `warc_gzipfile_data/gzip.json`
//! holds digests of the traces CPython 3.11-3.15 produced, per group of cases; every Python
//! version must match the parameter family it maps to.
//!
//! To regenerate the fixture:
//! `GZIP_CASES_OUT=<dir> cargo test --test warc_gzipfile -- --ignored dump_cases`, then for each
//! Python `python -I tests/warc_gzipfile_data/trace.py gzip <dir> <part.json> [detail dir]` and
//! `trace.py merge tests/warc_gzipfile_data/gzip.json <part.json>...`. With
//! `GZIP_TRACE_DETAIL=<detail dir>` the test names the cases that differ.
#![allow(
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::print_stderr,
    clippy::too_many_lines
)]
// a test harness: the deflate encoder packs bit fields, the case table is long, the run reports its counts

use std::collections::BTreeMap;
use std::fmt::Write as _;

use flate2::Crc;
use pyroid_core::warc::gzipfile::{GzipFile, GzipParams};
use pyroid_core::warc::{Error, PyExc, Source};
use sha1::{Digest, Sha1};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/warc_gzipfile_data");

// ---------------------------------------------------------------------------------------------
// Deterministic inputs
// ---------------------------------------------------------------------------------------------

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        usize::try_from(self.next() % n as u64).unwrap()
    }

    fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }
}

const WORDS: &[&str] = &[
    "the",
    "of",
    "and",
    "to",
    "in",
    "data",
    "record",
    "archive",
    "crawl",
    "page",
    "html",
    "body",
    "div",
    "class",
    "content",
    "news",
    "with",
    "from",
    "this",
    "that",
    "for",
    "are",
    "was",
    "which",
    "their",
    "text",
    "link",
    "href",
    "https",
    "example",
    "org",
    "com",
    "index",
    "main",
    "header",
    "footer",
    "script",
    "style",
    "width",
    "height",
    "image",
    "title",
    "meta",
    "charset",
    "utf",
    "lang",
    "en",
    "search",
    "menu",
    "item",
    "list",
    "table",
    "row",
    "cell",
    "span",
    "strong",
    "about",
    "contact",
    "privacy",
    "terms",
    "copyright",
    "2026",
    "March",
    "update",
];

/// A WARC response record of about `len` bytes: headers, HTTP headers, an HTML-ish body.
fn warc_record(rng: &mut Rng, len: usize) -> Vec<u8> {
    let mut body = String::new();
    while body.len() < len {
        match rng.below(20) {
            0 => body.push_str("\r\n"),
            1 => {
                let _ = write!(
                    body,
                    "<a href=\"https://example.org/{}\">",
                    rng.below(100_000)
                );
            }
            2 => {
                let _ = write!(body, "{} ", rng.next() % 1_000_000);
            }
            3 => body.push_str("</div>\n<div class=\"content\">"),
            _ => {
                body.push_str(WORDS[rng.below(WORDS.len())]);
                body.push(' ');
            }
        }
    }
    let mut out = format!(
        "WARC/1.0\r\nWARC-Type: response\r\nWARC-Record-ID: <urn:uuid:{:016x}{:016x}>\r\n\
         WARC-Target-URI: https://example.org/{}\r\nContent-Type: application/http; msgtype=response\r\n\
         Content-Length: {}\r\n\r\nHTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n",
        rng.next(),
        rng.next(),
        rng.below(1000),
        body.len() + 64
    )
    .into_bytes();
    out.extend_from_slice(body.as_bytes());
    out.extend_from_slice(b"\r\n\r\n");
    out
}

struct Bits {
    out: Vec<u8>,
    acc: u64,
    n: u32,
}

impl Bits {
    fn new() -> Self {
        Self {
            out: Vec::new(),
            acc: 0,
            n: 0,
        }
    }

    fn put(&mut self, v: u32, bits: u32) {
        self.acc |= u64::from(v) << self.n;
        self.n += bits;
        while self.n >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }

    /// A Huffman code (stored MSB first).
    fn code(&mut self, code: u32, len: u8) {
        let mut r = 0;
        for i in 0..len {
            r |= ((code >> i) & 1) << (len - 1 - i);
        }
        self.put(r, u32::from(len));
    }

    fn align(&mut self) {
        if self.n > 0 {
            self.put(0, 8 - self.n);
        }
    }
}

#[derive(Clone, Copy)]
enum Tok {
    Lit(u8),
    Match(u16, u16),
}

impl Tok {
    fn len(self) -> usize {
        match self {
            Tok::Lit(_) => 1,
            Tok::Match(l, _) => usize::from(l),
        }
    }
}

/// Greedy LZ77 with a 32 KiB window and short hash chains.
fn lz77(data: &[u8]) -> Vec<Tok> {
    const NONE: usize = usize::MAX;
    let hash = |i: usize| {
        (usize::from(data[i]) << 10 ^ usize::from(data[i + 1]) << 5 ^ usize::from(data[i + 2]))
            & 0x7fff
    };
    let mut head = vec![NONE; 1 << 15];
    let mut prev = vec![NONE; data.len()];
    let insert = |i: usize, head: &mut Vec<usize>, prev: &mut Vec<usize>| {
        if i + 3 <= data.len() {
            let h = hash(i);
            prev[i] = head[h];
            head[h] = i;
        }
    };
    let mut toks = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let (mut best, mut dist) = (0, 0);
        if i + 3 <= data.len() {
            let mut cand = head[hash(i)];
            let mut chain = 0;
            while cand != NONE && i - cand <= 32768 && chain < 16 {
                let max = (data.len() - i).min(258);
                let l = (0..max)
                    .take_while(|&k| data[cand + k] == data[i + k])
                    .count();
                if l > best {
                    best = l;
                    dist = i - cand;
                }
                cand = prev[cand];
                chain += 1;
            }
        }
        if best >= 3 {
            toks.push(Tok::Match(best as u16, dist as u16));
            for k in i..i + best {
                insert(k, &mut head, &mut prev);
            }
            i += best;
        } else {
            toks.push(Tok::Lit(data[i]));
            insert(i, &mut head, &mut prev);
            i += 1;
        }
    }
    toks
}

const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn len_sym(l: u16) -> usize {
    if l == 258 {
        28
    } else {
        LEN_BASE.iter().rposition(|&b| b <= l).unwrap()
    }
}

fn dist_sym(d: u16) -> usize {
    DIST_BASE.iter().rposition(|&b| b <= d).unwrap()
}

/// Huffman code lengths (deterministic tree, frequencies halved until within `limit`).
fn huff_lengths(freq: &[u64], limit: u8) -> Vec<u8> {
    let mut f = freq.to_vec();
    loop {
        // nodes: (weight, children); leaves first.
        let mut weight: Vec<u64> = Vec::new();
        let mut kids: Vec<Option<(usize, usize)>> = Vec::new();
        let mut leaf_of: Vec<usize> = Vec::new();
        let mut active = Vec::new();
        for (s, &w) in f.iter().enumerate() {
            if w > 0 {
                active.push(weight.len());
                weight.push(w);
                kids.push(None);
                leaf_of.push(s);
            }
        }
        let mut lens = vec![0u8; f.len()];
        if active.len() == 1 {
            lens[leaf_of[0]] = 1;
            return lens;
        }
        while active.len() > 1 {
            active.sort_by_key(|&i| (weight[i], i));
            let (a, b) = (active.remove(0), active.remove(0));
            active.push(weight.len());
            weight.push(weight[a] + weight[b]);
            kids.push(Some((a, b)));
        }
        let mut depth = vec![0u8; weight.len()];
        let mut max = 0;
        for i in (0..weight.len()).rev() {
            if let Some((a, b)) = kids[i] {
                depth[a] = depth[i] + 1;
                depth[b] = depth[i] + 1;
            }
        }
        for (i, s) in leaf_of.iter().enumerate() {
            lens[*s] = depth[i];
            max = max.max(depth[i]);
        }
        if max <= limit {
            return lens;
        }
        for w in &mut f {
            if *w > 0 {
                *w = w.div_ceil(2);
            }
        }
    }
}

fn canonical(lens: &[u8]) -> Vec<u32> {
    let mut count = [0u32; 16];
    for &l in lens {
        count[usize::from(l)] += 1;
    }
    count[0] = 0;
    let mut next = [0u32; 16];
    let mut code = 0;
    for b in 1..16 {
        code = (code + count[b - 1]) << 1;
        next[b] = code;
    }
    lens.iter()
        .map(|&l| {
            if l == 0 {
                0
            } else {
                let c = next[usize::from(l)];
                next[usize::from(l)] += 1;
                c
            }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Stored,
    Fixed,
    Dynamic,
}

fn fixed_lengths() -> (Vec<u8>, Vec<u8>) {
    let mut ll = vec![8u8; 288];
    ll[144..256].fill(9);
    ll[256..280].fill(7);
    (ll, vec![5u8; 30])
}

fn write_tokens(bits: &mut Bits, toks: &[Tok], ll: &[u8], dl: &[u8]) {
    let (lc, dc) = (canonical(ll), canonical(dl));
    for &t in toks {
        match t {
            Tok::Lit(b) => bits.code(lc[usize::from(b)], ll[usize::from(b)]),
            Tok::Match(l, d) => {
                let s = len_sym(l);
                bits.code(lc[257 + s], ll[257 + s]);
                bits.put(u32::from(l - LEN_BASE[s]), u32::from(LEN_EXTRA[s]));
                let s = dist_sym(d);
                bits.code(dc[s], dl[s]);
                bits.put(u32::from(d - DIST_BASE[s]), u32::from(DIST_EXTRA[s]));
            }
        }
    }
    bits.code(lc[256], ll[256]);
}

fn dynamic_block(bits: &mut Bits, toks: &[Tok], last: bool) {
    let mut lf = vec![0u64; 286];
    let mut df = vec![0u64; 30];
    for &t in toks {
        match t {
            Tok::Lit(b) => lf[usize::from(b)] += 1,
            Tok::Match(l, d) => {
                lf[257 + len_sym(l)] += 1;
                df[dist_sym(d)] += 1;
            }
        }
    }
    lf[256] += 1;
    if lf.iter().filter(|&&f| f > 0).count() < 2 {
        lf[0] += 1;
    }
    while df.iter().filter(|&&f| f > 0).count() < 2 {
        let z = df.iter().position(|&f| f == 0).unwrap();
        df[z] = 1;
    }
    let ll = huff_lengths(&lf, 15);
    let dl = huff_lengths(&df, 15);
    let nlit = (257..=286)
        .rev()
        .find(|&n| ll[n - 1] != 0)
        .unwrap_or(257)
        .max(257);
    let ndist = (1..=30).rev().find(|&n| dl[n - 1] != 0).unwrap_or(1);
    let all: Vec<u8> = ll[..nlit].iter().chain(&dl[..ndist]).copied().collect();
    // Run-length code the lengths: (symbol, extra value, extra bits).
    let mut rle = Vec::new();
    let mut i = 0;
    while i < all.len() {
        let l = all[i];
        let run = all[i..].iter().take_while(|&&x| x == l).count();
        if l == 0 {
            let mut r = run;
            while r >= 11 {
                let k = r.min(138);
                rle.push((18u8, (k - 11) as u32, 7u32));
                r -= k;
            }
            if r >= 3 {
                rle.push((17, (r - 3) as u32, 3));
                r = 0;
            }
            rle.extend(std::iter::repeat_n((0, 0, 0), r));
        } else {
            rle.push((l, 0, 0));
            let mut r = run - 1;
            while r >= 3 {
                let k = r.min(6);
                rle.push((16, (k - 3) as u32, 2));
                r -= k;
            }
            rle.extend(std::iter::repeat_n((l, 0, 0), r));
        }
        i += run;
    }
    let mut cf = vec![0u64; 19];
    for &(s, _, _) in &rle {
        cf[usize::from(s)] += 1;
    }
    if cf.iter().filter(|&&f| f > 0).count() < 2 {
        let z = cf.iter().position(|&f| f == 0).unwrap();
        cf[z] = 1;
    }
    let cl = huff_lengths(&cf, 7);
    let cc = canonical(&cl);
    let nclen = (4..=19)
        .rev()
        .find(|&n| cl[CL_ORDER[n - 1]] != 0)
        .unwrap_or(4);
    bits.put(u32::from(last), 1);
    bits.put(2, 2);
    bits.put((nlit - 257) as u32, 5);
    bits.put((ndist - 1) as u32, 5);
    bits.put((nclen - 4) as u32, 4);
    for &s in &CL_ORDER[..nclen] {
        bits.put(u32::from(cl[s]), 3);
    }
    for &(s, v, n) in &rle {
        bits.code(cc[usize::from(s)], cl[usize::from(s)]);
        bits.put(v, n);
    }
    write_tokens(bits, toks, &ll, &dl);
}

/// Raw deflate of `data`, cut into blocks following `plan` (block type, input bytes; the plan
/// repeats; the last block is final). Matches may reach back into earlier blocks.
fn deflate(data: &[u8], plan: &[(Block, usize)]) -> Vec<u8> {
    let toks = lz77(data);
    let mut bits = Bits::new();
    let (mut t, mut at, mut step) = (0, 0, 0);
    loop {
        let (kind, size) = plan[step % plan.len()];
        step += 1;
        let (t0, at0) = (t, at);
        while t < toks.len() && at - at0 < size {
            at += toks[t].len();
            t += 1;
        }
        let last = t == toks.len();
        match kind {
            Block::Stored => {
                let mut chunk = &data[at0..at];
                loop {
                    let n = chunk.len().min(65535);
                    let fin = last && n == chunk.len();
                    bits.put(u32::from(fin), 1);
                    bits.put(0, 2);
                    bits.align();
                    bits.put(n as u32, 16);
                    bits.put(!(n as u32) & 0xffff, 16);
                    bits.out.extend_from_slice(&chunk[..n]);
                    chunk = &chunk[n..];
                    if chunk.is_empty() {
                        break;
                    }
                }
            }
            Block::Fixed => {
                bits.put(u32::from(last), 1);
                bits.put(1, 2);
                let (ll, dl) = fixed_lengths();
                write_tokens(&mut bits, &toks[t0..t], &ll, &dl);
            }
            Block::Dynamic => dynamic_block(&mut bits, &toks[t0..t], last),
        }
        if last {
            break;
        }
    }
    bits.align();
    bits.out
}

#[derive(Clone, Default)]
struct Header {
    flags: u8,
    mtime: u32,
    extra: Vec<u8>,
    name: Vec<u8>,
    comment: Vec<u8>,
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc::new();
    c.update(data);
    c.sum()
}

/// A gzip member; returns the bytes and the offset of its deflate data.
fn member(data: &[u8], plan: &[(Block, usize)], h: &Header) -> (Vec<u8>, usize) {
    let mut out = vec![0x1f, 0x8b, 8, h.flags];
    out.extend_from_slice(&h.mtime.to_le_bytes());
    out.extend_from_slice(&[0, 3]);
    if h.flags & 4 != 0 {
        out.extend_from_slice(&(h.extra.len() as u16).to_le_bytes());
        out.extend_from_slice(&h.extra);
    }
    if h.flags & 8 != 0 {
        out.extend_from_slice(&h.name);
        out.push(0);
    }
    if h.flags & 16 != 0 {
        out.extend_from_slice(&h.comment);
        out.push(0);
    }
    if h.flags & 2 != 0 {
        out.extend_from_slice(&((crc32(&out) & 0xffff) as u16).to_le_bytes());
    }
    let at = out.len();
    out.extend_from_slice(&deflate(data, plan));
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    (out, at)
}

/// Where a member sits in a base file.
struct Span {
    start: usize,
    deflate: usize,
    trailer: usize,
    end: usize,
}

struct Base {
    name: &'static str,
    data: Vec<u8>,
    spans: Vec<Span>,
    plain_len: usize,
}

struct BaseBuilder {
    base: Base,
}

impl BaseBuilder {
    fn new(name: &'static str) -> Self {
        Self {
            base: Base {
                name,
                data: Vec::new(),
                spans: Vec::new(),
                plain_len: 0,
            },
        }
    }

    fn member(mut self, data: &[u8], plan: &[(Block, usize)], h: &Header) -> Self {
        let (m, at) = member(data, plan, h);
        let start = self.base.data.len();
        self.base.spans.push(Span {
            start,
            deflate: start + at,
            trailer: start + m.len() - 8,
            end: start + m.len(),
        });
        self.base.data.extend_from_slice(&m);
        self.base.plain_len += data.len();
        self
    }

    fn raw(mut self, bytes: &[u8]) -> Self {
        self.base.data.extend_from_slice(bytes);
        self
    }
}

use Block::{Dynamic as D, Fixed as F, Stored as S};

fn bases() -> Vec<Base> {
    let mut rng = Rng(0x5eed_2026_0312);
    let h0 = Header::default();
    let mut v = Vec::new();

    let mut b = BaseBuilder::new("warc");
    for (len, plan) in [
        (3000, vec![(D, 1 << 20)]),
        (40_000, vec![(D, 16_384)]),
        (1200, vec![(F, 1 << 20)]),
        (9000, vec![(D, 1 << 20)]),
        (25_000, vec![(D, 8000), (S, 4000), (F, 4000), (D, 1 << 20)]),
        (600, vec![(S, 1 << 20)]),
    ] {
        let rec = warc_record(&mut rng, len);
        b = b.member(&rec, &plan, &h0);
    }
    v.push(b.base);

    let mut b = BaseBuilder::new("big");
    let t1 = warc_record(&mut rng, 200_000);
    let r1 = rng.bytes(150_000);
    let t2 = warc_record(&mut rng, 70_000);
    let mut t3 = warc_record(&mut rng, 150_000);
    t3.extend_from_slice(&rng.bytes(20_000));
    t3.extend_from_slice(&warc_record(&mut rng, 90_000));
    b = b
        .member(&t1, &[(D, 32_768)], &h0)
        .member(&r1, &[(S, 1 << 20)], &h0)
        .member(&t2, &[(F, 1 << 20)], &h0)
        .member(b"", &[(F, 0)], &h0)
        .member(
            &t3,
            &[(D, 65_536), (S, 20_000), (F, 10_000), (D, 1 << 20)],
            &h0,
        );
    v.push(b.base);

    let mut b = BaseBuilder::new("flags");
    let hdrs = [
        Header {
            flags: 0,
            mtime: 0,
            ..Header::default()
        },
        Header {
            flags: 8,
            mtime: 1,
            name: b"file.warc".to_vec(),
            ..Header::default()
        },
        Header {
            flags: 4,
            mtime: 0xffff_ffff,
            extra: Vec::new(),
            ..Header::default()
        },
        Header {
            flags: 4,
            mtime: 1_700_000_000,
            extra: b"AB\x03\x00xyz".to_vec(),
            ..Header::default()
        },
        Header {
            flags: 16,
            comment: b"hello world".to_vec(),
            ..Header::default()
        },
        Header {
            flags: 2,
            mtime: 12345,
            ..Header::default()
        },
        Header {
            flags: 2 | 4 | 8 | 16,
            mtime: 77,
            extra: rng.bytes(300),
            name: b"name\xe9.warc".to_vec(),
            comment: b"c".to_vec(),
        },
        Header {
            flags: 1 | 8,
            name: b"t".to_vec(),
            ..Header::default()
        },
    ];
    for h in &hdrs {
        let len = 800 + rng.below(2500);
        let rec = warc_record(&mut rng, len);
        b = b.member(&rec, &[(D, 1 << 20)], h);
    }
    v.push(b.base);

    let mut content = warc_record(&mut rng, 3000);
    content.extend_from_slice(&rng.bytes(70_000));
    content.extend_from_slice(&warc_record(&mut rng, 40_000));
    let plan = [
        (S, 0),
        (F, 0),
        (D, 2000),
        (S, 0),
        (S, 1),
        (F, 1),
        (D, 1),
        (S, 65_535),
        (D, 0),
        (S, 4000),
        (D, 30_000),
        (F, 500),
        (S, 0),
        (D, 1 << 20),
    ];
    v.push(BaseBuilder::new("blocks").member(&content, &plan, &h0).base);

    let mut b = BaseBuilder::new("padded");
    let recs: Vec<Vec<u8>> = [2000, 5000, 1000]
        .iter()
        .map(|&n| warc_record(&mut rng, n))
        .collect();
    b = b.member(&recs[0], &[(D, 1 << 20)], &h0).raw(&[0]);
    b = b.member(&recs[1], &[(D, 1 << 20)], &h0).raw(&[0; 300]);
    b = b.member(&recs[2], &[(F, 1 << 20)], &h0).raw(&[0; 9]);
    v.push(b.base);

    let rec = warc_record(&mut rng, 30_000);
    v.push(
        BaseBuilder::new("one")
            .member(&rec, &[(D, 1 << 20)], &h0)
            .base,
    );

    let b = BaseBuilder::new("tiny")
        .member(b"x", &[(F, 1 << 20)], &h0)
        .member(b"", &[(F, 0)], &h0)
        .member(b"", &[(S, 0)], &h0)
        .member(b"ab", &[(S, 1), (D, 1)], &h0);
    v.push(b.base);

    let mut rep = b"abc".repeat(40_000);
    rep.resize(rep.len() + 50_000, 0);
    rep.extend_from_slice(&warc_record(&mut rng, 2000).repeat(30));
    v.push(
        BaseBuilder::new("repeat")
            .member(&rep, &[(D, 1 << 20)], &h0)
            .member(&rep[..20_000], &[(F, 1 << 20)], &h0)
            .base,
    );

    let r = rng.bytes(140_000);
    v.push(
        BaseBuilder::new("random")
            .member(&r, &[(S, 1 << 20)], &h0)
            .base,
    );
    v
}

#[derive(Clone, Copy, Debug)]
enum Src {
    Bytes,
    /// At most this many bytes per read.
    Short(usize),
    /// The read calls after the first `n` raise `OSError("boom")`.
    Fail(usize),
}

impl Src {
    fn spec(self) -> String {
        match self {
            Src::Bytes => "bytes".to_owned(),
            Src::Short(n) => format!("short:{n}"),
            Src::Fail(n) => format!("fail:{n}"),
        }
    }
}

struct Case {
    group: String,
    name: String,
    data: Vec<u8>,
    src: Src,
    /// `GzipFile.read(n)`; -1 for `read()`.
    n: i64,
}

const READ_SIZES: [i64; 7] = [16_384, 100, 8192, 65_536, 131_073, 1 << 20, -1];

fn flip(data: &[u8], at: usize, bit: usize) -> Vec<u8> {
    let mut d = data.to_vec();
    d[at] ^= 1 << (bit % 8);
    d
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    let mut add = |group: &str, name: String, data: Vec<u8>, src: Src, n: i64| {
        out.push(Case {
            group: group.to_owned(),
            name,
            data,
            src,
            n,
        });
    };
    let bases = bases();
    let mut rng = Rng(0x0dd_ba11);
    for b in &bases {
        let (name, data, len) = (b.name, &b.data, b.data.len());
        let small = len <= 12_000;
        // Valid inputs, every read size and source behaviour.
        for n in READ_SIZES {
            add("valid", format!("{name}/n{n}"), data.clone(), Src::Bytes, n);
        }
        if b.plain_len <= 20_000 {
            add("valid", format!("{name}/n1"), data.clone(), Src::Bytes, 1);
        }
        let shorts: &[usize] = if small {
            &[1, 7, 1000]
        } else {
            &[7, 1000, 50_000]
        };
        for &k in shorts {
            add(
                "short",
                format!("{name}/short{k}"),
                data.clone(),
                Src::Short(k),
                16_384,
            );
            add(
                "short",
                format!("{name}/short{k}/n8192"),
                data.clone(),
                Src::Short(k),
                8192,
            );
        }
        for k in [0, 1, 2, 3, 5, 8, 13] {
            add(
                "fail",
                format!("{name}/fail{k}"),
                data.clone(),
                Src::Fail(k),
                16_384,
            );
        }
        // Truncations: the first header, around every member boundary and trailer, and a stride.
        let mut cuts: Vec<usize> = (0..=40).collect();
        for s in &b.spans {
            cuts.extend(s.start.saturating_sub(12)..=s.start + 12);
            cuts.extend(s.deflate..s.deflate + 6);
            cuts.extend(s.trailer.saturating_sub(4)..=s.end);
        }
        cuts.extend((0..96).map(|i| i * len / 96));
        cuts.extend(len.saturating_sub(20)..len);
        cuts.retain(|&c| c < len);
        cuts.sort_unstable();
        cuts.dedup();
        let group = format!("trunc/{name}");
        for (i, &c) in cuts.iter().enumerate() {
            add(
                &group,
                format!("{name}/cut{c}"),
                data[..c].to_vec(),
                Src::Bytes,
                16_384,
            );
            if i % 4 == 0 {
                add(
                    &group,
                    format!("{name}/cut{c}/short"),
                    data[..c].to_vec(),
                    Src::Short(1000),
                    8192,
                );
            }
            if i % 9 == 0 {
                add(
                    &group,
                    format!("{name}/cut{c}/all"),
                    data[..c].to_vec(),
                    Src::Bytes,
                    -1,
                );
            }
        }
        // Damaged trailers and member headers.
        let group = format!("member/{name}");
        for (mi, s) in b.spans.iter().enumerate() {
            for (k, at, bit) in [
                ("crc0", s.trailer, 0),
                ("crc3", s.trailer + 3, 7),
                ("len0", s.trailer + 4, 0),
                ("len3", s.trailer + 7, 3),
            ] {
                add(
                    &group,
                    format!("{name}/m{mi}/{k}"),
                    flip(data, at, bit),
                    Src::Bytes,
                    16_384,
                );
            }
            for (k, at, val) in [
                ("magic0", s.start, 0x5f),
                ("magic1", s.start + 1, 0x8a),
                ("method", s.start + 2, 7),
            ] {
                let mut d = data.clone();
                d[at] = val;
                add(&group, format!("{name}/m{mi}/{k}"), d, Src::Bytes, 16_384);
            }
            for (k, bits) in [
                ("res", 0x20u8),
                ("hcrc", 2),
                ("extra", 4),
                ("fname", 8),
                ("comment", 16),
            ] {
                let mut d = data.clone();
                d[s.start + 3] ^= bits;
                add(
                    &group,
                    format!("{name}/m{mi}/flag-{k}"),
                    d,
                    Src::Bytes,
                    16_384,
                );
            }
        }
        // Trailing data after the last member.
        if ["warc", "one", "tiny", "padded"].contains(&name) {
            let one_member = member(b"tail", &[(F, 1 << 20)], &Header::default()).0;
            let garbage: Vec<(&str, Vec<u8>)> = vec![
                ("nul", b"\x00".to_vec()),
                ("one", b"\x01".to_vec()),
                ("id1", b"\x1f".to_vec()),
                ("magic", b"\x1f\x8b".to_vec()),
                ("method", b"\x1f\x8b\x08".to_vec()),
                (
                    "badmethod",
                    b"\x1f\x8b\x07\x00\x00\x00\x00\x00\x00\x03".to_vec(),
                ),
                ("zip", b"PK\x03\x04".to_vec()),
                ("random", rng.bytes(50)),
                ("zeros-garbage", [&[0u8; 20][..], b"garbage"].concat()),
                (
                    "no-trailer",
                    b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03\x03\x00".to_vec(),
                ),
                ("cut-member", one_member[..one_member.len() - 3].to_vec()),
                ("member", one_member.clone()),
                ("member-garbage", [&one_member[..], b"\x1f\x8bxx"].concat()),
                (
                    "hdr-random",
                    [
                        &b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\x03"[..],
                        &rng.bytes(5),
                    ]
                    .concat(),
                ),
            ];
            for (k, g) in garbage {
                let d = [&data[..], &g].concat();
                add(
                    "trailing",
                    format!("{name}/{k}"),
                    d.clone(),
                    Src::Bytes,
                    16_384,
                );
                add("trailing", format!("{name}/{k}/all"), d, Src::Bytes, -1);
            }
        }
        // Bit flips in the deflate data: every byte of each stream's first 24, then a stride.
        let per_member = match name {
            "big" => 16,
            "random" => 8,
            "warc" | "flags" | "tiny" => 24,
            _ => 48,
        };
        let group = format!("flip/{name}");
        for (mi, s) in b.spans.iter().enumerate() {
            let dlen = s.trailer - s.deflate;
            let mut at: Vec<usize> = (s.deflate..s.deflate + dlen.min(24)).collect();
            at.extend((0..per_member).map(|i| s.deflate + i * dlen / per_member));
            at.sort_unstable();
            at.dedup();
            for (i, &p) in at.iter().enumerate() {
                let d = flip(data, p, p * 3 + 1);
                if i % 5 == 0 {
                    add(
                        &group,
                        format!("{name}/m{mi}/flip{p}/n100"),
                        d.clone(),
                        Src::Bytes,
                        100,
                    );
                }
                add(
                    &group,
                    format!("{name}/m{mi}/flip{p}"),
                    d,
                    Src::Bytes,
                    16_384,
                );
            }
        }
    }
    for (k, d) in [
        ("empty", &b""[..]),
        ("nul", b"\x00"),
        ("id1", b"\x1f"),
        ("text", b"WARC/1.0\r\n"),
        ("zeros", &[0u8; 30]),
    ] {
        add("misc", format!("misc/{k}"), d.to_vec(), Src::Bytes, 16_384);
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Running a case
// ---------------------------------------------------------------------------------------------

struct TestSource<'a> {
    data: &'a [u8],
    pos: usize,
    src: Src,
    calls: usize,
}

impl Source for TestSource<'_> {
    fn read(&mut self, n: usize, out: &mut Vec<u8>) -> Result<(), Error> {
        self.calls += 1;
        let k = match self.src {
            Src::Fail(after) if self.calls > after => {
                return Err(Error::Source(Box::new("boom".to_owned())));
            }
            Src::Short(m) => n.min(m),
            _ => n,
        };
        let k = k.min(self.data.len() - self.pos);
        out.extend_from_slice(&self.data[self.pos..self.pos + k]);
        self.pos += k;
        Ok(())
    }
}

fn exc_name(e: &Error) -> (&'static str, String) {
    match e {
        Error::Py(PyExc::EofError, m) => ("EOFError", m.clone()),
        Error::Py(PyExc::BadGzipFile, m) => ("BadGzipFile", m.clone()),
        Error::Py(PyExc::ZlibError, m) => ("error", m.clone()),
        Error::Py(other, m) => panic!("unexpected {other:?}: {m}"),
        Error::Source(p) => (
            "OSError",
            p.downcast_ref::<String>().cloned().unwrap_or_default(),
        ),
    }
}

const MAX_CALLS: usize = 2_000_000;

/// The trace digest of one case, as trace.py computes it.
fn run_gzip(case: &Case, params: GzipParams, log: Option<&mut Vec<String>>) -> ([u8; 20], Stats) {
    let mut src = TestSource {
        data: &case.data,
        pos: 0,
        src: case.src,
        calls: 0,
    };
    let mut f = GzipFile::new(params);
    let mut h = Sha1::new();
    let mut stats = Stats::default();
    let mut log = log;
    let (mut excs, mut empties) = (0, 0);
    let mut buf = Vec::new();
    for _ in 0..MAX_CALLS {
        buf.clear();
        stats.calls += 1;
        let r = if case.n < 0 {
            f.read_all(&mut src, &mut buf)
        } else {
            f.read(&mut src, usize::try_from(case.n).unwrap(), &mut buf)
        };
        let stop = match r {
            Ok(()) => {
                h.update(b"O");
                h.update((buf.len() as u64).to_le_bytes());
                h.update(&buf);
                if let Some(l) = log.as_deref_mut() {
                    l.push(format!("ok {}", buf.len()));
                }
                stats.bytes += buf.len() as u64;
                if buf.is_empty() {
                    empties += 1;
                }
                empties >= 2
            }
            Err(e) => {
                let (t, m) = exc_name(&e);
                h.update(b"E");
                h.update(t.as_bytes());
                h.update(b"\0");
                h.update(m.as_bytes());
                h.update(b"\0");
                if let Some(l) = log.as_deref_mut() {
                    l.push(format!("{t}: {m}"));
                }
                *stats.exc.entry(t.to_owned()).or_default() += 1;
                excs += 1;
                excs >= 3
            }
        };
        h.update(b"T");
        h.update(f.tell().to_le_bytes());
        if stop {
            break;
        }
    }
    match f.mtime() {
        Some(m) => {
            h.update(b"M");
            h.update(u64::from(m).to_le_bytes());
        }
        None => h.update(b"N"),
    }
    (h.finalize().into(), stats)
}

#[derive(Default, Debug, PartialEq, Eq)]
struct Stats {
    calls: u64,
    bytes: u64,
    exc: BTreeMap<String, u64>,
}

/// The fixtures' short digest: the first 8 bytes as a decimal number (no letters for typos to read).
fn short(d: &[u8]) -> String {
    u64::from_be_bytes(d[..8].try_into().unwrap()).to_string()
}

fn hex(b: &[u8]) -> String {
    b.iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

/// Per group: (cases, digest); the totals; per case: the digest.
type Digests = (BTreeMap<String, (usize, String)>, Stats, Vec<String>);

fn group_digests(cases: &[Case], params: GzipParams) -> Digests {
    let mut groups: BTreeMap<String, (usize, Sha1)> = BTreeMap::new();
    let mut total = Stats::default();
    let mut per_case = Vec::new();
    for c in cases {
        let (d, s) = run_gzip(c, params, None);
        let g = groups
            .entry(c.group.clone())
            .or_insert_with(|| (0, Sha1::new()));
        g.0 += 1;
        g.1.update(d);
        per_case.push(hex(&d));
        total.calls += s.calls;
        total.bytes += s.bytes;
        for (k, v) in s.exc {
            *total.exc.entry(k).or_default() += v;
        }
    }
    let groups = groups
        .into_iter()
        .map(|(k, (n, h))| (k, (n, short(&h.finalize()))))
        .collect();
    (groups, total, per_case)
}

fn family(version: &str) -> GzipParams {
    let minor: u32 = version
        .trim_end_matches('t')
        .split('.')
        .nth(1)
        .unwrap()
        .parse()
        .unwrap();
    GzipParams::for_python(minor).unwrap()
}

fn load(name: &str) -> serde_json::Value {
    let text = std::fs::read_to_string(format!("{DATA_DIR}/{name}")).unwrap();
    serde_json::from_str(&text).unwrap()
}

#[test]
fn gzipfile_matches_cpython() {
    let fixture = load("gzip.json");
    let cases = cases();
    assert_eq!(
        usize::try_from(fixture["cases"].as_u64().unwrap()).unwrap(),
        cases.len(),
        "case list changed: regenerate gzip.json"
    );
    let pythons = fixture["pythons"].as_object().unwrap();
    let mut params: Vec<GzipParams> = Vec::new();
    for p in pythons.keys().map(|v| family(v)) {
        if !params.contains(&p) {
            params.push(p);
        }
    }
    // One thread per parameter family (the debug build is slow).
    let by_family: Vec<(GzipParams, Digests)> = std::thread::scope(|s| {
        let cases = &cases;
        let handles: Vec<_> = params
            .iter()
            .map(|&p| (p, s.spawn(move || group_digests(cases, p))))
            .collect();
        handles
            .into_iter()
            .map(|(p, h)| (p, h.join().unwrap()))
            .collect()
    });
    let mut failures = Vec::new();
    for (version, want) in pythons {
        let params = family(version);
        let (groups, stats, per_case) = &by_family.iter().find(|(p, _)| *p == params).unwrap().1;
        for (g, d) in want["groups"].as_object().unwrap() {
            let got = groups.get(g).map_or("missing", |(_, h)| h.as_str());
            if got != d.as_str().unwrap() {
                failures.push(format!("{version}: group {g}: {got} != {d}"));
            }
        }
        let want_calls = want["calls"].as_u64().unwrap();
        if stats.calls != want_calls {
            failures.push(format!(
                "{version}: {} read calls, CPython made {want_calls}",
                stats.calls
            ));
        }
        // With GZIP_TRACE_DETAIL=<dir> (trace.py's per-case digests), name the differing cases.
        if let Ok(dir) = std::env::var("GZIP_TRACE_DETAIL") {
            let detail: serde_json::Value = serde_json::from_str(
                &std::fs::read_to_string(format!("{dir}/detail-{version}.json")).unwrap(),
            )
            .unwrap();
            let mut shown = 0;
            for (i, c) in cases.iter().enumerate() {
                let want = detail["cases"][i].as_str().unwrap();
                if per_case[i] != want && shown < 10 {
                    shown += 1;
                    let mut log = Vec::new();
                    run_gzip(c, params, Some(&mut log));
                    let tail: Vec<&String> = log.iter().rev().take(6).collect();
                    failures.push(format!(
                        "{version}: case {i} {} differs; ours ends {tail:?}",
                        c.name
                    ));
                }
            }
        }
        eprintln!(
            "{version}: {} cases, {} read calls, {} bytes, {:?}",
            cases.len(),
            stats.calls,
            stats.bytes,
            stats.exc
        );
    }
    assert!(
        failures.is_empty(),
        "{} differences:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Write the cases for trace.py: `index.json` (group, name, source, n, offset, length) and
/// `inputs.bin`.
#[test]
#[ignore = "generates the inputs trace.py traces (GZIP_CASES_OUT=<dir>)"]
fn dump_cases() {
    let dir = std::env::var("GZIP_CASES_OUT").expect("GZIP_CASES_OUT");
    std::fs::create_dir_all(&dir).unwrap();
    let mut blob = Vec::new();
    let mut index = Vec::new();
    for c in cases() {
        index.push(serde_json::json!({
            "group": c.group, "name": c.name, "src": c.src.spec(), "n": c.n,
            "off": blob.len(), "len": c.data.len(),
        }));
        blob.extend_from_slice(&c.data);
    }
    std::fs::write(format!("{dir}/inputs.bin"), &blob).unwrap();
    std::fs::write(
        format!("{dir}/index.json"),
        serde_json::to_string(&index).unwrap(),
    )
    .unwrap();
    eprintln!("{} cases, {} input bytes", index.len(), blob.len());
}

#[test]
fn deflate_encoder_round_trips() {
    // The generator's streams are valid deflate (zlib-rs agrees), so damage is what we add.
    for b in bases() {
        let mut src = TestSource {
            data: &b.data,
            pos: 0,
            src: Src::Bytes,
            calls: 0,
        };
        let mut f = GzipFile::new(GzipParams::PY312);
        let mut out = Vec::new();
        f.read_all(&mut src, &mut out)
            .unwrap_or_else(|e| panic!("{}: {e:?}", b.name));
        assert_eq!(out.len(), b.plain_len, "{}", b.name);
    }
}
