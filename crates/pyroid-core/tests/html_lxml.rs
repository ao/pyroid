//! `pyroid_core::html` against lxml 6.1.3 on real pages: for each document, the tree
//! `trafilatura.utils.load_html` builds with `lxml.html.fromstring`, compared byte for byte in
//! the `.tree` line format (see `Document::dump`).
//!
//! Needs the sample sets (`raw.bin`, `meta.jsonl`, `lxml.tree`, optionally `repaired.bin`), so
//! it is ignored by default:
//!
//! ```sh
//! PYROID_HTMLX_DATA=/target/cc cargo test -p pyroid-core --release --test html_lxml -- --ignored --nocapture
//! ```
//!
//! `PYROID_HTMLX_SETS` picks the set directories under `PYROID_HTMLX_DATA` (default
//! `dev,eval`); `tests/html_data/lxml_sets.py` writes more sets in the same format (fuzzed and
//! mutated pages; `lxml_sets.py hashes` writes a set whose `meta.jsonl` holds the sha1 of each
//! tree instead of an `lxml.tree`). `load_html` repairs the string first
//! (`repair_faulty_html`); where a set has no `repaired.bin`, this test repeats that step, and
//! where it has one, the test checks its own repair against it.
#![allow(clippy::print_stderr)] // reports of the comparisons

use std::fmt::Write as _;
use std::io::{BufRead, Read};

use pyroid_core::html::lxml::{LxmlError, fromstring_bytes, fromstring_wtf8};
use pyroid_core::html::tree::Document;
use sha1::{Digest, Sha1};

fn read_bin(path: &str) -> std::io::Result<Vec<Vec<u8>>> {
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
    let mut u = [0u8; 4];
    f.read_exact(&mut u)?;
    let n = u32::from_le_bytes(u) as usize;
    let mut docs = Vec::with_capacity(n);
    for _ in 0..n {
        f.read_exact(&mut u)?;
        let mut b = vec![0u8; u32::from_le_bytes(u) as usize];
        f.read_exact(&mut b)?;
        docs.push(b);
    }
    Ok(docs)
}

/// Code points of surrogate-tolerant UTF-8.
fn code_points(s: &[u8]) -> Vec<u32> {
    let mut out = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let c = u32::from(s[i]);
        let n = match c {
            0..0x80 => 1,
            0x80..0xE0 => 2,
            0xE0..0xF0 => 3,
            _ => 4,
        }
        .min(s.len() - i);
        let cont = |k: usize| u32::from(s[i + k]) & 0x3F;
        out.push(match n {
            1 => c,
            2 => ((c & 0x1F) << 6) | cont(1),
            3 => ((c & 0x0F) << 12) | (cont(1) << 6) | cont(2),
            _ => ((c & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
        });
        i += n;
    }
    out
}

fn push_cp(out: &mut Vec<u8>, c: u32) {
    // Surrogates are encoded on their own, as Python's surrogatepass does.
    #[allow(clippy::cast_possible_truncation)]
    match c {
        0..0x80 => out.push(c as u8),
        0x80..0x800 => out.extend([0xC0 | (c >> 6) as u8, 0x80 | (c & 0x3F) as u8]),
        0x800..0x1_0000 => out.extend([
            0xE0 | (c >> 12) as u8,
            0x80 | ((c >> 6) & 0x3F) as u8,
            0x80 | (c & 0x3F) as u8,
        ]),
        _ => out.extend([
            0xF0 | (c >> 18) as u8,
            0x80 | ((c >> 12) & 0x3F) as u8,
            0x80 | ((c >> 6) & 0x3F) as u8,
            0x80 | (c & 0x3F) as u8,
        ]),
    }
}

/// `trafilatura.utils.repair_faulty_html` (trafilatura 2.3.1) on a str given as
/// surrogate-tolerant UTF-8, with `beginning = htmlstring[:50].lower()`. Only here to rebuild
/// the strings `load_html` passes to lxml where a set has no `repaired.bin`.
mod repair {
    use super::{code_points, push_cp};

    fn py_isspace(c: u32) -> bool {
        matches!(c, 0x09..=0x0D | 0x1C..=0x20 | 0x85 | 0xA0 | 0x1680 | 0x2000..=0x200A
            | 0x2028 | 0x2029 | 0x202F | 0x205F | 0x3000)
    }

    fn lower(c: u32) -> u32 {
        u8::try_from(c).map_or(c, |b| u32::from(b.to_ascii_lowercase()))
    }

    /// `DOCTYPE_TAG = re.compile("^< ?! ?DOCTYPE[^>]*/[^<>]*>", re.I)`: the end of the match.
    fn doctype_tag(line: &[u32]) -> Option<usize> {
        let mut i = 0;
        let at = |i: usize, c: u8| line.get(i) == Some(&u32::from(c));
        if !at(i, b'<') {
            return None;
        }
        i += 1;
        if at(i, b' ') {
            i += 1;
        }
        if !at(i, b'!') {
            return None;
        }
        i += 1;
        if at(i, b' ') {
            i += 1;
        }
        if !b"doctype"
            .iter()
            .enumerate()
            .all(|(k, &b)| line.get(i + k).map(|&c| lower(c)) == Some(u32::from(b)))
        {
            return None;
        }
        i += 7;
        // `[^>]*` runs to the first '>' (g); the last '/' before g must reach '>' without a '<'.
        let g = line[i..].iter().position(|&c| c == u32::from(b'>'))? + i;
        let slash = line[i..g].iter().rposition(|&c| c == u32::from(b'/'))? + i;
        if line[slash + 1..g].contains(&u32::from(b'<')) {
            return None;
        }
        Some(g + 1)
    }

    /// `FAULTY_HTML.sub(r"\1>", s, count=1)` with `FAULTY_HTML = (<html.*?)\s*/>`, re.I.
    fn faulty_html_sub(cps: &[u32]) -> Option<Vec<u32>> {
        let n = cps.len();
        for st in 0..n {
            if st + 5 > n
                || !b"<html"
                    .iter()
                    .enumerate()
                    .all(|(k, &b)| lower(cps[st + k]) == u32::from(b))
            {
                continue;
            }
            let mut k = st + 5;
            loop {
                let mut q = k;
                while q < n && py_isspace(cps[q]) {
                    q += 1;
                }
                if q + 1 < n && cps[q] == u32::from(b'/') && cps[q + 1] == u32::from(b'>') {
                    let mut out = cps[..k].to_vec();
                    out.push(u32::from(b'>'));
                    out.extend_from_slice(&cps[q + 2..]);
                    return Some(out);
                }
                if k >= n || cps[k] == u32::from(b'\n') {
                    break;
                }
                k += 1;
            }
        }
        None
    }

    /// Python's `str.splitlines()` line ends (after INVALID_XML_CHARS removed \v, \f, \x1c-\x1e).
    fn lines(cps: &[u32]) -> Vec<&[u32]> {
        let mut out = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < cps.len() {
            let c = cps[i];
            if matches!(
                c,
                0x0A | 0x0D | 0x0B | 0x0C | 0x1C..=0x1E | 0x85 | 0x2028 | 0x2029
            ) {
                out.push(&cps[start..i]);
                if c == 0x0D && cps.get(i + 1) == Some(&0x0A) {
                    i += 1;
                }
                start = i + 1;
            }
            i += 1;
        }
        if start < cps.len() {
            out.push(&cps[start..]);
        }
        out
    }

    pub fn repair_faulty_html(s: &[u8]) -> Vec<u8> {
        let orig = code_points(s);
        let beginning: Vec<u32> = orig.iter().take(50).map(|&c| lower(c)).collect();
        let has = |hay: &[u32], w: &[u8]| {
            hay.windows(w.len())
                .any(|x| x.iter().zip(w).all(|(&a, &b)| a == u32::from(b)))
        };
        // INVALID_XML_CHARS = [\x00-\x08\x0b\x0c\x0e-\x1f￾￿]
        let mut cps: Vec<u32> = orig
            .into_iter()
            .filter(|&c| !matches!(c, 0..=8 | 0x0B | 0x0C | 0x0E..=0x1F | 0xFFFE | 0xFFFF))
            .collect();
        if has(&beginning, b"doctype") {
            let nl = cps.iter().position(|&c| c == 0x0A);
            let (first, rest): (&[u32], &[u32]) = match nl {
                Some(p) => (&cps[..p], &cps[p + 1..]),
                None => (&cps[..], &[]),
            };
            let cut = doctype_tag(first).unwrap_or(0);
            let mut out = first[cut..].to_vec();
            out.push(0x0A);
            out.extend_from_slice(rest);
            cps = out;
        }
        let head = &cps[..cps.len().min(4096)];
        let fix = lines(head)
            .into_iter()
            .take(4)
            .any(|l| has(l, b"<html") && l.ends_with(&[u32::from(b'/'), u32::from(b'>')]));
        if fix {
            if let Some(out) = faulty_html_sub(&cps) {
                cps = out;
            }
        }
        let mut out = Vec::with_capacity(s.len());
        for c in cps {
            push_cp(&mut out, c);
        }
        out
    }
}

/// `trafilatura.utils.load_html` for a str, from the repaired string `s` on; `raw` is the
/// original string (its first 50 characters decide `is_dubious_html`). Also returns which lxml
/// call built the tree.
fn load_html(raw: &[u8], s: &[u8]) -> (Option<Document>, &'static str) {
    let head: Vec<u8> = code_points(raw)
        .into_iter()
        .take(50)
        .map(|c| u8::try_from(c).map_or(0, |b| b.to_ascii_lowercase()))
        .collect();
    let check_flag = !head.windows(4).any(|w| w == b"html");
    let len = |t: &Option<Document>| t.as_ref().map_or(0, |d| d.len(d.root));
    let (mut tree, mut path, fallback) = match fromstring_wtf8(s) {
        Ok(d) => (Some(d), "str", false),
        Err(LxmlError::ValueError(_)) => (fromstring_bytes(s).ok(), "bytes", true),
        Err(_) => (None, "str", false),
    };
    if (tree.is_none() || len(&tree) < 1) && !fallback {
        tree = fromstring_bytes(s).ok();
        path = "bytes";
    }
    if tree.is_some() && check_flag && len(&tree) < 2 {
        tree = None;
    }
    (tree, path)
}

fn check_set(dir: &str) -> usize {
    let raw = read_bin(&format!("{dir}/raw.bin")).expect("raw.bin");
    let repaired = read_bin(&format!("{dir}/repaired.bin")).ok();
    let meta: Vec<serde_json::Value> = std::fs::read_to_string(format!("{dir}/meta.jsonl"))
        .expect("meta.jsonl")
        .lines()
        .map(|l| serde_json::from_str(l).expect("meta line"))
        .collect();
    assert_eq!(meta.len(), raw.len());
    // The reference trees: lxml.tree, or (sets written by `lxml_sets.py hashes`) the sha1 of
    // each tree's lines in meta.jsonl.
    let mut lines = std::fs::File::open(format!("{dir}/lxml.tree"))
        .ok()
        .map(|f| std::io::BufReader::with_capacity(1 << 20, f).lines());
    let mut pending: Option<String> = None;
    let (mut same, mut diff, mut path_diff, mut repair_diff) = (0usize, 0usize, 0usize, 0usize);
    for (i, m) in meta.iter().enumerate() {
        let index = m["index"].as_u64().expect("index");
        let mut reference = String::new();
        if let Some(lines) = lines.as_mut() {
            // The document's ["D", index] line and everything up to the next one.
            reference = pending
                .take()
                .unwrap_or_else(|| lines.next().expect("D line").expect("read"));
            reference.push('\n');
            for l in lines.by_ref() {
                let l = l.expect("read");
                if l.starts_with("[\"D\"") {
                    pending = Some(l);
                    break;
                }
                reference.push_str(&l);
                reference.push('\n');
            }
        }
        let ours_repaired = repair::repair_faulty_html(&raw[i]);
        let s = match &repaired {
            Some(r) => {
                if r[i] != ours_repaired {
                    repair_diff += 1;
                    eprintln!("{dir}: doc {index}: repair_faulty_html differs from repaired.bin");
                }
                &r[i]
            }
            None => &ours_repaired,
        };
        let (doc, path) = load_html(&raw[i], s);
        if m["path"].as_str() != Some(path) {
            path_diff += 1;
            eprintln!(
                "{dir}: doc {index}: lxml call {path}, reference {}",
                m["path"]
            );
        }
        let mut tree = String::new();
        if let Some(d) = &doc {
            d.dump(d.root, true, &mut tree);
        }
        let identical = if lines.is_some() {
            format!("[\"D\", {index}]\n{tree}") == reference
        } else {
            let ours = doc.as_ref().map(|_| {
                Sha1::digest(tree.as_bytes())
                    .iter()
                    .fold(String::new(), |mut s, b| {
                        let _ = write!(s, "{b:02x}");
                        s
                    })
            });
            m["sha1"].as_str() == ours.as_deref()
        };
        if identical {
            same += 1;
        } else {
            diff += 1;
            if diff <= 20 {
                eprintln!("{dir}: doc {index}: tree differs");
            }
        }
    }
    eprintln!(
        "{dir}: {same} of {} trees identical, {diff} differ; lxml call differs on {path_diff}; \
         repair differs on {repair_diff}",
        meta.len()
    );
    diff + path_diff + repair_diff
}

#[test]
#[ignore = "needs the sample sets; set PYROID_HTMLX_DATA"]
fn trees_match_lxml() {
    let root = std::env::var("PYROID_HTMLX_DATA").unwrap_or_else(|_| "/target/cc".to_owned());
    let sets = std::env::var("PYROID_HTMLX_SETS").unwrap_or_else(|_| "dev,eval".to_owned());
    let mut bad = 0;
    for set in sets.split(',').filter(|s| !s.is_empty()) {
        bad += check_set(&format!("{root}/{set}"));
    }
    assert_eq!(bad, 0, "differences from lxml (see stderr)");
}
