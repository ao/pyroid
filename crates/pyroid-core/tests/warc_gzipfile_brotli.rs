//! Differential test of `warc::brotli` against brotlicffi 1.2.1.0's `Decompressor`, as warcio
//! 1.8.2 uses it (`decompress(data)` per block read, `flush()` at the end).
//!
//! `warc_gzipfile_data/brotli.json` holds streams made by brotlicffi's encoder and the digests
//! of brotlicffi's traces, per group of cases. The cases (whole streams fed in chunks of 1 to
//! 16384 bytes, truncations, bit flips, trailing data, garbage) are derived from the streams
//! here and in `warc_gzipfile_data/trace.py` by the same rules. A trace is each call's output
//! (length and bytes) or `str(e)` of its `brotlicffi.error`, then `flush()` and `is_finished()`.
//!
//! `known` lists cases where the port cannot give libbrotli's result. They are left out of the
//! group digests and must keep differing (a fixed port shows up). With brotli-decompressor 6.3
//! it is empty: every case matches. (5.0.3 differed in 41 cases, where a damaged command ran
//! past a metablock that was not the last one: libbrotli stops with
//! `_ERROR_FORMAT_BLOCK_LENGTH_2` and 5.0.3 decoded on; 6.2.0 added libbrotli's check.)
//!
//! Regenerate with `python -I tests/warc_gzipfile_data/trace.py brotli <out.json> [detail dir]
//! [known names file]`.
#![allow(clippy::cast_possible_truncation, clippy::print_stderr)] // base64 bit packing; the run reports its counts

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use pyroid_core::warc::Error;
use pyroid_core::warc::brotli::Decompressor;
use sha1::{Digest, Sha1};

const DATA_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/warc_gzipfile_data");

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

fn b64decode(s: &str) -> Vec<u8> {
    let val = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => panic!("bad base64"),
    };
    let mut out = Vec::new();
    let (mut acc, mut n) = (0u32, 0);
    for &c in s.as_bytes().iter().filter(|&&c| c != b'=') {
        acc = acc << 6 | u32::from(val(c));
        n += 6;
        if n >= 8 {
            n -= 8;
            out.push((acc >> n) as u8);
        }
    }
    out
}

fn flip(data: &[u8], at: usize, bit: usize) -> Vec<u8> {
    let mut d = data.to_vec();
    d[at] ^= 1 << (bit % 8);
    d
}

/// The cases, by the rules trace.py's `brotli_cases` follows: (name, input, chunk size; 0 = one
/// call with everything).
fn brotli_cases(bases: &[Vec<u8>]) -> Vec<(String, Vec<u8>, usize)> {
    let mut out = Vec::new();
    for (bi, b) in bases.iter().enumerate() {
        let len = b.len();
        for cs in [0, 1, 7, 100, 16_384] {
            if cs == 1 && len > 4096 {
                continue;
            }
            out.push((format!("b{bi}/valid/{cs}"), b.clone(), cs));
        }
        let step = (len / 150).max(1);
        for i in (0..len).step_by(step) {
            out.push((format!("b{bi}/trunc/{i}"), b[..i].to_vec(), 0));
        }
        for i in (0..len).step_by(step) {
            let m = flip(b, i, i * 5 + 3);
            if i % (step * 4) == 0 {
                out.push((format!("b{bi}/flip/{i}/100"), m.clone(), 100));
            }
            out.push((format!("b{bi}/flip/{i}"), m, 0));
        }
        for (ti, t) in [b"\x00".to_vec(), b"x".to_vec(), b.clone(), vec![0u8; 100]]
            .iter()
            .enumerate()
        {
            for cs in [0, 100] {
                out.push((format!("b{bi}/trail{ti}/{cs}"), [&b[..], t].concat(), cs));
            }
        }
    }
    let fixed: Vec<u8> = (0u8..=252).step_by(4).collect();
    for (gi, g) in [Vec::new(), vec![0], b"hello world".to_vec(), fixed]
        .into_iter()
        .enumerate()
    {
        out.push((format!("garbage{gi}"), g, 0));
    }
    out
}

fn run_brotli(data: &[u8], chunk: usize, log: Option<&mut Vec<String>>) -> [u8; 20] {
    let mut d = Decompressor::new();
    let mut h = Sha1::new();
    let mut log = log;
    let chunks: Vec<&[u8]> = if chunk == 0 || data.is_empty() {
        vec![data]
    } else {
        data.chunks(chunk).collect()
    };
    let mut out = Vec::new();
    for c in chunks {
        out.clear();
        match d.decompress(c, &mut out) {
            Ok(()) => {
                h.update(b"O");
                h.update((out.len() as u64).to_le_bytes());
                h.update(&out);
                if let Some(l) = log.as_deref_mut() {
                    l.push(format!("ok {}", out.len()));
                }
            }
            Err(Error::Py(_, m)) => {
                h.update(b"E");
                h.update(m.as_bytes());
                h.update(b"\0");
                if let Some(l) = log.as_deref_mut() {
                    l.push(m);
                }
            }
            Err(e) => panic!("unexpected {e:?}"),
        }
    }
    h.update(b"F");
    h.update((d.flush().len() as u64).to_le_bytes());
    h.update(if d.is_finished() { b"I1" } else { b"I0" });
    h.finalize().into()
}

#[test]
fn brotli_matches_brotlicffi() {
    let text = std::fs::read_to_string(format!("{DATA_DIR}/brotli.json")).unwrap();
    let fixture: serde_json::Value = serde_json::from_str(&text).unwrap();
    let bases: Vec<Vec<u8>> = fixture["bases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| b64decode(b.as_str().unwrap()))
        .collect();
    let cases = brotli_cases(&bases);
    assert_eq!(
        usize::try_from(fixture["cases"].as_u64().unwrap()).unwrap(),
        cases.len(),
        "case list changed: regenerate brotli.json"
    );
    let known: BTreeMap<&str, &str> = fixture["known"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| (k.as_str(), v["digest"].as_str().unwrap()))
        .collect();
    // With GZIP_TRACE_DETAIL=<dir> (trace.py's per-case digests and logs), name the differing cases.
    let detail: Option<serde_json::Value> = std::env::var("GZIP_TRACE_DETAIL").ok().map(|dir| {
        serde_json::from_str(&std::fs::read_to_string(format!("{dir}/detail-brotli.json")).unwrap())
            .unwrap()
    });
    let mut groups: BTreeMap<String, Sha1> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut seen = BTreeSet::new();
    for (i, (name, data, chunk)) in cases.iter().enumerate() {
        let d = run_brotli(data, *chunk, None);
        if let Some(want) = known.get(name.as_str()) {
            seen.insert(name.as_str());
            if short(&d) == *want {
                failures.push(format!(
                    "{name} now matches brotlicffi: drop it from `known`"
                ));
            }
            continue;
        }
        let group = name.split('/').take(2).collect::<Vec<_>>().join("/");
        groups.entry(group).or_default().update(d);
        if let Some(det) = &detail {
            if det["cases"][i].as_str().unwrap() != hex(&d) && failures.len() < 50 {
                let mut log = Vec::new();
                run_brotli(data, *chunk, Some(&mut log));
                failures.push(format!(
                    "case {i} {name}: ours {log:?}, brotlicffi {}",
                    det["logs"][i]
                ));
            }
        }
    }
    assert_eq!(
        seen.len(),
        known.len(),
        "`known` names a case that does not exist"
    );
    for (g, want) in fixture["groups"].as_object().unwrap() {
        let got = groups
            .remove(g)
            .map_or_else(|| "missing".to_owned(), |h| short(&h.finalize()));
        if got != want.as_str().unwrap() {
            failures.push(format!("group {g}: {got} != {want}"));
        }
    }
    assert!(
        groups.is_empty(),
        "groups missing from brotli.json: {:?}",
        groups.keys().collect::<Vec<_>>()
    );
    eprintln!(
        "brotli: {} cases, {} match brotlicffi, {} known differences",
        cases.len(),
        cases.len() - known.len(),
        known.len()
    );
    assert!(
        failures.is_empty(),
        "{} differences:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn after_the_end() {
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(format!("{DATA_DIR}/brotli.json")).unwrap())
            .unwrap();
    let stream = b64decode(fixture["bases"][0].as_str().unwrap());
    let mut d = Decompressor::new();
    let mut out = Vec::new();
    d.decompress(&stream, &mut out).unwrap();
    assert!(d.is_finished() && d.can_accept_more_data());
    let n = out.len();
    let err = d.decompress(b"x", &mut out).unwrap_err();
    assert_eq!(
        err.py().unwrap().1,
        "Decompression error: trailing data after stream."
    );
    assert_eq!(out.len(), n);
    d.decompress(b"", &mut out).unwrap();
    assert_eq!(out.len(), n);
    assert_eq!(pyroid_core::warc::brotli::decompress(&stream).unwrap(), out);
    assert_eq!(
        pyroid_core::warc::brotli::decompress(&stream[..stream.len() - 1])
            .unwrap_err()
            .py()
            .unwrap()
            .1,
        "Decompression error: incomplete compressed stream."
    );
}
