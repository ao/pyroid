//! The `html_parse` fuzz target's checks (`html_data/fuzz_check.rs`) on seeded random input:
//! byte noise, HTML token soup, and random edits of the `html_data/fromstring.jsonl` inputs.
//! It stands in for cargo-fuzz where libFuzzer cannot be built.
//!
//! `PYROID_HTML_FUZZ_ITERS` (default 3000), `PYROID_HTML_FUZZ_SEED` (default 1) and
//! `PYROID_HTML_FUZZ_LARGE=1` (also documents over 10 MB, which take the buffer emulation)
//! set the run, e.g.
//!
//! ```sh
//! PYROID_HTML_FUZZ_ITERS=1000000 PYROID_HTML_FUZZ_LARGE=1 \
//!     cargo test -p pyroid-core --release --test html_robust -- --nocapture
//! ```
#![allow(clippy::print_stderr)] // the run's summary

include!("html_data/fuzz_check.rs");

/// splitmix64.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        // `n` is small; the modulo bias does not matter here.
        #[allow(clippy::cast_possible_truncation)]
        let r = (self.next() % n as u64) as usize;
        r
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

const TOKENS: &[&[u8]] = &[
    b"<html>",
    b"</html>",
    b"<head>",
    b"</head>",
    b"<body>",
    b"</body>",
    b"<body/>",
    b"<html/>",
    b"<p>",
    b"</p>",
    b"<div class=a>",
    b"</div>",
    b"<b>",
    b"</b>",
    b"<table>",
    b"<tr>",
    b"<td>",
    b"</table>",
    b"<li>",
    b"<ul>",
    b"<select><option>",
    b"<br>",
    b"<br/>",
    b"<img src=x>",
    b"<script>",
    b"</script>",
    b"<style>",
    b"</style>",
    b"<textarea>",
    b"</textarea>",
    b"<title>",
    b"</title>",
    b"<plaintext>",
    b"<xmp>",
    b"<iframe>",
    b"<noembed>",
    b"<!--",
    b"-->",
    b"--!>",
    b"<!-- c -->",
    b"<!x>",
    b"<?php ?>",
    b"<!DOCTYPE html>",
    b"<!doctype x PUBLIC \"a\" \"b\">",
    b"<!DOCTYPE",
    b"<![CDATA[",
    b"]]>",
    b"</",
    b"<",
    b">",
    b"/>",
    b"=",
    b"\"",
    b"'",
    b" ",
    b"\t",
    b"\n",
    b"\r",
    b"\r\n",
    b"\x00",
    b"\x0c",
    b"&",
    b"&amp;",
    b"&amp",
    b"&lt",
    b"&notin;",
    b"&notit;",
    b"&#",
    b"&#x",
    b"&#65;",
    b"&#x1F600;",
    b"&#0;",
    b"&#128;",
    b"&#xD800;",
    b"&#99999999;",
    b"&#1;",
    b"&#xFFFE;",
    b"&CounterClockwise",
    b" a=b",
    b" checked",
    b" x='y'",
    b" CLASS=\"z\"",
    b" =",
    b" /",
    b"a",
    b"text",
    b"\xc3\xa9",
    b"\xe2\x82\xac",
    b"\xf0\x9f\x98\x80",
    b"\xed\xa0\x80",
    b"\xed\xb0\x80",
    b"\xff",
    b"\xc3",
    b"\xef\xbb\xbf",
    b"\xe0\x80\x80",
    b"<?xml version='1.0' encoding='utf-8'?>",
    b"<fb:like>",
    b"<O:P>",
    b"<DIV>",
    b"</DIV >",
    b"</p x=1>",
    b"</>",
    b"</3>",
    b"<a\x00b>",
];

fn soup(r: &mut Rng, max: usize) -> Vec<u8> {
    let mut out = Vec::new();
    for _ in 0..=r.below(max) {
        let t = *r.pick(TOKENS);
        out.extend_from_slice(t);
    }
    out
}

fn noise(r: &mut Rng) -> Vec<u8> {
    const SPECIAL: &[u8] = b"<>/!?-=&#;\"' \t\r\n\x00abcxyzDIVP";
    (0..r.below(300))
        .map(|_| {
            if r.below(4) == 0 {
                // Any byte.
                #[allow(clippy::cast_possible_truncation)]
                let b = r.next() as u8;
                b
            } else {
                *r.pick(SPECIAL)
            }
        })
        .collect()
}

#[allow(clippy::many_single_char_names)] // span ends i..j of the n bytes of s
fn mutate(r: &mut Rng, mut s: Vec<u8>) -> Vec<u8> {
    for _ in 0..=r.below(4) {
        let n = s.len();
        let (mut i, mut j) = (r.below(n + 1), r.below(n + 1));
        if i > j {
            std::mem::swap(&mut i, &mut j);
        }
        match r.below(6) {
            0 => {
                s.drain(i..j);
            }
            1 => {
                let dup = s[i..j].to_vec();
                s.splice(j..j, dup);
            }
            2 => {
                let t = soup(r, 6);
                s.splice(i..i, t);
            }
            3 => s.truncate(j),
            4 => {
                if i < n {
                    #[allow(clippy::cast_possible_truncation)]
                    let b = r.next() as u8;
                    s[i] = b;
                }
            }
            _ => {
                let deep = r
                    .pick(&[b"<div>".as_slice(), b"<b>", b"<td>", b"<font>"])
                    .repeat(r.below(80) + 220);
                s.splice(i..i, deep);
            }
        }
    }
    s
}

fn fixture_inputs() -> Vec<Vec<u8>> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/html_data/fromstring.jsonl"
    );
    let text = std::fs::read_to_string(path).expect("fixture");
    text.lines()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).expect("json");
            let i = &v["input"];
            if let Some(t) = i.get("text").and_then(|t| t.as_str()) {
                t.as_bytes().to_vec()
            } else {
                let h = i["hex"].as_str().expect("hex");
                (0..h.len())
                    .step_by(2)
                    .map(|k| u8::from_str_radix(&h[k..k + 2], 16).expect("hex digit"))
                    .collect()
            }
        })
        .collect()
}

#[test]
fn random_inputs_keep_the_invariants() {
    let env = |k: &str, d: u64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let iters = env("PYROID_HTML_FUZZ_ITERS", 3000);
    let mut r = Rng(env("PYROID_HTML_FUZZ_SEED", 1));
    let large = env("PYROID_HTML_FUZZ_LARGE", 0) == 1;
    let base = fixture_inputs();
    let mut bytes = 0usize;
    let mut big = 0;
    for k in 0..iters {
        let data = match r.below(4) {
            0 => noise(&mut r),
            1 => soup(&mut r, 120),
            _ => {
                let s = r.pick(&base).clone();
                mutate(&mut r, s)
            }
        };
        bytes += data.len();
        check(&data);
        if large && k % 20_000 == 19_999 {
            // Over 10 MB: the same checks through the buffer emulation.
            let unit = soup(&mut r, 40);
            let mut doc = unit.repeat(10_000_000 / unit.len().max(1) + 1);
            doc.extend_from_slice(&soup(&mut r, 20));
            check(&doc);
            big += 1;
        }
    }
    eprintln!("{iters} random documents ({bytes} bytes) and {big} over 10 MB checked");
}
