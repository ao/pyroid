// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! `dedup::simplify_text` against what CPython computes: datatrove 0.10.1 itself
//! under CPython 3.13, a vendored copy of `simplify_text` (only `re`, `regex`,
//! `unicodedata`) under 3.11, 3.12, 3.14 and 3.15. The data is recorded by
//! `tests/dedup_data/oracle.py` inside the pyroid-dev container:
//!
//! - `simplify_py3*.jsonl` (`oracle.py simplify`): random texts under many
//!   configs. Set `PYROID_SIMPLIFY_EXTRA` to a directory of more such files
//!   (e.g. a large uncommitted run) to replay those too.
//! - `simplify_digests_py3*.json` (`oracle.py digests`): a CRC-32 per block of
//!   4096 code points of `simplify_text` on each code point alone and in a few
//!   contexts, so every code point is checked without committing the outputs.
//!
//! The throughput test is `#[ignore]`d: `cargo test --release --test
//! dedup_simplify -- --ignored --nocapture`.

use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pyroid_core::dedup::simplify::simplify_text_into;
use pyroid_core::dedup::{TextNormConfig, UnicodeVersion, simplify_text};
use serde_json::Value;

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/dedup_data")
}

/// Config bit i is field i of datatrove's `TextNormConfig`.
fn config(bits: u64) -> TextNormConfig {
    TextNormConfig {
        lowercase: bits & 1 != 0,
        norm_whitespace: bits & 2 != 0,
        remove_punctuation: bits & 4 != 0,
        norm_unicode_diacritics: bits & 8 != 0,
        norm_numbers: bits & 16 != 0,
        norm_weekdays: bits & 32 != 0,
        norm_monthnames: bits & 64 != 0,
    }
}

fn files_in(dir: &Path, prefix: &str, suffix: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with(prefix) && n.ends_with(suffix))
        })
        .collect();
    out.sort();
    out
}

fn version_of(header: &Value) -> UnicodeVersion {
    let v = header["unicode"].as_str().expect("header has \"unicode\"");
    UnicodeVersion::from_unidata_version(v).unwrap_or_else(|| panic!("unknown Unicode {v}"))
}

const ALL_VERSIONS: [UnicodeVersion; 5] = [
    UnicodeVersion::V14_0,
    UnicodeVersion::V15_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V16_0,
    UnicodeVersion::V17_0,
];

/// Where `got` first differs from `want`, with a little context (texts can be
/// whole documents).
fn first_difference(want: &str, got: &str) -> String {
    let at = want
        .bytes()
        .zip(got.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    let at = (0..=at)
        .rev()
        .find(|&i| want.is_char_boundary(i) && got.is_char_boundary(i))
        .unwrap();
    let window = |s: &str| -> String {
        let start = (0..=at.saturating_sub(40))
            .rev()
            .find(|&i| s.is_char_boundary(i))
            .unwrap();
        s[start..].chars().take(80).collect()
    };
    format!(
        "at byte {at}:\n  want ...{:?}\n  got  ...{:?}",
        window(want),
        window(got)
    )
}

/// Replay one `oracle.py simplify` file; returns (version, texts, cases).
fn replay(path: &Path) -> (UnicodeVersion, usize, usize) {
    let data = std::fs::read_to_string(path).unwrap();
    let mut lines = data.lines();
    let header: Value = serde_json::from_str(lines.next().expect("header line")).unwrap();
    let v = version_of(&header);
    let (mut texts, mut cases) = (0, 0);
    let mut failures = Vec::new();
    let mut buf = String::new();
    for line in lines {
        let rec: Value = serde_json::from_str(line).unwrap();
        let text = rec["text"].as_str().unwrap();
        texts += 1;
        for (bits, expect) in rec["out"].as_object().unwrap() {
            let cfg = config(bits.parse().unwrap());
            let expect = expect.as_str().unwrap();
            let got = simplify_text(text, &cfg, v);
            cases += 1;
            let borrowed = matches!(got, Cow::Borrowed(_));
            let short: String = text.chars().take(200).collect();
            if got != expect || borrowed != (expect == text) {
                failures.push(format!(
                    "{v:?} config {bits} borrowed={borrowed} text {short:?}\n{}",
                    first_difference(expect, &got)
                ));
            }
            simplify_text_into(text, &cfg, v, &mut buf);
            if buf != expect {
                failures.push(format!(
                    "{v:?} config {bits} (into) text {short:?}\n{}",
                    first_difference(expect, &buf)
                ));
            }
        }
    }
    let n = failures.len();
    failures.truncate(10);
    assert!(
        n == 0,
        "{}: {n} mismatches, first:\n{}",
        path.display(),
        failures.join("\n")
    );
    (v, texts, cases)
}

#[test]
#[allow(clippy::print_stderr)]
fn golden_files_match_cpython() {
    let mut files = files_in(&data_dir(), "simplify_py", ".jsonl");
    let mut seen: Vec<UnicodeVersion> = Vec::new();
    for f in &files {
        let (v, _, _) = replay(f);
        seen.push(v);
    }
    for v in ALL_VERSIONS {
        assert!(
            seen.contains(&v),
            "no golden file for Unicode {}",
            v.as_str()
        );
    }
    if let Some(extra) = std::env::var_os("PYROID_SIMPLIFY_EXTRA") {
        files = files_in(Path::new(&extra), "", ".jsonl");
        assert!(
            !files.is_empty(),
            "no .jsonl files in PYROID_SIMPLIFY_EXTRA"
        );
        for f in &files {
            let (v, texts, cases) = replay(f);
            eprintln!(
                "{}: Unicode {}, {texts} texts, {cases} cases ok",
                f.display(),
                v.as_str()
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Every code point, through digests
// ---------------------------------------------------------------------------

const fn crc_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        #[allow(clippy::cast_possible_truncation)]
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        t[i] = c;
        i += 1;
    }
    t
}
static CRC_TABLE: [u32; 256] = crc_table();

/// zlib's `crc32(data, crc)`.
fn crc32(data: &[u8], crc: u32) -> u32 {
    let mut c = !crc;
    for &b in data {
        c = CRC_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
    }
    !c
}

/// The first mismatching block of one digest case, if any.
fn check_case(
    v: UnicodeVersion,
    bits: u64,
    template: &str,
    crcs: &str,
    block: u32,
) -> Option<String> {
    let cfg = config(bits);
    let (before, after) = template.split_once("{}").expect("template has {}");
    let mut text = String::new();
    let mut start = 0u32;
    for want in crcs.as_bytes().chunks(8) {
        let want = u32::from_str_radix(std::str::from_utf8(want).unwrap(), 16).unwrap();
        let mut crc = 0;
        for c in (start..start + block).filter_map(char::from_u32) {
            text.clear();
            text.push_str(before);
            text.push(c);
            text.push_str(after);
            let out = simplify_text(&text, &cfg, v);
            let len = u32::try_from(out.len()).unwrap();
            crc = crc32(out.as_bytes(), crc32(&len.to_le_bytes(), crc));
        }
        if crc != want {
            let end = start + block - 1;
            return Some(format!(
                "{v:?} config {bits} template {template:?}: block U+{start:04X}..U+{end:04X} differs"
            ));
        }
        start += block;
    }
    assert_eq!(start, 0x11_0000, "digest covers every code point");
    None
}

#[test]
fn every_code_point_matches_cpython() {
    let files = files_in(&data_dir(), "simplify_digests_py", ".json");
    let docs: Vec<Value> = files
        .iter()
        .map(|f| serde_json::from_str(&std::fs::read_to_string(f).unwrap()).unwrap())
        .collect();
    let mut seen: Vec<UnicodeVersion> = docs.iter().map(version_of).collect();
    seen.sort();
    assert_eq!(seen, ALL_VERSIONS, "one digest file per Unicode version");
    // One thread per (version, case): the debug build is slow, and this
    // keeps the whole check to a few seconds of wall time.
    let jobs: Vec<(UnicodeVersion, u64, String, String, u32)> = docs
        .iter()
        .flat_map(|d| {
            let v = version_of(d);
            let block = u32::try_from(d["block"].as_u64().unwrap()).unwrap();
            d["cases"].as_array().unwrap().iter().map(move |c| {
                (
                    v,
                    c["config"].as_u64().unwrap(),
                    c["template"].as_str().unwrap().to_owned(),
                    c["crc32"].as_str().unwrap().to_owned(),
                    block,
                )
            })
        })
        .collect();
    let threads = std::thread::available_parallelism()
        .map_or(4, usize::from)
        .min(8);
    let next = std::sync::atomic::AtomicUsize::new(0);
    let failures = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| {
                loop {
                    let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let Some((v, bits, template, crcs, block)) = jobs.get(i) else {
                        break;
                    };
                    if let Some(f) = check_case(*v, *bits, template, crcs, *block) {
                        failures.lock().unwrap().push(f);
                    }
                }
            });
        }
    });
    let failures = failures.into_inner().unwrap();
    assert!(
        failures.is_empty(),
        "{} of {} cases differ:\n{}",
        failures.len(),
        jobs.len(),
        failures.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Throughput (indicative)
// ---------------------------------------------------------------------------

const ASCII_SAMPLE: &str = "The Quick Brown Fox, born on 12 March 2019, jumped over 3.5 lazy dogs! \
    \"Really?\" asked Dr. Smith (aged 42) -- it's true: 1,000 people saw it; see http://example.com/a_b.\n\
    In the U.S., prices rose 4.2% in Q3; the CEO's memo [p. 7] said \"we'll grow\" {again}.\t";

const MIXED_SAMPLE: &str = "Café “quoted” — naïve résumé, Größe 3,5 m² … Москва, 12 ИЮНЯ 2024 г. \
    東京は日本の首都です。 Ελληνικά: ΟΔΟΣ ΣΟΦΙΑΣ. Ünïcödé façade — İstanbul’s 1.000 ÇİÇEK! ";

/// About 4 KiB documents cut from `sample`, `total` bytes in all.
fn docs(sample: &str, total: usize) -> Vec<String> {
    let mut doc = String::new();
    while doc.len() < 4096 {
        doc.push_str(sample);
    }
    let n = total / doc.len();
    vec![doc; n]
}

#[allow(clippy::print_stderr, clippy::cast_precision_loss)]
fn throughput(name: &str, docs: &[String], cfg: TextNormConfig) {
    let bytes: usize = docs.iter().map(String::len).sum();
    let ascii = docs
        .iter()
        .map(|d| d.bytes().filter(u8::is_ascii).count())
        .sum::<usize>();
    let mut out = String::new();
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        for d in docs {
            simplify_text_into(d, &cfg, UnicodeVersion::V15_1, &mut out);
            std::hint::black_box(&out);
        }
        best = best.min(t.elapsed().as_secs_f64());
    }
    eprintln!(
        "{name}: {:.0} MB/s ({bytes} bytes, {:.1}% ASCII, {} docs)",
        bytes as f64 / best / 1e6,
        100.0 * ascii as f64 / bytes as f64,
        docs.len()
    );
}

#[test]
#[ignore = "throughput numbers; run with --release -- --ignored --nocapture"]
fn simplify_throughput() {
    let default = TextNormConfig::default();
    let all = config(127);
    let ascii = docs(ASCII_SAMPLE, 32 << 20);
    let mixed = docs(MIXED_SAMPLE, 32 << 20);
    // Real corpora, if given: a directory of JSON files, each a list of texts.
    if let Some(dir) = std::env::var_os("PYROID_SIMPLIFY_BENCH_DIR") {
        for f in files_in(Path::new(&dir), "", ".json") {
            let docs: Vec<String> =
                serde_json::from_str(&std::fs::read_to_string(&f).unwrap()).unwrap();
            let name = f.file_stem().unwrap().to_string_lossy().into_owned();
            throughput(&format!("{name}, default config"), &docs, default);
            throughput(&format!("{name}, every flag"), &docs, all);
        }
    }
    for (name, docs) in [("ascii", &ascii), ("mixed", &mixed)] {
        throughput(&format!("{name}, default config"), docs, default);
        throughput(&format!("{name}, every flag"), docs, all);
        for (pass, bits) in [
            ("lowercase", 1),
            ("numbers", 16),
            ("punct+ws", 6),
            ("diacritics", 8),
        ] {
            throughput(&format!("{name}, {pass} only"), docs, config(bits));
        }
    }
}
