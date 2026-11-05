// Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
//! Differential tests: replay calls recorded from the real ftfy 6.3.1
//! (`tests/ftfy_data/oracle.py`) and require byte-identical results.
//!
//! * `ftfy_data/golden-*.jsonl` (committed) holds ftfy's own test inputs,
//!   recorded on one CPython and replayed under every Unicode version.
//! * `PYROID_FTFY_ORACLE=file1.jsonl:file2.jsonl` adds more recordings, such
//!   as large fuzz runs generated in the dev container.

use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};

use pyroid_core::ftfy::{
    self, ExplainedText, Normalization, TextFixerConfig, UnescapeHtml, UnicodeVersion,
};
use serde_json::Value;

/// A recorded string: plain JSON, or `{"h": hex}` (surrogatepass UTF-8).
fn text_of(v: &Value) -> Vec<u8> {
    match v {
        Value::String(s) => s.as_bytes().to_vec(),
        Value::Object(o) => {
            let h = o["h"].as_str().unwrap();
            (0..h.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
                .collect()
        }
        _ => panic!("bad recorded string {v}"),
    }
}

fn show(b: &[u8]) -> String {
    format!("{:?}", String::from_utf8_lossy(b))
}

/// Build the config ftfy would use for `fn(text, config?, **kwargs)`.
fn config_for(row: &Value, version: UnicodeVersion) -> Result<TextFixerConfig, String> {
    let fname = row["fn"].as_str().unwrap();
    let no_kwargs = serde_json::Map::new();
    let default_explain = matches!(fname, "fix_and_explain" | "fix_encoding_and_explain");
    let mut cfg = TextFixerConfig {
        explain: row["config_explain"].as_bool().unwrap_or(default_explain),
        unicode_version: version,
        ..TextFixerConfig::default()
    };
    for (k, v) in row
        .get("kwargs")
        .and_then(Value::as_object)
        .unwrap_or(&no_kwargs)
    {
        let flag = || v.as_bool().ok_or(format!("{k}={v}"));
        match k.as_str() {
            "unescape_html" => {
                cfg.unescape_html = match v {
                    Value::String(s) if s == "auto" => UnescapeHtml::Auto,
                    Value::Bool(true) => UnescapeHtml::Always,
                    Value::Bool(false) => UnescapeHtml::Never,
                    _ => return Err(format!("unescape_html={v}")),
                }
            }
            "normalization" => {
                cfg.normalization = match v {
                    Value::Null => None,
                    Value::String(s) => Some(Normalization::from_name(s).ok_or("bad form")?),
                    _ => return Err(format!("normalization={v}")),
                }
            }
            "max_decode_length" => {
                cfg.max_decode_length =
                    NonZeroUsize::new(v.as_u64().ok_or("bad length")? as usize).ok_or("zero")?;
            }
            "explain" => cfg.explain = flag()?,
            "remove_terminal_escapes" => cfg.remove_terminal_escapes = flag()?,
            "fix_encoding" => cfg.fix_encoding = flag()?,
            "restore_byte_a0" => cfg.restore_byte_a0 = flag()?,
            "replace_lossy_sequences" => cfg.replace_lossy_sequences = flag()?,
            "decode_inconsistent_utf8" => cfg.decode_inconsistent_utf8 = flag()?,
            "fix_c1_controls" => cfg.fix_c1_controls = flag()?,
            "fix_latin_ligatures" => cfg.fix_latin_ligatures = flag()?,
            "fix_character_width" => cfg.fix_character_width = flag()?,
            "uncurl_quotes" => cfg.uncurl_quotes = flag()?,
            "fix_line_breaks" => cfg.fix_line_breaks = flag()?,
            "fix_surrogates" => cfg.fix_surrogates = flag()?,
            "remove_control_chars" => cfg.remove_control_chars = flag()?,
            _ => return Err(format!("unknown option {k}")),
        }
    }
    Ok(cfg)
}

fn explanation_json(e: Option<&Vec<ftfy::ExplanationStep>>) -> Value {
    match e {
        None => Value::Null,
        Some(steps) => Value::Array(
            steps
                .iter()
                .map(|s| Value::Array(vec![s.action.as_str().into(), s.parameter.into()]))
                .collect(),
        ),
    }
}

/// Replay one recorded call; `Err` describes a mismatch.
fn replay(input: &[u8], row: &Value, version: UnicodeVersion) -> Result<(), String> {
    let input = input.to_vec();
    let fname = row["fn"].as_str().unwrap();
    let cfg = config_for(row, version)?;
    if let Some(err) = row.get("error") {
        return Err(format!("ftfy raised {err}; not replayable"));
    }
    let want = text_of(&row["out"]);
    let (got, expl): (Vec<u8>, Option<Value>) = match fname {
        "fix_text" => (
            ftfy::fix_text_wtf8(&input, &cfg).unwrap().into_owned(),
            None,
        ),
        "fix_text_segment" => (
            ftfy::fix_text_segment_wtf8(&input, &cfg)
                .unwrap()
                .into_owned(),
            None,
        ),
        "fix_encoding" => (
            ftfy::fix_encoding_wtf8(&input, &cfg).unwrap().into_owned(),
            None,
        ),
        "fix_and_explain" | "fix_encoding_and_explain" => {
            let ExplainedText { text, explanation } = if fname == "fix_and_explain" {
                ftfy::fix_and_explain_wtf8(&input, &cfg).unwrap()
            } else {
                ftfy::fix_encoding_and_explain_wtf8(&input, &cfg).unwrap()
            };
            (
                text.into_owned(),
                Some(explanation_json(explanation.as_ref())),
            )
        }
        _ => return Err(format!("unknown function {fname}")),
    };
    if got != want {
        return Err(format!(
            "{fname}({}, {}) = {} but ftfy gives {}",
            show(&input),
            row.get("kwargs").unwrap_or(&Value::Null),
            show(&got),
            show(&want)
        ));
    }
    if let Some(expl) = expl {
        if expl != row["explanation"] {
            return Err(format!(
                "{fname}({}) explanation {expl} but ftfy gives {}",
                show(&input),
                row["explanation"]
            ));
        }
    }
    Ok(())
}

fn version_of(path: &Path) -> UnicodeVersion {
    // Files are named `<kind>-<unicode version>.jsonl`, e.g. golden-15.0.0.jsonl.
    let stem = path.file_stem().unwrap().to_str().unwrap();
    let v = stem.rsplit('-').next().unwrap();
    UnicodeVersion::from_unidata_version(v).unwrap_or_else(|| panic!("no version in {stem}"))
}

/// Replay every file; `all_versions` replays each file under every
/// supported Unicode version instead of the one in its name.
fn run_files(files: &[PathBuf], all_versions: bool) {
    const VERSIONS: [UnicodeVersion; 5] = [
        UnicodeVersion::V14_0,
        UnicodeVersion::V15_0,
        UnicodeVersion::V15_1,
        UnicodeVersion::V16_0,
        UnicodeVersion::V17_0,
    ];
    let mut total = 0;
    let mut failures = Vec::new();
    for path in files {
        let versions = if all_versions {
            VERSIONS.to_vec()
        } else {
            vec![version_of(path)]
        };
        let data = std::fs::read_to_string(path).unwrap();
        for version in versions {
            for (n, line) in data.lines().enumerate() {
                let row: Value = serde_json::from_str(line).unwrap();
                let input = text_of(&row["in"]);
                for c in row["calls"].as_array().unwrap() {
                    if c.get("error").is_some() {
                        continue; // exceptions are covered by the Python-level tests
                    }
                    total += 1;
                    if let Err(e) = replay(&input, c, version) {
                        failures.push(format!("{}:{} [{version:?}]: {e}", path.display(), n + 1));
                    }
                }
            }
        }
    }
    let shown: Vec<_> = failures.iter().take(25).cloned().collect();
    assert!(
        failures.is_empty(),
        "{} of {total} recorded calls differ from ftfy:\n{}",
        failures.len(),
        shown.join("\n")
    );
    eprintln!("{total} recorded calls match ftfy");
}

#[test]
fn golden_recordings_match_ftfy() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ftfy_data");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("golden-")
                && p.extension().is_some_and(|e| e == "jsonl")
        })
        .collect();
    files.sort();
    assert!(
        !files.is_empty(),
        "no golden recordings in {}",
        dir.display()
    );
    // ftfy's output on its own test inputs is the same on every CPython
    // version (checked when recording), so replay it under each version.
    run_files(&files, true);
}

#[test]
fn extra_recordings_match_ftfy() {
    let Ok(list) = std::env::var("PYROID_FTFY_ORACLE") else {
        eprintln!("PYROID_FTFY_ORACLE not set; skipping");
        return;
    };
    let files: Vec<PathBuf> = list
        .split(':')
        .filter(|s| !s.is_empty())
        .map(PathBuf::from)
        .collect();
    run_files(&files, false);
}
