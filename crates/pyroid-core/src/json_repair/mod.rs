// Derived from json_repair 0.64.9 (MIT), © Stefano Baccianella; modified.
//! A port of json_repair 0.64.9 (`pyroid.json_repair`): repair the broken JSON
//! that LLMs emit, with json_repair's exact output and without unbounded recursion.
//! Linear on the input families where json_repair's lookaheads are quadratic. Inputs
//! where json_repair's parser backtracks (parses a value, drops it and parses again
//! a little further on) stay quadratic, as in json_repair; the fuzzer finds many
//! (`fuzz/fuzz_targets/json_repair.rs`).
//!
//! - [`value`]: the Python-typed value tree.
//! - [`chars`]: Python's `str` predicates per Unicode version, lone surrogates.
//! - [`pyjson`]: `json.loads` and `raw_decode` (the valid-JSON fast path).
//! - [`dumps`]: `json.dumps` for value trees.
//! - `parser`: json_repair's `JSONParser`.

pub mod chars;
pub mod dumps;
mod parser;
pub mod pyjson;
#[rustfmt::skip]
mod tables;
pub mod value;

pub use parser::{LogEntry, MAX_DEPTH, ParseError, ParseOptions};

use chars::CharProps;
use dumps::{DumpError, DumpOptions};
use value::Value;

/// Why a repair raised instead of returning.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RepairError {
    /// An exception json_repair raises (strict mode, nesting, `int()` limits).
    Parse(ParseError),
    /// `json.dumps` raised (`allow_nan=False`).
    Dump(DumpError),
    /// The kernel cannot represent this input exactly (a char in the range
    /// reserved for remapped surrogates); the caller delegates to json_repair.
    Unsupported,
}

impl From<ParseError> for RepairError {
    fn from(e: ParseError) -> Self {
        match e {
            ParseError::Unsupported => Self::Unsupported,
            e => Self::Parse(e),
        }
    }
}

/// The valid-JSON fast path: `Some` when `json.loads(text)` succeeds.
fn fast_path(text: &str, opts: &RepairOptions) -> Result<Option<Repaired>, RepairError> {
    if opts.skip_json_loads {
        return Ok(None);
    }
    match pyjson::loads(text, opts.max_int_digits) {
        Ok(value) => Ok(Some(Repaired {
            value,
            log: opts.logging.then(Vec::new),
            valid_json: true,
        })),
        Err(pyjson::DecodeError::ReservedCodePoint) => Err(RepairError::Unsupported),
        Err(_) => Ok(None),
    }
}

/// `repair_json` options that matter to the kernel (no `schema`, no `json_fd`
/// streaming: `load` reads the file first and sets `try_valid_json_suffix = false`).
#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct RepairOptions {
    pub skip_json_loads: bool,
    pub logging: bool,
    pub stream_stable: bool,
    pub strict: bool,
    /// `json_fd is None` (false for `load` / `from_file`).
    pub try_valid_json_suffix: bool,
    /// `sys.get_int_max_str_digits()` (0 = unlimited).
    pub max_int_digits: usize,
    pub props: CharProps,
}

impl RepairOptions {
    /// `repair_json`'s defaults for the given Unicode tables.
    pub fn new(props: CharProps) -> Self {
        Self {
            skip_json_loads: false,
            logging: false,
            stream_stable: false,
            strict: false,
            try_valid_json_suffix: true,
            max_int_digits: 4300,
            props,
        }
    }

    fn parse_options(&self) -> ParseOptions {
        ParseOptions {
            stream_stable: self.stream_stable,
            strict: self.strict,
            try_valid_json_suffix: self.try_valid_json_suffix,
            logging: self.logging,
            max_int_digits: self.max_int_digits,
            props: self.props,
        }
    }
}

/// A repaired value and, with `logging`, json_repair's repair log.
#[derive(Debug)]
pub struct Repaired {
    pub value: Value,
    pub log: Option<Vec<LogEntry>>,
    /// `json.loads` took the whole text (json_repair's `is_valid_json`): `repair_json`
    /// then writes an empty-string value as `'""'`, not as `''`.
    pub valid_json: bool,
}

/// `repair_json(text, return_objects=True, ...)`: the repaired value.
///
/// `text` is a Python `str`'s UTF-8: one holding a char in the range reserved
/// for remapped surrogates is [`RepairError::Unsupported`] (use
/// [`repair_chars`] for strs with lone surrogates).
pub fn repair(text: &str, opts: &RepairOptions) -> Result<Repaired, RepairError> {
    if chars::contains_reserved(text) {
        return Err(RepairError::Unsupported);
    }
    if let Some(r) = fast_path(text, opts)? {
        return Ok(r);
    }
    let chars: Vec<char> = text.chars().collect();
    repair_chars_parse(&chars, opts)
}

/// [`repair`] for text already decoded to chars (surrogates remapped).
pub fn repair_chars(chars: &[char], opts: &RepairOptions) -> Result<Repaired, RepairError> {
    if !opts.skip_json_loads {
        let s: String = chars.iter().collect();
        if let Some(r) = fast_path(&s, opts)? {
            return Ok(r);
        }
    }
    repair_chars_parse(chars, opts)
}

fn repair_chars_parse(chars: &[char], opts: &RepairOptions) -> Result<Repaired, RepairError> {
    let (value, log) = parser::parse(chars, &opts.parse_options())?;
    Ok(Repaired {
        value,
        log,
        valid_json: false,
    })
}

/// `repair_json(text, ...)` returning the string: `""` when the repair found no
/// value (an empty-string result from the parser), else `json.dumps(value, **dump_opts)`.
pub fn repair_json(
    text: &str,
    opts: &RepairOptions,
    dump: &DumpOptions,
) -> Result<String, RepairError> {
    let r = repair(text, opts)?;
    let out = dumps_result(&r, dump);
    value::drop_deep(r.value);
    out
}

/// The string `repair_json` returns for `r`.
pub fn dumps_result(r: &Repaired, dump: &DumpOptions) -> Result<String, RepairError> {
    if !r.valid_json && r.value.is_str("") {
        return Ok(String::new());
    }
    dumps::dumps(&r.value, dump).map_err(RepairError::Dump)
}

#[cfg(test)]
mod tests {
    //! The kernel against json_repair 0.64.9: every distinct input of json_repair's
    //! own test suite (`testdata/upstream_cases.json`, recorded with CPython 3.12 by
    //! `python -m pyroid_parity.json_repair_kernel_cases`), under four option sets.

    use super::chars::{CharProps, UnicodeVersion};
    use super::dumps::DumpOptions;
    use super::{ParseError, RepairError, RepairOptions, repair_json};

    fn opts() -> RepairOptions {
        RepairOptions::new(CharProps::new(UnicodeVersion::V15_0))
    }

    fn outcome(text: &str, o: &RepairOptions) -> serde_json::Value {
        match repair_json(text, o, &DumpOptions::default()) {
            Ok(s) => serde_json::json!({ "ok": s }),
            Err(RepairError::Parse(ParseError::Strict(msg))) => {
                serde_json::json!({ "err": "ValueError", "msg": msg })
            }
            Err(e) => serde_json::json!({ "err": format!("{e:?}") }),
        }
    }

    #[test]
    fn upstream_inputs_match_json_repair() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("testdata/upstream_cases.json")).expect("fixture");
        let cases = data["cases"].as_array().expect("cases");
        assert!(cases.len() > 400);
        let mut failures = Vec::new();
        for case in cases {
            let text = case["in"].as_str().expect("input");
            let variants = [
                ("default", opts()),
                (
                    "skip",
                    RepairOptions {
                        skip_json_loads: true,
                        ..opts()
                    },
                ),
                (
                    "strict",
                    RepairOptions {
                        strict: true,
                        ..opts()
                    },
                ),
                (
                    "stream",
                    RepairOptions {
                        stream_stable: true,
                        ..opts()
                    },
                ),
            ];
            for (name, o) in variants {
                let got = outcome(text, &o);
                if got != case[name] {
                    failures.push(format!("{name} {text:?}: got {got}, want {}", case[name]));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} diffs:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    #[test]
    fn examples() {
        let o = opts();
        let d = DumpOptions::default();
        for (input, want) in [
            ("{'a': 1, b: [1, 2,}", r#"{"a": 1, "b": [1, 2]}"#),
            ("[1, 2, 3", "[1, 2, 3]"),
            (r#"{"a": "unterminated"#, r#"{"a": "unterminated"}"#),
            ("```json\n{\"x\": [1, 2]}\n```", r#"{"x": [1, 2]}"#),
            ("{\"a\": True, \"b\": None}", r#"{"a": true, "b": null}"#),
            ("nothing", ""),
            ("", ""),
        ] {
            assert_eq!(repair_json(input, &o, &d).as_deref(), Ok(want), "{input:?}");
        }
        // Valid JSON too: the reserved range must never reach the fast path.
        for text in ["\"\u{10f800}\"", "{'a': '\u{10ffff}'}", r#""\udbff\udfff""#] {
            assert_eq!(
                repair_json(text, &o, &d),
                Err(RepairError::Unsupported),
                "{text:?}"
            );
        }
        let deep = "[".repeat(super::MAX_DEPTH + 1);
        assert_eq!(
            repair_json(&deep, &o, &d),
            Err(RepairError::Parse(ParseError::Depth))
        );
    }

    /// `MAX_DEPTH` counts the arrays and objects that nest, whatever their mix: an empty
    /// object that the parser reads again as an array (`[{[{`) is one level, not two (it
    /// raised at 668 levels). (A `{` in key position is skipped, so `{{` is one level.)
    #[test]
    fn max_depth_counts_levels_in_every_shape() {
        let (o, d) = (opts(), DumpOptions::default());
        for unit in [
            "[",
            "{\"a\":",
            "{\"a\":[",
            "[{",
            "{[",
            "[[{",
            "[{[",
            "[{\"a\": ",
        ] {
            let per = unit.chars().filter(|c| matches!(c, '[' | '{')).count();
            let n = super::MAX_DEPTH / per;
            assert!(
                repair_json(&unit.repeat(n), &o, &d).is_ok(),
                "{unit:?} x {n} ({} levels)",
                n * per
            );
            assert_eq!(
                repair_json(&unit.repeat(n + 1), &o, &d),
                Err(RepairError::Parse(ParseError::Depth)),
                "{unit:?} x {} ({} levels)",
                n + 1,
                (n + 1) * per
            );
        }
    }

    /// A deep valid prefix (`raw_decode` builds it without a depth limit) that the parser
    /// then fails on or discards must be dropped iteratively: a recursive drop of a million
    /// levels overflowed the stack and killed the process.
    #[test]
    fn deep_valid_prefix_is_dropped_without_recursion() {
        const N: usize = 1_000_000;
        let deep = format!("{}{}", "[".repeat(N), "]".repeat(N));
        let mut strict = opts();
        strict.strict = true;
        let cases = [
            // `parse()` errors on the second top-level value, holding the first.
            (format!("{deep} {}", "[".repeat(2000)), opts()),
            // Strict mode rejects a second top-level value.
            (format!("{deep} 1"), strict),
            // `try_parse_missing_opening_object` decodes `{` + input, then declines it.
            (format!("\"a\": {deep}}} }}"), opts()),
        ];
        // An 8 MiB stack, like CPython's main thread on Linux.
        std::thread::Builder::new()
            .stack_size(8 << 20)
            .spawn(move || {
                let d = DumpOptions::default();
                for (text, o) in &cases {
                    // Any outcome but a crash; the strict case raises json_repair's error.
                    let _ = repair_json(text, o, &d);
                }
                assert_eq!(
                    repair_json(&cases[1].0, &cases[1].1, &d),
                    Err(RepairError::Parse(ParseError::Strict(
                        "Multiple top-level JSON elements found in strict mode."
                    )))
                );
            })
            .expect("spawn")
            .join()
            .expect("no panic");
    }
}
