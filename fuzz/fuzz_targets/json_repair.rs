//! json_repair 0.64.5 (`pyroid_core::json_repair`): `repair`, `repair_chars`, `repair_json`
//! and `dumps_result`, under options from the first five input bytes.
//!
//! Input: `[repair dump extra n1 n2] a b rest...`, where `a` and `b` are the `n1` and `n2`
//! bytes after the header. Bits of `repair`: 0 `skip_json_loads`, 1 `logging`, 2
//! `stream_stable`, 3 `strict`, 4 `from_file` (no valid-suffix retry, as `load` and
//! `from_file`), 5-7 the Unicode tables (newest first: 0 is 17.0, what the binding uses until
//! the shim sets the interpreter's). Bits of `dump`: 0 `ensure_ascii=False`, 1
//! `allow_nan=False`, 2 `sort_keys`, 3-4 `indent` (None, 2, `"\t"`, 0), 5
//! `separators=(",", ":")`, 6 CPython 3.11's error message. Bits of `extra`: 0-1
//! `sys.get_int_max_str_digits()` (4300, 0, 640, 100000); 2-7 `e`. With `e = 0` the text is
//! `a + rest + b`; otherwise `rest` is `pre NUL mid NUL post` and the text is
//! `pre + a * r + mid + b * r + post` with `r = 2^e`, capped at 512 KiB of repeats, so
//! that a small `-max_len` still reaches deep nesting (`[` and `]` give a depth of 2^18).
//! All zero: `json_repair.repair_json(rest)`.
//!
//! The text goes in as the binding passes a Python str: UTF-8 through `repair`;
//! surrogatepass UTF-8 (lone surrogates) through `repair_chars` with the surrogates
//! remapped; other bytes decoded lossily.
//!
//! Size bounds, so that no input only measures a known cost: `indent` is dropped for texts
//! over 4 KiB (the indented output of deep nesting grows with the square of the depth);
//! `logging` and the first three checks below only run up to 8 KiB (json_repair's parser
//! rescans the rest of the text in places, so its work and its log can grow with the square
//! of the length, as in json_repair itself); texts over 16 KiB are only checked when the
//! repeated middle, `a * r + mid + b * r`, is valid JSON.
//!
//! Checked, besides "no panic, abort or stack overflow at any depth" (results are dropped
//! with `value::drop_deep`, as the binding does):
//! - deterministic: the same value, log and error twice;
//! - `repair` and `repair_chars` agree on UTF-8 text (the binding takes either for a str);
//! - `logging` changes only the log: the same value or error, a log exactly when logging,
//!   and an empty one when `json.loads` took the text (json_repair's `self.log` only appends);
//! - errors come from options that are on: `Strict` only in strict mode, `Int` only under a
//!   digit limit, `Dump` only with `allow_nan=False`; `NeedStack` never escapes;
//! - the string `repair_json` returns is `""` or valid JSON for `json.loads`, and repairing
//!   it again (`skip_json_loads=False`, the same dump options) gives it back: json_repair
//!   returns `json.dumps(value)`, and `json.loads` of that output takes the fast path to a
//!   value that dumps the same (an `""` result repairs to `""`).

#![no_main]

use libfuzzer_sys::fuzz_target;
use pyroid_core::json_repair::chars::{self, CharProps, UnicodeVersion};
use pyroid_core::json_repair::dumps::{self, DumpOptions};
use pyroid_core::json_repair::{
    self as jr, LogEntry, ParseError, RepairError, RepairOptions, Repaired, pyjson, value,
};

const HEADER: usize = 5;
/// Newest first, so a zero field selects 17.0.
const VERSIONS: [UnicodeVersion; 5] = [
    UnicodeVersion::V17_0,
    UnicodeVersion::V16_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V15_0,
    UnicodeVersion::V14_0,
];
/// `sys.set_int_max_str_digits` accepts 0 (no limit) and values from 640.
const DIGITS: [usize; 4] = [4300, 0, 640, 100_000];
/// Bytes of repeated `a` and `b` at most.
const MAX_REPEAT: usize = 1 << 19;
/// Longest text dumped with `indent`.
const MAX_INDENTED: usize = 4096;
/// Longest text checked for determinism, the char path and logging (logging is off for
/// longer ones: the log grows with the parser's work).
const MAX_CROSS: usize = 8 << 10;
/// Longest text checked unless its repeated middle is valid JSON.
const MAX_PARSE: usize = 16 << 10;

fn options(h: &[u8; HEADER]) -> (RepairOptions, DumpOptions) {
    let bit = |b: u8, i: u32| (b >> i) & 1 != 0;
    let r = h[0];
    let mut o = RepairOptions::new(CharProps::new(VERSIONS[usize::from(r >> 5) % 5]));
    o.skip_json_loads = bit(r, 0);
    o.logging = bit(r, 1);
    o.stream_stable = bit(r, 2);
    o.strict = bit(r, 3);
    o.try_valid_json_suffix = !bit(r, 4);
    o.max_int_digits = DIGITS[usize::from(h[2] & 3)];
    let d = h[1];
    let indent = match (d >> 3) & 3 {
        0 => None,
        1 => Some("  ".to_owned()),
        2 => Some("\t".to_owned()),
        _ => Some(String::new()),
    };
    // `json.dumps` defaults the item separator to "," when indenting.
    let (item, key) = if bit(d, 5) {
        (",", ":")
    } else if indent.is_some() {
        (",", ": ")
    } else {
        (", ", ": ")
    };
    let dump = DumpOptions {
        ensure_ascii: !bit(d, 0),
        allow_nan: !bit(d, 1),
        sort_keys: bit(d, 2),
        indent,
        item_separator: item.into(),
        key_separator: key.into(),
        python_3_11: bit(d, 6),
    };
    (o, dump)
}

/// A Python str as the binding hands it to the kernel.
enum Text {
    Str(String),
    /// The chars of a str with lone surrogates (remapped).
    Chars(Vec<char>),
}

fn run(text: &Text, o: &RepairOptions) -> Result<Repaired, RepairError> {
    match text {
        Text::Str(s) => jr::repair(s, o),
        Text::Chars(c) => jr::repair_chars(c, o),
    }
}

/// What `repair_json` returns as a string (the binding's mode 0: no logging).
fn repair_string(text: &Text, o: &RepairOptions, d: &DumpOptions) -> Result<String, RepairError> {
    let o = RepairOptions {
        logging: false,
        ..*o
    };
    match text {
        Text::Str(s) => jr::repair_json(s, &o, d),
        Text::Chars(c) => jr::repair_chars(c, &o).and_then(|r| {
            let out = jr::dumps_result(&r.value, d);
            value::drop_deep(r.value);
            out
        }),
    }
}

/// A repair's outcome in comparable form: the value as `json.dumps` writes it (iteratively;
/// `Value`'s own `PartialEq` and `Drop` recurse) and the log.
type Outcome = Result<(String, Option<Vec<LogEntry>>), RepairError>;

fn outcome(r: Result<Repaired, RepairError>) -> Outcome {
    r.map(|r| {
        let s = dumps::dumps(&r.value, &DumpOptions::default()).expect("allow_nan=True");
        value::drop_deep(r.value);
        (s, r.log)
    })
}

/// A short description of an outcome for panic messages (values can be megabytes).
fn brief(o: &Outcome) -> String {
    match o {
        Ok((s, log)) => format!(
            "Ok({} bytes, log {:?})",
            s.len(),
            log.as_ref().map(Vec::len)
        ),
        Err(e) => format!("Err({e:?})"),
    }
}

/// Whether `json.loads` succeeds on the text the fast path would see.
fn loads_ok(text: &Text, o: &RepairOptions) -> bool {
    let s: String = match text {
        Text::Str(s) if chars::contains_reserved(s) => return false,
        Text::Str(s) => s.clone(),
        Text::Chars(c) => c.iter().collect(),
    };
    match pyjson::loads(&s, o.max_int_digits) {
        Ok(v) => {
            value::drop_deep(v);
            true
        }
        Err(_) => false,
    }
}

fn check(text: &Text, len: usize, opts: &RepairOptions, dump: &DumpOptions) {
    let opts = &RepairOptions {
        logging: opts.logging && len <= MAX_CROSS,
        ..*opts
    };
    let first = outcome(run(text, opts));
    if let Err(e) = &first {
        match e {
            RepairError::Parse(ParseError::Strict(_)) => assert!(opts.strict, "Strict: {e:?}"),
            RepairError::Parse(ParseError::Int(_)) => assert_ne!(opts.max_int_digits, 0),
            RepairError::Parse(ParseError::NeedStack | ParseError::Unsupported) => {
                panic!("internal error escaped: {e:?}")
            }
            RepairError::Dump(_) => panic!("a value repair raised a dump error: {e:?}"),
            _ => {}
        }
    }
    if len <= MAX_CROSS {
        cross_check(text, opts, &first);
    }
    drop(first);
    output(text, opts, dump);
}

/// Determinism, the char path and logging, against the first outcome.
fn cross_check(text: &Text, opts: &RepairOptions, first: &Outcome) {
    let second = outcome(run(text, opts));
    assert!(
        *first == second,
        "not deterministic: {} then {}",
        brief(first),
        brief(&second)
    );

    if let Text::Str(s) = text
        && !chars::contains_reserved(s)
    {
        let c: Vec<char> = s.chars().collect();
        let via_chars = outcome(jr::repair_chars(&c, opts));
        assert!(
            *first == via_chars,
            "repair {} but repair_chars {}",
            brief(first),
            brief(&via_chars)
        );
    }

    let flipped = RepairOptions {
        logging: !opts.logging,
        ..*opts
    };
    let other = outcome(run(text, &flipped));
    match (first, &other) {
        (Ok((v, log)), Ok((w, log2))) => {
            assert!(v == w, "logging changed the value");
            assert_eq!(log.is_some(), opts.logging, "a log exactly when logging");
            assert_eq!(log2.is_some(), !opts.logging, "a log exactly when logging");
            let log = if opts.logging { log } else { log2 };
            if !opts.skip_json_loads && loads_ok(text, opts) {
                assert_eq!(log.as_ref().map(Vec::len), Some(0), "fast path, empty log");
            }
        }
        (Err(e), Err(f)) => assert_eq!(e, f, "logging changed the error"),
        _ => panic!(
            "logging changed the outcome: {} vs {}",
            brief(first),
            brief(&other)
        ),
    }
}

/// `repair_json`'s string: `""` or valid JSON that repairs to itself.
fn output(text: &Text, opts: &RepairOptions, dump: &DumpOptions) {
    let out = match repair_string(text, opts, dump) {
        Ok(out) => out,
        Err(RepairError::Dump(e)) => {
            assert!(!dump.allow_nan, "dump error with allow_nan: {e:?}");
            return;
        }
        Err(_) => return,
    };
    if !out.is_empty() {
        match pyjson::loads(&out, opts.max_int_digits) {
            Ok(v) => value::drop_deep(v),
            Err(e) => panic!("json.loads rejects repair_json's output ({e:?})"),
        }
    }
    let again_opts = RepairOptions {
        skip_json_loads: false,
        logging: false,
        stream_stable: false,
        strict: false,
        try_valid_json_suffix: true,
        ..*opts
    };
    // Remapped surrogates in the output are lone surrogates of the Python str.
    let again_text = if chars::contains_reserved(&out) {
        Text::Chars(out.chars().collect())
    } else {
        Text::Str(out.clone())
    };
    match repair_string(&again_text, &again_opts, dump) {
        Ok(again) => assert!(
            again == out,
            "repair_json(out) != out ({} vs {} bytes)",
            again.len(),
            out.len()
        ),
        // The escapes \udbff\udfff (two lone surrogates) decode to a reserved char: delegated.
        Err(RepairError::Unsupported) => {}
        Err(e) => panic!("repairing repair_json's output failed: {e:?}"),
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((h, body)) = data.split_first_chunk::<HEADER>() else {
        return;
    };
    let (opts, mut dump) = options(h);
    let (a, body) = body.split_at(usize::from(h[3]).min(body.len()));
    let (b, rest) = body.split_at(usize::from(h[4]).min(body.len()));
    let unit = a.len() + b.len();
    let mut r = 1usize << u32::from(h[2] >> 2).min(19);
    if unit > 0 {
        r = r.min(MAX_REPEAT / unit).max(1);
    }
    // `e > 0`: `rest` is `pre NUL mid NUL post`.
    let mut parts = rest.splitn(3, |&c| c == 0);
    let (pre, mid, post) = if h[2] >> 2 == 0 {
        (&[][..], rest, &[][..])
    } else {
        let pre = parts.next().unwrap_or_default();
        let mid = parts.next().unwrap_or_default();
        (pre, mid, parts.next().unwrap_or_default())
    };
    let mut bytes = Vec::with_capacity(unit * r + rest.len());
    bytes.extend_from_slice(pre);
    for _ in 0..r {
        bytes.extend_from_slice(a);
    }
    bytes.extend_from_slice(mid);
    for _ in 0..r {
        bytes.extend_from_slice(b);
    }
    bytes.extend_from_slice(post);
    let len = bytes.len();
    if len > MAX_PARSE {
        // json_repair's parser takes quadratic time on some inputs; long ones are checked only
        // when their bulk is valid JSON (deep nesting), which decodes in linear time.
        let m = &bytes[pre.len()..len - post.len()];
        let valid = std::str::from_utf8(m).is_ok_and(|s| {
            !chars::contains_reserved(s) && pyjson::loads(s, 0).map(value::drop_deep).is_ok()
        });
        if !valid {
            return;
        }
    }
    if len > MAX_INDENTED {
        dump.indent = None;
    }
    let text = match String::from_utf8(bytes) {
        Ok(s) => Text::Str(s),
        Err(e) => match chars::chars_from_wtf8(e.as_bytes()) {
            Some(Ok(c)) => Text::Chars(c),
            // A reserved code point: the binding delegates to json_repair.
            Some(Err(_)) => return,
            None => Text::Str(String::from_utf8_lossy(e.as_bytes()).into_owned()),
        },
    };
    check(&text, len, &opts, &dump);
});
