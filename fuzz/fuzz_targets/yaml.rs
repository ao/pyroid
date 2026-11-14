//! PyYAML 6.0.3's loader and emitter (`pyroid_core::yaml`): the reader, scanner, parser and
//! composer over arbitrary text, and the emitter over the events of what parsed.
//!
//! Input: `[flags] text...`. The text is the rest as UTF-8 (invalid bytes replaced), with the
//! characters PyYAML's reader rejects removed (the bindings raise `ReaderError` before the
//! scanner sees them). Bits of `flags`: 0-1 the emitter's `best_indent` (2, 4, 9, 3), 2
//! `canonical`, 3 `allow_unicode`, 4 a narrow `best_width` (20), 5-7 a reader `limit` (none,
//! or the first 1/8, 2/8, ... of the text, as for a file object with a decoding error there).
//!
//! Checked, besides "no panic and no stack overflow at any depth":
//! - deterministic: composing twice gives the same graphs or the same error;
//! - with a `limit`, the result is the full text's, or `NeedMore` (the pending reader error);
//! - the events of a stream that parses go through the emitter (as `yaml.emit` would) without
//!   a panic, and what it writes parses again.

#![no_main]

use libfuzzer_sys::fuzz_target;
use pyroid_core::yaml::composer::ComposerOptions;
use pyroid_core::yaml::emitter::{Emitter, EmitterOptions};
use pyroid_core::yaml::error::YamlError;
use pyroid_core::yaml::events::{Event, EventKind};
use pyroid_core::yaml::parser::Parser;
use pyroid_core::yaml::reader::{Reader, is_printable};
use pyroid_core::yaml::scanner::{Scanner, ScannerOptions};
use pyroid_core::yaml::{self, nodes::Graph};

fn compose(text: &str, limit: Option<usize>) -> Result<Vec<Graph>, YamlError> {
    let mut c = yaml::loader(
        text.to_owned(),
        limit,
        ScannerOptions::default(),
        ComposerOptions { max_depth: 2000 },
    );
    let mut out = Vec::new();
    while c.check_node()? {
        match c.get_node()? {
            Some(g) => out.push(g),
            None => break,
        }
    }
    Ok(out)
}

fn events(text: &str) -> Result<Vec<Event>, YamlError> {
    let mut p = Parser::new(Scanner::new(
        Reader::new(text.to_owned(), None),
        ScannerOptions::default(),
    ));
    let mut out = Vec::new();
    while let Some(e) = p.get_event()? {
        out.push(e);
        if out.len() > 100_000 {
            break;
        }
    }
    Ok(out)
}

fuzz_target!(|data: &[u8]| {
    let Some((&flags, rest)) = data.split_first() else {
        return;
    };
    let text: String = String::from_utf8_lossy(rest)
        .chars()
        .filter(|&c| is_printable(c))
        .collect();
    let full = compose(&text, None);
    assert_eq!(full, compose(&text, None), "not deterministic");
    let eighths = usize::from(flags >> 5);
    if eighths > 0 {
        let chars = text.chars().count();
        let limit = chars * eighths / 8;
        let prefix: String = text.chars().take(limit).collect();
        match compose(&prefix, Some(limit)) {
            Err(YamlError::NeedMore) => {}
            other => assert_eq!(other, full, "a limit changed the result"),
        }
    }
    let Ok(evs) = events(&text) else { return };
    let options = EmitterOptions {
        canonical: flags & 4 != 0,
        allow_unicode: flags & 8 != 0,
        best_indent: [2, 4, 9, 3][usize::from(flags & 3)],
        best_width: if flags & 16 != 0 { 20 } else { 80 },
        line_break: "\n",
        record_writes: false,
    };
    let mut emitter = Emitter::new(options);
    for mut e in evs {
        // The emitter takes what yaml.emit gets from yaml.parse.
        if let EventKind::StreamStart { encoding } = &mut e.kind {
            *encoding = None;
        }
        if emitter.emit(e).is_err() {
            return;
        }
    }
    let out = emitter.take_output();
    if let Err(e) = events(&out) {
        // A YAML error is possible (PyYAML's emitter does not promise round trips); a
        // Value/Overflow error from escapes it wrote itself would be a bug.
        assert!(
            matches!(e, YamlError::Marked(_)),
            "re-parse of emitted text: {e} for {out:?}"
        );
    }
});
