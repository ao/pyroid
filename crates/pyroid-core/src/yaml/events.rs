// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! Parser events (`events.py`): what the parser produces and the emitter consumes.

use super::error::Mark;

/// An event with PyYAML's `start_mark` and `end_mark` (the emitter ignores the marks).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Event {
    pub kind: EventKind,
    pub start_mark: Mark,
    pub end_mark: Mark,
}

impl Event {
    /// An event without marks (for the emitter).
    pub fn new(kind: EventKind) -> Self {
        Self {
            kind,
            start_mark: Mark::default(),
            end_mark: Mark::default(),
        }
    }
}

/// PyYAML's event classes. Fields keep PyYAML's names and meaning:
///
/// - `style` is the scalar style: `None` (plain, or PyYAML's `''`/`None`), or one of
///   `'"'`, `'\''`, `'|'`, `'>'`.
/// - `implicit` of a scalar is PyYAML's `(plain_implicit, quoted_implicit)` pair; of a
///   collection a single flag.
/// - `flow_style`: `Some(true)` flow, `Some(false)` block, `None` (the emitter decides).
/// - `version` and `tags` of a document start: the `%YAML` and `%TAG` directives. `tags` is
///   sorted by handle (the emitter writes them in `sorted()` order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EventKind {
    StreamStart {
        encoding: Option<String>,
    },
    StreamEnd,
    DocumentStart {
        explicit: bool,
        version: Option<(u64, u64)>,
        tags: Option<Vec<(String, String)>>,
    },
    DocumentEnd {
        explicit: bool,
    },
    Alias {
        anchor: String,
    },
    Scalar {
        anchor: Option<String>,
        tag: Option<String>,
        implicit: (bool, bool),
        value: String,
        style: Option<char>,
    },
    SequenceStart {
        anchor: Option<String>,
        tag: Option<String>,
        implicit: bool,
        flow_style: Option<bool>,
    },
    SequenceEnd,
    MappingStart {
        anchor: Option<String>,
        tag: Option<String>,
        implicit: bool,
        flow_style: Option<bool>,
    },
    MappingEnd,
}
