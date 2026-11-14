// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! The emitter (`emitter.py`): events to YAML text.
//!
//! A port of PyYAML's pure-Python `Emitter`, kept close to the original so the two can be
//! read side by side: the same state machine with an explicit state stack, the same event
//! lookahead (`need_more_events`), the same scalar analysis, style choice and writers. It
//! produces the same text for every event stream and option combination, and the same
//! `EmitterError` messages.
//!
//! The emitter writes text. The caller encodes it, and flushes the stream at the end (PyYAML's
//! `flush_stream` writes nothing). Because Python's `utf-16` codec puts a BOM in front of every
//! `stream.write`, [`Emitter::writes`] can record where each of PyYAML's writes ends.
//!
//! The events PyYAML grammar expects:
//!
//! ```text
//! stream   ::= STREAM-START document* STREAM-END
//! document ::= DOCUMENT-START node DOCUMENT-END
//! node     ::= SCALAR | sequence | mapping
//! sequence ::= SEQUENCE-START node* SEQUENCE-END
//! mapping  ::= MAPPING-START (node node)* MAPPING-END
//! ```

use std::borrow::Cow;
use std::collections::VecDeque;
use std::fmt::Write as _;

use super::error::{Arg, Msg, YamlError};
use super::events::{Event, EventKind};

/// Formatting options, normalized as PyYAML's `Emitter.__init__` normalizes its arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmitterOptions {
    pub canonical: bool,
    pub allow_unicode: bool,
    /// `indent` when `1 < indent < 10`, else 2.
    pub best_indent: usize,
    /// `width` when it is more than `2 * best_indent`, else 80. `u64::MAX` stands for
    /// `float('inf')`; a finite float width is passed as its floor (columns are integers).
    pub best_width: u64,
    /// `"\n"`, `"\r"` or `"\r\n"` (PyYAML's `best_line_break`).
    pub line_break: &'static str,
    /// Record the end of each of PyYAML's `stream.write` calls (see [`Emitter::writes`]).
    pub record_writes: bool,
}

impl Default for EmitterOptions {
    fn default() -> Self {
        Self {
            canonical: false,
            allow_unicode: false,
            best_indent: 2,
            best_width: 80,
            line_break: "\n",
            record_writes: false,
        }
    }
}

/// The emitter states: PyYAML's `expect_*` methods that `self.state` points to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    StreamStart,
    FirstDocumentStart,
    DocumentStart,
    DocumentEnd,
    DocumentRoot,
    FirstFlowSequenceItem,
    FlowSequenceItem,
    FirstFlowMappingKey,
    FlowMappingKey,
    FlowMappingSimpleValue,
    FlowMappingValue,
    FirstBlockSequenceItem,
    BlockSequenceItem,
    FirstBlockMappingKey,
    BlockMappingKey,
    BlockMappingSimpleValue,
    BlockMappingValue,
    Nothing,
}

/// The node context `expect_node` sets: PyYAML's `root_context`, `sequence_context`,
/// `mapping_context` and `simple_key_context` flags, which are never set two at a time
/// except `mapping` with `simple_key`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Context {
    Root,
    Sequence,
    Mapping,
    SimpleKey,
}

impl Context {
    fn mapping(self) -> bool {
        matches!(self, Self::Mapping | Self::SimpleKey)
    }
}

/// A chosen scalar style (PyYAML's `''`, `"'"`, `'"'`, `'|'`, `'>'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarStyle {
    Plain,
    SingleQuoted,
    DoubleQuoted,
    Literal,
    Folded,
}

/// PyYAML's `ScalarAnalysis` without `scalar` (the event's value) and `allow_double_quoted`
/// (always true, never read).
#[allow(clippy::struct_excessive_bools)] // PyYAML's flags, one to one
#[derive(Clone, Copy, Debug)]
struct ScalarAnalysis {
    /// `len(scalar)`, in code points.
    length: usize,
    empty: bool,
    multiline: bool,
    allow_flow_plain: bool,
    allow_block_plain: bool,
    allow_single_quoted: bool,
    allow_block: bool,
}

/// PyYAML's `Emitter`, writing to a string.
#[allow(clippy::struct_excessive_bools)] // PyYAML's flags, one to one
#[derive(Debug)]
pub struct Emitter {
    // Formatting details.
    canonical: bool,
    allow_unicode: bool,
    best_indent: usize,
    best_width: u64,
    best_line_break: &'static str,
    record_writes: bool,

    // The state machine with a stack of states for nested structures.
    states: Vec<State>,
    state: State,

    // The event queue (`self.events`); the current event is passed to the state methods.
    events: VecDeque<Event>,

    // The current indentation level and the stack of previous indents.
    indents: Vec<Option<usize>>,
    indent: Option<usize>,

    flow_level: usize,
    context: Context,

    // Characteristics of the last emitted character: the column (in code points), whether it
    // is a whitespace, whether it is an indentation character (indentation space, '-', '?',
    // or ':'). PyYAML also counts `line`, which it never reads.
    column: usize,
    whitespace: bool,
    indention: bool,

    // Whether the document requires an explicit document end indicator.
    open_ended: bool,

    // `tag_prefixes` (prefix -> handle), sorted by prefix as `prepare_tag` walks it.
    tag_prefixes: Vec<(Cow<'static, str>, Cow<'static, str>)>,

    // `prepared_anchor` (prepare_anchor returns the anchor itself, so a flag is enough),
    // `prepared_tag`, `analysis` and `style`, cached across the checks of one event.
    prepared_anchor: bool,
    prepared_tag: Option<String>,
    analysis: Option<ScalarAnalysis>,
    style: Option<ScalarStyle>,

    out: String,
    writes: Vec<usize>,
}

/// `EmitterError(msg)`.
fn emitter_error(msg: Msg) -> YamlError {
    YamlError::Emitter(msg)
}

/// The `TypeError` that PyYAML 6.0.3's `prepare_tag_prefix` raises for any character it
/// would percent-encode: it calls `ord()` on the `int` items of `ch.encode('utf-8')`.
///
/// `YamlError` has no `TypeError` variant yet, so this travels as `Value`; the bindings must
/// raise it as `TypeError`.
pub const TAG_PREFIX_TYPE_ERROR: &str = "ord() expected string of length 1, but int found";

fn tag_prefix_type_error() -> YamlError {
    YamlError::Type(TAG_PREFIX_TYPE_ERROR.to_owned())
}

impl Emitter {
    /// Create an emitter that expects a `StreamStart` event first.
    pub fn new(options: EmitterOptions) -> Self {
        Self {
            canonical: options.canonical,
            allow_unicode: options.allow_unicode,
            best_indent: options.best_indent,
            best_width: options.best_width,
            best_line_break: options.line_break,
            record_writes: options.record_writes,
            states: Vec::new(),
            state: State::StreamStart,
            events: VecDeque::with_capacity(4),
            indents: Vec::new(),
            indent: None,
            flow_level: 0,
            context: Context::Root,
            column: 0,
            whitespace: true,
            indention: true,
            open_ended: false,
            tag_prefixes: Vec::new(),
            prepared_anchor: false,
            prepared_tag: None,
            analysis: None,
            style: None,
            out: String::new(),
            writes: Vec::new(),
        }
    }

    /// Feed one event (`Emitter.emit`). Some events wait in the queue until the events after
    /// them are known: a document start for one more, a sequence start for two, a mapping
    /// start for three (fewer if the stream or collection ends first).
    pub fn emit(&mut self, event: Event) -> Result<(), YamlError> {
        self.events.push_back(event);
        while !self.need_more_events() {
            let Some(event) = self.events.pop_front() else {
                break;
            };
            self.run_state(&event)?;
        }
        Ok(())
    }

    /// Everything written so far (since the last [`take_output`](Self::take_output)).
    pub fn output(&self) -> &str {
        &self.out
    }

    /// Take the text written so far, and clear the recorded writes (their offsets point into
    /// the taken text: read [`writes`](Self::writes) first).
    pub fn take_output(&mut self) -> String {
        self.writes.clear();
        std::mem::take(&mut self.out)
    }

    /// The end offsets (in bytes into [`output`](Self::output)) of each `stream.write` call
    /// PyYAML makes, in order; empty writes repeat the previous offset. Only filled when
    /// [`EmitterOptions::record_writes`] is set.
    pub fn writes(&self) -> &[usize] {
        &self.writes
    }

    // In some cases, we wait for a few next events before emitting.

    fn need_more_events(&self) -> bool {
        let Some(event) = self.events.front() else {
            return true;
        };
        match event.kind {
            EventKind::DocumentStart { .. } => self.need_events(1),
            EventKind::SequenceStart { .. } => self.need_events(2),
            EventKind::MappingStart { .. } => self.need_events(3),
            _ => false,
        }
    }

    fn need_events(&self, count: usize) -> bool {
        let mut level: isize = 0;
        for event in self.events.iter().skip(1) {
            match event.kind {
                EventKind::DocumentStart { .. }
                | EventKind::SequenceStart { .. }
                | EventKind::MappingStart { .. } => level += 1,
                EventKind::DocumentEnd { .. } | EventKind::SequenceEnd | EventKind::MappingEnd => {
                    level -= 1;
                }
                EventKind::StreamEnd => level = -1,
                _ => {}
            }
            if level < 0 {
                return false;
            }
        }
        // `len(self.events) < count+1`
        self.events.len() <= count
    }

    fn increase_indent(&mut self, flow: bool, indentless: bool) {
        self.indents.push(self.indent);
        match self.indent {
            None => self.indent = Some(if flow { self.best_indent } else { 0 }),
            Some(indent) if !indentless => self.indent = Some(indent + self.best_indent),
            Some(_) => {}
        }
    }

    fn pop_indent(&mut self) {
        self.indent = self.indents.pop().flatten();
    }

    fn pop_state(&mut self) {
        // Every `expect_node` call follows a push, and every state that pops ends a node.
        self.state = self
            .states
            .pop()
            .expect("the emitter state stack is balanced");
    }

    fn run_state(&mut self, event: &Event) -> Result<(), YamlError> {
        match self.state {
            State::StreamStart => self.expect_stream_start(event),
            State::FirstDocumentStart => self.expect_document_start(event, true),
            State::DocumentStart => self.expect_document_start(event, false),
            State::DocumentEnd => self.expect_document_end(event),
            State::DocumentRoot => self.expect_document_root(event),
            State::FirstFlowSequenceItem => self.expect_first_flow_sequence_item(event),
            State::FlowSequenceItem => self.expect_flow_sequence_item(event),
            State::FirstFlowMappingKey => self.expect_first_flow_mapping_key(event),
            State::FlowMappingKey => self.expect_flow_mapping_key(event),
            State::FlowMappingSimpleValue => self.expect_flow_mapping_simple_value(event),
            State::FlowMappingValue => self.expect_flow_mapping_value(event),
            State::FirstBlockSequenceItem => self.expect_block_sequence_item(event, true),
            State::BlockSequenceItem => self.expect_block_sequence_item(event, false),
            State::FirstBlockMappingKey => self.expect_block_mapping_key(event, true),
            State::BlockMappingKey => self.expect_block_mapping_key(event, false),
            State::BlockMappingSimpleValue => self.expect_block_mapping_simple_value(event),
            State::BlockMappingValue => self.expect_block_mapping_value(event),
            State::Nothing => Err(event_error(&EXPECTED_NOTHING, event)),
        }
    }

    // Stream handlers.

    fn expect_stream_start(&mut self, event: &Event) -> Result<(), YamlError> {
        let EventKind::StreamStart { encoding } = &event.kind else {
            return Err(event_error(&EXPECTED_STREAM_START, event));
        };
        // write_stream_start: write a BOM if needed (the caller encodes it with the rest).
        if encoding.as_deref().is_some_and(|e| e.starts_with("utf-16")) {
            self.write("\u{FEFF}");
        }
        self.state = State::FirstDocumentStart;
        Ok(())
    }

    // Document handlers.

    fn expect_document_start(&mut self, event: &Event, first: bool) -> Result<(), YamlError> {
        match &event.kind {
            EventKind::DocumentStart {
                explicit,
                version,
                tags,
            } => {
                let tags = tags.as_deref().filter(|t| !t.is_empty());
                if (version.is_some() || tags.is_some()) && self.open_ended {
                    self.write_indicator("...", true, false, false);
                    self.write_indent();
                }
                if let Some((major, minor)) = *version {
                    let version_text = prepare_version(major, minor)?;
                    self.write_version_directive(&version_text);
                }
                // `self.tag_prefixes = self.DEFAULT_TAG_PREFIXES.copy()`
                self.tag_prefixes.clear();
                self.tag_prefixes.extend([
                    (Cow::Borrowed("!"), Cow::Borrowed("!")),
                    (Cow::Borrowed("tag:yaml.org,2002:"), Cow::Borrowed("!!")),
                ]);
                if let Some(tags) = tags {
                    let mut handles: Vec<&(String, String)> = tags.iter().collect();
                    handles.sort_by(|a, b| a.0.cmp(&b.0));
                    for (handle, prefix) in handles {
                        self.set_tag_prefix(prefix, handle);
                        prepare_tag_handle(handle)?;
                        prepare_tag_prefix(prefix)?;
                        self.write_tag_directive(handle, prefix);
                    }
                }
                let implicit = first
                    && !explicit
                    && !self.canonical
                    && version.is_none()
                    && tags.is_none()
                    && !self.check_empty_document(event);
                if !implicit {
                    self.write_indent();
                    self.write_indicator("---", true, false, false);
                    if self.canonical {
                        self.write_indent();
                    }
                }
                self.state = State::DocumentRoot;
                Ok(())
            }
            EventKind::StreamEnd => {
                if self.open_ended {
                    self.write_indicator("...", true, false, false);
                    self.write_indent();
                }
                // write_stream_end only flushes.
                self.state = State::Nothing;
                Ok(())
            }
            _ => Err(event_error(&EXPECTED_DOCUMENT_START, event)),
        }
    }

    fn expect_document_end(&mut self, event: &Event) -> Result<(), YamlError> {
        let EventKind::DocumentEnd { explicit } = event.kind else {
            return Err(event_error(&EXPECTED_DOCUMENT_END, event));
        };
        self.write_indent();
        if explicit {
            self.write_indicator("...", true, false, false);
            self.write_indent();
        }
        // flush_stream writes nothing.
        self.state = State::DocumentStart;
        Ok(())
    }

    fn expect_document_root(&mut self, event: &Event) -> Result<(), YamlError> {
        self.states.push(State::DocumentEnd);
        self.expect_node(event, Context::Root)
    }

    // Node handlers.

    fn expect_node(&mut self, event: &Event, context: Context) -> Result<(), YamlError> {
        self.context = context;
        match &event.kind {
            EventKind::Alias { .. } => self.expect_alias(event),
            EventKind::Scalar { .. } => {
                self.process_anchor(event, '&')?;
                self.process_tag(event)?;
                self.expect_scalar(event);
                Ok(())
            }
            EventKind::SequenceStart { flow_style, .. } => {
                self.process_anchor(event, '&')?;
                self.process_tag(event)?;
                if self.flow_level > 0
                    || self.canonical
                    || *flow_style == Some(true)
                    || self.check_empty_sequence(event)
                {
                    self.expect_flow_sequence();
                } else {
                    self.expect_block_sequence();
                }
                Ok(())
            }
            EventKind::MappingStart { flow_style, .. } => {
                self.process_anchor(event, '&')?;
                self.process_tag(event)?;
                if self.flow_level > 0
                    || self.canonical
                    || *flow_style == Some(true)
                    || self.check_empty_mapping(event)
                {
                    self.expect_flow_mapping();
                } else {
                    self.expect_block_mapping();
                }
                Ok(())
            }
            _ => Err(event_error(&EXPECTED_NODE, event)),
        }
    }

    fn expect_alias(&mut self, event: &Event) -> Result<(), YamlError> {
        // PyYAML's "anchor is not specified for alias" needs an `AliasEvent(None)`, which an
        // `EventKind::Alias` cannot hold.
        self.process_anchor(event, '*')?;
        self.pop_state();
        Ok(())
    }

    fn expect_scalar(&mut self, event: &Event) {
        self.increase_indent(true, false);
        self.process_scalar(event);
        self.pop_indent();
        self.pop_state();
    }

    // Flow sequence handlers.

    fn expect_flow_sequence(&mut self) {
        self.write_indicator("[", true, true, false);
        self.flow_level += 1;
        self.increase_indent(true, false);
        self.state = State::FirstFlowSequenceItem;
    }

    fn expect_first_flow_sequence_item(&mut self, event: &Event) -> Result<(), YamlError> {
        if matches!(event.kind, EventKind::SequenceEnd) {
            self.pop_indent();
            self.flow_level -= 1;
            self.write_indicator("]", false, false, false);
            self.pop_state();
            Ok(())
        } else {
            if self.canonical || self.past_width(self.column) {
                self.write_indent();
            }
            self.states.push(State::FlowSequenceItem);
            self.expect_node(event, Context::Sequence)
        }
    }

    fn expect_flow_sequence_item(&mut self, event: &Event) -> Result<(), YamlError> {
        if matches!(event.kind, EventKind::SequenceEnd) {
            self.pop_indent();
            self.flow_level -= 1;
            if self.canonical {
                self.write_indicator(",", false, false, false);
                self.write_indent();
            }
            self.write_indicator("]", false, false, false);
            self.pop_state();
            Ok(())
        } else {
            self.write_indicator(",", false, false, false);
            if self.canonical || self.past_width(self.column) {
                self.write_indent();
            }
            self.states.push(State::FlowSequenceItem);
            self.expect_node(event, Context::Sequence)
        }
    }

    // Flow mapping handlers.

    fn expect_flow_mapping(&mut self) {
        self.write_indicator("{", true, true, false);
        self.flow_level += 1;
        self.increase_indent(true, false);
        self.state = State::FirstFlowMappingKey;
    }

    fn expect_first_flow_mapping_key(&mut self, event: &Event) -> Result<(), YamlError> {
        if matches!(event.kind, EventKind::MappingEnd) {
            self.pop_indent();
            self.flow_level -= 1;
            self.write_indicator("}", false, false, false);
            self.pop_state();
            Ok(())
        } else {
            if self.canonical || self.past_width(self.column) {
                self.write_indent();
            }
            self.flow_mapping_key(event)
        }
    }

    fn expect_flow_mapping_key(&mut self, event: &Event) -> Result<(), YamlError> {
        if matches!(event.kind, EventKind::MappingEnd) {
            self.pop_indent();
            self.flow_level -= 1;
            if self.canonical {
                self.write_indicator(",", false, false, false);
                self.write_indent();
            }
            self.write_indicator("}", false, false, false);
            self.pop_state();
            Ok(())
        } else {
            self.write_indicator(",", false, false, false);
            if self.canonical || self.past_width(self.column) {
                self.write_indent();
            }
            self.flow_mapping_key(event)
        }
    }

    /// The tail both flow mapping key states share.
    fn flow_mapping_key(&mut self, event: &Event) -> Result<(), YamlError> {
        if !self.canonical && self.check_simple_key(event)? {
            self.states.push(State::FlowMappingSimpleValue);
            self.expect_node(event, Context::SimpleKey)
        } else {
            self.write_indicator("?", true, false, false);
            self.states.push(State::FlowMappingValue);
            self.expect_node(event, Context::Mapping)
        }
    }

    fn expect_flow_mapping_simple_value(&mut self, event: &Event) -> Result<(), YamlError> {
        self.write_indicator(":", false, false, false);
        self.states.push(State::FlowMappingKey);
        self.expect_node(event, Context::Mapping)
    }

    fn expect_flow_mapping_value(&mut self, event: &Event) -> Result<(), YamlError> {
        if self.canonical || self.past_width(self.column) {
            self.write_indent();
        }
        self.write_indicator(":", true, false, false);
        self.states.push(State::FlowMappingKey);
        self.expect_node(event, Context::Mapping)
    }

    // Block sequence handlers.

    fn expect_block_sequence(&mut self) {
        let indentless = self.context.mapping() && !self.indention;
        self.increase_indent(false, indentless);
        self.state = State::FirstBlockSequenceItem;
    }

    fn expect_block_sequence_item(&mut self, event: &Event, first: bool) -> Result<(), YamlError> {
        if !first && matches!(event.kind, EventKind::SequenceEnd) {
            self.pop_indent();
            self.pop_state();
            Ok(())
        } else {
            self.write_indent();
            self.write_indicator("-", true, false, true);
            self.states.push(State::BlockSequenceItem);
            self.expect_node(event, Context::Sequence)
        }
    }

    // Block mapping handlers.

    fn expect_block_mapping(&mut self) {
        self.increase_indent(false, false);
        self.state = State::FirstBlockMappingKey;
    }

    fn expect_block_mapping_key(&mut self, event: &Event, first: bool) -> Result<(), YamlError> {
        if !first && matches!(event.kind, EventKind::MappingEnd) {
            self.pop_indent();
            self.pop_state();
            Ok(())
        } else {
            self.write_indent();
            if self.check_simple_key(event)? {
                self.states.push(State::BlockMappingSimpleValue);
                self.expect_node(event, Context::SimpleKey)
            } else {
                self.write_indicator("?", true, false, true);
                self.states.push(State::BlockMappingValue);
                self.expect_node(event, Context::Mapping)
            }
        }
    }

    fn expect_block_mapping_simple_value(&mut self, event: &Event) -> Result<(), YamlError> {
        self.write_indicator(":", false, false, false);
        self.states.push(State::BlockMappingKey);
        self.expect_node(event, Context::Mapping)
    }

    fn expect_block_mapping_value(&mut self, event: &Event) -> Result<(), YamlError> {
        self.write_indent();
        self.write_indicator(":", true, false, true);
        self.states.push(State::BlockMappingKey);
        self.expect_node(event, Context::Mapping)
    }

    // Checkers.

    fn check_empty_sequence(&self, event: &Event) -> bool {
        matches!(event.kind, EventKind::SequenceStart { .. })
            && self
                .events
                .front()
                .is_some_and(|e| matches!(e.kind, EventKind::SequenceEnd))
    }

    fn check_empty_mapping(&self, event: &Event) -> bool {
        matches!(event.kind, EventKind::MappingStart { .. })
            && self
                .events
                .front()
                .is_some_and(|e| matches!(e.kind, EventKind::MappingEnd))
    }

    fn check_empty_document(&self, event: &Event) -> bool {
        if !matches!(event.kind, EventKind::DocumentStart { .. }) {
            return false;
        }
        // PyYAML also tests `event.implicit`, a 2-tuple, which is always true.
        matches!(
            self.events.front(),
            Some(Event { kind: EventKind::Scalar { anchor: None, tag: None, value, .. }, .. })
                if value.is_empty()
        )
    }

    fn check_simple_key(&mut self, event: &Event) -> Result<bool, YamlError> {
        let mut length = 0;
        if let Some(anchor) = node_anchor(event) {
            if !self.prepared_anchor {
                prepare_anchor(anchor)?;
                self.prepared_anchor = true;
            }
            length += anchor.chars().count();
        }
        if let Some(tag) = node_tag(event) {
            if self.prepared_tag.is_none() {
                self.prepared_tag = Some(self.prepare_tag(tag)?);
            }
            length += self
                .prepared_tag
                .as_deref()
                .map_or(0, |t| t.chars().count());
        }
        let mut simple_scalar = false;
        if let EventKind::Scalar { value, .. } = &event.kind {
            let analysis = self.analysis_of(value);
            length += analysis.length;
            simple_scalar = !analysis.empty && !analysis.multiline;
        }
        Ok(length < 128
            && (matches!(event.kind, EventKind::Alias { .. })
                || simple_scalar
                || self.check_empty_sequence(event)
                || self.check_empty_mapping(event)))
    }

    // Anchor, Tag, and Scalar processors.

    fn process_anchor(&mut self, event: &Event, indicator: char) -> Result<(), YamlError> {
        let Some(anchor) = node_anchor(event) else {
            self.prepared_anchor = false;
            return Ok(());
        };
        if !self.prepared_anchor {
            prepare_anchor(anchor)?;
        }
        // The prepared anchor is the anchor itself, and never empty.
        let mut data = String::with_capacity(anchor.len() + 1);
        data.push(indicator);
        data.push_str(anchor);
        self.write_indicator(&data, true, false, false);
        self.prepared_anchor = false;
        Ok(())
    }

    fn process_tag(&mut self, event: &Event) -> Result<(), YamlError> {
        let tag = match &event.kind {
            EventKind::Scalar { tag, implicit, .. } => {
                let style = if let Some(style) = self.style {
                    style
                } else {
                    let style = self.choose_scalar_style(event);
                    self.style = Some(style);
                    style
                };
                if (!self.canonical || tag.is_none())
                    && ((style == ScalarStyle::Plain && implicit.0)
                        || (style != ScalarStyle::Plain && implicit.1))
                {
                    self.prepared_tag = None;
                    return Ok(());
                }
                if implicit.0 && tag.is_none() {
                    self.prepared_tag = None;
                    Some("!")
                } else {
                    tag.as_deref()
                }
            }
            EventKind::SequenceStart { tag, implicit, .. }
            | EventKind::MappingStart { tag, implicit, .. } => {
                if (!self.canonical || tag.is_none()) && *implicit {
                    self.prepared_tag = None;
                    return Ok(());
                }
                tag.as_deref()
            }
            _ => None,
        };
        let Some(tag) = tag else {
            return Err(emitter_error(Msg::text("tag is not specified")));
        };
        let prepared = match self.prepared_tag.take() {
            Some(prepared) => prepared,
            None => self.prepare_tag(tag)?,
        };
        if !prepared.is_empty() {
            self.write_indicator(&prepared, true, false, false);
        }
        Ok(())
    }

    /// `self.analysis`, analyzing `value` first if it is not cached yet.
    fn analysis_of(&mut self, value: &str) -> ScalarAnalysis {
        if let Some(analysis) = self.analysis {
            analysis
        } else {
            let analysis = analyze_scalar(value, self.allow_unicode);
            self.analysis = Some(analysis);
            analysis
        }
    }

    fn choose_scalar_style(&mut self, event: &Event) -> ScalarStyle {
        let EventKind::Scalar {
            implicit,
            value,
            style,
            ..
        } = &event.kind
        else {
            return ScalarStyle::DoubleQuoted;
        };
        let analysis = self.analysis_of(value);
        let style = *style;
        if style == Some('"') || self.canonical {
            return ScalarStyle::DoubleQuoted;
        }
        let simple_key = self.context == Context::SimpleKey;
        if style.is_none()
            && implicit.0
            && !(simple_key && (analysis.empty || analysis.multiline))
            && ((self.flow_level > 0 && analysis.allow_flow_plain)
                || (self.flow_level == 0 && analysis.allow_block_plain))
        {
            return ScalarStyle::Plain;
        }
        if let Some(block @ ('|' | '>')) = style {
            if self.flow_level == 0 && !simple_key && analysis.allow_block {
                return if block == '|' {
                    ScalarStyle::Literal
                } else {
                    ScalarStyle::Folded
                };
            }
        }
        if (style.is_none() || style == Some('\''))
            && analysis.allow_single_quoted
            && !(simple_key && analysis.multiline)
        {
            return ScalarStyle::SingleQuoted;
        }
        ScalarStyle::DoubleQuoted
    }

    fn process_scalar(&mut self, event: &Event) {
        let EventKind::Scalar { value, .. } = &event.kind else {
            return;
        };
        let analysis = self.analysis_of(value);
        let style = match self.style {
            Some(style) => style,
            None => self.choose_scalar_style(event),
        };
        let split = self.context != Context::SimpleKey;
        match style {
            ScalarStyle::DoubleQuoted => self.write_double_quoted(value, split, analysis.length),
            ScalarStyle::SingleQuoted => self.write_single_quoted(value, split),
            ScalarStyle::Folded => self.write_folded(value),
            ScalarStyle::Literal => self.write_literal(value),
            ScalarStyle::Plain => self.write_plain(value, split),
        }
        self.analysis = None;
        self.style = None;
    }

    // Analyzers.

    /// `self.tag_prefixes[prefix] = handle`.
    fn set_tag_prefix(&mut self, prefix: &str, handle: &str) {
        match self
            .tag_prefixes
            .binary_search_by(|(p, _)| p.as_ref().cmp(prefix))
        {
            Ok(i) => self.tag_prefixes[i].1 = Cow::Owned(handle.to_owned()),
            Err(i) => self.tag_prefixes.insert(
                i,
                (Cow::Owned(prefix.to_owned()), Cow::Owned(handle.to_owned())),
            ),
        }
    }

    fn prepare_tag(&self, tag: &str) -> Result<String, YamlError> {
        if tag.is_empty() {
            return Err(emitter_error(Msg::text("tag must not be empty")));
        }
        if tag == "!" {
            return Ok(tag.to_owned());
        }
        let mut handle: Option<&str> = None;
        let mut suffix = tag;
        // The last match in sorted order wins.
        for (prefix, prefix_handle) in &self.tag_prefixes {
            if tag.starts_with(prefix.as_ref()) && (prefix == "!" || prefix.len() < tag.len()) {
                handle = Some(prefix_handle.as_ref());
                suffix = &tag[prefix.len()..];
            }
        }
        let bang_allowed = handle != Some("!");
        let named = handle.filter(|h| !h.is_empty());
        let mut text = String::with_capacity(suffix.len() + 4);
        text.push_str(named.unwrap_or("!<"));
        for ch in suffix.chars() {
            if is_uri_char(ch) || (ch == '!' && bang_allowed) {
                text.push(ch);
            } else {
                percent_encode(&mut text, ch);
            }
        }
        if named.is_none() {
            text.push('>');
        }
        Ok(text)
    }

    // Writers.

    /// One `self.stream.write(data)`.
    #[inline]
    fn write(&mut self, data: &str) {
        self.out.push_str(data);
        self.mark_write();
    }

    /// Record the end of a write whose text was just appended.
    #[inline]
    fn mark_write(&mut self) {
        if self.record_writes {
            self.writes.push(self.out.len());
        }
    }

    /// `self.column > self.best_width`.
    #[inline]
    fn past_width(&self, column: usize) -> bool {
        column as u64 > self.best_width
    }

    fn write_indicator(
        &mut self,
        indicator: &str,
        need_whitespace: bool,
        whitespace: bool,
        indention: bool,
    ) {
        if !self.whitespace && need_whitespace {
            self.out.push(' ');
            self.column += 1;
        }
        self.out.push_str(indicator);
        self.column += indicator.chars().count();
        self.mark_write();
        self.whitespace = whitespace;
        self.indention = self.indention && indention;
        self.open_ended = false;
    }

    fn write_indent(&mut self) {
        let indent = self.indent.unwrap_or(0);
        if !self.indention || self.column > indent || (self.column == indent && !self.whitespace) {
            self.write_line_break(None);
        }
        if self.column < indent {
            self.whitespace = true;
            self.out
                .extend(std::iter::repeat_n(' ', indent - self.column));
            self.column = indent;
            self.mark_write();
        }
    }

    /// `write_line_break(data)`: `None` writes `best_line_break`.
    fn write_line_break(&mut self, data: Option<char>) {
        self.whitespace = true;
        self.indention = true;
        self.column = 0;
        match data {
            None => self.out.push_str(self.best_line_break),
            Some(ch) => self.out.push(ch),
        }
        self.mark_write();
    }

    /// The line breaks of `text`, each written by itself: `'\n'` as `best_line_break`, the
    /// others as they are.
    fn write_breaks(&mut self, text: &str) {
        for br in text.chars() {
            self.write_line_break(if br == '\n' { None } else { Some(br) });
        }
    }

    fn write_version_directive(&mut self, version_text: &str) {
        self.out.push_str("%YAML ");
        self.out.push_str(version_text);
        self.mark_write();
        self.write_line_break(None);
    }

    fn write_tag_directive(&mut self, handle_text: &str, prefix_text: &str) {
        self.out.push_str("%TAG ");
        self.out.push_str(handle_text);
        self.out.push(' ');
        self.out.push_str(prefix_text);
        self.mark_write();
        self.write_line_break(None);
    }

    // Scalar streams. `start` and `end` are byte offsets into `text`; `start_n` and `end_n`
    // the same positions in code points, as PyYAML counts them.

    fn write_single_quoted(&mut self, text: &str, split: bool) {
        self.write_indicator("'", true, false, false);
        let mut spaces = false;
        let mut breaks = false;
        let (mut start, mut start_n, mut end_n) = (0, 0, 0);
        let mut chars = text.char_indices();
        loop {
            let (end, ch) = match chars.next() {
                Some((i, ch)) => (i, Some(ch)),
                None => (text.len(), None),
            };
            if spaces {
                if ch != Some(' ') {
                    // `end != len(text)` is `ch is not None`.
                    if start_n + 1 == end_n
                        && self.past_width(self.column)
                        && split
                        && start_n != 0
                        && ch.is_some()
                    {
                        self.write_indent();
                    } else {
                        self.write(&text[start..end]);
                        self.column += end_n - start_n;
                    }
                    start = end;
                    start_n = end_n;
                }
            } else if breaks {
                if !ch.is_some_and(is_break) {
                    if text.as_bytes().get(start) == Some(&b'\n') {
                        self.write_line_break(None);
                    }
                    self.write_breaks(&text[start..end]);
                    self.write_indent();
                    start = end;
                    start_n = end_n;
                }
            } else if ch.is_none_or(|ch| is_space_or_break(ch) || ch == '\'') && start_n < end_n {
                self.write(&text[start..end]);
                self.column += end_n - start_n;
                start = end;
                start_n = end_n;
            }
            if ch == Some('\'') {
                self.write("''");
                self.column += 2;
                start = end + 1;
                start_n = end_n + 1;
            }
            let Some(ch) = ch else { break };
            spaces = ch == ' ';
            breaks = is_break(ch);
            end_n += 1;
        }
        self.write_indicator("'", false, false, false);
    }

    fn write_double_quoted(&mut self, text: &str, split: bool, length: usize) {
        self.write_indicator("\"", true, false, false);
        let (mut start, mut start_n, mut end_n) = (0, 0, 0);
        let mut chars = text.char_indices();
        loop {
            let (end, ch) = match chars.next() {
                Some((i, ch)) => (i, Some(ch)),
                None => (text.len(), None),
            };
            if ch.is_none_or(|ch| needs_escape(ch, self.allow_unicode)) {
                if start_n < end_n {
                    self.write(&text[start..end]);
                    self.column += end_n - start_n;
                    start = end;
                    start_n = end_n;
                }
                if let Some(ch) = ch {
                    self.column += write_escape(&mut self.out, ch);
                    self.mark_write();
                    start = end + ch.len_utf8();
                    start_n = end_n + 1;
                }
            }
            // `start` can be one past `end` here, right after an escape: then
            // `text[start:end]` is empty and `end - start` is -1.
            if 0 < end_n
                && end_n + 1 < length
                && (ch == Some(' ') || start_n >= end_n)
                && (self.column + end_n) as u64 > (start_n as u64).saturating_add(self.best_width)
                && split
            {
                if start_n < end_n {
                    self.out.push_str(&text[start..end]);
                    self.column += end_n - start_n;
                    start = end;
                    start_n = end_n;
                }
                self.out.push('\\');
                self.column += 1;
                self.mark_write();
                self.write_indent();
                self.whitespace = false;
                self.indention = false;
                if text.as_bytes().get(start) == Some(&b' ') {
                    self.write("\\");
                    self.column += 1;
                }
            }
            if ch.is_none() {
                break;
            }
            end_n += 1;
        }
        self.write_indicator("\"", false, false, false);
    }

    fn determine_block_hints(&self, text: &str) -> String {
        let mut hints = String::new();
        let mut rev = text.chars().rev();
        if let (Some(first), Some(last)) = (text.chars().next(), rev.next()) {
            if is_space_or_break(first) {
                let _ = write!(hints, "{}", self.best_indent);
            }
            if !is_break(last) {
                hints.push('-');
            } else if rev.next().is_none_or(is_break) {
                // `len(text) == 1 or text[-2] in '\n\x85\u2028\u2029'`
                hints.push('+');
            }
        }
        hints
    }

    /// `write_indicator(indicator + hints, True)` for a block scalar, setting `open_ended`
    /// for a `+` hint, and the line break after it.
    fn write_block_header(&mut self, indicator: char, text: &str) {
        let hints = self.determine_block_hints(text);
        let mut data = String::with_capacity(hints.len() + 1);
        data.push(indicator);
        data.push_str(&hints);
        self.write_indicator(&data, true, false, false);
        if hints.ends_with('+') {
            self.open_ended = true;
        }
        self.write_line_break(None);
    }

    fn write_folded(&mut self, text: &str) {
        self.write_block_header('>', text);
        let mut leading_space = true;
        let mut spaces = false;
        let mut breaks = true;
        let (mut start, mut start_n, mut end_n) = (0, 0, 0);
        let mut chars = text.char_indices();
        loop {
            let (end, ch) = match chars.next() {
                Some((i, ch)) => (i, Some(ch)),
                None => (text.len(), None),
            };
            if breaks {
                if !ch.is_some_and(is_break) {
                    if !leading_space
                        && ch.is_some()
                        && ch != Some(' ')
                        && text.as_bytes().get(start) == Some(&b'\n')
                    {
                        self.write_line_break(None);
                    }
                    leading_space = ch == Some(' ');
                    self.write_breaks(&text[start..end]);
                    if ch.is_some() {
                        self.write_indent();
                    }
                    start = end;
                    start_n = end_n;
                }
            } else if spaces {
                if ch != Some(' ') {
                    if start_n + 1 == end_n && self.past_width(self.column) {
                        self.write_indent();
                    } else {
                        self.write(&text[start..end]);
                        self.column += end_n - start_n;
                    }
                    start = end;
                    start_n = end_n;
                }
            } else if ch.is_none_or(is_space_or_break) {
                self.write(&text[start..end]);
                self.column += end_n - start_n;
                if ch.is_none() {
                    self.write_line_break(None);
                }
                start = end;
                start_n = end_n;
            }
            let Some(ch) = ch else { break };
            breaks = is_break(ch);
            spaces = ch == ' ';
            end_n += 1;
        }
    }

    fn write_literal(&mut self, text: &str) {
        self.write_block_header('|', text);
        let mut breaks = true;
        let mut start = 0;
        let mut chars = text.char_indices();
        loop {
            let (end, ch) = match chars.next() {
                Some((i, ch)) => (i, Some(ch)),
                None => (text.len(), None),
            };
            if breaks {
                if !ch.is_some_and(is_break) {
                    self.write_breaks(&text[start..end]);
                    if ch.is_some() {
                        self.write_indent();
                    }
                    start = end;
                }
            } else if ch.is_none_or(is_break) {
                // PyYAML does not advance the column here.
                self.write(&text[start..end]);
                if ch.is_none() {
                    self.write_line_break(None);
                }
                start = end;
            }
            let Some(ch) = ch else { break };
            breaks = is_break(ch);
        }
    }

    fn write_plain(&mut self, text: &str, split: bool) {
        if self.context == Context::Root {
            self.open_ended = true;
        }
        if text.is_empty() {
            return;
        }
        if !self.whitespace {
            self.write(" ");
            self.column += 1;
        }
        self.whitespace = false;
        self.indention = false;
        let mut spaces = false;
        let mut breaks = false;
        let (mut start, mut start_n, mut end_n) = (0, 0, 0);
        let mut chars = text.char_indices();
        loop {
            let (end, ch) = match chars.next() {
                Some((i, ch)) => (i, Some(ch)),
                None => (text.len(), None),
            };
            if spaces {
                if ch != Some(' ') {
                    if start_n + 1 == end_n && self.past_width(self.column) && split {
                        self.write_indent();
                        self.whitespace = false;
                        self.indention = false;
                    } else {
                        self.write(&text[start..end]);
                        self.column += end_n - start_n;
                    }
                    start = end;
                    start_n = end_n;
                }
            } else if breaks {
                // At the end of the text PyYAML's `None not in '...'` raises a TypeError;
                // plain scalars never contain line breaks, so that is never reached.
                if !ch.is_some_and(is_break) {
                    if text.as_bytes().get(start) == Some(&b'\n') {
                        self.write_line_break(None);
                    }
                    self.write_breaks(&text[start..end]);
                    self.write_indent();
                    self.whitespace = false;
                    self.indention = false;
                    start = end;
                    start_n = end_n;
                }
            } else if ch.is_none_or(is_space_or_break) {
                self.write(&text[start..end]);
                self.column += end_n - start_n;
                start = end;
                start_n = end_n;
            }
            let Some(ch) = ch else { break };
            spaces = ch == ' ';
            breaks = is_break(ch);
            end_n += 1;
        }
    }
}

/// The anchor of a `NodeEvent` (alias, scalar, collection start).
fn node_anchor(event: &Event) -> Option<&str> {
    match &event.kind {
        EventKind::Alias { anchor } => Some(anchor),
        EventKind::Scalar { anchor, .. }
        | EventKind::SequenceStart { anchor, .. }
        | EventKind::MappingStart { anchor, .. } => anchor.as_deref(),
        _ => None,
    }
}

/// The tag of a scalar or collection start event.
fn node_tag(event: &Event) -> Option<&str> {
    match &event.kind {
        EventKind::Scalar { tag, .. }
        | EventKind::SequenceStart { tag, .. }
        | EventKind::MappingStart { tag, .. } => tag.as_deref(),
        _ => None,
    }
}

fn prepare_version(major: u64, minor: u64) -> Result<String, YamlError> {
    if major != 1 {
        // A version past i64::MAX cannot come from the parser or a Python int that fits u64
        // in practice; it would print as i64::MAX.
        let int = |n: u64| Arg::Int(i64::try_from(n).unwrap_or(i64::MAX));
        return Err(emitter_error(Msg {
            fmt: "unsupported YAML version: %d.%d",
            args: vec![int(major), int(minor)],
        }));
    }
    Ok(format!("{major}.{minor}"))
}

fn prepare_tag_handle(handle: &str) -> Result<(), YamlError> {
    if handle.is_empty() {
        return Err(emitter_error(Msg::text("tag handle must not be empty")));
    }
    if !handle.starts_with('!') || !handle.ends_with('!') {
        return Err(emitter_error(Msg::str(
            "tag handle must start and end with '!': %r",
            handle,
        )));
    }
    // `handle[1:-1]` ('!' is one byte; for the handle "!" it is empty).
    let inner = if handle.len() >= 2 {
        &handle[1..handle.len() - 1]
    } else {
        ""
    };
    for ch in inner.chars() {
        if !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '_') {
            return Err(emitter_error(Msg {
                fmt: "invalid character %r in the tag handle: %r",
                args: vec![Arg::Str(ch.to_string()), Arg::Str(handle.to_owned())],
            }));
        }
    }
    Ok(())
}

/// `prepare_tag_prefix` returns the prefix unchanged when every character is allowed, and
/// fails with a `TypeError` on the first one it would percent-encode.
fn prepare_tag_prefix(prefix: &str) -> Result<(), YamlError> {
    if prefix.is_empty() {
        return Err(emitter_error(Msg::text("tag prefix must not be empty")));
    }
    if prefix.chars().all(|ch| is_uri_char(ch) || ch == '!') {
        Ok(())
    } else {
        Err(tag_prefix_type_error())
    }
}

fn prepare_anchor(anchor: &str) -> Result<(), YamlError> {
    if anchor.is_empty() {
        return Err(emitter_error(Msg::text("anchor must not be empty")));
    }
    for ch in anchor.chars() {
        if !(ch.is_ascii_alphanumeric() || ch == '-' || ch == '_') {
            return Err(emitter_error(Msg {
                fmt: "invalid character %r in the anchor: %r",
                args: vec![Arg::Str(ch.to_string()), Arg::Str(anchor.to_owned())],
            }));
        }
    }
    Ok(())
}

/// The characters `prepare_tag` writes as they are (besides `!` when the handle is not `!`):
/// `'0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or ch in "-;/?:@&=+$,_.~*'()[]"`.
fn is_uri_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || "-;/?:@&=+$,_.~*'()[]".contains(ch)
}

/// `'%%%02X' % byte` for each UTF-8 byte of `ch`.
fn percent_encode(out: &mut String, ch: char) {
    let mut buf = [0; 4];
    for byte in ch.encode_utf8(&mut buf).bytes() {
        let _ = write!(out, "%{byte:02X}");
    }
}

/// `ch in '\n\x85\u2028\u2029'`.
#[inline]
fn is_break(ch: char) -> bool {
    matches!(ch, '\n' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// `ch in ' \n\x85\u2028\u2029'`.
#[inline]
fn is_space_or_break(ch: char) -> bool {
    ch == ' ' || is_break(ch)
}

/// `ch in '\0 \t\r\n\x85\u2028\u2029'`.
#[inline]
fn is_whitespace_or_nul(ch: char) -> bool {
    matches!(
        ch,
        '\0' | ' ' | '\t' | '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

#[allow(clippy::too_many_lines)] // one method in PyYAML, kept whole to compare line by line
fn analyze_scalar(scalar: &str, allow_unicode: bool) -> ScalarAnalysis {
    // Empty scalar is a special case.
    if scalar.is_empty() {
        return ScalarAnalysis {
            length: 0,
            empty: true,
            multiline: false,
            allow_flow_plain: false,
            allow_block_plain: true,
            allow_single_quoted: true,
            allow_block: false,
        };
    }

    // Indicators and special characters.
    let mut block_indicators = false;
    let mut flow_indicators = false;
    let mut line_breaks = false;
    let mut special_characters = false;

    // Important whitespace combinations.
    let mut leading_space = false;
    let mut leading_break = false;
    let mut trailing_space = false;
    let mut trailing_break = false;
    let mut break_space = false;
    let mut space_break = false;

    // Check document indicators.
    if scalar.starts_with("---") || scalar.starts_with("...") {
        block_indicators = true;
        flow_indicators = true;
    }

    // First character or preceded by a whitespace.
    let mut preceded_by_whitespace = true;
    // The previous character is a space; the previous character is a break.
    let mut previous_space = false;
    let mut previous_break = false;

    let mut chars = scalar.chars();
    let mut next = chars.next();
    let mut index = 0;
    while let Some(ch) = next {
        next = chars.next();
        // Last character or followed by a whitespace.
        let followed_by_whitespace = next.is_none_or(is_whitespace_or_nul);
        let last = next.is_none();

        // Check for indicators.
        if index == 0 {
            // Leading indicators are special characters.
            if "#,[]{}&*!|>'\"%@`".contains(ch) {
                flow_indicators = true;
                block_indicators = true;
            }
            if ch == '?' || ch == ':' {
                flow_indicators = true;
                if followed_by_whitespace {
                    block_indicators = true;
                }
            }
            if ch == '-' && followed_by_whitespace {
                flow_indicators = true;
                block_indicators = true;
            }
        } else {
            // Some indicators cannot appear within a scalar as well.
            if ",?[]{}".contains(ch) {
                flow_indicators = true;
            }
            if ch == ':' {
                flow_indicators = true;
                if followed_by_whitespace {
                    block_indicators = true;
                }
            }
            if ch == '#' && preceded_by_whitespace {
                flow_indicators = true;
                block_indicators = true;
            }
        }

        // Check for line breaks, special, and unicode characters.
        if is_break(ch) {
            line_breaks = true;
        }
        if !(ch == '\n' || ('\x20'..='\x7E').contains(&ch)) {
            if (ch == '\u{85}'
                || ('\u{A0}'..='\u{D7FF}').contains(&ch)
                || ('\u{E000}'..='\u{FFFD}').contains(&ch)
                || ('\u{10000}'..'\u{10FFFF}').contains(&ch))
                && ch != '\u{FEFF}'
            {
                if !allow_unicode {
                    special_characters = true;
                }
            } else {
                special_characters = true;
            }
        }

        // Detect important whitespace combinations.
        if ch == ' ' {
            if index == 0 {
                leading_space = true;
            }
            if last {
                trailing_space = true;
            }
            if previous_break {
                break_space = true;
            }
            previous_space = true;
            previous_break = false;
        } else if is_break(ch) {
            if index == 0 {
                leading_break = true;
            }
            if last {
                trailing_break = true;
            }
            if previous_space {
                space_break = true;
            }
            previous_space = false;
            previous_break = true;
        } else {
            previous_space = false;
            previous_break = false;
        }

        // Prepare for the next character.
        index += 1;
        preceded_by_whitespace = is_whitespace_or_nul(ch);
    }

    // Let's decide what styles are allowed.
    let mut allow_flow_plain = true;
    let mut allow_block_plain = true;
    let mut allow_single_quoted = true;
    let mut allow_block = true;

    // Leading and trailing whitespaces are bad for plain scalars.
    if leading_space || leading_break || trailing_space || trailing_break {
        allow_flow_plain = false;
        allow_block_plain = false;
    }

    // We do not permit trailing spaces for block scalars.
    if trailing_space {
        allow_block = false;
    }

    // Spaces at the beginning of a new line are only acceptable for block scalars.
    if break_space {
        allow_flow_plain = false;
        allow_block_plain = false;
        allow_single_quoted = false;
    }

    // Spaces followed by breaks, as well as special character are only allowed for double
    // quoted scalars.
    if space_break || special_characters {
        allow_flow_plain = false;
        allow_block_plain = false;
        allow_single_quoted = false;
        allow_block = false;
    }

    // Although the plain scalar writer supports breaks, we never emit multiline plain
    // scalars.
    if line_breaks {
        allow_flow_plain = false;
        allow_block_plain = false;
    }

    // Flow indicators are forbidden for flow plain scalars.
    if flow_indicators {
        allow_flow_plain = false;
    }

    // Block indicators are forbidden for block plain scalars.
    if block_indicators {
        allow_block_plain = false;
    }

    ScalarAnalysis {
        length: index,
        empty: false,
        multiline: line_breaks,
        allow_flow_plain,
        allow_block_plain,
        allow_single_quoted,
        allow_block,
    }
}

/// Whether `write_double_quoted` escapes `ch`: `ch in '"\\\x85\u2028\u2029\uFEFF'`, or `ch`
/// is not printable ASCII and not (with `allow_unicode`) in `\xA0-\uD7FF` or `\uE000-\uFFFD`.
#[inline]
fn needs_escape(ch: char, allow_unicode: bool) -> bool {
    matches!(
        ch,
        '"' | '\\' | '\u{85}' | '\u{2028}' | '\u{2029}' | '\u{FEFF}'
    ) || !(('\x20'..='\x7E').contains(&ch)
        || (allow_unicode
            && (('\u{A0}'..='\u{D7FF}').contains(&ch) || ('\u{E000}'..='\u{FFFD}').contains(&ch))))
}

/// Append the double-quoted escape of `ch` (`ESCAPE_REPLACEMENTS`, else `\xXX`, `\uXXXX` or
/// `\UXXXXXXXX`) and return its length.
fn write_escape(out: &mut String, ch: char) -> usize {
    let short = match ch {
        '\0' => '0',
        '\x07' => 'a',
        '\x08' => 'b',
        '\x09' => 't',
        '\x0A' => 'n',
        '\x0B' => 'v',
        '\x0C' => 'f',
        '\x0D' => 'r',
        '\x1B' => 'e',
        '"' => '"',
        '\\' => '\\',
        '\u{85}' => 'N',
        '\u{A0}' => '_',
        '\u{2028}' => 'L',
        '\u{2029}' => 'P',
        _ => {
            let code = u32::from(ch);
            return if code <= 0xFF {
                let _ = write!(out, "\\x{code:02X}");
                4
            } else if code <= 0xFFFF {
                let _ = write!(out, "\\u{code:04X}");
                6
            } else {
                let _ = write!(out, "\\U{code:08X}");
                10
            };
        }
    };
    out.push('\\');
    out.push(short);
    2
}

/// The formats of one "expected ..., but got %s" message, with `%s` of the event (PyYAML's
/// `Event.__repr__`) spelled out, so that Python's `%r` formats the event's strings. The
/// arrays are indexed by `anchor.is_some() | tag.is_some() << 1`; `implicit` is a `%s` of
/// `"True"`/`"False"`.
#[derive(Debug)]
struct GotFormats {
    stream_start: &'static str,
    stream_end: &'static str,
    document_start: &'static str,
    document_end: &'static str,
    alias: &'static str,
    scalar: [&'static str; 4],
    sequence_start: [&'static str; 4],
    sequence_end: &'static str,
    mapping_start: [&'static str; 4],
    mapping_end: &'static str,
}

macro_rules! got_formats {
    ($p:literal) => {
        GotFormats {
            stream_start: concat!($p, "StreamStartEvent()"),
            stream_end: concat!($p, "StreamEndEvent()"),
            document_start: concat!($p, "DocumentStartEvent()"),
            document_end: concat!($p, "DocumentEndEvent()"),
            alias: concat!($p, "AliasEvent(anchor=%r)"),
            scalar: [
                concat!(
                    $p,
                    "ScalarEvent(anchor=None, tag=None, implicit=(%s, %s), value=%r)"
                ),
                concat!(
                    $p,
                    "ScalarEvent(anchor=%r, tag=None, implicit=(%s, %s), value=%r)"
                ),
                concat!(
                    $p,
                    "ScalarEvent(anchor=None, tag=%r, implicit=(%s, %s), value=%r)"
                ),
                concat!(
                    $p,
                    "ScalarEvent(anchor=%r, tag=%r, implicit=(%s, %s), value=%r)"
                ),
            ],
            sequence_start: [
                concat!($p, "SequenceStartEvent(anchor=None, tag=None, implicit=%s)"),
                concat!($p, "SequenceStartEvent(anchor=%r, tag=None, implicit=%s)"),
                concat!($p, "SequenceStartEvent(anchor=None, tag=%r, implicit=%s)"),
                concat!($p, "SequenceStartEvent(anchor=%r, tag=%r, implicit=%s)"),
            ],
            sequence_end: concat!($p, "SequenceEndEvent()"),
            mapping_start: [
                concat!($p, "MappingStartEvent(anchor=None, tag=None, implicit=%s)"),
                concat!($p, "MappingStartEvent(anchor=%r, tag=None, implicit=%s)"),
                concat!($p, "MappingStartEvent(anchor=None, tag=%r, implicit=%s)"),
                concat!($p, "MappingStartEvent(anchor=%r, tag=%r, implicit=%s)"),
            ],
            mapping_end: concat!($p, "MappingEndEvent()"),
        }
    };
}

static EXPECTED_STREAM_START: GotFormats = got_formats!("expected StreamStartEvent, but got ");
static EXPECTED_NOTHING: GotFormats = got_formats!("expected nothing, but got ");
static EXPECTED_DOCUMENT_START: GotFormats = got_formats!("expected DocumentStartEvent, but got ");
static EXPECTED_DOCUMENT_END: GotFormats = got_formats!("expected DocumentEndEvent, but got ");
static EXPECTED_NODE: GotFormats = got_formats!("expected NodeEvent, but got ");

/// `EmitterError("expected ..., but got %s" % event)`.
fn event_error(formats: &GotFormats, event: &Event) -> YamlError {
    fn py_bool(b: bool) -> Arg {
        Arg::Str(if b { "True" } else { "False" }.to_owned())
    }
    /// The anchor and tag arguments, and the index of the format that has them.
    fn node(args: &mut Vec<Arg>, anchor: Option<&String>, tag: Option<&String>) -> usize {
        args.extend(anchor.map(|a| Arg::Str(a.clone())));
        args.extend(tag.map(|t| Arg::Str(t.clone())));
        usize::from(anchor.is_some()) | usize::from(tag.is_some()) << 1
    }
    let mut args = Vec::new();
    let fmt = match &event.kind {
        EventKind::StreamStart { .. } => formats.stream_start,
        EventKind::StreamEnd => formats.stream_end,
        EventKind::DocumentStart { .. } => formats.document_start,
        EventKind::DocumentEnd { .. } => formats.document_end,
        EventKind::Alias { anchor } => {
            args.push(Arg::Str(anchor.clone()));
            formats.alias
        }
        EventKind::Scalar {
            anchor,
            tag,
            implicit,
            value,
            ..
        } => {
            let i = node(&mut args, anchor.as_ref(), tag.as_ref());
            args.extend([
                py_bool(implicit.0),
                py_bool(implicit.1),
                Arg::Str(value.clone()),
            ]);
            formats.scalar[i]
        }
        EventKind::SequenceStart {
            anchor,
            tag,
            implicit,
            ..
        } => {
            let i = node(&mut args, anchor.as_ref(), tag.as_ref());
            args.push(py_bool(*implicit));
            formats.sequence_start[i]
        }
        EventKind::SequenceEnd => formats.sequence_end,
        EventKind::MappingStart {
            anchor,
            tag,
            implicit,
            ..
        } => {
            let i = node(&mut args, anchor.as_ref(), tag.as_ref());
            args.push(py_bool(*implicit));
            formats.mapping_start[i]
        }
        EventKind::MappingEnd => formats.mapping_end,
    };
    YamlError::Emitter(Msg { fmt, args })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    use std::io::BufRead as _;

    /// Python's `repr(s)` for ASCII text (non-ASCII is kept as it is; the fixtures carry
    /// Python's own `repr` for the strings they use).
    fn py_repr(out: &mut String, s: &str) {
        let quote = if s.contains('\'') && !s.contains('"') {
            '"'
        } else {
            '\''
        };
        out.push(quote);
        for ch in s.chars() {
            match ch {
                '\\' => out.push_str("\\\\"),
                '\t' => out.push_str("\\t"),
                '\n' => out.push_str("\\n"),
                '\r' => out.push_str("\\r"),
                _ if ch == quote => {
                    out.push('\\');
                    out.push(ch);
                }
                _ if ch < ' ' || ch == '\x7f' => {
                    let _ = write!(out, "\\x{:02x}", u32::from(ch));
                }
                _ => out.push(ch),
            }
        }
        out.push(quote);
    }

    /// `fmt % tuple(args)` as Python formats a [`Msg`]; `%r` of a str takes Python's `repr`
    /// from `reprs` when the fixture carries it.
    fn py_format(msg: &Msg, reprs: Option<&serde_json::Map<String, Value>>) -> String {
        if msg.args.is_empty() {
            return msg.fmt.to_owned();
        }
        let mut out = String::new();
        let mut args = msg.args.iter();
        let mut chars = msg.fmt.chars();
        while let Some(c) = chars.next() {
            if c != '%' {
                out.push(c);
                continue;
            }
            match (chars.next(), args.next()) {
                (Some('%'), _) => out.push('%'),
                (Some('r'), Some(Arg::Str(s))) => {
                    match reprs
                        .and_then(|r| r.get(s.as_str()))
                        .and_then(Value::as_str)
                    {
                        Some(r) => out.push_str(r),
                        None => py_repr(&mut out, s),
                    }
                }
                (Some('s'), Some(Arg::Str(s))) => out.push_str(s),
                (Some('d'), Some(Arg::Int(n))) => out.push_str(&n.to_string()),
                other => panic!("unexpected format {other:?} in {msg:?}"),
            }
        }
        out
    }

    fn opt_str(v: &Value) -> Option<String> {
        v.as_str().map(str::to_owned)
    }

    fn json_event(v: &Value) -> Event {
        let kind = match v["t"].as_str().unwrap() {
            "StreamStart" => EventKind::StreamStart {
                encoding: opt_str(&v["encoding"]),
            },
            "StreamEnd" => EventKind::StreamEnd,
            "DocumentStart" => EventKind::DocumentStart {
                explicit: v["explicit"].as_bool().unwrap(),
                version: v["version"]
                    .as_array()
                    .map(|a| (a[0].as_u64().unwrap(), a[1].as_u64().unwrap())),
                tags: v["tags"].as_array().map(|a| {
                    a.iter()
                        .map(|p| (opt_str(&p[0]).unwrap(), opt_str(&p[1]).unwrap()))
                        .collect()
                }),
            },
            "DocumentEnd" => EventKind::DocumentEnd {
                explicit: v["explicit"].as_bool().unwrap(),
            },
            "Alias" => EventKind::Alias {
                anchor: opt_str(&v["anchor"]).unwrap(),
            },
            "Scalar" => EventKind::Scalar {
                anchor: opt_str(&v["anchor"]),
                tag: opt_str(&v["tag"]),
                implicit: (
                    v["implicit"][0].as_bool().unwrap(),
                    v["implicit"][1].as_bool().unwrap(),
                ),
                value: opt_str(&v["value"]).unwrap(),
                style: v["style"].as_str().map(|s| s.chars().next().unwrap()),
            },
            t @ ("SequenceStart" | "MappingStart") => {
                let anchor = opt_str(&v["anchor"]);
                let tag = opt_str(&v["tag"]);
                let implicit = v["implicit"].as_bool().unwrap();
                let flow_style = v["flow_style"].as_bool();
                if t == "SequenceStart" {
                    EventKind::SequenceStart {
                        anchor,
                        tag,
                        implicit,
                        flow_style,
                    }
                } else {
                    EventKind::MappingStart {
                        anchor,
                        tag,
                        implicit,
                        flow_style,
                    }
                }
            }
            "SequenceEnd" => EventKind::SequenceEnd,
            "MappingEnd" => EventKind::MappingEnd,
            t => panic!("unknown event {t}"),
        };
        Event::new(kind)
    }

    fn json_options(v: &Value, record_writes: bool) -> EmitterOptions {
        EmitterOptions {
            canonical: v["canonical"].as_bool().unwrap(),
            allow_unicode: v["allow_unicode"].as_bool().unwrap(),
            best_indent: usize::try_from(v["best_indent"].as_u64().unwrap()).unwrap(),
            best_width: v["best_width"].as_u64().unwrap_or(u64::MAX),
            line_break: match v["line_break"].as_str().unwrap() {
                "\n" => "\n",
                "\r" => "\r",
                "\r\n" => "\r\n",
                other => panic!("line break {other:?}"),
            },
            record_writes,
        }
    }

    /// Feed `events` until the first error.
    fn run(options: EmitterOptions, events: Vec<Event>) -> (Emitter, Option<YamlError>) {
        let mut emitter = Emitter::new(options);
        for event in events {
            if let Err(e) = emitter.emit(event) {
                return (emitter, Some(e));
            }
        }
        (emitter, None)
    }

    /// Compare one fixture record; `None` when everything matches.
    fn check_record(rec: &Value) -> Option<String> {
        let events: Vec<Event> = rec["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(json_event)
            .collect();
        let (emitter, err) = run(json_options(&rec["options"], true), events.clone());
        let want = rec["output"].as_str().unwrap();
        if emitter.output() != want {
            return Some(format!(
                "output\n    want {want:?}\n    got  {:?}",
                emitter.output()
            ));
        }
        let want_writes: Vec<usize> = rec["writes"]
            .as_array()
            .unwrap()
            .iter()
            .map(|n| usize::try_from(n.as_u64().unwrap()).unwrap())
            .collect();
        if emitter.writes() != want_writes.as_slice() {
            let i = emitter
                .writes()
                .iter()
                .zip(&want_writes)
                .take_while(|(a, b)| a == b)
                .count();
            return Some(format!(
                "writes differ at #{i} ({} vs {} writes): want {:?} got {:?}",
                want_writes.len(),
                emitter.writes().len(),
                want_writes.get(i),
                emitter.writes().get(i)
            ));
        }
        let reprs = rec.get("reprs").and_then(Value::as_object);
        let got = match &err {
            None => None,
            Some(YamlError::Emitter(m)) => Some(("EmitterError".to_owned(), py_format(m, reprs))),
            Some(YamlError::Type(s)) if s == TAG_PREFIX_TYPE_ERROR => {
                Some(("TypeError".to_owned(), s.clone()))
            }
            Some(other) => Some(("other".to_owned(), format!("{other:?}"))),
        };
        let want_err = match &rec["error"] {
            Value::Null => None,
            e => Some((
                e["type"].as_str().unwrap().to_owned(),
                e["msg"].as_str().unwrap().to_owned(),
            )),
        };
        if got != want_err {
            return Some(format!("error\n    want {want_err:?}\n    got  {got:?}"));
        }
        // The same text without recording the writes.
        let (plain, _) = run(json_options(&rec["options"], false), events);
        if plain.output() != want || !plain.writes().is_empty() {
            return Some("output differs with record_writes off".to_owned());
        }
        None
    }

    /// Replay the fixtures of `.scratch/emitter/gen_fixtures.py` (comma-separated JSONL paths in
    /// `PYROID_YAML_EMITTER_FIXTURES`); skipped when the variable is unset.
    #[test]
    #[allow(clippy::print_stderr)]
    fn replay_fixtures() {
        let Ok(paths) = std::env::var("PYROID_YAML_EMITTER_FIXTURES") else {
            return;
        };
        let (mut total, mut failed) = (0, 0);
        let mut report = Vec::new();
        for path in paths.split(',').filter(|p| !p.is_empty()) {
            let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
            let (mut n, mut bad) = (0, 0);
            for (i, line) in std::io::BufReader::new(file).lines().enumerate() {
                let line = line.unwrap();
                if line.is_empty() {
                    continue;
                }
                let rec: Value = serde_json::from_str(&line).unwrap();
                n += 1;
                if let Some(why) = check_record(&rec) {
                    bad += 1;
                    if report.len() < 40 {
                        report.push(format!(
                            "{path}:{}: {} {}: {why}",
                            i + 1,
                            rec["src"],
                            rec["raw_options"]
                        ));
                    }
                }
            }
            eprintln!("{path}: {n} records, {bad} mismatches");
            total += n;
            failed += bad;
        }
        eprintln!("emitter fixtures: {total} records, {failed} mismatches");
        assert!(
            failed == 0,
            "{failed} of {total} records differ:\n{}",
            report.join("\n")
        );
    }

    // Hand-checked cases; the expected text is PyYAML 6.0.3's `yaml.emit` output for the same
    // events (.scratch/emitter/expect.py).

    fn plain(value: &str) -> EventKind {
        EventKind::Scalar {
            anchor: None,
            tag: None,
            implicit: (true, false),
            value: value.to_owned(),
            style: None,
        }
    }

    fn styled(value: &str, style: char) -> EventKind {
        EventKind::Scalar {
            anchor: None,
            tag: None,
            implicit: (true, false),
            value: value.to_owned(),
            style: Some(style),
        }
    }

    fn tagged(value: &str, tag: &str) -> EventKind {
        EventKind::Scalar {
            anchor: None,
            tag: Some(tag.to_owned()),
            implicit: (false, false),
            value: value.to_owned(),
            style: None,
        }
    }

    fn seq(flow: Option<bool>, items: Vec<EventKind>) -> Vec<EventKind> {
        let mut out = vec![EventKind::SequenceStart {
            anchor: None,
            tag: None,
            implicit: true,
            flow_style: flow,
        }];
        out.extend(items);
        out.push(EventKind::SequenceEnd);
        out
    }

    fn map(flow: Option<bool>, items: Vec<EventKind>) -> Vec<EventKind> {
        let mut out = vec![EventKind::MappingStart {
            anchor: None,
            tag: None,
            implicit: true,
            flow_style: flow,
        }];
        out.extend(items);
        out.push(EventKind::MappingEnd);
        out
    }

    fn doc_with(start: EventKind, body: Vec<EventKind>) -> Vec<EventKind> {
        let mut out = vec![EventKind::StreamStart { encoding: None }, start];
        out.extend(body);
        out.extend([
            EventKind::DocumentEnd { explicit: false },
            EventKind::StreamEnd,
        ]);
        out
    }

    fn doc(body: Vec<EventKind>) -> Vec<EventKind> {
        doc_with(
            EventKind::DocumentStart {
                explicit: false,
                version: None,
                tags: None,
            },
            body,
        )
    }

    fn emit_with(options: EmitterOptions, kinds: Vec<EventKind>) -> Result<String, YamlError> {
        let mut emitter = Emitter::new(options);
        for kind in kinds {
            emitter.emit(Event::new(kind))?;
        }
        Ok(emitter.take_output())
    }

    fn emit(options: EmitterOptions, kinds: Vec<EventKind>) -> String {
        emit_with(options, kinds).unwrap()
    }

    fn error(kinds: Vec<EventKind>) -> String {
        match emit_with(EmitterOptions::default(), kinds) {
            Err(YamlError::Emitter(m)) => py_format(&m, None),
            Err(YamlError::Type(s)) => format!("TypeError: {s}"),
            other => panic!("{other:?}"),
        }
    }

    fn ab_xy(flow: Option<bool>) -> Vec<EventKind> {
        let mut body = vec![plain("a"), plain("1"), plain("b")];
        body.extend(seq(None, vec![plain("x"), plain("y")]));
        doc(map(flow, body))
    }

    fn with(f: impl FnOnce(&mut EmitterOptions)) -> EmitterOptions {
        let mut options = EmitterOptions::default();
        f(&mut options);
        options
    }

    #[test]
    fn collections() {
        let d = EmitterOptions::default();
        assert_eq!(emit(d, ab_xy(None)), "a: 1\nb:\n- x\n- y\n");
        assert_eq!(emit(d, ab_xy(Some(true))), "{a: 1, b: [x, y]}\n");
        assert_eq!(
            emit(with(|o| o.canonical = true), ab_xy(None)),
            "---\n{\n  ? ! \"a\"\n  : ! \"1\",\n  ? ! \"b\"\n  : [\n    ! \"x\",\n    ! \"y\",\n  ],\n}\n"
        );
        let mut body = seq(None, vec![plain("a"), plain("b")]);
        body.extend([plain("v"), plain(&"k".repeat(130)), plain("w")]);
        assert_eq!(
            emit(d, doc(map(None, body))),
            format!("? - a\n  - b\n: v\n? {}\n: w\n", "k".repeat(130))
        );
    }

    #[test]
    fn scalar_styles() {
        let values = [
            "'x", "a\tb", "", " a", "a\nb", "-", "x: y", "#c", "\u{E9}", "a #b", "it's",
        ];
        assert_eq!(
            emit(
                EmitterOptions::default(),
                doc(seq(None, values.map(plain).to_vec()))
            ),
            "- ! '''x'\n- ! \"a\\tb\"\n-\n- ! ' a'\n- ! 'a\n\n  b'\n- ! '-'\n- ! 'x: y'\n- ! '#c'\n\
             - ! \"\\xE9\"\n- ! 'a #b'\n- it's\n"
        );
        let values = ["\u{E9}", "\u{1F600}", "\u{FEFF}", "\u{85}", "\u{2028}"];
        assert_eq!(
            emit(
                with(|o| o.allow_unicode = true),
                doc(seq(None, values.map(plain).to_vec()))
            ),
            "- \u{E9}\n- \u{1F600}\n- ! \"\\uFEFF\"\n- ! '\u{85}  '\n- ! '\u{2028}  '\n"
        );
    }

    #[test]
    fn block_scalars() {
        let body = vec![
            plain("a"),
            styled("x\ny\n", '|'),
            plain("b"),
            styled(
                "one two three four five six seven eight nine ten eleven twelve",
                '>',
            ),
            plain("c"),
            styled("keep\n\n", '|'),
            plain("d"),
            styled(" lead\nx", '>'),
        ];
        assert_eq!(
            emit(with(|o| o.best_width = 20), doc(map(None, body))),
            "a: ! |\n  x\n  y\nb: ! >-\n  one two three four five\n  six seven eight nine\n  \
             ten eleven twelve\nc: ! |+\n  keep\n\nd: ! >2-\n   lead\n  x\n"
        );
        let body = vec![plain("a"), plain("1"), plain("b"), styled("x\ny", '|')];
        let options = with(|o| {
            o.line_break = "\r\n";
            o.best_indent = 3;
        });
        assert_eq!(
            emit(options, doc(map(None, body))),
            "a: 1\r\nb: ! |-\r\n   x\r\n   y\r\n"
        );
    }

    #[test]
    fn open_ended_documents() {
        let d = EmitterOptions::default();
        assert_eq!(emit(d, doc(vec![plain("x")])), "x\n...\n");
        assert_eq!(
            emit(d, doc(vec![styled("x\n\n", '|')])),
            "! |+\n  x\n\n...\n"
        );
        let start = |version| EventKind::DocumentStart {
            explicit: false,
            version,
            tags: None,
        };
        let mut events = vec![
            EventKind::StreamStart { encoding: None },
            start(None),
            plain("a"),
        ];
        events.extend([
            EventKind::DocumentEnd { explicit: false },
            start(None),
            styled("b\n", '|'),
        ]);
        events.extend([
            EventKind::DocumentEnd { explicit: true },
            start(Some((1, 1))),
        ]);
        events.extend(map(None, vec![]));
        events.extend([
            EventKind::DocumentEnd { explicit: false },
            EventKind::StreamEnd,
        ]);
        assert_eq!(emit(d, events), "a\n--- ! |\n  b\n...\n%YAML 1.1\n--- {}\n");
    }

    #[test]
    fn line_width() {
        let words: Vec<EventKind> = (0..12).map(|i| plain(&format!("word{i}"))).collect();
        assert_eq!(
            emit(with(|o| o.best_width = 20), doc(seq(Some(true), words))),
            "[word0, word1, word2,\n  word3, word4, word5,\n  word6, word7, word8,\n  word9, word10, word11]\n"
        );
        let text: Vec<String> = (0..20).map(|i| format!("w{i}")).collect();
        let options = with(|o| {
            o.best_width = 20;
            o.best_indent = 4;
        });
        assert_eq!(
            emit(options, doc(vec![plain(&text.join(" "))])),
            "w0 w1 w2 w3 w4 w5 w6 w7\n    w8 w9 w10 w11 w12\n    w13 w14 w15 w16 w17\n    w18 w19\n...\n"
        );
        let text = "abc def\tghi jkl mno pqr stu vwx yz abc def ghi";
        assert_eq!(
            emit(with(|o| o.best_width = 20), doc(vec![styled(text, '"')])),
            "! \"abc def\\tghi jkl mno\\\n  \\ pqr stu vwx yz abc\\\n  \\ def ghi\"\n"
        );
        // An infinite width never wraps.
        let text: Vec<String> = (0..200).map(|i| format!("w{i}")).collect();
        let out = emit(
            with(|o| o.best_width = u64::MAX),
            doc(vec![plain(&text.join(" "))]),
        );
        assert_eq!(out, format!("{}\n...\n", text.join(" ")));
    }

    #[test]
    fn anchors_tags_directives() {
        let mut items = vec![tagged("x", "tag:example.com,2000:foo")];
        items.push(EventKind::Alias {
            anchor: "a".to_owned(),
        });
        items.extend([
            tagged("y", "tag:yaml.org,2002:str"),
            tagged("z", "!local thing"),
        ]);
        let mut body = seq(None, items);
        body[0] = EventKind::SequenceStart {
            anchor: Some("a".to_owned()),
            tag: Some("tag:yaml.org,2002:seq".to_owned()),
            implicit: false,
            flow_style: None,
        };
        let start = EventKind::DocumentStart {
            explicit: false,
            version: Some((1, 1)),
            tags: Some(vec![("!e!".to_owned(), "tag:example.com,2000:".to_owned())]),
        };
        assert_eq!(
            emit(EmitterOptions::default(), doc_with(start, body)),
            "%YAML 1.1\n%TAG !e! tag:example.com,2000:\n--- &a !!seq\n- !e!foo 'x'\n- *a\n\
             - !!str 'y'\n- !local%20thing 'z'\n"
        );
    }

    #[test]
    fn writes_and_bom() {
        let mut emitter = Emitter::new(with(|o| o.record_writes = true));
        for kind in ab_xy(None) {
            emitter.emit(Event::new(kind)).unwrap();
        }
        assert_eq!(emitter.writes(), (1..=16).collect::<Vec<_>>());
        let mut events = doc(vec![plain("x")]);
        events[0] = EventKind::StreamStart {
            encoding: Some("utf-16".to_owned()),
        };
        let mut emitter = Emitter::new(with(|o| o.record_writes = true));
        for kind in events {
            emitter.emit(Event::new(kind)).unwrap();
        }
        assert_eq!(emitter.output(), "\u{FEFF}x\n...\n");
        assert_eq!(emitter.writes(), [3, 4, 5, 8, 9]);
        assert_eq!(emitter.take_output(), "\u{FEFF}x\n...\n");
        assert!(emitter.writes().is_empty() && emitter.output().is_empty());
    }

    #[test]
    fn event_queue() {
        let mut emitter = Emitter::new(EmitterOptions::default());
        emitter
            .emit(Event::new(EventKind::StreamStart { encoding: None }))
            .unwrap();
        let start = EventKind::DocumentStart {
            explicit: false,
            version: None,
            tags: None,
        };
        emitter.emit(Event::new(start)).unwrap();
        // The document start waits for the next event (an empty scalar would make it explicit).
        assert_eq!(emitter.output(), "");
        emitter.emit(Event::new(plain("x"))).unwrap();
        assert_eq!(emitter.output(), "x");
        emitter
            .emit(Event::new(EventKind::DocumentEnd { explicit: false }))
            .unwrap();
        emitter.emit(Event::new(EventKind::StreamEnd)).unwrap();
        assert_eq!(emitter.output(), "x\n...\n");
    }

    #[test]
    fn errors() {
        let mut bad_anchor = doc(vec![plain("x")]);
        if let EventKind::Scalar { anchor, .. } = &mut bad_anchor[2] {
            *anchor = Some("a b".to_owned());
        }
        assert_eq!(
            error(bad_anchor),
            "invalid character ' ' in the anchor: 'a b'"
        );
        let tags = |h: &str, p: &str| EventKind::DocumentStart {
            explicit: false,
            version: None,
            tags: Some(vec![(h.to_owned(), p.to_owned())]),
        };
        assert_eq!(
            error(doc_with(tags("!a b!", "x"), vec![plain("x")])),
            "invalid character ' ' in the tag handle: '!a b!'"
        );
        assert_eq!(
            error(doc_with(tags("!a!", "a b"), vec![plain("x")])),
            "TypeError: ord() expected string of length 1, but int found"
        );
        let version = EventKind::DocumentStart {
            explicit: false,
            version: Some((2, 0)),
            tags: None,
        };
        assert_eq!(
            error(doc_with(version, vec![plain("x")])),
            "unsupported YAML version: 2.0"
        );
        assert_eq!(
            error(doc(vec![])),
            "expected NodeEvent, but got DocumentEndEvent()"
        );
        assert_eq!(error(doc(vec![tagged("x", "")])), "tag must not be empty");
        let mut no_tag = doc(vec![plain("x")]);
        if let EventKind::Scalar { implicit, .. } = &mut no_tag[2] {
            *implicit = (false, false);
        }
        assert_eq!(error(no_tag), "tag is not specified");
        let start = EventKind::DocumentStart {
            explicit: false,
            version: None,
            tags: None,
        };
        assert_eq!(
            error(vec![start, plain("x")]),
            "expected StreamStartEvent, but got DocumentStartEvent()"
        );
        let mut extra = doc(vec![plain("x")]);
        extra.insert(3, plain("it's"));
        assert_eq!(
            error(extra),
            "expected DocumentEndEvent, but got ScalarEvent(anchor=None, tag=None, \
             implicit=(True, False), value=\"it's\")"
        );
    }
}
