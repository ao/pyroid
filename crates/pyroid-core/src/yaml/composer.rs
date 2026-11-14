// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! The composer (`composer.py`): events to a node graph per document.
//!
//! PyYAML composes recursively; this composer keeps an explicit stack, so the depth limit
//! is a parameter (the bindings derive it from Python's recursion limit, see
//! [`ComposerOptions::max_depth`]).

use std::borrow::Cow;

use rustc_hash::FxHashMap;

use super::error::{ErrorKind, Mark, Msg, Result, YamlError};
use super::events::{Event, EventKind};
use super::nodes::{Graph, Node, NodeId, NodeKind};
use super::parser::Parser;
use super::resolver::{self, MAP, SEQ};

#[derive(Clone, Copy, Debug)]
pub struct ComposerOptions {
    /// Collections nested deeper than this raise `RecursionError` (PyYAML's recursive
    /// composer hits Python's recursion limit). `usize::MAX`: no limit.
    pub max_depth: usize,
}

impl Default for ComposerOptions {
    fn default() -> Self {
        Self {
            max_depth: usize::MAX,
        }
    }
}

/// Why composing stopped, beyond [`YamlError`].
fn composer_error(
    context: Option<Msg>,
    context_mark: Option<Mark>,
    problem: Msg,
    problem_mark: Mark,
) -> YamlError {
    YamlError::marked(
        ErrorKind::Composer,
        context,
        context_mark,
        problem,
        Some(problem_mark),
    )
}

enum Frame {
    Seq(NodeId),
    Map(NodeId, Option<NodeId>),
}

/// PyYAML's `Composer` over a [`Parser`].
#[derive(Debug)]
pub struct Composer {
    pub parser: Parser,
    options: ComposerOptions,
}

impl Composer {
    pub fn new(parser: Parser, options: ComposerOptions) -> Self {
        Self { parser, options }
    }

    fn check_stream_end(&mut self) -> Result<bool> {
        Ok(matches!(
            self.parser.peek_event()?.map(|e| &e.kind),
            Some(EventKind::StreamEnd)
        ))
    }

    fn get_event(&mut self) -> Result<Event> {
        self.parser
            .get_event()?
            .ok_or_else(|| YamlError::Value("composer read past the end of the stream".to_owned()))
    }

    /// `check_node()`: drop STREAM-START; is there another document.
    pub fn check_node(&mut self) -> Result<bool> {
        if matches!(
            self.parser.peek_event()?.map(|e| &e.kind),
            Some(EventKind::StreamStart { .. })
        ) {
            self.parser.get_event()?;
        }
        Ok(!self.check_stream_end()?)
    }

    /// `get_node()`: the next document (`None` at the end).
    pub fn get_node(&mut self) -> Result<Option<Graph>> {
        if self.check_stream_end()? {
            return Ok(None);
        }
        self.compose_document().map(Some)
    }

    /// `get_single_node()`.
    pub fn get_single_node(&mut self) -> Result<Option<Graph>> {
        self.parser.get_event()?; // STREAM-START
        let mut document = None;
        if !self.check_stream_end()? {
            document = Some(self.compose_document()?);
        }
        if !self.check_stream_end()? {
            let event = self.get_event()?;
            let start = document.as_ref().map(|d| d.nodes[d.root].start_mark);
            return Err(composer_error(
                Some(Msg::text("expected a single document in the stream")),
                start,
                Msg::text("but found another document"),
                event.start_mark,
            ));
        }
        self.parser.get_event()?; // STREAM-END
        Ok(document)
    }

    fn compose_document(&mut self) -> Result<Graph> {
        self.get_event()?; // DOCUMENT-START
        let mut graph = Graph::default();
        graph.root = self.compose_node(&mut graph)?;
        self.get_event()?; // DOCUMENT-END
        Ok(graph)
    }

    /// `compose_node(None, None)` for the document root, without recursion.
    #[allow(clippy::too_many_lines)] // compose_node and its three helpers, unrolled
    fn compose_node(&mut self, graph: &mut Graph) -> Result<NodeId> {
        let mut anchors: FxHashMap<String, NodeId> = FxHashMap::default();
        let mut stack: Vec<Frame> = Vec::new();
        'compose: loop {
            // Compose one node (or open a collection).
            let mut done: NodeId = {
                let event = self.get_peeked()?;
                if let EventKind::Alias { anchor } = &event.kind {
                    let anchor = anchor.clone();
                    let mark = event.start_mark;
                    self.parser.get_event()?;
                    match anchors.get(&anchor) {
                        Some(&id) => id,
                        None => {
                            return Err(composer_error(
                                None,
                                None,
                                Msg::str("found undefined alias %r", anchor),
                                mark,
                            ));
                        }
                    }
                } else {
                    let anchor = match &event.kind {
                        EventKind::Scalar { anchor, .. }
                        | EventKind::SequenceStart { anchor, .. }
                        | EventKind::MappingStart { anchor, .. } => anchor.clone(),
                        _ => None,
                    };
                    if let Some(a) = &anchor {
                        if let Some(&first) = anchors.get(a) {
                            return Err(composer_error(
                                Some(Msg::str(
                                    "found duplicate anchor %r; first occurrence",
                                    a.clone(),
                                )),
                                Some(graph.nodes[first].start_mark),
                                Msg::text("second occurrence"),
                                event.start_mark,
                            ));
                        }
                    }
                    let event = self.get_event()?;
                    let (start_mark, end_mark) = (event.start_mark, event.end_mark);
                    match event.kind {
                        EventKind::Scalar {
                            tag,
                            implicit,
                            value,
                            style,
                            ..
                        } => {
                            let tag = node_tag(
                                tag,
                                |t| t.is_none_or(|t| t == "!"),
                                || resolver::resolve_scalar(&value, implicit.0),
                            );
                            let id = graph.push(Node {
                                tag,
                                kind: NodeKind::Scalar { value, style },
                                start_mark,
                                end_mark,
                            });
                            if let Some(a) = anchor {
                                anchors.insert(a, id);
                            }
                            id
                        }
                        EventKind::SequenceStart {
                            tag, flow_style, ..
                        } => {
                            let tag = node_tag(tag, |t| t.is_none_or(|t| t == "!"), || SEQ);
                            let id = graph.push(Node {
                                tag,
                                kind: NodeKind::Sequence {
                                    items: Vec::new(),
                                    flow_style,
                                },
                                start_mark,
                                end_mark: start_mark,
                            });
                            if let Some(a) = anchor {
                                anchors.insert(a, id);
                            }
                            if stack.len() >= self.options.max_depth {
                                return Err(YamlError::Recursion);
                            }
                            stack.push(Frame::Seq(id));
                            match self.close_if_end(graph, id, false)? {
                                Some(id) => {
                                    stack.pop();
                                    id
                                }
                                None => continue 'compose,
                            }
                        }
                        EventKind::MappingStart {
                            tag, flow_style, ..
                        } => {
                            let tag = node_tag(tag, |t| t.is_none_or(|t| t == "!"), || MAP);
                            let id = graph.push(Node {
                                tag,
                                kind: NodeKind::Mapping {
                                    pairs: Vec::new(),
                                    flow_style,
                                },
                                start_mark,
                                end_mark: start_mark,
                            });
                            if let Some(a) = anchor {
                                anchors.insert(a, id);
                            }
                            if stack.len() >= self.options.max_depth {
                                return Err(YamlError::Recursion);
                            }
                            stack.push(Frame::Map(id, None));
                            match self.close_if_end(graph, id, true)? {
                                Some(id) => {
                                    stack.pop();
                                    id
                                }
                                None => continue 'compose,
                            }
                        }
                        _ => {
                            return Err(YamlError::Value(
                                "unexpected event while composing a node".to_owned(),
                            ));
                        }
                    }
                }
            };
            // Hand the finished node to its parent; close parents that end.
            loop {
                match stack.last_mut() {
                    None => return Ok(done),
                    Some(Frame::Seq(id)) => {
                        let id = *id;
                        if let NodeKind::Sequence { items, .. } = &mut graph.nodes[id].kind {
                            items.push(done);
                        }
                        match self.close_if_end(graph, id, false)? {
                            Some(id) => {
                                stack.pop();
                                done = id;
                            }
                            None => continue 'compose,
                        }
                    }
                    Some(Frame::Map(id, key)) => {
                        let id = *id;
                        match key.take() {
                            None => {
                                *key = Some(done);
                                continue 'compose;
                            }
                            Some(k) => {
                                if let NodeKind::Mapping { pairs, .. } = &mut graph.nodes[id].kind {
                                    pairs.push((k, done));
                                }
                                match self.close_if_end(graph, id, true)? {
                                    Some(id) => {
                                        stack.pop();
                                        done = id;
                                    }
                                    None => continue 'compose,
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn get_peeked(&mut self) -> Result<&Event> {
        self.parser
            .peek_event()?
            .ok_or_else(|| YamlError::Value("composer read past the end of the stream".to_owned()))
    }

    /// `while not self.check_event(SequenceEndEvent)` / `MappingEndEvent`: when it ends, take
    /// the end event and set the node's end mark.
    fn close_if_end(
        &mut self,
        graph: &mut Graph,
        id: NodeId,
        mapping: bool,
    ) -> Result<Option<NodeId>> {
        let ends = match self.parser.peek_event()?.map(|e| &e.kind) {
            Some(EventKind::SequenceEnd) => !mapping,
            Some(EventKind::MappingEnd) => mapping,
            _ => false,
        };
        if !ends {
            return Ok(None);
        }
        let end = self.get_event()?;
        graph.nodes[id].end_mark = end.end_mark;
        Ok(Some(id))
    }
}

/// The node's tag: the event's, or the resolved one when the event has none or `!`.
fn node_tag(
    tag: Option<String>,
    resolve_it: impl Fn(Option<&str>) -> bool,
    resolved: impl Fn() -> &'static str,
) -> Cow<'static, str> {
    if resolve_it(tag.as_deref()) {
        return Cow::Borrowed(resolved());
    }
    match tag {
        Some(t) => match resolver::intern(&t) {
            Some(s) => Cow::Borrowed(s),
            None => Cow::Owned(t),
        },
        None => Cow::Borrowed(resolved()),
    }
}
