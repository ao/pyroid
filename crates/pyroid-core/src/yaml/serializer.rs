// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! The serializer (`serializer.py`): a node graph to events, into an [`Emitter`].
//!
//! Nodes reached more than once get anchors `id001`, `id002`, ... in PyYAML's traversal order
//! (numbered per document); later visits become aliases.

use super::emitter::Emitter;
use super::error::Result;
use super::events::{Event, EventKind};
use super::nodes::{Graph, NodeId, NodeKind};
use super::resolver::{self, MAP, SEQ};

/// The document options of `yaml.dump_all`.
#[derive(Clone, Debug, Default)]
pub struct SerializerOptions {
    pub explicit_start: bool,
    pub explicit_end: bool,
    pub version: Option<(u64, u64)>,
    /// `%TAG` handles and prefixes (the emitter sorts them).
    pub tags: Option<Vec<(String, String)>>,
}

/// `Serializer.open()`.
pub fn open(emitter: &mut Emitter, encoding: Option<String>) -> Result<()> {
    emitter.emit(Event::new(EventKind::StreamStart { encoding }))
}

/// `Serializer.close()`.
pub fn close(emitter: &mut Emitter) -> Result<()> {
    emitter.emit(Event::new(EventKind::StreamEnd))
}

/// `Serializer.serialize(node)`: one document.
pub fn serialize(emitter: &mut Emitter, graph: &Graph, options: &SerializerOptions) -> Result<()> {
    emitter.emit(Event::new(EventKind::DocumentStart {
        explicit: options.explicit_start,
        version: options.version,
        tags: options.tags.clone(),
    }))?;
    let mut s = State {
        anchors: vec![Anchor::Unseen; graph.nodes.len()],
        serialized: vec![false; graph.nodes.len()],
        last_anchor_id: 0,
    };
    s.anchor_node(graph, graph.root);
    s.serialize_node(emitter, graph, graph.root)?;
    emitter.emit(Event::new(EventKind::DocumentEnd {
        explicit: options.explicit_end,
    }))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Anchor {
    Unseen,
    /// Seen once: PyYAML's `None`.
    Once,
    Named(String),
}

struct State {
    anchors: Vec<Anchor>,
    serialized: Vec<bool>,
    last_anchor_id: u64,
}

impl State {
    fn anchor_node(&mut self, graph: &Graph, id: NodeId) {
        match self.anchors[id] {
            Anchor::Once => {
                self.last_anchor_id += 1;
                self.anchors[id] = Anchor::Named(format!("id{:03}", self.last_anchor_id));
            }
            Anchor::Named(_) => {}
            Anchor::Unseen => {
                self.anchors[id] = Anchor::Once;
                match &graph.nodes[id].kind {
                    NodeKind::Scalar { .. } => {}
                    NodeKind::Sequence { items, .. } => {
                        for &item in items {
                            self.anchor_node(graph, item);
                        }
                    }
                    NodeKind::Mapping { pairs, .. } => {
                        for &(k, v) in pairs {
                            self.anchor_node(graph, k);
                            self.anchor_node(graph, v);
                        }
                    }
                }
            }
        }
    }

    fn serialize_node(&mut self, emitter: &mut Emitter, graph: &Graph, id: NodeId) -> Result<()> {
        let alias = match &self.anchors[id] {
            Anchor::Named(a) => Some(a.clone()),
            _ => None,
        };
        if self.serialized[id] {
            // PyYAML emits AliasEvent(None) for a node it saw once; serialized nodes always
            // have a name here (a second visit named them).
            return emitter.emit(Event::new(EventKind::Alias {
                anchor: alias.unwrap_or_default(),
            }));
        }
        self.serialized[id] = true;
        let node = &graph.nodes[id];
        let tag = node.tag.to_string();
        match &node.kind {
            NodeKind::Scalar { value, style } => {
                let detected = resolver::resolve_scalar(value, true);
                let default = resolver::resolve_scalar(value, false);
                let implicit = (node.tag == detected, node.tag == default);
                emitter.emit(Event::new(EventKind::Scalar {
                    anchor: alias,
                    tag: Some(tag),
                    implicit,
                    value: value.clone(),
                    style: *style,
                }))
            }
            NodeKind::Sequence { items, flow_style } => {
                let implicit = node.tag == SEQ;
                emitter.emit(Event::new(EventKind::SequenceStart {
                    anchor: alias,
                    tag: Some(tag),
                    implicit,
                    flow_style: *flow_style,
                }))?;
                for &item in items {
                    self.serialize_node(emitter, graph, item)?;
                }
                emitter.emit(Event::new(EventKind::SequenceEnd))
            }
            NodeKind::Mapping { pairs, flow_style } => {
                let implicit = node.tag == MAP;
                emitter.emit(Event::new(EventKind::MappingStart {
                    anchor: alias,
                    tag: Some(tag),
                    implicit,
                    flow_style: *flow_style,
                }))?;
                for &(k, v) in pairs {
                    self.serialize_node(emitter, graph, k)?;
                    self.serialize_node(emitter, graph, v)?;
                }
                emitter.emit(Event::new(EventKind::MappingEnd))
            }
        }
    }
}
