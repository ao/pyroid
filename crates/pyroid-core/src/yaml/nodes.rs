// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! Representation nodes (`nodes.py`), kept in an arena so that aliases share a node and the
//! constructor can mutate it (merge keys, as PyYAML's `flatten_mapping` does).

use std::borrow::Cow;

use super::error::Mark;

/// An index into [`Graph::nodes`].
pub type NodeId = usize;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Scalar {
        value: String,
        /// `None` for plain, else the quote or block indicator.
        style: Option<char>,
    },
    Sequence {
        items: Vec<NodeId>,
        flow_style: Option<bool>,
    },
    Mapping {
        pairs: Vec<(NodeId, NodeId)>,
        flow_style: Option<bool>,
    },
}

impl NodeKind {
    /// PyYAML's `Node.id`.
    pub const fn id(&self) -> &'static str {
        match self {
            Self::Scalar { .. } => "scalar",
            Self::Sequence { .. } => "sequence",
            Self::Mapping { .. } => "mapping",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub tag: Cow<'static, str>,
    pub kind: NodeKind,
    pub start_mark: Mark,
    pub end_mark: Mark,
}

/// The nodes of one document (or of one dumped object), and its root.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub root: NodeId,
}

impl Graph {
    pub fn push(&mut self, node: Node) -> NodeId {
        self.nodes.push(node);
        self.nodes.len() - 1
    }
}
