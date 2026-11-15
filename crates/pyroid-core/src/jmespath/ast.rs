// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! The node kinds of `jmespath/ast.py`. Children are in jmespath's order.

/// A comparator node's `value`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Comparator {
    Eq,
    Ne,
    Lt,
    Gt,
    Lte,
    Gte,
}

impl Comparator {
    /// `'eq'`, `'ne'`, ...
    pub fn name(self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Gt => "gt",
            Self::Lte => "lte",
            Self::Gte => "gte",
        }
    }

    /// The comparator for a `'eq'`-style name.
    pub fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "eq" => Self::Eq,
            "ne" => Self::Ne,
            "lt" => Self::Lt,
            "gt" => Self::Gt,
            "lte" => Self::Lte,
            "gte" => Self::Gte,
            _ => return None,
        })
    }
}

/// A node: `{'type': ..., 'children': [...], 'value': ...}`.
#[derive(Clone, Debug, PartialEq)]
pub enum Node<V> {
    /// `comparator(name, first, second)`.
    Comparator(Comparator, Box<Node<V>>, Box<Node<V>>),
    /// `current_node()`.
    Current,
    /// `expref(expression)`.
    Expref(Box<Node<V>>),
    /// `function_expression(name, args)`.
    FunctionExpression(V, Vec<Node<V>>),
    /// `field(name)`.
    Field(V),
    /// `filter_projection(left, right, comparator)`.
    FilterProjection(Box<Node<V>>, Box<Node<V>>, Box<Node<V>>),
    /// `flatten(node)`.
    Flatten(Box<Node<V>>),
    /// `identity()`.
    Identity,
    /// `index(index)`.
    Index(V),
    /// `index_expression(children)`.
    IndexExpression(Vec<Node<V>>),
    /// `key_val_pair(key_name, node)`.
    KeyValPair(V, Box<Node<V>>),
    /// `literal(literal_value)`.
    Literal(V),
    /// `multi_select_dict(nodes)` (the nodes are key-value pairs).
    MultiSelectDict(Vec<Node<V>>),
    /// `multi_select_list(nodes)`.
    MultiSelectList(Vec<Node<V>>),
    /// `or_expression(left, right)`.
    OrExpression(Box<Node<V>>, Box<Node<V>>),
    /// `and_expression(left, right)`.
    AndExpression(Box<Node<V>>, Box<Node<V>>),
    /// `not_expression(expr)`.
    NotExpression(Box<Node<V>>),
    /// `pipe(left, right)`.
    Pipe(Box<Node<V>>, Box<Node<V>>),
    /// `projection(left, right)`.
    Projection(Box<Node<V>>, Box<Node<V>>),
    /// `subexpression(children)`.
    Subexpression(Vec<Node<V>>),
    /// `slice(start, end, step)`.
    Slice(Option<V>, Option<V>, Option<V>),
    /// `value_projection(left, right)`.
    ValueProjection(Box<Node<V>>, Box<Node<V>>),
}

impl<V> Node<V> {
    /// `node['type']`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::Comparator(..) => "comparator",
            Self::Current => "current",
            Self::Expref(_) => "expref",
            Self::FunctionExpression(..) => "function_expression",
            Self::Field(_) => "field",
            Self::FilterProjection(..) => "filter_projection",
            Self::Flatten(_) => "flatten",
            Self::Identity => "identity",
            Self::Index(_) => "index",
            Self::IndexExpression(_) => "index_expression",
            Self::KeyValPair(..) => "key_val_pair",
            Self::Literal(_) => "literal",
            Self::MultiSelectDict(_) => "multi_select_dict",
            Self::MultiSelectList(_) => "multi_select_list",
            Self::OrExpression(..) => "or_expression",
            Self::AndExpression(..) => "and_expression",
            Self::NotExpression(_) => "not_expression",
            Self::Pipe(..) => "pipe",
            Self::Projection(..) => "projection",
            Self::Subexpression(_) => "subexpression",
            Self::Slice(..) => "slice",
            Self::ValueProjection(..) => "value_projection",
        }
    }
}
