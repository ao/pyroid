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
    /// Does this node have children? (A node without any is freed without recursion.)
    pub fn is_leaf(&self) -> bool {
        match self {
            Self::Current
            | Self::Field(_)
            | Self::Identity
            | Self::Index(_)
            | Self::Literal(_)
            | Self::Slice(..) => true,
            Self::FunctionExpression(_, v)
            | Self::IndexExpression(v)
            | Self::MultiSelectDict(v)
            | Self::MultiSelectList(v)
            | Self::Subexpression(v) => v.is_empty(),
            _ => false,
        }
    }

    /// The children, in jmespath's order.
    pub fn children(&self) -> impl Iterator<Item = &Self> {
        let (few, many): ([Option<&Self>; 3], &[Self]) = match self {
            Self::Comparator(_, l, r)
            | Self::OrExpression(l, r)
            | Self::AndExpression(l, r)
            | Self::Pipe(l, r)
            | Self::Projection(l, r)
            | Self::ValueProjection(l, r) => ([Some(&**l), Some(&**r), None], &[]),
            Self::FilterProjection(l, r, c) => ([Some(&**l), Some(&**r), Some(&**c)], &[]),
            Self::Expref(x)
            | Self::Flatten(x)
            | Self::KeyValPair(_, x)
            | Self::NotExpression(x) => ([Some(&**x), None, None], &[]),
            Self::FunctionExpression(_, v)
            | Self::IndexExpression(v)
            | Self::MultiSelectDict(v)
            | Self::MultiSelectList(v)
            | Self::Subexpression(v) => ([None; 3], v),
            Self::Current
            | Self::Field(_)
            | Self::Identity
            | Self::Index(_)
            | Self::Literal(_)
            | Self::Slice(..) => ([None; 3], &[]),
        };
        few.into_iter().flatten().chain(many)
    }

    /// The children, in jmespath's order, to change.
    pub fn children_mut(&mut self) -> impl Iterator<Item = &mut Self> {
        let (few, many): ([Option<&mut Self>; 3], &mut [Self]) = match self {
            Self::Comparator(_, l, r)
            | Self::OrExpression(l, r)
            | Self::AndExpression(l, r)
            | Self::Pipe(l, r)
            | Self::Projection(l, r)
            | Self::ValueProjection(l, r) => ([Some(&mut **l), Some(&mut **r), None], &mut []),
            Self::FilterProjection(l, r, c) => {
                ([Some(&mut **l), Some(&mut **r), Some(&mut **c)], &mut [])
            }
            Self::Expref(x)
            | Self::Flatten(x)
            | Self::KeyValPair(_, x)
            | Self::NotExpression(x) => ([Some(&mut **x), None, None], &mut []),
            Self::FunctionExpression(_, v)
            | Self::IndexExpression(v)
            | Self::MultiSelectDict(v)
            | Self::MultiSelectList(v)
            | Self::Subexpression(v) => ([None, None, None], v),
            Self::Current
            | Self::Field(_)
            | Self::Identity
            | Self::Index(_)
            | Self::Literal(_)
            | Self::Slice(..) => ([None, None, None], &mut []),
        };
        few.into_iter().flatten().chain(many)
    }

    /// Take out the first child that has children of its own (an identity is left in its place).
    fn take_first_inner_child(&mut self) -> Option<Self> {
        let take = |n: &mut Self| (!n.is_leaf()).then(|| std::mem::replace(n, Self::Identity));
        match self {
            Self::Comparator(_, l, r)
            | Self::OrExpression(l, r)
            | Self::AndExpression(l, r)
            | Self::Pipe(l, r)
            | Self::Projection(l, r)
            | Self::ValueProjection(l, r) => take(l).or_else(|| take(r)),
            Self::FilterProjection(l, r, c) => take(l).or_else(|| take(r)).or_else(|| take(c)),
            Self::Expref(x)
            | Self::Flatten(x)
            | Self::KeyValPair(_, x)
            | Self::NotExpression(x) => take(x),
            Self::FunctionExpression(_, v)
            | Self::IndexExpression(v)
            | Self::MultiSelectDict(v)
            | Self::MultiSelectList(v)
            | Self::Subexpression(v) => v.iter_mut().find_map(take),
            Self::Current
            | Self::Field(_)
            | Self::Identity
            | Self::Index(_)
            | Self::Literal(_)
            | Self::Slice(..) => None,
        }
    }

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

/// Left-associative chains (`a | b | c`, `a || b`, `a == b`, `a[][]`) are parsed in a loop, so
/// a tree can be as deep as its expression is long. Such a chain always runs through each
/// node's first child (the `left` the parser folds into the new node): unlink it with a loop.
/// Anything else nests only as deep as the parser's own (budgeted) recursion, and is dropped
/// recursively.
impl<V> Drop for Node<V> {
    fn drop(&mut self) {
        let mut next = self.take_first_inner_child();
        while let Some(mut n) = next {
            next = n.take_first_inner_child();
        }
    }
}
