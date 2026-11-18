// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! sqlparse's token objects (`sql.Token`, `sql.TokenList` and its subclasses) as an arena,
//! with the methods the grouping and the filters call, ported line by line.
//!
//! The arena mirrors the Python objects' state field for field, stale values included: a
//! group's `value` is set when it is built (the join of its children's values) and only
//! extended by `group_tokens(extend=True)`; `normalized` is set once; a token list is a
//! separate object, so two groups can share one (`sql.TokenList(stmt.tokens)` in the aligned
//! filter). Index arguments follow Python's list semantics (negative indices, slices that
//! clamp), and every place where sqlparse's Python code would raise returns
//! [`Fail::Python`], so the caller can run sqlparse's own code for the real exception.

use std::borrow::Cow;

use super::pystr;
use super::regex::Regex;
use super::ttype::{TT, Types, bits};
use crate::ftfy::UnicodeVersion;

/// A node id.
pub type Id = u32;
/// A token-list id.
pub type ListId = u32;
/// No list (a plain `Token`).
pub const NO_LIST: ListId = ListId::MAX;

/// Why a step did not finish.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Fail {
    /// sqlparse's Python code raises here (an `IndexError`, `TypeError`, ...): run it for the
    /// exception.
    Python,
    /// `SQLParseError(message)`.
    SqlParse(String),
}

/// The result of a step.
pub type R<T> = Result<T, Fail>;

/// The classes of `sqlparse.sql`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(missing_docs)]
pub enum Class {
    Token,
    TokenList,
    Statement,
    Identifier,
    IdentifierList,
    TypedLiteral,
    Parenthesis,
    SquareBrackets,
    Assignment,
    If,
    For,
    Comparison,
    Comment,
    Where,
    Over,
    Having,
    Case,
    Function,
    Begin,
    Operation,
    Values,
    Command,
}

impl Class {
    /// Every class, in `sql.py` order.
    pub const ALL: [Class; 22] = [
        Class::Token,
        Class::TokenList,
        Class::Statement,
        Class::Identifier,
        Class::IdentifierList,
        Class::TypedLiteral,
        Class::Parenthesis,
        Class::SquareBrackets,
        Class::Assignment,
        Class::If,
        Class::For,
        Class::Comparison,
        Class::Comment,
        Class::Where,
        Class::Over,
        Class::Having,
        Class::Case,
        Class::Function,
        Class::Begin,
        Class::Operation,
        Class::Values,
        Class::Command,
    ];

    /// The class name.
    pub fn name(self) -> &'static str {
        match self {
            Class::Token => "Token",
            Class::TokenList => "TokenList",
            Class::Statement => "Statement",
            Class::Identifier => "Identifier",
            Class::IdentifierList => "IdentifierList",
            Class::TypedLiteral => "TypedLiteral",
            Class::Parenthesis => "Parenthesis",
            Class::SquareBrackets => "SquareBrackets",
            Class::Assignment => "Assignment",
            Class::If => "If",
            Class::For => "For",
            Class::Comparison => "Comparison",
            Class::Comment => "Comment",
            Class::Where => "Where",
            Class::Over => "Over",
            Class::Having => "Having",
            Class::Case => "Case",
            Class::Function => "Function",
            Class::Begin => "Begin",
            Class::Operation => "Operation",
            Class::Values => "Values",
            Class::Command => "Command",
        }
    }

    /// `isinstance(x, self_class)` for an object of class `self`: `cls` is `self`, or
    /// `TokenList` (every group), or `Token` (everything).
    #[inline]
    pub fn is_a(self, cls: Class) -> bool {
        self == cls || cls == Class::Token || (cls == Class::TokenList && self != Class::Token)
    }

    /// The index in [`Class::ALL`].
    pub fn index(self) -> usize {
        self as usize
    }
}

/// A string value: a byte range of the source text, or owned.
#[derive(Clone, Debug)]
pub enum Str {
    /// `src[a..b]`.
    Span(u32, u32),
    /// Built by the port.
    Owned(Box<str>),
}

impl Str {
    /// The empty string.
    pub const EMPTY: Str = Str::Span(0, 0);

    /// An owned string.
    pub fn owned(s: impl Into<Box<str>>) -> Str {
        Str::Owned(s.into())
    }
}

/// One `sql.Token` (or `TokenList` subclass) object.
#[derive(Clone, Debug)]
pub struct Node {
    /// The Python class.
    pub class: Class,
    /// `.ttype`.
    pub ttype: Option<TT>,
    /// `.value`.
    pub value: Str,
    /// `.normalized`.
    pub normalized: Str,
    /// `.is_keyword`.
    pub is_keyword: bool,
    /// `.is_group`.
    pub is_group: bool,
    /// `.is_whitespace`.
    pub is_whitespace: bool,
    /// `.is_newline`.
    pub is_newline: bool,
    /// `.parent`.
    pub parent: Option<Id>,
    /// `.tokens` (groups only).
    pub list: ListId,
    /// The caller's object this node stands for (`u32::MAX`: none).
    pub origin: u32,
}

/// The arena: every node and token list of one run.
#[derive(Debug)]
pub struct Tree<'s> {
    /// The text the spans point into.
    pub src: &'s str,
    /// The nodes.
    pub nodes: Vec<Node>,
    /// The token lists.
    pub lists: Vec<Vec<Id>>,
    /// The token types.
    pub types: Types,
    /// The caller's Unicode version.
    pub v: UnicodeVersion,
}

fn id_of(n: usize) -> Id {
    Id::try_from(n).unwrap_or(Id::MAX)
}

/// Python's index normalisation for `list[i]`: `None` for an IndexError.
#[inline]
pub fn py_index(len: usize, i: isize) -> Option<usize> {
    let j = if i < 0 {
        i + isize::try_from(len).ok()?
    } else {
        i
    };
    let j = usize::try_from(j).ok()?;
    (j < len).then_some(j)
}

/// Python's slice bound normalisation (`list[a:b]`), for one bound.
#[inline]
pub fn py_bound(len: usize, i: isize) -> usize {
    let l = isize::try_from(len).unwrap_or(isize::MAX);
    let j = if i < 0 { (i + l).max(0) } else { i.min(l) };
    usize::try_from(j).unwrap_or(0)
}

impl<'s> Tree<'s> {
    /// An empty arena over `src`.
    pub fn new(src: &'s str, types: Types, v: UnicodeVersion) -> Tree<'s> {
        Tree {
            src,
            nodes: Vec::new(),
            lists: Vec::new(),
            types,
            v,
        }
    }

    /// The text of `s`.
    #[inline]
    pub fn s<'a>(&'a self, s: &'a Str) -> &'a str {
        match s {
            Str::Span(a, b) => &self.src[*a as usize..*b as usize],
            Str::Owned(o) => o,
        }
    }

    /// `node.value`.
    #[inline]
    pub fn val(&self, id: Id) -> &str {
        self.s(&self.nodes[id as usize].value)
    }

    /// `node.normalized`.
    #[inline]
    pub fn norm(&self, id: Id) -> &str {
        self.s(&self.nodes[id as usize].normalized)
    }

    /// The node.
    #[inline]
    pub fn n(&self, id: Id) -> &Node {
        &self.nodes[id as usize]
    }

    /// The node, mutably.
    #[inline]
    pub fn n_mut(&mut self, id: Id) -> &mut Node {
        &mut self.nodes[id as usize]
    }

    /// `sql.Token(ttype, value)`.
    pub fn token(&mut self, tt: TT, value: Str) -> Id {
        let w = self.types.info(tt).within;
        let is_keyword = w & bits::KEYWORD != 0;
        let normalized = if is_keyword {
            match pystr::upper(self.s(&value), self.v) {
                Cow::Borrowed(_) => value.clone(),
                Cow::Owned(u) => Str::owned(u),
            }
        } else {
            value.clone()
        };
        self.nodes.push(Node {
            class: Class::Token,
            ttype: Some(tt),
            value,
            normalized,
            is_keyword,
            is_group: false,
            is_whitespace: w & bits::WHITESPACE != 0,
            is_newline: w & bits::NEWLINE != 0,
            parent: None,
            list: NO_LIST,
            origin: u32::MAX,
        });
        id_of(self.nodes.len() - 1)
    }

    /// A new token list object holding `items`.
    pub fn new_list(&mut self, items: Vec<Id>) -> ListId {
        self.lists.push(items);
        id_of(self.lists.len() - 1)
    }

    /// `cls(tokens)` (`TokenList.__init__` on the list object `list`, which it keeps).
    pub fn group_on(&mut self, class: Class, list: ListId) -> Id {
        let id = id_of(self.nodes.len());
        let value = self.concat(&self.lists[list as usize].clone());
        for i in 0..self.lists[list as usize].len() {
            let t = self.lists[list as usize][i];
            self.nodes[t as usize].parent = Some(id);
        }
        self.nodes.push(Node {
            class,
            ttype: None,
            normalized: value.clone(),
            value,
            is_keyword: false,
            is_group: true,
            is_whitespace: false,
            is_newline: false,
            parent: None,
            list,
            origin: u32::MAX,
        });
        id
    }

    /// `cls(tokens)` on a new list.
    pub fn group(&mut self, class: Class, items: Vec<Id>) -> Id {
        let l = self.new_list(items);
        self.group_on(class, l)
    }

    /// `''.join(token.value for token in ids)`, as a span when the values are adjacent spans.
    pub fn concat(&self, ids: &[Id]) -> Str {
        let mut span: Option<(u32, u32)> = None;
        let mut contiguous = true;
        for &i in ids {
            match &self.nodes[i as usize].value {
                Str::Span(a, b) => {
                    if a == b {
                        continue;
                    }
                    span = match span {
                        None => Some((*a, *b)),
                        Some((x, y)) if y == *a => Some((x, *b)),
                        Some(_) => {
                            contiguous = false;
                            break;
                        }
                    }
                }
                Str::Owned(o) => {
                    if !o.is_empty() {
                        contiguous = false;
                        break;
                    }
                }
            }
        }
        if contiguous {
            return span.map_or(Str::EMPTY, |(a, b)| Str::Span(a, b));
        }
        let mut s = String::new();
        for &i in ids {
            s.push_str(self.val(i));
        }
        Str::owned(s)
    }

    /// `a + b` of two values.
    pub fn append(&self, a: &Str, b: &Str) -> Str {
        if let (Str::Span(x, y), Str::Span(p, q)) = (a, b) {
            if y == p {
                return Str::Span(*x, *q);
            }
            if p == q {
                return a.clone();
            }
            if x == y {
                return b.clone();
            }
        }
        let mut s = String::with_capacity(self.s(a).len() + self.s(b).len());
        s.push_str(self.s(a));
        s.push_str(self.s(b));
        Str::owned(s)
    }

    /// The children of group `id` (`id.tokens`); empty for a plain token.
    #[inline]
    pub fn kids(&self, id: Id) -> &[Id] {
        let l = self.nodes[id as usize].list;
        if l == NO_LIST {
            &[]
        } else {
            &self.lists[l as usize]
        }
    }

    /// The list id of group `id`.
    #[inline]
    pub fn list_of(&self, id: Id) -> ListId {
        self.nodes[id as usize].list
    }

    /// `len(id.tokens)`.
    #[inline]
    pub fn len(&self, id: Id) -> usize {
        self.kids(id).len()
    }

    /// `id.tokens[i]` (Python indexing).
    pub fn at(&self, id: Id, i: isize) -> R<Id> {
        let k = self.kids(id);
        py_index(k.len(), i).map(|j| k[j]).ok_or(Fail::Python)
    }

    /// `isinstance(node, cls)`.
    #[inline]
    pub fn isa(&self, id: Id, cls: Class) -> bool {
        self.nodes[id as usize].class.is_a(cls)
    }

    /// `isinstance(node, (cls, ...))`.
    #[inline]
    pub fn isa_any(&self, id: Id, classes: &[Class]) -> bool {
        let c = self.nodes[id as usize].class;
        classes.iter().any(|&k| c.is_a(k))
    }

    /// `node.ttype is tt`.
    #[inline]
    pub fn is(&self, id: Id, tt: TT) -> bool {
        self.nodes[id as usize].ttype == Some(tt)
    }

    /// `node.ttype in tt` (prefix; `False` for a group's `None`).
    #[inline]
    pub fn within(&self, id: Id, tt: TT) -> bool {
        self.nodes[id as usize]
            .ttype
            .is_some_and(|t| self.types.within(t, tt))
    }

    /// `node.ttype in (a, b, ...)` (a tuple: equality).
    #[inline]
    pub fn is_any(&self, id: Id, tts: &[TT]) -> bool {
        self.nodes[id as usize]
            .ttype
            .is_some_and(|t| tts.contains(&t))
    }

    /// `node.match(tt, None)`.
    #[inline]
    pub fn m0(&self, id: Id, tt: TT) -> bool {
        self.is(id, tt)
    }

    /// `node.match(tt, values)` (plain values; upper-cased for keywords).
    pub fn m(&self, id: Id, tt: TT, values: &[&str]) -> bool {
        if !self.is(id, tt) {
            return false;
        }
        let n = &self.nodes[id as usize];
        let norm = self.s(&n.normalized);
        if n.is_keyword {
            values.iter().any(|v| {
                if v.is_ascii() {
                    norm.len() == v.len()
                        && norm
                            .bytes()
                            .zip(v.bytes())
                            .all(|(a, b)| a == b.to_ascii_uppercase())
                } else {
                    *pystr::upper(v, self.v) == *norm
                }
            })
        } else {
            values.contains(&norm)
        }
    }

    /// `node.match(tt, patterns, regex=True)`: `pattern.search(node.normalized)` for each
    /// pattern; `ci` holds the patterns compiled with IGNORECASE (keywords), `cs` without.
    pub fn m_regex(&self, id: Id, tt: TT, ci: &[Regex], cs: &[Regex]) -> bool {
        if !self.is(id, tt) {
            return false;
        }
        let n = &self.nodes[id as usize];
        let norm = self.s(&n.normalized);
        let pats = if n.is_keyword { ci } else { cs };
        pats.iter().any(|p| search(p, norm))
    }

    /// `str(node)`: the value of a token, the joined values of a group's leaves.
    pub fn text(&self, id: Id) -> Cow<'_, str> {
        let n = &self.nodes[id as usize];
        if !n.is_group {
            return Cow::Borrowed(self.s(&n.value));
        }
        let mut out = String::new();
        let mut first: Option<&str> = None;
        let mut count = 0;
        self.for_leaves(id, |t| {
            let v = t.val_of_leaf();
            count += 1;
            if count == 1 {
                first = Some(v);
            } else {
                if count == 2 {
                    out.push_str(first.unwrap_or(""));
                }
                out.push_str(v);
            }
        });
        match count {
            0 => Cow::Borrowed(""),
            1 => Cow::Borrowed(first.unwrap_or("")),
            _ => Cow::Owned(out),
        }
    }

    /// Calls `f` for each leaf of `id` in order (`flatten()`, without recursion).
    pub fn for_leaves<'a>(&'a self, id: Id, mut f: impl FnMut(Leaf<'a, 's>)) {
        let n = &self.nodes[id as usize];
        if !n.is_group {
            f(Leaf { tree: self, id });
            return;
        }
        let mut stack: Vec<(ListId, usize)> = vec![(n.list, 0)];
        while let Some((l, i)) = stack.pop() {
            if l == NO_LIST {
                continue;
            }
            let list = &self.lists[l as usize];
            if i >= list.len() {
                continue;
            }
            stack.push((l, i + 1));
            let t = list[i];
            let tn = &self.nodes[t as usize];
            if tn.is_group {
                stack.push((tn.list, 0));
            } else {
                f(Leaf { tree: self, id: t });
            }
        }
    }

    /// The first leaf of `id` (`next(id.flatten())`); `None` if it has none (StopIteration).
    pub fn first_leaf(&self, id: Id) -> Option<Id> {
        let n = &self.nodes[id as usize];
        if !n.is_group {
            return Some(id);
        }
        let mut stack: Vec<(ListId, usize)> = vec![(n.list, 0)];
        while let Some((l, i)) = stack.pop() {
            if l == NO_LIST {
                continue;
            }
            let list = &self.lists[l as usize];
            if i >= list.len() {
                continue;
            }
            stack.push((l, i + 1));
            let t = list[i];
            let tn = &self.nodes[t as usize];
            if tn.is_group {
                stack.push((tn.list, 0));
            } else {
                return Some(t);
            }
        }
        None
    }

    /// The groups among `id`'s children (`get_sublists()`), read from the live list.
    #[inline]
    pub fn sublist_at(&self, id: Id, i: usize) -> Option<(usize, Option<Id>)> {
        let k = self.kids(id);
        if i >= k.len() {
            return None;
        }
        let t = k[i];
        Some((i + 1, self.nodes[t as usize].is_group.then_some(t)))
    }

    /// `token.within(cls)`: an ancestor is a `cls`.
    pub fn within_class(&self, id: Id, cls: Class) -> bool {
        let mut p = self.nodes[id as usize].parent;
        while let Some(x) = p {
            if self.isa(x, cls) {
                return true;
            }
            p = self.nodes[x as usize].parent;
        }
        false
    }

    // ------------------------------------------------------------------
    // TokenList methods
    // ------------------------------------------------------------------

    /// `tlist._token_matching(funcs, start, None, reverse)`.
    pub fn token_matching(
        &self,
        id: Id,
        start: isize,
        reverse: bool,
        mut f: impl FnMut(&Self, Id) -> bool,
    ) -> R<Option<(usize, Id)>> {
        let k = self.kids(id);
        let len = k.len();
        if reverse {
            // range(start - 2, -1, -1)
            let mut i = start - 2;
            while i >= 0 {
                let j = py_index(len, i).ok_or(Fail::Python)?;
                if f(self, k[j]) {
                    return Ok(Some((j, k[j])));
                }
                i -= 1;
            }
            return Ok(None);
        }
        let end = isize::try_from(len).unwrap_or(isize::MAX);
        let mut i = start;
        while i < end {
            let j = py_index(len, i).ok_or(Fail::Python)?;
            if f(self, k[j]) {
                // Python returns the loop index, which is negative for a negative start
                return Ok(Some((usize::try_from(i).unwrap_or(j), k[j])));
            }
            i += 1;
        }
        Ok(None)
    }

    /// `tlist.token_next_by(..., idx=idx)` with predicate `f` (`imt(tk, i, m, t)`).
    pub fn next_by(
        &self,
        id: Id,
        idx: isize,
        f: impl FnMut(&Self, Id) -> bool,
    ) -> R<Option<(usize, Id)>> {
        self.token_matching(id, idx + 1, false, f)
    }

    /// The matcher of `token_first`/`token_next`: not whitespace (if `skip_ws`), not a
    /// comment (if `skip_cm`).
    #[inline]
    pub fn not_skipped(&self, t: Id, skip_ws: bool, skip_cm: bool) -> bool {
        let n = &self.nodes[t as usize];
        !((skip_ws && n.is_whitespace)
            || (skip_cm
                && (self.within(t, super::ttype::builtin::COMMENT) || self.isa(t, Class::Comment))))
    }

    /// `tlist.token_first(skip_ws, skip_cm)`.
    pub fn token_first(&self, id: Id, skip_ws: bool, skip_cm: bool) -> R<Option<Id>> {
        Ok(self
            .token_matching(id, 0, false, |s, t| s.not_skipped(t, skip_ws, skip_cm))?
            .map(|x| x.1))
    }

    /// `tlist.token_next(idx, skip_ws, skip_cm)` (`idx=None` gives `(None, None)`).
    pub fn token_next(
        &self,
        id: Id,
        idx: Option<isize>,
        skip_ws: bool,
        skip_cm: bool,
    ) -> R<Option<(usize, Id)>> {
        let Some(idx) = idx else { return Ok(None) };
        self.token_matching(id, idx + 1, false, |s, t| {
            s.not_skipped(t, skip_ws, skip_cm)
        })
    }

    /// `tlist.token_prev(idx, skip_ws, skip_cm)`.
    pub fn token_prev(
        &self,
        id: Id,
        idx: Option<isize>,
        skip_ws: bool,
        skip_cm: bool,
    ) -> R<Option<(usize, Id)>> {
        let Some(idx) = idx else { return Ok(None) };
        self.token_matching(id, idx + 1, true, |s, t| s.not_skipped(t, skip_ws, skip_cm))
    }

    /// `tlist.token_index(token, start)` (ValueError if absent).
    pub fn token_index(&self, id: Id, token: Id, start: isize) -> R<usize> {
        let k = self.kids(id);
        let s = py_bound(k.len(), start);
        k[s..]
            .iter()
            .position(|&t| t == token)
            .map(|p| s + p)
            .ok_or(Fail::Python)
    }

    /// `tlist.group_tokens(cls, start, end, include_end, extend)`.
    pub fn group_tokens(
        &mut self,
        id: Id,
        cls: Class,
        start: isize,
        end: isize,
        include_end: bool,
        extend: bool,
    ) -> R<Id> {
        let l = self.list_of(id);
        let len = self.lists[l as usize].len();
        let start_tok = py_index(len, start)
            .map(|j| self.lists[l as usize][j])
            .ok_or(Fail::Python)?;
        let end_idx = end + isize::from(include_end);
        let grp;
        let subtokens: Vec<Id>;
        if extend && self.isa(start_tok, cls) {
            let a = py_bound(len, start + 1);
            let b = py_bound(len, end_idx).max(a);
            subtokens = self.lists[l as usize][a..b].to_vec();
            grp = start_tok;
            let gl = self.list_of(grp);
            if gl == NO_LIST {
                return Err(Fail::Python);
            }
            self.lists[gl as usize].extend_from_slice(&subtokens);
            self.lists[l as usize].drain(a..b);
            let add = self.concat(&subtokens);
            let nv = self.append(&self.nodes[grp as usize].value, &add);
            self.nodes[grp as usize].value = nv;
        } else {
            let a = py_bound(len, start);
            let b = py_bound(len, end_idx).max(a);
            subtokens = self.lists[l as usize][a..b].to_vec();
            grp = self.group(cls, subtokens.clone());
            self.lists[l as usize].splice(a..b, [grp]);
            self.nodes[grp as usize].parent = Some(id);
        }
        for t in subtokens {
            self.nodes[t as usize].parent = Some(grp);
        }
        Ok(grp)
    }

    /// `tlist.insert_before(where, token)` with an index.
    pub fn insert_before(&mut self, id: Id, at: isize, token: Id) {
        self.nodes[token as usize].parent = Some(id);
        let l = self.list_of(id);
        let list = &mut self.lists[l as usize];
        let i = py_bound(list.len(), at);
        list.insert(i, token);
    }

    /// `tlist.insert_before(where_token, token)`.
    pub fn insert_before_tok(&mut self, id: Id, at: Id, token: Id) -> R<()> {
        let i = self.token_index(id, at, 0)?;
        self.insert_before(id, isize::try_from(i).unwrap_or(0), token);
        Ok(())
    }

    /// `tlist.insert_after(where, token, skip_ws)` with an index.
    pub fn insert_after(&mut self, id: Id, at: isize, token: Id, skip_ws: bool) -> R<()> {
        let next = self.token_next(id, Some(at), skip_ws, false)?;
        self.nodes[token as usize].parent = Some(id);
        let l = self.list_of(id);
        let list = &mut self.lists[l as usize];
        match next {
            None => list.push(token),
            Some((nidx, _)) => {
                let i = py_bound(list.len(), isize::try_from(nidx).unwrap_or(0));
                list.insert(i, token);
            }
        }
        Ok(())
    }

    /// `tlist.insert_after(where_token, token, skip_ws)`.
    pub fn insert_after_tok(&mut self, id: Id, at: Id, token: Id, skip_ws: bool) -> R<()> {
        let i = self.token_index(id, at, 0)?;
        self.insert_after(id, isize::try_from(i).unwrap_or(0), token, skip_ws)
    }

    /// `tlist.tokens.insert(i, token)` (no parent change).
    pub fn list_insert(&mut self, l: ListId, at: isize, token: Id) {
        let list = &mut self.lists[l as usize];
        let i = py_bound(list.len(), at);
        list.insert(i, token);
    }

    /// `del tlist.tokens[i]` / `tlist.tokens.pop(i)`.
    pub fn list_remove_at(&mut self, l: ListId, at: isize) -> R<Id> {
        let list = &mut self.lists[l as usize];
        let j = py_index(list.len(), at).ok_or(Fail::Python)?;
        Ok(list.remove(j))
    }

    /// `tlist.tokens.remove(token)` (identity: tokens have no `__eq__`).
    pub fn list_remove(&mut self, l: ListId, token: Id) -> R<()> {
        let list = &mut self.lists[l as usize];
        let j = list.iter().position(|&t| t == token).ok_or(Fail::Python)?;
        list.remove(j);
        Ok(())
    }

    /// The `Case.get_cases(skip_ws)` lists: `(condition, value)`, `None` for ELSE's condition.
    pub fn get_cases(&self, id: Id, skip_ws: bool) -> Vec<(Option<Vec<Id>>, Vec<Id>)> {
        use super::ttype::builtin::{KEYWORD, WHITESPACE};
        #[derive(PartialEq, Clone, Copy)]
        enum Mode {
            Condition,
            Value,
            Off,
        }
        let mut ret: Vec<(Option<Vec<Id>>, Vec<Id>)> = Vec::new();
        let mut mode = Mode::Condition;
        for &t in self.kids(id) {
            if self.m(t, KEYWORD, &["CASE"]) || (skip_ws && self.within(t, WHITESPACE)) {
                continue;
            }
            if self.m(t, KEYWORD, &["WHEN"]) {
                ret.push((Some(Vec::new()), Vec::new()));
                mode = Mode::Condition;
            } else if self.m(t, KEYWORD, &["THEN"]) {
                mode = Mode::Value;
            } else if self.m(t, KEYWORD, &["ELSE"]) {
                ret.push((None, Vec::new()));
                mode = Mode::Value;
            } else if self.m(t, KEYWORD, &["END"]) {
                mode = Mode::Off;
            }
            if mode != Mode::Off && ret.is_empty() {
                ret.push((Some(Vec::new()), Vec::new()));
            }
            match mode {
                Mode::Condition => {
                    // only WHEN (which appends a fresh pair) and the start set this mode, so
                    // the last condition is never ELSE's None
                    if let Some((Some(c), _)) = ret.last_mut() {
                        c.push(t);
                    }
                }
                Mode::Value => {
                    if let Some(last) = ret.last_mut() {
                        last.1.push(t);
                    }
                }
                Mode::Off => {}
            }
        }
        ret
    }

    /// `IdentifierList.get_identifiers()`: children that are not whitespace or `,`.
    pub fn get_identifiers(&self, id: Id) -> Vec<Id> {
        self.kids(id)
            .iter()
            .copied()
            .filter(|&t| {
                !(self.nodes[t as usize].is_whitespace
                    || self.m(t, super::ttype::builtin::PUNCTUATION, &[","]))
            })
            .collect()
    }
}

/// One leaf seen by [`Tree::for_leaves`].
#[derive(Clone, Copy, Debug)]
pub struct Leaf<'a, 's> {
    tree: &'a Tree<'s>,
    /// The node.
    pub id: Id,
}

impl<'a> Leaf<'a, '_> {
    /// The leaf's value.
    pub fn val_of_leaf(&self) -> &'a str {
        self.tree.val(self.id)
    }
}

/// `pattern.search(s) is not None`.
pub fn search(p: &Regex, s: &str) -> bool {
    let mut scratch = super::regex::Scratch::default();
    let mut pos = 0;
    loop {
        let c = s[pos..].chars().next();
        if p.may_start_with(c) && p.match_with(s, pos, &mut scratch).is_some() {
            return true;
        }
        match c {
            Some(c) => pos += c.len_utf8(),
            None => return false,
        }
    }
}
