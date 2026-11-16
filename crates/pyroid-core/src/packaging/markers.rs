// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! packaging's `_tokenizer` and `_parser`: dependency specifiers (PEP 508) and environment
//! markers, with packaging 26.3's error messages and spans.
//!
//! The source is a slice of code points (a Python `str` can hold lone surrogates), and
//! positions are code-point indices, as packaging reports them. Quoted marker strings are
//! turned into values by the caller's [`Host`] (packaging uses `ast.literal_eval`).
//!
//! Marker trees are as deep as the input's parentheses, or as long as a loop of `m = m & other`
//! runs, so nothing here recurses: the parser keeps its open groups on a heap stack, and every
//! walk of a tree (formatting, comparing, dropping, and the evaluation and conversions in
//! `pyroid-py`) is a loop. Groups are shared ([`List`]), so combining two markers is O(1) and
//! `m & m` holds `m` once, as packaging's nested lists do.

use std::fmt;
use std::sync::Arc;

use crate::ftfy::UnicodeVersion;
use crate::text::is_py_whitespace;

/// A parse error: packaging's `ParserSyntaxError(message, span=...)`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct SyntaxError {
    /// The message (`"Expected ..."`).
    pub message: String,
    /// `(start, end)` in code points.
    pub span: (usize, usize),
}

/// A parse error, or an error from the [`Host`].
#[derive(Debug)]
pub enum Error<E> {
    /// packaging's `ParserSyntaxError`.
    Syntax(SyntaxError),
    /// The host's own error (for example a Python exception), passed through.
    Host(E),
}

/// What the parser asks of its caller.
pub trait Host {
    /// The host's error (a Python exception in `pyroid-py`).
    type Error;

    /// The value of a quoted string token (quotes included), as `ast.literal_eval` gives it;
    /// `Ok(None)` when packaging reports "Invalid quoted string".
    ///
    /// # Errors
    /// The host's own failure.
    fn quoted(&mut self, token: &[u32]) -> Result<Option<String>, Self::Error>;

    /// `canonicalize_name(value)` (`_normalize_extra_values` applies it to the literal compared
    /// with `extra`, `extras` or `dependency_groups`).
    ///
    /// # Errors
    /// The host's own failure.
    fn canonicalize(&mut self, value: &str) -> Result<String, Self::Error>;
}

/// One side of a marker comparison.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum MarkerVar {
    /// An environment variable (normalised: `os.name` is `os_name`).
    Variable(String),
    /// A quoted string's value.
    Value(String),
}

impl MarkerVar {
    /// The node's `value`.
    #[must_use]
    pub fn value(&self) -> &str {
        match self {
            MarkerVar::Variable(v) | MarkerVar::Value(v) => v,
        }
    }
}

/// An element of packaging's `MarkerList`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Elem {
    /// `(lhs, op, rhs)`.
    Item(MarkerVar, String, MarkerVar),
    /// A parenthesised (or combined) group.
    List(List),
    /// `"and"`.
    And,
    /// `"or"`.
    Or,
}

/// packaging's `MarkerList`: a group of [`Elem`]s, shared between the trees that hold it.
///
/// Cloning is O(1). Dropping, comparing and printing it with `{:?}` are loops, whatever the
/// depth.
#[derive(Clone, Default)]
pub struct List(Arc<Vec<Elem>>);

impl List {
    /// A group holding `items`.
    #[must_use]
    pub fn new(items: Vec<Elem>) -> Self {
        Self(Arc::new(items))
    }

    /// The group's elements.
    #[must_use]
    pub fn items(&self) -> &[Elem] {
        &self.0
    }

    /// The same group (shared, not equal): a key for memoising walks over shared sub-trees.
    #[must_use]
    pub fn id(&self) -> usize {
        Arc::as_ptr(&self.0).addr()
    }

    /// The elements to change in place, when this is the only reference to the group.
    pub fn get_mut(&mut self) -> Option<&mut Vec<Elem>> {
        Arc::get_mut(&mut self.0)
    }
}

impl std::ops::Deref for List {
    type Target = [Elem];
    fn deref(&self) -> &[Elem] {
        &self.0
    }
}

impl From<Vec<Elem>> for List {
    fn from(items: Vec<Elem>) -> Self {
        Self::new(items)
    }
}

impl Drop for List {
    fn drop(&mut self) {
        // The last reference to a group with sub-groups frees them with a work list, so a deep
        // tree never recurses. A shared group only loses a reference here.
        let Some(items) = Arc::get_mut(&mut self.0) else {
            return;
        };
        if !items.iter().any(|e| matches!(e, Elem::List(_))) {
            return;
        }
        let mut work = vec![std::mem::take(items)];
        while let Some(mut items) = work.pop() {
            for e in items.drain(..) {
                if let Elem::List(mut sub) = e {
                    if let Some(inner) = Arc::get_mut(&mut sub.0) {
                        if inner.iter().any(|e| matches!(e, Elem::List(_))) {
                            work.push(std::mem::take(inner));
                        }
                    }
                }
            }
        }
    }
}

impl PartialEq for List {
    fn eq(&self, other: &Self) -> bool {
        let mut work: Vec<(&[Elem], &[Elem])> = vec![(self, other)];
        while let Some((a, b)) = work.pop() {
            if a.len() != b.len() {
                return false;
            }
            for (x, y) in a.iter().zip(b) {
                match (x, y) {
                    (Elem::List(p), Elem::List(q)) => {
                        if !Arc::ptr_eq(&p.0, &q.0) {
                            work.push((p, q));
                        }
                    }
                    (Elem::Item(l1, o1, r1), Elem::Item(l2, o2, r2)) => {
                        if l1 != l2 || o1 != o2 || r1 != r2 {
                            return false;
                        }
                    }
                    (Elem::And, Elem::And) | (Elem::Or, Elem::Or) => {}
                    _ => return false,
                }
            }
        }
        true
    }
}

impl Eq for List {}

impl fmt::Debug for List {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `[Item(...), And, [...]]`, written with a work list (a derived Debug would recurse).
        enum Step<'a> {
            Open(&'a [Elem]),
            Elem(&'a Elem),
            Text(&'static str),
        }
        let mut work = vec![Step::Open(self)];
        while let Some(step) = work.pop() {
            match step {
                Step::Text(t) => f.write_str(t)?,
                Step::Elem(Elem::Item(l, o, r)) => write!(f, "Item({l:?}, {o:?}, {r:?})")?,
                Step::Elem(Elem::And) => f.write_str("And")?,
                Step::Elem(Elem::Or) => f.write_str("Or")?,
                Step::Elem(Elem::List(sub)) => work.push(Step::Open(sub)),
                Step::Open(sub) => {
                    f.write_str("[")?;
                    work.push(Step::Text("]"));
                    for (i, e) in sub.iter().enumerate().rev() {
                        work.push(Step::Elem(e));
                        if i > 0 {
                            work.push(Step::Text(", "));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

/// A parsed requirement; names, URL, extras and specifier are code-point spans of the source.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ParsedRequirement {
    /// The name.
    pub name: (usize, usize),
    /// The URL (`None`: no `@ url`).
    pub url: Option<(usize, usize)>,
    /// The extras, as written.
    pub extras: Vec<(usize, usize)>,
    /// The specifier tokens and commas, concatenated as packaging concatenates them.
    pub specifier: Vec<(usize, usize)>,
    /// The marker.
    pub marker: Option<List>,
}

const VARIABLES: &[&str] = &[
    "python_version",
    "python_full_version",
    "os_name",
    "os.name",
    "sys_platform",
    "sys.platform",
    "platform_release",
    "platform_system",
    "platform_version",
    "platform.version",
    "platform_machine",
    "platform.machine",
    "platform_python_implementation",
    "platform.python_implementation",
    "python_implementation",
    "implementation_name",
    "implementation_version",
    "extras",
    "extra",
    "dependency_groups",
];

/// The `Tokenizer`, with packaging's `DEFAULT_RULES`.
struct Tokenizer<'a> {
    s: &'a [u32],
    pos: usize,
    unicode: UnicodeVersion,
}

fn ch(c: u32) -> Option<char> {
    char::from_u32(c)
}

impl Tokenizer<'_> {
    fn at(&self, i: usize) -> Option<u32> {
        self.s.get(i).copied()
    }

    fn is_word(&self, i: usize) -> bool {
        self.at(i)
            .and_then(ch)
            .is_some_and(|c| crate::filters::unicode::is_word(c, self.unicode))
    }

    /// `\b` at `i`.
    fn boundary(&self, i: usize) -> bool {
        let before = i > 0 && self.is_word(i - 1);
        before != self.is_word(i)
    }

    fn lit(&self, i: usize, text: &str) -> bool {
        let t = text.as_bytes();
        self.s.len() >= i + t.len()
            && t.iter()
                .enumerate()
                .all(|(k, &b)| self.s[i + k] == u32::from(b))
    }

    fn is_ws(&self, i: usize) -> bool {
        matches!(self.at(i), Some(0x20 | 0x09))
    }

    /// The length of the `name` token at the current position, if it matches.
    fn rule(&self, name: Rule) -> Option<usize> {
        let p = self.pos;
        let c = self.at(p);
        let single = |x: u8| (c == Some(u32::from(x))).then_some(1);
        match name {
            Rule::LeftParen => single(b'('),
            Rule::RightParen => single(b')'),
            Rule::LeftBracket => single(b'['),
            Rule::RightBracket => single(b']'),
            Rule::Semicolon => single(b';'),
            Rule::Comma => single(b','),
            Rule::At => single(b'@'),
            Rule::QuotedString => {
                let q = c.filter(|&q| q == u32::from(b'\'') || q == u32::from(b'"'))?;
                let close = self.s[p + 1..].iter().position(|&x| x == q)?;
                Some(close + 2)
            }
            Rule::Op => ["===", "==", "~=", "!=", "<=", ">=", "<", ">"]
                .iter()
                .find(|o| self.lit(p, o))
                .map(|o| o.len()),
            Rule::BoolOp => self.word_token(&["or", "and"]),
            Rule::In => self.word_token(&["in"]),
            Rule::Not => self.word_token(&["not"]),
            Rule::Variable => self.word_token(VARIABLES),
            Rule::Specifier => crate::packaging::specifiers::match_len(&self.s[p..]),
            Rule::Url => {
                let n = self.s[p..]
                    .iter()
                    .take_while(|&&x| x != 0x20 && x != 0x09)
                    .count();
                (n > 0).then_some(n)
            }
            Rule::Identifier => {
                if !self.boundary(p) || !c.and_then(ch).is_some_and(|c| c.is_ascii_alphanumeric()) {
                    return None;
                }
                let mut e = p + 1;
                while self
                    .at(e)
                    .and_then(ch)
                    .is_some_and(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
                {
                    e += 1;
                }
                // Backtrack to the longest match that ends on a word boundary.
                while e > p + 1 && !self.boundary(e) {
                    e -= 1;
                }
                self.boundary(e).then_some(e - p)
            }
            Rule::VersionPrefixTrail => (self.lit(p, ".*")).then_some(2),
            Rule::VersionLocalLabelTrail => {
                if c != Some(u32::from(b'+')) {
                    return None;
                }
                let low = |x: Option<u32>| {
                    x.and_then(ch)
                        .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
                };
                let mut e = p + 1;
                if !low(self.at(e)) {
                    return None;
                }
                loop {
                    while low(self.at(e)) {
                        e += 1;
                    }
                    if self
                        .at(e)
                        .and_then(ch)
                        .is_some_and(|c| matches!(c, '-' | '_' | '.'))
                        && low(self.at(e + 1))
                    {
                        e += 1;
                    } else {
                        break;
                    }
                }
                Some(e - p)
            }
            Rule::Ws => {
                let n = (p..self.s.len()).take_while(|&i| self.is_ws(i)).count();
                (n > 0).then_some(n)
            }
            Rule::End => (p == self.s.len()).then_some(0),
        }
    }

    /// `\b(w1|w2|...)\b`, alternatives in order.
    fn word_token(&self, words: &[&str]) -> Option<usize> {
        let p = self.pos;
        if !self.boundary(p) {
            return None;
        }
        words
            .iter()
            .find(|w| self.lit(p, w) && self.boundary(p + w.len()))
            .map(|w| w.len())
    }

    fn check(&self, name: Rule) -> Option<usize> {
        self.rule(name)
    }

    fn consume(&mut self, name: Rule) {
        if let Some(n) = self.check(name) {
            self.pos += n;
        }
    }

    fn syntax<T>(
        &self,
        message: impl Into<String>,
        start: Option<usize>,
        end: Option<usize>,
    ) -> Result<T, SyntaxError> {
        Err(SyntaxError {
            message: message.into(),
            span: (start.unwrap_or(self.pos), end.unwrap_or(self.pos)),
        })
    }

    /// `expect(name, expected=...)`: the token's start and end.
    fn expect(&mut self, name: Rule, expected: &str) -> Result<(usize, usize), SyntaxError> {
        match self.check(name) {
            Some(n) => {
                let start = self.pos;
                self.pos += n;
                Ok((start, self.pos))
            }
            None => self.syntax(format!("Expected {expected}"), None, None),
        }
    }

    fn text(&self, span: (usize, usize)) -> String {
        self.s[span.0..span.1]
            .iter()
            .filter_map(|&c| ch(c))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
enum Rule {
    LeftParen,
    RightParen,
    LeftBracket,
    RightBracket,
    Semicolon,
    Comma,
    QuotedString,
    Op,
    BoolOp,
    In,
    Not,
    Variable,
    Specifier,
    At,
    Url,
    Identifier,
    VersionPrefixTrail,
    VersionLocalLabelTrail,
    Ws,
    End,
}

impl Rule {
    fn token(self) -> &'static str {
        match self {
            Rule::LeftParen => "LEFT_PARENTHESIS",
            Rule::RightParen => "RIGHT_PARENTHESIS",
            Rule::LeftBracket => "LEFT_BRACKET",
            Rule::RightBracket => "RIGHT_BRACKET",
            _ => "",
        }
    }
}

struct Parser<'a, H> {
    t: Tokenizer<'a>,
    host: &'a mut H,
}

type Res<T, E> = Result<T, Error<E>>;

fn syn<T, E>(r: Result<T, SyntaxError>) -> Res<T, E> {
    r.map_err(Error::Syntax)
}

impl<E, H: Host<Error = E>> Parser<'_, H> {
    /// `enclosing_tokens(open, close, around=...)`: the opening position, if it was there.
    fn open(&mut self, open: Rule) -> Option<usize> {
        let start = self.t.pos;
        self.t.check(open).map(|n| {
            self.t.pos += n;
            start
        })
    }

    fn close(
        &mut self,
        open_pos: Option<usize>,
        open: Rule,
        close: Rule,
        around: &str,
    ) -> Res<(), E> {
        let Some(start) = open_pos else {
            return Ok(());
        };
        match self.t.check(close) {
            Some(n) => {
                self.t.pos += n;
                Ok(())
            }
            None => syn(self.t.syntax(
                format!(
                    "Expected matching {} for {}, after {around}",
                    close.token(),
                    open.token()
                ),
                Some(start),
                None,
            )),
        }
    }

    fn requirement(&mut self) -> Res<ParsedRequirement, E> {
        self.t.consume(Rule::Ws);
        let name = syn(self.t.expect(
            Rule::Identifier,
            "package name at the start of dependency specifier",
        ))?;
        self.t.consume(Rule::Ws);
        let extras = self.extras()?;
        self.t.consume(Rule::Ws);
        let (url, specifier, marker) = self.requirement_details()?;
        syn(self.t.expect(Rule::End, "end of dependency specifier"))?;
        Ok(ParsedRequirement {
            name,
            url,
            extras,
            specifier,
            marker,
        })
    }

    #[allow(clippy::type_complexity)]
    fn requirement_details(
        &mut self,
    ) -> Res<(Option<(usize, usize)>, Vec<(usize, usize)>, Option<List>), E> {
        if let Some(n) = self.t.check(Rule::At) {
            self.t.pos += n;
            self.t.consume(Rule::Ws);
            let url_start = self.t.pos;
            let url = syn(self.t.expect(Rule::Url, "URL after @"))?;
            if self.t.check(Rule::End).is_some() {
                return Ok((Some(url), Vec::new(), None));
            }
            syn(self.t.expect(Rule::Ws, "whitespace after URL"))?;
            if self.t.check(Rule::End).is_some() {
                return Ok((Some(url), Vec::new(), None));
            }
            let marker =
                self.requirement_marker(url_start, "semicolon (after URL and whitespace)")?;
            return Ok((Some(url), Vec::new(), Some(marker)));
        }
        let specifier_start = self.t.pos;
        let specifier = self.specifier()?;
        self.t.consume(Rule::Ws);
        if self.t.check(Rule::End).is_some() {
            return Ok((None, specifier, None));
        }
        let expected = if specifier.is_empty() {
            "semicolon (after name with no version specifier)"
        } else {
            "comma (within version specifier), semicolon (after version specifier)"
        };
        let marker = self.requirement_marker(specifier_start, expected)?;
        Ok((None, specifier, Some(marker)))
    }

    fn requirement_marker(&mut self, span_start: usize, expected: &str) -> Res<List, E> {
        match self.t.check(Rule::Semicolon) {
            Some(n) => self.t.pos += n,
            None => {
                return syn(self.t.syntax(
                    format!("Expected {expected} or end"),
                    Some(span_start),
                    None,
                ));
            }
        }
        let marker = self.marker()?;
        self.t.consume(Rule::Ws);
        Ok(marker)
    }

    fn extras(&mut self) -> Res<Vec<(usize, usize)>, E> {
        if self.t.check(Rule::LeftBracket).is_none() {
            return Ok(Vec::new());
        }
        let open = self.open(Rule::LeftBracket);
        self.t.consume(Rule::Ws);
        let extras = self.extras_list()?;
        self.t.consume(Rule::Ws);
        self.close(open, Rule::LeftBracket, Rule::RightBracket, "extras")?;
        Ok(extras)
    }

    fn extras_list(&mut self) -> Res<Vec<(usize, usize)>, E> {
        let mut extras = Vec::new();
        let Some(n) = self.t.check(Rule::Identifier) else {
            return Ok(extras);
        };
        extras.push((self.t.pos, self.t.pos + n));
        self.t.pos += n;
        loop {
            self.t.consume(Rule::Ws);
            if self.t.check(Rule::Identifier).is_some() {
                return syn(self
                    .t
                    .syntax("Expected comma between extra names", None, None));
            }
            let Some(n) = self.t.check(Rule::Comma) else {
                break;
            };
            self.t.pos += n;
            self.t.consume(Rule::Ws);
            extras.push(syn(self
                .t
                .expect(Rule::Identifier, "extra name after comma"))?);
        }
        Ok(extras)
    }

    fn specifier(&mut self) -> Res<Vec<(usize, usize)>, E> {
        let open = self.open(Rule::LeftParen);
        self.t.consume(Rule::Ws);
        let parsed = self.version_many()?;
        self.t.consume(Rule::Ws);
        self.close(open, Rule::LeftParen, Rule::RightParen, "version specifier")?;
        Ok(parsed)
    }

    fn version_many(&mut self) -> Res<Vec<(usize, usize)>, E> {
        let mut parts = Vec::new();
        while let Some(n) = self.t.check(Rule::Specifier) {
            let span_start = self.t.pos;
            let token = (self.t.pos, self.t.pos + n);
            self.t.pos += n;
            parts.push(token);
            if self.t.check(Rule::VersionPrefixTrail).is_some() {
                let starts = |p: &str| self.t.lit(span_start, p);
                let message = if starts("!=") || (starts("==") && !starts("===")) {
                    ".* suffix cannot be used with pre-release, post-release, dev or local versions"
                } else {
                    ".* suffix can only be used with `==` or `!=` operators"
                };
                return syn(self
                    .t
                    .syntax(message, Some(span_start), Some(self.t.pos + 1)));
            }
            if self.t.check(Rule::VersionLocalLabelTrail).is_some() {
                return syn(self.t.syntax(
                    "Local version label can only be used with `==` or `!=` operators",
                    Some(span_start),
                    None,
                ));
            }
            self.t.consume(Rule::Ws);
            let Some(c) = self.t.check(Rule::Comma) else {
                break;
            };
            parts.push((self.t.pos, self.t.pos + c));
            self.t.pos += c;
            self.t.consume(Rule::Ws);
        }
        Ok(parts)
    }

    fn full_marker(&mut self) -> Res<List, E> {
        let m = self.marker()?;
        syn(self.t.expect(Rule::End, "end of marker expression"))?;
        Ok(m)
    }

    /// `_parse_marker`: `marker_atom (BOOLOP marker_atom)*`, where
    /// `marker_atom = WS? '(' WS? marker WS? ')' WS? | WS? marker_item WS?`.
    ///
    /// packaging recurses through `_parse_marker_atom` for each `(`; this keeps the open
    /// groups (their elements so far and where their `(` was) on a heap stack instead.
    fn marker(&mut self) -> Res<List, E> {
        // The innermost open group is last; the first entry is the marker itself (no `(`).
        let mut open: Vec<(Vec<Elem>, Option<usize>)> = vec![(Vec::new(), None)];
        loop {
            // The start of a marker_atom.
            self.t.consume(Rule::Ws);
            if self.t.check(Rule::LeftParen).is_some() {
                let at = self.open(Rule::LeftParen);
                self.t.consume(Rule::Ws);
                open.push((Vec::new(), at));
                continue;
            }
            let mut atom = self.marker_item()?;
            self.t.consume(Rule::Ws);
            // The atom is done: add it to its group, then close every group that ends here.
            loop {
                let Some((items, _)) = open.last_mut() else {
                    unreachable!("the outermost group is never closed here")
                };
                items.push(atom);
                if let Some(n) = self.t.check(Rule::BoolOp) {
                    let op = if self.t.lit(self.t.pos, "or") {
                        Elem::Or
                    } else {
                        Elem::And
                    };
                    self.t.pos += n;
                    items.push(op);
                    break;
                }
                let Some((items, at)) = open.pop() else {
                    unreachable!("the outermost group is never closed here")
                };
                if at.is_none() {
                    return Ok(List::new(items));
                }
                // The rest of the parenthesised marker_atom.
                self.t.consume(Rule::Ws);
                self.close(at, Rule::LeftParen, Rule::RightParen, "marker expression")?;
                self.t.consume(Rule::Ws);
                atom = Elem::List(List::new(items));
            }
        }
    }

    fn marker_item(&mut self) -> Res<Elem, E> {
        self.t.consume(Rule::Ws);
        let lhs = self.marker_var()?;
        self.t.consume(Rule::Ws);
        let op = self.marker_op()?;
        self.t.consume(Rule::Ws);
        let rhs = self.marker_var()?;
        self.t.consume(Rule::Ws);
        let (lhs, rhs) = self.normalize_extras(lhs, rhs).map_err(Error::Host)?;
        Ok(Elem::Item(lhs, op, rhs))
    }

    /// `_normalize_extras` for one item: the literal compared with `extra`, `extras` or
    /// `dependency_groups` is canonicalised.
    fn normalize_extras(
        &mut self,
        lhs: MarkerVar,
        rhs: MarkerVar,
    ) -> Result<(MarkerVar, MarkerVar), E> {
        Ok(match (lhs, rhs) {
            (MarkerVar::Variable(v), MarkerVar::Value(val)) if v == "extra" => {
                let val = self.host.canonicalize(&val)?;
                (MarkerVar::Variable(v), MarkerVar::Value(val))
            }
            (MarkerVar::Value(val), MarkerVar::Variable(v))
                if v == "extra" || v == "extras" || v == "dependency_groups" =>
            {
                (
                    MarkerVar::Value(self.host.canonicalize(&val)?),
                    MarkerVar::Variable(v),
                )
            }
            other => other,
        })
    }

    fn marker_var(&mut self) -> Res<MarkerVar, E> {
        if let Some(n) = self.t.check(Rule::Variable) {
            let name = self.t.text((self.t.pos, self.t.pos + n)).replace('.', "_");
            self.t.pos += n;
            let name = if name == "python_implementation" {
                "platform_python_implementation".to_owned()
            } else {
                name
            };
            return Ok(MarkerVar::Variable(name));
        }
        if let Some(n) = self.t.check(Rule::QuotedString) {
            let start = self.t.pos;
            self.t.pos += n;
            let token = &self.t.s[start..start + n];
            return match self.host.quoted(token) {
                Ok(Some(v)) => Ok(MarkerVar::Value(v)),
                Ok(None) => Err(Error::Syntax(SyntaxError {
                    message: "Invalid quoted string".into(),
                    span: (start, start + n),
                })),
                Err(e) => Err(Error::Host(e)),
            };
        }
        syn(self
            .t
            .syntax("Expected a marker variable or quoted string", None, None))
    }

    fn marker_op(&mut self) -> Res<String, E> {
        if let Some(n) = self.t.check(Rule::In) {
            self.t.pos += n;
            return Ok("in".into());
        }
        if let Some(n) = self.t.check(Rule::Not) {
            self.t.pos += n;
            syn(self.t.expect(Rule::Ws, "whitespace after 'not'"))?;
            syn(self.t.expect(Rule::In, "'in' after 'not'"))?;
            return Ok("not in".into());
        }
        if let Some(n) = self.t.check(Rule::Op) {
            let op = self.t.text((self.t.pos, self.t.pos + n));
            self.t.pos += n;
            return Ok(op);
        }
        syn(self.t.syntax(
            "Expected marker operator, one of <=, <, !=, ==, >=, >, ~=, ===, in, not in",
            None,
            None,
        ))
    }
}

/// The value of a quoted marker string that `ast.literal_eval` returns as is: no backslash,
/// line break, NUL or surrogate inside. `None`: the caller must ask Python.
#[must_use]
pub fn simple_quoted(token: &[u32]) -> Option<String> {
    let inner = token.get(1..token.len().checked_sub(1)?)?;
    let mut out = String::with_capacity(inner.len());
    for &c in inner {
        let c = char::from_u32(c)?;
        if matches!(c, '\\' | '\n' | '\r' | '\0') {
            return None;
        }
        out.push(c);
    }
    Some(out)
}

/// `parse_marker(source)`, with `_normalize_extra_values` applied.
///
/// # Errors
/// packaging's syntax errors, or the host's.
pub fn parse_marker<H: Host>(
    source: &[u32],
    unicode: UnicodeVersion,
    host: &mut H,
) -> Result<List, Error<H::Error>> {
    let mut p = Parser {
        t: Tokenizer {
            s: source,
            pos: 0,
            unicode,
        },
        host,
    };
    p.full_marker()
}

/// `parse_requirement(source)`, with the marker's extra values normalised.
///
/// # Errors
/// packaging's syntax errors, or the host's.
pub fn parse_requirement<H: Host>(
    source: &[u32],
    unicode: UnicodeVersion,
    host: &mut H,
) -> Result<ParsedRequirement, Error<H::Error>> {
    let mut p = Parser {
        t: Tokenizer {
            s: source,
            pos: 0,
            unicode,
        },
        host,
    };
    p.requirement()
}

/// `str(ParserSyntaxError)`: the message, the source and a marker line under the span.
#[must_use]
pub fn format_error(err: &SyntaxError, source: &str) -> String {
    let (a, b) = err.span;
    format!(
        "{}\n    {source}\n    {}{}^",
        err.message,
        " ".repeat(a),
        "~".repeat(b.saturating_sub(a))
    )
}

/// `canonicalize_name(name)` for an ASCII `name` (others need Python's `str.lower`).
#[must_use]
pub fn canonicalize_ascii(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_dash = false;
    for b in name.bytes() {
        let c = match b {
            b'_' | b'.' | b'-' => '-',
            _ => char::from(b.to_ascii_lowercase()),
        };
        if c == '-' {
            if last_dash {
                continue;
            }
            last_dash = true;
        } else {
            last_dash = false;
        }
        out.push(c);
    }
    out
}

/// `canonicalize_name` after Python's `str.lower()` (`lowered`): `_`/`.` to `-`, runs of `-`
/// collapsed.
#[must_use]
pub fn canonicalize_lowered(lowered: &str) -> String {
    let mut out = String::with_capacity(lowered.len());
    let mut last_dash = false;
    for c in lowered.chars() {
        let c = if matches!(c, '_' | '.') { '-' } else { c };
        if c == '-' {
            if last_dash {
                continue;
            }
            last_dash = true;
        } else {
            last_dash = false;
        }
        out.push(c);
    }
    out
}

/// `Value.serialize()`: double quotes unless the value has one, then single quotes; `None`
/// when it has both (packaging raises `ValueError`).
#[must_use]
pub fn serialize_value(v: &str) -> Option<String> {
    if !v.contains('"') {
        Some(format!("\"{v}\""))
    } else if !v.contains('\'') {
        Some(format!("'{v}'"))
    } else {
        None
    }
}

/// A marker value holds both quote characters, so it cannot be written (packaging raises
/// `ValueError`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QuoteError;

/// `_format_marker(marker, first)`; `Err` when a value has both quote characters.
///
/// packaging recurses into each group; this walks the tree with a work list. A group that a
/// tree holds several times (`m & m`) is written each time, as packaging writes it.
///
/// # Errors
/// [`QuoteError`] for a value that cannot be serialised.
pub fn format_marker(marker: &[Elem], first: bool) -> Result<String, QuoteError> {
    enum Step<'a> {
        Group(&'a [Elem], bool),
        Text(&'static str),
    }
    let mut out = String::new();
    let mut work = vec![Step::Group(marker, first)];
    while let Some(step) = work.pop() {
        let (mut items, first) = match step {
            Step::Text(t) => {
                out.push_str(t);
                continue;
            }
            Step::Group(items, first) => (items, first),
        };
        // `[[...]]` and `[(...)]`: unwrap, keeping `first` (a single item is written bare).
        loop {
            match items {
                [Elem::List(inner)] => items = inner,
                [Elem::Item(l, op, r)] => {
                    write_item(&mut out, l, op, r)?;
                    break;
                }
                _ => {
                    if !first {
                        out.push('(');
                        work.push(Step::Text(")"));
                    }
                    for (i, e) in items.iter().enumerate().rev() {
                        work.push(match e {
                            Elem::List(sub) => Step::Group(sub, false),
                            Elem::Item(..) => Step::Group(std::slice::from_ref(e), false),
                            Elem::And => Step::Text("and"),
                            Elem::Or => Step::Text("or"),
                        });
                        if i > 0 {
                            work.push(Step::Text(" "));
                        }
                    }
                    break;
                }
            }
        }
    }
    Ok(out)
}

fn write_item(out: &mut String, l: &MarkerVar, op: &str, r: &MarkerVar) -> Result<(), QuoteError> {
    let side = |out: &mut String, v: &MarkerVar| -> Result<(), QuoteError> {
        match v {
            MarkerVar::Variable(n) => out.push_str(n),
            MarkerVar::Value(s) => out.push_str(&serialize_value(s).ok_or(QuoteError)?),
        }
        Ok(())
    };
    side(out, l)?;
    out.push(' ');
    out.push_str(op);
    out.push(' ');
    side(out, r)
}

/// `str.strip()` (Python whitespace).
#[must_use]
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_whitespace)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Plain;

    impl Host for Plain {
        type Error = ();
        fn quoted(&mut self, token: &[u32]) -> Result<Option<String>, ()> {
            Ok(simple_quoted(token))
        }
        fn canonicalize(&mut self, value: &str) -> Result<String, ()> {
            Ok(canonicalize_ascii(value))
        }
    }

    fn cps(s: &str) -> Vec<u32> {
        s.chars().map(u32::from).collect()
    }

    fn parse(s: &str) -> Result<List, SyntaxError> {
        match parse_marker(&cps(s), UnicodeVersion::V17_0, &mut Plain) {
            Ok(l) => Ok(l),
            Err(Error::Syntax(e)) => Err(e),
            Err(Error::Host(())) => unreachable!(),
        }
    }

    fn text(s: &str) -> String {
        format_marker(&parse(s).unwrap(), true).unwrap()
    }

    /// Run `f` on a thread with a small stack: any recursion per nesting level overflows it.
    fn small_stack(f: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(64 << 10)
            .spawn(f)
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn formats_as_packaging_does() {
        assert_eq!(text("os_name=='a'"), r#"os_name == "a""#);
        assert_eq!(text("((os_name=='a'))"), r#"os_name == "a""#);
        assert_eq!(
            text("os_name=='a' and (python_version>'3' or extra=='X_Y')"),
            r#"os_name == "a" and (python_version > "3" or extra == "x-y")"#
        );
        assert_eq!(
            text("(os_name=='a' or os_name=='b') and ((sys_platform=='c'))"),
            r#"(os_name == "a" or os_name == "b") and sys_platform == "c""#
        );
        assert_eq!(text("'Foo.Bar' in extras"), r#""foo-bar" in extras"#);
        assert_eq!(format_marker(&List::default(), true).unwrap(), "");
        assert_eq!(format_marker(&List::default(), false).unwrap(), "()");
    }

    #[test]
    fn syntax_errors_keep_packagings_spans() {
        let e = parse("(os_name == 'a'").unwrap_err();
        assert_eq!(
            e.message,
            "Expected matching RIGHT_PARENTHESIS for LEFT_PARENTHESIS, after marker expression"
        );
        assert_eq!(e.span, (0, 15));
        let e = parse("os_name == 'a' and").unwrap_err();
        assert_eq!(e.message, "Expected a marker variable or quoted string");
        assert_eq!(e.span, (18, 18));
        let e = parse("(((").unwrap_err();
        assert_eq!(e.message, "Expected a marker variable or quoted string");
        assert_eq!(e.span, (3, 3));
        let e = parse("os_name == 'a')").unwrap_err();
        assert_eq!(e.message, "Expected end of marker expression");
    }

    #[test]
    fn deep_nesting_needs_no_stack() {
        small_stack(|| {
            let n = 200_000;
            let src = format!("{}os_name == 'nt'{}", "(".repeat(n), ")".repeat(n));
            let tree = parse(&src).unwrap();
            assert_eq!(format_marker(&tree, true).unwrap(), r#"os_name == "nt""#);
            assert_eq!(tree, parse(&src).unwrap());
            assert!(format!("{tree:?}").starts_with("[[[["));
            let e = parse(&"(".repeat(n)).unwrap_err();
            assert_eq!(e.span, (n, n));
            // Deep with a sibling at every level: printed with parentheses all the way down.
            let src = format!(
                "{}os_name == 'a'{}",
                "(os_name == 'b' or ".repeat(1000),
                ")".repeat(1000)
            );
            let printed = format_marker(&parse(&src).unwrap(), true).unwrap();
            assert_eq!(printed.matches('(').count(), 999);
            drop(tree);
        });
    }

    #[test]
    fn combined_groups_are_shared() {
        small_stack(|| {
            let other = parse("os_name == 'b'").unwrap();
            let mut m = parse("os_name == 'a'").unwrap();
            for _ in 0..100_000 {
                m = List::new(vec![Elem::List(m), Elem::And, Elem::List(other.clone())]);
            }
            let text = format_marker(&m, true).unwrap();
            assert!(text.starts_with(&format!("{}os_name == \"a\" and", "(".repeat(99_999))));
            let copy = m.clone();
            assert_eq!(copy.id(), m.id());
            assert_eq!(copy, m);
            drop(copy);
            drop(m);
            // m & m: one group held twice at each level, so 40 levels stay small.
            let mut m = parse("os_name == 'a'").unwrap();
            for _ in 0..40 {
                m = List::new(vec![Elem::List(m.clone()), Elem::Or, Elem::List(m)]);
            }
            let Elem::List(l) = &m[0] else { unreachable!() };
            let Elem::List(r) = &m[2] else { unreachable!() };
            assert_eq!(l.id(), r.id());
        });
    }
}
