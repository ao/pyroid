// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! packaging's `_tokenizer` and `_parser`: dependency specifiers (PEP 508) and environment
//! markers, with packaging 26.3's error messages and spans.
//!
//! The source is a slice of code points (a Python `str` can hold lone surrogates), and
//! positions are code-point indices, as packaging reports them. Quoted marker strings are
//! turned into values by the caller's `quoted` callback (packaging uses `ast.literal_eval`).

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

/// A parse error, or an error from the `quoted` callback.
#[derive(Debug)]
pub enum Error<E> {
    /// packaging's `ParserSyntaxError`.
    Syntax(SyntaxError),
    /// The callback's own error (for example a Python exception), passed through.
    Host(E),
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
    List(Vec<Elem>),
    /// `"and"`.
    And,
    /// `"or"`.
    Or,
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
    pub marker: Option<Vec<Elem>>,
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

struct Parser<'a, F> {
    t: Tokenizer<'a>,
    quoted: F,
}

type Res<T, E> = Result<T, Error<E>>;

fn syn<T, E>(r: Result<T, SyntaxError>) -> Res<T, E> {
    r.map_err(Error::Syntax)
}

impl<E, F: FnMut(&[u32]) -> Result<Option<String>, E>> Parser<'_, F> {
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
    ) -> Res<
        (
            Option<(usize, usize)>,
            Vec<(usize, usize)>,
            Option<Vec<Elem>>,
        ),
        E,
    > {
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

    fn requirement_marker(&mut self, span_start: usize, expected: &str) -> Res<Vec<Elem>, E> {
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

    fn full_marker(&mut self) -> Res<Vec<Elem>, E> {
        let m = self.marker()?;
        syn(self.t.expect(Rule::End, "end of marker expression"))?;
        Ok(m)
    }

    fn marker(&mut self) -> Res<Vec<Elem>, E> {
        let mut expression = vec![self.marker_atom()?];
        while let Some(n) = self.t.check(Rule::BoolOp) {
            let op = if self.t.lit(self.t.pos, "or") {
                Elem::Or
            } else {
                Elem::And
            };
            self.t.pos += n;
            let right = self.marker_atom()?;
            expression.push(op);
            expression.push(right);
        }
        Ok(expression)
    }

    fn marker_atom(&mut self) -> Res<Elem, E> {
        self.t.consume(Rule::Ws);
        let marker = if self.t.check(Rule::LeftParen).is_some() {
            let open = self.open(Rule::LeftParen);
            self.t.consume(Rule::Ws);
            let inner = self.marker()?;
            self.t.consume(Rule::Ws);
            self.close(open, Rule::LeftParen, Rule::RightParen, "marker expression")?;
            Elem::List(inner)
        } else {
            self.marker_item()?
        };
        self.t.consume(Rule::Ws);
        Ok(marker)
    }

    fn marker_item(&mut self) -> Res<Elem, E> {
        self.t.consume(Rule::Ws);
        let lhs = self.marker_var()?;
        self.t.consume(Rule::Ws);
        let op = self.marker_op()?;
        self.t.consume(Rule::Ws);
        let rhs = self.marker_var()?;
        self.t.consume(Rule::Ws);
        Ok(Elem::Item(lhs, op, rhs))
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
            return match (self.quoted)(token) {
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

/// `parse_marker(source)`.
///
/// # Errors
/// packaging's syntax errors, or the `quoted` callback's.
pub fn parse_marker<E>(
    source: &[u32],
    unicode: UnicodeVersion,
    quoted: impl FnMut(&[u32]) -> Result<Option<String>, E>,
) -> Result<Vec<Elem>, Error<E>> {
    let mut p = Parser {
        t: Tokenizer {
            s: source,
            pos: 0,
            unicode,
        },
        quoted,
    };
    p.full_marker()
}

/// `parse_requirement(source)`.
///
/// # Errors
/// packaging's syntax errors, or the `quoted` callback's.
pub fn parse_requirement<E>(
    source: &[u32],
    unicode: UnicodeVersion,
    quoted: impl FnMut(&[u32]) -> Result<Option<String>, E>,
) -> Result<ParsedRequirement, Error<E>> {
    let mut p = Parser {
        t: Tokenizer {
            s: source,
            pos: 0,
            unicode,
        },
        quoted,
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
/// # Errors
/// [`QuoteError`] for a value that cannot be serialised.
pub fn format_marker(marker: &[Elem], first: bool) -> Result<String, QuoteError> {
    if marker.len() == 1 {
        if let Elem::List(inner) = &marker[0] {
            return format_marker(inner, first);
        }
        if let Elem::Item(..) = &marker[0] {
            return format_elem(&marker[0], first);
        }
    }
    let mut parts = Vec::with_capacity(marker.len());
    for m in marker {
        parts.push(format_elem(m, false)?);
    }
    let joined = parts.join(" ");
    Ok(if first { joined } else { format!("({joined})") })
}

fn format_elem(e: &Elem, first: bool) -> Result<String, QuoteError> {
    match e {
        Elem::List(inner) => format_marker(inner, first),
        Elem::Item(l, op, r) => {
            let side = |v: &MarkerVar| match v {
                MarkerVar::Variable(n) => Ok(n.clone()),
                MarkerVar::Value(s) => serialize_value(s).ok_or(QuoteError),
            };
            Ok(format!("{} {op} {}", side(l)?, side(r)?))
        }
        Elem::And => Ok("and".into()),
        Elem::Or => Ok("or".into()),
    }
}

/// `str.strip()` (Python whitespace).
#[must_use]
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_whitespace)
}
