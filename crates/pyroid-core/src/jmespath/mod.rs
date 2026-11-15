// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! A port of jmespath 1.1.0's lexer and parser (`pyroid.jmespath`).
//!
//! The lexer and parser follow `jmespath/lexer.py` and `jmespath/parser.py` method by
//! method: the same tokens (types, values, `start`/`end` positions in code points), the same
//! AST (including the quirks: `{a: b c: d}` and `f(a b)` parse, `a.b.c` and `a[0][1]` grow one
//! node in place), and the same errors (kind, position, token, message), at the point where
//! jmespath raises them. Expressions are code points (`&[u32]`), so lone surrogates lex as
//! in Python.
//!
//! Values (identifiers, numbers, string and JSON literals) are made by a [`Host`]: the
//! bindings use Python's `int()` and `json.loads`, as jmespath does; tests use a mock.
//!
//! [`lexer`] ports `Lexer.tokenize`, [`parser`] ports `Parser._parse` (producing
//! [`ast::Node`]s), and [`ast`] has the node kinds of `jmespath/ast.py`.

pub mod ast;
pub mod lexer;
pub mod parser;
#[cfg(test)]
mod tests;

pub use ast::{Comparator, Node};
pub use lexer::{LexError, LexResult, Token, TokenKind, TokenValue, tokenize};
pub use parser::{Budget, ParseError, ParseMessage, ParseResult, parse};

/// An expression's code points (lone surrogates allowed), as Python indexes a `str`.
pub type Text = Vec<u32>;

/// What the lexer asks the caller for: the values Python would compute.
pub trait Host {
    /// A token value, AST value or literal (a Python object in the bindings).
    type Value: Clone;
    /// An error the host raised (propagated unchanged, like a Python exception).
    type Error;

    /// An unquoted identifier (ASCII letters, digits, `_`).
    fn identifier(&mut self, text: &str) -> Result<Self::Value, Self::Error>;
    /// A number token: Python's `int(text)` (`text` is `-?[0-9]+`).
    fn number(&mut self, text: &str) -> Result<Self::Value, Self::Error>;
    /// A raw string literal (`'...'` with `\'` already unescaped).
    fn raw_string(&mut self, text: &[u32]) -> Result<Self::Value, Self::Error>;
    /// `json.loads(text)`: `Ok(Ok(value))`; `Ok(Err(str(e)))` when it raises a `ValueError`;
    /// `Err(e)` for any other exception.
    fn json(&mut self, text: &[u32]) -> Result<Result<Self::Value, Text>, Self::Error>;
    /// `warnings.warn("deprecated string literal syntax", PendingDeprecationWarning)`.
    fn deprecated_literal(&mut self) -> Result<(), Self::Error>;
}

/// `s` as code points.
pub fn code_points(s: &str) -> Text {
    s.chars().map(u32::from).collect()
}
