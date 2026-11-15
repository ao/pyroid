// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! `jmespath/parser.py`: the top-down operator precedence (Pratt) parser.
//!
//! Each method mirrors the Python method of the same name. jmespath's parser is recursive,
//! so it raises `RecursionError` on deeply nested expressions; [`Budget`] counts the Python
//! frames jmespath would have open and stops where it would.

use super::ast::{Comparator, Node};
use super::lexer::{LexError, Token, TokenKind, TokenValue, tokenize};
use super::{Host, STACK_BUDGET, stack_position};

/// How many Python frames jmespath's parser may open below `Parser._parse`, and how much
/// native stack the port may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Budget {
    pub frames: usize,
    /// Bytes of stack below `parse`'s own frame.
    pub stack: usize,
}

impl Budget {
    /// No frame limit (the hard cap and [`STACK_BUDGET`] still apply).
    pub const UNLIMITED: Self = Self {
        frames: usize::MAX,
        stack: STACK_BUDGET,
    };
}

/// Nesting is cut off here whatever the budget, so the Rust stack stays small
/// (`RecursionError`, as jmespath at a raised recursion limit would eventually give).
pub const MAX_FRAMES: usize = 2000;

/// The message of a `ParseError`.
#[derive(Clone, Debug, PartialEq)]
pub enum ParseMessage {
    /// `'invalid token'`.
    InvalidToken,
    /// `'syntax error'`.
    SyntaxError,
    /// `"Unexpected token: %s" % value`.
    UnexpectedToken,
    /// `'Expecting: %s, got: %s' % (expected, actual_type)`, `expected` a type name.
    Expecting(TokenKind),
    /// The same with `['quoted_identifier', 'unquoted_identifier']` (a Python list).
    ExpectingIdentifier,
    /// `_parse_dot_rhs`: `"Expecting: %s, got: %s" % (allowed, type)`.
    ExpectingDotRhs,
    /// `"Invalid function name '%s'" % value`.
    InvalidFunctionName,
    /// `'Quoted identifier not allowed for function names.'`
    QuotedFunctionName,
}

/// Why parsing failed.
#[derive(Clone, Debug, PartialEq)]
pub enum ParseError<V, E> {
    /// From the lexer (`LexerError`, `EmptyExpressionError`, or the host).
    Lex(LexError<E>),
    /// `IncompleteExpressionError(start, value, type)` for the eof token.
    Incomplete { token: Token<V> },
    /// `ParseError(lex_position, token value, token type, message)`.
    Parse {
        lex_position: usize,
        token: Token<V>,
        message: ParseMessage,
    },
    /// Python would have run out of frames (`RecursionError`).
    Recursion,
}

/// The AST, or why parsing failed.
pub type ParseResult<V, E> = Result<Node<V>, ParseError<V, E>>;

/// Parse `expression` (code points) into jmespath's AST.
pub fn parse<H: Host>(
    expression: &[u32],
    host: &mut H,
    budget: Budget,
) -> ParseResult<H::Value, H::Error> {
    let tokens = tokenize(expression, host).map_err(|(_, e)| ParseError::Lex(e))?;
    let mut p = Parser {
        tokens,
        index: 0,
        depth: 0,
        limit: budget.frames.min(MAX_FRAMES),
        stack_base: stack_position(),
        stack: budget.stack,
    };
    let parsed = p.expression(0)?;
    if p.current() != TokenKind::Eof {
        let t = p.lookahead_token(0).clone();
        return Err(ParseError::Parse {
            lex_position: t.start,
            token: t,
            message: ParseMessage::UnexpectedToken,
        });
    }
    Ok(parsed)
}

type R<V, E, T = Node<V>> = Result<T, ParseError<V, E>>;

struct Parser<V> {
    tokens: Vec<Token<V>>,
    index: usize,
    /// Emulated Python frames open below `_parse`.
    depth: usize,
    limit: usize,
    stack_base: usize,
    stack: usize,
}

fn boxed<V>(n: Node<V>) -> Box<Node<V>> {
    Box::new(n)
}

impl<V: Clone> Parser<V> {
    /// Enter a method (one Python frame, plus one for the helpers it calls).
    fn enter<E>(&mut self) -> R<V, E, ()> {
        if self.depth + 2 > self.limit || stack_position().abs_diff(self.stack_base) > self.stack {
            return Err(ParseError::Recursion);
        }
        self.depth += 1;
        Ok(())
    }

    fn leave(&mut self) {
        self.depth -= 1;
    }

    fn advance(&mut self) {
        self.index += 1;
    }

    fn current(&self) -> TokenKind {
        self.tokens[self.index].kind
    }

    fn lookahead(&self, n: usize) -> TokenKind {
        self.tokens[self.index + n].kind
    }

    fn lookahead_token(&self, n: usize) -> &Token<V> {
        &self.tokens[self.index + n]
    }

    fn expression<E>(&mut self, binding_power: u32) -> R<V, E> {
        self.enter()?;
        let r = self.expression_body(binding_power);
        self.leave();
        r
    }

    fn expression_body<E>(&mut self, binding_power: u32) -> R<V, E> {
        let left_token = self.lookahead_token(0).clone();
        self.advance();
        let mut left = self.nud(left_token)?;
        let mut current = self.current();
        while binding_power < current.binding_power() {
            if !has_led(current) {
                let t = self.lookahead_token(0).clone();
                return Err(Self::raise_for_token(t, ParseMessage::InvalidToken));
            }
            self.advance();
            left = self.led(current, left)?;
            current = self.current();
        }
        Ok(left)
    }

    fn nud<E>(&mut self, token: Token<V>) -> R<V, E> {
        self.enter()?;
        let r = self.nud_body(token);
        self.leave();
        r
    }

    fn nud_body<E>(&mut self, token: Token<V>) -> R<V, E> {
        match token.kind {
            TokenKind::Literal => Ok(Node::Literal(value(token))),
            TokenKind::UnquotedIdentifier => Ok(Node::Field(value(token))),
            TokenKind::QuotedIdentifier => {
                let field = Node::Field(value(token));
                // A quoted identifier can't be a function name.
                if self.current() == TokenKind::Lparen {
                    let t = self.lookahead_token(0).clone();
                    return Err(ParseError::Parse {
                        lex_position: 0,
                        token: t,
                        message: ParseMessage::QuotedFunctionName,
                    });
                }
                Ok(field)
            }
            TokenKind::Star => {
                let left = Node::Identity;
                let right = if self.current() == TokenKind::Rbracket {
                    Node::Identity
                } else {
                    self.parse_projection_rhs(TokenKind::Star.binding_power())?
                };
                Ok(Node::ValueProjection(boxed(left), boxed(right)))
            }
            TokenKind::Filter => {
                // `_token_nud_filter` calls `_token_led_filter`: one more frame.
                self.enter()?;
                let r = self.led_filter(Node::Identity);
                self.leave();
                r
            }
            TokenKind::Lbrace => self.parse_multi_select_hash(),
            TokenKind::Lparen => {
                let expression = self.expression(0)?;
                self.match_(TokenKind::Rparen)?;
                Ok(expression)
            }
            TokenKind::Flatten => {
                let left = Node::Flatten(boxed(Node::Identity));
                let right = self.parse_projection_rhs(TokenKind::Flatten.binding_power())?;
                Ok(Node::Projection(boxed(left), boxed(right)))
            }
            TokenKind::Not => {
                let expr = self.expression(TokenKind::Not.binding_power())?;
                Ok(Node::NotExpression(boxed(expr)))
            }
            TokenKind::Lbracket => {
                if matches!(self.current(), TokenKind::Number | TokenKind::Colon) {
                    let right = self.parse_index_expression()?;
                    self.project_if_slice(Node::Identity, right)
                } else if self.current() == TokenKind::Star
                    && self.lookahead(1) == TokenKind::Rbracket
                {
                    self.advance();
                    self.advance();
                    let right = self.parse_projection_rhs(TokenKind::Star.binding_power())?;
                    Ok(Node::Projection(boxed(Node::Identity), boxed(right)))
                } else {
                    self.parse_multi_select_list()
                }
            }
            TokenKind::Current => Ok(Node::Current),
            TokenKind::Expref => {
                let expression = self.expression(TokenKind::Expref.binding_power())?;
                Ok(Node::Expref(boxed(expression)))
            }
            TokenKind::Eof => Err(ParseError::Incomplete { token }),
            _ => Err(Self::raise_for_token(token, ParseMessage::InvalidToken)),
        }
    }

    fn led<E>(&mut self, kind: TokenKind, left: Node<V>) -> R<V, E> {
        self.enter()?;
        let r = self.led_body(kind, left);
        self.leave();
        r
    }

    fn led_body<E>(&mut self, kind: TokenKind, left: Node<V>) -> R<V, E> {
        match kind {
            TokenKind::Dot => {
                if self.current() == TokenKind::Star {
                    // A projection.
                    self.advance();
                    let right = self.parse_projection_rhs(TokenKind::Dot.binding_power())?;
                    Ok(Node::ValueProjection(boxed(left), boxed(right)))
                } else {
                    let right = self.parse_dot_rhs(TokenKind::Dot.binding_power())?;
                    match left {
                        Node::Subexpression(mut children) => {
                            children.push(right);
                            Ok(Node::Subexpression(children))
                        }
                        left => Ok(Node::Subexpression(vec![left, right])),
                    }
                }
            }
            TokenKind::Pipe => {
                let right = self.expression(TokenKind::Pipe.binding_power())?;
                Ok(Node::Pipe(boxed(left), boxed(right)))
            }
            TokenKind::Or => {
                let right = self.expression(TokenKind::Or.binding_power())?;
                Ok(Node::OrExpression(boxed(left), boxed(right)))
            }
            TokenKind::And => {
                let right = self.expression(TokenKind::And.binding_power())?;
                Ok(Node::AndExpression(boxed(left), boxed(right)))
            }
            TokenKind::Lparen => self.led_lparen(left),
            TokenKind::Filter => self.led_filter(left),
            TokenKind::Eq => self.parse_comparator(left, Comparator::Eq, kind),
            TokenKind::Ne => self.parse_comparator(left, Comparator::Ne, kind),
            TokenKind::Gt => self.parse_comparator(left, Comparator::Gt, kind),
            TokenKind::Gte => self.parse_comparator(left, Comparator::Gte, kind),
            TokenKind::Lt => self.parse_comparator(left, Comparator::Lt, kind),
            TokenKind::Lte => self.parse_comparator(left, Comparator::Lte, kind),
            TokenKind::Flatten => {
                let left = Node::Flatten(boxed(left));
                let right = self.parse_projection_rhs(TokenKind::Flatten.binding_power())?;
                Ok(Node::Projection(boxed(left), boxed(right)))
            }
            TokenKind::Lbracket => {
                let token_kind = self.lookahead_token(0).kind;
                if matches!(token_kind, TokenKind::Number | TokenKind::Colon) {
                    let right = self.parse_index_expression()?;
                    match left {
                        // Grow the index expression in place.
                        Node::IndexExpression(mut children) => {
                            children.push(right);
                            Ok(Node::IndexExpression(children))
                        }
                        left => self.project_if_slice(left, right),
                    }
                } else {
                    // A projection.
                    self.match_(TokenKind::Star)?;
                    self.match_(TokenKind::Rbracket)?;
                    let right = self.parse_projection_rhs(TokenKind::Star.binding_power())?;
                    Ok(Node::Projection(boxed(left), boxed(right)))
                }
            }
            // `has_led` lets no other kind through.
            _ => {
                let t = self.lookahead_token(0).clone();
                Err(Self::raise_for_token(t, ParseMessage::InvalidToken))
            }
        }
    }

    fn led_lparen<E>(&mut self, left: Node<V>) -> R<V, E> {
        let Node::Field(name) = left else {
            // -1 is the '(' token, -2 the invalid function "name".
            let prev = self.tokens[self.index - 2].clone();
            return Err(Self::raise_for_token(
                prev,
                ParseMessage::InvalidFunctionName,
            ));
        };
        let mut args = Vec::new();
        while self.current() != TokenKind::Rparen {
            let expression = self.expression(0)?;
            if self.current() == TokenKind::Comma {
                self.match_(TokenKind::Comma)?;
            }
            args.push(expression);
        }
        self.match_(TokenKind::Rparen)?;
        Ok(Node::FunctionExpression(name, args))
    }

    fn led_filter<E>(&mut self, left: Node<V>) -> R<V, E> {
        let condition = self.expression(0)?;
        self.match_(TokenKind::Rbracket)?;
        let right = if self.current() == TokenKind::Flatten {
            Node::Identity
        } else {
            self.parse_projection_rhs(TokenKind::Filter.binding_power())?
        };
        Ok(Node::FilterProjection(
            boxed(left),
            boxed(right),
            boxed(condition),
        ))
    }

    fn parse_index_expression<E>(&mut self) -> R<V, E> {
        self.enter()?;
        let r = (|| {
            if self.lookahead(0) == TokenKind::Colon || self.lookahead(1) == TokenKind::Colon {
                self.parse_slice_expression()
            } else {
                let node = Node::Index(value(self.lookahead_token(0).clone()));
                self.advance();
                self.match_(TokenKind::Rbracket)?;
                Ok(node)
            }
        })();
        self.leave();
        r
    }

    fn parse_slice_expression<E>(&mut self) -> R<V, E> {
        self.enter()?;
        let r = (|| {
            let mut parts: [Option<V>; 3] = [None, None, None];
            let mut index = 0;
            let mut current = self.current();
            while current != TokenKind::Rbracket && index < 3 {
                if current == TokenKind::Colon {
                    index += 1;
                    if index == 3 {
                        let t = self.lookahead_token(0).clone();
                        return Err(Self::raise_for_token(t, ParseMessage::SyntaxError));
                    }
                    self.advance();
                } else if current == TokenKind::Number {
                    parts[index] = Some(value(self.lookahead_token(0).clone()));
                    self.advance();
                } else {
                    let t = self.lookahead_token(0).clone();
                    return Err(Self::raise_for_token(t, ParseMessage::SyntaxError));
                }
                current = self.current();
            }
            self.match_(TokenKind::Rbracket)?;
            let [start, end, step] = parts;
            Ok(Node::Slice(start, end, step))
        })();
        self.leave();
        r
    }

    fn project_if_slice<E>(&mut self, left: Node<V>, right: Node<V>) -> R<V, E> {
        self.enter()?;
        let r = (|| {
            let is_slice = matches!(right, Node::Slice(..));
            let index_expr = Node::IndexExpression(vec![left, right]);
            if is_slice {
                let rhs = self.parse_projection_rhs(TokenKind::Star.binding_power())?;
                Ok(Node::Projection(boxed(index_expr), boxed(rhs)))
            } else {
                Ok(index_expr)
            }
        })();
        self.leave();
        r
    }

    fn parse_comparator<E>(&mut self, left: Node<V>, cmp: Comparator, kind: TokenKind) -> R<V, E> {
        self.enter()?;
        let r = self
            .expression(kind.binding_power())
            .map(|right| Node::Comparator(cmp, boxed(left), boxed(right)));
        self.leave();
        r
    }

    fn parse_multi_select_list<E>(&mut self) -> R<V, E> {
        self.enter()?;
        let r = (|| {
            let mut expressions = Vec::new();
            loop {
                expressions.push(self.expression(0)?);
                if self.current() == TokenKind::Rbracket {
                    break;
                }
                self.match_(TokenKind::Comma)?;
            }
            self.match_(TokenKind::Rbracket)?;
            Ok(Node::MultiSelectList(expressions))
        })();
        self.leave();
        r
    }

    fn parse_multi_select_hash<E>(&mut self) -> R<V, E> {
        self.enter()?;
        let r = (|| {
            let mut pairs = Vec::new();
            loop {
                let key_token = self.lookahead_token(0).clone();
                // Check it is an identifier before taking its value.
                if !matches!(
                    self.current(),
                    TokenKind::QuotedIdentifier | TokenKind::UnquotedIdentifier
                ) {
                    let t = self.lookahead_token(0).clone();
                    return Err(Self::raise_maybe_eof(t, ParseMessage::ExpectingIdentifier));
                }
                self.advance();
                let key_name = value(key_token);
                self.match_(TokenKind::Colon)?;
                let node = self.expression(0)?;
                pairs.push(Node::KeyValPair(key_name, boxed(node)));
                if self.current() == TokenKind::Comma {
                    self.match_(TokenKind::Comma)?;
                } else if self.current() == TokenKind::Rbrace {
                    self.match_(TokenKind::Rbrace)?;
                    break;
                }
            }
            Ok(Node::MultiSelectDict(pairs))
        })();
        self.leave();
        r
    }

    fn parse_projection_rhs<E>(&mut self, binding_power: u32) -> R<V, E> {
        self.enter()?;
        let r = (|| {
            // Tokens below 10 stop a projection.
            let current = self.current();
            if current.binding_power() < 10 {
                Ok(Node::Identity)
            } else if current == TokenKind::Lbracket || current == TokenKind::Filter {
                self.expression(binding_power)
            } else if current == TokenKind::Dot {
                self.match_(TokenKind::Dot)?;
                self.parse_dot_rhs(binding_power)
            } else {
                let t = self.lookahead_token(0).clone();
                Err(Self::raise_for_token(t, ParseMessage::SyntaxError))
            }
        })();
        self.leave();
        r
    }

    fn parse_dot_rhs<E>(&mut self, binding_power: u32) -> R<V, E> {
        self.enter()?;
        let r = (|| {
            let lookahead = self.current();
            match lookahead {
                TokenKind::QuotedIdentifier | TokenKind::UnquotedIdentifier | TokenKind::Star => {
                    self.expression(binding_power)
                }
                TokenKind::Lbracket => {
                    self.match_(TokenKind::Lbracket)?;
                    self.parse_multi_select_list()
                }
                TokenKind::Lbrace => {
                    self.match_(TokenKind::Lbrace)?;
                    self.parse_multi_select_hash()
                }
                _ => {
                    let t = self.lookahead_token(0).clone();
                    Err(Self::raise_for_token(t, ParseMessage::ExpectingDotRhs))
                }
            }
        })();
        self.leave();
        r
    }

    fn match_<E>(&mut self, kind: TokenKind) -> R<V, E, ()> {
        if self.current() == kind {
            self.advance();
            Ok(())
        } else {
            let t = self.lookahead_token(0).clone();
            Err(Self::raise_maybe_eof(t, ParseMessage::Expecting(kind)))
        }
    }

    fn raise_for_token<E>(token: Token<V>, message: ParseMessage) -> ParseError<V, E> {
        ParseError::Parse {
            lex_position: token.start,
            token,
            message,
        }
    }

    fn raise_maybe_eof<E>(token: Token<V>, message: ParseMessage) -> ParseError<V, E> {
        if token.kind == TokenKind::Eof {
            return ParseError::Incomplete { token };
        }
        ParseError::Parse {
            lex_position: token.start,
            token,
            message,
        }
    }
}

/// Kinds with a `_token_led_*` method.
fn has_led(kind: TokenKind) -> bool {
    matches!(
        kind,
        TokenKind::Dot
            | TokenKind::Pipe
            | TokenKind::Or
            | TokenKind::And
            | TokenKind::Lparen
            | TokenKind::Filter
            | TokenKind::Eq
            | TokenKind::Ne
            | TokenKind::Gt
            | TokenKind::Gte
            | TokenKind::Lt
            | TokenKind::Lte
            | TokenKind::Flatten
            | TokenKind::Lbracket
    )
}

/// A value token's value (identifiers, numbers, literals). Text tokens never get here.
fn value<V>(token: Token<V>) -> V {
    match token.value {
        TokenValue::Value(v) => v,
        TokenValue::Text(_) => unreachable!("a text token used as a value"),
    }
}

/// The text of `_parse_dot_rhs`'s error, as Python formats it.
pub const DOT_RHS_ALLOWED: &str =
    "['quoted_identifier', 'unquoted_identifier', 'lbracket', 'lbrace']";

/// `['quoted_identifier', 'unquoted_identifier']`.
pub const IDENTIFIER_TYPES: &str = "['quoted_identifier', 'unquoted_identifier']";
