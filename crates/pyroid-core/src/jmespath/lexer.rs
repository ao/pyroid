// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! `jmespath/lexer.py`: `Lexer.tokenize`.

use super::{Host, Text};

/// A token type (`token['type']`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenKind {
    Dot,
    Star,
    Rbracket,
    Comma,
    Colon,
    Current,
    Lparen,
    Rparen,
    Lbrace,
    Rbrace,
    UnquotedIdentifier,
    QuotedIdentifier,
    Flatten,
    Filter,
    Lbracket,
    Literal,
    Or,
    Pipe,
    And,
    Expref,
    Number,
    Lte,
    Lt,
    Gte,
    Gt,
    Ne,
    Not,
    Eq,
    Eof,
}

impl TokenKind {
    /// The type's name, as jmespath spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Dot => "dot",
            Self::Star => "star",
            Self::Rbracket => "rbracket",
            Self::Comma => "comma",
            Self::Colon => "colon",
            Self::Current => "current",
            Self::Lparen => "lparen",
            Self::Rparen => "rparen",
            Self::Lbrace => "lbrace",
            Self::Rbrace => "rbrace",
            Self::UnquotedIdentifier => "unquoted_identifier",
            Self::QuotedIdentifier => "quoted_identifier",
            Self::Flatten => "flatten",
            Self::Filter => "filter",
            Self::Lbracket => "lbracket",
            Self::Literal => "literal",
            Self::Or => "or",
            Self::Pipe => "pipe",
            Self::And => "and",
            Self::Expref => "expref",
            Self::Number => "number",
            Self::Lte => "lte",
            Self::Lt => "lt",
            Self::Gte => "gte",
            Self::Gt => "gt",
            Self::Ne => "ne",
            Self::Not => "not",
            Self::Eq => "eq",
            Self::Eof => "eof",
        }
    }

    /// `Parser.BINDING_POWER[type]`.
    pub fn binding_power(self) -> u32 {
        match self {
            Self::Eof
            | Self::UnquotedIdentifier
            | Self::QuotedIdentifier
            | Self::Literal
            | Self::Rbracket
            | Self::Rparen
            | Self::Comma
            | Self::Rbrace
            | Self::Number
            | Self::Current
            | Self::Expref
            | Self::Colon => 0,
            Self::Pipe => 1,
            Self::Or => 2,
            Self::And => 3,
            Self::Eq | Self::Gt | Self::Lt | Self::Gte | Self::Lte | Self::Ne => 5,
            Self::Flatten => 9,
            Self::Star => 20,
            Self::Filter => 21,
            Self::Dot => 40,
            Self::Not => 45,
            Self::Lbrace => 50,
            Self::Lbracket => 55,
            Self::Lparen => 60,
        }
    }
}

/// `token['value']`.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenValue<V> {
    /// The token's text (`'.'`, `'[]'`, `'||'`, ..., `''` for eof).
    Text(&'static str),
    /// An identifier, number or literal, from the [`Host`].
    Value(V),
}

/// A token: `{'type': ..., 'value': ..., 'start': ..., 'end': ...}`.
#[derive(Clone, Debug, PartialEq)]
pub struct Token<V> {
    pub kind: TokenKind,
    pub value: TokenValue<V>,
    pub start: usize,
    /// jmespath's `end` (for literals and quoted identifiers it is the token's length).
    pub end: usize,
}

/// Why tokenizing stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum LexError<E> {
    /// `EmptyExpressionError`.
    Empty,
    /// `LexerError(lexer_position, lexer_value, message)`.
    Lexer {
        position: usize,
        value: Text,
        message: Text,
    },
    /// The host raised.
    Host(E),
}

const BACKSLASH: u32 = '\\' as u32;

fn is_start_identifier(c: u32) -> bool {
    matches!(c, 0x41..=0x5A | 0x61..=0x7A | 0x5F)
}

fn is_valid_identifier(c: u32) -> bool {
    is_start_identifier(c) || is_digit(c)
}

fn is_digit(c: u32) -> bool {
    (0x30..=0x39).contains(&c)
}

fn simple_token(c: u32) -> Option<(TokenKind, &'static str)> {
    Some(match char::from_u32(c)? {
        '.' => (TokenKind::Dot, "."),
        '*' => (TokenKind::Star, "*"),
        ']' => (TokenKind::Rbracket, "]"),
        ',' => (TokenKind::Comma, ","),
        ':' => (TokenKind::Colon, ":"),
        '@' => (TokenKind::Current, "@"),
        '(' => (TokenKind::Lparen, "("),
        ')' => (TokenKind::Rparen, ")"),
        '{' => (TokenKind::Lbrace, "{"),
        '}' => (TokenKind::Rbrace, "}"),
        _ => return None,
    })
}

fn text(s: &str) -> Text {
    super::code_points(s)
}

/// The tokens, or the tokens before the error and the error.
pub type LexResult<V, E> = Result<Vec<Token<V>>, (Vec<Token<V>>, LexError<E>)>;

/// The tokens of `expression` (code points), ending with eof. On an error, the tokens
/// yielded before it come back too (jmespath's `tokenize` is a generator).
pub fn tokenize<H: Host>(expression: &[u32], host: &mut H) -> LexResult<H::Value, H::Error> {
    let mut lexer = Lexer {
        chars: expression,
        position: 0,
        current: None,
    };
    let mut tokens = Vec::new();
    match lexer.run(host, &mut tokens) {
        Ok(()) => Ok(tokens),
        Err(e) => Err((tokens, e)),
    }
}

struct Lexer<'a> {
    chars: &'a [u32],
    position: usize,
    current: Option<u32>,
}

impl Lexer<'_> {
    fn length(&self) -> usize {
        self.chars.len()
    }

    /// `_next`: at the last character the position stays and `current` becomes `None`.
    fn next(&mut self) -> Option<u32> {
        if self.position + 1 == self.length() {
            self.current = None;
        } else {
            self.position += 1;
            self.current = Some(self.chars[self.position]);
        }
        self.current
    }

    // A line-by-line port of tokenize()'s loop: one branch per token kind.
    #[allow(clippy::too_many_lines)]
    fn run<H: Host>(
        &mut self,
        host: &mut H,
        tokens: &mut Vec<Token<H::Value>>,
    ) -> Result<(), LexError<H::Error>> {
        // _initialize_for_expression
        if self.chars.is_empty() {
            return Err(LexError::Empty);
        }
        self.position = 0;
        self.current = Some(self.chars[0]);
        while let Some(c) = self.current {
            if let Some((kind, value)) = simple_token(c) {
                tokens.push(Token {
                    kind,
                    value: TokenValue::Text(value),
                    start: self.position,
                    end: self.position + 1,
                });
                self.next();
            } else if is_start_identifier(c) {
                let start = self.position;
                let mut buff = String::new();
                buff.push(char::from_u32(c).unwrap_or_default());
                while let Some(d) = self.next().filter(|&d| is_valid_identifier(d)) {
                    buff.push(char::from_u32(d).unwrap_or_default());
                }
                let len = buff.len();
                tokens.push(Token {
                    kind: TokenKind::UnquotedIdentifier,
                    value: TokenValue::Value(host.identifier(&buff).map_err(LexError::Host)?),
                    start,
                    end: start + len,
                });
            } else if matches!(c, 0x20 | 0x09 | 0x0A | 0x0D) {
                self.next();
            } else if c == u32::from('[') {
                let start = self.position;
                let next = self.next();
                if next == Some(u32::from(']')) {
                    self.next();
                    tokens.push(Token {
                        kind: TokenKind::Flatten,
                        value: TokenValue::Text("[]"),
                        start,
                        end: start + 2,
                    });
                } else if next == Some(u32::from('?')) {
                    self.next();
                    tokens.push(Token {
                        kind: TokenKind::Filter,
                        value: TokenValue::Text("[?"),
                        start,
                        end: start + 2,
                    });
                } else {
                    tokens.push(Token {
                        kind: TokenKind::Lbracket,
                        value: TokenValue::Text("["),
                        start,
                        end: start + 1,
                    });
                }
            } else if c == u32::from('\'') {
                tokens.push(self.raw_string_literal(host)?);
            } else if c == u32::from('|') {
                tokens.push(self.match_or_else('|', TokenKind::Or, TokenKind::Pipe));
            } else if c == u32::from('&') {
                tokens.push(self.match_or_else('&', TokenKind::And, TokenKind::Expref));
            } else if c == u32::from('`') {
                tokens.push(self.literal(host)?);
            } else if is_digit(c) {
                let start = self.position;
                let buff = self.number();
                tokens.push(Token {
                    kind: TokenKind::Number,
                    value: TokenValue::Value(host.number(&buff).map_err(LexError::Host)?),
                    start,
                    end: start + buff.len(),
                });
            } else if c == u32::from('-') {
                let start = self.position;
                let buff = self.number();
                if buff.len() > 1 {
                    tokens.push(Token {
                        kind: TokenKind::Number,
                        value: TokenValue::Value(host.number(&buff).map_err(LexError::Host)?),
                        start,
                        end: start + buff.len(),
                    });
                } else {
                    return Err(LexError::Lexer {
                        position: start,
                        value: text(&buff),
                        message: text(&format!("Unknown token '{buff}'")),
                    });
                }
            } else if c == u32::from('"') {
                tokens.push(self.quoted_identifier(host)?);
            } else if c == u32::from('<') {
                tokens.push(self.match_or_else('=', TokenKind::Lte, TokenKind::Lt));
            } else if c == u32::from('>') {
                tokens.push(self.match_or_else('=', TokenKind::Gte, TokenKind::Gt));
            } else if c == u32::from('!') {
                tokens.push(self.match_or_else('=', TokenKind::Ne, TokenKind::Not));
            } else if c == u32::from('=') {
                if self.next() == Some(u32::from('=')) {
                    tokens.push(Token {
                        kind: TokenKind::Eq,
                        value: TokenValue::Text("=="),
                        start: self.position - 1,
                        end: self.position,
                    });
                    self.next();
                } else {
                    // At the end the position never advanced, so there is nothing to rewind.
                    let position = if self.current.is_none() {
                        self.position
                    } else {
                        self.position - 1
                    };
                    return Err(LexError::Lexer {
                        position,
                        value: text("="),
                        message: text("Unknown token '='"),
                    });
                }
            } else {
                let mut message = text("Unknown token ");
                message.push(c);
                return Err(LexError::Lexer {
                    position: self.position,
                    value: vec![c],
                    message,
                });
            }
        }
        let n = self.length();
        tokens.push(Token {
            kind: TokenKind::Eof,
            value: TokenValue::Text(""),
            start: n,
            end: n,
        });
        Ok(())
    }

    /// `_consume_number`: the current character and the digits after it.
    fn number(&mut self) -> String {
        let mut buff = String::new();
        buff.push(char::from_u32(self.current.unwrap_or(0)).unwrap_or_default());
        while let Some(d) = self.next().filter(|&d| is_digit(d)) {
            buff.push(char::from_u32(d).unwrap_or_default());
        }
        buff
    }

    /// `_consume_until`: the text up to `delimiter`, keeping `\` escapes as written.
    fn consume_until<E>(&mut self, delimiter: char) -> Result<Text, LexError<E>> {
        let delimiter = u32::from(delimiter);
        let start = self.position;
        let mut buff = Text::new();
        self.next();
        while self.current != Some(delimiter) {
            if self.current == Some(BACKSLASH) {
                buff.push(BACKSLASH);
                self.next();
            }
            let Some(c) = self.current else {
                let mut message = text("Unclosed ");
                message.push(delimiter);
                message.extend(text(" delimiter"));
                return Err(LexError::Lexer {
                    position: start,
                    value: self.chars[start..].to_vec(),
                    message,
                });
            };
            buff.push(c);
            self.next();
        }
        self.next();
        Ok(buff)
    }

    fn literal<H: Host>(&mut self, host: &mut H) -> Result<Token<H::Value>, LexError<H::Error>> {
        let start = self.position;
        let lexeme = replace_escape(&self.consume_until('`')?, '`');
        let value = if let Ok(v) = host.json(&lexeme).map_err(LexError::Host)? {
            v
        } else {
            // JEP-12's deprecated form: an invalid JSON value is read as a JSON string.
            let mut quoted = vec![u32::from('"')];
            quoted.extend(lstrip(&lexeme));
            quoted.push(u32::from('"'));
            let Ok(v) = host.json(&quoted).map_err(LexError::Host)? else {
                let mut message = text("Bad token ");
                message.extend(&lexeme);
                return Err(LexError::Lexer {
                    position: start,
                    value: self.chars[start..].to_vec(),
                    message,
                });
            };
            host.deprecated_literal().map_err(LexError::Host)?;
            v
        };
        Ok(Token {
            kind: TokenKind::Literal,
            value: TokenValue::Value(value),
            start,
            end: self.position - start,
        })
    }

    fn quoted_identifier<H: Host>(
        &mut self,
        host: &mut H,
    ) -> Result<Token<H::Value>, LexError<H::Error>> {
        let start = self.position;
        let mut lexeme = vec![u32::from('"')];
        lexeme.extend(self.consume_until('"')?);
        lexeme.push(u32::from('"'));
        let end = self.position - start;
        match host.json(&lexeme).map_err(LexError::Host)? {
            Ok(v) => Ok(Token {
                kind: TokenKind::QuotedIdentifier,
                value: TokenValue::Value(v),
                start,
                end,
            }),
            Err(message) => {
                // str(e).split(':')[0]
                let cut = message
                    .iter()
                    .position(|&c| c == u32::from(':'))
                    .unwrap_or(message.len());
                Err(LexError::Lexer {
                    position: start,
                    value: lexeme,
                    message: message[..cut].to_vec(),
                })
            }
        }
    }

    fn raw_string_literal<H: Host>(
        &mut self,
        host: &mut H,
    ) -> Result<Token<H::Value>, LexError<H::Error>> {
        let start = self.position;
        let lexeme = replace_escape(&self.consume_until('\'')?, '\'');
        let end = self.position - start;
        Ok(Token {
            kind: TokenKind::Literal,
            value: TokenValue::Value(host.raw_string(&lexeme).map_err(LexError::Host)?),
            start,
            end,
        })
    }

    fn match_or_else<V>(
        &mut self,
        expected: char,
        matched: TokenKind,
        other: TokenKind,
    ) -> Token<V> {
        let start = self.position;
        let current = self.current;
        let next = self.next();
        if next == Some(u32::from(expected)) {
            self.next();
            let value = match (current.and_then(char::from_u32), expected) {
                (Some('|'), '|') => "||",
                (Some('&'), '&') => "&&",
                (Some('<'), '=') => "<=",
                (Some('>'), '=') => ">=",
                _ => "!=",
            };
            return Token {
                kind: matched,
                value: TokenValue::Text(value),
                start,
                end: start + 1,
            };
        }
        let value = match current.and_then(char::from_u32) {
            Some('|') => "|",
            Some('&') => "&",
            Some('<') => "<",
            Some('>') => ">",
            _ => "!",
        };
        Token {
            kind: other,
            value: TokenValue::Text(value),
            start,
            end: start,
        }
    }
}

/// `s.replace('\\' + quote, quote)`.
fn replace_escape(s: &[u32], quote: char) -> Text {
    let quote = u32::from(quote);
    let mut out = Text::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        if s[i] == BACKSLASH && s.get(i + 1) == Some(&quote) {
            out.push(quote);
            i += 2;
        } else {
            out.push(s[i]);
            i += 1;
        }
    }
    out
}

/// Python's `str.isspace()` for one code point (what `str.lstrip()` strips).
pub fn is_python_space(c: u32) -> bool {
    matches!(
        c,
        0x09..=0x0D
            | 0x1C..=0x20
            | 0x85
            | 0xA0
            | 0x1680
            | 0x2000..=0x200A
            | 0x2028
            | 0x2029
            | 0x202F
            | 0x205F
            | 0x3000
    )
}

/// `s.lstrip()`.
fn lstrip(s: &[u32]) -> &[u32] {
    let start = s
        .iter()
        .position(|&c| !is_python_space(c))
        .unwrap_or(s.len());
    &s[start..]
}
