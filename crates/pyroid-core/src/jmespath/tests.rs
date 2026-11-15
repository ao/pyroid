//! Lexer and parser tests with a mock host (exact values are checked from Python).

use super::lexer::{LexError, Token, TokenKind, TokenValue, tokenize};
use super::parser::{Budget, ParseError, ParseMessage, parse};
use super::{Host, Node, Text, code_points};

/// Values as the mock host makes them.
#[derive(Clone, Debug, PartialEq)]
enum V {
    Ident(String),
    Int(String),
    Raw(String),
    Json(serde_json::Value),
}

#[derive(Default)]
struct Mock {
    warnings: usize,
}

fn string(text: &[u32]) -> String {
    text.iter()
        .map(|&c| char::from_u32(c).unwrap_or('\u{FFFD}'))
        .collect()
}

impl Host for Mock {
    type Value = V;
    type Error = String;

    fn identifier(&mut self, text: &str) -> Result<V, String> {
        Ok(V::Ident(text.to_owned()))
    }

    fn number(&mut self, text: &str) -> Result<V, String> {
        Ok(V::Int(text.to_owned()))
    }

    fn raw_string(&mut self, text: &[u32]) -> Result<V, String> {
        Ok(V::Raw(string(text)))
    }

    fn json(&mut self, text: &[u32]) -> Result<Result<V, Text>, String> {
        let s = string(text);
        Ok(serde_json::from_str(&s)
            .map(V::Json)
            .map_err(|e| code_points(&format!("Invalid: {e}"))))
    }

    fn deprecated_literal(&mut self) -> Result<(), String> {
        self.warnings += 1;
        Ok(())
    }
}

fn tokens(expr: &str) -> Vec<Token<V>> {
    tokenize(&code_points(expr), &mut Mock::default()).expect("tokenizes")
}

fn kinds(expr: &str) -> Vec<TokenKind> {
    tokens(expr).iter().map(|t| t.kind).collect()
}

fn lex_err(expr: &str) -> (usize, String, String) {
    match tokenize(&code_points(expr), &mut Mock::default()) {
        Err((
            _,
            LexError::Lexer {
                position,
                value,
                message,
            },
        )) => (position, string(&value), string(&message)),
        other => panic!("expected a LexerError for {expr:?}, got {other:?}"),
    }
}

fn ast(expr: &str) -> Node<V> {
    parse(&code_points(expr), &mut Mock::default(), Budget::UNLIMITED).expect("parses")
}

fn parse_err(expr: &str) -> ParseError<V, String> {
    parse(&code_points(expr), &mut Mock::default(), Budget::UNLIMITED).expect_err("fails")
}

fn field(s: &str) -> Node<V> {
    Node::Field(V::Ident(s.to_owned()))
}

#[test]
fn lexes_like_jmespath() {
    use TokenKind as K;
    assert_eq!(
        kinds("foo.bar[*].baz | a || b"),
        [
            K::UnquotedIdentifier,
            K::Dot,
            K::UnquotedIdentifier,
            K::Lbracket,
            K::Star,
            K::Rbracket,
            K::Dot,
            K::UnquotedIdentifier,
            K::Pipe,
            K::UnquotedIdentifier,
            K::Or,
            K::UnquotedIdentifier,
            K::Eof
        ]
    );
    assert_eq!(
        kinds("[]&&&!=!<=< >= > == [?@"),
        [
            K::Flatten,
            K::And,
            K::Expref,
            K::Ne,
            K::Not,
            K::Lte,
            K::Lt,
            K::Gte,
            K::Gt,
            K::Eq,
            K::Filter,
            K::Current,
            K::Eof
        ]
    );
    let t = tokens("foo.bar");
    assert_eq!((t[0].start, t[0].end), (0, 3));
    assert_eq!((t[1].start, t[1].end), (3, 4));
    assert_eq!((t[2].start, t[2].end), (4, 7));
    assert_eq!((t[3].start, t[3].end), (7, 7));
    // _match_or_else: `end` is start + 1 for two-character tokens, start for one.
    let t = tokens("a||b|c");
    assert_eq!((t[1].start, t[1].end), (1, 2));
    assert_eq!((t[3].start, t[3].end), (4, 4));
    let t = tokens("a == b");
    assert_eq!(
        (t[1].start, t[1].end, &t[1].value),
        (2, 3, &TokenValue::Text("=="))
    );
}

#[test]
fn numbers_and_literals() {
    let t = tokens("-24");
    assert_eq!(t[0].value, TokenValue::Value(V::Int("-24".into())));
    assert_eq!((t[0].start, t[0].end), (0, 3));
    // A closing delimiter at the very end: the position stays on it (`_next`).
    let t = tokens("`{{}`");
    assert_eq!(t[0].value, TokenValue::Value(V::Json("{{}".into())));
    assert_eq!((t[0].start, t[0].end, t[1].start), (0, 4, 5));
    let t = tokens("`2` ");
    assert_eq!(t[0].end, 3);
    let mut mock = Mock::default();
    tokenize(&code_points("`foo`"), &mut mock).expect("lexes");
    assert_eq!(mock.warnings, 1);
    let t = tokens(r"'it\'s'");
    assert_eq!(t[0].value, TokenValue::Value(V::Raw("it's".into())));
    let t = tokens(r#""a\"b""#);
    assert_eq!(t[0].value, TokenValue::Value(V::Json("a\"b".into())));
    let t = tokens(r"`a\`b`");
    assert_eq!(t[0].value, TokenValue::Value(V::Json("a`b".into())));
}

#[test]
fn lexer_errors() {
    assert!(matches!(
        tokenize(&[], &mut Mock::default()),
        Err((_, LexError::Empty))
    ));
    assert_eq!(
        lex_err("foo[0^]"),
        (5, "^".into(), "Unknown token ^".into())
    );
    assert_eq!(
        lex_err("foo-bar"),
        (3, "-".into(), "Unknown token '-'".into())
    );
    assert_eq!(
        lex_err("a = b"),
        (2, "=".into(), "Unknown token '='".into())
    );
    assert_eq!(lex_err("a ="), (2, "=".into(), "Unknown token '='".into()));
    assert_eq!(
        lex_err("foo.\"bar"),
        (4, "\"bar".into(), "Unclosed \" delimiter".into())
    );
    assert_eq!(
        lex_err("`foo\"bar`"),
        (0, "`foo\"bar`".into(), "Bad token foo\"bar".into())
    );
    assert_eq!(
        lex_err("foobar\\"),
        (6, "\\".into(), "Unknown token \\".into())
    );
    // Quoted identifiers keep the JSON error up to the first colon.
    let (pos, value, message) = lex_err("\"\\x\"");
    assert_eq!(
        (pos, value.as_str(), message.as_str()),
        (0, "\"\\x\"", "Invalid")
    );
    // The tokens before an error come back too.
    let Err((before, _)) = tokenize(&code_points("a.b ^"), &mut Mock::default()) else {
        panic!("expected an error");
    };
    assert_eq!(before.len(), 3);
}

#[test]
fn lone_surrogates_lex() {
    // A lone surrogate is an unknown token, at its code-point position.
    let expr = [u32::from('a'), u32::from('.'), 0xD800];
    let Err((
        _,
        LexError::Lexer {
            position, value, ..
        },
    )) = tokenize(&expr, &mut Mock::default())
    else {
        panic!("expected a LexerError");
    };
    assert_eq!((position, value), (2, vec![0xD800]));
}

#[test]
fn parses_like_jmespath() {
    assert_eq!(ast("foo"), field("foo"));
    assert_eq!(
        ast("foo.bar.baz"),
        Node::Subexpression(vec![field("foo"), field("bar"), field("baz")])
    );
    // Parentheses do not stop the in-place append.
    assert_eq!(
        ast("(a.b).c"),
        Node::Subexpression(vec![field("a"), field("b"), field("c")])
    );
    let Node::IndexExpression(children) = ast("a[0][1]") else {
        panic!("index expression");
    };
    assert_eq!(children.len(), 3);
    // A slice after an index expression is appended, not projected.
    let Node::IndexExpression(children) = ast("a[0][1:2]") else {
        panic!("index expression");
    };
    assert!(matches!(children[2], Node::Slice(..)));
    assert!(matches!(ast("a[1:2]"), Node::Projection(..)));
    assert!(matches!(ast("*"), Node::ValueProjection(..)));
    assert!(matches!(ast("[?a]"), Node::FilterProjection(..)));
    assert!(matches!(ast("a[?b][]"), Node::Projection(..)));
    assert!(matches!(ast("a || b && c"), Node::OrExpression(..)));
    // The grammar's quirks: no separators needed in hashes and argument lists.
    let Node::MultiSelectDict(pairs) = ast("{a: b c: d}") else {
        panic!("multi-select dict");
    };
    assert_eq!(pairs.len(), 2);
    let Node::FunctionExpression(name, args) = ast("f(a b,)") else {
        panic!("function");
    };
    assert_eq!((name, args.len()), (V::Ident("f".into()), 2));
    // A later number in a slice overwrites an earlier one.
    let Node::Projection(left, _) = ast("[1:2 3]") else {
        panic!("projection");
    };
    let Node::IndexExpression(children) = *left else {
        panic!("index expression");
    };
    assert_eq!(
        children[1],
        Node::Slice(Some(V::Int("1".into())), Some(V::Int("3".into())), None)
    );
}

#[test]
fn parse_errors() {
    let ParseError::Parse {
        lex_position,
        token,
        message,
    } = parse_err("foo]baz")
    else {
        panic!("ParseError");
    };
    assert_eq!(
        (lex_position, token.kind, message),
        (3, TokenKind::Rbracket, ParseMessage::UnexpectedToken)
    );
    assert!(matches!(
        parse_err("foo.{bar: baz,bar: bar"),
        ParseError::Incomplete { .. }
    ));
    assert!(matches!(
        parse_err("length(@,"),
        ParseError::Incomplete { .. }
    ));
    let ParseError::Parse {
        lex_position,
        message,
        ..
    } = parse_err("\"f\"(@)")
    else {
        panic!("ParseError");
    };
    assert_eq!(
        (lex_position, message),
        (0, ParseMessage::QuotedFunctionName)
    );
    let ParseError::Parse { token, message, .. } = parse_err("[0](@)") else {
        panic!("ParseError");
    };
    assert_eq!(
        (token.kind, message),
        (TokenKind::Rbracket, ParseMessage::InvalidFunctionName)
    );
    let ParseError::Parse { message, .. } = parse_err("a.1") else {
        panic!("ParseError");
    };
    assert_eq!(message, ParseMessage::ExpectingDotRhs);
    let ParseError::Parse { message, .. } = parse_err("{1: a}") else {
        panic!("ParseError");
    };
    assert_eq!(message, ParseMessage::ExpectingIdentifier);
    let ParseError::Parse { message, .. } = parse_err("a[b]") else {
        panic!("ParseError");
    };
    assert_eq!(message, ParseMessage::Expecting(TokenKind::Star));
    let ParseError::Parse { message, .. } = parse_err("[:::]") else {
        panic!("ParseError");
    };
    assert_eq!(message, ParseMessage::SyntaxError);
    let ParseError::Parse { message, .. } = parse_err("a*") else {
        panic!("ParseError");
    };
    assert_eq!(message, ParseMessage::InvalidToken);
    // Lexer errors win over parse errors (the tokens are listed first).
    assert!(matches!(
        parse_err("] ^"),
        ParseError::Lex(LexError::Lexer { .. })
    ));
}

#[test]
fn recursion_budget() {
    let deep = |n: usize| format!("{}a{}", "(".repeat(n), ")".repeat(n));
    let budget = Budget { frames: 100 };
    // Parentheses cost two frames a level (`_expression`, `_token_nud_lparen`).
    assert!(parse(&code_points(&deep(40)), &mut Mock::default(), budget).is_ok());
    assert_eq!(
        parse(&code_points(&deep(60)), &mut Mock::default(), budget),
        Err(ParseError::Recursion)
    );
    // The hard cap holds without a budget (a debug build needs a big stack to reach it).
    let deep_ok = std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(move || {
            let lists = format!("{}a{}", "[".repeat(100_000), "]".repeat(100_000));
            (
                parse(
                    &code_points(&deep(100_000)),
                    &mut Mock::default(),
                    Budget::UNLIMITED,
                ),
                parse(
                    &code_points(&lists),
                    &mut Mock::default(),
                    Budget::UNLIMITED,
                ),
                parse(
                    &code_points(&deep(990)),
                    &mut Mock::default(),
                    Budget::UNLIMITED,
                )
                .is_ok(),
            )
        })
        .expect("spawns")
        .join()
        .expect("joins");
    assert_eq!(deep_ok.0, Err(ParseError::Recursion));
    assert_eq!(deep_ok.1, Err(ParseError::Recursion));
    assert!(deep_ok.2);
}
