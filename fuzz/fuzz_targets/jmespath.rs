//! jmespath 1.1.0's lexer and parser (`pyroid_core::jmespath`) over arbitrary expressions.
//!
//! Input: `[flags] text...`. The text is the rest as UTF-8 (invalid bytes replaced), as code
//! points; with bit 2 of `flags`, every `~` becomes the lone surrogate U+D800 (Python strings
//! can hold them). Bits 0-1 pick the frame budget (none, 40, 200, 2000 frames). A mock host
//! stands in for Python: `json.loads` accepts a lexeme that starts like a JSON value.
//!
//! Checked, besides "no panic and no stack overflow at any depth":
//! - deterministic: tokenizing and parsing twice give the same result;
//! - token starts increase strictly and stay in range, the last token is `eof` at the end;
//!   a lexer error points inside the text;
//! - a parse succeeds only if tokenizing does, and fails with `Recursion` only when the budget
//!   or the hard cap is reached by nesting.

#![no_main]

use libfuzzer_sys::fuzz_target;
use pyroid_core::jmespath::{
    Budget, Host, LexError, ParseError, Text, TokenKind, parse, tokenize,
};

#[derive(Clone, Debug, PartialEq)]
enum V {
    Ident(String),
    Num(String),
    Raw(Text),
    Json(Text),
}

struct Mock;

impl Host for Mock {
    type Value = V;
    type Error = ();

    fn identifier(&mut self, text: &str) -> Result<V, ()> {
        Ok(V::Ident(text.to_owned()))
    }

    fn number(&mut self, text: &str) -> Result<V, ()> {
        Ok(V::Num(text.to_owned()))
    }

    fn raw_string(&mut self, text: &[u32]) -> Result<V, ()> {
        Ok(V::Raw(text.to_vec()))
    }

    fn json(&mut self, text: &[u32]) -> Result<Result<V, Text>, ()> {
        let ok = text.first().is_some_and(|&c| {
            matches!(char::from_u32(c), Some('0'..='9' | '-' | '[' | '{' | '"' | 't' | 'f' | 'n'))
        });
        Ok(if ok {
            Ok(V::Json(text.to_vec()))
        } else {
            Err("Expecting value: line 1 column 1 (char 0)".chars().map(u32::from).collect())
        })
    }

    fn deprecated_literal(&mut self) -> Result<(), ()> {
        Ok(())
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&flags, rest)) = data.split_first() else {
        return;
    };
    let text = String::from_utf8_lossy(rest);
    let expr: Vec<u32> = text
        .chars()
        .map(|c| if flags & 4 != 0 && c == '~' { 0xD800 } else { u32::from(c) })
        .collect();
    let frames = [usize::MAX, 40, 200, 2000][usize::from(flags & 3)];
    let budget = Budget {
        frames,
        ..Budget::UNLIMITED
    };

    let tokens = tokenize(&expr, &mut Mock);
    assert_eq!(tokens, tokenize(&expr, &mut Mock));
    match &tokens {
        Ok(t) => {
            assert!(t.windows(2).all(|w| w[0].start < w[1].start));
            let last = t.last().expect("eof");
            assert_eq!((last.kind, last.start), (TokenKind::Eof, expr.len()));
        }
        Err((before, LexError::Lexer { position, .. })) => {
            assert!(*position < expr.len());
            assert!(before.iter().all(|t| t.start <= *position));
        }
        Err((_, LexError::Empty)) => assert!(expr.is_empty()),
        Err((_, LexError::Host(()))) => unreachable!("the mock host never fails"),
    }

    let parsed = parse(&expr, &mut Mock, budget);
    assert_eq!(parsed, parse(&expr, &mut Mock, budget));
    match &parsed {
        Ok(_) => assert!(tokens.is_ok()),
        Err(ParseError::Lex(_)) => assert!(tokens.is_err()),
        Err(ParseError::Recursion) => {
            // Each nesting level needs at least one opening character.
            let openers = expr
                .iter()
                .filter(|&&c| matches!(char::from_u32(c), Some('(' | '[' | '{' | '!' | '&' | '|' | '.' | '=' | '<' | '>' | ',')))
                .count();
            assert!(openers * 4 + 4 >= frames.min(pyroid_core::jmespath::parser::MAX_FRAMES) / 2 || frames == usize::MAX);
        }
        Err(_) => assert!(tokens.is_ok()),
    }
});
