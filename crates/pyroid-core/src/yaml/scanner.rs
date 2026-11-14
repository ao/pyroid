// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! The scanner (`scanner.py`): characters to tokens, ported method by method.
//!
//! The scan loops that PyYAML writes as `while self.peek(length) ...: length += 1` walk the
//! UTF-8 text with a byte cursor here (and tell the reader how far they looked, for the
//! file-object case); everything else follows the Python line by line, including the order
//! in which errors are detected and their messages.

use std::collections::VecDeque;

use super::error::{Arg, ErrorKind, Mark, Msg, Result, YamlError};
use super::reader::{Reader, SURROGATE_BASE};
use super::tokens::{DirectiveValue, T, Token, TokenKind};

/// `'\0 \t\r\n\x85\u2028\u2029'`
#[inline]
const fn is_blankz(ch: char) -> bool {
    matches!(
        ch,
        '\0' | ' ' | '\t' | '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

/// `'\0\r\n\x85\u2028\u2029'`
#[inline]
const fn is_breakz(ch: char) -> bool {
    matches!(ch, '\0' | '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// `'\r\n\x85\u2028\u2029'`
#[inline]
const fn is_break(ch: char) -> bool {
    matches!(ch, '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// `'\0 \r\n\x85\u2028\u2029'` (no tab)
#[inline]
const fn is_space_breakz(ch: char) -> bool {
    matches!(
        ch,
        '\0' | ' ' | '\r' | '\n' | '\u{85}' | '\u{2028}' | '\u{2029}'
    )
}

/// `'0' <= ch <= '9' or 'A' <= ch <= 'Z' or 'a' <= ch <= 'z' or ch in '-_'`
#[inline]
const fn is_word(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '-' || ch == '_'
}

#[inline]
const fn is_hex(ch: char) -> bool {
    ch.is_ascii_hexdigit()
}

/// `n` as a signed indentation column (columns never come near `isize::MAX`).
#[inline]
fn signed(n: usize) -> isize {
    isize::try_from(n).unwrap_or(isize::MAX)
}

/// What the scanner needs from the interpreter.
#[derive(Clone, Copy, Debug)]
pub struct ScannerOptions {
    /// `sys.get_int_max_str_digits()` (0: no limit), for `int()` of a `%YAML` number.
    pub int_max_str_digits: usize,
}

impl Default for ScannerOptions {
    fn default() -> Self {
        Self {
            int_max_str_digits: 4300,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct SimpleKey {
    token_number: usize,
    required: bool,
    index: usize,
    line: usize,
    column: usize,
    mark: Mark,
}

/// PyYAML's `Scanner` over a [`Reader`].
#[derive(Debug)]
pub struct Scanner {
    pub reader: Reader,
    done: bool,
    flow_level: usize,
    tokens: VecDeque<Token>,
    tokens_taken: usize,
    indent: isize,
    indents: Vec<isize>,
    allow_simple_key: bool,
    /// PyYAML's dict keyed by flow level, in insertion order.
    possible_simple_keys: Vec<(usize, SimpleKey)>,
    /// A `\uD800`-style escape was decoded into U+10F800.. (see `reader::has_reserved`).
    pub surrogates: bool,
    options: ScannerOptions,
}

fn scanner_error(
    context: Option<Msg>,
    context_mark: Option<Mark>,
    problem: Msg,
    problem_mark: Mark,
) -> YamlError {
    YamlError::marked(
        ErrorKind::Scanner,
        context,
        context_mark,
        problem,
        Some(problem_mark),
    )
}

impl Scanner {
    pub fn new(reader: Reader, options: ScannerOptions) -> Self {
        let mut s = Self {
            reader,
            done: false,
            flow_level: 0,
            tokens: VecDeque::new(),
            tokens_taken: 0,
            indent: -1,
            indents: Vec::new(),
            allow_simple_key: true,
            possible_simple_keys: Vec::new(),
            surrogates: false,
            options,
        };
        s.fetch_stream_start();
        s
    }

    // Public methods.

    /// `check_token(*choices)`; no choices: is there any token.
    pub fn check_token(&mut self, choices: &[T]) -> Result<bool> {
        self.fill()?;
        Ok(match self.tokens.front() {
            Some(t) => choices.is_empty() || choices.contains(&t.kind.t()),
            None => false,
        })
    }

    /// `peek_token()`.
    pub fn peek_token(&mut self) -> Result<Option<&Token>> {
        self.fill()?;
        Ok(self.tokens.front())
    }

    /// `get_token()`.
    pub fn get_token(&mut self) -> Result<Option<Token>> {
        self.fill()?;
        let t = self.tokens.pop_front();
        if t.is_some() {
            self.tokens_taken += 1;
        }
        Ok(t)
    }

    #[inline]
    fn fill(&mut self) -> Result<()> {
        while self.need_more_tokens()? {
            let r = self.fetch_more_tokens();
            if self.reader.hit_limit() {
                return Err(YamlError::NeedMore);
            }
            r?;
        }
        Ok(())
    }

    // Private methods.

    fn need_more_tokens(&mut self) -> Result<bool> {
        if self.done {
            return Ok(false);
        }
        if self.tokens.is_empty() {
            return Ok(true);
        }
        self.stale_possible_simple_keys()?;
        Ok(self.next_possible_simple_key() == Some(self.tokens_taken))
    }

    fn fetch_more_tokens(&mut self) -> Result<()> {
        self.scan_to_next_token();
        self.stale_possible_simple_keys()?;
        self.unwind_indent(signed(self.reader.column));
        let ch = self.reader.peek0();
        match ch {
            '\0' => return self.fetch_stream_end(),
            '%' if self.check_directive() => return self.fetch_directive(),
            '-' if self.check_document_start() => {
                return self.fetch_document_indicator(TokenKind::DocumentStart);
            }
            '.' if self.check_document_end() => {
                return self.fetch_document_indicator(TokenKind::DocumentEnd);
            }
            _ => {}
        }
        match ch {
            '[' => self.fetch_flow_collection_start(TokenKind::FlowSequenceStart),
            '{' => self.fetch_flow_collection_start(TokenKind::FlowMappingStart),
            ']' => self.fetch_flow_collection_end(TokenKind::FlowSequenceEnd),
            '}' => self.fetch_flow_collection_end(TokenKind::FlowMappingEnd),
            ',' => self.fetch_flow_entry(),
            '-' if self.check_block_entry() => self.fetch_block_entry(),
            '?' if self.check_key() => self.fetch_key(),
            ':' if self.check_value() => self.fetch_value(),
            '*' => self.fetch_anchor(true),
            '&' => self.fetch_anchor(false),
            '!' => self.fetch_tag(),
            '|' if self.flow_level == 0 => self.fetch_block_scalar(false),
            '>' if self.flow_level == 0 => self.fetch_block_scalar(true),
            '\'' => self.fetch_flow_scalar(false),
            '"' => self.fetch_flow_scalar(true),
            _ if self.check_plain() => self.fetch_plain(),
            _ => Err(scanner_error(
                Some(Msg::text("while scanning for the next token")),
                None,
                Msg::ch("found character %r that cannot start any token", ch),
                self.reader.mark(),
            )),
        }
    }

    // Simple keys treatment.

    fn next_possible_simple_key(&self) -> Option<usize> {
        self.possible_simple_keys
            .iter()
            .map(|(_, k)| k.token_number)
            .min()
    }

    fn stale_possible_simple_keys(&mut self) -> Result<()> {
        let (line, index) = (self.reader.line, self.reader.index);
        let mut i = 0;
        while i < self.possible_simple_keys.len() {
            let key = self.possible_simple_keys[i].1;
            if key.line != line || index - key.index > 1024 {
                if key.required {
                    return Err(scanner_error(
                        Some(Msg::text("while scanning a simple key")),
                        Some(key.mark),
                        Msg::text("could not find expected ':'"),
                        self.reader.mark(),
                    ));
                }
                self.possible_simple_keys.remove(i);
            } else {
                i += 1;
            }
        }
        Ok(())
    }

    fn save_possible_simple_key(&mut self) -> Result<()> {
        let required = self.flow_level == 0 && self.indent == signed(self.reader.column);
        if self.allow_simple_key {
            self.remove_possible_simple_key()?;
            let token_number = self.tokens_taken + self.tokens.len();
            let key = SimpleKey {
                token_number,
                required,
                index: self.reader.index,
                line: self.reader.line,
                column: self.reader.column,
                mark: self.reader.mark(),
            };
            self.possible_simple_keys.push((self.flow_level, key));
        }
        Ok(())
    }

    fn remove_possible_simple_key(&mut self) -> Result<()> {
        if let Some(i) = self
            .possible_simple_keys
            .iter()
            .position(|(l, _)| *l == self.flow_level)
        {
            let key = self.possible_simple_keys[i].1;
            if key.required {
                return Err(scanner_error(
                    Some(Msg::text("while scanning a simple key")),
                    Some(key.mark),
                    Msg::text("could not find expected ':'"),
                    self.reader.mark(),
                ));
            }
            self.possible_simple_keys.remove(i);
        }
        Ok(())
    }

    // Indentation functions.

    fn unwind_indent(&mut self, column: isize) {
        if self.flow_level > 0 {
            return;
        }
        while self.indent > column {
            let mark = self.reader.mark();
            self.indent = self.indents.pop().unwrap_or(-1);
            self.push(TokenKind::BlockEnd, mark, mark);
        }
    }

    fn add_indent(&mut self, column: usize) -> bool {
        let column = signed(column);
        if self.indent < column {
            self.indents.push(self.indent);
            self.indent = column;
            return true;
        }
        false
    }

    #[inline]
    fn push(&mut self, kind: TokenKind, start_mark: Mark, end_mark: Mark) {
        self.tokens.push_back(Token {
            kind,
            start_mark,
            end_mark,
        });
    }

    // Fetchers.

    fn fetch_stream_start(&mut self) {
        let mark = self.reader.mark();
        self.push(TokenKind::StreamStart, mark, mark);
    }

    fn fetch_stream_end(&mut self) -> Result<()> {
        self.unwind_indent(-1);
        self.remove_possible_simple_key()?;
        self.allow_simple_key = false;
        self.possible_simple_keys.clear();
        let mark = self.reader.mark();
        self.push(TokenKind::StreamEnd, mark, mark);
        self.done = true;
        Ok(())
    }

    fn fetch_directive(&mut self) -> Result<()> {
        self.unwind_indent(-1);
        self.remove_possible_simple_key()?;
        self.allow_simple_key = false;
        let token = self.scan_directive()?;
        self.tokens.push_back(token);
        Ok(())
    }

    fn fetch_document_indicator(&mut self, kind: TokenKind) -> Result<()> {
        self.unwind_indent(-1);
        self.remove_possible_simple_key()?;
        self.allow_simple_key = false;
        let start_mark = self.reader.mark();
        self.reader.forward(3);
        let end_mark = self.reader.mark();
        self.push(kind, start_mark, end_mark);
        Ok(())
    }

    fn fetch_flow_collection_start(&mut self, kind: TokenKind) -> Result<()> {
        self.save_possible_simple_key()?;
        self.flow_level += 1;
        self.allow_simple_key = true;
        let start_mark = self.reader.mark();
        self.reader.forward_plain1(1);
        let end_mark = self.reader.mark();
        self.push(kind, start_mark, end_mark);
        Ok(())
    }

    fn fetch_flow_collection_end(&mut self, kind: TokenKind) -> Result<()> {
        self.remove_possible_simple_key()?;
        // PyYAML lets flow_level go negative on a stray ']' or '}'; it is only compared with 0.
        self.flow_level = self.flow_level.wrapping_sub(1);
        self.allow_simple_key = false;
        let start_mark = self.reader.mark();
        self.reader.forward_plain1(1);
        let end_mark = self.reader.mark();
        self.push(kind, start_mark, end_mark);
        Ok(())
    }

    fn fetch_flow_entry(&mut self) -> Result<()> {
        self.allow_simple_key = true;
        self.remove_possible_simple_key()?;
        let start_mark = self.reader.mark();
        self.reader.forward_plain1(1);
        let end_mark = self.reader.mark();
        self.push(TokenKind::FlowEntry, start_mark, end_mark);
        Ok(())
    }

    fn fetch_block_entry(&mut self) -> Result<()> {
        if self.flow_level == 0 {
            if !self.allow_simple_key {
                return Err(scanner_error(
                    None,
                    None,
                    Msg::text("sequence entries are not allowed here"),
                    self.reader.mark(),
                ));
            }
            if self.add_indent(self.reader.column) {
                let mark = self.reader.mark();
                self.push(TokenKind::BlockSequenceStart, mark, mark);
            }
        }
        self.allow_simple_key = true;
        self.remove_possible_simple_key()?;
        let start_mark = self.reader.mark();
        self.reader.forward_plain1(1);
        let end_mark = self.reader.mark();
        self.push(TokenKind::BlockEntry, start_mark, end_mark);
        Ok(())
    }

    fn fetch_key(&mut self) -> Result<()> {
        if self.flow_level == 0 {
            if !self.allow_simple_key {
                return Err(scanner_error(
                    None,
                    None,
                    Msg::text("mapping keys are not allowed here"),
                    self.reader.mark(),
                ));
            }
            if self.add_indent(self.reader.column) {
                let mark = self.reader.mark();
                self.push(TokenKind::BlockMappingStart, mark, mark);
            }
        }
        self.allow_simple_key = self.flow_level == 0;
        self.remove_possible_simple_key()?;
        let start_mark = self.reader.mark();
        self.reader.forward_plain1(1);
        let end_mark = self.reader.mark();
        self.push(TokenKind::Key, start_mark, end_mark);
        Ok(())
    }

    fn fetch_value(&mut self) -> Result<()> {
        if let Some(i) = self
            .possible_simple_keys
            .iter()
            .position(|(l, _)| *l == self.flow_level)
        {
            let key = self.possible_simple_keys.remove(i).1;
            let at = key.token_number - self.tokens_taken;
            self.tokens.insert(
                at,
                Token {
                    kind: TokenKind::Key,
                    start_mark: key.mark,
                    end_mark: key.mark,
                },
            );
            if self.flow_level == 0 && self.add_indent(key.column) {
                self.tokens.insert(
                    at,
                    Token {
                        kind: TokenKind::BlockMappingStart,
                        start_mark: key.mark,
                        end_mark: key.mark,
                    },
                );
            }
            self.allow_simple_key = false;
        } else {
            if self.flow_level == 0 && !self.allow_simple_key {
                return Err(scanner_error(
                    None,
                    None,
                    Msg::text("mapping values are not allowed here"),
                    self.reader.mark(),
                ));
            }
            if self.flow_level == 0 && self.add_indent(self.reader.column) {
                let mark = self.reader.mark();
                self.push(TokenKind::BlockMappingStart, mark, mark);
            }
            self.allow_simple_key = self.flow_level == 0;
            self.remove_possible_simple_key()?;
        }
        let start_mark = self.reader.mark();
        self.reader.forward_plain1(1);
        let end_mark = self.reader.mark();
        self.push(TokenKind::Value, start_mark, end_mark);
        Ok(())
    }

    fn fetch_anchor(&mut self, alias: bool) -> Result<()> {
        self.save_possible_simple_key()?;
        self.allow_simple_key = false;
        let token = self.scan_anchor(alias)?;
        self.tokens.push_back(token);
        Ok(())
    }

    fn fetch_tag(&mut self) -> Result<()> {
        self.save_possible_simple_key()?;
        self.allow_simple_key = false;
        let token = self.scan_tag()?;
        self.tokens.push_back(token);
        Ok(())
    }

    fn fetch_block_scalar(&mut self, folded: bool) -> Result<()> {
        self.allow_simple_key = true;
        self.remove_possible_simple_key()?;
        let token = self.scan_block_scalar(folded)?;
        self.tokens.push_back(token);
        Ok(())
    }

    fn fetch_flow_scalar(&mut self, double: bool) -> Result<()> {
        self.save_possible_simple_key()?;
        self.allow_simple_key = false;
        let token = self.scan_flow_scalar(double)?;
        self.tokens.push_back(token);
        Ok(())
    }

    fn fetch_plain(&mut self) -> Result<()> {
        self.save_possible_simple_key()?;
        self.allow_simple_key = false;
        let token = self.scan_plain();
        self.tokens.push_back(token);
        Ok(())
    }

    // Checkers.

    fn check_directive(&self) -> bool {
        self.reader.column == 0
    }

    fn check_document_start(&self) -> bool {
        self.reader.column == 0 && self.reader.prefix_is("---") && is_blankz(self.reader.peek(3))
    }

    fn check_document_end(&self) -> bool {
        self.reader.column == 0 && self.reader.prefix_is("...") && is_blankz(self.reader.peek(3))
    }

    fn check_block_entry(&self) -> bool {
        is_blankz(self.reader.peek(1))
    }

    fn check_key(&self) -> bool {
        self.flow_level > 0 || is_blankz(self.reader.peek(1))
    }

    fn check_value(&self) -> bool {
        self.flow_level > 0 || is_blankz(self.reader.peek(1))
    }

    fn check_plain(&self) -> bool {
        let ch = self.reader.peek0();
        !matches!(
            ch,
            '\0' | ' '
                | '\t'
                | '\r'
                | '\n'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
                | '-'
                | '?'
                | ':'
                | ','
                | '['
                | ']'
                | '{'
                | '}'
                | '#'
                | '&'
                | '*'
                | '!'
                | '|'
                | '>'
                | '\''
                | '"'
                | '%'
                | '@'
                | '`'
        ) || (!is_blankz(self.reader.peek(1))
            && (ch == '-' || (self.flow_level == 0 && matches!(ch, '?' | ':'))))
    }

    // Scanners.

    /// Walk from the cursor while `keep(ch)`: `(bytes, chars, BOMs)` of the run. The
    /// character that stops it is looked at too.
    #[inline]
    fn span(&self, keep: impl Fn(char) -> bool) -> (usize, usize, usize) {
        let text = self.reader.text().as_bytes();
        let start = self.reader.pos();
        let mut at = start;
        let mut chars = 0usize;
        let mut boms = 0usize;
        loop {
            let ch = match text.get(at) {
                None => '\0',
                Some(&b) if b < 0x80 => char::from(b),
                Some(_) => self.reader.char_at(at).0,
            };
            if !keep(ch) || ch == '\0' && at >= text.len() {
                break;
            }
            if ch == '\u{FEFF}' {
                boms += 1;
            }
            at += ch.len_utf8();
            chars += 1;
        }
        self.reader.need_ahead(chars);
        (at - start, chars, boms)
    }

    /// `while self.peek() == ' ': self.forward()`
    #[inline]
    fn skip_spaces(&mut self) {
        let text = self.reader.text().as_bytes();
        let start = self.reader.pos();
        let n = text[start..].iter().take_while(|&&b| b == b' ').count();
        self.reader.forward_span(n, n, 0);
    }

    /// `while self.peek() not in '\0\r\n\x85\u2028\u2029': self.forward()`
    #[inline]
    fn skip_to_break(&mut self) {
        let (bytes, chars, boms) = self.span(|ch| !is_breakz(ch));
        self.reader.forward_span(bytes, chars, boms);
    }

    fn scan_to_next_token(&mut self) {
        if self.reader.index == 0 && self.reader.peek0() == '\u{FEFF}' {
            self.reader.forward(1);
        }
        loop {
            self.skip_spaces();
            if self.reader.peek0() == '#' {
                self.skip_to_break();
            }
            if self.scan_line_break().is_some() {
                if self.flow_level == 0 {
                    self.allow_simple_key = true;
                }
            } else {
                break;
            }
        }
    }

    fn scan_directive(&mut self) -> Result<Token> {
        let start_mark = self.reader.mark();
        self.reader.forward(1);
        let name = self.scan_directive_name(start_mark)?;
        let value;
        let end_mark;
        if name == "YAML" {
            let (major, minor) = self.scan_yaml_directive_value(start_mark)?;
            value = DirectiveValue::Yaml(major, minor);
            end_mark = self.reader.mark();
        } else if name == "TAG" {
            let (handle, prefix) = self.scan_tag_directive_value(start_mark)?;
            value = DirectiveValue::Tag(handle, prefix);
            end_mark = self.reader.mark();
        } else {
            value = DirectiveValue::None;
            end_mark = self.reader.mark();
            self.skip_to_break();
        }
        self.scan_directive_ignored_line(start_mark)?;
        Ok(Token {
            kind: TokenKind::Directive { name, value },
            start_mark,
            end_mark,
        })
    }

    fn directive_error(&self, start_mark: Mark, problem: Msg) -> YamlError {
        scanner_error(
            Some(Msg::text("while scanning a directive")),
            Some(start_mark),
            problem,
            self.reader.mark(),
        )
    }

    fn scan_directive_name(&mut self, start_mark: Mark) -> Result<String> {
        let (bytes, chars, _) = self.span(is_word);
        if chars == 0 {
            let ch = self.reader.peek0();
            return Err(self.directive_error(
                start_mark,
                Msg::ch("expected alphabetic or numeric character, but found %r", ch),
            ));
        }
        let value = self.reader.prefix(chars).to_owned();
        self.reader.forward_span(bytes, chars, 0);
        let ch = self.reader.peek0();
        if !is_space_breakz(ch) {
            return Err(self.directive_error(
                start_mark,
                Msg::ch("expected alphabetic or numeric character, but found %r", ch),
            ));
        }
        Ok(value)
    }

    fn scan_yaml_directive_value(&mut self, start_mark: Mark) -> Result<(u64, u64)> {
        self.skip_spaces();
        let major = self.scan_yaml_directive_number(start_mark)?;
        if self.reader.peek0() != '.' {
            let ch = self.reader.peek0();
            return Err(self.directive_error(
                start_mark,
                Msg::ch("expected a digit or '.', but found %r", ch),
            ));
        }
        self.reader.forward_plain1(1);
        let minor = self.scan_yaml_directive_number(start_mark)?;
        if !is_space_breakz(self.reader.peek0()) {
            let ch = self.reader.peek0();
            return Err(self.directive_error(
                start_mark,
                Msg::ch("expected a digit or ' ', but found %r", ch),
            ));
        }
        Ok((major, minor))
    }

    fn scan_yaml_directive_number(&mut self, start_mark: Mark) -> Result<u64> {
        let ch = self.reader.peek0();
        if !ch.is_ascii_digit() {
            return Err(
                self.directive_error(start_mark, Msg::ch("expected a digit, but found %r", ch))
            );
        }
        let (bytes, chars, _) = self.span(|c| c.is_ascii_digit());
        let digits = self.reader.prefix(chars);
        let limit = self.options.int_max_str_digits;
        if limit > 0 && chars > limit {
            return Err(YamlError::Value(format!(
                "Exceeds the limit ({limit} digits) for integer string conversion: value has {chars} digits; use sys.set_int_max_str_digits() to increase the limit"
            )));
        }
        let value = digits.bytes().fold(0u64, |acc, d| {
            acc.saturating_mul(10).saturating_add(u64::from(d - b'0'))
        });
        self.reader.forward_span(bytes, chars, 0);
        Ok(value)
    }

    fn scan_tag_directive_value(&mut self, start_mark: Mark) -> Result<(String, String)> {
        self.skip_spaces();
        let handle = self.scan_tag_directive_handle(start_mark)?;
        self.skip_spaces();
        let prefix = self.scan_tag_directive_prefix(start_mark)?;
        Ok((handle, prefix))
    }

    fn scan_tag_directive_handle(&mut self, start_mark: Mark) -> Result<String> {
        let value = self.scan_tag_handle("directive", start_mark)?;
        let ch = self.reader.peek0();
        if ch != ' ' {
            return Err(self.directive_error(start_mark, Msg::ch("expected ' ', but found %r", ch)));
        }
        Ok(value)
    }

    fn scan_tag_directive_prefix(&mut self, start_mark: Mark) -> Result<String> {
        let value = self.scan_tag_uri("directive", start_mark)?;
        let ch = self.reader.peek0();
        if !is_space_breakz(ch) {
            return Err(self.directive_error(start_mark, Msg::ch("expected ' ', but found %r", ch)));
        }
        Ok(value)
    }

    fn scan_directive_ignored_line(&mut self, start_mark: Mark) -> Result<()> {
        self.skip_spaces();
        if self.reader.peek0() == '#' {
            self.skip_to_break();
        }
        let ch = self.reader.peek0();
        if !is_breakz(ch) {
            return Err(self.directive_error(
                start_mark,
                Msg::ch("expected a comment or a line break, but found %r", ch),
            ));
        }
        self.scan_line_break();
        Ok(())
    }

    fn scan_anchor(&mut self, alias: bool) -> Result<Token> {
        let start_mark = self.reader.mark();
        let context = if alias {
            "while scanning an alias"
        } else {
            "while scanning an anchor"
        };
        self.reader.forward_plain1(1);
        let (bytes, chars, _) = self.span(is_word);
        if chars == 0 {
            let ch = self.reader.peek0();
            return Err(scanner_error(
                Some(Msg::text(context)),
                Some(start_mark),
                Msg::ch("expected alphabetic or numeric character, but found %r", ch),
                self.reader.mark(),
            ));
        }
        let value = self.reader.prefix(chars).to_owned();
        self.reader.forward_span(bytes, chars, 0);
        let ch = self.reader.peek0();
        if !(is_blankz(ch) || matches!(ch, '?' | ':' | ',' | ']' | '}' | '%' | '@' | '`')) {
            return Err(scanner_error(
                Some(Msg::text(context)),
                Some(start_mark),
                Msg::ch("expected alphabetic or numeric character, but found %r", ch),
                self.reader.mark(),
            ));
        }
        let end_mark = self.reader.mark();
        let kind = if alias {
            TokenKind::Alias(value)
        } else {
            TokenKind::Anchor(value)
        };
        Ok(Token {
            kind,
            start_mark,
            end_mark,
        })
    }

    fn scan_tag(&mut self) -> Result<Token> {
        let start_mark = self.reader.mark();
        let mut ch = self.reader.peek(1);
        let handle;
        let suffix;
        if ch == '<' {
            handle = None;
            self.reader.forward(2);
            suffix = self.scan_tag_uri("tag", start_mark)?;
            if self.reader.peek0() != '>' {
                let c = self.reader.peek0();
                return Err(scanner_error(
                    Some(Msg::text("while parsing a tag")),
                    Some(start_mark),
                    Msg::ch("expected '>', but found %r", c),
                    self.reader.mark(),
                ));
            }
            self.reader.forward(1);
        } else if is_blankz(ch) {
            handle = None;
            suffix = "!".to_owned();
            self.reader.forward(1);
        } else {
            let mut length = 1;
            let mut use_handle = false;
            while !is_space_breakz(ch) {
                if ch == '!' {
                    use_handle = true;
                    break;
                }
                length += 1;
                ch = self.reader.peek(length);
            }
            if use_handle {
                handle = Some(self.scan_tag_handle("tag", start_mark)?);
            } else {
                handle = Some("!".to_owned());
                self.reader.forward(1);
            }
            suffix = self.scan_tag_uri("tag", start_mark)?;
        }
        let ch = self.reader.peek0();
        if !is_space_breakz(ch) {
            return Err(scanner_error(
                Some(Msg::text("while scanning a tag")),
                Some(start_mark),
                Msg::ch("expected ' ', but found %r", ch),
                self.reader.mark(),
            ));
        }
        let end_mark = self.reader.mark();
        Ok(Token {
            kind: TokenKind::Tag(handle, suffix),
            start_mark,
            end_mark,
        })
    }

    fn scan_block_scalar(&mut self, folded: bool) -> Result<Token> {
        let mut chunks = String::new();
        let start_mark = self.reader.mark();
        self.reader.forward(1);
        let (chomping, increment) = self.scan_block_scalar_indicators(start_mark)?;
        self.scan_block_scalar_ignored_line(start_mark)?;

        let mut min_indent = self.indent + 1;
        if min_indent < 1 {
            min_indent = 1;
        }
        let min_indent = usize::try_from(min_indent).unwrap_or(1);
        let mut breaks;
        let mut end_mark;
        let indent;
        if let Some(increment) = increment {
            indent = min_indent + increment - 1;
            (breaks, end_mark) = self.scan_block_scalar_breaks(indent);
        } else {
            let max_indent;
            (breaks, max_indent, end_mark) = self.scan_block_scalar_indentation();
            indent = min_indent.max(max_indent);
        }
        let mut line_break: &'static str = "";

        while self.reader.column == indent && self.reader.peek0() != '\0' {
            chunks.push_str(&breaks);
            let leading_non_space = !matches!(self.reader.peek0(), ' ' | '\t');
            let (bytes, chars, boms) = self.span(|c| !is_breakz(c));
            chunks.push_str(&self.reader.text()[self.reader.pos()..self.reader.pos() + bytes]);
            self.reader.forward_span(bytes, chars, boms);
            line_break = self.scan_line_break().unwrap_or("");
            (breaks, end_mark) = self.scan_block_scalar_breaks(indent);
            if self.reader.column == indent && self.reader.peek0() != '\0' {
                if folded
                    && line_break == "\n"
                    && leading_non_space
                    && !matches!(self.reader.peek0(), ' ' | '\t')
                {
                    if breaks.is_empty() {
                        chunks.push(' ');
                    }
                } else {
                    chunks.push_str(line_break);
                }
            } else {
                break;
            }
        }

        if chomping != Some(false) {
            chunks.push_str(line_break);
        }
        if chomping == Some(true) {
            chunks.push_str(&breaks);
        }
        let style = if folded { '>' } else { '|' };
        Ok(Token {
            kind: TokenKind::Scalar {
                value: chunks,
                plain: false,
                style: Some(style),
            },
            start_mark,
            end_mark,
        })
    }

    fn block_scalar_error(&self, start_mark: Mark, problem: Msg) -> YamlError {
        scanner_error(
            Some(Msg::text("while scanning a block scalar")),
            Some(start_mark),
            problem,
            self.reader.mark(),
        )
    }

    fn scan_block_scalar_indicators(
        &mut self,
        start_mark: Mark,
    ) -> Result<(Option<bool>, Option<usize>)> {
        let mut chomping = None;
        let mut increment = None;
        let mut ch = self.reader.peek0();
        if ch == '+' || ch == '-' {
            chomping = Some(ch == '+');
            self.reader.forward(1);
            ch = self.reader.peek0();
            if ch.is_ascii_digit() {
                let n = ch as usize - '0' as usize;
                if n == 0 {
                    return Err(self.block_scalar_error(
                        start_mark,
                        Msg::text("expected indentation indicator in the range 1-9, but found 0"),
                    ));
                }
                increment = Some(n);
                self.reader.forward(1);
            }
        } else if ch.is_ascii_digit() {
            let n = ch as usize - '0' as usize;
            if n == 0 {
                return Err(self.block_scalar_error(
                    start_mark,
                    Msg::text("expected indentation indicator in the range 1-9, but found 0"),
                ));
            }
            increment = Some(n);
            self.reader.forward(1);
            ch = self.reader.peek0();
            if ch == '+' || ch == '-' {
                chomping = Some(ch == '+');
                self.reader.forward(1);
            }
        }
        let ch = self.reader.peek0();
        if !is_space_breakz(ch) {
            return Err(self.block_scalar_error(
                start_mark,
                Msg::ch(
                    "expected chomping or indentation indicators, but found %r",
                    ch,
                ),
            ));
        }
        Ok((chomping, increment))
    }

    fn scan_block_scalar_ignored_line(&mut self, start_mark: Mark) -> Result<()> {
        self.skip_spaces();
        if self.reader.peek0() == '#' {
            self.skip_to_break();
        }
        let ch = self.reader.peek0();
        if !is_breakz(ch) {
            return Err(self.block_scalar_error(
                start_mark,
                Msg::ch("expected a comment or a line break, but found %r", ch),
            ));
        }
        self.scan_line_break();
        Ok(())
    }

    fn scan_block_scalar_indentation(&mut self) -> (String, usize, Mark) {
        let mut chunks = String::new();
        let mut max_indent = 0;
        let mut end_mark = self.reader.mark();
        loop {
            let ch = self.reader.peek0();
            if ch == ' ' {
                self.reader.forward_plain1(1);
                if self.reader.column > max_indent {
                    max_indent = self.reader.column;
                }
            } else if is_break(ch) {
                chunks.push_str(self.scan_line_break().unwrap_or(""));
                end_mark = self.reader.mark();
            } else {
                break;
            }
        }
        (chunks, max_indent, end_mark)
    }

    fn scan_block_scalar_breaks(&mut self, indent: usize) -> (String, Mark) {
        let mut chunks = String::new();
        let mut end_mark = self.reader.mark();
        while self.reader.column < indent && self.reader.peek0() == ' ' {
            self.reader.forward_plain1(1);
        }
        while is_break(self.reader.peek0()) {
            chunks.push_str(self.scan_line_break().unwrap_or(""));
            end_mark = self.reader.mark();
            while self.reader.column < indent && self.reader.peek0() == ' ' {
                self.reader.forward_plain1(1);
            }
        }
        (chunks, end_mark)
    }

    fn scan_flow_scalar(&mut self, double: bool) -> Result<Token> {
        let mut chunks = String::new();
        let start_mark = self.reader.mark();
        let quote = self.reader.peek0();
        self.reader.forward_plain1(1);
        self.scan_flow_scalar_non_spaces(double, start_mark, &mut chunks)?;
        while self.reader.peek0() != quote {
            self.scan_flow_scalar_spaces(double, start_mark, &mut chunks)?;
            self.scan_flow_scalar_non_spaces(double, start_mark, &mut chunks)?;
        }
        self.reader.forward_plain1(1);
        let end_mark = self.reader.mark();
        let style = if double { '"' } else { '\'' };
        Ok(Token {
            kind: TokenKind::Scalar {
                value: chunks,
                plain: false,
                style: Some(style),
            },
            start_mark,
            end_mark,
        })
    }

    fn scan_flow_scalar_non_spaces(
        &mut self,
        double: bool,
        start_mark: Mark,
        chunks: &mut String,
    ) -> Result<()> {
        loop {
            let (bytes, chars, boms) =
                self.span(|c| !matches!(c, '\'' | '"' | '\\') && !is_blankz(c));
            if chars > 0 {
                chunks.push_str(&self.reader.text()[self.reader.pos()..self.reader.pos() + bytes]);
                self.reader.forward_span(bytes, chars, boms);
            }
            let ch = self.reader.peek0();
            if !double && ch == '\'' && self.reader.peek(1) == '\'' {
                chunks.push('\'');
                self.reader.forward(2);
            } else if (double && ch == '\'') || (!double && (ch == '"' || ch == '\\')) {
                chunks.push(ch);
                self.reader.forward(1);
            } else if double && ch == '\\' {
                self.reader.forward(1);
                let ch = self.reader.peek0();
                if let Some(rep) = escape_replacement(ch) {
                    chunks.push(rep);
                    self.reader.forward(1);
                } else if let Some(length) = escape_code_length(ch) {
                    self.reader.forward(1);
                    for k in 0..length {
                        if !is_hex(self.reader.peek(k)) {
                            let c = self.reader.peek(k);
                            return Err(scanner_error(
                                Some(Msg::text("while scanning a double-quoted scalar")),
                                Some(start_mark),
                                Msg {
                                    fmt: "expected escape sequence of %d hexadecimal numbers, but found %r",
                                    args: vec![
                                        Arg::Int(i64::try_from(length).unwrap_or(i64::MAX)),
                                        Arg::Str(c.to_string()),
                                    ],
                                },
                                self.reader.mark(),
                            ));
                        }
                    }
                    let hex = self.reader.prefix(length);
                    let code = u32::from_str_radix(hex, 16).unwrap_or(u32::MAX);
                    if code > 0x7FFF_FFFF {
                        return Err(YamlError::Overflow(
                            "Python int too large to convert to C int".to_owned(),
                        ));
                    }
                    if code > 0x10_FFFF {
                        return Err(YamlError::Value(
                            "chr() arg not in range(0x110000)".to_owned(),
                        ));
                    }
                    if let Some(c) = char::from_u32(code) {
                        chunks.push(c);
                    } else {
                        // A lone surrogate: carried in U+10F800.. (the bindings map it back).
                        self.surrogates = true;
                        let mapped = char::from_u32(SURROGATE_BASE + (code - 0xD800));
                        chunks.push(mapped.unwrap_or('\u{FFFD}'));
                    }
                    self.reader.forward(length);
                } else if is_break(ch) {
                    self.scan_line_break();
                    self.scan_flow_scalar_breaks(start_mark, chunks)?;
                } else {
                    return Err(scanner_error(
                        Some(Msg::text("while scanning a double-quoted scalar")),
                        Some(start_mark),
                        Msg::ch("found unknown escape character %r", ch),
                        self.reader.mark(),
                    ));
                }
            } else {
                return Ok(());
            }
        }
    }

    fn scan_flow_scalar_spaces(
        &mut self,
        _double: bool,
        start_mark: Mark,
        chunks: &mut String,
    ) -> Result<()> {
        let (bytes, chars, _) = self.span(|c| c == ' ' || c == '\t');
        let ws_start = self.reader.pos();
        self.reader.forward_span(bytes, chars, 0);
        let ch = self.reader.peek0();
        if ch == '\0' {
            return Err(scanner_error(
                Some(Msg::text("while scanning a quoted scalar")),
                Some(start_mark),
                Msg::text("found unexpected end of stream"),
                self.reader.mark(),
            ));
        } else if is_break(ch) {
            let line_break = self.scan_line_break().unwrap_or("");
            let mut breaks = String::new();
            self.scan_flow_scalar_breaks(start_mark, &mut breaks)?;
            if line_break != "\n" {
                chunks.push_str(line_break);
            } else if breaks.is_empty() {
                chunks.push(' ');
            }
            chunks.push_str(&breaks);
        } else {
            chunks.push_str(&self.reader.text()[ws_start..ws_start + bytes]);
        }
        Ok(())
    }

    fn scan_flow_scalar_breaks(&mut self, start_mark: Mark, chunks: &mut String) -> Result<()> {
        loop {
            if (self.reader.prefix_is("---") || self.reader.prefix_is("..."))
                && is_blankz(self.reader.peek(3))
            {
                return Err(scanner_error(
                    Some(Msg::text("while scanning a quoted scalar")),
                    Some(start_mark),
                    Msg::text("found unexpected document separator"),
                    self.reader.mark(),
                ));
            }
            // PyYAML's prefix(3) asks for 3 characters even when they are not '---'.
            self.reader.need_ahead(2);
            let (bytes, chars, _) = self.span(|c| c == ' ' || c == '\t');
            self.reader.forward_span(bytes, chars, 0);
            if is_break(self.reader.peek0()) {
                chunks.push_str(self.scan_line_break().unwrap_or(""));
            } else {
                return Ok(());
            }
        }
    }

    fn scan_plain(&mut self) -> Token {
        let mut chunks = String::new();
        let start_mark = self.reader.mark();
        let mut end_mark = start_mark;
        let indent = self.indent + 1;
        let mut spaces: Option<String> = Some(String::new());
        let flow = self.flow_level > 0;
        loop {
            if self.reader.peek0() == '#' {
                break;
            }
            // while True: ch = self.peek(length); if <ends>: break; length += 1
            let text = self.reader.text();
            let bytes_all = text.as_bytes();
            let start = self.reader.pos();
            let mut at = start;
            let mut length = 0usize;
            let mut boms = 0usize;
            let mut looked;
            loop {
                let (ch, len) = self.reader.char_at(at);
                looked = length;
                if is_blankz(ch) {
                    break;
                }
                if ch == ':' {
                    let next = match bytes_all.get(at + 1) {
                        None => '\0',
                        Some(&b) if b < 0x80 => char::from(b),
                        Some(_) => self.reader.char_at(at + 1).0,
                    };
                    looked = length + 1;
                    if is_blankz(next) || (flow && matches!(next, ',' | '[' | ']' | '{' | '}')) {
                        break;
                    }
                    looked = length;
                }
                if flow && matches!(ch, ',' | '?' | '[' | ']' | '{' | '}') {
                    break;
                }
                if ch == '\u{FEFF}' {
                    boms += 1;
                }
                at += len;
                length += 1;
            }
            self.reader.need_ahead(looked);
            if length == 0 {
                break;
            }
            self.allow_simple_key = false;
            if let Some(s) = &spaces {
                chunks.push_str(s);
            }
            chunks.push_str(&text[start..at]);
            self.reader.forward_span(at - start, length, boms);
            end_mark = self.reader.mark();
            spaces = self.scan_plain_spaces();
            let stop = match &spaces {
                None => true,
                Some(s) => s.is_empty(),
            };
            if stop
                || self.reader.peek0() == '#'
                || (!flow && (signed(self.reader.column)) < indent)
            {
                break;
            }
        }
        Token {
            kind: TokenKind::Scalar {
                value: chunks,
                plain: true,
                style: None,
            },
            start_mark,
            end_mark,
        }
    }

    /// `None`: PyYAML's bare `return` (a document separator follows the line break).
    fn scan_plain_spaces(&mut self) -> Option<String> {
        let mut chunks = String::new();
        let ws_start = self.reader.pos();
        let n = self.reader.text().as_bytes()[ws_start..]
            .iter()
            .take_while(|&&b| b == b' ')
            .count();
        self.reader.need_ahead(n);
        self.reader.forward_span(n, n, 0);
        let ch = self.reader.peek0();
        if is_break(ch) {
            let line_break = self.scan_line_break().unwrap_or("");
            self.allow_simple_key = true;
            if self.at_document_separator() {
                return None;
            }
            let mut breaks = String::new();
            loop {
                let c = self.reader.peek0();
                if c == ' ' {
                    self.reader.forward_plain1(1);
                } else if is_break(c) {
                    breaks.push_str(self.scan_line_break().unwrap_or(""));
                    if self.at_document_separator() {
                        return None;
                    }
                } else {
                    break;
                }
            }
            if line_break != "\n" {
                chunks.push_str(line_break);
            } else if breaks.is_empty() {
                chunks.push(' ');
            }
            chunks.push_str(&breaks);
        } else if n > 0 {
            chunks.push_str(&self.reader.text()[ws_start..ws_start + n]);
        }
        Some(chunks)
    }

    /// `prefix = self.prefix(3); (prefix == '---' or prefix == '...') and self.peek(3) in ...`
    fn at_document_separator(&self) -> bool {
        self.reader.need_ahead(2);
        (self.reader.prefix_is("---") || self.reader.prefix_is("..."))
            && is_blankz(self.reader.peek(3))
    }

    fn scan_tag_handle(&mut self, name: &'static str, start_mark: Mark) -> Result<String> {
        let context = if name == "directive" {
            "while scanning a directive"
        } else {
            "while scanning a tag"
        };
        let ch = self.reader.peek0();
        if ch != '!' {
            return Err(scanner_error(
                Some(Msg::text(context)),
                Some(start_mark),
                Msg::ch("expected '!', but found %r", ch),
                self.reader.mark(),
            ));
        }
        let mut length = 1;
        let mut ch = self.reader.peek(length);
        if ch != ' ' {
            while is_word(ch) {
                length += 1;
                ch = self.reader.peek(length);
            }
            if ch != '!' {
                self.reader.forward(length);
                return Err(scanner_error(
                    Some(Msg::text(context)),
                    Some(start_mark),
                    Msg::ch("expected '!', but found %r", ch),
                    self.reader.mark(),
                ));
            }
            length += 1;
        }
        let value = self.reader.prefix(length).to_owned();
        self.reader.forward(length);
        Ok(value)
    }

    fn scan_tag_uri(&mut self, name: &'static str, start_mark: Mark) -> Result<String> {
        let mut chunks = String::new();
        let mut any = false;
        let mut length = 0;
        let mut ch = self.reader.peek(length);
        while ch.is_ascii_alphanumeric() || "-;/?:@&=+$,_.!~*'()[]%".contains(ch) {
            if ch == '%' {
                chunks.push_str(self.reader.prefix(length));
                self.reader.forward(length);
                length = 0;
                any = true;
                let s = self.scan_uri_escapes(name, start_mark)?;
                chunks.push_str(&s);
            } else {
                length += 1;
            }
            ch = self.reader.peek(length);
        }
        if length > 0 {
            chunks.push_str(self.reader.prefix(length));
            self.reader.forward(length);
            any = true;
        }
        if !any {
            let context = if name == "directive" {
                "while parsing a directive"
            } else {
                "while parsing a tag"
            };
            return Err(scanner_error(
                Some(Msg::text(context)),
                Some(start_mark),
                Msg::ch("expected URI, but found %r", ch),
                self.reader.mark(),
            ));
        }
        Ok(chunks)
    }

    fn scan_uri_escapes(&mut self, name: &'static str, start_mark: Mark) -> Result<String> {
        let context = if name == "directive" {
            "while scanning a directive"
        } else {
            "while scanning a tag"
        };
        let mut codes = Vec::new();
        let mark = self.reader.mark();
        while self.reader.peek0() == '%' {
            self.reader.forward(1);
            for k in 0..2 {
                if !is_hex(self.reader.peek(k)) {
                    let c = self.reader.peek(k);
                    return Err(scanner_error(
                        Some(Msg::text(context)),
                        Some(start_mark),
                        Msg::ch(
                            "expected URI escape sequence of 2 hexadecimal numbers, but found %r",
                            c,
                        ),
                        self.reader.mark(),
                    ));
                }
            }
            codes.push(u8::from_str_radix(self.reader.prefix(2), 16).unwrap_or(0));
            self.reader.forward(2);
        }
        match String::from_utf8(codes) {
            Ok(s) => Ok(s),
            Err(e) => Err(scanner_error(
                Some(Msg::text(context)),
                Some(start_mark),
                Msg {
                    fmt: "%s",
                    args: vec![Arg::Utf8DecodeError(e.into_bytes())],
                },
                mark,
            )),
        }
    }

    /// `scan_line_break()`: `None` for PyYAML's `''`.
    #[inline]
    fn scan_line_break(&mut self) -> Option<&'static str> {
        let ch = self.reader.peek0();
        match ch {
            '\r' | '\n' | '\u{85}' => {
                if self.reader.prefix_is("\r\n") {
                    self.reader.forward(2);
                } else {
                    self.reader.forward(1);
                }
                Some("\n")
            }
            '\u{2028}' => {
                self.reader.forward(1);
                Some("\u{2028}")
            }
            '\u{2029}' => {
                self.reader.forward(1);
                Some("\u{2029}")
            }
            _ => None,
        }
    }
}

fn escape_replacement(ch: char) -> Option<char> {
    Some(match ch {
        '0' => '\0',
        'a' => '\x07',
        'b' => '\x08',
        't' | '\t' => '\x09',
        'n' => '\x0A',
        'v' => '\x0B',
        'f' => '\x0C',
        'r' => '\x0D',
        'e' => '\x1B',
        ' ' => '\x20',
        '"' => '"',
        '\\' => '\\',
        '/' => '/',
        'N' => '\u{85}',
        '_' => '\u{A0}',
        'L' => '\u{2028}',
        'P' => '\u{2029}',
        _ => return None,
    })
}

fn escape_code_length(ch: char) -> Option<usize> {
    match ch {
        'x' => Some(2),
        'u' => Some(4),
        'U' => Some(8),
        _ => None,
    }
}
