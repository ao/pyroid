// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! json_repair's `JSONParser` (`json_parser.py`), ported line by line.
//!
//! The parser walks a char array with json_repair's index arithmetic, including
//! Python's negative-index wrap-around (`json_str[-1]` is the last char). Each
//! `parse_*` method mirrors the Python function of the same name; comments
//! name the places where the port differs in *how* (not *what*) it computes.

// json_repair's index arithmetic is signed (offsets can be negative, and
// Python's negative indices wrap); every cast between `isize` and `usize` here
// is of a value already clamped or checked to be in range.
#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]

mod array;
mod comment;
mod number;
mod object;
mod paren;
mod scan;
mod string;

use std::borrow::Cow;

use super::chars::CharProps;
use super::pyjson;
use super::value::{IntError, Value};

/// A `self.log(...)` entry: `{"text": ..., "context": ...}`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogEntry {
    pub text: String,
    pub context: String,
}

/// json_repair's `ContextValues`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Ctx {
    ObjectKey,
    ObjectValue,
    Array,
}

/// Why a parse stopped with an exception.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ParseError {
    /// `ValueError(message)` raised by strict mode.
    Strict(&'static str),
    /// Nesting beyond the supported depth.
    Depth,
    /// `ValueError` from `int()` inside `json.JSONDecoder.raw_decode` (json_repair does
    /// not catch it there).
    Int(IntError),
    /// json_repair's `assert isinstance(raw_key, str)` failed (`AssertionError`).
    Assertion,
    /// The parse needs more stack than the calling thread offers: rerun it on a
    /// thread with a big stack.
    NeedStack,
    /// `raw_decode` met an escaped surrogate pair for a char the kernel
    /// reserves (U+10F800..): the caller delegates the input.
    Unsupported,
}

pub(crate) type R<T> = Result<T, ParseError>;

/// The delimiters json_repair treats as string quotes (`STRING_DELIMITERS`).
#[inline]
pub(crate) fn is_delim(c: char) -> bool {
    matches!(c, '"' | '\'' | '\u{201c}' | '\u{201d}')
}

/// `c in STRING_DELIMITERS` for an optional char (`None` is never in it).
#[inline]
pub(crate) fn opt_delim(c: Option<char>) -> bool {
    c.is_some_and(is_delim)
}

/// Options that change parsing (`repair_json`'s keyword arguments).
#[derive(Clone, Copy, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct ParseOptions {
    pub stream_stable: bool,
    pub strict: bool,
    /// `json_fd is None`: plain strings try `raw_decode` on a valid suffix.
    pub try_valid_json_suffix: bool,
    pub logging: bool,
    /// `sys.get_int_max_str_digits()` (0 = unlimited).
    pub max_int_digits: usize,
    pub props: CharProps,
}

/// Maximum nesting of containers/comments/fences the parser follows.
pub const MAX_DEPTH: usize = 10_000;

/// Stack the parser may use on the calling thread before it asks for a rerun
/// on a dedicated thread (see [`ParseError::NeedStack`]).
const INLINE_STACK_BUDGET: usize = 256 * 1024;

#[allow(clippy::struct_excessive_bools)]
pub(crate) struct Parser<'a> {
    /// `self.json_str`, as chars. Copied only when one of the two places
    /// json_repair rewrites its input runs (duplicate-key split, escaped-object
    /// normalization).
    s: Cow<'a, [char]>,
    /// `self.index`. Signed: json_repair's index arithmetic can dip below 0.
    index: isize,
    ctx: Vec<Ctx>,
    deferred: Vec<Ctx>,
    pub(crate) logger: Option<Vec<LogEntry>>,
    stream_stable: bool,
    strict: bool,
    try_valid_json_suffix: bool,
    has_tried_valid_json_suffix: bool,
    max_int_digits: usize,
    props: CharProps,
    depth: usize,
    /// Address of a local in the entry frame; `None` = no stack budget (big-stack thread).
    stack_base: Option<usize>,
    scan: scan::Memo,
}

impl<'a> Parser<'a> {
    pub(crate) fn new(s: &'a [char], opts: &ParseOptions, stack_base: Option<usize>) -> Self {
        Self {
            s: Cow::Borrowed(s),
            index: 0,
            ctx: Vec::new(),
            deferred: Vec::new(),
            logger: opts.logging.then(Vec::new),
            stream_stable: opts.stream_stable,
            strict: opts.strict,
            try_valid_json_suffix: opts.try_valid_json_suffix,
            has_tried_valid_json_suffix: false,
            max_int_digits: opts.max_int_digits,
            props: opts.props,
            depth: 0,
            stack_base,
            scan: scan::Memo::default(),
        }
    }

    // ------------------------------------------------------------------
    // Buffer access (`self.json_str[...]` with Python semantics)
    // ------------------------------------------------------------------

    #[inline]
    fn len(&self) -> isize {
        self.s.len() as isize
    }

    /// `self.json_str[i]`: `None` for IndexError; negative `i` wraps like Python.
    #[inline]
    fn char_at(&self, i: isize) -> Option<char> {
        if i >= 0 {
            self.s.get(i as usize).copied()
        } else {
            let j = self.len() + i;
            if j < 0 {
                None
            } else {
                Some(self.s[j as usize])
            }
        }
    }

    /// `self.get_char_at(count)`.
    #[inline]
    fn get_char_at(&self, count: isize) -> Option<char> {
        self.char_at(self.index + count)
    }

    #[inline]
    fn cur(&self) -> Option<char> {
        self.get_char_at(0)
    }

    /// `self.json_str[a:b]` (Python slice clamping, non-negative bounds).
    fn slice(&self, a: isize, b: isize) -> &[char] {
        let n = self.len();
        let clamp = |x: isize| -> usize {
            if x < 0 {
                (n + x).max(0) as usize
            } else {
                x.min(n) as usize
            }
        };
        let (a, b) = (clamp(a), clamp(b));
        if a >= b { &[] } else { &self.s[a..b] }
    }

    #[inline]
    fn is_space(&self, c: char) -> bool {
        self.props.is_space(c)
    }

    #[inline]
    fn opt_space(&self, c: Option<char>) -> bool {
        c.is_some_and(|c| self.props.is_space(c))
    }

    /// Replace `self.s[a..b]` with `with` (the only buffer mutations).
    fn splice(&mut self, a: usize, b: usize, with: &[char]) {
        self.s.to_mut().splice(a..b, with.iter().copied());
        self.scan.invalidate();
    }

    // ------------------------------------------------------------------
    // Context (`JsonContext`)
    // ------------------------------------------------------------------

    #[inline]
    fn ctx_current(&self) -> Option<Ctx> {
        self.ctx.last().copied()
    }

    #[inline]
    fn ctx_empty(&self) -> bool {
        self.ctx.is_empty()
    }

    #[inline]
    fn ctx_has(&self, c: Ctx) -> bool {
        self.ctx.contains(&c)
    }

    #[inline]
    fn ctx_set(&mut self, c: Ctx) {
        self.ctx.push(c);
    }

    #[inline]
    fn ctx_reset(&mut self) {
        self.ctx.pop();
    }

    // ------------------------------------------------------------------
    // Logging
    // ------------------------------------------------------------------

    /// `self.log(text)`: no-op unless logging; `text` is only built when logging.
    #[inline]
    fn log(&mut self, text: impl FnOnce() -> String) {
        if self.logger.is_some() {
            self.log_now(text());
        }
    }

    #[cold]
    fn log_now(&mut self, text: String) {
        let window: isize = 10;
        let start = (self.index - window).max(0);
        let end = (self.index + window).min(self.len());
        let context: String = self.slice(start, end).iter().collect();
        if let Some(l) = self.logger.as_mut() {
            l.push(LogEntry { text, context });
        }
    }

    // ------------------------------------------------------------------
    // Recursion guard
    // ------------------------------------------------------------------

    /// Enter a recursive step: enforce [`MAX_DEPTH`] and the inline stack budget.
    #[inline]
    fn enter(&mut self) -> R<()> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(ParseError::Depth);
        }
        if let Some(base) = self.stack_base {
            let marker = 0u8;
            let here = std::ptr::addr_of!(marker).addr();
            if base.saturating_sub(here) > INLINE_STACK_BUDGET {
                return Err(ParseError::NeedStack);
            }
        }
        Ok(())
    }

    #[inline]
    fn leave(&mut self) {
        self.depth -= 1;
    }

    // ------------------------------------------------------------------
    // Whitespace and character skipping
    // ------------------------------------------------------------------

    /// `skip_whitespaces`.
    fn skip_whitespaces(&mut self) {
        while let Some(c) = self.cur() {
            if !self.is_space(c) {
                break;
            }
            self.index += 1;
        }
    }

    /// `skip_whitespace_and_comments`.
    fn skip_whitespace_and_comments(&mut self) {
        loop {
            self.skip_whitespaces();
            let c = self.cur();
            if c != Some('#') && !(c == Some('/') && matches!(self.get_char_at(1), Some('/' | '*')))
            {
                return;
            }
            self.skip_comment();
        }
    }

    /// `scroll_whitespaces(idx)`: the offset of the first non-space at or after `idx`.
    fn scroll_whitespaces(&mut self, idx: isize) -> isize {
        let start = self.index + idx;
        if start >= 0 {
            let end = self.next_non_space(start as usize);
            return end as isize - self.index;
        }
        self.scroll_whitespaces_slow(idx, false)
    }

    /// `scroll_whitespaces(idx, backwards=True)`.
    fn scroll_whitespaces_back(&self, idx: isize) -> isize {
        self.scroll_whitespaces_slow(idx, true)
    }

    fn scroll_whitespaces_slow(&self, mut idx: isize, backwards: bool) -> isize {
        let step = if backwards { -1 } else { 1 };
        let Some(mut c) = self.char_at(self.index + idx) else {
            return idx;
        };
        while self.index + idx >= 0 && self.is_space(c) {
            idx += step;
            match self.char_at(self.index + idx) {
                Some(n) => c = n,
                None => return idx,
            }
        }
        idx
    }

    /// `skip_to_character(character, idx)`: offset of the next unescaped char in
    /// `targets` at or after `idx` (or the distance to the end).
    fn skip_to_character(&mut self, targets: scan::Targets, idx: isize) -> isize {
        let start = self.index + idx;
        if start < 0 {
            return self.skip_to_character_slow(targets, idx);
        }
        match self.next_unescaped(targets, start as usize) {
            Some(i) => i as isize - self.index,
            None => self.len() - self.index,
        }
    }

    fn skip_to_character_slow(&self, targets: scan::Targets, idx: isize) -> isize {
        let n = self.len();
        let mut i = self.index + idx;
        let mut backslashes = 0usize;
        while i < n {
            let Some(ch) = self.char_at(i) else { break };
            if ch == '\\' {
                backslashes += 1;
                i += 1;
                continue;
            }
            if targets.contains(ch) && backslashes % 2 == 0 {
                return i - self.index;
            }
            backslashes = 0;
            i += 1;
        }
        n - self.index
    }

    // ------------------------------------------------------------------
    // Top level
    // ------------------------------------------------------------------

    /// `parse()`.
    pub(crate) fn parse(&mut self) -> R<Value> {
        let first = self.parse_json()?;
        if self.index >= self.len() {
            return Ok(first);
        }
        self.log(|| "The parser returned early, checking if there's more json elements".into());
        let mut json = vec![first];
        while self.index < self.len() {
            self.ctx.clear();
            self.deferred.clear();
            let is_comma_separated = self.next_top_level_value_is_comma_separated();
            let element_start = self.index;
            let j = self.parse_json()?;
            if self.strict && self.index > element_start {
                self.log(|| {
                    "Multiple top-level JSON elements found in strict mode, raising an error".into()
                });
                return Err(ParseError::Strict(
                    "Multiple top-level JSON elements found in strict mode.",
                ));
            }
            if j.truthy() {
                let last = json.last().expect("never empty");
                if !is_comma_separated && super::value::is_same_object(last, &j) {
                    // Treat repeated objects as updates: keep the newest value.
                    super::value::drop_deep(json.pop().expect("never empty"));
                } else if !last.truthy() {
                    super::value::drop_deep(json.pop().expect("never empty"));
                }
                json.push(j);
            } else {
                self.index += 1;
            }
        }
        if json.len() == 1 {
            self.log(|| {
                "There were no more elements, returning the element without the array".into()
            });
            return Ok(json.pop().expect("one element"));
        }
        Ok(Value::Array(json))
    }

    /// `_next_top_level_value_is_comma_separated`.
    fn next_top_level_value_is_comma_separated(&mut self) -> bool {
        let idx = self.scroll_whitespaces(0);
        if self.get_char_at(idx) == Some(',') {
            return true;
        }
        let mut i = self.index - 1;
        while i >= 0 && self.is_space(self.s[i as usize]) {
            i -= 1;
        }
        i >= 0 && self.s[i as usize] == ','
    }

    /// `_initial_container_has_non_comma_trailing_content`.
    fn initial_container_has_non_comma_trailing_content(&mut self) -> bool {
        if self.index != 0 {
            return true;
        }
        if let Some(r) = self.scan.initial_container {
            return r;
        }
        let r = self.initial_container_scan();
        self.scan.initial_container = Some(r);
        r
    }

    fn initial_container_scan(&self) -> bool {
        let mut containers: Vec<char> = Vec::new();
        let mut in_string = false;
        let mut escaped = false;
        let n = self.s.len();
        for (i, &ch) in self.s.iter().enumerate() {
            if in_string {
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            match ch {
                '"' => in_string = true,
                '{' | '[' => containers.push(ch),
                '}' | ']' => {
                    let Some(open) = containers.pop() else {
                        return false;
                    };
                    let expected = if open == '{' { '}' } else { ']' };
                    if ch != expected {
                        return false;
                    }
                    if containers.is_empty() {
                        let mut t = i + 1;
                        while t < n && self.is_space(self.s[t]) {
                            t += 1;
                        }
                        return t < n && self.s[t] != ',';
                    }
                }
                _ => {}
            }
        }
        false
    }

    /// `_try_parse_valid_json_value`.
    fn try_parse_valid_json_value(&mut self) -> R<Option<Value>> {
        if !self.try_valid_json_suffix || self.has_tried_valid_json_suffix || !self.ctx_empty() {
            return Ok(None);
        }
        self.has_tried_valid_json_suffix = true;
        let start = self.index.max(0) as usize;
        match pyjson::raw_decode(&self.s, start, self.max_int_digits) {
            Ok((value, end)) => {
                self.index = end as isize;
                Ok(Some(value))
            }
            Err(pyjson::DecodeError::Json) => Ok(None),
            Err(pyjson::DecodeError::Int(e)) => Err(ParseError::Int(e)),
            Err(pyjson::DecodeError::ReservedCodePoint) => Err(ParseError::Unsupported),
        }
    }

    /// `_try_parse_missing_opening_object`.
    fn try_parse_missing_opening_object(&mut self, delimiter: char) -> R<Option<Value>> {
        if self.scroll_whitespaces(-self.index) != 0 {
            return Ok(None);
        }
        let offset = self.skip_to_character(scan::Targets::one(delimiter), 1);
        if self.get_char_at(offset).is_none() {
            return Ok(None);
        }
        let offset = self.scroll_whitespaces(offset + 1);
        if self.get_char_at(offset) != Some(':') {
            return Ok(None);
        }
        let end_offset = self.scroll_whitespaces_back(self.len() - self.index - 1);
        if self.get_char_at(end_offset) != Some('}') {
            return Ok(None);
        }
        // Require a single insertion to explain the whole input.
        let mut candidate = Vec::with_capacity(self.s.len() + 1);
        candidate.push('{');
        candidate.extend_from_slice(&self.s);
        let (value, end) = match pyjson::raw_decode(&candidate, 0, self.max_int_digits) {
            Ok(r) => r,
            Err(pyjson::DecodeError::ReservedCodePoint) => return Err(ParseError::Unsupported),
            // Decoder limits (such as oversized integers) also decline.
            Err(_) => return Ok(None),
        };
        if candidate[end..]
            .iter()
            .any(|c| !matches!(c, ' ' | '\t' | '\r' | '\n'))
        {
            return Ok(None);
        }
        if self.strict {
            return Err(ParseError::Strict(
                "Missing opening object brace in strict mode.",
            ));
        }
        self.log(|| "Inserted missing opening object brace".into());
        self.index = self.len();
        Ok(Some(value))
    }

    /// `parse_json()` (no schema).
    fn parse_json(&mut self) -> R<Value> {
        if !self.deferred.is_empty() {
            let deferred = std::mem::take(&mut self.deferred);
            let n = deferred.len();
            self.ctx.extend(deferred);
            let r = self.parse_json();
            for _ in 0..n {
                self.ctx_reset();
            }
            return r;
        }
        let mut check_initial_quote = self.index == 0 && self.ctx_empty();
        loop {
            let Some(ch) = self.cur() else {
                return Ok(Value::empty_str());
            };
            if self.try_valid_json_suffix
                && matches!(ch, '{' | '[')
                && self.initial_container_has_non_comma_trailing_content()
            {
                if let Some(v) = self.try_parse_valid_json_value()? {
                    return Ok(v);
                }
            }
            match ch {
                '{' => {
                    self.index += 1;
                    return self.parse_object();
                }
                '[' => {
                    self.index += 1;
                    return Ok(Value::Array(self.parse_array(']')?));
                }
                '(' => {
                    // Keep top-level tuple detection conservative.
                    if !self.ctx_empty() || self.top_level_parenthesized_can_start_value() {
                        return self.parse_parenthesized();
                    }
                    self.index += 1;
                    continue;
                }
                _ => {}
            }
            if !self.ctx_empty() && (is_delim(ch) || self.props.is_alpha(ch)) {
                return self.parse_string();
            }
            if !self.ctx_empty() && (self.props.is_digit(ch) || ch == '-' || ch == '.') {
                return self.parse_number();
            }
            if ch == '#' || ch == '/' {
                return self.parse_comment();
            }
            if is_delim(ch) && check_initial_quote {
                check_initial_quote = false;
                if let Some(v) = self.try_parse_missing_opening_object(ch)? {
                    return Ok(v);
                }
            }
            self.index += 1;
        }
    }
}

/// Stack for the rerun of deep inputs: room for [`MAX_DEPTH`] levels (address
/// space only; pages are committed as they are touched).
const BIG_STACK: usize = 1 << 30;

/// Run the parser over `s`: the value and, with `logging`, the repair log.
pub(crate) fn parse(s: &[char], opts: &ParseOptions) -> R<(Value, Option<Vec<LogEntry>>)> {
    let marker = 0u8;
    let base = std::ptr::addr_of!(marker).addr();
    let mut p = Parser::new(s, opts, Some(base));
    match p.parse() {
        Err(ParseError::NeedStack) => {}
        Ok(v) => return Ok((v, p.logger.take())),
        Err(e) => return Err(e),
    }
    drop(p);
    // Deep input: rerun from scratch on a thread whose stack fits MAX_DEPTH.
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .name("pyroid-json-repair-deep".into())
            .stack_size(BIG_STACK)
            .spawn_scoped(scope, || {
                let mut p = Parser::new(s, opts, None);
                let v = p.parse()?;
                Ok((v, p.logger.take()))
            });
        match handle {
            Ok(h) => h.join().unwrap_or_else(|e| std::panic::resume_unwind(e)),
            // No thread (address space or thread limits): refuse as too deep.
            Err(_) => Err(ParseError::Depth),
        }
    })
}
