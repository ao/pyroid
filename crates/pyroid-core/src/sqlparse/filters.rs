// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! `sqlparse.filters`: the token filters (keyword/identifier case, string truncation), the
//! statement filters (strip comments, strip whitespace, spaces around operators, trailing
//! semicolons), the python/php output filters and the serializer (`SerializerUnicode` with
//! `utils.split_unquoted_newlines`).

use std::borrow::Cow;

use super::grouping::{RECURSION_MESSAGE, STACK_BUDGET};
use super::pystr;
use super::tree::{Class, Fail, Id, R, Str, Tree};
use super::ttype::{TT, Types, bits, builtin as T};
use crate::ftfy::UnicodeVersion;

#[allow(clippy::inline_always)]
#[inline(always)]
fn stack_position() -> usize {
    let marker = std::hint::black_box(0u8);
    std::ptr::from_ref(&marker).addr()
}

/// A recursion guard for the filters' walks over sublists.
#[derive(Clone, Copy, Debug)]
pub struct Depth {
    base: usize,
}

impl Depth {
    /// A guard based at the caller's frame.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn new() -> Depth {
        Depth {
            base: stack_position(),
        }
    }

    /// Fails with sqlparse's recursion message past the stack budget or the depth cap.
    #[allow(clippy::inline_always)]
    #[inline(always)]
    pub fn check(&self, depth: usize) -> R<()> {
        if depth > super::grouping::HARD_DEPTH
            || stack_position().abs_diff(self.base) > STACK_BUDGET
        {
            Err(Fail::SqlParse(RECURSION_MESSAGE.to_owned()))
        } else {
            Ok(())
        }
    }
}

impl Default for Depth {
    fn default() -> Self {
        Depth::new()
    }
}

/// `str.upper`, `str.lower` or `str.capitalize`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Case {
    /// `'upper'`.
    Upper,
    /// `'lower'`.
    Lower,
    /// `'capitalize'`.
    Capitalize,
}

impl Case {
    /// The conversion.
    pub fn apply(self, s: &str, v: UnicodeVersion) -> Cow<'_, str> {
        match self {
            Case::Upper => pystr::upper(s, v),
            Case::Lower => pystr::lower(s, v),
            Case::Capitalize => pystr::capitalize(s, v),
        }
    }
}

/// One item of the token stream: a type and a value.
pub type Item = (TT, Str);

fn set(item: &mut Str, s: Cow<'_, str>) {
    if let Cow::Owned(o) = s {
        *item = Str::owned(o);
    }
}

/// The preprocess filters, in `build_filter_stack`'s order.
#[derive(Clone, Debug, Default)]
pub struct TokenFilters {
    /// `KeywordCaseFilter(case)`.
    pub keyword_case: Option<Case>,
    /// `IdentifierCaseFilter(case)`.
    pub identifier_case: Option<Case>,
    /// `TruncateStringFilter(width, char)`.
    pub truncate: Option<(usize, String)>,
}

impl TokenFilters {
    /// Whether no filter is set.
    pub fn is_empty(&self) -> bool {
        self.keyword_case.is_none() && self.identifier_case.is_none() && self.truncate.is_none()
    }

    /// Apply the filters to one item.
    ///
    /// # Errors
    /// [`Fail::Python`] where the filter raises (`value.strip()[0]` of a blank name).
    pub fn apply(&self, src: &str, types: &Types, v: UnicodeVersion, item: &mut Item) -> R<()> {
        let text = |s: &Str| -> String {
            match s {
                Str::Span(a, b) => src[*a as usize..*b as usize].to_owned(),
                Str::Owned(o) => o.to_string(),
            }
        };
        if let Some(case) = self.keyword_case {
            if types.has(item.0, bits::KEYWORD) {
                let cur = text(&item.1);
                set(&mut item.1, Cow::Owned(case.apply(&cur, v).into_owned()));
                if let Str::Owned(o) = &item.1 {
                    if **o == *cur {
                        // unchanged: keep the cheaper span form if it was one
                    }
                }
            }
        }
        if let Some(case) = self.identifier_case {
            if item.0 == T::NAME || item.0 == T::STRING_SYMBOL {
                let cur = text(&item.1);
                let first = pystr::strip(&cur).chars().next().ok_or(Fail::Python)?;
                if first != '"' {
                    let conv = case.apply(&cur, v).into_owned();
                    item.1 = Str::owned(conv);
                }
            }
        }
        if let Some((width, ch)) = &self.truncate {
            if item.0 == T::STRING_SINGLE {
                let cur = text(&item.1);
                let chars: Vec<char> = cur.chars().collect();
                let (inner, quote): (&[char], &str) = if cur.starts_with("''") {
                    (py_slice(&chars, 2, -2), "''")
                } else {
                    (py_slice(&chars, 1, -1), "'")
                };
                if inner.len() > *width {
                    let mut s = String::with_capacity(cur.len() + ch.len());
                    s.push_str(quote);
                    s.extend(&inner[..*width]);
                    s.push_str(ch);
                    s.push_str(quote);
                    item.1 = Str::owned(s);
                }
            }
        }
        Ok(())
    }
}

/// `chars[a:b]` with Python's slice semantics.
fn py_slice(chars: &[char], a: isize, b: isize) -> &[char] {
    let lo = super::tree::py_bound(chars.len(), a);
    let hi = super::tree::py_bound(chars.len(), b);
    if hi <= lo { &[] } else { &chars[lo..hi] }
}

// ---------------------------------------------------------------------------
// Statement filters
// ---------------------------------------------------------------------------

/// `StripCommentsFilter().process(stmt)`.
///
/// # Errors
/// [`Fail::Python`] where sqlparse's code raises.
pub fn strip_comments(t: &mut Tree<'_>, stmt: Id, d: Depth, depth: usize) -> R<()> {
    d.check(depth)?;
    let mut i = 0;
    while let Some((next, sub)) = t.sublist_at(stmt, i) {
        if let Some(sg) = sub {
            strip_comments(t, sg, d, depth + 1)?;
        }
        i = next;
    }
    strip_comments_one(t, stmt)
}

/// `_get_insert_token(token)`: a newline token with the line breaks the comment ends with
/// (`re.search(r'([\r\n]+) *$', value)`), else a space.
fn insert_token(t: &mut Tree<'_>, comment: Id) -> Id {
    let value = t.val(comment).to_owned();
    match trailing_breaks(&value) {
        Some(g) => t.token(T::NEWLINE, Str::owned(g)),
        None => t.token(T::WHITESPACE, Str::owned(" ")),
    }
}

/// Group 1 of `re.search(r'([\r\n]+) *$', s)`.
fn trailing_breaks(s: &str) -> Option<&str> {
    let b = s.as_bytes();
    let n = b.len();
    let is_end = |x: usize| x == n || (x + 1 == n && b[x] == b'\n');
    let brk = |c: u8| c == b'\r' || c == b'\n';
    for p in 0..n {
        if !brk(b[p]) {
            continue;
        }
        let mut q = p;
        while q < n && brk(b[q]) {
            q += 1;
        }
        // greedy [\r\n]+ first, then shorter
        let mut k = q;
        while k > p {
            let mut r = k;
            if k == q {
                while r < n && b[r] == b' ' {
                    r += 1;
                }
            }
            // ` *` backs off too, but a space position is never an end
            if is_end(r) || (k < q && is_end(k)) {
                return Some(&s[p..k]);
            }
            k -= 1;
        }
    }
    None
}

fn strip_comments_one(t: &mut Tree<'_>, tlist: Id) -> R<()> {
    let next_comment = |t: &Tree<'_>, idx: isize| {
        t.next_by(tlist, idx, |s, x| {
            s.isa(x, Class::Comment) || s.within(x, T::COMMENT)
        })
    };
    let mut cur = next_comment(t, -1)?;
    while let Some((tidx, token)) = cur {
        let mut tidx = isize::try_from(tidx).unwrap_or(isize::MAX);
        let hint = |t: &Tree<'_>, x: Id| t.is_any(x, &[T::MULTILINE_HINT, T::SINGLE_HINT]);
        let mut is_hint = hint(t, token);
        if !is_hint && t.isa(token, Class::Comment) {
            if let Some(&first) = t.kids(token).first() {
                is_hint = hint(t, first);
            }
        }
        if is_hint {
            cur = next_comment(t, tidx)?;
            continue;
        }
        let prev = t.token_prev(tlist, Some(tidx), false, false)?.map(|x| x.1);
        let next = t.token_next(tlist, Some(tidx), false, false)?.map(|x| x.1);
        let paren_open = |t: &Tree<'_>, x: Id| t.m(x, T::PUNCTUATION, &["("]);
        let cond = match (prev, next) {
            (None, _) | (_, None) => true,
            (Some(p), Some(n)) => {
                t.n(p).is_whitespace
                    || paren_open(t, p)
                    || t.n(n).is_whitespace
                    || t.m(n, T::PUNCTUATION, &[")"])
            }
        };
        let l = t.list_of(tlist);
        if cond {
            if let Some(p) = prev {
                if !paren_open(t, p) {
                    let ins = insert_token(t, token);
                    t.list_insert(l, tidx, ins);
                }
            }
            t.list_remove(l, token)?;
            tidx -= 1;
        } else {
            let ins = insert_token(t, token);
            let list = &mut t.lists[l as usize];
            let j = super::tree::py_index(list.len(), tidx).ok_or(Fail::Python)?;
            list[j] = ins;
        }
        cur = next_comment(t, tidx)?;
    }
    Ok(())
}

/// `StripWhitespaceFilter().process(stmt)`.
///
/// # Errors
/// [`Fail::Python`] where sqlparse's code raises.
pub fn strip_whitespace(t: &mut Tree<'_>, stmt: Id, d: Depth, depth: usize) -> R<()> {
    d.check(depth)?;
    let mut i = 0;
    while let Some((next, sub)) = t.sublist_at(stmt, i) {
        if let Some(sg) = sub {
            strip_whitespace(t, sg, d, depth + 1)?;
        }
        i = next;
    }
    match t.n(stmt).class {
        Class::IdentifierList => stripws_identifierlist(t, stmt)?,
        Class::Parenthesis => stripws_parenthesis(t, stmt)?,
        _ => stripws_default(t, stmt),
    }
    if depth == 0 {
        let l = t.list_of(stmt);
        if let Some(&last) = t.lists[l as usize].last() {
            if t.n(last).is_whitespace {
                t.lists[l as usize].pop();
            }
        }
    }
    Ok(())
}

fn stripws_default(t: &mut Tree<'_>, tlist: Id) {
    let mut last_was_ws = false;
    let mut is_first_char = true;
    let l = t.list_of(tlist);
    let mut i = 0;
    while i < t.lists[l as usize].len() {
        let tok = t.lists[l as usize][i];
        let ws = t.n(tok).is_whitespace;
        if ws {
            t.n_mut(tok).value = if last_was_ws || is_first_char {
                Str::EMPTY
            } else {
                Str::owned(" ")
            };
        }
        last_was_ws = ws;
        is_first_char = false;
        i += 1;
    }
}

fn stripws_identifierlist(t: &mut Tree<'_>, tlist: Id) -> R<()> {
    let l = t.list_of(tlist);
    let snapshot: Vec<Id> = t.lists[l as usize].clone();
    let mut last_nl: Option<Id> = None;
    for tok in snapshot {
        if let Some(nl) = last_nl {
            if t.is(tok, T::PUNCTUATION) && t.val(tok) == "," {
                t.list_remove(l, nl)?;
            }
        }
        last_nl = t.n(tok).is_whitespace.then_some(tok);
    }
    stripws_default(t, tlist);
    Ok(())
}

fn stripws_parenthesis(t: &mut Tree<'_>, tlist: Id) -> R<()> {
    let l = t.list_of(tlist);
    let ok = {
        let k = &t.lists[l as usize];
        k.len() >= 2
            && t.m(k[0], T::PUNCTUATION, &["("])
            && t.m(k[k.len() - 1], T::PUNCTUATION, &[")"])
    };
    if !ok {
        stripws_default(t, tlist);
        return Ok(());
    }
    loop {
        let k = &t.lists[l as usize];
        let x = *k.get(1).ok_or(Fail::Python)?;
        if !t.n(x).is_whitespace {
            break;
        }
        t.lists[l as usize].remove(1);
    }
    loop {
        let k = &t.lists[l as usize];
        if k.len() < 2 {
            return Err(Fail::Python);
        }
        let x = k[k.len() - 2];
        if !t.n(x).is_whitespace {
            break;
        }
        let n = k.len();
        t.lists[l as usize].remove(n - 2);
    }
    let k = &t.lists[l as usize];
    let second_last = k[k.len() - 2];
    if t.n(second_last).is_group {
        let gl = t.list_of(second_last);
        loop {
            let last = *t.lists[gl as usize].last().ok_or(Fail::Python)?;
            if !t.n(last).is_whitespace {
                break;
            }
            t.lists[gl as usize].pop();
        }
    }
    stripws_default(t, tlist);
    Ok(())
}

/// `SpacesAroundOperatorsFilter().process(stmt)`.
///
/// # Errors
/// [`Fail::Python`] where sqlparse's code raises.
pub fn spaces_around_operators(t: &mut Tree<'_>, stmt: Id, d: Depth, depth: usize) -> R<()> {
    d.check(depth)?;
    let mut i = 0;
    while let Some((next, sub)) = t.sublist_at(stmt, i) {
        if let Some(sg) = sub {
            spaces_around_operators(t, sg, d, depth + 1)?;
        }
        i = next;
    }
    let pred = |s: &Tree<'_>, x: Id| s.is_any(x, &[T::OPERATOR, T::COMPARISON]);
    let mut cur = t.next_by(stmt, -1, pred)?;
    while let Some((tidx, _)) = cur {
        let mut tidx = isize::try_from(tidx).unwrap_or(isize::MAX);
        let next = t.token_next(stmt, Some(tidx), false, false)?;
        if let Some((_, n)) = next {
            if t.n(n).ttype != Some(T::WHITESPACE) {
                let ws = t.token(T::WHITESPACE, Str::owned(" "));
                t.insert_after(stmt, tidx, ws, true)?;
            }
        }
        let prev = t.token_prev(stmt, Some(tidx), false, false)?;
        if let Some((_, p)) = prev {
            if t.n(p).ttype != Some(T::WHITESPACE) {
                let ws = t.token(T::WHITESPACE, Str::owned(" "));
                t.insert_before(stmt, tidx, ws);
                tidx += 1;
            }
        }
        cur = t.next_by(stmt, tidx, pred)?;
    }
    Ok(())
}

/// `StripTrailingSemicolonFilter().process(stmt)`.
pub fn strip_trailing_semicolon(t: &mut Tree<'_>, stmt: Id) {
    let l = t.list_of(stmt);
    while let Some(&last) = t.lists[l as usize].last() {
        if t.n(last).is_whitespace || t.val(last) == ";" {
            t.lists[l as usize].pop();
        } else {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Output and serializer
// ---------------------------------------------------------------------------

/// `output_format`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// `OutputPythonFilter`.
    Python,
    /// `OutputPHPFilter`.
    Php,
}

/// `OutputFilter.process(stmt)` followed by `str(stmt)`: the statement's text after the
/// output filter replaced its tokens (a generator over the top-level tokens; groups are
/// written with their `value`, as the filter does).
pub fn output_text(t: &Tree<'_>, stmt: Id, kind: Output, count: usize) -> String {
    let (prefix, base) = match kind {
        Output::Python => ("", "sql"),
        Output::Php => ("$", "sql"),
    };
    let varname = if count > 1 {
        format!("{prefix}{base}{count}")
    } else {
        format!("{prefix}{base}")
    };
    let full = t.text(stmt);
    let has_nl = splitlines_count(pystr::strip(&full)) > 1;
    let mut out = String::new();
    if count > 1 {
        out.push('\n');
    }
    match kind {
        Output::Python => {
            out.push_str(&varname);
            out.push_str(" = ");
            if has_nl {
                out.push('(');
            }
            out.push('\'');
            for &tok in t.kids(stmt) {
                let n = t.n(tok);
                let value = t.val(tok);
                if n.is_whitespace && value.contains('\n') {
                    out.push_str(" '");
                    out.push('\n');
                    out.extend(std::iter::repeat_n(' ', pystr::py_len(&varname) + 4));
                    out.push('\'');
                    if let Some((_, after)) = value.split_once('\n') {
                        out.push_str(after);
                    }
                    continue;
                }
                out.push_str(&value.replace('\\', "\\\\").replace('\'', "\\'"));
            }
            out.push('\'');
            if has_nl {
                out.push(')');
            }
        }
        Output::Php => {
            out.push_str(&varname);
            out.push(' ');
            if has_nl {
                out.push(' ');
            }
            out.push_str("= \"");
            for &tok in t.kids(stmt) {
                let n = t.n(tok);
                let value = t.val(tok);
                if n.is_whitespace && value.contains('\n') {
                    out.push_str(" \";");
                    out.push('\n');
                    out.push_str(&varname);
                    out.push_str(" .= \"");
                    if let Some((_, after)) = value.split_once('\n') {
                        out.push_str(after);
                    }
                    continue;
                }
                out.push_str(
                    &value
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('$', "\\$"),
                );
            }
            out.push_str("\";");
        }
    }
    out
}

/// `len(s.splitlines())`.
pub fn splitlines_count(s: &str) -> usize {
    pystr::splitlines(s).len()
}

/// `SerializerUnicode.process`: `'\n'.join(line.rstrip() for line in
/// split_unquoted_newlines(text))`, appended to `out`.
pub fn serialize(text: &str, out: &mut String) {
    let b = text.as_bytes();
    let n = b.len();
    let mut line_start = 0;
    let mut pos = 0;
    // unclosed_before for '"' and "'"
    let mut unclosed = [0usize; 2];
    let mut first = true;
    let mut push_line = |out: &mut String, line: &str| {
        if !first {
            out.push('\n');
        }
        first = false;
        out.push_str(pystr::rstrip(line));
    };
    while pos < n {
        let c = b[pos];
        if c == b'\r' || c == b'\n' {
            push_line(out, &text[line_start..pos]);
            pos += if b[pos..].starts_with(b"\r\n") { 2 } else { 1 };
            line_start = pos;
        } else if c == b'"' || c == b'\'' {
            let k = usize::from(c == b'\'');
            if pos >= unclosed[k] {
                let end = string_content_end(text, pos + 1, c);
                if end < n && b[end] == c {
                    pos = end + 1;
                    continue;
                }
                unclosed[k] = end;
            }
            pos += 1;
        } else {
            // _UNQUOTED_TEXT = [^\r\n'"]+
            while pos < n && !matches!(b[pos], b'\r' | b'\n' | b'\'' | b'"') {
                pos += 1;
            }
        }
    }
    push_line(out, &text[line_start..]);
}

/// The end of `(?:[^q\\]|\\.)*` matched at `start` (`.` is any character but `\n`).
fn string_content_end(text: &str, start: usize, q: u8) -> usize {
    let b = text.as_bytes();
    let n = b.len();
    let mut i = start;
    while i < n {
        let c = b[i];
        if c == q {
            break;
        }
        if c == b'\\' {
            // `\\.`: the next character must exist and not be '\n'
            match text[i + 1..].chars().next() {
                Some(ch) if ch != '\n' => i += 1 + ch.len_utf8(),
                _ => break,
            }
        } else {
            i += text[i..].chars().next().map_or(1, char::len_utf8);
        }
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn breaks() {
        assert_eq!(trailing_breaks("-- c\n"), Some("\n"));
        assert_eq!(trailing_breaks("-- c\r\n"), Some("\r\n"));
        assert_eq!(trailing_breaks("/* a */"), None);
        assert_eq!(trailing_breaks("a\n\nb\n"), Some("\n"));
        assert_eq!(trailing_breaks("a\n  "), Some("\n"));
        assert_eq!(trailing_breaks("a\n\n"), Some("\n\n"));
    }

    #[test]
    fn serializer() {
        let mut s = String::new();
        serialize("a  \nb 'x\n' c\r\nd", &mut s);
        assert_eq!(s, "a\nb 'x\n' c\nd");
        let mut s = String::new();
        serialize("'unclosed\nx", &mut s);
        assert_eq!(s, "'unclosed\nx");
    }
}
