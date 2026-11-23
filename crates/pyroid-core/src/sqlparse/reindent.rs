// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! `sqlparse.filters.ReindentFilter` and `AlignedIndentFilter`, ported line by line.

use std::sync::OnceLock;

use super::filters::Depth;
use super::pystr;
use super::regex::{FLAG_IGNORECASE, FLAG_UNICODE, Regex};
use super::tree::{Class, Fail, Id, R, Str, Tree, py_index};
use super::ttype::builtin as T;
use crate::ftfy::UnicodeVersion;

/// The `regex=True` match patterns, compiled with and without IGNORECASE.
struct Patterns {
    reindent_ci: Vec<Regex>,
    reindent_cs: Vec<Regex>,
    aligned_ci: Vec<Regex>,
    aligned_cs: Vec<Regex>,
    join_ci: Vec<Regex>,
    join_cs: Vec<Regex>,
    by_ci: Vec<Regex>,
    by_cs: Vec<Regex>,
}

const REINDENT_SPLIT: [&str; 14] = [
    "FROM",
    "STRAIGHT_JOIN$",
    "JOIN$",
    "AND",
    "OR",
    "GROUP BY",
    "ORDER BY",
    "UNION",
    "VALUES",
    "SET",
    "BETWEEN",
    "EXCEPT",
    "HAVING",
    "LIMIT",
];
const JOIN_WORDS: &str =
    r"((LEFT\s+|RIGHT\s+|FULL\s+)?(INNER\s+|OUTER\s+|STRAIGHT\s+)?|(CROSS\s+|NATURAL\s+)?)?JOIN\b";
const BY_WORDS: &str = r"(GROUP|ORDER)\s+BY\b";
const ALIGNED_SPLIT: [&str; 14] = [
    "FROM", JOIN_WORDS, "ON", BY_WORDS, "WHERE", "AND", "OR", "HAVING", "LIMIT", "UNION", "VALUES",
    "SET", "BETWEEN", "EXCEPT",
];

fn compile(pats: &[&str], ci: bool, v: UnicodeVersion) -> Vec<Regex> {
    let flags = FLAG_UNICODE | if ci { FLAG_IGNORECASE } else { 0 };
    pats.iter()
        .map(|p| Regex::new(p, flags, v).unwrap_or_else(|e| panic!("{p}: {e}")))
        .collect()
}

fn patterns(v: UnicodeVersion) -> &'static Patterns {
    static CACHE: [OnceLock<Patterns>; 5] = [const { OnceLock::new() }; 5];
    let i = match v {
        UnicodeVersion::V14_0 => 0,
        UnicodeVersion::V15_0 => 1,
        UnicodeVersion::V15_1 => 2,
        UnicodeVersion::V16_0 => 3,
        UnicodeVersion::V17_0 => 4,
    };
    CACHE[i].get_or_init(|| Patterns {
        reindent_ci: compile(&REINDENT_SPLIT, true, v),
        reindent_cs: compile(&REINDENT_SPLIT, false, v),
        aligned_ci: compile(&ALIGNED_SPLIT, true, v),
        aligned_cs: compile(&ALIGNED_SPLIT, false, v),
        join_ci: compile(&[JOIN_WORDS], true, v),
        join_cs: compile(&[JOIN_WORDS], false, v),
        by_ci: compile(&[BY_WORDS], true, v),
        by_cs: compile(&[BY_WORDS], false, v),
    })
}

fn ix(i: usize) -> isize {
    isize::try_from(i).unwrap_or(isize::MAX)
}

/// The `ReindentFilter` options.
#[derive(Clone, Debug)]
pub struct ReindentOptions {
    /// `width` (`indent_width`).
    pub width: i64,
    /// `char` (`indent_char`).
    pub ch: char,
    /// `wrap_after`.
    pub wrap_after: i64,
    /// `comma_first`.
    pub comma_first: bool,
    /// `indent_after_first`.
    pub indent_after_first: bool,
    /// `indent_columns`.
    pub indent_columns: bool,
    /// `compact`.
    pub compact: bool,
}

/// `ReindentFilter`'s state across statements.
#[derive(Debug)]
pub struct Reindent {
    o: ReindentOptions,
    indent: i64,
    offset: i64,
    /// `str(self._last_stmt).endswith('\n')`, once there was a statement.
    pub last_stmt_ends_nl: Option<bool>,
    /// `len(self._last_func.value)`.
    last_func_len: Option<usize>,
}

/// The per-statement context: the arena and the current statement.
struct Cx<'a, 't, 's> {
    r: &'a mut Reindent,
    t: &'t mut Tree<'s>,
    curr: Id,
    d: Depth,
}

impl Reindent {
    /// A filter with options `o`.
    pub fn new(o: ReindentOptions) -> Reindent {
        let indent = i64::from(o.indent_after_first);
        Reindent {
            o,
            indent,
            offset: 0,
            last_stmt_ends_nl: None,
            last_func_len: None,
        }
    }

    /// `process(stmt)`; afterwards the caller sets [`Reindent::last_stmt_ends_nl`] from the
    /// statement's final text.
    ///
    /// # Errors
    /// [`Fail::Python`] where sqlparse's code raises.
    pub fn process(&mut self, t: &mut Tree<'_>, stmt: Id) -> R<()> {
        let mut cx = Cx {
            r: self,
            t,
            curr: stmt,
            d: Depth::new(),
        };
        cx.process(stmt, 0)?;
        if let Some(ends) = cx.r.last_stmt_ends_nl {
            let nl = if ends { "\n" } else { "\n\n" };
            let tok = cx.t.token(T::WHITESPACE, Str::owned(nl));
            let l = cx.t.list_of(stmt);
            cx.t.list_insert(l, 0, tok);
        }
        Ok(())
    }
}

impl Cx<'_, '_, '_> {
    fn leading_ws(&self) -> i64 {
        self.r.offset + self.r.indent * self.r.o.width
    }

    /// `_current_line_len(token)`.
    fn current_line_len(&self, token: Id) -> R<i64> {
        let t = &*self.t;
        let mut length: i64 = 0;
        let mut node = token;
        loop {
            if node == self.curr {
                break;
            }
            let Some(parent) = t.n(node).parent else {
                break;
            };
            let kids = t.kids(parent);
            let at = kids.iter().position(|&x| x == node).ok_or(Fail::Python)?;
            let mut stack: Vec<Id> = kids[..at].to_vec();
            while let Some(prev) = stack.pop() {
                if t.n(prev).is_group {
                    stack.extend_from_slice(t.kids(prev));
                    continue;
                }
                let value = t.val(prev);
                let size = i64::try_from(pystr::py_len(value)).unwrap_or(i64::MAX);
                if size == 0 {
                    continue;
                }
                let lines = pystr::splitlines(value);
                if lines.len() == 1 && i64::try_from(pystr::py_len(lines[0])).unwrap_or(0) == size {
                    length += size;
                    continue;
                }
                let dotted = format!("{value}.");
                let lines = pystr::splitlines(&dotted);
                let last = lines.last().map_or(0, |l| pystr::py_len(l));
                let tail = i64::try_from(last).unwrap_or(0) - 1 + length;
                if tail != 0 {
                    return Ok(tail);
                }
                if lines.len() > 2 {
                    return Ok(i64::try_from(pystr::py_len(lines[lines.len() - 2])).unwrap_or(0));
                }
                length = i64::try_from(pystr::py_len(lines[0])).unwrap_or(0);
            }
            node = parent;
        }
        Ok(length)
    }

    fn get_offset(&self, token: Id) -> R<i64> {
        Ok(self.current_line_len(token)? - self.leading_ws().max(0))
    }

    fn nl(&mut self, offset: i64) -> Id {
        let n = (self.leading_ws() + offset).max(0);
        let mut s = String::with_capacity(1 + usize::try_from(n).unwrap_or(0));
        s.push('\n');
        s.extend(std::iter::repeat_n(
            self.r.o.ch,
            usize::try_from(n).unwrap_or(0),
        ));
        self.t.token(T::WHITESPACE, Str::owned(s))
    }

    fn next_token(&self, tlist: Id, idx: isize, guard: usize) -> R<Option<(usize, Id)>> {
        self.d.check(guard)?;
        let p = patterns(self.t.v);
        let found = self.t.next_by(tlist, idx, |s, x| {
            s.m_regex(x, T::KEYWORD, &p.reindent_ci, &p.reindent_cs)
        })?;
        if let Some((tidx, tok)) = found {
            if self.t.norm(tok) == "BETWEEN" {
                let again = self.next_token(tlist, ix(tidx), guard + 1)?;
                if let Some((t2, tok2)) = again {
                    if self.t.norm(tok2) == "AND" {
                        return self.next_token(tlist, ix(t2), guard + 1);
                    }
                }
                return Ok(again);
            }
        }
        Ok(found)
    }

    fn split_kwds(&mut self, tlist: Id) -> R<()> {
        let mut cur = self.next_token(tlist, -1, 0)?;
        while let Some((tidx, _)) = cur {
            let mut tidx = ix(tidx);
            let prev = self.t.token_prev(tlist, Some(tidx), false, false)?;
            let uprev_nl = match prev {
                None => false, // str(None) == 'None'
                Some((_, p)) => {
                    let s = self.t.text(p);
                    s.ends_with('\n') || s.ends_with('\r')
                }
            };
            if let Some((pidx, p)) = prev {
                if self.t.n(p).is_whitespace {
                    let l = self.t.list_of(tlist);
                    self.t.list_remove_at(l, ix(pidx))?;
                    tidx -= 1;
                }
            }
            if !uprev_nl {
                let nl = self.nl(0);
                self.t.insert_before(tlist, tidx, nl);
                tidx += 1;
            }
            cur = self.next_token(tlist, tidx, 0)?;
        }
        Ok(())
    }

    fn split_statements(&mut self, tlist: Id) -> R<()> {
        let pred = |s: &Tree<'_>, x: Id| s.is_any(x, &[T::DML, T::DDL]);
        let mut cur = self.t.next_by(tlist, -1, pred)?;
        while let Some((tidx, _)) = cur {
            let mut tidx = ix(tidx);
            let prev = self.t.token_prev(tlist, Some(tidx), false, false)?;
            if let Some((pidx, p)) = prev {
                if self.t.n(p).is_whitespace {
                    let l = self.t.list_of(tlist);
                    self.t.list_remove_at(l, ix(pidx))?;
                    tidx -= 1;
                }
                let nl = self.nl(0);
                self.t.insert_before(tlist, tidx, nl);
                tidx += 1;
            }
            cur = self.t.next_by(tlist, tidx, pred)?;
        }
        Ok(())
    }

    fn process(&mut self, tlist: Id, depth: usize) -> R<()> {
        self.d.check(depth)?;
        match self.t.n(tlist).class {
            Class::Where => self.process_where(tlist, depth),
            Class::Parenthesis => self.process_parenthesis(tlist, depth),
            Class::Function => {
                let first = self.t.at(tlist, 0)?;
                self.r.last_func_len = Some(pystr::py_len(self.t.val(first)));
                self.process_default(tlist, true, depth)
            }
            Class::IdentifierList => self.process_identifierlist(tlist, depth),
            Class::Case => self.process_case(tlist, depth),
            Class::Values => self.process_values(tlist),
            _ => self.process_default(tlist, true, depth),
        }
    }

    fn process_where(&mut self, tlist: Id, depth: usize) -> R<()> {
        let found = self
            .t
            .next_by(tlist, -1, |s, x| s.m(x, T::KEYWORD, &["WHERE"]))?;
        let Some((tidx, _)) = found else {
            return Ok(());
        };
        let nl = self.nl(0);
        self.t.insert_before(tlist, ix(tidx), nl);
        self.r.indent += 1;
        self.process_default(tlist, true, depth)?;
        self.r.indent -= 1;
        Ok(())
    }

    fn process_parenthesis(&mut self, tlist: Id, depth: usize) -> R<()> {
        let dml = self
            .t
            .next_by(tlist, -1, |s, x| s.is_any(x, &[T::DML, T::DDL]))?
            .is_some();
        let first = self
            .t
            .next_by(tlist, -1, |s, x| s.m(x, T::PUNCTUATION, &["("]))?;
        let Some((_, first)) = first else {
            return Ok(());
        };
        let add = i64::from(dml);
        self.r.indent += add;
        if dml {
            let nl = self.nl(0);
            let l = self.t.list_of(tlist);
            self.t.list_insert(l, 0, nl);
        }
        let off = self.get_offset(first)? + 1;
        self.r.offset += off;
        self.process_default(tlist, !dml, depth)?;
        self.r.offset -= off;
        self.r.indent -= add;
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn process_identifierlist(&mut self, tlist: Id, depth: usize) -> R<()> {
        let mut identifiers = self.t.get_identifiers(tlist);
        let tab = self.r.o.ch == '\t';
        let num_offset = if self.r.o.indent_columns {
            let f = *identifiers.first().ok_or(Fail::Python)?;
            let _first = self.t.first_leaf(f).ok_or(Fail::Python)?;
            if tab { 1 } else { self.r.o.width }
        } else {
            if identifiers.is_empty() {
                return Err(Fail::Python);
            }
            let f = identifiers.remove(0);
            let first = self.t.first_leaf(f).ok_or(Fail::Python)?;
            if tab { 1 } else { self.get_offset(first)? }
        };
        if !self.t.within_class(tlist, Class::Function)
            && !self.t.within_class(tlist, Class::Values)
        {
            self.r.offset += num_offset;
            let mut position: i64 = 0;
            for &tok0 in &identifiers {
                let mut token = tok0;
                position += i64::try_from(pystr::py_len(self.t.val(token))).unwrap_or(0) + 1;
                if position > self.r.o.wrap_after - self.r.offset {
                    let mut adjust = 0;
                    if self.r.o.comma_first {
                        adjust = -2;
                        let at = self.t.token_index(tlist, token, 0)?;
                        let comma = self.t.token_prev(tlist, Some(ix(at)), true, false)?;
                        let Some((_, comma)) = comma else { continue };
                        token = comma;
                    }
                    let nl = self.nl(adjust);
                    self.t.insert_before_tok(tlist, token, nl)?;
                    if self.r.o.comma_first {
                        let at = self.t.token_index(tlist, token, 0)?;
                        let ws = self.t.token_next(tlist, Some(ix(at)), false, false)?;
                        if let Some((_, w)) = ws {
                            if !self.t.is(w, T::WHITESPACE) {
                                let sp = self.t.token(T::WHITESPACE, Str::owned(" "));
                                self.t.insert_after_tok(tlist, token, sp, true)?;
                            }
                        }
                    }
                    position = 0;
                }
            }
            self.r.offset -= num_offset;
        } else {
            // ensure whitespace: `for token in tlist` over the live list
            let mut i = 0;
            while i < self.t.len(tlist) {
                let token = self.t.kids(tlist)[i];
                let at = self.t.token_index(tlist, token, 0)?;
                let next_ws = self
                    .t
                    .token_next(tlist, Some(ix(at)), false, false)?
                    .map(|x| x.1);
                if self.t.val(token) == "," {
                    let nw = next_ws.ok_or(Fail::Python)?;
                    if !self.t.n(nw).is_whitespace {
                        let sp = self.t.token(T::WHITESPACE, Str::owned(" "));
                        self.t.insert_after_tok(tlist, token, sp, true)?;
                    }
                }
                i += 1;
            }
            let sum: i64 = identifiers
                .iter()
                .map(|&x| i64::try_from(pystr::py_len(self.t.val(x))).unwrap_or(0) + 1)
                .sum();
            let end_at = self.r.offset + sum;
            let mut adjusted_offset = 0;
            if self.r.o.wrap_after > 0 && end_at > self.r.o.wrap_after - self.r.offset {
                if let Some(len) = self.r.last_func_len {
                    adjusted_offset = -i64::try_from(len).unwrap_or(0) - 1;
                }
            }
            self.r.offset += adjusted_offset;
            self.r.indent += 1;
            if adjusted_offset < 0 {
                let f = *identifiers.first().ok_or(Fail::Python)?;
                let nl = self.nl(0);
                self.t.insert_before_tok(tlist, f, nl)?;
            }
            let mut position: i64 = 0;
            for &token in &identifiers {
                position += i64::try_from(pystr::py_len(self.t.val(token))).unwrap_or(0) + 1;
                if self.r.o.wrap_after > 0 && position > self.r.o.wrap_after - self.r.offset {
                    let nl = self.nl(0);
                    self.t.insert_before_tok(tlist, token, nl)?;
                    position = 0;
                }
            }
            self.r.indent -= 1;
            self.r.offset -= adjusted_offset;
        }
        self.process_default(tlist, true, depth)
    }

    fn process_case(&mut self, tlist: Id, depth: usize) -> R<()> {
        let cases = self.t.get_cases(tlist, false);
        let mut it = cases.into_iter();
        let (cond, _) = it.next().ok_or(Fail::Python)?;
        let cond = cond.ok_or(Fail::Python)?;
        let c0 = *cond.first().ok_or(Fail::Python)?;
        let first = self.t.first_leaf(c0).ok_or(Fail::Python)?;
        let t0 = self.t.at(tlist, 0)?;
        let o1 = self.get_offset(t0)?;
        self.r.offset += o1;
        let o2 = self.get_offset(first)?;
        self.r.offset += o2;
        for (cond, value) in it {
            let mut str_cond_len = 0usize;
            if let Some(c) = &cond {
                for &x in c {
                    str_cond_len += pystr::py_len(&self.t.text(x));
                }
            }
            let mut str_value_len = 0usize;
            for &x in &value {
                str_value_len += pystr::py_len(&self.t.text(x));
            }
            let end_pos = self.r.offset
                + 1
                + i64::try_from(str_cond_len).unwrap_or(0)
                + i64::try_from(str_value_len).unwrap_or(0);
            if !self.r.o.compact && end_pos > self.r.o.wrap_after {
                let token = match &cond {
                    None => *value.first().ok_or(Fail::Python)?,
                    Some(c) => *c.first().ok_or(Fail::Python)?,
                };
                let nl = self.nl(0);
                self.t.insert_before_tok(tlist, token, nl)?;
            }
        }
        let when = 5;
        self.r.offset += when;
        self.process_default(tlist, true, depth)?;
        self.r.offset -= when;
        self.r.offset -= o2;
        let end = self
            .t
            .next_by(tlist, -1, |s, x| s.m(x, T::KEYWORD, &["END"]))?;
        if let Some((end_idx, _)) = end {
            if !self.r.o.compact {
                let nl = self.nl(0);
                self.t.insert_before(tlist, ix(end_idx), nl);
            }
        }
        self.r.offset -= o1;
        Ok(())
    }

    fn process_values(&mut self, tlist: Id) -> R<()> {
        let nl = self.nl(0);
        self.t.insert_before(tlist, 0, nl);
        let paren = |s: &Tree<'_>, x: Id| s.isa(x, Class::Parenthesis);
        let mut cur = self.t.next_by(tlist, -1, paren)?;
        let first_token = cur.map(|x| x.1);
        while let Some((tidx, token)) = cur {
            let comma = self
                .t
                .next_by(tlist, ix(tidx), |s, x| s.m(x, T::PUNCTUATION, &[","]))?;
            if let Some((_, ptoken)) = comma {
                if self.r.o.comma_first {
                    let ft = first_token.ok_or(Fail::Python)?;
                    let off = self.get_offset(ft)? - 2;
                    let nl = self.nl(off);
                    self.t.insert_before_tok(tlist, ptoken, nl)?;
                } else {
                    let off = self.get_offset(token)?;
                    let nl = self.nl(off);
                    self.t.insert_after_tok(tlist, ptoken, nl, true)?;
                }
            }
            cur = self.t.next_by(tlist, ix(tidx), paren)?;
        }
        Ok(())
    }

    fn process_default(&mut self, tlist: Id, stmts: bool, depth: usize) -> R<()> {
        if stmts {
            self.split_statements(tlist)?;
        }
        self.split_kwds(tlist)?;
        let mut i = 0;
        while let Some((next, sub)) = self.t.sublist_at(tlist, i) {
            if let Some(sg) = sub {
                self.process(sg, depth + 1)?;
            }
            i = next;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AlignedIndentFilter
// ---------------------------------------------------------------------------

/// `AlignedIndentFilter(char)`.
#[derive(Debug)]
pub struct Aligned {
    ch: char,
    offset: i64,
    indent: i64,
}

const MAX_KWD_LEN: i64 = 6;

impl Aligned {
    /// A filter indenting with `ch`.
    pub fn new(ch: char) -> Aligned {
        Aligned {
            ch,
            offset: 0,
            indent: 0,
        }
    }

    /// `process(stmt)`.
    ///
    /// # Errors
    /// [`Fail::Python`] where sqlparse's code raises.
    pub fn process(&mut self, t: &mut Tree<'_>, stmt: Id) -> R<()> {
        let d = Depth::new();
        self.process_node(t, stmt, d, 0)
    }

    fn nl(&self, t: &mut Tree<'_>, offset: i64) -> Id {
        let indent = self.indent * (2 + MAX_KWD_LEN);
        let n = (MAX_KWD_LEN + offset + indent + self.offset).max(0);
        let mut s = String::from("\n");
        s.extend(std::iter::repeat_n(
            self.ch,
            usize::try_from(n).unwrap_or(0),
        ));
        t.token(T::WHITESPACE, Str::owned(s))
    }

    fn process_node(&mut self, t: &mut Tree<'_>, tlist: Id, d: Depth, depth: usize) -> R<()> {
        d.check(depth)?;
        match t.n(tlist).class {
            Class::Statement => {
                let l = t.list_of(tlist);
                if let Some(&f) = t.lists[l as usize].first() {
                    if t.n(f).is_whitespace && self.indent == 0 {
                        t.lists[l as usize].remove(0);
                    }
                }
                // self._process(sql.TokenList(tlist.tokens)): a new group on the same list
                let tl = t.group_on(Class::TokenList, l);
                self.process_node(t, tl, d, depth + 1)
            }
            Class::Parenthesis => {
                let sel = t.next_by(tlist, -1, |s, x| s.m(x, T::DML, &["SELECT"]))?;
                if sel.is_some() {
                    self.indent += 1;
                    let first = t.at(tlist, 0)?;
                    let nl = self.nl(t, -6);
                    t.insert_after_tok(tlist, first, nl, true)?;
                    self.process_default(t, tlist, d, depth)?;
                    self.indent -= 1;
                    let last = t.at(tlist, -1)?;
                    let nl = self.nl(t, 1);
                    t.insert_before_tok(tlist, last, nl)?;
                }
                Ok(())
            }
            Class::IdentifierList => {
                let mut ids = t.get_identifiers(tlist);
                if ids.is_empty() {
                    return Err(Fail::Python);
                }
                ids.remove(0);
                for tok in ids {
                    let nl = self.nl(t, 1);
                    t.insert_before_tok(tlist, tok, nl)?;
                }
                self.process_default(t, tlist, d, depth)
            }
            Class::Case => self.process_case(t, tlist),
            _ => self.process_default(t, tlist, d, depth),
        }
    }

    fn process_case(&mut self, t: &mut Tree<'_>, tlist: Id) -> R<()> {
        let offset_ = 10i64;
        let mut cases = t.get_cases(tlist, true);
        let end = t.next_by(tlist, -1, |s, x| s.m(x, T::KEYWORD, &["END"]))?;
        if let Some((_, e)) = end {
            cases.push((None, vec![e]));
        }
        let mut widths = Vec::with_capacity(cases.len());
        for (cond, _) in &cases {
            let w = match cond {
                Some(c) if !c.is_empty() => {
                    let mut n = 0usize;
                    for (k, &x) in c.iter().enumerate() {
                        if k > 0 {
                            n += 1;
                        }
                        n += pystr::py_len(&t.text(x));
                    }
                    n
                }
                _ => 0,
            };
            widths.push(i64::try_from(w).unwrap_or(0));
        }
        let max_w = widths.iter().copied().max().unwrap_or(0);
        for (i, (cond, value)) in cases.iter().enumerate() {
            let stmt = match cond {
                Some(c) if !c.is_empty() => c[0],
                _ => *value.first().ok_or(Fail::Python)?,
            };
            if i > 0 {
                let len = i64::try_from(pystr::py_len(&t.text(stmt))).unwrap_or(0);
                let nl = self.nl(t, offset_ - len);
                t.insert_before_tok(tlist, stmt, nl)?;
            }
            if let Some(c) = cond {
                if let Some(&last) = c.last() {
                    let n = usize::try_from(max_w - widths[i]).unwrap_or(0);
                    let ws: String = std::iter::repeat_n(self.ch, n).collect();
                    let tok = t.token(T::WHITESPACE, Str::owned(ws));
                    t.insert_after_tok(tlist, last, tok, true)?;
                }
            }
        }
        Ok(())
    }

    fn next_token(
        t: &Tree<'_>,
        tlist: Id,
        idx: isize,
        d: Depth,
        guard: usize,
    ) -> R<Option<(usize, Id)>> {
        d.check(guard)?;
        let p = patterns(t.v);
        let found = t.next_by(tlist, idx, |s, x| {
            s.m_regex(x, T::KEYWORD, &p.aligned_ci, &p.aligned_cs)
        })?;
        if let Some((tidx, tok)) = found {
            if t.norm(tok) == "BETWEEN" {
                let again = Self::next_token(t, tlist, ix(tidx), d, guard + 1)?;
                if let Some((t2, tok2)) = again {
                    if t.norm(tok2) == "AND" {
                        return Self::next_token(t, tlist, ix(t2), d, guard + 1);
                    }
                }
                return Ok(again);
            }
        }
        Ok(found)
    }

    fn split_kwds(&mut self, t: &mut Tree<'_>, tlist: Id, d: Depth) -> R<()> {
        let p = patterns(t.v);
        let mut cur = Self::next_token(t, tlist, -1, d, 0)?;
        while let Some((tidx, token)) = cur {
            let special = t.m_regex(token, T::KEYWORD, &p.join_ci, &p.join_cs)
                || t.m_regex(token, T::KEYWORD, &p.by_ci, &p.by_cs);
            let indent_len = if special {
                let v = t.val(token);
                let w = v
                    .split(crate::text::is_py_whitespace)
                    .find(|s| !s.is_empty())
                    .ok_or(Fail::Python)?;
                pystr::py_len(w)
            } else {
                pystr::py_len(&t.text(token))
            };
            let nl = self.nl(t, -i64::try_from(indent_len).unwrap_or(0));
            t.insert_before_tok(tlist, token, nl)?;
            let tidx = ix(tidx) + 1;
            cur = Self::next_token(t, tlist, tidx, d, 0)?;
        }
        Ok(())
    }

    fn process_default(&mut self, t: &mut Tree<'_>, tlist: Id, d: Depth, depth: usize) -> R<()> {
        self.split_kwds(t, tlist, d)?;
        let p = patterns(t.v);
        let mut i = 0;
        while let Some((next, sub)) = t.sublist_at(tlist, i) {
            if let Some(sg) = sub {
                let idx = t.token_index(tlist, sg, 0)?;
                let prev = t.token_prev(tlist, Some(ix(idx)), true, false)?;
                let off = match prev {
                    Some((_, pv)) if t.m_regex(pv, T::KEYWORD, &p.by_ci, &p.by_cs) => 3,
                    _ => 0,
                };
                self.offset += off;
                self.process_node(t, sg, d, depth + 1)?;
                self.offset -= off;
            }
            i = next;
        }
        Ok(())
    }
}

/// `tokens[i]` in Python indexing, for the callers here.
#[allow(dead_code)]
fn nth(v: &[Id], i: isize) -> R<Id> {
    py_index(v.len(), i).map(|j| v[j]).ok_or(Fail::Python)
}
