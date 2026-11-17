// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! `sqlparse.engine.grouping`, ported line by line: the passes of `group()` in order, the
//! `_group`/`_group_matching` walks over a snapshot of the token list with an index offset
//! into the live list, `@recurse`, `_is_delimiter`, and the DoS limits
//! (`MAX_GROUPING_DEPTH`, `MAX_GROUPING_TOKENS`).

use super::tree::{Class, Fail, Id, R, Tree};
use super::ttype::{TT, builtin as T};

/// The DoS limits (`None`: disabled).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Limits {
    /// `MAX_GROUPING_DEPTH`.
    pub depth: Option<i64>,
    /// `MAX_GROUPING_TOKENS`.
    pub tokens: Option<i64>,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            depth: Some(100),
            tokens: Some(10_000),
        }
    }
}

/// The deepest native recursion allowed whatever the limits say, so the Rust stack stays
/// small; sqlparse's own Python recursion overflows well before this
/// (`SQLParseError('Maximum recursion depth exceeded')` from `FilterStack.run`).
pub const HARD_DEPTH: usize = 2_000;

/// The message sqlparse raises when its recursion overflows.
pub const RECURSION_MESSAGE: &str = "Maximum recursion depth exceeded";

fn recursion() -> Fail {
    Fail::SqlParse(RECURSION_MESSAGE.to_owned())
}

/// The grouping context: the arena, the limits and a stack budget.
#[derive(Debug)]
pub struct Grouper<'t, 's> {
    /// The arena.
    pub t: &'t mut Tree<'s>,
    limits: Limits,
    stack_base: usize,
    stack_budget: usize,
}

#[allow(clippy::inline_always)]
#[inline(always)]
fn stack_position() -> usize {
    let marker = std::hint::black_box(0u8);
    std::ptr::from_ref(&marker).addr()
}

/// Native stack the grouping and the filters may use before they stop with
/// [`RECURSION_MESSAGE`] (musl threads get 128 KiB).
#[cfg(target_env = "musl")]
pub const STACK_BUDGET: usize = 80 << 10;
/// Native stack the grouping and the filters may use before they stop with
/// [`RECURSION_MESSAGE`] (musl threads get 128 KiB).
#[cfg(not(target_env = "musl"))]
pub const STACK_BUDGET: usize = 160 << 10;

/// `_DELIMITED_GROUPS`.
const DELIMITED: [Class; 6] = [
    Class::Parenthesis,
    Class::SquareBrackets,
    Class::Case,
    Class::If,
    Class::For,
    Class::Begin,
];

/// `T_NUMERICAL + T_STRING + T_NAME`.
const T_NSN: [TT; 8] = [
    T::NUMBER,
    T::INTEGER,
    T::FLOAT,
    T::STRING,
    T::STRING_SINGLE,
    T::STRING_SYMBOL,
    T::NAME,
    T::PLACEHOLDER,
];

/// The `_group` passes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pass {
    Typecasts,
    Tzcasts,
    TypedLiteral,
    TypedLiteralExtend,
    Period,
    As,
    Assignment,
    Comparison,
    Arrays,
    Operator,
    IdentifierList,
}

impl Pass {
    fn cls(self) -> Class {
        match self {
            Pass::Typecasts | Pass::Tzcasts | Pass::Period | Pass::As | Pass::Arrays => {
                Class::Identifier
            }
            Pass::TypedLiteral | Pass::TypedLiteralExtend => Class::TypedLiteral,
            Pass::Assignment => Class::Assignment,
            Pass::Comparison => Class::Comparison,
            Pass::Operator => Class::Operation,
            Pass::IdentifierList => Class::IdentifierList,
        }
    }

    fn extend(self) -> bool {
        !matches!(self, Pass::TypedLiteral | Pass::Comparison | Pass::Operator)
    }

    fn recurse(self) -> bool {
        self != Pass::Arrays
    }
}

impl<'t, 's> Grouper<'t, 's> {
    /// A grouper over `t`.
    pub fn new(t: &'t mut Tree<'s>, limits: Limits) -> Grouper<'t, 's> {
        Grouper {
            t,
            limits,
            stack_base: stack_position(),
            stack_budget: STACK_BUDGET,
        }
    }

    fn check_stack(&self, depth: usize) -> R<()> {
        if depth > HARD_DEPTH || stack_position().abs_diff(self.stack_base) > self.stack_budget {
            return Err(recursion());
        }
        Ok(())
    }

    /// The two limit checks at the top of `_group` and `_group_matching`.
    fn check_limits(&self, tlist: Id, depth: usize) -> R<()> {
        if let Some(d) = self.limits.depth {
            if i64::try_from(depth).unwrap_or(i64::MAX) > d {
                return Err(Fail::SqlParse(format!(
                    "Maximum grouping depth exceeded ({d})."
                )));
            }
        }
        if let Some(n) = self.limits.tokens {
            if i64::try_from(self.t.len(tlist)).unwrap_or(i64::MAX) > n {
                return Err(Fail::SqlParse(format!(
                    "Maximum number of tokens exceeded ({n})."
                )));
            }
        }
        self.check_stack(depth)
    }

    /// `group(stmt)`.
    ///
    /// # Errors
    /// `SQLParseError` for the limits, [`Fail::Python`] where sqlparse's code would raise.
    pub fn group(&mut self, stmt: Id) -> R<()> {
        self.recurse(stmt, &[Class::Comment], 0, Self::group_comments)?;
        self.matching(stmt, Class::SquareBrackets, 0)?;
        self.matching(stmt, Class::Parenthesis, 0)?;
        self.matching(stmt, Class::Case, 0)?;
        self.matching(stmt, Class::If, 0)?;
        self.matching(stmt, Class::For, 0)?;
        self.matching(stmt, Class::Begin, 0)?;
        self.recurse(stmt, &[Class::Over], 0, Self::group_over)?;
        self.recurse(stmt, &[Class::Function], 0, Self::group_functions)?;
        self.recurse(stmt, &[Class::Where], 0, Self::group_where)?;
        self.pass(stmt, Pass::Period, 0)?;
        self.pass(stmt, Pass::Arrays, 0)?;
        self.recurse(stmt, &[Class::Identifier], 0, Self::group_identifier)?;
        self.recurse(stmt, &[Class::Identifier], 0, Self::group_order)?;
        self.pass(stmt, Pass::Typecasts, 0)?;
        self.pass(stmt, Pass::Tzcasts, 0)?;
        self.pass(stmt, Pass::TypedLiteral, 0)?;
        self.pass(stmt, Pass::TypedLiteralExtend, 0)?;
        self.pass(stmt, Pass::Operator, 0)?;
        self.pass(stmt, Pass::Comparison, 0)?;
        self.pass(stmt, Pass::As, 0)?;
        self.recurse(stmt, &[], 0, Self::group_aliased)?;
        self.pass(stmt, Pass::Assignment, 0)?;
        self.recurse(stmt, &[], 0, Self::align_comments)?;
        self.pass(stmt, Pass::IdentifierList, 0)?;
        self.group_values(stmt)?;
        Ok(())
    }

    /// `@recurse(*skip)(f)(tlist)`: `f` on every sublist not of a `skip` class (depth first,
    /// children before parents), then on `tlist`.
    fn recurse(
        &mut self,
        tlist: Id,
        skip: &[Class],
        depth: usize,
        f: fn(&mut Self, Id) -> R<()>,
    ) -> R<()> {
        self.check_stack(depth)?;
        // `for sgroup in tlist.get_sublists()`: a live iteration over tlist.tokens
        let mut i = 0;
        while let Some((next, sub)) = self.t.sublist_at(tlist, i) {
            if let Some(sg) = sub {
                if !self.t.isa_any(sg, skip) {
                    self.recurse(sg, skip, depth + 1, f)?;
                }
            }
            i = next;
        }
        f(self, tlist)
    }

    // ------------------------------------------------------------------
    // _group_matching
    // ------------------------------------------------------------------

    fn open_close(&self, tok: Id, cls: Class) -> (bool, bool) {
        let t = &*self.t;
        match cls {
            Class::SquareBrackets => (
                t.m(tok, T::PUNCTUATION, &["["]),
                t.m(tok, T::PUNCTUATION, &["]"]),
            ),
            Class::Parenthesis => (
                t.m(tok, T::PUNCTUATION, &["("]),
                t.m(tok, T::PUNCTUATION, &[")"]),
            ),
            Class::Case => (
                t.m(tok, T::KEYWORD, &["CASE"]),
                t.m(tok, T::KEYWORD, &["END"]),
            ),
            Class::If => (
                t.m(tok, T::KEYWORD, &["IF"]),
                t.m(tok, T::KEYWORD, &["END IF"]),
            ),
            Class::For => (
                t.m(tok, T::KEYWORD, &["FOR", "FOREACH"]),
                t.m(tok, T::KEYWORD, &["END LOOP"]),
            ),
            Class::Begin => (
                t.m(tok, T::KEYWORD, &["BEGIN"]),
                t.m(tok, T::KEYWORD, &["END"]),
            ),
            _ => (false, false),
        }
    }

    /// `_group_matching(tlist, cls, depth)`.
    fn matching(&mut self, tlist: Id, cls: Class, depth: usize) -> R<()> {
        self.check_limits(tlist, depth)?;
        let mut opens: Vec<isize> = Vec::new();
        let mut tidx_offset: isize = 0;
        let token_list: Vec<Id> = self.t.kids(tlist).to_vec();
        for (idx, &token) in token_list.iter().enumerate() {
            let tidx = isize::try_from(idx).unwrap_or(isize::MAX) - tidx_offset;
            let n = self.t.n(token);
            if n.is_whitespace {
                continue;
            }
            if n.is_group && !self.t.isa(token, cls) {
                self.matching(token, cls, depth + 1)?;
                continue;
            }
            let (open, close) = self.open_close(token, cls);
            if open {
                opens.push(tidx);
            } else if close {
                let Some(open_idx) = opens.pop() else {
                    continue;
                };
                let close_idx = tidx;
                self.t
                    .group_tokens(tlist, cls, open_idx, close_idx, true, false)?;
                tidx_offset += close_idx - open_idx;
            }
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // _group and its passes
    // ------------------------------------------------------------------

    fn p_match(&self, p: Pass, tok: Id) -> bool {
        let t = &*self.t;
        match p {
            Pass::Typecasts => t.m(tok, T::PUNCTUATION, &["::"]),
            Pass::Tzcasts => t.is(tok, T::TZCAST),
            Pass::TypedLiteral => t.m0(tok, T::BUILTIN) || t.m(tok, T::KEYWORD, &["TIMESTAMP"]),
            Pass::TypedLiteralExtend => t.isa(tok, Class::TypedLiteral),
            Pass::Period => {
                t.m(tok, T::PUNCTUATION, &["."])
                    || t.m(tok, T::OPERATOR, &["->"])
                    || t.m(tok, T::OPERATOR, &["->>"])
            }
            Pass::As => {
                let n = t.n(tok);
                n.is_keyword && t.norm(tok) == "AS"
            }
            Pass::Assignment => t.m(tok, T::ASSIGNMENT, &[":="]),
            Pass::Comparison => t.is(tok, T::COMPARISON),
            Pass::Arrays => t.isa(tok, Class::SquareBrackets),
            Pass::Operator => t.is_any(tok, &[T::OPERATOR, T::WILDCARD]),
            Pass::IdentifierList => t.m(tok, T::PUNCTUATION, &[","]),
        }
    }

    fn p_valid_prev(&self, p: Pass, tok: Id) -> bool {
        let t = &*self.t;
        match p {
            Pass::Typecasts | Pass::Tzcasts | Pass::TypedLiteral | Pass::TypedLiteralExtend => true,
            Pass::Period => {
                t.isa_any(tok, &[Class::SquareBrackets, Class::Identifier])
                    || t.is_any(tok, &[T::NAME, T::STRING_SYMBOL])
            }
            Pass::As => t.norm(tok) == "NULL" || !t.n(tok).is_keyword,
            Pass::Arrays => {
                t.isa_any(
                    tok,
                    &[Class::SquareBrackets, Class::Identifier, Class::Function],
                ) || t.is_any(tok, &[T::NAME, T::STRING_SYMBOL])
            }
            _ => self.p_valid(p, Some(tok)),
        }
    }

    fn p_valid_next(&self, p: Pass, tok: Option<Id>) -> bool {
        let t = &*self.t;
        match p {
            Pass::Typecasts => tok.is_some(),
            Pass::Tzcasts => tok.is_some_and(|x| {
                t.n(x).is_whitespace || t.m(x, T::KEYWORD, &["AS"]) || t.m0(x, T::STRING_SINGLE)
            }),
            Pass::TypedLiteral => tok.is_some_and(|x| t.m0(x, T::STRING_SINGLE)),
            Pass::TypedLiteralExtend => tok.is_some_and(|x| {
                t.m(
                    x,
                    T::KEYWORD,
                    &["DAY", "HOUR", "MINUTE", "MONTH", "SECOND", "YEAR"],
                )
            }),
            Pass::Period | Pass::Arrays => true,
            Pass::As => tok.is_some_and(|x| !t.is_any(x, &[T::DML, T::DDL, T::CTE])),
            _ => self.p_valid(p, tok),
        }
    }

    /// The shared `valid` of the passes that use one function for both sides.
    fn p_valid(&self, p: Pass, tok: Option<Id>) -> bool {
        let t = &*self.t;
        let Some(x) = tok else { return false };
        match p {
            Pass::Assignment => !t.is_any(x, &[T::KEYWORD]),
            Pass::Comparison => {
                t.is_any(x, &T_NSN)
                    || t.isa_any(
                        x,
                        &[
                            Class::Parenthesis,
                            Class::Function,
                            Class::Identifier,
                            Class::Operation,
                            Class::TypedLiteral,
                        ],
                    )
                    || (t.n(x).is_keyword && t.norm(x) == "NULL")
            }
            Pass::Operator => {
                t.isa_any(
                    x,
                    &[
                        Class::SquareBrackets,
                        Class::Parenthesis,
                        Class::Function,
                        Class::Identifier,
                        Class::Operation,
                        Class::TypedLiteral,
                    ],
                ) || t.is_any(x, &T_NSN)
                    || t.m(
                        x,
                        T::KEYWORD,
                        &["CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP"],
                    )
            }
            Pass::IdentifierList => {
                t.isa_any(
                    x,
                    &[
                        Class::Function,
                        Class::Case,
                        Class::Identifier,
                        Class::Comparison,
                        Class::IdentifierList,
                        Class::Operation,
                    ],
                ) || t.m(x, T::KEYWORD, &["null", "role"])
                    || t.is_any(x, &T_NSN)
                    || t.is_any(x, &[T::KEYWORD, T::COMMENT, T::WILDCARD])
            }
            _ => true,
        }
    }

    /// `post(tlist, pidx, tidx, nidx)`: the range to group.
    fn p_post(
        &mut self,
        p: Pass,
        tlist: Id,
        pidx: Option<isize>,
        tidx: isize,
        nidx: Option<usize>,
    ) -> R<(isize, isize)> {
        let nidx_i = nidx.map(|n| isize::try_from(n).unwrap_or(isize::MAX));
        let pidx = pidx.ok_or(Fail::Python)?;
        match p {
            Pass::TypedLiteral | Pass::TypedLiteralExtend => {
                Ok((tidx, nidx_i.ok_or(Fail::Python)?))
            }
            Pass::Period => {
                let next = match nidx_i {
                    Some(n) => Some(self.t.at(tlist, n)?),
                    None => None,
                };
                let t = &*self.t;
                let valid = next.is_some_and(|x| {
                    t.isa_any(x, &[Class::SquareBrackets, Class::Function])
                        || t.is_any(
                            x,
                            &[T::NAME, T::STRING_SYMBOL, T::WILDCARD, T::STRING_SINGLE],
                        )
                });
                Ok(if valid {
                    (pidx, nidx_i.ok_or(Fail::Python)?)
                } else {
                    (pidx, tidx)
                })
            }
            Pass::Arrays => Ok((pidx, tidx)),
            Pass::Operator => {
                let tok = self.t.at(tlist, tidx)?;
                self.t.n_mut(tok).ttype = Some(T::OPERATOR);
                Ok((pidx, nidx_i.ok_or(Fail::Python)?))
            }
            Pass::Assignment => {
                let n = nidx_i.ok_or(Fail::Python)?;
                let semi = self
                    .t
                    .next_by(tlist, n, |s, x| s.m(x, T::PUNCTUATION, &[";"]))?;
                // `nidx = snidx or nidx`
                let n = match semi {
                    Some((s, _)) if s != 0 => isize::try_from(s).unwrap_or(isize::MAX),
                    _ => n,
                };
                Ok((pidx, n))
            }
            _ => Ok((pidx, nidx_i.ok_or(Fail::Python)?)),
        }
    }

    /// `_is_delimiter(tlist, token)`.
    fn is_delimiter(&self, tlist: Id, tok: Option<Id>) -> R<bool> {
        let Some(tok) = tok else { return Ok(false) };
        if !self.t.isa_any(tlist, &DELIMITED) {
            return Ok(false);
        }
        let first = self.t.at(tlist, 0)?;
        let last = self.t.at(tlist, -1)?;
        Ok(tok == first || tok == last)
    }

    /// `_group(tlist, cls, match, valid_prev, valid_next, post, extend, recurse, depth)`.
    fn pass(&mut self, tlist: Id, p: Pass, depth: usize) -> R<()> {
        self.check_limits(tlist, depth)?;
        let cls = p.cls();
        let mut tidx_offset: isize = 0;
        let mut pidx: Option<isize> = None;
        let mut prev: Option<Id> = None;
        let token_list: Vec<Id> = self.t.kids(tlist).to_vec();
        for (idx, &token) in token_list.iter().enumerate() {
            let tidx = isize::try_from(idx).unwrap_or(isize::MAX) - tidx_offset;
            if tidx < 0 {
                continue;
            }
            if self.t.n(token).is_whitespace {
                continue;
            }
            if p.recurse() && self.t.n(token).is_group && !self.t.isa(token, cls) {
                self.pass(token, p, depth + 1)?;
            }
            if self.p_match(p, token) {
                let next = self.t.token_next(tlist, Some(tidx), true, false)?;
                let (nidx, next_tok) = match next {
                    Some((i, t)) => (Some(i), Some(t)),
                    None => (None, None),
                };
                if let Some(pv) = prev {
                    if self.p_valid_prev(p, pv)
                        && self.p_valid_next(p, next_tok)
                        && !self.is_delimiter(tlist, Some(pv))?
                        && !self.is_delimiter(tlist, next_tok)?
                    {
                        let (from_idx, to_idx) = self.p_post(p, tlist, pidx, tidx, nidx)?;
                        let grp =
                            self.t
                                .group_tokens(tlist, cls, from_idx, to_idx, true, p.extend())?;
                        tidx_offset += to_idx - from_idx;
                        pidx = Some(from_idx);
                        prev = Some(grp);
                        continue;
                    }
                }
            }
            pidx = Some(tidx);
            prev = Some(token);
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // The other passes
    // ------------------------------------------------------------------

    fn idx(i: usize) -> isize {
        isize::try_from(i).unwrap_or(isize::MAX)
    }

    fn group_comments(&mut self, tlist: Id) -> R<()> {
        let mut cur = self.t.next_by(tlist, -1, |s, x| s.within(x, T::COMMENT))?;
        while let Some((tidx, _)) = cur {
            // token_not_matching(lambda tk: imt(tk, t=T.Comment) or tk.is_newline, idx=tidx)
            let end = self
                .t
                .token_matching(tlist, Self::idx(tidx), false, |s, x| {
                    !(s.within(x, T::COMMENT) || s.n(x).is_newline)
                })?;
            let Some((eidx, _)) = end else { break };
            let prev = self
                .t
                .token_prev(tlist, Some(Self::idx(eidx)), false, false)?;
            let eidx = prev.map(|p| Self::idx(p.0)).ok_or(Fail::Python)?;
            self.t
                .group_tokens(tlist, Class::Comment, Self::idx(tidx), eidx, true, false)?;
            cur = self
                .t
                .next_by(tlist, Self::idx(tidx), |s, x| s.within(x, T::COMMENT))?;
        }
        Ok(())
    }

    fn group_identifier(&mut self, tlist: Id) -> R<()> {
        let pred = |s: &Tree<'_>, x: Id| s.is_any(x, &[T::STRING_SYMBOL, T::NAME]);
        let mut cur = self.t.next_by(tlist, -1, pred)?;
        while let Some((tidx, _)) = cur {
            self.t.group_tokens(
                tlist,
                Class::Identifier,
                Self::idx(tidx),
                Self::idx(tidx),
                true,
                false,
            )?;
            cur = self.t.next_by(tlist, Self::idx(tidx), pred)?;
        }
        Ok(())
    }

    fn group_over(&mut self, tlist: Id) -> R<()> {
        let pred = |s: &Tree<'_>, x: Id| s.m(x, T::KEYWORD, &["OVER"]);
        let mut cur = self.t.next_by(tlist, -1, pred)?;
        while let Some((tidx, _)) = cur {
            let next = self
                .t
                .token_next(tlist, Some(Self::idx(tidx)), true, false)?;
            if let Some((nidx, nt)) = next {
                if self.t.isa(nt, Class::Parenthesis) || self.t.within(nt, T::NAME) {
                    self.t.group_tokens(
                        tlist,
                        Class::Over,
                        Self::idx(tidx),
                        Self::idx(nidx),
                        true,
                        false,
                    )?;
                }
            }
            cur = self.t.next_by(tlist, Self::idx(tidx), pred)?;
        }
        Ok(())
    }

    fn group_where(&mut self, tlist: Id) -> R<()> {
        let open = |s: &Tree<'_>, x: Id| s.m(x, T::KEYWORD, &["WHERE"]);
        let mut cur = self.t.next_by(tlist, -1, open)?;
        while let Some((tidx, _)) = cur {
            let close = self.t.next_by(tlist, Self::idx(tidx), |s, x| {
                s.m(
                    x,
                    T::KEYWORD,
                    &[
                        "ORDER BY",
                        "GROUP BY",
                        "LIMIT",
                        "UNION",
                        "UNION ALL",
                        "EXCEPT",
                        "INTERSECT",
                        "HAVING",
                        "RETURNING",
                        "INTO",
                    ],
                )
            })?;
            let end = match close {
                None => {
                    // tlist._groupable_tokens[-1]
                    let kids = self.t.kids(tlist);
                    let groupable: &[Id] = if self
                        .t
                        .isa_any(tlist, &[Class::Parenthesis, Class::SquareBrackets])
                    {
                        if kids.len() >= 2 {
                            &kids[1..kids.len() - 1]
                        } else {
                            &[]
                        }
                    } else {
                        kids
                    };
                    *groupable.last().ok_or(Fail::Python)?
                }
                Some((eidx, _)) => self.t.at(tlist, Self::idx(eidx) - 1)?,
            };
            let eidx = self.t.token_index(tlist, end, 0)?;
            self.t.group_tokens(
                tlist,
                Class::Where,
                Self::idx(tidx),
                Self::idx(eidx),
                true,
                false,
            )?;
            cur = self.t.next_by(tlist, Self::idx(tidx), open)?;
        }
        Ok(())
    }

    fn group_aliased(&mut self, tlist: Id) -> R<()> {
        let pred = |s: &Tree<'_>, x: Id| {
            s.isa_any(
                x,
                &[
                    Class::Parenthesis,
                    Class::Function,
                    Class::Case,
                    Class::Identifier,
                    Class::Operation,
                    Class::Comparison,
                ],
            ) || s.within(x, T::NUMBER)
        };
        let mut cur = self.t.next_by(tlist, -1, pred)?;
        while let Some((tidx, _)) = cur {
            let next = self
                .t
                .token_next(tlist, Some(Self::idx(tidx)), true, false)?;
            if let Some((nidx, nt)) = next {
                if self.t.isa(nt, Class::Identifier) {
                    self.t.group_tokens(
                        tlist,
                        Class::Identifier,
                        Self::idx(tidx),
                        Self::idx(nidx),
                        true,
                        true,
                    )?;
                }
            }
            cur = self.t.next_by(tlist, Self::idx(tidx), pred)?;
        }
        Ok(())
    }

    fn group_functions(&mut self, tlist: Id) -> R<()> {
        let (mut has_create, mut has_table, mut has_as) = (false, false, false);
        let v = self.t.v;
        for &tok in self.t.kids(tlist) {
            let val = self.t.val(tok);
            if super::pystr::upper_eq(val, "CREATE", v) {
                has_create = true;
            }
            if super::pystr::upper_eq(val, "TABLE", v) {
                has_table = true;
            }
            if super::pystr::upper_eq(val, "AS", v) {
                has_as = true;
            }
        }
        if has_create && has_table && !has_as {
            return Ok(());
        }
        let pred = |s: &Tree<'_>, x: Id| s.within(x, T::NAME);
        let mut cur = self.t.next_by(tlist, -1, pred)?;
        while let Some((tidx, _)) = cur {
            let next = self
                .t
                .token_next(tlist, Some(Self::idx(tidx)), true, false)?;
            if let Some((nidx, nt)) = next {
                if self.t.isa(nt, Class::Parenthesis) {
                    let over = self
                        .t
                        .token_next(tlist, Some(Self::idx(nidx)), true, false)?;
                    let eidx = match over {
                        Some((oidx, o)) if self.t.isa(o, Class::Over) => oidx,
                        _ => nidx,
                    };
                    self.t.group_tokens(
                        tlist,
                        Class::Function,
                        Self::idx(tidx),
                        Self::idx(eidx),
                        true,
                        false,
                    )?;
                }
            }
            cur = self.t.next_by(tlist, Self::idx(tidx), pred)?;
        }
        Ok(())
    }

    fn group_order(&mut self, tlist: Id) -> R<()> {
        let pred = |s: &Tree<'_>, x: Id| s.within(x, T::ORDER);
        let mut cur = self.t.next_by(tlist, -1, pred)?;
        while let Some((tidx, _)) = cur {
            let mut tidx = Self::idx(tidx);
            let prev = self.t.token_prev(tlist, Some(tidx), true, false)?;
            if let Some((pidx, pt)) = prev {
                if self.t.isa(pt, Class::Identifier) || self.t.within(pt, T::NUMBER) {
                    self.t.group_tokens(
                        tlist,
                        Class::Identifier,
                        Self::idx(pidx),
                        tidx,
                        true,
                        false,
                    )?;
                    tidx = Self::idx(pidx);
                }
            }
            cur = self.t.next_by(tlist, tidx, pred)?;
        }
        Ok(())
    }

    fn align_comments(&mut self, tlist: Id) -> R<()> {
        let pred = |s: &Tree<'_>, x: Id| s.isa(x, Class::Comment);
        let mut cur = self.t.next_by(tlist, -1, pred)?;
        while let Some((tidx, _)) = cur {
            let mut tidx = Self::idx(tidx);
            let prev = self.t.token_prev(tlist, Some(tidx), true, false)?;
            if let Some((pidx, pt)) = prev {
                if self.t.isa(pt, Class::TokenList) {
                    self.t.group_tokens(
                        tlist,
                        Class::TokenList,
                        Self::idx(pidx),
                        tidx,
                        true,
                        true,
                    )?;
                    tidx = Self::idx(pidx);
                }
            }
            cur = self.t.next_by(tlist, tidx, pred)?;
        }
        Ok(())
    }

    fn group_values(&mut self, tlist: Id) -> R<()> {
        let first = self
            .t
            .next_by(tlist, -1, |s, x| s.m(x, T::KEYWORD, &["VALUES"]))?;
        let Some((start_idx, _)) = first else {
            return Ok(());
        };
        let mut end_idx: Option<usize> = None;
        let mut cur = Some((start_idx, self.t.at(tlist, Self::idx(start_idx))?));
        while let Some((tidx, tok)) = cur {
            if self.t.isa(tok, Class::Parenthesis) {
                end_idx = Some(tidx);
            }
            cur = self
                .t
                .token_next(tlist, Some(Self::idx(tidx)), true, false)?;
        }
        if let Some(e) = end_idx {
            self.t.group_tokens(
                tlist,
                Class::Values,
                Self::idx(start_idx),
                Self::idx(e),
                true,
                true,
            )?;
        }
        Ok(())
    }
}
