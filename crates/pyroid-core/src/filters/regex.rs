// Derived from spaCy 3.8.18 (MIT), © ExplosionAI GmbH; modified.
//! The subset of Python's `re` that spaCy's English tokenizer patterns use,
//! with `re`'s semantics.
//!
//! The patterns are not transcribed: `generate_tables.py` parses spaCy's
//! compiled regexes with CPython's own parser (`re._parser`) and emits the
//! tree as [`Node`]s. Two matchers walk it:
//!
//! * [`Matcher`], a backtracking matcher with `re`'s leftmost-first,
//!   greedy-first priorities, for the prefix / suffix / infix searches. Their
//!   only unbounded repeats have one-character bodies, which run as loops, so
//!   recursion depth is bounded by the pattern, never by the input.
//! * [`Nfa`], a Thompson simulation for the URL pattern, where only "does it
//!   match" matters. It runs in linear time and constant stack for any input
//!   (the URL pattern nests unbounded repeats, which a backtracker would turn
//!   into one stack frame per host label).
//!
//! Positions are byte offsets into the haystack (always on char boundaries);
//! `re` counts code points, which gives the same matches.

use std::cmp::Ordering;

use super::unicode;
use crate::ftfy::UnicodeVersion;

/// `re`'s character categories (`\d`, `\w`, `\s` and their negations).
#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // any category the parsed patterns may use
pub(crate) enum Cat {
    Digit,
    NotDigit,
    Word,
    NotWord,
    Space,
    NotSpace,
}

impl Cat {
    #[inline]
    fn matches(self, c: char, v: UnicodeVersion) -> bool {
        match self {
            Cat::Digit => unicode::is_digit(c, v),
            Cat::NotDigit => !unicode::is_digit(c, v),
            Cat::Word => unicode::is_word(c, v),
            Cat::NotWord => !unicode::is_word(c, v),
            Cat::Space => unicode::is_space(c),
            Cat::NotSpace => !unicode::is_space(c),
        }
    }
}

/// A character class (`[...]`): ASCII membership precomputed by `re`
/// itself; above ASCII, explicit ranges, then categories, then negation.
#[derive(Debug)]
pub(crate) struct Class {
    pub negate: bool,
    pub ascii: [u64; 2],
    pub ranges: &'static [(u32, u32)],
    pub cats: &'static [Cat],
}

impl Class {
    #[inline]
    pub(crate) fn contains(&self, c: char, v: UnicodeVersion) -> bool {
        let u = c as u32;
        if u < 128 {
            return (self.ascii[(u >> 6) as usize] >> (u & 63)) & 1 == 1;
        }
        let mut hit = self
            .ranges
            .binary_search_by(|&(lo, hi)| {
                if hi < u {
                    Ordering::Less
                } else if lo > u {
                    Ordering::Greater
                } else {
                    Ordering::Equal
                }
            })
            .is_ok();
        if !hit {
            hit = self.cats.iter().any(|cat| cat.matches(c, v));
        }
        hit != self.negate
    }
}

/// One node of a parsed `re` pattern.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Node {
    /// One literal character.
    Lit(char),
    /// One character of a class.
    In(&'static Class),
    /// `^` (no MULTILINE): the start of the haystack.
    Bol,
    /// `$` (no MULTILINE): the end, or just before a final `\n`.
    Eol,
    /// `a|b|...`, tried in order.
    Branch(&'static [&'static [Node]]),
    /// A non-capturing group (CPython's parser usually inlines these).
    #[allow(dead_code)]
    Group(&'static [Node]),
    /// Greedy `{min,max}` (`max == u32::MAX` is unbounded).
    Repeat(u32, u32, &'static [Node]),
    /// `(?=...)`, or `(?!...)` when the flag is set.
    Ahead(bool, &'static [Node]),
    /// `(?<=...)` / `(?<!...)` of a fixed width in characters.
    Behind(bool, u32, &'static [Node]),
}

#[inline]
fn char_at(s: &str, pos: usize) -> Option<char> {
    s.get(pos..).and_then(|t| t.chars().next())
}

#[inline]
fn char_before(s: &str, pos: usize) -> Option<char> {
    s[..pos].chars().next_back()
}

/// Step back `n` characters from `pos`; `None` if the haystack is shorter.
fn back_chars(s: &str, pos: usize, n: u32) -> Option<usize> {
    let mut p = pos;
    for _ in 0..n {
        p -= char_before(s, p)?.len_utf8();
    }
    Some(p)
}

#[inline]
fn at_eol(s: &str, pos: usize) -> bool {
    pos == s.len() || (pos + 1 == s.len() && s.as_bytes()[pos] == b'\n')
}

/// Does `node` consume exactly one character (so a repeat of it can loop)?
#[inline]
fn single_char(nodes: &[Node]) -> Option<Node> {
    match nodes {
        [n @ (Node::Lit(_) | Node::In(_))] => Some(*n),
        _ => None,
    }
}

#[inline]
fn char_matches(node: Node, c: char, v: UnicodeVersion) -> bool {
    match node {
        Node::Lit(l) => l == c,
        Node::In(cls) => cls.contains(c, v),
        _ => unreachable!("not a one-character node"),
    }
}

/// Backtracking matcher over one haystack.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Matcher<'s> {
    pub s: &'s str,
    pub v: UnicodeVersion,
}

impl Matcher<'_> {
    /// `re`'s match of `nodes` at `pos`: the end of the highest-priority
    /// match, if any.
    pub(crate) fn match_at(&self, nodes: &[Node], pos: usize) -> Option<usize> {
        let mut end = None;
        self.seq(nodes, pos, &mut |p| {
            end = Some(p);
            true
        });
        end
    }

    /// Does `nodes` match at `pos` (any end)?
    fn matches_at(&self, nodes: &[Node], pos: usize) -> bool {
        self.seq(nodes, pos, &mut |_| true)
    }

    /// Match `nodes` at `pos`, then call `k` with each end position in
    /// priority order until it returns true.
    fn seq(&self, mut nodes: &[Node], mut pos: usize, k: &mut dyn FnMut(usize) -> bool) -> bool {
        loop {
            let Some((first, rest)) = nodes.split_first() else {
                return k(pos);
            };
            match *first {
                Node::Lit(_) | Node::In(_) => match char_at(self.s, pos) {
                    Some(c) if char_matches(*first, c, self.v) => pos += c.len_utf8(),
                    _ => return false,
                },
                Node::Bol => {
                    if pos != 0 {
                        return false;
                    }
                }
                Node::Eol => {
                    if !at_eol(self.s, pos) {
                        return false;
                    }
                }
                Node::Branch(alts) => {
                    return alts
                        .iter()
                        .any(|alt| self.seq(alt, pos, &mut |p| self.seq(rest, p, k)));
                }
                Node::Group(body) => return self.seq(body, pos, &mut |p| self.seq(rest, p, k)),
                Node::Repeat(min, max, body) => {
                    if let Some(one) = single_char(body) {
                        return self.repeat_one(one, min, max, rest, pos, k);
                    }
                    return self.repeat(body, min, max, 0, rest, pos, k);
                }
                Node::Ahead(negate, body) => {
                    if self.matches_at(body, pos) == negate {
                        return false;
                    }
                }
                Node::Behind(negate, width, body) => {
                    // sre: too little text behind fails `(?<=` and passes `(?<!`.
                    let ok = match back_chars(self.s, pos, width) {
                        Some(start) => self.matches_at(body, start),
                        None => false,
                    };
                    if ok == negate {
                        return false;
                    }
                }
            }
            nodes = rest;
        }
    }

    /// Greedy repeat of a one-character node, as a loop (sre's REPEAT_ONE).
    fn repeat_one(
        &self,
        one: Node,
        min: u32,
        max: u32,
        rest: &[Node],
        pos: usize,
        k: &mut dyn FnMut(usize) -> bool,
    ) -> bool {
        let mut n = 0u32;
        let mut p = pos;
        while n < max {
            match char_at(self.s, p) {
                Some(c) if char_matches(one, c, self.v) => {
                    p += c.len_utf8();
                    n += 1;
                }
                _ => break,
            }
        }
        if n < min {
            return false;
        }
        loop {
            if self.seq(rest, p, k) {
                return true;
            }
            if n == min {
                return false;
            }
            n -= 1;
            p -= char_before(self.s, p).map_or(0, char::len_utf8);
        }
    }

    /// Greedy repeat of a multi-node body, by recursion (bounded: only used
    /// by the patterns' small counted groups).
    #[allow(clippy::too_many_arguments)]
    fn repeat(
        &self,
        body: &[Node],
        min: u32,
        max: u32,
        count: u32,
        rest: &[Node],
        pos: usize,
        k: &mut dyn FnMut(usize) -> bool,
    ) -> bool {
        if count < max
            && self.seq(body, pos, &mut |p| {
                p != pos && self.repeat(body, min, max, count + 1, rest, p, k)
            })
        {
            return true;
        }
        count >= min && self.seq(rest, pos, k)
    }
}

// --------------------------------------------------------------------- NFA

#[derive(Clone, Copy, Debug)]
enum Inst {
    Char(Node),
    Split(usize, usize),
    Jmp(usize),
    Bol,
    Eol,
    Ahead(bool, &'static [Node]),
    Behind(bool, u32, &'static [Node]),
    Match,
}

/// A compiled pattern for boolean `re.match` (anchoring comes from the
/// pattern's own `^` / `$`), simulated breadth-first.
#[derive(Debug)]
pub(crate) struct Nfa {
    prog: Vec<Inst>,
}

impl Nfa {
    pub(crate) fn new(nodes: &'static [Node]) -> Self {
        let mut prog = Vec::new();
        compile(nodes, &mut prog);
        prog.push(Inst::Match);
        Self { prog }
    }

    /// `re.compile(pattern).match(s) is not None`.
    pub(crate) fn is_match(&self, s: &str, v: UnicodeVersion) -> bool {
        let m = Matcher { s, v };
        let n = self.prog.len();
        let mut cur = SparseSet::new(n);
        let mut next = SparseSet::new(n);
        let mut stack = Vec::new();
        if self.closure(0, 0, &m, &mut cur, &mut stack) {
            return true;
        }
        let mut pos = 0;
        for c in s.chars() {
            if cur.is_empty() {
                return false;
            }
            let npos = pos + c.len_utf8();
            next.clear();
            for i in 0..cur.len() {
                let pc = cur.dense[i];
                if let Inst::Char(node) = self.prog[pc] {
                    if char_matches(node, c, v)
                        && self.closure(pc + 1, npos, &m, &mut next, &mut stack)
                    {
                        return true;
                    }
                }
            }
            std::mem::swap(&mut cur, &mut next);
            pos = npos;
        }
        false
    }

    /// Add the epsilon closure of `pc` at `pos`; true if it reaches Match.
    fn closure(
        &self,
        pc: usize,
        pos: usize,
        m: &Matcher<'_>,
        set: &mut SparseSet,
        stack: &mut Vec<usize>,
    ) -> bool {
        stack.push(pc);
        while let Some(pc) = stack.pop() {
            if !set.insert(pc) {
                continue;
            }
            match self.prog[pc] {
                Inst::Char(_) => {}
                Inst::Match => return true,
                Inst::Jmp(a) => stack.push(a),
                Inst::Split(a, b) => {
                    stack.push(b);
                    stack.push(a);
                }
                Inst::Bol => {
                    if pos == 0 {
                        stack.push(pc + 1);
                    }
                }
                Inst::Eol => {
                    if at_eol(m.s, pos) {
                        stack.push(pc + 1);
                    }
                }
                Inst::Ahead(negate, body) => {
                    if m.matches_at(body, pos) != negate {
                        stack.push(pc + 1);
                    }
                }
                Inst::Behind(negate, width, body) => {
                    let ok = back_chars(m.s, pos, width).is_some_and(|st| m.matches_at(body, st));
                    if ok != negate {
                        stack.push(pc + 1);
                    }
                }
            }
        }
        false
    }
}

fn compile(nodes: &'static [Node], prog: &mut Vec<Inst>) {
    for &node in nodes {
        compile_one(node, prog);
    }
}

fn compile_one(node: Node, prog: &mut Vec<Inst>) {
    match node {
        Node::Lit(_) | Node::In(_) => prog.push(Inst::Char(node)),
        Node::Bol => prog.push(Inst::Bol),
        Node::Eol => prog.push(Inst::Eol),
        Node::Ahead(neg, body) => prog.push(Inst::Ahead(neg, body)),
        Node::Behind(neg, w, body) => prog.push(Inst::Behind(neg, w, body)),
        Node::Group(body) => compile(body, prog),
        Node::Branch(alts) => {
            // split L1, next; L1: alt0; jmp end; next: split L2, ...; last alt
            let mut jumps = Vec::new();
            for (i, alt) in alts.iter().enumerate() {
                if i + 1 < alts.len() {
                    let split = prog.len();
                    prog.push(Inst::Split(split + 1, 0));
                    compile(alt, prog);
                    jumps.push(prog.len());
                    prog.push(Inst::Jmp(0));
                    let after = prog.len();
                    prog[split] = Inst::Split(split + 1, after);
                } else {
                    compile(alt, prog);
                }
            }
            let end = prog.len();
            for j in jumps {
                prog[j] = Inst::Jmp(end);
            }
        }
        Node::Repeat(min, max, body) => {
            for _ in 0..min {
                compile(body, prog);
            }
            if max == u32::MAX {
                // L: split body, out; body; jmp L
                let split = prog.len();
                prog.push(Inst::Split(split + 1, 0));
                compile(body, prog);
                prog.push(Inst::Jmp(split));
                let out = prog.len();
                prog[split] = Inst::Split(split + 1, out);
            } else {
                // (body (body (...)?)?)?
                let mut splits = Vec::new();
                for _ in min..max {
                    splits.push(prog.len());
                    prog.push(Inst::Split(0, 0));
                    compile(body, prog);
                }
                let out = prog.len();
                for s in splits {
                    prog[s] = Inst::Split(s + 1, out);
                }
            }
        }
    }
}

/// A set of program counters with O(1) insert / clear (Briggs-Torczon).
#[derive(Debug)]
struct SparseSet {
    dense: Vec<usize>,
    sparse: Vec<usize>,
}

impl SparseSet {
    fn new(n: usize) -> Self {
        Self {
            dense: Vec::with_capacity(n),
            sparse: vec![0; n],
        }
    }

    fn len(&self) -> usize {
        self.dense.len()
    }

    fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    fn clear(&mut self) {
        self.dense.clear();
    }

    fn insert(&mut self, x: usize) -> bool {
        let i = self.sparse[x];
        if i < self.dense.len() && self.dense[i] == x {
            return false;
        }
        self.sparse[x] = self.dense.len();
        self.dense.push(x);
        true
    }
}

// ------------------------------------------------------ alternative dispatch

/// What an alternative's first consumed character can be.
#[derive(Clone, Copy, Debug)]
enum FirstItem {
    Lit(char),
    Class(&'static Class),
}

impl FirstItem {
    #[inline]
    fn contains(self, c: char, v: UnicodeVersion) -> bool {
        match self {
            FirstItem::Lit(l) => l == c,
            FirstItem::Class(cls) => cls.contains(c, v),
        }
    }

    fn may_be_non_ascii(self) -> bool {
        match self {
            FirstItem::Lit(l) => !l.is_ascii(),
            FirstItem::Class(cls) => cls.negate || !cls.ranges.is_empty() || !cls.cats.is_empty(),
        }
    }
}

/// Collect the possible first characters of `nodes` into `out`; true if
/// `nodes` can match without consuming one (then the set is incomplete).
fn first_items(nodes: &[Node], out: &mut Vec<FirstItem>) -> bool {
    for node in nodes {
        match *node {
            Node::Lit(c) => {
                out.push(FirstItem::Lit(c));
                return false;
            }
            Node::In(cls) => {
                out.push(FirstItem::Class(cls));
                return false;
            }
            Node::Bol | Node::Eol | Node::Ahead(..) | Node::Behind(..) => {}
            Node::Branch(alts) => {
                let mut nullable = false;
                for alt in alts {
                    nullable |= first_items(alt, out);
                }
                if !nullable {
                    return false;
                }
            }
            Node::Group(body) => {
                if !first_items(body, out) {
                    return false;
                }
            }
            Node::Repeat(min, _, body) => {
                if !first_items(body, out) && min > 0 {
                    return false;
                }
            }
        }
    }
    true
}

/// A top-level `a|b|...` (optionally after `^`), with each character's
/// candidate alternatives precomputed. [`Alternatives::match_at`] tries only
/// the alternatives that can start with the character at `pos`, in their
/// original order, so the result is `re`'s.
#[derive(Debug)]
pub(crate) struct Alternatives {
    alts: &'static [&'static [Node]],
    ascii: Vec<Vec<u16>>,
    /// Alternatives that may start with a non-ASCII character (`None`: any).
    other: Vec<(u16, Option<Vec<FirstItem>>)>,
}

impl Alternatives {
    /// `nodes` must be `[Branch]` or `[Bol, Branch]` (the `Bol` is then the
    /// caller's business: match only at 0).
    pub(crate) fn new(nodes: &'static [Node]) -> Self {
        let alts: &'static [&'static [Node]] = match nodes {
            [Node::Branch(alts)] | [Node::Bol, Node::Branch(alts)] => alts,
            _ => panic!("not a top-level alternation"),
        };
        let mut ascii = vec![Vec::new(); 128];
        let mut other = Vec::new();
        for (i, alt) in alts.iter().enumerate() {
            let i = u16::try_from(i).expect("fewer than 65536 alternatives");
            let mut items = Vec::new();
            let nullable = first_items(alt, &mut items);
            for (a, list) in ascii.iter_mut().enumerate() {
                let c = char::from(u8::try_from(a).expect("ASCII"));
                // `version` does not matter below 128: ASCII is precomputed.
                if nullable || items.iter().any(|it| it.contains(c, UnicodeVersion::V15_1)) {
                    list.push(i);
                }
            }
            if nullable {
                other.push((i, None));
            } else if items.iter().any(|it| it.may_be_non_ascii()) {
                other.push((i, Some(items)));
            }
        }
        Self { alts, ascii, other }
    }

    /// The end of `re`'s match of the alternation at `pos`, if any.
    #[inline]
    pub(crate) fn match_at(&self, m: &Matcher<'_>, pos: usize) -> Option<usize> {
        let c = char_at(m.s, pos)?;
        if c.is_ascii() {
            for &i in &self.ascii[c as usize] {
                if let Some(end) = m.match_at(self.alts[usize::from(i)], pos) {
                    return Some(end);
                }
            }
        } else {
            for (i, items) in &self.other {
                if items
                    .as_ref()
                    .is_none_or(|it| it.iter().any(|x| x.contains(c, m.v)))
                {
                    if let Some(end) = m.match_at(self.alts[usize::from(*i)], pos) {
                        return Some(end);
                    }
                }
            }
        }
        None
    }
}
