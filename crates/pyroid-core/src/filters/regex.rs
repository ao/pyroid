// Derived from spaCy 3.8.18 (MIT), © ExplosionAI GmbH; modified.
//! The subset of Python's `re` that spaCy's tokenizer patterns use (every
//! language datatrove tokenizes with a rule-based `spacy.blank`), with `re`'s
//! semantics.
//!
//! The patterns are not transcribed: `generate_tables.py` parses spaCy's
//! compiled regexes with CPython's own parser (`re._parser`) and emits the
//! tree as [`Node`]s. Two matchers walk it:
//!
//! * [`Matcher`], a backtracking matcher with `re`'s leftmost-first,
//!   greedy-first priorities, for the prefix / suffix / infix searches. Its
//!   unbounded repeats have one-character bodies, which run as loops, so
//!   recursion depth is bounded by the pattern, never by the input.
//! * [`Nfa`], a Thompson simulation in linear time and constant stack for any
//!   input: boolean `re.match` for the URL and `token_match` patterns, and,
//!   with thread priorities ([`Nfa::match_end`]), the match `re` would find
//!   for the few affix alternatives that repeat a multi-node group without
//!   bound (Greek, Portuguese), which a backtracker would turn into one stack
//!   frame per iteration.
//!
//! Case-insensitive patterns (French `token_match`) arrive with every
//! character already expanded into the set `re.IGNORECASE` matches, per
//! Unicode version where that differs ([`Node::InV`]).
//!
//! Positions are byte offsets into the haystack (always on char boundaries);
//! `re` counts code points, which gives the same matches.

use std::cmp::Ordering;

use super::unicode;
use crate::ftfy::UnicodeVersion;

/// Index of `version` in per-version tables (`UnicodeVersion` order).
pub(crate) fn version_index(version: UnicodeVersion) -> usize {
    match version {
        UnicodeVersion::V14_0 => 0,
        UnicodeVersion::V15_0 => 1,
        UnicodeVersion::V15_1 => 2,
        UnicodeVersion::V16_0 => 3,
        UnicodeVersion::V17_0 => 4,
    }
}

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
    /// One character of the class for the haystack's Unicode version
    /// (`[V14_0, V15_0, V15_1, V16_0, V17_0]`).
    InV(&'static [&'static Class; 5]),
    /// `^` (no MULTILINE): the start of the haystack.
    Bol,
    /// `$` (no MULTILINE): the end, or just before a final `\n`.
    Eol,
    /// `a|b|...`, tried in order.
    Branch(&'static [&'static [Node]]),
    /// A group (capturing or not: nothing here reads captures).
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
        [n @ (Node::Lit(_) | Node::In(_) | Node::InV(_))] => Some(*n),
        _ => None,
    }
}

#[inline]
fn char_matches(node: Node, c: char, v: UnicodeVersion) -> bool {
    match node {
        Node::Lit(l) => l == c,
        Node::In(cls) => cls.contains(c, v),
        Node::InV(classes) => classes[version_index(v)].contains(c, v),
        _ => unreachable!("not a one-character node"),
    }
}

/// Does `nodes` repeat a multi-node body without bound (outside lookarounds)?
/// The backtracking [`Matcher`] would recurse once per iteration of it.
fn has_unbounded_group(nodes: &[Node]) -> bool {
    nodes.iter().any(|node| match *node {
        Node::Repeat(_, max, body) => {
            (max == u32::MAX && single_char(body).is_none()) || has_unbounded_group(body)
        }
        Node::Branch(alts) => alts.iter().any(|a| has_unbounded_group(a)),
        Node::Group(body) => has_unbounded_group(body),
        _ => false,
    })
}

/// Backtracking matcher over one haystack.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Matcher<'s> {
    pub s: &'s str,
    pub v: UnicodeVersion,
}

impl Matcher<'_> {
    /// `re`'s match of `nodes` at `pos`: the end of the highest-priority
    /// match, if any. With `nonempty`, a match ending at `pos` does not count
    /// and the search goes on (sre's `must_advance`, after an empty
    /// `finditer` match).
    pub(crate) fn match_at(&self, nodes: &[Node], pos: usize, nonempty: bool) -> Option<usize> {
        let mut end = None;
        self.seq(nodes, pos, &mut |p| {
            if nonempty && p == pos {
                return false;
            }
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
                Node::Lit(_) | Node::In(_) | Node::InV(_) => match char_at(self.s, pos) {
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
    /// for counted groups; unbounded ones run on the [`Nfa`]).
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
    /// Try the first target, then the second (priority order).
    Split(usize, usize),
    Jmp(usize),
    Bol,
    Eol,
    Ahead(bool, &'static [Node]),
    Behind(bool, u32, &'static [Node]),
    Match,
}

/// A compiled pattern (anchoring comes from the pattern's own `^` / `$`),
/// simulated breadth-first.
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
            for &pc in &cur.dense {
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

    /// The end of the match `re` finds at `start` (leftmost-first: the
    /// highest-priority thread that reaches `Match` wins, as backtracking
    /// would find it). With `nonempty`, an empty match does not count.
    pub(crate) fn match_end(&self, m: &Matcher<'_>, start: usize, nonempty: bool) -> Option<usize> {
        let n = self.prog.len();
        let mut cur = SparseSet::new(n);
        let mut next = SparseSet::new(n);
        let mut stack = Vec::new();
        let mut best = None;
        let accept = |p: usize| !(nonempty && p == start);
        if self.closure_ordered(0, start, m, accept, &mut cur, &mut stack) {
            best = Some(start);
        }
        let mut pos = start;
        for c in m.s[start..].chars() {
            if cur.is_empty() {
                break;
            }
            let npos = pos + c.len_utf8();
            next.clear();
            for &pc in &cur.dense {
                if let Inst::Char(node) = self.prog[pc] {
                    if char_matches(node, c, m.v)
                        && self.closure_ordered(pc + 1, npos, m, accept, &mut next, &mut stack)
                    {
                        // Every thread after this one has a lower priority.
                        best = Some(npos);
                        break;
                    }
                }
            }
            std::mem::swap(&mut cur, &mut next);
            pos = npos;
        }
        best
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
                inst => Self::follow(inst, pc, pos, m, stack),
            }
        }
        false
    }

    /// [`Self::closure`] in priority order (depth first, first target of a
    /// split first): stops at the first accepted Match, dropping every
    /// lower-priority thread. A rejected Match is a dead thread.
    fn closure_ordered(
        &self,
        pc: usize,
        pos: usize,
        m: &Matcher<'_>,
        accept: impl Fn(usize) -> bool,
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
                Inst::Match => {
                    if accept(pos) {
                        stack.clear();
                        return true;
                    }
                }
                inst => Self::follow(inst, pc, pos, m, stack),
            }
        }
        false
    }

    /// Push the successors of the epsilon instruction `inst` (at `pc`), the
    /// highest-priority one last (it is popped first).
    #[inline]
    fn follow(inst: Inst, pc: usize, pos: usize, m: &Matcher<'_>, stack: &mut Vec<usize>) {
        match inst {
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
            Inst::Char(_) | Inst::Match => {}
        }
    }
}

fn compile(nodes: &'static [Node], prog: &mut Vec<Inst>) {
    for &node in nodes {
        compile_one(node, prog);
    }
}

fn compile_one(node: Node, prog: &mut Vec<Inst>) {
    match node {
        Node::Lit(_) | Node::In(_) | Node::InV(_) => prog.push(Inst::Char(node)),
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

/// A set of program counters with O(1) insert / clear (Briggs-Torczon),
/// iterated in insertion order.
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
            Node::InV(classes) => {
                out.extend(classes.iter().map(|c| FirstItem::Class(c)));
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

/// How one alternative is matched.
#[derive(Debug)]
enum Engine {
    Backtrack(&'static [Node]),
    Nfa(Nfa),
}

/// A top-level `a|b|...` (optionally after `^`), with each character's
/// candidate alternatives precomputed. [`Alternatives::match_at`] tries only
/// the alternatives that can start with the character at `pos`, in their
/// original order, so the result is `re`'s.
#[derive(Debug)]
pub(crate) struct Alternatives {
    alts: Vec<Engine>,
    /// The pattern starts with `^`: it only matches at 0.
    anchored: bool,
    /// Every match starts at 0 (the pattern, or each alternative, starts with `^`).
    all_anchored: bool,
    ascii: Vec<Vec<u16>>,
    /// Alternatives that may start with a non-ASCII character (`None`: any).
    other: Vec<(u16, Option<Vec<FirstItem>>)>,
    /// Alternatives that can match without consuming a character (the only
    /// candidates at the end of the haystack).
    nullable: Vec<u16>,
}

impl Alternatives {
    /// The alternatives of `nodes` (`[Branch]`, `[Bol, Branch]`, or any
    /// other pattern as a single alternative), each matched with `re`'s
    /// priorities. With `boolean`, every alternative is an [`Nfa`] (for
    /// [`Self::is_match`] only).
    pub(crate) fn new(nodes: &'static [Node], boolean: bool) -> Self {
        let (anchored, alts): (bool, Vec<&'static [Node]>) = match nodes {
            [Node::Branch(alts)] => (false, alts.to_vec()),
            [Node::Bol, Node::Branch(alts)] => (true, alts.to_vec()),
            [Node::Group(body)] => return Self::new(body, boolean),
            _ => (false, vec![nodes]),
        };
        let mut ascii = vec![Vec::new(); 128];
        let mut other = Vec::new();
        let mut nullable_alts = Vec::new();
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
                nullable_alts.push(i);
            } else if items.iter().any(|it| it.may_be_non_ascii()) {
                other.push((i, Some(items)));
            }
        }
        let all_anchored = anchored
            || alts
                .iter()
                .all(|alt| matches!(alt.first(), Some(Node::Bol)));
        let alts = alts
            .into_iter()
            .map(|alt| {
                if boolean || has_unbounded_group(alt) {
                    Engine::Nfa(Nfa::new(alt))
                } else {
                    Engine::Backtrack(alt)
                }
            })
            .collect();
        Self {
            alts,
            anchored,
            all_anchored,
            ascii,
            other,
            nullable: nullable_alts,
        }
    }

    /// Can a match only start at position 0 (the pattern, or every one of
    /// its alternatives, starts with `^`)?
    pub(crate) fn all_anchored(&self) -> bool {
        self.all_anchored
    }

    #[inline]
    fn try_alt(&self, i: u16, m: &Matcher<'_>, pos: usize, nonempty: bool) -> Option<usize> {
        match &self.alts[usize::from(i)] {
            Engine::Backtrack(nodes) => m.match_at(nodes, pos, nonempty),
            Engine::Nfa(nfa) => nfa.match_end(m, pos, nonempty),
        }
    }

    /// The end of `re`'s match of the alternation at `pos`, if any
    /// (`nonempty`: see [`Matcher::match_at`]).
    #[inline]
    pub(crate) fn match_at(&self, m: &Matcher<'_>, pos: usize, nonempty: bool) -> Option<usize> {
        if self.anchored && pos != 0 {
            return None;
        }
        let Some(c) = char_at(m.s, pos) else {
            return self
                .nullable
                .iter()
                .find_map(|&i| self.try_alt(i, m, pos, nonempty));
        };
        if c.is_ascii() {
            for &i in &self.ascii[c as usize] {
                if let Some(end) = self.try_alt(i, m, pos, nonempty) {
                    return Some(end);
                }
            }
        } else {
            for (i, items) in &self.other {
                if items
                    .as_ref()
                    .is_none_or(|it| it.iter().any(|x| x.contains(c, m.v)))
                {
                    if let Some(end) = self.try_alt(*i, m, pos, nonempty) {
                        return Some(end);
                    }
                }
            }
        }
        None
    }

    /// `re.match(pattern, s) is not None` (built with `boolean`).
    pub(crate) fn is_match(&self, s: &str, v: UnicodeVersion) -> bool {
        let run = |i: u16| match &self.alts[usize::from(i)] {
            Engine::Nfa(nfa) => nfa.is_match(s, v),
            Engine::Backtrack(nodes) => Matcher { s, v }.matches_at(nodes, 0),
        };
        let Some(c) = s.chars().next() else {
            return self.nullable.iter().any(|&i| run(i));
        };
        if c.is_ascii() {
            self.ascii[c as usize].iter().any(|&i| run(i))
        } else {
            self.other.iter().any(|(i, items)| {
                items
                    .as_ref()
                    .is_none_or(|it| it.iter().any(|x| x.contains(c, v)))
                    && run(*i)
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static DIGIT: Class = Class {
        negate: false,
        ascii: [0x3ff_0000_0000_0000, 0],
        ranges: &[],
        cats: &[],
    };
    // `(?:-[0-9]+)*$`, `-?(?:[0-9]-)*[0-9]` etc. as the generator emits them.
    static DASH_DIGITS: &[Node] = &[
        Node::Lit('-'),
        Node::Repeat(1, u32::MAX, &[Node::In(&DIGIT)]),
    ];
    static GROUPS: &[Node] = &[
        Node::Repeat(0, u32::MAX, &[Node::Group(DASH_DIGITS)]),
        Node::Lit('x'),
    ];

    #[test]
    fn priority_nfa_matches_backtracking() {
        let v = UnicodeVersion::V15_1;
        let nfa = Nfa::new(GROUPS);
        for s in ["x", "-1x", "-12-3x", "-1-x", "-1-2", "", "-", "-1-2-3-4x"] {
            let m = Matcher { s, v };
            assert_eq!(
                nfa.match_end(&m, 0, false),
                m.match_at(GROUPS, 0, false),
                "{s:?}"
            );
        }
        let long = "-1".repeat(300_000) + "x";
        let m = Matcher { s: &long, v };
        assert_eq!(nfa.match_end(&m, 0, false), Some(long.len()));
    }

    #[test]
    fn nonempty_rejects_empty_matches() {
        static OPT: &[Node] = &[Node::Repeat(0, 1, &[Node::Lit('a')])];
        let v = UnicodeVersion::V15_1;
        let m = Matcher { s: "b", v };
        assert_eq!(m.match_at(OPT, 0, false), Some(0));
        assert_eq!(m.match_at(OPT, 0, true), None);
        let nfa = Nfa::new(OPT);
        assert_eq!(nfa.match_end(&m, 0, false), Some(0));
        assert_eq!(nfa.match_end(&m, 0, true), None);
        let m = Matcher { s: "ab", v };
        assert_eq!(nfa.match_end(&m, 0, true), Some(1));
    }
}
