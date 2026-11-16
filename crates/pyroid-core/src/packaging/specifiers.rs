// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! `packaging.specifiers` and the range engine of `packaging._ranges`, as in packaging 26.3.
//!
//! [`is_valid`] decides `Specifier._regex.fullmatch(spec)` (the pattern is regular, and none of
//! its optional parts can block a later one, so a greedy pass decides). The range engine is
//! ported function by function: [`bounds_for_spec`], [`intersect`], [`matches_bounds_only`],
//! [`ranges_are_prerelease_only`] and the predicates `filter_by_ranges` applies.

use std::cmp::Ordering;

use super::num::Num;
use super::version::{self, Pre, Version, trim_zeros};
use crate::text::is_py_whitespace;

/// A specifier operator.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Op {
    /// `~=`
    Compatible,
    /// `==`
    Equal,
    /// `!=`
    NotEqual,
    /// `<=`
    LessEqual,
    /// `>=`
    GreaterEqual,
    /// `<`
    Less,
    /// `>`
    Greater,
    /// `===`
    Arbitrary,
}

impl Op {
    /// The operator as written.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Op::Compatible => "~=",
            Op::Equal => "==",
            Op::NotEqual => "!=",
            Op::LessEqual => "<=",
            Op::GreaterEqual => ">=",
            Op::Less => "<",
            Op::Greater => ">",
            Op::Arbitrary => "===",
        }
    }

    /// The operator `s` is, if it is one.
    #[must_use]
    pub fn from_str(s: &str) -> Option<Op> {
        Some(match s {
            "~=" => Op::Compatible,
            "==" => Op::Equal,
            "!=" => Op::NotEqual,
            "<=" => Op::LessEqual,
            ">=" => Op::GreaterEqual,
            "<" => Op::Less,
            ">" => Op::Greater,
            "===" => Op::Arbitrary,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// Specifier._regex
// ---------------------------------------------------------------------------

struct Scan<'a> {
    s: &'a [char],
    i: usize,
    /// Letters compare with `re.IGNORECASE`'s Unicode folds (the `~=` branch is outside the
    /// `(?a:...)` groups): `İ` and `ı` match `i`, `ſ` matches `s`.
    unicode_case: bool,
}

impl Scan<'_> {
    fn peek(&self) -> Option<char> {
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn sep(&mut self) -> bool {
        matches!(self.peek(), Some('-' | '_' | '.')) && {
            self.i += 1;
            true
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.peek().is_some_and(|c| c.is_ascii_digit()) {
            self.i += 1;
        }
        self.i - start
    }

    fn ws(&mut self) {
        while self.peek().is_some_and(is_py_whitespace) {
            self.i += 1;
        }
    }

    fn letter_eq(&self, c: char, lit: u8) -> bool {
        if c.is_ascii() {
            return (c as u8).eq_ignore_ascii_case(&lit);
        }
        self.unicode_case
            && match lit {
                b'i' => matches!(c, '\u{130}' | '\u{131}'),
                b's' => c == '\u{17f}',
                b'k' => c == '\u{212a}',
                _ => false,
            }
    }

    fn word(&mut self, words: &[&str]) -> bool {
        'next: for w in words {
            let w = w.as_bytes();
            if self.s.len() - self.i < w.len() {
                continue;
            }
            for (k, &b) in w.iter().enumerate() {
                if !self.letter_eq(self.s[self.i + k], b) {
                    continue 'next;
                }
            }
            self.i += w.len();
            return true;
        }
        false
    }

    /// `[-_\.]?WORD[-_\.]?[0-9]*`, optional.
    fn lettered(&mut self, words: &[&str]) {
        let save = self.i;
        self.sep();
        if self.word(words) {
            self.sep();
            self.digits();
        } else {
            self.i = save;
        }
    }

    /// `v?(?:[0-9]+!)?[0-9]+(?:\.[0-9]+)*`, with at least `min_parts` release parts.
    fn release(&mut self, min_parts: usize) -> bool {
        if matches!(self.peek(), Some('v' | 'V')) {
            self.i += 1;
        }
        let save = self.i;
        if self.digits() > 0 && self.eat('!') {
        } else {
            self.i = save;
        }
        if self.digits() == 0 {
            return false;
        }
        let mut parts = 1;
        while self.peek() == Some('.') && self.s.get(self.i + 1).is_some_and(char::is_ascii_digit)
        {
            self.i += 1;
            self.digits();
            parts += 1;
        }
        parts >= min_parts
    }

    /// The optional pre-, post- and dev-release parts.
    fn suffixes(&mut self) {
        self.lettered(&["alpha", "beta", "preview", "pre", "a", "b", "c", "rc"]);
        let save = self.i;
        if self.eat('-') && self.digits() > 0 {
        } else {
            self.i = save;
            self.lettered(&["post", "rev", "r"]);
        }
        self.lettered(&["dev"]);
    }

    /// `\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*`, optional (ASCII).
    fn local(&mut self) {
        let save = self.i;
        if !self.eat('+') {
            return;
        }
        let alnum = |c: Option<char>| c.is_some_and(|c| c.is_ascii_alphanumeric());
        if !alnum(self.peek()) {
            self.i = save;
            return;
        }
        loop {
            while alnum(self.peek()) {
                self.i += 1;
            }
            if matches!(self.peek(), Some('-' | '_' | '.')) && alnum(self.s.get(self.i + 1).copied())
            {
                self.i += 1;
            } else {
                break;
            }
        }
    }

    fn at_end(&self) -> bool {
        self.i == self.s.len()
    }
}

/// `Specifier._regex.fullmatch(spec)`: the specifier pattern with `re.VERBOSE | re.IGNORECASE`.
#[must_use]
pub fn is_valid(spec: &str) -> bool {
    let chars: Vec<char> = spec.trim_matches(is_py_whitespace).chars().collect();
    let mut s = Scan { s: &chars, i: 0, unicode_case: false };
    let rest = |k: usize| &chars[k.min(chars.len())..];
    if rest(0).starts_with(&['=', '=', '=']) {
        // === \s* [^\s;)]*
        s.i = 3;
        s.ws();
        return s.s[s.i..].iter().all(|&c| !is_py_whitespace(c) && c != ';' && c != ')');
    }
    if rest(0).starts_with(&['~', '=']) {
        s.i = 2;
        s.unicode_case = true;
        s.ws();
        if !s.release(2) {
            return false;
        }
        s.suffixes();
        return s.at_end();
    }
    if rest(0).starts_with(&['=', '=']) || rest(0).starts_with(&['!', '=']) {
        s.i = 2;
        s.ws();
        if !s.release(1) {
            return false;
        }
        let save = s.i;
        if s.eat('.') && s.eat('*') && s.at_end() {
            return true;
        }
        s.i = save;
        s.suffixes();
        s.local();
        return s.at_end();
    }
    let n = if rest(0).starts_with(&['<', '=']) || rest(0).starts_with(&['>', '=']) {
        2
    } else if matches!(chars.first(), Some('<' | '>')) {
        1
    } else {
        return false;
    };
    s.i = n;
    s.ws();
    if !s.release(1) {
        return false;
    }
    s.suffixes();
    s.at_end()
}

/// `Specifier.__init__`'s split of a valid `spec`: `(operator, version)`.
#[must_use]
pub fn split(spec: &str) -> (Op, &str) {
    let spec = spec.trim_matches(is_py_whitespace);
    let (op, n) = if spec.starts_with("===") {
        (Op::Arbitrary, 3)
    } else if let Some(op) = spec.get(..2).and_then(Op::from_str) {
        (op, 2)
    } else {
        (spec.get(..1).and_then(Op::from_str).unwrap_or(Op::Equal), 1)
    };
    (op, spec[n..].trim_matches(is_py_whitespace))
}

// ---------------------------------------------------------------------------
// The range engine (packaging._ranges)
// ---------------------------------------------------------------------------

/// Where a [`Point::Boundary`] sits relative to its version `V`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// After `V+local`, before `V.post0`.
    AfterLocals,
    /// After `V.postN`, before the next release.
    AfterPosts,
}

/// A bound's position: a version, or a point between two versions.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Point {
    /// A real version.
    Version(Version),
    /// `BoundaryVersion(V, kind)`.
    Boundary(Version, Kind),
}

/// A lower bound; `None` is `-inf`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Lower {
    /// The position.
    pub point: Option<Point>,
    /// `[` rather than `(`.
    pub inclusive: bool,
}

/// An upper bound; `None` is `+inf`.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Upper {
    /// The position.
    pub point: Option<Point>,
    /// `]` rather than `)`.
    pub inclusive: bool,
}

/// One interval.
pub type Interval = (Lower, Upper);

const NEG_INF: Lower = Lower { point: None, inclusive: false };
const POS_INF: Upper = Upper { point: None, inclusive: false };

fn lower(v: Version, inclusive: bool) -> Lower {
    Lower { point: Some(Point::Version(v)), inclusive }
}

fn upper(v: Version, inclusive: bool) -> Upper {
    Upper { point: Some(Point::Version(v)), inclusive }
}

/// `_ranges.trim_release`: trailing zeros off, one element kept.
fn trim_release(release: &[Num]) -> &[Num] {
    trim_zeros(release, true)
}

/// `MIN_VERSION`, `Version("0.dev0")`.
fn min_version() -> Version {
    let mut v = Version::release_only(vec![Num::ZERO]);
    v.dev = Some(Num::ZERO);
    v
}

fn lt(a: &Version, b: &Version) -> bool {
    a.cmp_key(b) == Ordering::Less
}

fn le(a: &Version, b: &Version) -> bool {
    a.cmp_key(b) != Ordering::Greater
}

/// Same epoch and (trimmed) release as `v`: `parsed` is in `v`'s release.
fn same_release(v: &Version, parsed: &Version) -> bool {
    if parsed.epoch != v.epoch {
        return false;
    }
    let trimmed = trim_release(&v.release);
    let other = &parsed.release;
    other.len() >= trimmed.len()
        && other[..trimmed.len()] == *trimmed
        && other[trimmed.len()..].iter().all(Num::is_zero)
}

/// `BoundaryVersion._is_family`.
fn is_family(v: &Version, kind: Kind, other: &Version) -> bool {
    if !same_release(v, other) || other.pre != v.pre {
        return false;
    }
    match kind {
        Kind::AfterLocals => other.post == v.post && other.dev == v.dev,
        Kind::AfterPosts => other.dev == v.dev || other.post.is_some(),
    }
}

/// `BoundaryVersion._order_key` compared: the version's key without its local label, then
/// an infinite local component; an `AFTER_POSTS` suffix has an infinite post number.
fn cmp_boundaries(a: &Version, ak: Kind, b: &Version, bk: Kind) -> Ordering {
    a.epoch
        .cmp(&b.epoch)
        .then_with(|| trim_zeros(&a.release, false).cmp(trim_zeros(&b.release, false)))
        .then_with(|| {
            let (sa, sb) = (a.suffix(), b.suffix());
            let head = sa.pre_rank.cmp(&sb.pre_rank).then_with(|| sa.pre_n.cmp(&sb.pre_n));
            head.then_with(|| match (ak, bk) {
                (Kind::AfterLocals, Kind::AfterLocals) => (
                    sa.post_rank,
                    &sa.post_n,
                    sa.dev_rank,
                    &sa.dev_n,
                )
                    .cmp(&(sb.post_rank, &sb.post_n, sb.dev_rank, &sb.dev_n)),
                (Kind::AfterPosts, Kind::AfterPosts) => Ordering::Equal,
                // (1, inf, 1, 0) against a real (post_rank, post_n, ...): post_rank <= 1 and
                // post_n < inf, so the AFTER_POSTS suffix is larger.
                (Kind::AfterPosts, Kind::AfterLocals) => Ordering::Greater,
                (Kind::AfterLocals, Kind::AfterPosts) => Ordering::Less,
            })
        })
}

/// `a == b` for two points, as Python evaluates it.
fn point_eq(a: &Point, b: &Point) -> bool {
    match (a, b) {
        (Point::Version(x), Point::Version(y)) => x.cmp_key(y) == Ordering::Equal,
        (Point::Boundary(x, xk), Point::Boundary(y, yk)) => {
            cmp_boundaries(x, *xk, y, *yk) == Ordering::Equal
        }
        // Neither side defines equality with the other: identity, false.
        _ => false,
    }
}

/// `boundary < version` (`BoundaryVersion.__lt__`).
fn boundary_lt_version(v: &Version, kind: Kind, other: &Version) -> bool {
    lt(v, other) && !is_family(v, kind, other)
}

/// `boundary > version` (`BoundaryVersion.__gt__`).
fn boundary_gt_version(v: &Version, kind: Kind, other: &Version) -> bool {
    le(other, v) || is_family(v, kind, other)
}

/// `a < b` for two points, as Python evaluates it (reflected operators included).
fn point_lt(a: &Point, b: &Point) -> bool {
    match (a, b) {
        (Point::Version(x), Point::Version(y)) => lt(x, y),
        (Point::Boundary(x, xk), Point::Boundary(y, yk)) => {
            cmp_boundaries(x, *xk, y, *yk) == Ordering::Less
        }
        // Version.__lt__(boundary) is NotImplemented: boundary.__gt__(version).
        (Point::Version(x), Point::Boundary(y, yk)) => boundary_gt_version(y, *yk, x),
        (Point::Boundary(x, xk), Point::Version(y)) => boundary_lt_version(x, *xk, y),
    }
}

/// `a > b` for two points.
fn point_gt(a: &Point, b: &Point) -> bool {
    match (a, b) {
        (Point::Version(x), Point::Version(y)) => lt(y, x),
        (Point::Boundary(x, xk), Point::Boundary(y, yk)) => {
            cmp_boundaries(x, *xk, y, *yk) == Ordering::Greater
        }
        (Point::Version(x), Point::Boundary(y, yk)) => boundary_lt_version(y, *yk, x),
        (Point::Boundary(x, xk), Point::Version(y)) => boundary_gt_version(x, *xk, y),
    }
}

fn opt_point_eq(a: Option<&Point>, b: Option<&Point>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(x), Some(y)) => point_eq(x, y),
        _ => false,
    }
}

/// `LowerBound.__eq__`.
fn lower_eq(a: &Lower, b: &Lower) -> bool {
    opt_point_eq(a.point.as_ref(), b.point.as_ref()) && a.inclusive == b.inclusive
}

/// `LowerBound.__lt__`.
fn lower_lt(a: &Lower, b: &Lower) -> bool {
    let Some(x) = &a.point else {
        return b.point.is_some();
    };
    let Some(y) = &b.point else {
        return false;
    };
    if !point_eq(x, y) {
        return point_lt(x, y);
    }
    a.inclusive && !b.inclusive
}

/// `UpperBound.__lt__`.
fn upper_lt(a: &Upper, b: &Upper) -> bool {
    let Some(x) = &a.point else {
        return false;
    };
    let Some(y) = &b.point else {
        return true;
    };
    if !point_eq(x, y) {
        return point_lt(x, y);
    }
    !a.inclusive && b.inclusive
}

/// `max(a, b)` of two lower bounds (`b` only if `b > a`, `__gt__` from `total_ordering`).
fn max_lower(a: &Lower, b: &Lower) -> Lower {
    let b_gt_a = !lower_lt(b, a) && !lower_eq(b, a);
    if b_gt_a { b.clone() } else { a.clone() }
}

/// `min(a, b)` of two upper bounds (`b` only if `b < a`).
fn min_upper(a: &Upper, b: &Upper) -> Upper {
    if upper_lt(b, a) { b.clone() } else { a.clone() }
}

/// `Version.__replace__` steps the engine uses (packaging validates nothing that can fail here).
fn with_dev(v: &Version, dev: Num) -> Version {
    let mut out = v.clone();
    out.dev = Some(dev);
    out.local = None;
    out
}

/// `least_version_above(boundary)`.
fn least_version_above(v: &Version, kind: Kind) -> Option<Version> {
    match kind {
        Kind::AfterLocals => {
            if let Some(d) = &v.dev {
                return Some(with_dev(v, d.plus_one()));
            }
            let next_post = v.post.as_ref().map_or(Num::ZERO, Num::plus_one);
            let mut out = v.clone();
            out.post = Some(next_post);
            out.dev = Some(Num::ZERO);
            out.local = None;
            Some(out)
        }
        Kind::AfterPosts => {
            let (k, n) = v.pre.as_ref()?;
            let mut out = v.clone();
            out.pre = Some((*k, n.plus_one()));
            out.post = None;
            out.dev = Some(Num::ZERO);
            out.local = None;
            Some(out)
        }
    }
}

/// `range_is_empty(lower, upper)`.
fn range_is_empty(lo: &Lower, up: &Upper) -> bool {
    let Some(u) = &up.point else {
        return false;
    };
    let Some(l) = &lo.point else {
        return !up.inclusive
            && match u {
                Point::Version(v) => le(v, &min_version()),
                Point::Boundary(..) => false,
            };
    };
    if let Point::Boundary(v, kind) = l {
        if let Some(successor) = least_version_above(v, *kind) {
            let s = Point::Version(successor);
            if point_eq(u, &s) {
                return !up.inclusive;
            }
            return point_lt(u, &s);
        }
    }
    if point_eq(l, u) {
        return !(lo.inclusive && up.inclusive);
    }
    point_gt(l, u)
}

/// `intersect_ranges(left, right)`.
#[must_use]
pub fn intersect(left: &[Interval], right: &[Interval]) -> Vec<Interval> {
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        let (ll, lu) = &left[i];
        let (rl, ru) = &right[j];
        let lo = max_lower(ll, rl);
        let up = min_upper(lu, ru);
        if !range_is_empty(&lo, &up) {
            out.push((lo, up));
        }
        if upper_lt(lu, ru) {
            i += 1;
        } else {
            j += 1;
        }
    }
    out
}

fn next_prefix_dev0(v: &Version) -> Version {
    let mut release = v.release.clone();
    if let Some(last) = release.last_mut() {
        *last = last.plus_one();
    }
    let mut out = Version::release_only(release);
    out.epoch = v.epoch.clone();
    out.dev = Some(Num::ZERO);
    out
}

fn base_dev0(v: &Version) -> Version {
    let mut out = Version::release_only(v.release.clone());
    out.epoch = v.epoch.clone();
    out.dev = Some(Num::ZERO);
    out
}

/// `wildcard_ranges(op, base)` (`op` is `==` or `!=`).
fn wildcard_ranges(op: Op, base: &Version) -> Vec<Interval> {
    let lo = base_dev0(base);
    let hi = next_prefix_dev0(base);
    if op == Op::Equal {
        return vec![(lower(lo, true), upper(hi, false))];
    }
    vec![(NEG_INF, upper(lo, false)), (lower(hi, true), POS_INF)]
}

/// `standard_ranges(op, version, has_local)`; `None` for `~=` with a one-part release (where
/// packaging's `__replace__(release=())` raises `InvalidVersion`).
fn standard_ranges(op: Op, v: &Version, has_local: bool) -> Option<Vec<Interval>> {
    Some(match op {
        Op::GreaterEqual => vec![(lower(v.clone(), true), POS_INF)],
        Op::LessEqual => vec![(
            NEG_INF,
            Upper { point: Some(Point::Boundary(v.clone(), Kind::AfterLocals)), inclusive: true },
        )],
        Op::Greater => {
            if let Some(d) = &v.dev {
                vec![(lower(with_dev(v, d.plus_one()), true), POS_INF)]
            } else if let Some(p) = &v.post {
                let mut b = v.clone();
                b.post = Some(p.plus_one());
                b.dev = Some(Num::ZERO);
                b.local = None;
                vec![(lower(b, true), POS_INF)]
            } else {
                vec![(
                    Lower {
                        point: Some(Point::Boundary(v.clone(), Kind::AfterPosts)),
                        inclusive: false,
                    },
                    POS_INF,
                )]
            }
        }
        Op::Less => {
            let bound = if v.is_prerelease() { v.clone() } else { with_dev(v, Num::ZERO) };
            if le(&bound, &min_version()) {
                return Some(Vec::new());
            }
            vec![(NEG_INF, upper(bound, false))]
        }
        Op::Equal | Op::NotEqual => {
            let up = if has_local {
                Point::Version(v.clone())
            } else {
                Point::Boundary(v.clone(), Kind::AfterLocals)
            };
            if op == Op::Equal {
                vec![(lower(v.clone(), true), Upper { point: Some(up), inclusive: true })]
            } else {
                vec![
                    (NEG_INF, upper(v.clone(), false)),
                    (Lower { point: Some(up), inclusive: false }, POS_INF),
                ]
            }
        }
        Op::Compatible => {
            if v.release.len() < 2 {
                return None;
            }
            let mut prefix = v.clone();
            prefix.release.pop();
            vec![(lower(v.clone(), true), upper(next_prefix_dev0(&prefix), false))]
        }
        Op::Arbitrary => vec![(NEG_INF, POS_INF)],
    })
}

/// `bounds_for_spec(op, version_str, version)`: `version` is the parsed `version_str` (its base,
/// without `.*`, for a wildcard).
#[must_use]
pub fn bounds_for_spec(op: Op, version_str: &str, v: &Version) -> Option<Vec<Interval>> {
    if op == Op::Arbitrary {
        return Some(vec![(NEG_INF, POS_INF)]);
    }
    if version_str.ends_with(".*") {
        return Some(wildcard_ranges(op, v));
    }
    standard_ranges(op, v, version_str.contains('+'))
}

/// `intersect_specifier_bounds`: the intersection of each specifier's ranges.
#[must_use]
pub fn intersect_all<'a>(mut per_spec: impl Iterator<Item = &'a [Interval]>) -> Vec<Interval> {
    let Some(first) = per_spec.next() else {
        return Vec::new();
    };
    let mut result = first.to_vec();
    for sub in per_spec {
        result = intersect(&result, sub);
        if result.is_empty() {
            break;
        }
    }
    result
}

/// `lower._above(parsed)` (`None`: no predicate, unbounded).
fn above(lo: &Lower, parsed: &Version) -> bool {
    match &lo.point {
        None => true,
        Some(Point::Version(v)) => {
            if lo.inclusive {
                le(v, parsed)
            } else {
                lt(v, parsed)
            }
        }
        Some(Point::Boundary(v, Kind::AfterPosts)) => {
            if le(parsed, v) {
                return false;
            }
            !(same_release(v, parsed) && parsed.pre == v.pre)
        }
        Some(Point::Boundary(v, Kind::AfterLocals)) => {
            if le(parsed, v) {
                return false;
            }
            !(same_release(v, parsed)
                && parsed.pre == v.pre
                && parsed.post == v.post
                && parsed.dev == v.dev)
        }
    }
}

/// `upper._below(parsed)`.
fn below(up: &Upper, parsed: &Version) -> bool {
    match &up.point {
        None => true,
        Some(Point::Version(v)) => {
            if up.inclusive {
                le(parsed, v)
            } else {
                lt(parsed, v)
            }
        }
        Some(Point::Boundary(v, Kind::AfterLocals)) => {
            if le(parsed, v) {
                return true;
            }
            same_release(v, parsed)
                && parsed.pre == v.pre
                && parsed.post == v.post
                && parsed.dev == v.dev
        }
        // BoundaryVersion.__ge__ (total_ordering): not (boundary < parsed).
        Some(Point::Boundary(v, Kind::AfterPosts)) => {
            !boundary_lt_version(v, Kind::AfterPosts, parsed)
        }
    }
}

/// `matches_bounds_only(ranges, version)`.
#[must_use]
pub fn matches_bounds_only(ranges: &[Interval], v: &Version) -> bool {
    for (lo, up) in ranges {
        if !above(lo, v) {
            return false;
        }
        if below(up, v) {
            return true;
        }
    }
    false
}

/// The `filter_by_ranges` membership test for one item (the single-range hot path and the
/// general loop give the same answer).
#[must_use]
pub fn in_ranges(ranges: &[Interval], v: &Version) -> bool {
    matches_bounds_only(ranges, v)
}

fn nearest_release_above_prerelease(v: &Version) -> Version {
    let mut out = v.clone();
    if v.pre.is_some() {
        out.pre = None;
        out.post = None;
        out.dev = None;
        out.local = None;
    } else {
        out.dev = None;
        out.local = None;
    }
    out
}

/// `_lowest_release_at_or_above(value)`.
fn lowest_release_at_or_above(p: Option<&Point>) -> Version {
    match p {
        None => Version::release_only(vec![Num::ZERO]),
        Some(Point::Boundary(v, _)) => {
            if v.is_prerelease() {
                return nearest_release_above_prerelease(v);
            }
            let mut out = v.clone();
            out.post = Some(v.post.as_ref().map_or(Num::ZERO, Num::plus_one));
            out.local = None;
            out
        }
        Some(Point::Version(v)) => {
            if v.is_prerelease() {
                nearest_release_above_prerelease(v)
            } else {
                v.clone()
            }
        }
    }
}

/// `ranges_are_prerelease_only(ranges)`.
#[must_use]
pub fn ranges_are_prerelease_only(ranges: &[Interval]) -> bool {
    for (lo, up) in ranges {
        let nearest = Point::Version(lowest_release_at_or_above(lo.point.as_ref()));
        let Some(u) = &up.point else {
            return false;
        };
        if point_lt(&nearest, u) {
            return false;
        }
        if point_eq(&nearest, u) && up.inclusive {
            return false;
        }
    }
    true
}

/// `_fast_match(specifier, parsed)`: `Some(answer)` without building ranges, or `None`.
#[must_use]
pub fn fast_match(op: Op, version_str: &str, spec_v: Option<&Version>, parsed: &Version) -> Option<bool> {
    if version_str.ends_with(".*") || parsed.local.is_some() {
        return None;
    }
    let spec_v = spec_v?;
    let o = parsed.cmp_key(spec_v);
    match op {
        Op::GreaterEqual => Some(o != Ordering::Less),
        Op::LessEqual => Some(o != Ordering::Greater),
        Op::Equal => Some(o == Ordering::Equal),
        Op::NotEqual => Some(o != Ordering::Equal),
        Op::Less | Op::Greater => {
            if parsed.epoch != spec_v.epoch
                || trim_release(&parsed.release) != trim_release(&spec_v.release)
            {
                Some(if op == Op::Less { o == Ordering::Less } else { o == Ordering::Greater })
            } else {
                None
            }
        }
        Op::Compatible | Op::Arbitrary => None,
    }
}

/// The parsed version of a specifier's version string (`_get_spec_version`): `None` for `===`
/// strings that are not versions, and for the case-folded spellings `~=` accepts.
#[must_use]
pub fn spec_version(version_str: &str) -> Option<Version> {
    version::parse(version_str).map(|p| p.version)
}

/// `canonicalize_version(spec_version, strip_trailing_zero=...)`.
#[must_use]
pub fn canonical_version(v: &Version, strip_trailing_zero: bool) -> String {
    if strip_trailing_zero { v.to_string_trimmed() } else { v.to_string_py() }
}

/// Whether `pre` is set (a/b/rc), for callers outside the crate.
#[must_use]
pub fn pre_letter(p: Pre) -> &'static str {
    p.letter()
}
