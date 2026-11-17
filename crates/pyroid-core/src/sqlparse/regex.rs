// Reproduces CPython's `re` (sre) matching semantics (PSF-2.0) for the pattern syntax
// sqlparse's lexer uses.
//! A small backtracking regex engine with `re`'s semantics, compiled from the pattern text.
//!
//! sqlparse lexes with `re.compile(rx, re.IGNORECASE | re.UNICODE).match(text, pos)` for each
//! pattern of its table, first match wins. To give the same tokens, every pattern is matched
//! here exactly as sre matches it:
//!
//! - `\w`, `\d`, `\s` and `\b` are Unicode (`str.isalnum() or "_"`, `str.isdecimal()`,
//!   `str.isspace()`), in the interpreter's Unicode version;
//! - IGNORECASE compares the simple lowercase mapping (`_sre.unicode_tolower`) of the text
//!   character with the pattern's, plus `re._casefix`'s extra equivalences (`i`/`ı`,
//!   `s`/`ſ`, ...); a character class is lowered once and its members folded at compile time,
//!   only if it holds a cased character (sre's `IN` vs `IN_UNI_IGNORE`);
//! - `.` is any character but `\n`; `$` matches at the end and before a final `\n`; `^`, `\A`,
//!   `\b` and lookbehinds see the text before `pos` (sre's `state->beginning` is the string
//!   start, not `pos`);
//! - alternation, greedy and lazy quantifiers backtrack in sre's order; lookarounds are atomic.
//!
//! Syntax outside that set ([`Error::Unsupported`]) is left to Python's `re`: possessive
//! quantifiers, atomic groups, conditionals, scoped inline flags, `re.VERBOSE`/`ASCII`/`LOCALE`,
//! `\N{...}`, repeated sub-patterns that can match the empty string, backreferences into
//! lookarounds, non-BMP ranges under IGNORECASE, surrogates.

use std::fmt;

use super::pystr;
use crate::ftfy::UnicodeVersion;

/// `re.IGNORECASE`.
pub const FLAG_IGNORECASE: u32 = 2;
/// `re.LOCALE`.
pub const FLAG_LOCALE: u32 = 4;
/// `re.MULTILINE`.
pub const FLAG_MULTILINE: u32 = 8;
/// `re.DOTALL`.
pub const FLAG_DOTALL: u32 = 16;
/// `re.UNICODE` (always set for str patterns).
pub const FLAG_UNICODE: u32 = 32;
/// `re.VERBOSE`.
pub const FLAG_VERBOSE: u32 = 64;
/// `re.DEBUG`.
pub const FLAG_DEBUG: u32 = 128;
/// `re.ASCII`.
pub const FLAG_ASCII: u32 = 256;

/// Why a pattern could not be compiled here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// Valid for `re`, but outside what this engine reproduces.
    Unsupported(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Unsupported(s) => write!(f, "unsupported regex syntax: {s}"),
        }
    }
}

impl std::error::Error for Error {}

fn unsupported<T>(what: impl Into<String>) -> Result<T, Error> {
    Err(Error::Unsupported(what.into()))
}

/// Character categories (`\w`, `\W`, `\d`, `\D`, `\s`, `\S`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cat {
    Word,
    NotWord,
    Digit,
    NotDigit,
    Space,
    NotSpace,
}

impl Cat {
    fn test(self, c: char, v: UnicodeVersion) -> bool {
        match self {
            Cat::Word => pystr::is_word(c, v),
            Cat::NotWord => !pystr::is_word(c, v),
            Cat::Digit => pystr::is_digit(c, v),
            Cat::NotDigit => !pystr::is_digit(c, v),
            Cat::Space => pystr::is_space(c),
            Cat::NotSpace => !pystr::is_space(c),
        }
    }
}

/// A parsed character class.
#[derive(Clone, Debug, Default)]
struct ClassSpec {
    negate: bool,
    ranges: Vec<(u32, u32)>,
    cats: Vec<Cat>,
}

/// A compiled character class, with sre's IGNORECASE treatment applied.
#[derive(Clone, Debug)]
struct Set {
    negate: bool,
    /// Lower the text character before testing (sre's `IN_UNI_IGNORE`).
    lower: bool,
    ascii: u128,
    /// Non-ASCII members, sorted and disjoint.
    ranges: Vec<(u32, u32)>,
    cats: Vec<Cat>,
}

impl Set {
    fn compile(spec: &ClassSpec, ignore: bool, v: UnicodeVersion) -> Result<Set, Error> {
        let cased = ignore
            && spec.ranges.iter().any(|&(a, b)| {
                (a..=b).any(|cp| char::from_u32(cp).is_some_and(|c| pystr::is_cased(c, v)))
            });
        let mut members: Vec<(u32, u32)> = Vec::new();
        if cased {
            for &(a, b) in &spec.ranges {
                if b > 0xFFFF {
                    return unsupported("a non-BMP range under IGNORECASE");
                }
                for cp in a..=b {
                    let Some(c) = char::from_u32(cp) else {
                        continue;
                    };
                    let lo = pystr::simple_lower(c, v);
                    members.push((u32::from(lo), u32::from(lo)));
                    for x in pystr::extra_cases(lo) {
                        members.push((u32::from(x), u32::from(x)));
                    }
                }
            }
        } else {
            members.extend_from_slice(&spec.ranges);
        }
        members.sort_unstable();
        let mut merged: Vec<(u32, u32)> = Vec::new();
        for (a, b) in members {
            match merged.last_mut() {
                Some(last) if a <= last.1.saturating_add(1) => last.1 = last.1.max(b),
                _ => merged.push((a, b)),
            }
        }
        let mut ascii = 0u128;
        let mut ranges = Vec::new();
        for (a, b) in merged {
            for cp in a..=b.min(0x7F) {
                ascii |= 1u128 << cp;
            }
            if b >= 0x80 {
                ranges.push((a.max(0x80), b));
            }
        }
        Ok(Set {
            negate: spec.negate,
            lower: cased,
            ascii,
            ranges,
            cats: spec.cats.clone(),
        })
    }

    #[inline]
    fn test(&self, c: char, v: UnicodeVersion) -> bool {
        let c = if self.lower {
            pystr::simple_lower(c, v)
        } else {
            c
        };
        let cp = u32::from(c);
        let hit = if cp < 0x80 {
            self.ascii & (1u128 << cp) != 0
        } else {
            let after = self.ranges.partition_point(|&(lo, _)| lo <= cp);
            after > 0 && cp <= self.ranges[after - 1].1
        } || self.cats.iter().any(|cat| cat.test(c, v));
        hit != self.negate
    }

    /// Whether a non-ASCII character could be a member.
    fn may_be_non_ascii(&self) -> bool {
        self.negate || self.lower || !self.ranges.is_empty() || !self.cats.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Parser (sre_parse's syntax)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Ast {
    Empty,
    Lit(char),
    Class(ClassSpec),
    Any,
    Start,
    /// `\A`: the string start, whatever the flags.
    StartAbs,
    End,
    EndAbs,
    WordB,
    NotWordB,
    Group(Option<usize>, Box<Ast>),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat {
        node: Box<Ast>,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    },
    Look {
        ahead: bool,
        negate: bool,
        node: Box<Ast>,
    },
    Backref(usize),
}

/// `MAXREPEAT`-free bound on counted repeats this engine expands.
const MAX_COUNT: u32 = 1000;

struct Parser<'a> {
    chars: Vec<char>,
    i: usize,
    groups: usize,
    names: Vec<(String, usize)>,
    _src: &'a str,
}

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.i).copied()
    }

    fn peek_at(&self, k: usize) -> Option<char> {
        self.chars.get(self.i + k).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn next(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.i += 1;
        Some(c)
    }

    fn alt(&mut self) -> Result<Ast, Error> {
        let mut branches = vec![self.concat()?];
        while self.eat('|') {
            branches.push(self.concat()?);
        }
        Ok(if branches.len() == 1 {
            branches.pop().unwrap_or(Ast::Empty)
        } else {
            Ast::Alt(branches)
        })
    }

    fn concat(&mut self) -> Result<Ast, Error> {
        let mut items = Vec::new();
        while let Some(c) = self.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            let atom = self.quantified(atom)?;
            items.push(atom);
        }
        Ok(match items.len() {
            0 => Ast::Empty,
            1 => items.pop().unwrap_or(Ast::Empty),
            _ => Ast::Concat(items),
        })
    }

    /// A `{m,n}` repeat at the cursor, if the text there is one (else `{` is a literal).
    fn counted(&mut self) -> Option<(u32, Option<u32>)> {
        let save = self.i;
        if !self.eat('{') {
            return None;
        }
        let num = |p: &mut Self| -> Option<Option<u32>> {
            let start = p.i;
            while p.peek().is_some_and(|c| c.is_ascii_digit()) {
                p.i += 1;
            }
            if p.i == start {
                return Some(None);
            }
            let s: String = p.chars[start..p.i].iter().collect();
            s.parse::<u32>().ok().map(Some)
        };
        let lo = num(self);
        let Some(lo) = lo else {
            self.i = save;
            return None;
        };
        let hi = if self.eat(',') {
            let Some(hi) = num(self) else {
                self.i = save;
                return None;
            };
            hi
        } else {
            lo
        };
        if !self.eat('}') {
            self.i = save;
            return None;
        }
        Some((lo.unwrap_or(0), hi))
    }

    fn quantified(&mut self, atom: Ast) -> Result<Ast, Error> {
        let (min, max) = match self.peek() {
            Some('*') => {
                self.i += 1;
                (0, None)
            }
            Some('+') => {
                self.i += 1;
                (1, None)
            }
            Some('?') => {
                self.i += 1;
                (0, Some(1))
            }
            Some('{') => match self.counted() {
                Some(r) => r,
                None => return Ok(atom),
            },
            _ => return Ok(atom),
        };
        if matches!(
            atom,
            Ast::Start
                | Ast::StartAbs
                | Ast::End
                | Ast::EndAbs
                | Ast::WordB
                | Ast::NotWordB
                | Ast::Look { .. }
        ) {
            return unsupported("a repeated assertion");
        }
        let greedy = !self.eat('?');
        if self.peek() == Some('+') {
            return unsupported("a possessive quantifier");
        }
        if max.is_some_and(|m| m < min) {
            return unsupported("a bad repeat range");
        }
        if min > MAX_COUNT || max.is_some_and(|m| m > MAX_COUNT) {
            return unsupported("a large counted repeat");
        }
        if matches!(self.peek(), Some('*' | '+' | '?' | '{')) && self.counted_or_quant() {
            return unsupported("a multiple repeat");
        }
        Ok(Ast::Repeat {
            node: Box::new(atom),
            min,
            max,
            greedy,
        })
    }

    fn counted_or_quant(&mut self) -> bool {
        match self.peek() {
            Some('{') => {
                let save = self.i;
                let r = self.counted().is_some();
                self.i = save;
                r
            }
            _ => true,
        }
    }

    fn hex(&mut self, n: usize) -> Result<u32, Error> {
        let mut v = 0u32;
        for _ in 0..n {
            let Some(d) = self.next().and_then(|c| c.to_digit(16)) else {
                return unsupported("a bad hex escape");
            };
            v = v * 16 + d;
        }
        Ok(v)
    }

    fn char_of(cp: u32) -> Result<char, Error> {
        match char::from_u32(cp) {
            Some(c) if !(0x10_F800..=0x10_FFFF).contains(&cp) => Ok(c),
            _ => unsupported("a surrogate or reserved code point"),
        }
    }

    /// An escape that stands for one character, after the backslash (`c` is the letter).
    fn char_escape(&mut self, c: char, in_class: bool) -> Result<Option<char>, Error> {
        Ok(Some(match c {
            'a' => '\x07',
            'f' => '\x0c',
            'n' => '\n',
            'r' => '\r',
            't' => '\t',
            'v' => '\x0b',
            'b' if in_class => '\x08',
            'x' => Self::char_of(self.hex(2)?)?,
            'u' => Self::char_of(self.hex(4)?)?,
            'U' => Self::char_of(self.hex(8)?)?,
            '0' => {
                // up to two more octal digits
                let mut v = 0u32;
                for _ in 0..2 {
                    match self.peek().and_then(|d| d.to_digit(8)) {
                        Some(d) => {
                            v = v * 8 + d;
                            self.i += 1;
                        }
                        None => break,
                    }
                }
                Self::char_of(v)?
            }
            c if c.is_ascii_alphanumeric() => return Ok(None),
            c => c,
        }))
    }

    fn class(&mut self) -> Result<Ast, Error> {
        let mut spec = ClassSpec {
            negate: self.eat('^'),
            ..ClassSpec::default()
        };
        let mut first = true;
        loop {
            let Some(c) = self.next() else {
                return unsupported("an unterminated class");
            };
            if c == ']' && !first {
                break;
            }
            first = false;
            let lo = if c == '\\' {
                let Some(e) = self.next() else {
                    return unsupported("a trailing backslash");
                };
                let cat = match e {
                    'w' => Some(Cat::Word),
                    'W' => Some(Cat::NotWord),
                    'd' => Some(Cat::Digit),
                    'D' => Some(Cat::NotDigit),
                    's' => Some(Cat::Space),
                    'S' => Some(Cat::NotSpace),
                    _ => None,
                };
                if let Some(cat) = cat {
                    spec.cats.push(cat);
                    continue;
                }
                match self.char_escape(e, true)? {
                    Some(ch) => ch,
                    None => {
                        if e.is_ascii_digit() {
                            // octal escape in a class
                            let mut v = e.to_digit(8).ok_or_else(|| {
                                Error::Unsupported("a bad escape in a class".into())
                            })?;
                            for _ in 0..2 {
                                match self.peek().and_then(|d| d.to_digit(8)) {
                                    Some(d) => {
                                        v = v * 8 + d;
                                        self.i += 1;
                                    }
                                    None => break,
                                }
                            }
                            Self::char_of(v)?
                        } else {
                            return unsupported(format!("the escape \\{e} in a class"));
                        }
                    }
                }
            } else if c == '[' && matches!(self.peek(), Some(':' | '=' | '.')) {
                return unsupported("a possible nested set");
            } else {
                c
            };
            // a range?
            if self.peek() == Some('-') && self.peek_at(1).is_some_and(|n| n != ']') {
                self.i += 1;
                let Some(h) = self.next() else {
                    return unsupported("an unterminated range");
                };
                let hi = if h == '\\' {
                    let Some(e) = self.next() else {
                        return unsupported("a trailing backslash");
                    };
                    match self.char_escape(e, true)? {
                        Some(ch) => ch,
                        None => return unsupported("a class escape as range end"),
                    }
                } else {
                    h
                };
                if hi < lo {
                    return unsupported("a bad range");
                }
                spec.ranges.push((u32::from(lo), u32::from(hi)));
            } else {
                spec.ranges.push((u32::from(lo), u32::from(lo)));
            }
        }
        Ok(Ast::Class(spec))
    }

    fn group_name(&mut self, close: char) -> Result<String, Error> {
        let mut name = String::new();
        loop {
            match self.next() {
                Some(c) if c == close => return Ok(name),
                Some(c) => name.push(c),
                None => return unsupported("an unterminated group name"),
            }
        }
    }

    fn atom(&mut self) -> Result<Ast, Error> {
        let Some(c) = self.next() else {
            return Ok(Ast::Empty);
        };
        Ok(match c {
            '.' => Ast::Any,
            '^' => Ast::Start,
            '$' => Ast::End,
            '[' => self.class()?,
            '(' => self.group()?,
            '*' | '+' | '?' => return unsupported("nothing to repeat"),
            '\\' => {
                let Some(e) = self.next() else {
                    return unsupported("a trailing backslash");
                };
                match e {
                    'w' | 'W' | 'd' | 'D' | 's' | 'S' => {
                        let cat = match e {
                            'w' => Cat::Word,
                            'W' => Cat::NotWord,
                            'd' => Cat::Digit,
                            'D' => Cat::NotDigit,
                            's' => Cat::Space,
                            _ => Cat::NotSpace,
                        };
                        Ast::Class(ClassSpec {
                            negate: false,
                            ranges: Vec::new(),
                            cats: vec![cat],
                        })
                    }
                    'b' => Ast::WordB,
                    'B' => Ast::NotWordB,
                    'A' => Ast::StartAbs,
                    'Z' => Ast::EndAbs,
                    '1'..='9' => {
                        // a group reference (1-99), or an octal escape of three digits
                        let mut n = e.to_digit(10).unwrap_or(0);
                        if let Some(d) = self.peek().and_then(|d| d.to_digit(10)) {
                            let oct = e.is_digit(8)
                                && self.peek_at(1).and_then(|x| x.to_digit(8)).is_some()
                                && d < 8;
                            if oct {
                                return unsupported("an octal escape");
                            }
                            n = n * 10 + d;
                            self.i += 1;
                        }
                        let n = n as usize;
                        if n > self.groups {
                            return unsupported("a reference to an open or missing group");
                        }
                        Ast::Backref(n)
                    }
                    _ => match self.char_escape(e, false)? {
                        Some(ch) => Ast::Lit(ch),
                        None => return unsupported(format!("the escape \\{e}")),
                    },
                }
            }
            c => Ast::Lit(c),
        })
    }

    fn group(&mut self) -> Result<Ast, Error> {
        if !self.eat('?') {
            self.groups += 1;
            let idx = self.groups;
            let inner = self.alt()?;
            if !self.eat(')') {
                return unsupported("an unterminated group");
            }
            return Ok(Ast::Group(Some(idx), Box::new(inner)));
        }
        let Some(k) = self.next() else {
            return unsupported("an unterminated group");
        };
        let (ahead, negate) = match k {
            ':' => {
                let inner = self.alt()?;
                if !self.eat(')') {
                    return unsupported("an unterminated group");
                }
                return Ok(Ast::Group(None, Box::new(inner)));
            }
            'P' => {
                if self.eat('<') {
                    let name = self.group_name('>')?;
                    self.groups += 1;
                    let idx = self.groups;
                    self.names.push((name, idx));
                    let inner = self.alt()?;
                    if !self.eat(')') {
                        return unsupported("an unterminated group");
                    }
                    return Ok(Ast::Group(Some(idx), Box::new(inner)));
                }
                if self.eat('=') {
                    let name = self.group_name(')')?;
                    let Some(&(_, idx)) = self.names.iter().find(|(n, _)| *n == name) else {
                        return unsupported("an unknown group name");
                    };
                    return Ok(Ast::Backref(idx));
                }
                return unsupported("(?P");
            }
            '=' => (true, false),
            '!' => (true, true),
            '<' => match self.next() {
                Some('=') => (false, false),
                Some('!') => (false, true),
                Some(_) => {
                    // (?<name>...) is 3.11+? sre only accepts it from 3.12 on as an alias.
                    return unsupported("(?<name>");
                }
                None => return unsupported("an unterminated group"),
            },
            '#' => {
                while let Some(c) = self.next() {
                    if c == ')' {
                        return Ok(Ast::Empty);
                    }
                }
                return unsupported("an unterminated comment");
            }
            _ => return unsupported(format!("(?{k}")),
        };
        let inner = self.alt()?;
        if !self.eat(')') {
            return unsupported("an unterminated lookaround");
        }
        Ok(Ast::Look {
            ahead,
            negate,
            node: Box::new(inner),
        })
    }
}

// ---------------------------------------------------------------------------
// Compiler
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Inst {
    Char(char),
    /// IGNORECASE literal: `lower(c)` is `lo` or one of `extra`.
    CharI {
        lo: char,
        extra: [char; 2],
    },
    Set(u32),
    /// A single-character `inst` repeated `min..=max` times, without a backtrack entry per
    /// character (sre's `REPEAT_ONE` / `MIN_REPEAT_ONE`).
    RepeatOne {
        item: u32,
        min: u32,
        max: u32,
        greedy: bool,
    },
    Any,
    AnyAll,
    Start,
    StartLine,
    End,
    EndLine,
    EndAbs,
    WordB,
    NotWordB,
    Split(u32, u32),
    Jmp(u32),
    Save(u32),
    Backref {
        group: u32,
        ignore: bool,
    },
    /// A lookaround: run `body..` (it ends in `Match`) at the position (`width` characters
    /// back for a lookbehind), then continue at `next`.
    Look {
        ahead: bool,
        negate: bool,
        body: u32,
        next: u32,
        width: u32,
    },
    Match,
}

/// The characters that can start a match (a conservative filter).
#[derive(Clone, Copy, Debug)]
struct First {
    ascii: u128,
    non_ascii: bool,
}

impl First {
    const NONE: First = First {
        ascii: 0,
        non_ascii: false,
    };
    const ALL: First = First {
        ascii: u128::MAX,
        non_ascii: true,
    };

    fn union(self, o: First) -> First {
        First {
            ascii: self.ascii | o.ascii,
            non_ascii: self.non_ascii || o.non_ascii,
        }
    }
}

/// A compiled pattern.
#[derive(Clone, Debug)]
pub struct Regex {
    prog: Vec<Inst>,
    /// Single-character items of `RepeatOne` and the sets, by index.
    sets: Vec<Set>,
    items: Vec<Inst>,
    ngroups: usize,
    /// Characters a match can start with; `None` if it can be empty.
    first: Option<First>,
    version: UnicodeVersion,
    /// `re.IGNORECASE` was set.
    ignore: bool,
    /// The pattern text and the flags it was compiled with.
    source: (Box<str>, u32),
}

struct Compiler {
    prog: Vec<Inst>,
    sets: Vec<Set>,
    items: Vec<Inst>,
    flags: u32,
    v: UnicodeVersion,
    in_look: usize,
    look_groups: Vec<usize>,
    /// The pattern has backreferences, so groups record their spans.
    captures: bool,
}

impl Compiler {
    fn ignore(&self) -> bool {
        self.flags & FLAG_IGNORECASE != 0
    }

    fn pc(&self) -> u32 {
        u32::try_from(self.prog.len()).unwrap_or(u32::MAX)
    }

    fn lit(&self, c: char) -> Inst {
        if self.ignore() && pystr::is_cased(c, self.v) {
            let lo = pystr::simple_lower(c, self.v);
            let mut extra = ['\0'; 2];
            for (slot, x) in extra.iter_mut().zip(pystr::extra_cases(lo)) {
                *slot = x;
            }
            Inst::CharI { lo, extra }
        } else {
            Inst::Char(c)
        }
    }

    fn set(&mut self, spec: &ClassSpec) -> Result<Inst, Error> {
        let set = Set::compile(spec, self.ignore(), self.v)?;
        self.sets.push(set);
        Ok(Inst::Set(
            u32::try_from(self.sets.len() - 1).unwrap_or(u32::MAX),
        ))
    }

    /// The single-character instruction for `node`, if it is one.
    fn single(&mut self, node: &Ast) -> Result<Option<Inst>, Error> {
        Ok(match node {
            Ast::Lit(c) => Some(self.lit(*c)),
            Ast::Class(spec) => Some(self.set(spec)?),
            Ast::Any => Some(if self.flags & FLAG_DOTALL != 0 {
                Inst::AnyAll
            } else {
                Inst::Any
            }),
            Ast::Group(None, inner) => self.single(inner)?,
            _ => None,
        })
    }

    fn emit(&mut self, node: &Ast) -> Result<(), Error> {
        match node {
            Ast::Empty => {}
            Ast::Lit(_) | Ast::Class(_) | Ast::Any => {
                let inst = self.single(node)?.unwrap_or(Inst::Match);
                self.prog.push(inst);
            }
            Ast::Start => self.prog.push(if self.flags & FLAG_MULTILINE != 0 {
                Inst::StartLine
            } else {
                Inst::Start
            }),
            Ast::StartAbs => self.prog.push(Inst::Start),
            Ast::End => self.prog.push(if self.flags & FLAG_MULTILINE != 0 {
                Inst::EndLine
            } else {
                Inst::End
            }),
            Ast::EndAbs => self.prog.push(Inst::EndAbs),
            Ast::WordB => self.prog.push(Inst::WordB),
            Ast::NotWordB => self.prog.push(Inst::NotWordB),
            Ast::Group(idx, inner) => {
                if let (Some(i), true) = (idx, self.captures) {
                    if self.in_look > 0 {
                        self.look_groups.push(*i);
                    }
                    let slot = u32::try_from(2 * i).unwrap_or(u32::MAX);
                    self.prog.push(Inst::Save(slot));
                    self.emit(inner)?;
                    self.prog.push(Inst::Save(slot + 1));
                } else {
                    self.emit(inner)?;
                }
            }
            Ast::Concat(items) => {
                for it in items {
                    self.emit(it)?;
                }
            }
            Ast::Alt(branches) => {
                let mut jumps = Vec::new();
                for (k, b) in branches.iter().enumerate() {
                    if k + 1 < branches.len() {
                        let split = self.prog.len();
                        self.prog.push(Inst::Split(0, 0));
                        self.emit(b)?;
                        jumps.push(self.prog.len());
                        self.prog.push(Inst::Jmp(0));
                        let next = self.pc();
                        self.prog[split] = Inst::Split(u32::try_from(split + 1).unwrap_or(0), next);
                    } else {
                        self.emit(b)?;
                    }
                }
                let end = self.pc();
                for j in jumps {
                    self.prog[j] = Inst::Jmp(end);
                }
            }
            Ast::Repeat {
                node,
                min,
                max,
                greedy,
            } => self.repeat(node, *min, *max, *greedy)?,
            Ast::Look {
                ahead,
                negate,
                node,
            } => {
                let width = if *ahead {
                    0
                } else {
                    match width(node) {
                        Some(w) => u32::try_from(w).unwrap_or(u32::MAX),
                        None => return unsupported("a variable-width lookbehind"),
                    }
                };
                let at = self.prog.len();
                self.prog.push(Inst::Match);
                let body = self.pc();
                self.in_look += 1;
                self.emit(node)?;
                self.in_look -= 1;
                self.prog.push(Inst::Match);
                let next = self.pc();
                self.prog[at] = Inst::Look {
                    ahead: *ahead,
                    negate: *negate,
                    body,
                    next,
                    width,
                };
            }
            Ast::Backref(n) => {
                if self.look_groups.contains(n) {
                    return unsupported("a reference to a group in a lookaround");
                }
                let ignore = self.ignore();
                self.prog.push(Inst::Backref {
                    group: u32::try_from(*n).unwrap_or(u32::MAX),
                    ignore,
                });
            }
        }
        Ok(())
    }

    fn repeat(
        &mut self,
        node: &Ast,
        min: u32,
        max: Option<u32>,
        greedy: bool,
    ) -> Result<(), Error> {
        if let Some(item) = self.single(node)? {
            self.items.push(item);
            let item = u32::try_from(self.items.len() - 1).unwrap_or(u32::MAX);
            self.prog.push(Inst::RepeatOne {
                item,
                min,
                max: max.unwrap_or(u32::MAX),
                greedy,
            });
            return Ok(());
        }
        if nullable(node) && max != Some(min) && max.is_none_or(|m| m > 1) {
            return unsupported("a repeated sub-pattern that can match the empty string");
        }
        for _ in 0..min {
            self.emit(node)?;
        }
        match max {
            None => {
                // L: split(body, exit); body; jmp L
                let l = self.prog.len();
                self.prog.push(Inst::Split(0, 0));
                self.emit(node)?;
                self.prog.push(Inst::Jmp(u32::try_from(l).unwrap_or(0)));
                let exit = self.pc();
                let body = u32::try_from(l + 1).unwrap_or(0);
                self.prog[l] = if greedy {
                    Inst::Split(body, exit)
                } else {
                    Inst::Split(exit, body)
                };
            }
            Some(max) => {
                let mut splits = Vec::new();
                for _ in min..max {
                    splits.push(self.prog.len());
                    self.prog.push(Inst::Split(0, 0));
                    self.emit(node)?;
                }
                let exit = self.pc();
                for s in splits {
                    let body = u32::try_from(s + 1).unwrap_or(0);
                    self.prog[s] = if greedy {
                        Inst::Split(body, exit)
                    } else {
                        Inst::Split(exit, body)
                    };
                }
            }
        }
        if self.prog.len() > 100_000 {
            return unsupported("a pattern too large to expand");
        }
        Ok(())
    }
}

fn has_backref(node: &Ast) -> bool {
    match node {
        Ast::Backref(_) => true,
        Ast::Group(_, inner) => has_backref(inner),
        Ast::Concat(items) | Ast::Alt(items) => items.iter().any(has_backref),
        Ast::Repeat { node, .. } | Ast::Look { node, .. } => has_backref(node),
        _ => false,
    }
}

fn has_look_group(node: &Ast, in_look: bool) -> bool {
    match node {
        Ast::Group(idx, inner) => (in_look && idx.is_some()) || has_look_group(inner, in_look),
        Ast::Concat(items) | Ast::Alt(items) => items.iter().any(|x| has_look_group(x, in_look)),
        Ast::Repeat { node, .. } => has_look_group(node, in_look),
        Ast::Look { node, .. } => has_look_group(node, true),
        _ => false,
    }
}

fn nullable(node: &Ast) -> bool {
    match node {
        Ast::Empty
        | Ast::Start
        | Ast::StartAbs
        | Ast::End
        | Ast::EndAbs
        | Ast::WordB
        | Ast::NotWordB
        | Ast::Look { .. }
        | Ast::Backref(_) => true,
        Ast::Lit(_) | Ast::Class(_) | Ast::Any => false,
        Ast::Group(_, inner) => nullable(inner),
        Ast::Concat(items) => items.iter().all(nullable),
        Ast::Alt(branches) => branches.iter().any(nullable),
        Ast::Repeat { node, min, .. } => *min == 0 || nullable(node),
    }
}

/// The fixed width (in characters) of `node`, if it has one.
fn width(node: &Ast) -> Option<usize> {
    match node {
        Ast::Empty
        | Ast::Start
        | Ast::StartAbs
        | Ast::End
        | Ast::EndAbs
        | Ast::WordB
        | Ast::NotWordB
        | Ast::Look { .. } => Some(0),
        Ast::Lit(_) | Ast::Class(_) | Ast::Any => Some(1),
        Ast::Group(_, inner) => width(inner),
        Ast::Concat(items) => items.iter().map(width).sum(),
        Ast::Alt(branches) => {
            let w = width(&branches[0])?;
            branches.iter().all(|b| width(b) == Some(w)).then_some(w)
        }
        Ast::Repeat { node, min, max, .. } => {
            if *max == Some(*min) {
                Some(width(node)? * (*min as usize))
            } else {
                None
            }
        }
        Ast::Backref(_) => None,
    }
}

impl Regex {
    /// Compile `pattern` with `re` flags `flags` for Unicode version `v`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for syntax or flags this engine does not reproduce.
    pub fn new(pattern: &str, flags: u32, v: UnicodeVersion) -> Result<Regex, Error> {
        let mut flags = flags;
        if flags & (FLAG_LOCALE | FLAG_VERBOSE | FLAG_ASCII | FLAG_DEBUG) != 0 {
            return unsupported("the flags LOCALE, VERBOSE, ASCII or DEBUG");
        }
        let mut p = Parser {
            chars: pattern.chars().collect(),
            i: 0,
            groups: 0,
            names: Vec::new(),
            _src: pattern,
        };
        // Global inline flags at the start, e.g. `(?i)`.
        while p.peek() == Some('(') && p.peek_at(1) == Some('?') {
            let save = p.i;
            p.i += 2;
            let mut f = 0;
            let mut ok = false;
            while let Some(c) = p.next() {
                match c {
                    'i' => f |= FLAG_IGNORECASE,
                    'm' => f |= FLAG_MULTILINE,
                    's' => f |= FLAG_DOTALL,
                    'u' => f |= FLAG_UNICODE,
                    ')' => {
                        ok = true;
                        break;
                    }
                    _ => break,
                }
            }
            if ok {
                flags |= f;
            } else {
                p.i = save;
                break;
            }
        }
        let ast = p.alt()?;
        if p.i != p.chars.len() {
            return unsupported("unbalanced parenthesis");
        }
        let mut c = Compiler {
            prog: Vec::new(),
            sets: Vec::new(),
            items: Vec::new(),
            flags,
            v,
            in_look: 0,
            look_groups: Vec::new(),
            captures: has_backref(&ast),
        };
        if c.captures && has_look_group(&ast, false) {
            return unsupported("a group inside a lookaround of a pattern with backreferences");
        }
        c.emit(&ast)?;
        c.prog.push(Inst::Match);
        let mut re = Regex {
            prog: c.prog,
            sets: c.sets,
            items: c.items,
            ngroups: p.groups,
            first: None,
            version: v,
            ignore: flags & FLAG_IGNORECASE != 0,
            source: (pattern.into(), flags),
        };
        re.first = re.first_set();
        Ok(re)
    }

    /// The pattern text and the flags given to [`Regex::new`].
    pub fn source(&self) -> (&str, u32) {
        (&self.source.0, self.source.1)
    }

    /// The Unicode version the pattern was compiled for.
    pub fn version(&self) -> UnicodeVersion {
        self.version
    }

    fn inst_first(&self, inst: &Inst) -> First {
        match inst {
            Inst::Char(c) => {
                let cp = u32::from(*c);
                if cp < 0x80 {
                    First {
                        ascii: 1u128 << cp,
                        non_ascii: false,
                    }
                } else {
                    First {
                        ascii: 0,
                        non_ascii: true,
                    }
                }
            }
            Inst::CharI { .. } => {
                let mut f = First::NONE;
                for cp in 0..128u32 {
                    let ch = char::from_u32(cp).unwrap_or('\0');
                    if self.test_single(inst, ch) {
                        f.ascii |= 1u128 << cp;
                    }
                }
                f.non_ascii = true;
                f
            }
            Inst::Set(i) => {
                let set = &self.sets[*i as usize];
                let mut f = First::NONE;
                for cp in 0..128u32 {
                    let ch = char::from_u32(cp).unwrap_or('\0');
                    if set.test(ch, self.version) {
                        f.ascii |= 1u128 << cp;
                    }
                }
                f.non_ascii = set.may_be_non_ascii();
                f
            }
            _ => First::ALL,
        }
    }

    /// The characters a match can start with (`None`: it can be empty, or unknown).
    fn first_set(&self) -> Option<First> {
        let mut seen = vec![false; self.prog.len()];
        let mut todo = vec![0usize];
        let mut f = First::NONE;
        while let Some(pc) = todo.pop() {
            if pc >= self.prog.len() || seen[pc] {
                continue;
            }
            seen[pc] = true;
            match &self.prog[pc] {
                Inst::Char(_) | Inst::CharI { .. } | Inst::Set(_) | Inst::Any | Inst::AnyAll => {
                    f = f.union(self.inst_first(&self.prog[pc]));
                }
                Inst::RepeatOne { item, min, .. } => {
                    f = f.union(self.inst_first(&self.items[*item as usize]));
                    if *min == 0 {
                        todo.push(pc + 1);
                    }
                }
                Inst::Split(a, b) => {
                    todo.push(*a as usize);
                    todo.push(*b as usize);
                }
                Inst::Jmp(a) => todo.push(*a as usize),
                Inst::Look { next, .. } => todo.push(*next as usize),
                Inst::Save(_)
                | Inst::Start
                | Inst::StartLine
                | Inst::End
                | Inst::EndLine
                | Inst::EndAbs
                | Inst::WordB
                | Inst::NotWordB => todo.push(pc + 1),
                Inst::Backref { .. } | Inst::Match => return None,
            }
        }
        Some(f)
    }

    /// Whether a match could start with `c` (a fast pre-check; `true` when unsure).
    #[inline]
    pub fn may_start_with(&self, c: Option<char>) -> bool {
        match (self.first, c) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some(f), Some(c)) => {
                let cp = u32::from(c);
                if cp < 0x80 {
                    f.ascii & (1u128 << cp) != 0
                } else {
                    f.non_ascii
                }
            }
        }
    }

    #[inline]
    fn test_single(&self, inst: &Inst, c: char) -> bool {
        match inst {
            Inst::Char(x) => c == *x,
            Inst::CharI { lo, extra } => {
                let l = pystr::simple_lower(c, self.version);
                l == *lo
                    || (extra[0] != '\0' && l == extra[0])
                    || (extra[1] != '\0' && l == extra[1])
            }
            Inst::Set(i) => self.sets[*i as usize].test(c, self.version),
            Inst::Any => c != '\n',
            Inst::AnyAll => true,
            _ => false,
        }
    }

    /// `pattern.match(text, pos)`: the end of the match starting at byte offset `pos`.
    pub fn match_at(&self, text: &str, pos: usize) -> Option<usize> {
        self.match_with(text, pos, &mut Scratch::default())
    }

    /// [`Regex::match_at`] reusing `scratch`'s buffers.
    pub fn match_with(&self, text: &str, pos: usize, scratch: &mut Scratch) -> Option<usize> {
        scratch.caps.clear();
        scratch.caps.resize(2 * (self.ngroups + 1), usize::MAX);
        scratch.stack.clear();
        self.exec(text, 0, pos, &mut scratch.caps, &mut scratch.stack)
    }

    fn word_at(&self, text: &str, pos: usize) -> bool {
        text[pos..]
            .chars()
            .next()
            .is_some_and(|c| pystr::is_word(c, self.version))
    }

    fn word_before(&self, text: &str, pos: usize) -> bool {
        text[..pos]
            .chars()
            .next_back()
            .is_some_and(|c| pystr::is_word(c, self.version))
    }

    #[allow(clippy::too_many_lines)]
    fn exec(
        &self,
        text: &str,
        pc0: usize,
        pos0: usize,
        caps: &mut [usize],
        stack: &mut Vec<Bt>,
    ) -> Option<usize> {
        let base = stack.len();
        let bytes = text.as_bytes();
        let (mut pc, mut pos) = (pc0, pos0);
        loop {
            let ok = match &self.prog[pc] {
                Inst::Char(c) => {
                    let c = *c;
                    if c.is_ascii() {
                        if bytes.get(pos) == Some(&(c as u8)) {
                            pos += 1;
                            pc += 1;
                            true
                        } else {
                            false
                        }
                    } else if text[pos..].starts_with(c) {
                        pos += c.len_utf8();
                        pc += 1;
                        true
                    } else {
                        false
                    }
                }
                inst @ (Inst::CharI { .. } | Inst::Set(_) | Inst::Any | Inst::AnyAll) => {
                    match text[pos..].chars().next() {
                        Some(c) if self.test_single(inst, c) => {
                            pos += c.len_utf8();
                            pc += 1;
                            true
                        }
                        _ => false,
                    }
                }
                Inst::RepeatOne {
                    item,
                    min,
                    max,
                    greedy,
                } => {
                    let item = &self.items[*item as usize];
                    let (min, max) = (*min, *max);
                    if *greedy {
                        let start = pos;
                        let mut n = 0u32;
                        let mut end = pos;
                        while n < max {
                            match text[end..].chars().next() {
                                Some(c) if self.test_single(item, c) => {
                                    end += c.len_utf8();
                                    n += 1;
                                }
                                _ => break,
                            }
                        }
                        if n < min {
                            false
                        } else {
                            if n > min {
                                stack.push(Bt::Greedy {
                                    pc: pc + 1,
                                    start,
                                    pos: end,
                                    count: n,
                                    min,
                                });
                            }
                            pos = end;
                            pc += 1;
                            true
                        }
                    } else {
                        let mut n = 0u32;
                        let mut end = pos;
                        let mut ok = true;
                        while n < min {
                            match text[end..].chars().next() {
                                Some(c) if self.test_single(item, c) => {
                                    end += c.len_utf8();
                                    n += 1;
                                }
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok {
                            if n < max {
                                stack.push(Bt::Lazy {
                                    pc: pc + 1,
                                    item: pc,
                                    pos: end,
                                    count: n,
                                    max,
                                });
                            }
                            pos = end;
                            pc += 1;
                        }
                        ok
                    }
                }
                Inst::Start => {
                    pc += 1;
                    pos == 0
                }
                Inst::StartLine => {
                    pc += 1;
                    pos == 0 || bytes[pos - 1] == b'\n'
                }
                Inst::End => {
                    pc += 1;
                    pos == text.len() || (pos + 1 == text.len() && bytes[pos] == b'\n')
                }
                Inst::EndLine => {
                    pc += 1;
                    pos == text.len() || bytes[pos] == b'\n'
                }
                Inst::EndAbs => {
                    pc += 1;
                    pos == text.len()
                }
                Inst::WordB | Inst::NotWordB => {
                    // sre: no boundary (and no non-boundary) in an empty string
                    let ok = !text.is_empty()
                        && ((self.word_before(text, pos) != self.word_at(text, pos))
                            == matches!(self.prog[pc], Inst::WordB));
                    pc += 1;
                    ok
                }
                Inst::Split(a, b) => {
                    stack.push(Bt::Alt {
                        pc: *b as usize,
                        pos,
                    });
                    pc = *a as usize;
                    true
                }
                Inst::Jmp(a) => {
                    pc = *a as usize;
                    true
                }
                Inst::Save(slot) => {
                    let slot = *slot as usize;
                    stack.push(Bt::Cap {
                        slot,
                        old: caps[slot],
                    });
                    caps[slot] = pos;
                    pc += 1;
                    true
                }
                Inst::Backref { group, ignore } => {
                    let g = *group as usize;
                    let (s, e) = (caps[2 * g], caps[2 * g + 1]);
                    if s == usize::MAX || e == usize::MAX {
                        false
                    } else {
                        let want = &text[s..e];
                        let mut p = pos;
                        let mut ok = true;
                        for wc in want.chars() {
                            match text[p..].chars().next() {
                                Some(tc)
                                    if tc == wc
                                        || (*ignore
                                            && pystr::simple_lower(tc, self.version)
                                                == pystr::simple_lower(wc, self.version)) =>
                                {
                                    p += tc.len_utf8();
                                }
                                _ => {
                                    ok = false;
                                    break;
                                }
                            }
                        }
                        if ok {
                            pos = p;
                            pc += 1;
                        }
                        ok
                    }
                }
                Inst::Look {
                    ahead,
                    negate,
                    body,
                    next,
                    width,
                } => {
                    let start = if *ahead {
                        Some(pos)
                    } else {
                        let mut p = pos;
                        let mut ok = true;
                        for _ in 0..*width {
                            if let Some(c) = text[..p].chars().next_back() {
                                p -= c.len_utf8();
                            } else {
                                ok = false;
                                break;
                            }
                        }
                        ok.then_some(p)
                    };
                    let (body, next, ahead, negate) = (*body, *next, *ahead, *negate);
                    let hit = match start {
                        None => false,
                        Some(p) => {
                            // atomic: the body's backtrack entries are gone once it returns
                            let r = self.exec(text, body as usize, p, caps, stack);
                            // a lookbehind's (fixed-width) body must end where the assertion is
                            match r {
                                Some(end) if !ahead => end == pos,
                                Some(_) => true,
                                None => false,
                            }
                        }
                    };
                    pc = next as usize;
                    hit != negate
                }
                Inst::Match => {
                    stack.truncate(base);
                    return Some(pos);
                }
            };
            if ok {
                continue;
            }
            // backtrack
            loop {
                if stack.len() == base {
                    return None;
                }
                match stack.pop() {
                    None => return None,
                    Some(Bt::Alt { pc: p, pos: q }) => {
                        pc = p;
                        pos = q;
                        break;
                    }
                    Some(Bt::Cap { slot, old }) => caps[slot] = old,
                    Some(Bt::Greedy {
                        pc: p,
                        start,
                        pos: q,
                        count,
                        min,
                    }) => {
                        // give back one character
                        let mut back = q;
                        if let Some(c) = text[start..q].chars().next_back() {
                            back -= c.len_utf8();
                        }
                        let count = count - 1;
                        if count > min {
                            stack.push(Bt::Greedy {
                                pc: p,
                                start,
                                pos: back,
                                count,
                                min,
                            });
                        }
                        pc = p;
                        pos = back;
                        break;
                    }
                    Some(Bt::Lazy {
                        pc: p,
                        item,
                        pos: q,
                        count,
                        max,
                    }) => {
                        let Inst::RepeatOne { item: it, .. } = self.prog[item] else {
                            return None;
                        };
                        let inst = &self.items[it as usize];
                        match text[q..].chars().next() {
                            Some(c) if self.test_single(inst, c) => {
                                let np = q + c.len_utf8();
                                let count = count + 1;
                                if count < max {
                                    stack.push(Bt::Lazy {
                                        pc: p,
                                        item,
                                        pos: np,
                                        count,
                                        max,
                                    });
                                }
                                pc = p;
                                pos = np;
                                break;
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }

    /// Whether IGNORECASE is set.
    pub fn ignorecase(&self) -> bool {
        self.ignore
    }
}

/// Reusable buffers for [`Regex::match_with`].
#[derive(Debug, Default)]
pub struct Scratch {
    caps: Vec<usize>,
    stack: Vec<Bt>,
}

#[derive(Clone, Copy, Debug)]
enum Bt {
    Alt {
        pc: usize,
        pos: usize,
    },
    Cap {
        slot: usize,
        old: usize,
    },
    Greedy {
        pc: usize,
        start: usize,
        pos: usize,
        count: u32,
        min: u32,
    },
    Lazy {
        pc: usize,
        item: usize,
        pos: usize,
        count: u32,
        max: u32,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::LATEST;
    const F: u32 = FLAG_IGNORECASE | FLAG_UNICODE;

    fn m(p: &str, s: &str, pos: usize) -> Option<usize> {
        Regex::new(p, F, V).expect(p).match_at(s, pos)
    }

    #[test]
    fn basics() {
        assert_eq!(m(r"\s+?", "   x", 0), Some(1));
        assert_eq!(m(r"(--|# ).*?(\r\n|\r|\n|$)", "-- a\r\nb", 0), Some(6));
        assert_eq!(m(r"(--|# ).*?(\r\n|\r|\n|$)", "-- abc", 0), Some(6));
        assert_eq!(m(r"(CASE|IN|VALUES|USING|FROM|AS)\b", "insert", 0), None);
        assert_eq!(m(r"(CASE|IN|VALUES|USING|FROM|AS)\b", "in (", 0), Some(2));
        assert_eq!(
            m(r"(CASE|IN|VALUES|USING|FROM|AS)\b", "ca\u{17f}e x", 0),
            Some(5)
        );
        assert_eq!(m(r"[A-ZÀ-Ü]\w*(?=\s*\.(?!\d))", "a.b", 0), Some(1));
        assert_eq!(m(r"[A-ZÀ-Ü]\w*(?=\s*\.(?!\d))", "a.1", 0), None);
        assert_eq!(m(r"(?<=\.)[A-ZÀ-Ü]\w*", "a.b", 2), Some(3));
        assert_eq!(m(r"(?<!\w)[$:?]\w+", "a:b", 1), None);
        assert_eq!(m(r"(?<!\w)[$:?]\w+", " :b", 1), Some(3));
        assert_eq!(
            m(r"(?![_A-ZÀ-Ü])-?(\d+(\.\d*)|\.\d+)(?![_A-ZÀ-Ü])", "1.5a", 0),
            Some(2)
        );
        assert_eq!(m(r"(?![_A-ZÀ-Ü])-?\d+(?![_A-ZÀ-Ü])", "123a", 0), Some(2));
        assert_eq!(m(r"'(''|\\'|[^'])*'", r"'a\'", 0), Some(4));
        assert_eq!(m(r"`(``|[^`])*`", "`a``", 0), Some(3));
        assert_eq!(
            m(
                r"((?<![\w\$])\$(?:[_A-ZÀ-Ü]\w*)?\$)[\s\S]*?\1",
                "$a$ x $a$",
                0
            ),
            Some(9)
        );
        assert_eq!(m(r"[A-ZÀ-Ü]", "\u{d7}", 0), Some(2));
        assert_eq!(m(r"[A-ZÀ-Ü]", "\u{f7}", 0), None);
        assert_eq!(m(r"k", "\u{212a}", 0), Some(3));
        assert_eq!(m(r"x{2,3}", "xxxx", 0), Some(3));
        assert_eq!(m(r"x{2,3}?", "xxxx", 0), Some(2));
        assert_eq!(m(r"a$", "a\n", 0), Some(1));
        assert_eq!(m("s", "\0", 0), None);
        assert_eq!(m("s", "\u{17f}", 0), Some(2));
        assert_eq!(
            Regex::new(r"\A", FLAG_MULTILINE | FLAG_UNICODE, V)
                .expect("ok")
                .match_at("a\nb", 2),
            None
        );
    }

    #[test]
    fn unsupported_syntax() {
        assert!(Regex::new(r"(a*)*", F, V).is_err());
        assert!(Regex::new(r"a*+", F, V).is_err());
        assert!(Regex::new(r"(?>a)", F, V).is_err());
        assert!(Regex::new(r"a", F | FLAG_VERBOSE, V).is_err());
    }
}
