// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! Marks and errors, as PyYAML's `error.py`, `reader.py` and the `*Error` classes build them.
//!
//! Messages stay unformatted: PyYAML formats them with Python's `%` operator (`%r` of a str
//! uses `repr`, which needs the interpreter's `str.isprintable`), so the bindings format
//! [`Msg::fmt`] with [`Msg::args`] in Python and get the same text.

use std::fmt::{self, Write as _};

/// A position in the stream: PyYAML's `Mark` without `name` and `buffer` (the bindings add
/// them). `index` counts code points from the start of the decoded stream; `line` and
/// `column` are 0-based, counted as PyYAML's `Reader.forward` counts them. For `str` and
/// `bytes` input PyYAML's `pointer` equals `index`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Mark {
    pub index: usize,
    pub line: usize,
    pub column: usize,
}

/// One `%` argument of a [`Msg`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Arg {
    /// A `str` (formatted by `%r` or `%s`, whichever the format has).
    Str(String),
    /// An `int` (`%d`).
    Int(i64),
    /// `str(exc)` of the `UnicodeDecodeError` that `bytes(..).decode('utf-8')` raises for these
    /// bytes (a `%s`).
    Utf8DecodeError(Vec<u8>),
}

/// A message as PyYAML writes it: `fmt % tuple(args)`, or `fmt` itself when `args` is empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Msg {
    pub fmt: &'static str,
    pub args: Vec<Arg>,
}

impl Msg {
    /// A message without arguments.
    pub const fn text(fmt: &'static str) -> Self {
        Self {
            fmt,
            args: Vec::new(),
        }
    }

    /// `fmt % (arg,)` with one `str` argument.
    pub fn str(fmt: &'static str, arg: impl Into<String>) -> Self {
        Self {
            fmt,
            args: vec![Arg::Str(arg.into())],
        }
    }

    /// `fmt % (arg,)` with one `char` argument (PyYAML's `%r % ch`).
    pub fn ch(fmt: &'static str, ch: char) -> Self {
        Self {
            fmt,
            args: vec![Arg::Str(ch.to_string())],
        }
    }

    /// `fmt % (n,)` with one `int` argument.
    pub fn int(fmt: &'static str, n: i64) -> Self {
        Self {
            fmt,
            args: vec![Arg::Int(n)],
        }
    }

    /// A rough rendering for Rust tests and debugging: `%r` as `'..'`, no escaping.
    pub fn render(&self) -> String {
        let mut out = String::new();
        let mut args = self.args.iter();
        let mut chars = self.fmt.chars().peekable();
        while let Some(c) = chars.next() {
            if c != '%' {
                out.push(c);
                continue;
            }
            match chars.next() {
                Some('%') | None => out.push('%'),
                Some(spec) => match args.next() {
                    Some(Arg::Str(s)) if spec == 'r' => {
                        out.push('\'');
                        out.push_str(s);
                        out.push('\'');
                    }
                    Some(Arg::Str(s)) => out.push_str(s),
                    Some(Arg::Int(n)) => out.push_str(&n.to_string()),
                    Some(Arg::Utf8DecodeError(b)) => {
                        let _ = write!(out, "<decode error {b:?}>");
                    }
                    None => out.push('?'),
                },
            }
        }
        out
    }
}

/// Which `MarkedYAMLError` subclass PyYAML raises.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    Scanner,
    Parser,
    Composer,
    Constructor,
}

impl ErrorKind {
    /// The PyYAML class name.
    pub const fn class_name(self) -> &'static str {
        match self {
            Self::Scanner => "ScannerError",
            Self::Parser => "ParserError",
            Self::Composer => "ComposerError",
            Self::Constructor => "ConstructorError",
        }
    }
}

/// A `MarkedYAMLError(context, context_mark, problem, problem_mark)`: PyYAML always passes
/// these four positionally (the `note` is never set on the safe path).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MarkedError {
    pub kind: ErrorKind,
    pub context: Option<Msg>,
    pub context_mark: Option<Mark>,
    pub problem: Option<Msg>,
    pub problem_mark: Option<Mark>,
}

/// A `ReaderError(name, position, character, encoding, reason)`. `character` is an `int` in
/// both of PyYAML's cases (a code point, or the byte `raw_buffer[exc.start]`), so its
/// `__str__` always takes the "unacceptable character #x%04x" branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReaderError {
    pub position: usize,
    pub character: u32,
    pub encoding: &'static str,
    pub reason: &'static str,
}

/// Why loading stopped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum YamlError {
    Reader(ReaderError),
    Marked(Box<MarkedError>),
    /// A Python `ValueError` PyYAML lets escape (e.g. `chr()` of an escape past U+10FFFF).
    Value(String),
    /// A Python `TypeError` PyYAML lets escape (`ord()` of a byte in the emitter's tag
    /// prefix escaping).
    Type(String),
    /// A Python `OverflowError` PyYAML lets escape (`chr()` of an escape past 0x7FFFFFFF).
    Overflow(String),
    /// The stream needed characters past a pending reader error (file objects): the bindings
    /// raise that error.
    NeedMore,
    /// An `EmitterError(message)`.
    Emitter(Msg),
    /// Nesting past the depth limit: Python's `RecursionError`, which PyYAML's recursive
    /// composer and representer hit.
    Recursion,
}

impl YamlError {
    pub fn marked(
        kind: ErrorKind,
        context: Option<Msg>,
        context_mark: Option<Mark>,
        problem: Msg,
        problem_mark: Option<Mark>,
    ) -> Self {
        Self::Marked(Box::new(MarkedError {
            kind,
            context,
            context_mark,
            problem: Some(problem),
            problem_mark,
        }))
    }
}

impl fmt::Display for YamlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reader(e) => write!(
                f,
                "unacceptable character #x{:04x}: {} (position {})",
                e.character, e.reason, e.position
            ),
            Self::Marked(e) => {
                write!(f, "{}:", e.kind.class_name())?;
                if let Some(c) = &e.context {
                    write!(f, " {}", c.render())?;
                }
                if let Some(m) = e.context_mark {
                    write!(f, " [{}:{}]", m.line + 1, m.column + 1)?;
                }
                if let Some(p) = &e.problem {
                    write!(f, " {}", p.render())?;
                }
                if let Some(m) = e.problem_mark {
                    write!(f, " [{}:{}]", m.line + 1, m.column + 1)?;
                }
                Ok(())
            }
            Self::Value(msg) => write!(f, "ValueError: {msg}"),
            Self::Overflow(msg) => write!(f, "OverflowError: {msg}"),
            Self::Type(msg) => write!(f, "TypeError: {msg}"),
            Self::NeedMore => f.write_str("reader error pending"),
            Self::Emitter(m) => write!(f, "EmitterError: {}", m.render()),
            Self::Recursion => f.write_str("RecursionError: maximum recursion depth exceeded"),
        }
    }
}

impl std::error::Error for YamlError {}

pub type Result<T> = std::result::Result<T, YamlError>;
