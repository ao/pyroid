// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! The `FilterStack.run` pipelines behind `format`, `split` and `parse`: lex, the token
//! filters, the statement splitter, then per statement grouping, the statement filters, the
//! output filter and the serializer.

use super::filters::{self, Depth, Item, Output, TokenFilters};
use super::grouping::{Grouper, Limits};
use super::lexer::{self, Config, LexError};
use super::reindent::{Aligned, Reindent, ReindentOptions};
use super::tree::{Class, Fail, Id, Str, Tree};
use super::ttype::{self, Types, bits};
use crate::ftfy::UnicodeVersion;

/// Why a pipeline stopped.
#[derive(Debug)]
pub enum Error<E> {
    /// `SQLParseError(message)`.
    SqlParse(String),
    /// sqlparse's Python code raises somewhere on this input: run it for the exception.
    Python,
    /// `NotImplementedError` (`right_margin`).
    NotImplemented,
    /// A lexer entry matched the empty string (`ValueError` from `itertools.islice`).
    EmptyMatch,
    /// The caller's matcher or progress check failed.
    External(E),
}

impl<E> From<Fail> for Error<E> {
    fn from(f: Fail) -> Self {
        match f {
            Fail::Python => Error::Python,
            Fail::SqlParse(m) => Error::SqlParse(m),
        }
    }
}

/// The caller's hooks: external regex entries and the Ctrl-C check ([`External`]).
pub use super::lexer::External as Hooks;

/// No external entries and no checks.
pub use super::lexer::NoExternal as NoHooks;

/// `format()`'s options, after `formatter.validate_options`.
#[derive(Clone, Debug, Default)]
pub struct FormatOptions {
    /// The preprocess filters.
    pub tokens: TokenFilters,
    /// `use_space_around_operators`.
    pub spaces_around_operators: bool,
    /// `strip_comments`.
    pub strip_comments: bool,
    /// `strip_whitespace or reindent`.
    pub strip_whitespace: bool,
    /// `reindent`.
    pub reindent: Option<ReindentOptions>,
    /// `reindent_aligned` (with `indent_char`).
    pub aligned: Option<char>,
    /// `right_margin` was given (its filter raises `NotImplementedError`).
    pub right_margin: bool,
    /// `output_format`.
    pub output: Option<Output>,
}

impl FormatOptions {
    fn grouping(&self) -> bool {
        self.spaces_around_operators
            || self.strip_comments
            || self.strip_whitespace
            || self.reindent.is_some()
            || self.aligned.is_some()
            || self.right_margin
    }
}

/// The lexed, filtered items of a text, handed out one statement at a time.
#[allow(missing_debug_implementations)]
pub struct Stream<S, E> {
    /// The text.
    pub src: S,
    /// The items.
    pub items: Vec<Item>,
    lex_error: Option<LexError<E>>,
    pos: usize,
    start: usize,
    splitter: super::splitter::Splitter,
    done: bool,
    types: Types,
    version: UnicodeVersion,
}

/// Lex `src` and apply the token filters.
///
/// # Errors
/// [`Error::Python`] where a token filter raises.
pub fn stream<S: AsRef<str>, H: Hooks>(
    src: S,
    cfg: &Config,
    filters: &TokenFilters,
    hooks: &mut H,
) -> Result<Stream<S, H::Error>, Error<H::Error>> {
    let (raw, err) = lexer::lex(src.as_ref(), cfg, hooks);
    let err = match err {
        Some(LexError::Interrupted(e)) => return Err(Error::External(e)),
        e => e,
    };
    let types = ttype::types();
    let v = cfg.version();
    let mut items: Vec<Item> = raw
        .iter()
        .map(|t| (t.tt, Str::Span(t.start, t.end)))
        .collect();
    if !filters.is_empty() {
        for it in &mut items {
            filters.apply(src.as_ref(), &types, v, it)?;
        }
    }
    Ok(Stream {
        src,
        items,
        lex_error: err,
        pos: 0,
        start: 0,
        splitter: super::splitter::Splitter::new(v),
        done: false,
        types,
        version: v,
    })
}

impl<S: AsRef<str>, E> Stream<S, E> {
    fn text<'a>(&'a self, s: &'a Str) -> &'a str {
        match s {
            Str::Span(a, b) => &self.src.as_ref()[*a as usize..*b as usize],
            Str::Owned(o) => o,
        }
    }

    /// The next statement's item range, or the lexer's error once the items run out.
    ///
    /// # Errors
    /// The splitter's [`Error::Python`], or the lexer's error after the last statement it
    /// completed.
    pub fn next_statement(&mut self) -> Result<Option<std::ops::Range<usize>>, Error<E>> {
        if self.done {
            return Ok(None);
        }
        while self.pos < self.items.len() {
            let tt = self.items[self.pos].0;
            let ws = self.types.has(tt, bits::WHITESPACE);
            let value: &str = match &self.items[self.pos].1 {
                Str::Span(a, b) => &self.src.as_ref()[*a as usize..*b as usize],
                Str::Owned(o) => o,
            };
            let split_before = self.splitter.push(&self.types, tt, value, ws)?;
            let i = self.pos;
            self.pos += 1;
            if split_before {
                let r = self.start..i;
                self.start = i;
                return Ok(Some(r));
            }
        }
        self.done = true;
        if let Some(e) = self.lex_error.take() {
            return Err(match e {
                LexError::EmptyMatch => Error::EmptyMatch,
                LexError::External(x) | LexError::Interrupted(x) => Error::External(x),
            });
        }
        if self.splitter.finish() {
            return Ok(Some(self.start..self.items.len()));
        }
        Ok(None)
    }

    /// `sql.Statement(tokens)` for `range`, in a new arena.
    pub fn statement(&self, range: std::ops::Range<usize>) -> (Tree<'_>, Id) {
        let mut t = Tree::new(self.src.as_ref(), self.types.clone(), self.version);
        let mut ids = Vec::with_capacity(range.len());
        for (tt, val) in &self.items[range] {
            ids.push(t.token(*tt, val.clone()));
        }
        let stmt = t.group(Class::Statement, ids);
        (t, stmt)
    }

    /// The item values of `range`: `(is_whitespace, value)`.
    pub fn values(&self, range: std::ops::Range<usize>) -> impl Iterator<Item = (bool, &str)> + '_ {
        self.items[range]
            .iter()
            .map(|(tt, s)| (self.types.has(*tt, bits::WHITESPACE), self.text(s)))
    }

    /// The span `src[a..b]` covering `range`, if every value there is an untouched span and
    /// they are adjacent.
    pub fn span_of(&self, range: std::ops::Range<usize>) -> Option<(usize, usize)> {
        let mut out: Option<(usize, usize)> = None;
        for (_, s) in &self.items[range] {
            let Str::Span(a, b) = s else { return None };
            let (a, b) = (*a as usize, *b as usize);
            out = match out {
                None => Some((a, b)),
                Some((x, y)) if y == a => Some((x, b)),
                Some(_) => return None,
            };
        }
        out
    }

    /// `str(stmt).strip()` of one statement of `split()`, after `StripTrailingSemicolonFilter`
    /// if `strip_semicolon`.
    pub fn split_text(
        &self,
        range: std::ops::Range<usize>,
        strip_semicolon: bool,
    ) -> std::borrow::Cow<'_, str> {
        split_text(
            self.src.as_ref(),
            &self.items,
            &self.types,
            range,
            strip_semicolon,
        )
    }
}

/// `str(stmt).strip()` of the items `range` of `src`, after `StripTrailingSemicolonFilter` if
/// `strip_semicolon`.
pub fn split_text<'a>(
    src: &'a str,
    items: &[Item],
    types: &Types,
    range: std::ops::Range<usize>,
    strip_semicolon: bool,
) -> std::borrow::Cow<'a, str> {
    let text = |s: &Str| -> &'a str {
        match s {
            Str::Span(a, b) => &src[*a as usize..*b as usize],
            Str::Owned(_) => "",
        }
    };
    let mut end = range.end;
    if strip_semicolon {
        while end > range.start {
            let (tt, val) = &items[end - 1];
            let v: std::borrow::Cow<'_, str> = match val {
                Str::Span(..) => std::borrow::Cow::Borrowed(text(val)),
                Str::Owned(o) => std::borrow::Cow::Owned(o.to_string()),
            };
            if types.has(*tt, bits::WHITESPACE) || v == ";" {
                end -= 1;
            } else {
                break;
            }
        }
    }
    let mut span: Option<(usize, usize)> = None;
    let mut contiguous = true;
    for (_, s) in &items[range.start..end] {
        match s {
            Str::Span(a, b) => {
                let (a, b) = (*a as usize, *b as usize);
                span = match span {
                    None => Some((a, b)),
                    Some((x, y)) if y == a => Some((x, b)),
                    Some(_) => {
                        contiguous = false;
                        break;
                    }
                };
            }
            Str::Owned(_) => {
                contiguous = false;
                break;
            }
        }
    }
    if contiguous {
        let (a, b) = span.unwrap_or((0, 0));
        return std::borrow::Cow::Borrowed(super::pystr::strip(&src[a..b]));
    }
    let mut s = String::new();
    for (_, v) in &items[range.start..end] {
        match v {
            Str::Span(a, b) => s.push_str(&src[*a as usize..*b as usize]),
            Str::Owned(o) => s.push_str(o),
        }
    }
    std::borrow::Cow::Owned(super::pystr::strip(&s).to_owned())
}

/// `format(sql, **options)`.
///
/// # Errors
/// See [`Error`].
pub fn format<H: Hooks>(
    src: &str,
    cfg: &Config,
    limits: Limits,
    opts: &FormatOptions,
    hooks: &mut H,
) -> Result<String, Error<H::Error>> {
    let mut st = stream(src, cfg, &opts.tokens, hooks)?;
    let grouping = opts.grouping();
    let mut reindent = opts.reindent.clone().map(Reindent::new);
    let mut aligned = opts.aligned.map(Aligned::new);
    let mut out = String::with_capacity(src.len() + src.len() / 4);
    let mut count = 0usize;
    loop {
        let range = match st.next_statement() {
            Ok(Some(r)) => r,
            Ok(None) => break,
            Err(e) => return Err(e),
        };
        hooks.tick().map_err(Error::External)?;
        let (mut t, stmt) = st.statement(range);
        let deep = super::grouping::nesting(&t, stmt) > super::grouping::BIG_STACK_NESTING;
        let (r, a) = (&mut reindent, &mut aligned);
        super::grouping::on_big_stack(deep, || -> Result<(), Fail> {
            if grouping {
                Grouper::new(&mut t, limits).group(stmt)?;
            }
            if opts.spaces_around_operators {
                filters::spaces_around_operators(&mut t, stmt, Depth::new(), 0)?;
            }
            if opts.strip_comments {
                filters::strip_comments(&mut t, stmt, Depth::new(), 0)?;
            }
            if opts.strip_whitespace {
                filters::strip_whitespace(&mut t, stmt, Depth::new(), 0)?;
            }
            if let Some(r) = r.as_mut() {
                r.process(&mut t, stmt)?;
            }
            if let Some(a) = a.as_mut() {
                a.process(&mut t, stmt)?;
            }
            Ok(())
        })?;
        if opts.right_margin {
            return Err(Error::NotImplemented);
        }
        count += 1;
        let text = match opts.output {
            Some(kind) => filters::output_text(&t, stmt, kind, count),
            None => t.text(stmt).into_owned(),
        };
        if let Some(r) = reindent.as_mut() {
            // str(self._last_stmt) when the next statement is processed: the output filter
            // left its tokens an exhausted generator
            r.last_stmt_ends_nl = Some(opts.output.is_none() && text.ends_with('\n'));
        }
        filters::serialize(&text, &mut out);
    }
    Ok(out)
}

/// `split(sql, strip_semicolon=...)`.
///
/// # Errors
/// See [`Error`].
pub fn split<'s, H: Hooks>(
    src: &'s str,
    cfg: &Config,
    strip_semicolon: bool,
    hooks: &mut H,
) -> Result<Vec<std::borrow::Cow<'s, str>>, Error<H::Error>> {
    let mut st = stream(src, cfg, &TokenFilters::default(), hooks)?;
    let mut ranges = Vec::new();
    while let Some(r) = st.next_statement()? {
        ranges.push(r);
    }
    Ok(ranges
        .into_iter()
        .map(|r| split_text(src, &st.items, &st.types, r, strip_semicolon))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sqlparse::ttype::builtin as T;

    const V: UnicodeVersion = UnicodeVersion::LATEST;

    fn cfg() -> Config {
        let kw: Vec<(String, u16)> = [
            ("SELECT", T::DML),
            ("FROM", T::KEYWORD),
            ("WHERE", T::KEYWORD),
            ("AND", T::KEYWORD),
            ("AS", T::KEYWORD),
            ("BEGIN", T::KEYWORD),
            ("END", T::KEYWORD),
        ]
        .iter()
        .map(|(k, t)| ((*k).to_owned(), *t))
        .collect();
        Config::new(Config::default_regexes(V), &[kw], V)
    }

    #[test]
    fn split_statements() {
        let c = cfg();
        let r = split("select 1; select 2;\n", &c, false, &mut NoHooks).expect("split");
        assert_eq!(r, vec!["select 1;", "select 2;"]);
        let r = split("select 1;", &c, true, &mut NoHooks).expect("split");
        assert_eq!(r, vec!["select 1"]);
    }

    #[test]
    fn format_reindent() {
        let c = cfg();
        let opts = FormatOptions {
            strip_whitespace: true,
            reindent: Some(ReindentOptions {
                width: 2,
                ch: ' ',
                wrap_after: 0,
                comma_first: false,
                indent_after_first: false,
                indent_columns: false,
                compact: false,
            }),
            ..FormatOptions::default()
        };
        let r = format(
            "select a, b from t where x = 1 and y = 2",
            &c,
            Limits::default(),
            &opts,
            &mut NoHooks,
        )
        .expect("format");
        assert_eq!(r, "select a,\n       b\nfrom t\nwhere x = 1\n  and y = 2");
    }
}
