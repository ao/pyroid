// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! sqlparse 0.6.1's lexer (`Lexer.get_tokens`): the delimited-spans pre-pass
//! (`keywords.find_delimited_spans`) and the ordered regex table, first match wins.

use rustc_hash::FxHashMap;

use super::pystr;
use super::regex::{Regex, Scratch};
use super::ttype::{TT, builtin};
use crate::ftfy::UnicodeVersion;

/// What an entry of the table does with its match.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    /// `(rx, tokens.X)`: a token of type X.
    Type(TT),
    /// `(rx, keywords.PROCESS_AS_KEYWORD)`: look the word up in the keyword dictionaries.
    Keyword,
    /// Any other action: sqlparse consumes the text and yields nothing.
    Drop,
}

/// How an entry matches.
#[derive(Clone, Debug)]
pub enum Matcher {
    /// Compiled here.
    Native(Box<Regex>),
    /// Matched by the caller (Python's `re`), by this index.
    External(usize),
}

/// One `(regex, action)` of `Lexer._SQL_REGEX`.
#[derive(Clone, Debug)]
pub struct Entry {
    /// The matcher.
    pub matcher: Matcher,
    /// The action.
    pub action: Action,
}

/// The lexer's configuration: `_SQL_REGEX` and `_keywords`, in order.
#[derive(Clone, Debug)]
pub struct Config {
    entries: Vec<Entry>,
    /// The keyword dictionaries merged, first dictionary wins.
    keywords: FxHashMap<Box<str>, TT>,
    /// Candidate entries by ASCII first character, and for any other character.
    ascii: Vec<Vec<u16>>,
    other: Vec<u16>,
    version: UnicodeVersion,
    /// Some entry is external.
    external: bool,
    /// The table is sqlparse 0.6.1's own `SQL_REGEX` (any actions): [`fast`] applies.
    fast: bool,
}

impl Config {
    /// A configuration from its entries and keyword dictionaries (in lookup order).
    ///
    /// # Panics
    /// Never in practice: more than 65,535 entries are refused by the caller.
    pub fn new(
        entries: Vec<Entry>,
        dicts: &[Vec<(String, TT)>],
        version: UnicodeVersion,
    ) -> Config {
        let mut keywords: FxHashMap<Box<str>, TT> = FxHashMap::default();
        for d in dicts {
            for (k, t) in d {
                keywords.entry(k.as_str().into()).or_insert(*t);
            }
        }
        let mut ascii = vec![Vec::new(); 128];
        let mut other = Vec::new();
        let mut external = false;
        for (i, e) in entries.iter().enumerate() {
            let i = u16::try_from(i).unwrap_or(u16::MAX);
            match &e.matcher {
                Matcher::Native(re) => {
                    for (cp, list) in ascii.iter_mut().enumerate() {
                        let c = char::from_u32(u32::try_from(cp).unwrap_or(0)).unwrap_or('\0');
                        if re.may_start_with(Some(c)) {
                            list.push(i);
                        }
                    }
                    // every non-ASCII character gets the same answer
                    if re.may_start_with(Some('\u{80}')) {
                        other.push(i);
                    }
                }
                Matcher::External(_) => {
                    external = true;
                    for list in &mut ascii {
                        list.push(i);
                    }
                    other.push(i);
                }
            }
        }
        let fast = entries.len() == SQL_REGEX.len()
            && entries
                .iter()
                .zip(SQL_REGEX)
                .all(|(e, (rx, _))| match &e.matcher {
                    Matcher::Native(re) => {
                        re.source()
                            == (
                                *rx,
                                super::regex::FLAG_IGNORECASE | super::regex::FLAG_UNICODE,
                            )
                    }
                    Matcher::External(_) => false,
                });
        Config {
            entries,
            keywords,
            ascii,
            other,
            version,
            external,
            fast,
        }
    }

    /// sqlparse 0.6.1's default table (`keywords.SQL_REGEX` and the keyword dictionaries are
    /// given by the caller; here only the regexes, for tests and benchmarks).
    ///
    /// # Panics
    /// If a pattern of the default table does not compile (a bug).
    pub fn default_regexes(version: UnicodeVersion) -> Vec<Entry> {
        use super::regex::{FLAG_IGNORECASE, FLAG_UNICODE};
        SQL_REGEX
            .iter()
            .map(|&(rx, action)| Entry {
                matcher: Matcher::Native(Box::new(
                    Regex::new(rx, FLAG_IGNORECASE | FLAG_UNICODE, version).expect(rx),
                )),
                action,
            })
            .collect()
    }

    /// The Unicode version the table was compiled for.
    pub fn version(&self) -> UnicodeVersion {
        self.version
    }

    /// Turn the fast paths for the default table off (tests compare both).
    #[must_use]
    pub fn without_fast_paths(mut self) -> Config {
        self.fast = false;
        self
    }

    /// Whether some entry needs the caller ([`Matcher::External`]).
    pub fn has_external(&self) -> bool {
        self.external
    }

    /// `Lexer.is_keyword(value)`: the type of `value`'s upper-case form in the first
    /// dictionary holding it, else `Name`.
    pub fn keyword_type(&self, value: &str) -> TT {
        if value.is_ascii() && value.len() <= 64 {
            let mut buf = [0u8; 64];
            let b = &mut buf[..value.len()];
            b.copy_from_slice(value.as_bytes());
            b.make_ascii_uppercase();
            let key = std::str::from_utf8(b).unwrap_or("");
            return self.keywords.get(key).copied().unwrap_or(builtin::NAME);
        }
        let up = pystr::upper(value, self.version);
        self.keywords.get(&*up).copied().unwrap_or(builtin::NAME)
    }
}

/// One lexed token: a type and a byte range of the text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawTok {
    /// The type.
    pub tt: TT,
    /// Byte offset of the start.
    pub start: u32,
    /// Byte offset of the end.
    pub end: u32,
}

/// Why lexing stopped early.
#[derive(Debug)]
pub enum LexError<E> {
    /// An entry matched the empty string at `pos` (sqlparse's `consume(iterable, -1)` then
    /// raises `ValueError` from `itertools.islice`).
    EmptyMatch,
    /// The caller's matcher failed.
    External(E),
    /// The caller's [`External::tick`] stopped the run (Ctrl-C): raise it at once.
    Interrupted(E),
}

/// The caller's matchers ([`Matcher::External`]).
pub trait External {
    /// The caller's error.
    type Error;
    /// `match(text, pos)` of external entry `index`: the end byte offset.
    ///
    /// # Errors
    /// Whatever the caller's matcher raises.
    fn match_at(
        &mut self,
        index: usize,
        text: &str,
        pos: usize,
    ) -> Result<Option<usize>, Self::Error>;

    /// Called now and then during long runs (between statements, every 64 Ki tokens): an
    /// error (a pending Ctrl-C) stops the run.
    ///
    /// # Errors
    /// Whatever the caller reports.
    fn tick(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

/// No external matchers.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoExternal;

impl External for NoExternal {
    type Error = std::convert::Infallible;
    fn match_at(&mut self, _: usize, _: &str, _: usize) -> Result<Option<usize>, Self::Error> {
        Ok(None)
    }
}

/// `find_delimited_spans`: opener and closer positions of dollar-quoted literals and
/// multiline comments.
struct Spans<'a> {
    /// Sorted by position: (start, opener end, tag index, type).
    openers: Vec<(usize, usize, usize, TT)>,
    /// Per tag: closer starts (ascending) and ends.
    closers: Vec<(Vec<usize>, Vec<usize>)>,
    tags: FxHashMap<&'a str, usize>,
}

impl<'a> Spans<'a> {
    fn tag(&mut self, t: &'a str) -> usize {
        let n = self.tags.len();
        let i = *self.tags.entry(t).or_insert(n);
        if i == self.closers.len() {
            self.closers.push((Vec::new(), Vec::new()));
        }
        i
    }

    fn find(text: &'a str, v: UnicodeVersion) -> Spans<'a> {
        let mut s = Spans {
            openers: Vec::new(),
            closers: Vec::new(),
            tags: FxHashMap::default(),
        };
        let has_dollar = text.contains('$');
        let has_comment_open = text.contains("/*");
        if !has_dollar && !has_comment_open {
            return s;
        }
        let b = text.as_bytes();
        if has_dollar {
            // _DOLLAR_QUOTE_DELIM = (?=(\$(?:[_A-ZÀ-Ü]\w*)?\$)), IGNORECASE: every position
            // where a tag starts, overlapping ones included.
            for (start, _) in text.match_indices('$') {
                let Some(end) = dollar_tag_end(text, start, v) else {
                    continue;
                };
                let tag = &text[start..end];
                let ti = s.tag(tag);
                s.closers[ti].0.push(start);
                s.closers[ti].1.push(end);
                // _DOLLAR_QUOTE_OPENER_OK = (?<![\w"$])
                let ok = text[..start]
                    .chars()
                    .next_back()
                    .is_none_or(|p| !(pystr::is_word(p, v) || p == '"' || p == '$'));
                if ok {
                    s.openers.push((start, end, ti, builtin::LITERAL));
                }
            }
        }
        if has_comment_open {
            let ti = s.tag("/*");
            // /\*\+ (non-overlapping) then /\*(?!\+): both start at a "/*", so every "/*"
            // is an opener of one kind (finditer of "/\*" never overlaps itself).
            let mut i = 0;
            while let Some(off) = text[i..].find("/*") {
                let start = i + off;
                if b.get(start + 2) == Some(&b'+') {
                    s.openers
                        .push((start, start + 3, ti, builtin::MULTILINE_HINT));
                } else {
                    s.openers.push((start, start + 2, ti, builtin::MULTILINE));
                }
                i = start + 2;
            }
            let mut i = 0;
            while let Some(off) = text[i..].find("*/") {
                let start = i + off;
                s.closers[ti].0.push(start);
                s.closers[ti].1.push(start + 2);
                i = start + 2;
            }
        }
        s.openers.sort_unstable_by_key(|o| o.0);
        s
    }

    /// `resolve(pos)` of the opener at index `oi`: the end of the span and its type.
    fn resolve(&self, oi: usize) -> Option<(usize, TT)> {
        let (_, opener_end, ti, tt) = self.openers[oi];
        let (starts, ends) = &self.closers[ti];
        let idx = starts.partition_point(|&x| x < opener_end);
        (idx < starts.len()).then(|| (ends[idx], tt))
    }
}

/// The end of the dollar-quote tag `\$(?:[_A-ZÀ-Ü]\w*)?\$` (IGNORECASE) starting at `start`.
fn dollar_tag_end(text: &str, start: usize, v: UnicodeVersion) -> Option<usize> {
    let rest = &text[start + 1..];
    let mut it = rest.char_indices();
    match it.next() {
        Some((_, '$')) => Some(start + 2),
        Some((_, c)) if is_tag_start(c, v) => {
            let mut end = rest.len();
            for (i, c) in it {
                if !pystr::is_word(c, v) {
                    end = i;
                    break;
                }
            }
            (rest[end..].starts_with('$')).then_some(start + 1 + end + 1)
        }
        _ => None,
    }
}

/// `[_A-ZÀ-Ü]` under IGNORECASE: `_` or a character whose simple lowercase mapping is in
/// `a-z`, `à-ö`, `×`, `ø-ü`, or is `ı`/`ſ` (re._casefix's extras of `i` and `s`).
pub fn is_tag_start(c: char, v: UnicodeVersion) -> bool {
    if c == '_' {
        return true;
    }
    let lo = u32::from(pystr::simple_lower(c, v));
    matches!(lo, 0x61..=0x7A | 0xE0..=0xF6 | 0xD7 | 0xF8..=0xFC | 0x131 | 0x17F)
}

/// Lex `text` with `cfg` (`Lexer.get_tokens`): the tokens, and why lexing stopped if it did
/// before the end.
///
/// # Panics
/// Never: texts of 4 GiB and more are refused by the caller.
pub fn lex<X: External>(
    text: &str,
    cfg: &Config,
    ext: &mut X,
) -> (Vec<RawTok>, Option<LexError<X::Error>>) {
    let v = cfg.version;
    let spans = Spans::find(text, v);
    let mut out: Vec<RawTok> = Vec::with_capacity(text.len() / 3 + 4);
    let mut scratch = Scratch::default();
    let mut oi = 0;
    let mut pos = 0;
    let n = text.len();
    let u = |x: usize| u32::try_from(x).unwrap_or(u32::MAX);
    'outer: while pos < n {
        if !spans.openers.is_empty() {
            while oi < spans.openers.len() && spans.openers[oi].0 < pos {
                oi += 1;
            }
            if oi < spans.openers.len() && spans.openers[oi].0 == pos {
                if let Some((end, tt)) = spans.resolve(oi) {
                    out.push(RawTok {
                        tt,
                        start: u(pos),
                        end: u(end),
                    });
                    pos = end;
                    continue;
                }
            }
        }
        if out.len() & 0xFFFF == 0xFFFF {
            if let Err(e) = ext.tick() {
                return (out, Some(LexError::Interrupted(e)));
            }
        }
        let c = text[pos..].chars().next().unwrap_or('\0');
        if cfg.fast {
            if let Some((i, end)) = fast::matched(text, pos, c, v) {
                let e = &cfg.entries[i];
                match e.action {
                    Action::Type(tt) => out.push(RawTok {
                        tt,
                        start: u(pos),
                        end: u(end),
                    }),
                    Action::Keyword => out.push(RawTok {
                        tt: cfg.keyword_type(&text[pos..end]),
                        start: u(pos),
                        end: u(end),
                    }),
                    Action::Drop => {}
                }
                pos = end;
                continue;
            }
        }
        let cands: &[u16] = if c.is_ascii() {
            &cfg.ascii[c as usize]
        } else {
            &cfg.other
        };
        for &i in cands {
            let e = &cfg.entries[usize::from(i)];
            let end = match &e.matcher {
                Matcher::Native(re) => re.match_with(text, pos, &mut scratch),
                Matcher::External(k) => match ext.match_at(*k, text, pos) {
                    Ok(m) => m,
                    Err(err) => return (out, Some(LexError::External(err))),
                },
            };
            let Some(end) = end else { continue };
            let end = end.max(pos);
            match e.action {
                Action::Type(tt) => out.push(RawTok {
                    tt,
                    start: u(pos),
                    end: u(end),
                }),
                Action::Keyword => out.push(RawTok {
                    tt: cfg.keyword_type(&text[pos..end]),
                    start: u(pos),
                    end: u(end),
                }),
                Action::Drop => {}
            }
            if end <= pos {
                // sqlparse yields the empty token, then consume(iterable, -1) raises
                return (out, Some(LexError::EmptyMatch));
            }
            pos = end;
            continue 'outer;
        }
        let end = pos + c.len_utf8();
        out.push(RawTok {
            tt: builtin::ERROR,
            start: u(pos),
            end: u(end),
        });
        pos = end;
    }
    (out, None)
}

/// Shortcuts for sqlparse 0.6.1's own table: at positions where the first matching entry
/// can be read off directly (whitespace, closed quotes, words), its index and end; `None`
/// leaves the position to the ordered regex loop. Each answer is what that loop gives
/// (`tests::fast_paths_agree` checks them against it).
mod fast {
    use super::super::pystr;
    use crate::ftfy::UnicodeVersion;

    const HINT_LINE: usize = 0;
    const COMMENT_LINE: usize = 1;
    const NEWLINE: usize = 2;
    const WHITESPACE: usize = 3;
    const BACKTICK: usize = 7;
    const KEYWORD_LIKE: usize = 13;
    const NAME_DOT: usize = 15;
    const NAME_AFTER_DOT: usize = 16;
    const NAME_CALL: usize = 17;
    const STRING_SINGLE: usize = 22;
    const STRING_SYMBOL: usize = 23;
    const WORD: usize = 45;

    /// `CASE|IN|VALUES|USING|FROM|AS` (entry 13).
    const KEYWORDS_13: [&str; 6] = ["CASE", "IN", "VALUES", "USING", "FROM", "AS"];
    /// The first words of entries 26 to 44 (JOIN, END, IF EXISTS, NOT NULL, ...).
    const MULTI: [&str; 30] = [
        "LEFT", "RIGHT", "FULL", "INNER", "OUTER", "STRAIGHT", "CROSS", "NATURAL", "JOIN", "END",
        "IF", "NOT", "ASC", "DESC", "NULLS", "UNION", "CREATE", "DOUBLE", "GROUP", "ORDER",
        "PRIMARY", "HANDLER", "GO", "LATERAL", "AT", "WITH", "LIKE", "ILIKE", "RLIKE", "REGEXP",
    ];

    /// The ASCII upper-case letter `re.IGNORECASE` matches `c` with, if any.
    fn fold(c: char) -> Option<u8> {
        match c {
            'a'..='z' | 'A'..='Z' => Some(c.to_ascii_uppercase() as u8),
            '\u{130}' | '\u{131}' => Some(b'I'),
            '\u{17f}' => Some(b'S'),
            '\u{212a}' => Some(b'K'),
            _ => None,
        }
    }

    /// Whether `word` matches one of `literals` (ASCII upper case) under IGNORECASE.
    fn is_one_of(word: &str, literals: &[&str]) -> bool {
        let mut buf = [0u8; 8];
        let mut n = 0;
        for c in word.chars() {
            let Some(f) = fold(c) else { return false };
            if n == buf.len() {
                return false;
            }
            buf[n] = f;
            n += 1;
        }
        literals.iter().any(|l| l.as_bytes() == &buf[..n])
    }

    /// The end of `q(qq|\\q|[^q])*q` (`backslash`: with the middle alternative) when its first
    /// path closes; `None` when it would have to backtrack (an unclosed quote).
    fn quoted(text: &str, pos: usize, q: u8, backslash: bool) -> Option<usize> {
        let b = text.as_bytes();
        let mut i = pos + 1;
        while i < b.len() {
            let c = b[i];
            if c == q {
                if b.get(i + 1) == Some(&q) {
                    i += 2;
                    continue;
                }
                return Some(i + 1);
            }
            if backslash && c == b'\\' && b.get(i + 1) == Some(&q) {
                i += 2;
                continue;
            }
            i += text[i..].chars().next().map_or(1, char::len_utf8);
        }
        None
    }

    pub(super) fn matched(
        text: &str,
        pos: usize,
        c: char,
        v: UnicodeVersion,
    ) -> Option<(usize, usize)> {
        let b = text.as_bytes();
        match c {
            '-' | '#' => line_comment(b, pos),
            '\r' => Some((
                NEWLINE,
                pos + if b.get(pos + 1) == Some(&b'\n') { 2 } else { 1 },
            )),
            '\n' => Some((NEWLINE, pos + 1)),
            '\'' => quoted(text, pos, b'\'', true).map(|e| (STRING_SINGLE, e)),
            '"' => quoted(text, pos, b'"', true).map(|e| (STRING_SYMBOL, e)),
            '`' => quoted(text, pos, b'`', false).map(|e| (BACKTICK, e)),
            c if pystr::is_space(c) => Some((WHITESPACE, pos + c.len_utf8())),
            c if pystr::is_word(c, v) && !pystr::is_digit(c, v) => word(text, pos, c, v),
            _ => None,
        }
    }

    /// `(--|# )\+.*?(\r\n|\r|\n|$)` (entry 0) or `(--|# ).*?(\r\n|\r|\n|$)` (entry 1): up to
    /// and including the first line break, else to the end. `None` without the prefix.
    fn line_comment(b: &[u8], pos: usize) -> Option<(usize, usize)> {
        let pre = b.get(pos..pos + 2);
        if pre != Some(b"--".as_slice()) && pre != Some(b"# ".as_slice()) {
            return None;
        }
        let hint = b.get(pos + 2) == Some(&b'+');
        let from = pos + 2 + usize::from(hint);
        let end = match memchr::memchr2(b'\r', b'\n', &b[from..]) {
            None => b.len(),
            Some(i) => {
                let i = from + i;
                if b[i] == b'\r' && b.get(i + 1) == Some(&b'\n') {
                    i + 2
                } else {
                    i + 1
                }
            }
        };
        Some((if hint { HINT_LINE } else { COMMENT_LINE }, end))
    }

    fn word(text: &str, pos: usize, c: char, v: UnicodeVersion) -> Option<(usize, usize)> {
        let b = text.as_bytes();
        let mut run = pos + c.len_utf8();
        for d in text[run..].chars() {
            if !pystr::is_word(d, v) {
                break;
            }
            run += d.len_utf8();
        }
        let w = &text[pos..run];
        if is_one_of(w, &KEYWORDS_13) {
            return Some((KEYWORD_LIKE, run));
        }
        if c != '_' && super::is_tag_start(c, v) {
            // [A-ZÀ-Ü]\w*(?=\s*\.(?!\d))
            let mut j = run;
            for d in text[run..].chars() {
                if !pystr::is_space(d) {
                    break;
                }
                j += d.len_utf8();
            }
            if b.get(j) == Some(&b'.')
                && !text[j + 1..]
                    .chars()
                    .next()
                    .is_some_and(|d| pystr::is_digit(d, v))
            {
                return Some((NAME_DOT, run));
            }
            // (?<=\.)[A-ZÀ-Ü]\w*
            if pos > 0 && b[pos - 1] == b'.' {
                return Some((NAME_AFTER_DOT, run));
            }
            // [A-ZÀ-Ü]\w*(?=\()
            if b.get(run) == Some(&b'(') {
                return Some((NAME_CALL, run));
            }
        }
        if is_one_of(w, &MULTI) {
            return None;
        }
        // \w[$#\w]*
        let mut end = run;
        for d in text[run..].chars() {
            if d == '$' || d == '#' || pystr::is_word(d, v) {
                end += d.len_utf8();
            } else {
                break;
            }
        }
        Some((WORD, end))
    }
}

/// sqlparse 0.6.1's `keywords.SQL_REGEX`.
pub const SQL_REGEX: &[(&str, Action)] = &[
    (
        r"(--|# )\+.*?(\r\n|\r|\n|$)",
        Action::Type(builtin::SINGLE_HINT),
    ),
    (r"(--|# ).*?(\r\n|\r|\n|$)", Action::Type(builtin::SINGLE)),
    (r"(\r\n|\r|\n)", Action::Type(builtin::NEWLINE)),
    (r"\s+?", Action::Type(builtin::WHITESPACE)),
    (r":=", Action::Type(builtin::ASSIGNMENT)),
    (r"::", Action::Type(builtin::PUNCTUATION)),
    (r"\*", Action::Type(builtin::WILDCARD)),
    (r"`(``|[^`])*`", Action::Type(builtin::NAME)),
    (r"´(´´|[^´])*´", Action::Type(builtin::NAME)),
    (r"\?", Action::Type(builtin::PLACEHOLDER)),
    (r"%(\(\w+\))?s", Action::Type(builtin::PLACEHOLDER)),
    (r"(?<!\w)[$:?]\w+", Action::Type(builtin::PLACEHOLDER)),
    (r"\\\w+", Action::Type(builtin::COMMAND)),
    (
        r"(CASE|IN|VALUES|USING|FROM|AS)\b",
        Action::Type(builtin::KEYWORD),
    ),
    (r"(@|##|#)[A-ZÀ-Ü]\w+", Action::Type(builtin::NAME)),
    (r"[A-ZÀ-Ü]\w*(?=\s*\.(?!\d))", Action::Type(builtin::NAME)),
    (r"(?<=\.)[A-ZÀ-Ü]\w*", Action::Type(builtin::NAME)),
    (r"[A-ZÀ-Ü]\w*(?=\()", Action::Type(builtin::NAME)),
    (r"-?0x[\dA-F]+", Action::Type(builtin::HEXADECIMAL)),
    (r"-?\d+(\.\d+)?E-?\d+", Action::Type(builtin::FLOAT)),
    (
        r"(?![_A-ZÀ-Ü])-?(\d+(\.\d*)|\.\d+)(?![_A-ZÀ-Ü])",
        Action::Type(builtin::FLOAT),
    ),
    (
        r"(?![_A-ZÀ-Ü])-?\d+(?![_A-ZÀ-Ü])",
        Action::Type(builtin::INTEGER),
    ),
    (r"'(''|\\'|[^'])*'", Action::Type(builtin::STRING_SINGLE)),
    (r#""(""|\\"|[^"])*""#, Action::Type(builtin::STRING_SYMBOL)),
    (r#"(""|".*?[^\\]")"#, Action::Type(builtin::STRING_SYMBOL)),
    (r"(?<![\w\])])(\[[^\]\[]+\])", Action::Type(builtin::NAME)),
    (
        r"((LEFT\s+|RIGHT\s+|FULL\s+)?(INNER\s+|OUTER\s+|STRAIGHT\s+)?|(CROSS\s+|NATURAL\s+)?)?JOIN\b",
        Action::Type(builtin::KEYWORD),
    ),
    (
        r"END(\s+IF|\s+LOOP|\s+WHILE|\s+FOR|\s+CASE)?\b",
        Action::Type(builtin::KEYWORD),
    ),
    (r"IF\s+(NOT\s+)?EXISTS\b", Action::Type(builtin::KEYWORD)),
    (r"NOT\s+NULL\b", Action::Type(builtin::KEYWORD)),
    (
        r"(ASC|DESC)(\s+NULLS\s+(FIRST|LAST))?\b",
        Action::Type(builtin::ORDER),
    ),
    (r"(ASC|DESC)\b", Action::Type(builtin::ORDER)),
    (r"NULLS\s+(FIRST|LAST)\b", Action::Type(builtin::ORDER)),
    (r"UNION\s+ALL\b", Action::Type(builtin::KEYWORD)),
    (r"CREATE(\s+OR\s+REPLACE)?\b", Action::Type(builtin::DDL)),
    (r"DOUBLE\s+PRECISION\b", Action::Type(builtin::BUILTIN)),
    (r"GROUP\s+BY\b", Action::Type(builtin::KEYWORD)),
    (r"ORDER\s+BY\b", Action::Type(builtin::KEYWORD)),
    (r"PRIMARY\s+KEY\b", Action::Type(builtin::KEYWORD)),
    (r"HANDLER\s+FOR\b", Action::Type(builtin::KEYWORD)),
    (r"GO(\s\d+)\b", Action::Type(builtin::KEYWORD)),
    (
        r"(LATERAL\s+VIEW\s+)(EXPLODE|INLINE|PARSE_URL_TUPLE|POSEXPLODE|STACK)\b",
        Action::Type(builtin::KEYWORD),
    ),
    (
        r"(AT|WITH')\s+TIME\s+ZONE\s+'[^']+'",
        Action::Type(builtin::TZCAST),
    ),
    (
        r"(NOT\s+)?(LIKE|ILIKE|RLIKE)\b",
        Action::Type(builtin::COMPARISON),
    ),
    (
        r"(NOT\s+)?(REGEXP)(\s+(BINARY))?\b",
        Action::Type(builtin::COMPARISON),
    ),
    (r"\w[$#\w]*", Action::Keyword),
    (r"[;:()\[\],\.]", Action::Type(builtin::PUNCTUATION)),
    (
        r"(\->>?|#>>?|@>|<@|\?\|?|\?&|\-|#\-)",
        Action::Type(builtin::OPERATOR),
    ),
    (r"[<>=~!]+", Action::Type(builtin::COMPARISON)),
    (r"[+/@#%^&|^-]+", Action::Type(builtin::OPERATOR)),
];

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::LATEST;

    fn cfg() -> Config {
        let kw = vec![
            ("SELECT".to_owned(), builtin::DML),
            ("FROM".to_owned(), builtin::KEYWORD),
            ("WHERE".to_owned(), builtin::KEYWORD),
        ];
        Config::new(Config::default_regexes(V), &[kw], V)
    }

    fn toks(s: &str) -> Vec<(TT, &str)> {
        let c = cfg();
        let (t, e) = lex(s, &c, &mut NoExternal);
        assert!(e.is_none());
        t.iter()
            .map(|t| (t.tt, &s[t.start as usize..t.end as usize]))
            .collect()
    }

    #[test]
    fn simple() {
        assert_eq!(
            toks("select a from b"),
            vec![
                (builtin::DML, "select"),
                (builtin::WHITESPACE, " "),
                (builtin::NAME, "a"),
                (builtin::WHITESPACE, " "),
                (builtin::KEYWORD, "from"),
                (builtin::WHITESPACE, " "),
                (builtin::NAME, "b"),
            ]
        );
    }

    #[test]
    fn fast_paths_agree() {
        let fast = cfg();
        let slow = cfg().without_fast_paths();
        assert!(fast.fast && !slow.fast);
        let alphabet: Vec<char> = "aZsSkKiI_0193 \t\r\n\x0b\x1c\u{a0}\u{2028}.,;:()[]'\"`\\$#@%?*-+/=<>!^&|~\
             \u{130}\u{131}\u{17f}\u{212a}\u{212b}\u{d7}\u{f7}\u{c0}\u{e9}\u{300}\u{661}\u{b2}\u{4e00}\u{10f800}"
            .chars()
            .collect();
        let words = [
            "select", "SELECT", "from", "join", "left", "IN", "in", "as", "As", "CASE", "values",
            "using", "end", "not", "null", "order", "by", "group", "if", "exists", "go", "with",
            "at", "time", "zone", "like", "regexp", "binary", "union", "all", "create", "or",
            "replace", "a.b", "f(", "x.1", ".y",
        ];
        let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..20_000 {
            let mut s = String::new();
            let len = (next() % 24) as usize;
            for _ in 0..len {
                let r = next();
                if r % 3 == 0 {
                    s.push_str(words[(r / 3) as usize % words.len()]);
                } else {
                    s.push(alphabet[(r / 3) as usize % alphabet.len()]);
                }
            }
            let a = lex(&s, &fast, &mut NoExternal).0;
            let b = lex(&s, &slow, &mut NoExternal).0;
            assert_eq!(a, b, "{s:?}");
        }
    }

    #[test]
    fn spans() {
        assert_eq!(toks("/* x */"), vec![(builtin::MULTILINE, "/* x */")]);
        assert_eq!(toks("$$a$$"), vec![(builtin::LITERAL, "$$a$$")]);
        assert_eq!(toks("$$$$"), vec![(builtin::LITERAL, "$$$$")]);
        assert_eq!(toks("/*/")[0], (builtin::OPERATOR, "/"));
        assert_eq!(
            toks("'a' /*+ h */")[2],
            (builtin::MULTILINE_HINT, "/*+ h */")
        );
    }
}
