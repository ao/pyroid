// Derived from python-dateutil 4.0.0 (BSD-3-Clause), © Gustavo Niemeyer, Paul Ganssle and
// dateutil contributors; modified.
//! `_timelex`: dateutil's tokenizer, character for character.
//!
//! Tokens are runs of letters (`str.isalpha`), runs of digits (`str.isdigit`), a single space
//! for any whitespace character, or any other single character. Dots and commas inside numbers
//! and words are kept while reading and split afterwards, as dateutil does. NUL characters are
//! skipped. Tokens borrow from the input unless a NUL interrupted them or a `,` became `.`.

use std::borrow::Cow;

use super::pystr::Props;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Start,
    /// `'a'`: letters.
    Word,
    /// `'0'`: digits.
    Num,
    /// `'a.'`: letters with a dot.
    WordDot,
    /// `'0.'`: digits with a dot (or comma).
    NumDot,
}

/// The token being read: `s[start..end]` while contiguous, else an owned copy.
struct Buf {
    start: usize,
    end: usize,
    chars: usize,
    owned: Option<String>,
    last: char,
}

impl Buf {
    fn new(at: usize, c: char) -> Self {
        Self {
            start: at,
            end: at + c.len_utf8(),
            chars: 1,
            owned: None,
            last: c,
        }
    }

    fn push(&mut self, s: &str, at: usize, c: char) {
        match &mut self.owned {
            Some(o) => o.push(c),
            None if at == self.end => self.end += c.len_utf8(),
            None => {
                let mut o = s[self.start..self.end].to_owned();
                o.push(c);
                self.owned = Some(o);
            }
        }
        self.chars += 1;
        self.last = c;
    }

    fn into_cow(self, s: &str) -> Cow<'_, str> {
        match self.owned {
            Some(o) => Cow::Owned(o),
            None => Cow::Borrowed(&s[self.start..self.end]),
        }
    }
}

/// Split a token on `.`/`,` keeping the separators (`re.split("([.,])", token)`), dropping
/// empty pieces after the first.
fn split_decimal<'a>(tok: Cow<'a, str>, out: &mut Vec<Cow<'a, str>>) -> Cow<'a, str> {
    let pieces = |t: &str| -> Vec<(usize, usize)> {
        let mut v = Vec::new();
        let mut from = 0;
        for (i, b) in t.bytes().enumerate() {
            if b == b'.' || b == b',' {
                v.push((from, i));
                v.push((i, i + 1));
                from = i + 1;
            }
        }
        v.push((from, t.len()));
        v
    };
    match tok {
        Cow::Borrowed(t) => {
            let p = pieces(t);
            for &(a, b) in &p[1..] {
                if b > a {
                    out.push(Cow::Borrowed(&t[a..b]));
                }
            }
            Cow::Borrowed(&t[p[0].0..p[0].1])
        }
        Cow::Owned(t) => {
            let p = pieces(&t);
            for &(a, b) in &p[1..] {
                if b > a {
                    out.push(Cow::Owned(t[a..b].to_owned()));
                }
            }
            Cow::Owned(t[p[0].0..p[0].1].to_owned())
        }
    }
}

struct Lexer<'a> {
    s: &'a str,
    pos: usize,
    /// `charstack` (at most one character).
    pushed: Option<(usize, char)>,
    /// `tokenstack`, in order (a queue).
    stack: Vec<Cow<'a, str>>,
    stack_at: usize,
    eof: bool,
    p: Props,
}

impl<'a> Lexer<'a> {
    /// `instream.read(1)`, skipping NULs.
    fn read(&mut self) -> Option<(usize, char)> {
        loop {
            let c = self.s[self.pos..].chars().next()?;
            let at = self.pos;
            self.pos += c.len_utf8();
            if c != '\0' {
                return Some((at, c));
            }
        }
    }

    #[allow(clippy::too_many_lines)]
    fn get_token(&mut self) -> Option<Cow<'a, str>> {
        if self.stack_at < self.stack.len() {
            let t = std::mem::take(&mut self.stack[self.stack_at]);
            self.stack_at += 1;
            if self.stack_at == self.stack.len() {
                self.stack.clear();
                self.stack_at = 0;
            }
            return Some(t);
        }
        let p = self.p;
        let mut seenletters = false;
        let mut tok: Option<Buf> = None;
        let mut space = false;
        let mut state = State::Start;
        while !self.eof {
            let next = match self.pushed.take() {
                Some(x) => Some(x),
                None => self.read(),
            };
            let Some((at, c)) = next else {
                self.eof = true;
                break;
            };
            match state {
                State::Start => {
                    tok = Some(Buf::new(at, c));
                    if p.is_alpha(c) {
                        state = State::Word;
                    } else if p.is_digit(c) {
                        state = State::Num;
                    } else {
                        space = Props::is_space(c);
                        break;
                    }
                }
                State::Word => {
                    seenletters = true;
                    let t = tok.as_mut().expect("started");
                    if p.is_alpha(c) {
                        t.push(self.s, at, c);
                    } else if c == '.' {
                        t.push(self.s, at, c);
                        state = State::WordDot;
                    } else {
                        self.pushed = Some((at, c));
                        break;
                    }
                }
                State::Num => {
                    let t = tok.as_mut().expect("started");
                    if p.is_digit(c) {
                        t.push(self.s, at, c);
                    } else if c == '.' || (c == ',' && t.chars >= 2) {
                        t.push(self.s, at, c);
                        state = State::NumDot;
                    } else {
                        self.pushed = Some((at, c));
                        break;
                    }
                }
                State::WordDot => {
                    seenletters = true;
                    let t = tok.as_mut().expect("started");
                    if c == '.' || p.is_alpha(c) {
                        t.push(self.s, at, c);
                    } else if p.is_digit(c) && t.last == '.' {
                        t.push(self.s, at, c);
                        state = State::NumDot;
                    } else {
                        self.pushed = Some((at, c));
                        break;
                    }
                }
                State::NumDot => {
                    let t = tok.as_mut().expect("started");
                    if c == '.' || p.is_digit(c) {
                        t.push(self.s, at, c);
                    } else if p.is_alpha(c) && t.last == '.' {
                        t.push(self.s, at, c);
                        state = State::WordDot;
                    } else {
                        self.pushed = Some((at, c));
                        break;
                    }
                }
            }
        }
        let buf = tok?;
        if space {
            return Some(Cow::Borrowed(" "));
        }
        let last = buf.last;
        let mut token = buf.into_cow(self.s);
        if matches!(state, State::WordDot | State::NumDot)
            && (seenletters
                || token.bytes().filter(|&b| b == b'.').count() > 1
                || matches!(last, '.' | ','))
        {
            token = split_decimal(token, &mut self.stack);
        }
        if state == State::NumDot && !token.contains('.') && token.contains(',') {
            token = Cow::Owned(token.replace(',', "."));
        }
        Some(token)
    }
}

/// `_timelex.split(s)`.
pub fn split(s: &str, p: Props) -> Vec<Cow<'_, str>> {
    let mut lx = Lexer {
        s,
        pos: 0,
        pushed: None,
        stack: Vec::new(),
        stack_at: 0,
        eof: false,
        p,
    };
    let mut out = Vec::with_capacity(16);
    while let Some(t) = lx.get_token() {
        out.push(t);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ftfy::UnicodeVersion;

    fn sp(s: &str) -> Vec<String> {
        split(s, Props::new(UnicodeVersion::V15_0))
            .into_iter()
            .map(Cow::into_owned)
            .collect()
    }

    #[test]
    fn tokens() {
        assert_eq!(
            sp("Sep.20.2009 4:30:21.447"),
            [
                "Sep", ".", "20", ".", "2009", " ", "4", ":", "30", ":", "21.447"
            ]
        );
        assert_eq!(sp("12,5"), ["12.5"]);
        assert_eq!(sp("1,5"), ["1", ",", "5"]);
        assert_eq!(sp("12,"), ["12", ","]);
        assert_eq!(sp("a.b"), ["a", ".", "b"]);
        assert_eq!(sp("12.a"), ["12.a"]);
        assert_eq!(sp("Jan\0uary\t1"), ["January", " ", "1"]);
        assert_eq!(sp("10h36m28.5s"), ["10", "h", "36", "m", "28.5", "s"]);
        assert_eq!(sp("12,5.3"), ["12,5.3"]);
        assert_eq!(sp(""), Vec::<String>::new());
        assert_eq!(sp("\0"), Vec::<String>::new());
        assert_eq!(sp("²³ x"), ["²³", " ", "x"]);
        assert_eq!(sp("1.2.3"), ["1", ".", "2", ".", "3"]);
        assert_eq!(sp("4."), ["4", "."]);
    }
}
