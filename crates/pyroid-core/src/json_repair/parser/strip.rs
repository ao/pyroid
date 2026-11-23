// Derived from json_repair 0.64.9 (MIT), © Stefano Baccianella; modified.
//! The scans `_repair_empty_object_result` and `_classify_empty_object_repair`
//! run over an empty object's body: `_strip_comments_for_empty_object_classification`,
//! then a look for content and for a `:` outside quotes.
//!
//! json_repair rebuilds the stripped body and rescans it for every empty object;
//! unclosed nesting (`{[{[{[...`) makes that once per level over the rest of the
//! input. Here the content check stops at the first char that survives, and the
//! `:` search is memoized: from a position where both state machines are in
//! their start state (outside quotes, no pending backslash, not in a comment),
//! the answer depends on that position only, so every such position a scan
//! passes is told the answer, and later scans stop as soon as they reach one.

use super::{Parser, is_delim};

/// The comment-stripping state machine (`_strip_comments_for_empty_object_classification`).
#[derive(Clone, Copy, Default)]
struct Strip {
    in_quote: Option<char>,
    backslashes: usize,
}

/// What one step of [`Strip`] did with the char at the cursor.
enum Step {
    /// The char survives (and the cursor moves past it).
    Keep(char),
    /// A comment was skipped; the cursor is past it.
    Skipped,
}

impl Strip {
    /// Process the char at `i` (of `s[..end]`); returns the step and the next index.
    fn step(&mut self, p: &Parser<'_>, i: usize, end: usize) -> (Step, usize) {
        let ch = p.s.at(i);
        let next = if i + 1 < end {
            Some(p.s.at(i + 1))
        } else {
            None
        };
        if ch == '\\' {
            self.backslashes += 1;
            return (Step::Keep(ch), i + 1);
        }
        if let Some(q) = self.in_quote {
            if ch == q && self.backslashes % 2 == 0 {
                self.in_quote = None;
            }
            self.backslashes = 0;
            return (Step::Keep(ch), i + 1);
        }
        if is_delim(ch) && self.backslashes % 2 == 0 {
            self.in_quote = Some(ch);
            self.backslashes = 0;
            return (Step::Keep(ch), i + 1);
        }
        self.backslashes = 0;
        if ch == '#' || (ch == '/' && next == Some('/')) {
            let mut j = i + if ch == '/' { 2 } else { 1 };
            while j < end && !matches!(p.s.at(j), '\n' | '\r') {
                j += 1;
            }
            return (Step::Skipped, j);
        }
        if ch == '/' && next == Some('*') {
            let mut j = i + 2;
            while j + 1 < end && !(p.s.at(j) == '*' && p.s.at(j + 1) == '/') {
                j += 1;
            }
            return (Step::Skipped, (j + 2).min(end));
        }
        (Step::Keep(ch), i + 1)
    }

    fn clean(self) -> bool {
        self.in_quote.is_none() && self.backslashes == 0
    }
}

/// The `:` search over the stripped chars (`_classify_empty_object_repair`'s loop).
#[derive(Clone, Copy, Default)]
struct Colon {
    in_quote: Option<char>,
    backslashes: usize,
}

impl Colon {
    /// Feed one stripped char; true if it is a `:` outside quotes.
    fn feed(&mut self, ch: char) -> bool {
        if ch == '\\' {
            self.backslashes += 1;
            return false;
        }
        let even = self.backslashes % 2 == 0;
        self.backslashes = 0;
        if let Some(q) = self.in_quote {
            if ch == q && even {
                self.in_quote = None;
            }
        } else if is_delim(ch) && even {
            self.in_quote = Some(ch);
        } else if ch == ':' && even {
            return true;
        }
        false
    }

    fn clean(self) -> bool {
        self.in_quote.is_none() && self.backslashes == 0
    }
}

/// Memo value for "no `:` up to the end".
const NO_COLON: u32 = u32::MAX;

impl Parser<'_> {
    /// Whether `strip_comments(s[a..b])` holds a non-space char.
    pub(crate) fn stripped_has_content(&self, a: usize, b: usize) -> bool {
        let b = b.min(self.s.len());
        let mut strip = Strip::default();
        let mut i = a;
        while i < b {
            let (step, next) = strip.step(self, i, b);
            if let Step::Keep(c) = step {
                if !self.is_space(c) {
                    return true;
                }
            }
            i = next;
        }
        false
    }

    /// The position of the first `:` outside quotes in `strip_comments(s[b0..])`
    /// (as `_classify_empty_object_repair` looks for it), memoized.
    pub(crate) fn stripped_colon(&mut self, b0: usize) -> Option<usize> {
        let n = self.s.len();
        if self.scan.colons.len() != n {
            self.scan.colons = vec![0; n];
        }
        let mut strip = Strip::default();
        let mut colon = Colon::default();
        let mut visited: Vec<usize> = Vec::new();
        let mut i = b0;
        let answer: u32 = loop {
            if i >= n {
                break NO_COLON;
            }
            if strip.clean() && colon.clean() {
                match self.scan.colons[i] {
                    0 => visited.push(i),
                    known => break known,
                }
            }
            let (step, next) = strip.step(self, i, n);
            if let Step::Keep(c) = step {
                if colon.feed(c) {
                    break u32::try_from(i).unwrap_or(NO_COLON - 1) + 1;
                }
            }
            i = next;
        };
        for v in visited {
            self.scan.colons[v] = answer;
        }
        (answer != NO_COLON).then(|| answer as usize - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::chars::{CharProps, UnicodeVersion};
    use super::super::object::strip_comments;
    use super::super::{ParseOptions, Parser, is_delim};

    /// `_classify_empty_object_repair`'s loop over an already stripped body.
    fn colon_in(stripped: &[char]) -> bool {
        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;
        for &ch in stripped {
            if ch == '\\' {
                backslashes += 1;
                continue;
            }
            if let Some(q) = in_quote {
                if ch == q && backslashes % 2 == 0 {
                    in_quote = None;
                }
            } else if is_delim(ch) && backslashes % 2 == 0 {
                in_quote = Some(ch);
            } else if ch == ':' && backslashes % 2 == 0 {
                return true;
            }
            backslashes = 0;
        }
        false
    }

    #[test]
    fn incremental_scans_match_the_reference() {
        let opts = ParseOptions {
            stream_stable: false,
            strict: false,
            try_valid_json_suffix: true,
            logging: false,
            max_int_digits: 4300,
            props: CharProps::new(UnicodeVersion::V15_0),
        };
        let alphabet: Vec<char> = "ab :\\\"'#/*\n {}".chars().collect();
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..300 {
            let len = (seed % 40) as usize;
            let s: Vec<char> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    alphabet[(seed % alphabet.len() as u64) as usize]
                })
                .collect();
            let mut p = Parser::new(&s, &opts, None);
            // Many queries per parser, so the memo is exercised.
            for a in 0..=s.len() {
                for b in a..=s.len() {
                    let stripped = strip_comments(&s[a..b]);
                    let content = stripped.iter().any(|&c| !c.is_whitespace());
                    assert_eq!(p.stripped_has_content(a, b), content, "{s:?} {a}..{b}");
                    let want = colon_in(&stripped);
                    let got = p.stripped_colon(a).is_some_and(|j| j < b);
                    assert_eq!(got, want, "{s:?} {a}..{b} colon");
                }
            }
        }
    }
}
