// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! datatrove's `SymbolLinesFormatter`: drop the lines made only of symbols.
//!
//! For each line of `text.splitlines()`: a non-blank line whose characters
//! are all in `symbols_to_remove` or `" "` is removed; a run of such lines
//! becomes one `replace_char` (when it is not empty); every other line is
//! kept. The kept pieces are joined with `"\n"`.

use std::borrow::Cow;

use super::tables;
use super::text::{CharSet, py_splitlines, py_strip};

/// `SymbolLinesFormatter` settings.
#[derive(Clone, Debug)]
pub struct SymbolLines {
    symbols: CharSet,
    replace: String,
}

impl Default for SymbolLines {
    /// datatrove's defaults: `PUNCTUATION_SET`, `replace_char=""`.
    fn default() -> Self {
        Self::new(None, String::new())
    }
}

impl SymbolLines {
    /// `symbols`: the characters of `symbols_to_remove` (only one-character
    /// strings can ever equal a character), or `None` for datatrove's
    /// `PUNCTUATION_SET`.
    pub fn new(symbols: Option<Vec<char>>, replace_char: String) -> Self {
        let symbols = match symbols {
            Some(chars) => CharSet::new(chars),
            None => CharSet::new(tables::PUNCTUATION_SET.iter().copied()),
        };
        Self {
            symbols,
            replace: replace_char,
        }
    }

    /// `format(text)`; borrowed when the result equals `text`.
    pub fn format<'a>(&self, text: &'a str) -> Cow<'a, str> {
        let mut out = String::with_capacity(text.len());
        let mut first = true;
        let mut push = |s: &str, out: &mut String| {
            if !first {
                out.push('\n');
            }
            first = false;
            out.push_str(s);
        };
        let mut in_removed_span = false;
        for line in py_splitlines(text) {
            let symbols_only = !py_strip(line).is_empty()
                && line.chars().all(|c| c == ' ' || self.symbols.contains(c));
            if symbols_only {
                if !in_removed_span {
                    if !self.replace.is_empty() {
                        push(&self.replace, &mut out);
                    }
                    in_removed_span = true;
                }
            } else {
                push(line, &mut out);
                in_removed_span = false;
            }
        }
        if out == text {
            Cow::Borrowed(text)
        } else {
            Cow::Owned(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults() {
        let f = SymbolLines::default();
        assert_eq!(f.format("a\n---\n***\nb"), "a\nb");
        assert_eq!(f.format("a\n  \nb"), "a\n  \nb");
        assert_eq!(f.format("a\r\nb\n"), "a\nb");
        assert_eq!(f.format("| x |\n|---|"), "| x |");
        assert!(matches!(f.format("a\nb"), Cow::Borrowed(_)));
        assert_eq!(f.format(""), "");
    }

    #[test]
    fn replace_char_once_per_run() {
        let f = SymbolLines::new(Some(vec!['|']), "\n".to_owned());
        assert_eq!(f.format("a\n|\n| |\nb\n|"), "a\n\n\nb\n\n");
        assert_eq!(f.format("a\n-\nb"), "a\n-\nb");
    }
}
