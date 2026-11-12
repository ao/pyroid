// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! `C4QualityFilter` (`datatrove/pipeline/filters/c4_filters.py`): line
//! removal and document rejection.

use std::borrow::Cow;

use super::text::{self, py_len};
use super::{FilterError, Reason, WordTokenizer, as_f64};

/// `END_PUNCTUATION`.
const END_PUNCTUATION: [char; 5] = ['.', '?', '!', '"', '\''];
/// `POLICY_SUBSTRINGS`.
const POLICY_SUBSTRINGS: [&[u8]; 6] = [
    b"terms of use",
    b"privacy policy",
    b"cookie policy",
    b"uses cookies",
    b"use of cookies",
    b"use cookies",
];

/// `C4QualityFilter(...)` with datatrove's parameter names and defaults.
/// The three counts are compared as Python would (`-1` disables
/// `min_num_sentences`, `min_words_per_line` and `max_word_length`).
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)] // datatrove's flags
pub struct C4Quality {
    pub split_paragraph: bool,
    pub remove_citations: bool,
    pub filter_no_terminal_punct: bool,
    pub min_num_sentences: f64,
    pub min_words_per_line: f64,
    pub max_word_length: f64,
    pub filter_lorem_ipsum: bool,
    pub filter_javascript: bool,
    pub filter_curly_bracket: bool,
    pub filter_policy: bool,
}

impl Default for C4Quality {
    fn default() -> Self {
        Self {
            split_paragraph: true,
            remove_citations: true,
            filter_no_terminal_punct: true,
            min_num_sentences: 5.0,
            min_words_per_line: 3.0,
            max_word_length: 1000.0,
            filter_lorem_ipsum: true,
            filter_javascript: true,
            filter_curly_bracket: true,
            filter_policy: true,
        }
    }
}

/// The outcome of [`C4Quality::filter`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum C4Outcome<'a> {
    /// Kept, with `doc.text` after line removal (the input when unchanged).
    Keep(Cow<'a, str>),
    /// Dropped (datatrove leaves `doc.text` untouched).
    Drop(Reason),
}

impl C4Quality {
    /// `C4QualityFilter.filter(doc)`: the decision and the new text.
    #[allow(clippy::float_cmp)] // Python's `!= -1` is an exact comparison
    pub fn filter<'a>(
        &self,
        text: &'a str,
        tok: &dyn WordTokenizer,
    ) -> Result<C4Outcome<'a>, FilterError> {
        let v = tok.unicode_version();
        let mut num_sentences = 0usize;
        let mut kept: Vec<Cow<'a, str>> = Vec::new();
        let mut folded = Vec::new();

        let segments: Vec<&'a str> = if self.split_paragraph {
            text::py_splitlines(text).collect()
        } else {
            sentence_segments(text, tok)?
        };
        for raw in segments {
            let stripped = text::py_strip(raw);
            // `words = line.split()` is taken before citations are removed.
            let mut n_words = 0usize;
            let mut too_long = false;
            for w in text::py_split(stripped) {
                n_words += 1;
                if self.max_word_length != -1.0 && as_f64(py_len(w)) > self.max_word_length {
                    too_long = true;
                }
            }
            if too_long {
                continue;
            }
            let line = if self.remove_citations {
                text::remove_citations(stripped, v)
            } else {
                Cow::Borrowed(stripped)
            };
            if self.filter_no_terminal_punct
                && (!line.ends_with(END_PUNCTUATION) || line.ends_with("..."))
            {
                continue;
            }
            if as_f64(n_words) < self.min_words_per_line {
                continue;
            }
            text::fold_lower_ascii(&line, &mut folded);
            if self.filter_lorem_ipsum && text::contains_bytes(&folded, b"lorem ipsum") {
                return Ok(C4Outcome::Drop(Reason::Fixed("lorem_ipsum")));
            }
            if self.filter_javascript && text::contains_bytes(&folded, b"javascript") {
                continue;
            }
            if self.filter_curly_bracket && line.contains('{') {
                return Ok(C4Outcome::Drop(Reason::Fixed("curly_bracket")));
            }
            if self.filter_policy
                && POLICY_SUBSTRINGS
                    .iter()
                    .any(|p| text::contains_bytes(&folded, p))
            {
                continue;
            }
            if self.min_num_sentences != -1.0 {
                num_sentences += if self.split_paragraph {
                    tok.count_sentences(&line)?
                } else {
                    1
                };
            }
            kept.push(line);
        }
        if as_f64(num_sentences) < self.min_num_sentences {
            return Ok(C4Outcome::Drop(Reason::Fixed("too_few_sentences")));
        }
        let sep = if self.split_paragraph { "\n" } else { " " };
        Ok(C4Outcome::Keep(join_strip(text, &kept, sep)))
    }
}

/// `sep.join(kept).strip()`, borrowing `original` when the result equals it.
fn join_strip<'a>(original: &'a str, kept: &[Cow<'a, str>], sep: &str) -> Cow<'a, str> {
    // Cheap equality check first: lengths must add up.
    let total: usize =
        kept.iter().map(|l| l.len()).sum::<usize>() + sep.len() * kept.len().saturating_sub(1);
    if total == original.len() {
        let mut pos = 0;
        let same = kept.iter().enumerate().all(|(i, l)| {
            if i > 0 {
                if !original[pos..].starts_with(sep) {
                    return false;
                }
                pos += sep.len();
            }
            let ok = original[pos..].starts_with(&**l);
            pos += l.len();
            ok
        });
        if same && text::py_strip(original).len() == original.len() {
            return Cow::Borrowed(original);
        }
    }
    let mut out = String::with_capacity(total);
    for (i, l) in kept.iter().enumerate() {
        if i > 0 {
            out.push_str(sep);
        }
        out.push_str(l);
    }
    let stripped = text::py_strip(&out);
    if stripped == original {
        return Cow::Borrowed(original);
    }
    if stripped.len() == out.len() {
        Cow::Owned(out)
    } else {
        Cow::Owned(stripped.to_owned())
    }
}

/// `split_into_sentences(text)` as datatrove slices it: consecutive
/// segments ending at each sentence end, the last running to the end.
fn sentence_segments<'a>(
    text: &'a str,
    tok: &dyn WordTokenizer,
) -> Result<Vec<&'a str>, FilterError> {
    let mut ends = Vec::new();
    tok.sentence_ends(text, &mut ends)?;
    if ends.is_empty() {
        return Ok(vec![text]);
    }
    let mut out = Vec::with_capacity(ends.len());
    let mut start = 0;
    for &e in &ends[..ends.len() - 1] {
        out.push(&text[start..e]);
        start = e;
    }
    out.push(&text[start..]);
    Ok(out)
}
