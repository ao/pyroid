// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! `FineWebQualityFilter` (`datatrove/pipeline/filters/fineweb_quality_filter.py`).

use std::collections::HashSet;

use rustc_hash::FxBuildHasher;

use super::text::{self, CharSet, py_len};
use super::{Doc, FilterError, Reason, Verdict, as_f64, ratio, tables};

/// `FineWebQualityFilter(...)` with datatrove's parameter names and defaults.
#[derive(Clone, Debug)]
pub struct FineWebQuality {
    pub line_punct_thr: f64,
    pub line_punct_exclude_zero: bool,
    /// `stop_chars`: line endings that count as punctuation (any strings;
    /// `None` = datatrove's `TERMINAL_PUNCTUATION`).
    pub stop_chars: Option<Vec<String>>,
    pub short_line_thr: f64,
    pub short_line_length: f64,
    pub char_duplicates_ratio: f64,
    pub new_line_ratio: f64,
}

impl Default for FineWebQuality {
    fn default() -> Self {
        Self {
            line_punct_thr: 0.12,
            line_punct_exclude_zero: false,
            stop_chars: None,
            short_line_thr: 0.67,
            short_line_length: 30.0,
            char_duplicates_ratio: 0.01,
            new_line_ratio: 0.3,
        }
    }
}

/// `line.endswith(stop_chars)`, precompiled.
#[derive(Debug)]
pub(crate) enum StopChars {
    Chars(CharSet),
    Strings(Vec<String>),
}

static DEFAULT_STOP: std::sync::LazyLock<StopChars> =
    std::sync::LazyLock::new(|| StopChars::new(None));

impl StopChars {
    fn new(stop: Option<&[String]>) -> Self {
        match stop {
            None => StopChars::Chars(CharSet::new(tables::TERMINAL_PUNCTUATION.iter().copied())),
            Some(list) if list.iter().all(|s| s.chars().count() == 1) => {
                StopChars::Chars(CharSet::new(list.iter().filter_map(|s| s.chars().next())))
            }
            Some(list) => StopChars::Strings(list.to_vec()),
        }
    }

    fn ends(&self, line: &str) -> bool {
        match self {
            StopChars::Chars(set) => line.chars().next_back().is_some_and(|c| set.contains(c)),
            StopChars::Strings(list) => list.iter().any(|s| line.ends_with(s.as_str())),
        }
    }
}

impl FineWebQuality {
    /// `FineWebQualityFilter.filter(doc)`.
    pub fn filter(&self, doc: &mut Doc<'_, '_>) -> Result<Verdict, FilterError> {
        let text = doc.text();
        let lines: Vec<&str> = text
            .split('\n')
            .filter(|l| !text::py_strip(l).is_empty())
            .collect();
        if lines.is_empty() {
            return Ok(Some(Reason::Fixed("empty")));
        }
        let custom;
        let stop = match self.stop_chars.as_deref() {
            None => &*DEFAULT_STOP,
            Some(list) => {
                custom = StopChars::new(Some(list));
                &custom
            }
        };
        let r = ratio(lines.iter().filter(|l| stop.ends(l)).count(), lines.len())?;
        if r < self.line_punct_thr && !(r == 0.0 && self.line_punct_exclude_zero) {
            return Ok(Some(Reason::Fixed("line_punct_ratio")));
        }
        let short = lines
            .iter()
            .filter(|l| as_f64(py_len(l)) <= self.short_line_length)
            .count();
        if ratio(short, lines.len())? > self.short_line_thr {
            return Ok(Some(Reason::Fixed("short_line_ratio")));
        }
        let mut seen: HashSet<&str, FxBuildHasher> = HashSet::default();
        let dup_chars: usize = lines
            .iter()
            .filter(|l| !seen.insert(l))
            .map(|l| py_len(l))
            .sum();
        let newlines = memchr::memchr_iter(b'\n', text.as_bytes()).count();
        if ratio(dup_chars, py_len(text) - newlines)? > self.char_duplicates_ratio {
            return Ok(Some(Reason::Fixed("char_dup_ratio")));
        }
        let n_words = doc.words()?.len();
        if ratio(newlines, n_words)? > self.new_line_ratio {
            return Ok(Some(Reason::Fixed("list_ratio")));
        }
        Ok(None)
    }
}
