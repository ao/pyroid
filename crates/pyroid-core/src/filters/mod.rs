// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace, and spaCy 3.8.18
// (MIT), © ExplosionAI GmbH; modified.
//! Quality filters with the same keep/drop decisions, drop reasons and (for
//! C4) rewritten text as datatrove 0.10.1:
//!
//! - [`GopherQuality`] (`GopherQualityFilter`)
//! - [`GopherRepetition`] (`GopherRepetitionFilter`)
//! - [`C4Quality`] (`C4QualityFilter`, line removal and document rejection)
//! - [`FineWebQuality`] (`FineWebQualityFilter`)
//!
//! Each config's `Default` is datatrove's default. Words come from a
//! [`WordTokenizer`]; [`tokenizer::SpacyEnglish`] reproduces datatrove's
//! English tokenizer (spaCy `blank("en")`), which is the only one so far.
//! Python `str` semantics (code-point lengths, `isspace`, `splitlines`,
//! `isalpha`, `lower`) are reproduced exactly, per Unicode version.

use std::borrow::Cow;
use std::fmt;

mod c4;
mod fineweb;
mod gopher;
mod regex;
mod tables;
mod text;
pub mod tokenizer;
pub(crate) mod unicode;

pub use c4::{C4Outcome, C4Quality};
pub use fineweb::FineWebQuality;
pub use gopher::{GopherQuality, GopherRepetition};

use crate::ftfy::UnicodeVersion;

/// Why a filter dropped a document: datatrove's `filter_reason` string.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reason {
    /// A fixed reason string, e.g. `"gopher_short_doc"`.
    Fixed(&'static str),
    /// `"top_{n}_gram"` (Gopher repetition).
    TopNGram(usize),
    /// `"duplicated_{n}_n_grams"` (Gopher repetition).
    DuplicatedNGrams(usize),
}

impl Reason {
    /// The reason string datatrove reports.
    pub fn as_str(&self) -> Cow<'static, str> {
        match *self {
            Reason::Fixed(s) => Cow::Borrowed(s),
            Reason::TopNGram(n) => Cow::Owned(format!("top_{n}_gram")),
            Reason::DuplicatedNGrams(n) => Cow::Owned(format!("duplicated_{n}_n_grams")),
        }
    }
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_str())
    }
}

/// A filter's decision: keep (`None`) or drop with a reason.
pub type Verdict = Option<Reason>;

/// The exceptions datatrove raises for some inputs, reproduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FilterError {
    /// Python `ZeroDivisionError` (only reachable with thresholds that
    /// disable the guarding check, e.g. `min_doc_words=None` on a document
    /// without words).
    ZeroDivision,
    /// Python `ValueError`: spaCy refuses texts of 2**30 characters or more
    /// (`[E025]`).
    TextTooLong(usize),
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterError::ZeroDivision => f.write_str("division by zero"),
            FilterError::TextTooLong(n) => write!(
                f,
                "[E025] String is too long: {n} characters. Max is 2**30."
            ),
        }
    }
}

impl std::error::Error for FilterError {}

/// A word tokenizer as datatrove uses it (`split_into_words`,
/// `split_into_sentences`). Implement it to add a language.
pub trait WordTokenizer: Sync + fmt::Debug {
    /// The Unicode version whose `str` semantics the filters reproduce.
    fn unicode_version(&self) -> UnicodeVersion;

    /// `split_into_words(text)`, appended to `out`.
    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError>;

    /// The end byte offsets of the sentences `span_tokenize(text)` yields,
    /// appended to `out` (none for a text without tokens).
    fn sentence_ends(&self, text: &str, out: &mut Vec<usize>) -> Result<(), FilterError>;

    /// `len(split_into_sentences(text))`.
    fn count_sentences(&self, text: &str) -> Result<usize, FilterError> {
        let mut ends = Vec::new();
        self.sentence_ends(text, &mut ends)?;
        Ok(ends.len().max(1))
    }
}

impl tokenizer::SpacyEnglish {
    fn check_len(text: &str) -> Result<(), FilterError> {
        // Characters <= bytes: only count when the bytes could reach the limit.
        if text.len() >= tokenizer::MAX_CHARS {
            let n = text::py_len(text);
            if n >= tokenizer::MAX_CHARS {
                return Err(FilterError::TextTooLong(n));
            }
        }
        Ok(())
    }
}

impl WordTokenizer for tokenizer::SpacyEnglish {
    fn unicode_version(&self) -> UnicodeVersion {
        self.version()
    }

    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        Self::check_len(text)?;
        TOKENS.with_borrow_mut(|toks| {
            toks.clear();
            self.tokens(text, toks);
            out.extend(toks.iter().filter(|t| !t.space).map(|t| t.text(text)));
        });
        Ok(())
    }

    fn sentence_ends(&self, text: &str, out: &mut Vec<usize>) -> Result<(), FilterError> {
        Self::check_len(text)?;
        TOKENS.with_borrow_mut(|toks| {
            toks.clear();
            self.tokens(text, toks);
            self.sentence_ends_in(text, toks, out);
        });
        Ok(())
    }
}

thread_local! {
    /// Reused token buffer (the filters run one document at a time per thread).
    static TOKENS: std::cell::RefCell<Vec<tokenizer::Token>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The datatrove language codes the English tokenizer serves: those whose
/// `load_word_tokenizer` is `SpaCyTokenizer("en")` (`"eng"`, the filters'
/// default, `"en"`, and proxies such as `"sco"`).
pub fn english_languages() -> &'static [&'static str] {
    tables::ENGLISH_LANGUAGES
}

/// The tokenizer for a datatrove language code, if pyroid supports it.
pub fn tokenizer_for(
    language: &str,
    version: UnicodeVersion,
) -> Option<&'static dyn WordTokenizer> {
    tables::ENGLISH_LANGUAGES
        .binary_search(&language)
        .ok()
        .map(|_| tokenizer::SpacyEnglish::get(version) as &dyn WordTokenizer)
}

/// Words of one document, tokenized at most once and shared by every filter
/// that needs them.
#[derive(Debug)]
pub struct Doc<'a, 't> {
    text: &'a str,
    tok: &'t dyn WordTokenizer,
    words: Option<Vec<&'a str>>,
}

impl<'a, 't> Doc<'a, 't> {
    /// A document to run filters on.
    pub fn new(text: &'a str, tok: &'t dyn WordTokenizer) -> Self {
        Self {
            text,
            tok,
            words: None,
        }
    }

    /// The document text.
    pub fn text(&self) -> &'a str {
        self.text
    }

    /// The tokenizer.
    pub fn tokenizer(&self) -> &'t dyn WordTokenizer {
        self.tok
    }

    /// `split_into_words(text)`, computed on first use.
    pub fn words(&mut self) -> Result<&[&'a str], FilterError> {
        if self.words.is_none() {
            let mut w = Vec::new();
            self.tok.words(self.text, &mut w)?;
            self.words = Some(w);
        }
        Ok(self.words.as_deref().unwrap_or_default())
    }
}

/// `a / b > t` etc. with Python's int/int true division (both exact in f64).
#[inline]
#[allow(clippy::cast_precision_loss)] // counts are far below 2**53
pub(crate) fn ratio(a: usize, b: usize) -> Result<f64, FilterError> {
    if b == 0 {
        return Err(FilterError::ZeroDivision);
    }
    Ok(a as f64 / b as f64)
}

/// A threshold that datatrove applies only when truthy (`if self.x and ...`).
/// `None` (and Python's 0 / 0.0 / None, mapped by the bindings) disables it.
pub type Threshold = Option<f64>;

#[allow(clippy::cast_precision_loss)]
#[inline]
pub(crate) fn as_f64(n: usize) -> f64 {
    n as f64
}

/// One step of a [`Pipeline`].
#[derive(Clone, Debug)]
pub enum Step {
    GopherQuality(GopherQuality),
    GopherRepetition(GopherRepetition),
    C4Quality(C4Quality),
    FineWebQuality(FineWebQuality),
}

/// A document's result after a [`Pipeline`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome<'a> {
    /// `None` if every step kept the document.
    pub reason: Option<Reason>,
    /// Index of the step that dropped it.
    pub step: Option<usize>,
    /// `doc.text` afterwards: C4 line removal applied by the steps that ran
    /// (borrowed when unchanged).
    pub text: Cow<'a, str>,
}

/// Filters applied in order, as datatrove's executor chains them: the first
/// drop wins, and a C4 step's rewritten text is what later steps see. Steps
/// that see the same text share one tokenization.
#[derive(Clone, Debug)]
pub struct Pipeline {
    pub steps: Vec<(Step, &'static dyn WordTokenizer)>,
}

impl Pipeline {
    /// Run every step on `text`.
    pub fn run<'a>(&self, text: &'a str) -> Result<Outcome<'a>, FilterError> {
        let mut cur: Cow<'a, str> = Cow::Borrowed(text);
        let mut i = 0;
        let mut dropped: Option<(usize, Reason)> = None;
        while i < self.steps.len() && dropped.is_none() {
            // Docs share words only when they share a tokenizer.
            let mut docs: Vec<Doc<'_, '_>> = Vec::new();
            let mut changed: Option<String> = None;
            while i < self.steps.len() {
                let (step, tok) = &self.steps[i];
                let verdict = match step {
                    Step::C4Quality(c) => match c.filter(&cur, *tok)? {
                        c4::C4Outcome::Drop(r) => Some(r),
                        c4::C4Outcome::Keep(Cow::Borrowed(_)) => None,
                        c4::C4Outcome::Keep(Cow::Owned(s)) => {
                            changed = Some(s);
                            i += 1;
                            break;
                        }
                    },
                    other => {
                        let j = if let Some(j) =
                            docs.iter().position(|d| std::ptr::addr_eq(d.tok, *tok))
                        {
                            j
                        } else {
                            docs.push(Doc::new(&cur, *tok));
                            docs.len() - 1
                        };
                        let doc = &mut docs[j];
                        match other {
                            Step::GopherQuality(f) => f.filter(doc)?,
                            Step::GopherRepetition(f) => f.filter(doc)?,
                            Step::FineWebQuality(f) => f.filter(doc)?,
                            Step::C4Quality(_) => unreachable!(),
                        }
                    }
                };
                if let Some(r) = verdict {
                    dropped = Some((i, r));
                    break;
                }
                i += 1;
            }
            drop(docs);
            match changed {
                Some(s) => cur = Cow::Owned(s),
                None => break,
            }
        }
        Ok(Outcome {
            reason: dropped.map(|d| d.1),
            step: dropped.map(|d| d.0),
            text: cur,
        })
    }
}
