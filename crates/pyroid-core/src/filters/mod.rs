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
//! [`WordTokenizer`]; [`tokenizer_for`] gives datatrove's tokenizer for a
//! language code where pyroid ports it: [`tokenizer::SpacyTokenizer`] (spaCy
//! `blank(code)` for 57 spaCy languages, `xx` included),
//! `whitespace::Whitespace` and `indic::IndicNlp`. [`backend`] names the
//! tokenizer datatrove uses for any code.
//! Python `str` semantics (code-point lengths, `isspace`, `splitlines`,
//! `isalpha`, `lower`) are reproduced exactly, per Unicode version.

use std::borrow::Cow;
use std::fmt;

mod c4;
mod fineweb;
mod gopher;
pub mod indic;
mod regex;
mod tables;
pub(crate) mod text;
pub mod tokenizer;
pub(crate) mod unicode;
pub mod whitespace;

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
    /// Python `ValueError: substring not found`: datatrove's
    /// `simple_span_tokenize` could not find a sentence in the text (some
    /// sentence splitters return text that is not a substring).
    SubstringNotFound,
    /// A step that tokenizes on its own (C4: per line) ran on pre-tokenized
    /// words ([`Pipeline::run_with_words`]).
    NeedsTokenizer,
    /// The word tokenizer failed (SudachiPy raises for a text that its
    /// normalization makes longer than 65,535 bytes).
    Tokenizer(crate::sudachi::TokenizeError),
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterError::ZeroDivision => f.write_str("division by zero"),
            FilterError::TextTooLong(n) => write!(
                f,
                "[E025] String is too long: {n} characters. Max is 2**30."
            ),
            FilterError::SubstringNotFound => f.write_str("substring not found"),
            FilterError::NeedsTokenizer => {
                f.write_str("this filter needs a word tokenizer, not pre-tokenized words")
            }
            FilterError::Tokenizer(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for FilterError {}

/// A word tokenizer as datatrove uses it (`split_into_words`,
/// `split_into_sentences`). Implement it to add a language.
pub trait WordTokenizer: Sync + fmt::Debug {
    /// The Unicode version whose `str` semantics the filters reproduce.
    fn unicode_version(&self) -> UnicodeVersion;

    /// `split_into_words(text)`, appended to `out`, for a tokenizer whose
    /// words are slices of `text`. One whose words may not be (IndicNLP joins
    /// `"1 , 000"` into `"1,000"`) returns [`FilterError::SubstringNotFound`]
    /// for such a text: use [`Self::words_cow`].
    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError>;

    /// `split_into_words(text)`, appended to `out`; a word may be a new string.
    /// This is what the filters and MinHash use.
    fn words_cow<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) -> Result<(), FilterError> {
        let mut words = Vec::new();
        self.words(text, &mut words)?;
        out.extend(words.into_iter().map(Cow::Borrowed));
        Ok(())
    }

    /// The end byte offsets of the sentences `span_tokenize(text)` yields,
    /// appended to `out` (none for a text without tokens).
    fn sentence_ends(&self, text: &str, out: &mut Vec<usize>) -> Result<(), FilterError>;

    /// `sent_tokenize(text)`: the sentences' texts, stripped, the empty ones
    /// dropped, appended to `out`.
    fn sentences<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) -> Result<(), FilterError> {
        // A sentence's text runs from its first token to its last; the
        // previous sentence ends at most one space (a token's `whitespace_`)
        // before that, which `strip()` removes anyway.
        let mut ends = Vec::new();
        self.sentence_ends(text, &mut ends)?;
        let mut start = 0;
        for end in ends {
            let s = text::py_strip(&text[start..end]);
            if !s.is_empty() {
                out.push(Cow::Borrowed(s));
            }
            start = end;
        }
        Ok(())
    }

    /// `len(split_into_sentences(text))`.
    fn count_sentences(&self, text: &str) -> Result<usize, FilterError> {
        let mut ends = Vec::new();
        self.sentence_ends(text, &mut ends)?;
        Ok(ends.len().max(1))
    }
}

impl tokenizer::SpacyTokenizer {
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

impl WordTokenizer for tokenizer::SpacyTokenizer {
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

    fn words_cow<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) -> Result<(), FilterError> {
        Self::check_len(text)?;
        TOKENS.with_borrow_mut(|toks| {
            toks.clear();
            self.tokens(text, toks);
            out.extend(
                toks.iter()
                    .filter(|t| !t.space)
                    .map(|t| Cow::Borrowed(t.text(text))),
            );
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

/// The word tokenizer datatrove 0.10.1 assigns to a language code
/// (`tokenizer_assignment.csv`, as `load_tokenizer_assignments` reads it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Backend {
    /// datatrove's class: `"SpaCyTokenizer"`, `"StanzaTokenizer"`,
    /// `"IndicNLPTokenizer"`, `"WhitespaceTokenizer"`, `"KiwiTokenizer"`, ...
    pub class: &'static str,
    /// The argument it is built with (a spaCy / Stanza / indic-nlp language
    /// code), or `""`.
    pub arg: &'static str,
}

/// The tokenizer datatrove uses for `language` (any of the 4,343 codes it
/// knows: `"deu_Latn"`, `"deu"`, `"de"`, `"de_Latn"`, ...), or `None` if
/// datatrove has no tokenizer for it (`load_word_tokenizer` raises).
pub fn backend(language: &str) -> Option<Backend> {
    let i = tables::LANGUAGES
        .binary_search_by(|(code, _)| (*code).cmp(language))
        .ok()?;
    let (class, arg) = tables::BACKENDS[usize::from(tables::LANGUAGES[i].1)];
    Some(Backend { class, arg })
}

/// Every datatrove language code with its tokenizer, sorted by code.
pub fn languages() -> impl Iterator<Item = (&'static str, Backend)> {
    tables::LANGUAGES.iter().map(|&(code, b)| {
        let (class, arg) = tables::BACKENDS[usize::from(b)];
        (code, Backend { class, arg })
    })
}

/// The datatrove language codes the English tokenizer serves: those whose
/// `load_word_tokenizer` is `SpaCyTokenizer("en")` (`"eng"`, the filters'
/// default, `"en"`, and proxies such as `"sco"`).
pub fn english_languages() -> &'static [&'static str] {
    static ENGLISH: std::sync::LazyLock<Vec<&'static str>> = std::sync::LazyLock::new(|| {
        languages()
            .filter(|(_, b)| b.class == "SpaCyTokenizer" && b.arg == "en")
            .map(|(code, _)| code)
            .collect()
    });
    &ENGLISH
}

/// The tokenizer for a datatrove language code, if pyroid ports the one
/// datatrove uses (see [`backend`] for the others).
pub fn tokenizer_for(
    language: &str,
    version: UnicodeVersion,
) -> Option<&'static dyn WordTokenizer> {
    let b = backend(language)?;
    match b.class {
        // jieba, as spaCy's ChineseTokenizer runs it.
        "SpaCyTokenizer" if b.arg == "zh" => {
            Some(crate::jieba::zh::ZhTokenizer::get(version) as &dyn WordTokenizer)
        }
        // SudachiPy, as datatrove's JapaneseTokenizer runs it, once a
        // dictionary is installed (`sudachi::ja::install`).
        "SpaCyTokenizer" if b.arg == "ja" => {
            crate::sudachi::ja::JaTokenizer::get(version).map(|t| t as &dyn WordTokenizer)
        }
        "SpaCyTokenizer" => {
            tokenizer::SpacyTokenizer::get(b.arg, version).map(|t| t as &dyn WordTokenizer)
        }
        "WhitespaceTokenizer" => Some(std::sync::LazyLock::force(
            &WHITESPACE[regex::version_index(version)],
        ) as &dyn WordTokenizer),
        "IndicNLPTokenizer" => {
            indic::IndicNlp::get(b.arg, version).map(|t| t as &dyn WordTokenizer)
        }
        _ => None,
    }
}

/// datatrove's `WhitespaceTokenizer` per Unicode version (words: spaCy `xx`).
static WHITESPACE: [std::sync::LazyLock<whitespace::Whitespace>; 5] = {
    fn make(v: UnicodeVersion) -> whitespace::Whitespace {
        whitespace::Whitespace::new(
            tokenizer::SpacyTokenizer::get("xx", v).expect("spaCy xx rules"),
        )
    }
    [
        std::sync::LazyLock::new(|| make(UnicodeVersion::V14_0)),
        std::sync::LazyLock::new(|| make(UnicodeVersion::V15_0)),
        std::sync::LazyLock::new(|| make(UnicodeVersion::V15_1)),
        std::sync::LazyLock::new(|| make(UnicodeVersion::V16_0)),
        std::sync::LazyLock::new(|| make(UnicodeVersion::V17_0)),
    ]
};

/// Words of one document, tokenized at most once and shared by every filter
/// that needs them.
#[derive(Debug)]
pub struct Doc<'a, 't> {
    text: &'a str,
    tok: &'t dyn WordTokenizer,
    words: Option<Vec<Cow<'a, str>>>,
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

    /// A document whose words are already known (from a tokenizer that runs
    /// outside pyroid, such as datatrove's own in Python). `tok` only
    /// supplies the Unicode version; its `words` is never called.
    pub fn with_words(text: &'a str, words: Vec<&'a str>, tok: &'t dyn WordTokenizer) -> Self {
        Self {
            text,
            tok,
            words: Some(words.into_iter().map(Cow::Borrowed).collect()),
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
    pub fn words(&mut self) -> Result<&[Cow<'a, str>], FilterError> {
        if self.words.is_none() {
            let mut w = Vec::new();
            self.tok.words_cow(self.text, &mut w)?;
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
    /// [`Self::run`] for a document whose words are already known (tokenized
    /// outside pyroid: by datatrove's own tokenizer, in Python, for the
    /// languages pyroid does not port). The text is never rewritten, so the
    /// pipeline must not contain [`Step::C4Quality`] (C4 tokenizes lines):
    /// that is [`FilterError::NeedsTokenizer`].
    pub fn run_with_words<'a>(
        &self,
        text: &'a str,
        words: Vec<&'a str>,
    ) -> Result<Outcome<'a>, FilterError> {
        let Some((_, tok)) = self.steps.first() else {
            return Ok(Outcome {
                reason: None,
                step: None,
                text: Cow::Borrowed(text),
            });
        };
        let mut doc = Doc::with_words(text, words, *tok);
        for (i, (step, _)) in self.steps.iter().enumerate() {
            let verdict = match step {
                Step::GopherQuality(f) => f.filter(&mut doc)?,
                Step::GopherRepetition(f) => f.filter(&mut doc)?,
                Step::FineWebQuality(f) => f.filter(&mut doc)?,
                Step::C4Quality(_) => return Err(FilterError::NeedsTokenizer),
            };
            if let Some(r) = verdict {
                return Ok(Outcome {
                    reason: Some(r),
                    step: Some(i),
                    text: Cow::Borrowed(text),
                });
            }
        }
        Ok(Outcome {
            reason: None,
            step: None,
            text: Cow::Borrowed(text),
        })
    }

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

// ---------------------------------------------------------------------------
// More datatrove 0.10.1 steps, one module each.
// ---------------------------------------------------------------------------
pub mod badwords;
pub mod pii;
pub mod symbol_lines;
pub mod url;
