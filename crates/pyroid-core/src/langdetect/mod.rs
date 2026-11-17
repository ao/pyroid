// Derived from langdetect 1.0.9 (Apache-2.0), © Cybozu Labs, Inc. and Michal Danilak; modified.
//! A port of langdetect 1.0.9 (`pyroid.langdetect`), itself a port of Cybozu Labs'
//! language-detection: the same text clean-up, n-grams, random draws and floating-point
//! operations, so detections equal langdetect's to the last bit for the same seed.
//!
//! - [`mt`]: CPython's `random.Random` (MT19937, `random`, `choice`, `gauss`).
//! - [`chars`]: `NGram.normalize` and `str.isupper` per Unicode version.
//! - [`text`]: `Detector.append` and `Detector.cleaning_text`.
//! - [`profiles`]: langdetect's `word_lang_prob_map`.
//! - [`detect`]: n-gram extraction and the trials of `_detect_block`.

pub mod chars;
pub mod detect;
pub mod mt;
pub mod profiles;
#[rustfmt::skip]
#[allow(clippy::unreadable_literal)]
mod tables;
pub mod text;

use chars::Upper;
use detect::{Outcome, Params};
use mt::Random;
use profiles::Frozen;

/// `max_text_length`'s default.
pub const MAX_TEXT_LENGTH: usize = 10_000;

/// Why a detection raised.
#[derive(Clone, Debug)]
pub enum Error {
    /// `LangDetectException(CantDetectError, 'No features in text.')`: no n-gram of the text
    /// is in the profiles. `self.langprob` stays `None`.
    NoFeatures,
    /// The trials raised (see [`detect::Failure`]); `langprob` is as far as they got.
    Trials(Outcome),
}

/// `_detect_block` on a detector's text: the cleaned text (if cleaning changed it) and the
/// outcome of the trials.
pub fn detect_block(
    text: &[u32],
    map: &Frozen,
    upper: &Upper,
    params: &Params,
    rng: &mut Random,
    tick: &mut dyn FnMut() -> bool,
) -> (Option<Vec<u32>>, Result<Outcome, Error>) {
    let cleaned = text::clean(text);
    let Some(ngrams) =
        detect::extract_ngrams_ticked(cleaned.as_deref().unwrap_or(text), map, upper, tick)
    else {
        let out = Outcome {
            langprob: Vec::new(),
            failure: Some(detect::Failure::Interrupted),
        };
        return (cleaned, Err(Error::Trials(out)));
    };
    if ngrams.is_empty() {
        return (cleaned, Err(Error::NoFeatures));
    }
    let out = detect::trials(&ngrams, map, params, rng, tick);
    let result = if out.failure.is_some() {
        Err(Error::Trials(out))
    } else {
        Ok(out)
    };
    (cleaned, result)
}

/// `detect_langs(text)` for a fresh detector: `append(text)` with `max_len`, then
/// `_detect_block`. Returns `langprob`.
pub fn detect_text(
    text: &[u32],
    max_len: usize,
    map: &Frozen,
    upper: &Upper,
    params: &Params,
    rng: &mut Random,
    tick: &mut dyn FnMut() -> bool,
) -> Result<Vec<f64>, Error> {
    let appended = text::append(text, max_len);
    detect_block(&appended, map, upper, params, rng, tick)
        .1
        .map(|o| o.langprob)
}
