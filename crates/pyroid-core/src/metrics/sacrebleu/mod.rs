// Derived from sacrebleu 2.6.0 (Apache-2.0), © Amazon.com, Inc. or its affiliates; modified.
//! BLEU and chrF statistics as computed by sacrebleu 2.6.0 (`pyroid.metrics.sacrebleu`).
//!
//! The kernels compute sacrebleu's integer sufficient statistics: BLEU's
//! `[hyp_len, ref_len, correct.., total..]` and chrF's best-reference
//! `[n_hyp, n_ref, n_match]` triples, per segment and summed. The scores
//! (`BLEU.compute_bleu`, `CHRF._compute_f_score`) are computed from them in
//! Python, so they are exactly sacrebleu's on every CPython; the only float
//! arithmetic here is chrF's per-reference f-score, which picks the best of
//! several references and is a step-for-step port ([`chrf::f_score`]).
//!
//! - [`pystr`]: Python's `str.split()` / `strip()` / `rstrip()` whitespace.
//! - [`tokenizers`]: the `13a`, `intl`, `zh`, `char` and `none` tokenizers
//!   (tables for `intl` and `zh` generated from sacrebleu and `regex` by
//!   `generate_tables.py`).
//! - [`ngrams`]: n-gram counting over interned ids with packed keys.
//! - [`bleu`], [`chrf`]: per-segment and corpus statistics, and the
//!   pre-computed reference caches (`BLEU(references=...)`).

pub mod bleu;
pub mod chrf;
pub mod ngrams;
pub mod pystr;
mod tables;
pub mod tokenizers;

pub use bleu::{BleuConfig, BleuRefCache, BleuStats};
pub use chrf::{ChrfConfig, ChrfRefCache, ChrfStats};
pub use tables::{REGEX_UNICODE_VERSION, REGEX_VERSION, SACREBLEU_VERSION};
pub use tokenizers::{IntlClass, Tokenizer, intl_class, is_chinese_char};
