//! Evaluation metrics for generated text, scored exactly as the incumbents do.
//!
//! - [`rouge`]: ROUGE-N (rouge1..rouge9), ROUGE-L and ROUGE-Lsum as computed by
//!   rouge-score 0.1.2, with its default tokenizer and NLTK 3.10.4's Porter
//!   stemmer (`pyroid.metrics.rouge_scorer`).
//! - [`sacrebleu`]: BLEU (tokenizers `13a`, `intl`, `char`, `zh`, `none`) and
//!   chrF / chrF++ as computed by sacrebleu 2.6.0 (`pyroid.metrics.sacrebleu`).
//!
//! Kernels return the incumbents' integer statistics and IEEE-identical
//! floats; the bindings build the incumbents' result objects.

pub mod rouge;
pub mod sacrebleu;
