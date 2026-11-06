// Derived from rouge-score 0.1.2 (Apache-2.0), © The rouge_score Authors; modified.
//! ROUGE as computed by rouge-score 0.1.2 (`rouge_score.rouge_scorer`):
//! ROUGE-N (`rouge1`..`rouge9`), ROUGE-L and ROUGE-Lsum, with rouge-score's
//! default tokenizer ([`tokenize`]) and NLTK 3.10.4's Porter stemmer
//! ([`porter`]).
//!
//! Scores are rouge-score's floats, bit for bit: the same integer counts,
//! divided and combined in the same operation order. Where rouge-score
//! returns `Score(precision=0, recall=0, fmeasure=0)` with Python *ints*
//! (ROUGE-L and ROUGE-Lsum when a side has no tokens), the result is
//! [`Score::IntZero`].
//!
//! ```
//! use pyroid_core::metrics::rouge::{RougeType, Score, Scorer};
//! let types = ["rouge1", "rougeL"].map(|t| RougeType::parse(t).unwrap());
//! let scorer = Scorer::new(types.to_vec(), true);
//! let scores = scorer.score("The quick brown fox", "the quick fox jumps");
//! assert_eq!(scores[0], Score::Float { precision: 0.75, recall: 0.75, fmeasure: 0.75 });
//! assert_eq!(scorer.score("", "x")[1], Score::IntZero);
//! ```

pub mod porter;
mod score;
pub mod tokenize;

pub use score::fmeasure;
use score::{Interner, Ngrams, Scratch, rouge_l, rouge_lsum, rouge_n};
use tokenize::Tokens;

/// A ROUGE type of `RougeScorer(rouge_types)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RougeType {
    /// `rouge<n>`: n-gram overlap (`n >= 1`; rouge-score accepts `1..=9`).
    N(usize),
    /// `rougeL`: longest common subsequence of the whole texts.
    L,
    /// `rougeLsum`: summary-level (union) LCS over sentences.
    LSum,
}

/// Why rouge-score rejects a rouge type (both are `ValueError`s).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TypeError {
    /// `rouge0`: "rougen requires positive n: %s".
    NonPositive,
    /// Anything else: "Invalid rouge type: %s".
    Invalid,
}

impl RougeType {
    /// Parse a type as `RougeScorer.score` does: `"rougeL"`, `"rougeLsum"`,
    /// or `re.match(r"rouge[0-9]$", t)` then `int(t[5:])`. As in Python, `$`
    /// also matches before a final newline, so `"rouge2\n"` is ROUGE-2.
    ///
    /// ```
    /// use pyroid_core::metrics::rouge::{RougeType, TypeError};
    /// assert_eq!(RougeType::parse("rouge2"), Ok(RougeType::N(2)));
    /// assert_eq!(RougeType::parse("rouge2\n"), Ok(RougeType::N(2)));
    /// assert_eq!(RougeType::parse("rouge0"), Err(TypeError::NonPositive));
    /// assert_eq!(RougeType::parse("rouge10"), Err(TypeError::Invalid));
    /// ```
    pub fn parse(s: &str) -> Result<Self, TypeError> {
        match s {
            "rougeL" => Ok(Self::L),
            "rougeLsum" => Ok(Self::LSum),
            _ => match s.as_bytes() {
                [b'r', b'o', b'u', b'g', b'e', d @ b'0'..=b'9']
                | [b'r', b'o', b'u', b'g', b'e', d @ b'0'..=b'9', b'\n'] => match d - b'0' {
                    0 => Err(TypeError::NonPositive),
                    n => Ok(Self::N(usize::from(n))),
                },
                _ => Err(TypeError::Invalid),
            },
        }
    }
}

/// One `Score(precision, recall, fmeasure)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Score {
    /// `Score(precision=0, recall=0, fmeasure=0)` with Python ints.
    IntZero,
    /// Floats, as rouge-score computes them.
    Float {
        precision: f64,
        recall: f64,
        fmeasure: f64,
    },
}

impl Score {
    /// `(precision, recall, fmeasure)` as floats.
    pub fn values(self) -> (f64, f64, f64) {
        match self {
            Self::IntZero => (0.0, 0.0, 0.0),
            Self::Float {
                precision,
                recall,
                fmeasure,
            } => (precision, recall, fmeasure),
        }
    }
}

/// `RougeScorer(rouge_types, use_stemmer)` with rouge-score's default
/// tokenizer: scores target/prediction texts.
#[derive(Clone, Debug)]
pub struct Scorer {
    types: Vec<RougeType>,
    stem: bool,
}

impl Scorer {
    /// A scorer for `types` (in order; repeats allowed).
    ///
    /// # Panics
    /// If a type is `N(0)`.
    pub fn new(types: Vec<RougeType>, use_stemmer: bool) -> Self {
        assert!(
            !types.contains(&RougeType::N(0)),
            "rougen requires positive n"
        );
        Self {
            types,
            stem: use_stemmer,
        }
    }

    /// The types scored, in order.
    pub fn types(&self) -> &[RougeType] {
        &self.types
    }

    /// `RougeScorer.score(target, prediction)`: one score per type, in order.
    /// rougeLsum sentences are the texts' `"\n"`-separated lines.
    pub fn score(&self, target: &str, prediction: &str) -> Vec<Score> {
        self.score_impl::<&str>(target, prediction, None)
    }

    /// [`Scorer::score`] with the rougeLsum sentences given (rouge-score's
    /// `split_summaries=True`, where nltk splits them; empty ones already
    /// dropped or not, either way). The texts are only tokenized for the
    /// other types.
    pub fn score_with_sentences<S: AsRef<str>>(
        &self,
        target: &str,
        prediction: &str,
        target_sentences: &[S],
        prediction_sentences: &[S],
    ) -> Vec<Score> {
        self.score_impl(
            target,
            prediction,
            Some((target_sentences, prediction_sentences)),
        )
    }

    #[allow(clippy::similar_names)] // t_/p_ (target/prediction) and ts_/ps_ (their sentences)
    fn score_impl<S: AsRef<str>>(
        &self,
        target: &str,
        prediction: &str,
        sentences: Option<(&[S], &[S])>,
    ) -> Vec<Score> {
        let only_lsum = self.types.iter().all(|t| *t == RougeType::LSum);
        let mut texts = (Tokens::default(), Tokens::default());
        if sentences.is_none() || !only_lsum {
            texts.0.push_text(target, true);
            texts.1.push_text(prediction, true);
        }
        let mut sents = (Tokens::default(), Tokens::default());
        let mut bounds = (Vec::new(), Vec::new());
        if let Some((t, p)) = sentences {
            for s in t {
                sents.0.push_text(s.as_ref(), true);
                bounds.0.push(sents.0.len());
            }
            for s in p {
                sents.1.push_text(s.as_ref(), true);
                bounds.1.push(sents.1.len());
            }
        }
        let mut interner = Interner::new(self.stem);
        let t_ids = token_ids(&mut interner, &texts.0);
        let p_ids = token_ids(&mut interner, &texts.1);
        let ts_ids = token_ids(&mut interner, &sents.0);
        let ps_ids = token_ids(&mut interner, &sents.1);
        let vocab = interner.len();
        let (t_sents, p_sents) = if sentences.is_some() {
            (split_at(&ts_ids, &bounds.0), split_at(&ps_ids, &bounds.1))
        } else {
            (
                split_at(&t_ids, texts.0.line_starts()),
                split_at(&p_ids, texts.1.line_starts()),
            )
        };
        let mut scratch = Scratch::default();
        let mut orders: Vec<usize> = self
            .types
            .iter()
            .filter_map(|t| match t {
                RougeType::N(n) => Some(*n),
                _ => None,
            })
            .collect();
        orders.sort_unstable();
        orders.dedup();
        let mut ngrams = Vec::with_capacity(orders.len());
        if !orders.is_empty() {
            let mut grams = Ngrams::new(&t_ids, &p_ids, vocab);
            for n in orders {
                ngrams.push((n, grams.score(n, &mut scratch)));
            }
        }
        let lcs = self
            .types
            .contains(&RougeType::L)
            .then(|| rouge_l(&t_ids, &p_ids, vocab, &mut scratch));
        let lsum = self
            .types
            .contains(&RougeType::LSum)
            .then(|| rouge_lsum(&t_sents, &p_sents, vocab, &mut scratch));
        self.types
            .iter()
            .map(|t| match t {
                RougeType::N(n) => ngrams
                    .iter()
                    .find(|(k, _)| k == n)
                    .map(|(_, s)| *s)
                    .expect("every order was scored"),
                RougeType::L => lcs.expect("rougeL was scored"),
                RougeType::LSum => lsum.expect("rougeLsum was scored"),
            })
            .collect()
    }
}

fn token_ids<'a>(interner: &mut Interner<'a>, tokens: &'a Tokens) -> Vec<u32> {
    tokens.iter().map(|t| interner.id(t)).collect()
}

/// The non-empty runs of `ids` between the token indices `cuts` (increasing).
fn split_at<'a>(ids: &'a [u32], cuts: &[usize]) -> Vec<&'a [u32]> {
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut start = 0;
    for &end in cuts.iter().chain(std::iter::once(&ids.len())) {
        if end > start {
            out.push(&ids[start..end]);
        }
        start = start.max(end);
    }
    out
}

/// Interned ids of token lists (a custom tokenizer's output; no stemming).
fn intern_lists<'a, S: AsRef<[u8]> + 'a>(
    interner: &mut Interner<'a>,
    lists: &[&'a [S]],
) -> Vec<Vec<u32>> {
    lists
        .iter()
        .map(|l| l.iter().map(|t| interner.id(t.as_ref())).collect())
        .collect()
}

/// rouge-score's `_score_ngrams(_create_ngrams(target, n), _create_ngrams(prediction, n))`
/// for token lists (`n >= 1`).
pub fn rouge_n_tokens<S: AsRef<[u8]>>(target: &[S], prediction: &[S], n: usize) -> Score {
    assert!(n >= 1, "rougen requires positive n");
    let mut interner = Interner::new(false);
    let ids = intern_lists(&mut interner, &[target, prediction]);
    rouge_n(&ids[0], &ids[1], n, interner.len(), &mut Scratch::default())
}

/// rouge-score's `_score_lcs(target, prediction)` for token lists.
pub fn rouge_l_tokens<S: AsRef<[u8]>>(target: &[S], prediction: &[S]) -> Score {
    let mut interner = Interner::new(false);
    let ids = intern_lists(&mut interner, &[target, prediction]);
    rouge_l(&ids[0], &ids[1], interner.len(), &mut Scratch::default())
}

/// rouge-score's `_summary_level_lcs(target_sentences, prediction_sentences)`
/// for lists of token lists.
pub fn rouge_lsum_tokens<S: AsRef<[u8]>, L: AsRef<[S]>>(target: &[L], prediction: &[L]) -> Score {
    let mut interner = Interner::new(false);
    let t: Vec<&[S]> = target.iter().map(AsRef::as_ref).collect();
    let p: Vec<&[S]> = prediction.iter().map(AsRef::as_ref).collect();
    let t_ids = intern_lists(&mut interner, &t);
    let p_ids = intern_lists(&mut interner, &p);
    let t_refs: Vec<&[u32]> = t_ids.iter().map(Vec::as_slice).collect();
    let p_refs: Vec<&[u32]> = p_ids.iter().map(Vec::as_slice).collect();
    rouge_lsum(&t_refs, &p_refs, interner.len(), &mut Scratch::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn types(names: &[&str]) -> Vec<RougeType> {
        names.iter().map(|t| RougeType::parse(t).unwrap()).collect()
    }

    fn f(precision: f64, recall: f64) -> Score {
        Score::Float {
            precision,
            recall,
            fmeasure: fmeasure(precision, recall),
        }
    }

    #[test]
    fn parse_types() {
        assert_eq!(RougeType::parse("rouge1"), Ok(RougeType::N(1)));
        assert_eq!(RougeType::parse("rouge9"), Ok(RougeType::N(9)));
        assert_eq!(RougeType::parse("rouge0\n"), Err(TypeError::NonPositive));
        for bad in [
            "rougen",
            "rouge10",
            "rouge",
            "Rouge1",
            "rouge1 ",
            "rouge1\n\n",
            "rougel",
            "rougeLSum",
            "rougeL\n",
            "rouge١",
            "",
        ] {
            assert_eq!(RougeType::parse(bad), Err(TypeError::Invalid), "{bad:?}");
        }
    }

    #[test]
    fn rouge_score_test_cases() {
        let s = Scorer::new(types(&["rouge1"]), false);
        assert_eq!(s.score("testing one two", "testing"), [f(1.0, 1.0 / 3.0)]);
        let s = Scorer::new(types(&["rouge2"]), false);
        assert_eq!(s.score("testing one two", "testing one"), [f(1.0, 0.5)]);
        let s = Scorer::new(types(&["rougeL"]), false);
        assert_eq!(
            s.score("testing one two", "testing one"),
            [f(1.0, 2.0 / 3.0)]
        );
        assert_eq!(
            s.score("testing one two", "testing two"),
            [f(1.0, 2.0 / 3.0)]
        );
        let s = Scorer::new(types(&["rouge1", "rouge2", "rougeL", "rougeLsum"]), false);
        assert_eq!(
            s.score("testing one two", ""),
            [f(0.0, 0.0), f(0.0, 0.0), Score::IntZero, Score::IntZero]
        );
        let s = Scorer::new(types(&["rougeLsum"]), false);
        assert_eq!(
            s.score("w1 w2 w3 w4 w5", "w1 w2 w6 w7 w8\nw1 w3 w8 w9 w5"),
            [f(0.4, 0.8)]
        );
        assert_eq!(s.score("w1 w2 w3 w4 w5", "/"), [Score::IntZero]);
        assert_eq!(s.score("", "w1"), [Score::IntZero]);
        let s = Scorer::new(types(&["rougeLsum"]), true);
        assert_eq!(
            s.score(
                "First sentence.\nSecond Sentence.",
                "Second sentence.\nFirst Sentence."
            ),
            [f(1.0, 1.0)]
        );
        assert_eq!(
            s.score(
                "First sentence. Second Sentence.",
                "Second sentence. First Sentence."
            ),
            [f(0.5, 0.5)]
        );
        assert_eq!(
            s.score_with_sentences(
                "First sentence. Second Sentence.",
                "Second sentence. First Sentence.",
                &["First sentence.", "Second Sentence."],
                &["Second sentence.", "First Sentence."],
            ),
            [f(1.0, 1.0)]
        );
    }

    #[test]
    fn repeated_and_unordered_types() {
        let s = Scorer::new(types(&["rouge2", "rougeL", "rouge1", "rouge2"]), false);
        let got = s.score("a b c a b", "a b a c");
        let one = |t: &str| Scorer::new(types(&[t]), false).score("a b c a b", "a b a c")[0];
        assert_eq!(
            got,
            [one("rouge2"), one("rougeL"), one("rouge1"), one("rouge2")]
        );
    }

    #[test]
    fn token_lists() {
        let t = ["1", "2", "3", "4", "5"];
        let c1 = ["2", "5", "3", "4"];
        assert_eq!(rouge_l_tokens(&t, &c1), f(0.75, 0.6));
        assert_eq!(rouge_n_tokens(&t, &c1, 1), f(1.0, 0.8));
        assert_eq!(rouge_l_tokens::<&str>(&[], &["x"]), Score::IntZero);
        let refs = [vec!["1", "2", "3", "4", "5"]];
        let cands = [vec!["1", "2", "6", "7", "8"], vec!["1", "3", "8", "9", "5"]];
        assert_eq!(rouge_lsum_tokens(&refs, &cands), f(0.4, 0.8));
        // Arbitrary strings, not only [a-z0-9].
        assert_eq!(rouge_n_tokens(&["Ab", "ß"], &["ß"], 1), f(1.0, 0.5));
    }

    #[test]
    fn split_at_cuts() {
        let ids = [1, 2, 3, 4];
        assert_eq!(split_at(&ids, &[0, 2, 2, 4]), [&[1, 2][..], &[3, 4]]);
        assert_eq!(split_at(&[], &[0]), Vec::<&[u32]>::new());
    }
}
