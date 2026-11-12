# Derived from sacrebleu 2.6.0 (Apache-2.0), © Amazon.com, Inc. or its affiliates; modified.
r"""A drop-in, Rust-backed replacement for sacrebleu 2.6.0's BLEU and chrF API.

```python
import pyroid.metrics.sacrebleu as sacrebleu

bleu = sacrebleu.corpus_bleu(hyps, [refs])  # BLEUScore, exactly sacrebleu's
print(bleu)  # the same line sacrebleu prints: "BLEU = ... (BP = ... ratio = ...)"
chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF2++
sacrebleu.sentence_bleu_batch(hyps, [refs], threads=8)  # one BLEUScore per pair
```

Every score, statistic, ``str()``/``repr()``, ``format()`` output, signature string, exception
and ``sacrebleu`` log warning is sacrebleu 2.6.0's. ``__version__``, like the ``version:``
field of signatures, is the sacrebleu API version reproduced. The integer n-gram statistics
are computed in Rust, in parallel on pyroid's thread pool. The score formulas
(``BLEU.compute_bleu``, ``CHRF._compute_f_score``), result classes and argument checks are
sacrebleu's own Python code, so scores are bit-for-bit sacrebleu's on the same interpreter.
To pick the best of several references, chrF uses a step-for-step Rust port of the f-score,
with the same operations in the same order.

pyroid handles these natively:

- the tokenizers ``13a`` (the default), ``intl``, ``zh`` (also via ``trg_lang="zh"``),
  ``char`` and ``none``;
- every smoothing method and value, ``lowercase`` (Python's ``str.lower()``), ``force``,
  ``effective_order`` and ``max_ngram_order``;
- several references per segment, including ``None`` holes (``nrefs:var``);
- chrF and chrF++ with any ``char_order``, ``word_order``, ``beta``, ``whitespace`` and
  ``eps_smoothing``;
- pre-computed references (``BLEU(references=...)``, ``CHRF(references=...)``).

Degenerate settings (e.g. ``max_ngram_order=0``, a non-int order,
``char_order + word_order == 0``) run sacrebleu's own code in pure Python, with sacrebleu's
results and errors.

These are delegated to the real sacrebleu, with one :class:`pyroid.FallbackWarning` per
function and reason: the ``ja-mecab``, ``ko-mecab``, ``spm``, ``flores101``, ``flores200``
and ``spBLEU-1K`` tokenizers (and ``trg_lang`` ``"ja"``/``"ko"``),
``corpus_score(..., n_bootstrap > 1)``, :class:`TER`, :func:`corpus_ter`,
:func:`sentence_ter`, :data:`DATASETS` and the test set helpers (``DELEGATED``). They need
sacrebleu installed; without it they raise :class:`pyroid.PyroidError` with an install hint.
Importing this module never imports sacrebleu.

pyroid extensions (keyword-only): ``threads=`` on :func:`corpus_bleu`,
:func:`raw_corpus_bleu`, :func:`corpus_chrf` and ``corpus_score`` (``None``: the whole pool;
``n``: at most ``n`` tasks; ``1``: the calling thread), and :func:`sentence_bleu_batch` /
:func:`sentence_chrf_batch`.

Differences from sacrebleu:

- Lone surrogates (``"\ud800"``) in a hypothesis or reference raise
  ``UnicodeEncodeError`` (with a note naming the item), as everywhere in pyroid. sacrebleu
  treats them as code points. Remove them first, e.g.
  ``s.encode("utf-8", "replace").decode("utf-8")``.
- Wrong item types that sacrebleu's own argument checks let through (a non-str hypothesis
  after the first, a non-str, non-``None`` reference) raise :class:`pyroid.InputError` (a
  ``TypeError``) naming the item. sacrebleu raises an ``AttributeError`` from deep inside.
  sacrebleu's checks themselves (``_check_*_args``) raise the same ``TypeError``s with the
  same messages.
- ``BLEU.tokenizer`` is a pyroid tokenizer object for the native tokenizers (``signature()``
  and ``__call__(line)``, the same output as sacrebleu's), and sacrebleu's own tokenizer
  object for the delegated ones.
- Pre-computed references are held in a native object. Changing ``max_ngram_order``,
  ``char_order`` or ``word_order`` of a metric after constructing it with ``references=``
  raises ``ValueError`` (sacrebleu would mix the old reference n-grams with new hypothesis
  ones).
- :func:`sentence_bleu_batch` logs sacrebleu's ``effective_order`` recommendation once per
  call, not once per sentence.
"""

from abc import ABCMeta
from collections import Counter
from collections.abc import KeysView, Mapping, Sequence
from typing import IO, Any, ClassVar, Final, Protocol, overload

from pyroid._typing import ArrowStrings

__all__ = [
    "BLEU",
    "CHRF",
    "DATASETS",
    "DELEGATED",
    "SACREBLEU_DIR",
    "TER",
    "BLEUScore",
    "BLEUSignature",
    "CHRFScore",
    "CHRFSignature",
    "Metric",
    "Score",
    "Signature",
    "__version__",
    "corpus_bleu",
    "corpus_chrf",
    "corpus_ter",
    "download_test_set",
    "extract_char_ngrams",
    "extract_word_ngrams",
    "get_available_testsets",
    "get_langpairs_for_testset",
    "get_reference_files",
    "get_source_file",
    "raw_corpus_bleu",
    "sentence_bleu",
    "sentence_bleu_batch",
    "sentence_chrf",
    "sentence_chrf_batch",
    "sentence_ter",
    "smart_open",
]

__version__: Final[str]
"""The sacrebleu release whose API and results this module reproduces: ``"2.6.0"``."""

DELEGATED: Final[tuple[str, ...]]
"""Names that are forwarded to the real sacrebleu (it must be installed)."""

SACREBLEU_DIR: str
"""Where sacrebleu stores test sets: ``$SACREBLEU``, else ``~/.sacrebleu`` (as sacrebleu)."""

DATASETS: Mapping[str, Any]
"""sacrebleu's test set registry (read-only; delegated to sacrebleu on first use)."""

class _Tokenizer(Protocol):
    def signature(self) -> str: ...
    def __call__(self, line: str, /) -> str: ...

# ---------------------------------------------------------------------------
# Scores and signatures
# ---------------------------------------------------------------------------

class Score:
    """The result of a metric (sacrebleu's ``Score``).

    ``str(score)`` and ``repr(score)`` are ``score.format()``.

    Args:
        name: The name of the underlying metric.
        score: A floating point number for the final metric.
    """

    name: str
    score: float
    def __init__(self, name: str, score: float) -> None: ...
    def format(
        self,
        width: int = 2,
        score_only: bool = False,
        signature: str = "",
        is_json: bool = False,
    ) -> str:
        """A pretty representation of the score, as sacrebleu prints it.

        Args:
            width: Floating point decimal precision width.
            score_only: If ``True``, and the format is not ``json``, returns a single score
                string.
            signature: A string representation of the metric's signature
                (``str(metric.get_signature())``).
            is_json: If ``True``, will output the score in JSON string.

        Returns:
            A plain or JSON-formatted string, e.g.
            ``"BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0 = ..."``.
        """

    def estimate_ci(self, scores: list[Score]) -> None:
        """Store the mean and 95% confidence interval of bootstrap resampled ``scores``.

        Args:
            scores: Scores of bootstrap resamples of the corpus.
        """

class Signature:
    """A sacreBLEU reproducibility signature; ``str()`` is ``format()``.

    Args:
        args: The metric's ``__dict__`` (must hold ``num_refs``: evaluate the metric first).

    Raises:
        ValueError: The metric has not been evaluated yet.
    """

    info: dict[str, Any]
    def __init__(self, args: dict[str, Any]) -> None: ...
    def format(self, short: bool = False) -> str:
        """The signature as a string, e.g. ``nrefs:1|case:mixed|...|version:2.6.0``.

        Args:
            short: If True, the abbreviated keys (``#:1|c:mixed|...|v:2.6.0``).

        Returns:
            The signature string.
        """

    def update(self, key: str, value: Any) -> None:
        """Add a new item or update an existing one.

        Args:
            key: The key to use in the dictionary.
            value: The associated value for the ``key``.
        """

class BLEUSignature(Signature):
    """The signature of a BLEU metric (``case``, ``eff``, ``tok``, ``smooth`` fields)."""

class CHRFSignature(Signature):
    """The signature of a chrF metric (``case``, ``eff``, ``nc``, ``nw``, ``space`` fields)."""

class BLEUScore(Score):
    """A BLEU score with its statistics.

    Args:
        score: The BLEU score.
        counts: Counts of correct n-grams, 1 <= n <= max_ngram_order (with add-k smoothing,
            including the added k, as in sacrebleu).
        totals: Counts of total n-grams, 1 <= n <= max_ngram_order.
        precisions: n-gram precisions (percentages).
        bp: The brevity penalty.
        sys_len: The cumulative system length.
        ref_len: The cumulative reference length.
    """

    bp: float
    counts: list[int]
    totals: list[int]
    sys_len: int
    ref_len: int
    precisions: list[float]
    prec_str: str
    ratio: float
    def __init__(
        self,
        score: float,
        counts: list[int],
        totals: list[int],
        precisions: list[float],
        bp: float,
        sys_len: int,
        ref_len: int,
    ) -> None: ...

class CHRFScore(Score):
    """A chrF score (named ``chrF2``, ``chrF2++``, ... after ``beta`` and ``word_order``).

    Args:
        score: The chrF (chrF++) score.
        char_order: The character n-gram order.
        word_order: The word n-gram order. If equals to 2, the metric is referred to as
            chrF++.
        beta: Determine the importance of recall w.r.t precision.
    """

    beta: int
    char_order: int
    word_order: int
    def __init__(self, score: float, char_order: int, word_order: int, beta: int) -> None: ...

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class Metric(metaclass=ABCMeta):
    """The base class of :class:`BLEU` and :class:`CHRF` (sacrebleu's ``Metric``)."""

    n_bootstrap: int | None
    seed: str | None
    num_refs: int
    """References per segment of the last evaluation (or the cache); -1 if they vary."""
    def __init__(self) -> None: ...
    def sentence_score(self, hypothesis: str, references: Sequence[str | None]) -> Score:
        """Score one hypothesis against its reference(s).

        Args:
            hypothesis: A single hypothesis string.
            references: Its reference strings (``None`` entries are skipped).

        Returns:
            The score.

        Raises:
            TypeError: sacrebleu's argument checks (e.g. ``hypothesis`` is not a str).
        """

    def corpus_score(
        self,
        hypotheses: Sequence[str],
        references: Sequence[Sequence[str | None]] | None,
        n_bootstrap: int = 1,
        *,
        threads: int | None = None,
    ) -> Score:
        """Score a corpus against one or more reference streams.

        Hypotheses and reference segments are zipped, as in sacrebleu: extra items of the
        longer one are ignored.

        Args:
            hypotheses: A sequence of hypothesis strings.
            references: Reference streams: ``references[k][i]`` is the k-th reference of
                hypothesis i (``None`` for a missing one). ``None`` (or empty) uses the
                references the metric was constructed with.
            n_bootstrap: If > 1, also estimate a 95% confidence interval by bootstrap
                resampling with that many samples (delegated to sacrebleu).
            threads: pyroid extension: ``None`` uses the whole thread pool, ``n`` at most
                ``n`` tasks at once, ``1`` the calling thread. Results are the same.

        Returns:
            The corpus score.

        Raises:
            TypeError: sacrebleu's argument checks.
            RuntimeError: No references given and none cached.
        """

    def get_signature(self) -> Signature:
        """The reproducibility signature (available once the metric has been evaluated).

        Raises:
            ValueError: The metric has not been evaluated yet.
        """

    def _extract_corpus_statistics(
        self, hypotheses: Sequence[str], references: Sequence[Sequence[str | None]] | None
    ) -> list[list[Any]]: ...

class BLEU(Metric):
    """The BLEU metric (``sacrebleu.BLEU``).

    Args:
        lowercase: If True, lowercased BLEU is computed.
        force: Ignore data that looks already tokenized (otherwise 100 hypotheses ending in
            ``" ."`` log a warning on the ``sacrebleu`` logger).
        tokenize: The tokenizer (one of ``TOKENIZERS``). If None, ``"13a"``, or the
            language's tokenizer for ``trg_lang`` ``"zh"``, ``"ja"``, ``"ko"``.
        smooth_method: ``"floor"``, ``"add-k"``, ``"exp"`` or ``"none"``.
        smooth_value: The value for ``floor`` and ``add-k`` (None: 0.1 and 1).
        max_ngram_order: The maximum n-gram order (default 4).
        effective_order: Stop including n-gram orders for which precision is 0 (use it for
            sentence-level BLEU).
        trg_lang: An optional language code to pick (or warn about) the tokenizer.
        references: Reference streams to pre-compute for scoring many systems.

    Raises:
        AssertionError: Unknown ``smooth_method`` (as sacrebleu).
        KeyError: Unknown ``tokenize``.
    """

    SMOOTH_DEFAULTS: ClassVar[dict[str, float | None]]
    TOKENIZERS: ClassVar[KeysView[str]]
    TOKENIZER_DEFAULT: ClassVar[str]
    trg_lang: str
    lowercase: bool
    smooth_value: float | None
    smooth_method: str
    max_ngram_order: int
    effective_order: bool
    tokenizer: _Tokenizer
    tokenizer_signature: str
    def __init__(
        self,
        lowercase: bool = False,
        force: bool = False,
        tokenize: str | None = None,
        smooth_method: str = "exp",
        smooth_value: float | None = None,
        max_ngram_order: int = 4,
        effective_order: bool = False,
        trg_lang: str = "",
        references: Sequence[Sequence[str | None]] | None = None,
    ) -> None: ...
    @staticmethod
    def compute_bleu(
        correct: list[int],
        total: list[int],
        sys_len: int,
        ref_len: int,
        smooth_method: str = "none",
        smooth_value: float | None = None,
        effective_order: bool = False,
        max_ngram_order: int = 4,
    ) -> BLEUScore:
        """BLEU from its sufficient statistics, with smoothing (sacrebleu's own code).

        Args:
            correct: Counts of correct n-grams, 1 <= n <= max_ngram_order (modified in place
                by ``add-k``).
            total: Counts of total n-grams, 1 <= n <= max_ngram_order.
            sys_len: The cumulative system length.
            ref_len: The cumulative reference length.
            smooth_method: ``"floor"``, ``"add-k"``, ``"exp"`` or ``"none"``.
            smooth_value: The value for ``floor`` and ``add-k`` (None: the default).
            effective_order: Stop at the first order without n-grams.
            max_ngram_order: The maximum n-gram order.

        Returns:
            The score.
        """

    def sentence_score(self, hypothesis: str, references: Sequence[str | None]) -> BLEUScore:
        """Sentence-level BLEU (logs a recommendation unless ``effective_order``).

        Args:
            hypothesis: A single hypothesis string.
            references: Its reference strings (``None`` entries are skipped).

        Returns:
            The score.
        """

    def corpus_score(
        self,
        hypotheses: Sequence[str],
        references: Sequence[Sequence[str | None]] | None,
        n_bootstrap: int = 1,
        *,
        threads: int | None = None,
    ) -> BLEUScore:
        """Corpus-level BLEU; see :meth:`Metric.corpus_score`.

        Args:
            hypotheses: A sequence of hypothesis strings.
            references: Reference streams (``references[k][i]``), or None for the cached ones.
            n_bootstrap: Bootstrap resamples for a confidence interval (delegated if > 1).
            threads: pyroid's thread pool setting.

        Returns:
            The score.
        """

    def get_signature(self) -> BLEUSignature:
        """The signature, e.g. ``nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0``."""

class CHRF(Metric):
    """The chrF and chrF++ metrics (``sacrebleu.CHRF``).

    Args:
        char_order: Character n-gram order.
        word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
        beta: Determine the importance of recall w.r.t precision.
        lowercase: Enable case-insensitivity.
        whitespace: If ``True``, include whitespaces when extracting character n-grams.
        eps_smoothing: If ``True``, applies epsilon smoothing similar to reference chrF++.py,
            NLTK and Moses implementations. Otherwise, it takes into account effective match
            order similar to sacreBLEU < 2.0.0.
        references: Reference streams to pre-compute for scoring many systems.
    """

    CHAR_ORDER: ClassVar[int]
    WORD_ORDER: ClassVar[int]
    BETA: ClassVar[int]
    beta: int
    char_order: int
    word_order: int
    order: int
    lowercase: bool
    whitespace: bool
    eps_smoothing: bool
    def __init__(
        self,
        char_order: int = 6,
        word_order: int = 0,
        beta: int = 2,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
        references: Sequence[Sequence[str | None]] | None = None,
    ) -> None: ...
    def sentence_score(self, hypothesis: str, references: Sequence[str | None]) -> CHRFScore:
        """Sentence-level chrF against the best of the references.

        Args:
            hypothesis: A single hypothesis string.
            references: Its reference strings (``None`` entries are skipped).

        Returns:
            The score.
        """

    def corpus_score(
        self,
        hypotheses: Sequence[str],
        references: Sequence[Sequence[str | None]] | None,
        n_bootstrap: int = 1,
        *,
        threads: int | None = None,
    ) -> CHRFScore:
        """Corpus-level chrF; see :meth:`Metric.corpus_score`.

        Args:
            hypotheses: A sequence of hypothesis strings.
            references: Reference streams (``references[k][i]``), or None for the cached ones.
            n_bootstrap: Bootstrap resamples for a confidence interval (delegated if > 1).
            threads: pyroid's thread pool setting.

        Returns:
            The score.
        """

    def get_signature(self) -> CHRFSignature:
        """The signature, e.g. ``nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0``."""

# ---------------------------------------------------------------------------
# Functions (sacrebleu.compat)
# ---------------------------------------------------------------------------

def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str | None]],
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    force: bool = False,
    lowercase: bool = False,
    tokenize: str = "13a",
    use_effective_order: bool = False,
    *,
    threads: int | None = None,
) -> BLEUScore:
    """BLEU of a corpus against one or more reference streams, like ``sacrebleu.corpus_bleu``.

        >>> corpus_bleu(["The cat sat on the mat."], [["The cat sat on a mat."]]).score
        48.892302243490086

    Args:
        hypotheses: A sequence of hypothesis strings.
        references: Reference streams: ``references[k][i]`` is the k-th reference of
            hypothesis i (``None`` for a missing one).
        smooth_method: ``"floor"``, ``"add-k"``, ``"exp"`` or ``"none"``.
        smooth_value: The value for ``floor`` and ``add-k`` (None: the default).
        force: Ignore data that looks already tokenized.
        lowercase: Lowercase the data.
        tokenize: The tokenizer to use.
        use_effective_order: Don't take into account n-gram orders without any match.
        threads: pyroid extension: ``None`` uses the whole thread pool, ``n`` at most ``n``
            tasks at once, ``1`` the calling thread.

    Returns:
        The score.
    """

def raw_corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str | None]],
    smooth_value: float | None = 0.1,
    *,
    threads: int | None = None,
) -> BLEUScore:
    """BLEU of the raw corpus: no tokenization, ``floor`` smoothing, effective order.

    Args:
        hypotheses: A sequence of hypothesis strings.
        references: Reference streams (``references[k][i]``).
        smooth_value: The ``floor`` smoothing value.
        threads: pyroid's thread pool setting (see :func:`corpus_bleu`).

    Returns:
        The score.
    """

def sentence_bleu(
    hypothesis: str,
    references: Sequence[str | None],
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    lowercase: bool = False,
    tokenize: str = "13a",
    use_effective_order: bool = True,
) -> BLEUScore:
    """BLEU of one hypothesis against its reference(s), like ``sacrebleu.sentence_bleu``.

    Args:
        hypothesis: A single hypothesis string.
        references: Its reference strings (``None`` entries are skipped).
        smooth_method: ``"floor"``, ``"add-k"``, ``"exp"`` or ``"none"``.
        smooth_value: The value for ``floor`` and ``add-k`` (None: the default).
        lowercase: Lowercase the data.
        tokenize: The tokenizer to use.
        use_effective_order: Don't take into account n-gram orders without any match.

    Returns:
        The score.
    """

@overload
def sentence_bleu_batch(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str | None] | ArrowStrings],
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    lowercase: bool = False,
    tokenize: str = "13a",
    use_effective_order: bool = True,
    *,
    threads: int | None = None,
) -> list[BLEUScore]:
    """:func:`sentence_bleu` of every hypothesis at once, on pyroid's thread pool.

    Element ``i`` equals ``sentence_bleu(hypotheses[i], [r[i] for r in references], ...)``.

    Args:
        hypotheses: The hypothesis strings: a list or tuple of str, or an Arrow string
            array/column (pyarrow, polars, ``pyroid.Column``, ...). Not a single str: use
            :func:`sentence_bleu` for one hypothesis.
        references: Reference streams, as for :func:`corpus_bleu`: ``references[k][i]`` is
            the k-th reference of hypothesis i (``None`` for a missing one). Every stream
            must have one entry per hypothesis. A stream may be an Arrow string column, whose
            nulls are missing references.
        smooth_method: ``"floor"``, ``"add-k"``, ``"exp"`` or ``"none"``.
        smooth_value: The value for ``floor`` and ``add-k`` (None: the default).
        lowercase: Lowercase the data.
        tokenize: The tokenizer to use.
        use_effective_order: Don't take into account n-gram orders without any match.
        threads: ``None`` uses the whole thread pool, ``n`` at most ``n`` tasks at once,
            ``1`` the calling thread.

    Returns:
        A list (for any input kind) with one score per hypothesis, and ``None`` for a null
        in an Arrow ``hypotheses`` column.

    Raises:
        pyroid.InputError: ``hypotheses`` is not a sequence of str or an Arrow string column,
            or a list or tuple of hypotheses holds ``None`` (the message gives its index), or
            ``references`` is not a sequence of streams (e.g. a flat list of str: pass
            ``[refs]``).
        ValueError: No streams, or a stream's length differs from the hypotheses'.
    """

@overload
def sentence_bleu_batch(
    hypotheses: ArrowStrings,
    references: Sequence[Sequence[str | None] | ArrowStrings],
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    lowercase: bool = False,
    tokenize: str = "13a",
    use_effective_order: bool = True,
    *,
    threads: int | None = None,
) -> list[BLEUScore | None]: ...
def corpus_chrf(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str | None]],
    char_order: int = 6,
    word_order: int = 0,
    beta: int = 2,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
    *,
    threads: int | None = None,
) -> CHRFScore:
    """Corpus-level chrF (``word_order=2``: chrF++), like ``sacrebleu.corpus_chrf``.

    Args:
        hypotheses: A sequence of hypothesis strings.
        references: Reference streams (``references[k][i]``; ``None`` for a missing one).
        char_order: Character n-gram order.
        word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
        beta: Determine the importance of recall w.r.t precision.
        remove_whitespace: If ``True``, removes whitespaces prior to character n-gram
            extraction.
        eps_smoothing: If ``True``, applies epsilon smoothing similar to reference chrF++.py,
            NLTK and Moses implementations. Otherwise, it takes into account effective match
            order similar to sacreBLEU < 2.0.0.
        threads: pyroid's thread pool setting (see :func:`corpus_bleu`).

    Returns:
        The score.
    """

def sentence_chrf(
    hypothesis: str,
    references: Sequence[str | None],
    char_order: int = 6,
    word_order: int = 0,
    beta: int = 2,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> CHRFScore:
    """Sentence-level chrF against the best reference, like ``sacrebleu.sentence_chrf``.

    Args:
        hypothesis: A single hypothesis string.
        references: Its reference strings (``None`` entries are skipped).
        char_order: Character n-gram order.
        word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
        beta: Determine the importance of recall w.r.t precision.
        remove_whitespace: If ``True``, removes whitespaces prior to character n-gram
            extraction.
        eps_smoothing: Epsilon smoothing instead of effective-order smoothing.

    Returns:
        The score.
    """

@overload
def sentence_chrf_batch(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str | None] | ArrowStrings],
    char_order: int = 6,
    word_order: int = 0,
    beta: int = 2,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
    *,
    threads: int | None = None,
) -> list[CHRFScore]:
    """:func:`sentence_chrf` of every hypothesis at once, on pyroid's thread pool.

    Element ``i`` equals ``sentence_chrf(hypotheses[i], [r[i] for r in references], ...)``.

    Args:
        hypotheses: The hypothesis strings, as for :func:`sentence_bleu_batch`.
        references: Reference streams (see :func:`sentence_bleu_batch`).
        char_order: Character n-gram order.
        word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
        beta: Determine the importance of recall w.r.t precision.
        remove_whitespace: If ``True``, removes whitespaces prior to character n-gram
            extraction.
        eps_smoothing: Epsilon smoothing instead of effective-order smoothing.
        threads: ``None`` uses the whole thread pool, ``n`` at most ``n`` tasks at once,
            ``1`` the calling thread.

    Returns:
        A list (for any input kind) with one score per hypothesis, and ``None`` for a null
        in an Arrow ``hypotheses`` column.

    Raises:
        pyroid.InputError: As for :func:`sentence_bleu_batch`.
        ValueError: No streams, or a stream's length differs from the hypotheses'.
    """

@overload
def sentence_chrf_batch(
    hypotheses: ArrowStrings,
    references: Sequence[Sequence[str | None] | ArrowStrings],
    char_order: int = 6,
    word_order: int = 0,
    beta: int = 2,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
    *,
    threads: int | None = None,
) -> list[CHRFScore | None]: ...
def extract_word_ngrams(tokens: list[str], n: int) -> Counter[str]:
    """Counts of the word n-grams of order ``n`` of ``tokens`` (joined with spaces).

    Args:
        tokens: A list of tokens.
        n: The order of n-grams.

    Returns:
        The n-gram counts.
    """

def extract_char_ngrams(line: str, n: int, include_whitespace: bool = False) -> Counter[str]:
    """Counts of the character n-grams of order ``n`` of ``line``.

    Args:
        line: A segment containing a sequence of words.
        n: The order of the n-grams.
        include_whitespace: If given, will not strip whitespaces from the line.

    Returns:
        The n-gram counts.
    """

def smart_open(file: str, mode: str = "rt", encoding: str = "utf-8") -> IO[Any]:
    """Open a plain or gzip-compressed (``.gz``) text file, like ``sacrebleu.smart_open``.

    Args:
        file: The file to open.
        mode: The file mode.
        encoding: The text encoding.

    Returns:
        The open file.
    """

# ---------------------------------------------------------------------------
# Delegated to the real sacrebleu
# ---------------------------------------------------------------------------

class TER:
    """sacrebleu's TER metric (delegated: every call runs a real ``sacrebleu.TER``).

    Args:
        normalized: Enable character normalization.
        no_punct: Remove punctuation.
        asian_support: Enable special treatment of Asian characters.
        case_sensitive: Enables case-sensitivity.
        references: Reference streams to pre-compute.

    Raises:
        pyroid.PyroidError: sacrebleu is not installed.
    """

    def __init__(
        self,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
        references: Sequence[Sequence[str]] | None = None,
    ) -> None: ...
    def corpus_score(
        self,
        hypotheses: Sequence[str],
        references: Sequence[Sequence[str]] | None,
        n_bootstrap: int = 1,
    ) -> Any:
        """``sacrebleu.TER.corpus_score`` (returns sacrebleu's ``TERScore``).

        Args:
            hypotheses: A sequence of hypothesis strings.
            references: Reference streams, or None for the cached ones.
            n_bootstrap: Bootstrap resamples for a confidence interval.
        """

    def sentence_score(self, hypothesis: str, references: Sequence[str]) -> Any:
        """``sacrebleu.TER.sentence_score`` (returns sacrebleu's ``TERScore``).

        Args:
            hypothesis: A single hypothesis string.
            references: Its reference strings.
        """

    def get_signature(self) -> Any:
        """``sacrebleu.TER.get_signature``."""

    def __getattr__(self, name: str) -> Any: ...

def corpus_ter(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    normalized: bool = False,
    no_punct: bool = False,
    asian_support: bool = False,
    case_sensitive: bool = False,
) -> Any:
    """``sacrebleu.corpus_ter`` (delegated; returns sacrebleu's ``TERScore``).

    Args:
        hypotheses: A sequence of hypothesis strings.
        references: Reference streams.
        normalized: Enable character normalization.
        no_punct: Remove punctuation.
        asian_support: Enable special treatment of Asian characters.
        case_sensitive: Enables case-sensitivity.
    """

def sentence_ter(
    hypothesis: str,
    references: Sequence[str],
    normalized: bool = False,
    no_punct: bool = False,
    asian_support: bool = False,
    case_sensitive: bool = False,
) -> Any:
    """``sacrebleu.sentence_ter`` (delegated; returns sacrebleu's ``TERScore``).

    Args:
        hypothesis: A single hypothesis string.
        references: Its reference strings.
        normalized: Enable character normalization.
        no_punct: Remove punctuation.
        asian_support: Enable special treatment of Asian characters.
        case_sensitive: Enables case-sensitivity.
    """

def download_test_set(test_set: str, langpair: str | None = None) -> Any:
    """``sacrebleu.download_test_set`` (delegated).

    Args:
        test_set: The test set to download.
        langpair: The language pair (all of them if None).
    """

def get_source_file(test_set: str, langpair: str) -> str:
    """``sacrebleu.get_source_file`` (delegated).

    Args:
        test_set: The test set.
        langpair: The language pair.
    """

def get_reference_files(test_set: str, langpair: str) -> list[str]:
    """``sacrebleu.get_reference_files`` (delegated).

    Args:
        test_set: The test set.
        langpair: The language pair.
    """

def get_available_testsets() -> list[str]:
    """``sacrebleu.get_available_testsets`` (delegated)."""

def get_langpairs_for_testset(testset: str) -> list[str]:
    """``sacrebleu.get_langpairs_for_testset`` (delegated).

    Args:
        testset: The test set.
    """
