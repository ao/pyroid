# Derived from sacrebleu 2.6.0 (Apache-2.0), © Amazon.com, Inc. or its affiliates; modified.
"""A drop-in, Rust-backed replacement for sacrebleu 2.6.0's BLEU and chrF API.

``import pyroid.metrics.sacrebleu as sacrebleu``: same functions, classes, scores (bit for
bit), ``str()``/signature strings, exceptions and ``sacrebleu`` log warnings. The integer
n-gram statistics are computed in Rust; the score formulas, result objects and argument
checks are sacrebleu's own Python code, so scores are exactly sacrebleu's on the same
interpreter. sacrebleu itself is only needed for the features in ``DELEGATED`` (and the
``ja-mecab``, ``ko-mecab``, ``spm``, ``flores101``, ``flores200`` and ``spBLEU-1K``
tokenizers, and ``n_bootstrap > 1``), which run the real sacrebleu with a one-time
:class:`pyroid.FallbackWarning`, or raise :class:`pyroid.PyroidError` if it is not
installed. Importing this module never imports sacrebleu. See ``sacrebleu.pyi``.
"""

from __future__ import annotations

import functools
import gzip
import json
import logging
import math
import operator
import os
import statistics
import sys
import threading
import warnings
from abc import ABCMeta, abstractmethod
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from importlib import import_module
from typing import Any, ClassVar, NoReturn

from pyroid import _versions
from pyroid._pyroid import FallbackWarning, InputError, PyroidError
from pyroid._pyroid import metrics as _native
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

#: The sacrebleu release whose API and results this module reproduces (also the
#: ``version:`` field of signatures).
__version__ = "2.6.0"

_versions.check(__name__, "sacrebleu", f"sacrebleu=={__version__}")

#: Names that are forwarded to the real sacrebleu (it must be installed).
DELEGATED = (
    "DATASETS",
    "TER",
    "corpus_ter",
    "download_test_set",
    "get_available_testsets",
    "get_langpairs_for_testset",
    "get_reference_files",
    "get_source_file",
    "sentence_ter",
)

sacrelogger = logging.getLogger("sacrebleu")

# Where sacrebleu stores downloaded test sets (sacrebleu.utils).
USERHOME = os.path.expanduser("~")
SACREBLEU_DIR = os.environ.get("SACREBLEU", os.path.join(USERHOME, ".sacrebleu"))

# The default for the maximum n-gram order when computing precisions
MAX_NGRAM_ORDER = 4

# ---------------------------------------------------------------------------
# Delegation to the real sacrebleu
# ---------------------------------------------------------------------------

_WARNED: set[str] = set()
_WARNED_LOCK = threading.Lock()


def _caller_stacklevel() -> int:
    """The ``stacklevel`` that points our caller's ``warnings.warn`` at the user's line.

    That is the first frame outside this module, however many of our functions are in between.
    """
    frame = sys._getframe(1)  # the function that calls warnings.warn: stacklevel 1
    level = 1
    while frame.f_back is not None and frame.f_globals.get("__name__") == __name__:
        frame = frame.f_back
        level += 1
    return level


def _first_time(key: str) -> bool:
    with _WARNED_LOCK:
        if key in _WARNED:
            return False
        _WARNED.add(key)
        return True


def _sacrebleu(name: str, reason: str) -> Any:
    """Import the real sacrebleu for ``name``, warning once per reason, or raise PyroidError."""
    try:
        import sacrebleu as real
    except ImportError:
        raise PyroidError(
            f"pyroid.metrics.sacrebleu.{name}: {reason} is not supported by pyroid's Rust "
            "port, and sacrebleu is not installed to fall back to (pip install sacrebleu==2.6.0)"
        ) from None
    if _first_time(f"{name}:{reason}"):
        warnings.warn(
            f"pyroid.metrics.sacrebleu.{name}: {reason} is not supported by the Rust port; "
            "delegating to sacrebleu (slow path). This warning is shown once.",
            FallbackWarning,
            stacklevel=_caller_stacklevel(),
        )
    return real


# ---------------------------------------------------------------------------
# sacrebleu.utils
# ---------------------------------------------------------------------------


def smart_open(file: str, mode: str = "rt", encoding: str = "utf-8") -> Any:
    """Open a plain or gzip-compressed (``.gz``) text file, as ``sacrebleu.smart_open``."""
    if file.endswith(".gz"):
        return gzip.open(file, mode=mode, encoding=encoding, newline="\n")
    return open(file, mode=mode, encoding=encoding, newline="\n")


def my_log(num: float) -> float:
    """Floors the log function: ``log(num)``, or -9999999999 for 0."""
    if num == 0.0:
        return -9999999999
    return math.log(num)


def sum_of_lists(lists: Any) -> Any:
    """Aggregates list of numeric lists by summing."""
    if len(lists) == 1:
        return lists[0]

    # Preserve datatype
    size = len(lists[0])
    init_val = type(lists[0][0])(0.0)
    total = [init_val] * size
    for ll in lists:
        for i in range(size):
            total[i] += ll[i]
    return total


# ---------------------------------------------------------------------------
# sacrebleu.metrics.helpers
# ---------------------------------------------------------------------------


def extract_all_word_ngrams(line: str, min_order: int, max_order: int) -> tuple[Counter[Any], int]:
    """All word n-grams of ``line`` (``min_order <= n <= max_order``) and its token count."""
    ngrams = []
    tokens = line.split()

    for n in range(min_order, max_order + 1):
        for i in range(0, len(tokens) - n + 1):  # noqa: PIE808
            ngrams.append(tuple(tokens[i : i + n]))  # noqa: PERF401

    return Counter(ngrams), len(tokens)


def extract_word_ngrams(tokens: list[str], n: int) -> Counter[str]:
    """Counts of the word n-grams of order ``n`` of ``tokens`` (joined with spaces).

    Args:
        tokens: A list of tokens.
        n: The order of n-grams.

    Returns:
        A Counter object with n-grams counts.
    """
    return Counter([" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def extract_char_ngrams(line: str, n: int, include_whitespace: bool = False) -> Counter[str]:
    """Counts of the character n-grams of order ``n`` of a sentence.

    Args:
        line: A segment containing a sequence of words.
        n: The order of the n-grams.
        include_whitespace: If given, will not strip whitespaces from the line.

    Returns:
        A dictionary containing ngrams and counts.
    """
    if not include_whitespace:
        line = "".join(line.split())

    return Counter([line[i : i + n] for i in range(len(line) - n + 1)])


def extract_all_char_ngrams(
    line: str, max_order: int, include_whitespace: bool = False
) -> list[Counter[str]]:
    """All character n-grams of ``line``, one Counter per order ``1..max_order``."""
    counters = []

    if not include_whitespace:
        line = "".join(line.split())

    for n in range(1, max_order + 1):
        ngrams = Counter([line[i : i + n] for i in range(len(line) - n + 1)])
        counters.append(ngrams)

    return counters


# ---------------------------------------------------------------------------
# sacrebleu.metrics.base
# ---------------------------------------------------------------------------


class Score:
    """A base score class to derive from.

    Args:
        name: The name of the underlying metric.
        score: A floating point number for the final metric.
    """

    def __init__(self, name: str, score: float):
        self.name = name
        self.score = score

        # Statistical test related fields
        self._mean = -1.0
        self._ci = -1.0

        # More info can be added right after the score
        self._verbose = ""

    def format(
        self,
        width: int = 2,
        score_only: bool = False,
        signature: str = "",
        is_json: bool = False,
    ) -> str:
        """Returns a pretty representation of the score.

        Args:
            width: Floating point decimal precision width.
            score_only: If ``True``, and the format is not ``json``, returns a single score
                string.
            signature: A string representation of the given ``Signature`` instance.
            is_json: If ``True``, will output the score in JSON string.

        Returns:
            A plain or JSON-formatted string representation.
        """
        d: dict[str, Any] = {
            "name": self.name,
            "score": float(f"{self.score:.{width}f}"),
            "signature": signature,
        }

        sc = f"{self.score:.{width}f}"

        if self._mean > 0:
            confidence_mean = f"{self._mean:.{width}f}"
            confidence_var = f"{self._ci:.{width}f}"
            confidence_str = f"μ = {confidence_mean} ± {confidence_var}"

            sc += f" ({confidence_str})"
            if is_json:
                d["confidence_mean"] = float(confidence_mean)
                d["confidence_var"] = float(confidence_var)
                d["confidence"] = confidence_str

        # Construct full score line
        full_score = f"{self.name}|{signature}" if signature else self.name
        full_score = f"{full_score} = {sc}"
        if self._verbose:
            full_score += f" {self._verbose}"
            d["verbose_score"] = self._verbose

        if score_only:
            return sc

        if is_json:
            for param in signature.split("|"):
                key, value = param.split(":")
                d[key] = value
            return json.dumps(d, indent=1, ensure_ascii=False)

        return full_score

    def estimate_ci(self, scores: list[Score]) -> None:
        """Stores the mean and the 95% confidence interval of bootstrap resampled scores.

        Args:
            scores: A list of ``Score`` objects obtained from bootstrap resampling for
                example.
        """
        # Sort the scores
        raw_scores = sorted([x.score for x in scores])
        n = len(raw_scores)

        # Get CI bounds (95%, i.e. 1/40 from left)
        lower_idx = n // 40
        upper_idx = n - lower_idx - 1
        lower, upper = raw_scores[lower_idx], raw_scores[upper_idx]
        self._ci = 0.5 * (upper - lower)
        self._mean = statistics.mean(raw_scores)

    def __repr__(self) -> str:
        """Returns a human readable score string."""
        return self.format()


class Signature:
    """A convenience class to represent sacreBLEU reproducibility signatures.

    Args:
        args: key-value dictionary passed from the actual metric instance.
    """

    def __init__(self, args: dict[str, Any]):
        # Global items that are shared across all metrics
        self._abbr = {
            "version": "v",
            "nrefs": "#",
            "test": "t",
            "lang": "l",
            "subset": "S",
            "origlang": "o",
            "bs": "bs",  # Bootstrap resampling trials
            "ar": "ar",  # Approximate randomization trials
            "seed": "rs",  # RNG's seed
        }

        if "num_refs" not in args:
            raise ValueError("Number of references unknown, please evaluate the metric first.")

        num_refs = args["num_refs"]
        if num_refs == -1:
            # Detect variable number of refs
            num_refs = "var"

        # Global items that are shared across all metrics
        # None's will be ignored
        self.info = {
            "version": __version__,
            "nrefs": num_refs,
            "bs": args.get("n_bootstrap"),
            "ar": None,
            "seed": args.get("seed"),
            "test": args.get("test_set"),
            "lang": args.get("langpair"),
            "origlang": args.get("origlang"),
            "subset": args.get("subset"),
        }

    def format(self, short: bool = False) -> str:
        """Returns a string representation of the signature.

        Args:
            short: If True, shortened signature is produced.

        Returns:
            A string representation of the signature.
        """
        pairs = []
        keys = list(self.info.keys())
        # keep version always at end
        keys.remove("version")
        for name in [*keys, "version"]:
            value = self.info[name]
            if value is not None:
                if isinstance(value, bool):
                    # Replace True/False with yes/no
                    value = "yes" if value else "no"
                final_name = self._abbr[name] if short else name
                pairs.append(f"{final_name}:{value}")

        return "|".join(pairs)

    def update(self, key: str, value: Any) -> None:
        """Add a new item or update an existing one.

        Args:
            key: The key to use in the dictionary.
            value: The associated value for the ``key``.
        """
        self.info[key] = value

    def __str__(self) -> str:
        """Returns a human-readable signature string."""
        return self.format()

    def __repr__(self) -> str:
        """Returns a human-readable signature string."""
        return self.format()


def _seq(hyps: Any) -> Any:
    """``hyps`` as a list or tuple (what ``zip`` would iterate)."""
    return hyps if type(hyps) in (list, tuple) else list(hyps)


def _streams(references: Any) -> list[Any]:
    """Reference streams as lists or tuples (``zip(*references)`` iterates each one)."""
    return [s if type(s) in (list, tuple) else list(s) for s in references]


def _n_ref_segments(streams: list[Any]) -> int:
    return min((len(s) for s in streams), default=0)


def _map_streams(streams: list[Any], n: int, f: Callable[[str], str]) -> list[list[Any]]:
    """``f`` over the first ``n`` references of every stream (``None`` stays ``None``)."""
    return [[None if x is None else f(x) for x in s[:n]] for s in streams]


def _raise_without_refs() -> NoReturn:
    """Raise the TypeError sacrebleu raises for a segment without references.

    (``hyp_ngram in ref_ngrams`` with ``ref_ngrams = None``; the message is the interpreter's.)
    """
    ref_ngrams: Any = None
    operator.contains(ref_ngrams, ())
    raise AssertionError("unreachable")  # pragma: no cover


def _warn_tokenized(tok_count: int) -> None:
    if tok_count >= 100:
        sacrelogger.warning("That's 100 lines that end in a tokenized period ('.')")
        sacrelogger.warning(
            "It looks like you forgot to detokenize your test data, which may hurt your score."
        )
        sacrelogger.warning(
            "If you insist your data is detokenized, or don't care, you can suppress this "
            "message with the `force` parameter."
        )


class Metric(metaclass=ABCMeta):
    """A base class for all metrics, as ``sacrebleu.metrics.base.Metric``."""

    # Each metric should define its Signature class' name here
    _SIGNATURE_TYPE: ClassVar[type[Signature]] = Signature

    def __init__(self) -> None:
        # The pre-computed reference cache
        self._ref_cache: Any = None

        # only useful for BLEU tokenized warnings. Set to True so that
        # warnings are not issued for other metrics.
        self._force = True

        # Will be used by the signature when bootstrap resampling
        self.n_bootstrap: int | None = None
        self.seed: str | None = None

    def _check_sentence_score_args(self, hyp: str, refs: Sequence[str]) -> None:
        """Performs sanity checks on ``sentence_score`` method's arguments."""
        prefix = self.__class__.__name__
        err_msg = None

        if not isinstance(hyp, str):
            err_msg = "The argument `hyp` should be a string."
        elif isinstance(refs, str) or not isinstance(refs, Sequence):
            err_msg = "The argument `refs` should be a sequence of strings."
        elif not isinstance(refs[0], str) and refs[0] is not None:
            err_msg = "Each element of `refs` should be a string."

        if err_msg:
            raise TypeError(f"{prefix}: {err_msg}")

    def _check_corpus_score_args(
        self, hyps: Sequence[str], refs: Sequence[Sequence[str]] | None
    ) -> None:
        """Performs sanity checks on ``corpus_score`` method's arguments."""
        prefix = self.__class__.__name__
        err_msg = None

        if not isinstance(hyps, Sequence):
            err_msg = "`hyps` should be a sequence of strings."
        elif not isinstance(hyps[0], str):
            err_msg = "Each element of `hyps` should be a string."
        elif any(line is None for line in hyps):
            err_msg = "Undefined line in hypotheses stream!"

        if refs is not None:
            if not isinstance(refs, Sequence):
                err_msg = "`refs` should be a sequence of sequence of strings."
            elif not isinstance(refs[0], Sequence):
                err_msg = "Each element of `refs` should be a sequence of strings."
            elif not isinstance(refs[0][0], str) and refs[0][0] is not None:
                err_msg = "`refs` should be a sequence of sequence of strings."

        if err_msg:
            raise TypeError(f"{prefix}: {err_msg}")

    @abstractmethod
    def _aggregate_and_compute(self, stats: list[list[Any]]) -> Any:
        """Computes the final score given the pre-computed match statistics."""

    @abstractmethod
    def _compute_score_from_stats(self, stats: list[Any]) -> Any:
        """Computes the final score from already aggregated statistics."""

    @abstractmethod
    def _preprocess_segment(self, sent: str) -> str:
        """A wrapper around the metric's tokenization and pre-processing logic."""

    @abstractmethod
    def _extract_reference_info(self, refs: Sequence[str]) -> dict[str, Any]:
        """Extract the information (n-grams) of a list of reference segments."""

    @abstractmethod
    def _compute_segment_statistics(self, hypothesis: str, ref_kwargs: dict[str, Any]) -> list[Any]:
        """The best match statistics of a (pre-processed) hypothesis across its references."""

    @abstractmethod
    def _statistics(
        self, hypotheses: Any, references: Any, want_rows: bool, threads: int | None
    ) -> tuple[list[Any] | None, list[list[Any]] | None]:
        """``(summed statistics, per-segment statistics)``; either may be None, not both."""

    def _py_cache_references(self, references: Sequence[Sequence[str]]) -> list[Any]:
        """``Metric._cache_references`` of sacrebleu (the pure-Python path)."""
        ref_cache = []

        # Decide on final number of refs here as well
        num_refs = set()

        for refs in zip(*references, strict=False):
            # Remove undefined references
            lines = [x for x in refs if x is not None]

            # Keep track of reference counts to allow variable reference
            # info in the signature
            num_refs.add(len(lines))

            lines = [self._preprocess_segment(x) for x in lines]

            # Get n-grams
            ref_cache.append(self._extract_reference_info(lines))

        if len(num_refs) == 1:
            self.num_refs = next(iter(num_refs))
        else:
            # A variable number of refs exist
            self.num_refs = -1

        return ref_cache

    def _py_extract_corpus_statistics(
        self, hypotheses: Sequence[str], references: Sequence[Sequence[str]] | None
    ) -> list[list[Any]]:
        """``Metric._extract_corpus_statistics`` of sacrebleu (the pure-Python path)."""
        # Pre-compute references
        # Don't store the cache as the user is explicitly passing refs
        if references:
            ref_cache = self._py_cache_references(references)
        elif self._ref_cache:
            ref_cache = self._ref_cache
            if not isinstance(ref_cache, list):
                raise ValueError(
                    "pyroid.metrics.sacrebleu: the metric's settings changed after its "
                    "references were cached; create a new metric object"
                )
        else:
            raise RuntimeError("No references provided and the cache is empty.")

        stats = []
        tok_count = 0

        for hyp, ref_kwargs in zip(hypotheses, ref_cache, strict=False):
            # Check for already-tokenized input problem (only for BLEU)
            if not self._force and hyp.endswith(" ."):
                tok_count += 1

            hyp = self._preprocess_segment(hyp)

            # Collect stats
            stats.append(self._compute_segment_statistics(hyp, ref_kwargs))

        _warn_tokenized(tok_count)
        return stats

    def _extract_corpus_statistics(
        self, hypotheses: Sequence[str], references: Sequence[Sequence[str]] | None
    ) -> list[list[Any]]:
        """Reads the corpus and returns sentence-level match statistics."""
        rows = self._statistics(hypotheses, references, True, None)[1]
        assert rows is not None
        return rows

    def sentence_score(self, hypothesis: str, references: Sequence[str]) -> Any:
        """Compute the metric for a single sentence against a single (or multiple) reference(s).

        Args:
            hypothesis: A single hypothesis string.
            references: A sequence of reference strings.

        Returns:
            A ``Score`` object.
        """
        self._check_sentence_score_args(hypothesis, references)

        totals, rows = self._statistics([hypothesis], [[refs] for refs in references], False, 1)
        return self._compute_score_from_stats(sum_of_lists(rows) if totals is None else totals)

    def corpus_score(
        self,
        hypotheses: Sequence[str],
        references: Sequence[Sequence[str]] | None,
        n_bootstrap: int = 1,
        *,
        threads: int | None = None,
    ) -> Any:
        """Compute the metric for a corpus against a single (or multiple) reference(s).

        Args:
            hypotheses: A sequence of hypothesis strings.
            references: A sequence of reference documents with document being defined as a
                sequence of reference strings. If ``None``, cached references will be used.
            n_bootstrap: If > 1, provides 95% confidence interval around true mean using
                bootstrap resampling with ``n_bootstrap`` samples (delegated to sacrebleu).
            threads: pyroid's thread pool setting (``None``: the whole pool; ``1``: the
                calling thread).

        Returns:
            A ``Score`` object.
        """
        self._check_corpus_score_args(hypotheses, references)

        # Collect corpus stats
        totals, stats = self._statistics(hypotheses, references, n_bootstrap > 1, threads)

        # Compute the actual system score
        actual_score = self._compute_score_from_stats(
            sum_of_lists(stats) if totals is None else totals
        )

        if n_bootstrap > 1:
            # Compute bootstrap estimate as well
            api = f"{self.__class__.__name__}.corpus_score"
            _sacrebleu(api, "n_bootstrap > 1")
            from sacrebleu.significance import _bootstrap_resample

            self.n_bootstrap = n_bootstrap
            self.seed, bs_scores = _bootstrap_resample(stats, self, n_bootstrap)
            actual_score.estimate_ci(bs_scores)

        return actual_score

    def get_signature(self) -> Signature:
        """Creates and returns the signature for the metric.

        The creation of signatures is delayed as the number of references is resolved only at
        the point of reference caching.
        """
        return self._SIGNATURE_TYPE(self.__dict__)


# ---------------------------------------------------------------------------
# sacrebleu.metrics.bleu
# ---------------------------------------------------------------------------

_TOKENIZERS = {
    "none": "tokenizer_none.NoneTokenizer",
    "zh": "tokenizer_zh.TokenizerZh",
    "13a": "tokenizer_13a.Tokenizer13a",
    "intl": "tokenizer_intl.TokenizerV14International",
    "char": "tokenizer_char.TokenizerChar",
    "ja-mecab": "tokenizer_ja_mecab.TokenizerJaMecab",
    "ko-mecab": "tokenizer_ko_mecab.TokenizerKoMecab",
    "spm": "tokenizer_spm.TokenizerSPM",
    "flores101": "tokenizer_spm.Flores101Tokenizer",
    "flores200": "tokenizer_spm.Flores200Tokenizer",
    ### Added for spBLEU-1K tokenizer by AbdelRahim Elmadany
    "spBLEU-1K": "tokenizer_spm.spBLEU1KTokenizer",
}

_NATIVE_TOKENIZERS = frozenset({"none", "zh", "13a", "intl", "char"})


class _NativeTokenizer:
    """One of sacrebleu's tokenizers (13a, intl, zh, char, none), run by pyroid's Rust port."""

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        self._name = name

    def signature(self) -> str:
        """The tokenizer's signature (its name)."""
        return self._name

    def __call__(self, line: str) -> str:
        """Tokenizes an input line with the tokenizer."""
        if self._name == "none":
            return line
        return _native.sb_tokenize(self._name, line)  # type: ignore[no-any-return]

    def __repr__(self) -> str:
        return f"<pyroid.metrics.sacrebleu tokenizer {self._name!r}>"


def _get_tokenizer(name: str, api: str) -> Callable[[], Any]:
    """The tokenizer class (a factory) for ``name``; non-native ones come from sacrebleu."""
    module_name, class_name = _TOKENIZERS[name].rsplit(".", 1)
    if name in _NATIVE_TOKENIZERS:
        return functools.partial(_NativeTokenizer, name)
    _sacrebleu(api, f"tokenize={name!r}")
    return getattr(import_module(f".tokenizers.{module_name}", "sacrebleu"), class_name)  # type: ignore[no-any-return]


def _native_tokenizer(tokenizer: Any) -> str | None:
    return tokenizer._name if type(tokenizer) is _NativeTokenizer else None


class BLEUSignature(Signature):
    """A convenience class to represent the reproducibility signature for BLEU.

    Args:
        args: key-value dictionary passed from the actual metric instance.
    """

    def __init__(self, args: dict[str, Any]):
        super().__init__(args)

        self._abbr.update(
            {
                "case": "c",
                "eff": "e",
                "tok": "tok",
                "smooth": "s",
            }
        )

        # Construct a combined string for smoothing method and value
        smooth_str = args["smooth_method"]
        smooth_def = BLEU.SMOOTH_DEFAULTS[smooth_str]

        # If the method requires a parameter, add it within brackets
        if smooth_def is not None:
            # the following can be None if the user wants to use the default
            smooth_val = args["smooth_value"]

            if smooth_val is None:
                smooth_val = smooth_def

            smooth_str += f"[{smooth_val:.2f}]"

        self.info.update(
            {
                "case": "lc" if args["lowercase"] else "mixed",
                "eff": "yes" if args["effective_order"] else "no",
                "tok": args["tokenizer_signature"],
                "smooth": smooth_str,
            }
        )


class BLEUScore(Score):
    """A convenience class to represent BLEU scores.

    Args:
        score: The BLEU score.
        counts: List of counts of correct ngrams, 1 <= n <= max_ngram_order
        totals: List of counts of total ngrams, 1 <= n <= max_ngram_order
        precisions: List of precisions, 1 <= n <= max_ngram_order
        bp: The brevity penalty.
        sys_len: The cumulative system length.
        ref_len: The cumulative reference length.
    """

    def __init__(
        self,
        score: float,
        counts: list[int],
        totals: list[int],
        precisions: list[float],
        bp: float,
        sys_len: int,
        ref_len: int,
    ):
        super().__init__("BLEU", score)
        self.bp = bp
        self.counts = counts
        self.totals = totals
        self.sys_len = sys_len
        self.ref_len = ref_len
        self.precisions = precisions

        self.prec_str = "/".join([f"{p:.1f}" for p in self.precisions])
        self.ratio = self.sys_len / self.ref_len if self.ref_len else 0

        # The verbose part of BLEU
        self._verbose = f"{self.prec_str} (BP = {self.bp:.3f} "
        self._verbose += f"ratio = {self.ratio:.3f} hyp_len = {self.sys_len:d} "
        self._verbose += f"ref_len = {self.ref_len:d})"


def _native_order(n: Any) -> bool:
    """Can the Rust port count n-grams up to this order (else: the pure-Python path)?"""
    return type(n) is int and 1 <= n <= 256


class BLEU(Metric):
    """Computes the BLEU metric given hypotheses and references (``sacrebleu.BLEU``).

    Args:
        lowercase: If True, lowercased BLEU is computed.
        force: Ignore data that looks already tokenized.
        tokenize: The tokenizer to use. If None, defaults to language-specific tokenizers
            with '13a' as the fallback default.
        smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none').
        smooth_value: The smoothing value for ``floor`` and ``add-k`` methods. ``None``
            falls back to default value.
        max_ngram_order: If given, it overrides the maximum n-gram order (default: 4) when
            computing precisions.
        effective_order: If ``True``, stop including n-gram orders for which precision is 0.
            This should be ``True``, if sentence-level BLEU will be computed.
        trg_lang: An optional language code to raise potential tokenizer warnings.
        references: A sequence of reference documents with document being defined as a
            sequence of reference strings. If given, the reference n-grams and lengths will
            be pre-computed and cached for faster BLEU computation across many systems.
    """

    SMOOTH_DEFAULTS: ClassVar[dict[str, float | None]] = {
        # The defaults for `floor` and `add-k` are obtained from the following paper
        # A Systematic Comparison of Smoothing Techniques for Sentence-Level BLEU
        # Boxing Chen and Colin Cherry
        # http://aclweb.org/anthology/W14-3346
        "none": None,  # No value is required
        "floor": 0.1,
        "add-k": 1,
        "exp": None,  # No value is required
    }

    TOKENIZERS = _TOKENIZERS.keys()

    # mteval-v13a.pl tokenizer unless Chinese or Japanese is provided
    TOKENIZER_DEFAULT = "13a"

    # Some language specific mappings to use if `trg_lang` is given
    # and the tokenizer is not explicitly specified
    _TOKENIZER_MAP: ClassVar[dict[str, str]] = {
        "zh": "zh",
        "ja": "ja-mecab",
        "ko": "ko-mecab",
    }

    _SIGNATURE_TYPE = BLEUSignature

    def __init__(
        self,
        lowercase: bool = False,
        force: bool = False,
        tokenize: str | None = None,
        smooth_method: str = "exp",
        smooth_value: float | None = None,
        max_ngram_order: int = MAX_NGRAM_ORDER,
        effective_order: bool = False,
        trg_lang: str = "",
        references: Sequence[Sequence[str]] | None = None,
    ):
        self._init(
            "BLEU",
            lowercase,
            force,
            tokenize,
            smooth_method,
            smooth_value,
            max_ngram_order,
            effective_order,
            trg_lang,
            references,
        )

    def _init(
        self,
        api: str,
        lowercase: bool = False,
        force: bool = False,
        tokenize: str | None = None,
        smooth_method: str = "exp",
        smooth_value: float | None = None,
        max_ngram_order: int = MAX_NGRAM_ORDER,
        effective_order: bool = False,
        trg_lang: str = "",
        references: Sequence[Sequence[str]] | None = None,
    ) -> None:
        """``BLEU.__init__``; ``api`` names the public entry point in fallback warnings."""
        super().__init__()

        self._force = force
        self.trg_lang = trg_lang
        self.lowercase = lowercase
        self.smooth_value = smooth_value
        self.smooth_method = smooth_method
        self.max_ngram_order = max_ngram_order
        self.effective_order = effective_order

        # Sanity check
        assert self.smooth_method in self.SMOOTH_DEFAULTS, (
            "Unknown smooth_method {self.smooth_method!r}"
        )

        # If the tokenizer wasn't specified, choose it according to the
        # following logic. We use 'v13a' except for ZH and JA. Note that
        # this logic can only be applied when sacrebleu knows the target
        # language, which is only the case for builtin datasets.
        if tokenize is None:
            best_tokenizer = self.TOKENIZER_DEFAULT

            # Set `zh` or `ja-mecab` or `ko-mecab` if target language is provided
            if self.trg_lang in self._TOKENIZER_MAP:
                best_tokenizer = self._TOKENIZER_MAP[self.trg_lang]
        else:
            best_tokenizer = tokenize
            if self.trg_lang == "zh" and best_tokenizer != "zh":
                sacrelogger.warning("Consider using the 'zh' or 'spm' tokenizer for Chinese.")
            if self.trg_lang == "ja" and best_tokenizer != "ja-mecab":
                sacrelogger.warning(
                    "Consider using the 'ja-mecab' or 'spm' tokenizer for Japanese."
                )
            if self.trg_lang == "ko" and best_tokenizer != "ko-mecab":
                sacrelogger.warning("Consider using the 'ko-mecab' or 'spm' tokenizer for Korean.")

        # Create the tokenizer
        self.tokenizer = _get_tokenizer(best_tokenizer, api)()

        # Build the signature
        self.tokenizer_signature = self.tokenizer.signature()

        if references is not None:
            # Pre-compute reference ngrams and lengths
            self._ref_cache = self._cache_references(references)

    @staticmethod
    def compute_bleu(
        correct: list[int],
        total: list[int],
        sys_len: int,
        ref_len: int,
        smooth_method: str = "none",
        smooth_value: float | None = None,
        effective_order: bool = False,
        max_ngram_order: int = MAX_NGRAM_ORDER,
    ) -> BLEUScore:
        """Computes BLEU score from its sufficient statistics with smoothing.

        Smoothing methods (citing "A Systematic Comparison of Smoothing Techniques for
        Sentence-Level BLEU", Boxing Chen and Colin Cherry, WMT 2014:
        http://aclweb.org/anthology/W14-3346)

        - none: No smoothing.
        - floor: Method 1 (requires small positive value (0.1 in the paper) to be set)
        - add-k: Method 2 (Generalizing Lin and Och, 2004)
        - exp: Method 3 (NIST smoothing method i.e. in use with mteval-v13a.pl)

        Args:
            correct: List of counts of correct ngrams, 1 <= n <= max_ngram_order
            total: List of counts of total ngrams, 1 <= n <= max_ngram_order
            sys_len: The cumulative system length
            ref_len: The cumulative reference length
            smooth_method: The smoothing method to use ('floor', 'add-k', 'exp' or 'none')
            smooth_value: The smoothing value for ``floor`` and ``add-k`` methods. ``None``
                falls back to default value.
            effective_order: If ``True``, stop including n-gram orders for which precision
                is 0. This should be ``True``, if sentence-level BLEU will be computed.
            max_ngram_order: If given, it overrides the maximum n-gram order (default: 4)
                when computing precisions.

        Returns:
            A ``BLEUScore`` instance.
        """
        assert smooth_method in BLEU.SMOOTH_DEFAULTS, "Unknown smooth_method {smooth_method!r}"

        # Fetch the default value for floor and add-k
        if smooth_value is None:
            smooth_value = BLEU.SMOOTH_DEFAULTS[smooth_method]

        # Compute brevity penalty
        bp = 1.0
        if sys_len < ref_len:
            bp = math.exp(1 - ref_len / sys_len) if sys_len > 0 else 0.0

        # n-gram precisions
        precisions = [0.0] * max_ngram_order

        # Early stop if there are no matches (#141)
        if not any(correct):
            return BLEUScore(0.0, correct, total, precisions, bp, sys_len, ref_len)

        smooth_mteval = 1.0
        eff_order = max_ngram_order
        for n in range(1, len(precisions) + 1):
            if smooth_method == "add-k" and n > 1:
                correct[n - 1] += smooth_value  # type: ignore[assignment,operator]
                total[n - 1] += smooth_value  # type: ignore[assignment,operator]

            if total[n - 1] == 0:
                break

            # If the system guesses no i-grams, 1 <= i <= max_ngram_order,
            # the BLEU score is 0 (technically undefined). This is a problem for sentence
            # level BLEU or a corpus of short sentences, where systems will get
            # no credit if sentence lengths fall under the max_ngram_order threshold.
            # This fix scales max_ngram_order to the observed maximum order.
            # It is only available through the API and off by default
            if effective_order:
                eff_order = n

            if correct[n - 1] == 0:
                if smooth_method == "exp":
                    smooth_mteval *= 2
                    precisions[n - 1] = 100.0 / (smooth_mteval * total[n - 1])
                elif smooth_method == "floor":
                    precisions[n - 1] = 100.0 * smooth_value / total[n - 1]  # type: ignore[operator]
            else:
                precisions[n - 1] = 100.0 * correct[n - 1] / total[n - 1]

        # Compute BLEU score
        score = bp * math.exp(sum([my_log(p) for p in precisions[:eff_order]]) / eff_order)

        return BLEUScore(score, correct, total, precisions, bp, sys_len, ref_len)

    def _preprocess_segment(self, sent: str) -> str:
        """Given a sentence, lowercases (optionally) and tokenizes it."""
        if self.lowercase:
            sent = sent.lower()
        return self.tokenizer(sent.rstrip())  # type: ignore[no-any-return]

    def _compute_score_from_stats(self, stats: list[int]) -> BLEUScore:
        """Computes the final score from already aggregated statistics."""
        return self.compute_bleu(
            correct=stats[2 : 2 + self.max_ngram_order],
            total=stats[2 + self.max_ngram_order :],
            sys_len=int(stats[0]),
            ref_len=int(stats[1]),
            smooth_method=self.smooth_method,
            smooth_value=self.smooth_value,
            effective_order=self.effective_order,
            max_ngram_order=self.max_ngram_order,
        )

    def _aggregate_and_compute(self, stats: list[list[int]]) -> BLEUScore:
        """Computes the final BLEU score given the pre-computed corpus statistics."""
        return self._compute_score_from_stats(sum_of_lists(stats))

    def _get_closest_ref_len(self, hyp_len: int, ref_lens: list[int]) -> int:
        """The reference length closest to the hypothesis length (the shorter one on ties)."""
        closest_diff, closest_len = -1, -1

        for ref_len in ref_lens:
            diff = abs(hyp_len - ref_len)
            if closest_diff == -1 or diff < closest_diff:
                closest_diff = diff
                closest_len = ref_len
            elif diff == closest_diff and ref_len < closest_len:
                closest_len = ref_len

        return closest_len

    def _extract_reference_info(self, refs: Sequence[str]) -> dict[str, Any]:
        """The merged n-gram counts and lengths of a list of reference segments."""
        ngrams: Any = None
        ref_lens = []

        for ref in refs:
            # extract n-grams for this ref
            this_ngrams, ref_len = extract_all_word_ngrams(ref, 1, self.max_ngram_order)
            ref_lens.append(ref_len)

            if ngrams is None:
                # Set it directly for first set of refs
                ngrams = this_ngrams
            else:
                # Merge counts across multiple references
                # The below loop is faster than `ngrams |= this_ngrams`
                for ngram, count in this_ngrams.items():
                    ngrams[ngram] = max(ngrams[ngram], count)

        return {"ref_ngrams": ngrams, "ref_lens": ref_lens}

    def _compute_segment_statistics(self, hypothesis: str, ref_kwargs: dict[str, Any]) -> list[int]:
        """The match statistics of a (pre-processed) hypothesis against its references."""
        ref_ngrams, ref_lens = ref_kwargs["ref_ngrams"], ref_kwargs["ref_lens"]

        # Extract n-grams for the hypothesis
        hyp_ngrams, hyp_len = extract_all_word_ngrams(hypothesis, 1, self.max_ngram_order)

        ref_len = self._get_closest_ref_len(hyp_len, ref_lens)

        # Count the stats
        # Although counter has its internal & and | operators, this is faster
        correct = [0 for i in range(self.max_ngram_order)]
        total = correct[:]
        for hyp_ngram, hyp_count in hyp_ngrams.items():
            # n-gram order
            n = len(hyp_ngram) - 1
            # count hypothesis n-grams
            total[n] += hyp_count
            # count matched n-grams
            if hyp_ngram in ref_ngrams:
                correct[n] += min(hyp_count, ref_ngrams[hyp_ngram])

        # Return a flattened list for efficient computation
        return [hyp_len, ref_len, *correct, *total]

    def _cache_references(self, references: Sequence[Sequence[str]]) -> Any:
        """Pre-compute the references: a native cache, or sacrebleu's list on the Python path."""
        if not _native_order(self.max_ngram_order):
            return self._py_cache_references(references)
        streams = _streams(references)
        name = _native_tokenizer(self.tokenizer)
        if name is None:
            streams = _map_streams(streams, _n_ref_segments(streams), self._preprocess_segment)
            name = "none"
        elif self.lowercase:
            streams = _map_streams(streams, _n_ref_segments(streams), str.lower)
        cache = _native._BleuRefCache(streams, name, self.max_ngram_order)
        self.num_refs = cache.num_refs
        return cache

    def _hyps_for_native(self, hyps: Any, n: int) -> tuple[Any, str, int | None]:
        """Hypotheses as the kernel takes them, and the tokenizer name and " ." count.

        The count is None when the kernel can count it itself.
        """
        name = _native_tokenizer(self.tokenizer)
        if name is None:
            hyps = hyps[:n]
            tok_count = 0 if self._force else sum(1 for h in hyps if h.endswith(" ."))
            return [self._preprocess_segment(h) for h in hyps], "none", tok_count
        if self.lowercase:
            return [h.lower() for h in hyps[:n]], name, None
        return hyps, name, None

    def _statistics(
        self, hypotheses: Any, references: Any, want_rows: bool, threads: int | None
    ) -> tuple[list[Any] | None, list[list[Any]] | None]:
        if not _native_order(self.max_ngram_order):
            return None, self._py_extract_corpus_statistics(hypotheses, references)
        hyps = _seq(hypotheses)
        if references:
            streams = _streams(references)
            n_ref = _n_ref_segments(streams)
            n = min(len(hyps), n_ref)
            name = _native_tokenizer(self.tokenizer)
            if name is None:
                streams = _map_streams(streams, n_ref, self._preprocess_segment)
            elif self.lowercase:
                streams = _map_streams(streams, n_ref, str.lower)
            hyps, name, tok_count = self._hyps_for_native(hyps, n)
            out = _native.bleu_corpus_stats(
                hyps,
                streams,
                name,
                self.max_ngram_order,
                tok_count is None and not self._force,
                want_rows,
                threads=threads,
            )
        elif self._ref_cache:
            cache = self._ref_cache
            if (
                not isinstance(cache, _native._BleuRefCache)
                or cache.max_order != self.max_ngram_order
            ):
                raise ValueError(
                    "pyroid.metrics.sacrebleu: max_ngram_order changed after the references "
                    "were cached; create a new BLEU object"
                )
            hyps, name, tok_count = self._hyps_for_native(hyps, min(len(hyps), len(cache)))
            out = _native.bleu_cache_stats(
                cache,
                hyps,
                name,
                tok_count is None and not self._force,
                want_rows,
                threads=threads,
            )
        else:
            raise RuntimeError("No references provided and the cache is empty.")
        totals, n_segments, num_refs, periods, type_error_at, rows = out
        if references:
            self.num_refs = num_refs
        if type_error_at is not None:
            _raise_without_refs()
        _warn_tokenized(periods if tok_count is None else tok_count)
        if n_segments == 0:
            return None, []
        return totals, rows

    def sentence_score(self, hypothesis: str, references: Sequence[str]) -> BLEUScore:
        """Compute the metric for a single sentence against a single (or multiple) reference(s).

        Args:
            hypothesis: A single hypothesis string.
            references: A sequence of reference strings.

        Returns:
            A ``BLEUScore`` object.
        """
        if not self.effective_order:
            sacrelogger.warning(
                "It is recommended to enable `effective_order` for sentence-level BLEU."
            )
        return super().sentence_score(hypothesis, references)  # type: ignore[no-any-return]

    def _sentence_score_batch(
        self, hypotheses: Any, references: Any, threads: int | None
    ) -> list[BLEUScore | None]:
        """``sentence_score(hypotheses[i], [r[i] for r in references])`` for every ``i``."""
        hyps, streams = _batch_args(hypotheses, references)
        if not self.effective_order:
            sacrelogger.warning(
                "It is recommended to enable `effective_order` for sentence-level BLEU."
            )
        n = len(hyps)
        hyps, streams, keep = _without_nulls(hyps, streams)
        if not hyps:
            return [None] * n
        if _native_order(self.max_ngram_order):
            rows = self._statistics(hyps, streams, True, threads)[1]
        else:
            rows = [
                self._py_extract_corpus_statistics([h], [[s[i]] for s in streams])[0]
                for i, h in enumerate(hyps)
            ]
        assert rows is not None
        return _with_nulls([self._compute_score_from_stats(row) for row in rows], keep, n)


# ---------------------------------------------------------------------------
# sacrebleu.metrics.chrf
# ---------------------------------------------------------------------------


class CHRFSignature(Signature):
    """A convenience class to represent the reproducibility signature for chrF.

    Args:
        args: key-value dictionary passed from the actual metric instance.
    """

    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        self._abbr.update(
            {
                "case": "c",
                "eff": "e",
                "nc": "nc",
                "nw": "nw",
                "space": "s",
            }
        )

        self.info.update(
            {
                "case": "lc" if args["lowercase"] else "mixed",
                "eff": "yes" if not args["eps_smoothing"] else "no",
                "nc": args["char_order"],
                "nw": args["word_order"],
                "space": "yes" if args["whitespace"] else "no",
            }
        )


class CHRFScore(Score):
    """A convenience class to represent chrF scores.

    Args:
        score: The chrF (chrF++) score.
        char_order: The character n-gram order.
        word_order: The word n-gram order. If equals to 2, the metric is referred to as
            chrF++.
        beta: Determine the importance of recall w.r.t precision.
    """

    def __init__(self, score: float, char_order: int, word_order: int, beta: int):
        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order

        # Add + signs to denote chrF+ variant
        name = f"chrF{self.beta}" + "+" * self.word_order

        super().__init__(name, score)


class CHRF(Metric):
    """Computes the chrF(++) metric given hypotheses and references (``sacrebleu.CHRF``).

    Args:
        char_order: Character n-gram order.
        word_order: Word n-gram order. If equals to 2, the metric is referred to as chrF++.
        beta: Determine the importance of recall w.r.t precision.
        lowercase: Enable case-insensitivity.
        whitespace: If ``True``, include whitespaces when extracting character n-grams.
        eps_smoothing: If ``True``, applies epsilon smoothing similar to reference
            chrF++.py, NLTK and Moses implementations. Otherwise, it takes into account
            effective match order similar to sacreBLEU < 2.0.0.
        references: A sequence of reference documents with document being defined as a
            sequence of reference strings. If given, the reference n-grams will be
            pre-computed and cached for faster re-computation across many systems.
    """

    # Maximum character n-gram order to take into account
    CHAR_ORDER = 6

    # chrF+ additionally takes into account some of the word n-grams
    WORD_ORDER = 0

    # Defaults to 2 (per http://www.aclweb.org/anthology/W16-2341)
    BETA = 2

    # Cache string.punctuation for chrF+' punctuation stripper
    _PUNCTS = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    _SIGNATURE_TYPE = CHRFSignature

    def __init__(
        self,
        char_order: int = CHAR_ORDER,
        word_order: int = WORD_ORDER,
        beta: int = BETA,
        lowercase: bool = False,
        whitespace: bool = False,
        eps_smoothing: bool = False,
        references: Sequence[Sequence[str]] | None = None,
    ):
        super().__init__()

        self.beta = beta
        self.char_order = char_order
        self.word_order = word_order
        self.order = self.char_order + self.word_order
        self.lowercase = lowercase
        self.whitespace = whitespace
        self.eps_smoothing = eps_smoothing

        if references is not None:
            # Pre-compute reference ngrams
            self._ref_cache = self._cache_references(references)

    @staticmethod
    def _get_match_statistics(hyp_ngrams: Counter[Any], ref_ngrams: Counter[Any]) -> list[int]:
        """``[hyp count, ref count, matches]`` of hypothesis and reference n-grams."""
        # Counter's internal intersection is not that fast, count manually
        match_count, hyp_count = 0, 0
        for ng, count in hyp_ngrams.items():
            hyp_count += count
            if ng in ref_ngrams:
                match_count += min(count, ref_ngrams[ng])

        return [
            # Don't count hits if no reference exists for that n-gram
            hyp_count if ref_ngrams else 0,
            sum(ref_ngrams.values()),
            match_count,
        ]

    def _remove_punctuation(self, sent: str) -> list[str]:
        """Separates out punctuations from beginning and end of words for chrF."""
        tokenized = []
        for w in sent.split():
            if len(w) == 1:
                tokenized.append(w)
            else:
                # NOTE: This splits '(hi)' to '(hi' and ')' (issue #124)
                if w[-1] in self._PUNCTS:
                    tokenized += [w[:-1], w[-1]]
                elif w[0] in self._PUNCTS:
                    tokenized += [w[0], w[1:]]
                else:
                    tokenized.append(w)
        return tokenized

    def _preprocess_segment(self, sent: str) -> str:
        """Given a sentence, apply optional lowercasing."""
        return sent.lower() if self.lowercase else sent

    def _compute_f_score(self, statistics: list[int]) -> float:
        """Compute the chrF score given the n-gram match statistics.

        Args:
            statistics: A flattened list of 3 * (``char_order`` + ``word_order``) elements
                giving the [hyp, ref, match] counts for each order.

        Returns:
            The final f_beta score between [0, 100].
        """
        eps = 1e-16
        score = 0.0
        effective_order = 0
        factor = self.beta**2
        avg_prec, avg_rec = 0.0, 0.0

        for i in range(self.order):
            n_hyp, n_ref, n_match = statistics[3 * i : 3 * i + 3]

            # chrF++.py style EPS smoothing (also used by Moses and NLTK)
            prec = n_match / n_hyp if n_hyp > 0 else eps
            rec = n_match / n_ref if n_ref > 0 else eps

            denom = factor * prec + rec
            score += ((1 + factor) * prec * rec / denom) if denom > 0 else eps

            # sacreBLEU <2.0.0 style effective order smoothing
            if n_hyp > 0 and n_ref > 0:
                avg_prec += prec
                avg_rec += rec
                effective_order += 1

        if self.eps_smoothing:
            return 100 * score / self.order

        if effective_order == 0:
            avg_prec = avg_rec = 0.0
        else:
            avg_prec /= effective_order
            avg_rec /= effective_order

        if avg_prec + avg_rec:
            score = (1 + factor) * avg_prec * avg_rec
            score /= (factor * avg_prec) + avg_rec
            return 100 * score
        else:
            return 0.0

    def _compute_score_from_stats(self, stats: list[int]) -> CHRFScore:
        """Computes the final score from already aggregated statistics."""
        return CHRFScore(self._compute_f_score(stats), self.char_order, self.word_order, self.beta)

    def _aggregate_and_compute(self, stats: list[list[int]]) -> CHRFScore:
        """Computes the final score given the pre-computed corpus statistics."""
        return self._compute_score_from_stats(sum_of_lists(stats))

    def _extract_reference_info(self, refs: Sequence[str]) -> dict[str, list[list[Counter[Any]]]]:
        """The character and word n-grams of each reference segment."""
        ngrams = []

        for ref in refs:
            # extract character n-grams
            stats: list[Counter[Any]] = extract_all_char_ngrams(
                ref, self.char_order, self.whitespace
            )

            # Check chrF+ mode
            if self.word_order > 0:
                ref_words = self._remove_punctuation(ref)

                for n in range(self.word_order):
                    stats.append(extract_word_ngrams(ref_words, n + 1))  # noqa: PERF401

            ngrams.append(stats)

        return {"ref_ngrams": ngrams}

    def _compute_segment_statistics(self, hypothesis: str, ref_kwargs: dict[str, Any]) -> list[int]:
        """The best-reference match statistics of a (pre-processed) hypothesis."""
        best_stats: list[int] = []
        best_f_score = -1.0

        # extract character n-grams
        all_hyp_ngrams: list[Counter[Any]] = extract_all_char_ngrams(
            hypothesis, self.char_order, self.whitespace
        )

        # Check chrF+ mode to see if we'll add word n-grams as well
        if self.word_order > 0:
            # Primitive tokenization: separate out punctuations
            hwords = self._remove_punctuation(hypothesis)
            _range = range(1, self.word_order + 1)
            all_hyp_ngrams.extend([extract_word_ngrams(hwords, n) for n in _range])

        # Iterate over multiple references, pick the one with best F score
        for _ref_ngrams in ref_kwargs["ref_ngrams"]:
            stats = []
            # Traverse all orders
            for h, r in zip(all_hyp_ngrams, _ref_ngrams, strict=False):
                stats.extend(self._get_match_statistics(h, r))
            f_score = self._compute_f_score(stats)

            if f_score > best_f_score:
                best_f_score = f_score
                best_stats = stats

        return best_stats

    def _native_factors(self) -> tuple[float, float] | None:
        """``(float(beta ** 2), float(1 + beta ** 2))``, or None for the pure-Python path.

        None when the Rust port cannot run these settings exactly as Python computes them.
        """
        c, w, beta = self.char_order, self.word_order, self.beta
        if not (type(c) is int and type(w) is int and 0 <= c <= 256 and 0 <= w <= 256):
            return None
        if c + w == 0 or self.order != c + w:
            return None
        if type(beta) is int:
            if abs(beta) >= 1 << 26:
                return None
        elif type(beta) is not float or not math.isfinite(beta * beta):
            return None
        factor = beta**2
        return float(factor), float(1 + factor)

    def _cache_references(self, references: Sequence[Sequence[str]]) -> Any:
        """Pre-compute the references: a native cache, or sacrebleu's list on the Python path."""
        if self._native_factors() is None:
            return self._py_cache_references(references)
        streams = _streams(references)
        if self.lowercase:
            streams = _map_streams(streams, _n_ref_segments(streams), str.lower)
        cache = _native._ChrfRefCache(streams, self.char_order, self.word_order, self.whitespace)
        self.num_refs = cache.num_refs
        return cache

    def _statistics(
        self, hypotheses: Any, references: Any, want_rows: bool, threads: int | None
    ) -> tuple[list[Any] | None, list[list[Any]] | None]:
        factors = self._native_factors()
        if factors is None:
            return None, self._py_extract_corpus_statistics(hypotheses, references)
        hyps = _seq(hypotheses)
        settings = (
            self.char_order,
            self.word_order,
            self.whitespace,
            *factors,
            self.eps_smoothing,
            want_rows,
        )
        if references:
            streams = _streams(references)
            n_ref = _n_ref_segments(streams)
            if self.lowercase:
                streams = _map_streams(streams, n_ref, str.lower)
                hyps = [h.lower() for h in hyps[: min(len(hyps), n_ref)]]
            totals, n_segments, num_refs, rows = _native.chrf_corpus_stats(
                hyps, streams, *settings, threads=threads
            )
            self.num_refs = num_refs
        elif self._ref_cache:
            cache = self._ref_cache
            if not isinstance(cache, _native._ChrfRefCache) or cache.settings[:2] != (
                self.char_order,
                self.word_order,
            ):
                raise ValueError(
                    "pyroid.metrics.sacrebleu: char_order or word_order changed after the "
                    "references were cached; create a new CHRF object"
                )
            if self.lowercase:
                hyps = [h.lower() for h in hyps[: min(len(hyps), len(cache))]]
            totals, n_segments, _, rows = _native.chrf_cache_stats(
                cache, hyps, *settings, threads=threads
            )
        else:
            raise RuntimeError("No references provided and the cache is empty.")
        if n_segments == 0:
            return None, []
        return totals, rows

    def _sentence_score_batch(
        self, hypotheses: Any, references: Any, threads: int | None
    ) -> list[CHRFScore | None]:
        """``sentence_score(hypotheses[i], [r[i] for r in references])`` for every ``i``."""
        hyps, streams = _batch_args(hypotheses, references)
        n = len(hyps)
        hyps, streams, keep = _without_nulls(hyps, streams)
        if not hyps:
            return [None] * n
        if self._native_factors() is not None:
            rows = self._statistics(hyps, streams, True, threads)[1]
        else:
            rows = [
                self._py_extract_corpus_statistics([h], [[s[i]] for s in streams])[0]
                for i, h in enumerate(hyps)
            ]
        assert rows is not None
        return _with_nulls([self._compute_score_from_stats(row) for row in rows], keep, n)


def _is_arrow(x: Any) -> bool:
    return not isinstance(x, (str, list, tuple)) and (
        hasattr(x, "__arrow_c_stream__") or hasattr(x, "__arrow_c_array__")
    )


def _input_error(message: str, name: str) -> InputError:
    err = InputError(message)
    err.add_note(f"while processing {name!r}")
    return err


def _batch_args(hypotheses: Any, references: Any) -> tuple[Any, list[Any]]:
    """Checked arguments of the ``sentence_*_batch`` functions.

    Arrow columns become lists, with ``None`` for each null.
    """
    if _is_arrow(hypotheses):
        hyps = _native.batch_texts(hypotheses, "hypotheses")
    else:
        if isinstance(hypotheses, str) or not isinstance(hypotheses, Sequence):
            raise _input_error(
                "hypotheses must be a sequence of str or an Arrow string array, got "
                f"{type(hypotheses).__qualname__}",
                "hypotheses",
            )
        hyps = _seq(hypotheses)
        for i, h in enumerate(hyps):
            if h is None:
                raise _input_error(f"hypotheses[{i}] must be a str, got NoneType", "hypotheses")
    if isinstance(references, str) or not isinstance(references, Sequence):
        raise _input_error(
            "references must be a sequence of reference streams (references[k][i] is the "
            f"k-th reference of hypothesis i), got {type(references).__qualname__}",
            "references",
        )
    references = [
        _native.batch_texts(s, f"references[{k}]") if _is_arrow(s) else s
        for k, s in enumerate(references)
    ]
    for k, s in enumerate(references):
        if isinstance(s, str) or not isinstance(s, Sequence):
            raise _input_error(
                f"references[{k}] must be a sequence of str or None (one per hypothesis) or an "
                f"Arrow string array, got {type(s).__qualname__}; with one reference per "
                "hypothesis, pass [refs]",
                "references",
            )
    streams = _streams(references)
    if not streams:
        raise ValueError("references must hold at least one reference stream")
    for k, s in enumerate(streams):
        if len(s) != len(hyps):
            raise ValueError(
                f"references[{k}] has {len(s)} references, but there are {len(hyps)} hypotheses"
            )
    return hyps, streams


def _without_nulls(hyps: Any, streams: list[Any]) -> tuple[Any, list[Any], list[int] | None]:
    """The segments whose hypothesis is not ``None`` (an Arrow null), and their indices.

    The indices are None when every segment is kept.
    """
    keep = [i for i, h in enumerate(hyps) if h is not None]
    if len(keep) == len(hyps):
        return hyps, streams, None
    return [hyps[i] for i in keep], [[s[i] for i in keep] for s in streams], keep


def _with_nulls(scores: list[Any], keep: list[int] | None, n: int) -> list[Any]:
    """``scores`` of the kept segments back in place, ``None`` for the others."""
    if keep is None:
        return scores
    out: list[Any] = [None] * n
    for i, s in zip(keep, scores, strict=True):
        out[i] = s
    return out


# ---------------------------------------------------------------------------
# sacrebleu.compat
# ---------------------------------------------------------------------------


def _bleu(api: str, **kwargs: Any) -> BLEU:
    """A BLEU object made for the public function ``api``."""
    metric = BLEU.__new__(BLEU)
    metric._init(api, **kwargs)
    return metric


def corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    force: bool = False,
    lowercase: bool = False,
    tokenize: str = BLEU.TOKENIZER_DEFAULT,
    use_effective_order: bool = False,
    *,
    threads: int | None = None,
) -> BLEUScore:
    """Computes BLEU for a corpus against a single (or multiple) reference(s).

    See ``sacrebleu.pyi`` for the arguments.
    """
    metric = _bleu(
        "corpus_bleu",
        lowercase=lowercase,
        force=force,
        tokenize=tokenize,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=use_effective_order,
    )

    return metric.corpus_score(hypotheses, references, threads=threads)  # type: ignore[no-any-return]


def raw_corpus_bleu(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    smooth_value: float | None = BLEU.SMOOTH_DEFAULTS["floor"],
    *,
    threads: int | None = None,
) -> BLEUScore:
    """Computes BLEU on the raw corpus: no tokenization, ``floor`` smoothing (0.1 by default).

    See ``sacrebleu.pyi`` for the arguments.
    """
    return corpus_bleu(
        hypotheses,
        references,
        smooth_method="floor",
        smooth_value=smooth_value,
        force=True,
        tokenize="none",
        use_effective_order=True,
        threads=threads,
    )


def sentence_bleu(
    hypothesis: str,
    references: Sequence[str],
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    lowercase: bool = False,
    tokenize: str = BLEU.TOKENIZER_DEFAULT,
    use_effective_order: bool = True,
) -> BLEUScore:
    """Computes BLEU for a single sentence against a single (or multiple) reference(s).

    See ``sacrebleu.pyi`` for the arguments.
    """
    metric = _bleu(
        "sentence_bleu",
        lowercase=lowercase,
        tokenize=tokenize,
        force=False,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=use_effective_order,
    )

    return metric.sentence_score(hypothesis, references)


def sentence_bleu_batch(
    hypotheses: Sequence[str] | ArrowStrings,
    references: Sequence[Sequence[str | None] | ArrowStrings],
    smooth_method: str = "exp",
    smooth_value: float | None = None,
    lowercase: bool = False,
    tokenize: str = BLEU.TOKENIZER_DEFAULT,
    use_effective_order: bool = True,
    *,
    threads: int | None = None,
) -> list[BLEUScore | None]:
    """:func:`sentence_bleu` of every hypothesis at once, on pyroid's thread pool.

    See ``sacrebleu.pyi`` for the arguments.
    """
    metric = _bleu(
        "sentence_bleu_batch",
        lowercase=lowercase,
        tokenize=tokenize,
        force=False,
        smooth_method=smooth_method,
        smooth_value=smooth_value,
        effective_order=use_effective_order,
    )
    return metric._sentence_score_batch(hypotheses, references, threads)


def corpus_chrf(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    char_order: int = CHRF.CHAR_ORDER,
    word_order: int = CHRF.WORD_ORDER,
    beta: int = CHRF.BETA,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
    *,
    threads: int | None = None,
) -> CHRFScore:
    """Computes chrF for a corpus against a single (or multiple) reference(s).

    See ``sacrebleu.pyi`` for the arguments.
    """
    metric = CHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=not remove_whitespace,
        eps_smoothing=eps_smoothing,
    )
    return metric.corpus_score(hypotheses, references, threads=threads)  # type: ignore[no-any-return]


def sentence_chrf(
    hypothesis: str,
    references: Sequence[str],
    char_order: int = CHRF.CHAR_ORDER,
    word_order: int = CHRF.WORD_ORDER,
    beta: int = CHRF.BETA,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
) -> CHRFScore:
    """Computes chrF for a single sentence against a single (or multiple) reference(s).

    See ``sacrebleu.pyi`` for the arguments.
    """
    metric = CHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=not remove_whitespace,
        eps_smoothing=eps_smoothing,
    )
    return metric.sentence_score(hypothesis, references)  # type: ignore[no-any-return]


def sentence_chrf_batch(
    hypotheses: Sequence[str] | ArrowStrings,
    references: Sequence[Sequence[str | None] | ArrowStrings],
    char_order: int = CHRF.CHAR_ORDER,
    word_order: int = CHRF.WORD_ORDER,
    beta: int = CHRF.BETA,
    remove_whitespace: bool = True,
    eps_smoothing: bool = False,
    *,
    threads: int | None = None,
) -> list[CHRFScore | None]:
    """:func:`sentence_chrf` of every hypothesis at once, on pyroid's thread pool.

    See ``sacrebleu.pyi`` for the arguments.
    """
    metric = CHRF(
        char_order=char_order,
        word_order=word_order,
        beta=beta,
        whitespace=not remove_whitespace,
        eps_smoothing=eps_smoothing,
    )
    return metric._sentence_score_batch(hypotheses, references, threads)


# ---------------------------------------------------------------------------
# Delegated to the real sacrebleu
# ---------------------------------------------------------------------------


class TER:
    """sacrebleu's TER metric (delegated: wraps a real ``sacrebleu.TER``).

    Args:
        normalized: Enable character normalization.
        no_punct: Remove punctuation.
        asian_support: Enable special treatment of Asian characters.
        case_sensitive: Enables case-sensitivity.
        references: Reference streams to pre-compute.
    """

    def __init__(
        self,
        normalized: bool = False,
        no_punct: bool = False,
        asian_support: bool = False,
        case_sensitive: bool = False,
        references: Sequence[Sequence[str]] | None = None,
    ):
        real = _sacrebleu("TER", "TER")
        self._ter = real.TER(
            normalized=normalized,
            no_punct=no_punct,
            asian_support=asian_support,
            case_sensitive=case_sensitive,
            references=references,
        )

    def corpus_score(
        self,
        hypotheses: Sequence[str],
        references: Sequence[Sequence[str]] | None,
        n_bootstrap: int = 1,
    ) -> Any:
        """``sacrebleu.TER.corpus_score`` (returns a ``sacrebleu.metrics.ter.TERScore``)."""
        return self._ter.corpus_score(hypotheses, references, n_bootstrap)

    def sentence_score(self, hypothesis: str, references: Sequence[str]) -> Any:
        """``sacrebleu.TER.sentence_score`` (returns a ``sacrebleu.metrics.ter.TERScore``)."""
        return self._ter.sentence_score(hypothesis, references)

    def get_signature(self) -> Any:
        """``sacrebleu.TER.get_signature``."""
        return self._ter.get_signature()

    def __getattr__(self, name: str) -> Any:
        if name == "_ter":
            raise AttributeError(name)
        return getattr(self._ter, name)


def corpus_ter(
    hypotheses: Sequence[str],
    references: Sequence[Sequence[str]],
    normalized: bool = False,
    no_punct: bool = False,
    asian_support: bool = False,
    case_sensitive: bool = False,
) -> Any:
    """``sacrebleu.corpus_ter`` (delegated)."""
    real = _sacrebleu("corpus_ter", "TER")
    return real.corpus_ter(
        hypotheses,
        references,
        normalized=normalized,
        no_punct=no_punct,
        asian_support=asian_support,
        case_sensitive=case_sensitive,
    )


def sentence_ter(
    hypothesis: str,
    references: Sequence[str],
    normalized: bool = False,
    no_punct: bool = False,
    asian_support: bool = False,
    case_sensitive: bool = False,
) -> Any:
    """``sacrebleu.sentence_ter`` (delegated)."""
    real = _sacrebleu("sentence_ter", "TER")
    return real.sentence_ter(
        hypothesis,
        references,
        normalized=normalized,
        no_punct=no_punct,
        asian_support=asian_support,
        case_sensitive=case_sensitive,
    )


class _Datasets(Mapping[str, Any]):
    """``sacrebleu.DATASETS``, read from sacrebleu on first use."""

    def _real(self) -> Mapping[str, Any]:
        return _sacrebleu("DATASETS", "the test set registry").DATASETS  # type: ignore[no-any-return]

    def __getitem__(self, key: str) -> Any:
        return self._real()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._real())

    def __len__(self) -> int:
        return len(self._real())

    def __repr__(self) -> str:
        return "<sacrebleu.DATASETS (delegated to sacrebleu on first use)>"


DATASETS: Mapping[str, Any] = _Datasets()


def download_test_set(test_set: str, langpair: str | None = None) -> Any:
    """``sacrebleu.download_test_set`` (delegated)."""
    return _sacrebleu("download_test_set", "test sets").download_test_set(test_set, langpair)


def get_source_file(test_set: str, langpair: str) -> str:
    """``sacrebleu.get_source_file`` (delegated)."""
    return _sacrebleu("get_source_file", "test sets").get_source_file(test_set, langpair)  # type: ignore[no-any-return]


def get_reference_files(test_set: str, langpair: str) -> list[str]:
    """``sacrebleu.get_reference_files`` (delegated)."""
    real = _sacrebleu("get_reference_files", "test sets")
    return real.get_reference_files(test_set, langpair)  # type: ignore[no-any-return]


def get_available_testsets() -> list[str]:
    """``sacrebleu.get_available_testsets`` (delegated)."""
    return _sacrebleu("get_available_testsets", "test sets").get_available_testsets()  # type: ignore[no-any-return]


def get_langpairs_for_testset(testset: str) -> list[str]:
    """``sacrebleu.get_langpairs_for_testset`` (delegated)."""
    real = _sacrebleu("get_langpairs_for_testset", "test sets")
    return real.get_langpairs_for_testset(testset)  # type: ignore[no-any-return]
