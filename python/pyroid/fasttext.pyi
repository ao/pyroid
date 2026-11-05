"""fastText inference in Rust (``pyroid.fasttext``)."""

import os
from typing import Any, Literal, TypeAlias, final, overload

import numpy as np
import numpy.typing as npt

from pyroid import PyroidError
from pyroid._typing import ArrowStrings

__all__ = ["Model", "ModelError", "load_model"]

_F32: TypeAlias = npt.NDArray[np.float32]
_F64: TypeAlias = npt.NDArray[np.float64]
_I64: TypeAlias = npt.NDArray[np.int64]
_Batch: TypeAlias = list[str] | tuple[str, ...] | ArrowStrings

class ModelError(PyroidError, ValueError):
    """A fastText model file is malformed, truncated, or not a fastText model.

    A ``ValueError``, like the errors fastText's own loader raises.
    """

def load_model(path: str | os.PathLike[str]) -> Model:
    """Load a fastText model: a ``.bin`` file, or a quantized ``.ftz``.

    The whole file is read and validated in one pass (the GIL is released
    meanwhile). The model is immutable: share it across threads.

    Args:
        path: The model file.

    Returns:
        The loaded :class:`Model`.

    Raises:
        ModelError: The file is not a fastText model (``"... has wrong file
            format!"``), is truncated, or is internally inconsistent (sizes or
            ids that fastText itself would read out of bounds).
        OSError: The file cannot be opened or read (``FileNotFoundError``,
            ``PermissionError``, ...; fastText raises ``ValueError`` here).
    """

@final
class Model:
    """A loaded fastText model.

    The inference methods of fastText's ``_FastText``, with the same results.
    :meth:`predict` and :meth:`get_sentence_vector` also accept a batch: a list
    or tuple of str, or an Arrow string array/column, processed on pyroid's
    thread pool without the GIL.
    """

    @overload
    def predict(
        self,
        text: str,
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
        *,
        threads: int | None = None,
    ) -> tuple[tuple[str, ...], _F64]:
        r"""Predict the most likely labels of each line of text.

        As fastText: the text is split on spaces, tabs, vertical tabs, form
        feeds, carriage returns and NULs; tokens with the label prefix are
        ignored; the end-of-line token ``</s>`` is always added (so an empty
        line still has predictions). Probabilities are fastText's, which add
        ``1e-5`` and so can slightly exceed 1.

        Args:
            text: One line (no ``"\n"``), or a batch of lines: a list or tuple
                of str, or an Arrow string array/column.
            k: Number of labels to return, best first; ``-1`` for all of them.
            threshold: Only return labels with at least this probability.
            on_unicode_error: Error handler for decoding labels that are not
                valid UTF-8 (``"strict"``, ``"replace"``, ...).
            threads: ``None`` uses pyroid's shared thread pool; ``n`` runs at
                most ``n`` tasks at once; ``1`` runs on the calling thread.

        Returns:
            For a str: ``(labels, probabilities)``, a tuple of str and a
            float64 array. For a batch: ``(list of label lists, list of float32
            arrays)``, one entry per line (``None`` for Arrow nulls).

        Raises:
            ValueError: A line contains ``"\n"``; ``k`` is 0 or below -1; the
                model is unsupervised.
            TypeError: A line contains a lone surrogate (as fastText).
            pyroid.InputError: ``text`` is not a str or a batch of str.
            RuntimeError: The model's weights produce NaN (``"Encountered NaN."``).
        """

    @overload
    def predict(
        self,
        text: _Batch,
        k: int = 1,
        threshold: float = 0.0,
        on_unicode_error: str = "strict",
        *,
        threads: int | None = None,
    ) -> tuple[list[list[str]], list[_F32]]: ...
    @overload
    def get_sentence_vector(self, text: str, *, threads: int | None = None) -> _F32:
        r"""The sentence vector of a line (float32, ``get_dimension()`` values).

        Supervised models average the input rows ``predict`` uses (the hidden
        layer); unsupervised models average the L2-normalized vectors of the
        whitespace-separated words.

        Args:
            text: One line (no ``"\n"``), or a batch of lines.
            threads: As for :meth:`predict`.

        Returns:
            A 1-D float32 array for a str; a 2-D ``(len(batch), dim)`` float32
            array for a batch (rows of NaN for Arrow nulls).

        Raises:
            ValueError: A line contains ``"\n"``.
            TypeError: A line contains a lone surrogate.
        """

    @overload
    def get_sentence_vector(self, text: _Batch, *, threads: int | None = None) -> _F32: ...
    def get_word_vector(self, word: str) -> _F32:
        """The vector of ``word``: the mean of its subword rows (float32).

        Works for any word: unknown words use their character n-grams.
        """

    def get_input_vector(self, ind: int) -> _F32:
        """Row ``ind`` of the input matrix (float32).

        Raises:
            IndexError: ``ind`` is outside the input matrix.
        """

    @overload
    def get_words(
        self, include_freq: Literal[False] = False, on_unicode_error: str = "strict"
    ) -> list[str]:
        """The words of the dictionary (no labels, no subwords).

        Args:
            include_freq: Also return each word's training count.
            on_unicode_error: Error handler for words that are not valid UTF-8.

        Returns:
            The words, or ``(words, counts)`` with an int64 array of counts.
        """

    @overload
    def get_words(
        self, include_freq: Literal[True], on_unicode_error: str = "strict"
    ) -> tuple[list[str], _I64]: ...
    @overload
    def get_labels(
        self, include_freq: Literal[False] = False, on_unicode_error: str = "strict"
    ) -> list[str]:
        """The labels, prefix included. Unsupervised models return their words.

        Args:
            include_freq: Also return each label's training count.
            on_unicode_error: Error handler for labels that are not valid UTF-8.

        Returns:
            The labels, or ``(labels, counts)`` with an int64 array of counts.
        """

    @overload
    def get_labels(
        self, include_freq: Literal[True], on_unicode_error: str = "strict"
    ) -> tuple[list[str], _I64]: ...
    @property
    def words(self) -> list[str]:
        """The dictionary's words (``get_words()``, computed once)."""

    @property
    def labels(self) -> list[str]:
        """The model's labels (``get_labels()``, computed once)."""

    @overload
    def get_line(self, text: str, on_unicode_error: str = "strict") -> tuple[list[str], list[str]]:
        r"""Split a line into its word tokens (``</s>`` included) and known labels.

        Args:
            text: One line (no ``"\n"``), or a batch of lines.
            on_unicode_error: Accepted for compatibility (tokens of a str are
                always valid UTF-8).

        Returns:
            ``(words, labels)`` for a str; ``(list of word lists, list of label
            lists)`` for a batch.

        Raises:
            ValueError: A line contains ``"\n"``.
        """

    @overload
    def get_line(
        self, text: _Batch, on_unicode_error: str = "strict"
    ) -> tuple[list[list[str]], list[list[str]]]: ...
    def get_subwords(self, word: str, on_unicode_error: str = "strict") -> tuple[list[str], _I64]:
        """The word (if known) and its character n-grams, with their input rows.

        As in fastText, a pruned (quantized) model lists n-grams that have no
        row, so the two results can differ in length.
        """

    def get_word_id(self, word: str) -> int:
        """Dictionary id of ``word`` (labels included), or -1."""

    def get_label_id(self, label: str) -> int:
        """Label index of ``label``, or -1.

        fastText's arithmetic: a known *word* gives a negative id other than -1.
        """

    def get_subword_id(self, subword: str) -> int:
        """The input-matrix row a character n-gram hashes to (no pruning).

        Raises:
            ValueError: The model has no n-gram buckets.
        """

    def get_dimension(self) -> int:
        """Dimension of the word and sentence vectors."""

    def is_quantized(self) -> bool:
        """True for a quantized (``.ftz``) model."""

    def get_input_matrix(self) -> _F32:
        """A copy of the input matrix (float32, ``rows x dim``).

        Raises:
            ValueError: The model is quantized (``"Can't get quantized Matrix"``).
        """

    def get_output_matrix(self) -> _F32:
        """A copy of the output matrix (float32, ``rows x dim``).

        Raises:
            ValueError: The model is quantized (``"Can't get quantized Matrix"``).
        """

    def __getitem__(self, word: str, /) -> _F32: ...
    def __contains__(self, word: Any, /) -> bool: ...
