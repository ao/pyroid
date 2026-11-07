"""Part-of-speech tagging, delegated to ``jieba.posseg`` (jieba 0.42.1).

pyroid does not port ``jieba.posseg``. Every call here goes to jieba, with a
one-time :class:`pyroid.FallbackWarning` (:class:`pyroid.PyroidError` if jieba
is not installed), after the words added to ``pyroid.jieba.dt`` are replayed
onto ``jieba.dt``.
"""

from __future__ import annotations

from typing import Any

from pyroid.jieba import _delegate


def _posseg() -> Any:
    return _delegate.submodule("posseg")


def cut(sentence: Any, HMM: bool = True, use_paddle: bool = False) -> Any:  # noqa: N803
    """``jieba.posseg.cut``: yields ``pair(word, flag)``."""
    return _posseg().cut(sentence, HMM, use_paddle)


def lcut(sentence: Any, HMM: bool = True, use_paddle: bool = False) -> list[Any]:  # noqa: N803
    """``jieba.posseg.lcut``: a list of ``pair(word, flag)``."""
    return _posseg().lcut(sentence, HMM, use_paddle)  # type: ignore[no-any-return]


def POSTokenizer(tokenizer: Any = None) -> Any:  # noqa: N802  jieba's class name
    """``jieba.posseg.POSTokenizer``; a pyroid ``Tokenizer`` is replaced by jieba's copy of it."""
    from pyroid import jieba as pj

    mod = _posseg()
    if isinstance(tokenizer, pj.Tokenizer):
        tokenizer = _delegate.upstream(tokenizer, "jieba.posseg")
    return mod.POSTokenizer(tokenizer)


def __getattr__(name: str) -> Any:
    return getattr(_posseg(), name)
