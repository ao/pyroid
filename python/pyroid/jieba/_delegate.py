"""Calls that ``pyroid.jieba`` hands to the real jieba 0.42.1.

Part-of-speech tagging (``posseg``), keyword extraction (``analyse``), paddle mode and the
default dictionary's file are not ported. Before each delegated call, the words added to
the pyroid tokenizer involved (``add_word``, ``del_word``, ``load_userdict``,
``suggest_freq(..., tune=True)``) are replayed, in order, onto its jieba counterpart:
``jieba.dt`` for ``pyroid.jieba.dt``, else a ``jieba.Tokenizer`` with the same dictionary.
"""

from __future__ import annotations

import importlib
import sys
import threading
import warnings
import weakref
from typing import Any

from pyroid._pyroid import FallbackWarning, PyroidError

_LOCK = threading.RLock()
_WARNED: set[str] = set()
#: pyroid Tokenizer -> [jieba Tokenizer, the pyroid tokenizer's generation, ops replayed]
_UPSTREAM: weakref.WeakKeyDictionary[Any, list[Any]] = weakref.WeakKeyDictionary()


def _stacklevel() -> int:
    """``stacklevel`` for warnings.warn that points at the first frame outside pyroid."""
    frame = sys._getframe(1)
    level = 1
    while frame.f_back is not None and str(frame.f_globals.get("__name__", "")).startswith(
        "pyroid."
    ):
        frame = frame.f_back
        level += 1
    return level


def jieba_module(what: str) -> Any:
    """The real jieba, after a one-time :class:`pyroid.FallbackWarning` for ``what``."""
    try:
        jieba = importlib.import_module("jieba")
    except ImportError as e:
        raise PyroidError(
            f"pyroid.jieba does not port {what}; it delegates that to jieba, which is not "
            "installed (pip install jieba==0.42.1)"
        ) from e
    with _LOCK:
        first = what not in _WARNED
        _WARNED.add(what)
    if first:
        warnings.warn(
            f"pyroid.jieba: {what} is not ported to Rust; calling jieba {jieba.__version__}",
            FallbackWarning,
            stacklevel=_stacklevel(),
        )
    return jieba


def upstream(tok: Any, what: str) -> Any:
    """Jieba's counterpart of the pyroid Tokenizer ``tok``, with ``tok``'s added words replayed."""
    jieba = jieba_module(what)
    from pyroid import jieba as pj

    tok.check_initialized()
    with _LOCK:
        st = _UPSTREAM.get(tok)
        if st is None or st[1] != tok._generation:
            if tok is pj.dt:
                up = jieba.dt
                if up.dictionary != tok.dictionary:
                    with up.lock:
                        up.dictionary = tok.dictionary
                        up.initialized = False
            else:
                up = jieba.Tokenizer(tok.dictionary)
            st = [up, tok._generation, 0]
            _UPSTREAM[tok] = st
        up = st[0]
        ops = tok._ops
        for word, freq, tag in ops[st[2] :]:
            up.add_word(word, freq, tag)
        st[2] = len(ops)
        return up


def submodule(name: str) -> Any:
    """``jieba.<name>`` (``posseg`` or ``analyse``), after replaying ``pyroid.jieba.dt``."""
    from pyroid import jieba as pj

    upstream(pj.dt, f"jieba.{name}")
    return importlib.import_module(f"jieba.{name}")


def paddle_enabled() -> bool:
    """Whether jieba's paddle mode was enabled (``enable_paddle``)."""
    compat = sys.modules.get("jieba._compat")
    return bool(compat is not None and compat.check_paddle_install.get("is_paddle_installed"))


def enable_paddle() -> None:
    """``jieba.enable_paddle()``, without its ``pip install`` of paddlepaddle-tiny."""
    jieba = jieba_module("paddle mode")
    try:
        importlib.import_module("paddle")
    except ImportError as e:
        raise PyroidError(
            "pyroid.jieba: paddle mode needs paddlepaddle-tiny "
            "(pip install paddlepaddle-tiny==1.6.1); unlike jieba, pyroid does not install it"
        ) from e
    jieba.enable_paddle()


def paddle_cut(sentence: Any) -> Any:
    """``jieba.cut(sentence, use_paddle=True)`` on ``jieba.dt``."""
    from pyroid import jieba as pj

    return upstream(pj.dt, "paddle mode").cut(sentence, use_paddle=True)
