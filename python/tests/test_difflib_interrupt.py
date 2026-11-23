"""Ctrl-C during a long pyroid.difflib call.

A SIGINT that arrives while a long native call runs (matching, a diff of large inputs, a batch)
raises ``KeyboardInterrupt`` within a fraction of a second, and the module keeps working afterwards:
the call checks for signals about every 0.1 s, and a batch stops between items. Before, the
interrupt came only when the call had finished (52.7 s into a 200,000-line ``unified_diff``). Each
case runs in a fresh interpreter, sized to run for seconds uninterrupted; the signal comes after
0.3 s, or after 0.02 s (before the call first looks for it).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable

import pytest

RunPython = Callable[..., subprocess.CompletedProcess[str]]

pytestmark = pytest.mark.thread_unsafe(reason="runs subprocesses that signal themselves")

#: Seconds from the signal to the KeyboardInterrupt: the call asks every 0.1 s; the rest is
#: headroom for a loaded CI runner and a debug build.
LATENCY = 2.0

#: A case whose call ends before the signal (a fast runner) runs again with this many times the
#: input, up to the last factor.
SCALES = (1, 2, 4)

CHILD = r"""
import signal, sys, threading, time
import pyroid.difflib as pd

def alternating(n):
    # Every other line changed: line matching is quadratic in the number of lines.
    return [f"{i}\n" for i in range(n)], [f"{i}\n" if i % 2 else f"x{i}\n" for i in range(n)]

if case == "opcodes":
    n = 14_000 * scale
    a, b = [i % 40 for i in range(n)], [(i * 7) % 40 for i in range(n)]
    call = lambda: pd.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    small = lambda: pd.SequenceMatcher(None, a[:50], b[:50]).get_opcodes()
elif case == "ratio":
    n = 14_000 * scale
    a, b = [i % 40 for i in range(n)], [(i * 7) % 40 for i in range(n)]
    call = lambda: pd.SequenceMatcher(None, a, b, autojunk=False).ratio()
    small = lambda: pd.SequenceMatcher(None, a[:50], b[:50]).ratio()
elif case == "find_longest_match":
    a = [0] * (60_000 * scale)
    call = lambda: pd.SequenceMatcher(None, a, a, autojunk=False).find_longest_match()
    small = lambda: pd.SequenceMatcher(None, a[:50], a[:50]).find_longest_match()
elif case == "unified_diff":
    a, b = alternating(40_000 * scale)
    call = lambda: list(pd.unified_diff(a, b))
    small = lambda: list(pd.unified_diff(a[:50], b[:50]))
elif case == "context_diff":
    a, b = alternating(40_000 * scale)
    call = lambda: list(pd.context_diff(a, b))
    small = lambda: list(pd.context_diff(a[:50], b[:50]))
elif case == "ndiff":
    a, b = alternating(40_000 * scale)
    call = lambda: list(pd.ndiff(a, b))
    small = lambda: list(pd.ndiff(a[:50], b[:50]))
elif case == "ndiff_similar":
    # 3.11 to 3.13: Differ compares every pair of lines of a replaced block.
    n = 5_000 * scale
    a = [f"value = compute(x{i % 97}, y{i % 89})\n" for i in range(n)]
    b = [f"value = compute(x{i % 89}, y{i % 97})\n" for i in range(n)]
    call = lambda: list(pd.ndiff(a, b))
    small = lambda: list(pd.ndiff(a[:50], b[:50]))
elif case == "make_table":
    a, b = alternating(30_000 * scale)
    call = lambda: pd.HtmlDiff().make_table(a, b)
    small = lambda: pd.HtmlDiff().make_table(a[:50], b[:50])
elif case == "get_close_matches":
    word = "abcdefghij" * 20
    poss = [word[:i] + "x" + word[i + 1:] for i in range(200)] * (1_000 * scale)
    call = lambda: pd.get_close_matches(word, poss, threads=threads)
    small = lambda: pd.get_close_matches(word, poss[:50], threads=threads)
elif case == "ratio_batch":
    word = "abcdefghij" * 20
    a = [word[:i] + "x" + word[i + 1:] for i in range(200)] * (1_000 * scale)
    call = lambda: pd.ratio_batch(a, a[::-1], threads=threads)
    small = lambda: pd.ratio_batch(a[:50], a[:50], threads=threads)
elif case == "close_matches_batch":
    names = [f"name {i * 7919 % 100_003} street" for i in range(20_000)]
    words = names[: 300 * scale]
    call = lambda: pd.close_matches_batch(words, names, threads=threads)
    small = lambda: pd.close_matches_batch(words[:5], names[:50], threads=threads)
elif case == "close_matches_batch_few":
    # Under 128 words: one word at a time, each on all threads.
    names = [f"name {i * 7919 % 1_000_003} street" for i in range(200_000)]
    words = names[: 40 * scale]
    call = lambda: pd.close_matches_batch(words, names, threads=threads)
    small = lambda: pd.close_matches_batch(words[:5], names[:50], threads=threads)
else:
    raise SystemExit(f"unknown case {case}")

fired = []
def fire():
    fired.append(time.perf_counter())
    signal.raise_signal(signal.SIGINT)

t0 = time.perf_counter()
timer = threading.Timer(delay, fire)
timer.start()
done = []
try:
    call()
    done.append(time.perf_counter())
    timer.cancel()
    timer.join()
except KeyboardInterrupt:
    if not done:
        print("interrupted", round(time.perf_counter() - fired[0], 3))
except BaseException as e:
    print("raised", type(e).__module__ + "." + type(e).__name__, e)
if done:
    # The call ended before the signal (or the signal came just after it): too fast to measure.
    print("finished", round(done[0] - t0, 3))
small()
print("after ok")
"""

CASES = [
    ("opcodes", 1),
    ("ratio", 1),
    ("find_longest_match", 1),
    ("unified_diff", 1),
    ("context_diff", 1),
    ("ndiff", 1),
    pytest.param(
        "ndiff_similar",
        1,
        marks=pytest.mark.skipif(
            sys.version_info >= (3, 14), reason="3.14's Differ compares a window of lines"
        ),
    ),
    ("make_table", 1),
    ("get_close_matches", 1),
    ("get_close_matches", 2),
    ("ratio_batch", 1),
    ("ratio_batch", 2),
    ("close_matches_batch", 1),
    ("close_matches_batch", 2),
    ("close_matches_batch_few", 1),
    ("close_matches_batch_few", 2),
]


@pytest.mark.parametrize("delay", [0.3, 0.02])
@pytest.mark.parametrize(("case", "threads"), CASES)
def test_ctrl_c_interrupts_a_long_call(
    run_python: RunPython, case: str, threads: int, delay: float
) -> None:
    for scale in SCALES:
        head = f"case, threads, scale, delay = {case!r}, {threads}, {scale}, {delay}\n"
        r = run_python(head + CHILD, timeout=600)
        assert r.returncode == 0, r.stderr
        lines = r.stdout.split()
        assert lines[-2:] == ["after", "ok"], r.stdout + r.stderr
        if lines[0] != "finished":
            break
    assert lines[0] == "interrupted", r.stdout + r.stderr
    assert float(lines[1]) < LATENCY, r.stdout
