"""Containers that another thread changes while a pyroid call reads them.

On free-threaded CPython a list, dict or set can change while a pyroid call reads it, and on GIL
builds while the call runs Python code or releases the GIL. pyo3's dict and set iterators panic
when that happens (a dateutil test once died of it and hung the run-parallel pass), and a length
read before the items can be stale. Each test here has one thread keep changing what a module's
functions read while eight threads call them, and checks that:

- no call raises, and above all none raises ``pyo3_runtime.PanicException`` (a pyroid bug);
- no thread hangs;
- a list is read once, in one step: every result is the function's result for the list as it was
  at one moment (one of the states the changing thread puts it in), never a mix of two states.

The changing thread replaces the whole contents in one step (``lst[:] = ...``), so the states are
known in advance; the expected results are computed first, on one thread.
"""

from __future__ import annotations

import functools
import io
import sys
import threading
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import pyroid

pytestmark = pytest.mark.thread_unsafe(reason="starts its own threads")

#: Callers, as in the dateutil regression.
CALLERS = 8
#: How long the callers keep calling (each makes at least MIN_CALLS calls).
SECONDS = 0.4
MIN_CALLS = 3
#: A thread still running after this long is taken as hung.
HANG = 60.0


def _is_panic(e: BaseException) -> bool:
    return type(e).__name__ == "PanicException"


def hammer(
    call: Callable[[], Any],
    change: Callable[[int], None],
    allowed: Sequence[Any] | None = None,
    *,
    expected: tuple[type[BaseException], ...] = (),
) -> int:
    """Run `call` on CALLERS threads while another runs `change(0)`, `change(1)`, ...

    Every result must equal one of `allowed` (if given); an exception of a type in `expected`
    is the result ("raised", its type). Returns the number of calls made.
    """
    stop = threading.Event()
    errors: list[BaseException] = []
    bad: list[Any] = []
    calls = [0] * CALLERS

    def changer() -> None:
        i = 0
        try:
            while not stop.is_set():
                change(i)
                i += 1
        except BaseException as e:  # noqa: BLE001 - reported below
            errors.append(e)

    def caller(k: int) -> None:
        end = time.monotonic() + SECONDS
        try:
            while calls[k] < MIN_CALLS or time.monotonic() < end:
                out = _outcome(call, expected)
                if allowed is not None and not any(_same(out, a) for a in allowed):
                    bad.append(out)
                calls[k] += 1
        except BaseException as e:  # noqa: BLE001 - a PanicException is a BaseException
            errors.append(e)

    threads = [threading.Thread(target=caller, args=(k,), daemon=True) for k in range(CALLERS)]
    mutator = threading.Thread(target=changer, daemon=True)
    mutator.start()
    for t in threads:
        t.start()
    for t in threads:
        t.join(HANG)
    stop.set()
    mutator.join(HANG)
    hung = [t for t in [*threads, mutator] if t.is_alive()]
    assert not hung, f"{len(hung)} thread(s) still running after {HANG} s"
    panics = [e for e in errors if _is_panic(e)]
    if panics:
        raise AssertionError(f"PanicException: {panics[0]}") from panics[0]
    if errors:
        raise errors[0]
    assert not bad, f"{len(bad)} results match no state of the input, e.g. {bad[0]!r}"
    return sum(calls)


def _outcome(call: Callable[[], Any], expected: tuple[type[BaseException], ...]) -> Any:
    """`call()`, or ("raised", the exception type) for an exception of a type in `expected`."""
    try:
        return call()
    except expected as e:
        return ("raised", type(e))


def _same(a: Any, b: Any) -> bool:
    np = sys.modules.get("numpy")
    if np is not None and (isinstance(a, np.ndarray) or isinstance(b, np.ndarray)):
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(map(_same, a, b))
    return bool(a == b)


def states(target: list[Any], variants: Sequence[Sequence[Any]]) -> Callable[[int], None]:
    """A `change` that puts `target` in each of `variants` in turn, each in one step."""

    def change(i: int) -> None:
        target[:] = variants[i % len(variants)]

    return change


def over_states(
    fn: Callable[[list[Any]], Any], variants: Sequence[Sequence[Any]], **kw: Any
) -> None:
    """`fn(lst)` on many threads while `lst` cycles through `variants` (the common case)."""
    allowed = [_outcome(functools.partial(fn, list(v)), kw.get("expected", ())) for v in variants]
    lst = list(variants[0])
    hammer(lambda: fn(lst), states(lst, variants), allowed, **kw)


#: Two states of a document list: different lengths and contents, so a mix is visible.
DOCS_A = ["The  cat sat on the mat. It was   happy."] * 150 + ["café crème"]
DOCS_B = ["A dog\tbarked at the moon all night long."] * 90
VARIANTS: tuple[list[str], ...] = (DOCS_A, DOCS_B, [])


def test_text() -> None:
    import pyroid.text as t

    over_states(t.normalize_whitespace, VARIANTS)
    over_states(lambda d: t.min_chars(d, 20), VARIANTS)


def test_ftfy() -> None:
    import pyroid.ftfy as ftfy

    mojibake = (["cafÃ© crÃ¨me"] * 120, ["âœ” done"] * 40)
    over_states(ftfy.fix_text_batch, mojibake)
    # Lone surrogates take a second route that reads the list again.
    over_states(ftfy.fix_text_batch, (["ok \udce9 x"] * 70, ["bad \ud800"] * 30 + ["fine"]))


def test_translit() -> None:
    import pyroid.translit as tr

    over_states(tr.anyascii, (["Ελληνικά καλά"] * 100, ["Привет"] * 60))
    over_states(tr.anyascii, (["x \ud800"] * 50, ["ü"] * 20))  # the surrogate route
    over_states(tr.is_ascii, (["abc"] * 80, ["абв"] * 30))


def test_json_repair() -> None:
    import pyroid.json_repair as jr

    plain = (['{"a": 1,'] * 100, ["[1, 2"] * 40)
    over_states(jr.repair_json_batch, plain)
    pytest.importorskip("json_repair")
    # A document with a character the port hands to json_repair: the fallback reads the list too.
    odd = ['{"k": "\U0010f800"'] * 3 + ['{"a": 1,'] * 20
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        jr.repair_json_batch(odd)  # its one FallbackWarning, here, on one thread
    over_states(jr.repair_json_batch, (odd, ['{"b": 2'] * 7))


def test_trafilatura() -> None:
    import pyroid.trafilatura as T

    page = "<html><body><article><p>{}</p></article></body></html>"
    a = [page.format("First page text, long enough to be kept as the main content.")] * 12
    b = [page.format("Second page: another paragraph that trafilatura extracts.")] * 5
    over_states(T.extract_batch, (a, b))


@pytest.fixture(scope="module")
def ft_model(tmp_path_factory: pytest.TempPathFactory) -> Any:
    pytest.importorskip("numpy")
    import pyroid.fasttext as ft

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_fasttext import SPECS, build
    finally:
        sys.path.pop(0)
    path = tmp_path_factory.mktemp("mutation") / "model.bin"
    path.write_bytes(build(SPECS["softmax-dense"]).data)
    return ft.load_model(str(path))


def test_fasttext(ft_model: Any) -> None:
    def predict(d: list[str]) -> Any:
        labels, probs = ft_model.predict(d, k=1)
        return list(labels), [p.tolist() for p in probs]

    over_states(predict, (["the quick brown fox"] * 100, ["lazy dogs sleep"] * 40))
    # A newline in a document is an error, reported for the list as it was read.
    over_states(predict, (["a b"] * 30, ["a\nb"] * 10), expected=(ValueError,))


def test_filters() -> None:
    import pyroid.filters as F

    over_states(F.gopher_quality, VARIANTS)
    chain = [F.GopherRepetition(), F.C4Quality(), F.FineWebQuality()]
    over_states(lambda d: F.apply(d, chain), VARIANTS)
    # A config's word list, changed while the filter runs.
    words = ["the", "and", "of"]
    cfg = F.GopherQuality(min_stop_words=2, stop_words=words)
    word_states = (["the", "and", "of"], ["cat", "dog"])
    allowed = [F.gopher_quality(DOCS_A, F.GopherQuality(min_stop_words=2, stop_words=w)) for w in word_states]
    hammer(lambda: F.gopher_quality(DOCS_A, cfg), states(words, word_states), allowed)


def test_filters_pack_lists() -> None:
    # The private packer reads three lists; they used to be copied by a length read first.
    from pyroid._pyroid import filters as native

    _, shape = native._to_list(["x"] * 40)
    keep: list[Any] = [True] * 40
    reason: list[Any] = [None] * 40
    text: list[Any] = [None] * 40

    def change(i: int) -> None:
        n = 40 if i % 2 else 7
        keep[:] = [True] * n
        reason[:] = [None] * n
        text[:] = [None] * n

    hammer(
        lambda: native._pack(shape, "keep", keep, reason, text),
        change,
        expected=(ValueError,),  # lengths that differ from each other or from the shape's
    )


def test_dedup() -> None:
    import pyroid.dedup as D

    near = ["one two three four five six seven eight nine ten eleven twelve"] * 80
    other = [f"doc {i} has words of its own, nothing else is like it at all {i}" for i in range(50)]
    variants: tuple[list[str], ...] = (near + other, other, [])
    over_states(D.exact_dedup, variants)
    over_states(D.minhash_dedup, variants)
    over_states(D.simplify_text, variants)


def test_datasketch() -> None:
    np = pytest.importorskip("numpy")
    import pyroid.datasketch as ds

    tokens = ([b"a", b"b", b"c"] * 50, [b"x", b"y"] * 20)

    def update(d: list[bytes]) -> Any:
        m = ds.MinHash(num_perm=64)
        m.update_batch(d)
        return m.hashvalues

    over_states(update, tokens)
    # MinHash.bulk takes any iterable and goes through it as datasketch does (Python's iterator,
    # a list item by item): no one-step read to check, only that nothing breaks.
    docs: list[Any] = [[b"a", b"b"]] * 60
    hammer(
        lambda: ds.MinHash.bulk(docs, num_perm=32),
        states(docs, ([[b"a", b"b"]] * 60, [[b"c"], [b"d", b"e"]] * 10)),
    )
    over_states(lambda d: ds.minhash_batch(d, num_perm=32), (DOCS_A, DOCS_B))

    # One index used from every thread: queries while another thread inserts.
    lsh = ds.MinHashLSH(threshold=0.5, num_perm=64)
    sigs = ds.minhash_batch(["alpha beta gamma", "delta epsilon", "alpha beta"], num_perm=64)
    lsh.insert_batch(["k0", "k1", "k2"], sigs)
    batch = np.repeat(sigs[:1], 4, axis=0)
    hammer(lambda: lsh.query_batch(batch), lambda i: lsh.insert(f"b{i}", ds.MinHash(num_perm=64)))
    m = ds.MinHash(num_perm=64)
    m.update_batch([b"alpha", b"beta"])
    hammer(lambda: lsh.query(m), lambda i: lsh.insert(f"n{i}", m), expected=(ValueError,))


def test_jieba() -> None:
    import pyroid.jieba as jieba

    tok = jieba.Tokenizer()
    text = "我们在北京天安门广场看升旗"
    tok.lcut(text)  # build the dictionary first
    word = "看升旗"
    tok.add_word(word, 100000)
    with_word = tok.lcut(text)
    tok.del_word(word)
    without = tok.lcut(text)
    assert with_word != without

    def change(i: int) -> None:
        if i % 2:
            tok.add_word(word, 100000)
        else:
            tok.del_word(word)

    hammer(lambda: tok.lcut(text), change, [with_word, without])
    tok.del_word(word)
    over_states(lambda d: tok.lcut(d), (["我们在北京"] * 50, ["天安门广场"] * 20))


def test_nltk() -> None:
    import pyroid.nltk as nltk

    params = nltk.PunktParameters()
    tok = nltk.PunktSentenceTokenizer(params)
    text = "Dr. Smith went home. He slept."
    split = tok.tokenize(text)
    params.abbrev_types.add("dr")
    joined = tok.tokenize(text)
    assert split != joined

    def change(i: int) -> None:
        if i % 2:
            params.abbrev_types.add("dr")
        else:
            params.abbrev_types.discard("dr")

    hammer(lambda: tok.tokenize(text), change, [split, joined])
    # Whatever the threads cached meanwhile, the next call sees the parameters as they are now.
    params.abbrev_types.add("dr")
    assert tok.tokenize(text) == joined
    params.abbrev_types.discard("dr")
    assert tok.tokenize(text) == split
    batch = (["Hello, world!"] * 60, ["It's 3 p.m. now."] * 25)
    over_states(lambda d: nltk.word_tokenize(d, preserve_line=True), batch)
    over_states(tok.tokenize, batch)


def test_nemo() -> None:
    import pyroid.nemo as nemo

    over_states(lambda d: nemo.repeating_top_ngrams(d, 2), VARIANTS)
    scores = [0.5] * 30
    over_states(lambda langs: nemo.lang_id_strings(scores, langs), (["en"] * 30, ["fr"] * 30))
    patterns = ["drop me"]
    docs = ["keep\ndrop me\nkeep too", "drop me", "x\ny"] * 30
    pattern_states = (["drop me"], ["keep", "x"])
    allowed = [nemo.remove_lines(docs, list(p)) for p in pattern_states]
    hammer(lambda: nemo.remove_lines(docs, patterns), states(patterns, pattern_states), allowed)


def test_metrics() -> None:
    from pyroid.metrics import rouge_scorer, sacrebleu

    types = ["rouge1", "rougeL"]
    scorer = rouge_scorer.RougeScorer(types)
    type_states = (["rouge1", "rougeL"], ["rouge2"])
    target, prediction = "the cat sat on the mat", "the cat lay on a mat"
    allowed = [rouge_scorer.RougeScorer(list(t)).score(target, prediction) for t in type_states]
    hammer(lambda: scorer.score(target, prediction), states(types, type_states), allowed)

    refs = ["the cat sat on the mat"] * 40
    hyp_states = (["the cat sat on a mat"] * 40, ["a dog ran"] * 40)
    hyps = list(hyp_states[0])
    scores = [sacrebleu.corpus_bleu(list(h), [refs]).score for h in hyp_states]
    hammer(lambda: sacrebleu.corpus_bleu(hyps, [refs]).score, states(hyps, hyp_states), scores)


def test_warc() -> None:
    import pyroid.warc as warc

    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from test_warc import response, warcinfo
    finally:
        sys.path.pop(0)
    body = b"<html><body>" + b"x" * 200_000 + b"</body></html>"
    data = warcinfo() + response(body) + response(b"<html>two</html>")

    class Slow(io.BytesIO):
        """A file object whose read() lets other threads run (as a network or disk read does)."""

        def read(self, n: int | None = -1) -> bytes:
            time.sleep(0)
            return super().read(n)

    it = iter(warc.ArchiveIterator(Slow(data)))
    next(it)  # warcinfo
    rec = next(it)
    done = threading.Event()
    got: list[bytes] = []

    def read() -> None:
        while chunk := rec.raw_stream.read(100):
            got.append(chunk)
        done.set()

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    # Reading the record's `limit` while another thread reads the record used to deadlock on
    # GIL builds (the read held the record's lock across fh.read(), which released the GIL).
    def wait(i: int) -> None:
        done.wait(0.01)

    hammer(lambda: rec.raw_stream.limit, wait)
    reader.join(HANG)
    assert done.is_set() and not reader.is_alive()
    assert b"".join(got).endswith(body)

    # One iterator shared by every thread: each record comes out once.
    seen: list[Any] = []
    shared = iter(warc.ArchiveIterator(io.BytesIO(warcinfo() + response() * 200)))

    def take() -> None:
        for rec in shared:
            seen.append(rec.rec_headers.get_header("WARC-Type"))
            return

    hammer(take, lambda i: time.sleep(0.001))
    assert seen.count("response") == 200 and seen.count("warcinfo") == 1
