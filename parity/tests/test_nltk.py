"""NLTK parity: the ``--impl`` candidate vs live NLTK 3.10.4's tokenizers on real text.

    pytest tests/test_nltk.py                                  # self-check (candidate = nltk)
    pytest tests/test_nltk.py --impl pyroid.nltk               # pyroid vs NLTK
    pytest tests/test_nltk.py --impl pyroid.nltk --parity-limit 200

Every document is tokenized whole and compared with NLTK's outcome: the sentences, spans or
tokens, the detokenized str, or the exception's type and message.

Models: NLTK's ``punkt_tab`` (it has no licence, so it is never committed). The installed copy if
``nltk.data.find`` has it; else punkt_tab.zip, downloaded once into ``$PYROID_DATA/nltk_data``
(or a tmp dir) and checked against its sha256, goes first on ``nltk.data.path``. Offline, the
Punkt checks skip and say so; the Treebank checks need no model.

Corpora:
    fineweb          fineweb/fineweb-dev-10k.parquet (English)
    adversarial      parity/adversarial.parquet (8 languages, mojibake, controls, near-duplicates, edge cases)
    fineweb2-<code>  fineweb2/<code>.parquet for deu_Latn, fra_Latn, rus_Cyrl and spa_Latn (pinned)
    fw2-<code>       $PYROID_NLTK_FW2/<code>-test-5k.parquet for the other punkt_tab languages (FineWeb-2
                     test splits; not pinned, so skipped unless the variable names a directory with them)

Checks (suite ``nltk`` in parity-report.json):
    <corpus>-<lang>-sent           sent_tokenize(text, lang)
    <corpus>-<lang>-sent_noalign   PunktTokenizer(lang).tokenize(text, realign_boundaries=False)
    <corpus>-<lang>-spans          list(PunktTokenizer(lang).span_tokenize(text))
    <corpus>-<lang>-spans_noalign  the same with realign_boundaries=False
    <corpus>-<lang>-sents          PunktTokenizer(lang).tokenize_sents / span_tokenize_sents (BATCH docs)
    <corpus>-<lang>-word           word_tokenize(text, lang), in the corpus's own language
                                   Every punkt_tab language runs on fineweb and adversarial;
                                   fineweb2-<code> and fw2-<code> run in their own language.
    <corpus>-treebank[_ptb]        TreebankWordTokenizer().tokenize(text[, convert_parentheses=True])
    <corpus>-treebank_spans        list(TreebankWordTokenizer().span_tokenize(text))
    <corpus>-nltkword[_ptb|_spans] the same with NLTKWordTokenizer
    <corpus>-word_preserve         word_tokenize(text, preserve_line=True)
    <corpus>-tokenize_sents        tokenize_sents of both word tokenizers (BATCH docs a call)
    <corpus>-detok                 TreebankWordDetokenizer().detokenize(NLTK's NLTKWordTokenizer tokens)
    <corpus>-detok_ptb             detokenize(NLTK's TreebankWordTokenizer convert_parentheses tokens,
                                   convert_parentheses=True)
    <corpus>-align                 align_tokens(NLTK's NLTKWordTokenizer tokens, text)
    <corpus>-regexp[_spans]        RegexpTokenizer(pattern, gaps, discard_empty).tokenize / span_tokenize
                                   for every pattern pyroid runs natively and every gaps/discard_empty
    <corpus>-regexp_fallback       patterns and flags pyroid hands to NLTK (still NLTK's results)
    <corpus>-wordpunct, -blankline wordpunct_tokenize, blankline_tokenize
    <corpus>-whitespace            WhitespaceTokenizer().tokenize / span_tokenize
    <corpus>-regexp_tokenize       regexp_tokenize(text, pattern, gaps, discard_empty)
    <corpus>-regexp_sents          RegexpTokenizer.tokenize_sents (BATCH docs a call)
    batch-<check>                 the candidate's batch call (``sent_tokenize(list_of_docs)``, ...,
                                   BATCH docs a call) vs NLTK per document; skipped for a candidate
                                   that takes no list (NLTK itself)

A candidate runs on the first 500 documents of each corpus, unless ``--parity-limit`` says
otherwise (``--parity-limit 0``: every document, about 2 hours). Without ``--impl`` the candidate
is NLTK itself (the first 1000 documents of each corpus), so the checks only show that NLTK is
deterministic here.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import types
import urllib.request
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from pyroid_parity import data as D
from pyroid_parity import multilingual as ML
from pyroid_parity.plugin import _impl, effective_limit, need, need_corpus, oracle_pin, reference
from pyroid_parity.report import Collector

SUITE = "nltk"
NLTK_VERSION = "3.10.4"
#: NLTK runs its patterns with the third-party ``regex`` module; its Unicode tables are part of the oracle.
REGEX_VERSION = "2026.9.29"
#: Documents per batch call (and per tokenize_sents call).
BATCH = 1000
#: Documents per corpus for a candidate when ``--parity-limit`` is not given. Every document of
#: every corpus takes about 2 hours of NLTK time (14 M comparisons); ``--parity-limit 0`` runs them.
CANDIDATE_DOCS = 500


def _limit(config: pytest.Config, impl: Any) -> int:
    """``--parity-limit`` if given (0: every document), else ``CANDIDATE_DOCS`` for a candidate."""
    if config.getoption("--parity-limit") is None and not impl.is_reference:
        return CANDIDATE_DOCS
    return effective_limit(config, impl)


PUNKT_TAB_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"
PUNKT_TAB_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"

#: The punkt_tab models and their FineWeb-2 language codes (English: the FineWeb dev slice).
LANGUAGES: dict[str, str | None] = {
    "czech": "ces_Latn",
    "danish": "dan_Latn",
    "dutch": "nld_Latn",
    "english": None,
    "estonian": "ekk_Latn",
    "finnish": "fin_Latn",
    "french": "fra_Latn",
    "german": "deu_Latn",
    "greek": "ell_Grek",
    "italian": "ita_Latn",
    "malayalam": "mal_Mlym",
    "norwegian": "nob_Latn",
    "polish": "pol_Latn",
    "portuguese": "por_Latn",
    "russian": "rus_Cyrl",
    "slovene": "slv_Latn",
    "spanish": "spa_Latn",
    "swedish": "swe_Latn",
    "turkish": "tur_Latn",
}
#: The languages whose FineWeb-2 slices are pinned (bench/manifest.json).
PINNED = ("deu_Latn", "fra_Latn", "rus_Cyrl", "spa_Latn")
FW2_ENV = "PYROID_NLTK_FW2"
FW2 = tuple(c for c in LANGUAGES.values() if c is not None and c not in PINNED)

#: Each corpus's own language (word_tokenize runs in it).
OWN: dict[str, str] = {"fineweb": "english", "adversarial": "english"}
for _lang, _code in LANGUAGES.items():
    if _code is not None:
        OWN[f"fineweb2-{_code}" if _code in PINNED else f"fw2-{_code}"] = _lang

CORPORA = ("fineweb", "adversarial", *(f"fineweb2-{c}" for c in PINNED), *(f"fw2-{c}" for c in FW2))
PUNKT_CASES = [
    *(("fineweb", lang) for lang in LANGUAGES),
    *(("adversarial", lang) for lang in LANGUAGES),
    *((corpus, OWN[corpus]) for corpus in CORPORA[2:]),
]

try:
    import nltk
    import regex
    from nltk.tokenize.punkt import PunktTokenizer as _RefPunkt
    from nltk.tokenize.util import align_tokens as _ref_align
except ImportError as e:
    nltk = None
    _NO_ORACLE: str | None = f"NLTK (the oracle) is not installed here: {e}"
else:
    _WANT = (oracle_pin("nltk", NLTK_VERSION), oracle_pin("regex", REGEX_VERSION))
    _NO_ORACLE = (
        f"needs nltk {_WANT[0]} with regex {_WANT[1]}, found {nltk.__version__} with {regex.__version__}"
        if (nltk.__version__, regex.__version__) != _WANT
        else None
    )

# pyroid's "delegating to NLTK" notice says which path ran, not what the result is (impl.py).
pytestmark = (
    [pytest.mark.filterwarnings("ignore::pyroid.FallbackWarning")]
    if importlib.util.find_spec("pyroid")
    else []
)


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, ...]:
    """``("ok", result)``, generators listed, or ``("raise", type name, message)``."""
    try:
        r = fn(*args, **kwargs)
        if isinstance(r, types.GeneratorType):
            r = list(r)
        return ("ok", r)
    except Exception as e:  # noqa: BLE001  the exception is part of the outcome
        return ("raise", type(e).__name__, str(e))


def _focus(got: Any, want: Any) -> tuple[Any, Any]:
    """For the report: the first differing items of two ``("ok", list)`` outcomes, else both whole."""
    if got[0] == want[0] == "ok" and isinstance(got[1], list) and isinstance(want[1], list):
        g, w = got[1], want[1]
        i = next((i for i, (a, b) in enumerate(zip(g, w, strict=False)) if a != b), min(len(g), len(w)))
        return (f"[{i}:] of {len(g)}", g[i : i + 3]), (f"[{i}:] of {len(w)}", w[i : i + 3])
    return got, want


# ------------------------------------------------------------------------------------------------ models


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_english() -> bool:
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError:
        return False
    return True


def _expose(root: Path) -> None:
    """``root`` first on nltk.data.path and NLTK_DATA (what pyroid reads without NLTK)."""
    if str(root) not in nltk.data.path:
        nltk.data.path.insert(0, str(root))
    env = [p for p in os.environ.get("NLTK_DATA", "").split(os.pathsep) if p]
    if str(root) not in env:
        os.environ["NLTK_DATA"] = os.pathsep.join([str(root), *env])
    for name in ("pyroid.nltk", "nltk.tokenize"):
        cached = getattr(sys.modules.get(name), "_get_punkt_tokenizer", None)
        if cached is not None:
            cached.cache_clear()


def _install_punkt_tab(tmp: Path) -> str | None:
    """Make punkt_tab visible to NLTK (installed, cached or downloaded); None, or why it is not."""
    if _find_english():
        return None
    root = D.DATA / "nltk_data"
    try:
        (root / "tokenizers").mkdir(parents=True, exist_ok=True)
        writable = os.access(root / "tokenizers", os.W_OK)
    except OSError:
        writable = False
    if not writable:
        root = tmp
    dest = root / "tokenizers" / "punkt_tab.zip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file() or _sha256(dest) != PUNKT_TAB_SHA256:
        part = dest.with_name(f"{dest.name}.{os.getpid()}.part")
        try:
            with urllib.request.urlopen(PUNKT_TAB_URL, timeout=60) as r:
                part.write_bytes(r.read())
        except OSError as e:  # offline
            part.unlink(missing_ok=True)
            return f"punkt_tab is not installed and {PUNKT_TAB_URL} could not be downloaded: {e}"
        digest = _sha256(part)
        if digest != PUNKT_TAB_SHA256:
            part.unlink()
            pytest.fail(f"{PUNKT_TAB_URL} has sha256 {digest}, not the pinned {PUNKT_TAB_SHA256}")
        os.replace(part, dest)
    _expose(root)
    if not _find_english():
        pytest.fail(f"NLTK does not find the punkt_tab models in {dest}")
    return None


@pytest.fixture(scope="session")
def punkt_problem(tmp_path_factory: pytest.TempPathFactory) -> str | None:
    """None once punkt_tab is visible to NLTK and the candidate; else why the Punkt checks skip."""
    if _NO_ORACLE:
        return _NO_ORACLE
    return _install_punkt_tab(tmp_path_factory.mktemp("nltk_data"))


@pytest.fixture(scope="session")
def nltk_impl(request: pytest.FixtureRequest, punkt_problem: str | None) -> Any:
    # After punkt_problem: GenericNltk calls sent_tokenize to tell whether it takes a list.
    return _impl(request, SUITE)


@pytest.fixture(scope="module", autouse=True)
def _oracle() -> None:
    reference(SUITE, "nltk", nltk.__version__ if nltk is not None else None, None)


def oracle(parity: Collector, checks: list[str], punkt: str | None = None) -> None:
    """Skip (and record why) if NLTK, or for Punkt checks its models, cannot run here."""
    why = _NO_ORACLE or punkt
    if why:
        for check in checks:
            parity.skip(SUITE, check, why)
        pytest.skip(why)


# ------------------------------------------------------------------------------------------------ corpora


def _fw2_path(code: str) -> Path | None:
    d = os.environ.get(FW2_ENV)
    p = Path(d) / f"{code}-test-5k.parquet" if d else None
    return p if p is not None and p.is_file() else None


@cache
def _load(corpus: str) -> tuple[list[str], list[str]]:
    if corpus == "fineweb":
        c = need_corpus("dev")
    elif corpus == "adversarial":
        c = need_corpus("adversarial")
    elif corpus.startswith("fineweb2-"):
        code = corpus.removeprefix("fineweb2-")
        need(ML.slice_rel(code))
        c = ML.fineweb2(code)
    else:
        code = corpus.removeprefix("fw2-")
        p = _fw2_path(code)
        assert p is not None
        cols = D.read_parquet_columns(p, ["id", "text"])
        return [f"{code}:{i}" for i in cols["id"]], cols["text"]
    return list(c.ids), list(c.texts)


def _docs(parity: Collector, corpus: str, limit: int, checks: list[str]) -> tuple[list[str], list[str]]:
    """``(ids, texts)`` of a corpus, the first ``limit`` documents (0: all)."""
    code = corpus.removeprefix("fw2-")
    if corpus.startswith("fw2-") and _fw2_path(code) is None:
        why = f"{corpus} is not pinned: set {FW2_ENV} to a directory with {code}-test-5k.parquet"
        for check in checks:
            parity.skip(SUITE, check, why)
        pytest.skip(why)
    ids, texts = _load(corpus)
    return (ids[:limit], texts[:limit]) if limit else (ids, texts)


def _uses(parity: Collector, corpus: str) -> None:
    rel = {"fineweb": D.CORPORA["dev"], "adversarial": D.CORPORA["adversarial"]}.get(corpus)
    if corpus.startswith("fineweb2-"):
        rel = ML.slice_rel(corpus.removeprefix("fineweb2-"))
    if rel is not None:  # the fw2-* files are not pinned
        parity.uses(SUITE, rel)


def _chunks(n: int) -> Iterator[range]:
    for lo in range(0, n, BATCH):
        yield range(lo, min(lo + BATCH, n))


class _Tally:
    """Records per-document comparisons; remembers the checks with a difference."""

    def __init__(self, parity: Collector, prefix: str) -> None:
        self.parity, self.prefix = parity, prefix
        self.bad: dict[str, list[str]] = {}

    def __call__(self, check: str, case: str, text: str, got: Any, want: Any) -> None:
        name = (
            f"{self.prefix}-{check}" if not check.startswith("batch-") else f"batch-{self.prefix}-{check[6:]}"
        )
        same = got == want
        g, w = (got, want) if same else _focus(got, want)
        if not self.parity.record(SUITE, name, same, case=case, input=text, got=g, want=w):
            self.bad.setdefault(name, []).append(case)

    def assert_clean(self, n: int) -> None:
        summary = "; ".join(f"{c}: {len(v)} of {n} differ, e.g. {v[:3]}" for c, v in self.bad.items())
        assert not self.bad, summary


def _takes_lists(f: Callable[[Any], Any], one: Callable[[str], Any]) -> bool:
    """Whether ``f(list_of_str)`` gives one result per document (``one(doc)`` each)."""
    try:
        return f(["A b. C d.", ""]) == [one("A b. C d."), one("")]
    except Exception:  # noqa: BLE001  NLTK's tokenizers raise for a list
        return False


def _align_tokens(mod: Any) -> Callable[..., Any] | None:
    f = getattr(mod, "align_tokens", None)
    if f is None:  # NLTK keeps it in nltk.tokenize.util
        for name in (f"{mod.__name__}.util", f"{mod.__name__}.tokenize.util"):
            f = getattr(sys.modules.get(name), "align_tokens", None) or f
    return f


# ------------------------------------------------------------------------------------------------ checks


def test_oracle_is_pinned_nltk(parity: Collector) -> None:
    oracle(parity, ["oracle-version"])
    got = f"nltk {nltk.__version__}, regex {regex.__version__}"
    assert parity.record(SUITE, "oracle-version", True, got=got)


def _slices(text: str, spans: tuple[Any, ...]) -> tuple[Any, ...]:
    """NLTK's ``sentences_from_text``: the text of each span."""
    return ("ok", [text[s:e] for s, e in spans[1]]) if spans[0] == "ok" else spans


@pytest.mark.parametrize(("corpus", "lang"), PUNKT_CASES, ids=[f"{c}-{lang}" for c, lang in PUNKT_CASES])
def test_punkt(
    nltk_impl: Any,
    parity: Collector,
    punkt_problem: str | None,
    request: pytest.FixtureRequest,
    corpus: str,
    lang: str,
) -> None:
    """Per document, then tokenize_sents and (for a candidate with batch input) batch calls."""
    own = OWN[corpus] == lang
    modes = ["sent", "sent_noalign", "spans", "spans_noalign", "sents"] + (["word"] if own else [])
    batch = [m for m in modes if m != "sents"]
    prefix = f"{corpus}-{lang}"
    checks = [f"{prefix}-{m}" for m in modes]
    oracle(parity, checks, punkt_problem)
    ids, texts = _docs(parity, corpus, _limit(request.config, nltk_impl), checks)
    _uses(parity, corpus)
    mod = nltk_impl.obj.mod
    ref, cand = _RefPunkt(lang), mod.PunktTokenizer(lang)
    has_batch = nltk_impl.obj.has_batch
    if not has_batch:
        for m in batch:
            parity.skip(SUITE, f"batch-{prefix}-{m}", f"{nltk_impl.module}.sent_tokenize takes no list")
    tally = _Tally(parity, prefix)
    candidate: dict[str, Callable[[str], Any]] = {
        "sent": lambda t: mod.sent_tokenize(t, lang),
        "sent_noalign": lambda t: cand.tokenize(t, realign_boundaries=False),
        "spans": lambda t: list(cand.span_tokenize(t)),
        "spans_noalign": lambda t: list(cand.span_tokenize(t, realign_boundaries=False)),
        "word": lambda t: mod.word_tokenize(t, lang),
    }
    batched: dict[str, Callable[[list[str]], Any]] = {
        "sent": lambda ts: mod.sent_tokenize(ts, lang),
        "sent_noalign": lambda ts: cand.tokenize(ts, realign_boundaries=False),
        "spans": lambda ts: cand.span_tokenize(ts),
        "spans_noalign": lambda ts: cand.span_tokenize(ts, realign_boundaries=False),
        "word": lambda ts: mod.word_tokenize(ts, lang),
    }
    for chunk in _chunks(len(texts)):
        wants: dict[str, list[tuple[Any, ...]]] = {m: [] for m in batch}
        for i in chunk:
            t = texts[i]
            spans = outcome(ref.span_tokenize, t)
            spans0 = outcome(ref.span_tokenize, t, realign_boundaries=False)
            want = {"sent": _slices(t, spans), "sent_noalign": _slices(t, spans0), "spans": spans}
            want["spans_noalign"] = spans0
            if own:
                want["word"] = outcome(nltk.word_tokenize, t, lang)
            for m, w in want.items():
                wants[m].append(w)
                tally(m, ids[i], t, outcome(candidate[m], t), w)
        docs, chunk_ids = [texts[i] for i in chunk], [ids[i] for i in chunk]
        # tokenize_sents / span_tokenize_sents: NLTK's own list API, for every candidate.
        _compare_batch(tally, "sents", cand.tokenize_sents, docs, chunk_ids, wants["sent"])
        _compare_batch(tally, "sents", cand.span_tokenize_sents, docs, chunk_ids, wants["spans"])
        if has_batch:
            for m in batch:
                _compare_batch(tally, f"batch-{m}", batched[m], docs, chunk_ids, wants[m])
    tally.assert_clean(len(ids))


def _compare_batch(
    tally: _Tally,
    check: str,
    f: Callable[[list[str]], Any],
    docs: list[str],
    ids: list[str],
    wants: list[tuple[Any, ...]],
) -> None:
    """One call of ``f`` over the documents whose reference did not raise, compared per document."""
    keep = [k for k, w in enumerate(wants) if w[0] == "ok"]
    got = outcome(f, [docs[k] for k in keep])
    if got[0] != "ok" or len(got[1]) != len(keep):
        tally(check, f"{ids[0]}..{ids[-1]}", "", got, ("ok", f"{len(keep)} results"))
        return
    for k, g in zip(keep, got[1], strict=True):
        g = list(g) if isinstance(g, types.GeneratorType) else g
        tally(check, ids[k], docs[k], ("ok", g), wants[k])


TREEBANK_MODES = (
    "treebank",
    "treebank_ptb",
    "treebank_spans",
    "nltkword",
    "nltkword_ptb",
    "nltkword_spans",
    "word_preserve",
    "detok",
    "detok_ptb",
    "align",
)
TREEBANK_BATCH = ("treebank", "treebank_ptb", "nltkword", "nltkword_ptb", "word_preserve")


@pytest.mark.parametrize("corpus", CORPORA)
def test_treebank(nltk_impl: Any, parity: Collector, request: pytest.FixtureRequest, corpus: str) -> None:
    """Both word tokenizers, the detokenizer and align_tokens, per document, then in batches."""
    checks = [f"{corpus}-{m}" for m in (*TREEBANK_MODES, "tokenize_sents")]
    oracle(parity, checks)
    ids, texts = _docs(parity, corpus, _limit(request.config, nltk_impl), checks)
    _uses(parity, corpus)
    mod = nltk_impl.obj.mod
    tb, nw, detok = mod.TreebankWordTokenizer(), mod.NLTKWordTokenizer(), mod.TreebankWordDetokenizer()
    ref_tb, ref_nw = nltk.TreebankWordTokenizer(), nltk.NLTKWordTokenizer()
    ref_detok = nltk.TreebankWordDetokenizer()
    align = _align_tokens(mod)
    if align is None:
        parity.skip(SUITE, f"{corpus}-align", f"{nltk_impl.module} has no align_tokens")
    has_batch = _takes_lists(tb.tokenize, tb.tokenize)
    if not has_batch:
        for m in TREEBANK_BATCH:
            parity.skip(
                SUITE, f"batch-{corpus}-{m}", f"{nltk_impl.module}.TreebankWordTokenizer takes no list"
            )
    batched: dict[str, Callable[[list[str]], Any]] = {
        "treebank": lambda ts: tb.tokenize(ts),
        "treebank_ptb": lambda ts: tb.tokenize(ts, convert_parentheses=True),
        "nltkword": lambda ts: nw.tokenize(ts),
        "nltkword_ptb": lambda ts: nw.tokenize(ts, convert_parentheses=True),
        "word_preserve": lambda ts: mod.word_tokenize(ts, preserve_line=True),
    }
    tally = _Tally(parity, corpus)
    for chunk in _chunks(len(texts)):
        wants: dict[str, list[tuple[Any, ...]]] = {m: [] for m in TREEBANK_BATCH}
        for i in chunk:
            t, doc = texts[i], ids[i]
            toks, ptb = outcome(ref_nw.tokenize, t), outcome(ref_tb.tokenize, t, convert_parentheses=True)
            want = {
                "treebank": outcome(ref_tb.tokenize, t),
                "treebank_ptb": ptb,
                "nltkword": toks,
                "nltkword_ptb": outcome(ref_nw.tokenize, t, convert_parentheses=True),
                "word_preserve": toks,  # word_tokenize(preserve_line=True) is NLTKWordTokenizer's tokenize
                "treebank_spans": outcome(ref_tb.span_tokenize, t),
                "nltkword_spans": outcome(ref_nw.span_tokenize, t),
            }
            for m in TREEBANK_BATCH:
                wants[m].append(want[m])
            got = {
                "treebank": outcome(tb.tokenize, t),
                "treebank_ptb": outcome(tb.tokenize, t, convert_parentheses=True),
                "nltkword": outcome(nw.tokenize, t),
                "nltkword_ptb": outcome(nw.tokenize, t, convert_parentheses=True),
                "word_preserve": outcome(mod.word_tokenize, t, preserve_line=True),
                "treebank_spans": outcome(tb.span_tokenize, t),
                "nltkword_spans": outcome(nw.span_tokenize, t),
            }
            if toks[0] == "ok":
                want["detok"] = outcome(ref_detok.detokenize, toks[1])
                got["detok"] = outcome(detok.detokenize, list(toks[1]))
                if align is not None:
                    want["align"] = outcome(_ref_align, toks[1], t)
                    got["align"] = outcome(align, list(toks[1]), t)
            if ptb[0] == "ok":
                want["detok_ptb"] = outcome(ref_detok.detokenize, ptb[1], convert_parentheses=True)
                got["detok_ptb"] = outcome(detok.detokenize, list(ptb[1]), convert_parentheses=True)
            for m, w in want.items():
                tally(m, doc, t, got[m], w)
        docs, chunk_ids = [texts[i] for i in chunk], [ids[i] for i in chunk]
        _compare_batch(tally, "tokenize_sents", tb.tokenize_sents, docs, chunk_ids, wants["treebank"])
        _compare_batch(tally, "tokenize_sents", nw.tokenize_sents, docs, chunk_ids, wants["nltkword"])
        if has_batch:
            for m in TREEBANK_BATCH:
                _compare_batch(tally, f"batch-{m}", batched[m], docs, chunk_ids, wants[m])
    tally.assert_clean(len(ids))


#: The patterns pyroid's RegexpTokenizer runs natively, and every gaps/discard_empty use of them
#: (discard_empty only matters with gaps).
REGEXP_PATTERNS = (
    r"\w+",
    r"\w+|[^\w\s]+",
    r"\w+|\$[\d\.]+|\S+",
    r"[\w']+",
    r"\S+",
    r"\s+",
    r"[a-zA-Z]+",
    r"(?:(?<!\s)\s*)?\n\s*\n\s*",
)
REGEXP_USES = ((False, True), (True, True), (True, False))
#: Patterns and flags that run in NLTK (pyroid's fallback): the results must still be NLTK's.
REGEXP_FALLBACKS = [
    (r"[A-Z][a-z]+|\d+(?:[.,]\d+)*", False, True, None),
    (r"[,\.\?!\"]\s*", True, True, None),
    (r"\w+", False, True, 2),  # re.IGNORECASE
    (r"\b", True, False, None),
]
REGEXP_MISC = ("regexp_fallback", "wordpunct", "blankline", "whitespace", "regexp_tokenize")


def _label(pattern: str, gaps: bool, discard: bool) -> str:
    return f"{pattern!r}{'/gaps' if gaps else ''}{'' if discard else '/keep-empty'}"


@pytest.mark.parametrize("pattern", REGEXP_PATTERNS)
@pytest.mark.parametrize("corpus", CORPORA)
def test_regexp(
    nltk_impl: Any, parity: Collector, request: pytest.FixtureRequest, corpus: str, pattern: str
) -> None:
    """RegexpTokenizer(pattern, gaps, discard_empty) per document, tokenize_sents, batches.

    One test per pattern: NLTK takes minutes over a whole corpus, and pytest-timeout allows
    600 s a test. The checks are shared by all patterns (``<corpus>-regexp``, ...).
    """
    checks = [f"{corpus}-{m}" for m in ("regexp", "regexp_spans", "regexp_sents")]
    oracle(parity, checks)
    ids, texts = _docs(parity, corpus, _limit(request.config, nltk_impl), checks)
    _uses(parity, corpus)
    mod = nltk_impl.obj.mod
    ref_mod = importlib.import_module("nltk.tokenize.regexp")
    uses = [(pattern, g, d) for g, d in REGEXP_USES]
    ours = [mod.RegexpTokenizer(*u) for u in uses]
    refs = [ref_mod.RegexpTokenizer(*u) for u in uses]
    has_batch = _takes_lists(ours[0].tokenize, ours[0].tokenize)
    if not has_batch:
        parity.skip(SUITE, f"batch-{corpus}-regexp", f"{nltk_impl.module}.RegexpTokenizer takes no list")
    tally = _Tally(parity, corpus)
    for chunk in _chunks(len(texts)):
        wants: list[list[tuple[Any, ...]]] = [[] for _ in uses]
        for i in chunk:
            t = texts[i]
            for k, (tok, ref) in enumerate(zip(ours, refs, strict=True)):
                case = f"{ids[i]}:{_label(*uses[k])}"
                want = outcome(ref.tokenize, t)
                wants[k].append(want)
                tally("regexp", case, t, outcome(tok.tokenize, t), want)
                tally("regexp_spans", case, t, outcome(tok.span_tokenize, t), outcome(ref.span_tokenize, t))
        docs = [texts[i] for i in chunk]
        for k, tok in enumerate(ours):
            labelled = [f"{ids[i]}:{_label(*uses[k])}" for i in chunk]
            _compare_batch(tally, "regexp_sents", tok.tokenize_sents, docs, labelled, wants[k])
            if has_batch:
                _compare_batch(tally, "batch-regexp", tok.tokenize, docs, labelled, wants[k])
    tally.assert_clean(len(ids))


@pytest.mark.parametrize("corpus", CORPORA)
def test_regexp_misc(nltk_impl: Any, parity: Collector, request: pytest.FixtureRequest, corpus: str) -> None:
    """The subclasses and functions, regexp_tokenize, and patterns pyroid hands to NLTK."""
    checks = [f"{corpus}-{m}" for m in REGEXP_MISC]
    oracle(parity, checks)
    ids, texts = _docs(parity, corpus, _limit(request.config, nltk_impl), checks)
    _uses(parity, corpus)
    mod = nltk_impl.obj.mod
    ref_mod = importlib.import_module("nltk.tokenize.regexp")

    def make(m: Any, p: str, g: bool, d: bool, f: int | None) -> Any:
        return m.RegexpTokenizer(p, g, d) if f is None else m.RegexpTokenizer(p, g, d, f)

    fallbacks = [(make(mod, *c), make(ref_mod, *c)) for c in REGEXP_FALLBACKS]
    ws, ref_ws = mod.WhitespaceTokenizer(), ref_mod.WhitespaceTokenizer()
    tally = _Tally(parity, corpus)
    for doc, t in zip(ids, texts, strict=True):
        for k, (tok, ref) in enumerate(fallbacks):
            case = f"{doc}:{REGEXP_FALLBACKS[k][0]!r}"
            tally("regexp_fallback", case, t, outcome(tok.tokenize, t), outcome(ref.tokenize, t))
            tally("regexp_fallback", case, t, outcome(tok.span_tokenize, t), outcome(ref.span_tokenize, t))
        tally("wordpunct", doc, t, outcome(mod.wordpunct_tokenize, t), outcome(ref_mod.wordpunct_tokenize, t))
        tally("blankline", doc, t, outcome(mod.blankline_tokenize, t), outcome(ref_mod.blankline_tokenize, t))
        tally("whitespace", doc, t, outcome(ws.tokenize, t), outcome(ref_ws.tokenize, t))
        tally("whitespace", doc, t, outcome(ws.span_tokenize, t), outcome(ref_ws.span_tokenize, t))
        for p, (g, d) in zip(REGEXP_PATTERNS[1::3], REGEXP_USES, strict=True):
            tally(
                "regexp_tokenize",
                f"{doc}:{p!r}",
                t,
                outcome(mod.regexp_tokenize, t, p, g, d),
                outcome(ref_mod.regexp_tokenize, t, p, g, d),
            )
    tally.assert_clean(len(ids))
