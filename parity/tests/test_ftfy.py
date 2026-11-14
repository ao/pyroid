"""ftfy parity: the ``--impl`` candidate vs live ftfy 6.3.1 and its pinned outputs.

    pytest tests/test_ftfy.py                                   # self-check (candidate = ftfy)
    pytest tests/test_ftfy.py --impl pyroid.ftfy                # pyroid vs ftfy
    HYPOTHESIS_PROFILE=ci pytest tests/test_ftfy.py --impl pyroid.ftfy

Checks (suite ``ftfy`` in parity-report.json):
    golden-fix_text        fix_text(original) == live ftfy on all v6.3.1 cases, ftfy's known failures included
    golden-fix_encoding    fix_encoding(original) == the case's "fixed-encoding" (else "fixed"), "pass" cases
    golden-extra-layer     the original garbled once more (UTF-8 read as latin-1), as ftfy's own tests do
    hypothesis-*           generated input: outcome == live ftfy's, where an outcome is ("ok", result) or
                           ("raise", ExceptionTypeName); exception types are part of parity
    batch                  fix_text_batch(texts, **kw) == [ftfy.fix_text(t, **kw) for t in texts]
    errors                 bad arguments: same outcome and same warning categories, except that a candidate's
                           pyroid.FallbackWarning (delegation notice, once per process) is not compared
    corpus-<name>          fix_text(doc) == pinned ftfy.fix_text(doc) (parity/ftfy-ref.parquet), all docs
    api-signature, api-config   signatures and TextFixerConfig defaults mirror ftfy's (non-reference only)
"""

from __future__ import annotations

import importlib.metadata as md
import inspect
import json
import warnings
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import ftfy
import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from pyroid_parity import data as D
from pyroid_parity import strategies as S
from pyroid_parity.impl import warning_categories
from pyroid_parity.plugin import need, need_corpus, oracle_pin, reference
from pyroid_parity.report import Collector

FTFY_VERSION = "6.3.1"
SUITE = "ftfy"
GOLDEN = Path(__file__).resolve().parents[1] / "data" / f"ftfy-{FTFY_VERSION}" / "test_cases.json"
CASES: list[dict[str, str]] = json.loads(GOLDEN.read_text("utf-8"))
IDS = [
    f"{i:03d}-{c['label']}" for i, c in enumerate(CASES)
]  # labels repeat; strict mode rejects duplicate ids
PASSING = [(case, i) for case, i in zip(CASES, IDS, strict=True) if case["expect"] == "pass"]
REF = "parity/ftfy-ref.parquet"
CHUNK = 256

Outcome = tuple[str, Any]


def outcome(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Outcome:
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as e:  # noqa: BLE001  exception types are part of parity
        return ("raise", type(e).__name__)


def check(
    parity: Collector, name: str, got: Any, want: Any, *, case: str | None = None, input: Any = None
) -> None:
    same = parity.record(SUITE, name, got == want, case=case, input=input, got=got, want=want)
    assert same, f"{name} {case or ''}\ninput: {input!r}\n  got: {got!r}\n want: {want!r}"


@pytest.fixture(scope="module", autouse=True)
def _oracle() -> None:
    reference(SUITE, "ftfy", md.version("ftfy"), None)


@pytest.fixture(scope="module")
def candidate_mod(ftfy_impl: Any) -> Any:
    mod = getattr(ftfy_impl.obj, "mod", None)
    if mod is None:
        pytest.skip(f"adapter {ftfy_impl.adapter} exposes no module")
    return mod


def test_live_oracle_is_pinned_ftfy() -> None:
    assert md.version("ftfy") == oracle_pin("ftfy", FTFY_VERSION)


# ------------------------------------------------------------------------------------------- golden corpus


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_golden_fix_text(ftfy_impl: Any, parity: Collector, case: dict[str, str]) -> None:
    orig = case["original"]
    check(
        parity,
        "golden-fix_text",
        outcome(ftfy_impl.obj.fix_text, orig),
        outcome(ftfy.fix_text, orig),
        case=case["label"],
        input=orig,
    )


@pytest.mark.parametrize(("case", "cid"), PASSING, ids=[cid for _, cid in PASSING])
def test_golden_fix_encoding(ftfy_impl: Any, parity: Collector, case: dict[str, str], cid: str) -> None:
    orig = case["original"]
    want = ("ok", case.get("fixed-encoding", case["fixed"]))
    check(
        parity, "golden-fix_encoding", outcome(ftfy_impl.obj.fix_encoding, orig), want, case=cid, input=orig
    )


@pytest.mark.parametrize(("case", "cid"), PASSING, ids=[cid for _, cid in PASSING])
def test_golden_extra_layer(ftfy_impl: Any, parity: Collector, case: dict[str, str], cid: str) -> None:
    extra_bad = case["original"].encode("utf-8").decode("latin-1")
    check(
        parity,
        "golden-extra-layer",
        outcome(ftfy_impl.obj.fix_text, extra_bad),
        outcome(ftfy.fix_text, extra_bad),
        case=cid,
        input=extra_bad,
    )


# ------------------------------------------------------------------------------------- Hypothesis differential

VOILA = "voil\xc3  le travail"  # restore_byte_a0: the NBSP after "\xc3" was turned into a space
PAIR = S.utf16_pair("\N{GRINNING FACE}")


@given(text=S.documents())
@example(text=VOILA)
@example(text="voil\xc3\xa0 le travail")
@example(text=chr(0xDC00))  # lone low surrogate -> U+FFFD
@example(text=PAIR)  # UTF-16 pair -> the astral character
@example(text=PAIR[::-1])  # reversed pair -> two U+FFFD
@example(text="a < b\n&eacute; &EACUTE;")  # unescape_html="auto": off from the first '<' onwards
@example(text="&eacute;\na < b\n&eacute;")
@example(
    text="\x92 \x85 \r\n \N{ZERO WIDTH NO-BREAK SPACE}\x00 \x1b[31mred\x1b[0m \N{LATIN SMALL LIGATURE FI}"
)
@example(text="")
def test_fix_text(ftfy_impl: Any, parity: Collector, text: str) -> None:
    check(
        parity,
        "hypothesis-fix_text",
        outcome(ftfy_impl.obj.fix_text, text),
        outcome(ftfy.fix_text, text),
        input=text,
    )


@given(text=S.documents(), kwargs=S.configs())
@example(
    text=VOILA, kwargs={"restore_byte_a0": False}
)  # ignored: fix_text drops the config for fix_encoding...
@example(text=VOILA, kwargs={"restore_byte_a0": False, "explain": True})  # ...unless explain=True
@example(text=PAIR, kwargs={"max_decode_length": 1})  # segments split the pair
@example(text=chr(0xDC00), kwargs={"fix_surrogates": False})
@example(
    text="\x1b[31mred\x1b[0m <b>&EACUTE;</b>",
    kwargs={"unescape_html": True, "remove_terminal_escapes": False},
)
@example(
    text="e\N{COMBINING ACUTE ACCENT}\N{ANGSTROM SIGN}\N{ROMAN NUMERAL TWELVE}",
    kwargs={"normalization": "NFKD"},
)
def test_fix_text_kwargs(ftfy_impl: Any, parity: Collector, text: str, kwargs: dict[str, Any]) -> None:
    check(
        parity,
        "hypothesis-fix_text-kwargs",
        outcome(ftfy_impl.obj.fix_text, text, **kwargs),
        outcome(ftfy.fix_text, text, **kwargs),
        input=(text, kwargs),
    )


@given(text=S.documents(), kwargs=S.configs())
@example(text=VOILA, kwargs={})
@example(text=VOILA, kwargs={"restore_byte_a0": False})
@example(text=chr(0xDC00), kwargs={})  # fix_encoding leaves surrogates alone
@example(text=S.garble(PAIR, [S.CP1252]), kwargs={})  # CESU-8 mojibake -> utf-8-variants
@example(text="\xc3\xa9", kwargs={"fix_encoding": False})
def test_fix_encoding(ftfy_impl: Any, parity: Collector, text: str, kwargs: dict[str, Any]) -> None:
    check(
        parity,
        "hypothesis-fix_encoding",
        outcome(ftfy_impl.obj.fix_encoding, text, **kwargs),
        outcome(ftfy.fix_encoding, text, **kwargs),
        input=(text, kwargs),
    )


@given(text=S.documents(), kwargs=S.configs())
@example(text=VOILA, kwargs={"restore_byte_a0": False})  # an explicit config keeps explain=True
def test_fix_text_config_object(
    candidate_mod: Any, parity: Collector, text: str, kwargs: dict[str, Any]
) -> None:
    config_cls = getattr(candidate_mod, "TextFixerConfig", None)
    if config_cls is None:
        pytest.skip("candidate exports no TextFixerConfig")
    got = outcome(lambda: candidate_mod.fix_text(text, config_cls(**kwargs)))
    want = outcome(lambda: ftfy.fix_text(text, ftfy.TextFixerConfig(**kwargs)))
    check(parity, "hypothesis-config-object", got, want, input=(text, kwargs))


@given(batch=st.lists(S.texts(), max_size=16), kwargs=S.configs())
@example(batch=[], kwargs={})
@example(batch=["a < b", "&eacute;"], kwargs={})  # "auto" unescaping must not leak across documents
@example(batch=[chr(0xD800), PAIR, "x"], kwargs={})
def test_batch(ftfy_impl: Any, parity: Collector, batch: list[str], kwargs: dict[str, Any]) -> None:
    got = outcome(ftfy_impl.obj.fix_text_batch, batch, **kwargs)
    want = outcome(lambda: [ftfy.fix_text(t, **kwargs) for t in batch])
    check(parity, "batch", got, want, input=(batch, kwargs))


# ---------------------------------------------------------------------------------------------- bad arguments

BAD_CALLS: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = [
    ("fix_text", (b"caf\xc3\xa9",), {}),  # UnicodeError
    ("fix_encoding", (b"caf\xc3\xa9",), {}),
    ("fix_text", (None,), {}),  # TypeError
    ("fix_encoding", (None,), {}),
    ("fix_text", (42,), {}),
    ("fix_text", ("x", {}), {}),  # AttributeError: a dict is not a TextFixerConfig
    (
        "fix_text",
        ("x",),
        {"no_such_option": True},
    ),  # namedtuple._replace: TypeError on 3.13+, ValueError on 3.12
    ("fix_encoding", ("x",), {"no_such_option": True}),
    ("fix_text", ("x",), {"normalization": "NFX"}),  # ValueError from unicodedata.normalize
    ("fix_encoding", ("x",), {"normalization": "NFX"}),  # ok: fix_encoding never normalizes
    ("fix_text", ("x",), {"normalization": 1}),  # TypeError
    ("fix_text", ("xyz",), {"max_decode_length": 1.5}),  # TypeError: slice indices must be integers
    ("fix_text", ("&eacute;",), {"fix_entities": True}),  # DeprecationWarning, then acts as unescape_html
    ("fix_text", ("&eacute;",), {"unescape_html": "sometimes"}),  # any truthy value unescapes
]


def outcome_and_warnings(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Outcome, list[str]]:
    """The outcome and the warning categories parity compares: all but pyroid.FallbackWarning (see impl)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = outcome(fn, *args, **kwargs)
    return out, warning_categories(caught)


@pytest.mark.parametrize(
    ("name", "args", "kwargs"), BAD_CALLS, ids=[f"{i:02d}-{c[0]}" for i, c in enumerate(BAD_CALLS)]
)
def test_bad_arguments(
    ftfy_impl: Any, parity: Collector, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> None:
    got = outcome_and_warnings(getattr(ftfy_impl.obj, name), *args, **kwargs)
    want = outcome_and_warnings(getattr(ftfy, name), *args, **kwargs)
    check(parity, "errors", got, want, case=name, input=(args, kwargs))


# ----------------------------------------------------------------------------------------------------- corpora


@cache
def _pinned_fixed(path: Path) -> dict[str, str]:
    cols = D.read_parquet_columns(path, ["id", "fixed"])
    out = dict(zip(cols["id"], cols["fixed"], strict=True))
    if len(out) != len(cols["id"]):
        raise D.DataError(f"{REF}: ids are not unique")
    return out


def _window(got: Outcome, want: Outcome, before: int = 40, after: int = 80) -> tuple[Any, Any, int]:
    """Shrink a long-document diff to the region around the first differing character."""
    if got[0] != "ok" or not isinstance(got[1], str):
        return got, want, 0
    a, b = got[1], want[1]
    pos = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), min(len(a), len(b)))
    lo = max(0, pos - before)
    return a[lo : pos + after], b[lo : pos + after], pos


def _candidate_outcomes(obj: Any, parity: Collector, label: str, texts: list[str]) -> list[Outcome]:
    """fix_text_batch on a chunk; if it raises (or returns the wrong length), fall back to fix_text per doc."""
    batch = outcome(obj.fix_text_batch, texts)
    if batch[0] == "ok" and len(batch[1]) == len(texts):
        return [("ok", r) for r in batch[1]]
    per_doc = [outcome(obj.fix_text, t) for t in texts]
    if all(o[0] == "ok" for o in per_doc):  # the batch API failed where fix_text did not
        got = batch if batch[0] == "raise" else ("ok", f"{len(batch[1])} results")
        parity.record(
            SUITE, "batch", False, case=label, input=f"{len(texts)} docs", got=got, want="per-doc results"
        )
    return per_doc


@pytest.mark.parametrize("name", ["dev", "adversarial"])
def test_corpus(ftfy_impl: Any, parity: Collector, parity_limit: int, name: str) -> None:
    ref_path = need(REF)
    corpus = need_corpus(name, parity_limit)
    entry = D.load_manifest()["files"][REF]
    assert name in entry.get("corpora", [name]), f"{REF} was not generated for corpus {name!r}"
    ref_version = entry.get("oracle", {}).get("ftfy")
    reference(SUITE, "ftfy", ref_version, None)
    parity.uses(SUITE, REF, D.CORPORA[name])
    assert ref_version == FTFY_VERSION, f"{REF} was generated by ftfy {ref_version}, not {FTFY_VERSION}"
    pinned = _pinned_fixed(ref_path)
    missing = [i for i in corpus.ids if i not in pinned]
    assert not missing, f"{REF} lacks {len(missing)} {name} ids, e.g. {missing[:5]}"

    check_name, diffs = f"corpus-{name}", []
    for start in range(0, len(corpus), CHUNK):
        ids, texts = corpus.ids[start : start + CHUNK], corpus.texts[start : start + CHUNK]
        got_all = _candidate_outcomes(ftfy_impl.obj, parity, f"{name}[{start}:{start + len(ids)}]", texts)
        for doc_id, text, got in zip(ids, texts, got_all, strict=True):
            want = ("ok", pinned[doc_id])
            if got == want:
                parity.record(SUITE, check_name, True)
                continue
            g, w, pos = _window(got, want)
            parity.record(
                SUITE,
                check_name,
                False,
                case=f"{doc_id} @{pos}",
                input=text[max(0, pos - 40) : pos + 80],
                got=g,
                want=w,
            )
            diffs.append(doc_id)
    assert not diffs, (
        f"{len(diffs)}/{len(corpus)} {name} docs differ from pinned ftfy.fix_text; first: {diffs[:10]}"
    )


# -------------------------------------------------------------------------------------------------- API mirror


def _shape(fn: Callable[..., Any]) -> list[tuple[str, str, Any]]:
    return [(p.name, p.kind.name, p.default) for p in inspect.signature(fn).parameters.values()]


def _not_reference(ftfy_impl: Any, parity: Collector, check_name: str) -> None:
    if ftfy_impl.is_reference:
        parity.skip(SUITE, check_name, "candidate is ftfy itself")
        pytest.skip("candidate is ftfy itself")


@pytest.mark.parametrize("name", ["fix_text", "fix_encoding"])
def test_signature_mirrors_ftfy(ftfy_impl: Any, candidate_mod: Any, parity: Collector, name: str) -> None:
    _not_reference(ftfy_impl, parity, "api-signature")
    check(
        parity,
        "api-signature",
        outcome(lambda: _shape(getattr(candidate_mod, name))),
        ("ok", _shape(getattr(ftfy, name))),
        case=name,
    )


def test_config_mirrors_ftfy(ftfy_impl: Any, candidate_mod: Any, parity: Collector) -> None:
    _not_reference(ftfy_impl, parity, "api-config")
    config_cls = getattr(candidate_mod, "TextFixerConfig", None)
    if config_cls is None:
        parity.skip(SUITE, "api-config", "candidate exports no TextFixerConfig")
        pytest.skip("candidate exports no TextFixerConfig")
    got = outcome(lambda: dict(config_cls._field_defaults))
    check(
        parity, "api-config", got, ("ok", dict(ftfy.TextFixerConfig._field_defaults)), case="TextFixerConfig"
    )
