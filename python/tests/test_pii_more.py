"""pyroid.filters.pii_replace / PIIReplacer: datatrove 0.10.1's PIIFormatter, byte for byte.

The reference below is datatrove's own code (its two regexes, ``re.sub`` with a rotating
replacement and ``ipaddress``), so these tests need no datatrove; the live-datatrove tests skip
without it.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.filters as pf
from pyroid._pyroid import _testing

# datatrove 0.10.1 pipeline/formatters/pii.py
EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:(?:[A-Za-z0-9](?:["
    r"A-Za-z0-9-]*[A-Za-z0-9])?\.)+[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?|\[(?:(?:25[0-5]|2[0-4][0-9]|["
    r"01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?|[A-Za-z0-9-]*[A-Za-z0-9]:)])"
)
IP_RE = re.compile(
    r"(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"
)


class Reference:
    """datatrove's PIIFormatter, transcribed (state: one counter per replacer)."""

    def __init__(self, cfg: pf.PII) -> None:
        self.cfg = cfg
        self.i = [0, 0]

    def _rep(self, v: tuple[str, ...] | str) -> tuple[str, ...]:
        return (v,) if isinstance(v, str) else tuple(v)

    def _sub(
        self, k: int, regex: re.Pattern[str], reps: tuple[str, ...], ok: Any, text: str
    ) -> str:
        def repl(m: re.Match[str]) -> str:
            if ok is not None and not ok(m.group(0)):
                return m.group(0)
            r = reps[self.i[k]]
            self.i[k] = (self.i[k] + 1) % len(reps)
            return r

        return regex.sub(repl, text)

    def format(self, text: str) -> str:
        c = self.cfg

        def valid(ip: str) -> bool:
            try:
                addr = ipaddress.ip_address(ip)
            except ValueError:
                return False
            return not c.only_remove_public_ips or addr.is_global

        if c.remove_emails:
            text = self._sub(0, EMAIL_RE, self._rep(c.email_replacement), None, text)
        if c.remove_ips:
            text = self._sub(1, IP_RE, self._rep(c.ip_replacement), valid, text)
        return text


PIECES = [
    *"aZ09._-@[]:!#$%&'*+/=?^`{|}~ \n",
    "é",
    "ß",
    "̀",
    "٣",
    "²",
    "\u2160",
    "25",
    "255",
    "256",
    "01",
]
FRAGMENT = st.one_of(
    st.lists(st.sampled_from(PIECES), max_size=25).map("".join),
    st.from_regex(r"[a-z0-9._%+-]{1,6}@[a-z0-9-]{1,4}(\.[a-z0-9-]{1,4}){0,3}", fullmatch=True),
    st.from_regex(r"[0-9]{1,4}(\.[0-9]{1,4}){2,5}", fullmatch=True),
    st.from_regex(r"[a-z]{1,3}@\[[0-9]{1,3}(\.[0-9]{1,3}){3}\]", fullmatch=True),
    st.sampled_from(["8.8.8.8", "10.0.0.1", "192.168.1.1", "100.64.0.1", "08.8.8.8", "1.2.3.4"]),
    st.text(st.characters(exclude_categories=["Cs"]), max_size=6),
)
DOC = st.lists(FRAGMENT, max_size=6).map("".join)
CONFIG = st.builds(
    pf.PII,
    remove_emails=st.booleans(),
    remove_ips=st.booleans(),
    only_remove_public_ips=st.booleans(),
    email_replacement=st.sampled_from(
        [pf.PII_EMAIL_REPLACEMENT, "only@one.org", ("a@b.cc", "d@e.ff", "1.2.3.4")]
    ),
    ip_replacement=st.sampled_from([pf.PII_IP_REPLACEMENT, "9.9.9.9", ("1.1.1.1", "2.2.2.2")]),
)

DOCS = [
    "",
    "no pii here",
    "mail me at john.doe@example.com or jane@mail.co.uk, ok?",
    "servers 8.8.8.8 and 10.0.0.1 and 1.1.1.1; version 1.2.3.4.5",
    "a@[127.0.0.1] b@[1.2.3.x9:] c@b.c- d@b.c.-x",
    "ünïcödé-x@y.zz and éx@y.zz",
    "1234.5.6.7 and 1.2.3.256 and 01.2.3.4",
]


def test_reference_semantics_on_examples() -> None:
    ref = Reference(pf.PII())
    assert pf.pii_replace(DOCS) == [ref.format(t) for t in DOCS]


@settings(max_examples=300, deadline=None, suppress_health_check=list(HealthCheck))
@given(st.lists(DOC, min_size=1, max_size=8), CONFIG, st.integers(1, 4))
def test_matches_reference(texts: list[str], cfg: pf.PII, split: int) -> None:
    ref = Reference(cfg)
    want = [ref.format(t) for t in texts]
    r = pf.PIIReplacer(cfg)
    got = r(texts[:split]) + r(texts[split:])
    assert got == want
    assert [r.email_index, r.ip_index] == ref.i


@pytest.mark.parametrize("kind", [list, tuple])
def test_container_kinds_and_identity(kind: Callable[..., Any]) -> None:
    texts = kind(DOCS)
    out = pf.pii_replace(texts)
    assert type(out) is kind
    for a, b in zip(texts, out, strict=True):
        if a == b:
            assert a is b  # unchanged documents come back as the same objects
    s = "x@y.zz"
    assert pf.pii_replace(s) == "email@example.com"
    assert pf.pii_replace("nothing") == "nothing"


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    values = [*DOCS, None, "z@a.bb"]
    col = pf.pii_replace(_testing.column(values, large=large, chunks=chunks, name="t"))
    assert col.type == ("large_string" if large else "string")
    assert col.name == "t"
    ref = Reference(pf.PII())
    assert col.to_pylist() == [None if v is None else ref.format(v) for v in values]


def test_arrow_with_pyarrow(pa: Any) -> None:
    values = [*DOCS, None]
    want = pf.pii_replace([v or "" for v in values])
    for arr in (
        pa.array(values),
        pa.array(values, pa.large_string()),
        pa.array(values, pa.string_view()),
        pa.chunked_array([values[:3], values[3:]]),
        pa.array(["pad", *values])[1:],
    ):
        got = pf.pii_replace(arr).to_pylist()
        assert got == [None if v is None else w for v, w in zip(values, want, strict=True)]


def test_counters_carry_over_and_are_assignable() -> None:
    r = pf.PIIReplacer()
    assert r(["a@b.cc"]) == ["email@example.com"]
    assert r(["a@b.cc"]) == ["firstname.lastname@example.org"]
    assert (r.email_index, r.ip_index) == (0, 0)
    r.ip_index = 5
    assert r("8.8.8.8") == "18.104.22.168"
    assert r.ip_index == 0
    # A fresh function call starts at 0 again.
    want = ["email@example.com", "firstname.lastname@example.org"]
    assert pf.pii_replace(["a@b.cc", "a@b.cc"]) == want
    assert pf.pii_replace(["a@b.cc", "a@b.cc"]) == want


def test_only_public_ips() -> None:
    text = "8.8.8.8 10.0.0.1 127.0.0.1 100.64.0.1 192.0.0.9 08.8.8.8"
    ref = Reference(pf.PII())
    assert pf.pii_replace(text) == ref.format(text)
    allr = Reference(pf.PII(only_remove_public_ips=False))
    assert pf.pii_replace(text, only_remove_public_ips=False) == allr.format(text)
    # A leading zero never parses (ipaddress), so it is never replaced.
    assert pf.pii_replace("08.8.8.8", only_remove_public_ips=False) == "08.8.8.8"


def test_empty_replacements_raise_like_datatrove() -> None:
    with pytest.raises(IndexError):
        pf.pii_replace("a@b.cc", email_replacement=())
    assert pf.pii_replace("no match", email_replacement=()) == "no match"
    with pytest.raises(TypeError):
        pf.PIIReplacer(email_replacement=(1, 2))  # type: ignore[arg-type]


def test_lone_surrogate_raises() -> None:
    with pytest.raises(UnicodeEncodeError):
        pf.pii_replace(["ok", "bad \ud800 a@b.cc"])


def test_wrong_input_type() -> None:
    with pytest.raises(pyroid.InputError):
        pf.pii_replace(b"bytes")  # type: ignore[call-overload]


@pytest.mark.parametrize("threads", [None, 1, 2, 4])
def test_threads_equivalence(threads: int | None) -> None:
    docs = [f"{i}: user{i}@host{i % 7}.com from 8.8.{i % 256}.{i % 200}" for i in range(500)]
    assert pf.pii_replace(docs, threads=threads) == pf.pii_replace(docs, threads=1)


@pytest.mark.thread_unsafe(reason="10k documents")
@pytest.mark.timeout(120)
def test_ten_thousand_documents() -> None:
    docs = [f"doc {i} mail a{i}@b.com ip 1.{i % 250}.2.3 " * 20 for i in range(10_000)]
    out = pf.pii_replace(docs)
    ref = Reference(pf.PII())
    assert out[:50] == [ref.format(t) for t in docs[:50]]
    assert len(out) == 10_000


# ------------------------------------------------------------------------- live datatrove

DATA = Path(os.environ.get("PYROID_DATA", "/data"))


def _datatrove_pii() -> Any:
    return pytest.importorskip("datatrove.pipeline.formatters.pii")


@settings(max_examples=200, deadline=None, suppress_health_check=list(HealthCheck))
@given(st.lists(DOC, min_size=1, max_size=6), CONFIG)
def test_matches_live_datatrove(texts: list[str], cfg: pf.PII) -> None:
    dt = _datatrove_pii()
    ref = dt.PIIFormatter(**cfg._asdict())
    want = [ref.format(t) for t in texts]
    r = pf.PIIReplacer(cfg)
    assert r(texts) == want
    assert (r.email_index, r.ip_index) == (
        ref.emails_replacer._replace_i,
        ref.ip_replacer._replace_i,
    )


@pytest.mark.thread_unsafe(reason="whole corpus through datatrove")
@pytest.mark.timeout(600)
def test_corpus_matches_live_datatrove() -> None:
    dt = _datatrove_pii()
    pq = pytest.importorskip("pyarrow.parquet")
    files = [DATA / "fineweb" / "fineweb-dev-10k.parquet", DATA / "parity" / "adversarial.parquet"]
    if not all(f.exists() for f in files):
        pytest.skip("pinned corpora not available")
    texts = [
        t for f in files for t in pq.read_table(f, columns=["text"]).column("text").to_pylist()
    ]
    ref = dt.PIIFormatter()
    want = [ref.format(t) for t in texts]
    r = pf.PIIReplacer()
    got = [t for i in range(0, len(texts), 997) for t in r(texts[i : i + 997])]
    assert sum(g != w for g, w in zip(got, want, strict=True)) == 0


def test_drop_in_step_matches_datatrove() -> None:
    dt = _datatrove_pii()
    data = pytest.importorskip("datatrove.data")
    from pyroid.integrations import datatrove as pdt

    texts = [*DOCS, *(f"x{i}@y.zz 8.8.{i}.1" for i in range(30))]
    for kw in ({}, {"only_remove_public_ips": False, "ip_replacement": "1.1.1.1"}):
        stock = dt.PIIFormatter(**kw)
        mine = pdt.PIIFormatter(**kw, batch_size=7)
        want = [
            d.text for d in stock.run(data.Document(text=t, id=str(i)) for i, t in enumerate(texts))
        ]
        got = [
            d.text for d in mine.run(data.Document(text=t, id=str(i)) for i, t in enumerate(texts))
        ]
        assert got == want
        assert mine.format("a@b.cc") == stock.format("a@b.cc")
        assert mine.emails_replacer._replace_i == stock.emails_replacer._replace_i
        assert mine.ip_replacer._replace_i == stock.ip_replacer._replace_i
