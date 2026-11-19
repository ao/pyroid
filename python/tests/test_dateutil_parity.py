"""pyroid.dateutil.parser against python-dateutil: Hypothesis over date-like strings.

Every keyword and `default` value, lone surrogates, NUL and digits from other scripts; the
results must be the same values, types, tzinfo types and reprs, and the same exceptions with
the same messages.
"""

from __future__ import annotations

import datetime
import warnings
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pyroid
import pyroid.dateutil.parser as pp

# python-dateutil is untyped for these tests (the mypy hook does not install it).
du: Any = pytest.importorskip("dateutil.parser")
tz: Any = pytest.importorskip("dateutil.tz")

PIECES = (
    [str(i) for i in range(61)]
    + ["%02d" % i for i in range(13)]  # noqa: UP031
    + ["1999", "2003", "99", "00", "100", "9999", "10000", "123456", "20030925", "200309251049"]
    + ["20030925104941", "19990101T23", "235959.5", "1.5", "10.25", "3,5", "12,5", "0.000001"]
    + ["Jan", "January", "feb", "Sep", "Sept", "Mar.", "May", "dec", "Thu", "Thursday", "sun"]
    + ["am", "pm", "AM", "PM", "a", "p", "A.M.", "P.M.", "a.m.", "m", "UTC", "GMT", "Z", "z"]
    + ["BRST", "EST", "CET", "PST", "XYZ", "ABCDEF", "utc", "h", "s", "hours", "minute"]
    + ["of", "at", "on", "and", "ad", "T", "t", "st", "nd", "rd", "th", " ", "  ", "\t", "-"]
    + ["/", ":", ".", ",", ";", "'", "+", "(", ")", "_", "\x00", "\n", "　", "inf", "nan"]
    + ["١٢", "٣", "²", "½", "é", "ß", "İ", "K", "\ud800", "\udfff", "\U0010f801"]  # noqa: RUF001
)

text = st.lists(
    st.one_of(st.sampled_from(PIECES), st.text(min_size=1, max_size=3)), min_size=0, max_size=14
).map("".join)

defaults = st.sampled_from(
    [
        None,
        datetime.datetime(2003, 9, 25),
        datetime.datetime(2024, 2, 29, 13, 14, 15, 161718),
        datetime.datetime(2003, 1, 31, tzinfo=datetime.UTC),
        datetime.datetime(9999, 12, 31),
        datetime.datetime(1, 1, 1, fold=1),
    ]
)
tzinfos = st.sampled_from(
    [None, {"BRST": -10800}, {"EST": tz.tzoffset("EST", -18000), "CET": 3600}, {}]
)


def rep(r: Any) -> Any:
    if isinstance(r, tuple):
        return ("tuple", rep(r[0]), r[1])
    return (type(r), r.isoformat(), repr(r.tzinfo), type(r.tzinfo), r.fold)


def outcome(f: Any, s: str, kw: dict[str, Any]) -> Any:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out: tuple[Any, ...]
        try:
            out = ("ok", rep(f(s, **kw)))
        except Exception as e:
            out = ("err", type(e), str(e), type(e.__cause__))
    seen = [(x.category, str(x.message)) for x in w if x.category is not pyroid.FallbackWarning]
    return out, seen


@pytest.mark.thread_unsafe(reason="catch_warnings")
@settings(max_examples=3000, deadline=None)
@given(
    s=text,
    default=defaults,
    dayfirst=st.sampled_from([None, True, False]),
    yearfirst=st.sampled_from([None, True, False]),
    fuzzy=st.booleans(),
    fuzzy_with_tokens=st.booleans(),
    ignoretz=st.booleans(),
    tzinfos=tzinfos,
)
def test_parse(
    s: str,
    default: Any,
    dayfirst: Any,
    yearfirst: Any,
    fuzzy: bool,
    fuzzy_with_tokens: bool,
    ignoretz: bool,
    tzinfos: Any,
) -> None:
    kw = {
        "default": default,
        "dayfirst": dayfirst,
        "yearfirst": yearfirst,
        "fuzzy": fuzzy,
        "fuzzy_with_tokens": fuzzy_with_tokens,
        "ignoretz": ignoretz,
        "tzinfos": tzinfos,
    }
    if default is None:
        kw.pop("default")
    assert outcome(pp.parse, s, kw) == outcome(du.parse, s, kw)


iso_text = st.lists(
    st.sampled_from(
        [
            *["2003", "-", "09", "25", "T", "10", ":", "49", "41", ".", ",", "5", "123456789"],
            *[
                "Z",
                "+",
                "05",
                "30",
                "W",
                "39",
                "4",
                "268",
                " ",
                "_",
                "24",
                "00",
                "60",
                "0",
                "z",
                "é",
            ],
        ]
    ),
    min_size=0,
    max_size=12,
).map("".join)


@pytest.mark.thread_unsafe(reason="catch_warnings")
@settings(max_examples=3000, deadline=None)
@given(s=iso_text, sep=st.sampled_from([None, "T", " "]))
def test_isoparse(s: str, sep: str | None) -> None:
    import importlib.metadata

    installed = importlib.metadata.version("python-dateutil")
    if installed not in pp.DATEUTIL_VERSIONS:
        pytest.skip(f"python-dateutil {installed} is not reproduced")
    a = outcome(du.isoparser(sep).isoparse, s, {})
    b = outcome(pp.isoparser(sep).isoparse, s, {})
    assert a == b
