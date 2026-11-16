"""pyroid.dateutil.parser: API, objects, batches, fallbacks and hostile inputs.

The differential checks against python-dateutil itself are in test_dateutil_parity.py.
"""

from __future__ import annotations

import contextlib
import datetime
import decimal
import gc
import io
import pickle
import subprocess
import sys
import textwrap
import threading
import warnings
from typing import Any, cast

import pytest

import pyroid
import pyroid.dateutil.parser as pp
import pyroid.integrations

# python-dateutil is untyped for these tests (the mypy hook does not install it).
du: Any = pytest.importorskip("dateutil.parser")
tz: Any = pytest.importorskip("dateutil.tz")
# The private parts of pyroid's classes that the tests reach into.
ppx: Any = pp

D = datetime.datetime
DEFAULT = D(2003, 9, 25)


def same(a: Any, b: Any) -> None:
    """Equal values of the same type, with the same tzinfo type and repr."""
    assert type(a) is type(b)
    if isinstance(a, tuple):
        assert len(a) == len(b)
        same(a[0], b[0])
        assert a[1] == b[1]
        return
    assert a == b
    assert repr(a) == repr(b)
    assert type(a.tzinfo) is type(b.tzinfo)
    assert a.fold == b.fold


def outcome(f: Any, *args: Any, **kwargs: Any) -> Any:
    # The FallbackWarning says which path ran, not what the result is.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        try:
            return ("ok", f(*args, **kwargs))
        except Exception as e:
            return ("err", type(e), str(e), type(e.__cause__))


def check(s: Any, **kwargs: Any) -> None:
    a = outcome(du.parse, s, **kwargs)
    b = outcome(pp.parse, s, **kwargs)
    if a[0] == "ok" and b[0] == "ok":
        same(a[1], b[1])
    else:
        assert a == b


# --- the API -------------------------------------------------------------------------------


def test_versions() -> None:
    assert pp.DATEUTIL_VERSION == "4.0.0"
    assert pp.DATEUTIL_VERSIONS == ("4.0.0", "3.0.0", "2.9.0.post0", "2.9.0", "2.8.2")


def test_functions_pickle() -> None:
    # Pool.map(parse, ...) pickles parse by reference: its module must hold it.
    # Round-trips objects this test made (no untrusted data).
    for f in (pp.parse, pp.isoparse, pp.parse_batch, pp.isoparse_batch):
        assert pickle.loads(pickle.dumps(f)) is f
    p = pickle.loads(pickle.dumps(pp.DEFAULTPARSER.parse))
    assert p("2003-09-25 10:49") == D(2003, 9, 25, 10, 49)


@pytest.mark.thread_unsafe(reason="starts a process pool")
def test_parse_in_a_spawned_process_pool() -> None:
    import multiprocessing

    with multiprocessing.get_context("spawn").Pool(1) as pool:
        got = pool.map(pp.parse, ["2003-09-25", "Sep 26 2003 10:00"])
        assert pool.map(pp.isoparse, ["2003-09-25T10:00"]) == [D(2003, 9, 25, 10)]
    assert got == [D(2003, 9, 25), D(2003, 9, 26, 10)]


def test_classes_are_dateutils_own() -> None:
    assert pp.ParserError is du.ParserError
    assert pp.UnknownTimezoneWarning is du.UnknownTimezoneWarning
    with pytest.raises(du.ParserError, match="Unknown string format: nonsense"):
        pp.parse("nonsense")


def test_common_strings() -> None:
    for s in [
        "2003-09-25T10:49:41.5-03:00",
        "Thu Sep 25 10:36:28 BRST 2003",
        "Thu, 25 Sep 2003 10:49:41 -0300",
        "10/09/03",
        "Sep 25 2003",
        "25 Sep 2003 10:49 PM",
        "5:50 A.M. on June 13, 1990",
        "1996.07.10 AD at 15:08:56 PDT",
        "Tuesday, April 12, 1952 AD 3:30:42pm PST",
        "December.0031.30",
        "10h36m28.5s",
        "19990101T235959.59",
        "2003-09-25T10:49:41Z",
        "Tue, 02 Jan 2024 10:00:00 GMT",
        "10:00 GMT+3",
        "0:00 PM, PST",
        "12 pm",
        "",
        " ",
        "foo",
    ]:
        for kw in ({}, {"default": DEFAULT}, {"dayfirst": True}, {"yearfirst": True}):
            check(s, **kw)


def test_result_types() -> None:
    r = pp.parse("2003-09-25T10:49:41Z")
    assert r.tzinfo is not None
    r = pp.parse("2003-09-25T10:49:41+05:30")
    assert isinstance(r.tzinfo, tz.tzoffset)
    assert r.tzinfo == tz.tzoffset(None, 19800)
    r = pp.parse("2003-09-25 10:49:41 BRST", tzinfos={"BRST": -10800})
    assert r.tzinfo == tz.tzoffset("BRST", -10800)


def test_fuzzy_with_tokens() -> None:
    s = "Today is 25 of September of 2003, exactly at 10:49:41 with timezone -03:00."
    check(s, fuzzy_with_tokens=True)
    check("Today is January 1, 2047 at 8:21:00AM", fuzzy_with_tokens=True)
    check("http://biz.yahoo.com/ipo/p/600221.html", fuzzy_with_tokens=True)


def test_inputs() -> None:
    check(b"2003-09-25 10:49")
    check(bytearray(b"2003-09-25 10:49"))
    a = outcome(du.parse, io.StringIO("2003-09-25 10:49"))
    b = outcome(pp.parse, io.StringIO("2003-09-25 10:49"))
    same(a[1], b[1])
    a = outcome(du.parse, io.StringIO("nonsense"))
    b = outcome(pp.parse, io.StringIO("nonsense"))
    assert a[0] == b[0] == "err"
    assert a[1] is b[1]
    with pytest.raises(TypeError, match="Parser must be a string or character stream, not int"):
        pp.parse(20030925)  # type: ignore[call-overload]
    with pytest.raises(UnicodeDecodeError):
        pp.parse(b"\xff2003")


class OddStream:
    """A stream whose ``read(1)`` gives ``chunks``, then `last` forever; counts the reads."""

    def __init__(self, chunks: list[Any], last: Any) -> None:
        self.chunks = list(chunks)
        self.last = last
        self.reads = 0

    def read(self, n: int = -1) -> Any:
        self.reads += 1
        if self.reads > 10_000:
            raise RuntimeError("read past the 10,000th character")
        return self.chunks.pop(0) if self.chunks else self.last


@pytest.mark.parametrize(
    ("chunks", "last"),
    [
        ([], 5),  # read() returns an int: dateutil fails at the first character
        ([], True),
        (list("2003-09-25 "), 7),
        (list("Sep 2"), "5 "),  # two characters at once, then a never-ending stream
        ([], b"2"),
        (["2", "0", "0", "3"], b""),
        ([], []),
        (list("2003-09-25"), ""),
        (["1", "\x00", "0", ":", "3", "0"], ""),
    ],
)
def test_odd_streams_are_read_as_dateutil_reads_them(chunks: list[Any], last: Any) -> None:
    a, b = OddStream(chunks, last), OddStream(chunks, last)
    ra = outcome(du.parse, a, default=DEFAULT)
    rb = outcome(pp.parse, b, default=DEFAULT)
    if ra[0] == rb[0] == "ok":
        same(ra[1], rb[1])
    else:
        assert ra[:2] == rb[:2]
        assert ra[2].replace(repr(a), "S") == rb[2].replace(repr(b), "S")
    assert a.reads == b.reads


def test_stream_errors_hold_the_callers_stream() -> None:
    for make in (
        lambda: io.BytesIO(b"nonsense"),
        lambda: io.BytesIO(b"Feb 30 2021"),
        lambda: OddStream([*"nonsen", "se"], ""),
        lambda: OddStream(list("nonsense"), ""),
    ):
        for f in (du.parse, pp.parse, pp.DEFAULTPARSER.parse):
            s = make()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", pyroid.FallbackWarning)
                with pytest.raises(du.ParserError) as e:
                    f(s)  # type: ignore[arg-type]
            assert e.value.args[1] is s


def test_unexpected_keyword_is_dateutils_type_error() -> None:
    a = outcome(du.parse, "2003", fuzzzy=True)
    b = outcome(pp.parse, "2003", fuzzzy=True)
    assert a[0] == b[0] == "err"
    assert a[1] is b[1] is TypeError
    assert a[2] == b[2]


def test_parse_keywords() -> None:
    # parse takes dateutil's keywords as named parameters: None is the same as not passing one.
    kw = dict.fromkeys(["dayfirst", "yearfirst", "fuzzy", "fuzzy_with_tokens", "ignoretz"])
    check("01/02/03 10:00 +0300", default=None, tzinfos=None, **kw)
    check("01/02/03", dayfirst=True, default=DEFAULT)
    # The tzoffset objects are dateutil's: its factory gives the same one while it lives.
    z = pp.parse("2003-09-25 10:00 +0300").tzinfo
    assert z is tz.tzoffset(None, 10800)
    assert pp.parse("2003-09-26 10:00 +0300").tzinfo is z
    assert pp.parse("2003-09-26 10:00 +0300 (BRST)").tzinfo is tz.tzoffset("BRST", 10800)


@pytest.mark.thread_unsafe(reason="replaces DEFAULTPARSER")
def test_parse_passes_its_keywords_on(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    class Spy(ppx.parser):  # type: ignore[misc]
        def parse(self, timestr: Any, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return super().parse(timestr, **kwargs)

    monkeypatch.setattr(pp, "DEFAULTPARSER", Spy())
    assert pp.parse("01/02/03", dayfirst=True, default=DEFAULT) == D(2003, 2, 1)
    assert seen == [{"default": DEFAULT, "dayfirst": True}]


def test_tzinfos() -> None:
    for tzinfos in (
        {"BRST": -10800},
        {"BRST": tz.tzoffset("BRST", -10800)},
        {"BRST": "BRST-3BRDT"},
        {"BRST": None},
        lambda name, offset: tz.tzoffset(name, 3600) if name else None,
    ):
        check("2003-09-25 10:49:41 BRST", tzinfos=tzinfos)
        check("2003-09-25 10:49:41", tzinfos=tzinfos)
    check("2003-09-25 10:49:41 BRST", tzinfos={"BRST": 1.5})


def test_unknown_timezone_warning() -> None:
    for f in (du.parse, pp.parse):
        with pytest.warns(du.UnknownTimezoneWarning, match="tzname XYZ identified"):
            f("2003-09-25 10:49:41 XYZ")


def test_default() -> None:
    check("10:49", default=D(2024, 2, 29, 1, 2, 3, 4))
    check("Feb", default=D(2024, 1, 31))
    check("Feb 2023", default=D(2024, 1, 31))
    check("10:49", default=D(2003, 9, 25, tzinfo=datetime.UTC))
    check("10:49", default=D(2003, 9, 25, fold=1))
    check("Thu", default=D(2003, 9, 25, fold=1))
    check("Fri", default=D(9999, 12, 31))

    class Sub(D):
        pass

    check("10:49", default=Sub(2003, 9, 25))
    check("10:49", default=datetime.date(2003, 9, 25))
    check("10:49", default=42)
    r = pp.parse("10:49")
    assert (
        r.date() == datetime.date.today()
        or r.date() == datetime.date.today() - datetime.timedelta(1)
    )


def test_errors_and_messages() -> None:
    for s in ["32/13/2020", "2020-13", "Feb 30 2021", "99999999999999999999", "25:00", "1" * 30]:
        check(s)
        check(s, fuzzy=True)
    check("1" * 5000)


def test_parser_error_pickles() -> None:
    with pytest.raises(pp.ParserError) as info:
        pp.parse("nonsense")
    # Round-trips an exception this test made (no untrusted data).
    e2 = pickle.loads(pickle.dumps(info.value))
    assert str(e2) == str(info.value)
    assert repr(e2) == repr(info.value)


# --- parser, parserinfo -------------------------------------------------------------------


class German(ppx.parserinfo):  # type: ignore[misc]
    # dateutil's convention: plain class-level lists.
    MONTHS = [  # noqa: RUF012
        ("Jan", "Januar"),
        ("Feb", "Februar"),
        ("Mär", "März"),
        ("Apr", "April"),
        ("Mai", "Mai"),
        ("Jun", "Juni"),
        ("Jul", "Juli"),
        ("Aug", "August"),
        ("Sep", "September"),
        ("Okt", "Oktober"),
        ("Nov", "November"),
        ("Dez", "Dezember"),
    ]
    WEEKDAYS = [  # noqa: RUF012
        ("Mo", "Montag"),
        ("Di", "Dienstag"),
        ("Mi", "Mittwoch"),
        ("Do", "Donnerstag"),
        ("Fr", "Freitag"),
        ("Sa", "Samstag"),
        ("So", "Sonntag"),
    ]
    JUMP = [*pp.parserinfo.JUMP, "um", "uhr", "den"]  # noqa: RUF012


class DuGerman(du.parserinfo):  # type: ignore[misc]
    MONTHS = German.MONTHS
    WEEKDAYS = German.WEEKDAYS
    JUMP = German.JUMP


@pytest.mark.parametrize(
    "s", ["Donnerstag, den 25. September 2003 um 10:49 Uhr", "25. MÄRZ 2003", "Mi 5 Dez"]
)
def test_custom_parserinfo_runs_natively(s: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", pyroid.FallbackWarning)
        b = outcome(pp.parse, s, German(), default=DEFAULT)
        c = outcome(pp.parse, s, DuGerman(), default=DEFAULT)
    a = outcome(du.parse, s, DuGerman(), default=DEFAULT)
    assert a == b == c


def test_parserinfo_changes_are_seen() -> None:
    info: Any = pp.parserinfo()
    p = pp.parser(info)
    assert p.parse("25 Sep 2003") == D(2003, 9, 25)
    with pytest.raises(pp.ParserError):
        p.parse("25 Septembre 2003")
    info._months["septembre"] = 8
    assert p.parse("25 Septembre 2003") == D(2003, 9, 25)
    info.dayfirst = True
    assert p.parse("01/02/03") == D(2003, 2, 1)
    info._year, info._century = 1950, 1900
    assert p.parse("01/02/99") == D(1999, 2, 1)
    info.TZOFFSET = {"BRST": -10800}  # this instance's, so parallel runs do not share it
    assert p.parse("10:00 BRST").utcoffset() == datetime.timedelta(hours=-3)
    info.TZOFFSET["BRDT"] = -7200
    assert p.parse("10:00 BRDT").utcoffset() == datetime.timedelta(hours=-2)


@pytest.mark.thread_unsafe(reason="starts its own threads")
def test_tables_changed_while_other_threads_parse() -> None:
    # Regression: the tables are copied before they are read. Iterating a dict that another
    # thread changes made PyO3 panic ("dictionary changed size during iteration").
    info: Any = pp.parserinfo()
    info.TZOFFSET = {}
    p = pp.parser(info)
    stop = threading.Event()
    errors: list[BaseException] = []

    def change() -> None:
        i = 0
        while not stop.is_set():
            info.TZOFFSET[f"Z{i % 50:02d}"] = 3600
            info._jump[f"w{i % 50}"] = 0
            if i % 2:
                info.TZOFFSET.pop(f"Z{(i - 1) % 50:02d}", None)
                info._jump.pop(f"w{(i - 1) % 50}", None)
            i += 1

    def parse() -> None:
        try:
            for _ in range(3000):
                p.parse("2003-09-25 10:00 at noon", default=DEFAULT, fuzzy=True)
        except BaseException as e:  # a PanicException is a BaseException
            errors.append(e)

    changer = threading.Thread(target=change)
    changer.start()
    workers = [threading.Thread(target=parse) for _ in range(8)]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    stop.set()
    changer.join()
    assert not errors, errors[0]


@pytest.mark.thread_unsafe(reason="replaces sys.unraisablehook")
def test_parse_in_del_while_a_cache_drops_entries(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: the per-thread caches dropped old entries while their RefCell was borrowed. The
    # last reference to a parserinfo (or to a table value) going with it ran __del__, whose parse
    # borrowed the same cache: PyO3 panicked ("already mutably borrowed"), reported as unraisable.
    unraisable: list[Any] = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)
    seen: list[D] = []

    class DelInfo(ppx.parserinfo):  # type: ignore[misc]
        def __del__(self) -> None:
            seen.append(pp.parse("2020-01-02"))

    for _ in range(4):
        assert pp.parser(DelInfo()).parse("10:00", default=DEFAULT) == D(2003, 9, 25, 10)

    class DelInt(int):
        def __del__(self) -> None:
            seen.append(pp.parse("2020-01-02"))

    for _ in range(12):
        info: Any = pp.parserinfo()
        info.TZOFFSET = {"XYZ": DelInt(3600)}
        got = pp.parse("10:00 XYZ", parserinfo=info, default=DEFAULT)
        assert got.utcoffset() == datetime.timedelta(hours=1)
        del info, got
    gc.collect()
    assert not unraisable, unraisable[0].exc_value
    assert len(seen) >= 3
    assert all(d == D(2020, 1, 2) for d in seen)


class LoudInfo(ppx.parserinfo):  # type: ignore[misc]
    def convertyear(self, year: int, century_specified: bool = False) -> int:
        return int(super().convertyear(year, century_specified)) + 1


class DuLoudInfo(du.parserinfo):  # type: ignore[misc]
    def convertyear(self, year: int, century_specified: bool = False) -> int:
        return int(super().convertyear(year, century_specified)) + 1


@pytest.mark.thread_unsafe(reason="pytest.warns")
def test_parserinfo_method_override_runs_dateutils_code() -> None:
    ppx._WARNED.clear()  # the warning is shown once per process
    with pytest.warns(pyroid.FallbackWarning, match="overriding convertyear"):
        b = pp.parse("01/02/03", LoudInfo())
    assert b == du.parse("01/02/03", DuLoudInfo())


class MyParser(ppx.parser):  # type: ignore[misc]
    def _parsems(self, value: str) -> tuple[int, int]:
        s, us = super()._parsems(value)
        return int(s), int(us) + 1


@pytest.mark.thread_unsafe(reason="pytest.warns")
def test_parser_hook_override_runs_dateutils_code() -> None:
    ppx._WARNED.clear()  # the warning is shown once per process
    with pytest.warns(pyroid.FallbackWarning, match="overriding _parsems"):
        assert MyParser().parse("10:20:30.5").microsecond == 500_001


class NumericParser(ppx.parser):  # type: ignore[misc]
    def _parse_numeric_token(self, *args: Any, **kwargs: Any) -> Any:
        return super()._parse_numeric_token(*args, **kwargs)


class OwnParseParser(ppx.parser):  # type: ignore[misc]
    def _parse(self, timestr: Any, **kwargs: Any) -> Any:
        return super()._parse(timestr, **kwargs)


@pytest.mark.thread_unsafe(reason="records warnings")
@pytest.mark.parametrize(
    ("call", "reason"),
    [
        (lambda: pp.parse("18446744073709551616 Sep 2003"), "a number past 64 bits"),
        (lambda: NumericParser().parse("Sep 25 2003"), "overriding _parse_numeric_token"),
        (lambda: OwnParseParser().parse("Sep 25 2003"), "overriding _parse"),
        (lambda: pp.parse("01/02/03", LoudInfo()), "overriding convertyear"),
        (lambda: pp.parse(io.BytesIO(b"Sep 25 2003")), "this stream"),  # type: ignore[arg-type]
        pytest.param(
            lambda: pp.parse("10." + "1" * 25 + "h"), "26 significant digits", id="long hour"
        ),
        pytest.param(
            lambda: pp.parse("10:00." + "1" * 27), "26 significant digits", id="long minute"
        ),
    ],
)
def test_one_fallback_warning_per_reason(call: Any, reason: str) -> None:
    ppx._WARNED.clear()  # the warning is shown once per process
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(2):
            with contextlib.suppress(ValueError):
                call()
    got = [x for x in w if x.category is pyroid.FallbackWarning]
    assert len(got) == 1, [str(x.message) for x in got]
    assert reason in str(got[0].message)
    assert str(got[0].message).count("Rust port") == 1
    assert got[0].filename == __file__


@pytest.mark.thread_unsafe(reason="records warnings")
@pytest.mark.parametrize(
    ("s", "falls_back"),
    [
        # The decimal context matters only where dateutil computes with the number (an hour or
        # minute with a fraction); the fraction of a second is read as text, natively too.
        ("10:00:00." + "1" * 40, False),
        ("10:00." + "1" * 26, False),
        ("10:00." + "1" * 27, True),
        ("10." + "1" * 24 + "h", False),
        ("10." + "1" * 25 + "h", True),
        ("10." + "9" * 40 + "m", True),
    ],
)
def test_long_fractions_match_dateutil(s: str, falls_back: bool) -> None:
    check(s, default=DEFAULT)
    ppx._WARNED.clear()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with contextlib.suppress(ValueError):  # dateutil's own error, for "10.99...9m"
            pp.parse(s, default=DEFAULT)
    assert any(x.category is pyroid.FallbackWarning for x in w) is falls_back


def test_private_parse() -> None:
    res, tokens = ppx.DEFAULTPARSER._parse("Sep 25 2003 10:49", fuzzy_with_tokens=True)
    ref, ref_tokens = du.DEFAULTPARSER._parse("Sep 25 2003 10:49", fuzzy_with_tokens=True)
    assert repr(res) == repr(ref)
    assert tokens == ref_tokens
    assert ppx.DEFAULTPARSER._parse("nonsense") == (None, None)


# --- isoparse ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "s",
    [
        "2003-09-25T10:49:41.5-03:00",
        "2003-09-25T10:49:41Z",
        "2003-09-25",
        "2003-W39-4",
        "2003268",
        "2014- 4-21",
        "20200728121240",
        "2014-04-21T24:00",
        "9999-12-31T24:00",
        "0000-01-01",
        "2014-04-21T10:20+-1:00",
        "2003-09-25T10:49:41+00:00",
        "x",
        "2003-02-30",
    ],
)
@pytest.mark.parametrize("compat", ["4.0.0", "2.9.0.post0"])
def test_isoparse_modes(s: str, compat: str) -> None:
    from pyroid.dateutil import _isoreference

    ref: Any = cast(Any, _isoreference).isoparser()
    ref._lenient = compat != "4.0.0"
    a = outcome(ref.isoparse, s)
    b = outcome(ppx.isoparser(compat=compat).isoparse, s)
    if a[0] == b[0] == "ok":
        same(a[1], b[1])
    else:
        assert a == b


def test_isoparse_follows_installed() -> None:
    import importlib.metadata

    installed = importlib.metadata.version("python-dateutil")
    a = outcome(du.isoparse, "2014- 4-21")
    b = outcome(pp.isoparse, "2014- 4-21")
    assert a[0] == b[0], installed
    with pytest.raises(ValueError, match="compat must be one of"):
        pp.isoparser(compat="1.0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Separator must be"):
        pp.isoparser(sep="12")


def test_isoparse_inputs() -> None:
    for s in [b"2003-09-25", "2003-09-25T10:00é", 20030925]:
        assert outcome(du.isoparse, s) == outcome(pp.isoparse, s)
    a = outcome(du.isoparse, io.StringIO("2003-09-25"))
    b = outcome(pp.isoparse, io.StringIO("2003-09-25"))
    same(a[1], b[1])


def test_isoparse_type_errors_are_dateutils() -> None:
    # dateutil's isoparse is DEFAULT_ISOPARSER.isoparse, so its TypeErrors name isoparser.isoparse.
    for args, kwargs in [((), {}), (("2003-09-25",), {"sep": "T"}), (("2003-09-25", "T"), {})]:
        for f in (pp.isoparse, pp.DEFAULT_ISOPARSER.isoparse):
            assert outcome(f, *args, **kwargs) == outcome(du.isoparse, *args, **kwargs)
    assert outcome(pp.isoparse, dt_str="2003-09-25") == outcome(du.isoparse, dt_str="2003-09-25")
    import inspect

    assert str(inspect.signature(pp.isoparse)) == "(dt_str)"
    assert str(inspect.signature(pp.parse)) == "(timestr, parserinfo=None, **kwargs)"


# --- batches -------------------------------------------------------------------------------

STRINGS = ["2003-09-25", "Sep 26 2003 10:00", "10/27/03 1:02:03.5", "25.09.2003"]


def test_parse_batch_kinds() -> None:
    want = [pp.parse(s) for s in STRINGS]
    assert pp.parse_batch(STRINGS) == want
    assert pp.parse_batch(tuple(STRINGS)) == tuple(want)
    assert pp.parse_batch(STRINGS[0]) == want[0]
    assert pp.parse_batch(STRINGS, default=DEFAULT, dayfirst=True) == [
        pp.parse(s, default=DEFAULT, dayfirst=True) for s in STRINGS
    ]


def test_parse_batch_errors() -> None:
    with pytest.raises(pp.ParserError) as e:
        pp.parse_batch(["2003-09-25", "nonsense"])
    assert "while parsing item 1" in "".join(getattr(e.value, "__notes__", []))
    assert pp.parse_batch(["2003-09-25", "nonsense", "Feb 30 2021"], errors="coerce") == [
        D(2003, 9, 25),
        None,
        None,
    ]
    with pytest.raises(ValueError, match="errors must be"):
        pp.parse_batch(STRINGS, errors="ignore")  # type: ignore[call-overload]
    with pytest.raises(TypeError, match="fuzzy_with_tokens"):
        pp.parse_batch(STRINGS, fuzzy_with_tokens=True)  # type: ignore[call-overload]


def test_parse_batch_threads() -> None:
    xs = [
        f"{2000 + i % 30}-{1 + i % 12:02d}-{1 + i % 28:02d} {i % 24}:{i % 60:02d}"
        for i in range(20000)
    ]
    want = [pp.parse(s) for s in xs]
    for threads in (None, 1, 2, 7):
        assert pp.parse_batch(xs, threads=threads) == want


def test_parse_batch_arrow() -> None:
    pa = pytest.importorskip("pyarrow")
    col = pp.parse_batch(pa.array(["2003-09-25 10:00", None, "Sep 26 2003 10:00:00.5"]))
    assert col.type == "timestamp[us]"
    arr = pa.array(col)
    assert arr.type == pa.timestamp("us")
    assert arr.to_pylist() == [D(2003, 9, 25, 10), None, D(2003, 9, 26, 10, 0, 0, 500000)]
    col = pp.parse_batch(
        pa.chunked_array([["2003-09-25 10:00Z", "2003-09-25 10:00+05:30"], ["x"]]),
        errors="coerce",
    )
    assert col.type == "timestamp[us, tz=UTC]"
    assert pa.array(col).to_pylist() == [
        D(2003, 9, 25, 10, tzinfo=datetime.UTC),
        D(2003, 9, 25, 4, 30, tzinfo=datetime.UTC),
        None,
    ]
    with pytest.raises(ValueError, match="mix naive and aware"):
        pp.parse_batch(pa.array(["10:00Z", "10:00"]))
    assert pp.parse_batch(pa.array(["10:00Z", "10:00"]), ignoretz=True).type == "timestamp[us]"


def test_parse_batch_arrow_without_pyarrow() -> None:
    from pyroid._pyroid import _testing

    col = pp.parse_batch(_testing.column(["2003-09-25", None], large=True, chunks=1))
    assert col.to_pylist() == [D(2003, 9, 25), None]


class _Month13(D):
    @property
    def month(self) -> int:
        return 13


class _FarOffset(D):
    def utcoffset(self) -> Any:
        return datetime.timedelta.max


def test_parse_batch_arrow_checks_datetime_subclass_fields() -> None:
    # Regression: an Arrow result is read through the datetime's attributes, which a subclass can
    # override; invalid fields panicked (`expect`), and a huge offset could overflow.
    from pyroid._pyroid import _testing

    col = _testing.column(["2020-01-02 10:00"], large=False, chunks=1)
    with pytest.raises(ValueError, match="not a valid date and time for an Arrow timestamp"):
        pp.parse_batch(col, default=_Month13(2003, 9, 25))
    with pytest.raises(ValueError, match=r"UTC offset .* out of range for an Arrow timestamp"):
        pp.parse_batch(col, default=_FarOffset(2003, 9, 25))
    first = pp.parse_batch(["2020-01-02 10:00"], default=_Month13(2003, 9, 25))[0]
    assert first is not None
    assert first.day == 2


def test_batches_take_lone_surrogates() -> None:
    # A str with lone surrogates is parsed as parse() parses it (the kernel's input refuses it).
    xs = ["2003-09-25", "\ud800 2003-09-25 10:00", "\ud800", "Sep 26 2003"]
    want = []
    for x in xs:
        r = outcome(pp.parse, x, default=DEFAULT, fuzzy=True)
        want.append(r[1] if r[0] == "ok" else None)
    assert want[1] == D(2003, 9, 25, 10)
    assert pp.parse_batch(xs, default=DEFAULT, fuzzy=True, errors="coerce") == want
    assert pp.parse_batch(tuple(xs), default=DEFAULT, fuzzy=True, errors="coerce") == tuple(want)
    assert pp.parse_batch(xs[1], default=DEFAULT, fuzzy=True) == want[1]
    with pytest.raises(pp.ParserError) as e:
        pp.parse_batch(xs, default=DEFAULT, fuzzy=True)
    assert "while parsing item 2" in "".join(getattr(e.value, "__notes__", []))
    with pytest.raises(pyroid.InputError, match="expected str at index 3, got int"):
        pp.parse_batch(["\ud800", "2003", "2004", 5])  # type: ignore[list-item]
    assert pp.isoparse_batch(["2003-09-25", "\ud800"], errors="coerce") == [D(2003, 9, 25), None]
    with pytest.raises(ValueError, match="only ASCII") as e2:
        pp.isoparse_batch(["2003-09-25", "\ud800"])
    assert "while parsing item 1" in "".join(getattr(e2.value, "__notes__", []))


def test_isoparse_batch() -> None:
    xs = ["2003-09-25T10:49:41", "2003-W39-4", "x"]
    assert pp.isoparse_batch(xs, errors="coerce") == [pp.isoparse(xs[0]), pp.isoparse(xs[1]), None]
    assert pp.isoparse_batch(xs[:2], compat="2.9.0.post0", sep="T") == [
        pp.isoparser("T", compat="2.9.0.post0").isoparse(x) for x in xs[:2]
    ]
    with pytest.raises(ValueError, match="ISO string too short"):
        pp.isoparse_batch(xs)
    pa = pytest.importorskip("pyarrow")
    col = pp.isoparse_batch(pa.array(["2003-09-25T10:49:41+01:00", None]))
    assert pa.array(col).to_pylist() == [D(2003, 9, 25, 9, 49, 41, tzinfo=datetime.UTC), None]


# --- hostile inputs, fallbacks ---------------------------------------------------------------


@pytest.mark.parametrize(
    "s",
    [
        "\x00" * 1000 + "2003-09-25",
        "2003\x00-09\x00-25",
        "\ud800 2003-09-25",
        "2003-09-25 \udfff",
        "\U0010f801 2003-09-25",
        "\ud800 \U0010f801 2003",
        "1" * 100_000,
        "1 " * 50_000,
        "-" * 10_000,
        "10:" + "1" * 40 + ".5",
        "10:20:30." + "9" * 40,
        "10.999999999999999999999999999999h",
        "99999999999999999999:00",
        "10:00 +99999999999999999999:00",
        "9999-12-31 Fri",
    ],
)
def test_hostile(s: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        check(s, default=DEFAULT)
        check(s, default=DEFAULT, fuzzy_with_tokens=True)


@pytest.mark.thread_unsafe(reason="changes the thread's decimal context")
def test_decimal_context() -> None:
    with decimal.localcontext() as ctx, warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.FallbackWarning)
        ctx.prec = 2
        check("10:20:30", default=DEFAULT)
        check("10.5h", default=DEFAULT)


@pytest.mark.thread_unsafe(reason="subprocesses")
@pytest.mark.parametrize("zone", ["UTC", "Europe/London", "America/New_York", "Asia/Kolkata"])
def test_local_zones(zone: str) -> None:
    import os
    import zoneinfo

    if sys.platform == "win32":
        pytest.skip("TZ takes POSIX zone names only on Unix")
    try:
        zoneinfo.ZoneInfo(zone)
    except (zoneinfo.ZoneInfoNotFoundError, ValueError):
        pytest.skip(f"no tz database entry for {zone} here")
    code = textwrap.dedent(
        """
        import datetime, time
        import dateutil.parser as du, pyroid.dateutil.parser as pp
        D = datetime.datetime(2003, 1, 25)
        names = ["UTC", "GMT", "BST", "EST", "EDT", "IST", "Z"]
        for s in [f"10:00 {n}" for n in names] + [f"2003-07-01 10:00 {n}" for n in names]:
            a, b = du.parse(s, default=D), pp.parse(s, default=D)
            ka = (a, repr(a), type(a.tzinfo), a.tzname())
            assert ka == (b, repr(b), type(b.tzinfo), b.tzname()), (s, a, b)
        print("ok", time.tzname)
        """
    )
    r = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TZ": zone},
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith("ok")


def test_parse_in_threads() -> None:
    xs = [f"Sep {1 + i % 28} 2003 {i % 24}:{i % 60:02d} PM" for i in range(2000)]
    want = [outcome(du.parse, s, default=DEFAULT) for s in xs]
    got: list[Any] = [None] * len(xs)

    def run(k: int) -> None:
        for i in range(k, len(xs), 8):
            got[i] = outcome(pp.parse, xs[i], default=DEFAULT)

    ts = [threading.Thread(target=run, args=(k,)) for k in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert got == want


# --- Ctrl-C during a batch -------------------------------------------------------------------

#: A child that raises SIGINT in itself 0.3 s into a batch, and prints how long the
#: KeyboardInterrupt took to arrive, and how long the whole batch would take (from a tenth of it).
SIGINT_CHILD = r"""
import signal, sys, threading, time
import pyroid.dateutil.parser as P
from pyroid._pyroid import _testing

case, threads, scale = sys.argv[1], int(sys.argv[2]) or None, int(sys.argv[3])
n = 1_000_000 * scale
if case == "list":
    make = lambda k: ["Sep 26 2003 10:00:00.5"] * k
    call = lambda d: P.parse_batch(d, threads=threads, errors="coerce")
elif case == "arrow":
    make = lambda k: _testing.column(["Sep 26 2003 10:00:00.5"] * k, chunks=8)
    call = lambda d: P.parse_batch(d, threads=threads)
elif case == "iso":
    make = lambda k: _testing.column(["2003-09-26T10:00:00.5+01:00"] * k, chunks=8)
    call = lambda d: P.isoparse_batch(d, threads=threads)
else:
    raise SystemExit(f"unknown case {case}")
tenth = make(n // 10)
call(tenth)
t = time.perf_counter()
call(tenth)
whole = (time.perf_counter() - t) * 10
docs = make(n)

fired = []
def fire():
    fired.append(time.perf_counter())
    signal.raise_signal(signal.SIGINT)

t0 = time.perf_counter()
timer = threading.Timer(0.3, fire)
timer.start()
done = []
try:
    call(docs)
    done.append(time.perf_counter())
    timer.cancel()
    timer.join()
except KeyboardInterrupt:
    if not done:
        print("interrupted", round(time.perf_counter() - fired[0], 3), round(whole, 3))
except BaseException as e:
    print("raised", type(e).__module__ + "." + type(e).__name__, e)
if done:
    print("finished", round(done[0] - t0, 3))
print("after", len(P.parse_batch(["2003-09-26"] * 200, threads=threads)))
"""


@pytest.mark.thread_unsafe(reason="runs subprocesses that signal themselves")
@pytest.mark.parametrize(("case", "threads"), [("list", 1), ("list", 0), ("arrow", 1), ("iso", 2)])
def test_ctrl_c_interrupts_a_batch(run_python: Any, case: str, threads: int) -> None:
    for scale in (1, 4, 16):
        code = f"import sys; sys.argv[1:] = {[case, str(threads), str(scale)]!r}\n" + SIGINT_CHILD
        proc = run_python(code, timeout=600.0)
        lines = proc.stdout.splitlines()
        # Too fast to tell (a batch that ends within a second of the signal): more input.
        if not (lines and lines[0].startswith(("finished ", "interrupted "))):
            break
        if lines[0].startswith("interrupted ") and float(lines[0].split()[2]) > 1.3:
            break
    assert proc.returncode == 0, proc.stdout + proc.stderr
    word, latency, whole = lines[0].split()
    assert word == "interrupted", lines
    # The batch checks every 50 ms (and every 1,024 rows it builds), not at its end; the rest is
    # headroom for a loaded runner and a debug build.
    assert float(latency) < min(2.0, float(whole) / 2), lines
    assert lines[1] == "after 200", lines


SIGINT_HANDLER = r"""
import signal, threading, time
import pyroid.dateutil.parser as P

hits = []
signal.signal(signal.SIGINT, lambda *a: hits.append(time.perf_counter()))
docs = ["Sep 26 2003 10:00:00.5"] * 2_000_000
threading.Timer(0.3, signal.raise_signal, (signal.SIGINT,)).start()
t0 = time.perf_counter()
out = P.parse_batch(docs, threads=1)
done = time.perf_counter()
print(len(out) == len(docs), len(hits), (hits[0] < done) if hits else None, done - t0 > 0.3)
"""


@pytest.mark.thread_unsafe(reason="runs a subprocess that signals itself")
def test_a_signal_handler_that_returns_lets_the_batch_finish(run_python: Any) -> None:
    proc = run_python(SIGINT_HANDLER, timeout=600.0)
    assert proc.returncode == 0, proc.stderr
    ok, hits, during, long_enough = proc.stdout.split()
    assert (ok, hits) == ("True", "1"), proc.stdout
    if long_enough == "True":  # the handler ran during the call, not after it
        assert during == "True", proc.stdout


# --- time zones without python-dateutil -------------------------------------------------------


def _tz_ops(z: Any) -> list[Any]:
    """What a ``tzinfo`` gives, as comparable values (``repr`` for datetimes)."""
    naive = D(2003, 9, 25, 10)
    aware = D(2003, 9, 25, 10, tzinfo=datetime.UTC)
    ops: list[Any] = [
        lambda: repr(z),
        lambda: str(z),
        lambda: z.utcoffset(naive),
        lambda: z.utcoffset(None),
        lambda: z.dst(naive),
        lambda: z.tzname(naive),
        lambda: repr(z.fromutc(naive.replace(tzinfo=z))),
        lambda: z.fromutc(naive),
        lambda: z.fromutc(5),
        lambda: repr(aware.astimezone(z)),
        lambda: repr(datetime.datetime.fromtimestamp(1_064_484_000, z)),
        lambda: z.is_ambiguous(naive),
        lambda: naive.replace(tzinfo=z) - aware,
        lambda: z == z,
        lambda: z != 5,
    ]
    return [outcome(op) for op in ops]


def test_tz_copies_behave_as_dateutils() -> None:
    from pyroid.dateutil import _tzlite

    lite: Any = _tzlite
    for ours, theirs in [
        (lite.tzutc(), tz.tzutc()),
        (lite.UTC, tz.UTC),
        (lite.tzoffset(None, 10800), tz.tzoffset(None, 10800)),
        (lite.tzoffset("BRST", -10800), tz.tzoffset("BRST", -10800)),
        (lite.tzlocal(), tz.tzlocal()),
    ]:
        assert _tz_ops(ours) == _tz_ops(theirs), ours


@pytest.mark.thread_unsafe(reason="subprocess")
def test_tz_without_dateutil(run_python: Any) -> None:
    r = run_python(
        """
        import sys
        sys.modules["dateutil"] = None  # not installed, as far as imports go
        import datetime
        import pyroid.dateutil.parser as P
        utc = datetime.datetime(2003, 9, 25, 10, tzinfo=datetime.timezone.utc)
        for s in ("2003-09-25 10:00 +0300", "2003-09-25 10:00 UTC", "2003-09-25T10:00Z"):
            z = P.parse(s).tzinfo
            print(type(z).__module__, utc.astimezone(z), datetime.datetime.fromtimestamp(0, z))
        """,
        env={"TZ": "UTC"},
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [
        "pyroid.dateutil._tzlite 2003-09-25 13:00:00+03:00 1970-01-01 03:00:00+03:00",
        "pyroid.dateutil._tzlite 2003-09-25 10:00:00+00:00 1970-01-01 00:00:00+00:00",
        "pyroid.dateutil._tzlite 2003-09-25 10:00:00+00:00 1970-01-01 00:00:00+00:00",
    ]


# --- pyroid.patch("dateutil") ------------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="patches dateutil")
def test_patch() -> None:
    import dateutil.parser
    import dateutil.parser._parser
    import dateutil.parser.isoparser

    original = (dateutil.parser.parse, dateutil.parser.isoparse)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.integrations.PatchWarning)
        pyroid.patch("dateutil")
    try:
        assert dateutil.parser.parse is not original[0]
        assert dateutil.parser._parser.parse is dateutil.parser.parse
        # dateutil.parser.isoparser is the class; the module is in sys.modules.
        assert sys.modules["dateutil.parser.isoparser"].isoparse is dateutil.parser.isoparse
        assert dateutil.parser.parse("2003-09-25 10:49") == D(2003, 9, 25, 10, 49)
        assert dateutil.parser.isoparse("2003-09-25T10:49") == D(2003, 9, 25, 10, 49)
        with pytest.raises(dateutil.parser.ParserError):
            dateutil.parser.parse("nonsense")
        # dateutil's own DEFAULTPARSER (and its parserinfo) still decides.
        info = dateutil.parser._parser.DEFAULTPARSER.info
        info._months["septembre"] = 8
        try:
            assert dateutil.parser.parse("25 septembre 2003") == D(2003, 9, 25)
        finally:
            del info._months["septembre"]
        assert dateutil.parser.parse("01/02/03", dateutil.parser.parserinfo(dayfirst=True)) == D(
            2003, 2, 1
        )
        # Pool.map(dateutil.parser.parse, ...) pickles them by reference to dateutil's modules.
        # (Round-trips objects this test made: no untrusted data.)
        for f in (dateutil.parser.parse, dateutil.parser.isoparse):
            assert pickle.loads(pickle.dumps(f)) is f
    finally:
        pyroid.unpatch("dateutil")
    assert (dateutil.parser.parse, dateutil.parser.isoparse) == original


@pytest.mark.thread_unsafe(reason="patches dateutil")
def test_patched_isoparse_type_errors_are_dateutils() -> None:
    import dateutil.parser

    want = [outcome(dateutil.parser.isoparse), outcome(du.isoparse, "2003-09-25", sep="T")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pyroid.integrations.PatchWarning)
        pyroid.patch("dateutil")
    try:
        assert outcome(dateutil.parser.isoparse) == want[0]
        assert outcome(dateutil.parser.isoparse, "2003-09-25", sep="T") == want[1]
    finally:
        pyroid.unpatch("dateutil")


# --- versions -------------------------------------------------------------------------------


@pytest.mark.thread_unsafe(reason="changes module state and warns")
def test_version_warning_and_isoparse_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from pyroid.dateutil import _upstream

    for release in pp.DATEUTIL_VERSIONS:
        monkeypatch.setattr(_upstream, "INSTALLED_VERSION", release)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ppx._version_check()
        lenient = release != "4.0.0"
        assert ppx.isoparser()._lenient is lenient
        assert ppx.isoparser(compat=release)._lenient is lenient
    # 2.8.2's and 2.9.0's isoparse is 2.9.0.post0's.
    for release in ("2.8.2", "2.9.0"):
        monkeypatch.setattr(_upstream, "INSTALLED_VERSION", release)
        assert ppx.isoparser().isoparse("2014- 4-21") == D(2014, 4, 21)
        assert ppx.isoparser().isoparse("20200728121240") == D(2020, 7, 28, 21, 24)
    # Other releases warn, and get the closer of the two forms.
    for release, lenient in (("2.8.1", True), ("2.7.5", True), ("4.1.0", False)):
        monkeypatch.setattr(_upstream, "INSTALLED_VERSION", release)
        try:
            from pyroid import _versions  # type: ignore[attr-defined]  # noqa: F401
        except ImportError:
            with pytest.warns(RuntimeWarning, match=rf"but python-dateutil {release} is installed"):
                ppx._version_check()
        assert ppx.isoparser()._lenient is lenient
    monkeypatch.setattr(_upstream, "INSTALLED_VERSION", None)
    assert ppx.isoparser()._lenient is False


FAKE_RELEASE = """
import sys
sys.path.insert(0, {site!r})
import runpy
runpy.run_path({script!r}, run_name="__main__")
"""


@pytest.mark.thread_unsafe(reason="subprocess")
@pytest.mark.parametrize("release", ["2.8.1", "2.8.2", "2.9.0"])
def test_import_under_another_release(run_python: Any, tmp_path: Any, release: str) -> None:
    # The installed release as importlib.metadata reads it (dateutil itself is 4.0.0's code).
    site = tmp_path / "site"
    meta = site / f"python_dateutil-{release}.dist-info"
    meta.mkdir(parents=True)
    (meta / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: python-dateutil\nVersion: {release}\n", encoding="utf-8"
    )
    script = tmp_path / "uses_dateutil.py"
    script.write_text(
        "import sys\nimport pyroid.dateutil.parser\n"
        "print(pyroid.dateutil.parser.isoparse('2014- 4-21'))\n",
        encoding="utf-8",
    )
    r = run_python(FAKE_RELEASE.format(site=str(site), script=str(script)))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "2014-04-21 00:00:00"
    warned = f"python-dateutil {release} is installed" in r.stderr
    assert warned is (release == "2.8.1"), r.stderr
    if warned:
        # At the line that imports pyroid.dateutil.parser, not at sys:1.
        assert f"{script}:2: " in r.stderr, r.stderr


@pytest.mark.thread_unsafe(reason="subprocess")
def test_pyroid_run(tmp_path: Any) -> None:
    pytest.importorskip("pyarrow", reason="the pyroid command line needs pyarrow (pyroid[cli])")
    script = tmp_path / "uses_dateutil.py"
    script.write_text(
        "import dateutil.parser\n"
        "from pyroid._pyroid import dateutil as native\n"
        "assert dateutil.parser.parse is native.upstream.parse\n"
        "print(dateutil.parser.parse('Thu Sep 25 10:36:28 2003'))\n",
        encoding="utf-8",
    )
    r = subprocess.run(
        [sys.executable, "-m", "pyroid", "run", "-p", "dateutil", str(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "2003-09-25 10:36:28"
