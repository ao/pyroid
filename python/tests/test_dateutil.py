"""pyroid.dateutil.parser: API, objects, batches, fallbacks and hostile inputs.

The differential checks against python-dateutil itself are in test_dateutil_parity.py.
"""

from __future__ import annotations

import datetime
import decimal
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
    assert pp.DATEUTIL_VERSIONS == ("4.0.0", "3.0.0", "2.9.0.post0")


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


def test_unexpected_keyword_is_dateutils_type_error() -> None:
    a = outcome(du.parse, "2003", fuzzzy=True)
    b = outcome(pp.parse, "2003", fuzzzy=True)
    assert a[0] == b[0] == "err"
    assert a[1] is b[1] is TypeError
    assert a[2] == b[2]


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
    pp.parserinfo.TZOFFSET["BRST"] = -10800
    try:
        assert p.parse("10:00 BRST").utcoffset() == datetime.timedelta(hours=-3)
    finally:
        del pp.parserinfo.TZOFFSET["BRST"]


class LoudInfo(ppx.parserinfo):  # type: ignore[misc]
    def convertyear(self, year: int, century_specified: bool = False) -> int:
        return int(super().convertyear(year, century_specified)) + 1


class DuLoudInfo(du.parserinfo):  # type: ignore[misc]
    def convertyear(self, year: int, century_specified: bool = False) -> int:
        return int(super().convertyear(year, century_specified)) + 1


@pytest.mark.thread_unsafe(reason="pytest.warns")
def test_parserinfo_method_override_runs_dateutils_code() -> None:
    with pytest.warns(pyroid.FallbackWarning, match="overriding convertyear"):
        b = pp.parse("01/02/03", LoudInfo())
    assert b == du.parse("01/02/03", DuLoudInfo())


class MyParser(ppx.parser):  # type: ignore[misc]
    def _parsems(self, value: str) -> tuple[int, int]:
        s, us = super()._parsems(value)
        return int(s), int(us) + 1


@pytest.mark.thread_unsafe(reason="pytest.warns")
def test_parser_hook_override_runs_dateutils_code() -> None:
    with pytest.warns(pyroid.FallbackWarning, match="overriding _parsems"):
        assert MyParser().parse("10:20:30.5").microsecond == 500_001


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
    finally:
        pyroid.unpatch("dateutil")
    assert (dateutil.parser.parse, dateutil.parser.isoparse) == original


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
    monkeypatch.setattr(_upstream, "INSTALLED_VERSION", "2.8.2")
    try:
        from pyroid import _versions  # type: ignore[attr-defined]  # noqa: F401
    except ImportError:
        with pytest.warns(RuntimeWarning, match=r"reproduces python-dateutil 4\.0\.0.*2\.8\.2"):
            ppx._version_check()
    assert ppx.isoparser()._lenient is False
    monkeypatch.setattr(_upstream, "INSTALLED_VERSION", None)
    assert ppx.isoparser()._lenient is False


@pytest.mark.thread_unsafe(reason="subprocess")
def test_pyroid_run(tmp_path: Any) -> None:
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
