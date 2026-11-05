# Derived from ftfy 6.3.1 (Apache-2.0), © Robyn Speer; modified.
"""pyroid.ftfy against the real ftfy 6.3.1: the API mirror and fixed cases.

The randomized differential tests are in test_ftfy_parity.py.
"""

# The test data is full of look-alike characters on purpose.
# ruff: noqa: RUF001

from __future__ import annotations

import inspect
import json
import sys
import threading
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

import pyroid
from pyroid import ftfy as candidate

ftfy: Any = pytest.importorskip("ftfy")
if ftfy.__version__ != "6.3.1":
    pytest.skip(f"needs ftfy 6.3.1, found {ftfy.__version__}", allow_module_level=True)

DATA = Path(__file__).resolve().parents[2] / "crates/pyroid-core/tests/ftfy_data"
CASES = json.loads((DATA / "test_cases.json").read_text("utf-8")) if DATA.is_dir() else []
needs_data = pytest.mark.skipif(not CASES, reason="test data not in this checkout")


def outcome(fn: Any, *args: Any, **kwargs: Any) -> tuple[str, Any]:
    """The result, or the exception type: both are part of parity."""
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as e:
        return ("raise", type(e).__name__)


@needs_data
@pytest.mark.parametrize("case", CASES, ids=[f"{i}" for i in range(len(CASES))])
def test_golden_corpus(case: dict[str, str]) -> None:
    orig = case["original"]
    for text in (orig, orig.encode("utf-8").decode("latin-1")):
        assert candidate.fix_text(text) == ftfy.fix_text(text)
        assert candidate.fix_and_explain(text) == ftfy.fix_and_explain(text)
        assert candidate.fix_encoding_and_explain(text) == ftfy.fix_encoding_and_explain(text)
        assert candidate.fix_text_segment(text) == ftfy.fix_text_segment(text)
    if case["expect"] == "pass":
        assert candidate.fix_text(orig) == case["fixed"]
        assert candidate.fix_encoding(orig) == case.get("fixed-encoding", case["fixed"])


@pytest.mark.parametrize(
    ("text", "kwargs", "expected"),
    [
        ("âœ” No problems", {}, "✔ No problems"),
        ("&macr;\\_(ã\x83\x84)_/&macr;", {}, "¯\\_(ツ)_/¯"),
        # ftfy's docstring says "...", but ftfy 6.3.1 itself returns U+2026.
        ("Broken text&hellip; it&#x2019;s ﬂubberiﬁc!", {}, "Broken text\u2026 it's flubberific!"),
        ("ＬＯＵＤ　ＮＯＩＳＥＳ", {}, "LOUD NOISES"),
        ("&amp;\n<html>\n&amp;", {}, "&\n<html>\n&amp;"),
        ("&amp;\n<html>\n&amp;", {"unescape_html": True}, "&\n<html>\n&"),
        ("&amp;\n<html>\n&amp;", {"unescape_html": False}, "&amp;\n<html>\n&amp;"),
        ("\001\033[36;44mfoo", {}, "foo"),
        (chr(0xD83D) + chr(0xDCA9), {}, "💩"),
        (chr(0xDCA9) + chr(0xD83D), {}, "��"),
    ],
)
def test_documented_examples(text: str, kwargs: dict[str, Any], expected: str) -> None:
    assert candidate.fix_text(text, **kwargs) == expected == ftfy.fix_text(text, **kwargs)


def test_explanations() -> None:
    assert candidate.fix_encoding_and_explain("sÃ³") == (
        "só",
        [("encode", "latin-1"), ("decode", "utf-8")],
    )
    result = candidate.fix_encoding_and_explain("voilÃ le travail")
    assert result.explanation == [
        ("encode", "latin-1"),
        ("transcode", "restore_byte_a0"),
        ("decode", "utf-8"),
    ]
    assert isinstance(result, candidate.ExplainedText)
    assert all(isinstance(s, candidate.ExplanationStep) for s in result.explanation or [])
    assert result.explanation is not None
    assert repr(result.explanation[0]) == "('encode', 'latin-1')"
    assert candidate.fix_and_explain("\001\033[36;44mfoo") == ftfy.fix_and_explain(
        "\001\033[36;44mfoo"
    )
    assert candidate.fix_and_explain("x", explain=False) == ("x", None)


@pytest.mark.parametrize(
    "name",
    [
        "fix_text",
        "fix_encoding",
        "fix_and_explain",
        "fix_encoding_and_explain",
        "fix_text_segment",
        "apply_plan",
        "guess_bytes",
        "explain_unicode",
        "fix_file",
        "ftfy",
    ],
)
def test_signature_mirrors_ftfy(name: str) -> None:
    def shape(f: Any) -> list[tuple[str, Any, Any]]:
        return [(p.name, p.kind, p.default) for p in inspect.signature(f).parameters.values()]

    assert shape(getattr(candidate, name)) == shape(getattr(ftfy, name))


def test_config_mirrors_ftfy() -> None:
    assert candidate.TextFixerConfig._fields == ftfy.TextFixerConfig._fields
    assert candidate.TextFixerConfig._field_defaults == ftfy.TextFixerConfig._field_defaults
    assert candidate.ExplanationStep._fields == ftfy.ExplanationStep._fields
    assert candidate.ExplainedText._fields == ftfy.ExplainedText._fields


def test_accepts_configs_of_either_type() -> None:
    text = "â€œmojibakeâ€\x9d &amp; ﬁ"
    for cfg in (
        ftfy.TextFixerConfig(uncurl_quotes=False),
        candidate.TextFixerConfig(uncurl_quotes=False),
    ):
        assert candidate.fix_text(text, cfg) == ftfy.fix_text(text, cfg)
        assert candidate.fix_and_explain(text, cfg) == ftfy.fix_and_explain(text, cfg)


def test_explain_false_ignores_encoding_suboptions_like_ftfy() -> None:
    # ftfy's fix_text calls fix_encoding(text) without the config when
    # explain is off, so restore_byte_a0=False has no effect there.
    text = "voilÃ le travail"
    for explain in (False, True):
        cfg = candidate.TextFixerConfig(restore_byte_a0=False, explain=explain)
        assert candidate.fix_text(text, cfg) == ftfy.fix_text(
            text, ftfy.TextFixerConfig(restore_byte_a0=False, explain=explain)
        )


def test_bytes_raise_unicode_error() -> None:
    for fn in (
        "fix_text",
        "fix_and_explain",
        "fix_encoding",
        "fix_encoding_and_explain",
        "fix_text_segment",
    ):
        with pytest.raises(UnicodeError) as got:
            getattr(candidate, fn)(b"bytes")
        with pytest.raises(UnicodeError) as want:
            getattr(ftfy, fn)(b"bytes")
        assert str(got.value) == str(want.value)


def test_unknown_option_raises_like_ftfy() -> None:
    # NamedTuple._replace raises ValueError before 3.13 and TypeError after.
    def error(fn: Any) -> tuple[type[BaseException], str]:
        with pytest.raises((ValueError, TypeError)) as e:
            fn("x", not_an_option=True)
        return type(e.value), str(e.value)

    assert error(candidate.fix_text) == error(ftfy.fix_text)
    assert error(candidate.fix_and_explain) == error(ftfy.fix_and_explain)


def test_fix_entities_is_deprecated() -> None:
    example = "&amp;\n<html>\n&amp;"
    with pytest.deprecated_call():
        assert candidate.fix_text(example, fix_entities=True) == "&\n<html>\n&"


def test_non_str_input_raises_type_error() -> None:
    for bad in (None, 3, ["a"]):
        assert outcome(candidate.fix_text, bad)[0] == "raise"
    with pytest.raises(TypeError):
        candidate.fix_text(None)  # type: ignore[arg-type]


def test_str_subclass_gives_exact_str() -> None:
    class S(str):
        pass

    for text in ("plain", "â€œ", "lone \udc00"):
        options: list[dict[str, Any]] = [{}, {"fix_surrogates": False}]
        for kwargs in options:
            got = candidate.fix_text(S(text), **kwargs)
            assert type(got) is str
            assert got == ftfy.fix_text(text, **kwargs)
            assert type(candidate.fix_text_segment(S(text), **kwargs)) is str
            assert type(candidate.fix_and_explain(S(text), **kwargs).text) is str


def test_delegates_unsupported_options_with_one_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(candidate, "_WARNED", set())
    with pytest.warns(RuntimeWarning, match="delegating to ftfy"):
        assert candidate.fix_text("abc", max_decode_length=5.0) == ftfy.fix_text(
            "abc", max_decode_length=5.0
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        candidate.fix_text("abc", max_decode_length=5.0)  # warned only once
    assert outcome(candidate.fix_text, "x", normalization="nfc") == outcome(
        ftfy.fix_text, "x", normalization="nfc"
    )


def test_delegation_without_ftfy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ftfy", None)
    with pytest.raises(pyroid.PyroidError, match="not installed"):
        candidate.fix_text("abc", normalization="XYZ")
    with pytest.raises(pyroid.PyroidError):
        candidate.guess_bytes(b"abc")
    # Everything else needs no ftfy at all.
    assert candidate.fix_text("Ã³") == "ó"


def test_max_decode_length_below_one_is_refused() -> None:
    # ftfy loops forever here; pyroid raises instead (a documented difference).
    with pytest.raises(ValueError, match="at least 1"):
        candidate.fix_text("abc", max_decode_length=0)


def test_delegated_helpers_match() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mojibake = "schÃ¶n"
        text, plan = candidate.fix_and_explain(mojibake)
        assert candidate.apply_plan(mojibake, plan) == text == "schön"  # type: ignore[arg-type]
        assert candidate.guess_bytes("ünicode".encode()) == ftfy.guess_bytes("ünicode".encode())


def test_long_strings() -> None:
    line = "Ã©tÃ© â€œquotedâ€\x9d " * 60_000  # about 1.2M code points, one segment
    for text in (line, line + "\n" + line, "x" * 2_500_000 + "Ã©"):
        assert candidate.fix_text(text) == ftfy.fix_text(text)
    short = line[:20_000]
    assert candidate.fix_text(short, max_decode_length=7) == ftfy.fix_text(
        short, max_decode_length=7
    )


def test_uses_this_interpreters_unicode_version() -> None:
    assert candidate._UNICODE_VERSION == unicodedata.unidata_version


def test_concurrent_callers() -> None:
    corpus = [
        "voilÃ  le travail",
        "crÃ¨me brÃ»lÃ©e",
        "&lt;b&gt;",
        "\U0001f600",
        "ﬁne",
        "“x”",
        chr(0xD800),
    ] * 50
    expected = [ftfy.fix_text(t) for t in corpus]
    barrier = threading.Barrier(8)

    def work(_: int) -> list[str]:
        barrier.wait()
        return [candidate.fix_text(t) for t in corpus]

    with ThreadPoolExecutor(8) as ex:
        assert all(r == expected for r in ex.map(work, range(8)))


def test_gil_stays_disabled() -> None:
    import sysconfig

    if not sysconfig.get_config_var("Py_GIL_DISABLED"):
        pytest.skip("not a free-threaded build")
    assert not getattr(sys, "_is_gil_enabled")()  # noqa: B009  (3.13+ only)
