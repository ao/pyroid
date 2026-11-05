"""pyroid.translit: API surface, examples, container kinds, identity, Arrow, errors, threads.

The contract (docs/dev/MODULES.md): ``anyascii`` and ``unidecode`` are transforms (the
caller's container kind comes back, and unchanged, i.e. pure-ASCII, documents are the
*original* objects); ``is_ascii`` is a predicate (a mask of the same kind). A single str of
>= 4096 UTF-8 bytes and batches of more than 64 documents take different code paths
(detached / thread pool) from small inputs, and strs holding lone surrogates take a WTF-8
path of their own, so each is covered on both sides. ``unidecode``'s ``errors=`` modes and
surrogate warnings have their own suite; here it runs in its default mode, where it gives
the same output as ``anyascii``. Parity with the ``anyascii`` package is in
test_translit_parity.py.
"""

import inspect
import subprocess
from collections.abc import Callable, Sequence
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

import pyroid
import pyroid.translit as tr
from pyroid._pyroid import _testing

#: Thresholds in the bindings: a single str this long (in UTF-8 bytes) is
#: processed detached; a batch longer than this uses the thread pool.
DETACH_BYTES = 4096
INLINE_BATCH = 64

#: The two transforms. Without lone surrogates, unidecode's default mode
#: (errors="ignore") gives exactly anyascii's output.
TRANSFORMS: list[Callable[..., Any]] = [tr.anyascii, tr.unidecode]
transforms = pytest.mark.parametrize("fn", TRANSFORMS, ids=["anyascii", "unidecode"])
ALL_FUNCS: list[Callable[..., Any]] = [tr.anyascii, tr.unidecode, tr.is_ascii]
all_funcs = pytest.mark.parametrize("fn", ALL_FUNCS, ids=["anyascii", "unidecode", "is_ascii"])

# (input, anyascii output). Every non-ASCII character here either has a
# mapping or is one anyascii removes (combining marks, joiners, private use,
# noncharacters, C1 controls), so unidecode's default mode agrees.
EXAMPLES: list[tuple[str, str]] = [
    (
        "\u5317\u4eac \u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac \U0001f600 caf\xe9",
        "BeiJing Ellinika :grinning: cafe",
    ),
    ("\u041f\u0440\u0438\u0432\u0435\u0442", "Privet"),
    ("\xdf", "ss"),
    ("\xc6", "Ae"),
    ("€", "EUR"),
    ("\xbd", "1/2"),
    ("\u2026", "..."),
    ("\u201cquoted\u201d", '"quoted"'),
    ("\xa0", " "),
    ("a\u0301", "a"),  # a combining mark maps to ""
    ("한국어", "HanGugEo"),
    ("日本語", "RiBenYu"),
    ("ภาษาไทย", "phasaaithy"),
    ("☃", ":snowman2:"),
    ("\U0001f44d\U0001f3fd", ":thumbsup:"),  # the skin-tone modifier maps to ""
    ("\ufb01", "fi"),
    ("\uff37", "W"),
    ("\u216b", "XII"),
    ("\u0149", "'n"),
    ("a\u200db", "ab"),
    ("\ufeffa", "a"),
    ("\u0640", ""),  # Arabic tatweel
    ("\ue000", ""),  # private use: no mapping
    ("\U0010ffff", ""),  # noncharacter: no mapping
    ("\x80", ""),  # C1 control: no mapping
    ("", ""),
    ("plain ascii\n\t\x00\x7f", "plain ascii\n\t\x00\x7f"),
]
EXAMPLE_IDS = [f"ex{i}" for i in range(len(EXAMPLES))]
NON_ASCII_EXAMPLES = [(s, w) for s, w in EXAMPLES if not s.isascii()]

# (input with lone surrogates, anyascii output): anyascii drops them, and a
# high/low pair stored as two code points is two lone surrogates, not a char.
SURROGATE_CASES: list[tuple[str, str]] = [
    ("a\ud800b", "ab"),
    ("\udfff", ""),
    ("\ud83d\ude00", ""),
    ("\xe9\udc80日", "eRi"),
    ("\ud800\ud800\ud800x", "x"),
    ("\U0001f600\udbff", ":grinning:"),
    ("a" * DETACH_BYTES + "\udc00\xe9", "a" * DETACH_BYTES + "e"),
]
SURROGATE_IDS = [f"surrogates{i}" for i in range(len(SURROGATE_CASES))]


class MyStr(str):
    """A str subclass: accepted as a document, returned as itself if unchanged."""

    __slots__ = ()


def _notes(exc: BaseException) -> str:
    return "\n".join(getattr(exc, "__notes__", []))


def mixed_docs(n: int) -> tuple[list[str], list[str]]:
    """``n`` documents (ASCII at even indexes only) and their anyascii outputs."""
    docs: list[str] = []
    want: list[str] = []
    for i in range(n):
        if i % 2 == 0:
            docs.append(f"doc {i}")
            want.append(f"doc {i}")
        else:
            src, out = NON_ASCII_EXAMPLES[i % len(NON_ASCII_EXAMPLES)]
            docs.append(f"{src} {i}")
            want.append(f"{out} {i}")
    return docs, want


# ---- module surface --------------------------------------------------------


def test_module_surface() -> None:
    assert tr.__all__ == [
        "ANYASCII_VERSION",
        "UnidecodeError",
        "anyascii",
        "is_ascii",
        "unidecode",
        "unidecode_expect_ascii",
        "unidecode_expect_nonascii",
    ]
    assert tr.ANYASCII_VERSION == "0.3.3"
    for fn in ALL_FUNCS:
        assert fn.__module__ == "pyroid.translit"
        assert fn.__doc__
    assert tr.unidecode_expect_ascii is tr.unidecode
    assert tr.unidecode_expect_nonascii is tr.unidecode


def test_signatures() -> None:
    assert str(inspect.signature(tr.anyascii)) == "(string, *, threads=None)"
    assert str(inspect.signature(tr.is_ascii)) == "(string, *, threads=None)"
    assert (
        str(inspect.signature(tr.unidecode))
        == "(string, errors='ignore', replace_str='?', *, threads=None)"
    )


@all_funcs
def test_string_keyword(fn: Callable[..., Any]) -> None:
    want = False if fn is tr.is_ascii else "e"
    assert fn(string="\xe9") == want
    assert fn(string=["\xe9"]) == [want]
    assert fn(string="\xe9", threads=1) == want


@all_funcs
def test_threads_is_keyword_only(fn: Callable[..., Any]) -> None:
    extra = ("ignore", "?") if fn is tr.unidecode else ()
    with pytest.raises(TypeError):
        fn("a", *extra, 2)
    with pytest.raises(TypeError):
        fn()


def test_results_are_plain_python_objects() -> None:
    assert not isinstance(tr.anyascii(["\xe9"]), pyroid.Column)
    assert not isinstance(tr.is_ascii(["\xe9"]), pyroid.Column)


@pytest.mark.thread_unsafe(reason="runs a subprocess")
def test_import_pyroid_does_not_import_translit(
    run_python: Callable[..., subprocess.CompletedProcess[str]],
) -> None:
    proc = run_python(
        """
        import sys, pyroid
        assert "pyroid.translit" not in sys.modules, sorted(sys.modules)
        import pyroid.translit as tr
        assert tr.anyascii("caf\\xe9") == "cafe"
        assert tr.unidecode(["\\u5317\\u4eac"]) == ["BeiJing"]
        print("ok")
        """
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


# ---- examples --------------------------------------------------------------


def test_readme_example() -> None:
    doc = "北京 Ελληνικά 😀 café"
    assert tr.anyascii(doc) == "BeiJing Ellinika :grinning: cafe"
    assert tr.unidecode(doc) == "BeiJing Ellinika :grinning: cafe"


@transforms
@pytest.mark.parametrize(("doc", "want"), EXAMPLES, ids=EXAMPLE_IDS)
def test_examples(fn: Callable[..., Any], doc: str, want: str) -> None:
    out = fn(doc)
    assert out == want
    assert out.isascii()
    assert fn([doc]) == [want]
    assert fn((doc,)) == (want,)
    assert tr.is_ascii(doc) is doc.isascii()


@transforms
def test_every_ascii_char_is_kept(fn: Callable[..., Any]) -> None:
    ascii_chars = "".join(map(chr, range(128)))
    assert fn(ascii_chars) is ascii_chars
    assert fn(list(ascii_chars)) == list(ascii_chars)
    assert tr.is_ascii(list(ascii_chars)) == [True] * 128


# ---- container kinds -------------------------------------------------------


@transforms
def test_str_in_str_out(fn: Callable[..., Any]) -> None:
    out = fn("caf\xe9")
    assert type(out) is str
    assert out == "cafe"


@transforms
@pytest.mark.parametrize("n", [0, 1, INLINE_BATCH, INLINE_BATCH + 1, 1000])
def test_list_in_list_out(fn: Callable[..., Any], n: int) -> None:
    docs, want = mixed_docs(n)
    out = fn(docs)
    assert type(out) is list
    assert out == want
    assert all(type(o) is str for o in out)


@transforms
@pytest.mark.parametrize("n", [0, 1, INLINE_BATCH, INLINE_BATCH + 1, 1000])
def test_tuple_in_tuple_out(fn: Callable[..., Any], n: int) -> None:
    docs, want = mixed_docs(n)
    out = fn(tuple(docs))
    assert type(out) is tuple
    assert out == tuple(want)


@pytest.mark.parametrize("n", [0, 1, INLINE_BATCH, INLINE_BATCH + 1, 1000])
def test_is_ascii_kinds(n: int) -> None:
    docs, _ = mixed_docs(n)
    want = [d.isascii() for d in docs]
    mask = tr.is_ascii(docs)
    assert type(mask) is list
    assert mask == want
    assert all(type(m) is bool for m in mask)
    tmask = tr.is_ascii(tuple(docs))
    assert type(tmask) is tuple
    assert tmask == tuple(want)


def test_is_ascii_str_gives_the_bool_singletons() -> None:
    assert tr.is_ascii("abc") is True
    assert tr.is_ascii("") is True
    assert tr.is_ascii("\x7f") is True
    assert tr.is_ascii("\x80") is False
    assert tr.is_ascii("a\ud800") is False
    assert tr.is_ascii(MyStr("abc")) is True


def test_is_ascii_ignores_an_isascii_override() -> None:
    class Liar(str):
        __slots__ = ()

        def isascii(self) -> bool:
            return not str.isascii(self)

    assert tr.is_ascii(Liar("\xe9")) is False
    assert tr.is_ascii([Liar("a"), Liar("\xe9")]) == [True, False]
    assert tr.is_ascii((Liar("a"),)) == (True,)


@all_funcs
def test_input_is_not_modified(fn: Callable[..., Any]) -> None:
    docs = ["caf\xe9", "ok", "a\ud800"]
    fn(docs[:2])
    if fn is not tr.unidecode:  # unidecode warns on the surrogate (tested below)
        fn(docs)
    assert docs == ["caf\xe9", "ok", "a\ud800"]


@pytest.mark.parametrize(
    ("fn", "surrogate"),
    [
        (tr.anyascii, False),
        (tr.unidecode, False),
        (tr.is_ascii, False),
        # unidecode warns on lone surrogates: see test_unidecode_warns_on_a_lone_surrogate.
        (tr.anyascii, True),
        (tr.is_ascii, True),
    ],
    ids=["anyascii", "unidecode", "is_ascii", "anyascii-wtf8", "is_ascii-wtf8"],
)
def test_list_and_tuple_subclasses_are_accepted(fn: Callable[..., Any], surrogate: bool) -> None:
    class MyList(list[str]):
        pass

    class MyTuple(tuple[str, ...]):
        __slots__ = ()

    docs = ["caf\xe9", "ok", "x\udc00" if surrogate else "x"]
    want: list[Any] = [d.isascii() for d in docs] if fn is tr.is_ascii else ["cafe", "ok", "x"]
    out = fn(MyList(docs))
    assert isinstance(out, list)
    assert list(out) == want
    tout = fn(MyTuple(docs))
    assert isinstance(tout, tuple)
    assert tuple(tout) == tuple(want)


# ---- identity of unchanged documents ---------------------------------------


@transforms
@pytest.mark.parametrize(
    "doc",
    ["", "a", "plain words", "\x00\x7f", "word " * 1000, MyStr("sub"), MyStr("x" * 5000)],
    ids=["empty", "a", "words", "controls", "long", "subclass", "long-subclass"],
)
def test_ascii_str_is_the_same_object(fn: Callable[..., Any], doc: str) -> None:
    assert fn(doc) is doc


@transforms
@pytest.mark.parametrize(
    "doc",
    [
        "\xe9",
        "a\u0301",
        "\u200b",
        "\U0001f600",
        "caf\xe9 " * 1000,
        MyStr("\xe9"),
        MyStr("\xe9" * 3000),
    ],
    ids=["e-acute", "combining", "zwsp", "emoji", "long", "subclass", "long-subclass"],
)
def test_non_ascii_str_is_a_new_exact_str(fn: Callable[..., Any], doc: str) -> None:
    out = fn(doc)
    assert out is not doc
    assert type(out) is str
    assert out.isascii()


@transforms
@pytest.mark.parametrize("nbytes", [DETACH_BYTES - 1, DETACH_BYTES, DETACH_BYTES + 1, 100_000])
def test_single_str_on_both_sides_of_the_detach_threshold(
    fn: Callable[..., Any], nbytes: int
) -> None:
    unchanged = "a" * nbytes
    assert fn(unchanged) is unchanged
    assert tr.is_ascii(unchanged) is True
    changed = "a" * (nbytes - 2) + "\xe9"
    assert len(changed.encode()) == nbytes
    assert fn(changed) == "a" * (nbytes - 2) + "e"
    assert tr.is_ascii(changed) is False
    head = "日" + "a" * (nbytes - 3)  # 3 bytes, at the start this time
    assert len(head.encode()) == nbytes
    assert fn(head) == "Ri" + "a" * (nbytes - 3)


@transforms
@pytest.mark.parametrize("nchars", [DETACH_BYTES // 2 - 1, DETACH_BYTES // 2, DETACH_BYTES])
def test_threshold_counts_bytes_not_chars(fn: Callable[..., Any], nchars: int) -> None:
    doc = "\xe9" * nchars  # 2 UTF-8 bytes per char
    assert fn(doc) == "e" * nchars
    assert tr.is_ascii(doc) is False


@transforms
@pytest.mark.parametrize("n", [1, INLINE_BATCH, INLINE_BATCH + 1, 5000])
@pytest.mark.parametrize("kind", [list, tuple], ids=["list", "tuple"])
def test_batch_identity(fn: Callable[..., Any], kind: Callable[[list[str]], Any], n: int) -> None:
    raw, want = mixed_docs(n)
    docs = kind(raw)
    out = fn(docs)
    assert len(out) == n
    for i, (o, d, w) in enumerate(zip(out, docs, want, strict=True)):
        if i % 2 == 0:
            assert o is d, i
        else:
            assert o is not d, i
            assert type(o) is str, i
            assert o == w, i


@transforms
def test_same_object_repeated_in_a_batch(fn: Callable[..., Any]) -> None:
    s, u = "keep me", "change m\xe9"
    out = fn([s, u, s, u] * 50)
    assert all(o is s for o in out[::2])
    assert out[1::2] == ["change me"] * 100


@transforms
@pytest.mark.parametrize("n", [1, INLINE_BATCH + 1])
def test_str_subclass_documents(fn: Callable[..., Any], n: int) -> None:
    keep = MyStr("cafe")
    change = MyStr("caf\xe9")
    batch = fn([keep, change] * n)
    assert all(o is keep for o in batch[::2])
    assert all(type(o) is str for o in batch[1::2])
    assert batch[1::2] == ["cafe"] * n
    assert tr.is_ascii([keep, change] * n) == [True, False] * n


# ---- lone surrogates -------------------------------------------------------


@pytest.mark.parametrize(("doc", "want"), SURROGATE_CASES, ids=SURROGATE_IDS)
def test_anyascii_drops_lone_surrogates(doc: str, want: str) -> None:
    out = tr.anyascii(doc)
    assert type(out) is str
    assert out == want
    assert tr.anyascii([doc]) == [want]
    assert tr.anyascii((doc,)) == (want,)
    sub = tr.anyascii(MyStr(doc))
    assert type(sub) is str
    assert sub == want
    assert tr.is_ascii(doc) is False


@pytest.mark.parametrize("n", [1, INLINE_BATCH + 1, 1000])
@pytest.mark.parametrize("kind", [list, tuple], ids=["list", "tuple"])
def test_surrogates_in_a_mixed_batch(kind: Callable[[list[str]], Any], n: int) -> None:
    keep, keep_sub = "plain", MyStr("plain sub")
    raw: list[str] = []
    want: list[str] = []
    for i in range(n):
        pick = i % 5
        if pick == 0:
            raw.append(keep)
            want.append(keep)
        elif pick == 1:
            raw.append(keep_sub)
            want.append(keep_sub)
        elif pick == 2:
            src, out = SURROGATE_CASES[i % len(SURROGATE_CASES)]
            raw.append(src)
            want.append(out)
        elif pick == 3:
            src, out = NON_ASCII_EXAMPLES[i % len(NON_ASCII_EXAMPLES)]
            raw.append(src)
            want.append(out)
        else:
            raw.append(MyStr(f"\udcff{i}"))
            want.append(str(i))
    docs = kind(raw)
    out = tr.anyascii(docs)
    assert type(out) is kind
    assert list(out) == want
    for i, (o, d) in enumerate(zip(out, docs, strict=True)):
        if d.isascii():
            assert o is d, i
        else:
            assert type(o) is str, i
    assert tr.is_ascii(docs) == kind([d.isascii() for d in raw])


@pytest.mark.parametrize("threads", [None, 1, 2, 4])
def test_surrogate_batches_and_threads(threads: int | None) -> None:
    raw = [src for src, _ in SURROGATE_CASES] * 300 + ["ascii"] * 100
    want = [w for _, w in SURROGATE_CASES] * 300 + ["ascii"] * 100
    out = tr.anyascii(raw, threads=threads)
    assert out == want
    assert all(o is d for o, d in zip(out[-100:], raw[-100:], strict=True))


@pytest.mark.thread_unsafe(reason="pytest.warns")
def test_unidecode_warns_on_a_lone_surrogate() -> None:
    with pytest.warns(RuntimeWarning, match="Surrogate character"):
        out = tr.unidecode("a\ud800b")
    assert out == "ab"


# ---- Arrow without pyarrow -------------------------------------------------

ARROW_VALUES: list[str | None] = [
    "caf\xe9",
    None,
    "plain",
    "",
    "北京",
    None,
    "\U0001f600 " * 20,
    "",
]
ARROW_WANT: list[str | None] = [
    "cafe",
    None,
    "plain",
    "",
    "BeiJing",
    None,
    ":grinning: " * 20,
    "",
]


@transforms
@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_arrow_without_pyarrow(fn: Callable[..., Any], large: bool, chunks: int) -> None:
    col = _testing.column(ARROW_VALUES, large=large, chunks=chunks, name="text")
    out = fn(col)
    assert isinstance(out, pyroid.Column)
    assert out.type == ("large_string" if large else "string")
    assert out.name == "text"
    assert out.null_count == 2
    assert out.num_chunks == chunks
    assert len(out) == len(ARROW_VALUES)
    assert out.to_pylist() == ARROW_WANT
    # A Column goes straight back in; its documents are all ASCII now.
    assert fn(out).to_pylist() == ARROW_WANT


@pytest.mark.parametrize("large", [False, True])
@pytest.mark.parametrize("chunks", [1, 3])
def test_is_ascii_arrow_without_pyarrow(large: bool, chunks: int) -> None:
    col = _testing.column(ARROW_VALUES, large=large, chunks=chunks, name="text")
    mask = tr.is_ascii(col)
    assert isinstance(mask, pyroid.Column)
    assert mask.type == "bool"
    assert mask.name == "text"
    assert mask.null_count == 2
    assert mask.to_pylist() == [None if v is None else v.isascii() for v in ARROW_VALUES]


@all_funcs
def test_arrow_empty_and_all_null(fn: Callable[..., Any]) -> None:
    assert fn(_testing.column([])).to_pylist() == []
    out = fn(_testing.column([None, None, None], chunks=2))
    assert out.to_pylist() == [None, None, None]
    assert out.null_count == 3


@all_funcs
@pytest.mark.parametrize("n", [INLINE_BATCH, INLINE_BATCH + 1, 5000])
def test_arrow_batches_on_both_sides_of_the_pool_threshold(fn: Callable[..., Any], n: int) -> None:
    docs, want = mixed_docs(n)
    values: list[str | None] = [None if i % 7 == 3 else d for i, d in enumerate(docs)]
    if fn is tr.is_ascii:
        expect: list[Any] = [None if v is None else v.isascii() for v in values]
    else:
        expect = [None if v is None else w for v, w in zip(values, want, strict=True)]
    for large in (False, True):
        assert fn(_testing.column(values, large=large, chunks=2)).to_pylist() == expect


@transforms
def test_arrow_long_row(fn: Callable[..., Any]) -> None:
    long = "caf\xe9 " * 2000
    col = _testing.column(["x", long, None])
    assert fn(col).to_pylist() == ["x", "cafe " * 2000, None]


# ---- Arrow with pyarrow ----------------------------------------------------


def _pa_arrays(pa: Any, values: list[str | None]) -> list[tuple[str, Any, str]]:
    """(id, array, expected output type) for each Arrow layout pyroid takes."""
    skip = "sk\xefp"
    return [
        ("string", pa.array(values, pa.string()), "string"),
        ("large_string", pa.array(values, pa.large_string()), "large_string"),
        ("string_view", pa.array(values, pa.string_view()), "string"),
        ("chunked", pa.chunked_array([values[:3], [], values[3:]], pa.string()), "string"),
        (
            "chunked-large",
            pa.chunked_array([values[:2], values[2:]], pa.large_string()),
            "large_string",
        ),
        ("sliced", pa.array(["skip", skip, *values], pa.string())[2:], "string"),
        ("sliced-large", pa.array([skip, *values], pa.large_string())[1:], "large_string"),
        ("sliced-view", pa.array([skip, *values], pa.string_view())[1:], "string"),
        (
            "chunked-slices",
            pa.chunked_array(
                [pa.array([skip, *values[:4]])[1:], pa.array(["s", skip, *values[4:]])[2:]]
            ),
            "string",
        ),
    ]


@transforms
def test_arrow_with_pyarrow(pa: Any, fn: Callable[..., Any]) -> None:
    for name, arr, out_type in _pa_arrays(pa, ARROW_VALUES):
        out = fn(arr)
        assert isinstance(out, pyroid.Column), name
        assert out.type == out_type, name
        assert out.null_count == 2, name
        assert out.to_pylist() == ARROW_WANT, name
        assert pa.chunked_array(out).to_pylist() == ARROW_WANT, name
        assert pa.chunked_array(out).type == getattr(pa, out_type)(), name
        assert pa.array(out).to_pylist() == ARROW_WANT, name


def test_is_ascii_with_pyarrow(pa: Any) -> None:
    want = [None if v is None else v.isascii() for v in ARROW_VALUES]
    for name, arr, _ in _pa_arrays(pa, ARROW_VALUES):
        mask = tr.is_ascii(arr)
        assert mask.type == "bool", name
        assert mask.null_count == 2, name
        assert mask.to_pylist() == want, name
        assert pa.chunked_array(mask).type == pa.bool_(), name
        assert pa.chunked_array(mask).to_pylist() == want, name


@all_funcs
def test_pyarrow_empty_inputs(pa: Any, fn: Callable[..., Any]) -> None:
    for arr in (
        pa.array([], pa.string()),
        pa.array([], pa.large_string()),
        pa.array([], pa.string_view()),
        pa.chunked_array([], pa.string()),
        pa.array(["\xe9"])[1:],
    ):
        assert fn(arr).to_pylist() == []


@all_funcs
def test_pyarrow_wrong_types(pa: Any, fn: Callable[..., Any]) -> None:
    for arr in (
        pa.array([1, 2]),
        pa.array([b"a"], pa.binary()),
        pa.array(["a"]).dictionary_encode(),
        pa.table({"text": ["a"]}),
    ):
        with pytest.raises(pyroid.InputError):
            fn(arr)


def _binary_as(pa: Any, values: Sequence[bytes | None], type_name: str) -> Any:
    """Arrow string data that is not valid UTF-8 (a binary array viewed as strings)."""
    binary = {"string": "binary", "large_string": "large_binary", "string_view": "binary_view"}
    arr = pa.array(values, type=getattr(pa, binary[type_name])())
    return arr.view(getattr(pa, type_name)())


@all_funcs
@pytest.mark.parametrize("type_name", ["string", "large_string", "string_view"])
@pytest.mark.parametrize(
    "bad", [b"\xff", b"caf\xc3", b"\xed\xa0\x80", b"long enough \xc3\xa9 \xfe"]
)
def test_invalid_utf8_in_arrow(pa: Any, fn: Callable[..., Any], type_name: str, bad: bytes) -> None:
    arr = _binary_as(pa, [b"ok", bad, b"fine"], type_name)
    with pytest.raises(UnicodeDecodeError) as info:
        fn(arr)
    assert "row 1 of chunk 0" in _notes(info.value)


@all_funcs
def test_invalid_utf8_in_a_later_chunk(pa: Any, fn: Callable[..., Any]) -> None:
    good = pa.array(["caf\xe9"] * 100)
    bad = _binary_as(pa, [b"ok"] * 70 + [b"\x80"], "string")
    with pytest.raises(UnicodeDecodeError) as info:
        fn(pa.chunked_array([good, bad]))
    assert "row 70 of chunk 1" in _notes(info.value)


# ---- errors ----------------------------------------------------------------


@all_funcs
@pytest.mark.parametrize(
    "bad",
    [3, None, 1.5, b"bytes", bytearray(b"x"), memoryview(b"x"), {"a"}, {"a": 1}, object()],
    ids=["int", "None", "float", "bytes", "bytearray", "memoryview", "set", "dict", "object"],
)
def test_wrong_input_types(fn: Callable[..., Any], bad: Any) -> None:
    with pytest.raises(pyroid.InputError) as info:
        fn(bad)
    assert isinstance(info.value, TypeError)
    assert isinstance(info.value, pyroid.PyroidError)
    if isinstance(bad, bytes | bytearray | memoryview):
        assert "decode" in str(info.value)


@all_funcs
@pytest.mark.parametrize("kind", [list, tuple], ids=["list", "tuple"])
@pytest.mark.parametrize("item", [3, None, b"x", ["a"]], ids=["int", "None", "bytes", "list"])
def test_non_str_item_names_the_index(
    fn: Callable[..., Any], kind: Callable[[list[Any]], Any], item: Any
) -> None:
    with pytest.raises(pyroid.InputError, match="index 2"):
        fn(kind(["a", "\xe9", item, "b"]))
    big: list[Any] = ["a"] * (INLINE_BATCH + 10)
    big[INLINE_BATCH + 5] = item
    with pytest.raises(pyroid.InputError, match=f"index {INLINE_BATCH + 5}"):
        fn(kind(big))


@pytest.mark.parametrize("fn", [tr.anyascii, tr.is_ascii], ids=["anyascii", "is_ascii"])
def test_non_str_item_with_lone_surrogates(fn: Callable[..., Any]) -> None:
    after: list[Any] = ["a\ud800", "ok", 3]
    with pytest.raises(pyroid.InputError, match="index 2"):
        fn(after)
    before: list[Any] = ["ok", None, "a\ud800"]
    with pytest.raises(pyroid.InputError, match="index 1"):
        fn(tuple(before))


@all_funcs
@pytest.mark.parametrize("threads", [0, -1, -(2**70)])
@pytest.mark.parametrize("arg", ["a", ["\xe9"], ("\xe9",)], ids=["str", "list", "tuple"])
def test_bad_thread_counts(fn: Callable[..., Any], threads: int, arg: Any) -> None:
    with pytest.raises(ValueError, match="threads") as info:
        fn(arg, threads=threads)
    assert not isinstance(info.value, TypeError)


@all_funcs
@pytest.mark.parametrize("threads", ["2", 1.5, b"1"], ids=["str", "float", "bytes"])
def test_wrong_thread_types(fn: Callable[..., Any], threads: Any) -> None:
    with pytest.raises(pyroid.InputError, match="threads"):
        fn(["\xe9"], threads=threads)


@pytest.mark.parametrize("fn", [tr.anyascii, tr.is_ascii], ids=["anyascii", "is_ascii"])
def test_bad_threads_with_lone_surrogates(fn: Callable[..., Any]) -> None:
    with pytest.raises(ValueError, match="threads"):
        fn(["a\ud800"], threads=0)


@all_funcs
def test_arrow_producer_errors_propagate(fn: Callable[..., Any]) -> None:
    class Producer:
        def __init__(self, exc: Exception) -> None:
            self.exc = exc

        def __arrow_c_stream__(self, requested_schema: object | None = None) -> object:
            raise self.exc

    with pytest.raises(ZeroDivisionError):
        fn(Producer(ZeroDivisionError("boom")))
    # Even the error type that sends strs down the lone-surrogate path.
    with pytest.raises(UnicodeEncodeError, match="boom"):
        fn(Producer(UnicodeEncodeError("utf-8", "x", 0, 1, "boom")))


# ---- threads ---------------------------------------------------------------


def _thread_docs() -> tuple[list[str], list[str]]:
    docs, want = mixed_docs(3000)
    long_src, long_out = "caf\xe9 北京 " * 600, "cafe BeiJing " * 600
    return [*docs, long_src, "x" * 9000], [*want, long_out, "x" * 9000]


@transforms
@pytest.mark.parametrize("threads", [None, 1, 2, 4])
def test_threads_give_identical_results(fn: Callable[..., Any], threads: int | None) -> None:
    docs, want = _thread_docs()
    assert fn(docs, threads=threads) == want
    assert fn(tuple(docs), threads=threads) == tuple(want)
    col = _testing.column(docs, chunks=3)
    assert fn(col, threads=threads).to_pylist() == want


@pytest.mark.parametrize("threads", [None, 1, 2, 4])
def test_is_ascii_threads(threads: int | None) -> None:
    docs, _ = _thread_docs()
    want = [d.isascii() for d in docs]
    assert tr.is_ascii(docs, threads=threads) == want
    assert tr.is_ascii(_testing.column(docs, chunks=3), threads=threads).to_pylist() == want


@all_funcs
@pytest.mark.timeout(120)
def test_twenty_thousand_documents(fn: Callable[..., Any]) -> None:
    docs, want = mixed_docs(20_000)
    if fn is tr.is_ascii:
        want_mask = [d.isascii() for d in docs]
        assert fn(docs) == want_mask
        assert fn(_testing.column(docs, chunks=4)).to_pylist() == want_mask
    else:
        assert fn(docs) == want
        assert fn(_testing.column(docs, large=True, chunks=4)).to_pylist() == want


# ---- properties ------------------------------------------------------------

# Any code point, lone surrogates included, with extra weight on characters
# anyascii maps to "" or has no mapping for.
any_char = st.one_of(
    st.characters(),
    st.characters(categories=["Cs"]),
    st.sampled_from(["\u0301", "\u200d", "\ufe0f", "\U0001f3fd", "\ue000", "\x80", "\u0640"]),
)
any_text = st.text(any_char, max_size=30)
utf8_text = st.text(st.characters(codec="utf-8"), max_size=30)


@given(docs=st.lists(any_text, max_size=12))
def test_is_ascii_matches_str_isascii(docs: list[str]) -> None:
    want = [d.isascii() for d in docs]
    assert tr.is_ascii(docs) == want
    assert tr.is_ascii(tuple(docs)) == tuple(want)
    assert tr.is_ascii(docs * 7) == want * 7  # past the pool threshold when docs has 10+
    for d in docs:
        assert tr.is_ascii(d) is d.isascii()


@given(docs=st.lists(st.none() | utf8_text, max_size=12), large=st.booleans())
def test_is_ascii_arrow_matches_str_isascii(docs: list[str | None], large: bool) -> None:
    col = _testing.column(docs, large=large, chunks=2 if len(docs) > 1 else 1)
    assert tr.is_ascii(col).to_pylist() == [None if d is None else d.isascii() for d in docs]


@given(docs=st.lists(st.none() | utf8_text, max_size=12))
def test_is_ascii_pyarrow_matches_str_isascii(pa: Any, docs: list[str | None]) -> None:
    want = [None if d is None else d.isascii() for d in docs]
    for arr in (pa.array(docs, pa.string()), pa.array(docs, pa.string_view())):
        assert tr.is_ascii(arr).to_pylist() == want


@given(docs=st.lists(any_text, max_size=12))
def test_anyascii_output_is_ascii_and_ascii_input_is_kept(docs: list[str]) -> None:
    out = tr.anyascii(docs)
    assert type(out) is list
    assert len(out) == len(docs)
    for o, d in zip(out, docs, strict=True):
        assert o.isascii()
        if d.isascii():
            assert o is d
        else:
            assert type(o) is str
            assert o is not d
        assert tr.anyascii(o) is o
        assert tr.anyascii(d) == o


@given(docs=st.lists(utf8_text, max_size=12))
def test_unidecode_default_is_anyascii_without_surrogates(docs: list[str]) -> None:
    out = tr.unidecode(docs)
    assert out == tr.anyascii(docs)
    for o, d in zip(out, docs, strict=True):
        assert o.isascii()
        if d.isascii():
            assert o is d
