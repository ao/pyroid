"""pyroid.nemo's string modifiers vs NeMo Curator 1.4.0's ``DocumentModifier`` code.

The reference functions below are NeMo Curator 1.4.0's ``modify_document`` bodies (Apache-2.0,
© NVIDIA CORPORATION), copied so the tests run where NeMo does not install (Python 3.14+). When
``nemo_curator`` is importable, the same properties also run against NeMo's own classes.
"""

from __future__ import annotations

import re
import sys
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import pyroid.nemo as nemo

# --- NeMo Curator 1.4.0 reference (nemo_curator/stages/text/modifiers/string/*.py) -------------

POLICY = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "privacy overview",
    "use of cookies",
    "use cookies",
    "privacy & cookies policy",
    "privacy and cookies policy",
    "this website uses cookies to improve your experience while you navigate through the "
    "website. out of these cookies, the cookies that are categorized as necessary are stored "
    "on your browser as they are essential for the working of basic functionalities of the "
    "website. we also use third-party cookies that help us analyze and understand how you use "
    "this website. these cookies will be stored in your browser only with your consent. you "
    "also have the option to opt-out of these cookies. but opting out of some of these "
    "cookies may have an effect on your browsing experience.",
    "necessary cookies are absolutely essential for the website to function properly. this "
    "category only includes cookies that ensures basic functionalities and security features "
    "of the website. these cookies do not store any personal information.",
    "any cookies that may not be particularly necessary for the website to function and is "
    "used specifically to collect user personal data via analytics, ads, other embedded "
    "contents are termed as non-necessary cookies. it is mandatory to procure user consent "
    "prior to running these cookies on your website.",
    "this site uses cookies, including for analytics, personalization, and advertising "
    "purposes. for more information or to change your cookie settings, click here.",
    "if you continue to browse this site without changing your cookie settings, you agree to "
    "this use. acceptread more",
]


def _top_or_bottom_only(indices: list[int], n: int) -> bool:
    if len(indices) == n:
        return False
    contiguous = all(indices[i] + 1 == indices[i + 1] for i in range(len(indices) - 1))
    return contiguous and (indices[0] == 0 or indices[-1] == n - 1)


def ref_boilerplate(text: str, remove_if_at_top_or_bottom: bool = True) -> str:
    indices = []
    paragraphs = text.split("\n\n")
    for idx, paragraph in enumerate(paragraphs):
        p = paragraph.strip().lower()
        if "lorem ipsum" in p:
            return ""
        if any(s in p for s in POLICY):
            if not remove_if_at_top_or_bottom:
                return ""
            indices.append(idx)
    if not indices:
        return text
    if _top_or_bottom_only(indices, len(paragraphs)):
        return "\n\n".join(p for i, p in enumerate(paragraphs) if i not in indices)
    return text


def ref_markdown(text: str) -> str:
    out = []
    for line in text.split("\n"):
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"\*(.*?)\*", r"\1", line)
        line = re.sub(r"_(.*?)_", r"\1", line)
        line = re.sub(r"\[.*?\]\((.*?)\)", r"\1", line)
        out.append(line)
    return "\n".join(out)


def ref_newlines(text: str) -> str:
    text = re.sub(r"(\n){3,}", "\n\n", text)
    return re.sub(r"(\r\n){3,}", "\r\n\r\n", text)


def ref_quotation(text: str) -> str:
    if len(text.strip()) > 2 and text[0] == '"' and text[-1] == '"':  # noqa: SIM102
        if "\n" not in text.strip() or text.split("\n")[0][-1] != '"':
            text = text[1:-1]
    return text


_URL = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)


def ref_urls(text: str) -> str:
    return _URL.sub("", text)


def ref_lines(text: str, patterns: list[str]) -> str:
    return "\n".join(line for line in text.split("\n") if line not in patterns)


def ref_slice(
    text: str,
    left: int | str | None = 0,
    right: int | str | None = None,
    include_left: bool = True,
    include_right: bool = True,
    strip: bool = True,
) -> str:
    if isinstance(left, int):
        li = left
    elif isinstance(left, str):
        f = text.find(left)
        if f == -1:
            return ""
        li = f if include_left else f + len(left)
    else:
        li = 0
    if isinstance(right, int):
        ri = right
    elif isinstance(right, str):
        f = text.rfind(right)
        if f == -1:
            return ""
        ri = f + len(right) if include_right else f
    else:
        ri = len(text)
    result = text[li:ri]
    return result.strip() if strip else result


def ref_label(text: str, label: object) -> str:
    text = text.replace("\n", " ").replace("__label__", " ")
    return f"{label} {text}"


# --- strategies ----------------------------------------------------------------------------------

_PIECES = [
    "*",
    "**",
    "***",
    "_",
    "__",
    "[",
    "]",
    "(",
    ")",
    "](",
    "[a](b)",
    "\n",
    "\n\n",
    "\n\n\n",
    "\r\n",
    "\r\n\r\n\r\n",
    '"',
    " ",
    "\t",
    "\x0b",
    "\x1c",
    "\x85",
    "\xa0",
    "\u2028",
    "\u3000",
    "http",
    "https",
    "HTTP",
    "httpS",
    "http\u017f",
    "://",
    "www.",
    "WWW.",
    "ww",
    "w",
    "terms of use",
    "Privacy Policy",
    "coo\u212aie policy",
    "pr\u0130vacy policy",
    "lorem ipsum",
    "LOREM IPSUM",
    "use cookies",
    "privacy & cookies policy",
    *(p.upper() for p in POLICY[9:]),
    "a",
    "b",
    "é",
    "\U0001f600",
    "__label__",
    "x.y",
    "%20",
]

texts = st.lists(st.one_of(st.sampled_from(_PIECES), st.text(max_size=4)), max_size=24).map("".join)

Pair = tuple[Callable[..., Any], Callable[..., Any]]
SIMPLE: dict[str, Pair] = {
    "remove_markdown": (nemo.remove_markdown, ref_markdown),
    "normalize_newlines": (nemo.normalize_newlines, ref_newlines),
    "remove_quotation": (nemo.remove_quotation, ref_quotation),
    "remove_urls": (nemo.remove_urls, ref_urls),
}


@pytest.mark.parametrize("name", sorted(SIMPLE))
@given(text=texts)
@settings(max_examples=400)
def test_simple_modifiers_match_reference(name: str, text: str) -> None:
    ours, ref = SIMPLE[name]
    assert ours(text) == ref(text)


@pytest.mark.parametrize("remove", [True, False])
@given(paragraphs=st.lists(texts, max_size=6))
@settings(max_examples=400)
def test_boilerplate_matches_reference(remove: bool, paragraphs: list[str]) -> None:
    text = "\n\n".join(paragraphs)
    assert nemo.remove_boilerplate_strings(
        text, remove_if_at_top_or_bottom=remove
    ) == ref_boilerplate(text, remove)


@given(
    text=texts,
    patterns=st.lists(st.sampled_from(["", "a", "*", " ", "terms of use", "b"]), max_size=3),
)
def test_remove_lines_matches_reference(text: str, patterns: list[str]) -> None:
    assert nemo.remove_lines(text, patterns) == ref_lines(text, patterns)


bounds = st.one_of(
    st.none(),
    st.integers(-12, 12),
    st.sampled_from([True, False, 2**70, -(2**70)]),
    st.sampled_from(_PIECES),
)


@given(
    text=texts, left=bounds, right=bounds, il=st.booleans(), ir=st.booleans(), strip=st.booleans()
)
@settings(max_examples=600)
def test_slice_text_matches_reference(
    text: str, left: Any, right: Any, il: bool, ir: bool, strip: bool
) -> None:
    assert nemo.slice_text(text, left, right, il, ir, strip) == ref_slice(
        text, left, right, il, ir, strip
    )


@given(text=texts, label=st.sampled_from(["__label__hq", "", "x y", 5, None]))
def test_fasttext_label_matches_reference(text: str, label: object) -> None:
    assert nemo.fasttext_label(text, label) == ref_label(text, label)


def test_url_ignorecase_letters() -> None:
    # Every character that re.IGNORECASE lets stand for a letter of the pattern.
    for cp in range(sys.maxunicode + 1):
        c = chr(cp)
        if 0xD800 <= cp < 0xE000:
            continue
        for template in (
            "{c}ttp://x",
            "h{c}tp://x",
            "htt{c}://x",
            "http{c}://x",
            "{c}ww.x",
            "ww{c}.x",
        ):
            text = template.format(c=c)
            assert nemo.remove_urls(text) == ref_urls(text), (hex(cp), template)


def test_containers_and_identity() -> None:
    docs = ["no change", "**bold**", "see www.x.y now"]
    out = nemo.remove_markdown(docs)
    assert out == ["no change", "bold", "see www.x.y now"]
    assert out[0] is docs[0]
    assert nemo.remove_urls(tuple(docs)) == ("no change", "**bold**", "see  now")
    pa = pytest.importorskip("pyarrow")
    col = nemo.normalize_newlines(pa.array(["a\n\n\n\nb", None]))
    assert pa.chunked_array(col).to_pylist() == ["a\n\nb", None]


def test_threads_agree() -> None:
    docs = [f"**{i}** [x](y{i})\n\n\n_z_" for i in range(5000)]
    want = [ref_markdown(d) for d in docs]
    for threads in (None, 1, 2):
        assert nemo.remove_markdown(docs, threads=threads) == want


# --- the same properties against NeMo itself, where it installs -----------------------------------


@pytest.fixture(scope="module")
def nemo_modifiers() -> Any:
    return pytest.importorskip(
        "nemo_curator.stages.text.modifiers.string", reason="NeMo Curator not installed"
    )


@given(text=texts)
@settings(max_examples=300)
def test_live_nemo(nemo_modifiers: Any, text: str) -> None:
    m = nemo_modifiers
    assert nemo.remove_markdown(text) == m.MarkdownRemover().modify_document(text)
    assert nemo.normalize_newlines(text) == m.NewlineNormalizer().modify_document(text)
    assert nemo.remove_quotation(text) == m.QuotationRemover().modify_document(text)
    assert nemo.remove_urls(text) == m.UrlRemover().modify_document(text)
    for remove in (True, False):
        want = m.BoilerPlateStringModifier(remove_if_at_top_or_bottom=remove).modify_document(text)
        assert nemo.remove_boilerplate_strings(text, remove_if_at_top_or_bottom=remove) == want
    assert nemo.remove_lines(text, ["a", ""]) == m.LineRemover(["a", ""]).modify_document(text)
    assert nemo.slice_text(text, "a", "b", False, True) == m.Slicer(
        "a", "b", False, True
    ).modify_document(text)
