# Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Derived from NeMo Curator 1.4.0 (Apache-2.0), © NVIDIA CORPORATION; modified.
"""NeMo Curator 1.4.0's heuristic and fastText document filters, vendored for the tests only.

The ``score_document`` methods of ``nemo_curator/stages/text/filters/heuristic/string.py``,
``.../heuristic/repetition/repetition.py`` and ``.../filters/fasttext/fasttext_filters.py``, with
the constants and helpers they use (``utils/constants.py``, ``utils/text_utils.py``), copied
unchanged. Changes: one module instead of several; ``keep_document``, model loading and the
pipeline machinery left out; ``get_word_splitter("zh")`` falls back to ``pyroid.jieba`` when
jieba is not installed, and ``"ja"`` raises ImportError without MeCab, as NeMo does.
``test_nemo.py`` checks pyroid against this module on every interpreter, and against NeMo
Curator itself where it is installed.
"""

# ruff: noqa: ANN001, ANN201, ANN202, D101, D102, D103, D107, PLR2004, RUF001, SIM114, RET505
# mypy: ignore-errors

import re
from collections.abc import Callable

import regex

# --- utils/constants.py -----------------------------------------------------------------------

end_marks = (".", "?", "!", '"', "'")
ellipsis_marks = {"...", "[...]", "…", "(...)", "[…]", "-»", "read more..", "read more"}
policy_substrings = [
    "terms of use",
    "privacy policy",
    "cookie policy",
    "uses cookies",
    "privacy overview",
    "use of cookies",
    "use cookies",
    "privacy & cookies policy",
    "privacy and cookies policy",
    "This website uses cookies to improve your experience while you "
    "navigate through the website. Out of these cookies, the cookies "
    "that are categorized as necessary are stored on your browser as they "
    "are essential for the working of basic functionalities of the website. "
    "We also use third-party cookies that help us analyze and understand how "
    "you use this website. These cookies will be stored in your browser only "
    "with your consent. You also have the option to opt-out of these "
    "cookies. But opting out of some of these cookies may have an effect "
    "on your browsing experience.".lower(),
    "Necessary cookies are absolutely essential for the website to "
    "function properly. This category only includes cookies that "
    "ensures basic functionalities and security features of the website. "
    "These cookies do not store any personal information.".lower(),
    "Any cookies that may not be particularly necessary for the website "
    "to function and is used specifically to collect user personal data "
    "via analytics, ads, other embedded contents are termed as non-necessary "
    "cookies. It is mandatory to procure user consent prior to running these "
    "cookies on your website.".lower(),
    "This site uses cookies, including for analytics, personalization, and "
    "advertising purposes. For more information or to change your "
    "cookie settings, click here.".lower(),
    "If you continue to browse this site without changing your cookie "
    "settings, you agree to this use. AcceptRead More".lower(),
]
white_space_list = ["\t", "\n", "\r", "\b", " "]
common_english_words = {"the", "be", "to", "of", "and", "that", "have", "with"}
bullet_list = {
    "•",
    "‣",
    "⁃",
    "⁌",
    "⁍",
    "∙",
    "○",
    "●",
    "◘",
    "◦",
    "⦾",
    "⦿",
}

regex_alpha = regex.compile("[[:alpha:]]")
regex_digit = regex.compile("[[:digit:]]")
regex_alphanum = re.compile("[a-zA-Z0-9\n?!,.]")
regex_url = re.compile(r"https?://(?:[A-Za-z0-9$\-_@.&+/:=?#~]|[!*\(\),]|(?:%[0-9A-Fa-f]{2}))+")
regex_paren = re.compile(r"{|}|⟨|⟩|\[|\]|\(|\)")
regex_hash = re.compile("#+")

# --- utils/text_utils.py ----------------------------------------------------------------------


def get_word_splitter(language: str) -> Callable[[str], list[str]]:
    language = language.lower()

    if language == "zh":
        try:
            import jieba
        except ImportError:  # pyroid.jieba reproduces jieba 0.42.1's cut()
            import pyroid.jieba as jieba

        def jieba_splitter(text: str) -> list[str]:
            return list(jieba.cut(text))

        return jieba_splitter

    elif language == "ja":
        import MeCab

        def mecab_splitter(text: str) -> list[str]:
            mecab = MeCab.Tagger()
            parsed = mecab.parse(text)
            lines = parsed.strip().split("\n")
            return [line.split("\t")[0] for line in lines if line and line != "EOS"]

        return mecab_splitter

    else:

        def default_splitter(text: str) -> list[str]:
            return text.split()

        return default_splitter


def get_paragraphs(document: str) -> list[str]:
    return document.split("\n\n")


def get_sentences(document: str) -> list[str]:
    return [x for x in document.split("\n") if len(x.strip()) > 0]


def get_ngrams(input_list: list[str], n: int) -> list[tuple[str, ...]]:
    return list(zip(*[input_list[i:] for i in range(n)], strict=False))


# --- filters/doc_filter.py --------------------------------------------------------------------


class DocumentFilter:
    def __init__(self):
        super().__init__()
        self._name = self.__class__.__name__
        self._sentences = None
        self._paragraphs = None
        self._ngrams = None


# --- filters/heuristic/string.py --------------------------------------------------------------


class NonAlphaNumericFilter(DocumentFilter):
    def __init__(self, max_non_alpha_numeric_to_text_ratio: float = 0.25):
        super().__init__()
        self._cutoff = max_non_alpha_numeric_to_text_ratio
        self._name = "alpha_numeric"

    def score_document(self, text: str) -> float:
        nchar = len(text)
        return (nchar - len(regex_alphanum.findall(text))) / nchar if nchar > 0 else 1.0


class SymbolsToWordsFilter(DocumentFilter):
    def __init__(self, max_symbol_to_word_ratio: float = 0.1, lang: str = "en"):
        super().__init__()
        self._cutoff = max_symbol_to_word_ratio
        self._word_splitter = get_word_splitter(lang)
        self._name = "symbol_to_word"

    def score_document(self, text: str) -> float:
        num_symbol_words = 0
        words = self._word_splitter(text.strip())
        for w in words:
            word = w.strip()
            symbol_ratio = len(regex_hash.findall(word)) / len(word)
            if word in ellipsis_marks or symbol_ratio > 0.5:
                num_symbol_words += 1
        return num_symbol_words / len(words)


class NumbersFilter(DocumentFilter):
    def __init__(self, max_number_to_text_ratio: float = 0.15):
        super().__init__()
        self._cutoff = max_number_to_text_ratio
        self._name = "numbers_ratio"

    def score_document(self, text: str) -> float:
        nchar = len(text)
        return len(regex_digit.findall(text)) / nchar if nchar > 0 else 1.0


class UrlsFilter(DocumentFilter):
    def __init__(self, max_url_to_text_ratio: float = 0.2, url_regex=None):
        super().__init__()
        self._cutoff = max_url_to_text_ratio
        self._name = "urls_ratio"
        self._url_regex = (
            re.compile(url_regex) if isinstance(url_regex, str) else (url_regex or regex_url)
        )

    def score_document(self, text: str) -> float:
        all_urls = self._url_regex.findall(text)
        url_chars = sum([len(url) for url in all_urls])
        nchar = len(text)
        return url_chars / nchar if nchar > 0 else 1.0


class BulletsFilter(DocumentFilter):
    def __init__(self, max_bullet_lines_ratio: float = 0.9):
        super().__init__()
        self._cutoff = max_bullet_lines_ratio
        self._sentences = None
        self._name = "bullet_ratio"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        num_bullet_lines = 0
        for sentence in sentences:
            for bullet in bullet_list:
                if sentence.strip().startswith(bullet):
                    num_bullet_lines += 1
                    break
        return num_bullet_lines / len(sentences)


class WhiteSpaceFilter(DocumentFilter):
    def __init__(self, max_white_space_ratio: float = 0.25):
        super().__init__()
        self._cutoff = max_white_space_ratio
        self._name = "white_space"

    def score_document(self, text: str) -> float:
        nchar = len(text)
        return len([x for x in text if x in white_space_list]) / nchar if nchar > 0 else 1.0


class ParenthesesFilter(DocumentFilter):
    def __init__(self, max_parentheses_ratio: float = 0.1):
        super().__init__()
        self._max_parentheses_ratio = max_parentheses_ratio
        self._name = "parentheses_ratio"

    def score_document(self, text: str) -> float:
        nchar = len(text)
        return len(regex_paren.findall(text)) / nchar if nchar > 0 else 1.0


class LongWordFilter(DocumentFilter):
    def __init__(self, max_word_length: int = 1000, lang: str = "en"):
        super().__init__()
        self._max_word_length = max_word_length
        self._word_splitter = get_word_splitter(lang)
        self._name = "max_word_length"

    def score_document(self, text: str) -> float:
        return max(len(w) for w in self._word_splitter(text.strip()))


class WordCountFilter(DocumentFilter):
    def __init__(self, min_words: int = 50, max_words: int = 100000, lang: str = "en"):
        super().__init__()
        self._min_words = min_words
        self._max_words = max_words
        self._word_splitter = get_word_splitter(lang)
        self._name = "word_count"

    def score_document(self, text: str) -> float:
        return len(self._word_splitter(text.strip()))


class BoilerPlateStringFilter(DocumentFilter):
    def __init__(
        self,
        remove_if_at_top_or_bottom: bool = True,
        max_boilerplate_string_ratio: float = 0.4,
    ):
        super().__init__()
        self._remove_if_at_top_or_bottom = remove_if_at_top_or_bottom
        self._max_boilerplate_string_ratio = max_boilerplate_string_ratio
        self._boilerplate_paragraph_indices = []
        self._max_ratio = 1.0
        self._name = "boilerplate_string_ratio"

    def score_document(self, text: str) -> float:
        boilerplate_paragraph_count = 0
        paragraphs = get_paragraphs(text)
        for _idx, each_paragraph in enumerate(paragraphs):
            paragraph = each_paragraph.strip().lower()
            if "lorem ipsum" in paragraph:
                return self._max_ratio
            if any(p in paragraph for p in policy_substrings):
                boilerplate_paragraph_count += 1
        return boilerplate_paragraph_count / len(paragraphs)


class MeanWordLengthFilter(DocumentFilter):
    def __init__(
        self,
        min_mean_word_length: int = 3,
        max_mean_word_length: int = 10,
        lang: str = "en",
    ):
        super().__init__()
        self._min_cutoff = min_mean_word_length
        self._max_cutoff = max_mean_word_length
        self._word_splitter = get_word_splitter(lang)
        self._name = "mean_word_length"

    def score_document(self, text: str) -> float:
        word_lens = [len(w) for w in self._word_splitter(text.strip()) if len(w) > 0]
        return sum(word_lens) / len(word_lens)


class PunctuationFilter(DocumentFilter):
    def __init__(self, max_num_sentences_without_endmark_ratio: float = 0.85):
        super().__init__()
        self._cutoff = max_num_sentences_without_endmark_ratio
        self._name = "punctuation"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        num_sentence_without_endmarks = len(
            [s for s in sentences if not s.strip().endswith(end_marks)]
        )
        return num_sentence_without_endmarks / len(sentences)


class EllipsisFilter(DocumentFilter):
    def __init__(self, max_num_lines_ending_with_ellipsis_ratio: float = 0.3):
        super().__init__()
        self._cutoff = max_num_lines_ending_with_ellipsis_ratio
        self._name = "ellipsis"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        num_lines_ending_with_ellipsis = 0
        for sentence in sentences:
            for ellipsis in ellipsis_marks:
                if sentence.strip().lower().endswith(ellipsis):
                    num_lines_ending_with_ellipsis += 1
                    break
        return num_lines_ending_with_ellipsis / len(sentences)


class CommonEnglishWordsFilter(DocumentFilter):
    def __init__(self, min_num_common_words: int = 2, stop_at_false: bool = True):
        super().__init__()
        self._cutoff = min_num_common_words
        self._stop_at_false = stop_at_false
        self._word_splitter = get_word_splitter("en")
        self._name = "common_english_words"

    def score_document(self, text: str) -> int:
        common_word_counter = 0
        for word in self._word_splitter(text.strip()):
            if word in common_english_words:
                common_word_counter += 1
            if self._stop_at_false and common_word_counter >= self._cutoff:
                return common_word_counter

        return common_word_counter


class WordsWithoutAlphabetsFilter(DocumentFilter):
    def __init__(self, min_words_with_alphabets: float = 0.8, lang: str = "en"):
        super().__init__()
        self._cutoff = min_words_with_alphabets
        self._word_splitter = get_word_splitter(lang)
        self._name = "words_without_alphabets"

    def score_document(self, text: str) -> float:
        num_english_alpha = 0
        words = self._word_splitter(text.strip())
        for word in words:
            if regex_alpha.search(word):
                num_english_alpha += 1

        return num_english_alpha / len(words)


class PornographicUrlsFilter(DocumentFilter):
    def __init__(self, url_regex=None):
        super().__init__()
        self._url_regex = (
            re.compile(url_regex) if isinstance(url_regex, str) else (url_regex or regex_url)
        )

    def score_document(self, text: str) -> int:
        all_urls = self._url_regex.findall(text)
        for url in all_urls:
            if "porn" in url:
                return 1

        return 0


class SubstringFilter(DocumentFilter):
    def __init__(self, substring: str, position: str):
        super().__init__()
        self._substring = substring
        if position not in ["prefix", "suffix", "any"]:
            msg = f"Invalid position: {position}. Must be one of: prefix, suffix, any."
            raise ValueError(msg)
        self._position = position

    def score_document(self, text: str) -> int:
        if self._position == "prefix":
            return int(text.startswith(self._substring))
        elif self._position == "suffix":
            return int(text.endswith(self._substring))
        elif self._position == "any":
            return int(self._substring in text)
        else:
            msg = f"Invalid position: {self._position}. Must be one of: prefix, suffix, any."
            raise ValueError(msg)


# --- filters/heuristic/repetition/repetition.py -----------------------------------------------


class RepeatedLinesFilter(DocumentFilter):
    def __init__(self, max_repeated_line_fraction: float = 0.7):
        super().__init__()
        self._cutoff = max_repeated_line_fraction
        self._name = "repeated_lines"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)
        return len(set(sentences)) / len(sentences)


class RepeatedParagraphsFilter(DocumentFilter):
    def __init__(self, max_repeated_paragraphs_ratio: float = 0.7):
        super().__init__()
        self._max_repeated_paragraphs_ratio = max_repeated_paragraphs_ratio
        self._name = "repeated_paragraphs"

    def score_document(self, text: str) -> float:
        paragraphs = self._paragraphs
        if paragraphs is None:
            paragraphs = get_paragraphs(text)
        return len(set(paragraphs)) / len(paragraphs)


class RepeatedLinesByCharFilter(DocumentFilter):
    def __init__(self, max_repeated_lines_char_ratio: float = 0.8):
        super().__init__()
        self._cutoff = max_repeated_lines_char_ratio
        self._name = "repeated_lines_char"

    def score_document(self, text: str) -> float:
        sentences = self._sentences
        if sentences is None:
            sentences = get_sentences(text)

        return len("".join(set(sentences))) / len("".join(sentences))


class RepeatedParagraphsByCharFilter(DocumentFilter):
    def __init__(self, max_repeated_paragraphs_char_ratio: float = 0.8):
        super().__init__()
        self._cutoff = max_repeated_paragraphs_char_ratio
        self._name = "repeated_paragraphs_char"

    def score_document(self, text: str) -> float:
        paragraphs = self._paragraphs
        if paragraphs is None:
            paragraphs = get_paragraphs(text)

        return len("".join(set(paragraphs))) / len("".join(paragraphs))


class RepeatingTopNGramsFilter(DocumentFilter):
    def __init__(self, n: int = 2, max_repeating_ngram_ratio: float = 0.2, lang: str = "en"):
        super().__init__()
        self._n = n
        self._cutoff = max_repeating_ngram_ratio
        self._max_ratio = 1.0
        self._word_splitter = get_word_splitter(lang)
        self._name = f"repeating_top_{n}grams"

    def score_document(self, text: str) -> float:
        ngrams = self._ngrams
        if ngrams is None:
            split_text = self._word_splitter(text.strip())
            if len(split_text) < self._n:
                return self._max_ratio
            ngrams = get_ngrams(split_text, self._n)
        unique_ngrams = set(ngrams)
        counts = {
            ngram: {"freq": 0, "num_chars": sum(len(word) for word in ngram)}
            for ngram in unique_ngrams
        }
        for ngram in ngrams:
            counts[ngram]["freq"] += 1
        most_frqnt_ngram = " ".join(max(counts, key=lambda x: counts[x]["freq"]))
        nchar = len(text)
        len_diff = nchar - len(text.replace(most_frqnt_ngram, ""))
        return len_diff / nchar if nchar > 0 else 1.0


class RepeatingDuplicateNGramsFilter(DocumentFilter):
    def __init__(
        self, n: int = 2, max_repeating_duplicate_ngram_ratio: float = 0.2, lang: str = "en"
    ):
        super().__init__()
        self._n = n
        self._cutoff = max_repeating_duplicate_ngram_ratio
        self._max_ratio = 1.0
        self._word_splitter = get_word_splitter(lang)
        self._name = f"repeating_dup_{n}gram"

    def score_document(self, text: str) -> float:
        ngrams = self._ngrams
        if ngrams is None:
            split_text = self._word_splitter(text.strip())
            if len(split_text) < self._n:
                return self._max_ratio
            ngrams = get_ngrams(split_text, self._n)

        counts = {}
        duplicated_nchar = 0
        overlapping_ngrams = 0
        for ngram in ngrams:
            counts[ngram] = counts.get(ngram, 0) + 1
            if counts[ngram] > 1:
                duplicated_ngrams = sum(len(gram) for gram in ngram[overlapping_ngrams:])
                nspaces = min(self._n - overlapping_ngrams, self._n - 1)
                duplicated_nchar += duplicated_ngrams + nspaces
                overlapping_ngrams = self._n
            overlapping_ngrams = max(overlapping_ngrams - 1, 0)

        nchar = len(text)
        return duplicated_nchar / nchar if nchar > 0 else 1.0


# --- filters/fasttext/fasttext_filters.py (score_document only; the model is passed in) -------


class FastTextQualityFilter(DocumentFilter):
    def __init__(self, model, label: str = "__label__hq"):
        super().__init__()
        self._fasttext_quality_filter_model = model
        self._label = label
        self._name = "fasttext_quality_filter"

    def score_document(self, text: str) -> float:
        model = self._fasttext_quality_filter_model

        text = text.replace("\n", " ").replace("__label__", " ")
        label, score = model.predict([text])
        document_score = score[0][0].item()
        if label[0][0] != self._label:
            document_score = 1 - document_score

        return document_score


class FastTextLangId(DocumentFilter):
    def __init__(self, model):
        super().__init__()
        self._fasttext_langid_model = model
        self._name = "lang_id"

    def score_document(self, text: str) -> str:
        model = self._fasttext_langid_model

        pp = text.strip().replace("\n", " ")
        label, score = model.predict([pp], k=1)
        score = score[0][0].item()
        lang_code = label[0][0].removeprefix("__label__")

        return str([score, lang_code])
