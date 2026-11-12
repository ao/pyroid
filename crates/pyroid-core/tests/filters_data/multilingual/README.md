# Multilingual tokenizer goldens

What datatrove 0.10.1 returns from `split_into_words(text, lang)` and
`len(split_into_sentences(text, lang))`, for every tokenizer it assigns that a
Rust port can reproduce: the 57 rule-based spaCy blank models, the
`WhitespaceTokenizer` and the `IndicNLPTokenizer`. One file per tokenizer,
recorded through one representative datatrove language code (`"l"` in every
row). `../oracle.py` (see its docstring) records them; the English golden is
`../golden-tokenizer.jsonl`.

Recorded on Linux aarch64: CPython 3.13.18 (Unicode 15.1.0),
datatrove 0.10.1, spaCy 3.8.18, regex 2026.9.29, pyarrow 26.0.1; the
`golden-indic-*` files also with indic-nlp-library 0.92 (no urduhack, see
below).

## Rows

```
{"t": text, "w": words, "s": sentence count, "u": "15.1.0", "l": "deu_Latn"}
{"t": text, "w": words, "raise": "ValueError", "at": "s", "u": ..., "l": ...}
```

A row that raised in `split_into_sentences` (`"at": "s"`) keeps its words
(`"w"`); a row that raised in `split_into_words` (`"at": "w"`) has none. Rows
from the language's FineWeb-2 slice carry `"k"`: `"doc"` (a whole document,
with its `"id"`), `"line"` (a C4-style line: `C4QualityFilter(split_paragraph=True)`
input) or `"excerpt"` (a random cut of a document, at any code point). Fuzz rows
carry no `"k"`. Texts containing `IS_ALPHA` are never generated (spaCy's symbol
table bug: see `oracle.py isalpha`).

## How the rows were chosen

* **Real text** (de, fr, es, ru, ar, indic-hi: the slices in `/data/fineweb2`):
  the 3 shortest documents, 80 C4-style lines of at most 120 characters picked
  greedily for diverse punctuation and scripts, 80 random excerpts of 8-120
  characters.
* **Fuzz** (seed 20261105): 3,600 distinct texts from the language's fuzzer
  (`LangFuzzer`, `WsFuzzer` or `IndicFuzzer` in `oracle.py`), all recorded,
  then kept within a 60 kB budget (at most 900 rows): first a greedy cover of
  what the rows exercise (split points between glued tokens, token shapes,
  exception keys, sentence counts, exceptions raised) per byte, then random
  short rows. The fuzzer draws on the tokenizer's own data: every exception key
  (`nlp.tokenizer.rules`, re-cased, re-punctuated, mangled, glued to affixes
  and brackets; the keys spanning white space, which spaCy's special-case
  matcher handles), strings sampled from each prefix / suffix / infix pattern
  and from `token_match` (fr, hu, tr; lookarounds spelled out, case flipped
  where the pattern ignores case), the characters those patterns name, the
  letters, marks and digits of the scripts datatrove assigns to the tokenizer,
  the commonest words of the FineWeb-2 slice and spaCy's stop words, elisions
  and hyphens in their Unicode variants, numbers with separators and
  inflections, URLs, e-mails, white space kinds and English noise.
  `golden-ws.jsonl` adds the `WhitespaceTokenizer` sentence regex's terminal
  punctuation, quotes and upper-case-letter-period acronyms; `golden-indic-*`
  the trivial tokenizer's punctuation, dandas, the `1 , 000` numbers it
  re-joins, and the abbreviations the sentence splitter keeps.

Each row is recorded with one call of each function, by one of 6 workers forked
from a warm process. The same rows come out with 6 or 2 workers, and when re-recorded in
reverse order in one process (`rerecord`): checked for all 61 files; nothing in
these recordings depends on history.

## Re-recording

```
# datatrove 0.10.1 + spacy 3.8.18 + regex + pyarrow (+ indic-nlp-library 0.92 for indic-*)
python oracle.py goldens multilingual                              # spaCy and ws
python oracle.py goldens multilingual indic-hi indic-ur indic-bn   # Indic venv
python oracle.py --lang deu_Latn golden multilingual/golden-de.jsonl   # one file
```

The fuzz texts depend on the interpreter's `unicodedata` (and `regex`'s
script tables): re-record on CPython 3.13 to reproduce the files byte for byte.

## Files

One file per tokenizer, recorded with the datatrove code in the second column:
the tokenizer's own language (its `code_3_script`; where datatrove has several,
the one FineWeb-2 uses: `ekk_Latn` for et, `lvs_Latn` for lv, `als_Latn` for
sq, `srp_Cyrl` for sr, `zsm_Latn` for ms, `tsn_Latn` for tn, `nob_Latn` for nb).
datatrove assigns no Tagalog code to `tl`, so `fil_Latn`; `xx` serves 967 codes
(the biggest FineWeb-2 ones `und_Cyrl` and `uzn_Cyrl`), recorded with
`uzn_Latn`; `ws` is `kat_Geor`. Every code datatrove sends to one tokenizer
gives the same results (the class and its argument are all `load_word_tokenizer`
uses), so one file covers them all. "Real" counts the FineWeb-2 rows; the
others are fuzz. About 650 rows fit the 60 kB budget: the fixed part of a row
(`{"t":"","w":[],"s":1,"u":"15.1.0","l":"deu_Latn"}`) is 50 bytes.

| file | `--lang` | datatrove tokenizer | rows | real | size | raised |
|---|---|---|--:|--:|--:|---|
| `golden-af.jsonl` | `afr_Latn` | SpaCyTokenizer('af') | 654 | 0 | 60 kB | - |
| `golden-am.jsonl` | `amh_Ethi` | SpaCyTokenizer('am') | 627 | 0 | 60 kB | - |
| `golden-ar.jsonl` | `arb_Arab` | SpaCyTokenizer('ar') | 752 | 120 | 98 kB | - |
| `golden-az.jsonl` | `azj_Latn` | SpaCyTokenizer('az') | 643 | 0 | 60 kB | - |
| `golden-bg.jsonl` | `bul_Cyrl` | SpaCyTokenizer('bg') | 624 | 0 | 60 kB | - |
| `golden-ca.jsonl` | `cat_Latn` | SpaCyTokenizer('ca') | 661 | 0 | 60 kB | - |
| `golden-cs.jsonl` | `ces_Latn` | SpaCyTokenizer('cs') | 660 | 0 | 60 kB | - |
| `golden-da.jsonl` | `dan_Latn` | SpaCyTokenizer('da') | 664 | 0 | 60 kB | - |
| `golden-de.jsonl` | `deu_Latn` | SpaCyTokenizer('de') | 773 | 120 | 87 kB | - |
| `golden-dsb.jsonl` | `dsb_Latn` | SpaCyTokenizer('dsb') | 660 | 0 | 60 kB | - |
| `golden-el.jsonl` | `ell_Grek` | SpaCyTokenizer('el') | 627 | 0 | 60 kB | - |
| `golden-en.jsonl` | `eng_Latn` | SpaCyTokenizer('en') | 651 | 0 | 60 kB | - |
| `golden-es.jsonl` | `spa_Latn` | SpaCyTokenizer('es') | 788 | 128 | 88 kB | - |
| `golden-et.jsonl` | `ekk_Latn` | SpaCyTokenizer('et') | 653 | 0 | 60 kB | - |
| `golden-eu.jsonl` | `eus_Latn` | SpaCyTokenizer('eu') | 657 | 0 | 60 kB | - |
| `golden-fa.jsonl` | `fas_Arab` | SpaCyTokenizer('fa') | 626 | 0 | 60 kB | - |
| `golden-fi.jsonl` | `fin_Latn` | SpaCyTokenizer('fi') | 645 | 0 | 60 kB | - |
| `golden-fo.jsonl` | `fao_Latn` | SpaCyTokenizer('fo') | 642 | 0 | 60 kB | - |
| `golden-fr.jsonl` | `fra_Latn` | SpaCyTokenizer('fr') | 763 | 116 | 87 kB | - |
| `golden-ga.jsonl` | `gle_Latn` | SpaCyTokenizer('ga') | 641 | 0 | 60 kB | - |
| `golden-grc.jsonl` | `grc_Grek` | SpaCyTokenizer('grc') | 612 | 0 | 60 kB | - |
| `golden-he.jsonl` | `heb_Hebr` | SpaCyTokenizer('he') | 645 | 0 | 60 kB | - |
| `golden-hr.jsonl` | `hrv_Latn` | SpaCyTokenizer('hr') | 646 | 0 | 60 kB | - |
| `golden-hsb.jsonl` | `hsb_Latn` | SpaCyTokenizer('hsb') | 648 | 0 | 60 kB | - |
| `golden-hu.jsonl` | `hun_Latn` | SpaCyTokenizer('hu') | 646 | 0 | 60 kB | - |
| `golden-hy.jsonl` | `hye_Armn` | SpaCyTokenizer('hy') | 627 | 0 | 60 kB | - |
| `golden-id.jsonl` | `ind_Latn` | SpaCyTokenizer('id') | 637 | 0 | 60 kB | - |
| `golden-is.jsonl` | `isl_Latn` | SpaCyTokenizer('is') | 653 | 0 | 60 kB | - |
| `golden-it.jsonl` | `ita_Latn` | SpaCyTokenizer('it') | 659 | 0 | 60 kB | - |
| `golden-ky.jsonl` | `kir_Cyrl` | SpaCyTokenizer('ky') | 626 | 0 | 60 kB | - |
| `golden-la.jsonl` | `lat_Latn` | SpaCyTokenizer('la') | 650 | 0 | 60 kB | - |
| `golden-lb.jsonl` | `ltz_Latn` | SpaCyTokenizer('lb') | 651 | 0 | 60 kB | - |
| `golden-lg.jsonl` | `lug_Latn` | SpaCyTokenizer('lg') | 647 | 0 | 60 kB | - |
| `golden-lij.jsonl` | `lij_Latn` | SpaCyTokenizer('lij') | 647 | 0 | 60 kB | - |
| `golden-lt.jsonl` | `lit_Latn` | SpaCyTokenizer('lt') | 644 | 0 | 60 kB | - |
| `golden-lv.jsonl` | `lvs_Latn` | SpaCyTokenizer('lv') | 644 | 0 | 60 kB | - |
| `golden-mk.jsonl` | `mkd_Cyrl` | SpaCyTokenizer('mk') | 630 | 0 | 60 kB | - |
| `golden-ms.jsonl` | `zsm_Latn` | SpaCyTokenizer('ms') | 643 | 0 | 60 kB | - |
| `golden-nb.jsonl` | `nob_Latn` | SpaCyTokenizer('nb') | 660 | 0 | 60 kB | - |
| `golden-nl.jsonl` | `nld_Latn` | SpaCyTokenizer('nl') | 661 | 0 | 60 kB | - |
| `golden-pl.jsonl` | `pol_Latn` | SpaCyTokenizer('pl') | 650 | 0 | 60 kB | - |
| `golden-pt.jsonl` | `por_Latn` | SpaCyTokenizer('pt') | 650 | 0 | 60 kB | - |
| `golden-ro.jsonl` | `ron_Latn` | SpaCyTokenizer('ro') | 645 | 0 | 60 kB | - |
| `golden-ru.jsonl` | `rus_Cyrl` | SpaCyTokenizer('ru') | 737 | 109 | 95 kB | - |
| `golden-sk.jsonl` | `slk_Latn` | SpaCyTokenizer('sk') | 642 | 0 | 60 kB | - |
| `golden-sl.jsonl` | `slv_Latn` | SpaCyTokenizer('sl') | 646 | 0 | 60 kB | - |
| `golden-sq.jsonl` | `als_Latn` | SpaCyTokenizer('sq') | 653 | 0 | 60 kB | - |
| `golden-sr.jsonl` | `srp_Cyrl` | SpaCyTokenizer('sr') | 629 | 0 | 60 kB | - |
| `golden-sv.jsonl` | `swe_Latn` | SpaCyTokenizer('sv') | 651 | 0 | 60 kB | - |
| `golden-ti.jsonl` | `tir_Ethi` | SpaCyTokenizer('ti') | 632 | 0 | 60 kB | - |
| `golden-tl.jsonl` | `fil_Latn` | SpaCyTokenizer('tl') | 654 | 0 | 60 kB | - |
| `golden-tn.jsonl` | `tsn_Latn` | SpaCyTokenizer('tn') | 656 | 0 | 60 kB | - |
| `golden-tr.jsonl` | `tur_Latn` | SpaCyTokenizer('tr') | 643 | 0 | 60 kB | - |
| `golden-tt.jsonl` | `tat_Cyrl` | SpaCyTokenizer('tt') | 621 | 0 | 60 kB | - |
| `golden-uk.jsonl` | `ukr_Cyrl` | SpaCyTokenizer('uk') | 633 | 0 | 60 kB | - |
| `golden-xx.jsonl` | `uzn_Latn` | SpaCyTokenizer('xx') | 618 | 0 | 60 kB | - |
| `golden-yo.jsonl` | `yor_Latn` | SpaCyTokenizer('yo') | 640 | 0 | 60 kB | - |
| `golden-ws.jsonl` | `kat_Geor` | WhitespaceTokenizer('') | 597 | 0 | 60 kB | - |
| `golden-indic-hi.jsonl` | `hin_Deva` | IndicNLPTokenizer('hi') | 688 | 107 | 105 kB | ValueError@s 7 |
| `golden-indic-ur.jsonl` | `urd_Arab` | IndicNLPTokenizer('ur') | 444 | 0 | 60 kB | ModuleNotFoundError@s 444 |
| `golden-indic-bn.jsonl` | `ben_Beng` | IndicNLPTokenizer('bn') | 571 | 0 | 60 kB | ValueError@s 8 |

## Notes for the port

* **Urdu (`golden-indic-ur.jsonl`).** `IndicNLPTokenizer("ur").sent_tokenize`
  calls indic-nlp-library's `sentence_split(text, "ur")`, which imports
  `urduhack` — not a dependency of datatrove or indic-nlp-library, and
  urduhack 1.1.1 imports TensorFlow at package import. Without it every
  `split_into_sentences` raises `ModuleNotFoundError`, which is what this file
  records (words kept). The uncommitted `/target/oracle-ml/indic-ur-urduhack-fuzz.jsonl`
  was recorded with urduhack's own rule-based splitter (`eos.py`) loaded
  without TensorFlow (`/target/oracle-ml/urduhack_shim.py`, which expects
  `urduhack==1.1.1` installed with `--no-deps` next to indic-nlp-library).
* **`ValueError` rows (indic-hi, indic-bn).** `sentence_split` re-joins
  sentences it merged with a single space; when the text had other white space
  there, the sentence is not a substring and datatrove's
  `simple_span_tokenize` (`text.index(sentence)`) raises. Real text does this
  too: 9 of the 250 `hin_Deva` slice documents (and 12 of their 2,761 C4-style
  lines) raise in `split_into_sentences`.
* **Turkish `token_match`** is `^(abbr)|(nums)$`: the first alternative is
  anchored at the start only, so with `re.match` any chunk that starts with an
  inflected abbreviation is one token (`Dr.'abc!!!` → `["Dr.'abc!!!"]`).
* **French `token_match`** is case-insensitive (`(?iu)`, 1.45 MB of pattern);
  the exceptions are not (`peut-être` is one token, `PEUT-ÊTRE` four).
* **indic-nlp-library's trivial tokenizer**: the punctuation class is
  `[string.punctuation + ...]` written inside `[...]`, so `\` escapes `]` and
  the backslash itself is not punctuation; the number re-join
  (`([0-9]+ [,.:/] )+[0-9]+`) skips a number at the very start of the text
  (`if start > prev`).
