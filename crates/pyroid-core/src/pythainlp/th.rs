// Derived from PyThaiNLP 5.4.0 (Apache-2.0), © 2016-2026 PyThaiNLP Project,
// and datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! datatrove's word tokenizer for Thai: `ThaiTokenizer`.
//!
//! - `word_tokenize(text) = strip_strings(word_tokenize(text,
//!   keep_whitespace=False, engine="newmm-safe"))`
//! - `sent_tokenize(text) = strip_strings(sent_tokenize(text))` (crfcut)
//! - `span_tokenize(text) = simple_span_tokenize(text, sent_tokenize(text))`
//!
//! The stripped sentences are found in order where crfcut cut them: the text
//! between two of them is whitespace, so `text.index(sent, start)` finds
//! each at its own place. One sentence spans the whole text.
//!
//! The word list and model are the ones [`install`]ed in this process.

use std::borrow::Cow;
use std::sync::{Mutex, OnceLock};

use super::{Thai, ThaiError};
use crate::filters::{FilterError, WordTokenizer};
use crate::ftfy::UnicodeVersion;

static THAI: OnceLock<Thai> = OnceLock::new();
static LOADING: Mutex<()> = Mutex::new(());

/// Load the word list and sentence model from PyThaiNLP's corpus directory
/// for the Thai tokenizer, once per process. Later calls return the data
/// already loaded, whatever their directory.
pub fn install(corpus_dir: impl AsRef<std::path::Path>) -> Result<&'static Thai, ThaiError> {
    install_with(corpus_dir, || {})
}

/// [`install`], calling `loaded` if this call is the one that read the files.
pub fn install_with(
    corpus_dir: impl AsRef<std::path::Path>,
    loaded: impl FnOnce(),
) -> Result<&'static Thai, ThaiError> {
    if let Some(t) = THAI.get() {
        return Ok(t);
    }
    let _guard = LOADING
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(t) = THAI.get() {
        return Ok(t);
    }
    let t = Thai::open(corpus_dir)?;
    let t = THAI.get_or_init(|| t);
    loaded();
    Ok(t)
}

/// The data [`install`] loaded, if any.
pub fn installed() -> Option<&'static Thai> {
    THAI.get()
}

/// datatrove's `ThaiTokenizer` for one Unicode version (only `re`'s `\d`
/// depends on it).
#[derive(Debug)]
pub struct ThTokenizer {
    version: UnicodeVersion,
}

static TH: [ThTokenizer; 5] = [
    ThTokenizer {
        version: UnicodeVersion::V14_0,
    },
    ThTokenizer {
        version: UnicodeVersion::V15_0,
    },
    ThTokenizer {
        version: UnicodeVersion::V15_1,
    },
    ThTokenizer {
        version: UnicodeVersion::V16_0,
    },
    ThTokenizer {
        version: UnicodeVersion::V17_0,
    },
];

impl ThTokenizer {
    /// The shared instance for `version`, if the data is [`install`]ed.
    pub fn get(version: UnicodeVersion) -> Option<&'static ThTokenizer> {
        installed()?;
        TH.iter().find(|t| t.version == version)
    }

    fn thai() -> &'static Thai {
        installed().expect("ThTokenizer::get checks that the data is installed")
    }

    /// datatrove's `span_tokenize(text)`: `(start, end)` character offsets,
    /// appended to `out`.
    ///
    /// # Errors
    /// [`FilterError::ThaiTextTooLong`] for a text of [`super::MAX_CHARS`]
    /// characters or more.
    pub fn spans(&self, text: &str, out: &mut Vec<(usize, usize)>) -> Result<(), FilterError> {
        super::check_len(text)?;
        let mut r = Vec::new();
        Self::thai().datatrove_sentence_ranges(text, self.version, &mut r);
        if r.len() == 1 {
            out.push((0, text.chars().count()));
            return Ok(());
        }
        let (mut at_b, mut at_c) = (0usize, 0usize);
        let mut chars = |b: usize| {
            at_c += text[at_b..b].chars().count();
            at_b = b;
            at_c
        };
        for (s, e) in r {
            let s = chars(s);
            let e = chars(e);
            out.push((s, e));
        }
        Ok(())
    }
}

impl WordTokenizer for ThTokenizer {
    fn unicode_version(&self) -> UnicodeVersion {
        self.version
    }

    fn words<'a>(&self, text: &'a str, out: &mut Vec<&'a str>) -> Result<(), FilterError> {
        super::check_len(text)?;
        Self::thai().datatrove_words(text, self.version, out);
        Ok(())
    }

    fn sentence_ends(&self, text: &str, out: &mut Vec<usize>) -> Result<(), FilterError> {
        super::check_len(text)?;
        let mut r = Vec::new();
        Self::thai().datatrove_sentence_ranges(text, self.version, &mut r);
        if r.len() == 1 {
            out.push(text.len());
        } else {
            out.extend(r.into_iter().map(|(_, e)| e));
        }
        Ok(())
    }

    fn sentences<'a>(&self, text: &'a str, out: &mut Vec<Cow<'a, str>>) -> Result<(), FilterError> {
        super::check_len(text)?;
        let mut s = Vec::new();
        Self::thai().datatrove_sentences(text, self.version, &mut s);
        out.extend(s.into_iter().map(Cow::Borrowed));
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::print_stderr)] // prints why a test was skipped
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    /// The tokenizer, if `PYROID_PYTHAINLP_CORPUS` names PyThaiNLP's corpus
    /// directory.
    fn tok() -> Option<(&'static Thai, &'static ThTokenizer)> {
        let Some(dir) = std::env::var_os("PYROID_PYTHAINLP_CORPUS") else {
            eprintln!("skipped, set PYROID_PYTHAINLP_CORPUS");
            return None;
        };
        let thai = install(dir).expect("PyThaiNLP's corpus directory");
        Some((thai, ThTokenizer::get(V).expect("installed")))
    }

    struct Case {
        t: &'static str,
        nm: &'static [&'static str],
        toks: &'static [&'static str],
        lab: &'static str,
        crf: &'static [&'static str],
        w: &'static [&'static str],
        s: &'static [&'static str],
        sp: &'static [(usize, usize)],
    }

    // Recorded from PyThaiNLP 5.4.0 and datatrove 0.10.1 (Python 3.13):
    // `newmm.segment(t)`, `word_tokenize(t)`, the raw crfcut labels,
    // `crfcut.segment(t)`, and `ThaiTokenizer`'s `word_tokenize`,
    // `sent_tokenize` and `span_tokenize`.
    const CASES: [Case; 3] = [
        Case {
            t: "วันนี้อากาศดีมาก ฉันไปตลาดเวลา 12:30 น. ซื้อผลไม้ 1,234.5 บาท แล้วกลับบ้าน",
            nm: &[
                "วันนี้",
                "อากาศ",
                "ดีมาก",
                " ",
                "ฉัน",
                "ไป",
                "ตลาด",
                "เวลา",
                " ",
                "12",
                ":30",
                " ",
                "น.",
                " ",
                "ซื้อ",
                "ผลไม้",
                " ",
                "1,234.5",
                " ",
                "บาท",
                " ",
                "แล้ว",
                "กลับบ้าน",
            ],
            toks: &[
                "วันนี้",
                "อากาศ",
                "ดีมาก",
                " ",
                "ฉัน",
                "ไป",
                "ตลาด",
                "เวลา",
                " ",
                "12:30",
                " ",
                "น.",
                " ",
                "ซื้อ",
                "ผลไม้",
                " ",
                "1,234.5",
                " ",
                "บาท",
                " ",
                "แล้ว",
                "กลับบ้าน",
            ],
            lab: "IIIEIIIIIIIIIIIIIIIEIE",
            crf: &[
                "วันนี้อากาศดีมาก ",
                "ฉันไปตลาดเวลา 12:30 น.",
                " ซื้อผลไม้ 1,234.5 บาท ",
                "แล้วกลับบ้าน",
            ],
            w: &[
                "วันนี้",
                "อากาศ",
                "ดีมาก",
                "ฉัน",
                "ไป",
                "ตลาด",
                "เวลา",
                "12:30",
                "น.",
                "ซื้อ",
                "ผลไม้",
                "1,234.5",
                "บาท",
                "แล้ว",
                "กลับบ้าน",
            ],
            s: &[
                "วันนี้อากาศดีมาก",
                "ฉันไปตลาดเวลา 12:30 น.",
                "ซื้อผลไม้ 1,234.5 บาท",
                "แล้วกลับบ้าน",
            ],
            sp: &[(0, 16), (17, 39), (40, 61), (62, 74)],
        },
        Case {
            t: "ผมชอบกินข้าว! คุณล่ะครับ?  \n",
            nm: &[
                "ผม",
                "ชอบ",
                "กินข้าว",
                "!",
                " ",
                "คุณ",
                "ล่ะ",
                "ครับ",
                "?",
                "  ",
                "\n",
            ],
            toks: &[
                "ผม",
                "ชอบ",
                "กินข้าว",
                "!",
                " ",
                "คุณ",
                "ล่ะ",
                "ครับ",
                "?",
                "  ",
                "\n",
            ],
            lab: "IIIIEIIIIIE",
            crf: &["ผมชอบกินข้าว!", " คุณล่ะครับ?", "  \n"],
            w: &["ผม", "ชอบ", "กินข้าว", "!", "คุณ", "ล่ะ", "ครับ", "?"],
            s: &["ผมชอบกินข้าว!", "คุณล่ะครับ?"],
            sp: &[(0, 13), (14, 25)],
        },
        Case {
            t: " \t\n",
            nm: &[" \t", "\n"],
            toks: &[" \t", "\n"],
            lab: "IE",
            crf: &[" \t\n"],
            w: &[],
            s: &[],
            sp: &[],
        },
    ];

    #[test]
    fn matches_pythainlp_and_datatrove() {
        let Some((thai, tok)) = tok() else {
            return;
        };
        for c in &CASES {
            let t = c.t;
            let mut out = Vec::new();
            thai.newmm(t, false, V, &mut out);
            assert_eq!(out, c.nm, "newmm {t:?}");
            out.clear();
            thai.newmm(t, true, V, &mut out);
            assert_eq!(out, c.nm, "newmm safe {t:?}");
            out.clear();
            thai.word_tokenize(t, V, &mut out);
            assert_eq!(out, c.toks, "word_tokenize {t:?}");
            let mut labs = Vec::new();
            thai.crfcut_labels(&out, &mut labs);
            let labs: String = labs.iter().map(|&e| if e { 'E' } else { 'I' }).collect();
            assert_eq!(labs, c.lab, "labels {t:?}");
            out.clear();
            thai.crfcut(t, V, &mut out);
            assert_eq!(out, c.crf, "crfcut {t:?}");
            out.clear();
            tok.words(t, &mut out).unwrap();
            assert_eq!(out, c.w, "words {t:?}");
            let mut sents = Vec::new();
            tok.sentences(t, &mut sents).unwrap();
            assert_eq!(sents, c.s, "sentences {t:?}");
            let mut spans = Vec::new();
            tok.spans(t, &mut spans).expect("a short text");
            assert_eq!(spans, c.sp, "spans {t:?}");
            let mut ends = Vec::new();
            tok.sentence_ends(t, &mut ends).unwrap();
            let want: Vec<usize> =
                c.sp.iter()
                    .map(|&(_, e)| t.char_indices().nth(e).map_or(t.len(), |(b, _)| b))
                    .collect();
            assert_eq!(ends, want, "sentence ends {t:?}");
        }
        assert!(thai.num_words() > 60_000);
        assert_eq!(tok.unicode_version(), V);
        assert_eq!(tok.count_sentences(" ").unwrap(), 1);
    }

    #[test]
    fn one_sentence_spans_the_text() {
        let Some((_, tok)) = tok() else {
            return;
        };
        // Recorded from datatrove 0.10.1: one sentence is `(0, len(text))`,
        // its surrounding whitespace included.
        let t = "  ฉันไปโรงเรียน \n";
        let mut spans = Vec::new();
        tok.spans(t, &mut spans).expect("a short text");
        assert_eq!(spans, [(0, 17)]);
        let mut ends = Vec::new();
        tok.sentence_ends(t, &mut ends).unwrap();
        assert_eq!(ends, [t.len()]);
    }

    #[test]
    fn get_needs_install() {
        // Without the corpus nothing is installed in this process (unless
        // another test installed it).
        assert_eq!(ThTokenizer::get(V).is_some(), installed().is_some());
    }
}
