// Derived from NLTK 3.10.4 nltk/stem/porter.py (Apache-2.0), © NLTK Project; modified.
//! NLTK's Porter stemmer: `nltk.stem.porter.PorterStemmer().stem(word)` in its
//! default `NLTK_EXTENSIONS` mode (nltk 3.10.4), which rouge-score's
//! `DefaultTokenizer(use_stemmer=True)` applies to every token longer than 3.
//!
//! The port works on ASCII bytes: the ROUGE tokenizer only produces
//! `[a-z0-9]+` tokens. Rules, their order and every special case follow
//! nltk's code exactly (the irregular-forms pool, the `len <= 2` early
//! return, the NLTK-only `ies`/`ied`/`alli`/`fulli`/`logi` rules, and the
//! "first rule whose suffix matches decides" semantics of `_apply_rule_list`).
//!
//! Porter, M. "An algorithm for suffix stripping." Program 14.3 (1980): 130-137.

use std::borrow::Cow;

/// `PorterStemmer().stem(word)` for an ASCII word.
///
/// Uppercase ASCII letters are lowercased first, as nltk does. Bytes other
/// than ASCII letters are consonants (nltk's rule for digits); non-ASCII
/// input is processed byte by byte, which is *not* what nltk does with
/// non-ASCII text (it works on code points), so pass ASCII only.
///
/// ```
/// use pyroid_core::metrics::rouge::porter::stem;
/// assert_eq!(&*stem(b"running"), b"run");
/// assert_eq!(&*stem(b"generalizations"), b"gener");
/// assert_eq!(&*stem(b"dying"), b"die"); // nltk's irregular forms
/// assert_eq!(&*stem(b"is"), b"is"); // words of 1-2 letters are kept
/// ```
pub fn stem(word: &[u8]) -> Cow<'_, [u8]> {
    let mut out = Vec::new();
    stem_into(word, &mut out);
    if out == word {
        Cow::Borrowed(word)
    } else {
        Cow::Owned(out)
    }
}

/// [`stem`] written into `out` (cleared first), so a caller stemming many
/// words reuses one buffer.
///
/// ```
/// use pyroid_core::metrics::rouge::porter::stem_into;
/// let mut out = Vec::new();
/// stem_into(b"Caresses", &mut out);
/// assert_eq!(out, b"caress");
/// ```
pub fn stem_into(word: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.extend_from_slice(word);
    out.make_ascii_lowercase();
    if let Some(form) = irregular_form(out) {
        out.clear();
        out.extend_from_slice(form);
        return;
    }
    // nltk tests `len(word)` (before lowercasing); ASCII lowercasing keeps it.
    if word.len() <= 2 {
        return;
    }
    step1a(out);
    step1b(out);
    step1c(out);
    step2(out);
    step3(out);
    step4(out);
    step5a(out);
    step5b(out);
}

/// nltk's `pool` of irregular forms (`NLTK_EXTENSIONS` only).
fn irregular_form(w: &[u8]) -> Option<&'static [u8]> {
    Some(match w {
        b"sky" | b"skies" => b"sky",
        b"dying" => b"die",
        b"lying" => b"lie",
        b"tying" => b"tie",
        b"news" => b"news",
        b"innings" | b"inning" => b"inning",
        b"outings" | b"outing" => b"outing",
        b"cannings" | b"canning" => b"canning",
        b"howe" => b"howe",
        b"proceed" => b"proceed",
        b"exceed" => b"exceed",
        b"succeed" => b"succeed",
        _ => return None,
    })
}

fn is_vowel(c: u8) -> bool {
    matches!(c, b'a' | b'e' | b'i' | b'o' | b'u')
}

/// `_is_consonant(word, i)`: a letter other than a, e, i, o, u, and other
/// than a `y` preceded by a consonant (a run of `y`s alternates).
fn is_consonant(w: &[u8], mut i: usize) -> bool {
    if is_vowel(w[i]) {
        return false;
    }
    if w[i] == b'y' {
        let mut negate = false;
        while i > 0 && w[i] == b'y' {
            negate = !negate;
            i -= 1;
        }
        // A consonant iff (w[i] is not a vowel) != negate.
        return is_vowel(w[i]) == negate;
    }
    true
}

/// `_consonant_flags(word)`: `is_consonant` for every position, left to right.
fn consonant_flags(w: &[u8]) -> impl Iterator<Item = bool> + '_ {
    let mut prev = true;
    w.iter().enumerate().map(move |(i, &c)| {
        let cons = if is_vowel(c) {
            false
        } else if c == b'y' {
            i == 0 || !prev
        } else {
            true
        };
        prev = cons;
        cons
    })
}

/// `_measure(stem)`: m in `[C](VC){m}[V]`, the number of vowel→consonant steps.
fn measure(s: &[u8]) -> usize {
    let mut m = 0;
    let mut prev_vowel = false;
    for cons in consonant_flags(s) {
        if prev_vowel && cons {
            m += 1;
        }
        prev_vowel = !cons;
    }
    m
}

/// `_contains_vowel(stem)`.
fn contains_vowel(s: &[u8]) -> bool {
    consonant_flags(s).any(|cons| !cons)
}

/// `_ends_double_consonant(word)` (condition *d).
fn ends_double_consonant(w: &[u8]) -> bool {
    let n = w.len();
    n >= 2 && w[n - 1] == w[n - 2] && is_consonant(w, n - 1)
}

/// `_ends_cvc(word)` (condition *o), with the NLTK extension for 2 letters.
fn ends_cvc(w: &[u8]) -> bool {
    let n = w.len();
    (n >= 3
        && is_consonant(w, n - 3)
        && !is_consonant(w, n - 2)
        && is_consonant(w, n - 1)
        && !matches!(w[n - 1], b'w' | b'x' | b'y'))
        || (n == 2 && !is_consonant(w, 0) && is_consonant(w, 1))
}

/// The condition of a suffix rule, evaluated on the stem (the word minus the suffix).
#[derive(Clone, Copy, Debug)]
enum Cond {
    Always,
    /// `_has_positive_measure(stem)`.
    MeasureGt0,
    /// `_measure(stem) > 1`.
    MeasureGt1,
    /// Step 4's `(m>1 and (*S or *T)) ION`.
    Ion,
    /// Step 2's NLTK `logi -> log`: `_has_positive_measure(word[:-3])` (the stem keeps its `l`).
    Logi,
}

/// `(suffix, replacement, condition)`.
type Rule = (&'static [u8], &'static [u8], Cond);

/// `_apply_rule_list(word, rules)` for rules without the `*d` / empty-suffix
/// specials: the first rule whose suffix matches decides; if its condition
/// fails the word is returned unchanged (no further rules are tried).
fn apply_rules(w: &mut Vec<u8>, rules: &[Rule]) {
    let Some(&last) = w.last() else { return };
    for &(suffix, replacement, cond) in rules {
        // Suffixes are non-empty. Rules ending in another letter can never
        // match together with this one, so skipping them keeps rule order.
        if suffix[suffix.len() - 1] == last && w.ends_with(suffix) {
            let k = w.len() - suffix.len();
            let stem = &w[..k];
            let ok = match cond {
                Cond::Always => true,
                Cond::MeasureGt0 => measure(stem) > 0,
                Cond::MeasureGt1 => measure(stem) > 1,
                Cond::Ion => measure(stem) > 1 && matches!(stem.last(), Some(b's' | b't')),
                Cond::Logi => measure(&w[..w.len() - 3]) > 0,
            };
            if ok {
                w.truncate(k);
                w.extend_from_slice(replacement);
            }
            return;
        }
    }
}

/// Step 1a: plurals (`sses`, `ies`, `ss`, `s`), with NLTK's 4-letter `ies -> ie`.
fn step1a(w: &mut Vec<u8>) {
    if w.len() == 4 && w.ends_with(b"ies") {
        w.pop(); // "ies" -> "ie"
        return;
    }
    apply_rules(
        w,
        &[
            (b"sses", b"ss", Cond::Always),
            (b"ies", b"i", Cond::Always),
            (b"ss", b"ss", Cond::Always),
            (b"s", b"", Cond::Always),
        ],
    );
}

/// Step 1b: `eed`, `ed`, `ing` and the tidying-up rules after `ed`/`ing`,
/// with NLTK's `ied` rule first.
fn step1b(w: &mut Vec<u8>) {
    if w.ends_with(b"ied") {
        let n = w.len();
        w.truncate(n - 3);
        w.extend_from_slice(if n == 4 { b"ie" } else { b"i" });
        return;
    }
    if w.ends_with(b"eed") {
        if measure(&w[..w.len() - 3]) > 0 {
            w.pop(); // "eed" -> "ee"
        }
        return;
    }
    let mut cut = None;
    for suffix in [&b"ed"[..], b"ing"] {
        if w.ends_with(suffix) {
            let k = w.len() - suffix.len();
            if contains_vowel(&w[..k]) {
                cut = Some(k);
                break;
            }
        }
    }
    let Some(k) = cut else { return };
    w.truncate(k); // the intermediate stem
    if w.ends_with(b"at") || w.ends_with(b"bl") || w.ends_with(b"iz") {
        w.push(b'e'); // AT -> ATE, BL -> BLE, IZ -> IZE
        return;
    }
    if ends_double_consonant(w) {
        // (*d and not (*L or *S or *Z)) -> single letter
        if !matches!(w[w.len() - 1], b'l' | b's' | b'z') {
            w.pop();
        }
        return;
    }
    // (m=1 and *o) -> E
    if measure(w) == 1 && ends_cvc(w) {
        w.push(b'e');
    }
}

/// Step 1c: `y -> i` when the `y` follows a consonant that is not the first letter.
fn step1c(w: &mut [u8]) {
    let n = w.len();
    if n > 0 && w[n - 1] == b'y' {
        let stem = &w[..n - 1];
        if stem.len() > 1 && is_consonant(stem, stem.len() - 1) {
            w[n - 1] = b'i';
        }
    }
}

const STEP2: &[Rule] = &[
    (b"ational", b"ate", Cond::MeasureGt0),
    (b"tional", b"tion", Cond::MeasureGt0),
    (b"enci", b"ence", Cond::MeasureGt0),
    (b"anci", b"ance", Cond::MeasureGt0),
    (b"izer", b"ize", Cond::MeasureGt0),
    (b"bli", b"ble", Cond::MeasureGt0),
    (b"alli", b"al", Cond::MeasureGt0),
    (b"entli", b"ent", Cond::MeasureGt0),
    (b"eli", b"e", Cond::MeasureGt0),
    (b"ousli", b"ous", Cond::MeasureGt0),
    (b"ization", b"ize", Cond::MeasureGt0),
    (b"ation", b"ate", Cond::MeasureGt0),
    (b"ator", b"ate", Cond::MeasureGt0),
    (b"alism", b"al", Cond::MeasureGt0),
    (b"iveness", b"ive", Cond::MeasureGt0),
    (b"fulness", b"ful", Cond::MeasureGt0),
    (b"ousness", b"ous", Cond::MeasureGt0),
    (b"aliti", b"al", Cond::MeasureGt0),
    (b"iviti", b"ive", Cond::MeasureGt0),
    (b"biliti", b"ble", Cond::MeasureGt0),
    (b"fulli", b"ful", Cond::MeasureGt0),
    (b"logi", b"log", Cond::Logi),
];

/// Step 2. NLTK applies `alli -> al` first and, if it fires, runs step 2
/// again on the result (which can't end in `alli`, so this loops at most once).
fn step2(w: &mut Vec<u8>) {
    while w.ends_with(b"alli") && measure(&w[..w.len() - 4]) > 0 {
        let n = w.len();
        w.truncate(n - 2); // "alli" -> "al"
    }
    apply_rules(w, STEP2);
}

const STEP3: &[Rule] = &[
    (b"icate", b"ic", Cond::MeasureGt0),
    (b"ative", b"", Cond::MeasureGt0),
    (b"alize", b"al", Cond::MeasureGt0),
    (b"iciti", b"ic", Cond::MeasureGt0),
    (b"ical", b"ic", Cond::MeasureGt0),
    (b"ful", b"", Cond::MeasureGt0),
    (b"ness", b"", Cond::MeasureGt0),
];

fn step3(w: &mut Vec<u8>) {
    apply_rules(w, STEP3);
}

const STEP4: &[Rule] = &[
    (b"al", b"", Cond::MeasureGt1),
    (b"ance", b"", Cond::MeasureGt1),
    (b"ence", b"", Cond::MeasureGt1),
    (b"er", b"", Cond::MeasureGt1),
    (b"ic", b"", Cond::MeasureGt1),
    (b"able", b"", Cond::MeasureGt1),
    (b"ible", b"", Cond::MeasureGt1),
    (b"ant", b"", Cond::MeasureGt1),
    (b"ement", b"", Cond::MeasureGt1),
    (b"ment", b"", Cond::MeasureGt1),
    (b"ent", b"", Cond::MeasureGt1),
    (b"ion", b"", Cond::Ion),
    (b"ou", b"", Cond::MeasureGt1),
    (b"ism", b"", Cond::MeasureGt1),
    (b"ate", b"", Cond::MeasureGt1),
    (b"iti", b"", Cond::MeasureGt1),
    (b"ous", b"", Cond::MeasureGt1),
    (b"ive", b"", Cond::MeasureGt1),
    (b"ize", b"", Cond::MeasureGt1),
];

fn step4(w: &mut Vec<u8>) {
    apply_rules(w, STEP4);
}

/// Step 5a: `(m>1) E ->` and `(m=1 and not *o) E ->` (both conditions are tried).
fn step5a(w: &mut Vec<u8>) {
    if w.last() == Some(&b'e') {
        let stem = &w[..w.len() - 1];
        let m = measure(stem);
        if m > 1 || (m == 1 && !ends_cvc(stem)) {
            w.pop();
        }
    }
}

/// Step 5b: `(m > 1 and *d and *L) -> single letter`, measured on the word minus one `l`.
fn step5b(w: &mut Vec<u8>) {
    if w.ends_with(b"ll") && measure(&w[..w.len() - 1]) > 1 {
        w.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s(word: &str) -> String {
        String::from_utf8(stem(word.as_bytes()).into_owned()).unwrap()
    }

    #[test]
    fn paper_examples() {
        // Examples from Porter's paper, as nltk's NLTK_EXTENSIONS mode stems them.
        for (word, want) in [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "tie"),
            ("caress", "caress"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("bled", "bled"),
            ("motoring", "motor"),
            ("sing", "sing"),
            ("conflated", "conflat"),
            ("troubled", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("tanned", "tan"),
            ("falling", "fall"),
            ("hissing", "hiss"),
            ("fizzed", "fizz"),
            ("failing", "fail"),
            ("filing", "file"),
            ("happy", "happi"),
            ("sky", "sky"),
            ("relational", "relat"),
            ("conditional", "condit"),
            ("rational", "ration"),
            ("valenci", "valenc"),
            ("digitizer", "digit"),
            ("conformabli", "conform"),
            ("radicalli", "radic"),
            ("differentli", "differ"),
            ("vileli", "vile"),
            ("analogousli", "analog"),
            ("vietnamization", "vietnam"),
            ("predication", "predic"),
            ("operator", "oper"),
            ("feudalism", "feudal"),
            ("decisiveness", "decis"),
            ("hopefulness", "hope"),
            ("callousness", "callous"),
            ("formaliti", "formal"),
            ("sensitiviti", "sensit"),
            ("sensibiliti", "sensibl"),
            ("triplicate", "triplic"),
            ("formative", "form"),
            ("formalize", "formal"),
            ("electriciti", "electr"),
            ("electrical", "electr"),
            ("hopeful", "hope"),
            ("goodness", "good"),
            ("revival", "reviv"),
            ("allowance", "allow"),
            ("inference", "infer"),
            ("airliner", "airlin"),
            ("gyroscopic", "gyroscop"),
            ("adjustable", "adjust"),
            ("defensible", "defens"),
            ("irritant", "irrit"),
            ("replacement", "replac"),
            ("adjustment", "adjust"),
            ("dependent", "depend"),
            ("adoption", "adopt"),
            ("homologou", "homolog"),
            ("communism", "commun"),
            ("activate", "activ"),
            ("angulariti", "angular"),
            ("homologous", "homolog"),
            ("effective", "effect"),
            ("bowdlerize", "bowdler"),
            ("probate", "probat"),
            ("rate", "rate"),
            ("cease", "ceas"),
            ("controll", "control"),
            ("roll", "roll"),
        ] {
            assert_eq!(s(word), want, "{word}");
        }
    }

    #[test]
    fn nltk_extensions() {
        for (word, want) in [
            // irregular forms
            ("skies", "sky"),
            ("dying", "die"),
            ("lying", "lie"),
            ("tying", "tie"),
            ("news", "news"),
            ("innings", "inning"),
            ("outings", "outing"),
            ("cannings", "canning"),
            ("howe", "howe"),
            ("proceed", "proceed"),
            ("exceed", "exceed"),
            ("succeed", "succeed"),
            // 4-letter ies / ied, spied -> spi
            ("dies", "die"),
            ("died", "die"),
            ("spied", "spi"),
            ("flies", "fli"),
            // y -> i only after a consonant that is not the first letter
            ("enjoy", "enjoy"),
            ("spy", "spi"),
            ("by", "by"),
            // fulli, logi
            ("hopefulli", "hope"),
            ("archaeologi", "archaeolog"),
            ("geologi", "geolog"),
            // short words
            ("a", "a"),
            ("is", "is"),
            ("as", "as"),
            ("Running", "run"),
            ("1990s", "1990"),
            ("yyyy", "yyyi"),
        ] {
            assert_eq!(s(word), want, "{word}");
        }
    }

    #[test]
    fn consonants_and_measure() {
        assert!(is_consonant(b"toy", 2)); // y after a vowel
        assert!(!is_consonant(b"syzygy", 1)); // y after a consonant
        assert!(is_consonant(b"y", 0));
        assert!(!is_consonant(b"yy", 1));
        assert!(is_consonant(b"yyy", 2));
        for (w, m) in [
            ("tr", 0),
            ("ee", 0),
            ("tree", 0),
            ("y", 0),
            ("by", 0),
            ("trouble", 1),
            ("oats", 1),
            ("trees", 1),
            ("ivy", 1),
            ("troubles", 2),
            ("private", 2),
            ("oaten", 2),
            ("orrery", 2),
            ("", 0),
        ] {
            assert_eq!(measure(w.as_bytes()), m, "{w}");
        }
        // consonant_flags agrees with the iterative definition everywhere.
        for w in ["yyyyay", "ayyy", "syzygy", "toy", "yay", "y0y1yy"] {
            let flags: Vec<bool> = consonant_flags(w.as_bytes()).collect();
            let direct: Vec<bool> = (0..w.len())
                .map(|i| is_consonant(w.as_bytes(), i))
                .collect();
            assert_eq!(flags, direct, "{w}");
        }
        assert!(!contains_vowel(b""));
        assert!(!contains_vowel(b"y"));
        assert!(contains_vowel(b"ty"));
    }

    #[test]
    fn unchanged_words_are_borrowed() {
        let w = b"sing";
        assert!(matches!(stem(w), Cow::Borrowed(b) if b.as_ptr() == w.as_ptr()));
        assert!(matches!(stem(b"cats"), Cow::Owned(_)));
    }
}
