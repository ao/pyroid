// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc., and numpy
// (BSD-3-Clause), © NumPy Developers; modified.
//! datatrove's `C4BadWordsFilter`.
//!
//! datatrove drops a document when `badwords_regex.search(doc.text.lower())`
//! finds one of its language's words, where the regex is
//! `(?:\W|^)(w1|w2|...)(?:\W|$)` (`"|".join(words)` for `ja`, `th` and `zh`,
//! which need no word separation), unless `keep_fraction > 0` and
//! `uniform() < keep_fraction` with numpy's `default_rng(seed)`.
//!
//! - [`BadWords`]: the regex search, as a byte trie walked from every
//!   position where a word may start (whether `re` finds *a* match does not
//!   depend on the alternation order, so the order of datatrove's set does
//!   not matter); `\w` is `re`'s, for the interpreter's Unicode version.
//! - [`Pcg64`] and [`seed_sequence`]: numpy's `PCG64` bit generator seeded by
//!   `SeedSequence`, and `Generator.uniform()`, bit for bit.

use std::borrow::Cow;

use super::unicode::is_word;
use crate::dedup::{TextNormConfig, simplify_text};
use crate::ftfy::UnicodeVersion;

/// The languages datatrove has a word list for (`_BADWORDS_LANGS`).
pub const LANGUAGES: [&str; 28] = [
    "ar",
    "cs",
    "da",
    "de",
    "en",
    "eo",
    "es",
    "fa",
    "fi",
    "fil",
    "fr",
    "fr-CA-u-sd-caqc",
    "hi",
    "hu",
    "it",
    "ja",
    "kab",
    "ko",
    "nl",
    "no",
    "pl",
    "pt",
    "ru",
    "sv",
    "th",
    "tlh",
    "tr",
    "zh",
];

/// Languages whose words match anywhere, without word separation.
pub const NO_BOUNDARY_LANGUAGES: [&str; 3] = ["ja", "th", "zh"];

/// `text.lower()` (CPython's, per Unicode version).
pub fn py_lower(text: &str, version: UnicodeVersion) -> Cow<'_, str> {
    let cfg = TextNormConfig {
        lowercase: true,
        norm_whitespace: false,
        remove_punctuation: false,
        norm_unicode_diacritics: false,
        norm_numbers: false,
        norm_weekdays: false,
        norm_monthnames: false,
    };
    // simplify_text(lowercase only) is `text.lower().strip()`. `lower()` maps
    // whitespace to itself and nothing else to whitespace, and whitespace
    // blocks CPython's only context rule (Final_Sigma), so the stripped
    // whitespace comes back from the original.
    let core = simplify_text(text, &cfg, version);
    let start = text.len() - text.trim_start_matches(crate::text::is_py_whitespace).len();
    let end = text.trim_end_matches(crate::text::is_py_whitespace).len();
    if start >= end {
        return Cow::Borrowed(text);
    }
    match core {
        c if *c == text[start..end] => Cow::Borrowed(text),
        c => {
            let mut s = String::with_capacity(c.len() + text.len() - (end - start));
            s.push_str(&text[..start]);
            s.push_str(&c);
            s.push_str(&text[end..]);
            Cow::Owned(s)
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Node {
    /// Children by next byte, sorted.
    next: Vec<(u8, u32)>,
    /// A word ends here.
    end: bool,
}

/// One language's word list, compiled.
#[derive(Clone, Debug)]
pub struct BadWords {
    nodes: Vec<Node>,
    /// First bytes of the words (bit per byte value).
    first: [u64; 4],
    /// `(?:\W|^)...(?:\W|$)` around the words (every language but ja/th/zh).
    boundaries: bool,
}

impl BadWords {
    /// Compile `words` (any order, duplicates allowed; they are matched
    /// literally, as `re.escape` makes them).
    pub fn new<S: AsRef<str>>(words: impl IntoIterator<Item = S>, boundaries: bool) -> Self {
        let mut nodes = vec![Node::default()];
        let mut first = [0u64; 4];
        for w in words {
            let w = w.as_ref().as_bytes();
            if let Some(&b) = w.first() {
                first[usize::from(b >> 6)] |= 1 << (b & 63);
            }
            let mut n = 0usize;
            for &b in w {
                n = match nodes[n].next.binary_search_by_key(&b, |e| e.0) {
                    Ok(k) => nodes[n].next[k].1 as usize,
                    Err(k) => {
                        let id = u32::try_from(nodes.len()).expect("trie fits in u32");
                        nodes[n].next.insert(k, (b, id));
                        nodes.push(Node::default());
                        id as usize
                    }
                };
            }
            nodes[n].end = true;
        }
        Self {
            nodes,
            first,
            boundaries,
        }
    }

    /// Whether words must be flanked by non-word characters.
    pub fn boundaries(&self) -> bool {
        self.boundaries
    }

    /// `badwords_regex.search(text.lower()) is not None`.
    pub fn found(&self, text: &str, version: UnicodeVersion) -> bool {
        let lower = py_lower(text, version);
        self.found_in_lowered(&lower, version)
    }

    /// [`Self::found`] for text that is already `lower()`ed.
    pub fn found_in_lowered(&self, text: &str, version: UnicodeVersion) -> bool {
        let empty_word = self.nodes[0].end;
        if !self.boundaries {
            return empty_word
                || (0..text.len())
                    .any(|i| text.is_char_boundary(i) && self.walk(text, i, |_| true));
        }
        let right_ok = |j: usize| {
            text[j..]
                .chars()
                .next()
                .is_none_or(|c| !is_word(c, version))
        };
        // Starts: the beginning, and after every non-word character.
        if self.walk(text, 0, right_ok) || (empty_word && right_ok(0)) {
            return true;
        }
        for (i, c) in text.char_indices() {
            if is_word(c, version) {
                continue;
            }
            let j = i + c.len_utf8();
            if self.walk(text, j, right_ok) || (empty_word && right_ok(j)) {
                return true;
            }
        }
        false
    }

    /// Does a (non-empty) word start at `i` and end where `ok(end)`?
    #[inline]
    fn walk(&self, text: &str, i: usize, ok: impl Fn(usize) -> bool) -> bool {
        let b = text.as_bytes();
        let Some(&c0) = b.get(i) else {
            return false;
        };
        if self.first[usize::from(c0 >> 6)] & (1 << (c0 & 63)) == 0 {
            return false;
        }
        let mut n = 0usize;
        for (k, &c) in b[i..].iter().enumerate() {
            match self.nodes[n].next.binary_search_by_key(&c, |e| e.0) {
                Ok(x) => n = self.nodes[n].next[x].1 as usize,
                Err(_) => return false,
            }
            // Words are whole UTF-8 strings, so a word end is a char boundary.
            if self.nodes[n].end && ok(i + k + 1) {
                return true;
            }
        }
        false
    }
}

// --------------------------------------------------------------------------
// numpy's PCG64 + SeedSequence (numpy/random/_pcg64.pyx, bit_generator.pyx,
// src/pcg64/pcg64.h).
// --------------------------------------------------------------------------

const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

/// numpy's `PCG64` state: a 128-bit LCG with the XSL-RR output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pcg64 {
    pub state: u128,
    pub inc: u128,
}

impl Pcg64 {
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
    }

    /// `pcg64_set_seed(seed, inc)`: numpy's `pcg_setseq_128_srandom_r`.
    pub fn from_seed(initstate: u128, initseq: u128) -> Self {
        let mut g = Self {
            state: 0,
            inc: (initseq << 1) | 1,
        };
        g.step();
        g.state = g.state.wrapping_add(initstate);
        g.step();
        g
    }

    /// `next_uint64`.
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        #[allow(clippy::cast_possible_truncation)] // the halves of the state
        let xored = ((s >> 64) as u64) ^ (s as u64);
        #[allow(clippy::cast_possible_truncation)] // the top 6 bits
        let rot = (s >> 122) as u32;
        xored.rotate_right(rot)
    }

    /// `Generator.uniform()` (`low=0.0, high=1.0`): `next_double`.
    #[allow(clippy::cast_precision_loss)] // 53 bits: exact
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
    }
}

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const POOL_SIZE: usize = 4;

fn hashmix(value: u32, hash_const: &mut u32) -> u32 {
    let mut v = value ^ *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    v = v.wrapping_mul(*hash_const);
    v ^ (v >> 16)
}

fn mix(x: u32, y: u32) -> u32 {
    let r = MIX_MULT_L
        .wrapping_mul(x)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(y));
    r ^ (r >> 16)
}

/// `PCG64(SeedSequence(entropy))` where `entropy` is the seed's 32-bit words,
/// least significant first (`_coerce_to_uint32_array`: `[0]` for 0), without
/// a spawn key: what `numpy.random.default_rng(seed)` builds for an int seed.
pub fn seed_sequence(entropy: &[u32]) -> Pcg64 {
    let mut hc = INIT_A;
    let mut pool = [0u32; POOL_SIZE];
    for (i, p) in pool.iter_mut().enumerate() {
        *p = hashmix(entropy.get(i).copied().unwrap_or(0), &mut hc);
    }
    for src in 0..POOL_SIZE {
        for dst in 0..POOL_SIZE {
            if src != dst {
                pool[dst] = mix(pool[dst], hashmix(pool[src], &mut hc));
            }
        }
    }
    for &e in entropy.iter().skip(POOL_SIZE) {
        for p in &mut pool {
            *p = mix(*p, hashmix(e, &mut hc));
        }
    }
    // generate_state(4, uint64): 8 words, viewed as 4 little-endian u64s.
    let mut hb = INIT_B;
    let mut words = [0u32; 8];
    for (i, w) in words.iter_mut().enumerate() {
        let mut v = pool[i % POOL_SIZE] ^ hb;
        hb = hb.wrapping_mul(MULT_B);
        v = v.wrapping_mul(hb);
        *w = v ^ (v >> 16);
    }
    let u = |k: usize| u64::from(words[2 * k]) | (u64::from(words[2 * k + 1]) << 32);
    let seed = (u128::from(u(0)) << 64) | u128::from(u(1));
    let inc = (u128::from(u(2)) << 64) | u128::from(u(3));
    Pcg64::from_seed(seed, inc)
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    #[test]
    fn boundaries() {
        let b = BadWords::new(["bad", "two words", "x-y"], true);
        assert!(b.found("this is bad", V));
        assert!(b.found("BAD.", V));
        assert!(b.found("bad", V));
        assert!(!b.found("badly", V));
        assert!(!b.found("_bad", V));
        assert!(!b.found("ébad", V));
        assert!(b.found("-bad-", V));
        assert!(b.found("Two Words!", V));
        assert!(b.found("a x-y b", V));
        assert!(!b.found("", V));
        let cjk = BadWords::new(["坏"], false);
        assert!(cjk.found("很坏的", V));
        assert!(!cjk.found("很好", V));
    }

    #[test]
    fn empty_word() {
        let b = BadWords::new([""], true);
        assert!(b.found("", V));
        assert!(!b.found("a b", V));
        assert!(b.found(" ", V));
        assert!(!b.found("ab", V));
        assert!(b.found("a  b", V));
        assert!(b.found(" ab", V));
        assert!(BadWords::new([""], false).found("anything", V));
    }

    #[test]
    fn lower_keeps_outer_whitespace() {
        assert_eq!(py_lower("  ABC Σ  ", V), "  abc σ  ");
        assert_eq!(py_lower("ΑΣ ", V), "ας ");
        assert!(matches!(py_lower(" abc ", V), Cow::Borrowed(_)));
        assert_eq!(py_lower("İ", V), "i\u{307}");
    }

    #[test]
    fn numpy_default_rng() {
        // numpy 2.5.4: default_rng(seed).bit_generator.state, then three
        // .uniform() draws (as f64 bits). Seeds 0, 1, 42, 2**32 + 5, 2**200 + 7.
        #[allow(clippy::type_complexity)]
        let cases: [(&[u32], u128, u128, [u64; 3]); 5] = [
            (
                &[0],
                0x1aa1b5345996452d09585eb7a69561e3,
                0x418ddadb3af71a82588133bc447873a9,
                [0x3fe461fd79fb3850, 0x3fd1442f7e20b674, 0x3fa4fa7b529d9bd0],
            ),
            (
                &[1],
                0x9c5b484bfedb756c2a6e7d6f320fbc7e,
                0x922af2da2645f895a19857b95740937b,
                [0x3fe060d7be6f245c, 0x3fee6a32d7782a55, 0x3fc273d27b04760c],
            ),
            (
                &[42],
                0xcea44f6798798f2aacbc7c9d68860ac8,
                0xfa505436c9a8416e66caf2e28d25abff,
                [0x3fe8c43f79a2db24, 0x3fdc16959869e47e, 0x3feb79a2584ddb42],
            ),
            (
                &[5, 1],
                0x2931d0c6dbee79bfece8058fe085a345,
                0x271168ffcdb9ee72bdd751b35ac873c3,
                [0x3fe8c647d8643bd7, 0x3fde20520871ce80, 0x3fe644a6db1c1d82],
            ),
            (
                &[7, 0, 0, 0, 0, 0, 256],
                0x6561e8596ba750359c8737013f1902d2,
                0xce61f6b3634bc82d580e75670d294bed,
                [0x3fec29c317a2d499, 0x3fd685935c8e4b0e, 0x3fe132947d08103c],
            ),
        ];
        for (entropy, state, inc, draws) in cases {
            let mut g = seed_sequence(entropy);
            assert_eq!((g.state, g.inc), (state, inc), "{entropy:?}");
            for d in draws {
                assert_eq!(g.uniform().to_bits(), d, "{entropy:?}");
            }
        }
    }
}
