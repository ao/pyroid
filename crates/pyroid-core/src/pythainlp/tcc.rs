// Derived from PyThaiNLP 5.4.0 (Apache-2.0), © 2016-2026 PyThaiNLP Project,
// and datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! Thai Character Clusters as `pythainlp.tokenize.tcc_p` finds them (the
//! improved rules that newmm uses, not `tcc.py`).
//!
//! `_PAT_TCC` is the alternation of 29 templates, expanded as the module
//! does (`k` -> `(cc?[dิ]?[์])?`, then `c` -> `[ก-ฮ]`, `t` -> `[่-๋]?`,
//! `d` -> `ูุ`). `re` tries the alternatives left to right and returns the
//! first one that matches (not the longest), backtracking inside an
//! alternative: greedy `?` first takes the character, then skips it. Each
//! cluster is matched on the remaining text (`_PAT_TCC.match(text[p:])`), so
//! `$` in the lookahead `(?=[เ-ไก-ฮ]|$)` holds at the end of the text and
//! just before a final `"\n"`. A position where nothing matches is a cluster
//! of one character.

/// A character class of the expanded templates.
#[derive(Clone, Copy, Debug)]
enum Set {
    Lit(char),
    Range(char, char),
    Any(&'static [char]),
}

impl Set {
    #[inline]
    fn has(self, c: char) -> bool {
        match self {
            Set::Lit(x) => c == x,
            Set::Range(lo, hi) => (lo..=hi).contains(&c),
            Set::Any(cs) => cs.contains(&c),
        }
    }
}

/// One element of a template.
#[derive(Clone, Copy, Debug)]
enum El {
    /// One character of the set.
    One(Set),
    /// `[...]?`: an optional character of the set, greedy.
    Opt(Set),
    /// `(?=[เ-ไก-ฮ]|$)`.
    Ahead,
    /// `k`: `([ก-ฮ][ก-ฮ]?[ูุิ]?[์])?`. It is the last element of every
    /// template that has it.
    K,
}

use El::{Ahead, K, One, Opt};

/// `c`: `[ก-ฮ]`.
const C: Set = Set::Range('\u{e01}', '\u{e2e}');
/// `t` without its `?`: `[่-๋]`.
const T: Set = Set::Range('\u{e48}', '\u{e4b}');
const fn l(c: char) -> Set {
    Set::Lit(c)
}

/// The group inside `k`: `[ก-ฮ][ก-ฮ]?[ูุิ]?[์]`.
const K_GROUP: [El; 4] = [
    One(C),
    Opt(C),
    Opt(Set::Any(&['\u{e39}', '\u{e38}', '\u{e34}'])),
    One(l('\u{e4c}')),
];

/// `_RE_TCC`, expanded, in order.
const TEMPLATES: [&[El]; 29] = [
    // เc็ck
    &[One(l('\u{e40}')), One(C), One(l('\u{e47}')), One(C), K],
    // เcctาะk
    &[
        One(l('\u{e40}')),
        One(C),
        One(C),
        Opt(T),
        One(l('\u{e32}')),
        One(l('\u{e30}')),
        K,
    ],
    // เccีtยะk
    &[
        One(l('\u{e40}')),
        One(C),
        One(C),
        One(l('\u{e35}')),
        Opt(T),
        One(l('\u{e22}')),
        One(l('\u{e30}')),
        K,
    ],
    // เccีtย(?=[เ-ไก-ฮ]|$)k
    &[
        One(l('\u{e40}')),
        One(C),
        One(C),
        One(l('\u{e35}')),
        Opt(T),
        One(l('\u{e22}')),
        Ahead,
        K,
    ],
    // เcc็ck
    &[
        One(l('\u{e40}')),
        One(C),
        One(C),
        One(l('\u{e47}')),
        One(C),
        K,
    ],
    // เcิc์ck
    &[
        One(l('\u{e40}')),
        One(C),
        One(l('\u{e34}')),
        One(C),
        One(l('\u{e4c}')),
        One(C),
        K,
    ],
    // เcิtck
    &[
        One(l('\u{e40}')),
        One(C),
        One(l('\u{e34}')),
        Opt(T),
        One(C),
        K,
    ],
    // เcีtยะ?k
    &[
        One(l('\u{e40}')),
        One(C),
        One(l('\u{e35}')),
        Opt(T),
        One(l('\u{e22}')),
        Opt(l('\u{e30}')),
        K,
    ],
    // เcืtอะ?k
    &[
        One(l('\u{e40}')),
        One(C),
        One(l('\u{e37}')),
        Opt(T),
        One(l('\u{e2d}')),
        Opt(l('\u{e30}')),
        K,
    ],
    // เc[ิีุู]tย(?=[เ-ไก-ฮ]|$)k
    &[
        One(l('\u{e40}')),
        One(C),
        One(Set::Any(&['\u{e34}', '\u{e35}', '\u{e38}', '\u{e39}'])),
        Opt(T),
        One(l('\u{e22}')),
        Ahead,
        K,
    ],
    // เctา?ะ?k
    &[
        One(l('\u{e40}')),
        One(C),
        Opt(T),
        Opt(l('\u{e32}')),
        Opt(l('\u{e30}')),
        K,
    ],
    // cัtวะk
    &[
        One(C),
        One(l('\u{e31}')),
        Opt(T),
        One(l('\u{e27}')),
        One(l('\u{e30}')),
        K,
    ],
    // c[ัื]tc[ุิะ]?k
    &[
        One(C),
        One(Set::Any(&['\u{e31}', '\u{e37}'])),
        Opt(T),
        One(C),
        Opt(Set::Any(&['\u{e38}', '\u{e34}', '\u{e30}'])),
        K,
    ],
    // c[ิุู]์
    &[
        One(C),
        One(Set::Any(&['\u{e34}', '\u{e38}', '\u{e39}'])),
        One(l('\u{e4c}')),
    ],
    // c[ะ-ู]tk
    &[One(C), One(Set::Range('\u{e30}', '\u{e39}')), Opt(T), K],
    // cรรc์
    &[
        One(C),
        One(l('\u{e23}')),
        One(l('\u{e23}')),
        One(C),
        One(l('\u{e4c}')),
    ],
    // c็
    &[One(C), One(l('\u{e47}'))],
    // ct[ะาำ]?k
    &[
        One(C),
        Opt(T),
        Opt(Set::Any(&['\u{e30}', '\u{e32}', '\u{e33}'])),
        K,
    ],
    // ck
    &[One(C), K],
    // แc็c
    &[One(l('\u{e41}')), One(C), One(l('\u{e47}')), One(C)],
    // แcc์
    &[One(l('\u{e41}')), One(C), One(C), One(l('\u{e4c}'))],
    // แctะ
    &[One(l('\u{e41}')), One(C), Opt(T), One(l('\u{e30}'))],
    // แcc็c
    &[One(l('\u{e41}')), One(C), One(C), One(l('\u{e47}')), One(C)],
    // แccc์
    &[One(l('\u{e41}')), One(C), One(C), One(C), One(l('\u{e4c}'))],
    // โctะ
    &[One(l('\u{e42}')), One(C), Opt(T), One(l('\u{e30}'))],
    // [เ-ไ]ct
    &[One(Set::Range('\u{e40}', '\u{e44}')), One(C), Opt(T)],
    // ก็
    &[One(l('\u{e01}')), One(l('\u{e47}'))],
    // อึ
    &[One(l('\u{e2d}')), One(l('\u{e36}'))],
    // หึ
    &[One(l('\u{e2b}')), One(l('\u{e36}'))],
];

/// `$` (no MULTILINE) at `pos` of `s`: the end, or before a final `"\n"`.
#[inline]
pub(super) fn at_end(s: &[char], pos: usize) -> bool {
    pos == s.len() || (pos + 1 == s.len() && s[pos] == '\n')
}

/// The end of the first match of `seq` at `pos`, in `re`'s backtracking
/// order, or `None`.
fn match_seq(s: &[char], pos: usize, seq: &[El]) -> Option<usize> {
    let Some((&first, rest)) = seq.split_first() else {
        return Some(pos);
    };
    let here = s.get(pos).copied();
    match first {
        One(set) => {
            if here.is_some_and(|c| set.has(c)) {
                match_seq(s, pos + 1, rest)
            } else {
                None
            }
        }
        Opt(set) => {
            if here.is_some_and(|c| set.has(c)) {
                if let Some(e) = match_seq(s, pos + 1, rest) {
                    return Some(e);
                }
            }
            match_seq(s, pos, rest)
        }
        Ahead => {
            let ok = here
                .is_some_and(|c| matches!(c, '\u{e40}'..='\u{e44}' | '\u{e01}'..='\u{e2e}'))
                || at_end(s, pos);
            if ok { match_seq(s, pos, rest) } else { None }
        }
        K => {
            debug_assert!(rest.is_empty(), "k ends its template");
            Some(match_seq(s, pos, &K_GROUP).unwrap_or(pos))
        }
    }
}

/// The length of the cluster at `pos` (at least 1): the first template that
/// matches there, or one character.
#[inline]
pub(super) fn cluster_len(s: &[char], pos: usize) -> usize {
    for t in TEMPLATES {
        if let Some(e) = match_seq(s, pos, t) {
            return e - pos;
        }
    }
    1
}

/// `tcc_pos_array(s)`: `out[i]` is whether a cluster ends at `i`
/// (`out.len() == s.len() + 1`; `out[0]` is false).
pub(super) fn pos_array(s: &[char], out: &mut Vec<bool>) {
    out.clear();
    out.resize(s.len() + 1, false);
    let mut p = 0;
    while p < s.len() {
        p += cluster_len(s, p);
        out[p] = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens(t: &str) -> Vec<usize> {
        let s: Vec<char> = t.chars().collect();
        let mut out = Vec::new();
        let mut p = 0;
        while p < s.len() {
            let n = cluster_len(&s, p);
            out.push(n);
            p += n;
        }
        out
    }

    #[test]
    fn k_is_last() {
        for t in TEMPLATES {
            for (i, e) in t.iter().enumerate() {
                if matches!(e, K) {
                    assert_eq!(i + 1, t.len());
                }
            }
        }
    }

    #[test]
    fn clusters_match_pythainlp() {
        // Recorded from PyThaiNLP 5.4.0 (`tcc_p.segment`), lengths in
        // code points.
        assert_eq!(lens(""), [0usize; 0]);
        assert_eq!(lens("ประเทศไทย"), [1, 2, 2, 1, 2, 1]);
        assert_eq!(lens("สวัสดีครับ"), [1, 3, 2, 1, 3]);
        assert_eq!(lens("abc ไทย"), [1, 1, 1, 1, 2, 1]);
        assert_eq!(lens("แก้ไข"), [3, 2]);
        assert_eq!(lens("กรรม์"), [5]);
        // `(?=[เ-ไก-ฮ]|$)`: `$` also holds before a final "\n".
        assert_eq!(lens("เปลี่ย"), [6]);
        assert_eq!(lens("เปลี่ยน"), [6, 1]);
        assert_eq!(lens("เปลี่ย\n"), [6, 1]);
        assert_eq!(lens("เปลี่ย\n\n"), [2, 3, 1, 1, 1]);
        assert_eq!(lens("เปลี่ย1"), [2, 3, 1, 1]);
    }
}
