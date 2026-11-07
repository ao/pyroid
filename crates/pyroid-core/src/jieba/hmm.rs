// Derived from jieba 0.42.1 (MIT), © 2013 Sun Junyi; modified.
//! `jieba.finalseg`: the BMES hidden Markov model that segments runs of
//! characters the dictionary route left as single characters.
//!
//! The probabilities are jieba's `prob_start.py`, `prob_trans.py` and
//! `prob_emit.py`, stored bit for bit in `data/hmm.bin` (see `generate.py`).
//! `finalseg.cut` only ever runs the Viterbi on characters in
//! U+4E00..=U+9FD5 (its `re_han`), so the emission table covers that range.

use std::sync::LazyLock;

use super::ForceSplit;

/// `finalseg.MIN_FLOAT`.
pub(crate) const MIN_FLOAT: f64 = -3.14e100;

/// First code point of `re_han` (`[一-鿕]`).
pub(crate) const HAN_FIRST: u32 = 0x4E00;
/// Last code point of `re_han`.
pub(crate) const HAN_LAST: u32 = 0x9FD5;
const HAN_COUNT: usize = (HAN_LAST - HAN_FIRST + 1) as usize;

static HMM_BIN: &[u8] = include_bytes!("data/hmm.bin");

/// States in index order; the order is also the order of their letters
/// (`'B' < 'E' < 'M' < 'S'`), which breaks `max` ties in `viterbi`.
const B: usize = 0;
const E: usize = 1;
const M: usize = 2;
const S: usize = 3;

/// `PrevStatus`, in jieba's order.
const PREV: [[usize; 2]; 4] = [
    [E, S], // B: 'ES'
    [B, M], // E: 'BM'
    [M, B], // M: 'MB'
    [S, E], // S: 'SE'
];

struct Model {
    start: [f64; 4],
    /// `trans[from][to]`, `MIN_FLOAT` where jieba has no entry.
    trans: [[f64; 4]; 4],
    /// Per state: the distinct emission log-probabilities, index 0 being
    /// `MIN_FLOAT` (no entry for the character).
    values: [Vec<f64>; 4],
    /// Per character of `re_han`: the value index of each state.
    index: Vec<[u16; 4]>,
}

static MODEL: LazyLock<Model> = LazyLock::new(|| parse(HMM_BIN));

struct Reader<'a> {
    b: &'a [u8],
}

impl Reader<'_> {
    fn take<const N: usize>(&mut self) -> [u8; N] {
        let (head, rest) = self.b.split_at(N);
        self.b = rest;
        head.try_into().expect("N bytes")
    }
    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take())
    }
    fn u16(&mut self) -> u16 {
        u16::from_le_bytes(self.take())
    }
    fn f64(&mut self) -> f64 {
        f64::from_bits(u64::from_le_bytes(self.take()))
    }
}

/// `data/hmm.bin`: `b"JBHM"`, u32 version 1, start\[4\], trans\[4\]\[4\],
/// then per state a u32 count and that many f64 values, then per character of
/// U+4E00..=U+9FD5 four u16 value indices (B, E, M, S). Little-endian.
fn parse(bin: &[u8]) -> Model {
    let mut r = Reader { b: bin };
    assert_eq!(&r.take::<4>(), b"JBHM", "hmm.bin magic");
    assert_eq!(r.u32(), 1, "hmm.bin version");
    let mut start = [0.0; 4];
    for v in &mut start {
        *v = r.f64();
    }
    let mut trans = [[0.0; 4]; 4];
    for row in &mut trans {
        for v in row {
            *v = r.f64();
        }
    }
    let values: [Vec<f64>; 4] = std::array::from_fn(|_| {
        let n = r.u32() as usize;
        let mut v = Vec::with_capacity(n + 1);
        v.push(MIN_FLOAT);
        for _ in 0..n {
            v.push(r.f64());
        }
        v
    });
    let index: Vec<[u16; 4]> = (0..HAN_COUNT)
        .map(|_| std::array::from_fn(|_| r.u16()))
        .collect();
    assert!(r.b.is_empty(), "hmm.bin trailing bytes");
    for (s, ix) in index.iter().flat_map(|ix| ix.iter().enumerate()) {
        assert!((*ix as usize) < values[s].len(), "hmm.bin index");
    }
    Model {
        start,
        trans,
        values,
        index,
    }
}

impl Model {
    #[inline]
    fn emit(&self, state: usize, c: char) -> f64 {
        let i = (c as u32).wrapping_sub(HAN_FIRST) as usize;
        match self.index.get(i) {
            Some(ix) => self.values[state][ix[state] as usize],
            None => MIN_FLOAT,
        }
    }
}

/// `max([(p0, s0), (p1, s1)])`: tuples compare by probability, then by the
/// state's letter; the first of equals wins.
#[inline]
fn max2(a: (f64, usize), b: (f64, usize)) -> (f64, usize) {
    #[allow(clippy::float_cmp)] // Python's tuple order: == on the floats
    let b_greater = if b.0 == a.0 { b.1 > a.1 } else { b.0 > a.0 };
    if b_greater { b } else { a }
}

/// Reused buffers for the Viterbi.
#[derive(Debug, Default)]
pub(crate) struct Scratch {
    chars: Vec<char>,
    /// Back pointers: `back[t][y]` is the best previous state of `y` at `t`.
    back: Vec<[u8; 4]>,
    states: Vec<u8>,
}

/// `viterbi(obs, 'BMES', start_P, trans_P, emit_P)`: the best state of each
/// character, into `s.states`.
#[allow(clippy::many_single_char_names)] // obs, states, V as in jieba
fn viterbi(m: &Model, s: &mut Scratch) {
    let n = s.chars.len();
    s.back.clear();
    s.back.resize(n, [0; 4]);
    let mut v = [0.0f64; 4];
    for y in [B, M, E, S] {
        v[y] = m.start[y] + m.emit(y, s.chars[0]);
    }
    for t in 1..n {
        let c = s.chars[t];
        let mut next = [0.0f64; 4];
        for y in [B, M, E, S] {
            let em = m.emit(y, c);
            let [y0, y1] = PREV[y];
            let best = max2(
                (v[y0] + m.trans[y0][y] + em, y0),
                (v[y1] + m.trans[y1][y] + em, y1),
            );
            next[y] = best.0;
            #[allow(clippy::cast_possible_truncation)] // a state index
            {
                s.back[t][y] = best.1 as u8;
            }
        }
        v = next;
    }
    let (_, mut state) = max2((v[E], E), (v[S], S));
    s.states.clear();
    s.states.resize(n, 0);
    for t in (0..n).rev() {
        #[allow(clippy::cast_possible_truncation)]
        {
            s.states[t] = state as u8;
        }
        state = s.back[t][state] as usize;
    }
}

/// `finalseg.__cut` on a run of `re_han` characters (`blk`, non-empty),
/// with `Force_Split_Words` applied as `finalseg.cut` does.
fn cut_han<'a>(blk: &'a str, fs: &ForceSplit, s: &mut Scratch, out: &mut Vec<&'a str>) {
    let m = &*MODEL;
    s.chars.clear();
    s.chars.extend(blk.chars());
    viterbi(m, s);
    let mut emit = |w: &'a str| {
        if !fs.is_empty() && fs.contains(w) {
            out.extend(w.char_indices().map(|(i, c)| &w[i..i + c.len_utf8()]));
        } else {
            out.push(w);
        }
    };
    let (mut begin, mut next) = (0usize, 0usize);
    for ((i, c), &st) in blk.char_indices().zip(&s.states) {
        let end = i + c.len_utf8();
        match st as usize {
            B => begin = i,
            E => {
                emit(&blk[begin..end]);
                next = end;
            }
            S => {
                emit(&blk[i..end]);
                next = end;
            }
            _ => {}
        }
    }
    if next < blk.len() {
        emit(&blk[next..]);
    }
}

#[inline]
pub(crate) fn is_re_han(c: char) -> bool {
    (HAN_FIRST..=HAN_LAST).contains(&(c as u32))
}

/// `finalseg.re_skip.split(blk)`'s non-empty pieces, for a `blk` of the
/// characters jieba's blocks can hold besides `re_han` (ASCII letters and
/// digits, `+#&._%-`): `[a-zA-Z0-9]+(?:\.\d+)?%?` matches and the text
/// between them. `\d` only meets ASCII here.
fn split_skip<'a>(blk: &'a str, out: &mut Vec<&'a str>) {
    let b = blk.as_bytes();
    let n = b.len();
    let (mut last, mut i) = (0usize, 0usize);
    while i < n {
        if !b[i].is_ascii_alphanumeric() {
            // Not ASCII (cannot happen for jieba's blocks) or not a match start.
            i += if b[i] < 0x80 {
                1
            } else {
                blk[i..].chars().next().map_or(1, char::len_utf8)
            };
            continue;
        }
        let start = i;
        while i < n && b[i].is_ascii_alphanumeric() {
            i += 1;
        }
        if i + 1 < n && b[i] == b'.' && b[i + 1].is_ascii_digit() {
            i += 1;
            while i < n && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        if i < n && b[i] == b'%' {
            i += 1;
        }
        if start > last {
            out.push(&blk[last..start]);
        }
        out.push(&blk[start..i]);
        last = i;
    }
    if last < n {
        out.push(&blk[last..]);
    }
}

/// `finalseg.cut(buf)`.
pub(crate) fn cut<'a>(buf: &'a str, fs: &ForceSplit, s: &mut Scratch, out: &mut Vec<&'a str>) {
    let mut rest = buf;
    while !rest.is_empty() {
        let han = rest.chars().next().is_some_and(is_re_han);
        let end = rest
            .char_indices()
            .find(|&(_, c)| is_re_han(c) != han)
            .map_or(rest.len(), |(i, _)| i);
        let (blk, tail) = rest.split_at(end);
        if han {
            cut_han(blk, fs, s, out);
        } else {
            split_skip(blk, out);
        }
        rest = tail;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_loads() {
        let m = &*MODEL;
        // prob_start.py
        assert_eq!(
            m.start[B].to_bits(),
            (-0.262_686_608_092_500_16f64).to_bits()
        );
        assert_eq!(m.start[E], MIN_FLOAT);
        assert_eq!(m.start[M], MIN_FLOAT);
        assert_eq!(
            m.start[S].to_bits(),
            (-1.465_263_339_853_767_8f64).to_bits()
        );
        // prob_trans.py: B -> E
        assert_eq!(
            m.trans[B][E].to_bits(),
            (-0.510_825_623_765_99f64).to_bits()
        );
        // prob_emit.py: 'B': {'一': -3.6544978750449433, ...}
        assert_eq!(
            m.emit(B, '\u{4e00}').to_bits(),
            (-3.654_497_875_044_943_3f64).to_bits()
        );
        assert_eq!(m.emit(B, '\u{4e02}'), MIN_FLOAT);
    }

    fn fcut(s: &str) -> Vec<&str> {
        let mut out = Vec::new();
        cut(s, &ForceSplit::default(), &mut Scratch::default(), &mut out);
        out
    }

    #[test]
    fn skip_split() {
        assert_eq!(fcut("3.14.15"), ["3.14", ".", "15"]);
        assert_eq!(fcut("a.b_c-5%"), ["a", ".", "b", "_", "c", "-", "5%"]);
        assert_eq!(fcut("1.5e3"), ["1.5", "e3"]);
        assert_eq!(fcut("+#"), ["+#"]);
    }

    #[test]
    fn viterbi_new_words() {
        // jieba's README: 杭研 is not in the dictionary; the HMM finds it.
        assert_eq!(fcut("杭研"), ["杭研"]);
    }
}
