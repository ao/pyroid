//! `difflib.SequenceMatcher`'s algorithm over dense element ids.
//!
//! Elements are `u32` ids: the distinct elements of `b` are numbered `0..n` in
//! first-occurrence order (the order of CPython's `b2j` keys), and an element
//! of `a` that is not in `b` is [`NONE`]. Every loop, comparison and tie-break
//! follows CPython's `Lib/difflib.py` (identical in 3.11 to 3.15), so the
//! blocks, opcodes and ratios are the same values, the ratios the same bits.

use super::stop::Ticker;

/// The id of an element of `a` that does not occur in `b`.
pub const NONE: u32 = u32::MAX;

/// Work units `find_longest_match` gathers before it ticks.
const TICK_WORK: usize = 1 << 12;

/// What `__chain_b` decided about an element of `b`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// In `b2j`.
    Normal,
    /// `isjunk(elt)` was true: in `bjunk`, not in `b2j`.
    Junk,
    /// Popular under `autojunk`: in `bpopular`, not in `b2j`.
    Popular,
}

/// `set_seq2(b)`'s result: `b` as ids, and `b2j` (for every id, the
/// positions of `b` holding it, junk and popular ones included: they are also
/// `fullbcount`).
#[derive(Debug, Clone)]
pub struct BIndex {
    b: Vec<u32>,
    kind: Vec<Kind>,
    starts: Vec<u32>,
    pos: Vec<u32>,
}

impl BIndex {
    /// Index `b` (ids below `keys`, numbered by first occurrence). `junk[id]`
    /// is `isjunk`'s verdict (empty: no `isjunk`); `autojunk` is the
    /// truthiness of the matcher's `autojunk`.
    ///
    /// # Panics
    ///
    /// If `b` has `u32::MAX` elements or more (the bindings delegate those).
    pub fn new(b: Vec<u32>, keys: usize, junk: &[bool], autojunk: bool) -> Self {
        let mut index = Self {
            b,
            kind: Vec::new(),
            starts: Vec::new(),
            pos: Vec::new(),
        };
        index.reindex(keys, junk, autojunk);
        index
    }

    /// [`Self::new`] for `b`'s ids in `ids` (left empty), reusing this index's memory.
    pub fn rebuild(&mut self, ids: &mut Vec<u32>, keys: usize, junk: &[bool], autojunk: bool) {
        std::mem::swap(&mut self.b, ids);
        ids.clear();
        self.reindex(keys, junk, autojunk);
    }

    fn reindex(&mut self, keys: usize, junk: &[bool], autojunk: bool) {
        let b = &self.b;
        assert!(u32::try_from(b.len()).is_ok_and(|n| n < u32::MAX));
        let starts = &mut self.starts;
        starts.clear();
        starts.resize(keys + 1, 0);
        for &id in b {
            starts[id as usize + 1] += 1;
        }
        for k in 0..keys {
            starts[k + 1] += starts[k];
        }
        // Fill the positions back to front, counting each start down to its final value.
        let pos = &mut self.pos;
        pos.clear();
        pos.resize(b.len(), 0);
        for (j, &id) in b.iter().enumerate().rev() {
            let end = &mut starts[id as usize + 1];
            *end -= 1;
            #[allow(clippy::cast_possible_truncation)] // checked above
            {
                pos[*end as usize] = j as u32;
            }
        }
        // starts[id + 1] now holds id's start: shift back by one.
        starts.copy_within(1.., 0);
        starts[keys] = u32::try_from(b.len()).unwrap_or(u32::MAX);
        let kind = &mut self.kind;
        kind.clear();
        kind.extend((0..keys).map(|id| {
            if junk.get(id).copied().unwrap_or(false) {
                Kind::Junk
            } else {
                Kind::Normal
            }
        }));
        let n = b.len();
        if autojunk && n >= 200 {
            let ntest = n / 100 + 1;
            for (id, k) in kind.iter_mut().enumerate() {
                if *k == Kind::Normal && (starts[id + 1] - starts[id]) as usize > ntest {
                    *k = Kind::Popular;
                }
            }
        }
    }

    /// `b` as ids.
    pub fn b(&self) -> &[u32] {
        &self.b
    }

    /// Number of distinct elements of `b`.
    pub fn keys(&self) -> usize {
        self.kind.len()
    }

    /// What `__chain_b` made of element `id`.
    pub fn kind(&self, id: u32) -> Kind {
        self.kind[id as usize]
    }

    /// Every position of `id` in `b`, in increasing order.
    pub fn positions(&self, id: u32) -> &[u32] {
        let id = id as usize;
        &self.pos[self.starts[id] as usize..self.starts[id + 1] as usize]
    }

    /// `b2j[id]`: the positions if `id` is neither junk nor popular, else none.
    fn b2j(&self, id: u32) -> &[u32] {
        if id == NONE || self.kind[id as usize] != Kind::Normal {
            &[]
        } else {
            self.positions(id)
        }
    }

    /// How often `id` occurs in `b` (`fullbcount[elt]`).
    pub fn count(&self, id: u32) -> usize {
        let id = id as usize;
        (self.starts[id + 1] - self.starts[id]) as usize
    }

    fn is_junk(&self, j: usize) -> bool {
        self.kind[self.b[j] as usize] == Kind::Junk
    }

    /// Scratch space for [`Self::find_longest_match`] on this `b`.
    pub fn scratch(&self) -> Scratch {
        let mut s = Scratch::default();
        s.fit(self.b.len());
        s
    }

    /// `find_longest_match(alo, ahi, blo, bhi)`: `(i, j, k)` such that
    /// `a[i:i+k] == b[j:j+k]`, with CPython's tie-breaks (the earliest-ending
    /// block in `a`, then the earliest in `b`) and its extension through junk.
    /// Bounds must satisfy `alo <= ahi <= len(a)` and `blo <= bhi <= len(b)`.
    #[allow(clippy::similar_names, clippy::many_single_char_names)] // CPython's names
    pub fn find_longest_match(
        &self,
        a: &[u32],
        alo: usize,
        ahi: usize,
        blo: usize,
        bhi: usize,
        s: &mut Scratch,
    ) -> (usize, usize, usize) {
        let b = &self.b;
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        // Work since the last tick: told to the ticker in lumps, to keep the rows cheap.
        let mut work = 0usize;
        // j2len[j + 1] is the length of the match ending with a[i - 1] and
        // b[j] (CPython's dict, rebuilt for every i).
        for (i, &id) in a.iter().enumerate().take(ahi).skip(alo) {
            let positions = self.b2j(id);
            let first = positions.partition_point(|&j| (j as usize) < blo);
            // Stopped (Ctrl-C): the caller discards the result. Here, at the
            // top of a row, `j2len` holds only the entries in `touched`.
            work += 1 + positions.len() - first;
            if work >= TICK_WORK {
                if s.ticker.tick(work) {
                    break;
                }
                work = 0;
            }
            for &j in &positions[first..] {
                let j = j as usize;
                if j >= bhi {
                    break;
                }
                let k = s.j2len[j] + 1;
                s.newj2len[j + 1] = k;
                s.newtouched.push(j + 1);
                let k = k as usize;
                if k > bestsize {
                    besti = i + 1 - k;
                    bestj = j + 1 - k;
                    bestsize = k;
                }
            }
            for &t in &s.touched {
                s.j2len[t] = 0;
            }
            s.touched.clear();
            std::mem::swap(&mut s.j2len, &mut s.newj2len);
            std::mem::swap(&mut s.touched, &mut s.newtouched);
        }
        for &t in &s.touched {
            s.j2len[t] = 0;
        }
        s.touched.clear();
        s.ticker.tick(work); // the rest of this call's work (a stop shows at the next tick)

        // Extend the best by non-junk elements on each end, then by junk
        // (CPython's four loops, in order).
        while besti > alo && bestj > blo && !self.is_junk(bestj - 1) && a[besti - 1] == b[bestj - 1]
        {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && !self.is_junk(bestj + bestsize)
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        while besti > alo && bestj > blo && self.is_junk(bestj - 1) && a[besti - 1] == b[bestj - 1]
        {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi
            && bestj + bestsize < bhi
            && self.is_junk(bestj + bestsize)
            && a[besti + bestsize] == b[bestj + bestsize]
        {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    }

    /// [`Self::find_longest_match`] with reusable buffers.
    pub fn find_longest_match_with(
        &self,
        a: &[u32],
        alo: usize,
        ahi: usize,
        blo: usize,
        bhi: usize,
        w: &mut Work,
    ) -> (usize, usize, usize) {
        w.scratch.fit(self.b.len());
        self.find_longest_match(a, alo, ahi, blo, bhi, &mut w.scratch)
    }

    /// The blocks `get_matching_blocks` finds, in the order it finds them
    /// (unsorted, adjacent ones not merged), in `w.found`.
    #[allow(clippy::similar_names, clippy::many_single_char_names)] // CPython's names
    fn find_blocks(&self, a: &[u32], w: &mut Work) {
        let (la, lb) = (a.len(), self.b.len());
        w.scratch.fit(lb);
        w.found.clear();
        w.queue.clear();
        w.queue.push((0, la, 0, lb));
        while let Some((alo, ahi, blo, bhi)) = w.queue.pop() {
            let (i, j, k) = self.find_longest_match(a, alo, ahi, blo, bhi, &mut w.scratch);
            // One unit per block (find_longest_match ticks only for long rows of work).
            if w.scratch.ticker.tick(1) {
                return;
            }
            if k > 0 {
                w.found.push(Block {
                    a: i,
                    b: j,
                    size: k,
                });
                if alo < i && blo < j {
                    w.queue.push((alo, i, blo, j));
                }
                if i + k < ahi && j + k < bhi {
                    w.queue.push((i + k, ahi, j + k, bhi));
                }
            }
        }
    }

    /// `get_matching_blocks()`: the non-adjacent matching blocks in order,
    /// ending with the `(len(a), len(b), 0)` sentinel.
    pub fn matching_blocks(&self, a: &[u32]) -> Vec<Block> {
        self.matching_blocks_with(a, &mut Work::default())
    }

    /// [`Self::matching_blocks`] with reusable buffers.
    #[allow(clippy::similar_names, clippy::many_single_char_names)] // CPython's names
    pub fn matching_blocks_with(&self, a: &[u32], w: &mut Work) -> Vec<Block> {
        self.find_blocks(a, w);
        let blocks = &mut w.found;
        blocks.sort_unstable_by_key(|m| (m.a, m.b, m.size));
        let mut out = Vec::with_capacity(blocks.len() + 1);
        let (mut i1, mut j1, mut k1) = (0, 0, 0);
        for m in blocks.iter() {
            if i1 + k1 == m.a && j1 + k1 == m.b {
                k1 += m.size;
            } else {
                if k1 > 0 {
                    out.push(Block {
                        a: i1,
                        b: j1,
                        size: k1,
                    });
                }
                (i1, j1, k1) = (m.a, m.b, m.size);
            }
        }
        if k1 > 0 {
            out.push(Block {
                a: i1,
                b: j1,
                size: k1,
            });
        }
        out.push(Block {
            a: a.len(),
            b: self.b.len(),
            size: 0,
        });
        out
    }

    /// The number of matched elements of `get_matching_blocks()` (`ratio()`'s numerator).
    pub fn matched(&self, a: &[u32], w: &mut Work) -> usize {
        self.find_blocks(a, w);
        w.found.iter().map(|m| m.size).sum()
    }

    /// `quick_ratio()`'s match count: for every element, the smaller of its
    /// counts in `a` and in `b` (junk and popular elements included).
    pub fn quick_matches(&self, a: &[u32]) -> usize {
        self.quick_matches_with(a, &mut Work::default())
    }

    /// [`Self::quick_matches`] with reusable buffers.
    pub fn quick_matches_with(&self, a: &[u32], w: &mut Work) -> usize {
        let avail = &mut w.avail;
        avail.clear();
        avail.resize(self.keys(), 0);
        let mut matches = 0;
        for &id in a {
            if id == NONE {
                continue;
            }
            let used = &mut avail[id as usize];
            if (*used as usize) < self.count(id) {
                matches += 1;
            }
            *used = used.saturating_add(1);
        }
        matches
    }
}

/// Buffers reused across matches (one per thread), and the thread's
/// interrupt checks.
#[derive(Debug, Default)]
pub struct Work {
    pub(crate) scratch: Scratch,
    queue: Vec<(usize, usize, usize, usize)>,
    found: Vec<Block>,
    avail: Vec<u32>,
    /// `a` as ids, for the caller to fill.
    pub ids: Vec<u32>,
    /// Junk verdicts, for the caller to fill.
    pub junk: Vec<bool>,
}

/// Reusable buffers for [`BIndex::find_longest_match`]: all zeros between calls.
#[derive(Debug, Default)]
pub struct Scratch {
    j2len: Vec<u32>,
    newj2len: Vec<u32>,
    touched: Vec<usize>,
    newtouched: Vec<usize>,
    /// Ctrl-C checks (off unless the bindings turned them on).
    pub(crate) ticker: Ticker,
}

impl Work {
    /// Ctrl-C checks for the loops that use this `Work`.
    pub fn ticker(&mut self) -> &mut Ticker {
        &mut self.scratch.ticker
    }
}

impl Scratch {
    /// Room for a `b` of `lb` elements.
    fn fit(&mut self, lb: usize) {
        if self.j2len.len() < lb + 1 {
            self.j2len.resize(lb + 1, 0);
            self.newj2len.resize(lb + 1, 0);
        }
    }
}

/// One `difflib.Match`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Block {
    pub a: usize,
    pub b: usize,
    pub size: usize,
}

/// An opcode tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Replace,
    Delete,
    Insert,
    Equal,
}

impl Tag {
    /// The tag as CPython spells it.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Replace => "replace",
            Self::Delete => "delete",
            Self::Insert => "insert",
            Self::Equal => "equal",
        }
    }
}

/// One `get_opcodes()` entry: `(tag, i1, i2, j1, j2)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opcode {
    pub tag: Tag,
    pub i1: usize,
    pub i2: usize,
    pub j1: usize,
    pub j2: usize,
}

/// `get_opcodes()` from `get_matching_blocks()`.
pub fn opcodes(blocks: &[Block]) -> Vec<Opcode> {
    let (mut i, mut j) = (0, 0);
    let mut out = Vec::with_capacity(2 * blocks.len());
    for m in blocks {
        let tag = if i < m.a && j < m.b {
            Some(Tag::Replace)
        } else if i < m.a {
            Some(Tag::Delete)
        } else if j < m.b {
            Some(Tag::Insert)
        } else {
            None
        };
        if let Some(tag) = tag {
            out.push(Opcode {
                tag,
                i1: i,
                i2: m.a,
                j1: j,
                j2: m.b,
            });
        }
        i = m.a + m.size;
        j = m.b + m.size;
        if m.size > 0 {
            out.push(Opcode {
                tag: Tag::Equal,
                i1: m.a,
                i2: i,
                j1: m.b,
                j2: j,
            });
        }
    }
    out
}

/// The number of matched elements of `get_matching_blocks()`.
pub fn matches(blocks: &[Block]) -> usize {
    blocks.iter().map(|m| m.size).sum()
}

/// CPython's `_calculate_ratio(matches, length)`: `2.0 * matches / length`,
/// or `1.0` for two empty sequences (the same float operations).
#[allow(clippy::cast_precision_loss)] // exact below 2**53, as in CPython
pub fn calculate_ratio(matches: usize, length: usize) -> f64 {
    if length == 0 {
        1.0
    } else {
        2.0 * matches as f64 / length as f64
    }
}

/// `real_quick_ratio()` for sequences of these lengths.
pub fn real_quick_ratio(la: usize, lb: usize) -> f64 {
    calculate_ratio(la.min(lb), la + lb)
}

/// `get_grouped_opcodes(n)`'s groups. Values are `i64` because a negative
/// `n` makes CPython produce indices outside the sequences.
pub fn grouped_opcodes(codes: &[Opcode], n: i64) -> Vec<Vec<GroupedOpcode>> {
    #[allow(clippy::cast_possible_wrap)] // sequence lengths are far below 2**63
    let mut codes: Vec<GroupedOpcode> = codes
        .iter()
        .map(|c| GroupedOpcode {
            tag: c.tag,
            i1: c.i1 as i64,
            i2: c.i2 as i64,
            j1: c.j1 as i64,
            j2: c.j2 as i64,
        })
        .collect();
    if codes.is_empty() {
        codes.push(GroupedOpcode {
            tag: Tag::Equal,
            i1: 0,
            i2: 1,
            j1: 0,
            j2: 1,
        });
    }
    if codes[0].tag == Tag::Equal {
        let c = &mut codes[0];
        c.i1 = c.i1.max(c.i2 - n);
        c.j1 = c.j1.max(c.j2 - n);
    }
    let last = codes.len() - 1;
    if codes[last].tag == Tag::Equal {
        let c = &mut codes[last];
        c.i2 = c.i2.min(c.i1 + n);
        c.j2 = c.j2.min(c.j1 + n);
    }
    let nn = n + n;
    let mut groups = Vec::new();
    let mut group = Vec::new();
    for c in codes {
        let GroupedOpcode {
            tag,
            mut i1,
            i2,
            mut j1,
            j2,
        } = c;
        if tag == Tag::Equal && i2 - i1 > nn {
            group.push(GroupedOpcode {
                tag,
                i1,
                i2: i2.min(i1 + n),
                j1,
                j2: j2.min(j1 + n),
            });
            groups.push(std::mem::take(&mut group));
            i1 = i1.max(i2 - n);
            j1 = j1.max(j2 - n);
        }
        group.push(GroupedOpcode {
            tag,
            i1,
            i2,
            j1,
            j2,
        });
    }
    if !group.is_empty() && (group.len() != 1 || group[0].tag != Tag::Equal) {
        groups.push(group);
    }
    groups
}

/// One opcode of [`grouped_opcodes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupedOpcode {
    pub tag: Tag,
    pub i1: i64,
    pub i2: i64,
    pub j1: i64,
    pub j2: i64,
}

/// Python's `range(len)[start:stop]` for a list of `len` items: the indices
/// `seq[start:stop]` reads (negative values count from the end, both clamp).
pub fn py_slice(len: usize, start: i64, stop: i64) -> std::ops::Range<usize> {
    #[allow(clippy::cast_possible_wrap)]
    let n = len as i64;
    let clamp = |x: i64| -> i64 { if x < 0 { (x + n).max(0) } else { x.min(n) } };
    let (s, e) = (clamp(start), clamp(stop));
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // both are in 0..=n
    if e > s {
        s as usize..e as usize
    } else {
        s as usize..s as usize
    }
}
