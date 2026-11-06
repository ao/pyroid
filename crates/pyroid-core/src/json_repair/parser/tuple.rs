// Derived from json_repair 0.64.5 (MIT), © Stefano Baccianella; modified.
//! `parenthesized_is_explicit_tuple` in linear time for nested tuples.
//!
//! json_repair scans from every `(` it parses to the first `)` or `,` at its
//! own nesting level zero; for unclosed nested tuples (`[((((...`) every scan
//! runs to the end of the input. The scans share their work here.
//!
//! A scan from `q` keeps three floored counters (parentheses, brackets,
//! braces) that start at 0. A floored counter is the walk minus its running
//! minimum: with `G` the unfloored prefix sum, the counter after position `i`
//! is `G(i) - min(G[q..=i])`. It is 0 exactly when no position in `[q, i)`
//! holds a smaller `G` than `G(i)`, i.e. when `prev_smaller(i) < q`. So the
//! scan from `q` ends at the first `)`/`,` at `t > q` with
//! `max(prev_smaller_N(t-1), prev_smaller_S(t-1), prev_smaller_B(t-1)) < q`:
//! one pass computes that bound for every candidate `t`, and a min-segment
//! tree finds the first candidate past `q` under the bound.
//!
//! Quote state: a scan starts outside quotes. One built from `root` sees the
//! same quote state as a scan from a later `(` at `q` exactly when `q` is
//! outside quotes in its view, so such queries share its index.

use super::is_delim;

/// A tuple index built by one scan from `root` to the end of the buffer.
#[derive(Debug)]
pub(crate) struct TupleIndex {
    root: usize,
    /// Bit `i - root`: the scan is outside quotes when it reaches position `i`.
    outside: Vec<u64>,
    /// Candidate terminators (`)` or `,` outside quotes), ascending.
    cand_pos: Vec<u32>,
    /// `max(prev_smaller(t - 1)) + 1` per candidate (0: no smaller value).
    cand_lim: Vec<u32>,
    /// Min-segment tree over `cand_lim` (leaves from `size`).
    tree: Vec<u32>,
    size: usize,
}

/// The monotonic stack that gives each position its previous strictly smaller value.
struct PrevSmaller {
    stack: Vec<(i64, u32)>,
}

impl PrevSmaller {
    fn new(root: u32) -> Self {
        Self {
            stack: vec![(0, root)],
        }
    }

    /// Push `(value, pos)`; returns `prev_smaller(pos) + 1` (0 for none).
    #[inline]
    fn push(&mut self, value: i64, pos: u32) -> u32 {
        while self.stack.last().is_some_and(|&(v, _)| v >= value) {
            self.stack.pop();
        }
        let r = self.stack.last().map_or(0, |&(_, p)| p + 1);
        self.stack.push((value, pos));
        r
    }
}

impl TupleIndex {
    /// Scan `len` chars (via `at`) from the `(` at `root`.
    pub(crate) fn build(root: usize, len: usize, at: impl Fn(usize) -> char) -> Option<Self> {
        let root32 = u32::try_from(root).ok()?;
        u32::try_from(len).ok()?;
        let span = len - root;
        let mut outside = vec![0u64; span.div_ceil(64)];
        let mut cand_pos = Vec::new();
        let mut cand_lim = Vec::new();
        let mut g = [0i64; 3];
        let mut stacks = [
            PrevSmaller::new(root32),
            PrevSmaller::new(root32),
            PrevSmaller::new(root32),
        ];
        // `prev_smaller + 1` of the previous position, per counter.
        let mut prev = [0u32; 3];
        let mut in_quote: Option<char> = None;
        let mut backslashes = 0usize;
        for i in root + 1..len {
            let c = at(i);
            if in_quote.is_none() {
                outside[(i - root) / 64] |= 1 << ((i - root) % 64);
            }
            let structural = if c == '\\' {
                backslashes += 1;
                false
            } else if let Some(q) = in_quote {
                if c == q && backslashes % 2 == 0 {
                    in_quote = None;
                }
                backslashes = 0;
                false
            } else if is_delim(c) && backslashes % 2 == 0 {
                in_quote = Some(c);
                backslashes = 0;
                false
            } else {
                backslashes = 0;
                true
            };
            if structural {
                match c {
                    ')' | ',' => {
                        cand_pos.push(i as u32);
                        cand_lim.push(prev[0].max(prev[1]).max(prev[2]));
                    }
                    _ => {}
                }
                match c {
                    '(' => g[0] += 1,
                    ')' => g[0] -= 1,
                    '[' => g[1] += 1,
                    ']' => g[1] -= 1,
                    '{' => g[2] += 1,
                    '}' => g[2] -= 1,
                    _ => {}
                }
            }
            for (k, st) in stacks.iter_mut().enumerate() {
                prev[k] = st.push(g[k], i as u32);
            }
        }
        let size = cand_lim.len().next_power_of_two().max(1);
        let mut tree = vec![u32::MAX; 2 * size];
        tree[size..size + cand_lim.len()].copy_from_slice(&cand_lim);
        for k in (1..size).rev() {
            tree[k] = tree[2 * k].min(tree[2 * k + 1]);
        }
        Some(Self {
            root,
            outside,
            cand_pos,
            cand_lim,
            tree,
            size,
        })
    }

    /// Whether a scan from the `(` at `q` sees the same quote state as this one.
    pub(crate) fn covers(&self, q: usize) -> bool {
        if q == self.root {
            return true;
        }
        q > self.root
            && self
                .outside
                .get((q - self.root) / 64)
                .is_some_and(|w| w >> ((q - self.root) % 64) & 1 == 1)
    }

    /// The first terminator (`)` or `,` at level zero) of the scan from `q`.
    pub(crate) fn terminator(&self, q: usize) -> Option<usize> {
        let q32 = u32::try_from(q).ok()?;
        let k0 = self.cand_pos.partition_point(|&p| p <= q32);
        self.first_at_most(k0, q32)
            .map(|k| self.cand_pos[k] as usize)
    }

    /// The first candidate index `>= k0` whose bound is `<= limit`.
    fn first_at_most(&self, k0: usize, limit: u32) -> Option<usize> {
        if k0 >= self.cand_lim.len() {
            return None;
        }
        // Walk up from leaf k0 to find a subtree to the right holding a match.
        let mut node = self.size + k0;
        if self.tree[node] <= limit {
            return Some(k0);
        }
        loop {
            // Move to the next subtree to the right.
            while node % 2 == 1 {
                node /= 2;
                if node == 0 {
                    return None;
                }
            }
            node += 1;
            if self.tree[node] <= limit {
                break;
            }
        }
        // Descend to the leftmost leaf under the limit.
        while node < self.size {
            node *= 2;
            if self.tree[node] > limit {
                node += 1;
            }
        }
        let k = node - self.size;
        (k < self.cand_lim.len()).then_some(k)
    }
}

#[cfg(test)]
mod tests {
    use super::TupleIndex;

    /// The reference scan (the Python loop), returning the terminator position.
    fn direct(s: &[char], q: usize) -> Option<usize> {
        let (mut n, mut sq, mut br) = (0usize, 0usize, 0usize);
        let mut in_quote: Option<char> = None;
        let mut bs = 0usize;
        for (i, &c) in s.iter().enumerate().skip(q + 1) {
            if c == '\\' {
                bs += 1;
                continue;
            }
            if let Some(qc) = in_quote {
                if c == qc && bs % 2 == 0 {
                    in_quote = None;
                }
                bs = 0;
                continue;
            }
            if super::is_delim(c) && bs % 2 == 0 {
                in_quote = Some(c);
                bs = 0;
                continue;
            }
            bs = 0;
            let top = n == 0 && sq == 0 && br == 0;
            match c {
                ')' | ',' if top => return Some(i),
                '(' => n += 1,
                ')' => n = n.saturating_sub(1),
                '[' => sq += 1,
                ']' => sq = sq.saturating_sub(1),
                '{' => br += 1,
                '}' => br = br.saturating_sub(1),
                _ => {}
            }
        }
        None
    }

    #[test]
    fn matches_the_direct_scan() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let alphabet: Vec<char> = "(((()))[]{},,\"'\\ ab".chars().collect();
        for _ in 0..400 {
            let len = 1 + (seed % 60) as usize;
            let s: Vec<char> = (0..len)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    alphabet[(seed % alphabet.len() as u64) as usize]
                })
                .collect();
            for root in (0..s.len()).filter(|&i| s[i] == '(') {
                let Some(ix) = TupleIndex::build(root, s.len(), |i| s[i]) else {
                    continue;
                };
                for q in (root..s.len()).filter(|&i| s[i] == '(' && ix.covers(i)) {
                    assert_eq!(ix.terminator(q), direct(&s, q), "{s:?} root {root} q {q}");
                }
            }
        }
    }
}
