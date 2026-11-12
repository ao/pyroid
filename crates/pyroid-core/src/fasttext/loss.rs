// Part of pyroid's port of fastText (https://github.com/facebookresearch/fastText).
// fastText: Copyright (c) 2016-present, Facebook, Inc. MIT License.

//! The prediction heads (`loss.cc`): softmax, hierarchical softmax, and the
//! sigmoid heads of one-vs-all and negative sampling, plus fastText's top-k
//! heap.
//!
//! Each expression keeps fastText's C++ types: `std::exp` on a `float` is
//! `expf`, the unqualified `exp` in `SoftmaxLoss` is C's `double exp`, and
//! `std_log` adds `1e-5` in double precision. The heap replays libstdc++'s
//! `push_heap`/`pop_heap`/`sort_heap`, so equal scores come out in the same
//! order as the Linux reference.

use super::matrix::{Matrix, NaN};

/// `(log-probability, label)`, fastText's `Predictions` element.
pub type Scored = (f32, i32);

/// `std_log(x) = std::log(x + 1e-5)` with `real` (f32) argument and result.
#[inline]
pub fn std_log(x: f32) -> f32 {
    (f64::from(x) + 1e-5).ln() as f32
}

// ---------------------------------------------------------------- libstdc++ heap

/// fastText's `comparePairs`: `l.first > r.first`. Used as libstdc++'s
/// "less", it makes `heap[0]` the *smallest* score.
#[inline]
fn comp(l: Scored, r: Scored) -> bool {
    l.0 > r.0
}

/// `std::__push_heap`.
fn sift_up(h: &mut [Scored], mut hole: usize, top: usize, value: Scored) {
    while hole > top {
        let parent = (hole - 1) / 2;
        if !comp(h[parent], value) {
            break;
        }
        h[hole] = h[parent];
        hole = parent;
    }
    h[hole] = value;
}

/// `std::__adjust_heap` over `h[..len]`.
fn adjust_heap(h: &mut [Scored], mut hole: usize, len: usize, value: Scored) {
    let top = hole;
    let mut second = hole;
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if comp(h[second], h[second - 1]) {
            second -= 1;
        }
        h[hole] = h[second];
        hole = second;
    }
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        h[hole] = h[second - 1];
        hole = second - 1;
    }
    sift_up(h, hole, top, value);
}

/// `std::push_heap(h.begin(), h.end(), comparePairs)`.
pub fn push_heap(h: &mut [Scored]) {
    if let Some(last) = h.len().checked_sub(1) {
        let value = h[last];
        sift_up(h, last, 0, value);
    }
}

/// `std::pop_heap(h.begin(), h.end(), comparePairs)`: moves the top to the end.
pub fn pop_heap(h: &mut [Scored]) {
    let n = h.len();
    if n > 1 {
        let last = n - 1;
        let value = h[last];
        h[last] = h[0];
        adjust_heap(h, 0, last, value);
    }
}

/// `std::sort_heap(h.begin(), h.end(), comparePairs)`: best score first.
pub fn sort_heap(h: &mut [Scored]) {
    let mut last = h.len();
    while last > 1 {
        last -= 1;
        let value = h[last];
        h[last] = h[0];
        adjust_heap(h, 0, last, value);
    }
}

/// Push `(score, i)` and keep at most `k` entries (`findKBest`/`dfs` body).
#[inline]
fn offer(heap: &mut Vec<Scored>, k: usize, item: Scored) {
    heap.push(item);
    push_heap(heap);
    if heap.len() > k {
        pop_heap(heap);
        heap.pop();
    }
}

// ---------------------------------------------------------------- heads

const SIGMOID_TABLE_SIZE: usize = 512;
const MAX_SIGMOID: f32 = 8.0;

#[derive(Debug, Clone, Copy)]
pub struct Node {
    left: i32,
    right: i32,
}

#[derive(Debug)]
pub enum Head {
    Softmax {
        nlabels: usize,
    },
    /// `ova` and `ns` predict alike: a table sigmoid per label.
    Sigmoid {
        table: Box<[f32]>,
        nlabels: usize,
    },
    /// Hierarchical softmax over a Huffman tree of the label counts.
    Hierarchical {
        tree: Vec<Node>,
        osz: usize,
    },
}

impl Head {
    pub fn softmax(nlabels: usize) -> Self {
        Self::Softmax { nlabels }
    }

    /// `Loss::Loss`: `t_sigmoid_`.
    pub fn sigmoid(nlabels: usize) -> Self {
        let table = (0..=SIGMOID_TABLE_SIZE)
            .map(|i| {
                // real x = real(i * 2 * MAX_SIGMOID) / SIGMOID_TABLE_SIZE - MAX_SIGMOID;
                let x = (i * 16) as f32 / SIGMOID_TABLE_SIZE as f32 - MAX_SIGMOID;
                // 1.0 / (1.0 + std::exp(-x)): expf, then double arithmetic.
                (1.0f64 / (1.0f64 + f64::from((-x).exp()))) as f32
            })
            .collect();
        Self::Sigmoid { table, nlabels }
    }

    /// `HierarchicalSoftmaxLoss::buildTree` (only the children matter for
    /// prediction). `counts` has at least one label. Fails if the counts
    /// make a node its own descendant (counts of 1e15 or more), which would
    /// send fastText's search into endless recursion.
    pub fn hierarchical(counts: &[i64]) -> Result<Self, String> {
        let osz = counts.len();
        let n = 2 * osz - 1;
        let mut tree = vec![
            Node {
                left: -1,
                right: -1,
            };
            n
        ];
        let mut count: Vec<i64> = vec![1_000_000_000_000_000; n];
        count[..osz].copy_from_slice(counts);
        let mut leaf = osz as i64 - 1;
        let mut node = osz;
        for i in osz..n {
            let mut mini = [0usize; 2];
            for m in &mut mini {
                // `leaf >= 0 && tree_[leaf].count < tree_[node].count`
                if leaf >= 0 && count[leaf as usize] < count[node] {
                    *m = leaf as usize;
                    leaf -= 1;
                } else {
                    *m = node;
                    node += 1;
                }
                if *m >= i {
                    return Err(format!(
                        "label counts do not form a hierarchical-softmax tree (node {i})"
                    ));
                }
            }
            tree[i] = Node {
                left: mini[0] as i32,
                right: mini[1] as i32,
            };
            count[i] = count[mini[0]].wrapping_add(count[mini[1]]);
        }
        Ok(Self::Hierarchical { tree, osz })
    }

    /// `Loss::sigmoid`: the table lookup.
    #[inline]
    fn table_sigmoid(table: &[f32], x: f32) -> f32 {
        if x < -MAX_SIGMOID {
            0.0
        } else if x > MAX_SIGMOID {
            1.0
        } else {
            // int64_t((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE / MAX_SIGMOID / 2)
            let i = ((x + MAX_SIGMOID) * SIGMOID_TABLE_SIZE as f32 / MAX_SIGMOID / 2.0) as i64;
            table[i.clamp(0, SIGMOID_TABLE_SIZE as i64) as usize]
        }
    }

    /// `Loss::predict` (or the hierarchical override): the best `k` labels
    /// with score at least `threshold`, best first, into `heap`.
    /// `output` is scratch space for one score per label.
    pub fn predict(
        &self,
        wo: &Matrix,
        hidden: &[f32],
        k: usize,
        threshold: f32,
        output: &mut Vec<f32>,
        heap: &mut Vec<Scored>,
    ) -> Result<(), NaN> {
        heap.clear();
        match self {
            Self::Softmax { nlabels } => {
                output.clear();
                for i in 0..*nlabels {
                    output.push(wo.dot_row(hidden, i)?);
                }
                if let Some(&first) = output.first() {
                    let mut max = first;
                    for &o in output.iter() {
                        // std::max(output[i], max)
                        max = if o < max { max } else { o };
                    }
                    let mut z = 0.0f32;
                    for o in output.iter_mut() {
                        // Unqualified `exp` on a float is C's double exp().
                        *o = f64::from(*o - max).exp() as f32;
                        z += *o;
                    }
                    for o in output.iter_mut() {
                        *o /= z;
                    }
                }
                find_k_best(k, threshold, heap, output);
            }
            Self::Sigmoid { table, nlabels } => {
                output.clear();
                for i in 0..*nlabels {
                    output.push(Self::table_sigmoid(table, wo.dot_row(hidden, i)?));
                }
                find_k_best(k, threshold, heap, output);
            }
            Self::Hierarchical { tree, osz } => {
                dfs(tree, *osz, wo, hidden, k, threshold, heap)?;
            }
        }
        sort_heap(heap);
        Ok(())
    }
}

/// `Loss::findKBest`.
fn find_k_best(k: usize, threshold: f32, heap: &mut Vec<Scored>, output: &[f32]) {
    for (i, &o) in output.iter().enumerate() {
        if o < threshold {
            continue;
        }
        let score = std_log(o);
        if heap.len() == k && score < heap[0].0 {
            continue;
        }
        offer(heap, k, (score, i as i32));
    }
}

/// `HierarchicalSoftmaxLoss::dfs`, with an explicit stack (deep trees must
/// not overflow a worker thread's stack). Children are visited left first and
/// each node is tested on entry, as in the recursion.
fn dfs(
    tree: &[Node],
    osz: usize,
    wo: &Matrix,
    hidden: &[f32],
    k: usize,
    threshold: f32,
    heap: &mut Vec<Scored>,
) -> Result<(), NaN> {
    let log_threshold = std_log(threshold);
    let mut stack: Vec<(usize, f32)> = vec![(2 * osz - 2, 0.0)];
    while let Some((node, score)) = stack.pop() {
        if score < log_threshold {
            continue;
        }
        if heap.len() == k && score < heap[0].0 {
            continue;
        }
        let Node { left, right } = tree[node];
        if left == -1 && right == -1 {
            offer(heap, k, (score, node as i32));
            continue;
        }
        let f = wo.dot_row(hidden, node - osz)?;
        // f = 1. / (1 + std::exp(-f)): expf, float add, double division.
        let f = (1.0f64 / f64::from(1.0f32 + (-f).exp())) as f32;
        // std_log(1.0 - f): the double difference is rounded to the float parameter.
        let left_score = score + std_log((1.0f64 - f64::from(f)) as f32);
        let right_score = score + std_log(f);
        stack.push((right as usize, right_score));
        stack.push((left as usize, left_score));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference heap operations: a direct transcription of libstdc++'s
    /// algorithms on indices, checked against our port on random inputs with
    /// many ties.
    #[test]
    fn heap_keeps_k_best_in_order() {
        let mut seed = 12345u64;
        let mut next = || {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (seed >> 33) as u32
        };
        for k in 1..8 {
            let mut heap: Vec<Scored> = Vec::new();
            let mut all = Vec::new();
            for i in 0..50 {
                let s = (next() % 5) as f32; // many ties
                all.push((s, i));
                if heap.len() == k && s < heap[0].0 {
                    continue;
                }
                offer(&mut heap, k, (s, i));
            }
            sort_heap(&mut heap);
            // Scores are the k largest, in non-increasing order.
            let mut want: Vec<f32> = all.iter().map(|p| p.0).collect();
            want.sort_by(|a, b| b.total_cmp(a));
            let got: Vec<f32> = heap.iter().map(|p| p.0).collect();
            assert_eq!(got, want[..k]);
        }
    }

    #[test]
    fn std_log_adds_1e5_in_double() {
        assert_eq!(std_log(0.0), (1e-5f64).ln() as f32);
        assert_eq!(std_log(1.0), (1.0f64 + 1e-5).ln() as f32);
    }

    #[test]
    fn huffman_tree_matches_fasttext_shape() {
        // counts sorted descending, as fastText stores labels.
        let Ok(Head::Hierarchical { tree, osz }) = Head::hierarchical(&[5, 3, 2, 1]) else {
            unreachable!()
        };
        assert_eq!(osz, 4);
        // Node 4 joins the two smallest leaves (3: count 1, 2: count 2).
        assert_eq!((tree[4].left, tree[4].right), (3, 2));
        // Node 5: leaf 1 (count 3) and node 4 (count 3): leaf 1 is not < 3.
        assert_eq!((tree[5].left, tree[5].right), (4, 1));
        // Root.
        assert_eq!((tree[6].left, tree[6].right), (0, 5));
        // A count of 1e15 would make node 1 its own child.
        assert!(Head::hierarchical(&[1_000_000_000_000_000, 1]).is_err());
    }
}
