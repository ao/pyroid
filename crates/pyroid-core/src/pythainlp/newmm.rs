// Derived from PyThaiNLP 5.4.0 (Apache-2.0), © 2016-2026 PyThaiNLP Project,
// and datatrove 0.10.1 (Apache-2.0), © HuggingFace; modified.
//! `pythainlp.tokenize.newmm.segment` (maximal matching constrained by TCC
//! boundaries) and `tokenize._utils.rejoin_formatted_num`, on code points.
//!
//! `_onecut` is followed step by step: the priority queue of candidate
//! positions (no duplicates), `_extend_graph` with its cutoff at
//! `_MAX_GRAPH_SIZE = 50` edges, the breadth-first search for the first path
//! when one candidate is left, and `_find_skip_end` for text outside the
//! dictionary. Positions only grow while `_onecut` runs, so the queue is a
//! flag per position with a forward scan, and every position is extended at
//! most once.
//!
//! Safe mode (`safe_mode=True`) cuts a text of 140 or more characters into
//! chunks first (`_split_chunks`, `_find_cut_pos`: the last space in
//! characters 100..140, else the start of the last longest token of that
//! window) and segments each chunk on its own.

use super::tcc;
use super::trie::Trie;
use crate::filters::unicode::is_digit;
use crate::ftfy::UnicodeVersion;

const MAX_GRAPH_SIZE: usize = 50;
const TEXT_SCAN_BEGIN: usize = 100;
const TEXT_SCAN_END: usize = 140;

const NONE: u32 = u32::MAX;

/// Buffers `_onecut` reuses; positions index them.
#[derive(Debug, Default)]
pub(super) struct Work {
    /// `tcc_pos_array(text)`.
    valid: Vec<bool>,
    /// Whether a position is in `pos_list`.
    queued: Vec<bool>,
    /// `graph[p]` is `edges[lo[p]..hi[p]]`.
    lo: Vec<u32>,
    hi: Vec<u32>,
    edges: Vec<u32>,
    /// Breadth-first search: the BFS round that reached a position, and
    /// from where.
    seen: Vec<u32>,
    parent: Vec<u32>,
    queue: Vec<u32>,
    path: Vec<u32>,
    /// `_find_cut_pos`'s tokens of the scan window.
    window: Vec<u32>,
}

fn pos32(p: usize) -> u32 {
    u32::try_from(p).expect("texts under 2**32 characters")
}

/// `[ก-ฮ]`.
#[inline]
fn is_consonant(c: char) -> bool {
    ('\u{e01}'..='\u{e2e}').contains(&c)
}

/// The end of `_PAT_NONTHAI.match(s, pos)`, if it matches:
/// `[-a-zA-Z]+|\d+([,\.]\d+)*|[ \t]+|\r?\n|[^฀-๿ \t\r\n]+`,
/// alternatives in that order.
pub(super) fn nonthai_end(s: &[char], pos: usize, v: UnicodeVersion) -> Option<usize> {
    let &c = s.get(pos)?;
    let run = |from: usize, f: &dyn Fn(char) -> bool| {
        from + s[from..].iter().take_while(|&&c| f(c)).count()
    };
    if c == '-' || c.is_ascii_alphabetic() {
        return Some(run(pos, &|c| c == '-' || c.is_ascii_alphabetic()));
    }
    let digit = |c: char| is_digit(c, v);
    if digit(c) {
        let mut j = run(pos, &digit);
        while j + 1 < s.len() && matches!(s[j], ',' | '.') && digit(s[j + 1]) {
            j = run(j + 1, &digit);
        }
        return Some(j);
    }
    match c {
        ' ' | '\t' => return Some(run(pos, &|c| c == ' ' || c == '\t')),
        '\n' => return Some(pos + 1),
        '\r' => return (s.get(pos + 1) == Some(&'\n')).then_some(pos + 2),
        _ => {}
    }
    let other = |c: char| !matches!(c, '\u{e00}'..='\u{e7f}' | ' ' | '\t' | '\r' | '\n');
    other(c).then(|| run(pos, &other))
}

/// `_PAT_THAI_TWOCHARS.match(word)`: `[ก-ฮ]{,2}$`.
fn is_two_consonants(word: &[char]) -> bool {
    let k = word
        .iter()
        .take(2)
        .take_while(|&&c| is_consonant(c))
        .count();
    (0..=k).rev().any(|j| tcc::at_end(word, j))
}

/// `_find_skip_end`: the end of the out-of-dictionary word at `begin`.
fn find_skip_end(
    s: &[char],
    begin: usize,
    valid: &[bool],
    trie: &Trie,
    v: UnicodeVersion,
) -> usize {
    if let Some(e) = nonthai_end(s, begin, v) {
        return e;
    }
    for pos in begin + 1..s.len() {
        if valid[pos] {
            let word = trie
                .prefixes(s, pos)
                .any(|n| valid[pos + n] && !is_two_consonants(&s[pos..pos + n]));
            if word || nonthai_end(s, pos, v).is_some() {
                return pos;
            }
        }
    }
    s.len()
}

impl Work {
    /// `_onecut(s, trie)`: the end of each word, appended to `out` (offset by
    /// `base`).
    pub(super) fn onecut(
        &mut self,
        s: &[char],
        trie: &Trie,
        v: UnicodeVersion,
        base: usize,
        out: &mut Vec<u32>,
    ) {
        let n = s.len();
        tcc::pos_array(s, &mut self.valid);
        for buf in [&mut self.lo, &mut self.hi, &mut self.seen] {
            buf.clear();
            buf.resize(n + 1, 0);
        }
        self.queued.clear();
        self.queued.resize(n + 1, false);
        self.parent.clear();
        self.parent.resize(n + 1, NONE);
        self.edges.clear();
        let mut round = 0u32;

        // `pos_list`: `queued` from `scan` on, `queued_n` entries.
        self.queued[0] = true;
        let mut queued_n = 1usize;
        let mut scan = 0usize;
        let mut end_pos = 0usize;
        let mut graph_size = 0usize;
        loop {
            while !self.queued[scan] {
                scan += 1;
            }
            if scan >= n {
                break;
            }
            // heappop
            let begin = scan;
            self.queued[begin] = false;
            queued_n -= 1;
            scan += 1;

            // _extend_graph
            self.lo[begin] = pos32(self.edges.len());
            for len in trie.prefixes(s, begin) {
                let end = begin + len;
                if self.valid[end] {
                    self.edges.push(pos32(end));
                    graph_size += 1;
                    if !self.queued[end] {
                        self.queued[end] = true;
                        queued_n += 1;
                    }
                    if graph_size > MAX_GRAPH_SIZE {
                        break;
                    }
                }
            }
            self.hi[begin] = pos32(self.edges.len());

            if queued_n == 1 {
                let mut goal = scan;
                while !self.queued[goal] {
                    goal += 1;
                }
                round += 1;
                self.first_path(end_pos, goal, round);
                // graph.clear(): positions before `goal` are never read again.
                graph_size = 0;
                out.extend(self.path[1..].iter().map(|&p| pos32(base) + p));
                end_pos = goal;
            } else if queued_n == 0 {
                end_pos = find_skip_end(s, begin, &self.valid, trie, v);
                graph_size = 0;
                out.push(pos32(base + end_pos));
                self.queued[end_pos] = true;
                queued_n = 1;
            }
        }
    }

    /// `next(_bfs_paths_graph(graph, start, goal))`, into `self.path`.
    fn first_path(&mut self, start: usize, goal: usize, round: u32) {
        let goal = pos32(goal);
        self.queue.clear();
        self.queue.push(pos32(start));
        self.seen[start] = round;
        let mut head = 0;
        while head < self.queue.len() {
            let vertex = self.queue[head];
            head += 1;
            let v = vertex as usize;
            for i in self.lo[v] as usize..self.hi[v] as usize {
                let pos = self.edges[i];
                if pos == goal {
                    self.path.clear();
                    self.path.push(goal);
                    let mut at = vertex;
                    while at as usize != start {
                        self.path.push(at);
                        at = self.parent[at as usize];
                    }
                    self.path.push(pos32(start));
                    self.path.reverse();
                    return;
                }
                let p = pos as usize;
                if self.seen[p] != round {
                    self.seen[p] = round;
                    self.parent[p] = vertex;
                    self.queue.push(pos);
                }
            }
        }
        unreachable!("every queued position is reachable from the last cut");
    }

    /// `_find_cut_pos(t, trie)` for `len(t) >= 140`.
    fn find_cut_pos(&mut self, t: &[char], trie: &Trie, v: UnicodeVersion) -> usize {
        let sample = &t[TEXT_SCAN_BEGIN..TEXT_SCAN_END];
        if let Some(i) = sample.iter().rposition(|&c| c == ' ') {
            return i + 1 + TEXT_SCAN_BEGIN;
        }
        let mut ends = std::mem::take(&mut self.window);
        ends.clear();
        self.onecut(sample, trie, v, 0, &mut ends);
        // The last longest token (`>=`).
        let (mut max_idx, mut max_len, mut prev) = (0usize, 0usize, 0usize);
        for (i, &e) in ends.iter().enumerate() {
            let len = e as usize - prev;
            if len >= max_len {
                max_len = len;
                max_idx = i;
            }
            prev = e as usize;
        }
        let cut = TEXT_SCAN_BEGIN
            + if max_idx == 0 {
                0
            } else {
                ends[max_idx - 1] as usize
            };
        self.window = ends;
        cut
    }

    /// `newmm.segment(s, trie, safe_mode)`: the end of each word, in
    /// characters, appended to `out`.
    pub(super) fn segment(
        &mut self,
        s: &[char],
        trie: &Trie,
        safe: bool,
        v: UnicodeVersion,
        out: &mut Vec<u32>,
    ) {
        if s.is_empty() {
            return;
        }
        if !safe || s.len() < TEXT_SCAN_END {
            self.onecut(s, trie, v, 0, out);
            return;
        }
        let mut at = 0;
        while s.len() - at >= TEXT_SCAN_END {
            let cut = self.find_cut_pos(&s[at..], trie, v);
            self.onecut(&s[at..at + cut], trie, v, at, out);
            at += cut;
        }
        if at < s.len() {
            self.onecut(&s[at..], trie, v, at, out);
        }
    }
}

/// The matches of `(\d+[\.\,:])+\d+` in `s` (`finditer`), as character
/// ranges.
fn formatted_numbers(s: &[char], v: UnicodeVersion) -> impl Iterator<Item = (usize, usize)> + '_ {
    let digit = move |c: char| is_digit(c, v);
    let run = move |from: usize| from + s[from..].iter().take_while(|&&c| digit(c)).count();
    let mut next = 0;
    std::iter::from_fn(move || {
        while next < s.len() {
            if !digit(s[next]) {
                next += 1;
                continue;
            }
            let start = next;
            // Digit runs separated by one separator: `pairs` "digits
            // separator" pairs, then the end of the match.
            let mut at = run(start);
            let mut pairs = 0;
            let end = loop {
                if at < s.len() && matches!(s[at], '.' | ',' | ':') {
                    pairs += 1;
                    if at + 1 < s.len() && digit(s[at + 1]) {
                        at = run(at + 1);
                        continue;
                    }
                    // No digits after the last separator: give it back.
                    break (pairs >= 2).then_some(at);
                }
                break (pairs >= 1).then_some(at);
            };
            match end {
                Some(end) => {
                    next = end;
                    return Some((start, end));
                }
                // A later start in the same digit run fails the same way.
                None => next = run(start),
            }
        }
        None
    })
}

/// `rejoin_formatted_num(segments)`: `ends` are the segments' ends in `s`
/// (they cover it from 0); the joined segments' ends are appended to `out`.
pub(super) fn rejoin_formatted_num(
    s: &[char],
    ends: &[u32],
    v: UnicodeVersion,
    out: &mut Vec<u32>,
) {
    let mut matches = formatted_numbers(s, v);
    let mut pos = 0usize;
    let mut idx = 0usize;
    let mut m = matches.next();
    while idx < ends.len() {
        let Some((start, end)) = m else {
            break;
        };
        if pos >= start {
            let mut joined = false;
            while pos < end && idx < ends.len() {
                pos = ends[idx] as usize;
                idx += 1;
                joined = true;
            }
            if joined {
                out.push(pos32(pos));
            }
            m = matches.next();
        } else {
            pos = ends[idx] as usize;
            out.push(ends[idx]);
            idx += 1;
        }
    }
    out.extend_from_slice(&ends[idx..]);
}

#[cfg(test)]
mod tests {
    use super::*;

    const V: UnicodeVersion = UnicodeVersion::V15_1;

    fn chars(t: &str) -> Vec<char> {
        t.chars().collect()
    }

    fn nonthai(t: &str) -> Option<usize> {
        nonthai_end(&chars(t), 0, V)
    }

    #[test]
    fn nonthai_alternatives() {
        assert_eq!(nonthai("ab-c1"), Some(4));
        assert_eq!(nonthai("12,3.4.x"), Some(6));
        assert_eq!(nonthai("12,,3"), Some(2));
        assert_eq!(nonthai("๑๒.๓ก"), Some(4));
        assert_eq!(nonthai(" \t x"), Some(3));
        assert_eq!(nonthai("\r\n\n"), Some(2));
        assert_eq!(nonthai("\rx"), None);
        assert_eq!(nonthai("€_ab ก"), Some(4));
        assert_eq!(nonthai("กa"), None);
        assert_eq!(nonthai("\u{e5b}"), None);
        assert_eq!(nonthai(""), None);
    }

    #[test]
    fn two_consonants() {
        for (w, want) in [
            ("", true),
            ("ก", true),
            ("กข", true),
            ("กขค", false),
            ("กา", false),
            ("ก\n", true),
            ("กข\n", true),
            ("a", false),
        ] {
            assert_eq!(is_two_consonants(&chars(w)), want, "{w:?}");
        }
    }

    fn numbers(t: &str) -> Vec<(usize, usize)> {
        formatted_numbers(&chars(t), V).collect()
    }

    #[test]
    fn formatted_number_matches() {
        assert_eq!(numbers("1.2."), [(0, 3)]);
        assert_eq!(numbers("1.2.3"), [(0, 5)]);
        assert_eq!(numbers("a12:30b 1,234.5"), [(1, 6), (8, 15)]);
        assert_eq!(numbers("1..2 3. 4,5,"), [(8, 11)]);
        assert_eq!(numbers("12"), []);
        assert_eq!(numbers("๑.๒"), [(0, 3)]);
    }

    #[test]
    fn graph_cutoff_and_safe_mode() {
        // Recorded from PyThaiNLP 5.4.0: `newmm.segment(t, Trie(words))` and
        // with `safe_mode=True`, segment lengths. Every position is a TCC
        // boundary, so the graph passes 50 edges.
        let mut words: Vec<String> = (1..=20).map(|k| "ก".repeat(k)).collect();
        words.push("ขา".into());
        let trie = Trie::new(words.iter().map(String::as_str));
        let seg = |t: &str, safe: bool| {
            let s = chars(t);
            let (mut w, mut ends) = (Work::default(), Vec::new());
            w.segment(&s, &trie, safe, V, &mut ends);
            let mut prev = 0;
            ends.iter()
                .map(|&e| {
                    let n = e as usize - prev;
                    prev = e as usize;
                    n
                })
                .collect::<Vec<usize>>()
        };
        let k60 = "ก".repeat(60);
        assert_eq!(seg(&k60, false), [1, 20, 1, 20, 18]);
        assert_eq!(seg(&k60, true), [1, 20, 1, 20, 18]);
        let k150 = "ก".repeat(150);
        assert_eq!(
            seg(&k150, false),
            [1, 20, 1, 20, 1, 20, 1, 20, 1, 20, 1, 20, 1, 20, 3]
        );
        // No space in characters 100..140: cut at the last longest token.
        assert_eq!(
            seg(&k150, true),
            [1, 20, 1, 20, 1, 20, 1, 20, 17, 1, 20, 1, 20, 7]
        );
        let spaced = format!("{}{}{}", "ก".repeat(120), " ", "ก".repeat(40));
        let want = [1, 20, 1, 20, 1, 20, 1, 20, 1, 20, 15, 1, 1, 20, 19];
        assert_eq!(seg(&spaced, false), want);
        assert_eq!(seg(&spaced, true), want);
        let mixed = format!("{}{}", "กขากา".repeat(3), "ก".repeat(45));
        assert_eq!(seg(&mixed, false), [1, 2, 3, 2, 3, 2, 2, 1, 20, 1, 20, 3]);
        assert_eq!(seg("", false), [0usize; 0]);
    }

    fn rejoin(segs: &[&str]) -> Vec<String> {
        let text: String = segs.concat();
        let s = chars(&text);
        let mut ends = Vec::new();
        let mut at = 0;
        for seg in segs {
            at += seg.chars().count();
            ends.push(pos32(at));
        }
        let mut out = Vec::new();
        rejoin_formatted_num(&s, &ends, V, &mut out);
        let mut prev = 0;
        out.iter()
            .map(|&e| {
                let w: String = s[prev..e as usize].iter().collect();
                prev = e as usize;
                w
            })
            .collect()
    }

    #[test]
    fn rejoin_like_pythainlp() {
        // The examples in `rejoin_formatted_num`'s docstring.
        assert_eq!(
            rejoin(&[
                "ขณะ",
                "นี้",
                "เวลา",
                " ",
                "12",
                ":",
                "00น",
                " ",
                "อัตรา",
                "แลกเปลี่ยน",
                " ",
                "1",
                ",",
                "234",
                ".",
                "5",
                " ",
                "baht/zeny"
            ]),
            [
                "ขณะ",
                "นี้",
                "เวลา",
                " ",
                "12:00น",
                " ",
                "อัตรา",
                "แลกเปลี่ยน",
                " ",
                "1,234.5",
                " ",
                "baht/zeny"
            ]
        );
        assert_eq!(
            rejoin(&[
                "IP",
                " ",
                "address",
                " ",
                "ของ",
                "คุณ",
                "คือ",
                " ",
                "127",
                ".",
                "0",
                ".",
                "0",
                ".",
                "1",
                " ",
                "ครับ"
            ]),
            [
                "IP",
                " ",
                "address",
                " ",
                "ของ",
                "คุณ",
                "คือ",
                " ",
                "127.0.0.1",
                " ",
                "ครับ"
            ]
        );
        // A segment across a match start stays; the rest of the match joins.
        assert_eq!(rejoin(&["a1", ".", "2", "b"]), ["a1", ".2", "b"]);
        // A match already passed adds nothing.
        assert_eq!(rejoin(&["1.2x3.4"]), ["1.2x3.4"]);
    }
}
