// Derived from sudachi.rs 0.7.0 (Apache-2.0), © Works Applications Co., Ltd.;
// modified.
//! The lattice, its best path, the OOV providers (`MeCabOovPlugin`,
//! `SimpleOovPlugin`), the path rewriters (`JoinNumericPlugin`,
//! `JoinKatakanaOovPlugin`) and the A-unit split: everything from the
//! rewritten text to the morphemes.

use std::borrow::Cow;

use super::Dictionary;
use super::chars::cat;
use super::dic::{WordInfo, is_oov, oov_word_id};
use super::input::{Input, TokenizeError};
use super::numeric::{NumError, NumericParser};

/// `Node` (positions are modified-text character indices).
#[derive(Clone, Copy, Debug)]
struct Node {
    begin: u16,
    end: u16,
    left_id: u16,
    right_id: u16,
    cost: i16,
    word_id: u32,
}

const BOS_EOS: u32 = 0xffff_fff0;

/// `NodeIdx`: `(end, index in that end's list)`.
type NodeIdx = (u16, u16);
const NO_NODE: NodeIdx = (u16::MAX, u16::MAX);

/// `Lattice`.
#[derive(Default)]
pub(crate) struct Lattice {
    /// `VNode`s: `(right_id, total_cost)`.
    ends: Vec<Vec<(u16, i32)>>,
    ends_full: Vec<Vec<Node>>,
    indices: Vec<Vec<NodeIdx>>,
    eos: Option<NodeIdx>,
    size: usize,
}

impl Lattice {
    fn reset(&mut self, length: usize) {
        fn reset_vec<T>(v: &mut Vec<Vec<T>>, target: usize) {
            for x in v.iter_mut() {
                x.clear();
            }
            if v.len() < target {
                v.resize_with(target, || Vec::with_capacity(16));
            }
        }
        reset_vec(&mut self.ends, length + 1);
        reset_vec(&mut self.ends_full, length + 1);
        reset_vec(&mut self.indices, length + 1);
        self.eos = None;
        self.size = length + 1;
        self.ends[0].push((0, 0));
    }

    #[inline]
    fn connect_node(&self, dic: &Dictionary, r: &Node) -> (NodeIdx, i32) {
        let begin = usize::from(r.begin);
        let node_cost = i32::from(r.cost);
        let mut min_cost = i32::MAX;
        let mut prev = NO_NODE;
        for (i, &(right_id, total)) in self.ends[begin].iter().enumerate() {
            if total == i32::MAX {
                continue;
            }
            let connect = i32::from(dic.dic.connect_cost(right_id, r.left_id));
            // Wrapping, as sudachi.rs's release builds.
            let new_cost = total.wrapping_add(connect).wrapping_add(node_cost);
            if new_cost < min_cost {
                min_cost = new_cost;
                #[allow(clippy::cast_possible_truncation)] // < u16::MAX nodes per end
                {
                    prev = (r.begin, i as u16);
                }
            }
        }
        (prev, min_cost)
    }

    fn insert(&mut self, dic: &Dictionary, node: Node) {
        let (idx, cost) = self.connect_node(dic, &node);
        let end = usize::from(node.end);
        self.ends[end].push((node.right_id, cost));
        self.indices[end].push(idx);
        self.ends_full[end].push(node);
    }

    fn connect_eos(&mut self, dic: &Dictionary) -> Result<(), TokenizeError> {
        #[allow(clippy::cast_possible_truncation)] // the text is at most u16::MAX bytes
        let at = (self.size - 1) as u16;
        let node = Node {
            begin: at,
            end: at,
            left_id: 0,
            right_id: 0,
            cost: 0,
            word_id: BOS_EOS + 1,
        };
        let (idx, cost) = self.connect_node(dic, &node);
        if cost == i32::MAX {
            return Err(TokenizeError::EosBosDisconnect);
        }
        self.eos = Some(idx);
        Ok(())
    }

    /// `(node, total cost)`.
    fn node(&self, (end, i): NodeIdx) -> (&Node, i32) {
        let (end, i) = (usize::from(end), usize::from(i));
        (&self.ends_full[end][i], self.ends[end][i].1)
    }

    /// `get_minimum_node`: the first node over `begin..end` with the lowest
    /// word cost.
    fn minimum_node(&self, begin: usize, end: usize) -> Option<(&Node, i32)> {
        let nodes = self.ends_full.get(end)?;
        nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| usize::from(n.begin) == begin)
            .min_by_key(|(_, n)| n.cost)
            .map(|(i, n)| (n, self.ends[end][i].1))
    }

    fn top_path(&self, out: &mut Vec<NodeIdx>) {
        let Some(mut idx) = self.eos else {
            return;
        };
        // `fill_top_path` starts with the EOS node's predecessor.
        out.push(idx);
        loop {
            let prev = self.indices[usize::from(idx.0)][usize::from(idx.1)];
            if prev.0 == 0 {
                break;
            }
            out.push(prev);
            idx = prev;
        }
        out.reverse();
    }
}

/// Where a path node's normalized form comes from (`StringsCache`).
#[derive(Clone, Debug)]
enum Norm {
    /// The headword of a dictionary word.
    Word(u32),
    /// An OOV's own text: a byte range of the modified text.
    Text(usize, usize),
    /// A joined node's.
    Owned(String),
    /// Not computed: never read after the last path rewrite.
    Unused,
}

/// A `ResultNode` with the parts of its `WordInfo` tokenization uses.
#[derive(Clone, Debug)]
struct PathNode {
    begin: usize,
    end: usize,
    begin_b: usize,
    end_b: usize,
    word_id: u32,
    pos_id: u16,
    index_form_length: i16,
    a_split: Vec<u32>,
    norm: Norm,
}

impl PathNode {
    fn oov(begin: usize, end: usize, begin_b: usize, end_b: usize, word_id: u32) -> Self {
        // `WordInfo::new_oov(entry as u16, curr_slice.len() as i16, ...)`.
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let len = (end_b - begin_b) as i16;
        #[allow(clippy::cast_possible_truncation)]
        let pos_id = (word_id & super::dic::WORD_MASK) as u16;
        Self {
            begin,
            end,
            begin_b,
            end_b,
            word_id,
            pos_id,
            index_form_length: len,
            a_split: Vec::new(),
            norm: Norm::Text(begin_b, end_b),
        }
    }

    fn word(
        dic: &Dictionary,
        (begin, end, begin_b, end_b): (usize, usize, usize, usize),
        word_id: u32,
    ) -> Result<Self, TokenizeError> {
        let WordInfo {
            pos_id,
            index_form_length,
            normalized_form,
            a_split,
        } = dic
            .dic
            .word_info(word_id)
            .map_err(|_| TokenizeError::Internal)?;
        let norm = if normalized_form == word_id {
            Norm::Word(word_id)
        } else {
            Norm::Word(normalized_form)
        };
        Ok(Self {
            begin,
            end,
            begin_b,
            end_b,
            word_id,
            pos_id,
            index_form_length,
            a_split,
            norm,
        })
    }

    fn normalized_form<'a>(
        &'a self,
        dic: &Dictionary,
        input: &'a Input,
    ) -> Result<Cow<'a, str>, TokenizeError> {
        Ok(match &self.norm {
            Norm::Word(w) => Cow::Owned(dic.dic.headword(*w).map_err(|_| TokenizeError::Internal)?),
            Norm::Text(a, b) => Cow::Borrowed(&input.modified[*a..*b]),
            Norm::Owned(s) => Cow::Borrowed(s),
            Norm::Unused => return Err(TokenizeError::Internal),
        })
    }

    /// The normalized form's only byte, if it is one ASCII character (what
    /// `JoinNumericPlugin` compares with `","` and `"."`), without building
    /// the string.
    fn normalized_ascii(
        &self,
        dic: &Dictionary,
        input: &Input,
    ) -> Result<Option<u8>, TokenizeError> {
        let one = |s: &str| match s.as_bytes() {
            &[b] => Some(b),
            _ => None,
        };
        Ok(match &self.norm {
            Norm::Word(w) => dic
                .dic
                .headword_ascii(*w)
                .map_err(|_| TokenizeError::Internal)?,
            Norm::Text(a, b) => one(&input.modified[*a..*b]),
            Norm::Owned(s) => one(s),
            Norm::Unused => return Err(TokenizeError::Internal),
        })
    }
}

/// One morpheme: a byte range of the original text, its POS and word id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Morpheme {
    /// Start byte in the text (`Morpheme.begin()`, in bytes).
    pub begin: usize,
    /// End byte in the text.
    pub end: usize,
    /// `part_of_speech_id()`.
    pub pos_id: u16,
    /// `word_id()`: the dictionary entry, or `0xf000_0000 | pos_id` for an
    /// out-of-vocabulary word.
    pub word_id: u32,
}

/// Per-thread buffers.
#[derive(Default)]
pub(crate) struct State {
    pub input: Input,
    lattice: Lattice,
    ids: Vec<NodeIdx>,
    edits: Vec<(std::ops::Range<usize>, super::input::Repl<'static>)>,
}

impl State {
    /// `StatefulTokenizer::do_tokenize` in mode A, then the morphemes.
    pub fn tokenize(
        &mut self,
        dic: &Dictionary,
        text: &str,
        out: &mut Vec<Morpheme>,
    ) -> Result<(), TokenizeError> {
        self.input.start(text)?;
        self.rewrite_input(dic)?;
        self.input.build(&dic.chars);
        if self.input.modified.is_empty() {
            return Ok(());
        }
        self.build_lattice(dic)?;
        let mut path = self.best_path(dic)?;
        path = join_numeric(dic, &self.input, path)?;
        path = join_katakana_oov(dic, &self.input, &self.lattice, path)?;
        let input = &self.input;
        let orig = |b: usize| input.m2o[b];
        for node in split_a(dic, input, path)? {
            let (begin, end) = (orig(node.begin_b), orig(node.end_b));
            if begin > end
                || !input.original.is_char_boundary(begin)
                || !input.original.is_char_boundary(end)
            {
                return Err(TokenizeError::Internal);
            }
            out.push(Morpheme {
                begin,
                end,
                pos_id: node.pos_id,
                word_id: node.word_id,
            });
        }
        Ok(())
    }

    fn rewrite_input(&mut self, dic: &Dictionary) -> Result<(), TokenizeError> {
        // The edits borrow the dictionary's replacement strings only until
        // they are committed; `Repl<'static>` holds owned or static ones.
        let mut edits: Vec<(std::ops::Range<usize>, super::input::Repl<'_>)> = Vec::new();
        dic.default_input.edits(&self.input, &mut edits);
        self.input.commit(&mut edits)?;
        super::input::prolonged_sound_marks(&self.input, &mut self.edits);
        self.input.commit(&mut self.edits)?;
        super::input::ignore_yomigana(&self.input, &dic.chars, &mut self.edits);
        self.input.commit(&mut self.edits)?;
        Ok(())
    }

    fn build_lattice(&mut self, dic: &Dictionary) -> Result<(), TokenizeError> {
        let input = &self.input;
        let lattice = &mut self.lattice;
        let n = input.chars.len();
        lattice.reset(n);
        let bytes = input.modified.as_bytes();
        let mut entries: Vec<(u32, usize)> = Vec::new();
        for ch_off in 0..n {
            if lattice.ends[ch_off].is_empty() {
                continue;
            }
            let byte_off = input.c2b[ch_off];
            let mut created = false;
            entries.clear();
            dic.dic
                .lookup(bytes, byte_off, |w, end| entries.push((w, end)));
            for &(word_id, end) in &entries {
                if end < bytes.len() && !input.bow[end] {
                    continue;
                }
                let (left, right, cost) = dic.dic.word_param(word_id);
                let end_c = input.b2c[end];
                #[allow(clippy::cast_possible_truncation)] // at most u16::MAX bytes
                let node = Node {
                    begin: ch_off as u16,
                    end: end_c as u16,
                    left_id: u16::from_le_bytes(left.to_le_bytes()),
                    right_id: u16::from_le_bytes(right.to_le_bytes()),
                    cost,
                    word_id,
                };
                created = true;
                lattice.insert(dic, node);
            }
            // `insert_oovs`.
            if input.can_oov_bow(ch_off) {
                created |= mecab_oov(dic, input, lattice, ch_off, created);
                created |= simple_oov(dic, input, lattice, ch_off, created);
            }
            if !created {
                created = simple_oov(dic, input, lattice, ch_off, created);
            }
            if !created {
                return Err(TokenizeError::EosBosDisconnect);
            }
        }
        lattice.connect_eos(dic)
    }

    /// `resolve_best_path`.
    fn best_path(&mut self, dic: &Dictionary) -> Result<Vec<PathNode>, TokenizeError> {
        self.ids.clear();
        self.lattice.top_path(&mut self.ids);
        let mut path = Vec::with_capacity(self.ids.len());
        for &id in &self.ids {
            let (node, _) = self.lattice.node(id);
            let (b, e) = (usize::from(node.begin), usize::from(node.end));
            let (bb, eb) = (self.input.c2b[b], self.input.c2b[e]);
            path.push(if is_oov(node.word_id) {
                PathNode::oov(b, e, bb, eb, node.word_id)
            } else {
                PathNode::word(dic, (b, e, bb, eb), node.word_id)?
            });
        }
        Ok(path)
    }
}

/// `MeCabOovPlugin::provide_oov`.
fn mecab_oov(
    dic: &Dictionary,
    input: &Input,
    lattice: &mut Lattice,
    offset: usize,
    other: bool,
) -> bool {
    let char_len = input.cont[offset];
    if char_len == 0 {
        return false;
    }
    let mut created = false;
    let n = input.chars.len();
    let mut bits = input.cat[offset];
    while bits != 0 {
        let bit = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let Some(info) = dic.mecab.categories[bit] else {
            continue;
        };
        if !info.is_invoke && other {
            continue;
        }
        let oovs = &dic.mecab.oovs[bit];
        if oovs.is_empty() {
            continue;
        }
        let mut llength = char_len;
        let mut push = |len: usize, lattice: &mut Lattice| {
            for oov in oovs {
                #[allow(clippy::cast_possible_truncation)]
                lattice.insert(
                    dic,
                    Node {
                        begin: offset as u16,
                        end: (offset + len) as u16,
                        left_id: u16::from_le_bytes(oov.left_id.to_le_bytes()),
                        right_id: u16::from_le_bytes(oov.right_id.to_le_bytes()),
                        cost: oov.cost,
                        word_id: oov_word_id(oov.pos_id),
                    },
                );
                created = true;
            }
        };
        if info.is_group {
            push(char_len, lattice);
            llength -= 1;
        }
        for i in 1..=info.length as usize {
            let sublength = (offset + i).min(n) - offset;
            if sublength > llength {
                break;
            }
            push(sublength, lattice);
        }
    }
    created
}

/// `SimpleOovPlugin::provide_oov`.
fn simple_oov(
    dic: &Dictionary,
    input: &Input,
    lattice: &mut Lattice,
    offset: usize,
    other: bool,
) -> bool {
    if other {
        return false;
    }
    let length = input.word_candidate_length(offset);
    #[allow(clippy::cast_possible_truncation)]
    lattice.insert(
        dic,
        Node {
            begin: offset as u16,
            end: (offset + length) as u16,
            left_id: dic.simple.left_id,
            right_id: dic.simple.right_id,
            cost: dic.simple.cost,
            word_id: oov_word_id(dic.simple.pos_id),
        },
    );
    true
}

/// `i16 as usize` (sign-extending), as sudachi.rs reads `index_form_length`.
#[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
fn widen(x: i16) -> usize {
    x as isize as usize
}

/// `usize as i16` (truncating).
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn narrow(x: usize) -> i16 {
    x as i16
}

/// `concat_nodes`: `path[begin..end]` becomes one node with the first one's
/// POS.
fn concat_nodes(path: &mut Vec<PathNode>, begin: usize, end: usize, normalized: String) {
    let index_form_length = path[begin..end]
        .iter()
        .fold(0usize, |a, n| a.wrapping_add(widen(n.index_form_length)));
    let pos_id = path[begin].pos_id;
    let node = PathNode {
        begin: path[begin].begin,
        end: path[end - 1].end,
        begin_b: path[begin].begin_b,
        end_b: path[end - 1].end_b,
        word_id: oov_word_id(pos_id),
        pos_id,
        index_form_length: narrow(index_form_length),
        a_split: Vec::new(),
        norm: Norm::Owned(normalized),
    };
    path[begin] = node;
    path.drain(begin + 1..end);
}

/// `JoinNumericPlugin::concat` (with `enableNormalize`).
fn join_numeric_concat(
    dic: &Dictionary,
    input: &Input,
    path: &mut Vec<PathNode>,
    begin: usize,
    end: usize,
    parser: &mut NumericParser,
) -> Result<(), TokenizeError> {
    if path[begin].pos_id != dic.numeric_pos {
        return Ok(());
    }
    let normalized = parser.get_normalized();
    if end - begin > 1 || normalized != path[begin].normalized_form(dic, input)? {
        concat_nodes(path, begin, end, normalized);
    }
    Ok(())
}

/// `JoinNumericPlugin::rewrite_gen`.
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn join_numeric(
    dic: &Dictionary,
    input: &Input,
    mut path: Vec<PathNode>,
) -> Result<Vec<PathNode>, TokenizeError> {
    let mut begin_idx: i64 = -1;
    let mut comma_as_digit = true;
    let mut period_as_digit = true;
    let mut parser = NumericParser::new();
    let mut i: i64 = -1;
    while i < path.len() as i64 - 1 {
        i += 1;
        let node = &path[i as usize];
        let ctypes = input.cat_of_range(node.begin..node.end);
        // `s.len() == 1`: the normalized form is one ASCII character.
        let one = node.normalized_ascii(dic, input)?;
        if ctypes & (cat::NUMERIC | cat::KANJINUMERIC) != 0
            || (comma_as_digit && one == Some(b','))
            || (period_as_digit && one == Some(b'.'))
        {
            if begin_idx < 0 {
                parser.clear();
                begin_idx = i;
            }
            let s = node.normalized_form(dic, input)?;
            for c in s.chars() {
                if !parser.append(c) {
                    if begin_idx >= 0 {
                        // sudachi.rs restarts the run with commas (periods) no longer digits
                        // even when they already were not; that restart repeats the same
                        // state forever (a numeric word whose normalized form starts with
                        // "," does it), so here the run just ends, as on other errors. Inputs
                        // sudachi.rs finishes are unaffected.
                        if parser.error_state == NumError::Comma && comma_as_digit {
                            comma_as_digit = false;
                            i = begin_idx - 1;
                        } else if parser.error_state == NumError::Point && period_as_digit {
                            period_as_digit = false;
                            i = begin_idx - 1;
                        }
                        begin_idx = -1;
                    }
                    break;
                }
            }
            continue;
        }
        let c = one.map_or(char::MAX, char::from);
        if begin_idx >= 0 {
            if parser.done() {
                join_numeric_concat(
                    dic,
                    input,
                    &mut path,
                    begin_idx as usize,
                    i as usize,
                    &mut parser,
                )?;
                i = begin_idx + 1;
            } else {
                let ss = path[i as usize - 1].normalized_ascii(dic, input)?;
                if (parser.error_state == NumError::Comma && ss == Some(b','))
                    || (parser.error_state == NumError::Point && ss == Some(b'.'))
                {
                    join_numeric_concat(
                        dic,
                        input,
                        &mut path,
                        begin_idx as usize,
                        i as usize - 1,
                        &mut parser,
                    )?;
                    i = begin_idx + 2;
                }
            }
        }
        begin_idx = -1;
        if !comma_as_digit && c != ',' {
            comma_as_digit = true;
        }
        if !period_as_digit && c != '.' {
            period_as_digit = true;
        }
    }
    if begin_idx >= 0 {
        let len = path.len();
        if parser.done() {
            join_numeric_concat(dic, input, &mut path, begin_idx as usize, len, &mut parser)?;
        } else {
            let ss = path[len - 1].normalized_ascii(dic, input)?;
            if (parser.error_state == NumError::Comma && ss == Some(b','))
                || (parser.error_state == NumError::Point && ss == Some(b'.'))
            {
                join_numeric_concat(
                    dic,
                    input,
                    &mut path,
                    begin_idx as usize,
                    len - 1,
                    &mut parser,
                )?;
            }
        }
    }
    Ok(path)
}

/// `JoinKatakanaOovPlugin::rewrite_gen` (`minLength` 3).
fn join_katakana_oov(
    dic: &Dictionary,
    input: &Input,
    lattice: &Lattice,
    mut path: Vec<PathNode>,
) -> Result<Vec<PathNode>, TokenizeError> {
    const MIN_LENGTH: usize = 3;
    let is_katakana = |n: &PathNode| input.cat_of_range(n.begin..n.end) & cat::KATAKANA != 0;
    let can_oov_bow = |n: &PathNode| input.cat[n.begin] & cat::NOOOVBOW == 0;
    let mut i = 0;
    while i < path.len() {
        let node = &path[i];
        if !(is_oov(node.word_id) || node.end - node.begin < MIN_LENGTH) || !is_katakana(node) {
            i += 1;
            continue;
        }
        // `begin = i - 1`, walked back over katakana nodes (as i32 in sudachi.rs).
        let mut begin = i;
        while begin > 0 && is_katakana(&path[begin - 1]) {
            begin -= 1;
        }
        let mut end = i + 1;
        while end < path.len() && is_katakana(&path[end]) {
            end += 1;
        }
        while begin != end && !can_oov_bow(&path[begin]) {
            begin += 1;
        }
        if end - begin > 1 {
            concat_oov_nodes(dic, input, lattice, &mut path, begin, end)?;
            i = begin + 1;
        }
        i += 1;
    }
    Ok(path)
}

/// `concat_oov_nodes` with the katakana OOV POS.
fn concat_oov_nodes(
    dic: &Dictionary,
    input: &Input,
    lattice: &Lattice,
    path: &mut Vec<PathNode>,
    begin: usize,
    end: usize,
) -> Result<(), TokenizeError> {
    let (begin_b, end_b) = (path[begin].begin_b, path[end - 1].end_b);
    let (node_begin, node_end) = (path[begin].begin, path[end - 1].end);
    let existing = lattice
        .minimum_node(node_begin, node_end)
        .filter(|(n, _)| n.word_id != BOS_EOS && n.word_id != BOS_EOS + 1);
    if let Some((existing, _cost)) = existing {
        let (b, e) = (usize::from(existing.begin), usize::from(existing.end));
        let mut node = if is_oov(existing.word_id) {
            PathNode::oov(b, e, input.c2b[b], input.c2b[e], existing.word_id)
        } else {
            PathNode::word(dic, (b, e, 0, 0), existing.word_id)?
        };
        node.begin_b = begin_b;
        node.end_b = end_b;
        path[begin] = node;
        path.drain(begin + 1..end);
        return Ok(());
    }
    let index_form_length = path[begin..end]
        .iter()
        .fold(0usize, |a, n| a.wrapping_add(widen(n.index_form_length)));
    let pos_id = dic.katakana_pos;
    let node = PathNode {
        begin: node_begin,
        end: node_end,
        begin_b,
        end_b,
        word_id: oov_word_id(pos_id),
        pos_id,
        index_form_length: narrow(index_form_length),
        a_split: Vec::new(),
        norm: Norm::Unused,
    };
    path[begin] = node;
    path.drain(begin + 1..end);
    Ok(())
}

/// `split_path` in mode A.
fn split_a(
    dic: &Dictionary,
    input: &Input,
    path: Vec<PathNode>,
) -> Result<Vec<PathNode>, TokenizeError> {
    if path.iter().all(|n| n.a_split.len() <= 1) {
        return Ok(path);
    }
    let mut out = Vec::with_capacity(path.len() * 3 / 2);
    for node in path {
        if node.a_split.len() <= 1 {
            out.push(node);
            continue;
        }
        let (mut char_start, mut byte_start) = (node.begin, node.begin_b);
        let last = node.a_split.len() - 1;
        for (k, &word_id) in node.a_split.iter().enumerate() {
            let mut part = PathNode::word(dic, (0, 0, 0, 0), word_id)?;
            let (char_end, byte_end) = if k == last {
                (node.end, node.end_b)
            } else {
                let byte_end = byte_start.wrapping_add(widen(part.index_form_length));
                let char_end = *input.b2c.get(byte_end).ok_or(TokenizeError::Internal)?;
                (char_end, byte_end)
            };
            part.begin = char_start;
            part.end = char_end;
            part.begin_b = byte_start;
            part.end_b = byte_end;
            char_start = char_end;
            byte_start = byte_end;
            out.push(part);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::Dictionary;

    /// A dictionary the `sudachi_dic` fuzz target found (its input layout, `[n] text dic`;
    /// the dictionary lacks the bulk of its connection matrix, which the target appends as
    /// zeros): there "1" is word 45, whose normalized form is ",". sudachi.rs's
    /// `JoinNumericPlugin` fails on that comma at the start of the run, restarts the run
    /// with commas no longer digits (which they already were not), and so loops forever.
    #[test]
    fn join_numeric_ends_a_run_it_cannot_restart() {
        let input = include_bytes!("testdata/join-numeric-loop.bin");
        let (&n, rest) = input.split_first().expect("input");
        let (_, dic) = rest.split_at(usize::from(n));
        let mut file = vec![0u8; dic.len() + 72_000_000];
        file[..dic.len()].copy_from_slice(dic);
        let d = Dictionary::from_bytes(file).expect("the dictionary loads");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut out = Vec::new();
            let r = d.tokenize("東京都に住む1,234人", &mut out);
            tx.send((r, out)).expect("send");
        });
        let (r, out) = rx
            .recv_timeout(Duration::from_secs(60))
            .expect("tokenize returns");
        assert_eq!(r, Ok(()));
        let got: Vec<(usize, usize, u16, u32)> = out
            .iter()
            .map(|m| (m.begin, m.end, m.pos_id, m.word_id))
            .collect();
        let oov = |pos: u16| 0xf000_0000 | u32::from(pos);
        // "1" and "," stay word 45: no numeric run is joined there.
        assert_eq!(
            got,
            [
                (0, 3, 2, oov(2)),
                (3, 9, 2, oov(2)),
                (9, 12, 2, oov(2)),
                (12, 15, 2, oov(2)),
                (15, 18, 2, oov(2)),
                (18, 19, 11, 45),
                (19, 20, 11, 45),
                (20, 23, 1, oov(1)),
                (23, 26, 2, oov(2)),
            ]
        );
    }
}
