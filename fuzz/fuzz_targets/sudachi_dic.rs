//! The Sudachi system dictionary reader (`pyroid_core::sudachi`): `Dictionary::from_bytes` on
//! a fuzzed `SudachiBinaryDic` file, then `tokenize` over a short text, which reads the trie,
//! the word-id table, the word parameters and infos, the headwords and the connection matrix.
//!
//! Input: `[n] text dic...`: `text` is the next `min(n, 32)` bytes (decoded lossily), the rest
//! is the dictionary file. A dictionary that loads needs SudachiPy's default plugins' POS and a
//! connection matrix of at least 5968 x 5968 cells (71 MB), so the target also loads the file
//! with 72,000,000 zero bytes appended: the seeds end with the matrix block, holding its first
//! cells (those of left id 0, so the fuzzer can change some costs) and leaving the rest to the
//! padding. That padded load only runs when every block but the matrix lies in the input and
//! the word-id table is at most 128 bytes: a trie value may list any number of entries, each
//! a lattice node, and tokenizing is quadratic in them (see the README).
//!
//! Checked, besides "no panic" on both loads:
//! - a loaded dictionary has the comment and signature of its header, and `pos` answers for
//!   exactly the ids below `pos_count`;
//! - `tokenize` succeeds or reports invalid dictionary data or a disconnected lattice (never
//!   a too-long input, for 32 bytes); its morphemes are byte ranges of the text on char
//!   boundaries, each starting where the previous one ended; an OOV morpheme's word id is
//!   `0xf000_0000 | pos_id`, with a POS the dictionary has;
//! - deterministic: the same morphemes again, also after tokenizing another text (the
//!   lattice and input buffers are reused per thread, as in sudachi.rs).
//!
//! `PYROID_FUZZ_SEEDS=<dir>` writes the seed inputs (small dictionaries built here) to `<dir>`.

#![no_main]

use std::collections::BTreeMap;

use libfuzzer_sys::fuzz_target;
use pyroid_core::sudachi::{Dictionary, Morpheme, TokenizeError};

const MAX_TEXT: usize = 32;
const PAD: usize = 72_000_000;
const MAX_WORD_TABLE: u64 = 128;
const MAGIC: &[u8] = b"SudachiBinaryDic";
const OTHER: &str = "東京都に住む1,234人";

// ------------------------------------------------------------------ the description

struct Reader<'a> {
    d: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.d.get(self.p..self.p.checked_add(n)?)?;
        self.p += n;
        Some(s)
    }

    /// `varint64`, as the reader decodes it.
    fn varint(&mut self) -> Option<u64> {
        let mut v = 0u64;
        for i in 0..9 {
            let b = u64::from(self.take(1)?[0]);
            v |= (b & 0x7f) << (7 * i);
            if b < 0x80 {
                return Some(v);
            }
        }
        let b = u64::from(self.take(1)?[0]);
        (b < 2).then_some(v | (b << 63))
    }

    fn string(&mut self) -> Option<&'a [u8]> {
        let n = usize::try_from(self.varint()?).ok()?;
        self.take(n)
    }
}

struct Description<'a> {
    comment: &'a [u8],
    signature: &'a [u8],
    blocks: Vec<(&'a [u8], u64, u64)>,
}

/// The header and block table, read leniently (whatever the reader rejects fails to load).
fn description(d: &[u8]) -> Option<Description<'_>> {
    let mut r = Reader { d, p: 0 };
    if r.take(MAGIC.len())? != MAGIC {
        return None;
    }
    r.take(24)?;
    let comment = r.string()?;
    let signature = r.string()?;
    r.string()?;
    r.varint()?;
    r.varint()?;
    let n = r.varint()?;
    let mut blocks = Vec::new();
    for _ in 0..n {
        let name = r.string()?;
        blocks.push((name, r.varint()?, r.varint()?));
    }
    Some(Description {
        comment,
        signature,
        blocks,
    })
}

/// Whether the padded load may tokenize: only the matrix may reach into the padding, and the
/// word-id table is small.
fn bounded(desc: &Description<'_>, len: usize) -> bool {
    let len = len as u64;
    for name in ["POS", "TrieIndex", "WordPointers", "Entries", "Strings"] {
        let Some(&(_, start, size)) = desc.blocks.iter().find(|b| b.0 == name.as_bytes()) else {
            continue;
        };
        if start.checked_add(size).is_none_or(|end| end > len) {
            return false;
        }
        if name == "WordPointers" && size > MAX_WORD_TABLE {
            return false;
        }
    }
    true
}

// ------------------------------------------------------------------ checks

fn morphemes(d: &Dictionary, text: &str, ms: &[Morpheme]) {
    let mut at = None;
    for m in ms {
        assert!(m.begin <= m.end && m.end <= text.len(), "{m:?} in {text:?}");
        assert!(
            text.is_char_boundary(m.begin) && text.is_char_boundary(m.end),
            "{m:?} in {text:?}"
        );
        if let Some(prev) = at {
            assert_eq!(m.begin, prev, "morphemes are contiguous: {ms:?}");
        }
        at = Some(m.end);
        if m.word_id >> 28 == 0xf {
            assert_eq!(m.word_id & 0x0fff_ffff, u32::from(m.pos_id), "{m:?}");
            assert!(d.pos(m.pos_id).is_some(), "OOV POS {m:?}");
        }
    }
}

fn check(d: &Dictionary, desc: &Description<'_>, text: &str) {
    assert_eq!(d.comment().as_bytes(), desc.comment);
    assert_eq!(d.signature().as_bytes(), desc.signature);
    let n = d.pos_count();
    let n16 = u16::try_from(n).expect("an i16 count");
    assert!(d.pos(n16).is_none());
    if n > 0 {
        assert!(d.pos(n16 - 1).is_some());
    }

    let mut first = Vec::new();
    let r = d.tokenize(text, &mut first);
    assert!(
        !matches!(r, Err(TokenizeError::InputTooLong(..))),
        "{r:?} for {} bytes",
        text.len()
    );
    if r.is_ok() {
        morphemes(d, text, &first);
    }
    let mut again = Vec::new();
    assert_eq!(d.tokenize(text, &mut again), r);
    assert_eq!(again, first, "tokenize is deterministic");
    let mut other = Vec::new();
    let _ = d.tokenize(OTHER, &mut other);
    again.clear();
    assert_eq!(d.tokenize(text, &mut again), r);
    assert_eq!(again, first, "the per-thread state does not leak");
}

fuzz_target!(
    init: {
        if let Some(dir) = std::env::var_os("PYROID_FUZZ_SEEDS") {
            seeds::write(std::path::Path::new(&dir));
        }
    },
    |data: &[u8]| {
        let Some((&n, rest)) = data.split_first() else {
            return;
        };
        let (text, dic) = rest.split_at(usize::from(n).min(MAX_TEXT).min(rest.len()));
        let text = String::from_utf8_lossy(text);
        let _ = Dictionary::from_bytes(dic.to_vec());
        let Some(desc) = description(dic) else {
            return;
        };
        if !bounded(&desc, dic.len()) {
            return;
        }
        // Zeroed pages are mapped lazily: only what the reader touches costs anything.
        let mut file = vec![0u8; dic.len() + PAD];
        file[..dic.len()].copy_from_slice(dic);
        if let Ok(d) = Dictionary::from_bytes(file) {
            check(&d, &desc, &text);
        }
    }
);

// ------------------------------------------------------------------ seeds

/// Small system dictionaries in the layout the reader takes (`dic.rs`): header, block table,
/// POS table, double-array trie, word-id table, entries, UTF-16 headwords, then the matrix.
mod seeds {
    use super::{BTreeMap, MAGIC, MAX_TEXT};

    const UNK_DEF: &str = include_str!("../../crates/pyroid-core/src/sudachi/resources/unk.def");
    const DIM: i16 = 5969;

    pub struct Word {
        pub surface: &'static str,
        pub left: i16,
        pub right: i16,
        pub cost: i16,
        pub pos: [&'static str; 6],
        /// Indices of the words of its A-unit split.
        pub split: &'static [usize],
    }

    const fn w(
        surface: &'static str,
        left: i16,
        right: i16,
        cost: i16,
        pos: [&'static str; 6],
    ) -> Word {
        Word {
            surface,
            left,
            right,
            cost,
            pos,
            split: &[],
        }
    }

    const PLACE: [&str; 6] = ["名詞", "固有名詞", "地名", "一般", "*", "*"];
    const NOUN: [&str; 6] = ["名詞", "普通名詞", "一般", "*", "*", "*"];
    const NUM: [&str; 6] = ["名詞", "数詞", "*", "*", "*", "*"];
    const PARTICLE: [&str; 6] = ["助詞", "格助詞", "*", "*", "*", "*"];
    const VERB: [&str; 6] = ["動詞", "一般", "*", "*", "五段-マ行", "終止形-一般"];
    const COMMA: [&str; 6] = ["補助記号", "読点", "*", "*", "*", "*"];
    const POINT: [&str; 6] = ["補助記号", "句点", "*", "*", "*", "*"];
    const SUFFIX: [&str; 6] = ["接尾辞", "名詞的", "一般", "*", "*", "*"];

    fn words() -> Vec<Word> {
        vec![
            w("東京", 1, 1, 3000, PLACE),
            w("都", 2, 2, 4000, NOUN),
            Word {
                split: &[0, 1],
                ..w("東京都", 1, 2, 2000, PLACE)
            },
            w("に", 0, 3, 500, PARTICLE),
            w("住む", 0, 4, 3500, VERB),
            w("京都", 1, 1, 2500, PLACE),
            w("一", 0, 5, 2000, NUM),
            w("二", 0, 5, 2000, NUM),
            w("十", 0, 5, 2200, NUM),
            w("1", 0, 5, 1500, NUM),
            w("2", 0, 5, 1500, NUM),
            w(",", 0, 6, 1000, COMMA),
            w(".", 0, 7, 1000, POINT),
            w("カタカナ", 0, 8, 5000, NOUN),
            w("に", 0, 9, 800, SUFFIX),
            w("東", 0, 10, 4500, NOUN),
            Word {
                split: &[6, 8],
                ..w("一十", 0, 5, 1000, NUM)
            },
        ]
    }

    fn varint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    fn string(out: &mut Vec<u8>, s: &str) {
        varint(out, s.len() as u64);
        out.extend_from_slice(s.as_bytes());
    }

    fn utf16(out: &mut Vec<u8>, s: &str) {
        let units: Vec<u16> = s.encode_utf16().collect();
        out.extend_from_slice(&(units.len() as i16).to_le_bytes());
        for u in units {
            out.extend_from_slice(&u.to_le_bytes());
        }
    }

    /// The POS the default plugins look up, then `extra`.
    fn pos_table(extra: &[[&'static str; 6]]) -> Vec<[String; 6]> {
        let mut table: Vec<[String; 6]> = Vec::new();
        let mut add = |p: [&str; 6]| {
            let p = p.map(str::to_owned);
            if !table.contains(&p) {
                table.push(p);
            }
        };
        add(["補助記号", "一般", "*", "*", "*", "*"]);
        add(NUM);
        add(NOUN);
        for line in UNK_DEF.lines().filter(|l| !l.trim().is_empty()) {
            let cols: Vec<&str> = line.trim().split(',').collect();
            add(cols[4..10].try_into().expect("unk.def has 10 columns"));
        }
        for &p in extra {
            add(p);
        }
        table
    }

    /// A darts-clone double array for `keys` (byte strings, values < 2^31).
    fn trie(keys: &BTreeMap<Vec<u8>, u32>) -> Vec<u32> {
        #[derive(Default)]
        struct Node {
            kids: BTreeMap<u8, Node>,
            value: Option<u32>,
        }
        let mut root = Node::default();
        for (k, &v) in keys {
            let mut n = &mut root;
            for &b in k {
                n = n.kids.entry(b).or_default();
            }
            n.value = Some(v);
        }
        fn place(n: &Node, pos: usize, label: u32, units: &mut Vec<u32>, used: &mut Vec<bool>) {
            let free = |used: &Vec<bool>, p: usize| used.get(p).is_none_or(|u| !u);
            let mut base = 1;
            while (n.value.is_some() && !free(used, base))
                || n.kids.keys().any(|&k| !free(used, base ^ usize::from(k)))
            {
                base += 1;
            }
            let mut mark = |p: usize| {
                if used.len() <= p {
                    used.resize(p + 1, false);
                    units.resize(p + 1, 0);
                }
                used[p] = true;
            };
            if n.value.is_some() {
                mark(base);
            }
            for &k in n.kids.keys() {
                mark(base ^ usize::from(k));
            }
            let offset = pos ^ base;
            assert!(offset < 1 << 21);
            units[pos] = label | (u32::from(n.value.is_some()) << 8) | ((offset as u32) << 10);
            if let Some(v) = n.value {
                units[base] = (1 << 31) | v;
            }
            for (&k, kid) in &n.kids {
                place(kid, base ^ usize::from(k), u32::from(k), units, used);
            }
        }
        let mut units = vec![0u32];
        let mut used = vec![true];
        place(&root, 0, 0, &mut units, &mut used);
        units
    }

    /// The dictionary file: `words`, a POS table with `extra`, and the first `cells` of the
    /// connection matrix (the rest comes from the target's padding).
    pub fn dictionary(words: &[Word], extra: &[[&'static str; 6]], cells: &[i16]) -> Vec<u8> {
        let mut extra = extra.to_vec();
        extra.extend(words.iter().map(|w| w.pos));
        let table = pos_table(&extra);
        let mut pos = Vec::new();
        pos.extend_from_slice(&(table.len() as i16).to_le_bytes());
        for p in &table {
            for part in p {
                utf16(&mut pos, part);
            }
        }

        // Entries (8-byte slots: a word id is its offset / 8) and headwords.
        let mut ids = Vec::new();
        let mut at = 0usize;
        for w in words {
            ids.push(at / 8);
            at += (32 + 4 * w.split.len()).div_ceil(8) * 8;
        }
        let mut entries = Vec::new();
        let mut strings: Vec<u8> = Vec::new();
        for (i, w) in words.iter().enumerate() {
            let units = w.surface.encode_utf16().count() as u32;
            assert!(units <= 19, "a short string pointer");
            let ptr = (units << 27) | (strings.len() as u32 / 2);
            for u in w.surface.encode_utf16() {
                strings.extend_from_slice(&u.to_le_bytes());
            }
            let pos_id = table
                .iter()
                .position(|p| p.iter().zip(&w.pos).all(|(a, b)| a == b))
                .expect("in the table") as i16;
            let mut e = Vec::new();
            for x in [w.left, w.right, w.cost, pos_id] {
                e.extend_from_slice(&x.to_le_bytes());
            }
            let id = ids[i] as u32;
            for x in [ptr, ptr, id, id] {
                e.extend_from_slice(&x.to_le_bytes());
            }
            e.extend_from_slice(&(w.surface.len() as i16).to_le_bytes());
            e.extend_from_slice(&[0, 0, w.split.len() as u8, 0, 0, 0]);
            for &s in w.split {
                e.extend_from_slice(&(ids[s] as u32).to_le_bytes());
            }
            e.resize(e.len().div_ceil(8) * 8, 0);
            entries.extend(e);
        }

        // The word-id table and the trie over the surfaces.
        let mut by_key: BTreeMap<Vec<u8>, Vec<u32>> = BTreeMap::new();
        for (i, w) in words.iter().enumerate() {
            by_key
                .entry(w.surface.as_bytes().to_vec())
                .or_default()
                .push(ids[i] as u32);
        }
        let mut word_ptrs = Vec::new();
        let mut values = BTreeMap::new();
        for (k, list) in &by_key {
            values.insert(k.clone(), word_ptrs.len() as u32);
            varint(&mut word_ptrs, list.len() as u64);
            let mut prev = 0;
            for &id in list {
                varint(&mut word_ptrs, u64::from(id - prev));
                prev = id;
            }
        }
        assert!(word_ptrs.len() as u64 <= super::MAX_WORD_TABLE);
        let trie: Vec<u8> = trie(&values).iter().flat_map(|u| u.to_le_bytes()).collect();

        let mut conn = Vec::new();
        conn.extend_from_slice(&DIM.to_le_bytes());
        conn.extend_from_slice(&DIM.to_le_bytes());
        for c in cells {
            conn.extend_from_slice(&c.to_le_bytes());
        }
        let matrix = 4 + 2 * (DIM as u64) * (DIM as u64);

        let parts: [(&str, &[u8]); 5] = [
            ("POS", &pos),
            ("TrieIndex", &trie),
            ("WordPointers", &word_ptrs),
            ("Entries", &entries),
            ("Strings", &strings),
        ];
        // The header's length depends on the offsets it holds: iterate to a fixed point.
        let mut header_len = 0;
        loop {
            let mut h = Vec::new();
            h.extend_from_slice(MAGIC);
            for x in [1u64, 1_731_196_800, 0] {
                h.extend_from_slice(&x.to_le_bytes());
            }
            string(&mut h, "pyroid fuzz seed");
            string(&mut h, &format!("seed-{}-{}", words.len(), cells.len()));
            string(&mut h, "");
            varint(&mut h, words.len() as u64);
            varint(&mut h, words.len() as u64);
            varint(&mut h, 6);
            let mut off = header_len as u64;
            for (name, b) in parts {
                string(&mut h, name);
                varint(&mut h, off);
                varint(&mut h, b.len() as u64);
                off += b.len() as u64;
            }
            string(&mut h, "ConnMatrix");
            varint(&mut h, off);
            varint(&mut h, matrix);
            if h.len() == header_len {
                for (_, b) in parts {
                    h.extend_from_slice(b);
                }
                h.extend_from_slice(&conn);
                return h;
            }
            header_len = h.len();
        }
    }

    fn input(text: &str, dic: &[u8]) -> Vec<u8> {
        assert!(text.len() <= MAX_TEXT);
        let mut v = vec![text.len() as u8];
        v.extend_from_slice(text.as_bytes());
        v.extend_from_slice(dic);
        v
    }

    pub fn write(dir: &std::path::Path) {
        std::fs::create_dir_all(dir).expect("seed dir");
        let cells: Vec<i16> = (0..64).map(|i| ((i * 37) % 200 - 100) as i16).collect();
        let full = dictionary(&words(), &[], &cells);
        let none = dictionary(&[], &[], &[]);
        let few = dictionary(&words()[..6], &[], &cells[..8]);
        let mut n = 0;
        let mut put = |text: &str, dic: &[u8]| {
            std::fs::write(dir.join(format!("seed-{n:02}")), input(text, dic)).expect("write");
            n += 1;
        };
        for text in [
            "東京都に住む",
            "京都に住む",
            "一二十1,2.2",
            "カタカナとカタカナ",
            "東京ディズニーランド",
            "ｱｲｳｴｵ123abc",
            "一十二",
            "東京（とうきょう）",
        ] {
            put(text, &full);
        }
        put("テスト 123 abc", &none);
        put("東京都に", &few);
        put("", &full);
    }
}
