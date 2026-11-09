// Derived from sudachi.rs 0.7.0 (Apache-2.0), © Works Applications Co., Ltd.;
// modified.
//! Reader for SudachiDict's binary system dictionary, in the format
//! SudachiPy 0.7 reads (`SudachiBinaryDic`, description version 1).
//!
//! The file is read into memory once and every block is accessed in place:
//! the connection matrix, the POS table, the double-array trie, the word-id
//! table, the entries (word parameters and word infos) and the UTF-16
//! strings. Only what tokenization needs is decoded.

use std::fmt;
use std::ops::Range;

/// Why a dictionary could not be read.
#[derive(Debug)]
pub enum DicError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The data is not a dictionary pyroid can read, or is corrupt.
    Format(String),
}

impl fmt::Display for DicError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DicError::Io(e) => write!(f, "{e}"),
            DicError::Format(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for DicError {}

impl From<std::io::Error> for DicError {
    fn from(e: std::io::Error) -> Self {
        DicError::Io(e)
    }
}

fn bad(what: &str) -> DicError {
    DicError::Format(format!("invalid Sudachi dictionary: {what}"))
}

/// The v0 header magics of older dictionaries (SudachiPy 0.6 and earlier).
const V0_VERSIONS: [u64; 5] = [
    0x7366_d3f1_8bd1_11e7,
    0xce9f_011a_9239_4434,
    0xa50f_3118_8bd2_11e7,
    0x9fde_b5a9_0168_d868,
    0xca98_1175_6ff6_4fb0,
];
const MAGIC: &[u8] = b"SudachiBinaryDic";

/// The part of a word id that is the entry id.
pub(crate) const WORD_MASK: u32 = 0x0fff_ffff;

/// `WordId::oov(pos_id)`.
#[inline]
pub(crate) fn oov_word_id(pos_id: u16) -> u32 {
    (0xf << 28) | u32::from(pos_id)
}

/// `WordId::is_oov()`.
#[inline]
pub(crate) fn is_oov(word_id: u32) -> bool {
    word_id >> 28 == 0xf
}

/// `WordRef::resolve(DictId::SYSTEM)`: a system reference is its own word id;
/// a user one (never present in a system dictionary) is moved to dictionary 0.
#[inline]
fn resolve_ref(raw: u32) -> u32 {
    if raw >> 28 == 0 { raw } else { raw & WORD_MASK }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], DicError> {
        let end = self.pos.checked_add(n).ok_or_else(|| bad("truncated"))?;
        let s = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| bad("truncated"))?;
        self.pos = end;
        Ok(s)
    }

    fn u8(&mut self) -> Result<u8, DicError> {
        Ok(self.take(1)?[0])
    }

    fn u64(&mut self) -> Result<u64, DicError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes(b.try_into().expect("8 bytes")))
    }

    fn i16(&mut self) -> Result<i16, DicError> {
        let b = self.take(2)?;
        Ok(i16::from_le_bytes([b[0], b[1]]))
    }

    /// `varint64`: little-endian base-128, at most 10 bytes.
    fn varint64(&mut self) -> Result<u64, DicError> {
        let mut v = 0u64;
        for i in 0..9 {
            let b = u64::from(self.u8()?);
            v |= (b & 0x7f) << (7 * i);
            if b < 0x80 {
                return Ok(v);
            }
        }
        let b = u64::from(self.u8()?);
        if b < 0x02 {
            Ok(v | (b << 63))
        } else {
            Err(bad("varint"))
        }
    }

    fn varint32(&mut self) -> Result<u32, DicError> {
        u32::try_from(self.varint64()?).map_err(|_| bad("varint32"))
    }

    fn utf8(&mut self) -> Result<String, DicError> {
        let n = self.varint32()? as usize;
        let b = self.take(n)?;
        std::str::from_utf8(b)
            .map(str::to_owned)
            .map_err(|_| bad("utf-8 string"))
    }

    /// `utf16_string`: an i16 length in code units, then the units.
    fn utf16(&mut self) -> Result<String, DicError> {
        let n = self.i16()?;
        if n == 0 {
            return Ok(String::new());
        }
        // A negative length is a huge `usize` in sudachi.rs: never in range.
        let n = usize::try_from(n).map_err(|_| bad("utf-16 string"))?;
        let b = self.take(n * 2)?;
        decode_utf16(b).ok_or_else(|| bad("utf-16 string"))
    }
}

fn decode_utf16(b: &[u8]) -> Option<String> {
    char::decode_utf16(b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])))
        .collect::<Result<String, _>>()
        .ok()
}

#[inline]
fn u32_at(data: &[u8], i: usize) -> Option<u32> {
    let b = data.get(i..i + 4)?;
    Some(u32::from_le_bytes(b.try_into().ok()?))
}

/// What tokenization needs from one entry's word info.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WordInfo {
    pub pos_id: u16,
    /// `index_form_length`: the byte length of the word's index form.
    pub index_form_length: i16,
    /// `normalized_form`, resolved: the word whose headword is the
    /// normalized form.
    pub normalized_form: u32,
    /// The A-unit split, resolved word ids.
    pub a_split: Vec<u32>,
}

/// A SudachiDict system dictionary.
pub(crate) struct Dic {
    data: Box<[u8]>,
    pub comment: String,
    pub signature: String,
    pub creation_time: u64,
    pos: Vec<[Box<str>; 6]>,
    conn: Range<usize>,
    num_left: usize,
    num_right: usize,
    trie: Range<usize>,
    word_ptrs: Range<usize>,
    entries: Range<usize>,
    strings: Range<usize>,
}

impl fmt::Debug for Dic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Dic")
            .field("signature", &self.signature)
            .field("comment", &self.comment)
            .field("bytes", &self.data.len())
            .finish_non_exhaustive()
    }
}

impl Dic {
    /// Parse a system dictionary (`BinaryDictionary::load_system`).
    #[allow(clippy::too_many_lines)] // one pass over the description, in file order
    pub fn new(data: Box<[u8]>) -> Result<Self, DicError> {
        let mut r = Reader {
            data: &data,
            pos: 0,
        };
        let first = r.u64()?;
        if V0_VERSIONS.contains(&first) {
            return Err(DicError::Format(
                "this Sudachi dictionary is in the format of SudachiPy 0.6 or older; pyroid reads \
                 the format of SudachiPy 0.7 (sudachidict-core 20261015 or newer)"
                    .into(),
            ));
        }
        r.pos = 0;
        if r.take(MAGIC.len())? != MAGIC {
            return Err(bad("magic bytes"));
        }
        let version = r.u64()?;
        if version != 1 {
            return Err(DicError::Format(format!(
                "unsupported Sudachi dictionary description version {version}"
            )));
        }
        let creation_time = r.u64()?;
        let _flags = r.u64()?;
        let comment = r.utf8()?;
        let signature = r.utf8()?;
        let reference = r.utf8()?;
        if !reference.is_empty() {
            return Err(DicError::Format(
                "this is a Sudachi user dictionary, not a system dictionary".into(),
            ));
        }
        let _num_indexed = r.varint32()?;
        let _num_total = r.varint32()?;
        let n_blocks = r.varint32()?;
        let mut blocks: Vec<(String, Range<usize>)> = Vec::new();
        for _ in 0..n_blocks {
            let name = r.utf8()?;
            let start = r.varint64()?;
            let size = r.varint64()?;
            let start = usize::try_from(start).map_err(|_| bad("block offset"))?;
            let size = usize::try_from(size).map_err(|_| bad("block size"))?;
            let end = start.checked_add(size).ok_or_else(|| bad("block size"))?;
            blocks.push((name, start..end));
        }
        let block = |name: &str| -> Result<Range<usize>, DicError> {
            // The first block with the name, as `Description::slice`.
            let range = blocks
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, r)| r.clone())
                .ok_or_else(|| DicError::Format(format!("Dictionary part not found: {name}")))?;
            if range.end > data.len() {
                return Err(DicError::Format(format!(
                    "Dictionary part out of range: {}..{}",
                    range.start, range.end
                )));
            }
            Ok(range)
        };
        let conn_block = block("ConnMatrix")?;
        let pos_block = block("POS")?;
        let trie = block("TrieIndex")?;
        let word_ptrs = block("WordPointers")?;
        let entries = block("Entries")?;
        let strings = block("Strings")?;

        let mut r = Reader {
            data: &data[conn_block.clone()],
            pos: 0,
        };
        let num_left = usize::try_from(r.i16()?).map_err(|_| bad("connection matrix"))?;
        let num_right = usize::try_from(r.i16()?).map_err(|_| bad("connection matrix"))?;
        let conn = conn_block.start + 4..conn_block.start + 4 + num_left * num_right * 2;
        if conn.end > conn_block.end {
            return Err(bad("connection matrix"));
        }

        let mut r = Reader {
            data: &data[pos_block],
            pos: 0,
        };
        let n_pos = usize::try_from(r.i16()?).map_err(|_| bad("POS table"))?;
        let mut pos = Vec::with_capacity(n_pos);
        for _ in 0..n_pos {
            let mut p: [Box<str>; 6] = Default::default();
            for part in &mut p {
                *part = r.utf16()?.into_boxed_str();
            }
            pos.push(p);
        }
        let trie = trie.start..trie.start + (trie.len() / 4) * 4;
        Ok(Self {
            data,
            comment,
            signature,
            creation_time,
            pos,
            conn,
            num_left,
            num_right,
            trie,
            word_ptrs,
            entries,
            strings,
        })
    }

    /// The POS table.
    pub fn pos_list(&self) -> &[[Box<str>; 6]] {
        &self.pos
    }

    /// `Grammar::get_part_of_speech_id`: the first POS equal to `pos`.
    pub fn pos_id(&self, pos: &[&str]) -> Option<u16> {
        if pos.len() != 6 {
            return None;
        }
        self.pos
            .iter()
            .position(|p| p.iter().zip(pos).all(|(a, b)| &**a == *b))
            .and_then(|i| u16::try_from(i).ok())
    }

    pub fn num_left(&self) -> usize {
        self.num_left
    }

    pub fn num_right(&self) -> usize {
        self.num_right
    }

    /// `ConnectionMatrix::cost(left, right)`.
    #[inline]
    pub fn connect_cost(&self, left: u16, right: u16) -> i16 {
        let i = usize::from(right) * self.num_left + usize::from(left);
        let at = self.conn.start + 2 * i;
        match self.data.get(at..at + 2) {
            Some(b) if at + 2 <= self.conn.end => i16::from_le_bytes([b[0], b[1]]),
            // sudachi.rs reads out of bounds here (only debug-asserted); ids
            // come from the dictionary and the plugins' checked settings.
            _ => i16::MAX,
        }
    }

    #[inline]
    fn trie_unit(&self, i: usize) -> Option<u32> {
        let at = self.trie.start.checked_add(i.checked_mul(4)?)?;
        if at + 4 > self.trie.end {
            return None;
        }
        u32_at(&self.data, at)
    }

    /// Every dictionary word that starts at byte `start` of `input`, in the
    /// order of `Lexicon::lookup`: by end, then in word-id-table order.
    #[inline]
    pub fn lookup(&self, input: &[u8], start: usize, mut f: impl FnMut(u32, usize)) {
        let Some(root) = self.trie_unit(0) else {
            return;
        };
        let mut node = offset(root);
        for (i, &k) in input.iter().enumerate().skip(start) {
            let k = usize::from(k);
            node ^= k;
            let Some(unit) = self.trie_unit(node) else {
                return;
            };
            if label(unit) != k {
                return;
            }
            node ^= offset(unit);
            if has_leaf(unit) {
                let Some(v) = self.trie_unit(node) else {
                    return;
                };
                self.word_ids(v & 0x7fff_ffff, |w| f(w, i + 1));
            }
        }
    }

    /// The entry ids of a trie value (`WordIdTable::entries`).
    #[inline]
    fn word_ids(&self, index: u32, mut f: impl FnMut(u32)) {
        let table = &self.data[self.word_ptrs.clone()];
        let Some(rest) = table.get(index as usize..) else {
            return;
        };
        let mut r = Reader { data: rest, pos: 0 };
        let Ok(n) = r.varint32() else {
            return;
        };
        let mut sum = 0u32;
        for _ in 0..n {
            let Ok(d) = r.varint32() else {
                return;
            };
            sum = sum.wrapping_add(d);
            f(sum & WORD_MASK);
        }
    }

    #[inline]
    fn entry(&self, word_id: u32) -> Option<&[u8]> {
        let off = ((word_id & WORD_MASK) as usize) << 3;
        let start = self.entries.start.checked_add(off)?;
        if start >= self.entries.end {
            return None;
        }
        self.data.get(start..self.entries.end)
    }

    /// `(left_id, right_id, cost)` of a dictionary word.
    #[inline]
    pub fn word_param(&self, word_id: u32) -> (i16, i16, i16) {
        match self.entry(word_id).and_then(|e| e.get(..6)) {
            Some(b) => (
                i16::from_le_bytes([b[0], b[1]]),
                i16::from_le_bytes([b[2], b[3]]),
                i16::from_le_bytes([b[4], b[5]]),
            ),
            None => (0, 0, i16::MAX),
        }
    }

    /// The word info fields tokenization uses (`WordInfoParser` with the
    /// tokenizer's subset: every field, plus the A split).
    pub fn word_info(&self, word_id: u32) -> Result<WordInfo, DicError> {
        let e = self.entry(word_id).ok_or_else(|| bad("word id"))?;
        let fixed = e.get(..32).ok_or_else(|| bad("word info"))?;
        let pos_id = i16::from_le_bytes([fixed[6], fixed[7]]);
        let normalized_form = u32::from_le_bytes([fixed[16], fixed[17], fixed[18], fixed[19]]);
        let index_form_length = i16::from_le_bytes([fixed[24], fixed[25]]);
        let c_len = i8::from_le_bytes([fixed[26]]);
        let b_len = i8::from_le_bytes([fixed[27]]);
        let a_len = i8::from_le_bytes([fixed[28]]);
        // A negative C length is a huge `usize` in sudachi.rs.
        let c_len = usize::try_from(c_len).map_err(|_| bad("word info"))?;
        let read = |at: usize, n: usize| -> Result<Vec<u32>, DicError> {
            (0..n)
                .map(|i| {
                    u32_at(e, at + 4 * i)
                        .map(resolve_ref)
                        .ok_or_else(|| bad("word info"))
                })
                .collect()
        };
        let c = read(32, c_len)?;
        let mut at = 32 + 4 * c_len;
        let b = if b_len < 0 {
            c
        } else {
            let n = b_len.unsigned_abs() as usize;
            let v = read(at, n)?;
            at += 4 * n;
            v
        };
        let a = if a_len < 0 {
            b
        } else {
            read(at, a_len.unsigned_abs() as usize)?
        };
        Ok(WordInfo {
            pos_id: u16::from_le_bytes(pos_id.to_le_bytes()),
            index_form_length,
            normalized_form: resolve_ref(normalized_form),
            a_split: a,
        })
    }

    /// The UTF-16 code units of a dictionary word's headword.
    fn headword_units(&self, word_id: u32) -> Result<&[u8], DicError> {
        let e = self.entry(word_id).ok_or_else(|| bad("word id"))?;
        let ptr = u32_at(e, 8).ok_or_else(|| bad("word info"))?;
        let (length, offset) = decode_string_pointer(ptr);
        let strings = &self.data[self.strings.clone()];
        let start = (offset as usize) * 2;
        let end = start + (length as usize) * 2;
        strings.get(start..end).ok_or_else(|| bad("string pointer"))
    }

    /// A dictionary word's headword (`StringsCache::headword`).
    pub fn headword(&self, word_id: u32) -> Result<String, DicError> {
        decode_utf16(self.headword_units(word_id)?).ok_or_else(|| bad("utf-16 string"))
    }

    /// The headword's only byte, if it is one ASCII character.
    pub fn headword_ascii(&self, word_id: u32) -> Result<Option<u8>, DicError> {
        Ok(match self.headword_units(word_id)? {
            &[lo, 0] if lo < 0x80 => Some(lo),
            _ => None,
        })
    }
}

#[inline]
fn has_leaf(unit: u32) -> bool {
    (unit >> 8) & 1 == 1
}

#[inline]
fn label(unit: u32) -> usize {
    (unit & ((1 << 31) | 0xff)) as usize
}

#[inline]
fn offset(unit: u32) -> usize {
    ((unit >> 10) << ((unit & (1 << 9)) >> 6)) as usize
}

/// `StringPointer::decode`: `(length, offset)` in UTF-16 code units.
fn decode_string_pointer(encoded: u32) -> (u32, u32) {
    const MAX_SIMPLE_LENGTH: u32 = (1 << 5) - 1 - 12;
    let base_value = encoded >> 27;
    let additional_length_bits = base_value.saturating_sub(MAX_SIMPLE_LENGTH);
    let non_fixed_length = (encoded & 0x07ff_0000) >> (16 + 12 - additional_length_bits);
    let implicit_bit = (1u32 << 12) >> (13 - additional_length_bits);
    let length = (base_value - additional_length_bits) + (non_fixed_length | implicit_bit);
    let alignment = additional_length_bits.saturating_sub(1);
    let offset = (encoded & (0x07ff_ffff >> alignment)) << alignment;
    (length, offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_pointer_round_trips_sudachi_encodings() {
        // Encodings from `StringPointer::encode` in sudachi.rs 0.7.0.
        assert_eq!(decode_string_pointer(0), (0, 0));
        assert_eq!(decode_string_pointer((3 << 27) | 0x2a), (3, 0x2a));
        assert_eq!(decode_string_pointer((19 << 27) | 0x7), (19, 0x7));
    }

    #[test]
    fn varints() {
        let mut r = Reader {
            data: &[0x7f, 0x80, 0x01, 0xff, 0xff, 0xff, 0xff, 0x0f],
            pos: 0,
        };
        assert_eq!(r.varint64().unwrap(), 0x7f);
        assert_eq!(r.varint64().unwrap(), 0x80);
        assert_eq!(r.varint32().unwrap(), u32::MAX);
    }
}
