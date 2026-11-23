// Part of pyroid's port of fastText (https://github.com/facebookresearch/fastText).
// fastText: Copyright (c) 2016-present, Facebook, Inc. MIT License.

//! The dictionary: words and labels, the word hash table, subword (character
//! n-gram) ids, word n-gram buckets and the pruning index of quantized models.
//!
//! Integer semantics follow fastText's C++ exactly, including its quirks:
//! - the FNV-1a hash XORs each byte *sign-extended* (`uint32_t(int8_t(c))`);
//! - word hashes are stored as `int32_t` and sign-extended to `uint64_t` when
//!   word n-grams are combined;
//! - `minn`/`maxn` are compared against `size_t` lengths (so negative values
//!   wrap), while `addSubwords` tests `maxn <= 0` as a signed int.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::Read;

use super::args::Args;
use super::error::LoadError;
use super::io::Reader;

/// End-of-sentence token. `predict` sees one at the end of every line.
pub const EOS: &[u8] = b"</s>";
const BOW: u8 = b'<';
const EOW: u8 = b'>';
/// The label prefix. It is not stored in model files: every model uses
/// fastText's default.
pub const LABEL_PREFIX: &[u8] = b"__label__";

/// The largest `maxn` and `wordNgrams` a model may declare. fastText sets no
/// limit, but `computeSubwords` takes up to `maxn` hash steps (and ids) per
/// byte of a word, `addWordNgrams` up to `wordNgrams` ids per token, and a
/// negative `maxn` (compared as a huge `size_t`) is unbounded: without a
/// cap, a corrupt header turns one long dictionary entry or input token into
/// quadratic work and gigabytes of ids. fastText's defaults and autotune stay
/// at or below 6 and 5.
const MAX_NGRAM: i32 = 32;

/// The longest run of occupied slots the word hash table (`word2int`, linear
/// probing) may have. A lookup scans the rest of the run it lands in, so a
/// dictionary crafted to collide (FNV-1a collisions are cheap to find) would
/// make loading quadratic and every token lookup slow. At fastText's load
/// factor of 0.7 the longest run of random hashes grows like
/// `ln(n) / 0.057`, about 300 for 30M entries, and a run past 1024 has a
/// probability near 1e-18 even then (lid.176.ftz's longest run is 75).
const MAX_PROBE_RUN: usize = 1024;

/// One slot of the word hash table: an entry id (`-1`: empty) and the FNV
/// hash of the entry, so a probe compares the bytes only when the hashes
/// match (equal bytes have equal hashes, so the slot found is the same).
#[derive(Clone, Copy, Debug)]
struct Slot {
    id: i32,
    hash: u32,
}

const EMPTY: Slot = Slot { id: -1, hash: 0 };

/// The longest cyclic run of occupied slots in `table`.
fn longest_run(table: &[Slot]) -> usize {
    // `ceil(size / 0.7) > size`: a non-empty table has an empty slot.
    let Some(start) = table.iter().position(|e| e.id == -1) else {
        return table.len();
    };
    let (mut best, mut run) = (0, 0);
    for k in 1..=table.len() {
        if table[(start + k) % table.len()].id == -1 {
            run = 0;
        } else {
            run += 1;
            best = best.max(run);
        }
    }
    best
}

const FNV_OFFSET: u32 = 2_166_136_261;
const FNV_PRIME: u32 = 16_777_619;

/// One FNV-1a step with fastText's signed-char quirk.
#[inline]
fn fnv_step(h: u32, byte: u8) -> u32 {
    let x = i32::from(byte as i8) as u32;
    (h ^ x).wrapping_mul(FNV_PRIME)
}

/// `Dictionary::hash`: FNV-1a over the bytes, each byte sign-extended.
#[inline]
pub fn hash(bytes: &[u8]) -> u32 {
    bytes.iter().fold(FNV_OFFSET, |h, &b| fnv_step(h, b))
}

/// fastText's token separators (`Dictionary::readWord`).
#[inline]
pub fn is_separator(b: u8) -> bool {
    matches!(b, b' ' | b'\n' | b'\r' | b'\t' | 0x0b | 0x0c | 0)
}

#[inline]
fn is_continuation(b: u8) -> bool {
    b & 0xC0 == 0x80
}

/// `entry_type` in fastText.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    Word,
    Label,
}

/// Identity-ish hasher for `i32` keys (the pruning index is looked up once
/// per n-gram, so SipHash would dominate).
#[derive(Default)]
pub struct IntHasher(u64);

impl Hasher for IntHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    fn write_i32(&mut self, i: i32) {
        let x = u64::from(i as u32);
        self.0 = x.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type PruneMap = HashMap<i32, i32, BuildHasherDefault<IntHasher>>;

/// `a % d` for a fixed `d` without a division instruction (Lemire, Kaser
/// and Kurz, "Faster Remainder by Direct Computation", 2019): exact for
/// every `u32` `a` and `d >= 1`. Each token's word-table lookup and each
/// character n-gram take one.
#[derive(Clone, Copy, Debug)]
struct FastMod {
    d: u64,
    /// `ceil(2^64 / d)` (0 for `d = 1`, which still gives 0).
    m: u64,
}

impl FastMod {
    fn new(d: u32) -> Self {
        let d = u64::from(d.max(1));
        Self {
            d,
            m: (u64::MAX / d).wrapping_add(1),
        }
    }

    #[inline]
    fn rem(self, a: u32) -> u32 {
        let low = self.m.wrapping_mul(u64::from(a));
        ((u128::from(low) * u128::from(self.d)) >> 64) as u32
    }
}

/// How many bucket ids the pruning index's bitmap covers at most (2 MiB of
/// bits). Real models have 2M buckets (256 KiB); ids past the bitmap go to
/// the map directly.
const MAX_PRUNE_BITS: usize = 1 << 24;

/// The keys of the pruning index in `0..len` as a bitmap: most n-grams of a
/// quantized model were pruned, and one bit test rejects them before a
/// hash-map probe.
#[derive(Debug, Default)]
struct PruneBits {
    words: Vec<u64>,
    /// Keys at or past `words.len() * 64` exist (they skip the bitmap).
    beyond: bool,
}

impl PruneBits {
    /// The bitmap of `map`'s keys for lookups of ids in `0..bucket`.
    fn new(map: &PruneMap, bucket: i32) -> Self {
        let max = map.keys().copied().max().unwrap_or(-1);
        let (Ok(max), Ok(bucket)) = (usize::try_from(max), usize::try_from(bucket)) else {
            return Self::default();
        };
        let len = (max + 1).min(bucket).min(MAX_PRUNE_BITS);
        let mut words = vec![0u64; len.div_ceil(64)];
        // Every key the bitmap's words cover gets its bit, so a clear bit is a sure "no".
        let covered = words.len() * 64;
        let mut beyond = false;
        for &k in map.keys() {
            match usize::try_from(k) {
                Ok(k) if k < covered => words[k / 64] |= 1 << (k % 64),
                Ok(_) => beyond = true,
                Err(_) => {}
            }
        }
        Self { words, beyond }
    }

    /// `false` only if `id` (non-negative) is certainly not a key.
    #[inline]
    fn may_contain(&self, id: i32) -> bool {
        let i = id as usize;
        match self.words.get(i / 64) {
            Some(w) => (w >> (i % 64)) & 1 != 0,
            None => self.beyond,
        }
    }
}

/// Reusable buffers for tokenizing one line.
#[derive(Debug, Default)]
pub struct LineScratch {
    /// Input-matrix rows of the line, in fastText's order.
    pub ids: Vec<i32>,
    hashes: Vec<i32>,
    word: Vec<u8>,
}

impl LineScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

#[derive(Debug)]
pub struct Dictionary {
    /// All entry strings back to back; entry `i` is `bytes[ends[i-1]..ends[i]]`.
    bytes: Vec<u8>,
    ends: Vec<usize>,
    counts: Vec<i64>,
    nwords: usize,
    nlabels: usize,
    ntokens: i64,
    pruneidx_size: i64,
    pruneidx: PruneMap,
    prune_bits: PruneBits,
    /// Open-addressing table `hash % len -> entry id` (fastText's `word2int_`).
    word2int: Vec<Slot>,
    /// `% word2int.len()` (the length is below 2^32: `ceil(i32::MAX / 0.7)`).
    slot_mod: FastMod,
    /// Subword ids of every entry (CSR): `[i, n-gram ids...]` (`initNgrams`).
    sub_ids: Vec<i32>,
    sub_ends: Vec<usize>,
    // The arguments the tokenizer needs, pre-converted to fastText's
    // comparison semantics.
    minn: u64,
    maxn: u64,
    minn_signed: i32,
    maxn_signed: i32,
    bucket: i32,
    /// `% bucket` for character n-gram hashes (`bucket` as `u32`).
    bucket_mod: FastMod,
    word_ngrams: i32,
}

impl Dictionary {
    pub(crate) fn load<R: Read>(r: &mut Reader<R>, args: &Args) -> Result<Self, LoadError> {
        let size = r.i32("dictionary size")?;
        let nwords = r.i32("dictionary nwords")?;
        let nlabels = r.i32("dictionary nlabels")?;
        let ntokens = r.i64("dictionary ntokens")?;
        let pruneidx_size = r.i64("dictionary pruneidx_size")?;
        let (Ok(size), Ok(nwords), Ok(nlabels)) = (
            usize::try_from(size),
            usize::try_from(nwords),
            usize::try_from(nlabels),
        ) else {
            return Err(LoadError::Invalid(format!(
                "negative dictionary sizes (size {size}, nwords {nwords}, nlabels {nlabels})"
            )));
        };
        if nwords.checked_add(nlabels) != Some(size) {
            return Err(LoadError::Invalid(format!(
                "dictionary size {size} is not nwords {nwords} + nlabels {nlabels}"
            )));
        }
        // Each entry takes at least 10 bytes (NUL, count, type).
        r.require("dictionary entries", (size as u64).saturating_mul(10))?;

        let mut bytes = Vec::new();
        let mut ends = Vec::with_capacity(size);
        let mut counts = Vec::with_capacity(size);
        for i in 0..size {
            r.cstring(&mut bytes, "dictionary entry")?;
            ends.push(bytes.len());
            counts.push(r.i64("dictionary entry count")?);
            let kind = match r.u8("dictionary entry type")? {
                0 => EntryKind::Word,
                1 => EntryKind::Label,
                other => {
                    return Err(LoadError::Invalid(format!(
                        "dictionary entry {i} has type {other} (expected 0 = word, 1 = label)"
                    )));
                }
            };
            // fastText sorts entries by type: words first, then labels.
            let want = if i < nwords {
                EntryKind::Word
            } else {
                EntryKind::Label
            };
            if kind != want {
                return Err(LoadError::Invalid(format!(
                    "dictionary entry {i} is a {kind:?}, but entries 0..{nwords} must be words \
                     and the rest labels"
                )));
            }
        }

        let mut pruneidx = PruneMap::default();
        if pruneidx_size > 0 {
            r.require(
                "dictionary pruning index",
                u64::try_from(pruneidx_size)
                    .unwrap_or(u64::MAX)
                    .saturating_mul(8),
            )?;
            for _ in 0..pruneidx_size {
                let first = r.i32("dictionary pruning index")?;
                let second = r.i32("dictionary pruning index")?;
                pruneidx.insert(first, second); // later pairs win, as with fastText's map
            }
        }

        let prune_bits = PruneBits::new(&pruneidx, args.bucket);
        let mut dict = Self {
            bytes,
            ends,
            counts,
            nwords,
            nlabels,
            ntokens,
            pruneidx_size,
            pruneidx,
            prune_bits,
            word2int: Vec::new(),
            slot_mod: FastMod::new(1),
            sub_ids: Vec::new(),
            sub_ends: Vec::with_capacity(size),
            minn: i64::from(args.minn) as u64,
            maxn: i64::from(args.maxn) as u64,
            minn_signed: args.minn,
            maxn_signed: args.maxn,
            bucket: args.bucket,
            bucket_mod: FastMod::new(args.bucket as u32),
            word_ngrams: args.word_ngrams,
        };
        dict.check_hashing()?;
        dict.init_ngrams();
        dict.init_word2int()?;
        Ok(dict)
    }

    /// Can tokenizing produce character n-gram or word n-gram hashes?
    pub fn hashes_ngrams(&self) -> bool {
        let subwords = self.maxn >= self.minn.max(1);
        subwords || self.word_ngrams > 1
    }

    fn check_hashing(&self) -> Result<(), LoadError> {
        if !(0..=MAX_NGRAM).contains(&self.maxn_signed) {
            return Err(LoadError::Invalid(format!(
                "maxn is {}, outside 0..={MAX_NGRAM}",
                self.maxn_signed
            )));
        }
        if self.word_ngrams > MAX_NGRAM {
            return Err(LoadError::Invalid(format!(
                "wordNgrams is {}, above {MAX_NGRAM}",
                self.word_ngrams
            )));
        }
        if self.hashes_ngrams() && self.bucket <= 0 {
            return Err(LoadError::Invalid(format!(
                "bucket is {} but the model uses n-grams (minn {}, maxn {}, wordNgrams {})",
                self.bucket, self.minn_signed, self.maxn_signed, self.word_ngrams
            )));
        }
        Ok(())
    }

    /// Largest input-matrix row this dictionary can produce, plus one.
    pub fn rows_needed(&self) -> u64 {
        let words = self.nwords as u64;
        if !self.hashes_ngrams() || self.pruneidx_size == 0 {
            return words;
        }
        if self.pruneidx_size > 0 {
            // Only ids that are in the pruning index survive.
            let max = self.pruneidx.values().copied().max().unwrap_or(-1);
            return words + u64::try_from(i64::from(max) + 1).unwrap_or(0);
        }
        let bucket = self.bucket as u64;
        words + bucket
    }

    /// Ids in the pruning index that are negative (they would index before
    /// the n-gram rows).
    pub fn has_negative_pruned_ids(&self) -> bool {
        self.pruneidx_size > 0 && self.pruneidx.values().any(|&v| v < 0)
    }

    pub fn is_pruned(&self) -> bool {
        self.pruneidx_size >= 0
    }

    pub fn nwords(&self) -> usize {
        self.nwords
    }

    pub fn nlabels(&self) -> usize {
        self.nlabels
    }

    pub fn size(&self) -> usize {
        self.ends.len()
    }

    pub fn ntokens(&self) -> i64 {
        self.ntokens
    }

    /// Entry `i`'s bytes (words `0..nwords`, then labels).
    pub fn entry(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        &self.bytes[start..self.ends[i]]
    }

    pub fn count(&self, i: usize) -> i64 {
        self.counts[i]
    }

    pub fn kind(&self, i: usize) -> EntryKind {
        if i < self.nwords {
            EntryKind::Word
        } else {
            EntryKind::Label
        }
    }

    /// Counts of the labels, in label order (`getCounts(label)`).
    pub fn label_counts(&self) -> &[i64] {
        &self.counts[self.nwords..]
    }

    /// Label `lid`'s bytes.
    pub fn label(&self, lid: usize) -> &[u8] {
        self.entry(self.nwords + lid)
    }

    fn subwords_of(&self, i: usize) -> &[i32] {
        let start = if i == 0 { 0 } else { self.sub_ends[i - 1] };
        &self.sub_ids[start..self.sub_ends[i]]
    }

    /// `pushHash`: map an n-gram bucket to its input row, through the
    /// pruning index of a quantized model.
    #[inline]
    fn push_hash(&self, out: &mut Vec<i32>, id: i32) {
        if self.pruneidx_size == 0 || id < 0 {
            return;
        }
        let id = if self.pruneidx_size > 0 {
            if !self.prune_bits.may_contain(id) {
                return;
            }
            match self.pruneidx.get(&id) {
                Some(&v) => v,
                None => return,
            }
        } else {
            id
        };
        out.push((self.nwords as i32).wrapping_add(id));
    }

    /// `computeSubwords` over `word` (which already carries `<` and `>`).
    /// With `substrings`, also records each n-gram's byte range in `word`.
    fn compute_subwords(
        &self,
        word: &[u8],
        out: &mut Vec<i32>,
        mut substrings: Option<&mut Vec<(usize, usize)>>,
    ) {
        let len = word.len();
        for i in 0..len {
            if is_continuation(word[i]) {
                continue;
            }
            // The hash of word[i..j], extended byte by byte: the same value
            // fastText gets by hashing each n-gram string from scratch.
            let mut h = FNV_OFFSET;
            let mut j = i;
            let mut n: u64 = 1;
            while j < len && n <= self.maxn {
                h = fnv_step(h, word[j]);
                j += 1;
                while j < len && is_continuation(word[j]) {
                    h = fnv_step(h, word[j]);
                    j += 1;
                }
                if n >= self.minn && !(n == 1 && (i == 0 || j == len)) {
                    // `hash % bucket` is unsigned and below bucket <= i32::MAX
                    // (bucket > 0 whenever n-grams are hashed: `check_hashing`).
                    self.push_hash(out, self.bucket_mod.rem(h) as i32);
                    if let Some(s) = substrings.as_deref_mut() {
                        s.push((i, j));
                    }
                }
                n += 1;
            }
        }
    }

    fn init_ngrams(&mut self) {
        let mut ids = Vec::new();
        let mut word = Vec::new();
        for i in 0..self.size() {
            ids.push(i as i32);
            let entry = self.entry(i);
            if entry != EOS {
                word.clear();
                word.push(BOW);
                word.extend_from_slice(entry);
                word.push(EOW);
                self.compute_subwords(&word, &mut ids, None);
            }
            self.sub_ends.push(ids.len());
        }
        self.sub_ids = ids;
    }

    fn init_word2int(&mut self) -> Result<(), LoadError> {
        // `int32_t word2intsize = std::ceil(size_ / 0.7);`
        let len = (self.size() as f64 / 0.7).ceil() as usize;
        self.word2int = vec![EMPTY; len];
        self.slot_mod = FastMod::new(len as u32);
        let clustered = || {
            LoadError::Invalid(format!(
                "the dictionary's hash table has a run of more than {MAX_PROBE_RUN} occupied \
                 slots (entries crafted to collide)"
            ))
        };
        for i in 0..self.size() {
            let h = hash(self.entry(i));
            let slot = self.find_bounded(self.entry(i), h).ok_or_else(clustered)?;
            self.word2int[slot] = Slot {
                id: i as i32,
                hash: h,
            };
        }
        // Entries on consecutive home slots form a long run without any long
        // probe while inserting; every lookup landing in it would scan it.
        if longest_run(&self.word2int) > MAX_PROBE_RUN {
            return Err(clustered());
        }
        Ok(())
    }

    /// `find` for building the table: `None` once it has passed
    /// `MAX_PROBE_RUN` occupied slots (so building stays linear).
    fn find_bounded(&self, w: &[u8], h: u32) -> Option<usize> {
        let len = self.word2int.len();
        let mut id = self.slot_mod.rem(h) as usize;
        for _ in 0..=MAX_PROBE_RUN {
            let e = self.word2int[id];
            if e.id == -1 || (e.hash == h && self.entry(e.id as usize) == w) {
                return Some(id);
            }
            id = if id + 1 == len { 0 } else { id + 1 };
        }
        None
    }

    /// `find`: the slot of `w` (or the empty slot where it would go).
    fn find(&self, w: &[u8], h: u32) -> usize {
        let len = self.word2int.len();
        // `h % len` and `(id + 1) % len`, without divisions.
        let mut id = self.slot_mod.rem(h) as usize;
        loop {
            let e = self.word2int[id];
            if e.id == -1 || (e.hash == h && self.entry(e.id as usize) == w) {
                return id;
            }
            id = if id + 1 == len { 0 } else { id + 1 };
        }
    }

    /// `getId`: entry id of `w`, or -1. `h` must be `hash(w)`.
    pub fn get_id(&self, w: &[u8], h: u32) -> i32 {
        if self.word2int.is_empty() {
            return -1;
        }
        self.word2int[self.find(w, h)].id
    }

    pub fn id(&self, w: &[u8]) -> i32 {
        self.get_id(w, hash(w))
    }

    /// `getType(token)` for a token that is not in the dictionary.
    fn kind_of_token(w: &[u8]) -> EntryKind {
        if w.starts_with(LABEL_PREFIX) {
            EntryKind::Label
        } else {
            EntryKind::Word
        }
    }

    /// The kind of a token, known or not.
    fn token_kind(&self, w: &[u8], wid: i32) -> EntryKind {
        match usize::try_from(wid) {
            Ok(i) => self.kind(i),
            Err(_) => Self::kind_of_token(w),
        }
    }

    /// `addSubwords`.
    fn add_subwords(&self, out: &mut Vec<i32>, token: &[u8], wid: i32, buf: &mut Vec<u8>) {
        match usize::try_from(wid) {
            Err(_) => {
                if token != EOS {
                    buf.clear();
                    buf.push(BOW);
                    buf.extend_from_slice(token);
                    buf.push(EOW);
                    self.compute_subwords(buf, out, None);
                }
            }
            Ok(_) if self.maxn_signed <= 0 => out.push(wid),
            Ok(i) => out.extend_from_slice(self.subwords_of(i)),
        }
    }

    /// `addWordNgrams`.
    fn add_word_ngrams(&self, out: &mut Vec<i32>, hashes: &[i32]) {
        if self.word_ngrams <= 1 {
            return;
        }
        let bucket = self.bucket as u64;
        for i in 0..hashes.len() {
            let mut h = i64::from(hashes[i]) as u64;
            let stop = (i as i32).wrapping_add(self.word_ngrams);
            let mut j = i + 1;
            while j < hashes.len() && (j as i32) < stop {
                h = h
                    .wrapping_mul(116_049_371)
                    .wrapping_add(i64::from(hashes[j]) as u64);
                self.push_hash(out, (h % bucket) as i32);
                j += 1;
            }
        }
    }

    /// `getLine` for prediction: the input-matrix rows of one line of
    /// `text`, as if `text + "\n"` were read with `readWord`. Tokens are
    /// split on fastText's separators; a `'\n'` (or a literal `</s>`
    /// token) ends the line. Label tokens are skipped. Rows go to
    /// `scratch.ids`.
    pub fn line_ids(&self, text: &[u8], scratch: &mut LineScratch) {
        let LineScratch { ids, hashes, word } = scratch;
        ids.clear();
        hashes.clear();
        Self::for_each_token(text, |token| {
            let h = hash(token);
            let wid = self.get_id(token, h);
            if self.token_kind(token, wid) == EntryKind::Word {
                self.add_subwords(ids, token, wid, word);
                hashes.push(h as i32);
            }
        });
        self.add_word_ngrams(ids, hashes);
    }

    /// Call `f` with each token of `text + "\n"` up to and including the
    /// first end-of-sentence (`readWord`/`getLine`).
    pub fn for_each_token<'t>(text: &'t [u8], mut f: impl FnMut(&'t [u8])) {
        let mut start = None;
        for (i, &b) in text.iter().enumerate() {
            if is_separator(b) {
                if let Some(s) = start.take() {
                    let token = &text[s..i];
                    f(token);
                    if token == EOS {
                        return;
                    }
                }
                if b == b'\n' {
                    f(EOS);
                    return;
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(s) = start {
            let token = &text[s..];
            f(token);
            if token == EOS {
                return;
            }
        }
        f(EOS);
    }

    /// `getLine` as the Python binding exposes it: the line's word tokens
    /// (known or not, `</s>` included) and its known labels.
    pub fn line_tokens<'t>(&self, text: &'t [u8]) -> (Vec<&'t [u8]>, Vec<&'t [u8]>) {
        let mut words = Vec::new();
        let mut labels = Vec::new();
        Self::for_each_token(text, |token| {
            let wid = self.id(token);
            match self.token_kind(token, wid) {
                EntryKind::Word => words.push(token),
                EntryKind::Label if wid >= 0 => labels.push(token),
                EntryKind::Label => {}
            }
        });
        (words, labels)
    }

    /// `getSubwords(word)`: the input rows that make up a word vector.
    pub fn word_ids(&self, word: &[u8], out: &mut Vec<i32>) {
        out.clear();
        match usize::try_from(self.id(word)) {
            Ok(i) => out.extend_from_slice(self.subwords_of(i)),
            Err(_) => {
                if word != EOS {
                    let mut buf = Vec::with_capacity(word.len() + 2);
                    buf.push(BOW);
                    buf.extend_from_slice(word);
                    buf.push(EOW);
                    self.compute_subwords(&buf, out, None);
                }
            }
        }
    }

    /// `getSubwords(word, ngrams, substrings)` (the Python `get_subwords`):
    /// the word itself if known, then each character n-gram. In a pruned
    /// model, n-grams outside the pruning index have a substring but no id,
    /// exactly as in fastText, so the two lists can differ in length.
    pub fn subwords(&self, word: &[u8]) -> (Vec<Vec<u8>>, Vec<i32>) {
        let mut ids = Vec::new();
        let mut strings = Vec::new();
        let i = self.id(word);
        if let Ok(u) = usize::try_from(i) {
            ids.push(i);
            strings.push(self.entry(u).to_vec());
        }
        if word != EOS {
            let mut buf = Vec::with_capacity(word.len() + 2);
            buf.push(BOW);
            buf.extend_from_slice(word);
            buf.push(EOW);
            let mut ranges = Vec::new();
            self.compute_subwords(&buf, &mut ids, Some(&mut ranges));
            strings.extend(ranges.into_iter().map(|(a, b)| buf[a..b].to_vec()));
        }
        (strings, ids)
    }

    /// `getSubwordId`: `nwords + hash(subword) % bucket` (no pruning), or
    /// `None` when the model has no buckets.
    pub fn subword_id(&self, subword: &[u8]) -> Option<i64> {
        let bucket = u32::try_from(self.bucket).ok().filter(|&b| b > 0)?;
        Some(self.nwords as i64 + i64::from(hash(subword) % bucket))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_sign_extends_bytes() {
        // ASCII: plain FNV-1a.
        assert_eq!(hash(b""), FNV_OFFSET);
        assert_eq!(hash(b"a"), (FNV_OFFSET ^ 0x61).wrapping_mul(FNV_PRIME));
        // 0xC3 is XORed as 0xFFFF_FFC3, not 0xC3.
        assert_eq!(
            hash(&[0xC3]),
            (FNV_OFFSET ^ 0xFFFF_FFC3).wrapping_mul(FNV_PRIME)
        );
        assert_ne!(hash(&[0xC3]), (FNV_OFFSET ^ 0xC3).wrapping_mul(FNV_PRIME));
    }

    #[test]
    fn separators() {
        for b in [b' ', b'\n', b'\r', b'\t', 0x0b, 0x0c, 0] {
            assert!(is_separator(b));
        }
        for b in [b'a', 0x1c, 0x85, 0xa0] {
            assert!(!is_separator(b));
        }
    }

    #[test]
    fn fast_mod_is_exact() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as u32
        };
        let edges = [0, 1, 2, 3, 0x7FFF_FFFF, 0x8000_0000, u32::MAX - 1, u32::MAX];
        for d in [1, 2, 3, 7, 10, 1 << 20, 2_000_000, 2_000_001, 0x7FFF_FFFF] {
            let f = FastMod::new(d);
            for &a in &edges {
                assert_eq!(f.rem(a), a % d, "{a} % {d}");
            }
            for _ in 0..100_000 {
                let a = next();
                assert_eq!(f.rem(a), a % d, "{a} % {d}");
            }
        }
        for _ in 0..10_000 {
            let d = next().max(1);
            let f = FastMod::new(d);
            for _ in 0..10 {
                let a = next();
                assert_eq!(f.rem(a), a % d, "{a} % {d}");
            }
        }
    }

    #[test]
    fn prune_bits_never_reject_a_key() {
        let mut map = PruneMap::default();
        for (k, v) in [(0, 5), (63, 1), (64, 2), (1_999_999, 3), (-4, 7)] {
            map.insert(k, v);
        }
        let bits = PruneBits::new(&map, 2_000_000);
        for k in [0, 63, 64, 1_999_999] {
            assert!(bits.may_contain(k));
        }
        for k in [1, 62, 65, 1_000_000, 1_999_998, 2_000_000] {
            assert!(!bits.may_contain(k));
        }
        // Keys past the bitmap's cap go to the map.
        map.insert(i32::MAX - 1, 9);
        let bits = PruneBits::new(&map, i32::MAX);
        assert_eq!(bits.words.len(), MAX_PRUNE_BITS / 64);
        assert!(bits.beyond);
        assert!(bits.may_contain(i32::MAX - 1));
        assert!(bits.may_contain(i32::MAX - 2));
        assert!(!bits.may_contain(65));
        // No non-negative key: nothing passes.
        let mut neg = PruneMap::default();
        neg.insert(-1, 0);
        assert!(!PruneBits::new(&neg, 10).may_contain(3));
    }
}
