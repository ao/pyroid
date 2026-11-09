// Derived from CRFsuite 0.12.2 (BSD-3-Clause), © 2007-2010 Naoaki Okazaki,
// and python-crfsuite 0.9.12 (MIT), © 2014-2017 ScrapingHub Inc. and
// contributors; modified.
//! CRFsuite's first-order linear-chain CRF (`crf1d`) models: the model file
//! reader and the Viterbi tagger, with the same labels as python-crfsuite's
//! `Tagger.tag`.
//!
//! - [`Model::from_bytes`] reads a model file (`lCRF`/`FOMC`, version 100):
//!   the label and attribute dictionaries (CQDB, read through their
//!   backward links), the state features of each attribute and the
//!   transition weights.
//! - [`Tagger::tag`] scores a sequence of items and returns the best label
//!   sequence. The arithmetic is CRFsuite's, in the same order:
//!   `crf1dt_state_score` adds each feature's weight times the attribute's
//!   value, attribute by attribute in the item's order and feature by feature
//!   in the model's order, and `crf1dc_viterbi` keeps the first maximum
//!   (`max < score`), so ties resolve the same way.
//!
//! An item is a list of `(attribute, value)` pairs, as python-crfsuite
//! builds them: a string feature `s` is `(s, 1.0)`; a dict feature
//! `{key: str}` is `(key + ":" + str, 1.0)`; `{key: bool}` is `(key, 1.0 or
//! 0.0)`. CRFsuite looks attributes up as C strings, so an attribute is cut at
//! its first NUL byte ([`Model::attr_id`]), and unknown attributes are
//! dropped.

use std::fmt;

use rustc_hash::FxHashMap;

/// Why a model could not be read.
#[derive(Debug)]
pub enum ModelError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The data is not a CRFsuite model pyroid can read, or is corrupt.
    Format(String),
}

impl fmt::Display for ModelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelError::Io(e) => write!(f, "{e}"),
            ModelError::Format(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for ModelError {}

impl From<std::io::Error> for ModelError {
    fn from(e: std::io::Error) -> Self {
        ModelError::Io(e)
    }
}

fn bad(what: &str) -> ModelError {
    ModelError::Format(format!("invalid CRFsuite model: {what}"))
}

/// Little-endian reads with bounds checks.
struct Bytes<'a>(&'a [u8]);

impl Bytes<'_> {
    fn u32(&self, at: usize) -> Result<u32, ModelError> {
        let b = self
            .0
            .get(at..at.checked_add(4).ok_or_else(|| bad("offset overflow"))?)
            .ok_or_else(|| bad("truncated"))?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn off(&self, at: usize) -> Result<usize, ModelError> {
        Ok(self.u32(at)? as usize)
    }

    fn f64(&self, at: usize) -> Result<f64, ModelError> {
        let b = self
            .0
            .get(at..at.checked_add(8).ok_or_else(|| bad("offset overflow"))?)
            .ok_or_else(|| bad("truncated"))?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(f64::from_bits(u64::from_le_bytes(a)))
    }

    fn tag(&self, at: usize, tag: [u8; 4]) -> Result<(), ModelError> {
        if self.0.get(at..at + 4) == Some(&tag[..]) {
            Ok(())
        } else {
            Err(bad(&format!(
                "missing {} chunk",
                String::from_utf8_lossy(&tag)
            )))
        }
    }
}

/// `CQDB`'s magic and byte-order check.
const CQDB: &[u8; 4] = b"CQDB";
const CQDB_BYTEORDER: u32 = 0x6244_5371;
/// `sizeof(header_t)` + 256 table references of a CQDB.
const CQDB_HEADER: usize = 24;
const CQDB_TABLES: usize = 256;

/// The strings of a CQDB, by id (`cqdb_to_string` for every id).
fn cqdb_strings(data: &[u8], start: usize) -> Result<Vec<Option<Vec<u8>>>, ModelError> {
    let db = data.get(start..).ok_or_else(|| bad("CQDB offset"))?;
    let r = Bytes(db);
    r.tag(0, *CQDB)?;
    if r.u32(12)? != CQDB_BYTEORDER {
        return Err(bad("CQDB byte order"));
    }
    let size = r.off(4)?;
    if size > db.len() {
        return Err(bad("CQDB size"));
    }
    let bwd_size = r.off(16)?;
    let bwd_offset = r.off(20)?;
    // The number of records: half the size of every hash table.
    let mut num = 0usize;
    for i in 0..CQDB_TABLES {
        num += r.off(CQDB_HEADER + 8 * i + 4)? / 2;
    }
    if bwd_offset == 0 {
        return Err(bad("CQDB without backward links"));
    }
    let n = bwd_size.min(num);
    let mut out = Vec::with_capacity(n);
    for id in 0..n {
        let rec = r.off(bwd_offset + 4 * id)?;
        if rec == 0 {
            out.push(None);
            continue;
        }
        let ksize = r.off(rec + 4)?;
        let key = db
            .get(rec + 8..rec + 8 + ksize)
            .ok_or_else(|| bad("CQDB key"))?;
        // `ksize` counts the terminating NUL; the C string ends at the first.
        let end = key.iter().position(|&b| b == 0).unwrap_or(key.len());
        out.push(Some(key[..end].to_vec()));
    }
    Ok(out)
}

/// A CRFsuite `crf1d` model.
#[derive(Clone)]
pub struct Model {
    labels: Vec<Box<str>>,
    attrs: FxHashMap<Box<[u8]>, u32>,
    /// Attribute id -> its range of `state`.
    attr_refs: Vec<(u32, u32)>,
    /// `(label, weight)` of the state features, attribute by attribute.
    state: Vec<(u32, f64)>,
    /// `trans[i * L + j]`: the weight of the transition from label `i` to `j`.
    trans: Vec<f64>,
}

impl fmt::Debug for Model {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Model")
            .field("labels", &self.labels)
            .field("attributes", &self.attrs.len())
            .field("state_features", &self.state.len())
            .finish_non_exhaustive()
    }
}

const HEADER_SIZE: usize = 48;
const CHUNK_SIZE: usize = 12;
const FEATURE_SIZE: usize = 20;

impl Model {
    /// Read a model file.
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Self, ModelError> {
        Self::from_bytes(&std::fs::read(path)?)
    }

    /// A model from the bytes of its file.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ModelError> {
        let r = Bytes(data);
        if data.len() <= HEADER_SIZE || &data[..4] != b"lCRF" {
            return Err(bad("not a CRFsuite model file"));
        }
        if &data[8..12] != b"FOMC" {
            return Err(bad("not a first-order CRF (FOMC) model"));
        }
        if r.u32(12)? != 100 {
            return Err(bad("unsupported version"));
        }
        let num_labels = r.off(20)?;
        let num_attrs = r.off(24)?;
        let off_features = r.off(28)?;
        let off_labels = r.off(32)?;
        let off_attrs = r.off(36)?;
        let off_labelrefs = r.off(40)?;
        let off_attrrefs = r.off(44)?;
        if num_labels == 0 {
            return Err(bad("no labels"));
        }

        let labels = cqdb_strings(data, off_labels)?
            .into_iter()
            .take(num_labels)
            .map(|s| {
                let s = s.ok_or_else(|| bad("missing label"))?;
                String::from_utf8(s)
                    .map(String::into_boxed_str)
                    .map_err(|_| bad("label is not UTF-8"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if labels.len() != num_labels {
            return Err(bad("label count"));
        }
        let mut attrs = FxHashMap::default();
        for (id, s) in cqdb_strings(data, off_attrs)?.into_iter().enumerate() {
            let Some(s) = s else { continue };
            if id < num_attrs {
                // `cqdb_to_id` finds the first record of a key; a valid model
                // has each key once.
                let id = u32::try_from(id).map_err(|_| bad("attribute id"))?;
                attrs.entry(s.into_boxed_slice()).or_insert(id);
            }
        }

        r.tag(off_features, *b"FEAT")?;
        let num_features = r.off(off_features + 8)?;
        let feature = |fid: usize| -> Result<(u32, f64), ModelError> {
            if fid >= num_features {
                return Err(bad("feature id"));
            }
            let at = off_features + CHUNK_SIZE + FEATURE_SIZE * fid;
            let dst = r.u32(at + 8)?;
            if dst as usize >= num_labels {
                return Err(bad("feature label"));
            }
            Ok((dst, r.f64(at + 12)?))
        };
        // The feature ids a reference chunk lists for item `i`.
        let refs = |off: usize, tag: &[u8; 4], i: usize| -> Result<Vec<usize>, ModelError> {
            r.tag(off, *tag)?;
            let at = r.off(off + CHUNK_SIZE + 4 * i)?;
            let n = r.off(at)?;
            (0..n).map(|k| r.off(at + 4 + 4 * k)).collect()
        };

        let mut trans = vec![0.0; num_labels * num_labels];
        for i in 0..num_labels {
            for fid in refs(off_labelrefs, b"LFRF", i)? {
                let (dst, w) = feature(fid)?;
                trans[i * num_labels + dst as usize] = w;
            }
        }
        let mut attr_refs = Vec::with_capacity(num_attrs);
        let mut state = Vec::new();
        for a in 0..num_attrs {
            let start = u32::try_from(state.len()).map_err(|_| bad("too many features"))?;
            for fid in refs(off_attrrefs, b"AFRF", a)? {
                state.push(feature(fid)?);
            }
            let end = u32::try_from(state.len()).map_err(|_| bad("too many features"))?;
            attr_refs.push((start, end));
        }
        Ok(Self {
            labels,
            attrs,
            attr_refs,
            state,
            trans,
        })
    }

    /// The number of labels.
    pub fn num_labels(&self) -> usize {
        self.labels.len()
    }

    /// The labels, by id.
    pub fn labels(&self) -> &[Box<str>] {
        &self.labels
    }

    /// The id of `label`.
    pub fn label_id(&self, label: &str) -> Option<usize> {
        self.labels.iter().position(|l| &**l == label)
    }

    /// The number of attributes.
    pub fn num_attrs(&self) -> usize {
        self.attr_refs.len()
    }

    /// Every attribute with its id, in no particular order.
    pub fn attributes(&self) -> impl Iterator<Item = (&[u8], u32)> {
        self.attrs.iter().map(|(k, &v)| (&**k, v))
    }

    /// The id of `attr` as CRFsuite looks it up: as a C string, so only the
    /// bytes before the first NUL count.
    #[inline]
    pub fn attr_id(&self, attr: &[u8]) -> Option<u32> {
        let key = match memchr::memchr(0, attr) {
            Some(i) => &attr[..i],
            None => attr,
        };
        self.attrs.get(key).copied()
    }

    /// The state features `(label, weight)` of an attribute.
    #[inline]
    fn features(&self, aid: u32) -> &[(u32, f64)] {
        let (s, e) = self.attr_refs[aid as usize];
        &self.state[s as usize..e as usize]
    }
}

/// Scores and decodes sequences for one [`Model`]; reuse it to keep its
/// buffers.
#[derive(Debug)]
pub struct Tagger<'m> {
    model: &'m Model,
    /// `STATE_SCORE(t, l)`, item by item.
    state: Vec<f64>,
    /// `ALPHA_SCORE`.
    alpha: Vec<f64>,
    /// `BACKWARD_EDGE_AT`.
    back: Vec<u32>,
    items: usize,
}

impl<'m> Tagger<'m> {
    /// A tagger for `model`.
    pub fn new(model: &'m Model) -> Self {
        Self {
            model,
            state: Vec::new(),
            alpha: Vec::new(),
            back: Vec::new(),
            items: 0,
        }
    }

    /// The model.
    pub fn model(&self) -> &'m Model {
        self.model
    }

    /// Start a sequence (`Tagger::set`).
    pub fn clear(&mut self) {
        self.state.clear();
        self.items = 0;
    }

    /// Append an item with no attributes yet; [`Self::add`] adds them.
    pub fn push_item(&mut self) {
        let l = self.model.num_labels();
        self.state.resize(self.state.len() + l, 0.0);
        self.items += 1;
    }

    /// Add the attribute `aid` with `value` to the last item
    /// (`crf1dt_state_score`'s inner loop).
    #[inline]
    pub fn add(&mut self, aid: u32, value: f64) {
        let l = self.model.num_labels();
        let base = self.state.len() - l;
        for &(dst, w) in self.model.features(aid) {
            self.state[base + dst as usize] += w * value;
        }
    }

    /// Add the attribute `attr` (looked up as [`Model::attr_id`] does; an
    /// unknown one is dropped) with value 1.0 to the last item.
    #[inline]
    pub fn add_str(&mut self, attr: &[u8]) {
        if let Some(aid) = self.model.attr_id(attr) {
            self.add_one(aid);
        }
    }

    /// [`Self::add`] with value 1.0 (`w * 1.0` is `w`).
    #[inline]
    pub fn add_one(&mut self, aid: u32) {
        let l = self.model.num_labels();
        let base = self.state.len() - l;
        for &(dst, w) in self.model.features(aid) {
            self.state[base + dst as usize] += w;
        }
    }

    /// The number of items.
    pub fn len(&self) -> usize {
        self.items
    }

    /// Whether there are no items.
    pub fn is_empty(&self) -> bool {
        self.items == 0
    }

    /// The best label sequence (`crf1dc_viterbi`), one label id per item,
    /// written to `out` (cleared first). Empty for an empty sequence.
    pub fn viterbi(&mut self, out: &mut Vec<u32>) {
        out.clear();
        let t_len = self.items;
        if t_len == 0 {
            return;
        }
        let l = self.model.num_labels();
        let trans = &self.model.trans;
        self.alpha.clear();
        self.alpha.extend_from_slice(&self.state[..l]);
        self.alpha.resize(t_len * l, 0.0);
        self.back.clear();
        self.back.resize(t_len * l, 0);
        for t in 1..t_len {
            let (done, cur) = self.alpha.split_at_mut(t * l);
            let prev = &done[(t - 1) * l..];
            let state = &self.state[t * l..(t + 1) * l];
            let back = &mut self.back[t * l..(t + 1) * l];
            for j in 0..l {
                let mut max = -f64::MAX;
                let mut arg: Option<u32> = None;
                for (i, &p) in prev.iter().enumerate() {
                    let score = p + trans[i * l + j];
                    if max < score {
                        max = score;
                        arg = Some(u32::try_from(i).expect("label id"));
                    }
                }
                // CRFsuite keeps the old backward link when no score beats
                // -DBL_MAX (only possible with infinite or NaN weights).
                if let Some(i) = arg {
                    back[j] = i;
                }
                cur[j] = max + state[j];
            }
        }
        let last = &self.alpha[(t_len - 1) * l..t_len * l];
        let mut max = -f64::MAX;
        let mut label = 0u32;
        for (i, &a) in last.iter().enumerate() {
            if max < a {
                max = a;
                label = u32::try_from(i).expect("label id");
            }
        }
        out.resize(t_len, 0);
        out[t_len - 1] = label;
        for t in (0..t_len - 1).rev() {
            out[t] = self.back[(t + 1) * l + out[t + 1] as usize];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model file with labels `["A", "B"]`, attributes `["x", "y"]`, state
    /// features x->A 1.5, x->B 0.5, y->B 2.0 and transitions A->B -1.0,
    /// B->A 0.25, written as CRFsuite's `crf1dmw_*` writers lay it out.
    #[allow(clippy::too_many_lines)]
    fn tiny_model() -> Vec<u8> {
        fn u32le(v: &mut Vec<u8>, x: u32) {
            v.extend_from_slice(&x.to_le_bytes());
        }
        fn cqdb(keys: &[&str]) -> Vec<u8> {
            // Header, 256 table refs (all empty but table 0, which holds
            // 2 * len(keys) buckets so the record count comes out right),
            // the records, the backward links.
            let mut recs = Vec::new();
            let mut offs = Vec::new();
            let data_start = CQDB_HEADER + 8 * CQDB_TABLES;
            for (id, k) in keys.iter().enumerate() {
                offs.push(u32::try_from(data_start + recs.len()).unwrap());
                u32le(&mut recs, u32::try_from(id).unwrap());
                u32le(&mut recs, u32::try_from(k.len() + 1).unwrap());
                recs.extend_from_slice(k.as_bytes());
                recs.push(0);
            }
            let bwd_offset = data_start + recs.len();
            let size = bwd_offset + 4 * keys.len();
            let mut v = Vec::new();
            v.extend_from_slice(CQDB);
            u32le(&mut v, u32::try_from(size).unwrap());
            u32le(&mut v, 0);
            u32le(&mut v, CQDB_BYTEORDER);
            u32le(&mut v, u32::try_from(keys.len()).unwrap());
            u32le(&mut v, u32::try_from(bwd_offset).unwrap());
            for i in 0..CQDB_TABLES {
                // A non-zero offset with num buckets; never read here.
                u32le(&mut v, u32::from(i == 0));
                u32le(
                    &mut v,
                    if i == 0 {
                        2 * u32::try_from(keys.len()).unwrap()
                    } else {
                        0
                    },
                );
            }
            v.extend_from_slice(&recs);
            for o in offs {
                u32le(&mut v, o);
            }
            v
        }
        // Features: 0..3 state, 3..5 transitions.
        let feats: [(u32, u32, u32, f64); 5] = [
            (0, 0, 0, 1.5),
            (0, 0, 1, 0.5),
            (0, 1, 1, 2.0),
            (1, 0, 1, -1.0),
            (1, 1, 0, 0.25),
        ];
        let mut body = Vec::new();
        let off_features = HEADER_SIZE;
        body.extend_from_slice(b"FEAT");
        u32le(&mut body, 0);
        u32le(&mut body, 5);
        for (ty, src, dst, w) in feats {
            u32le(&mut body, ty);
            u32le(&mut body, src);
            u32le(&mut body, dst);
            body.extend_from_slice(&w.to_bits().to_le_bytes());
        }
        let off_labels = HEADER_SIZE + body.len();
        body.extend(cqdb(&["A", "B"]));
        let off_attrs = HEADER_SIZE + body.len();
        body.extend(cqdb(&["x", "y"]));
        // A refs chunk: header, offsets, then each list (count, fids).
        let refs = |body: &mut Vec<u8>, tag: &[u8; 4], lists: &[&[u32]]| -> usize {
            let off = HEADER_SIZE + body.len();
            body.extend_from_slice(tag);
            u32le(body, 0);
            u32le(body, u32::try_from(lists.len()).unwrap());
            let mut at = off + CHUNK_SIZE + 4 * lists.len();
            for l in lists {
                u32le(body, u32::try_from(at).unwrap());
                at += 4 + 4 * l.len();
            }
            for l in lists {
                u32le(body, u32::try_from(l.len()).unwrap());
                for &f in *l {
                    u32le(body, f);
                }
            }
            off
        };
        let off_labelrefs = refs(&mut body, b"LFRF", &[&[3], &[4]]);
        let off_attrrefs = refs(&mut body, b"AFRF", &[&[0, 1], &[2]]);
        let mut v = Vec::new();
        v.extend_from_slice(b"lCRF");
        u32le(&mut v, u32::try_from(HEADER_SIZE + body.len()).unwrap());
        v.extend_from_slice(b"FOMC");
        for x in [
            100,
            0,
            2,
            2,
            off_features,
            off_labels,
            off_attrs,
            off_labelrefs,
            off_attrrefs,
        ] {
            u32le(&mut v, u32::try_from(x).unwrap());
        }
        v.extend(body);
        v
    }

    #[test]
    fn reads_and_tags_a_tiny_model() {
        let m = Model::from_bytes(&tiny_model()).unwrap();
        assert_eq!(m.labels().len(), 2);
        assert_eq!(m.label_id("B"), Some(1));
        assert_eq!(m.attr_id(b"x"), Some(0));
        assert_eq!(m.attr_id(b"y\0junk"), Some(1));
        assert_eq!(m.attr_id(b"z"), None);
        let mut t = Tagger::new(&m);
        let mut out = Vec::new();
        t.viterbi(&mut out);
        assert_eq!(out, Vec::<u32>::new());
        // x then y. Into B at t=1, A (1.5 - 1.0) and B (0.5 + 0.0) tie at
        // 0.5: the first maximum, A, is the backward link. The path ending
        // in B scores 2.5, the one ending in A 1.5.
        t.clear();
        t.push_item();
        t.add_str(b"x");
        t.push_item();
        t.add_str(b"y");
        t.viterbi(&mut out);
        assert_eq!(out, [0, 1]);
        // Unknown and zero-valued attributes change nothing.
        t.clear();
        t.push_item();
        t.add_str(b"nope");
        t.add(1, 0.0);
        t.viterbi(&mut out);
        assert_eq!(out, [0]);
    }

    #[test]
    fn rejects_garbage() {
        assert!(Model::from_bytes(b"").is_err());
        assert!(Model::from_bytes(&[0u8; 64]).is_err());
        let mut m = tiny_model();
        m.truncate(m.len() - 10);
        assert!(Model::from_bytes(&m).is_err());
    }
}
