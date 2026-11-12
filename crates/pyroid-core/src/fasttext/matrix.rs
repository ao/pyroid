// Part of pyroid's port of fastText (https://github.com/facebookresearch/fastText).
// fastText: Copyright (c) 2016-present, Facebook, Inc. MIT License.

//! Input and output matrices: dense (`DenseMatrix`) and product-quantized
//! (`QuantMatrix` + `ProductQuantizer`, as written by `fasttext quantize`).
//!
//! Floating-point operations happen in fastText's order with fastText's
//! roundings, so results are bit-identical to the reference build. The one
//! platform-dependent step is [`fused`] (see there).

use std::io::Read;

use super::error::LoadError;
use super::io::Reader;

/// `acc + a * b` in the quantized kernels (`mulcode`, `addcode`) and
/// `Vector::norm`, as the reference builds compute it.
///
/// The published fastText wheels are built by GCC with `-O3`, which
/// contracts these loops into fused multiply-adds where the base ISA has
/// them: on aarch64 the reference rounds once, on baseline x86-64 twice.
/// The dense `dotRow` is vectorized instead: products four at a time, added
/// in order, and only its scalar tail is fused (see [`Matrix::dot_row`]).
/// Measured against fasttext-predict 0.9.2.4 on aarch64: with this split, all
/// 26,464 pinned lid.176 predictions, every synthetic qout/qnorm model and
/// dense models of every dim from 1 to 33 are bit-identical;
/// against upstream fastText 0.9.2 (`fasttext-numpy2-wheel`), so are
/// unsupervised sentence vectors (`norm`).
#[inline]
pub fn fused(acc: f32, a: f32, b: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, acc)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        acc + a * b
    }
}

/// A dot product produced NaN (fastText's `EncounteredNaNError`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NaN;

#[derive(Debug)]
pub struct Dense {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

/// `ProductQuantizer` (8 bits per code: 256 centroids per sub-quantizer).
#[derive(Debug)]
pub struct ProductQuantizer {
    nsubq: usize,
    dsub: usize,
    lastdsub: usize,
    centroids: Vec<f32>,
}

const KSUB: usize = 256;

#[derive(Debug)]
pub struct Quant {
    rows: usize,
    cols: usize,
    codes: Vec<u8>,
    pq: ProductQuantizer,
    /// Per-row norm codes and their quantizer, when `qnorm` was used.
    norm: Option<(Vec<u8>, ProductQuantizer)>,
}

#[derive(Debug)]
pub enum Matrix {
    Dense(Dense),
    Quant(Quant),
}

fn dims<R: Read>(r: &mut Reader<R>, what: &'static str) -> Result<(usize, usize), LoadError> {
    let m = r.i64(what)?;
    let n = r.i64(what)?;
    match (usize::try_from(m), usize::try_from(n)) {
        (Ok(m), Ok(n)) => Ok((m, n)),
        _ => Err(LoadError::Invalid(format!(
            "{what} has negative shape {m} x {n}"
        ))),
    }
}

impl Dense {
    fn load<R: Read>(r: &mut Reader<R>, what: &'static str) -> Result<Self, LoadError> {
        let (rows, cols) = dims(r, what)?;
        let count = (rows as u64)
            .checked_mul(cols as u64)
            .ok_or_else(|| LoadError::Invalid(format!("{what}: {rows} x {cols} overflows")))?;
        let data = r.f32s(count, what)?;
        Ok(Self { rows, cols, data })
    }

    #[inline]
    fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

impl ProductQuantizer {
    fn load<R: Read>(r: &mut Reader<R>, what: &'static str) -> Result<Self, LoadError> {
        let dim = r.i32(what)?;
        let nsubq = r.i32(what)?;
        let dsub = r.i32(what)?;
        let lastdsub = r.i32(what)?;
        let bad = || {
            LoadError::Invalid(format!(
                "{what}: inconsistent product quantizer (dim {dim}, nsubq {nsubq}, dsub {dsub}, \
                 lastdsub {lastdsub})"
            ))
        };
        let (Ok(udim), Ok(nsubq), Ok(dsub), Ok(lastdsub)) = (
            usize::try_from(dim),
            usize::try_from(nsubq),
            usize::try_from(dsub),
            usize::try_from(lastdsub),
        ) else {
            return Err(bad());
        };
        // The layout `get_centroids` assumes: nsubq - 1 blocks of dsub
        // columns, then one of lastdsub.
        if nsubq == 0
            || dsub == 0
            || lastdsub == 0
            || lastdsub > dsub
            || (nsubq - 1)
                .checked_mul(dsub)
                .and_then(|x| x.checked_add(lastdsub))
                != Some(udim)
        {
            return Err(bad());
        }
        let centroids = r.f32s(udim as u64 * KSUB as u64, what)?;
        Ok(Self {
            nsubq,
            dsub,
            lastdsub,
            centroids,
        })
    }

    fn dim(&self) -> usize {
        (self.nsubq - 1) * self.dsub + self.lastdsub
    }

    /// `get_centroids(m, i)`: centroid `i` of sub-quantizer `m`.
    #[inline]
    fn centroid(&self, m: usize, i: u8) -> &[f32] {
        let i = usize::from(i);
        if m == self.nsubq - 1 {
            let start = m * KSUB * self.dsub + i * self.lastdsub;
            &self.centroids[start..start + self.lastdsub]
        } else {
            let start = (m * KSUB + i) * self.dsub;
            &self.centroids[start..start + self.dsub]
        }
    }

    /// `mulcode`: `alpha * <x, decode(code)>`.
    #[inline]
    fn mulcode(&self, x: &[f32], code: &[u8], alpha: f32) -> f32 {
        let mut res = 0.0f32;
        for (m, &c) in code.iter().enumerate() {
            let cent = self.centroid(m, c);
            let xs = &x[m * self.dsub..m * self.dsub + cent.len()];
            for (&a, &b) in xs.iter().zip(cent) {
                res = fused(res, a, b);
            }
        }
        res * alpha
    }

    /// `addcode`: `x += alpha * decode(code)`.
    #[inline]
    fn addcode(&self, x: &mut [f32], code: &[u8], alpha: f32) {
        for (m, &c) in code.iter().enumerate() {
            let cent = self.centroid(m, c);
            let xs = &mut x[m * self.dsub..m * self.dsub + cent.len()];
            for (xv, &cv) in xs.iter_mut().zip(cent) {
                *xv = fused(*xv, alpha, cv);
            }
        }
    }
}

impl Quant {
    fn load<R: Read>(r: &mut Reader<R>, what: &'static str) -> Result<Self, LoadError> {
        let qnorm = r.bool(what)?;
        let (rows, cols) = dims(r, what)?;
        let codesize = r.i32(what)?;
        let codesize = u64::try_from(codesize)
            .map_err(|_| LoadError::Invalid(format!("{what}: negative code size {codesize}")))?;
        let codes = r.bytes(codesize, what)?;
        let pq = ProductQuantizer::load(r, what)?;
        if pq.dim() != cols {
            return Err(LoadError::Invalid(format!(
                "{what}: quantizer dimension {} does not match the {cols} columns",
                pq.dim()
            )));
        }
        // An overflowing product is as short of codes as any (`None` would
        // compare below every `Some`).
        let need = (rows as u64).checked_mul(pq.nsubq as u64);
        if need.is_none_or(|need| need > codes.len() as u64) {
            return Err(LoadError::Invalid(format!(
                "{what}: {} code bytes for {rows} rows of {} sub-quantizers",
                codes.len(),
                pq.nsubq
            )));
        }
        let norm = if qnorm {
            let norm_codes = r.bytes(rows as u64, what)?;
            let npq = ProductQuantizer::load(r, what)?;
            Some((norm_codes, npq))
        } else {
            None
        };
        Ok(Self {
            rows,
            cols,
            codes,
            pq,
            norm,
        })
    }

    #[inline]
    fn code(&self, i: usize) -> &[u8] {
        let n = self.pq.nsubq;
        &self.codes[i * n..(i + 1) * n]
    }

    /// The row's norm (1 without `qnorm`).
    #[inline]
    fn norm(&self, i: usize) -> f32 {
        match &self.norm {
            Some((codes, npq)) => npq.centroid(0, codes[i])[0],
            None => 1.0,
        }
    }
}

impl Matrix {
    pub(crate) fn load<R: Read>(
        r: &mut Reader<R>,
        quantized: bool,
        what: &'static str,
    ) -> Result<Self, LoadError> {
        Ok(if quantized {
            Self::Quant(Quant::load(r, what)?)
        } else {
            Self::Dense(Dense::load(r, what)?)
        })
    }

    pub fn rows(&self) -> usize {
        match self {
            Self::Dense(d) => d.rows,
            Self::Quant(q) => q.rows,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Dense(d) => d.cols,
            Self::Quant(q) => q.cols,
        }
    }

    pub fn as_dense(&self) -> Option<&Dense> {
        match self {
            Self::Dense(d) => Some(d),
            Self::Quant(_) => None,
        }
    }

    /// `addRowToVector(x, i)`: `x += row i`. `x.len()` must be `cols`.
    #[inline]
    pub fn add_row(&self, x: &mut [f32], i: usize) {
        match self {
            Self::Dense(d) => {
                for (xv, &a) in x.iter_mut().zip(d.row(i)) {
                    *xv += a;
                }
            }
            Self::Quant(q) => q.pq.addcode(x, q.code(i), q.norm(i)),
        }
    }

    /// `dotRow(v, i)`. Dense rows check for NaN like fastText; quantized
    /// rows do not (nor does fastText).
    #[inline]
    pub fn dot_row(&self, v: &[f32], i: usize) -> Result<f32, NaN> {
        match self {
            Self::Dense(d) => {
                let row = d.row(i);
                // GCC vectorizes the products four at a time and adds them in
                // order (two roundings); the scalar tail of `cols % 4` terms
                // is contracted into fused multiply-adds.
                let body = row.len() - row.len() % 4;
                let mut acc = 0.0f32;
                for (&a, &b) in row[..body].iter().zip(v) {
                    acc += a * b;
                }
                for (&a, &b) in row[body..].iter().zip(&v[body..]) {
                    acc = fused(acc, a, b);
                }
                if acc.is_nan() { Err(NaN) } else { Ok(acc) }
            }
            Self::Quant(q) => Ok(q.pq.mulcode(v, q.code(i), q.norm(i))),
        }
    }
}
