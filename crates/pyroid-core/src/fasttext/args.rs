// Part of pyroid's port of fastText (https://github.com/facebookresearch/fastText).
// fastText: Copyright (c) 2016-present, Facebook, Inc. MIT License.

//! The model header: training arguments stored in the file (`Args::load`).

use std::io::Read;

use super::error::LoadError;
use super::io::Reader;

/// `model_name` in fastText.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelKind {
    Cbow,
    Skipgram,
    Supervised,
}

/// `loss_name` in fastText.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LossKind {
    /// Hierarchical softmax (`hs`), e.g. lid.176.
    HierarchicalSoftmax,
    /// Negative sampling (`ns`).
    NegativeSampling,
    /// Softmax.
    Softmax,
    /// One-vs-all (`ova`), independent sigmoids.
    OneVsAll,
}

/// The arguments a model was trained with, as stored in its header. Fields
/// that only matter for training are kept for introspection.
#[derive(Clone, Debug, PartialEq)]
pub struct Args {
    pub dim: i32,
    pub ws: i32,
    pub epoch: i32,
    pub min_count: i32,
    pub neg: i32,
    pub word_ngrams: i32,
    pub loss: LossKind,
    pub model: ModelKind,
    pub bucket: i32,
    pub minn: i32,
    pub maxn: i32,
    pub lr_update_rate: i32,
    pub t: f64,
}

impl Args {
    pub(crate) fn load<R: Read>(r: &mut Reader<R>) -> Result<Self, LoadError> {
        let dim = r.i32("args.dim")?;
        let ws = r.i32("args.ws")?;
        let epoch = r.i32("args.epoch")?;
        let min_count = r.i32("args.minCount")?;
        let neg = r.i32("args.neg")?;
        let word_ngrams = r.i32("args.wordNgrams")?;
        let loss = match r.i32("args.loss")? {
            1 => LossKind::HierarchicalSoftmax,
            2 => LossKind::NegativeSampling,
            3 => LossKind::Softmax,
            4 => LossKind::OneVsAll,
            other => {
                return Err(LoadError::Invalid(format!(
                    "unknown loss {other} (expected 1-4: hs, ns, softmax, ova)"
                )));
            }
        };
        let model = match r.i32("args.model")? {
            1 => ModelKind::Cbow,
            2 => ModelKind::Skipgram,
            3 => ModelKind::Supervised,
            other => {
                return Err(LoadError::Invalid(format!(
                    "unknown model type {other} (expected 1-3: cbow, skipgram, supervised)"
                )));
            }
        };
        Ok(Self {
            dim,
            ws,
            epoch,
            min_count,
            neg,
            word_ngrams,
            loss,
            model,
            bucket: r.i32("args.bucket")?,
            minn: r.i32("args.minn")?,
            maxn: r.i32("args.maxn")?,
            lr_update_rate: r.i32("args.lrUpdateRate")?,
            t: r.f64("args.t")?,
        })
    }
}
