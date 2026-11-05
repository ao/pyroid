// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! Configuration, mirroring datatrove's `MinhashConfig`, `HashConfig` and
//! `TextNormConfig` field for field, with the same defaults.

use std::fmt;

pub use crate::ftfy::UnicodeVersion;

/// The shingle hash function (`HashConfig.hash_fc`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum HashFn {
    /// The first 4 or 8 bytes of SHA-1, little-endian.
    Sha1,
    /// xxHash32 or xxHash64 with seed 0 (datatrove's default).
    #[default]
    Xxhash,
}

impl HashFn {
    /// Parse datatrove's spelling (`"sha1"`, `"xxhash"`).
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "sha1" => Some(Self::Sha1),
            "xxhash" => Some(Self::Xxhash),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sha1 => "sha1",
            Self::Xxhash => "xxhash",
        }
    }
}

/// Hash width (`HashConfig.precision`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Precision {
    P32,
    #[default]
    P64,
}

impl Precision {
    pub fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            32 => Some(Self::P32),
            64 => Some(Self::P64),
            _ => None,
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            Self::P32 => 32,
            Self::P64 => 64,
        }
    }
}

/// datatrove's `HashConfig` (default: xxhash, 64 bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct HashConfig {
    pub precision: Precision,
    pub hash_fc: HashFn,
}

/// datatrove's `TextNormConfig`: what `simplify_text` does before shingling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(clippy::struct_excessive_bools)]
pub struct TextNormConfig {
    pub lowercase: bool,
    pub norm_whitespace: bool,
    pub remove_punctuation: bool,
    pub norm_unicode_diacritics: bool,
    pub norm_numbers: bool,
    pub norm_weekdays: bool,
    pub norm_monthnames: bool,
}

impl Default for TextNormConfig {
    fn default() -> Self {
        Self {
            lowercase: true,
            norm_whitespace: true,
            remove_punctuation: true,
            norm_unicode_diacritics: true,
            norm_numbers: true,
            norm_weekdays: false,
            norm_monthnames: false,
        }
    }
}

/// datatrove's `MinhashConfig` (defaults: 5-grams, 14 buckets of 8 hashes,
/// seed 1, xxhash/64, the default `TextNormConfig`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MinhashConfig {
    pub n_grams: usize,
    pub num_buckets: usize,
    pub hashes_per_bucket: usize,
    pub seed: u32,
    pub norm_config: TextNormConfig,
    pub hash_config: HashConfig,
}

impl Default for MinhashConfig {
    fn default() -> Self {
        Self {
            n_grams: 5,
            num_buckets: 14,
            hashes_per_bucket: 8,
            seed: 1,
            norm_config: TextNormConfig::default(),
            hash_config: HashConfig::default(),
        }
    }
}

/// A configuration value pyroid does not accept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConfigError(pub String);

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for ConfigError {}

impl MinhashConfig {
    /// Total hashes per signature (`num_buckets * hashes_per_bucket`).
    pub fn num_hashes(&self) -> usize {
        self.num_buckets * self.hashes_per_bucket
    }

    /// Reject values datatrove cannot run with (or that would make every
    /// signature empty).
    pub fn validate(&self) -> Result<(), ConfigError> {
        for (name, v) in [
            ("n_grams", self.n_grams),
            ("num_buckets", self.num_buckets),
            ("hashes_per_bucket", self.hashes_per_bucket),
        ] {
            if v == 0 {
                return Err(ConfigError(format!("{name} must be at least 1")));
            }
        }
        if self
            .num_buckets
            .checked_mul(self.hashes_per_bucket)
            .is_none_or(|n| n > 1 << 20)
        {
            return Err(ConfigError(
                "num_buckets * hashes_per_bucket must be at most 2**20".to_owned(),
            ));
        }
        Ok(())
    }
}
