// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! datasketch's MinHash arithmetic: the SHA1 hash functions, the three
//! permutation schemes and the signature update.
//!
//! For an input hash `h` and permutation `k`, datasketch 2.0.0 computes
//! (`MinHash.update` / `update_batch`, numpy integer arithmetic, which wraps):
//!
//! - `"legacy"`: `((a_k * h + b_k) mod 2**64) % (2**61 - 1) & 0xFFFF_FFFF`,
//!   in `uint64`;
//! - `"affine32"`: `a_k * fmix32(h) + b_k` in `uint32`;
//! - `"affine64"`: `a_k * fmix64(h) + b_k` in `uint64`;
//!
//! and keeps the minimum per permutation. Here `h` is `sha1_hash32` (legacy,
//! affine32) or `sha1_hash64` (affine64) of the token, or a value the
//! caller's own hash function returned.

pub use crate::dedup::minhash::sha1_prefix64;

/// `2**61 - 1`, datasketch's `_mersenne_prime`.
pub const MERSENNE_PRIME: u64 = (1 << 61) - 1;

/// The largest legacy hash value (`_max_hash`, `2**32 - 1`).
pub const LEGACY_MAX: u64 = (1 << 32) - 1;

/// datasketch's permutation schemes (`MinHash(scheme=...)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Scheme {
    /// The scheme of datasketch < 2.0.0: values below 2**32 stored as `uint64`.
    Legacy,
    /// The default since 2.0.0: `uint32` values.
    Affine32,
    /// 64-bit affine permutations: `uint64` values.
    Affine64,
}

impl Scheme {
    /// The scheme's name, as datasketch spells it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Legacy => "legacy",
            Self::Affine32 => "affine32",
            Self::Affine64 => "affine64",
        }
    }

    /// The scheme for a datasketch name.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "legacy" => Some(Self::Legacy),
            "affine32" => Some(Self::Affine32),
            "affine64" => Some(Self::Affine64),
            _ => None,
        }
    }

    /// The value an empty MinHash holds (`_SCHEME_MAX_HASHES`).
    pub fn max_value(self) -> u64 {
        match self {
            Self::Legacy | Self::Affine32 => LEGACY_MAX,
            Self::Affine64 => u64::MAX,
        }
    }
}

/// `datasketch.hashfunc.sha1_hash32`: the first 4 bytes of SHA1, little-endian.
#[inline]
pub fn sha1_hash32(bytes: &[u8]) -> u32 {
    // The low half of the 8-byte little-endian prefix.
    #[allow(clippy::cast_possible_truncation)]
    let h = sha1_prefix64(bytes) as u32;
    h
}

/// `datasketch.hashfunc.sha1_hash64`: the first 8 bytes of SHA1, little-endian.
#[inline]
pub fn sha1_hash64(bytes: &[u8]) -> u64 {
    sha1_prefix64(bytes)
}

/// The 32-bit MurmurHash3 finalizer (`_fmix(hv, 32)`).
#[inline]
pub fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
}

/// The 64-bit MurmurHash3 finalizer (`_fmix(hv, 64)`).
#[inline]
pub fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

/// `x % (2**61 - 1)` for any `u64`.
#[inline]
fn mod_mersenne(x: u64) -> u64 {
    // x = hi * 2**61 + lo with hi < 8, and 2**61 = 1 (mod p): x = hi + lo (mod p),
    // and hi + lo <= p + 7, so one conditional subtraction finishes it.
    let y = (x & MERSENNE_PRIME) + (x >> 61);
    if y >= MERSENNE_PRIME {
        y - MERSENNE_PRIME
    } else {
        y
    }
}

/// The permutation parameters of one MinHash (`MinHash.permutations`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Permutations {
    Legacy { a: Vec<u64>, b: Vec<u64> },
    Affine32 { a: Vec<u32>, b: Vec<u32> },
    Affine64 { a: Vec<u64>, b: Vec<u64> },
}

impl Permutations {
    pub fn scheme(&self) -> Scheme {
        match self {
            Self::Legacy { .. } => Scheme::Legacy,
            Self::Affine32 { .. } => Scheme::Affine32,
            Self::Affine64 { .. } => Scheme::Affine64,
        }
    }

    /// The number of permutations.
    pub fn len(&self) -> usize {
        match self {
            Self::Legacy { a, .. } | Self::Affine64 { a, .. } => a.len(),
            Self::Affine32 { a, .. } => a.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Signature values of one scheme: `u64` for legacy and affine64, `u32` for
/// affine32.
pub trait Value: Copy + Ord + Send + Sync + 'static {}
impl Value for u32 {}
impl Value for u64 {}

/// Fold the hashes `hs` into `hv`, legacy scheme. `hs` are the hash
/// function's outputs (any `u64`: numpy's `a * h` wraps for 64-bit values).
pub fn update_legacy(hv: &mut [u64], a: &[u64], b: &[u64], hs: impl IntoIterator<Item = u64>) {
    let n = hv.len();
    let (a, b) = (&a[..n], &b[..n]);
    for h in hs {
        for k in 0..n {
            let v = mod_mersenne(a[k].wrapping_mul(h).wrapping_add(b[k])) & LEGACY_MAX;
            hv[k] = hv[k].min(v);
        }
    }
}

/// Fold the hashes `hs` (before `fmix32`) into `hv`, affine32 scheme.
pub fn update_affine32(hv: &mut [u32], a: &[u32], b: &[u32], hs: impl IntoIterator<Item = u32>) {
    let n = hv.len();
    let (a, b) = (&a[..n], &b[..n]);
    for h in hs {
        let h = fmix32(h);
        for k in 0..n {
            hv[k] = hv[k].min(a[k].wrapping_mul(h).wrapping_add(b[k]));
        }
    }
}

/// Fold the hashes `hs` (before `fmix64`) into `hv`, affine64 scheme.
pub fn update_affine64(hv: &mut [u64], a: &[u64], b: &[u64], hs: impl IntoIterator<Item = u64>) {
    let n = hv.len();
    let (a, b) = (&a[..n], &b[..n]);
    for h in hs {
        let h = fmix64(h);
        for k in 0..n {
            hv[k] = hv[k].min(a[k].wrapping_mul(h).wrapping_add(b[k]));
        }
    }
}

/// Which SHA1 prefix hashes the tokens (`sha1_hash32` or `sha1_hash64`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sha1Bits {
    B32,
    B64,
}

impl Sha1Bits {
    #[inline]
    pub fn hash(self, bytes: &[u8]) -> u64 {
        match self {
            Self::B32 => u64::from(sha1_hash32(bytes)),
            Self::B64 => sha1_hash64(bytes),
        }
    }
}

/// A signature buffer of the scheme's value type.
#[derive(Debug)]
pub enum Signature<'a> {
    U64(&'a mut [u64]),
    U32(&'a mut [u32]),
}

/// Why tokens can't be folded natively.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateError {
    /// A hash value does not fit the scheme's width (affine32 and a 64-bit hash).
    HashTooWide,
    /// The signature's value type is not the scheme's.
    WrongSignatureType,
}

impl Permutations {
    /// Fold already-computed hash values into `sig`.
    ///
    /// For affine32 every value must be below 2**32 (datasketch's
    /// `_hash_input_array` raises otherwise); then `Err(HashTooWide)` and
    /// `sig` is unchanged.
    pub fn update_hashes(&self, sig: Signature<'_>, hs: &[u64]) -> Result<(), UpdateError> {
        match (self, sig) {
            (Self::Legacy { a, b }, Signature::U64(hv)) => {
                update_legacy(hv, a, b, hs.iter().copied());
            }
            (Self::Affine64 { a, b }, Signature::U64(hv)) => {
                update_affine64(hv, a, b, hs.iter().copied());
            }
            (Self::Affine32 { a, b }, Signature::U32(hv)) => {
                if hs.iter().any(|&h| h > u64::from(u32::MAX)) {
                    return Err(UpdateError::HashTooWide);
                }
                #[allow(clippy::cast_possible_truncation)]
                update_affine32(hv, a, b, hs.iter().map(|&h| h as u32));
            }
            _ => return Err(UpdateError::WrongSignatureType),
        }
        Ok(())
    }

    /// Hash every token with `bits` and fold it into `sig`.
    pub fn update_tokens<'t>(
        &self,
        sig: Signature<'_>,
        bits: Sha1Bits,
        tokens: impl IntoIterator<Item = &'t [u8]>,
    ) -> Result<(), UpdateError> {
        match (self, sig, bits) {
            (Self::Legacy { a, b }, Signature::U64(hv), _) => {
                update_legacy(hv, a, b, tokens.into_iter().map(|t| bits.hash(t)));
            }
            (Self::Affine64 { a, b }, Signature::U64(hv), _) => {
                update_affine64(hv, a, b, tokens.into_iter().map(|t| bits.hash(t)));
            }
            (Self::Affine32 { a, b }, Signature::U32(hv), Sha1Bits::B32) => {
                update_affine32(hv, a, b, tokens.into_iter().map(sha1_hash32));
            }
            (Self::Affine32 { .. }, Signature::U32(_), Sha1Bits::B64) => {
                return Err(UpdateError::HashTooWide);
            }
            _ => return Err(UpdateError::WrongSignatureType),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha1_prefixes() {
        // struct.unpack("<I"/"<Q", hashlib.sha1(b"abc").digest()[:4]/[:8]):
        // sha1(b"abc") = a9993e364706816aba3e2571...
        assert_eq!(sha1_hash32(b"abc"), 0x363e_99a9);
        assert_eq!(sha1_hash64(b"abc"), 0x6a81_0647_363e_99a9);
    }

    #[test]
    fn fmix_known_values() {
        // MurmurHash3's finalizers map 0 to 0 and are bijections.
        assert_eq!(fmix32(0), 0);
        assert_eq!(fmix64(0), 0);
        assert_eq!(fmix32(1), 0x514e_28b7);
        assert_eq!(fmix64(1), 0xb456_bcfc_34c2_cb2c);
    }

    #[test]
    fn mersenne_reduction() {
        for x in [
            0,
            1,
            MERSENNE_PRIME - 1,
            MERSENNE_PRIME,
            MERSENNE_PRIME + 1,
            u64::MAX,
            u64::MAX - 7,
            1 << 63,
        ] {
            assert_eq!(mod_mersenne(x), x % MERSENNE_PRIME, "{x}");
        }
    }

    #[test]
    fn affine32_rejects_wide_hashes() {
        let p = Permutations::Affine32 {
            a: vec![1, 3],
            b: vec![0, 0],
        };
        let mut hv = [u32::MAX; 2];
        assert_eq!(
            p.update_hashes(Signature::U32(&mut hv), &[1 << 32]),
            Err(UpdateError::HashTooWide)
        );
        assert_eq!(hv, [u32::MAX; 2]);
        p.update_hashes(Signature::U32(&mut hv), &[0]).unwrap();
        assert_eq!(hv, [0, 0]);
    }
}
