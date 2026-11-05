// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! Stage 1 (`MinhashDedupSignature`): shingles and signatures.
//!
//! For one document datatrove computes
//!
//! ```python
//! words = word_tokenizer.word_tokenize(simplify_text(text, config.norm_config))
//! shingles = [hash_fc(" ".join(g)) for g in ngrams(words, config.n_grams)]   # uint64
//! phv = (shingles[:, None] * a + b) % _mersenne_prime    # uint64, * and + wrap mod 2**64
//! if precision == 32: phv &= 0xFFFFFFFF
//! signature = phv.min(axis=0)                            # num_buckets * hashes_per_bucket
//! ```
//!
//! A document with fewer than `n_grams` words has no shingles and no
//! signature: datatrove never writes it to a bucket, so it is never removed.

use sha1::{Digest, Sha1};
use twox_hash::{XxHash32, XxHash64};

use super::config::{ConfigError, HashFn, MinhashConfig, Precision, UnicodeVersion};
use super::rng::{MERSENNE_PRIME, permutation_parameters};
use super::simplify::simplify_text;
use super::tokenize::WordTokenizer;

/// SHA-1's initial state.
const SHA1_H0: [u32; 5] = [
    0x6745_2301,
    0xefcd_ab89,
    0x98ba_dcfe,
    0x1032_5476,
    0xc3d2_e1f0,
];

/// The first 8 bytes of `sha1(bytes)`, little-endian (datatrove's
/// `sha1_hash64`; `sha1_hash32` is the low half).
///
/// Shingles are short, so most fit one 64-byte block (<= 55 bytes plus
/// padding): those go straight to the compression function (hardware SHA-1
/// where the CPU has it) without the streaming `Digest` machinery.
#[inline]
pub fn sha1_prefix64(bytes: &[u8]) -> u64 {
    let (h0, h1) = if bytes.len() <= 55 {
        let mut block = [0u8; 64];
        block[..bytes.len()].copy_from_slice(bytes);
        block[bytes.len()] = 0x80;
        block[56..].copy_from_slice(&((bytes.len() as u64) * 8).to_be_bytes());
        let mut state = SHA1_H0;
        sha1::block_api::compress(&mut state, std::slice::from_ref(&block));
        (state[0], state[1])
    } else {
        let d = Sha1::digest(bytes);
        (
            u32::from_be_bytes([d[0], d[1], d[2], d[3]]),
            u32::from_be_bytes([d[4], d[5], d[6], d[7]]),
        )
    };
    // Digest bytes are the state words big-endian; read them back little-endian.
    u64::from(h0.swap_bytes()) | (u64::from(h1.swap_bytes()) << 32)
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

/// Everything stage 1 needs for one configuration: the config, the
/// interpreter's Unicode version and the permutation parameters. Cheap to
/// share between threads.
#[derive(Clone, Debug)]
pub struct MinHasher {
    config: MinhashConfig,
    unicode: UnicodeVersion,
    a: Box<[u64]>,
    b: Box<[u64]>,
}

impl MinHasher {
    pub fn new(config: MinhashConfig, unicode: UnicodeVersion) -> Result<Self, ConfigError> {
        config.validate()?;
        let (a, b) = permutation_parameters(config.seed, config.num_hashes());
        Ok(Self {
            config,
            unicode,
            a: a.into_boxed_slice(),
            b: b.into_boxed_slice(),
        })
    }

    pub fn config(&self) -> &MinhashConfig {
        &self.config
    }

    pub fn unicode(&self) -> UnicodeVersion {
        self.unicode
    }

    /// Hashes per signature.
    pub fn num_hashes(&self) -> usize {
        self.a.len()
    }

    /// The permutation parameters `(a, b)` (datatrove's `parameters`).
    pub fn parameters(&self) -> (&[u64], &[u64]) {
        (&self.a, &self.b)
    }

    /// datatrove's `hash_fc` on the UTF-8 bytes of one shingle string.
    #[inline]
    pub fn hash_shingle(&self, bytes: &[u8]) -> u64 {
        let hc = self.config.hash_config;
        match (hc.hash_fc, hc.precision) {
            (HashFn::Xxhash, Precision::P64) => XxHash64::oneshot(0, bytes),
            (HashFn::Xxhash, Precision::P32) => u64::from(XxHash32::oneshot(0, bytes)),
            (HashFn::Sha1, Precision::P64) => sha1_prefix64(bytes),
            (HashFn::Sha1, Precision::P32) => sha1_prefix64(bytes) & u64::from(u32::MAX),
        }
    }

    /// datatrove's `get_shingles`, given the document's words: the hash of
    /// every `n_grams`-word window joined by single spaces. Replaces `out`.
    pub fn shingles_from_tokens<S: AsRef<str>>(&self, tokens: &[S], out: &mut Vec<u64>) {
        out.clear();
        let n = self.config.n_grams;
        if tokens.len() < n {
            return;
        }
        out.reserve(tokens.len() + 1 - n);
        let mut buf: Vec<u8> = Vec::with_capacity(64);
        for window in tokens.windows(n) {
            buf.clear();
            for (j, t) in window.iter().enumerate() {
                if j > 0 {
                    buf.push(b' ');
                }
                buf.extend_from_slice(t.as_ref().as_bytes());
            }
            out.push(self.hash_shingle(&buf));
        }
    }

    /// datatrove's `get_signature`, flattened bucket by bucket into `out`
    /// (`num_hashes` values). Returns `false`, leaving `out` unspecified,
    /// when there are no shingles (datatrove writes no signature then).
    pub fn signature_from_shingles(&self, shingles: &[u64], out: &mut [u64]) -> bool {
        assert_eq!(out.len(), self.num_hashes(), "signature buffer size");
        if shingles.is_empty() {
            return false;
        }
        let mask = match self.config.hash_config.precision {
            Precision::P32 => u64::from(u32::MAX),
            Precision::P64 => u64::MAX,
        };
        // Four permutations at a time: independent dependency chains keep
        // the multipliers busy while each minimum stays in a register.
        let mut h = 0;
        let n = self.num_hashes();
        while h + 4 <= n {
            let (a0, a1, a2, a3) = (self.a[h], self.a[h + 1], self.a[h + 2], self.a[h + 3]);
            let (b0, b1, b2, b3) = (self.b[h], self.b[h + 1], self.b[h + 2], self.b[h + 3]);
            let (mut m0, mut m1, mut m2, mut m3) = (u64::MAX, u64::MAX, u64::MAX, u64::MAX);
            for &s in shingles {
                m0 = m0.min(mod_mersenne(s.wrapping_mul(a0).wrapping_add(b0)) & mask);
                m1 = m1.min(mod_mersenne(s.wrapping_mul(a1).wrapping_add(b1)) & mask);
                m2 = m2.min(mod_mersenne(s.wrapping_mul(a2).wrapping_add(b2)) & mask);
                m3 = m3.min(mod_mersenne(s.wrapping_mul(a3).wrapping_add(b3)) & mask);
            }
            out[h..h + 4].copy_from_slice(&[m0, m1, m2, m3]);
            h += 4;
        }
        for (k, o) in out.iter_mut().enumerate().skip(h) {
            let (a, b) = (self.a[k], self.b[k]);
            *o = shingles
                .iter()
                .map(|&s| mod_mersenne(s.wrapping_mul(a).wrapping_add(b)) & mask)
                .min()
                .unwrap_or(u64::MAX);
        }
        true
    }

    /// Signature from the document's words (see [`Self::shingles_from_tokens`]).
    pub fn signature_from_tokens<S: AsRef<str>>(&self, tokens: &[S], out: &mut [u64]) -> bool {
        let mut shingles = Vec::new();
        self.shingles_from_tokens(tokens, &mut shingles);
        self.signature_from_shingles(&shingles, out)
    }

    /// The whole of stage 1 for one document: `simplify_text`, the word
    /// tokenizer, shingles, signature.
    pub fn signature<T: WordTokenizer + ?Sized>(
        &self,
        text: &str,
        tokenizer: &T,
        out: &mut [u64],
    ) -> bool {
        let simplified = simplify_text(text, &self.config.norm_config, self.unicode);
        let mut tokens = Vec::new();
        tokenizer.word_tokenize(&simplified, &mut tokens);
        self.signature_from_tokens(&tokens, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dedup::config::HashConfig;

    fn hasher(hash_fc: HashFn, precision: Precision) -> MinHasher {
        let config = MinhashConfig {
            hash_config: HashConfig { precision, hash_fc },
            ..MinhashConfig::default()
        };
        MinHasher::new(config, UnicodeVersion::LATEST).unwrap()
    }

    #[test]
    fn mod_mersenne_matches_rem() {
        for x in [
            0,
            1,
            MERSENNE_PRIME - 1,
            MERSENNE_PRIME,
            MERSENNE_PRIME + 1,
            2 * MERSENNE_PRIME,
            u64::MAX,
            u64::MAX - 7,
            0x1fff_ffff_ffff_fff8,
            0xe000_0000_0000_0000,
            0xffff_ffff_0000_0001,
        ] {
            assert_eq!(mod_mersenne(x), x % MERSENNE_PRIME, "{x:#x}");
        }
        let mut x = 0x9e37_79b9_7f4a_7c15_u64;
        for _ in 0..100_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            assert_eq!(mod_mersenne(x), x % MERSENNE_PRIME, "{x:#x}");
        }
    }

    #[test]
    fn shingle_hashes_match_python() {
        // struct.unpack("<Q", hashlib.sha1(b"a b c d e").digest()[:8])[0], etc.
        // sha1(b"a b c d e") = 8795a3b65131dc10...
        let s = b"a b c d e";
        assert_eq!(
            hasher(HashFn::Sha1, Precision::P64).hash_shingle(s),
            0x10dc_3151_b6a3_9587
        );
        assert_eq!(
            hasher(HashFn::Sha1, Precision::P32).hash_shingle(s),
            0xb6a3_9587
        );
        // xxhash64(b"") with seed 0 is the published test vector.
        assert_eq!(
            hasher(HashFn::Xxhash, Precision::P64).hash_shingle(b""),
            0xef46_db37_51d8_e999
        );
        assert_eq!(
            hasher(HashFn::Xxhash, Precision::P32).hash_shingle(b""),
            0x02cc_5d05
        );
    }

    #[test]
    fn sha1_fast_path_matches_digest() {
        let data: Vec<u8> = (0..300u32).map(|i| (i * 131 % 251) as u8).collect();
        for len in 0..data.len() {
            let d = Sha1::digest(&data[..len]);
            let want = u64::from_le_bytes([d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7]]);
            assert_eq!(sha1_prefix64(&data[..len]), want, "len {len}");
        }
    }

    #[test]
    fn short_documents_have_no_signature() {
        let h = hasher(HashFn::Sha1, Precision::P64);
        let mut out = vec![0; h.num_hashes()];
        assert!(!h.signature_from_tokens(&["a", "b", "c", "d"], &mut out));
        assert!(h.signature_from_tokens(&["a", "b", "c", "d", "e"], &mut out));
        let mut sh = Vec::new();
        h.shingles_from_tokens(&["a", "b", "c", "d", "e", "f"], &mut sh);
        assert_eq!(sh.len(), 2);
        assert_eq!(sh[0], h.hash_shingle(b"a b c d e"));
        assert_eq!(sh[1], h.hash_shingle(b"b c d e f"));
    }

    #[test]
    fn signature_is_min_over_permutations() {
        for p in [Precision::P32, Precision::P64] {
            let h = hasher(HashFn::Xxhash, p);
            let shingles = [1u64, 0xdead_beef, u64::MAX, 42, 1 << 63];
            let mut out = vec![0; h.num_hashes()];
            assert!(h.signature_from_shingles(&shingles, &mut out));
            let (a, b) = h.parameters();
            for k in 0..h.num_hashes() {
                let want = shingles
                    .iter()
                    .map(|&s| {
                        let v = u64::try_from(
                            u128::from(s.wrapping_mul(a[k]).wrapping_add(b[k]))
                                % u128::from(MERSENNE_PRIME),
                        )
                        .unwrap();
                        if p == Precision::P32 {
                            v & 0xffff_ffff
                        } else {
                            v
                        }
                    })
                    .min()
                    .unwrap();
                assert_eq!(out[k], want);
            }
        }
    }
}
