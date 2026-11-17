// Derived from CPython's Modules/_randommodule.c and Lib/random.py (PSF-2.0), and from
// MT19937 by Makoto Matsumoto and Takuji Nishimura (BSD-3-Clause); modified.
//! CPython's `random.Random`, as far as langdetect uses it: the Mersenne Twister
//! (`_random.Random`), `random()`, `getrandbits()`, `_randbelow()` / `choice()` and `gauss()`.
//!
//! Every operation draws the same 32-bit words in the same order as CPython 3.11-3.15, so a
//! generator seeded like a `random.Random` gives the same floats and choices, bit for bit.

use std::f64::consts::PI;
use std::hint::black_box;

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;
/// `random.TWOPI`.
const TWOPI: f64 = 2.0 * PI;

/// The 624 state words of a seeded generator, before its first draw (`index == 624`).
pub type SeededState = [u32; N];

/// A `random.Random`: MT19937 state plus `gauss()`'s cached second value.
#[derive(Clone, Debug)]
pub struct Random {
    state: [u32; N],
    index: usize,
    gauss_next: Option<f64>,
}

fn init_genrand(s: u32) -> [u32; N] {
    let mut mt = [0u32; N];
    mt[0] = s;
    for i in 1..N {
        // i < 624 fits in u32.
        #[allow(clippy::cast_possible_truncation)]
        let k = i as u32;
        mt[i] = 1_812_433_253u32
            .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
            .wrapping_add(k);
    }
    mt
}

/// CPython's `init_by_array`: the state `random.seed(n)` makes from the 32-bit words of `n`
/// (least significant first), or from `os.urandom` words for `seed(None)`.
pub fn init_by_array(key: &[u32]) -> SeededState {
    let key: &[u32] = if key.is_empty() { &[0] } else { key };
    let mut mt = init_genrand(19_650_218);
    let mut i = 1usize;
    let mut j = 0usize;
    let mut k = N.max(key.len());
    while k > 0 {
        // j < key.len(); CPython adds `(uint32_t)j`, so truncation is the reference behaviour.
        #[allow(clippy::cast_possible_truncation)]
        let jj = j as u32;
        mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(jj);
        i += 1;
        j += 1;
        if i >= N {
            mt[0] = mt[N - 1];
            i = 1;
        }
        if j >= key.len() {
            j = 0;
        }
        k -= 1;
    }
    k = N - 1;
    while k > 0 {
        #[allow(clippy::cast_possible_truncation)]
        let ii = i as u32;
        mt[i] =
            (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(ii);
        i += 1;
        if i >= N {
            mt[0] = mt[N - 1];
            i = 1;
        }
        k -= 1;
    }
    mt[0] = 0x8000_0000;
    mt
}

/// The 32-bit words of `|n|`, least significant first, as `random.seed(n)` splits an int.
pub fn int_key(n: u128) -> Vec<u32> {
    let mut key = Vec::with_capacity(4);
    let mut v = n;
    loop {
        // Low 32 bits of v.
        #[allow(clippy::cast_possible_truncation)]
        key.push(v as u32);
        v >>= 32;
        if v == 0 {
            break;
        }
    }
    key
}

/// Keys for generators seeded "from the operating system" in a batch: xoshiro256** seeded
/// with entropy once per batch, 624 words per document (CPython uses 624 words of
/// `os.urandom` for each `seed(None)`).
#[derive(Clone, Debug)]
pub struct EntropyKeys {
    s: [u64; 4],
}

impl EntropyKeys {
    /// A key stream from 32 bytes of entropy.
    pub fn new(seed: [u8; 32]) -> Self {
        let mut s = [0u64; 4];
        for (i, w) in s.iter_mut().enumerate() {
            let mut b = [0u8; 8];
            b.copy_from_slice(&seed[i * 8..i * 8 + 8]);
            *w = u64::from_le_bytes(b);
        }
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// The seeded state for the next document.
    pub fn next_state(&mut self) -> SeededState {
        let mut key = [0u32; N];
        for pair in key.chunks_mut(2) {
            let v = self.next_u64();
            // Low and high halves.
            #[allow(clippy::cast_possible_truncation)]
            {
                pair[0] = v as u32;
                pair[1] = (v >> 32) as u32;
            }
        }
        init_by_array(&key)
    }
}

impl Random {
    /// A generator in a seeded state (as right after `random.seed(...)`).
    pub fn from_seeded(state: &SeededState) -> Self {
        Self {
            state: *state,
            index: N,
            gauss_next: None,
        }
    }

    /// A generator in the state `random.Random.getstate()` describes: 624 words, the index
    /// (0..=624) and `gauss_next`. `None` if the index is out of range.
    pub fn from_state(words: &[u32; N], index: usize, gauss_next: Option<f64>) -> Option<Self> {
        (index <= N).then_some(Self {
            state: *words,
            index,
            gauss_next,
        })
    }

    /// One 32-bit word (`genrand_uint32`).
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    #[cold]
    fn twist(&mut self) {
        let mt = &mut self.state;
        let mag = |y: u32| if y & 1 == 0 { 0 } else { MATRIX_A };
        for kk in 0..N - M {
            let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag(y);
        }
        for kk in N - M..N - 1 {
            let y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
            mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ mag(y);
        }
        let y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag(y);
        self.index = 0;
    }

    /// `random.random()`: a float in [0, 1) from two words.
    #[inline]
    pub fn random(&mut self) -> f64 {
        let a = self.next_u32() >> 5;
        let b = self.next_u32() >> 6;
        (f64::from(a) * 67_108_864.0 + f64::from(b)) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// `random.getrandbits(k)` for `k <= 64` (words assembled least significant first, as
    /// CPython does on little-endian machines).
    #[inline]
    pub fn getrandbits(&mut self, k: u32) -> u64 {
        if k == 0 {
            return 0;
        }
        if k <= 32 {
            return u64::from(self.next_u32() >> (32 - k));
        }
        let mut out = 0u64;
        let mut left = k;
        let mut shift = 0;
        while left > 0 {
            let mut r = self.next_u32();
            if left < 32 {
                r >>= 32 - left;
            }
            out |= u64::from(r) << shift;
            shift += 32;
            left = left.saturating_sub(32);
        }
        out
    }

    /// `random._randbelow(n)` (`_randbelow_with_getrandbits`) for `n > 0`.
    #[inline]
    pub fn randbelow(&mut self, n: u64) -> u64 {
        let k = u64::BITS - n.leading_zeros();
        let mut r = self.getrandbits(k);
        while r >= n {
            r = self.getrandbits(k);
        }
        r
    }

    /// `random.gauss(mu, sigma)`.
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        let z = if let Some(z) = self.gauss_next.take() {
            z
        } else {
            let x2pi = self.random() * TWOPI;
            let g2rad = (-2.0 * libm_log(1.0 - self.random())).sqrt();
            let z = libm_cos(x2pi) * g2rad;
            self.gauss_next = Some(libm_sin(x2pi) * g2rad);
            z
        };
        mu + z * sigma
    }
}

// `math.log`, `math.cos` and `math.sin` call the C library's `log`, `cos` and `sin`; so do
// these. `black_box` keeps LLVM from fusing the cos and sin of one argument into one `sincos`
// call, whose last bit is not guaranteed to match.
#[inline(never)]
fn libm_log(x: f64) -> f64 {
    black_box(x).ln()
}

#[inline(never)]
fn libm_cos(x: f64) -> f64 {
    black_box(x).cos()
}

#[inline(never)]
fn libm_sin(x: f64) -> f64 {
    black_box(x).sin()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from CPython 3.12: `r = random.Random(0)`.
    #[test]
    fn seed_zero_matches_cpython() {
        let mut r = Random::from_seeded(&init_by_array(&int_key(0)));
        // random.Random(0).getrandbits(32) == 3626764237
        assert_eq!(r.next_u32(), 3_626_764_237);
        let mut r = Random::from_seeded(&init_by_array(&int_key(0)));
        // random.Random(0).random() == 0.8444218515250481
        assert_eq!(r.random().to_bits(), 0.844_421_851_525_048_1f64.to_bits());
    }

    #[test]
    fn big_int_seed_and_wide_getrandbits() {
        // random.Random(2**70 + 5).getrandbits(64) == 7192407739748854955
        let mut r = Random::from_seeded(&init_by_array(&int_key((1u128 << 70) + 5)));
        assert_eq!(r.getrandbits(64), 7_192_407_739_748_854_955);
    }

    #[test]
    fn int_keys_split_into_words() {
        assert_eq!(int_key(0), vec![0]);
        assert_eq!(int_key(1 << 32), vec![0, 1]);
        assert_eq!(int_key(u128::from(u32::MAX)), vec![u32::MAX]);
    }

    #[test]
    fn randbelow_stays_below() {
        let mut r = Random::from_seeded(&init_by_array(&[42]));
        for n in [1u64, 2, 3, 7, 1000, u64::MAX] {
            for _ in 0..100 {
                assert!(r.randbelow(n) < n);
            }
        }
    }
}
