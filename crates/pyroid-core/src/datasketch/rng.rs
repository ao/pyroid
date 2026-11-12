// Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
//! The permutation parameters, drawn exactly as datasketch draws them with
//! numpy's legacy `np.random.RandomState(seed)`.
//!
//! `RandomState` is MT19937 seeded with `init_genrand` (`mt19937_seed`);
//! 64-bit draws are two 32-bit outputs, high word first (`mt19937_next64`);
//! bounded integers use masked rejection sampling, with 32-bit draws when
//! the range fits 32 bits (numpy's `random_bounded_uint{32,64}_fill` with
//! `use_masked=True`). The same generator backs `pyroid.dedup`
//! (`crate::dedup`), which draws datatrove's parameters in another order.
//!
//! datasketch's draws per scheme (`MinHash._init_permutations`):
//!
//! ```python
//! gen = np.random.RandomState(seed)
//! # "legacy": one (a, b) pair per permutation, as scalars
//! [(gen.randint(1, _mersenne_prime, dtype=np.uint64),
//!   gen.randint(0, _mersenne_prime, dtype=np.uint64)) for _ in range(num_perm)]
//! # "affine32" / "affine64" (w = 32 / 64): two arrays
//! a = gen.randint(0, 1 << (w - 1), num_perm, dtype=dtype) * 2 + 1
//! b = gen.randint(0, 1 << w, num_perm, dtype=dtype)
//! ```

use super::minhash::{MERSENNE_PRIME, Permutations};

const N: usize = 624;
const M: usize = 397;

/// MT19937 as numpy's legacy seeding sets it up.
#[derive(Clone, Debug)]
pub struct Mt19937 {
    state: [u32; N],
    pos: usize,
}

impl Mt19937 {
    /// `mt19937_seed` (`init_genrand`): what `RandomState(seed)` does for an
    /// integer seed in `0..2**32`.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            // i < 624, so the cast is exact.
            #[allow(clippy::cast_possible_truncation)]
            let i32_ = i as u32;
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i32_);
        }
        Self { state, pos: N }
    }

    fn refill(&mut self) {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        for i in 0..N {
            let y = (self.state[i] & UPPER) | (self.state[(i + 1) % N] & LOWER);
            let mag = if y & 1 == 0 { 0 } else { MATRIX_A };
            self.state[i] = self.state[(i + M) % N] ^ (y >> 1) ^ mag;
        }
        self.pos = 0;
    }

    /// `mt19937_next32`.
    pub fn next_u32(&mut self) -> u32 {
        if self.pos >= N {
            self.refill();
        }
        let mut y = self.state[self.pos];
        self.pos += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `mt19937_next64`: the first 32-bit output is the high word.
    pub fn next_u64(&mut self) -> u64 {
        let hi = u64::from(self.next_u32());
        let lo = u64::from(self.next_u32());
        (hi << 32) | lo
    }

    /// One `RandomState.randint(low, high, dtype=np.uint64)` (`high`
    /// exclusive, `low < high`): `random_bounded_uint64_fill` with one value.
    pub fn randint_u64(&mut self, low: u64, high: u64) -> u64 {
        debug_assert!(low < high);
        let rng = high - 1 - low;
        if rng == 0 {
            return low;
        }
        if let Ok(rng32) = u32::try_from(rng) {
            // numpy draws 32-bit values when the range fits in 32 bits.
            return low + u64::from(self.bounded_u32(rng32));
        }
        if rng == u64::MAX {
            return low.wrapping_add(self.next_u64());
        }
        let mask = gen_mask(rng);
        loop {
            let v = self.next_u64() & mask;
            if v <= rng {
                return low + v;
            }
        }
    }

    /// One bounded 32-bit draw in `0..=rng` (`buffered_bounded_masked_uint32`,
    /// or a plain draw when `rng` is `u32::MAX`).
    fn bounded_u32(&mut self, rng: u32) -> u32 {
        if rng == u32::MAX {
            return self.next_u32();
        }
        // gen_mask(rng) < 2**32 here.
        #[allow(clippy::cast_possible_truncation)]
        let mask = gen_mask(u64::from(rng)) as u32;
        loop {
            let v = self.next_u32() & mask;
            if v <= rng {
                return v;
            }
        }
    }
}

/// The smallest `2**k - 1 >= max` (numpy's `gen_mask`).
fn gen_mask(max: u64) -> u64 {
    if max == 0 {
        0
    } else {
        u64::MAX >> max.leading_zeros()
    }
}

/// datasketch's permutation parameters for `scheme`, `seed` and `num_perm`.
pub fn permutations(scheme: super::minhash::Scheme, seed: u32, num_perm: usize) -> Permutations {
    use super::minhash::Scheme;
    let mut rng = Mt19937::new(seed);
    match scheme {
        Scheme::Legacy => {
            let mut a = Vec::with_capacity(num_perm);
            let mut b = Vec::with_capacity(num_perm);
            for _ in 0..num_perm {
                a.push(rng.randint_u64(1, MERSENNE_PRIME));
                b.push(rng.randint_u64(0, MERSENNE_PRIME));
            }
            Permutations::Legacy { a, b }
        }
        Scheme::Affine32 => {
            // randint(0, 2**31, n, uint32) * 2 + 1, then randint(0, 2**32, n, uint32).
            let a = (0..num_perm)
                .map(|_| rng.bounded_u32((1 << 31) - 1) * 2 + 1)
                .collect();
            let b = (0..num_perm).map(|_| rng.next_u32()).collect();
            Permutations::Affine32 { a, b }
        }
        Scheme::Affine64 => {
            // randint(0, 2**63, n, uint64) * 2 + 1, then randint(0, 2**64, n, uint64).
            let a = (0..num_perm)
                .map(|_| rng.randint_u64(0, 1 << 63).wrapping_mul(2).wrapping_add(1))
                .collect();
            let b = (0..num_perm).map(|_| rng.next_u64()).collect();
            Permutations::Affine64 { a, b }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::minhash::Scheme;
    use super::*;

    #[test]
    fn mt19937_reference_output() {
        // The reference implementation's init_genrand(5489) outputs.
        let mut mt = Mt19937::new(5489);
        let first: Vec<u32> = (0..3).map(|_| mt.next_u32()).collect();
        assert_eq!(first, [3_499_211_612, 581_869_302, 3_890_346_734]);
        let mut mt = Mt19937::new(5489);
        let v = (0..10_000).map(|_| mt.next_u32()).last().unwrap();
        assert_eq!(v, 4_123_659_995);
    }

    #[test]
    fn masks() {
        assert_eq!(gen_mask(1), 1);
        assert_eq!(gen_mask(2), 3);
        assert_eq!(gen_mask((1 << 61) - 3), (1 << 61) - 1);
        assert_eq!(gen_mask(u64::MAX), u64::MAX);
    }

    #[test]
    fn numpy_values() {
        // From numpy 2.5.4, as datasketch 2.0.0 draws them:
        //   MinHash(num_perm=4, seed=1, scheme=...).permutations
        let Permutations::Legacy { a, b } = permutations(Scheme::Legacy, 1, 2) else {
            unreachable!()
        };
        assert_eq!(a, [775_169_054_918_279_404, 2_109_959_069_025_162]);
        assert_eq!(b, [1_758_426_461_858_698_312, 965_365_488_286_768_773]);
        let Permutations::Affine32 { a, b } = permutations(Scheme::Affine32, 1, 3) else {
            unreachable!()
        };
        assert_eq!(a, [3_582_191_691, 4_270_784_983, 1_892_572_953]);
        assert_eq!(b, [4_005_303_368, 491_263, 550_290_313]);
        let Permutations::Affine64 { a, b } = permutations(Scheme::Affine64, 1, 2) else {
            unreachable!()
        };
        assert_eq!(a, [15_385_396_165_118_722_519, 8_128_538_942_144_784_529]);
        assert_eq!(b, [2_109_959_069_025_161, 5_577_051_506_714_156_677]);
    }
}
