// Derived from datatrove 0.10.1 (Apache-2.0), © HuggingFace Inc.; modified.
//! The permutation parameters, drawn exactly as datatrove draws them:
//!
//! ```python
//! gen = np.random.RandomState(seed)
//! a = gen.randint(1, _mersenne_prime, dtype=np.uint64, size=(1, num_hashes))
//! b = gen.randint(0, _mersenne_prime, dtype=np.uint64, size=(1, num_hashes))
//! ```
//!
//! That is numpy's legacy `RandomState`: MT19937 seeded with `init_genrand`,
//! 64-bit draws made of two 32-bit outputs (high word first), and bounded
//! integers by masked rejection sampling (numpy's
//! `random_bounded_uint64_fill(..., use_masked=True)`).

/// MT19937 (Matsumoto & Nishimura), as numpy's legacy seeding sets it up.
struct Mt19937 {
    state: [u32; N],
    pos: usize,
}

const N: usize = 624;
const M: usize = 397;

impl Mt19937 {
    /// `mt19937_seed` / `init_genrand`.
    fn new(seed: u32) -> Self {
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

    fn next_u32(&mut self) -> u32 {
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

    /// numpy's `mt19937_next64`: the first 32-bit output is the high word.
    fn next_u64(&mut self) -> u64 {
        let hi = u64::from(self.next_u32());
        let lo = u64::from(self.next_u32());
        (hi << 32) | lo
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

/// `RandomState.randint(low, high, size=n, dtype=np.uint64)` (`high` exclusive).
fn randint_u64(rng: &mut Mt19937, low: u64, high: u64, n: usize) -> Vec<u64> {
    debug_assert!(low < high);
    let range = high - 1 - low;
    if range == 0 {
        return vec![low; n];
    }
    let mask = gen_mask(range);
    if u32::try_from(range).is_ok() {
        // numpy draws 32-bit values for ranges that fit in 32 bits.
        if range == u64::from(u32::MAX) {
            return (0..n).map(|_| low + u64::from(rng.next_u32())).collect();
        }
        return (0..n)
            .map(|_| {
                loop {
                    let v = u64::from(rng.next_u32()) & mask;
                    if v <= range {
                        break low + v;
                    }
                }
            })
            .collect();
    }
    if range == u64::MAX {
        return (0..n).map(|_| low.wrapping_add(rng.next_u64())).collect();
    }
    (0..n)
        .map(|_| {
            loop {
                let v = rng.next_u64() & mask;
                if v <= range {
                    break low + v;
                }
            }
        })
        .collect()
}

/// `2**61 - 1`, datatrove's `_mersenne_prime`.
pub const MERSENNE_PRIME: u64 = (1 << 61) - 1;

/// datatrove's `MinhashDedupSignature.parameters`: `(a, b)`, `num_hashes` each.
pub fn permutation_parameters(seed: u32, num_hashes: usize) -> (Vec<u64>, Vec<u64>) {
    let mut rng = Mt19937::new(seed);
    let a = randint_u64(&mut rng, 1, MERSENNE_PRIME, num_hashes);
    let b = randint_u64(&mut rng, 0, MERSENNE_PRIME, num_hashes);
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mt19937_reference_output() {
        // The reference implementation's init_genrand(5489) first outputs.
        let mut mt = Mt19937::new(5489);
        let first: Vec<u32> = (0..3).map(|_| mt.next_u32()).collect();
        assert_eq!(first, [3_499_211_612, 581_869_302, 3_890_346_734]);
        // 10000th output for seed 5489 (C++ std::mt19937 conformance value).
        let mut mt = Mt19937::new(5489);
        let v = (0..10_000).map(|_| mt.next_u32()).last().unwrap();
        assert_eq!(v, 4_123_659_995);
    }

    #[test]
    fn masks() {
        assert_eq!(gen_mask(1), 1);
        assert_eq!(gen_mask(2), 3);
        assert_eq!(gen_mask(MERSENNE_PRIME - 2), MERSENNE_PRIME);
        assert_eq!(gen_mask(u64::MAX), u64::MAX);
    }

    #[test]
    fn parameters_in_range() {
        let (a, b) = permutation_parameters(1, 112);
        assert_eq!((a.len(), b.len()), (112, 112));
        assert!(a.iter().all(|&x| (1..MERSENNE_PRIME).contains(&x)));
        assert!(b.iter().all(|&x| x < MERSENNE_PRIME));
    }
}
