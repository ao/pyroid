// Derived from packaging 26.3 (Apache-2.0 OR BSD-2-Clause), © Donald Stufft and individual
// contributors; modified.
//! Non-negative integers of any size, as packaging's `int(...)` of a digit run gives them.

use std::cmp::Ordering;
use std::fmt;

/// A non-negative integer: a `u64`, or the decimal digits (no leading zeros) of a larger one.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum Num {
    /// A value that fits in a `u64`.
    Small(u64),
    /// The digits of a value above `u64::MAX`, without leading zeros.
    Big(Box<str>),
}

impl Num {
    /// Zero.
    pub const ZERO: Num = Num::Small(0);

    /// `int(digits)` for a non-empty run of ASCII digits.
    #[must_use]
    pub fn from_digits(digits: &str) -> Num {
        let trimmed = digits.trim_start_matches('0');
        if trimmed.len() <= 19 {
            // At most 19 digits always fit in a u64.
            return Num::Small(trimmed.parse().unwrap_or(0));
        }
        match trimmed.parse::<u64>() {
            Ok(v) => Num::Small(v),
            Err(_) => Num::Big(trimmed.into()),
        }
    }

    /// The value, if it fits in a `u64`.
    #[must_use]
    pub fn small(&self) -> Option<u64> {
        match self {
            Num::Small(v) => Some(*v),
            Num::Big(_) => None,
        }
    }

    /// Is this zero?
    #[must_use]
    pub fn is_zero(&self) -> bool {
        matches!(self, Num::Small(0))
    }

    /// `self + 1`.
    #[must_use]
    pub fn plus_one(&self) -> Num {
        match self {
            Num::Small(v) => match v.checked_add(1) {
                Some(n) => Num::Small(n),
                None => Num::Big("18446744073709551616".into()),
            },
            Num::Big(d) => Num::Big(add_one(d).into()),
        }
    }

    /// `self - 1`, for a non-zero value.
    #[must_use]
    pub fn minus_one(&self) -> Num {
        match self {
            Num::Small(v) => Num::Small(v.saturating_sub(1)),
            Num::Big(d) => Num::from_digits(&sub_one(d)),
        }
    }

    /// The value as big-endian bytes, without leading zero bytes (`int.from_bytes(b, "big")`
    /// gives it back; unlike `int(digits)`, that is not subject to Python's limit on integer
    /// string conversion).
    #[must_use]
    pub fn to_be_bytes(&self) -> Vec<u8> {
        let digits = match self {
            Num::Small(v) => {
                let b = v.to_be_bytes();
                let first = b.iter().position(|&x| x != 0).unwrap_or(b.len());
                return b[first..].to_vec();
            }
            Num::Big(d) => d.as_bytes(),
        };
        // Little-endian base-2**32 limbs; each step multiplies by 10**k and adds k digits.
        let mut limbs: Vec<u32> = Vec::with_capacity(digits.len() / 9 + 1);
        for chunk in digits.chunks(9) {
            let mut mul: u64 = 1;
            let mut add: u64 = 0;
            for &b in chunk {
                mul *= 10;
                add = add * 10 + u64::from(b - b'0');
            }
            let mut carry = add;
            for limb in &mut limbs {
                let x = u64::from(*limb) * mul + carry;
                #[allow(clippy::cast_possible_truncation)]
                {
                    *limb = x as u32;
                }
                carry = x >> 32;
            }
            while carry > 0 {
                #[allow(clippy::cast_possible_truncation)]
                limbs.push(carry as u32);
                carry >>= 32;
            }
        }
        let mut out: Vec<u8> = limbs.iter().rev().flat_map(|l| l.to_be_bytes()).collect();
        let first = out.iter().position(|&x| x != 0).unwrap_or(out.len());
        out.drain(..first);
        out
    }

    /// The number of decimal digits.
    #[must_use]
    pub fn digit_count(&self) -> usize {
        match self {
            Num::Small(v) => v.checked_ilog10().map_or(1, |l| l as usize + 1),
            Num::Big(d) => d.len(),
        }
    }

    /// CPython's `hash(int(self))`: the value modulo the hash modulus (2**61 - 1 on 64-bit).
    #[must_use]
    pub fn py_hash(&self) -> i64 {
        match self {
            Num::Small(v) => {
                // The modulus is below 2**63.
                #[allow(clippy::cast_possible_wrap, clippy::cast_possible_truncation)]
                let h = (u128::from(*v) % u128::from(HASH_MODULUS)) as i64;
                h
            }
            Num::Big(d) => {
                let mut acc: u128 = 0;
                for b in d.bytes() {
                    acc = (acc * 10 + u128::from(b - b'0')) % u128::from(HASH_MODULUS);
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
                let h = acc as i64;
                h
            }
        }
    }
}

impl Ord for Num {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Num::Small(a), Num::Small(b)) => a.cmp(b),
            (Num::Small(_), Num::Big(_)) => Ordering::Less,
            (Num::Big(_), Num::Small(_)) => Ordering::Greater,
            (Num::Big(a), Num::Big(b)) => a.len().cmp(&b.len()).then_with(|| a.cmp(b)),
        }
    }
}

impl PartialOrd for Num {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for Num {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Num::Small(v) => write!(f, "{v}"),
            Num::Big(d) => f.write_str(d),
        }
    }
}

impl From<u64> for Num {
    fn from(v: u64) -> Self {
        Num::Small(v)
    }
}

/// CPython's `_PyHASH_MODULUS` (`sys.hash_info.modulus`).
#[cfg(target_pointer_width = "64")]
pub const HASH_MODULUS: u64 = (1 << 61) - 1;
/// CPython's `_PyHASH_MODULUS` (`sys.hash_info.modulus`).
#[cfg(not(target_pointer_width = "64"))]
pub const HASH_MODULUS: u64 = (1 << 31) - 1;

fn add_one(digits: &str) -> String {
    let mut out: Vec<u8> = digits.bytes().collect();
    let mut i = out.len();
    loop {
        if i == 0 {
            out.insert(0, b'1');
            break;
        }
        i -= 1;
        if out[i] == b'9' {
            out[i] = b'0';
        } else {
            out[i] += 1;
            break;
        }
    }
    String::from_utf8(out).unwrap_or_default()
}

fn sub_one(digits: &str) -> String {
    let mut out: Vec<u8> = digits.bytes().collect();
    let mut i = out.len();
    while i > 0 {
        i -= 1;
        if out[i] == b'0' {
            out[i] = b'9';
        } else {
            out[i] -= 1;
            break;
        }
    }
    String::from_utf8(out).unwrap_or_default()
}

/// CPython's tuple hash, one item at a time (no allocation).
#[derive(Clone, Copy, Debug)]
pub struct TupleHasher {
    acc: u64,
    len: u64,
}

#[cfg(target_pointer_width = "64")]
const XX: (u64, u64, u64, u32) = (
    11_400_714_785_074_694_791,
    14_029_467_366_897_019_727,
    2_870_177_450_012_600_261,
    31,
);
#[cfg(not(target_pointer_width = "64"))]
const XX: (u64, u64, u64, u32) = (2_654_435_761, 2_246_822_519, 374_761_393, 13);

impl TupleHasher {
    /// An empty tuple's state.
    #[must_use]
    pub fn new() -> Self {
        Self { acc: XX.2, len: 0 }
    }

    /// Add an item whose hash is `lane`.
    pub fn add(&mut self, lane: i64) {
        #[cfg(target_pointer_width = "64")]
        {
            #[allow(clippy::cast_sign_loss)]
            let lane = lane as u64;
            self.acc = self
                .acc
                .wrapping_add(lane.wrapping_mul(XX.1))
                .rotate_left(XX.3)
                .wrapping_mul(XX.0);
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let (lane, acc) = (lane as u32, self.acc as u32);
            #[allow(clippy::cast_possible_truncation)]
            let acc = acc
                .wrapping_add(lane.wrapping_mul(XX.1 as u32))
                .rotate_left(XX.3)
                .wrapping_mul(XX.0 as u32);
            self.acc = u64::from(acc);
        }
        self.len += 1;
    }

    /// The tuple's hash.
    #[must_use]
    pub fn finish(self) -> i64 {
        #[cfg(target_pointer_width = "64")]
        {
            let acc = self.acc.wrapping_add(self.len ^ (XX.2 ^ 0x0035_d373));
            if acc == u64::MAX {
                return 1_546_275_796;
            }
            #[allow(clippy::cast_possible_wrap)]
            let h = acc as i64;
            h
        }
        #[cfg(not(target_pointer_width = "64"))]
        {
            #[allow(clippy::cast_possible_truncation)]
            let acc =
                (self.acc as u32).wrapping_add((self.len as u32) ^ ((XX.2 as u32) ^ 0x0035_d373));
            if acc == u32::MAX {
                return 1_546_275_796;
            }
            #[allow(clippy::cast_possible_wrap)]
            let h = i64::from(acc as i32);
            h
        }
    }
}

impl Default for TupleHasher {
    fn default() -> Self {
        Self::new()
    }
}

/// CPython's hash of a tuple whose items hash to `lanes` (`tuplehash`, xxHash-based, 3.8+).
#[must_use]
pub fn tuple_hash(lanes: &[i64]) -> i64 {
    #[cfg(target_pointer_width = "64")]
    {
        const P1: u64 = 11_400_714_785_074_694_791;
        const P2: u64 = 14_029_467_366_897_019_727;
        const P5: u64 = 2_870_177_450_012_600_261;
        let mut acc = P5;
        for &lane in lanes {
            #[allow(clippy::cast_sign_loss)]
            let lane = lane as u64;
            acc = acc.wrapping_add(lane.wrapping_mul(P2));
            acc = acc.rotate_left(31);
            acc = acc.wrapping_mul(P1);
        }
        acc = acc.wrapping_add((lanes.len() as u64) ^ (P5 ^ 0x0035_d373));
        if acc == u64::MAX {
            return 1_546_275_796;
        }
        #[allow(clippy::cast_possible_wrap)]
        let h = acc as i64;
        h
    }
    #[cfg(not(target_pointer_width = "64"))]
    {
        const P1: u32 = 2_654_435_761;
        const P2: u32 = 2_246_822_519;
        const P5: u32 = 374_761_393;
        let mut acc = P5;
        for &lane in lanes {
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let lane = lane as u32;
            acc = acc.wrapping_add(lane.wrapping_mul(P2));
            acc = acc.rotate_left(13);
            acc = acc.wrapping_mul(P1);
        }
        #[allow(clippy::cast_possible_truncation)]
        let n = lanes.len() as u32;
        acc = acc.wrapping_add(n ^ (P5 ^ 0x0035_d373));
        if acc == u32::MAX {
            return 1_546_275_796;
        }
        #[allow(clippy::cast_possible_wrap)]
        let h = i64::from(acc as i32);
        h
    }
}

/// CPython's `hash(n)` for a small int (`-1` hashes as `-2`).
#[must_use]
pub fn int_hash(n: i64) -> i64 {
    #[allow(clippy::cast_possible_wrap)]
    let m = HASH_MODULUS as i64;
    // CPython hashes a negative int as -(|n| mod M); Rust's `%` truncates the same way.
    let h = n % m;
    if h == -1 { -2 } else { h }
}
