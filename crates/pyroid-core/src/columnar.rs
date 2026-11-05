//! Arrow-layout buffers, built without an Arrow dependency.
//!
//! The bindings export these through the Arrow C Data Interface. Layouts follow
//! the Arrow columnar format: validity and boolean values are LSB-first
//! bitmaps; string arrays are `len + 1` offsets into one UTF-8 data buffer.

use std::fmt;

use crate::par::{self, PoolError, Threads};

/// Rows per parallel copy task when filling a string data buffer.
const COPY_BLOCK_ROWS: usize = 2048;

/// Offset type of a string array: `i32` (Arrow `utf8`) or `i64` (`large_utf8`).
pub trait Offset: Copy + Send + Sync + fmt::Debug + PartialEq + 'static {
    /// Largest total byte length this offset type can address.
    const MAX: usize;
    /// Convert a byte position that is known to be `<= Self::MAX`.
    fn from_usize(v: usize) -> Self;
    /// Convert back to a byte position.
    fn to_usize(self) -> usize;
}

impl Offset for i32 {
    #[allow(clippy::cast_sign_loss)] // i32::MAX is positive
    const MAX: usize = i32::MAX as usize;
    fn from_usize(v: usize) -> Self {
        Self::try_from(v).expect("offset checked against Offset::MAX")
    }
    fn to_usize(self) -> usize {
        usize::try_from(self).expect("offsets are non-negative")
    }
}

impl Offset for i64 {
    #[allow(clippy::cast_possible_truncation)] // only reachable on <64-bit targets
    const MAX: usize = if usize::BITS >= 64 {
        i64::MAX as usize
    } else {
        usize::MAX
    };
    fn from_usize(v: usize) -> Self {
        Self::try_from(v).expect("offset checked against Offset::MAX")
    }
    fn to_usize(self) -> usize {
        usize::try_from(self).expect("offsets are non-negative")
    }
}

/// Building buffers failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BuildError {
    /// The strings don't fit the offset type (e.g. > 2 GiB in a `utf8` array).
    Overflow {
        /// Total bytes needed so far.
        bytes: usize,
        /// Largest total the offset type supports.
        max: usize,
    },
    /// The thread pool could not be started.
    Pool(PoolError),
}

impl fmt::Display for BuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow { bytes, max } => write!(
                f,
                "output needs more than {max} bytes in one Arrow string array (at least {bytes}); \
                 pass large_string input to get large_string output"
            ),
            Self::Pool(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for BuildError {}

impl From<PoolError> for BuildError {
    fn from(e: PoolError) -> Self {
        Self::Pool(e)
    }
}

/// Is bit `i` of an LSB-first bitmap set?
pub fn get_bit(bitmap: &[u8], i: usize) -> bool {
    bitmap[i / 8] & (1 << (i % 8)) != 0
}

/// Pack `len` booleans into an LSB-first bitmap.
pub fn bitmap(bits: impl IntoIterator<Item = bool>, len: usize) -> Vec<u8> {
    let mut out = vec![0u8; len.div_ceil(8)];
    for (i, bit) in bits.into_iter().enumerate().take(len) {
        if bit {
            out[i / 8] |= 1 << (i % 8);
        }
    }
    out
}

/// Buffers of an Arrow `utf8` (`O = i32`) or `large_utf8` (`O = i64`) array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StringBuffers<O> {
    /// Number of rows.
    pub len: usize,
    /// `len + 1` offsets; `offsets[0] == 0`.
    pub offsets: Vec<O>,
    /// Concatenated UTF-8 of every non-null row.
    pub data: Vec<u8>,
    /// Validity bitmap; `None` when there are no nulls.
    pub validity: Option<Vec<u8>>,
    /// Number of null rows.
    pub null_count: usize,
}

impl<O: Offset> StringBuffers<O> {
    /// Build from per-row values (`None` = null). The bytes are copied into one
    /// data buffer in parallel, per `threads`.
    pub fn build<S: AsRef<str> + Sync>(
        values: &[Option<S>],
        threads: Threads,
    ) -> Result<Self, BuildError> {
        let len = values.len();
        let mut offsets = Vec::with_capacity(len + 1);
        offsets.push(O::from_usize(0));
        let mut total = 0usize;
        let mut null_count = 0usize;
        for v in values {
            match v {
                Some(s) => total = total.saturating_add(s.as_ref().len()),
                None => null_count += 1,
            }
            if total > O::MAX {
                return Err(BuildError::Overflow {
                    bytes: total,
                    max: O::MAX,
                });
            }
            offsets.push(O::from_usize(total));
        }
        let validity = (null_count > 0).then(|| bitmap(values.iter().map(Option::is_some), len));

        // `vec![0; n]` is a calloc: large buffers come back as untouched zero
        // pages, which the parallel copy below faults in.
        let mut data = vec![0u8; total];
        let mut blocks = Vec::with_capacity(len.div_ceil(COPY_BLOCK_ROWS));
        let mut rest: &mut [u8] = &mut data;
        for (b, rows) in values.chunks(COPY_BLOCK_ROWS).enumerate() {
            let start = b * COPY_BLOCK_ROWS;
            let bytes = offsets[start + rows.len()].to_usize() - offsets[start].to_usize();
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(bytes);
            blocks.push((rows, head));
            rest = tail;
        }
        par::for_each(blocks, len, threads, |(rows, mut out)| {
            for s in rows.iter().flatten() {
                let src = s.as_ref().as_bytes();
                let (dst, tail) = std::mem::take(&mut out).split_at_mut(src.len());
                dst.copy_from_slice(src);
                out = tail;
            }
        })?;
        Ok(Self {
            len,
            offsets,
            data,
            validity,
            null_count,
        })
    }

    /// Row `i`, or `None` if it is null. Panics if `i >= len`.
    pub fn get(&self, i: usize) -> Option<&[u8]> {
        assert!(i < self.len, "row {i} out of range for {} rows", self.len);
        if self.validity.as_deref().is_some_and(|v| !get_bit(v, i)) {
            return None;
        }
        Some(&self.data[self.offsets[i].to_usize()..self.offsets[i + 1].to_usize()])
    }

    /// Concatenate several arrays into one.
    pub fn concat(parts: &[&Self]) -> Result<Self, BuildError> {
        let len = parts.iter().map(|p| p.len).sum();
        let total: usize = parts.iter().map(|p| p.data.len()).sum();
        if total > O::MAX {
            return Err(BuildError::Overflow {
                bytes: total,
                max: O::MAX,
            });
        }
        let mut offsets = Vec::with_capacity(len + 1);
        offsets.push(O::from_usize(0));
        let mut data = Vec::with_capacity(total);
        for p in parts {
            let base = data.len();
            offsets.extend(
                p.offsets[1..]
                    .iter()
                    .map(|o| O::from_usize(base + o.to_usize())),
            );
            data.extend_from_slice(&p.data);
        }
        let validity = concat_validity(parts.iter().map(|p| (p.validity.as_deref(), p.len)), len);
        Ok(Self {
            len,
            offsets,
            data,
            validity,
            null_count: parts.iter().map(|p| p.null_count).sum(),
        })
    }
}

/// Buffers of an Arrow `bool` array.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoolBuffers {
    /// Number of rows.
    pub len: usize,
    /// Values bitmap (bits of null rows are 0).
    pub values: Vec<u8>,
    /// Validity bitmap; `None` when there are no nulls.
    pub validity: Option<Vec<u8>>,
    /// Number of null rows.
    pub null_count: usize,
}

impl BoolBuffers {
    /// Build from per-row values (`None` = null).
    pub fn build(values: &[Option<bool>]) -> Self {
        let len = values.len();
        let null_count = values.iter().filter(|v| v.is_none()).count();
        Self {
            len,
            values: bitmap(values.iter().map(|v| *v == Some(true)), len),
            validity: (null_count > 0).then(|| bitmap(values.iter().map(Option::is_some), len)),
            null_count,
        }
    }

    /// Row `i`, or `None` if it is null. Panics if `i >= len`.
    pub fn get(&self, i: usize) -> Option<bool> {
        assert!(i < self.len, "row {i} out of range for {} rows", self.len);
        if self.validity.as_deref().is_some_and(|v| !get_bit(v, i)) {
            return None;
        }
        Some(get_bit(&self.values, i))
    }

    /// Concatenate several arrays into one.
    pub fn concat(parts: &[&Self]) -> Self {
        let len = parts.iter().map(|p| p.len).sum();
        let values = bitmap(
            parts
                .iter()
                .flat_map(|p| (0..p.len).map(|i| get_bit(&p.values, i))),
            len,
        );
        Self {
            len,
            values,
            validity: concat_validity(parts.iter().map(|p| (p.validity.as_deref(), p.len)), len),
            null_count: parts.iter().map(|p| p.null_count).sum(),
        }
    }
}

/// Concatenate validity bitmaps; `None` (all valid) if no part has one.
fn concat_validity<'a>(
    parts: impl Iterator<Item = (Option<&'a [u8]>, usize)> + Clone,
    len: usize,
) -> Option<Vec<u8>> {
    if parts.clone().all(|(v, _)| v.is_none()) {
        return None;
    }
    let bits = parts.flat_map(|(v, n)| (0..n).map(move |i| v.is_none_or(|v| get_bit(v, i))));
    Some(bitmap(bits, len))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(b: &StringBuffers<i32>) -> Vec<Option<String>> {
        (0..b.len)
            .map(|i| b.get(i).map(|s| String::from_utf8(s.to_vec()).unwrap()))
            .collect()
    }

    #[test]
    fn bitmaps() {
        let b = bitmap(
            [true, false, true, true, false, false, false, false, true],
            9,
        );
        assert_eq!(b, vec![0b0000_1101, 0b0000_0001]);
        assert!(get_bit(&b, 8));
        assert!(!get_bit(&b, 1));
        assert_eq!(bitmap([], 0), Vec::<u8>::new());
    }

    #[test]
    fn strings_without_nulls() {
        let v = [Some("a"), Some(""), Some("héllo")];
        let b = StringBuffers::<i32>::build(&v, Threads::Auto).unwrap();
        assert_eq!(b.offsets, vec![0, 1, 1, 7]);
        assert_eq!(b.data, "ahéllo".as_bytes());
        assert_eq!(b.validity, None);
        assert_eq!(b.null_count, 0);
    }

    #[test]
    fn strings_with_nulls() {
        let v = [None, Some("xy"), None];
        let b = StringBuffers::<i64>::build(&v, Threads::Auto).unwrap();
        assert_eq!(b.offsets, vec![0, 0, 2, 2]);
        assert_eq!(b.validity, Some(vec![0b010]));
        assert_eq!(b.null_count, 2);
        assert_eq!(b.get(1), Some(&b"xy"[..]));
        assert_eq!(b.get(0), None);
    }

    #[test]
    fn many_rows_parallel_copy_matches_serial() {
        let owned: Vec<Option<String>> = (0..20_000)
            .map(|i| (i % 7 != 0).then(|| "ü".repeat(i % 13)))
            .collect();
        let par = StringBuffers::<i32>::build(&owned, Threads::Auto).unwrap();
        let ser = StringBuffers::<i32>::build(&owned, Threads::ONE).unwrap();
        assert_eq!(par, ser);
        assert_eq!(rows(&par), owned);
    }

    #[test]
    fn empty_strings_array() {
        let b = StringBuffers::<i32>::build::<&str>(&[], Threads::Auto).unwrap();
        assert_eq!(b.offsets, vec![0]);
        assert_eq!(b.data, Vec::<u8>::new());
    }

    #[test]
    fn concat_strings() {
        let a = StringBuffers::<i32>::build(&[Some("ab"), None], Threads::ONE).unwrap();
        let b = StringBuffers::<i32>::build(&[Some("c")], Threads::ONE).unwrap();
        let c = StringBuffers::concat(&[&a, &b]).unwrap();
        assert_eq!(c.offsets, vec![0, 2, 2, 3]);
        assert_eq!(c.validity, Some(vec![0b101]));
        assert_eq!(c.null_count, 1);
        let none = StringBuffers::concat(&[&b, &b]).unwrap();
        assert_eq!(none.validity, None);
    }

    #[test]
    fn bools() {
        let b = BoolBuffers::build(&[Some(true), None, Some(false), Some(true)]);
        assert_eq!(b.values, vec![0b1001]);
        assert_eq!(b.validity, Some(vec![0b1101]));
        assert_eq!(b.null_count, 1);
        assert_eq!(b.get(1), None);
        let c = BoolBuffers::concat(&[&b, &BoolBuffers::build(&[Some(true)])]);
        assert_eq!(c.len, 5);
        assert_eq!(c.get(4), Some(true));
        assert_eq!(c.get(1), None);
        assert_eq!(c.validity, Some(vec![0b1_1101]));
    }

    #[test]
    fn overflow_is_an_error() {
        // Fake a tiny offset type to exercise the overflow path cheaply.
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Tiny(u8);
        impl Offset for Tiny {
            const MAX: usize = 4;
            fn from_usize(v: usize) -> Self {
                Self(u8::try_from(v).unwrap())
            }
            fn to_usize(self) -> usize {
                usize::from(self.0)
            }
        }
        let err =
            StringBuffers::<Tiny>::build(&[Some("abc"), Some("de")], Threads::ONE).unwrap_err();
        assert_eq!(err, BuildError::Overflow { bytes: 5, max: 4 });
        assert!(err.to_string().contains("large_string"));
    }
}
