//! Per-thread scratch buffers that outlive a document.
//!
//! Kernels keep their buffers in thread-locals, so a batch allocates nothing
//! per document. A buffer grows to the longest document its thread has seen,
//! and on a pool of many threads that capacity adds up: 64 threads that have
//! each seen a long document hold 64 copies of its buffers after the call.
//! Kernels give a buffer back once it has grown past [`KEEP_BYTES`].

/// Capacity a scratch buffer keeps between documents.
pub const KEEP_BYTES: usize = 1 << 20;

/// Has `v` grown past [`KEEP_BYTES`]?
pub fn too_big<T>(v: &Vec<T>) -> bool {
    v.capacity().saturating_mul(std::mem::size_of::<T>()) > KEEP_BYTES
}

/// Empty `v` and drop its capacity beyond [`KEEP_BYTES`], if it grew past it.
pub fn trim<T>(v: &mut Vec<T>) {
    if too_big(v) {
        v.clear();
        v.shrink_to(KEEP_BYTES / std::mem::size_of::<T>().max(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_keeps_small_buffers_and_shrinks_big_ones() {
        let mut small: Vec<u64> = Vec::with_capacity(16);
        small.push(1);
        trim(&mut small);
        assert_eq!(small, [1]);
        let mut big: Vec<u64> = vec![0; 1 << 20];
        assert!(too_big(&big));
        trim(&mut big);
        assert!(big.is_empty() && !too_big(&big));
    }
}
