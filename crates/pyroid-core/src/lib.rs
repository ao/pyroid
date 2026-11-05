//! Pure-Rust kernels behind pyroid.
//!
//! This crate has no Python dependency: everything here is plain Rust that is
//! unit-tested with `cargo test`. The `pyroid-py` crate wraps it for Python.
#![forbid(unsafe_code)]

/// Length of `text` in UTF-8 bytes.
///
/// **Placeholder.** It exists only to prove the core → bindings → wheel
/// pipeline end to end, and will be removed once real kernels land.
///
/// ```
/// assert_eq!(pyroid_core::utf8_len("héllo"), 6);
/// ```
#[must_use]
pub fn utf8_len(text: &str) -> usize {
    text.len()
}

#[cfg(test)]
mod tests {
    use super::utf8_len;

    #[test]
    fn empty() {
        assert_eq!(utf8_len(""), 0);
    }

    #[test]
    fn ascii() {
        assert_eq!(utf8_len("pyroid"), 6);
    }

    #[test]
    fn multibyte() {
        // 1-, 2-, 3- and 4-byte UTF-8 sequences.
        assert_eq!(utf8_len("a"), 1);
        assert_eq!(utf8_len("é"), 2);
        assert_eq!(utf8_len("€"), 3);
        assert_eq!(utf8_len("😀"), 4);
        assert_eq!(utf8_len("aé€😀"), 10);
    }

    #[test]
    fn counts_bytes_not_chars() {
        let s = "naïve café";
        assert_eq!(s.chars().count(), 10);
        assert_eq!(utf8_len(s), 12);
    }

    #[test]
    fn nul_and_controls() {
        assert_eq!(utf8_len("\0\r\n\u{7f}"), 4);
        assert_eq!(utf8_len("\u{85}"), 2);
    }
}
