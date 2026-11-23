// Derived from json_repair 0.64.9 (MIT), © Stefano Baccianella; modified.
//! The parser's text buffer: json_repair's `self.json_str`, as chars.
//!
//! json_repair rewrites its input in two places: it inserts a `{` where a
//! duplicate key starts a new object, and it replaces an escaped object with
//! its unescaped form. Rebuilding the whole string each time is quadratic
//! (`[{"id": 0 "id": 1 ...}]` splits once per record), so the buffer is a gap
//! buffer: it borrows the input until the first edit, then keeps a gap where
//! the last edit happened. The edits happen at the parse position, which only
//! moves forward, so the gap moves O(n) in total.

use std::borrow::Cow;

#[derive(Debug)]
pub(crate) struct Buf<'a> {
    data: Cow<'a, [char]>,
    /// Logical index of the gap; `data[gap_start..gap_start + gap_len]` is unused.
    gap_start: usize,
    gap_len: usize,
}

impl<'a> Buf<'a> {
    pub(crate) fn new(s: &'a [char]) -> Self {
        Self {
            data: Cow::Borrowed(s),
            gap_start: s.len(),
            gap_len: 0,
        }
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.data.len() - self.gap_len
    }

    /// The char at logical index `i < len()`.
    #[inline]
    pub(crate) fn at(&self, i: usize) -> char {
        if i < self.gap_start {
            self.data[i]
        } else {
            self.data[i + self.gap_len]
        }
    }

    #[inline]
    pub(crate) fn get(&self, i: usize) -> Option<char> {
        (i < self.len()).then(|| self.at(i))
    }

    /// `[a, b)` (clamped); borrowed unless it straddles the gap.
    pub(crate) fn slice(&self, a: usize, b: usize) -> Cow<'_, [char]> {
        let b = b.min(self.len());
        if a >= b {
            return Cow::Borrowed(&[]);
        }
        if b <= self.gap_start {
            Cow::Borrowed(&self.data[a..b])
        } else if a >= self.gap_start {
            Cow::Borrowed(&self.data[a + self.gap_len..b + self.gap_len])
        } else {
            let mut v = Vec::with_capacity(b - a);
            v.extend_from_slice(&self.data[a..self.gap_start]);
            v.extend_from_slice(&self.data[self.gap_start + self.gap_len..b + self.gap_len]);
            Cow::Owned(v)
        }
    }

    /// Every char, in order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = char> + '_ {
        self.data[..self.gap_start]
            .iter()
            .chain(&self.data[self.gap_start + self.gap_len..])
            .copied()
    }

    /// The whole text as one slice (closes the gap if there is one).
    pub(crate) fn contiguous(&mut self) -> &[char] {
        if self.gap_len > 0 {
            self.move_gap(self.len());
        }
        let n = self.len();
        &self.data[..n]
    }

    fn move_gap(&mut self, to: usize) {
        let gap_len = self.gap_len;
        let gap_start = self.gap_start;
        let data = self.data.to_mut();
        if to < gap_start {
            data.copy_within(to..gap_start, to + gap_len);
        } else if to > gap_start {
            data.copy_within(gap_start + gap_len..to + gap_len, gap_start);
        }
        self.gap_start = to;
    }

    /// Replace logical `[a, b)` with `with`.
    pub(crate) fn splice(&mut self, a: usize, b: usize, with: &[char]) {
        let b = b.min(self.len());
        let a = a.min(b);
        self.move_gap(b);
        // Delete [a, b): widen the gap to the left.
        self.gap_len += b - a;
        self.gap_start = a;
        if with.len() > self.gap_len {
            let grow = (with.len() - self.gap_len).max(self.len() / 8).max(64);
            let data = self.data.to_mut();
            let tail = data.len() - (self.gap_start + self.gap_len);
            data.splice(
                self.gap_start..self.gap_start,
                std::iter::repeat_n('\0', grow),
            );
            self.gap_len += grow;
            debug_assert_eq!(data.len() - (self.gap_start + self.gap_len), tail);
        }
        let data = self.data.to_mut();
        data[self.gap_start..self.gap_start + with.len()].copy_from_slice(with);
        self.gap_start += with.len();
        self.gap_len -= with.len();
    }
}

#[cfg(test)]
mod tests {
    use super::Buf;

    fn text(b: &Buf<'_>) -> String {
        b.iter().collect()
    }

    #[test]
    fn edits_and_reads() {
        let src: Vec<char> = "abcdefgh".chars().collect();
        let mut b = Buf::new(&src);
        b.splice(3, 3, &['{']);
        assert_eq!(text(&b), "abc{defgh");
        b.splice(6, 6, &['{']);
        assert_eq!(text(&b), "abc{de{fgh");
        b.splice(1, 1, &['X']);
        assert_eq!(text(&b), "aXbc{de{fgh");
        b.splice(2, 5, &['Y', 'Z']);
        assert_eq!(text(&b), "aXYZde{fgh");
        assert_eq!(b.len(), 10);
        assert_eq!(b.at(6), '{');
        assert_eq!(b.get(10), None);
        assert_eq!(&*b.slice(1, 8), &['X', 'Y', 'Z', 'd', 'e', '{', 'f']);
        assert_eq!(b.contiguous().iter().collect::<String>(), "aXYZde{fgh");
        for i in 0..200 {
            let at = b.len() - (i % 3);
            b.splice(at, at, &['{']);
        }
        assert_eq!(b.len(), 210);
        assert_eq!(src.iter().collect::<String>(), "abcdefgh");
    }
}
