// Derived from CPython's Python/formatter_unicode.c and Python/pystrtod.c (PSF-2.0); modified.
//! `format(x, spec)` for `float` and `int`, and `repr(float)`, as CPython prints them.
//!
//! [`Spec::parse`] accepts the part of the format-spec mini-language that has meant the same
//! in every CPython since 3.11: `[[fill]align][sign][z][#][0][width][,|_][.precision][type]`
//! with float types `e E f F g G %` (or none) and int type `d` (or none). Anything else
//! (`n`, `x`, grouping with zero padding, locale, out-of-range widths) gives `None`: the
//! caller asks Python, which also raises its own errors for invalid specs.
//!
//! Digits come from Rust's formatter, which rounds the exact binary value correctly with ties
//! to even, as CPython's dtoa does: `{:.N}` for `f`, `{:.Ne}` for `e` and for `g`'s
//! significant digits, `{:e}` for the shortest round-tripping digits of `repr`.

use std::fmt::Write as _;

/// Widths and precisions above this are left to Python.
const MAX_WIDTH: usize = 1 << 16;

/// A parsed format spec.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Spec {
    fill: char,
    /// `<`, `>`, `=`, `^`, or none (numbers: `>`).
    align: Option<char>,
    /// `-` (default), `+` or space.
    sign: char,
    no_neg_zero: bool,
    alt: bool,
    width: usize,
    grouping: Option<char>,
    precision: Option<usize>,
    ty: Option<char>,
}

fn is_align(c: char) -> bool {
    matches!(c, '<' | '>' | '=' | '^')
}

impl Spec {
    /// Parse `spec`; `None` if it is outside the supported subset.
    pub fn parse(spec: &str, for_int: bool) -> Option<Self> {
        let chars: Vec<char> = spec.chars().collect();
        let mut pos = 0;
        let mut fill = ' ';
        let mut fill_given = false;
        let mut align = None;
        if chars.len() >= 2 && is_align(chars[1]) {
            fill = chars[0];
            fill_given = true;
            align = Some(chars[1]);
            pos = 2;
        } else if !chars.is_empty() && is_align(chars[0]) {
            align = Some(chars[0]);
            pos = 1;
        }
        let mut sign = '-';
        if let Some(&c @ ('+' | '-' | ' ')) = chars.get(pos) {
            sign = c;
            pos += 1;
        }
        let mut no_neg_zero = false;
        if chars.get(pos) == Some(&'z') {
            no_neg_zero = true;
            pos += 1;
        }
        let mut alt = false;
        if chars.get(pos) == Some(&'#') {
            alt = true;
            pos += 1;
        }
        let mut zero = false;
        if !fill_given && chars.get(pos) == Some(&'0') {
            zero = true;
            fill = '0';
            if align.is_none() {
                align = Some('=');
            }
            pos += 1;
        }
        let digits = |pos: &mut usize| -> Option<Option<usize>> {
            let start = *pos;
            let mut n: usize = 0;
            while let Some(d) = chars.get(*pos).and_then(|c| c.to_digit(10)) {
                n = n.checked_mul(10)?.checked_add(d as usize)?;
                if n > MAX_WIDTH {
                    return None;
                }
                *pos += 1;
            }
            Some((*pos > start).then_some(n))
        };
        let width = digits(&mut pos)?.unwrap_or(0);
        let mut grouping = None;
        if let Some(&c @ (',' | '_')) = chars.get(pos) {
            grouping = Some(c);
            pos += 1;
            if matches!(chars.get(pos), Some(',' | '_')) {
                return None;
            }
        }
        let mut precision = None;
        if chars.get(pos) == Some(&'.') {
            pos += 1;
            precision = Some(digits(&mut pos)??);
        }
        let ty = match chars.len() - pos {
            0 => None,
            1 => Some(chars[pos]),
            _ => return None,
        };
        // Zero padding with grouping puts separators into the padding: left to Python.
        if grouping.is_some() && (zero || align == Some('=')) {
            return None;
        }
        if for_int {
            if no_neg_zero || precision.is_some() || !matches!(ty, None | Some('d')) {
                return None;
            }
        } else if !matches!(ty, None | Some('e' | 'E' | 'f' | 'F' | 'g' | 'G' | '%')) {
            return None;
        }
        Some(Self {
            fill,
            align,
            sign,
            no_neg_zero,
            alt,
            width,
            grouping,
            precision,
            ty,
        })
    }

    /// `format(x, spec)` for a float.
    #[allow(clippy::many_single_char_names)] // CPython's names
    pub fn format_float(&self, x: f64) -> String {
        let mut out = String::with_capacity(24);
        self.format_float_into(x, &mut out);
        out
    }

    /// [`Spec::format_float`], appended to `out`.
    pub fn format_float_into(&self, x: f64, out: &mut String) {
        let start = out.len();
        let body = out;
        let upper = matches!(self.ty, Some('E' | 'F' | 'G'));
        let mut neg = x.is_sign_negative() && !x.is_nan();
        let mut x = x;
        let pct = self.ty == Some('%');
        if pct {
            x *= 100.0;
            neg = x.is_sign_negative() && !x.is_nan();
        }
        let finite = x.is_finite();
        if x.is_nan() {
            body.push_str("nan");
        } else if x.is_infinite() {
            body.push_str("inf");
        } else {
            let a = x.abs();
            match self.ty {
                Some('f' | 'F' | '%') => {
                    let p = self.precision.unwrap_or(6);
                    if !fixed_fast(body, a, p) {
                        let _ = write!(body, "{a:.p$}");
                    }
                    if self.alt && p == 0 {
                        body.push('.');
                    }
                }
                Some('e' | 'E') => {
                    let p = self.precision.unwrap_or(6);
                    let _ = write!(body, "{a:.p$e}");
                    let at = start + body[start..].find('e').expect("exponent form");
                    let exp: i64 = body[at + 1..].parse().expect("exponent");
                    body.truncate(at);
                    if self.alt && p == 0 {
                        body.push('.');
                    }
                    push_exponent(body, exp, 'e');
                }
                Some('g' | 'G') => {
                    let p = self.precision.unwrap_or(6).max(1);
                    if a == 0.0 || self.alt || !g_fast(body, a, p, false) {
                        let (digits, decpt) = significant(a, p);
                        build(body, digits.as_slice(), decpt, Code::G, p, false, self.alt);
                    }
                }
                None => match self.precision {
                    None => {
                        let (digits, decpt) = shortest(a);
                        build(body, digits.as_slice(), decpt, Code::R, 0, true, self.alt);
                    }
                    Some(p) => {
                        let p = p.max(1);
                        if a == 0.0 || self.alt || !g_fast(body, a, p, true) {
                            let (digits, decpt) = significant(a, p);
                            build(body, digits.as_slice(), decpt, Code::G, p, true, self.alt);
                        }
                    }
                },
                Some(_) => unreachable!("rejected by parse"),
            }
            if self.no_neg_zero && !body[start..].bytes().any(|b| (b'1'..=b'9').contains(&b)) {
                neg = false;
            }
        }
        if pct {
            body.push('%');
        }
        if upper {
            body[start..].make_ascii_uppercase();
        }
        if finite {
            if let Some(sep) = self.grouping {
                let grouped = group(&body[start..], sep);
                body.truncate(start);
                body.push_str(&grouped);
            }
        }
        self.pad_into(body, start, neg);
    }

    /// `format(v, spec)` for an int.
    pub fn format_int(&self, v: i128) -> String {
        let mut out = String::with_capacity(24);
        self.format_int_into(v, &mut out);
        out
    }

    /// [`Spec::format_int`], appended to `out`.
    pub fn format_int_into(&self, v: i128, out: &mut String) {
        let start = out.len();
        match u64::try_from(v.unsigned_abs()) {
            Ok(n) => push_u64(out, n, 0),
            Err(_) => {
                let _ = write!(out, "{}", v.unsigned_abs());
            }
        }
        if let Some(sep) = self.grouping {
            let grouped = group(&out[start..], sep);
            out.truncate(start);
            out.push_str(&grouped);
        }
        self.pad_into(out, start, v < 0);
    }

    /// Sign and padding around the body at `out[start..]`.
    fn pad_into(&self, out: &mut String, start: usize, neg: bool) {
        if self.width == 0 {
            if neg {
                out.insert(start, '-');
            } else if self.sign == '+' || self.sign == ' ' {
                out.insert(start, self.sign);
            }
            return;
        }
        let body = out.split_off(start);
        let padded = self.pad(neg, body);
        out.push_str(&padded);
    }

    fn pad(&self, neg: bool, mut body: String) -> String {
        let sign = if neg {
            "-"
        } else {
            match self.sign {
                '+' => "+",
                ' ' => " ",
                _ => "",
            }
        };
        let len = sign.len() + body.chars().count();
        if self.width <= len {
            body.insert_str(0, sign);
            return body;
        }
        let body = body.as_str();
        let mut out = String::with_capacity(len.max(self.width) + 4);
        let n = self.width - len;
        let fill = |out: &mut String, k: usize| out.extend(std::iter::repeat_n(self.fill, k));
        match self.align.unwrap_or('>') {
            '<' => {
                out.push_str(sign);
                out.push_str(body);
                fill(&mut out, n);
            }
            '^' => {
                fill(&mut out, n / 2);
                out.push_str(sign);
                out.push_str(body);
                fill(&mut out, n - n / 2);
            }
            '=' => {
                out.push_str(sign);
                fill(&mut out, n);
                out.push_str(body);
            }
            _ => {
                fill(&mut out, n);
                out.push_str(sign);
                out.push_str(body);
            }
        }
        out
    }
}

/// Insert `sep` every three digits of the leading digit run of `body`.
fn group(body: &str, sep: char) -> String {
    let n = body.bytes().take_while(u8::is_ascii_digit).count();
    let mut out = String::with_capacity(body.len() + n / 3);
    for (i, c) in body[..n].chars().enumerate() {
        if i > 0 && (n - i) % 3 == 0 {
            out.push(sep);
        }
        out.push(c);
    }
    out.push_str(&body[n..]);
    out
}

/// `e+05`-style exponent (sign always, at least two digits).
fn push_exponent(out: &mut String, exp: i64, e: char) {
    out.push(e);
    out.push(if exp < 0 { '-' } else { '+' });
    let _ = write!(out, "{:02}", exp.unsigned_abs());
}

/// Digits from Rust's formatter, on the stack when they fit.
struct Digits {
    stack: [u8; 64],
    len: usize,
    heap: Vec<u8>,
}

impl Digits {
    fn as_slice(&self) -> &[u8] {
        if self.heap.is_empty() {
            &self.stack[..self.len]
        } else {
            &self.heap
        }
    }
}

/// A `fmt::Write` into a fixed buffer (fails when it is full).
struct StackWriter {
    buf: [u8; 96],
    len: usize,
}

impl std::fmt::Write for StackWriter {
    fn write_str(&mut self, s: &str) -> std::fmt::Result {
        let b = s.as_bytes();
        let end = self.len + b.len();
        if end > self.buf.len() {
            return Err(std::fmt::Error);
        }
        self.buf[self.len..end].copy_from_slice(b);
        self.len = end;
        Ok(())
    }
}

const POW10: [u128; 39] = {
    let mut t = [1u128; 39];
    let mut i = 1;
    while i < 39 {
        t[i] = t[i - 1] * 10;
        i += 1;
    }
    t
};

/// `round(a * 10**s)` exactly, ties to even, for a finite `a >= 0`: `None` when the integers
/// do not fit in 128 bits.
#[allow(clippy::many_single_char_names)] // a = m * 2^e, as in the docs
fn scaled_round(a: f64, s: i64) -> Option<u128> {
    let bits = a.to_bits();
    let exp_bits = i64::try_from((bits >> 52) & 0x7ff).ok()?;
    let frac = u128::from(bits & ((1u64 << 52) - 1));
    let (m, e) = if exp_bits == 0 {
        (frac, -1074)
    } else {
        (frac | (1 << 52), exp_bits - 1075)
    };
    if m == 0 {
        return Some(0);
    }
    // a * 10^s = m * 2^e * 10^s = num / den
    let mut num = m;
    let mut den: u128 = 1;
    let pow = |k: i64| POW10.get(usize::try_from(k).ok()?).copied();
    if (0..20).contains(&s) {
        // m < 2^53 and 10^s < 2^64: the product fits.
        num *= POW10[usize::try_from(s).ok()?];
    } else if s >= 0 {
        num = num.checked_mul(pow(s)?)?;
    } else {
        den = pow(-s)?;
    }
    let shift = |x: u128, k: i64| -> Option<u128> {
        let k = u32::try_from(k).ok()?;
        (k < x.leading_zeros()).then(|| x << k)
    };
    if e >= 0 {
        num = shift(num, e)?;
    } else {
        den = shift(den, -e)?;
    }
    let (q, r) = if den.is_power_of_two() {
        let k = den.trailing_zeros();
        (num >> k, num & (den - 1))
    } else {
        (num / den, num % den)
    };
    let half = den - r; // compare r with den / 2 without overflow
    Some(if r > half || (r == half && q & 1 == 1) {
        q + 1
    } else {
        q
    })
}

/// `{a:.p$}` (correctly rounded, ties to even) when the integers fit; `false` otherwise.
fn fixed_fast(out: &mut String, a: f64, p: usize) -> bool {
    if p > 30 || a >= 1e15 {
        return false;
    }
    let Some(q) = scaled_round(a, i64::try_from(p).unwrap_or(i64::MAX)) else {
        return false;
    };
    let unit = POW10[p];
    let (int, frac) = (q / unit, q % unit);
    let (Ok(int), Ok(frac)) = (u64::try_from(int), u64::try_from(frac)) else {
        let _ = write!(out, "{int}");
        if p > 0 {
            let _ = write!(out, ".{frac:0p$}");
        }
        return true;
    };
    push_u64(out, int, 0);
    if p > 0 {
        out.push('.');
        push_u64(out, frac, p);
    }
    true
}

/// Append `n` in decimal, zero-padded to at least `width` digits.
pub fn push_u64(out: &mut String, mut n: u64, width: usize) {
    let mut buf = [b'0'; 40];
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        #[allow(clippy::cast_possible_truncation)]
        let digit = (n % 10) as u8;
        buf[i] = b'0' + digit;
        n /= 10;
    }
    let start = i.min(buf.len().saturating_sub(width.max(1)));
    out.push_str(std::str::from_utf8(&buf[start..]).expect("ascii digits"));
}

/// `round(a)` to `p` significant digits as an integer of exactly `p` digits, and the decimal
/// point position, by exact integer arithmetic (up to 17 digits).
fn sig_int(a: f64, p: usize) -> Option<(u64, i64)> {
    if p > 17 || !a.is_finite() || a <= 0.0 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation)]
    let mut k = a.log10().floor() as i64;
    let p_i = i64::try_from(p).ok()?;
    for _ in 0..3 {
        let q = scaled_round(a, p_i - 1 - k)?;
        if q >= POW10[p] {
            if q == POW10[p] {
                return Some((u64::try_from(POW10[p - 1]).ok()?, k + 2));
            }
            k += 1;
        } else if q < POW10[p - 1] {
            k -= 1;
        } else {
            return Some((u64::try_from(q).ok()?, k + 1));
        }
    }
    None
}

/// [`significant`] by exact integer arithmetic, for up to 17 digits.
fn significant_fast(a: f64, p: usize) -> Option<(Digits, i64)> {
    let (q, decpt) = sig_int(a, p)?;
    Some(digits_of(u128::from(q), decpt))
}

/// `'g'` with `p` significant digits (no `#`) for `a > 0`, straight into `out`, as
/// [`build`] lays it out; `false` when the integers do not fit.
#[allow(clippy::many_single_char_names)]
fn g_fast(out: &mut String, a: f64, p: usize, add_dot_0: bool) -> bool {
    let Some((mut q, decpt)) = sig_int(a, p) else {
        return false;
    };
    let mut nd = p;
    while nd > 1 && q % 10 == 0 {
        q /= 10;
        nd -= 1;
    }
    let mut buf = [b'0'; 20];
    let mut i = buf.len();
    while q > 0 {
        i -= 1;
        #[allow(clippy::cast_possible_truncation)]
        let digit = (q % 10) as u8;
        buf[i] = b'0' + digit;
        q /= 10;
    }
    let digits = std::str::from_utf8(&buf[buf.len() - nd..]).expect("ascii digits");
    let nd_i = i64::try_from(nd).expect("small");
    let p_i = i64::try_from(p).expect("small");
    let use_exp = decpt <= -4 || decpt > (if add_dot_0 { p_i - 1 } else { p_i });
    if use_exp {
        out.push_str(&digits[..1]);
        if nd > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        push_exponent(out, decpt - 1, 'e');
    } else if decpt <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n(
            '0',
            usize::try_from(-decpt).expect("small"),
        ));
        out.push_str(digits);
    } else if decpt >= nd_i {
        out.push_str(digits);
        out.extend(std::iter::repeat_n(
            '0',
            usize::try_from(decpt - nd_i).expect("small"),
        ));
        if add_dot_0 {
            out.push_str(".0");
        }
    } else {
        let d = usize::try_from(decpt).expect("positive");
        out.push_str(&digits[..d]);
        out.push('.');
        out.push_str(&digits[d..]);
    }
    true
}

fn digits_of(q: u128, decpt: i64) -> (Digits, i64) {
    let mut d = Digits {
        stack: [0; 64],
        len: 0,
        heap: Vec::new(),
    };
    let mut tmp = [0u8; 40];
    let mut n = 0;
    let mut q = u64::try_from(q).expect("at most 17 digits");
    while q > 0 {
        #[allow(clippy::cast_possible_truncation)]
        let digit = (q % 10) as u8;
        tmp[n] = b'0' + digit;
        n += 1;
        q /= 10;
    }
    let mut start = 0;
    while start < n - 1 && tmp[start] == b'0' {
        start += 1; // trailing zeros of the number (tmp is reversed)
    }
    for i in (start..n).rev() {
        d.stack[d.len] = tmp[i];
        d.len += 1;
    }
    (d, decpt)
}

/// The first `p` significant digits of `a > 0` (correctly rounded), without trailing zeros,
/// and the decimal point position (`a = 0.d1d2... * 10^decpt`): dtoa's mode 2.
fn significant(a: f64, p: usize) -> (Digits, i64) {
    if a == 0.0 {
        return split_exp(b"0e0");
    }
    if let Some(r) = significant_fast(a, p) {
        return r;
    }
    let mut w = StackWriter {
        buf: [0; 96],
        len: 0,
    };
    if write!(w, "{a:.prec$e}", prec = p - 1).is_ok() {
        return split_exp(&w.buf[..w.len]);
    }
    split_exp(format!("{a:.prec$e}", prec = p - 1).as_bytes())
}

/// The shortest round-tripping digits of `a` (dtoa's mode 0).
fn shortest(a: f64) -> (Digits, i64) {
    if a == 0.0 {
        return split_exp(b"0e0");
    }
    let mut w = StackWriter {
        buf: [0; 96],
        len: 0,
    };
    write!(w, "{a:e}").expect("a shortest float fits");
    split_exp(&w.buf[..w.len])
}

fn split_exp(s: &[u8]) -> (Digits, i64) {
    let at = s.iter().position(|&b| b == b'e').expect("exponent form");
    let mut d = Digits {
        stack: [0; 64],
        len: 0,
        heap: Vec::new(),
    };
    let mantissa = s[..at].iter().copied().filter(u8::is_ascii_digit);
    if at <= d.stack.len() {
        for b in mantissa {
            d.stack[d.len] = b;
            d.len += 1;
        }
        while d.len > 1 && d.stack[d.len - 1] == b'0' {
            d.len -= 1;
        }
    } else {
        d.heap = mantissa.collect();
        while d.heap.len() > 1 && d.heap.last() == Some(&b'0') {
            d.heap.pop();
        }
    }
    let exp: i64 = std::str::from_utf8(&s[at + 1..])
        .expect("ascii")
        .parse()
        .expect("exponent");
    (d, exp + 1)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Code {
    G,
    R,
}

/// CPython's `format_float_short` layout for `'g'` and `'r'` from dtoa digits.
fn build(
    out: &mut String,
    digits: &[u8],
    decpt: i64,
    code: Code,
    precision: usize,
    add_dot_0: bool,
    alt: bool,
) {
    let digits_len = i64::try_from(digits.len()).expect("small");
    let precision = i64::try_from(precision).expect("small");
    let mut decpt = decpt;
    let mut vdigits_end = digits_len;
    let mut use_exp = false;
    match code {
        Code::G => {
            if decpt <= -4 || decpt > (if add_dot_0 { precision - 1 } else { precision }) {
                use_exp = true;
            }
            if alt {
                vdigits_end = precision;
            }
        }
        Code::R => {
            if decpt <= -4 || decpt > 16 {
                use_exp = true;
            }
        }
    }
    let mut exp = 0;
    if use_exp {
        exp = decpt - 1;
        decpt = 1;
    }
    let vdigits_start = if decpt <= 0 { decpt - 1 } else { 0 };
    vdigits_end = if !use_exp && add_dot_0 {
        vdigits_end.max(decpt + 1)
    } else {
        vdigits_end.max(decpt)
    };
    let zeros = |out: &mut String, n: i64| {
        if let Ok(n) = usize::try_from(n) {
            out.extend(std::iter::repeat_n('0', n));
        }
    };
    let digits = std::str::from_utf8(digits).expect("ascii digits");
    if decpt <= 0 {
        zeros(out, decpt - vdigits_start);
        out.push('.');
        zeros(out, -decpt);
    } else {
        zeros(out, -vdigits_start);
    }
    if 0 < decpt && decpt <= digits_len {
        let d = usize::try_from(decpt).expect("positive");
        out.push_str(&digits[..d]);
        out.push('.');
        out.push_str(&digits[d..]);
    } else {
        out.push_str(digits);
    }
    if digits_len < decpt {
        zeros(out, decpt - digits_len);
        out.push('.');
        zeros(out, vdigits_end - decpt);
    } else {
        zeros(out, vdigits_end - digits_len);
    }
    if out.ends_with('.') && !alt {
        out.pop();
    }
    if use_exp {
        push_exponent(out, exp, 'e');
    }
}

/// `repr(x)` (and `str(x)`, `format(x, "")`) for a float.
pub fn float_repr(x: f64) -> String {
    Spec::parse("", false)
        .expect("empty spec parses")
        .format_float(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(x: f64, spec: &str) -> String {
        Spec::parse(spec, false)
            .unwrap_or_else(|| panic!("{spec:?}"))
            .format_float(x)
    }

    #[test]
    fn repr_like_python() {
        let cases = [
            (1.5, "1.5"),
            (1.0, "1.0"),
            (-0.0, "-0.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (9_999_999_999_999_998.0, "9999999999999998.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (123_456.789, "123456.789"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (f64::NAN, "nan"),
            (-f64::NAN, "nan"),
            (1e-310, "1e-310"),
            (f64::MAX, "1.7976931348623157e+308"),
            (0.1 + 0.2, "0.30000000000000004"),
        ];
        for (x, want) in cases {
            assert_eq!(float_repr(x), want, "{x:?}");
        }
    }

    #[test]
    fn g_like_python() {
        let cases = [
            (1.0, "g", "1"),
            (0.0, "g", "0"),
            (-0.0, "g", "-0"),
            (-0.0, "zg", "0"),
            (123_456.0, "g", "123456"),
            (1_234_567.0, "g", "1.23457e+06"),
            (0.0001, "g", "0.0001"),
            (0.00001, "g", "1e-05"),
            (41.9999, "g", "41.9999"),
            (451.0, "g", "451"),
            (2.5, ".0g", "2"),
            (1.0, "#g", "1.00000"),
            (1.0, ".3", "1.0"),
            (123.0, ".3", "1.23e+02"),
            (12.0, ".3", "12.0"),
            (1234.5, ",.2f", "1,234.50"),
            (1234.5, "_g", "1_234.5"),
            (1e20, ",.0f", "100,000,000,000,000,000,000"),
            (0.125, ".2f", "0.12"),
            (0.5, ".0f", "0"),
            (1.0, "#.0f", "1."),
            (1.0, ".0e", "1e+00"),
            (1.0, "#.0e", "1.e+00"),
            (12_345.678, ".3e", "1.235e+04"),
            (12_345.678, "E", "1.234568E+04"),
            (f64::INFINITY, "F", "INF"),
            (f64::NAN, "+g", "+nan"),
            (-f64::NAN, "g", "nan"),
            (0.25, ".1%", "25.0%"),
            (f64::INFINITY, "%", "inf%"),
            (1.5, "10", "       1.5"),
            (1.5, "<10", "1.5       "),
            (1.5, "^10", "   1.5    "),
            (-1.5, "010.2f", "-000001.50"),
            (-1.5, "*=10.2f", "-*****1.50"),
            (1.5, "+.1f", "+1.5"),
            (1.5, " .1f", " 1.5"),
            (-0.001, "z.2f", "0.00"),
            (-0.001, ".2f", "-0.00"),
            (1e16, "#", "1.e+16"),
            (5.0, "<010", "5.00000000"),
        ];
        for (x, spec, want) in cases {
            assert_eq!(f(x, spec), want, "{x:?} {spec:?}");
        }
    }

    /// The integer fast paths give Rust's (CPython's) digits.
    #[test]
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    fn fast_paths_match_rusts_formatter() {
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut values: Vec<f64> = vec![
            0.125,
            0.5,
            1.5,
            2.5,
            0.25,
            0.375,
            1e22,
            1e23,
            5e-324,
            1e-5,
            0.000_015,
            999_999.5,
            9_999_995.0,
            0.999_999_5,
            1.0,
            10.0,
            100.0,
            1e15,
            123_456.5,
            2f64.powi(52),
            2f64.powi(53),
        ];
        let n = if cfg!(debug_assertions) {
            20_000
        } else {
            300_000
        };
        for _ in 0..n {
            let r = next();
            values.push(f64::from_bits(r >> 1)); // any positive double
            let mag = (r % 40) as i32 - 20;
            values.push((r >> 11) as f64 / (1u64 << 53) as f64 * 10f64.powi(mag));
            values.push(
                ((r >> 20) % 2_000_000) as f64
                    / [1.0, 2.0, 4.0, 8.0, 16.0, 100.0, 1000.0][(r % 7) as usize],
            );
        }
        for &a in &values {
            if !a.is_finite() || a <= 0.0 {
                continue;
            }
            for p in [1usize, 2, 3, 6, 10, 15, 17] {
                if let Some((d, decpt)) = significant_fast(a, p) {
                    let want = split_exp(format!("{a:.prec$e}", prec = p - 1).as_bytes());
                    assert_eq!(
                        (d.as_slice(), decpt),
                        (want.0.as_slice(), want.1),
                        "{a:e} p={p}"
                    );
                }
            }
            for p in [1usize, 3, 6, 12, 17] {
                for add_dot_0 in [false, true] {
                    let mut got = String::new();
                    if g_fast(&mut got, a, p, add_dot_0) {
                        let mut want = String::new();
                        let d = split_exp(format!("{a:.prec$e}", prec = p - 1).as_bytes());
                        build(&mut want, d.0.as_slice(), d.1, Code::G, p, add_dot_0, false);
                        assert_eq!(got, want, "{a:e} p={p} {add_dot_0}");
                    }
                }
            }
            for p in [0usize, 1, 2, 3, 6] {
                let mut got = String::new();
                if fixed_fast(&mut got, a, p) {
                    assert_eq!(got, format!("{a:.p$}"), "{a:e} p={p}");
                }
            }
        }
    }

    #[test]
    fn ints_like_python() {
        let i = |v: i128, spec: &str| Spec::parse(spec, true).unwrap().format_int(v);
        assert_eq!(i(5, ""), "5");
        assert_eq!(i(-1_234_567, ","), "-1,234,567");
        assert_eq!(i(1_234, "_"), "1_234");
        assert_eq!(i(5, "04d"), "0005");
        assert_eq!(i(-5, "04d"), "-005");
        assert_eq!(i(5, "+"), "+5");
        assert_eq!(i(5, "^5"), "  5  ");
        assert_eq!(i(i128::from(i64::MIN), ""), "-9223372036854775808");
    }

    #[test]
    fn unsupported_specs_are_left_to_python() {
        for spec in [
            "n",
            "x",
            "c",
            "010,.2f",
            "=,10",
            ".",
            ".2,f",
            ",_",
            "s",
            "abc",
            "99999999999",
            ".2fx",
        ] {
            assert!(Spec::parse(spec, false).is_none(), "{spec:?}");
        }
        for spec in ["z", ".2", "x", "f", "n"] {
            assert!(Spec::parse(spec, true).is_none(), "{spec:?}");
        }
    }
}
