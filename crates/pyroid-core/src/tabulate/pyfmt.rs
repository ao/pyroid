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
        let upper = matches!(self.ty, Some('E' | 'F' | 'G'));
        let mut neg = x.is_sign_negative() && !x.is_nan();
        let mut body = String::new();
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
                    let _ = write!(body, "{a:.p$}");
                    if self.alt && p == 0 {
                        body.push('.');
                    }
                }
                Some('e' | 'E') => {
                    let p = self.precision.unwrap_or(6);
                    let s = format!("{a:.p$e}");
                    let (m, e) = s.split_once('e').expect("exponent form");
                    body.push_str(m);
                    if self.alt && p == 0 {
                        body.push('.');
                    }
                    push_exponent(&mut body, e.parse().expect("exponent"), 'e');
                }
                Some('g' | 'G') => {
                    let p = self.precision.unwrap_or(6).max(1);
                    let (digits, decpt) = significant(a, p);
                    build(&mut body, &digits, decpt, Code::G, p, false, self.alt);
                }
                None => match self.precision {
                    None => {
                        let (digits, decpt) = shortest(a);
                        build(&mut body, &digits, decpt, Code::R, 0, true, self.alt);
                    }
                    Some(p) => {
                        let p = p.max(1);
                        let (digits, decpt) = significant(a, p);
                        build(&mut body, &digits, decpt, Code::G, p, true, self.alt);
                    }
                },
                Some(_) => unreachable!("rejected by parse"),
            }
            if self.no_neg_zero && !body.bytes().any(|b| (b'1'..=b'9').contains(&b)) {
                neg = false;
            }
        }
        if pct {
            body.push('%');
        }
        if upper {
            body.make_ascii_uppercase();
        }
        if finite {
            if let Some(sep) = self.grouping {
                body = group(&body, sep);
            }
        }
        self.pad(neg, &body)
    }

    /// `format(v, spec)` for an int.
    pub fn format_int(&self, v: i128) -> String {
        let mut body = v.unsigned_abs().to_string();
        if let Some(sep) = self.grouping {
            body = group(&body, sep);
        }
        self.pad(v < 0, &body)
    }

    fn pad(&self, neg: bool, body: &str) -> String {
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
        let mut out = String::with_capacity(len.max(self.width) + 4);
        if self.width <= len {
            out.push_str(sign);
            out.push_str(body);
            return out;
        }
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

/// The first `p` significant digits of `a > 0` (correctly rounded), without trailing zeros,
/// and the decimal point position (`a = 0.d1d2... * 10^decpt`): dtoa's mode 2.
fn significant(a: f64, p: usize) -> (Vec<u8>, i64) {
    if a == 0.0 {
        return (vec![b'0'], 1);
    }
    let s = format!("{a:.prec$e}", prec = p - 1);
    split_exp(&s)
}

/// The shortest round-tripping digits of `a` (dtoa's mode 0).
fn shortest(a: f64) -> (Vec<u8>, i64) {
    if a == 0.0 {
        return (vec![b'0'], 1);
    }
    split_exp(&format!("{a:e}"))
}

fn split_exp(s: &str) -> (Vec<u8>, i64) {
    let (m, e) = s.split_once('e').expect("exponent form");
    let mut digits: Vec<u8> = m.bytes().filter(u8::is_ascii_digit).collect();
    while digits.len() > 1 && digits.last() == Some(&b'0') {
        digits.pop();
    }
    let exp: i64 = e.parse().expect("exponent");
    (digits, exp + 1)
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
