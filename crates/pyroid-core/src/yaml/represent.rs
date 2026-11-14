// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! Pure parts of `SafeRepresenter`: the text of floats and ints.

use std::fmt::Write as _;

/// Python's `repr(x)` for a finite float: the shortest digits that round-trip, in fixed
/// notation when the decimal exponent is in -4..16, else in exponent notation (`1e+16`,
/// `1.5e-05`), with `.0` added to integral fixed values.
pub fn python_float_repr(x: f64) -> String {
    // `{:e}` gives the shortest round-trip length: "d.ddde<exp>". Python's repr takes, of the
    // strings of that length, the one nearest the exact value, ties to even; Rust's exact
    // formatting at that precision rounds the same way (the shortest one may not, on a tie).
    let shortest = format!("{:e}", x.abs());
    let len = shortest
        .split_once('e')
        .map_or(1, |(m, _)| m.chars().filter(char::is_ascii_digit).count());
    let exact = format!("{:.*e}", len.saturating_sub(1), x.abs());
    let sci = if exact.parse::<f64>().ok() == Some(x.abs()) {
        exact
    } else {
        shortest
    };
    let (mantissa, exp) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exp.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let n = i32::try_from(digits.len()).unwrap_or(i32::MAX);
    let decpt = exp + 1;
    let mut out = String::with_capacity(24);
    if x.is_sign_negative() {
        out.push('-');
    }
    if x == 0.0 {
        out.push_str("0.0");
        return out;
    }
    if !(-4 < decpt && decpt <= 16) {
        out.push_str(&digits[..1]);
        if n > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let e = decpt - 1;
        let _ = write!(out, "e{}{:02}", if e < 0 { '-' } else { '+' }, e.abs());
    } else if decpt <= 0 {
        out.push_str("0.");
        for _ in 0..-decpt {
            out.push('0');
        }
        out.push_str(digits);
    } else if decpt >= n {
        out.push_str(digits);
        for _ in 0..decpt - n {
            out.push('0');
        }
        out.push_str(".0");
    } else {
        let d = usize::try_from(decpt).unwrap_or(0);
        out.push_str(&digits[..d]);
        out.push('.');
        out.push_str(&digits[d..]);
    }
    out
}

/// `SafeRepresenter.represent_float`'s text.
pub fn float_text(x: f64) -> String {
    if x.is_nan() {
        return ".nan".to_owned();
    }
    if x == f64::INFINITY {
        return ".inf".to_owned();
    }
    if x == f64::NEG_INFINITY {
        return "-.inf".to_owned();
    }
    // repr(data).lower(), and '.0' before an 'e' when there is no '.'.
    let r = python_float_repr(x);
    if !r.contains('.') && r.contains('e') {
        return r.replacen('e', ".0e", 1);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reprs() {
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (1.5, "1.5"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (123_456_789_012_345_680.0, "1.2345678901234568e+17"),
            (0.0001, "0.0001"),
            (0.000_01, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (-2.5e-300, "-2.5e-300"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (100.0, "100.0"),
            (12.34, "12.34"),
            // Exact ties between two shortest candidates: Python rounds them to even.
            (1_711.531_066_894_531_2, "1711.5310668945312"),
            (101.633_331_298_828_12, "101.63333129882812"),
            (-122.433_013_916_015_62, "-122.43301391601562"),
        ];
        for &(x, want) in cases {
            assert_eq!(python_float_repr(x), want, "{x}");
        }
        assert_eq!(float_text(1e17), "1.0e+17");
        assert_eq!(float_text(f64::NAN), ".nan");
        assert_eq!(float_text(-f64::INFINITY), "-.inf");
    }
}
