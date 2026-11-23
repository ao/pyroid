// Derived from tabulate 0.10.0 (MIT), © 2011-2020 Sergey Astanin and contributors; modified.
//! Which strings tabulate takes for numbers: CPython's `int(s)` and `float(s)` grammars,
//! tabulate's thousands-separator regex, `_type` for `str` cells and `_afterpoint`.
//!
//! The grammars are exact for ASCII text. For other text CPython first maps Unicode decimal
//! digits to ASCII digits and Unicode whitespace to spaces, so a string with a non-ASCII char
//! can only be a number if every such char is a decimal digit or a space. When Rust's Unicode
//! tables are at least as new as the interpreter's (`unicode_ok`), a non-ASCII char that is
//! neither `is_numeric` nor `is_whitespace` (supersets of both) settles it; any other case is
//! [`Tri::Ask`]: the caller asks the interpreter.

/// An answer, or "ask Python".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tri {
    No,
    Yes,
    Ask,
}

/// `Py_ISSPACE`: the ASCII whitespace `int()` and `float()` strip.
#[inline]
fn is_c_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn strip_c_space(s: &[u8]) -> &[u8] {
    let start = s.iter().position(|&b| !is_c_space(b)).unwrap_or(s.len());
    let end = s
        .iter()
        .rposition(|&b| !is_c_space(b))
        .map_or(start, |i| i + 1);
    &s[start..end]
}

/// Could a string holding these chars be accepted by `int()` or `float()`? `false` is certain.
pub fn may_be_number(s: &str, unicode_ok: bool) -> bool {
    s.chars().all(|c| {
        if c.is_ascii() {
            let b = c as u8;
            b.is_ascii_digit()
                || is_c_space(b)
                || matches!(
                    b.to_ascii_lowercase(),
                    b'+' | b'-' | b'.' | b'_' | b'e' | b'i' | b'n' | b'f' | b't' | b'y' | b'a'
                )
        } else {
            !unicode_ok || c.is_numeric() || c.is_whitespace()
        }
    })
}

/// `int(s)` succeeds, for ASCII `s` (`max_digits`: `sys.get_int_max_str_digits()`, 0 = none).
pub fn int_ok_ascii(s: &[u8], max_digits: usize) -> bool {
    let t = strip_c_space(s);
    let t = match t.first() {
        Some(b'+' | b'-') => &t[1..],
        _ => t,
    };
    if t.first() == Some(&b'_') {
        return false;
    }
    let mut digits = 0usize;
    let mut prev = 0u8;
    for &b in t {
        if b.is_ascii_digit() {
            digits += 1;
        } else if b == b'_' {
            if prev == b'_' {
                return false;
            }
        } else {
            return false;
        }
        prev = b;
    }
    digits > 0 && prev != b'_' && !(max_digits > 0 && digits > max_digits)
}

/// `float(s)` for ASCII `s`: the value, or `None` where CPython raises `ValueError`.
pub fn float_ascii(s: &[u8]) -> Option<f64> {
    let t = strip_c_space(s);
    if t.is_empty() {
        return None;
    }
    let cleaned;
    let t = if t.contains(&b'_') {
        let mut out = Vec::with_capacity(t.len());
        let mut prev = 0u8;
        for &b in t {
            if b == b'_' {
                if !prev.is_ascii_digit() {
                    return None;
                }
            } else {
                if prev == b'_' && !b.is_ascii_digit() {
                    return None;
                }
                out.push(b);
            }
            prev = b;
        }
        if prev == b'_' {
            return None;
        }
        cleaned = out;
        &cleaned[..]
    } else {
        t
    };
    let (neg, body) = match t.first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body.eq_ignore_ascii_case(b"inf") || body.eq_ignore_ascii_case(b"infinity") {
        return Some(if neg {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }
    if body.eq_ignore_ascii_case(b"nan") {
        return Some(if neg { -f64::NAN } else { f64::NAN });
    }
    // digits [. digits] [e [sign] digits], at least one mantissa digit.
    let mut i = 0;
    let mut mantissa = 0;
    while i < body.len() && body[i].is_ascii_digit() {
        i += 1;
        mantissa += 1;
    }
    if i < body.len() && body[i] == b'.' {
        i += 1;
        while i < body.len() && body[i].is_ascii_digit() {
            i += 1;
            mantissa += 1;
        }
    }
    if mantissa == 0 {
        return None;
    }
    if i < body.len() && (body[i] == b'e' || body[i] == b'E') {
        i += 1;
        if i < body.len() && (body[i] == b'+' || body[i] == b'-') {
            i += 1;
        }
        let start = i;
        while i < body.len() && body[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return None;
        }
    }
    if i != body.len() {
        return None;
    }
    // The grammar is a subset of Rust's, and both round correctly.
    let text = std::str::from_utf8(t).ok()?;
    text.parse::<f64>().ok()
}

/// tabulate's `_float_with_thousands_separators` (`re.match`, so `$` also matches before one
/// trailing `"\n"`): `[+-]?d{1,3}(,ddd)*(\.d*)?`, or `(\.d+)?` (the empty string included).
pub fn thousands_match(s: &str) -> bool {
    let b = s.as_bytes();
    let b = b.strip_suffix(b"\n").unwrap_or(b);
    if b.is_empty() {
        return true;
    }
    let digit_run = |from: usize| b[from..].iter().take_while(|c| c.is_ascii_digit()).count();
    if b[0] == b'.' {
        let n = digit_run(1);
        if n > 0 && 1 + n == b.len() {
            return true;
        }
    }
    let mut i = usize::from(b[0] == b'+' || b[0] == b'-');
    let lead = digit_run(i);
    if !(1..=3).contains(&lead) {
        return false;
    }
    i += lead;
    while i < b.len() && b[i] == b',' {
        if digit_run(i + 1) != 3 {
            return false;
        }
        i += 4;
    }
    if i < b.len() && b[i] == b'.' {
        i += 1 + digit_run(i + 1);
    }
    i == b.len()
}

/// tabulate's `_isnumber(s)` for a `str`.
pub fn is_number(s: &str, unicode_ok: bool) -> Tri {
    if !s.is_ascii() {
        return if may_be_number(s, unicode_ok) {
            Tri::Ask
        } else {
            Tri::No
        };
    }
    match float_ascii(s.as_bytes()) {
        None => Tri::No,
        Some(f) if !f.is_finite() => {
            let lower = s.to_ascii_lowercase();
            if matches!(lower.as_str(), "inf" | "-inf" | "nan") {
                Tri::Yes
            } else {
                Tri::No
            }
        }
        Some(_) => Tri::Yes,
    }
}

/// tabulate's `_isint(s)` for a `str`: `int(s)` succeeds.
pub fn is_int(s: &str, max_digits: usize, unicode_ok: bool) -> Tri {
    if !s.is_ascii() {
        return if may_be_number(s, unicode_ok) {
            Tri::Ask
        } else {
            Tri::No
        };
    }
    if int_ok_ascii(s.as_bytes(), max_digits) {
        Tri::Yes
    } else {
        Tri::No
    }
}

/// The column types of tabulate, least generic first (`_more_generic`'s order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ColType {
    None = 0,
    Bool = 1,
    Int = 2,
    Float = 3,
    Bytes = 4,
    Str = 5,
}

impl ColType {
    /// From `_more_generic`'s rank.
    pub fn from_rank(rank: u8) -> Self {
        match rank {
            0 => Self::None,
            1 => Self::Bool,
            2 => Self::Int,
            3 => Self::Float,
            4 => Self::Bytes,
            _ => Self::Str,
        }
    }

    /// `ct in [int, float]`: numbers take `numalign`.
    pub fn is_numeric(self) -> bool {
        matches!(self, Self::Int | Self::Float)
    }
}

/// tabulate's `_type(s, numparse=numparse)` for a `str` already stripped of ANSI codes;
/// `None` means "ask Python".
pub fn str_type(s: &str, numparse: bool, max_digits: usize, unicode_ok: bool) -> Option<ColType> {
    if s.is_empty() {
        return Some(ColType::None);
    }
    if s == "True" || s == "False" {
        return Some(ColType::Bool);
    }
    if !numparse {
        return Some(ColType::Str);
    }
    // int(), float() and the thousands regex all need one of these first (after spaces).
    if let Some(&c) = s.as_bytes().iter().find(|&&b| !is_c_space(b)) {
        if c.is_ascii()
            && !matches!(
                c,
                b'0'..=b'9' | b'+' | b'-' | b'.' | b'i' | b'I' | b'n' | b'N'
            )
        {
            return Some(ColType::Str);
        }
    }
    if !s.is_ascii() {
        // The thousands regex is ASCII-only, so only int() and float() could accept it.
        return if may_be_number(s, unicode_ok) {
            None
        } else {
            Some(ColType::Str)
        };
    }
    let thousands = thousands_match(s);
    if int_ok_ascii(s.as_bytes(), max_digits) || (thousands && !s.contains('.')) {
        return Some(ColType::Int);
    }
    if is_number(s, unicode_ok) == Tri::Yes || thousands {
        return Some(ColType::Float);
    }
    Some(ColType::Str)
}

/// tabulate's `_afterpoint(s)`: digits after the decimal point (or after the `e`), `-1` for
/// non-numbers and ints; `None` means "ask Python".
pub fn afterpoint(s: &str, max_digits: usize, unicode_ok: bool) -> Option<isize> {
    if let Some(n) = afterpoint_plain(s, max_digits) {
        return Some(n);
    }
    if !s.is_ascii() {
        return if may_be_number(s, unicode_ok) {
            None
        } else {
            Some(-1)
        };
    }
    if !(is_number(s, unicode_ok) == Tri::Yes || thousands_match(s)) {
        return Some(-1);
    }
    if int_ok_ascii(s.as_bytes(), max_digits) {
        return Some(-1);
    }
    let pos = s.rfind('.').or_else(|| s.to_ascii_lowercase().rfind('e'));
    Some(pos.map_or(-1, |p| {
        isize::try_from(s.len() - p - 1).unwrap_or(isize::MAX)
    }))
}

/// `_afterpoint` for text that is plainly a decimal literal (`[+-]?d*(.d*)?([eE][+-]?ddd)?`,
/// far from overflow): the common case, without parsing the value. `None` for anything else.
fn afterpoint_plain(s: &str, max_digits: usize) -> Option<isize> {
    let b = s.as_bytes();
    let digits = |from: usize| b[from..].iter().take_while(|c| c.is_ascii_digit()).count();
    let mut i = usize::from(matches!(b.first(), Some(b'+' | b'-')));
    let int_digits = digits(i);
    i += int_digits;
    let mut dot = None;
    let mut frac_digits = 0;
    if b.get(i) == Some(&b'.') {
        dot = Some(i);
        frac_digits = digits(i + 1);
        i += 1 + frac_digits;
    }
    if int_digits + frac_digits == 0 || int_digits > 300 {
        return None;
    }
    let mut exp = None;
    if matches!(b.get(i), Some(b'e' | b'E')) {
        let at = i;
        i += 1 + usize::from(matches!(b.get(i + 1), Some(b'+' | b'-')));
        let n = digits(i);
        if n == 0 || n > 2 {
            return None;
        }
        // At most 300 + 99 digits before the point: keep clear of overflow to inf.
        if int_digits + 99 > 300 {
            return None;
        }
        i += n;
        exp = Some(at);
    }
    if i != b.len() {
        return None;
    }
    if dot.is_none() && exp.is_none() {
        // An int (int() takes it, within the digit limit).
        return (max_digits == 0 || int_digits <= max_digits).then_some(-1);
    }
    let pos = dot.or(exp)?;
    isize::try_from(b.len() - pos - 1).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_grammar() {
        for ok in ["1", " 12 ", "-0", "+7", "007", "1_000", "\t5\n", "0_0"] {
            assert!(int_ok_ascii(ok.as_bytes(), 4300), "{ok:?}");
        }
        for bad in [
            "", " ", "+", "1.0", "1e5", "_1", "1_", "1__0", "- 1", "0x10", "\x1c5", "1 2",
        ] {
            assert!(!int_ok_ascii(bad.as_bytes(), 4300), "{bad:?}");
        }
        let long = "1".repeat(4301);
        assert!(!int_ok_ascii(long.as_bytes(), 4300));
        assert!(int_ok_ascii(long.as_bytes(), 0));
        assert!(int_ok_ascii("1".repeat(4300).as_bytes(), 4300));
    }

    #[test]
    fn float_grammar() {
        let ok = [
            ("1", 1.0),
            (" 1.5 ", 1.5),
            ("1.", 1.0),
            (".5", 0.5),
            ("1e5", 1e5),
            ("1E+5", 1e5),
            ("-1e-5", -1e-5),
            ("1_000.000_1", 1000.0001),
            ("1e1_0", 1e10),
            ("1e400", f64::INFINITY),
            ("-InFiNiTy", f64::NEG_INFINITY),
            ("+inf", f64::INFINITY),
        ];
        for (s, v) in ok {
            assert_eq!(float_ascii(s.as_bytes()), Some(v), "{s:?}");
        }
        assert!(float_ascii(b"nan").is_some_and(f64::is_nan));
        assert!(float_ascii(b"-NaN").is_some_and(|f| f.is_nan() && f.is_sign_negative()));
        for bad in [
            "", ".", "e5", "1e", "1e+", "1_e5", "1e_5", "_1", "1_", "inf_", "infini", "1nan",
            "0x1p3", "1 2", "--1", "\x1c1",
        ] {
            assert_eq!(float_ascii(bad.as_bytes()), None, "{bad:?}");
        }
    }

    #[test]
    fn thousands_regex() {
        for ok in [
            "",
            "\n",
            "1",
            "12",
            "123",
            "1,000",
            "+1,000.1234",
            "-1,000.",
            "1,000\n",
            ".5",
            "1.",
        ] {
            assert!(thousands_match(ok), "{ok:?}");
        }
        for bad in [
            ".",
            "1000",
            "1,0000",
            "1,00",
            ",000",
            "1,000\n\n",
            "+.5",
            "1,000.5x",
            " 1",
        ] {
            assert!(!thousands_match(bad), "{bad:?}");
        }
    }

    #[test]
    fn str_types_like_tabulate() {
        let t = |s: &str| str_type(s, true, 4300, true);
        assert_eq!(t(""), Some(ColType::None));
        assert_eq!(t("True"), Some(ColType::Bool));
        assert_eq!(t("12"), Some(ColType::Int));
        assert_eq!(t(" 12 "), Some(ColType::Int));
        assert_eq!(t("1,000"), Some(ColType::Int));
        assert_eq!(t("\n"), Some(ColType::Int));
        assert_eq!(t("1.5"), Some(ColType::Float));
        assert_eq!(t("1,000.5"), Some(ColType::Float));
        assert_eq!(t("1e5"), Some(ColType::Float));
        assert_eq!(t("inf"), Some(ColType::Float));
        assert_eq!(t("-INF"), Some(ColType::Float));
        assert_eq!(t("infinity"), Some(ColType::Str));
        assert_eq!(t("+inf"), Some(ColType::Str));
        assert_eq!(t("-nan"), Some(ColType::Str));
        assert_eq!(t("1e400"), Some(ColType::Str));
        assert_eq!(t("spam"), Some(ColType::Str));
        assert_eq!(t("日本"), Some(ColType::Str));
        assert_eq!(t("١٢"), None);
        assert_eq!(str_type("12", false, 4300, true), Some(ColType::Str));
        assert_eq!(str_type("日本", true, 4300, false), None);
        assert_eq!(str_type("日本x", true, 4300, false), Some(ColType::Str));
    }

    #[test]
    fn afterpoint_like_tabulate() {
        let a = |s: &str| afterpoint(s, 4300, true);
        assert_eq!(a("123.45"), Some(2));
        assert_eq!(a("1001"), Some(-1));
        assert_eq!(a("eggs"), Some(-1));
        assert_eq!(a("123e45"), Some(2));
        assert_eq!(a("123,456.78"), Some(2));
        assert_eq!(a("1e+05"), Some(3));
        assert_eq!(a("inf"), Some(-1));
        assert_eq!(a(""), Some(-1));
        assert_eq!(a("1."), Some(0));
        assert_eq!(a(".5"), Some(1));
        assert_eq!(a("-.5e3"), Some(3));
        assert_eq!(a("1e5"), Some(1));
        assert_eq!(a("1e400"), Some(-1));
        assert_eq!(a(&format!("{}e99", "9".repeat(250))), Some(-1));
        assert_eq!(a(" 1.5"), Some(1));
        assert_eq!(a("1,234"), Some(-1));
        assert_eq!(afterpoint(&"1".repeat(4301), 4300, true), Some(-1));
        // The plain fast path and the general one agree.
        let general = |s: &str| -> isize {
            if !(is_number(s, true) == Tri::Yes || thousands_match(s))
                || int_ok_ascii(s.as_bytes(), 4300)
            {
                return -1;
            }
            let pos = s.rfind('.').or_else(|| s.to_ascii_lowercase().rfind('e'));
            pos.map_or(-1, |p| isize::try_from(s.len() - p - 1).unwrap())
        };
        for s in [
            "12",
            "-3.25",
            "1.5e+06",
            "+.5",
            "0.000",
            "1e-05",
            "123456.789",
            "5.",
            "-0",
        ] {
            assert_eq!(afterpoint_plain(s, 4300), Some(general(s)), "{s}");
        }
    }
}
