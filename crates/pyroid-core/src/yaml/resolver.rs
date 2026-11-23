// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! The resolver (`resolver.py`): PyYAML's implicit tags for plain scalars, YAML 1.1 style.
//!
//! PyYAML picks the candidate regexes by the first character of the value, in the order they
//! were added (bool, float, int, merge, null, timestamp, value, yaml), and `regexp.match`s
//! each. The patterns end in `$`, which also matches before one trailing `'\n'`; the
//! matchers below take the value with that newline removed. Each matcher accepts exactly
//! the language of PyYAML's pattern (`[0-9]` is ASCII only).

pub const STR: &str = "tag:yaml.org,2002:str";
pub const SEQ: &str = "tag:yaml.org,2002:seq";
pub const MAP: &str = "tag:yaml.org,2002:map";
pub const BOOL: &str = "tag:yaml.org,2002:bool";
pub const FLOAT: &str = "tag:yaml.org,2002:float";
pub const INT: &str = "tag:yaml.org,2002:int";
pub const MERGE: &str = "tag:yaml.org,2002:merge";
pub const NULL: &str = "tag:yaml.org,2002:null";
pub const TIMESTAMP: &str = "tag:yaml.org,2002:timestamp";
pub const VALUE: &str = "tag:yaml.org,2002:value";
pub const YAML: &str = "tag:yaml.org,2002:yaml";
pub const BINARY: &str = "tag:yaml.org,2002:binary";
pub const OMAP: &str = "tag:yaml.org,2002:omap";
pub const PAIRS: &str = "tag:yaml.org,2002:pairs";
pub const SET: &str = "tag:yaml.org,2002:set";

/// The standard tag with this name, as a `'static` str (to avoid allocating).
pub fn intern(tag: &str) -> Option<&'static str> {
    Some(match tag {
        STR => STR,
        SEQ => SEQ,
        MAP => MAP,
        BOOL => BOOL,
        FLOAT => FLOAT,
        INT => INT,
        MERGE => MERGE,
        NULL => NULL,
        TIMESTAMP => TIMESTAMP,
        VALUE => VALUE,
        BINARY => BINARY,
        OMAP => OMAP,
        PAIRS => PAIRS,
        SET => SET,
        _ => return None,
    })
}

/// `Resolver.resolve(ScalarNode, value, implicit)` for the stock resolvers (no path resolvers).
pub fn resolve_scalar(value: &str, implicit_plain: bool) -> &'static str {
    if implicit_plain {
        if let Some(tag) = implicit_tag(value) {
            return tag;
        }
    }
    STR
}

/// The implicit tag of `value`, if one of PyYAML's implicit resolvers matches it.
pub fn implicit_tag(value: &str) -> Option<&'static str> {
    let Some(&first) = value.as_bytes().first() else {
        return Some(NULL); // the '' resolver: null's `^(?:~|null|Null|NULL|)$` matches ''.
    };
    let v = value.strip_suffix('\n').unwrap_or(value).as_bytes();
    match first {
        b'y' | b'Y' | b't' | b'T' | b'f' | b'F' | b'o' | b'O' => is_bool(v).then_some(BOOL),
        b'n' | b'N' => {
            if is_bool(v) {
                Some(BOOL)
            } else {
                is_null(v).then_some(NULL)
            }
        }
        b'-' | b'+' => {
            if is_float(v) {
                Some(FLOAT)
            } else {
                is_int(v).then_some(INT)
            }
        }
        b'0'..=b'9' => {
            if is_float(v) {
                Some(FLOAT)
            } else if is_int(v) {
                Some(INT)
            } else {
                is_timestamp(v).then_some(TIMESTAMP)
            }
        }
        b'.' => is_float(v).then_some(FLOAT),
        b'<' => (v == b"<<").then_some(MERGE),
        b'~' => is_null(v).then_some(NULL),
        b'=' => (v == b"=").then_some(VALUE),
        b'!' | b'&' | b'*' => (v.len() == 1).then_some(YAML),
        _ => None,
    }
}

fn is_bool(v: &[u8]) -> bool {
    matches!(
        v,
        b"yes"
            | b"Yes"
            | b"YES"
            | b"no"
            | b"No"
            | b"NO"
            | b"true"
            | b"True"
            | b"TRUE"
            | b"false"
            | b"False"
            | b"FALSE"
            | b"on"
            | b"On"
            | b"ON"
            | b"off"
            | b"Off"
            | b"OFF"
    )
}

fn is_null(v: &[u8]) -> bool {
    matches!(v, b"" | b"~" | b"null" | b"Null" | b"NULL")
}

#[inline]
fn digit(b: u8) -> bool {
    b.is_ascii_digit()
}

#[inline]
fn digit_(b: u8) -> bool {
    b.is_ascii_digit() || b == b'_'
}

/// Consume `[-+]?` and return the rest.
#[inline]
fn sign(v: &[u8]) -> &[u8] {
    match v.first() {
        Some(b'-' | b'+') => &v[1..],
        _ => v,
    }
}

/// `(?::[0-5]?[0-9])+` at the start of `v`: the rest after the last group, or `None`.
fn sexagesimal_groups(mut v: &[u8]) -> Option<&[u8]> {
    let mut groups = 0;
    while v.first() == Some(&b':') {
        let digits = v[1..].iter().take_while(|b| b.is_ascii_digit()).count();
        // One digit, or two with the first in 0-5 (a third digit can never follow a group).
        let ok = digits == 1 || (digits == 2 && v[1] <= b'5');
        if !ok {
            return None;
        }
        v = &v[1 + digits..];
        groups += 1;
    }
    (groups > 0).then_some(v)
}

/// `(?:[eE][-+][0-9]+)?$`
fn exponent_end(v: &[u8]) -> bool {
    match v {
        [] => true,
        [b'e' | b'E', b'-' | b'+', rest @ ..] => !rest.is_empty() && rest.iter().all(|&b| digit(b)),
        _ => false,
    }
}

#[allow(clippy::many_single_char_names)]
pub fn is_float(v: &[u8]) -> bool {
    // [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+][0-9]+)?
    let s = sign(v);
    if s.first().is_some_and(|&b| digit(b)) {
        let n = s.iter().take_while(|&&b| digit_(b)).count();
        let rest = &s[n..];
        if rest.first() == Some(&b'.') {
            let r = &rest[1..];
            let m = r.iter().take_while(|&&b| digit_(b)).count();
            if exponent_end(&r[m..]) {
                return true;
            }
        }
        // [-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
        if sexagesimal_groups(rest)
            .is_some_and(|r| r.first() == Some(&b'.') && r[1..].iter().all(|&b| digit_(b)))
        {
            return true;
        }
    }
    // \.[0-9][0-9_]*(?:[eE][-+][0-9]+)?   (no sign)
    if v.first() == Some(&b'.') && v.get(1).is_some_and(|&b| digit(b)) {
        let r = &v[1..];
        let m = r.iter().take_while(|&&b| digit_(b)).count();
        if exponent_end(&r[m..]) {
            return true;
        }
    }
    // [-+]?\.(?:inf|Inf|INF)
    if matches!(s, b".inf" | b".Inf" | b".INF") {
        return true;
    }
    // \.(?:nan|NaN|NAN)   (no sign)
    matches!(v, b".nan" | b".NaN" | b".NAN")
}

pub fn is_int(v: &[u8]) -> bool {
    let s = sign(v);
    match s {
        // [-+]?0b[0-1_]+
        [b'0', b'b', rest @ ..]
            if !rest.is_empty() && rest.iter().all(|&b| matches!(b, b'0' | b'1' | b'_')) =>
        {
            true
        }
        // [-+]?0x[0-9a-fA-F_]+
        [b'0', b'x', rest @ ..]
            if !rest.is_empty() && rest.iter().all(|&b| b.is_ascii_hexdigit() || b == b'_') =>
        {
            true
        }
        // [-+]?0[0-7_]+ and the bare 0 of [-+]?(?:0|[1-9][0-9_]*)
        [b'0', rest @ ..] => rest.iter().all(|&b| matches!(b, b'0'..=b'7' | b'_')),
        // [-+]?[1-9][0-9_]* and [-+]?[1-9][0-9_]*(?::[0-5]?[0-9])+
        [b'1'..=b'9', rest @ ..] => {
            let n = rest.iter().take_while(|&&b| digit_(b)).count();
            let after = &rest[n..];
            after.is_empty() || sexagesimal_groups(after).is_some_and(<[u8]>::is_empty)
        }
        _ => false,
    }
}

/// `n` ASCII digits at the start of `v`: the rest.
#[inline]
fn digits_n(v: &[u8], n: usize) -> Option<&[u8]> {
    (v.len() >= n && v[..n].iter().all(|&b| digit(b))).then(|| &v[n..])
}

/// `[0-9][0-9]?` at the start of `v`: the rest. `then` must hold for the rest (a 1- or
/// 2-digit run is ambiguous only when another digit follows, which never matches `then`).
#[inline]
fn digits_1_2(v: &[u8]) -> Option<(&[u8], usize)> {
    let n = v.iter().take(2).take_while(|&&b| digit(b)).count();
    (n > 0).then(|| (&v[n..], n))
}

/// The parts of a timestamp, as the constructor's `timestamp_regexp` groups them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Timestamp<'a> {
    pub year: &'a str,
    pub month: &'a str,
    pub day: &'a str,
    /// `(hour, minute, second, fraction)`.
    pub time: Option<(&'a str, &'a str, &'a str, Option<&'a str>)>,
    /// `Some("Z")`, or the sign, hour and minute of the offset.
    pub tz: Option<Tz<'a>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tz<'a> {
    Utc,
    Offset {
        minus: bool,
        hour: &'a str,
        minute: Option<&'a str>,
    },
}

/// `SafeConstructor.timestamp_regexp.match(value)` (the trailing-`'\n'` rule included).
#[allow(clippy::many_single_char_names)]
pub fn parse_timestamp(value: &str) -> Option<Timestamp<'_>> {
    let v = value.strip_suffix('\n').unwrap_or(value);
    let b = v.as_bytes();
    let s = |r: &[u8]| -> &str { &v[b.len() - r.len()..] };
    let r = digits_n(b, 4)?;
    let year = &v[..4];
    let r = r.strip_prefix(b"-")?;
    let (r2, n) = digits_1_2(r)?;
    let month = &s(r)[..n];
    let r = r2.strip_prefix(b"-")?;
    let (r2, n) = digits_1_2(r)?;
    let day = &s(r)[..n];
    let r = r2;
    if r.is_empty() {
        return Some(Timestamp {
            year,
            month,
            day,
            time: None,
            tz: None,
        });
    }
    // (?:[Tt]|[ \t]+)
    let r = match r[0] {
        b'T' | b't' => &r[1..],
        b' ' | b'\t' => {
            let n = r.iter().take_while(|&&c| c == b' ' || c == b'\t').count();
            &r[n..]
        }
        _ => return None,
    };
    let (r2, n) = digits_1_2(r)?;
    let hour = &s(r)[..n];
    let r = r2.strip_prefix(b":")?;
    let rest = digits_n(r, 2)?;
    let minute = &s(r)[..2];
    let r = rest.strip_prefix(b":")?;
    let rest = digits_n(r, 2)?;
    let second = &s(r)[..2];
    let mut r = rest;
    let mut fraction = None;
    if r.first() == Some(&b'.') {
        let n = r[1..].iter().take_while(|&&c| digit(c)).count();
        fraction = Some(&s(&r[1..])[..n]);
        r = &r[1 + n..];
    }
    let time = Some((hour, minute, second, fraction));
    if r.is_empty() {
        return Some(Timestamp {
            year,
            month,
            day,
            time,
            tz: None,
        });
    }
    // (?:[ \t]*(?P<tz>Z|(?P<tz_sign>[-+])(?P<tz_hour>[0-9][0-9]?)(?::(?P<tz_minute>[0-9][0-9]))?))?
    let n = r.iter().take_while(|&&c| c == b' ' || c == b'\t').count();
    let r = &r[n..];
    let tz = match r.first() {
        Some(b'Z') if r.len() == 1 => Tz::Utc,
        Some(&c @ (b'-' | b'+')) => {
            let r1 = &r[1..];
            let (r2, n) = digits_1_2(r1)?;
            let hour = &s(r1)[..n];
            let minute = if r2.is_empty() {
                None
            } else {
                let r3 = r2.strip_prefix(b":")?;
                let rest = digits_n(r3, 2)?;
                if !rest.is_empty() {
                    return None;
                }
                Some(&s(r3)[..2])
            };
            Tz::Offset {
                minus: c == b'-',
                hour,
                minute,
            }
        }
        _ => return None,
    };
    Some(Timestamp {
        year,
        month,
        day,
        time,
        tz: Some(tz),
    })
}

/// The resolver's timestamp pattern (stricter than the constructor's: a bare date needs
/// two-digit month and day).
pub fn is_timestamp(v: &[u8]) -> bool {
    // [0-9]{4}-[0-9]{2}-[0-9]{2}
    if v.len() == 10
        && digits_n(v, 4).is_some()
        && v[4] == b'-'
        && digit(v[5])
        && digit(v[6])
        && v[7] == b'-'
        && digit(v[8])
        && digit(v[9])
    {
        return true;
    }
    // The long form: the constructor's pattern with a mandatory time part.
    std::str::from_utf8(v)
        .ok()
        .and_then(parse_timestamp)
        .is_some_and(|t| t.time.is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_tags() {
        let cases: &[(&str, &str)] = &[
            ("", NULL),
            ("~", NULL),
            ("null", NULL),
            ("nULL", STR),
            ("yes", BOOL),
            ("No", BOOL),
            ("ON", BOOL),
            ("true\n", BOOL),
            ("true\n\n", STR),
            ("1", INT),
            ("0", INT),
            ("-0", INT),
            ("00", INT),
            ("08", STR),
            ("0_7", INT),
            ("0b1_0", INT),
            ("0b", STR),
            ("0x_", INT),
            ("0xFF", INT),
            ("1_000", INT),
            ("1:20", INT),
            ("1:60", STR),
            ("1:5:59", INT),
            ("190:20:30.15", FLOAT),
            ("1.0", FLOAT),
            ("1.", FLOAT),
            ("-.5", STR),
            (".5", FLOAT),
            ("1e10", STR),
            ("1.0e10", STR),
            ("1.0e+10", FLOAT),
            ("6.8523015e+5", FLOAT),
            ("685.230_15e+03", FLOAT),
            (".inf", FLOAT),
            ("-.Inf", FLOAT),
            ("-.nan", STR),
            (".NaN", FLOAT),
            ("2001-12-14", TIMESTAMP),
            ("2001-1-14", STR),
            ("2001-12-14t21:59:43.10-05:00", TIMESTAMP),
            ("2001-12-14 21:59:43.10 -5", TIMESTAMP),
            ("2001-12-14 21:59:43.10 ", STR),
            ("2001-12-14  21:59:43.10Z", TIMESTAMP),
            ("2002-1-1 1:00:00", TIMESTAMP),
            ("<<", MERGE),
            ("=", VALUE),
            ("_1", STR),
            ("+", STR),
        ];
        for (v, tag) in cases {
            assert_eq!(resolve_scalar(v, true), *tag, "{v:?}");
        }
    }

    #[test]
    fn timestamps() {
        let t = parse_timestamp("2001-12-14t21:59:43.10-05:00").unwrap();
        assert_eq!((t.year, t.month, t.day), ("2001", "12", "14"));
        assert_eq!(t.time, Some(("21", "59", "43", Some("10"))));
        assert_eq!(
            t.tz,
            Some(Tz::Offset {
                minus: true,
                hour: "05",
                minute: Some("00")
            })
        );
        let t = parse_timestamp("2001-1-2").unwrap();
        assert_eq!((t.month, t.day, t.time), ("1", "2", None));
        assert!(parse_timestamp("2001-1-2 ").is_none());
        assert!(parse_timestamp("2001-01-02 1:02:03 Z").is_some());
        assert!(parse_timestamp("2001-01-02 1:02:03 Zx").is_none());
        assert_eq!(
            parse_timestamp("2001-01-02T1:02:03.")
                .unwrap()
                .time
                .unwrap()
                .3,
            Some("")
        );
    }
}
