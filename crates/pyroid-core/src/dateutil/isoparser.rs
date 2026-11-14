// Derived from python-dateutil 4.0.0 and 2.9.0.post0 (BSD-3-Clause), © Gustavo Niemeyer,
// Paul Ganssle and dateutil contributors; modified.
//! `isoparser.isoparse` on the ASCII bytes of the string, in both releases' modes:
//!
//! - [`Mode::Strict`] (dateutil 4.0.0): each numeric field is exactly N ASCII digits
//!   (`_parse_digits`), and an offset's minutes must fill the string.
//! - [`Mode::Lenient`] (2.9.0.post0 and 3.0.0): fields are read with `int(bytes)`, which takes
//!   surrounding whitespace, a sign and underscores (`2014- 4-21`, `+-1:00`).
//!
//! The kernel only decides *whether* dateutil raises; the caller then runs dateutil's own code
//! for the exception, so its type and message are exact. It also leaves the `datetime`
//! construction (and its range errors) to the caller.

use super::cal;

/// Which release's `isoparse` to reproduce.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// dateutil 4.0.0.
    Strict,
    /// dateutil 2.9.0.post0 and 3.0.0.
    Lenient,
}

/// The time zone `_parse_tzstr` returns.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IsoTz {
    /// `tz.UTC`.
    Utc,
    /// `tz.tzoffset(None, seconds)`.
    Offset(i64),
}

/// What `isoparse` passes to `datetime(...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Iso {
    /// year, month, day, hour, minute, second, microsecond.
    pub fields: [i64; 7],
    pub tz: Option<IsoTz>,
    /// The hour was 24: dateutil builds hour 0 and adds a day.
    pub plus_day: bool,
}

/// dateutil raises (`ValueError`, or an error from `date()`/`timedelta` arithmetic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Raises;

type R<T> = Result<T, Raises>;

/// `int(b)` for ASCII bytes, base 10.
fn int_bytes(b: &[u8]) -> R<i64> {
    let is_ws = |c: &u8| matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r');
    let start = b.iter().position(|c| !is_ws(c)).ok_or(Raises)?;
    let end = b.iter().rposition(|c| !is_ws(c)).expect("non-empty") + 1;
    let t = &b[start..end];
    let (neg, body) = match t.first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if body.is_empty() || body[0] == b'_' || body[body.len() - 1] == b'_' {
        return Err(Raises);
    }
    let mut v: i64 = 0;
    let mut prev_us = false;
    for &c in body {
        if c == b'_' {
            if prev_us {
                return Err(Raises);
            }
            prev_us = true;
            continue;
        }
        prev_us = false;
        if !c.is_ascii_digit() {
            return Err(Raises);
        }
        // Slices are at most 4 bytes long, except an offset's minutes in lenient mode.
        v = v
            .checked_mul(10)
            .and_then(|v| v.checked_add(i64::from(c - b'0')))
            .ok_or(Raises)?;
    }
    Ok(if neg { -v } else { v })
}

/// `b[a:b]` with Python's clamping.
fn sl(b: &[u8], a: usize, e: usize) -> &[u8] {
    let n = b.len();
    let a = a.min(n);
    &b[a..e.min(n).max(a)]
}

struct P {
    mode: Mode,
}

impl P {
    /// `_parse_digits(s, pos, n)` (strict) or `int(s[pos:pos + n])` (lenient).
    fn digits(&self, s: &[u8], pos: usize, n: usize) -> R<i64> {
        let d = sl(s, pos, pos + n);
        match self.mode {
            Mode::Lenient => int_bytes(d),
            Mode::Strict => {
                if d.len() != n || !d.iter().all(u8::is_ascii_digit) {
                    return Err(Raises);
                }
                Ok(d.iter().fold(0, |a, &c| a * 10 + i64::from(c - b'0')))
            }
        }
    }

    fn isodate(&self, s: &[u8]) -> R<([i64; 3], usize)> {
        self.isodate_common(s).or_else(|_| self.isodate_uncommon(s))
    }

    fn isodate_common(&self, s: &[u8]) -> R<([i64; 3], usize)> {
        let len = s.len();
        let mut c = [1, 1, 1];
        if len < 4 {
            return Err(Raises);
        }
        c[0] = self.digits(s, 0, 4)?;
        let mut pos = 4;
        if pos >= len {
            return Ok((c, pos));
        }
        let has_sep = s[pos] == b'-';
        if has_sep {
            pos += 1;
        }
        if len - pos < 2 {
            return Err(Raises);
        }
        c[1] = self.digits(s, pos, 2)?;
        pos += 2;
        if pos >= len {
            return if has_sep { Ok((c, pos)) } else { Err(Raises) };
        }
        if has_sep {
            if s[pos] != b'-' {
                return Err(Raises);
            }
            pos += 1;
        }
        if len - pos < 2 {
            return Err(Raises);
        }
        c[2] = self.digits(s, pos, 2)?;
        Ok((c, pos + 2))
    }

    fn isodate_uncommon(&self, s: &[u8]) -> R<([i64; 3], usize)> {
        if s.len() < 4 {
            return Err(Raises);
        }
        let year = self.digits(s, 0, 4)?;
        let has_sep = sl(s, 4, 5) == b"-";
        let mut pos = 4 + usize::from(has_sep);
        let ord = if sl(s, pos, pos + 1) == b"W" {
            // YYYY-?Www-?D?
            pos += 1;
            let weekno = self.digits(s, pos, 2)?;
            pos += 2;
            let mut dayno = 1;
            if s.len() > pos {
                if (sl(s, pos, pos + 1) == b"-") != has_sep {
                    return Err(Raises);
                }
                pos += usize::from(has_sep);
                dayno = self.digits(s, pos, 1)?;
                pos += 1;
            }
            weekdate(year, weekno, dayno)?
        } else {
            // YYYYDDD or YYYY-DDD
            if s.len() - pos < 3 {
                return Err(Raises);
            }
            let ordinal = self.digits(s, pos, 3)?;
            pos += 3;
            if ordinal < 1 || ordinal > 365 + i64::from(is_leap(year)) {
                return Err(Raises);
            }
            date_ordinal(year, 1, 1)? + ordinal - 1
        };
        let ymd = from_ordinal(ord)?;
        Ok((ymd, pos))
    }

    fn isotime(&self, t: &[u8]) -> R<([i64; 4], Option<IsoTz>)> {
        let len = t.len();
        let mut c = [0i64; 4];
        let mut tz = None;
        let mut pos = 0;
        let mut comp: i32 = -1;
        if len < 2 {
            return Err(Raises);
        }
        let mut has_sep = false;
        while pos < len && comp < 5 {
            comp += 1;
            if matches!(t[pos], b'-' | b'+' | b'Z' | b'z') {
                tz = Some(self.tzstr(&t[pos..])?);
                pos = len;
                break;
            }
            if comp == 1 && t[pos] == b':' {
                has_sep = true;
                pos += 1;
            } else if comp == 2 && has_sep {
                if sl(t, pos, pos + 1) != b":" {
                    return Err(Raises);
                }
                pos += 1;
            }
            if comp < 3 {
                c[usize::try_from(comp).expect("0..3")] = self.digits(t, pos, 2)?;
                pos += 2;
            }
            if comp == 3 {
                // _FRACTION_REGEX = [\.,]([0-9]+)
                let rest = sl(t, pos, len);
                let n = rest
                    .iter()
                    .skip(1)
                    .take_while(|c| c.is_ascii_digit())
                    .count();
                if !matches!(rest.first(), Some(b'.' | b',')) || n == 0 {
                    continue;
                }
                let us = &rest[1..=n.min(6)];
                let v = us.iter().fold(0i64, |a, &d| a * 10 + i64::from(d - b'0'));
                c[3] = v * 10i64.pow(u32::try_from(6 - us.len()).expect("<= 6"));
                pos += 1 + n;
            }
        }
        if pos < len {
            return Err(Raises);
        }
        if c[0] == 24 && c[1..4].iter().any(|&x| x != 0) {
            return Err(Raises);
        }
        Ok((c, tz))
    }

    fn tzstr(&self, z: &[u8]) -> R<IsoTz> {
        if z == b"Z" || z == b"z" {
            return Ok(IsoTz::Utc);
        }
        if !matches!(z.len(), 3 | 5 | 6) {
            return Err(Raises);
        }
        let mult = match z[0] {
            b'-' => -1,
            b'+' => 1,
            _ => return Err(Raises),
        };
        let hours = self.digits(z, 1, 2)?;
        let minutes = if z.len() == 3 {
            0
        } else {
            let pos = if z[3] == b':' { 4 } else { 3 };
            match self.mode {
                Mode::Strict => {
                    if z.len() != pos + 2 {
                        return Err(Raises);
                    }
                    self.digits(z, pos, 2)?
                }
                Mode::Lenient => int_bytes(&z[pos..])?,
            }
        };
        if hours == 0 && minutes == 0 {
            return Ok(IsoTz::Utc);
        }
        if minutes > 59 || hours > 23 {
            return Err(Raises);
        }
        Ok(IsoTz::Offset(mult * (hours * 60 + minutes) * 60))
    }
}

fn is_leap(y: i64) -> bool {
    cal::is_leap(y.into())
}

/// `date(year, 1, day).toordinal()`; `date()` raises outside 1..9999 (left to dateutil's code).
fn date_ordinal(year: i64, month: i64, day: i64) -> R<i64> {
    cal::ordinal(year, month, day).ok_or(Raises)
}

/// `date.fromordinal(n)`; outside 0001-01-01..9999-12-31 the `timedelta` arithmetic raises
/// `OverflowError` (left to dateutil's code).
fn from_ordinal(n: i64) -> R<[i64; 3]> {
    cal::from_ordinal(n).ok_or(Raises)
}

/// `_calculate_weekdate(year, week, day)` as an ordinal.
fn weekdate(year: i64, week: i64, day: i64) -> R<i64> {
    if !(1..54).contains(&week) || !(1..8).contains(&day) {
        return Err(Raises);
    }
    let jan4 = date_ordinal(year, 1, 4)?;
    // isocalendar()[2] is isoweekday: ordinal 1 (0001-01-01) is a Monday.
    let isoweekday = (jan4 - 1).rem_euclid(7) + 1;
    let week1 = jan4 - (isoweekday - 1);
    if week1 < 1 {
        return Err(Raises);
    }
    Ok(week1 + (week - 1) * 7 + (day - 1))
}

/// `isoparser(sep).isoparse(s)` on the ASCII bytes `s`: the `datetime(...)` arguments, or
/// [`Raises`].
pub fn isoparse(s: &[u8], sep: Option<u8>, mode: Mode) -> R<Iso> {
    let p = P { mode };
    let (date, pos) = p.isodate(s)?;
    let mut fields = [date[0], date[1], date[2], 0, 0, 0, 0];
    let mut tz = None;
    if s.len() > pos {
        if sep.is_none_or(|c| s[pos] == c) {
            let (t, z) = p.isotime(&s[pos + 1..])?;
            fields[3..7].copy_from_slice(&t);
            tz = z;
        } else {
            return Err(Raises);
        }
    }
    let plus_day = fields[3] == 24;
    if plus_day {
        fields[3] = 0;
    }
    Ok(Iso {
        fields,
        tz,
        plus_day,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iso(s: &str, m: Mode) -> R<Iso> {
        isoparse(s.as_bytes(), None, m)
    }

    #[test]
    fn modes() {
        let s = iso("2014- 4-21", Mode::Lenient).unwrap();
        assert_eq!(s.fields[..3], [2014, 4, 21]);
        assert!(iso("2014- 4-21", Mode::Strict).is_err());
        // Any character separates date and time: "1" here, then 21:24:0.
        let s = iso("20200728121240", Mode::Lenient).unwrap();
        assert_eq!(s.fields, [2020, 7, 28, 21, 24, 0, 0]);
        assert!(iso("20200728121240", Mode::Strict).is_err());
        assert_eq!(
            iso("2020-01-01T10:00+-1:00", Mode::Lenient).unwrap().tz,
            Some(IsoTz::Offset(-3600))
        );
        assert!(iso("2020-01-01T10:00+-1:00", Mode::Strict).is_err());
    }

    #[test]
    fn forms() {
        for m in [Mode::Strict, Mode::Lenient] {
            let s = iso("2020-W01-3T24:00Z", m).unwrap();
            assert_eq!(s.fields[..4], [2020, 1, 1, 0]);
            assert!(s.plus_day);
            assert_eq!(s.tz, Some(IsoTz::Utc));
            assert_eq!(iso("2020-060", m).unwrap().fields[..3], [2020, 2, 29]);
            assert_eq!(iso("2021W527", m).unwrap().fields[..3], [2022, 1, 2]);
            let s = iso("2017-03-04T10:20:30.1234567-05:30", m).unwrap();
            assert_eq!(s.fields, [2017, 3, 4, 10, 20, 30, 123_456]);
            assert_eq!(s.tz, Some(IsoTz::Offset(-19800)));
            assert_eq!(iso("2017", m).unwrap().fields[..3], [2017, 1, 1]);
            assert!(iso("2017-0", m).is_err());
            assert!(iso("201", m).is_err());
            assert!(iso("2017-03-04T10:20:30:", m).is_err());
            assert!(iso("2017-03-04T1020:30", m).is_err());
            assert!(iso("0000-W01", m).is_err());
        }
    }
}
