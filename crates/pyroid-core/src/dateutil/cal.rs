//! Proleptic Gregorian calendar arithmetic, as `datetime.date` does it (years 1..=9999).

/// Whether `y` is a leap year (`calendar.isleap`).
pub fn is_leap(y: i64) -> bool {
    y.rem_euclid(4) == 0 && (y.rem_euclid(100) != 0 || y.rem_euclid(400) == 0)
}

/// `calendar.monthrange(year, month)[1]` (`None`: `IllegalMonthError`).
pub fn month_days(year: i64, month: i64) -> Option<i64> {
    const DAYS: [i64; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=12).contains(&month) {
        return None;
    }
    let m = usize::try_from(month).ok()?;
    Some(DAYS[m] + i64::from(month == 2 && is_leap(year)))
}

/// `date(y, m, d).toordinal()` for a valid date in 1..=9999.
pub fn ordinal(year: i64, month: i64, day: i64) -> Option<i64> {
    const BEFORE: [i64; 13] = [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334];
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return None;
    }
    let md = month_days(year, month)?;
    if !(1..=md).contains(&day) {
        return None;
    }
    let y = year - 1;
    let m = usize::try_from(month).ok()?;
    Some(
        y * 365 + y / 4 - y / 100
            + y / 400
            + BEFORE[m]
            + i64::from(month > 2 && is_leap(year))
            + day,
    )
}

/// The last ordinal `datetime.date` allows (9999-12-31).
pub const MAX_ORDINAL: i64 = 3_652_059;

/// `date.fromordinal(n)` as (year, month, day), for 1..=MAX_ORDINAL.
pub fn from_ordinal(n: i64) -> Option<[i64; 3]> {
    if !(1..=MAX_ORDINAL).contains(&n) {
        return None;
    }
    let (mut lo, mut hi) = (1i64, 9999i64);
    while lo < hi {
        let mid = (lo + hi + 1) / 2;
        if ordinal(mid, 1, 1)? <= n {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    let year = lo;
    let mut doy = n - ordinal(year, 1, 1)? + 1;
    let mut month = 1;
    loop {
        let md = month_days(year, month)?;
        if doy <= md {
            break;
        }
        doy -= md;
        month += 1;
    }
    Some([year, month, doy])
}

/// `date.weekday()` (Monday is 0) of a valid ordinal.
pub fn weekday(ordinal: i64) -> i64 {
    (ordinal - 1).rem_euclid(7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        assert_eq!(ordinal(1, 1, 1), Some(1));
        assert_eq!(ordinal(9999, 12, 31), Some(MAX_ORDINAL));
        assert_eq!(ordinal(2024, 2, 30), None);
        for n in [1, 59, 60, 365, 366, 730_120, 738_000, MAX_ORDINAL] {
            let [y, m, d] = from_ordinal(n).unwrap();
            assert_eq!(ordinal(y, m, d), Some(n));
        }
        assert_eq!(weekday(ordinal(2024, 1, 1).unwrap()), 0);
        assert_eq!(month_days(1900, 2), Some(28));
        assert_eq!(month_days(2000, 2), Some(29));
        assert_eq!(month_days(2000, 13), None);
    }
}
