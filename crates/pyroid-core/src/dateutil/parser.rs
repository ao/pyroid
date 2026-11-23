// Derived from python-dateutil 4.0.0 (BSD-3-Clause), © Gustavo Niemeyer, Paul Ganssle and
// dateutil contributors; modified.
//! `parser._parse` and its helpers, statement by statement: the token loop,
//! `_parse_numeric_token`, the hms and am/pm logic, `_ymd` and `resolve_ymd`, and
//! `parserinfo.validate`. The datetime itself is built by the caller (`_build_naive` /
//! `_build_tzaware` need Python objects).

use std::borrow::Cow;
use std::cmp::Ordering;

use super::info::{ParserInfo, Settings, Word};
use super::pystr::{Fail, Int, Num, Props, Reason, py_find_dot, py_int, py_len, py_slice};
use super::timelex;

/// `parser._result`: the fields `_parse` fills in (`None` stays `None`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Res {
    pub year: Option<Int>,
    pub month: Option<Int>,
    pub day: Option<Int>,
    pub weekday: Option<Int>,
    pub hour: Option<Int>,
    pub minute: Option<Int>,
    pub second: Option<Int>,
    pub microsecond: Option<Int>,
    pub tzname: Option<String>,
    pub tzoffset: Option<Int>,
    pub ampm: Option<Int>,
    pub century_specified: bool,
}

impl Res {
    /// `len(res)`: how many of the slots are not `None` (`any_unused_tokens` always is).
    pub fn len(&self) -> usize {
        [
            self.year.is_some(),
            self.month.is_some(),
            self.day.is_some(),
            self.weekday.is_some(),
            self.hour.is_some(),
            self.minute.is_some(),
            self.second.is_some(),
            self.microsecond.is_some(),
            self.tzname.is_some(),
            self.tzoffset.is_some(),
            self.ampm.is_some(),
        ]
        .iter()
        .filter(|&&b| b)
        .count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// `_parse`'s keyword arguments (`None` for dayfirst/yearfirst: the parserinfo's).
#[derive(Clone, Copy, Debug, Default)]
pub struct ParseOptions {
    pub dayfirst: Option<bool>,
    pub yearfirst: Option<bool>,
    pub fuzzy: bool,
    pub fuzzy_with_tokens: bool,
    /// `sys.get_int_max_str_digits()` (0: no limit).
    pub max_str_digits: usize,
}

/// What `_parse` returned.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parsed {
    pub res: Res,
    /// The fuzzy tokens (only with `fuzzy_with_tokens`).
    pub skipped: Option<Vec<String>>,
    /// The `decimal` context precision the `Decimal` arithmetic needed (0: none ran). The
    /// caller checks the thread's context against it.
    pub decimal_precision: u32,
    /// The most fraction digits of a number that arithmetic ran on: `d % 1` and `60 * (d % 1)`
    /// have exponents from `-decimal_scale` to 0, which the context's `Emin` and `clamp` must
    /// allow.
    pub decimal_scale: usize,
    /// Whether one of those numbers had a nonzero fraction, so that `60 * (d % 1)` ran (up to
    /// 59.99..., which the context's `Emax` must allow).
    pub decimal_fraction: bool,
}

/// `_parse(timestr, ...)`: `Ok(Some(..))`, `Ok(None)` for dateutil's `(None, None)`, or
/// `Err(reason)` when the Python reference must run instead.
pub fn parse(
    s: &str,
    info: &ParserInfo,
    settings: Settings,
    p: Props,
    opts: ParseOptions,
) -> Result<Option<Parsed>, Reason> {
    let mut l: Vec<Word<'_>> = Vec::with_capacity(16);
    timelex::lex(s, p, |t| l.push(Word::new(t, info, p)));
    let mut cx = Cx {
        info,
        settings,
        p,
        fuzzy: opts.fuzzy || opts.fuzzy_with_tokens,
        max_digits: opts.max_str_digits,
        precision: 0,
        scale: 0,
        fraction: false,
    };
    let dayfirst = opts.dayfirst.unwrap_or(settings.dayfirst);
    let yearfirst = opts.yearfirst.unwrap_or(settings.yearfirst);
    let mut res = Res::default();
    let mut skipped = Vec::new();
    match cx.run(&mut l, &mut res, &mut skipped, yearfirst, dayfirst) {
        Ok(()) => {}
        Err(Fail::Invalid) => return Ok(None),
        Err(Fail::Reference(r)) => return Err(r),
    }
    validate(info, settings, p, &mut res);
    let skipped = opts
        .fuzzy_with_tokens
        .then(|| recombine_skipped(&l, &skipped));
    Ok(Some(Parsed {
        res,
        skipped,
        decimal_precision: cx.precision,
        decimal_scale: cx.scale,
        decimal_fraction: cx.fraction,
    }))
}

/// `parserinfo.validate(res)`.
fn validate(info: &ParserInfo, settings: Settings, p: Props, res: &mut Res) {
    if let Some(y) = res.year {
        res.year = Some(settings.convertyear(y, res.century_specified));
    }
    let tzname = res.tzname.as_deref();
    if (res.tzoffset == Some(0) && tzname.is_none_or(str::is_empty))
        || matches!(tzname, Some("Z" | "z"))
    {
        res.tzname = Some("UTC".to_owned());
        res.tzoffset = Some(0);
    } else if res.tzoffset != Some(0)
        && tzname.is_some_and(|n| !n.is_empty() && info.word(n, p).utczone())
    {
        res.tzoffset = Some(0);
    }
}

/// `_recombine_skipped(tokens, skipped_idxs)`.
fn recombine_skipped(l: &[Word<'_>], idxs: &[usize]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (i, &idx) in idxs.iter().enumerate() {
        if i > 0 && idx.checked_sub(1) == Some(idxs[i - 1]) {
            out.last_mut().expect("i > 0").push_str(&l[idx].raw);
        } else {
            out.push(l[idx].raw.clone().into_owned());
        }
    }
    out
}

/// `_ymd.append`'s labels (dateutil never passes `'D'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Label {
    Y,
    M,
}

/// A value `_ymd.append` receives.
enum YVal<'t> {
    Str(&'t str),
    Dec(&'t Num<'t>),
    Int(Int),
}

/// `_ymd`.
#[derive(Debug, Default)]
struct Ymd {
    /// The first four values inline, the rest in `extra` (dateutil raises past three anyway).
    head: [Int; 4],
    extra: Vec<Int>,
    n: usize,
    century_specified: bool,
    dstridx: Option<usize>,
    mstridx: Option<usize>,
    ystridx: Option<usize>,
}

/// `calendar.monthrange(year, month)[1]`; `IllegalMonthError` (a ValueError) outside 1..12.
fn month_days(year: Int, month: Int) -> Result<Int, Fail> {
    const DAYS: [Int; 13] = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    if !(1..=12).contains(&month) {
        return Err(Fail::Invalid);
    }
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let m = usize::try_from(month).expect("1..=12");
    Ok(DAYS[m] + Int::from(month == 2 && leap))
}

impl Ymd {
    fn len(&self) -> usize {
        self.n
    }

    /// `self[i]`.
    fn get(&self, i: usize) -> Int {
        if i < 4 {
            self.head[i]
        } else {
            self.extra[i - 4]
        }
    }

    /// `self[i]` with Python's negative indexing.
    fn at(&self, i: isize) -> Int {
        let n = isize::try_from(self.n).expect("small");
        let j = if i < 0 { i + n } else { i };
        self.get(usize::try_from(j).expect("in range"))
    }

    fn could_be_day(&self, value: &Num<'_>) -> Result<bool, Fail> {
        if self.dstridx.is_some() {
            return Ok(false);
        }
        // `1 <= value <= monthrange(...)[1]`: monthrange only runs when `1 <= value`.
        if value.cmp_int(1) == Ordering::Less {
            return Ok(false);
        }
        let Some(mi) = self.mstridx else {
            return Ok(value.cmp_int(31) != Ordering::Greater);
        };
        let month = self.get(mi);
        let year = self.ystridx.map_or(2000, |yi| self.get(yi));
        let days = month_days(year, month)?;
        Ok(value.cmp_int(days) != Ordering::Greater)
    }

    fn append(&mut self, cx: &Cx<'_>, val: YVal<'_>, label: Option<Label>) -> Result<(), Fail> {
        let mut label = label;
        let big = match &val {
            YVal::Str(s) => cx.p.str_isdigit(s) && py_len(s) > 2,
            YVal::Dec(d) => d.cmp_int(100) == Ordering::Greater,
            YVal::Int(i) => *i > 100,
        };
        if big {
            self.century_specified = true;
            if !matches!(label, None | Some(Label::Y)) {
                return Err(Fail::Invalid);
            }
            label = Some(Label::Y);
        }
        let n = match val {
            YVal::Str(s) => py_int(cx.p, s, cx.max_digits)?,
            YVal::Dec(d) => d.trunc()?,
            YVal::Int(i) => i,
        };
        if self.n < 4 {
            self.head[self.n] = n;
        } else {
            self.extra.push(n);
        }
        self.n += 1;
        let idx = Some(self.n - 1);
        match label {
            Some(Label::M) => {
                if self.mstridx.is_some() {
                    return Err(Fail::Invalid);
                }
                self.mstridx = idx;
            }
            Some(Label::Y) => {
                if self.ystridx.is_some() {
                    return Err(Fail::Invalid);
                }
                self.ystridx = idx;
            }
            None => {}
        }
        Ok(())
    }

    /// `resolve_ymd(yearfirst, dayfirst)` -> (year, month, day).
    #[allow(clippy::similar_names)]
    fn resolve(&self, yearfirst: bool, dayfirst: bool) -> Result<[Option<Int>; 3], Fail> {
        let len = self.len();
        let strids: Vec<(usize, usize)> = [self.ystridx, self.mstridx, self.dstridx]
            .iter()
            .enumerate()
            .filter_map(|(k, v)| v.map(|v| (k, v)))
            .collect();
        if (len == strids.len() && !strids.is_empty()) || (len == 3 && strids.len() == 2) {
            // _resolve_from_stridxs
            let mut out = [None; 3];
            let mut slots = [None; 3];
            for &(k, v) in &strids {
                slots[k] = Some(v);
            }
            if len == 3 && strids.len() == 2 {
                let used: Vec<usize> = strids.iter().map(|&(_, v)| v).collect();
                let missing = (0..3).find(|x| !used.contains(x)).expect("one missing");
                let key = (0..3).find(|&k| slots[k].is_none()).expect("one key");
                slots[key] = Some(missing);
            }
            for k in 0..3 {
                if let Some(v) = slots[k] {
                    out[k] = Some(self.get(v));
                }
            }
            return Ok(out);
        }
        let (mut year, mut month, mut day) = (None, None, None);
        let mstridx = self.mstridx;
        let s = |i: usize| self.get(i);
        if len > 3 {
            return Err(Fail::Invalid);
        } else if len == 1 || (mstridx.is_some() && len == 2) {
            let other = if let Some(mi) = mstridx {
                month = Some(s(mi));
                self.at(isize::try_from(mi).expect("small") - 1)
            } else {
                s(0)
            };
            if len > 1 || mstridx.is_none() {
                if other > 31 {
                    year = Some(other);
                } else {
                    day = Some(other);
                }
            }
        } else if len == 2 {
            if s(0) > 31 {
                (year, month) = (Some(s(0)), Some(s(1)));
            } else if s(1) > 31 {
                (month, year) = (Some(s(0)), Some(s(1)));
            } else if dayfirst && s(1) <= 12 {
                (day, month) = (Some(s(0)), Some(s(1)));
            } else {
                (month, day) = (Some(s(0)), Some(s(1)));
            }
        } else if len == 3 {
            let (a, b, c) = (Some(s(0)), Some(s(1)), Some(s(2)));
            match mstridx {
                Some(0) => {
                    if s(1) > 31 {
                        (month, year, day) = (a, b, c);
                    } else {
                        (month, day, year) = (a, b, c);
                    }
                }
                Some(1) => {
                    if s(0) > 31 || (yearfirst && s(2) <= 31) {
                        (year, month, day) = (a, b, c);
                    } else {
                        (day, month, year) = (a, b, c);
                    }
                }
                Some(2) => {
                    if s(1) > 31 {
                        (day, year, month) = (a, b, c);
                    } else {
                        (year, day, month) = (a, b, c);
                    }
                }
                _ => {
                    if s(0) > 31
                        || self.ystridx == Some(0)
                        || (yearfirst && s(1) <= 12 && s(2) <= 31)
                    {
                        if dayfirst && s(2) <= 12 {
                            (year, day, month) = (a, b, c);
                        } else {
                            (year, month, day) = (a, b, c);
                        }
                    } else if s(0) > 12 || (dayfirst && s(1) <= 12) {
                        (day, month, year) = (a, b, c);
                    } else {
                        (month, day, year) = (a, b, c);
                    }
                }
            }
        }
        Ok([year, month, day])
    }
}

struct Cx<'i> {
    info: &'i ParserInfo,
    settings: Settings,
    p: Props,
    fuzzy: bool,
    max_digits: usize,
    precision: u32,
    scale: usize,
    fraction: bool,
}

/// `tokens[i]`, or `IndexError` (dateutil returns `None` for it).
fn tok<'l, 'a>(l: &'l [Word<'a>], i: usize) -> Result<&'l Word<'a>, Fail> {
    l.get(i).ok_or(Fail::Invalid)
}

impl Cx<'_> {
    fn jump(w: &Word<'_>) -> bool {
        w.w.jump()
    }

    fn weekday(w: &Word<'_>) -> Option<Int> {
        w.w.weekday()
    }

    fn month(w: &Word<'_>) -> Option<Int> {
        w.w.month().map(|m| m + 1)
    }

    fn hms(w: &Word<'_>) -> Option<Int> {
        w.w.hms()
    }

    fn ampm(w: &Word<'_>) -> Option<Int> {
        w.w.ampm()
    }

    fn pertain(w: &Word<'_>) -> bool {
        w.w.pertain()
    }

    fn utczone(w: &Word<'_>) -> bool {
        w.w.utczone()
    }

    /// `info.tzoffset(name)`: 0 when `name` (not lowered) is a key of `_utczone`.
    fn tzoffset(&self, name: &str) -> Option<Int> {
        if self.info.utczone.contains_key(name) {
            return Some(0);
        }
        self.info.tzoffset.get(name).copied()
    }

    fn int(&self, s: &str) -> Result<Int, Fail> {
        py_int(self.p, s, self.max_digits)
    }

    /// `_to_decimal(val)`: `ValueError` unless finite.
    fn dec<'s>(&self, s: &'s str) -> Result<Num<'s>, Fail> {
        Num::parse(self.p, s).ok_or(Fail::Invalid)
    }

    fn note_precision(&mut self, d: &Num<'_>) -> Result<(), Fail> {
        let need = d.precision_needed()?;
        self.precision = self.precision.max(need);
        self.scale = self.scale.max(d.frac.len());
        self.fraction |= d.has_fraction();
        Ok(())
    }

    /// `_parse_min_sec(value)` -> (minute, second).
    fn min_sec(&mut self, d: &Num<'_>) -> Result<(Int, Option<Int>), Fail> {
        let minute = d.trunc()?;
        self.note_precision(d)?;
        let second = if d.has_fraction() {
            Some(d.frac_times_60()?)
        } else {
            None
        };
        Ok((minute, second))
    }

    /// `_parsems(value)` -> (seconds, microseconds).
    fn parsems(&self, value: &str) -> Result<(Int, Int), Fail> {
        if !value.contains('.') {
            return Ok((self.int(value)?, 0));
        }
        let mut parts = value.split('.');
        let (i, f) = (parts.next().expect("one"), parts.next().expect("two"));
        if parts.next().is_some() {
            return Err(Fail::Invalid); // too many values to unpack
        }
        let mut f6: String = f.to_owned();
        let flen = py_len(f);
        if flen < 6 {
            f6.extend(std::iter::repeat_n('0', 6 - flen));
        }
        let f6 = py_slice(&f6, 0, Some(6));
        Ok((self.int(i)?, self.int(f6)?))
    }

    /// `_could_be_tzname(hour, tzname, tzoffset, token)`.
    fn could_be_tzname(&self, res: &Res, check_offset: bool, token: &str) -> bool {
        res.hour.is_some()
            && res.tzname.is_none()
            && (!check_offset || res.tzoffset.is_none())
            && py_len(token) <= 5
            && (token.bytes().all(|b| b.is_ascii_uppercase())
                || self.info.utczone_list.iter().any(|z| &**z == token))
    }

    /// `_ampm_valid(hour, ampm, fuzzy)`.
    fn ampm_valid(&self, res: &Res) -> Result<bool, Fail> {
        let mut ok = true;
        if self.fuzzy && res.ampm.is_some() {
            ok = false;
        }
        match res.hour {
            None => {
                if self.fuzzy {
                    ok = false;
                } else {
                    return Err(Fail::Invalid);
                }
            }
            Some(h) if !(0..=12).contains(&h) => {
                if self.fuzzy {
                    ok = false;
                } else {
                    return Err(Fail::Invalid);
                }
            }
            Some(_) => {}
        }
        Ok(ok)
    }

    /// `_find_hms_idx(idx, tokens, info, allow_jump=True)`.
    fn find_hms_idx(idx: usize, l: &[Word<'_>]) -> Option<usize> {
        let len_l = l.len();
        if idx + 1 < len_l && Self::hms(&l[idx + 1]).is_some() {
            Some(idx + 1)
        } else if idx + 2 < len_l && l[idx + 1].is(" ") && Self::hms(&l[idx + 2]).is_some() {
            Some(idx + 2)
        } else if idx > 0 && Self::hms(&l[idx - 1]).is_some() {
            Some(idx - 1)
        } else if 1 < idx
            && idx == len_l - 1
            && l[idx - 1].is(" ")
            && Self::hms(&l[idx - 2]).is_some()
        {
            Some(idx - 2)
        } else {
            None
        }
    }

    /// `_assign_hms(res, value_repr, hms)`.
    fn assign_hms(&mut self, res: &mut Res, value_repr: &str, hms: Int) -> Result<(), Fail> {
        let d = self.dec(value_repr)?;
        if hms == 0 {
            res.hour = Some(d.trunc()?);
            self.note_precision(&d)?;
            if d.has_fraction() {
                res.minute = Some(d.frac_times_60()?);
            }
        } else if hms == 1 {
            let (m, s) = self.min_sec(&d)?;
            res.minute = Some(m);
            res.second = s;
        } else if hms == 2 {
            let (s, us) = self.parsems(value_repr)?;
            res.second = Some(s);
            res.microsecond = Some(us);
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn numeric_token(
        &mut self,
        l: &[Word<'_>],
        mut idx: usize,
        ymd: &mut Ymd,
        res: &mut Res,
    ) -> Result<usize, Fail> {
        let value_repr: &str = &l[idx].raw;
        // `_to_decimal`; "Unknown numeric token" is a ValueError too.
        let value = self.dec(value_repr)?;
        let len_li = py_len(value_repr);
        let len_l = l.len();
        if ymd.len() == 3
            && matches!(len_li, 2 | 4)
            && res.hour.is_none()
            && (idx + 1 >= len_l || (!l[idx + 1].is(":") && Self::hms(&l[idx + 1]).is_none()))
        {
            // 19990101T23[59]
            let s = value_repr;
            res.hour = Some(self.int(py_slice(s, 0, Some(2)))?);
            if len_li == 4 {
                res.minute = Some(self.int(py_slice(s, 2, None))?);
            }
        } else if len_li == 6 || (len_li > 6 && py_find_dot(value_repr) == Some(6)) {
            // YYMMDD or HHMMSS[.ss]
            let s = value_repr;
            if ymd.len() == 0 && !s.contains('.') {
                ymd.append(self, YVal::Str(py_slice(s, 0, Some(2))), None)?;
                ymd.append(self, YVal::Str(py_slice(s, 2, Some(4))), None)?;
                ymd.append(self, YVal::Str(py_slice(s, 4, None)), None)?;
            } else {
                res.hour = Some(self.int(py_slice(s, 0, Some(2)))?);
                res.minute = Some(self.int(py_slice(s, 2, Some(4)))?);
                let (sec, us) = self.parsems(py_slice(s, 4, None))?;
                res.second = Some(sec);
                res.microsecond = Some(us);
            }
        } else if matches!(len_li, 8 | 12 | 14) {
            // YYYYMMDD
            let s = value_repr;
            ymd.append(self, YVal::Str(py_slice(s, 0, Some(4))), Some(Label::Y))?;
            ymd.append(self, YVal::Str(py_slice(s, 4, Some(6))), None)?;
            ymd.append(self, YVal::Str(py_slice(s, 6, Some(8))), None)?;
            if len_li > 8 {
                res.hour = Some(self.int(py_slice(s, 8, Some(10)))?);
                res.minute = Some(self.int(py_slice(s, 10, Some(12)))?);
                if len_li > 12 {
                    res.second = Some(self.int(py_slice(s, 12, None))?);
                }
            }
        } else if let Some(hms_idx) = Self::find_hms_idx(idx, l) {
            // HH[ ]h or MM[ ]m or SS[.ss][ ]s; _parse_hms
            let hms = if hms_idx > idx {
                let h = Self::hms(&l[hms_idx]).expect("found");
                idx = hms_idx;
                h
            } else {
                Self::hms(&l[hms_idx]).expect("found") + 1
            };
            self.assign_hms(res, value_repr, hms)?;
        } else if idx + 2 < len_l && l[idx + 1].is(":") {
            // HH:MM[:SS[.ss]]
            res.hour = Some(value.trunc()?);
            let v2 = self.dec(&l[idx + 2].raw)?;
            let (m, s) = self.min_sec(&v2)?;
            res.minute = Some(m);
            res.second = s;
            if idx + 4 < len_l && l[idx + 3].is(":") {
                let (sec, us) = self.parsems(&l[idx + 4].raw)?;
                res.second = Some(sec);
                res.microsecond = Some(us);
                idx += 2;
            }
            idx += 2;
        } else if idx + 1 < len_l && matches!(&*l[idx + 1].raw, "-" | "/" | ".") {
            let sep: &str = &l[idx + 1].raw;
            ymd.append(self, YVal::Str(value_repr), None)?;
            if idx + 2 < len_l && !Self::jump(&l[idx + 2]) {
                if self.p.str_isdigit(&l[idx + 2].raw) {
                    // 01-01[-01]
                    ymd.append(self, YVal::Str(&l[idx + 2].raw), None)?;
                } else {
                    // 01-Jan[-01]
                    match Self::month(&l[idx + 2]) {
                        Some(m) => ymd.append(self, YVal::Int(m), Some(Label::M))?,
                        None => return Err(Fail::Invalid),
                    }
                }
                if idx + 3 < len_l && l[idx + 3].is(sep) {
                    // We have three members
                    let t4 = tok(l, idx + 4)?;
                    match Self::month(t4) {
                        Some(m) => ymd.append(self, YVal::Int(m), Some(Label::M))?,
                        None => ymd.append(self, YVal::Str(&t4.raw), None)?,
                    }
                    idx += 2;
                }
                idx += 1;
            }
            idx += 1;
        } else if idx + 1 >= len_l || Self::jump(&l[idx + 1]) {
            match l.get(idx + 2).and_then(|w| Self::ampm(w)) {
                Some(ap) => {
                    // 12 am
                    let hour = value.trunc()?;
                    res.hour = Some(adjust_ampm(hour, ap));
                    idx += 1;
                }
                None => {
                    // Year, month or day
                    ymd.append(self, YVal::Dec(&value), None)?;
                }
            }
            idx += 1;
        } else if let Some(ap) = Self::ampm(&l[idx + 1])
            .filter(|_| value.cmp_int(0) != Ordering::Less && value.cmp_int(24) == Ordering::Less)
        {
            // 12am
            let hour = value.trunc()?;
            res.hour = Some(adjust_ampm(hour, ap));
            idx += 1;
        } else if ymd.could_be_day(&value)? {
            ymd.append(self, YVal::Dec(&value), None)?;
        } else if !self.fuzzy {
            return Err(Fail::Invalid);
        }
        Ok(idx)
    }

    #[allow(clippy::too_many_lines)]
    fn run(
        &mut self,
        l: &mut [Word<'_>],
        res: &mut Res,
        skipped: &mut Vec<usize>,
        yearfirst: bool,
        dayfirst: bool,
    ) -> Result<(), Fail> {
        let mut ymd = Ymd::default();
        let len_l = l.len();
        let mut i = 0;
        while i < len_l {
            if float_ok(self.p, &l[i].raw) {
                // Numeric token
                i = self.numeric_token(l, i, &mut ymd, res)?;
            } else if let Some(wd) = Self::weekday(&l[i]) {
                res.weekday = Some(wd);
            } else if let Some(m) = Self::month(&l[i]) {
                ymd.append(self, YVal::Int(m), Some(Label::M))?;
                if i + 1 < len_l {
                    if matches!(&*l[i + 1].raw, "-" | "/") {
                        // Jan-01[-99]
                        let sep = l[i + 1].raw.clone();
                        ymd.append(self, YVal::Str(&tok(l, i + 2)?.raw), None)?;
                        if i + 3 < len_l && l[i + 3].raw == sep {
                            // Jan-01-99
                            ymd.append(self, YVal::Str(&tok(l, i + 4)?.raw), None)?;
                            i += 2;
                        }
                        i += 2;
                    } else if i + 4 < len_l
                        && l[i + 1].is(" ")
                        && l[i + 3].is(" ")
                        && Self::pertain(&l[i + 2])
                    {
                        // Jan of 01
                        if self.p.str_isdigit(&l[i + 4].raw) {
                            let value = self.int(&l[i + 4].raw)?;
                            let year = self.settings.convertyear(value, false).to_string();
                            ymd.append(self, YVal::Str(&year), Some(Label::Y))?;
                        }
                        i += 4;
                    }
                }
            } else if let Some(ap) = Self::ampm(&l[i]) {
                if self.ampm_valid(res)? {
                    res.hour = Some(adjust_ampm(res.hour.expect("valid"), ap));
                    res.ampm = Some(ap);
                } else if self.fuzzy {
                    skipped.push(i);
                }
            } else if self.could_be_tzname(res, true, &l[i].raw) {
                let name = l[i].raw.clone().into_owned();
                res.tzoffset = self.tzoffset(&name);
                res.tzname = Some(name);
                // GMT+3 / BRST+3: "my time +3 is GMT", so the sign is flipped.
                if i + 1 < len_l && matches!(&*l[i + 1].raw, "+" | "-") {
                    let flipped = if l[i + 1].is("+") { "-" } else { "+" };
                    l[i + 1] = Word::new(Cow::Borrowed(flipped), self.info, self.p);
                    res.tzoffset = None;
                    if Self::utczone(&l[i]) {
                        res.tzname = None;
                    }
                }
            } else if res.hour.is_some() && matches!(&*l[i].raw, "+" | "-") {
                let signal: Int = if l[i].is("+") { 1 } else { -1 };
                let t1 = tok(l, i + 1)?;
                let len_li = py_len(&t1.raw);
                let (hour_offset, min_offset);
                if len_li == 4 {
                    // -0300
                    hour_offset = self.int(py_slice(&t1.raw, 0, Some(2)))?;
                    min_offset = self.int(py_slice(&t1.raw, 2, None))?;
                } else if i + 2 < len_l && l[i + 2].is(":") {
                    // -03:00
                    hour_offset = self.int(&t1.raw)?;
                    min_offset = self.int(&tok(l, i + 3)?.raw)?;
                    i += 2;
                } else if len_li <= 2 {
                    // -[0]3
                    hour_offset = self.int(py_slice(&t1.raw, 0, Some(2)))?;
                    min_offset = 0;
                } else {
                    return Err(Fail::Invalid);
                }
                let secs = hour_offset
                    .checked_mul(3600)
                    .and_then(|h| min_offset.checked_mul(60).and_then(|m| h.checked_add(m)))
                    .and_then(|x| x.checked_mul(signal))
                    .ok_or(Fail::Reference(Reason::BigNumber))?;
                res.tzoffset = Some(secs);
                // Look for a timezone name between parenthesis: -0300 (BRST)
                if i + 5 < len_l
                    && Self::jump(&l[i + 2])
                    && l[i + 3].is("(")
                    && l[i + 5].is(")")
                    && 3 <= py_len(&l[i + 4].raw)
                    && self.could_be_tzname(res, false, &l[i + 4].raw)
                {
                    res.tzname = Some(l[i + 4].raw.clone().into_owned());
                    i += 4;
                }
                i += 1;
            } else if !(Self::jump(&l[i]) || self.fuzzy) {
                return Err(Fail::Invalid);
            } else {
                skipped.push(i);
            }
            i += 1;
        }
        let [year, month, day] = ymd.resolve(yearfirst, dayfirst)?;
        res.century_specified = ymd.century_specified;
        res.year = year;
        res.month = month;
        res.day = day;
        Ok(())
    }
}

/// `_adjust_ampm(hour, ampm)`.
fn adjust_ampm(hour: Int, ampm: Int) -> Int {
    if hour < 12 && ampm == 1 {
        hour + 12
    } else if hour == 12 && ampm == 0 {
        0
    } else {
        hour
    }
}

/// Whether `float(token)` succeeds: `Nd+(.Nd+)?` or (any case) `inf`, `infinity`, `nan`.
fn float_ok(p: Props, t: &str) -> bool {
    let b = t.as_bytes();
    match b.first() {
        Some(c) if c.is_ascii_digit() || !c.is_ascii() => {}
        Some(b'i' | b'I' | b'n' | b'N') => {
            return t.eq_ignore_ascii_case("inf")
                || t.eq_ignore_ascii_case("infinity")
                || t.eq_ignore_ascii_case("nan");
        }
        _ => return false,
    }
    let mut seen_dot = false;
    let mut digits_before = 0usize;
    let mut digits_after = 0usize;
    for c in t.chars() {
        if c == '.' {
            if seen_dot || digits_before == 0 {
                return false;
            }
            seen_dot = true;
        } else if p.is_decimal(c) {
            if seen_dot {
                digits_after += 1;
            } else {
                digits_before += 1;
            }
        } else {
            return false;
        }
    }
    digits_before > 0 && (!seen_dot || digits_after > 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ftfy::UnicodeVersion;

    fn run(s: &str, fuzzy: bool) -> Option<Res> {
        let info = ParserInfo::default_tables();
        let p = Props::new(UnicodeVersion::V15_0);
        let opts = ParseOptions {
            fuzzy,
            max_str_digits: 4300,
            ..ParseOptions::default()
        };
        parse(s, &info, Settings::default_for(2026), p, opts)
            .expect("no reference")
            .map(|x| x.res)
    }

    fn ymdhms(r: &Res) -> [Option<Int>; 7] {
        [
            r.year,
            r.month,
            r.day,
            r.hour,
            r.minute,
            r.second,
            r.microsecond,
        ]
    }

    #[test]
    fn common_forms() {
        let r = run("2003-09-25T10:49:41.5-03:00", false).unwrap();
        assert_eq!(
            ymdhms(&r),
            [
                Some(2003),
                Some(9),
                Some(25),
                Some(10),
                Some(49),
                Some(41),
                Some(500_000)
            ]
        );
        assert_eq!(r.tzoffset, Some(-10800));
        let r = run("Thu Sep 25 10:36:28 BRST 2003", false).unwrap();
        assert_eq!(
            ymdhms(&r),
            [
                Some(2003),
                Some(9),
                Some(25),
                Some(10),
                Some(36),
                Some(28),
                Some(0)
            ]
        );
        assert_eq!(r.tzname.as_deref(), Some("BRST"));
        assert_eq!(r.weekday, Some(3));
        let r = run("10/09/03", false).unwrap();
        assert_eq!(ymdhms(&r)[..3], [Some(2003), Some(10), Some(9)]);
        let r = run("Jan of 01", false).unwrap();
        assert_eq!(ymdhms(&r)[..3], [Some(2001), Some(1), None]);
        let r = run("10h36m28.5s", false).unwrap();
        assert_eq!(
            ymdhms(&r)[3..],
            [Some(10), Some(36), Some(28), Some(500_000)]
        );
        let r = run("5:50 A.M. on June 13, 1990", false).unwrap();
        assert_eq!(
            ymdhms(&r)[..5],
            [Some(1990), Some(6), Some(13), Some(5), Some(50)]
        );
        let r = run("12 pm", false).unwrap();
        assert_eq!(r.hour, Some(12));
        let r = run("Tue, 02 Jan 2024 10:00:00 GMT", false).unwrap();
        assert_eq!(r.tzname.as_deref(), Some("GMT"));
        assert_eq!(r.tzoffset, Some(0));
        let r = run("10:00 GMT+3", false).unwrap();
        assert_eq!(r.tzname, None);
        assert_eq!(r.tzoffset, Some(-10800));
    }

    #[test]
    fn failures() {
        assert!(run("foo", false).is_none());
        assert!(run("am", false).is_none());
        assert!(run("inf", true).is_none());
        assert!(run("13:00 pm", false).is_none());
        assert!(run(" ", false).unwrap().is_empty());
        let r = run("Today is January 1, 2047 at 8:21:00AM", true).unwrap();
        assert_eq!(
            ymdhms(&r)[..6],
            [Some(2047), Some(1), Some(1), Some(8), Some(21), Some(0)]
        );
    }

    #[test]
    fn fuzzy_tokens() {
        let info = ParserInfo::default_tables();
        let p = Props::new(UnicodeVersion::V15_0);
        let opts = ParseOptions {
            fuzzy_with_tokens: true,
            max_str_digits: 4300,
            ..ParseOptions::default()
        };
        let s = "Today is January 1, 2047 at 8:21:00AM";
        let x = parse(s, &info, Settings::default_for(2026), p, opts)
            .unwrap()
            .unwrap();
        // The docstring shows three tokens; the code gives four (the space after the comma).
        assert_eq!(x.skipped.unwrap(), ["Today is ", " ", " ", "at "]);
    }
}
