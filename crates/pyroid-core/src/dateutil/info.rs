// Derived from python-dateutil 4.0.0 (BSD-3-Clause), © Gustavo Niemeyer, Paul Ganssle and
// dateutil contributors; modified.
//! `parserinfo`: the word tables the parser consults, compiled from a Python `parserinfo`
//! instance (its `_jump`, `_weekdays`, ... dicts, `TZOFFSET`, `UTCZONE` and settings).

use rustc_hash::FxHashMap;

use super::pystr::{Int, Props};

/// One of parserinfo's `_convert`ed dicts: lowered name -> index.
pub type Table = FxHashMap<Box<str>, i64>;

/// A compiled `parserinfo`.
#[derive(Clone, Debug, Default)]
pub struct ParserInfo {
    pub jump: Table,
    pub weekdays: Table,
    pub months: Table,
    pub hms: Table,
    pub ampm: Table,
    pub utczone: Table,
    pub pertain: Table,
    /// `TZOFFSET` (case-sensitive names); `None` values are left out (`.get` gives `None` too).
    pub tzoffset: FxHashMap<Box<str>, Int>,
    /// `UTCZONE`, the list (`_could_be_tzname` tests `token in self.info.UTCZONE`).
    pub utczone_list: Vec<Box<str>>,
}

/// The parserinfo attributes read on every call (they are plain instance attributes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    pub dayfirst: bool,
    pub yearfirst: bool,
    /// `_year` and `_century` (set from `time.localtime()` when the parserinfo was made).
    pub year: Int,
    pub century: Int,
}

impl Settings {
    /// A default `parserinfo()` made in `year`.
    pub fn default_for(year: Int) -> Self {
        Self {
            dayfirst: false,
            yearfirst: false,
            year,
            century: year / 100 * 100,
        }
    }

    /// `convertyear(year, century_specified)`.
    pub fn convertyear(&self, year: Int, century_specified: bool) -> Int {
        let mut year = year;
        if year < 100 && !century_specified {
            year += self.century;
            if year >= self.year + 50 {
                year -= 100;
            } else if year < self.year - 50 {
                year += 100;
            }
        }
        year
    }
}

fn convert(lists: &[&[&str]]) -> Table {
    let mut t = Table::default();
    for (i, names) in lists.iter().enumerate() {
        for n in *names {
            t.insert(
                n.to_lowercase().into_boxed_str(),
                i64::try_from(i).expect("small"),
            );
        }
    }
    t
}

impl ParserInfo {
    /// dateutil's default `parserinfo()` tables.
    pub fn default_tables() -> Self {
        let jump: Vec<&[&str]> = [
            " ", ".", ",", ";", "-", "/", "'", "at", "on", "and", "ad", "m", "t", "of", "st", "nd",
            "rd", "th",
        ]
        .iter()
        .map(std::slice::from_ref)
        .collect();
        Self {
            jump: convert(&jump),
            weekdays: convert(&[
                &["Mon", "Monday"],
                &["Tue", "Tuesday"],
                &["Wed", "Wednesday"],
                &["Thu", "Thursday"],
                &["Fri", "Friday"],
                &["Sat", "Saturday"],
                &["Sun", "Sunday"],
            ]),
            months: convert(&[
                &["Jan", "January"],
                &["Feb", "February"],
                &["Mar", "March"],
                &["Apr", "April"],
                &["May", "May"],
                &["Jun", "June"],
                &["Jul", "July"],
                &["Aug", "August"],
                &["Sep", "Sept", "September"],
                &["Oct", "October"],
                &["Nov", "November"],
                &["Dec", "December"],
            ]),
            hms: convert(&[
                &["h", "hour", "hours"],
                &["m", "minute", "minutes"],
                &["s", "second", "seconds"],
            ]),
            ampm: convert(&[&["am", "a"], &["pm", "p"]]),
            utczone: convert(&[&["UTC"], &["GMT"], &["Z"], &["z"]]),
            pertain: convert(&[&["of"]]),
            tzoffset: FxHashMap::default(),
            utczone_list: ["UTC", "GMT", "Z", "z"]
                .iter()
                .map(|s| (*s).into())
                .collect(),
        }
    }
}

/// A token with its `name.lower()` (computed once).
#[derive(Clone, Debug)]
pub(crate) struct Word<'a> {
    pub raw: std::borrow::Cow<'a, str>,
    pub low: std::borrow::Cow<'a, str>,
}

impl<'a> Word<'a> {
    pub fn new(raw: std::borrow::Cow<'a, str>, p: Props) -> Self {
        let low = match &raw {
            std::borrow::Cow::Borrowed(s) => p.lower(s),
            std::borrow::Cow::Owned(s) => std::borrow::Cow::Owned(p.lower(s).into_owned()),
        };
        Self { raw, low }
    }

    pub fn is(&self, s: &str) -> bool {
        self.raw == s
    }
}
