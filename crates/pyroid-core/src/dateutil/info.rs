// Derived from python-dateutil 4.0.0 (BSD-3-Clause), © Gustavo Niemeyer, Paul Ganssle and
// dateutil contributors; modified.
//! `parserinfo`: the word tables the parser consults, compiled from a Python `parserinfo`
//! instance (its `_jump`, `_weekdays`, ... dicts, `TZOFFSET`, `UTCZONE` and settings).

use rustc_hash::FxHashMap;

use super::pystr::{Int, Props};

/// One of parserinfo's `_convert`ed dicts: lowered name -> index.
pub type Table = FxHashMap<Box<str>, i64>;

/// What the tables say about one word (`name.lower()` looked up in each of them).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordInfo {
    flags: u8,
    /// weekday, month, hms, ampm values ([`NONE`]: not in that table).
    values: [i16; 4],
}

/// No value: the word is not in that table.
const NONE: i16 = i16::MIN;
const JUMP: u8 = 1;
const PERTAIN: u8 = 2;
const UTCZONE: u8 = 4;

impl Default for WordInfo {
    fn default() -> Self {
        Self {
            flags: 0,
            values: [NONE; 4],
        }
    }
}

impl WordInfo {
    pub fn jump(self) -> bool {
        self.flags & JUMP != 0
    }
    pub fn pertain(self) -> bool {
        self.flags & PERTAIN != 0
    }
    pub fn utczone(self) -> bool {
        self.flags & UTCZONE != 0
    }
    fn value(self, i: usize) -> Option<i64> {
        let v = self.values[i];
        (v != NONE).then_some(i64::from(v))
    }
    pub fn weekday(self) -> Option<i64> {
        self.value(0)
    }
    pub fn month(self) -> Option<i64> {
        self.value(1)
    }
    pub fn hms(self) -> Option<i64> {
        self.value(2)
    }
    pub fn ampm(self) -> Option<i64> {
        self.value(3)
    }
}

/// The largest table value the kernel takes (others make the caller run dateutil's code).
pub const MAX_TABLE_VALUE: i64 = i16::MAX as i64;

/// A compiled `parserinfo`.
#[derive(Clone, Debug)]
pub struct ParserInfo {
    /// `_utczone` (`tzoffset(name)` looks the name up without lowering it).
    pub utczone: Table,
    /// `TZOFFSET` (case-sensitive names); `None` values are left out (`.get` gives `None` too).
    pub tzoffset: FxHashMap<Box<str>, Int>,
    /// `UTCZONE`, the list (`_could_be_tzname` tests `token in self.info.UTCZONE`).
    pub utczone_list: Vec<Box<str>>,
    /// Every key of the seven tables, with what each table says about it.
    words: FxHashMap<Box<str>, WordInfo>,
    /// [`Self::words`] for the one-character ASCII words, by byte (already lowered).
    ascii1: [WordInfo; 128],
    /// Whether a key starts with an ASCII digit (else digit tokens need no lookup).
    digit_keys: bool,
}

/// parserinfo's seven `_convert`ed tables.
#[derive(Clone, Debug, Default)]
pub struct Tables {
    pub jump: Table,
    pub weekdays: Table,
    pub months: Table,
    pub hms: Table,
    pub ampm: Table,
    pub utczone: Table,
    pub pertain: Table,
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
    /// The compiled form of `tables`, `TZOFFSET` and `UTCZONE`.
    pub fn new(t: Tables, tzoffset: FxHashMap<Box<str>, Int>, utczone_list: Vec<Box<str>>) -> Self {
        let mut words: FxHashMap<Box<str>, WordInfo> = FxHashMap::default();
        for (table, flag) in [
            (&t.jump, JUMP),
            (&t.pertain, PERTAIN),
            (&t.utczone, UTCZONE),
        ] {
            for k in table.keys() {
                words.entry(k.clone()).or_default().flags |= flag;
            }
        }
        for (i, table) in [&t.weekdays, &t.months, &t.hms, &t.ampm]
            .into_iter()
            .enumerate()
        {
            for (k, &v) in table {
                let v = i16::try_from(v)
                    .ok()
                    .filter(|&v| v != NONE)
                    .expect("checked by the caller");
                words.entry(k.clone()).or_default().values[i] = v;
            }
        }
        let mut ascii1 = [WordInfo::default(); 128];
        for (b, slot) in ascii1.iter_mut().enumerate() {
            let c = char::from(u8::try_from(b).expect("< 128"));
            let mut buf = [0u8; 4];
            if let Some(w) = words.get(c.encode_utf8(&mut buf) as &str) {
                *slot = *w;
            }
        }
        let digit_keys = words
            .keys()
            .any(|k| k.as_bytes().first().is_some_and(u8::is_ascii_digit));
        Self {
            utczone: t.utczone,
            tzoffset,
            utczone_list,
            words,
            ascii1,
            digit_keys,
        }
    }

    /// What the tables say about `name` (`name.lower()` looked up).
    pub fn word(&self, name: &str, p: Props) -> WordInfo {
        let b = name.as_bytes();
        if let [c] = b {
            if c.is_ascii() {
                return self.ascii1[usize::from(c.to_ascii_lowercase())];
            }
        }
        if !self.digit_keys && b.first().is_some_and(u8::is_ascii_digit) {
            return WordInfo::default();
        }
        if name.is_ascii() {
            if !b.iter().any(u8::is_ascii_uppercase) {
                return self.words.get(name).copied().unwrap_or_default();
            }
            let mut buf = [0u8; 32];
            if b.len() <= buf.len() {
                let low = &mut buf[..b.len()];
                low.copy_from_slice(b);
                low.make_ascii_lowercase();
                let low = std::str::from_utf8(low).expect("ASCII");
                return self.words.get(low).copied().unwrap_or_default();
            }
        }
        self.words
            .get(p.lower(name).as_ref())
            .copied()
            .unwrap_or_default()
    }

    /// dateutil's default `parserinfo()`.
    pub fn default_tables() -> Self {
        let jump: Vec<&[&str]> = [
            " ", ".", ",", ";", "-", "/", "'", "at", "on", "and", "ad", "m", "t", "of", "st", "nd",
            "rd", "th",
        ]
        .iter()
        .map(std::slice::from_ref)
        .collect();
        let t = Tables {
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
        };
        let utczone_list = ["UTC", "GMT", "Z", "z"]
            .iter()
            .map(|s| (*s).into())
            .collect();
        Self::new(t, FxHashMap::default(), utczone_list)
    }
}

/// A token with what the tables say about it (looked up once).
#[derive(Clone, Debug)]
pub(crate) struct Word<'a> {
    pub raw: std::borrow::Cow<'a, str>,
    pub w: WordInfo,
}

impl<'a> Word<'a> {
    pub fn new(raw: std::borrow::Cow<'a, str>, info: &ParserInfo, p: Props) -> Self {
        let w = info.word(&raw, p);
        Self { raw, w }
    }

    pub fn is(&self, s: &str) -> bool {
        self.raw == s
    }
}
