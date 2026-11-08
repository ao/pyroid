// Derived from sudachi.rs 0.7.0 (Apache-2.0), © Works Applications Co., Ltd.;
// modified.
//! Character categories (`char.def`) and the OOV definitions (`unk.def`).

use super::dic::{Dic, DicError};

/// `CategoryType` bits.
pub(crate) mod cat {
    pub const DEFAULT: u32 = 1 << 0;
    pub const KANJI: u32 = 1 << 2;
    pub const NUMERIC: u32 = 1 << 4;
    pub const ALPHA: u32 = 1 << 5;
    pub const HIRAGANA: u32 = 1 << 6;
    pub const KATAKANA: u32 = 1 << 7;
    pub const KANJINUMERIC: u32 = 1 << 8;
    pub const GREEK: u32 = 1 << 9;
    pub const CYRILLIC: u32 = 1 << 10;
    pub const NOOOVBOW: u32 = 1 << 30;
    pub const NOOOVEOW: u32 = 1 << 31;
    /// `CategoryType::all()`.
    pub const ALL: u32 = u32::MAX;

    /// The named flags, in declaration (and bit) order.
    pub const NAMES: [(&str, u32); 18] = [
        ("DEFAULT", 1 << 0),
        ("SPACE", 1 << 1),
        ("KANJI", 1 << 2),
        ("SYMBOL", 1 << 3),
        ("NUMERIC", 1 << 4),
        ("ALPHA", 1 << 5),
        ("HIRAGANA", 1 << 6),
        ("KATAKANA", 1 << 7),
        ("KANJINUMERIC", 1 << 8),
        ("GREEK", 1 << 9),
        ("CYRILLIC", 1 << 10),
        ("USER1", 1 << 11),
        ("USER2", 1 << 12),
        ("USER3", 1 << 13),
        ("USER4", 1 << 14),
        ("NOOOVBOW", 1 << 30),
        ("NOOOVEOW", 1 << 31),
        ("ALL", 0x3fff_ffff),
    ];

    /// bitflags' `from_str` for one name (no `|`).
    pub fn parse(name: &str) -> Option<u32> {
        NAMES.iter().find(|(n, _)| *n == name).map(|&(_, b)| b)
    }
}

pub(crate) const CHAR_DEF: &str = include_str!("resources/char.def");
pub(crate) const UNK_DEF: &str = include_str!("resources/unk.def");
pub(crate) const REWRITE_DEF: &str = include_str!("resources/rewrite.def");

fn format_err(file: &str, line: usize) -> DicError {
    DicError::Format(format!("invalid {file} at line {line}"))
}

/// `CharacterCategory`: the categories of every code point.
#[derive(Debug, Clone)]
pub(crate) struct CharCategory {
    boundaries: Vec<u32>,
    categories: Vec<u32>,
}

impl CharCategory {
    /// `CharacterCategory::from_reader` + `compile`.
    pub fn parse(def: &str) -> Result<Self, DicError> {
        let mut ranges: Vec<(u32, u32, u32)> = Vec::new();
        for (i, line) in def.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || !line.starts_with("0x") {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 2 {
                return Err(format_err("char.def", i));
            }
            let r: Vec<&str> = cols[0].split("..").collect();
            let hex = |s: &str| {
                u32::from_str_radix(s.trim_start_matches("0x"), 16)
                    .map_err(|_| format_err("char.def", i))
            };
            let begin = hex(r[0])?;
            let end = if r.len() > 1 {
                hex(r[1])? + 1
            } else {
                begin + 1
            };
            if begin >= end || char::from_u32(begin).is_none() || char::from_u32(end).is_none() {
                return Err(format_err("char.def", i));
            }
            let mut categories = 0;
            for elem in cols[1..].iter().take_while(|e| !e.starts_with('#')) {
                categories |= cat::parse(elem).ok_or_else(|| format_err("char.def", i))?;
            }
            ranges.push((begin, end, categories));
        }
        Ok(Self::compile(&ranges))
    }

    fn compile(ranges: &[(u32, u32, u32)]) -> Self {
        if ranges.is_empty() {
            return Self {
                boundaries: Vec::new(),
                categories: vec![cat::DEFAULT],
            };
        }
        let mut boundaries: Vec<u32> = ranges.iter().flat_map(|r| [r.0, r.1]).collect();
        boundaries.sort_unstable();
        boundaries.dedup();
        let mut categories = vec![0u32; boundaries.len()];
        for &(begin, end, cats) in ranges {
            let start = boundaries.binary_search(&begin).expect("a boundary") + 1;
            for i in start..boundaries.len() {
                if boundaries[i] > end {
                    break;
                }
                categories[i] |= cats;
            }
        }
        categories[0] = cat::DEFAULT;
        let mut final_boundaries = Vec::with_capacity(boundaries.len());
        let mut final_categories = Vec::with_capacity(categories.len());
        let mut last_category = categories[0];
        let mut last_boundary = boundaries[0];
        for i in 1..categories.len() {
            if categories[i] == last_category {
                last_boundary = boundaries[i];
                continue;
            }
            final_boundaries.push(last_boundary);
            final_categories.push(last_category);
            last_category = categories[i];
            last_boundary = boundaries[i];
        }
        final_categories.push(last_category);
        final_boundaries.push(last_boundary);
        for c in &mut final_categories {
            if *c == 0 {
                *c = cat::DEFAULT;
            }
        }
        final_categories.push(cat::DEFAULT);
        Self {
            boundaries: final_boundaries,
            categories: final_categories,
        }
    }

    /// `get_category_types`.
    #[inline]
    pub fn get(&self, c: char) -> u32 {
        if self.boundaries.is_empty() {
            return cat::DEFAULT;
        }
        match self.boundaries.binary_search(&u32::from(c)) {
            Ok(i) => self.categories[i + 1],
            Err(i) => self.categories[i],
        }
    }
}

/// One `char.def` category header line, as the MeCab OOV plugin reads it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct CategoryInfo {
    pub is_invoke: bool,
    pub is_group: bool,
    pub length: u32,
}

/// One `unk.def` line.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Oov {
    pub left_id: i16,
    pub right_id: i16,
    pub cost: i16,
    pub pos_id: u16,
}

/// The MeCab OOV plugin's tables, indexed by category bit.
#[derive(Debug, Clone)]
pub(crate) struct MeCabOov {
    pub categories: [Option<CategoryInfo>; 32],
    pub oovs: [Vec<Oov>; 32],
}

impl MeCabOov {
    /// `MeCabOovPlugin::set_up` with the default `char.def` and `unk.def`.
    pub fn new(dic: &Dic) -> Result<Self, DicError> {
        let mut categories: [Option<CategoryInfo>; 32] = [None; 32];
        for (i, line) in CHAR_DEF.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("0x") {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            if cols.len() < 4 {
                return Err(format_err("char.def", i));
            }
            let t = cat::parse(cols[0]).ok_or_else(|| format_err("char.def", i))?;
            // Composite names (`ALL`) are not single bits; the default file has none.
            if t.count_ones() != 1 {
                return Err(format_err("char.def", i));
            }
            let slot = &mut categories[t.trailing_zeros() as usize];
            if slot.is_some() {
                return Err(format_err("char.def", i));
            }
            *slot = Some(CategoryInfo {
                is_invoke: cols[1] == "1",
                is_group: cols[2] == "1",
                length: cols[3].parse().map_err(|_| format_err("char.def", i))?,
            });
        }
        let mut oovs: [Vec<Oov>; 32] = Default::default();
        for (i, line) in UNK_DEF.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() < 10 {
                return Err(format_err("unk.def", i));
            }
            let t = cat::parse(cols[0]).ok_or_else(|| format_err("unk.def", i))?;
            if t.count_ones() != 1 || categories[t.trailing_zeros() as usize].is_none() {
                return Err(format_err("unk.def", i));
            }
            let num = |s: &str| s.parse::<i16>().map_err(|_| format_err("unk.def", i));
            let pos_id = dic.pos_id(&cols[4..10]).ok_or_else(|| {
                DicError::Format(format!(
                    "POS {} was not in the dictionary, user-defined POS are forbidden",
                    cols[4..10].join(",")
                ))
            })?;
            let oov = Oov {
                left_id: num(cols[1])?,
                right_id: num(cols[2])?,
                cost: num(cols[3])?,
                pos_id,
            };
            // `as usize` in sudachi.rs: a negative id is out of range.
            let in_range = |id: i16, n: usize| usize::try_from(id).is_ok_and(|id| id <= n);
            if !in_range(oov.left_id, dic.num_left()) || !in_range(oov.right_id, dic.num_right()) {
                return Err(format_err("unk.def", i));
            }
            oovs[t.trailing_zeros() as usize].push(oov);
        }
        Ok(Self { categories, oovs })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_char_def() {
        let c = CharCategory::parse(CHAR_DEF).unwrap();
        assert_eq!(c.get(' '), 1 << 1);
        assert_eq!(c.get('a'), cat::ALPHA);
        assert_eq!(c.get('一'), cat::KANJI | cat::KANJINUMERIC);
        assert_eq!(c.get('ァ'), cat::KATAKANA | cat::NOOOVBOW);
        assert_eq!(c.get('\u{10ffff}'), cat::DEFAULT);
        assert_eq!(c.get('\0'), cat::DEFAULT);
    }
}
