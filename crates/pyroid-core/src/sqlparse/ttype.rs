// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! Token types: sqlparse's `_TokenType` tuples (`Token.Keyword.DML` is `("Keyword", "DML")`).
//!
//! A type is a `u16` id into a process-wide registry of paths; one id per path. The types
//! sqlparse's own lexer, grouping and filters use have fixed ids ([`builtin`]); others (from a
//! customised keyword or regex table, or from a tree built by other code) are added with
//! [`intern`]. `ttype is X`, `ttype == X` and `X in (..)` compare ids; `ttype in X` (the
//! `_TokenType.__contains__` prefix test) is [`Types::within`].

use std::sync::{Arc, RwLock};

/// A token type id.
pub type TT = u16;

/// The fixed ids, in the order of [`BUILTIN_PATHS`].
#[allow(missing_docs)]
pub mod builtin {
    use super::TT;
    pub const TOKEN: TT = 0;
    pub const TEXT: TT = 1;
    pub const WHITESPACE: TT = 2;
    pub const NEWLINE: TT = 3;
    pub const ERROR: TT = 4;
    pub const OTHER: TT = 5;
    pub const KEYWORD: TT = 6;
    pub const DML: TT = 7;
    pub const DDL: TT = 8;
    pub const CTE: TT = 9;
    pub const DCL: TT = 10;
    pub const ORDER: TT = 11;
    pub const TZCAST: TT = 12;
    pub const NAME: TT = 13;
    pub const PLACEHOLDER: TT = 14;
    pub const BUILTIN: TT = 15;
    pub const LITERAL: TT = 16;
    pub const STRING: TT = 17;
    pub const STRING_SINGLE: TT = 18;
    pub const STRING_SYMBOL: TT = 19;
    pub const NUMBER: TT = 20;
    pub const HEXADECIMAL: TT = 21;
    pub const FLOAT: TT = 22;
    pub const INTEGER: TT = 23;
    pub const PUNCTUATION: TT = 24;
    pub const OPERATOR: TT = 25;
    pub const COMPARISON: TT = 26;
    pub const WILDCARD: TT = 27;
    pub const COMMENT: TT = 28;
    pub const SINGLE: TT = 29;
    pub const SINGLE_HINT: TT = 30;
    pub const MULTILINE: TT = 31;
    pub const MULTILINE_HINT: TT = 32;
    pub const ASSIGNMENT: TT = 33;
    pub const GENERIC: TT = 34;
    pub const COMMAND: TT = 35;
}

/// The paths of the fixed ids (index = id).
pub const BUILTIN_PATHS: &[&[&str]] = &[
    &[],
    &["Text"],
    &["Text", "Whitespace"],
    &["Text", "Whitespace", "Newline"],
    &["Error"],
    &["Other"],
    &["Keyword"],
    &["Keyword", "DML"],
    &["Keyword", "DDL"],
    &["Keyword", "CTE"],
    &["Keyword", "DCL"],
    &["Keyword", "Order"],
    &["Keyword", "TZCast"],
    &["Name"],
    &["Name", "Placeholder"],
    &["Name", "Builtin"],
    &["Literal"],
    &["Literal", "String"],
    &["Literal", "String", "Single"],
    &["Literal", "String", "Symbol"],
    &["Literal", "Number"],
    &["Literal", "Number", "Hexadecimal"],
    &["Literal", "Number", "Float"],
    &["Literal", "Number", "Integer"],
    &["Punctuation"],
    &["Operator"],
    &["Operator", "Comparison"],
    &["Wildcard"],
    &["Comment"],
    &["Comment", "Single"],
    &["Comment", "Single", "Hint"],
    &["Comment", "Multiline"],
    &["Comment", "Multiline", "Hint"],
    &["Assignment"],
    &["Generic"],
    &["Generic", "Command"],
];

/// Bits of [`Info::within`] for the parents the port tests often.
#[allow(missing_docs)]
pub mod bits {
    pub const KEYWORD: u32 = 1 << 0;
    pub const WHITESPACE: u32 = 1 << 1;
    pub const NEWLINE: u32 = 1 << 2;
    pub const COMMENT: u32 = 1 << 3;
    pub const NAME: u32 = 1 << 4;
    pub const LITERAL: u32 = 1 << 5;
    pub const STRING: u32 = 1 << 6;
    pub const NUMBER: u32 = 1 << 7;
    pub const OPERATOR: u32 = 1 << 8;
    pub const PUNCTUATION: u32 = 1 << 9;
    pub const WILDCARD: u32 = 1 << 10;
    pub const DML: u32 = 1 << 11;
    pub const DDL: u32 = 1 << 12;
    pub const CTE: u32 = 1 << 13;
    pub const TEXT: u32 = 1 << 14;
    pub const COMPARISON: u32 = 1 << 15;
}

const BIT_PARENTS: &[(TT, u32)] = &[
    (builtin::KEYWORD, bits::KEYWORD),
    (builtin::WHITESPACE, bits::WHITESPACE),
    (builtin::NEWLINE, bits::NEWLINE),
    (builtin::COMMENT, bits::COMMENT),
    (builtin::NAME, bits::NAME),
    (builtin::LITERAL, bits::LITERAL),
    (builtin::STRING, bits::STRING),
    (builtin::NUMBER, bits::NUMBER),
    (builtin::OPERATOR, bits::OPERATOR),
    (builtin::PUNCTUATION, bits::PUNCTUATION),
    (builtin::WILDCARD, bits::WILDCARD),
    (builtin::DML, bits::DML),
    (builtin::DDL, bits::DDL),
    (builtin::CTE, bits::CTE),
    (builtin::TEXT, bits::TEXT),
    (builtin::COMPARISON, bits::COMPARISON),
];

/// One registered type.
#[derive(Clone, Debug)]
pub struct Info {
    /// The tuple's items.
    pub path: Vec<Box<str>>,
    /// The parent type (`None` for the root `Token`).
    pub parent: Option<TT>,
    /// [`bits`] of the types this one is `in` (itself included).
    pub within: u32,
}

/// A snapshot of the registry.
#[derive(Clone, Debug)]
pub struct Types {
    infos: Arc<Vec<Info>>,
}

impl Types {
    /// The registered info of `t`.
    #[inline]
    pub fn info(&self, t: TT) -> &Info {
        &self.infos[usize::from(t)]
    }

    /// `t in parent` (`_TokenType.__contains__`: `parent` is a prefix of `t`).
    pub fn within(&self, t: TT, parent: TT) -> bool {
        let mut cur = Some(t);
        while let Some(c) = cur {
            if c == parent {
                return true;
            }
            cur = self.infos[usize::from(c)].parent;
        }
        false
    }

    /// `t in X` for one of the [`bits`] parents.
    #[inline]
    pub fn has(&self, t: TT, bit: u32) -> bool {
        self.infos[usize::from(t)].within & bit != 0
    }

    /// The number of registered types.
    pub fn len(&self) -> usize {
        self.infos.len()
    }

    /// Never true: the root is always registered.
    pub fn is_empty(&self) -> bool {
        self.infos.is_empty()
    }

    /// The id of `path`, if registered.
    pub fn find(&self, path: &[&str]) -> Option<TT> {
        self.infos
            .iter()
            .position(|i| {
                i.path.len() == path.len() && i.path.iter().zip(path).all(|(a, b)| &**a == *b)
            })
            .and_then(|p| TT::try_from(p).ok())
    }
}

fn registry() -> &'static RwLock<Arc<Vec<Info>>> {
    static REG: std::sync::OnceLock<RwLock<Arc<Vec<Info>>>> = std::sync::OnceLock::new();
    REG.get_or_init(|| {
        let mut infos: Vec<Info> = Vec::new();
        for path in BUILTIN_PATHS {
            add(&mut infos, path);
        }
        RwLock::new(Arc::new(infos))
    })
}

fn add(infos: &mut Vec<Info>, path: &[&str]) -> TT {
    if let Some(i) = infos.iter().position(|i| {
        i.path.len() == path.len() && i.path.iter().zip(path).all(|(a, b)| &**a == *b)
    }) {
        return TT::try_from(i).unwrap_or(0);
    }
    let parent = if path.is_empty() {
        None
    } else {
        Some(add(infos, &path[..path.len() - 1]))
    };
    let id = TT::try_from(infos.len()).unwrap_or(TT::MAX);
    let mut within = parent.map_or(0, |p| infos[usize::from(p)].within);
    for &(b, bit) in BIT_PARENTS {
        if b == id {
            within |= bit;
        }
    }
    infos.push(Info {
        path: path.iter().map(|s| Box::<str>::from(*s)).collect(),
        parent,
        within,
    });
    id
}

/// The current registry.
pub fn types() -> Types {
    let reg = registry()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    Types {
        infos: Arc::clone(&reg),
    }
}

/// Registry is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Full;

/// The id of `path`, registering it (and its parents) if needed.
///
/// # Errors
/// [`Full`] after 65,000 distinct types.
pub fn intern(path: &[&str]) -> Result<TT, Full> {
    if let Some(t) = types().find(path) {
        return Ok(t);
    }
    let mut reg = registry()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if reg.len() + path.len() > 65_000 {
        return Err(Full);
    }
    let mut infos = (**reg).clone();
    let t = add(&mut infos, path);
    *reg = Arc::new(infos);
    Ok(t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_and_prefixes() {
        let t = types();
        for (i, p) in BUILTIN_PATHS.iter().enumerate() {
            assert_eq!(t.find(p), TT::try_from(i).ok(), "{p:?}");
        }
        assert!(t.within(builtin::NEWLINE, builtin::WHITESPACE));
        assert!(t.has(builtin::NEWLINE, bits::WHITESPACE));
        assert!(t.has(builtin::DML, bits::KEYWORD));
        assert!(!t.has(builtin::NAME, bits::KEYWORD));
        assert!(t.within(builtin::SINGLE_HINT, builtin::COMMENT));
        let x = intern(&["Keyword", "Custom"]).expect("room");
        let t = types();
        assert!(t.has(x, bits::KEYWORD) && t.within(x, builtin::KEYWORD));
        assert_eq!(intern(&["Keyword", "Custom"]), Ok(x));
    }
}
