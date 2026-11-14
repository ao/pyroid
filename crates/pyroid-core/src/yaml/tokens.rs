// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! Scanner tokens (`tokens.py`).

use super::error::Mark;

/// The value of a `%YAML` or `%TAG` directive (`None` for other directives).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirectiveValue {
    None,
    /// `(major, minor)`; numbers too big for a `u64` saturate (they are never 1).
    Yaml(u64, u64),
    /// `(handle, prefix)`.
    Tag(String, String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TokenKind {
    StreamStart,
    StreamEnd,
    Directive {
        name: String,
        value: DirectiveValue,
    },
    DocumentStart,
    DocumentEnd,
    BlockSequenceStart,
    BlockMappingStart,
    BlockEnd,
    FlowSequenceStart,
    FlowMappingStart,
    FlowSequenceEnd,
    FlowMappingEnd,
    Key,
    Value,
    BlockEntry,
    FlowEntry,
    Alias(String),
    Anchor(String),
    /// `(handle, suffix)`; `handle` is `None` for verbatim `!<...>` tags and a lone `!`.
    Tag(Option<String>, String),
    /// `style`: `None` for plain scalars, else `'\''`, `'"'`, `'|'` or `'>'`.
    Scalar {
        value: String,
        plain: bool,
        style: Option<char>,
    },
}

impl TokenKind {
    /// PyYAML's `Token.id`.
    pub const fn id(&self) -> &'static str {
        match self {
            Self::StreamStart => "<stream start>",
            Self::StreamEnd => "<stream end>",
            Self::Directive { .. } => "<directive>",
            Self::DocumentStart => "<document start>",
            Self::DocumentEnd => "<document end>",
            Self::BlockSequenceStart => "<block sequence start>",
            Self::BlockMappingStart => "<block mapping start>",
            Self::BlockEnd => "<block end>",
            Self::FlowSequenceStart => "[",
            Self::FlowMappingStart => "{",
            Self::FlowSequenceEnd => "]",
            Self::FlowMappingEnd => "}",
            Self::Key => "?",
            Self::Value => ":",
            Self::BlockEntry => "-",
            Self::FlowEntry => ",",
            Self::Alias(_) => "<alias>",
            Self::Anchor(_) => "<anchor>",
            Self::Tag(..) => "<tag>",
            Self::Scalar { .. } => "<scalar>",
        }
    }
}

/// A token and its marks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub start_mark: Mark,
    pub end_mark: Mark,
}

/// The kinds `check_token` asks about, without payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum T {
    StreamStart,
    StreamEnd,
    Directive,
    DocumentStart,
    DocumentEnd,
    BlockSequenceStart,
    BlockMappingStart,
    BlockEnd,
    FlowSequenceStart,
    FlowMappingStart,
    FlowSequenceEnd,
    FlowMappingEnd,
    Key,
    Value,
    BlockEntry,
    FlowEntry,
    Alias,
    Anchor,
    Tag,
    Scalar,
}

impl TokenKind {
    /// The payload-free kind.
    pub const fn t(&self) -> T {
        match self {
            Self::StreamStart => T::StreamStart,
            Self::StreamEnd => T::StreamEnd,
            Self::Directive { .. } => T::Directive,
            Self::DocumentStart => T::DocumentStart,
            Self::DocumentEnd => T::DocumentEnd,
            Self::BlockSequenceStart => T::BlockSequenceStart,
            Self::BlockMappingStart => T::BlockMappingStart,
            Self::BlockEnd => T::BlockEnd,
            Self::FlowSequenceStart => T::FlowSequenceStart,
            Self::FlowMappingStart => T::FlowMappingStart,
            Self::FlowSequenceEnd => T::FlowSequenceEnd,
            Self::FlowMappingEnd => T::FlowMappingEnd,
            Self::Key => T::Key,
            Self::Value => T::Value,
            Self::BlockEntry => T::BlockEntry,
            Self::FlowEntry => T::FlowEntry,
            Self::Alias(_) => T::Alias,
            Self::Anchor(_) => T::Anchor,
            Self::Tag(..) => T::Tag,
            Self::Scalar { .. } => T::Scalar,
        }
    }
}
