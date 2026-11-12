// Derived from sudachi.rs 0.7.0 (Apache-2.0), © Works Applications Co., Ltd.;
// modified.
//! Japanese morphological analysis with the same morphemes as SudachiPy
//! 0.7.0 in split mode A with its default configuration, read from an
//! installed SudachiDict system dictionary (`sudachidict_core`'s
//! `system.dic`).
//!
//! This is a port of the parts of sudachi.rs (SudachiPy's Rust core) that
//! tokenization uses:
//!
//! - the dictionary reader (the `SudachiBinaryDic` format of SudachiPy 0.7);
//! - the input-text plugins (`DefaultInputTextPlugin` with `rewrite.def`,
//!   `ProlongedSoundMarkPlugin`, `IgnoreYomiganaPlugin`);
//! - the lattice with the MeCab and simple OOV providers (`char.def`,
//!   `unk.def`), the Viterbi best path (same tie-breaking), the path
//!   rewriters (`JoinNumericPlugin`, `JoinKatakanaOovPlugin`) and the A-unit
//!   split.
//!
//! SudachiPy's resource files (`char.def`, `unk.def`, `rewrite.def`) are
//! bundled; the dictionary is not (it is about 200 MB). Normalization and
//! case mapping follow Unicode 17.0, as SudachiPy 0.7.0's build does.
//!
//! - [`Dictionary`]: a loaded dictionary; [`Dictionary::tokenize`] gives the
//!   morphemes of a text.
//! - [`ja`]: datatrove's Japanese word tokenizer (`SpaCyTokenizer("ja")`) for
//!   the filters and MinHash.

use std::cell::RefCell;
use std::path::Path;

mod chars;
mod dic;
mod input;
pub mod ja;
mod lattice;
mod numeric;

pub use dic::DicError;
pub use input::TokenizeError;
pub use lattice::Morpheme;

/// `SimpleOovPlugin`'s settings.
#[derive(Debug, Clone, Copy)]
struct SimpleOov {
    left_id: u16,
    right_id: u16,
    cost: i16,
    pos_id: u16,
}

/// A SudachiDict system dictionary with SudachiPy 0.7.0's default plugins.
#[derive(Debug)]
pub struct Dictionary {
    dic: dic::Dic,
    chars: chars::CharCategory,
    default_input: input::DefaultInputText,
    mecab: chars::MeCabOov,
    simple: SimpleOov,
    numeric_pos: u16,
    katakana_pos: u16,
}

thread_local! {
    static STATE: RefCell<lattice::State> = RefCell::new(lattice::State::default());
}

impl Dictionary {
    /// Read a system dictionary file (`system.dic`).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, DicError> {
        Self::from_bytes(std::fs::read(path)?)
    }

    /// A system dictionary from its bytes.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, DicError> {
        let dic = dic::Dic::new(data.into_boxed_slice())?;
        let chars = chars::CharCategory::parse(chars::CHAR_DEF)?;
        let default_input = input::DefaultInputText::new()?;
        let mecab = chars::MeCabOov::new(&dic)?;
        let pos = |p: [&str; 6]| {
            dic.pos_id(&p).ok_or_else(|| {
                DicError::Format(format!(
                    "POS {} was not in the dictionary, user-defined POS are forbidden",
                    p.join(",")
                ))
            })
        };
        let simple_pos = pos(["補助記号", "一般", "*", "*", "*", "*"])?;
        let numeric_pos = pos(["名詞", "数詞", "*", "*", "*", "*"])?;
        let katakana_pos = pos(["名詞", "普通名詞", "一般", "*", "*", "*"])?;
        // `leftId`/`rightId` 5968, `cost` 3857, checked as `CheckParams` does.
        if 5968 > dic.num_left() || 5968 > dic.num_right() {
            return Err(DicError::Format(
                "the simple OOV plugin's ids are outside the connection matrix".into(),
            ));
        }
        Ok(Self {
            dic,
            chars,
            default_input,
            mecab,
            simple: SimpleOov {
                left_id: 5968,
                right_id: 5968,
                cost: 3857,
                pos_id: simple_pos,
            },
            numeric_pos,
            katakana_pos,
        })
    }

    /// The morphemes of `text`, appended to `out`: `tokenizer.tokenize(text)`
    /// for a SudachiPy tokenizer in split mode A, as byte ranges of `text`
    /// (empty for the characters normalization folded into the previous
    /// morpheme).
    pub fn tokenize(&self, text: &str, out: &mut Vec<Morpheme>) -> Result<(), TokenizeError> {
        STATE.with_borrow_mut(|state| state.tokenize(self, text, out))
    }

    /// The POS of `pos_id`: six fields, `"*"` when unset.
    pub fn pos(&self, pos_id: u16) -> Option<&[Box<str>; 6]> {
        self.dic.pos_list().get(usize::from(pos_id))
    }

    /// The number of POS in the dictionary.
    pub fn pos_count(&self) -> usize {
        self.dic.pos_list().len()
    }

    /// The dictionary's signature (its build identifier).
    pub fn signature(&self) -> &str {
        &self.dic.signature
    }

    /// The dictionary's description comment.
    pub fn comment(&self) -> &str {
        &self.dic.comment
    }

    /// When the dictionary was built, in seconds since the Unix epoch.
    pub fn creation_time(&self) -> u64 {
        self.dic.creation_time
    }
}

#[cfg(test)]
mod replay;
