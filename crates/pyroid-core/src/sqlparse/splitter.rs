// Derived from sqlparse 0.6.1 (BSD-3-Clause), © the sqlparse authors and contributors;
// modified.
//! `sqlparse.engine.StatementSplitter` (0.6.0's stack-based rewrite), over a stream of
//! `(ttype, value)` items.

use super::pystr;
use super::tree::{Fail, R};
use super::ttype::{TT, Types, bits, builtin as T};
use crate::ftfy::UnicodeVersion;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Block {
    Begin,
    Declare,
    For,
    While,
    Loop,
    If,
    Case,
}

/// The splitter's state (`_reset()`).
#[derive(Debug)]
pub struct Splitter {
    block_stack: Vec<Block>,
    parenthesis_level: i64,
    unconfirmed_start: Option<Block>,
    is_create: bool,
    seen_begin: bool,
    consume_ws: bool,
    level: i64,
    /// Items of the current statement: `(ttype, is_whitespace)`.
    current: usize,
    current_all_ws: bool,
    v: UnicodeVersion,
}

impl Splitter {
    /// A fresh splitter.
    pub fn new(v: UnicodeVersion) -> Splitter {
        Splitter {
            block_stack: Vec::new(),
            parenthesis_level: 0,
            unconfirmed_start: None,
            is_create: false,
            seen_begin: false,
            consume_ws: false,
            level: 0,
            current: 0,
            current_all_ws: true,
            v,
        }
    }

    fn reset(&mut self) {
        let v = self.v;
        *self = Splitter::new(v);
    }

    fn top(&self) -> Option<Block> {
        self.block_stack.last().copied()
    }

    fn nested_block(&mut self, unified: &str) -> Option<i64> {
        match unified {
            "FOR" => {
                self.unconfirmed_start = Some(Block::For);
                return Some(0);
            }
            "WHILE" => {
                self.unconfirmed_start = Some(Block::While);
                return Some(0);
            }
            _ => {}
        }
        if unified == "LOOP" || unified == "DO" {
            if let Some(u @ (Block::For | Block::While)) = self.unconfirmed_start {
                self.block_stack.push(u);
                self.unconfirmed_start = None;
                return Some(1);
            }
            if unified == "LOOP" {
                self.block_stack.push(Block::Loop);
                return Some(1);
            }
        }
        if unified == "IF" {
            self.block_stack.push(Block::If);
            return Some(1);
        }
        if unified == "CASE" {
            self.block_stack.push(Block::Case);
            return Some(1);
        }
        None
    }

    fn closing_keyword(&mut self, unified: &str) -> i64 {
        let want = match unified {
            "END IF" => Some(&[Block::If][..]),
            "END FOR" => Some(&[Block::For][..]),
            "END WHILE" => Some(&[Block::While][..]),
            "END LOOP" => Some(&[Block::Loop, Block::For, Block::While][..]),
            "END CASE" => Some(&[Block::Case][..]),
            "END" => {
                self.block_stack.pop();
                return -1;
            }
            _ => None,
        };
        if let Some(w) = want {
            if self.top().is_some_and(|t| w.contains(&t)) {
                self.block_stack.pop();
                return -1;
            }
        }
        0
    }

    fn change_splitlevel(&mut self, types: &Types, tt: TT, value: &str) -> i64 {
        if tt == T::PUNCTUATION && value == ";" {
            self.unconfirmed_start = None;
            if self.seen_begin {
                self.seen_begin = false;
                if self.top() == Some(Block::Begin) {
                    self.block_stack.pop();
                    return -1;
                }
            }
            return 0;
        }
        if tt == T::PUNCTUATION && value == "(" {
            self.parenthesis_level += 1;
            return 1;
        } else if tt == T::PUNCTUATION && value == ")" {
            self.parenthesis_level = (self.parenthesis_level - 1).max(0);
            return -1;
        } else if !types.has(tt, bits::KEYWORD) {
            return 0;
        }
        let unified = pystr::upper(value, self.v);
        let unified = &*unified;
        if tt == T::DDL && unified.starts_with("CREATE") {
            self.is_create = true;
            return 0;
        }
        if unified == "DECLARE" && self.is_create && self.block_stack.is_empty() {
            self.block_stack.push(Block::Declare);
            return 1;
        }
        if unified == "BEGIN" {
            self.seen_begin = true;
            if self.top() == Some(Block::Declare) {
                self.block_stack.pop();
                self.block_stack.push(Block::Begin);
                return 0;
            }
            self.block_stack.push(Block::Begin);
            return 1;
        }
        if self.seen_begin
            && (tt == T::KEYWORD || tt == T::NAME)
            && matches!(
                unified,
                "TRANSACTION"
                    | "WORK"
                    | "TRAN"
                    | "DISTRIBUTED"
                    | "DEFERRED"
                    | "IMMEDIATE"
                    | "EXCLUSIVE"
            )
        {
            self.seen_begin = false;
            if self.top() == Some(Block::Begin) {
                self.block_stack.pop();
                return -1;
            }
            return 0;
        }
        if self.block_stack.contains(&Block::Begin) {
            if let Some(r) = self.nested_block(unified) {
                return r;
            }
        }
        self.closing_keyword(unified)
    }

    /// Feed one item. `Ok(true)`: the statement so far ended *before* this item (yield it,
    /// then this item starts the next one).
    ///
    /// # Errors
    /// [`Fail::Python`] where sqlparse's code raises (`value.split()[0]` of a blank keyword).
    pub fn push(&mut self, types: &Types, tt: TT, value: &str, is_whitespace: bool) -> R<bool> {
        let mut split_before = false;
        // EOS_TTYPE = T.Whitespace, T.Comment.Single (a tuple: equality)
        if self.consume_ws && tt != T::WHITESPACE && tt != T::SINGLE {
            split_before = true;
            self.reset();
        }
        self.level += self.change_splitlevel(types, tt, value);
        self.current += 1;
        self.current_all_ws &= is_whitespace;
        if tt == T::PUNCTUATION && value == ";" {
            self.seen_begin = false;
            if self.level <= 0 && !self.block_stack.contains(&Block::Begin) {
                self.consume_ws = true;
            }
        } else if tt == T::KEYWORD {
            let first = value
                .split(crate::text::is_py_whitespace)
                .find(|s| !s.is_empty());
            match first {
                Some("GO") => self.consume_ws = true,
                Some(_) => {
                    if !pystr::upper_eq(value, "BEGIN", self.v) {
                        self.seen_begin = false;
                    }
                }
                None => return Err(Fail::Python),
            }
        } else if !(tt == T::WHITESPACE
            || tt == T::NEWLINE
            || tt == T::SINGLE
            || tt == T::MULTILINE)
        {
            self.seen_begin = false;
        }
        Ok(split_before)
    }

    /// At the end: whether the pending statement is yielded (it has a non-whitespace token).
    pub fn finish(&self) -> bool {
        self.current > 0 && !self.current_all_ws
    }
}
