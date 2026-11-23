// Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
//! A port of PyYAML 6.0.3's pure-Python safe loader and dumper (`pyroid.yaml`).
//!
//! The modules follow PyYAML's: the [`reader`] decodes and checks the stream, the
//! [`scanner`] makes tokens, the [`parser`] events, the [`composer`] a node graph per
//! document; the bindings construct Python objects from the nodes. For dumping, the bindings
//! represent Python objects as nodes, the [`serializer`] turns them into events and the
//! [`emitter`] writes the text. Errors carry PyYAML's messages and marks (see [`error`]).

pub mod composer;
pub mod emitter;
pub mod error;
pub mod events;
pub mod nodes;
pub mod parser;
pub mod reader;
pub mod represent;
pub mod resolver;
pub mod scanner;
pub mod serializer;
pub mod tokens;

use composer::{Composer, ComposerOptions};
use error::Result;
use nodes::Graph;
use parser::Parser;
use reader::Reader;
use scanner::{Scanner, ScannerOptions};

/// A loader over already decoded and checked text (see [`reader::find_non_printable`]).
pub fn loader(
    text: String,
    limit: Option<usize>,
    scanner: ScannerOptions,
    composer: ComposerOptions,
) -> Composer {
    Composer::new(
        Parser::new(Scanner::new(Reader::new(text, limit), scanner)),
        composer,
    )
}

/// Every document of `text` as a node graph (`yaml.compose_all`), for tests and tools.
pub fn compose_all(text: &str) -> Result<Vec<Graph>> {
    let mut c = loader(
        text.to_owned(),
        None,
        ScannerOptions::default(),
        ComposerOptions::default(),
    );
    let mut out = Vec::new();
    while c.check_node()? {
        match c.get_node()? {
            Some(g) => out.push(g),
            None => break,
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
