// Derived from libxml2 2.14.6 (MIT), © Daniel Veillard and contributors, and lxml 6.1.3
// (BSD-3-Clause), © Infrae; modified.
//! An HTML parser that builds the same trees as `lxml.html.fromstring` with lxml 6.1.3 and its
//! bundled libxml2 2.14.6, for the parser options Trafilatura uses.
//!
//! - [`tree`]: the document arena, in lxml's element model (`.text`, `.tail`, ordered
//!   attributes), with lxml's editing operations.
//! - [`parser`]: a port of libxml2's HTML parser (HTMLparser.c with the SAX2 tree builder).
//! - [`lxml`]: lxml's string input handling and `lxml.html.fromstring`, with the errors lxml
//!   raises.
//!
//! ```
//! use pyroid_core::html::lxml;
//!
//! let doc = lxml::fromstring_str("<p>Hello <b>world</b>").unwrap();
//! assert_eq!(doc.tag_name(doc.root), "p");
//! assert_eq!(doc.text_content(doc.root), "Hello world");
//! ```
//!
//! The tables in `tables.rs` and `entities.rs` are generated from the libxml2 sources by
//! `generate_tables.py`.

mod entities;
pub mod lxml;
pub mod parser;
mod tables;
pub mod tree;
