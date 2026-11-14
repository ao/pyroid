//! Smoke tests of the loader pipeline (the parity suites compare against PyYAML itself).

use super::compose_all;
use super::error::YamlError;
use super::nodes::{Graph, NodeId, NodeKind};

/// A compact rendering of a node: `tag-suffix:value`, `[..]`, `{k: v}`.
fn show(g: &Graph, id: NodeId) -> String {
    let n = &g.nodes[id];
    let tag = n.tag.strip_prefix("tag:yaml.org,2002:").unwrap_or(&n.tag);
    match &n.kind {
        NodeKind::Scalar { value, .. } => format!("{tag}:{value}"),
        NodeKind::Sequence { items, .. } => {
            format!(
                "[{}]",
                items
                    .iter()
                    .map(|&i| show(g, i))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        NodeKind::Mapping { pairs, .. } => format!(
            "{{{}}}",
            pairs
                .iter()
                .map(|&(k, v)| format!("{}: {}", show(g, k), show(g, v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn one(text: &str) -> String {
    let docs = compose_all(text).unwrap_or_else(|e| panic!("{text:?}: {e}"));
    docs.iter()
        .map(|g| show(g, g.root))
        .collect::<Vec<_>>()
        .join(" --- ")
}

fn err(text: &str) -> String {
    match compose_all(text) {
        Ok(_) => panic!("{text:?} composed"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn block_and_flow() {
    assert_eq!(
        one("a: 1\nb: [x, 'y', \"z\"]\n"),
        "{str:a: int:1, str:b: [str:x, str:y, str:z]}"
    );
    assert_eq!(
        one("- a\n- - b\n  - c\n- {d: e}\n"),
        "[str:a, [str:b, str:c], {str:d: str:e}]"
    );
    assert_eq!(
        one("key:\n- a\n- b\nother: ~\n"),
        "{str:key: [str:a, str:b], str:other: null:~}"
    );
    assert_eq!(one("? complex\n: value\n"), "{str:complex: str:value}");
    assert_eq!(one("[a: b, c]"), "[{str:a: str:b}, str:c]");
}

#[test]
fn scalars() {
    assert_eq!(one("|\n  line1\n  line2\n"), "str:line1\nline2\n");
    assert_eq!(
        one(">-\n  folded\n  text\n\n  para\n"),
        "str:folded text\npara"
    );
    assert_eq!(one("'it''s'"), "str:it's");
    assert_eq!(
        one("\"tab\\there\\x41\\u00e9\""),
        "str:tab\there\u{41}\u{e9}"
    );
    assert_eq!(one("plain\n  continued\n"), "str:plain continued");
    assert_eq!(one("! true"), "bool:true");
    assert_eq!(one("! 'true'"), "bool:true");
    assert_eq!(one("!!str 1"), "str:1");
    assert_eq!(one("!foo bar"), "!foo:bar");
}

#[test]
fn documents() {
    assert_eq!(one("--- a\n--- b\n...\n"), "str:a --- str:b");
    assert_eq!(one("%YAML 1.1\n--- x\n"), "str:x");
    assert_eq!(
        one("%TAG !e! tag:example.com,2000:\n--- !e!foo bar\n"),
        "tag:example.com,2000:foo:bar"
    );
    assert_eq!(one(""), "");
    assert_eq!(one("\u{feff}a"), "str:a");
}

#[test]
fn aliases_share_nodes() {
    let docs = compose_all("a: &x [1]\nb: *x\n").unwrap();
    let g = &docs[0];
    let NodeKind::Mapping { pairs, .. } = &g.nodes[g.root].kind else {
        panic!()
    };
    assert_eq!(pairs[0].1, pairs[1].1);
}

#[test]
fn errors() {
    assert_eq!(
        err("a: b: c"),
        "ScannerError: mapping values are not allowed here [1:5]"
    );
    assert_eq!(
        err("[a, b"),
        "ParserError: while parsing a flow sequence [1:1] expected ',' or ']', but got '<stream end>' [1:6]"
    );
    assert_eq!(err("*x"), "ComposerError: found undefined alias 'x' [1:1]");
    assert_eq!(
        err("- &a x\n- &a y\n"),
        "ComposerError: found duplicate anchor 'a'; first occurrence [1:3] second occurrence [2:3]"
    );
    assert_eq!(
        err("key: 'unterminated"),
        "ScannerError: while scanning a quoted scalar [1:6] found unexpected end of stream [1:19]"
    );
    assert_eq!(err("\"\\Uffffffff\""), "chr(0xffffffff) failed");
    assert_eq!(err("\"\\U00110000\""), "chr(0x110000) failed");
    assert_eq!(one("a\n:b"), "str:a :b");
    assert_eq!(
        err("a: 1\n:b"),
        "ScannerError: while scanning a simple key [2:1] could not find expected ':' [2:3]"
    );
    assert!(matches!(compose_all("\t"), Err(YamlError::Marked(_))));
}
