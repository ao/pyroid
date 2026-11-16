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

/// Serialize `graph` as one document with the default options.
fn dump_graph(graph: &Graph) -> String {
    use super::emitter::{Emitter, EmitterOptions};
    use super::serializer::{self, SerializerOptions};
    let mut emitter = Emitter::new(EmitterOptions::default());
    serializer::open(&mut emitter, None).unwrap();
    serializer::serialize(&mut emitter, graph, &SerializerOptions::default()).unwrap();
    serializer::close(&mut emitter).unwrap();
    emitter.output().to_owned()
}

fn seq(graph: &mut Graph, items: Vec<NodeId>, flow: bool) -> NodeId {
    graph.push(super::nodes::Node {
        tag: std::borrow::Cow::Borrowed(super::resolver::SEQ),
        kind: NodeKind::Sequence {
            items,
            flow_style: Some(flow),
        },
        start_mark: super::error::Mark::default(),
        end_mark: super::error::Mark::default(),
    })
}

#[test]
fn serializer_anchors_in_pyyaml_order() {
    // [[x], [x], y, y] with x and y shared: x is seen first, so it is id001.
    let mut g = Graph::default();
    let second = seq(&mut g, Vec::new(), true);
    let first = seq(&mut g, Vec::new(), true);
    let one = seq(&mut g, vec![first], true);
    let two = seq(&mut g, vec![first], true);
    g.root = seq(&mut g, vec![one, two, second, second], false);
    assert_eq!(
        dump_graph(&g),
        "- [&id001 []]\n- [*id001]\n- &id002 []\n- *id002\n"
    );
}

#[test]
fn deep_graphs_serialize_on_a_small_stack() {
    // The walks keep their own stacks: 200,000 levels fit in a 64 KiB thread.
    let out = std::thread::Builder::new()
        .stack_size(64 << 10)
        .spawn(|| {
            let mut g = Graph::default();
            let mut id = seq(&mut g, Vec::new(), true);
            for _ in 0..200_000 {
                id = seq(&mut g, vec![id], false);
            }
            g.root = id;
            dump_graph(&g)
        })
        .unwrap()
        .join()
        .unwrap();
    assert!(out.starts_with("- - - "));
    assert!(out.ends_with("- []\n"));
    assert_eq!(out.matches('-').count(), 200_000);
}

/// A stream handed out `k` characters at a time.
#[derive(Debug)]
struct Pieces(std::collections::VecDeque<String>);

impl super::reader::Source for Pieces {
    fn pull(&mut self) -> Result<(String, bool), ()> {
        Ok(match self.0.pop_front() {
            Some(piece) => (piece, false),
            None => (String::new(), true),
        })
    }
}

/// `compose_all` over `text` read `k` characters at a time: the graphs, or the error.
fn compose_stream(text: &str, k: usize) -> Result<Vec<Graph>, String> {
    use super::composer::{Composer, ComposerOptions};
    use super::parser::Parser;
    use super::reader::Reader;
    use super::scanner::{Scanner, ScannerOptions};
    let chars: Vec<char> = text.chars().collect();
    let pieces = chars.chunks(k).map(|c| c.iter().collect()).collect();
    let reader = Reader::stream(Box::new(Pieces(pieces)));
    let parser = Parser::new(Scanner::new(reader, ScannerOptions::default()));
    let mut c = Composer::new(parser, ComposerOptions::default());
    let mut out = Vec::new();
    loop {
        match c.check_node() {
            Ok(false) => return Ok(out),
            Ok(true) => {}
            Err(e) => return Err(e.to_string()),
        }
        match c.get_node() {
            Ok(Some(g)) => out.push(g),
            Ok(None) => return Ok(out),
            Err(e) => return Err(e.to_string()),
        }
    }
}

fn check_stream(text: &str) {
    let whole = compose_all(text).map_err(|e| e.to_string());
    for k in [1, 2, 3, 5, 8, 13, 64] {
        assert_eq!(
            compose_stream(text, k),
            whole,
            "{text:?} read {k} at a time"
        );
    }
}

#[test]
fn streams_compose_as_whole_texts() {
    for text in [
        "a: 1\nb: [x, 'y', \"z\"]\n",
        "- a\n- - b\n  - c\n- {d: e}\n",
        "key:\n- a\n- b\nother: ~\n",
        "? complex\n: value\n",
        "[a: b, c]",
        "--- a\n--- b\n...\n",
        "%YAML 1.1\n--- x\n",
        "%TAG !e! tag:example.com,2000:\n--- !e!foo bar\n",
        "|\n  line1\n  line2\n",
        ">-\n  folded\n  text\n\n  para\n",
        "\"tab\\there\\x41\\u00e9 \\ud800\"",
        "plain\n  continued\n",
        "\u{feff}a",
        "# comment\n# more\na: 1 # c\nb: &x [*x]\n",
        "a: b: c",
        "[a, b",
        "*x",
        "- &a x\n- &a y\n",
        "key: 'unterminated",
        "a\n:b",
        "a: 1\n:b",
        "\t",
        "",
    ] {
        check_stream(text);
    }
}

#[test]
fn streams_compose_as_whole_texts_fuzzed() {
    // Random strings of YAML's significant characters, read in small pieces: every rollback
    // of a fetch that ran out of text must leave the same tokens as one pass over the text.
    // PYROID_YAML_STREAM_FUZZ sets the number of strings (default 3000).
    let n: usize = std::env::var("PYROID_YAML_STREAM_FUZZ")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3000);
    let alphabet: Vec<char> = "ab1 :-[]{},?'\"#&*!|>\n\n  \t%.~\\x\u{e9}"
        .chars()
        .collect();
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for _ in 0..n {
        let len = usize::try_from(next() % 60).unwrap_or(0);
        let text: String = (0..len)
            .map(|_| alphabet[usize::try_from(next()).unwrap_or(0) % alphabet.len()])
            .collect();
        check_stream(&text);
    }
}
