// The body of the `html_parse` fuzz target (fuzz/fuzz_targets/html_parse.rs) and of the
// random-input loop in tests/html_robust.rs, which both `include!` this file. Uses only
// `pyroid_core`.
//
// For any input bytes:
// - `fromstring_bytes` returns; so does `fromstring_wtf8` when `ftfy::validate_wtf8` accepts
//   the input (and it returns `InvalidInput` otherwise), and `fromstring_str` equals
//   `fromstring_wtf8` on valid UTF-8;
// - a second call gives the same result;
// - every returned tree is well formed: parent, child and sibling links agree, `nchild`
//   counts the children, depth stays within the parser's 256 levels, element and attribute
//   names have no ASCII capitals, no text or tail is `Some("")`, and the text and attribute
//   values add up to at most three bytes per input byte (U+FFFD for one bad byte).

use pyroid_core::html::lxml::{LxmlError, fromstring_bytes, fromstring_str, fromstring_wtf8};
use pyroid_core::html::tree::{Document, Kind, NIL, NodeId};

fn dump_or_error(r: &Result<Document, LxmlError>) -> String {
    match r {
        Ok(d) => {
            let mut s = String::new();
            d.dump(d.root, true, &mut s);
            s
        }
        Err(e) => format!("error {e:?}"),
    }
}

fn check_links(d: &Document, input_len: usize) {
    let n = d.nodes.len();
    assert!(d.root != NIL && (d.root as usize) < n);
    assert!(n <= input_len + 8, "{n} nodes from {input_len} bytes");
    let mut payload = 0usize;
    let mut seen = vec![false; n];
    // Walk every top-level element (the root's top ancestor and its siblings).
    let mut top = d.top(d.root);
    while let Some(p) = d.prev(top) {
        top = p;
    }
    let mut stack: Vec<(NodeId, usize)> = Vec::new();
    let mut t = top;
    loop {
        assert_eq!(d.n(t).parent, NIL);
        stack.push((t, 1));
        let next = d.n(t).next;
        if next == NIL {
            break;
        }
        assert_eq!(d.n(next).prev, t);
        t = next;
    }
    while let Some((id, depth)) = stack.pop() {
        assert!(!seen[id as usize], "node {id} reached twice");
        seen[id as usize] = true;
        assert!(depth <= 256, "depth {depth}");
        let node = d.n(id);
        if node.kind == Kind::Element {
            let tag = d.name(node.tag);
            assert!(
                !tag.is_empty() && !tag.bytes().any(|b| b.is_ascii_uppercase()),
                "{tag:?}"
            );
            for a in &node.attrs {
                assert!(!d.name(a.name).bytes().any(|b| b.is_ascii_uppercase()));
                payload += a.value.len();
            }
        }
        for s in [&node.text, &node.tail].into_iter().flatten() {
            assert!(!s.is_empty(), "empty text node");
            payload += s.len();
        }
        let mut count = 0;
        let mut prev = NIL;
        let mut c = node.first;
        while c != NIL {
            let cn = d.n(c);
            assert_eq!(cn.parent, id);
            assert_eq!(cn.prev, prev);
            stack.push((c, depth + 1));
            count += 1;
            prev = c;
            c = cn.next;
        }
        assert_eq!(node.last, prev);
        assert_eq!(node.nchild, count);
    }
    assert!(seen[d.root as usize]);
    assert!(
        payload <= 3 * input_len,
        "{payload} bytes of text from {input_len}"
    );
}

pub fn check(data: &[u8]) {
    let bytes = fromstring_bytes(data);
    if let Ok(d) = &bytes {
        check_links(d, data.len());
    }
    assert_eq!(
        dump_or_error(&fromstring_bytes(data)),
        dump_or_error(&bytes)
    );
    let wtf8 = fromstring_wtf8(data);
    match pyroid_core::ftfy::validate_wtf8(data) {
        Ok(()) => {
            if let Ok(d) = &wtf8 {
                check_links(d, data.len());
            }
            assert!(!matches!(wtf8, Err(LxmlError::InvalidInput(_))));
            if let Ok(s) = std::str::from_utf8(data) {
                assert_eq!(dump_or_error(&fromstring_str(s)), dump_or_error(&wtf8));
            }
        }
        Err(at) => assert_eq!(wtf8.err(), Some(LxmlError::InvalidInput(at))),
    }
}
