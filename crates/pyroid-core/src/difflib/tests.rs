//! Cases from CPython's difflib documentation and docstrings (the full
//! differential tests run against the interpreter's difflib in python/tests).

use super::differ::{CharJunk, DiffOp, DifferVersion, Lines, Side, compare};
use super::format::{format_range_context, format_range_unified};
use super::intern::{CharInterner, Interner};
use super::matcher::{
    BIndex, Block, Tag, calculate_ratio, grouped_opcodes, matches, opcodes, py_slice,
};
use super::mdiff::{Cell, Text, mdiff};
use super::{StrMatcher, nlargest};

fn cps(s: &str) -> Vec<u32> {
    s.chars().map(u32::from).collect()
}

fn matcher(b: &str, junk: impl Fn(u32) -> bool) -> StrMatcher {
    StrMatcher::new(&cps(b), junk, true)
}

fn blocks(a: &str, b: &str, junk: impl Fn(u32) -> bool) -> Vec<(usize, usize, usize)> {
    let m = matcher(b, junk);
    m.index()
        .matching_blocks(&m.encode(&cps(a)))
        .into_iter()
        .map(|Block { a, b, size }| (a, b, size))
        .collect()
}

#[test]
fn docs_matching_blocks_and_ratio() {
    let space = |c| c == u32::from(b' ');
    let (a, b) = (
        "private Thread currentThread;",
        "private volatile Thread currentThread;",
    );
    assert_eq!(blocks(a, b, space), [(0, 0, 8), (8, 17, 21), (29, 38, 0)]);
    let r = matcher(b, space).ratio(&cps(a));
    assert_eq!(r.to_bits(), (2.0 * 29.0 / 67.0_f64).to_bits());
    assert_eq!(
        blocks("abxcd", "abcd", |_| false),
        [(0, 0, 2), (3, 2, 2), (5, 4, 0)]
    );
}

#[test]
fn docs_find_longest_match() {
    let m = matcher("abcd abcd", |_| false);
    let a = m.encode(&cps(" abcd"));
    let mut s = m.index().scratch();
    assert_eq!(
        m.index().find_longest_match(&a, 0, 5, 0, 9, &mut s),
        (0, 4, 5)
    );
    let m = matcher("abcd abcd", |c| c == u32::from(b' '));
    let a = m.encode(&cps(" abcd"));
    let mut s = m.index().scratch();
    assert_eq!(
        m.index().find_longest_match(&a, 0, 5, 0, 9, &mut s),
        (1, 0, 4)
    );
    let m = matcher("ab", |_| false);
    let a = m.encode(&cps("ab"));
    let mut s = m.index().scratch();
    assert_eq!(
        m.index().find_longest_match(&a, 0, 1, 0, 1, &mut s),
        (0, 0, 1)
    );
}

#[test]
fn docs_opcodes() {
    let m = matcher("abycdf", |_| false);
    let b = m.index().matching_blocks(&m.encode(&cps("qabxcd")));
    let ops: Vec<_> = opcodes(&b)
        .into_iter()
        .map(|o| (o.tag.as_str(), o.i1, o.i2, o.j1, o.j2))
        .collect();
    assert_eq!(
        ops,
        [
            ("delete", 0, 1, 0, 0),
            ("equal", 1, 3, 0, 2),
            ("replace", 3, 4, 2, 3),
            ("equal", 4, 6, 3, 5),
            ("insert", 6, 6, 5, 6),
        ]
    );
}

#[test]
fn ratios() {
    assert_eq!(calculate_ratio(0, 0).to_bits(), 1.0f64.to_bits());
    let m = matcher("abcd", |_| false);
    assert_eq!(m.ratio(&cps("bcde")), 0.75);
    assert_eq!(m.quick_ratio(&cps("bcde")), 0.75);
    assert_eq!(m.quick_ratio(&cps("dcba")), 1.0);
    let m = matcher("", |_| false);
    assert_eq!(m.ratio(&[]), 1.0);
    assert_eq!(m.ratio(&cps("a")), 0.0);
}

#[test]
fn autojunk_drops_popular_elements() {
    // test_difflib's TestAutojunk: 200 'b's, then the popular 'b' is ignored.
    let b: String = "b".repeat(200);
    let a: String = format!("a{b}");
    let m = matcher(&b, |_| false);
    assert_eq!(m.ratio(&cps(&a)), 0.0);
    let m = StrMatcher::new(&cps(&b), |_| false, false);
    assert!(m.ratio(&cps(&a)) > 0.99);
}

#[test]
fn grouped() {
    let lines_a: Vec<u32> = (0..40).collect();
    let mut lines_b = lines_a.clone();
    lines_b[20] = 99;
    let mut ids = Interner::default();
    let b: Vec<u32> = lines_b.iter().map(|&x| ids.intern(x).0).collect();
    let a: Vec<u32> = lines_a.iter().map(|x| ids.get(x)).collect();
    let index = BIndex::new(b, ids.len(), &[], true);
    let codes = opcodes(&index.matching_blocks(&a));
    let groups = grouped_opcodes(&codes, 3);
    assert_eq!(groups.len(), 1);
    let g: Vec<_> = groups[0]
        .iter()
        .map(|o| (o.tag, o.i1, o.i2, o.j1, o.j2))
        .collect();
    assert_eq!(
        g,
        [
            (Tag::Equal, 17, 20, 17, 20),
            (Tag::Replace, 20, 21, 20, 21),
            (Tag::Equal, 21, 24, 21, 24),
        ]
    );
    assert_eq!(
        grouped_opcodes(&opcodes(&index.matching_blocks(index.b())), 3),
        Vec::<Vec<_>>::new()
    );
    assert_eq!(matches(&index.matching_blocks(&a)), 39);
}

#[test]
fn ranges() {
    assert_eq!(format_range_unified(3, 3), "3,0");
    assert_eq!(format_range_unified(3, 4), "4");
    assert_eq!(format_range_unified(0, 4), "1,4");
    assert_eq!(format_range_context(3, 3), "3");
    assert_eq!(format_range_context(3, 4), "4");
    assert_eq!(format_range_context(0, 4), "1,4");
    assert_eq!(py_slice(5, 1, 3), 1..3);
    assert_eq!(py_slice(5, 3, -1), 3..4);
    assert_eq!(py_slice(5, 4, 2), 4..4);
    assert_eq!(py_slice(5, -9, 99), 0..5);
}

fn render(ops: &[DiffOp], a: &[&str], b: &[&str]) -> Vec<String> {
    ops.iter()
        .map(|op| match op {
            DiffOp::Line(tag, side, i) => {
                let line = if *side == Side::A { a[*i] } else { b[*i] };
                format!("{} {line}", char::from(*tag))
            }
            DiffOp::Tags(t) => format!("? {t}\n"),
        })
        .collect()
}

fn ndiff(a: &[&str], b: &[&str], version: DifferVersion) -> (Vec<DiffOp>, Vec<String>) {
    let mut ids = Interner::default();
    let bi: Vec<u32> = b.iter().map(|s| ids.intern(*s).0).collect();
    let ai: Vec<u32> = a.iter().map(|s| ids.get(*s)).collect();
    let ac: Vec<Vec<u32>> = a.iter().map(|s| cps(s)).collect();
    let bc: Vec<Vec<u32>> = b.iter().map(|s| cps(s)).collect();
    let ar: Vec<&[u32]> = ac.iter().map(Vec::as_slice).collect();
    let br: Vec<&[u32]> = bc.iter().map(Vec::as_slice).collect();
    let lines = Lines {
        a: &ai,
        b: &bi,
        b_keys: ids.len(),
        b_junk: &[],
        a_chars: &ar,
        b_chars: &br,
    };
    let ops = compare(&lines, CharJunk::SpaceTab, version);
    let text = render(&ops, a, b);
    (ops, text)
}

#[test]
fn docs_ndiff() {
    let a = ["one\n", "two\n", "three\n"];
    let b = ["ore\n", "tree\n", "emu\n"];
    for v in [DifferVersion::Classic, DifferVersion::Windowed] {
        let (_, out) = ndiff(&a, &b, v);
        assert_eq!(
            out.concat(),
            "- one\n?  ^\n+ ore\n?  ^\n- two\n- three\n?  -\n+ tree\n+ emu\n"
        );
    }
}

#[test]
fn qformat_keeps_tabs() {
    let (_, out) = ndiff(
        &["\tabcDefghiJkl\n"],
        &["\tabcdefGhijkl\n"],
        DifferVersion::Windowed,
    );
    assert_eq!(
        out,
        [
            "- \tabcDefghiJkl\n",
            "? \t   ^  ^  ^\n",
            "+ \tabcdefGhijkl\n",
            "? \t   ^  ^  ^\n"
        ]
    );
}

#[test]
fn mdiff_rows() {
    let a = ["one\n", "two\n", "three\n"];
    let b = ["ore\n", "tree\n", "emu\n"];
    let (ops, _) = ndiff(&a, &b, DifferVersion::Windowed);
    let ac: Vec<Vec<u32>> = a.iter().map(|s| cps(s)).collect();
    let bc: Vec<Vec<u32>> = b.iter().map(|s| cps(s)).collect();
    let ar: Vec<&[u32]> = ac.iter().map(Vec::as_slice).collect();
    let br: Vec<&[u32]> = bc.iter().map(Vec::as_slice).collect();
    let rows = mdiff(&ops, &ar, &br, None);
    let marked = |s: &str| Text::Marked(cps(s));
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].from, Cell::Line(1, marked("o\0^n\u{1}e\n")));
    assert_eq!(rows[0].to, Cell::Line(1, marked("o\0^r\u{1}e\n")));
    assert_eq!(rows[1].from, Cell::Line(2, marked("\0-two\n\u{1}")));
    assert_eq!(rows[1].to, Cell::Blank);
    assert_eq!(rows[2].from, Cell::Line(3, marked("t\0-h\u{1}ree\n")));
    assert_eq!(rows[2].to, Cell::Line(2, Text::Line(Side::B, 1)));
    assert_eq!(rows[3].from, Cell::Blank);
    assert_eq!(rows[3].to, Cell::Line(3, marked("\0+emu\n\u{1}")));
    assert!(rows.iter().all(|r| r.flag == Some(true)));
}

#[test]
fn nlargest_ties() {
    let texts = ["b", "a", "b", "c"];
    let scored = [(0.5, 0), (0.5, 1), (0.5, 2), (0.25, 3)];
    let pick = nlargest(&scored, 3, |x, y| texts[x].cmp(texts[y]));
    assert_eq!(pick, [0, 2, 1]);
}

#[test]
fn interners() {
    let (chars, ids) = CharInterner::build(cps("abca\u{e9}"));
    assert_eq!(ids, [0, 1, 2, 0, 3]);
    assert_eq!(chars.get(u32::from('\u{e9}')), 3);
    assert_eq!(chars.get(u32::from('z')), super::matcher::NONE);
}

fn stop_now(_first: bool) -> super::stop::Poll {
    super::stop::Poll::Stop
}

#[test]
#[allow(clippy::many_single_char_names)]
fn checks_stop_long_work_early() {
    use super::matcher::Work;
    use std::time::{Duration, Instant};
    // Quadratic: every line differs from the one opposite, so each match is one
    // line long and the remaining range barely shrinks. 200,000 lines take
    // a minute; stopped at the first check (100 ms in), the call ends soon after.
    let n: u32 = 200_000;
    let b: Vec<u32> = (0..n).collect();
    let a: Vec<u32> = (0..n)
        .map(|i| if i % 2 == 0 { super::matcher::NONE } else { i })
        .collect();
    let index = BIndex::new(b, n as usize, &[], false);
    let mut w = Work::default();
    w.ticker().start(stop_now);
    let t = Instant::now();
    let found = index.matching_blocks_with(&a, &mut w);
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    assert!(w.ticker().finish());
    // Incomplete, but blocks are still sorted, non-overlapping, and end in the sentinel.
    assert!(found.len() < n as usize / 2);
    assert_eq!(
        found.last(),
        Some(&Block {
            a: n as usize,
            b: n as usize,
            size: 0
        })
    );
    // Without checks the same Work gives the full result.
    let full = index.matching_blocks_with(&a[..2_000], &mut w);
    assert_eq!(full.len(), 1_000 + 1);

    // Differ stops between pairs of lines too.
    let la: Vec<String> = (0..3_000)
        .map(|i| format!("value = f(x{}, y{})\n", i % 97, i % 89))
        .collect();
    let lb: Vec<String> = (0..3_000)
        .map(|i| format!("value = f(x{}, y{})\n", i % 89, i % 97))
        .collect();
    let mut keys: Interner<String> = Interner::default();
    let bi: Vec<u32> = lb.iter().map(|l| keys.intern(l.clone()).0).collect();
    let ai: Vec<u32> = la.iter().map(|l| keys.get(l)).collect();
    let (ac, bc): (Vec<Vec<u32>>, Vec<Vec<u32>>) = (
        la.iter().map(|l| cps(l)).collect(),
        lb.iter().map(|l| cps(l)).collect(),
    );
    let (ac, bc): (Vec<&[u32]>, Vec<&[u32]>) = (
        ac.iter().map(Vec::as_slice).collect(),
        bc.iter().map(Vec::as_slice).collect(),
    );
    let lines = Lines {
        a: &ai,
        b: &bi,
        b_keys: keys.len(),
        b_junk: &[],
        a_chars: &ac,
        b_chars: &bc,
    };
    w.ticker().start(stop_now);
    let t = Instant::now();
    let ops =
        super::differ::compare_with(&lines, CharJunk::SpaceTab, DifferVersion::Classic, &mut w);
    assert!(t.elapsed() < Duration::from_secs(2), "{:?}", t.elapsed());
    assert!(w.ticker().finish());
    assert!(ops.len() < 2 * la.len());
}
