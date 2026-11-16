use std::cmp::Ordering;

use super::num::{Num, tuple_hash};
use super::version::{Local, Pre, Version, parse};

fn v(s: &str) -> Version {
    parse(s).unwrap_or_else(|| panic!("{s:?} should parse")).version
}

#[test]
fn parses_the_spellings() {
    let x = v(" V1!2.0.3-RC.4-post.5_DEV6+Ubuntu-1_a.07 ");
    assert_eq!(x.epoch, Num::Small(1));
    assert_eq!(x.release, vec![Num::Small(2), Num::Small(0), Num::Small(3)]);
    assert_eq!(x.pre, Some((Pre::Rc, Num::Small(4))));
    assert_eq!(x.post, Some(Num::Small(5)));
    assert_eq!(x.dev, Some(Num::Small(6)));
    assert_eq!(
        x.local,
        Some(vec![
            Local::Str("ubuntu".into()),
            Local::Num(Num::Small(1)),
            Local::Str("a".into()),
            Local::Num(Num::Small(7)),
        ])
    );
    assert_eq!(x.to_string_py(), "1!2.0.3rc4.post5.dev6+ubuntu.1.a.7");
    assert_eq!(v("1.0-1").post, Some(Num::Small(1)));
    assert_eq!(v("1.0.r").post, Some(Num::Small(0)));
    assert_eq!(v("1.0rev2").post, Some(Num::Small(2)));
    assert_eq!(v("1.0a").pre, Some((Pre::A, Num::ZERO)));
    assert_eq!(v("1.0alpha1").pre, Some((Pre::A, Num::Small(1))));
    assert_eq!(v("1.0preview1").pre, Some((Pre::Rc, Num::Small(1))));
    assert_eq!(v("1.0c1").pre, Some((Pre::Rc, Num::Small(1))));
    assert_eq!(v("1.0a-1").pre, Some((Pre::A, Num::Small(1))));
    assert_eq!(v("1.0a.dev").dev, Some(Num::ZERO));
    assert_eq!(v("1.0.dev-").dev, Some(Num::ZERO));
    assert_eq!(v("1.0a-").pre, Some((Pre::A, Num::ZERO)));
    assert_eq!(v("1.0-dev1").to_string_py(), "1.0.dev1");
}

#[test]
fn rejects_what_the_pattern_rejects() {
    for bad in [
        "", " ", "1.", ".1", "1..0", "1.0+", "1.0+a..b", "1.0+a.", "1.0ab", "foo", "1.0\u{e9}",
        "1.0+\u{fc}", "v", "1!", "1.0-", "1.0.post.1.", "1.0 1", "\u{0661}", "1.0+a b",
    ] {
        assert!(parse(bad).is_none(), "{bad:?} should not parse");
    }
}

#[test]
fn strips_python_whitespace() {
    assert_eq!(v("\u{1c}\u{3000}1.0\u{85}\u{a0}").to_string_py(), "1.0");
    assert!(parse("\u{200b}1.0").is_none());
}

#[test]
fn orders_as_cmpkey() {
    let order = [
        "1.0.dev0", "1.0a1.dev1", "1.0a1", "1.0a1.post1.dev1", "1.0a1.post1", "1.0b1", "1.0rc1",
        "1.0", "1.0+abc", "1.0+abc.1", "1.0+1", "1.0.post1.dev1", "1.0.post1", "1.0.1",
    ];
    for w in order.windows(2) {
        assert_eq!(v(w[0]).cmp_key(&v(w[1])), Ordering::Less, "{} < {}", w[0], w[1]);
    }
    assert_eq!(v("1.0").cmp_key(&v("1.0.0.0")), Ordering::Equal);
    assert_eq!(v("1.0+ABC").cmp_key(&v("1.0+abc")), Ordering::Equal);
    assert_eq!(v("1.0+01").cmp_key(&v("1.0+1")), Ordering::Equal);
}

#[test]
fn big_numbers_compare_by_value() {
    let big = v("1.123456789012345678901234567890");
    assert!(matches!(big.release[1], Num::Big(_)));
    assert_eq!(big.cmp_key(&v("1.18446744073709551615")), Ordering::Greater);
    assert_eq!(v("1.18446744073709551615").release[1], Num::Small(u64::MAX));
    assert_eq!(v("1.000000000000000000000000000000001").release[1], Num::Small(1));
    let p = parse("1.000000000000000000000000000000001").unwrap();
    assert_eq!(p.long_runs.len(), 1);
}

#[test]
fn hashes_as_cpython() {
    // hash((0, (1,), (3, 0, 0, 0, 1, 0))) and friends, from CPython 3.12 on 64-bit.
    #[cfg(target_pointer_width = "64")]
    {
        assert_eq!(tuple_hash(&[]), 5_740_354_900_026_072_187);
        assert_eq!(tuple_hash(&[1]), -6_644_214_454_873_602_895);
        assert_eq!(tuple_hash(&[1, 2]), -3_550_055_125_485_641_917);
    }
}

#[test]
fn trims_for_str() {
    assert_eq!(v("1.0.0").to_string_trimmed(), "1");
    assert_eq!(v("0.0").to_string_trimmed(), "0");
    assert_eq!(v("1.0.0rc1+x").to_string_trimmed(), "1rc1+x");
    assert_eq!(v("2!1.0.dev1").base_version(), "2!1.0");
    assert_eq!(v("1.0+x").public(), "1.0");
}
