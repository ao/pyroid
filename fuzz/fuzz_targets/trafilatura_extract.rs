//! `pyroid_core::trafilatura`: `load_html`, `extract` and `bare_extraction` on arbitrary HTML,
//! with options from the first input byte.
//!
//! Input: `[options] html...`. Bits of the options byte: 0-1 the focus (0 balanced,
//! 1 precision, 2 recall, 3 balanced), 2 `include_comments` off, 3 `fast`, 4
//! `include_tables` off, 5 deduplicate, 6-7 pick one of four Unicode versions. All zero:
//! `trafilatura.extract(html)` with CPython 3.15's Unicode 17. The HTML is the rest of the input
//! as given (`load_html` takes surrogatepass UTF-8; other bytes are rejected, as no Python str
//! produces them) and, when it is not valid UTF-8, its lossy decoding too.
//!
//! Checked, besides "no panic" (and bounded time and memory, by libFuzzer):
//! - deterministic: the same results twice;
//! - `extract` equals `nfc(strip(bare.text + "\n" + xmltotxt(commentsbody)))` the way
//!   determine_returnstring builds it: without comments, `extract` is `bare.text` stripped
//!   and normalized, so they agree on None and on the text being empty;
//! - with deduplicate, the cache never holds more than its 4096 entries.

#![no_main]

use libfuzzer_sys::fuzz_target;
use pyroid_core::ftfy::UnicodeVersion;
use pyroid_core::trafilatura::{self, Focus, LruCache, Options};

const VERSIONS: [UnicodeVersion; 4] = [
    UnicodeVersion::V17_0,
    UnicodeVersion::V15_1,
    UnicodeVersion::V15_0,
    UnicodeVersion::V14_0,
];

fn options(b: u8) -> (Options, bool) {
    let focus = match b & 3 {
        1 => Focus::Precision,
        2 => Focus::Recall,
        _ => Focus::Balanced,
    };
    let opts = Options {
        focus,
        comments: b & 4 == 0,
        fast: b & 8 != 0,
        tables: b & 16 == 0,
        unicode: VERSIONS[usize::from(b >> 6)],
        ..Options::default()
    };
    (opts, b & 32 != 0)
}

fn run(html: &[u8], opts: &Options, dedup: bool) {
    let mut cache = LruCache::default();
    let a = trafilatura::extract(html, opts, dedup.then_some(&mut cache));
    if dedup {
        // a second pass sees the first one's entries
        let _ = trafilatura::extract(html, opts, Some(&mut cache));
        assert!(cache.len() <= 4096);
    } else {
        assert_eq!(
            a,
            trafilatura::extract(html, opts, None),
            "extract is deterministic"
        );
    }
    let bare = trafilatura::bare_extraction(html, opts, None);
    let bare2 = trafilatura::bare_extraction(html, opts, None);
    assert_eq!(bare, bare2, "bare_extraction is deterministic");
    if !dedup && !opts.comments {
        // the dedup cache is off: both see the same document
        match (&a, &bare) {
            (Ok(None), Ok(None)) | (Err(_), _) | (_, Err(_)) => {}
            (Ok(Some(t)), Ok(Some(x))) => {
                assert_eq!(
                    t.is_empty(),
                    x.text
                        .trim_matches(pyroid_core::text::is_py_whitespace)
                        .is_empty()
                );
            }
            (a, b) => panic!("extract {a:?} but bare_extraction {b:?}"),
        }
    }
}

fuzz_target!(|data: &[u8]| {
    let Some((&b, html)) = data.split_first() else {
        return;
    };
    let (opts, dedup) = options(b);
    let _ = trafilatura::load_html(html);
    run(html, &opts, dedup);
    if std::str::from_utf8(html).is_err() {
        let lossy = String::from_utf8_lossy(html);
        run(lossy.as_bytes(), &opts, dedup);
    }
});
