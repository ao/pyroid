//! `zlib.decompressobj` errors that zlib-rs's fast path finds: CPython's messages (zlib keeps
//! the message `inflate_fast` sets; zlib-rs replaces it with "repeated call with bad state").

use pyroid_core::warc::{DecompressObj, PyExc};

/// The gzip member that holds byte 667 of the parity suite's synthetic file `flip/gz-3-at-667`
/// (a WARC record with a bit flipped at byte 166 of the member).
const MEMBER: &str = concat!(
    "1f8b08000000000000ff65526d6f9b3010fe8e94ffe0fd00c7405ed4501a694a262ddaa4461153d57e33708e2d818dcc",
    "6569fefd6c036dd47dc0d8e77b5eeece2fdf4f3b96cce359f4e276b4b87590110b7d67740f6370cfd105d3385dd338a1",
    "715ac48b2c5e66f1ea6d4c3841656c4d0ffb8ce417abb3cb45d5590d695aa6ab9226e922a1cb7221287fa82a1a8b75cc",
    "97620db0596c67d1ce68048df437e833ca8cac93cd6771b0c3bbae51154765349388dd2369fb33baaba72f460b6ecf80",
    "f4cfe990119f983106efbced1a9857a6651d3f039b2a3dd44e400905353df25b63783daa21bca393699b59348b7e16c5",
    "d1b52771d5c7e4f9d757671fb98fa492dcf6804f1714f4e1ffb2568b8de7cbbfed9f77c5ebf107f1a86d94875f2e81d7",
    "db1c1536b06d0282f46881b7040d4109167236dce62ce4467969ea9bfb755b945c93b05c2568d2425b822552f56e8c7e",
    "2c441b24ee763c39326230ac4238c344184b783f808d085be5f40761c2719eb32e080d28177759266cb885f0296f006e",
    "de05fac3c83cf05c3d9173e350636957e9383410c96b7275b50de05a9da1774e6d25d55fcfe905fa292cac693d3517e8",
    "4cf8494e050db6c6b227b4a70f6ebcd65d0d037c4406ced60403be87e62e726fe3b3037ab8f482c1f87dc77d9b47b31a",
    "aed3d69953a1bfe56d1aa99aecb030c39c0d4fc13f8f7f8f0d666c8e030000",
);

fn member() -> Vec<u8> {
    (0..MEMBER.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&MEMBER[i..i + 2], 16).unwrap())
        .collect()
}

/// The errors of `decompress()` calls on `chunk`-byte pieces: (offset of the piece, message).
fn errors(chunk: usize) -> Vec<(usize, String)> {
    let data = member();
    let mut d = DecompressObj::new(31);
    let mut out = Vec::new();
    let mut got = Vec::new();
    for (i, piece) in data.chunks(chunk).enumerate() {
        if let Err(e) = d.decompress(piece, &mut out) {
            let (exc, msg) = e.py().unwrap();
            assert_eq!(exc, PyExc::ZlibError);
            got.push((i * chunk, msg.to_owned()));
        }
    }
    got
}

// CPython 3.13 (zlib 1.3.2): `zlib.decompressobj(31)` fed the member in pieces.
const TOO_FAR: &str = "Error -3 while decompressing data: invalid distance too far back";

#[test]
fn fast_path_error_keeps_zlibs_message() {
    assert_eq!(member().len(), 511);
    for (chunk, first) in [(100, 100), (7, 161), (16384, 0)] {
        let got = errors(chunk);
        assert_eq!(got[0], (first, TOO_FAR.to_owned()), "pieces of {chunk}");
        assert!(
            got.iter().all(|(_, m)| m == TOO_FAR),
            "pieces of {chunk}: {got:?}"
        );
        assert_eq!(got.len(), member().len().div_ceil(chunk) - first / chunk);
    }
}
