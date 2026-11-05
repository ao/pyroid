//! MinHash stage 1 against numpy / datatrove 0.10.1 recordings
//! (`tests/dedup_data/`, written by `oracle.py`).

use pyroid_core::dedup::{HashConfig, HashFn, MinHasher, MinhashConfig, Precision, UnicodeVersion};
use serde_json::Value;

fn data(name: &str) -> Value {
    let path = format!("{}/tests/dedup_data/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect(&path)).expect("json")
}

fn u64s(v: &Value) -> Vec<u64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_u64().expect("u64"))
        .collect()
}

/// `np.random.RandomState(seed).randint(1|0, 2**61 - 1, dtype=np.uint64, size=(1, n))`.
#[test]
fn permutation_parameters_match_numpy() {
    let cases = data("numpy-params.json");
    let cases = cases.as_array().expect("cases");
    assert!(cases.len() >= 5);
    for case in cases {
        let seed = u32::try_from(case["seed"].as_u64().unwrap()).unwrap();
        let n = usize::try_from(case["n"].as_u64().unwrap()).unwrap();
        // Any split of n into buckets gives the same flat parameters.
        let config = MinhashConfig {
            seed,
            num_buckets: n,
            hashes_per_bucket: 1,
            ..MinhashConfig::default()
        };
        let h = MinHasher::new(config, UnicodeVersion::LATEST).unwrap();
        let (a, b) = h.parameters();
        assert_eq!(a, u64s(&case["a"]), "a, seed {seed}");
        assert_eq!(b, u64s(&case["b"]), "b, seed {seed}");
    }
}

#[test]
fn all_hash_configs_construct() {
    for hash_fc in [HashFn::Sha1, HashFn::Xxhash] {
        for precision in [Precision::P32, Precision::P64] {
            let config = MinhashConfig {
                hash_config: HashConfig { precision, hash_fc },
                ..MinhashConfig::default()
            };
            let h = MinHasher::new(config, UnicodeVersion::V15_1).unwrap();
            assert_eq!(h.num_hashes(), 112);
        }
    }
    let bad = MinhashConfig {
        n_grams: 0,
        ..MinhashConfig::default()
    };
    assert!(MinHasher::new(bad, UnicodeVersion::LATEST).is_err());
}
