//! The `(b, r)` searches of `MinHashLSH` and `MinHashLSHEnsemble`
//! (`datasketch::params`) and the `scipy.integrate.quad` port they use
//! (`datasketch::quad`), against values recorded from datasketch 2.0.0 with
//! scipy 1.18.1 on Linux aarch64, glibc 2.36 (`tests/datasketch_data/params.json`;
//! floats are `float.hex`).
//!
//! The integrands call libm's `pow`, in datasketch as in `f64::powf`, and
//! libms differ in the last bit of `pow` on a few inputs: glibc's FMA code
//! path (aarch64, and x86_64 with AVX2 and FMA) and its SSE2 path (other
//! x86_64 CPUs) disagree on about 3 in 10,000 of the bases and exponents seen
//! here. The reference therefore carries `pow_probes`, inputs on which those
//! two paths differ. If this platform's `pow` reproduces all of them,
//! probabilities, integrals and QUADPACK's counters must match bit for bit;
//! otherwise the tests report the differences and do not fail on them.
//! Argmins must match exactly everywhere (they do on both glibc paths).
//!
//! The larger searches are `#[ignore]`d (slow in debug builds):
//! `cargo test --release -p pyroid-core --test datasketch_params -- --ignored --nocapture`.
//! `probabilities_match_datasketch_full_grid` also needs `DATASKETCH_PARAMS_FULL=<dir>`, a
//! directory with the full recordings (`full_lsh_probs.json`,
//! `full_ens_probs.json`: every probability the argmin grid evaluates), which
//! are too large to commit; it skips without it.

#![allow(clippy::print_stderr)] // reports of the comparisons

use std::sync::LazyLock;
use std::time::Instant;

use pyroid_core::datasketch::params::{
    ensemble_false_negative, ensemble_false_positive, ensemble_optimal_param, lsh_false_negative,
    lsh_false_positive, lsh_optimal_param,
};
use pyroid_core::datasketch::quad::{DEFAULT_EPSABS, DEFAULT_EPSREL, DEFAULT_LIMIT, qagse};
use rayon::prelude::*;
use serde_json::Value;

static DATA: LazyLock<Value> = LazyLock::new(|| {
    let path = format!(
        "{}/tests/datasketch_data/params.json",
        env!("CARGO_MANIFEST_DIR")
    );
    serde_json::from_str(&std::fs::read_to_string(&path).expect(&path)).expect("json")
});

/// Whether this platform's `pow` gives the recorded results on `pow_probes`.
static LIBM_MATCHES: LazyLock<bool> = LazyLock::new(|| {
    let probes = rows("pow_probes");
    assert!(probes.len() >= 100);
    let differ = probes
        .iter()
        .filter(|p| {
            let (x, y) = (hex(&p[0]), f64::from(u32::try_from(uint(&p[1])).unwrap()));
            !same(x.powf(y), hex(&p[2]))
        })
        .count();
    if differ > 0 {
        eprintln!(
            "pow differs from the reference platform's on {differ} of {} probes: \
             value differences are reported, not asserted",
            probes.len()
        );
    }
    differ == 0
});

/// Parse Python's `float.hex` output (`0x1.8000000000000p-1`, `-0x0.0p+0`, `nan`, ...).
fn parse_hex(s: &str) -> f64 {
    match s {
        "nan" => return f64::NAN,
        "inf" => return f64::INFINITY,
        "-inf" => return f64::NEG_INFINITY,
        _ => {}
    }
    let (negative, rest) = s.strip_prefix('-').map_or((false, s), |r| (true, r));
    let rest = rest.strip_prefix("0x").expect(s);
    let (mantissa, exp) = rest.split_once('p').expect(s);
    let exp: i32 = exp.parse().expect(s);
    let (lead, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    assert!(frac.len() <= 13, "{s}");
    let frac = u64::from_str_radix(&format!("{frac:0<13}"), 16).expect(s);
    let magnitude = match lead {
        "1" => {
            let biased = u64::try_from(exp + 1023).expect(s);
            assert!((1..=2046).contains(&biased), "{s}");
            (biased << 52) | frac
        }
        "0" => {
            assert!(frac == 0 || exp == -1022, "{s}");
            frac
        }
        _ => panic!("{s}"),
    };
    f64::from_bits(magnitude | (u64::from(negative) << 63))
}

fn hex(v: &Value) -> f64 {
    parse_hex(v.as_str().expect("hex string"))
}

fn uint(v: &Value) -> usize {
    usize::try_from(v.as_u64().expect("unsigned")).expect("usize")
}

fn rows(key: &str) -> &'static [Value] {
    DATA[key].as_array().expect(key)
}

fn weights() -> Vec<(f64, f64)> {
    rows("weights")
        .iter()
        .map(|w| (hex(&w[0]), hex(&w[1])))
        .collect()
}

/// Bitwise equality, treating every NaN as equal (Python prints them all as `nan`).
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// Distance in units in the last place (0 for equal values and for +0 vs -0).
fn ulps(a: f64, b: f64) -> u128 {
    let key = |x: f64| {
        let bits = x.to_bits();
        let magnitude = i128::from(bits & !(1 << 63));
        if bits >> 63 == 1 {
            -magnitude
        } else {
            magnitude
        }
    };
    (key(a) - key(b)).unsigned_abs()
}

/// Counts of a bitwise comparison, with the first few differences.
#[derive(Default)]
struct Tally {
    total: usize,
    differ: usize,
    max_ulps: u128,
    examples: Vec<String>,
}

impl Tally {
    fn check(&mut self, got: f64, want: f64, what: impl FnOnce() -> String) {
        self.total += 1;
        if !same(got, want) {
            self.differ += 1;
            let d = if got.is_nan() || want.is_nan() {
                u128::MAX
            } else {
                ulps(got, want)
            };
            self.max_ulps = self.max_ulps.max(d);
            if self.examples.len() < 10 {
                self.examples
                    .push(format!("{}: got {got:e}, want {want:e} ({d} ulp)", what()));
            }
        }
    }

    /// Report, and fail on a difference if this platform's `pow` is the reference's.
    fn assert_bitwise(&self, name: &str) {
        eprintln!(
            "{name}: {} values, {} not bitwise identical (max {} ulp)",
            self.total, self.differ, self.max_ulps
        );
        assert!(self.total > 0, "{name}: nothing compared");
        if *LIBM_MATCHES {
            assert!(self.differ == 0, "{name}:\n{}", self.examples.join("\n"));
        } else if self.differ > 0 {
            eprintln!("{name}, first differences:\n{}", self.examples.join("\n"));
        }
    }
}

fn lsh_fp(b: f64, r: f64) -> impl Fn(f64) -> f64 {
    move |s| 1.0 - (1.0 - s.powf(r)).powf(b)
}

fn lsh_fn(b: f64, r: f64) -> impl Fn(f64) -> f64 {
    move |s| 1.0 - (1.0 - (1.0 - s.powf(r)).powf(b))
}

fn ens_fp(b: f64, r: f64, xq: f64) -> impl Fn(f64) -> f64 {
    move |t| 1.0 - (1.0 - (t / (1.0 + xq - t)).powf(r)).powf(b)
}

fn ens_fn(b: f64, r: f64, xq: f64) -> impl Fn(f64) -> f64 {
    move |t| 1.0 - (1.0 - (1.0 - (t / (1.0 + xq - t)).powf(r)).powf(b))
}

/// `_quadpack._qagse(f, a, b, (), 1, 1.49e-8, 1.49e-8, 50)` on datasketch's
/// integrands: result, abserr, neval, ier, last and (for some) the work lists.
#[test]
#[allow(clippy::cast_precision_loss)] // b and r are small
fn qagse_matches_scipy_on_datasketch_integrands() {
    let mut values = Tally::default();
    let mut paths = Vec::new(); // cases whose counters or work lists differ
    let mut with_lists = 0;
    for case in rows("qagse") {
        let (b, r) = (uint(&case["b"]) as f64, uint(&case["r"]) as f64);
        let (lo, hi) = (hex(&case["a"]), hex(&case["bound"]));
        let kind = case["kind"].as_str().unwrap();
        let (eps_abs, eps_rel, limit) = (DEFAULT_EPSABS, DEFAULT_EPSREL, DEFAULT_LIMIT);
        let got = match kind {
            "lsh_fp" => qagse(lsh_fp(b, r), lo, hi, eps_abs, eps_rel, limit),
            "lsh_fn" => qagse(lsh_fn(b, r), lo, hi, eps_abs, eps_rel, limit),
            "ens_fp" => qagse(
                ens_fp(b, r, hex(&case["xq"])),
                lo,
                hi,
                eps_abs,
                eps_rel,
                limit,
            ),
            "ens_fn" => qagse(
                ens_fn(b, r, hex(&case["xq"])),
                lo,
                hi,
                eps_abs,
                eps_rel,
                limit,
            ),
            _ => panic!("{kind}"),
        };
        let label = || format!("{case}");
        values.check(got.result, hex(&case["result"]), label);
        values.check(got.abserr, hex(&case["abserr"]), label);
        let counters = (got.neval, usize::from(got.ier), got.last);
        let want = (
            uint(&case["neval"]),
            uint(&case["ier"]),
            uint(&case["last"]),
        );
        let mut same_path = counters == want;
        if let Some(lists) = case["lists"].as_array() {
            with_lists += 1;
            let got_lists = [&got.alist, &got.blist, &got.rlist, &got.elist];
            for (i, (got_list, want)) in got_lists.into_iter().zip(lists).enumerate() {
                let want = want.as_array().unwrap();
                same_path &= got_list.len() == want.len();
                for (g, w) in got_list.iter().zip(want) {
                    values.check(*g, hex(w), || format!("list {i} of {case}"));
                }
            }
            let want_iord: Vec<usize> = lists[4].as_array().unwrap().iter().map(uint).collect();
            same_path &= got.iord.get(..want_iord.len()) == Some(want_iord.as_slice());
        }
        if !same_path {
            paths.push(format!("{counters:?} vs {want:?}: {case}"));
        }
    }
    assert!(with_lists > 0);
    eprintln!(
        "qagse: {} cases, {} with other counters or iord",
        rows("qagse").len(),
        paths.len()
    );
    if *LIBM_MATCHES {
        assert!(paths.is_empty(), "{}", paths.join("\n"));
    }
    values.assert_bitwise("qagse");
}

#[test]
fn lsh_probabilities_match_datasketch() {
    let mut tally = Tally::default();
    for row in rows("lsh_probabilities") {
        let (t, b, r) = (hex(&row[0]), uint(&row[1]), uint(&row[2]));
        tally.check(lsh_false_positive(t, b, r), hex(&row[3]), || {
            format!("fp {row}")
        });
        tally.check(lsh_false_negative(t, b, r), hex(&row[4]), || {
            format!("fn {row}")
        });
    }
    tally.assert_bitwise("lsh probabilities");
}

#[test]
fn ensemble_probabilities_match_datasketch() {
    let mut tally = Tally::default();
    for row in rows("ensemble_probabilities") {
        let (t, b, r, xq) = (hex(&row[0]), uint(&row[1]), uint(&row[2]), hex(&row[3]));
        tally.check(ensemble_false_positive(t, b, r, xq), hex(&row[4]), || {
            format!("fp {row}")
        });
        tally.check(ensemble_false_negative(t, b, r, xq), hex(&row[5]), || {
            format!("fn {row}")
        });
    }
    tally.assert_bitwise("ensemble probabilities");
}

/// Compare `lsh_optimal_param` with every recorded argmin with `num_perm <= max_perm`.
fn check_lsh_grid(max_perm: usize) {
    let weights = weights();
    let num_perms: Vec<usize> = rows("lsh_num_perms").iter().map(uint).collect();
    let mut cases: Vec<(f64, usize, usize, (usize, usize))> = Vec::new();
    for row in rows("lsh_optimal_param") {
        let t = hex(&row[0]);
        let opt: Vec<usize> = row[1].as_array().unwrap().iter().map(uint).collect();
        assert_eq!(opt.len(), 2 * num_perms.len() * weights.len());
        for (ni, &n) in num_perms.iter().enumerate() {
            for wi in 0..weights.len() {
                let k = 2 * (ni * weights.len() + wi);
                if n <= max_perm {
                    cases.push((t, n, wi, (opt[k], opt[k + 1])));
                }
            }
        }
    }
    let start = Instant::now();
    let wrong: Vec<String> = cases
        .par_iter()
        .filter_map(|&(t, n, wi, want)| {
            let (fpw, fnw) = weights[wi];
            let got = lsh_optimal_param(t, n, fpw, fnw);
            (got != want).then(|| {
                format!(
                    "threshold {t:e}, num_perm {n}, weights {:?}: got {got:?}, want {want:?}",
                    weights[wi]
                )
            })
        })
        .collect();
    let largest = cases.iter().map(|c| c.1).max().unwrap_or(0);
    eprintln!(
        "lsh_optimal_param: {} searches, num_perm up to {largest}, {} differ ({:.1} s)",
        cases.len(),
        wrong.len(),
        start.elapsed().as_secs_f64()
    );
    assert_ne!(cases.len(), 0);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

#[test]
fn lsh_optimal_param_matches_datasketch() {
    // num_perm up to 128 for every threshold and weight pair (a few seconds
    // in a debug build); the full grid is the ignored test below.
    check_lsh_grid(128);
}

#[test]
#[ignore = "slow in debug builds; run with --release -- --ignored"]
fn lsh_optimal_param_matches_datasketch_full_grid() {
    check_lsh_grid(usize::MAX);
}

/// `MinHashLSHEnsemble(...).params`: one search per recorded `xq` with
/// `max_r = m`. The recordings include constructions that datasketch then
/// rejects (`ValueError: The number of bands are too small (b < 2)`), after
/// computing these params.
#[test]
fn ensemble_params_match_datasketch() {
    let xqs: Vec<f64> = rows("ensemble_xqs").iter().map(hex).collect();
    assert_eq!(xqs.len(), 10);
    let cases = rows("ensemble_params");
    for row in cases {
        let (t, n, m) = (hex(&row[0]), uint(&row[1]), uint(&row[3]));
        let (fpw, fnw) = (hex(&row[4][0]), hex(&row[4][1]));
        let want: Vec<(usize, usize)> = row[5]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (uint(&p[0]), uint(&p[1])))
            .collect();
        let got: Vec<(usize, usize)> = xqs
            .par_iter()
            .map(|&xq| ensemble_optimal_param(t, n, m, xq, fpw, fnw))
            .collect();
        assert_eq!(got, want, "{row}");
    }
    eprintln!("ensemble params: {} constructions", cases.len());
    assert_ne!(cases.len(), 0);
}

/// Direct `lshensemble._optimal_param` calls, including a NaN threshold,
/// `xq = 1` and `max_r` 1 and 3.
#[test]
fn ensemble_optimal_param_matches_datasketch() {
    let xqs: Vec<f64> = rows("ensemble_direct_xqs").iter().map(hex).collect();
    let cases = rows("ensemble_optimal_param");
    for row in cases {
        let (t, n, m) = (hex(&row[0]), uint(&row[1]), uint(&row[2]));
        let (fpw, fnw) = (hex(&row[3][0]), hex(&row[3][1]));
        let opt: Vec<usize> = row[4].as_array().unwrap().iter().map(uint).collect();
        assert_eq!(opt.len(), 2 * xqs.len());
        for (i, &xq) in xqs.iter().enumerate() {
            let want = (opt[2 * i], opt[2 * i + 1]);
            assert_eq!(
                ensemble_optimal_param(t, n, m, xq, fpw, fnw),
                want,
                "xq {xq:e}: {row}"
            );
        }
    }
    assert_ne!(cases.len(), 0);
}

/// Every probability the full argmin grid evaluates (LSH: 109 thresholds x
/// every `b * r <= 512`; ensemble: 15 thresholds x 10 `xq` x every
/// `b * r <= 256` with `r <= 16`), from `$DATASKETCH_PARAMS_FULL`.
#[test]
#[ignore = "needs DATASKETCH_PARAMS_FULL; run with --release -- --ignored"]
fn probabilities_match_datasketch_full_grid() {
    let Ok(dir) = std::env::var("DATASKETCH_PARAMS_FULL") else {
        eprintln!("DATASKETCH_PARAMS_FULL is not set; skipped");
        return;
    };
    let load = |name: &str| -> Value {
        let path = format!("{dir}/{name}");
        serde_json::from_str(&std::fs::read_to_string(&path).expect(&path)).expect("json")
    };
    let pairs = |v: &Value| -> Vec<(usize, usize)> {
        v["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| (uint(&p[0]), uint(&p[1])))
            .collect()
    };

    let lsh = load("full_lsh_probs.json");
    let lsh_pairs = pairs(&lsh);
    let tallies: Vec<Tally> = lsh["rows"]
        .as_array()
        .unwrap()
        .par_iter()
        .map(|row| {
            let t = hex(&row[0]);
            let (fps, fns) = (row[1].as_array().unwrap(), row[2].as_array().unwrap());
            let mut tally = Tally::default();
            for (j, &(b, r)) in lsh_pairs.iter().enumerate() {
                tally.check(lsh_false_positive(t, b, r), hex(&fps[j]), || {
                    format!("fp {t:e} {b} {r}")
                });
                tally.check(lsh_false_negative(t, b, r), hex(&fns[j]), || {
                    format!("fn {t:e} {b} {r}")
                });
            }
            tally
        })
        .collect();
    merge(tallies).assert_bitwise("full lsh probability grid");

    let ens = load("full_ens_probs.json");
    let ens_pairs = pairs(&ens);
    let tallies: Vec<Tally> = ens["rows"]
        .as_array()
        .unwrap()
        .par_iter()
        .flat_map_iter(|row| {
            let t = hex(&row[0]);
            let ens_pairs = &ens_pairs;
            row[1].as_array().unwrap().iter().map(move |per_xq| {
                let xq = hex(&per_xq[0]);
                let (fps, fns) = (per_xq[1].as_array().unwrap(), per_xq[2].as_array().unwrap());
                let mut tally = Tally::default();
                for (j, &(b, r)) in ens_pairs.iter().enumerate() {
                    tally.check(ensemble_false_positive(t, b, r, xq), hex(&fps[j]), || {
                        format!("fp {t:e} {b} {r} {xq:e}")
                    });
                    tally.check(ensemble_false_negative(t, b, r, xq), hex(&fns[j]), || {
                        format!("fn {t:e} {b} {r} {xq:e}")
                    });
                }
                tally
            })
        })
        .collect();
    merge(tallies).assert_bitwise("full ensemble probability grid");
}

fn merge(tallies: Vec<Tally>) -> Tally {
    let mut all = Tally::default();
    for t in tallies {
        all.total += t.total;
        all.differ += t.differ;
        all.max_ulps = all.max_ulps.max(t.max_ulps);
        all.examples.extend(t.examples);
    }
    all.examples.truncate(20);
    all
}

/// Indicative timings: `cargo test --release -p pyroid-core --test
/// datasketch_params -- --ignored --nocapture time_datasketch`.
#[test]
#[ignore = "benchmark"]
fn time_datasketch_lsh_optimal_param() {
    for (t, n) in [(0.9, 128), (0.9, 256), (0.5, 512)] {
        let mut times = Vec::new();
        let mut opt = (0, 0);
        for _ in 0..20 {
            let start = Instant::now();
            opt = std::hint::black_box(lsh_optimal_param(std::hint::black_box(t), n, 0.5, 0.5));
            times.push(start.elapsed().as_secs_f64());
        }
        times.sort_by(f64::total_cmp);
        eprintln!(
            "lsh_optimal_param({t}, {n}, 0.5, 0.5) = {opt:?}: min {:.2} ms, median {:.2} ms (20 runs, one thread)",
            times[0] * 1e3,
            times[times.len() / 2] * 1e3
        );
    }
}
