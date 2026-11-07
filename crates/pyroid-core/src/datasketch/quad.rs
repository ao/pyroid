// Derived from scipy 1.18.1 (BSD-3-Clause), © 2024 SciPy developers; modified.
//! QUADPACK's `dqagse`, as `scipy.integrate.quad` runs it for finite limits.
//!
//! This is a port of the C translation of QUADPACK in scipy 1.18.1,
//! `scipy/integrate/__quadpack.c`: [`qagse`] is `dqagse` (adaptive bisection
//! with Wynn's epsilon extrapolation), and the private `dqk21` (21-point
//! Gauss-Kronrod rule), `dqpsrt` (error-list ordering) and `dqelg` (epsilon
//! table) are ported with it. [`quad`] is the part of `scipy.integrate.quad`
//! (`scipy/integrate/_quadpack_py.py`) that applies to finite limits and the
//! default options. QUADPACK itself is by R. Piessens, E. de Doncker-Kapenga,
//! C. Überhuber and D. Kahaner (1983).
//!
//! Every floating-point operation happens in the same order as in the C, with
//! the same `fmax`/`fmin` and comparison semantics for NaN, and without fused
//! multiply-adds (scipy 1.18.1's `_quadpack` for aarch64 has none either), so
//! for the same integrand values the results are bit-for-bit scipy's,
//! including `abserr`, `neval`, `ier` and the work lists. Indices are 0-based,
//! as in the C. The machine constants are `d1mach`'s: `epmach = 2**-52`,
//! `uflow = 2**-1022` and `oflow = f64::MAX`.
#![allow(clippy::float_cmp)] // QUADPACK compares floats exactly, by design

/// `d1mach(4)`: the largest relative spacing, `np.finfo(np.float64).eps`.
const EPMACH: f64 = f64::EPSILON;
/// `d1mach(1)`: the smallest positive normal magnitude, `np.finfo(np.float64).tiny`.
const UFLOW: f64 = f64::MIN_POSITIVE;
/// `d1mach(2)`: the largest finite magnitude, `np.finfo(np.float64).max`.
const OFLOW: f64 = f64::MAX;

/// The defaults of `scipy.integrate.quad`.
pub const DEFAULT_EPSABS: f64 = 1.49e-8;
/// See [`DEFAULT_EPSABS`].
pub const DEFAULT_EPSREL: f64 = 1.49e-8;
/// See [`DEFAULT_EPSABS`].
pub const DEFAULT_LIMIT: usize = 50;

// Gauss quadrature weights and Kronrod abscissae and weights, as evaluated with
// 80 decimal digit arithmetic by L. W. Fullerton, Bell Labs, Nov. 1981; the
// literals are QUADPACK's, digit for digit.
#[allow(clippy::excessive_precision, clippy::unreadable_literal)] // copied from QUADPACK
const WG: [f64; 5] = [
    0.066671344308688137593568809893332,
    0.149451349150580593145776339657697,
    0.219086362515982043995534934228163,
    0.269266719309996355091226921569469,
    0.295524224714752870173892994651338,
];
#[allow(clippy::excessive_precision, clippy::unreadable_literal)] // copied from QUADPACK
const XGK: [f64; 11] = [
    0.995657163025808080735527280689003,
    0.973906528517171720077964012084452,
    0.930157491355708226001207180059508,
    0.865063366688984510732096688423493,
    0.780817726586416897063717578345042,
    0.679409568299024406234327365114874,
    0.562757134668604683339000099272694,
    0.433395394129247190799265943165784,
    0.294392862701460198131126603103866,
    0.148874338981631210884826001129720,
    0.000000000000000000000000000000000,
];
#[allow(clippy::excessive_precision, clippy::unreadable_literal)] // copied from QUADPACK
const WGK: [f64; 11] = [
    0.011694638867371874278064396062192,
    0.032558162307964727478818972459390,
    0.054755896574351996031381300244580,
    0.075039674810919952767043140916190,
    0.093125454583697605535065465083366,
    0.109387158802297641899210590325805,
    0.123491976262065851077958109831074,
    0.134709217311473325928054001771707,
    0.142775938577060080797094273138717,
    0.147739104901338491374841515972068,
    0.149445554002916905664936468389821,
];

/// What `dqagse` returns: scipy's `_quadpack._qagse(..., full_output=1)`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuadResult {
    /// The approximation to the integral.
    pub result: f64,
    /// The estimate of the absolute error.
    pub abserr: f64,
    /// The number of integrand evaluations.
    pub neval: usize,
    /// QUADPACK's error code: 0 on success, 1 (subdivision limit reached),
    /// 2 (roundoff error), 3 (bad integrand behaviour), 4 (no convergence),
    /// 5 (probably divergent), 6 (invalid input). `scipy.integrate.quad` warns
    /// with `IntegrationWarning` for 1 to 5 and raises `ValueError` for 6.
    pub ier: u8,
    /// The number of subintervals produced.
    pub last: usize,
    /// Left end points of the subintervals (the first `last` of scipy's `alist`).
    pub alist: Vec<f64>,
    /// Right end points of the subintervals (`blist`).
    pub blist: Vec<f64>,
    /// Integral approximations on the subintervals (`rlist`).
    pub rlist: Vec<f64>,
    /// Error estimates on the subintervals (`elist`).
    pub elist: Vec<f64>,
    /// 0-based pointers into `elist` (`iord`). Only the first `last` entries are
    /// kept, and of those only the first `k` are ordered, with `k = last` if
    /// `last <= limit / 2 + 2` and `k = limit + 1 - last` otherwise.
    pub iord: Vec<usize>,
}

/// Integrate `f` over the finite interval `[a, b]`, as `scipy.integrate.quad(f, a, b)`.
///
/// Uses scipy's defaults (`epsabs=1.49e-8`, `epsrel=1.49e-8`, `limit=50`) and
/// returns `(result, abserr)`, ignoring QUADPACK's error code as `quad` does
/// (it only warns). As in `quad`, an empty interval (`a == b`) returns
/// `(0.0, 0.0)` without calling `f`, and `b < a` integrates over `[b, a]` and
/// negates the result. Python's `min`/`max` choose the limits, so a NaN limit
/// behaves as it does in scipy. Infinite limits are not supported: scipy sends
/// them to `dqagie`, which is not ported.
pub fn quad(mut f: impl FnMut(f64) -> f64, a: f64, b: f64) -> (f64, f64) {
    if a == b {
        return (0.0, 0.0);
    }
    let flip = b < a;
    // `min(a, b), max(a, b)`: the first argument unless the second compares
    // smaller (larger).
    let lo = if b < a { b } else { a };
    let hi = if b > a { b } else { a };
    // The work arrays live on the stack: `quad` does not return them.
    let mut alist = [0.0; DEFAULT_LIMIT];
    let mut blist = [0.0; DEFAULT_LIMIT];
    let mut rlist = [0.0; DEFAULT_LIMIT];
    let mut elist = [0.0; DEFAULT_LIMIT];
    let mut iord = [0; DEFAULT_LIMIT];
    let w = Work {
        alist: &mut alist,
        blist: &mut blist,
        rlist: &mut rlist,
        elist: &mut elist,
        iord: &mut iord,
    };
    let (result, abserr, _, _) = dqagse(&mut f, lo, hi, DEFAULT_EPSABS, DEFAULT_EPSREL, w);
    (if flip { -result } else { result }, abserr)
}

/// QUADPACK's `dqagse`: integrate `f` over `[a, b]` to within
/// `max(epsabs, epsrel * |integral|)`, using at most `limit` subintervals.
///
/// Port of `dqagse` in scipy 1.18.1's `__quadpack.c`, called the way
/// `_quadpack._qagse` calls it: `limit == 0` returns `ier = 6` without
/// evaluating `f`, as scipy's wrapper does.
pub fn qagse(
    mut f: impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    limit: usize,
) -> QuadResult {
    let mut alist = vec![0.0; limit];
    let mut blist = vec![0.0; limit];
    let mut rlist = vec![0.0; limit];
    let mut elist = vec![0.0; limit];
    let mut iord = vec![0; limit];
    let (result, abserr, ier, last) = if limit == 0 {
        (0.0, 0.0, 6, 0)
    } else {
        let w = Work {
            alist: &mut alist,
            blist: &mut blist,
            rlist: &mut rlist,
            elist: &mut elist,
            iord: &mut iord,
        };
        dqagse(&mut f, a, b, epsabs, epsrel, w)
    };
    alist.truncate(last);
    blist.truncate(last);
    rlist.truncate(last);
    elist.truncate(last);
    iord.truncate(last);
    QuadResult {
        result,
        abserr,
        // LINE140; the invalid-input return skips it and leaves 0.
        neval: if last == 0 { 0 } else { 42 * last - 21 },
        ier,
        last,
        alist,
        blist,
        rlist,
        elist,
        iord,
    }
}

/// Where `dqagse` goes after its main loop (the C's labels).
enum Tail {
    /// LINE115: sum `rlist` for the result, `abserr = errsum`.
    Sum,
    /// LINE110: the divergence test.
    Divergence,
    /// LINE130: keep `result` and `abserr`.
    Done,
}

/// `dqagse`'s work arrays, all `limit` long.
struct Work<'a> {
    alist: &'a mut [f64],
    blist: &'a mut [f64],
    rlist: &'a mut [f64],
    elist: &'a mut [f64],
    iord: &'a mut [usize],
}

/// The body of `dqagse` on work arrays of length `limit >= 1`; returns
/// `(result, abserr, ier, last)`.
#[allow(clippy::too_many_lines)] // one routine in the C
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x >= y)` keeps the C's NaN behaviour
#[allow(
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::manual_midpoint
)] // QUADPACK's names and `0.5 * (a + b)`
fn dqagse(
    f: &mut impl FnMut(f64) -> f64,
    a: f64,
    b: f64,
    epsabs: f64,
    epsrel: f64,
    w: Work<'_>,
) -> (f64, f64, u8, usize) {
    let Work {
        alist,
        blist,
        rlist,
        elist,
        iord,
    } = w;
    let limit = alist.len();
    let mut ier: u8 = 0;
    alist[0] = a;
    blist[0] = b;
    let mut ierror = 0;
    let mut erlarg = 0.0;
    let mut ertest = 0.0;
    let mut correc = 0.0;
    let mut small = 0.0;

    if epsabs <= 0.0 && epsrel < (50.0 * EPMACH).max(0.5e-28) {
        return (0.0, 0.0, 6, 0);
    }

    // First approximation to the integral. As in the C, `defabs` receives
    // dqk21's `resabs` and `resabs` its `resasc`.
    let (mut result, mut abserr, defabs, resabs) = dqk21(f, a, b);

    // Test on accuracy.
    let dres = result.abs();
    let mut errbnd = epsabs.max(epsrel * dres);
    let mut last = 1;
    rlist[0] = result;
    elist[0] = abserr;
    iord[0] = 0;

    if abserr <= 100.0 * EPMACH * defabs && abserr > errbnd {
        ier = 2;
    }
    if limit == 1 {
        ier = 1;
    }
    if ier != 0 || (abserr <= errbnd && abserr != resabs) || abserr == 0.0 {
        return (result, abserr, ier, last);
    }

    // Initialization.
    let mut rlist2 = [0.0; 52];
    let mut res3la = [0.0; 3];
    rlist2[0] = result;
    let mut errmax = abserr;
    let mut maxerr = 0;
    let mut area = result;
    let mut errsum = abserr;
    abserr = OFLOW;
    let mut nrmax = 0;
    let mut nres = 0;
    let mut numrl2 = 1;
    let mut ktmin = 0;
    let mut extrap = false;
    let mut noext = false;
    let mut iroff1 = 0;
    let mut iroff2 = 0;
    let mut iroff3 = 0;
    let ksgn = if dres >= (1.0 - 50.0 * EPMACH) * defabs {
        1
    } else {
        -1
    };

    // Main loop. It always leaves through `break` (ier is set to 1 once `last`
    // reaches `limit`), so `l` is the iteration it left in, as the C's `L`.
    let mut l = 0;
    let mut sum_rlist = false; // `goto LINE115` from inside the loop
    for iteration in 1..limit {
        l = iteration;
        last = l + 1;

        // Bisect the subinterval with the nrmax-th largest error estimate.
        let a1 = alist[maxerr];
        let b1 = 0.5 * (alist[maxerr] + blist[maxerr]);
        let a2 = b1;
        let b2 = blist[maxerr];
        let erlast = errmax;
        let (area1, error1, _, defab1) = dqk21(f, a1, b1);
        let (area2, error2, _, defab2) = dqk21(f, a2, b2);

        // Improve previous approximations to integral and error and test for
        // accuracy.
        let area12 = area1 + area2;
        let error12 = error1 + error2;
        errsum = errsum + error12 - errmax;
        area = area + area12 - rlist[maxerr];

        if defab1 != error1 && defab2 != error2 {
            if !((rlist[maxerr] - area12).abs() > 1.0e-5 * area12.abs() || error12 < 0.99 * errmax)
            {
                if extrap {
                    iroff2 += 1;
                } else {
                    iroff1 += 1;
                }
            }
            if l > 9 && error12 > errmax {
                iroff3 += 1;
            }
        }

        rlist[maxerr] = area1;
        rlist[l] = area2;
        errbnd = epsabs.max(epsrel * area.abs());

        // Test for roundoff error and eventually set error flag.
        if iroff1 + iroff2 >= 10 || iroff3 >= 20 {
            ier = 2;
        }
        if iroff2 >= 5 {
            ierror = 3;
        }

        // Set error flag in the case that the number of subintervals equals limit.
        if last == limit {
            ier = 1;
        }

        // Set error flag in the case of bad integrand behaviour at some points
        // of the integration range.
        if a1.abs().max(b2.abs()) <= (1.0 + 100.0 * EPMACH) * (a2.abs() + 1000.0 * UFLOW) {
            ier = 4;
        }

        // Append the newly-created intervals to the list.
        if error2 > error1 {
            alist[maxerr] = a2;
            alist[l] = a1;
            blist[l] = b1;
            rlist[maxerr] = area2;
            rlist[l] = area1;
            elist[maxerr] = error2;
            elist[l] = error1;
        } else {
            alist[l] = a2;
            blist[maxerr] = b1;
            blist[l] = b2;
            elist[maxerr] = error1;
            elist[l] = error2;
        }

        // Maintain the descending ordering in the list of error estimates and
        // select the subinterval with the nrmax-th largest error estimate (to
        // be bisected next).
        dqpsrt(
            limit,
            last,
            &mut maxerr,
            &mut errmax,
            elist,
            iord,
            &mut nrmax,
        );

        if errsum <= errbnd {
            sum_rlist = true;
            break;
        }
        if ier != 0 {
            break;
        }
        if l == 1 {
            // LINE80
            small = (b - a).abs() * 0.375;
            erlarg = errsum;
            ertest = errbnd;
            rlist2[1] = area;
            continue;
        }
        if noext {
            continue;
        }
        erlarg -= erlast;
        if (b1 - a1).abs() > small {
            erlarg += error12;
        }
        if !extrap {
            // Test whether the interval to be bisected next is the smallest interval.
            if (blist[maxerr] - alist[maxerr]).abs() > small {
                continue;
            }
            extrap = true;
            nrmax = 1;
        }
        // 40
        if !(ierror == 3 || erlarg <= ertest) {
            // The smallest interval has the largest error. Before bisecting,
            // decrease the sum of the errors over the larger intervals (erlarg)
            // and perform extrapolation.
            let jupbnd = if last > 2 + limit / 2 {
                limit + 3 - last
            } else {
                last
            };
            let mut larger_found = false; // `goto LINE90`
            let first = nrmax;
            for _ in first..jupbnd {
                maxerr = iord[nrmax];
                errmax = elist[maxerr];
                if (blist[maxerr] - alist[maxerr]).abs() > small {
                    larger_found = true;
                    break;
                }
                nrmax += 1;
            }
            if larger_found {
                continue;
            }
        }
        // LINE60: perform extrapolation.
        numrl2 += 1;
        rlist2[numrl2] = area;
        let (reseps, abseps) = dqelg(&mut numrl2, &mut rlist2, &mut res3la, &mut nres);
        ktmin += 1;
        if ktmin > 5 && abserr < 1.0e-3 * errsum {
            ier = 5;
        }
        if !(abseps >= abserr) {
            ktmin = 0;
            abserr = abseps;
            result = reseps;
            correc = erlarg;
            ertest = epsabs.max(epsrel * reseps.abs());
            if abserr <= ertest {
                break;
            }
        }
        // LINE70: prepare bisection of the smallest interval.
        if numrl2 == 0 {
            noext = true;
        }
        if ier == 5 {
            break;
        }
        maxerr = iord[0];
        errmax = elist[maxerr];
        nrmax = 0;
        extrap = false;
        small *= 0.5;
        erlarg = errsum;
    }

    // Set final result and error estimate.
    let tail = if sum_rlist || abserr == OFLOW {
        Tail::Sum
    } else if ier + ierror == 0 {
        Tail::Divergence
    } else {
        if ierror == 3 {
            abserr += correc;
        }
        if ier == 0 {
            ier = 3;
        }
        if result != 0.0 && area != 0.0 {
            // LINE105
            if abserr / result.abs() > errsum / area.abs() {
                Tail::Sum
            } else {
                Tail::Divergence
            }
        } else if abserr > errsum {
            Tail::Sum
        } else if area == 0.0 {
            Tail::Done
        } else {
            Tail::Divergence
        }
    };
    match tail {
        Tail::Divergence => {
            // Test on divergence.
            if !(ksgn == -1 && result.abs().max(area.abs()) <= defabs * 0.01) {
                let ratio = result / area;
                if 0.01 > ratio || ratio > 100.0 || errsum > area.abs() {
                    ier = 6;
                }
            }
        }
        Tail::Sum => {
            // Compute global integral sum (from +0.0, as the C).
            result = 0.0;
            for &v in &rlist[..=l] {
                result += v;
            }
            abserr = errsum;
        }
        Tail::Done => {}
    }
    // LINE130
    if ier > 2 {
        ier -= 1;
    }
    (result, abserr, ier, last)
}

/// QUADPACK's `dqk21`: the 21-point Kronrod rule (with the embedded 10-point
/// Gauss rule for the error estimate) over `[a, b]`.
///
/// Returns `(result, abserr, resabs, resasc)`: the integral, the error
/// estimate, the integral of `|f|` and the integral of `|f - mean|`.
#[allow(clippy::similar_names, clippy::manual_midpoint)] // QUADPACK's names and `0.5 * (a + b)`
fn dqk21(f: &mut impl FnMut(f64) -> f64, a: f64, b: f64) -> (f64, f64, f64, f64) {
    let centr = 0.5 * (a + b);
    let hlgth = 0.5 * (b - a);
    let dhlgth = hlgth.abs();

    // Compute the 21-point Kronrod approximation to the integral, and estimate
    // the absolute error.
    let mut fv1 = [0.0; 10];
    let mut fv2 = [0.0; 10];
    let fc = f(centr);
    let mut resg = 0.0;
    let mut resk = WGK[10] * fc;
    let mut resabs = resk.abs();

    for (j, wg) in WG.iter().enumerate() {
        let jtw = 2 * j + 1;
        let absc = hlgth * XGK[jtw];
        let fval1 = f(centr - absc);
        let fval2 = f(centr + absc);
        fv1[jtw] = fval1;
        fv2[jtw] = fval2;
        let fsum = fval1 + fval2;
        resg += wg * fsum;
        resk += WGK[jtw] * fsum;
        resabs += WGK[jtw] * (fval1.abs() + fval2.abs());
    }

    for j in 0..5 {
        let jtwm1 = 2 * j;
        let absc = hlgth * XGK[jtwm1];
        let fval1 = f(centr - absc);
        let fval2 = f(centr + absc);
        fv1[jtwm1] = fval1;
        fv2[jtwm1] = fval2;
        let fsum = fval1 + fval2;
        resk += WGK[jtwm1] * fsum;
        resabs += WGK[jtwm1] * (fval1.abs() + fval2.abs());
    }

    let reskh = resk * 0.5;
    let mut resasc = WGK[10] * (fc - reskh).abs();
    for ((w, v1), v2) in WGK.iter().zip(&fv1).zip(&fv2) {
        resasc += w * ((v1 - reskh).abs() + (v2 - reskh).abs());
    }

    let result = resk * hlgth;
    resabs *= dhlgth;
    resasc *= dhlgth;
    let mut abserr = ((resk - resg) * hlgth).abs();

    if resasc != 0.0 && abserr != 0.0 {
        abserr = resasc * 1.0_f64.min((200.0 * abserr / resasc).powf(1.5));
    }
    if resabs > UFLOW / (50.0 * EPMACH) {
        abserr = (EPMACH * 50.0 * resabs).max(abserr);
    }
    (result, abserr, resabs, resasc)
}

/// QUADPACK's `dqpsrt`: keep `iord` pointing at the error estimates in
/// descending order after a bisection, and select the `nrmax`-th largest.
#[allow(clippy::similar_names)] // QUADPACK's names (`errmax`, `ermax`)
fn dqpsrt(
    limit: usize,
    last: usize,
    maxerr: &mut usize,
    ermax: &mut f64,
    elist: &[f64],
    iord: &mut [usize],
    nrmax: &mut usize,
) {
    if last <= 2 {
        iord[0] = 0;
        iord[1] = 1;
        *maxerr = iord[*nrmax];
        *ermax = elist[*maxerr];
        return;
    }

    // This part is only executed if, due to a difficult integrand, subdivision
    // increased the error estimate. In the normal case the insert procedure
    // should start after the nrmax-th largest error estimate.
    let errmax = elist[*maxerr];
    while *nrmax > 0 && errmax > elist[iord[*nrmax - 1]] {
        iord[*nrmax] = iord[*nrmax - 1];
        *nrmax -= 1;
    }

    // The number of elements to keep in descending order depends on the
    // number of subdivisions still allowed.
    let jupbn = if last > limit / 2 + 2 {
        limit - last + 2
    } else {
        last - 1
    };
    let errmin = elist[last - 1];

    // Insert errmax by traversing the list top-down.
    let jbnd = jupbn - 1;
    let ibeg = *nrmax + 1;
    let mut insert_at = None; // `goto LINE60`
    if ibeg <= jbnd {
        for i in ibeg..=jbnd {
            if errmax >= elist[iord[i]] {
                insert_at = Some(i);
                break;
            }
            iord[i - 1] = iord[i];
        }
    }
    let Some(i) = insert_at else {
        iord[jbnd] = *maxerr;
        iord[jupbn] = last - 1;
        *maxerr = iord[*nrmax];
        *ermax = elist[*maxerr];
        return;
    };

    // LINE60: insert errmin by traversing the list bottom-up.
    iord[i - 1] = *maxerr;
    let mut k = jbnd;
    for _ in i..=jbnd {
        if errmin < elist[iord[k]] {
            break;
        }
        iord[k + 1] = iord[k];
        k -= 1;
    }
    if errmin < elist[iord[k]] {
        iord[k + 1] = last - 1;
    } else {
        iord[i] = last - 1;
    }
    *maxerr = iord[*nrmax];
    *ermax = elist[*maxerr];
}

/// QUADPACK's `dqelg`: Wynn's epsilon algorithm on the 0-based `epstab`,
/// whose newest element is `epstab[n]`.
///
/// Returns `(result, abserr)` and updates `n`, `epstab`, `res3la` (the last
/// three results) and `nres` (the number of calls) in place.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(x > y)` keeps the C's NaN behaviour
#[allow(clippy::similar_names)] // QUADPACK's names
fn dqelg(
    n: &mut usize,
    epstab: &mut [f64; 52],
    res3la: &mut [f64; 3],
    nres: &mut usize,
) -> (f64, f64) {
    // The table can hold limexp + 1 elements (0-based index limexp); when it
    // is full, the upper diagonal is deleted.
    const LIMEXP: usize = 49;
    let starting_n = *n;
    *nres += 1;
    let mut abserr = OFLOW;
    let mut result = epstab[*n];
    if *n < 2 {
        abserr = abserr.max(5.0 * EPMACH * result.abs());
        return (result, abserr);
    }

    epstab[*n + 2] = epstab[*n];
    let newelm = *n / 2;
    epstab[*n] = OFLOW;
    let mut k1 = *n;
    for i in 0..newelm {
        let k2 = k1 - 1;
        let k3 = k1 - 2;
        let mut res = epstab[k1 + 2];
        let e0 = epstab[k3];
        let e1 = epstab[k2];
        let e2 = res;
        let e1abs = e1.abs();
        let delta2 = e2 - e1;
        let err2 = delta2.abs();
        let tol2 = e2.abs().max(e1abs) * EPMACH;
        let delta3 = e1 - e0;
        let err3 = delta3.abs();
        let tol3 = e1abs.max(e0.abs()) * EPMACH;
        if !(err2 > tol2 || err3 > tol3) {
            // e0, e1 and e2 are equal to within machine accuracy: convergence
            // is assumed.
            result = res;
            abserr = err2 + err3;
            abserr = abserr.max(5.0 * EPMACH * result.abs());
            return (result, abserr);
        }

        let e3 = epstab[k1];
        epstab[k1] = e1;
        let delta1 = e1 - e3;
        let err1 = delta1.abs();
        let tol1 = e1abs.max(e3.abs()) * EPMACH;

        // If two elements are very close to each other, omit a part of the
        // table by adjusting the value of n.
        if err1 <= tol1 || err2 <= tol2 || err3 <= tol3 {
            *n = i + i;
            break;
        }
        let ss = 1.0 / delta1 + 1.0 / delta2 - 1.0 / delta3;
        let epsinf = (ss * e1).abs();

        // Test to detect irregular behaviour in the table, and eventually omit
        // a part of the table by adjusting the value of n.
        if !(epsinf > 1e-4) {
            *n = i + i;
            break;
        }

        // Compute a new element and eventually adjust the value of result.
        res = e1 + 1.0 / ss;
        epstab[k1] = res;
        k1 -= 2;
        let error = err2 + (res - e2).abs() + err3;
        if !(error > abserr) {
            abserr = error;
            result = res;
        }
    }

    // Shift the table.
    if *n == LIMEXP {
        *n = 2 * (LIMEXP / 2);
    }
    let n2_rem = starting_n % 2;
    for i in 0..=newelm {
        epstab[2 * i + n2_rem] = epstab[2 * i + 2 + n2_rem];
    }
    if *n != starting_n {
        // The C copies epstab[first..=first + n] to epstab[0..=n], one by one: a forward copy.
        let first = starting_n - *n;
        epstab.copy_within(first..=first + *n, 0);
    }
    if *nres < 4 {
        res3la[*nres - 1] = result;
        abserr = OFLOW;
    } else {
        abserr =
            (result - res3la[2]).abs() + (result - res3la[1]).abs() + (result - res3la[0]).abs();
        res3la[0] = res3la[1];
        res3la[1] = res3la[2];
        res3la[2] = result;
    }
    abserr = abserr.max(5.0 * EPMACH * result.abs());
    (result, abserr)
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)] // values as Python prints them
mod tests {
    use super::*;

    #[test]
    fn machine_constants_are_d1mach() {
        assert_eq!(EPMACH, 2.0_f64.powi(-52));
        assert_eq!(EPMACH, 2.220446049250313e-16);
        assert_eq!(UFLOW, 2.2250738585072014e-308);
        assert_eq!(OFLOW, 1.7976931348623157e+308);
    }

    #[test]
    fn empty_and_reversed_intervals_follow_quad() {
        let mut calls = 0;
        assert_eq!(
            quad(
                |_| {
                    calls += 1;
                    1.0
                },
                0.25,
                0.25
            ),
            (0.0, 0.0)
        );
        assert_eq!(calls, 0);
        let (fwd, err_fwd) = quad(|x| x * x, 0.0, 4.0);
        let (rev, err_rev) = quad(|x| x * x, 4.0, 0.0);
        assert_eq!(rev, -fwd);
        assert_eq!(err_rev, err_fwd);
    }

    #[test]
    fn limit_zero_and_invalid_tolerances_return_ier_6() {
        let r = qagse(|x| x, 0.0, 1.0, 1.49e-8, 1.49e-8, 0);
        assert_eq!(
            (r.result, r.abserr, r.ier, r.neval, r.last),
            (0.0, 0.0, 6, 0, 0)
        );
        let r = qagse(|x| x, 0.0, 1.0, 0.0, 1e-20, 50);
        assert_eq!(
            (r.result, r.abserr, r.ier, r.neval, r.last),
            (0.0, 0.0, 6, 0, 0)
        );
    }

    #[test]
    fn limit_one_stops_after_the_first_rule() {
        let r = qagse(|x| 1.0 / x.sqrt(), 0.0, 1.0, 1.49e-8, 1.49e-8, 1);
        assert_eq!((r.ier, r.neval, r.last), (1, 21, 1));
    }

    #[test]
    fn nan_limits_behave_as_in_scipy() {
        // min(0.0, nan) and max(0.0, nan) are both 0.0: the interval is empty.
        assert_eq!(quad(|x| x + 1.0, 0.0, f64::NAN), (0.0, 0.0));
        // min(nan, 1.0) and max(nan, 1.0) are both nan.
        assert!(quad(|x| x, f64::NAN, 1.0).0.is_nan());
    }

    // Values recorded from scipy 1.18.1 (`_quadpack._qagse(f, a, b, (), 1)`); the
    // integrands use only correctly rounded operations, so they are the same on
    // every platform.
    fn check(f: impl FnMut(f64) -> f64, a: f64, b: f64, want: (f64, f64, usize, u8, usize)) {
        let r = qagse(f, a, b, DEFAULT_EPSABS, DEFAULT_EPSREL, DEFAULT_LIMIT);
        assert_eq!(
            (
                r.result.to_bits(),
                r.abserr.to_bits(),
                r.neval,
                r.ier,
                r.last
            ),
            (want.0.to_bits(), want.1.to_bits(), want.2, want.3, want.4),
            "got {r:?}"
        );
    }

    #[test]
    fn matches_scipy_on_reference_integrands() {
        // The quad docstring's examples (the docstring's x**2 value is from an
        // older build; scipy 1.18.1 gives these).
        check(
            |x| x * x,
            0.0,
            4.0,
            (21.333333333333336, 2.368475785867001e-13, 21, 0, 1),
        );
        let step = |x: f64| if x <= 0.0 { 1.0 } else { 0.0 };
        check(step, -1.0, 1.0, (1.0, 1.1102230246251565e-14, 63, 0, 2));
        check(
            step,
            -1.0,
            100.0,
            (1.0000000002199108, 1.0189464580163188e-08, 1365, 0, 33),
        );
        check(step, -1.0, 10000.0, (0.0, 0.0, 21, 0, 1));
        // End-point and interior singularities (extrapolation).
        check(
            |x| 1.0 / x.sqrt(),
            0.0,
            1.0,
            (1.9999999999999984, 5.773159728050814e-15, 231, 0, 6),
        );
        check(
            |x| 1.0 / (x - 1.0 / 3.0).abs().sqrt(),
            0.0,
            1.0,
            (2.7876937002347075, 2.2648549702353193e-14, 231, 0, 6),
        );
        // A divergent integral: the subdivision limit is reached.
        check(
            |x| 1.0 / x,
            0.0,
            1.0,
            (41.67684067538809, 9.35056037314051, 2079, 1, 50),
        );
    }

    #[test]
    fn nan_integrand_reports_roundoff_as_scipy() {
        let r = qagse(
            |_| f64::NAN,
            0.0,
            1.0,
            DEFAULT_EPSABS,
            DEFAULT_EPSREL,
            DEFAULT_LIMIT,
        );
        assert!(r.result.is_nan() && r.abserr.is_nan());
        assert_eq!((r.neval, r.ier, r.last), (441, 2, 11));
    }
}
