//! `ratio_batch` and `close_matches_batch`: `SequenceMatcher` over many
//! strings at once, on the shared pool, with Ctrl-C checked between chunks.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyString};
use pyroid_core::difflib::{StrMatcher, nlargest, with_work};
use pyroid_core::par;
use rustc_hash::FxHashSet;

use crate::batch::{BatchError, rows};
use crate::errors::input_error;
use crate::input::{self, TextInput, Threads, text_arg};

/// Pairs (or words) per chunk between checks for signals.
const CHUNK: usize = 1 << 14;

fn code_points(s: &str) -> impl Iterator<Item = u32> + '_ {
    s.chars().map(u32::from)
}

fn char_len(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// `isjunk`'s verdicts for every distinct character of `texts` (it is called
/// once per character, here on the calling thread), or `None`.
fn junk_set(isjunk: &Bound<'_, PyAny>, texts: &[Option<&str>]) -> PyResult<Option<FxHashSet<u32>>> {
    if isjunk.is_none() || !isjunk.is_truthy()? {
        return Ok(None);
    }
    let py = isjunk.py();
    let mut seen: Vec<char> = py.detach(|| {
        let mut seen = FxHashSet::default();
        for s in texts.iter().flatten() {
            seen.extend(s.chars());
        }
        seen.into_iter().collect()
    });
    seen.sort_unstable();
    let mut junk = FxHashSet::default();
    let mut buf = [0; 4];
    for c in seen {
        if isjunk
            .call1((PyString::new(py, c.encode_utf8(&mut buf)),))?
            .is_truthy()?
        {
            junk.insert(u32::from(c));
        }
    }
    Ok(Some(junk))
}

/// `SequenceMatcher(isjunk, a[i], b[i], autojunk).ratio()` for every `i`.
#[pyfunction]
#[pyo3(signature = (a, b, *, isjunk=None, autojunk=true, threads=None))]
#[allow(clippy::many_single_char_names)] // difflib's names
pub fn ratio_batch<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    isjunk: Option<&Bound<'py, PyAny>>,
    autojunk: bool,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyList>> {
    let ta = text_arg(a, threads, "a")?;
    let tb = text_arg(b, threads, "b")?;
    let ra = rows(&ta).map_err(|e| e.into_pyerr(py))?;
    let rb = rows(&tb).map_err(|e| e.into_pyerr(py))?;
    let (sa, sb) = (
        matches!(ta, TextInput::Str(_)),
        matches!(tb, TextInput::Str(_)),
    );
    let n = match (sa, sb) {
        (true, true) => 1,
        (true, false) => rb.len(),
        (false, true) => ra.len(),
        (false, false) if ra.len() == rb.len() => ra.len(),
        (false, false) => {
            return Err(PyValueError::new_err(format!(
                "a and b must have the same length (or one be a str), got {} and {}",
                ra.len(),
                rb.len()
            )));
        }
    };
    let none = py.None().into_bound(py);
    let junk = junk_set(isjunk.unwrap_or(&none), &rb)?;
    let junk_char = |c: u32| junk.as_ref().is_some_and(|j| j.contains(&c));
    let t = input::threads(threads);
    let fixed_b = sb.then(|| {
        rb[0].map(|s| {
            let cps: Vec<u32> = code_points(s).collect();
            StrMatcher::new(&cps, junk_char, autojunk)
        })
    });
    let pair = |i: usize| -> Option<f64> {
        let x = ra[if sa { 0 } else { i }]?;
        match &fixed_b {
            Some(Some(m)) => Some(with_work(|w| m.ratio_with(code_points(x), w))),
            Some(None) => None,
            None => {
                let y = rb[i]?;
                Some(with_work(|w| {
                    thread_local! {
                        static M: std::cell::RefCell<StrMatcher> = std::cell::RefCell::new(StrMatcher::empty());
                    }
                    M.with(|m| {
                        let mut m = m.borrow_mut();
                        m.reset(code_points(y), junk_char, autojunk, w);
                        m.ratio_with(code_points(x), w)
                    })
                }))
            }
        }
    };
    let mut out: Vec<Option<f64>> = Vec::with_capacity(n);
    let mut start = 0;
    while start < n {
        let len = CHUNK.min(n - start);
        let part = py
            .detach(|| par::map(len, t, |i| pair(start + i)))
            .map_err(|e| BatchError::from(e).into_pyerr(py))?;
        out.extend(part);
        start += len;
        py.check_signals()?;
    }
    PyList::new(py, out)
}

/// `get_close_matches(word, possibilities, n, cutoff)` for every word, after
/// `close_matches_batch`'s argument checks (`n > 0`, `0 <= cutoff <= 1`).
#[pyfunction]
#[pyo3(signature = (words, possibilities, n, cutoff, threads=None))]
#[allow(clippy::many_single_char_names)] // difflib's names
pub fn _close_matches_batch<'py>(
    py: Python<'py>,
    words: &Bound<'py, PyAny>,
    possibilities: &Bound<'py, PyAny>,
    n: usize,
    cutoff: f64,
    threads: Option<Threads>,
) -> PyResult<Bound<'py, PyList>> {
    let tw = text_arg(words, threads, "words")?;
    let tp = text_arg(possibilities, threads, "possibilities")?;
    if matches!(tp, TextInput::Str(_)) {
        return Err(input_error(
            py,
            "possibilities must be a list, tuple or Arrow array of str, not a str",
        ));
    }
    let rw = rows(&tw).map_err(|e| e.into_pyerr(py))?;
    let rp = rows(&tp).map_err(|e| e.into_pyerr(py))?;
    let lens: Vec<usize> = rp.iter().map(|s| s.map_or(0, char_len)).collect();
    let t = input::threads(threads);
    let one = |word: &str, t: par::Threads| -> Result<Vec<usize>, BatchError> {
        let cps: Vec<u32> = code_points(word).collect();
        let m = StrMatcher::new(&cps, |_| false, true);
        let scored: Vec<Option<f64>> = par::map(rp.len(), t, |i| {
            let x = rp[i]?;
            with_work(|w| m.close_match_with(lens[i], code_points(x), cutoff, w))
        })?;
        let scored: Vec<(f64, usize)> = scored
            .into_iter()
            .enumerate()
            .filter_map(|(i, r)| r.map(|r| (r, i)))
            .collect();
        Ok(nlargest(&scored, n, |x, y| {
            let (x, y) = (rp[x].unwrap_or(""), rp[y].unwrap_or(""));
            x.as_bytes().cmp(y.as_bytes())
        }))
    };
    let mut picked: Vec<Option<Vec<usize>>> = Vec::with_capacity(rw.len());
    if rw.len() >= 2 * par::MIN_DOCS_PER_TASK {
        // Many words: one word per task.
        let mut start = 0;
        while start < rw.len() {
            let len = (CHUNK / 16).max(1).min(rw.len() - start);
            let part = py
                .detach(|| {
                    par::try_map(len, t, |i| {
                        rw[start + i]
                            .map(|word| one(word, par::Threads::ONE))
                            .transpose()
                    })
                })
                .map_err(|e| e.into_pyerr(py))?;
            picked.extend(part);
            start += len;
            py.check_signals()?;
        }
    } else {
        for word in &rw {
            let found = match word {
                None => None,
                Some(word) => Some(py.detach(|| one(word, t)).map_err(|e| e.into_pyerr(py))?),
            };
            picked.push(found);
            py.check_signals()?;
        }
    }
    let objs = crate::batch::to_list(py, &tp)?;
    let out = PyList::empty(py);
    for found in picked {
        match found {
            None => out.append(py.None())?,
            Some(found) => {
                let items = found
                    .iter()
                    .map(|&i| objs.get_item(i))
                    .collect::<PyResult<Vec<_>>>()?;
                out.append(PyList::new(py, items)?)?;
            }
        }
    }
    Ok(out)
}
