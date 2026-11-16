// Derived from jmespath 1.1.0 (MIT), © Amazon.com, Inc. or its affiliates; modified.
//! The lexer's values, made as jmespath makes them: `int(buff)`, `json.loads(lexeme)`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyString};
use pyroid_core::jmespath::{Host, Text};

use super::Config;

pub(crate) struct PyHost<'py, 'c> {
    pub py: Python<'py>,
    pub cfg: &'c Config,
}

/// A `str` from code points (lone surrogates included).
pub(crate) fn py_str<'py>(py: Python<'py>, cps: &[u32]) -> PyResult<Bound<'py, PyString>> {
    if let Some(s) = cps
        .iter()
        .map(|&c| char::from_u32(c))
        .collect::<Option<String>>()
    {
        return Ok(PyString::new(py, &s));
    }
    let bytes: Vec<u8> = cps.iter().flat_map(|c| c.to_le_bytes()).collect();
    Ok(PyBytes::new(py, &bytes)
        .call_method1("decode", ("utf-32-le", "surrogatepass"))?
        .cast_into::<PyString>()?)
}

/// A `str`'s code points (lone surrogates included).
pub(crate) fn code_points(s: &Bound<'_, PyString>) -> PyResult<Text> {
    if let Ok(t) = s.to_str() {
        return Ok(t.chars().map(u32::from).collect());
    }
    let b = s.call_method1("encode", ("utf-32-le", "surrogatepass"))?;
    let b = b.cast::<PyBytes>()?.as_bytes();
    Ok(b.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// A value made by the host: always `Some` (the conversion to an `Expr` takes it out, so it is
/// moved, not copied: in abi3 builds every reference-count change is a call into libpython).
pub(crate) type Value<'py> = Option<Bound<'py, PyAny>>;

impl<'py> Host for PyHost<'py, '_> {
    type Value = Value<'py>;
    type Error = PyErr;

    fn identifier(&mut self, text: &str) -> PyResult<Self::Value> {
        // Interned, as Python source keys are: dict lookups then hit by identity.
        Ok(Some(PyString::intern(self.py, text).into_any()))
    }

    fn number(&mut self, text: &str) -> PyResult<Self::Value> {
        self.py.get_type::<PyInt>().call1((text,)).map(Some)
    }

    fn raw_string(&mut self, text: &[u32]) -> PyResult<Self::Value> {
        Ok(Some(py_str(self.py, text)?.into_any()))
    }

    fn json(&mut self, text: &[u32]) -> PyResult<Result<Self::Value, Text>> {
        let s = py_str(self.py, text)?;
        match self.cfg.loads.bind(self.py).call1((s,)) {
            Ok(v) => Ok(Ok(Some(v))),
            Err(e) if e.is_instance_of::<PyValueError>(self.py) => {
                let msg = e.value(self.py).str()?;
                Ok(Err(code_points(&msg)?))
            }
            Err(e) => Err(e),
        }
    }

    fn deprecated_literal(&mut self) -> PyResult<()> {
        self.cfg.warn_literal.bind(self.py).call0()?;
        Ok(())
    }
}
