//! `engine.grouping.group(stmt)` on a statement made of Python token objects: read the tree
//! into the arena, group it natively, and write the changes back onto the same objects (the
//! leaves and lists keep their identity, new groups are new objects), as sqlparse's in-place
//! grouping leaves them.

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyList, PyString};
use pyroid_core::json_repair::chars as wtf8;
use pyroid_core::sqlparse::grouping::Limits;
use pyroid_core::sqlparse::tree::{Class, Fail, Id, ListId, NO_LIST, Node, Str, Tree};
use pyroid_core::sqlparse::ttype::{self, TT, bits};
use rustc_hash::FxHashMap;

use super::{Lim, limit, not_implemented, py_str, setup, sql_error, tt_of, type_objects};

/// One node as read from Python.
struct Read {
    ttype: Option<TT>,
    value: Str,
    parent: Option<Id>,
}

struct Reader<'py> {
    py: Python<'py>,
    classes: FxHashMap<usize, Class>,
    nodes: Vec<Node>,
    lists: Vec<Vec<Id>>,
    objs: Vec<Bound<'py, PyAny>>,
    list_objs: Vec<Bound<'py, PyList>>,
    by_obj: FxHashMap<usize, Id>,
    by_list: FxHashMap<usize, ListId>,
    src: String,
    remapped: bool,
    initial: Vec<Read>,
    /// The classes have an `is_newline` slot (else it is read off the token type).
    newline_slot: bool,
    types: ttype::Types,
}

/// The text of a Python str value (`None`: not an exact str, or a surrogate case pyroid
/// leaves to Python).
fn str_value(v: &Bound<'_, PyAny>) -> PyResult<Option<(String, bool)>> {
    let py = v.py();
    let Ok(s) = v.cast::<PyString>() else {
        return Ok(None);
    };
    if !s.get_type().is(py.get_type::<PyString>()) {
        return Ok(None);
    }
    if let Ok(t) = s.to_str() {
        return Ok(Some((t.to_owned(), false)));
    }
    let bytes = py
        .get_type::<PyString>()
        .call_method1("encode", (s, "utf-8", "surrogatepass"))?
        .cast_into::<pyo3::types::PyBytes>()?;
    match wtf8::chars_from_wtf8(bytes.as_bytes()) {
        Some(Ok(chars)) => Ok(Some((chars.into_iter().collect(), true))),
        _ => Ok(None),
    }
}

fn exact_bool(v: &Bound<'_, PyAny>) -> Option<bool> {
    v.cast::<PyBool>()
        .ok()
        .map(pyo3::types::PyBoolMethods::is_true)
}

impl<'py> Reader<'py> {
    fn text(&mut self, v: &Bound<'py, PyAny>, leaf: bool) -> PyResult<Option<Str>> {
        let Some((s, rem)) = str_value(v)? else {
            return Ok(None);
        };
        if rem {
            self.remapped = true;
        } else if wtf8::contains_reserved(&s) {
            // a char standing for a surrogate elsewhere would be ambiguous
            if self.remapped {
                return Ok(None);
            }
        }
        if leaf {
            let a = u32::try_from(self.src.len()).unwrap_or(u32::MAX);
            self.src.push_str(&s);
            let b = u32::try_from(self.src.len()).unwrap_or(u32::MAX);
            Ok(Some(Str::Span(a, b)))
        } else {
            Ok(Some(Str::owned(s)))
        }
    }

    /// Read `root` and everything below it (`None`: not a plain sqlparse tree).
    #[allow(clippy::too_many_lines)]
    fn read(&mut self, root: &Bound<'py, PyAny>) -> PyResult<Option<Id>> {
        let py = self.py;
        let mut todo: Vec<Bound<'py, PyAny>> = vec![root.clone()];
        let mut pending_parent: Vec<(Id, Bound<'py, PyAny>)> = Vec::new();
        let first = self.nodes.len();
        while let Some(obj) = todo.pop() {
            let ptr = obj.as_ptr() as usize;
            if self.by_obj.contains_key(&ptr) {
                continue;
            }
            let Some(&class) = self.classes.get(&(obj.get_type().as_ptr() as usize)) else {
                return Ok(None);
            };
            let id = Id::try_from(self.nodes.len()).unwrap_or(Id::MAX);
            self.by_obj.insert(ptr, id);
            let ttype_obj = obj.getattr(intern!(py, "ttype"))?;
            let ttype = if ttype_obj.is_none() {
                None
            } else {
                match tt_of(&ttype_obj)? {
                    Some(t) => Some(t),
                    None => return Ok(None),
                }
            };
            let flags = [
                obj.getattr(intern!(py, "is_keyword"))?,
                obj.getattr(intern!(py, "is_group"))?,
                obj.getattr(intern!(py, "is_whitespace"))?,
            ];
            let mut fl = [false; 4];
            for (slot, f) in fl.iter_mut().zip(&flags) {
                let Some(b) = exact_bool(f) else {
                    return Ok(None);
                };
                *slot = b;
            }
            fl[3] = if self.newline_slot {
                let Some(b) = exact_bool(&obj.getattr(intern!(py, "is_newline"))?) else {
                    return Ok(None);
                };
                b
            } else {
                // sqlparse 0.5.0's tokens have no is_newline: 0.5.1's `ttype in T.Newline`
                ttype.is_some_and(|t| self.types.has(t, bits::NEWLINE))
            };
            let is_list_class = class != Class::Token;
            let Some(value) = self.text(&obj.getattr(intern!(py, "value"))?, !is_list_class)?
            else {
                return Ok(None);
            };
            let norm_obj = obj.getattr(intern!(py, "normalized"))?;
            let value_obj = obj.getattr(intern!(py, "value"))?;
            let normalized = if norm_obj.is(&value_obj) {
                value.clone()
            } else {
                match self.text(&norm_obj, false)? {
                    Some(n) => n,
                    None => return Ok(None),
                }
            };
            let parent_obj = obj.getattr(intern!(py, "parent"))?;
            let mut list = NO_LIST;
            if is_list_class {
                let tokens = obj.getattr(intern!(py, "tokens"))?;
                let Ok(tl) = tokens.cast::<PyList>() else {
                    return Ok(None);
                };
                if !tl.get_type().is(py.get_type::<PyList>()) {
                    return Ok(None);
                }
                let lptr = tl.as_ptr() as usize;
                list = if let Some(&l) = self.by_list.get(&lptr) {
                    l
                } else {
                    let l = ListId::try_from(self.lists.len()).unwrap_or(ListId::MAX);
                    self.by_list.insert(lptr, l);
                    self.lists.push(Vec::new());
                    self.list_objs.push(tl.clone());
                    // children are filled in once every object has an id
                    for item in tl.to_tuple().iter().rev() {
                        todo.push(item);
                    }
                    l
                };
            }
            self.nodes.push(Node {
                class,
                ttype,
                value: value.clone(),
                normalized,
                is_keyword: fl[0],
                is_group: fl[1],
                is_whitespace: fl[2],
                is_newline: fl[3],
                parent: None,
                list,
                origin: id,
            });
            self.objs.push(obj.clone());
            self.initial.push(Read {
                ttype,
                value,
                parent: None,
            });
            pending_parent.push((id, parent_obj));
        }
        // the lists' items, from one snapshot each
        for (l, lo) in self.list_objs.iter().enumerate() {
            let items: Vec<Id> = lo
                .to_tuple()
                .iter()
                .filter_map(|x| self.by_obj.get(&(x.as_ptr() as usize)).copied())
                .collect();
            if items.len() != lo.len() {
                return Ok(None);
            }
            self.lists[l] = items;
        }
        for (id, p) in pending_parent {
            let parent = if p.is_none() {
                None
            } else {
                self.by_obj.get(&(p.as_ptr() as usize)).copied()
            };
            self.nodes[id as usize].parent = parent;
            self.initial[id as usize].parent = parent;
        }
        Ok(self
            .by_obj
            .get(&(root.as_ptr() as usize))
            .copied()
            .filter(|&r| r as usize >= first))
    }
}

fn same(a: &Str, b: &Str, t: &Tree<'_>) -> bool {
    match (a, b) {
        (Str::Span(x, y), Str::Span(p, q)) => x == p && y == q,
        _ => t.s(a) == t.s(b),
    }
}

/// `group(stmt)` with the limits: the statement (grouped in place), or `NotImplemented`.
#[pyfunction]
#[pyo3(signature = (stmt, max_depth, max_tokens))]
pub fn _group<'py>(
    stmt: &Bound<'py, PyAny>,
    max_depth: Option<&Bound<'py, PyAny>>,
    max_tokens: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = stmt.py();
    let (Lim::Is(depth), Lim::Is(tokens)) = (limit(max_depth), limit(max_tokens)) else {
        return Ok(not_implemented(py));
    };
    let s = setup()?;
    let mut classes = FxHashMap::default();
    for (cls, class) in s.classes.iter().zip(Class::ALL) {
        classes.insert(cls.as_ptr() as usize, class);
    }
    let mut r = Reader {
        py,
        classes,
        nodes: Vec::new(),
        lists: Vec::new(),
        objs: Vec::new(),
        list_objs: Vec::new(),
        by_obj: FxHashMap::default(),
        by_list: FxHashMap::default(),
        src: String::new(),
        remapped: false,
        initial: Vec::new(),
        newline_slot: s.newline,
        types: ttype::types(),
    };
    let Some(root) = r.read(stmt)? else {
        return Ok(not_implemented(py));
    };
    if r.nodes[root as usize].list == NO_LIST {
        return Ok(not_implemented(py));
    }
    let Reader {
        nodes,
        lists,
        objs,
        list_objs,
        src,
        remapped,
        initial,
        ..
    } = r;
    let n_orig = nodes.len();
    let n_lists = lists.len();
    let initial_lists = lists.clone();
    let mut tree = Tree::new(&src, ttype::types(), s.version);
    tree.nodes = nodes;
    tree.lists = lists;
    let result =
        pyroid_core::sqlparse::grouping::group_statement(&mut tree, root, Limits { depth, tokens });
    if result == Err(Fail::Python) {
        return Ok(not_implemented(py));
    }
    write_back(
        py,
        &tree,
        root,
        &objs,
        &list_objs,
        &initial,
        &initial_lists,
        n_orig,
        n_lists,
        remapped,
    )?;
    match result {
        Ok(()) => Ok(stmt.clone()),
        Err(Fail::SqlParse(m)) => Err(sql_error(py, m)?),
        Err(Fail::Python) => Ok(not_implemented(py)),
    }
}

/// Set the arena's state on the Python objects: new groups become new objects, the original
/// objects get the fields grouping changed, the lists their new items.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn write_back<'py>(
    py: Python<'py>,
    t: &Tree<'_>,
    root: Id,
    objs: &[Bound<'py, PyAny>],
    list_objs: &[Bound<'py, PyList>],
    initial: &[Read],
    initial_lists: &[Vec<Id>],
    n_orig: usize,
    n_lists: usize,
    remapped: bool,
) -> PyResult<()> {
    let s = setup()?;
    let types = type_objects(py);
    let new = s.new.bind(py);
    let tt_obj = |tt: Option<TT>| -> PyResult<Bound<'py, PyAny>> {
        Ok(match tt {
            None => py.None().into_bound(py),
            Some(x) => types
                .get(usize::from(x))
                .and_then(Option::as_ref)
                .map(|o| o.bind(py).clone())
                .ok_or_else(|| pyo3::exceptions::PyTypeError::new_err("unregistered token type"))?,
        })
    };
    // every node reachable from the root
    let mut reach: Vec<Id> = Vec::new();
    let mut seen = vec![false; t.nodes.len()];
    let mut seen_list = vec![false; t.lists.len()];
    let mut stack = vec![root];
    while let Some(id) = stack.pop() {
        if std::mem::replace(&mut seen[id as usize], true) {
            continue;
        }
        reach.push(id);
        let l = t.n(id).list;
        if l != NO_LIST && !std::mem::replace(&mut seen_list[l as usize], true) {
            stack.extend(t.lists[l as usize].iter().copied());
        }
    }
    let mut out: Vec<Option<Bound<'py, PyAny>>> = vec![None; t.nodes.len()];
    let true_ = PyBool::new(py, true).to_owned().into_any();
    let false_ = PyBool::new(py, false).to_owned().into_any();
    let b = |x: bool| if x { true_.clone() } else { false_.clone() };
    for &id in &reach {
        let n = t.n(id);
        let i = id as usize;
        if i < n_orig {
            let obj = objs[i].clone();
            if n.ttype != initial[i].ttype {
                obj.setattr(intern!(py, "ttype"), tt_obj(n.ttype)?)?;
            }
            if !same(&n.value, &initial[i].value, t) {
                obj.setattr(intern!(py, "value"), py_str(py, t.val(id), remapped)?)?;
            }
            out[i] = Some(obj);
        } else {
            let obj = new.call1((s.classes[n.class.index()].bind(py),))?;
            let value = py_str(py, t.val(id), remapped)?;
            let normalized = if same(&n.value, &n.normalized, t) {
                value.clone()
            } else {
                py_str(py, t.norm(id), remapped)?
            };
            obj.setattr(intern!(py, "value"), &value)?;
            obj.setattr(intern!(py, "normalized"), &normalized)?;
            obj.setattr(intern!(py, "ttype"), tt_obj(n.ttype)?)?;
            obj.setattr(intern!(py, "is_keyword"), b(n.is_keyword))?;
            obj.setattr(intern!(py, "is_group"), b(n.is_group))?;
            obj.setattr(intern!(py, "is_whitespace"), b(n.is_whitespace))?;
            if s.newline {
                obj.setattr(intern!(py, "is_newline"), b(n.is_newline))?;
            }
            out[i] = Some(obj);
        }
    }
    let item = |k: Id| {
        out[k as usize]
            .clone()
            .unwrap_or_else(|| py.None().into_bound(py))
    };
    let mut new_lists: Vec<Option<Bound<'py, PyList>>> = vec![None; t.lists.len()];
    for (l, items) in t.lists.iter().enumerate() {
        if !seen_list[l] {
            continue;
        }
        if l < n_lists {
            if *items != initial_lists[l] {
                let lo = &list_objs[l];
                let vals: Vec<Bound<'py, PyAny>> = items.iter().map(|&k| item(k)).collect();
                lo.as_any()
                    .set_item(pyo3::types::PySlice::full(py), PyList::new(py, vals)?)?;
            }
        } else {
            let vals: Vec<Bound<'py, PyAny>> = items.iter().map(|&k| item(k)).collect();
            new_lists[l] = Some(PyList::new(py, vals)?);
        }
    }
    for &id in &reach {
        let i = id as usize;
        let n = t.n(id);
        let Some(obj) = &out[i] else { continue };
        if i >= n_orig {
            if n.list != NO_LIST {
                let l = n.list as usize;
                let lo: Bound<'py, PyAny> = if l < n_lists {
                    list_objs[l].clone().into_any()
                } else {
                    new_lists[l]
                        .clone()
                        .map_or_else(|| py.None().into_bound(py), Bound::into_any)
                };
                obj.setattr(intern!(py, "tokens"), lo)?;
            }
            let parent = n.parent.map_or_else(|| py.None().into_bound(py), item);
            obj.setattr(intern!(py, "parent"), parent)?;
        } else if let Some(p) = n.parent {
            if initial[i].parent != Some(p) {
                obj.setattr(intern!(py, "parent"), item(p))?;
            }
        }
    }
    Ok(())
}
