"""Copy the stubs' docstrings into the Python modules, so ``help()`` shows them at run time.

The API docs are written in the ``.pyi`` stubs (``python/pyroid/**/*.pyi``). Every public
function, class and method of a ``.py`` module that its stub documents gets the stub's
docstring, as written there (for an ``@overload``ed name, the first documented overload's).
Edit the stub, then run this script.

    python docs/scripts/sync_docstrings.py           # rewrite the .py files
    python docs/scripts/sync_docstrings.py --check   # exit 1 if any would change

Native functions (in ``crates/pyroid-py``) take their docstrings from the Rust doc comments.
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "python" / "pyroid"


def _defs(body: list[ast.stmt]) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef]:
    out: dict[str, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef] = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # The first definition with a docstring wins (the documented @overload).
            seen = out.get(node.name)
            if seen is None or (
                ast.get_docstring(seen, clean=False) is None
                and ast.get_docstring(node, clean=False) is not None
            ):
                out[node.name] = node
    return out


def _doc_node(node: ast.AST) -> ast.Constant | None:
    body = getattr(node, "body", None)
    first = body[0] if body else None
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.value
    return None


def _reindent(line: str, old: int, indent: str) -> str:
    if not line.strip():
        return ""
    if line[:old].strip():  # less indented than the docstring: keep it as written
        return line
    return indent + line[old:]


def _edits(
    stub_src: str,
    stub_body: list[ast.stmt],
    py_body: list[ast.stmt],
    out: list[tuple[int, int, int, list[str]]],
) -> None:
    stub = _defs(stub_body)
    for name, node in _defs(py_body).items():
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            continue
        twin = stub.get(name)
        if twin is None:
            continue
        if isinstance(node, ast.ClassDef) and isinstance(twin, ast.ClassDef):
            _edits(stub_src, twin.body, node.body, out)
        src_doc = _doc_node(twin)
        if src_doc is None:
            continue
        text = ast.get_source_segment(stub_src, src_doc)
        if text is None:
            continue
        indent = " " * (node.body[0].col_offset if node.body else node.col_offset + 4)
        lines = text.splitlines()
        stub_indent = src_doc.col_offset
        new = [indent + lines[0]] + [_reindent(ln, stub_indent, indent) for ln in lines[1:]]
        old = _doc_node(node)
        if old is not None:
            out.append((old.lineno - 1, old.end_lineno or old.lineno, 0, new))
        else:
            # No docstring: insert one before the first statement of the body.
            first = node.body[0]
            decorators = getattr(first, "decorator_list", None)
            at = (decorators[0].lineno if decorators else first.lineno) - 1
            out.append((at, at, 1, new))


def sync(py: Path, check: bool) -> bool:
    stub_path = py.with_suffix(".pyi")
    if not stub_path.exists():
        return False
    stub_src = stub_path.read_text(encoding="utf-8")
    src = py.read_text(encoding="utf-8")
    lines = src.splitlines(keepends=False)
    edits: list[tuple[int, int, int, list[str]]] = []
    _edits(stub_src, ast.parse(stub_src).body, ast.parse(src).body, edits)
    for start, end, _insert, new in sorted(edits, key=lambda e: e[0], reverse=True):
        lines[start:end] = new
    text = "\n".join(lines) + ("\n" if src.endswith("\n") else "")
    if text == src:
        return False
    name = py.relative_to(ROOT.parents[1])
    if check:
        sys.stderr.write(f"{name}: docstrings differ from the stub; run {__file__}\n")
    else:
        py.write_text(text, encoding="utf-8")
        sys.stdout.write(f"updated {name}\n")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="only report; exit 1 on a difference")
    args = ap.parse_args()
    changed = [sync(py, args.check) for py in sorted(ROOT.rglob("*.py"))]
    return 1 if args.check and any(changed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
