"""Griffe extension for pyroid's API reference (loaded from mkdocs.yml).

pyroid's public modules are thin ``.py`` re-exports of the native extension
(``fix_text = _native.fix_text``), and their types and docs live in the
hand-written ``.pyi`` stubs next to them. Griffe would document the ``.py``
files (it keeps the runtime module when both exist, and skips stub members
whose kind differs, e.g. an attribute at runtime but a function in the stub),
so this extension:

1. replaces every ``pyroid.<mod>`` (and ``pyroid.<pkg>.<mod>``, e.g. ``pyroid.metrics.sacrebleu``)
   that has a ``<mod>.pyi`` with a module visited from the stub alone (statically: nothing is
   imported);
2. gives each stub function that only has ``@overload``s (stubs have no
   implementation) a documented entry: the docstring of its first overload,
   with every overload attached (mkdocs.yml shows only the overloads);
3. turns the Sphinx roles the stubs use (``:func:`x```, ``:class:`pyroid.Column```)
   into mkdocstrings cross-references when the target is documented, and into
   code spans otherwise, and drops reST's ``::`` literal-block marker;
4. rewrites a ``Raises:`` section that only says "As for :func:`x`." into a sentence
   (it is not a google-style ``Type: description`` pair).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import griffe

_ROLE = re.compile(r":(?:func|class|meth|attr|data|exc|mod|obj):`~?([A-Za-z_][\w.]*)`")
_LITERAL_MARKER = re.compile(r"::$", re.MULTILINE)
# A "Raises:" section that only says "As for :func:`x`." (not a google-style "Type: text" pair).
_RAISES_AS_FOR = re.compile(r"^Raises:\n\s+As for (.+?)\.?$", re.MULTILINE)


def _promote_overloads(obj: griffe.Module | griffe.Class) -> None:
    for name, overloads in list(obj.overloads.items()):
        if not overloads or name in obj.members:
            continue
        first = next((o for o in overloads if o.docstring), overloads[0])

        # The general signature: each parameter's and the return's annotations are the union of
        # the overloads' (overloads are shown as well, above the docstring).
        def union(annotations: list[Any]) -> str | None:
            names = list(dict.fromkeys(str(a) for a in annotations if a is not None))
            return " | ".join(names) or None

        params = griffe.Parameters(
            *(
                griffe.Parameter(
                    p.name,
                    kind=p.kind,
                    default=p.default,
                    annotation=union(
                        [
                            o.parameters[p.name].annotation
                            for o in overloads
                            if p.name in o.parameters
                        ]
                    ),
                )
                for p in first.parameters
            )
        )
        func = griffe.Function(
            name,
            parameters=params,
            returns=union([o.returns for o in overloads]),
            lineno=overloads[0].lineno,
            endlineno=overloads[-1].endlineno,
            docstring=first.docstring,
        )
        func.overloads = overloads
        if func.docstring is not None:
            func.docstring.parent = func
        obj.set_member(name, func)
        del obj.overloads[name]
    for member in list(obj.members.values()):
        if not member.is_alias and (member.is_class or member.is_module):
            _promote_overloads(member)


def _members(obj: griffe.Object) -> list[griffe.Object]:
    out = []
    for member in obj.members.values():
        if member.is_alias:
            continue
        out.append(member)
        if member.is_class or member.is_module:
            out.extend(_members(member))
    return out


def _swap_stubs(parent: griffe.Module, loader: griffe.GriffeLoader) -> None:
    """Replace each submodule that has a ``.pyi`` with the stub; recurse into subpackages."""
    for name, member in list(parent.members.items()):
        if member.is_alias or not member.is_module:
            continue
        path = member.filepath
        if not isinstance(path, Path):
            continue
        stub = path.with_suffix(".pyi")
        if path.suffix == ".pyi" or not stub.exists():
            # A package without its own stub (pyroid.metrics): its modules may have stubs.
            if path.name == "__init__.py":
                _swap_stubs(member, loader)
            continue
        code = stub.read_text(encoding="utf-8")
        loader.lines_collection[stub] = code.splitlines()
        module = griffe.visit(
            name,
            stub,
            code,
            parent=parent,
            docstring_parser=loader.docstring_parser,
            docstring_options=loader.docstring_options,
            lines_collection=loader.lines_collection,
            modules_collection=loader.modules_collection,
        )
        # set_member() would merge the new module into the runtime one: remove it first.
        old = parent.members.pop(name)
        parent.set_member(name, module)
        for alias in old.aliases.values():
            alias.target = module


class PyroidStubs(griffe.Extension):
    """Document pyroid from its stubs, and resolve the stubs' Sphinx roles."""

    def on_package(self, *, pkg: griffe.Module, loader: griffe.GriffeLoader, **kwargs: Any) -> None:
        """Swap runtime modules for their stubs, then rewrite docstring roles."""
        if pkg.name != "pyroid":
            return
        _swap_stubs(pkg, loader)
        _promote_overloads(pkg)
        for obj in [pkg, *_members(pkg)]:
            if obj.docstring is not None:
                obj.docstring.value = self._rewrite(obj, obj.docstring.value, pkg)

    @staticmethod
    def _resolve(obj: griffe.Object, target: str, pkg: griffe.Module) -> str | None:
        scopes = []
        scope: griffe.Object | None = obj
        while scope is not None:
            scopes.append(scope.path)
            scope = scope.parent
        candidates = [f"{s}.{target}" for s in scopes] + [target]
        for path in candidates:
            if not path.startswith("pyroid"):
                continue
            try:
                member = pkg.modules_collection.get_member(path)
            except (KeyError, ValueError, griffe.AliasResolutionError):
                continue
            # An import of another package (``import runpy`` inside a module) is not documented
            # here: link to what it imports, which the Python inventory resolves.
            if member.is_alias and not member.target_path.startswith("pyroid"):
                return member.target_path
            # Re-exports (pyroid.Column is pyroid._pyroid.Column) link to the public path.
            return path.replace("pyroid._pyroid.", "pyroid.", 1) if path.count(".") == 2 else path
        return None

    def _rewrite(self, obj: griffe.Object, text: str, pkg: griffe.Module) -> str:
        def sub(m: re.Match[str]) -> str:
            target = m.group(1)
            path = self._resolve(obj, target, pkg)
            return f"[`{target}`][{path}]" if path else f"`{target}`"

        text = _RAISES_AS_FOR.sub(r"Raises the same exceptions as \1.", text)
        return _LITERAL_MARKER.sub(":", _ROLE.sub(sub, text))
