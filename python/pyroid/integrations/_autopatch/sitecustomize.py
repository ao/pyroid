"""pyroid's start-up hook for ``$PYROID_PATCH`` (not a module to import yourself).

Python imports ``sitecustomize`` at start-up from the first ``sys.path`` entry that has one.
``pyroid run`` puts this directory first on ``PYTHONPATH`` (see
``pyroid.integrations._patch.HOOK_DIR``), so the Python processes it starts are patched too:
when ``PYROID_PATCH`` is set, this applies it with ``pyroid.integrations.patch``. Then it runs
the ``sitecustomize`` it shadows (the next one on ``sys.path``), if there is one, so the
environment's own still runs.

It only patches if this interpreter can import pyroid (a subprocess may run another Python),
and never raises because of pyroid. The code sticks to syntax that any Python 3 can parse.
Imported under any other name than ``sitecustomize`` it does nothing.
"""

import os
import sys


def _activate() -> None:
    if not os.environ.get("PYROID_PATCH", "").strip():
        return
    try:
        from pyroid.integrations import _patch
    except Exception:  # another interpreter, or a pyroid without this module
        return
    _patch._autopatch()


def _chain() -> None:
    from importlib.machinery import PathFinder

    def resolved(entry: str) -> str:
        return os.path.normcase(os.path.realpath(entry or os.curdir))

    here = resolved(os.path.dirname(os.path.abspath(__file__)))
    others = [p for p in sys.path if not (isinstance(p, str) and resolved(p) == here)]
    spec = PathFinder.find_spec("sitecustomize", others)
    if spec is None or spec.loader is None:
        return
    from importlib.util import module_from_spec

    module = module_from_spec(spec)
    # The import of this module returns whatever sys.modules holds once it has run.
    sys.modules["sitecustomize"] = module
    spec.loader.exec_module(module)


# Only as Python's start-up hook: imported as pyroid.integrations._autopatch.sitecustomize
# (a tool walking the package, say) it does nothing.
if __name__ == "sitecustomize":
    try:
        _activate()
    finally:
        _chain()
