"""Stand-in for tabulate's ``tabulate/__init__.py``, which ``test/test_cli.py`` runs as a script.

Run the tests from the ``cli`` directory: the command line is pyroid's
(``python -m pyroid.tabulate``), or the installed tabulate's with
``PYROID_TABULATE_IMPL=tabulate``.
"""

import os

if os.environ.get("PYROID_TABULATE_IMPL", "pyroid") == "pyroid":
    from pyroid.tabulate import _main
else:
    from tabulate import _main

_main()
