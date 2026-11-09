"""``python -m pyroid``: the ``pyroid`` command line (see :mod:`pyroid.cli`)."""

import sys

from pyroid.cli import main

# The guard matters: ``--workers`` starts spawned processes, which import this module again
# (as ``__mp_main__``) and must not run the command a second time.
if __name__ == "__main__":
    sys.exit(main())
