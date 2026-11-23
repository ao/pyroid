"""Run tabulate 0.10.0's own tests (``test/``) against pyroid.tabulate.

``pyroid.integrations.patch("tabulate")`` runs before the test modules import ``tabulate``, so
``from tabulate import tabulate`` (and ``tabulate._CustomTextWrap`` and the other private
helpers they test) are pyroid's. ``PYROID_TABULATE_IMPL=tabulate`` runs them against the
installed tabulate instead.
"""

import os
import warnings

if os.environ.get("PYROID_TABULATE_IMPL", "pyroid") == "pyroid":
    import pyroid.integrations

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pyroid.integrations.patch("tabulate")
    import tabulate

    assert tabulate.__pyroid__, "the tests would run against the installed tabulate"
