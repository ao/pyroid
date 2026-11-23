# packaging 26.3 test suite

Copied unchanged from the packaging 26.3 sdist on PyPI (`packaging-26.3.tar.gz`, sha256
94edc256424af38762eb31306eed28beb9f0efc50a8837492c9d6fd6004aed79;
https://github.com/pypa/packaging, Apache-2.0 OR BSD-2-Clause, see `LICENSE`, `LICENSE.APACHE` and
`LICENSE.BSD`): `tests/` holds its tests of versions, specifiers, markers, requirements, utils,
ranges, metadata, pylock and dependency groups, the property-based tests, and the data files they
read. The tests of platform tags (`test_tags`, `test_manylinux`, `test_musllinux`, `test_elffile`),
licenses, direct URLs and errors are not copied: pyroid does not port those modules.

`python/tests/test_packaging_patch.py::test_packagings_own_test_suite` runs them with
`pyroid.patch("packaging")` on (through the `python/tests/_packaging_upstream.py` plugin), so the
tests import pyroid's classes as packaging's. `SHA256SUMS` pins every file.
