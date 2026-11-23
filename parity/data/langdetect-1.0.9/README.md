# langdetect 1.0.9 test suite

Copied unchanged from the langdetect 1.0.9 sdist on PyPI (`langdetect-1.0.9.tar.gz`, sha256
cbc1fef89f8d062739774bd51eda3da3274006b3661d199c2655f6b3f6d605a0;
https://github.com/Mimino666/langdetect, Apache-2.0, see `LICENSE` and `NOTICE`): `tests/` holds
its tests of the detector, the factory and `Language`.

`python/tests/test_langdetect_patch.py` runs them with `pyroid.patch("langdetect")` on, both
with langdetect installed (its classes then run pyroid's code) and without it (the tests then
import pyroid's classes as langdetect's). `SHA256SUMS` pins every file.
