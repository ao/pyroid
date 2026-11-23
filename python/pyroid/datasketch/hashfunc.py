# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
"""datasketch's SHA1-based hash functions (``datasketch.hashfunc``), unchanged.

:class:`~pyroid.datasketch.MinHash` recognises these functions (and
datasketch's own) and hashes natively; called directly, they are plain
``hashlib`` code, so they accept and reject exactly what datasketch's do.
"""

import hashlib
import struct
from typing import Any

__all__ = ["sha1_hash32", "sha1_hash64"]


def sha1_hash32(data: Any) -> int:
    """Return the first 4 bytes of ``sha1(data)`` as a little-endian int."""
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]  # type: ignore[no-any-return]


def sha1_hash64(data: Any) -> int:
    """Return the first 8 bytes of ``sha1(data)`` as a little-endian int."""
    return struct.unpack("<Q", hashlib.sha1(data).digest()[:8])[0]  # type: ignore[no-any-return]
