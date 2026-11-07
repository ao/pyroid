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
    """A 32-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 32-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 32 bits.
    """
    return struct.unpack("<I", hashlib.sha1(data).digest()[:4])[0]  # type: ignore[no-any-return]


def sha1_hash64(data: Any) -> int:
    """A 64-bit hash function based on SHA1.

    Args:
        data (bytes): the data to generate 64-bit integer hash from.

    Returns:
        int: an integer hash value that can be encoded using 64 bits.
    """
    return struct.unpack("<Q", hashlib.sha1(data).digest()[:8])[0]  # type: ignore[no-any-return]
