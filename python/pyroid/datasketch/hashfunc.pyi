# Derived from datasketch 2.0.0 (MIT), © 2015 ekzhu; modified.
# Type stubs for pyroid.datasketch.hashfunc.
"""datasketch's SHA1-based hash functions (``datasketch.hashfunc``), unchanged.

:class:`pyroid.datasketch.MinHash` recognises these two (and datasketch's own)
and hashes natively. Called directly they are plain ``hashlib`` code, so they
take and reject what datasketch's take and reject (``bytes``-like data only).
"""

from typing_extensions import Buffer

__all__ = ["sha1_hash32", "sha1_hash64"]

def sha1_hash32(data: Buffer) -> int:
    """Return the first 4 bytes of ``sha1(data)`` as a little-endian int."""

def sha1_hash64(data: Buffer) -> int:
    """Return the first 8 bytes of ``sha1(data)`` as a little-endian int."""
