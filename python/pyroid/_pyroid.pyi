# Type stubs for the pyroid._pyroid extension module (crates/pyroid-py).
# Hand-written: keep in sync with crates/pyroid-py/src/lib.rs.
from typing import Final

__version__: Final[str]

# Base class for all pyroid errors.
class PyroidError(Exception): ...

# Length of `text` in UTF-8 bytes. Placeholder that proves the pipeline.
def utf8_len(text: str, /) -> int: ...
