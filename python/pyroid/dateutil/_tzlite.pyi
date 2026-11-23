# Private: python-dateutil 4.0.0's tzutc, tzoffset, tzlocal and enfold, kept untyped.
from typing import Any

__all__ = ["UTC", "enfold", "tzlocal", "tzoffset", "tzutc"]

UTC: Any
enfold: Any
tzlocal: Any
tzoffset: Any
tzutc: Any

def __getattr__(name: str) -> Any: ...
