# Private: the python-dateutil objects pyroid's parser returns and raises.
from typing import Any

__all__ = ["INSTALLED_VERSION", "ParserError", "UnknownTimezoneWarning", "relativedelta", "tz"]

INSTALLED_VERSION: str | None
ParserError: type[ValueError]
UnknownTimezoneWarning: type[Warning]
relativedelta: Any
tz: Any

def __getattr__(name: str) -> Any: ...
