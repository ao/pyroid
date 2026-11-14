# Derived from PyYAML 6.0.3 (MIT), © Kirill Simonov and Ingy döt Net; modified.
"""PyYAML's ``SafeConstructor`` code for the scalars the native loader does not handle.

The native constructor converts the usual forms itself (decimal ints, floats, booleans,
timestamps); the rest (explicitly tagged values such as ``!!int " 0x1F "``, Unicode
digits, invalid values) come here, so results and exceptions are PyYAML's.
"""

from __future__ import annotations

import base64
import binascii
import datetime
import re
from typing import Any

inf_value = 1e300
while inf_value != inf_value * inf_value:
    inf_value *= inf_value
#: ``SafeConstructor.nan_value``: one float object shared by every ``.nan``, as in PyYAML.
nan_value = -inf_value / inf_value

bool_values = {
    "yes": True,
    "no": False,
    "true": True,
    "false": False,
    "on": True,
    "off": False,
}


def construct_bool(value: str) -> bool:
    return bool_values[value.lower()]


def construct_int(value: str) -> int:
    value = value.replace("_", "")
    sign = +1
    if value[0] == "-":
        sign = -1
    if value[0] in "+-":
        value = value[1:]
    if value == "0":
        return 0
    elif value.startswith("0b"):
        return sign * int(value[2:], 2)
    elif value.startswith("0x"):
        return sign * int(value[2:], 16)
    elif value[0] == "0":
        return sign * int(value, 8)
    elif ":" in value:
        digits = [int(part) for part in value.split(":")]
        digits.reverse()
        base = 1
        result = 0
        for digit in digits:
            result += digit * base
            base *= 60
        return sign * result
    else:
        return sign * int(value)


def construct_float(value: str) -> float:
    value = value.replace("_", "").lower()
    sign = +1
    if value[0] == "-":
        sign = -1
    if value[0] in "+-":
        value = value[1:]
    if value == ".inf":
        return sign * inf_value
    elif value == ".nan":
        return nan_value
    elif ":" in value:
        digits = [float(part) for part in value.split(":")]
        digits.reverse()
        base = 1
        result = 0.0
        for digit in digits:
            result += digit * base
            base *= 60
        return sign * result
    else:
        return sign * float(value)


def construct_binary(value: str) -> bytes | str:
    """The bytes, or the message of the ``ConstructorError`` PyYAML raises."""
    try:
        data = value.encode("ascii")
    except UnicodeEncodeError as exc:
        return "failed to convert base64 data into ascii: %s" % exc
    try:
        return base64.decodebytes(data)
    except binascii.Error as exc:
        return "failed to decode base64 data: %s" % exc


timestamp_regexp = re.compile(
    r"""^(?P<year>[0-9][0-9][0-9][0-9])
                -(?P<month>[0-9][0-9]?)
                -(?P<day>[0-9][0-9]?)
                (?:(?:[Tt]|[ \t]+)
                (?P<hour>[0-9][0-9]?)
                :(?P<minute>[0-9][0-9])
                :(?P<second>[0-9][0-9])
                (?:\.(?P<fraction>[0-9]*))?
                (?:[ \t]*(?P<tz>Z|(?P<tz_sign>[-+])(?P<tz_hour>[0-9][0-9]?)
                (?::(?P<tz_minute>[0-9][0-9]))?))?)?$""",
    re.X,
)


def construct_timestamp(value: str) -> Any:
    match = timestamp_regexp.match(value)
    values = match.groupdict()  # type: ignore[union-attr]  # PyYAML's AttributeError
    year = int(values["year"])
    month = int(values["month"])
    day = int(values["day"])
    if not values["hour"]:
        return datetime.date(year, month, day)
    hour = int(values["hour"])
    minute = int(values["minute"])
    second = int(values["second"])
    fraction: Any = 0
    tzinfo = None
    if values["fraction"]:
        fraction = values["fraction"][:6]
        while len(fraction) < 6:
            fraction += "0"
        fraction = int(fraction)
    if values["tz_sign"]:
        tz_hour = int(values["tz_hour"])
        tz_minute = int(values["tz_minute"] or 0)
        delta = datetime.timedelta(hours=tz_hour, minutes=tz_minute)
        if values["tz_sign"] == "-":
            delta = -delta
        tzinfo = datetime.timezone(delta)
    elif values["tz"]:
        tzinfo = datetime.timezone.utc
    return datetime.datetime(year, month, day, hour, minute, second, fraction, tzinfo=tzinfo)


HELPERS = {
    "int": construct_int,
    "float": construct_float,
    "bool": construct_bool,
    "binary": construct_binary,
    "timestamp": construct_timestamp,
}
