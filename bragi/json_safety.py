"""Low-allocation structural limits for untrusted JSON payloads."""

from __future__ import annotations


class JsonSafetyError(ValueError):
    """Raised when JSON structure exceeds configured resource bounds."""


def validate_json_structure(
    payload: bytes,
    *,
    max_nodes: int,
    max_depth: int,
) -> int:
    """Bound non-string JSON separators before materializing Python objects."""
    nodes = 1
    depth = 0
    in_string = False
    escaped = False
    for byte in payload:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:
                escaped = True
            elif byte == 0x22:
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in {0x7B, 0x5B}:
            depth += 1
            nodes += 1
            if depth > max_depth:
                raise JsonSafetyError("JSON nesting is too deep")
        elif byte in {0x7D, 0x5D}:
            depth -= 1
        elif byte == 0x2C:
            nodes += 1
        if nodes > max_nodes:
            raise JsonSafetyError("JSON contains too many values")
    return nodes
