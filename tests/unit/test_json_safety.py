from __future__ import annotations

import pytest

from bragi.json_safety import JsonSafetyError, validate_json_structure


def test_json_structure_counts_primitive_array_values() -> None:
    with pytest.raises(JsonSafetyError, match="too many values"):
        validate_json_structure(
            b"[0,0,0,0]",
            max_nodes=4,
            max_depth=8,
        )


def test_json_structure_ignores_delimiters_inside_strings() -> None:
    validate_json_structure(
        b'{"value":"[[[,,,]]"}',
        max_nodes=4,
        max_depth=2,
    )


def test_json_structure_bounds_nesting_depth() -> None:
    with pytest.raises(JsonSafetyError, match="too deep"):
        validate_json_structure(
            b"[[[0]]]",
            max_nodes=10,
            max_depth=2,
        )
