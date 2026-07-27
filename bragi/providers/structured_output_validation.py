"""Local validation for provider-enforced structured output."""

from __future__ import annotations

from typing import Any

from jsonschema import Draft202012Validator


class StructuredOutputValidationError(ValueError):
    """A structured response did not satisfy its requested JSON Schema."""

    def __init__(self, diagnostics: dict[str, object]) -> None:
        self.diagnostics = diagnostics
        super().__init__("Structured provider response violated its JSON Schema")


def validate_structured_output(
    data: Any,
    *,
    schema: dict[str, Any],
    schema_name: str,
) -> None:
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(data),
        key=lambda error: (
            tuple(str(part) for part in error.absolute_path),
            tuple(str(part) for part in error.absolute_schema_path),
        ),
    )
    if not errors:
        return
    diagnostics: dict[str, object] = {
        "schema_name": schema_name,
        "error_count": len(errors),
        "errors": [
            {
                "schema_path": _json_path(tuple(error.absolute_schema_path)),
                "validator": str(error.validator),
                "message": "Value does not satisfy schema constraint",
            }
            for error in errors[:20]
        ],
    }
    if len(errors) > 20:
        diagnostics["errors_truncated"] = True
    raise StructuredOutputValidationError(diagnostics)


def _json_path(parts: tuple[object, ...]) -> str:
    path = "$"
    for part in parts:
        if isinstance(part, int):
            path += f"[{part}]"
        else:
            path += f".{part}"
    return path
