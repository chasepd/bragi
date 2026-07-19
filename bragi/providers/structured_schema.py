"""Strict JSON Schema helpers for provider structured-output requests."""


_COMBINATOR_KEYS = ("anyOf", "oneOf", "allOf")
_JsonSchema = dict[str, object]


def normalize_strict_json_schema(schema: _JsonSchema) -> _JsonSchema:
    """Return an OpenAI/Venice-style strict object schema."""
    normalized = _normalize_schema(
        _deepcopy_schema_value(schema),
        is_root=True,
        nullable=False,
    )
    if not isinstance(normalized, dict):
        raise ValueError("Structured output schema must be an object")
    validate_strict_json_schema(normalized)
    return normalized


def validate_strict_json_schema(schema: _JsonSchema) -> None:
    _validate_schema(schema, path="$")


def _normalize_schema(
    schema: object,
    *,
    is_root: bool,
    nullable: bool,
) -> object:
    if not isinstance(schema, dict):
        return schema

    normalized: _JsonSchema = {
        key: _deepcopy_schema_value(value)
        for key, value in schema.items()
        if key
        not in {
            "properties",
            "required",
            "additionalProperties",
            "items",
            *_COMBINATOR_KEYS,
        }
    }
    schema_type = normalized.get("type")
    allows_object = _allows_type(schema_type, "object") or "properties" in schema
    properties = schema.get("properties")
    has_combinator = _normalize_combinators(
        schema,
        normalized,
        nullable=nullable,
    )

    if allows_object and (not isinstance(properties, dict) or not properties):
        if is_root:
            normalized["type"] = _type_with(schema_type, "object", nullable=nullable)
            normalized["additionalProperties"] = False
            normalized["properties"] = {}
            normalized["required"] = []
            return normalized
        normalized["type"] = _replace_type(
            schema_type,
            old="object",
            new="string",
            nullable=nullable,
        )
        description = str(normalized.get("description", "")).strip()
        suffix = "Use plain text; do not emit a free-form object."
        normalized["description"] = f"{description} {suffix}".strip()
        return normalized

    if allows_object:
        if not isinstance(properties, dict):
            raise ValueError("Object schema properties must be a mapping")
        original_required = {
            item for item in schema.get("required", []) if isinstance(item, str)
        }
        normalized_properties: dict[str, object] = {}
        for name, property_schema in properties.items():
            if not isinstance(name, str):
                continue
            normalized_properties[name] = _normalize_schema(
                property_schema,
                is_root=False,
                nullable=name not in original_required,
            )
        normalized["type"] = _type_with(schema_type, "object", nullable=nullable)
        normalized["additionalProperties"] = False
        normalized["properties"] = normalized_properties
        normalized["required"] = list(normalized_properties)
        if nullable:
            _add_null_enum(normalized)
        return normalized

    if "items" in schema:
        normalized["items"] = _normalize_schema(
            schema["items"],
            is_root=False,
            nullable=False,
        )

    if nullable and (not has_combinator or schema_type is not None):
        normalized["type"] = _type_with(schema_type, None, nullable=True)
        _add_null_enum(normalized)
    return normalized


def _deepcopy_schema_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _deepcopy_schema_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_deepcopy_schema_value(item) for item in value]
    return value


def _normalize_combinators(
    schema: _JsonSchema,
    normalized: _JsonSchema,
    *,
    nullable: bool,
) -> bool:
    has_combinator = False
    for key in _COMBINATOR_KEYS:
        branches = schema.get(key)
        if branches is None:
            continue
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"{key} must be a non-empty schema list")
        branch_nullable = nullable and key == "allOf"
        normalized_branches = [
            _normalize_schema(
                branch,
                is_root=False,
                nullable=branch_nullable,
            )
            for branch in branches
        ]
        if (
            nullable
            and key in {"anyOf", "oneOf"}
            and not any(_schema_allows_null(branch) for branch in normalized_branches)
        ):
            normalized_branches.append({"type": "null"})
        normalized[key] = normalized_branches
        has_combinator = True
    return has_combinator


def _validate_schema(schema: object, *, path: str) -> None:
    if not isinstance(schema, dict):
        raise ValueError(f"{path}: schema node must be an object")
    schema_type = schema.get("type")
    for key in _COMBINATOR_KEYS:
        branches = schema.get(key)
        if branches is None:
            continue
        if not isinstance(branches, list) or not branches:
            raise ValueError(f"{path}.{key}: must be a non-empty schema list")
        for index, branch in enumerate(branches):
            _validate_schema(branch, path=f"{path}.{key}[{index}]")
    if _allows_type(schema_type, "object") or "properties" in schema:
        if schema.get("additionalProperties") is not False:
            raise ValueError(
                f"{path}: object schema must set additionalProperties=false"
            )
        properties = schema.get("properties")
        if not isinstance(properties, dict):
            raise ValueError(f"{path}: object schema must define properties")
        required = schema.get("required")
        if not isinstance(required, list):
            raise ValueError(f"{path}: object schema must define required")
        if set(required) != set(properties):
            raise ValueError(f"{path}: every object property must be required")
        for name, property_schema in properties.items():
            _validate_schema(property_schema, path=f"{path}.properties.{name}")
    if _allows_type(schema_type, "array"):
        items = schema.get("items")
        if items is not None:
            _validate_schema(items, path=f"{path}.items")


def _allows_type(schema_type: object, expected: str) -> bool:
    if schema_type == expected:
        return True
    if isinstance(schema_type, list):
        return expected in schema_type
    return False


def _schema_allows_null(schema: object) -> bool:
    if not isinstance(schema, dict):
        return False
    if _allows_type(schema.get("type"), "null"):
        return True
    enum = schema.get("enum")
    if isinstance(enum, list) and None in enum:
        return True
    for key in ("anyOf", "oneOf"):
        branches = schema.get(key)
        if isinstance(branches, list) and any(
            _schema_allows_null(branch) for branch in branches
        ):
            return True
    branches = schema.get("allOf")
    if isinstance(branches, list) and branches:
        return all(_schema_allows_null(branch) for branch in branches)
    return False


def _type_with(
    schema_type: object,
    preferred: str | None,
    *,
    nullable: bool,
) -> str | list[str]:
    types: list[str] = []
    if isinstance(schema_type, str):
        types = [schema_type]
    elif isinstance(schema_type, list):
        types = [item for item in schema_type if isinstance(item, str)]
    if preferred is not None and preferred not in types:
        types.insert(0, preferred)
    if not types and preferred is not None:
        types = [preferred]
    if nullable and "null" not in types:
        types.append("null")
    if len(types) == 1:
        return types[0]
    return types


def _replace_type(
    schema_type: object,
    *,
    old: str,
    new: str,
    nullable: bool,
) -> str | list[str]:
    types: list[str]
    if isinstance(schema_type, str):
        types = [schema_type]
    elif isinstance(schema_type, list):
        types = [item for item in schema_type if isinstance(item, str)]
    else:
        types = [old]
    replaced = [new if item == old else item for item in types]
    if new not in replaced:
        replaced.append(new)
    seen: list[str] = []
    for item in replaced:
        if item not in seen:
            seen.append(item)
    if nullable and "null" not in seen:
        seen.append("null")
    if len(seen) == 1:
        return seen[0]
    return seen


def _add_null_enum(schema: _JsonSchema) -> None:
    enum = schema.get("enum")
    if isinstance(enum, list) and None not in enum:
        schema["enum"] = [*enum, None]
