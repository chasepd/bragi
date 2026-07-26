"""Content-rating provenance stored inside portable scenario content."""

from __future__ import annotations

import json
from collections.abc import Mapping

from bragi.content_rating_instructions import CONTENT_RATING_UNCLASSIFIED

SCENARIO_SOURCE_CONTENT_KEY = "_source"
SCENARIO_CONTENT_RATING_KEY = "content_rating"
SCENARIO_SECTION_CONTENT_RATINGS_KEY = "section_content_ratings"


def scenario_content_rating(content_json: str) -> str:
    """Return a scenario's stored aggregate rating or the legacy sentinel."""

    try:
        content = json.loads(content_json)
    except (json.JSONDecodeError, TypeError):
        return CONTENT_RATING_UNCLASSIFIED
    if not isinstance(content, Mapping):
        return CONTENT_RATING_UNCLASSIFIED
    source = content.get(SCENARIO_SOURCE_CONTENT_KEY)
    if not isinstance(source, Mapping):
        return CONTENT_RATING_UNCLASSIFIED
    value = source.get(SCENARIO_CONTENT_RATING_KEY)
    return (
        value.strip()
        if isinstance(value, str) and value.strip()
        else CONTENT_RATING_UNCLASSIFIED
    )


def metadata_with_scenario_content_ratings(
    metadata: Mapping[str, object] | None,
    *,
    aggregate_rating: str,
    section_ratings: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Copy draft source metadata and attach portable rating provenance."""

    updated = dict(metadata or {})
    updated[SCENARIO_CONTENT_RATING_KEY] = aggregate_rating
    if section_ratings:
        updated[SCENARIO_SECTION_CONTENT_RATINGS_KEY] = dict(section_ratings)
    else:
        updated.pop(SCENARIO_SECTION_CONTENT_RATINGS_KEY, None)
    return updated
