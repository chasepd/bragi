from __future__ import annotations

import json

from bragi.services.scenario_content_rating import (
    metadata_with_scenario_content_ratings,
    scenario_content_rating,
)


def test_scenario_content_rating_reads_portable_source_metadata() -> None:
    content_json = json.dumps(
        {
            "opening_message": "The gate opens.",
            "_source": {"content_rating": "r"},
        }
    )

    assert scenario_content_rating(content_json) == "r"


def test_legacy_or_malformed_scenario_rating_is_unclassified() -> None:
    assert scenario_content_rating("{}") == "unclassified"
    assert scenario_content_rating("{not-json") == "unclassified"


def test_scenario_rating_metadata_preserves_existing_source_data() -> None:
    metadata = metadata_with_scenario_content_ratings(
        {"origin": "ai_draft"},
        aggregate_rating="pg-13",
        section_ratings={"premise": "pg", "opening_message": "pg-13"},
    )

    assert metadata == {
        "origin": "ai_draft",
        "content_rating": "pg-13",
        "section_content_ratings": {
            "premise": "pg",
            "opening_message": "pg-13",
        },
    }
