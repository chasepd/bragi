from __future__ import annotations

import pytest

from bragi.scene_facts import scene_fact_conflict_key, validate_scene_fact_shape


@pytest.mark.parametrize("reference_type", ["object", "environment"])
def test_scene_local_subject_rejects_provider_generated_id(
    reference_type: str,
) -> None:
    with pytest.raises(ValueError, match="cannot use IDs"):
        validate_scene_fact_shape(
            fact_type="environment_state",
            subject_type=reference_type,
            subject_id="provider-generated-id",
            subject_label="Brass Key",
            aspect="position",
            value="on the table",
        )


@pytest.mark.parametrize("reference_type", ["object", "environment"])
def test_scene_local_target_rejects_provider_generated_id(
    reference_type: str,
) -> None:
    with pytest.raises(ValueError, match="cannot use IDs"):
        validate_scene_fact_shape(
            fact_type="line_of_sight",
            subject_type="character",
            subject_id="character-mara",
            subject_label="Mara",
            target_type=reference_type,
            target_id="provider-generated-id",
            target_label="Brass Key",
            value="visible",
        )


def test_scene_local_conflict_key_ignores_non_durable_reference_ids() -> None:
    with_id = scene_fact_conflict_key(
        fact_type="object_location",
        subject_type="object",
        subject_id="provider-generated-id",
        subject_label="Brass Key",
    )
    normalized_label = scene_fact_conflict_key(
        fact_type="object_location",
        subject_type="object",
        subject_id=None,
        subject_label="brass   key",
    )

    assert with_id == normalized_label
