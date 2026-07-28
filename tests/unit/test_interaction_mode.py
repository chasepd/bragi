from __future__ import annotations

import pytest

from bragi.interaction_mode import InteractionMode, normalize_interaction_mode


def test_interaction_mode_exposes_public_values_and_legacy_default() -> None:
    assert InteractionMode.ROLEPLAY.value == "roleplay"
    assert InteractionMode.STORYTELLER.value == "storyteller"
    assert normalize_interaction_mode(None) is InteractionMode.ROLEPLAY


@pytest.mark.parametrize("value", ["", "cinematic"])
def test_interaction_mode_rejects_unknown_values(value: str) -> None:
    with pytest.raises(ValueError, match="Unknown interaction mode"):
        normalize_interaction_mode(value)
