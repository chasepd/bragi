from __future__ import annotations

from typing import cast

from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.manual_confirmation import (
    manual_character_registry_confirmation_enabled,
    manual_memory_confirmation_enabled,
    manual_state_change_confirmation_enabled,
)


class FakeRepositories:
    def __init__(self, settings: dict[str, object]) -> None:
        self.settings = settings

    def get_app_setting(self, key: str) -> object:
        return self.settings.get(key)

    def get_effective_setting(
        self,
        key: str,
        *,
        save_id: str | None = None,
        user_id: str | None = None,
        scenario_id: str | None = None,
    ) -> object:
        return self.settings.get(key)


def test_manual_confirmation_helpers_return_boolean_settings_only() -> None:
    repositories = FakeRepositories(
        {
            "manual_confirmation_memories_enabled": True,
            "manual_confirmation_character_registry_enabled": False,
            "manual_confirmation_state_changes_enabled": "true",
        }
    )

    typed_repositories = cast(PersistenceRepositories, repositories)

    assert manual_memory_confirmation_enabled(typed_repositories) is True
    assert manual_character_registry_confirmation_enabled(typed_repositories) is False
    assert manual_state_change_confirmation_enabled(typed_repositories) is False
