from __future__ import annotations

from bragi.services.character_locks import reconcile_character_presence_locks


def test_reconcile_character_presence_locks_preserves_locked_membership() -> None:
    assert reconcile_character_presence_locks(
        current_present_ids={"locked-present", "unlocked-leaving"},
        proposed_present_ids={"locked-absent", "unlocked-entering"},
        locked_character_ids={"locked-present", "locked-absent"},
    ) == {"locked-present", "unlocked-entering"}
