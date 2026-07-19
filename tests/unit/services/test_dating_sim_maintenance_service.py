from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

from bragi.persistence.migrations import migrate_database
from bragi.persistence.repositories import PersistenceRepositories
from bragi.services.dating_sim_maintenance_service import (
    DatingSimMaintenanceService,
)


@pytest.fixture
def repositories(tmp_path: Path) -> Iterator[PersistenceRepositories]:
    database_path = tmp_path / "bragi.sqlite3"
    migrate_database(database_path)

    with sqlite3.connect(database_path) as connection:
        yield PersistenceRepositories(connection)


def test_dry_run_reports_route_backfill_without_mutating_or_quoting(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_save_with_romance_option(repositories)
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika Arai meets Ren outside class and they exchange numbers.",
    )
    service = DatingSimMaintenanceService(repositories)

    report = service.inspect_save(save_id)

    assert report.applied_count == 0
    assert repositories.list_dating_route_states(save_id) == []
    assert [repair.npc_character_id for repair in report.reviewable_repairs] == [
        npc_id
    ]
    repair = report.reviewable_repairs[0]
    assert repair.stage == "contact_exchanged"
    assert repair.first_met_message_id is not None
    payload = report.to_result()
    reviewable = cast(list[dict[str, object]], payload["reviewable_repairs"])
    assert "Mika Arai meets Ren" not in repr(payload)
    assert "source_message_ids" in reviewable[0]
    assert "evidence_text" not in reviewable[0]


def test_include_evidence_text_adds_local_redacted_snippets(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _dating_save_with_romance_option(repositories)
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika Arai gives Ren her number. token=secret-value",
    )

    report = DatingSimMaintenanceService(repositories).inspect_save(
        save_id,
        include_evidence_text=True,
    )

    payload = report.to_result()
    reviewable = cast(list[dict[str, object]], payload["reviewable_repairs"])
    evidence = cast(str, reviewable[0]["evidence_text"])
    assert "Mika Arai gives Ren her number" in evidence
    assert "token=[redacted]" in evidence


def test_dry_run_reports_countdown_scene_time_cleanup(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_save_with_romance_option(repositories)
    message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika watches the countdown timer show 03:45 remaining.",
    )
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save_id,
        in_world_time="Monday 03:45",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=0,
        present_character_ids=[npc_id],
        source_message_id=message.id,
    )

    report = DatingSimMaintenanceService(repositories).inspect_save(save_id)

    assert [repair.repair_id for repair in report.deterministic_repairs] == [
        f"scene-time-cleanup:{snapshot.id}"
    ]
    assert report.deterministic_repairs[0].field_path == "in_world_time"
    assert report.deterministic_repairs[0].proposed_value == "Monday morning"
    saved_snapshot = repositories.get_scene_snapshot(save_id)
    assert saved_snapshot is not None
    assert saved_snapshot.in_world_time == "Monday 03:45"


def test_apply_countdown_scene_time_cleanup_syncs_canonical_time(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, npc_id = _dating_save_with_romance_option(repositories)
    message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika watches the countdown timer show 03:45 remaining.",
    )
    snapshot = repositories.upsert_scene_snapshot(
        save_id=save_id,
        in_world_time="Monday 03:45",
        time_of_day="morning",
        day_of_week="monday",
        world_day_index=0,
        present_character_ids=[npc_id],
        source_message_id=message.id,
    )
    repositories.connection.execute(
        """
        UPDATE scene_snapshots
        SET world_time_period_label = 'festival week'
        WHERE id = ?
        """,
        (snapshot.id,),
    )
    repositories.commit()
    service = DatingSimMaintenanceService(repositories)

    applied = service.apply_repairs(
        save_id,
        repair_ids=[f"scene-time-cleanup:{snapshot.id}"],
        confirm_save_id=save_id,
    )

    assert applied.applied_count == 1
    saved_snapshot = repositories.get_scene_snapshot(save_id)
    assert saved_snapshot is not None
    assert saved_snapshot.in_world_time == "Monday festival week morning"
    assert saved_snapshot.world_time_phase == "morning"
    assert saved_snapshot.world_time_clock_minutes is None
    assert saved_snapshot.world_time_period_label == "festival week"
    assert saved_snapshot.world_time_source_message_id == message.id


def test_dry_run_identifies_clear_date_completed_anchor(
    repositories: PersistenceRepositories,
) -> None:
    save_id, _player_id, _npc_id = _dating_save_with_romance_option(repositories)
    repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="After the first date, Mika Arai thanks Ren for walking her home.",
    )

    report = DatingSimMaintenanceService(repositories).inspect_save(save_id)

    assert report.reviewable_repairs[0].stage == "early_dating"
    assert report.reviewable_repairs[0].dates_completed == 1


def test_apply_requires_confirmation_and_selected_repairs(
    repositories: PersistenceRepositories,
) -> None:
    save_id, player_id, npc_id = _dating_save_with_romance_option(repositories)
    message = repositories.append_message(
        save_id=save_id,
        role="narrator",
        speaker_name="Narrator",
        body="Mika Arai and Ren exchange numbers, then plan a first date.",
    )
    repositories.upsert_scene_snapshot(
        save_id=save_id,
        world_day_index=2,
        source_message_id=message.id,
    )
    service = DatingSimMaintenanceService(repositories)
    dry_run = service.inspect_save(save_id)
    repair_id = dry_run.reviewable_repairs[0].repair_id

    with pytest.raises(ValueError, match="confirm_save_id"):
        service.apply_repairs(save_id, repair_ids=[repair_id], confirm_save_id="")
    with pytest.raises(ValueError, match="repair_ids"):
        service.apply_repairs(save_id, repair_ids=[], confirm_save_id=save_id)

    applied = service.apply_repairs(
        save_id,
        repair_ids=[repair_id],
        confirm_save_id=save_id,
    )

    assert applied.applied_count == 1
    route = repositories.get_dating_route_state_for_pair(save_id, player_id, npc_id)
    assert route is not None
    assert route.stage == "first_date_planned"
    assert route.first_met_message_id == message.id
    assert route.source_message_id == message.id
    audits = repositories.list_context_update_audit(save_id)
    assert audits[-1].entity_type == "dating_route_state"
    assert audits[-1].operation == "dating_sim_maintenance_apply"


def _dating_save_with_romance_option(
    repositories: PersistenceRepositories,
) -> tuple[str, str, str]:
    scenario = repositories.create_scenario(
        type="dating_sim",
        title="Last Summer",
        premise="A summer of route choices.",
        player_role="Transfer student",
        content={
            "player_character_name": "Ren Takahashi",
            "romance_options": "Mika Arai is the class president.",
        },
    )
    save = repositories.create_save(scenario_id=scenario.id, title="Summer Save")
    player = repositories.add_character(
        save_id=save.id,
        name="Ren Takahashi",
        aliases=["Ren"],
        is_player_character=True,
        met=True,
    )
    npc = repositories.add_character(
        save_id=save.id,
        name="Mika Arai",
        relationships={player.name: "romance option for Ren Takahashi"},
        status="available romance option at scenario start",
        met=True,
    )
    return save.id, player.id, npc.id
