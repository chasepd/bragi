# Narration Query Index Audit

Issue: #668

## Method

The audit compared `EXPLAIN QUERY PLAN` output for the save-scoped list queries
used by narration context search, deterministic context assembly, and post-turn
context updates.

The "before" plans used a migrated database with the schema-39 indexes removed.
The "after" plans used the same query shapes with the schema-39 indexes present.
The goal was planner shape, not a timing benchmark: save-scoped active-row reads
should use targeted indexes and avoid temporary sort B-trees where practical.

## Added Indexes

| Query | Before | After |
| --- | --- | --- |
| Active messages by save ordered by rowid | Full `messages` scan | `idx_messages_save_active_row_order` |
| Active world state by save ordered by key | Existing unique save/key index, no active-row filter support | `idx_world_state_save_active_key` |
| Active context sources by save/type/title/created | Existing unique index plus temp B-tree sort | `idx_context_sources_save_active_type_title_created` |
| Active context observations by save/created | Existing save/created index plus active-row filtering and temp sort | `idx_context_observations_save_active_created` |
| Active locations by save/name/created | Existing unique save/id index plus temp sort | `idx_locations_save_active_name_created` |
| Active characters by save/name/created | Full `characters` scan plus temp sort | `idx_characters_save_active_name_created` |
| Active threads by save/priority/created | Full `active_threads` scan plus temp sort | `idx_active_threads_save_active_priority_created` |
| State changes by save/created | Full `state_changes` scan plus temp sort | `idx_state_changes_save_created` |
| Active memories by save/created | Full `memories` scan plus temp sort | `idx_memories_save_active_created` |
| Active summaries by save/created | Full `summaries` scan plus temp sort | `idx_summaries_save_active_created` |
| Active media assets by save/created | Full `media_assets` scan plus temp sort | `idx_media_assets_save_active_created` |

Most new indexes are partial active-row indexes. That keeps archived/deleted
audit and restore data out of the hot narration indexes while preserving the
cold audit paths unchanged.

## Skipped Indexes

- `entity_links`: `list_entity_links()` already uses the unique
  `(save_id, entity_type, entity_id, target_type, target_id, relation)` autoindex.
- `character_knowledge_edges`: targeted save/character and save/target indexes
  already exist for the current access patterns.
- `message_visibility`: targeted save/message and save/character indexes already
  exist for the current access patterns.
- `list_messages(include_deleted=True)` and `list_all_media_assets()` remain
  cold audit/export paths and still prefer behavioral simplicity over extra
  indexes in this PR.
