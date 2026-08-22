# Multi-User Auth Policy

This document defines Bragi's multi-user authorization baseline. It is the
source of truth for the first authenticated web implementation; frontend-only
enforcement is never sufficient.

Authentication is required by default for the web app. Public unauthenticated
routes are limited to static assets, health, login/logout, current-session
checks, and first-admin bootstrap. A fresh install starts in bootstrap mode until
the first active admin is created. The bootstrap flow starts a session and claims
previously unowned single-user saves for that first admin so legacy trusted-LAN
data remains reachable after migration. Remote LAN bootstrap requires a
one-time setup token from `BRAGI_WEB_BOOTSTRAP_TOKEN`; local loopback bootstrap
does not require the token. Login and bootstrap credential failures are
throttled by source address and username before expensive password verification
continues. Newly created or reset local passwords must be at least 12
characters, while existing shorter passwords may still log in until reset.

Sessions use the `bragi_session` HttpOnly cookie with `SameSite=Lax`. The cookie
is secure on HTTPS, and deployments behind TLS-terminating reverse proxies should
set `BRAGI_WEB_SECURE_COOKIES=1`. Reverse proxies and custom LAN hostnames must
also keep Bragi's host/origin checks aligned with `BRAGI_WEB_ALLOWED_HOSTS` and
`BRAGI_WEB_ALLOWED_ORIGINS`; do not rely on browser-only controls for
authorization.

## Roles And Sharing

- `admin`: can administer users, see and manage every save, manage global
  settings, provider keys, diagnostics, debug prompts, provider payloads, and
  model routing.
- `user`: can use normal roleplay features for owned or shared saves, including
  chat, world data, characters, media, scenarios, and save/scenario/character
  import or export when the target object is accessible.
- `child`: can access owned or assigned saves. Child users may read and
  participate in owned or assigned roleplay saves, but cannot import, export,
  delete, manage provider/model settings, inspect diagnostics, inspect debug
  prompts or provider payloads, or use unsafe media/provider controls. They may
  generate and regenerate media when the generated prompt passes their content
  rating and the provider path enforces safe mode. Child media generation
  currently requires Venice; non-Venice primary and fallback providers fail
  closed before receiving the media request. Uploads, deletion, and reference
  management remain blocked.

Saves are the primary authorization boundary. Regular users and child users can
only see saves they own or are assigned to. Admin users can see all saves.
Scenario templates are shared read-only by default; changing or deleting shared
scenario definitions is admin-only until scenario ownership is implemented.

Provider keys are global application secrets. They are never user-scoped in v1,
and only admins can read key presence, set keys, clear keys, refresh model lists,
or change global model routing.

Scoped settings are stored by global, user, scenario, or save scope. Save-scoped
runtime settings are read in this order: save override, scenario override, user
override, then global default. User presentation settings are read as user
override then global default. Global provider/model/routing/diagnostic settings
are admin-only. Normal users may write save-scoped runtime settings for
accessible saves and their own user-scoped presentation settings. Child users may
write only child-allowed user-scoped settings: pending-jobs display mode and a G
or PG content rating. Admins may grant a child account PG-13. Adult accounts may
select G, PG, PG-13, R, or Unrated; PG-13 is the adult default, while PG is the
child default. Unrated disables the rating gate but does not disable independent
provider safeguards. The narrator fade-to-black classifier is enabled by default
and configurable for adult accounts; it is always enabled for child accounts.

Existing trusted-LAN installs will be assigned during bootstrap. The first admin
bootstrap flow owns migrated single-user saves unless a later migration step
chooses a stricter assignment policy.

`BragiRuntime.active_save_id` is a legacy process-global fallback for
unauthenticated/direct-controller flows and must not be treated as an
authorization boundary. Authenticated routes must resolve save access per request
or per user/session before using a save ID.

## Route Categories

- Public: static assets, non-sensitive health checks, and auth/bootstrap
  endpoints only.
- Authenticated: requires a valid session but is not tied to one save.
- Save-scoped: requires a valid session and access to the referenced save.
- User-scoped: requires a valid session and access to the referenced user-owned
  or user-created object.
- Admin-only: requires `admin`.

| Route | Category | Policy |
| --- | --- | --- |
| `/api/health` | Public | Return a non-sensitive liveness response only. |
| `/api/bootstrap/status` | Public | Report whether first-admin bootstrap and a remote setup token are required without exposing app data. |
| `/api/bootstrap/admin` | Public | Create the first admin only while no active admin exists, require a setup token for remote clients, throttle failures, then start a session. |
| `/api/auth/login` | Public | Throttle failures, validate credentials, and start an HttpOnly cookie session. |
| `/api/auth/logout` | Public | Revoke the current session when present and clear the session cookie. |
| `/api/auth/session` | Public | Return bootstrap status and the current user when a valid session exists, otherwise return `user: null` without exposing app data. |
| `/api/auth/me` | Public | Return the current user for a valid session, otherwise `401`. |
| `/api/admin/users` | Admin-only | List users or create users with roles; never return password hashes. |
| `/api/admin/users/{user_id}` | Admin-only | Change role or status; the last active admin cannot be removed, disabled, or demoted. |
| `/api/admin/users/{user_id}/password` | Admin-only | Set a local password and revoke stale sessions; self-reset keeps the current session. |
| `/api/admin/dating-sim-maintenance` | Admin-only | Inspect dating-sim saves and explicitly apply selected route/time maintenance repairs; dry-run reports redact transcript text by default. |
| `/api/diagnostics` | Authenticated | Admins may read global metadata-only diagnostics; normal users may read active-save health for accessible saves; child users are blocked. |
| `/api/runtime` | Save-scoped | Build runtime only for an accessible save. |
| `/api/runtime/shell` | Save-scoped | Build the bounded startup runtime shell only for an accessible save; heavy side resources stay on dedicated save-scoped routes. |
| `/api/runtime/world-time` | Save-scoped | Correct the current in-world time only for an accessible save; child allowed for owned or assigned saves. |
| `/api/saves` | Authenticated | List saves visible to the current user. |
| `/api/saves/{save_id}/chronicle` | Save-scoped | Read paged chronicle messages only for an accessible save. |
| `/api/saves/{save_id}/media` | Save-scoped | Read scene media history only for an accessible save. |
| `/api/saves/{save_id}/engine-health` | Save-scoped | Read metadata-only engine diagnostics for an accessible save; child blocked. |
| `/api/chat-history` | Save-scoped | Read history only for an accessible save. |
| `/api/chat/submission-status` | Save-scoped | Read queued/running state only for an accessible save. |
| `/api/chat/timing-summary` | Save-scoped | Read metadata-only narrator timing, terminal-outcome, and adaptive-route aggregates for an accessible save. |
| `/api/character-texts` | Save-scoped | Read visible phone contacts, contact repair candidates, and thread summaries only for an accessible save. |
| `/api/character-texts/threads/{thread_id}` | Save-scoped | Read character text thread history only when the thread belongs to an accessible save. |
| `/api/character-texts/groups` | Save-scoped | Create character text group threads only in an accessible save when role policy allows chat and the player has every selected character's number. |
| `/api/character-texts/threads/{thread_id}/read` | Save-scoped | Mark incoming character texts read only when the thread belongs to an accessible save; child allowed for owned or assigned saves. |
| `/api/character-texts/threads/{thread_id}/send` | Save-scoped | Send player texts to an existing character text thread only when the thread belongs to an accessible save and role policy allows chat. |
| `/api/character-texts/threads/{thread_id}/send-image` | Save-scoped | Send image-bearing player texts to an existing character text thread only when the thread belongs to an accessible save and role policy allows chat and media uploads. |
| `/api/character-texts/contacts/{character_id}` | Save-scoped | Manually correct phone contact permissions only for characters in an accessible save; child allowed for owned or assigned saves. |
| `/api/saves/{save_id}/load` | Save-scoped | Load only an accessible save. |
| `/api/saves/{save_id}/rename` | Save-scoped | Rename only accessible saves when role policy allows save mutation; child blocked. |
| `/api/saves/{save_id}` | Save-scoped | Delete only if role policy allows deletion. |
| `/api/saves/{save_id}/events` | Save-scoped | Stream events only for an accessible save. |
| `/api/scenarios` | Authenticated | List shared scenario templates. |
| `/api/scenarios/{scenario_id}/definition` | Authenticated/Admin-only | GET reads shared definitions for authenticated users; POST edits are admin-only. |
| `/api/scenarios/{scenario_id}/character-starters/reference-image/upload` | Admin-only | Upload shared scenario starter reference images only as admin. |
| `/api/scenarios/{scenario_id}/character-starters/reference-image/remove` | Admin-only | Remove shared scenario starter reference images only as admin. |
| `/api/scenarios/{scenario_id}/character-starters/reference-images/{image_id}` | Authenticated | Read shared scenario starter reference images for authenticated users. |
| `/api/scenarios/manual` | Authenticated | Create scenario templates for the current user/admin policy. |
| `/api/scenarios/{scenario_id}/start` | Authenticated | Start a save only from an allowed scenario. |
| `/api/scenarios/{scenario_id}` | Admin-only | Delete shared scenario templates only as admin. |
| `/api/scenarios/draft` | Authenticated | Create drafts for the current user. |
| `/api/scenarios/continuation-draft` | Save-scoped | Create continuation draft only from an accessible save. |
| `/api/scenarios/draft/save` | Authenticated | Save drafts under current user/admin policy. |
| `/api/scenarios/draft/character-starters/generate` | Authenticated | Generate draft character starters for the current user; child blocked. |
| `/api/scenarios/draft/section` | Authenticated | Regenerate draft sections for the current user. |
| `/api/worlds` | Authenticated | List persistent worlds allowed by the current account content-rating policy. |
| `/api/worlds/{world_id}` | Authenticated | Read a persistent world allowed by the current account content-rating policy. |
| `/api/worlds/manual` | Authenticated | Create a persistent world from reviewed setting prose. |
| `/api/worlds/draft` | Authenticated | Generate an AI-assisted persistent-world draft subject to the account content-rating policy. |
| `/api/worlds/draft/save` | Authenticated | Save a reviewed persistent-world draft. |
| `/api/worlds/{world_id}/definition` | Admin-only | Edit a persistent world; linked scenarios use the edited world for future saves. |
| `/api/worlds/{world_id}` (DELETE) | Admin-only | Delete only an unlinked persistent world. |
| `/api/scenarios/{scenario_id}/persistent-world` | Admin-only | Link or unlink a persistent world from a scenario. |
| `/api/chat` | Save-scoped | Submit chat only to an accessible save; child allowed for owned or assigned saves. |
| `/api/chat/retry` | Save-scoped | Retry an interrupted player turn for an accessible save, or an interrupted timeskip when role policy allows save mutation. |
| `/api/chat/continue` | Save-scoped | Continue narration only for an accessible Storyteller-mode save when role policy allows chat. |
| `/api/chat/look-around` | Save-scoped | Ask side-channel scene questions only for an accessible save; child allowed for owned or assigned saves. |
| `/api/character-texts/send` | Save-scoped | Send side-channel character texts only to an accessible save when texts are enabled; child allowed for owned or assigned saves. |
| `/api/character-texts/send-image` | Save-scoped | Send image-bearing side-channel character texts only to an accessible save when texts are enabled and role policy allows chat and media uploads. |
| `/api/character-texts/spontaneous` | Save-scoped | Ask an accessible character text thread to produce a character-originated side-channel text only when both contact directions are allowed; child allowed for owned or assigned saves. |
| `/api/character-texts/message-edit` | Save-scoped | Save character text message edits without resubmitting only in an accessible save; child blocked. |
| `/api/character-texts/delete-from-here` | Save-scoped | Delete character text conversation suffixes only in an accessible save; child blocked. |
| `/api/character-texts/edit` | Save-scoped | Edit and replay player character texts only in an accessible save; child blocked. |
| `/api/chat/timeskip` | Save-scoped | Timeskip only when role can write to the save. |
| `/api/chat/cancel` | Save-scoped | Compatibility fallback for callers without a tracked job ID; cancel only the current user's accessible save job, and prefer `/api/jobs/{job_id}/cancel` when a job ID is available. Child cannot cancel another user's active chat job. |
| `/api/action-choices/regenerate` | Save-scoped | Regenerate generated action choices only for an accessible save; child allowed as normal chat participation. |
| `/api/chat/regenerate` | Save-scoped | Regenerate only messages in an accessible save. |
| `/api/runtime/custom-instructions` | Save-scoped | Update only accessible save instructions; child blocked. |
| `/api/chat/edit` | Save-scoped | Edit only messages in an accessible save; child blocked unless explicitly allowed later. |
| `/api/chat/message-edit` | Save-scoped | Save message text edits without resubmitting only in an accessible save; child blocked. |
| `/api/chat/narrator-edit` | Save-scoped | Edit narrator text only in an accessible save; child blocked. |
| `/api/chat/delete-from-here` | Save-scoped | Delete timeline only when role policy allows destructive edits. |
| `/api/chat/fork-from-here` | Save-scoped | Fork only from an accessible save. |
| `/api/messages/{message_id}/scene-presence` | Save-scoped | GET reads message scene presence for an accessible save; POST replaces presence only when role can mutate the save. |
| `/api/media/generate` | Save-scoped | Generate media only for an accessible save. Child requests are allowed only when the generated prompt passes the account rating and Venice is the provider, with safe mode forced on and unsafe fallbacks disabled. |
| `/api/media/generate-character-image` | Save-scoped | Generate a solo image for a selected present character with an existing reference image in an accessible save. Child content safeguards apply. |
| `/api/media/initial` | Save-scoped | Generate initial media only for an accessible save. Child content safeguards apply. |
| `/api/media/character-reference/upload` | Save-scoped | Upload references only for an accessible save; child blocked. |
| `/api/media/character-reference/remove` | Save-scoped | Remove references only for an accessible save; child blocked. |
| `/api/media/{asset_id}/prompt` | Save-scoped | Read editable raw prompts only for generated image media in an accessible save; child access is limited to safeguarded regeneration workflows. |
| `/api/media/{asset_id}/animate` | Save-scoped | Animate only media in an accessible save. Child content safeguards apply. |
| `/api/media/{asset_id}/regenerate` | Save-scoped | Regenerate only media in an accessible save. Child content safeguards apply to edited prompts. |
| `/api/media/{asset_id}/set-character-reference` | Save-scoped | Set references only for media in an accessible save; child blocked. |
| `/api/media/{asset_id}` | Save-scoped | Read/delete only media in an accessible save; deletion blocked for child. |
| `/api/media/{asset_id}/thumbnail` | Save-scoped | Read only thumbnails for media in an accessible save. |
| `/api/world-data` | Save-scoped | Read world data only for an accessible save. |
| `/api/world-data/apply` | Save-scoped | Apply world edits only when role can mutate the save. |
| `/api/world-data/time-loop/baseline` | Save-scoped | Capture a loop reset baseline only when role can mutate the save. |
| `/api/world-data/time-loop/reset` | Save-scoped | Reset a loop only when role can mutate the save. |
| `/api/world-data/context-cleanup` | Save-scoped | Cleanup only accessible saves; child blocked. |
| `/api/world-data/summary-backfill` | Save-scoped | Compact long-save summaries only for accessible saves; child blocked. |
| `/api/world-data/suggestion-review` | Save-scoped | Review queued suggestions only for accessible saves; child blocked. |
| `/api/world-data/context-retention` | Save-scoped | Expire stale suggestions and prune world-data support history only for accessible saves; child blocked. |
| `/api/world-data/guided-cleanup` | Save-scoped | Guided cleanup only accessible saves; child blocked. |
| `/api/characters` | Save-scoped | Read characters only for an accessible save. |
| `/api/characters/{character_id}/image/generate` | Save-scoped | Generate a registry picture from an existing character reference image in an accessible save. Child content safeguards apply. |
| `/api/characters/{character_id}/reference-image/generate` | Save-scoped | Generate only for characters in an accessible save. Child content safeguards apply. |
| `/api/characters/{character_id}/reference-image/upload` | Save-scoped | Upload only for characters in an accessible save; child blocked. |
| `/api/characters/{character_id}/reference-image/set` | Save-scoped | Set only for characters in an accessible save; child blocked. |
| `/api/characters/{character_id}/reference-image/remove` | Save-scoped | Remove only for characters in an accessible save; child blocked. |
| `/api/characters/apply` | Save-scoped | Apply character edits only when role can mutate the save. |
| `/api/characters/{character_id}/enhance-field` | Save-scoped | Enhance character fields only when role can mutate the save. |
| `/api/characters/{character_id}/knowledge/apply` | Save-scoped | Apply knowledge edits only when role can mutate the save. |
| `/api/character-bundles/export/{character_id}` | Save-scoped | Export only allowed characters from accessible saves; child blocked. Character bundle private-note export is admin-only. |
| `/api/character-bundles/preview` | Save-scoped | Preview import only for accessible target saves; child blocked. |
| `/api/character-bundles/import/{preview_id}` | User-scoped | Import only previews created by this user/session and allowed target save. |
| `/api/settings` | Authenticated | Return only settings visible to the current role. |
| `/api/settings/shell` | Authenticated | Return only lightweight workbench presentation settings visible to the current role. |
| `/api/settings/providers` | Authenticated | Return only provider settings visible to the current role; admin-only secret storage warnings stay hidden from non-admin users. |
| `/api/settings/models` | Authenticated | Return normalized global model settings only when visible to the current role. |
| `/api/settings/openrouter` | Authenticated | Return OpenRouter routing settings only when visible to the current role. |
| `/api/settings/save` | Authenticated | Return only role-visible settings for the requested accessible save. |
| `/api/log/client` | Authenticated | Accept sanitized client logs from authenticated sessions. |
| `/api/settings/provider-key` | Admin-only | Set provider keys only as admin. |
| `/api/settings/provider-key/{provider}` | Admin-only | Clear provider keys only as admin. |
| `/api/settings/model-refresh/{provider}` | Admin-only | Refresh global provider models only as admin. |
| `/api/settings/model-preference` | Admin-only | Change global model preferences only as admin. |
| `/api/settings/model-preference/{task}` | Admin-only | Clear global model preferences only as admin. |
| `/api/settings/model-thinking` | Admin-only | Change global model thinking-level preferences only as admin. |
| `/api/settings/model-thinking/{task}` | Admin-only | Clear global model thinking-level preferences only as admin. |
| `/api/settings/model-routing-profiles` | Admin-only | Save or overwrite global model routing profiles only as admin. |
| `/api/settings/model-routing-profiles/{profile_id}/apply` | Admin-only | Apply saved global model routing profiles only as admin. |
| `/api/settings/model-routing-profiles/{profile_id}` | Admin-only | Delete saved global model routing profiles only as admin. |
| `/api/settings/scoped` | Authenticated | Allow only role-visible global/user/save scoped settings; reject hidden keys server-side. |
| `/api/settings/local` | Authenticated | Legacy alias for `/api/settings/scoped`. |
| `/api/bundles/preview` | Authenticated | Preview save import only for roles allowed to import; child blocked. |
| `/api/bundles/import/{preview_id}` | User-scoped | Import only previews created by this user/session. |
| `/api/scenario-bundles/preview` | Authenticated | Preview scenario import only for roles allowed to import; child blocked. |
| `/api/scenario-bundles/import/{preview_id}` | User-scoped | Import only previews created by this user/session. |
| `/api/scenario-bundles/export/{scenario_id}` | Authenticated | Export only scenarios visible to the current role; child blocked. |
| `/api/persistent-world-bundles/export/{world_id}` | Authenticated | Export a persistent world; child users are blocked by import/export policy. |
| `/api/persistent-world-bundles/preview` | Authenticated | Preview persistent-world import; child users are blocked. |
| `/api/persistent-world-bundles/import/{preview_id}` | User-scoped | Import only a persistent-world preview created by this user/session. |
| `/api/bundles/export` | Save-scoped | Export only an accessible save; child blocked. |
| `/api/jobs` | Authenticated | List only jobs visible to the current user. |
| `/api/jobs/{job_id}` | User-scoped | Read only jobs created by or visible to the current user. |
| `/api/jobs/{job_id}/cancel` | User-scoped | Cancel only jobs created by or visible to the current user; child can cancel only jobs they created. |
| `/api/jobs/{job_id}/events` | User-scoped | Stream only job events visible to the current user. |
| `/api/jobs/{job_id}/steps` | User-scoped | Read metadata-only persisted job step diagnostics only for jobs created by or visible to the current user; prompt, payload, result, and free-text error content stay hidden. |
| `/api/jobs/{job_id}/diagnostics` | User-scoped/Admin detail | Read structured terminal job diagnostics for a visible job; normal users receive metadata-only detail, admins may inspect bounded captured prompts and structured provider/Bragi failure data, and child users are blocked. |
