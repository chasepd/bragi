"""Settings and provider model preference service."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Protocol, runtime_checkable

from bragi.app_logging import (
    exception_log_fields,
    log_error_event,
    log_event,
    set_debug_logging_enabled,
)
from bragi.model_tasks import is_retired_model_task
from bragi.persistence.models import ModelPreferenceRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ProviderCapability,
    ProviderCatalogEntry,
    ProviderConfigStatus,
    ProviderModel,
    ProviderModelListResponse,
    ProviderModelPricing,
    ProviderThinkingLevelSupport,
    VideoProvider,
)
from bragi.providers.errors import (
    ProviderError,
    ProviderErrorCategory,
    map_exception_to_category,
)
from bragi.services.agentic_context import (
    PLAN_FIRST_NARRATOR_DEFAULT,
    PLAN_FIRST_NARRATOR_SETTING,
)
from bragi.services.character_action_planning_service import (
    CHARACTER_ACTION_PLANNING_ENABLED_DEFAULT,
    CHARACTER_ACTION_PLANNING_ENABLED_SETTING,
    CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING,
    sanitize_character_action_planning_max_concurrency,
)
from bragi.services.character_text_service import (
    CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING,
    CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING,
    DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT,
    DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS,
    sanitize_character_text_proactive_random_chance_percent,
    sanitize_character_text_proactive_random_cooldown_turns,
)
from bragi.services.chat_history_settings import (
    DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW,
    DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW,
    NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    RECENT_NARRATOR_MESSAGE_WINDOW_SETTING,
    RECENT_PLAYER_MESSAGE_WINDOW_SETTING,
    ChatHistoryWindowSettings,
    chat_history_window_settings,
    narrator_planner_chat_history_window_settings,
    sanitize_recent_message_window,
)
from bragi.services.content_rating import (
    CONTENT_FILTER_RATING_SETTING,
    FADE_TO_BLACK_ENABLED_SETTING,
    sanitize_content_rating,
    set_user_content_rating,
)
from bragi.services.context_assembly import (
    DEFAULT_CONTEXT_BUDGET_ADAPTIVE_FRACTION,
    DEFAULT_CONTEXT_BUDGET_FIXED_TOTAL_CHARS,
    DEFAULT_CONTEXT_BUDGET_MODE,
    ContextBudgetSettings,
    context_budget_settings,
)
from bragi.services.director_pressure_service import (
    DIRECTOR_PRESSURE_ENABLED_DEFAULT,
    DIRECTOR_PRESSURE_ENABLED_SETTING,
)
from bragi.services.generation_settings import (
    CHAT_MAX_OUTPUT_TOKENS_SETTING,
    CHAT_TEMPERATURE_SETTING,
    DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
    DEFAULT_CHAT_TEMPERATURE,
    DEFAULT_IMAGE_DIMENSION_PRESET,
    IMAGE_DIMENSION_PRESET_SETTING,
    MODEL_THINKING_PREFERENCES_SETTING,
    OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING,
    THINKING_LEVEL_OFF,
    THINKING_LEVEL_PROVIDER_DEFAULT,
    sanitize_chat_max_output_tokens,
    sanitize_chat_temperature,
    sanitize_image_dimension_preset,
    sanitize_model_thinking_preferences,
    sanitize_openrouter_chat_reasoning_overrides,
    sanitize_thinking_level,
)
from bragi.services.image_style_settings import (
    DEFAULT_IMAGE_STYLE_PRESET,
    IMAGE_STYLE_PRESET_SETTING,
    sanitize_image_style_preset,
)
from bragi.services.job_lifecycle import JobLifecycleService
from bragi.services.manual_confirmation import (
    MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING,
    MANUAL_CONFIRMATION_MEMORIES_SETTING,
    MANUAL_CONFIRMATION_STATE_CHANGES_SETTING,
)
from bragi.services.model_capabilities import (
    STRUCTURED_OUTPUT_CAPABILITIES,
    model_supports_any_capability,
)
from bragi.services.model_preferences import (
    CONTENT_SAFETY_PURPOSE,
    clear_save_model_override_preference,
    clear_save_model_thinking_preference,
    model_preference_for_selector,
    roleplay_model_purpose,
    save_model_thinking_preference,
    set_save_model_override_preference,
    set_save_model_thinking_preference,
)
from bragi.services.model_routing_profiles import (
    MODEL_ROUTING_PROFILES_SETTING,
    default_model_routing_profiles,
    sanitize_model_routing_profiles,
)
from bragi.services.openrouter_routing_settings import (
    OPENROUTER_ROUTING_PROFILES_SETTING,
    default_openrouter_routing_profiles,
    sanitize_openrouter_routing_profiles,
)
from bragi.services.pending_jobs_settings import (
    DEFAULT_PENDING_JOBS_DISPLAY_MODE,
    PENDING_JOBS_DISPLAY_MODE_SETTING,
    sanitize_pending_jobs_display_mode,
)
from bragi.services.phrase_denylist import (
    GENERATED_PHRASE_DENYLIST_SETTING,
    SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
    sanitize_generated_phrase_denylist,
)
from bragi.services.post_turn_inference import (
    POST_TURN_INFERENCE_MODE_DEFAULT,
    POST_TURN_INFERENCE_MODE_SETTING,
    sanitize_post_turn_inference_mode,
)
from bragi.services.provider_diagnostics import (
    record_provider_error,
    record_provider_response,
)
from bragi.services.scenario_evolution_policy import (
    DEFAULT_SCENARIO_EVOLUTION_TURN_INTERVAL,
    SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING,
    sanitize_scenario_evolution_turn_interval,
)
from bragi.services.secrets import SecretStorageError, SecretStore
from bragi.services.settings_policy import (
    ScopedSettingPolicy,
    scoped_setting_policy,
)
from bragi.services.text_script_policy import (
    DEFAULT_SCRIPT_GUARD_MODE,
    SCRIPT_GUARD_MODE_SETTING,
    sanitize_script_guard_mode,
)
from bragi.services.user_narration_guidance import (
    DEFAULT_USER_NARRATION_GUIDANCE,
    USER_NARRATION_GUIDANCE_SETTING,
    sanitize_user_narration_guidance,
)


@dataclass(frozen=True)
class ProviderRefreshResult:
    provider: str
    configured: bool
    authenticated: bool
    model_count: int
    error: str | None


class ModelListingProvider(Protocol):
    provider_name: str

    async def validate_config(self) -> ProviderConfigStatus: ...

    async def list_models(self) -> list[ProviderModel]: ...


@runtime_checkable
class MetadataModelListingProvider(Protocol):
    async def list_models_with_metadata(self) -> ProviderModelListResponse: ...


@runtime_checkable
class ProviderCatalogListingProvider(Protocol):
    async def list_providers(self) -> list[ProviderCatalogEntry]: ...


class SettingsService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: Mapping[str, ModelListingProvider],
        secret_store: SecretStore,
        log_file_path: Path | None = None,
    ) -> None:
        self.repositories = repositories
        self.providers = providers
        self.secret_store = secret_store
        self.log_file_path = log_file_path
        self.jobs = JobLifecycleService(repositories=repositories)

    def set_provider_api_key(self, provider: str, api_key: str) -> None:
        normalized_key = api_key.strip()
        if not normalized_key:
            raise ValueError("API key must not be blank")
        self.secret_store.set_api_key(provider, normalized_key)
        self.repositories.upsert_provider_config(
            provider=provider,
            enabled=True,
            has_api_key=True,
        )
        log_event("settings.provider_api_key_saved", provider=provider)

    def clear_provider_api_key(self, provider: str) -> None:
        self.secret_store.delete_api_key(provider)
        self.repositories.upsert_provider_config(
            provider=provider,
            enabled=False,
            has_api_key=False,
            last_model_refresh_at=None,
            last_error=None,
        )
        log_event("settings.provider_api_key_cleared", provider=provider)

    async def refresh_provider_models(self, provider: str) -> ProviderRefreshResult:
        client = self.providers[provider]
        job = self.jobs.create_running(
            type="model_refresh",
            payload={"provider": provider},
            collect_provider_diagnostics=True,
        )
        started_at = perf_counter()
        log_event("settings.model_refresh_started", provider=provider)
        try:
            has_api_key = self.secret_store.has_api_key(provider)
        except SecretStorageError as exc:
            self.jobs.fail(job.id, error=str(exc) or exc.__class__.__name__)
            result = self._record_provider_exception(
                provider=provider,
                has_api_key=True,
                exc=exc,
            )
            log_error_event(
                "settings.model_refresh_failed",
                provider=provider,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            return result
        try:
            status = await client.validate_config()
        except Exception as exc:
            record_provider_error(
                task="model_listing",
                provider=provider,
                exc=exc,
            )
            self.jobs.fail(job.id, error=str(exc) or exc.__class__.__name__)
            result = self._record_provider_exception(provider, has_api_key, exc)
            log_error_event(
                "settings.model_refresh_failed",
                provider=provider,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            return result
        record_provider_response(
            task="model_listing",
            provider=provider,
            raw_metadata=status.diagnostics,
        )

        self.repositories.upsert_provider_config(
            provider=provider,
            enabled=status.configured,
            has_api_key=has_api_key,
            last_error=status.error,
        )
        if not status.configured or not status.authenticated:
            self.jobs.fail(
                job.id,
                error=status.error or "Provider is not configured or authenticated",
                result={
                    "configured": status.configured,
                    "authenticated": status.authenticated,
                },
            )
            result = ProviderRefreshResult(
                provider=provider,
                configured=status.configured,
                authenticated=status.authenticated,
                model_count=0,
                error=status.error,
            )
            log_error_event(
                "settings.model_refresh_failed",
                provider=provider,
                configured=status.configured,
                authenticated=status.authenticated,
                duration_ms=_elapsed_ms(started_at),
                error=status.error,
            )
            return result

        try:
            listing = await self._list_models_with_metadata(client)
        except Exception as exc:
            record_provider_error(
                task="model_listing",
                provider=provider,
                exc=exc,
            )
            self.jobs.fail(job.id, error=str(exc) or exc.__class__.__name__)
            result = self._record_provider_exception(provider, has_api_key, exc)
            log_error_event(
                "settings.model_refresh_failed",
                provider=provider,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            return result
        models = listing.models
        record_provider_response(
            task="model_listing",
            provider=provider,
            raw_metadata=listing.raw_metadata,
        )
        catalog_entries = await self._list_provider_catalog(provider, client)

        try:
            self.repositories.begin_transaction()
            for model in models:
                self._save_provider_model(model, client=client)
            self.repositories.mark_missing_provider_models_unavailable(
                provider=provider,
                available_model_ids={model.model_id for model in models},
            )
            if catalog_entries is not None:
                self.repositories.replace_provider_catalog_entries(
                    provider=provider,
                    entries=[
                        provider_catalog_entry_json(entry)
                        for entry in catalog_entries
                    ],
                )
            self.repositories.upsert_provider_config(
                provider=provider,
                enabled=True,
                has_api_key=has_api_key,
                last_model_refresh_at=_utc_now(),
                last_error=None,
            )
            self.repositories.commit_transaction()
        except Exception as exc:
            self.repositories.rollback_transaction()
            self.jobs.fail(job.id, error=str(exc) or exc.__class__.__name__)
            result = self._record_provider_exception(provider, has_api_key, exc)
            log_error_event(
                "settings.model_refresh_failed",
                provider=provider,
                duration_ms=_elapsed_ms(started_at),
                **exception_log_fields(exc),
            )
            return result
        result = ProviderRefreshResult(
            provider=provider,
            configured=True,
            authenticated=True,
            model_count=len(models),
            error=None,
        )
        log_event(
            "settings.model_refresh_succeeded",
            provider=provider,
            model_count=result.model_count,
            duration_ms=_elapsed_ms(started_at),
        )
        self.jobs.succeed(job.id, result={"model_count": result.model_count})
        return result

    async def _list_models_with_metadata(
        self,
        client: ModelListingProvider,
    ) -> ProviderModelListResponse:
        if isinstance(client, MetadataModelListingProvider):
            return await client.list_models_with_metadata()
        return ProviderModelListResponse(models=await client.list_models())

    async def _list_provider_catalog(
        self,
        provider: str,
        client: ModelListingProvider,
    ) -> list[ProviderCatalogEntry] | None:
        if not isinstance(client, ProviderCatalogListingProvider):
            return None
        try:
            catalog_entries = await client.list_providers()
        except Exception as exc:
            record_provider_error(
                task="provider_catalog",
                provider=provider,
                exc=exc,
            )
            log_error_event(
                "settings.provider_catalog_refresh_failed",
                provider=provider,
                **exception_log_fields(exc),
            )
            return None
        log_event(
            "settings.provider_catalog_refresh_succeeded",
            provider=provider,
            provider_count=len(catalog_entries),
        )
        return catalog_entries

    def set_model_preference(
        self,
        task: str,
        provider: str,
        model_id: str,
        *,
        save_id: str | None = None,
    ) -> None:
        if is_retired_model_task(task):
            raise ValueError("Model task is retired")
        if (
            roleplay_model_purpose(task) == CONTENT_SAFETY_PURPOSE
            and not model_supports_any_capability(
                self.repositories,
                provider=provider,
                model_id=model_id,
                required=STRUCTURED_OUTPUT_CAPABILITIES,
            )
        ):
            raise ValueError("Safety Agent model must support structured output")
        if save_id is not None:
            self._set_save_model_preference(
                task=task,
                provider=provider,
                model_id=model_id,
                save_id=save_id,
            )
            return
        self.repositories.set_model_preference(
            task=task,
            provider=provider,
            model_id=model_id,
        )
        thinking_preferences = sanitize_model_thinking_preferences(
            self.repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING)
        )
        thinking_preference = thinking_preferences.get(task)
        if thinking_preference is not None and (
            thinking_preference["provider"] != provider
            or thinking_preference["model_id"] != model_id
        ):
            self.clear_model_thinking_preference(task)
        log_event(
            "settings.model_preference_saved",
            task=task,
            provider=provider,
            model=model_id,
        )

    def get_model_preference(
        self,
        task: str,
        *,
        save_id: str | None = None,
    ) -> ModelPreferenceRecord | None:
        if save_id is not None:
            return model_preference_for_selector(
                self.repositories,
                task,
                save_id=save_id,
            )
        return self.repositories.get_model_preference(task)

    def clear_model_preference(
        self,
        task: str,
        *,
        save_id: str | None = None,
    ) -> None:
        if save_id is not None:
            clear_save_model_override_preference(
                self.repositories,
                save_id=save_id,
                task=task,
            )
            log_event("settings.save_model_preference_cleared", task=task)
            return
        self.repositories.clear_model_preference(task)
        self.clear_model_thinking_preference(task)
        log_event("settings.model_preference_cleared", task=task)

    def _set_save_model_preference(
        self,
        *,
        task: str,
        provider: str,
        model_id: str,
        save_id: str,
    ) -> None:
        self._require_save(save_id, "model preference")
        normalized_provider = provider.strip().casefold()
        normalized_model_id = model_id.strip()
        inherited = model_preference_for_selector(self.repositories, task)
        if (
            inherited is not None
            and inherited.provider == normalized_provider
            and inherited.model_id == normalized_model_id
        ):
            clear_save_model_override_preference(
                self.repositories,
                save_id=save_id,
                task=task,
            )
        else:
            set_save_model_override_preference(
                self.repositories,
                save_id=save_id,
                task=task,
                provider=normalized_provider,
                model_id=normalized_model_id,
            )
        thinking_preference = save_model_thinking_preference(
            self.repositories,
            save_id=save_id,
            task=task,
        )
        if thinking_preference is not None and (
            thinking_preference["provider"] != normalized_provider
            or thinking_preference["model_id"] != normalized_model_id
        ):
            self.clear_model_thinking_preference(task, save_id=save_id)
        log_event(
            "settings.save_model_preference_saved",
            task=task,
            provider=normalized_provider,
            model=normalized_model_id,
        )

    def set_model_thinking_preference(
        self,
        *,
        task: str,
        provider: str,
        model_id: str,
        level: str,
        save_id: str | None = None,
    ) -> None:
        if is_retired_model_task(task):
            raise ValueError("Model task is retired")
        sanitized_level = sanitize_thinking_level(level)
        if sanitized_level is None:
            raise ValueError("Unknown thinking level")
        if sanitized_level == THINKING_LEVEL_PROVIDER_DEFAULT:
            self.clear_model_thinking_preference(task, save_id=save_id)
            return
        normalized_provider = provider.strip().casefold()
        normalized_model_id = model_id.strip()
        support = _thinking_support_for_model(
            self.repositories,
            provider=normalized_provider,
            model_id=normalized_model_id,
        )
        if support is None:
            raise ValueError("Selected model does not support thinking level")
        if sanitized_level == THINKING_LEVEL_OFF:
            if support.get("mandatory") is True:
                raise ValueError("Selected model requires thinking")
        elif sanitized_level not in _thinking_support_levels(support):
            raise ValueError("Thinking level is not supported by selected model")
        if save_id is not None:
            self._require_save(save_id, "model thinking preference")
            set_save_model_thinking_preference(
                self.repositories,
                save_id=save_id,
                task=task,
                provider=normalized_provider,
                model_id=normalized_model_id,
                level=sanitized_level,
            )
        else:
            preferences = sanitize_model_thinking_preferences(
                self.repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING)
            )
            preferences[task] = {
                "provider": normalized_provider,
                "model_id": normalized_model_id,
                "level": sanitized_level,
            }
            self.repositories.set_app_setting(
                MODEL_THINKING_PREFERENCES_SETTING,
                preferences,
            )
        log_event(
            "settings.model_thinking_preference_saved",
            task=task,
            provider=normalized_provider,
            model=normalized_model_id,
            level=sanitized_level,
        )

    def clear_model_thinking_preference(
        self,
        task: str,
        *,
        save_id: str | None = None,
    ) -> None:
        if save_id is not None:
            clear_save_model_thinking_preference(
                self.repositories,
                save_id=save_id,
                task=task,
            )
            log_event("settings.save_model_thinking_preference_cleared", task=task)
            return
        preferences = sanitize_model_thinking_preferences(
            self.repositories.get_app_setting(MODEL_THINKING_PREFERENCES_SETTING)
        )
        if task in preferences:
            preferences.pop(task)
            self.repositories.set_app_setting(
                MODEL_THINKING_PREFERENCES_SETTING,
                preferences,
            )
        log_event("settings.model_thinking_preference_cleared", task=task)

    def _require_save(self, save_id: str, label: str) -> None:
        if self.repositories.get_save(save_id) is None:
            raise ValueError(f"{label} requires a save")

    def set_scoped_app_setting(
        self,
        key: str,
        value: object,
        *,
        save_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        policy = scoped_setting_policy(key)
        if key == RECENT_PLAYER_MESSAGE_WINDOW_SETTING:
            value = sanitize_recent_message_window(
                value,
                default=ChatHistoryWindowSettings.defaults().player_messages,
            )
        elif key == RECENT_NARRATOR_MESSAGE_WINDOW_SETTING:
            value = sanitize_recent_message_window(
                value,
                default=ChatHistoryWindowSettings.defaults().narrator_messages,
            )
        elif key == NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING:
            value = sanitize_recent_message_window(
                value,
                default=narrator_planner_chat_history_window_settings(
                    self.repositories,
                    save_id=save_id,
                ).player_messages,
            )
        elif key == NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING:
            value = sanitize_recent_message_window(
                value,
                default=narrator_planner_chat_history_window_settings(
                    self.repositories,
                    save_id=save_id,
                ).narrator_messages,
            )
        elif key == SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING:
            value = sanitize_scenario_evolution_turn_interval(value)
        elif key == CHARACTER_ACTION_PLANNING_MAX_CONCURRENCY_SETTING:
            value = sanitize_character_action_planning_max_concurrency(value)
        elif key == CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING:
            value = sanitize_character_text_proactive_random_chance_percent(value)
        elif key == CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING:
            value = sanitize_character_text_proactive_random_cooldown_turns(value)
        elif key == IMAGE_STYLE_PRESET_SETTING:
            if save_id is None or self.repositories.get_save(save_id) is None:
                raise ValueError("Image style preset requires a save")
            value = sanitize_image_style_preset(value)
        elif key == CHAT_TEMPERATURE_SETTING:
            value = sanitize_chat_temperature(value)
        elif key == CHAT_MAX_OUTPUT_TOKENS_SETTING:
            value = sanitize_chat_max_output_tokens(value)
        elif key == IMAGE_DIMENSION_PRESET_SETTING:
            value = sanitize_image_dimension_preset(value)
        elif key == OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING:
            value = sanitize_openrouter_chat_reasoning_overrides(value)
        elif key == MODEL_THINKING_PREFERENCES_SETTING:
            value = sanitize_model_thinking_preferences(value)
        elif key == OPENROUTER_ROUTING_PROFILES_SETTING:
            value = sanitize_openrouter_routing_profiles(value)
        elif key == MODEL_ROUTING_PROFILES_SETTING:
            value = sanitize_model_routing_profiles(value)
        elif key == PENDING_JOBS_DISPLAY_MODE_SETTING:
            value = sanitize_pending_jobs_display_mode(value)
        elif key == POST_TURN_INFERENCE_MODE_SETTING:
            value = sanitize_post_turn_inference_mode(value)
        elif key == USER_NARRATION_GUIDANCE_SETTING:
            value = sanitize_user_narration_guidance(value)
        elif key == CONTENT_FILTER_RATING_SETTING:
            if user_id is not None:
                set_user_content_rating(
                    self.repositories,
                    user_id=user_id,
                    rating=value,
                )
                log_event("settings.scoped_setting_saved", key=key)
                return
            value = sanitize_content_rating(value)
        elif key == FADE_TO_BLACK_ENABLED_SETTING and not isinstance(value, bool):
            raise ValueError("Fade to black setting must be boolean")
        elif key == SCRIPT_GUARD_MODE_SETTING:
            value = sanitize_script_guard_mode(value)
        elif key in {
            GENERATED_PHRASE_DENYLIST_SETTING,
            SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
        }:
            value = sanitize_generated_phrase_denylist(value)
        if policy.scope == "save" and save_id is not None:
            if self.repositories.get_save(save_id) is None:
                raise ValueError(f"{key} requires a save")
            self.repositories.set_scoped_setting(
                scope="save",
                scope_id=save_id,
                key=key,
                value=value,
            )
        elif policy.scope == "user" and user_id is not None:
            self.repositories.set_scoped_setting(
                scope="user",
                scope_id=user_id,
                key=key,
                value=value,
            )
        else:
            self.repositories.set_app_setting(key, value)
        if key == "debug_logging_enabled":
            set_debug_logging_enabled(bool(value))
        log_event("settings.scoped_setting_saved", key=key)

    def set_local_setting(
        self,
        key: str,
        value: object,
        *,
        save_id: str | None = None,
        user_id: str | None = None,
    ) -> None:
        self.set_scoped_app_setting(
            key,
            value,
            save_id=save_id,
            user_id=user_id,
        )

    def get_scoped_app_setting(
        self,
        key: str,
        default: object = None,
        *,
        save_id: str | None = None,
        user_id: str | None = None,
    ) -> object:
        if key == IMAGE_STYLE_PRESET_SETTING and save_id is None:
            return default if default is not None else DEFAULT_IMAGE_STYLE_PRESET
        if key == IMAGE_STYLE_PRESET_SETTING:
            value = self.repositories.get_scoped_setting(
                scope="save",
                scope_id=save_id,
                key=IMAGE_STYLE_PRESET_SETTING,
            )
            if value is not None:
                return sanitize_image_style_preset(value)
            return default if default is not None else DEFAULT_IMAGE_STYLE_PRESET
        try:
            policy = scoped_setting_policy(key)
        except ValueError:
            policy = ScopedSettingPolicy(scope="global")
        value = self.repositories.get_effective_setting(
            key,
            save_id=save_id if policy.scope == "save" else None,
            user_id=user_id if policy.scope == "user" else None,
        )
        if value is not None:
            if key == IMAGE_STYLE_PRESET_SETTING:
                return sanitize_image_style_preset(value)
            if key == USER_NARRATION_GUIDANCE_SETTING:
                return sanitize_user_narration_guidance(value)
            if key == CONTENT_FILTER_RATING_SETTING:
                return sanitize_content_rating(value)
            if key == POST_TURN_INFERENCE_MODE_SETTING:
                return sanitize_post_turn_inference_mode(value)
            if key == SCRIPT_GUARD_MODE_SETTING:
                return sanitize_script_guard_mode(value)
            if key in {
                GENERATED_PHRASE_DENYLIST_SETTING,
                SAVE_GENERATED_PHRASE_DENYLIST_SETTING,
            }:
                return sanitize_generated_phrase_denylist(value)
            if key == CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING:
                return sanitize_character_text_proactive_random_chance_percent(value)
            if key == CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING:
                return sanitize_character_text_proactive_random_cooldown_turns(value)
            if key == NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING:
                return sanitize_recent_message_window(
                    value,
                    default=narrator_planner_chat_history_window_settings(
                        self.repositories,
                        save_id=save_id,
                    ).player_messages,
                )
            if key == NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING:
                return sanitize_recent_message_window(
                    value,
                    default=narrator_planner_chat_history_window_settings(
                        self.repositories,
                        save_id=save_id,
                    ).narrator_messages,
                )
            return value
        if default is not None:
            return default
        if key == NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING:
            return narrator_planner_chat_history_window_settings(
                self.repositories,
                save_id=save_id,
            ).player_messages
        if key == NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING:
            return narrator_planner_chat_history_window_settings(
                self.repositories,
                save_id=save_id,
            ).narrator_messages
        return _default_local_setting(key)

    def get_local_setting(
        self,
        key: str,
        default: object = None,
        *,
        save_id: str | None = None,
        user_id: str | None = None,
    ) -> object:
        return self.get_scoped_app_setting(
            key,
            default,
            save_id=save_id,
            user_id=user_id,
        )

    def get_context_budget_settings(
        self,
        *,
        save_id: str | None = None,
    ) -> ContextBudgetSettings:
        return context_budget_settings(self.repositories, save_id=save_id)

    def set_context_budget_settings(
        self,
        settings: ContextBudgetSettings,
        *,
        save_id: str | None = None,
    ) -> None:
        self._set_settings_values(
            {
                "context_budget_mode": settings.mode,
                "context_budget_fixed_total_chars": settings.fixed_total_chars,
                "context_budget_adaptive_fraction": settings.adaptive_fraction,
            },
            save_id=save_id,
        )
        log_event("settings.context_budget_saved")

    def get_chat_history_window_settings(
        self,
        *,
        save_id: str | None = None,
    ) -> ChatHistoryWindowSettings:
        return chat_history_window_settings(self.repositories, save_id=save_id)

    def set_chat_history_window_settings(
        self,
        settings: ChatHistoryWindowSettings,
        *,
        save_id: str | None = None,
    ) -> None:
        self._set_settings_values(
            {
                RECENT_PLAYER_MESSAGE_WINDOW_SETTING: sanitize_recent_message_window(
                    settings.player_messages,
                    default=ChatHistoryWindowSettings.defaults().player_messages,
                ),
                RECENT_NARRATOR_MESSAGE_WINDOW_SETTING: sanitize_recent_message_window(
                    settings.narrator_messages,
                    default=ChatHistoryWindowSettings.defaults().narrator_messages,
                ),
            },
            save_id=save_id,
        )
        log_event("settings.chat_history_window_saved")

    def get_narrator_planner_chat_history_window_settings(
        self,
        *,
        save_id: str | None = None,
    ) -> ChatHistoryWindowSettings:
        return narrator_planner_chat_history_window_settings(
            self.repositories,
            save_id=save_id,
        )

    def set_narrator_planner_chat_history_window_settings(
        self,
        settings: ChatHistoryWindowSettings,
        *,
        save_id: str | None = None,
    ) -> None:
        self._set_settings_values(
            {
                NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING: (
                    sanitize_recent_message_window(
                        settings.player_messages,
                        default=(
                            DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW
                        ),
                    )
                ),
                NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING: (
                    sanitize_recent_message_window(
                        settings.narrator_messages,
                        default=(
                            DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW
                        ),
                    )
                ),
            },
            save_id=save_id,
        )
        log_event("settings.narrator_planner_chat_history_window_saved")

    def secret_storage_warning(self) -> str | None:
        # File fallback is the expected default for web and container deployments.
        return None

    def _set_settings_values(
        self,
        values: Mapping[str, object],
        *,
        save_id: str | None,
    ) -> None:
        for key, value in values.items():
            if save_id is None:
                self.repositories.set_app_setting(key, value)
            else:
                self.repositories.set_scoped_setting(
                    scope="save",
                    scope_id=save_id,
                    key=key,
                    value=value,
                )

    def _save_provider_model(
        self,
        model: ProviderModel,
        *,
        client: ModelListingProvider,
    ) -> None:
        self.repositories.save_provider_model(
            provider=model.provider,
            model_id=model.model_id,
            display_name=model.display_name,
            capabilities=sorted(
                capability.value
                for capability in _runtime_supported_capabilities(
                    model.capabilities,
                    client=client,
                )
            ),
            supported_parameters=sorted(
                parameter.value for parameter in model.supported_parameters
            ),
            context_window=model.context_window,
            pricing=model_pricing_json(model.pricing),
            thinking=model_thinking_json(model.thinking),
        )

    def _record_provider_exception(
        self,
        provider: str,
        has_api_key: bool,
        exc: Exception,
    ) -> ProviderRefreshResult:
        if isinstance(exc, SecretStorageError):
            category = ProviderErrorCategory.SECRET_STORAGE_ERROR
        elif isinstance(exc, ProviderError):
            category = exc.category
        else:
            category = map_exception_to_category(exc)
        error = category.value
        self.repositories.upsert_provider_config(
            provider=provider,
            enabled=True,
            has_api_key=has_api_key,
            last_error=error,
        )
        return ProviderRefreshResult(
            provider=provider,
            configured=True,
            authenticated=False,
            model_count=0,
            error=error,
        )


def model_capabilities_json(model: ProviderModel) -> str:
    return json.dumps(
        sorted(capability.value for capability in model.capabilities),
        separators=(",", ":"),
    )


def model_pricing_json(pricing: ProviderModelPricing | None) -> dict[str, str]:
    if pricing is None:
        return {}
    values = {
        "input_per_million_tokens_usd": pricing.input_per_million_tokens_usd,
        "output_per_million_tokens_usd": pricing.output_per_million_tokens_usd,
        "cache_read_per_million_tokens_usd": pricing.cache_read_per_million_tokens_usd,
        "cache_write_per_million_tokens_usd": (
            pricing.cache_write_per_million_tokens_usd
        ),
        "request_usd": pricing.request_usd,
        "image_usd": pricing.image_usd,
        "note": pricing.note,
    }
    return {key: value for key, value in values.items() if value}


def model_thinking_json(
    thinking: ProviderThinkingLevelSupport | None,
) -> dict[str, object]:
    if thinking is None:
        return {}
    values: dict[str, object | None] = {
        "levels": list(thinking.levels),
        "default_level": thinking.default_level,
        "default_enabled": thinking.default_enabled,
        "mandatory": thinking.mandatory,
        "supports_max_tokens": thinking.supports_max_tokens,
    }
    return {key: value for key, value in values.items() if value is not None}


def provider_catalog_entry_json(entry: ProviderCatalogEntry) -> dict[str, object]:
    return {
        "slug": entry.slug,
        "name": entry.name,
        "privacy_policy_url": entry.privacy_policy_url,
        "terms_of_service_url": entry.terms_of_service_url,
        "status_page_url": entry.status_page_url,
        "headquarters": entry.headquarters,
        "datacenters": list(entry.datacenters),
    }


def _runtime_supported_capabilities(
    capabilities: frozenset[ProviderCapability],
    *,
    client: ModelListingProvider,
) -> frozenset[ProviderCapability]:
    if isinstance(client, VideoProvider):
        return capabilities
    video_capabilities = {
        ProviderCapability.TEXT_TO_VIDEO,
        ProviderCapability.IMAGE_TO_VIDEO,
        ProviderCapability.IMAGE_PLUS_TEXT_TO_VIDEO,
    }
    return capabilities - video_capabilities


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _elapsed_ms(started_at: float) -> int:
    return int((perf_counter() - started_at) * 1000)


def _thinking_support_for_model(
    repositories: PersistenceRepositories,
    *,
    provider: str,
    model_id: str,
) -> dict[str, object] | None:
    for model in repositories.list_provider_models(provider.strip()):
        if model.model_id != model_id.strip() or not model.available:
            continue
        levels = _thinking_support_levels(model.thinking)
        return model.thinking if levels else None
    return None


def _thinking_support_levels(support: Mapping[str, object]) -> tuple[str, ...]:
    levels = support.get("levels")
    if not isinstance(levels, list | tuple):
        return ()
    sanitized: list[str] = []
    for item in levels:
        level = sanitize_thinking_level(item)
        if level is None or level == THINKING_LEVEL_OFF:
            continue
        if level == THINKING_LEVEL_PROVIDER_DEFAULT:
            continue
        sanitized.append(level)
    return tuple(sanitized)


def _default_local_setting(key: str) -> object | None:
    defaults: dict[str, object] = {
        "context_budget_mode": DEFAULT_CONTEXT_BUDGET_MODE,
        "context_budget_fixed_total_chars": DEFAULT_CONTEXT_BUDGET_FIXED_TOTAL_CHARS,
        "context_budget_adaptive_fraction": DEFAULT_CONTEXT_BUDGET_ADAPTIVE_FRACTION,
        RECENT_PLAYER_MESSAGE_WINDOW_SETTING: (
            ChatHistoryWindowSettings.defaults().player_messages
        ),
        RECENT_NARRATOR_MESSAGE_WINDOW_SETTING: (
            ChatHistoryWindowSettings.defaults().narrator_messages
        ),
        NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW_SETTING: (
            DEFAULT_NARRATOR_PLANNER_RECENT_PLAYER_MESSAGE_WINDOW
        ),
        NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW_SETTING: (
            DEFAULT_NARRATOR_PLANNER_RECENT_NARRATOR_MESSAGE_WINDOW
        ),
        PLAN_FIRST_NARRATOR_SETTING: PLAN_FIRST_NARRATOR_DEFAULT,
        DIRECTOR_PRESSURE_ENABLED_SETTING: DIRECTOR_PRESSURE_ENABLED_DEFAULT,
        CHARACTER_ACTION_PLANNING_ENABLED_SETTING: (
            CHARACTER_ACTION_PLANNING_ENABLED_DEFAULT
        ),
        MANUAL_CONFIRMATION_MEMORIES_SETTING: False,
        MANUAL_CONFIRMATION_CHARACTER_REGISTRY_SETTING: False,
        MANUAL_CONFIRMATION_STATE_CHANGES_SETTING: False,
        SCENARIO_EVOLUTION_TURN_INTERVAL_SETTING: (
            DEFAULT_SCENARIO_EVOLUTION_TURN_INTERVAL
        ),
        IMAGE_STYLE_PRESET_SETTING: DEFAULT_IMAGE_STYLE_PRESET,
        CHAT_TEMPERATURE_SETTING: DEFAULT_CHAT_TEMPERATURE,
        CHAT_MAX_OUTPUT_TOKENS_SETTING: DEFAULT_CHAT_MAX_OUTPUT_TOKENS,
        IMAGE_DIMENSION_PRESET_SETTING: DEFAULT_IMAGE_DIMENSION_PRESET,
        OPENROUTER_CHAT_REASONING_OVERRIDES_SETTING: {},
        MODEL_THINKING_PREFERENCES_SETTING: {},
        OPENROUTER_ROUTING_PROFILES_SETTING: default_openrouter_routing_profiles(),
        MODEL_ROUTING_PROFILES_SETTING: default_model_routing_profiles(),
        PENDING_JOBS_DISPLAY_MODE_SETTING: DEFAULT_PENDING_JOBS_DISPLAY_MODE,
        POST_TURN_INFERENCE_MODE_SETTING: POST_TURN_INFERENCE_MODE_DEFAULT,
        SCRIPT_GUARD_MODE_SETTING: DEFAULT_SCRIPT_GUARD_MODE,
        GENERATED_PHRASE_DENYLIST_SETTING: "",
        SAVE_GENERATED_PHRASE_DENYLIST_SETTING: "",
        USER_NARRATION_GUIDANCE_SETTING: DEFAULT_USER_NARRATION_GUIDANCE,
        CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_SETTING: (
            DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_CHANCE_PERCENT
        ),
        CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_SETTING: (
            DEFAULT_CHARACTER_TEXT_PROACTIVE_RANDOM_COOLDOWN_TURNS
        ),
    }
    return defaults.get(key)
