"""Shared fallback helpers for provider calls."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import cast

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import ModelPreferenceRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ChatRequest,
    ChatResponse,
    ProviderClient,
    StructuredOutputProvider,
    StructuredOutputRequest,
    StructuredOutputResponse,
    ToolCallProvider,
    ToolCallRequest,
)
from bragi.providers.errors import ProviderError, ProviderErrorCategory
from bragi.services.generation_settings import (
    chat_request_with_reasoning_override,
    request_with_model_thinking_preference,
)
from bragi.services.model_capabilities import (
    CHAT_CAPABILITIES,
    MODEL_LACKS_CAPABILITY_REASON,
    MODEL_MISSING_REASON,
    MODEL_UNAVAILABLE_REASON,
    STRUCTURED_OUTPUT_CAPABILITIES,
    TOOL_CALLING_CAPABILITIES,
    check_model_capabilities,
    known_model_is_unavailable,
    model_supports_any_capability,
)
from bragi.services.model_preferences import (
    EXTRACTION_TOOL_FALLBACK_MODEL_ID,
    EXTRACTION_TOOL_FALLBACK_PROVIDER,
    recommended_tool_call_fallback_preference,
    roleplay_model_preference,
)
from bragi.services.openrouter_routing_settings import (
    request_with_openrouter_routing,
)
from bragi.services.provider_diagnostics import (
    record_provider_error,
    record_provider_response,
)
from bragi.services.request_budget import (
    budget_chat_request,
    budget_structured_output_request,
    budget_tool_call_request,
)

CHAT_FALLBACK_ENABLED_SETTING = "chat_fallback_enabled"
STRUCTURED_OUTPUT_FALLBACK_ENABLED_SETTING = (
    "structured_output_fallback_enabled"
)
TOOL_CALL_FALLBACK_ENABLED_SETTING = "tool_call_fallback_enabled"
CHAT_FALLBACK_TASK = "chat_fallback"
STRUCTURED_OUTPUT_FALLBACK_TASK = "structured_output_fallback"
TOOL_CALL_FALLBACK_TASK = "tool_call_fallback"
STRUCTURED_OUTPUT_FALLBACK_PREFERENCE_TASKS = (
    STRUCTURED_OUTPUT_FALLBACK_TASK,
    "full_roleplay_structured_output_fallback",
    "fantasy_roleplay_structured_output_fallback",
    "science_fiction_roleplay_structured_output_fallback",
    "first_contact_exploration_structured_output_fallback",
    "survival_expedition_structured_output_fallback",
    "time_loop_structured_output_fallback",
    "investigation_mystery_structured_output_fallback",
    "heist_infiltration_structured_output_fallback",
    "political_intrigue_structured_output_fallback",
    "dating_sim_structured_output_fallback",
)
TOOL_CALL_FALLBACK_PREFERENCE_TASKS = (
    TOOL_CALL_FALLBACK_TASK,
    "full_roleplay_tool_call_fallback",
    "fantasy_roleplay_tool_call_fallback",
    "science_fiction_roleplay_tool_call_fallback",
    "first_contact_exploration_tool_call_fallback",
    "survival_expedition_tool_call_fallback",
    "time_loop_tool_call_fallback",
    "investigation_mystery_tool_call_fallback",
    "heist_infiltration_tool_call_fallback",
    "political_intrigue_tool_call_fallback",
    "dating_sim_tool_call_fallback",
)

async def chat_with_fallback(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    request: ChatRequest,
    task: str,
    save_id: str | None = None,
    diagnostic_context: dict[str, object] | None = None,
) -> ChatResponse:
    request = request_with_openrouter_routing(
        repositories,
        chat_request_with_reasoning_override(
            repositories,
            request,
            task=task,
            save_id=save_id,
        ),
        task=task,
        save_id=save_id,
    )
    provider = providers[request.provider]
    if known_model_is_unavailable(
        repositories,
        provider=request.provider,
        model_id=request.model_id,
    ):
        raise ValueError(f"Chat model is unavailable: {request.model_id}")
    try:
        request = budget_chat_request(
            repositories,
            request,
            task=task,
        )
        response = await provider.chat(request)
    except ProviderError as exc:
        if not _should_try_fallback_for_error(exc):
            record_provider_error(
                task=task,
                provider=request.provider,
                model_id=request.model_id,
                exc=exc,
                extra=diagnostic_context,
            )
            raise
        fallback = _fallback_chat_request(
            repositories=repositories,
            providers=providers,
            request=request,
            save_id=save_id,
            task=task,
        )
        if fallback is None:
            reason = _chat_fallback_skip_reason(
                repositories=repositories,
                providers=providers,
                save_id=save_id,
            )
            log_event(
                "provider.chat_fallback_skipped",
                provider=request.provider,
                model=request.model_id,
                task=task,
                reason=reason,
            )
            enriched = _with_fallback_skipped_reason(exc, reason)
            record_provider_error(
                task=task,
                provider=request.provider,
                model_id=request.model_id,
                exc=enriched,
                extra=diagnostic_context,
            )
            raise enriched from exc
        record_provider_error(
            task=task,
            provider=request.provider,
            model_id=request.model_id,
            exc=exc,
            extra=diagnostic_context,
        )
        log_event(
            "provider.chat_fallback_started",
            provider=fallback.provider,
            model=fallback.model_id,
            task=task,
        )
        try:
            fallback = budget_chat_request(
                repositories,
                fallback,
                task=task,
            )
            response = await providers[fallback.provider].chat(fallback)
        except ProviderError as fallback_exc:
            enriched = _with_fallback_attempted(
                fallback_exc,
                provider=fallback.provider,
                model_id=fallback.model_id,
            )
            log_error_event(
                "provider.chat_fallback_failed",
                provider=fallback.provider,
                model=fallback.model_id,
                task=task,
                **exception_log_fields(enriched),
            )
            record_provider_error(
                task=task,
                provider=fallback.provider,
                model_id=fallback.model_id,
                exc=enriched,
                extra=diagnostic_context,
            )
            raise enriched from fallback_exc
        record_provider_response(
            task=task,
            provider=response.provider,
            model_id=response.model_id,
            raw_metadata=response.raw_metadata,
            extra=diagnostic_context,
        )
        return response
    record_provider_response(
        task=task,
        provider=response.provider,
        model_id=response.model_id,
        raw_metadata=response.raw_metadata,
        extra=diagnostic_context,
    )
    if response.body.strip():
        return response
    fallback = _fallback_chat_request(
        repositories=repositories,
        providers=providers,
        request=request,
        save_id=save_id,
        task=task,
    )
    if fallback is None:
        reason = _chat_fallback_skip_reason(
            repositories=repositories,
            providers=providers,
            save_id=save_id,
        )
        log_event(
            "provider.chat_fallback_skipped",
            provider=request.provider,
            model=request.model_id,
            task=task,
            reason=reason,
        )
        return response
    log_event(
        "provider.chat_fallback_started",
        provider=fallback.provider,
        model=fallback.model_id,
        task=task,
    )
    try:
        fallback = budget_chat_request(
            repositories,
            fallback,
            task=task,
        )
        response = await providers[fallback.provider].chat(fallback)
    except ProviderError as fallback_exc:
        enriched = _with_fallback_attempted(
            fallback_exc,
            provider=fallback.provider,
            model_id=fallback.model_id,
        )
        log_error_event(
            "provider.chat_fallback_failed",
            provider=fallback.provider,
            model=fallback.model_id,
            task=task,
            **exception_log_fields(enriched),
        )
        record_provider_error(
            task=task,
            provider=fallback.provider,
            model_id=fallback.model_id,
            exc=enriched,
            extra=diagnostic_context,
        )
        raise enriched from fallback_exc
    record_provider_response(
        task=task,
        provider=response.provider,
        model_id=response.model_id,
        raw_metadata=response.raw_metadata,
        extra=diagnostic_context,
    )
    return response


async def structured_output_with_fallback(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    request: StructuredOutputRequest,
    task: str,
    save_id: str | None,
    diagnostic_context: dict[str, object] | None = None,
) -> StructuredOutputResponse:
    request = request_with_openrouter_routing(
        repositories,
        request_with_model_thinking_preference(
            repositories,
            request,
            task=task,
            save_id=save_id,
        ),
        task=task,
        save_id=save_id,
    )
    provider = providers[request.provider]
    structured_diagnostic_context = {
        "schema_name": request.schema_name,
        **(diagnostic_context or {}),
    }
    if not isinstance(provider, StructuredOutputProvider):
        raise ValueError(f"{task} provider does not support structured output")
    model_error = _structured_output_model_requirement_error(
        repositories=repositories,
        provider=request.provider,
        model_id=request.model_id,
    )
    if model_error is not None:
        raise ValueError(model_error)
    try:
        request = budget_structured_output_request(
            repositories,
            request,
            task=task,
        )
        response = await provider.generate_structured_output(request)
    except ProviderError as exc:
        if not _should_try_fallback_for_error(exc):
            record_provider_error(
                task=task,
                provider=request.provider,
                model_id=request.model_id,
                exc=exc,
                extra=structured_diagnostic_context,
            )
            raise
        fallback = _fallback_structured_output_request(
            repositories=repositories,
            providers=providers,
            request=request,
            save_id=save_id,
            task=task,
        )
        if fallback is None:
            reason = _structured_output_fallback_skip_reason(
                repositories=repositories,
                providers=providers,
                save_id=save_id,
            )
            log_event(
                "provider.structured_output_fallback_skipped",
                provider=request.provider,
                model=request.model_id,
                task=task,
                reason=reason,
            )
            enriched = _with_fallback_skipped_reason(exc, reason)
            record_provider_error(
                task=task,
                provider=request.provider,
                model_id=request.model_id,
                exc=enriched,
                extra=structured_diagnostic_context,
            )
            raise enriched from exc
        fallback_provider = providers[fallback.provider]
        if not isinstance(fallback_provider, StructuredOutputProvider):
            reason = "fallback_provider_unavailable"
            log_event(
                "provider.structured_output_fallback_skipped",
                provider=request.provider,
                model=request.model_id,
                task=task,
                reason=reason,
            )
            enriched = _with_fallback_skipped_reason(exc, reason)
            record_provider_error(
                task=task,
                provider=request.provider,
                model_id=request.model_id,
                exc=enriched,
                extra=structured_diagnostic_context,
            )
            raise enriched from exc
        record_provider_error(
            task=task,
            provider=request.provider,
            model_id=request.model_id,
            exc=exc,
            extra=structured_diagnostic_context,
        )
        structured_fallback_provider = cast(
            StructuredOutputProvider,
            fallback_provider,
        )
        log_event(
            "provider.structured_output_fallback_started",
            provider=fallback.provider,
            model=fallback.model_id,
            task=task,
        )
        try:
            fallback = budget_structured_output_request(
                repositories,
                fallback,
                task=task,
            )
            response = await structured_fallback_provider.generate_structured_output(
                fallback
            )
        except ProviderError as fallback_exc:
            enriched = _with_fallback_attempted(
                fallback_exc,
                provider=fallback.provider,
                model_id=fallback.model_id,
            )
            log_error_event(
                "provider.structured_output_fallback_failed",
                provider=fallback.provider,
                model=fallback.model_id,
                task=task,
                **exception_log_fields(enriched),
            )
            record_provider_error(
                task=task,
                provider=fallback.provider,
                model_id=fallback.model_id,
                exc=enriched,
                extra=structured_diagnostic_context,
            )
            raise enriched from fallback_exc
        record_provider_response(
            task=task,
            provider=response.provider,
            model_id=response.model_id,
            raw_metadata=response.raw_metadata,
            extra=structured_diagnostic_context,
        )
        return response
    record_provider_response(
        task=task,
        provider=response.provider,
        model_id=response.model_id,
        raw_metadata=response.raw_metadata,
        extra=structured_diagnostic_context,
    )
    return response


def _fallback_chat_request(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    request: ChatRequest,
    save_id: str | None,
    task: str,
) -> ChatRequest | None:
    preference = _fallback_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=CHAT_FALLBACK_TASK,
    )
    if preference is None or preference.provider not in providers:
        return None
    if not _model_supports(
        repositories=repositories,
        preference=preference,
        required=CHAT_CAPABILITIES,
    ):
        return None
    fallback = chat_request_with_reasoning_override(
        repositories,
        replace(
            request,
            provider=preference.provider,
            model_id=preference.model_id,
            reasoning=None,
            openrouter_provider_routing=None,
        ),
        task=CHAT_FALLBACK_TASK,
        save_id=save_id,
    )
    return request_with_openrouter_routing(
        repositories,
        fallback,
        task=task,
        save_id=save_id,
    )


def _chat_fallback_skip_reason(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    save_id: str | None,
) -> str:
    preference = _fallback_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=CHAT_FALLBACK_TASK,
    )
    if preference is None:
        return "no_fallback_model"
    check = check_model_capabilities(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=CHAT_CAPABILITIES,
    )
    if check.reason == MODEL_UNAVAILABLE_REASON:
        return "fallback_model_unavailable"
    if preference.provider not in providers:
        return "fallback_provider_unavailable"
    if not check.supported:
        return "fallback_model_lacks_required_capabilities"
    return "fallback_provider_unavailable"


def _fallback_structured_output_request(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    request: StructuredOutputRequest,
    save_id: str | None,
    task: str,
) -> StructuredOutputRequest | None:
    preference = _fallback_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=STRUCTURED_OUTPUT_FALLBACK_TASK,
    )
    if preference is None or preference.provider not in providers:
        return None
    provider: object = providers[preference.provider]
    if not isinstance(provider, StructuredOutputProvider):
        return None
    if not _model_supports(
        repositories=repositories,
        preference=preference,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    ):
        return None
    return request_with_openrouter_routing(
        repositories,
        request_with_model_thinking_preference(
            repositories,
            replace(
                request,
                provider=preference.provider,
                model_id=preference.model_id,
                reasoning=None,
                openrouter_provider_routing=None,
            ),
            task=STRUCTURED_OUTPUT_FALLBACK_TASK,
            save_id=save_id,
        ),
        task=task,
        save_id=save_id,
    )


def tool_call_fallback_request(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    request: ToolCallRequest,
    save_id: str | None,
    task: str = TOOL_CALL_FALLBACK_TASK,
) -> ToolCallRequest | None:
    preference = _fallback_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=TOOL_CALL_FALLBACK_TASK,
    )
    if preference is None or preference.provider not in providers:
        return None
    provider: object = providers[preference.provider]
    if not isinstance(provider, ToolCallProvider):
        return None
    if not _model_supports(
        repositories=repositories,
        preference=preference,
        required=TOOL_CALLING_CAPABILITIES,
    ):
        return None
    fallback = request_with_openrouter_routing(
        repositories,
        request_with_model_thinking_preference(
            repositories,
            replace(
                request,
                provider=preference.provider,
                model_id=preference.model_id,
                reasoning=None,
                openrouter_provider_routing=None,
            ),
            task=TOOL_CALL_FALLBACK_TASK,
            save_id=save_id,
        ),
        task=TOOL_CALL_FALLBACK_TASK,
        save_id=save_id,
    )
    return budget_tool_call_request(
        repositories,
        fallback,
        task=task,
    )


def tool_call_fallback_skip_reason(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    save_id: str | None,
) -> str:
    preference = _fallback_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=TOOL_CALL_FALLBACK_TASK,
    )
    if preference is None:
        recommended_check = check_model_capabilities(
            repositories,
            provider=EXTRACTION_TOOL_FALLBACK_PROVIDER,
            model_id=EXTRACTION_TOOL_FALLBACK_MODEL_ID,
            required=TOOL_CALLING_CAPABILITIES,
        )
        if recommended_check.reason == MODEL_UNAVAILABLE_REASON:
            return "fallback_model_unavailable"
        if recommended_check.found and not recommended_check.supported:
            return "fallback_model_lacks_required_capabilities"
        return "no_fallback_model"
    check = check_model_capabilities(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=TOOL_CALLING_CAPABILITIES,
    )
    if check.reason == MODEL_UNAVAILABLE_REASON:
        return "fallback_model_unavailable"
    provider: object | None = providers.get(preference.provider)
    if not isinstance(provider, ToolCallProvider):
        return "fallback_provider_unavailable"
    if not check.supported:
        return "fallback_model_lacks_required_capabilities"
    return "fallback_provider_unavailable"


def structured_output_fallback_request(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    request: StructuredOutputRequest,
    save_id: str | None,
    task: str,
) -> StructuredOutputRequest | None:
    fallback = _fallback_structured_output_request(
        repositories=repositories,
        providers=providers,
        request=request,
        save_id=save_id,
        task=task,
    )
    if fallback is None:
        return None
    return budget_structured_output_request(
        repositories,
        fallback,
        task=task,
    )


def structured_output_fallback_skip_reason(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    save_id: str | None,
) -> str:
    return _structured_output_fallback_skip_reason(
        repositories=repositories,
        providers=providers,
        save_id=save_id,
    )


def _structured_output_fallback_skip_reason(
    *,
    repositories: PersistenceRepositories,
    providers: dict[str, ProviderClient],
    save_id: str | None,
) -> str:
    preference = _fallback_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=STRUCTURED_OUTPUT_FALLBACK_TASK,
    )
    if preference is None:
        return "no_fallback_model"
    check = check_model_capabilities(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    )
    if check.reason == MODEL_UNAVAILABLE_REASON:
        return "fallback_model_unavailable"
    provider: object | None = providers.get(preference.provider)
    if not isinstance(provider, StructuredOutputProvider):
        return "fallback_provider_unavailable"
    if not check.supported:
        return "fallback_model_lacks_required_capabilities"
    return "fallback_provider_unavailable"


def _structured_output_model_requirement_error(
    *,
    repositories: PersistenceRepositories,
    provider: str,
    model_id: str,
) -> str | None:
    check = check_model_capabilities(
        repositories,
        provider=provider,
        model_id=model_id,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    )
    if check.reason == MODEL_MISSING_REASON:
        return (
            "Structured-output model is not in the provider model catalog: "
            f"{model_id}"
        )
    if check.reason == MODEL_UNAVAILABLE_REASON:
        return f"Structured-output model is unavailable: {model_id}"
    if check.reason == MODEL_LACKS_CAPABILITY_REASON:
        return (
            "Structured-output model does not advertise structured output: "
            f"{model_id}"
        )
    return None


def _with_fallback_skipped_reason(
    exc: ProviderError,
    reason: str,
) -> ProviderError:
    return replace(
        exc,
        fallback_attempted=False,
        fallback_skipped_reason=reason,
        fallback_provider=None,
        fallback_model_id=None,
    )


async def recover_tool_call_shape_with_structured_output[T](
    *,
    error: ProviderError,
    task: str,
    provider: str,
    model_id: str,
    structured_run: Callable[[], Awaitable[T]],
) -> T:
    """Recover a failed tool-call request through the structured-output route.

    When the tool-call failure means the configured model cannot serve
    tool-calling requests (HTTP 404 model_not_found from the provider router),
    rerun the same selection through the structured-output route. Other error
    categories keep the caller's existing same-shape semantics: the original
    error is re-raised unchanged. When the structured route also fails, an
    enriched ProviderError records the shape attempt in the fallback fields,
    keeping any more specific fallback identity the structured route already
    reported. Non-ProviderError failures from the structured route are wrapped
    into an enriched ProviderError so callers always see the provider error
    contract.
    """
    if error.category is not ProviderErrorCategory.MODEL_NOT_FOUND:
        raise error
    log_event(
        "provider.tool_call_shape_recovery_started",
        provider=provider,
        model=model_id,
        task=task,
    )
    try:
        result = await structured_run()
    except ProviderError as exc:
        log_error_event(
            "provider.tool_call_shape_recovery_failed",
            provider=provider,
            model=model_id,
            task=task,
            **exception_log_fields(exc),
        )
        raise _with_shape_recovery_failure(
            exc,
            task=task,
            provider=provider,
            model_id=model_id,
        ) from exc
    except (TimeoutError, ValueError, TypeError) as exc:
        log_error_event(
            "provider.tool_call_shape_recovery_failed",
            provider=provider,
            model=model_id,
            task=task,
            **exception_log_fields(exc),
        )
        raise _with_shape_recovery_failure(
            _shape_recovery_wrapped_error(exc),
            task=task,
            provider=provider,
            model_id=model_id,
        ) from exc
    log_event(
        "provider.tool_call_shape_recovery_succeeded",
        provider=provider,
        model=model_id,
        task=task,
    )
    return result


def _shape_recovery_wrapped_error(exc: Exception) -> ProviderError:
    category = (
        ProviderErrorCategory.NETWORK_ERROR
        if isinstance(exc, TimeoutError)
        else ProviderErrorCategory.PROVIDER_ERROR
    )
    return ProviderError(
        category=category,
        message=str(exc) or exc.__class__.__name__,
    )


def _with_shape_recovery_failure(
    exc: ProviderError,
    *,
    task: str,
    provider: str,
    model_id: str,
) -> ProviderError:
    enriched = (
        exc
        if exc.fallback_attempted is True
        else provider_error_with_fallback_attempted(
            exc,
            provider=provider,
            model_id=model_id,
        )
    )
    return replace(
        enriched,
        diagnostics={
            **enriched.diagnostics,
            "shape_recovery": {
                "task": task,
                "provider": provider,
                "model_id": model_id,
                "failed": True,
            },
        },
    )


def provider_error_with_fallback_skipped_reason(
    exc: ProviderError,
    reason: str,
) -> ProviderError:
    return _with_fallback_skipped_reason(exc, reason)


def _with_fallback_attempted(
    exc: ProviderError,
    *,
    provider: str,
    model_id: str,
) -> ProviderError:
    return replace(
        exc,
        fallback_attempted=True,
        fallback_skipped_reason=None,
        fallback_provider=provider,
        fallback_model_id=model_id,
    )


def provider_error_with_fallback_attempted(
    exc: ProviderError,
    *,
    provider: str,
    model_id: str,
) -> ProviderError:
    return _with_fallback_attempted(exc, provider=provider, model_id=model_id)


def _fallback_preference(
    *,
    repositories: PersistenceRepositories,
    save_id: str | None,
    purpose: str,
) -> ModelPreferenceRecord | None:
    default: ModelPreferenceRecord | None = None
    if purpose == TOOL_CALL_FALLBACK_TASK:
        default = recommended_tool_call_fallback_preference(repositories)
    if save_id is None:
        return repositories.get_model_preference(purpose) or default
    return roleplay_model_preference(
        repositories=repositories,
        save_id=save_id,
        purpose=purpose,
    ) or default


def _model_supports(
    *,
    repositories: PersistenceRepositories,
    preference: ModelPreferenceRecord,
    required: frozenset[str],
) -> bool:
    return model_supports_any_capability(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=required,
    )


def _should_try_fallback_for_error(exc: ProviderError) -> bool:
    return exc.category in {
        ProviderErrorCategory.CONTENT_BLOCKED,
        ProviderErrorCategory.CONTEXT_LIMIT_EXCEEDED,
        ProviderErrorCategory.MODEL_NOT_FOUND,
        ProviderErrorCategory.RATE_LIMITED,
        ProviderErrorCategory.NETWORK_ERROR,
        ProviderErrorCategory.PROVIDER_ERROR,
        ProviderErrorCategory.IMAGE_GENERATION_FAILED,
        ProviderErrorCategory.STRUCTURED_OUTPUT_INVALID,
    }
