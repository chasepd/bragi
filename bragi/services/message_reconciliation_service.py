"""Reconcile deterministic app data after message edits."""

from __future__ import annotations

from dataclasses import dataclass

from bragi.app_logging import exception_log_fields, log_error_event, log_event
from bragi.persistence.models import MessageRevisionRecord, ModelPreferenceRecord
from bragi.persistence.repositories import PersistenceRepositories
from bragi.providers.contracts import (
    ProviderClient,
    StructuredOutputProvider,
    ToolCallProvider,
)
from bragi.services.context_update_service import (
    ContextRegistrySelection,
    ContextRegistrySelectionRequest,
    ContextUpdateExtractor,
    ContextUpdateService,
    StructuredProviderContextUpdater,
    ToolCallingProviderContextUpdater,
)
from bragi.services.message_correction import MessageCorrectionContext
from bragi.services.model_capabilities import (
    STRUCTURED_OUTPUT_CAPABILITIES,
    TOOL_CALLING_CAPABILITIES,
    model_supports_any_capability,
)
from bragi.services.model_preferences import roleplay_model_preference
from bragi.services.state_service import (
    StateExtractor,
    StateService,
    StructuredProviderStateExtractor,
    ToolCallingProviderStateExtractor,
)


@dataclass(frozen=True)
class MessageReconciliationResult:
    status: str
    state_status: str = "skipped"
    context_status: str = "skipped"
    error: str | None = None


class MessageReconciliationService:
    def __init__(
        self,
        *,
        repositories: PersistenceRepositories,
        providers: dict[str, ProviderClient],
    ) -> None:
        self.repositories = repositories
        self.providers = providers

    async def reconcile_revision(
        self,
        *,
        revision: MessageRevisionRecord,
    ) -> MessageReconciliationResult:
        context = MessageCorrectionContext(
            message_id=revision.message_id,
            previous_body=revision.previous_body,
            new_body=revision.new_body,
            diff_unified=revision.diff_unified,
            message_role=self._message_role(
                save_id=revision.save_id,
                message_id=revision.message_id,
            ),
        )
        state_extractor = self._state_extractor(revision.save_id)
        context_extractor = self._context_extractor(revision.save_id)
        if state_extractor is None and context_extractor is None:
            self.repositories.mark_message_revision_reconciled(
                revision.id,
                status="skipped",
                error="No typed reconciliation provider configured",
            )
            log_event(
                "message_reconciliation.skipped",
                save_id=revision.save_id,
                message_id=revision.message_id,
                revision_id=revision.id,
            )
            return MessageReconciliationResult(status="skipped")

        state_status = "skipped"
        context_status = "skipped"
        try:
            if state_extractor is not None:
                await StateService(
                    repositories=self.repositories,
                    extractor=state_extractor,
                ).extract_and_apply_message_correction(
                    save_id=revision.save_id,
                    source_message_id=revision.message_id,
                    correction_context=context,
                )
                state_status = "succeeded"
            if context_extractor is not None:
                await ContextUpdateService(
                    repositories=self.repositories,
                    extractor=context_extractor,
                    world_data_enricher=None,
                    registry_selector=_FallbackContextRegistrySelector(),
                ).update_after_message_correction(
                    save_id=revision.save_id,
                    source_message_id=revision.message_id,
                    correction_context=context,
                )
                context_status = "succeeded"
        except Exception as exc:
            error = str(exc) or exc.__class__.__name__
            self.repositories.mark_message_revision_reconciled(
                revision.id,
                status="failed",
                error=error,
            )
            log_error_event(
                "message_reconciliation.failed",
                save_id=revision.save_id,
                message_id=revision.message_id,
                revision_id=revision.id,
                **exception_log_fields(exc),
            )
            return MessageReconciliationResult(
                status="failed",
                state_status=state_status,
                context_status=context_status,
                error=error,
            )

        self.repositories.mark_message_revision_reconciled(
            revision.id,
            status="succeeded",
        )
        log_event(
            "message_reconciliation.succeeded",
            save_id=revision.save_id,
            message_id=revision.message_id,
            revision_id=revision.id,
            state_status=state_status,
            context_status=context_status,
        )
        return MessageReconciliationResult(
            status="succeeded",
            state_status=state_status,
            context_status=context_status,
        )

    def _message_role(self, *, save_id: str, message_id: str) -> str:
        for message in self.repositories.list_messages(save_id, include_deleted=True):
            if message.id == message_id:
                return message.role
        return "message"

    def _state_extractor(self, save_id: str) -> StateExtractor | None:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="state_memory",
        )
        if preference is None:
            return None
        provider = self.providers.get(preference.provider)
        if provider is None:
            return None
        if (
            isinstance(provider, ToolCallProvider)
            and _supports_tool_calling(
                self.repositories,
                preference,
            )
        ):
            return ToolCallingProviderStateExtractor(
                provider=provider,
                provider_name=preference.provider,
                model_id=preference.model_id,
                repositories=self.repositories,
                providers=self.providers,
            )
        if (
            isinstance(provider, StructuredOutputProvider)
            and _supports_structured_output(
                self.repositories,
                preference,
            )
        ):
            return StructuredProviderStateExtractor(
                provider=provider,
                provider_name=preference.provider,
                model_id=preference.model_id,
                repositories=self.repositories,
                providers=self.providers,
            )
        return None

    def _context_extractor(self, save_id: str) -> ContextUpdateExtractor | None:
        preference = roleplay_model_preference(
            repositories=self.repositories,
            save_id=save_id,
            purpose="context_update",
        )
        if preference is None:
            return None
        provider = self.providers.get(preference.provider)
        if provider is None:
            return None
        if (
            isinstance(provider, ToolCallProvider)
            and _supports_tool_calling(
                self.repositories,
                preference,
            )
        ):
            return ToolCallingProviderContextUpdater(
                provider=provider,
                provider_name=preference.provider,
                model_id=preference.model_id,
                repositories=self.repositories,
                providers=self.providers,
            )
        if (
            isinstance(provider, StructuredOutputProvider)
            and _supports_structured_output(
                self.repositories,
                preference,
            )
        ):
            return StructuredProviderContextUpdater(
                provider=provider,
                provider_name=preference.provider,
                model_id=preference.model_id,
                repositories=self.repositories,
                providers=self.providers,
            )
        return None


def _supports_tool_calling(
    repositories: PersistenceRepositories,
    preference: ModelPreferenceRecord,
) -> bool:
    return model_supports_any_capability(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=TOOL_CALLING_CAPABILITIES,
    )


class _FallbackContextRegistrySelector:
    async def select_context(
        self,
        _request: ContextRegistrySelectionRequest,
    ) -> ContextRegistrySelection:
        return ContextRegistrySelection()


def _supports_structured_output(
    repositories: PersistenceRepositories,
    preference: ModelPreferenceRecord,
) -> bool:
    return model_supports_any_capability(
        repositories,
        provider=preference.provider,
        model_id=preference.model_id,
        required=STRUCTURED_OUTPUT_CAPABILITIES,
    )
