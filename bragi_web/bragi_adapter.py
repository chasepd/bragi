"""Lazy compatibility boundary for Bragi application internals."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from importlib import import_module
from types import ModuleType
from typing import Any

_INSTALL_HINT = (
    "Run `uv sync` from the Bragi checkout, or reinstall Bragi so the web "
    "package can import the local Bragi core."
)


class BragiCompatibilityError(RuntimeError):
    """Raised when the installed Bragi package is missing or incompatible."""


@dataclass(frozen=True)
class BragiLoggingBindings:
    log_debug_event: Any
    log_error_event: Any
    log_event: Any
    redact_log_value: Any
    redact_text: Any


@dataclass(frozen=True)
class BragiRuntimeBindings:
    configure_logging: Any
    migrate_database: Any
    PersistenceRepositories: Any
    ensure_private_dir: Any
    OpenRouterClient: Any
    VeniceClient: Any
    ProviderClient: Any
    FakeProviderClient: Any
    SystemSecretStore: Any
    SettingsService: Any
    BragiRuntime: Any
    build_settings_model: Any
    record_current_job_step: Any
    record_job_step: Any
    runtime_job_step: Any
    runtime_telemetry_context: Any
    wrap_provider_clients_for_telemetry: Any


@dataclass(frozen=True)
class BragiApiBindings:
    CharacterRegistryEdits: Any
    CharacterKnowledgeAction: Any
    CharacterRegistryRow: Any
    CharacterRegistryService: Any
    MemoryEdit: Any
    ScenarioEdit: Any
    SummaryEdit: Any
    WorldDataCharacterRow: Any
    WorldDataEdits: Any
    WorldDataEntityLinkRow: Any
    WorldDataLocationRow: Any
    WorldDataLossConditionRow: Any
    WorldDataMemoryRow: Any
    WorldDataSceneRow: Any
    WorldDataService: Any
    WorldDataStateRow: Any
    WorldDataSuggestionGroupRow: Any
    WorldDataSuggestionRow: Any
    WorldDataSummaryRow: Any
    WorldDataThreadRow: Any
    ManualScenarioInput: Any


@dataclass(frozen=True)
class BragiSettingsBindings:
    build_settings_model: Any
    build_provider_settings_model: Any
    build_local_settings_model: Any
    configuration_diagnostics: Any


@dataclass(frozen=True)
class BragiDiagnosticsBindings:
    DiagnosticsService: Any
    EngineHealthService: Any
    redact_diagnostic_text: Any


class LazyBragiApiSymbol:
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def __name__(self) -> str:
        return self._name

    def resolve(self) -> Any:
        return getattr(bragi_api_bindings(), self._name)

    def __call__(self, *args: object, **kwargs: object) -> Any:
        return self.resolve()(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.resolve(), name)


def lazy_bragi_api_symbol(name: str) -> LazyBragiApiSymbol:
    return LazyBragiApiSymbol(name)


@cache
def bragi_logging_bindings() -> BragiLoggingBindings:
    logging_module = _module("bragi.app_logging")
    redaction_module = _module("bragi.redaction")
    return BragiLoggingBindings(
        log_debug_event=_required(logging_module, "log_debug_event"),
        log_error_event=_required(logging_module, "log_error_event"),
        log_event=_required(logging_module, "log_event"),
        redact_log_value=_required(redaction_module, "redact_log_value"),
        redact_text=_required(redaction_module, "redact_text"),
    )


@cache
def bragi_runtime_bindings() -> BragiRuntimeBindings:
    app_logging = _module("bragi.app_logging")
    persistence = _module("bragi.persistence")
    repositories = _module("bragi.persistence.repositories")
    private_files = _module("bragi.private_files")
    providers = _module("bragi.providers")
    provider_contracts = _module("bragi.providers.contracts")
    fake_provider = _module("bragi.providers.fake")
    secrets = _module("bragi.services.secrets")
    settings_service = _module("bragi.services.settings_service")
    runtime = _module("bragi.application.controller")
    settings = _module("bragi.application.settings")
    runtime_telemetry = _module("bragi.services.runtime_telemetry")
    return BragiRuntimeBindings(
        configure_logging=_required(app_logging, "configure_logging"),
        migrate_database=_required(persistence, "migrate_database"),
        PersistenceRepositories=_required(repositories, "PersistenceRepositories"),
        ensure_private_dir=_required(private_files, "ensure_private_dir"),
        OpenRouterClient=_required(providers, "OpenRouterClient"),
        VeniceClient=_required(providers, "VeniceClient"),
        ProviderClient=_required(provider_contracts, "ProviderClient"),
        FakeProviderClient=_required(fake_provider, "FakeProviderClient"),
        SystemSecretStore=_required(secrets, "SystemSecretStore"),
        SettingsService=_required(settings_service, "SettingsService"),
        BragiRuntime=_required(runtime, "BragiRuntime"),
        build_settings_model=_required(settings, "build_settings_model"),
        record_current_job_step=_required(
            runtime_telemetry,
            "record_current_job_step",
        ),
        record_job_step=_required(runtime_telemetry, "record_job_step"),
        runtime_job_step=_required(runtime_telemetry, "runtime_job_step"),
        runtime_telemetry_context=_required(
            runtime_telemetry,
            "runtime_telemetry_context",
        ),
        wrap_provider_clients_for_telemetry=_required(
            runtime_telemetry,
            "wrap_provider_clients_for_telemetry",
        ),
    )


@cache
def bragi_api_bindings() -> BragiApiBindings:
    character_registry = _module("bragi.services.character_registry_service")
    world_data = _module("bragi.services.world_data_service")
    runtime = _module("bragi.application.controller")
    return BragiApiBindings(
        CharacterRegistryEdits=_required(
            character_registry,
            "CharacterRegistryEdits",
        ),
        CharacterKnowledgeAction=_required(
            character_registry,
            "CharacterKnowledgeAction",
        ),
        CharacterRegistryRow=_required(character_registry, "CharacterRegistryRow"),
        CharacterRegistryService=_required(
            character_registry,
            "CharacterRegistryService",
        ),
        MemoryEdit=_required(world_data, "MemoryEdit"),
        ScenarioEdit=_required(world_data, "ScenarioEdit"),
        SummaryEdit=_required(world_data, "SummaryEdit"),
        WorldDataCharacterRow=_required(world_data, "WorldDataCharacterRow"),
        WorldDataEdits=_required(world_data, "WorldDataEdits"),
        WorldDataEntityLinkRow=_required(world_data, "WorldDataEntityLinkRow"),
        WorldDataLocationRow=_required(world_data, "WorldDataLocationRow"),
        WorldDataLossConditionRow=_required(world_data, "WorldDataLossConditionRow"),
        WorldDataMemoryRow=_required(world_data, "WorldDataMemoryRow"),
        WorldDataSceneRow=_required(world_data, "WorldDataSceneRow"),
        WorldDataService=_required(world_data, "WorldDataService"),
        WorldDataStateRow=_required(world_data, "WorldDataStateRow"),
        WorldDataSuggestionGroupRow=_required(
            world_data,
            "WorldDataSuggestionGroupRow",
        ),
        WorldDataSuggestionRow=_required(world_data, "WorldDataSuggestionRow"),
        WorldDataSummaryRow=_required(world_data, "WorldDataSummaryRow"),
        WorldDataThreadRow=_required(world_data, "WorldDataThreadRow"),
        ManualScenarioInput=_required(runtime, "ManualScenarioInput"),
    )


@cache
def bragi_settings_bindings() -> BragiSettingsBindings:
    settings = _module("bragi.application.settings")
    return BragiSettingsBindings(
        build_settings_model=_required(settings, "build_settings_model"),
        build_provider_settings_model=_required(
            settings,
            "build_provider_settings_model",
        ),
        build_local_settings_model=_required(
            settings,
            "build_local_settings_model",
        ),
        configuration_diagnostics=_required(settings, "configuration_diagnostics"),
    )


@cache
def bragi_diagnostics_bindings() -> BragiDiagnosticsBindings:
    diagnostics = _module("bragi.services.diagnostics_service")
    engine_health = _module("bragi.services.engine_health_service")
    return BragiDiagnosticsBindings(
        DiagnosticsService=_required(diagnostics, "DiagnosticsService"),
        EngineHealthService=_required(engine_health, "EngineHealthService"),
        redact_diagnostic_text=_required(diagnostics, "redact_diagnostic_text"),
    )


def resolve_lazy_symbol(value: object) -> Any:
    if isinstance(value, LazyBragiApiSymbol):
        return value.resolve()
    return value


def _module(module_name: str) -> ModuleType:
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise BragiCompatibilityError(
            _message(f"could not import {module_name}")
        ) from exc
    if not isinstance(module, ModuleType):
        raise BragiCompatibilityError(_message(f"{module_name} is not a module"))
    return module


def _required(module: ModuleType, name: str) -> Any:
    try:
        return getattr(module, name)
    except AttributeError as exc:
        raise BragiCompatibilityError(
            _message(f"{module.__name__} is missing required symbol {name}")
        ) from exc


def _message(detail: str) -> str:
    return (
        "Bragi Web requires compatible Bragi application modules, but "
        f"{detail}. {_INSTALL_HINT}"
    )
