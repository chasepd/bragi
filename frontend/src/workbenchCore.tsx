import React, { useCallback, useDeferredValue, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider, keepPreviousData, useInfiniteQuery, useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  Archive,
  ArrowDown,
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  ChevronLeft,
  ChevronUp,
  Clock,
  Download,
  Edit3,
  Eye,
  FileWarning,
  Film,
  FileText,
  GitBranch,
  History,
  Image,
  Info,
  Italic,
  KeyRound,
  Loader2,
  LogOut,
  MessageSquare,
  MessageSquareText,
  PanelRight,
  Play,
  Plus,
  Quote,
  RefreshCw,
  RemoveFormatting,
  Save,
  Search,
  Send,
  Settings,
  ShieldCheck,
  Smartphone,
  Square,
  Trash2,
  Upload,
  Users,
  Wand2,
  X
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  api,
  AdminUser,
  ApiError,
  BundlePreview,
  CharacterBundlePreview,
  CharacterEnhanceField,
  CharacterFieldEnhanceResult,
  CharacterKnowledgeApplyResult,
  CharacterKnowledgeTarget,
  CharacterReferenceImage,
  CharacterRow,
  CharacterRegistryModel,
  CharacterRegistryApplyResult,
  CharacterTextAttachment,
  CharacterTextContact,
  CharacterTextMessage,
  CharacterTextsModel,
  CharacterTextThread,
  ChatSubmissionStatus,
  ChatTimingSummary,
  ChatTurnDelta,
  ChatHistoryMessage,
  ChatHistoryModel,
  ChatBundleExportResult,
  ChronicleModel,
  ChronicleMessage,
  deleteJson,
  DiagnosticsModel,
  DiagnosticEntry,
  EngineHealthModel,
  EngineHealthWarning,
  Job,
  JobStepsModel,
  JobStepSummary,
  NarratorDraft,
  isPostTurnCatchupProgress,
  logClientEvent,
  MarkdownBlock,
  MediaAsset,
  MediaModel,
  MaintenanceJobDiagnostic,
  ModelOption,
  OpenRouterRoutingProfile,
  OpenRouterProviderCatalogEntry,
  OpenRouterRoutingSettings as OpenRouterRoutingSettingsModel,
  OpenRouterRoutingTaskOverride,
  postJson,
  ProviderCard,
  PersistentWorld,
  RuntimePerformanceReport,
  RuntimePerformanceRow,
  RuntimeSlowOperation,
  RuntimeModel,
  RuntimeWorldTime,
  SchedulerHealthReport,
  SchedulerHealthTask,
  Scenario,
  ScenarioBundlePreview,
  ScenarioStarterReferenceImage,
  ScenarioDraft,
  SaveEvent,
  ScenePresenceModel,
  SaveListItem,
  ScenarioWizardFlow,
  setUnauthorizedHandler,
  SettingsModel,
  TaskModelSelector,
  ThinkingLevelControl,
  ToggleControl,
  TerminalJobSummary,
  WebEventEntry,
  watchJob,
  watchSave,
  WorldDataApplyResult,
  WorldDataModel,
  WorldDataScenario,
  WorldDataSuggestionGroupRow
} from "./api";

import { BrandLockup, BrandMark } from "./brand";
import { PersistentWorldLibrary } from "./persistentWorldLibrary";
import "./styles.css";

function scheduleClientPaintEvent(
  event: string,
  startedAtMs: number,
  target: Element,
) {
  window.requestAnimationFrame(() => {
    window.requestAnimationFrame(() => {
      if (!target.isConnected) return;
      logClientEvent("info", event, {
        duration_ms: Math.max(0, Math.round(performance.now() - startedAtMs))
      });
    });
  });
}

declare global {
  interface ImportMetaEnv {
    readonly MODE?: string;
  }

  interface ImportMeta {
    readonly env: ImportMetaEnv;
  }
}

const queryClient = new QueryClient();
const LazyScenarioDialog = React.lazy(() => (
  import("./scenarioDialog").then((module) => ({ default: module.ScenarioDialog }))
));
const LazyCharacterTextPhone = React.lazy(() => (
  import("./characterTextPhone").then((module) => ({ default: module.CharacterTextPhone }))
));
const LazyHistoryPanel = React.lazy(() => (
  import("./mediaPanel").then((module) => ({ default: module.HistoryPanel }))
));
const LazyMediaPanel = React.lazy(() => (
  import("./mediaPanel").then((module) => ({ default: module.MediaPanel }))
));
const loadWorldPanel = () => import("./worldPanel");
const loadCharactersPanel = () => import("./charactersPanel");
const loadSettingsPanel = () => import("./settingsPanel");
const LazyWorldPanel = React.lazy(() => (
  loadWorldPanel().then((module) => ({ default: module.WorldPanel }))
));
const LazyCharactersPanel = React.lazy(() => (
  loadCharactersPanel().then((module) => ({ default: module.CharactersPanel }))
));
const LazySettingsPanel = React.lazy(() => (
  loadSettingsPanel().then((module) => ({ default: module.SettingsPanel }))
));
type PanelName = "media" | "history" | "world" | "characters" | "settings";
type WorkbenchRefreshTarget =
  | "runtime"
  | "scenarios"
  | "worlds"
  | "world"
  | "characters"
  | "scene-presence"
  | "chat-history"
  | "character-texts"
  | "character-text-thread"
  | "jobs-active"
  | "chat-status"
  | "chat-timing"
  | "settings"
  | "media";
type QueuedWorkbenchRefresh = {
  saveId: string | null;
  targets: Set<WorkbenchRefreshTarget>;
};
const WORKBENCH_REFRESH_DEBOUNCE_MS = 50;
const RUNTIME_FRESH_SUPPRESS_MS = 5000;
const ALL_WORKBENCH_REFRESH_TARGETS: readonly WorkbenchRefreshTarget[] = [
  "runtime",
  "scenarios",
  "world",
  "characters",
  "scene-presence",
  "chat-history",
  "character-texts",
  "character-text-thread",
  "jobs-active",
  "chat-status"
];
const RUNTIME_MODEL_SIDE_EFFECT_REFRESH_TARGETS: readonly WorkbenchRefreshTarget[] = [
  "world",
  "characters",
  "scene-presence",
  "chat-history",
  "character-texts",
  "character-text-thread",
  "jobs-active",
  "chat-status"
];
const CHAT_TURN_DELTA_REFRESH_TARGETS: readonly WorkbenchRefreshTarget[] = [
  "jobs-active",
  "chat-status",
  "chat-timing",
];
type SettingsTab = "providers" | "openrouter" | "models" | "save" | "local" | "diagnostics" | "users";
type SettingsSummaryTone = "neutral" | "healthy" | "warning" | "attention";
type SettingsSummaryModel = {
  title: string;
  detail: string;
  helper?: string;
  facts?: string[];
  tone?: SettingsSummaryTone;
};
type RunJobOptions = {
  applyResult?: boolean;
  clearPendingMessages?: boolean;
  resumeFromEventCursor?: number;
  recovered?: boolean;
  paintStartedAtMs?: number;
  onSucceeded?: (result: unknown) => void;
  onFailed?: (error: string, job: Job) => void;
  onFinished?: (job: Job) => void;
  allowCrossSaveCompletion?: boolean;
  allowInactiveSave?: boolean;
};
type RunJob = (job: Job, options?: RunJobOptions) => () => void;
type SaveExportState = string | ChatBundleExportResult;
type SaveExportStates = Record<string, SaveExportState>;
type SetSaveExportStates = React.Dispatch<React.SetStateAction<SaveExportStates>>;
type SaveExportReady = { active: boolean; export: ChatBundleExportResult | null };
type SaveExportRecoveryAction = "consume" | "prepare" | "restart";
type ClearSaveExportRecovery = (saveId: string, action: SaveExportRecoveryAction) => void;
type SaveExports = [SaveExportStates, SetSaveExportStates, ClearSaveExportRecovery?];
const SAVE_EXPORT_RECOVERY_WINDOW_MS = 15_000;
const SAVE_EXPORT_RECOVERY_MAX_WINDOW_MS = 10 * 60_000;
function setSaveExportState(setStates: SetSaveExportStates, saveId: string, state?: SaveExportState) {
  setStates((current) => {
    if (state !== undefined) return { ...current, [saveId]: state };
    const next = { ...current };
    delete next[saveId];
    return next;
  });
}
type JobActionRequest = {
  key: string;
  path: string;
  body: unknown;
  fallbackError?: string;
};

function saveEventPayloadRecord(event: SaveEvent): Record<string, unknown> {
  return event.payload && typeof event.payload === "object" && !Array.isArray(event.payload)
    ? event.payload as Record<string, unknown>
    : {};
}

function saveEventReason(event: SaveEvent): string | null {
  const reason = saveEventPayloadRecord(event).reason;
  return typeof reason === "string" && reason.trim() ? reason : null;
}

function refreshTargetForPanel(panel: PanelName): WorkbenchRefreshTarget {
  if (panel === "media") return "media";
  if (panel === "history") return "chat-history";
  if (panel === "world") return "world";
  if (panel === "characters") return "characters";
  return "settings";
}

function runtimeChangedRefreshTargets(reason: string | null, panel: PanelName): WorkbenchRefreshTarget[] {
  const base: WorkbenchRefreshTarget[] = ["runtime"];
  if (!reason) return [...base, refreshTargetForPanel(panel)];
  if ([
    "save_model_preference_updated",
    "save_model_preference_cleared",
    "save_model_thinking_updated",
    "save_model_thinking_cleared",
    "local_setting_updated"
  ].includes(reason)) {
    return [...base, "settings"];
  }
  if ([
    "character_reference_uploaded",
    "character_reference_set",
    "character_reference_removed",
    "media_deleted"
  ].includes(reason)) {
    return [...base, "media", "characters"];
  }
  if ([
    "characters_applied",
    "character_field_enhanced",
    "character_knowledge_applied",
    "character_registry_maintenance",
    "character_bundle_imported"
  ].includes(reason)) {
    return [...base, "characters", "world", "scene-presence"];
  }
  if ([
    "world_data_applied",
    "world_suggestion_review",
    "state_pruning",
    "state_extraction_retry",
    "context_update_retry",
    "character_text_world_update_retry",
    "world_context_retention",
    "memory_consolidation"
  ].includes(reason)) {
    return [...base, "world", "characters", "scene-presence", "chat-history"];
  }
  if (reason === "chat") {
    return [
      ...base,
      "chat-history",
      "world",
      "characters",
      "scene-presence",
      "character-texts",
      "character-text-thread",
      "chat-status",
      "jobs-active"
    ];
  }
  if (["save_created", "save_forked", "save_imported", "save_renamed", "save_deleted"].includes(reason)) {
    return [...base, "scenarios"];
  }
  if (["messages_deleted", "world_time_corrected", "custom_instructions_updated"].includes(reason)) {
    return [...base, "chat-history"];
  }
  const panelTarget = refreshTargetForPanel(panel);
  return panelTarget === "media" ? base : [...base, panelTarget];
}

function saveEventRefreshTargets(event: SaveEvent, panel: PanelName): WorkbenchRefreshTarget[] {
  if (event.type === "runtime_changed") return runtimeChangedRefreshTargets(saveEventReason(event), panel);
  if (event.type === "world_data_changed") return runtimeChangedRefreshTargets(saveEventReason(event), panel);
  if (event.type === "job_changed") return [];
  if (event.type === "character_texts_changed") return ["character-texts", "character-text-thread", "chat-status"];
  if (event.type === "worlds_changed") return ["worlds", "scenarios"];
  if (event.type === "scenarios_changed") return ["scenarios", "worlds"];
  if (event.type === "saves_changed") return ["runtime", "scenarios"];
  if (event.type === "save_deleted") return ["runtime", "scenarios", "jobs-active", "chat-status"];
  return ALL_WORKBENCH_REFRESH_TARGETS.slice();
}

function jobFromSaveEvent(event: SaveEvent): Job | null {
  if (!event.payload || typeof event.payload !== "object") return null;
  const job = (event.payload as { job?: unknown }).job;
  if (!job || typeof job !== "object") return null;
  const record = job as Record<string, unknown>;
  const statuses: Job["status"][] = ["queued", "running", "succeeded", "failed", "cancelled"];
  if (
    typeof record.id !== "string"
    || !record.id
    || typeof record.type !== "string"
    || !record.type
    || typeof record.status !== "string"
    || !statuses.includes(record.status as Job["status"])
  ) return null;
  const saveId = typeof record.save_id === "string" ? record.save_id : event.save_id;
  if (!saveId || (event.save_id && saveId !== event.save_id)) return null;
  const scope = record.scope as Job["scope"];
  const validScope = scope
    && typeof scope.kind === "string"
    && typeof scope.id === "string"
    ? scope
    : null;
  return {
    id: record.id,
    type: record.type,
    save_id: saveId,
    scope: validScope,
    status: record.status as Job["status"],
    completion_level: typeof record.completion_level === "string"
      ? record.completion_level as Job["completion_level"]
      : null,
    result: null,
    error: typeof record.error === "string" ? record.error : null,
    created_at: typeof record.created_at === "number" ? record.created_at : undefined,
    updated_at: typeof record.updated_at === "number" ? record.updated_at : undefined,
    latest_progress: record.latest_progress ?? null,
    event_cursor: typeof record.event_cursor === "number" ? record.event_cursor : undefined,
  };
}

function useJobActionRunner(runJob: RunJob) {
  const pendingKeysRef = useRef(new Set<string>());
  const stateGenerationRef = useRef(0);
  const [pendingJobActionKeys, setPendingJobActionKeys] = useState<Set<string>>(() => new Set());
  const [jobActionErrors, setJobActionErrors] = useState<Record<string, string>>({});

  const clearJobActionState = useCallback(() => {
    stateGenerationRef.current += 1;
    pendingKeysRef.current.clear();
    setPendingJobActionKeys(new Set());
    setJobActionErrors({});
  }, []);

  const startJobAction = useCallback(async (request: JobActionRequest) => {
    if (pendingKeysRef.current.has(request.key)) return;
    const stateGeneration = stateGenerationRef.current;
    const reportError = (error: string) => setJobActionErrors((current) => ({
      ...current,
      [request.key]: error
    }));
    pendingKeysRef.current.add(request.key);
    setPendingJobActionKeys(new Set(pendingKeysRef.current));
    setJobActionErrors((current) => ({ ...current, [request.key]: "" }));
    try {
      const job = await postJson<Job>(request.path, request.body);
      runJob(job, { onFailed: reportError });
    } catch (failure) {
      if (stateGeneration !== stateGenerationRef.current) return;
      reportError(failure instanceof Error ? failure.message : "Background job failed.");
    } finally {
      pendingKeysRef.current.delete(request.key);
      if (stateGeneration === stateGenerationRef.current) {
        setPendingJobActionKeys(new Set(pendingKeysRef.current));
      }
    }
  }, [runJob]);

  return {
    clearJobActionState,
    jobActionErrors,
    pendingJobActionKeys,
    startJobAction
  };
}
type PendingJobsDisplayMode = "compact" | "expanded" | "expanded_full";
type ShellSettingsModel = Pick<SettingsModel, "pending_jobs_display_mode">;
type CurrentUser = { id: string; username: string; role: string; status: string };
type AuthResponse = { user: CurrentUser };
type BootstrapStatus = { admin_exists: boolean; bootstrap_required: boolean; setup_token_required: boolean };
type AuthSessionResponse = { bootstrap: BootstrapStatus; user: CurrentUser | null };
type SessionState =
  | { status: "checking" }
  | { status: "bootstrap"; message?: string; setupTokenRequired: boolean }
  | { status: "login"; message?: string }
  | { status: "authenticated"; user: CurrentUser }
  | { status: "error"; message: string };
export type TrackedJobPhase = { name: string; status: string };
export type TrackedJob = { job: Job; progress: string; phases?: TrackedJobPhase[] };
type PendingChronicleMessage = ChronicleMessage & {
  pending_after_message_id?: string | null;
  pending_save_id?: string | null;
  paint_started_at_ms?: number;
  pending_kind?: "player" | "narrator_placeholder";
  pending_progress?: string;
  pending_draft?: string;
  pending_started_at_ms?: number;
  pending_timing_estimate?: ChatTimingSummary["estimate"];
};
type NarratorPaintMeasurement = {
  jobId: string;
  messageId: string;
  saveId: string;
  startedAtMs: number;
};
type LocalCharacterTextMessage = CharacterTextMessage & {
  local_after_message_id?: string | null;
};
type CharacterTextSendVariables = {
  saveId: string;
  characterId: string | null;
  threadKey: string;
  draftKey: string;
  threadId: string;
  isGroupThread?: boolean;
  body: string;
  photo?: File | null;
  localId: string;
  afterMessageId: string | null;
};
type CharacterTextSpontaneousVariables = {
  saveId: string;
  characterId: string;
};
type WorldDataTab = typeof WORLD_DATA_TABS[number];
type WorldDataEditTab = WorldDataTab;
type CharacterEditorTab = "profile" | "agency" | "knowledge" | "pictures" | "locks";
type WorldDataRow = Record<string, unknown>;
type LookAroundAnswer = {
  query: string;
  answer: string;
  markdownBlocks?: MarkdownBlock[];
  updateCounts?: Record<string, number>;
};
type ScenarioEditorSection = { id: string; key: string; value: string };
type ScenarioEditorStarter = {
  id: string;
  starter_id: string;
  name: string;
  aliases_text: string;
  role: string;
  age: string;
  known_state: string;
  appearance: string;
  visual_notes: string;
  personality: string;
  voice: string;
  texting_style: string;
  goals: string;
  motivations: string;
  boundaries: string;
  relationships_json: string;
  status: string;
  met: boolean;
  locked_fields_text: string;
  reference_image: ScenarioStarterReferenceImage | null;
};
type ScenarioStarterReferencePatch = Pick<
  ScenarioEditorStarter,
  "starter_id" | "appearance" | "visual_notes" | "locked_fields_text" | "reference_image"
>;
type ScenarioStarterTextField =
  | "aliases_text"
  | "role"
  | "age"
  | "status"
  | "locked_fields_text"
  | "known_state"
  | "appearance"
  | "visual_notes"
  | "personality"
  | "voice"
  | "texting_style"
  | "goals"
  | "motivations"
  | "boundaries"
  | "relationships_json";
const STARTER_INPUT_FIELDS = [
  ["aliases_text", "Aliases", "aliases"],
  ["role", "Role", "role"],
  ["age", "Age", "age"],
  ["status", "Status", "status"],
  ["locked_fields_text", "Locked Fields", "locked fields"]
] as const satisfies readonly (readonly [ScenarioStarterTextField, string, string])[];
const STARTER_TEXTAREA_FIELDS = [
  ["known_state", "History", "history"],
  ["appearance", "Appearance", "appearance"],
  ["visual_notes", "Visual Notes", "visual notes"],
  ["personality", "Personality", "personality"],
  ["voice", "Voice", "voice"],
  ["texting_style", "Texting Style", "texting style"],
  ["goals", "Goals", "goals"],
  ["motivations", "Motivations", "motivations"],
  ["boundaries", "Boundaries", "boundaries"],
  ["relationships_json", "Relationships", "relationships"]
] as const satisfies readonly (readonly [ScenarioStarterTextField, string, string])[];
type ScenarioEditorCore = {
  interaction_mode: "roleplay" | "storyteller";
  title: string;
  premise: string;
  player_character_name: string;
  player_role: string;
};
type ScenarioEditorDraftState = {
  core: ScenarioEditorCore;
  sections: ScenarioEditorSection[];
  starters: ScenarioEditorStarter[];
};
type ScenarioEditorValue = {
  scenario_id?: string;
  scenario_type: string;
  interaction_mode: "roleplay" | "storyteller";
  title: string;
  premise: string;
  player_character_name: string;
  player_role: string;
  content_sections: ScenarioEditorSection[];
  character_starters: ScenarioEditorStarter[];
};
type ScenarioEditPayload = {
  interaction_mode: "roleplay" | "storyteller";
  title: string;
  premise: string;
  player_character_name: string;
  player_role: string;
  content_sections: [string, string][];
  character_starters: {
    starter_id?: string;
    name: string;
    aliases: string[];
    role: string;
    age: string;
    known_state: string;
    appearance: string;
    visual_notes: string;
    personality: string;
    voice: string;
    texting_style: string;
    goals: string;
    motivations: string;
    boundaries: string;
    relationships: Record<string, unknown>;
    status: string;
    met: boolean;
    locked_fields: string[];
    reference_image?: ScenarioStarterReferenceImage | null;
  }[];
};
type ScenarioSectionGroup = { label: string; section_ids: string[] };
type ModelCapabilityFamily = "chat" | "structured_output" | "tool_calling" | "image_generation" | "image_to_image" | "text_to_video" | "image_to_video" | "vision";
type ModelRoutingLaneId =
  "narrator"
  | "narrator_planner"
  | "narrator_verifier"
  | "content_safety"
  | "character_agents"
  | "action_choices"
  | "director_pressure"
  | "context_selector"
  | "observation_memory"
  | "world_updates"
  | "character_registry_maintenance"
  | "context_cleanup"
  | "state_pruning"
  | "scenario_evolution"
  | "scenario_writer"
  | "summarization"
  | "image_prompt"
  | "character_image_description"
  | "image_generation"
  | "image_edit"
  | "video_generation"
  | "image_animation"
  | "narrator_fallback"
  | "text_fallback"
  | "structured_fallback"
  | "tool_fallback"
  | "image_fallback"
  | "image_edit_fallback"
  | "video_fallback";
type ResizeSide = "left" | "right";
type MobileSheetName = "library" | PanelName;
type WorkbenchLayout = { leftRailWidth: number; rightPanelWidth: number };
type WorkbenchLayoutStyle = React.CSSProperties & {
  "--left-rail-width": string;
  "--right-panel-width": string;
};
type LibraryTab = "saves" | "scenarios" | "worlds";
type SortDirection = "asc" | "desc";
type SaveSortKey = "last_opened" | "title" | "created" | "updated" | "scenario_title";
type ScenarioSortKey = "updated" | "title" | "created" | "save_count" | "type";
type ScenarioUsageFilter = "all" | "used" | "unused";
type LibraryControlsState = {
  activeTab: LibraryTab;
  saveQuery: string;
  saveSort: SaveSortKey;
  saveDirection: SortDirection;
  scenarioQuery: string;
  scenarioSort: ScenarioSortKey;
  scenarioDirection: SortDirection;
  scenarioType: string;
  scenarioUsage: ScenarioUsageFilter;
};
type ScopedLibraryControlsState = {
  userId: string | null;
  state: LibraryControlsState;
};
type ModelRoutingLaneMeta = {
  id: ModelRoutingLaneId;
  label: string;
  title: string;
  capabilities: readonly ModelCapabilityFamily[];
  targetPurposes: readonly string[];
  icon: LucideIcon;
};
type ModelRoutingLane = ModelRoutingLaneMeta & {
  selectors: TaskModelSelector[];
  options: ModelOption[];
};
type ModelRoutingLaneGroupMeta = {
  label: string;
  lanes: readonly ModelRoutingLaneMeta[];
};
type ModelRoutingLaneGroup = {
  label: string;
  lanes: ModelRoutingLane[];
};
type ModelSelectorGroup = {
  label: string;
  selectors: TaskModelSelector[];
};
const THINKING_LEVEL_PROVIDER_DEFAULT = "provider_default";
const THINKING_LEVEL_OFF = "off";

function canUseChildRestrictedControls(currentUser?: CurrentUser | null) {
  return currentUser?.role !== "child";
}

function canUseAdminControls(currentUser?: CurrentUser | null) {
  return !currentUser || currentUser.role === "admin";
}

function useDialogJobWatcher() {
  const watcherRef = useRef<{ token: number; stop: () => void } | null>(null);
  const tokenRef = useRef(0);
  const stopCurrent = useCallback(() => {
    tokenRef.current += 1;
    watcherRef.current?.stop();
    watcherRef.current = null;
  }, []);
  const watchDialogJob = useCallback((
    jobId: string,
    onUpdate: (job: Job) => void,
    onEvent?: (name: string, data: unknown) => void,
    saveId?: string | null
  ) => {
    stopCurrent();
    const token = tokenRef.current;
    let active = true;
    let stopWatcher: () => void = () => undefined;
    const stop = () => {
      if (!active) return;
      active = false;
      stopWatcher();
      if (watcherRef.current?.token === token) watcherRef.current = null;
    };
    stopWatcher = watchJob(
      jobId,
      (job) => {
        if (!active || tokenRef.current !== token) return;
        if (["succeeded", "failed", "cancelled"].includes(job.status)) {
          active = false;
          if (watcherRef.current?.token === token) watcherRef.current = null;
        }
        onUpdate(job);
      },
      onEvent
        ? (name, data) => {
            if (active && tokenRef.current === token) onEvent(name, data);
          }
        : undefined,
      saveId
    );
    watcherRef.current = { token, stop };
    return stop;
  }, [stopCurrent]);
  useEffect(() => () => stopCurrent(), [stopCurrent]);
  return watchDialogJob;
}

type DialogFocusElement = HTMLDivElement | HTMLFormElement | HTMLElement;

function useDialogFocus<T extends DialogFocusElement>(dialogRef: React.RefObject<T>, onClose: () => void) {
  const closeRef = useRef(onClose);
  const previousFocusRef = useRef<HTMLElement | null>(
    typeof document !== "undefined" && document.activeElement instanceof HTMLElement ? document.activeElement : null
  );
  useEffect(() => {
    closeRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const dialog = dialogRef.current;
    const focusTarget = dialog ? firstFocusableElement(dialog) ?? dialog : null;
    focusTarget?.focus();
    return () => {
      const previous = previousFocusRef.current;
      if (previous?.isConnected) previous.focus();
    };
  }, [dialogRef]);

  return useCallback((event: React.KeyboardEvent<T>) => {
    if (event.defaultPrevented) return;
    const dialog = dialogRef.current;
    if (!dialog) return;
    const target = event.target;
    if (target instanceof Node && !dialog.contains(target)) return;
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      closeRef.current();
      return;
    }
    if (event.key !== "Tab") return;
    const focusable = focusableElements(dialog);
    if (!focusable.length) {
      event.preventDefault();
      dialog.focus();
      return;
    }
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    const active = document.activeElement;
    if (event.shiftKey) {
      if (active === first || !dialog.contains(active)) {
        event.preventDefault();
        last.focus();
      }
      return;
    }
    if (active === last || !dialog.contains(active)) {
      event.preventDefault();
      first.focus();
    }
  }, [dialogRef]);
}

function firstFocusableElement(container: HTMLElement): HTMLElement | null {
  const autofocus = container.querySelector<HTMLElement>("[autofocus]");
  if (autofocus && isFocusableElement(autofocus)) return autofocus;
  return focusableElements(container)[0] ?? null;
}

function focusableElements(container: HTMLElement): HTMLElement[] {
  return Array.from(container.querySelectorAll<HTMLElement>(
    [
      "button:not([disabled])",
      "input:not([disabled])",
      "select:not([disabled])",
      "textarea:not([disabled])",
      "a[href]",
      "[tabindex]:not([tabindex='-1'])"
    ].join(",")
  )).filter(isFocusableElement);
}

function isFocusableElement(element: HTMLElement): boolean {
  return !element.matches("[disabled], [aria-hidden='true']");
}

function ModalBackdrop({ children }: { children: React.ReactNode }) {
  const content = <div className="modal-backdrop">{children}</div>;
  return typeof document === "undefined" ? content : createPortal(content, document.body);
}

function DialogPanel({
  className,
  titleId,
  label,
  onClose,
  children
}: {
  className: string;
  titleId?: string;
  label?: string;
  onClose: () => void;
  children: React.ReactNode;
}) {
  const dialogRef = useRef<HTMLDivElement | null>(null);
  const onKeyDown = useDialogFocus(dialogRef, onClose);
  return (
    <div
      ref={dialogRef}
      className={className}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-label={titleId ? undefined : label}
      tabIndex={-1}
      onKeyDown={onKeyDown}
    >
      {children}
    </div>
  );
}

function DialogForm({
  className,
  titleId,
  label,
  onClose,
  onSubmit,
  children
}: {
  className: string;
  titleId?: string;
  label?: string;
  onClose: () => void;
  onSubmit: React.FormEventHandler<HTMLFormElement>;
  children: React.ReactNode;
}) {
  const dialogRef = useRef<HTMLFormElement | null>(null);
  const onKeyDown = useDialogFocus(dialogRef, onClose);
  return (
    <form
      ref={dialogRef}
      className={className}
      role="dialog"
      aria-modal="true"
      aria-labelledby={titleId}
      aria-label={titleId ? undefined : label}
      tabIndex={-1}
      onKeyDown={onKeyDown}
      onSubmit={onSubmit}
    >
      {children}
    </form>
  );
}

function InlineNotice({ children, className = "", polite = false }: { children: React.ReactNode; className?: string; polite?: boolean }) {
  return (
    <p className={["settings-warning", className].filter(Boolean).join(" ")} role={polite ? undefined : "alert"} aria-live={polite ? "polite" : undefined}>
      {children}
    </p>
  );
}

function ChatSubmissionStatusNotice({
  message,
  retrying,
  onRetry
}: {
  message: string;
  retrying: boolean;
  onRetry: () => void;
}) {
  return (
    <InlineNotice className="composer-error submission-status-notice">
      <span>{message} Chat actions are disabled until status refreshes.</span>
      <button
        type="button"
        className="secondary-command compact"
        aria-label="Retry chat status"
        disabled={retrying}
        onClick={onRetry}
      >
        {retrying ? <Loader2 className="spin" size={14} aria-hidden="true" /> : <RefreshCw size={14} aria-hidden="true" />}
        <span>{retrying ? "Retrying" : "Retry"}</span>
      </button>
    </InlineNotice>
  );
}

function EditorDirtyStatus({ dirty, canDiscard, onDiscard }: { dirty: boolean; canDiscard: boolean; onDiscard: () => void }) {
  if (!dirty) return null;
  return (
    <div className="editor-dirty-row">
      <span role="status" aria-live="polite">Unsaved changes</span>
      {canDiscard ? (
        <button type="button" className="secondary-command compact" onClick={onDiscard}>
          Discard changes
        </button>
      ) : null}
    </div>
  );
}

type SegmentOption<T extends string> = {
  value: T;
  label: string;
  title?: string;
  disabled?: boolean;
  tabId?: string;
  panelId?: string;
};

function SegmentedTabs<T extends string>({
  options,
  value,
  onChange,
  label,
  className = "segmented",
  renderOption
}: {
  options: SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  label: string;
  className?: string;
  renderOption?: (option: SegmentOption<T>) => React.ReactNode;
}) {
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const selectedIndex = options.findIndex((option) => option.value === value);
  const focusableIndexes = options.reduce<number[]>((indexes, option, index) => (
    option.disabled ? indexes : [...indexes, index]
  ), []);
  const selectedTabbable = selectedIndex >= 0 && !options[selectedIndex]?.disabled ? selectedIndex : null;
  const tabbableIndex = selectedTabbable ?? focusableIndexes[0] ?? -1;
  const focusTab = useCallback((index: number) => {
    const option = options[index];
    if (!option || option.disabled) return;
    onChange(option.value);
    tabRefs.current[index]?.focus();
  }, [onChange, options]);
  const onTabKeyDown = useCallback((index: number, event: React.KeyboardEvent<HTMLButtonElement>) => {
    if (event.defaultPrevented || !focusableIndexes.length) return;
    let nextIndex: number | null = null;
    if (event.key === "Home") {
      nextIndex = focusableIndexes[0];
    } else if (event.key === "End") {
      nextIndex = focusableIndexes[focusableIndexes.length - 1];
    } else if (event.key === "ArrowRight" || event.key === "ArrowDown" || event.key === "ArrowLeft" || event.key === "ArrowUp") {
      const currentPosition = focusableIndexes.includes(index)
        ? focusableIndexes.indexOf(index)
        : Math.max(0, focusableIndexes.indexOf(tabbableIndex));
      const direction = event.key === "ArrowRight" || event.key === "ArrowDown" ? 1 : -1;
      nextIndex = focusableIndexes[(currentPosition + direction + focusableIndexes.length) % focusableIndexes.length];
    }
    if (nextIndex === null) return;
    event.preventDefault();
    focusTab(nextIndex);
  }, [focusTab, focusableIndexes, tabbableIndex]);
  return (
    <div className={className} role="tablist" aria-label={label}>
      {options.map((option, index) => (
        <button
          ref={(element) => {
            tabRefs.current[index] = element;
          }}
          type="button"
          role="tab"
          key={option.value}
          id={option.tabId}
          aria-selected={value === option.value}
          aria-controls={option.panelId}
          className={value === option.value ? "active" : ""}
          title={option.title}
          disabled={option.disabled}
          tabIndex={index === tabbableIndex ? 0 : -1}
          onClick={() => onChange(option.value)}
          onKeyDown={(event) => onTabKeyDown(index, event)}
        >
          {renderOption ? renderOption(option) : option.label}
        </button>
      ))}
    </div>
  );
}

const WORLD_DATA_TABS = ["scenario", "scene", "world_state", "memories", "context_inputs", "summaries", "locations", "characters", "threads", "links", "suggestion_groups", "audit"] as const;
const READONLY_WORLD_TABS = new Set<WorldDataTab>(["context_inputs", "audit"]);
const WORLD_DATA_PAGE_SIZE = 80;
type WorldDataTabGroupId = "review" | "scene" | "knowledge" | "people" | "advanced";
type WorldDataTabGroup = {
  id: WorldDataTabGroupId;
  label: string;
  description: string;
  tabs: readonly WorldDataTab[];
};
const WORLD_DATA_TAB_GROUPS: readonly WorldDataTabGroup[] = [
  {
    id: "review",
    label: "Review",
    description: "Pending suggestions that need approval before they reach the world.",
    tabs: ["suggestion_groups"]
  },
  {
    id: "scene",
    label: "Scene",
    description: "The current scene and the context Bragi used for the latest narrator turn.",
    tabs: ["scene", "context_inputs"]
  },
  {
    id: "knowledge",
    label: "Knowledge",
    description: "Facts, memories, summaries, and links that ground the chronicle.",
    tabs: ["world_state", "memories", "summaries", "links"]
  },
  {
    id: "people",
    label: "People & Places",
    description: "Characters, locations, and the threads that connect them.",
    tabs: ["characters", "locations", "threads"]
  },
  {
    id: "advanced",
    label: "Advanced",
    description: "Scenario definition and the audit trail of world-data edits.",
    tabs: ["scenario", "audit"]
  }
];
const WORLD_DATA_TAB_GROUP_BY_TAB: ReadonlyMap<WorldDataTab, WorldDataTabGroup> = (() => {
  const map = new Map<WorldDataTab, WorldDataTabGroup>();
  for (const group of WORLD_DATA_TAB_GROUPS) {
    for (const tab of group.tabs) map.set(tab, group);
  }
  return map;
})();
const FIRST_TAB_BY_GROUP: ReadonlyMap<WorldDataTabGroupId, WorldDataTab> = (() => {
  const map = new Map<WorldDataTabGroupId, WorldDataTab>();
  for (const group of WORLD_DATA_TAB_GROUPS) {
    const first = group.tabs[0];
    if (first) map.set(group.id, first);
  }
  return map;
})();
function worldDataTabGroup(tab: WorldDataTab): WorldDataTabGroup {
  return WORLD_DATA_TAB_GROUP_BY_TAB.get(tab) ?? WORLD_DATA_TAB_GROUPS[WORLD_DATA_TAB_GROUPS.length - 1];
}
function pendingSuggestionGroupCount(model: WorldDataModel | undefined): number {
  return worldSuggestionGroupRows(worldTabValue(model, "suggestion_groups"))
    .filter((row) => row.status === "pending")
    .length;
}
const TEXTAREA_FIELD_NAMES = new Set(["body", "description", "visual_description", "known_state", "appearance", "personality", "voice", "texting_style", "private_notes", "goals", "motivations", "current_intent", "boundaries", "attitude_toward_player", "cooperation_conditions", "premise", "situation", "objective", "reason", "explanation", "choice_style", "opening_message"]);
const CHARACTER_LOCK_FIELDS = [
  ["name", "Name"],
  ["aliases", "Aliases"],
  ["role", "Role"],
  ["age", "Age"],
  ["known_state", "History"],
  ["met", "Met"],
  ["appearance", "Appearance"],
  ["visual_notes", "Visual notes"],
  ["current_clothing", "Current clothing"],
  ["personality", "Personality"],
  ["voice", "Voice"],
  ["texting_style", "Texting style"],
  ["relationships", "Relationships"],
  ["goals", "Goals"],
  ["motivations", "Motivations"],
  ["current_intent", "Current intent"],
  ["boundaries", "Boundaries"],
  ["attitude_toward_player", "Attitude toward player"],
  ["cooperation_conditions", "Cooperation conditions"],
  ["status", "Status"],
  ["location_id", "Location"],
  ["private_notes", "Private notes"],
  ["present", "Present"]
] as const;
const CHARACTER_LOCK_FIELD_IDS: ReadonlySet<string> = new Set(CHARACTER_LOCK_FIELDS.map(([id]) => id));
export function mergeReferenceUploadLocks(server: string[], baseline: string[], current: string[]): string[] {
  const merged = new Set(server);
  for (const field of CHARACTER_LOCK_FIELD_IDS) {
    if (field === "appearance" || field === "visual_notes") merged.add(field);
    else if (baseline.includes(field) !== current.includes(field)) {
      if (current.includes(field)) merged.add(field);
      else merged.delete(field);
    }
  }
  return [
    ...CHARACTER_LOCK_FIELDS.map(([id]) => id).filter((id) => merged.has(id)),
    ...server.filter((field) => !CHARACTER_LOCK_FIELD_IDS.has(field))
  ];
}
const CHARACTER_LOCK_FIELD_ALIASES: Record<string, string> = {
  aliases_text: "aliases",
  relationships_json: "relationships"
};
const CHARACTER_AUTO_ENHANCE_FIELDS: readonly CharacterEnhanceField[] = [
  "known_state",
  "appearance",
  "visual_notes",
  "personality",
  "voice",
  "texting_style",
  "goals",
  "motivations",
  "current_intent",
  "boundaries",
  "attitude_toward_player",
  "cooperation_conditions",
  "status",
  "relationships"
];
const CHARACTER_AUTO_ENHANCE_FIELD_SET: ReadonlySet<string> = new Set(CHARACTER_AUTO_ENHANCE_FIELDS);
const CHARACTER_AUTO_ENHANCE_LABELS: Record<CharacterEnhanceField, string> = {
  known_state: "History",
  appearance: "Appearance",
  visual_notes: "Visual notes",
  personality: "Personality",
  voice: "Voice",
  texting_style: "Texting Style",
  goals: "Goals",
  motivations: "Motivations",
  current_intent: "Current Intent",
  boundaries: "Boundaries",
  attitude_toward_player: "Attitude Toward Player",
  cooperation_conditions: "Cooperation Conditions",
  status: "Status",
  relationships: "Relationships"
};
const SCENE_LOCK_FIELDS = [
  ["current_location_id", "Loc"],
  ["situation", "Scene"],
  ["objective", "Goal"],
  ["in_world_time", "Time"],
  ["time_of_day", "Old TOD"],
  ["day_of_week", "Old DOW"],
  ["world_day_index", "Old #"],
  ["world_time_day_label", "Day"],
  ["world_time_day_index", "#"],
  ["world_time_phase", "Phase"],
  ["world_time_clock_minutes", "Clock"],
  ["world_time_period_label", "Era"],
  ["weather", "WX"],
  ["mood", "Mood"],
  ["nearby_objects_text", "Props"],
  ["hazards_text", "Risk"],
  ["present_character_ids_text", "Cast"]
] as const;
const WORLD_TIME_OF_DAY_OPTIONS = [
  "morning",
  "late_morning",
  "afternoon",
  "evening",
  "night"
] as const;
const LOCATION_LOCK_FIELDS = [
  ["name", "Name"],
  ["aliases_text", "Aliases"],
  ["description", "Description"],
  ["visual_description", "Visual description"],
  ["parent_location_id", "Parent location"],
  ["connections_text", "Connections"],
  ["status", "Status"],
  ["hazards_text", "Hazards"]
] as const;
const THREAD_LOCK_FIELDS = [
  ["title", "Title"],
  ["description", "Description"],
  ["status", "Status"],
  ["priority", "Priority"],
  ["visibility", "Visibility"],
  ["related_entities_text", "Related entities"]
] as const;

function openDownloadInNewTab(url: string) {
  window.open(url, "_blank", "noopener,noreferrer");
}

function isChatBundleExportResult(result: unknown): result is ChatBundleExportResult {
  if (!result || typeof result !== "object") return false;
  const candidate = result as Partial<ChatBundleExportResult>;
  return (
    candidate.kind === "chat_bundle_export"
    && typeof candidate.download_url === "string"
    && typeof candidate.filename === "string"
  );
}

const SCENARIO_CORE_SECTION_IDS = new Set(["title", "premise", "setup_line", "starting_scene", "player_character_name", "player_role", "character_starters"]);
const FULL_ROLEPLAY_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "World", section_ids: ["worldbuilding", "lore", "locations", "factions"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["current_scene"] }
];
const FANTASY_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Fantasy World", section_ids: ["magic_system", "realms_and_places", "factions_and_orders", "myths_and_creatures", "quest_stakes"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["current_scene"] }
];
const SCIENCE_FICTION_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Science Fiction World", section_ids: ["technology_level", "setting_scope", "species_and_intelligences", "factions_and_institutions", "mission_stakes"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["current_scene"] }
];
const FIRST_CONTACT_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Mission", section_ids: ["mission_profile", "ship_or_base_status"] },
  { label: "Discovery", section_ids: ["exploration_target", "knowledge_state", "discoveries_and_samples", "hazards_and_escalation"] },
  { label: "Contact", section_ids: ["unknown_intelligence", "translation_progress"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["current_scene"] }
];
const SURVIVAL_EXPEDITION_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Expedition", section_ids: ["expedition_goal", "route_options", "travel_progress"] },
  { label: "Supplies", section_ids: ["resource_inventory"] },
  { label: "Conditions", section_ids: ["environmental_conditions", "hazards_and_events", "camp_status"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["worldbuilding", "lore", "locations", "factions", "current_scene"] }
];
const TIME_LOOP_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Loop Rules", section_ids: ["loop_premise", "reset_trigger", "loop_duration", "objective", "failure_conditions"] },
  { label: "Reset State", section_ids: ["starting_state", "baseline_world_state"] },
  { label: "Schedule", section_ids: ["loop_schedule", "current_loop_state"] },
  { label: "Persistence", section_ids: ["persistent_knowledge", "persistence_exceptions", "npc_memory_rules"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["worldbuilding", "lore", "locations", "factions", "current_scene"] }
];
const INVESTIGATION_MYSTERY_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Case", section_ids: ["case_facts", "case_status"] },
  { label: "Evidence", section_ids: ["clues", "timeline", "red_herrings", "hidden_truth"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["locations", "factions", "current_scene"] }
];
const HEIST_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Target & Objectives", section_ids: ["target_location", "objectives_and_stakes"] },
  { label: "Intel", section_ids: ["intel_and_access"] },
  { label: "Security", section_ids: ["security_model", "alert_and_heat"] },
  { label: "Tools & Complications", section_ids: ["loadout_and_tools", "complications"] },
  { label: "Exit & Consequences", section_ids: ["extraction_routes", "aftermath"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["locations", "factions", "current_scene"] }
];
const POLITICAL_INTRIGUE_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Arena", section_ids: ["political_arena", "central_conflict"] },
  { label: "Factions", section_ids: ["political_factions", "alliances_and_rivalries"] },
  { label: "Leverage", section_ids: ["secrets_and_leverage", "reputation_and_standing", "obligations_and_favors", "public_private_knowledge"] },
  { label: "Pressure", section_ids: ["event_calendar", "political_pressure"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["worldbuilding", "lore", "locations", "factions", "current_scene"] }
];
const SETTLEMENT_BUILDER_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Community", section_ids: ["settlement_profile"] },
  { label: "Operations", section_ids: ["resources_and_indicators", "projects_and_facilities"] },
  { label: "Pressure", section_ids: ["threats_and_opportunities", "calendar_and_deadlines"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["worldbuilding", "lore", "locations", "factions", "current_scene"] }
];
const MONSTER_HUNT_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Hunt", section_ids: ["hunt_profile", "target_profile", "hunt_status"] },
  { label: "Investigation", section_ids: ["leads_and_clues", "hunt_locations"] },
  { label: "Pressure", section_ids: ["preparation_state"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["locations", "factions", "current_scene"] }
];
const ROAD_TRIP_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Journey", section_ids: ["journey_profile", "route_and_stops", "journey_progress"] },
  { label: "Relationship Threads", section_ids: ["relationship_threads"] },
  { label: "Road Pressure", section_ids: ["transport_and_supplies", "recurring_pressures"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["worldbuilding", "lore", "locations", "factions", "current_scene"] }
];
const MERCHANT_TRADE_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Trade Route", section_ids: ["trade_profile", "markets_and_stops"] },
  { label: "Cargo & Contracts", section_ids: ["cargo_inventory", "contracts_and_debts"] },
  { label: "Risk & Standing", section_ids: ["route_hazards", "profit_and_loss"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "Continuity", section_ids: ["worldbuilding", "lore", "locations", "factions", "current_scene"] }
];
const DATING_SIM_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Player", section_ids: ["player_character_profile"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "World", section_ids: ["worldbuilding", "lore", "locations", "factions", "current_scene"] }
];
const CYOA_SCENARIO_SECTION_GROUPS: ScenarioSectionGroup[] = [
  { label: "Choices", section_ids: ["choice_style"] },
  { label: "Opening", section_ids: ["tone_genre", "opening_message"] },
  { label: "World", section_ids: ["worldbuilding", "lore", "locations", "factions", "current_scene"] }
];
const SCENARIO_TYPE_LABELS: Record<string, string> = {
  full_roleplay: "Generic Roleplay",
  fantasy_roleplay: "Fantasy",
  science_fiction_roleplay: "Science Fiction",
  first_contact_exploration: "First Contact / Exploration",
  survival_expedition: "Survival Expedition",
  time_loop: "Time Loop",
  investigation_mystery: "Investigation Mystery",
  heist_infiltration: "Heist / Infiltration",
  political_intrigue: "Political Intrigue",
  settlement_builder: "Settlement Builder",
  monster_hunt_bounty: "Monster Hunt / Bounty",
  road_trip_pilgrimage: "Road Trip / Pilgrimage",
  merchant_trade_route: "Merchant / Trade Route",
  dating_sim: "Dating Sim",
  choose_your_own_adventure: "Choose Your Own Adventure"
};
type ScenarioForm = {
  scenario_type: string;
  scenario_types: string[];
  interaction_mode: "roleplay" | "storyteller";
  action_choices_enabled: boolean;
  persistent_world_id: string;
  title: string;
  premise: string;
  player_role: string;
  player_character_name: string;
  player_character_profile: string;
  magic_system: string;
  realms_and_places: string;
  factions_and_orders: string;
  myths_and_creatures: string;
  quest_stakes: string;
  technology_level: string;
  setting_scope: string;
  species_and_intelligences: string;
  factions_and_institutions: string;
  mission_stakes: string;
  mission_profile: string;
  ship_or_base_status: string;
  exploration_target: string;
  unknown_intelligence: string;
  knowledge_state: string;
  translation_progress: string;
  discoveries_and_samples: string;
  hazards_and_escalation: string;
  expedition_goal: string;
  route_options: string;
  resource_inventory: string;
  environmental_conditions: string;
  hazards_and_events: string;
  camp_status: string;
  travel_progress: string;
  loop_premise: string;
  reset_trigger: string;
  loop_duration: string;
  starting_state: string;
  objective: string;
  failure_conditions: string;
  baseline_world_state: string;
  loop_schedule: string;
  persistent_knowledge: string;
  persistence_exceptions: string;
  npc_memory_rules: string;
  current_loop_state: string;
  case_facts: string;
  clues: string;
  timeline: string;
  red_herrings: string;
  hidden_truth: string;
  case_status: string;
  target_location: string;
  objectives_and_stakes: string;
  intel_and_access: string;
  security_model: string;
  alert_and_heat: string;
  loadout_and_tools: string;
  complications: string;
  extraction_routes: string;
  aftermath: string;
  political_arena: string;
  political_factions: string;
  central_conflict: string;
  secrets_and_leverage: string;
  reputation_and_standing: string;
  obligations_and_favors: string;
  alliances_and_rivalries: string;
  event_calendar: string;
  political_pressure: string;
  public_private_knowledge: string;
  settlement_profile: string;
  resources_and_indicators: string;
  projects_and_facilities: string;
  threats_and_opportunities: string;
  calendar_and_deadlines: string;
  hunt_profile: string;
  target_profile: string;
  leads_and_clues: string;
  hunt_locations: string;
  preparation_state: string;
  hunt_status: string;
  journey_profile: string;
  route_and_stops: string;
  transport_and_supplies: string;
  recurring_pressures: string;
  relationship_threads: string;
  journey_progress: string;
  trade_profile: string;
  cargo_inventory: string;
  markets_and_stops: string;
  contracts_and_debts: string;
  route_hazards: string;
  profit_and_loss: string;
  tone_genre: string;
  choice_style: string;
  opening_message: string;
};
type ScenarioFormTextField = Exclude<keyof ScenarioForm, "action_choices_enabled" | "persistent_world_id" | "scenario_types" | "interaction_mode">;
type ScenarioDraftPrefill = {
  scenario_type: string;
  scenario_types: string[];
  action_choices_enabled: boolean;
  seed: string;
  interaction_mode?: "roleplay" | "storyteller";
  persistent_world_id?: string | null;
};
const MANUAL_BASE_SECTION_IDS = new Set(["title", "premise", "player_character_name", "player_role", "opening_message"]);
const MANUAL_SCENARIO_TEXTAREA_FIELDS = new Set([
  "premise",
  "player_character_profile",
  "magic_system",
  "realms_and_places",
  "factions_and_orders",
  "myths_and_creatures",
  "quest_stakes",
  "technology_level",
  "setting_scope",
  "species_and_intelligences",
  "factions_and_institutions",
  "mission_stakes",
  "mission_profile",
  "ship_or_base_status",
  "exploration_target",
  "unknown_intelligence",
  "knowledge_state",
  "translation_progress",
  "discoveries_and_samples",
  "hazards_and_escalation",
  "expedition_goal",
  "route_options",
  "resource_inventory",
  "environmental_conditions",
  "hazards_and_events",
  "camp_status",
  "travel_progress",
  "loop_premise",
  "reset_trigger",
  "loop_duration",
  "starting_state",
  "objective",
  "failure_conditions",
  "baseline_world_state",
  "loop_schedule",
  "persistent_knowledge",
  "persistence_exceptions",
  "npc_memory_rules",
  "current_loop_state",
  "case_facts",
  "clues",
  "timeline",
  "red_herrings",
  "hidden_truth",
  "case_status",
  "target_location",
  "objectives_and_stakes",
  "intel_and_access",
  "security_model",
  "alert_and_heat",
  "loadout_and_tools",
  "complications",
  "extraction_routes",
  "aftermath",
  "political_arena",
  "political_factions",
  "central_conflict",
  "secrets_and_leverage",
  "reputation_and_standing",
  "obligations_and_favors",
  "alliances_and_rivalries",
  "event_calendar",
  "political_pressure",
  "public_private_knowledge",
  "settlement_profile",
  "resources_and_indicators",
  "projects_and_facilities",
  "threats_and_opportunities",
  "calendar_and_deadlines",
  "hunt_profile",
  "target_profile",
  "leads_and_clues",
  "hunt_locations",
  "preparation_state",
  "hunt_status",
  "journey_profile",
  "route_and_stops",
  "transport_and_supplies",
  "recurring_pressures",
  "relationship_threads",
  "journey_progress",
  "trade_profile",
  "cargo_inventory",
  "markets_and_stops",
  "contracts_and_debts",
  "route_hazards",
  "profit_and_loss",
  "tone_genre",
  "choice_style",
  "opening_message"
]);
const HIDDEN_WORLD_FIELDS = new Set(["original_key", "source_message_ids", "consolidated"]);
const VIRTUAL_LIST_INITIAL_RECT = { width: 390, height: 520 };
const CHRONICLE_ESTIMATED_ROW_HEIGHT = 156;
const CHRONICLE_ROW_GAP = 16;
const CHRONICLE_ROW_OVERSCAN = 6;
const CHARACTER_TEXT_ESTIMATED_ROW_HEIGHT = 88;
const CHARACTER_TEXT_ROW_GAP = 10;
const CHARACTER_TEXT_ROW_OVERSCAN = 8;
type VirtualElementRect = { width: number; height: number };

function virtualElementRect(element: HTMLElement | null): VirtualElementRect {
  return {
    width: element?.clientWidth || VIRTUAL_LIST_INITIAL_RECT.width,
    height: element?.clientHeight || VIRTUAL_LIST_INITIAL_RECT.height
  };
}

function observeVirtualElementRect(
  element: HTMLElement | null,
  callback: (rect: VirtualElementRect) => void
) {
  const report = () => callback(virtualElementRect(element));
  report();
  if (!element || typeof ResizeObserver === "undefined") return undefined;
  const observer = new ResizeObserver(report);
  observer.observe(element);
  return () => observer.disconnect();
}

function observeVirtualElementOffset(
  element: HTMLElement | null,
  callback: (offset: number, isScrolling: boolean) => void
) {
  callback(element?.scrollTop ?? 0, false);
  if (!element) return undefined;
  const report = () => callback(element.scrollTop, true);
  element.addEventListener("scroll", report, { passive: true });
  return () => element.removeEventListener("scroll", report);
}

function setScrollTopAndNotify(element: HTMLElement, scrollTop: number) {
  element.scrollTop = scrollTop;
  if (typeof Event === "function") {
    element.dispatchEvent(new Event("scroll"));
  }
}

function initialVirtualBottomOffset(count: number, estimatedSize: number, gap: number) {
  if (count <= 0) return 0;
  const estimatedTotal = (count * estimatedSize) + (Math.max(0, count - 1) * gap);
  return Math.max(0, estimatedTotal - VIRTUAL_LIST_INITIAL_RECT.height);
}
const MODEL_CAPABILITY_ALIASES: Record<ModelCapabilityFamily, readonly string[]> = {
  chat: ["chat", "chat_completion", "text"],
  structured_output: ["structured_output", "structured", "json_schema"],
  tool_calling: ["tool_calling", "tools", "function_calling"],
  image_generation: ["image_generation", "image"],
  image_to_image: ["image_to_image", "image_edit", "image_editing", "edit", "inpaint"],
  text_to_video: ["text_to_video", "video_generation", "video"],
  image_to_video: ["image_to_video", "image_plus_text_to_video", "image_text_to_video", "image_animation"],
  vision: ["vision", "image_input", "image_understanding", "image_analysis", "multimodal"]
};
const MODEL_ROUTING_GROUPS: readonly ModelRoutingLaneGroupMeta[] = [
  {
    label: "Narration",
    lanes: [
      {
        id: "narrator",
        label: "Narrator",
        title: "Narrator chat models.",
        capabilities: ["chat"],
        targetPurposes: ["chat"],
        icon: MessageSquare
      },
      {
        id: "narrator_planner",
        label: "Narrator Planner",
        title: "Plan-first narrator model.",
        capabilities: ["structured_output"],
        targetPurposes: ["response_planning"],
        icon: FileText
      },
      {
        id: "narrator_verifier",
        label: "Narrator Verifier",
        title: "Narrator verification models.",
        capabilities: ["structured_output"],
        targetPurposes: ["response_verification", "npc_knowledge_audit"],
        icon: Eye
      },
      {
        id: "content_safety",
        label: "Safety Agent",
        title: "Rating-aware content safety review model.",
        capabilities: ["structured_output"],
        targetPurposes: ["content_safety"],
        icon: ShieldCheck
      }
    ]
  },
  {
    label: "Cast & Direction",
    lanes: [
      {
        id: "character_agents",
        label: "Character Agents",
        title: "NPC presence and intent models.",
        capabilities: ["structured_output"],
        targetPurposes: ["character_presence_assessment", "character_intent_planning", "character_action_planning"],
        icon: Users
      },
      {
        id: "action_choices",
        label: "Action Choices",
        title: "Player action-choice model.",
        capabilities: ["structured_output"],
        targetPurposes: ["action_choice_generation"],
        icon: MessageSquareText
      },
      {
        id: "director_pressure",
        label: "Director Pressure",
        title: "External pressure model.",
        capabilities: ["structured_output"],
        targetPurposes: ["director_pressure"],
        icon: Wand2
      }
    ]
  },
  {
    label: "Context & World",
    lanes: [
      {
        id: "context_selector",
        label: "Context Selector",
        title: "Local context selection model.",
        capabilities: ["structured_output", "tool_calling"],
        targetPurposes: ["context_search"],
        icon: Search
      },
      {
        id: "observation_memory",
        label: "Observation & Memory",
        title: "Observation and memory models.",
        capabilities: ["structured_output"],
        targetPurposes: ["fact_observation", "memory_curation"],
        icon: BookOpen
      },
      {
        id: "world_updates",
        label: "World Updates",
        title: "World update models.",
        capabilities: ["structured_output", "tool_calling"],
        targetPurposes: ["state_memory", "context_update", "character_enhancement"],
        icon: GitBranch
      },
      {
        id: "character_registry_maintenance",
        label: "Character Registry Maintenance",
        title: "Character registry model.",
        capabilities: ["structured_output", "tool_calling"],
        targetPurposes: ["character_registry_maintenance"],
        icon: FileText
      },
      {
        id: "context_cleanup",
        label: "Context Cleanup",
        title: "Stale-context cleanup models.",
        capabilities: ["structured_output", "tool_calling"],
        targetPurposes: ["context_cleanup_scan", "context_cleanup_actions", "guided_context_cleanup", "context_cleanup"],
        icon: FileText
      },
      {
        id: "state_pruning",
        label: "State Pruning",
        title: "World-state pruning model.",
        capabilities: ["structured_output", "tool_calling"],
        targetPurposes: ["state_pruning"],
        icon: FileText
      },
      {
        id: "scenario_evolution",
        label: "Scenario Evolution",
        title: "Scenario change model.",
        capabilities: ["structured_output", "tool_calling"],
        targetPurposes: ["scenario_evolution"],
        icon: FileText
      }
    ]
  },
  {
    label: "Authoring & Media",
    lanes: [
      {
        id: "scenario_writer",
        label: "Scenario Writer",
        title: "Scenario generation and per-section draft models.",
        capabilities: ["chat"],
        targetPurposes: ["scenario_generation"],
        icon: BookOpen
      },
      {
        id: "summarization",
        label: "Summarization",
        title: "Chronicle summary model.",
        capabilities: ["chat"],
        targetPurposes: ["summarization"],
        icon: MessageSquareText
      },
      {
        id: "image_prompt",
        label: "Image Prompt",
        title: "Scene image prompt model.",
        capabilities: ["chat"],
        targetPurposes: ["image_prompt"],
        icon: MessageSquareText
      },
      {
        id: "character_image_description",
        label: "Image Details",
        title: "Image Details.",
        capabilities: ["vision"],
        targetPurposes: ["character_image_description"],
        icon: Eye
      },
      {
        id: "image_generation",
        label: "Image Generation",
        title: "Scene image generation models.",
        capabilities: ["image_generation"],
        targetPurposes: ["image_generation"],
        icon: Image
      },
      {
        id: "image_edit",
        label: "Image Edit",
        title: "Image edit models.",
        capabilities: ["image_to_image"],
        targetPurposes: [
          "image_to_image_generation",
          "scene_image_edit_generation",
          "character_image_edit_generation",
          "text_message_image_edit_generation"
        ],
        icon: Edit3
      },
      {
        id: "video_generation",
        label: "Video Generation",
        title: "Text-to-video generation models.",
        capabilities: ["text_to_video"],
        targetPurposes: ["video_generation"],
        icon: Play
      },
      {
        id: "image_animation",
        label: "Image Animation",
        title: "Image-to-video animation models.",
        capabilities: ["image_to_video"],
        targetPurposes: ["image_animation"],
        icon: Play
      }
    ]
  }
];
const MODEL_FALLBACK_LANES: readonly ModelRoutingLaneMeta[] = [
  {
    id: "narrator_fallback",
    label: "Narrator Fallback",
    title: "Fallback model for blocked or failed narrator turns.",
    capabilities: ["chat"],
    targetPurposes: ["narrator_fallback"],
    icon: MessageSquare
  },
  {
    id: "text_fallback",
    label: "Background Text Fallback",
    title: "Fallback for background text.",
    capabilities: ["chat"],
    targetPurposes: ["chat_fallback"],
    icon: RefreshCw
  },
  {
    id: "structured_fallback",
    label: "Structured Fallback",
    title: "Fallback model for structured-output failures.",
    capabilities: ["structured_output"],
    targetPurposes: ["structured_output_fallback"],
    icon: RefreshCw
  },
  {
    id: "tool_fallback",
    label: "Tool Fallback",
    title: "Fallback model for tool/function-call failures.",
    capabilities: ["tool_calling"],
    targetPurposes: ["tool_call_fallback"],
    icon: RefreshCw
  },
  {
    id: "image_fallback",
    label: "Image Fallback",
    title: "Fallback for image generation.",
    capabilities: ["image_generation"],
    targetPurposes: ["image_fallback"],
    icon: RefreshCw
  },
  {
    id: "image_edit_fallback",
    label: "Image Edit Fallback",
    title: "Fallback model for blocked or failed image edits.",
    capabilities: ["image_to_image"],
    targetPurposes: ["image_edit_fallback"],
    icon: RefreshCw
  },
  {
    id: "video_fallback",
    label: "Video Fallback",
    title: "Fallback for video generation.",
    capabilities: ["text_to_video"],
    targetPurposes: ["video_fallback"],
    icon: RefreshCw
  }
];
const SETTINGS_TAB_TOOLTIPS: Record<SettingsTab, string> = {
  providers: "Configure provider keys and refresh available models.",
  openrouter: "Tune OpenRouter provider routing, privacy, cost, and performance preferences.",
  models: "Choose which models Bragi uses for each task.",
  save: "Control options stored with the active save.",
  local: "Control account and local instance preferences.",
  diagnostics: "View diagnostics, failed background jobs, and recent web events.",
  users: "Manage Bragi users and local passwords."
};
const SAVE_SCOPED_SETTING_KEYS = new Set([
  "turn_responsiveness_mode",
  "automatic_summarization_enabled",
  "summarization_context_pressure_threshold",
  "show_summarization_activity",
  "agentic_context_pipeline_enabled",
  "plan_first_narrator_enabled",
  "director_pressure_enabled",
  "director_pressure_guidance",
  "character_action_planning_enabled",
  "character_action_planning_max_concurrency",
  "character_texts_enabled",
  "character_text_proactive_random_chance_percent",
  "character_text_proactive_random_cooldown_turns",
  "post_turn_inference_mode",
  "npc_knowledge_audit_mode",
  "response_checking_enabled",
  "automatic_image_generation_enabled",
  "automatic_media_mode",
  "image_generation_frequency",
  "venice_image_safe_mode",
  "chat_temperature_enabled",
  "chat_temperature",
  "chat_max_output_tokens_enabled",
  "chat_max_output_tokens",
  "image_dimension_preset",
  "image_style_preset",
  "narrator_planner_recent_player_message_window",
  "narrator_planner_recent_narrator_message_window",
  "recent_player_message_window",
  "recent_narrator_message_window",
  "context_budget_mode",
  "context_budget_fixed_total_chars",
  "context_budget_adaptive_fraction",
  "manual_confirmation_memories_enabled",
  "manual_confirmation_character_registry_enabled",
  "manual_confirmation_state_changes_enabled",
  "generated_text_script_guard_mode",
  "save_generated_phrase_denylist",
  "scenario_evolution_turn_interval"
]);
const TASK_MODEL_TOOLTIPS: Record<string, string> = {
  chat: "Sets the default model Bragi uses for narrator chat responses.",
  chat_full_roleplay: "Sets the model Bragi uses for generic roleplay narrator responses.",
  chat_fantasy_roleplay: "Sets the model Bragi uses for fantasy narrator responses.",
  chat_science_fiction_roleplay: "Sets the model Bragi uses for science fiction narrator responses.",
  chat_first_contact_exploration: "Sets the model Bragi uses for first-contact and exploration narrator responses.",
  chat_survival_expedition: "Sets the model Bragi uses for survival expedition narrator responses.",
  chat_time_loop: "Sets the model Bragi uses for time loop narrator responses.",
  chat_investigation_mystery: "Sets the model Bragi uses for investigation mystery narrator responses.",
  chat_heist_infiltration: "Sets the model Bragi uses for heist and infiltration narrator responses.",
  chat_political_intrigue: "Sets the model Bragi uses for political intrigue narrator responses.",
  chat_dating_sim: "Sets the model Bragi uses for dating sim narrator responses.",
  context_search: "Sets the model Bragi uses to select relevant local context sources.",
  fact_observation: "Sets the structured-output model Bragi uses to extract factual observations around narrator turns.",
  memory_curation: "Sets the structured-output model Bragi uses to curate observations into memory suggestions.",
  response_planning: "Sets the structured-output model Bragi uses to prepare the plan-first narrator turn plan.",
  response_verification: "Sets the structured-output model Bragi uses to verify narrator output before committing turn updates.",
  state_memory: "Sets the model Bragi uses to extract and update structured world state and memories.",
  context_update: "Sets the model Bragi uses to propose world and memory updates.",
  character_enhancement: "Sets the model Bragi uses when auto-enhancing character profile fields.",
  action_choice_generation: "Sets the structured-output model Bragi uses to suggest player action choices after narrator turns.",
  character_presence_assessment: "Sets the structured-output model Bragi uses to assess ambiguous NPC scene entry/exit before narrator turns.",
  character_intent_planning: "Sets the structured-output model Bragi uses for dating-route intent profiling fallbacks. NPC intents are planned by the narrator planner.",
  context_cleanup_scan: "Sets the model Bragi uses to scan transcript chunks for possible stale-context cleanup notes.",
  context_cleanup_actions: "Sets the model Bragi uses to propose automatic cleanup actions for stale context records.",
  guided_context_cleanup: "Sets the model Bragi uses for user-directed cleanup action proposals.",
  context_cleanup: "Sets the model Bragi uses when cleaning stale context records.",
  state_pruning: "Sets the model Bragi uses to compact low-value world state.",
  scenario_generation: "Sets the model Bragi uses while generating new scenario sections.",
  scenario_evolution: "Sets the model Bragi uses to detect scenario changes after turns.",
  director_pressure: "Sets the structured-output model Bragi uses to assess story tension and plan external pressure.",
  character_action_planning: "Legacy fallback model for character-agent NPC planning.",
  npc_knowledge_audit: "Sets the structured-output model Bragi uses to audit NPC knowledge boundaries after narrator verification.",
  content_safety: "Sets the structured-output safety agent Bragi uses to enforce the selected content-rating ceiling.",
  summarization: "Sets the model Bragi uses to summarize older chronicle context.",
  image_prompt: "Sets the model Bragi uses to turn scene context into image prompts.",
  image_generation: "Sets the model Bragi uses to generate scene images.",
  image_to_image_generation: "Sets the default image-edit model Bragi uses when a flow-specific image edit override is not set.",
  scene_image_edit_generation: "Sets the image-edit model Bragi uses when scene images are generated from character reference images.",
  character_image_edit_generation: "Sets the image-edit model Bragi uses when character images are generated from reference images.",
  text_message_image_edit_generation: "Sets the image-edit model Bragi uses when text-message images are generated from reference images.",
  video_generation: "Sets the model Bragi uses for text-to-video work when available.",
  image_animation: "Sets the model Bragi uses to animate existing images when available.",
  narrator_fallback: "Sets the backup chat model used when primary narrator output is blocked.",
  chat_fallback: "Sets the backup chat model used when primary background text output is blocked.",
  structured_output_fallback: "Sets the backup structured-output model used after primary failures.",
  tool_call_fallback: "Sets the backup tool-calling model used after primary tool-call failures.",
  image_fallback: "Sets the backup image model used when primary image generation fails.",
  image_edit_fallback: "Sets the backup image-edit model used when primary image edits fail.",
  video_fallback: "Sets the backup video model used when primary video work fails.",
  character_image_description: "Image Details.",
  character_registry_maintenance: "Sets the model Bragi uses to maintain character registry entries."
};
const SETTING_TOOLTIPS: Record<string, string> = {
  turn_responsiveness_mode: "Quality keeps full helper work; Responsive bounds foreground helpers.",
  automatic_summarization_enabled: "When enabled, Bragi summarizes older chronicle context as saves grow.",
  summarization_context_pressure_threshold: "Controls how full the context budget can get before summarization is eligible.",
  show_summarization_activity: "Shows summarization work in the status area instead of keeping it quiet.",
  agentic_context_pipeline_enabled: "Runs structured observation, memory curation, narrator planning, and response verification around narrator turns.",
  plan_first_narrator_enabled: "Uses the structured turn plan alongside normal continuity context for narrator turns.",
  director_pressure_enabled: "Lets Bragi assess completed turns for conservative external story pressure when tension stalls.",
  director_pressure_guidance: "Binding Director guidance. When set, it runs after every verified turn; canon, agency, autonomy, and safety still apply.",
  character_action_planning_enabled: "Runs one structured planning call per active NPC before narrator turns to update scene presence and NPC actions.",
  character_action_planning_max_concurrency: "Limits how many NPC action planning calls can run at the same time.",
  character_texts_enabled: "Enables side-channel phone text threads with characters for this save.",
  character_text_proactive_random_chance_percent: "Controls automatic proactive character texts after turns. 0 disables all proactive texts; higher values enable strong triggers and set the ambient random text chance.",
  character_text_proactive_random_cooldown_turns: "Sets how many narrator turns Bragi waits between proactive character texts for the same contact; 0 means no cooldown.",
  post_turn_inference_mode: "Controls how Bragi handles legacy narrator-prose state and memory inference after verified planned commits.",
  npc_knowledge_audit_mode: "Controls whether NPC knowledge-boundary audit findings block narration or stay diagnostic after one retry.",
  response_checking_enabled: "When disabled, skips narrator verification and NPC knowledge auditing for this save. Deterministic script and phrase safeguards remain enabled.",
  chat_fallback_enabled: "When chat output looks blocked or refused, retry with the configured fallback model.",
  structured_output_fallback_enabled: "Retries structured maintenance work with the fallback model when the primary model fails.",
  tool_call_fallback_enabled: "Retries tool-call maintenance work with the fallback model when the primary model fails.",
  image_fallback_enabled: "Retries failed image generation and image edits with the configured image fallback models.",
  video_fallback_enabled: "Retries blocked or unavailable video work with the configured video fallback model.",
  venice_image_safe_mode: "Requests Venice media safe mode for image generation and video models that advertise support.",
  content_filter_rating: "Maximum rating for generated narration and media prompts.",
  fade_to_black_enabled: "Fades explicit sexual narration to black.",
  debug_logging_enabled: "Writes extra local diagnostics for troubleshooting.",
  pending_jobs_display_mode: "Controls how much detail appears in the pending jobs tray.",
  user_narration_guidance: "Sets account-level narrator guidance for saves without save-specific response guidance.",
  automatic_image_generation_enabled: "Generates scene images automatically after eligible narrator turns.",
  automatic_media_mode: "Chooses which media type automatic generation creates; image is currently supported.",
  image_style_preset: "Adds a reusable visual style hint to generated image prompts.",
  image_generation_frequency: "Sets how many narrator turns Bragi waits between automatic scene images; 0 disables the interval.",
  chat_temperature_enabled: "Enables a custom chat temperature when the selected provider model reports support for it.",
  chat_temperature: "Controls randomness for chat models that support temperature.",
  chat_max_output_tokens_enabled: "Enables a custom chat response token limit when the selected provider model reports support for it.",
  chat_max_output_tokens: "Caps chat response length for models that support max output tokens.",
  image_dimension_preset: "Requests an image size preset when the selected image model reports support for dimensions.",
  narrator_planner_recent_player_message_window: "Sets how many recent player messages the narrator planner receives as direct transcript context.",
  narrator_planner_recent_narrator_message_window: "Sets how many recent narrator messages the narrator planner receives as direct transcript context.",
  recent_player_message_window: "Sets how many recent player messages the narrator prose writer receives as direct transcript context.",
  recent_narrator_message_window: "Sets how many recent narrator messages the narrator prose writer receives as direct transcript context.",
  context_budget_mode: "Chooses how Bragi limits assembled narrator context.",
  context_budget_fixed_total_chars: "Sets the fixed character budget used when fixed context budgeting is active.",
  context_budget_adaptive_fraction: "Sets the share of available context Bragi uses in adaptive budget mode.",
  manual_confirmation_memories: "Queue extracted memories for review before adding them to the save.",
  manual_confirmation_memories_enabled: "Queue extracted memories for review before adding them to the save.",
  manual_confirmation_character_registry: "Queue new character registry entries for review before adding them to the save.",
  manual_confirmation_character_registry_enabled: "Queue new character registry entries for review before adding them to the save.",
  manual_confirmation_state_changes: "Queue proposed world-state changes for review before applying them.",
  manual_confirmation_state_changes_enabled: "Queue proposed world-state changes for review before applying them.",
  generated_text_script_guard_mode: "Rejects generated memory and context text that introduces scripts absent from the source text.",
  generated_phrase_denylist: "Rejects generated narrator, text, and voice examples containing globally denied phrases.",
  save_generated_phrase_denylist: "Adds save-specific phrases to the global denylist."
};
const OPENROUTER_ROUTING_TOOLTIPS: Record<string, string> = {
  profile: "Choose whether these routing rules apply globally or only to a task family.",
  use_custom_profile: "Override the global OpenRouter routing profile for this task family.",
  sort: "Choose how OpenRouter prioritizes providers: default load balancing, lowest price, highest throughput, or lowest latency.",
  sort_partition: "Controls whether sorting stays grouped by model or ranks all fallback model endpoints together.",
  allow_fallbacks: "Controls whether OpenRouter may try backup providers when the preferred provider is unavailable.",
  require_parameters: "Only route to providers that support every parameter Bragi sends.",
  order: "Provider slugs OpenRouter should try first, one per line.",
  only: "Provider slugs OpenRouter is allowed to use, one per line.",
  ignore: "Provider slugs OpenRouter should skip, one per line.",
  data_collection: "Controls whether providers that may store request data are allowed.",
  zdr: "Restrict routing to OpenRouter endpoints marked for zero data retention.",
  enforce_distillable_text: "Restrict routing to models that allow text distillation.",
  quantizations: "Limit routing to providers using the selected model precision levels.",
  preferred_min_throughput: "Prefer providers meeting these tokens-per-second percentile thresholds.",
  preferred_max_latency: "Prefer providers meeting these response-latency percentile thresholds.",
  max_price: "Do not route to providers above these price limits.",
  provider_object: "Shows the exact OpenRouter provider object Bragi will send for this profile."
};
const OPENROUTER_QUANTIZATION_TOOLTIPS: Record<string, string> = {
  int4: "Integer 4-bit quantization; usually cheaper or faster, but can reduce output quality.",
  int8: "Integer 8-bit quantization; more compact than full precision with possible quality tradeoffs.",
  fp4: "Floating-point 4-bit quantization; usually cheaper or faster, but can reduce output quality.",
  fp6: "Floating-point 6-bit quantization; lower bit depth can be cheaper or faster, but less faithful.",
  fp8: "Floating-point 8-bit quantization; often faster or cheaper than 16-bit precision, with possible quality tradeoffs.",
  fp16: "Floating-point 16-bit precision; common higher-quality inference precision.",
  bf16: "Brain floating-point 16-bit precision; common higher-quality inference precision.",
  fp32: "Floating-point 32-bit precision; highest precision option when available.",
  unknown: "Provider did not publish a quantization level."
};
const OPENROUTER_MAX_PRICE_TOOLTIPS: Record<string, string> = {
  prompt: "Maximum prompt-token price per million tokens OpenRouter may route to.",
  completion: "Maximum completion-token price per million tokens OpenRouter may route to.",
  request: "Maximum per-request price OpenRouter may route to.",
  image: "Maximum per-image price OpenRouter may route to."
};
const WORKBENCH_LAYOUT_STORAGE_KEY = "bragi-web:workbench-layout:v1";
const SELECTED_SAVE_STORAGE_KEY = "bragi-web:selected-save-id:v1";
const LIBRARY_CONTROLS_STORAGE_KEY = "bragi-web:library-controls:v1";
const CHARACTER_TEXT_SEEN_STORAGE_KEY = "bragi-web:character-text-seen:v1";
const WORKBENCH_STACKED_BREAKPOINT = 1050;
const WORKBENCH_CHRONICLE_MIN_WIDTH = 360;
const WORKBENCH_RESIZE_STEP = 16;
const WORKBENCH_RESIZE_FAST_STEP = 48;
const DEFAULT_WORKBENCH_LAYOUT: WorkbenchLayout = { leftRailWidth: 278, rightPanelWidth: 370 };
const WORKBENCH_STACKED_QUERY = "(max-width: 1050px)";
const WORKBENCH_MOBILE_QUERY = "(max-width: 760px), (pointer: coarse) and (max-height: 520px)";
const WORKBENCH_LAYOUT_LIMITS: Record<ResizeSide, { min: number; max: number }> = {
  left: { min: 220, max: 420 },
  right: { min: 300, max: 560 }
};
const DEFAULT_LIBRARY_CONTROLS_STATE: LibraryControlsState = {
  activeTab: "saves",
  saveQuery: "",
  saveSort: "updated",
  saveDirection: "desc",
  scenarioQuery: "",
  scenarioSort: "updated",
  scenarioDirection: "desc",
  scenarioType: "all",
  scenarioUsage: "all"
};
const PANEL_BUTTONS: [PanelName, string, LucideIcon][] = [
  ["media", "Media", Image],
  ["history", "History", History],
  ["world", "World", Archive],
  ["characters", "Characters", Users],
  ["settings", "Settings", Settings]
];

function loadWorkbenchLayout(): WorkbenchLayout {
  if (typeof window === "undefined") return DEFAULT_WORKBENCH_LAYOUT;
  try {
    const stored = window.localStorage.getItem(WORKBENCH_LAYOUT_STORAGE_KEY);
    if (!stored) return DEFAULT_WORKBENCH_LAYOUT;
    const parsed = JSON.parse(stored) as unknown;
    if (!isWorkbenchLayout(parsed)) return DEFAULT_WORKBENCH_LAYOUT;
    return constrainWorkbenchLayout(parsed);
  } catch {
    return DEFAULT_WORKBENCH_LAYOUT;
  }
}

function saveWorkbenchLayout(layout: WorkbenchLayout) {
  try {
    window.localStorage.setItem(WORKBENCH_LAYOUT_STORAGE_KEY, JSON.stringify(layout));
  } catch {
    // Local persistence is a convenience; private or locked-down storage should not break the app.
  }
}

function selectedSaveStorageKey(userId: string | null) {
  return userId ? `${SELECTED_SAVE_STORAGE_KEY}:${userId}` : SELECTED_SAVE_STORAGE_KEY;
}

function loadSelectedSaveId(userId: string | null = null): string | null {
  if (typeof window === "undefined") return null;
  try {
    const scoped = window.localStorage.getItem(selectedSaveStorageKey(userId));
    const stored = scoped ?? (userId ? window.localStorage.getItem(SELECTED_SAVE_STORAGE_KEY) : null);
    return stored?.trim() || null;
  } catch {
    return null;
  }
}

function saveSelectedSaveId(saveId: string | null, userId: string | null = null) {
  try {
    const key = selectedSaveStorageKey(userId);
    if (saveId) {
      window.localStorage.setItem(key, saveId);
      if (userId) window.localStorage.removeItem(SELECTED_SAVE_STORAGE_KEY);
    } else {
      window.localStorage.removeItem(key);
      if (userId) window.localStorage.removeItem(SELECTED_SAVE_STORAGE_KEY);
    }
  } catch {
    // Local persistence is optional; the runtime response remains the source of save data.
  }
}

function characterTextSeenStorageKey(userId: string | null, saveId: string | null) {
  const userScope = userId || "anonymous";
  const saveScope = saveId || "no-save";
  return `${CHARACTER_TEXT_SEEN_STORAGE_KEY}:${userScope}:${saveScope}`;
}

function loadCharacterTextSeenState(
  userId: string | null,
  saveId: string | null,
): Record<string, string> {
  if (typeof window === "undefined" || !saveId) return {};
  try {
    const stored = window.localStorage.getItem(
      characterTextSeenStorageKey(userId, saveId),
    );
    if (!stored) return {};
    const parsed = JSON.parse(stored) as unknown;
    if (!parsed || typeof parsed !== "object") return {};
    return Object.fromEntries(
      Object.entries(parsed as Record<string, unknown>).flatMap(([threadId, messageId]) => (
        typeof messageId === "string" && threadId.trim()
          ? [[threadId, messageId]]
          : []
      )),
    );
  } catch {
    return {};
  }
}

function saveCharacterTextSeenState(
  userId: string | null,
  saveId: string | null,
  state: Record<string, string>,
) {
  if (!saveId) return;
  try {
    window.localStorage.setItem(
      characterTextSeenStorageKey(userId, saveId),
      JSON.stringify(state),
    );
  } catch {
    // Unread markers are local convenience state; the inbox still reflects server data.
  }
}

function incomingCharacterTextContacts(
  model: CharacterTextsModel | undefined,
  seenState: Record<string, string>,
): CharacterTextContact[] {
  if (!model?.enabled) return [];
  return model.contacts.filter((contact) => (
    Boolean(contact.thread_id)
    && Boolean(contact.latest_message_id)
    && contact.latest_message_sender === "character"
    && !contact.latest_message_read_at
    && seenState[contact.thread_id ?? ""] !== contact.latest_message_id
  ));
}

function libraryControlsStorageKey(userId: string | null) {
  return userId ? `${LIBRARY_CONTROLS_STORAGE_KEY}:${userId}` : LIBRARY_CONTROLS_STORAGE_KEY;
}

function loadLibraryControlsState(userId: string | null = null): LibraryControlsState {
  if (typeof window === "undefined") return DEFAULT_LIBRARY_CONTROLS_STATE;
  try {
    const stored = window.localStorage.getItem(libraryControlsStorageKey(userId));
    if (!stored) return DEFAULT_LIBRARY_CONTROLS_STATE;
    return normalizeLibraryControlsState(JSON.parse(stored) as unknown);
  } catch {
    return DEFAULT_LIBRARY_CONTROLS_STATE;
  }
}

function saveLibraryControlsState(userId: string | null, state: LibraryControlsState) {
  try {
    window.localStorage.setItem(libraryControlsStorageKey(userId), JSON.stringify(state));
  } catch {
    // Search and sort preferences are nice-to-have; blocked storage should not break library use.
  }
}

function normalizeLibraryControlsState(value: unknown): LibraryControlsState {
  if (!value || typeof value !== "object") return DEFAULT_LIBRARY_CONTROLS_STATE;
  const candidate = value as Partial<Record<keyof LibraryControlsState, unknown>>;
  return {
    activeTab: candidate.activeTab === "scenarios" || candidate.activeTab === "worlds"
      ? candidate.activeTab
      : "saves",
    saveQuery: typeof candidate.saveQuery === "string" ? candidate.saveQuery : "",
    saveSort: isSaveSortKey(candidate.saveSort) ? candidate.saveSort : "updated",
    saveDirection: candidate.saveDirection === "asc" ? "asc" : "desc",
    scenarioQuery: typeof candidate.scenarioQuery === "string" ? candidate.scenarioQuery : "",
    scenarioSort: isScenarioSortKey(candidate.scenarioSort) ? candidate.scenarioSort : "updated",
    scenarioDirection: candidate.scenarioDirection === "asc" ? "asc" : "desc",
    scenarioType: typeof candidate.scenarioType === "string" && candidate.scenarioType ? candidate.scenarioType : "all",
    scenarioUsage: isScenarioUsageFilter(candidate.scenarioUsage) ? candidate.scenarioUsage : "all"
  };
}

function isSaveSortKey(value: unknown): value is SaveSortKey {
  return value === "last_opened" || value === "title" || value === "created" || value === "updated" || value === "scenario_title";
}

function isScenarioSortKey(value: unknown): value is ScenarioSortKey {
  return value === "updated" || value === "title" || value === "created" || value === "save_count" || value === "type";
}

function isScenarioUsageFilter(value: unknown): value is ScenarioUsageFilter {
  return value === "all" || value === "used" || value === "unused";
}

function isWorkbenchLayout(value: unknown): value is WorkbenchLayout {
  if (!value || typeof value !== "object") return false;
  const layout = value as Partial<WorkbenchLayout>;
  return isFiniteNumber(layout.leftRailWidth) && isFiniteNumber(layout.rightPanelWidth);
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function constrainWorkbenchLayout(layout: WorkbenchLayout, resizedSide?: ResizeSide): WorkbenchLayout {
  let leftRailWidth = clamp(Math.round(layout.leftRailWidth), WORKBENCH_LAYOUT_LIMITS.left.min, WORKBENCH_LAYOUT_LIMITS.left.max);
  let rightPanelWidth = clamp(Math.round(layout.rightPanelWidth), WORKBENCH_LAYOUT_LIMITS.right.min, WORKBENCH_LAYOUT_LIMITS.right.max);
  const viewportWidth = typeof window === "undefined" ? 0 : window.innerWidth;
  if (viewportWidth > WORKBENCH_STACKED_BREAKPOINT) {
    const minimumSideWidth = WORKBENCH_LAYOUT_LIMITS.left.min + WORKBENCH_LAYOUT_LIMITS.right.min;
    const maximumSideWidth = Math.max(minimumSideWidth, viewportWidth - WORKBENCH_CHRONICLE_MIN_WIDTH);
    const overflow = leftRailWidth + rightPanelWidth - maximumSideWidth;
    if (overflow > 0) {
      if (resizedSide === "left") {
        leftRailWidth = Math.max(WORKBENCH_LAYOUT_LIMITS.left.min, leftRailWidth - overflow);
      } else if (resizedSide === "right") {
        rightPanelWidth = Math.max(WORKBENCH_LAYOUT_LIMITS.right.min, rightPanelWidth - overflow);
      } else {
        const rightReduction = Math.min(overflow, rightPanelWidth - WORKBENCH_LAYOUT_LIMITS.right.min);
        rightPanelWidth -= rightReduction;
        leftRailWidth = Math.max(WORKBENCH_LAYOUT_LIMITS.left.min, leftRailWidth - (overflow - rightReduction));
      }
    }
  }
  return { leftRailWidth, rightPanelWidth };
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(Math.max(value, minimum), maximum);
}

function useMediaQuery(query: string) {
  const [matches, setMatches] = useState(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return undefined;
    const media = window.matchMedia(query);
    const update = () => setMatches(media.matches);
    update();
    if (typeof media.addEventListener === "function") {
      media.addEventListener("change", update);
      return () => media.removeEventListener("change", update);
    }
    media.addListener?.(update);
    return () => media.removeListener?.(update);
  }, [query]);

  return matches;
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <SessionShell />
    </QueryClientProvider>
  );
}

function SessionShell() {
  const client = useQueryClient();
  const [session, setSession] = useState<SessionState>({ status: "checking" });
  const [sessionReload, setSessionReload] = useState(0);

  const markLoggedOut = useCallback((message?: string) => {
    client.clear();
    setSession({ status: "login", message });
  }, [client]);

  const handleAuthenticated = useCallback((user: CurrentUser) => {
    client.clear();
    setSession({ status: "authenticated", user });
  }, [client]);

  useEffect(() => {
    let active = true;
    async function loadSession() {
      try {
        const session = await api<AuthSessionResponse>("/api/auth/session");
        if (!active) return;
        const bootstrap = session.bootstrap;
        if (bootstrap.bootstrap_required) {
          setSession({ status: "bootstrap", setupTokenRequired: bootstrap.setup_token_required });
          return;
        }
        if (session.user) {
          handleAuthenticated(session.user);
          return;
        }
        setSession({ status: "login" });
      } catch (failure) {
        if (!active) return;
        setSession({
          status: "error",
          message: failure instanceof Error ? failure.message : "Could not load session"
        });
      }
    }
    void loadSession();
    return () => {
      active = false;
    };
  }, [handleAuthenticated, sessionReload]);

  useEffect(() => {
    setUnauthorizedHandler(() => markLoggedOut("Session expired. Log in again to continue."));
    return () => setUnauthorizedHandler(null);
  }, [markLoggedOut]);

  const logout = useCallback(async () => {
    try {
      await postJson<{ ok: boolean }>("/api/auth/logout", {});
    } finally {
      markLoggedOut();
    }
  }, [markLoggedOut]);

  if (session.status === "authenticated") {
    return <Workbench currentUser={session.user} onLogout={logout} />;
  }
  if (session.status === "bootstrap") {
    return (
      <AuthPanel
        mode="bootstrap"
        message={session.message}
        setupTokenRequired={session.setupTokenRequired}
        onAuthenticated={handleAuthenticated}
      />
    );
  }
  if (session.status === "login") {
    return (
      <AuthPanel
        mode="login"
        message={session.message}
        onAuthenticated={handleAuthenticated}
      />
    );
  }
  if (session.status === "error") {
    return (
      <main className="auth-shell">
        <section className="auth-panel" aria-live="polite">
          <p className="eyebrow">Bragi</p>
          <h1>Could not open Bragi</h1>
          <InlineNotice>{session.message}</InlineNotice>
          <button
            type="button"
            onClick={() => {
              setSession({ status: "checking" });
              setSessionReload((value) => value + 1);
            }}
          >
            Retry
          </button>
        </section>
      </main>
    );
  }
  return (
    <main className="auth-shell">
      <section className="auth-panel" aria-live="polite">
        <p className="eyebrow">Bragi</p>
        <h1>Opening Bragi</h1>
        <Loader2 className="spin" size={20} aria-hidden="true" />
      </section>
    </main>
  );
}

function AuthPanel({
  mode,
  message,
  setupTokenRequired = false,
  onAuthenticated
}: {
  mode: "bootstrap" | "login";
  message?: string;
  setupTokenRequired?: boolean;
  onAuthenticated: (user: CurrentUser) => void;
}) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [setupToken, setSetupToken] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const isBootstrap = mode === "bootstrap";
  const title = isBootstrap ? "Create first admin" : "Log in to Bragi";
  const buttonLabel = isBootstrap ? "Create admin" : "Log in";
  const endpoint = isBootstrap ? "/api/bootstrap/admin" : "/api/auth/login";

  async function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);
    try {
      const response = await postJson<AuthResponse>(
        endpoint,
        isBootstrap && setupTokenRequired
          ? { username, password, setup_token: setupToken }
          : { username, password }
      );
      onAuthenticated(response.user);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Authentication failed");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <main className="auth-shell">
      <form className="auth-panel" onSubmit={submit}>
        <p className="eyebrow">Bragi</p>
        <h1>{title}</h1>
        {message ? <InlineNotice polite>{message}</InlineNotice> : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <label className="auth-field">
          <span>Username</span>
          <input
            autoComplete="username"
            autoFocus
            value={username}
            onChange={(event) => setUsername(event.currentTarget.value)}
          />
        </label>
        <label className="auth-field">
          <span>Password</span>
          <input
            autoComplete={isBootstrap ? "new-password" : "current-password"}
            minLength={isBootstrap ? 12 : undefined}
            type="password"
            value={password}
            onChange={(event) => setPassword(event.currentTarget.value)}
          />
        </label>
        {isBootstrap && setupTokenRequired ? (
          <label className="auth-field">
            <span>Setup token</span>
            <input
              autoComplete="one-time-code"
              type="password"
              value={setupToken}
              onChange={(event) => setSetupToken(event.currentTarget.value)}
            />
          </label>
        ) : null}
        <button
          type="submit"
          disabled={submitting || !username.trim() || !password || (setupTokenRequired && !setupToken)}
        >
          {submitting ? <Loader2 className="spin" size={16} aria-hidden="true" /> : null}
          {buttonLabel}
        </button>
      </form>
    </main>
  );
}

function installGlobalErrorLogging() {
  window.addEventListener("error", (event) => {
    logClientEvent("error", "client.window.error", {
      component: "window",
      error_name: event.error instanceof Error ? event.error.name : "Error",
      error_message: event.message,
      route: window.location.pathname
    });
  });
  window.addEventListener("unhandledrejection", (event) => {
    const reason = event.reason;
    logClientEvent("error", "client.window.unhandled_rejection", {
      component: "window",
      error_name: reason instanceof Error ? reason.name : "UnhandledRejection",
      error_message: reason instanceof Error ? reason.message : String(reason),
      route: window.location.pathname
    });
  });
}

function mountApp() {
  installGlobalErrorLogging();
  ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
}

function chatSubmissionStatusPath(activeSaveId: string | null) {
  return activeSaveId
    ? `/api/chat/submission-status?save_id=${encodeURIComponent(activeSaveId)}`
    : "/api/chat/submission-status";
}

function chatTimingSummaryPath(activeSaveId: string) {
  return `/api/chat/timing-summary?save_id=${encodeURIComponent(activeSaveId)}`;
}

function runtimePath(saveId: string | null) {
  return saveId
    ? `/api/runtime/shell?save_id=${encodeURIComponent(saveId)}`
    : "/api/runtime/shell";
}

function runtimeQueryKey(saveId: string | null) {
  return ["runtime", saveId] as const;
}

function chroniclePagePath(saveId: string, beforeMessageId: string, limit?: number) {
  const params = new URLSearchParams({ before_message_id: beforeMessageId });
  if (limit !== undefined) params.set("limit", String(limit));
  return `/api/saves/${encodeURIComponent(saveId)}/chronicle?${params.toString()}`;
}

function mediaPath(saveId: string) {
  return `/api/saves/${encodeURIComponent(saveId)}/media`;
}

function invalidateScenePresenceQueries(client: QueryClient, saveId: string | null) {
  client.invalidateQueries({ queryKey: saveId ? ["scene-presence", saveId] : ["scene-presence"] });
}

const CHAT_HISTORY_PAGE_LIMIT = 80;

function chatHistoryPath(activeSaveId: string | null, filter: string, beforeMessageId?: string | null) {
  const params = new URLSearchParams({ filter, limit: String(CHAT_HISTORY_PAGE_LIMIT) });
  if (activeSaveId) params.set("save_id", activeSaveId);
  if (beforeMessageId) params.set("before_message_id", beforeMessageId);
  return `/api/chat-history?${params.toString()}`;
}

function worldDataPath(activeSaveId: string | null) {
  return activeSaveId
    ? `/api/world-data?save_id=${encodeURIComponent(activeSaveId)}`
    : "/api/world-data";
}

function charactersPath(activeSaveId: string | null) {
  return activeSaveId
    ? `/api/characters?save_id=${encodeURIComponent(activeSaveId)}`
    : "/api/characters";
}

function characterTextsPath(activeSaveId: string | null) {
  return activeSaveId
    ? `/api/character-texts?save_id=${encodeURIComponent(activeSaveId)}`
    : "/api/character-texts";
}

function characterTextThreadPath(activeSaveId: string | null, threadId: string) {
  const query = activeSaveId ? `?save_id=${encodeURIComponent(activeSaveId)}` : "";
  return `/api/character-texts/threads/${encodeURIComponent(threadId)}${query}`;
}

function characterTextThreadReadPath(threadId: string) {
  return `/api/character-texts/threads/${encodeURIComponent(threadId)}/read`;
}

function characterTextContactPath(characterId: string) {
  return `/api/character-texts/contacts/${encodeURIComponent(characterId)}`;
}

function mediaAssetPath(assetId: string, activeSaveId: string | null) {
  const query = activeSaveId ? `?save_id=${encodeURIComponent(activeSaveId)}` : "";
  return `/api/media/${encodeURIComponent(assetId)}${query}`;
}

function mediaAssetPromptPath(assetId: string, activeSaveId: string | null) {
  const query = activeSaveId ? `?save_id=${encodeURIComponent(activeSaveId)}` : "";
  return `/api/media/${encodeURIComponent(assetId)}/prompt${query}`;
}

function mediaAssetThumbnailPath(assetId: string, activeSaveId: string | null) {
  const query = activeSaveId ? `?save_id=${encodeURIComponent(activeSaveId)}` : "";
  return `/api/media/${encodeURIComponent(assetId)}/thumbnail${query}`;
}

function scenarioStarterReferenceThumbnailPath(scenarioId: string, imageId: string) {
  return `/api/scenarios/${encodeURIComponent(scenarioId)}/character-starters/reference-images/${encodeURIComponent(imageId)}/thumbnail`;
}

function activeJobsPath(activeSaveId: string | null) {
  return activeSaveId
    ? `/api/jobs?status=active&save_id=${encodeURIComponent(activeSaveId)}`
    : "/api/jobs?status=active";
}

function apiRead<T>(path: string, signal?: AbortSignal): Promise<T> {
  return api<T>(path, { signal });
}

const SETTINGS_BACKGROUND_STALE_MS = 60_000;

type DiagnosticsFilters = {
  request_id: string;
  job_id: string;
  route: string;
  component: string;
  since: string;
  limit: string;
};

const EMPTY_DIAGNOSTICS_FILTERS: DiagnosticsFilters = {
  request_id: "",
  job_id: "",
  route: "",
  component: "",
  since: "",
  limit: ""
};
const DIAGNOSTICS_STALE_MS = 5_000;
type TerminalJobStatusFilter = "terminal" | "failed" | "cancelled" | "succeeded";

function diagnosticsPath(activeSaveId: string | null, filters: DiagnosticsFilters) {
  const params = new URLSearchParams();
  if (activeSaveId) params.set("save_id", activeSaveId);
  Object.entries(filters).forEach(([key, value]) => {
    const trimmed = value.trim();
    if (trimmed) params.set(key, trimmed);
  });
  const query = params.toString();
  return query ? `/api/diagnostics?${query}` : "/api/diagnostics";
}

function terminalJobsPath(activeSaveId: string | null, status: TerminalJobStatusFilter, filters: DiagnosticsFilters) {
  const params = new URLSearchParams({ status });
  if (activeSaveId) params.set("save_id", activeSaveId);
  const since = filters.since.trim();
  const limit = filters.limit.trim();
  if (since) params.set("since", since);
  if (limit) params.set("limit", limit);
  return `/api/jobs?${params.toString()}`;
}

function jobStepsPath(job: TerminalJobSummary) {
  const params = new URLSearchParams();
  if (job.save_id) params.set("save_id", job.save_id);
  const query = params.toString();
  return `/api/jobs/${encodeURIComponent(job.id)}/steps${query ? `?${query}` : ""}`;
}

function jobDiagnosticsPath(job: Pick<TerminalJobSummary, "id" | "save_id">) {
  const params = new URLSearchParams();
  if (job.save_id) params.set("save_id", job.save_id);
  const query = params.toString();
  return `/api/jobs/${encodeURIComponent(job.id)}/diagnostics${query ? `?${query}` : ""}`;
}

function diagnosticsBundleFilename(generatedAt?: string) {
  const stamp = (generatedAt ?? "bundle").replace(/[^A-Za-z0-9._-]/g, "-");
  return `bragi-diagnostics-${stamp}.json`;
}

function diagnosticsQueryKey(activeSaveId: string | null, filters: DiagnosticsFilters) {
  return ["diagnostics", activeSaveId, filters.request_id, filters.job_id, filters.route, filters.component, filters.since, filters.limit] as const;
}

function diagnosticsQueryOptions(activeSaveId: string | null, filters: DiagnosticsFilters) {
  return {
    queryKey: diagnosticsQueryKey(activeSaveId, filters),
    queryFn: ({ signal }: { signal: AbortSignal }) => apiRead<DiagnosticsModel>(diagnosticsPath(activeSaveId, filters), signal),
    staleTime: DIAGNOSTICS_STALE_MS
  };
}

function jobBelongsToActiveSave(job: Job, activeSaveId: string | null) {
  return job.save_id === undefined || job.save_id === null || job.save_id === activeSaveId;
}

function runtimeEventBelongsToWatchedJob(
  runtime: RuntimeModel,
  job: Job,
  activeSaveId: string | null,
) {
  return runtime.active_save_id === activeSaveId
    && (job.save_id === undefined || job.save_id === null || runtime.active_save_id === job.save_id);
}

function chatTurnDeltaBelongsToWatchedJob(
  delta: ChatTurnDelta,
  job: Job,
  activeSaveId: string | null,
) {
  return delta.save_id === activeSaveId
    && (job.save_id === undefined || job.save_id === null || delta.save_id === job.save_id);
}

function worldTimeLabel(worldTime: RuntimeWorldTime | null | undefined): string {
  const display = worldTime?.display?.trim();
  if (display) return display;
  return "Time unknown";
}

function worldDayInputValue(worldTime: RuntimeWorldTime | null | undefined): string {
  return worldTime?.day_index === null || worldTime?.day_index === undefined
    ? ""
    : String(worldTime.day_index);
}

function parsedWorldDayIndex(value: string): number | null {
  const trimmed = value.trim();
  return trimmed ? Number(trimmed) : null;
}

function worldDayIndexIsInvalid(value: string): boolean {
  const trimmed = value.trim();
  if (!trimmed) return false;
  if (!/^\d+$/.test(trimmed)) return true;
  const parsed = Number(trimmed);
  return !Number.isSafeInteger(parsed);
}

function WorldTimeControl({
  worldTime,
  activeSaveId,
  onRuntimeChanged
}: {
  worldTime?: RuntimeWorldTime | null;
  activeSaveId: string | null;
  onRuntimeChanged: (model: RuntimeModel) => void;
}) {
  const [editingSaveId, setEditingSaveId] = useState<string | null>(null);
  const display = worldTimeLabel(worldTime);
  useEffect(() => {
    if (editingSaveId && editingSaveId !== activeSaveId) setEditingSaveId(null);
  }, [activeSaveId, editingSaveId]);
  if (!activeSaveId && !worldTime) return null;
  return (
    <div className="world-time-control">
      <button
        type="button"
        className="world-time-chip"
        title="Correct world time"
        aria-label="Correct world time"
        disabled={!activeSaveId}
        onClick={() => setEditingSaveId(activeSaveId)}
      >
        <Clock size={16} aria-hidden="true" />
        <span>World time</span>
        <strong>{display}</strong>
      </button>
      {editingSaveId && editingSaveId === activeSaveId ? (
        <WorldTimeEditorDialog
          worldTime={worldTime}
          activeSaveId={editingSaveId}
          onClose={() => setEditingSaveId(null)}
          onSaved={onRuntimeChanged}
        />
      ) : null}
    </div>
  );
}

function WorldTimeEditorDialog({
  worldTime,
  activeSaveId,
  onClose,
  onSaved
}: {
  worldTime?: RuntimeWorldTime | null;
  activeSaveId: string | null;
  onClose: () => void;
  onSaved: (model: RuntimeModel) => void;
}) {
  const titleId = React.useId();
  const [dayLabel, setDayLabel] = useState(worldTime?.day_label ?? "");
  const [phase, setPhase] = useState(worldTime?.phase ?? "");
  const [worldDay, setWorldDay] = useState(worldDayInputValue(worldTime));
  const worldDayInvalid = worldDayIndexIsInvalid(worldDay);
  const updateWorldTime = useMutation({
    mutationFn: () => postJson<RuntimeModel>("/api/runtime/world-time", {
      save_id: activeSaveId,
      day_index: parsedWorldDayIndex(worldDay),
      day_label: dayLabel.trim(),
      phase
    }),
    onSuccess: (model) => {
      onSaved(model);
      onClose();
    }
  });

  function submit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!activeSaveId || worldDayInvalid) return;
    updateWorldTime.mutate();
  }

  return (
    <ModalBackdrop>
      <DialogForm className="preview-dialog world-time-dialog" titleId={titleId} onClose={onClose} onSubmit={submit}>
        <header>
          <h2 id={titleId}>Correct world time</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <div className="world-time-grid">
          <label className="world-time-field">
            <span>Day label</span>
            <input
              autoFocus
              value={dayLabel}
              maxLength={80}
              onChange={(event) => setDayLabel(event.currentTarget.value)}
            />
          </label>
          <label className="world-time-field">
            <span>Phase</span>
            <select value={phase} onChange={(event) => setPhase(event.currentTarget.value)}>
              <option value="">Unknown</option>
              {WORLD_TIME_OF_DAY_OPTIONS.map((option) => (
                <option key={option} value={option}>{labelize(option)}</option>
              ))}
            </select>
          </label>
          <label className="world-time-field">
            <span>Day index</span>
            <input
              inputMode="numeric"
              min="0"
              pattern="[0-9]*"
              value={worldDay}
              onChange={(event) => setWorldDay(event.currentTarget.value)}
            />
          </label>
        </div>
        {worldDayInvalid ? <InlineNotice>World day must be zero or greater.</InlineNotice> : null}
        {updateWorldTime.error ? (
          <InlineNotice>{updateWorldTime.error instanceof Error ? updateWorldTime.error.message : "Could not save world time"}</InlineNotice>
        ) : null}
        <div className="command-row end">
          <button type="button" onClick={onClose}>Cancel</button>
          <button
            type="submit"
            className="primary-command compact"
            disabled={!activeSaveId || worldDayInvalid || updateWorldTime.isPending}
          >
            {updateWorldTime.isPending ? <Loader2 className="spin" size={16} aria-hidden="true" /> : null}
            {updateWorldTime.isPending ? "Saving" : "Save time"}
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function Workbench({
  currentUser = null,
  onLogout
}: {
  currentUser?: CurrentUser | null;
  onLogout?: () => void;
}) {
  const client = useQueryClient();
  const [panel, setPanel] = useState<PanelName>("media");
  const panelRef = useRef(panel);
  panelRef.current = panel;
  const [draftOpen, setDraftOpen] = useState(false);
  const [phoneOpen, setPhoneOpen] = useState(false);
  const [lookAroundOpen, setLookAroundOpen] = useState(false);
  const [lookAroundInitialQuery, setLookAroundInitialQuery] = useState("");
  const [lookAroundAnswer, setLookAroundAnswer] = useState<LookAroundAnswer | null>(null);
  const [topbarExpanded, setTopbarExpanded] = useState(true);
  const [draftInitialMode, setDraftInitialMode] = useState<"manual" | "draft">("manual");
  const [draftPrefill, setDraftPrefill] = useState<ScenarioDraftPrefill | null>(null);
  const [mobileSheet, setMobileSheet] = useState<MobileSheetName | null>(null);
  const [trackedJobs, setTrackedJobs] = useState<Record<string, TrackedJob>>({});
  const [narratorDrafts, setNarratorDrafts] = useState<Record<string, NarratorDraft>>({});
  const [scenarioRefreshVersion, setScenarioRefreshVersion] = useState(0);
  const [saveExportStates, setSaveExportStates] = useState<SaveExportStates>({});
  const saveExportRecoveryDeadlineRef = useRef<Map<string, number>>(new Map());
  const saveExportRecoveryMaxDeadlineRef = useRef<Map<string, number>>(new Map());
  const consumedSaveExportRecoveryRef = useRef<Set<string>>(new Set());
  const saveExportRecoveryGenerationRef = useRef<Map<string, number>>(new Map());
  const jobWatchers = useRef<Record<string, () => void>>({});
  const jobRunOptionsRef = useRef<Record<string, RunJobOptions>>({});
  const queuedRefreshesRef = useRef<Map<string, QueuedWorkbenchRefresh>>(new Map());
  const refreshFlushTimerRef = useRef<number | null>(null);
  const runtimeFreshTimerRef = useRef<number | null>(null);
  const runtimeFreshUntilRef = useRef(0);
  const runtimeFreshSaveIdRef = useRef<string | null>(null);
  const activeSaveIdRef = useRef<string | null>(null);
  const hasTrackedChatJobRef = useRef(false);
  const currentUserId = currentUser?.id ?? null;
  const [selectedSaveId, setSelectedSaveId] = useState<string | null>(() => loadSelectedSaveId(currentUserId));
  const [paintedRuntimeSaveKey, setPaintedRuntimeSaveKey] = useState<string | null>(null);
  const [seenTextMessageIdsByThread, setSeenTextMessageIdsByThread] = useState<Record<string, string>>({});
  const [pendingSaveId, setPendingSaveId] = useState<string | null>(null);
  const [saveSelectionError, setSaveSelectionError] = useState("");
  const [pendingMessage, setPendingMessage] = useState<PendingChronicleMessage | null>(null);
  const [narratorPaintMeasurement, setNarratorPaintMeasurement] = useState<NarratorPaintMeasurement | null>(null);
  const [layout, setLayout] = useState<WorkbenchLayout>(loadWorkbenchLayout);
  const [resizingSide, setResizingSide] = useState<ResizeSide | null>(null);
  const layoutDrag = useRef<{ side: ResizeSide; startX: number; startLayout: WorkbenchLayout } | null>(null);
  const isStackedWorkbench = useMediaQuery(WORKBENCH_STACKED_QUERY);
  const isMobileWorkbench = useMediaQuery(WORKBENCH_MOBILE_QUERY);
  const hasTrackedChatJob = Object.values(trackedJobs).some(({ job }) => isChatJobType(job.type));
  hasTrackedChatJobRef.current = hasTrackedChatJob;
  const runtime = useQuery({
    queryKey: runtimeQueryKey(selectedSaveId),
    queryFn: async ({ signal }) => {
      try {
        return await apiRead<RuntimeModel>(runtimePath(selectedSaveId), signal);
      } catch (failure) {
        if (selectedSaveId && failure instanceof ApiError && failure.status === 404) {
          setSelectedSaveId(null);
          setSaveSelectionError("Selected save is no longer available.");
          return await apiRead<RuntimeModel>(runtimePath(null), signal);
        }
        throw failure;
      }
    },
    placeholderData: keepPreviousData,
    retry: (failureCount, failure) => (
      !(failure instanceof ApiError && failure.status < 500) && failureCount < 3
    )
  });
  const model = runtime.data;
  const runtimeLoadError = runtime.error instanceof Error ? runtime.error.message : "";
  const activeSaveId = model?.active_save_id ?? selectedSaveId ?? null;
  const saveExportRecoveryIds = useMemo(() => {
    const ids = (model?.saves ?? []).map((save) => save.save_id);
    if (activeSaveId && !ids.includes(activeSaveId)) ids.unshift(activeSaveId);
    return ids;
  }, [activeSaveId, model?.saves]);
  useEffect(() => {
    const now = Date.now();
    const retainedIds = new Set(saveExportRecoveryIds);
    for (const saveId of saveExportRecoveryIds) {
      if (!saveExportRecoveryDeadlineRef.current.has(saveId)) {
        saveExportRecoveryDeadlineRef.current.set(saveId, now + SAVE_EXPORT_RECOVERY_WINDOW_MS);
      }
      if (!saveExportRecoveryMaxDeadlineRef.current.has(saveId)) {
        saveExportRecoveryMaxDeadlineRef.current.set(saveId, now + SAVE_EXPORT_RECOVERY_MAX_WINDOW_MS);
      }
    }
    for (const saveId of saveExportRecoveryDeadlineRef.current.keys()) {
      if (!retainedIds.has(saveId)) {
        saveExportRecoveryDeadlineRef.current.delete(saveId);
        saveExportRecoveryMaxDeadlineRef.current.delete(saveId);
      }
    }
  }, [saveExportRecoveryIds]);
  const activeSave = model?.saves?.find((save) => save.save_id === activeSaveId);
  const activeSaveSupported = activeSave?.supported !== false;
  useEffect(() => {
    if (!activeSaveSupported) setPhoneOpen(false);
  }, [activeSaveSupported]);
  activeSaveIdRef.current = activeSaveId;
  const activeRuntimeSaveKey = activeSaveId ?? "__none__";
  useEffect(() => {
    if (model) setPaintedRuntimeSaveKey(activeRuntimeSaveKey);
  }, [activeRuntimeSaveKey, model]);
  const shellSettings = useQuery({
    queryKey: ["settings", "shell"],
    queryFn: ({ signal }) => apiRead<ShellSettingsModel>("/api/settings/shell", signal),
    staleTime: 30_000
  });
  const activeJobs = useQuery({
    queryKey: ["jobs", "active", activeSaveId],
    queryFn: ({ signal }) => apiRead<{ jobs: Job[] }>(activeJobsPath(activeSaveId), signal),
    enabled: Boolean(model),
    retry: false
  });
  const readySaveExports = useQueries({
    queries: saveExportRecoveryIds.map((saveId) => ({
      queryKey: ["export-ready", saveId],
      queryFn: async ({ signal }: { signal: AbortSignal }) => {
        const generation = saveExportRecoveryGenerationRef.current.get(saveId) ?? 0;
        const ready = await apiRead<SaveExportReady>(
          `/api/bundles/export/ready?save_id=${encodeURIComponent(saveId)}`,
          signal,
        );
        if (
          (saveExportRecoveryGenerationRef.current.get(saveId) ?? 0) !== generation
          || consumedSaveExportRecoveryRef.current.has(saveId)
        ) {
          return { active: false, export: null };
        }
        if (ready.active) {
          const now = Date.now();
          const maximum = saveExportRecoveryMaxDeadlineRef.current.get(saveId)
            ?? now + SAVE_EXPORT_RECOVERY_MAX_WINDOW_MS;
          saveExportRecoveryMaxDeadlineRef.current.set(saveId, maximum);
          saveExportRecoveryDeadlineRef.current.set(
            saveId,
            Math.min(now + SAVE_EXPORT_RECOVERY_WINDOW_MS, maximum),
          );
        }
        return ready;
      },
      enabled: Boolean(model && canUseChildRestrictedControls(currentUser)),
      refetchInterval: (query: { state: { data?: SaveExportReady } }) => (
        !query.state.data?.export
        && (
          saveId === activeSaveId
          || saveExportStates[saveId] === "pending"
          || query.state.data?.active === true
        )
        && !consumedSaveExportRecoveryRef.current.has(saveId)
        && Date.now() < (saveExportRecoveryDeadlineRef.current.get(saveId) ?? 0)
      ) ? 3_000 : false,
      retry: false,
    })),
  });
  const clearSaveExportRecovery = useCallback((saveId: string, action: SaveExportRecoveryAction) => {
    const generation = saveExportRecoveryGenerationRef.current.get(saveId) ?? 0;
    saveExportRecoveryGenerationRef.current.set(saveId, generation + 1);
    const now = Date.now();
    if (action !== "restart") {
      consumedSaveExportRecoveryRef.current.add(saveId);
      saveExportRecoveryDeadlineRef.current.set(saveId, now);
      saveExportRecoveryMaxDeadlineRef.current.set(saveId, now);
    } else {
      consumedSaveExportRecoveryRef.current.delete(saveId);
      saveExportRecoveryDeadlineRef.current.set(saveId, now + SAVE_EXPORT_RECOVERY_WINDOW_MS);
      saveExportRecoveryMaxDeadlineRef.current.set(saveId, now + SAVE_EXPORT_RECOVERY_MAX_WINDOW_MS);
    }
    void client.cancelQueries({ queryKey: ["export-ready", saveId], exact: true });
    client.setQueryData(["export-ready", saveId], { active: false, export: null });
  }, [client]);
  useEffect(() => {
    setSaveExportStates((current) => {
      let next = current;
      for (const [index, saveId] of saveExportRecoveryIds.entries()) {
        const ready = readySaveExports[index]?.data;
        if (!ready) continue;
        const recovered = ready.export;
        const existing = next[saveId];
        if (!isChatBundleExportResult(recovered)) {
          if (ready.active || !isChatBundleExportResult(existing)) continue;
          if (next === current) next = { ...current };
          delete next[saveId];
          continue;
        }
        if (existing !== undefined && existing !== "pending") continue;
        if (next === current) next = { ...current };
        next[saveId] = recovered;
      }
      return next;
    });
  }, [readySaveExports, saveExportRecoveryIds]);
  const chatSubmissionStatus = useQuery({
    queryKey: ["chat", "submission-status", activeSaveId],
    queryFn: ({ signal }) => apiRead<ChatSubmissionStatus>(chatSubmissionStatusPath(activeSaveId), signal),
    enabled: Boolean(model),
    retry: false
  });
  const chatTimingSummary = useQuery({
    queryKey: ["chat", "timing-summary", activeSaveId],
    queryFn: ({ signal }) => apiRead<ChatTimingSummary>(chatTimingSummaryPath(activeSaveId!), signal),
    enabled: Boolean(model && activeSaveId),
    staleTime: 30_000,
    retry: false
  });
  const characterTextsSummary = useQuery({
    queryKey: ["character-texts", activeSaveId],
    queryFn: ({ signal }) => apiRead<CharacterTextsModel>(characterTextsPath(activeSaveId), signal),
    enabled: Boolean(
      activeSaveId
      && model?.character_texts_enabled
      && paintedRuntimeSaveKey === activeRuntimeSaveKey
    ),
    retry: false
  });
  const unreadCharacterTextContacts = incomingCharacterTextContacts(
    characterTextsSummary.data,
    seenTextMessageIdsByThread,
  );
  const unreadCharacterTextCount = unreadCharacterTextContacts.length;
  const markCharacterTextThreadSeen = useCallback((threadId: string | null | undefined, messageId: string | null | undefined) => {
    if (!threadId || !messageId) return;
    setSeenTextMessageIdsByThread((current) => {
      if (current[threadId] === messageId) return current;
      const next = { ...current, [threadId]: messageId };
      saveCharacterTextSeenState(currentUserId, activeSaveIdRef.current, next);
      return next;
    });
  }, [currentUserId]);

  const refreshScenarioLibrary = useCallback(() => {
    setScenarioRefreshVersion((current) => current + 1);
  }, []);

  const applyWorkbenchRefreshTargets = useCallback((saveId: string | null, targets: Iterable<WorkbenchRefreshTarget>) => {
    const uniqueTargets = new Set(targets);
    for (const target of uniqueTargets) {
      if (target === "runtime") {
        if (runtimeFreshSaveIdRef.current === saveId && runtimeFreshUntilRef.current > Date.now()) continue;
        client.invalidateQueries({ queryKey: runtimeQueryKey(saveId) });
      }
      if (target === "scenarios") {
        client.invalidateQueries({ queryKey: ["scenarios"] });
        refreshScenarioLibrary();
      }
      if (target === "worlds") {
        client.invalidateQueries({ queryKey: ["persistent-worlds"] });
      }
      if (target === "world") {
        client.invalidateQueries({ queryKey: saveId ? ["world", saveId] : ["world"] });
      }
      if (target === "characters") {
        client.invalidateQueries({ queryKey: saveId ? ["characters", saveId] : ["characters"] });
      }
      if (target === "scene-presence") {
        invalidateScenePresenceQueries(client, saveId);
      }
      if (target === "chat-history") {
        client.invalidateQueries({ queryKey: saveId ? ["chat-history", saveId] : ["chat-history"] });
      }
      if (target === "character-texts") {
        client.invalidateQueries({ queryKey: saveId ? ["character-texts", saveId] : ["character-texts"] });
      }
      if (target === "character-text-thread") {
        client.invalidateQueries({ queryKey: saveId ? ["character-text-thread", saveId] : ["character-text-thread"] });
      }
      if (target === "jobs-active") {
        client.invalidateQueries({ queryKey: ["jobs", "active", saveId] });
      }
      if (target === "chat-status") {
        client.invalidateQueries({ queryKey: ["chat", "submission-status", saveId] });
      }
      if (target === "chat-timing") {
        client.invalidateQueries({ queryKey: ["chat", "timing-summary", saveId] });
      }
      if (target === "settings") {
        client.invalidateQueries({ queryKey: ["settings"] });
      }
      if (target === "media") {
        client.invalidateQueries({ queryKey: saveId ? ["media", saveId] : ["media"] });
      }
    }
  }, [client, refreshScenarioLibrary]);

  const flushQueuedRefreshes = useCallback(() => {
    refreshFlushTimerRef.current = null;
    const queued = Array.from(queuedRefreshesRef.current.values());
    queuedRefreshesRef.current.clear();
    for (const refresh of queued) {
      applyWorkbenchRefreshTargets(refresh.saveId, refresh.targets);
    }
  }, [applyWorkbenchRefreshTargets]);

  const queueWorkbenchRefresh = useCallback((saveId: string | null, targets: Iterable<WorkbenchRefreshTarget>) => {
    const key = saveId ?? "__global__";
    const current = queuedRefreshesRef.current.get(key);
    if (current) {
      for (const target of targets) current.targets.add(target);
    } else {
      queuedRefreshesRef.current.set(key, { saveId, targets: new Set(targets) });
    }
    if (refreshFlushTimerRef.current !== null) return;
    refreshFlushTimerRef.current = window.setTimeout(flushQueuedRefreshes, WORKBENCH_REFRESH_DEBOUNCE_MS);
  }, [flushQueuedRefreshes]);

  const refreshWorkbench = useCallback((
    saveId: string | null = activeSaveIdRef.current,
    targets: Iterable<WorkbenchRefreshTarget> = ALL_WORKBENCH_REFRESH_TARGETS,
  ) => {
    queueWorkbenchRefresh(saveId, targets);
  }, [queueWorkbenchRefresh]);

  const noteRuntimeModelFresh = useCallback((saveId: string | null) => {
    if (runtimeFreshTimerRef.current !== null) window.clearTimeout(runtimeFreshTimerRef.current);
    const freshUntil = Date.now() + RUNTIME_FRESH_SUPPRESS_MS;
    runtimeFreshSaveIdRef.current = saveId;
    runtimeFreshUntilRef.current = freshUntil;
    runtimeFreshTimerRef.current = window.setTimeout(() => {
      runtimeFreshTimerRef.current = null;
      runtimeFreshSaveIdRef.current = null;
      runtimeFreshUntilRef.current = 0;
    }, RUNTIME_FRESH_SUPPRESS_MS);
  }, []);

  const refreshForSaveEvent = useCallback((event: SaveEvent) => {
    const saveId = event.save_id ?? activeSaveIdRef.current;
    if (event.type === "job_changed") {
      const changedJob = jobFromSaveEvent(event);
      if (!changedJob || !saveId) {
        queueWorkbenchRefresh(saveId, ["jobs-active", "chat-status"]);
        return;
      }
      const queryKey = ["jobs", "active", saveId] as const;
      const currentJobs = client.getQueryData<{ jobs: Job[] }>(queryKey)?.jobs ?? [];
      const existing = currentJobs.find((job) => job.id === changedJob.id);
      const terminal = ["succeeded", "failed", "cancelled"].includes(changedJob.status);
      let stoppedWatcher = false;
      void client.cancelQueries({ queryKey, exact: true });
      client.setQueryData<{ jobs: Job[] }>(queryKey, (current) => {
        const jobs = current?.jobs ?? [];
        if (terminal) {
          return { jobs: jobs.filter((job) => job.id !== changedJob.id) };
        }
        const matched = jobs.some((job) => job.id === changedJob.id);
        return {
          jobs: matched
            ? jobs.map((job) => job.id === changedJob.id ? { ...job, ...changedJob } : job)
            : [...jobs, changedJob],
        };
      });
      if (terminal) {
        if (!jobRunOptionsRef.current[changedJob.id]?.onFailed) {
          stoppedWatcher = Object.prototype.hasOwnProperty.call(jobWatchers.current, changedJob.id);
          if (stoppedWatcher) jobWatchers.current[changedJob.id]();
          delete jobWatchers.current[changedJob.id];
          delete jobRunOptionsRef.current[changedJob.id];
        }
        setTrackedJobs((current) => {
          if (!(changedJob.id in current)) return current;
          const next = { ...current };
          delete next[changedJob.id];
          return next;
        });
        setNarratorDrafts((current) => {
          if (!(changedJob.id in current)) return current;
          const next = { ...current };
          delete next[changedJob.id];
          return next;
        });
        if (stoppedWatcher && isChatJobType(changedJob.type)) {
          setPendingMessage((current) => (
            current?.pending_save_id === saveId ? null : current
          ));
        }
      }
      if (!terminal) {
        setTrackedJobs((current) => {
          const tracked = current[changedJob.id];
          if (!tracked) return current;
          return {
            ...current,
            [changedJob.id]: trackedActiveJob(changedJob, tracked),
          };
        });
      }
      if (terminal || (!existing && isChatJobType(changedJob.type))) {
        queueWorkbenchRefresh(
          saveId,
          terminal && stoppedWatcher && isChatJobType(changedJob.type)
            ? ["runtime", "jobs-active", "chat-status"]
            : ["jobs-active", "chat-status"],
        );
      }
      if (terminal && mediaChangingJob(changedJob)) {
        queueWorkbenchRefresh(saveId, ["media"]);
      }
      return;
    }
    queueWorkbenchRefresh(saveId, saveEventRefreshTargets(event, panelRef.current));
  }, [client, queueWorkbenchRefresh]);

  const applyRuntimeModel = useCallback((nextModel: RuntimeModel, fallbackSaveId: string | null = activeSaveIdRef.current) => {
    const nextSaveId = nextModel.active_save_id ?? fallbackSaveId;
    if (nextSaveId) setSelectedSaveId(nextSaveId);
    setSaveSelectionError("");
    client.setQueryData(runtimeQueryKey(nextSaveId), nextModel);
    noteRuntimeModelFresh(nextSaveId);
    queueWorkbenchRefresh(nextSaveId, RUNTIME_MODEL_SIDE_EFFECT_REFRESH_TARGETS);
  }, [client, noteRuntimeModelFresh, queueWorkbenchRefresh]);

  const applyChatTurnDelta = useCallback((delta: ChatTurnDelta) => {
    if (delta.requires_full_refresh) return false;
    if (delta.save_id !== activeSaveIdRef.current) return false;
    let applied = false;
    client.setQueryData<RuntimeModel>(runtimeQueryKey(delta.save_id), (current) => {
      if (!current || current.active_save_id !== delta.save_id) return current;
      applied = true;
      return applyChatTurnDeltaToRuntimeModel(current, delta);
    });
    if (!applied) return false;
    setSelectedSaveId(delta.save_id);
    setSaveSelectionError("");
    noteRuntimeModelFresh(delta.save_id);
    queueWorkbenchRefresh(delta.save_id, CHAT_TURN_DELTA_REFRESH_TARGETS);
    return true;
  }, [client, noteRuntimeModelFresh, queueWorkbenchRefresh]);

  useEffect(() => {
    saveSelectedSaveId(selectedSaveId, currentUserId);
  }, [currentUserId, selectedSaveId]);

  useEffect(() => {
    setSeenTextMessageIdsByThread(
      loadCharacterTextSeenState(currentUserId, activeSaveId),
    );
  }, [activeSaveId, currentUserId]);

  useEffect(() => {
    if (!model) return;
    const modelSaveIds = new Set((model.saves ?? []).map((save) => save.save_id));
    if (!selectedSaveId && model.active_save_id) {
      setSelectedSaveId(model.active_save_id);
      return;
    }
    if (selectedSaveId && !modelSaveIds.has(selectedSaveId)) {
      setSelectedSaveId(model.active_save_id ?? null);
      setSaveSelectionError("Selected save is no longer available.");
    }
  }, [model, selectedSaveId]);

  useEffect(() => {
    if (!activeSaveId) return undefined;
    return watchSave(activeSaveId, refreshForSaveEvent, () => {
      refreshWorkbench(activeSaveId);
    }, async (signal) => {
      async function refreshFallback<T>(queryKey: readonly unknown[], path: string) {
        const value = await apiRead<T>(path, signal);
        if (!signal.aborted && activeSaveIdRef.current === activeSaveId) {
          client.setQueryData(queryKey, value);
        }
      }
      const refreshes: Promise<unknown>[] = [
        refreshFallback<{ jobs: Job[] }>(
          ["jobs", "active", activeSaveId],
          activeJobsPath(activeSaveId)
        ),
        refreshFallback<ChatSubmissionStatus>(
          ["chat", "submission-status", activeSaveId],
          chatSubmissionStatusPath(activeSaveId)
        )
      ];
      if (
        hasTrackedChatJobRef.current
        && runtimeFreshUntilRef.current <= Date.now()
      ) {
        refreshes.push(refreshFallback<RuntimeModel>(
          runtimeQueryKey(activeSaveId),
          runtimePath(activeSaveId)
        ));
      }
      const results = await Promise.allSettled(refreshes);
      return results.every((result) => result.status === "fulfilled");
    });
  }, [activeSaveId, client, refreshForSaveEvent, refreshWorkbench]);

  useEffect(() => {
    setPendingMessage((current) => pendingMessageForActiveSave(current, activeSaveId));
    setNarratorPaintMeasurement(null);
    setLookAroundAnswer(null);
  }, [activeSaveId]);

  const openScenarioDialog = (mode: "manual" | "draft" = "manual") => {
    setDraftPrefill(null);
    setDraftInitialMode(mode);
    setDraftOpen(true);
  };
  const reuseScenarioPrompt = useCallback(async (scenario: Scenario) => {
    const definition = await api<WorldDataModel>(`/api/scenarios/${scenario.scenario_id}/definition`);
    const prompt = definition.scenario?.generation_prompt?.trim();
    if (!prompt) throw new Error("Scenario does not have a saved AI prompt.");
    const scenarioType = definition.scenario?.scenario_type || scenario.scenario_type;
    const persistentWorldId = Object.prototype.hasOwnProperty.call(definition, "persistent_world")
      ? definition.persistent_world?.world_id ?? null
      : scenario.persistent_world_id ?? null;
    setDraftPrefill({
      scenario_type: scenarioType,
      scenario_types: normalizedScenarioTypes(scenarioType, scenario.scenario_types),
      action_choices_enabled: Boolean(scenario.action_choices_enabled),
      interaction_mode: definition.scenario?.interaction_mode
        ?? scenario.interaction_mode
        ?? "roleplay",
      persistent_world_id: persistentWorldId,
      seed: prompt
    });
    setDraftInitialMode("draft");
    setDraftOpen(true);
  }, []);
  const openLookAround = useCallback((initialQuery = "") => {
    setLookAroundInitialQuery(initialQuery);
    setLookAroundAnswer(null);
    setLookAroundOpen(true);
  }, []);

  const selectSave = useCallback(async (saveId: string) => {
    setPendingSaveId(saveId);
    setSaveSelectionError("");
    try {
      const nextModel = await postJson<RuntimeModel>(`/api/saves/${saveId}/load`, {});
      applyRuntimeModel(nextModel, saveId);
      return true;
    } catch (failure) {
      setSaveSelectionError(failure instanceof Error ? failure.message : "Could not load save");
      return false;
    } finally {
      setPendingSaveId((current) => (current === saveId ? null : current));
    }
  }, [applyRuntimeModel]);

  const setLayoutWidth = useCallback((side: ResizeSide, width: number) => {
    setLayout((current) => constrainWorkbenchLayout({
      leftRailWidth: side === "left" ? width : current.leftRailWidth,
      rightPanelWidth: side === "right" ? width : current.rightPanelWidth
    }, side));
  }, []);

  const onLayoutPointerMove = useCallback((event: PointerEvent) => {
    const drag = layoutDrag.current;
    if (!drag) return;
    const delta = event.clientX - drag.startX;
    setLayoutWidth(
      drag.side,
      drag.side === "left"
        ? drag.startLayout.leftRailWidth + delta
        : drag.startLayout.rightPanelWidth - delta
    );
  }, [setLayoutWidth]);

  const stopLayoutResize = useCallback(() => {
    layoutDrag.current = null;
    setResizingSide(null);
  }, []);

  const startLayoutResize = useCallback((side: ResizeSide, event: React.PointerEvent<HTMLDivElement>) => {
    if (event.button !== 0) return;
    event.preventDefault();
    layoutDrag.current = { side, startX: event.clientX, startLayout: layout };
    setResizingSide(side);
  }, [layout]);

  const onLayoutKeyDown = useCallback((side: ResizeSide, event: React.KeyboardEvent<HTMLDivElement>) => {
    const currentWidth = side === "left" ? layout.leftRailWidth : layout.rightPanelWidth;
    const limits = WORKBENCH_LAYOUT_LIMITS[side];
    const step = event.shiftKey ? WORKBENCH_RESIZE_FAST_STEP : WORKBENCH_RESIZE_STEP;
    let nextWidth: number | null = null;

    if (event.key === "ArrowLeft") nextWidth = currentWidth + (side === "left" ? -step : step);
    if (event.key === "ArrowRight") nextWidth = currentWidth + (side === "left" ? step : -step);
    if (event.key === "Home") nextWidth = limits.min;
    if (event.key === "End") nextWidth = limits.max;

    if (nextWidth === null) return;
    event.preventDefault();
    setLayoutWidth(side, nextWidth);
  }, [layout.leftRailWidth, layout.rightPanelWidth, setLayoutWidth]);

  useEffect(() => {
    saveWorkbenchLayout(layout);
  }, [layout]);

  useEffect(() => {
    const onWindowResize = () => setLayout((current) => constrainWorkbenchLayout(current));
    window.addEventListener("resize", onWindowResize);
    return () => window.removeEventListener("resize", onWindowResize);
  }, []);

  useEffect(() => {
    if (!resizingSide) return undefined;
    window.addEventListener("pointermove", onLayoutPointerMove);
    window.addEventListener("pointerup", stopLayoutResize);
    window.addEventListener("pointercancel", stopLayoutResize);
    return () => {
      window.removeEventListener("pointermove", onLayoutPointerMove);
      window.removeEventListener("pointerup", stopLayoutResize);
      window.removeEventListener("pointercancel", stopLayoutResize);
    };
  }, [onLayoutPointerMove, resizingSide, stopLayoutResize]);

  const runJob = useCallback<RunJob>((created, requestedOptions) => {
    if (
      !jobBelongsToActiveSave(created, activeSaveIdRef.current)
      && requestedOptions?.allowInactiveSave !== true
    ) {
      delete jobRunOptionsRef.current[created.id];
      return () => undefined;
    }
    const priorOptions = jobRunOptionsRef.current[created.id];
    if (jobWatchers.current[created.id]) {
      if (requestedOptions && !requestedOptions.recovered) {
        jobRunOptionsRef.current[created.id] = {
          ...priorOptions,
          ...requestedOptions,
          applyResult: requestedOptions.applyResult ?? true,
          recovered: false,
        };
      }
      return jobWatchers.current[created.id];
    }
    const options = requestedOptions
      ? { ...priorOptions, ...requestedOptions }
      : priorOptions ?? { applyResult: true };
    jobRunOptionsRef.current[created.id] = options;
    const currentOptions = () => jobRunOptionsRef.current[created.id] ?? options;
    let narratorPaintRequested = false;
    const requestNarratorPaint = (result: unknown) => {
      const activeOptions = currentOptions();
      if (
        created.type !== "chat_turn"
        || activeOptions.paintStartedAtMs === undefined
        || narratorPaintRequested
      ) return;
      const messageId = narratorMessageIdFromResult(result);
      const saveId = created.save_id ?? activeSaveIdRef.current;
      if (!messageId || !saveId) return;
      narratorPaintRequested = true;
      setNarratorPaintMeasurement({
        jobId: created.id,
        messageId,
        saveId,
        startedAtMs: activeOptions.paintStartedAtMs,
      });
    };
    if (created.status !== "queued" && created.status !== "running") {
      const activeOptions = currentOptions();
      if (created.status === "succeeded") {
        requestNarratorPaint(created.result);
        activeOptions.onSucceeded?.(created.result);
      }
      if (created.status === "failed") {
        activeOptions.onFailed?.(created.error || "Background job failed.", created);
      }
      activeOptions.onFinished?.(created);
      if (activeOptions.clearPendingMessages !== false) setPendingMessage(null);
      delete jobRunOptionsRef.current[created.id];
      refreshWorkbench(
        activeSaveIdRef.current,
        mediaChangingJob(created)
          ? ["jobs-active", "media"]
          : ALL_WORKBENCH_REFRESH_TARGETS,
      );
      return () => undefined;
    }
    setTrackedJobs((current) => {
      const existing = current[created.id];
      return {
        ...current,
        [created.id]: trackedActiveJob(created, existing)
      };
    });
    const stop = watchJob(
      created.id,
      (done) => {
        const activeOptions = currentOptions();
        const appliesToCurrentSave = jobBelongsToActiveSave(done, activeSaveIdRef.current);
        let appliedRuntimeResult = false;
        let appliedChatDelta = false;
        setTrackedJobs((current) => {
          const next = { ...current };
          delete next[done.id];
          return next;
        });
        setNarratorDrafts((current) => {
          if (!(done.id in current)) return current;
          const next = { ...current };
          delete next[done.id];
          return next;
        });
        delete jobWatchers.current[done.id];
        if (appliesToCurrentSave && activeOptions.applyResult !== false && done.status === "succeeded") {
          if (isRuntimeModel(done.result)) {
            appliedRuntimeResult = true;
            applyRuntimeModel(done.result);
            requestNarratorPaint(done.result);
          } else if (isChatTurnDelta(done.result)) {
            appliedChatDelta = applyChatTurnDelta(done.result);
            if (appliedChatDelta) requestNarratorPaint(done.result);
          }
        }
        if (appliesToCurrentSave && done.status === "succeeded") {
          applyCharacterTextJobResult(client, done.result, done.save_id ?? activeSaveIdRef.current);
        }
        const canCompleteAcrossSaves = (
          appliesToCurrentSave || activeOptions.allowCrossSaveCompletion === true
        );
        if (canCompleteAcrossSaves && done.status === "succeeded") {
          activeOptions.onSucceeded?.(done.result);
        }
        const isActionChoiceJob = (
          done.type === "action_choice_generate"
          || done.type === "action_choice_regenerate"
        );
        if (
          appliesToCurrentSave
          && isActionChoiceJob
          && (done.status === "failed" || done.status === "cancelled")
        ) {
          const error = done.status === "failed" ? done.error || "Background job failed." : null;
          appliedRuntimeResult = true;
          client.setQueryData<RuntimeModel>(
            runtimeQueryKey(done.save_id ?? activeSaveIdRef.current),
            (current) => current?.action_choices
              && (
                !current.action_choices.generation_job
                || current.action_choices.generation_job.id === done.id
              )
              ? {
                ...current,
                action_choices: {
                  ...current.action_choices,
                  generation_job: null,
                  generation_error: error
                }
              }
              : current
          );
        }
        if (canCompleteAcrossSaves && done.status === "failed") {
          const error = done.error || "Background job failed.";
          if (appliesToCurrentSave && isActionChoiceJob) {
            appliedRuntimeResult = true;
          }
          activeOptions.onFailed?.(error, done);
        }
        if (canCompleteAcrossSaves) activeOptions.onFinished?.(done);
        if (appliesToCurrentSave) {
          if (activeOptions.clearPendingMessages !== false) {
            setPendingMessage(null);
          }
          refreshWorkbench(
            activeSaveIdRef.current,
            mediaChangingJob(done)
              ? ["jobs-active", "media"]
              : appliedChatDelta
              ? CHAT_TURN_DELTA_REFRESH_TARGETS
              : appliedRuntimeResult
                ? RUNTIME_MODEL_SIDE_EFFECT_REFRESH_TARGETS
                : ALL_WORKBENCH_REFRESH_TARGETS,
          );
        }
        delete jobRunOptionsRef.current[done.id];
        if (done.type === "model_refresh") client.invalidateQueries({ queryKey: ["settings"] });
      },
      (name, data) => {
        if (jobWatchers.current[created.id] !== stop) return;
        if (
          name === "runtime"
          && isRuntimeModel(data)
          && runtimeEventBelongsToWatchedJob(data, created, activeSaveIdRef.current)
        ) {
          setPendingMessage(null);
          applyRuntimeModel(data);
          requestNarratorPaint(data);
          client.invalidateQueries({ queryKey: ["chat", "submission-status", data.active_save_id ?? activeSaveIdRef.current] });
        }
        if (
          name === "chat_turn_delta"
          && isChatTurnDelta(data)
          && chatTurnDeltaBelongsToWatchedJob(data, created, activeSaveIdRef.current)
        ) {
          setPendingMessage(null);
          setNarratorDrafts((current) => {
            if (!(created.id in current)) return current;
            const next = { ...current };
            delete next[created.id];
            return next;
          });
          if (applyChatTurnDelta(data)) {
            requestNarratorPaint(data);
          } else {
            refreshWorkbench(data.save_id, ALL_WORKBENCH_REFRESH_TARGETS);
          }
          client.invalidateQueries({ queryKey: ["chat", "submission-status", data.save_id] });
        }
        if (name === "narrator_draft" && isNarratorDraft(data) && jobBelongsToActiveSave(created, activeSaveIdRef.current)) {
          setNarratorDrafts((current) => ({ ...current, [created.id]: data }));
        }
        if (name === "progress") {
          const phases = postTurnProgressPhases(data);
          setTrackedJobs((current) => {
            const tracked = current[created.id];
            if (!tracked) return current;
            const replacesCatchup = isPostTurnCatchupProgress(data)
              || isPostTurnCatchupProgress(tracked.job.latest_progress);
            return {
              ...current,
              [created.id]: {
                ...tracked,
                job: { ...tracked.job, latest_progress: data },
                progress: progressLabel(data),
                phases: phases ?? (replacesCatchup ? undefined : tracked.phases)
              }
            };
          });
        }
        if (name === "completion_level" && isCompletionLevelEvent(data)) {
          setTrackedJobs((current) => {
            const tracked = current[created.id];
            if (!tracked) return current;
            return {
              ...current,
              [created.id]: {
                ...tracked,
                job: { ...tracked.job, completion_level: data.completion_level }
              }
            };
          });
        }
      },
      created.save_id ?? null,
      currentOptions().resumeFromEventCursor
    );
    jobWatchers.current[created.id] = stop;
    return stop;
  }, [applyChatTurnDelta, applyRuntimeModel, client, refreshWorkbench]);

  const startContinuationDraft = async (chapterStartInstructions = "") => {
    const created = await postJson<Job>("/api/scenarios/continuation-draft", {
      save_id: model?.active_save_id ?? null,
      chapter_start_instructions: chapterStartInstructions.trim()
    });
    runJob(created, {
      onSucceeded: (result) => {
        if (isRuntimeModel(result) && result.scenario_draft) {
          client.setQueryData(runtimeQueryKey(result.active_save_id ?? activeSaveIdRef.current), result);
          openScenarioDialog("draft");
        }
      }
    });
  };

  useEffect(() => {
    return () => {
      Object.values(jobWatchers.current).forEach((stop) => stop());
      jobWatchers.current = {};
      jobRunOptionsRef.current = {};
      if (refreshFlushTimerRef.current !== null) {
        window.clearTimeout(refreshFlushTimerRef.current);
        refreshFlushTimerRef.current = null;
      }
      if (runtimeFreshTimerRef.current !== null) {
        window.clearTimeout(runtimeFreshTimerRef.current);
        runtimeFreshTimerRef.current = null;
      }
      runtimeFreshSaveIdRef.current = null;
      runtimeFreshUntilRef.current = 0;
    };
  }, []);

  useEffect(() => {
    for (const active of activeJobs.data?.jobs ?? []) {
      runJob(active, {
        applyResult: false,
        clearPendingMessages: false,
        resumeFromEventCursor: active.event_cursor,
        recovered: true,
      });
    }
  }, [activeJobs.data?.jobs, runJob]);

  const openingActionChoiceJob = model?.action_choices?.generation_job;
  useEffect(() => {
    if (
      !openingActionChoiceJob
      || !["queued", "running"].includes(openingActionChoiceJob.status)
    ) {
      return;
    }
    runJob(openingActionChoiceJob, {
      applyResult: false,
      clearPendingMessages: false,
      resumeFromEventCursor: openingActionChoiceJob.event_cursor,
      recovered: true,
    });
  }, [
    openingActionChoiceJob?.id,
    openingActionChoiceJob?.status,
    openingActionChoiceJob?.event_cursor,
    runJob
  ]);

  useEffect(() => {
    setTrackedJobs((current) => {
      const next: Record<string, TrackedJob> = {};
      for (const [jobId, tracked] of Object.entries(current)) {
        if (
          jobBelongsToActiveSave(tracked.job, activeSaveId)
          || jobRunOptionsRef.current[jobId]?.allowCrossSaveCompletion === true
        ) {
          next[jobId] = tracked;
          continue;
        }
        jobWatchers.current[jobId]?.();
        delete jobWatchers.current[jobId];
        delete jobRunOptionsRef.current[jobId];
      }
      return next;
    });
    setNarratorDrafts((current) => {
      let pruned = false;
      const next: Record<string, NarratorDraft> = {};
      for (const [jobId, draft] of Object.entries(current)) {
        if (draft.save_id !== activeSaveId) {
          pruned = true;
          continue;
        }
        next[jobId] = draft;
      }
      return pruned ? next : current;
    });
  }, [activeSaveId]);

  useEffect(() => {
    setMobileSheet((current) => {
      if (isMobileWorkbench) return current;
      if (isStackedWorkbench) return current === "library" ? current : null;
      return null;
    });
  }, [isMobileWorkbench, isStackedWorkbench]);

  const cancelTrackedJob = async (tracked: TrackedJob) => {
    setTrackedJobs((current) => ({
      ...current,
      [tracked.job.id]: { ...tracked, progress: "Cancelling" }
    }));
    try {
      const jobSaveId = tracked.job.save_id ?? null;
      const saveQuery = jobSaveId ? `?save_id=${encodeURIComponent(jobSaveId)}` : "";
      await postJson(`/api/jobs/${encodeURIComponent(tracked.job.id)}/cancel${saveQuery}`, {});
    } catch (failure) {
      setTrackedJobs((current) => ({
        ...current,
        [tracked.job.id]: {
          ...tracked,
          progress: failure instanceof Error ? failure.message : "Cancel failed"
        }
      }));
    }
  };

  const pendingJobs = Object.values(trackedJobs)
    .filter((tracked) => jobBelongsToActiveSave(tracked.job, activeSaveId))
    .filter((tracked) => tracked.job.type !== "character_text_send")
    .sort((left, right) => (left.job.created_at ?? 0) - (right.job.created_at ?? 0));
  const busyCharacterTextThreadIds = new Set([
    ...(activeJobs.data?.jobs ?? []),
    ...Object.values(trackedJobs).map(({ job }) => job),
  ].flatMap((job) => (
    ["queued", "running"].includes(job.status)
      && job.scope?.kind === "character_text_thread"
      ? [job.scope.id]
      : []
  )));
  const sceneArrivalMessageIds = new Set(
    pendingJobs
      .map(({ job }) => sceneArrivalSourceMessageId(job))
      .filter((messageId): messageId is string => Boolean(messageId)),
  );
  const activeSaveChatBlockers = pendingJobs.filter(({ job }) => jobBlocksChatSubmission(job, activeSaveId));
  const backendChatBlocker = chatSubmissionStatus.data?.blocking_job_id
    ? pendingJobs.find((tracked) => tracked.job.id === chatSubmissionStatus.data?.blocking_job_id && isChatJobType(tracked.job.type))
    : undefined;
  const chatBlockerJobs = backendChatBlocker ? [backendChatBlocker] : activeSaveChatBlockers;
  const hasActiveSaveChatBlocker = chatBlockerJobs.length > 0;
  const chatSubmissionStatusLoadError = chatSubmissionStatus.error instanceof Error
    ? chatSubmissionStatus.error.message
    : "";
  const chatSubmissionStatusMissing = chatSubmissionStatus.isSuccess && !chatSubmissionStatus.data;
  const chatSubmissionStatusNotice = chatSubmissionStatusLoadError
    ? `Chat submission status could not be loaded. ${chatSubmissionStatusLoadError}`
    : chatSubmissionStatusMissing
      ? "Chat submission status could not be loaded."
      : "";
  const chatCanSubmit = chatSubmissionStatus.data?.can_submit === true && !chatSubmissionStatusNotice;
  const lookAroundJobInFlight = chatBlockerJobs.some(({ job }) => job.type === "look_around");
  const unrelatedChatBlockerActive = chatBlockerJobs.some(({ job }) => job.type !== "look_around");
  const lookAroundDisabled = !activeSaveId
    || !activeSaveSupported
    || !model?.composer_enabled
    || unrelatedChatBlockerActive
    || (!lookAroundJobInFlight && (hasActiveSaveChatBlocker || !chatCanSubmit));
  useEffect(() => {
    setLookAroundOpen(false);
    setLookAroundInitialQuery("");
    setLookAroundAnswer(null);
  }, [activeSaveId]);
  useEffect(() => {
    if (!lookAroundDisabled) return;
    setLookAroundOpen(false);
    setLookAroundInitialQuery("");
    setLookAroundAnswer(null);
  }, [lookAroundDisabled]);
  const pendingJobsDisplayMode = pendingJobsDisplayModeFromSettings(shellSettings.data?.pending_jobs_display_mode?.selected);
  const shellStyle: WorkbenchLayoutStyle = {
    "--left-rail-width": `${layout.leftRailWidth}px`,
    "--right-panel-width": `${layout.rightPanelWidth}px`
  };
  const shellClass = [
    "app-shell",
    resizingSide ? "is-resizing-layout" : "",
    isMobileWorkbench ? "mobile-app-shell" : ""
  ].filter(Boolean).join(" ");
  const isStackedDesktopWorkbench = isStackedWorkbench && !isMobileWorkbench;
  const openPanelSheet = (nextPanel: PanelName) => {
    setPanel(nextPanel);
    setMobileSheet(nextPanel);
  };
  const closeMobileSheet = () => setMobileSheet(null);
  const activeMobilePanel = mobileSheet && mobileSheet !== "library" ? mobileSheet : null;
  const persistedChronicleMessages = model?.chronicle?.messages ?? [];
  const pendingAfterMessageId = persistedChronicleMessages[persistedChronicleMessages.length - 1]?.message_id ?? null;
  const activePendingMessage = pendingMessageForActiveSave(pendingMessage, activeSaveId);
  const activeNarratorJob = chatBlockerJobs.find(({ job }) => isChatJobType(job.type)) ?? null;
  const activeNarratorDraft = activeNarratorJob
    ? narratorDrafts[activeNarratorJob.job.id]?.draft ?? null
    : null;
  const activeNarratorPlaceholder: PendingChronicleMessage | null = activePendingMessage ? {
    message_id: "pending-narrator-placeholder",
    role: "narrator",
    speaker_name: null,
    body: activeNarratorDraft || activeNarratorJob?.progress || "Preparing narrator response",
    actions: [],
    pending_after_message_id: activePendingMessage.message_id,
    pending_save_id: activeSaveId,
    paint_started_at_ms: activePendingMessage.paint_started_at_ms,
    pending_kind: "narrator_placeholder",
    pending_progress: activeNarratorJob?.progress || "Preparing narrator response",
    pending_draft: activeNarratorDraft ?? undefined,
    pending_started_at_ms: activePendingMessage.paint_started_at_ms,
    pending_timing_estimate: chatTimingSummary.data?.estimate ?? null,
  } : null;
  const activePendingMessages = [activePendingMessage, activeNarratorPlaceholder]
    .filter((message): message is PendingChronicleMessage => Boolean(message));
  const chatInputDisabled = !activeSaveSupported || !model?.composer_enabled || Boolean(activePendingMessage) || hasActiveSaveChatBlocker || !chatCanSubmit;

  return (
    <div className={shellClass} style={shellStyle}>
      {!isMobileWorkbench ? (
        <>
          <LeftRail
            model={model}
            scenarioRefreshVersion={scenarioRefreshVersion}
            currentUser={currentUser}
            onChanged={refreshWorkbench}
            onSelectSave={selectSave}
            pendingSaveId={pendingSaveId}
            saveSelectionError={saveSelectionError}
            onNew={() => openScenarioDialog("manual")}
            onContinuationDraft={startContinuationDraft}
            onReuseScenarioPrompt={reuseScenarioPrompt}
            activePanel={panel}
            setPanel={setPanel}
            compactLibrary={isStackedDesktopWorkbench}
            onOpenLibrary={() => setMobileSheet("library")}
            runJob={runJob}
            saveExports={[saveExportStates, setSaveExportStates, clearSaveExportRecovery]}
          />
          {!isStackedWorkbench ? (
            <WorkbenchResizeHandle
              side="left"
              active={resizingSide === "left"}
              value={layout.leftRailWidth}
              onPointerDown={startLayoutResize}
              onKeyDown={onLayoutKeyDown}
            />
          ) : null}
        </>
      ) : null}
      <main className="chronicle-pane">
        <div className="topbar">
          {topbarExpanded ? (
            <div>
              <p className="eyebrow">{model?.model_indicator ?? "No model selected"}</p>
              <h1>{model?.active_save_title ?? "Bragi Workbench"}</h1>
              <div className="topbar-meta">
                <span>{model?.scenario_title ?? "No scenario loaded"}</span>
                {model?.scene_title ? <span>{model.scene_title}</span> : null}
              </div>
              <WorldTimeControl
                worldTime={model?.world_time ?? null}
                activeSaveId={activeSaveSupported ? activeSaveId : null}
                onRuntimeChanged={applyRuntimeModel}
              />
            </div>
          ) : null}
          <div className="topbar-actions" style={{ marginLeft: "auto" }}>
            {runtimeLoadError || model?.error ? (
              <p className="topbar-error" role="alert">{runtimeLoadError || model?.error}</p>
            ) : null}
            {topbarExpanded ? (
              <>
                <button
                  type="button"
                  className="icon-button"
                  title="Look around"
                  aria-label="Look around"
                  disabled={lookAroundDisabled}
                  onClick={() => openLookAround("")}
                >
                  <Search size={16} aria-hidden="true" />
                </button>
                {model?.character_texts_enabled ? (
                  <button
                    type="button"
                    className="icon-button phone-button"
                    title={unreadCharacterTextCount ? `Open phone, ${unreadCharacterTextCount} unread` : "Open phone"}
                    aria-label={unreadCharacterTextCount ? `Open phone, ${unreadCharacterTextCount} unread` : "Open phone"}
                    disabled={!activeSaveId || !activeSaveSupported}
                    onClick={() => setPhoneOpen(true)}
                  >
                    <Smartphone size={16} aria-hidden="true" />
                    {unreadCharacterTextCount ? (
                      <span className="phone-unread-badge">{unreadCharacterTextCount}</span>
                    ) : null}
                  </button>
                ) : null}
                {currentUser && onLogout ? (
                  <div className="session-chip">
                    <span>{currentUser.username}</span>
                    <small>{currentUser.role}</small>
                    <button
                      type="button"
                      className="icon-button"
                      title="Log out"
                      aria-label="Log out"
                      onClick={onLogout}
                    >
                      <LogOut size={16} aria-hidden="true" />
                    </button>
                  </div>
                ) : null}
              </>
            ) : null}
            <button
              type="button"
              className="icon-button"
              title={topbarExpanded ? "Collapse top bar" : "Expand top bar"}
              aria-label={topbarExpanded ? "Collapse top bar" : "Expand top bar"}
              aria-expanded={topbarExpanded}
              style={{ marginLeft: "auto" }}
              onClick={() => setTopbarExpanded((expanded) => !expanded)}
            >
              {topbarExpanded ? (
                <ChevronUp size={16} aria-hidden="true" />
              ) : (
                <ChevronDown size={16} aria-hidden="true" />
              )}
            </button>
          </div>
        </div>
        {!activeSaveSupported ? (
          <InlineNotice>{activeSave?.unsupported_reason || "This save is no longer supported. Export or delete it from the library."}</InlineNotice>
        ) : null}
        {model?.continuity_degraded ? (
          <InlineNotice polite>
            Continuity updates are still catching up. Bragi is using the recent chronicle as the source of truth until repair completes.
          </InlineNotice>
        ) : null}
        <Chronicle
          key={activeSaveId ?? "no-save"}
          model={model}
          runJob={runJob}
          pendingMessages={activePendingMessages}
          narratorPaintMeasurement={narratorPaintMeasurement}
          onRuntimeChanged={applyRuntimeModel}
          currentUser={currentUser}
          mutationsDisabled={!activeSaveSupported}
          sceneArrivalMessageIds={sceneArrivalMessageIds}
          onCancelNarrator={activeNarratorJob ? () => cancelTrackedJob(activeNarratorJob) : undefined}
        />
        <PendingJobsTray
          jobs={pendingJobs}
          mode={pendingJobsDisplayMode}
          onCancel={cancelTrackedJob}
        />
        {chatSubmissionStatusNotice ? (
          <ChatSubmissionStatusNotice
            message={chatSubmissionStatusNotice}
            retrying={chatSubmissionStatus.isFetching}
            onRetry={() => {
              void chatSubmissionStatus.refetch();
            }}
          />
        ) : null}
        {model?.interaction_mode !== "storyteller" && model?.action_choices_enabled ? (
          <CyoaActionPicker
            disabled={chatInputDisabled}
            runJob={runJob}
            activeSaveId={activeSaveId}
            actionChoices={model?.action_choices ?? null}
            generationActive={pendingJobs.some(({ job }) => (
              job.type === "action_choice_generate"
              || job.type === "action_choice_regenerate"
            ))}
            generationRecoveryPending={activeJobs.isPending}
            pendingAfterMessageId={pendingAfterMessageId}
            onPendingMessage={setPendingMessage}
          />
        ) : (
          <Composer
            disabled={chatInputDisabled}
            runJob={runJob}
            activeSaveId={activeSaveId}
            pendingAfterMessageId={pendingAfterMessageId}
            onPendingMessage={setPendingMessage}
            storytellerMode={model?.interaction_mode === "storyteller"}
          />
        )}
      </main>
      {!isMobileWorkbench ? (
        <>
          {!isStackedWorkbench ? (
            <WorkbenchResizeHandle
              side="right"
              active={resizingSide === "right"}
              value={layout.rightPanelWidth}
              onPointerDown={startLayoutResize}
              onKeyDown={onLayoutKeyDown}
            />
          ) : null}
          <RightPanel panel={panel} model={model} runJob={runJob} currentUser={currentUser} openLookAround={openLookAround} readOnly={!activeSaveSupported} onContentSafetyChanged={refreshScenarioLibrary} />
        </>
      ) : (
        <>
          <MobileDock
            activePanel={panel}
            activeSheet={mobileSheet}
            onOpenLibrary={() => setMobileSheet("library")}
            onOpenPanel={openPanelSheet}
          />
          {activeMobilePanel ? (
            <MobileSheet title={panelLabel(activeMobilePanel)} icon={panelIcon(activeMobilePanel, 18)} onClose={closeMobileSheet}>
              <RightPanel panel={activeMobilePanel} model={model} runJob={runJob} currentUser={currentUser} openLookAround={openLookAround} readOnly={!activeSaveSupported} onContentSafetyChanged={refreshScenarioLibrary} />
            </MobileSheet>
          ) : null}
        </>
      )}
      {(isStackedWorkbench || isMobileWorkbench) && mobileSheet === "library" ? (
        <MobileSheet title="Library" icon={<BookOpen size={18} />} onClose={closeMobileSheet}>
          <LibraryControls
            model={model}
            scenarioRefreshVersion={scenarioRefreshVersion}
            currentUser={currentUser}
            onChanged={refreshWorkbench}
            onSelectSave={selectSave}
            pendingSaveId={pendingSaveId}
            saveSelectionError={saveSelectionError}
            onNew={() => openScenarioDialog("manual")}
            onContinuationDraft={startContinuationDraft}
            onReuseScenarioPrompt={reuseScenarioPrompt}
            onAfterAction={closeMobileSheet}
            runJob={runJob}
            saveExports={[saveExportStates, setSaveExportStates, clearSaveExportRecovery]}
          />
        </MobileSheet>
      ) : null}
      {phoneOpen && activeSaveSupported ? (
        <React.Suspense fallback={null}>
          <LazyCharacterTextPhone
            activeSaveId={activeSaveId}
            disabled={!activeSaveSupported}
            busyThreadIds={busyCharacterTextThreadIds}
            runJob={runJob}
            currentUser={currentUser}
            seenTextMessageIdsByThread={seenTextMessageIdsByThread}
            onThreadSeen={markCharacterTextThreadSeen}
            onClose={() => setPhoneOpen(false)}
          />
        </React.Suspense>
      ) : null}
      {lookAroundOpen && activeSaveSupported ? (
        <LookAroundDialog
          activeSaveId={activeSaveId}
          runJob={runJob}
          disabled={lookAroundDisabled}
          initialQuery={lookAroundInitialQuery}
          answer={lookAroundAnswer}
          onAnswer={setLookAroundAnswer}
          onClose={() => setLookAroundOpen(false)}
        />
      ) : null}
      {draftOpen ? (
        <React.Suspense fallback={null}>
          <LazyScenarioDialog model={model} initialMode={draftInitialMode} initialDraftPrefill={draftPrefill ?? undefined} currentUser={currentUser} onClose={() => setDraftOpen(false)} onRuntimeChanged={applyRuntimeModel} onScenarioListChanged={refreshScenarioLibrary} runJob={runJob} />
        </React.Suspense>
      ) : null}
    </div>
  );
}

function WorkbenchResizeHandle({
  side,
  active,
  value,
  onPointerDown,
  onKeyDown
}: {
  side: ResizeSide;
  active: boolean;
  value: number;
  onPointerDown: (side: ResizeSide, event: React.PointerEvent<HTMLDivElement>) => void;
  onKeyDown: (side: ResizeSide, event: React.KeyboardEvent<HTMLDivElement>) => void;
}) {
  const label = side === "left" ? "Resize left rail" : "Resize right panel";
  const limits = WORKBENCH_LAYOUT_LIMITS[side];
  return (
    <div
      role="separator"
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemin={limits.min}
      aria-valuemax={limits.max}
      aria-valuenow={value}
      aria-valuetext={`${value}px`}
      tabIndex={0}
      className={`layout-resize-handle ${side}-resize-handle ${active ? "active" : ""}`}
      onPointerDown={(event) => onPointerDown(side, event)}
      onKeyDown={(event) => onKeyDown(side, event)}
    />
  );
}

function LeftRail(props: {
  model?: RuntimeModel;
  scenarios?: Scenario[];
  scenarioRefreshVersion?: number;
  currentUser?: CurrentUser | null;
  onChanged: () => void;
  onSelectSave: (saveId: string) => Promise<boolean | void>;
  pendingSaveId: string | null;
  saveSelectionError: string;
  onNew: () => void;
  onContinuationDraft?: (chapterStartInstructions: string) => Promise<void>;
  onReuseScenarioPrompt?: (scenario: Scenario) => Promise<void>;
  activePanel: PanelName;
  setPanel: (panel: PanelName) => void;
  compactLibrary?: boolean;
  onOpenLibrary?: () => void;
  runJob?: RunJob;
  saveExports?: SaveExports;
}) {
  return (
    <aside className={`left-rail ${props.compactLibrary ? "compact-library-rail" : ""}`}>
      <div className="brand-block">
        <BrandLockup compact={props.compactLibrary} />
      </div>
      {props.compactLibrary ? (
        <button
          type="button"
          className="compact-library-button"
          aria-label="Open library"
          title="Open library"
          onClick={props.onOpenLibrary}
        >
          <BookOpen size={18} aria-hidden="true" />
          <span>Library</span>
        </button>
      ) : (
        <LibraryControls
          model={props.model}
          scenarios={props.scenarios}
          scenarioRefreshVersion={props.scenarioRefreshVersion}
          currentUser={props.currentUser}
          onChanged={props.onChanged}
          onSelectSave={props.onSelectSave}
          pendingSaveId={props.pendingSaveId}
          saveSelectionError={props.saveSelectionError}
          onNew={props.onNew}
          onContinuationDraft={props.onContinuationDraft}
          onReuseScenarioPrompt={props.onReuseScenarioPrompt}
          runJob={props.runJob}
          saveExports={props.saveExports}
        />
      )}
      <PanelNav activePanel={props.activePanel} setPanel={props.setPanel} />
    </aside>
  );
}

function LibraryControls(props: {
  model?: RuntimeModel;
  scenarios?: Scenario[];
  scenarioRefreshVersion?: number;
  currentUser?: CurrentUser | null;
  onChanged: () => void;
  onSelectSave: (saveId: string) => Promise<boolean | void>;
  pendingSaveId: string | null;
  saveSelectionError: string;
  onNew: () => void;
  onContinuationDraft?: (chapterStartInstructions: string) => Promise<void>;
  onReuseScenarioPrompt?: (scenario: Scenario) => Promise<void>;
  onAfterAction?: () => void;
  runJob?: RunJob;
  saveExports?: SaveExports;
}) {
  const [confirm, setConfirm] = useState<{ kind: "save" | "scenario"; id: string; title: string } | null>(null);
  const [renamingSave, setRenamingSave] = useState<SaveListItem | null>(null);
  const [editingScenario, setEditingScenario] = useState<Scenario | null>(null);
  const [startingScenario, setStartingScenario] = useState<Scenario | null>(null);
  const [chapterDialogOpen, setChapterDialogOpen] = useState(false);
  const [scenarioReuseError, setScenarioReuseError] = useState("");
  const [reusingScenarioId, setReusingScenarioId] = useState("");
  const localSaveExports = useState<SaveExportStates>({});
  const [saveExportStates, setSaveExportStates] = props.saveExports ?? localSaveExports;
  const clearSaveExportRecovery = props.saveExports?.[2];
  const libraryUserId = props.currentUser?.id ?? null;
  const [scopedLibraryState, setScopedLibraryState] = useState<ScopedLibraryControlsState>(() => ({
    userId: libraryUserId,
    state: loadLibraryControlsState(libraryUserId)
  }));
  const libraryState = scopedLibraryState.userId === libraryUserId
    ? scopedLibraryState.state
    : loadLibraryControlsState(libraryUserId);
  const tabBaseId = React.useId();
  const childRestrictedControlsAllowed = canUseChildRestrictedControls(props.currentUser);
  const adminControlsAllowed = canUseAdminControls(props.currentUser);
  const saves = props.model?.saves ?? [];
  const [scenarioLoad, setScenarioLoad] = useState<{
    scenarios: Scenario[];
    loading: boolean;
    error: string;
    loadedAt: number;
  }>({
    scenarios: props.scenarios ?? [],
    loading: false,
    error: "",
    loadedAt: props.scenarios === undefined ? 0 : Date.now()
  });
  const shouldLoadScenarios = props.scenarios === undefined && libraryState.activeTab === "scenarios";
  // Loading is set inside this effect; depending on it aborts the request we just started.
  useEffect(() => {
    if (!shouldLoadScenarios) {
      setScenarioLoad((current) => (
        current.loading ? { ...current, loading: false } : current
      ));
      return undefined;
    }
    if (
      scenarioLoad.loading
      || (scenarioLoad.loadedAt > 0 && Date.now() - scenarioLoad.loadedAt < 60_000)
    ) {
      return undefined;
    }
    const controller = new AbortController();
    setScenarioLoad((current) => ({ ...current, loading: true, error: "" }));
    apiRead<{ scenarios: Scenario[] }>("/api/scenarios", controller.signal)
      .then((payload) => {
        setScenarioLoad({
          scenarios: payload.scenarios,
          loading: false,
          error: "",
          loadedAt: Date.now()
        });
      })
      .catch((failure: unknown) => {
        if (controller.signal.aborted) return;
        setScenarioLoad((current) => ({
          ...current,
          loading: false,
          error: failure instanceof Error ? failure.message : "Could not load scenarios",
          loadedAt: Date.now()
        }));
      });
    return () => controller.abort();
  }, [scenarioLoad.loadedAt, shouldLoadScenarios]);
  useEffect(() => {
    if (props.scenarios === undefined) return;
    setScenarioLoad({
      scenarios: props.scenarios,
      loading: false,
      error: "",
      loadedAt: Date.now()
    });
  }, [props.scenarios]);
  useEffect(() => {
    if (props.scenarios !== undefined) return;
    setScenarioLoad((current) => (
      current.loadedAt === 0
        ? current
        : { ...current, loadedAt: 0 }
    ));
  }, [props.scenarioRefreshVersion, props.scenarios]);
  const refreshLocalScenarios = useCallback(() => {
    setScenarioLoad((current) => ({
      ...current,
      loading: false,
      loadedAt: libraryState.activeTab === "scenarios" ? 0 : current.loadedAt
    }));
    props.onChanged();
  }, [libraryState.activeTab, props.onChanged]);
  const scenarioItems = props.scenarios ?? scenarioLoad.scenarios;
  const scenariosLoading = props.scenarios === undefined && scenarioLoad.loading;
  const scenariosError = props.scenarios === undefined ? scenarioLoad.error : "";
  const scenarioTypes = useMemo(
    () => Array.from(new Set(scenarioItems.flatMap(scenarioTypeValues))).sort(compareText),
    [scenarioItems]
  );
  const effectiveScenarioType = scenarioTypes.includes(libraryState.scenarioType)
    ? libraryState.scenarioType
    : "all";
  const visibleSaves = useMemo(
    () => sortedSaves(
      saves.filter((save) => librarySearchMatches(libraryState.saveQuery, [
        save.title,
        save.scenario_title,
        save.scenario_id,
        save.save_id
      ])),
      libraryState.saveSort,
      libraryState.saveDirection
    ),
    [saves, libraryState.saveQuery, libraryState.saveSort, libraryState.saveDirection]
  );
  const visibleScenarios = useMemo(
    () => sortedScenarios(
      scenarioItems.filter((scenario) => (
        librarySearchMatches(libraryState.scenarioQuery, [
          scenario.title,
          scenario.premise,
          scenario.player_role,
          scenario.persistent_world_title,
          scenarioTypesLabel(scenario.scenario_types, scenario.scenario_type),
          scenario.scenario_id
        ])
        && (effectiveScenarioType === "all" || scenarioTypeValues(scenario).includes(effectiveScenarioType))
        && scenarioUsageMatches(scenario, libraryState.scenarioUsage)
      )),
      libraryState.scenarioSort,
      libraryState.scenarioDirection
    ),
    [
      scenarioItems,
      libraryState.scenarioQuery,
      libraryState.scenarioSort,
      libraryState.scenarioDirection,
      libraryState.scenarioUsage,
      effectiveScenarioType
    ]
  );

  useEffect(() => {
    setScopedLibraryState((current) => (
      current.userId === libraryUserId
        ? current
        : { userId: libraryUserId, state: loadLibraryControlsState(libraryUserId) }
    ));
  }, [libraryUserId]);

  useEffect(() => {
    if (scopedLibraryState.userId !== libraryUserId) return;
    saveLibraryControlsState(libraryUserId, scopedLibraryState.state);
  }, [libraryUserId, scopedLibraryState]);

  const updateLibraryState = useCallback((patch: Partial<LibraryControlsState>) => {
    setScopedLibraryState((current) => {
      const currentState = current.userId === libraryUserId
        ? current.state
        : loadLibraryControlsState(libraryUserId);
      return {
        userId: libraryUserId,
        state: { ...currentState, ...patch }
      };
    });
  }, [libraryUserId]);
  const finishAfterSaveSelection = async (saveId: string) => {
    const selected = await props.onSelectSave(saveId);
    if (selected === false) return;
    props.onChanged();
    props.onAfterAction?.();
  };
  const activeSaveTitle = props.model?.saves?.find((save) => save.active)?.title
    ?? props.model?.active_save_title
    ?? "Current save";
  const activeSaveSupported = props.model?.saves?.find((save) => save.active)?.supported !== false;
  const currentRenamingSave = renamingSave
    ? saves.find((save) => save.save_id === renamingSave.save_id && save.supported !== false) ?? null
    : null;
  const currentEditingScenario = editingScenario
    ? scenarioItems.find((scenario) => scenario.scenario_id === editingScenario.scenario_id && scenario.supported !== false) ?? null
    : null;
  const currentStartingScenario = startingScenario
    ? scenarioItems.find((scenario) => scenario.scenario_id === startingScenario.scenario_id && scenario.supported !== false) ?? null
    : null;
  useEffect(() => {
    if (renamingSave && !currentRenamingSave) setRenamingSave(null);
    if (editingScenario && !currentEditingScenario) setEditingScenario(null);
    if (startingScenario && !currentStartingScenario) setStartingScenario(null);
    if (chapterDialogOpen && !activeSaveSupported) setChapterDialogOpen(false);
  }, [
    activeSaveSupported,
    chapterDialogOpen,
    currentEditingScenario,
    currentRenamingSave,
    currentStartingScenario,
    editingScenario,
    renamingSave,
    startingScenario
  ]);
  const tabOptions: SegmentOption<LibraryTab>[] = [
    {
      value: "saves",
      label: `Saves (${saves.length})`,
      tabId: `${tabBaseId}-saves-tab`,
      panelId: `${tabBaseId}-saves-panel`
    },
    {
      value: "scenarios",
      label: `Scenarios (${scenarioItems.length})`,
      tabId: `${tabBaseId}-scenarios-tab`,
      panelId: `${tabBaseId}-scenarios-panel`
    },
    {
      value: "worlds",
      label: "Worlds",
      tabId: `${tabBaseId}-worlds-tab`,
      panelId: `${tabBaseId}-worlds-panel`
    }
  ];
  return (
    <div className="library-controls">
      <button
        className="primary-command"
        onClick={() => {
          props.onNew();
          props.onAfterAction?.();
        }}
      >
        <Plus size={16} /> New scenario
      </button>
      <div className="library-manager">
        <SegmentedTabs
          className="segmented library-tabs"
          label="Library sections"
          value={libraryState.activeTab}
          onChange={(activeTab) => updateLibraryState({ activeTab })}
          options={tabOptions}
        />
        {libraryState.activeTab === "saves" ? (
          <section
            id={`${tabBaseId}-saves-panel`}
            className="library-panel"
            role="tabpanel"
            aria-labelledby={`${tabBaseId}-saves-tab`}
          >
            {props.saveSelectionError ? <InlineNotice>{props.saveSelectionError}</InlineNotice> : null}
            {childRestrictedControlsAllowed ? (
              <SaveBundleControls
                hasActiveSave={Boolean(props.model?.active_save_id)}
                exportEnabled={activeSaveSupported}
                activeSaveId={props.model?.active_save_id ?? null}
                onImported={(saveId) => {
                  if (saveId) {
                    void finishAfterSaveSelection(saveId);
                    return;
                  }
                  props.onChanged();
                  props.onAfterAction?.();
                }}
                runJob={props.runJob}
                saveExports={[saveExportStates, setSaveExportStates, clearSaveExportRecovery]}
              />
            ) : null}
            <button
              type="button"
              className="secondary-command chapter-command"
              disabled={!props.model?.active_save_id || !props.onContinuationDraft || !activeSaveSupported}
              onClick={() => setChapterDialogOpen(true)}
            >
              <BookOpen size={15} /> <span>New chapter from current save</span>
            </button>
            <div className="library-toolbar">
              <label className="library-search">
                <Search size={15} aria-hidden="true" />
                <input
                  aria-label="Search saves"
                  value={libraryState.saveQuery}
                  placeholder="Search saves"
                  autoComplete="off"
                  spellCheck={false}
                  onChange={(event) => updateLibraryState({ saveQuery: event.target.value })}
                />
                <button
                  type="button"
                  aria-label="Clear save search"
                  title="Clear save search"
                  disabled={!libraryState.saveQuery}
                  onClick={() => updateLibraryState({ saveQuery: "" })}
                >
                  <X size={14} />
                </button>
              </label>
              <div className="library-filter-row">
                <label>
                  <span>Sort</span>
                  <select
                    aria-label="Sort saves"
                    value={libraryState.saveSort}
                    onChange={(event) => {
                      const saveSort = event.target.value as SaveSortKey;
                      updateLibraryState({
                        saveSort,
                        saveDirection: defaultSaveSortDirection(saveSort)
                      });
                    }}
                  >
                    <option value="last_opened">Last opened</option>
                    <option value="title">Title</option>
                    <option value="scenario_title">Scenario</option>
                    <option value="updated">Updated</option>
                    <option value="created">Created</option>
                  </select>
                </label>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Reverse save sort direction"
                  title={libraryState.saveDirection === "asc" ? "Ascending" : "Descending"}
                  onClick={() => updateLibraryState({ saveDirection: flipSortDirection(libraryState.saveDirection) })}
                >
                  {libraryState.saveDirection === "asc" ? <ArrowUp size={15} /> : <ArrowDown size={15} />}
                </button>
              </div>
              <p className="library-result-count" aria-live="polite">{visibleSaves.length} of {saves.length} saves</p>
            </div>
            <div className="stack-list library-list">
              {visibleSaves.map((save) => {
                const exportState = saveExportStates[save.save_id];
                const exportDownload = isChatBundleExportResult(exportState) ? exportState : null;
                return (
                <div className={`library-row ${save.active ? "active" : ""}`} key={save.save_id}>
                  <button
                    className="library-row-main"
                    title={`Load ${save.title}`}
                    aria-label={`Load ${save.title}`}
                    disabled={props.pendingSaveId !== null || save.supported === false}
                    onClick={async () => {
                      await finishAfterSaveSelection(save.save_id);
                    }}
                  >
                    <Save size={14} />
                    <span className="library-row-copy">
                      <strong>{save.title}</strong>
                      <small>{saveLibraryMeta(save)}</small>
                      {save.supported === false && save.unsupported_reason ? <small>{save.unsupported_reason}</small> : null}
                    </span>
                    {save.active ? <span className="library-pill current">Current</span> : null}
                    {save.supported === false ? <span className="library-pill unsupported">Unsupported</span> : null}
                  </button>
                  {childRestrictedControlsAllowed ? (
                    <div className="row-tools library-row-tools">
                      {exportDownload && (save.supported === false || save.save_id !== props.model?.active_save_id) ? (
                        <a
                          className={touchActionClassName("download-link")}
                          href={exportDownload.download_url}
                          download={exportDownload.filename}
                          title="Download save bundle"
                          aria-label={`Download ${save.title} export`}
                          onClick={() => {
                            clearSaveExportRecovery?.(save.save_id, "consume");
                            window.setTimeout(() => {
                              setSaveExportState(setSaveExportStates, save.save_id);
                            }, 0);
                          }}
                        >
                          <TouchActionContents icon={<Download size={14} />} label="Download" />
                        </a>
                      ) : save.supported !== false ? (
                        <button type="button" className={touchActionClassName()} title="Rename save" aria-label={`Rename ${save.title}`} onClick={() => setRenamingSave(save)}>
                          <TouchActionContents icon={<Edit3 size={14} />} label="Rename" />
                        </button>
                      ) : (
                        <button
                          type="button"
                          className={touchActionClassName()}
                          title="Export save bundle"
                          aria-label={`Export ${save.title}`}
                          disabled={!props.runJob || saveExportStates[save.save_id] === "pending"}
                          onClick={() => {
                            if (!props.runJob) return;
                            if (saveExportStates[save.save_id] === "pending") return;
                            clearSaveExportRecovery?.(save.save_id, "prepare");
                            setSaveExportState(setSaveExportStates, save.save_id, "pending");
                            void postJson<Job>("/api/bundles/export", {
                              save_id: save.save_id,
                              include_revision_history: false
                            }).then((job) => {
                              clearSaveExportRecovery?.(save.save_id, "restart");
                              props.runJob?.(job, {
                                allowInactiveSave: true,
                                allowCrossSaveCompletion: true,
                                onSucceeded: (result) => {
                                  if (isChatBundleExportResult(result)) {
                                    setSaveExportState(setSaveExportStates, save.save_id, result);
                                  } else {
                                    setSaveExportState(setSaveExportStates, save.save_id, "Save export completed without a download.");
                                  }
                                },
                                onFailed: (error) => {
                                  setSaveExportState(setSaveExportStates, save.save_id, error);
                                },
                                onFinished: (finished) => {
                                  if (finished.status !== "cancelled") return;
                                  setSaveExportState(setSaveExportStates, save.save_id, "Save export cancelled.");
                                }
                              });
                            }, (failure) => {
                              setSaveExportState(
                                setSaveExportStates,
                                save.save_id,
                                failure instanceof Error ? failure.message : "Could not start save export"
                              );
                            });
                          }}
                        >
                          <TouchActionContents
                            icon={<Download size={14} />}
                            label={exportState === "pending" ? "Exporting..." : "Export"}
                          />
                        </button>
                      )}
                      <button type="button" className={touchActionClassName("destructive-action")} title="Delete save" aria-label={`Delete ${save.title}`} onClick={() => setConfirm({ kind: "save", id: save.save_id, title: save.title })}>
                        <TouchActionContents icon={<Trash2 size={14} />} label="Delete" />
                      </button>
                    </div>
                  ) : null}
                  {exportDownload && (save.supported === false || save.save_id !== props.model?.active_save_id) ? (
                    <small role="status">Save export ready.</small>
                  ) : typeof exportState === "string"
                    && exportState !== "pending"
                    && (save.supported === false || save.save_id !== props.model?.active_save_id) ? (
                    <small
                      role="alert"
                    >
                      {exportState}
                    </small>
                  ) : null}
                </div>
                );
              })}
              {!saves.length ? <p className="empty">No saves yet</p> : null}
              {saves.length && !visibleSaves.length ? <p className="empty">No saves match</p> : null}
            </div>
          </section>
        ) : libraryState.activeTab === "scenarios" ? (
          <section
            id={`${tabBaseId}-scenarios-panel`}
            className="library-panel"
            role="tabpanel"
            aria-labelledby={`${tabBaseId}-scenarios-tab`}
          >
            {childRestrictedControlsAllowed ? <ScenarioBundleUpload onImported={refreshLocalScenarios} /> : null}
            {scenarioReuseError ? <InlineNotice>{scenarioReuseError}</InlineNotice> : null}
            {scenariosLoading ? <p className="muted">Loading scenarios...</p> : null}
            {scenariosError ? <InlineNotice>{scenariosError}</InlineNotice> : null}
            <div className="library-toolbar">
              <label className="library-search">
                <Search size={15} aria-hidden="true" />
                <input
                  aria-label="Search scenarios"
                  value={libraryState.scenarioQuery}
                  placeholder="Search scenarios"
                  autoComplete="off"
                  spellCheck={false}
                  onChange={(event) => updateLibraryState({ scenarioQuery: event.target.value })}
                />
                <button
                  type="button"
                  aria-label="Clear scenario search"
                  title="Clear scenario search"
                  disabled={!libraryState.scenarioQuery}
                  onClick={() => updateLibraryState({ scenarioQuery: "" })}
                >
                  <X size={14} />
                </button>
              </label>
              <div className="library-filter-row">
                <label>
                  <span>Sort</span>
                  <select
                    aria-label="Sort scenarios"
                    value={libraryState.scenarioSort}
                    onChange={(event) => {
                      const scenarioSort = event.target.value as ScenarioSortKey;
                      updateLibraryState({
                        scenarioSort,
                        scenarioDirection: defaultScenarioSortDirection(scenarioSort)
                      });
                    }}
                  >
                    <option value="updated">Updated</option>
                    <option value="title">Title</option>
                    <option value="save_count">Save count</option>
                    <option value="type">Type</option>
                    <option value="created">Created</option>
                  </select>
                </label>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Reverse scenario sort direction"
                  title={libraryState.scenarioDirection === "asc" ? "Ascending" : "Descending"}
                  onClick={() => updateLibraryState({ scenarioDirection: flipSortDirection(libraryState.scenarioDirection) })}
                >
                  {libraryState.scenarioDirection === "asc" ? <ArrowUp size={15} /> : <ArrowDown size={15} />}
                </button>
              </div>
              <div className="library-filter-row split">
                <label>
                  <span>Type</span>
                  <select
                    aria-label="Scenario type"
                    value={effectiveScenarioType}
                    onChange={(event) => updateLibraryState({ scenarioType: event.target.value })}
                  >
                    <option value="all">All types</option>
                    {scenarioTypes.map((scenarioType) => (
                      <option key={scenarioType} value={scenarioType}>{scenarioTypeLabel(scenarioType)}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Usage</span>
                  <select
                    aria-label="Scenario usage"
                    value={libraryState.scenarioUsage}
                    onChange={(event) => updateLibraryState({ scenarioUsage: event.target.value as ScenarioUsageFilter })}
                  >
                    <option value="all">All usage</option>
                    <option value="used">In use</option>
                    <option value="unused">Unused</option>
                  </select>
                </label>
              </div>
              <p className="library-result-count" aria-live="polite">{visibleScenarios.length} of {scenarioItems.length} scenarios</p>
            </div>
            <div className="stack-list library-list">
              {visibleScenarios.map((scenario) => (
                <div className="library-row" key={scenario.scenario_id}>
                  <button
                    className="library-row-main"
                    title={`Start ${scenario.title}`}
                    aria-label={`Start ${scenario.title}`}
                    disabled={scenario.supported === false}
                    onClick={() => setStartingScenario(scenario)}
                  >
                    <Play size={14} />
                    <span className="library-row-copy">
                      <strong>{scenario.title}</strong>
                      <small>{scenarioLibraryMeta(scenario)}</small>
                      {scenario.supported === false && scenario.unsupported_reason ? <small>{scenario.unsupported_reason}</small> : null}
                    </span>
                    <span className="library-pill">{scenario.save_count}</span>
                    {scenario.supported === false ? <span className="library-pill unsupported">Unsupported</span> : null}
                  </button>
                  {(scenario.has_generation_prompt && props.onReuseScenarioPrompt) || childRestrictedControlsAllowed || adminControlsAllowed ? (
                    <div className="row-tools library-row-tools">
                      {scenario.supported !== false && scenario.has_generation_prompt && props.onReuseScenarioPrompt ? (
                        <button
                          type="button"
                          className={touchActionClassName()}
                          title="Reuse AI prompt"
                          aria-label={`Reuse prompt for ${scenario.title}`}
                          disabled={reusingScenarioId === scenario.scenario_id}
                          onClick={async () => {
                            setScenarioReuseError("");
                            setReusingScenarioId(scenario.scenario_id);
                            try {
                              await props.onReuseScenarioPrompt?.(scenario);
                              props.onAfterAction?.();
                            } catch (failure) {
                              setScenarioReuseError(failure instanceof Error ? failure.message : "Could not reuse scenario prompt");
                            } finally {
                              setReusingScenarioId((current) => current === scenario.scenario_id ? "" : current);
                            }
                          }}
                        >
                          <TouchActionContents
                            icon={reusingScenarioId === scenario.scenario_id ? <Loader2 className="spin" size={14} /> : <Wand2 size={14} />}
                            label="Reuse"
                          />
                        </button>
                      ) : null}
                      {childRestrictedControlsAllowed ? (
                        <button type="button" className={touchActionClassName()} title="Export scenario bundle" aria-label={`Export ${scenario.title}`} onClick={() => openDownloadInNewTab(`/api/scenario-bundles/export/${scenario.scenario_id}`)}>
                          <TouchActionContents icon={<Download size={14} />} label="Export" />
                        </button>
                      ) : null}
                      {adminControlsAllowed ? (
                        <>
                          {scenario.supported !== false ? (
                            <button type="button" className={touchActionClassName()} title="Edit scenario definition" aria-label={`Edit ${scenario.title}`} onClick={() => setEditingScenario(scenario)}>
                              <TouchActionContents icon={<Edit3 size={14} />} label="Edit" />
                            </button>
                          ) : null}
                          <button type="button" className={touchActionClassName("destructive-action")} title="Delete scenario" aria-label={`Delete ${scenario.title}`} onClick={() => setConfirm({ kind: "scenario", id: scenario.scenario_id, title: scenario.title })}>
                            <TouchActionContents icon={<Trash2 size={14} />} label="Delete" />
                          </button>
                        </>
                      ) : null}
                    </div>
                  ) : null}
                </div>
              ))}
              {!scenariosLoading && !scenarioItems.length ? <p className="empty">No saved scenarios</p> : null}
              {scenarioItems.length && !visibleScenarios.length ? <p className="empty">No scenarios match</p> : null}
            </div>
          </section>
        ) : (
          <PersistentWorldLibrary
            adminControlsAllowed={adminControlsAllowed}
            allowImportExport={childRestrictedControlsAllowed}
            onChanged={props.onChanged}
          />
        )}
      </div>
      {currentRenamingSave ? (
        <RenameSaveModal
          save={currentRenamingSave}
          onCancel={() => setRenamingSave(null)}
          onRenamed={() => {
            setRenamingSave(null);
            props.onChanged();
          }}
        />
      ) : null}
      {currentStartingScenario ? (
        <StartScenarioModal
          scenario={currentStartingScenario}
          onCancel={() => setStartingScenario(null)}
          onStarted={(result) => {
            setStartingScenario(null);
            if (result.active_save_id) {
              void finishAfterSaveSelection(result.active_save_id);
              refreshLocalScenarios();
              return;
            }
            refreshLocalScenarios();
            props.onAfterAction?.();
          }}
        />
      ) : null}
      {confirm ? (
        <ConfirmModal
          title={`Delete ${confirm.kind}?`}
          body={confirm.title}
          confirmLabel="Delete"
          destructive
          onCancel={() => setConfirm(null)}
          onConfirm={async () => {
            await deleteJson(confirm.kind === "save" ? `/api/saves/${confirm.id}` : `/api/scenarios/${confirm.id}`);
            setConfirm(null);
            if (confirm.kind === "scenario") {
              refreshLocalScenarios();
            } else {
              props.onChanged();
            }
          }}
        />
      ) : null}
      {currentEditingScenario ? (
        <ScenarioDefinitionModal
          scenario={currentEditingScenario}
          onClose={() => setEditingScenario(null)}
          onSaved={() => {
            setEditingScenario(null);
            refreshLocalScenarios();
          }}
        />
      ) : null}
      {chapterDialogOpen && activeSaveSupported ? (
        <NewChapterDialog
          activeSaveTitle={activeSaveTitle}
          onCancel={() => setChapterDialogOpen(false)}
          onSubmit={async (chapterStartInstructions) => {
            if (!props.onContinuationDraft) throw new Error("Could not draft chapter scenario");
            await props.onContinuationDraft(chapterStartInstructions);
            setChapterDialogOpen(false);
            props.onAfterAction?.();
          }}
        />
      ) : null}
    </div>
  );
}

function librarySearchMatches(query: string, values: Array<string | null | undefined>) {
  const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
  if (!terms.length) return true;
  const haystack = values.filter(Boolean).join(" ").toLocaleLowerCase();
  return terms.every((term) => haystack.includes(term));
}

function sortedSaves(saves: SaveListItem[], sortKey: SaveSortKey, direction: SortDirection) {
  return [...saves].sort((left, right) => withSortDirection(compareSaveSort(left, right, sortKey), direction));
}

function sortedScenarios(scenarios: Scenario[], sortKey: ScenarioSortKey, direction: SortDirection) {
  return [...scenarios].sort((left, right) => withSortDirection(compareScenarioSort(left, right, sortKey), direction));
}

function compareSaveSort(left: SaveListItem, right: SaveListItem, sortKey: SaveSortKey) {
  if (sortKey === "title") return compareText(left.title, right.title);
  if (sortKey === "scenario_title") return compareText(left.scenario_title ?? "", right.scenario_title ?? "");
  if (sortKey === "created") return compareDate(left.created_at, right.created_at);
  if (sortKey === "updated") return compareDate(left.updated_at, right.updated_at);
  return compareDate(left.last_opened_at, right.last_opened_at);
}

function compareScenarioSort(left: Scenario, right: Scenario, sortKey: ScenarioSortKey) {
  if (sortKey === "title") return compareText(left.title, right.title);
  if (sortKey === "type") {
    return compareText(
      scenarioTypesLabel(left.scenario_types, left.scenario_type),
      scenarioTypesLabel(right.scenario_types, right.scenario_type)
    );
  }
  if (sortKey === "created") return compareDate(left.created_at, right.created_at);
  if (sortKey === "save_count") return left.save_count - right.save_count;
  return compareDate(left.updated_at, right.updated_at);
}

function compareText(left: string | null | undefined, right: string | null | undefined) {
  return (left ?? "").localeCompare(right ?? "", undefined, { numeric: true, sensitivity: "base" });
}

function compareDate(left: string | null | undefined, right: string | null | undefined) {
  return libraryDateValue(left) - libraryDateValue(right);
}

function withSortDirection(value: number, direction: SortDirection) {
  return direction === "asc" ? value : -value;
}

function defaultSaveSortDirection(sortKey: SaveSortKey): SortDirection {
  return sortKey === "title" || sortKey === "scenario_title" ? "asc" : "desc";
}

function defaultScenarioSortDirection(sortKey: ScenarioSortKey): SortDirection {
  return sortKey === "title" || sortKey === "type" ? "asc" : "desc";
}

function flipSortDirection(direction: SortDirection): SortDirection {
  return direction === "asc" ? "desc" : "asc";
}

function scenarioUsageMatches(scenario: Scenario, usage: ScenarioUsageFilter) {
  if (usage === "used") return scenario.save_count > 0;
  if (usage === "unused") return scenario.save_count === 0;
  return true;
}

function scenarioTypeValues(scenario: Scenario): string[] {
  return normalizedScenarioTypes(scenario.scenario_type, scenario.scenario_types);
}

function saveLibraryMeta(save: SaveListItem) {
  const parts = [
    save.scenario_title || "No scenario title",
    save.last_opened_at ? `opened ${formatLibraryDate(save.last_opened_at)}` : "",
    save.updated_at ? `updated ${formatLibraryDate(save.updated_at)}` : ""
  ].filter(Boolean);
  return parts.join(" - ");
}

function scenarioLibraryMeta(scenario: Scenario) {
  const parts = [
    scenarioTypesLabel(scenario.scenario_types, scenario.scenario_type),
    scenario.persistent_world_title ? `setting: ${scenario.persistent_world_title}` : "",
    `${scenario.save_count} ${scenario.save_count === 1 ? "save" : "saves"}`,
    scenario.updated_at ? `updated ${formatLibraryDate(scenario.updated_at)}` : ""
  ].filter(Boolean);
  return parts.join(" - ");
}

function formatLibraryDate(value: string) {
  const time = libraryDateValue(value);
  if (!time) return "unknown";
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric"
  }).format(new Date(time));
}

function libraryDateValue(value: string | null | undefined) {
  if (!value) return 0;
  const normalized = value.includes("T") ? value : value.replace(" ", "T");
  const parsed = Date.parse(normalized);
  return Number.isFinite(parsed) ? parsed : 0;
}

function PanelNav({ activePanel, setPanel }: { activePanel: PanelName; setPanel: (panel: PanelName) => void }) {
  return (
    <nav className="panel-nav" aria-label="Workbench panels">
      {PANEL_BUTTONS.map(([name, label, Icon]) => (
        <button
          key={String(name)}
          className={activePanel === name ? "active" : ""}
          onClick={() => setPanel(name)}
          onFocus={() => prefetchPanel(name)}
          onPointerEnter={() => prefetchPanel(name)}
          title={String(label)}
          aria-label={String(label)}
        >
          <Icon size={17} />
        </button>
      ))}
    </nav>
  );
}

function MobileDock({
  activePanel,
  activeSheet,
  onOpenLibrary,
  onOpenPanel
}: {
  activePanel: PanelName;
  activeSheet: MobileSheetName | null;
  onOpenLibrary: () => void;
  onOpenPanel: (panel: PanelName) => void;
}) {
  return (
    <nav className="mobile-dock" aria-label="Mobile navigation">
      <button
        type="button"
        className={activeSheet === "library" ? "active" : ""}
        aria-label="Library"
        title="Library"
        onClick={onOpenLibrary}
      >
        <BookOpen size={19} />
      </button>
      {PANEL_BUTTONS.map(([name, label, Icon]) => (
        <button
          type="button"
          key={name}
          className={activePanel === name || activeSheet === name ? "active" : ""}
          aria-label={label}
          title={label}
          aria-current={activeSheet === name ? "page" : undefined}
          onClick={() => onOpenPanel(name)}
          onFocus={() => prefetchPanel(name)}
          onPointerEnter={() => prefetchPanel(name)}
        >
          <Icon size={19} />
        </button>
      ))}
    </nav>
  );
}

function MobileSheet({ title, icon, onClose, children }: { title: string; icon: React.ReactNode; onClose: () => void; children: React.ReactNode }) {
  const titleId = React.useId();
  const sheetRef = useRef<HTMLElement | null>(null);
  const onKeyDown = useDialogFocus(sheetRef, onClose);

  return (
    <div className="mobile-sheet-backdrop" onClick={onClose}>
      <section
        ref={sheetRef}
        className="mobile-sheet"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        tabIndex={-1}
        onClick={(event) => event.stopPropagation()}
        onKeyDown={onKeyDown}
      >
        <header className="mobile-sheet-header">
          <div>
            {icon}
            <h2 id={titleId}>{title}</h2>
          </div>
          <button type="button" onClick={onClose} title="Close" aria-label="Close">
            <X size={18} />
          </button>
        </header>
        <div className="mobile-sheet-content">{children}</div>
      </section>
    </div>
  );
}

function panelLabel(panel: PanelName) {
  return PANEL_BUTTONS.find(([name]) => name === panel)?.[1] ?? labelize(panel);
}

function panelIcon(panel: PanelName, size = 17) {
  const Icon = PANEL_BUTTONS.find(([name]) => name === panel)?.[2] ?? PanelRight;
  return <Icon size={size} />;
}

function prefetchPanel(panel: PanelName) {
  if (panel === "world") void loadWorldPanel();
  if (panel === "characters") void loadCharactersPanel();
  if (panel === "settings") void loadSettingsPanel();
}

function NewChapterDialog({
  activeSaveTitle,
  onCancel,
  onSubmit
}: {
  activeSaveTitle: string;
  onCancel: () => void;
  onSubmit: (chapterStartInstructions: string) => Promise<void>;
}) {
  const [chapterStartInstructions, setChapterStartInstructions] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();

  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog"
        titleId={titleId}
        onClose={onCancel}
        onSubmit={async (event) => {
          event.preventDefault();
          setBusy(true);
          setError("");
          try {
            await onSubmit(chapterStartInstructions.trim());
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : "Could not draft chapter scenario");
            setBusy(false);
          }
        }}
      >
        <header>
          <div>
            <h2 id={titleId}>New chapter from current save</h2>
            <p className="muted">{activeSaveTitle}</p>
          </div>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Start instructions</span>
          <textarea
            className="tall-field"
            value={chapterStartInstructions}
            onChange={(event) => setChapterStartInstructions(event.target.value)}
            aria-label="Start instructions"
            autoFocus
            placeholder="Example: Start the next chapter the following morning after everyone wakes."
          />
        </label>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onCancel}>Cancel</button>
          <button className="primary-command compact" disabled={busy}>
            {busy ? <Loader2 className="spin" size={15} /> : <BookOpen size={15} />} Start chapter draft
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function RenameSaveModal({
  save,
  onCancel,
  onRenamed
}: {
  save: SaveListItem;
  onCancel: () => void;
  onRenamed: (result: RuntimeModel) => void;
}) {
  const [saveTitle, setSaveTitle] = useState(save.title);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();

  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog"
        titleId={titleId}
        onClose={onCancel}
        onSubmit={async (event) => {
          event.preventDefault();
          const title = saveTitle.trim();
          if (!title) {
            setError("Save title is required");
            return;
          }
          setBusy(true);
          setError("");
          try {
            const result = await postJson<RuntimeModel>(
              `/api/saves/${encodeURIComponent(save.save_id)}/rename`,
              { title }
            );
            onRenamed(result);
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : "Save title could not be changed.");
            setBusy(false);
          }
        }}
      >
        <header>
          <div>
            <h2 id={titleId}>Rename save</h2>
            <p className="muted">{save.title}</p>
          </div>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Save Title</span>
          <input value={saveTitle} onChange={(event) => setSaveTitle(event.target.value)} autoFocus />
        </label>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onCancel}>Cancel</button>
          <button className="primary-command compact" disabled={busy}>
            {busy ? <Loader2 className="spin" size={15} /> : <Edit3 size={15} />} Rename
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function StartScenarioModal({ scenario, onCancel, onStarted }: { scenario: Scenario; onCancel: () => void; onStarted: (result: RuntimeModel) => void }) {
  const [saveTitle, setSaveTitle] = useState(scenario.title);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog"
        titleId={titleId}
        onClose={onCancel}
        onSubmit={async (event) => {
          event.preventDefault();
          setBusy(true);
          try {
            const result = await postJson<RuntimeModel>(`/api/scenarios/${scenario.scenario_id}/start`, { save_title: saveTitle.trim() || scenario.title });
            onStarted(result);
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : "Could not start scenario");
          } finally {
            setBusy(false);
          }
        }}
      >
        <header>
          <div>
            <h2 id={titleId}>Start Scenario</h2>
            <p className="muted">{scenario.premise || scenario.player_role || "Create a new save from this scenario."}</p>
          </div>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Save Title</span>
          <input value={saveTitle} onChange={(event) => setSaveTitle(event.target.value)} autoFocus />
        </label>
        {scenario.save_count ? <p className="muted">{scenario.save_count} existing saves use this scenario.</p> : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onCancel}>Cancel</button>
          <button className="primary-command compact" disabled={busy}>
            {busy ? <Loader2 className="spin" size={15} /> : <Play size={15} />} Start
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function ScenarioBundleUpload({ onImported }: { onImported: () => void }) {
  const [pending, setPending] = useState<{ preview_id: string; preview: ScenarioBundlePreview } | null>(null);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  return (
    <div className="scenario-import-row">
      <button
        type="button"
        className="upload-button scenario-upload-button"
        aria-label="Import scenario bundle"
        onClick={() => inputRef.current?.click()}
      >
        <Upload size={15} /> Import scenario
      </button>
      <input
        ref={inputRef}
        className="upload-input"
        aria-label="Scenario bundle file"
        type="file"
        accept=".bragi-scenario"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (!file) return;
          const form = new FormData();
          form.append("file", file);
          try {
            setPending(await api<{ preview_id: string; preview: ScenarioBundlePreview }>("/api/scenario-bundles/preview", { method: "POST", body: form }));
            setError("");
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : "Scenario import preview failed");
          }
        }}
      />
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      {pending ? (
        <ScenarioBundlePreviewModal
          pending={pending}
          onCancel={() => setPending(null)}
          onImported={() => {
            setPending(null);
            onImported();
          }}
        />
      ) : null}
    </div>
  );
}

function ScenarioBundlePreviewModal({
  pending,
  onCancel,
  onImported
}: {
  pending: { preview_id: string; preview: ScenarioBundlePreview };
  onCancel: () => void;
  onImported: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog" titleId={titleId} onClose={onCancel}>
        <header>
          <h2 id={titleId}>Import scenario bundle?</h2>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <div className="preview-title">
          <strong>{pending.preview.title}</strong>
          <span>{scenarioTypeLabel(pending.preview.scenario_type)}</span>
        </div>
        <div className="preview-grid">
          <div><span>Scenario</span><strong>{pending.preview.scenario_id || "New"}</strong></div>
          <div><span>Type</span><strong>{scenarioTypeLabel(pending.preview.scenario_type)}</strong></div>
          <div><span>Bundle</span><strong>v{pending.preview.bundle_version}</strong></div>
        </div>
        <p className="muted">This adds the bundled reusable scenario to the local Scenario Library.</p>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onCancel}>Cancel</button>
          <button
            type="button"
            className="primary-command compact"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              setError("");
              try {
                await postJson(`/api/scenario-bundles/import/${pending.preview_id}`, {});
                onImported();
              } catch (failure) {
                setError(failure instanceof Error ? failure.message : "Could not import scenario");
              } finally {
                setBusy(false);
              }
            }}
          >
            Import
          </button>
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function chronicleMessages(model: RuntimeModel | undefined, pendingMessages: PendingChronicleMessage | PendingChronicleMessage[] | null): ChronicleMessage[] {
  const persisted = model?.chronicle?.messages ?? [];
  const pendingList = Array.isArray(pendingMessages) ? pendingMessages : pendingMessages ? [pendingMessages] : [];
  return pendingList.reduce<ChronicleMessage[]>((messages, pendingMessage) => {
    if (hasPersistedPendingMessage(messages, pendingMessage)) return messages;
    return [...messages, pendingMessage];
  }, persisted);
}

function mergeChroniclePage(model: RuntimeModel | undefined, page: ChronicleModel): RuntimeModel | undefined {
  if (!model) return model;
  const currentMessages = model.chronicle?.messages ?? [];
  const seen = new Set(page.messages.map((message) => message.message_id));
  const messages = [
    ...page.messages,
    ...currentMessages.filter((message) => !seen.has(message.message_id))
  ];
  return {
    ...model,
    chronicle: {
      ...model.chronicle,
      messages,
      has_more_before: Boolean(page.has_more_before),
      oldest_message_id: page.oldest_message_id ?? messages[0]?.message_id ?? null
    }
  };
}

function hasPersistedPendingMessage(persisted: ChronicleMessage[], pendingMessage: PendingChronicleMessage) {
  if (pendingMessage.pending_kind === "narrator_placeholder") return false;
  const anchorMessageId = pendingMessage.pending_after_message_id;
  if (anchorMessageId !== undefined) {
    const anchorIndex = anchorMessageId === null
      ? -1
      : persisted.findIndex((message) => message.message_id === anchorMessageId);
    const candidates = anchorIndex >= 0 || anchorMessageId === null
      ? persisted.slice(anchorIndex + 1)
      : persisted;
    return candidates.some((message) => matchesPendingMessage(message, pendingMessage));
  }

  const latest = persisted[persisted.length - 1];
  return Boolean(latest && matchesPendingMessage(latest, pendingMessage));
}

function matchesPendingMessage(message: ChronicleMessage, pendingMessage: PendingChronicleMessage) {
  return message.role === pendingMessage.role && message.body.trim() === pendingMessage.body.trim();
}

function pendingMessageForActiveSave(message: PendingChronicleMessage | null, activeSaveId: string | null) {
  return message?.pending_save_id === activeSaveId ? message : null;
}

function broadTimingRange(estimate: NonNullable<ChatTimingSummary["estimate"]>): string {
  const broadSeconds = (milliseconds: number) => Math.max(
    5,
    Math.round(milliseconds / 5_000) * 5,
  );
  const lower = broadSeconds(estimate.p50_ms);
  const upper = Math.max(lower, broadSeconds(estimate.p95_ms));
  return `${lower}–${upper}s`;
}

function narratorMessageIdFromResult(result: unknown): string | null {
  if (isChatTurnDelta(result)) return result.narrator_message_id;
  if (result && typeof result === "object") {
    const messageId = (result as { narrator_message_id?: unknown }).narrator_message_id;
    if (typeof messageId === "string" && messageId) return messageId;
  }
  if (!isRuntimeModel(result)) return null;
  const messages = result.chronicle?.messages ?? [];
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message?.role === "narrator") return message.message_id;
  }
  return null;
}

function chronicleJobActionKey(messageId: string, actionId: string) {
  return `${messageId}:${actionId}`;
}

function chronicleJobActionRequest(
  actionId: string,
  messageId: string,
  activeSaveId: string | null
): Omit<JobActionRequest, "key"> | null {
  const body = { message_id: messageId, save_id: activeSaveId };
  if (actionId === "regenerate-message") {
    return {
      path: "/api/chat/regenerate",
      body
    };
  }
  if (actionId === "retry-interrupted-turn") {
    return {
      path: "/api/chat/retry",
      body
    };
  }
  if (actionId === "generate-scene-image") {
    return {
      path: "/api/media/generate",
      body
    };
  }
  return null;
}

type ChronicleMessageRowProps = {
  message: ChronicleMessage;
  activeSaveId: string | null;
  jobActionErrors: Record<string, string>;
  pendingJobActionKeys: Set<string>;
  onAction: (actionId: string, message: ChronicleMessage) => void;
  mutationsDisabled?: boolean;
  storytellerMode?: boolean;
  sceneImageArriving?: boolean;
  onCancelNarrator?: () => void;
};

function NarratorPlaceholderRow({
  message,
  onCancel,
}: {
  message: PendingChronicleMessage;
  onCancel?: () => void;
}) {
  const startedAtMs = message.pending_started_at_ms;
  const [, setTick] = useState(0);
  const elapsedSeconds = startedAtMs === undefined
    ? 0
    : Math.max(0, Math.floor((performance.now() - startedAtMs) / 1_000));
  useEffect(() => {
    if (startedAtMs === undefined) return;
    const timer = window.setInterval(() => setTick((value) => value + 1), 250);
    return () => window.clearInterval(timer);
  }, [startedAtMs]);
  return (
    <article
      className={message.pending_draft
        ? "message narrator narrator-placeholder has-draft"
        : "message narrator narrator-placeholder"}
      data-message-id={message.message_id}
      data-draft={message.pending_draft ? "true" : "false"}
      role="status"
      aria-label={message.pending_draft ? "Narrator response preview" : "Narrator response progress"}
    >
      <header>
        <span>Narrator</span>
        {elapsedSeconds >= 3 ? <small>{elapsedSeconds}s elapsed</small> : null}
      </header>
      {message.pending_draft ? (
        <div className="narrator-placeholder-body narrator-draft-body">
          <span>{message.pending_draft}</span>
        </div>
      ) : (
        <div className="narrator-placeholder-body">
          <Loader2 className="spin" size={16} aria-hidden="true" />
          <span>{message.pending_progress || "Preparing narrator response"}</span>
        </div>
      )}
      {message.pending_draft ? (
        <small className="narrator-placeholder-estimate">Draft preview — not yet checked</small>
      ) : message.pending_timing_estimate ? (
        <small className="narrator-placeholder-estimate">
          Recent turns: about {broadTimingRange(message.pending_timing_estimate)}
        </small>
      ) : null}
      {onCancel ? (
        <button type="button" onClick={onCancel} aria-label="Cancel narrator response">
          <Square size={13} aria-hidden="true" />
          Cancel
        </button>
      ) : null}
    </article>
  );
}

const ChronicleMessageRow = React.memo(function ChronicleMessageRow({
  message,
  activeSaveId,
  jobActionErrors,
  pendingJobActionKeys,
  onAction,
  mutationsDisabled = false,
  storytellerMode = false,
  sceneImageArriving = false,
  onCancelNarrator,
}: ChronicleMessageRowProps) {
  const pendingMessage = message as PendingChronicleMessage;
  if (pendingMessage.pending_kind === "narrator_placeholder") {
    return (
      <NarratorPlaceholderRow
        message={pendingMessage}
        onCancel={onCancelNarrator}
      />
    );
  }
  const isDirection = storytellerMode && message.role === "player";
  const messageActionErrors = message.actions
    .map((action) => {
      const key = chronicleJobActionKey(message.message_id, action.action_id);
      return { key, error: jobActionErrors[key] };
    })
    .filter((item) => item.error);
  return (
    <article
      className={`message ${isDirection ? "direction" : message.role}`}
      data-message-id={message.message_id}
    >
      <header>
        <span>{isDirection ? "Direction" : message.speaker_name || message.role}</span>
        {message.revision_count ? <small className="message-edited">Edited</small> : null}
        <div className="message-actions">
          {message.actions
            .filter((action) => !message.interrupted_turn || ![
              "retry-interrupted-turn",
              "edit-and-resubmit-message",
              "delete-messages-from-here"
            ].includes(action.action_id))
            .map((action) => {
            const actionKey = chronicleJobActionKey(message.message_id, action.action_id);
            const jobRequest = chronicleJobActionRequest(action.action_id, message.message_id, activeSaveId);
            const actionPending = Boolean(jobRequest && pendingJobActionKeys.has(actionKey));
            return (
              <button
                key={action.action_id}
                type="button"
                className={touchActionClassName(action.action_id === "delete-messages-from-here" && "destructive-action")}
                title={action.label}
                aria-label={action.label}
                disabled={
                  message.message_id === "pending-player-message"
                  || actionPending
                  || mutationsDisabled
                }
                onClick={() => onAction(action.action_id, message)}
              >
                <TouchActionContents
                  icon={actionPending ? <Loader2 className="spin" size={14} aria-hidden="true" /> : actionIcon(action.action_id)}
                  label={action.label}
                />
              </button>
            );
          })}
        </div>
      </header>
      {messageActionErrors.map(({ key, error }) => (
        error ? <InlineNotice key={key} className="message-action-error">{error}</InlineNotice> : null
      ))}
      <MarkdownView message={message} />
      {sceneImageArriving ? (
        <span
          className="scene-arriving-tile"
          role="status"
          aria-label="Scene image arriving"
        />
      ) : null}
      {message.interrupted_turn ? (
        <div className="interrupted-turn" role="alert">
          <div>
            <strong>
              {message.interrupted_turn.status === "cancelled"
                ? "Turn cancelled"
                : "Turn interrupted"}
            </strong>
            <p>{message.interrupted_turn.reason}</p>
          </div>
          <div className="interrupted-turn-actions">
            {message.actions
              .filter((action) => [
                "retry-interrupted-turn",
                "edit-and-resubmit-message",
                "delete-messages-from-here"
              ].includes(action.action_id))
              .map((action) => {
                const actionKey = chronicleJobActionKey(
                  message.message_id,
                  action.action_id
                );
                const actionPending = pendingJobActionKeys.has(actionKey);
                return (
                  <button
                    key={action.action_id}
                    type="button"
                    className={touchActionClassName(
                      action.action_id === "delete-messages-from-here"
                        && "destructive-action"
                    )}
                    disabled={actionPending || mutationsDisabled}
                    title={action.label}
                    aria-label={action.label}
                    onClick={() => onAction(action.action_id, message)}
                  >
                    {actionPending ? "Working…" : action.label}
                  </button>
                );
              })}
          </div>
        </div>
      ) : null}
    </article>
  );
});

function Chronicle({
  model,
  runJob,
  pendingMessage,
  pendingMessages,
  narratorPaintMeasurement = null,
  currentUser = null,
  mutationsDisabled = false,
  sceneArrivalMessageIds = new Set<string>(),
  onCancelNarrator,
}: {
  model?: RuntimeModel;
  runJob: RunJob;
  pendingMessage?: PendingChronicleMessage | null;
  pendingMessages?: PendingChronicleMessage[];
  narratorPaintMeasurement?: NarratorPaintMeasurement | null;
  onRuntimeChanged?: (model: RuntimeModel) => void;
  currentUser?: CurrentUser | null;
  mutationsDisabled?: boolean;
  sceneArrivalMessageIds?: ReadonlySet<string>;
  onCancelNarrator?: () => void;
}) {
  const localPendingMessages = pendingMessages
    ?? (pendingMessage ? [pendingMessage] : []);
  const messages = chronicleMessages(model, localPendingMessages);
  const optimisticPaintMeasurement = localPendingMessages.find(
    (message) => (
      message.pending_kind !== "narrator_placeholder"
      && message.paint_started_at_ms !== undefined
    ),
  ) ?? null;
  const placeholderPaintMeasurement = localPendingMessages.find(
    (message) => (
      message.pending_kind === "narrator_placeholder"
      && message.paint_started_at_ms !== undefined
    ),
  ) ?? null;
  const activeSaveId = model?.active_save_id ?? null;
  const canMutatePresence = !mutationsDisabled && currentUser?.role !== "child";
  const [editing, setEditing] = useState<ChronicleMessage | null>(null);
  const [feedbackMessage, setFeedbackMessage] = useState<ChronicleMessage | null>(null);
  const [forkFromHere, setForkFromHere] = useState<ChronicleMessage | null>(null);
  const [deleteFromHere, setDeleteFromHere] = useState<ChronicleMessage | null>(null);
  const [presenceMessage, setPresenceMessage] = useState<ChronicleMessage | null>(null);
  const [characterImageMessage, setCharacterImageMessage] = useState<ChronicleMessage | null>(null);
  const [inspection, setInspection] = useState<{ title: string; text: string } | null>(null);
  const {
    clearJobActionState,
    jobActionErrors,
    pendingJobActionKeys,
    startJobAction
  } = useJobActionRunner(runJob);
  useEffect(() => {
    clearJobActionState();
  }, [activeSaveId, clearJobActionState]);
  useEffect(() => {
    setEditing(null);
    setFeedbackMessage(null);
    setForkFromHere(null);
    setDeleteFromHere(null);
    setPresenceMessage(null);
    setCharacterImageMessage(null);
  }, [activeSaveId, mutationsDisabled]);
  const refreshScenePresenceDependencies = () => {
    queryClient.invalidateQueries({ queryKey: ["runtime"] });
    queryClient.invalidateQueries({ queryKey: ["media"] });
    queryClient.invalidateQueries({ queryKey: ["characters"] });
    invalidateScenePresenceQueries(queryClient, activeSaveId);
    if (activeSaveId) {
      queryClient.invalidateQueries({ queryKey: ["characters", activeSaveId] });
    }
  };
  const scrollRef = useRef<HTMLElement | null>(null);
  const reportedPaintMeasurementsRef = useRef(new Set<string>());
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const scrollMetricsRef = useRef<{ activeSaveId: string | null; scrollHeight: number } | null>(null);
  const chronicleAnchorFrameRef = useRef<number | null>(null);
  const [hasNewContentBelow, setHasNewContentBelow] = useState(false);
  const [olderChronicleLoading, setOlderChronicleLoading] = useState(false);
  const [olderChronicleError, setOlderChronicleError] = useState("");
  const firstMessage = messages[0] ?? null;
  const latestMessage = messages[messages.length - 1] ?? null;
  const messageWindowSignal = [
    activeSaveId ?? "",
    messages.length,
    firstMessage?.message_id ?? "",
    latestMessage?.message_id ?? "",
    latestMessage?.revision_count ?? 0,
    latestMessage?.body.length ?? 0
  ].join(":");
  useLayoutEffect(() => {
    const root = scrollRef.current;
    if (!root) return;
    const recordPaint = (
      event: string,
      messageId: string,
      startedAtMs: number,
      token: string,
    ) => {
      if (reportedPaintMeasurementsRef.current.has(token)) return;
      const target = root.querySelector(
        `[data-message-id="${CSS.escape(messageId)}"]`,
      );
      if (!target) return;
      reportedPaintMeasurementsRef.current.add(token);
      scheduleClientPaintEvent(event, startedAtMs, target);
    };
    if (optimisticPaintMeasurement?.paint_started_at_ms !== undefined) {
      recordPaint(
        "client.chat.optimistic_player_painted",
        optimisticPaintMeasurement.message_id,
        optimisticPaintMeasurement.paint_started_at_ms,
        `optimistic:${optimisticPaintMeasurement.paint_started_at_ms}`,
      );
    }
    if (placeholderPaintMeasurement?.paint_started_at_ms !== undefined) {
      recordPaint(
        "client.chat.placeholder_painted",
        placeholderPaintMeasurement.message_id,
        placeholderPaintMeasurement.paint_started_at_ms,
        `placeholder:${placeholderPaintMeasurement.paint_started_at_ms}`,
      );
    }
    if (narratorPaintMeasurement?.saveId === activeSaveId) {
      recordPaint(
        "client.chat.narrator_painted",
        narratorPaintMeasurement.messageId,
        narratorPaintMeasurement.startedAtMs,
        `narrator:${narratorPaintMeasurement.jobId}`,
      );
    }
  }, [messageWindowSignal, narratorPaintMeasurement, optimisticPaintMeasurement, placeholderPaintMeasurement]);
  const oldestMessageId = model?.chronicle?.oldest_message_id ?? messages[0]?.message_id ?? null;
  const chronicleVirtualizer = useVirtualizer<HTMLElement, HTMLDivElement>({
    count: messages.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => CHRONICLE_ESTIMATED_ROW_HEIGHT,
    getItemKey: (index) => `${activeSaveId ?? ""}:${messages[index]?.message_id ?? index}`,
    gap: CHRONICLE_ROW_GAP,
    initialOffset: () => initialVirtualBottomOffset(messages.length, CHRONICLE_ESTIMATED_ROW_HEIGHT, CHRONICLE_ROW_GAP),
    initialRect: VIRTUAL_LIST_INITIAL_RECT,
    observeElementOffset: (_instance, callback) => observeVirtualElementOffset(scrollRef.current, callback),
    observeElementRect: (_instance, callback) => observeVirtualElementRect(scrollRef.current, callback),
    overscan: CHRONICLE_ROW_OVERSCAN,
    useFlushSync: false,
    measureElement: (element) => {
      const height = element.getBoundingClientRect().height;
      return height > 0 ? height : CHRONICLE_ESTIMATED_ROW_HEIGHT;
    }
  });
  const virtualChronicleRows = chronicleVirtualizer.getVirtualItems();
  const chronicleIsNearBottom = useCallback((node: HTMLElement, scrollHeight = node.scrollHeight) => {
    return scrollHeight - node.scrollTop - node.clientHeight <= 96;
  }, []);
  const cancelPendingChronicleAnchor = useCallback(() => {
    if (chronicleAnchorFrameRef.current === null) return;
    window.cancelAnimationFrame(chronicleAnchorFrameRef.current);
    chronicleAnchorFrameRef.current = null;
  }, []);
  const scrollToChronicleBottom = useCallback(() => {
    const node = scrollRef.current;
    if (!node) return;
    if (messages.length) {
      chronicleVirtualizer.scrollToIndex(messages.length - 1, { align: "end" });
    }
    const scrollHeight = Math.max(node.scrollHeight, chronicleVirtualizer.getTotalSize());
    bottomRef.current?.scrollIntoView?.({ block: "end" });
    setScrollTopAndNotify(node, scrollHeight);
    scrollMetricsRef.current = { activeSaveId, scrollHeight };
    setHasNewContentBelow(false);
  }, [activeSaveId, chronicleVirtualizer, messages.length]);
  useLayoutEffect(() => {
    const node = scrollRef.current;
    if (!node) return;
    const previousMetrics = scrollMetricsRef.current;
    const activeSaveChanged = previousMetrics?.activeSaveId !== activeSaveId;
    const wasNearBottom = previousMetrics ? chronicleIsNearBottom(node, previousMetrics.scrollHeight) : true;
    if (activeSaveChanged || wasNearBottom) {
      scrollToChronicleBottom();
      const frame = window.requestAnimationFrame(() => {
        chronicleAnchorFrameRef.current = null;
        scrollToChronicleBottom();
      });
      chronicleAnchorFrameRef.current = frame;
      return () => {
        if (chronicleAnchorFrameRef.current === frame) {
          window.cancelAnimationFrame(frame);
          chronicleAnchorFrameRef.current = null;
        }
      };
    }
    scrollMetricsRef.current = { activeSaveId, scrollHeight: node.scrollHeight };
    setHasNewContentBelow(true);
    return undefined;
  }, [activeSaveId, chronicleIsNearBottom, messageWindowSignal, scrollToChronicleBottom]);
  const onChronicleScroll = () => {
    const node = scrollRef.current;
    if (!node) return;
    if (chronicleIsNearBottom(node)) {
      setHasNewContentBelow(false);
    } else {
      cancelPendingChronicleAnchor();
    }
    scrollMetricsRef.current = { activeSaveId, scrollHeight: node.scrollHeight };
  };
  const loadOlderChronicle = useCallback(async () => {
    if (!activeSaveId || !oldestMessageId || olderChronicleLoading) return;
    cancelPendingChronicleAnchor();
    const node = scrollRef.current;
    const previousScrollHeight = node?.scrollHeight ?? 0;
    setOlderChronicleLoading(true);
    setOlderChronicleError("");
    try {
      const page = await api<ChronicleModel>(chroniclePagePath(activeSaveId, oldestMessageId));
      queryClient.setQueryData<RuntimeModel>(
        runtimeQueryKey(activeSaveId),
        (current) => mergeChroniclePage(current, page),
      );
      window.requestAnimationFrame(() => {
        if (!node) return;
        const nextScrollHeight = node.scrollHeight;
        setScrollTopAndNotify(node, node.scrollTop + nextScrollHeight - previousScrollHeight);
        scrollMetricsRef.current = {
          activeSaveId,
          scrollHeight: nextScrollHeight
        };
      });
    } catch (failure) {
      setOlderChronicleError(
        failure instanceof Error ? failure.message : "Could not load earlier chronicle",
      );
    } finally {
      setOlderChronicleLoading(false);
    }
  }, [activeSaveId, cancelPendingChronicleAnchor, oldestMessageId, olderChronicleLoading]);
  const handleChronicleAction = useCallback((actionId: string, message: ChronicleMessage) => {
    const jobRequest = chronicleJobActionRequest(actionId, message.message_id, activeSaveId);
    if (jobRequest) {
      void startJobAction({
        key: chronicleJobActionKey(message.message_id, actionId),
        ...jobRequest
      });
      return;
    }
    if (actionId === "edit-and-resubmit-message" || actionId === "edit-narrator-message") setEditing(message);
    if (actionId === "regenerate-message-with-feedback") setFeedbackMessage(message);
    if (actionId === "fork-from-here") setForkFromHere(message);
    if (actionId === "delete-messages-from-here") setDeleteFromHere(message);
    if (actionId === "view-characters-present") setPresenceMessage(message);
    if (actionId === "generate-character-image") setCharacterImageMessage(message);
    if (actionId === "inspect-debug-prompt" || actionId === "inspect-provider-payload") {
      const action = message.actions.find((candidate) => candidate.action_id === actionId);
      setInspection({ title: action?.label ?? "Inspect", text: action?.detail_text ?? "" });
    }
  }, [activeSaveId, startJobAction]);
  return (
    <div className="chronicle-region">
      <section
        className="chronicle-scroll"
        ref={scrollRef}
        role="log"
        aria-label="Chronicle"
        aria-live={olderChronicleLoading ? "off" : "polite"}
        aria-relevant="additions text"
        aria-atomic="false"
        onScroll={onChronicleScroll}
      >
      {model?.chronicle?.has_more_before ? (
        <button
          type="button"
          className="chronicle-load-earlier"
          onClick={loadOlderChronicle}
          disabled={olderChronicleLoading}
        >
          {olderChronicleLoading ? <Loader2 className="spin" size={15} /> : <ArrowUp size={15} />}
          {olderChronicleLoading ? "Loading..." : "Load earlier"}
        </button>
      ) : null}
      {olderChronicleError ? <InlineNotice>{olderChronicleError}</InlineNotice> : null}
      {messages.length ? (
        <div
          className="chronicle-virtual-list"
          style={{ height: `${chronicleVirtualizer.getTotalSize()}px` }}
        >
          {virtualChronicleRows.map((virtualRow) => {
            const message = messages[virtualRow.index];
            if (!message) return null;
            return (
              <div
                key={virtualRow.key}
                ref={chronicleVirtualizer.measureElement}
                className="chronicle-virtual-row"
                data-index={virtualRow.index}
                style={{ transform: `translateY(${virtualRow.start}px)` }}
              >
                <ChronicleMessageRow
                  message={message}
                  activeSaveId={activeSaveId}
                  jobActionErrors={jobActionErrors}
                  pendingJobActionKeys={pendingJobActionKeys}
                  mutationsDisabled={mutationsDisabled}
                  storytellerMode={model?.interaction_mode === "storyteller"}
                  sceneImageArriving={sceneArrivalMessageIds.has(message.message_id)}
                  onAction={handleChronicleAction}
                  onCancelNarrator={onCancelNarrator}
                />
              </div>
            );
          })}
        </div>
      ) : (
        <EmptyState icon={<MessageSquare size={34} />} title="No chronicle loaded" body="Create, import, or start a saved scenario to begin." />
      )}
      {editing ? <EditMessageModal message={editing} activeSaveId={activeSaveId} onClose={() => setEditing(null)} runJob={runJob} /> : null}
      {feedbackMessage ? <RegenerateFeedbackModal message={feedbackMessage} activeSaveId={activeSaveId} onClose={() => setFeedbackMessage(null)} runJob={runJob} /> : null}
      {presenceMessage ? (
        <ScenePresenceDialog
          message={presenceMessage}
          activeSaveId={activeSaveId}
          canMutate={canMutatePresence}
          onClose={() => setPresenceMessage(null)}
          onSaved={refreshScenePresenceDependencies}
        />
      ) : null}
      {characterImageMessage ? (
        <CharacterImageChooserDialog
          message={characterImageMessage}
          activeSaveId={activeSaveId}
          runJob={runJob}
          onClose={() => setCharacterImageMessage(null)}
          onStarted={refreshScenePresenceDependencies}
        />
      ) : null}
      {forkFromHere ? (
        <ConfirmModal
          title="Fork from here?"
          body="This creates a new save ending at the selected message and switches to it."
          confirmLabel="Fork from here"
          onCancel={() => setForkFromHere(null)}
          onConfirm={async () => {
            const sourceMessageId = forkFromHere.message_id;
            await startJobAction({
              key: chronicleJobActionKey(sourceMessageId, "fork-from-here"),
              path: "/api/chat/fork-from-here",
              body: {
                message_id: sourceMessageId,
                save_id: activeSaveId
              }
            });
            setForkFromHere(null);
          }}
        />
      ) : null}
      {deleteFromHere ? (
        <ConfirmModal
          title="Delete from here?"
          body="This hides the selected turn and every later turn from the chronicle and future narrator context."
          confirmLabel="Delete from here"
          destructive
          onCancel={() => setDeleteFromHere(null)}
          onConfirm={async () => {
            const job = await postJson<Job>("/api/chat/delete-from-here", {
              message_id: deleteFromHere.message_id,
              save_id: activeSaveId
            });
            setDeleteFromHere(null);
            runJob(job);
          }}
        />
      ) : null}
      {inspection ? <InspectModal title={inspection.title} text={inspection.text} onClose={() => setInspection(null)} /> : null}
      <div ref={bottomRef} aria-hidden="true" />
      </section>
    {hasNewContentBelow ? (
      <button type="button" className="chronicle-jump-latest" onClick={scrollToChronicleBottom}>
        <ArrowDown size={16} aria-hidden="true" />
        Jump to latest
      </button>
    ) : null}
    </div>
  );
}

function PendingJobsTray({
  jobs,
  mode = "compact",
  onCancel
}: {
  jobs: TrackedJob[];
  mode?: PendingJobsDisplayMode;
  onCancel: (job: TrackedJob) => void;
}) {
  const [compactDetailsOpen, setCompactDetailsOpen] = useState(false);
  const displayedJobs = visiblePendingJobs(jobs, mode);
  if (!displayedJobs.length) return null;
  const expanded = mode === "expanded" || mode === "expanded_full";
  const expandedFull = mode === "expanded_full";
  const compactCanExpand = !expanded && displayedJobs.length > 1;
  const compactDetailsVisible = compactCanExpand && compactDetailsOpen;
  return (
    <section className={expanded ? "pending-jobs-tray expanded" : "pending-jobs-tray compact"} aria-label="Pending jobs" aria-live="polite">
      <div className="pending-jobs-head">
        <span>Pending jobs</span>
        <strong>{displayedJobs.length}</strong>
      </div>
      <div className="pending-job-list">
        {expanded ? (
          displayedJobs.map((tracked) => (
            <PendingJobRow key={tracked.job.id} tracked={tracked} onCancel={onCancel} expanded expandedFull={expandedFull} />
          ))
        ) : (
          <>
            <PendingJobsCompactSummary
              jobs={displayedJobs}
              detailsOpen={compactDetailsVisible}
              canShowDetails={compactCanExpand}
              onCancel={onCancel}
              onToggleDetails={() => setCompactDetailsOpen((open) => !open)}
            />
            {compactDetailsVisible ? (
              displayedJobs.map((tracked) => (
                <PendingJobRow key={tracked.job.id} tracked={tracked} onCancel={onCancel} />
              ))
            ) : null}
          </>
        )}
      </div>
    </section>
  );
}

function visiblePendingJobs(jobs: TrackedJob[], mode: PendingJobsDisplayMode): TrackedJob[] {
  if (mode === "expanded_full") return jobs;
  return jobs.filter((tracked) => tracked.job.type !== "state_pruning");
}

function trackedActiveJob(created: Job, existing?: TrackedJob): TrackedJob {
  const firstProgress = created.status === "running" ? "Running" : "Queued";
  const latestProgress = created.latest_progress ?? null;
  const progressFromJob = latestProgress === null ? null : progressLabel(latestProgress);
  const phasesFromJob = postTurnProgressPhases(latestProgress);
  const preservedProgress = existing?.progress && !["Queued", "Running"].includes(existing.progress) ? existing.progress : null;
  const replacesCatchup = isPostTurnCatchupProgress(latestProgress)
    || isPostTurnCatchupProgress(existing?.job.latest_progress);
  const nextProgress = phasesFromJob || replacesCatchup ? progressFromJob : null;
  return {
    job: created,
    progress: nextProgress ?? preservedProgress ?? progressFromJob ?? firstProgress,
    phases: phasesFromJob ?? (replacesCatchup ? undefined : existing?.phases)
  };
}

function PendingJobsCompactSummary({
  jobs,
  detailsOpen = false,
  canShowDetails = false,
  onCancel,
  onToggleDetails
}: {
  jobs: TrackedJob[];
  detailsOpen?: boolean;
  canShowDetails?: boolean;
  onCancel: (job: TrackedJob) => void;
  onToggleDetails?: () => void;
}) {
  const summary = jobs.length === 1
    ? postTurnCompletionLabel(jobs[0].job) || jobs[0].progress
    : compactJobGroups(jobs).map((group) => group.count > 1 ? `${group.label} x${group.count}` : group.label).join("; ");
  return (
    <div className="pending-job-row compact-summary">
      <Loader2 className="spin" size={15} />
      <div>
        <strong>Active jobs</strong>
        {jobs.length === 1 ? <span>{jobTypeLabel(jobs[0].job.type)}</span> : null}
        <span>{summary}</span>
      </div>
      {jobs.length === 1 ? (
        <button
          type="button"
          title="Cancel job"
          aria-label={`Cancel ${jobTypeLabel(jobs[0].job.type)}`}
          onClick={() => onCancel(jobs[0])}
        >
          <Square size={14} />
        </button>
      ) : canShowDetails ? (
        <button
          type="button"
          className="pending-job-details-toggle"
          aria-expanded={detailsOpen}
          aria-label={detailsOpen ? "Hide pending jobs" : "Show pending jobs"}
          title={detailsOpen ? "Hide pending jobs" : "Show pending jobs"}
          onClick={onToggleDetails}
        >
          <ChevronDown size={15} aria-hidden="true" />
        </button>
      ) : (
        <span aria-hidden="true" />
      )}
    </div>
  );
}

function PendingJobRow({
  tracked,
  onCancel,
  expanded = false,
  expandedFull = false
}: {
  tracked: TrackedJob;
  onCancel: (job: TrackedJob) => void;
  expanded?: boolean;
  expandedFull?: boolean;
}) {
  const label = jobTypeLabel(tracked.job.type);
  const progress = postTurnCompletionLabel(tracked.job)
    ?? (tracked.progress || labelize(tracked.job.status));
  const visiblePhases = expanded ? visiblePostTurnPhases(tracked.phases, expandedFull) : [];
  const longRunningHint = useLongRunningJobHint(tracked);
  return (
    <div className={visiblePhases.length ? "pending-job-row expanded-with-phases" : "pending-job-row"}>
      <Loader2 className="spin" size={15} />
      <div>
        <strong>{label}</strong>
        <span>{progress}</span>
        {longRunningHint ? (
          <span className="pending-job-hint">{longRunningHint}</span>
        ) : null}
        {visiblePhases.length ? (
          <div className="pending-job-phases">
            {visiblePhases.map((phase) => (
              <div className="pending-job-phase" key={phase.name}>
                <span>{postTurnPhaseLabel(phase.name)}</span>
                <strong>{postTurnPhaseStatusLabel(phase.status)}</strong>
              </div>
            ))}
          </div>
        ) : null}
      </div>
      <button
        type="button"
        title="Cancel job"
        aria-label={`Cancel ${label}`}
        onClick={() => onCancel(tracked)}
      >
        <Square size={14} />
      </button>
    </div>
  );
}

const LONG_RUNNING_JOB_HINT_SECONDS = 60;
const CANCEL_STUCK_HINT_TOTAL_SECONDS = 15;

function useLongRunningJobHint(tracked: TrackedJob): string | null {
  const [now, setNow] = useState(() => Date.now());
  const createdAt = tracked.job.created_at;
  useEffect(() => {
    if (!createdAt) return;
    const timer = window.setInterval(() => setNow(Date.now()), 5000);
    return () => window.clearInterval(timer);
  }, [createdAt]);
  if (!createdAt) return null;
  const elapsedSeconds = Math.max(0, Math.floor((now - createdAt * 1000) / 1000));
  const isCancelling = tracked.progress === "Cancelling";
  if (isCancelling && elapsedSeconds >= CANCEL_STUCK_HINT_TOTAL_SECONDS) {
    return "Cancelling. The current provider call cannot be interrupted until it completes; you can keep waiting or close this view.";
  }
  if (elapsedSeconds >= LONG_RUNNING_JOB_HINT_SECONDS) {
    return "This is taking longer than usual. You can cancel and try again.";
  }
  return null;
}

function EmptyState({
  icon,
  title,
  body,
  action
}: {
  icon: React.ReactNode;
  title: string;
  body: string;
  action?: React.ReactNode;
}) {
  return (
    <div className="empty-state">
      {icon}
      <h2>{title}</h2>
      <p>{body}</p>
      {action ? <div className="empty-state-action">{action}</div> : null}
    </div>
  );
}

function ScenePresenceDialog({
  message,
  activeSaveId,
  canMutate,
  onClose,
  onSaved
}: {
  message: ChronicleMessage;
  activeSaveId: string | null;
  canMutate: boolean;
  onClose: () => void;
  onSaved: () => void;
}) {
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const titleId = React.useId();
  const presence = useQuery({
    queryKey: ["scene-presence", activeSaveId, message.message_id],
    queryFn: () => api<ScenePresenceModel>(`/api/messages/${encodeURIComponent(message.message_id)}/scene-presence?save_id=${encodeURIComponent(activeSaveId ?? "")}`),
    enabled: Boolean(activeSaveId)
  });
  useEffect(() => {
    if (!presence.data) return;
    setSelectedIds(new Set(presence.data.characters.filter((character) => character.present).map((character) => character.character_id)));
  }, [presence.data]);
  const toggle = (characterId: string) => {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(characterId)) next.delete(characterId);
      else next.add(characterId);
      return next;
    });
  };
  const save = async () => {
    if (!activeSaveId || !canMutate) return;
    setSaving(true);
    setError("");
    try {
      await postJson<ScenePresenceModel>(`/api/messages/${encodeURIComponent(message.message_id)}/scene-presence`, {
        save_id: activeSaveId,
        character_ids: Array.from(selectedIds)
      });
      onSaved();
      onClose();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not save scene presence");
    } finally {
      setSaving(false);
    }
  };
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog scene-presence-dialog" titleId={titleId} onClose={onClose}>
        <header>
          <h2 id={titleId}>Characters present</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        {presence.isLoading ? <p className="muted">Loading...</p> : null}
        {presence.error instanceof Error ? <InlineNotice>{presence.error.message}</InlineNotice> : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="presence-list">
          {(presence.data?.characters ?? []).map((character) => {
            const checked = selectedIds.has(character.character_id);
            return (
              <label className="presence-row" key={character.character_id}>
                <input
                  type="checkbox"
                  checked={checked}
                  disabled={!canMutate}
                  onChange={() => toggle(character.character_id)}
                />
                <span>
                  <strong>{character.name}</strong>
                  <small>{[
                    character.is_player_character ? "Player" : "",
                    character.status,
                    character.has_reference_image ? "Reference" : ""
                  ].filter(Boolean).join(" · ")}</small>
                </span>
              </label>
            );
          })}
        </div>
        <div className="command-row end">
          <button type="button" onClick={onClose}>{canMutate ? "Cancel" : "Close"}</button>
          {canMutate ? (
            <button type="button" className="primary-command compact" disabled={saving || presence.isLoading} onClick={save}>
              <Save size={15} /> Save
            </button>
          ) : null}
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function CharacterImageChooserDialog({
  message,
  activeSaveId,
  runJob,
  onClose,
  onStarted
}: {
  message: ChronicleMessage;
  activeSaveId: string | null;
  runJob: RunJob;
  onClose: () => void;
  onStarted: () => void;
}) {
  const [selectedCharacterId, setSelectedCharacterId] = useState("");
  const [error, setError] = useState("");
  const titleId = React.useId();
  const presence = useQuery({
    queryKey: ["scene-presence", activeSaveId, message.message_id],
    queryFn: () => api<ScenePresenceModel>(`/api/messages/${encodeURIComponent(message.message_id)}/scene-presence?save_id=${encodeURIComponent(activeSaveId ?? "")}`),
    enabled: Boolean(activeSaveId)
  });
  const eligible = useMemo(
    () => (presence.data?.characters ?? []).filter((character) => (
      character.present && character.has_reference_image
    )),
    [presence.data?.characters]
  );
  useEffect(() => {
    if (!selectedCharacterId && eligible[0]) setSelectedCharacterId(eligible[0].character_id);
  }, [eligible, selectedCharacterId]);
  const selectedCharacter = eligible.find((character) => character.character_id === selectedCharacterId) ?? eligible[0] ?? null;
  const start = async () => {
    if (!activeSaveId || !selectedCharacter) return;
    setError("");
    try {
      const job = await postJson<Job>("/api/media/generate-character-image", {
        save_id: activeSaveId,
        message_id: message.message_id,
        character_id: selectedCharacter.character_id
      });
      onStarted();
      onClose();
      runJob(job, { onSucceeded: onStarted });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not start character image");
    }
  };
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog character-image-dialog" titleId={titleId} onClose={onClose}>
        <header>
          <h2 id={titleId}>Generate character image</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        {presence.isLoading ? <p className="muted">Loading...</p> : null}
        {presence.error instanceof Error ? <InlineNotice>{presence.error.message}</InlineNotice> : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="character-image-choice-list">
          {eligible.map((character) => (
            <label className="character-image-choice" key={character.character_id}>
              <input
                type="radio"
                name={`character-image-${message.message_id}`}
                checked={selectedCharacter?.character_id === character.character_id}
                onChange={() => setSelectedCharacterId(character.character_id)}
              />
              {character.reference_image ? (
                <img
                  src={mediaAssetThumbnailPath(character.reference_image.media_asset_id, activeSaveId)}
                  alt={character.name}
                  loading="lazy"
                  decoding="async"
                />
              ) : (
                <span className="character-image-choice-placeholder"><Users size={18} /></span>
              )}
              <span>
                <strong>{character.name}</strong>
                <small>{character.status || "Present"}</small>
              </span>
            </label>
          ))}
        </div>
        {!presence.isLoading && !eligible.length ? <InlineNotice>No present characters have reference images.</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onClose}>Cancel</button>
          <button type="button" className="primary-command compact" disabled={!selectedCharacter} onClick={start}>
            <Image size={15} /> Generate
          </button>
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function EditMessageModal({
  message,
  activeSaveId,
  onClose,
  runJob
}: {
  message: ChronicleMessage;
  activeSaveId: string | null;
  onClose: () => void;
  runJob: RunJob;
}) {
  const [body, setBody] = useState(message.body);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();
  const isNarratorEdit = message.role === "narrator";
  const isSystemEdit = message.role === "system";
  const unchanged = body.trim() === message.body.trim();
  const submitEdit = async (mode: "save" | "resubmit") => {
    setBusy(true);
    setError("");
    try {
      if (mode === "resubmit") {
        runJob(await postJson<Job>(
          message.role === "system" ? "/api/chat/retry" : "/api/chat/edit",
          { message_id: message.message_id, body, save_id: activeSaveId }
        ));
      } else if (isNarratorEdit) {
        runJob(await postJson<Job>("/api/chat/narrator-edit", {
          message_id: message.message_id,
          body,
          save_id: activeSaveId
        }));
      } else {
        runJob(await postJson<Job>("/api/chat/message-edit", {
          message_id: message.message_id,
          body,
          save_id: activeSaveId
        }));
      }
      onClose();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not submit edit");
      setBusy(false);
    }
  };
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog message-edit-dialog"
        titleId={titleId}
        onClose={onClose}
        onSubmit={async (event) => {
          event.preventDefault();
          await submitEdit(isNarratorEdit ? "save" : "resubmit");
        }}
      >
        <header>
          <h2 id={titleId}>Edit message</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Message</span>
          <textarea className="tall-field" value={body} onChange={(event) => setBody(event.target.value)} />
        </label>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end message-edit-actions">
          <button type="button" onClick={onClose}>Cancel</button>
          {isNarratorEdit ? (
            <button type="submit" className="primary-command compact" disabled={busy || !body.trim() || unchanged}>
              <Save size={15} /> Save
            </button>
          ) : (
            <>
              {!isSystemEdit ? (
                <button type="button" className="primary-command compact" disabled={busy || !body.trim() || unchanged} onClick={() => submitEdit("save")}>
                  <Save size={15} /> Edit without Resubmit
                </button>
              ) : null}
              <button type="submit" className="primary-command compact" disabled={busy || !body.trim()}>
                <RefreshCw size={15} /> Resubmit
              </button>
            </>
          )}
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function RegenerateFeedbackModal({ message, activeSaveId, onClose, runJob }: { message: ChronicleMessage; activeSaveId: string | null; onClose: () => void; runJob: RunJob }) {
  const [feedback, setFeedback] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog message-edit-dialog"
        titleId={titleId}
        onClose={onClose}
        onSubmit={async (event) => {
          event.preventDefault();
          setBusy(true);
          setError("");
          try {
            runJob(await postJson<Job>("/api/chat/regenerate", {
              message_id: message.message_id,
              save_id: activeSaveId,
              regeneration_feedback: feedback
            }));
            onClose();
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : "Could not start regeneration");
            setBusy(false);
          }
        }}
      >
        <header>
          <h2 id={titleId}>Regenerate with feedback</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Feedback</span>
          <textarea
            className="tall-field"
            value={feedback}
            onChange={(event) => setFeedback(event.target.value)}
            aria-label="Feedback"
          />
        </label>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end message-edit-actions">
          <button type="button" onClick={onClose}>Cancel</button>
          <button type="submit" className="primary-command compact" disabled={busy || !feedback.trim()}>
            <RefreshCw size={15} /> Regenerate
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

const MarkdownView = React.memo(function MarkdownView({
  message,
  body,
  markdownBlocks,
  className = "prose"
}: {
  message?: ChronicleMessage;
  body?: string;
  markdownBlocks?: MarkdownBlock[];
  className?: string;
}) {
  const resolvedBody = body ?? message?.body ?? "";
  const blocks = markdownBlocks ?? message?.markdown_blocks ?? [];
  if (!blocks.length) return <div className={className}>{resolvedBody}</div>;
  return (
    <div className={className}>
      {blocks.map((block, index) => {
        const kind = block.block_type ?? block.kind;
        if (kind === "code_block") return <pre key={index}><code>{block.text ?? ""}</code></pre>;
        if (kind === "blockquote") return <blockquote key={index}>{renderSpans(block.spans, block.text)}</blockquote>;
        if (kind === "bullet_item") return <p key={index}>- {renderSpans(block.spans, block.text)}</p>;
        if (kind === "numbered_item") return <p key={index}>{block.ordinal ?? index + 1}. {renderSpans(block.spans, block.text)}</p>;
        if (kind === "thematic_break") return <hr key={index} />;
        return <p key={index}>{renderSpans(block.spans, block.text)}</p>;
      })}
    </div>
  );
});

function renderSpans(spans: { kind: string; text: string; target?: string | null }[] | undefined, fallback?: string) {
  if (!spans?.length) return fallback ?? "";
  return spans.map((span, index) => {
    if (span.kind === "strong") return <strong key={index}>{span.text}</strong>;
    if (span.kind === "emphasis") return <em key={index}>{span.text}</em>;
    if (span.kind === "inline_code") return <code key={index}>{span.text}</code>;
    if (span.kind === "link" && span.target) {
      const href = safeMarkdownLinkTarget(span.target);
      if (href) return <a key={index} href={href} rel="noreferrer" target="_blank">{span.text}</a>;
    }
    return <React.Fragment key={index}>{span.text}</React.Fragment>;
  });
}

const SAFE_MARKDOWN_LINK_PROTOCOLS = new Set(["http:", "https:", "mailto:"]);
const EXPLICIT_URL_PROTOCOL = /^[a-z][a-z\d+.-]*:/i;

function safeMarkdownLinkTarget(target: string) {
  const href = target.trim();
  if (!href) return null;

  try {
    const baseOrigin = markdownLinkBaseOrigin();
    const parsed = new URL(href, baseOrigin);
    if (!EXPLICIT_URL_PROTOCOL.test(href)) return parsed.origin === baseOrigin ? href : null;
    return SAFE_MARKDOWN_LINK_PROTOCOLS.has(parsed.protocol) ? href : null;
  } catch {
    return null;
  }
}

function markdownLinkBaseOrigin() {
  if (typeof window === "undefined" || !window.location.origin || window.location.origin === "null") {
    return "http://localhost";
  }
  return window.location.origin;
}

function InspectModal({ title, text, onClose }: { title: string; text: string; onClose: () => void }) {
  const sections = splitInspectionText(text);
  const [activeTab, setActiveTab] = useState<"sources" | "raw">("sources");
  const titleId = React.useId();
  const tabId = React.useId();
  const sourcesTabId = `${tabId}-sources-tab`;
  const rawTabId = `${tabId}-raw-tab`;
  const sourcesPanelId = `${tabId}-sources-panel`;
  const rawPanelId = `${tabId}-raw-panel`;
  const activeTabId = activeTab === "raw" ? rawTabId : sourcesTabId;
  const activePanelId = activeTab === "raw" ? rawPanelId : sourcesPanelId;
  const activeText = sections.raw !== null && activeTab === "raw" ? sections.raw : sections.sources;
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog wide-dialog" titleId={titleId} onClose={onClose}>
        <header>
          <h2 id={titleId}>{title}</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        {sections.raw !== null ? (
          <SegmentedTabs
            className="segmented inspect-tabs"
            label="Model request views"
            value={activeTab}
            onChange={setActiveTab}
            options={[
              { value: "sources", label: "Sources", tabId: sourcesTabId, panelId: sourcesPanelId },
              { value: "raw", label: "Raw", tabId: rawTabId, panelId: rawPanelId }
            ]}
          />
        ) : null}
        <pre
          className={`debug-pre ${sections.raw !== null ? "inspect-pre" : ""}`}
          role={sections.raw !== null ? "tabpanel" : undefined}
          id={sections.raw !== null ? activePanelId : undefined}
          aria-labelledby={sections.raw !== null ? activeTabId : undefined}
        >
          {activeText || "No debug payload captured"}
        </pre>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function splitInspectionText(text: string): { sources: string; raw: string | null } {
  const marker = "\nRaw requests\n";
  const index = text.indexOf(marker);
  if (index === -1) return { sources: text, raw: null };
  return {
    sources: text.slice(0, index),
    raw: text.slice(index + marker.length)
  };
}

function actionIcon(actionId: string) {
  if (actionId === "retry-interrupted-turn") return <RefreshCw size={14} />;
  if (actionId === "edit-and-resubmit-message") return <Edit3 size={14} />;
  if (actionId === "edit-text-message") return <Edit3 size={14} />;
  if (actionId === "correct-character-text-message") return <Edit3 size={14} />;
  if (actionId === "edit-and-resubmit-text-message") return <RefreshCw size={14} />;
  if (actionId === "delete-text-messages-from-here") return <Trash2 size={14} />;
  if (actionId === "edit-narrator-message") return <Edit3 size={14} />;
  if (actionId === "regenerate-message-with-feedback") return <MessageSquareText size={14} />;
  if (actionId === "regenerate-message") return <RefreshCw size={14} />;
  if (actionId === "fork-from-here") return <GitBranch size={14} />;
  if (actionId === "delete-messages-from-here") return <Trash2 size={14} />;
  if (actionId === "generate-scene-image") return <Image size={14} />;
  if (actionId === "view-characters-present") return <Users size={14} />;
  if (actionId === "generate-character-image") return <Users size={14} />;
  if (actionId === "inspect-debug-prompt" || actionId === "inspect-provider-payload") return <Eye size={14} />;
  return <FileText size={14} />;
}

function TouchActionContents({ icon, label }: { icon: React.ReactNode; label: string }) {
  return (
    <>
      {icon}
      <span className="touch-action-label">{label}</span>
    </>
  );
}

function touchActionClassName(...classNames: (string | false | null | undefined)[]) {
  return ["touch-labeled-action", ...classNames].filter(Boolean).join(" ");
}

function LookAroundDialog({
  activeSaveId,
  runJob,
  disabled,
  initialQuery = "",
  answer,
  onAnswer,
  onClose
}: {
  activeSaveId: string | null;
  runJob: RunJob;
  disabled: boolean;
  initialQuery?: string;
  answer: LookAroundAnswer | null;
  onAnswer: (answer: LookAroundAnswer | null) => void;
  onClose: () => void;
}) {
  const titleId = "look-around-title";
  const [query, setQuery] = useState(initialQuery);
  const [submitError, setSubmitError] = useState("");
  useEffect(() => {
    setQuery(initialQuery);
  }, [initialQuery]);
  const lookAround = useMutation({
    mutationFn: async (submittedQuery: string) => {
      if (!activeSaveId) throw new Error("No save loaded");
      return postJson<Job>("/api/chat/look-around", {
        save_id: activeSaveId,
        query: submittedQuery
      });
    },
    onSuccess: (job, submittedQuery) => {
      setSubmitError("");
      runJob(job, {
        applyResult: false,
        onSucceeded: (result) => {
          const parsed = lookAroundAnswerFromResult(result, submittedQuery);
          if (parsed) onAnswer(parsed);
        }
      });
    },
    onError: (error) => {
      setSubmitError(error instanceof Error ? error.message : "Could not look around");
    }
  });
  const busy = lookAround.isPending;
  const canSubmit = Boolean(activeSaveId && query.trim() && !disabled && !busy);
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog look-around-dialog" titleId={titleId} onClose={onClose}>
        <div className="dialog-title-row">
          <div>
            <p className="eyebrow">Observation</p>
            <h2 id={titleId}>Look Around</h2>
          </div>
          <button type="button" className="icon-button" aria-label="Close Look Around" title="Close" onClick={onClose}>
            <X size={16} aria-hidden="true" />
          </button>
        </div>
        <form
          className="look-around-form"
          onSubmit={(event) => {
            event.preventDefault();
            const submittedQuery = query.trim();
            if (!canSubmit || !submittedQuery) return;
            lookAround.mutate(submittedQuery);
          }}
        >
          <textarea
            value={query}
            onChange={(event) => setQuery(event.currentTarget.value)}
            disabled={disabled || busy}
            placeholder="Inspect the brass lens..."
            aria-label="Look Around question"
            autoFocus
          />
          <button type="submit" className="primary-command compact" disabled={!canSubmit}>
            {busy ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <Search size={15} aria-hidden="true" />}
            Look
          </button>
        </form>
        {submitError ? <InlineNotice>{submitError}</InlineNotice> : null}
        {answer ? (
          <section className="look-around-answer" aria-live="polite">
            <div className="look-around-query">{answer.query}</div>
            <MarkdownView
              body={answer.answer}
              markdownBlocks={answer.markdownBlocks}
              className="look-around-answer-body"
            />
            {answer.updateCounts?.suggestions ? (
              <small>{answer.updateCounts.suggestions} world update suggestion queued.</small>
            ) : null}
          </section>
        ) : (
          <p className="muted">No observation yet.</p>
        )}
      </DialogPanel>
    </ModalBackdrop>
  );
}

function lookAroundAnswerFromResult(
  result: unknown,
  fallbackQuery: string,
): LookAroundAnswer | null {
  if (!result || typeof result !== "object") return null;
  const record = result as Record<string, unknown>;
  const answer = typeof record.answer === "string" ? record.answer.trim() : "";
  if (!answer) return null;
  const updateCounts = (
    record.update_counts && typeof record.update_counts === "object"
      ? (record.update_counts as Record<string, number>)
      : undefined
  );
  const markdownBlocks = Array.isArray(record.answer_markdown_blocks)
    ? (record.answer_markdown_blocks as MarkdownBlock[])
    : undefined;
  return {
    query: typeof record.query === "string" && record.query.trim()
      ? record.query.trim()
      : fallbackQuery,
    answer,
    markdownBlocks,
    updateCounts
  };
}

type ChatSubmitVariables = {
  body: string;
  saveId: string | null;
  key: string;
  paintStartedAtMs: number;
};

function clientTurnId(): string {
  return crypto.randomUUID?.() ?? "10000000-1000-4000-8000-100000000000".replace(
    /[018]/g,
    (digit) => (
      Number(digit)
      ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (Number(digit) / 4)))
    ).toString(16)
  );
}

function chatSubmitVariables(
  body: string,
  saveId: string | null,
  previous: ChatSubmitVariables | null,
  paintStartedAtMs: number,
): ChatSubmitVariables {
  if (previous && previous.saveId === saveId && previous.body.trim() === body.trim()) {
    return { ...previous, paintStartedAtMs };
  }
  return { body, saveId, key: clientTurnId(), paintStartedAtMs };
}

function pendingPlayerChronicleMessage(
  body: string,
  saveId: string | null,
  pendingAfterMessageId: string | null,
  paintStartedAtMs: number,
): PendingChronicleMessage {
  return {
    message_id: "pending-player-message",
    role: "player",
    speaker_name: null,
    body,
    actions: [],
    markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: body }] }],
    pending_after_message_id: pendingAfterMessageId,
    pending_save_id: saveId,
    paint_started_at_ms: paintStartedAtMs,
    pending_kind: "player",
  };
}

type ComposerFormatActionId = "narration" | "dialogue" | "text_message" | "clear";
type ComposerFormatResult = {
  body: string;
  selectionStart: number;
  selectionEnd: number;
};
type ComposerFormatAction = {
  actionId: ComposerFormatActionId;
  label: string;
  title: string;
  shortcut: string;
  icon: LucideIcon;
};

const COMPOSER_FORMAT_ACTIONS: readonly ComposerFormatAction[] = [
  {
    actionId: "narration",
    label: "Format as narration",
    title: "Narration (Alt+N)",
    shortcut: "Alt+N",
    icon: Italic
  },
  {
    actionId: "dialogue",
    label: "Format as dialogue",
    title: "Dialogue (Alt+Q)",
    shortcut: "Alt+Q",
    icon: Quote
  },
  {
    actionId: "text_message",
    label: "Format as text message",
    title: "Text message (Alt+M)",
    shortcut: "Alt+M",
    icon: MessageSquareText
  },
  {
    actionId: "clear",
    label: "Clear roleplay formatting",
    title: "Clear roleplay formatting (Alt+0)",
    shortcut: "Alt+0",
    icon: RemoveFormatting
  }
];

function formatComposerBody(
  body: string,
  selectionStart: number,
  selectionEnd: number,
  actionId: ComposerFormatActionId
): ComposerFormatResult {
  const start = Math.max(0, Math.min(selectionStart, selectionEnd, body.length));
  const end = Math.max(0, Math.min(Math.max(selectionStart, selectionEnd), body.length));
  if (actionId === "narration") return formatComposerWrappedSelection(body, start, end, "*", "*");
  if (actionId === "dialogue") return formatComposerWrappedSelection(body, start, end, "\"", "\"");
  if (actionId === "text_message") return formatComposerLineSelection(body, start, end, formatComposerTextMessageLines);
  return formatComposerTextSelection(body, start, end, clearComposerRoleplayFormatting);
}

function formatComposerWrappedSelection(
  body: string,
  selectionStart: number,
  selectionEnd: number,
  prefix: string,
  suffix: string
): ComposerFormatResult {
  const range = selectionStart === selectionEnd
    ? currentComposerLineRange(body, selectionStart)
    : { start: selectionStart, end: selectionEnd };
  const selected = body.slice(range.start, range.end);
  if (!selected) {
    const replacement = `${prefix}${suffix}`;
    return replaceComposerRange(
      body,
      range.start,
      range.end,
      replacement,
      range.start + prefix.length,
      range.start + prefix.length
    );
  }
  if (hasComposerWrapper(selected, prefix, suffix)) {
    const replacement = selected.slice(prefix.length, selected.length - suffix.length);
    return replaceComposerRange(
      body,
      range.start,
      range.end,
      replacement,
      range.start,
      range.start + replacement.length
    );
  }
  const replacement = `${prefix}${selected}${suffix}`;
  return replaceComposerRange(
    body,
    range.start,
    range.end,
    replacement,
    range.start + prefix.length,
    range.start + prefix.length + selected.length
  );
}

function formatComposerLineSelection(
  body: string,
  selectionStart: number,
  selectionEnd: number,
  formatter: (text: string) => string
): ComposerFormatResult {
  const range = selectionStart === selectionEnd
    ? currentComposerLineRange(body, selectionStart)
    : selectedComposerLineRange(body, selectionStart, selectionEnd);
  const selected = body.slice(range.start, range.end);
  const replacement = formatter(selected);
  return replaceComposerRange(
    body,
    range.start,
    range.end,
    replacement,
    range.start,
    range.start + replacement.length
  );
}

function formatComposerTextSelection(
  body: string,
  selectionStart: number,
  selectionEnd: number,
  formatter: (text: string) => string
): ComposerFormatResult {
  const range = selectionStart === selectionEnd
    ? currentComposerLineRange(body, selectionStart)
    : { start: selectionStart, end: selectionEnd };
  const selected = body.slice(range.start, range.end);
  const replacement = formatter(selected);
  return replaceComposerRange(
    body,
    range.start,
    range.end,
    replacement,
    range.start,
    range.start + replacement.length
  );
}

function currentComposerLineRange(body: string, position: number) {
  const start = body.lastIndexOf("\n", Math.max(0, position - 1)) + 1;
  const nextBreak = body.indexOf("\n", position);
  return {
    start,
    end: nextBreak === -1 ? body.length : nextBreak
  };
}

function selectedComposerLineRange(body: string, selectionStart: number, selectionEnd: number) {
  const start = body.lastIndexOf("\n", Math.max(0, selectionStart - 1)) + 1;
  const endProbe = selectionEnd > selectionStart && body[selectionEnd - 1] === "\n"
    ? selectionEnd - 1
    : selectionEnd;
  const nextBreak = body.indexOf("\n", endProbe);
  return {
    start,
    end: nextBreak === -1 ? body.length : nextBreak
  };
}

function replaceComposerRange(
  body: string,
  start: number,
  end: number,
  replacement: string,
  selectionStart: number,
  selectionEnd: number
): ComposerFormatResult {
  return {
    body: `${body.slice(0, start)}${replacement}${body.slice(end)}`,
    selectionStart,
    selectionEnd
  };
}

function hasComposerWrapper(text: string, prefix: string, suffix: string) {
  if (!text.startsWith(prefix) || !text.endsWith(suffix) || text.length < prefix.length + suffix.length) {
    return false;
  }
  if (prefix === "*" && (text.startsWith("**") || text.endsWith("**"))) {
    return false;
  }
  return true;
}

function formatComposerTextMessageLines(text: string) {
  const lines = text.split("\n");
  const messageLines = lines.filter((line) => line.trim());
  const shouldClear = Boolean(messageLines.length) && messageLines.every((line) => /^\s*>\s?/.test(line));
  return lines
    .map((line) => {
      if (!line.trim()) return line;
      if (shouldClear) return line.replace(/^(\s*)>\s?/, "$1");
      return line.replace(/^(\s*)/, "$1> ");
    })
    .join("\n");
}

function clearComposerRoleplayFormatting(text: string) {
  return text
    .split("\n")
    .map((line) => {
      const unquoted = line.replace(/^(\s*)>\s?/, "$1");
      return stripComposerLineWrapper(stripComposerLineWrapper(unquoted, "*", "*"), "\"", "\"");
    })
    .join("\n");
}

function stripComposerLineWrapper(line: string, prefix: string, suffix: string) {
  const leading = line.match(/^\s*/)?.[0] ?? "";
  const trailing = line.match(/\s*$/)?.[0] ?? "";
  const innerStart = leading.length;
  const innerEnd = line.length - trailing.length;
  const content = line.slice(innerStart, innerEnd);
  if (!hasComposerWrapper(content, prefix, suffix)) return line;
  return `${leading}${content.slice(prefix.length, content.length - suffix.length)}${trailing}`;
}

function composerFormatActionForKey(event: React.KeyboardEvent<HTMLTextAreaElement>): ComposerFormatActionId | null {
  if (!event.altKey || event.ctrlKey || event.metaKey) return null;
  const key = event.key.toLowerCase();
  if (key === "n") return "narration";
  if (key === "q") return "dialogue";
  if (key === "m") return "text_message";
  if (key === "0") return "clear";
  return null;
}

function Composer({
  disabled,
  runJob,
  activeSaveId,
  pendingAfterMessageId = null,
  onPendingMessage,
  storytellerMode = false
}: {
  disabled: boolean;
  runJob: RunJob;
  activeSaveId: string | null;
  pendingAfterMessageId?: string | null;
  onPendingMessage: (message: PendingChronicleMessage | null) => void;
  storytellerMode?: boolean;
}) {
  const [body, setBody] = useState("");
  const [submitError, setSubmitError] = useState("");
  const [timeskipOpen, setTimeskipOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const activeSaveIdRef = useRef(activeSaveId);
  activeSaveIdRef.current = activeSaveId;
  const submittingSaveIdRef = useRef<string | null | undefined>(undefined);
  const [submittingSaveId, setSubmittingSaveId] = useState<string | null | undefined>(undefined);
  const continuingSaveIdRef = useRef<string | null | undefined>(undefined);
  const [continuingSaveId, setContinuingSaveId] = useState<string | null | undefined>(undefined);
  const timeskipSubmittingRef = useRef(false);
  const failedSubmitRef = useRef<ChatSubmitVariables | null>(null);
  const failedContinueRef = useRef<{ saveId: string | null; key: string } | null>(null);
  const failedTimeskipRef = useRef<{ instruction: string; saveId: string | null; key: string } | null>(null);
  const submit = useMutation({
    mutationFn: (submitted: ChatSubmitVariables) => postJson<Job>("/api/chat", { body: submitted.body, speaker_name: null, save_id: submitted.saveId, client_turn_id: submitted.key }),
    onSuccess: (job, submitted) => {
      failedSubmitRef.current = null;
      if (submitted.saveId !== activeSaveIdRef.current) return;
      setSubmitError("");
      runJob(job, { paintStartedAtMs: submitted.paintStartedAtMs });
    },
    onError: (error, submitted) => {
      failedSubmitRef.current = submitted;
      if (submitted.saveId !== activeSaveIdRef.current) return;
      setBody((currentBody) => currentBody ? currentBody : submitted.body);
      onPendingMessage(null);
      setSubmitError(error instanceof Error ? error.message : "Could not send message");
    },
    onSettled: (_data, _error, submitted) => {
      if (submitted && submittingSaveIdRef.current === submitted.saveId) {
        submittingSaveIdRef.current = undefined;
        setSubmittingSaveId(undefined);
      }
    }
  });
  const timeskip = useMutation({
    mutationFn: (submitted: { instruction: string; saveId: string | null; key: string }) => postJson<Job>("/api/chat/timeskip", { instruction: submitted.instruction, save_id: submitted.saveId, client_turn_id: submitted.key }),
    onSuccess: (job) => {
      failedTimeskipRef.current = null;
      runJob(job);
      setTimeskipOpen(false);
    },
    onError: (_error, submitted) => {
      failedTimeskipRef.current = submitted;
    },
    onSettled: () => {
      timeskipSubmittingRef.current = false;
    }
  });
  const continueStory = useMutation({
    mutationFn: (submitted: { saveId: string | null; key: string }) => postJson<Job>("/api/chat/continue", { save_id: submitted.saveId, client_turn_id: submitted.key }),
    onSuccess: (job, submitted) => {
      failedContinueRef.current = null;
      const submittedSaveId = submitted.saveId;
      if (submittedSaveId !== activeSaveIdRef.current) return;
      setSubmitError("");
      runJob(job);
    },
    onError: (error, submitted) => {
      failedContinueRef.current = submitted;
      const submittedSaveId = submitted.saveId;
      if (submittedSaveId !== activeSaveIdRef.current) return;
      setSubmitError(error instanceof Error ? error.message : "Could not continue story");
    },
    onSettled: (_data, _error, submitted) => {
      const submittedSaveId = submitted.saveId;
      if (continuingSaveIdRef.current === submittedSaveId) {
        continuingSaveIdRef.current = undefined;
        setContinuingSaveId(undefined);
      }
    }
  });
  useEffect(() => {
    setSubmitError("");
    setTimeskipOpen(false);
    failedSubmitRef.current = null;
    failedContinueRef.current = null;
    failedTimeskipRef.current = null;
  }, [activeSaveId]);
  useEffect(() => {
    if (disabled) setTimeskipOpen(false);
  }, [disabled]);

  const applyFormat = useCallback((actionId: ComposerFormatActionId, restoreFocus: "always" | "while-focused" = "always") => {
    const textarea = textareaRef.current;
    const sourceBody = textarea?.value ?? body;
    const selectionStart = textarea?.selectionStart ?? sourceBody.length;
    const selectionEnd = textarea?.selectionEnd ?? selectionStart;
    const next = formatComposerBody(sourceBody, selectionStart, selectionEnd, actionId);
    const activeElement = document.activeElement;
    setBody(next.body);
    window.requestAnimationFrame(() => {
      const current = textareaRef.current;
      if (!current) return;
      if (restoreFocus === "while-focused" && document.activeElement !== activeElement) return;
      current.focus();
      current.setSelectionRange(next.selectionStart, next.selectionEnd);
    });
  }, [body]);

  const submitBusy = submittingSaveId === activeSaveId || submittingSaveIdRef.current === activeSaveId;
  const timeskipBusy = timeskip.isPending || timeskipSubmittingRef.current;
  const continueBusy = continuingSaveId === activeSaveId || continuingSaveIdRef.current === activeSaveId;
  const composerMutationBusy = submitBusy || timeskipBusy || continueBusy;
  const timeskipDisabled = disabled || composerMutationBusy;
  return (
    <>
      <form
        className="composer"
        onSubmit={(event) => {
          event.preventDefault();
          if (!disabled && !composerMutationBusy && body.trim()) {
            const submittedBody = body;
            const submittedSaveId = activeSaveId;
            submittingSaveIdRef.current = submittedSaveId;
            setSubmittingSaveId(submittedSaveId);
            setSubmitError("");
            setBody("");
            const paintStartedAtMs = performance.now();
            onPendingMessage(pendingPlayerChronicleMessage(
              submittedBody,
              submittedSaveId,
              pendingAfterMessageId,
              paintStartedAtMs,
            ));
            submit.mutate(chatSubmitVariables(
              submittedBody,
              submittedSaveId,
              failedSubmitRef.current,
              paintStartedAtMs,
            ));
          }
        }}
      >
        <div className="composer-fields">
          <div className="composer-format-toolbar" role="toolbar" aria-label="Message formatting">
            {COMPOSER_FORMAT_ACTIONS.map((action) => {
              const Icon = action.icon;
              return (
                <button
                  key={action.actionId}
                  type="button"
                  className="composer-format-button"
                  title={action.title}
                  aria-label={action.label}
                  aria-keyshortcuts={action.shortcut}
                  onClick={() => applyFormat(action.actionId, "always")}
                >
                  <Icon size={16} aria-hidden="true" />
                </button>
              );
            })}
          </div>
          {storytellerMode ? (
            <div className="storyteller-quick-actions">
              <button
                type="button"
                className="storyteller-continue-button"
                disabled={disabled || composerMutationBusy}
                onClick={() => {
                  continuingSaveIdRef.current = activeSaveId;
                  setContinuingSaveId(activeSaveId);
                  setSubmitError("");
                  const previous = failedContinueRef.current;
                  continueStory.mutate(
                    previous?.saveId === activeSaveId
                      ? previous
                      : { saveId: activeSaveId, key: clientTurnId() }
                  );
                }}
              >
                {continueBusy ? (
                  <Loader2 className="spin" size={15} aria-hidden="true" />
                ) : (
                  <BookOpen size={15} aria-hidden="true" />
                )}
                {continueBusy ? "Continuing…" : "Continue story"}
              </button>
            </div>
          ) : null}
          <textarea
            ref={textareaRef}
            value={body}
            onChange={(event) => setBody(event.target.value)}
            onKeyDown={(event) => {
              const actionId = composerFormatActionForKey(event);
              if (actionId === null) return;
              event.preventDefault();
              applyFormat(actionId, "while-focused");
            }}
            placeholder={storytellerMode ? "Guide what happens next…" : "Say what you do..."}
            aria-label="Message"
          />
          {submitError ? <InlineNotice className="composer-error">{submitError}</InlineNotice> : null}
        </div>
        <button
          type="button"
          className="composer-action-button"
          disabled={timeskipDisabled}
          title="Timeskip"
          aria-label="Timeskip"
          onClick={() => setTimeskipOpen(true)}
        >
          <Clock size={18} />
        </button>
        <button className="composer-action-button" disabled={disabled || !body.trim() || composerMutationBusy} title="Send">
          <Send size={18} />
        </button>
      </form>
      {timeskipOpen && !disabled ? (
        <TimeskipDialog
          busy={timeskip.isPending || timeskipSubmittingRef.current}
          error={timeskip.error instanceof Error ? timeskip.error.message : ""}
          onCancel={() => {
            if (!timeskip.isPending && !timeskipSubmittingRef.current) setTimeskipOpen(false);
          }}
          onSubmit={(instruction) => {
            if (timeskipDisabled || !instruction.trim()) return;
            timeskipSubmittingRef.current = true;
            const previous = failedTimeskipRef.current;
            const submitted = (
              previous
              && previous.saveId === activeSaveId
              && previous.instruction.trim() === instruction.trim()
            ) ? { ...previous, instruction } : {
              instruction,
              saveId: activeSaveId,
              key: clientTurnId()
            };
            timeskip.mutate(submitted);
          }}
        />
      ) : null}
    </>
  );
}

function CyoaActionPicker({
  disabled,
  runJob,
  activeSaveId,
  actionChoices,
  generationActive = false,
  generationRecoveryPending = false,
  pendingAfterMessageId = null,
  onPendingMessage
}: {
  disabled: boolean;
  runJob: RunJob;
  activeSaveId: string | null;
  actionChoices: RuntimeModel["action_choices"];
  generationActive?: boolean;
  generationRecoveryPending?: boolean;
  pendingAfterMessageId?: string | null;
  onPendingMessage: (message: PendingChronicleMessage | null) => void;
}) {
  const [manualOpen, setManualOpen] = useState(false);
  const [manualBody, setManualBody] = useState("");
  const [submitError, setSubmitError] = useState("");
  const activeSaveIdRef = useRef(activeSaveId);
  activeSaveIdRef.current = activeSaveId;
  const submittingSaveIdRef = useRef<string | null | undefined>(undefined);
  const [submittingSaveId, setSubmittingSaveId] = useState<string | null | undefined>(undefined);
  const failedSubmitRef = useRef<ChatSubmitVariables | null>(null);
  const submit = useMutation({
    mutationFn: (submitted: ChatSubmitVariables) => postJson<Job>("/api/chat", { body: submitted.body, speaker_name: null, save_id: submitted.saveId, client_turn_id: submitted.key }),
    onSuccess: (job, submitted) => {
      failedSubmitRef.current = null;
      if (submitted.saveId !== activeSaveIdRef.current) return;
      setSubmitError("");
      setManualBody("");
      runJob(job, { paintStartedAtMs: submitted.paintStartedAtMs });
    },
    onError: (error, submitted) => {
      failedSubmitRef.current = submitted;
      if (submitted.saveId !== activeSaveIdRef.current) return;
      setManualBody((currentBody) => currentBody ? currentBody : submitted.body);
      onPendingMessage(null);
      setSubmitError(error instanceof Error ? error.message : "Could not send message");
    },
    onSettled: (_data, _error, submitted) => {
      if (submitted && submittingSaveIdRef.current === submitted.saveId) {
        submittingSaveIdRef.current = undefined;
        setSubmittingSaveId(undefined);
      }
    }
  });
  const regenerate = useMutation({
    mutationFn: (submitted: { narratorMessageId: string; saveId: string | null }) => postJson<Job>("/api/action-choices/regenerate", {
      message_id: submitted.narratorMessageId,
      save_id: submitted.saveId
    }),
    onSuccess: (job, submitted) => {
      if (submitted.saveId !== activeSaveIdRef.current) return;
      setSubmitError("");
      runJob(job);
    },
    onError: (error, submitted) => {
      if (submitted.saveId !== activeSaveIdRef.current) return;
      setSubmitError(error instanceof Error ? error.message : "Could not regenerate options");
    }
  });
  useEffect(() => {
    setManualOpen(false);
    setManualBody("");
    setSubmitError("");
    failedSubmitRef.current = null;
  }, [activeSaveId, actionChoices?.narrator_message_id]);

  const choices = [...(actionChoices?.choices ?? [])].sort((left, right) => left.ordinal - right.ordinal);
  const submitBusy = submittingSaveId === activeSaveId || submittingSaveIdRef.current === activeSaveId;
  const regenerateBusy = regenerate.isPending;
  const choicesGenerating = generationActive
    || generationRecoveryPending
    || Boolean(actionChoices?.generation_job)
    || regenerateBusy;
  const canSubmit = !disabled && !submitBusy;
  const canSubmitChoice = canSubmit && !choicesGenerating;
  const canRegenerate = !disabled && !submitBusy && !regenerateBusy && !choicesGenerating && Boolean(activeSaveId && actionChoices?.narrator_message_id);
  const submitBody = (body: string, allowed = canSubmit) => {
    const submittedBody = body.trim();
    if (!allowed || !submittedBody) return;
    const submittedSaveId = activeSaveId;
    submittingSaveIdRef.current = submittedSaveId;
    setSubmittingSaveId(submittedSaveId);
    setSubmitError("");
    const paintStartedAtMs = performance.now();
    onPendingMessage(pendingPlayerChronicleMessage(
      submittedBody,
      submittedSaveId,
      pendingAfterMessageId,
      paintStartedAtMs,
    ));
    submit.mutate(chatSubmitVariables(
      submittedBody,
      submittedSaveId,
      failedSubmitRef.current,
      paintStartedAtMs,
    ));
  };
  const regenerateOptions = () => {
    if (!canRegenerate || !actionChoices?.narrator_message_id) return;
    setSubmitError("");
    regenerate.mutate({
      narratorMessageId: actionChoices.narrator_message_id,
      saveId: activeSaveId
    });
  };

  return (
    <section className="cyoa-picker" aria-label="Choose your next action">
      <ol className="cyoa-choice-list" aria-label="Generated actions" role="list">
        {choices.map((choice, index) => (
          <li key={choice.choice_id} className="cyoa-choice-item" role="listitem">
            <button
              type="button"
              className="cyoa-choice-button"
              disabled={!canSubmitChoice}
              onClick={() => submitBody(choice.body, canSubmitChoice)}
            >
              <span className="cyoa-choice-ordinal" aria-hidden="true">{index + 1}</span>
              <span className="cyoa-choice-body">{choice.body}</span>
            </button>
          </li>
        ))}
      </ol>
      <div className="cyoa-actions">
        <div className={manualOpen ? "cyoa-custom cyoa-custom-open" : "cyoa-custom"}>
          <button
            type="button"
            className="cyoa-custom-toggle"
            disabled={disabled || submitBusy}
            aria-label="Write your own"
            aria-expanded={manualOpen}
            onClick={() => setManualOpen((current) => !current)}
          >
            <Edit3 size={17} aria-hidden="true" />
            <span>Write your own</span>
          </button>
          {choicesGenerating ? (
            <span className="cyoa-generation-status" role="status">Generating choices...</span>
          ) : null}
          {manualOpen ? (
            <form
              className="cyoa-manual-form"
              onSubmit={(event) => {
                event.preventDefault();
                submitBody(manualBody);
              }}
            >
              <textarea
                value={manualBody}
                onChange={(event) => setManualBody(event.target.value)}
                placeholder="Say what you do..."
                aria-label="Custom action"
              />
              <button className="composer-action-button" disabled={!canSubmit || !manualBody.trim()} title="Send">
                <Send size={18} aria-hidden="true" />
              </button>
            </form>
          ) : null}
        </div>
        {actionChoices?.narrator_message_id ? (
          <button
            type="button"
            className="cyoa-regenerate-button"
            disabled={!canRegenerate}
            title="Regenerate options"
            aria-label="Regenerate options"
            onClick={regenerateOptions}
          >
            <RefreshCw className={regenerateBusy ? "spin" : undefined} size={17} aria-hidden="true" />
            <span>Regenerate Options</span>
          </button>
        ) : null}
      </div>
      {actionChoices?.generation_error ? <InlineNotice className="composer-error">{actionChoices.generation_error}</InlineNotice> : null}
      {submitError ? <InlineNotice className="composer-error">{submitError}</InlineNotice> : null}
    </section>
  );
}

function TimeskipDialog({
  busy,
  error,
  onCancel,
  onSubmit
}: {
  busy: boolean;
  error: string;
  onCancel: () => void;
  onSubmit: (instruction: string) => void;
}) {
  const [instruction, setInstruction] = useState("");
  const titleId = React.useId();
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog timeskip-dialog"
        titleId={titleId}
        onClose={onCancel}
        onSubmit={(event) => {
          event.preventDefault();
          if (!busy && instruction.trim()) onSubmit(instruction.trim());
        }}
      >
        <div className="dialog-title-row">
          <h2 id={titleId}>Timeskip</h2>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close" disabled={busy}>
            <X size={16} />
          </button>
        </div>
        <label>
          Timeskip instructions
          <textarea
            className="tall-field"
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            placeholder="Skip to the next morning at the inn..."
            autoFocus
          />
        </label>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="modal-actions">
          <button type="button" onClick={onCancel} disabled={busy}>Cancel</button>
          <button type="submit" className="primary-command compact" disabled={busy || !instruction.trim()}>
            <Clock size={15} /> Timeskip
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function RightPanel({
  panel,
  model,
  runJob,
  currentUser = null,
  openLookAround,
  readOnly = false,
  onContentSafetyChanged
}: {
  panel: PanelName;
  model?: RuntimeModel;
  runJob: RunJob;
  currentUser?: CurrentUser | null;
  openLookAround?: (query: string) => void;
  readOnly?: boolean;
  onContentSafetyChanged?: () => void;
}) {
  const effectiveCurrentUser: CurrentUser | null = readOnly
    ? { id: "unsupported-save", username: "Recovery", role: "child", status: "active" }
    : currentUser;
  if (panel === "media") {
    return (
      <React.Suspense fallback={<RightPanelFallback title="Scene Media" icon={<Image size={18} />} />}>
        <LazyMediaPanel model={model} runJob={runJob} currentUser={effectiveCurrentUser} />
      </React.Suspense>
    );
  }
  if (panel === "history") {
    return (
      <React.Suspense fallback={<RightPanelFallback title="History" icon={<History size={18} />} />}>
        <LazyHistoryPanel activeSaveId={model?.active_save_id ?? null} />
      </React.Suspense>
    );
  }
  if (panel === "world") {
    return (
      <React.Suspense fallback={<RightPanelFallback title="World Data" icon={<Archive size={18} />} />}>
        <LazyWorldPanel model={model} runJob={runJob} currentUser={effectiveCurrentUser} openLookAround={readOnly ? undefined : openLookAround} />
      </React.Suspense>
    );
  }
  if (panel === "characters") {
    return (
      <React.Suspense fallback={<RightPanelFallback title="Characters" icon={<Users size={18} />} />}>
        <LazyCharactersPanel
          activeSaveId={model?.active_save_id ?? null}
          runJob={runJob}
          currentUser={effectiveCurrentUser}
          characterTextsEnabled={!readOnly && model?.character_texts_enabled === true}
        />
      </React.Suspense>
    );
  }
  return (
    <React.Suspense fallback={<RightPanelFallback title="Settings" icon={<Settings size={18} />} />}>
      <LazySettingsPanel
        runJob={runJob}
        activeSaveId={readOnly ? null : model?.active_save_id ?? null}
        storytellerMode={model?.interaction_mode === "storyteller"}
        currentUser={effectiveCurrentUser}
        onContentSafetyChanged={onContentSafetyChanged}
      />
    </React.Suspense>
  );
}

function RightPanelFallback({ title, icon }: { title: string; icon: React.ReactNode }) {
  return (
    <aside className="right-panel" aria-busy="true">
      <PanelHeader icon={icon} title={title} />
      <p className="muted">Loading...</p>
    </aside>
  );
}

function initialMediaDraftLabel(_scenarioType: string) {
  return "Generate opening image";
}

function SaveBundleControls({
  hasActiveSave,
  exportEnabled,
  activeSaveId,
  onImported,
  runJob,
  saveExports
}: {
  hasActiveSave: boolean;
  exportEnabled: boolean;
  activeSaveId: string | null;
  onImported: (saveId: string | null) => void;
  runJob?: RunJob;
  saveExports: SaveExports;
}) {
  const [saveExportStates, setSaveExportStates, clearSaveExportRecovery] = saveExports;
  const exportState = activeSaveId ? saveExportStates[activeSaveId] : undefined;
  const exporting = exportEnabled && exportState === "pending";
  const exportDownload = exportEnabled && isChatBundleExportResult(exportState) ? exportState : null;
  const exportError = exportEnabled && typeof exportState === "string" && exportState !== "pending"
    ? exportState
    : "";
  const storyLogPath = activeSaveId
    ? `/api/story-logs/export?save_id=${encodeURIComponent(activeSaveId)}`
    : "/api/story-logs/export";
  const startExport = async () => {
    if (!runJob || !hasActiveSave || !exportEnabled || !activeSaveId || exporting) return;
    clearSaveExportRecovery?.(activeSaveId, "prepare");
    setSaveExportState(setSaveExportStates, activeSaveId, "pending");
    try {
      const created = await postJson<Job>("/api/bundles/export", {
        save_id: activeSaveId,
        include_revision_history: false
      });
      clearSaveExportRecovery?.(activeSaveId, "restart");
      runJob(created, {
        allowInactiveSave: true,
        allowCrossSaveCompletion: true,
        onSucceeded: (result) => {
          if (!isChatBundleExportResult(result)) {
            setSaveExportState(setSaveExportStates, activeSaveId, "Save export completed without a download.");
            return;
          }
          setSaveExportState(setSaveExportStates, activeSaveId, result);
        },
        onFailed: (error) => {
          setSaveExportState(setSaveExportStates, activeSaveId, error);
        },
        onFinished: (job) => {
          if (job.status === "cancelled") {
            setSaveExportState(setSaveExportStates, activeSaveId, "Save export cancelled.");
          }
        }
      });
    } catch (failure) {
      setSaveExportState(
        setSaveExportStates,
        activeSaveId,
        failure instanceof Error ? failure.message : "Could not start save export"
      );
    }
  };
  return (
    <div className="command-row save-bundle-actions">
      {exportDownload ? (
        <a
          className="download-link"
          href={exportDownload.download_url}
          download={exportDownload.filename}
          title="Download completed save export"
          aria-label="Download completed save export"
          onClick={() => {
            if (activeSaveId) clearSaveExportRecovery?.(activeSaveId, "consume");
            window.setTimeout(() => {
              if (activeSaveId) setSaveExportState(setSaveExportStates, activeSaveId);
            }, 0);
          }}
        >
          <Download size={14} /> Download
        </a>
      ) : (
        <button
          title="Export active save bundle"
          aria-label="Export active save"
          disabled={!hasActiveSave || !exportEnabled || !runJob || exporting}
          onClick={() => void startExport()}
        >
          <Download size={14} /> {exporting ? "Exporting..." : "Export"}
        </button>
      )}
      <button
        title="Export active save story log"
        aria-label="Export story log"
        disabled={!hasActiveSave}
        onClick={() => openDownloadInNewTab(storyLogPath)}
      >
        <FileText size={14} /> Story log
      </button>
      <SaveBundleUpload onImported={onImported} />
      {exportDownload ? <InlineNotice className="save-export-ready" polite>Save export ready.</InlineNotice> : null}
      {exportError ? <InlineNotice>{exportError}</InlineNotice> : null}
    </div>
  );
}

function SaveBundleUpload({ onImported }: { onImported: (saveId: string | null) => void }) {
  const [pending, setPending] = useState<{ preview_id: string; preview: BundlePreview } | null>(null);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  return (
    <>
      <button
        type="button"
        className="upload-button"
        aria-label="Import save bundle"
        onClick={() => inputRef.current?.click()}
      >
        <Upload size={15} /> Import
      </button>
      <input
        ref={inputRef}
        className="upload-input"
        aria-label="Save bundle file"
        type="file"
        accept=".bragi-chat"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (!file) return;
          const form = new FormData();
          form.append("file", file);
          try {
            setPending(await api<{ preview_id: string; preview: BundlePreview }>("/api/bundles/preview", { method: "POST", body: form }));
            setError("");
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : "Import preview failed");
          }
        }}
      />
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      {pending ? (
        <PreviewModal
          title="Import save bundle?"
          preview={pending.preview}
          detail="This will restore the bundled save into Bragi Web."
          confirmLabel="Import"
          onCancel={() => setPending(null)}
          onConfirm={async () => {
            const result = await postJson<RuntimeModel>(`/api/bundles/import/${pending.preview_id}`, {});
            setPending(null);
            onImported(result.active_save_id ?? null);
          }}
        />
      ) : null}
    </>
  );
}


function PanelHeader({ icon, title }: { icon: React.ReactNode; title: string }) {
  return <header className="panel-header"><span className="panel-header-icon">{icon}</span><h2>{title}</h2><BrandMark /></header>;
}

function scenarioEditorValue(scenario: WorldDataScenario | Record<string, unknown>): ScenarioEditorValue {
  const record = scenario as Record<string, unknown>;
  return {
    scenario_id: textValue(record.scenario_id),
    scenario_type: textValue(record.scenario_type) || "full_roleplay",
    interaction_mode: record.interaction_mode === "storyteller" ? "storyteller" : "roleplay",
    title: textValue(record.title),
    premise: textValue(record.premise),
    player_character_name: textValue(record.player_character_name),
    player_role: textValue(record.player_role),
    content_sections: scenarioEditorSections(record.content_sections),
    character_starters: scenarioEditorStarters(record.character_starters)
  };
}

function scenarioEditorSections(value: unknown): ScenarioEditorSection[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item, index) => {
    if (!Array.isArray(item) || item.length !== 2) return [];
    const key = textValue(item[0]).trim();
    if (!key || SCENARIO_CORE_SECTION_IDS.has(key)) return [];
    return [{ id: `${key}:${index}`, key, value: textValue(item[1]) }];
  });
}

function scenarioEditorStarters(value: unknown): ScenarioEditorStarter[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((item, index) => {
    if (!item || typeof item !== "object") return [];
    const record = item as Record<string, unknown>;
    const name = textValue(record.name).trim();
    if (!name) return [];
    return [{
      id: `starter:${index}:${name}`,
      starter_id: textValue(record.starter_id),
      name,
      aliases_text: stringListText(record.aliases),
      role: textValue(record.role),
      age: textValue(record.age),
      known_state: textValue(record.known_state),
      appearance: textValue(record.appearance),
      visual_notes: textValue(record.visual_notes),
      personality: textValue(record.personality),
      voice: textValue(record.voice),
      texting_style: textValue(record.texting_style),
      goals: textValue(record.goals),
      motivations: textValue(record.motivations),
      boundaries: textValue(record.boundaries),
      relationships_json: JSON.stringify(scenarioObjectValue(record.relationships), null, 2),
      status: textValue(record.status),
      met: typeof record.met === "boolean" ? record.met : true,
      locked_fields_text: stringListText(record.locked_fields),
      reference_image: scenarioStarterReferenceImage(record.reference_image)
    }];
  });
}

function scenarioSectionEditorGroups(scenarioType: string): ScenarioSectionGroup[] {
  if (scenarioType === "fantasy_roleplay") return FANTASY_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "science_fiction_roleplay") return SCIENCE_FICTION_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "first_contact_exploration") return FIRST_CONTACT_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "survival_expedition") return SURVIVAL_EXPEDITION_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "time_loop") return TIME_LOOP_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "investigation_mystery") return INVESTIGATION_MYSTERY_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "heist_infiltration") return HEIST_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "political_intrigue") return POLITICAL_INTRIGUE_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "settlement_builder") return SETTLEMENT_BUILDER_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "monster_hunt_bounty") return MONSTER_HUNT_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "road_trip_pilgrimage") return ROAD_TRIP_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "merchant_trade_route") return MERCHANT_TRADE_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "dating_sim") return DATING_SIM_SCENARIO_SECTION_GROUPS;
  if (scenarioType === "choose_your_own_adventure") return CYOA_SCENARIO_SECTION_GROUPS;
  return FULL_ROLEPLAY_SCENARIO_SECTION_GROUPS;
}

function scenarioEditPayload(scenario: ScenarioEditorValue): { edit: ScenarioEditPayload } | { error: string } {
  const title = scenario.title.trim();
  const premise = scenario.premise.trim();
  const playerRole = scenario.player_role.trim();
  if (!title) return { error: "Title is required" };
  if (!premise) return { error: "Premise is required" };
  if (scenario.interaction_mode === "roleplay" && !playerRole) {
    return { error: "Player role is required" };
  }
  const contentSections: [string, string][] = [];
  const seen = new Set<string>();
  for (const section of scenario.content_sections) {
    const key = section.key.trim();
    if (!key) return { error: "Section key is required" };
    if (SCENARIO_CORE_SECTION_IDS.has(key)) return { error: "Section keys cannot use core field names" };
    if (seen.has(key)) return { error: "Section keys must be unique" };
    seen.add(key);
    contentSections.push([key, section.value]);
  }
  const starterPayload = scenarioStarterPayload(scenario.character_starters);
  if ("error" in starterPayload) return starterPayload;
  return {
    edit: {
      interaction_mode: scenario.interaction_mode,
      title,
      premise,
      player_character_name: scenario.player_character_name.trim(),
      player_role: playerRole,
      content_sections: contentSections,
      character_starters: starterPayload.value
    }
  };
}

function scenarioStarterPayload(
  starters: ScenarioEditorStarter[]
): { value: ScenarioEditPayload["character_starters"] } | { error: string } {
  const characterStarters: ScenarioEditPayload["character_starters"] = [];
  const seenStarterNames = new Set<string>();
  for (const starter of starters) {
    const name = starter.name.trim();
    if (!name) return { error: "Starter name is required" };
    const key = name.toLocaleLowerCase();
    if (seenStarterNames.has(key)) return { error: "Starter names must be unique" };
    seenStarterNames.add(key);
    const relationships = parseObjectJson(starter.relationships_json, `Relationships for ${name}`);
    if ("error" in relationships) return relationships;
    characterStarters.push({
      ...(starter.starter_id.trim() ? { starter_id: starter.starter_id.trim() } : {}),
      name,
      aliases: csvValues(starter.aliases_text),
      role: starter.role.trim(),
      age: starter.age.trim(),
      known_state: starter.known_state.trim(),
      appearance: starter.appearance.trim(),
      visual_notes: starter.visual_notes.trim(),
      personality: starter.personality.trim(),
      voice: starter.voice.trim(),
      texting_style: starter.texting_style.trim(),
      goals: starter.goals.trim(),
      motivations: starter.motivations.trim(),
      boundaries: starter.boundaries.trim(),
      relationships: relationships.value,
      status: starter.status.trim(),
      met: starter.met,
      locked_fields: csvValues(starter.locked_fields_text),
      reference_image: starter.reference_image
    });
  }
  return { value: characterStarters };
}

function textValue(value: unknown): string {
  if (value === null || value === undefined) return "";
  return String(value);
}

function stringListText(value: unknown): string {
  if (!Array.isArray(value)) return "";
  return value.filter((item): item is string => typeof item === "string" && item.trim().length > 0).join(", ");
}

function scenarioObjectValue(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}

function scenarioStarterReferenceImage(value: unknown): ScenarioStarterReferenceImage | null {
  if (!value || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const id = textValue(record.id).trim();
  if (!id) return null;
  return {
    id,
    path: textValue(record.path),
    thumbnail_path: typeof record.thumbnail_path === "string" || record.thumbnail_path === null ? record.thumbnail_path : null,
    mime_type: textValue(record.mime_type) || "image/png",
    prompt_preview: textValue(record.prompt_preview) || "Uploaded character reference image",
    source: typeof record.source === "string" || record.source === null ? record.source : null,
    created_at: typeof record.created_at === "string" || record.created_at === null ? record.created_at : null,
    bundle_path: typeof record.bundle_path === "string" || record.bundle_path === null ? record.bundle_path : null
  };
}

function parseObjectJson(value: string, label: string): { value: Record<string, unknown> } | { error: string } {
  const text = value.trim();
  if (!text) return { value: {} };
  try {
    const parsed = JSON.parse(text);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return { error: `${label} must be a JSON object` };
    return { value: parsed as Record<string, unknown> };
  } catch {
    return { error: `${label} must be valid JSON` };
  }
}

function csvValues(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function stableEditorSnapshot(value: unknown): string {
  return JSON.stringify(stableSnapshotValue(value));
}

function stableSnapshotValue(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableSnapshotValue);
  if (value && typeof value === "object") {
    return Object.keys(value as Record<string, unknown>).sort().reduce<Record<string, unknown>>((result, key) => {
      const item = (value as Record<string, unknown>)[key];
      if (item !== undefined) result[key] = stableSnapshotValue(item);
      return result;
    }, {});
  }
  return value;
}

function scenarioEditorDraftState(scenario: ScenarioEditorValue): ScenarioEditorDraftState {
  return {
    core: {
      interaction_mode: scenario.interaction_mode,
      title: scenario.title,
      premise: scenario.premise,
      player_character_name: scenario.player_character_name,
      player_role: scenario.player_role
    },
    sections: scenario.content_sections.map((section) => ({ ...section })),
    starters: scenario.character_starters.map((starter) => ({ ...starter }))
  };
}

function scenarioEditorDraftSnapshot(draft: ScenarioEditorDraftState): string {
  return stableEditorSnapshot({
    core: draft.core,
    sections: draft.sections.map((section) => ({ key: section.key, value: section.value })),
    starters: draft.starters.map(({ id: _id, ...starter }) => starter)
  });
}

function scenarioStarterReferencePatch(
  updatedStarters: ScenarioEditorStarter[],
  targetStarter: ScenarioEditorStarter
): ScenarioStarterReferencePatch | null {
  const targetStarterId = targetStarter.starter_id.trim();
  if (targetStarterId) {
    const starter = updatedStarters.find(
      (candidate) => candidate.starter_id.trim() === targetStarterId
    );
    if (starter) return scenarioStarterReferencePatchFor(starter);
  }

  const targetIndex = scenarioEditorStarterIndex(targetStarter.id);
  if (targetIndex !== null) {
    const starter = updatedStarters[targetIndex];
    if (starter) return scenarioStarterReferencePatchFor(starter);
  }

  const targetName = targetStarter.name.trim();
  if (!targetName) return null;
  const matches = updatedStarters.filter(
    (starter) => starter.name.trim() === targetName
  );
  if (matches.length !== 1) return null;
  return scenarioStarterReferencePatchFor(matches[0]);
}

function scenarioEditorStarterIndex(id: string): number | null {
  const match = /^starter:(\d+):/.exec(id);
  if (!match) return null;
  const index = Number(match[1]);
  return Number.isInteger(index) ? index : null;
}

function scenarioStarterReferencePatchFor(
  starter: ScenarioEditorStarter
): ScenarioStarterReferencePatch {
  const { starter_id, appearance, visual_notes, locked_fields_text, reference_image } = starter;
  return {
    starter_id, appearance, visual_notes, locked_fields_text, reference_image
  };
}

function scenarioEditorValueFromDraft(scenario: ScenarioEditorValue, draft: ScenarioEditorDraftState): ScenarioEditorValue {
  return {
    ...scenario,
    ...draft.core,
    content_sections: draft.sections,
    character_starters: draft.starters
  };
}

function ScenarioStarterReferenceField({
  scenarioId,
  starter,
  disabled,
  onError,
  onUpdated,
  onPendingChange
}: {
  scenarioId: string;
  starter: ScenarioEditorStarter;
  disabled: boolean;
  onError: (message: string) => void;
  onUpdated: (model: WorldDataModel, starter: ScenarioEditorStarter) => void;
  onPendingChange: (pending: boolean) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [busy, setBusy] = useState<"upload" | "remove" | null>(null);
  const reference = starter.reference_image;
  const hasReference = Boolean(reference);
  const canUse = Boolean(scenarioId && !disabled && busy === null);
  useEffect(() => {
    if (busy === null) return undefined;
    onPendingChange(true);
    return () => onPendingChange(false);
  }, [busy, onPendingChange]);
  const uploadFile = async (file: File) => {
    if (!scenarioId) return;
    setBusy("upload");
    onError("");
    const form = new FormData();
    if (starter.starter_id.trim()) form.append("starter_id", starter.starter_id.trim());
    else form.append("starter_name", starter.name.trim());
    form.append("replace_existing", hasReference ? "true" : "false");
    form.append("file", file);
    try {
      onUpdated(await api<WorldDataModel>(
        `/api/scenarios/${encodeURIComponent(scenarioId)}/character-starters/reference-image/upload`,
        { method: "POST", body: form }
      ), starter);
    } catch (failure) {
      onError(failure instanceof Error ? failure.message : "Could not upload reference image");
    } finally {
      setBusy(null);
    }
  };
  const removeImage = async () => {
    if (!scenarioId) return;
    setBusy("remove");
    onError("");
    try {
      onUpdated(await postJson<WorldDataModel>(
        `/api/scenarios/${encodeURIComponent(scenarioId)}/character-starters/reference-image/remove`,
        {
          starter_id: starter.starter_id.trim() || null,
          starter_name: starter.name.trim()
        }
      ), starter);
    } catch (failure) {
      onError(failure instanceof Error ? failure.message : "Could not remove reference image");
    } finally {
      setBusy(null);
    }
  };
  return (
    <div className="scenario-starter-reference">
      <div className="scenario-starter-reference-preview">
        {reference && scenarioId ? (
          <img
            src={scenarioStarterReferenceThumbnailPath(scenarioId, reference.id)}
            alt={reference.prompt_preview || `${starter.name} reference image`}
            loading="lazy"
            decoding="async"
          />
        ) : (
          <span><Image size={18} /></span>
        )}
      </div>
      <div className="scenario-starter-reference-tools">
        <button
          type="button"
          disabled={!canUse}
          aria-label={`${hasReference ? "Replace" : "Upload"} ${starter.name || "starter"} reference image`}
          onClick={() => inputRef.current?.click()}
        >
          {busy === "upload" ? <Loader2 className="spin" size={14} /> : <Upload size={14} />}
        </button>
        {hasReference ? (
          <button
            type="button"
            className="danger-command"
            disabled={!canUse}
            aria-label={`Remove ${starter.name || "starter"} reference image`}
            onClick={() => void removeImage()}
          >
            {busy === "remove" ? <Loader2 className="spin" size={14} /> : <X size={14} />}
          </button>
        ) : null}
        <input
          ref={inputRef}
          className="upload-input"
          aria-label={`${hasReference ? "Replace" : "Upload"} ${starter.name || "starter"} reference image file`}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(event) => {
            const file = event.target.files?.[0];
            event.target.value = "";
            if (file) void uploadFile(file);
          }}
        />
      </div>
    </div>
  );
}

function ScenarioStructuredEditor({
  scenario,
  onSave,
  onCancel,
  saveLabel = "Save scenario",
  onDirtyChange,
  starterReferenceImages = false,
  interactionModeEditable = false
}: {
  scenario: ScenarioEditorValue;
  onSave: (edit: ScenarioEditPayload) => Promise<void>;
  onCancel?: () => void;
  saveLabel?: string;
  onDirtyChange?: (dirty: boolean) => void;
  starterReferenceImages?: boolean;
  interactionModeEditable?: boolean;
}) {
  const initialDraft = scenarioEditorDraftState(scenario);
  const incomingSnapshot = scenarioEditorDraftSnapshot(initialDraft);
  const incomingSnapshotRef = useRef(incomingSnapshot);
  const [core, setCore] = useState<ScenarioEditorCore>(() => initialDraft.core);
  const [sections, setSections] = useState<ScenarioEditorSection[]>(() => initialDraft.sections);
  const [starters, setStarters] = useState<ScenarioEditorStarter[]>(() => initialDraft.starters);
  const [savedDraft, setSavedDraft] = useState<ScenarioEditorDraftState>(() => initialDraft);
  const [newSectionKey, setNewSectionKey] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [starterImagePendingCount, setStarterImagePendingCount] = useState(0);
  const [discardAction, setDiscardAction] = useState<"reset" | "close" | null>(null);
  const nextSectionId = useRef(sections.length + 1);
  const nextStarterId = useRef(starters.length + 1);
  const applyDraft = useCallback((draft: ScenarioEditorDraftState) => {
    setCore(draft.core);
    setSections(draft.sections.map((section) => ({ ...section })));
    setStarters(draft.starters.map((starter) => ({ ...starter })));
    nextSectionId.current = draft.sections.length + 1;
    nextStarterId.current = draft.starters.length + 1;
  }, []);
  useEffect(() => {
    if (incomingSnapshotRef.current === incomingSnapshot) return;
    incomingSnapshotRef.current = incomingSnapshot;
    applyDraft(initialDraft);
    setSavedDraft(initialDraft);
    setDiscardAction(null);
    setError("");
  }, [applyDraft, incomingSnapshot, initialDraft]);
  const groups = scenarioSectionEditorGroups(scenario.scenario_type);
  const knownSectionIds = new Set(groups.flatMap((group) => group.section_ids));
  const sectionByKey = (sectionId: string) => sections.find((section) => section.key === sectionId);
  const currentDraft = { core, sections, starters };
  const currentSnapshot = scenarioEditorDraftSnapshot(currentDraft);
  const savedSnapshot = scenarioEditorDraftSnapshot(savedDraft);
  const hasUnsavedChanges = currentSnapshot !== savedSnapshot;
  const hasPendingStarterImageOperation = starterImagePendingCount > 0;
  const requestCancel = () => {
    if (!onCancel) return;
    if (hasUnsavedChanges) {
      setDiscardAction("close");
      return;
    }
    onCancel();
  };
  useEffect(() => {
    onDirtyChange?.(hasUnsavedChanges);
  }, [hasUnsavedChanges, onDirtyChange]);
  const updateCore = (
    key: keyof typeof core,
    value: string
  ) => setCore((current) => ({ ...current, [key]: value }));
  const updateSectionValue = (sectionId: string, value: string) => {
    setSections((current) => current.map((section) => section.key === sectionId ? { ...section, value } : section));
  };
  const updateSectionKey = (id: string, key: string) => {
    setSections((current) => current.map((section) => section.id === id ? { ...section, key } : section));
  };
  const removeSection = (id: string) => setSections((current) => current.filter((section) => section.id !== id));
  const updateStarter = (id: string, patch: Partial<ScenarioEditorStarter>) => {
    setStarters((current) => current.map((starter) => starter.id === id ? { ...starter, ...patch } : starter));
  };
  const updateStarterImagePending = useCallback((pending: boolean) => {
    setStarterImagePendingCount((count) => Math.max(0, count + (pending ? 1 : -1)));
  }, []);
  const removeStarter = (id: string) => {
    setStarters((current) => current.filter((starter) => starter.id !== id));
  };
  const addStarter = () => {
    setStarters((current) => [
      ...current,
      {
        id: `new-starter:${nextStarterId.current++}`,
        starter_id: "",
        name: "",
        aliases_text: "",
        role: "",
        age: "",
        known_state: "",
        appearance: "",
        visual_notes: "",
        personality: "",
        voice: "",
        texting_style: "",
        goals: "",
        motivations: "",
        boundaries: "",
        relationships_json: "{}",
        status: "",
        met: true,
        locked_fields_text: "",
        reference_image: null
      }
    ]);
    setError("");
  };
  const moveSection = (id: string, offset: -1 | 1) => {
    setSections((current) => {
      const index = current.findIndex((section) => section.id === id);
      const nextIndex = index + offset;
      if (index < 0 || nextIndex < 0 || nextIndex >= current.length) return current;
      const next = [...current];
      const [item] = next.splice(index, 1);
      next.splice(nextIndex, 0, item);
      return next;
    });
  };
  const addSection = () => {
    const key = newSectionKey.trim();
    if (!key) {
      setError("Section key is required");
      return;
    }
    if (sections.some((section) => section.key.trim() === key)) {
      setError("Section keys must be unique");
      return;
    }
    setSections((current) => [...current, { id: `new:${nextSectionId.current++}`, key, value: "" }]);
    setNewSectionKey("");
    setError("");
  };
  const hiddenSectionIds = core.interaction_mode === "storyteller"
    ? new Set(["player_character_profile", "choice_style"])
    : new Set<string>();
  const visibleGroups = groups
    .map((group) => ({
      ...group,
      section_ids: group.section_ids.filter(
        (sectionId) => sectionByKey(sectionId) && !hiddenSectionIds.has(sectionId)
      )
    }))
    .filter((group) => group.section_ids.length > 0);
  const customSections = sections.filter(
    (section) => (
      !knownSectionIds.has(section.key)
      && !hiddenSectionIds.has(section.key)
    )
  );
  const submit = async () => {
    if (hasPendingStarterImageOperation) {
      setError("Wait for reference image updates to finish before saving.");
      return;
    }
    const draft = { core, sections, starters };
    const payload = scenarioEditPayload(scenarioEditorValueFromDraft(scenario, draft));
    if ("error" in payload) {
      setError(payload.error);
      return;
    }
    try {
      setSaving(true);
      await onSave(payload.edit);
      setSavedDraft(draft);
      setError("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not save scenario");
    } finally {
      setSaving(false);
    }
  };
  const applyScenarioModel = (
    model: WorldDataModel,
    targetStarter: ScenarioEditorStarter
  ) => {
    if (!model.scenario) return;
    const updatedScenario = scenarioEditorValue(model.scenario);
    const patch = scenarioStarterReferencePatch(
      updatedScenario.character_starters,
      targetStarter
    );
    if (!patch) return;
    setStarters((current) => current.map((starter) => (
      starter.id === targetStarter.id
        ? {
            ...starter,
            ...patch,
            locked_fields_text: mergeReferenceUploadLocks(
              csvValues(patch.locked_fields_text),
              csvValues(targetStarter.locked_fields_text),
              csvValues(starter.locked_fields_text)
            ).join(", ")
          }
        : starter
    )));
    setSavedDraft((current) => ({
      ...current,
      starters: current.starters.map((starter) => (
        starter.id === targetStarter.id
          ? { ...starter, ...patch }
          : starter
      ))
    }));
    setError("");
  };
  const canManageStarterImages = Boolean(
    starterReferenceImages && scenario.scenario_id && !hasUnsavedChanges && !saving
  );

  return (
    <div className="scenario-structured-editor">
      <EditorDirtyStatus
        dirty={hasUnsavedChanges}
        canDiscard={hasUnsavedChanges}
        onDiscard={() => setDiscardAction("reset")}
      />
      <div className="scenario-core-grid">
        <label className="field-label">
          <span>Interaction Mode</span>
          <select
            aria-label="Interaction Mode"
            disabled={!interactionModeEditable}
            value={core.interaction_mode}
            onChange={(event) => updateCore("interaction_mode", event.target.value)}
          >
            <option value="roleplay">Roleplay</option>
            <option value="storyteller">Storyteller</option>
          </select>
        </label>
        <label className="field-label">
          <span>Title</span>
          <input required value={core.title} onChange={(event) => updateCore("title", event.target.value)} />
        </label>
        {core.interaction_mode === "roleplay" ? <label className="field-label">
          <span>Player Character</span>
          <input value={core.player_character_name} onChange={(event) => updateCore("player_character_name", event.target.value)} />
        </label> : null}
        {core.interaction_mode === "roleplay" ? <label className="field-label">
          <span>Player Role</span>
          <input required value={core.player_role} onChange={(event) => updateCore("player_role", event.target.value)} />
        </label> : null}
        <label className="field-label scenario-premise-field">
          <span>Premise / Setup</span>
          <textarea required value={core.premise} onChange={(event) => updateCore("premise", event.target.value)} />
        </label>
      </div>
      {visibleGroups.map((group) => (
        <details className="model-group scenario-section-group" key={group.label} open>
          <summary>
            <div>
              <strong>{group.label}</strong>
              <span>{group.section_ids.length} sections</span>
            </div>
          </summary>
          <div className="scenario-section-list">
            {group.section_ids.map((sectionId) => (
              <label className="field-label" key={sectionId}>
                <span>{labelize(sectionId)}</span>
                <textarea value={sectionByKey(sectionId)?.value ?? ""} onChange={(event) => updateSectionValue(sectionId, event.target.value)} />
              </label>
            ))}
          </div>
        </details>
      ))}
      <details className="model-group scenario-section-group" open>
        <summary>
          <div>
            <strong>Character starters</strong>
            <span>{starters.length} characters</span>
          </div>
        </summary>
        <div className="scenario-starter-list">
          {starters.map((starter, index) => (
            <div className="scenario-starter-row" key={starter.id}>
              <div className="scenario-starter-grid">
                <label className="field-label">
                  <span>Name</span>
                  <input
                    aria-label={`Starter ${index + 1} name`}
                    value={starter.name}
                    onChange={(event) => updateStarter(starter.id, { name: event.target.value })}
                  />
                </label>
                {STARTER_INPUT_FIELDS.map(([field, label, suffix]) => (
                  <label className="field-label" key={field}>
                    <span>{label}</span>
                    <input
                      aria-label={`Starter ${starter.name || index + 1} ${suffix}`}
                      value={starter[field]}
                      onChange={(event) => updateStarter(
                        starter.id,
                        { [field]: event.target.value } as Partial<ScenarioEditorStarter>
                      )}
                    />
                  </label>
                ))}
                <label className="toggle-row compact-toggle scenario-starter-met">
                  <input
                    type="checkbox"
                    checked={starter.met}
                    onChange={(event) => updateStarter(starter.id, { met: event.target.checked })}
                  />
                  <span>Met</span>
                </label>
                {starterReferenceImages ? (
                  <ScenarioStarterReferenceField
                    scenarioId={scenario.scenario_id ?? ""}
                    starter={starter}
                    disabled={!canManageStarterImages}
	                    onError={setError}
	                    onUpdated={applyScenarioModel}
	                    onPendingChange={updateStarterImagePending}
	                  />
                ) : null}
                {STARTER_TEXTAREA_FIELDS.map(([field, label, suffix]) => (
                  <label className="field-label scenario-starter-wide" key={field}>
                    <span>{label}</span>
                    <textarea
                      className={field === "relationships_json" ? "json-editor compact-json-editor" : undefined}
                      aria-label={`Starter ${starter.name || index + 1} ${suffix}`}
                      value={starter[field]}
                      onChange={(event) => updateStarter(
                        starter.id,
                        { [field]: event.target.value } as Partial<ScenarioEditorStarter>
                      )}
                    />
                  </label>
                ))}
              </div>
              <div className="scenario-starter-tools">
                <button
                  type="button"
                  className={touchActionClassName("destructive-action")}
                  title="Remove"
                  aria-label={`Remove ${starter.name || "starter"}`}
                  onClick={() => removeStarter(starter.id)}
                >
                  <TouchActionContents icon={<Trash2 size={14} />} label="Remove" />
                </button>
              </div>
            </div>
          ))}
          {!starters.length ? <p className="empty">No character starters</p> : null}
          <button type="button" className="secondary-command scenario-add-starter" onClick={addStarter}>
            <Plus size={15} /> Add starter
          </button>
        </div>
      </details>
      <details className="model-group scenario-section-group" open>
        <summary>
          <div>
            <strong>Custom sections</strong>
            <span>{customSections.length} sections</span>
          </div>
        </summary>
        <div className="scenario-custom-list">
          {customSections.map((section) => (
            <div className="scenario-custom-section" key={section.id}>
              <label className="field-label">
                <span>Section Key</span>
                <input aria-label={`Section key ${section.key || "new section"}`} value={section.key} onChange={(event) => updateSectionKey(section.id, event.target.value)} />
              </label>
              <label className="field-label scenario-custom-body">
                <span>Section Body</span>
                <textarea aria-label={`Section body ${section.key || "new section"}`} value={section.value} onChange={(event) => setSections((current) => current.map((item) => item.id === section.id ? { ...item, value: event.target.value } : item))} />
              </label>
              <div className="scenario-custom-tools">
                <button type="button" className={touchActionClassName()} title="Move up" aria-label={`Move ${section.key || "section"} up`} onClick={() => moveSection(section.id, -1)}>
                  <TouchActionContents icon={<ArrowUp size={14} />} label="Up" />
                </button>
                <button type="button" className={touchActionClassName()} title="Move down" aria-label={`Move ${section.key || "section"} down`} onClick={() => moveSection(section.id, 1)}>
                  <TouchActionContents icon={<ArrowDown size={14} />} label="Down" />
                </button>
                <button type="button" className={touchActionClassName("destructive-action")} title="Remove" aria-label={`Remove ${section.key || "section"}`} onClick={() => removeSection(section.id)}>
                  <TouchActionContents icon={<Trash2 size={14} />} label="Remove" />
                </button>
              </div>
            </div>
          ))}
          {!customSections.length ? <p className="empty">No custom sections</p> : null}
          <div className="scenario-add-section">
            <label className="field-label">
              <span>New section key</span>
              <input value={newSectionKey} onChange={(event) => setNewSectionKey(event.target.value)} />
            </label>
            <button type="button" className="secondary-command" onClick={addSection}>
              <Plus size={15} /> Add section
            </button>
          </div>
        </div>
      </details>
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      <div className="command-row end scenario-editor-actions">
        {onCancel ? <button type="button" onClick={requestCancel}>Cancel</button> : null}
        <button type="button" className="primary-command compact" disabled={saving || hasPendingStarterImageOperation || !hasUnsavedChanges} onClick={submit}>
          {saving ? <Loader2 className="spin" size={15} /> : <Save size={15} />} {saveLabel}
        </button>
      </div>
      {discardAction ? (
        <ConfirmModal
          title="Discard changes?"
          body="Unsaved scenario edits will be lost."
          confirmLabel="Discard"
          destructive
          onCancel={() => setDiscardAction(null)}
          onConfirm={async () => {
            if (discardAction === "close") {
              setDiscardAction(null);
              onCancel?.();
              return;
            }
            applyDraft(savedDraft);
            setNewSectionKey("");
            setError("");
            setDiscardAction(null);
          }}
        />
      ) : null}
    </div>
  );
}

function ScenarioDefinitionModal({ scenario, onClose, onSaved }: { scenario: Scenario; onClose: () => void; onSaved: () => void }) {
  const definition = useQuery({ queryKey: ["scenario-definition", scenario.scenario_id], queryFn: () => api<WorldDataModel>(`/api/scenarios/${scenario.scenario_id}/definition`) });
  const worlds = useQuery({ queryKey: ["persistent-worlds"], queryFn: () => api<{ worlds: PersistentWorld[] }>("/api/worlds") });
  const editorScenario = definition.data?.scenario ? scenarioEditorValue(definition.data.scenario) : null;
  const [selectedWorldId, setSelectedWorldId] = useState("");
  const [linkingWorld, setLinkingWorld] = useState(false);
  const [linkError, setLinkError] = useState("");
  const [editorDirty, setEditorDirty] = useState(false);
  const [discardCloseOpen, setDiscardCloseOpen] = useState(false);
  const titleId = "scenario-definition-title";
  useEffect(() => {
    setSelectedWorldId(definition.data?.persistent_world?.world_id ?? "");
  }, [definition.data?.persistent_world?.world_id]);
  const requestClose = () => {
    if (editorDirty) {
      setDiscardCloseOpen(true);
      return;
    }
    onClose();
  };
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog wide-dialog scenario-definition-dialog" titleId={titleId} onClose={requestClose}>
        <header>
          <div>
            <h2 id={titleId}>Edit scenario: {scenario.title}</h2>
            <p className="muted">{scenario.premise || scenario.player_role}</p>
          </div>
          <button type="button" onClick={requestClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        {definition.error instanceof Error ? <InlineNotice>{definition.error.message}</InlineNotice> : null}
        {linkError ? <InlineNotice>{linkError}</InlineNotice> : null}
        {!definition.error && !editorScenario ? <p className="empty">Loading scenario definition...</p> : null}
        <div className="scenario-world-link">
          <label className="field-label">
            <span>Persistent world</span>
            <select value={selectedWorldId} disabled={linkingWorld || editorDirty} onChange={(event) => setSelectedWorldId(event.target.value)}>
              <option value="">Standalone setting</option>
              {worlds.data?.worlds?.map((world) => <option key={world.world_id} value={world.world_id}>{world.title}</option>)}
            </select>
          </label>
          <button
            type="button"
            className="secondary-command compact"
            disabled={linkingWorld || editorDirty || selectedWorldId === (definition.data?.persistent_world?.world_id ?? "")}
            onClick={async () => {
              setLinkError("");
              setLinkingWorld(true);
              try {
                await postJson(`/api/scenarios/${scenario.scenario_id}/persistent-world`, { persistent_world_id: selectedWorldId || null });
                onSaved();
              } catch (error) {
                setLinkError(error instanceof Error ? error.message : "Unable to update the persistent world.");
              } finally {
                setLinkingWorld(false);
              }
            }}
          >
            {linkingWorld ? <Loader2 className="spin" size={14} /> : <Save size={14} />} Apply setting
          </button>
        </div>
        {editorScenario ? (
          <ScenarioStructuredEditor
            scenario={editorScenario}
            onCancel={onClose}
            onDirtyChange={setEditorDirty}
            starterReferenceImages
            interactionModeEditable
            onSave={async (edit) => {
              await postJson(`/api/scenarios/${scenario.scenario_id}/definition`, { edit });
              onSaved();
            }}
          />
        ) : null}
        {discardCloseOpen ? (
          <ConfirmModal
            title="Discard changes?"
            body="Unsaved scenario edits will be lost."
            confirmLabel="Discard"
            destructive
            onCancel={() => setDiscardCloseOpen(false)}
            onConfirm={async () => {
              setDiscardCloseOpen(false);
              onClose();
            }}
          />
        ) : null}
      </DialogPanel>
    </ModalBackdrop>
  );
}

function WorldDataExplorer({
  model,
  editable,
  activeTab,
  setActiveTab,
  onSaveTab
}: {
  model?: WorldDataModel;
  editable: boolean;
  activeTab: WorldDataTab;
  setActiveTab: (tab: WorldDataTab) => void;
  onSaveTab: (tab: WorldDataEditTab, value: unknown) => Promise<void>;
}) {
  const [search, setSearch] = useState("");
  const [visibleRowLimit, setVisibleRowLimit] = useState(WORLD_DATA_PAGE_SIZE);
  const deferredSearch = useDeferredValue(search);
  const activeValue = worldTabValue(model, activeTab);
  const activeEditTab = worldEditTab(model, activeTab);
  const query = normalizedWorldSearchText(deferredSearch.trim());
  const indexedRowsByTab = useMemo(() => WORLD_DATA_TABS.reduce<Record<WorldDataTab, IndexedWorldRow[]>>((result, tab) => {
    result[tab] = indexedWorldRows(worldTabValue(model, tab));
    return result;
  }, {} as Record<WorldDataTab, IndexedWorldRow[]>), [model]);
  const counts = useMemo(() => WORLD_DATA_TABS.reduce<Record<WorldDataTab, number>>((result, tab) => {
    result[tab] = matchingIndexedWorldRows(indexedRowsByTab[tab], query).length;
    return result;
  }, {} as Record<WorldDataTab, number>), [indexedRowsByTab, query]);
  const rows = useMemo(() => matchingIndexedWorldRows(indexedRowsByTab[activeTab], query), [activeTab, indexedRowsByTab, query]);
  const visibleRows = rows.slice(0, visibleRowLimit);
  const totalMatches = WORLD_DATA_TABS.reduce((total, tab) => total + counts[tab], 0);
  const activeGroup = worldDataTabGroup(activeTab);
  const groupCounts = WORLD_DATA_TAB_GROUPS.map((group) => ({
    group,
    matches: query
      ? group.tabs.reduce((total, tab) => total + counts[tab], 0)
      : group.tabs.reduce((total, tab) => total + indexedRowsByTab[tab].length, 0),
    sectionsWithMatches: query
      ? group.tabs.filter((tab) => counts[tab] > 0).length
      : group.tabs.filter((tab) => indexedRowsByTab[tab].length > 0).length
  }));
  const matchingSectionCount = groupCounts.reduce((total, entry) => total + entry.sectionsWithMatches, 0);
  const reviewGroup = WORLD_DATA_TAB_GROUPS.find((group) => group.id === "review");
  const pendingReviewCount = reviewGroup ? pendingSuggestionGroupCount(model) : 0;
  const selectGroup = (groupId: WorldDataTabGroupId) => {
    if (groupId === activeGroup.id) return;
    const fallback = FIRST_TAB_BY_GROUP.get(groupId);
    if (fallback) setActiveTab(fallback);
  };
  const saveRow = (row: WorldDataRow) => onSaveTab(activeEditTab, Array.isArray(activeValue) ? [row] : row);
  const activeScenario = activeTab === "scenario" && model?.scenario ? scenarioEditorValue(model.scenario) : null;
  const activeRowCount = indexedRowsByTab[activeTab].length;
  const renderedRowCount = Math.min(visibleRows.length, rows.length);
  const hasMoreRows = activeTab !== "scenario" && activeTab !== "suggestion_groups" && rows.length > visibleRows.length;
  const searchStatus = (() => {
    if (!query) return rows.length > visibleRows.length ? `${renderedRowCount} of ${activeRowCount} shown` : `${activeRowCount} shown`;
    if (!totalMatches) return "No matches";
    if (matchingSectionCount > 1) return `${totalMatches} matches in ${matchingSectionCount} sections`;
    return `${totalMatches} matches`;
  })();
  useEffect(() => {
    setVisibleRowLimit(WORLD_DATA_PAGE_SIZE);
  }, [activeTab, model?.active_save_id, query]);

  return (
    <div className="world-explorer">
      <div className="world-search">
        <Search size={15} aria-hidden="true" />
        <input
          value={search}
          onChange={(event) => setSearch(event.target.value)}
          placeholder="Find facts, characters, locations..."
          aria-label="Search world data"
        />
        <span>{searchStatus}</span>
      </div>
      <SegmentedTabs
        className="tab-strip world-tabs world-tab-groups"
        label="World data workflow groups"
        value={activeGroup.id}
        onChange={(value: WorldDataTabGroupId) => selectGroup(value)}
        options={WORLD_DATA_TAB_GROUPS.map((group) => ({
          value: group.id,
          label: group.label,
          title: group.description
        }))}
        renderOption={(option) => {
          const groupId = option.value as WorldDataTabGroupId;
          const entry = groupCounts.find((candidate) => candidate.group.id === groupId);
          const showReviewBadge = query
            ? false
            : groupId === "review" && pendingReviewCount > 0;
          const badgeValue = (() => {
            if (query) return entry?.matches ?? 0;
            if (showReviewBadge) return pendingReviewCount;
            return entry?.matches ?? 0;
          })();
          return (
            <>
              <span>{option.label}</span>
              <small>{showReviewBadge ? `${pendingReviewCount} pending` : badgeValue}</small>
            </>
          );
        }}
      />
      <SegmentedTabs
        className="tab-strip world-tabs world-tab-sections"
        label={`${activeGroup.label} sections`}
        value={activeTab}
        onChange={setActiveTab}
        options={activeGroup.tabs.map((name) => ({
          value: name,
          label: worldTabLabel(name),
          title: worldTabTooltip(name)
        }))}
        renderOption={(option) => {
          const name = option.value as WorldDataTab;
          return (
            <>
              <span>{option.label}</span>
              <small>{query ? counts[name] : indexedRowsByTab[name].length}</small>
            </>
          );
        }}
      />
      {editable && activeTab !== "scenario" && activeTab !== "suggestion_groups" && !READONLY_WORLD_TABS.has(activeTab) ? (
        <div className="world-raw-tools">
          <JsonEditSection
            value={activeValue}
            emptyLabel="No world data"
            buttonLabel="Raw JSON"
            onSave={(value) => onSaveTab(activeEditTab, value)}
          />
        </div>
      ) : null}
      {activeTab === "scenario" ? (
        <div className="world-record-list">
          {activeScenario && editable ? (
            <ScenarioStructuredEditor scenario={activeScenario} onSave={(edit) => onSaveTab("scenario", edit)} />
          ) : activeValue ? (
            <DataViewer value={activeValue} emptyLabel="No scenario" />
          ) : (
            <p className="empty">No scenario</p>
          )}
        </div>
      ) : activeTab === "suggestion_groups" ? (
        <WorldSuggestionGroupList
          rows={worldSuggestionGroupRows(activeValue)}
          editable={editable}
          onAction={(row, action) => onSaveTab("suggestion_groups", [{ ...row, action }])}
        />
      ) : (
        <div className="world-record-list">
          {visibleRows.map(({ row, index }) => (
            <WorldDataCard
              key={worldRowKey(activeTab, row, index)}
              tab={activeTab}
              row={row}
              index={index}
              readonly={!editable || READONLY_WORLD_TABS.has(activeTab)}
              onSave={saveRow}
            />
          ))}
          {hasMoreRows ? (
            <button
              type="button"
              className="secondary-command world-show-more"
              onClick={() => setVisibleRowLimit((current) => current + WORLD_DATA_PAGE_SIZE)}
            >
              Show more world data
            </button>
          ) : null}
          {!rows.length ? <p className="empty">{query ? "No matching world data" : "No world data"}</p> : null}
        </div>
      )}
    </div>
  );
}

function WorldDataCard({
  tab,
  row,
  index,
  readonly,
  onSave
}: {
  tab: WorldDataTab;
  row: WorldDataRow;
  index: number;
  readonly: boolean;
  onSave: (row: WorldDataRow) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [error, setError] = useState("");
  const open = editing || expanded;
  const save = async (next: WorldDataRow) => {
    try {
      await onSave(next);
      setError("");
      setEditing(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not save world data");
    }
  };

  return (
    <details
      className="world-card"
      open={open}
      onToggle={(event) => setExpanded(event.currentTarget.open)}
    >
      <summary>
        <div>
          <strong>{worldRowTitle(tab, row, index)}</strong>
          <span>{worldRowSubtitle(tab, row)}</span>
        </div>
        <div className="world-card-tools">
          {worldRowPills(tab, row).map((pill) => <small key={pill}>{pill}</small>)}
          {!readonly ? (
            <button
              type="button"
              className={touchActionClassName()}
              title="Edit"
              aria-label={`Edit ${worldRowTitle(tab, row, index)}`}
              onClick={(event) => {
                event.preventDefault();
                setExpanded(true);
                setEditing(true);
              }}
            >
              <TouchActionContents icon={<Edit3 size={14} />} label="Edit" />
            </button>
          ) : null}
        </div>
      </summary>
      {editing ? (
        tab === "world_state" ? (
          <WorldStateRowEditor row={row} error={error} onCancel={() => setEditing(false)} onSave={save} />
        ) : (
          <GenericWorldRowEditor tab={tab} row={row} error={error} onCancel={() => setEditing(false)} onSave={save} />
        )
      ) : expanded ? (
        <>
          <DataViewer value={row} emptyLabel="No world data" />
          {error ? <InlineNotice>{error}</InlineNotice> : null}
        </>
      ) : null}
    </details>
  );
}

function WorldStateRowEditor({
  row,
  error,
  onCancel,
  onSave
}: {
  row: WorldDataRow;
  error: string;
  onCancel: () => void;
  onSave: (row: WorldDataRow) => Promise<void>;
}) {
  const [draft, setDraft] = useState<WorldDataRow>(row);
  const [jsonText, setJsonText] = useState(prettyJsonText(row.value_json));
  const [localError, setLocalError] = useState("");
  const update = (key: string, value: unknown) => setDraft((current) => ({ ...current, [key]: value }));
  return (
    <div className="world-editor-grid">
      <label className="field-label">
        <span>Key</span>
        <input value={String(draft.key ?? "")} onChange={(event) => update("key", event.target.value)} />
      </label>
      <label className="field-label">
        <span>Category</span>
        <input value={String(draft.category ?? "")} onChange={(event) => update("category", event.target.value)} />
      </label>
      <label className="field-label">
        <span>Confidence</span>
        <input type="number" min={0} max={1} step={0.01} value={String(draft.confidence ?? 0)} onChange={(event) => update("confidence", Number(event.target.value))} />
      </label>
      <label className="toggle-row compact-toggle world-archive-toggle">
        <input type="checkbox" checked={Boolean(draft.archived)} onChange={(event) => update("archived", event.target.checked)} />
        <span>Archive</span>
      </label>
      <label className="field-label world-json-field">
        <span>Fact Value JSON</span>
        <textarea className="json-editor compact-json-editor" value={jsonText} onChange={(event) => setJsonText(event.target.value)} />
      </label>
      {draft.source_message_id ? <p className="muted world-source">Source message: {String(draft.source_message_id)}</p> : null}
      {localError || error ? <InlineNotice>{localError || error}</InlineNotice> : null}
      <div className="command-row end world-editor-actions">
        <button type="button" onClick={onCancel}>Cancel</button>
        <button
          type="button"
          className="primary-command compact"
          onClick={() => {
            try {
              const parsed = JSON.parse(jsonText || "{}");
              if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
                setLocalError("Fact value must be a JSON object");
                return;
              }
              setLocalError("");
              onSave({ ...draft, value_json: JSON.stringify(parsed) });
            } catch {
              setLocalError("Fact value must be valid JSON");
            }
          }}
        >
          <Save size={15} /> Save
        </button>
      </div>
    </div>
  );
}

function GenericWorldRowEditor({
  tab,
  row,
  error,
  onCancel,
  onSave
}: {
  tab: WorldDataTab;
  row: WorldDataRow;
  error: string;
  onCancel: () => void;
  onSave: (row: WorldDataRow) => Promise<void>;
}) {
  const [draft, setDraft] = useState<WorldDataRow>(row);
  const lockFields = lockableWorldFields(tab);
  const jsonFieldNames = Object.entries(row)
    .filter(([key, value]) => {
      if (key === "locked_fields" && lockFields.length) return false;
      return value !== null && typeof value === "object";
    })
    .map(([key]) => key);
  const [jsonText, setJsonText] = useState<Record<string, string>>(() => Object.fromEntries(
    jsonFieldNames.map((key) => [key, JSON.stringify(row[key], null, 2)])
  ));
  const [localError, setLocalError] = useState("");
  const update = (key: string, value: unknown) => setDraft((current) => ({ ...current, [key]: value }));
  const fields = Object.entries(draft).filter(([key]) => {
    if (HIDDEN_WORLD_FIELDS.has(key)) return false;
    return !(key === "locked_fields" && lockFields.length);
  });
  return (
    <div className="world-editor-stack">
      {lockFields.length ? (
        <WorldLockedFieldsEditor
          fields={lockFields}
          value={draft.locked_fields}
          onChange={(next) => update("locked_fields", next)}
        />
      ) : null}
      {fields.map(([key, value]) => (
        <WorldFieldEditor
          key={key}
          name={key}
          value={value}
          jsonText={jsonText[key]}
          onChange={(next) => update(key, next)}
          onJsonTextChange={(next) => setJsonText((current) => ({ ...current, [key]: next }))}
        />
      ))}
      {localError || error ? <InlineNotice>{localError || error}</InlineNotice> : null}
      <div className="command-row end">
        <button type="button" onClick={onCancel}>Cancel</button>
        <button
          type="button"
          className="primary-command compact"
          onClick={() => {
            const next = { ...draft };
            try {
              for (const key of jsonFieldNames) next[key] = JSON.parse(jsonText[key] || "null");
            } catch {
              setLocalError("Nested field JSON must be valid");
              return;
            }
            setLocalError("");
            onSave(next);
          }}
        >
          <Save size={15} /> Save
        </button>
      </div>
    </div>
  );
}

type WorldSuggestionAction = "apply" | "reject" | "dismiss";
type WorldLockField = readonly [string, string];

function WorldSuggestionGroupList({
  rows,
  editable,
  onAction
}: {
  rows: WorldDataSuggestionGroupRow[];
  editable: boolean;
  onAction: (row: WorldDataSuggestionGroupRow, action: WorldSuggestionAction) => Promise<void>;
}) {
  if (!rows.length) return <p className="empty">No pending suggestions</p>;
  return (
    <div className="world-record-list suggestion-review-list">
      {rows.map((row) => (
        <WorldSuggestionGroupCard
          key={row.group_id}
          row={row}
          editable={editable}
          onAction={onAction}
        />
      ))}
    </div>
  );
}

function WorldSuggestionGroupCard({
  row,
  editable,
  onAction
}: {
  row: WorldDataSuggestionGroupRow;
  editable: boolean;
  onAction: (row: WorldDataSuggestionGroupRow, action: WorldSuggestionAction) => Promise<void>;
}) {
  const [busyAction, setBusyAction] = useState<WorldSuggestionAction | null>(null);
  const [error, setError] = useState("");
  const run = async (action: WorldSuggestionAction) => {
    try {
      setBusyAction(action);
      setError("");
      await onAction(row, action);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not update suggestion");
    } finally {
      setBusyAction(null);
    }
  };
  return (
    <article className="world-card suggestion-review-card">
      <header className="suggestion-review-header">
        <div>
          <strong>{row.field_path}</strong>
          <span>{suggestionTargetLabel(row)}</span>
        </div>
        <div className="world-card-tools">
          <small>{row.suggestion_count} grouped</small>
          <small>{Math.round(row.confidence * 100)}%</small>
          <small>{row.status}</small>
        </div>
      </header>
      <div className="suggestion-review-body">
        <div className="kv-list">
          <div className="kv-row">
            <span>Value</span>
            <strong>{suggestionValuePreview(row.proposed_value_json)}</strong>
          </div>
          {row.source_message_ids_text ? (
            <div className="kv-row">
              <span>Sources</span>
              <strong>{row.source_message_ids_text}</strong>
            </div>
          ) : null}
        </div>
        {row.reason ? <p className="muted">{row.reason}</p> : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
      </div>
      {editable ? (
        <div className="suggestion-actions">
          <button
            type="button"
            className="primary-command compact"
            disabled={busyAction !== null}
            aria-label={`Apply suggestion ${row.field_path}`}
            onClick={() => void run("apply")}
          >
            {busyAction === "apply" ? <Loader2 className="spin" size={14} /> : <Check size={14} />} Apply
          </button>
          <button
            type="button"
            disabled={busyAction !== null}
            aria-label={`Reject suggestion ${row.field_path}`}
            onClick={() => void run("reject")}
          >
            <X size={14} /> Reject
          </button>
          <button
            type="button"
            disabled={busyAction !== null}
            aria-label={`Dismiss suggestion ${row.field_path}`}
            onClick={() => void run("dismiss")}
          >
            <Clock size={14} /> Dismiss
          </button>
        </div>
      ) : null}
    </article>
  );
}

function WorldLockedFieldsEditor({
  fields,
  value,
  onChange
}: {
  fields: readonly WorldLockField[];
  value: unknown;
  onChange: (value: string[]) => void;
}) {
  const locked = worldLockedFields(value);
  const selected = new Set(locked);
  const known = new Set(fields.map(([field]) => field));
  const setLocked = (field: string, enabled: boolean) => {
    const next = new Set(locked);
    if (enabled) {
      next.add(field);
    } else {
      next.delete(field);
    }
    onChange([
      ...locked.filter((candidate) => !known.has(candidate) && next.has(candidate)),
      ...fields.map(([candidate]) => candidate).filter((candidate) => next.has(candidate))
    ]);
  };
  return (
    <div className="field-label world-lock-fields">
      <span>Locked Fields</span>
      <div className="checkbox-grid world-lock-grid">
        {fields.map(([field, label]) => (
          <label key={field}>
            <input
              type="checkbox"
              checked={selected.has(field)}
              aria-label={`Lock ${labelize(field)}`}
              onChange={(event) => setLocked(field, event.target.checked)}
            />
            {`Lock ${label}`}
          </label>
        ))}
      </div>
    </div>
  );
}

function WorldFieldEditor({
  name,
  value,
  jsonText,
  onChange,
  onJsonTextChange
}: {
  name: string;
  value: unknown;
  jsonText?: string;
  onChange: (value: unknown) => void;
  onJsonTextChange: (value: string) => void;
}) {
  if (typeof value === "boolean") {
    return (
      <label className="toggle-row compact-toggle">
        <input type="checkbox" checked={value} onChange={(event) => onChange(event.target.checked)} />
        <span>{labelize(name)}</span>
      </label>
    );
  }
  if (typeof value === "number") {
    return (
      <label className="field-label">
        <span>{labelize(name)}</span>
        <input type="number" value={String(value)} onChange={(event) => onChange(Number(event.target.value))} />
      </label>
    );
  }
  if (value !== null && typeof value === "object") {
    return (
      <label className="field-label">
        <span>{labelize(name)}</span>
        <textarea className="json-editor compact-json-editor" value={jsonText ?? ""} onChange={(event) => onJsonTextChange(event.target.value)} />
      </label>
    );
  }
  const text = value === null || value === undefined ? "" : String(value);
  const useTextarea = TEXTAREA_FIELD_NAMES.has(name) || text.length > 90;
  return (
    <label className="field-label">
      <span>{labelize(name)}</span>
      {useTextarea ? (
        <textarea value={text} onChange={(event) => onChange(event.target.value)} />
      ) : (
        <input value={text} onChange={(event) => onChange(event.target.value)} />
      )}
    </label>
  );
}

function DataViewer({ value, emptyLabel }: { value: unknown; emptyLabel: string }) {
  if (value === null || value === undefined || value === "" || (Array.isArray(value) && value.length === 0)) return <p className="empty">{emptyLabel}</p>;
  if (Array.isArray(value)) {
    return (
      <div className="data-summary">
        {value.map((item, index) => (
          <details className="entity-detail" key={index}>
            <summary>
              <strong>{itemTitle(item, index)}</strong>
              <span>{itemSubtitle(item)}</span>
            </summary>
            <DataViewer value={item} emptyLabel={emptyLabel} />
          </details>
        ))}
      </div>
    );
  }
  if (typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>);
    if (!entries.length) return <p className="empty">{emptyLabel}</p>;
    return (
      <div className="kv-list">
        {entries.map(([key, item]) => (
          <div className="kv-row" key={key}>
            <span>{labelize(key)}</span>
            <strong>{formatValue(item)}</strong>
          </div>
        ))}
      </div>
    );
  }
  return <p className="muted">{String(value)}</p>;
}

function JsonEditSection({ value, emptyLabel, buttonLabel = "Edit", onSave }: { value: unknown; emptyLabel: string; buttonLabel?: string; onSave: (value: unknown) => Promise<void> }) {
  const [editing, setEditing] = useState(false);
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  if (!editing) {
    return (
      <>
        <div className="command-row end">
          <button
            type="button"
            onClick={() => {
              setText(JSON.stringify(value ?? null, null, 2));
              setEditing(true);
              setError("");
            }}
          >
            <Edit3 size={15} /> {buttonLabel}
          </button>
          {buttonLabel === "Edit" ? null : <span className="sr-only">{emptyLabel}</span>}
        </div>
        {buttonLabel === "Edit" ? <DataViewer value={value} emptyLabel={emptyLabel} /> : null}
      </>
    );
  }
  const editorLabel = buttonLabel === "Raw JSON" ? "Raw JSON" : `${emptyLabel} JSON`;
  return (
    <div className="settings-stack">
      <textarea className="json-editor" value={text} onChange={(event) => setText(event.target.value)} aria-label={editorLabel} />
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      <div className="command-row end">
        <button type="button" onClick={() => setEditing(false)}>Cancel</button>
        <button
          type="button"
          className="primary-command compact"
          onClick={async () => {
            try {
              await onSave(JSON.parse(text));
              setEditing(false);
              setError("");
            } catch (failure) {
              setError(failure instanceof Error ? failure.message : "Could not save edits");
            }
          }}
        >
          <Save size={15} /> Save
        </button>
      </div>
    </div>
  );
}

function PreviewModal({
  title,
  preview,
  detail,
  confirmLabel,
  onCancel,
  onConfirm,
  extra
}: {
  title: string;
  preview: BundlePreview;
  detail: string;
  confirmLabel: string;
  onCancel: () => void;
  onConfirm: () => Promise<void>;
  extra?: React.ReactNode;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog" titleId={titleId} onClose={onCancel}>
        <header>
          <h2 id={titleId}>{title}</h2>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <div className="preview-title">
          <strong>{preview.title}</strong>
          <span>{preview.scenario_title || "Untitled scenario"}</span>
        </div>
        <div className="preview-grid">
          <div><span>Messages</span><strong>{preview.message_count}</strong></div>
          <div><span>Media</span><strong>{preview.media_count}</strong></div>
          <div><span>Bundle</span><strong>v{preview.bundle_version}</strong></div>
        </div>
        {extra}
        <p className="muted">{detail}</p>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onCancel}>Cancel</button>
          <button
            className="primary-command compact"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              setError("");
              try {
                await onConfirm();
              } catch (failure) {
                setError(failure instanceof Error ? failure.message : "Could not complete action");
              } finally {
                setBusy(false);
              }
            }}
          >
            {confirmLabel}
          </button>
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function ConfirmModal({
  title,
  body,
  confirmLabel,
  destructive = false,
  disabled = false,
  onCancel,
  onConfirm
}: {
  title: string;
  body: string;
  confirmLabel: string;
  destructive?: boolean;
  disabled?: boolean;
  onCancel: () => void;
  onConfirm: () => Promise<void>;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog" titleId={titleId} onClose={onCancel}>
        <header>
          <h2 id={titleId}>{title}</h2>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <p className="muted">{body}</p>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onCancel}>Cancel</button>
          <button
            className={destructive ? "danger-command compact" : "primary-command compact"}
            disabled={busy || disabled}
            onClick={async () => {
              setBusy(true);
              setError("");
              try {
                await onConfirm();
              } catch (failure) {
                setError(failure instanceof Error ? failure.message : "Could not complete action");
              } finally {
                setBusy(false);
              }
            }}
          >
            {confirmLabel}
          </button>
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function selectedOption(selector: TaskModelSelector): ModelOption | undefined {
  return selector.options.find((option) => option.provider === selector.selected_provider && option.model_id === selector.selected_model_id);
}

function ModelPricingLine({ option, showUnavailable = false }: { option?: ModelOption; showUnavailable?: boolean }) {
  if (!option && !showUnavailable) return null;
  return (
    <p className="model-pricing-line">
      {modelPricingDisplayLabel(option?.pricing) ?? "Pricing unavailable"}
    </p>
  );
}

function hasAction(message: ChronicleMessage, ...ids: string[]) {
  return message.actions.some((action) => ids.includes(action.action_id));
}

function modelOptionSelectLabel(option: ModelOption) {
  const pricing = modelPricingCompactLabel(option.pricing);
  const label = modelOptionLabel(option.display_name, option.provider, option.model_id);
  return pricing ? `${label} · ${pricing}` : label;
}

function modelOptionLabel(displayName: string, provider: string, modelId: string) {
  return `${displayName} - ${provider}/${modelId}`;
}

function modelPricingDisplayLabel(pricing?: ModelOption["pricing"] | null): string | null {
  if (!pricing) return null;
  const tokenPricing = modelTokenPricingDisplayLabel(pricing);
  const cachePricing = modelCachePricingDisplayLabel(pricing);
  const fixedPricing = modelFixedPricingDisplayLabel(pricing);
  const note = pricing.note?.trim() || null;
  return [tokenPricing, cachePricing, fixedPricing, note].filter(Boolean).join(" · ") || null;
}

function modelPricingCompactLabel(pricing?: ModelOption["pricing"] | null): string | null {
  if (!pricing) return null;
  const tokenPricing = modelTokenPricingCompactLabel(pricing);
  const cachePricing = modelCachePricingCompactLabel(pricing);
  if (tokenPricing || cachePricing) return [tokenPricing, cachePricing].filter(Boolean).join(" · ");
  if (pricing.image_usd) {
    const image = formatUsd(pricing.image_usd);
    if (image) return `${image}/image`;
  }
  if (pricing.request_usd) {
    const request = formatUsd(pricing.request_usd);
    if (request) return `${request}/request`;
  }
  return pricing.note?.trim() || null;
}

function modelTokenPricingCompactLabel(pricing: NonNullable<ModelOption["pricing"]>): string | null {
  const input = formatUsd(pricing.input_per_million_tokens_usd);
  const output = formatUsd(pricing.output_per_million_tokens_usd);
  if (input && output) return `${input} in / ${output} out per 1M`;
  if (input) return `${input} in per 1M`;
  if (output) return `${output} out per 1M`;
  return null;
}

function modelTokenPricingDisplayLabel(pricing: NonNullable<ModelOption["pricing"]>): string | null {
  const input = formatUsd(pricing.input_per_million_tokens_usd);
  const output = formatUsd(pricing.output_per_million_tokens_usd);
  if (input && output) return `Input ${input} / output ${output} per 1M tokens`;
  if (input) return `Input ${input} per 1M tokens`;
  if (output) return `Output ${output} per 1M tokens`;
  return null;
}

function modelCachePricingCompactLabel(pricing: NonNullable<ModelOption["pricing"]>): string | null {
  const read = formatUsd(pricing.cache_read_per_million_tokens_usd);
  const write = formatUsd(pricing.cache_write_per_million_tokens_usd);
  if (read && write) return `cache ${read} read / ${write} write per 1M`;
  if (read) return `cache ${read} read per 1M`;
  if (write) return `cache ${write} write per 1M`;
  return null;
}

function modelCachePricingDisplayLabel(pricing: NonNullable<ModelOption["pricing"]>): string | null {
  const read = formatUsd(pricing.cache_read_per_million_tokens_usd);
  const write = formatUsd(pricing.cache_write_per_million_tokens_usd);
  if (read && write) return `Cache read ${read} / write ${write} per 1M tokens`;
  if (read) return `Cache read ${read} per 1M tokens`;
  if (write) return `Cache write ${write} per 1M tokens`;
  return null;
}

function modelFixedPricingDisplayLabel(pricing: NonNullable<ModelOption["pricing"]>): string | null {
  const image = formatUsd(pricing.image_usd);
  if (image) return `${image} per image`;
  const request = formatUsd(pricing.request_usd);
  if (request) return `${request} per request`;
  return null;
}

function formatUsd(value?: string | null): string | null {
  if (!value) return null;
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < 0) return null;
  const minimumFractionDigits = parsed === 0 ? 0 : 2;
  return `$${parsed.toLocaleString("en-US", {
    minimumFractionDigits,
    maximumFractionDigits: 8
  })}`;
}

function taskLabel(task: string) {
  const normalized = task
    .replace(/^full_roleplay_/, "")
    .replace(/^fantasy_roleplay_/, "")
    .replace(/^science_fiction_roleplay_/, "")
    .replace(/^first_contact_exploration_/, "")
    .replace(/^survival_expedition_/, "")
    .replace(/^time_loop_/, "")
    .replace(/^investigation_mystery_/, "")
    .replace(/^heist_infiltration_/, "")
    .replace(/^political_intrigue_/, "")
    .replace(/^dating_sim_/, "")
    .replace(/^shared_roleplay_/, "");
  return fallbackTaskLabels[normalized] ?? labelize(normalized);
}

const fallbackTaskLabels: Record<string, string> = {
  fact_observation: "Fact Observation",
  memory_curation: "Memory Curation",
  response_planning: "Narrator Planner",
  response_verification: "Narrator Verifier",
  content_safety: "Safety Agent",
  director_pressure: "Director Pressure",
  action_choice_generation: "Action Choice Generation",
  character_presence_assessment: "Character Presence Assessment",
  character_intent_planning: "Character Intent Planning",
  character_action_planning: "Character Action Planning",
  character_enhancement: "Character Enhancement",
  character_registry_maintenance: "Character Registry Maintenance",
  context_cleanup_scan: "Context Cleanup Scan",
  context_cleanup_actions: "Context Cleanup Actions",
  guided_context_cleanup: "Guided Context Cleanup",
  context_cleanup: "Context Cleanup",
  scenario_evolution: "Scenario Evolution",
  npc_knowledge_audit: "NPC Knowledge Audit",
  narrator_fallback: "Narrator Fallback",
  chat_fallback: "Background Text Fallback",
  structured_output_fallback: "Structured Output Fallback",
  tool_call_fallback: "Tool Fallback",
  image_to_image_generation: "Default Image Edit",
  scene_image_edit_generation: "Scene Image Edit",
  character_image_edit_generation: "Character Image Edit",
  text_message_image_edit_generation: "Text Message Image Edit",
  character_image_description: "Image Details",
  image_fallback: "Image Fallback",
  image_edit_fallback: "Image Edit Fallback",
  video_fallback: "Video Fallback"
};

function settingLabel(settingKey: string) {
  return fallbackSettingLabels[settingKey] ?? labelize(settingKey);
}

const fallbackSettingLabels: Record<string, string> = {
  turn_responsiveness_mode: "Turn Responsiveness Mode",
  agentic_context_pipeline_enabled: "Agentic Context Pipeline",
  plan_first_narrator_enabled: "Plan-First Narrator",
  director_pressure_enabled: "Director Pressure",
  character_action_planning_enabled: "Character Action Planning",
  character_action_planning_max_concurrency: "Character Action Planning Max Concurrency",
  character_text_proactive_random_chance_percent: "Proactive Text Chance",
  character_text_proactive_random_cooldown_turns: "Proactive Text Cooldown",
  post_turn_inference_mode: "Post-Turn Inference Mode",
  npc_knowledge_audit_mode: "NPC Knowledge Audit Mode",
  response_checking_enabled: "Response Checking",
  generated_text_script_guard_mode: "Generated Text Script Guard",
  retry_count: "Retries after first",
  provider_call_deadline_seconds: "Provider Call Deadline (seconds)",
  generated_phrase_denylist: "Global Phrase Denylist",
  save_generated_phrase_denylist: "Save Phrase Denylist",
  narrator_planner_recent_player_message_window: "Planner Player Messages",
  narrator_planner_recent_narrator_message_window: "Planner Narrator Messages",
  recent_player_message_window: "Prose Player Messages",
  recent_narrator_message_window: "Prose Narrator Messages",
  chat_fallback_enabled: "Chat Fallback Enabled",
  chat_temperature_enabled: "Chat Temperature Enabled",
  chat_max_output_tokens_enabled: "Chat Max Output Tokens Enabled",
  structured_output_fallback_enabled: "Structured Output Fallback Enabled",
  tool_call_fallback_enabled: "Tool Fallback Enabled",
  image_fallback_enabled: "Image Fallback Enabled",
  video_fallback_enabled: "Video Fallback Enabled",
  venice_image_safe_mode: "Venice Media Safe Mode",
  content_filter_rating: "Content Filtering Level",
  fade_to_black_enabled: "Fade Explicit Content to Black",
  pending_jobs_display_mode: "Pending Jobs Display Mode",
  user_narration_guidance: "Narration Guidance"
};

function pendingJobsDisplayModeLabel(mode: string) {
  return pendingJobsDisplayModeLabels[mode] ?? labelize(mode);
}

const pendingJobsDisplayModeLabels: Record<string, string> = {
  compact: "Compact",
  expanded: "Expanded Turns",
  expanded_full: "Expanded Full"
};

function npcKnowledgeAuditModeLabel(mode: string) {
  return npcKnowledgeAuditModeLabels[mode] ?? labelize(mode);
}

const npcKnowledgeAuditModeLabels: Record<string, string> = {
  soft_fail: "Soft fail",
  hard_fail: "Hard fail"
};

function scriptGuardModeLabel(mode: string) {
  return scriptGuardModeLabels[mode] ?? labelize(mode);
}

const scriptGuardModeLabels: Record<string, string> = {
  source_aware_reject: "Source-aware reject",
  latin_only_reject: "Latin-only reject",
  off: "Off"
};

function postTurnInferenceModeLabel(mode: string) {
  return postTurnInferenceModeLabels[mode] ?? labelize(mode);
}

const postTurnInferenceModeLabels: Record<string, string> = {
  legacy: "Legacy",
  hybrid: "Hybrid",
  plan_owned: "Plan-owned"
};

function imageStylePresetLabel(preset: string) {
  return imageStylePresetLabels[preset] ?? labelize(preset);
}

const imageStylePresetLabels: Record<string, string> = {
  none: "No preset",
  realistic: "Realistic",
  anime: "Anime",
  cartoon: "Cartoon",
  cinematic: "Cinematic",
  concept_art: "Concept Art",
  digital_painting: "Digital Painting",
  watercolor: "Watercolor",
  oil_painting: "Oil Painting",
  comic_book: "Comic Book",
  colored_pencil: "Colored Pencil",
  sketch: "Sketch",
  ink: "Ink",
  pixel_art: "Pixel Art",
  three_d_render: "3D Render",
  low_poly: "Low Poly"
};

function imageDimensionPresetLabel(preset: string) {
  return imageDimensionPresetLabels[preset] ?? labelize(preset);
}

const imageDimensionPresetLabels: Record<string, string> = {
  provider_default: "Provider default",
  square_1024x1024: "Square 1024x1024",
  landscape_1024x768: "Landscape 1024x768",
  portrait_768x1024: "Portrait 768x1024",
  wide_1024x576: "Wide 1024x576",
  tall_576x1024: "Tall 576x1024"
};

function thinkingLevelLabel(level: string) {
  return thinkingLevelLabels[level] ?? labelize(level);
}

const thinkingLevelLabels: Record<string, string> = {
  provider_default: "Provider default",
  off: "Off",
  max: "Max",
  xhigh: "Extra high",
  high: "High",
  medium: "Medium",
  low: "Low",
  minimal: "Minimal",
  none: "None"
};

function worldSuggestionGroupRows(value: unknown): WorldDataSuggestionGroupRow[] {
  return worldRows(value).flatMap(({ row }) => {
    const group = row as Partial<WorldDataSuggestionGroupRow>;
    if (typeof group.group_id !== "string" || !group.group_id) return [];
    if (typeof group.field_path !== "string") return [];
    return [{
      group_id: group.group_id,
      suggestion_ids: Array.isArray(group.suggestion_ids) ? group.suggestion_ids.filter((id): id is string => typeof id === "string") : [],
      update_type: typeof group.update_type === "string" ? group.update_type : "",
      entity_type: typeof group.entity_type === "string" ? group.entity_type : "",
      entity_id: typeof group.entity_id === "string" || group.entity_id === null ? group.entity_id : null,
      field_path: group.field_path,
      proposed_value_json: typeof group.proposed_value_json === "string" ? group.proposed_value_json : "",
      status: typeof group.status === "string" ? group.status : "",
      reason: typeof group.reason === "string" ? group.reason : "",
      confidence: typeof group.confidence === "number" ? group.confidence : 0,
      source_message_ids_text: typeof group.source_message_ids_text === "string" ? group.source_message_ids_text : "",
      suggestion_count: typeof group.suggestion_count === "number" ? group.suggestion_count : 1,
      action: typeof group.action === "string" ? group.action : ""
    }];
  });
}

function suggestionTargetLabel(row: WorldDataSuggestionGroupRow): string {
  const entity = row.entity_id ? `${row.entity_type}/${row.entity_id}` : row.entity_type;
  return [row.update_type, entity].filter(Boolean).join(" · ");
}

function suggestionValuePreview(value: string): string {
  try {
    return compactInlineTitle(JSON.stringify(JSON.parse(value)), value);
  } catch {
    return compactInlineTitle(value, "");
  }
}

function lockableWorldFields(tab: WorldDataTab): readonly WorldLockField[] {
  if (tab === "scene") return SCENE_LOCK_FIELDS;
  if (tab === "locations") return LOCATION_LOCK_FIELDS;
  if (tab === "threads") return THREAD_LOCK_FIELDS;
  return [];
}

function worldLockedFields(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((field): field is string => typeof field === "string" && field.trim().length > 0);
}

function capabilityLabel(capability: string) {
  return capability.split("_").join(" ");
}

type IndexedWorldRow = { row: WorldDataRow; index: number; searchText: string };

function indexedWorldRows(value: unknown): IndexedWorldRow[] {
  return worldRows(value).map(({ row, index }) => ({
    row,
    index,
    searchText: worldSearchText(row)
  }));
}

function matchingIndexedWorldRows(rows: IndexedWorldRow[], query: string): { row: WorldDataRow; index: number }[] {
  const normalizedQuery = normalizedWorldSearchText(query);
  if (!normalizedQuery) return rows.map(({ row, index }) => ({ row, index }));
  return rows
    .filter((entry) => entry.searchText.includes(normalizedQuery))
    .map(({ row, index }) => ({ row, index }));
}

function matchingWorldRows(value: unknown, query: string): { row: WorldDataRow; index: number }[] {
  return matchingIndexedWorldRows(indexedWorldRows(value), query);
}

function worldTabValue(model: WorldDataModel | undefined, tab: WorldDataTab): unknown {
  if (!model) return undefined;
  return model[tab];
}

function worldEditTab(_model: WorldDataModel | undefined, tab: WorldDataTab): WorldDataEditTab {
  return tab;
}

function worldRows(value: unknown): { row: WorldDataRow; index: number }[] {
  if (value === null || value === undefined || value === "") return [];
  if (Array.isArray(value)) {
    return value
      .map((item, index) => ({ row: worldRowObject(item), index }))
      .filter(({ row }) => Object.keys(row).length > 0);
  }
  if (typeof value === "object") return [{ row: value as WorldDataRow, index: 0 }];
  return [{ row: { value }, index: 0 }];
}

function worldRowObject(item: unknown): WorldDataRow {
  if (item && typeof item === "object" && !Array.isArray(item)) return item as WorldDataRow;
  return { value: item };
}

function worldTabSize(value: unknown): number {
  return worldRows(value).length;
}

function normalizedWorldSearchText(value: string): string {
  return value.toLowerCase().replace(/[-_]+/g, " ");
}

function worldSearchText(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (Array.isArray(value)) return normalizedWorldSearchText(value.map(worldSearchText).join(" "));
  if (typeof value === "object") {
    return normalizedWorldSearchText(Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => `${key} ${worldSearchText(item)}`)
      .join(" "));
  }
  return normalizedWorldSearchText(String(value));
}

function worldRowKey(tab: WorldDataTab, row: WorldDataRow, index: number): string {
  const id = row.row_id ?? row.state_id ?? row.memory_id ?? row.context_source_id ?? row.summary_id ?? row.snapshot_id ?? row.location_id ?? row.character_id ?? row.thread_id ?? row.link_id ?? row.group_id ?? row.suggestion_id ?? row.audit_id ?? row.condition_id ?? row.outcome_id ?? row.scenario_id;
  return `${tab}:${String(id ?? index)}`;
}

function worldRowTitle(tab: WorldDataTab, row: WorldDataRow, index: number): string {
  if (tab === "world_state") return String(row.key || `Fact ${index + 1}`);
  if (tab === "memories") return compactInlineTitle(row.body, `Memory ${index + 1}`);
  if (tab === "context_inputs") return String(row.title || `Context input ${index + 1}`);
  if (tab === "scene") return String(row.situation || row.objective || "Scene");
  if (tab === "scenario") return String(row.title || "Scenario");
  if (tab === "links") return `${String(row.entity_type ?? "Entity")} -> ${String(row.target_type ?? "Target")}`;
  return itemTitle(row, index);
}

function worldRowSubtitle(tab: WorldDataTab, row: WorldDataRow): string {
  if (tab === "world_state") return previewWorldStateValue(row);
  if (tab === "memories") return memorySubtitle(row);
  if (tab === "context_inputs") return contextInputSubtitle(row);
  if (tab === "scene") return [row.current_location_id, row.weather, row.mood].filter(Boolean).join(" · ");
  if (tab === "links") return [row.entity_id, row.relation, row.target_id].filter(Boolean).join(" · ");
  return itemSubtitle(row) || conciseRowPreview(row);
}

function worldRowPills(tab: WorldDataTab, row: WorldDataRow): string[] {
  const pills: string[] = [];
  if (tab === "memories") {
    if (row.consolidated) pills.push("consolidated");
    const count = sourceMessageCount(row);
    if (count !== null) pills.push(sourceCountLabel(count));
  }
  if (tab === "context_inputs") {
    if (typeof row.fact_type === "string" && row.fact_type) pills.push(row.fact_type);
    if (typeof row.importance === "number") pills.push(`${Math.round(row.importance * 100)}%`);
    if (typeof row.source_message_count === "number") pills.push(sourceCountLabel(row.source_message_count));
  }
  if (typeof row.category === "string" && row.category) pills.push(row.category);
  if (typeof row.status === "string" && row.status) pills.push(row.status);
  if (typeof row.confidence === "number") pills.push(`${Math.round(row.confidence * 100)}%`);
  if (Array.isArray(row.locked_fields) && row.locked_fields.length) pills.push(`${row.locked_fields.length} locked`);
  if (row.archived) pills.push("archived");
  return pills.slice(0, 3);
}

function memorySubtitle(row: WorldDataRow): string {
  const parts = [
    typeof row.tags_text === "string" && row.tags_text ? row.tags_text : null,
    typeof row.source_message_id === "string" && row.source_message_id ? `source ${row.source_message_id}` : null
  ];
  return parts.filter(Boolean).join(" · ") || compactInlineTitle(row.body, "");
}

function contextInputSubtitle(row: WorldDataRow): string {
  return [
    contextSourceLabel(row),
    compactInlineTitle(row.body, "")
  ].filter(Boolean).join(" · ");
}

function contextSourceLabel(row: WorldDataRow): string {
  const sourceType = typeof row.source_type === "string" ? row.source_type : "";
  const sourceId = typeof row.source_id === "string" ? row.source_id : "";
  if (sourceType && sourceId) return `${sourceType}:${sourceId}`;
  return sourceType || sourceId;
}

function sourceMessageCount(row: WorldDataRow): number | null {
  if (Array.isArray(row.source_message_ids)) return row.source_message_ids.length;
  if (typeof row.source_message_count === "number") return row.source_message_count;
  return row.source_message_id ? 1 : null;
}

function sourceCountLabel(count: number): string {
  return `${count} source${count === 1 ? "" : "s"}`;
}

function compactInlineTitle(value: unknown, fallback: string): string {
  if (typeof value !== "string") return fallback;
  const compact = value.replace(/\s+/g, " ").trim();
  if (!compact) return fallback;
  return compact.length > 96 ? `${compact.slice(0, 95).trim()}...` : compact;
}

function previewWorldStateValue(row: WorldDataRow): string {
  if (typeof row.value_json !== "string" || !row.value_json) return "";
  try {
    const parsed = JSON.parse(row.value_json);
    return conciseRowPreview(parsed);
  } catch {
    return row.value_json.slice(0, 120);
  }
}

function conciseRowPreview(value: unknown): string {
  if (value === null || value === undefined || value === "") return "";
  if (Array.isArray(value)) return `${value.length} items`;
  if (typeof value !== "object") return String(value).slice(0, 120);
  return Object.entries(value as Record<string, unknown>)
    .filter(([, item]) => item !== null && item !== undefined && item !== "")
    .slice(0, 3)
    .map(([key, item]) => `${labelize(key)}: ${formatValue(item)}`)
    .join(" · ");
}

function prettyJsonText(value: unknown): string {
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  return JSON.stringify(value ?? {}, null, 2);
}

function progressLabel(data: unknown) {
  const phases = postTurnProgressPhases(data);
  if (data && typeof data === "object" && "status_text" in data) return String((data as { status_text: unknown }).status_text);
  if (phases) return `Post-turn: ${phases.map((phase) => `${postTurnPhaseLabel(phase.name)} ${postTurnPhaseStatusLabel(phase.status)}`).join(", ")}`;
  if (data && typeof data === "object" && "section_id" in data && "status" in data) {
    const progress = data as { section_id: unknown; status: unknown; completed_count?: unknown; total_count?: unknown };
    const count = progress.completed_count !== undefined && progress.total_count !== undefined ? ` ${progress.completed_count}/${progress.total_count}` : "";
    return `${labelize(String(progress.section_id))}: ${String(progress.status)}${count}`;
  }
  if (data && typeof data === "object" && "label" in data) return String((data as { label: unknown }).label);
  if (data && typeof data === "object" && "status" in data) return String((data as { status: unknown }).status);
  return "Working";
}

function postTurnProgressPhases(data: unknown): TrackedJobPhase[] | undefined {
  if (!data || typeof data !== "object" || !("jobs" in data)) return undefined;
  const jobs = (data as { jobs?: unknown }).jobs;
  if (!Array.isArray(jobs)) return undefined;
  const phases = jobs.flatMap((job) => {
    if (!job || typeof job !== "object") return [];
    const phase = job as { name?: unknown; status?: unknown };
    if (typeof phase.name !== "string" || typeof phase.status !== "string") return [];
    return [{ name: phase.name, status: phase.status }];
  });
  return phases.length ? phases : undefined;
}

function visiblePostTurnPhases(phases: TrackedJobPhase[] | undefined, includeFull: boolean): TrackedJobPhase[] {
  if (!phases) return [];
  if (includeFull) return phases;
  return phases.filter((phase) => phase.name !== "pruning");
}

function postTurnPhaseLabel(name: string): string {
  const labels: Record<string, string> = {
    submission: "Submitting",
    classification: "Content classification",
    history: "History check",
    input: "Saving input",
    character_planning: "Character planning",
    context_selection: "Context selection",
    prompt: "Prompt prep",
    narrator: "Narrator response",
    response_checks: "Response checks",
    save_narration: "Saving narration",
    action_choices: "Action choices",
    summary: "History summary",
    state: "World state",
    context: "Context update",
    proactive_text: "Proactive text",
    director: "Director pressure",
    scenario: "Scenario evolution",
    characters: "Character cleanup",
    image: "Automatic image",
    post_turn_catchup: "Prior turn continuity"
  };
  return labels[name] ?? labelize(name);
}

function postTurnPhaseStatusLabel(status: string): string {
  return status.split("_").join(" ");
}

function pendingJobsDisplayModeFromSettings(value: unknown): PendingJobsDisplayMode {
  if (value === "expanded" || value === "expanded_full") return value;
  return "compact";
}

function compactJobGroups(jobs: TrackedJob[]): { label: string; count: number }[] {
  const groups = new Map<string, { label: string; count: number }>();
  for (const tracked of jobs) {
    const existing = groups.get(tracked.job.type);
    if (existing) {
      existing.count += 1;
    } else {
      groups.set(tracked.job.type, { label: jobTypeLabel(tracked.job.type), count: 1 });
    }
  }
  return [...groups.values()];
}

function jobTypeLabel(type: string) {
  const labels: Record<string, string> = {
    chat_turn: "Chat turn",
    chat_bundle_export: "Exporting save",
    look_around: "Looking around",
    chat_regenerate: "Regenerating message",
    action_choice_regenerate: "Regenerating options",
    character_text_send: "Sending text",
    character_text_message_edit: "Editing text",
    character_text_edit: "Replaying text edit",
    character_text_delete: "Deleting texts",
    chat_edit: "Replaying edit",
    message_edit: "Editing message",
    narrator_edit: "Editing narrator message",
    chat_delete_from_here: "Deleting messages",
    chat_fork_from_here: "Forking save",
    image_generation: "Generating image",
    character_image_generation: "Generating character image",
    initial_image_generation: "Generating opening image",
    image_regeneration: "Regenerating image",
    image_animation: "Animating image",
    character_reference_image: "Generating character reference image",
    character_reference_upload: "Uploading reference image",
    character_reference_set: "Setting reference image",
    media_delete: "Deleting media",
    context_cleanup: "Cleaning context",
    guided_context_cleanup: "Guided cleanup",
    summary_backfill: "Compacting history",
    world_suggestion_review: "Reviewing suggestions",
    world_context_retention: "Pruning world history",
    state_pruning: "Cleaning world state",
    model_refresh: "Refreshing models",
    scenario_draft: "Drafting scenario",
    scenario_section: "Regenerating section",
    scenario_character_starters: "Generating character starters"
  };
  return labels[type] ?? labelize(type);
}

const POST_TURN_COMPLETION_LABELS = {
  response_committed: "Response ready",
  continuity_ready: "Continuity ready",
  optional_enrichments_complete: "Optional complete"
} as const;

function postTurnCompletionLabel(job: Job): string | null {
  if (job.type !== "post_turn_background" || !job.completion_level) return null;
  return POST_TURN_COMPLETION_LABELS[job.completion_level];
}

function isCompletionLevelEvent(data: unknown): data is { completion_level: NonNullable<Job["completion_level"]> } {
  if (!data || typeof data !== "object") return false;
  const level = (data as { completion_level?: unknown }).completion_level;
  return typeof level === "string" && level in POST_TURN_COMPLETION_LABELS;
}

function isChatJobType(type: string) {
  return type === "chat_turn"
    || type === "look_around"
    || type === "chat_regenerate"
    || type === "chat_edit"
    || type === "message_edit"
    || type === "narrator_edit"
    || type === "chat_delete_from_here"
    || type === "chat_fork_from_here";
}

function mediaChangingJob(job: Pick<Job, "type">) {
  return /(image|media|video|character_reference)/.test(job.type);
}

function sceneArrivalSourceMessageId(job: Job): string | null {
  const sourceMessageId = (
    job.latest_progress as { source_message_id?: unknown } | null
  )?.source_message_id;
  return job.type === "automatic_image_generation"
    && typeof sourceMessageId === "string" ? sourceMessageId : null;
}

function jobBlocksChatSubmission(job: Job, activeSaveId: string | null) {
  return Boolean(activeSaveId) && job.save_id === activeSaveId && isChatJobType(job.type);
}

function scenarioSectionResultText(result: unknown, sectionId: string): string | null {
  if (typeof result === "string") return result;
  if (!result || typeof result !== "object") return null;
  const direct = result as Record<string, unknown>;
  if (typeof direct.text === "string") return direct.text;
  if (typeof direct.value === "string") return direct.value;
  const draft = direct.scenario_draft;
  if (!draft || typeof draft !== "object") return null;
  const sections = (draft as { sections?: unknown }).sections;
  if (Array.isArray(sections)) {
    for (const entry of sections) {
      if (Array.isArray(entry) && entry[0] === sectionId && typeof entry[1] === "string") return entry[1];
    }
  }
  return null;
}

function runtimeResultError(result: unknown): string | null {
  if (!result || typeof result !== "object") return null;
  const payload = result as Record<string, unknown>;
  if (typeof payload.error === "string" && payload.error) return payload.error;
  if (typeof payload.failure_text === "string" && payload.failure_text) return payload.failure_text;
  return null;
}

function labelize(value: string) {
  return value
    .split("_")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

function scenarioTypeLabel(value: string) {
  return SCENARIO_TYPE_LABELS[value] ?? labelize(value);
}

function scenarioTypesLabel(values: string[] | undefined, fallback: string) {
  return normalizedScenarioTypes(fallback, values).map(scenarioTypeLabel).join(" / ");
}

function worldTabLabel(tab: WorldDataTab): string {
  if (tab === "suggestion_groups") return "Suggestions";
  if (tab === "scene") return "Current Scene";
  return labelize(tab);
}

const WORLD_TAB_TOOLTIPS: Record<WorldDataTab, string> = {
  scenario: "Scenario definition and opening message authoring.",
  scene: "The current scene snapshot: situation, location, and time-of-day markers.",
  world_state: "Structured facts the chronicle treats as durable world state.",
  memories: "Memories distilled from earlier narrator turns.",
  context_inputs: "Read-only inputs Bragi used to compose the latest narrator turn.",
  summaries: "Compressed summaries of older chronicle context.",
  locations: "Locations in the world and their connections.",
  characters: "Character records in the active save.",
  threads: "Open plot threads, relationships, and ongoing quests.",
  links: "Relations between world entities (entity -> target).",
  suggestion_groups: "Pending world-data suggestions queued for review.",
  audit: "Audit trail of accepted, rejected, and superseded world-data edits."
};
function worldTabTooltip(tab: WorldDataTab): string {
  return WORLD_TAB_TOOLTIPS[tab] ?? worldTabLabel(tab);
}

function settingTooltip(settingKey: string): string {
  return SETTING_TOOLTIPS[settingKey] ?? "Changes how this setting affects Bragi behavior.";
}

export function modelSelectorPurpose(task: string): string {
  if (task.startsWith("scenario_generation_section_")) return "scenario_generation";
  if (
    task === "chat_full_roleplay" ||
    task === "chat_fantasy_roleplay" ||
    task === "chat_science_fiction_roleplay" ||
    task === "chat_first_contact_exploration" ||
    task === "chat_survival_expedition" ||
    task === "chat_time_loop" ||
    task === "chat_investigation_mystery" ||
    task === "chat_heist_infiltration" ||
    task === "chat_political_intrigue" ||
    task === "chat_dating_sim"
  ) return "chat";
  for (const prefix of ["full_roleplay_", "fantasy_roleplay_", "science_fiction_roleplay_", "first_contact_exploration_", "survival_expedition_", "time_loop_", "investigation_mystery_", "heist_infiltration_", "political_intrigue_", "dating_sim_", "shared_roleplay_"]) {
    if (task.startsWith(prefix)) return task.slice(prefix.length);
  }
  return task;
}

function taskModelTooltip(task: string): string {
  const purpose = modelSelectorPurpose(task);
  return TASK_MODEL_TOOLTIPS[task] ?? TASK_MODEL_TOOLTIPS[purpose] ?? `Sets the model Bragi uses for ${taskLabel(task)}.`;
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "None";
  if (Array.isArray(value)) return `${value.length} items`;
  if (typeof value === "object") return itemTitle(value, 0);
  return String(value);
}

function itemTitle(item: unknown, index: number): string {
  if (item && typeof item === "object") {
    const row = item as Record<string, unknown>;
    return String(row.title ?? row.name ?? row.label ?? row.id ?? row.character_id ?? row.location_id ?? `Item ${index + 1}`);
  }
  return String(item);
}

function itemSubtitle(item: unknown): string {
  if (item && typeof item === "object") {
    const row = item as Record<string, unknown>;
    return String(row.status ?? row.type ?? row.kind ?? row.updated_at ?? "");
  }
  return "";
}

function isRuntimeModel(value: unknown): value is RuntimeModel {
  return Boolean(value && typeof value === "object" && "chronicle" in value && "saves" in value);
}

function isChatTurnDelta(value: unknown): value is ChatTurnDelta {
  if (!value || typeof value !== "object") return false;
  const delta = value as Partial<ChatTurnDelta>;
  return (
    delta.kind === "chat_turn_delta"
    && delta.version === 1
    && typeof delta.save_id === "string"
    && Array.isArray(delta.messages)
  );
}

function isNarratorDraft(value: unknown): value is NarratorDraft {
  if (!value || typeof value !== "object") return false;
  const draft = value as Partial<NarratorDraft>;
  return (
    draft.kind === "narrator_draft"
    && draft.version === 1
    && typeof draft.save_id === "string"
    && typeof draft.draft === "string"
  );
}

export function applyChatTurnDeltaToRuntimeModel(model: RuntimeModel, delta: ChatTurnDelta): RuntimeModel {
  const replacements = new Map(delta.messages.map((message) => [message.message_id, message]));
  const seen = new Set<string>();
  const mergedMessages = model.chronicle.messages.map((message) => {
    seen.add(message.message_id);
    return replacements.get(message.message_id) ?? message;
  });
  for (const message of delta.messages) {
    if (!seen.has(message.message_id)) {
      mergedMessages.push(message);
      seen.add(message.message_id);
    }
  }
  const saves = delta.save
    ? [
        delta.save,
        ...model.saves.filter((save) => save.save_id !== delta.save?.save_id)
      ]
    : model.saves;
  return {
    ...model,
    active_save_id: delta.save_id,
    active_save_title: delta.save?.title ?? model.active_save_title,
    chronicle: {
      ...model.chronicle,
      messages: mergedMessages
    },
    action_choices: delta.action_choices,
    saves,
    status: delta.status,
    error: delta.error,
    continuity_degraded: delta.continuity_degraded ?? model.continuity_degraded,
    retry_pending: delta.retry_pending ?? model.retry_pending
  };
}

function applyCharacterTextJobResult(client: QueryClient, result: unknown, fallbackSaveId: string | null): boolean {
  if (!isCharacterTextThreadJobResult(result)) return false;
  const saveId = typeof result.save_id === "string" ? result.save_id : fallbackSaveId;
  const thread = result.thread;
  client.setQueryData(["character-text-thread", saveId, thread.id], thread);
  client.setQueryData<CharacterTextsModel>(["character-texts", saveId], (current) => (
    current ? characterTextsModelWithUpdatedThread(current, thread) : current
  ));
  return true;
}

function isCharacterTextThreadJobResult(value: unknown): value is { save_id?: string; thread: CharacterTextThread } {
  if (!value || typeof value !== "object" || !("thread" in value)) return false;
  const thread = (value as { thread?: unknown }).thread;
  return Boolean(
    thread &&
    typeof thread === "object" &&
    typeof (thread as Partial<CharacterTextThread>).id === "string" &&
    Array.isArray((thread as Partial<CharacterTextThread>).messages)
  );
}

function characterTextsModelWithUpdatedThread(model: CharacterTextsModel, thread: CharacterTextThread): CharacterTextsModel {
  const latest = thread.messages[thread.messages.length - 1] ?? null;
  const updateContact = (contact: CharacterTextContact): CharacterTextContact => {
    if (contact.thread_id !== thread.id) return contact;
    return {
      ...contact,
      latest_message_id: latest?.id ?? null,
      latest_message_body: latest?.body ?? "",
      latest_message_markdown_blocks: latest?.markdown_blocks ?? [],
      latest_message_sender: latest?.sender ?? null,
      latest_message_at: latest?.created_at ?? null,
      latest_message_read_at: latest?.read_at ?? null
    };
  };
  return {
    ...model,
    threads: model.threads.some((existing) => existing.id === thread.id)
      ? model.threads.map((existing) => existing.id === thread.id ? thread : existing)
      : [thread, ...model.threads],
    contacts: model.contacts.map(updateContact),
    repair_contacts: model.repair_contacts.map(updateContact)
  };
}

function isCharacterRegistryModel(value: unknown): value is CharacterRegistryModel {
  return Boolean(value && typeof value === "object" && "active_save_id" in value && "characters" in value);
}

function scenarioFlow(model: RuntimeModel | undefined, scenarioType: string): ScenarioWizardFlow | undefined {
  return (model?.scenario_wizard?.flows ?? defaultFlows()).find((flow) => flow.flow_id === scenarioType);
}

function scenarioCreationFlow(model: RuntimeModel | undefined, scenarioTypes: string[]): ScenarioWizardFlow | undefined {
  const flows = model?.scenario_wizard?.flows ?? defaultFlows();
  const selectedFlows = scenarioTypes
    .map((scenarioType) => flows.find((flow) => flow.flow_id === scenarioType))
    .filter((flow): flow is ScenarioWizardFlow => Boolean(flow));
  if (selectedFlows.length <= 1) return selectedFlows[0] ?? scenarioFlow(model, scenarioTypes[0] ?? "full_roleplay");
  return {
    flow_id: selectedFlows.map((flow) => flow.flow_id).join("+"),
    label: selectedFlows.map((flow) => flow.label).join(" / "),
    seed_prompt: `Describe a ${selectedFlows.map((flow) => flow.label).join(" / ")} hybrid scenario with all required setup, genre-specific details, tone, and visible opening narration.`,
    editable_section_ids: mergedScenarioSectionIds(selectedFlows.map((flow) => flow.editable_section_ids)),
    review_groups: mergedScenarioReviewGroups(selectedFlows)
  };
}

function mergedScenarioSectionIds(sectionGroups: string[][]): string[] {
  const openingIds = ["tone_genre", "choice_style", "opening_message"];
  const body: string[] = [];
  const opening: string[] = [];
  const seen = new Set<string>();
  sectionGroups.forEach((sectionIds) => {
    sectionIds.forEach((sectionId) => {
      if (seen.has(sectionId)) return;
      seen.add(sectionId);
      if (openingIds.includes(sectionId)) opening.push(sectionId);
      else body.push(sectionId);
    });
  });
  return [...body, ...openingIds.filter((sectionId) => opening.includes(sectionId))];
}

function mergedScenarioReviewGroups(flows: ScenarioWizardFlow[]): { label: string; section_ids: string[] }[] {
  const groups: { label: string; section_ids: string[] }[] = [];
  const seenSections = new Set<string>();
  flows.forEach((flow) => {
    flow.review_groups.forEach((group) => {
      const sectionIds = group.section_ids.filter((sectionId) => !seenSections.has(sectionId));
      sectionIds.forEach((sectionId) => seenSections.add(sectionId));
      if (!sectionIds.length) return;
      const label = group.label === "Core" || group.label === "Opening"
        ? group.label
        : `${flow.label}: ${group.label}`;
      const existing = groups.find((item) => item.label === label);
      if (existing) existing.section_ids.push(...sectionIds);
      else groups.push({ label, section_ids: sectionIds });
    });
  });
  return groups;
}

function normalizedScenarioTypes(primary: string, scenarioTypes: string[] | undefined): string[] {
  const selected = [primary, ...(scenarioTypes ?? []).filter((scenarioType) => scenarioType !== primary)];
  return [...new Set(selected)].slice(0, 2);
}

function defaultSecondaryScenarioType(flows: ScenarioWizardFlow[], primary: string): string {
  return flows.find((flow) => flow.flow_id !== primary)?.flow_id ?? primary;
}

function defaultFlows(): ScenarioWizardFlow[] {
  return [
    {
      flow_id: "full_roleplay",
      label: "Generic Roleplay",
      seed_prompt: "Describe the genre, premise, player role, tone, and visible opening narration. Leave room for the world to emerge in play.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "fantasy_roleplay",
      label: "Fantasy",
      seed_prompt: "Describe the fantasy premise, player role, magic, realms, factions, myths or creatures, quest stakes, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "magic_system", "realms_and_places", "factions_and_orders", "myths_and_creatures", "quest_stakes", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Fantasy World", section_ids: ["magic_system", "realms_and_places", "factions_and_orders", "myths_and_creatures", "quest_stakes"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "science_fiction_roleplay",
      label: "Science Fiction",
      seed_prompt: "Describe the science fiction premise, player role, technology, setting scope, species or intelligences, factions or institutions, mission stakes, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "technology_level", "setting_scope", "species_and_intelligences", "factions_and_institutions", "mission_stakes", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Science Fiction World", section_ids: ["technology_level", "setting_scope", "species_and_intelligences", "factions_and_institutions", "mission_stakes"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "first_contact_exploration",
      label: "First Contact / Exploration",
      seed_prompt: "Describe the first contact or exploration mission, unknown world or anomaly, ship/base status, alien or ambiguous intelligence, translation progress, discoveries, hazards, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "mission_profile", "ship_or_base_status", "exploration_target", "unknown_intelligence", "knowledge_state", "translation_progress", "discoveries_and_samples", "hazards_and_escalation", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Mission", section_ids: ["mission_profile", "ship_or_base_status"] },
        { label: "Discovery", section_ids: ["exploration_target", "knowledge_state", "discoveries_and_samples", "hazards_and_escalation"] },
        { label: "Contact", section_ids: ["unknown_intelligence", "translation_progress"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "survival_expedition",
      label: "Survival Expedition",
      seed_prompt: "Describe the survival expedition premise, player role, goal, route options, supplies, environmental conditions, hazards, camp status, travel progress, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "expedition_goal", "route_options", "resource_inventory", "environmental_conditions", "hazards_and_events", "camp_status", "travel_progress", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Expedition", section_ids: ["expedition_goal", "route_options", "travel_progress"] },
        { label: "Supplies", section_ids: ["resource_inventory"] },
        { label: "Conditions", section_ids: ["environmental_conditions", "hazards_and_events", "camp_status"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "time_loop",
      label: "Time Loop",
      seed_prompt: "Describe the time loop premise, reset trigger, loop duration, starting state, objective, failure conditions, baseline world state, schedule, persistent knowledge, persistence exceptions, NPC memory rules, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "loop_premise", "reset_trigger", "loop_duration", "starting_state", "objective", "failure_conditions", "baseline_world_state", "loop_schedule", "persistent_knowledge", "persistence_exceptions", "npc_memory_rules", "current_loop_state", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Loop Rules", section_ids: ["loop_premise", "reset_trigger", "loop_duration", "objective", "failure_conditions"] },
        { label: "Reset State", section_ids: ["starting_state", "baseline_world_state"] },
        { label: "Schedule", section_ids: ["loop_schedule", "current_loop_state"] },
        { label: "Persistence", section_ids: ["persistent_knowledge", "persistence_exceptions", "npc_memory_rules"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "investigation_mystery",
      label: "Investigation Mystery",
      seed_prompt: "Describe the mystery premise, case facts, clues, timeline, red herrings, hidden truth, case status, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "case_facts", "clues", "timeline", "red_herrings", "hidden_truth", "case_status", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Case", section_ids: ["case_facts", "case_status"] },
        { label: "Evidence", section_ids: ["clues", "timeline", "red_herrings", "hidden_truth"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "heist_infiltration",
      label: "Heist / Infiltration",
      seed_prompt: "Describe the heist or infiltration target, objectives, intel, access, security model, alert or heat state, loadout, complications, extraction, aftermath, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "target_location", "objectives_and_stakes", "intel_and_access", "security_model", "alert_and_heat", "loadout_and_tools", "complications", "extraction_routes", "aftermath", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Target & Objectives", section_ids: ["target_location", "objectives_and_stakes"] },
        { label: "Intel", section_ids: ["intel_and_access"] },
        { label: "Security", section_ids: ["security_model", "alert_and_heat"] },
        { label: "Tools & Complications", section_ids: ["loadout_and_tools", "complications"] },
        { label: "Exit & Consequences", section_ids: ["extraction_routes", "aftermath"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "political_intrigue",
      label: "Political Intrigue",
      seed_prompt: "Describe the political arena, factions, central conflict, secrets, reputation, obligations, alliances, event calendar, timed political pressure, public and private knowledge, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "political_arena", "political_factions", "central_conflict", "secrets_and_leverage", "reputation_and_standing", "obligations_and_favors", "alliances_and_rivalries", "event_calendar", "political_pressure", "public_private_knowledge", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Arena", section_ids: ["political_arena", "central_conflict"] },
        { label: "Factions", section_ids: ["political_factions", "alliances_and_rivalries"] },
        { label: "Leverage", section_ids: ["secrets_and_leverage", "reputation_and_standing", "obligations_and_favors", "public_private_knowledge"] },
        { label: "Pressure", section_ids: ["event_calendar", "political_pressure"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "settlement_builder",
      label: "Settlement Builder",
      seed_prompt: "Describe the settlement premise, resources, projects, facilities, threats, opportunities, calendar pressure, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "settlement_profile", "resources_and_indicators", "projects_and_facilities", "threats_and_opportunities", "calendar_and_deadlines", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Community", section_ids: ["settlement_profile"] },
        { label: "Operations", section_ids: ["resources_and_indicators", "projects_and_facilities"] },
        { label: "Pressure", section_ids: ["threats_and_opportunities", "calendar_and_deadlines"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "monster_hunt_bounty",
      label: "Monster Hunt / Bounty",
      seed_prompt: "Describe the hunt or bounty premise, target, clues, locations, preparation state, current hunt status, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "hunt_profile", "target_profile", "leads_and_clues", "hunt_locations", "preparation_state", "hunt_status", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Hunt", section_ids: ["hunt_profile", "target_profile", "hunt_status"] },
        { label: "Investigation", section_ids: ["leads_and_clues", "hunt_locations"] },
        { label: "Pressure", section_ids: ["preparation_state"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "road_trip_pilgrimage",
      label: "Road Trip / Pilgrimage",
      seed_prompt: "Describe the journey premise, route, stops, transport, supplies, recurring pressures, relationships, progress, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "journey_profile", "route_and_stops", "transport_and_supplies", "recurring_pressures", "relationship_threads", "journey_progress", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Journey", section_ids: ["journey_profile", "route_and_stops", "journey_progress"] },
        { label: "Relationship Threads", section_ids: ["relationship_threads"] },
        { label: "Road Pressure", section_ids: ["transport_and_supplies", "recurring_pressures"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "merchant_trade_route",
      label: "Merchant / Trade Route",
      seed_prompt: "Describe the trade premise, route, cargo, markets, contracts, debts, route hazards, profit and loss pressure, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_role", "trade_profile", "cargo_inventory", "markets_and_stops", "contracts_and_debts", "route_hazards", "profit_and_loss", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_role"] },
        { label: "Trade Route", section_ids: ["trade_profile", "markets_and_stops"] },
        { label: "Cargo & Contracts", section_ids: ["cargo_inventory", "contracts_and_debts"] },
        { label: "Risk & Standing", section_ids: ["route_hazards", "profit_and_loss"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
    {
      flow_id: "dating_sim",
      label: "Dating Sim",
      seed_prompt: "Describe the player character, dating sim premise, tone, and visible opening narration.",
      editable_section_ids: ["title", "premise", "player_character_name", "player_character_profile", "player_role", "tone_genre", "opening_message"],
      review_groups: [
        { label: "Core", section_ids: ["title", "premise", "player_character_name", "player_character_profile", "player_role"] },
        { label: "Opening", section_ids: ["tone_genre", "opening_message"] }
      ],
    },
  ];
}

export { actionIcon, apiRead, App, applyCharacterTextJobResult, canUseAdminControls, canUseChildRestrictedControls, capabilityLabel, CHARACTER_AUTO_ENHANCE_FIELD_SET, CHARACTER_AUTO_ENHANCE_LABELS, CHARACTER_LOCK_FIELD_ALIASES, CHARACTER_LOCK_FIELD_IDS, CHARACTER_LOCK_FIELDS, CHARACTER_TEXT_ESTIMATED_ROW_HEIGHT, CHARACTER_TEXT_ROW_GAP, CHARACTER_TEXT_ROW_OVERSCAN, charactersPath, characterTextContactPath, characterTextSeenStorageKey, characterTextsPath, characterTextThreadPath, characterTextThreadReadPath, chatHistoryPath, Chronicle, chronicleMessages, compactInlineTitle, Composer, conciseRowPreview, ConfirmModal, CyoaActionPicker, DataViewer, defaultFlows, defaultSecondaryScenarioType, DIAGNOSTICS_STALE_MS, diagnosticsBundleFilename, diagnosticsQueryOptions, DialogForm, DialogPanel, EditorDirtyStatus, EMPTY_DIAGNOSTICS_FILTERS, EmptyState, fallbackTaskLabels, focusableElements, formatUsd, hasAction, imageDimensionPresetLabel, imageStylePresetLabel, incomingCharacterTextContacts, initialMediaDraftLabel, initialVirtualBottomOffset, InlineNotice, installGlobalErrorLogging, invalidateScenePresenceQueries, isCharacterRegistryModel, isRuntimeModel, jobDiagnosticsPath, jobStepsPath, jobTypeLabel, labelize, LeftRail, MANUAL_BASE_SECTION_IDS, MANUAL_SCENARIO_TEXTAREA_FIELDS, MarkdownView, matchingWorldRows, mediaAssetPath, mediaAssetPromptPath, mediaAssetThumbnailPath, mediaPath, mergeChroniclePage, ModalBackdrop, MODEL_CAPABILITY_ALIASES, MODEL_FALLBACK_LANES, MODEL_ROUTING_GROUPS, modelOptionLabel, modelOptionSelectLabel, modelPricingCompactLabel, modelPricingDisplayLabel, ModelPricingLine, mountApp, normalizedScenarioTypes, npcKnowledgeAuditModeLabel, observeVirtualElementOffset, observeVirtualElementRect, openDownloadInNewTab, OPENROUTER_MAX_PRICE_TOOLTIPS, OPENROUTER_QUANTIZATION_TOOLTIPS, OPENROUTER_ROUTING_TOOLTIPS, PanelHeader, pendingJobsDisplayModeLabel, PendingJobsTray, postTurnInferenceModeLabel, PreviewModal, progressLabel, queryClient as workbenchQueryClient, runtimeQueryKey, runtimeResultError, SAVE_SCOPED_SETTING_KEYS, ScenarioBundleUpload, scenarioCreationFlow, scenarioEditorStarters, STARTER_INPUT_FIELDS, scenarioSectionEditorGroups, scenarioSectionResultText, STARTER_TEXTAREA_FIELDS, scenarioStarterPayload, scriptGuardModeLabel, SegmentedTabs, selectedOption, setScrollTopAndNotify, settingLabel, SETTINGS_TAB_TOOLTIPS, settingTooltip, taskLabel, taskModelTooltip, terminalJobsPath, THINKING_LEVEL_OFF, THINKING_LEVEL_PROVIDER_DEFAULT, thinkingLevelLabel, touchActionClassName, TouchActionContents, trackedActiveJob, useDialogFocus, useDialogJobWatcher, useJobActionRunner, useMediaQuery, VIRTUAL_LIST_INITIAL_RECT, virtualElementRect, Workbench, WORKBENCH_MOBILE_QUERY, WorldDataExplorer, worldDataPath, worldRows, worldRowSubtitle, worldRowTitle };
export type { CharacterEditorTab, CharacterTextSendVariables, CharacterTextSpontaneousVariables, CurrentUser, DiagnosticsFilters, LocalCharacterTextMessage, ModelCapabilityFamily, ModelRoutingLane, ModelRoutingLaneGroup, ModelRoutingLaneGroupMeta, ModelRoutingLaneMeta, ModelSelectorGroup, RunJob, ScenarioDraftPrefill, ScenarioEditorStarter, ScenarioForm, ScenarioFormTextField, SegmentOption, TerminalJobStatusFilter };
