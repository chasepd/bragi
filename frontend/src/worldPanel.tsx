import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  BookOpen,
  Check,
  ChevronDown,
  Clock,
  Download,
  Edit3,
  Eye,
  FileText,
  FileWarning,
  GitBranch,
  Image,
  Info,
  Loader2,
  MessageSquareText,
  PanelRight,
  Play,
  Plus,
  Save,
  Search,
  Trash2,
  Upload,
  Users,
  Wand2,
  X
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import {
  api,
  deleteJson,
  postJson
} from "./api";
import type {
  AdminUser,
  BundlePreview,
  CharacterBundlePreview,
  CharacterEnhanceField,
  CharacterFieldEnhanceResult,
  CharacterKnowledgeApplyResult,
  CharacterKnowledgeTarget,
  CharacterReferenceImage,
  CharacterRegistryApplyResult,
  CharacterRegistryModel,
  CharacterRow,
  ChatHistoryModel,
  DiagnosticEntry,
  DiagnosticsModel,
  EngineHealthModel,
  EngineHealthWarning,
  Job,
  JobStepsModel,
  JobStepSummary,
  MaintenanceJobDiagnostic,
  ModelOption,
  OpenRouterProviderCatalogEntry,
  OpenRouterRoutingProfile,
  OpenRouterRoutingSettings as OpenRouterRoutingSettingsModel,
  OpenRouterRoutingTaskOverride,
  ProviderCard,
  RuntimeModel,
  RuntimePerformanceReport,
  RuntimePerformanceRow,
  RuntimeSlowOperation,
  SchedulerHealthReport,
  SchedulerHealthTask,
  SettingsModel,
  TaskModelSelector,
  TerminalJobSummary,
  ThinkingLevelControl,
  ToggleControl,
  WebEventEntry,
  WorldDataApplyResult,
  WorldDataModel,
  WorldDataScenario
} from "./api";
import {
  actionIcon,
  apiRead,
  canUseAdminControls,
  canUseChildRestrictedControls,
  conciseRowPreview,
  ConfirmModal,
  DataViewer,
  DialogForm,
  DialogPanel,
  EmptyState,
  formatUsd,
  imageDimensionPresetLabel,
  imageStylePresetLabel,
  InlineNotice,
  labelize,
  MarkdownView,
  mediaAssetPath,
  mediaAssetThumbnailPath,
  ModalBackdrop,
  ModelPricingLine,
  modelOptionLabel,
  modelOptionSelectLabel,
  modelPricingCompactLabel,
  PanelHeader,
  PreviewModal,
  runtimeQueryKey,
  SAVE_SCOPED_SETTING_KEYS,
  SegmentedTabs,
  selectedOption,
  settingsPath,
  settingsQueryOptions,
  settingLabel,
  settingTooltip,
  taskLabel,
  taskModelTooltip,
  touchActionClassName,
  TouchActionContents,
  useDialogFocus,
  useJobActionRunner,
  useMediaQuery,
  worldDataPath,
  WorldDataExplorer,
  worldRows,
  worldRowTitle,
  worldRowSubtitle,
} from "./workbenchCore";
import type { CurrentUser, RunJob } from "./workbenchCore";

type WorldDataRow = Record<string, unknown>;
type WorldDataTab = "scenario" | "scene" | "world_state" | "memories" | "context_inputs" | "summaries" | "locations" | "characters" | "threads" | "links" | "suggestion_groups" | "audit";
type WorldDataEditTab = WorldDataTab;

const FIRST_CONTACT_BOARD_ROWS: readonly { key: string; label: string }[] = [
  { key: "contact.mission", label: "Mission" },
  { key: "contact.crew", label: "Crew" },
  { key: "contact.base", label: "Ship / Base" },
  { key: "contact.target", label: "Target" },
  { key: "contact.intelligence", label: "Intelligence" },
  { key: "contact.knowledge", label: "Knowledge" },
  { key: "contact.translation", label: "Translation" },
  { key: "contact.discoveries", label: "Discoveries" },
  { key: "contact.hazards", label: "Hazards" }
];

const FIRST_CONTACT_BOARD_PREFIXES = [
  "contact.",
  "mission.",
  "ship.",
  "base.",
  "crew.",
  "site.",
  "translation.",
  "discovery.",
  "sample.",
  "escalation."
] as const;

const FIRST_CONTACT_OBSERVATION_ACTIONS: readonly {
  label: string;
  query: string;
  icon: LucideIcon;
}[] = [
  {
    label: "Scan target",
    icon: Search,
    query: "Scan the current exploration target, focusing on observable sensor findings, hazards, and unanswered questions."
  },
  {
    label: "Analyze sample",
    icon: FileText,
    query: "Analyze the current discovery or sample, focusing on what can be inferred without advancing time or consuming resources."
  },
  {
    label: "Review translation",
    icon: MessageSquareText,
    query: "Review current translation progress, including confirmed meanings, open hypotheses, and likely misunderstandings."
  },
  {
    label: "Log hypothesis",
    icon: Plus,
    query: "Log a first-contact hypothesis from the current evidence, clearly separating observed facts from speculation."
  }
];

const INVESTIGATION_CASE_SCENARIO_ROWS: readonly { key: string; label: string }[] = [
  { key: "case_facts", label: "Case Facts" },
  { key: "case_status", label: "Case Status" }
];

const INVESTIGATION_CASE_WORLD_PREFIXES = [
  "case.",
  "case_",
  "clue.",
  "clue_",
  "evidence.",
  "evidence_",
  "suspect.",
  "suspect_",
  "timeline.",
  "timeline_"
] as const;

const INVESTIGATION_CASE_WORLD_CATEGORIES = new Set([
  "case",
  "clue",
  "evidence",
  "investigation",
  "suspect",
  "timeline"
]);

const INVESTIGATION_CASE_BLOCKED_TOKENS = [
  "hidden_truth",
  "hidden truth",
  "hidden.",
  "hidden_",
  "private.",
  "private_",
  "red_herring",
  "red herring",
  "secret.",
  "secret_"
] as const;

const INVESTIGATION_CASE_HIDDEN_MARKERS = [
  "gm-only",
  "hidden",
  "not known",
  "private",
  "secret",
  "undiscovered",
  "unrevealed"
] as const;

const INVESTIGATION_CASE_PUBLIC_MARKERS = [
  "discovered",
  "known to player",
  "player",
  "public",
  "revealed",
  "visible"
] as const;

const INVESTIGATION_CASE_VISIBILITY_FIELDS = new Set([
  "access",
  "audience",
  "discovered",
  "discovery_status",
  "known_to",
  "known_to_player",
  "player_visible",
  "public",
  "reveal_state",
  "revealed",
  "status",
  "visibility"
]);

const INVESTIGATION_CASE_OBSERVATION_ACTIONS: readonly {
  label: string;
  query: string;
  icon: LucideIcon;
}[] = [
  {
    label: "Review case file",
    icon: BookOpen,
    query: "Review the public case file, focusing on known facts, current case status, and open questions without revealing hidden truth."
  },
  {
    label: "Examine evidence",
    icon: Search,
    query: "Examine the evidence currently available to the player, separating confirmed clues from assumptions and keeping hidden solution material unrevealed."
  },
  {
    label: "Check suspects",
    icon: Users,
    query: "Check the currently known suspects, witnesses, and persons of interest, focusing on public alibis, motives, contradictions, and unanswered questions."
  }
];

const HEIST_BOARD_ROWS: readonly { key: string; label: string }[] = [
  { key: "heist.target", label: "Target" },
  { key: "heist.objectives", label: "Objectives" },
  { key: "heist.crew", label: "Crew" },
  { key: "heist.intel", label: "Intel / Access" },
  { key: "heist.security", label: "Security" },
  { key: "heist.alert", label: "Alert / Heat" },
  { key: "heist.loadout", label: "Loadout" },
  { key: "heist.complications", label: "Complications" },
  { key: "heist.extraction", label: "Extraction" },
  { key: "heist.aftermath", label: "Aftermath" }
];

const HEIST_OBSERVATION_ACTIONS: readonly {
  label: string;
  query: string;
  icon: LucideIcon;
}[] = [
  {
    label: "Case area",
    icon: Search,
    query: "Case the current heist area, focusing on visible access points, patrol movement, cover, and immediate risks without advancing time."
  },
  {
    label: "Review intel",
    icon: BookOpen,
    query: "Review current heist intel and access, separating confirmed facts from assumptions and noting unresolved questions."
  },
  {
    label: "Check security",
    icon: FileWarning,
    query: "Review the current heist security model, alert state, heat, alarms, and response posture from known information."
  },
  {
    label: "Plan extraction",
    icon: GitBranch,
    query: "Review extraction options for the current heist, focusing on viable exits, fallback routes, complications, and consequences."
  }
];

const HEIST_ALERT_FIELDS = ["summary", "level", "alarm", "heat", "response"] as const;

const SURVIVAL_EXPEDITION_BOARD_ROWS: readonly { key: string; label: string }[] = [
  { key: "expedition.goal", label: "Goal" },
  { key: "expedition.route", label: "Route" },
  { key: "expedition.party", label: "Party" },
  { key: "expedition.resources", label: "Resources" },
  { key: "expedition.environment", label: "Environment" },
  { key: "expedition.hazards", label: "Hazards" },
  { key: "expedition.camp", label: "Camp" },
  { key: "expedition.progress", label: "Progress" }
];

const SURVIVAL_EXPEDITION_PRESSURE_KEYS = new Set([
  "expedition.resources",
  "expedition.camp",
  "expedition.progress"
]);

const SURVIVAL_EXPEDITION_ACTIONS: readonly {
  label: string;
  query: string;
  icon: LucideIcon;
}[] = [
  {
    label: "Travel leg",
    icon: Play,
    query: "Travel the next expedition leg, advancing route pressure, time, supplies, weather, hazards, and party condition from the current expedition state."
  },
  {
    label: "Make camp",
    icon: Archive,
    query: "Make camp for the survival expedition, focusing on shelter, rest, watches, weather exposure, hazards, and what changes in camp status."
  },
  {
    label: "Ration supplies",
    icon: FileWarning,
    query: "Review and ration expedition supplies, focusing on food, water, medicine, gear condition, scarcity, and consequences for the party."
  },
  {
    label: "Choose route",
    icon: GitBranch,
    query: "Choose the next survival expedition route option, comparing travel time, resource cost, weather exposure, hazards, and party risk."
  }
];

type ManagementBoardAction = string;

type ManagementBoardConfig = {
  scenarioType: string;
  title: string;
  extraPrefixes: readonly string[];
  actions: readonly ManagementBoardAction[];
  stateIsVisible?: (row: WorldDataRow) => boolean;
};

const MANAGEMENT_BOARD_CONFIGS: readonly ManagementBoardConfig[] = [
  {
    scenarioType: "settlement_builder",
    title: "Settlement Board",
    extraPrefixes: ["settlement.", "project.", "resource."],
    actions: [
      "Allocate resources"
    ]
  },
  {
    scenarioType: "time_loop",
    title: "Loop Clock",
    extraPrefixes: ["loop."],
    actions: [
      "Review remembered facts",
      "Advance to known event",
      "Review reset rules"
    ]
  },
  {
    scenarioType: "political_intrigue",
    title: "Political Ledger",
    extraPrefixes: ["intrigue.", "faction.", "obligation.", "alliance."],
    actions: [
      "Call in favor",
      "Review leverage",
      "Track standing",
      "Check calendar"
    ],
    stateIsVisible: politicalIntrigueStateIsVisible
  },
  {
    scenarioType: "monster_hunt_bounty",
    title: "Hunt Board",
    extraPrefixes: ["hunt.", "target.", "clue."],
    actions: [
      "Review leads",
      "Track target",
      "Prepare gear",
      "Check rival pressure"
    ]
  },
  {
    scenarioType: "road_trip_pilgrimage",
    title: "Journey Board",
    extraPrefixes: ["journey.", "stop.", "companion.", "vehicle."],
    actions: [
      "Choose next stop",
      "Travel leg",
      "Check transport",
      "Review companion threads"
    ]
  },
  {
    scenarioType: "merchant_trade_route",
    title: "Trade Ledger",
    extraPrefixes: ["trade.", "cargo.", "contract.", "debt.", "market."],
    actions: [
      "Review cargo",
      "Settle contract",
      "Record debt",
      "Check market",
      "Plot route"
    ]
  }
];

const WORLD_PANEL_JOB_ACTIONS: readonly {
  key: string;
  path: string;
  label: string;
  fallbackError: string;
  icon: LucideIcon;
}[] = [
  {
    key: "world:suggestion-review",
    path: "/api/world-data/suggestion-review",
    label: "Review suggestions",
    fallbackError: "Could not start suggestion review",
    icon: Check
  },
  {
    key: "world:context-retention",
    path: "/api/world-data/context-retention",
    label: "Run retention",
    fallbackError: "Could not start context retention",
    icon: Clock
  },
  {
    key: "world:context-cleanup",
    path: "/api/world-data/context-cleanup",
    label: "Cleanup context",
    fallbackError: "Could not start context cleanup",
    icon: Wand2
  }
];

export function WorldPanel({
  model,
  runJob,
  currentUser = null,
  openLookAround
}: {
  model?: RuntimeModel;
  runJob: RunJob;
  currentUser?: CurrentUser | null;
  openLookAround?: (query: string) => void;
}) {
  const activeSaveId = model?.active_save_id ?? null;
  const world = useQuery({
    queryKey: ["world", activeSaveId],
    queryFn: () => api<WorldDataModel>(worldDataPath(activeSaveId)),
    enabled: Boolean(activeSaveId)
  });
  const client = useQueryClient();
  const [tab, setTab] = useState<WorldDataTab>("scenario");
  const [guidanceOpen, setGuidanceOpen] = useState(false);
  const [guidedCleanupOpen, setGuidedCleanupOpen] = useState(false);
  const loopActionInFlight = useRef(false);
  const worldDataMatchesActiveSave = Boolean(activeSaveId && world.data?.active_save_id === activeSaveId);
  const worldDataStale = Boolean(activeSaveId && world.data && world.data.active_save_id !== activeSaveId);
  const canMutateWorld = canUseChildRestrictedControls(currentUser);
  const {
    clearJobActionState,
    jobActionErrors,
    pendingJobActionKeys,
    startJobAction
  } = useJobActionRunner(runJob);
  useEffect(() => {
    clearJobActionState();
  }, [activeSaveId, clearJobActionState]);
  const applyWorldData = async (targetTab: WorldDataEditTab, value: unknown) => {
    if (!worldDataMatchesActiveSave || !world.data?.scenario || !activeSaveId) {
      throw new Error("World data is still loading for the active save");
    }
    const edits = targetTab === "scenario"
      ? { scenario: value }
      : { scenario: world.data.scenario, [targetTab]: value };
    const result = await postJson<WorldDataApplyResult>("/api/world-data/apply", {
      active_save_id: activeSaveId,
      edits
    });
    client.setQueryData(["world", activeSaveId], result.model);
    client.invalidateQueries({ queryKey: ["runtime"] });
    client.invalidateQueries({ queryKey: ["characters"] });
  };
  const applyLoopClockAction = async (path: string) => {
    if (!activeSaveId || !canMutateWorld || loopActionInFlight.current) return;
    loopActionInFlight.current = true;
    try {
      const result = await postJson<WorldDataModel>(path, { save_id: activeSaveId });
      client.setQueryData(["world", activeSaveId], result);
      client.invalidateQueries({ queryKey: ["runtime"] });
    } catch {
      window.alert("Could not update loop time");
    } finally {
      loopActionInFlight.current = false;
    }
  };
  return (
    <aside className="right-panel">
      <PanelHeader icon={<Archive size={18} />} title="World Data" />
      {canMutateWorld ? (
        <>
          <button className="secondary-command" onClick={() => setGuidanceOpen(true)} disabled={!model?.active_save_id}>
            <Edit3 size={15} /> Save response guidance
          </button>
          {WORLD_PANEL_JOB_ACTIONS.map((action) => {
            const Icon = action.icon;
            const pending = pendingJobActionKeys.has(action.key);
            return (
              <button
                key={action.key}
                type="button"
                className="secondary-command"
                onClick={() => {
                  void startJobAction({
                    key: action.key,
                    path: action.path,
                    body: { save_id: activeSaveId },
                    fallbackError: action.fallbackError
                  });
                }}
                disabled={!activeSaveId || pending}
              >
                {pending ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <Icon size={15} aria-hidden="true" />}
                {action.label}
              </button>
            );
          })}
          <button className="secondary-command" onClick={() => setGuidedCleanupOpen(true)} disabled={!model?.active_save_id}>
            <Wand2 size={15} /> Guided cleanup
          </button>
          {model?.active_scenario_type === "time_loop" ? (
            <>
              <button
                className="secondary-command"
                type="button"
                disabled={!activeSaveId || !world.data?.scene}
                onClick={() => {
                  void applyLoopClockAction("/api/world-data/time-loop/baseline");
                }}
              >
                <Clock size={15} /> Capture reset baseline
              </button>
              <button
                className="secondary-command"
                type="button"
                disabled={!activeSaveId || !world.data?.scene}
                onClick={() => {
                  void applyLoopClockAction("/api/world-data/time-loop/reset");
                }}
              >
                <Clock size={15} /> Reset loop
              </button>
            </>
          ) : null}
          {WORLD_PANEL_JOB_ACTIONS.map((action) => (
            jobActionErrors[action.key] ? (
              <InlineNotice key={action.key} className="world-job-action-error">
                {jobActionErrors[action.key]}
              </InlineNotice>
            ) : null
          ))}
        </>
      ) : null}
      {guidanceOpen && canMutateWorld ? (
        <ResponseGuidanceModal
          model={model}
          onClose={() => setGuidanceOpen(false)}
        />
      ) : null}
      {guidedCleanupOpen && canMutateWorld ? (
        <GuidedCleanupModal
          model={model}
          onClose={() => setGuidedCleanupOpen(false)}
          runJob={runJob}
        />
      ) : null}
      {world.data?.error ? <InlineNotice>{world.data.error}</InlineNotice> : null}
      {worldDataStale ? <InlineNotice polite>Refreshing world data for active save...</InlineNotice> : null}
      <SurvivalExpeditionBoard
        model={world.data}
        editable={worldDataMatchesActiveSave && canMutateWorld}
        openLookAround={openLookAround}
        onSavePressure={(row) => applyWorldData("world_state", [row])}
      />
      <FirstContactOperationsBoard
        model={world.data}
        openLookAround={openLookAround}
      />
      <InvestigationCaseBoard
        model={world.data}
        openLookAround={openLookAround}
      />
      <HeistInfiltrationBoard
        model={world.data}
        editable={worldDataMatchesActiveSave && canMutateWorld}
        openLookAround={openLookAround}
        onSaveAlert={(row) => applyWorldData("world_state", [row])}
      />
      <ManagementTemplateBoard
        model={world.data}
        editable={worldDataMatchesActiveSave && canMutateWorld}
        openLookAround={openLookAround}
        onSaveSummary={(row) => applyWorldData("world_state", [row])}
      />
      <WorldDataExplorer
        model={world.data}
        editable={worldDataMatchesActiveSave && canMutateWorld}
        activeTab={tab}
        setActiveTab={setTab}
        onSaveTab={applyWorldData}
      />
    </aside>
  );
}

function SurvivalExpeditionBoard({
  model,
  editable,
  openLookAround,
  onSavePressure
}: {
  model?: WorldDataModel;
  editable: boolean;
  openLookAround?: (query: string) => void;
  onSavePressure: (row: WorldDataRow) => Promise<void>;
}) {
  const [editingKey, setEditingKey] = useState<string | null>(null);
  useEffect(() => {
    setEditingKey(null);
  }, [model?.active_save_id]);
  if (model?.scenario?.scenario_type !== "survival_expedition") return null;
  const rows = survivalExpeditionBoardRows(model);
  const editingRow = rows.find((row) => row.key === editingKey) ?? null;
  const titleId = "survival-expedition-board-title";
  return (
    <section className="genre-board survival-expedition-board" aria-labelledby={titleId}>
      <div className="genre-board-header">
        <div>
          <p className="eyebrow">Expedition</p>
          <h2 id={titleId}>Expedition Dashboard</h2>
        </div>
        {rows.length ? <small>{rows.length} records</small> : null}
      </div>
      <div className="genre-board-actions" aria-label="Survival expedition actions">
        {SURVIVAL_EXPEDITION_ACTIONS.map((action) => {
          const Icon = action.icon;
          return (
            <button
              key={action.label}
              type="button"
              className="secondary-command compact"
              disabled={!openLookAround}
              onClick={() => openLookAround?.(action.query)}
            >
              <Icon size={15} aria-hidden="true" />
              {action.label}
            </button>
          );
        })}
      </div>
      {editable && editingRow ? (
        <ExpeditionPressureEditor
          key={editingRow.key}
          label={editingRow.label}
          row={editingRow.row}
          onCancel={() => setEditingKey(null)}
          onSave={async (row) => {
            await onSavePressure(row);
            setEditingKey(null);
          }}
        />
      ) : null}
      {rows.length ? (
        <div className="genre-board-grid">
          {rows.map((row) => {
            const canEditRow = editable && SURVIVAL_EXPEDITION_PRESSURE_KEYS.has(row.key);
            const editLabel = `Edit ${row.label.toLowerCase()}`;
            return (
              <article key={row.key} className="genre-board-state">
                <div className="genre-board-state-header">
                  <span>{row.label}</span>
                  {canEditRow ? (
                    <button
                      type="button"
                      className={touchActionClassName()}
                      title={editLabel}
                      aria-label={editLabel}
                      onClick={() => setEditingKey(row.key)}
                    >
                      <TouchActionContents icon={<Edit3 size={14} />} label="Edit" />
                    </button>
                  ) : null}
                </div>
                <strong>{row.summary}</strong>
                {row.category ? <small>{row.category}</small> : null}
              </article>
            );
          })}
        </div>
      ) : (
        <p className="empty">No expedition state yet.</p>
      )}
    </section>
  );
}

function ExpeditionPressureEditor({
  label,
  row,
  onCancel,
  onSave
}: {
  label: string;
  row: WorldDataRow;
  onCancel: () => void;
  onSave: (row: WorldDataRow) => Promise<void>;
}) {
  const [summary, setSummary] = useState(() => worldStateSummary(row));
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const save = async () => {
    const nextSummary = summary.trim();
    if (!nextSummary) {
      setError(`${label} summary is required`);
      return;
    }
    const nextValue = expeditionPressureValue(row);
    nextValue.summary = nextSummary;
    try {
      setSaving(true);
      setError("");
      await onSave({
        ...row,
        value_json: JSON.stringify(nextValue)
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : `Could not save ${label.toLowerCase()}`);
      setSaving(false);
    }
  };
  return (
    <form
      className="expedition-pressure-editor"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <label className="field-label">
        <span>{label} summary</span>
        <textarea
          value={summary}
          onChange={(event) => setSummary(event.target.value)}
        />
      </label>
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      <div className="command-row end">
        <button type="button" onClick={onCancel} disabled={saving}>Cancel</button>
        <button type="submit" className="primary-command compact" disabled={saving}>
          {saving ? <Loader2 className="spin" size={15} /> : <Save size={15} />}
          Save {label.toLowerCase()}
        </button>
      </div>
    </form>
  );
}

function survivalExpeditionBoardRows(model: WorldDataModel): {
  key: string;
  label: string;
  summary: string;
  category: string;
  row: WorldDataRow;
}[] {
  const stateRows = worldRows(model.world_state);
  const rowsByKey = new Map(
    stateRows
      .map(({ row }) => row)
      .filter((row) => typeof row.key === "string")
      .map((row) => [String(row.key), row] as const)
  );
  const configuredKeys = new Set(SURVIVAL_EXPEDITION_BOARD_ROWS.map((item) => item.key));
  const configuredRows = SURVIVAL_EXPEDITION_BOARD_ROWS.flatMap((item) => {
    const row = rowsByKey.get(item.key);
    if (!row) return [];
    return [survivalExpeditionBoardRow(item.key, item.label, row)];
  });
  const extraRows = stateRows
    .map(({ row }) => row)
    .filter((row) => {
      const key = typeof row.key === "string" ? row.key : "";
      return key && !configuredKeys.has(key) && key.startsWith("expedition.");
    })
    .sort((left, right) => String(left.key).localeCompare(String(right.key)))
    .map((row) => {
      const key = String(row.key);
      return survivalExpeditionBoardRow(
        key,
        labelize(key.replace(/^expedition\./, "").replace(/[.-]/g, "_")),
        row,
      );
    });
  return [...configuredRows, ...extraRows];
}

function survivalExpeditionBoardRow(
  key: string,
  label: string,
  row: WorldDataRow,
): {
  key: string;
  label: string;
  summary: string;
  category: string;
  row: WorldDataRow;
} {
  return {
    key,
    label,
    summary: worldStateSummary(row),
    category: typeof row.category === "string" ? row.category : "",
    row
  };
}

function expeditionPressureValue(row: WorldDataRow): Record<string, unknown> {
  const parsed = parsedWorldStateValue(row);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return { ...(parsed as Record<string, unknown>) };
  }
  return {};
}

function FirstContactOperationsBoard({
  model,
  openLookAround
}: {
  model?: WorldDataModel;
  openLookAround?: (query: string) => void;
}) {
  if (model?.scenario?.scenario_type !== "first_contact_exploration") return null;
  const rows = firstContactBoardRows(model);
  const titleId = "first-contact-board-title";
  return (
    <section className="genre-board first-contact-board" aria-labelledby={titleId}>
      <div className="genre-board-header">
        <div>
          <p className="eyebrow">Operations</p>
          <h2 id={titleId}>First Contact</h2>
        </div>
        {rows.length ? <small>{rows.length} records</small> : null}
      </div>
      <div className="genre-board-actions" aria-label="First-contact observations">
        {FIRST_CONTACT_OBSERVATION_ACTIONS.map((action) => {
          const Icon = action.icon;
          return (
            <button
              key={action.label}
              type="button"
              className="secondary-command compact"
              disabled={!openLookAround}
              onClick={() => openLookAround?.(action.query)}
            >
              <Icon size={15} aria-hidden="true" />
              {action.label}
            </button>
          );
        })}
      </div>
      {rows.length ? (
        <div className="genre-board-grid">
          {rows.map((row) => (
            <article key={row.key} className="genre-board-state">
              <span>{row.label}</span>
              <strong>{row.summary}</strong>
              {row.category ? <small>{row.category}</small> : null}
            </article>
          ))}
        </div>
      ) : (
        <p className="empty">No first-contact state yet.</p>
      )}
    </section>
  );
}

function firstContactBoardRows(model: WorldDataModel): {
  key: string;
  label: string;
  summary: string;
  category: string;
}[] {
  const stateRows = worldRows(model.world_state);
  const rowsByKey = new Map(
    stateRows
      .map(({ row }) => row)
      .filter((row) => typeof row.key === "string")
      .map((row) => [String(row.key), row] as const)
  );
  const configuredKeys = new Set(FIRST_CONTACT_BOARD_ROWS.map((item) => item.key));
  const configuredRows = FIRST_CONTACT_BOARD_ROWS.flatMap((item) => {
    const row = rowsByKey.get(item.key);
    if (!row) return [];
    return [firstContactBoardRow(item.key, item.label, row)];
  });
  const extraRows = stateRows
    .map(({ row }) => row)
    .filter((row) => {
      const key = typeof row.key === "string" ? row.key : "";
      return (
        key
        && !configuredKeys.has(key)
        && FIRST_CONTACT_BOARD_PREFIXES.some((prefix) => key.startsWith(prefix))
      );
    })
    .sort((left, right) => String(left.key).localeCompare(String(right.key)))
    .map((row) => {
      const key = String(row.key);
      return firstContactBoardRow(key, labelize(key.replace(/^contact\./, "")), row);
    });
  return [...configuredRows, ...extraRows];
}

function firstContactBoardRow(
  key: string,
  label: string,
  row: WorldDataRow,
): {
  key: string;
  label: string;
  summary: string;
  category: string;
} {
  return {
    key,
    label,
    summary: firstContactStateSummary(row),
    category: typeof row.category === "string" ? row.category : ""
  };
}

function firstContactStateSummary(row: WorldDataRow): string {
  return worldStateSummary(row);
}

function InvestigationCaseBoard({
  model,
  openLookAround
}: {
  model?: WorldDataModel;
  openLookAround?: (query: string) => void;
}) {
  if (model?.scenario?.scenario_type !== "investigation_mystery") return null;
  const rows = investigationCaseBoardRows(model);
  const titleId = "investigation-case-board-title";
  return (
    <section className="genre-board investigation-case-board" aria-labelledby={titleId}>
      <div className="genre-board-header">
        <div>
          <p className="eyebrow">Case</p>
          <h2 id={titleId}>Case Board</h2>
        </div>
        {rows.length ? <small>{rows.length} records</small> : null}
      </div>
      <div className="genre-board-actions" aria-label="Investigation observations">
        {INVESTIGATION_CASE_OBSERVATION_ACTIONS.map((action) => {
          const Icon = action.icon;
          return (
            <button
              key={action.label}
              type="button"
              className="secondary-command compact"
              disabled={!openLookAround}
              onClick={() => openLookAround?.(action.query)}
            >
              <Icon size={15} aria-hidden="true" />
              {action.label}
            </button>
          );
        })}
      </div>
      {rows.length ? (
        <div className="genre-board-grid">
          {rows.map((row) => (
            <article key={row.key} className="genre-board-state">
              <span>{row.label}</span>
              <strong>{row.summary}</strong>
              {row.category ? <small>{row.category}</small> : null}
            </article>
          ))}
        </div>
      ) : (
        <p className="empty">No public case state yet.</p>
      )}
    </section>
  );
}

function investigationCaseBoardRows(model: WorldDataModel): {
  key: string;
  label: string;
  summary: string;
  category: string;
}[] {
  const scenarioRows = INVESTIGATION_CASE_SCENARIO_ROWS.flatMap((item) => {
    const summary = scenarioSectionText(model.scenario, item.key);
    if (!summary) return [];
    return [{
      key: `scenario.${item.key}`,
      label: item.label,
      summary,
      category: "scenario"
    }];
  });
  const stateRows = worldRows(model.world_state)
    .map(({ row }) => row)
    .filter(investigationCaseStateIsVisible)
    .sort((left, right) => String(left.key ?? "").localeCompare(String(right.key ?? "")))
    .map((row) => {
      const key = String(row.key ?? "");
      return {
        key,
        label: investigationCaseStateLabel(row),
        summary: worldStateSummary(row),
        category: typeof row.category === "string" ? row.category : ""
      };
    })
    .filter((row) => row.summary);
  return [...scenarioRows, ...stateRows];
}

function scenarioSectionText(
  scenario: WorldDataScenario | null | undefined,
  key: string
): string {
  const section = scenario?.content_sections?.find(([sectionKey]) => sectionKey === key);
  const value = section?.[1];
  return typeof value === "string" ? value.trim() : "";
}

function investigationCaseStateIsVisible(row: WorldDataRow): boolean {
  const key = typeof row.key === "string" ? row.key.toLowerCase() : "";
  const category = typeof row.category === "string" ? row.category.toLowerCase() : "";
  const identity = `${key} ${category}`;
  if (INVESTIGATION_CASE_BLOCKED_TOKENS.some((token) => identity.includes(token))) {
    return false;
  }
  const matchesCaseState = (
    INVESTIGATION_CASE_WORLD_PREFIXES.some((prefix) => key.startsWith(prefix))
    || INVESTIGATION_CASE_WORLD_CATEGORIES.has(category)
  );
  if (!matchesCaseState) return false;
  const visibility = investigationCaseStateVisibility(row);
  if (visibility.hidden) return false;
  if (investigationCaseStateRequiresPublicMarker(key, category)) {
    return visibility.public;
  }
  return true;
}

function investigationCaseStateRequiresPublicMarker(key: string, category: string): boolean {
  return (
    key.startsWith("clue.")
    || key.startsWith("clue_")
    || key.startsWith("evidence.")
    || key.startsWith("evidence_")
    || category === "clue"
    || category === "evidence"
  );
}

function investigationCaseStateVisibility(row: WorldDataRow): {
  hidden: boolean;
  public: boolean;
} {
  const parsed = parsedWorldStateValue(row);
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { hidden: false, public: false };
  }
  const record = parsed as Record<string, unknown>;
  let hidden = false;
  let publicVisible = false;
  for (const [key, value] of Object.entries(record)) {
    const normalizedKey = key.toLowerCase();
    if (!INVESTIGATION_CASE_VISIBILITY_FIELDS.has(normalizedKey)) continue;
    if (typeof value === "boolean") {
      if (
        value
        && ["discovered", "known_to_player", "player_visible", "public", "revealed"].includes(normalizedKey)
      ) {
        publicVisible = true;
      }
      if (
        value
        && ["gm_only", "hidden", "private", "secret", "undiscovered", "unrevealed"].includes(normalizedKey)
      ) {
        hidden = true;
      }
      continue;
    }
    if (typeof value !== "string") continue;
    const normalizedValue = value.toLowerCase();
    const valueHidden = INVESTIGATION_CASE_HIDDEN_MARKERS.some((marker) => normalizedValue.includes(marker));
    if (valueHidden) {
      hidden = true;
    }
    if (!valueHidden && INVESTIGATION_CASE_PUBLIC_MARKERS.some((marker) => normalizedValue.includes(marker))) {
      publicVisible = true;
    }
  }
  return { hidden, public: publicVisible };
}

function parsedWorldStateValue(row: WorldDataRow): unknown {
  const valueJson = typeof row.value_json === "string" ? row.value_json : "";
  if (!valueJson) return null;
  try {
    return JSON.parse(valueJson) as unknown;
  } catch {
    return null;
  }
}

function investigationCaseStateLabel(row: WorldDataRow): string {
  const key = typeof row.key === "string" ? row.key : "";
  const category = typeof row.category === "string" ? row.category : "";
  const compactKey = key.replace(/^(case|clue|evidence|suspect|timeline)[._-]?/, "");
  return labelize(compactKey || category || "case note");
}

function worldStateSummary(row: WorldDataRow): string {
  const valueJson = typeof row.value_json === "string" ? row.value_json : "";
  if (!valueJson) return conciseRowPreview(row) || "Recorded";
  const parsed = parsedWorldStateValue(row);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const summary = (parsed as Record<string, unknown>).summary;
    if (typeof summary === "string" && summary.trim()) return summary.trim();
  }
  if (parsed !== null) return conciseRowPreview(parsed) || "Recorded";
  return valueJson;
}

function HeistInfiltrationBoard({
  model,
  editable,
  openLookAround,
  onSaveAlert
}: {
  model?: WorldDataModel;
  editable: boolean;
  openLookAround?: (query: string) => void;
  onSaveAlert: (row: WorldDataRow) => Promise<void>;
}) {
  const [editingHeat, setEditingHeat] = useState(false);
  useEffect(() => {
    setEditingHeat(false);
  }, [model?.active_save_id]);
  if (model?.scenario?.scenario_type !== "heist_infiltration") return null;
  const rows = heistBoardRows(model);
  const alertRow = heistAlertRow(model);
  const titleId = "heist-infiltration-board-title";
  return (
    <section className="genre-board heist-board" aria-labelledby={titleId}>
      <div className="genre-board-header">
        <div>
          <p className="eyebrow">Operation</p>
          <h2 id={titleId}>Heist Board</h2>
        </div>
        {rows.length ? <small>{rows.length} records</small> : null}
      </div>
      <div className="genre-board-actions" aria-label="Heist observations">
        {HEIST_OBSERVATION_ACTIONS.map((action) => {
          const Icon = action.icon;
          return (
            <button
              key={action.label}
              type="button"
              className="secondary-command compact"
              disabled={!openLookAround}
              onClick={() => openLookAround?.(action.query)}
            >
              <Icon size={15} aria-hidden="true" />
              {action.label}
            </button>
          );
        })}
      </div>
      {editable && alertRow && !editingHeat ? (
        <button
          type="button"
          className="secondary-command compact heist-heat-toggle"
          onClick={() => setEditingHeat(true)}
        >
          <Edit3 size={15} aria-hidden="true" />
          Adjust heat
        </button>
      ) : null}
      {editable && editingHeat && alertRow ? (
        <HeistAlertEditor
          row={alertRow}
          onCancel={() => setEditingHeat(false)}
          onSave={async (row) => {
            await onSaveAlert(row);
            setEditingHeat(false);
          }}
        />
      ) : null}
      {rows.length ? (
        <div className="genre-board-grid">
          {rows.map((row) => (
            <article key={row.key} className="genre-board-state">
              <span>{row.label}</span>
              <strong>{row.summary}</strong>
              {row.category ? <small>{row.category}</small> : null}
            </article>
          ))}
        </div>
      ) : (
        <p className="empty">No heist state yet.</p>
      )}
    </section>
  );
}

function HeistAlertEditor({
  row,
  onCancel,
  onSave
}: {
  row: WorldDataRow;
  onCancel: () => void;
  onSave: (row: WorldDataRow) => Promise<void>;
}) {
  const [draft, setDraft] = useState(() => heistAlertDraft(row));
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const update = (key: keyof HeistAlertDraft, value: string) => {
    setDraft((current) => ({ ...current, [key]: value }));
  };
  const save = async () => {
    const nextValue = heistAlertValueWithDraft(row, draft);
    if (!heistAlertDraftHasContent(draft)) {
      setError("At least one alert field is required");
      return;
    }
    try {
      setSaving(true);
      setError("");
      await onSave({
        ...row,
        value_json: JSON.stringify(nextValue)
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not save heist heat");
      setSaving(false);
    }
  };
  return (
    <form
      className="heist-alert-editor"
      onSubmit={(event) => {
        event.preventDefault();
        void save();
      }}
    >
      <label className="field-label">
        <span>Alert summary</span>
        <textarea
          value={draft.summary}
          onChange={(event) => update("summary", event.target.value)}
        />
      </label>
      <label className="field-label">
        <span>Alert level</span>
        <input value={draft.level} onChange={(event) => update("level", event.target.value)} />
      </label>
      <label className="field-label">
        <span>Alarm state</span>
        <input value={draft.alarm} onChange={(event) => update("alarm", event.target.value)} />
      </label>
      <label className="field-label">
        <span>Heat</span>
        <input value={draft.heat} onChange={(event) => update("heat", event.target.value)} />
      </label>
      <label className="field-label">
        <span>Response</span>
        <input value={draft.response} onChange={(event) => update("response", event.target.value)} />
      </label>
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      <div className="command-row end">
        <button type="button" onClick={onCancel} disabled={saving}>Cancel</button>
        <button type="submit" className="primary-command compact" disabled={saving}>
          {saving ? <Loader2 className="spin" size={15} /> : <Save size={15} />}
          Save heat
        </button>
      </div>
    </form>
  );
}

type HeistAlertDraft = {
  summary: string;
  level: string;
  alarm: string;
  heat: string;
  response: string;
};

function heistBoardRows(model: WorldDataModel): {
  key: string;
  label: string;
  summary: string;
  category: string;
}[] {
  const stateRows = worldRows(model.world_state);
  const rowsByKey = new Map(
    stateRows
      .map(({ row }) => row)
      .filter((row) => typeof row.key === "string")
      .map((row) => [String(row.key), row] as const)
  );
  const configuredKeys = new Set(HEIST_BOARD_ROWS.map((item) => item.key));
  const configuredRows = HEIST_BOARD_ROWS.flatMap((item) => {
    const row = rowsByKey.get(item.key);
    if (!row) return [];
    return [heistBoardRow(item.key, item.label, row)];
  });
  const extraRows = stateRows
    .map(({ row }) => row)
    .filter((row) => {
      const key = typeof row.key === "string" ? row.key : "";
      return key && !configuredKeys.has(key) && key.startsWith("heist.");
    })
    .sort((left, right) => String(left.key).localeCompare(String(right.key)))
    .map((row) => {
      const key = String(row.key);
      return heistBoardRow(
        key,
        labelize(key.replace(/^heist\./, "").replace(/[.-]/g, "_")),
        row,
      );
    });
  return [...configuredRows, ...extraRows];
}

function heistBoardRow(
  key: string,
  label: string,
  row: WorldDataRow,
): {
  key: string;
  label: string;
  summary: string;
  category: string;
} {
  return {
    key,
    label,
    summary: worldStateSummary(row),
    category: typeof row.category === "string" ? row.category : ""
  };
}

function heistAlertRow(model: WorldDataModel): WorldDataRow | null {
  return worldRows(model.world_state)
    .map(({ row }) => row)
    .find((row) => row.key === "heist.alert") ?? null;
}

function heistAlertDraft(row: WorldDataRow): HeistAlertDraft {
  const value = heistAlertValue(row);
  return {
    summary: stringRecordValue(value, "summary"),
    level: stringRecordValue(value, "level"),
    alarm: stringRecordValue(value, "alarm"),
    heat: stringRecordValue(value, "heat"),
    response: stringRecordValue(value, "response")
  };
}

function heistAlertValue(row: WorldDataRow): Record<string, unknown> {
  const parsed = parsedWorldStateValue(row);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return { ...(parsed as Record<string, unknown>) };
  }
  return {};
}

function heistAlertValueWithDraft(
  row: WorldDataRow,
  draft: HeistAlertDraft
): Record<string, unknown> {
  const next = heistAlertValue(row);
  for (const field of HEIST_ALERT_FIELDS) {
    const value = draft[field].trim();
    if (value) {
      next[field] = value;
    } else {
      delete next[field];
    }
  }
  return next;
}

function heistAlertDraftHasContent(draft: HeistAlertDraft): boolean {
  return HEIST_ALERT_FIELDS.some((field) => draft[field].trim());
}

function stringRecordValue(record: Record<string, unknown>, key: string): string {
  const value = record[key];
  return typeof value === "string" ? value : "";
}

function ManagementTemplateBoard({
  model,
  editable,
  openLookAround,
  onSaveSummary
}: {
  model?: WorldDataModel;
  editable: boolean;
  openLookAround?: (query: string) => void;
  onSaveSummary: (row: WorldDataRow) => Promise<void>;
}) {
  const [editingKey, setEditingKey] = useState<string | null>(null);
  const scenarioType = model?.scenario?.scenario_type ?? "";
  const config = MANAGEMENT_BOARD_CONFIGS.find((item) => item.scenarioType === scenarioType) ?? null;
  useEffect(() => {
    setEditingKey(null);
  }, [model?.active_save_id, scenarioType]);
  if (!model || !config) return null;
  const rows = managementBoardRows(model, config);
  const editingRow = rows.find((row): row is ManagementBoardRow & { row: WorldDataRow } => (
    row.key === editingKey && row.row !== null && managementBoardRowCanEdit(row.row)
  )) ?? null;
  const titleId = "management-template-board-title";
  return (
    <section className="genre-board" aria-labelledby={titleId}>
      <div className="genre-board-header">
        <div>
          <h2 id={titleId}>{config.title}</h2>
        </div>
      </div>
      <div className="genre-board-actions">
        {config.actions.map((label) => {
          return (
            <button
              key={label}
              type="button"
              className="secondary-command compact"
              disabled={!openLookAround}
              onClick={() => openLookAround?.(label)}
            >
              <Search size={15} aria-hidden="true" />
              {label}
            </button>
          );
        })}
      </div>
      {editable && editingRow ? (
        <ExpeditionPressureEditor
          key={editingRow.key}
          label={editingRow.label}
          row={editingRow.row}
          onCancel={() => setEditingKey(null)}
          onSave={async (row) => {
            await onSaveSummary(row);
            setEditingKey(null);
          }}
        />
      ) : null}
      {rows.length ? (
        <div className="genre-board-grid">
          {rows.map((row) => {
            const stateRow = row.row;
            const canEditRow = editable && stateRow !== null && managementBoardRowCanEdit(stateRow);
            const editLabel = `Edit ${row.label.toLowerCase()}`;
            return (
              <article key={row.key} className="genre-board-state">
                <div className="genre-board-state-header">
                  <span>{row.label}</span>
                  {canEditRow ? (
                    <button
                      type="button"
                      className={touchActionClassName()}
                      title={editLabel}
                      aria-label={editLabel}
                      onClick={() => setEditingKey(row.key)}
                    >
                      <TouchActionContents icon={<Edit3 size={14} />} label="Edit" />
                    </button>
                  ) : null}
                </div>
                <strong>{row.summary}</strong>
                {row.category ? <small>{row.category}</small> : null}
              </article>
            );
          })}
        </div>
      ) : null}
    </section>
  );
}

type ManagementBoardRow = {
  key: string;
  label: string;
  summary: string;
  category: string;
  row: WorldDataRow | null;
};

function managementBoardRows(
  model: WorldDataModel,
  config: ManagementBoardConfig
): ManagementBoardRow[] {
  const stateRows = worldRows(model.world_state);
  const visibleStateRows = stateRows
    .map(({ row }) => row)
    .filter((row) => !config.stateIsVisible || config.stateIsVisible(row));
  const stateBackedRows = visibleStateRows
    .filter((row) => {
      const key = typeof row.key === "string" ? row.key : "";
      return (
        config.extraPrefixes.some((prefix) => key.startsWith(prefix))
      );
    })
    .sort((left, right) => String(left.key).localeCompare(String(right.key)))
    .map((row) => {
      const key = String(row.key);
      return managementBoardRow(key, managementBoardExtraLabel(key, config), row);
    });
  if (stateBackedRows.length) return stateBackedRows;
  return managementBoardScenarioRows(model.scenario, config);
}

function managementBoardRow(
  key: string,
  label: string,
  row: WorldDataRow
): ManagementBoardRow {
  return {
    key,
    label,
    summary: worldStateSummary(row),
    category: typeof row.category === "string" ? row.category : "",
    row
  };
}

function managementBoardScenarioRows(
  scenario: WorldDataScenario | null | undefined,
  config: ManagementBoardConfig
): ManagementBoardRow[] {
  return (scenario?.content_sections ?? []).flatMap(([key, value]) => {
    const summary = typeof value === "string" ? value.trim() : "";
    if (
      !summary
      || key === "tone_genre"
      || key === "opening_message"
      || key === "choice_style"
      || (config.stateIsVisible && /secret|hidden|private/.test(key.toLowerCase()))
    ) return [];
    return [{
      key,
      label: labelize(key),
      summary,
      category: "",
      row: null
    }];
  });
}

function managementBoardExtraLabel(key: string, config: ManagementBoardConfig): string {
  const prefix = config.extraPrefixes.find((item) => key.startsWith(item)) ?? "";
  return labelize(key.slice(prefix.length).replace(/[.-]/g, "_"));
}

function managementBoardRowCanEdit(row: WorldDataRow): boolean {
  return Boolean(managementBoardObjectValue(row));
}

function managementBoardObjectValue(row: WorldDataRow): Record<string, unknown> | null {
  const parsed = parsedWorldStateValue(row);
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    return { ...(parsed as Record<string, unknown>) };
  }
  return null;
}

function politicalIntrigueStateIsVisible(row: WorldDataRow): boolean {
  const key = typeof row.key === "string" ? row.key.toLowerCase() : "";
  const category = typeof row.category === "string" ? row.category.toLowerCase() : "";
  const parsed = parsedWorldStateValue(row);
  let hidden = false;
  let publicVisible = false;
  if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
    const record = parsed as Record<string, unknown>;
    for (const [field, value] of Object.entries(record)) {
      const normalizedField = field.toLowerCase();
      if (typeof value === "boolean") {
        if (["discovered", "known_to_player", "player_visible", "public", "revealed"].includes(normalizedField)) {
          if (value) publicVisible = true;
          else hidden = true;
        }
        if (value && ["gm_only", "hidden", "private", "secret", "undiscovered", "unrevealed"].includes(normalizedField)) {
          hidden = true;
        }
        continue;
      }
      if (typeof value !== "string" || !INVESTIGATION_CASE_VISIBILITY_FIELDS.has(normalizedField)) continue;
      const normalizedValue = value.toLowerCase().replace(/_/g, " ");
      const valueHidden = INVESTIGATION_CASE_HIDDEN_MARKERS.some((marker) => normalizedValue.includes(marker)) || normalizedValue.includes("not visible");
      if (valueHidden) {
        hidden = true;
      }
      if (!valueHidden && INVESTIGATION_CASE_PUBLIC_MARKERS.some((marker) => normalizedValue.includes(marker))) {
        publicVisible = true;
      }
    }
  }
  if (hidden) return false;
  if (publicVisible) return true;
  const valueJson = typeof row.value_json === "string" ? row.value_json.toLowerCase() : "";
  const text = `${key} ${category} ${valueJson}`;
  return !/gm[-_]|hidden|private|secret|undiscovered|unrevealed/.test(text);
}

function GuidedCleanupModal({ model, onClose, runJob }: { model?: RuntimeModel; onClose: () => void; runJob: RunJob }) {
  const [instruction, setInstruction] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const titleId = React.useId();
  const submit = async () => {
    const text = instruction.trim();
    if (!text) {
      setError("Cleanup instructions are required");
      return;
    }
    try {
      setSubmitting(true);
      const job = await postJson<Job>("/api/world-data/guided-cleanup", {
        save_id: model?.active_save_id ?? null,
        instruction: text
      });
      runJob(job);
      onClose();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not queue guided cleanup");
      setSubmitting(false);
    }
  };
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog"
        titleId={titleId}
        onClose={onClose}
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <header>
          <h2 id={titleId}>Guided cleanup</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Cleanup instructions</span>
          <textarea
            className="tall-field"
            value={instruction}
            onChange={(event) => setInstruction(event.target.value)}
            aria-label="Cleanup instructions"
          />
        </label>
        <div className="command-row end">
          {error ? <InlineNotice>{error}</InlineNotice> : null}
          <button type="button" onClick={onClose}>Cancel</button>
          <button className="primary-command compact" disabled={submitting || !instruction.trim()}>
            <Wand2 size={15} /> Run cleanup
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function ResponseGuidanceModal({ model, onClose }: { model?: RuntimeModel; onClose: () => void }) {
  const client = useQueryClient();
  const [customInstructions, setCustomInstructions] = useState(model?.custom_instructions ?? "");
  const titleId = React.useId();
  const updateGuidance = useMutation({
    mutationFn: (value: string) => postJson<RuntimeModel>("/api/runtime/custom-instructions", {
      save_id: model?.active_save_id ?? null,
      custom_instructions: value
    }),
    onSuccess: (nextModel) => {
      client.setQueryData(runtimeQueryKey(nextModel.active_save_id ?? model?.active_save_id ?? null), nextModel);
      client.invalidateQueries({ queryKey: runtimeQueryKey(nextModel.active_save_id ?? model?.active_save_id ?? null) });
      onClose();
    }
  });
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog"
        titleId={titleId}
        onClose={onClose}
        onSubmit={(event) => {
          event.preventDefault();
          updateGuidance.mutate(customInstructions);
        }}
      >
        <header>
          <h2 id={titleId}>Save response guidance</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Save response guidance</span>
          <textarea
            className="tall-field"
            value={customInstructions}
            onChange={(event) => setCustomInstructions(event.target.value)}
            aria-label="Save response guidance"
          />
        </label>
        <div className="command-row end">
          {updateGuidance.error ? <InlineNotice>{updateGuidance.error instanceof Error ? updateGuidance.error.message : "Could not save response guidance"}</InlineNotice> : null}
          <button type="button" onClick={onClose}>Cancel</button>
          <button type="button" disabled={updateGuidance.isPending || !customInstructions} onClick={() => updateGuidance.mutate("")}>
            <X size={15} /> Clear
          </button>
          <button className="primary-command compact" disabled={updateGuidance.isPending}>
            <Save size={15} /> Save
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}
