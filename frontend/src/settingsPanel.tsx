import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  ArrowDown,
  ArrowUp,
  Check,
  ChevronDown,
  Clock,
  Download,
  Edit3,
  FileText,
  FileWarning,
  GitBranch,
  History,
  Image,
  Info,
  KeyRound,
  Loader2,
  PanelRight,
  Play,
  Plus,
  RefreshCw,
  Save,
  Search,
  Settings,
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
  JobDiagnosticsModel,
  JobStepsModel,
  JobStepSummary,
  LocalSettingsModel,
  MaintenanceJobDiagnostic,
  ModelOption,
  OpenRouterProviderCatalogEntry,
  OpenRouterRoutingProfile,
  OpenRouterRoutingSettings as OpenRouterRoutingSettingsModel,
  OpenRouterRoutingTaskOverride,
  ProviderCard,
  ProviderSettingsModel,
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
  WorldDataModel
} from "./api";
import {
  actionIcon,
  apiRead,
  canUseAdminControls,
  canUseChildRestrictedControls,
  capabilityLabel,
  ConfirmModal,
  DataViewer,
  DIAGNOSTICS_STALE_MS,
  diagnosticsBundleFilename,
  diagnosticsQueryOptions,
  DialogForm,
  DialogPanel,
  EmptyState,
  formatUsd,
  fallbackTaskLabels,
  imageDimensionPresetLabel,
  imageStylePresetLabel,
  InlineNotice,
  jobDiagnosticsPath,
  jobStepsPath,
  labelize,
  MarkdownView,
  mediaAssetPath,
  mediaAssetThumbnailPath,
  ModalBackdrop,
  MODEL_CAPABILITY_ALIASES,
  MODEL_FALLBACK_LANES,
  MODEL_ROUTING_GROUPS,
  ModelPricingLine,
  modelOptionLabel,
  modelOptionSelectLabel,
  modelSelectorPurpose,
  modelPricingDisplayLabel,
  modelPricingCompactLabel,
  npcKnowledgeAuditModeLabel,
  OPENROUTER_MAX_PRICE_TOOLTIPS,
  OPENROUTER_QUANTIZATION_TOOLTIPS,
  OPENROUTER_ROUTING_TOOLTIPS,
  PanelHeader,
  pendingJobsDisplayModeLabel,
  postTurnInferenceModeLabel,
  PreviewModal,
  SAVE_SCOPED_SETTING_KEYS,
  SegmentedTabs,
  selectedOption,
  SETTINGS_TAB_TOOLTIPS,
  EMPTY_DIAGNOSTICS_FILTERS,
  settingsQueryOptions,
  settingLabel,
  settingTooltip,
  scriptGuardModeLabel,
  taskLabel,
  taskModelTooltip,
  terminalJobsPath,
  THINKING_LEVEL_OFF,
  thinkingLevelLabel,
  THINKING_LEVEL_PROVIDER_DEFAULT,
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
import type { CurrentUser, DiagnosticsFilters, ModelCapabilityFamily, ModelRoutingLane, ModelRoutingLaneGroup, ModelRoutingLaneGroupMeta, ModelRoutingLaneMeta, ModelSelectorGroup, RunJob, TerminalJobStatusFilter } from "./workbenchCore";


type SettingsTab = "providers" | "openrouter" | "models" | "save" | "local" | "diagnostics" | "users";
type SettingsPanelData = Partial<SettingsModel>;
type SettingsSummaryTone = "neutral" | "healthy" | "warning" | "attention";
type SettingsSummaryModel = {
  title: string;
  detail: string;
  helper?: string;
  facts?: string[];
  tone?: SettingsSummaryTone;
};
type SimpleModelSelectorMeta = readonly [
  string,
  readonly string[],
  readonly ModelCapabilityFamily[],
  readonly string[],
  readonly ModelCapabilityFamily[]
];
type SimpleModelSelectorGroup = {
  label: string;
  mainSelectors: TaskModelSelector[];
  mainOptions: ModelOption[];
  fallbackSelectors: TaskModelSelector[];
  fallbackOptions: ModelOption[];
};

const SETTINGS_SECTION_STALE_MS = 60_000;
const LOCAL_SETTING_KEYS = new Set([
  "pending_jobs_display_mode",
  "user_narration_guidance",
  "content_filter_rating",
  "fade_to_black_enabled",
  "debug_logging_enabled"
]);
const SIMPLE_MODEL_SELECTOR_GROUPS: readonly SimpleModelSelectorMeta[] = [
  [
    "Structured Output / Tool Calls",
    structuredToolModelPurposes(),
    ["structured_output", "tool_calling"],
    ["structured_output_fallback", "tool_call_fallback"],
    ["structured_output", "tool_calling"]
  ],
  [
    "Prose",
    routingLanePurposes("narrator", "scenario_writer", "summarization", "image_prompt"),
    ["chat"],
    ["narrator_fallback", "chat_fallback"],
    ["chat"]
  ],
  [
    "Image Generation",
    routingLanePurposes("image_generation"),
    ["image_generation"],
    ["image_fallback"],
    ["image_generation"]
  ],
  [
    "Image Edit",
    routingLanePurposes("image_edit"),
    ["image_to_image"],
    ["image_edit_fallback"],
    ["image_to_image"]
  ],
  [
    "Video Generation",
    routingLanePurposes("video_generation", "image_animation"),
    ["text_to_video", "image_to_video"],
    ["video_fallback"],
    ["text_to_video"]
  ]
];

function contentRatingLabel(rating: string) {
  return ({ g: "G", pg: "PG", "pg-13": "PG-13", r: "R", unrated: "Unrated" } as Record<string, string>)[rating]
    ?? labelize(rating);
}

function providerSettingsQueryOptions() {
  return {
    queryKey: ["settings", "providers"] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) => apiRead<ProviderSettingsModel>("/api/settings/providers", signal),
    staleTime: SETTINGS_SECTION_STALE_MS
  };
}

function localSettingsQueryOptions() {
  return {
    queryKey: ["settings", "local"] as const,
    queryFn: ({ signal }: { signal: AbortSignal }) => apiRead<LocalSettingsModel>("/api/settings/local", signal),
    staleTime: SETTINGS_SECTION_STALE_MS
  };
}

function visibleSettingsSectionsForRole(role?: string | null): SettingsTab[] {
  if (role === "child") return ["local"];
  if (role === "user") return ["save", "local", "diagnostics"];
  const tabs: SettingsTab[] = ["providers", "openrouter", "models", "save", "local", "diagnostics"];
  if (role === "admin") tabs.push("users");
  return tabs;
}

function tabUsesFullSettings(tab: SettingsTab): boolean {
  return tab === "openrouter" || tab === "models" || tab === "save";
}

export function SettingsPanel({
  runJob,
  activeSaveId = null,
  storytellerMode = false,
  currentUser = null,
  onContentSafetyChanged
}: {
  runJob: RunJob;
  activeSaveId?: string | null;
  storytellerMode?: boolean;
  currentUser?: CurrentUser | null;
  onContentSafetyChanged?: () => void;
}) {
  const [tab, setTab] = useState<SettingsTab>("providers");
  const isAdmin = currentUser?.role === "admin";
  const client = useQueryClient();
  const visibleSectionSet = useMemo(
    () => new Set(visibleSettingsSectionsForRole(currentUser?.role)),
    [currentUser?.role]
  );
  const sectionVisible = useCallback(
    (section: SettingsTab) => visibleSectionSet.has(section),
    [visibleSectionSet]
  );
  const settings = useQuery({
    ...settingsQueryOptions(activeSaveId),
    enabled: tabUsesFullSettings(tab) && sectionVisible(tab)
  });
  const providerSettings = useQuery({
    ...providerSettingsQueryOptions(),
    enabled: tab === "providers" && sectionVisible("providers")
  });
  const localSettings = useQuery({
    ...localSettingsQueryOptions(),
    enabled: tab === "local" && sectionVisible("local")
  });
  const activeSettings: SettingsPanelData | undefined = (
    tab === "providers"
      ? providerSettings.data
      : tab === "local"
        ? localSettings.data
        : settings.data
  );
  const activeSettingsLoading = (
    tab === "providers"
      ? providerSettings.isLoading
      : tab === "local"
        ? localSettings.isLoading
        : tabUsesFullSettings(tab)
          ? settings.isLoading
          : false
  );
  const activeSettingsError = (
    tab === "providers"
      ? providerSettings.error
      : tab === "local"
        ? localSettings.error
        : tabUsesFullSettings(tab)
          ? settings.error
          : null
  );
  const updateScoped = useMutation({
    mutationFn: ({
      key,
      value,
      save_id
    }: {
      key: string;
      value: unknown;
      save_id?: string | null;
    }) => {
      const payload: { key: string; value: unknown; save_id?: string } = { key, value };
      if (save_id) payload.save_id = save_id;
      return postJson("/api/settings/scoped", payload);
    },
    onSuccess: async (_data, variables) => {
      if (SAVE_SCOPED_SETTING_KEYS.has(variables.key)) {
        client.invalidateQueries({ queryKey: ["settings", "full"] });
        client.invalidateQueries({ queryKey: ["runtime"] });
      } else if (LOCAL_SETTING_KEYS.has(variables.key)) {
        client.invalidateQueries({ queryKey: ["settings", "local"] });
        client.invalidateQueries({ queryKey: ["settings", "shell"] });
        if (variables.key === "content_filter_rating") {
          onContentSafetyChanged?.();
          await client.resetQueries();
        }
      } else {
        client.invalidateQueries({ queryKey: ["settings", "full"] });
      }
    }
  });
  const updateSetting = (key: string, value: unknown) => updateScoped.mutate({
    key,
    value,
    save_id: SAVE_SCOPED_SETTING_KEYS.has(key) ? activeSaveId : null
  });
  const tabOptions = useMemo(() => {
    const allTabOptions: { value: SettingsTab; label: string; title: string }[] = [
      { value: "providers", label: "Providers", title: SETTINGS_TAB_TOOLTIPS.providers },
      { value: "openrouter", label: "OpenRouter", title: SETTINGS_TAB_TOOLTIPS.openrouter },
      { value: "models", label: "Models", title: SETTINGS_TAB_TOOLTIPS.models },
      { value: "save", label: "Save", title: SETTINGS_TAB_TOOLTIPS.save },
      { value: "local", label: "Local", title: SETTINGS_TAB_TOOLTIPS.local },
      { value: "diagnostics", label: "Diagnostics", title: SETTINGS_TAB_TOOLTIPS.diagnostics }
    ];
    if (isAdmin) {
      allTabOptions.push({ value: "users", label: "Users", title: SETTINGS_TAB_TOOLTIPS.users });
    }
    return allTabOptions.filter((option) => sectionVisible(option.value));
  }, [isAdmin, sectionVisible]);
  useEffect(() => {
    if (tabOptions.length && !tabOptions.some((option) => option.value === tab)) {
      setTab(tabOptions[0].value);
    }
  }, [tab, tabOptions]);
  return (
    <aside className="right-panel">
      <PanelHeader icon={<Settings size={18} />} title="Settings" />
      <SegmentedTabs
        className="settings-tab-grid"
        label="Settings sections"
        value={tab}
        onChange={setTab}
        options={tabOptions}
      />
      {activeSettingsLoading ? <p className="muted">Loading settings...</p> : null}
      {activeSettingsError instanceof Error ? <InlineNotice>{activeSettingsError.message}</InlineNotice> : null}
      {updateScoped.error instanceof Error ? <InlineNotice>{updateScoped.error.message}</InlineNotice> : null}
      {!activeSettingsLoading ? (
        <SettingsPanelSummary
          tab={tab}
          settings={activeSettings}
          activeSaveId={activeSaveId}
          currentUser={currentUser}
        />
      ) : null}
      {tab === "providers" && sectionVisible("providers") ? (
        <ProviderSettings
          providers={providerSettings.data?.provider_cards ?? []}
          secretWarning={secretStorageWarning(providerSettings.data)}
          onRefresh={async (provider) => {
            runJob(await postJson<Job>(`/api/settings/model-refresh/${provider}`, {}));
            client.invalidateQueries({ queryKey: ["settings", "providers"] });
            client.invalidateQueries({ queryKey: ["settings", "full"] });
          }}
        />
      ) : null}
      {tab === "openrouter" && sectionVisible("openrouter") ? <OpenRouterRoutingSettings settings={settings.data} updateLocal={updateSetting} disabled={updateScoped.isPending} /> : null}
      {tab === "models" && sectionVisible("models") ? <ModelSettings settings={settings.data} updateLocal={updateSetting} disabled={updateScoped.isPending} /> : null}
      {tab === "save" && sectionVisible("save") ? (
        <SaveSettingsControls
          settings={settings.data}
          activeSaveId={activeSaveId}
          storytellerMode={storytellerMode}
          updateLocal={updateSetting}
          disabled={updateScoped.isPending}
          runJob={runJob}
          canRunMaintenance={canUseChildRestrictedControls(currentUser)}
        />
      ) : null}
      {tab === "local" && sectionVisible("local") ? (
        <LocalSettingsControls
          settings={localSettings.data}
          updateLocal={updateSetting}
          disabled={updateScoped.isPending}
        />
      ) : null}
      {tab === "diagnostics" && sectionVisible("diagnostics") ? (
        <DiagnosticsSettings activeSaveId={activeSaveId} isAdmin={isAdmin} />
      ) : null}
      {tab === "users" && isAdmin && sectionVisible("users") ? <UserManagementSettings /> : null}
    </aside>
  );
}

function SettingsPanelSummary({
  tab,
  settings,
  activeSaveId,
  currentUser
}: {
  tab: SettingsTab;
  settings?: SettingsPanelData;
  activeSaveId: string | null;
  currentUser: CurrentUser | null;
}) {
  const summary = settingsSummaryForTab(tab, settings, activeSaveId, currentUser);
  return (
    <section className={`settings-summary ${summary.tone ?? "neutral"}`} aria-label="Settings summary">
      <div className="settings-summary-main">
        <h3>{summary.title}</h3>
        <p>{summary.detail}</p>
        {summary.helper ? <small>{summary.helper}</small> : null}
      </div>
      {summary.facts?.length ? (
        <div className="settings-summary-facts">
          {summary.facts.map((fact) => <span key={fact}>{fact}</span>)}
        </div>
      ) : null}
    </section>
  );
}

function settingsSummaryForTab(
  tab: SettingsTab,
  settings: SettingsPanelData | undefined,
  activeSaveId: string | null,
  currentUser: CurrentUser | null
): SettingsSummaryModel {
  switch (tab) {
    case "providers":
      return providerSettingsSummary(settings);
    case "openrouter":
      return openRouterSettingsSummary(settings);
    case "models":
      return modelSettingsSummary(settings);
    case "save":
      return saveSettingsSummary(settings, activeSaveId);
    case "local":
      return {
        title: "Local preferences",
        detail: "Applies to this account and browser.",
        helper: "Use these for workbench display and account-level narrator guidance.",
        facts: [
          settings?.pending_jobs_display_mode ? `Pending jobs: ${pendingJobsDisplayModeLabel(settings.pending_jobs_display_mode.selected)}` : "Workbench controls",
          settings?.debug_logging?.enabled ? "Debug logging enabled" : "Debug logging off"
        ]
      };
    case "diagnostics": {
      return {
        title: "Diagnostics health",
        detail: "Refreshes independently from settings configuration.",
        facts: [activeSaveId ? "Active save scoped" : "Global diagnostics"],
        tone: "neutral"
      };
    }
    case "users":
      return {
        title: "User management",
        detail: currentUser?.role === "admin"
          ? "Admin-only controls for local accounts, roles, and password resets."
          : "User management is only available to admins.",
        facts: currentUser ? [`Signed in as ${currentUser.username}`, `${labelize(currentUser.role)} role`] : ["Admin only"],
        tone: currentUser?.role === "admin" ? "neutral" : "warning"
      };
  }
}

function secretStorageWarning(settings?: SettingsPanelData): string | null {
  return settings?.secret_storage_warning ?? null;
}

function providerSettingsSummary(settings?: SettingsPanelData): SettingsSummaryModel {
  const providers = settings?.provider_cards ?? [];
  if (!providers.length) {
    return {
      title: "Provider setup",
      detail: "No providers configured",
      helper: "Add provider keys before refreshing model lists.",
      tone: "warning"
    };
  }
  const readyCount = providers.filter((provider) => provider.enabled && provider.has_api_key).length;
  const needsKeyCount = providers.filter((provider) => provider.enabled && !provider.has_api_key).length;
  const errorCount = providers.filter((provider) => Boolean(provider.last_error)).length;
  const facts = [
    needsKeyCount ? `${pluralCount(needsKeyCount, "provider")} needs an API key` : "All enabled providers have keys",
    errorCount ? `${pluralCount(errorCount, "provider")} reporting errors` : `${providers.reduce((total, provider) => total + provider.model_count, 0)} models cached`
  ];
  return {
    title: "Provider setup",
    detail: `${readyCount} of ${providers.length} providers ready`,
    helper: "Paste provider keys, then refresh model lists after key changes.",
    facts,
    tone: needsKeyCount || errorCount ? "warning" : "healthy"
  };
}

function openRouterSettingsSummary(settings?: SettingsPanelData): SettingsSummaryModel {
  const routing = settings?.openrouter_routing;
  if (!routing) {
    return {
      title: "OpenRouter routing",
      detail: "OpenRouter routing settings are not available.",
      helper: "Enable OpenRouter provider settings to tune endpoint routing.",
      tone: "neutral"
    };
  }
  const sortLabel = openRouterSortLabel(routing.global_profile.sort);
  const filterCount = routing.global_profile.order.length + routing.global_profile.only.length + routing.global_profile.ignore.length;
  const catalogCount = routing.provider_catalog?.length ?? 0;
  return {
    title: "OpenRouter routing",
    detail: routing.global_profile.sort === "default"
      ? "Global profile uses OpenRouter default sorting"
      : `Global profile sorts by ${sortLabel}`,
    helper: "Start with the global profile, then open advanced routing for provider filters, privacy, and price limits.",
    facts: [
      `${pluralCount(routing.task_overrides.length, "task override")} available`,
      filterCount ? `${pluralCount(filterCount, "provider filter")} active` : "No global provider filters",
      catalogCount ? `${pluralCount(catalogCount, "cached provider")}` : "No cached providers"
    ]
  };
}

function modelSettingsSummary(settings?: SettingsPanelData): SettingsSummaryModel {
  if (!settings?.task_model_selectors || !settings.roleplay_model_groups) {
    return {
      title: "Model routing",
      detail: "Loading model selectors.",
      tone: "neutral"
    };
  }
  const modelSettings = settings as SettingsModel;
  const groups = modelRoutingLaneGroups(modelSettings, MODEL_ROUTING_GROUPS);
  const fallbacks = modelRoutingLanes(modelSettings, MODEL_FALLBACK_LANES);
  const laneCount = groups.reduce((count, group) => count + group.lanes.length, 0) + fallbacks.length;
  const selectorCount = allModelSelectors(modelSettings).length;
  const savedProfileCount = settings.model_routing_profiles?.profiles.length ?? 0;
  return {
    title: "Model routing",
    detail: `${pluralCount(selectorCount, "task selector")} available`,
    helper: "Use common model sets first; open advanced routing for per-lane and per-task overrides.",
    facts: [
      `${pluralCount(laneCount, "routing lane")}`,
      settings.roleplay_shared_models?.enabled ? "Shared roleplay model set" : "Split roleplay model sets",
      savedProfileCount ? `${pluralCount(savedProfileCount, "saved profile")}` : "No saved profiles"
    ]
  };
}

function saveSettingsSummary(settings: SettingsPanelData | undefined, activeSaveId: string | null): SettingsSummaryModel {
  const facts = [
    settings?.automatic_summarization?.enabled ? "Summarization on" : "Summarization off",
    settings?.automatic_image_generation?.enabled ? "Automatic images on" : "Automatic images off"
  ];
  return {
    title: "Save settings",
    detail: activeSaveId ? "Editing active save controls" : "Load a save to edit save-scoped controls.",
    helper: "Common controls are shown first; lower sections tune automation, safety, history, and budget behavior.",
    facts,
    tone: activeSaveId ? "neutral" : "warning"
  };
}

function pluralCount(count: number, singular: string, plural = `${singular}s`): string {
  return `${count} ${count === 1 ? singular : plural}`;
}

function routingLanePurposes(...laneIds: string[]): string[] {
  const targetIds = new Set(laneIds);
  return MODEL_ROUTING_GROUPS
    .flatMap((group) => group.lanes)
    .filter((lane) => targetIds.has(lane.id))
    .flatMap((lane) => lane.targetPurposes);
}

function structuredToolModelPurposes(): string[] {
  return MODEL_ROUTING_GROUPS
    .flatMap((group) => group.lanes)
    .filter((lane) => lane.capabilities.some((capability) => capability === "structured_output" || capability === "tool_calling"))
    .flatMap((lane) => lane.targetPurposes)
    .concat("dating_route_profile");
}

function AdvancedSettingsSection({
  title,
  summary,
  defaultOpen = false,
  children,
  className = ""
}: {
  title: string;
  summary: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
  className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  const regionId = React.useId();
  return (
    <section className={`settings-subsection advanced-settings-section ${open ? "expanded" : ""} ${className}`.trim()}>
      <button
        type="button"
        className="advanced-settings-toggle"
        aria-expanded={open}
        aria-controls={regionId}
        onClick={() => setOpen((current) => !current)}
      >
        <span>
          <strong>{title}</strong>
          <small>{summary}</small>
        </span>
        <ChevronDown size={16} aria-hidden="true" />
      </button>
      {open ? (
        <div id={regionId} role="region" aria-label={title} className="advanced-settings-body">
          {children}
        </div>
      ) : null}
    </section>
  );
}

function UserManagementSettings() {
  const client = useQueryClient();
  const [newUsername, setNewUsername] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [newRole, setNewRole] = useState("user");
  const [passwordDrafts, setPasswordDrafts] = useState<Record<string, string>>({});
  const users = useQuery({
    queryKey: ["admin", "users"],
    queryFn: () => api<{ users: AdminUser[] }>("/api/admin/users")
  });
  const invalidateUsers = () => client.invalidateQueries({ queryKey: ["admin", "users"] });
  const createUser = useMutation({
    mutationFn: () => postJson<{ user: AdminUser }>("/api/admin/users", {
      username: newUsername.trim(),
      password: newPassword,
      role: newRole
    }),
    onSuccess: () => {
      setNewUsername("");
      setNewPassword("");
      setNewRole("user");
      invalidateUsers();
    }
  });
  const updateUser = useMutation({
    mutationFn: ({ userId, role, status, contentRating }: { userId: string; role?: string; status?: string; contentRating?: string }) => (
      api<{ user: AdminUser }>(`/api/admin/users/${encodeURIComponent(userId)}`, {
        method: "PATCH",
        body: JSON.stringify({ role, status, content_rating: contentRating })
      })
    ),
    onSuccess: invalidateUsers
  });
  const resetPassword = useMutation({
    mutationFn: ({ userId, password }: { userId: string; password: string }) => (
      postJson<{ user: AdminUser }>(`/api/admin/users/${encodeURIComponent(userId)}/password`, { password })
    ),
    onSuccess: (_data, variables) => {
      setPasswordDrafts((current) => {
        const next = { ...current };
        delete next[variables.userId];
        return next;
      });
      invalidateUsers();
    }
  });
  const error = [users.error, createUser.error, updateUser.error, resetPassword.error]
    .find((failure): failure is Error => failure instanceof Error);

  return (
    <div className="settings-stack user-management">
      <section className="settings-subsection">
        <h3>Create User</h3>
        <form
          className="admin-user-create"
          onSubmit={(event) => {
            event.preventDefault();
            if (!newUsername.trim() || !newPassword || createUser.isPending) return;
            createUser.mutate();
          }}
        >
          <label>
            New username
            <input value={newUsername} onChange={(event) => setNewUsername(event.currentTarget.value)} autoComplete="off" />
          </label>
          <label>
            New password
            <input value={newPassword} onChange={(event) => setNewPassword(event.currentTarget.value)} type="password" autoComplete="new-password" minLength={12} />
          </label>
          <label>
            New role
            <select value={newRole} onChange={(event) => setNewRole(event.currentTarget.value)}>
              <option value="user">User</option>
              <option value="child">Child</option>
              <option value="admin">Admin</option>
            </select>
          </label>
          <button className="primary-command compact" disabled={createUser.isPending || !newUsername.trim() || !newPassword}>
            {createUser.isPending ? <Loader2 className="spin" size={15} /> : <Plus size={15} />}
            Create user
          </button>
        </form>
      </section>
      <section className="settings-subsection">
        <h3>Users</h3>
        {users.isLoading ? <p className="muted">Loading users...</p> : null}
        {error ? <InlineNotice>{error.message}</InlineNotice> : null}
        <div className="admin-user-list">
          {(users.data?.users ?? []).map((user) => {
            const passwordDraft = passwordDrafts[user.id] ?? "";
            return (
              <div className="admin-user-row" key={user.id}>
                <div className="admin-user-identity">
                  <strong>{user.username}</strong>
                  <small>{user.role} · {user.status}</small>
                </div>
                <select
                  aria-label={`${user.username} role`}
                  value={user.role}
                  disabled={updateUser.isPending}
                  onChange={(event) => updateUser.mutate({ userId: user.id, role: event.currentTarget.value })}
                >
                  <option value="admin">Admin</option>
                  <option value="user">User</option>
                  <option value="child">Child</option>
                </select>
                <select
                  aria-label={`${user.username} status`}
                  value={user.status}
                  disabled={updateUser.isPending}
                  onChange={(event) => updateUser.mutate({ userId: user.id, status: event.currentTarget.value })}
                >
                  <option value="active">Active</option>
                  <option value="disabled">Disabled</option>
                </select>
                <select
                  aria-label={`${user.username} content rating`}
                  value={user.content_rating ?? (user.role === "child" ? "pg" : "pg-13")}
                  disabled={updateUser.isPending}
                  onChange={(event) => updateUser.mutate({ userId: user.id, contentRating: event.currentTarget.value })}
                >
                  {(user.role === "child" ? ["g", "pg", "pg-13"] : ["g", "pg", "pg-13", "r", "unrated"])
                    .map((rating) => <option key={rating} value={rating}>{contentRatingLabel(rating)}</option>)}
                </select>
                <input
                  aria-label={`${user.username} new password`}
                  type="password"
                  autoComplete="new-password"
                  minLength={12}
                  value={passwordDraft}
                  onChange={(event) => {
                    const value = event.currentTarget.value;
                    setPasswordDrafts((current) => ({
                      ...current,
                      [user.id]: value
                    }));
                  }}
                />
                <button
                  type="button"
                  className="secondary-command compact"
                  disabled={resetPassword.isPending || !passwordDraft}
                  aria-label={`Reset ${user.username} password`}
                  onClick={() => resetPassword.mutate({ userId: user.id, password: passwordDraft })}
                >
                  <KeyRound size={15} />
                  Reset
                </button>
              </div>
            );
          })}
        </div>
        {!users.isLoading && !(users.data?.users?.length ?? 0) ? <p className="empty">No users</p> : null}
      </section>
    </div>
  );
}

function ProviderSettings({ providers, secretWarning, onRefresh }: { providers: ProviderCard[]; secretWarning: string | null; onRefresh: (provider: string) => void }) {
  return (
    <div className="settings-stack">
      {secretWarning ? <InlineNotice>{secretWarning}</InlineNotice> : null}
      {providers.map((provider) => (
        <ProviderKeyForm key={provider.provider} provider={provider} onRefresh={() => onRefresh(provider.provider)} />
      ))}
      {!providers.length ? <p className="empty">No providers configured</p> : null}
    </div>
  );
}

function ProviderKeyForm({ provider, onRefresh }: { provider: ProviderCard; onRefresh: () => void }) {
  const client = useQueryClient();
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const keyHelpId = React.useId();
  const trimmedKey = key.trim();
  const keyHelp = `Paste the API key Bragi uses when calling ${provider.provider}.`;
  return (
    <form
      className="provider-card"
      onSubmit={async (event) => {
        event.preventDefault();
        if (!trimmedKey) {
          setError("");
          return;
        }
        setPending(true);
        try {
          await postJson("/api/settings/provider-key", { provider: provider.provider, api_key: trimmedKey });
          setKey("");
          setError("");
          client.invalidateQueries({ queryKey: ["settings", "providers"] });
        } catch (failure) {
          setError(failure instanceof Error ? failure.message : "Could not save key");
        } finally {
          setPending(false);
        }
      }}
    >
      <div className="provider-card-head">
        <div>
          <strong>{provider.provider}</strong>
          <span>{provider.refresh_status}</span>
        </div>
        <span className={provider.enabled ? "status-pill active" : "status-pill"}>{provider.enabled ? "Enabled" : "Disabled"}</span>
      </div>
      <div className="provider-form">
        <input
          type="password"
          autoComplete="off"
          aria-label={`${provider.provider} API key`}
          value={key}
          onChange={(event) => setKey(event.target.value)}
          placeholder={provider.has_api_key ? "Key saved" : "API key"}
          title={keyHelp}
          aria-describedby={keyHelpId}
          disabled={pending}
        />
        <button title="Store this provider key locally for future requests." aria-label={`Save ${provider.provider} API key`} disabled={pending || !trimmedKey}><Save size={15} /></button>
        <button
          type="button"
          title={`Remove the saved API key for ${provider.provider}.`}
          aria-label={`Remove ${provider.provider} API key`}
          disabled={pending || !provider.has_api_key}
          onClick={async () => {
            setPending(true);
            try {
              await deleteJson(`/api/settings/provider-key/${encodeURIComponent(provider.provider)}`);
              setKey("");
              setError("");
              client.invalidateQueries({ queryKey: ["settings", "providers"] });
            } catch (failure) {
              setError(failure instanceof Error ? failure.message : "Could not remove key");
            } finally {
              setPending(false);
            }
          }}
        >
          <Trash2 size={15} />
        </button>
        <button type="button" title={`Fetch the current model list for ${provider.provider}.`} aria-label={`Refresh ${provider.provider} models`} onClick={onRefresh} disabled={pending}><RefreshCw size={15} /></button>
      </div>
      <small id={keyHelpId} className="provider-card-help">
        {keyHelp} Save stores the key locally; refresh fetches the current model list.
      </small>
      <div className="provider-meta">
        <span>{provider.model_count} models</span>
        {provider.last_model_refresh_at ? <span>{provider.last_model_refresh_at}</span> : null}
      </div>
      {provider.last_error || error ? <InlineNotice>{provider.last_error || error}</InlineNotice> : null}
    </form>
  );
}

function RoleplaySharedModelSettings({ settings, updateLocal, disabled }: { settings: SettingsModel; updateLocal: (key: string, value: unknown) => void; disabled: boolean }) {
  const sharedModels = settings.roleplay_shared_models;
  if (!sharedModels) return null;
  const sharedTooltip = "Use one model set for all roleplay modes instead of separate full-roleplay and character-interaction defaults.";
  return (
    <section className="settings-subsection">
      <h3>Roleplay Model Sets</h3>
      <label className="toggle-row" title={sharedTooltip}>
        <input
          type="checkbox"
          checked={sharedModels.enabled}
          disabled={disabled}
          title={sharedTooltip}
          onChange={(event) => updateLocal(sharedModels.setting_key, event.target.checked)}
        />
        <span>
          <strong>Shared roleplay models</strong>
          <small>Use one roleplay model set across scenario types</small>
        </span>
      </label>
    </section>
  );
}

function ModelSettings({ settings, updateLocal, disabled }: { settings?: SettingsModel; updateLocal: (key: string, value: unknown) => void; disabled: boolean }) {
  if (!settings) return null;
  const groups = modelRoutingLaneGroups(settings, MODEL_ROUTING_GROUPS);
  const fallbacks = modelRoutingLanes(settings, MODEL_FALLBACK_LANES);
  const routedTasks = routedModelTaskSet(groups, fallbacks);
  const otherSelectors = allModelSelectors(settings).filter((selector) => !routedTasks.has(selector.task));
  return (
    <div className="settings-stack">
      <ModelRoutingProfileControls settings={settings} />
      <SimpleModelSelectorSettings settings={settings} disabled={disabled} />
      <RoleplaySharedModelSettings settings={settings} updateLocal={updateLocal} disabled={disabled} />
      {settings.retry_count || settings.provider_call_deadline_seconds ? (
        <AdvancedSettingsSection
          title="Retry policy"
          summary="Retries and total provider-call deadline."
        >
          {settings.retry_count ? <NumberSetting control={settings.retry_count} disabled={disabled} updateLocal={updateLocal} /> : null}
          {settings.provider_call_deadline_seconds ? <NumberSetting control={settings.provider_call_deadline_seconds} disabled={disabled} updateLocal={updateLocal} /> : null}
        </AdvancedSettingsSection>
      ) : null}
      {groups.length || fallbacks.length || otherSelectors.length ? (
        <AdvancedSettingsSection
          title="Advanced model routing"
          summary="Per-lane model selectors, fallback model lanes, and unmapped task overrides."
        >
          <ModelRoutingSettings groups={groups} fallbacks={fallbacks} />
          {otherSelectors.length ? (
            <section className="settings-subsection">
              <h3>Other Model Tasks</h3>
              <ModelSelectorGroups selectors={otherSelectors} emptyLabel="No other model tasks" />
            </section>
          ) : null}
        </AdvancedSettingsSection>
      ) : null}
    </div>
  );
}

function SimpleModelSelectorSettings({ settings, disabled }: { settings: SettingsModel; disabled: boolean }) {
  const groups = simpleModelSelectorGroups(settings);
  if (!groups.length) return null;
  return (
    <section className="settings-subsection">
      <h3>Simple Model Selectors</h3>
      {groups.map((group) => (
        <SimpleModelSelectorRow key={group.label} group={group} disabled={disabled} />
      ))}
    </section>
  );
}

function SimpleModelSelectorRow({ group, disabled }: { group: SimpleModelSelectorGroup; disabled: boolean }) {
  const client = useQueryClient();
  const commonMainSelection = commonSelectedModelValue(group.mainSelectors, group.mainOptions);
  const commonFallbackSelection = commonSelectedModelValue(group.fallbackSelectors, group.fallbackOptions);
  const [mainValue, setMainValue] = useState(commonMainSelection);
  const [fallbackValue, setFallbackValue] = useState(commonFallbackSelection);
  const [mainThinking, setMainThinking] = useState(commonSelectedThinkingValue(group.mainSelectors, commonMainSelection));
  const [fallbackThinking, setFallbackThinking] = useState(commonSelectedThinkingValue(group.fallbackSelectors, commonFallbackSelection));
  const [error, setError] = useState("");

  useEffect(() => {
    setMainValue(commonMainSelection);
    setFallbackValue(commonFallbackSelection);
    setMainThinking(commonSelectedThinkingValue(group.mainSelectors, commonMainSelection));
    setFallbackThinking(commonSelectedThinkingValue(group.fallbackSelectors, commonFallbackSelection));
  }, [commonMainSelection, commonFallbackSelection, group]);

  const savePreferences = useMutation({
    mutationFn: async ({ main, fallback, mainThinkingLevel, fallbackThinkingLevel }: { main: string; fallback: string; mainThinkingLevel: string; fallbackThinkingLevel: string }) => {
      await saveSimpleModelPreferences(group.mainSelectors, main);
      await saveSimpleModelPreferences(group.fallbackSelectors, fallback);
      await saveSimpleModelThinking(group.mainSelectors, main, mainThinkingLevel);
      await saveSimpleModelThinking(group.fallbackSelectors, fallback, fallbackThinkingLevel);
    },
    onSuccess: () => {
      setError("");
    },
    onError: (failure) => setError(failure instanceof Error ? failure.message : "Could not save simple model selectors"),
    onSettled: () => {
      client.invalidateQueries({ queryKey: ["settings", "full"] });
      client.invalidateQueries({ queryKey: ["runtime"] });
    }
  });

  const controlsDisabled = disabled || savePreferences.isPending;
  const canApply = Boolean(mainValue || fallbackValue);
  const mainDraftOption = modelOptionForValue(group.mainOptions, mainValue);
  const fallbackDraftOption = modelOptionForValue(group.fallbackOptions, fallbackValue);
  const mainThinkingControl = thinkingControlForModelOption(mainDraftOption, mainThinking, group.label);
  const fallbackThinkingControl = thinkingControlForModelOption(fallbackDraftOption, fallbackThinking, group.label);

  return (
    <div className="model-routing-row">
      <strong>{group.label}</strong>
      <div className="model-routing-actions">
        <div className="field-label">
          <span>Main</span>
          {simpleModelOptionSelect(
            `${group.label} main model`,
            mainValue,
            group.mainOptions,
            controlsDisabled || !group.mainSelectors.length || !group.mainOptions.length,
            (value) => {
              setMainValue(value);
              setMainThinking(THINKING_LEVEL_PROVIDER_DEFAULT);
            }
          )}
          <ThinkingLevelSelect
            control={mainThinkingControl}
            label={`${group.label} main thinking level`}
            disabled={controlsDisabled || !group.mainOptions.length}
            onChange={setMainThinking}
          />
        </div>
        <div className="field-label">
          <span>Fallback</span>
          {simpleModelOptionSelect(
            `${group.label} fallback model`,
            fallbackValue,
            group.fallbackOptions,
            controlsDisabled || !group.fallbackSelectors.length || !group.fallbackOptions.length,
            (value) => {
              setFallbackValue(value);
              setFallbackThinking(THINKING_LEVEL_PROVIDER_DEFAULT);
            }
          )}
          <ThinkingLevelSelect
            control={fallbackThinkingControl}
            label={`${group.label} fallback thinking level`}
            disabled={controlsDisabled || !group.fallbackOptions.length}
            onChange={setFallbackThinking}
          />
        </div>
        <button
          type="button"
          className="primary-command compact"
          disabled={controlsDisabled || !canApply}
          onClick={() => savePreferences.mutate({
            main: mainValue,
            fallback: fallbackValue,
            mainThinkingLevel: mainThinking,
            fallbackThinkingLevel: fallbackThinking
          })}
        >
          {savePreferences.isPending ? <Loader2 size={15} /> : <Check size={15} />}
          Apply
        </button>
      </div>
      {error ? <InlineNotice>{error}</InlineNotice> : null}
    </div>
  );
}

async function saveSimpleModelPreferences(selectors: TaskModelSelector[], value: string): Promise<boolean> {
  const preference = modelPreferenceFromValue(value);
  if (!preference) return false;
  for (const selector of selectors) {
    await postJson("/api/settings/model-preference", {
      task: selector.task,
      provider: preference.provider,
      model_id: preference.model_id
    });
  }
  return true;
}

async function saveSimpleModelThinking(
  selectors: TaskModelSelector[],
  value: string,
  thinkingLevel: string
): Promise<boolean> {
  const preference = modelPreferenceFromValue(value);
  if (!preference || !selectors.length) return false;
  for (const selector of selectors) {
    if (thinkingLevel === THINKING_LEVEL_PROVIDER_DEFAULT) {
      await deleteJson(modelThinkingPreferencePath(selector.task));
    } else {
      await postJson("/api/settings/model-thinking", {
        task: selector.task,
        provider: preference.provider,
        model_id: preference.model_id,
        level: thinkingLevel
      });
    }
  }
  return true;
}

function simpleModelOptionSelect(
  label: string,
  value: string,
  options: ModelOption[],
  disabled: boolean,
  onChange: (value: string) => void
) {
  return (
    <select value={value} disabled={disabled} aria-label={label} onChange={(event) => onChange(event.target.value)}>
      <option value="">{options.length ? "Choose" : "None"}</option>
      {options.map((option) => (
        <option key={modelOptionValue(option)} value={modelOptionValue(option)}>
          {modelOptionSelectLabel(option)}
        </option>
      ))}
    </select>
  );
}

function ModelRoutingProfileControls({ settings }: { settings: SettingsModel }) {
  const profileSettings = settings.model_routing_profiles;
  const client = useQueryClient();
  const [error, setError] = useState("");
  const profiles = profileSettings?.profiles ?? [];
  const profileSignature = profiles.map((profile) => `${profile.id}:${profile.name}:${profile.preference_count}`).join("\u0001");
  const initialProfileId = profileSettings?.last_loaded_profile_id ?? profiles[0]?.id ?? "";
  const [selectedProfileId, setSelectedProfileId] = useState(initialProfileId);
  const selectedProfile = profiles.find((profile) => profile.id === selectedProfileId) ?? null;
  const [profileName, setProfileName] = useState(selectedProfile?.name ?? "");
  const trimmedName = profileName.trim();
  const hasProfiles = profiles.length > 0;

  useEffect(() => {
    const nextProfileId = profileSettings?.last_loaded_profile_id ?? profiles[0]?.id ?? "";
    setSelectedProfileId(nextProfileId);
    setProfileName(profiles.find((profile) => profile.id === nextProfileId)?.name ?? "");
  }, [profileSettings?.last_loaded_profile_id, profileSignature]);

  const invalidate = (includeRuntime = false) => {
    client.invalidateQueries({ queryKey: ["settings", "full"] });
    if (includeRuntime) client.invalidateQueries({ queryKey: ["runtime"] });
  };
  const saveProfile = useMutation({
    mutationFn: ({ name, profile_id }: { name: string; profile_id?: string }) => (
      postJson<{ profile?: { id?: string } }>("/api/settings/model-routing-profiles", profile_id ? { name, profile_id } : { name })
    ),
    onSuccess: (result) => {
      setError("");
      if (typeof result.profile?.id === "string") setSelectedProfileId(result.profile.id);
      invalidate();
    },
    onError: (failure) => setError(failure instanceof Error ? failure.message : "Could not save model routing profile")
  });
  const loadProfile = useMutation({
    mutationFn: (profileId: string) => postJson(`/api/settings/model-routing-profiles/${encodeURIComponent(profileId)}/apply`, {}),
    onSuccess: () => {
      setError("");
      invalidate(true);
    },
    onError: (failure) => setError(failure instanceof Error ? failure.message : "Could not load model routing profile")
  });
  const deleteProfile = useMutation({
    mutationFn: (profileId: string) => deleteJson(`/api/settings/model-routing-profiles/${encodeURIComponent(profileId)}`),
    onSuccess: () => {
      setError("");
      invalidate();
    },
    onError: (failure) => setError(failure instanceof Error ? failure.message : "Could not delete model routing profile")
  });
  const controlsDisabled = saveProfile.isPending || loadProfile.isPending || deleteProfile.isPending;

  if (!profileSettings) return null;

  const selectProfile = (profileId: string) => {
    setSelectedProfileId(profileId);
    setProfileName(profiles.find((profile) => profile.id === profileId)?.name ?? "");
  };

  return (
    <section className="settings-subsection model-profile-section">
      <h3>Model Profiles</h3>
      <div className="model-profile-controls">
        <label className="field-label">
          <span>Saved profile</span>
          <select value={selectedProfileId} disabled={!hasProfiles || controlsDisabled} onChange={(event) => selectProfile(event.target.value)}>
            {!hasProfiles ? <option value="">No saved profiles</option> : null}
            {profiles.map((profile) => (
              <option key={profile.id} value={profile.id}>{profile.name}</option>
            ))}
          </select>
        </label>
        <label className="field-label">
          <span>Profile name</span>
          <input value={profileName} disabled={controlsDisabled} onChange={(event) => setProfileName(event.target.value)} />
        </label>
        <div className="command-row model-profile-actions">
          <button type="button" disabled={!trimmedName || controlsDisabled} onClick={() => saveProfile.mutate({ name: trimmedName })}>
            {saveProfile.isPending ? <Loader2 size={15} /> : <Save size={15} />}
            Save new profile
          </button>
          <button type="button" disabled={!selectedProfile || !trimmedName || controlsDisabled} onClick={() => selectedProfile && saveProfile.mutate({ name: trimmedName, profile_id: selectedProfile.id })}>
            <Check size={15} />
            Overwrite profile
          </button>
          <button type="button" disabled={!selectedProfile || controlsDisabled} onClick={() => selectedProfile && loadProfile.mutate(selectedProfile.id)}>
            {loadProfile.isPending ? <Loader2 size={15} /> : <RefreshCw size={15} />}
            Load profile
          </button>
          <button type="button" disabled={!selectedProfile || controlsDisabled} onClick={() => selectedProfile && deleteProfile.mutate(selectedProfile.id)}>
            {deleteProfile.isPending ? <Loader2 size={15} /> : <Trash2 size={15} />}
            Delete profile
          </button>
        </div>
      </div>
      {selectedProfile ? (
        <div className="model-routing-chips">
          <span>{selectedProfile.preference_count === 1 ? "1 override" : `${selectedProfile.preference_count} overrides`}</span>
          <span>{selectedProfile.roleplay_shared_models_enabled ? "shared roleplay" : "split roleplay"}</span>
        </div>
      ) : null}
      {error ? <InlineNotice>{error}</InlineNotice> : null}
    </section>
  );
}

function ModelRoutingSettings({ groups, fallbacks, saveId = null }: { groups: ModelRoutingLaneGroup[]; fallbacks: ModelRoutingLane[]; saveId?: string | null }) {
  if (!groups.length && !fallbacks.length) return null;
  return (
    <section className="settings-subsection model-routing-section">
      {groups.length ? (
        <>
          <h3>Routing Lanes</h3>
          <div className="model-routing-groups">
            {groups.map((group) => (
              <section key={group.label} className="model-routing-group">
                <h4>{group.label}</h4>
                <div className="model-routing-grid">
                  {group.lanes.map((lane) => (
                    <ModelRoutingLaneRow key={lane.id} lane={lane} saveId={saveId} />
                  ))}
                </div>
              </section>
            ))}
          </div>
        </>
      ) : null}
      {fallbacks.length ? (
        <section className="settings-subsection model-routing-fallbacks">
          <h3>Fallback Models</h3>
          <div className="model-routing-grid compact">
            {fallbacks.map((lane) => (
              <ModelRoutingLaneRow key={lane.id} lane={lane} compact saveId={saveId} />
            ))}
          </div>
        </section>
      ) : null}
    </section>
  );
}

function ModelSelectorGroups({ selectors, emptyLabel, saveId = null }: { selectors: TaskModelSelector[]; emptyLabel: string; saveId?: string | null }) {
  const groups = modelSelectorGroups(selectors);
  if (!selectors.length) return <p className="empty">{emptyLabel}</p>;
  return (
    <div className="model-selector-groups">
      {groups.map((group) => (
        <section key={group.label} className="model-selector-group">
          <h4>{group.label}</h4>
          <div className="model-selector-list">
            {group.selectors.map((selector) => (
              <ModelSelector key={selector.task} selector={selector} saveId={saveId} />
            ))}
          </div>
        </section>
      ))}
    </div>
  );
}

function ModelRoutingLaneRow({ lane, compact = false, saveId = null }: { lane: ModelRoutingLane; compact?: boolean; saveId?: string | null }) {
  const client = useQueryClient();
  const commonSelected = commonSelectedModelValue(lane.selectors, lane.options);
  const optionSignature = lane.options.map(modelOptionValue).join("\u0001");
  const [selectedValue, setSelectedValue] = useState(commonSelected);
  const [thinkingValue, setThinkingValue] = useState(commonSelectedThinkingValue(lane.selectors, commonSelected));
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  useEffect(() => {
    setSelectedValue(commonSelected);
    setThinkingValue(commonSelectedThinkingValue(lane.selectors, commonSelected));
  }, [commonSelected, optionSignature, lane.selectors]);

  const savePreferences = useMutation({
    mutationFn: async ({ provider, model_id, thinking_level }: { provider: string; model_id: string; thinking_level: string }) => {
      for (const selector of lane.selectors) {
        await postJson("/api/settings/model-preference", saveId ? { task: selector.task, provider, model_id, save_id: saveId } : { task: selector.task, provider, model_id });
      }
      for (const selector of lane.selectors) {
        if (thinking_level === THINKING_LEVEL_PROVIDER_DEFAULT) {
          await deleteJson(modelThinkingPreferencePath(selector.task, saveId));
        } else {
          await postJson("/api/settings/model-thinking", {
            task: selector.task,
            provider,
            model_id,
            level: thinking_level,
            ...(saveId ? { save_id: saveId } : {})
          });
        }
      }
    },
    onSuccess: () => {
      setError("");
    },
    onError: (failure) => setError(failure instanceof Error ? failure.message : "Could not save model preferences"),
    onSettled: () => {
      client.invalidateQueries({ queryKey: ["settings", "full"] });
      client.invalidateQueries({ queryKey: ["runtime"] });
    }
  });

  const applySelection = () => {
    const [provider, model_id] = selectedValue.split("\u0000");
    if (provider && model_id) {
      savePreferences.mutate({
        provider,
        model_id,
        thinking_level: thinkingControl.selected
      });
    }
  };
  const Icon = lane.icon;
  const modelCountLabel = lane.selectors.length === 1 ? "1 task" : `${lane.selectors.length} tasks`;
  const draftOption = modelOptionForValue(lane.options, selectedValue);
  const thinkingControl = thinkingControlForModelOption(draftOption, thinkingValue, lane.id);
  const rowClassName = ["model-routing-row", compact ? "compact" : "", open ? "expanded" : ""].filter(Boolean).join(" ");
  const taskListId = `model-routing-${lane.id}-tasks`;

  return (
    <div className={rowClassName} title={lane.title}>
      <div className="model-routing-summary">
        <button
          type="button"
          className="model-routing-toggle"
          aria-expanded={open}
          aria-controls={taskListId}
          title={lane.title}
          onClick={() => setOpen((current) => !current)}
        >
          <span className="model-routing-head">
            <span className="model-routing-icon" aria-hidden="true"><Icon size={16} /></span>
            <span className="model-routing-title">
              <strong>{lane.label}</strong>
              <span className="model-routing-chips">
                <span>{capabilityRequirementLabel(lane.capabilities)}</span>
                <span>{modelCountLabel}</span>
              </span>
            </span>
          </span>
          <ChevronDown className="model-routing-chevron" size={16} aria-hidden="true" />
        </button>
        <div className="model-routing-actions">
          <ModelOptionSelect
            label={`${lane.label} model`}
            title={lane.title}
            value={selectedValue}
            options={lane.options}
            disabled={!lane.options.length || savePreferences.isPending}
            emptyLabel="No common compatible models"
            placeholderLabel={commonSelectionPlaceholder(lane.selectors, commonSelected)}
            onChange={(value) => {
              setSelectedValue(value);
              setThinkingValue(THINKING_LEVEL_PROVIDER_DEFAULT);
            }}
          />
          <ThinkingLevelSelect
            control={thinkingControl}
            label={`${lane.label} thinking level`}
            disabled={!lane.options.length || savePreferences.isPending}
            onChange={setThinkingValue}
          />
          <button type="button" className="primary-command compact" disabled={!lane.options.length || !selectedValue || savePreferences.isPending} onClick={applySelection}>
            {savePreferences.isPending ? <Loader2 size={15} /> : <Check size={15} />}
            Apply
          </button>
          <ModelPricingLine option={draftOption} />
        </div>
      </div>
      {open ? (
        <div id={taskListId} className="model-routing-body">
          <div className="model-routing-task-list">
            {lane.selectors.map((selector) => (
              <ModelSelector key={selector.task} selector={selector} labelOverride={routingLaneTaskLabel(selector)} saveId={saveId} />
            ))}
          </div>
        </div>
      ) : null}
      {error ? <InlineNotice>{error}</InlineNotice> : null}
    </div>
  );
}

function modelRoutingLaneGroups(settings: SettingsModel, groups: readonly ModelRoutingLaneGroupMeta[]): ModelRoutingLaneGroup[] {
  return groups.flatMap((group) => {
    const lanes = modelRoutingLanes(settings, group.lanes);
    if (!lanes.length) return [];
    return [{ label: group.label, lanes }];
  });
}

function routedModelTaskSet(groups: ModelRoutingLaneGroup[], fallbacks: ModelRoutingLane[]): Set<string> {
  const tasks = new Set<string>();
  for (const group of groups) {
    for (const lane of group.lanes) {
      for (const selector of lane.selectors) {
        tasks.add(selector.task);
      }
    }
  }
  for (const lane of fallbacks) {
    for (const selector of lane.selectors) {
      tasks.add(selector.task);
    }
  }
  return tasks;
}

function modelRoutingLanes(settings: SettingsModel, metas: readonly ModelRoutingLaneMeta[]): ModelRoutingLane[] {
  return metas.flatMap((meta) => {
    const selectors = selectorsForLane(settings, meta);
    if (!selectors.length) return [];
    return [{
      ...meta,
      selectors,
      options: commonModelOptions(selectors, meta.capabilities)
    }];
  });
}

function simpleModelSelectorGroups(settings: SettingsModel): SimpleModelSelectorGroup[] {
  const selectors = allModelSelectors(settings);
  return SIMPLE_MODEL_SELECTOR_GROUPS.flatMap(([label, mainPurposes, mainCapabilities, fallbackPurposes, fallbackCapabilities]) => {
    const mainSelectors = selectorsForPurposes(selectors, mainPurposes);
    const fallbackSelectors = selectorsForPurposes(selectors, fallbackPurposes);
    if (!mainSelectors.length && !fallbackSelectors.length) return [];
    return [{
      label,
      mainSelectors,
      mainOptions: commonModelOptions(mainSelectors, mainCapabilities),
      fallbackSelectors,
      fallbackOptions: commonModelOptions(fallbackSelectors, fallbackCapabilities)
    }];
  });
}

function selectorsForPurposes(selectors: TaskModelSelector[], purposes: readonly string[]): TaskModelSelector[] {
  const targetPurposes = new Set(purposes);
  return selectors.filter((selector) => targetPurposes.has(modelSelectorPurpose(selector.task)));
}

function selectorsForLane(settings: SettingsModel, meta: ModelRoutingLaneMeta): TaskModelSelector[] {
  const targetPurposes = new Set(meta.targetPurposes);
  return allModelSelectors(settings).filter((selector) => targetPurposes.has(modelSelectorPurpose(selector.task)));
}

function modelSelectorGroups(selectors: TaskModelSelector[]): ModelSelectorGroup[] {
  const remaining = new Map(selectors.map((selector) => [selector.task, selector]));
  const groups: ModelSelectorGroup[] = [];
  for (const group of modelSelectorGroupMetas()) {
    const groupedSelectors = selectorsForModelGroup(selectors, remaining, group.lanes);
    if (groupedSelectors.length) groups.push({ label: group.label, selectors: groupedSelectors });
  }
  if (remaining.size) groups.push({ label: "Other", selectors: Array.from(remaining.values()) });
  return groups;
}

function modelSelectorGroupMetas(): readonly ModelRoutingLaneGroupMeta[] {
  return [
    ...MODEL_ROUTING_GROUPS,
    {
      label: "Fallback Models",
      lanes: MODEL_FALLBACK_LANES
    }
  ];
}

function selectorsForModelGroup(
  selectors: TaskModelSelector[],
  remaining: Map<string, TaskModelSelector>,
  lanes: readonly ModelRoutingLaneMeta[]
): TaskModelSelector[] {
  const groupedSelectors: TaskModelSelector[] = [];
  for (const lane of lanes) {
    const targetPurposes = new Set(lane.targetPurposes);
    for (const selector of selectors) {
      if (!remaining.has(selector.task)) continue;
      if (!targetPurposes.has(modelSelectorPurpose(selector.task))) continue;
      groupedSelectors.push(selector);
      remaining.delete(selector.task);
    }
  }
  return groupedSelectors;
}

function allModelSelectors(settings: SettingsModel): TaskModelSelector[] {
  const selectors = new Map<string, TaskModelSelector>();
  for (const selector of settings.task_model_selectors) {
    if (isRetiredModelTask(selector.task)) continue;
    selectors.set(selector.task, selector);
  }
  for (const group of settings.roleplay_model_groups) {
    for (const selector of group.selectors) {
      if (isRetiredModelTask(selector.task)) continue;
      if (!selectors.has(selector.task)) selectors.set(selector.task, selector);
    }
  }
  for (const selector of settings.scenario_section_model_selectors ?? []) {
    if (!selectors.has(selector.task)) selectors.set(selector.task, selector);
  }
  return Array.from(selectors.values());
}

function isRetiredModelTask(task: string): boolean {
  return task === "chat_character_interaction" || task.startsWith("character_interaction_");
}

function OpenRouterRoutingSettings({
  settings,
  updateLocal,
  disabled
}: {
  settings?: SettingsModel;
  updateLocal: (key: string, value: unknown) => void;
  disabled: boolean;
}) {
  const routing = settings?.openrouter_routing;
  const [selectedProfile, setSelectedProfile] = useState("global");
  if (!routing) return null;
  const providerCatalog = routing.provider_catalog ?? [];
  const selectedOverride = routing.task_overrides.find((override) => override.task_family === selectedProfile) ?? null;
  const activeProfile = selectedOverride?.profile ?? routing.global_profile;
  const effectivePayload = selectedOverride ? selectedOverride.effective_provider_payload : routing.global_provider_payload;
  const controlsDisabled = disabled || Boolean(selectedOverride && !selectedOverride.enabled);
  const selectedProfileLabel = selectedOverride?.label ?? "Global";
  const catalogStatus = providerCatalog.length
    ? `${providerCatalog.length} cached providers${routing.provider_catalog_refreshed_at ? ` - refreshed ${routing.provider_catalog_refreshed_at}` : ""}`
    : "No cached providers";
  const saveGlobalProfile = (profile: OpenRouterRoutingProfile) => updateLocal(
    routing.setting_key,
    openRouterRoutingSettingValue(routing, { globalProfile: profile })
  );
  const saveTaskOverride = (override: OpenRouterRoutingTaskOverride) => updateLocal(
    routing.setting_key,
    openRouterRoutingSettingValue(routing, { taskOverride: override })
  );
  const saveActiveProfile = (profile: OpenRouterRoutingProfile) => {
    if (!selectedOverride) {
      saveGlobalProfile(profile);
      return;
    }
    saveTaskOverride({ ...selectedOverride, profile });
  };
  const setProfileField = <K extends keyof OpenRouterRoutingProfile>(key: K, value: OpenRouterRoutingProfile[K]) => {
    saveActiveProfile({ ...activeProfile, [key]: value });
  };
  return (
    <div className="settings-stack openrouter-routing-panel openrouter-cockpit">
      <section className="settings-subsection openrouter-command-band">
        <div className="openrouter-section-head">
          <div>
            <h3>Routing Profile</h3>
            <span>{selectedProfileLabel}</span>
          </div>
          <span className="openrouter-catalog-status">{catalogStatus}</span>
        </div>
        <div className="openrouter-profile-grid">
          <label className="field-label" title={OPENROUTER_ROUTING_TOOLTIPS.profile}>
            <span>Profile</span>
            <select
              value={selectedProfile}
              disabled={disabled}
              title={OPENROUTER_ROUTING_TOOLTIPS.profile}
              aria-label="Profile"
              onChange={(event) => setSelectedProfile(event.target.value)}
            >
              <option value="global">Global</option>
              {routing.task_overrides.map((override) => (
                <option key={override.task_family} value={override.task_family}>{override.label}</option>
              ))}
            </select>
            <small className="setting-helper">{OPENROUTER_ROUTING_TOOLTIPS.profile}</small>
          </label>
          {selectedOverride ? (
            <label className="toggle-row compact-toggle openrouter-custom-toggle" title={OPENROUTER_ROUTING_TOOLTIPS.use_custom_profile}>
              <input
                type="checkbox"
                checked={selectedOverride.enabled}
                disabled={disabled}
                title={OPENROUTER_ROUTING_TOOLTIPS.use_custom_profile}
                aria-label="Use Custom Profile"
                onChange={(event) => saveTaskOverride({ ...selectedOverride, enabled: event.target.checked })}
              />
              <span>
                <strong>Use Custom Profile</strong>
                <small>{OPENROUTER_ROUTING_TOOLTIPS.use_custom_profile}</small>
              </span>
            </label>
          ) : null}
        </div>
      </section>
      <section className="settings-subsection openrouter-routing-band">
        <div className="openrouter-section-head">
          <div>
            <h3>Routing Strategy</h3>
          </div>
        </div>
        <div className="openrouter-control-grid">
        <label className="field-label" title={OPENROUTER_ROUTING_TOOLTIPS.sort}>
          <span>Sort</span>
          <select value={activeProfile.sort} disabled={controlsDisabled} title={OPENROUTER_ROUTING_TOOLTIPS.sort} aria-label="Sort" onChange={(event) => setProfileField("sort", event.target.value)}>
            {routing.sort_options.map((option) => <option key={option} value={option}>{openRouterSortLabel(option)}</option>)}
          </select>
          <small className="setting-helper">{OPENROUTER_ROUTING_TOOLTIPS.sort}</small>
        </label>
        <label className="field-label" title={OPENROUTER_ROUTING_TOOLTIPS.sort_partition}>
          <span>Partition</span>
          <select value={activeProfile.sort_partition} disabled={controlsDisabled || activeProfile.sort === "default"} title={OPENROUTER_ROUTING_TOOLTIPS.sort_partition} aria-label="Partition" onChange={(event) => setProfileField("sort_partition", event.target.value)}>
            {routing.partition_options.map((option) => <option key={option} value={option}>{openRouterPartitionLabel(option)}</option>)}
          </select>
          <small className="setting-helper">{OPENROUTER_ROUTING_TOOLTIPS.sort_partition}</small>
        </label>
        <label className="field-label" title={OPENROUTER_ROUTING_TOOLTIPS.allow_fallbacks}>
          <span>Fallbacks</span>
          <select value={allowFallbacksSelectValue(activeProfile.allow_fallbacks)} disabled={controlsDisabled} title={OPENROUTER_ROUTING_TOOLTIPS.allow_fallbacks} aria-label="Fallbacks" onChange={(event) => setProfileField("allow_fallbacks", openRouterAllowFallbacksValue(event.target.value))}>
            <option value="default">OpenRouter default</option>
            <option value="true">Allow</option>
            <option value="false">Disable</option>
          </select>
          <small className="setting-helper">{OPENROUTER_ROUTING_TOOLTIPS.allow_fallbacks}</small>
        </label>
        <label className="toggle-row compact-toggle" title={OPENROUTER_ROUTING_TOOLTIPS.require_parameters}>
          <input type="checkbox" checked={activeProfile.require_parameters} disabled={controlsDisabled} title={OPENROUTER_ROUTING_TOOLTIPS.require_parameters} aria-label="Require Parameters" onChange={(event) => setProfileField("require_parameters", event.target.checked)} />
          <span>
            <strong>Require Parameters</strong>
            <small>{OPENROUTER_ROUTING_TOOLTIPS.require_parameters}</small>
          </span>
        </label>
        </div>
      </section>
      <AdvancedSettingsSection
        title="Advanced OpenRouter routing"
        summary="Provider filters, privacy and quantization gates, performance limits, and the provider object preview."
      >
        <section className="settings-subsection openrouter-routing-band">
          <div className="openrouter-section-head">
            <div>
              <h3>Provider Filters</h3>
            </div>
          </div>
          <div className="openrouter-provider-filter-grid">
            <OpenRouterProviderFilterSetting label="Order" tooltip={OPENROUTER_ROUTING_TOOLTIPS.order} value={activeProfile.order} catalog={providerCatalog} ordered disabled={controlsDisabled} onChange={(value) => setProfileField("order", value)} />
            <OpenRouterProviderFilterSetting label="Only" tooltip={OPENROUTER_ROUTING_TOOLTIPS.only} value={activeProfile.only} catalog={providerCatalog} disabled={controlsDisabled} onChange={(value) => setProfileField("only", value)} />
            <OpenRouterProviderFilterSetting label="Ignore" tooltip={OPENROUTER_ROUTING_TOOLTIPS.ignore} value={activeProfile.ignore} catalog={providerCatalog} disabled={controlsDisabled} onChange={(value) => setProfileField("ignore", value)} />
          </div>
        </section>
        <section className="settings-subsection openrouter-routing-band">
          <div className="openrouter-section-head">
            <div>
              <h3>Privacy & Quantization</h3>
            </div>
          </div>
          <div className="openrouter-control-grid">
          <label className="field-label" title={OPENROUTER_ROUTING_TOOLTIPS.data_collection}>
            <span>Data Collection</span>
            <select value={activeProfile.data_collection} disabled={controlsDisabled} title={OPENROUTER_ROUTING_TOOLTIPS.data_collection} aria-label="Data Collection" onChange={(event) => setProfileField("data_collection", event.target.value)}>
              {routing.data_collection_options.map((option) => <option key={option} value={option}>{openRouterDataCollectionLabel(option)}</option>)}
            </select>
            <small className="setting-helper">{OPENROUTER_ROUTING_TOOLTIPS.data_collection}</small>
          </label>
          <label className="toggle-row compact-toggle" title={OPENROUTER_ROUTING_TOOLTIPS.zdr}>
            <input type="checkbox" checked={activeProfile.zdr} disabled={controlsDisabled} title={OPENROUTER_ROUTING_TOOLTIPS.zdr} aria-label="Zero Data Retention" onChange={(event) => setProfileField("zdr", event.target.checked)} />
            <span>
              <strong>Zero Data Retention</strong>
              <small>{OPENROUTER_ROUTING_TOOLTIPS.zdr}</small>
            </span>
          </label>
          <label className="toggle-row compact-toggle" title={OPENROUTER_ROUTING_TOOLTIPS.enforce_distillable_text}>
            <input type="checkbox" checked={activeProfile.enforce_distillable_text} disabled={controlsDisabled} title={OPENROUTER_ROUTING_TOOLTIPS.enforce_distillable_text} aria-label="Distillable Text" onChange={(event) => setProfileField("enforce_distillable_text", event.target.checked)} />
            <span>
              <strong>Distillable Text</strong>
              <small>{OPENROUTER_ROUTING_TOOLTIPS.enforce_distillable_text}</small>
            </span>
          </label>
          </div>
          <div className="openrouter-checkbox-grid" title={OPENROUTER_ROUTING_TOOLTIPS.quantizations}>
            {routing.quantization_options.map((option) => {
              const tooltip = openRouterQuantizationTooltip(option);
              return (
                <label key={option} className="toggle-row compact-toggle" title={tooltip}>
                  <input
                    type="checkbox"
                    checked={activeProfile.quantizations.includes(option)}
                    disabled={controlsDisabled}
                    title={tooltip}
                    aria-label={option}
                    onChange={(event) => setProfileField("quantizations", toggleString(activeProfile.quantizations, option, event.target.checked))}
                  />
                  <span>
                    <strong>{option}</strong>
                    <small>{tooltip}</small>
                  </span>
                </label>
              );
            })}
          </div>
        </section>
        <section className="settings-subsection openrouter-routing-band">
          <div className="openrouter-section-head">
            <div>
              <h3>Performance & Price</h3>
            </div>
          </div>
          <OpenRouterNumberMapSetting
            label="Min Throughput"
            tooltip={OPENROUTER_ROUTING_TOOLTIPS.preferred_min_throughput}
            keys={routing.percentile_options}
            values={activeProfile.preferred_min_throughput}
            disabled={controlsDisabled}
            tooltipForKey={(key) => openRouterPercentileTooltip("throughput", key)}
            onChange={(values) => setProfileField("preferred_min_throughput", values)}
          />
          <OpenRouterNumberMapSetting
            label="Max Latency"
            tooltip={OPENROUTER_ROUTING_TOOLTIPS.preferred_max_latency}
            keys={routing.percentile_options}
            values={activeProfile.preferred_max_latency}
            disabled={controlsDisabled}
            tooltipForKey={(key) => openRouterPercentileTooltip("latency", key)}
            onChange={(values) => setProfileField("preferred_max_latency", values)}
          />
          <OpenRouterNumberMapSetting
            label="Max Price"
            tooltip={OPENROUTER_ROUTING_TOOLTIPS.max_price}
            keys={routing.max_price_fields}
            values={activeProfile.max_price}
            disabled={controlsDisabled}
            allowZero
            tooltipForKey={openRouterMaxPriceTooltip}
            onChange={(values) => setProfileField("max_price", values)}
          />
        </section>
        <section className="settings-subsection openrouter-routing-band">
          <div className="openrouter-section-head">
            <div>
              <h3 title={OPENROUTER_ROUTING_TOOLTIPS.provider_object}>Provider Object</h3>
              <span>{OPENROUTER_ROUTING_TOOLTIPS.provider_object}</span>
            </div>
          </div>
          <pre className="json-editor compact-json-editor openrouter-preview" title={OPENROUTER_ROUTING_TOOLTIPS.provider_object}>{JSON.stringify(effectivePayload, null, 2)}</pre>
        </section>
      </AdvancedSettingsSection>
    </div>
  );
}

function OpenRouterProviderFilterSetting({
  label,
  tooltip,
  value,
  catalog,
  ordered = false,
  disabled,
  onChange
}: {
  label: string;
  tooltip: string;
  value: string[];
  catalog: OpenRouterProviderCatalogEntry[];
  ordered?: boolean;
  disabled: boolean;
  onChange: (value: string[]) => void;
}) {
  const [query, setQuery] = useState("");
  const catalogBySlug = useMemo(() => new Map(catalog.map((entry) => [entry.slug, entry])), [catalog]);
  const suggestions = matchingOpenRouterProviders(catalog, value, query).slice(0, 6);
  const customSlug = sanitizeOpenRouterProviderSlug(query);
  const customAvailable = Boolean(customSlug && !value.includes(customSlug) && !catalogBySlug.has(customSlug));
  const addProvider = (slug: string) => {
    if (disabled || value.includes(slug)) return;
    onChange([...value, slug]);
    setQuery("");
  };
  const removeProvider = (slug: string) => {
    if (disabled) return;
    onChange(value.filter((item) => item !== slug));
  };
  const moveProvider = (index: number, direction: -1 | 1) => {
    if (disabled) return;
    const nextIndex = index + direction;
    if (nextIndex < 0 || nextIndex >= value.length) return;
    const next = [...value];
    [next[index], next[nextIndex]] = [next[nextIndex], next[index]];
    onChange(next);
  };
  return (
    <div className="openrouter-provider-filter" title={tooltip}>
      <div className="openrouter-provider-filter-head">
        <span>{label}</span>
        <small>{value.length ? `${value.length} selected` : "No filter"}</small>
      </div>
      <small className="setting-helper">{tooltip}</small>
      <div className="openrouter-provider-search">
        <Search size={14} aria-hidden="true" />
        <input
          value={query}
          disabled={disabled}
          aria-label={`${label} provider search`}
          autoComplete="off"
          placeholder="Search or type slug"
          spellCheck={false}
          title={tooltip}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter") return;
            event.preventDefault();
            if (suggestions[0]) addProvider(suggestions[0].slug);
            else if (customAvailable && customSlug) addProvider(customSlug);
          }}
        />
        <button
          type="button"
          aria-label={`Clear ${label} provider search`}
          title={`Clear ${label} provider search`}
          disabled={disabled || !query}
          onClick={() => setQuery("")}
        >
          <X size={13} />
        </button>
      </div>
      <div className="openrouter-provider-suggestions">
        {suggestions.map((entry) => (
          <button
            key={entry.slug}
            type="button"
            className="openrouter-provider-suggestion"
            disabled={disabled}
            aria-label={`Add ${entry.name} to ${label}`}
            title={`${entry.slug}${entry.datacenters.length ? ` - ${entry.datacenters.join(", ")}` : ""}`}
            onClick={() => addProvider(entry.slug)}
          >
            <span>{entry.name}</span>
            <small>{entry.slug}</small>
          </button>
        ))}
        <button
          type="button"
          className="openrouter-provider-suggestion custom"
          disabled={disabled || !customAvailable || !customSlug}
          aria-label={`Add custom provider to ${label}`}
          title={customSlug ? `Add ${customSlug}` : "Type a provider slug to add it"}
          onClick={() => customSlug && addProvider(customSlug)}
        >
          <span>Custom slug</span>
          <small>{customSlug || "provider/variant"}</small>
        </button>
      </div>
      <div className="openrouter-provider-chip-list">
        {value.length ? value.map((slug, index) => {
          const entry = catalogBySlug.get(slug);
          const name = entry?.name ?? slug;
          return (
            <span key={`${slug}:${index}`} className={entry ? "openrouter-provider-chip" : "openrouter-provider-chip custom"}>
              <span className="openrouter-provider-chip-text">
                <strong>{name}</strong>
                <small>{entry ? slug : "Custom slug"}</small>
              </span>
              {ordered ? (
                <>
                  <button type="button" disabled={disabled || index === 0} aria-label={`Move ${name} up in ${label}`} title={`Move ${name} up`} onClick={() => moveProvider(index, -1)}>
                    <ArrowUp size={12} />
                  </button>
                  <button type="button" disabled={disabled || index === value.length - 1} aria-label={`Move ${name} down in ${label}`} title={`Move ${name} down`} onClick={() => moveProvider(index, 1)}>
                    <ArrowDown size={12} />
                  </button>
                </>
              ) : null}
              <button type="button" disabled={disabled} aria-label={`Remove ${name} from ${label}`} title={`Remove ${name}`} onClick={() => removeProvider(slug)}>
                <X size={12} />
              </button>
            </span>
          );
        }) : (
          <span className="openrouter-provider-empty">OpenRouter default</span>
        )}
      </div>
    </div>
  );
}

function OpenRouterNumberMapSetting({
  label,
  tooltip,
  keys,
  values,
  disabled,
  allowZero = false,
  tooltipForKey,
  onChange
}: {
  label: string;
  tooltip: string;
  keys: string[];
  values: Record<string, number>;
  disabled: boolean;
  allowZero?: boolean;
  tooltipForKey: (key: string) => string;
  onChange: (values: Record<string, number>) => void;
}) {
  return (
    <div className="openrouter-number-group" title={tooltip}>
      <span title={tooltip}>{label}</span>
      <small className="setting-helper">{tooltip}</small>
      <div className="openrouter-number-grid">
        {keys.map((key) => {
          const keyTooltip = tooltipForKey(key);
          return (
            <label key={key} className="field-label" title={keyTooltip}>
              <span>{labelize(key)}</span>
              <OptionalRoutingNumberInput
                value={values[key] ?? null}
                disabled={disabled}
                allowZero={allowZero}
                title={keyTooltip}
                ariaLabel={`${label} ${labelize(key)}`}
                onCommit={(value) => onChange(openRouterNumberMapWithValue(values, key, value))}
              />
              <small className="setting-helper">{keyTooltip}</small>
            </label>
          );
        })}
      </div>
    </div>
  );
}

function OptionalRoutingNumberInput({
  value,
  disabled,
  allowZero,
  title,
  ariaLabel,
  onCommit
}: {
  value: number | null;
  disabled: boolean;
  allowZero: boolean;
  title: string;
  ariaLabel: string;
  onCommit: (value: number | null) => void;
}) {
  const [draft, setDraft] = useState(value === null ? "" : String(value));
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setDraft(value === null ? "" : String(value));
  }, [focused, value]);
  const commit = () => {
    setFocused(false);
    const text = draft.trim();
    if (!text) {
      if (value !== null) onCommit(null);
      return;
    }
    const parsed = Number(text);
    if (!Number.isFinite(parsed) || parsed < 0 || (!allowZero && parsed === 0)) {
      setDraft(value === null ? "" : String(value));
      return;
    }
    if (parsed !== value) onCommit(parsed);
  };
  return (
    <input
      type="number"
      min={allowZero ? 0 : 0.000001}
      step="any"
      value={draft}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      onFocus={() => setFocused(true)}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commit}
    />
  );
}

function openRouterRoutingSettingValue(
  routing: OpenRouterRoutingSettingsModel,
  changes: {
    globalProfile?: OpenRouterRoutingProfile;
    taskOverride?: OpenRouterRoutingTaskOverride;
  } = {}
) {
  const taskOverrides = Object.fromEntries(
    routing.task_overrides.map((override) => {
      const next = changes.taskOverride?.task_family === override.task_family ? changes.taskOverride : override;
      return [next.task_family, { enabled: next.enabled, profile: next.profile }];
    })
  );
  return {
    global: changes.globalProfile ?? routing.global_profile,
    task_overrides: taskOverrides
  };
}

function openRouterNumberMapWithValue(values: Record<string, number>, key: string, value: number | null) {
  const next = { ...values };
  if (value === null) delete next[key];
  else next[key] = value;
  return next;
}

function toggleString(values: string[], value: string, enabled: boolean) {
  if (enabled) return values.includes(value) ? values : [...values, value];
  return values.filter((item) => item !== value);
}

function sanitizeOpenRouterProviderSlug(value: string) {
  const slug = value.trim().toLowerCase();
  if (!slug || slug.length > 128) return null;
  return /^[a-z0-9._/-]+$/.test(slug) ? slug : null;
}

function matchingOpenRouterProviders(
  catalog: OpenRouterProviderCatalogEntry[],
  selected: string[],
  query: string
) {
  const selectedSlugs = new Set(selected);
  const normalizedQuery = query.trim().toLowerCase();
  const tokens = normalizedQuery.split(/\s+/).filter(Boolean);
  return catalog.filter((entry) => {
    if (selectedSlugs.has(entry.slug)) return false;
    if (!tokens.length) return true;
    const fields = [
      entry.slug,
      entry.name,
      entry.headquarters ?? "",
      ...(entry.datacenters ?? [])
    ].map((field) => field.toLowerCase());
    return tokens.every((token) => fields.some((field) => field.includes(token)));
  });
}

function allowFallbacksSelectValue(value: boolean | null) {
  if (value === null) return "default";
  return value ? "true" : "false";
}

function openRouterAllowFallbacksValue(value: string) {
  if (value === "true") return true;
  if (value === "false") return false;
  return null;
}

function openRouterSortLabel(option: string) {
  const labels: Record<string, string> = {
    default: "OpenRouter default",
    price: "Price",
    throughput: "Throughput",
    latency: "Latency"
  };
  return labels[option] ?? labelize(option);
}

function openRouterPartitionLabel(option: string) {
  const labels: Record<string, string> = {
    model: "Model",
    none: "None"
  };
  return labels[option] ?? labelize(option);
}

function openRouterDataCollectionLabel(option: string) {
  const labels: Record<string, string> = {
    allow: "Allow",
    deny: "Deny"
  };
  return labels[option] ?? labelize(option);
}

function openRouterQuantizationTooltip(option: string) {
  return OPENROUTER_QUANTIZATION_TOOLTIPS[option] ?? `Limit routing to providers that report ${option} quantization.`;
}

function openRouterPercentileTooltip(kind: "throughput" | "latency", percentile: string) {
  const percentileText = `${labelize(percentile)} means the ${percentile.slice(1)}th percentile OpenRouter performance metric.`;
  if (kind === "throughput") {
    return `${percentileText} Prefer providers at or above this throughput in tokens per second.`;
  }
  return `${percentileText} Prefer providers at or below this latency in seconds.`;
}

function openRouterMaxPriceTooltip(field: string) {
  return OPENROUTER_MAX_PRICE_TOOLTIPS[field] ?? `Maximum ${labelize(field).toLowerCase()} price OpenRouter may route to.`;
}

function commonModelOptions(selectors: TaskModelSelector[], capabilities: readonly ModelCapabilityFamily[]): ModelOption[] {
  const [first, ...rest] = selectors;
  if (!first) return [];
  const firstOptions = uniqueAvailableModelOptions(first, capabilities);
  const commonKeys = new Set(firstOptions.map(modelOptionValue));
  for (const selector of rest) {
    const selectorKeys = new Set(uniqueAvailableModelOptions(selector, capabilities).map(modelOptionValue));
    for (const key of Array.from(commonKeys)) {
      if (!selectorKeys.has(key)) commonKeys.delete(key);
    }
  }
  return firstOptions.filter((option) => commonKeys.has(modelOptionValue(option)));
}

function uniqueAvailableModelOptions(selector: TaskModelSelector, capabilities: readonly ModelCapabilityFamily[]): ModelOption[] {
  const seen = new Set<string>();
  const options: ModelOption[] = [];
  for (const option of selector.options) {
    const key = modelOptionValue(option);
    if (!option.available || seen.has(key) || !modelOptionHasAnyCapability(option, capabilities)) continue;
    seen.add(key);
    options.push(option);
  }
  return options;
}

function commonSelectedModelValue(selectors: TaskModelSelector[], options: ModelOption[]): string {
  const [first, ...rest] = selectors;
  if (!first) return "";
  const selectedValue = modelPreferenceValue(first.selected_provider, first.selected_model_id);
  if (!selectedValue || !options.some((option) => modelOptionValue(option) === selectedValue)) return "";
  return rest.every((selector) => modelPreferenceValue(selector.selected_provider, selector.selected_model_id) === selectedValue) ? selectedValue : "";
}

function commonSelectionPlaceholder(selectors: TaskModelSelector[], commonSelected: string): string {
  if (commonSelected) return "Choose model";
  const selectedValues = selectors
    .map((selector) => modelPreferenceValue(selector.selected_provider, selector.selected_model_id))
    .filter(Boolean);
  return new Set(selectedValues).size > 1 ? "Mixed selections" : "Choose model";
}

function capabilityRequirementLabel(capabilities: readonly ModelCapabilityFamily[]): string {
  const normalized = new Set(capabilities);
  if (normalized.has("structured_output") && normalized.has("tool_calling") && normalized.size === 2) {
    return "structured output or tool calling";
  }
  return capabilities.map(capabilityLabel).join(" or ");
}

function modelOptionHasAnyCapability(option: ModelOption, capabilities: readonly ModelCapabilityFamily[]): boolean {
  return capabilities.some((capability) => modelOptionHasCapability(option, capability));
}

function modelOptionHasCapability(option: ModelOption, capability: ModelCapabilityFamily): boolean {
  const aliases = new Set(MODEL_CAPABILITY_ALIASES[capability]);
  return option.capabilities.some((value) => aliases.has(normalizedCapability(value)));
}

function normalizedCapability(value: string): string {
  return value.trim().toLowerCase().replace(/-/g, "_");
}

function modelPreferenceValue(provider: string | null, modelId: string | null) {
  return provider && modelId ? `${provider}\u0000${modelId}` : "";
}

function modelPreferenceFromValue(value: string): { provider: string; model_id: string } | null {
  const [provider, model_id] = value.split("\u0000");
  return provider && model_id ? { provider, model_id } : null;
}

function modelPreferencePath(task: string, saveId?: string | null) {
  const base = `/api/settings/model-preference/${encodeURIComponent(task)}`;
  return saveId ? `${base}?save_id=${encodeURIComponent(saveId)}` : base;
}

function modelThinkingPreferencePath(task: string, saveId?: string | null) {
  const base = `/api/settings/model-thinking/${encodeURIComponent(task)}`;
  return saveId ? `${base}?save_id=${encodeURIComponent(saveId)}` : base;
}

function modelOptionValue(option: ModelOption) {
  return modelPreferenceValue(option.provider, option.model_id);
}

function modelOptionForValue(options: ModelOption[], value: string): ModelOption | undefined {
  return value ? options.find((option) => modelOptionValue(option) === value) : undefined;
}

function thinkingControlForModelOption(
  option: ModelOption | undefined,
  selected: string,
  task: string
): ThinkingLevelControl {
  const support = option?.thinking ?? null;
  if (!option || !support || !support.levels.length) {
    return {
      setting_key: "model_thinking_preferences",
      task,
      selected: THINKING_LEVEL_PROVIDER_DEFAULT,
      supported: false,
      options: [THINKING_LEVEL_PROVIDER_DEFAULT],
      provider: option?.provider ?? null,
      model_id: option?.model_id ?? null,
      disabled_reason: option ? "Selected model does not support thinking level" : "Choose a model first"
    };
  }
  const options = [
    THINKING_LEVEL_PROVIDER_DEFAULT,
    ...(support.mandatory ? [] : [THINKING_LEVEL_OFF]),
    ...support.levels
  ];
  return {
    setting_key: "model_thinking_preferences",
    task,
    selected: options.includes(selected) ? selected : THINKING_LEVEL_PROVIDER_DEFAULT,
    supported: true,
    options,
    provider: option.provider,
    model_id: option.model_id,
    default_level: support.default_level,
    default_enabled: support.default_enabled,
    mandatory: support.mandatory,
    disabled_reason: null
  };
}

function commonSelectedThinkingValue(selectors: TaskModelSelector[], selectedValue: string): string {
  const [provider, model_id] = selectedValue.split("\u0000");
  if (!provider || !model_id) return THINKING_LEVEL_PROVIDER_DEFAULT;
  const values = selectors.map((selector) => {
    if (selector.selected_provider !== provider || selector.selected_model_id !== model_id) {
      return THINKING_LEVEL_PROVIDER_DEFAULT;
    }
    return selector.thinking?.selected ?? THINKING_LEVEL_PROVIDER_DEFAULT;
  });
  const [first, ...rest] = values;
  return first && rest.every((value) => value === first) ? first : THINKING_LEVEL_PROVIDER_DEFAULT;
}

function matchingModelOptions(options: ModelOption[], query: string): ModelOption[] {
  const normalizedQuery = normalizeModelSearchText(query);
  if (!normalizedQuery) return options;
  const tokens = modelSearchTokens(normalizedQuery);
  return options
    .map((option, index) => ({
      option,
      index,
      score: modelOptionSearchScore(option, normalizedQuery, tokens)
    }))
    .filter((result): result is { option: ModelOption; index: number; score: number } => result.score !== null)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .map((result) => result.option);
}

function modelOptionSearchScore(option: ModelOption, normalizedQuery: string, tokens: string[]): number | null {
  const [displayName, provider, modelId, ...capabilities] = modelOptionSearchFields(option);
  const searchText = [displayName, provider, modelId, ...capabilities].join(" ");
  const compactText = [displayName, provider, modelId, ...capabilities].join("");
  if (!tokens.every((token) => searchText.includes(token) || compactText.includes(token))) return null;

  let score = searchText.includes(normalizedQuery) ? 50 : 0;
  for (const token of tokens) {
    if (displayName.startsWith(token)) score += 24;
    else if (displayName.includes(token)) score += 16;

    if (modelId.startsWith(token)) score += 18;
    else if (modelId.includes(token)) score += 12;

    if (provider.startsWith(token)) score += 10;
    else if (provider.includes(token)) score += 6;

    if (capabilities.some((capability) => capability.includes(token))) score += 3;
  }
  return score;
}

function modelOptionSearchFields(option: ModelOption): string[] {
  return [
    option.display_name,
    option.provider,
    option.model_id,
    modelPricingCompactLabel(option.pricing) ?? "",
    modelPricingDisplayLabel(option.pricing) ?? "",
    ...option.capabilities
  ].map(normalizeModelSearchText);
}

function normalizeModelSearchText(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function modelSearchTokens(query: string): string[] {
  return query.split(/\s+/).filter(Boolean);
}

function modelOptionMatchLabel(count: number): string {
  return count === 1 ? "1 match" : `${count} matches`;
}

function routingLaneTaskLabel(selector: TaskModelSelector): string {
  if (selector.label) return selector.label;
  const roleplayTask = roleplayTaskParts(selector.task);
  if (!roleplayTask) return taskLabel(selector.task);
  const baseLabel = fallbackTaskLabels[roleplayTask.baseTask] ?? labelize(roleplayTask.baseTask);
  return `${roleplayTask.roleplayLabel} ${baseLabel}`;
}

function roleplayTaskParts(task: string): { roleplayLabel: string; baseTask: string } | null {
  const chatTasks: Record<string, string> = {
    chat_full_roleplay: "Full Roleplay",
    chat_fantasy_roleplay: "Fantasy Roleplay",
    chat_science_fiction_roleplay: "Science Fiction Roleplay",
    chat_first_contact_exploration: "First Contact / Exploration",
    chat_survival_expedition: "Survival Expedition",
    chat_time_loop: "Time Loop",
    chat_investigation_mystery: "Investigation Mystery",
    chat_heist_infiltration: "Heist / Infiltration",
    chat_political_intrigue: "Political Intrigue",
    chat_dating_sim: "Dating Sim"
  };
  const chatRoleplay = chatTasks[task];
  if (chatRoleplay) return { roleplayLabel: chatRoleplay, baseTask: "chat" };
  for (const [prefix, roleplayLabel] of ROLEPLAY_TASK_PREFIX_LABELS) {
    if (task.startsWith(prefix)) return { roleplayLabel, baseTask: task.slice(prefix.length) };
  }
  return null;
}

const ROLEPLAY_TASK_PREFIX_LABELS: readonly [string, string][] = [
  ["full_roleplay_", "Full Roleplay"],
  ["fantasy_roleplay_", "Fantasy Roleplay"],
  ["science_fiction_roleplay_", "Science Fiction Roleplay"],
  ["first_contact_exploration_", "First Contact / Exploration"],
  ["survival_expedition_", "Survival Expedition"],
  ["time_loop_", "Time Loop"],
  ["investigation_mystery_", "Investigation Mystery"],
  ["heist_infiltration_", "Heist / Infiltration"],
  ["political_intrigue_", "Political Intrigue"],
  ["dating_sim_", "Dating Sim"],
  ["shared_roleplay_", "Shared Roleplay"]
];

function SaveSettingsControls({
  settings,
  activeSaveId,
  storytellerMode,
  updateLocal,
  disabled,
  runJob,
  canRunMaintenance
}: {
  settings?: SettingsModel;
  activeSaveId: string | null;
  storytellerMode: boolean;
  updateLocal: (key: string, value: unknown) => void;
  disabled: boolean;
  runJob: RunJob;
  canRunMaintenance: boolean;
}) {
  const [applySummaryWindows, setApplySummaryWindows] = useState(false);
  const saveDisabled = disabled || !activeSaveId;
  const summaryBackfillDisabled = saveDisabled || !canRunMaintenance;
  if (!settings) return null;
  return (
    <div className="settings-stack">
      {!activeSaveId ? <p className="muted">Load a save to edit save options.</p> : null}
      <SaveModelOverrideSettings settings={settings} activeSaveId={activeSaveId} />
      <section className="settings-subsection">
        <h3>Summarization</h3>
        {settings.automatic_summarization ? <ToggleSetting control={settings.automatic_summarization} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
        {settings.summarization_context_pressure_threshold ? <NumberSetting control={settings.summarization_context_pressure_threshold} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
        {settings.summarization_visibility ? <ToggleSetting control={settings.summarization_visibility} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
        <label className="toggle-row compact-toggle" title="Also reduce save-scoped recent chat history windows to the recommended compact defaults after the backfill succeeds.">
          <input
            type="checkbox"
            checked={applySummaryWindows}
            disabled={summaryBackfillDisabled}
            onChange={(event) => setApplySummaryWindows(event.target.checked)}
          />
          <span>Apply Recommended Chat History Windows</span>
        </label>
        <button
          type="button"
          className="primary-command compact"
          disabled={summaryBackfillDisabled}
          title={activeSaveId ? "Summarize older chronicle messages into one active save summary." : "Load a save before compacting history."}
          onClick={async () => {
            if (!activeSaveId) return;
            runJob(await postJson<Job>("/api/world-data/summary-backfill", {
              save_id: activeSaveId,
              apply_recommended_windows: applySummaryWindows
            }));
          }}
        >
          <Archive size={15} /> Compact History
        </button>
      </section>
      {settings.agentic_context_pipeline ? (
        <section className="settings-subsection">
          <h3>Context Automation</h3>
          <ToggleSetting control={settings.agentic_context_pipeline} disabled={saveDisabled} updateLocal={updateLocal} />
          {settings.plan_first_narrator ? (
            <ToggleSetting control={settings.plan_first_narrator} disabled={saveDisabled} updateLocal={updateLocal} />
          ) : null}
          {settings.director_pressure ? (
            <ToggleSetting control={settings.director_pressure} disabled={saveDisabled} updateLocal={updateLocal} />
          ) : null}
          {settings.director_pressure_guidance ? (
            <TextSetting control={settings.director_pressure_guidance} disabled={saveDisabled} updateLocal={updateLocal} />
          ) : null}
          {settings.character_action_planning ? (
            <ToggleSetting control={settings.character_action_planning} disabled={saveDisabled} updateLocal={updateLocal} />
          ) : null}
          {settings.character_action_planning_max_concurrency ? (
            <NumberSetting control={settings.character_action_planning_max_concurrency} disabled={saveDisabled} updateLocal={updateLocal} />
          ) : null}
          {!storytellerMode && settings.character_texts ? (
            <ToggleSetting control={settings.character_texts} disabled={saveDisabled} updateLocal={updateLocal} />
          ) : null}
          {!storytellerMode && settings.character_text_proactive_random_chance ? (
            <NumberSetting control={settings.character_text_proactive_random_chance} disabled={saveDisabled} updateLocal={updateLocal} />
          ) : null}
          {!storytellerMode && settings.character_text_proactive_random_cooldown ? (
            <NumberSetting control={settings.character_text_proactive_random_cooldown} disabled={saveDisabled} updateLocal={updateLocal} />
          ) : null}
          {settings.post_turn_inference_mode ? (
            <ChoiceSetting control={settings.post_turn_inference_mode} disabled={saveDisabled} updateLocal={updateLocal} optionLabel={postTurnInferenceModeLabel} />
          ) : null}
        </section>
      ) : null}
      <section className="settings-subsection">
        <h3>Media Automation</h3>
        {settings.automatic_image_generation ? <ToggleSetting control={settings.automatic_image_generation} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
        {settings.automatic_media_mode ? <ChoiceSetting control={settings.automatic_media_mode} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
        {settings.image_style_preset ? (
          <ChoiceSetting
            control={settings.image_style_preset}
            disabled={saveDisabled}
            updateLocal={updateLocal}
            optionLabel={imageStylePresetLabel}
          />
        ) : null}
        {settings.image_frequency ? <NumberSetting control={settings.image_frequency} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
      </section>
      <section className="settings-subsection">
        <h3>Provider Generation</h3>
        {settings.chat_temperature ? <OptionalNumberSetting control={settings.chat_temperature} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
        {settings.chat_max_output_tokens ? <OptionalNumberSetting control={settings.chat_max_output_tokens} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
        {settings.image_dimension_preset ? <SupportedChoiceSetting control={settings.image_dimension_preset} disabled={saveDisabled} updateLocal={updateLocal} optionLabel={imageDimensionPresetLabel} /> : null}
      </section>
      <section className="settings-subsection">
        <h3>Safety</h3>
        {settings.npc_knowledge_audit_mode ? <ChoiceSetting control={settings.npc_knowledge_audit_mode} disabled={saveDisabled} updateLocal={updateLocal} optionLabel={npcKnowledgeAuditModeLabel} /> : null}
        {settings.generated_text_script_guard_mode ? <ChoiceSetting control={settings.generated_text_script_guard_mode} disabled={saveDisabled} updateLocal={updateLocal} optionLabel={scriptGuardModeLabel} /> : null}
        {settings.generated_phrase_denylist ? <TextSetting control={settings.generated_phrase_denylist} disabled={disabled} updateLocal={updateLocal} /> : null}
        {settings.save_generated_phrase_denylist ? <TextSetting control={settings.save_generated_phrase_denylist} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
        {settings.venice_image_safe_mode ? <ToggleSetting control={settings.venice_image_safe_mode} disabled={saveDisabled} updateLocal={updateLocal} /> : null}
      </section>
      <section className="settings-subsection">
        <h3>Chat History</h3>
        {settings.chat_history ? (
          <>
            <h4>Narrator Planner</h4>
            <NumberSetting control={settings.chat_history.planner_player_messages} disabled={saveDisabled} updateLocal={updateLocal} />
            <NumberSetting control={settings.chat_history.planner_narrator_messages} disabled={saveDisabled} updateLocal={updateLocal} />
            <h4>Narrator Prose</h4>
            <NumberSetting control={settings.chat_history.player_messages} disabled={saveDisabled} updateLocal={updateLocal} />
            <NumberSetting control={settings.chat_history.narrator_messages} disabled={saveDisabled} updateLocal={updateLocal} />
          </>
        ) : null}
      </section>
      <section className="settings-subsection">
        <h3>Context Budget</h3>
        {settings.context_budget ? (
          <>
            <ChoiceSetting control={settings.context_budget.mode} disabled={saveDisabled} updateLocal={updateLocal} />
            <NumberSetting control={settings.context_budget.fixed_total_chars} disabled={saveDisabled} updateLocal={updateLocal} />
            <NumberSetting control={{ ...settings.context_budget.adaptive_fraction, step: 0.01 }} disabled={saveDisabled} updateLocal={updateLocal} />
          </>
        ) : null}
      </section>
      {settings.manual_confirmation ? (
        <section className="settings-subsection">
          <h3>Manual Confirmation</h3>
          <ToggleSetting control={settings.manual_confirmation.memories} disabled={saveDisabled} updateLocal={updateLocal} />
          <ToggleSetting control={settings.manual_confirmation.character_registry} disabled={saveDisabled} updateLocal={updateLocal} />
          <ToggleSetting control={settings.manual_confirmation.state_changes} disabled={saveDisabled} updateLocal={updateLocal} />
        </section>
      ) : null}
    </div>
  );
}

function SaveModelOverrideSettings({ settings, activeSaveId }: { settings: SettingsModel; activeSaveId: string | null }) {
  const selectors = settings.save_model_override_selectors ?? [];
  if (!activeSaveId || !selectors.length) return null;
  const overrideSettings: SettingsModel = {
    ...settings,
    task_model_selectors: selectors,
    roleplay_model_groups: [],
    scenario_section_model_selectors: []
  };
  const groups = modelRoutingLaneGroups(overrideSettings, MODEL_ROUTING_GROUPS);
  const fallbacks = modelRoutingLanes(overrideSettings, MODEL_FALLBACK_LANES);
  const routedTasks = routedModelTaskSet(groups, fallbacks);
  const otherSelectors = selectors.filter((selector) => !routedTasks.has(selector.task));
  return (
    <AdvancedSettingsSection
      title="Model Overrides"
      summary="Override inherited server routing for this save."
    >
      <ModelRoutingSettings groups={groups} fallbacks={fallbacks} saveId={activeSaveId} />
      {otherSelectors.length ? (
        <section className="settings-subsection">
          <h3>Other Model Tasks</h3>
          <ModelSelectorGroups selectors={otherSelectors} emptyLabel="No other model tasks" saveId={activeSaveId} />
        </section>
      ) : null}
    </AdvancedSettingsSection>
  );
}

function LocalSettingsControls({ settings, updateLocal, disabled }: { settings?: LocalSettingsModel; updateLocal: (key: string, value: unknown) => void; disabled: boolean }) {
  if (!settings) return null;
  const unrated = settings.content_rating?.selected === "unrated";
  return (
    <div className="settings-stack">
      <section className="settings-subsection">
        <h3>Workbench</h3>
        {settings.pending_jobs_display_mode ? <ChoiceSetting control={settings.pending_jobs_display_mode} disabled={disabled} updateLocal={updateLocal} optionLabel={pendingJobsDisplayModeLabel} /> : null}
        {settings.user_narration_guidance ? <TextSetting control={settings.user_narration_guidance} disabled={disabled} updateLocal={updateLocal} /> : null}
      </section>
      {settings.content_rating || settings.fade_to_black ? (
        <section className="settings-subsection">
          <h3>Content Safety</h3>
          {settings.content_rating ? <ChoiceSetting control={settings.content_rating} disabled={disabled} updateLocal={updateLocal} optionLabel={contentRatingLabel} /> : null}
          {settings.fade_to_black ? <ToggleSetting control={settings.fade_to_black} disabled={disabled || unrated} updateLocal={updateLocal} /> : null}
          {unrated ? <p className="muted">Unrated skips Safety Agent review and never triggers fade-to-black.</p> : null}
        </section>
      ) : null}
      {settings.debug_logging ? (
        <section className="settings-subsection">
          <h3>Local Recorder</h3>
          <ToggleSetting control={settings.debug_logging} disabled={disabled} updateLocal={updateLocal} />
        </section>
      ) : null}
    </div>
  );
}

function DiagnosticsSettings({ activeSaveId, isAdmin }: { activeSaveId: string | null; isAdmin: boolean }) {
  const [filters, setFilters] = useState<DiagnosticsFilters>({ ...EMPTY_DIAGNOSTICS_FILTERS });
  const [bundleStatus, setBundleStatus] = useState("");
  const diagnostics = useQuery(diagnosticsQueryOptions(activeSaveId, filters));
  const report = diagnostics.data;
  const summary = diagnosticsCockpitSummary(report);
  const HealthIcon = summary.health.icon;
  const updateFilter = (key: keyof DiagnosticsFilters, value: string) => {
    setFilters((current) => ({ ...current, [key]: value }));
    setBundleStatus("");
  };
  const supportBundleText = report ? JSON.stringify(report, null, 2) : "";
  const copySupportBundle = async () => {
    if (!supportBundleText || !navigator.clipboard?.writeText) return;
    await navigator.clipboard.writeText(supportBundleText);
    setBundleStatus("Copied");
  };
  const downloadSupportBundle = () => {
    if (!supportBundleText) return;
    const blob = new Blob([supportBundleText], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = diagnosticsBundleFilename(report?.generated_at);
    link.click();
    URL.revokeObjectURL(url);
    setBundleStatus("Downloaded");
  };
  return (
    <section className="diagnostics-cockpit" aria-label="Diagnostics ops cockpit">
      <header className={`diagnostics-hero ${summary.health.tone}`}>
        <span className="diagnostics-hero-icon" aria-hidden="true">
          <HealthIcon size={20} />
        </span>
        <div>
          <span className="diagnostics-eyebrow">Ops Cockpit</span>
          <h3>Diagnostics</h3>
          <p>{summary.health.detail}</p>
        </div>
        <strong className={`diagnostics-health-pill ${summary.health.tone}`}>{summary.health.label}</strong>
      </header>
      <div className="diagnostics-toolbar">
        <label>
          <span>Request ID</span>
          <input value={filters.request_id} aria-label="Request ID filter" onChange={(event) => updateFilter("request_id", event.target.value)} />
        </label>
        <label>
          <span>Job ID</span>
          <input value={filters.job_id} aria-label="Job ID filter" onChange={(event) => updateFilter("job_id", event.target.value)} />
        </label>
        <label>
          <span>Route</span>
          <input value={filters.route} aria-label="Route filter" onChange={(event) => updateFilter("route", event.target.value)} />
        </label>
        <label>
          <span>Component</span>
          <input value={filters.component} aria-label="Component filter" onChange={(event) => updateFilter("component", event.target.value)} />
        </label>
        <label>
          <span>Since</span>
          <input value={filters.since} aria-label="Since filter" placeholder="2026-07-09T12:00:00Z" onChange={(event) => updateFilter("since", event.target.value)} />
        </label>
        <label>
          <span>Limit</span>
          <input value={filters.limit} aria-label="Limit filter" inputMode="numeric" onChange={(event) => updateFilter("limit", event.target.value)} />
        </label>
        <button type="button" className="icon-button" title="Refresh diagnostics" aria-label="Refresh diagnostics" onClick={() => void diagnostics.refetch()}>
          <RefreshCw size={16} />
        </button>
        <button type="button" className="secondary-button" disabled={!report} onClick={() => void copySupportBundle()}>
          <FileText size={15} /> Copy support bundle
        </button>
        <button type="button" className="secondary-button" disabled={!report} onClick={downloadSupportBundle}>
          <Download size={15} /> Download
        </button>
        {bundleStatus ? <small>{bundleStatus}</small> : null}
      </div>
      {diagnostics.isLoading ? <p className="muted">Loading diagnostics...</p> : null}
      {diagnostics.error instanceof Error ? <InlineNotice>{diagnostics.error.message}</InlineNotice> : null}
      <div className="diagnostic-stat-grid">
        <DiagnosticStat icon={FileWarning} label="Signals" value={summary.signalCount} detail={summary.signalDetail} tone={summary.signalIssueCount ? "attention" : summary.signalCount ? "neutral" : "healthy"} />
        <DiagnosticStat icon={GitBranch} label="Failed Jobs" value={summary.failedJobCount} detail={summary.jobDetail} tone={summary.failedJobCount ? "attention" : summary.cancelledJobCount ? "warning" : "healthy"} />
        <DiagnosticStat icon={History} label="Error Events" value={summary.errorEventCount} detail={summary.eventDetail} tone={summary.errorEventCount ? "attention" : summary.warningEventCount ? "warning" : "healthy"} />
        <DiagnosticStat icon={Clock} label="Perf Rows" value={summary.performanceRowCount} detail={summary.performanceDetail} tone={summary.failedOperationCount ? "attention" : "neutral"} />
        <DiagnosticStat icon={Check} label="Save Health" value={summary.saveHealthWarningCount} detail={summary.saveHealthDetail} tone={summary.saveHealthWarningCount ? "warning" : summary.hasActiveSaveHealth ? "healthy" : "neutral"} />
        <DiagnosticStat icon={GitBranch} label="Scheduler" value={summary.schedulerIssueCount} detail={summary.schedulerDetail} tone={summary.schedulerIssueCount ? "attention" : summary.schedulerTaskCount ? "healthy" : "neutral"} />
      </div>
      <section className="diagnostic-panel">
        <div className="diagnostic-section-header">
          <span>Active Save Health</span>
          <small>Continuity and prompt assembly metadata for the current save</small>
        </div>
        <ActiveSaveHealthPanel health={report?.active_save_health ?? null} />
      </section>
      <section className="diagnostic-panel">
        <div className="diagnostic-section-header">
          <span>Signal Board</span>
          <small>Provider, storage, and configuration warnings</small>
        </div>
        <DiagnosticsList diagnostics={report?.signals ?? []} isAdmin={isAdmin} />
      </section>
      <section className="diagnostic-panel">
        <div className="diagnostic-section-header">
          <span>Scheduler Health</span>
          <small>Persisted maintenance task state</small>
        </div>
        <SchedulerHealthList report={report?.scheduler_health} isAdmin={isAdmin} />
      </section>
      <section className="diagnostic-panel">
        <div className="diagnostic-section-header">
          <span>Performance</span>
          <small>Recent runtime averages by job, step, and model</small>
        </div>
        <RuntimePerformanceList report={report?.runtime_performance ?? undefined} />
      </section>
      <section className="diagnostic-panel">
        <div className="diagnostic-section-header">
          <span>Job Failures</span>
          <small>Terminal maintenance work that needs attention</small>
        </div>
        <MaintenanceJobsList jobs={report?.maintenance_jobs ?? []} isAdmin={isAdmin} />
      </section>
      <section className="diagnostic-panel">
        <div className="diagnostic-section-header">
          <span>Job History</span>
          <small>Safe terminal job metadata and step drilldown</small>
        </div>
        <TerminalJobHistoryPanel activeSaveId={activeSaveId} filters={filters} isAdmin={isAdmin} />
      </section>
      <section className="diagnostic-panel">
        <div className="diagnostic-section-header">
          <span>Event Stream</span>
          <small>Recent client, API, and job events</small>
        </div>
        <WebEventsList events={report?.web_events ?? []} />
      </section>
    </section>
  );
}

type DiagnosticsHealthTone = "healthy" | "warning" | "attention";
type DiagnosticStatTone = DiagnosticsHealthTone | "neutral";
type DiagnosticsHealth = {
  label: "Healthy" | "Warnings" | "Attention";
  tone: DiagnosticsHealthTone;
  detail: string;
  icon: LucideIcon;
};
type DiagnosticsCockpitSummary = {
  health: DiagnosticsHealth;
  signalCount: number;
  signalIssueCount: number;
  signalDetail: string;
  failedJobCount: number;
  cancelledJobCount: number;
  jobDetail: string;
  errorEventCount: number;
  warningEventCount: number;
  eventDetail: string;
  performanceRowCount: number;
  failedOperationCount: number;
  performanceDetail: string;
  hasActiveSaveHealth: boolean;
  saveHealthWarningCount: number;
  saveHealthDetail: string;
  schedulerTaskCount: number;
  schedulerIssueCount: number;
  schedulerDetail: string;
};

function DiagnosticStat({
  icon: Icon,
  label,
  value,
  detail,
  tone = "neutral"
}: {
  icon: LucideIcon;
  label: string;
  value: number | string;
  detail: string;
  tone?: DiagnosticStatTone;
}) {
  return (
    <div className={`diagnostic-stat ${tone}`}>
      <span aria-hidden="true"><Icon size={15} /></span>
      <div>
        <strong>{value}</strong>
        <small>{label}</small>
      </div>
      <em>{detail}</em>
    </div>
  );
}

function diagnosticsCockpitSummary(report?: DiagnosticsModel): DiagnosticsCockpitSummary {
  const diagnostics = report?.signals ?? [];
  const jobs = report?.maintenance_jobs ?? [];
  const events = report?.web_events ?? [];
  const performanceRows = runtimePerformanceRows(report?.runtime_performance ?? undefined);
  const activeSaveHealth = report?.active_save_health ?? null;
  const schedulerHealth = report?.scheduler_health;
  const diagnosticIssueCount = diagnostics.filter((entry) => Boolean(entry.error || entry.retry_summary)).length;
  const failedJobCount = jobs.filter((job) => job.status === "failed").length;
  const cancelledJobCount = jobs.filter((job) => job.status === "cancelled").length;
  const errorEventCount = events.filter((event) => event.level === "error").length;
  const warningEventCount = events.filter((event) => event.level === "warning" || event.level === "warn").length;
  const failedOperationCount = performanceRows.reduce((count, row) => count + row.failed_count + row.cancelled_count, 0);
  const saveHealthWarningCount = activeSaveHealth?.warnings?.length ?? 0;
  const schedulerIssueCount = (schedulerHealth?.summary.failed ?? 0) + (schedulerHealth?.summary.overdue ?? 0);
  const schedulerTaskCount = schedulerHealth?.summary.total ?? 0;
  const attentionCount = diagnosticIssueCount + failedJobCount + errorEventCount + schedulerIssueCount;
  const warningCount = cancelledJobCount + warningEventCount + saveHealthWarningCount;
  let health: DiagnosticsHealth;
  if (attentionCount) {
    health = {
      label: "Attention",
      tone: "attention",
      detail: `${attentionCount} signal${attentionCount === 1 ? "" : "s"} need attention across diagnostics, jobs, events, or scheduler state.`,
      icon: FileWarning
    };
  } else if (warningCount) {
    health = {
      label: "Warnings",
      tone: "warning",
      detail: `${warningCount} warning${warningCount === 1 ? "" : "s"} found, but no hard failures are active.`,
      icon: GitBranch
    };
  } else {
    health = {
      label: "Healthy",
      tone: "healthy",
      detail: "No active diagnostic signals are reporting trouble right now.",
      icon: Check
    };
  }
  return {
    health,
    signalCount: diagnostics.length,
    signalIssueCount: diagnosticIssueCount,
    signalDetail: diagnosticIssueCount ? `${diagnosticIssueCount} need attention` : "clear",
    failedJobCount,
    cancelledJobCount,
    jobDetail: jobs.length ? `${jobs.length} terminal tracked` : "none",
    errorEventCount,
    warningEventCount,
    eventDetail: events.length ? `${events.length} recent events` : "quiet",
    performanceRowCount: performanceRows.length,
    failedOperationCount,
    performanceDetail: performanceRows.length ? failedOperationCount ? `${failedOperationCount} failed ops` : "latency only" : "no samples",
    hasActiveSaveHealth: Boolean(activeSaveHealth),
    saveHealthWarningCount,
    saveHealthDetail: activeSaveHealth ? saveHealthWarningCount ? `${saveHealthWarningCount} warning${saveHealthWarningCount === 1 ? "" : "s"}` : "clear" : "no save",
    schedulerTaskCount,
    schedulerIssueCount,
    schedulerDetail: schedulerTaskCount ? schedulerIssueCount ? `${schedulerIssueCount} need attention` : `${schedulerTaskCount} tracked` : "none"
  };
}

function runtimePerformanceRows(report?: RuntimePerformanceReport): RuntimePerformanceRow[] {
  if (!report) return [];
  return [
    ...(report.job_averages ?? []),
    ...(report.step_averages ?? []),
    ...(report.model_averages ?? [])
  ];
}

function ActiveSaveHealthPanel({ health }: { health: EngineHealthModel | null }) {
  if (!health) return <p className="empty">No active save health data</p>;
  const metrics = [
    ["Messages", health.active_message_count],
    ["Pending Suggestions", health.pending_suggestion_count],
    ["Stale Suggestions", health.stale_pending_suggestion_count],
    ["Summaries", health.summary_count],
    ["Failed Continuity Jobs", health.recent_failed_continuity_job_count],
    ["Pending Observations", health.observation_curation.pending_count],
    ["Eligible Observations", health.observation_curation.eligible_count],
    ["Leased Observations", health.observation_curation.leased_count],
    ["Curation Attempts", health.observation_curation.total_attempt_count],
    ["Curation Failures", health.observation_curation.terminal_failure_count]
  ];
  return (
    <div className="diagnostic-lane">
      <div className="diagnostic-metric-strip">
        {metrics.map(([label, value]) => (
          <span key={label}><em>{label}</em>{value}</span>
        ))}
      </div>
      {health.warnings.length ? (
        <div className="web-event-list compact">
          {health.warnings.map((warning) => (
            <div className={`web-event-row diagnostic-signal-row ${engineHealthWarningTone(warning)}`} key={warning.code}>
              <span className={`web-event-level diagnostic-badge ${engineHealthWarningTone(warning)}`}>{warning.severity}</span>
              <div>
                <strong>{labelize(warning.code)}</strong>
                <small>{[warning.message, warning.count !== null && warning.count !== undefined ? `${warning.count}` : null].filter(Boolean).join(" · ")}</small>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="empty">No active save health warnings</p>
      )}
    </div>
  );
}

function engineHealthWarningTone(warning: EngineHealthWarning): DiagnosticStatTone {
  if (warning.severity === "error" || warning.severity === "critical") return "attention";
  if (warning.severity === "warning") return "warning";
  return "neutral";
}

function SchedulerHealthList({ report, isAdmin = false }: { report?: SchedulerHealthReport; isAdmin?: boolean }) {
  const tasks = report?.tasks ?? [];
  if (!tasks.length) return <p className="empty">No scheduler tasks tracked</p>;
  return (
    <div className="web-event-list">
      {tasks.map((task) => (
        <div className={`web-event-row diagnostic-signal-row ${schedulerTaskTone(task)}`} key={task.task_id}>
          <span className={`web-event-level diagnostic-badge ${schedulerTaskTone(task)}`}>{task.status}</span>
          <div>
            <strong>{labelize(task.task_type)}</strong>
            <small>{schedulerTaskSummary(task)}</small>
            {task.last_job_id ? (
              <JobDiagnosticDisclosure
                job={{
                  id: task.last_job_id,
                  save_id: task.save_id
                }}
                isAdmin={isAdmin}
              />
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function schedulerTaskTone(task: SchedulerHealthTask): DiagnosticStatTone {
  if (task.status === "failed" || task.status === "overdue") return "attention";
  if (task.status === "leased" || task.failure_count) return "warning";
  if (task.status === "healthy") return "healthy";
  return "neutral";
}

function schedulerTaskSummary(task: SchedulerHealthTask): string {
  return [
    task.error,
    task.skip_reason ? `skipped ${task.skip_reason}` : null,
    task.failure_count ? `${task.failure_count} failure${task.failure_count === 1 ? "" : "s"}` : null,
    task.last_job_id ? `job ${task.last_job_id}` : null,
    task.save_id,
    task.next_run_at
  ].filter(Boolean).join(" · ");
}

function RuntimePerformanceList({ report }: { report?: RuntimePerformanceReport }) {
  if (!report) return <p className="empty">No runtime performance data</p>;
  const sections = runtimePerformanceSections(report);
  const slowest = report.slowest_recent ?? [];
  if (!sections.some((section) => section.rows.length) && !slowest.length) {
    return <p className="empty">No runtime performance data</p>;
  }
  return (
    <div className="runtime-performance-grid">
      {sections.map((section) => (
        <div className="runtime-performance-section diagnostic-lane" key={section.title}>
          <header>
            <h4>{section.title}</h4>
            <span>{section.rows.length}</span>
          </header>
          {section.rows.length ? (
            <div className="web-event-list compact">
              {section.rows.map((row, index) => (
                <div className={`web-event-row runtime-performance-row diagnostic-signal-row ${runtimePerformanceTone(row)}`} key={`${section.kind}-${runtimePerformanceTitle(row, section.kind)}-${index}`}>
                  <span className="web-event-level info diagnostic-count-badge">{row.success_count}</span>
                  <div>
                    <strong>{runtimePerformanceTitle(row, section.kind)}</strong>
                    <small>{runtimePerformanceSummary(row)}</small>
                    <RuntimeMetricStrip row={row} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="empty">No {section.title.toLowerCase()} data</p>
          )}
        </div>
      ))}
      {slowest.length ? (
        <div className="runtime-performance-section diagnostic-lane">
          <header>
            <h4>Slowest Recent</h4>
            <span>{slowest.length}</span>
          </header>
          <div className="web-event-list compact">
            {slowest.map((operation) => (
              <div className={`web-event-row runtime-performance-row diagnostic-signal-row ${slowOperationTone(operation)}`} key={operation.job_id}>
                <span className={`web-event-level info diagnostic-count-badge ${slowOperationTone(operation)}`}>{operation.status}</span>
                <div>
                  <strong>{slowOperationTitle(operation)}</strong>
                  <small>{slowOperationSummary(operation)}</small>
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function runtimePerformanceSections(report: RuntimePerformanceReport): { title: string; kind: "job" | "step" | "model"; rows: RuntimePerformanceRow[] }[] {
  return [
    { title: "Jobs", kind: "job", rows: report.job_averages ?? [] },
    { title: "Steps", kind: "step", rows: report.step_averages ?? [] },
    { title: "Models", kind: "model", rows: report.model_averages ?? [] }
  ];
}

function runtimePerformanceTitle(row: RuntimePerformanceRow, kind: "job" | "step" | "model"): string {
  if (kind === "job") return labelize(row.job_type ?? "job");
  if (kind === "step") return labelize(row.step_name ?? "step");
  const model = [row.provider, row.model].filter(Boolean).join(" / ");
  return model || labelize(row.task ?? "model");
}

function runtimePerformanceSummary(row: RuntimePerformanceRow): string {
  return [
    row.task ? labelize(row.task) : null,
    row.average_duration_ms !== null ? `avg ${formatDurationMs(row.average_duration_ms)}` : null,
    row.p50_duration_ms !== null && row.p50_duration_ms !== undefined ? `p50 ${formatDurationMs(row.p50_duration_ms)}` : null,
    row.p95_duration_ms !== null && row.p95_duration_ms !== undefined ? `p95 ${formatDurationMs(row.p95_duration_ms)}` : null,
    row.latest_duration_ms !== null ? `latest ${formatDurationMs(row.latest_duration_ms)}` : null,
    row.p95_queue_wait_ms !== null && row.p95_queue_wait_ms !== undefined ? `queue p95 ${formatDurationMs(row.p95_queue_wait_ms)}` : null,
    row.failure_rate ? `fail ${formatPercent(row.failure_rate)}` : null,
    `ok ${row.success_count}`,
    row.sample_count ? `samples ${row.sample_count}` : null,
    row.failed_count ? `failed ${row.failed_count}` : null,
    row.cancelled_count ? `cancelled ${row.cancelled_count}` : null,
    row.skipped_count ? `skipped ${row.skipped_count}` : null,
    row.latest_completed_at
  ].filter(Boolean).join(" · ");
}

function formatDurationMs(value: number): string {
  if (value >= 1000) return `${(value / 1000).toFixed(value >= 10000 ? 0 : 1)}s`;
  return `${value} ms`;
}

function formatPercent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function runtimePerformanceTone(row: RuntimePerformanceRow): DiagnosticStatTone {
  if (row.failed_count || row.cancelled_count) return "attention";
  if (row.skipped_count) return "warning";
  return "healthy";
}

function slowOperationTone(operation: RuntimeSlowOperation): DiagnosticStatTone {
  if (operation.status === "failed") return "attention";
  if (operation.status === "cancelled") return "warning";
  return "neutral";
}

function slowOperationTitle(operation: RuntimeSlowOperation): string {
  return diagnosticLabel(operation.slowest_step_name ?? operation.job_type);
}

function slowOperationSummary(operation: RuntimeSlowOperation): string {
  return [
    operation.job_id,
    labelize(operation.job_type),
    typeof operation.duration_ms === "number" ? formatDurationMs(operation.duration_ms) : null,
    typeof operation.queue_wait_ms === "number" ? `queue ${formatDurationMs(operation.queue_wait_ms)}` : null,
    operation.slowest_step_duration_ms !== null && operation.slowest_step_duration_ms !== undefined ? `step ${formatDurationMs(operation.slowest_step_duration_ms)}` : null,
    operation.provider,
    operation.model,
    operation.completed_at
  ].filter(Boolean).join(" · ");
}

function diagnosticLabel(value: string): string {
  return labelize(value.replace(/[.-]/g, "_"));
}

type JobDiagnosticRef = {
  id: string;
  save_id: string | null;
};

function JobDiagnosticDisclosure({ job, isAdmin }: { job: JobDiagnosticRef; isAdmin: boolean }) {
  const controlId = React.useId();
  const [expanded, setExpanded] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [detail, setDetail] = useState<JobDiagnosticsModel | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const detailForJob = detail?.job_id === job.id && (detail.save_id ?? null) === (job.save_id ?? null) ? detail : null;
  useEffect(() => {
    if (!expanded && !modalOpen) return;
    if (detailForJob) return;
    let cancelled = false;
    setLoading(true);
    setError("");
    void apiRead<JobDiagnosticsModel>(jobDiagnosticsPath(job))
      .then((value) => {
        if (!cancelled) setDetail(value);
      })
      .catch((failure) => {
        if (!cancelled) setError(failure instanceof Error ? failure.message : "Could not load request details");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [detailForJob, expanded, job, modalOpen]);
  return (
    <div className="job-diagnostic-disclosure">
      <button
        type="button"
        className="diagnostic-expand-button"
        aria-expanded={expanded}
        aria-controls={controlId}
        onClick={() => setExpanded((current) => !current)}
      >
        <ChevronDown size={14} aria-hidden="true" />
        {expanded ? "Hide request details" : "Expand request details"}
      </button>
      {expanded ? (
        <div id={controlId} className="job-diagnostic-inline">
          {loading ? <p className="muted">Loading request details...</p> : null}
          {error ? <InlineNotice>{error}</InlineNotice> : null}
          {detailForJob ? (
            <JobDiagnosticSummary detail={detailForJob} isAdmin={isAdmin} onOpenModal={() => setModalOpen(true)} />
          ) : null}
        </div>
      ) : null}
      {modalOpen && detailForJob ? (
        <JobDiagnosticModal detail={detailForJob} onClose={() => setModalOpen(false)} />
      ) : null}
    </div>
  );
}

function JobDiagnosticSummary({
  detail,
  isAdmin,
  onOpenModal
}: {
  detail: JobDiagnosticsModel;
  isAdmin: boolean;
  onOpenModal: () => void;
}) {
  const request = detail.diagnostics.request ?? {};
  const provider = detail.diagnostics.provider ?? {};
  const bragi = detail.diagnostics.bragi ?? {};
  const summary = [
    request.origin?.label,
    request.task ? diagnosticLabel(request.task) : null,
    [request.provider, request.model].filter(Boolean).join(" / "),
    typeof provider.error_category === "string" ? provider.error_category : null,
    typeof provider.http_status === "number" ? `HTTP ${provider.http_status}` : null,
    bragi.error
  ].filter(Boolean).join(" · ");
  return (
    <div className="job-diagnostic-summary">
      <small className="muted">{summary || "No request summary"}</small>
      {isAdmin ? (
        <button type="button" className="secondary-button compact" onClick={onOpenModal}>
          Open full details
        </button>
      ) : null}
      {!detail.detail_available ? <small className="muted">Legacy job metadata.</small> : null}
    </div>
  );
}

function JobDiagnosticModal({ detail, onClose }: { detail: JobDiagnosticsModel; onClose: () => void }) {
  const titleId = React.useId();
  return (
    <ModalBackdrop>
      <DialogPanel className="diagnostic-detail-dialog" titleId={titleId} onClose={onClose}>
        <header>
          <div>
            <span className="diagnostics-eyebrow">Admin detail</span>
            <h2 id={titleId}>{labelize(detail.job_type)}</h2>
            <p className="muted">{detail.job_id} · {detail.status}</p>
          </div>
          <button type="button" onClick={onClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <pre className="diagnostic-detail-pre">{JSON.stringify(detail.diagnostics, null, 2)}</pre>
        <div className="command-row end">
          <button type="button" onClick={onClose}>Close</button>
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function RuntimeMetricStrip({ row }: { row: RuntimePerformanceRow }) {
  const metrics = [
    ["avg", row.average_duration_ms],
    ["p50", row.p50_duration_ms],
    ["p95", row.p95_duration_ms],
    ["latest", row.latest_duration_ms],
    ["min", row.min_duration_ms],
    ["max", row.max_duration_ms],
    ["queue p95", row.p95_queue_wait_ms]
  ].filter((metric): metric is [string, number] => typeof metric[1] === "number");
  if (!metrics.length) return null;
  return (
    <div className="diagnostic-metric-strip" aria-hidden="true">
      {metrics.map(([label, value]) => (
        <span key={label}><em>{label}</em>{formatDurationMs(value)}</span>
      ))}
    </div>
  );
}

function TerminalJobHistoryPanel({ activeSaveId, filters, isAdmin }: { activeSaveId: string | null; filters: DiagnosticsFilters; isAdmin: boolean }) {
  const [status, setStatus] = useState<TerminalJobStatusFilter>("terminal");
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const jobsQuery = useQuery({
    queryKey: ["jobs", "terminal", activeSaveId, status, filters.since, filters.limit],
    queryFn: ({ signal }) => apiRead<{ jobs: TerminalJobSummary[] }>(terminalJobsPath(activeSaveId, status, filters), signal),
    staleTime: DIAGNOSTICS_STALE_MS
  });
  const jobs = jobsQuery.data?.jobs ?? [];
  const selectedJob = jobs.find((job) => job.id === selectedJobId) ?? null;
  const stepsQuery = useQuery({
    queryKey: ["job-steps", selectedJob?.id, selectedJob?.save_id],
    queryFn: ({ signal }) => {
      if (!selectedJob) throw new Error("No job selected");
      return apiRead<JobStepsModel>(jobStepsPath(selectedJob), signal);
    },
    enabled: Boolean(selectedJob),
    staleTime: DIAGNOSTICS_STALE_MS
  });
  return (
    <div className="diagnostic-lane">
      <div className="diagnostics-toolbar compact">
        <label>
          <span>Status</span>
          <select aria-label="Terminal job status" value={status} onChange={(event) => setStatus(event.target.value as TerminalJobStatusFilter)}>
            <option value="terminal">Terminal</option>
            <option value="failed">Failed</option>
            <option value="cancelled">Cancelled</option>
            <option value="succeeded">Succeeded</option>
          </select>
        </label>
      </div>
      {jobsQuery.isLoading ? <p className="muted">Loading job history...</p> : null}
      {jobsQuery.error instanceof Error ? <InlineNotice>{jobsQuery.error.message}</InlineNotice> : null}
      <TerminalJobsList
        jobs={jobs}
        isAdmin={isAdmin}
        selectedJobId={selectedJobId}
        steps={stepsQuery.data}
        stepsLoading={stepsQuery.isFetching}
        onSelectJob={(job) => setSelectedJobId((current) => current === job.id ? null : job.id)}
      />
    </div>
  );
}

function TerminalJobsList({
  jobs,
  isAdmin = false,
  selectedJobId,
  steps,
  stepsLoading = false,
  onSelectJob
}: {
  jobs: TerminalJobSummary[];
  isAdmin?: boolean;
  selectedJobId: string | null;
  steps?: JobStepsModel;
  stepsLoading?: boolean;
  onSelectJob: (job: TerminalJobSummary) => void;
}) {
  if (!jobs.length) return <p className="empty">No terminal job history</p>;
  return (
    <div className="web-event-list">
      {jobs.map((job) => {
        const selected = selectedJobId === job.id;
        return (
          <div className={`web-event-row diagnostic-signal-row ${terminalJobTone(job)}`} key={job.id}>
            <span className={`web-event-level diagnostic-badge ${terminalJobTone(job)}`}>{job.status}</span>
            <div>
              <button type="button" className="text-button inline" onClick={() => onSelectJob(job)}>
                <strong>{labelize(job.type)}</strong>
              </button>
              <small>{terminalJobSummary(job)}</small>
              <JobDiagnosticDisclosure job={job} isAdmin={isAdmin} />
              {selected ? (
                <TerminalJobSteps
                  steps={steps?.job_id === job.id ? steps.steps : []}
                  loading={stepsLoading && steps?.job_id !== job.id}
                />
              ) : null}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function TerminalJobSteps({ steps, loading }: { steps: JobStepSummary[]; loading: boolean }) {
  if (loading) return <p className="muted">Loading steps...</p>;
  if (!steps.length) return <p className="empty">No step metadata</p>;
  return (
    <div className="web-event-list compact">
      {steps.map((step) => (
        <div className={`web-event-row diagnostic-signal-row ${step.status === "failed" ? "attention" : "neutral"}`} key={step.id}>
          <span className="web-event-level diagnostic-badge neutral">{step.status}</span>
          <div>
            <strong>{diagnosticLabel(step.name)}</strong>
            <small>{terminalJobStepSummary(step)}</small>
            <DataViewer value={step.metadata} emptyLabel="No step metadata" />
          </div>
        </div>
      ))}
    </div>
  );
}

function terminalJobTone(job: TerminalJobSummary): DiagnosticStatTone {
  if (job.status === "failed") return "attention";
  if (job.status === "cancelled") return "warning";
  return "neutral";
}

function terminalJobSummary(job: TerminalJobSummary): string {
  return [
    job.status,
    typeof job.duration_ms === "number" ? formatDurationMs(job.duration_ms) : null,
    typeof job.queue_wait_ms === "number" ? `queue ${formatDurationMs(job.queue_wait_ms)}` : null,
    job.step_count ? `${job.step_count} step${job.step_count === 1 ? "" : "s"}` : null,
    job.error,
    job.completed_at
  ].filter(Boolean).join(" · ");
}

function terminalJobStepSummary(step: JobStepSummary): string {
  return [
    step.provider,
    step.model,
    step.task ? labelize(step.task) : null,
    typeof step.duration_ms === "number" ? formatDurationMs(step.duration_ms) : null,
    step.completed_at
  ].filter(Boolean).join(" · ");
}

function DiagnosticsList({ diagnostics, isAdmin = false }: { diagnostics: DiagnosticEntry[]; isAdmin?: boolean }) {
  if (!diagnostics.length) return <p className="empty">No active diagnostic signals</p>;
  return (
    <div className="web-event-list">
      {diagnostics.map((entry, index) => (
        <div className={`web-event-row diagnostic-signal-row ${diagnosticEntryTone(entry)}`} key={`${entry.kind}-${entry.provider ?? entry.job_type ?? index}`}>
          <span className={`web-event-level diagnostic-badge ${diagnosticEntryTone(entry)}`}>{entry.kind}</span>
          <div>
            <strong>{entry.provider ?? entry.job_type ?? entry.path ?? "Diagnostic"}</strong>
            <small>{[entry.error, entry.retry_summary, entry.save_id].filter(Boolean).join(" · ")}</small>
            {entry.job_id ? (
              <JobDiagnosticDisclosure
                job={{
                  id: entry.job_id,
                  save_id: entry.save_id ?? null
                }}
                isAdmin={isAdmin}
              />
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function diagnosticEntryTone(entry: DiagnosticEntry): DiagnosticStatTone {
  if (entry.error || entry.retry_summary) return "attention";
  return "neutral";
}

function MaintenanceJobsList({ jobs, isAdmin = false }: { jobs: MaintenanceJobDiagnostic[]; isAdmin?: boolean }) {
  if (!jobs.length) return <p className="empty">No failed job diagnostics</p>;
  return (
    <div className="web-event-list">
      {jobs.map((job) => (
        <div className={`web-event-row diagnostic-signal-row ${maintenanceJobTone(job)}`} key={job.job_id}>
          <span className={`web-event-level diagnostic-badge ${maintenanceJobTone(job)}`}>{job.status}</span>
          <div>
            <strong>{labelize(job.job_type)}</strong>
            <small>{maintenanceJobSummary(job)}</small>
            <DataViewer value={job.metrics} emptyLabel="No batch details" />
            <JobDiagnosticDisclosure
              job={{
                id: job.job_id,
                save_id: job.save_id
              }}
              isAdmin={isAdmin}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function maintenanceJobTone(job: MaintenanceJobDiagnostic): DiagnosticStatTone {
  if (job.status === "failed") return "attention";
  if (job.status === "cancelled") return "warning";
  return "neutral";
}

function maintenanceJobSummary(job: MaintenanceJobDiagnostic): string {
  return [
    job.summary,
    job.error,
    job.completed_at
  ].filter(Boolean).join(" · ");
}

function WebEventsList({ events }: { events: WebEventEntry[] }) {
  if (!events.length) return <p className="empty">No recent web events</p>;
  return (
    <div className="web-event-list">
      {events.map((event, index) => (
        <div className={`web-event-row diagnostic-signal-row ${webEventTone(event)}`} key={`${event.timestamp}-${event.event}-${index}`}>
          <span className={`web-event-level diagnostic-badge ${webEventTone(event)}`}>{event.level}</span>
          <div>
            <strong>{event.event}</strong>
            <small>{webEventSummary(event)}</small>
          </div>
        </div>
      ))}
    </div>
  );
}

function webEventTone(event: WebEventEntry): DiagnosticStatTone {
  if (event.level === "error") return "attention";
  if (event.level === "warning" || event.level === "warn") return "warning";
  if (event.level === "info") return "healthy";
  return "neutral";
}

function webEventSummary(event: WebEventEntry): string {
  return [
    event.request_id ? `request ${event.request_id}` : null,
    event.job_id ? `job ${event.job_id}` : null,
    event.save_id ? `save ${event.save_id}` : null,
    event.status_code ? `${event.status_code}` : null,
    event.status ?? event.job_status ?? null,
    event.duration_ms !== undefined && event.duration_ms !== null ? `${event.duration_ms} ms` : null,
    event.error_class ?? event.error ?? null,
    event.route ?? event.job_type ?? event.task_type ?? event.component ?? null
  ].filter(Boolean).join(" · ");
}

function ToggleSetting({ control, disabled, updateLocal }: { control: { setting_key: string; enabled: boolean }; disabled: boolean; updateLocal: (key: string, value: unknown) => void }) {
  const tooltip = settingTooltip(control.setting_key);
  const label = settingLabel(control.setting_key);
  return (
    <label className="toggle-row compact-toggle" title={tooltip}>
      <input
        type="checkbox"
        checked={control.enabled}
        disabled={disabled}
        title={tooltip}
        aria-label={label}
        onChange={(event) => updateLocal(control.setting_key, event.target.checked)}
      />
      <span>
        <strong>{label}</strong>
        <small>{tooltip}</small>
      </span>
    </label>
  );
}

function TextSetting({ control, disabled, updateLocal }: { control: { setting_key: string; value: string }; disabled: boolean; updateLocal: (key: string, value: unknown) => void }) {
  const tooltip = settingTooltip(control.setting_key);
  const label = settingLabel(control.setting_key);
  const [draft, setDraft] = useState(control.value ?? "");
  useEffect(() => {
    setDraft(control.value ?? "");
  }, [control.setting_key, control.value]);
  const savedValue = (control.value ?? "").trim();
  const draftValue = draft.trim();
  return (
    <div className="text-setting" title={tooltip}>
      <label className="field-label">
        <span>{label}</span>
        <textarea
          className="tall-field"
          value={draft}
          disabled={disabled}
          title={tooltip}
          aria-label={label}
          onChange={(event) => setDraft(event.target.value)}
        />
        <small className="setting-helper">{tooltip}</small>
      </label>
      <div className="command-row end">
        <button type="button" disabled={disabled || !draftValue} onClick={() => {
          setDraft("");
          updateLocal(control.setting_key, "");
        }}>
          <X size={15} /> Clear
        </button>
        <button type="button" className="primary-command compact" disabled={disabled || draftValue === savedValue} onClick={() => updateLocal(control.setting_key, draft)}>
          <Save size={15} /> Save
        </button>
      </div>
    </div>
  );
}

function SyncedNumberInput({
  value,
  minimum,
  maximum,
  step,
  disabled,
  title,
  ariaLabel,
  onCommit
}: {
  value: number;
  minimum: number;
  maximum?: number | null;
  step?: number;
  disabled: boolean;
  title?: string;
  ariaLabel?: string;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState(String(value));
  const [focused, setFocused] = useState(false);
  useEffect(() => {
    if (!focused) setDraft(String(value));
  }, [focused, value]);
  const commitDraft = () => {
    setFocused(false);
    const nextValue = Number(draft);
    if (!draft.trim() || !Number.isFinite(nextValue)) {
      setDraft(String(value));
      return;
    }
    if (nextValue === value) return;
    onCommit(nextValue);
  };
  return (
    <input
      type="number"
      min={minimum}
      max={maximum ?? undefined}
      step={step ?? 1}
      value={draft}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
      onFocus={() => setFocused(true)}
      onChange={(event) => setDraft(event.target.value)}
      onBlur={commitDraft}
    />
  );
}

function NumberSetting({ control, disabled, updateLocal }: { control: { setting_key: string; value: number; minimum: number; maximum?: number | null; step?: number }; disabled: boolean; updateLocal: (key: string, value: unknown) => void }) {
  const tooltip = settingTooltip(control.setting_key);
  const label = settingLabel(control.setting_key);
  return (
    <label className="field-label" title={tooltip}>
      <span>{label}</span>
      <SyncedNumberInput
        value={control.value}
        minimum={control.minimum}
        maximum={control.maximum}
        step={control.step}
        disabled={disabled}
        title={tooltip}
        ariaLabel={label}
        onCommit={(value) => updateLocal(control.setting_key, value)}
      />
      <small className="setting-helper">{tooltip}</small>
    </label>
  );
}

function OptionalNumberSetting({
  control,
  disabled,
  updateLocal
}: {
  control: { setting_key: string; enabled_setting_key: string; enabled: boolean; supported: boolean; value: number; minimum: number; maximum: number; step?: number };
  disabled: boolean;
  updateLocal: (key: string, value: unknown) => void;
}) {
  const supported = control.supported;
  const tooltip = supported ? settingTooltip(control.setting_key) : "Refresh model metadata for a selected provider model that supports this parameter.";
  const enabledLabel = settingLabel(control.enabled_setting_key);
  const valueLabel = labelize(control.setting_key);
  return (
    <div className="optional-number-setting" title={tooltip}>
      <label className="toggle-row compact-toggle">
        <input
          type="checkbox"
          checked={control.enabled}
          disabled={disabled || !supported}
          aria-label={enabledLabel}
          onChange={(event) => updateLocal(control.enabled_setting_key, event.target.checked)}
        />
        <span>
          <strong>{enabledLabel}</strong>
          <small>{tooltip}</small>
        </span>
      </label>
      <label className="field-label">
        <span>{valueLabel}</span>
        <SyncedNumberInput
          value={control.value}
          minimum={control.minimum}
          maximum={control.maximum}
          step={control.step}
          disabled={disabled || !supported || !control.enabled}
          ariaLabel={valueLabel}
          onCommit={(value) => updateLocal(control.setting_key, value)}
        />
      </label>
    </div>
  );
}

function ChoiceSetting({ control, disabled, updateLocal, optionLabel = labelize }: { control: { setting_key: string; selected: string; options: string[] }; disabled: boolean; updateLocal: (key: string, value: unknown) => void; optionLabel?: (option: string) => string }) {
  const tooltip = settingTooltip(control.setting_key);
  const label = settingLabel(control.setting_key);
  return (
    <label className="field-label" title={tooltip}>
      <span>{label}</span>
      <select value={control.selected} disabled={disabled} title={tooltip} aria-label={label} onChange={(event) => updateLocal(control.setting_key, event.target.value)}>
        {control.options.map((option) => <option key={option} value={option}>{optionLabel(option)}</option>)}
      </select>
      <small className="setting-helper">{tooltip}</small>
    </label>
  );
}

function SupportedChoiceSetting({ control, disabled, updateLocal, optionLabel = labelize }: { control: { setting_key: string; selected: string; options: string[]; supported: boolean }; disabled: boolean; updateLocal: (key: string, value: unknown) => void; optionLabel?: (option: string) => string }) {
  const tooltip = control.supported ? settingTooltip(control.setting_key) : "Refresh model metadata for a selected provider model that supports this parameter.";
  const label = settingLabel(control.setting_key);
  return (
    <label className="field-label" title={tooltip}>
      <span>{label}</span>
      <select value={control.selected} disabled={disabled || !control.supported} title={tooltip} aria-label={label} onChange={(event) => updateLocal(control.setting_key, event.target.value)}>
        {control.options.map((option) => <option key={option} value={option}>{optionLabel(option)}</option>)}
      </select>
      <small className="setting-helper">{tooltip}</small>
    </label>
  );
}

function ThinkingLevelSelect({
  control,
  label,
  disabled,
  onChange
}: {
  control: ThinkingLevelControl;
  label: string;
  disabled: boolean;
  onChange: (level: string) => void;
}) {
  const tooltip = thinkingLevelTooltip(control);
  return (
    <label className="field-label thinking-level-control" title={tooltip}>
      <span>Thinking</span>
      <select
        value={control.selected}
        disabled={disabled || !control.supported}
        title={tooltip}
        aria-label={label}
        onChange={(event) => onChange(event.target.value)}
      >
        {control.options.map((option) => (
          <option key={option} value={option}>{thinkingLevelLabel(option)}</option>
        ))}
      </select>
      <small className="setting-helper">{tooltip}</small>
    </label>
  );
}

function thinkingLevelTooltip(control: ThinkingLevelControl): string {
  if (!control.supported) {
    return control.disabled_reason || "Selected model does not support thinking level.";
  }
  const defaultLabel = control.default_level ? thinkingLevelLabel(control.default_level) : "provider default";
  if (control.mandatory) return `This model requires thinking. Provider default is ${defaultLabel}.`;
  return `Optional model thinking level. Provider default is ${defaultLabel}.`;
}

function ModelSelector({ selector, labelOverride, saveId = null }: { selector: TaskModelSelector; labelOverride?: string; saveId?: string | null }) {
  const client = useQueryClient();
  const [error, setError] = useState("");
  const selectedValue = modelPreferenceValue(selector.selected_provider, selector.selected_model_id);
  const label = labelOverride ?? selector.label ?? taskLabel(selector.task);
  const tooltip = taskModelTooltip(selector.task);
  const savePreference = useMutation({
    mutationFn: ({ provider, model_id }: { provider: string; model_id: string }) => postJson(
      "/api/settings/model-preference",
      saveId ? { task: selector.task, provider, model_id, save_id: saveId } : { task: selector.task, provider, model_id }
    ),
    onSuccess: () => {
      setError("");
      client.invalidateQueries({ queryKey: ["settings", "full"] });
      client.invalidateQueries({ queryKey: ["runtime"] });
    },
    onError: (failure) => setError(failure instanceof Error ? failure.message : "Could not save model preference")
  });
  const clearPreference = useMutation({
    mutationFn: () => deleteJson(modelPreferencePath(selector.task, saveId)),
    onSuccess: () => {
      setError("");
      client.invalidateQueries({ queryKey: ["settings", "full"] });
      client.invalidateQueries({ queryKey: ["runtime"] });
    },
    onError: (failure) => setError(failure instanceof Error ? failure.message : "Could not clear model preference")
  });
  const saveThinkingPreference = useMutation({
    mutationFn: async ({ level, provider, model_id }: { level: string; provider: string; model_id: string }) => {
      if (level === THINKING_LEVEL_PROVIDER_DEFAULT) {
        await deleteJson(modelThinkingPreferencePath(selector.task, saveId));
        return;
      }
      await postJson("/api/settings/model-thinking", {
        task: selector.task,
        provider,
        model_id,
        level,
        ...(saveId ? { save_id: saveId } : {})
      });
    },
    onSuccess: () => {
      setError("");
      client.invalidateQueries({ queryKey: ["settings", "full"] });
      client.invalidateQueries({ queryKey: ["runtime"] });
    },
    onError: (failure) => setError(failure instanceof Error ? failure.message : "Could not save thinking level")
  });
  const inherited = modelPreferenceValue(selector.inherited_provider ?? null, selector.inherited_model_id ?? null);
  const priceOption = selectedValue
    ? selectedOption(selector)
    : modelOptionForValue(selector.options, inherited);
  const showUnavailablePricing = Boolean(selectedValue || priceOption);
  return (
    <div className="model-selector" title={tooltip}>
      <div className="model-selector-label">
        <strong title={tooltip}>{label}</strong>
        <span>{selector.task}</span>
        {!selectedValue && inherited ? <small>Default: {selector.inherited_provider} / {selector.inherited_model_id}</small> : null}
      </div>
      <ModelOptionSelect
        label={`${label} model`}
        title={tooltip}
        value={selectedValue}
        options={selector.options}
        disabled={!selector.options.length || savePreference.isPending || clearPreference.isPending}
        emptyLabel="No compatible models"
        placeholderLabel={!selectedValue ? (inherited ? "Use default" : "Choose model") : undefined}
        unavailableOption={
          selector.options.length && !selector.selected_available && selector.selected_provider && selector.selected_model_id
            ? { value: selectedValue, label: `${modelOptionLabel(selector.selected_model_id, selector.selected_provider, selector.selected_model_id)} unavailable` }
            : undefined
        }
        onChange={(value) => {
          const [provider, model_id] = value.split("\u0000");
          if (provider && model_id) savePreference.mutate({ provider, model_id });
        }}
      />
      <ModelPricingLine option={priceOption} showUnavailable={showUnavailablePricing} />
      {selector.thinking ? (
        <ThinkingLevelSelect
          control={selector.thinking}
          label={`${label} thinking level`}
          disabled={savePreference.isPending || clearPreference.isPending || saveThinkingPreference.isPending}
          onChange={(level) => {
            if (!selector.selected_provider || !selector.selected_model_id) return;
            saveThinkingPreference.mutate({
              provider: selector.selected_provider,
              model_id: selector.selected_model_id,
              level
            });
          }}
        />
      ) : null}
      {selector.clearable ? (
        <div className="command-row end model-clear-row">
          <button
            type="button"
            disabled={clearPreference.isPending}
            onClick={() => clearPreference.mutate()}
          >
            <X size={14} /> Use default
          </button>
        </div>
      ) : null}
      <div className="capability-row">
        {selectedOption(selector)?.capabilities.slice(0, 3).map((capability) => (
          <span key={capability}>{capabilityLabel(capability)}</span>
        ))}
      </div>
      {selector.warning ? <InlineNotice>{selector.warning}</InlineNotice> : null}
      {error ? <InlineNotice>{error}</InlineNotice> : null}
    </div>
  );
}

function ModelOptionSelect({
  label,
  title,
  value,
  options,
  disabled,
  emptyLabel,
  placeholderLabel,
  unavailableOption,
  onChange
}: {
  label: string;
  title: string;
  value: string;
  options: ModelOption[];
  disabled: boolean;
  emptyLabel: string;
  placeholderLabel?: string;
  unavailableOption?: { value: string; label: string };
  onChange: (value: string) => void;
}) {
  const [query, setQuery] = useState("");
  const optionSignature = options.map(modelOptionValue).join("\u0001");
  useEffect(() => setQuery(""), [optionSignature]);
  const filteredOptions = matchingModelOptions(options, query);
  const selectedFiltered = Boolean(value && filteredOptions.some((option) => modelOptionValue(option) === value));
  const selectedOptionValue = value && !selectedFiltered
    ? options.find((option) => modelOptionValue(option) === value)
    : undefined;
  const showSearch = options.length > 1;
  const matchLabel = query ? modelOptionMatchLabel(filteredOptions.length) : "";
  return (
    <div className="model-option-control">
      {showSearch ? (
        <div className="model-option-search">
          <Search size={15} aria-hidden="true" />
          <input
            value={query}
            disabled={disabled}
            aria-label={`${label} search`}
            autoComplete="off"
            placeholder="Search models"
            spellCheck={false}
            title={`Search ${label}`}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button
            type="button"
            aria-label={`Clear ${label} search`}
            title={`Clear ${label} search`}
            disabled={disabled || !query}
            onClick={() => setQuery("")}
          >
            <X size={14} />
          </button>
          {matchLabel ? <span className="model-option-match-count" aria-live="polite">{matchLabel}</span> : null}
        </div>
      ) : null}
      <select
        value={value}
        disabled={disabled}
        aria-label={label}
        title={title}
        onChange={(event) => onChange(event.target.value)}
      >
        {!options.length ? <option value="">{emptyLabel}</option> : null}
        {unavailableOption ? <option value={unavailableOption.value}>{unavailableOption.label}</option> : null}
        {selectedOptionValue ? (
          <option value={modelOptionValue(selectedOptionValue)} hidden>
            {modelOptionSelectLabel(selectedOptionValue)}
          </option>
        ) : null}
        {options.length && !filteredOptions.length ? <option value="">No matching models</option> : null}
        {filteredOptions.length && placeholderLabel ? <option value="">{placeholderLabel}</option> : null}
        {filteredOptions.map((option) => (
          <option key={modelOptionValue(option)} value={modelOptionValue(option)}>
            {modelOptionSelectLabel(option)}
          </option>
        ))}
      </select>
    </div>
  );
}


export { DiagnosticsList, MaintenanceJobsList, RuntimePerformanceList, TerminalJobsList, WebEventsList };
