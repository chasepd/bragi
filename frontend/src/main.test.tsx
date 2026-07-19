import React from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, beforeEach, afterEach, vi } from "vitest";
import type { CharacterTextContact, CharacterTextMessage, CharacterTextThread, CharacterTextsModel, ChatSubmissionStatus, ChatTurnDelta, CharacterRegistryModel, DiagnosticsModel, Job, ModelOption, RuntimeModel, Scenario, ScenarioContentSection, ScenarioDraft, SettingsModel, TaskModelSelector, WorldDataModel } from "./api";

const EXPECTED_IMAGE_STYLE_PRESETS = [
  "none",
  "realistic",
  "anime",
  "cartoon",
  "cinematic",
  "concept_art",
  "digital_painting",
  "watercolor",
  "oil_painting",
  "comic_book",
  "colored_pencil",
  "sketch",
  "ink",
  "pixel_art",
  "three_d_render",
  "low_poly"
];

beforeEach(() => {
  vi.unstubAllGlobals();
  window.localStorage.clear();
  document.body.innerHTML = '<div id="root"></div>';
});

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

type EventSourceDoubleInstance = {
  url: string;
  closed: boolean;
  closeCalls: number;
  dispatch: (name: string, data: unknown) => void;
  dispatchRaw: (name: string, data: string) => void;
  dispatchNativeError: () => void;
  close: () => void;
};

function installEventSourceDouble(): EventSourceDoubleInstance[] {
  const sources: EventSourceDoubleInstance[] = [];
  class EventSourceDouble {
    url: string;
    closed = false;
    closeCalls = 0;
    listeners: Record<string, ((event: Event) => void)[]> = {};
    onerror: ((event: Event) => void) | null = null;

    constructor(url: string) {
      this.url = url;
      sources.push(this);
    }

    addEventListener(name: string, listener: (event: Event) => void) {
      this.listeners[name] = [...(this.listeners[name] ?? []), listener];
    }

    close() {
      this.closed = true;
      this.closeCalls += 1;
    }

    dispatch(name: string, data: unknown) {
      this.dispatchRaw(name, JSON.stringify(data));
    }

    dispatchRaw(name: string, data: string) {
      for (const listener of this.listeners[name] ?? []) {
        listener({ data } as MessageEvent);
      }
    }

    dispatchNativeError() {
      const event = new Event("error");
      for (const listener of this.listeners.error ?? []) {
        listener(event);
      }
      this.onerror?.(event);
    }
  }
  vi.stubGlobal("EventSource", EventSourceDouble);
  return sources;
}

function deferred<T>() {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason?: unknown) => void = () => undefined;
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise;
    reject = rejectPromise;
  });
  return { promise, resolve, reject };
}

function formDataTextEntries(body: BodyInit | null | undefined) {
  expect(body).toBeInstanceOf(FormData);
  const entries: Record<string, string> = {};
  (body as FormData).forEach((value, key) => {
    entries[key] = typeof value === "string" ? value : value.name;
  });
  return entries;
}

function formDataValue(body: BodyInit | null | undefined, key: string) {
  expect(body).toBeInstanceOf(FormData);
  return (body as FormData).get(key);
}

function stubWorkbenchMedia(matches: boolean | ((query: string) => boolean)) {
  vi.stubGlobal("matchMedia", vi.fn().mockImplementation((query: string) => ({
    matches: typeof matches === "function" ? matches(query) : matches,
    media: query,
    onchange: null,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn()
  })));
}

function runtimeModel(overrides: Partial<RuntimeModel> = {}): RuntimeModel {
  return {
    saves: [],
    active_save_id: "save-1",
    active_save_title: "Lantern Keep",
    active_scenario_type: null,
    scenario_title: "Lantern Keep",
    scene_title: "Beacon",
    chronicle: { messages: [] },
    media: null,
    action_choices: null,
    model_indicator: "fake / chat",
    failed_save: false,
    composer_enabled: true,
    failure_text: null,
    status: null,
    error: null,
    ...overrides
  };
}

function cyoaActionChoices(
  overrides: Partial<NonNullable<RuntimeModel["action_choices"]>> = {}
): NonNullable<RuntimeModel["action_choices"]> {
  return {
    narrator_message_id: "narrator-1",
    choices: [
      { choice_id: "choice-1", ordinal: 0, body: "Open the brass door" },
      { choice_id: "choice-2", ordinal: 1, body: "Question the masked guide" },
      { choice_id: "choice-3", ordinal: 2, body: "Search the moonlit shelves" },
      { choice_id: "choice-4", ordinal: 3, body: "Retreat to the courtyard" }
    ],
    ...overrides
  };
}

function scenarioFixture(overrides: Partial<Scenario> = {}): Scenario {
  return {
    scenario_id: "scenario-1",
    scenario_type: "full_roleplay",
    title: "Lantern Keep",
    premise: "A mountain beacon is going dark.",
    player_role: "Keeper",
    opening_message: null,
    save_count: 0,
    has_generation_prompt: false,
    action_choices_enabled: false,
    ...overrides
  };
}

function modelOption(
  model_id: string,
  display_name: string,
  capabilities: string[],
  provider = "fake",
  pricing: ModelOption["pricing"] = null,
  thinking: ModelOption["thinking"] = null
): ModelOption {
  return {
    provider,
    model_id,
    display_name,
    available: true,
    capabilities,
    pricing,
    thinking
  };
}

function modelThinkingSupport(levels: string[] = ["high", "low"]): ModelOption["thinking"] {
  return {
    levels,
    default_level: levels[levels.length - 1] ?? null,
    default_enabled: true,
    mandatory: false,
    supports_max_tokens: false
  };
}

function thinkingControl(
  task: string,
  selected = "provider_default",
  supported = true,
  model_id = "alpha"
): TaskModelSelector["thinking"] {
  return {
    setting_key: "model_thinking_preferences",
    task,
    selected,
    supported,
    options: supported ? ["provider_default", "off", "high", "low"] : ["provider_default"],
    provider: "fake",
    model_id,
    default_level: supported ? "low" : null,
    default_enabled: supported ? true : null,
    mandatory: false,
    disabled_reason: supported ? null : "Selected model does not support thinking level"
  };
}

function modelSelector(task: string, options: ModelOption[], selectedModelId: string | null = options[0]?.model_id ?? null, extra: Partial<TaskModelSelector> = {}): TaskModelSelector {
  const selectedOption = selectedModelId ? options.find((option) => option.model_id === selectedModelId) ?? null : null;
  return {
    task,
    selected_provider: selectedOption?.provider ?? null,
    selected_model_id: selectedModelId,
    selected_available: Boolean(selectedOption?.available),
    warning: null,
    options,
    ...extra
  };
}

function modelSettingsPayload(overrides: Partial<SettingsModel> = {}): SettingsModel {
  return {
    provider_cards: [],
    task_model_selectors: [],
    roleplay_shared_models: { setting_key: "roleplay_shared_models", enabled: true },
    roleplay_model_groups: [],
    scenario_section_model_selectors: [],
    ...overrides
  };
}

function characterRegistryPayload(overrides: Partial<CharacterRegistryModel> = {}): CharacterRegistryModel {
  return {
    active_save_id: "save-1",
    characters: [
      {
        character_id: "character-1",
        name: "Mara",
        aliases_text: "Signal runner",
        role: "Scout",
        age: "early 30s",
        known_state: "Knows the beacon lens is cracked.",
        met: true,
        appearance: "",
        visual_notes: "",
        current_clothing: "",
        personality: "",
        voice: "",
        texting_style: "",
        relationships_json: "{}",
        goals: "Keep the beacon lit.",
        motivations: "Protect the lower village.",
        current_intent: "Guard the lens stair.",
        boundaries: "Will not leave the tower.",
        attitude_toward_player: "Trusts the player under pressure.",
        cooperation_conditions: "Helps after proof the lens can hold.",
        status: "present",
        location_id: "location-beacon",
        private_notes: "",
        present: true,
        protected_from_maintenance: false,
        linked_memory_ids: ["memory-1"],
        linked_state_ids: [],
        linked_summary_ids: ["summary-1"]
      },
      {
        character_id: "character-2",
        name: "Ilyra",
        aliases_text: "Fog rival",
        role: "Navigator",
        age: "",
        known_state: "",
        met: false,
        appearance: "",
        visual_notes: "",
        current_clothing: "",
        personality: "",
        voice: "",
        relationships_json: "{}",
        status: "away",
        location_id: null,
        private_notes: "",
        present: false,
        linked_memory_ids: [],
        linked_state_ids: [],
        linked_summary_ids: []
      }
    ],
    link_targets: [
      {
        target_type: "memory",
        target_id: "memory-1",
        title: "Lens memory",
        body: "Mara saw the cracked lens.",
        tags: ["beacon"],
        importance: 0.7,
        linked_character_ids: ["character-1"]
      },
      {
        target_type: "summary",
        target_id: "summary-1",
        title: "Beacon summary",
        body: "Mara knows the beacon nearly failed.",
        linked_character_ids: ["character-1"]
      },
      {
        target_type: "summary",
        target_id: "summary-2",
        title: "Fog route",
        body: "Ilyra charted a safe fog route.",
        linked_character_ids: []
      }
    ],
    location_choices: [
      ["location-beacon", "Beacon Tower"],
      ["location-docks", "Fog Docks"]
    ],
    ...overrides
  };
}

function worldDataPayload(overrides: Partial<WorldDataModel> = {}): WorldDataModel {
  return {
    active_save_id: "save-1",
    scenario: {
      scenario_id: "scenario-1",
      scenario_type: "full_roleplay",
      title: "Lantern Keep",
      premise: "A watchtower.",
      player_character_name: "Mara",
      player_role: "Keeper",
      content_sections: []
    },
    scene: { snapshot_id: "scene-1", situation: "Fog over the beacon", weather: "fog" },
    world_state: [],
    memories: [
      {
        memory_id: "memory-1",
        body: "Storm memory",
        tags_text: "weather",
        importance: 0.7,
        metadata: { source: "manual" },
        archived: false
      }
    ],
    summaries: [
      {
        summary_id: "summary-1",
        title: "Beacon summary",
        summary_text: "The beacon almost failed.",
        status: "active",
        metadata: { scope: "scene" }
      }
    ],
    locations: [
      {
        location_id: "location-1",
        name: "Beacon Tower",
        description: "A tower above the fog.",
        status: "active"
      }
    ],
    threads: [
      {
        thread_id: "thread-1",
        title: "Repair the beacon",
        status: "open",
        objective: "Find a replacement lens."
      }
    ],
    links: [
      {
        link_id: "link-1",
        entity_type: "character",
        entity_id: "character-1",
        relation: "knows",
        target_type: "memory",
        target_id: "memory-1",
        confidence: 0.8
      }
    ],
    audit: [],
    ...overrides
  };
}

type OpenRouterRoutingTestModel = NonNullable<SettingsModel["openrouter_routing"]>;
type OpenRouterRoutingTestProfile = OpenRouterRoutingTestModel["global_profile"];

function openRouterRoutingProfile(overrides: Partial<OpenRouterRoutingTestProfile> = {}): OpenRouterRoutingTestProfile {
  return {
    order: [],
    allow_fallbacks: null,
    require_parameters: false,
    data_collection: "allow",
    zdr: false,
    enforce_distillable_text: false,
    only: [],
    ignore: [],
    quantizations: [],
    sort: "default",
    sort_partition: "model",
    preferred_min_throughput: {},
    preferred_max_latency: {},
    max_price: {},
    ...overrides
  };
}

function openRouterRoutingSettings(overrides: Partial<OpenRouterRoutingTestModel> = {}): OpenRouterRoutingTestModel {
  const globalProfile = openRouterRoutingProfile({ sort: "price" });
  return {
    setting_key: "openrouter_routing_profiles",
    global_profile: globalProfile,
    global_provider_payload: { sort: "price" },
    task_overrides: [
      {
        task_family: "narrator",
        label: "Narrator",
        enabled: false,
        profile: openRouterRoutingProfile(),
        provider_payload: {},
        effective_provider_payload: { sort: "price" }
      },
      {
        task_family: "background_text",
        label: "Background Text",
        enabled: false,
        profile: openRouterRoutingProfile(),
        provider_payload: {},
        effective_provider_payload: { sort: "price" }
      }
    ],
    provider_catalog: [],
    provider_catalog_refreshed_at: null,
    sort_options: ["default", "price", "throughput", "latency"],
    partition_options: ["model", "none"],
    data_collection_options: ["allow", "deny"],
    quantization_options: ["int4", "int8", "fp8"],
    percentile_options: ["p50", "p90"],
    max_price_fields: ["prompt", "completion", "request", "image"],
    ...overrides
  };
}

function modelRoutingProfilesSettings(overrides: Partial<NonNullable<SettingsModel["model_routing_profiles"]>> = {}): NonNullable<SettingsModel["model_routing_profiles"]> {
  return {
    setting_key: "model_routing_profiles",
    last_loaded_profile_id: null,
    profiles: [],
    ...overrides
  };
}

function settingsFetch(settingsPayload: SettingsModel) {
  return vi.fn().mockImplementation((path: string) => Promise.resolve({
    ok: true,
    json: async () => settingsPayloadForPath(path, settingsPayload)
  }));
}

function settingsFetchSequence(settingsPayloads: SettingsModel[]) {
  let settingsReadIndex = 0;
  return vi.fn().mockImplementation((path: string) => Promise.resolve({
    ok: true,
    json: async () => {
      if (!isAnySettingsReadPath(path)) return {};
      const payload = settingsPayloads[Math.min(settingsReadIndex, settingsPayloads.length - 1)];
      settingsReadIndex += 1;
      return settingsPayloadForPath(path, payload);
    }
  }));
}

function isSettingsReadPath(path: string) {
  return path === "/api/settings" || path.startsWith("/api/settings?");
}

function isAnySettingsReadPath(path: string) {
  return isSettingsReadPath(path)
    || path === "/api/settings/providers"
    || path === "/api/settings/local";
}

function settingsPayloadForPath(path: string, settingsPayload: SettingsModel) {
  if (path === "/api/settings/providers") {
    return {
      provider_cards: settingsPayload.provider_cards,
      secret_storage_warning: settingsPayload.secret_storage_warning
    };
  }
  if (path === "/api/settings/local") {
    return {
      pending_jobs_display_mode: settingsPayload.pending_jobs_display_mode,
      user_narration_guidance: settingsPayload.user_narration_guidance,
      content_rating: settingsPayload.content_rating,
      fade_to_black: settingsPayload.fade_to_black,
      debug_logging: settingsPayload.debug_logging
    };
  }
  return isSettingsReadPath(path) ? settingsPayload : {};
}

function diagnosticsPayload(overrides: Partial<DiagnosticsModel> = {}): DiagnosticsModel {
  return {
    generated_at: "2026-07-08T12:00:00Z",
    filters: { save_id: null, categories: [], limit: 50, since: null },
    signals: [],
    maintenance_jobs: [],
    runtime_performance: { job_averages: [], step_averages: [], model_averages: [] },
    scheduler_health: {
      summary: { total: 0, healthy: 0, overdue: 0, leased: 0, failed: 0, disabled: 0 },
      tasks: []
    },
    web_events: [],
    active_save_health: null,
    ...overrides
  };
}

async function renderModelSettings(settingsPayload: SettingsModel) {
  const fetchMock = settingsFetch(settingsPayload);
  vi.stubGlobal("fetch", fetchMock);
  const { SettingsPanel } = await import("./main");

  render(
    <QueryClientProvider client={new QueryClient()}>
      <SettingsPanel runJob={vi.fn()} />
    </QueryClientProvider>
  );

  await userEvent.click(await screen.findByRole("tab", { name: "Models" }));
  const advancedRouting = screen.queryByRole("button", { name: /advanced model routing/i });
  if (advancedRouting && advancedRouting.getAttribute("aria-expanded") !== "true") {
    await userEvent.click(advancedRouting);
  }
  return fetchMock;
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function routingLane(label: string): HTMLElement {
  const row = screen.getByText(label).closest(".model-routing-row");
  expect(row).not.toBeNull();
  return row as HTMLElement;
}

async function expandRoutingLane(label: string): Promise<HTMLElement> {
  const row = routingLane(label);
  const toggle = within(row).getAllByRole("button", {
    name: new RegExp(escapeRegExp(label), "i")
  }).find((button) => button.hasAttribute("aria-expanded"));
  if (!toggle) throw new Error(`Could not find ${label} routing lane toggle`);
  if (toggle.getAttribute("aria-expanded") !== "true") {
    await userEvent.click(toggle);
  }
  return row;
}

async function expandOpenRouterAdvanced(): Promise<HTMLElement> {
  const toggle = screen.getByRole("button", { name: /advanced openrouter routing/i });
  if (toggle.getAttribute("aria-expanded") !== "true") {
    await userEvent.click(toggle);
  }
  const region = screen.getByRole("region", { name: /advanced openrouter routing/i });
  return region;
}

async function renderDiagnosticsSettings(
  settingsPayload: SettingsModel,
  diagnostics: DiagnosticsModel = diagnosticsPayload(),
  activeSaveId: string | null = null
) {
  const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
    ok: true,
    json: async () => {
      if (isAnySettingsReadPath(path)) return settingsPayloadForPath(path, settingsPayload);
      if (path.startsWith("/api/diagnostics")) return diagnostics;
      return {};
    }
  }));
  vi.stubGlobal("fetch", fetchMock);
  const { SettingsPanel } = await import("./main");

  render(
    <QueryClientProvider client={new QueryClient()}>
      <SettingsPanel runJob={vi.fn()} activeSaveId={activeSaveId} />
    </QueryClientProvider>
  );

  await userEvent.click(await screen.findByRole("tab", { name: "Diagnostics" }));
  return fetchMock;
}

async function renderAdminUserSettings(fetchMock: ReturnType<typeof vi.fn>) {
  vi.stubGlobal("fetch", fetchMock);
  const { SettingsPanel } = await import("./main");

  render(
    <QueryClientProvider client={new QueryClient()}>
      <SettingsPanel
        runJob={vi.fn()}
        currentUser={{ id: "admin-1", username: "Mira", role: "admin", status: "active" }}
      />
    </QueryClientProvider>
  );

  await userEvent.click(await screen.findByRole("tab", { name: "Users" }));
}

function workbenchFetch(
  activeJobs: Job[],
  model: RuntimeModel = runtimeModel(),
  scenarios: Scenario[] = [],
  submissionStatus: ChatSubmissionStatus = {
    save_id: model.active_save_id,
    can_submit: Boolean(model.active_save_id),
    reason: model.active_save_id ? null : "no_save",
    blocking_job_id: null,
    blocking_job_status: null
  },
  settings: Partial<SettingsModel> = {},
  characterTexts: {
    model: CharacterTextsModel;
    threads: Record<string, CharacterTextThread>;
    sendJob?: Job;
    spontaneousJob?: Job;
    contactUpdateModel?: CharacterTextsModel | ((path: string, init?: RequestInit) => CharacterTextsModel);
  } | null = null
) {
  return vi.fn().mockImplementation((path: string, init?: RequestInit) => Promise.resolve({
    ok: true,
    json: async () => {
      if (path.startsWith("/api/runtime")) return model;
      if (path === "/api/scenarios") return { scenarios };
      if (path.startsWith("/api/jobs?status=active")) return { jobs: activeJobs };
      if (path === "/api/settings/shell") {
        return {
          pending_jobs_display_mode: modelSettingsPayload(settings).pending_jobs_display_mode
        };
      }
      if (isAnySettingsReadPath(path)) return settingsPayloadForPath(path, modelSettingsPayload(settings));
      if (path.startsWith("/api/chat/submission-status")) return submissionStatus;
      if (path === "/api/character-texts/send-image" && characterTexts?.sendJob) return characterTexts.sendJob;
      if (path === "/api/character-texts/spontaneous" && characterTexts?.spontaneousJob) return characterTexts.spontaneousJob;
      if (path.startsWith("/api/character-texts/contacts/") && characterTexts) {
        if (typeof characterTexts.contactUpdateModel === "function") {
          return characterTexts.contactUpdateModel(path, init);
        }
        return characterTexts.contactUpdateModel ?? characterTexts.model;
      }
      if (path.startsWith("/api/character-texts/threads/") && path.endsWith("/send-image") && characterTexts?.sendJob) {
        return characterTexts.sendJob;
      }
      if (path.startsWith("/api/character-texts/threads/") && path.endsWith("/read") && characterTexts) {
        const threadId = decodeURIComponent(path.split("/api/character-texts/threads/")[1].split("/read")[0]);
        return {
          save_id: characterTexts.model.save_id,
          thread: characterTexts.threads[threadId],
          updated_message_ids: []
        };
      }
      if (path.startsWith("/api/character-texts/threads/") && characterTexts) {
        const threadId = decodeURIComponent(path.split("/api/character-texts/threads/")[1].split("?")[0]);
        return characterTexts.threads[threadId];
      }
      if (path.startsWith("/api/character-texts") && characterTexts) return characterTexts.model;
      if (path.startsWith("/api/jobs/") && path.includes("/cancel")) return { cancelled: true };
      if (path === "/api/chat/cancel") return { cancelled: true };
      return {};
    }
  }));
}

function characterTextsPayload(options: {
  rowanReferenceAssetId?: string | null;
  rowanContactOverrides?: Partial<CharacterTextContact>;
} = {}): { model: CharacterTextsModel; threads: Record<string, CharacterTextThread> } {
  const permission = (
    allowed: boolean,
    source: "none" | "chronicle" | "text_message" | "manual_or_legacy",
    reason: string,
  ) => ({
    allowed,
    source,
    reason,
    source_message_id: source === "chronicle" ? "message-phone" : null,
    source_text_message_id: source === "text_message" ? "text-phone" : null
  });
  const rowanReferenceImage = options.rowanReferenceAssetId === undefined
    ? null
    : options.rowanReferenceAssetId === null
      ? null
      : {
          media_asset_id: options.rowanReferenceAssetId,
          mime_type: "image/png",
          prompt_preview: "Rowan portrait",
          provider: "local",
          model: "upload",
          created_at: "2026-07-01T12:00:00Z",
          source: "uploaded"
        };
  const rowanContact: CharacterTextContact = {
    id: "character-rowan",
    name: "Rowan",
    contact_name: "",
    role: "CS major",
    status: "At the lab",
    is_player_character: false,
    player_has_character_number: true,
    character_has_player_number: true,
    player_number_permission: permission(true, "chronicle", "You can text them. Detected in the Chronicle."),
    character_number_permission: permission(true, "text_message", "They can text you. You texted them first."),
    thread_id: "thread-rowan",
    latest_message_id: "text-2",
    latest_message_body: "**Absolutely.** Bring `notes` after class.",
    latest_message_markdown_blocks: [
      {
        kind: "paragraph",
        spans: [
          { kind: "strong", text: "Absolutely." },
          { kind: "text", text: " Bring " },
          { kind: "inline_code", text: "notes" },
          { kind: "text", text: " after class." }
        ]
      }
    ],
    latest_message_sender: "character",
    latest_message_at: "2026-07-01T12:02:00Z",
    latest_message_read_at: null,
    reference_image: rowanReferenceImage,
    ...(options.rowanContactOverrides ?? {})
  };
  const mayaRepairContact: CharacterTextContact = {
    id: "character-maya",
    name: "Maya",
    role: "Club president",
    status: "Busy",
    is_player_character: false,
    player_has_character_number: false,
    character_has_player_number: false,
    player_number_permission: permission(false, "none", "You do not have this character's number."),
    character_number_permission: permission(false, "none", "They cannot text you yet."),
    thread_id: "thread-maya",
    latest_message_id: "text-maya-1",
    latest_message_body: "North campus Starbucks.",
    latest_message_markdown_blocks: [
      {
        kind: "paragraph",
        spans: [{ kind: "text", text: "North campus Starbucks." }]
      }
    ],
    latest_message_sender: "player",
    latest_message_at: "2026-07-01T12:01:00Z",
    latest_message_read_at: null,
    reference_image: null
  };
  return {
    model: {
      save_id: "save-1",
      enabled: true,
      contacts: [rowanContact],
      repair_contacts: [rowanContact, mayaRepairContact],
      threads: []
    },
    threads: {
      "thread-rowan": {
        id: "thread-rowan",
        character_id: "character-rowan",
        title: "Rowan",
        status: "active",
        created_at: "2026-07-01T12:00:00Z",
        updated_at: "2026-07-01T12:02:00Z",
        messages: [
          {
            id: "text-1",
            thread_id: "thread-rowan",
            character_id: "character-rowan",
            sender: "player",
            body: "Can we talk about **algorithms**?",
            delivery_status: "sent",
            markdown_blocks: [
              {
                kind: "paragraph",
                spans: [
                  { kind: "text", text: "Can we talk about " },
                  { kind: "strong", text: "algorithms" },
                  { kind: "text", text: "?" }
                ]
              }
            ]
          },
          {
            id: "text-2",
            thread_id: "thread-rowan",
            character_id: "character-rowan",
            sender: "character",
            body: "**Absolutely.** Bring `notes` after class.",
            delivery_status: "sent",
            markdown_blocks: [
              {
                kind: "paragraph",
                spans: [
                  { kind: "strong", text: "Absolutely." },
                  { kind: "text", text: " Bring " },
                  { kind: "inline_code", text: "notes" },
                  { kind: "text", text: " after class." }
                ]
              }
            ]
          }
        ]
      },
      "thread-maya": {
        id: "thread-maya",
        character_id: "character-maya",
        title: "Maya",
        status: "active",
        created_at: "2026-07-01T12:00:00Z",
        updated_at: "2026-07-01T12:01:00Z",
        messages: []
      }
    }
  };
}

it("cancels pending workbench read requests when the workbench unmounts", async () => {
  installEventSourceDouble();
  const pendingSignals: Record<string, AbortSignal | undefined> = {};
  const fetchMock = vi.fn().mockImplementation((rawPath: string, init?: RequestInit) => {
    const path = String(rawPath);
    if (path.startsWith("/api/runtime")) {
      return Promise.resolve({
        ok: true,
        json: async () => runtimeModel()
      });
    }
    if (
      path === "/api/settings/shell" ||
      path === "/api/saves/save-1/media" ||
      path.startsWith("/api/jobs?status=active") ||
      path.startsWith("/api/chat/submission-status")
    ) {
      pendingSignals[path] = init?.signal ?? undefined;
      return new Promise(() => undefined);
    }
    return Promise.resolve({
      ok: true,
      json: async () => ({})
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  const { Workbench } = await import("./main");

  const { unmount } = render(
    <QueryClientProvider client={new QueryClient()}>
      <Workbench />
    </QueryClientProvider>
  );

  await waitFor(() => expect(pendingSignals["/api/settings/shell"]).toBeDefined());
  await waitFor(() => expect(pendingSignals["/api/saves/save-1/media"]).toBeDefined());
  await waitFor(() => expect(pendingSignals["/api/jobs?status=active&save_id=save-1"]).toBeDefined());
  await waitFor(() => expect(pendingSignals["/api/chat/submission-status?save_id=save-1"]).toBeDefined());

  unmount();

  expect(pendingSignals["/api/settings/shell"]?.aborted).toBe(true);
  expect(pendingSignals["/api/saves/save-1/media"]?.aborted).toBe(true);
  expect(pendingSignals["/api/jobs?status=active&save_id=save-1"]?.aborted).toBe(true);
  expect(pendingSignals["/api/chat/submission-status?save_id=save-1"]?.aborted).toBe(true);
});

it("loads shell runtime first and defers noncritical workbench requests", async () => {
  installEventSourceDouble();
  stubWorkbenchMedia(false);
  const shellModel = runtimeModel({
    saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }],
    media: null
  });
  const fetchMock = vi.fn().mockImplementation((rawPath: string) => {
    const path = String(rawPath);
    const ok = (payload: unknown) => Promise.resolve({
      ok: true,
      status: 200,
      json: async () => payload
    });
    if (path.startsWith("/api/runtime/shell")) return ok(shellModel);
    if (path.startsWith("/api/runtime")) return ok(shellModel);
    if (path === "/api/settings/shell") {
      return ok({
        pending_jobs_display_mode: modelSettingsPayload().pending_jobs_display_mode
      });
    }
    if (isSettingsReadPath(path)) return ok(modelSettingsPayload());
    if (path.startsWith("/api/jobs?status=active")) return ok({ jobs: [] });
    if (path.startsWith("/api/chat/submission-status")) {
      return ok({
        save_id: "save-1",
        can_submit: true,
        reason: null,
        blocking_job_id: null,
        blocking_job_status: null
      });
    }
    if (path === "/api/saves/save-1/media") {
      return ok({
        latest_scene_media: null,
        latest_scene_image: null,
        image_history: [],
        media_history: []
      });
    }
    if (path === "/api/scenarios") {
      return ok({ scenarios: [scenarioFixture()] });
    }
    return ok({});
  });
  vi.stubGlobal("fetch", fetchMock);
  const { Workbench } = await import("./main");

  render(
    <QueryClientProvider client={new QueryClient()}>
      <Workbench currentUser={{ id: "user-1", username: "Mira", role: "admin", status: "active" }} />
    </QueryClientProvider>
  );

  expect(await screen.findByRole("heading", { name: "Lantern Keep" })).toBeInTheDocument();
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/runtime/shell", expect.anything()));
  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/saves/save-1/media", expect.anything()));
  expect(fetchMock.mock.calls.some(([path]) => path === "/api/runtime")).toBe(false);
  expect(fetchMock.mock.calls.some(([path]) => isSettingsReadPath(String(path)) && path !== "/api/settings/shell")).toBe(false);
  expect(fetchMock.mock.calls.some(([path]) => path === "/api/scenarios")).toBe(false);

  await userEvent.click(screen.getByRole("tab", { name: /Scenarios/ }));

  await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios", expect.anything()));
  expect(await screen.findByRole("button", { name: "Start Lantern Keep" })).toBeInTheDocument();
});

it("keeps the pending scenario library request alive while showing loading state", async () => {
  installEventSourceDouble();
  stubWorkbenchMedia(false);
  type FetchResponse = {
    ok: boolean;
    status: number;
    statusText?: string;
    json: () => Promise<unknown>;
  };
  const shellModel = runtimeModel({
    saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }],
    media: null
  });
  const scenarioResponse = deferred<FetchResponse>();
  let scenarioSignal: AbortSignal | undefined;
  const ok = (payload: unknown): Promise<FetchResponse> => Promise.resolve({
    ok: true,
    status: 200,
    json: async () => payload
  });
  const fetchMock = vi.fn().mockImplementation((rawPath: string, init?: RequestInit) => {
    const path = String(rawPath);
    if (path.startsWith("/api/runtime")) return ok(shellModel);
    if (path === "/api/settings/shell") {
      return ok({
        pending_jobs_display_mode: modelSettingsPayload().pending_jobs_display_mode
      });
    }
    if (path.startsWith("/api/jobs?status=active")) return ok({ jobs: [] });
    if (path.startsWith("/api/chat/submission-status")) {
      return ok({
        save_id: "save-1",
        can_submit: true,
        reason: null,
        blocking_job_id: null,
        blocking_job_status: null
      });
    }
    if (path === "/api/saves/save-1/media") {
      return ok({
        latest_scene_media: null,
        latest_scene_image: null,
        image_history: [],
        media_history: []
      });
    }
    if (path === "/api/scenarios") {
      scenarioSignal = init?.signal ?? undefined;
      return scenarioResponse.promise;
    }
    return ok({});
  });
  vi.stubGlobal("fetch", fetchMock);
  const { Workbench } = await import("./main");

  render(
    <QueryClientProvider client={new QueryClient()}>
      <Workbench currentUser={{ id: "user-1", username: "Mira", role: "admin", status: "active" }} />
    </QueryClientProvider>
  );

  await userEvent.click(await screen.findByRole("tab", { name: /Scenarios/ }));

  expect(await screen.findByText("Loading scenarios...")).toBeInTheDocument();
  await waitFor(() => expect(scenarioSignal).toBeDefined());
  expect(scenarioSignal?.aborted).toBe(false);

  scenarioResponse.resolve({
    ok: true,
    status: 200,
    json: async () => ({
      scenarios: [scenarioFixture({ title: "Fog Gate", premise: "A gate in the fog." })]
    })
  });

  expect(await screen.findByRole("button", { name: "Start Fog Gate" })).toBeInTheDocument();
});

function sessionFetch({
  bootstrapRequired,
  setupTokenRequired = false,
  meStatus = 200,
  runtimeStatus = 200,
  runtimeDetail = "Authentication required",
  model = runtimeModel()
}: {
  bootstrapRequired: boolean;
  setupTokenRequired?: boolean;
  meStatus?: number;
  runtimeStatus?: number;
  runtimeDetail?: string;
  model?: RuntimeModel;
}) {
  return vi.fn().mockImplementation((rawPath: string, init?: RequestInit) => {
    const path = String(rawPath);
    const method = init?.method ?? "GET";
    const ok = (payload: unknown) => Promise.resolve({
      ok: true,
      status: 200,
      json: async () => payload
    });
    const failure = (status: number, detail: string) => Promise.resolve({
      ok: false,
      status,
      statusText: detail,
      json: async () => ({ detail })
    });

    if (path === "/api/bootstrap/status") {
      return ok({
        admin_exists: !bootstrapRequired,
        bootstrap_required: bootstrapRequired,
        setup_token_required: setupTokenRequired
      });
    }
    if (path === "/api/auth/session") {
      return ok({
        bootstrap: {
          admin_exists: !bootstrapRequired,
          bootstrap_required: bootstrapRequired,
          setup_token_required: setupTokenRequired
        },
        user: !bootstrapRequired && meStatus === 200
          ? { id: "user-1", username: "Mira", role: "admin", status: "active" }
          : null
      });
    }
    if (path === "/api/bootstrap/admin" && method === "POST") {
      return ok({ user: { id: "user-1", username: "Mira", role: "admin", status: "active" } });
    }
    if (path === "/api/auth/login" && method === "POST") {
      return ok({ user: { id: "user-1", username: "Mira", role: "admin", status: "active" } });
    }
    if (path === "/api/auth/logout" && method === "POST") {
      return ok({ ok: true });
    }
    if (path === "/api/auth/me") {
      return meStatus === 200
        ? ok({ user: { id: "user-1", username: "Mira", role: "admin", status: "active" } })
        : failure(meStatus, "Authentication required");
    }
    if (path.startsWith("/api/runtime")) {
      return runtimeStatus === 200 ? ok(model) : failure(runtimeStatus, runtimeDetail);
    }
    if (path === "/api/scenarios") return ok({ scenarios: [] });
    if (path === "/api/settings") return ok(modelSettingsPayload());
    if (path.startsWith("/api/jobs?status=active")) return ok({ jobs: [] });
    if (path.startsWith("/api/chat/submission-status")) {
      return ok({
        save_id: model.active_save_id,
        can_submit: Boolean(model.active_save_id),
        reason: null,
        blocking_job_id: null,
        blocking_job_status: null
      });
    }
    if (path === "/api/log/client") return ok({ ok: true });
    return ok({});
  });
}

describe("frontend helpers", () => {
  it("offers supported scenario creation modes without retired Venice or single-character flows", async () => {
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = screen.getByRole("dialog", { name: "New scenario" });
    expect(within(dialog).getByRole("tab", { name: "Manual" })).toBeInTheDocument();
    expect(within(dialog).getByRole("tab", { name: "AI draft" })).toBeInTheDocument();
    expect(within(dialog).getByRole("tab", { name: "Dating Sim" })).toBeInTheDocument();
    expect(within(dialog).queryByRole("tab", { name: "Venice" })).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("tab", { name: "Single-character Dating Sim" })).not.toBeInTheDocument();
  });
  it("does not add a JSON content type to bodyless API reads", async () => {
    const { api } = await import("./api");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);

    await api("/api/saves");
    await api("/api/settings", { body: null });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    const nullBodyInit = fetchMock.mock.calls[1][1] as RequestInit;
    expect(init.headers).toBeUndefined();
    expect(nullBodyInit.headers).toBeUndefined();
  });

  it("logs failed API responses with metadata only", async () => {
    const { api } = await import("./api");
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/log/client"
        ? { ok: true, json: async () => ({ ok: true }) }
        : { ok: false, status: 500, statusText: "Server Error", json: async () => ({ detail: "boom" }) }
    ));
    vi.stubGlobal("fetch", fetchMock);

    await expect(api("/api/chat", { method: "POST", body: JSON.stringify({ body: "secret chat text" }) })).rejects.toThrow("boom");

    const logCall = fetchMock.mock.calls.find(([path]) => path === "/api/log/client");
    expect(logCall).toBeTruthy();
    const payload = JSON.parse(String(logCall?.[1].body));
    expect(payload).toMatchObject({
      level: "error",
      event: "client.api.failed",
      fields: { method: "POST", path: "/api/chat", status_code: 500 }
    });
    expect(String(logCall?.[1].body)).not.toContain("secret chat text");
  });

  it("adds JSON content type only when unsafe API helpers send JSON bodies", async () => {
    const { deleteJson, postJson } = await import("./api");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);

    await postJson("/api/chat", { body: "Light the beacon" });
    await deleteJson("/api/saves/save-1");
    await deleteJson("/api/saves/save-2", { reason: "mistake" });

    const postInit = fetchMock.mock.calls[0][1] as RequestInit;
    const deleteInit = fetchMock.mock.calls[1][1] as RequestInit;
    const deleteWithBodyInit = fetchMock.mock.calls[2][1] as RequestInit;
    expect(postInit.headers).toMatchObject({
      "Content-Type": "application/json",
      "X-Bragi-Api-Request": "1"
    });
    expect(deleteInit.headers).toEqual({ "X-Bragi-Api-Request": "1" });
    expect(deleteWithBodyInit.headers).toMatchObject({
      "Content-Type": "application/json",
      "X-Bragi-Api-Request": "1"
    });
  });

  it("adds the Bragi write guard header without overriding FormData content type", async () => {
    const { api } = await import("./api");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    const form = new FormData();
    form.append("file", new Blob(["bundle"]), "save.bragi");

    await api("/api/bundles/preview", { method: "POST", body: form });

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.headers).toEqual({ "X-Bragi-Api-Request": "1" });
  });

  it("sanitizes client log fields before sending", async () => {
    const { logClientEvent } = await import("./api");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchMock);

    logClientEvent("error", "client.test", {
      component: "Composer",
      body: "secret body",
      api_key: "secret-key",
      error_message: "x".repeat(400)
    });

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/log/client", expect.anything()));
    const payload = JSON.parse(String(fetchMock.mock.calls[0][1].body));
    expect(payload.fields.component).toBe("Composer");
    expect(payload.fields.error_message).toHaveLength(240);
    expect(fetchMock.mock.calls[0][1].headers).toMatchObject({
      "Content-Type": "application/json",
      "X-Bragi-Api-Request": "1"
    });
    expect(String(fetchMock.mock.calls[0][1].body)).not.toContain("secret body");
    expect(String(fetchMock.mock.calls[0][1].body)).not.toContain("secret-key");
  });

  it("keeps the dev API proxy on the browser origin for write guards", async () => {
    const { apiProxy } = await import("./devProxy");

    expect(apiProxy).toMatchObject({
      target: "http://127.0.0.1:8787",
      changeOrigin: false
    });
  });

  it("logs job SSE fallback and terminal failures", async () => {
    const { watchJob } = await import("./api");
    let source: EventSourceDouble | null = null;
    class EventSourceDouble {
      onerror: (() => void) | null = null;
      constructor() {
        source = this;
      }
      addEventListener() {}
      close() {}
    }
    vi.stubGlobal("EventSource", EventSourceDouble);
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/log/client"
        ? { ok: true, json: async () => ({ ok: true }) }
        : {
            ok: true,
            json: async () => ({
              id: "job-1",
              type: "chat_turn",
              status: "failed",
              result: null,
              error: "provider echoed Mara private prompt with api_key=live-secret"
            })
          }
    ));
    vi.stubGlobal("fetch", fetchMock);

    watchJob("job-1", vi.fn());
    expect(source).not.toBeNull();
    const createdSource = source as unknown as { onerror: (() => void) | null };
    createdSource.onerror?.();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/job-1", expect.anything()));
    const logBodies = fetchMock.mock.calls
      .filter(([path]) => path === "/api/log/client")
      .map(([, init]) => String(init.body));
    expect(logBodies.some((body) => body.includes("client.job.sse_fallback"))).toBe(true);
    expect(logBodies.some((body) => body.includes("client.job.failed"))).toBe(true);
    expect(logBodies.join("\n")).not.toContain("Mara private prompt");
    expect(logBodies.join("\n")).not.toContain("live-secret");
    expect(logBodies.join("\n")).not.toContain("api_key");
  });

  it("falls back from native EventSource errors without parsing empty event data", async () => {
    const { watchJob } = await import("./api");
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/log/client"
        ? { ok: true, json: async () => ({ ok: true }) }
        : { ok: true, json: async () => ({ id: "job-1", type: "chat_turn", status: "succeeded", result: null, error: null }) }
    ));
    vi.stubGlobal("fetch", fetchMock);

    const updates = vi.fn();
    watchJob("job-1", updates);

    expect(() => sources[0].dispatchNativeError()).not.toThrow();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/job-1", expect.anything()));
    expect(updates).toHaveBeenCalledWith(expect.objectContaining({ id: "job-1", status: "succeeded" }));
  });

  it("falls back from malformed job done events", async () => {
    const { watchJob } = await import("./api");
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/log/client"
        ? { ok: true, json: async () => ({ ok: true }) }
        : { ok: true, json: async () => ({ id: "job-1", type: "chat_turn", status: "succeeded", result: null, error: null }) }
    ));
    vi.stubGlobal("fetch", fetchMock);

    const updates = vi.fn();
    watchJob("job-1", updates);

    expect(() => sources[0].dispatchRaw("done", "{not json")).not.toThrow();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/job-1", expect.anything()));
    expect(updates).toHaveBeenCalledWith(expect.objectContaining({ id: "job-1", status: "succeeded" }));
    const logBodies = fetchMock.mock.calls
      .filter(([path]) => path === "/api/log/client")
      .map(([, init]) => String(init.body));
    expect(logBodies.some((body) => body.includes("client.sse.parse_failed"))).toBe(true);
    expect(logBodies.join("\n")).not.toContain("{not json");
  });

  it("falls back from malformed job progress events without emitting progress", async () => {
    const { watchJob } = await import("./api");
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/log/client"
        ? { ok: true, json: async () => ({ ok: true }) }
        : { ok: true, json: async () => ({ id: "job-1", type: "chat_turn", status: "succeeded", result: null, error: null }) }
    ));
    vi.stubGlobal("fetch", fetchMock);

    const updates = vi.fn();
    const events = vi.fn();
    watchJob("job-1", updates, events);

    expect(() => sources[0].dispatchRaw("progress", "{not json")).not.toThrow();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/job-1", expect.anything()));
    expect(events).not.toHaveBeenCalled();
    expect(updates).toHaveBeenCalledWith(expect.objectContaining({ id: "job-1", status: "succeeded" }));
  });

  it("delivers chat turn delta job events", async () => {
    const { watchJob } = await import("./api");
    const sources = installEventSourceDouble();
    const events = vi.fn();
    const stop = watchJob("job-1", vi.fn(), events, "save-1");
    const delta: ChatTurnDelta = {
      kind: "chat_turn_delta",
      version: 1,
      save_id: "save-1",
      status: "Turn complete",
      error: null,
      player_message_id: "player-1",
      narrator_message_id: "narrator-1",
      messages: [
        { message_id: "player-1", role: "player", speaker_name: "Keeper", body: "Light the beacon", actions: [] },
        { message_id: "narrator-1", role: "narrator", speaker_name: null, body: "The bell answers.", actions: [] }
      ],
      action_choices: null,
      save: {
        save_id: "save-1",
        title: "Lantern Keep",
        active: true,
        scenario_id: "scenario-1",
        scenario_title: "Lantern Keep"
      },
      fallback_used: false,
      context_trimmed: false
    };

    sources[0].dispatch("chat_turn_delta", delta);
    stop();

    expect(events).toHaveBeenCalledWith("chat_turn_delta", delta);
  });

  it("scopes job event streams and fallback polling to a save", async () => {
    const { watchJob } = await import("./api");
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/log/client"
        ? { ok: true, json: async () => ({ ok: true }) }
        : { ok: true, json: async () => ({ id: "job-1", type: "chat_turn", save_id: "save-1", status: "succeeded", result: null, error: null }) }
    ));
    vi.stubGlobal("fetch", fetchMock);

    const updates = vi.fn();
    watchJob("job-1", updates, undefined, "save-1");

    expect(sources[0].url).toBe("/api/jobs/job-1/events?save_id=save-1");
    sources[0].dispatchNativeError();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/job-1?save_id=save-1", expect.anything()));
    expect(updates).toHaveBeenCalledWith(expect.objectContaining({ id: "job-1", save_id: "save-1", status: "succeeded" }));
  });

  it("ignores duplicate and out-of-order save events", async () => {
    const { watchSave } = await import("./api");
    const sources = installEventSourceDouble();
    const updates = vi.fn();

    const stop = watchSave("save-1", updates);

    expect(sources[0].url).toBe("/api/saves/save-1/events");
    act(() => {
      sources[0].dispatch("runtime_changed", {
        event_id: 2,
        save_id: "save-1",
        type: "runtime_changed",
        payload: { step: "first" }
      });
      sources[0].dispatch("runtime_changed", {
        event_id: 2,
        save_id: "save-1",
        type: "runtime_changed",
        payload: { step: "duplicate" }
      });
      sources[0].dispatch("runtime_changed", {
        event_id: 1,
        save_id: "save-1",
        type: "runtime_changed",
        payload: { step: "older" }
      });
      sources[0].dispatch("runtime_changed", {
        event_id: 3,
        save_id: "save-1",
        type: "runtime_changed",
        payload: { step: "newer" }
      });
    });
    stop();

    expect(updates).toHaveBeenCalledTimes(2);
    expect(updates).toHaveBeenNthCalledWith(1, expect.objectContaining({ event_id: 2 }));
    expect(updates).toHaveBeenNthCalledWith(2, expect.objectContaining({ event_id: 3 }));
  });

  it("recovers from malformed save events without delivering stale data", async () => {
    const { watchSave } = await import("./api");
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ ok: true }) });
    vi.stubGlobal("fetch", fetchMock);
    const updates = vi.fn();
    const recover = vi.fn();

    const stop = watchSave("save-1", updates, recover);

    expect(() => sources[0].dispatchRaw("runtime_changed", "{not json")).not.toThrow();
    stop();

    expect(updates).not.toHaveBeenCalled();
    expect(recover).toHaveBeenCalledTimes(1);
    const logBodies = fetchMock.mock.calls
      .filter(([path]) => path === "/api/log/client")
      .map(([, init]) => String(init.body));
    expect(logBodies.some((body) => body.includes("client.sse.parse_failed"))).toBe(true);
    expect(logBodies.join("\n")).not.toContain("{not json");
  });

  it("stops fallback polling when a job is no longer known", async () => {
    const { watchJob } = await import("./api");
    const sources = installEventSourceDouble();
    const setTimeoutSpy = vi.spyOn(window, "setTimeout");
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/log/client"
        ? { ok: true, json: async () => ({ ok: true }) }
        : { ok: false, status: 404, statusText: "Not Found", json: async () => ({ detail: "Unknown job" }) }
    ));
    vi.stubGlobal("fetch", fetchMock);

    const updates = vi.fn();
    watchJob("job-1", updates);
    await act(async () => {
      sources[0].dispatchNativeError();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(updates).toHaveBeenCalledWith(expect.objectContaining({
      id: "job-1",
      status: "cancelled",
      error: "Unknown job"
    }));
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/jobs/job-1")).toHaveLength(1);
    expect(setTimeoutSpy).not.toHaveBeenCalled();
    expect(fetchMock.mock.calls.some(([path, init]) => path === "/api/log/client" && String(init.body).includes("client.job.stale"))).toBe(true);
    setTimeoutSpy.mockRestore();
  });

  it("keeps retrying fallback polling after transient job read failures", async () => {
    const { watchJob } = await import("./api");
    const sources = installEventSourceDouble();
    let retryPoll: (() => void) | undefined;
    const setTimeoutSpy = vi.spyOn(window, "setTimeout").mockImplementation((callback: TimerHandler) => {
      retryPoll = callback as () => void;
      return 1;
    });
    const fetchMock = vi.fn()
      .mockImplementationOnce((path: string) => Promise.resolve(path === "/api/log/client" ? { ok: true, json: async () => ({ ok: true }) } : { ok: true, json: async () => ({ ok: true }) }))
      .mockRejectedValueOnce(new TypeError("network down"))
      .mockImplementation((path: string) => Promise.resolve(
        path === "/api/log/client"
          ? { ok: true, json: async () => ({ ok: true }) }
          : { ok: true, json: async () => ({ id: "job-1", type: "chat_turn", status: "succeeded", result: null, error: null }) }
      ));
    vi.stubGlobal("fetch", fetchMock);

    const updates = vi.fn();
    watchJob("job-1", updates);
    await act(async () => {
      sources[0].dispatchNativeError();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(fetchMock).toHaveBeenCalledWith("/api/jobs/job-1", expect.anything());
    expect(retryPoll).toBeDefined();
    await act(async () => {
      retryPoll?.();
      await Promise.resolve();
      await Promise.resolve();
    });

    expect(updates).toHaveBeenCalledWith(expect.objectContaining({ id: "job-1", status: "succeeded" }));
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/jobs/job-1").length).toBeGreaterThanOrEqual(2);
    setTimeoutSpy.mockRestore();
  });

  it("does not mount the workbench as an import side effect in tests", async () => {
    vi.stubGlobal("fetch", vi.fn());

    const root = document.getElementById("root")!;
    await import("./main");

    expect(root.childElementCount).toBe(0);
  });

  it("applies chat turn deltas without duplicating messages", async () => {
    const { applyChatTurnDeltaToRuntimeModel } = await import("./main");
    const model = runtimeModel({
      saves: [
        {
          save_id: "save-1",
          title: "Lantern Keep",
          active: true,
          scenario_id: "scenario-1",
          scenario_title: "Lantern Keep"
        }
      ],
      chronicle: {
        messages: [
          { message_id: "opening-1", role: "narrator", speaker_name: null, body: "The beacon snaps awake.", actions: [] }
        ]
      }
    });
    const delta: ChatTurnDelta = {
      kind: "chat_turn_delta",
      version: 1,
      save_id: "save-1",
      status: "Turn complete",
      error: null,
      player_message_id: "player-1",
      narrator_message_id: "narrator-1",
      messages: [
        { message_id: "player-1", role: "player", speaker_name: "Keeper", body: "Light the beacon", actions: [] },
        { message_id: "narrator-1", role: "narrator", speaker_name: null, body: "The bell answers.", actions: [] }
      ],
      action_choices: cyoaActionChoices({ narrator_message_id: "narrator-1" }),
      save: {
        save_id: "save-1",
        title: "Lantern Keep Updated",
        active: true,
        scenario_id: "scenario-1",
        scenario_title: "Lantern Keep"
      },
      fallback_used: false,
      context_trimmed: false
    };

    const once = applyChatTurnDeltaToRuntimeModel(model, delta);
    const twice = applyChatTurnDeltaToRuntimeModel(once, delta);

    expect(twice.chronicle.messages.map((message) => message.message_id)).toEqual([
      "opening-1",
      "player-1",
      "narrator-1"
    ]);
    expect(twice.chronicle.messages[2].body).toBe("The bell answers.");
    expect(twice.action_choices?.narrator_message_id).toBe("narrator-1");
    expect(twice.saves[0].title).toBe("Lantern Keep Updated");
    expect(twice.status).toBe("Turn complete");
    expect(twice.error).toBeNull();
  });

  it("hides pending character text rows even after reply text exists", async () => {
    const { isPlaceholderCharacterTextDelivery } = await import("./main");
    const pendingCharacterMessage = {
      id: "text-1",
      thread_id: "thread-1",
      character_id: "character-1",
      sender: "character",
      body: "Found the ticket stub. Sending proof.",
      delivery_status: "pending",
      attachments: [],
      actions: []
    } as unknown as CharacterTextMessage;

    expect(isPlaceholderCharacterTextDelivery(pendingCharacterMessage)).toBe(true);
    expect(isPlaceholderCharacterTextDelivery({
      ...pendingCharacterMessage,
      delivery_status: "sent"
    })).toBe(false);
    expect(isPlaceholderCharacterTextDelivery({
      ...pendingCharacterMessage,
      sender: "player",
      delivery_status: "pending"
    })).toBe(false);
  });

  it("global error handlers log uncaught browser errors", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true });
    vi.stubGlobal("fetch", fetchMock);
    const { installGlobalErrorLogging } = await import("./main");

    installGlobalErrorLogging();
    window.dispatchEvent(new ErrorEvent("error", { message: "Render failed", error: new TypeError("Render failed") }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/log/client", expect.anything()));
    expect(String(fetchMock.mock.calls[0][1].body)).toContain("client.window.error");
    expect(String(fetchMock.mock.calls[0][1].body)).toContain("TypeError");
  });

  it("renders recent web event diagnostics rows", async () => {
    const { WebEventsList } = await import("./main");

    render(
      <WebEventsList
        events={[
          {
            timestamp: "2026-05-25T12:00:00Z",
            level: "error",
            event: "client.api.failed",
            status_code: 500,
            duration_ms: 12,
            error_class: "ApiError",
            route: "/api/chat"
          }
        ]}
      />
    );

    expect(screen.getByText("client.api.failed")).toBeInTheDocument();
    expect(screen.getByText("500 · 12 ms · ApiError · /api/chat")).toBeInTheDocument();
  });

  it("renders chat history rows and changes filters", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      const query = new URL(String(path), "http://bragi.local").searchParams;
      const filter = query.get("filter") ?? "all";
      const beforeMessageId = query.get("before_message_id");
      const withImages = filter === "with_images";
      const messages = withImages
        ? [
            {
              message_id: "message-2",
              role: "narrator",
              role_label: "Narrator",
              speaker_name: null,
              body: "The beacon catches the fog.",
              markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: "The beacon catches the fog." }] }],
              style_class: "narrator",
              provider: "fake",
              model: "fake-chat",
              provider_model_label: "fake / fake-chat",
              token_estimate: 42,
              created_at: "2026-05-29T12:00:00Z",
              image_count: 1
            }
          ]
        : beforeMessageId
          ? [
              {
                message_id: "message-1",
                role: "player",
                role_label: "Player",
                speaker_name: "Mara",
                body: "I light the lantern.",
                markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: "I light the lantern." }] }],
                style_class: "player",
                provider: null,
                model: null,
                token_estimate: null,
                created_at: "2026-05-29T11:59:00Z",
                image_count: 0
              }
            ]
          : [
              {
                message_id: "message-2",
                role: "narrator",
                role_label: "Narrator",
                speaker_name: null,
                body: "The beacon catches the fog.",
                markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: "The beacon catches the fog." }] }],
                style_class: "narrator",
                provider: "fake",
                model: "fake-chat",
                provider_model_label: "fake / fake-chat",
                token_estimate: 42,
                created_at: "2026-05-29T12:00:00Z",
                image_count: 1
              }
            ];
      return Promise.resolve({
        ok: true,
        json: async () => ({
        active_save_id: "save-1",
        active_save_title: "Lantern Keep",
        selected_filter: filter,
        filter_options: [
          { filter_id: "all", label: "All", active: filter === "all" },
          { filter_id: "with_images", label: "With images", active: withImages }
        ],
        messages,
        total_message_count: 3,
        matching_message_count: withImages ? 1 : 3,
        has_more_before: !withImages && !beforeMessageId,
        oldest_message_id: messages[0]?.message_id ?? null,
        empty_title: "",
        empty_detail: ""
      })
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { HistoryPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <HistoryPanel activeSaveId="save-1" />
      </QueryClientProvider>
    );

    expect(await screen.findByText("The beacon catches the fog.")).toBeInTheDocument();
    expect(screen.getByText("fake / fake-chat")).toBeInTheDocument();
    expect(screen.getByText("42 tokens")).toBeInTheDocument();
    expect(screen.getByText("1 images")).toBeInTheDocument();
    expect(screen.getByText("Showing 1 of 3 matching · 3 total")).toBeInTheDocument();
    expect(String(fetchMock.mock.calls[0][0])).toContain("limit=80");

    await userEvent.click(screen.getByRole("button", { name: /load earlier/i }));

    expect(await screen.findByText("I light the lantern.")).toBeInTheDocument();
    const older = screen.getByText("I light the lantern.");
    const latest = screen.getByText("The beacon catches the fog.");
    expect(Boolean(older.compareDocumentPosition(latest) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => String(path).includes("before_message_id=message-2"))).toBe(true));

    await userEvent.click(screen.getByRole("tab", { name: "With images" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => String(path).includes("filter=with_images"))).toBe(true));
    expect(screen.getByText("Showing 1 of 1 matching · 3 total")).toBeInTheDocument();
  });

  it("submits regenerate feedback from narrator actions", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: "job-1",
        type: "chat_regenerate",
        status: "queued",
        result: null,
        error: null
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "A long answer.",
            actions: [
              {
                action_id: "regenerate-message-with-feedback",
                label: "Regenerate with feedback"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={runJob} pendingMessage={null} />);

    await userEvent.click(screen.getByTitle("Regenerate with feedback"));
    await userEvent.type(screen.getByLabelText("Feedback"), "Make it shorter.");
    await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/regenerate", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/chat/regenerate");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      message_id: "narrator-1",
      save_id: "save-1",
      regeneration_feedback: "Make it shorter."
    });
    expect(runJob).toHaveBeenCalledWith({
      id: "job-1",
      type: "chat_regenerate",
      status: "queued",
      result: null,
      error: null
    });
  });

  it("keeps regenerate feedback open and usable when submission fails", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      json: async () => ({ detail: "Regeneration could not start." })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "A long answer.",
            actions: [
              {
                action_id: "regenerate-message-with-feedback",
                label: "Regenerate with feedback"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={runJob} pendingMessage={null} />);

    await userEvent.click(screen.getByTitle("Regenerate with feedback"));
    await userEvent.type(screen.getByLabelText("Feedback"), "Make it shorter.");
    await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));

    const dialog = screen.getByRole("dialog", { name: "Regenerate with feedback" });
    expect(await within(dialog).findByText("Regeneration could not start.")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Regenerate" })).toBeEnabled();
    expect(within(dialog).getByLabelText("Feedback")).toHaveValue("Make it shorter.");
    expect(runJob).not.toHaveBeenCalled();
  });

  it("keeps high-impact chronicle action names visible for touch layouts", async () => {
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "A long answer.",
            actions: [
              { action_id: "regenerate-message", label: "Regenerate" },
              { action_id: "delete-messages-from-here", label: "Delete from here" }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={vi.fn()} pendingMessage={null} />);

    const regenerate = screen.getByRole("button", { name: "Regenerate" });
    expect(regenerate).toHaveClass("touch-labeled-action");
    expect(within(regenerate).getByText("Regenerate")).toHaveClass("touch-action-label");
    const deleteFromHere = screen.getByRole("button", { name: "Delete from here" });
    expect(deleteFromHere).toHaveClass("touch-labeled-action");
    expect(within(deleteFromHere).getByText("Delete from here")).toHaveClass("touch-action-label");
  });

  it("submits scene images directly and character images through the chooser", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      if (String(path).includes("/scene-presence")) {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            save_id: "save-1",
            message_id: "narrator-1",
            latest_message: true,
            characters: [
              {
                character_id: "character-oracle",
                name: "Oracle",
                present: true,
                has_reference_image: true,
                reference_image: {
                  media_asset_id: "reference-1",
                  mime_type: "image/png",
                  prompt_preview: "Oracle reference",
                  provider: "local",
                  model: "upload"
                },
                is_player_character: false,
                status: "present"
              }
            ]
          })
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          id: path === "/api/media/generate-character-image" ? "job-character-image" : "job-scene-image",
          type: path === "/api/media/generate-character-image" ? "character_image_generation" : "image_generation",
          status: "queued",
          result: null,
          error: null
        })
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      active_scenario_type: "dating_sim",
      chronicle: {
        messages: [
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "The oracle turns toward the moonlit window.",
            actions: [
              {
                action_id: "generate-scene-image",
                label: "Generate image of this scene"
              },
              {
                action_id: "generate-character-image",
                label: "Generate image of a character"
              }
            ]
          }
        ]
      }
    });

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Chronicle model={model} runJob={runJob} pendingMessage={null} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByTitle("Generate image of this scene"));
    await userEvent.click(screen.getByTitle("Generate image of a character"));
    const dialog = await screen.findByRole("dialog", { name: "Generate character image" });
    await userEvent.click(within(dialog).getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/media/generate-character-image", expect.anything()));
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/media/generate",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ message_id: "narrator-1", save_id: "save-1" })
      })
    );
    const characterImageCall = fetchMock.mock.calls.find(
      ([path, init]) => path === "/api/media/generate-character-image" && init?.method === "POST"
    );
    expect(JSON.parse(String(characterImageCall?.[1].body))).toEqual({
      save_id: "save-1",
      message_id: "narrator-1",
      character_id: "character-oracle"
    });
    expect(runJob).toHaveBeenCalledWith(expect.objectContaining({ id: "job-scene-image" }));
    expect(runJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: "job-character-image" }),
      expect.objectContaining({ onSucceeded: expect.any(Function) })
    );
  });

  it("invalidates scene-presence cache for the active save after reference changes", async () => {
    const { invalidateScenePresenceQueries } = await import("./main");
    const client = new QueryClient();
    const invalidateSpy = vi.spyOn(client, "invalidateQueries");

    invalidateScenePresenceQueries(client, "save-1");
    invalidateScenePresenceQueries(client, null);

    expect(invalidateSpy).toHaveBeenNthCalledWith(1, {
      queryKey: ["scene-presence", "save-1"]
    });
    expect(invalidateSpy).toHaveBeenNthCalledWith(2, { queryKey: ["scene-presence"] });
  });

  it("edits characters present for a chronicle message", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (String(path).includes("/scene-presence") && init?.method !== "POST") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            save_id: "save-1",
            message_id: "narrator-1",
            latest_message: true,
            characters: [
              {
                character_id: "character-oracle",
                name: "Oracle",
                present: true,
                has_reference_image: true,
                reference_image: null,
                is_player_character: false,
                status: "present"
              },
              {
                character_id: "character-mara",
                name: "Mara",
                present: false,
                has_reference_image: false,
                reference_image: null,
                is_player_character: true,
                status: ""
              }
            ]
          })
        });
      }
      return Promise.resolve({
        ok: true,
        json: async () => ({
          save_id: "save-1",
          message_id: "narrator-1",
          latest_message: true,
          characters: []
        })
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      active_scenario_type: "dating_sim",
      chronicle: {
        messages: [
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "The oracle turns toward the moonlit window.",
            actions: [
              {
                action_id: "view-characters-present",
                label: "Characters present"
              }
            ]
          }
        ]
      }
    });

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Chronicle model={model} runJob={vi.fn()} pendingMessage={null} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByTitle("Characters present"));
    const dialog = await screen.findByRole("dialog", { name: "Characters present" });
    await userEvent.click(within(dialog).getByLabelText(/Mara/));
    await userEvent.click(within(dialog).getByRole("button", { name: "Save" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/messages/narrator-1/scene-presence",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          save_id: "save-1",
          character_ids: ["character-oracle", "character-mara"]
        })
      })
    ));
  });

  it("confirms delete-from-here actions before refreshing runtime", async () => {
    const job = { id: "job-delete", type: "chat_delete_from_here", save_id: "save-1", status: "queued", result: null, error: null };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => job
    });
    vi.stubGlobal("fetch", fetchMock);
    const onRuntimeChanged = vi.fn();
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      active_save_id: "save-1",
      chronicle: {
        messages: [
          {
            message_id: "narrator-2",
            role: "narrator",
            speaker_name: null,
            body: "The second answer.",
            actions: [
              {
                action_id: "delete-messages-from-here",
                label: "Delete from here"
              }
            ]
          }
        ]
      }
    });

    render(
      <Chronicle
        model={model}
        runJob={runJob}
        pendingMessage={null}
        onRuntimeChanged={onRuntimeChanged}
      />
    );

    await userEvent.click(screen.getByTitle("Delete from here"));
    expect(screen.getByRole("dialog", { name: "Delete from here?" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/chat/delete-from-here")).toBe(false);

    await userEvent.click(screen.getByTitle("Delete from here"));
    const dialog = screen.getByRole("dialog", { name: "Delete from here?" });
    await userEvent.click(within(dialog).getByRole("button", { name: "Delete from here" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/delete-from-here", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/chat/delete-from-here");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      message_id: "narrator-2",
      save_id: "save-1"
    });
    expect(runJob).toHaveBeenCalledWith(job);
    expect(onRuntimeChanged).not.toHaveBeenCalled();
  });

  it("confirms fork-from-here actions before switching to the returned save", async () => {
    const updated = runtimeModel({
      active_save_id: "fork-1",
      active_save_title: "Lantern Keep - fork after narrator 2",
      chronicle: { messages: [] },
      status: "Save forked"
    });
    const job = { id: "job-fork", type: "chat_fork_from_here", save_id: "save-1", status: "queued", result: null, error: null };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => job
    });
    vi.stubGlobal("fetch", fetchMock);
    const onRuntimeChanged = vi.fn();
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      active_save_id: "save-1",
      chronicle: {
        messages: [
          {
            message_id: "narrator-2",
            role: "narrator",
            speaker_name: null,
            body: "The second answer.",
            actions: [
              {
                action_id: "fork-from-here",
                label: "Fork from here"
              }
            ]
          }
        ]
      }
    });

    render(
      <Chronicle
        model={model}
        runJob={runJob}
        pendingMessage={null}
        onRuntimeChanged={onRuntimeChanged}
      />
    );

    await userEvent.click(screen.getByTitle("Fork from here"));
    expect(screen.getByRole("dialog", { name: "Fork from here?" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/chat/fork-from-here")).toBe(false);

    await userEvent.click(screen.getByTitle("Fork from here"));
    const dialog = screen.getByRole("dialog", { name: "Fork from here?" });
    await userEvent.click(within(dialog).getByRole("button", { name: "Fork from here" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/fork-from-here", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/chat/fork-from-here");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      message_id: "narrator-2",
      save_id: "save-1"
    });
    expect(runJob).toHaveBeenCalledWith(job, expect.objectContaining({ onSucceeded: expect.any(Function) }));
    const options = runJob.mock.calls[0][1];
    options.onSucceeded(updated);
    expect(onRuntimeChanged).toHaveBeenCalledWith(updated);
  });

  it("keeps player edit choices in the mobile-safe dialog footer", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: "job-chat-edit",
        type: "chat_edit",
        status: "queued",
        result: null,
        error: null
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "player-1",
            role: "player",
            speaker_name: "Keeper",
            body: "Hold the line.",
            actions: [
              {
                action_id: "edit-and-resubmit-message",
                label: "Edit this message"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={runJob} pendingMessage={null} />);

    await userEvent.click(screen.getByTitle("Edit this message"));

    const dialog = screen.getByRole("dialog", { name: "Edit message" });
    expect(dialog).toHaveClass("message-edit-dialog");
    expect(dialog.querySelector(".message-edit-actions")).not.toBeNull();
    expect(within(dialog).getByRole("button", { name: "Edit without Resubmit" })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Resubmit" })).toBeInTheDocument();
    const editor = within(dialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "Hold the east line.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Resubmit" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/edit", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/chat/edit");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      message_id: "player-1",
      save_id: "save-1",
      body: "Hold the east line."
    });
    expect(runJob).toHaveBeenCalledWith({
      id: "job-chat-edit",
      type: "chat_edit",
      status: "queued",
      result: null,
      error: null
    });
  });

  it("closes open Chronicle mutation dialogs when the active save becomes unsupported", async () => {
    const { Chronicle } = await import("./main");
    const supportedModel = runtimeModel({
      active_save_id: "save-1",
      chronicle: {
        messages: [{
          message_id: "player-1",
          role: "player",
          speaker_name: "Keeper",
          body: "Hold the line.",
          actions: [{ action_id: "edit-and-resubmit-message", label: "Edit this message" }]
        }]
      }
    });
    const { rerender } = render(
      <Chronicle model={supportedModel} runJob={vi.fn()} pendingMessage={null} mutationsDisabled={false} />
    );

    await userEvent.click(screen.getByTitle("Edit this message"));
    expect(screen.getByRole("dialog", { name: "Edit message" })).toBeInTheDocument();

    rerender(
      <Chronicle
        model={runtimeModel({ ...supportedModel, active_save_id: "save-retired" })}
        runJob={vi.fn()}
        pendingMessage={null}
        mutationsDisabled
      />
    );

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Edit message" })).not.toBeInTheDocument());
    expect(screen.getByTitle("Edit this message")).toBeDisabled();
  });

  it("saves player edits without replaying the turn", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: "job-message-edit",
        type: "message_edit",
        status: "queued",
        result: null,
        error: null
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      active_save_id: "save-1",
      chronicle: {
        messages: [
          {
            message_id: "player-1",
            role: "player",
            speaker_name: "Keeper",
            body: "Hold the line.",
            actions: [
              {
                action_id: "edit-and-resubmit-message",
                label: "Edit this message"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={runJob} pendingMessage={null} />);

    await userEvent.click(screen.getByTitle("Edit this message"));
    const dialog = screen.getByRole("dialog", { name: "Edit message" });
    const editor = within(dialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "Hold the east line.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Edit without Resubmit" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/message-edit", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/chat/message-edit");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      message_id: "player-1",
      save_id: "save-1",
      body: "Hold the east line."
    });
    expect(runJob).toHaveBeenCalledWith({
      id: "job-message-edit",
      type: "message_edit",
      status: "queued",
      result: null,
      error: null
    });
  });

  it("saves narrator edits without replaying the turn", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: "job-narrator-edit",
        type: "narrator_edit",
        status: "queued",
        result: null,
        error: null
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      active_save_id: "save-1",
      chronicle: {
        messages: [
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "The corridor floods with ash.",
            revision_count: 1,
            edited_at: "2026-06-02 15:30:00",
            actions: [
              {
                action_id: "edit-narrator-message",
                label: "Edit this message"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={runJob} pendingMessage={null} />);

    expect(screen.getByText("Edited")).toBeInTheDocument();
    await userEvent.click(screen.getByTitle("Edit this message"));
    const dialog = screen.getByRole("dialog", { name: "Edit message" });
    const editor = within(dialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "The corridor holds steady.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Save" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/narrator-edit", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/chat/narrator-edit");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      message_id: "narrator-1",
      save_id: "save-1",
      body: "The corridor holds steady."
    });
    expect(runJob).toHaveBeenCalledWith({
      id: "job-narrator-edit",
      type: "narrator_edit",
      status: "queued",
      result: null,
      error: null
    });
  });

  it("keeps player edit replay open and usable when submission fails", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      statusText: "Conflict",
      json: async () => ({ detail: "Player edit could not be replayed." })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "player-1",
            role: "player",
            speaker_name: "Keeper",
            body: "Hold the line.",
            actions: [
              {
                action_id: "edit-and-resubmit-message",
                label: "Edit this message"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={runJob} pendingMessage={null} />);

    await userEvent.click(screen.getByTitle("Edit this message"));
    const dialog = screen.getByRole("dialog", { name: "Edit message" });
    const editor = within(dialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "Hold the east line.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Resubmit" }));

    expect(await within(dialog).findByText("Player edit could not be replayed.")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Resubmit" })).toBeEnabled();
    expect(editor).toHaveValue("Hold the east line.");
    expect(runJob).not.toHaveBeenCalled();
  });

  it("keeps player edit without resubmit open and usable when submission fails", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      statusText: "Conflict",
      json: async () => ({ detail: "Player edit could not be saved." })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "player-1",
            role: "player",
            speaker_name: "Keeper",
            body: "Hold the line.",
            actions: [
              {
                action_id: "edit-and-resubmit-message",
                label: "Edit this message"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={runJob} pendingMessage={null} />);

    await userEvent.click(screen.getByTitle("Edit this message"));
    const dialog = screen.getByRole("dialog", { name: "Edit message" });
    const editor = within(dialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "Hold the east line.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Edit without Resubmit" }));

    expect(await within(dialog).findByText("Player edit could not be saved.")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Edit without Resubmit" })).toBeEnabled();
    expect(editor).toHaveValue("Hold the east line.");
    expect(runJob).not.toHaveBeenCalled();
  });

  it("keeps narrator edit open and usable when submission fails", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Server Error",
      json: async () => ({ detail: "Narrator edit could not be saved." })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "The corridor floods with ash.",
            actions: [
              {
                action_id: "edit-narrator-message",
                label: "Edit this message"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={runJob} pendingMessage={null} />);

    await userEvent.click(screen.getByTitle("Edit this message"));
    const dialog = screen.getByRole("dialog", { name: "Edit message" });
    const editor = within(dialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "The corridor holds steady.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Save" }));

    expect(await within(dialog).findByText("Narrator edit could not be saved.")).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Save" })).toBeEnabled();
    expect(editor).toHaveValue("The corridor holds steady.");
    expect(runJob).not.toHaveBeenCalled();
  });

  it("splits captured model requests into sources and raw tabs", async () => {
    const { Chronicle } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "The beacon waits.",
            actions: [
              {
                action_id: "inspect-debug-prompt",
                label: "Inspect prompt",
                detail_text: "Source cards\n\nNarrator prompt\nRequest\n- Provider and model\n  openrouter / qwen\n\nRaw requests\n{\"messages\":[]}"
              }
            ]
          }
        ]
      }
    });

    render(<Chronicle model={model} runJob={vi.fn()} pendingMessage={null} />);

    const opener = screen.getByTitle("Inspect prompt");
    opener.focus();
    expect(opener).toHaveFocus();
    await userEvent.click(opener);

    const dialog = screen.getByRole("dialog", { name: "Inspect prompt" });
    expect(dialog).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Sources" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel", { name: "Sources" })).toHaveTextContent("Provider and model");
    expect(screen.getByText(/Provider and model/)).toBeInTheDocument();
    expect(screen.queryByText("{\"messages\":[]}")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Raw" }));

    expect(screen.getByRole("tab", { name: "Raw" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel", { name: "Raw" })).toHaveTextContent("{\"messages\":[]}");
    expect(screen.getByText("{\"messages\":[]}")).toBeInTheDocument();

    await userEvent.keyboard("{Escape}");

    expect(screen.queryByRole("dialog", { name: "Inspect prompt" })).not.toBeInTheDocument();
    expect(opener).toHaveFocus();
  });

  it("renders maintenance job diagnostics with batch metrics", async () => {
    const { MaintenanceJobsList } = await import("./main");

    render(
      <MaintenanceJobsList
        jobs={[
          {
            job_id: "job-1",
            job_type: "state_pruning",
            status: "failed",
            save_id: null,
            error: "provider timed out",
            started_at: "2026-05-27T12:00:00Z",
            completed_at: "2026-05-27T12:01:00Z",
            summary: "1/3 batches, 8 archived, 2 rejected",
            metrics: {
              completed_batch_count: 1,
              batch_count: 3,
              archived_count: 8,
              rejected_count: 2
            }
          }
        ]}
      />
    );

    expect(screen.getByText("State Pruning")).toBeInTheDocument();
    expect(screen.getByText("1/3 batches, 8 archived, 2 rejected · provider timed out · 2026-05-27T12:01:00Z")).toBeInTheDocument();
    expect(screen.getByText("Completed Batch Count")).toBeInTheDocument();
    expect(screen.getByText("Batch Count")).toBeInTheDocument();
  });

  it("renders memory consolidation diagnostics", async () => {
    const { MaintenanceJobsList } = await import("./main");

    render(
      <MaintenanceJobsList
        jobs={[
          {
            job_id: "job-1",
            job_type: "memory_consolidation",
            status: "failed",
            save_id: "save-1",
            error: "provider timed out",
            started_at: "2026-05-27T12:00:00Z",
            completed_at: "2026-05-27T12:01:00Z",
            summary: "42 active, 3 proposed, 1 rewritten, 4 archived, 2 rejected",
            metrics: {
              active_memory_count: 42,
              proposed_cluster_count: 3,
              rewritten_count: 1,
              archived_count: 4,
              rejected_count: 2
            }
          }
        ]}
      />
    );

    expect(screen.getByText("Memory Consolidation")).toBeInTheDocument();
    expect(screen.getByText("42 active, 3 proposed, 1 rewritten, 4 archived, 2 rejected · provider timed out · 2026-05-27T12:01:00Z")).toBeInTheDocument();
    expect(screen.getByText("Proposed Cluster Count")).toBeInTheDocument();
    expect(screen.getByText("Rewritten Count")).toBeInTheDocument();
  });

  it("expands job diagnostics and opens the admin detail modal", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        job_id: "job-1",
        job_type: "character_reference_image",
        save_id: "save-1",
        status: "failed",
        detail_level: "admin",
        detail_available: true,
        diagnostics: {
          request: {
            origin: {
              kind: "manual_character_reference",
              label: "Manual character reference image"
            },
            task: "image_generation",
            provider: "fake",
            model: "fake-image",
            prompt: "A lantern keeper in the rain."
          },
          provider: { error_category: "content_blocked", http_status: 400 },
          bragi: { status: "failed", error: "Provider rejected the image." },
          timing: { duration_ms: 1200 }
        }
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const { MaintenanceJobsList } = await import("./main");

    render(
      <MaintenanceJobsList
        isAdmin
        jobs={[{
          job_id: "job-1",
          job_type: "character_reference_image",
          status: "failed",
          save_id: "save-1",
          error: "provider failed",
          started_at: "2026-07-10T12:00:00Z",
          completed_at: "2026-07-10T12:00:01Z",
          summary: "Image request failed",
          metrics: {}
        }]}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Expand request details" }));
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/jobs/job-1/diagnostics?save_id=save-1",
      expect.anything()
    );
    expect(await screen.findByText(/Manual character reference image/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Open full details" }));

    const dialog = screen.getByRole("dialog", { name: "Character Reference Image" });
    expect(dialog).toHaveTextContent("A lantern keeper in the rain.");
    expect(dialog).toHaveTextContent("content_blocked");
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: "Character Reference Image" })).not.toBeInTheDocument();
  });

  it("refetches expanded diagnostics when the signal job changes", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      return Promise.resolve({
        ok: true,
        json: async () => url.includes("job-2")
          ? {
              job_id: "job-2",
              job_type: "character_text_image_generation",
              save_id: "save-1",
              status: "failed",
              detail_level: "admin",
              detail_available: true,
              diagnostics: {
                request: { origin: { kind: "character_text_attachment", label: "Character text message image" } },
                provider: { error_category: "provider_error" },
                bragi: { status: "failed" }
              }
            }
          : {
              job_id: "job-1",
              job_type: "image_generation",
              save_id: "save-1",
              status: "failed",
              detail_level: "admin",
              detail_available: true,
              diagnostics: {
                request: { origin: { kind: "manual_scene_image", label: "Manual scene image" } },
                provider: { error_category: "content_blocked" },
                bragi: { status: "failed" }
              }
            }
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { DiagnosticsList } = await import("./main");
    const initialSignals = [{
      kind: "job",
      job_id: "job-1",
      job_type: "image_generation",
      save_id: "save-1",
      error: "failed"
    }];

    const { rerender } = render(<DiagnosticsList diagnostics={initialSignals} isAdmin />);

    await userEvent.click(screen.getByRole("button", { name: "Expand request details" }));
    expect(await screen.findByText(/Manual scene image/)).toBeInTheDocument();
    rerender(<DiagnosticsList diagnostics={[{ ...initialSignals[0], job_id: "job-2" }]} isAdmin />);

    expect(await screen.findByText(/Character text message image/)).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/jobs/job-2/diagnostics?save_id=save-1",
      expect.anything()
    );
  });

  it("renders runtime performance diagnostics", async () => {
    const { RuntimePerformanceList } = await import("./main");

    render(
      <RuntimePerformanceList
        report={{
          job_averages: [
            {
              job_type: "chat_turn",
              step_name: null,
              provider: null,
              model: null,
              task: null,
              success_count: 2,
              sample_count: 3,
              failed_count: 1,
              cancelled_count: 0,
              skipped_count: 0,
              average_duration_ms: 1200,
              p50_duration_ms: 900,
              p95_duration_ms: 1500,
              min_duration_ms: 900,
              max_duration_ms: 1500,
              latest_duration_ms: 999,
              average_queue_wait_ms: 250,
              p95_queue_wait_ms: 500,
              failure_rate: 1 / 3,
              latest_completed_at: "2026-06-01T12:00:00Z"
            }
          ],
          step_averages: [
            {
              job_type: null,
              step_name: "state",
              provider: null,
              model: null,
              task: null,
              success_count: 1,
              sample_count: 2,
              failed_count: 0,
              cancelled_count: 0,
              skipped_count: 1,
              average_duration_ms: 80,
              p50_duration_ms: 80,
              p95_duration_ms: 80,
              min_duration_ms: 80,
              max_duration_ms: 80,
              latest_duration_ms: 0,
              average_queue_wait_ms: null,
              p95_queue_wait_ms: null,
              failure_rate: 0,
              latest_completed_at: "2026-06-01T12:00:01Z"
            }
          ],
          model_averages: [
            {
              job_type: null,
              step_name: null,
              provider: "fake",
              model: "fake-chat",
              task: "chat",
              success_count: 3,
              sample_count: 3,
              failed_count: 0,
              cancelled_count: 0,
              skipped_count: 0,
              average_duration_ms: 70,
              p50_duration_ms: 70,
              p95_duration_ms: 90,
              min_duration_ms: 50,
              max_duration_ms: 90,
              latest_duration_ms: 65,
              average_queue_wait_ms: null,
              p95_queue_wait_ms: null,
              failure_rate: 0,
              latest_completed_at: "2026-06-01T12:00:02Z"
            }
          ],
          slowest_recent: [
            {
              job_id: "job-slow",
              save_id: "save-1",
              job_type: "chat_turn",
              status: "failed",
              started_at: "2026-06-01T12:00:00Z",
              completed_at: "2026-06-01T12:00:07Z",
              duration_ms: 7000,
              queue_wait_ms: 500,
              slowest_step_name: "provider.chat",
              slowest_step_duration_ms: 6500,
              provider: "fake",
              model: "fake-chat",
              task: "chat"
            }
          ]
        }}
      />
    );

    expect(screen.getByText("Jobs")).toBeInTheDocument();
    expect(screen.getByText("Chat Turn")).toBeInTheDocument();
    expect(screen.getByText(/avg 1.2s/)).toBeInTheDocument();
    expect(screen.getByText(/p50 900 ms/)).toBeInTheDocument();
    expect(screen.getByText(/p95 1.5s/)).toBeInTheDocument();
    expect(screen.getByText(/fail 33%/)).toBeInTheDocument();
    expect(screen.getByText(/queue p95 500 ms/)).toBeInTheDocument();
    expect(screen.getByText("State")).toBeInTheDocument();
    expect(screen.getByText(/skipped 1/)).toBeInTheDocument();
    expect(screen.getByText("fake / fake-chat")).toBeInTheDocument();
    expect(screen.getByText(/Chat · avg 70 ms/)).toBeInTheDocument();
    expect(screen.getByText("Slowest Recent")).toBeInTheDocument();
    expect(screen.getByText(/Provider Chat/)).toBeInTheDocument();
    expect(screen.getByText(/job-slow/)).toBeInTheDocument();
  });

  it("renders runtime performance empty state", async () => {
    const { RuntimePerformanceList } = await import("./main");

    render(
      <RuntimePerformanceList
        report={{ job_averages: [], step_averages: [], model_averages: [] }}
      />
    );

    expect(screen.getByText("No runtime performance data")).toBeInTheDocument();
  });

  it("renders terminal job history with step drilldown metadata", async () => {
    const { TerminalJobsList } = await import("./main");

    render(
      <TerminalJobsList
        jobs={[
          {
            id: "job-1",
            type: "chat_turn",
            save_id: "save-1",
            status: "failed",
            created_at: "2026-06-01T12:00:00Z",
            started_at: "2026-06-01T12:00:03Z",
            completed_at: "2026-06-01T12:00:10Z",
            duration_ms: 7000,
            queue_wait_ms: 3000,
            step_count: 1,
            error: "Background job failed. Check diagnostics for details."
          }
        ]}
        selectedJobId="job-1"
        steps={{
          job_id: "job-1",
          steps: [
            {
              id: "step-1",
              name: "provider.chat",
              status: "failed",
              provider: "fake",
              model: "fake-chat",
              task: "chat",
              started_at: "2026-06-01T12:00:03Z",
              completed_at: "2026-06-01T12:00:09Z",
              duration_ms: 6000,
              metadata: { token_total: 123 }
            }
          ]
        }}
        onSelectJob={vi.fn()}
      />
    );

    expect(screen.getByText("Chat Turn")).toBeInTheDocument();
    expect(screen.getByText(/failed · 7.0s · queue 3.0s/)).toBeInTheDocument();
    expect(screen.getByText("Provider Chat")).toBeInTheDocument();
    expect(screen.getByText("Token Total")).toBeInTheDocument();
    expect(screen.getByText("123")).toBeInTheDocument();
    expect(screen.queryByText(/private/)).not.toBeInTheDocument();
  });

  it("recognizes backend message action ids", async () => {
    const { hasAction } = await import("./main");

    expect(
      hasAction(
        {
          message_id: "m1",
          role: "narrator",
          speaker_name: null,
          body: "Scene",
          actions: [{ action_id: "regenerate-message", label: "Regenerate" }]
        },
        "regenerate-message",
        "regenerate"
      )
    ).toBe(true);
  });

  it("disables a chronicle regenerate action while it starts a job", async () => {
    const { Chronicle } = await import("./main");
    const response = deferred<{ ok: boolean; json: () => Promise<Job> }>();
    const fetchMock = vi.fn().mockImplementation((path: string) => (
      path === "/api/chat/regenerate"
        ? response.promise
        : Promise.resolve({ ok: true, json: async () => ({}) })
    ));
    const job = {
      id: "job-regenerate",
      type: "chat_regenerate",
      save_id: "save-1",
      status: "queued",
      result: null,
      error: null
    } satisfies Job;
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <Chronicle
        model={runtimeModel({
          chronicle: {
            messages: [
              {
                message_id: "narrator-1",
                role: "narrator",
                speaker_name: null,
                body: "The beacon flickers.",
                actions: [{ action_id: "regenerate-message", label: "Regenerate" }]
              }
            ]
          }
        })}
        runJob={runJob}
        pendingMessage={null}
      />
    );

    const regenerate = screen.getByRole("button", { name: "Regenerate" });
    await userEvent.click(regenerate);

    await waitFor(() => expect(regenerate).toBeDisabled());
    expect(regenerate.querySelector(".spin")).not.toBeNull();

    response.resolve({ ok: true, json: async () => job });

    await waitFor(() => expect(runJob).toHaveBeenCalledWith(job));
    expect(regenerate).toBeEnabled();
  });

  it("shows a chronicle regenerate action error without starting a job", async () => {
    const { Chronicle } = await import("./main");
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/chat/regenerate"
        ? {
            ok: false,
            status: 409,
            statusText: "Conflict",
            json: async () => ({ detail: "Regeneration is already running." })
          }
        : { ok: true, json: async () => ({}) }
    ));
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <Chronicle
        model={runtimeModel({
          chronicle: {
            messages: [
              {
                message_id: "narrator-1",
                role: "narrator",
                speaker_name: null,
                body: "The beacon flickers.",
                actions: [{ action_id: "regenerate-message", label: "Regenerate" }]
              }
            ]
          }
        })}
        runJob={runJob}
        pendingMessage={null}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Regenerate" }));

    expect(await screen.findByText("Regeneration is already running.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Regenerate" })).toBeEnabled();
    expect(runJob).not.toHaveBeenCalled();
  });

  it("renders generic object data without crashing", async () => {
    const { DataViewer } = await import("./main");

    render(<DataViewer value={{ scenario_title: "Lantern Keep", message_count: 3 }} emptyLabel="Empty" />);

    expect(screen.getByText("Scenario Title")).toBeInTheDocument();
    expect(screen.getByText("Lantern Keep")).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
  });

  it("keeps preview modals open and shows confirm failures", async () => {
    const { PreviewModal } = await import("./main");

    render(
      <PreviewModal
        title="Import save bundle?"
        preview={{
          save_id: "save-1",
          title: "Lantern Keep",
          scenario_title: "Storm Sea",
          message_count: 2,
          media_count: 1,
          bundle_version: 1,
          created_at: null,
          updated_at: null,
          exported_at: null
        }}
        detail="This will restore the bundled save into Bragi Web."
        confirmLabel="Import"
        onCancel={vi.fn()}
        onConfirm={async () => {
          throw new Error("Bundle import failed: scenario already exists");
        }}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Import" }));

    expect(await screen.findByText("Bundle import failed: scenario already exists")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Import save bundle?" })).toBeInTheDocument();
  });

  it("traps keyboard focus inside modal dialogs and restores the opener", async () => {
    const { PreviewModal } = await import("./main");
    const user = userEvent.setup();
    const onCancel = vi.fn();

    function Harness() {
      const [open, setOpen] = React.useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>Open import</button>
          <button type="button">Outside action</button>
          {open ? (
            <PreviewModal
              title="Import save bundle?"
              preview={{
                save_id: "save-1",
                title: "Lantern Keep",
                scenario_title: "Storm Sea",
                message_count: 2,
                media_count: 1,
                bundle_version: 1,
                created_at: null,
                updated_at: null,
                exported_at: null
              }}
              detail="This will restore the bundled save into Bragi Web."
              confirmLabel="Import"
              onCancel={() => {
                onCancel();
                setOpen(false);
              }}
              onConfirm={async () => undefined}
              extra={<button type="button">Inspect bundle</button>}
            />
          ) : null}
        </>
      );
    }

    render(<Harness />);

    const opener = screen.getByRole("button", { name: "Open import" });
    await user.click(opener);
    const dialog = screen.getByRole("dialog", { name: "Import save bundle?" });
    const close = within(dialog).getByLabelText("Close");
    const importButton = within(dialog).getByRole("button", { name: "Import" });

    await waitFor(() => expect(close).toHaveFocus());
    await user.tab({ shift: true });
    expect(importButton).toHaveFocus();
    await user.tab();
    expect(close).toHaveFocus();
    await user.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Import save bundle?" })).not.toBeInTheDocument());
    expect(opener).toHaveFocus();
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("supports roving keyboard navigation for segmented tabs", async () => {
    const { SegmentedTabs } = await import("./main");
    const user = userEvent.setup();

    function Harness() {
      const [value, setValue] = React.useState<"alpha" | "beta" | "gamma" | "delta">("alpha");
      return (
        <SegmentedTabs
          label="Workbench modes"
          value={value}
          onChange={setValue}
          options={[
            { value: "alpha", label: "Alpha" },
            { value: "beta", label: "Beta", disabled: true },
            { value: "gamma", label: "Gamma" },
            { value: "delta", label: "Delta" }
          ]}
        />
      );
    }

    render(<Harness />);

    const alpha = screen.getByRole("tab", { name: "Alpha" });
    const beta = screen.getByRole("tab", { name: "Beta" });
    const gamma = screen.getByRole("tab", { name: "Gamma" });
    const delta = screen.getByRole("tab", { name: "Delta" });

    expect(alpha).toHaveAttribute("aria-selected", "true");
    expect(alpha).toHaveAttribute("tabIndex", "0");
    expect(gamma).toHaveAttribute("tabIndex", "-1");
    expect(beta).toBeDisabled();

    alpha.focus();
    await user.keyboard("{ArrowRight}");
    expect(gamma).toHaveFocus();
    expect(gamma).toHaveAttribute("aria-selected", "true");
    expect(alpha).toHaveAttribute("tabIndex", "-1");
    expect(gamma).toHaveAttribute("tabIndex", "0");

    await user.keyboard("{End}");
    expect(delta).toHaveFocus();
    expect(delta).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{ArrowRight}");
    expect(alpha).toHaveFocus();
    expect(alpha).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{ArrowLeft}");
    expect(delta).toHaveFocus();
    expect(delta).toHaveAttribute("aria-selected", "true");

    await user.keyboard("{Home}");
    expect(alpha).toHaveFocus();
    expect(alpha).toHaveAttribute("aria-selected", "true");
  });

  it("filters world data across tabs and saves an inline world-state correction", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "Keeper",
        player_role: "Keeper",
        content_sections: []
      },
      scene: { snapshot_id: "scene-1", situation: "A beacon waits", weather: "clear" },
      world_state: [
        {
          row_id: "state-1",
          key: "storm",
          original_key: "storm",
          category: "weather",
          confidence: 0.9,
          value_json: "{\"intensity\":\"high\"}",
          source_message_id: null,
          archived: false
        },
        {
          row_id: "state-2",
          key: "lantern",
          original_key: "lantern",
          category: "object",
          confidence: 0.8,
          value_json: "{\"lit\":false}",
          source_message_id: null,
          archived: false
        }
      ],
      memories: [{ memory_id: "memory-1", body: "Storm clouds gathered.", tags_text: "weather", importance: 0.7 }],
      suggestions: [],
      audit: [{ audit_id: "audit-1", operation: "manual_world_data_edit", field_path: "storm" }]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? { model: { ...worldPayload, world_state: [{ ...worldPayload.world_state[0], value_json: "{\"intensity\":\"low\"}" }] }, state_archive_count: 0, memory_archive_count: 0, summary_archive_count: 0 }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^Knowledge/ }));
    const worldStateTab = screen.getByRole("tab", { name: /world state/i });
    await userEvent.click(worldStateTab);
    expect(worldStateTab).toHaveAttribute("aria-selected", "true");
    await userEvent.type(screen.getByLabelText("Search world data"), "storm");

    expect(screen.getAllByText("storm").length).toBeGreaterThan(0);
    expect(screen.queryByText("lantern")).not.toBeInTheDocument();
    expect(screen.getByText("3 matches in 3 sections")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Edit storm" }));
    fireEvent.change(screen.getByLabelText("Fact Value JSON"), { target: { value: "{\"intensity\":\"low\"}" } });
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scenario: worldPayload.scenario,
        world_state: [
          {
            row_id: "state-1",
            key: "storm",
            value_json: "{\"intensity\":\"low\"}"
          }
        ]
      }
    });
  });

  it("edits the active-save scenario from the world panel with structured fields", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "Mara",
        player_role: "Keeper",
        content_sections: [
          ["tone_genre", "Beacon mystery"]
        ]
      },
      world_state: [],
      memories: []
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? { model: { ...worldPayload, scenario: { ...worldPayload.scenario, premise: "A repaired tower." } }, state_archive_count: 0, memory_archive_count: 0, summary_archive_count: 0 }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    expect(await screen.findByLabelText("Title")).toHaveValue("Lantern Keep");
    expect(screen.queryByRole("button", { name: /raw json/i })).not.toBeInTheDocument();
    const saveScenario = screen.getByRole("button", { name: "Save scenario" });
    expect(saveScenario).toBeDisabled();
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("Premise / Setup"), { target: { value: "A repaired tower." } });
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    expect(saveScenario).toBeEnabled();
    expect(screen.queryByLabelText("Starting Scene")).not.toBeInTheDocument();
    fireEvent.change(screen.getByLabelText("New section key"), { target: { value: "current_scene" } });
    await userEvent.click(screen.getByRole("button", { name: "Add section" }));
    fireEvent.change(screen.getByLabelText("Current Scene"), { target: { value: "Mara stands under the repaired lens." } });
    await userEvent.click(saveScenario);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/apply", expect.anything()));
    await waitFor(() => expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument());
    expect(saveScenario).toBeDisabled();
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scenario: {
          title: "Lantern Keep",
          premise: "A repaired tower.",
          player_character_name: "Mara",
          player_role: "Keeper",
          content_sections: expect.arrayContaining([
            ["tone_genre", "Beacon mystery"],
            ["current_scene", "Mara stands under the repaired lens."]
          ])
        }
      }
    });
  });

  it("confirms before discarding unsaved scenario definition edits", async () => {
    const scenario = scenarioFixture({
      title: "Fog Gate",
      premise: "A gate in the fog.",
      player_role: "Keeper"
    });
    const definitionPayload: any = {
      active_save_id: null,
      scenario: {
        ...scenario,
        player_character_name: "Mara",
        content_sections: [["tone_genre", "Fog mystery"]]
      }
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/scenario-1/definition" ? definitionPayload : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const onChanged = vi.fn();
    const { LeftRail } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <LeftRail
          model={runtimeModel()}
          scenarios={[scenario]}
          onChanged={onChanged}
          onSelectSave={vi.fn()}
          pendingSaveId={null}
          saveSelectionError=""
          onNew={vi.fn()}
          activePanel="media"
          setPanel={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (1)" }));
    await userEvent.click(screen.getByRole("button", { name: "Edit Fog Gate" }));
    const dialog = await screen.findByRole("dialog", { name: /edit scenario: fog gate/i });
    const saveButton = within(dialog).getByRole("button", { name: "Save scenario" });
    expect(saveButton).toBeDisabled();

    fireEvent.change(within(dialog).getByLabelText("Title"), { target: { value: "Fog Gate Revised" } });
    expect(within(dialog).getByText("Unsaved changes")).toBeInTheDocument();
    expect(saveButton).toBeEnabled();

    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));
    let confirm = screen.getByRole("dialog", { name: "Discard changes?" });
    await userEvent.click(within(confirm).getByRole("button", { name: "Cancel" }));
    expect(screen.getByRole("dialog", { name: /edit scenario: fog gate/i })).toBeInTheDocument();
    expect(within(dialog).getByLabelText("Title")).toHaveValue("Fog Gate Revised");

    await userEvent.click(within(dialog).getByRole("button", { name: "Close" }));
    confirm = screen.getByRole("dialog", { name: "Discard changes?" });
    await userEvent.click(within(confirm).getByRole("button", { name: "Discard" }));

    await waitFor(() => expect(screen.queryByRole("dialog", { name: /edit scenario: fog gate/i })).not.toBeInTheDocument());
    expect(fetchMock.mock.calls.some(([path, init]) => path === "/api/scenarios/scenario-1/definition" && (init as RequestInit | undefined)?.body)).toBe(false);
    expect(onChanged).not.toHaveBeenCalled();
  });

  it("blocks stale world-data edits after the active save changes", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "Keeper",
        player_role: "Keeper",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-1",
          key: "storm",
          original_key: "storm",
          category: "weather",
          confidence: 0.9,
          value_json: "{\"intensity\":\"high\"}",
          source_message_id: null,
          archived: false
        }
      ]
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel({ active_save_id: "save-2", active_save_title: "Signal Tower" })} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^Knowledge/ }));
    await userEvent.click(screen.getByRole("tab", { name: /world state/i }));

    expect(screen.getAllByText("storm").length).toBeGreaterThan(0);
    expect(screen.queryByRole("button", { name: "Edit storm" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /raw json/i })).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/world-data/apply")).toBe(false);
  });

  it("shows context inputs and consolidated memory evidence in world data", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "Keeper",
        player_role: "Keeper",
        content_sections: []
      },
      memories: [
        {
          memory_id: "memory-1",
          body: "Storm clouds gathered over the beacon.",
          tags_text: "weather",
          importance: 0.7,
          source_message_id: "message-1",
          source_message_ids: ["message-1", "message-2"],
          consolidated: true,
          archived: false
        }
      ],
      context_inputs: [
        {
          context_source_id: "context-1",
          source_type: "memory",
          source_id: "memory-1",
          title: "Storm memory",
          body: "Storm clouds gathered over the beacon.",
          fact_type: "weather",
          importance: 0.82,
          source_message_count: 2,
          token_estimate: 48
        }
      ]
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^Scene/ }));
    await userEvent.click(screen.getByRole("tab", { name: /context inputs/i }));

    expect(screen.getAllByText("Storm memory").length).toBeGreaterThan(0);
    expect(screen.getByText(/memory:memory-1/)).toBeInTheDocument();
    expect(screen.getByText("2 sources")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /raw json/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /^Knowledge/ }));
    await userEvent.click(screen.getByRole("tab", { name: /memories/i }));

    expect(screen.getAllByText("Storm clouds gathered over the beacon.").length).toBeGreaterThan(0);
    expect(screen.getByText("consolidated")).toBeInTheDocument();
    expect(screen.getByText("2 sources")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /edit storm clouds gathered/i }));

    expect(screen.queryByText("Source Message Ids")).not.toBeInTheDocument();
    expect(screen.queryByText("Consolidated")).not.toBeInTheDocument();
  });

  it("hides deprecated loss-condition world data tabs", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      loss_conditions: [
        {
          condition_id: "condition-1",
          name: "Beacon collapse",
          description: "Legacy condition.",
          status: "active"
        }
      ],
      active_loss_outcome: {
        outcome_id: "outcome-1",
        condition_name: "Mission complete",
        explanation: "Mara seals the gate and the scenario is over.",
        outcome_type: "player_dead",
        confidence: 0.96
      }
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await screen.findByRole("tab", { name: /scenario/i });
    expect(screen.queryByRole("tab", { name: /loss conditions/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /active terminal outcome/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /active loss outcome/i })).not.toBeInTheDocument();
    expect(screen.queryByText("Beacon collapse")).not.toBeInTheDocument();
    expect(screen.queryByText("Mission complete")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /raw json/i })).not.toBeInTheDocument();
  });

  it("saves response guidance from the world panel", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "",
        player_role: "Keeper",
        content_sections: []
      }
    };
    const runtimeModel = {
      saves: [],
      active_save_id: "save-1",
      active_save_title: "Lantern Keep",
      active_scenario_type: null,
      custom_instructions: "Keep narration brisk.",
      scenario_title: "Lantern Keep",
      scene_title: "Beacon",
      chronicle: { messages: [] },
      media: null,
      action_choices: null,
      model_indicator: "fake / chat",
      failed_save: false,
      composer_enabled: true,
      failure_text: null,
      status: null,
      error: null
    } satisfies RuntimeModel;
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/runtime/custom-instructions") {
          const body = JSON.parse(String(init?.body));
          return {
            ...runtimeModel,
            custom_instructions: body.custom_instructions,
            status: "Response guidance saved"
          };
        }
        return worldPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /save response guidance/i }));
    const guidanceDialog = screen.getByRole("dialog", { name: "Save response guidance" });
    const editor = within(guidanceDialog).getByRole("textbox", { name: "Save response guidance" });
    expect(editor).toHaveValue("Keep narration brisk.");
    await userEvent.clear(editor);
    await userEvent.type(editor, "Keep replies concise.");
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/runtime/custom-instructions", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/runtime/custom-instructions");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      save_id: "save-1",
      custom_instructions: "Keep replies concise."
    });
  });

  it("shows pending and error state for world suggestion review", async () => {
    const review = deferred<{ ok: boolean; status?: number; statusText?: string; json: () => Promise<unknown> }>();
    const fetchMock = vi.fn().mockImplementation((rawPath: string) => {
      const path = String(rawPath);
      if (path === "/api/world-data/suggestion-review") return review.promise;
      if (path === "/api/log/client") return Promise.resolve({ ok: true, json: async () => ({ ok: true }) });
      return Promise.resolve({ ok: true, json: async () => worldDataPayload() });
    });
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={runJob} />
      </QueryClientProvider>
    );

    const reviewButton = await screen.findByRole("button", { name: /Review suggestions/i });
    await userEvent.click(reviewButton);

    await waitFor(() => expect(reviewButton).toBeDisabled());
    expect(reviewButton.querySelector(".spin")).not.toBeNull();

    review.resolve({
      ok: false,
      status: 500,
      statusText: "Server Error",
      json: async () => ({ detail: "Could not start suggestion review." })
    });

    expect(await screen.findByText("Could not start suggestion review.")).toBeInTheDocument();
    expect(reviewButton).toBeEnabled();
    expect(runJob).not.toHaveBeenCalled();
  });

  it("queues guided cleanup instructions from the world panel", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "",
        player_role: "Keeper",
        content_sections: []
      },
      suggestions: []
    };
    const runtimeModel = {
      saves: [],
      active_save_id: "save-1",
      active_save_title: "Lantern Keep",
      active_scenario_type: null,
      custom_instructions: "",
      scenario_title: "Lantern Keep",
      scene_title: "Beacon",
      chronicle: { messages: [] },
      media: null,
      action_choices: null,
      model_indicator: "fake / chat",
      failed_save: false,
      composer_enabled: true,
      failure_text: null,
      status: null,
      error: null
    } satisfies RuntimeModel;
    const job = {
      id: "job-1",
      type: "guided_context_cleanup",
      status: "queued",
      result: null,
      error: null
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/guided-cleanup" ? job : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel} runJob={runJob} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /guided cleanup/i }));
    await userEvent.type(screen.getByLabelText("Cleanup instructions"), "Archive the resolved storm thread.");
    await userEvent.click(screen.getByRole("button", { name: /run cleanup/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/guided-cleanup", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/guided-cleanup");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      save_id: "save-1",
      instruction: "Archive the resolved storm thread."
    });
    expect(runJob).toHaveBeenCalledWith(job);
  });

  it("queues regular cleanup for the active save from the world panel", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "",
        player_role: "Keeper",
        content_sections: []
      },
      suggestions: []
    };
    const runtimeModel = {
      saves: [],
      active_save_id: "save-1",
      active_save_title: "Lantern Keep",
      active_scenario_type: null,
      custom_instructions: "",
      scenario_title: "Lantern Keep",
      scene_title: "Beacon",
      chronicle: { messages: [] },
      media: null,
      action_choices: null,
      model_indicator: "fake / chat",
      failed_save: false,
      composer_enabled: true,
      failure_text: null,
      status: null,
      error: null
    } satisfies RuntimeModel;
    const job = {
      id: "job-1",
      type: "context_cleanup",
      status: "queued",
      result: null,
      error: null
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/context-cleanup" ? job : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel} runJob={runJob} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /cleanup context/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/context-cleanup", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/context-cleanup");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      save_id: "save-1"
    });
    expect(runJob).toHaveBeenCalledWith(job);
  });

  it("disables regular cleanup with no active save", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({ active_save_id: null }) });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel({ active_save_id: null })} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("button", { name: /cleanup context/i })).toBeDisabled();
  });

  it("blocks invalid world-state JSON before saving", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "",
        player_role: "Keeper",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-1",
          key: "storm",
          original_key: "storm",
          category: "weather",
          confidence: 0.9,
          value_json: "{\"intensity\":\"high\"}",
          source_message_id: null,
          archived: false
        }
      ]
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^Knowledge/ }));
    await userEvent.click(screen.getByRole("tab", { name: /world state/i }));
    await userEvent.click(screen.getByRole("button", { name: "Edit storm" }));
    fireEvent.change(screen.getByLabelText("Fact Value JSON"), { target: { value: "{nope" } });
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    expect(await screen.findByText("Fact value must be valid JSON")).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/world-data/apply")).toBe(false);
  });

  it("edits generic world data rows across memories, summaries, locations, threads, and links", async () => {
    const worldPayload = worldDataPayload();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? { model: worldPayload, state_archive_count: 0, memory_archive_count: 0, summary_archive_count: 0 }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const edits = [
      {
        group: "Knowledge",
        tab: /memories/i,
        editButton: "Edit Storm memory",
        label: "Body",
        value: "Mara remembers the repaired beacon.",
        expectedTab: "memories",
        expected: { memory_id: "memory-1", body: "Mara remembers the repaired beacon." }
      },
      {
        group: "Knowledge",
        tab: /summaries/i,
        editButton: "Edit Beacon summary",
        label: "Summary Text",
        value: "The beacon is repaired.",
        expectedTab: "summaries",
        expected: { summary_id: "summary-1", summary_text: "The beacon is repaired." }
      },
      {
        group: "People & Places",
        tab: /locations/i,
        editButton: "Edit Beacon Tower",
        label: "Description",
        value: "A repaired tower above the fog.",
        expectedTab: "locations",
        expected: { location_id: "location-1", description: "A repaired tower above the fog." }
      },
      {
        group: "People & Places",
        tab: /threads/i,
        editButton: "Edit Repair the beacon",
        label: "Status",
        value: "resolved",
        expectedTab: "threads",
        expected: { thread_id: "thread-1", status: "resolved" }
      },
      {
        group: "Knowledge",
        tab: /links/i,
        editButton: "Edit character -> memory",
        label: "Relation",
        value: "trusts",
        expectedTab: "links",
        expected: { link_id: "link-1", relation: "trusts" }
      }
    ];

    for (const edit of edits) {
      const escapedGroup = edit.group.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      await userEvent.click(await screen.findByRole("tab", { name: new RegExp(`^${escapedGroup}( |\\d|$)`) }));
      await userEvent.click(screen.getByRole("tab", { name: edit.tab }));
      await userEvent.click(screen.getByRole("button", { name: edit.editButton }));
      const field = screen.getByLabelText(edit.label);
      await userEvent.clear(field);
      await userEvent.type(field, edit.value);
      const applyCallsBefore = fetchMock.mock.calls.filter(([path]) => path === "/api/world-data/apply").length;
      await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

      await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/world-data/apply")).toHaveLength(applyCallsBefore + 1));
      const applyCalls = fetchMock.mock.calls.filter(([path]) => path === "/api/world-data/apply");
      const applyCall = applyCalls[applyCalls.length - 1];
      expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
        active_save_id: "save-1",
        edits: {
          scenario: worldPayload.scenario,
          [edit.expectedTab]: [edit.expected]
        }
      });
    }
  });

  it("blocks nested generic world-data JSON errors before saving", async () => {
    const worldPayload = worldDataPayload();
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^Knowledge/ }));
    await userEvent.click(screen.getByRole("tab", { name: /memories/i }));
    await userEvent.click(screen.getByRole("button", { name: "Edit Storm memory" }));
    fireEvent.change(screen.getByLabelText("Metadata"), { target: { value: "{nope" } });
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    expect(await screen.findByText("Nested field JSON must be valid")).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/world-data/apply")).toBe(false);
  });

  it("saves raw world-data JSON and keeps invalid JSON local", async () => {
    const worldPayload = worldDataPayload();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? { model: worldPayload, state_archive_count: 0, memory_archive_count: 0, summary_archive_count: 0 }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^People/ }));
    await userEvent.click(screen.getByRole("tab", { name: /locations/i }));
    await userEvent.click(screen.getByRole("button", { name: /raw json/i }));
    const editor = screen.getByLabelText("Raw JSON");
    fireEvent.change(editor, { target: { value: "{nope" } });
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    expect(await screen.findByRole("alert")).toHaveTextContent(/json/i);
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/world-data/apply")).toBe(false);

    fireEvent.change(editor, {
      target: {
        value: JSON.stringify([
          {
            location_id: "location-1",
            name: "Signal Tower",
            description: "A repaired tower.",
            status: "safe"
          }
        ])
      }
    });
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scenario: worldPayload.scenario,
        locations: [
          {
            location_id: "location-1",
            name: "Signal Tower",
            status: "safe"
          }
        ]
      }
    });
  });

  it("keeps audit read-only and exposes grouped suggestions in their own tab", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "",
        player_role: "Keeper",
        content_sections: []
      },
      suggestion_groups: [
        {
          group_id: "group-1",
          suggestion_ids: ["suggestion-1", "suggestion-2"],
          update_type: "update",
          entity_type: "world_state",
          entity_id: "state-1",
          field_path: "storm",
          proposed_value_json: "{\"intensity\":\"low\"}",
          status: "pending",
          reason: "The storm eased.",
          confidence: 0.7,
          source_message_ids_text: "message-1",
          suggestion_count: 2,
          action: ""
        }
      ],
      suggestions: [
        {
          suggestion_id: "suggestion-1",
          update_type: "update",
          entity_type: "world_state",
          entity_id: "state-1",
          field_path: "storm",
          proposed_value_json: "{\"intensity\":\"low\"}",
          status: "pending",
          reason: "The storm eased.",
          confidence: 0.7,
          source_message_ids_text: "",
          action: ""
        },
        {
          suggestion_id: "suggestion-2",
          update_type: "update",
          entity_type: "world_state",
          entity_id: "state-1",
          field_path: "stale storm",
          proposed_value_json: "{\"intensity\":\"high\"}",
          status: "superseded",
          reason: "A newer update landed.",
          confidence: 0.7,
          source_message_ids_text: "",
          action: ""
        }
      ],
      audit: [{ audit_id: "audit-1", operation: "manual_world_data_edit", field_path: "storm" }]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? { model: worldPayload, state_archive_count: 0, memory_archive_count: 0, summary_archive_count: 0 }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("tab", { name: /audit/i }));
    expect(screen.queryByRole("button", { name: /raw json/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /edit storm/i })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: /^Review/ }));
    await userEvent.click(screen.getByRole("tab", { name: /suggestions/i }));

    expect(await screen.findByText("storm")).toBeInTheDocument();
    expect(screen.getByText("2 grouped")).toBeInTheDocument();
    expect(screen.getByText("The storm eased.")).toBeInTheDocument();
    expect(screen.queryByText("hidden individual")).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/world-data/apply")).toBe(false);
  });

  it("applies grouped world-data suggestions from the suggestions tab", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "",
        player_role: "Keeper",
        content_sections: []
      },
      suggestion_groups: [
        {
          group_id: "group-1",
          suggestion_ids: ["suggestion-1", "suggestion-2"],
          update_type: "update",
          entity_type: "world_state",
          entity_id: "state-1",
          field_path: "storm",
          proposed_value_json: "{\"intensity\":\"low\"}",
          status: "pending",
          reason: "The storm eased.",
          confidence: 0.7,
          source_message_ids_text: "message-1",
          suggestion_count: 2,
          action: ""
        }
      ],
      suggestions: [
        {
          suggestion_id: "suggestion-1",
          update_type: "update",
          entity_type: "world_state",
          entity_id: "state-1",
          field_path: "hidden individual",
          proposed_value_json: "{\"intensity\":\"low\"}",
          status: "pending",
          reason: "The storm eased.",
          confidence: 0.7,
          source_message_ids_text: "",
          action: ""
        }
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? { model: worldPayload, state_archive_count: 0, memory_archive_count: 0, summary_archive_count: 0 }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^Review/ }));
    await userEvent.click(screen.getByRole("tab", { name: /suggestions/i }));
    await userEvent.click(screen.getByRole("button", { name: /apply suggestion storm/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scenario: worldPayload.scenario,
        suggestion_groups: [
          {
            group_id: "group-1",
            action: "apply"
          }
        ]
      }
    });
    expect(screen.queryByText("hidden individual")).not.toBeInTheDocument();
  });

  it("queues manual suggestion review and retention jobs", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "",
        player_role: "Keeper",
        content_sections: []
      },
      suggestion_groups: []
    };
    const reviewJob = { id: "job-review", type: "world_suggestion_review", save_id: "save-1", status: "queued", result: null, error: null };
    const retentionJob = { id: "job-retention", type: "world_context_retention", save_id: "save-1", status: "queued", result: null, error: null };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/world-data/suggestion-review") return reviewJob;
        if (path === "/api/world-data/context-retention") return retentionJob;
        return worldPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={runJob} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /review suggestions/i }));
    await userEvent.click(screen.getByRole("button", { name: /run retention/i }));

    await waitFor(() => expect(runJob).toHaveBeenCalledWith(reviewJob));
    expect(runJob).toHaveBeenCalledWith(retentionJob);
    expect(JSON.parse(String(fetchMock.mock.calls.find(([path]) => path === "/api/world-data/suggestion-review")?.[1].body))).toEqual({ save_id: "save-1" });
    expect(JSON.parse(String(fetchMock.mock.calls.find(([path]) => path === "/api/world-data/context-retention")?.[1].body))).toEqual({ save_id: "save-1" });
  });

  it("can explicitly unlock world-data rows with locked field controls", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "",
        player_role: "Keeper",
        content_sections: []
      },
      scene: {
        snapshot_id: "scene-1",
        situation: "Fog over the beacon",
        world_time_day_index: 5,
        locked_fields: [
          "in_world_time",
          "time_of_day",
          "day_of_week",
          "world_day_index",
          "world_time_day_label",
          "world_time_day_index",
          "world_time_phase",
          "world_time_clock_minutes",
          "world_time_period_label"
        ]
      },
      locations: [
        {
          location_id: "location-1",
          name: "Signal Tower",
          aliases_text: "",
          description: "A tower.",
          visual_description: "",
          parent_location_id: null,
          connections_text: "",
          status: "sealed",
          hazards_text: "",
          source_message_id: null,
          locked_fields: ["description", "status"]
        }
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? { model: worldPayload, state_archive_count: 0, memory_archive_count: 0, summary_archive_count: 0 }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^People/ }));
    await userEvent.click(screen.getByRole("tab", { name: /locations/i }));
    await userEvent.click(await screen.findByRole("button", { name: /edit signal tower/i }));
    const statusLock = screen.getByLabelText("Lock Status");
    expect(statusLock).toBeChecked();

    await userEvent.click(statusLock);
    await waitFor(() => expect(statusLock).not.toBeChecked());
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scenario: worldPayload.scenario,
        locations: [
          {
            location_id: "location-1",
            locked_fields: ["description"]
          }
        ]
      }
    });

    await userEvent.click(screen.getByRole("tab", { name: /^Scene/ }));
    await userEvent.click(
      await screen.findByRole("button", { name: /edit fog over the beacon/i })
    );
    const sceneTimeLockLabels = [
      "Lock In World Time",
      "Lock Time Of Day",
      "Lock Day Of Week",
      "Lock World Day Index",
      "Lock World Time Day Label",
      "Lock World Time Day Index",
      "Lock World Time Phase",
      "Lock World Time Clock Minutes",
      "Lock World Time Period Label"
    ];
    for (const label of sceneTimeLockLabels) {
      expect(screen.getByLabelText(label)).toBeChecked();
    }
    for (const label of sceneTimeLockLabels) {
      const lock = screen.getByLabelText(label);
      await userEvent.click(lock);
      await waitFor(() => expect(lock).not.toBeChecked());
    }
    await userEvent.click(screen.getByRole("button", { name: /^save$/i }));

    await waitFor(() => {
      expect(
        fetchMock.mock.calls.filter(([path]) => path === "/api/world-data/apply")
      ).toHaveLength(2);
    });
    const applyCalls = fetchMock.mock.calls.filter(
      ([path]) => path === "/api/world-data/apply"
    );
    const sceneApplyCall = applyCalls[applyCalls.length - 1];
    expect(JSON.parse(String(sceneApplyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scene: {
          snapshot_id: "scene-1",
          locked_fields: []
        }
      }
    });
  });

  it("groups world data sections into review, scene, knowledge, people, and advanced workflows", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "Mara",
        player_role: "Keeper",
        content_sections: []
      },
      scene: { snapshot_id: "scene-1", situation: "Fog over the beacon" },
      world_state: [{ row_id: "state-1", key: "storm", original_key: "storm", category: "weather", confidence: 0.8, value_json: "{\"intensity\":\"low\"}", archived: false }],
      memories: [{ memory_id: "memory-1", body: "Storm clouds gathered." }],
      suggestion_groups: [{
        group_id: "group-1",
        suggestion_ids: ["s-1"],
        update_type: "update",
        entity_type: "world_state",
        entity_id: "state-1",
        field_path: "storm",
        proposed_value_json: "{\"intensity\":\"low\"}",
        status: "pending",
        reason: "Calmer weather.",
        confidence: 0.7,
        source_message_ids_text: "",
        suggestion_count: 1,
        action: ""
      }]
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    for (const groupLabel of ["Review", "Scene", "Knowledge", "People & Places", "Advanced"]) {
      const escaped = groupLabel.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
      expect(await screen.findByRole("tab", { name: new RegExp(`^${escaped}( |\\d|$)`) })).toBeInTheDocument();
    }

    const reviewTab = screen.getByRole("tab", { name: /^Review/ });
    await waitFor(() => expect(reviewTab).toHaveTextContent("1 pending"));
    await userEvent.click(reviewTab);
    expect(screen.getByRole("tab", { name: /suggestions/i })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /world state/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /^Knowledge/ }));
    expect(screen.getByRole("tab", { name: /world state/i })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /suggestions/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /^Advanced/ }));
    expect(screen.getByRole("tab", { name: /scenario/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /audit/i })).toBeInTheDocument();
  });

  it("surfaces first-contact mission state and opens scan observations", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-contact",
        scenario_type: "first_contact_exploration",
        title: "Songs Under Europa",
        premise: "A survey crew finds patterned signals beneath the ice.",
        player_character_name: "Dr. Mara Voss",
        player_role: "Mission linguist",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-mission",
          key: "contact.mission",
          category: "mission",
          confidence: 1,
          value_json: "{\"summary\":\"Survey the hidden ocean.\"}",
          archived: false
        },
        {
          row_id: "state-translation",
          key: "contact.translation",
          category: "translation",
          confidence: 1,
          value_json: "{\"summary\":\"Three descending pulses may mean open water.\"}",
          archived: false
        },
        {
          row_id: "state-hazards",
          key: "contact.hazards",
          category: "threat",
          confidence: 1,
          value_json: "{\"summary\":\"Thermal fissures are spreading.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const openLookAround = vi.fn();
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "first_contact_exploration" })}
          runJob={vi.fn()}
          openLookAround={openLookAround}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("heading", { name: "First Contact" })).toBeInTheDocument();
    expect(screen.getByText("Survey the hidden ocean.")).toBeInTheDocument();
    expect(screen.getByText("Three descending pulses may mean open water.")).toBeInTheDocument();
    expect(screen.getByText("Thermal fissures are spreading.")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Scan target" }));

    expect(openLookAround).toHaveBeenCalledWith(
      expect.stringContaining("Scan the current exploration target")
    );
  });

  it("shows an empty first-contact board when seeded state is missing", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-contact",
        scenario_type: "first_contact_exploration",
        title: "Songs Under Europa",
        premise: "A survey crew finds patterned signals beneath the ice.",
        player_character_name: "Dr. Mara Voss",
        player_role: "Mission linguist",
        content_sections: []
      },
      world_state: []
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "first_contact_exploration" })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("heading", { name: "First Contact" })).toBeInTheDocument();
    expect(screen.getByText("No first-contact state yet.")).toBeInTheDocument();
  });

  it("hides the first-contact board for other scenario types", async () => {
    const worldPayload = worldDataPayload({
      world_state: [
        {
          row_id: "state-mission",
          key: "contact.mission",
          category: "mission",
          confidence: 1,
          value_json: "{\"summary\":\"Survey the hidden ocean.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} openLookAround={vi.fn()} />
      </QueryClientProvider>
    );

    await screen.findByRole("tab", { name: /^Advanced/ });
    expect(screen.queryByRole("heading", { name: "First Contact" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Scan target" })).not.toBeInTheDocument();
  });

  it("surfaces investigation case state without hidden solution material", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-case",
        scenario_type: "investigation_mystery",
        title: "Museum of Broken Hours",
        premise: "A curator disappears during a public gala.",
        player_character_name: "Inspector Mara Voss",
        player_role: "Lead investigator",
        content_sections: [
          ["case_facts", "Curator Elian Vale vanished from a sealed gallery."],
          ["case_status", "Unresolved; only public facts are known."],
          ["clues", "Undiscovered watch log gap points to the restoration lift."],
          ["timeline", "Public alarm at 9:21; hidden lift movement at 9:12."],
          ["red_herrings", "The bloody glove belongs to a mannequin repair."],
          ["hidden_truth", "Sera hid the smuggling ledger in the restoration lift."]
        ]
      },
      world_state: [
        {
          row_id: "state-clue",
          key: "clue.watch_log",
          category: "clue",
          confidence: 1,
          value_json: "{\"summary\":\"Watch log gap from 9:10 to 9:18 is confirmed.\",\"discovery_status\":\"discovered\"}",
          archived: false
        },
        {
          row_id: "state-timeline",
          key: "timeline.public_alarm",
          category: "timeline",
          confidence: 1,
          value_json: "{\"summary\":\"Public alarm sounded at 9:21.\",\"visibility\":\"public\"}",
          archived: false
        },
        {
          row_id: "state-hidden-clue",
          key: "clue.ledger",
          category: "clue",
          confidence: 1,
          value_json: "{\"summary\":\"Sera hid the ledger in the restoration lift.\",\"discovery_status\":\"undiscovered\"}",
          archived: false
        },
        {
          row_id: "state-red-herring",
          key: "case.red_herring.glove",
          category: "red_herring",
          confidence: 1,
          value_json: "{\"summary\":\"The bloody glove belongs to a mannequin repair.\"}",
          archived: false
        },
        {
          row_id: "state-hidden-truth",
          key: "case.hidden_truth",
          category: "truth",
          confidence: 1,
          value_json: "{\"summary\":\"Sera hid the smuggling ledger in the restoration lift.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const openLookAround = vi.fn();
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "investigation_mystery" })}
          runJob={vi.fn()}
          openLookAround={openLookAround}
          currentUser={{ id: "user-1", username: "child", role: "child", status: "active" }}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("heading", { name: "Case Board" })).toBeInTheDocument();
    expect(screen.getByText("Curator Elian Vale vanished from a sealed gallery.")).toBeInTheDocument();
    expect(screen.getByText("Unresolved; only public facts are known.")).toBeInTheDocument();
    expect(screen.getByText("Watch log gap from 9:10 to 9:18 is confirmed.")).toBeInTheDocument();
    expect(screen.getByText("Public alarm sounded at 9:21.")).toBeInTheDocument();
    expect(screen.queryByText(/restoration lift/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/bloody glove/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/hidden lift movement/i)).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Examine evidence" }));

    expect(openLookAround).toHaveBeenCalledWith(
      expect.stringContaining("Examine the evidence currently available to the player")
    );
  });

  it("hides the investigation case board for other scenario types", async () => {
    const worldPayload = worldDataPayload({
      world_state: [
        {
          row_id: "state-case",
          key: "case.status",
          category: "case",
          confidence: 1,
          value_json: "{\"summary\":\"The public case remains unresolved.\",\"visibility\":\"public\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} openLookAround={vi.fn()} />
      </QueryClientProvider>
    );

    await screen.findByRole("tab", { name: /^Advanced/ });
    expect(screen.queryByRole("heading", { name: "Case Board" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Review case file" })).not.toBeInTheDocument();
  });

  it("surfaces heist operation state and opens heist observations", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-heist",
        scenario_type: "heist_infiltration",
        title: "Skybank Treaty Job",
        premise: "A crew must steal a treaty from a floating bank.",
        player_character_name: "Mara Voss",
        player_role: "Crew planner",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-alert",
          key: "heist.alert",
          category: "threat",
          confidence: 1,
          value_json: "{\"level\":\"low\",\"alarm\":\"inactive\",\"heat\":\"minimal\",\"summary\":\"Suspicion low; alarm inactive.\"}",
          archived: false
        },
        {
          row_id: "state-security",
          key: "heist.security",
          category: "security",
          confidence: 1,
          value_json: "{\"summary\":\"Clockwork cameras active; west lock disabled.\"}",
          archived: false
        },
        {
          row_id: "state-extraction",
          key: "heist.extraction",
          category: "objective",
          confidence: 1,
          value_json: "{\"summary\":\"Primary storm skiff; fallback service stairs.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const openLookAround = vi.fn();
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "heist_infiltration" })}
          runJob={vi.fn()}
          openLookAround={openLookAround}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("heading", { name: "Heist Board" })).toBeInTheDocument();
    expect(screen.getByText("Clockwork cameras active; west lock disabled.")).toBeInTheDocument();
    expect(screen.getByText("Suspicion low; alarm inactive.")).toBeInTheDocument();
    expect(screen.getByText("Primary storm skiff; fallback service stairs.")).toBeInTheDocument();
    expect(screen.getByText("Security")).toBeInTheDocument();
    expect(screen.getByText("Alert / Heat")).toBeInTheDocument();
    expect(screen.getByText("Extraction")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Check security" }));

    expect(openLookAround).toHaveBeenCalledWith(
      expect.stringContaining("Review the current heist security model")
    );
  });

  it("saves heist alert adjustments through world-data apply", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-heist",
        scenario_type: "heist_infiltration",
        title: "Skybank Treaty Job",
        premise: "A crew must steal a treaty from a floating bank.",
        player_character_name: "Mara Voss",
        player_role: "Crew planner",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-alert",
          key: "heist.alert",
          original_key: "heist.alert",
          category: "threat",
          confidence: 1,
          value_json: "{\"level\":\"low\",\"alarm\":\"inactive\",\"heat\":\"minimal\",\"summary\":\"Suspicion low; alarm inactive.\",\"response\":\"guards relaxed\"}",
          archived: false
        },
        {
          row_id: "state-security",
          key: "heist.security",
          original_key: "heist.security",
          category: "security",
          confidence: 1,
          value_json: "{\"summary\":\"Clockwork cameras active.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? {
            model: {
              ...worldPayload,
              world_state: [
                {
                  ...(worldPayload.world_state?.[0] as Record<string, unknown>),
                  value_json: "{\"level\":\"high\",\"alarm\":\"active\",\"heat\":\"contained\",\"summary\":\"Alarm active; guards converge.\",\"response\":\"east post sealing gallery\"}"
                }
              ]
            },
            state_archive_count: 0,
            memory_archive_count: 0,
            summary_archive_count: 0
          }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "heist_infiltration" })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Adjust heat" }));
    fireEvent.change(screen.getByLabelText("Alert summary"), { target: { value: "Alarm active; guards converge." } });
    fireEvent.change(screen.getByLabelText("Alert level"), { target: { value: "high" } });
    fireEvent.change(screen.getByLabelText("Alarm state"), { target: { value: "active" } });
    fireEvent.change(screen.getByLabelText("Heat"), { target: { value: "contained" } });
    fireEvent.change(screen.getByLabelText("Response"), { target: { value: "east post sealing gallery" } });
    await userEvent.click(screen.getByRole("button", { name: "Save heat" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scenario: worldPayload.scenario,
        world_state: [
          {
            row_id: "state-alert",
            key: "heist.alert",
            category: "threat",
            value_json: JSON.stringify({
              level: "high",
              alarm: "active",
              heat: "contained",
              summary: "Alarm active; guards converge.",
              response: "east post sealing gallery"
            })
          }
        ]
      }
    });
  });

  it("keeps heist heat controls read-only for child users", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-heist",
        scenario_type: "heist_infiltration",
        title: "Skybank Treaty Job",
        premise: "A crew must steal a treaty from a floating bank.",
        player_character_name: "Mara Voss",
        player_role: "Crew planner",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-alert",
          key: "heist.alert",
          category: "threat",
          confidence: 1,
          value_json: "{\"level\":\"low\",\"alarm\":\"inactive\",\"summary\":\"Suspicion low.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "heist_infiltration" })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
          currentUser={{ id: "user-1", username: "child", role: "child", status: "active" }}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("heading", { name: "Heist Board" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Adjust heat" })).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/world-data/apply")).toBe(false);
  });

  it("hides the heist board for other scenario types", async () => {
    const worldPayload = worldDataPayload({
      world_state: [
        {
          row_id: "state-alert",
          key: "heist.alert",
          category: "threat",
          confidence: 1,
          value_json: "{\"summary\":\"Suspicion low.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} openLookAround={vi.fn()} />
      </QueryClientProvider>
    );

    await screen.findByRole("tab", { name: /^Advanced/ });
    expect(screen.queryByRole("heading", { name: "Heist Board" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Check security" })).not.toBeInTheDocument();
  });

  it("surfaces survival expedition state and opens expedition actions", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-expedition",
        scenario_type: "survival_expedition",
        title: "Whiteout Pass",
        premise: "A rescue caravan must cross a frozen mountain pass.",
        player_character_name: "Mara Voss",
        player_role: "Expedition lead",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-goal",
          key: "expedition.goal",
          category: "expedition",
          confidence: 1,
          value_json: "{\"summary\":\"Reach Northwatch before the fever medicine spoils.\"}",
          archived: false
        },
        {
          row_id: "state-resources",
          key: "expedition.resources",
          category: "inventory",
          confidence: 1,
          value_json: "{\"summary\":\"Food: 2 days. Water: 1 skin.\"}",
          archived: false
        },
        {
          row_id: "state-camp",
          key: "expedition.camp",
          category: "expedition",
          confidence: 1,
          value_json: "{\"summary\":\"Emergency bivouac below the ridge.\"}",
          archived: false
        },
        {
          row_id: "state-progress",
          key: "expedition.progress",
          category: "objective",
          confidence: 1,
          value_json: "{\"summary\":\"18 of 80 miles traveled; blizzard delay active.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const openLookAround = vi.fn();
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "survival_expedition" })}
          runJob={vi.fn()}
          openLookAround={openLookAround}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("heading", { name: "Expedition Dashboard" })).toBeInTheDocument();
    expect(screen.getByText("Reach Northwatch before the fever medicine spoils.")).toBeInTheDocument();
    expect(screen.getByText("Food: 2 days. Water: 1 skin.")).toBeInTheDocument();
    expect(screen.getByText("Emergency bivouac below the ridge.")).toBeInTheDocument();
    expect(screen.getByText("18 of 80 miles traveled; blizzard delay active.")).toBeInTheDocument();
    expect(screen.getByText("Resources")).toBeInTheDocument();
    expect(screen.getByText("Camp")).toBeInTheDocument();
    expect(screen.getByText("Progress")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Travel leg" }));

    expect(openLookAround).toHaveBeenCalledWith(
      expect.stringContaining("Travel the next expedition leg")
    );
  });

  it("saves survival expedition pressure through world-data apply", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-expedition",
        scenario_type: "survival_expedition",
        title: "Whiteout Pass",
        premise: "A rescue caravan must cross a frozen mountain pass.",
        player_character_name: "Mara Voss",
        player_role: "Expedition lead",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-resources",
          key: "expedition.resources",
          original_key: "expedition.resources",
          category: "inventory",
          confidence: 1,
          value_json: "{\"summary\":\"Food: 2 days. Water: 1 skin.\",\"ration\":\"standard\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? {
            model: {
              ...worldPayload,
              world_state: [
                {
                  ...(worldPayload.world_state?.[0] as Record<string, unknown>),
                  value_json: "{\"summary\":\"Food: 1 day. Water: half a skin.\",\"ration\":\"standard\"}"
                }
              ]
            },
            state_archive_count: 0,
            memory_archive_count: 0,
            summary_archive_count: 0
          }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "survival_expedition" })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Edit resources" }));
    fireEvent.change(screen.getByLabelText("Resources summary"), { target: { value: "Food: 1 day. Water: half a skin." } });
    await userEvent.click(screen.getByRole("button", { name: "Save resources" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scenario: worldPayload.scenario,
        world_state: [
          {
            row_id: "state-resources",
            key: "expedition.resources",
            category: "inventory",
            value_json: JSON.stringify({
              summary: "Food: 1 day. Water: half a skin.",
              ration: "standard"
            })
          }
        ]
      }
    });
  });

  it("keeps survival expedition pressure controls read-only for child users", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-expedition",
        scenario_type: "survival_expedition",
        title: "Whiteout Pass",
        premise: "A rescue caravan must cross a frozen mountain pass.",
        player_character_name: "Mara Voss",
        player_role: "Expedition lead",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-resources",
          key: "expedition.resources",
          category: "inventory",
          confidence: 1,
          value_json: "{\"summary\":\"Food: 2 days. Water: 1 skin.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "survival_expedition" })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
          currentUser={{ id: "user-1", username: "child", role: "child", status: "active" }}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("heading", { name: "Expedition Dashboard" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit resources" })).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/world-data/apply")).toBe(false);
  });

  it("hides the survival expedition board for other scenario types", async () => {
    const worldPayload = worldDataPayload({
      world_state: [
        {
          row_id: "state-resources",
          key: "expedition.resources",
          category: "inventory",
          confidence: 1,
          value_json: "{\"summary\":\"Food: 2 days. Water: 1 skin.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} openLookAround={vi.fn()} />
      </QueryClientProvider>
    );

    await screen.findByRole("tab", { name: /^Advanced/ });
    expect(screen.queryByRole("heading", { name: "Expedition Dashboard" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Travel leg" })).not.toBeInTheDocument();
  });

  it.each([
    {
      scenarioType: "settlement_builder",
      heading: "Settlement Board",
      actions: ["Allocate resources"],
      rows: [
        ["state-resources", "settlement.resources", "resource", "Timber: 18. Morale: steady."],
        ["state-projects", "settlement.projects", "project", "Clinic foundation complete."],
        ["state-extra", "project.aqueduct.status", "project", "Aqueduct survey pending."]
      ],
      labels: ["Resources", "Projects", "Aqueduct Status"]
    },
    {
      scenarioType: "time_loop",
      heading: "Loop Clock",
      actions: [
        "Review remembered facts",
        "Advance to known event",
        "Review reset rules"
      ],
      rows: [
        ["state-current", "loop.current", "loop", "Loop 3, afternoon phase."],
        ["state-knowledge", "loop.knowledge", "loop_persistent", "Tower code persists for Mara."],
        ["state-extra", "loop.npc_memory", "loop_boundary", "NPCs reset unless marked by salt."]
      ],
      labels: ["Current", "Knowledge", "Npc Memory"]
    },
    {
      scenarioType: "political_intrigue",
      heading: "Political Ledger",
      actions: [
        "Call in favor",
        "Review leverage",
        "Track standing",
        "Check calendar"
      ],
      rows: [
        ["state-standing", "intrigue.standing", "reputation", "Reformers trust Mara; Old Families resist."],
        ["state-obligations", "intrigue.obligations", "obligation", "Orro owes Mara one endorsement."],
        ["state-extra", "obligation.orro.owed_to_mara", "obligation", "The endorsement is due before midnight."]
      ],
      labels: ["Standing", "Obligations", "Orro Owed To Mara"]
    },
    {
      scenarioType: "monster_hunt_bounty",
      heading: "Hunt Board",
      actions: [
        "Review leads",
        "Track target",
        "Prepare gear",
        "Check rival pressure"
      ],
      rows: [
        ["state-leads", "hunt.leads", "clue", "Blood spoor points north."],
        ["state-preparation", "hunt.preparation", "inventory", "Silver traps packed."],
        ["state-extra", "clue.blood_trail.status", "clue", "Blood trail is fresh."]
      ],
      labels: ["Leads", "Preparation", "Blood Trail Status"]
    },
    {
      scenarioType: "road_trip_pilgrimage",
      heading: "Journey Board",
      actions: [
        "Choose next stop",
        "Travel leg",
        "Check transport",
        "Review companion threads"
      ],
      rows: [
        ["state-route", "journey.route", "location", "Old highway to Saint Orla."],
        ["state-progress", "journey.progress", "objective", "Three of nine shrines visited."],
        ["state-extra", "stop.old_bridge.threads", "relationship", "The argument at Old Bridge is unresolved."]
      ],
      labels: ["Route", "Progress", "Old Bridge Threads"]
    },
    {
      scenarioType: "merchant_trade_route",
      heading: "Trade Ledger",
      actions: [
        "Review cargo",
        "Settle contract",
        "Record debt",
        "Check market",
        "Plot route"
      ],
      rows: [
        ["state-cargo", "trade.cargo", "inventory", "Silk bales intact."],
        ["state-contracts", "trade.contracts", "contract", "Harbor delivery due at dawn."],
        ["state-extra", "debt.harbor_guild.status", "contract", "Harbor guild debt is overdue."]
      ],
      labels: ["Cargo", "Contracts", "Harbor Guild Status"]
    }
  ])("surfaces %s management state and opens board actions", async ({ scenarioType, heading, actions, rows, labels }) => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: `scenario-${scenarioType}`,
        scenario_type: scenarioType,
        title: heading,
        premise: "A managed scenario.",
        player_character_name: "Mara Voss",
        player_role: "Lead",
        content_sections: []
      },
      world_state: rows.map(([row_id, key, category, summary]) => ({
        row_id,
        key,
        category,
        confidence: 1,
        value_json: JSON.stringify({ summary }),
        archived: false
      }))
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const openLookAround = vi.fn();
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: scenarioType })}
          runJob={vi.fn()}
          openLookAround={openLookAround}
        />
      </QueryClientProvider>
    );

    const boardHeading = await screen.findByRole("heading", { name: heading });
    const board = boardHeading.closest("section");
    expect(board).not.toBeNull();
    const boardView = within(board as HTMLElement);
    for (const [, , , summary] of rows) {
      expect(boardView.getByText(summary)).toBeInTheDocument();
    }
    for (const label of labels) {
      expect(boardView.getByText(label)).toBeInTheDocument();
    }

    for (const label of actions) {
      await userEvent.click(boardView.getByRole("button", { name: label }));
      expect(openLookAround).toHaveBeenCalledWith(expect.stringContaining(label));
    }
  });

  it.each([
    ["settlement_builder", "monster_hunt_bounty", "Settlement Board", "Allocate resources"],
    ["time_loop", "political_intrigue", "Loop Clock", "Review remembered facts"],
    ["political_intrigue", "time_loop", "Political Ledger", "Call in favor"],
    ["monster_hunt_bounty", "settlement_builder", "Hunt Board", "Review leads"],
    ["road_trip_pilgrimage", "merchant_trade_route", "Journey Board", "Travel leg"],
    ["merchant_trade_route", "road_trip_pilgrimage", "Trade Ledger", "Check market"]
  ])("hides the %s management board for other scenario types", async (_scenarioType, otherScenarioType, heading, actionLabel) => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: `scenario-${otherScenarioType}`,
        scenario_type: otherScenarioType,
        title: "Other board",
        premise: "Another managed scenario.",
        player_character_name: "Mara Voss",
        player_role: "Lead",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-management",
          key: "settlement.resources",
          category: "resource",
          confidence: 1,
          value_json: "{\"summary\":\"Timber: 18.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} openLookAround={vi.fn()} />
      </QueryClientProvider>
    );

    await screen.findByRole("tab", { name: /^Advanced/ });
    expect(screen.queryByRole("heading", { name: heading })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: actionLabel })).not.toBeInTheDocument();
  });

  it.each([
    ["settlement_builder", "Settlement Board", "settlement.resources", "Resources", "Timber: 12. Morale: strained."],
    ["time_loop", "Loop Clock", "loop.current", "Current", "Loop 4, dusk phase."],
    ["political_intrigue", "Political Ledger", "intrigue.standing", "Standing", "Harbor guild support is wavering."],
    ["monster_hunt_bounty", "Hunt Board", "hunt.leads", "Leads", "New claw marks found near the mill."],
    ["road_trip_pilgrimage", "Journey Board", "journey.progress", "Progress", "Four of nine shrines visited."],
    ["merchant_trade_route", "Trade Ledger", "trade.contracts", "Contracts", "Harbor delivery marked complete."]
  ])("saves %s board summary edits through world-data apply", async (scenarioType, heading, key, label, nextSummary) => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: `scenario-${scenarioType}`,
        scenario_type: scenarioType,
        title: heading,
        premise: "A managed scenario.",
        player_character_name: "Mara Voss",
        player_role: "Lead",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-management",
          key,
          original_key: key,
          category: "resource",
          confidence: 1,
          value_json: JSON.stringify({ summary: "Original summary.", note: "preserved" }),
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/apply"
        ? {
            model: {
              ...worldPayload,
              world_state: [
                {
                  ...(worldPayload.world_state?.[0] as Record<string, unknown>),
                  value_json: JSON.stringify({ summary: nextSummary, note: "preserved" })
                }
              ]
            },
            state_archive_count: 0,
            memory_archive_count: 0,
            summary_archive_count: 0
          }
        : worldPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: scenarioType })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: `Edit ${label.toLowerCase()}` }));
    fireEvent.change(screen.getByLabelText(`${label} summary`), { target: { value: nextSummary } });
    await userEvent.click(screen.getByRole("button", { name: `Save ${label.toLowerCase()}` }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        scenario: worldPayload.scenario,
        world_state: [
          {
            row_id: "state-management",
            key,
            value_json: JSON.stringify({
              summary: nextSummary,
              note: "preserved"
            })
          }
        ]
      }
    });
  });

  it.each([
    ["settlement_builder", "Settlement Board", "settlement.resources", "Resources"],
    ["time_loop", "Loop Clock", "loop.current", "Current"],
    ["political_intrigue", "Political Ledger", "intrigue.standing", "Standing"],
    ["monster_hunt_bounty", "Hunt Board", "hunt.leads", "Leads"],
    ["road_trip_pilgrimage", "Journey Board", "journey.progress", "Progress"],
    ["merchant_trade_route", "Trade Ledger", "trade.contracts", "Contracts"]
  ])("keeps %s management edits read-only for child users", async (scenarioType, heading, key, label) => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: `scenario-${scenarioType}`,
        scenario_type: scenarioType,
        title: heading,
        premise: "A managed scenario.",
        player_character_name: "Mara Voss",
        player_role: "Lead",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-management",
          key,
          category: "resource",
          confidence: 1,
          value_json: "{\"summary\":\"Original summary.\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: scenarioType })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
          currentUser={{ id: "user-1", username: "child", role: "child", status: "active" }}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("heading", { name: heading })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: `Edit ${label.toLowerCase()}` })).not.toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/world-data/apply")).toBe(false);
  });

  it.each([
    {
      scenarioType: "time_loop",
      heading: "Loop Clock",
      sections: [
        ["current_loop_state", "Loop 1, dawn phase."],
        ["persistent_knowledge", "Tower code persists for the player."]
      ],
      summaries: ["Loop 1, dawn phase.", "Tower code persists for the player."]
    },
    {
      scenarioType: "political_intrigue",
      heading: "Political Ledger",
      sections: [
        ["reputation_and_standing", "Reformers trust Mara."],
        ["obligations_and_favors", "Orro owes Mara one endorsement."]
      ],
      summaries: ["Reformers trust Mara.", "Orro owes Mara one endorsement."]
    },
    {
      scenarioType: "monster_hunt_bounty",
      heading: "Hunt Board",
      sections: [
        ["leads_and_clues", "Blue sap points north."],
        ["preparation_state", "Silver traps packed."]
      ],
      summaries: ["Blue sap points north.", "Silver traps packed."]
    },
    {
      scenarioType: "road_trip_pilgrimage",
      heading: "Journey Board",
      sections: [
        ["transport_and_supplies", "One wagon, two mules."],
        ["journey_progress", "Day one to Lantern Ford."]
      ],
      summaries: ["One wagon, two mules.", "Day one to Lantern Ford."]
    },
    {
      scenarioType: "merchant_trade_route",
      heading: "Trade Ledger",
      sections: [
        ["cargo_inventory", "Cedar oil jars intact."],
        ["contracts_and_debts", "Harbor delivery due at dawn."]
      ],
      summaries: ["Cedar oil jars intact.", "Harbor delivery due at dawn."]
    }
  ])("surfaces %s scenario setup when management world state is not seeded", async ({ scenarioType, heading, sections, summaries }) => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: `scenario-${scenarioType}`,
        scenario_type: scenarioType,
        title: heading,
        premise: "A managed scenario.",
        player_character_name: "Mara Voss",
        player_role: "Lead",
        content_sections: sections as ScenarioContentSection[]
      },
      world_state: []
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: scenarioType })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
        />
      </QueryClientProvider>
    );

    const boardHeading = await screen.findByRole("heading", { name: heading });
    const board = boardHeading.closest("section");
    expect(board).not.toBeNull();
    const boardView = within(board as HTMLElement);
    for (const summary of summaries) {
      expect(boardView.getByText(summary)).toBeInTheDocument();
    }
    expect(boardView.queryByText("No records yet.")).not.toBeInTheDocument();
  });

  it("keeps private political intrigue rows off the player-facing ledger unless explicitly visible", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-political_intrigue",
        scenario_type: "political_intrigue",
        title: "Political Ledger",
        premise: "A managed scenario.",
        player_character_name: "Mara Voss",
        player_role: "Lead",
        content_sections: []
      },
      world_state: [
        {
          row_id: "state-standing",
          key: "intrigue.standing",
          category: "reputation",
          confidence: 1,
          value_json: "{\"summary\":\"Reformers trust Mara.\"}",
          archived: false
        },
        {
          row_id: "state-hidden-secret",
          key: "intrigue.secrets.hidden_vote",
          category: "leverage",
          confidence: 1,
          value_json: "{\"summary\":\"Private blackmail evidence remains unrevealed.\",\"visibility\":\"private\"}",
          archived: false
        },
        {
          row_id: "state-not-visible",
          key: "intrigue.hidden_plan",
          category: "leverage",
          confidence: 1,
          value_json: "{\"summary\":\"Council ambush remains concealed.\",\"visibility\":\"not_visible\"}",
          archived: false
        },
        {
          row_id: "state-player-visible-false",
          key: "intrigue.standing.vote",
          category: "reputation",
          confidence: 1,
          value_json: "{\"summary\":\"Council vote moved to dawn.\",\"player_visible\":false}",
          archived: false
        },
        {
          row_id: "state-public-secret",
          key: "intrigue.secrets.public_censure",
          category: "leverage",
          confidence: 1,
          value_json: "{\"summary\":\"Public censure threat is revealed.\",\"visibility\":\"player-visible\"}",
          archived: false
        }
      ]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "political_intrigue" })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
        />
      </QueryClientProvider>
    );

    const boardHeading = await screen.findByRole("heading", { name: "Political Ledger" });
    const board = boardHeading.closest("section");
    expect(board).not.toBeNull();
    const boardView = within(board as HTMLElement);
    expect(boardView.getByText("Reformers trust Mara.")).toBeInTheDocument();
    expect(boardView.getByText("Public censure threat is revealed.")).toBeInTheDocument();
    expect(boardView.queryByText("Private blackmail evidence remains unrevealed.")).not.toBeInTheDocument();
    expect(boardView.queryByText("Council ambush remains concealed.")).not.toBeInTheDocument();
    expect(boardView.queryByText("Council vote moved to dawn.")).not.toBeInTheDocument();
  });

  it("does not use political secret scenario sections as player-facing ledger fallbacks", async () => {
    const worldPayload = worldDataPayload({
      scenario: {
        scenario_id: "scenario-political_intrigue",
        scenario_type: "political_intrigue",
        title: "Political Ledger",
        premise: "A managed scenario.",
        player_character_name: "Mara Voss",
        player_role: "Lead",
        content_sections: [
          ["reputation_and_standing", "Reformers trust Mara."],
          ["secrets_and_leverage", "Only Mara knows Orro moved the silver."],
          ["public_private_knowledge", "Only the player knows the regent is blackmailed."]
        ]
      },
      world_state: []
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel
          model={runtimeModel({ active_scenario_type: "political_intrigue" })}
          runJob={vi.fn()}
          openLookAround={vi.fn()}
        />
      </QueryClientProvider>
    );

    const boardHeading = await screen.findByRole("heading", { name: "Political Ledger" });
    const board = boardHeading.closest("section");
    expect(board).not.toBeNull();
    const boardView = within(board as HTMLElement);
    expect(boardView.getByText("Reformers trust Mara.")).toBeInTheDocument();
    expect(boardView.queryByText("Only Mara knows Orro moved the silver.")).not.toBeInTheDocument();
    expect(boardView.queryByText("Only the player knows the regent is blackmailed.")).not.toBeInTheDocument();
  });

  it("hides the pending review badge when there are no pending suggestions", async () => {
    const worldPayload = worldDataPayload({
      suggestion_groups: [{
        group_id: "group-1",
        suggestion_ids: ["s-1"],
        update_type: "update",
        entity_type: "world_state",
        entity_id: null,
        field_path: "storm",
        proposed_value_json: "{\"intensity\":\"low\"}",
        status: "superseded",
        reason: "Old.",
        confidence: 0.6,
        source_message_ids_text: "",
        suggestion_count: 1,
        action: ""
      }]
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const reviewTab = await screen.findByRole("tab", { name: /^Review/ });
    await waitFor(() => expect(reviewTab).toHaveTextContent("1"));
    expect(reviewTab).not.toHaveTextContent("pending");
    expect(screen.queryByRole("button", { name: /apply suggestion/i })).not.toBeInTheDocument();
  });

  it("switches world data sections when the user changes groups", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "Mara",
        player_role: "Keeper",
        content_sections: []
      },
      scene: { snapshot_id: "scene-1", situation: "Fog over the beacon" },
      context_inputs: [{ context_source_id: "ctx-1", source_type: "memory", source_id: "m1", title: "Storm memory", body: "Storm clouds.", fact_type: "weather", importance: 0.6, source_message_count: 1, token_estimate: 20 }],
      locations: [{ location_id: "loc-1", name: "Beacon Tower", description: "Tower above the fog." }],
      characters: [{ character_id: "char-1", name: "Mara", role: "Keeper" }]
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("tab", { name: /^Advanced/ })).toHaveAttribute("aria-selected", "true");

    await userEvent.click(screen.getByRole("tab", { name: /^Scene/ }));
    expect(screen.getByRole("tab", { name: /context inputs/i })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /characters/i })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: /context inputs/i }));
    expect(screen.getAllByText("Storm memory").length).toBeGreaterThan(0);

    await userEvent.click(screen.getByRole("tab", { name: /^People/ }));
    expect(screen.getByRole("tab", { name: /characters/i })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /locations/i })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: /context inputs/i })).not.toBeInTheDocument();
  });

  it("reports world data search matches grouped by section across workflows", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Lantern Keep",
        premise: "A watchtower.",
        player_character_name: "Mara",
        player_role: "Keeper",
        content_sections: []
      },
      scene: { snapshot_id: "scene-1", situation: "Beacon waits in fog" },
      world_state: [
        { row_id: "state-1", key: "storm", original_key: "storm", category: "weather", confidence: 0.8, value_json: "{\"intensity\":\"low\"}", archived: false },
        { row_id: "state-2", key: "lantern", original_key: "lantern", category: "object", confidence: 0.7, value_json: "{\"lit\":true}", archived: false }
      ],
      memories: [{ memory_id: "memory-1", body: "Storm clouds gathered." }],
      locations: [{ location_id: "loc-1", name: "Beacon Tower", description: "Above the fog." }]
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const knowledgeTab = await screen.findByRole("tab", { name: /^Knowledge/ });
    await waitFor(() => expect(knowledgeTab).toHaveTextContent("3"));

    await userEvent.type(screen.getByLabelText("Search world data"), "storm");

    expect(await screen.findByText("2 matches in 2 sections")).toBeInTheDocument();
    const knowledgeAfter = screen.getByRole("tab", { name: /^Knowledge/ });
    expect(knowledgeAfter).toHaveTextContent("2");
    const sceneTab = screen.getByRole("tab", { name: /^Scene/ });
    expect(sceneTab).toHaveTextContent("0");
    const peopleTab = screen.getByRole("tab", { name: /^People/ });
    expect(peopleTab).toHaveTextContent("0");
    expect(screen.queryByText("lantern")).not.toBeInTheDocument();
  });

  it("pages large world data lists but still finds matches beyond the first page", async () => {
    const worldState = Array.from({ length: 95 }, (_, index) => {
      const number = index + 1;
      return {
        row_id: `state-${number}`,
        key: number === 95 ? "distant-storm-cache" : `beacon-${number}`,
        original_key: number === 95 ? "distant-storm-cache" : `beacon-${number}`,
        category: "test",
        confidence: 0.8,
        value_json: JSON.stringify({ note: number === 95 ? "Hidden storm supplies" : `Supply ${number}` }),
        source_message_id: null,
        archived: false
      };
    });
    const worldPayload = {
      active_save_id: "save-1",
      scenario: null,
      world_state: worldState,
      memories: [],
      summaries: [],
      links: [],
      suggestions: [],
      suggestion_groups: []
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^Knowledge/ }));
    await userEvent.click(screen.getByRole("tab", { name: /world state/i }));
    expect(await screen.findByText("beacon-1")).toBeInTheDocument();
    expect(screen.getByText("beacon-80")).toBeInTheDocument();
    expect(screen.queryByText("beacon-81")).not.toBeInTheDocument();
    expect(screen.queryByText("distant-storm-cache")).not.toBeInTheDocument();

    await userEvent.type(screen.getByLabelText("Search world data"), "distant storm");

    expect(await screen.findByText("distant-storm-cache")).toBeInTheDocument();
    expect(screen.queryByText("beacon-1")).not.toBeInTheDocument();
  });

  it("reveals additional world data rows on demand", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: null,
      world_state: Array.from({ length: 95 }, (_, index) => {
        const number = index + 1;
        return {
          row_id: `state-${number}`,
          key: `beacon-${number}`,
          original_key: `beacon-${number}`,
          category: "test",
          confidence: 0.8,
          value_json: JSON.stringify({ note: `Supply ${number}` }),
          source_message_id: null,
          archived: false
        };
      }),
      memories: [],
      summaries: [],
      links: [],
      suggestions: [],
      suggestion_groups: []
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload }));
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^Knowledge/ }));
    await userEvent.click(screen.getByRole("tab", { name: /world state/i }));
    expect(await screen.findByText("beacon-80")).toBeInTheDocument();
    expect(screen.queryByText("beacon-95")).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: /show more world data/i }));

    expect(await screen.findByText("beacon-95")).toBeInTheDocument();
  });

  it("does not render nested world row details until a row is expanded", async () => {
    const worldPayload = {
      active_save_id: "save-1",
      scenario: null,
      locations: [
        {
          location_id: "loc-1",
          name: "Sealed Door",
          status: "quiet",
          deep_secret: "Only visible after expansion"
        }
      ],
      world_state: [],
      memories: [],
      summaries: [],
      links: [],
      suggestions: [],
      suggestion_groups: []
    };
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => worldPayload }));
    const { WorldPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <WorldPanel model={runtimeModel()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /^People/ }));
    await userEvent.click(screen.getByRole("tab", { name: /locations/i }));
    expect(await screen.findByText("Sealed Door")).toBeInTheDocument();
    expect(screen.queryByText("Only visible after expansion")).not.toBeInTheDocument();

    await userEvent.click(screen.getByText("Sealed Door"));

    expect(await screen.findByText("Only visible after expansion")).toBeInTheDocument();
  });

  it("labels snake case controls", async () => {
    const { labelize } = await import("./main");

    expect(labelize("active_loss_outcome")).toBe("Active Loss Outcome");
  });

  it("renders chronicle markdown blocks", async () => {
    const { MarkdownView } = await import("./main");

    render(
      <MarkdownView
        message={{
          message_id: "m1",
          role: "narrator",
          speaker_name: null,
          body: "**Fallback**",
          actions: [],
          markdown_blocks: [
            { kind: "paragraph", spans: [{ kind: "strong", text: "Bold" }, { kind: "text", text: " move" }] },
            { kind: "code_block", text: "roll()" }
          ]
        }}
      />
    );

    expect(screen.getByText("Bold")).toBeInTheDocument();
    expect(screen.getByText("roll()")).toBeInTheDocument();
  });

  it("renders safe markdown links as anchors", async () => {
    const { MarkdownView } = await import("./main");

    render(
      <MarkdownView
        message={{
          message_id: "m1",
          role: "narrator",
          speaker_name: null,
          body: "",
          actions: [],
          markdown_blocks: [
            {
              kind: "paragraph",
              spans: [
                { kind: "link", text: "Docs", target: "https://example.test/guide" },
                { kind: "text", text: " " },
                { kind: "link", text: "Email", target: "mailto:keeper@example.test" },
                { kind: "text", text: " " },
                { kind: "link", text: "Local", target: "/chronicle" }
              ]
            }
          ]
        }}
      />
    );

    expect(screen.getByRole("link", { name: "Docs" })).toHaveAttribute("href", "https://example.test/guide");
    expect(screen.getByRole("link", { name: "Email" })).toHaveAttribute("href", "mailto:keeper@example.test");
    expect(screen.getByRole("link", { name: "Local" })).toHaveAttribute("href", "/chronicle");
  });

  it("renders unsafe markdown link targets as plain text", async () => {
    const { MarkdownView } = await import("./main");

    const { container } = render(
      <MarkdownView
        message={{
          message_id: "m1",
          role: "narrator",
          speaker_name: null,
          body: "",
          actions: [],
          markdown_blocks: [
            {
              kind: "paragraph",
              spans: [
                { kind: "link", text: "Script", target: "javascript:alert(1)" },
                { kind: "text", text: " " },
                { kind: "link", text: "Data", target: "data:text/html,<svg onload=alert(1)>" }
              ]
            }
          ]
        }}
      />
    );

    expect(container.querySelector(".prose")?.textContent).toBe("Script Data");
    expect(screen.queryByRole("link", { name: "Script" })).not.toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Data" })).not.toBeInTheDocument();
  });

  it("does not duplicate an optimistic player message after it is persisted", async () => {
    const { chronicleMessages } = await import("./main");
    const model = {
      saves: [],
      active_save_id: "save-1",
      active_save_title: "Lantern Keep",
      active_scenario_type: null,
      scenario_title: "Lantern Keep",
      scene_title: "Beacon",
      chronicle: {
        messages: [
          { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] },
          { message_id: "m2", role: "player", speaker_name: "Keeper", body: "Light the beacon", actions: [] }
        ]
      },
      media: null,
      action_choices: null,
      model_indicator: "",
      failed_save: false,
      composer_enabled: true,
      failure_text: null,
      status: null,
      error: null
    };
    const pendingMessage = {
      message_id: "pending-player-message",
      role: "player",
      speaker_name: null,
      body: "Light the beacon",
      actions: [],
      markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: "Light the beacon" }] }]
    };

    expect(chronicleMessages(model, pendingMessage).map((message) => message.message_id)).toEqual(["m1", "m2"]);
    expect(chronicleMessages({ ...model, chronicle: { messages: model.chronicle.messages.slice(0, 1) } }, pendingMessage).map((message) => message.message_id)).toEqual([
      "m1",
      "pending-player-message"
    ]);
  });

  it("does not show an optimistic player artifact after the narrator response persists", async () => {
    const { chronicleMessages } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m0", role: "player", speaker_name: "Keeper", body: "Light the beacon", actions: [] },
          { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] },
          { message_id: "m2", role: "player", speaker_name: "Keeper", body: "Light the beacon", actions: [] },
          { message_id: "m3", role: "narrator", speaker_name: null, body: "The bell answers.", actions: [] }
        ]
      }
    });
    const pendingMessage = {
      message_id: "pending-player-message",
      role: "player",
      speaker_name: null,
      body: "Light the beacon",
      actions: [],
      markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: "Light the beacon" }] }],
      pending_after_message_id: "m1"
    };

    expect(chronicleMessages(model, pendingMessage).map((message) => message.message_id)).toEqual(["m0", "m1", "m2", "m3"]);
    expect(chronicleMessages({ ...model, chronicle: { messages: model.chronicle.messages.slice(0, 2) } }, pendingMessage).map((message) => message.message_id)).toEqual([
      "m0",
      "m1",
      "pending-player-message"
    ]);
  });

  it("prepends older chronicle pages without duplicating loaded messages", async () => {
    const { mergeChroniclePage } = await import("./main");
    const model = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m3", role: "player", speaker_name: "Keeper", body: "Current first", actions: [] },
          { message_id: "m4", role: "narrator", speaker_name: null, body: "Latest", actions: [] }
        ],
        has_more_before: true,
        oldest_message_id: "m3"
      }
    });

    const merged = mergeChroniclePage(model, {
      messages: [
        { message_id: "m1", role: "narrator", speaker_name: null, body: "Earliest", actions: [] },
        { message_id: "m2", role: "player", speaker_name: "Keeper", body: "Earlier", actions: [] },
        { message_id: "m3", role: "player", speaker_name: "Keeper", body: "Current first", actions: [] }
      ],
      has_more_before: false,
      oldest_message_id: "m1"
    });

    expect(merged?.chronicle.messages.map((message) => message.message_id)).toEqual([
      "m1",
      "m2",
      "m3",
      "m4"
    ]);
    expect(merged?.chronicle.has_more_before).toBe(false);
    expect(merged?.chronicle.oldest_message_id).toBe("m1");
  });

  it("formats section and status_text progress events", async () => {
    const { progressLabel } = await import("./main");

    expect(progressLabel({ status_text: "Post-turn: memories running" })).toBe("Post-turn: memories running");
    expect(progressLabel({ status_text: "Selecting context", jobs: [{ name: "context_selection", status: "running" }] })).toBe(
      "Selecting context"
    );
    expect(progressLabel({ jobs: [{ name: "state", status: "running" }, { name: "context", status: "pending" }, { name: "director", status: "pending" }, { name: "characters", status: "pending" }] })).toBe(
      "Post-turn: World state running, Context update pending, Director pressure pending, Character cleanup pending"
    );
    expect(progressLabel({ label: "Retrying chat request (attempt 2 of 3)..." })).toBe("Retrying chat request (attempt 2 of 3)...");
    expect(progressLabel({ section_id: "opening_message", status: "complete", completed_count: 2, total_count: 4 })).toBe(
      "Opening Message: complete 2/4"
    );
  });

  it("renders compact pending jobs as grouped summaries", async () => {
    const { PendingJobsTray } = await import("./main");
    const onCancel = vi.fn();

    render(
      <PendingJobsTray
        mode="compact"
        jobs={[
          {
            job: { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 },
            progress: "Submitting turn"
          },
          {
            job: { id: "job-2", type: "image_generation", status: "queued", result: null, error: null, created_at: 2 },
            progress: "Queued"
          }
        ]}
        onCancel={onCancel}
      />
    );

    expect(screen.getByLabelText("Pending jobs")).toBeInTheDocument();
    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("Active jobs")).toBeInTheDocument();
    expect(screen.getByText("Chat turn; Generating image")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Show pending jobs" }));
    await userEvent.click(screen.getByRole("button", { name: "Cancel Generating image" }));
    expect(onCancel).toHaveBeenCalledWith(expect.objectContaining({ job: expect.objectContaining({ id: "job-2" }) }));
    expect(screen.getByRole("button", { name: "Hide pending jobs" })).toBeInTheDocument();
  });

  it("uses compact history wording for summary backfill jobs", async () => {
    const { jobTypeLabel } = await import("./main");

    expect(jobTypeLabel("summary_backfill")).toBe("Compacting history");
    expect(jobTypeLabel("message_edit")).toBe("Editing message");
  });

  it("renders expanded pending jobs with post-turn phase detail", async () => {
    const { PendingJobsTray } = await import("./main");

    render(
      <PendingJobsTray
        mode="expanded"
        jobs={[
          {
            job: { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 },
            progress: "Post-turn: World state running, Context update pending, Character cleanup pending",
            phases: [
              { name: "state", status: "running" },
              { name: "context", status: "pending" },
              { name: "characters", status: "pending" }
            ]
          },
          {
            job: { id: "job-2", type: "image_generation", status: "queued", result: null, error: null, created_at: 2 },
            progress: "Queued"
          }
        ]}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByText("2")).toBeInTheDocument();
    expect(screen.getByText("Chat turn")).toBeInTheDocument();
    expect(screen.getByText("World state")).toBeInTheDocument();
    expect(screen.getByText("running")).toBeInTheDocument();
    expect(screen.getByText("Context update")).toBeInTheDocument();
    expect(screen.getByText("Character cleanup")).toBeInTheDocument();
    expect(screen.getByText("Generating image")).toBeInTheDocument();
  });

  it("renders expanded pending jobs with pre-narrator phase detail", async () => {
    const { PendingJobsTray } = await import("./main");

    render(
      <PendingJobsTray
        mode="expanded"
        jobs={[
          {
            job: { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 },
            progress: "Selecting context",
            phases: [
              { name: "submission", status: "succeeded" },
              { name: "history", status: "succeeded" },
              { name: "input", status: "succeeded" },
              { name: "time_state", status: "skipped" },
              { name: "character_planning", status: "skipped" },
              { name: "context_selection", status: "running" },
              { name: "prompt", status: "pending" },
              { name: "narrator", status: "pending" },
              { name: "response_checks", status: "pending" },
              { name: "save_narration", status: "pending" },
              { name: "action_choices", status: "pending" }
            ]
          }
        ]}
        onCancel={vi.fn()}
      />
    );

    expect(screen.getByText("Selecting context")).toBeInTheDocument();
    expect(screen.getByText("Submitting")).toBeInTheDocument();
    expect(screen.getByText("History check")).toBeInTheDocument();
    expect(screen.getByText("Saving input")).toBeInTheDocument();
    expect(screen.getByText("World time")).toBeInTheDocument();
    expect(screen.getByText("Character planning")).toBeInTheDocument();
    expect(screen.getByText("Context selection")).toBeInTheDocument();
    expect(screen.getByText("Prompt prep")).toBeInTheDocument();
    expect(screen.getByText("Narrator response")).toBeInTheDocument();
    expect(screen.getByText("Response checks")).toBeInTheDocument();
    expect(screen.getByText("Saving narration")).toBeInTheDocument();
    expect(screen.getByText("Action choices")).toBeInTheDocument();
  });

  it("hides automatic state pruning outside expanded full mode", async () => {
    const { PendingJobsTray } = await import("./main");
    const jobs: import("./main").TrackedJob[] = [
      {
        job: { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 },
        progress: "Submitting turn"
      },
      {
        job: { id: "job-2", type: "state_pruning", status: "running", result: null, error: null, created_at: 2 },
        progress: "Cleaning world state"
      }
    ];

    const { rerender } = render(<PendingJobsTray mode="compact" jobs={jobs} onCancel={vi.fn()} />);

    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("Chat turn")).toBeInTheDocument();
    expect(screen.queryByText("Cleaning world state")).not.toBeInTheDocument();

    rerender(<PendingJobsTray mode="expanded" jobs={jobs} onCancel={vi.fn()} />);
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByText("Chat turn")).toBeInTheDocument();
    expect(screen.queryByText("Cleaning world state")).not.toBeInTheDocument();
  });

  it("shows automatic state pruning in expanded full mode", async () => {
    const { PendingJobsTray } = await import("./main");
    const onCancel = vi.fn();

    render(
      <PendingJobsTray
        mode="expanded_full"
        jobs={[
          {
            job: { id: "job-1", type: "state_pruning", status: "running", result: null, error: null, created_at: 1 },
            progress: "Cleaning world state"
          }
        ]}
        onCancel={onCancel}
      />
    );

    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getAllByText("Cleaning world state").length).toBeGreaterThan(0);
    await userEvent.click(screen.getByRole("button", { name: "Cancel Cleaning world state" }));
    expect(onCancel).toHaveBeenCalledTimes(1);
  });

  it("preserves expanded post-turn phases when active job polling refreshes a tracked job", async () => {
    const { trackedActiveJob } = await import("./main");
    const existing = {
      job: { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 } satisfies Job,
      progress: "Post-turn: World state running, Character cleanup pending",
      phases: [
        { name: "state", status: "running" },
        { name: "characters", status: "pending" }
      ]
    };

    expect(trackedActiveJob({ ...existing.job, updated_at: 2 }, existing)).toEqual({
      job: { ...existing.job, updated_at: 2 },
      progress: "Post-turn: World state running, Character cleanup pending",
      phases: existing.phases
    });
  });

  it("preserves detailed post-turn progress when polling returns stale generic progress", async () => {
    const { trackedActiveJob } = await import("./main");
    const existing = {
      job: { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 } satisfies Job,
      progress: "Post-turn: World state running, Character cleanup pending",
      phases: [
        { name: "state", status: "running" },
        { name: "characters", status: "pending" }
      ]
    };
    const job = {
      ...existing.job,
      updated_at: 2,
      latest_progress: { label: "Updating world state" }
    } satisfies Job;

    expect(trackedActiveJob(job, existing)).toEqual({
      job,
      progress: "Post-turn: World state running, Character cleanup pending",
      phases: existing.phases
    });
  });

  it("hydrates expanded post-turn phases from active job progress", async () => {
    const { trackedActiveJob } = await import("./main");
    const job = {
      id: "job-1",
      type: "chat_turn",
      status: "running",
      result: null,
      error: null,
      created_at: 1,
      latest_progress: {
        jobs: [
          { name: "state", status: "complete" },
          { name: "context", status: "running" },
          { name: "characters", status: "pending" }
        ]
      }
    } satisfies Job;

    expect(trackedActiveJob(job)).toEqual({
      job,
      progress: "Post-turn: World state complete, Context update running, Character cleanup pending",
      phases: [
        { name: "state", status: "complete" },
        { name: "context", status: "running" },
        { name: "characters", status: "pending" }
      ]
    });
  });

  it("replaces generic active job progress with latest progress", async () => {
    const { trackedActiveJob } = await import("./main");
    const job = {
      id: "job-1",
      type: "chat_turn",
      status: "running",
      result: null,
      error: null,
      created_at: 1,
      latest_progress: {
        jobs: [
          { name: "state", status: "running" },
          { name: "context", status: "pending" }
        ]
      }
    } satisfies Job;

    expect(trackedActiveJob(job, { job: { ...job, latest_progress: null }, progress: "Running" }).progress).toBe(
      "Post-turn: World state running, Context update pending"
    );
  });

  it("replaces stale active job progress with latest phase progress", async () => {
    const { trackedActiveJob } = await import("./main");
    const job = {
      id: "job-1",
      type: "chat_turn",
      status: "running",
      result: null,
      error: null,
      created_at: 1,
      latest_progress: {
        jobs: [
          { name: "state", status: "complete" },
          { name: "context", status: "running" }
        ]
      }
    } satisfies Job;

    expect(
      trackedActiveJob(job, {
        job: { ...job, latest_progress: { label: "Updating world state" } },
        progress: "Updating world state"
      })
    ).toEqual({
      job,
      progress: "Post-turn: World state complete, Context update running",
      phases: [
        { name: "state", status: "complete" },
        { name: "context", status: "running" }
      ]
    });
  });

  it("tracks every active job from the active jobs endpoint separately", async () => {
    installEventSourceDouble();
    vi.stubGlobal(
      "fetch",
      workbenchFetch(
        [
          { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 },
          { id: "job-2", type: "image_generation", status: "queued", result: null, error: null, created_at: 2 },
          { id: "job-3", type: "image_generation", status: "running", result: null, error: null, created_at: 3 }
        ],
        runtimeModel(),
        [],
        undefined,
        { pending_jobs_display_mode: { setting_key: "pending_jobs_display_mode", selected: "expanded", options: ["compact", "expanded", "expanded_full"] } }
      )
    );
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const trays = await screen.findAllByLabelText("Pending jobs");
    const tray = trays[trays.length - 1];
    expect(within(tray).getByText("3")).toBeInTheDocument();
    expect(within(tray).getByText("Chat turn")).toBeInTheDocument();
    expect(within(tray).getAllByText("Generating image")).toHaveLength(2);
    expect(within(tray).getByText("Queued")).toBeInTheDocument();
    expect(within(tray).getAllByText("Running").length).toBeGreaterThanOrEqual(2);
  });

  it("applies completed chronicle edit job results to the visible runtime", async () => {
    const sources = installEventSourceDouble();
    const initialModel = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "player-1",
            role: "player",
            speaker_name: "Keeper",
            body: "Hold the line.",
            actions: [{ action_id: "edit-and-resubmit-message", label: "Edit this message" }]
          },
          {
            message_id: "narrator-1",
            role: "narrator",
            speaker_name: null,
            body: "The old answer lands.",
            actions: []
          }
        ]
      }
    });
    const editedModel = runtimeModel({
      chronicle: {
        messages: [
          {
            message_id: "player-2",
            role: "player",
            speaker_name: "Keeper",
            body: "Hold the east line.",
            revision_count: 1,
            actions: []
          },
          {
            message_id: "narrator-2",
            role: "narrator",
            speaker_name: null,
            body: "The revised answer lands.",
            actions: []
          }
        ]
      }
    });
    let currentModel = initialModel;
    const baseFetch = workbenchFetch([], currentModel);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path.startsWith("/api/runtime")) {
        return Promise.resolve({ ok: true, json: async () => currentModel });
      }
      if (path === "/api/chat/edit") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "job-chat-edit",
            type: "chat_edit",
            save_id: "save-1",
            status: "queued",
            result: null,
            error: null
          })
        });
      }
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByTitle("Edit this message"));
    const dialog = screen.getByRole("dialog", { name: "Edit message" });
    const editor = within(dialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "Hold the east line.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Resubmit" }));

    await waitFor(() => expect(sources.some((source) => (
      source.url === "/api/jobs/job-chat-edit/events?save_id=save-1"
    ))).toBe(true));
    currentModel = editedModel;
    act(() => {
      for (const source of sources) {
        source.dispatch("done", {
          id: "job-chat-edit",
          type: "chat_edit",
          save_id: "save-1",
          status: "succeeded",
          result: editedModel,
          error: null
        });
      }
    });

    expect(await screen.findByText("The revised answer lands.")).toBeInTheDocument();
    expect(screen.queryByText("The old answer lands.")).not.toBeInTheDocument();
  });

  it("submits Look Around and displays the returned observation", async () => {
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((rawPath: string, init?: RequestInit) => {
      const path = String(rawPath);
      const method = init?.method ?? "GET";
      const ok = (payload: unknown) => Promise.resolve({
        ok: true,
        json: async () => payload
      });
      if (path.startsWith("/api/runtime")) return ok(runtimeModel());
      if (path === "/api/scenarios") return ok({ scenarios: [] });
      if (path === "/api/settings") return ok(modelSettingsPayload());
      if (path.startsWith("/api/jobs?status=active")) return ok({ jobs: [] });
      if (path.startsWith("/api/chat/submission-status")) {
        return ok({
          save_id: "save-1",
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        });
      }
      if (path === "/api/chat/look-around" && method === "POST") {
        return ok({
          id: "job-look",
          type: "look_around",
          save_id: "save-1",
          status: "queued",
          result: null,
          error: null
        });
      }
      return ok({});
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Look around" }));
    await userEvent.type(screen.getByLabelText("Look Around question"), "Inspect the brass lens.");
    await userEvent.click(screen.getByRole("button", { name: "Look" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/look-around", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/chat/look-around");
    expect(JSON.parse(String(call?.[1].body))).toMatchObject({
      save_id: "save-1",
      query: "Inspect the brass lens."
    });
    await waitFor(() => expect(sources.find((source) => source.url === "/api/jobs/job-look/events?save_id=save-1")).toBeTruthy());
    act(() => {
      sources.find((source) => source.url === "/api/jobs/job-look/events?save_id=save-1")?.dispatch("done", {
        id: "job-look",
        type: "look_around",
        save_id: "save-1",
        status: "succeeded",
        result: {
          answer: "The brass lens hides a locked prism.",
          update_counts: { observations: 1, suggestions: 0, memories: 0, context_sources: 0 }
        },
        error: null
      });
    });

    expect(await screen.findByText("The brass lens hides a locked prism.")).toBeInTheDocument();
  });

  it("renders markdown blocks from a Look Around answer", async () => {
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((rawPath: string, init?: RequestInit) => {
      const path = String(rawPath);
      const method = init?.method ?? "GET";
      const ok = (payload: unknown) => Promise.resolve({
        ok: true,
        json: async () => payload
      });
      if (path.startsWith("/api/runtime")) return ok(runtimeModel());
      if (path === "/api/scenarios") return ok({ scenarios: [] });
      if (path === "/api/settings") return ok(modelSettingsPayload());
      if (path.startsWith("/api/jobs?status=active")) return ok({ jobs: [] });
      if (path.startsWith("/api/chat/submission-status")) {
        return ok({
          save_id: "save-1",
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        });
      }
      if (path === "/api/chat/look-around" && method === "POST") {
        return ok({
          id: "job-look-md",
          type: "look_around",
          save_id: "save-1",
          status: "queued",
          result: null,
          error: null
        });
      }
      return ok({});
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Look around" }));
    await userEvent.type(screen.getByLabelText("Look Around question"), "Inspect the brass lens.");
    await userEvent.click(screen.getByRole("button", { name: "Look" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/look-around", expect.anything()));
    await waitFor(() => expect(sources.find((source) => source.url === "/api/jobs/job-look-md/events?save_id=save-1")).toBeTruthy());
    act(() => {
      sources.find((source) => source.url === "/api/jobs/job-look-md/events?save_id=save-1")?.dispatch("done", {
        id: "job-look-md",
        type: "look_around",
        save_id: "save-1",
        status: "succeeded",
        result: {
          answer:
            "The brass lens rests on a velvet pad.\n- faintly etched runes\n- a small keyhole\nUse the **iron key** to open it.",
          query: "Inspect the brass lens.",
          answer_markdown_blocks: [
            {
              kind: "paragraph",
              spans: [{ kind: "text", text: "The brass lens rests on a velvet pad." }]
            },
            {
              kind: "bullet_item",
              list_kind: "bullet",
              marker: "•",
              ordinal: null,
              spans: [{ kind: "text", text: "faintly etched runes" }]
            },
            {
              kind: "bullet_item",
              list_kind: "bullet",
              marker: "•",
              ordinal: null,
              spans: [{ kind: "text", text: "a small keyhole" }]
            },
            {
              kind: "paragraph",
              spans: [
                { kind: "text", text: "Use the " },
                { kind: "strong", text: "iron key" },
                { kind: "text", text: " to open it." }
              ]
            }
          ],
          update_counts: { observations: 1, suggestions: 0, memories: 0, context_sources: 0 }
        },
        error: null
      });
    });

    const body = await screen.findByText("The brass lens rests on a velvet pad.");
    const answerBody = body.closest(".look-around-answer-body");
    expect(answerBody).not.toBeNull();
    expect(answerBody?.querySelectorAll("p")).toHaveLength(4);
    const paragraphs = Array.from(answerBody?.querySelectorAll("p") ?? []);
    expect(paragraphs.map((p) => p.textContent)).toEqual([
      "The brass lens rests on a velvet pad.",
      "- faintly etched runes",
      "- a small keyhole",
      "Use the iron key to open it.",
    ]);
    expect(answerBody?.querySelector("strong")?.textContent).toBe("iron key");
    expect(document.querySelector(".look-around-query")?.textContent).toBe("Inspect the brass lens.");
  });

  it("uses expanded pending job display mode from settings", async () => {
    installEventSourceDouble();
    vi.stubGlobal(
      "fetch",
      workbenchFetch(
        [
          { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 },
          { id: "job-2", type: "image_generation", status: "running", result: null, error: null, created_at: 2 }
        ],
        runtimeModel(),
        [],
        undefined,
        { pending_jobs_display_mode: { setting_key: "pending_jobs_display_mode", selected: "expanded", options: ["compact", "expanded", "expanded_full"] } }
      )
    );
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const trays = await screen.findAllByLabelText("Pending jobs");
    const tray = trays[trays.length - 1];
    expect(within(tray).getByText("2")).toBeInTheDocument();
    expect(within(tray).getByText("Chat turn")).toBeInTheDocument();
    expect(within(tray).getByText("Generating image")).toBeInTheDocument();
    expect(within(tray).queryByText("Active jobs")).not.toBeInTheDocument();
  });

  it("renders expanded desktop pending job phases from active job progress", async () => {
    stubWorkbenchMedia(false);
    installEventSourceDouble();
    vi.stubGlobal(
      "fetch",
      workbenchFetch(
        [
          {
            id: "job-1",
            type: "chat_turn",
            status: "running",
            result: null,
            error: null,
            created_at: 1,
            latest_progress: {
              jobs: [
                { name: "state", status: "complete" },
                { name: "context", status: "running" },
                { name: "characters", status: "pending" }
              ]
            }
          }
        ],
        runtimeModel(),
        [],
        undefined,
        { pending_jobs_display_mode: { setting_key: "pending_jobs_display_mode", selected: "expanded", options: ["compact", "expanded", "expanded_full"] } }
      )
    );
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const trays = await screen.findAllByLabelText("Pending jobs");
    const tray = trays[trays.length - 1];
    expect(within(tray).getByText("World state")).toBeInTheDocument();
    expect(within(tray).getByText("Context update")).toBeInTheDocument();
    expect(within(tray).getByText("Character cleanup")).toBeInTheDocument();
  });

  it("places active job status directly above the composer", async () => {
    installEventSourceDouble();
    vi.stubGlobal(
      "fetch",
      workbenchFetch([
        { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 }
      ])
    );
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const trays = await screen.findAllByLabelText("Pending jobs");
    const composers = screen.getAllByRole("textbox", { name: "Message" });
    const tray = trays[trays.length - 1];
    const composer = composers[composers.length - 1].closest("form");

    expect(composer).not.toBeNull();
    expect(Boolean(tray.compareDocumentPosition(composer as Element) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
  });

  it("shows in-world time and lets the player correct it", async () => {
    installEventSourceDouble();
    let currentModel = runtimeModel({
      world_time: {
        snapshot_id: "scene-1",
        day_index: 5,
        day_label: "friday",
        phase: "evening",
        clock_minutes: 1275,
        period_label: "festival week",
        source_message_id: null,
        confidence: null,
        display: "Friday festival week evening at 21:15; world day index 5"
      }
    });
    const ok = (payload: unknown) => Promise.resolve({
      ok: true,
      json: async () => payload
    });
    const fetchMock = vi.fn().mockImplementation((rawPath: string, init?: RequestInit) => {
      const path = String(rawPath);
      const method = init?.method ?? "GET";
      if (path.startsWith("/api/runtime") && path !== "/api/runtime/world-time") return ok(currentModel);
      if (path === "/api/runtime/world-time" && method === "POST") {
        currentModel = {
          ...currentModel,
          world_time: {
            snapshot_id: "scene-1",
            day_index: 12,
            day_label: "friday",
            phase: "night",
            clock_minutes: 1275,
            period_label: "festival week",
            source_message_id: null,
            confidence: null,
            display: "Friday festival week night at 21:15; world day index 12"
          }
        };
        return ok(currentModel);
      }
      if (path === "/api/scenarios") return ok({ scenarios: [] });
      if (path.startsWith("/api/jobs?status=active")) return ok({ jobs: [] });
      if (path === "/api/settings/shell") return ok({ pending_jobs_display_mode: modelSettingsPayload().pending_jobs_display_mode });
      if (isSettingsReadPath(path)) return ok(modelSettingsPayload());
      if (path.startsWith("/api/chat/submission-status")) {
        return ok({
          save_id: "save-1",
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        });
      }
      return ok({});
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect(
      await screen.findByText("Friday festival week evening at 21:15; world day index 5")
    ).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Correct world time" }));
    await userEvent.clear(screen.getByLabelText("Day label"));
    await userEvent.type(screen.getByLabelText("Day label"), "friday");
    await userEvent.selectOptions(screen.getByLabelText("Phase"), "night");
    await userEvent.clear(screen.getByLabelText("Day index"));
    await userEvent.type(screen.getByLabelText("Day index"), "12");
    await userEvent.click(screen.getByRole("button", { name: "Save time" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/runtime/world-time", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/runtime/world-time");
    expect(JSON.parse(String(call?.[1].body))).toEqual({
      save_id: "save-1",
      day_index: 12,
      day_label: "friday",
      phase: "night"
    });
    expect(
      await screen.findByText("Friday festival week night at 21:15; world day index 12")
    ).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Correct world time" })).not.toBeInTheDocument();
  });

  it("closes world-time and Look Around mutation dialogs when runtime support changes", async () => {
    installEventSourceDouble();
    const supported = runtimeModel({
      world_time: { snapshot_id: "scene-1", day_index: 1, day_label: "monday", phase: "day", clock_minutes: null, period_label: "", source_message_id: null, confidence: null, display: "Monday" }
    });
    const fetchMock = workbenchFetch([], supported, []);
    vi.stubGlobal("fetch", fetchMock);
    const client = new QueryClient();
    const { Workbench, runtimeQueryKey } = await import("./main");
    render(<QueryClientProvider client={client}><Workbench /></QueryClientProvider>);
    const setRuntime = (model: RuntimeModel) => {
      client.setQueryData(runtimeQueryKey(null), model);
      client.setQueryData(runtimeQueryKey("save-1"), model);
    };

    await userEvent.click(await screen.findByRole("button", { name: "Correct world time" }));
    expect(screen.getByRole("dialog", { name: "Correct world time" })).toBeInTheDocument();
    act(() => setRuntime(runtimeModel({
      ...supported,
      saves: [{ save_id: "save-1", title: "Retired", active: true, supported: false, unsupported_reason: "Retired." }]
    })));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Correct world time" })).not.toBeInTheDocument());

    act(() => setRuntime(supported));
    await userEvent.click(await screen.findByRole("button", { name: "Look around" }));
    expect(screen.getByRole("dialog", { name: "Look Around" })).toBeInTheDocument();
    act(() => setRuntime(runtimeModel({ ...supported, composer_enabled: false })));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Look Around" })).not.toBeInTheDocument());

    act(() => setRuntime(supported));
    await userEvent.click(await screen.findByRole("button", { name: "Look around" }));
    act(() => setRuntime(runtimeModel({
      ...supported,
      active_save_id: "save-2",
      saves: [{ save_id: "save-2", title: "Next", active: true, supported: true }]
    })));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Look Around" })).not.toBeInTheDocument());
  });

  it("disables chat submit from the backend submission status while keeping typed-ahead text", async () => {
    installEventSourceDouble();
    const fetchMock = workbenchFetch(
      [{ id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 }],
      runtimeModel(),
      [],
      {
        save_id: "save-1",
        can_submit: false,
        reason: "chat_turn_active",
        blocking_job_id: "job-1",
        blocking_job_status: "running"
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const textarea = await screen.findByRole("textbox", { name: "Message" });
    await userEvent.type(textarea, "Prepare the follow-up bit");

    expect(textarea).toHaveValue("Prepare the follow-up bit");
    expect(screen.getByTitle("Send")).toBeDisabled();
    expect(screen.getByText("Chat is waiting for Chat turn")).toBeInTheDocument();
    expect(screen.getByText("Running")).toBeInTheDocument();

    fireEvent.submit(textarea.closest("form") as HTMLFormElement);

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/chat")).toBe(false);
  });

  it("keeps an unexpectedly active unsupported save read-only", async () => {
    installEventSourceDouble();
    const unsupportedModel = runtimeModel({
      character_texts_enabled: true,
      saves: [{
        save_id: "save-1",
        title: "Retired Chronicle",
        active: true,
        supported: false,
        unsupported_reason: "Single-character Dating Sim is no longer supported."
      }]
    });
    const fetchMock = workbenchFetch([], unsupportedModel, [], {
      save_id: "save-1",
      can_submit: true,
      reason: null,
      blocking_job_id: null,
      blocking_job_status: null
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect((await screen.findAllByText("Single-character Dating Sim is no longer supported.")).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByTitle("Send")).toBeDisabled();
    expect(screen.getByTitle("Timeskip")).toBeDisabled();
    expect(screen.getByTitle("Open phone")).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Generate opening image" })).not.toBeInTheDocument();
  });

  it("shows a retryable composer notice when chat submission status fails and recovers", async () => {
    installEventSourceDouble();
    let statusRequests = 0;
    const ok = (payload: unknown) => Promise.resolve({
      ok: true,
      json: async () => payload
    });
    const failure = (detail: string) => Promise.resolve({
      ok: false,
      status: 503,
      statusText: "Service Unavailable",
      json: async () => ({ detail })
    });
    const fetchMock = vi.fn().mockImplementation((rawPath: string) => {
      const path = String(rawPath);
      if (path.startsWith("/api/runtime")) return ok(runtimeModel());
      if (path === "/api/scenarios") return ok({ scenarios: [] });
      if (path.startsWith("/api/jobs?status=active")) return ok({ jobs: [] });
      if (path === "/api/settings/shell") {
        return ok({
          pending_jobs_display_mode: modelSettingsPayload().pending_jobs_display_mode
        });
      }
      if (isSettingsReadPath(path)) return ok(modelSettingsPayload());
      if (path.startsWith("/api/chat/submission-status")) {
        statusRequests += 1;
        if (statusRequests === 1) return failure("Submission status is temporarily unavailable.");
        return ok({
          save_id: "save-1",
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        });
      }
      if (path === "/api/log/client") return ok({ ok: true });
      return ok({});
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const textarea = await screen.findByRole("textbox", { name: "Message" });
    expect(await screen.findByText(/Chat submission status could not be loaded/i)).toBeInTheDocument();
    await userEvent.type(textarea, "Hold the response until status recovers.");

    expect(textarea).toHaveValue("Hold the response until status recovers.");
    expect(screen.getByTitle("Send")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Look around" })).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: "Retry chat status" }));

    await waitFor(() => {
      expect(screen.queryByText(/Chat submission status could not be loaded/i)).not.toBeInTheDocument();
    });
    expect(statusRequests).toBeGreaterThanOrEqual(2);
    expect(textarea).toHaveValue("Hold the response until status recovers.");
    expect(screen.getByTitle("Send")).toBeEnabled();
    expect(screen.getByRole("button", { name: "Look around" })).toBeEnabled();
  });

  it.each([
    "chat_turn",
    "chat_regenerate",
    "chat_edit",
    "message_edit",
    "narrator_edit"
  ])("keeps chat submit disabled while %s is active even if submission status is stale", async (jobType) => {
    installEventSourceDouble();
    const fetchMock = workbenchFetch(
      [{ id: "job-1", type: jobType, save_id: "save-1", status: "running", result: null, error: null, created_at: 1 }],
      runtimeModel(),
      [],
      {
        save_id: "save-1",
        can_submit: true,
        reason: null,
        blocking_job_id: null,
        blocking_job_status: null
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const textarea = await screen.findByRole("textbox", { name: "Message" });
    await userEvent.type(textarea, "Queue the next move");

    expect(textarea).toHaveValue("Queue the next move");
    expect(screen.getByTitle("Send")).toBeDisabled();

    fireEvent.submit(textarea.closest("form") as HTMLFormElement);

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/chat")).toBe(false);
  });

  it("does not locally disable chat submit for unrelated active save jobs", async () => {
    installEventSourceDouble();
    const fetchMock = workbenchFetch(
      [{ id: "job-1", type: "image_generation", save_id: "save-1", status: "running", result: null, error: null, created_at: 1 }],
      runtimeModel(),
      [],
      {
        save_id: "save-1",
        can_submit: true,
        reason: null,
        blocking_job_id: null,
        blocking_job_status: null
      }
    );
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const textarea = await screen.findByRole("textbox", { name: "Message" });
    await userEvent.type(textarea, "Queue the next move");

    expect(screen.getByTitle("Send")).toBeEnabled();
  });

  // Regression: switching saves must detach old-save jobs so late Save A events
  // never rewrite Save B's UI. This is intentional save-switch behavior.
  it("hides old-save jobs after switching saves and ignores their late runtime events", async () => {
    const sources = installEventSourceDouble();
    let model = runtimeModel({
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: true },
        { save_id: "save-2", title: "Signal Tower", active: false }
      ],
      chronicle: {
        messages: [
          { message_id: "save-1-message", role: "narrator", speaker_name: null, body: "Save A text.", actions: [] }
        ]
      }
    });
    const saveTwo = runtimeModel({
      active_save_id: "save-2",
      active_save_title: "Signal Tower",
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: false },
        { save_id: "save-2", title: "Signal Tower", active: true }
      ],
      chronicle: {
        messages: [
          { message_id: "save-2-message", role: "narrator", speaker_name: null, body: "Save B text.", actions: [] }
        ]
      }
    });
    const activeJobs = [{ id: "job-1", type: "chat_turn", save_id: "save-1", status: "running", result: null, error: null, created_at: 1 } satisfies Job];
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) return model;
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: activeJobs };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return {
          save_id: model.active_save_id,
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        };
        if (path === "/api/saves/save-2/load") {
          model = saveTwo;
          return saveTwo;
        }
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("Save A text.")).toBeInTheDocument());
    expect(await screen.findByLabelText("Pending jobs")).toBeInTheDocument();

    const jobSource = sources.find((source) => source.url.startsWith("/api/jobs/"));
    expect(jobSource).toBeTruthy();
    act(() => {
      jobSource?.dispatch("narrator_draft", {
        message: {
          message_id: "pending-narrator-message",
          role: "narrator",
          speaker_name: "Narrator",
          body: "Save A draft in progress.",
          markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: "Save A draft in progress." }] }],
          actions: []
        }
      });
    });
    expect(await screen.findByText("Save A draft in progress.")).toBeInTheDocument();

    await userEvent.click(await screen.findByRole("button", { name: "Load Signal Tower" }));

    await waitFor(() => expect(screen.getByText("Save B text.")).toBeInTheDocument());
    await waitFor(() => expect(screen.queryByLabelText("Pending jobs")).not.toBeInTheDocument());
    expect(screen.queryByText("Save A draft in progress.")).not.toBeInTheDocument();

    act(() => {
      jobSource?.dispatch("runtime", runtimeModel({
        active_save_id: "save-1",
        active_save_title: "Lantern Keep",
        chronicle: {
          messages: [
            { message_id: "late-save-1", role: "narrator", speaker_name: null, body: "Late Save A result.", actions: [] }
          ]
        }
      }));
      jobSource?.dispatch("done", {
        id: "job-1",
        type: "chat_turn",
        save_id: "save-1",
        status: "succeeded",
        result: runtimeModel({
          active_save_id: "save-1",
          active_save_title: "Lantern Keep",
          chronicle: {
            messages: [
              { message_id: "late-save-1-done", role: "narrator", speaker_name: null, body: "Late Save A done.", actions: [] }
            ]
          }
        }),
        error: null
      });
    });

    expect(screen.getByText("Save B text.")).toBeInTheDocument();
    expect(screen.queryByText("Late Save A result.")).not.toBeInTheDocument();
    expect(screen.queryByText("Late Save A done.")).not.toBeInTheDocument();
  });

  it("hides an optimistic submitted message after switching saves before the chat job resolves", async () => {
    installEventSourceDouble();
    let resolveChat: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const chatResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveChat = resolve;
    });
    let model = runtimeModel({
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: true },
        { save_id: "save-2", title: "Signal Tower", active: false }
      ],
      chronicle: {
        messages: [
          { message_id: "save-1-message", role: "narrator", speaker_name: null, body: "Save A text.", actions: [] }
        ]
      }
    });
    const saveTwo = runtimeModel({
      active_save_id: "save-2",
      active_save_title: "Signal Tower",
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: false },
        { save_id: "save-2", title: "Signal Tower", active: true }
      ],
      chronicle: {
        messages: [
          { message_id: "save-2-message", role: "narrator", speaker_name: null, body: "Save B text.", actions: [] }
        ]
      }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/chat") return await chatResponse.then((response) => response.json());
        if (path.startsWith("/api/runtime")) return model;
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return {
          save_id: model.active_save_id,
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        };
        if (path === "/api/saves/save-2/load") {
          model = saveTwo;
          return saveTwo;
        }
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("Save A text.")).toBeInTheDocument());
    const textarea = screen.getByRole("textbox", { name: "Message" });
    await userEvent.type(textarea, "Save A pending turn.");
    await waitFor(() => expect(screen.getByTitle("Send")).toBeEnabled());
    fireEvent.submit(textarea.closest("form") as HTMLFormElement);
    expect(await screen.findByText("Save A pending turn.")).toBeInTheDocument();

    await userEvent.click(await screen.findByRole("button", { name: "Load Signal Tower" }));

    await waitFor(() => expect(screen.getByText("Save B text.")).toBeInTheDocument());
    expect(screen.queryByText("Save A pending turn.")).not.toBeInTheDocument();

    resolveChat({
      ok: true,
      json: async () => ({ id: "job-save-a", type: "chat_turn", save_id: "save-1", status: "queued", result: null, error: null })
    });

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/chat")).toBe(true));
    expect(screen.getByText("Save B text.")).toBeInTheDocument();
    expect(screen.queryByText("Save A pending turn.")).not.toBeInTheDocument();
  });

  it("ignores stale submit failures after switching saves", async () => {
    installEventSourceDouble();
    let resolveChat: (response: { ok: boolean; status: number; statusText: string; json: () => Promise<unknown> }) => void = () => undefined;
    const chatResponse = new Promise<{ ok: boolean; status: number; statusText: string; json: () => Promise<unknown> }>((resolve) => {
      resolveChat = resolve;
    });
    let model = runtimeModel({
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: true },
        { save_id: "save-2", title: "Signal Tower", active: false }
      ],
      chronicle: {
        messages: [
          { message_id: "save-1-message", role: "narrator", speaker_name: null, body: "Save A text.", actions: [] }
        ]
      }
    });
    const saveTwo = runtimeModel({
      active_save_id: "save-2",
      active_save_title: "Signal Tower",
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: false },
        { save_id: "save-2", title: "Signal Tower", active: true }
      ],
      chronicle: {
        messages: [
          { message_id: "save-2-message", role: "narrator", speaker_name: null, body: "Save B text.", actions: [] }
        ]
      }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      if (path === "/api/chat") return chatResponse;
      return Promise.resolve({
        ok: true,
        json: async () => {
          if (path === "/api/log/client") return { ok: true };
          if (path.startsWith("/api/runtime")) return model;
          if (path === "/api/scenarios") return { scenarios: [] };
          if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
          if (path === "/api/settings") return modelSettingsPayload();
          if (path.startsWith("/api/chat/submission-status")) return {
            save_id: model.active_save_id,
            can_submit: true,
            reason: null,
            blocking_job_id: null,
            blocking_job_status: null
          };
          if (path === "/api/saves/save-2/load") {
            model = saveTwo;
            return saveTwo;
          }
          return {};
        }
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("Save A text.")).toBeInTheDocument());
    const textarea = screen.getByRole("textbox", { name: "Message" });
    await userEvent.type(textarea, "Save A doomed turn.");
    await waitFor(() => expect(screen.getByTitle("Send")).toBeEnabled());
    fireEvent.submit(textarea.closest("form") as HTMLFormElement);

    await userEvent.click(await screen.findByRole("button", { name: "Load Signal Tower" }));
    await waitFor(() => expect(screen.getByText("Save B text.")).toBeInTheDocument());

    resolveChat({
      ok: false,
      status: 500,
      statusText: "Server Error",
      json: async () => ({ detail: "Old save failed." })
    });

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/log/client")).toBe(true));
    expect(screen.queryByText("Old save failed.")).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Message" })).not.toHaveValue("Save A doomed turn.");
    expect(screen.queryByText("Save A doomed turn.")).not.toBeInTheDocument();
  });

  it("stores save selection locally and requests scoped runtime after switching saves", async () => {
    installEventSourceDouble();
    const saveOne = runtimeModel({
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: true },
        { save_id: "save-2", title: "Signal Tower", active: false }
      ],
      chronicle: {
        messages: [
          { message_id: "save-1-message", role: "narrator", speaker_name: null, body: "Save A text.", actions: [] }
        ]
      }
    });
    const saveTwo = runtimeModel({
      active_save_id: "save-2",
      active_save_title: "Signal Tower",
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: false },
        { save_id: "save-2", title: "Signal Tower", active: true }
      ],
      chronicle: {
        messages: [
          { message_id: "save-2-message", role: "narrator", speaker_name: null, body: "Save B text.", actions: [] }
        ]
      }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/runtime/shell?save_id=save-2") return saveTwo;
        if (path.startsWith("/api/runtime")) return saveOne;
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return {
          save_id: saveOne.active_save_id,
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        };
        if (path === "/api/saves/save-2/load") {
          return saveTwo;
        }
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("Save A text.")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Load Signal Tower" }));

    await waitFor(() => expect(screen.getByText("Save B text.")).toBeInTheDocument());
    expect(window.localStorage.getItem("bragi-web:selected-save-id:v1")).toBe("save-2");
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/runtime/shell?save_id=save-2")).toBe(true));
  });

  it("uses user-scoped stored save selections when authenticated", async () => {
    installEventSourceDouble();
    window.localStorage.setItem("bragi-web:selected-save-id:v1", "save-1");
    window.localStorage.setItem("bragi-web:selected-save-id:v1:user-1", "save-2");
    const saveOne = runtimeModel({
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: true },
        { save_id: "save-2", title: "Signal Tower", active: false }
      ],
      chronicle: {
        messages: [
          { message_id: "save-1-message", role: "narrator", speaker_name: null, body: "Save A text.", actions: [] }
        ]
      }
    });
    const saveTwo = runtimeModel({
      active_save_id: "save-2",
      active_save_title: "Signal Tower",
      scenario_title: "Signal Tower",
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: false },
        { save_id: "save-2", title: "Signal Tower", active: true }
      ],
      chronicle: {
        messages: [
          { message_id: "save-2-message", role: "narrator", speaker_name: null, body: "Save B text.", actions: [] }
        ]
      }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/runtime/shell?save_id=save-2") return saveTwo;
        if (path.startsWith("/api/runtime")) return saveOne;
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return {
          save_id: saveTwo.active_save_id,
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        };
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench currentUser={{ id: "user-1", username: "Mira", role: "user", status: "active" }} />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("Save B text.")).toBeInTheDocument());
    expect(window.localStorage.getItem("bragi-web:selected-save-id:v1:user-1")).toBe("save-2");
    expect(window.localStorage.getItem("bragi-web:selected-save-id:v1")).toBeNull();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/runtime/shell?save_id=save-2")).toBe(true);
  });

  it("clears stale stored save selection after scoped runtime returns not found", async () => {
    installEventSourceDouble();
    window.localStorage.setItem("bragi-web:selected-save-id:v1", "save-2");
    const saveOne = runtimeModel({
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: true }
      ],
      chronicle: {
        messages: [
          { message_id: "save-1-message", role: "narrator", speaker_name: null, body: "Save A text.", actions: [] }
        ]
      }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/runtime/shell?save_id=save-2"
        ? {
          ok: false,
          status: 404,
          statusText: "Not Found",
          json: async () => ({ detail: "Unknown save id: save-2" })
        }
        : {
          ok: true,
          json: async () => {
            if (path.startsWith("/api/runtime")) return saveOne;
            if (path === "/api/scenarios") return { scenarios: [] };
            if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
            if (path === "/api/settings") return modelSettingsPayload();
            if (path.startsWith("/api/chat/submission-status")) return {
              save_id: "save-1",
              can_submit: true,
              reason: null,
              blocking_job_id: null,
              blocking_job_status: null
            };
            if (path === "/api/log/client") return { ok: true };
            return {};
          }
        }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("Save A text.")).toBeInTheDocument());
    await waitFor(() => (
      expect(window.localStorage.getItem("bragi-web:selected-save-id:v1")).toBe("save-1")
    ));
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/runtime/shell?save_id=save-2")).toBe(true);
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/runtime/shell")).toBe(true);
  });

  it("keeps the current save selected when scoped load fails", async () => {
    installEventSourceDouble();
    const saveOne = runtimeModel({
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: true },
        { save_id: "save-2", title: "Signal Tower", active: false }
      ],
      chronicle: {
        messages: [
          { message_id: "save-1-message", role: "narrator", speaker_name: null, body: "Save A text.", actions: [] }
        ]
      }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/saves/save-2/load"
        ? {
          ok: false,
          status: 404,
          statusText: "Not Found",
          json: async () => ({ detail: "Unknown save id: save-2" })
        }
        : {
          ok: true,
          json: async () => {
            if (path.startsWith("/api/runtime")) return saveOne;
            if (path === "/api/scenarios") return { scenarios: [] };
            if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
            if (path === "/api/settings") return modelSettingsPayload();
            if (path.startsWith("/api/chat/submission-status")) return {
              save_id: "save-1",
              can_submit: true,
              reason: null,
              blocking_job_id: null,
              blocking_job_status: null
            };
            if (path === "/api/log/client") return { ok: true };
            return {};
          }
        }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("Save A text.")).toBeInTheDocument());
    await userEvent.click(screen.getByRole("button", { name: "Load Signal Tower" }));

    await waitFor(() => expect(screen.getByText("Unknown save id: save-2")).toBeInTheDocument());
    expect(screen.getByText("Save A text.")).toBeInTheDocument();
    expect(window.localStorage.getItem("bragi-web:selected-save-id:v1")).toBe("save-1");
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/runtime/shell?save_id=save-2")).toBe(false);
  });

  it("refetches scoped runtime when the selected save event stream changes", async () => {
    const sources = installEventSourceDouble();
    let runtimeCalls = 0;
    let scenarioCalls = 0;
    let showUpdatedRuntime = false;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) {
          runtimeCalls += 1;
          return showUpdatedRuntime
            ? runtimeModel({
              chronicle: {
                messages: [
                  { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] },
                  { message_id: "m2", role: "narrator", speaker_name: null, body: "The bell answers.", actions: [] }
                ]
              }
            })
            : runtimeModel({
              chronicle: {
                messages: [
                  { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] }
                ]
              }
            });
        }
        if (path === "/api/scenarios") {
          scenarioCalls += 1;
          return { scenarios: [] };
        }
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return {
          save_id: "save-1",
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        };
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("The beacon waits.")).toBeInTheDocument());
    await waitFor(() => expect(sources.some((source) => source.url === "/api/saves/save-1/events")).toBe(true));
    const runtimeCallsBeforeEvent = runtimeCalls;
    const scenarioCallsBeforeEvent = scenarioCalls;

    act(() => {
      showUpdatedRuntime = true;
      sources.find((source) => source.url === "/api/saves/save-1/events")?.dispatch("runtime_changed", {
        event_id: 1,
        save_id: "save-1",
        type: "runtime_changed",
        payload: { reason: "chat" }
      });
    });

    await waitFor(() => expect(runtimeCalls).toBeGreaterThan(runtimeCallsBeforeEvent));
    expect(await screen.findByText("The bell answers.")).toBeInTheDocument();
    expect(scenarioCalls).toBe(scenarioCallsBeforeEvent);
  });

  it("does not refetch runtime immediately after a live runtime job event applies a fresh model", async () => {
    const sources = installEventSourceDouble();
    const initialModel = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] }
        ]
      }
    });
    const updatedModel = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] },
          { message_id: "m2", role: "narrator", speaker_name: null, body: "The bell answers.", actions: [] }
        ]
      }
    });
    let runtimeCalls = 0;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) {
          runtimeCalls += 1;
          return initialModel;
        }
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) {
          return {
            jobs: [
              { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1, save_id: "save-1" }
            ]
          };
        }
        if (path === "/api/settings/shell") return { pending_jobs_display_mode: modelSettingsPayload().pending_jobs_display_mode };
        if (path.startsWith("/api/chat/submission-status")) return {
          save_id: "save-1",
          can_submit: false,
          reason: "chat_turn_active",
          blocking_job_id: "job-1",
          blocking_job_status: "running"
        };
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("The beacon waits.")).toBeInTheDocument());
    await screen.findByLabelText("Pending jobs");
    await waitFor(() => expect(sources.some((source) => source.url.startsWith("/api/jobs/job-1/events"))).toBe(true));
    const runtimeCallsBeforeEvent = runtimeCalls;

    act(() => {
      sources.find((source) => source.url.startsWith("/api/jobs/job-1/events"))?.dispatch("runtime", updatedModel);
    });

    expect(await screen.findByText("The bell answers.")).toBeInTheDocument();
    await act(async () => {
      await Promise.resolve();
    });
    expect(runtimeCalls).toBe(runtimeCallsBeforeEvent);
  });

  it("refreshes only active job and chat status queries for job save events", async () => {
    const sources = installEventSourceDouble();
    let runtimeCalls = 0;
    let activeJobCalls = 0;
    let submissionStatusCalls = 0;
    let worldCalls = 0;
    let characterCalls = 0;
    let historyCalls = 0;
    let scenarioCalls = 0;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) {
          runtimeCalls += 1;
          return runtimeModel();
        }
        if (path === "/api/scenarios") {
          scenarioCalls += 1;
          return { scenarios: [] };
        }
        if (path.startsWith("/api/jobs?status=active")) {
          activeJobCalls += 1;
          return { jobs: [] };
        }
        if (path.startsWith("/api/chat/submission-status")) {
          submissionStatusCalls += 1;
          return {
            save_id: "save-1",
            can_submit: true,
            reason: null,
            blocking_job_id: null,
            blocking_job_status: null
          };
        }
        if (path.startsWith("/api/world-data")) {
          worldCalls += 1;
          return { active_save_id: "save-1", world_state: [] };
        }
        if (path.startsWith("/api/characters")) {
          characterCalls += 1;
          return { active_save_id: "save-1", characters: [] };
        }
        if (path.startsWith("/api/chat-history")) {
          historyCalls += 1;
          return { messages: [], has_more_before: false, oldest_message_id: null, total_message_count: 0, matching_message_count: 0 };
        }
        if (path === "/api/settings/shell") return { pending_jobs_display_mode: modelSettingsPayload().pending_jobs_display_mode };
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(sources.some((source) => source.url === "/api/saves/save-1/events")).toBe(true));
    const countsBeforeEvent = {
      runtimeCalls,
      activeJobCalls,
      submissionStatusCalls,
      worldCalls,
      characterCalls,
      historyCalls,
      scenarioCalls
    };

    act(() => {
      sources.find((source) => source.url === "/api/saves/save-1/events")?.dispatch("job_changed", {
        event_id: 1,
        save_id: "save-1",
        type: "job_changed",
        payload: { job: { id: "job-1", type: "chat_turn", status: "running" } }
      });
    });

    await waitFor(() => expect(activeJobCalls).toBeGreaterThan(countsBeforeEvent.activeJobCalls));
    await waitFor(() => expect(submissionStatusCalls).toBeGreaterThan(countsBeforeEvent.submissionStatusCalls));
    expect(runtimeCalls).toBe(countsBeforeEvent.runtimeCalls);
    expect(worldCalls).toBe(countsBeforeEvent.worldCalls);
    expect(characterCalls).toBe(countsBeforeEvent.characterCalls);
    expect(historyCalls).toBe(countsBeforeEvent.historyCalls);
    expect(scenarioCalls).toBe(countsBeforeEvent.scenarioCalls);
  });

  it("polls the runtime while a chat job is active so missed SSE updates still show the narrator response", async () => {
    installEventSourceDouble();
    const initialModel = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] }
        ]
      }
    });
    const updatedModel = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] },
          { message_id: "m2", role: "player", speaker_name: "Keeper", body: "Light the beacon.", actions: [] },
          { message_id: "m3", role: "narrator", speaker_name: null, body: "The bell answers.", actions: [] }
        ]
      }
    });
    let runtimeCalls = 0;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) {
          runtimeCalls += 1;
          return runtimeCalls === 1 ? initialModel : updatedModel;
        }
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) {
          return {
            jobs: [
              { id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 }
            ]
          };
        }
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("The beacon waits.")).toBeInTheDocument());
    await screen.findByLabelText("Pending jobs");

    await waitFor(() => expect(screen.getByText("The bell answers.")).toBeInTheDocument(), { timeout: 2500 });
  });

  it("uses a mobile app shell without desktop resize handles", async () => {
    stubWorkbenchMedia(true);
    vi.stubGlobal("fetch", workbenchFetch([]));
    const { Workbench } = await import("./main");

    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("navigation", { name: "Mobile navigation" })).toBeInTheDocument();
    expect(container.querySelector(".app-shell")).toHaveClass("mobile-app-shell");
    expect(screen.queryByRole("separator", { name: "Resize left rail" })).not.toBeInTheDocument();
    expect(screen.queryByRole("separator", { name: "Resize right panel" })).not.toBeInTheDocument();
  });

  it("keeps library access available in the stacked tablet layout", async () => {
    stubWorkbenchMedia((query) => query.includes("1050px"));
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }]
    });
    const scenarios: Scenario[] = [
      scenarioFixture({
        scenario_id: "scenario-1",
        title: "Fog Gate",
        premise: "A gate in the fog.",
        save_count: 0
      })
    ];
    vi.stubGlobal("fetch", workbenchFetch([], model, scenarios));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("button", { name: "Open library" })).toBeInTheDocument();
    expect(screen.queryByRole("navigation", { name: "Mobile navigation" })).not.toBeInTheDocument();
    expect(screen.queryByRole("separator", { name: "Resize left rail" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "New scenario" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Open library" }));

    const sheet = screen.getByRole("dialog", { name: "Library" });
    expect(within(sheet).getByRole("button", { name: "New scenario" })).toBeInTheDocument();
    expect(within(sheet).getByRole("button", { name: "Load Lantern Keep" })).toBeInTheDocument();
    await userEvent.click(within(sheet).getByRole("tab", { name: /Scenarios/ }));
    expect(await within(sheet).findByRole("button", { name: "Start Fog Gate" })).toBeInTheDocument();
  });

  it("keeps the mobile composer before the dock in the installed-app shell", async () => {
    stubWorkbenchMedia(true);
    vi.stubGlobal("fetch", workbenchFetch([]));
    const { Workbench } = await import("./main");

    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await screen.findByRole("navigation", { name: "Mobile navigation" });
    const composer = container.querySelector(".composer");
    const dock = container.querySelector(".mobile-dock");

    expect(composer).not.toBeNull();
    expect(dock).not.toBeNull();
    expect(Boolean((composer as Element).compareDocumentPosition(dock as Element) & Node.DOCUMENT_POSITION_FOLLOWING)).toBe(true);
  });

  it("shows an unread phone badge for incoming character texts", async () => {
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const phoneButton = await screen.findByRole("button", {
      name: "Open phone, 1 unread",
    });

    expect(within(phoneButton).getByText("1")).toBeInTheDocument();

    await userEvent.click(phoneButton);

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "Open phone" })).toBeInTheDocument();
    });
  });

  it("does not show an unread phone badge for server-read incoming character texts", async () => {
    const textPayload = characterTextsPayload({
      rowanContactOverrides: {
        latest_message_read_at: "2026-07-01T12:03:00Z"
      }
    });
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("button", { name: "Open phone" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Open phone, 1 unread" })).not.toBeInTheDocument();
  });

  it("persists phone thread read state when the thread is visible", async () => {
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    const fetchMock = workbenchFetch([], model, [], undefined, {}, textPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Open phone, 1 unread" }));

    await waitFor(() => expect(
      fetchMock.mock.calls.some(([path]) => path === "/api/character-texts/threads/thread-rowan/read")
    ).toBe(true));
    const readCalls = fetchMock.mock.calls.filter(
      ([path]) => path === "/api/character-texts/threads/thread-rowan/read",
    );
    expect(readCalls).toHaveLength(1);
    expect(JSON.parse(String(readCalls[0][1]?.body))).toEqual({
      save_id: "save-1",
      through_message_id: "text-2"
    });

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.click(within(phone).getByRole("button", { name: "Open text thread for Rowan" }));

    await waitFor(() => expect(
      fetchMock.mock.calls.filter(
        ([path]) => path === "/api/character-texts/threads/thread-rowan/read",
      )
    ).toHaveLength(1));
  });

  it("does not mark the first mobile phone thread read before it is opened", async () => {
    stubWorkbenchMedia(true);
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });

    const seenStorageKey = "bragi-web:character-text-seen:v1:anonymous:save-1";
    expect(window.localStorage.getItem(seenStorageKey)).toBeNull();

    await userEvent.click(within(phone).getByRole("button", { name: "Open text thread for Rowan" }));

    expect(JSON.parse(
      window.localStorage.getItem(seenStorageKey) ?? "{}",
    )).toEqual({ "thread-rowan": "text-2" });
    expect(within(phone).getByRole("button", { name: "Back to contacts" })).toBeInTheDocument();
  });

  it("renders character text threads as markdown phone conversations", async () => {
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const rowanButton = within(phone).getByRole("button", { name: "Open text thread for Rowan" });
    expect(rowanButton).toHaveClass("selected");
    expect(within(rowanButton).queryByText(/CS major/)).not.toBeInTheDocument();
    expect(within(rowanButton).queryByText(/At the lab/)).not.toBeInTheDocument();
    expect(within(rowanButton).getByText("Absolutely.")).toBeInTheDocument();
    expect(within(rowanButton).getByText("Absolutely.").tagName).toBe("STRONG");
    expect(within(rowanButton).getByText("notes").tagName).toBe("CODE");
    expect(within(phone).getByText("algorithms").tagName).toBe("STRONG");

    expect(within(phone).queryByRole("button", { name: "Open text thread for Maya" })).not.toBeInTheDocument();
  });

  it("renders a bounded window for long character text threads", async () => {
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      value: 360
    });
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      value: 12000
    });
    const textPayload = characterTextsPayload();
    const messages: CharacterTextThread["messages"] = Array.from({ length: 80 }, (_, index) => {
      const ordinal = index + 1;
      const body = `Thread message ${ordinal}`;
      return {
        id: `text-${ordinal}`,
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: index % 2 ? "character" : "player",
        body,
        delivery_status: "sent",
        created_at: `2026-07-01T12:${String(index % 60).padStart(2, "0")}:00Z`,
        markdown_blocks: [
          {
            kind: "paragraph",
            spans: [{ kind: "text", text: body }]
          }
        ]
      };
    });
    textPayload.threads["thread-rowan"].messages = messages;
    textPayload.model.contacts = textPayload.model.contacts.map((contact) => (
      contact.id === "character-rowan"
        ? {
            ...contact,
            latest_message_id: "text-80",
            latest_message_body: "Thread message 80",
            latest_message_markdown_blocks: messages[79].markdown_blocks,
            latest_message_sender: "character",
            latest_message_at: "2026-07-01T12:59:00Z"
          }
        : contact
    ));
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const messagesPane = phone.querySelector(".character-text-messages") as HTMLElement | null;
    expect(messagesPane).not.toBeNull();
    await waitFor(() => expect(within(messagesPane as HTMLElement).getByText("Thread message 80")).toBeInTheDocument());
    expect(messagesPane?.querySelectorAll(".character-text-bubble").length).toBeLessThan(40);
    expect(within(messagesPane as HTMLElement).queryByText("Thread message 20")).not.toBeInTheDocument();
  });

  it("requests a spontaneous character text from the open thread", async () => {
    installEventSourceDouble();
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    const fetchMock = workbenchFetch([], model, [], undefined, {}, {
      ...textPayload,
      spontaneousJob: {
        id: "job-spontaneous-text",
        type: "character_text_spontaneous",
        save_id: "save-1",
        status: "queued",
        result: null,
        error: null
      }
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.click(within(phone).getByRole("button", { name: "Ask Rowan to text you" }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/character-texts/spontaneous",
        expect.anything(),
      );
    });
    expect(JSON.parse(String(
      fetchMock.mock.calls.find(([path]) => path === "/api/character-texts/spontaneous")?.[1]?.body
    ))).toEqual({
      save_id: "save-1",
      character_id: "character-rowan"
    });
  });

  it("renders generated character text attachments and failures", async () => {
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages = textPayload.threads["thread-rowan"].messages.map((message) => (
      message.id === "text-2"
        ? {
            ...message,
            attachments: [
              {
                id: "attachment-ticket",
                kind: "object_context_image",
                status: "succeeded",
                media_asset_id: "media-ticket",
                mime_type: "image/png",
                provider: "local",
                model: "upload",
                prompt_preview: "creased arcade ticket stub",
                error: null,
                created_at: "2026-07-01T12:03:00Z"
              },
              {
                id: "attachment-failed",
                kind: "character_image",
                status: "failed",
                media_asset_id: null,
                mime_type: null,
                prompt_preview: "Rowan selfie",
                error: "Image provider failed",
                created_at: "2026-07-01T12:04:00Z"
              }
            ]
          }
        : message
    ));
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const attachmentButton = within(phone).getByRole("button", {
      name: "Open text attachment"
    });
    expect(within(attachmentButton).getByRole("img", {
      name: "creased arcade ticket stub"
    })).toHaveAttribute("src", "/api/media/media-ticket/thumbnail?save_id=save-1");
    expect(within(phone).getByRole("status")).toHaveTextContent(
      "Image provider failed"
    );

    await userEvent.click(attachmentButton);

    const preview = await screen.findByRole("dialog", { name: "text attachment" });
    expect(within(preview).getByRole("img", {
      name: "creased arcade ticket stub"
    })).toHaveAttribute("src", "/api/media/media-ticket?save_id=save-1");
    expect(
      within(preview).queryByRole("button", { name: "Regenerate with edits" })
    ).not.toBeInTheDocument();
  });

  it("regenerates character text attachments from an edited raw prompt", async () => {
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages = textPayload.threads["thread-rowan"].messages.map((message) => (
      message.id === "text-2"
        ? {
            ...message,
            attachments: [
              {
                id: "attachment-selfie",
                kind: "character_image",
                status: "succeeded",
                media_asset_id: "media-selfie",
                mime_type: "image/png",
                provider: "fake",
                model: "fake-image",
                prompt_preview: "Rowan selfie",
                error: null,
                created_at: "2026-07-01T12:03:00Z"
              }
            ]
          }
        : message
    ));
    const job: Job = {
      id: "job-regenerate-text-image",
      type: "image_generation",
      status: "queued",
      result: null,
      error: null,
      created_at: 1,
      save_id: "save-1"
    };
    const baseFetch = workbenchFetch([], runtimeModel({ character_texts_enabled: true }), [], undefined, {}, textPayload);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/media/media-selfie/prompt?save_id=save-1") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            media_asset_id: "media-selfie",
            prompt: "Rowan grinning under neon"
          })
        });
      }
      if (path === "/api/media/media-selfie/regenerate") {
        return Promise.resolve({
          ok: true,
          json: async () => job
        });
      }
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { CharacterTextPhone } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharacterTextPhone
          activeSaveId="save-1"
          disabled={false}
          runJob={runJob}
          seenTextMessageIdsByThread={{}}
          onThreadSeen={vi.fn()}
          onClose={vi.fn()}
        />
      </QueryClientProvider>
    );

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.click(await within(phone).findByRole("button", {
      name: "Open text image"
    }));
    const preview = await screen.findByRole("dialog", { name: "text image" });
    await userEvent.click(within(preview).getByRole("button", { name: "Regenerate with edits" }));
    const promptDialog = await screen.findByRole("dialog", { name: "Regenerate with edits" });
    const promptField = await within(promptDialog).findByLabelText("Image prompt");
    expect(promptField).toHaveValue("Rowan grinning under neon");
    await userEvent.clear(promptField);
    await userEvent.type(promptField, "Rowan grinning under red arcade neon");
    await userEvent.click(within(promptDialog).getByRole("button", { name: "Regenerate" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/media/media-selfie/regenerate",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          save_id: "save-1",
          prompt: "Rowan grinning under red arcade neon"
        })
      })
    ));
    expect(runJob).toHaveBeenCalledWith(job);
  });

  it("renders no secondary text in a phone contact row that has no messages", async () => {
    const textPayload = characterTextsPayload();
    const quietContact: CharacterTextContact = {
      id: "character-quiet",
      name: "Quiet",
      contact_name: "",
      role: "neighbor",
      status: "Away",
      is_player_character: false,
      player_has_character_number: true,
      character_has_player_number: true,
      thread_id: "thread-quiet",
      latest_message_id: null,
      latest_message_body: "",
      latest_message_markdown_blocks: [],
      latest_message_sender: null,
      latest_message_at: null,
      reference_image: null
    };
    textPayload.model.contacts = [...textPayload.model.contacts, quietContact];
    textPayload.model.repair_contacts = [
      ...textPayload.model.repair_contacts,
      quietContact
    ];
    textPayload.threads["thread-quiet"] = {
      id: "thread-quiet",
      character_id: "character-quiet",
      title: "Quiet",
      status: "active",
      created_at: "2026-07-01T12:00:00Z",
      updated_at: "2026-07-01T12:00:00Z",
      messages: []
    };
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });

    const rowanButton = within(phone).getByRole("button", { name: "Open text thread for Rowan" });
    expect(rowanButton.querySelector(".character-text-contact-preview")).not.toBeNull();
    expect(within(rowanButton).queryByText(/CS major/)).not.toBeInTheDocument();
    expect(within(rowanButton).queryByText(/At the lab/)).not.toBeInTheDocument();

    const quietButton = within(phone).getByRole("button", { name: "Open text thread for Quiet" });
    expect(quietButton.querySelector(".character-text-contact-preview")).toBeNull();
    expect(within(quietButton).queryByText(/neighbor/)).not.toBeInTheDocument();
    expect(within(quietButton).queryByText(/Away/)).not.toBeInTheDocument();
    expect(within(quietButton).queryByText(/Contact/)).not.toBeInTheDocument();
  });

  it("shows only the contact name in the selected phone thread header", async () => {
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const threadHeader = phone.querySelector(".character-text-thread-header");
    expect(threadHeader).not.toBeNull();

    expect(within(threadHeader as HTMLElement).getByText("Rowan")).toBeInTheDocument();
    expect(within(threadHeader as HTMLElement).queryByText(/At the lab/)).not.toBeInTheDocument();
    expect(within(threadHeader as HTMLElement).queryByText(/CS major/)).not.toBeInTheDocument();
    expect(within(threadHeader as HTMLElement).queryByText(/Messages/)).not.toBeInTheDocument();
  });

  it("hides character role and status from the Add-contact dialog rows", async () => {
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.click(within(phone).getByRole("button", { name: "Add contact" }));
    const addContact = await screen.findByRole("dialog", { name: "Add contact" });

    expect(within(addContact).getByText("Maya")).toBeInTheDocument();
    expect(within(addContact).queryByText(/Club president/)).not.toBeInTheDocument();
    expect(within(addContact).queryByText(/Busy/)).not.toBeInTheDocument();
    expect(within(addContact).queryByText(/Contact/)).not.toBeInTheDocument();
  });

  it("updates the phone contact preview when the runtime reports that texts changed", async () => {
    const sources = installEventSourceDouble();
    const textPayload = characterTextsPayload();
    const updatedModel: CharacterTextsModel = {
      ...textPayload.model,
      contacts: textPayload.model.contacts.map((contact) => (
        contact.id === "character-rowan"
          ? {
              ...contact,
              latest_message_id: "text-player-1",
              latest_message_body: "Heading to the arcade after class.",
              latest_message_markdown_blocks: [
                {
                  kind: "paragraph",
                  spans: [{ kind: "text", text: "Heading to the arcade after class." }]
                }
              ],
              latest_message_sender: "player",
              latest_message_at: "2026-07-01T12:03:00Z"
            }
          : contact
      ))
    };
    let showUpdatedModel = false;
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      if (path.startsWith("/api/runtime")) return Promise.resolve({
        ok: true,
        json: async () => runtimeModel({ character_texts_enabled: true })
      });
      if (path.startsWith("/api/character-texts/threads/")) return Promise.resolve({
        ok: true,
        json: async () => textPayload.threads["thread-rowan"]
      });
      if (path.startsWith("/api/character-texts")) {
        return Promise.resolve({
          ok: true,
          json: async () => (showUpdatedModel ? updatedModel : textPayload.model)
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock as unknown as typeof fetch);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(
      sources.some((source) => source.url === "/api/saves/save-1/events")
    ).toBe(true));

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const rowanButton = within(phone).getByRole("button", { name: "Open text thread for Rowan" });
    expect(within(rowanButton).getByText("Absolutely.")).toBeInTheDocument();

    act(() => {
      showUpdatedModel = true;
      sources
        .find((source) => source.url === "/api/saves/save-1/events")
        ?.dispatch("character_texts_changed", {
          event_id: 1,
          save_id: "save-1",
          type: "character_texts_changed"
        });
    });

    await waitFor(() => expect(
      rowanButton.querySelector(".character-text-contact-preview")?.textContent
    ).toMatch(/Heading to the arcade after class/));
    expect(within(rowanButton).queryByText(/Absolutely\./)).not.toBeInTheDocument();
  });

  it("shows the reference photo as avatar and falls back to initials", async () => {
    const textPayload = characterTextsPayload({ rowanReferenceAssetId: "asset-rowan-1" });
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const rowanButton = within(phone).getByRole("button", { name: "Open text thread for Rowan" });
    const rowanAvatar = rowanButton.querySelector(".character-text-avatar img.character-text-avatar-image");
    expect(rowanAvatar).not.toBeNull();
    expect(rowanAvatar?.getAttribute("src")).toBe("/api/media/asset-rowan-1/thumbnail?save_id=save-1");
    expect(rowanAvatar?.getAttribute("alt")).toBe("");

    await userEvent.click(within(phone).getByRole("button", { name: "Add contact" }));
    const addContact = await screen.findByRole("dialog", { name: "Add contact" });
    const mayaRow = within(addContact).getByText("Maya").closest(".character-text-repair-row");
    expect(mayaRow).not.toBeNull();
    expect(mayaRow?.querySelector(".character-text-avatar img.character-text-avatar-image")).toBeNull();
    expect(within(mayaRow as HTMLElement).getByText("M")).toBeInTheDocument();

    const threadHeader = phone.querySelector(".character-text-thread-header");
    expect(threadHeader).not.toBeNull();
    const threadHeaderAvatar = threadHeader?.querySelector(".character-text-avatar.large img.character-text-avatar-image");
    expect(threadHeaderAvatar).not.toBeNull();
    expect(threadHeaderAvatar?.getAttribute("src")).toBe("/api/media/asset-rowan-1/thumbnail?save_id=save-1");
  });

  it("falls back to initials when the reference photo fails to load", async () => {
    const textPayload = characterTextsPayload({ rowanReferenceAssetId: "asset-rowan-broken" });
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const rowanButton = within(phone).getByRole("button", { name: "Open text thread for Rowan" });
    const rowanImage = rowanButton.querySelector(".character-text-avatar img.character-text-avatar-image");
    expect(rowanImage).not.toBeNull();
    fireEvent.error(rowanImage as HTMLElement);

    await waitFor(() => {
      expect(rowanButton.querySelector(".character-text-avatar img")).toBeNull();
      expect(within(rowanButton).getByText("R")).toBeInTheDocument();
    });

    const threadHeader = phone.querySelector(".character-text-thread-header");
    const headerImage = threadHeader?.querySelector(".character-text-avatar.large img.character-text-avatar-image");
    expect(headerImage).not.toBeNull();
    fireEvent.error(headerImage as HTMLElement);
    await waitFor(() => {
      expect(threadHeader?.querySelector(".character-text-avatar.large img")).toBeNull();
      expect(within(threadHeader as HTMLElement).getByText("R")).toBeInTheDocument();
    });
  });

  it("keeps the phone composer available while chat submission is blocked", async () => {
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    const blockedStatus: ChatSubmissionStatus = {
      save_id: "save-1",
      can_submit: false,
      reason: "chat_turn_active",
      blocking_job_id: "job-1",
      blocking_job_status: "running"
    };
    vi.stubGlobal("fetch", workbenchFetch([], model, [], blockedStatus, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    expect(within(phone).queryByText("Reply pending")).not.toBeInTheDocument();
    const composer = within(phone).getByRole("textbox", { name: "Message Rowan" });
    expect(composer).not.toBeDisabled();

    await userEvent.type(composer, "Still free?");

    expect(within(phone).getByRole("button", { name: "Send text" })).toBeEnabled();
  });

  it("renders per-message character text delivery states", async () => {
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages = [
      {
        ...textPayload.threads["thread-rowan"].messages[0],
        sender: "character",
        body: "",
        markdown_blocks: [],
        delivery_status: "pending"
      },
      textPayload.threads["thread-rowan"].messages[1],
      {
        id: "text-3",
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: "player",
        body: "Did this go through?",
        delivery_status: "failed",
        delivery_error: "Provider request failed"
      }
    ];
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    expect(within(phone).queryByText("Pending")).not.toBeInTheDocument();
    expect(within(phone).getByText("Failed")).toBeInTheDocument();
    expect(within(phone).getByText("Provider request failed")).toBeInTheDocument();
    expect(within(phone).getByText("Rowan is typing...")).toBeInTheDocument();
    expect(within(phone).getByRole("textbox", { name: "Message Rowan" })).toBeDisabled();
    expect(within(phone).getByRole("button", { name: "Ask Rowan to text you" })).toBeDisabled();
  });

  it("renders failed empty character text attempts as explicit errors", async () => {
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages = [
      {
        ...textPayload.threads["thread-rowan"].messages[0],
        id: "text-failed-character",
        sender: "character",
        body: "",
        markdown_blocks: [],
        delivery_status: "failed",
        delivery_error: "Text provider returned an empty reply"
      }
    ];
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    expect(within(phone).getByText("Failed")).toBeInTheDocument();
    expect(within(phone).getByText("Text provider returned an empty reply")).toBeInTheDocument();
    expect(within(phone).queryByText("Rowan is typing...")).not.toBeInTheDocument();
  });

  it("renders phone message metadata and sends multiline drafts", async () => {
    installEventSourceDouble();
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages = [
      {
        id: "text-1",
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: "player",
        body: "Can we talk after class?",
        delivery_status: "sent",
        created_at: "2026-07-01T12:00:00Z",
        in_world_sent_at: "Friday evening after class",
        delivered_at: "2026-07-01T12:00:04Z",
        read_at: "2026-07-01T12:01:00Z",
        markdown_blocks: [
          {
            kind: "paragraph",
            spans: [{ kind: "text", text: "Can we talk after class?" }]
          }
        ]
      },
      {
        id: "text-2",
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: "character",
        body: "Meet me by the arcade.",
        delivery_status: "sent",
        created_at: "2026-07-01T12:02:00Z",
        in_world_sent_at: "Friday evening after class",
        delivered_at: "2026-07-01T12:02:01Z",
        reply_to_message_id: "text-1",
        proactive_reason: "Follow up on Rowan's route plan.",
        proactive_trigger_type: "dating_route",
        markdown_blocks: [
          {
            kind: "paragraph",
            spans: [{ kind: "text", text: "Meet me by the arcade." }]
          }
        ]
      },
      {
        id: "text-3",
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: "character",
        body: "Bring the brass token.",
        delivery_status: "sent",
        created_at: "2026-07-01T12:03:00Z",
        markdown_blocks: [
          {
            kind: "paragraph",
            spans: [{ kind: "text", text: "Bring the brass token." }]
          }
        ]
      }
    ];
    const model = runtimeModel({ character_texts_enabled: true });
    const fetchMock = workbenchFetch([], model, [], undefined, {}, {
      ...textPayload,
      sendJob: {
        id: "job-text-send",
        type: "character_text_send",
        save_id: "save-1",
        status: "queued",
        result: null,
        error: null
      }
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });

    expect(within(phone).getByText("Jul 1, 2026")).toBeInTheDocument();
    expect(within(phone).getAllByText("Friday evening after class").length).toBeGreaterThan(0);
    expect(within(phone).getByText("Read")).toBeInTheDocument();
    expect(within(phone).getByRole("button", { name: "Show why this text arrived" })).toBeInTheDocument();

    await userEvent.click(within(phone).getByRole("button", { name: "Show why this text arrived" }));
    expect(within(phone).getByText("Follow up on Rowan's route plan.")).toBeInTheDocument();
    const groupedBubble = within(phone).getByText("Bring the brass token.").closest(".character-text-bubble");
    expect(groupedBubble).toHaveClass("grouped");

    const composer = within(phone).getByRole("textbox", { name: "Message Rowan" });
    expect(composer.tagName).toBe("TEXTAREA");
    const photo = new File(["fake-image"], "gate-key.png", { type: "image/png" });
    await userEvent.upload(within(phone).getByLabelText("Photo"), photo);
    fireEvent.change(composer, { target: { value: "Line one\nLine two" } });
    await userEvent.click(within(phone).getByRole("button", { name: "Send text" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/character-texts/send-image",
      expect.anything(),
    ));
    const sendCall = fetchMock.mock.calls.find(([path]) => path === "/api/character-texts/send-image");
    expect(formDataTextEntries(sendCall?.[1]?.body)).toEqual({
      save_id: "save-1",
      character_id: "character-rowan",
      body: "Line one\nLine two",
      file: "gate-key.png"
    });
    const uploadedPhoto = formDataValue(sendCall?.[1]?.body, "file");
    expect(uploadedPhoto).toBeInstanceOf(File);
    expect((uploadedPhoto as File).name).toBe(photo.name);
    expect((uploadedPhoto as File).type).toBe(photo.type);
  });

  it("sends selected photos with group character texts", async () => {
    const textPayload = characterTextsPayload();
    const groupThread: CharacterTextThread = {
      id: "thread-arcade",
      character_id: null,
      title: "Arcade Crew",
      kind: "group",
      status: "active",
      participants: [
        { character_id: "character-rowan", name: "Rowan", ordinal: 0 },
        { character_id: "character-maya", name: "Maya", ordinal: 1 }
      ],
      created_at: "2026-07-01T12:00:00Z",
      updated_at: "2026-07-01T12:00:00Z",
      messages: []
    };
    textPayload.model.threads = [groupThread];
    textPayload.threads[groupThread.id] = groupThread;
    const model = runtimeModel({ character_texts_enabled: true });
    const fetchMock = workbenchFetch([], model, [], undefined, {}, {
      ...textPayload,
      sendJob: {
        id: "job-group-text-send",
        type: "character_text_send",
        save_id: "save-1",
        status: "queued",
        result: null,
        error: null
      }
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.click(within(phone).getByRole("button", { name: "Open text thread for Arcade Crew" }));
    const photo = new File(["group-image"], "arcade-map.png", { type: "image/png" });
    await userEvent.upload(within(phone).getByLabelText("Photo"), photo);
    fireEvent.change(within(phone).getByRole("textbox", { name: "Message Arcade Crew" }), {
      target: { value: "Can everyone see this?" }
    });
    await userEvent.click(within(phone).getByRole("button", { name: "Send text" }));

    const path = "/api/character-texts/threads/thread-arcade/send-image";
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(path, expect.anything()));
    const sendCall = fetchMock.mock.calls.find(([calledPath]) => calledPath === path);
    expect(formDataTextEntries(sendCall?.[1]?.body)).toEqual({
      save_id: "save-1",
      body: "Can everyone see this?",
      file: "arcade-map.png"
    });
    const uploadedPhoto = formDataValue(sendCall?.[1]?.body, "file");
    expect(uploadedPhoto).toBeInstanceOf(File);
    expect((uploadedPhoto as File).name).toBe(photo.name);
    expect((uploadedPhoto as File).type).toBe(photo.type);
    expect(formDataValue(sendCall?.[1]?.body, "character_id")).toBeNull();
  });

  it("hides the text photo picker for child users", async () => {
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench currentUser={{ id: "child-1", username: "Ilyra", role: "child", status: "active" }} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    expect(within(phone).queryByLabelText("Photo")).not.toBeInTheDocument();
    const composer = within(phone).getByRole("textbox", { name: "Message Rowan" }).closest("form");
    expect(composer).toHaveClass("character-text-compose");
    expect(composer?.querySelector("input[type='file']")).toBeNull();
    expect(composer?.childElementCount).toBe(2);
  });

  it("shows player-sent character texts immediately while send acceptance is pending", async () => {
    installEventSourceDouble();
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    let resolveSend: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const sendResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveSend = resolve;
    });
    const baseFetch = workbenchFetch([], model, [], undefined, {}, textPayload);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/character-texts/send-image") return sendResponse;
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const composer = within(phone).getByRole("textbox", { name: "Message Rowan" }) as HTMLInputElement;
    await userEvent.type(composer, "Still free?");
    await userEvent.click(within(phone).getByRole("button", { name: "Send text" }));

    expect(await within(phone).findByText("Still free?")).toBeInTheDocument();
    expect(within(phone).getByText("Pending")).toBeInTheDocument();
    expect(composer.value).toBe("");
    expect(within(phone).getByRole("button", { name: "Send text" })).toBeDisabled();
    expect(formDataTextEntries(fetchMock.mock.calls.find(([path]) => path === "/api/character-texts/send-image")?.[1]?.body)).toEqual({
      save_id: "save-1",
      character_id: "character-rowan",
      body: "Still free?"
    });

    resolveSend({
      ok: true,
      json: async () => ({
        id: "job-text-send",
        type: "character_text_send",
        save_id: "save-1",
        status: "queued",
        result: null,
        error: null
      })
    });
  });

  it("replaces the optimistic character text with the refreshed server message", async () => {
    const sources = installEventSourceDouble();
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    let resolveSend: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const sendResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveSend = resolve;
    });
    const baseFetch = workbenchFetch([], model, [], undefined, {}, textPayload);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/character-texts/send-image") return sendResponse;
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.type(within(phone).getByRole("textbox", { name: "Message Rowan" }), "See you soon.");
    await userEvent.click(within(phone).getByRole("button", { name: "Send text" }));

    expect(await within(phone).findByText("See you soon.")).toBeInTheDocument();
    expect(within(phone).getByText("Pending")).toBeInTheDocument();

    textPayload.threads["thread-rowan"].messages = [
      ...textPayload.threads["thread-rowan"].messages,
      {
        id: "text-3",
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: "player",
        body: "See you soon.",
        delivery_status: "pending",
        delivery_job_id: "job-text-send"
      }
    ];
    resolveSend({
      ok: true,
      json: async () => ({
        id: "job-text-send",
        type: "character_text_send",
        save_id: "save-1",
        status: "queued",
        result: null,
        error: null
      })
    });

    await waitFor(() => {
      expect(within(phone).getAllByText("See you soon.")).toHaveLength(1);
      expect(within(phone).getByText("Pending")).toBeInTheDocument();
    });
    await waitFor(() => expect(sources.some((source) => (
      source.url === "/api/jobs/job-text-send/events?save_id=save-1"
    ))).toBe(true));

    textPayload.threads["thread-rowan"].messages = textPayload.threads["thread-rowan"].messages.map((message) => (
      message.id === "text-3"
        ? { ...message, delivery_status: "sent" }
        : message
    ));
    act(() => {
      for (const source of sources) {
        source.dispatch("done", {
          id: "job-text-send",
          type: "character_text_send",
          save_id: "save-1",
          status: "succeeded",
          result: null,
          error: null
        });
      }
    });

    await waitFor(() => {
      expect(within(phone).getAllByText("See you soon.")).toHaveLength(1);
      expect(within(phone).queryByText("Pending")).not.toBeInTheDocument();
    });
  });

  it("does not clear a pending chronicle message when a text send job finishes", async () => {
    const sources = installEventSourceDouble();
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    let resolveChat: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const chatResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveChat = resolve;
    });
    const baseFetch = workbenchFetch([], model, [], undefined, {}, textPayload);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/chat") return chatResponse;
      if (path === "/api/character-texts/send-image") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "job-text-send",
            type: "character_text_send",
            save_id: "save-1",
            status: "queued",
            result: null,
            error: null
          })
        });
      }
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const chatComposer = await screen.findByRole("textbox", { name: "Message" });
    await userEvent.type(chatComposer, "A chat turn is still pending.");
    await userEvent.click(screen.getByTitle("Send"));
    expect(await screen.findByText("A chat turn is still pending.")).toBeInTheDocument();

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.type(within(phone).getByRole("textbox", { name: "Message Rowan" }), "Text while chat runs.");
    await userEvent.click(within(phone).getByRole("button", { name: "Send text" }));

    await waitFor(() => expect(sources.some((source) => (
      source.url === "/api/jobs/job-text-send/events?save_id=save-1"
    ))).toBe(true));
    act(() => {
      for (const source of sources) {
        source.dispatch("done", {
          id: "job-text-send",
          type: "character_text_send",
          save_id: "save-1",
          status: "succeeded",
          result: null,
          error: null
        });
      }
    });

    expect(screen.getByText("A chat turn is still pending.")).toBeInTheDocument();
    resolveChat({
      ok: true,
      json: async () => ({
        id: "job-chat",
        type: "chat_turn",
        save_id: "save-1",
        status: "queued",
        result: null,
        error: null
      })
    });
  });

  it("confirms character text delete-from-here actions before starting the delete job", async () => {
    const sources = installEventSourceDouble();
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages[0] = {
      ...textPayload.threads["thread-rowan"].messages[0],
      actions: [{ action_id: "delete-text-messages-from-here", label: "Delete from here" }]
    };
    const model = runtimeModel({ character_texts_enabled: true });
    const deleteJob = {
      id: "job-text-delete",
      type: "character_text_delete",
      save_id: "save-1",
      status: "queued",
      result: null,
      error: null
    };
    const baseFetch = workbenchFetch([], model, [], undefined, {}, textPayload);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/character-texts/delete-from-here") {
        return Promise.resolve({
          ok: true,
          json: async () => deleteJob
        });
      }
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const deleteFromHereAction = within(phone).getByRole("button", { name: "Delete from here" });
    expect(deleteFromHereAction).toHaveClass("touch-labeled-action");
    expect(within(deleteFromHereAction).getByText("Delete from here")).toHaveClass("touch-action-label");
    await userEvent.click(deleteFromHereAction);
    expect(screen.getByRole("dialog", { name: "Delete from here?" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/character-texts/delete-from-here")).toBe(false);

    await userEvent.click(within(phone).getByRole("button", { name: "Delete from here" }));
    const dialog = screen.getByRole("dialog", { name: "Delete from here?" });
    await userEvent.click(within(dialog).getByRole("button", { name: "Delete from here" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/character-texts/delete-from-here", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/character-texts/delete-from-here");
    expect(JSON.parse(String(call?.[1]?.body))).toEqual({
      save_id: "save-1",
      text_message_id: "text-1"
    });
    await waitFor(() => expect(sources.some((source) => (
      source.url === "/api/jobs/job-text-delete/events?save_id=save-1"
    ))).toBe(true));
  });

  it("does not clear a pending chronicle message when a text edit job finishes", async () => {
    const sources = installEventSourceDouble();
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages[0] = {
      ...textPayload.threads["thread-rowan"].messages[0],
      actions: [{ action_id: "edit-and-resubmit-text-message", label: "Edit text" }]
    };
    const model = runtimeModel({ character_texts_enabled: true });
    let resolveChat: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const chatResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveChat = resolve;
    });
    const baseFetch = workbenchFetch([], model, [], undefined, {}, textPayload);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/chat") return chatResponse;
      if (path === "/api/character-texts/edit") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "job-text-edit",
            type: "character_text_edit",
            save_id: "save-1",
            status: "queued",
            result: null,
            error: null
          })
        });
      }
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const chatComposer = await screen.findByRole("textbox", { name: "Message" });
    await userEvent.type(chatComposer, "A chat turn remains pending through text edit.");
    await userEvent.click(screen.getByTitle("Send"));
    expect(await screen.findByText("A chat turn remains pending through text edit.")).toBeInTheDocument();

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.click(within(phone).getByRole("button", { name: "Edit text" }));
    const editDialog = await screen.findByRole("dialog", { name: "Edit text" });
    const editor = within(editDialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "Can we talk after lab?");
    await userEvent.click(within(editDialog).getByRole("button", { name: "Resubmit" }));

    await waitFor(() => expect(sources.some((source) => (
      source.url === "/api/jobs/job-text-edit/events?save_id=save-1"
    ))).toBe(true));
    const updatedThread = {
      ...textPayload.threads["thread-rowan"],
      messages: [
        {
          ...textPayload.threads["thread-rowan"].messages[0],
          body: "Can we talk after lab?",
          markdown_blocks: [
            {
              kind: "paragraph",
              spans: [{ kind: "text", text: "Can we talk after lab?" }]
            }
          ]
        },
        {
          ...textPayload.threads["thread-rowan"].messages[1],
          id: "text-3",
          body: "Meet me by the lab doors.",
          markdown_blocks: [
            {
              kind: "paragraph",
              spans: [{ kind: "text", text: "Meet me by the lab doors." }]
            }
          ]
        }
      ]
    };
    textPayload.threads["thread-rowan"] = updatedThread;
    textPayload.model = {
      ...textPayload.model,
      contacts: textPayload.model.contacts.map((contact) => (
        contact.thread_id === "thread-rowan"
          ? {
              ...contact,
              latest_message_id: "text-3",
              latest_message_body: "Meet me by the lab doors.",
              latest_message_markdown_blocks: updatedThread.messages[1].markdown_blocks,
              latest_message_sender: "character"
            }
          : contact
      )),
      repair_contacts: textPayload.model.repair_contacts.map((contact) => (
        contact.thread_id === "thread-rowan"
          ? {
              ...contact,
              latest_message_id: "text-3",
              latest_message_body: "Meet me by the lab doors.",
              latest_message_markdown_blocks: updatedThread.messages[1].markdown_blocks,
              latest_message_sender: "character"
            }
          : contact
      ))
    };
    act(() => {
      for (const source of sources) {
        source.dispatch("done", {
          id: "job-text-edit",
          type: "character_text_edit",
          save_id: "save-1",
          status: "succeeded",
          result: {
            save_id: "save-1",
            thread: updatedThread,
            player_message: updatedThread.messages[0],
            reply: updatedThread.messages[1],
            revision: { id: "revision-1" }
          },
          error: null
        });
      }
    });

    expect(screen.getByText("A chat turn remains pending through text edit.")).toBeInTheDocument();
    await waitFor(() => {
      expect(within(phone).getAllByText("Meet me by the lab doors.").length).toBeGreaterThan(0);
    });
    resolveChat({
      ok: true,
      json: async () => ({
        id: "job-chat",
        type: "chat_turn",
        save_id: "save-1",
        status: "queued",
        result: null,
        error: null
      })
    });
  });

  it("disables unchanged character text resubmit", async () => {
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages[0] = {
      ...textPayload.threads["thread-rowan"].messages[0],
      actions: [{ action_id: "edit-and-resubmit-text-message", label: "Edit text" }]
    };
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.click(within(phone).getByRole("button", { name: "Edit text" }));
    const editDialog = await screen.findByRole("dialog", { name: "Edit text" });

    expect(within(editDialog).getByRole("button", { name: "Resubmit" })).toBeDisabled();
  });

  it("shows failed character text resubmit jobs in the phone UI", async () => {
    const sources = installEventSourceDouble();
    const textPayload = characterTextsPayload();
    textPayload.threads["thread-rowan"].messages[0] = {
      ...textPayload.threads["thread-rowan"].messages[0],
      actions: [{ action_id: "edit-and-resubmit-text-message", label: "Edit text" }]
    };
    const model = runtimeModel({ character_texts_enabled: true });
    const baseFetch = workbenchFetch([], model, [], undefined, {}, textPayload);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/character-texts/edit") {
        return Promise.resolve({
          ok: true,
          json: async () => ({
            id: "job-text-edit",
            type: "character_text_edit",
            save_id: "save-1",
            status: "queued",
            result: null,
            error: null
          })
        });
      }
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    await userEvent.click(within(phone).getByRole("button", { name: "Edit text" }));
    const editDialog = await screen.findByRole("dialog", { name: "Edit text" });
    const editor = within(editDialog).getByLabelText("Message");
    await userEvent.clear(editor);
    await userEvent.type(editor, "Can we talk after lab?");
    await userEvent.click(within(editDialog).getByRole("button", { name: "Resubmit" }));

    await waitFor(() => expect(sources.some((source) => (
      source.url === "/api/jobs/job-text-edit/events?save_id=save-1"
    ))).toBe(true));
    act(() => {
      for (const source of sources) {
        source.dispatch("done", {
          id: "job-text-edit",
          type: "character_text_edit",
          save_id: "save-1",
          status: "failed",
          result: null,
          error: "Text resubmit failed."
        });
      }
    });

    expect(await within(phone).findByText("Text resubmit failed.")).toBeInTheDocument();
  });

  it("keeps failed locally accepted character texts visible", async () => {
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    const baseFetch = workbenchFetch([], model, [], undefined, {}, textPayload);
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/character-texts/send-image") {
        return Promise.resolve({
          ok: false,
          status: 400,
          statusText: "Bad Request",
          json: async () => ({ detail: "No chat model configured" })
        });
      }
      return baseFetch(path, init);
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const composer = within(phone).getByRole("textbox", { name: "Message Rowan" }) as HTMLInputElement;
    await userEvent.type(composer, "This should stay visible.");
    await userEvent.click(within(phone).getByRole("button", { name: "Send text" }));

    expect(await within(phone).findByText("This should stay visible.")).toBeInTheDocument();
    await waitFor(() => expect(within(phone).getByText("Failed")).toBeInTheDocument());
    expect(within(phone).getAllByText("No chat model configured").length).toBeGreaterThan(0);
    expect(composer.value).toBe("");
    expect(composer).not.toBeDisabled();
  });

  it("lets users manually repair phone contact permissions", async () => {
    const textPayload = characterTextsPayload();
    const repairModel = (_path: string, init?: RequestInit): CharacterTextsModel => {
      const body = JSON.parse(String(init?.body ?? "{}")) as {
        player_has_character_number?: boolean;
        character_has_player_number?: boolean;
      };
      const updatedRepairContacts = textPayload.model.repair_contacts.map((contact) => (
        contact.id === "character-maya"
          ? {
              ...contact,
              player_has_character_number: Boolean(body.player_has_character_number),
              character_has_player_number: Boolean(body.character_has_player_number),
              player_number_permission: {
                allowed: Boolean(body.player_has_character_number),
                source: body.player_has_character_number ? "manual_or_legacy" : "none",
                reason: body.player_has_character_number
                  ? "You can text them. Saved manually."
                  : "You do not have this character's number.",
                source_message_id: null,
                source_text_message_id: null
              },
              character_number_permission: {
                allowed: Boolean(body.character_has_player_number),
                source: body.character_has_player_number ? "manual_or_legacy" : "none",
                reason: body.character_has_player_number
                  ? "They can text you. Saved manually."
                  : "They cannot text you yet.",
                source_message_id: null,
                source_text_message_id: null
              }
            }
          : contact
      ));
      return {
        ...textPayload.model,
        contacts: updatedRepairContacts.filter((contact) => contact.player_has_character_number),
        repair_contacts: updatedRepairContacts
      };
    };
    const model = runtimeModel({ character_texts_enabled: true });
    const fetchMock = workbenchFetch(
      [],
      model,
      [],
      undefined,
      {},
      { ...textPayload, contactUpdateModel: repairModel }
    );
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Open phone" }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    expect(within(phone).queryByRole("button", { name: "Open text thread for Maya" })).not.toBeInTheDocument();
    expect(within(phone).queryByRole("checkbox", { name: "You can text Rowan" })).not.toBeInTheDocument();

    await userEvent.click(within(phone).getByRole("button", { name: "Add contact" }));
    const addContact = await screen.findByRole("dialog", { name: "Add contact" });
    expect(within(addContact).getByText("Maya")).toBeInTheDocument();
    const playerHasNumber = within(addContact).getByRole("checkbox", {
      name: "You can text Maya"
    });
    const characterHasNumber = within(addContact).getByRole("checkbox", {
      name: "Maya can text you"
    });
    expect(within(addContact).getByText("You do not have this character's number.")).toBeInTheDocument();
    expect(within(addContact).getByText("They cannot text you yet.")).toBeInTheDocument();
    expect(playerHasNumber).not.toBeChecked();
    expect(characterHasNumber).not.toBeChecked();

    await userEvent.click(playerHasNumber);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/character-texts/contacts/character-maya",
      expect.objectContaining({ method: "POST" })
    ));
    const contactUpdateCall = fetchMock.mock.calls.find(([path]) => (
      String(path) === "/api/character-texts/contacts/character-maya"
    ));
    expect(contactUpdateCall).toBeDefined();
    expect(JSON.parse(String(contactUpdateCall?.[1]?.body))).toEqual({
      save_id: "save-1",
      player_has_character_number: true,
      character_has_player_number: false
    });
    expect(await within(addContact).findByRole("checkbox", {
      name: "You can text Maya"
    })).toBeChecked();
    expect(await within(addContact).findByText("You can text them. Saved manually.")).toBeInTheDocument();
    expect(await within(phone).findByRole("button", {
      name: "Open text thread for Maya"
    })).toBeInTheDocument();
    const mayaButton = within(phone).getByRole("button", {
      name: "Open text thread for Maya"
    });
    expect(within(mayaButton).getByText("You can text them.")).toBeInTheDocument();

    await userEvent.click(within(addContact).getByRole("button", { name: "Close Add contact" }));
    expect(
      await within(phone).findByRole("textbox", { name: "Message Maya" })
    ).not.toBeDisabled();
    expect(within(phone).queryByRole("checkbox", {
      name: "You can text Maya"
    })).not.toBeInTheDocument();

    await userEvent.click(within(phone).getByRole("button", { name: "Add contact" }));
    const reopenedAddContact = await screen.findByRole("dialog", { name: "Add contact" });
    await userEvent.click(within(reopenedAddContact).getByRole("checkbox", {
      name: "You can text Maya"
    }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/character-texts/contacts/character-maya",
      expect.objectContaining({ method: "POST" })
    ));
    const contactUpdateCalls = fetchMock.mock.calls.filter(([path]) => (
      String(path) === "/api/character-texts/contacts/character-maya"
    ));
    const finalContactUpdateCall = contactUpdateCalls[contactUpdateCalls.length - 1];
    expect(JSON.parse(String(finalContactUpdateCall?.[1]?.body))).toEqual({
      save_id: "save-1",
      player_has_character_number: false,
      character_has_player_number: false
    });
    await waitFor(() => expect(within(phone).queryByRole("button", {
      name: "Open text thread for Maya"
    })).not.toBeInTheDocument());
  });

  it("uses character contact name in the phone when set", async () => {
    const textPayload = characterTextsPayload({
      rowanContactOverrides: { contact_name: "Row (lab partner)" }
    });
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    expect(within(phone).getByRole("button", {
      name: "Open text thread for Row (lab partner)"
    })).toHaveClass("selected");
    expect(within(phone).getByRole("region", {
      name: "Conversation with Row (lab partner)"
    })).toBeInTheDocument();
    expect(within(phone).getByRole("textbox", {
      name: "Message Row (lab partner)"
    })).toBeInTheDocument();
  });

  it("falls back to character name when contact name is empty or blank", async () => {
    const textPayload = characterTextsPayload({
      rowanContactOverrides: { contact_name: "   " }
    });
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const rowanButton = within(phone).getByRole("button", {
      name: "Open text thread for Rowan"
    });
    expect(rowanButton).toHaveClass("selected");
    expect(within(rowanButton).getByText("Rowan")).toBeInTheDocument();
  });

  it("uses an inbox-first mobile phone flow", async () => {
    stubWorkbenchMedia(true);
    const textPayload = characterTextsPayload();
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));

    const phone = await screen.findByRole("dialog", { name: "Phone" });
    expect(within(phone).getByRole("button", { name: "Open text thread for Rowan" })).toBeInTheDocument();
    expect(within(phone).queryByRole("textbox", { name: "Message Rowan" })).not.toBeInTheDocument();

    await userEvent.click(within(phone).getByRole("button", { name: "Open text thread for Rowan" }));

    expect(await within(phone).findByRole("textbox", { name: "Message Rowan" })).toBeInTheDocument();
    expect(within(phone).getByRole("button", { name: "Back to contacts" })).toBeInTheDocument();
    expect(within(phone).queryByRole("button", { name: "Open text thread for Maya" })).not.toBeInTheDocument();

    await userEvent.click(within(phone).getByRole("button", { name: "Back to contacts" }));

    expect(within(phone).getByRole("button", { name: "Open text thread for Rowan" })).toBeInTheDocument();
    expect(within(phone).queryByRole("textbox", { name: "Message Rowan" })).not.toBeInTheDocument();
  });

  it("keeps long mobile phone text messages wrapped inside the viewport", async () => {
    stubWorkbenchMedia(true);
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      value: 1200
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      value: 420
    });
    const longToken = "crystal-lens-calibration-" + "x".repeat(96);
    const latestBody = `Latest field notes ${longToken}`;
    const textPayload = characterTextsPayload();
    textPayload.model.contacts = textPayload.model.contacts.map((contact) => (
      contact.id === "character-rowan"
        ? {
            ...contact,
            latest_message_id: "text-long",
            latest_message_body: latestBody,
            latest_message_markdown_blocks: [
              {
                kind: "paragraph",
                spans: [{ kind: "text", text: latestBody }]
              }
            ],
            latest_message_sender: "character",
            latest_message_at: "2026-07-01T12:04:00Z"
          }
        : contact
    ));
    textPayload.threads["thread-rowan"].messages = [
      {
        id: "text-1",
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: "player",
        body: "Can we talk after class?",
        delivery_status: "sent",
        markdown_blocks: [
          {
            kind: "paragraph",
            spans: [{ kind: "text", text: "Can we talk after class?" }]
          }
        ]
      },
      {
        id: "text-2",
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: "character",
        body: "Meet me by the arcade.",
        delivery_status: "sent",
        markdown_blocks: [
          {
            kind: "paragraph",
            spans: [{ kind: "text", text: "Meet me by the arcade." }]
          }
        ]
      },
      {
        id: "text-long",
        thread_id: "thread-rowan",
        character_id: "character-rowan",
        sender: "character",
        body: latestBody,
        delivery_status: "sent",
        markdown_blocks: [
          {
            kind: "paragraph",
            spans: [{ kind: "text", text: latestBody }]
          }
        ]
      }
    ];
    const model = runtimeModel({ character_texts_enabled: true });
    vi.stubGlobal("fetch", workbenchFetch([], model, [], undefined, {}, textPayload));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: /Open phone/ }));
    const phone = await screen.findByRole("dialog", { name: "Phone" });
    const phoneTitle = within(phone).getByText("Phone").closest(".character-text-phone-title");
    expect(phoneTitle).not.toBeNull();
    expect(within(phoneTitle as HTMLElement).getByRole("button", { name: "Close phone" })).toBeInTheDocument();
    const rowanButton = within(phone).getByRole("button", { name: "Open text thread for Rowan" });
    const contactCopy = rowanButton.querySelector(".character-text-contact-copy") as HTMLElement | null;
    expect(contactCopy).not.toBeNull();
    expect(contactCopy?.tagName).toBe("DIV");
    expect(within(rowanButton).getByText(latestBody)).toBeInTheDocument();

    await userEvent.click(rowanButton);

    const body = within(phone).getByText(latestBody).closest(".character-text-bubble-body") as HTMLElement | null;
    const paragraph = body?.querySelector("p") as HTMLElement | null;
    const bubble = body?.closest(".character-text-bubble") as HTMLElement | null;
    const messages = body?.closest(".character-text-messages") as HTMLElement | null;
    expect(within(phone).getByRole("textbox", { name: "Message Rowan" })).toBeInTheDocument();
    expect(phone.querySelector(".character-text-inbox")).toBeNull();
    expect(body).not.toBeNull();
    expect(paragraph).not.toBeNull();
    expect(bubble).not.toBeNull();
    expect(messages).not.toBeNull();
    expect(messages).toHaveClass("character-text-messages");
    expect(bubble).toHaveClass("character-text-bubble");
    expect(body).toHaveClass("character-text-bubble-body");
    await waitFor(() => expect(messages?.scrollTop).toBe(1200));
  });

  it("opens mobile library controls as a sheet", async () => {
    stubWorkbenchMedia(true);
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }]
    });
    const scenarios: Scenario[] = [
      {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Fog Gate",
        premise: "A gate in the fog.",
        player_role: "Keeper",
        opening_message: "The fog opens.",
        save_count: 0
      }
    ];
    vi.stubGlobal("fetch", workbenchFetch([], model, scenarios));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Library" }));

    const sheet = screen.getByRole("dialog", { name: "Library" });
    expect(within(sheet).getByRole("button", { name: "New scenario" })).toBeInTheDocument();
    expect(within(sheet).getByRole("button", { name: "Load Lantern Keep" })).toBeInTheDocument();
    const renameSave = within(sheet).getByRole("button", { name: "Rename Lantern Keep" });
    expect(renameSave).toHaveClass("touch-labeled-action");
    expect(within(renameSave).getByText("Rename")).toHaveClass("touch-action-label");
    await userEvent.click(within(sheet).getByRole("tab", { name: /Scenarios/ }));
    expect(await within(sheet).findByRole("button", { name: "Start Fog Gate" })).toBeInTheDocument();
    const editScenario = within(sheet).getByRole("button", { name: "Edit Fog Gate" });
    expect(editScenario).toHaveClass("touch-labeled-action");
    expect(within(editScenario).getByText("Edit")).toHaveClass("touch-action-label");
  });

  it("opens mobile library dialogs above the sheet layer", async () => {
    stubWorkbenchMedia(true);
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }]
    });
    vi.stubGlobal("fetch", workbenchFetch([], model));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Library" }));
    const sheet = screen.getByRole("dialog", { name: "Library" });
    const renameButton = within(sheet).getByRole("button", { name: "Rename Lantern Keep" });

    await userEvent.click(renameButton);

    const dialog = screen.getByRole("dialog", { name: "Rename save" });
    expect(dialog.closest(".mobile-sheet")).toBeNull();
    expect(dialog.parentElement).toHaveClass("modal-backdrop");
    expect(dialog.parentElement?.parentElement).toBe(document.body);
    within(dialog).getByLabelText("Save Title").focus();
    expect(within(dialog).getByLabelText("Save Title")).toHaveFocus();
    await userEvent.tab({ shift: true });
    expect(within(dialog).getByLabelText("Close")).toHaveFocus();

    await userEvent.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Rename save" })).not.toBeInTheDocument());
    expect(screen.getByRole("dialog", { name: "Library" })).toBeInTheDocument();
    expect(renameButton).toHaveFocus();
  });

  it("hides child-blocked library controls while keeping save and scenario starts available", async () => {
    const onChanged = vi.fn();
    const onSelectSave = vi.fn().mockResolvedValue(undefined);
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }]
        })}
        scenarios={[scenarioFixture({ title: "Fog Gate", premise: "A gate in the fog." })]}
        currentUser={{ id: "child-1", username: "Ilyra", role: "child", status: "active" }}
        onChanged={onChanged}
        onSelectSave={onSelectSave}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: "New scenario" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load Lantern Keep" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (1)" }));
    expect(screen.getByRole("button", { name: "Start Fog Gate" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Import save bundle" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Export active save" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Rename Lantern Keep" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete Lantern Keep" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Import scenario bundle" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Export Fog Gate" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit Fog Gate" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete Fog Gate" })).not.toBeInTheDocument();
  });

  it("keeps unsupported legacy library records recoverable without allowing play or mutation", async () => {
    const onSelectSave = vi.fn();
    const onContinuationDraft = vi.fn();
    const onReuseScenarioPrompt = vi.fn();
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          active_save_id: "save-retired",
          saves: [{
            save_id: "save-retired",
            title: "Retired Chronicle",
            active: true,
            supported: false,
            unsupported_reason: "Single-character Dating Sim is no longer supported."
          }]
        })}
        scenarios={[scenarioFixture({
          scenario_id: "scenario-retired",
          title: "Retired Scenario",
          has_generation_prompt: true,
          supported: false,
          unsupported_reason: "Single-character Dating Sim is no longer supported."
        })]}
        currentUser={{ id: "admin-1", username: "Mira", role: "admin", status: "active" }}
        onChanged={vi.fn()}
        onSelectSave={onSelectSave}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        onContinuationDraft={onContinuationDraft}
        onReuseScenarioPrompt={onReuseScenarioPrompt}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    expect(screen.getByText("Unsupported")).toBeInTheDocument();
    expect(screen.getByText("Single-character Dating Sim is no longer supported.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load Retired Chronicle" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Rename Retired Chronicle" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export Retired Chronicle" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Delete Retired Chronicle" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "New chapter from current save" })).toBeDisabled();

    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (1)" }));
    expect(screen.getAllByText("Unsupported")).toHaveLength(1);
    expect(screen.getByRole("button", { name: "Start Retired Scenario" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Reuse prompt for Retired Scenario" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Edit Retired Scenario" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export Retired Scenario" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Delete Retired Scenario" })).toBeEnabled();
    expect(onSelectSave).not.toHaveBeenCalled();
  });

  it("closes library mutation dialogs when refreshed resources become unsupported", async () => {
    const { LeftRail } = await import("./main");
    const currentUser = { id: "admin-1", username: "Mira", role: "admin", status: "active" };
    const supportedSave = { save_id: "save-1", title: "Lantern Keep", active: true, supported: true };
    const unsupportedSave = { ...supportedSave, supported: false, unsupported_reason: "Retired scenario type." };
    const supportedScenario = scenarioFixture({ scenario_id: "scenario-1", title: "Fog Gate", supported: true });
    const unsupportedScenario = { ...supportedScenario, supported: false, unsupported_reason: "Retired scenario type." };
    const props = {
      currentUser,
      onChanged: vi.fn(),
      onSelectSave: vi.fn(),
      pendingSaveId: null,
      saveSelectionError: "",
      onNew: vi.fn(),
      onContinuationDraft: vi.fn(),
      activePanel: "media" as const,
      setPanel: vi.fn()
    };
    const supportedModel = runtimeModel({ active_save_id: "save-1", saves: [supportedSave] });
    const unsupportedModel = runtimeModel({ active_save_id: "save-1", saves: [unsupportedSave] });
    const client = new QueryClient();
    const rail = (model: RuntimeModel, scenarios: Scenario[]) => (
      <QueryClientProvider client={client}>
        <LeftRail {...props} model={model} scenarios={scenarios} />
      </QueryClientProvider>
    );
    const { rerender } = render(rail(supportedModel, [supportedScenario]));

    await userEvent.click(screen.getByRole("button", { name: "Rename Lantern Keep" }));
    expect(screen.getByRole("dialog", { name: "Rename save" })).toBeInTheDocument();
    rerender(rail(unsupportedModel, [supportedScenario]));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Rename save" })).not.toBeInTheDocument());

    rerender(rail(supportedModel, [supportedScenario]));
    await userEvent.click(screen.getByRole("button", { name: "New chapter from current save" }));
    expect(screen.getByRole("dialog", { name: "New chapter from current save" })).toBeInTheDocument();
    rerender(rail(unsupportedModel, [supportedScenario]));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "New chapter from current save" })).not.toBeInTheDocument());

    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (1)" }));
    rerender(rail(supportedModel, [supportedScenario]));
    await userEvent.click(screen.getByRole("button", { name: "Start Fog Gate" }));
    expect(screen.getByRole("dialog", { name: "Start Scenario" })).toBeInTheDocument();
    rerender(rail(supportedModel, [unsupportedScenario]));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Start Scenario" })).not.toBeInTheDocument());

    rerender(rail(supportedModel, [supportedScenario]));
    await userEvent.click(screen.getByRole("button", { name: "Edit Fog Gate" }));
    expect(screen.getByRole("dialog", { name: "Edit scenario: Fog Gate" })).toBeInTheDocument();
    rerender(rail(supportedModel, [unsupportedScenario]));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Edit scenario: Fog Gate" })).not.toBeInTheDocument());
  });

  it("loads saves from the left rail and refreshes library state", async () => {
    const onChanged = vi.fn();
    const onSelectSave = vi.fn().mockResolvedValue(undefined);
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [
            { save_id: "save-1", title: "Lantern Keep", active: true },
            { save_id: "save-2", title: "Signal Tower", active: false }
          ]
        })}
        scenarios={[]}
        onChanged={onChanged}
        onSelectSave={onSelectSave}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Load Signal Tower" }));

    await waitFor(() => expect(onSelectSave).toHaveBeenCalledWith("save-2"));
    expect(onChanged).toHaveBeenCalledTimes(1);
  });

  it("keeps the lazy scenario request alive while the tab shows loading", async () => {
    const scenario = scenarioFixture({
      title: "Fog Gate",
      premise: "A gate in the fog.",
      save_count: 0
    });
    let scenarioSignal: AbortSignal | undefined;
    let resolveScenarios: () => void = () => undefined;
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/scenarios") {
        scenarioSignal = init?.signal ?? undefined;
        return new Promise<{ ok: boolean; json: () => Promise<{ scenarios: Scenario[] }> }>((resolve, reject) => {
          resolveScenarios = () => resolve({
            ok: true,
            json: async () => ({ scenarios: [scenario] })
          });
          init?.signal?.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }]
        })}
        onChanged={vi.fn()}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("tab", { name: /Scenarios/ }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios", expect.anything()));
    expect(screen.getByText("Loading scenarios...")).toBeInTheDocument();
    expect(scenarioSignal).toBeDefined();
    expect(scenarioSignal?.aborted).toBe(false);

    await act(async () => {
      resolveScenarios();
    });

    expect(await screen.findByRole("button", { name: "Start Fog Gate" })).toBeInTheDocument();
  });

  it("searches sorts and filters large library lists in the left rail", async () => {
    window.localStorage.clear();
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [
            {
              save_id: "save-1",
              title: "Lantern Keep",
              active: true,
              scenario_id: "scenario-ashfall",
              scenario_title: "Ashfall Keep",
              created_at: "2026-05-01 00:00:00",
              updated_at: "2026-05-03 00:00:00",
              last_opened_at: "2026-05-04 00:00:00"
            },
            {
              save_id: "save-2",
              title: "Signal Tower",
              active: false,
              scenario_id: "scenario-harbor",
              scenario_title: "Glass Harbor",
              created_at: "2026-05-02 00:00:00",
              updated_at: "2026-05-04 00:00:00",
              last_opened_at: "2026-05-06 00:00:00"
            },
            {
              save_id: "save-3",
              title: "Moon Archive",
              active: false,
              scenario_id: "scenario-ashfall",
              scenario_title: "Ashfall Keep",
              created_at: "2026-05-03 00:00:00",
              updated_at: "2026-05-05 00:00:00",
              last_opened_at: "2026-05-05 00:00:00"
            }
          ]
        })}
        scenarios={[
          scenarioFixture({
            scenario_id: "scenario-ashfall",
            title: "Ashfall Keep",
            premise: "A storm keep above the ash sea.",
            player_role: "Warden",
            save_count: 2,
            created_at: "2026-05-01 00:00:00",
            updated_at: "2026-05-05 00:00:00"
          }),
          scenarioFixture({
            scenario_id: "scenario-harbor",
            scenario_type: "science_fiction_roleplay",
            scenario_types: ["science_fiction_roleplay", "dating_sim"],
            title: "Glass Harbor",
            premise: "A drowned harbor rings at low tide.",
            player_role: "Harbor warden",
            save_count: 0,
            created_at: "2026-05-02 00:00:00",
            updated_at: "2026-05-04 00:00:00"
          })
        ]}
        onChanged={vi.fn()}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    expect(screen.getByLabelText("Sort saves")).toHaveValue("updated");
    expect(screen.getByText("Moon Archive").compareDocumentPosition(screen.getByText("Signal Tower")) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    await userEvent.selectOptions(screen.getByLabelText("Sort saves"), "last_opened");
    expect(screen.getByText("Signal Tower").compareDocumentPosition(screen.getByText("Moon Archive")) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    await userEvent.type(screen.getByLabelText("Search saves"), "ashfall");
    expect(screen.getByRole("button", { name: "Load Lantern Keep" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Load Moon Archive" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Load Signal Tower" })).not.toBeInTheDocument();

    await userEvent.clear(screen.getByLabelText("Search saves"));
    await userEvent.selectOptions(screen.getByLabelText("Sort saves"), "title");
    expect(screen.getByText("Lantern Keep").compareDocumentPosition(screen.getByText("Moon Archive")) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();

    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (2)" }));
    await userEvent.selectOptions(screen.getByLabelText("Scenario usage"), "unused");
    expect(screen.getByRole("button", { name: "Start Glass Harbor" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start Ashfall Keep" })).not.toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText("Scenario type"), "dating_sim");
    expect(screen.getByRole("button", { name: "Start Glass Harbor" })).toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText("Scenario type"), "full_roleplay");
    expect(screen.getByText("No scenarios match")).toBeInTheDocument();
  });

  it("remembers library manager preferences locally", async () => {
    window.localStorage.clear();
    const { LeftRail } = await import("./main");
    const model = runtimeModel({
      saves: [
        { save_id: "save-1", title: "Lantern Keep", active: true, scenario_title: "Ashfall Keep" },
        { save_id: "save-2", title: "Signal Tower", active: false, scenario_title: "Glass Harbor" }
      ]
    });
    const scenarios = [
      scenarioFixture({ title: "Fog Gate", premise: "A gate in the fog." }),
      scenarioFixture({
        scenario_id: "scenario-2",
        scenario_type: "dating_sim",
        title: "Market Duel",
        premise: "A duel under bright awnings.",
        save_count: 0
      })
    ];
    const props = {
      model,
      scenarios,
      currentUser: { id: "user-1", username: "Mira", role: "admin", status: "active" },
      onChanged: vi.fn(),
      onSelectSave: vi.fn(),
      pendingSaveId: null,
      saveSelectionError: "",
      onNew: vi.fn(),
      activePanel: "media" as const,
      setPanel: vi.fn()
    };

    const { unmount } = render(<LeftRail {...props} />);
    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (2)" }));
    await userEvent.type(screen.getByLabelText("Search scenarios"), "market");
    await userEvent.selectOptions(screen.getByLabelText("Scenario usage"), "unused");
    expect(screen.getByRole("button", { name: "Start Market Duel" })).toBeInTheDocument();
    unmount();

    render(<LeftRail {...props} />);

    expect(screen.getByRole("tab", { name: "Scenarios (2)" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("Search scenarios")).toHaveValue("market");
    expect(screen.getByLabelText("Scenario usage")).toHaveValue("unused");
    expect(screen.getByRole("button", { name: "Start Market Duel" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start Fog Gate" })).not.toBeInTheDocument();
  });

  it("does not write stale library preferences across user scopes", async () => {
    window.localStorage.setItem("bragi-web:library-controls:v1:user-2", JSON.stringify({
      activeTab: "scenarios",
      scenarioQuery: "market",
      scenarioUsage: "unused"
    }));
    const setItemSpy = vi.spyOn(Storage.prototype, "setItem");
    const { LeftRail } = await import("./main");
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Keep", active: true, scenario_title: "Ashfall Keep" }]
    });
    const scenarios = [
      scenarioFixture({ title: "Fog Gate", premise: "A gate in the fog.", save_count: 1 }),
      scenarioFixture({
        scenario_id: "scenario-2",
        scenario_type: "dating_sim",
        title: "Market Duel",
        premise: "A duel under bright awnings.",
        save_count: 0
      })
    ];
    const props = {
      model,
      scenarios,
      currentUser: { id: "user-1", username: "Mira", role: "admin", status: "active" },
      onChanged: vi.fn(),
      onSelectSave: vi.fn(),
      pendingSaveId: null,
      saveSelectionError: "",
      onNew: vi.fn(),
      activePanel: "media" as const,
      setPanel: vi.fn()
    };

    const { rerender } = render(<LeftRail {...props} />);
    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (2)" }));
    await userEvent.type(screen.getByLabelText("Search scenarios"), "fog");
    expect(screen.getByRole("button", { name: "Start Fog Gate" })).toBeInTheDocument();
    setItemSpy.mockClear();

    rerender(
      <LeftRail
        {...props}
        currentUser={{ id: "user-2", username: "Ilyra", role: "admin", status: "active" }}
      />
    );

    await waitFor(() => expect(screen.getByLabelText("Search scenarios")).toHaveValue("market"));
    const userTwoWrites = setItemSpy.mock.calls
      .filter(([key]) => key === "bragi-web:library-controls:v1:user-2")
      .map(([, value]) => String(value));
    expect(userTwoWrites.some((value) => value.includes("fog"))).toBe(false);
    expect(screen.getByRole("button", { name: "Start Market Duel" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Start Fog Gate" })).not.toBeInTheDocument();
  });

  it("renames saves from a focused modal in the left rail", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => runtimeModel({
        active_save_title: "Lantern Run Revised",
        saves: [{ save_id: "save-1", title: "Lantern Run Revised", active: true }]
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const onChanged = vi.fn();
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [{ save_id: "save-1", title: "Lantern Run", active: true }]
        })}
        scenarios={[]}
        onChanged={onChanged}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Rename Lantern Run" }));
    const dialog = screen.getByRole("dialog", { name: "Rename save" });
    expect(within(dialog).getByLabelText("Save Title")).toHaveValue("Lantern Run");

    await userEvent.clear(within(dialog).getByLabelText("Save Title"));
    await userEvent.type(within(dialog).getByLabelText("Save Title"), "Lantern Run Revised");
    await userEvent.click(within(dialog).getByRole("button", { name: "Rename" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/saves/save-1/rename", expect.anything()));
    const renameCall = fetchMock.mock.calls.find(([path]) => path === "/api/saves/save-1/rename");
    expect(JSON.parse(String(renameCall?.[1].body))).toEqual({ title: "Lantern Run Revised" });
    expect(onChanged).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Rename save" })).not.toBeInTheDocument());
  });

  it("keeps save rename validation local for blank titles", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [{ save_id: "save-1", title: "Lantern Run", active: true }]
        })}
        scenarios={[]}
        onChanged={vi.fn()}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Rename Lantern Run" }));
    const dialog = screen.getByRole("dialog", { name: "Rename save" });
    await userEvent.clear(within(dialog).getByLabelText("Save Title"));
    await userEvent.click(within(dialog).getByRole("button", { name: "Rename" }));

    expect(await within(dialog).findByText("Save title is required")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: "Rename save" })).toBeInTheDocument();
  });

  it("keeps the save rename modal open when the API rejects the change", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/saves/save-1/rename"
        ? { ok: false, status: 409, statusText: "Conflict", json: async () => ({ detail: "Save title could not be changed." }) }
        : { ok: true, json: async () => ({ ok: true }) }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const onChanged = vi.fn();
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [{ save_id: "save-1", title: "Lantern Run", active: true }]
        })}
        scenarios={[]}
        onChanged={onChanged}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Rename Lantern Run" }));
    const dialog = screen.getByRole("dialog", { name: "Rename save" });
    await userEvent.clear(within(dialog).getByLabelText("Save Title"));
    await userEvent.type(within(dialog).getByLabelText("Save Title"), "Lantern Run Revised");
    await userEvent.click(within(dialog).getByRole("button", { name: "Rename" }));

    expect(await within(dialog).findByText("Save title could not be changed.")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Rename save" })).toBeInTheDocument();
    expect(onChanged).not.toHaveBeenCalled();
  });

  it("confirms save and scenario deletes before refreshing the library", async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);
    const onChanged = vi.fn();
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [{ save_id: "save-1", title: "Lantern Run", active: true }]
        })}
        scenarios={[scenarioFixture({ title: "Fog Gate", premise: "A gate in the fog." })]}
        onChanged={onChanged}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Delete Lantern Run" }));
    expect(screen.getByRole("dialog", { name: "Delete save?" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/saves/save-1", expect.anything()));
    const saveDelete = fetchMock.mock.calls.find(([path]) => path === "/api/saves/save-1");
    expect(saveDelete?.[1]).toMatchObject({ method: "DELETE" });
    expect(onChanged).toHaveBeenCalledTimes(1);
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Delete save?" })).not.toBeInTheDocument());

    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (1)" }));
    await userEvent.click(screen.getByRole("button", { name: "Delete Fog Gate" }));
    expect(screen.getByRole("dialog", { name: "Delete scenario?" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/scenario-1", expect.anything()));
    const scenarioDelete = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/scenario-1");
    expect(scenarioDelete?.[1]).toMatchObject({ method: "DELETE" });
    expect(onChanged).toHaveBeenCalledTimes(2);
  });

  it("keeps destructive delete confirmations open when deletion fails", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/saves/save-1"
        ? { ok: false, status: 409, statusText: "Conflict", json: async () => ({ detail: "Cannot delete the active save." }) }
        : { ok: true, json: async () => ({}) }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const onChanged = vi.fn();
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({
          saves: [{ save_id: "save-1", title: "Lantern Run", active: true }]
        })}
        scenarios={[]}
        onChanged={onChanged}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Delete Lantern Run" }));
    await userEvent.click(screen.getByRole("button", { name: "Delete" }));

    expect(await screen.findByText("Cannot delete the active save.")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Delete save?" })).toBeInTheDocument();
    expect(onChanged).not.toHaveBeenCalled();
  });

  it("edits saved scenario definitions with structured fields and custom sections", async () => {
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }]
    });
    const scenarios: Scenario[] = [
      {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Fog Gate",
        premise: "A gate in the fog.",
        player_role: "Keeper",
        opening_message: "The fog opens.",
        save_count: 0
      }
    ];
    const definitionPayload: any = {
      active_save_id: null,
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Fog Gate",
        premise: "A gate in the fog.",
        player_character_name: "Mara",
        player_role: "Keeper",
        content_sections: [
          ["tone_genre", "Fog mystery"],
          ["hidden_clock", "The gate opens at midnight."],
          ["rumor_board", "Pilgrims leave warnings on brass tags."]
        ],
        character_starters: [
          {
            name: "Captain Ilyra",
            aliases: ["Ilyra"],
            role: "Watch captain",
            known_state: "She guards the fog gate.",
            appearance: "Bronze cloak clasp.",
            visual_notes: "Straight silhouette in lantern haze.",
            personality: "Decisive and guarded.",
            voice: "Low clipped orders.",
            texting_style: "Short formal replies, no emoji.",
            relationships: { Mara: "wary ally" },
            goals: "Hold the fog gate until dawn.",
            motivations: "Protect the lower town from gate weather.",
            boundaries: "Will not open the gate without a brass writ.",
            status: "waiting near the gate",
            met: true,
            locked_fields: ["appearance"]
          }
        ]
      }
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) return model;
        if (path === "/api/scenarios") return { scenarios };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return { save_id: "save-1", can_submit: true, reason: null, blocking_job_id: null, blocking_job_status: null };
        if (path === "/api/scenarios/scenario-1/definition") return definitionPayload;
        return { model: definitionPayload };
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /Scenarios/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Edit Fog Gate" }));
    const dialog = await screen.findByRole("dialog", { name: /edit scenario: fog gate/i });

    expect(within(dialog).getByLabelText("Title")).toHaveValue("Fog Gate");
    expect(within(dialog).getByLabelText("Tone Genre")).toHaveValue("Fog mystery");
    expect(within(dialog).queryByDisplayValue(/"scenario_id"/)).not.toBeInTheDocument();
    expect(within(dialog).getByText("Character starters")).toBeInTheDocument();
    expect(within(dialog).getByLabelText("Starter Captain Ilyra role")).toHaveValue("Watch captain");
    expect(within(dialog).getByLabelText("Starter Captain Ilyra goals")).toHaveValue("Hold the fog gate until dawn.");
    expect(within(dialog).queryByLabelText("Section body character_starters")).not.toBeInTheDocument();
    const uploadReference = within(dialog).getByRole("button", { name: "Upload Captain Ilyra reference image" });
    expect(uploadReference).toBeEnabled();

    fireEvent.change(within(dialog).getByLabelText("Title"), { target: { value: "Fog Gate Revised" } });
    expect(uploadReference).toBeDisabled();
    fireEvent.change(within(dialog).getByLabelText("Player Role"), { target: { value: "Keeper of paths" } });
    fireEvent.change(within(dialog).getByLabelText("Starter Captain Ilyra role"), { target: { value: "Gate captain" } });
    fireEvent.change(within(dialog).getByLabelText("Starter Captain Ilyra appearance"), { target: { value: "Fog-damp bronze cloak clasp." } });
    fireEvent.change(within(dialog).getByLabelText("Starter Captain Ilyra goals"), { target: { value: "Hold the revised fog gate until dawn." } });
    await userEvent.click(within(dialog).getByRole("button", { name: "Move rumor_board up" }));
    await userEvent.click(within(dialog).getByRole("button", { name: "Remove hidden_clock" }));
    fireEvent.change(within(dialog).getByLabelText("Section body rumor_board"), { target: { value: "Pilgrims leave warnings on brass tags and wax seals." } });
    fireEvent.change(within(dialog).getByLabelText("New section key"), { target: { value: "omen_table" } });
    await userEvent.click(within(dialog).getByRole("button", { name: "Add section" }));
    fireEvent.change(within(dialog).getByLabelText("Section body omen_table"), { target: { value: "Three omens are carved under the gate." } });
    await userEvent.click(within(dialog).getByRole("button", { name: "Save scenario" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/scenario-1/definition", expect.anything()));
    const saveCall = fetchMock.mock.calls.find(([path, init]) => path === "/api/scenarios/scenario-1/definition" && (init as RequestInit | undefined)?.body);
    const savedBody = JSON.parse(String(saveCall?.[1].body));
    expect(savedBody).toMatchObject({
      edit: {
        title: "Fog Gate Revised",
        premise: "A gate in the fog.",
        player_character_name: "Mara",
        player_role: "Keeper of paths",
        content_sections: expect.arrayContaining([
          ["tone_genre", "Fog mystery"],
          ["rumor_board", "Pilgrims leave warnings on brass tags and wax seals."],
          ["omen_table", "Three omens are carved under the gate."]
        ]),
        character_starters: [
          expect.objectContaining({
            name: "Captain Ilyra",
            aliases: ["Ilyra"],
            role: "Gate captain",
            appearance: "Fog-damp bronze cloak clasp.",
            texting_style: "Short formal replies, no emoji.",
            goals: "Hold the revised fog gate until dawn.",
            motivations: "Protect the lower town from gate weather.",
            boundaries: "Will not open the gate without a brass writ.",
            relationships: { Mara: "wary ally" },
            met: true,
            locked_fields: ["appearance"]
          })
        ]
      }
    });
    const savedSections = savedBody.edit.content_sections as [string, string][];
    expect(savedSections.some(([key]) => key === "hidden_clock")).toBe(false);
    expect(savedSections.some(([key]) => key === "character_starters")).toBe(false);
    expect(savedSections.findIndex(([key]) => key === "rumor_board")).toBeLessThan(savedSections.findIndex(([key]) => key === "omen_table"));
  });

  it("uploads and removes saved scenario starter reference images", async () => {
    const scenario: Scenario = {
      scenario_id: "scenario-1",
      scenario_type: "full_roleplay",
      title: "Fog Gate",
      premise: "A gate in the fog.",
      player_role: "Keeper",
      opening_message: "The fog opens.",
      save_count: 0
    };
    const definitionPayload: any = {
      active_save_id: null,
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Fog Gate",
        premise: "A gate in the fog.",
        player_character_name: "Mara",
        player_role: "Keeper",
        content_sections: [["tone_genre", "Fog mystery"]],
        character_starters: [
          {
            name: "Captain Ilyra",
            aliases: [],
            role: "Watch captain",
            known_state: "",
            appearance: "",
            visual_notes: "",
            personality: "",
            voice: "",
            texting_style: "",
            relationships: {},
            goals: "",
            motivations: "",
            boundaries: "",
            status: "",
            met: true,
            locked_fields: [],
            reference_image: null
          }
        ]
      }
    };
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      if (path === "/api/scenarios/scenario-1/character-starters/reference-image/upload") {
        definitionPayload.scenario.character_starters[0] = {
          ...definitionPayload.scenario.character_starters[0],
          starter_id: "starter-ilyra",
          reference_image: {
            id: "starter-ref-1",
            path: "scenario-starters/scenario-1/starter-ref-1.png",
            thumbnail_path: "scenario-starters/scenario-1/thumbnails/starter-ref-1.png",
            mime_type: "image/png",
            prompt_preview: "Uploaded character reference image",
            source: "uploaded",
            created_at: "2026-07-12T00:00:00+00:00"
          }
        };
        return Promise.resolve({ ok: true, json: async () => definitionPayload });
      }
      if (path === "/api/scenarios/scenario-1/character-starters/reference-image/remove") {
        definitionPayload.scenario.character_starters[0] = {
          ...definitionPayload.scenario.character_starters[0],
          reference_image: null
        };
        return Promise.resolve({ ok: true, json: async () => definitionPayload });
      }
      return Promise.resolve({
        ok: true,
        json: async () => {
          if (path.startsWith("/api/runtime")) return runtimeModel();
          if (path === "/api/scenarios") return { scenarios: [scenario] };
          if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
          if (path === "/api/settings") return modelSettingsPayload();
          if (path.startsWith("/api/chat/submission-status")) return { save_id: "save-1", can_submit: true, reason: null, blocking_job_id: null, blocking_job_status: null };
          if (path === "/api/scenarios/scenario-1/definition") return definitionPayload;
          return {};
        }
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /Scenarios/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Edit Fog Gate" }));
    const dialog = await screen.findByRole("dialog", { name: /edit scenario: fog gate/i });
    const file = new File(["starter image"], "ilyra.png", { type: "image/png" });

    await userEvent.upload(within(dialog).getByLabelText("Upload Captain Ilyra reference image file"), file);
    expect(await within(dialog).findByAltText("Uploaded character reference image")).toBeInTheDocument();
    const uploadCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/scenario-1/character-starters/reference-image/upload");
    const uploadForm = uploadCall?.[1]?.body as FormData;
    expect(uploadForm.get("starter_name")).toBe("Captain Ilyra");
    expect(uploadForm.get("replace_existing")).toBe("false");

    await userEvent.click(within(dialog).getByRole("button", { name: "Remove Captain Ilyra reference image" }));

    await waitFor(() => {
      expect(within(dialog).queryByAltText("Uploaded character reference image")).not.toBeInTheDocument();
    });
    const removeCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/scenario-1/character-starters/reference-image/remove");
    expect(JSON.parse(String(removeCall?.[1]?.body))).toEqual({
      starter_id: "starter-ilyra",
      starter_name: "Captain Ilyra"
    });
  });

  it("preserves starter edits made while reference image upload is pending", async () => {
    const scenario: Scenario = {
      scenario_id: "scenario-1",
      scenario_type: "full_roleplay",
      title: "Fog Gate",
      premise: "A gate in the fog.",
      player_role: "Keeper",
      opening_message: "The fog opens.",
      save_count: 0
    };
    const definitionPayload: any = {
      active_save_id: null,
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Fog Gate",
        premise: "A gate in the fog.",
        player_character_name: "Mara",
        player_role: "Keeper",
        content_sections: [["tone_genre", "Fog mystery"]],
        character_starters: [
          {
            name: "Captain Ilyra",
            aliases: [],
            role: "Watch captain",
            known_state: "",
            appearance: "",
            visual_notes: "",
            personality: "",
            voice: "",
            texting_style: "",
            relationships: {},
            goals: "",
            motivations: "",
            boundaries: "",
            status: "",
            met: true,
            locked_fields: [],
            reference_image: null
          }
        ]
      }
    };
    const uploadedPayload = {
      ...definitionPayload,
      scenario: {
        ...definitionPayload.scenario,
        character_starters: [
          {
            ...definitionPayload.scenario.character_starters[0],
            starter_id: "starter-ilyra",
            role: "Watch captain",
            reference_image: {
              id: "starter-ref-1",
              path: "scenario-starters/scenario-1/starter-ref-1.png",
              thumbnail_path: "scenario-starters/scenario-1/thumbnails/starter-ref-1.png",
              mime_type: "image/png",
              prompt_preview: "Uploaded character reference image",
              source: "uploaded",
              created_at: "2026-07-12T00:00:00+00:00"
            }
          }
        ]
      }
    };
    const uploadResponse = deferred<{
      ok: boolean;
      json: () => Promise<typeof uploadedPayload>;
    }>();
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      if (path === "/api/scenarios/scenario-1/character-starters/reference-image/upload") {
        return uploadResponse.promise;
      }
      return Promise.resolve({
        ok: true,
        json: async () => {
          if (path.startsWith("/api/runtime")) return runtimeModel();
          if (path === "/api/scenarios") return { scenarios: [scenario] };
          if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
          if (path === "/api/settings") return modelSettingsPayload();
          if (path.startsWith("/api/chat/submission-status")) return { save_id: "save-1", can_submit: true, reason: null, blocking_job_id: null, blocking_job_status: null };
          if (path === "/api/scenarios/scenario-1/definition") return definitionPayload;
          return {};
        }
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /Scenarios/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Edit Fog Gate" }));
    const dialog = await screen.findByRole("dialog", { name: /edit scenario: fog gate/i });
    const file = new File(["starter image"], "ilyra.png", { type: "image/png" });

    await userEvent.upload(within(dialog).getByLabelText("Upload Captain Ilyra reference image file"), file);
    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/scenarios/scenario-1/character-starters/reference-image/upload",
        expect.anything()
      );
    });
    fireEvent.change(within(dialog).getByLabelText("Starter Captain Ilyra role"), {
      target: { value: "Harbor captain" }
    });
    const saveButton = within(dialog).getByRole("button", { name: "Save scenario" });
    await waitFor(() => expect(saveButton).toBeDisabled());
    expect(fetchMock.mock.calls.some(([path, init]) => (
      path === "/api/scenarios/scenario-1/definition"
      && Boolean((init as RequestInit | undefined)?.body)
    ))).toBe(false);
    await act(async () => {
      uploadResponse.resolve({
        ok: true,
        json: async () => uploadedPayload
      });
      await uploadResponse.promise;
    });

    expect(await within(dialog).findByAltText("Uploaded character reference image")).toBeInTheDocument();
    expect(within(dialog).getByLabelText("Starter Captain Ilyra role")).toHaveValue("Harbor captain");
    await waitFor(() => expect(saveButton).toBeEnabled());

    await userEvent.click(saveButton);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/scenario-1/definition", expect.anything()));
    const saveCall = fetchMock.mock.calls.find(([path, init]) => path === "/api/scenarios/scenario-1/definition" && (init as RequestInit | undefined)?.body);
    const savedBody = JSON.parse(String(saveCall?.[1].body));
    expect(savedBody.edit.character_starters[0]).toMatchObject({
      starter_id: "starter-ilyra",
      role: "Harbor captain",
      reference_image: expect.objectContaining({ id: "starter-ref-1" })
    });
  });

  it("blocks duplicate scenario section keys before saving", async () => {
    const scenario: Scenario = {
      scenario_id: "scenario-1",
      scenario_type: "full_roleplay",
      title: "Fog Gate",
      premise: "A gate in the fog.",
      player_role: "Keeper",
      opening_message: "The fog opens.",
      save_count: 0
    };
    const definitionPayload = {
      active_save_id: null,
      scenario: {
        ...scenario,
        player_character_name: "",
        content_sections: [
          ["tone_genre", "Fog mystery"],
          ["hidden_clock", "The gate opens at midnight."]
        ]
      }
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) return runtimeModel();
        if (path === "/api/scenarios") return { scenarios: [scenario] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return { save_id: "save-1", can_submit: true, reason: null, blocking_job_id: null, blocking_job_status: null };
        if (path === "/api/scenarios/scenario-1/definition") return definitionPayload;
        return { model: definitionPayload };
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /Scenarios/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Edit Fog Gate" }));
    const dialog = await screen.findByRole("dialog", { name: /edit scenario: fog gate/i });
    fireEvent.change(within(dialog).getByLabelText("Section key hidden_clock"), { target: { value: "tone_genre" } });
    await userEvent.click(within(dialog).getByRole("button", { name: "Save scenario" }));

    expect(await within(dialog).findByText("Section keys must be unique")).toBeInTheDocument();
    const postCall = fetchMock.mock.calls.find(([path, init]) => path === "/api/scenarios/scenario-1/definition" && (init as RequestInit | undefined)?.body);
    expect(postCall).toBeUndefined();
  });

  it("keeps scenario definition editing open when saving fails", async () => {
    const scenario = scenarioFixture({
      title: "Fog Gate",
      premise: "A gate in the fog.",
      player_role: "Keeper"
    });
    const definitionPayload = {
      active_save_id: null,
      scenario: {
        ...scenario,
        player_character_name: "Mara",
        content_sections: [["tone_genre", "Fog mystery"]]
      }
    };
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => Promise.resolve(
      path === "/api/scenarios/scenario-1/definition" && init?.method === "POST"
        ? { ok: false, status: 500, statusText: "Server Error", json: async () => ({ detail: "Scenario definition could not be saved." }) }
        : {
            ok: true,
            json: async () => {
              if (path.startsWith("/api/runtime")) return runtimeModel();
              if (path === "/api/scenarios") return { scenarios: [scenario] };
              if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
              if (path === "/api/settings") return modelSettingsPayload();
              if (path.startsWith("/api/chat/submission-status")) return { save_id: "save-1", can_submit: true, reason: null, blocking_job_id: null, blocking_job_status: null };
              if (path === "/api/scenarios/scenario-1/definition") return definitionPayload;
              return {};
            }
          }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /Scenarios/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Edit Fog Gate" }));
    const dialog = await screen.findByRole("dialog", { name: /edit scenario: fog gate/i });
    fireEvent.change(within(dialog).getByLabelText("Title"), { target: { value: "Fog Gate Revised" } });
    await userEvent.click(within(dialog).getByRole("button", { name: "Save scenario" }));

    expect(await within(dialog).findByText("Scenario definition could not be saved.")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: /edit scenario: fog gate/i })).toBeInTheDocument();
  });

  it("opens and closes mobile panel sheets from the dock", async () => {
    stubWorkbenchMedia(true);
    vi.stubGlobal("fetch", workbenchFetch([]));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Media" }));
    expect(screen.getByRole("dialog", { name: "Media" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(screen.queryByRole("dialog", { name: "Media" })).not.toBeInTheDocument();
    expect(screen.getByRole("textbox", { name: "Message" })).toBeInTheDocument();
  });

  it("uses the action picker instead of the composer when action choices are enabled", async () => {
    const model = runtimeModel({
      active_scenario_type: "full_roleplay",
      action_choices_enabled: true,
      action_choices: cyoaActionChoices()
    });
    vi.stubGlobal("fetch", workbenchFetch([], model));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("button", { name: "Open the brass door" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Write your own" })).toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Message" })).not.toBeInTheDocument();
  });

  it("closes mobile sheets with Escape and backdrop clicks while restoring focus", async () => {
    stubWorkbenchMedia(true);
    vi.stubGlobal("fetch", workbenchFetch([]));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const mediaButton = await screen.findByRole("button", { name: "Media" });
    await userEvent.click(mediaButton);
    let sheet = screen.getByRole("dialog", { name: "Media" });
    expect(within(sheet).getByRole("button", { name: "Close" })).toHaveFocus();

    await userEvent.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Media" })).not.toBeInTheDocument());
    expect(mediaButton).toHaveFocus();

    await userEvent.click(mediaButton);
    sheet = screen.getByRole("dialog", { name: "Media" });
    await userEvent.click(sheet);
    expect(screen.getByRole("dialog", { name: "Media" })).toBeInTheDocument();

    fireEvent.click(sheet.parentElement as HTMLElement);

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Media" })).not.toBeInTheDocument());
  });

  it("traps keyboard focus inside mobile sheets", async () => {
    stubWorkbenchMedia(true);
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Keep", active: true }]
    });
    vi.stubGlobal("fetch", workbenchFetch([], model));
    const { Workbench } = await import("./main");
    const user = userEvent.setup();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await user.click(await screen.findByRole("button", { name: "Library" }));
    const sheet = screen.getByRole("dialog", { name: "Library" });
    const close = within(sheet).getByRole("button", { name: "Close" });
    const sheetButtons = within(sheet).getAllByRole("button")
      .filter((button) => !(button as HTMLButtonElement).disabled);

    expect(close).toHaveFocus();
    await user.tab({ shift: true });
    expect(document.activeElement).toBe(sheetButtons[sheetButtons.length - 1]);
    await user.tab();
    expect(close).toHaveFocus();
  });

  it("renders accessible workbench resize handles with default dimensions", async () => {
    vi.stubGlobal("fetch", workbenchFetch([]));
    const { Workbench } = await import("./main");

    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect(await within(container).findByRole("separator", { name: "Resize left rail" })).toHaveAttribute("aria-valuenow", "278");
    expect(within(container).getByRole("separator", { name: "Resize right panel" })).toHaveAttribute("aria-valuenow", "370");
    const shell = container.querySelector(".app-shell") as HTMLElement;
    expect(shell.style.getPropertyValue("--left-rail-width")).toBe("278px");
    expect(shell.style.getPropertyValue("--right-panel-width")).toBe("370px");
  });

  it("uses shell settings for the workbench and loads full Settings on demand", async () => {
    const model = runtimeModel();
    const shellSettings = {
      pending_jobs_display_mode: {
        setting_key: "pending_jobs_display_mode",
        selected: "compact",
        options: ["compact", "expanded", "expanded_full"]
      }
    };
    const fullSettings = modelSettingsPayload({
      provider_cards: [
        {
          provider: "fake",
          enabled: true,
          has_api_key: true,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "Models available",
          last_error: null
        }
      ]
    });
    const fetchMock = vi.fn().mockImplementation((rawPath: string) => {
      const path = String(rawPath);
      return Promise.resolve({
        ok: true,
        json: async () => {
          if (path.startsWith("/api/runtime")) return model;
          if (path === "/api/scenarios") return { scenarios: [] };
          if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
          if (path === "/api/settings/shell") return shellSettings;
          if (path === "/api/settings/providers") {
            return {
              provider_cards: fullSettings.provider_cards,
              secret_storage_warning: fullSettings.secret_storage_warning
            };
          }
          if (path === "/api/settings?save_id=save-1") return fullSettings;
          if (path.startsWith("/api/chat/submission-status")) {
            return {
              save_id: model.active_save_id,
              can_submit: true,
              reason: null,
              blocking_job_id: null,
              blocking_job_status: null
            };
          }
          return {};
        }
      });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const fullSettingsReads = () => fetchMock.mock.calls.filter(([path]) => path === "/api/settings?save_id=save-1");
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/shell")).toBe(true));
    expect(fullSettingsReads()).toHaveLength(0);
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings")).toBe(false);

    await userEvent.click(await screen.findByRole("button", { name: "Settings" }));
    expect(await screen.findByRole("heading", { name: "Settings" })).toBeInTheDocument();
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/providers")).toBe(true));
    expect(fullSettingsReads()).toHaveLength(0);
    await userEvent.click(screen.getByRole("tab", { name: "Save" }));
    await waitFor(() => expect(fullSettingsReads()).toHaveLength(1));
  });

  it("resizes workbench segments with the keyboard and persists dimensions", async () => {
    vi.stubGlobal("fetch", workbenchFetch([]));
    const { Workbench } = await import("./main");

    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const shell = container.querySelector(".app-shell") as HTMLElement;
    const leftHandle = await within(container).findByRole("separator", { name: "Resize left rail" });
    const rightHandle = within(container).getByRole("separator", { name: "Resize right panel" });

    fireEvent.keyDown(leftHandle, { key: "ArrowRight" });
    expect(shell.style.getPropertyValue("--left-rail-width")).toBe("294px");

    fireEvent.keyDown(rightHandle, { key: "ArrowLeft" });
    expect(shell.style.getPropertyValue("--right-panel-width")).toBe("386px");

    fireEvent.keyDown(rightHandle, { key: "Home" });
    expect(shell.style.getPropertyValue("--right-panel-width")).toBe("300px");

    await waitFor(() => {
      expect(JSON.parse(window.localStorage.getItem("bragi-web:workbench-layout:v1") ?? "{}")).toMatchObject({
        leftRailWidth: 294,
        rightPanelWidth: 300
      });
    });
  });

  it("restores persisted workbench segment dimensions", async () => {
    window.localStorage.setItem("bragi-web:workbench-layout:v1", JSON.stringify({ leftRailWidth: 332, rightPanelWidth: 444 }));
    vi.stubGlobal("fetch", workbenchFetch([]));
    const { Workbench } = await import("./main");

    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await within(container).findByRole("separator", { name: "Resize left rail" });
    const shell = container.querySelector(".app-shell") as HTMLElement;
    expect(shell.style.getPropertyValue("--left-rail-width")).toBe("332px");
    expect(shell.style.getPropertyValue("--right-panel-width")).toBe("444px");
  });

  it("falls back to default workbench dimensions for invalid persisted layout data", async () => {
    window.localStorage.setItem("bragi-web:workbench-layout:v1", JSON.stringify({ leftRailWidth: "wide", rightPanelWidth: 9000 }));
    vi.stubGlobal("fetch", workbenchFetch([]));
    const { Workbench } = await import("./main");

    const { container } = render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await within(container).findByRole("separator", { name: "Resize left rail" });
    const shell = container.querySelector(".app-shell") as HTMLElement;
    expect(shell.style.getPropertyValue("--left-rail-width")).toBe("278px");
    expect(shell.style.getPropertyValue("--right-panel-width")).toBe("370px");
  });

  it("cancels chat jobs through the job and runtime cancel endpoints", async () => {
    installEventSourceDouble();
    const fetchMock = workbenchFetch([
      { id: "job-1", type: "chat_turn", save_id: "save-1", status: "running", result: null, error: null, created_at: 1 }
    ]);
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const cancelButtons = await screen.findAllByRole("button", { name: "Cancel Chat turn" });
    await userEvent.click(cancelButtons[cancelButtons.length - 1]);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/job-1/cancel?save_id=save-1", expect.objectContaining({ method: "POST" })));
    const chatCancel = fetchMock.mock.calls.find(([path]) => path === "/api/chat/cancel");
    expect(chatCancel).toBeTruthy();
    expect(JSON.parse(String(chatCancel?.[1].body))).toEqual({ save_id: "save-1" });
  });

  it("cancels non-chat jobs without runtime chat cancellation", async () => {
    installEventSourceDouble();
    const fetchMock = workbenchFetch([
      { id: "job-2", type: "image_generation", save_id: "save-1", status: "running", result: null, error: null, created_at: 1 }
    ]);
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const cancelButtons = await screen.findAllByRole("button", { name: "Cancel Generating image" });
    await userEvent.click(cancelButtons[cancelButtons.length - 1]);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/jobs/job-2/cancel?save_id=save-1", expect.objectContaining({ method: "POST" })));
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/chat/cancel")).toBe(false);
  });

  it("removes pending jobs when their watcher reaches a terminal state", async () => {
    const sources = installEventSourceDouble();
    const activeJobs = [{ id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 } satisfies Job];
    const model = runtimeModel();
    vi.stubGlobal("fetch", workbenchFetch(activeJobs, model));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect((await screen.findAllByText("Active jobs")).length).toBeGreaterThan(0);

    act(() => {
      activeJobs.splice(0);
      for (const source of sources) {
        source.dispatch("done", {
          id: "job-1",
          type: "chat_turn",
          status: "succeeded",
          result: runtimeModel({ status: "Turn complete" }),
          error: null
        });
      }
    });

    await waitFor(() => expect(screen.queryByLabelText("Pending jobs")).not.toBeInTheDocument());
  });

  it("removes pending jobs when fallback polling discovers a stale job id", async () => {
    const sources = installEventSourceDouble();
    const activeJobs = [{ id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 } satisfies Job];
    const model = runtimeModel();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: path !== "/api/jobs/job-1",
      status: path === "/api/jobs/job-1" ? 404 : 200,
      statusText: path === "/api/jobs/job-1" ? "Not Found" : "OK",
      json: async () => {
        if (path.startsWith("/api/runtime")) return model;
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: activeJobs };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return {
          save_id: model.active_save_id,
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        };
        if (path === "/api/jobs/job-1") return { detail: "Unknown job" };
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect((await screen.findAllByText("Active jobs")).length).toBeGreaterThan(0);

    activeJobs.splice(0);
    act(() => {
      for (const source of sources) source.dispatchNativeError();
    });

    await waitFor(() => expect(screen.queryByLabelText("Pending jobs")).not.toBeInTheDocument());
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/jobs/job-1")).toBe(true);
  });

  it("renders expanded pending job phases from live structured progress events", async () => {
    const sources = installEventSourceDouble();
    const activeJobs = [{ id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 } satisfies Job];
    const model = runtimeModel();
    vi.stubGlobal(
      "fetch",
      workbenchFetch(
        activeJobs,
        model,
        [],
        undefined,
        { pending_jobs_display_mode: { setting_key: "pending_jobs_display_mode", selected: "expanded", options: ["compact", "expanded", "expanded_full"] } }
      )
    );
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    const trays = await screen.findAllByLabelText("Pending jobs");
    const tray = trays[trays.length - 1];
    expect(within(tray).getByText("Chat turn")).toBeInTheDocument();

    act(() => {
      for (const source of sources) {
        source.dispatch("progress", { label: "Updating world state" });
      }
    });
    expect(within(tray).getByText("Updating world state")).toBeInTheDocument();
    expect(within(tray).queryByText("World state")).not.toBeInTheDocument();

    act(() => {
      for (const source of sources) {
        source.dispatch("progress", {
          jobs: [
            { name: "state", status: "running" },
            { name: "context", status: "pending" },
            { name: "characters", status: "pending" }
          ]
        });
      }
    });

    expect(within(tray).getByText("World state")).toBeInTheDocument();
    expect(within(tray).getByText("Context update")).toBeInTheDocument();
    expect(within(tray).getByText("Character cleanup")).toBeInTheDocument();
  });

  it("renders narrator draft job events as a transient chronicle message", async () => {
    const sources = installEventSourceDouble();
    const activeJobs = [{ id: "job-1", type: "chat_turn", status: "running", result: null, error: null, created_at: 1 } satisfies Job];
    const model = runtimeModel();
    vi.stubGlobal("fetch", workbenchFetch(activeJobs, model));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    expect((await screen.findAllByText("Active jobs")).length).toBeGreaterThan(0);
    const jobSources = () => sources.filter((source) => source.url.startsWith("/api/jobs/"));
    const dispatchDraft = (body: string) => {
      for (const source of jobSources()) {
        source.dispatch("narrator_draft", {
          message: {
            message_id: "pending-narrator-message",
            role: "narrator",
            speaker_name: "Narrator",
            body,
            markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: body }] }],
            actions: []
          }
        });
      }
    };

    act(() => {
      dispatchDraft("The bell");
    });

    expect(await screen.findByText("The bell")).toBeInTheDocument();

    act(() => {
      dispatchDraft("The bell answers.");
    });

    expect(await screen.findByText("The bell answers.")).toBeInTheDocument();

    act(() => {
      activeJobs.splice(0);
      model.chronicle = {
        messages: [{
          message_id: "narrator-1",
          role: "narrator",
          speaker_name: "Narrator",
          body: "The bell answers.",
          actions: [],
          markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: "The bell answers." }] }]
        }]
      };
      jobSources()[0].dispatch("done", {
        id: "job-1",
        type: "chat_turn",
        status: "succeeded",
        result: { ...model, status: "Turn complete" },
        error: null
      });
    });

    await waitFor(() => expect(screen.queryByLabelText("Pending jobs")).not.toBeInTheDocument());
    expect(screen.getAllByText("The bell answers.")).toHaveLength(1);
  });

  it("reorders saves when a completed chat job returns fresher save activity", async () => {
    const sources = installEventSourceDouble();
    const activeJobs = [
      { id: "job-1", type: "chat_turn", save_id: "save-1", status: "running", result: null, error: null, created_at: 1 } satisfies Job
    ];
    const initialModel = runtimeModel({
      active_save_id: "save-1",
      active_save_title: "Lantern Keep",
      saves: [
        {
          save_id: "save-1",
          title: "Lantern Keep",
          active: true,
          updated_at: "2026-05-01 00:00:00"
        },
        {
          save_id: "save-2",
          title: "Signal Tower",
          active: false,
          updated_at: "2026-05-02 00:00:00"
        }
      ],
      chronicle: {
        messages: [
          { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] }
        ]
      }
    });
    const updatedModel = runtimeModel({
      ...initialModel,
      saves: [
        {
          save_id: "save-1",
          title: "Lantern Keep",
          active: true,
          updated_at: "2026-05-03 00:00:00"
        },
        {
          save_id: "save-2",
          title: "Signal Tower",
          active: false,
          updated_at: "2026-05-02 00:00:00"
        }
      ],
      chronicle: {
        messages: [
          { message_id: "m1", role: "narrator", speaker_name: null, body: "The beacon waits.", actions: [] },
          { message_id: "m2", role: "player", speaker_name: "Keeper", body: "Light the beacon.", actions: [] },
          { message_id: "m3", role: "narrator", speaker_name: null, body: "The bell answers.", actions: [] }
        ]
      }
    });
    let runtime = initialModel;
    vi.stubGlobal("fetch", vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) return runtime;
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: activeJobs };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return {
          save_id: "save-1",
          can_submit: true,
          reason: null,
          blocking_job_id: null,
          blocking_job_status: null
        };
        return {};
      }
    })));
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByText("The beacon waits.")).toBeInTheDocument());
    expect(
      screen.getByRole("button", { name: "Load Signal Tower" })
        .compareDocumentPosition(screen.getByRole("button", { name: "Load Lantern Keep" }))
      & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();

    act(() => {
      runtime = updatedModel;
      activeJobs.splice(0);
      sources
        .find((source) => source.url.startsWith("/api/jobs/job-1/events"))
        ?.dispatch("done", {
          id: "job-1",
          type: "chat_turn",
          save_id: "save-1",
          status: "succeeded",
          result: updatedModel,
          error: null
        });
    });

    await waitFor(() => expect(screen.getByText("The bell answers.")).toBeInTheDocument());
    expect(
      screen.getByRole("button", { name: "Load Lantern Keep" })
        .compareDocumentPosition(screen.getByRole("button", { name: "Load Signal Tower" }))
      & Node.DOCUMENT_POSITION_FOLLOWING
    ).toBeTruthy();
  });

  it("extracts regenerated scenario section text from runtime model results", async () => {
    const { runtimeResultError, scenarioSectionResultText } = await import("./main");

    expect(scenarioSectionResultText("Raw section", "opening_message")).toBe("Raw section");
    expect(
      scenarioSectionResultText(
        {
          scenario_draft: {
            sections: [
              ["title", "Lantern Keep"],
              ["opening_message", "The beacon wakes."]
            ]
          }
        },
        "opening_message"
      )
    ).toBe("The beacon wakes.");
    expect(scenarioSectionResultText({ scenario_draft: { sections: [] } }, "opening_message")).toBeNull();
    expect(runtimeResultError({ error: "No scenario generation model preference configured" })).toBe(
      "No scenario generation model preference configured"
    );
  });

  it("creates manual scenarios, applies returned runtime, and refreshes an open scenario list", async () => {
    const initialModel = runtimeModel({
      saves: [],
      active_save_id: null,
      active_save_title: null,
      scenario_title: null,
      composer_enabled: false
    });
    const createdModel = runtimeModel({
      saves: [{ save_id: "save-manual", title: "Mist Run", active: true }],
      active_save_id: "save-manual",
      active_save_title: "Mist Run",
      scenario_title: "Mist Run"
    });
    let created = false;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) return created ? createdModel : initialModel;
        if (path === "/api/scenarios") {
          return {
            scenarios: created
              ? [scenarioFixture({ scenario_id: "scenario-manual", title: "Mist Run", premise: "The fog has teeth." })]
              : []
          };
        }
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path.startsWith("/api/chat/submission-status")) return { save_id: null, can_submit: false, reason: "no_save", blocking_job_id: null, blocking_job_status: null };
        if (path === "/api/scenarios/manual") {
          created = true;
          return createdModel;
        }
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /Scenarios/ }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/scenarios")).toHaveLength(1));

    await userEvent.click(await screen.findByRole("button", { name: "New scenario" }));
    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.type(within(dialog).getByLabelText("Title"), "Mist Run");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "The fog has teeth.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Keeper");
    await userEvent.type(within(dialog).getByLabelText("Player Character"), "Mara");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "The lamps hiss awake.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toEqual({
      scenario_type: "full_roleplay",
      scenario_types: ["full_roleplay"],
      action_choices_enabled: false,
      title: "Mist Run",
      premise: "The fog has teeth.",
      player_role: "Keeper",
      player_character_name: "Mara",
      player_character_profile: "",
      romance_options: "",
      magic_system: "",
      realms_and_places: "",
      factions_and_orders: "",
      myths_and_creatures: "",
      quest_stakes: "",
      technology_level: "",
      setting_scope: "",
      species_and_intelligences: "",
      factions_and_institutions: "",
      mission_stakes: "",
      mission_profile: "",
      crew_and_command: "",
      ship_or_base_status: "",
      exploration_target: "",
      unknown_intelligence: "",
      knowledge_state: "",
      translation_progress: "",
      discoveries_and_samples: "",
      hazards_and_escalation: "",
      expedition_goal: "",
      route_options: "",
      party_roster: "",
      resource_inventory: "",
      environmental_conditions: "",
      hazards_and_events: "",
      camp_status: "",
      travel_progress: "",
      loop_premise: "",
      reset_trigger: "",
      loop_duration: "",
      starting_state: "",
      objective: "",
      failure_conditions: "",
      baseline_world_state: "",
      loop_schedule: "",
      persistent_knowledge: "",
      persistence_exceptions: "",
      npc_memory_rules: "",
      current_loop_state: "",
      case_facts: "",
      suspects: "",
      clues: "",
      timeline: "",
      red_herrings: "",
      hidden_truth: "",
      case_status: "",
      target_location: "",
      objectives_and_stakes: "",
      crew_and_contacts: "",
      intel_and_access: "",
      security_model: "",
      alert_and_heat: "",
      loadout_and_tools: "",
      complications: "",
      extraction_routes: "",
      aftermath: "",
      political_arena: "",
      political_factions: "",
      major_npcs: "",
      central_conflict: "",
      secrets_and_leverage: "",
      reputation_and_standing: "",
      obligations_and_favors: "",
      alliances_and_rivalries: "",
      event_calendar: "",
      political_pressure: "",
      public_private_knowledge: "",
      settlement_profile: "",
      population_and_residents: "",
      resources_and_indicators: "",
      projects_and_facilities: "",
      threats_and_opportunities: "",
      calendar_and_deadlines: "",
      hunt_profile: "",
      target_profile: "",
      leads_and_clues: "",
      hunt_locations: "",
      rivals_and_factions: "",
      preparation_state: "",
      hunt_status: "",
      journey_profile: "",
      route_and_stops: "",
      traveling_party: "",
      transport_and_supplies: "",
      recurring_pressures: "",
      relationship_threads: "",
      journey_progress: "",
      trade_profile: "",
      cargo_inventory: "",
      markets_and_stops: "",
      contracts_and_debts: "",
      route_hazards: "",
      reputation_and_contacts: "",
      profit_and_loss: "",
      tone_genre: "",
      choice_style: "",
      opening_message: "The lamps hiss awake."
    });
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "New scenario" })).not.toBeInTheDocument());
    expect(await screen.findByRole("heading", { name: "Mist Run", level: 1 })).toBeInTheDocument();

    await waitFor(() => {
      expect(fetchMock.mock.calls.filter(([path]) => path === "/api/scenarios").length).toBeGreaterThanOrEqual(2);
    });
    expect(await screen.findByRole("button", { name: "Start Mist Run" })).toBeInTheDocument();
  });

  it("creates manual hybrid scenarios with fields from both genres", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-hybrid",
            active_save_title: "Starlit Hearts",
            active_scenario_type: "science_fiction_roleplay",
            scenario_title: "Starlit Hearts"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByRole("tab", { name: "Science Fiction" }));
    await userEvent.click(within(dialog).getByLabelText("Hybrid"));
    await userEvent.selectOptions(within(dialog).getByLabelText("Second genre"), "dating_sim");
    await userEvent.type(within(dialog).getByLabelText("Title"), "Starlit Hearts");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A diplomatic station turns courtship into first contact.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Envoy-pilot");
    await userEvent.type(within(dialog).getByLabelText("Technology Level"), "Near-future orbital habitat and alien translators.");
    await userEvent.type(within(dialog).getByLabelText("Romance Options"), "A xenolinguist, station marshal, and visiting envoy.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "The airlock opens on the first reception.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "science_fiction_roleplay",
      scenario_types: ["science_fiction_roleplay", "dating_sim"],
      title: "Starlit Hearts",
      premise: "A diplomatic station turns courtship into first contact.",
      player_role: "Envoy-pilot",
      technology_level: "Near-future orbital habitat and alien translators.",
      romance_options: "A xenolinguist, station marshal, and visiting envoy.",
      opening_message: "The airlock opens on the first reception."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "science_fiction_roleplay",
      active_save_id: "save-hybrid"
    }));
  });

  it("creates manual investigation mystery scenarios with case fields", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-mystery",
            active_save_title: "Broken Hours",
            active_scenario_type: "investigation_mystery",
            scenario_title: "Broken Hours"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByRole("tab", { name: "Investigation Mystery" }));
    await userEvent.type(within(dialog).getByLabelText("Title"), "Broken Hours");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A curator vanishes during a gala.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Lead investigator");
    await userEvent.type(within(dialog).getByLabelText("Case Facts"), "The east gallery was sealed.");
    await userEvent.type(within(dialog).getByLabelText("Suspects"), "Sera Holt has a false alibi.");
    await userEvent.type(within(dialog).getByLabelText("Clues"), "The watch log skips eight minutes.");
    await userEvent.type(within(dialog).getByLabelText("Timeline"), "Public alarm at 9:21; hidden lift movement at 9:12.");
    await userEvent.type(within(dialog).getByLabelText("Red Herrings"), "The bloody glove is from a mannequin.");
    await userEvent.type(within(dialog).getByLabelText("Hidden Truth"), "Sera hid the ledger in the lift.");
    await userEvent.type(within(dialog).getByLabelText("Case Status"), "Unresolved.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "Rain taps the museum glass.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "investigation_mystery",
      title: "Broken Hours",
      premise: "A curator vanishes during a gala.",
      player_role: "Lead investigator",
      case_facts: "The east gallery was sealed.",
      suspects: "Sera Holt has a false alibi.",
      clues: "The watch log skips eight minutes.",
      timeline: "Public alarm at 9:21; hidden lift movement at 9:12.",
      red_herrings: "The bloody glove is from a mannequin.",
      hidden_truth: "Sera hid the ledger in the lift.",
      case_status: "Unresolved.",
      opening_message: "Rain taps the museum glass."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "investigation_mystery",
      active_save_id: "save-mystery"
    }));
  });

  it("creates manual heist infiltration scenarios with security fields", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-heist",
            active_save_title: "Treaty Job",
            active_scenario_type: "heist_infiltration",
            scenario_title: "Skybank Treaty Job"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByRole("tab", { name: "Heist / Infiltration" }));
    await userEvent.type(within(dialog).getByLabelText("Title"), "Skybank Treaty Job");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A crew must steal a treaty from a floating bank.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Crew planner");
    await userEvent.type(within(dialog).getByLabelText("Target Location"), "Skybank vault above the storm moorings.");
    await userEvent.type(within(dialog).getByLabelText("Objectives And Stakes"), "Recover the treaty and avoid war.");
    await userEvent.type(within(dialog).getByLabelText("Crew And Contacts"), "Tavi runs locks; Venn is the inside clerk.");
    await userEvent.type(within(dialog).getByLabelText("Intel And Access"), "Guard shift changes at bell three.");
    await userEvent.type(within(dialog).getByLabelText("Security Model"), "Clockwork cameras and a silent alarm.");
    await userEvent.type(within(dialog).getByLabelText("Alert And Heat"), "Suspicion low; alarm inactive.");
    await userEvent.type(within(dialog).getByLabelText("Loadout And Tools"), "Forged badges, lockpicks, smoke pellets.");
    await userEvent.type(within(dialog).getByLabelText("Complications"), "A rival crew shadows the job.");
    await userEvent.type(within(dialog).getByLabelText("Extraction Routes"), "Primary storm skiff; fallback service stairs.");
    await userEvent.type(within(dialog).getByLabelText("Aftermath"), "Clean success keeps heat low.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "The skybank bell strikes three.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "heist_infiltration",
      title: "Skybank Treaty Job",
      premise: "A crew must steal a treaty from a floating bank.",
      player_role: "Crew planner",
      target_location: "Skybank vault above the storm moorings.",
      objectives_and_stakes: "Recover the treaty and avoid war.",
      crew_and_contacts: "Tavi runs locks; Venn is the inside clerk.",
      intel_and_access: "Guard shift changes at bell three.",
      security_model: "Clockwork cameras and a silent alarm.",
      alert_and_heat: "Suspicion low; alarm inactive.",
      loadout_and_tools: "Forged badges, lockpicks, smoke pellets.",
      complications: "A rival crew shadows the job.",
      extraction_routes: "Primary storm skiff; fallback service stairs.",
      aftermath: "Clean success keeps heat low.",
      opening_message: "The skybank bell strikes three."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "heist_infiltration",
      active_save_id: "save-heist"
    }));
  });

  it("creates manual political intrigue scenarios with faction fields", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-intrigue",
            active_save_title: "Ash Council",
            active_scenario_type: "political_intrigue",
            scenario_title: "Council of Ash"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByRole("tab", { name: "Political Intrigue" }));
    await userEvent.type(within(dialog).getByLabelText("Title"), "Council of Ash");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A council vote will decide the harbor.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Swing-vote envoy");
    await userEvent.type(within(dialog).getByLabelText("Political Arena"), "Harbor council chamber and public galleries.");
    await userEvent.type(within(dialog).getByLabelText("Political Factions"), "Guilds, Old Families, and dock unions.");
    await userEvent.type(within(dialog).getByLabelText("Major Npcs"), "Duchess Salen needs Mara's vote; Orro owes a favor.");
    await userEvent.type(within(dialog).getByLabelText("Central Conflict"), "A midnight vote can replace the regent.");
    await userEvent.type(within(dialog).getByLabelText("Secrets And Leverage"), "Only Mara knows Orro moved missing silver.");
    await userEvent.type(within(dialog).getByLabelText("Reputation And Standing"), "Mara is trusted by reformers.");
    await userEvent.type(within(dialog).getByLabelText("Obligations And Favors"), "Orro owes Mara one endorsement.");
    await userEvent.type(within(dialog).getByLabelText("Alliances And Rivalries"), "Reformers court Mara; old houses resist.");
    await userEvent.type(within(dialog).getByLabelText("Event Calendar"), "Dawn hearing, noon procession, midnight vote.");
    await userEvent.type(within(dialog).getByLabelText("Political Pressure"), "The midnight vote proceeds unless delayed.");
    await userEvent.type(within(dialog).getByLabelText("Public Private Knowledge"), "The public knows the vote is close; Mara knows the favor.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "The council bell rings.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "political_intrigue",
      title: "Council of Ash",
      premise: "A council vote will decide the harbor.",
      player_role: "Swing-vote envoy",
      political_arena: "Harbor council chamber and public galleries.",
      political_factions: "Guilds, Old Families, and dock unions.",
      major_npcs: "Duchess Salen needs Mara's vote; Orro owes a favor.",
      central_conflict: "A midnight vote can replace the regent.",
      secrets_and_leverage: "Only Mara knows Orro moved missing silver.",
      reputation_and_standing: "Mara is trusted by reformers.",
      obligations_and_favors: "Orro owes Mara one endorsement.",
      alliances_and_rivalries: "Reformers court Mara; old houses resist.",
      event_calendar: "Dawn hearing, noon procession, midnight vote.",
      political_pressure: "The midnight vote proceeds unless delayed.",
      public_private_knowledge: "The public knows the vote is close; Mara knows the favor.",
      opening_message: "The council bell rings."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "political_intrigue",
      active_save_id: "save-intrigue"
    }));
  });

  it("creates manual first-contact exploration scenarios with discovery fields", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-contact",
            active_save_title: "Europa Contact",
            active_scenario_type: "first_contact_exploration",
            scenario_title: "Songs Under Europa"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByRole("tab", { name: "First Contact / Exploration" }));
    await userEvent.type(within(dialog).getByLabelText("Title"), "Songs Under Europa");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A survey crew finds patterned signals under the ice.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Mission linguist");
    await userEvent.type(within(dialog).getByLabelText("Mission Profile"), "Survey the hidden ocean.");
    await userEvent.type(within(dialog).getByLabelText("Crew And Command"), "Commander Reyes leads the mission.");
    await userEvent.type(within(dialog).getByLabelText("Ship Or Base Status"), "Habitat heat is stable for 42 hours.");
    await userEvent.type(within(dialog).getByLabelText("Exploration Target"), "A black-water cavern beneath the ice.");
    await userEvent.type(within(dialog).getByLabelText("Unknown Intelligence"), "An unseen singer answers sonar.");
    await userEvent.type(within(dialog).getByLabelText("Knowledge State"), "Observed songs; unknown intent.");
    await userEvent.type(within(dialog).getByLabelText("Translation Progress"), "Three descending pulses may mean open water.");
    await userEvent.type(within(dialog).getByLabelText("Discoveries And Samples"), "Metallic spores remain quarantined.");
    await userEvent.type(within(dialog).getByLabelText("Hazards And Escalation"), "Thermal fissures are spreading.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "Blue light pulses beneath the ice.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "first_contact_exploration",
      title: "Songs Under Europa",
      premise: "A survey crew finds patterned signals under the ice.",
      player_role: "Mission linguist",
      mission_profile: "Survey the hidden ocean.",
      crew_and_command: "Commander Reyes leads the mission.",
      ship_or_base_status: "Habitat heat is stable for 42 hours.",
      exploration_target: "A black-water cavern beneath the ice.",
      unknown_intelligence: "An unseen singer answers sonar.",
      knowledge_state: "Observed songs; unknown intent.",
      translation_progress: "Three descending pulses may mean open water.",
      discoveries_and_samples: "Metallic spores remain quarantined.",
      hazards_and_escalation: "Thermal fissures are spreading.",
      opening_message: "Blue light pulses beneath the ice."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "first_contact_exploration",
      active_save_id: "save-contact"
    }));
  });

  it("creates manual survival expedition scenarios", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-expedition",
            active_save_title: "Whiteout Pass",
            active_scenario_type: "survival_expedition",
            scenario_title: "Whiteout Pass"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByRole("tab", { name: "Survival Expedition" }));
    await userEvent.type(within(dialog).getByLabelText("Title"), "Whiteout Pass");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A relief party crosses a storm-locked pass.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Expedition lead");
    await userEvent.type(within(dialog).getByLabelText("Expedition Goal"), "Reach Northwatch before the fever spreads.");
    await userEvent.type(within(dialog).getByLabelText("Route Options"), "Cliff road is faster; forest route has fuel.");
    await userEvent.type(within(dialog).getByLabelText("Party Roster"), "Mara guides two scouts and a medic.");
    await userEvent.type(within(dialog).getByLabelText("Resource Inventory"), "Food for nine days; medicine for three patients.");
    await userEvent.type(within(dialog).getByLabelText("Environmental Conditions"), "Late winter whiteout with ice-glazed slopes.");
    await userEvent.type(within(dialog).getByLabelText("Hazards And Events"), "Avalanches, frostbite, and wolf sign.");
    await userEvent.type(within(dialog).getByLabelText("Camp Status"), "Two canvas tents and one cracked stove.");
    await userEvent.type(within(dialog).getByLabelText("Travel Progress"), "0 of 80 miles; retreat is open for one day.");
    await userEvent.type(within(dialog).getByLabelText("Tone Genre"), "Gritty expedition survival.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "Snow erases the last wagon tracks.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "survival_expedition",
      title: "Whiteout Pass",
      premise: "A relief party crosses a storm-locked pass.",
      player_role: "Expedition lead",
      expedition_goal: "Reach Northwatch before the fever spreads.",
      route_options: "Cliff road is faster; forest route has fuel.",
      party_roster: "Mara guides two scouts and a medic.",
      resource_inventory: "Food for nine days; medicine for three patients.",
      environmental_conditions: "Late winter whiteout with ice-glazed slopes.",
      hazards_and_events: "Avalanches, frostbite, and wolf sign.",
      camp_status: "Two canvas tents and one cracked stove.",
      travel_progress: "0 of 80 miles; retreat is open for one day.",
      tone_genre: "Gritty expedition survival.",
      opening_message: "Snow erases the last wagon tracks."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "survival_expedition",
      active_save_id: "save-expedition"
    }));
  });

  it("creates manual merchant trade route scenarios", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-trade",
            active_save_title: "Ledger Road",
            active_scenario_type: "merchant_trade_route",
            scenario_title: "Ledger Road"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByRole("tab", { name: "Merchant / Trade Route" }));
    await userEvent.type(within(dialog).getByLabelText("Title"), "Ledger Road");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A caravan must turn debt into profit.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Caravan factor");
    await userEvent.type(within(dialog).getByLabelText("Trade Profile"), "Run cedar oil from Kesh Gate to Red Harbor.");
    await userEvent.type(within(dialog).getByLabelText("Cargo Inventory"), "Cedar oil: 20 jars.");
    await userEvent.type(within(dialog).getByLabelText("Markets And Stops"), "Red Harbor needs oil.");
    await userEvent.type(within(dialog).getByLabelText("Contracts And Debts"), "Deliver ten jars in twelve days.");
    await userEvent.type(within(dialog).getByLabelText("Route Hazards"), "Tariff patrols and bridge bandits.");
    await userEvent.type(within(dialog).getByLabelText("Reputation And Contacts"), "Trusted by Kesh brokers.");
    await userEvent.type(within(dialog).getByLabelText("Profit And Loss"), "One lost crate erases profit.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "The creditor stamps the contract.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "merchant_trade_route",
      title: "Ledger Road",
      premise: "A caravan must turn debt into profit.",
      player_role: "Caravan factor",
      trade_profile: "Run cedar oil from Kesh Gate to Red Harbor.",
      cargo_inventory: "Cedar oil: 20 jars.",
      markets_and_stops: "Red Harbor needs oil.",
      contracts_and_debts: "Deliver ten jars in twelve days.",
      route_hazards: "Tariff patrols and bridge bandits.",
      reputation_and_contacts: "Trusted by Kesh brokers.",
      profit_and_loss: "One lost crate erases profit.",
      opening_message: "The creditor stamps the contract."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "merchant_trade_route",
      active_save_id: "save-trade"
    }));
  });

  it("creates manual time loop scenarios with reset and persistence fields", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-loop",
            active_save_title: "Bell Loop",
            active_scenario_type: "time_loop",
            scenario_title: "Bellwether Day"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByRole("tab", { name: "Time Loop" }));
    await userEvent.type(within(dialog).getByLabelText("Title"), "Bellwether Day");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A harbor festival repeats until the bell is saved.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Loop-aware archivist");
    await userEvent.type(within(dialog).getByLabelText("Loop Premise"), "The same festival day repeats.");
    await userEvent.type(within(dialog).getByLabelText("Reset Trigger"), "The drowned bell tolls at midnight.");
    await userEvent.type(within(dialog).getByLabelText("Loop Duration"), "Twenty-four hours.");
    await userEvent.type(within(dialog).getByLabelText("Starting State"), "Mara wakes in the archive loft.");
    await userEvent.type(within(dialog).getByLabelText("Objective"), "Prevent the bell from sinking.");
    await userEvent.type(within(dialog).getByLabelText("Failure Conditions"), "The bell sinks or midnight arrives.");
    await userEvent.type(within(dialog).getByLabelText("Baseline World State"), "The harbor resets to dawn.");
    await userEvent.type(within(dialog).getByLabelText("Loop Schedule"), "09:00 parade; 23:45 sabotage.");
    await userEvent.type(within(dialog).getByLabelText("Persistent Knowledge"), "Tower code persists for the player.");
    await userEvent.type(within(dialog).getByLabelText("Persistence Exceptions"), "A salt mark persists.");
    await userEvent.type(within(dialog).getByLabelText("Npc Memory Rules"), "NPCs reset unless excepted.");
    await userEvent.type(within(dialog).getByLabelText("Current Loop State"), "Loop 1, dawn phase.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "The same bell rings dawn again.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "time_loop",
      title: "Bellwether Day",
      premise: "A harbor festival repeats until the bell is saved.",
      player_role: "Loop-aware archivist",
      loop_premise: "The same festival day repeats.",
      reset_trigger: "The drowned bell tolls at midnight.",
      loop_duration: "Twenty-four hours.",
      starting_state: "Mara wakes in the archive loft.",
      objective: "Prevent the bell from sinking.",
      failure_conditions: "The bell sinks or midnight arrives.",
      baseline_world_state: "The harbor resets to dawn.",
      loop_schedule: "09:00 parade; 23:45 sabotage.",
      persistent_knowledge: "Tower code persists for the player.",
      persistence_exceptions: "A salt mark persists.",
      npc_memory_rules: "NPCs reset unless excepted.",
      current_loop_state: "Loop 1, dawn phase.",
      opening_message: "The same bell rings dawn again."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "time_loop",
      active_save_id: "save-loop"
    }));
  });

  it("creates manual scenarios with action choices and choice style guidance", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/manual"
        ? runtimeModel({
            active_save_id: "save-cyoa",
            active_save_title: "Clockwork Labyrinth",
            active_scenario_type: "full_roleplay",
            action_choices_enabled: true,
            scenario_title: "Clockwork Labyrinth"
          })
        : {}
    }));
    const onRuntimeChanged = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = await screen.findByRole("dialog", { name: "New scenario" });
    await userEvent.click(within(dialog).getByLabelText("Action choices"));
    await userEvent.type(within(dialog).getByLabelText("Title"), "Clockwork Labyrinth");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "A city-sized machine wakes under the rain.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Lost machinist");
    await userEvent.type(within(dialog).getByLabelText("Choice Style"), "Four concise, risky actions.");
    await userEvent.type(within(dialog).getByLabelText("Opening Message"), "The gears begin to turn.");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/manual", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/manual");
    expect(JSON.parse(String(createCall?.[1].body))).toMatchObject({
      scenario_type: "full_roleplay",
      action_choices_enabled: true,
      title: "Clockwork Labyrinth",
      premise: "A city-sized machine wakes under the rain.",
      player_role: "Lost machinist",
      choice_style: "Four concise, risky actions.",
      opening_message: "The gears begin to turn."
    });
    expect(onRuntimeChanged).toHaveBeenCalledWith(expect.objectContaining({
      active_scenario_type: "full_roleplay",
      action_choices_enabled: true,
      active_save_id: "save-cyoa"
    }));
  });

  it("keeps manual scenario creation open when submission fails", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/scenarios/manual"
        ? { ok: false, status: 400, statusText: "Bad Request", json: async () => ({ detail: "Manual scenario could not be created." }) }
        : { ok: true, json: async () => ({}) }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const onClose = vi.fn();
    const onRuntimeChanged = vi.fn();
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={onClose} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const dialog = screen.getByRole("dialog", { name: "New scenario" });
    await userEvent.type(within(dialog).getByLabelText("Title"), "Mist Run");
    await userEvent.type(within(dialog).getByLabelText("Premise"), "The fog has teeth.");
    await userEvent.type(within(dialog).getByLabelText("Player Role"), "Keeper");
    await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

    expect(await within(dialog).findByText("Manual scenario could not be created.")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "New scenario" })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: "Create" })).toBeEnabled();
    expect(onClose).not.toHaveBeenCalled();
    expect(onRuntimeChanged).not.toHaveBeenCalled();
  });

  it("resets scenario draft textareas when a new draft replaces the old one", async () => {
    const sources = installEventSourceDouble();
    const oldDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "old seed",
      source_metadata: [],
      sections: [
        ["title", "Old Draft"],
        ["premise", "Old premise"],
        ["player_character_name", "Mara"],
        ["player_role", "Keeper"]
      ] as [string, string][]
    };
    const newDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "new seed",
      source_metadata: [],
      sections: [
        ["title", "New Draft"],
        ["premise", "New premise"],
        ["player_character_name", "Iris"],
        ["player_role", "Cartographer"]
      ] as [string, string][]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/scenarios/draft") {
          return { id: "job-draft", type: "scenario_draft", status: "queued", result: null, error: null };
        }
        if (path === "/api/scenarios/draft/save") return runtimeModel();
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel({ scenario_draft: oldDraft })} onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("tab", { name: "AI draft" }));
    const title = screen.getByLabelText("Title");
    await userEvent.clear(title);
    await userEvent.type(title, "Locally edited old draft");
    await userEvent.type(
      screen.getByPlaceholderText(
        "Describe the genre, premise, player role, tone, and visible opening narration. Leave room for the world to emerge in play.",
      ),
      "new idea",
    );
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(sources).toHaveLength(1));
    act(() => {
      sources[0].dispatch("progress", { status_text: "Drafting new scenario" });
    });
    expect(await screen.findByRole("status")).toHaveTextContent("Drafting new scenario");

    act(() => {
      sources[0].dispatch("done", {
        id: "job-draft",
        type: "scenario_draft",
        status: "succeeded",
        result: runtimeModel({ scenario_draft: newDraft }),
        error: null
      });
    });

    await waitFor(() => expect(screen.getByLabelText("Title")).toHaveValue("New Draft"));
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /save draft/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/draft/save", expect.anything()));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/draft/save");
    expect(JSON.parse(String(saveCall?.[1].body)).sections).toMatchObject({
      title: "New Draft",
      premise: "New premise",
      player_character_name: "Iris",
      player_role: "Cartographer"
    });
  });

  it("prefills the AI draft seed from a reusable scenario prompt", async () => {
    const { ScenarioDialog } = await import("./main");
    const staleDraft: ScenarioDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "old seed",
      source_metadata: [],
      sections: [["title", "Old Draft"]]
    };

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog
          model={runtimeModel({ scenario_draft: staleDraft })}
          initialMode="draft"
          initialDraftPrefill={{
            scenario_type: "full_roleplay",
            scenario_types: ["full_roleplay"],
            action_choices_enabled: false,
            seed: "A bell tower that only answers at low tide."
          }}
          onClose={vi.fn()}
          onRuntimeChanged={vi.fn()}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    expect(screen.getByRole("textbox", { name: "Scenario seed" })).toHaveValue(
      "A bell tower that only answers at low tide.",
    );
    expect(screen.queryByDisplayValue("Old Draft")).not.toBeInTheDocument();
  });

  it("renders scenario draft progress as neutral status text", async () => {
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/draft"
        ? { id: "job-draft", type: "scenario_draft", status: "queued", result: null, error: null }
        : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("tab", { name: "AI draft" }));
    await userEvent.type(
      screen.getByPlaceholderText(
        "Describe the genre, premise, player role, tone, and visible opening narration. Leave room for the world to emerge in play.",
      ),
      "new idea",
    );
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(sources).toHaveLength(1));
    act(() => {
      sources[0].dispatch("progress", {
        section_id: "opening_message",
        status: "running",
        completed_count: 1,
        total_count: 4
      });
    });

    expect(await screen.findByRole("status")).toHaveTextContent("Opening Message: running 1/4");
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("clears scenario draft progress and shows an error when the draft job fails", async () => {
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/draft"
        ? { id: "job-draft", type: "scenario_draft", status: "queued", result: null, error: null }
        : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("tab", { name: "AI draft" }));
    await userEvent.type(
      screen.getByPlaceholderText(
        "Describe the genre, premise, player role, tone, and visible opening narration. Leave room for the world to emerge in play.",
      ),
      "new idea",
    );
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(sources).toHaveLength(1));
    act(() => {
      sources[0].dispatch("progress", { status_text: "Drafting opening scene" });
    });
    expect(await screen.findByRole("status")).toHaveTextContent("Drafting opening scene");

    act(() => {
      sources[0].dispatch("done", {
        id: "job-draft",
        type: "scenario_draft",
        status: "failed",
        result: null,
        error: "No scenario draft model is configured."
      });
    });

    expect(await screen.findByRole("alert")).toHaveTextContent("No scenario draft model is configured.");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate" })).toBeEnabled();
  });

  it("resets scenario draft busy state when job creation fails", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/log/client"
        ? { ok: true, json: async () => ({}) }
        : {
            ok: false,
            status: 429,
            statusText: "Too Many Requests",
            json: async () => ({
              detail: "Too many active jobs; wait for one to finish before starting another."
            })
          }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("tab", { name: "AI draft" }));
    await userEvent.type(
      screen.getByPlaceholderText(
        "Describe the genre, premise, player role, tone, and visible opening narration. Leave room for the world to emerge in play.",
      ),
      "new idea",
    );
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));

    expect(await screen.findByText("Too many active jobs; wait for one to finish before starting another.")).toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate" })).toBeEnabled();
  });

  it("closes scenario draft watchers when the dialog unmounts", async () => {
    const sources = installEventSourceDouble();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/draft"
        ? { id: "job-draft", type: "scenario_draft", status: "queued", result: null, error: null }
        : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    const view = render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel()} onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("tab", { name: "AI draft" }));
    await userEvent.type(
      screen.getByPlaceholderText(
        "Describe the genre, premise, player role, tone, and visible opening narration. Leave room for the world to emerge in play.",
      ),
      "new idea",
    );
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(sources).toHaveLength(1));
    expect(sources[0].closed).toBe(false);

    view.unmount();

    expect(sources[0].closed).toBe(true);
    expect(sources[0].closeCalls).toBe(1);
  });

  it("sends current draft context when regenerating a scenario section", async () => {
    const sources = installEventSourceDouble();
    const draft: ScenarioDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "fog seed",
      source_metadata: [],
      sections: [
        ["title", "Fog Gate"],
        ["premise", "Old premise"]
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/draft/section"
        ? { id: "job-section", type: "scenario_section", status: "queued", result: null, error: null }
        : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel({ scenario_draft: draft })} initialMode="draft" onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const premiseRow = screen.getByLabelText("Premise").closest("label");
    expect(premiseRow).not.toBeNull();
    await userEvent.click(within(premiseRow as HTMLElement).getByRole("button", { name: /regenerate/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/draft/section", expect.anything()));
    const regenerateCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/draft/section");
    expect(JSON.parse(String(regenerateCall?.[1].body))).toEqual({
      scenario_type: "full_roleplay",
      scenario_types: ["full_roleplay"],
      seed: "fog seed",
      section_id: "premise",
      action_choices_enabled: false,
      sections: {
        title: "Fog Gate",
        premise: "Old premise"
      }
    });

    act(() => {
      sources[0].dispatch("done", {
        id: "job-section",
        type: "scenario_section",
        status: "succeeded",
        result: {
          scenario_draft: {
            sections: [
              ["title", "Fog Gate"],
              ["premise", "Fresh premise"]
            ]
          }
        },
        error: null
      });
    });

    await waitFor(() => expect(screen.getByLabelText("Premise")).toHaveValue("Fresh premise"));
  });

  it("surfaces scenario section regeneration failures", async () => {
    const sources = installEventSourceDouble();
    const draft: ScenarioDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "fog seed",
      source_metadata: [],
      sections: [
        ["title", "Fog Gate"],
        ["premise", "Old premise"]
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/draft/section"
        ? { id: "job-section", type: "scenario_section", status: "queued", result: null, error: null }
        : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel({ scenario_draft: draft })} initialMode="draft" onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const premiseRow = screen.getByLabelText("Premise").closest("label");
    expect(premiseRow).not.toBeNull();
    await userEvent.click(within(premiseRow as HTMLElement).getByRole("button", { name: /regenerate/i }));

    await waitFor(() => expect(sources).toHaveLength(1));
    act(() => {
      sources[0].dispatch("done", {
        id: "job-section",
        type: "scenario_section",
        status: "failed",
        result: null,
        error: "No scenario section model is configured."
      });
    });

    expect(await screen.findByText("No scenario section model is configured.")).toBeInTheDocument();
    expect(screen.getByLabelText("Premise")).toHaveValue("Old premise");
  });

  it("surfaces scenario section regeneration request failures", async () => {
    const draft: ScenarioDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "fog seed",
      source_metadata: [],
      sections: [
        ["title", "Fog Gate"],
        ["premise", "Old premise"]
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/scenarios/draft/section"
        ? { ok: false, status: 503, statusText: "Service Unavailable", json: async () => ({ detail: "Scenario section could not be queued." }) }
        : { ok: true, json: async () => ({}) }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel({ scenario_draft: draft })} initialMode="draft" onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const premiseRow = screen.getByLabelText("Premise").closest("label");
    expect(premiseRow).not.toBeNull();
    await userEvent.click(within(premiseRow as HTMLElement).getByRole("button", { name: /regenerate/i }));

    expect(await screen.findByText("Scenario section could not be queued.")).toBeInTheDocument();
    expect(screen.getByLabelText("Premise")).toHaveValue("Old premise");
    expect(within(premiseRow as HTMLElement).getByRole("button", { name: /regenerate/i })).toBeEnabled();
  });

  it("closes replacement scenario section watchers and ignores stale section completions", async () => {
    const sources = installEventSourceDouble();
    let sectionJobs = 0;
    const draft: ScenarioDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "section seed",
      source_metadata: [],
      sections: [
        ["title", "Old Title"],
        ["premise", "Old premise"]
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/scenarios/draft/section") {
          sectionJobs += 1;
          return { id: `job-section-${sectionJobs}`, type: "scenario_section", status: "queued", result: null, error: null };
        }
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel({ scenario_draft: draft })} initialMode="draft" onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const regenerateTitle = screen.getAllByRole("button", { name: /regenerate/i })[0];
    await userEvent.click(regenerateTitle);
    await waitFor(() => expect(sources).toHaveLength(1));
    await userEvent.click(regenerateTitle);
    await waitFor(() => expect(sources).toHaveLength(2));

    expect(sources[0].closed).toBe(true);
    expect(sources[1].closed).toBe(false);

    act(() => {
      sources[0].dispatch("done", {
        id: "job-section-1",
        type: "scenario_section",
        status: "succeeded",
        result: "Stale Title",
        error: null
      });
      sources[1].dispatch("done", {
        id: "job-section-2",
        type: "scenario_section",
        status: "succeeded",
        result: "Fresh Title",
        error: null
      });
    });

    await waitFor(() => expect(screen.getByLabelText("Title")).toHaveValue("Fresh Title"));
    expect(screen.getByLabelText("Premise")).toHaveValue("Old premise");
  });

  it("shows continuation-only draft sections in an editable group", async () => {
    const draft: ScenarioDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "continuation seed",
      source_metadata: [["origin", "save_continuation"]],
      sections: [
        ["title", "Chapter Two"],
        ["premise", "The next bell debt begins."],
        ["player_character_name", "Mara"],
        ["player_role", "Harbor warden"],
        ["tone_genre", "Nautical noir"],
        ["current_scene", "Mara is negotiating under the quay."],
        ["characters", "Mara keeps her clipped voice and debt to Ren."],
        ["opening_message", "The bell tolls once."]
      ] as [string, string][]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/draft/save" ? runtimeModel() : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel({ scenario_draft: draft })} initialMode="draft" onClose={vi.fn()} onRuntimeChanged={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    expect(screen.getByText("Continuity")).toBeInTheDocument();
    expect(screen.getByLabelText("Current Scene")).toHaveValue("Mara is negotiating under the quay.");
    expect(screen.getByLabelText("Characters")).toHaveValue("Mara keeps her clipped voice and debt to Ren.");
    await userEvent.clear(screen.getByLabelText("Current Scene"));
    await userEvent.type(screen.getByLabelText("Current Scene"), "Mara stands before the opened bell.");
    await userEvent.click(screen.getByRole("button", { name: /save draft/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/draft/save", expect.anything()));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/draft/save");
    expect(JSON.parse(String(saveCall?.[1].body))).toMatchObject({
      sections: {
        current_scene: "Mara stands before the opened bell.",
        characters: "Mara keeps her clipped voice and debt to Ren."
      },
      source_metadata: { origin: "save_continuation" }
    });
  });

  it("keeps scenario draft open and editable when saving fails", async () => {
    const draft: ScenarioDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "save seed",
      source_metadata: [],
      sections: [
        ["title", "Fog Gate"],
        ["premise", "Old premise"],
        ["player_role", "Keeper"]
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/scenarios/draft/save"
        ? { ok: false, status: 500, statusText: "Server Error", json: async () => ({ detail: "Draft could not be saved." }) }
        : { ok: true, json: async () => ({}) }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const onClose = vi.fn();
    const onRuntimeChanged = vi.fn();
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog model={runtimeModel({ scenario_draft: draft })} initialMode="draft" onClose={onClose} onRuntimeChanged={onRuntimeChanged} runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const premise = screen.getByLabelText("Premise");
    await userEvent.clear(premise);
    await userEvent.type(premise, "Fresh premise");
    await userEvent.click(screen.getByRole("button", { name: /save draft/i }));

    expect(await screen.findByText("Draft could not be saved.")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "New scenario" })).toBeInTheDocument();
    expect(screen.getByLabelText("Premise")).toHaveValue("Fresh premise");
    expect(screen.getByRole("button", { name: /save draft/i })).toBeEnabled();
    expect(onClose).not.toHaveBeenCalled();
    expect(onRuntimeChanged).not.toHaveBeenCalled();
  });

  it("hides draft initial media generation for child users", async () => {
    const draft: ScenarioDraft = {
      scenario_type: "dating_sim",
      regeneration_seed: "portrait seed",
      source_metadata: [],
      sections: [
        ["title", "Sigil Keeper"],
        ["character_name", "Sigil Keeper"],
        ["opening_message", "The sigil flares."]
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/scenarios/draft/save"
        ? runtimeModel({
            active_scenario_type: "dating_sim",
            chronicle: {
              messages: [
                {
                  message_id: "narrator-1",
                  role: "narrator",
                  speaker_name: null,
                  body: "The sigil flares.",
                  actions: []
                }
              ]
            }
          })
        : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { ScenarioDialog } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ScenarioDialog
          model={runtimeModel({ scenario_draft: draft })}
          initialMode="draft"
          currentUser={{ id: "child-1", username: "Ilyra", role: "child", status: "active" }}
          onClose={vi.fn()}
          onRuntimeChanged={vi.fn()}
          runJob={runJob}
        />
      </QueryClientProvider>
    );

    expect(screen.queryByLabelText("Generate character reference image")).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /save draft/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/draft/save", expect.anything()));
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/media/initial")).toBe(false);
    expect(runJob).not.toHaveBeenCalled();
  });

  it("starts continuation draft jobs from the active save with optional instructions", async () => {
    const sources = installEventSourceDouble();
    const draft: ScenarioDraft = {
      scenario_type: "full_roleplay",
      regeneration_seed: "continuation seed",
      source_metadata: [["origin", "save_continuation"]],
      sections: [
        ["title", "Chapter Two"],
        ["premise", "The next bell debt begins."],
        ["player_character_name", "Mara"],
        ["player_role", "Harbor warden"]
      ] as [string, string][]
    };
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Run", active: true }]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) return model;
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return { save_id: "save-1", can_submit: true, reason: null, blocking_job_id: null, blocking_job_status: null };
        if (path === "/api/scenarios/continuation-draft") return { id: "job-chapter", type: "scenario_draft", status: "queued", result: null, error: null };
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "New chapter from current save" }));

    const dialog = await screen.findByRole("dialog", { name: "New chapter from current save" });
    expect(within(dialog).getByText("Lantern Run")).toBeInTheDocument();
    await userEvent.type(
      within(dialog).getByLabelText("Start instructions"),
      "Characters are going to bed, start the next chapter the following morning."
    );
    await userEvent.click(within(dialog).getByRole("button", { name: "Start chapter draft" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/continuation-draft", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/continuation-draft");
    expect(JSON.parse(String(createCall?.[1].body))).toEqual({
      save_id: "save-1",
      chapter_start_instructions: "Characters are going to bed, start the next chapter the following morning."
    });
    const jobSources = () => sources.filter((source) => source.url.startsWith("/api/jobs/"));
    await waitFor(() => expect(jobSources()).toHaveLength(1));
    act(() => {
      jobSources()[0].dispatch("done", {
        id: "job-chapter",
        type: "scenario_draft",
        status: "succeeded",
        result: runtimeModel({ scenario_draft: draft }),
        error: null
      });
    });

    expect(await screen.findByRole("heading", { name: "New scenario" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "AI draft" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByLabelText("Title")).toHaveValue("Chapter Two");
  });

  it("queues continuation drafts with blank chapter instructions", async () => {
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Run", active: true }]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) return model;
        if (path === "/api/scenarios") return { scenarios: [] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings") return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) return { save_id: "save-1", can_submit: true, reason: null, blocking_job_id: null, blocking_job_status: null };
        if (path === "/api/scenarios/continuation-draft") return { id: "job-chapter", type: "scenario_draft", status: "queued", result: null, error: null };
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "New chapter from current save" }));
    const dialog = await screen.findByRole("dialog", { name: "New chapter from current save" });
    await userEvent.click(within(dialog).getByRole("button", { name: "Start chapter draft" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/continuation-draft", expect.anything()));
    const createCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/continuation-draft");
    expect(JSON.parse(String(createCall?.[1].body))).toEqual({
      save_id: "save-1",
      chapter_start_instructions: ""
    });
  });

  it("shows initial media failures and re-enables the action", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/media/initial"
        ? { ok: false, status: 500, statusText: "Server Error", json: async () => ({ detail: "Initial media failed." }) }
        : { ok: true, json: async () => ({}) }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const { MediaPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            chronicle: {
              messages: [
                { message_id: "narrator-1", role: "narrator", speaker_name: null, body: "The beacon wakes.", actions: [] }
              ]
            },
            media: { latest_scene_image: null, image_history: [], media_history: [] }
          })}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    const generate = screen.getByRole("button", { name: "Generate opening image" });
    await userEvent.click(generate);

    expect(await screen.findByText("Initial media failed.")).toBeInTheDocument();
    expect(generate).toBeEnabled();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/media/initial",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ message_id: "narrator-1", save_id: "save-1" })
      })
    );
  });

  it("moves focus into media previews and restores it after Escape closes", async () => {
    const { ImagePreview } = await import("./main");
    const user = userEvent.setup();

    function MediaPreviewHarness() {
      const [open, setOpen] = React.useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>Open media</button>
          {open ? (
            <ImagePreview
              asset={{
                id: "media-1",
                source_message_id: "message-1",
                type: "image",
                mime_type: "image/png",
                prompt_preview: "A beacon",
                status: "succeeded",
                created_at: null
              }}
              activeSaveId="save-1"
              onClose={() => setOpen(false)}
              runJob={vi.fn()}
            />
          ) : null}
        </>
      );
    }

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPreviewHarness />
      </QueryClientProvider>
    );

    const opener = screen.getByRole("button", { name: "Open media" });
    opener.focus();
    await user.click(opener);
    const dialog = screen.getByRole("dialog", { name: "Image · succeeded" });

    await waitFor(() => expect(within(dialog).getByLabelText("Close")).toHaveFocus());
    await user.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Image · succeeded" })).not.toBeInTheDocument());
    expect(opener).toHaveFocus();
  });

  it("moves focus into animation prompts and restores it after Escape closes", async () => {
    const { ImagePreview } = await import("./main");
    const user = userEvent.setup();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            source_media_asset_id: null,
            type: "image",
            mime_type: "image/png",
            prompt_preview: "A beacon",
            status: "succeeded",
            created_at: null,
            can_animate: true
          }}
          activeSaveId="save-1"
          onClose={vi.fn()}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    const animate = screen.getByRole("button", { name: "Animate this" });
    animate.focus();
    await user.click(animate);
    const animationDialog = screen.getByRole("dialog", { name: "Animate image" });

    await waitFor(() => expect(within(animationDialog).getByLabelText("Close")).toHaveFocus());
    await user.keyboard("{Escape}");

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Animate image" })).not.toBeInTheDocument());
    expect(screen.getByRole("dialog", { name: "Image · succeeded" })).toBeInTheDocument();
    expect(animate).toHaveFocus();
  });

  it("resets media mutation modals when mutation permission or the active save changes", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({ media_asset_id: "media-1", prompt: "A beacon" }) }));
    const { ImagePreview } = await import("./main");
    const asset = {
      id: "media-1",
      source_message_id: "message-1",
      type: "image",
      mime_type: "image/png",
      provider: "fake",
      model: "fake-image",
      source_message: "The beacon wakes.",
      prompt_preview: "A beacon",
      status: "succeeded",
      created_at: null,
      can_animate: true
    };
    const client = new QueryClient();
    const preview = (canMutate: boolean, activeSaveId = "save-1") => (
      <QueryClientProvider client={client}>
        <ImagePreview asset={asset} activeSaveId={activeSaveId} canMutate={canMutate} onClose={vi.fn()} runJob={vi.fn()} />
      </QueryClientProvider>
    );
    const { rerender } = render(preview(true));

    await userEvent.click(screen.getByRole("button", { name: "Regenerate with edits" }));
    expect(await screen.findByRole("dialog", { name: "Regenerate with edits" })).toBeInTheDocument();
    rerender(preview(false));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Regenerate with edits" })).not.toBeInTheDocument());
    rerender(preview(true));
    expect(screen.queryByRole("dialog", { name: "Regenerate with edits" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Animate this" }));
    expect(screen.getByRole("dialog", { name: "Animate image" })).toBeInTheDocument();
    rerender(preview(true, "save-2"));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Animate image" })).not.toBeInTheDocument());

    await userEvent.click(screen.getByRole("button", { name: "Delete" }));
    expect(screen.getByRole("dialog", { name: "Delete image?" })).toBeInTheDocument();
    rerender(preview(false, "save-2"));
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Delete image?" })).not.toBeInTheDocument());
  });

  it("regenerates previewed media from an edited raw prompt", async () => {
    const job: Job = {
      id: "job-regenerate",
      type: "image_generation",
      status: "queued",
      result: null,
      error: null,
      created_at: 1,
      save_id: "save-1"
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/media/media-1/prompt?save_id=save-1") {
          return {
            media_asset_id: "media-1",
            prompt: "A beacon above a storm sea"
          };
        }
        if (path === "/api/media/media-1/regenerate") return job;
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const onClose = vi.fn();
    const { ImagePreview } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            type: "image",
            mime_type: "image/png",
            provider: "fake",
            model: "fake-image",
            source_message: "The beacon wakes.",
            prompt_preview: "A beacon",
            status: "succeeded",
            created_at: null
          }}
          activeSaveId="save-1"
          onClose={onClose}
          runJob={runJob}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Regenerate with edits" }));
    const dialog = await screen.findByRole("dialog", { name: "Regenerate with edits" });
    const promptField = await within(dialog).findByLabelText("Image prompt");
    expect(promptField).toHaveValue("A beacon above a storm sea");
    await userEvent.clear(promptField);
    await userEvent.type(promptField, "A brighter beacon over glassy water");
    await userEvent.click(within(dialog).getByRole("button", { name: "Regenerate" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/media/media-1/regenerate",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          save_id: "save-1",
          prompt: "A brighter beacon over glassy water"
        })
      })
    ));
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/media/media-1/prompt?save_id=save-1",
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    );
    expect(runJob).toHaveBeenCalledWith(job);
    expect(onClose).toHaveBeenCalled();
  });

  it("confirms previewed media deletion before calling the runtime API", async () => {
    const { ImagePreview } = await import("./main");
    const job = {
      id: "job-media-delete",
      type: "media_delete",
      status: "queued",
      result: null,
      error: null
    };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => job
    });
    vi.stubGlobal("fetch", fetchMock);
    const onClose = vi.fn();
    const runJob = vi.fn();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            type: "image",
            mime_type: "image/png",
            provider: "fake",
            model: "fake-image",
            source_message: "The beacon wakes.",
            prompt_preview: "A beacon",
            prompt: "A beacon above a storm sea",
            status: "succeeded",
            created_at: "2026-05-26T02:00:00Z"
          }}
          activeSaveId="save-1"
          onClose={onClose}
          runJob={runJob}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: /delete/i }));
    const confirm = screen.getByRole("dialog", { name: "Delete image?" });
    expect(within(confirm).getByText("Prompt: A beacon")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();

    await userEvent.click(within(confirm).getByRole("button", { name: "Delete" }));
    await waitFor(() => expect(onClose).toHaveBeenCalled());
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/media/media-1?save_id=save-1",
      expect.objectContaining({ method: "DELETE" })
    );
    expect(runJob).toHaveBeenCalledWith(job);
  });

  it("cancels previewed media deletion without mutating the asset", async () => {
    const { ImagePreview } = await import("./main");
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const onClose = vi.fn();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            type: "image",
            mime_type: "image/png",
            provider: "fake",
            model: "fake-image",
            source_message: "The beacon wakes.",
            prompt_preview: "A beacon",
            status: "succeeded",
            created_at: "2026-05-26T02:00:00Z"
          }}
          activeSaveId="save-1"
          onClose={onClose}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: /delete/i }));
    await userEvent.click(within(screen.getByRole("dialog", { name: "Delete image?" })).getByRole("button", { name: "Cancel" }));

    expect(fetchMock).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: "Image · succeeded" })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: "Delete image?" })).not.toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("keeps media previews open and shows delete failures", async () => {
    const { ImagePreview } = await import("./main");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 409,
      statusText: "Conflict",
      json: async () => ({ detail: "Media is still referenced." })
    });
    vi.stubGlobal("fetch", fetchMock);
    const onClose = vi.fn();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            type: "image",
            mime_type: "image/png",
            provider: "fake",
            model: "fake-image",
            source_message: "The beacon wakes.",
            prompt_preview: "A beacon",
            status: "succeeded",
            created_at: null
          }}
          activeSaveId="save-1"
          onClose={onClose}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: /delete/i }));
    await userEvent.click(within(screen.getByRole("dialog", { name: "Delete image?" })).getByRole("button", { name: "Delete" }));

    expect(await screen.findByText("Media is still referenced.")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Delete image?" })).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Image · succeeded" })).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("submits optional motion guidance when animating previewed media", async () => {
    const { ImagePreview } = await import("./main");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        id: "job-animation",
        type: "image_animation",
        status: "queued",
        result: null,
        error: null
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const onClose = vi.fn();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            source_media_asset_id: null,
            type: "image",
            mime_type: "image/png",
            provider: "fake",
            model: "fake-image",
            source_message: "The beacon wakes.",
            prompt_preview: "A beacon",
            prompt: "A beacon above a storm sea",
            status: "succeeded",
            created_at: "2026-05-26T02:00:00Z",
            can_animate: true
          }}
          activeSaveId="save-1"
          onClose={onClose}
          runJob={runJob}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Animate this" }));
    await userEvent.type(screen.getByLabelText("Motion guidance"), "make the lamp flame breathe");
    await userEvent.click(screen.getByRole("button", { name: "Start animation" }));

    await waitFor(() => expect(runJob).toHaveBeenCalledWith(expect.objectContaining({ id: "job-animation" })));
    expect(onClose).toHaveBeenCalled();
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/media/media-1/animate",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          save_id: "save-1",
          motion_prompt: "make the lamp flame breathe"
        })
      })
    );
  });

  it("keeps animation dialogs open and shows animation failures", async () => {
    const { ImagePreview } = await import("./main");
    const fetchMock = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Server Error",
      json: async () => ({ detail: "Animation queue failed." })
    });
    vi.stubGlobal("fetch", fetchMock);
    const onClose = vi.fn();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            source_media_asset_id: null,
            type: "image",
            mime_type: "image/png",
            provider: "fake",
            model: "fake-image",
            source_message: "The beacon wakes.",
            prompt_preview: "A beacon",
            status: "succeeded",
            created_at: null,
            can_animate: true
          }}
          activeSaveId="save-1"
          onClose={onClose}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Animate this" }));
    await userEvent.type(screen.getByLabelText("Motion guidance"), "make the fog drift");
    await userEvent.click(screen.getByRole("button", { name: "Start animation" }));

    expect(await screen.findByText("Animation queue failed.")).toBeInTheDocument();
    expect(screen.getByRole("dialog", { name: "Animate image" })).toBeInTheDocument();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("does not offer animation for ineligible media", async () => {
    const { ImagePreview } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            source_media_asset_id: null,
            type: "image",
            mime_type: "image/png",
            prompt_preview: "A beacon",
            status: "succeeded",
            created_at: null,
            can_animate: false
          }}
          activeSaveId="save-1"
          onClose={vi.fn()}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    expect(screen.queryByRole("button", { name: "Animate this" })).not.toBeInTheDocument();
  });

  it("lets child users generate media without exposing media management", async () => {
    const { MediaPanel, ImagePreview } = await import("./main");

    const childUser = { id: "child-1", username: "Ilyra", role: "child", status: "active" };
    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            chronicle: {
              messages: [
                { message_id: "narrator-1", role: "narrator", speaker_name: null, body: "The beacon wakes.", actions: [] }
              ]
            },
            media: { latest_scene_image: null, image_history: [], media_history: [] }
          })}
          runJob={vi.fn()}
          currentUser={childUser}
        />
      </QueryClientProvider>
    );

    expect(screen.getByRole("button", { name: "Generate opening image" })).toBeInTheDocument();
    cleanup();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            active_scenario_type: "dating_sim",
            chronicle: {
              messages: [
                { message_id: "narrator-1", role: "narrator", speaker_name: null, body: "The beacon wakes.", actions: [] }
              ]
            },
            media: { latest_scene_image: null, image_history: [], media_history: [] }
          })}
          runJob={vi.fn()}
          currentUser={childUser}
        />
      </QueryClientProvider>
    );

    expect(screen.getByRole("button", { name: "Generate opening image" })).toBeInTheDocument();
    cleanup();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-1",
            source_message_id: "message-1",
            type: "image",
            mime_type: "image/png",
            provider: "fake",
            model: "fake-image",
            prompt_preview: "A beacon",
            status: "succeeded",
            created_at: null,
            can_animate: true,
            can_set_character_reference: true
          }}
          activeSaveId="save-1"
          onClose={vi.fn()}
          runJob={vi.fn()}
          canGenerate
          canManage={false}
        />
      </QueryClientProvider>
    );

    expect(screen.getByRole("dialog", { name: "Image · succeeded" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Set as reference" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Animate this" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Regenerate with edits" })).toBeInTheDocument();
    cleanup();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "reference-media-1",
            source_message_id: "message-1",
            type: "image",
            mime_type: "image/png",
            provider: "fake",
            model: "fake-image",
            prompt_preview: "A character portrait",
            status: "succeeded",
            created_at: null,
            can_animate: true,
            is_character_reference: true
          }}
          activeSaveId="save-1"
          onClose={vi.fn()}
          runJob={vi.fn()}
          canGenerate
          canManage={false}
        />
      </QueryClientProvider>
    );

    expect(screen.getByRole("button", { name: "Animate this" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Regenerate with edits" })).not.toBeInTheDocument();
  });

  it("renders generated video assets in media history and preview", async () => {
    const { MediaPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            media: {
              latest_scene_image: null,
              image_history: [],
              image_animation_available: true,
              media_history: [
                {
                  id: "media-video",
                  source_message_id: "message-1",
                  source_media_asset_id: "media-image",
                  type: "video",
                  mime_type: "video/mp4",
                  prompt_preview: "The beacon starts moving",
                  status: "succeeded",
                  created_at: null,
                  can_animate: false
                }
              ]
            }
          })}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Open full media viewer for The beacon starts moving" }));

    expect(screen.getByLabelText("Video preview")).toBeInTheDocument();
    expect(screen.queryByRole("img")).not.toBeInTheDocument();
  });

  it("uses latest scene media for the primary media preview", async () => {
    const { MediaPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            media: {
              latest_scene_media: {
                id: "media-video",
                source_message_id: "message-1",
                source_media_asset_id: "media-image",
                type: "video",
                mime_type: "video/webm",
                provider: "fake",
                model: "fake-video",
                source_message: "The beacon starts moving.",
                prompt_preview: "The beacon starts moving",
                status: "succeeded",
                created_at: "2026-05-31T23:59:00Z",
                metadata: { duration_seconds: 4 },
                source_media: {
                  id: "media-image",
                  type: "image",
                  mime_type: "image/png",
                  prompt_preview: "Original beacon",
                  source_message_id: "message-1",
                  created_at: "2026-05-31T23:58:00Z"
                },
                file_available: true,
                can_animate: false
              },
              latest_scene_image: {
                id: "media-image",
                source_message_id: "message-1",
                type: "image",
                mime_type: "image/png",
                prompt_preview: "Original beacon",
                status: "succeeded",
                created_at: "2026-05-31T23:58:00Z"
              },
              image_history: [],
              image_animation_available: true,
              media_history: []
            }
          })}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Open full media viewer for The beacon starts moving" }));

    expect(screen.getByLabelText("Video preview")).toBeInTheDocument();
    expect(screen.getByText("Video · succeeded")).toBeInTheDocument();
    expect(screen.getByText("image · Original beacon · image/png")).toBeInTheDocument();
    expect(screen.getByText('{"duration_seconds":4}')).toBeInTheDocument();
  });

  it("selects media history items before opening the full viewer", async () => {
    const { MediaPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            media: {
              latest_scene_media: null,
              latest_scene_image: {
                id: "media-latest",
                source_message_id: "message-1",
                type: "image",
                mime_type: "image/png",
                prompt_preview: "Opening beacon",
                status: "succeeded",
                created_at: null
              },
              image_history: [],
              media_history: [
                {
                  id: "media-storm",
                  source_message_id: "message-2",
                  type: "image",
                  mime_type: "image/png",
                  prompt_preview: "Storm ship",
                  status: "succeeded",
                  created_at: null
                }
              ]
            }
          })}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Select Storm ship" }));

    expect(screen.queryByText("Image · succeeded")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Open full media viewer for Storm ship" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Open full media viewer for Storm ship" }));

    const dialog = screen.getByRole("dialog", { name: "Image · succeeded" });
    expect(within(dialog).getByText("Storm ship")).toBeInTheDocument();
  });

  it("uses thumbnails for media history tiles and originals for full previews", async () => {
    const { MediaPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            media: {
              latest_scene_media: null,
              latest_scene_image: {
                id: "media-latest",
                source_message_id: "message-1",
                type: "image",
                mime_type: "image/png",
                prompt_preview: "Opening beacon",
                status: "succeeded",
                created_at: null
              },
              image_history: [],
              media_history: [
                {
                  id: "media-storm",
                  source_message_id: "message-2",
                  type: "image",
                  mime_type: "image/png",
                  prompt_preview: "Storm ship",
                  status: "succeeded",
                  created_at: null,
                  thumbnail_path: "save-1/thumbnails/media-storm.png"
                }
              ]
            }
          })}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    const thumbnail = screen.getByRole("button", { name: "Select Storm ship" });
    const thumbnailImage = within(thumbnail).getByRole("img", { name: "Storm ship" });
    expect(thumbnailImage).toHaveAttribute("src", "/api/media/media-storm/thumbnail?save_id=save-1");
    expect(thumbnailImage).toHaveAttribute("loading", "lazy");
    expect(thumbnailImage).toHaveAttribute("decoding", "async");

    await userEvent.click(thumbnail);

    const primaryPreview = screen.getByRole("button", { name: "Open full media viewer for Storm ship" });
    expect(within(primaryPreview).getByRole("img", {
      name: "Storm ship"
    })).toHaveAttribute("src", "/api/media/media-storm?save_id=save-1");

    await userEvent.click(primaryPreview);

    const dialog = screen.getByRole("dialog", { name: "Image · succeeded" });
    expect(within(dialog).getByRole("img", {
      name: "Storm ship"
    })).toHaveAttribute("src", "/api/media/media-storm?save_id=save-1");
  });

  it("keeps media selection keyboard behavior in sync with pointer clicks", async () => {
    const { MediaPanel } = await import("./main");
    const user = userEvent.setup();

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            media: {
              latest_scene_media: null,
              latest_scene_image: {
                id: "media-latest",
                source_message_id: "message-1",
                type: "image",
                mime_type: "image/png",
                prompt_preview: "Opening beacon",
                status: "succeeded",
                created_at: null
              },
              image_history: [],
              media_history: [
                {
                  id: "media-storm",
                  source_message_id: "message-2",
                  type: "image",
                  mime_type: "image/png",
                  prompt_preview: "Storm ship",
                  status: "succeeded",
                  created_at: null
                }
              ]
            }
          })}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    const thumbnail = screen.getByRole("button", { name: "Select Storm ship" });
    thumbnail.focus();
    await user.keyboard("{Enter}");

    expect(screen.queryByText("Image · succeeded")).not.toBeInTheDocument();

    const primaryPreview = screen.getByRole("button", { name: "Open full media viewer for Storm ship" });
    primaryPreview.focus();
    await user.keyboard(" ");

    expect(screen.getByRole("dialog", { name: "Image · succeeded" })).toBeInTheDocument();
  });

  it("marks the selected media thumbnail for assistive tech and visual scanning", async () => {
    const { MediaPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <MediaPanel
          model={runtimeModel({
            media: {
              latest_scene_media: null,
              latest_scene_image: {
                id: "media-latest",
                source_message_id: "message-1",
                type: "image",
                mime_type: "image/png",
                prompt_preview: "Opening beacon",
                status: "succeeded",
                created_at: null
              },
              image_history: [],
              media_history: [
                {
                  id: "media-storm",
                  source_message_id: "message-2",
                  type: "image",
                  mime_type: "image/png",
                  prompt_preview: "Storm ship",
                  status: "succeeded",
                  created_at: null
                }
              ]
            }
          })}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    const thumbnail = screen.getByRole("button", { name: "Select Storm ship" });
    expect(thumbnail).not.toHaveAttribute("aria-current");

    await userEvent.click(thumbnail);

    expect(thumbnail).toHaveAttribute("aria-current", "true");
    expect(thumbnail).toHaveClass("selected");
  });

  it("sets an eligible generated character image as the reference from preview", async () => {
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => runtimeModel({
        media: {
          character_reference_image: {
            id: "media-character",
            source_message_id: "message-1",
            source_media_asset_id: "media-reference",
            type: "image",
            mime_type: "image/png",
            prompt_preview: "Generated character image",
            status: "succeeded",
            created_at: null,
            is_character_reference: true,
            can_set_character_reference: false
          },
          latest_scene_image: null,
          image_history: [],
          media_history: []
        },
        status: "Character reference image updated"
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const onClose = vi.fn();
    const { ImagePreview } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-character",
            source_message_id: "message-1",
            source_media_asset_id: "media-reference",
            type: "image",
            mime_type: "image/png",
            prompt_preview: "Generated character image",
            status: "succeeded",
            created_at: null,
            metadata: { kind: "character_image" },
            is_character_reference: false,
            can_set_character_reference: true
          }}
          activeSaveId="save-1"
          onClose={onClose}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Set as reference" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/media/media-character/set-character-reference",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ save_id: "save-1" })
      })
    ));
    expect(onClose).toHaveBeenCalled();
  });

  it("shows the character name first in character image details", async () => {
    const { ImagePreview } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-character",
            source_message_id: "message-1",
            source_media_asset_id: "media-reference",
            type: "image",
            mime_type: "image/png",
            provider: "venice",
            model: "nano-banana-2",
            prompt_preview: "Mara in the storm",
            status: "succeeded",
            created_at: "2026-07-09 14:05:32",
            character_name: "Mara",
            metadata: { kind: "character_image" }
          }}
          activeSaveId="save-1"
          onClose={vi.fn()}
          runJob={vi.fn()}
          canMutate={false}
        />
      </QueryClientProvider>
    );

    const rows = document.body.querySelectorAll(".kv-row");
    expect(rows.length).toBeGreaterThan(0);
    expect(within(rows[0] as HTMLElement).getByText("Character Name")).toBeInTheDocument();
    expect(within(rows[0] as HTMLElement).getByText("Mara")).toBeInTheDocument();
  });

  it("shows a missing media placeholder instead of broken playback", async () => {
    const { ImagePreview } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <ImagePreview
          asset={{
            id: "media-video",
            source_message_id: "message-1",
            type: "video",
            mime_type: "video/mp4",
            prompt_preview: "Missing video",
            status: "succeeded",
            created_at: null,
            file_available: false
          }}
          activeSaveId="save-1"
          onClose={vi.fn()}
          runJob={vi.fn()}
        />
      </QueryClientProvider>
    );

    expect(screen.getByRole("status")).toHaveTextContent("Video file unavailable");
    expect(screen.queryByLabelText("Video preview")).not.toBeInTheDocument();
  });

  it("lets the runtime infer the player speaker name", async () => {
    const { Composer } = await import("./main");
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/chat"
        ? { id: "job-1", type: "chat_turn", status: "queued", result: null, error: null }
        : { saves: [], active_save_id: "save-1", chronicle: { messages: [] }, composer_enabled: true }
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    expect(screen.queryByPlaceholderText("Speaker")).not.toBeInTheDocument();
    await userEvent.type(screen.getByRole("textbox", { name: "Message" }), "Light the beacon");
    await userEvent.click(screen.getByTitle("Send"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const chatCall = fetchMock.mock.calls.find(([path]) => path === "/api/chat");
    expect(JSON.parse(String(chatCall?.[1].body))).toMatchObject({
      body: "Light the beacon",
      save_id: "save-1",
      speaker_name: null
    });
  });

  it("renders CYOA action choices without the default composer textarea", async () => {
    const { CyoaActionPicker } = await import("./main");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CyoaActionPicker
          disabled={false}
          runJob={vi.fn()}
          activeSaveId="save-1"
          actionChoices={cyoaActionChoices({
            choices: [
              { choice_id: "choice-3", ordinal: 2, body: "Search the moonlit shelves" },
              { choice_id: "choice-1", ordinal: 0, body: "Open the brass door" },
              { choice_id: "choice-4", ordinal: 3, body: "Retreat to the courtyard" },
              { choice_id: "choice-2", ordinal: 1, body: "Question the masked guide" }
            ]
          })}
          pendingAfterMessageId="narrator-1"
          onPendingMessage={vi.fn()}
        />
      </QueryClientProvider>
    );

    const generatedActions = screen.getByRole("list", { name: "Generated actions" });
    const generatedButtons = within(generatedActions).getAllByRole("button");
    expect(generatedButtons).toHaveLength(4);
    expect(generatedButtons[0]).toHaveAccessibleName("Open the brass door");
    expect(generatedButtons[1]).toHaveAccessibleName("Question the masked guide");
    expect(generatedButtons[2]).toHaveAccessibleName("Search the moonlit shelves");
    expect(generatedButtons[3]).toHaveAccessibleName("Retreat to the courtyard");
    expect(within(generatedActions).queryByRole("button", { name: "Write your own" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Write your own" })).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByRole("textbox", { name: "Message" })).not.toBeInTheDocument();
    expect(screen.queryByRole("textbox", { name: "Custom action" })).not.toBeInTheDocument();
  });

  it("starts CYOA action choice regeneration from the latest narrator message", async () => {
    const { CyoaActionPicker } = await import("./main");
    const job = {
      id: "job-choice-1",
      type: "action_choice_regenerate",
      save_id: "save-1",
      status: "queued",
      result: null,
      error: null
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/action-choices/regenerate" ? job : {}
    }));
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CyoaActionPicker
          disabled={false}
          runJob={runJob}
          activeSaveId="save-1"
          actionChoices={cyoaActionChoices()}
          pendingAfterMessageId="narrator-1"
          onPendingMessage={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: /regenerate options/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/action-choices/regenerate",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          message_id: "narrator-1",
          save_id: "save-1"
        })
      })
    ));
    expect(runJob).toHaveBeenCalledWith(job);
  });

  it("submits a selected CYOA action as a normal chat message", async () => {
    const { CyoaActionPicker } = await import("./main");
    const job = { id: "job-1", type: "chat_turn", save_id: "save-1", status: "queued", result: null, error: null };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/chat" ? job : {}
    }));
    const runJob = vi.fn();
    const onPendingMessage = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CyoaActionPicker
          disabled={false}
          runJob={runJob}
          activeSaveId="save-1"
          actionChoices={cyoaActionChoices()}
          pendingAfterMessageId="narrator-1"
          onPendingMessage={onPendingMessage}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Search the moonlit shelves" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const chatCall = fetchMock.mock.calls.find(([path]) => path === "/api/chat");
    expect(JSON.parse(String(chatCall?.[1].body))).toMatchObject({
      body: "Search the moonlit shelves",
      save_id: "save-1",
      speaker_name: null
    });
    expect(onPendingMessage).toHaveBeenCalledWith(expect.objectContaining({
      body: "Search the moonlit shelves",
      pending_after_message_id: "narrator-1"
    }));
    await waitFor(() => expect(runJob).toHaveBeenCalledWith(job));
  });

  it("submits a custom CYOA action through the same chat path", async () => {
    const { CyoaActionPicker } = await import("./main");
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/chat"
        ? { id: "job-1", type: "chat_turn", save_id: "save-1", status: "queued", result: null, error: null }
        : {}
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CyoaActionPicker
          disabled={false}
          runJob={vi.fn()}
          activeSaveId="save-1"
          actionChoices={cyoaActionChoices()}
          pendingAfterMessageId="narrator-1"
          onPendingMessage={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Write your own" }));
    const textarea = screen.getByRole("textbox", { name: "Custom action" });
    await userEvent.type(textarea, "Climb through the observatory window");
    await userEvent.click(screen.getByTitle("Send"));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const chatCall = fetchMock.mock.calls.find(([path]) => path === "/api/chat");
    expect(JSON.parse(String(chatCall?.[1].body))).toMatchObject({
      body: "Climb through the observatory window",
      save_id: "save-1",
      speaker_name: null
    });
  });

  it("blocks duplicate CYOA submissions while a choice is pending", async () => {
    const { CyoaActionPicker } = await import("./main");
    let resolveChat: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const chatResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveChat = resolve;
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => (
      path === "/api/chat"
        ? chatResponse
        : Promise.resolve({ ok: true, json: async () => ({}) })
    ));
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CyoaActionPicker
          disabled={false}
          runJob={runJob}
          activeSaveId="save-1"
          actionChoices={cyoaActionChoices()}
          pendingAfterMessageId="narrator-1"
          onPendingMessage={vi.fn()}
        />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByRole("button", { name: "Open the brass door" }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/chat")).toHaveLength(1));
    expect(screen.getByRole("button", { name: "Question the masked guide" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Write your own" })).toBeDisabled();

    await userEvent.click(screen.getByRole("button", { name: "Question the masked guide" }));
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/chat")).toHaveLength(1);

    resolveChat({
      ok: true,
      json: async () => ({ id: "job-1", type: "chat_turn", save_id: "save-1", status: "queued", result: null, error: null })
    });
    await waitFor(() => expect(runJob).toHaveBeenCalledTimes(1));
  });

  it("formats selected composer text as narration and submits the markdown body", async () => {
    const { Composer } = await import("./main");
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/chat"
        ? { id: "job-1", type: "chat_turn", status: "queued", result: null, error: null }
        : {}
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement;
    await userEvent.type(textarea, "The hall goes quiet");
    textarea.setSelectionRange(0, textarea.value.length);
    await userEvent.click(screen.getByRole("button", { name: "Format as narration" }));

    expect(textarea).toHaveValue("*The hall goes quiet*");

    await userEvent.click(screen.getByTitle("Send"));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat", expect.anything()));
    const chatCall = fetchMock.mock.calls.find(([path]) => path === "/api/chat");
    expect(JSON.parse(String(chatCall?.[1].body))).toMatchObject({
      body: "*The hall goes quiet*",
      save_id: "save-1",
      speaker_name: null
    });
  });

  it("formats the current composer line as dialogue when no text is selected", async () => {
    const { Composer } = await import("./main");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement;
    await userEvent.type(textarea, "I wait\nSay hi\nLater");
    textarea.setSelectionRange(9, 9);
    await userEvent.click(screen.getByRole("button", { name: "Format as dialogue" }));

    expect(textarea).toHaveValue("I wait\n\"Say hi\"\nLater");
  });

  it("formats selected composer lines as a text message block", async () => {
    const { Composer } = await import("./main");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement;
    await userEvent.type(textarea, "first ping\nsecond ping");
    textarea.setSelectionRange(0, textarea.value.length);
    await userEvent.click(screen.getByRole("button", { name: "Format as text message" }));

    expect(textarea).toHaveValue("> first ping\n> second ping");
  });

  it("clears supported roleplay formatting from selected composer text", async () => {
    const { Composer } = await import("./main");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement;
    await userEvent.type(textarea, "*Footsteps*\n> meet me outside\n\"Fine.\"");
    textarea.setSelectionRange(0, textarea.value.length);
    await userEvent.click(screen.getByRole("button", { name: "Clear roleplay formatting" }));

    expect(textarea).toHaveValue("Footsteps\nmeet me outside\nFine.");
  });

  it("clears only selected roleplay formatting inside a composer line", async () => {
    const { Composer } = await import("./main");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement;
    await userEvent.type(textarea, "She says \"Fine.\" now");
    textarea.setSelectionRange(9, 16);
    await userEvent.click(screen.getByRole("button", { name: "Clear roleplay formatting" }));

    expect(textarea).toHaveValue("She says Fine. now");
  });

  it("supports composer formatting shortcuts only while the message field is focused", async () => {
    const { Composer } = await import("./main");
    const user = userEvent.setup();
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) }));

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" }) as HTMLTextAreaElement;
    await user.type(textarea, "Listen close");
    textarea.setSelectionRange(0, textarea.value.length);
    await user.keyboard("{Alt>}n{/Alt}");

    expect(textarea).toHaveValue("*Listen close*");

    const dialogueButton = screen.getByRole("button", { name: "Format as dialogue" });
    dialogueButton.focus();
    await new Promise<void>((resolve) => window.requestAnimationFrame(() => resolve()));

    expect(dialogueButton).toHaveFocus();

    await user.keyboard("{Alt>}q{/Alt}");

    expect(textarea).toHaveValue("*Listen close*");
  });

  it("opens and cancels the timeskip dialog without submitting", async () => {
    const { Composer } = await import("./main");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByTitle("Timeskip"));
    const dialog = screen.getByRole("dialog", { name: "Timeskip" });
    await userEvent.type(
      within(dialog).getByLabelText("Timeskip instructions"),
      "Skip to dawn at the city gates."
    );
    await userEvent.click(within(dialog).getByRole("button", { name: "Cancel" }));

    expect(screen.queryByRole("dialog", { name: "Timeskip" })).not.toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("submits timeskip instructions as a chat turn job", async () => {
    const { Composer } = await import("./main");
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/chat/timeskip"
        ? { id: "job-1", type: "chat_turn", save_id: "save-1", status: "queued", result: null, error: null }
        : {}
    }));
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={runJob} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByTitle("Timeskip"));
    const dialog = screen.getByRole("dialog", { name: "Timeskip" });
    await userEvent.type(
      within(dialog).getByLabelText("Timeskip instructions"),
      "Skip to dawn at the city gates."
    );
    await userEvent.click(within(dialog).getByRole("button", { name: "Timeskip" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/chat/timeskip", expect.anything()));
    const timeskipCall = fetchMock.mock.calls.find(([path]) => path === "/api/chat/timeskip");
    expect(JSON.parse(String(timeskipCall?.[1].body))).toMatchObject({
      instruction: "Skip to dawn at the city gates.",
      save_id: "save-1"
    });
    await waitFor(() => expect(runJob).toHaveBeenCalledWith(expect.objectContaining({
      id: "job-1",
      type: "chat_turn",
      save_id: "save-1"
    })));
  });

  it("keeps the composer textarea editable while submit is disabled", async () => {
    const { Composer } = await import("./main");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={true} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" });
    await userEvent.type(textarea, "Plotting the next questionable decision");

    expect(textarea).toBeEnabled();
    expect(textarea).toHaveValue("Plotting the next questionable decision");
    expect(screen.getByTitle("Send")).toBeDisabled();
    expect(screen.getByTitle("Timeskip")).toBeDisabled();

    fireEvent.submit(textarea.closest("form") as HTMLFormElement);

    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("closes Timeskip when chat mutations become disabled or the active save changes", async () => {
    const { Composer } = await import("./main");
    const props = { runJob: vi.fn(), onPendingMessage: vi.fn() };
    const { rerender } = render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer {...props} disabled={false} activeSaveId="save-1" />
      </QueryClientProvider>
    );

    await userEvent.click(screen.getByTitle("Timeskip"));
    expect(screen.getByRole("dialog", { name: "Timeskip" })).toBeInTheDocument();
    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <Composer {...props} disabled activeSaveId="save-1" />
      </QueryClientProvider>
    );
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Timeskip" })).not.toBeInTheDocument());

    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <Composer {...props} disabled={false} activeSaveId="save-1" />
      </QueryClientProvider>
    );
    await userEvent.click(screen.getByTitle("Timeskip"));
    rerender(
      <QueryClientProvider client={new QueryClient()}>
        <Composer {...props} disabled={false} activeSaveId="save-2" />
      </QueryClientProvider>
    );
    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Timeskip" })).not.toBeInTheDocument());
  });

  it("shows composer submit failures inline for failed responses", async () => {
    const { Composer } = await import("./main");
    const failureCases = [
      [409, "Conflict", "A turn is already active for this save."],
      [429, "Too Many Requests", "Too many active jobs; wait for one to finish before starting another."],
      [500, "Server Error", "The provider fell over mid-sentence."]
    ] as const;

    for (const [status, statusText, detail] of failureCases) {
      cleanup();
      const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
        path === "/api/chat"
          ? { ok: false, status, statusText, json: async () => ({ detail }) }
          : { ok: true, json: async () => ({}) }
      ));
      const onPendingMessage = vi.fn();
      vi.stubGlobal("fetch", fetchMock);

      render(
        <QueryClientProvider client={new QueryClient()}>
          <Composer disabled={false} runJob={vi.fn()} activeSaveId="save-1" onPendingMessage={onPendingMessage} />
        </QueryClientProvider>
      );

      const textarea = screen.getByRole("textbox", { name: "Message" });
      await userEvent.type(textarea, "Light the beacon");
      fireEvent.submit(textarea.closest("form") as HTMLFormElement);

      expect(await screen.findByRole("alert")).toHaveTextContent(detail);
      expect(textarea).toHaveValue("Light the beacon");
      expect(onPendingMessage).toHaveBeenCalledWith(expect.objectContaining({ body: "Light the beacon" }));
      expect(onPendingMessage).toHaveBeenLastCalledWith(null);
    }
  });

  it("clears stale composer submit errors when retrying", async () => {
    const { Composer } = await import("./main");
    let chatRequests = 0;
    let resolveSecondChat: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const secondChatResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveSecondChat = resolve;
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      if (path !== "/api/chat") return Promise.resolve({ ok: true, json: async () => ({}) });
      chatRequests += 1;
      return chatRequests === 1
        ? Promise.resolve({
            ok: false,
            status: 500,
            statusText: "Server Error",
            json: async () => ({ detail: "The first send failed." })
          })
        : secondChatResponse;
    });
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={runJob} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" });
    const form = textarea.closest("form") as HTMLFormElement;
    await userEvent.type(textarea, "Try the risky move");
    fireEvent.submit(form);

    expect(await screen.findByRole("alert")).toHaveTextContent("The first send failed.");

    fireEvent.submit(form);
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument());

    resolveSecondChat({
      ok: true,
      json: async () => ({ id: "job-1", type: "chat_turn", status: "queued", result: null, error: null })
    });

    await waitFor(() => expect(runJob).toHaveBeenCalledWith(expect.objectContaining({ id: "job-1" })));
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("preserves typed-ahead composer text when the previous submit queues", async () => {
    const { Composer } = await import("./main");
    let resolveChat: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const chatResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveChat = resolve;
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => (
      path === "/api/chat"
        ? chatResponse
        : Promise.resolve({ ok: true, json: async () => ({}) })
    ));
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={runJob} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" });
    await userEvent.type(textarea, "Light the beacon");
    fireEvent.submit(textarea.closest("form") as HTMLFormElement);

    await waitFor(() => expect(textarea).toHaveValue(""));
    expect(screen.getByTitle("Timeskip")).toBeDisabled();
    await userEvent.type(textarea, "Prepare the follow-up bit");

    resolveChat({
      ok: true,
      json: async () => ({ id: "job-1", type: "chat_turn", status: "queued", result: null, error: null })
    });

    await waitFor(() => expect(runJob).toHaveBeenCalledTimes(1));
    expect(textarea).toHaveValue("Prepare the follow-up bit");
    const chatCall = fetchMock.mock.calls.find(([path]) => path === "/api/chat");
    expect(JSON.parse(String(chatCall?.[1].body))).toMatchObject({
      body: "Light the beacon",
      save_id: "save-1",
      speaker_name: null
    });
  });

  it("guards duplicate composer submits before the pending state rerenders", async () => {
    const { Composer } = await import("./main");
    let resolveChat: (response: { ok: boolean; json: () => Promise<Job> }) => void = () => undefined;
    const chatResponse = new Promise<{ ok: boolean; json: () => Promise<Job> }>((resolve) => {
      resolveChat = resolve;
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => (
      path === "/api/chat"
        ? chatResponse
        : Promise.resolve({ ok: true, json: async () => ({}) })
    ));
    const runJob = vi.fn();
    vi.stubGlobal("fetch", fetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Composer disabled={false} runJob={runJob} activeSaveId="save-1" onPendingMessage={vi.fn()} />
      </QueryClientProvider>
    );

    const textarea = screen.getByRole("textbox", { name: "Message" });
    await userEvent.type(textarea, "Light the beacon");
    const form = textarea.closest("form");
    expect(form).not.toBeNull();

    fireEvent.submit(form as HTMLFormElement);
    fireEvent.submit(form as HTMLFormElement);

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/chat").length).toBeGreaterThan(0));
    await act(async () => {
      await Promise.resolve();
    });
    expect(fetchMock.mock.calls.filter(([path]) => path === "/api/chat")).toHaveLength(1);
    resolveChat({
      ok: true,
      json: async () => ({ id: "job-1", type: "chat_turn", status: "queued", result: null, error: null })
    });
    await waitFor(() => expect(runJob).toHaveBeenCalledTimes(1));
  });

  it("starts saved scenarios from an in-app modal instead of window prompt", async () => {
    const { LeftRail } = await import("./main");
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => ({}) });
    const promptMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    vi.stubGlobal("prompt", promptMock);

    render(
      <LeftRail
        model={{
          saves: [],
          active_save_id: null,
          active_save_title: null,
          active_scenario_type: null,
          scenario_title: null,
          scene_title: "",
          chronicle: { messages: [] },
          media: null,
          action_choices: null,
          model_indicator: "",
          failed_save: false,
          composer_enabled: false,
          failure_text: null,
          status: null,
          error: null
        }}
        scenarios={[
          {
            scenario_id: "scenario-1",
            scenario_type: "full_roleplay",
            title: "Lantern Keep",
            premise: "A mountain beacon is going dark.",
            player_role: "Keeper",
            opening_message: null,
            save_count: 2
          }
        ]}
        onChanged={vi.fn()}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (1)" }));
    await userEvent.click(screen.getByRole("button", { name: "Start Lantern Keep" }));

    expect(promptMock).not.toHaveBeenCalled();
    expect(screen.getByRole("dialog", { name: "Start Scenario" })).toBeInTheDocument();
    expect(screen.getByLabelText("Save Title")).toHaveValue("Lantern Keep");

    await userEvent.clear(screen.getByLabelText("Save Title"));
    await userEvent.type(screen.getByLabelText("Save Title"), "Lantern Keep Run");
    await userEvent.click(screen.getByRole("button", { name: "Start" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/scenario-1/start", expect.anything()));
    const startCall = fetchMock.mock.calls.find(([path]) => path === "/api/scenarios/scenario-1/start");
    expect(JSON.parse(String(startCall?.[1].body))).toEqual({ save_title: "Lantern Keep Run" });
  });

  it("reuses saved scenario generation prompts from the scenario library", async () => {
    installEventSourceDouble();
    const model = runtimeModel({
      saves: [{ save_id: "save-1", title: "Lantern Run", active: true }]
    });
    const scenario = scenarioFixture({
      scenario_id: "scenario-1",
      scenario_type: "full_roleplay",
      scenario_types: ["full_roleplay", "dating_sim"],
      title: "Fog Gate",
      premise: "A gate in the fog.",
      player_role: "Keeper",
      save_count: 0,
      has_generation_prompt: true,
      action_choices_enabled: true
    });
    const definitionPayload = {
      active_save_id: null,
      scenario: {
        scenario_id: "scenario-1",
        scenario_type: "full_roleplay",
        title: "Fog Gate",
        premise: "A gate in the fog.",
        player_character_name: "Mara",
        player_role: "Keeper",
        generation_prompt: "A fog gate romance with action choices.",
        content_sections: [
          ["_scenario_genres", ["full_roleplay", "dating_sim"]],
          ["action_choices_enabled", true]
        ],
        character_starters: []
      }
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path.startsWith("/api/runtime")) return model;
        if (path === "/api/scenarios") return { scenarios: [scenario] };
        if (path.startsWith("/api/jobs?status=active")) return { jobs: [] };
        if (path === "/api/settings/shell") return { pending_jobs_display_mode: "compact" };
        if (isSettingsReadPath(path)) return modelSettingsPayload();
        if (path.startsWith("/api/chat/submission-status")) {
          return { save_id: "save-1", can_submit: true, reason: null, blocking_job_id: null, blocking_job_status: null };
        }
        if (path === "/api/scenarios/scenario-1/definition") return definitionPayload;
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { Workbench } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <Workbench />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: /Scenarios/ }));
    await userEvent.click(await screen.findByRole("button", { name: "Reuse prompt for Fog Gate" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenarios/scenario-1/definition", expect.anything()));
    expect(await screen.findByRole("dialog", { name: "New scenario" })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "AI draft" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("textbox", { name: "Scenario seed" })).toHaveValue(
      "A fog gate romance with action choices.",
    );
    expect(screen.getByLabelText("Action choices")).toBeChecked();
  });

  it("imports and exports reusable scenario bundles from the scenario library", async () => {
    const openMock = vi.fn();
    vi.stubGlobal("open", openMock);
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/scenario-bundles/preview") {
          return {
            preview_id: "preview-1",
            preview: {
              scenario_id: "scenario-imported",
              title: "Imported Lantern",
              scenario_type: "full_roleplay",
              bundle_version: 1,
              created_at: null,
              updated_at: null,
              exported_at: null
            }
          };
        }
        if (path === "/api/scenario-bundles/import/preview-1") return { status: "Imported scenario" };
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const onChanged = vi.fn();
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel()}
        scenarios={[
          {
            scenario_id: "scenario-1",
            scenario_type: "full_roleplay",
            title: "Lantern Keep",
            premise: "A mountain beacon is going dark.",
            player_role: "Keeper",
            opening_message: null,
            save_count: 0
          }
        ]}
        onChanged={onChanged}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("tab", { name: "Scenarios (1)" }));
    await userEvent.click(screen.getByRole("button", { name: "Export Lantern Keep" }));
    expect(openMock).toHaveBeenCalledWith("/api/scenario-bundles/export/scenario-1", "_blank", "noopener,noreferrer");

    const file = new File(["scenario"], "scenario.bragi-scenario", { type: "application/octet-stream" });
    await userEvent.upload(screen.getByLabelText("Scenario bundle file"), file);
    expect(await screen.findByRole("dialog", { name: "Import scenario bundle?" })).toBeInTheDocument();
    expect(screen.getByText("Imported Lantern")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Import" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/scenario-bundles/import/preview-1", expect.anything()));
    expect(onChanged).toHaveBeenCalled();
  });

  it("uses keyboard-focusable buttons for bundle imports", async () => {
    const clickSpy = vi
      .spyOn(HTMLInputElement.prototype, "click")
      .mockImplementation(() => undefined);
    const { LeftRail } = await import("./main");
    try {
      render(
        <LeftRail
          model={runtimeModel({ saves: [{ save_id: "save-1", title: "Lantern Run", active: true }] })}
          scenarios={[
            {
              scenario_id: "scenario-1",
              scenario_type: "full_roleplay",
              title: "Lantern Keep",
              premise: "A mountain beacon is going dark.",
              player_role: "Keeper",
              opening_message: null,
              save_count: 0
            }
          ]}
          onChanged={vi.fn()}
          onSelectSave={vi.fn()}
          pendingSaveId={null}
          saveSelectionError=""
          onNew={vi.fn()}
          activePanel="media"
          setPanel={vi.fn()}
        />
      );

      expect(screen.getByRole("button", { name: "Import save bundle" })).toBeInTheDocument();
      await userEvent.click(screen.getByRole("tab", { name: "Scenarios (1)" }));
      const scenarioImport = screen.getByRole("button", { name: "Import scenario bundle" });

      scenarioImport.focus();
      expect(scenarioImport).toHaveFocus();
      await userEvent.keyboard("{Enter}");

      await userEvent.click(screen.getByRole("tab", { name: "Saves (1)" }));
      const saveImport = screen.getByRole("button", { name: "Import save bundle" });
      saveImport.focus();
      expect(saveImport).toHaveFocus();
      await userEvent.keyboard(" ");

      expect(clickSpy).toHaveBeenCalledTimes(2);
    } finally {
      clickSpy.mockRestore();
    }
  });

  it("labels icon-only panel navigation for assistive technology", async () => {
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={undefined}
        scenarios={[]}
        onChanged={vi.fn()}
        onSelectSave={vi.fn()}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    expect(screen.getByRole("button", { name: "Media" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "History" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "World" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Characters" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Sync" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Settings" })).toBeInTheDocument();
  });

  it("imports and exports active save bundles from the saves section", async () => {
    const openMock = vi.fn();
    vi.stubGlobal("open", openMock);
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/bundles/preview") {
          return {
            preview_id: "preview-save-1",
            preview: {
              save_id: "save-imported",
              title: "Imported Run",
              scenario_title: "Lantern Keep",
              message_count: 4,
              media_count: 1,
              bundle_version: 1,
              created_at: null,
              updated_at: null,
              exported_at: null
            }
          };
        }
        if (path === "/api/bundles/import/preview-save-1") return runtimeModel({ active_save_id: "save-imported" });
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const onChanged = vi.fn();
    const onSelectSave = vi.fn();
    const { LeftRail } = await import("./main");

    render(
      <LeftRail
        model={runtimeModel({ saves: [{ save_id: "save-1", title: "Lantern Run", active: true }] })}
        scenarios={[]}
        onChanged={onChanged}
        onSelectSave={onSelectSave}
        pendingSaveId={null}
        saveSelectionError=""
        onNew={vi.fn()}
        activePanel="media"
        setPanel={vi.fn()}
      />
    );

    await userEvent.click(screen.getByRole("button", { name: "Export active save" }));
    expect(openMock).toHaveBeenCalledWith("/api/bundles/export?save_id=save-1", "_blank", "noopener,noreferrer");

    const file = new File(["save"], "save.bragi-chat", { type: "application/octet-stream" });
    await userEvent.upload(screen.getByLabelText("Save bundle file"), file);
    expect(await screen.findByRole("dialog", { name: "Import save bundle?" })).toBeInTheDocument();
    expect(screen.getByText("Imported Run")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Import" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/bundles/import/preview-save-1", expect.anything()));
    expect(onChanged).toHaveBeenCalled();
    expect(onSelectSave).toHaveBeenCalledWith("save-imported");
    expect(fetchMock.mock.calls.some(([path]) => String(path).startsWith("/api/sync/"))).toBe(false);
  });

  it("renders the character panel read-only for child users", async () => {
    const characterPayload = characterRegistryPayload();
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => characterPayload
    });
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel
          activeSaveId="save-1"
          runJob={vi.fn(() => vi.fn())}
          currentUser={{ id: "child-1", username: "Ilyra", role: "child", status: "active" }}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByText("Mara")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Import character bundle" })).not.toBeInTheDocument();
    expect(screen.queryByTitle("Add character")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Export Mara" })).not.toBeInTheDocument();

    await userEvent.click(screen.getByText("Mara"));
    expect(screen.getByLabelText("Name")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Save character" })).toBeDisabled();
  });

  it("adds, filters, and archives characters from the character panel", async () => {
    const characterPayload = characterRegistryPayload();
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/characters/apply"
        ? { model: characterPayload, created_count: 1, updated_count: 0, archived_count: 0 }
        : characterPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    expect(await screen.findByText("Mara")).toBeInTheDocument();
    expect(screen.queryByText("away")).not.toBeInTheDocument();

    await userEvent.click(screen.getByTitle("Add character"));
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/characters/apply")).toBe(false);
    const draftSummary = await screen.findByText("New character");
    await userEvent.click(draftSummary);
    const draftDetails = draftSummary.closest("details");
    if (!draftDetails) throw new Error("Expected draft character details");
    await userEvent.click(within(draftDetails).getByRole("button", { name: "Save character" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/apply", expect.anything()));
    const addCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/apply");
    expect(JSON.parse(String(addCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      auto_enhance_created_agency: true,
      edits: {
        characters: [
          {
            character_id: "",
            name: "New character",
            relationships_json: "{}",
            present: true
          }
        ]
      }
    });

    await userEvent.type(screen.getByLabelText("Search characters"), "fog rival");
    expect(screen.queryByText("Mara")).not.toBeInTheDocument();
    expect(screen.getByText("No characters found")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("tab", { name: "All" }));
    expect(await screen.findByText("away")).toBeInTheDocument();

    await userEvent.click(screen.getByText("away"));
    const archiveButton = screen.getByTitle("Archive character");
    expect(archiveButton).toBeDisabled();
    await userEvent.type(screen.getByLabelText("Archive confirmation"), "DELETE");
    expect(archiveButton).toBeEnabled();
    await userEvent.click(archiveButton);

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/characters/apply")).toHaveLength(2));
    const archiveCalls = fetchMock.mock.calls.filter(([path]) => path === "/api/characters/apply");
    const archiveCall = archiveCalls[archiveCalls.length - 1];
    expect(JSON.parse(String(archiveCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      edits: {
        characters: [
          {
            character_id: "character-2",
            name: "Ilyra",
            archived: true
          }
        ]
      }
    });
  });

  it("saves character booleans, location, merge target, and save failures", async () => {
    const characterPayload = characterRegistryPayload();
    let failNextApply = false;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/characters/apply" && failNextApply
        ? { ok: false, status: 500, statusText: "Server Error", json: async () => ({ detail: "Character save failed." }) }
        : {
            ok: true,
            json: async () => path === "/api/characters/apply"
              ? { model: characterPayload, created_count: 0, updated_count: 1, archived_count: 0 }
              : characterPayload
          }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.clear(screen.getByLabelText("Age"));
    await userEvent.type(screen.getByLabelText("Age"), "late 40s");
    await userEvent.clear(screen.getByLabelText("History"));
    await userEvent.type(screen.getByLabelText("History"), "Knows the fog route.");
    await userEvent.click(screen.getByLabelText("Met"));
    await userEvent.click(screen.getByLabelText("Protected From Maintenance"));
    await userEvent.click(screen.getByLabelText("Player Character"));
    expect(screen.getByLabelText("Present")).toBeDisabled();
    await userEvent.selectOptions(screen.getByLabelText("Location"), "location-docks");
    await userEvent.selectOptions(screen.getByLabelText("Merge target"), "character-2");
    await userEvent.click(screen.getByRole("tab", { name: "Agency" }));
    expect(screen.getByRole("button", { name: /Auto-enhance Goals/i })).toBeInTheDocument();
    await userEvent.clear(screen.getByLabelText("Current Intent"));
    await userEvent.type(screen.getByLabelText("Current Intent"), "Demand proof before sharing the failsafe.");
    await userEvent.clear(screen.getByLabelText("Cooperation Conditions"));
    await userEvent.type(screen.getByLabelText("Cooperation Conditions"), "Helps after Mara shows the brass warrant.");
    await userEvent.click(screen.getByRole("button", { name: "Save character" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/apply", expect.anything()));
    const successCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/apply");
    expect(JSON.parse(String(successCall?.[1].body))).toMatchObject({
      edits: {
        characters: [
          {
            character_id: "character-1",
            age: "late 40s",
            known_state: "Knows the fog route.",
            met: false,
            protected_from_maintenance: true,
            is_player_character: true,
            present: true,
            location_id: "location-docks",
            merge_into_character_id: "character-2",
            current_intent: "Demand proof before sharing the failsafe.",
            cooperation_conditions: "Helps after Mara shows the brass warrant."
          }
        ]
      }
    });

    failNextApply = true;
    await userEvent.click(screen.getByRole("tab", { name: "Profile" }));
    await userEvent.clear(screen.getByLabelText("Role"));
    await userEvent.type(screen.getByLabelText("Role"), "Keeper");
    await userEvent.click(screen.getByRole("button", { name: "Save character" }));

    expect(await screen.findByText("Character save failed.")).toBeInTheDocument();
    expect(screen.getByLabelText("Role")).toHaveValue("Keeper");
  });

  it("shows saves and discards unsaved character edits", async () => {
    let characterPayload = characterRegistryPayload();
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/apply" && init?.method === "POST") {
          const body = JSON.parse(String(init.body));
          const firstCharacter = characterPayload.characters?.[0];
          const secondCharacter = characterPayload.characters?.[1];
          if (!firstCharacter || !secondCharacter) throw new Error("Expected character fixtures");
          characterPayload = characterRegistryPayload({
            characters: [
              { ...firstCharacter, ...body.edits.characters[0] },
              secondCharacter
            ]
          });
          return { model: characterPayload, created_count: 0, updated_count: 1, archived_count: 0 };
        }
        return characterPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    const saveButton = screen.getByRole("button", { name: "Save character" });
    expect(saveButton).toBeDisabled();
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument();

    await userEvent.clear(screen.getByLabelText("Role"));
    await userEvent.type(screen.getByLabelText("Role"), "Signal keeper");
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();
    expect(saveButton).toBeEnabled();

    await userEvent.click(saveButton);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/apply", expect.anything()));
    await waitFor(() => expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument());
    expect(saveButton).toBeDisabled();

    await userEvent.clear(screen.getByLabelText("Role"));
    await userEvent.type(screen.getByLabelText("Role"), "Unsaved scout");
    await userEvent.click(screen.getByRole("button", { name: "Discard changes" }));
    expect(screen.getByRole("dialog", { name: "Discard changes?" })).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Cancel" }));
    expect(screen.getByLabelText("Role")).toHaveValue("Unsaved scout");
    expect(screen.getByText("Unsaved changes")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Discard changes" }));
    await userEvent.click(screen.getByRole("button", { name: "Discard" }));

    expect(screen.getByLabelText("Role")).toHaveValue("Signal keeper");
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument();
    expect(saveButton).toBeDisabled();
  });

  it("auto-enhances character narrative fields from the current editor row", async () => {
    const basePayload = characterRegistryPayload();
    const mara = basePayload.characters?.[0];
    const ilyra = basePayload.characters?.[1];
    if (!mara || !ilyra) throw new Error("Expected character fixtures");
    let latestPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          appearance: "Ash-dusted cloak.",
          visual_notes: "",
          locked_fields: []
        },
        ilyra
      ]
    });
    const enhancedPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          known_state: "Knows the repaired lens.",
          appearance: "Ash-dusted cloak.\n\nCopper lens-key on a black cord.",
          visual_notes: "",
          locked_fields: ["appearance"]
        },
        ilyra
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/character-1/enhance-field") {
          latestPayload = enhancedPayload;
          return {
            model: enhancedPayload,
            character_id: "character-1",
            field_name: "appearance",
            created_count: 0,
            updated_count: 1,
            archived_count: 0,
            error: null
          };
        }
        return latestPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    expect(screen.queryByRole("button", { name: "Auto-enhance Role" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Auto-enhance Visual notes" })).toBeInTheDocument();

    await userEvent.clear(screen.getByLabelText("History"));
    await userEvent.type(screen.getByLabelText("History"), "Knows the repaired lens.");
    await userEvent.click(screen.getByRole("button", { name: "Auto-enhance Appearance" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/character-1/enhance-field", expect.anything()));
    const enhanceCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/character-1/enhance-field");
    expect(JSON.parse(String(enhanceCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      field_name: "appearance",
      character: {
        character_id: "character-1",
        known_state: "Knows the repaired lens.",
        appearance: "Ash-dusted cloak."
      }
    });
    expect(screen.getByLabelText("Appearance")).toHaveValue("Ash-dusted cloak.\n\nCopper lens-key on a black cord.");
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument();
  });

  it("auto-enhances character texting style from the current editor row", async () => {
    const basePayload = characterRegistryPayload();
    const mara = basePayload.characters?.[0];
    const ilyra = basePayload.characters?.[1];
    if (!mara || !ilyra) throw new Error("Expected character fixtures");
    let latestPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          texting_style: "Short replies after midnight.",
          locked_fields: []
        },
        ilyra
      ]
    });
    const enhancedPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          role: "Beacon courier",
          texting_style: "Short replies after midnight.\n\nLowercase bursts, double texts when worried, one sparkle emoji max.",
          locked_fields: ["texting_style"]
        },
        ilyra
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/character-1/enhance-field") {
          latestPayload = enhancedPayload;
          return {
            model: enhancedPayload,
            character_id: "character-1",
            field_name: "texting_style",
            created_count: 0,
            updated_count: 1,
            archived_count: 0,
            error: null
          };
        }
        return latestPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel
          activeSaveId="save-1"
          runJob={vi.fn(() => vi.fn())}
          characterTextsEnabled
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.clear(screen.getByLabelText("Role"));
    await userEvent.type(screen.getByLabelText("Role"), "Beacon courier");
    await userEvent.click(screen.getByRole("button", { name: "Auto-enhance Texting Style" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/character-1/enhance-field", expect.anything()));
    const enhanceCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/character-1/enhance-field");
    expect(JSON.parse(String(enhanceCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      field_name: "texting_style",
      character: {
        character_id: "character-1",
        role: "Beacon courier",
        texting_style: "Short replies after midnight."
      }
    });
    expect(screen.getByLabelText("Texting Style")).toHaveValue(
      "Short replies after midnight.\n\nLowercase bursts, double texts when worried, one sparkle emoji max."
    );
    expect(screen.queryByText("Unsaved changes")).not.toBeInTheDocument();
  });

  it("shows a notice when character enhancement returns no new field details", async () => {
    const basePayload = characterRegistryPayload();
    const mara = basePayload.characters?.[0];
    const ilyra = basePayload.characters?.[1];
    if (!mara || !ilyra) throw new Error("Expected character fixtures");
    let latestPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          role: "Scout",
          appearance: "Ash-dusted cloak.",
          locked_fields: []
        },
        ilyra
      ]
    });
    const noopPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          role: "Beacon courier",
          appearance: "Ash-dusted cloak.",
          locked_fields: []
        },
        ilyra
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/character-1/enhance-field") {
          latestPayload = noopPayload;
          return {
            model: noopPayload,
            character_id: "character-1",
            field_name: "appearance",
            created_count: 0,
            updated_count: 1,
            archived_count: 0,
            field_changed: false,
            notice: "No new Appearance details were found; the field was left unchanged.",
            error: null
          };
        }
        return latestPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.clear(screen.getByLabelText("Role"));
    await userEvent.type(screen.getByLabelText("Role"), "Beacon courier");
    await userEvent.click(screen.getByRole("button", { name: "Auto-enhance Appearance" }));

    expect(await screen.findByText("No new Appearance details were found; the field was left unchanged.")).toBeInTheDocument();
    expect(screen.getByLabelText("Appearance")).toHaveValue("Ash-dusted cloak.");
    expect(screen.getByLabelText("Role")).toHaveValue("Beacon courier");
  });

  it("auto-enhances character agency fields from the Agency tab", async () => {
    const basePayload = characterRegistryPayload();
    const mara = basePayload.characters?.[0];
    const ilyra = basePayload.characters?.[1];
    if (!mara || !ilyra) throw new Error("Expected character fixtures");
    let latestPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          goals: "Protect the beacon.",
          locked_fields: []
        },
        ilyra
      ]
    });
    const enhancedPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          goals: "Protect the beacon.\n\nKeep the red lens stable until dawn.",
          locked_fields: ["goals"]
        },
        ilyra
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/character-1/enhance-field") {
          latestPayload = enhancedPayload;
          return {
            model: enhancedPayload,
            character_id: "character-1",
            field_name: "goals",
            created_count: 0,
            updated_count: 1,
            archived_count: 0,
            error: null
          };
        }
        return latestPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.click(screen.getByRole("tab", { name: "Agency" }));
    await userEvent.click(screen.getByRole("button", { name: "Auto-enhance Goals" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/character-1/enhance-field", expect.anything()));
    const enhanceCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/character-1/enhance-field");
    expect(JSON.parse(String(enhanceCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      field_name: "goals",
      character: {
        character_id: "character-1",
        goals: "Protect the beacon."
      }
    });
    expect(screen.getByLabelText("Goals")).toHaveValue("Protect the beacon.\n\nKeep the red lens stable until dawn.");
  });

  it("links and unlinks character summary knowledge targets", async () => {
    const initialPayload = characterRegistryPayload();
    const mara = initialPayload.characters?.[0];
    const ilyra = initialPayload.characters?.[1];
    if (!mara || !ilyra) throw new Error("Expected character fixtures");
    const unlinkedPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          linked_summary_ids: []
        },
        ilyra
      ],
      link_targets: initialPayload.link_targets?.map((target) => {
        if (target.target_id === "summary-1") return { ...target, linked_character_ids: [] };
        return target;
      })
    });
    const linkedPayload = characterRegistryPayload({
      characters: [
        {
          ...mara,
          linked_summary_ids: ["summary-2"]
        },
        ilyra
      ],
      link_targets: initialPayload.link_targets?.map((target) => {
        if (target.target_id === "summary-1") return { ...target, linked_character_ids: [] };
        if (target.target_id === "summary-2") return { ...target, linked_character_ids: ["character-1"] };
        return target;
      })
    });
    let latestPayload = initialPayload;
    let applyCount = 0;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/character-1/knowledge/apply") {
          applyCount += 1;
          latestPayload = applyCount === 1 ? unlinkedPayload : linkedPayload;
          return { model: latestPayload, created_count: 0, updated_count: 1, archived_count: 0 };
        }
        return latestPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.click(screen.getByRole("tab", { name: "Knowledge" }));
    await userEvent.click(screen.getByRole("button", { name: "Unlink summary Beacon summary" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/character-1/knowledge/apply", expect.anything()));
    const unlinkCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/character-1/knowledge/apply");
    expect(JSON.parse(String(unlinkCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      actions: [{ action: "unlink", target_type: "summary", target_id: "summary-1" }]
    });

    await userEvent.click(await screen.findByRole("button", { name: "Link summary Fog route" }));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/characters/character-1/knowledge/apply")).toHaveLength(2));
    const linkCalls = fetchMock.mock.calls.filter(([path]) => path === "/api/characters/character-1/knowledge/apply");
    const linkCall = linkCalls[linkCalls.length - 1];
    expect(JSON.parse(String(linkCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      actions: [{ action: "link", target_type: "summary", target_id: "summary-2" }]
    });
  });

  it("edits character aliases from the character panel", async () => {
    const characterPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "Signal runner",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: ["memory-1", "missing-memory"],
          linked_state_ids: ["missing-state"],
          linked_summary_ids: ["missing-summary"]
        }
      ],
      link_targets: [
        {
          target_type: "memory",
          target_id: "memory-1",
          title: "Memory",
          body: "Known active-save memory",
          linked_character_ids: ["character-1"]
        }
      ],
      location_choices: []
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/characters/apply"
        ? { model: characterPayload, created_count: 0, updated_count: 1, archived_count: 0 }
        : characterPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    const aliases = screen.getByLabelText("Aliases");
    await userEvent.clear(aliases);
    await userEvent.type(aliases, "Signal runner, Ember");
    await userEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      edits: {
        characters: [
          {
            character_id: "character-1",
            aliases_text: "Signal runner, Ember",
            linked_memory_ids: ["memory-1"],
            linked_state_ids: [],
            linked_summary_ids: []
          }
        ]
      }
    });
  });

  it("exposes the contact name field when texting is enabled for the save", async () => {
    const characterPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          contact_name: "Mar",
          texting_style: "Short replies after midnight.",
          aliases_text: "",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: []
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/characters/apply"
        ? { model: characterPayload, created_count: 0, updated_count: 1, archived_count: 0 }
        : characterPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel
          activeSaveId="save-1"
          runJob={vi.fn(() => vi.fn())}
          characterTextsEnabled
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await screen.findByLabelText("Aliases");
    expect(await screen.findByDisplayValue("Mar")).toBeInTheDocument();
    expect(screen.getByText("Contact Name")).toBeInTheDocument();
    expect(screen.getByLabelText("Texting Style")).toHaveValue("Short replies after midnight.");
  });

  it("saves character texting style from the character panel", async () => {
    const characterPayload = characterRegistryPayload({
      characters: [
        {
          ...characterRegistryPayload().characters![0],
          texting_style: "Short replies after midnight."
        }
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/characters/apply"
        ? { model: characterPayload, created_count: 0, updated_count: 1, archived_count: 0 }
        : characterPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel
          activeSaveId="save-1"
          runJob={vi.fn(() => vi.fn())}
          characterTextsEnabled
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    const textingStyle = screen.getByLabelText("Texting Style");
    await userEvent.clear(textingStyle);
    await userEvent.type(
      textingStyle,
      "Lowercase bursts, double texts when worried, one sparkle emoji max."
    );
    await userEvent.click(screen.getByRole("button", { name: "Save character" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      edits: {
        characters: [
          {
            character_id: "character-1",
            texting_style: "Lowercase bursts, double texts when worried, one sparkle emoji max."
          }
        ]
      }
    });
  });

  it("hides the contact name field when texting is disabled for the save", async () => {
    const characterPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          contact_name: "",
          aliases_text: "",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: []
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve({
      ok: true,
      json: async () => characterPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel
          activeSaveId="save-1"
          runJob={vi.fn(() => vi.fn())}
          characterTextsEnabled={false}
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    expect(screen.getByLabelText("Aliases")).toBeInTheDocument();
    expect(screen.queryByPlaceholderText("Optional phone contact name")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Texting Style")).not.toBeInTheDocument();
    expect(screen.queryByText("Phone nickname; falls back to the character name when blank.")).not.toBeInTheDocument();
  });

  it("creates and links character knowledge from dossier controls without raw JSON", async () => {
    const initialPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "Signal runner",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{\"Ilyra\":\"trusted contact\"}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: ["memory-1"],
          linked_state_ids: [],
          linked_summary_ids: []
        }
      ],
      link_targets: [
        {
          target_type: "memory",
          target_id: "memory-1",
          title: "Memory",
          body: "Mara knows Ilyra keeps a copper key.",
          tags: ["mara"],
          importance: 0.7,
          linked_character_ids: ["character-1"]
        },
        {
          target_type: "world_state",
          target_id: "state-1",
          title: "beacon.lens",
          body: "Failsafe: copper notch",
          value: { failsafe: "copper notch" },
          category: "artifact",
          confidence: 0.8,
          linked_character_ids: []
        },
        {
          target_type: "summary",
          target_id: "summary-1",
          title: "Summary",
          body: "Ilyra warned Mara about the red lens.",
          linked_character_ids: []
        }
      ],
      location_choices: []
    };
    const updatedPayload = {
      ...initialPayload,
      characters: [
        {
          ...initialPayload.characters[0],
          linked_memory_ids: ["memory-1", "memory-2"],
          linked_state_ids: ["state-1"],
          linked_summary_ids: []
        }
      ],
      link_targets: [
        ...initialPayload.link_targets.map((target) => target.target_id === "state-1" ? { ...target, linked_character_ids: ["character-1"] } : target),
        {
          target_type: "memory",
          target_id: "memory-2",
          title: "Memory",
          body: "Mara knows the copper key is hidden in the bell frame.",
          tags: ["mara", "key"],
          importance: 0.72,
          linked_character_ids: ["character-1"]
        }
      ]
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/characters/character-1/knowledge/apply"
        ? { model: updatedPayload, created_count: 1, updated_count: 0, archived_count: 0 }
        : initialPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.click(screen.getByRole("tab", { name: "Knowledge" }));

    expect(screen.queryByLabelText("Knowledge Links")).not.toBeInTheDocument();
    expect(screen.queryByText(/raw json|relationships_json|value_json|fact value json/i)).not.toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Link fact beacon.lens" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/character-1/knowledge/apply", expect.anything()));
    let applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/character-1/knowledge/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      actions: [{ action: "link", target_type: "world_state", target_id: "state-1" }]
    });

    await userEvent.click(await screen.findByRole("button", { name: "Add memory" }));
    fireEvent.change(screen.getByLabelText("Memory body"), { target: { value: "Mara knows the copper key is hidden in the bell frame." } });
    fireEvent.change(screen.getByLabelText("Tags"), { target: { value: "mara, key" } });
    fireEvent.change(screen.getByLabelText("Importance"), { target: { value: "0.72" } });
    await userEvent.click(screen.getByRole("button", { name: "Save memory" }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/characters/character-1/knowledge/apply").length).toBeGreaterThanOrEqual(2));
    const knowledgeCalls = fetchMock.mock.calls.filter(([path]) => path === "/api/characters/character-1/knowledge/apply");
    applyCall = knowledgeCalls[knowledgeCalls.length - 1];
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      active_save_id: "save-1",
      actions: [
        {
          action: "create_memory",
          body: "Mara knows the copper key is hidden in the bell frame.",
          tags: ["mara", "key"],
          importance: 0.72
        }
      ]
    });
  });

  it("edits character relationships with structured rows instead of JSON", async () => {
    const characterPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "Signal runner",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{\"Ilyra\":\"trusted contact\"}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: []
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/characters/apply"
        ? { model: characterPayload, created_count: 0, updated_count: 1, archived_count: 0 }
        : characterPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));

    expect(screen.queryByText(/relationships_json|\{\"Ilyra\"/i)).not.toBeInTheDocument();
    expect(screen.getByLabelText("Relationship name Ilyra")).toHaveValue("Ilyra");
    const removeRelationship = screen.getByRole("button", { name: "Remove relationship Ilyra" });
    expect(removeRelationship).toHaveClass("touch-labeled-action");
    expect(within(removeRelationship).getByText("Remove")).toHaveClass("touch-action-label");
    fireEvent.change(screen.getByLabelText("Relationship note Ilyra"), { target: { value: "trusted contact and lens-key witness" } });
    await userEvent.click(screen.getByRole("button", { name: "Save character" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      edits: {
        characters: [
          {
            character_id: "character-1",
            relationships_json: "{\"Ilyra\":\"trusted contact and lens-key witness\"}"
          }
        ]
      }
    });
  });

  it("submits explicit character fact locks from the character panel", async () => {
    const characterPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "Signal runner",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "Ash-dusted cloak.",
          visual_notes: "",
          current_clothing: "Sleeveless gray work tunic.",
          personality: "",
          voice: "Low and clipped.",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: [],
          locked_fields: ["voice"]
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/characters/apply"
        ? { model: characterPayload, created_count: 0, updated_count: 1, archived_count: 0 }
        : characterPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.clear(screen.getByLabelText("Current clothing"));
    await userEvent.type(
      screen.getByLabelText("Current clothing"),
      "Borrowed green raincoat over a linen shirt."
    );
    await userEvent.click(screen.getByRole("tab", { name: "Locks" }));
    expect(screen.getByLabelText("Lock Voice")).toBeChecked();
    await userEvent.click(screen.getByLabelText("Lock Voice"));
    await userEvent.click(screen.getByLabelText("Lock Appearance"));
    await userEvent.click(screen.getByLabelText("Lock Current clothing"));
    await userEvent.click(screen.getByRole("button", { name: /save/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/apply", expect.anything()));
    const applyCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/apply");
    expect(JSON.parse(String(applyCall?.[1].body))).toMatchObject({
      edits: {
        characters: [
          {
            character_id: "character-1",
            current_clothing: "Borrowed green raincoat over a linen shirt.",
            locked_fields: ["appearance", "current_clothing"]
          }
        ]
      }
    });
  });

  it("imports and exports character bundles from the character panel", async () => {
    const openMock = vi.fn();
    vi.stubGlobal("open", openMock);
    const characterPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "Signal runner",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: []
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const importedPayload = {
      ...characterPayload,
      characters: [
        ...characterPayload.characters,
        {
          ...characterPayload.characters[0],
          character_id: "character-imported",
          name: "Mara of the North"
        }
      ]
    };
    let latestPayload = characterPayload;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/character-bundles/preview") {
          return {
            preview_id: "preview-character-1",
            preview: {
              character_id: "character-source",
              name: "Mara",
              suggested_name: "Mara (imported)",
              name_conflict: true,
              media_count: 1,
              bundle_version: 1,
              aliases: ["Ember"],
              role: "Signal runner",
              known_state: "Carries the amber lens.",
              appearance: "Ash-dusted cloak.",
              current_clothing: "Borrowed green raincoat over a linen shirt.",
              personality: "Careful and dry-witted.",
              voice: "Low and clipped.",
              status: "traveling",
              created_at: null,
              updated_at: null,
              exported_at: null,
              skipped_media_count: 0,
              warnings: ["Reference images will be restored as character links."]
            }
          };
        }
        if (path === "/api/character-bundles/import/preview-character-1") {
          latestPayload = importedPayload;
          return importedPayload;
        }
        return latestPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await screen.findByText("Mara");
    await userEvent.click(screen.getByRole("button", { name: "Export Mara" }));
    const exportDialog = await screen.findByRole("dialog", { name: "Export character bundle?" });
    expect(within(exportDialog).getByLabelText("Include private notes")).not.toBeChecked();
    await userEvent.click(within(exportDialog).getByRole("button", { name: "Export" }));
    expect(openMock).toHaveBeenCalledWith("/api/character-bundles/export/character-1", "_blank", "noopener,noreferrer");

    const file = new File(["character"], "mara.bragi-character", { type: "application/octet-stream" });
    await userEvent.upload(screen.getByLabelText("Character bundle file"), file);
    expect(await screen.findByRole("dialog", { name: "Import character?" })).toBeInTheDocument();
    expect(screen.getByLabelText("Import name")).toHaveValue("Mara (imported)");
    expect(screen.getAllByText("Signal runner").length).toBeGreaterThan(0);
    expect(screen.getByText("Careful and dry-witted.")).toBeInTheDocument();
    expect(screen.getByText("A character with this name already exists.")).toBeInTheDocument();

    await userEvent.clear(screen.getByLabelText("Import name"));
    await userEvent.type(screen.getByLabelText("Import name"), "Mara of the North");
    await userEvent.click(screen.getByRole("button", { name: "Import" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/character-bundles/import/preview-character-1", expect.anything()));
    const importCall = fetchMock.mock.calls.find(([path]) => path === "/api/character-bundles/import/preview-character-1");
    expect(JSON.parse(String(importCall?.[1].body))).toEqual({
      active_save_id: "save-1",
      name: "Mara of the North"
    });
    expect(
      (await screen.findAllByText("Mara of the North")).length
    ).toBeGreaterThan(0);
  });

  it("lets admins explicitly opt in before exporting character private notes", async () => {
    const openMock = vi.fn();
    vi.stubGlobal("open", openMock);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        active_save_id: "save-1",
        characters: [
          {
            character_id: "character-1",
            name: "Mara",
            aliases_text: "",
            role: "Signal runner",
            known_state: "",
            met: true,
            appearance: "",
            visual_notes: "",
            personality: "",
            voice: "",
            relationships_json: "{}",
            status: "present",
            location_id: null,
            private_notes: "Keep the lens secret.",
            present: true,
            linked_memory_ids: [],
            linked_state_ids: [],
            linked_summary_ids: [],
            reference_image: null
          }
        ],
        link_targets: [],
        location_choices: []
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel
          activeSaveId="save-1"
          runJob={vi.fn(() => vi.fn())}
          currentUser={{ id: "admin-1", username: "Mira", role: "admin", status: "active" }}
        />
      </QueryClientProvider>
    );

    await screen.findByText("Mara");
    await userEvent.click(screen.getByRole("button", { name: "Export Mara" }));
    const exportDialog = await screen.findByRole("dialog", { name: "Export character bundle?" });
    await userEvent.click(within(exportDialog).getByLabelText("Include private notes"));
    await userEvent.click(within(exportDialog).getByRole("button", { name: "Export" }));

    expect(openMock).toHaveBeenCalledWith(
      "/api/character-bundles/export/character-1?include_private_notes=1",
      "_blank",
      "noopener,noreferrer"
    );
  });

  it("hides private notes export opt-in from regular users", async () => {
    const openMock = vi.fn();
    vi.stubGlobal("open", openMock);
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        active_save_id: "save-1",
        characters: [
          {
            character_id: "character-1",
            name: "Mara",
            aliases_text: "",
            role: "Signal runner",
            known_state: "",
            met: true,
            appearance: "",
            visual_notes: "",
            personality: "",
            voice: "",
            relationships_json: "{}",
            status: "present",
            location_id: null,
            private_notes: "Keep the lens secret.",
            present: true,
            linked_memory_ids: [],
            linked_state_ids: [],
            linked_summary_ids: [],
            reference_image: null
          }
        ],
        link_targets: [],
        location_choices: []
      })
    });
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel
          activeSaveId="save-1"
          runJob={vi.fn(() => vi.fn())}
          currentUser={{ id: "user-1", username: "Rook", role: "user", status: "active" }}
        />
      </QueryClientProvider>
    );

    await screen.findByText("Mara");
    await userEvent.click(screen.getByRole("button", { name: "Export Mara" }));
    const exportDialog = await screen.findByRole("dialog", { name: "Export character bundle?" });
    expect(within(exportDialog).queryByLabelText("Include private notes")).not.toBeInTheDocument();
    await userEvent.click(within(exportDialog).getByRole("button", { name: "Export" }));

    expect(openMock).toHaveBeenCalledWith(
      "/api/character-bundles/export/character-1",
      "_blank",
      "noopener,noreferrer"
    );
  });

  it("manages scoped character reference images from the character panel", async () => {
    const basePayload: CharacterRegistryModel = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: [],
          reference_image: null
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const baseCharacter = basePayload.characters?.[0];
    if (!baseCharacter) throw new Error("Expected character fixture");
    const withReference: CharacterRegistryModel = {
      ...basePayload,
      characters: [
        {
          ...baseCharacter,
          reference_image: {
            media_asset_id: "media-reference-1",
            mime_type: "image/png",
            prompt_preview: "Uploaded character reference image",
            provider: "local",
            model: "upload",
            created_at: null,
            source: "uploaded"
          }
        }
      ]
    };
    let latestPayload: CharacterRegistryModel = basePayload;
    const job = {
      id: "job-reference",
      type: "character_reference_image",
      status: "queued",
      result: null,
      error: null,
      created_at: 1,
      save_id: "save-1"
    };
    const uploadJob = {
      id: "job-reference-upload",
      type: "character_reference_upload",
      status: "queued",
      result: null,
      error: null,
      created_at: 2,
      save_id: "save-1"
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/character-1/reference-image/generate") return job;
        if (path === "/api/characters/character-1/reference-image/upload") {
          latestPayload = withReference;
          return uploadJob;
        }
        if (path === "/api/characters/character-1/reference-image/remove") {
          latestPayload = basePayload;
          return basePayload;
        }
        return latestPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn((_: unknown, options?: { onSucceeded?: (result: unknown) => void }) => {
      options?.onSucceeded?.({});
      return vi.fn();
    });
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={runJob} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.click(screen.getByRole("button", { name: "Generate" }));
    expect(runJob).toHaveBeenCalledWith(expect.objectContaining({ id: "job-reference" }), expect.anything());
    const generateCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/character-1/reference-image/generate");
    expect(JSON.parse(String(generateCall?.[1].body))).toMatchObject({
      save_id: "save-1",
      replace_existing: false
    });

    const file = new File([new Uint8Array([137, 80, 78, 71])], "mara.png", { type: "image/png" });
    await userEvent.upload(screen.getByLabelText("Upload character reference image"), file);
    await screen.findByAltText("Uploaded character reference image");
    const uploadCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/character-1/reference-image/upload");
    expect(uploadCall?.[1].body).toBeInstanceOf(FormData);

    await userEvent.click(screen.getByRole("button", { name: "Remove" }));
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/characters/character-1/reference-image/remove", expect.anything()));
    const removeCall = fetchMock.mock.calls.find(([path]) => path === "/api/characters/character-1/reference-image/remove");
    expect(JSON.parse(String(removeCall?.[1].body))).toEqual({ save_id: "save-1" });
  });

  it("generates registry character pictures from the pictures tab", async () => {
    const characterPayload: CharacterRegistryModel = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: [],
          reference_image: {
            media_asset_id: "media-reference-1",
            mime_type: "image/png",
            prompt_preview: "Mara reference",
            provider: "local",
            model: "upload",
            created_at: null,
            source: "uploaded"
          },
          generated_images: [
            {
              media_asset_id: "media-generated-1",
              mime_type: "image/png",
              prompt_preview: "Mara in the storm",
              provider: "venice",
              model: "image-to-image",
              created_at: null,
              source: "generated"
            }
          ]
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const job = {
      id: "job-character-picture",
      type: "character_image_generation",
      status: "queued",
      result: null,
      error: null,
      created_at: 1,
      save_id: "save-1"
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/character-1/image/generate") return job;
        return characterPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn((_: unknown, options?: { onSucceeded?: (result: unknown) => void }) => {
      options?.onSucceeded?.({});
      return vi.fn();
    });
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={runJob} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.click(screen.getByRole("tab", { name: "Pictures" }));
    expect(screen.getByAltText("Mara in the storm")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Generate image of this character" }));
    const dialog = await screen.findByRole("dialog", { name: "Generate image of Mara" });
    expect(within(dialog).getByAltText("Mara reference")).toBeInTheDocument();
    await userEvent.type(within(dialog).getByLabelText("Instructions"), "winter cloak");
    await userEvent.click(within(dialog).getByRole("button", { name: "Generate" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/characters/character-1/image/generate",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          save_id: "save-1",
          instructions: "winter cloak"
        })
      })
    ));
    expect(runJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: "job-character-picture" }),
      expect.objectContaining({ onSucceeded: expect.any(Function) })
    );
  });

  it("sets a registry character picture as the character reference", async () => {
    const initialPayload: CharacterRegistryModel = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: [],
          reference_image: {
            media_asset_id: "media-reference-1",
            mime_type: "image/png",
            prompt_preview: "Mara reference",
            provider: "local",
            model: "upload",
            created_at: null,
            source: "uploaded"
          },
          generated_images: [
            {
              media_asset_id: "media-generated-1",
              mime_type: "image/png",
              prompt_preview: "Mara in the storm",
              provider: "venice",
              model: "image-to-image",
              created_at: null,
              source: "generated"
            }
          ]
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const swappedPayload: CharacterRegistryModel = {
      ...initialPayload,
      characters: [
        {
          ...initialPayload.characters![0],
          reference_image: {
            media_asset_id: "media-generated-1",
            mime_type: "image/png",
            prompt_preview: "Mara in the storm",
            provider: "venice",
            model: "image-to-image",
            created_at: null,
            source: "generated"
          },
          generated_images: [
            {
              media_asset_id: "media-reference-1",
              mime_type: "image/png",
              prompt_preview: "Mara reference",
              provider: "local",
              model: "upload",
              created_at: null,
              source: "uploaded"
            }
          ]
        }
      ]
    };
    const job = {
      id: "job-reference-set",
      type: "character_reference_set",
      status: "queued",
      result: null,
      error: null,
      created_at: 1,
      save_id: "save-1"
    };
    let latestPayload = initialPayload;
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/characters/character-1/reference-image/set") return job;
        return latestPayload;
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn((_: unknown, options?: { applyResult?: boolean; onSucceeded?: (result: unknown) => void }) => {
      latestPayload = swappedPayload;
      options?.onSucceeded?.(swappedPayload);
      return vi.fn();
    });
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={runJob} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    await userEvent.click(screen.getByRole("tab", { name: "Pictures" }));
    await userEvent.click(screen.getByRole("button", { name: "Make reference: Mara in the storm" }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith(
      "/api/characters/character-1/reference-image/set",
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({
          save_id: "save-1",
          media_asset_id: "media-generated-1"
        })
      })
    ));
    expect(runJob).toHaveBeenCalledWith(
      expect.objectContaining({ id: "job-reference-set" }),
      expect.objectContaining({
        applyResult: false,
        onSucceeded: expect.any(Function)
      })
    );
    expect(await screen.findByAltText("Mara reference")).toBeInTheDocument();
    expect(screen.queryByAltText("Mara in the storm")).not.toBeInTheDocument();
  });

  it("blocks blank character names in the character panel", async () => {
    const characterPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: []
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => characterPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-1" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));
    const name = screen.getByPlaceholderText("Name");
    await userEvent.clear(name);
    await userEvent.click(screen.getByRole("button", { name: /save/i }));

    expect(screen.getByText("Character name must not be blank")).toBeInTheDocument();
    expect(name).toHaveAttribute("aria-invalid", "true");
    expect(screen.getByRole("button", { name: /save/i })).toBeDisabled();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/characters/apply")).toBe(false);
  });

  it("blocks stale character edits after the active save changes", async () => {
    const characterPayload = {
      active_save_id: "save-1",
      characters: [
        {
          character_id: "character-1",
          name: "Mara",
          aliases_text: "Signal runner",
          role: "Scout",
          known_state: "",
          met: true,
          appearance: "",
          visual_notes: "",
          personality: "",
          voice: "",
          relationships_json: "{}",
          status: "present",
          location_id: null,
          private_notes: "",
          present: true,
          linked_memory_ids: [],
          linked_state_ids: [],
          linked_summary_ids: []
        }
      ],
      link_targets: [],
      location_choices: []
    };
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, json: async () => characterPayload });
    vi.stubGlobal("fetch", fetchMock);
    const { CharactersPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <CharactersPanel activeSaveId="save-2" runJob={vi.fn(() => vi.fn())} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByText("Mara"));

    expect(screen.getByLabelText("Aliases")).toBeDisabled();
    expect(screen.getByTitle("Add character")).toBeDisabled();
    expect(screen.getByRole("button", { name: /save/i })).toBeDisabled();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/characters/apply")).toBe(false);
  });

  it("exposes the chronicle transcript as a polite live log", async () => {
    const { Chronicle } = await import("./main");

    render(
      <Chronicle
        model={runtimeModel({
          chronicle: {
            messages: [
              { message_id: "m1", role: "player", speaker_name: "Keeper", body: "First", actions: [] },
              { message_id: "m2", role: "narrator", speaker_name: null, body: "Latest", actions: [] }
            ]
          }
        })}
        runJob={vi.fn()}
        pendingMessage={null}
      />
    );

    const log = screen.getByRole("log", { name: "Chronicle" });
    expect(log).toHaveClass("chronicle-scroll");
    expect(log).toHaveAttribute("aria-live", "polite");
    expect(log).toHaveAttribute("aria-relevant", "additions text");
    expect(log).toHaveAttribute("aria-atomic", "false");
  });

  it("renders a bounded chronicle window for long saves", async () => {
    const { Chronicle } = await import("./main");
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      value: 360
    });
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      value: 12000
    });
    const messages: RuntimeModel["chronicle"]["messages"] = Array.from({ length: 80 }, (_, index) => ({
      message_id: `m${index + 1}`,
      role: index % 2 ? "narrator" : "player",
      speaker_name: index % 2 ? null : "Keeper",
      body: `Chronicle message ${index + 1}`,
      actions: []
    }));

    render(
      <Chronicle
        model={runtimeModel({ chronicle: { messages } })}
        runJob={vi.fn()}
        pendingMessage={null}
      />
    );

    const log = screen.getByRole("log", { name: "Chronicle" });
    await waitFor(() => expect(within(log).getByText("Chronicle message 80")).toBeInTheDocument());
    expect(log.querySelectorAll(".message").length).toBeLessThan(40);
    expect(within(log).queryByText("Chronicle message 20")).not.toBeInTheDocument();
  });

  it("suppresses chronicle live announcements while loading earlier history", async () => {
    const { Chronicle } = await import("./main");
    const page = deferred<{ ok: boolean; json: () => Promise<{ messages: RuntimeModel["chronicle"]["messages"]; has_more_before: boolean; oldest_message_id: string }> }>();
    const fetchMock = vi.fn().mockImplementation((path: string) => (
      path.startsWith("/api/saves/save-1/chronicle")
        ? page.promise
        : Promise.resolve({ ok: true, json: async () => ({}) })
    ));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <Chronicle
        model={runtimeModel({
          chronicle: {
            has_more_before: true,
            oldest_message_id: "m1",
            messages: [
              { message_id: "m1", role: "player", speaker_name: "Keeper", body: "First", actions: [] }
            ]
          }
        })}
        runJob={vi.fn()}
        pendingMessage={null}
      />
    );

    const log = screen.getByRole("log", { name: "Chronicle" });
    expect(log).toHaveAttribute("aria-live", "polite");

    await userEvent.click(screen.getByRole("button", { name: /Load earlier/i }));

    expect(log).toHaveAttribute("aria-live", "off");

    page.resolve({
      ok: true,
      json: async () => ({
        messages: [
          { message_id: "m0", role: "narrator", speaker_name: null, body: "Earlier", actions: [] }
        ],
        has_more_before: false,
        oldest_message_id: "m0"
      })
    });

    await waitFor(() => expect(log).toHaveAttribute("aria-live", "polite"));
  });

  it("keeps chronicle scroll position stable when older messages are prepended", async () => {
    const { Chronicle } = await import("./main");
    let scrollHeightValue = 900;
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      get: () => scrollHeightValue
    });
    const page = deferred<{ ok: boolean; json: () => Promise<{ messages: RuntimeModel["chronicle"]["messages"]; has_more_before: boolean; oldest_message_id: string }> }>();
    const fetchMock = vi.fn().mockImplementation((path: string) => (
      path.startsWith("/api/saves/save-1/chronicle")
        ? page.promise
        : Promise.resolve({ ok: true, json: async () => ({}) })
    ));
    vi.stubGlobal("fetch", fetchMock);

    render(
      <Chronicle
        model={runtimeModel({
          chronicle: {
            has_more_before: true,
            oldest_message_id: "m2",
            messages: [
              { message_id: "m2", role: "player", speaker_name: "Keeper", body: "Current top", actions: [] },
              { message_id: "m3", role: "narrator", speaker_name: null, body: "Latest", actions: [] }
            ]
          }
        })}
        runJob={vi.fn()}
        pendingMessage={null}
      />
    );

    const log = screen.getByRole("log", { name: "Chronicle" });
    log.scrollTop = 240;
    fireEvent.scroll(log);
    await userEvent.click(screen.getByRole("button", { name: /Load earlier/i }));
    scrollHeightValue = 1120;
    page.resolve({
      ok: true,
      json: async () => ({
        messages: [
          { message_id: "m1", role: "narrator", speaker_name: null, body: "Older", actions: [] }
        ],
        has_more_before: false,
        oldest_message_id: "m1"
      })
    });

    await waitFor(() => expect(log.scrollTop).toBe(460));
  });

  it("starts the chronicle at the latest message", async () => {
    const { Chronicle } = await import("./main");
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      value: 720
    });

    render(
      <Chronicle
        model={{
          saves: [],
          active_save_id: "save-1",
          active_save_title: "Lantern Keep",
          active_scenario_type: null,
          scenario_title: "Lantern Keep",
          scene_title: "Beacon",
          chronicle: {
            messages: [
              { message_id: "m1", role: "player", speaker_name: "Keeper", body: "First", actions: [] },
              { message_id: "m2", role: "narrator", speaker_name: null, body: "Latest", actions: [] }
            ]
          },
          media: null,
          action_choices: null,
          model_indicator: "",
          failed_save: false,
          composer_enabled: true,
          failure_text: null,
          status: null,
          error: null
        }}
        runJob={vi.fn()}
        pendingMessage={null}
      />
    );

    expect(screen.getByText("Latest").closest(".chronicle-scroll")?.scrollTop).toBe(720);
  });

  it("auto-scrolls chronicle updates when the reader is near the latest message", async () => {
    const { Chronicle } = await import("./main");
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      value: 1000
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      value: 300
    });
    const firstModel = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m1", role: "player", speaker_name: "Keeper", body: "First", actions: [] }
        ]
      }
    });
    const secondModel = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m1", role: "player", speaker_name: "Keeper", body: "First", actions: [] },
          { message_id: "m2", role: "narrator", speaker_name: null, body: "Latest", actions: [] }
        ]
      }
    });

    const { rerender } = render(<Chronicle model={firstModel} runJob={vi.fn()} pendingMessage={null} />);
    const scroll = screen.getByText("First").closest(".chronicle-scroll") as HTMLElement;
    scroll.scrollTop = 630;

    rerender(<Chronicle model={secondModel} runJob={vi.fn()} pendingMessage={null} />);

    expect(scroll.scrollTop).toBe(1000);
    expect(screen.queryByRole("button", { name: "Jump to latest" })).not.toBeInTheDocument();
  });

  it("preserves chronicle position and offers a latest jump when updates arrive while scrolled up", async () => {
    const { Chronicle } = await import("./main");
    Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
      configurable: true,
      value: 1000
    });
    Object.defineProperty(HTMLElement.prototype, "clientHeight", {
      configurable: true,
      value: 300
    });
    const firstModel = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m1", role: "player", speaker_name: "Keeper", body: "First", actions: [] }
        ]
      }
    });
    const secondModel = runtimeModel({
      chronicle: {
        messages: [
          { message_id: "m1", role: "player", speaker_name: "Keeper", body: "First", actions: [] },
          { message_id: "m2", role: "narrator", speaker_name: null, body: "Latest", actions: [] }
        ]
      }
    });

    const { rerender } = render(<Chronicle model={firstModel} runJob={vi.fn()} pendingMessage={null} />);
    const scroll = screen.getByText("First").closest(".chronicle-scroll") as HTMLElement;
    scroll.scrollTop = 120;

    rerender(<Chronicle model={secondModel} runJob={vi.fn()} pendingMessage={null} />);

    expect(scroll.scrollTop).toBe(120);
    await userEvent.click(screen.getByRole("button", { name: "Jump to latest" }));
    expect(scroll.scrollTop).toBe(1000);
    expect(screen.queryByRole("button", { name: "Jump to latest" })).not.toBeInTheDocument();
  });

  it("separates save-scoped settings from local settings", async () => {
    const settingsPayload = {
      provider_cards: [
        {
          provider: "fake",
          enabled: true,
          has_api_key: false,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "ready",
          last_error: null
        }
      ],
      task_model_selectors: [],
      roleplay_shared_models: { setting_key: "roleplay_shared_models", enabled: true },
      roleplay_model_groups: [],
      visible_sections: ["providers", "models", "save", "local", "diagnostics"],
      automatic_summarization: { setting_key: "automatic_summarization_enabled", enabled: true },
      summarization_context_pressure_threshold: { setting_key: "summarization_context_pressure_threshold", value: 0.75, minimum: 0.1, maximum: 1, step: 0.05 },
      summarization_visibility: { setting_key: "show_summarization_activity", enabled: false },
      agentic_context_pipeline: { setting_key: "agentic_context_pipeline_enabled", enabled: false },
      plan_first_narrator: { setting_key: "plan_first_narrator_enabled", enabled: true },
      director_pressure: { setting_key: "director_pressure_enabled", enabled: true },
      character_action_planning: { setting_key: "character_action_planning_enabled", enabled: true },
      character_action_planning_max_concurrency: { setting_key: "character_action_planning_max_concurrency", value: 20, minimum: 1, maximum: 20, step: 1 },
      npc_knowledge_audit_mode: { setting_key: "npc_knowledge_audit_mode", selected: "soft_fail", options: ["soft_fail", "hard_fail"] },
      generated_phrase_denylist: { setting_key: "generated_phrase_denylist", value: "That's not nothing" },
      save_generated_phrase_denylist: { setting_key: "save_generated_phrase_denylist", value: "save-only phrase" },
      chat_fallback: { setting_key: "chat_fallback_enabled", enabled: false },
      structured_output_fallback: { setting_key: "structured_output_fallback_enabled", enabled: false },
      tool_call_fallback: { setting_key: "tool_call_fallback_enabled", enabled: false },
      image_fallback: { setting_key: "image_fallback_enabled", enabled: false },
      video_fallback: { setting_key: "video_fallback_enabled", enabled: false },
      venice_image_safe_mode: { setting_key: "venice_image_safe_mode", enabled: true },
      debug_logging: { setting_key: "debug_logging_enabled", enabled: false },
      pending_jobs_display_mode: { setting_key: "pending_jobs_display_mode", selected: "compact", options: ["compact", "expanded", "expanded_full"] },
      user_narration_guidance: { setting_key: "user_narration_guidance", value: "Keep it punchy." },
      content_rating: { setting_key: "content_filter_rating", selected: "pg-13", options: ["g", "pg", "pg-13", "r", "unrated"], admin_granted: false },
      fade_to_black: { setting_key: "fade_to_black_enabled", enabled: true },
      automatic_image_generation: { setting_key: "automatic_image_generation_enabled", enabled: true },
      automatic_media_mode: { setting_key: "automatic_media_mode", selected: "image", options: ["off", "image"] },
      image_style_preset: { setting_key: "image_style_preset", selected: "none", options: EXPECTED_IMAGE_STYLE_PRESETS },
      image_frequency: { setting_key: "image_generation_frequency", value: 3, minimum: 0, maximum: 999, step: 1 },
      manual_confirmation: {
        memories: { setting_key: "manual_confirmation_memories", enabled: false },
        character_registry: { setting_key: "manual_confirmation_character_registry", enabled: false },
        state_changes: { setting_key: "manual_confirmation_state_changes", enabled: false }
      },
      chat_history: {
        planner_player_messages: { setting_key: "narrator_planner_recent_player_message_window", value: 10, minimum: 0, maximum: 24, step: 1 },
        planner_narrator_messages: { setting_key: "narrator_planner_recent_narrator_message_window", value: 9, minimum: 0, maximum: 24, step: 1 },
        player_messages: { setting_key: "recent_player_message_window", value: 8, minimum: 0, maximum: 24, step: 1 },
        narrator_messages: { setting_key: "recent_narrator_message_window", value: 7, minimum: 0, maximum: 24, step: 1 }
      },
      context_budget: {
        mode: { setting_key: "context_budget_mode", selected: "adaptive", options: ["adaptive", "fixed"] },
        fixed_total_chars: { setting_key: "context_budget_fixed_total_chars", value: 24000, minimum: 1, step: 1 },
        adaptive_fraction: { setting_key: "context_budget_adaptive_fraction", value: 0.6, minimum: 0.01, maximum: 1 }
      }
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => isAnySettingsReadPath(path) ? settingsPayloadForPath(path, settingsPayload) : { saves: [], chronicle: { messages: [] } }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} activeSaveId="save-1" />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("tab", { name: "Providers" })).toHaveAttribute("title", expect.stringContaining("provider keys"));
    expect(screen.getByRole("tab", { name: "Save" })).toHaveAttribute("title", expect.stringContaining("active save"));
    expect(screen.getByRole("tab", { name: "Local" })).toHaveAttribute("title", expect.stringContaining("account"));
    expect(await screen.findByPlaceholderText("API key")).toHaveAttribute("title", "Paste the API key Bragi uses when calling fake.");
    expect(screen.getByTitle("Store this provider key locally for future requests.")).toBeInTheDocument();
    expect(screen.getByTitle("Fetch the current model list for fake.")).toBeInTheDocument();
    await userEvent.click(await screen.findByRole("tab", { name: "Save" }));
    expect(screen.getByRole("tab", { name: "Save" })).toHaveAttribute("aria-selected", "true");

    expect(screen.getByText("Summarization")).toBeInTheDocument();
    expect(screen.getByText("Context Automation")).toBeInTheDocument();
    expect(screen.getByText("Media Automation")).toBeInTheDocument();
    expect(screen.getByText("Chat History")).toBeInTheDocument();
    expect(screen.getByText("Narrator Planner")).toBeInTheDocument();
    expect(screen.getByText("Narrator Prose")).toBeInTheDocument();
    expect(screen.getByLabelText("Planner Player Messages")).toHaveValue(10);
    expect(screen.getByLabelText("Planner Narrator Messages")).toHaveValue(9);
    expect(screen.getByLabelText("Prose Player Messages")).toHaveValue(8);
    expect(screen.getByLabelText("Prose Narrator Messages")).toHaveValue(7);
    expect(screen.getByText("Context Budget")).toBeInTheDocument();
    expect(screen.queryByText("Workbench")).not.toBeInTheDocument();
    expect(screen.getByText("Automatic Summarization Enabled")).toBeInTheDocument();
    expect(screen.getByText("Automatic Summarization Enabled").closest("label")).toHaveAttribute("title", expect.stringContaining("summarizes older"));
    expect(screen.getByText("Agentic Context Pipeline").closest("label")).toHaveAttribute("title", expect.stringContaining("observation"));
    expect(screen.getByText("Plan-First Narrator").closest("label")).toHaveAttribute("title", expect.stringContaining("turn plan"));
    const characterConcurrency = screen.getByLabelText("Character Action Planning Max Concurrency");
    expect(characterConcurrency).toHaveValue(20);
    expect(characterConcurrency.closest("label")).toHaveAttribute("title", expect.stringContaining("NPC action planning"));
    const auditMode = screen.getByLabelText("NPC Knowledge Audit Mode");
    expect(auditMode.closest("label")).toHaveAttribute("title", expect.stringContaining("audit findings"));
    expect(within(auditMode).getByRole("option", { name: "Soft fail" })).toBeInTheDocument();
    expect(within(auditMode).getByRole("option", { name: "Hard fail" })).toBeInTheDocument();
    const globalPhraseDenylist = screen.getByRole("textbox", { name: "Global Phrase Denylist" });
    expect(globalPhraseDenylist).toHaveValue("That's not nothing");
    const savePhraseDenylist = screen.getByRole("textbox", { name: "Save Phrase Denylist" });
    expect(savePhraseDenylist).toHaveValue("save-only phrase");
    await userEvent.clear(savePhraseDenylist);
    await userEvent.type(savePhraseDenylist, "save-only phrase\nsave-only phrase");
    await userEvent.click(within(savePhraseDenylist.closest(".text-setting") as HTMLElement).getByRole("button", { name: /^save$/i }));
    await waitFor(() => expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .some(([, options]) => JSON.parse(String(options.body)).key === "save_generated_phrase_denylist")).toBe(true));
    const savePhraseCall = fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .find(([, options]) => JSON.parse(String(options.body)).key === "save_generated_phrase_denylist");
    expect(JSON.parse(String(savePhraseCall?.[1].body))).toEqual({
      key: "save_generated_phrase_denylist",
      value: "save-only phrase\nsave-only phrase",
      save_id: "save-1"
    });
    await userEvent.clear(globalPhraseDenylist);
    await userEvent.type(globalPhraseDenylist, "global phrase");
    await userEvent.click(within(globalPhraseDenylist.closest(".text-setting") as HTMLElement).getByRole("button", { name: /^save$/i }));
    await waitFor(() => expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .some(([, options]) => JSON.parse(String(options.body)).key === "generated_phrase_denylist")).toBe(true));
    const globalPhraseCall = fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .find(([, options]) => JSON.parse(String(options.body)).key === "generated_phrase_denylist");
    expect(JSON.parse(String(globalPhraseCall?.[1].body))).toEqual({
      key: "generated_phrase_denylist",
      value: "global phrase"
    });
    expect(screen.queryByText("Structured Output Fallback Enabled")).not.toBeInTheDocument();
    expect(screen.getByText("Venice Media Safe Mode").closest("label")).toHaveAttribute("title", expect.stringContaining("media safe mode"));
    expect(screen.queryByText("Uncensored Structured Output Fallback Enabled")).not.toBeInTheDocument();
    const imageStyle = screen.getByLabelText("Image Style Preset");
    expect(imageStyle.closest("label")).toHaveAttribute("title", expect.stringContaining("visual style"));
    expect(within(imageStyle).getByRole("option", { name: "No preset" })).toBeInTheDocument();
    expect(within(imageStyle).getByRole("option", { name: "3D Render" })).toBeInTheDocument();
    await userEvent.selectOptions(imageStyle, "pixel_art");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/settings/scoped", expect.anything()));
    const styleCall = fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .find(([, options]) => JSON.parse(String(options.body)).key === "image_style_preset");
    expect(JSON.parse(String(styleCall?.[1].body))).toEqual({
      key: "image_style_preset",
      value: "pixel_art",
      save_id: "save-1"
    });
    await userEvent.click(screen.getByLabelText("Agentic Context Pipeline"));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/scoped").length).toBeGreaterThanOrEqual(2));
    const agenticContextCall = fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .find(([, options]) => JSON.parse(String(options.body)).key === "agentic_context_pipeline_enabled");
    expect(JSON.parse(String(agenticContextCall?.[1].body))).toEqual({
      key: "agentic_context_pipeline_enabled",
      value: true,
      save_id: "save-1"
    });
    await userEvent.click(screen.getByLabelText("Plan-First Narrator"));
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/scoped").length).toBeGreaterThanOrEqual(3));
    const planFirstCall = fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .find(([, options]) => JSON.parse(String(options.body)).key === "plan_first_narrator_enabled");
    expect(JSON.parse(String(planFirstCall?.[1].body))).toEqual({
      key: "plan_first_narrator_enabled",
      value: false,
      save_id: "save-1"
    });
    await userEvent.selectOptions(auditMode, "hard_fail");
    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/scoped").length).toBeGreaterThanOrEqual(4));
    const auditModeCall = fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .find(([, options]) => JSON.parse(String(options.body)).key === "npc_knowledge_audit_mode");
    expect(JSON.parse(String(auditModeCall?.[1].body))).toEqual({
      key: "npc_knowledge_audit_mode",
      value: "hard_fail",
      save_id: "save-1"
    });
    expect(screen.getByText("Context Budget Adaptive Fraction")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Models" }));
    expect(screen.getByText("Fallback Behavior")).toBeInTheDocument();
    expect(screen.getByText("Structured Output Fallback Enabled")).toBeInTheDocument();
    expect(screen.getByText("Structured Output Fallback Enabled").closest("label")).toHaveAttribute("title", expect.stringContaining("structured maintenance"));

    await userEvent.click(screen.getByRole("tab", { name: "Local" }));
    expect(screen.getByRole("tab", { name: "Local" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByText("Workbench")).toBeInTheDocument();
    expect(screen.getByText("Local Recorder")).toBeInTheDocument();
    expect(screen.getByText("Content Safety")).toBeInTheDocument();
    const contentRating = screen.getByLabelText("Content Filtering Level");
    expect(contentRating).toHaveValue("pg-13");
    expect(within(contentRating).getByRole("option", { name: "PG-13" })).toBeInTheDocument();
    expect(within(contentRating).getByRole("option", { name: "Unrated" })).toBeInTheDocument();
    expect(screen.getByLabelText("Fade Explicit Content to Black")).toBeChecked();
    const pendingJobsMode = screen.getByLabelText("Pending Jobs Display Mode");
    expect(pendingJobsMode.closest("label")).toHaveAttribute("title", expect.stringContaining("pending jobs tray"));
    expect(within(pendingJobsMode).getByRole("option", { name: "Compact" })).toBeInTheDocument();
    expect(within(pendingJobsMode).getByRole("option", { name: "Expanded Turns" })).toBeInTheDocument();
    expect(within(pendingJobsMode).getByRole("option", { name: "Expanded Full" })).toBeInTheDocument();
    const accountGuidance = screen.getByRole("textbox", { name: "Narration Guidance" });
    expect(accountGuidance).toHaveValue("Keep it punchy.");
    await userEvent.clear(accountGuidance);
    await userEvent.type(accountGuidance, "Keep narrator responses to two paragraphs or less.");
    const accountGuidanceControl = accountGuidance.closest(".text-setting");
    expect(accountGuidanceControl).not.toBeNull();
    await userEvent.click(within(accountGuidanceControl as HTMLElement).getByRole("button", { name: /^save$/i }));
    await waitFor(() => expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .some(([, options]) => JSON.parse(String(options.body)).key === "user_narration_guidance")).toBe(true));
    const guidanceCall = fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .find(([, options]) => JSON.parse(String(options.body)).key === "user_narration_guidance");
    expect(JSON.parse(String(guidanceCall?.[1].body))).toEqual({
      key: "user_narration_guidance",
      value: "Keep narrator responses to two paragraphs or less."
    });
    await userEvent.selectOptions(pendingJobsMode, "expanded");
    await waitFor(() => expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .some(([, options]) => JSON.parse(String(options.body)).key === "pending_jobs_display_mode")).toBe(true));
    const displayModeCall = fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .find(([, options]) => JSON.parse(String(options.body)).key === "pending_jobs_display_mode");
    expect(JSON.parse(String(displayModeCall?.[1].body))).toEqual({
      key: "pending_jobs_display_mode",
      value: "expanded"
    });
    await userEvent.selectOptions(contentRating, "r");
    await userEvent.click(screen.getByLabelText("Fade Explicit Content to Black"));
    await waitFor(() => expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .some(([, options]) => JSON.parse(String(options.body)).key === "fade_to_black_enabled")).toBe(true));
    expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .map(([, options]) => JSON.parse(String(options.body))))
      .toEqual(expect.arrayContaining([
        { key: "content_filter_rating", value: "r" },
        { key: "fade_to_black_enabled", value: false }
      ]));
  });

  it("posts narrator planner history settings with the active save id", async () => {
    const settingsPayload = modelSettingsPayload({
      visible_sections: ["save"],
      chat_history: {
        planner_player_messages: { setting_key: "narrator_planner_recent_player_message_window", value: 10, minimum: 0, maximum: 24, step: 1 },
        planner_narrator_messages: { setting_key: "narrator_planner_recent_narrator_message_window", value: 9, minimum: 0, maximum: 24, step: 1 },
        player_messages: { setting_key: "recent_player_message_window", value: 8, minimum: 0, maximum: 24, step: 1 },
        narrator_messages: { setting_key: "recent_narrator_message_window", value: 7, minimum: 0, maximum: 24, step: 1 }
      }
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} activeSaveId="save-1" />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "Save" }));
    const plannerPlayer = screen.getByLabelText("Planner Player Messages");
    await userEvent.clear(plannerPlayer);
    await userEvent.type(plannerPlayer, "12");
    fireEvent.blur(plannerPlayer);

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/scoped")).toBe(true));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/scoped");
    expect(JSON.parse(String(saveCall?.[1].body))).toEqual({
      key: "narrator_planner_recent_player_message_window",
      value: 12,
      save_id: "save-1"
    });
  });

  it("posts character action planning concurrency with the active save id", async () => {
    const settingsPayload = modelSettingsPayload({
      visible_sections: ["save"],
      agentic_context_pipeline: { setting_key: "agentic_context_pipeline_enabled", enabled: true },
      character_action_planning_max_concurrency: {
        setting_key: "character_action_planning_max_concurrency",
        value: 20,
        minimum: 1,
        maximum: 20,
        step: 1
      }
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} activeSaveId="save-1" />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "Save" }));
    const concurrency = screen.getByLabelText("Character Action Planning Max Concurrency");
    await userEvent.clear(concurrency);
    await userEvent.type(concurrency, "12");
    fireEvent.blur(concurrency);

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/scoped")).toBe(true));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/scoped");
    expect(JSON.parse(String(saveCall?.[1].body))).toEqual({
      key: "character_action_planning_max_concurrency",
      value: 12,
      save_id: "save-1"
    });
  });

  it("posts save model overrides with the active save id", async () => {
    const chatOptions = [
      modelOption("chat-a", "Chat A", ["chat"]),
      modelOption("chat-b", "Chat B", ["chat"])
    ];
    const settingsPayload = modelSettingsPayload({
      visible_sections: ["save"],
      save_model_override_selectors: [
        modelSelector("chat", chatOptions, "chat-a", {
          inherited_provider: "fake",
          inherited_model_id: "chat-a",
          clearable: false
        })
      ]
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={vi.fn()}
          activeSaveId="save-1"
          currentUser={{ id: "admin-1", username: "admin", role: "admin", status: "active" }}
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "Save" }));
    await userEvent.click(screen.getByRole("button", { name: /model overrides/i }));
    const narratorSelect = screen.getByLabelText("Narrator model");
    const row = narratorSelect.closest(".model-routing-row");
    expect(row).not.toBeNull();
    await userEvent.selectOptions(narratorSelect, "fake\u0000chat-b");
    await userEvent.click(within(row as HTMLElement).getByRole("button", { name: /apply/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/settings/model-preference", expect.anything()));
    const preferenceCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/model-preference");
    expect(JSON.parse(String(preferenceCall?.[1].body))).toEqual({
      task: "chat",
      provider: "fake",
      model_id: "chat-b",
      save_id: "save-1"
    });
  });

  it("dispatches summary backfill from save settings with optional window changes", async () => {
    const settingsPayload = modelSettingsPayload({
      visible_sections: ["save"],
      automatic_summarization: { setting_key: "automatic_summarization_enabled", enabled: true }
    });
    const summaryBackfillJob: Job = {
      id: "job-summary-backfill",
      type: "summary_backfill",
      status: "queued",
      result: null,
      error: null,
      save_id: "save-1"
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/world-data/summary-backfill" ? summaryBackfillJob : settingsPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={runJob}
          activeSaveId="save-1"
          currentUser={{ id: "user-1", username: "Mira", role: "user", status: "active" }}
        />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByRole("tab", { name: "Save" })).toHaveAttribute("aria-selected", "true"));
    await userEvent.click(await screen.findByLabelText("Apply Recommended Chat History Windows"));
    await userEvent.click(screen.getByRole("button", { name: /compact history/i }));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/world-data/summary-backfill", expect.anything()));
    const call = fetchMock.mock.calls.find(([path]) => path === "/api/world-data/summary-backfill");
    expect(JSON.parse(String(call?.[1].body))).toEqual({
      save_id: "save-1",
      apply_recommended_windows: true
    });
    expect(runJob).toHaveBeenCalledWith(summaryBackfillJob);
  });

  it("hides admin settings sections for standard users", async () => {
    const settingsPayload = modelSettingsPayload({
      visible_sections: ["save", "local"],
      pending_jobs_display_mode: { setting_key: "pending_jobs_display_mode", selected: "compact", options: ["compact", "expanded", "expanded_full"] },
      automatic_summarization: { setting_key: "automatic_summarization_enabled", enabled: true },
      image_style_preset: { setting_key: "image_style_preset", selected: "none", options: EXPECTED_IMAGE_STYLE_PRESETS },
      manual_confirmation: {
        memories: { setting_key: "manual_confirmation_memories_enabled", enabled: false },
        character_registry: { setting_key: "manual_confirmation_character_registry_enabled", enabled: false },
        state_changes: { setting_key: "manual_confirmation_state_changes_enabled", enabled: false }
      }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => isAnySettingsReadPath(path) ? settingsPayloadForPath(path, settingsPayload) : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={vi.fn()}
          activeSaveId="save-1"
          currentUser={{ id: "user-1", username: "Mira", role: "user", status: "active" }}
        />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.queryByRole("tab", { name: "Providers" })).not.toBeInTheDocument());
    expect(screen.queryByRole("tab", { name: "Models" })).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Diagnostics" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Review" })).not.toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Save" })).toHaveAttribute("aria-selected", "true");
    expect(await screen.findByText("Summarization")).toBeInTheDocument();
    expect(screen.queryByText("Workbench")).not.toBeInTheDocument();
    expect(screen.queryByPlaceholderText("API key")).not.toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText("Image Style Preset"), "anime");
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/settings/scoped", expect.anything()));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/scoped");
    expect(JSON.parse(String(saveCall?.[1].body))).toEqual({
      key: "image_style_preset",
      value: "anime",
      save_id: "save-1"
    });

    expect(screen.getByText("Manual Confirmation Memories Enabled")).toBeInTheDocument();
  });

  it("shows only scoped local settings for child users", async () => {
    const settingsPayload = modelSettingsPayload({
      visible_sections: ["local"],
      pending_jobs_display_mode: {
        setting_key: "pending_jobs_display_mode",
        selected: "compact",
        options: ["compact", "expanded", "expanded_full"]
      },
      content_rating: {
        setting_key: "content_filter_rating",
        selected: "pg",
        options: ["g", "pg"],
        admin_granted: false
      }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => isAnySettingsReadPath(path) ? settingsPayloadForPath(path, settingsPayload) : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={vi.fn()}
          activeSaveId="save-1"
          currentUser={{ id: "child-1", username: "Ilyra", role: "child", status: "active" }}
        />
      </QueryClientProvider>
    );

    await waitFor(() => {
      expect(screen.getByRole("tab", { name: "Local" })).toHaveAttribute("aria-selected", "true");
    });
    expect(screen.queryByRole("tab", { name: "Providers" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Models" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Diagnostics" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Users" })).not.toBeInTheDocument();

    const pendingJobsMode = await screen.findByLabelText("Pending Jobs Display Mode");
    expect(within(pendingJobsMode).getByRole("option", { name: "Compact" })).toBeInTheDocument();
    await userEvent.selectOptions(pendingJobsMode, "expanded");
    const contentRating = screen.getByLabelText("Content Filtering Level");
    expect(within(contentRating).getByRole("option", { name: "G" })).toBeInTheDocument();
    expect(within(contentRating).getByRole("option", { name: "PG" })).toBeInTheDocument();
    expect(within(contentRating).queryByRole("option", { name: "PG-13" })).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Fade Explicit Content to Black")).not.toBeInTheDocument();

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/settings/scoped", expect.anything()));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/scoped");
    expect(JSON.parse(String(saveCall?.[1].body))).toEqual({
      key: "pending_jobs_display_mode",
      value: "expanded"
    });
  });

  it("shows visible settings summaries and helper text across tabs", async () => {
    const settingsPayload = modelSettingsPayload({
      provider_cards: [
        {
          provider: "fake",
          enabled: true,
          has_api_key: true,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "ready",
          last_error: null
        },
        {
          provider: "openrouter",
          enabled: true,
          has_api_key: false,
          model_count: 0,
          last_model_refresh_at: null,
          refresh_status: "needs_key",
          last_error: null
        }
      ],
      openrouter_routing: openRouterRoutingSettings(),
      task_model_selectors: [
        modelSelector("chat", [modelOption("chat-a", "Chat A", ["chat"])]),
        modelSelector("context_update", [modelOption("struct-a", "Struct A", ["structured_output"])])
      ],
      automatic_summarization: { setting_key: "automatic_summarization_enabled", enabled: true },
      automatic_image_generation: { setting_key: "automatic_image_generation_enabled", enabled: false },
      pending_jobs_display_mode: { setting_key: "pending_jobs_display_mode", selected: "compact", options: ["compact", "expanded"] },
      user_narration_guidance: { setting_key: "user_narration_guidance", value: "" }
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={vi.fn()}
          activeSaveId="save-1"
          currentUser={{ id: "admin-1", username: "Mira", role: "admin", status: "active" }}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByText("Provider setup")).toBeInTheDocument();
    expect(screen.getByText("1 of 2 providers ready")).toBeInTheDocument();
    expect(screen.getByText("1 provider needs an API key")).toBeInTheDocument();
    expect(screen.getByText("Paste provider keys, then refresh model lists after key changes.")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "OpenRouter" }));
    expect(screen.getByText("OpenRouter routing")).toBeInTheDocument();
    expect(screen.getByText("Global profile sorts by Price")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Models" }));
    expect(screen.getByText("Model routing")).toBeInTheDocument();
    expect(screen.getByText("2 task selectors available")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Save" }));
    expect(screen.getByText("Save settings")).toBeInTheDocument();
    expect(screen.getByText("Editing active save controls")).toBeInTheDocument();
    expect(screen.getByText("When enabled, Bragi summarizes older chronicle context as saves grow.")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Local" }));
    expect(screen.getByText("Local preferences")).toBeInTheDocument();
    expect(screen.getByText("Applies to this account and browser.")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Diagnostics" }));
    expect(screen.getByText("Diagnostics health")).toBeInTheDocument();
    expect(screen.getByText("No active diagnostic signals are reporting trouble right now.")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("tab", { name: "Users" }));
    expect(screen.getByText("User management")).toBeInTheDocument();
    expect(screen.getByText("Admin-only controls for local accounts, roles, and password resets.")).toBeInTheDocument();
  });

  it("loads provider settings without requesting the full settings payload", async () => {
    const providerPayload = {
      provider_cards: [
        {
          provider: "fake",
          enabled: true,
          has_api_key: false,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "No API key",
          last_error: null
        }
      ],
      secret_storage_warning: null
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/settings/providers" ? providerPayload : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    expect(await screen.findByLabelText("fake API key")).toBeInTheDocument();
    expect(fetchMock).toHaveBeenCalledWith("/api/settings/providers", expect.anything());
    expect(fetchMock.mock.calls.some(([path]) => isSettingsReadPath(String(path)))).toBe(false);
  });

  it("loads local settings without requesting the full settings payload", async () => {
    const providerPayload = {
      provider_cards: [],
      secret_storage_warning: null
    };
    const localPayload = {
      pending_jobs_display_mode: { setting_key: "pending_jobs_display_mode", selected: "expanded", options: ["compact", "expanded", "expanded_full"] },
      user_narration_guidance: { setting_key: "user_narration_guidance", value: "Keep replies short." },
      debug_logging: { setting_key: "debug_logging_enabled", enabled: false }
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => {
        if (path === "/api/settings/providers") return providerPayload;
        if (path === "/api/settings/local") return localPayload;
        return {};
      }
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={vi.fn()}
          currentUser={{ id: "admin-1", username: "Mira", role: "admin", status: "active" }}
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "Local" }));

    expect(await screen.findByDisplayValue("Keep replies short.")).toBeInTheDocument();
    expect(screen.getByLabelText("Pending Jobs Display Mode")).toHaveValue("expanded");
    expect(fetchMock).toHaveBeenCalledWith("/api/settings/local", expect.anything());
    expect(fetchMock.mock.calls.some(([path]) => isSettingsReadPath(String(path)))).toBe(false);
  });

  it("dispatches provider model refresh jobs and shows provider warnings", async () => {
    const refreshJob: Job = {
      id: "job-model-refresh",
      type: "model_refresh",
      status: "queued",
      result: null,
      error: null,
      created_at: 1
    };
    const settingsPayload = modelSettingsPayload({
      provider_cards: [
        {
          provider: "fake",
          enabled: true,
          has_api_key: true,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "ready",
          last_error: "Previous model refresh failed."
        }
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => path === "/api/settings/model-refresh/fake" ? refreshJob : settingsPayload
    }));
    vi.stubGlobal("fetch", fetchMock);
    const runJob = vi.fn();
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={runJob} />
      </QueryClientProvider>
    );

    expect(await screen.findByText("Previous model refresh failed.")).toBeInTheDocument();
    await userEvent.click(screen.getByTitle("Fetch the current model list for fake."));

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/settings/model-refresh/fake", expect.anything()));
    expect(runJob).toHaveBeenCalledWith(refreshJob);
  });

  it("shows provider key mutation failures and restores busy controls", async () => {
    const settingsPayload = modelSettingsPayload({
      provider_cards: [
        {
          provider: "fake",
          enabled: true,
          has_api_key: true,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "ready",
          last_error: null
        }
      ]
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve(
      path === "/api/settings/provider-key"
        ? { ok: false, status: 500, statusText: "Server Error", json: async () => ({ detail: "Key storage failed." }) }
        : path === "/api/settings/provider-key/fake"
          ? { ok: false, status: 500, statusText: "Server Error", json: async () => ({ detail: "Key removal failed." }) }
          : { ok: true, json: async () => settingsPayload }
    ));
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} activeSaveId="save-1" />
      </QueryClientProvider>
    );

    const keyInput = await screen.findByLabelText("fake API key");
    const saveButton = screen.getByTitle("Store this provider key locally for future requests.");
    await userEvent.type(keyInput, "secret-value");
    await userEvent.click(saveButton);

    expect(await screen.findByText("Key storage failed.")).toBeInTheDocument();
    expect(keyInput).toBeEnabled();
    expect(saveButton).toBeEnabled();

    await userEvent.click(screen.getByRole("button", { name: "Remove fake API key" }));

    expect(await screen.findByText("Key removal failed.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove fake API key" })).toBeEnabled();
  });

  it("disables save settings while saving and surfaces mutation failures", async () => {
    let resolveLocalSave: (response: unknown) => void = () => undefined;
    const localSave = new Promise((resolve) => {
      resolveLocalSave = resolve;
    });
    const settingsPayload = modelSettingsPayload({
      automatic_summarization: { setting_key: "automatic_summarization_enabled", enabled: true },
      image_frequency: { setting_key: "image_generation_frequency", value: 3, minimum: 0, maximum: 999, step: 1 }
    });
    const fetchMock = vi.fn().mockImplementation((path: string) => {
      if (path === "/api/settings/scoped") return localSave;
      return Promise.resolve({ ok: true, json: async () => isAnySettingsReadPath(path) ? settingsPayloadForPath(path, settingsPayload) : {} });
    });
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} activeSaveId="save-1" />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "Save" }));
    await userEvent.click(screen.getByLabelText("Automatic Summarization Enabled"));

    await waitFor(() => expect(screen.getByLabelText("Image Generation Frequency")).toBeDisabled());

    resolveLocalSave({
      ok: false,
      status: 500,
      statusText: "Server Error",
      json: async () => ({ detail: "Could not persist local setting." })
    });

    expect(await screen.findByText("Could not persist local setting.")).toBeInTheDocument();
    expect(screen.getByLabelText("Image Generation Frequency")).toBeEnabled();
  });

  it("keeps save settings visible but disabled when no save is active", async () => {
    const settingsPayload = modelSettingsPayload({
      visible_sections: ["save"],
      automatic_summarization: { setting_key: "automatic_summarization_enabled", enabled: true },
      image_frequency: { setting_key: "image_generation_frequency", value: 3, minimum: 0, maximum: 999, step: 1 }
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={vi.fn()}
          activeSaveId={null}
          currentUser={{ id: "user-1", username: "Mira", role: "user", status: "active" }}
        />
      </QueryClientProvider>
    );

    await waitFor(() => expect(screen.getByRole("tab", { name: "Save" })).toHaveAttribute("aria-selected", "true"));
    expect(await screen.findByText("Load a save to edit save options.")).toBeInTheDocument();
    expect(await screen.findByLabelText("Automatic Summarization Enabled")).toBeDisabled();
    expect(screen.getByLabelText("Image Generation Frequency")).toBeDisabled();
    expect(screen.getByRole("button", { name: /compact history/i })).toBeDisabled();
  });

  it("saves and clears provider API keys from provider settings", async () => {
    const noKeyPayload = modelSettingsPayload({
      provider_cards: [
        {
          provider: "fake",
          enabled: false,
          has_api_key: false,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "No API key",
          last_error: null
        }
      ]
    });
    const fetchMock = settingsFetch(noKeyPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    const keyInput = await screen.findByLabelText("fake API key");
    expect(keyInput).toHaveAttribute("type", "password");
    expect(keyInput).toHaveAttribute("autocomplete", "off");
    expect(keyInput).toHaveAttribute("placeholder", "API key");
    const saveButton = screen.getByTitle("Store this provider key locally for future requests.");
    expect(saveButton).toBeDisabled();

    await userEvent.type(keyInput, "secret-value");
    expect(screen.queryByText("secret-value")).not.toBeInTheDocument();
    expect(saveButton).not.toBeDisabled();
    await userEvent.click(saveButton);

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/settings/provider-key", expect.anything()));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/provider-key");
    expect(JSON.parse(String(saveCall?.[1].body))).toEqual({
      provider: "fake",
      api_key: "secret-value"
    });

    cleanup();
    const savedPayload = modelSettingsPayload({
      provider_cards: [
        {
          provider: "fake",
          enabled: true,
          has_api_key: true,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "Configured; not refreshed",
          last_error: null
        }
      ]
    });
    const clearedPayload = modelSettingsPayload({
      provider_cards: [
        {
          provider: "fake",
          enabled: false,
          has_api_key: false,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "No API key",
          last_error: null
        }
      ]
    });
    const clearFetchMock = settingsFetchSequence([savedPayload, clearedPayload]);
    vi.stubGlobal("fetch", clearFetchMock);

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("button", { name: "Remove fake API key" }));

    await waitFor(() => expect(clearFetchMock).toHaveBeenCalledWith("/api/settings/provider-key/fake", expect.anything()));
    expect(await screen.findByPlaceholderText("API key")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Remove fake API key" })).toBeDisabled();
  });

  it("shows secret storage diagnostics near provider key controls", async () => {
    const warning = "Secret storage health check failed.";
    const settingsPayload = modelSettingsPayload({
      provider_cards: [
        {
          provider: "fake",
          enabled: false,
          has_api_key: false,
          model_count: 2,
          last_model_refresh_at: null,
          refresh_status: "No API key",
          last_error: null
        }
      ],
      secret_storage_warning: warning
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("alert")).toHaveTextContent(warning);
    expect(screen.getByLabelText("fake API key")).toBeInTheDocument();
  });

  it("renders diagnostics as an ops cockpit with health and signal summaries", async () => {
    await renderDiagnosticsSettings(modelSettingsPayload({
      debug_logging: { setting_key: "debug_logging_enabled", enabled: true },
      visible_sections: ["diagnostics"]
    }), diagnosticsPayload({
      signals: [
        { kind: "provider", provider: "fake", error: "authentication failed", retry_summary: "Retries exhausted after 3 attempts" }
      ],
      maintenance_jobs: [
        {
          job_id: "job-1",
          job_type: "state_pruning",
          status: "failed",
          save_id: "save-1",
          error: "provider timed out",
          started_at: "2026-05-27T12:00:00Z",
          completed_at: "2026-05-27T12:01:00Z",
          summary: "1/3 batches, 8 archived, 2 rejected",
          metrics: { completed_batch_count: 1, batch_count: 3 }
        }
      ],
      runtime_performance: {
        job_averages: [
          {
            job_type: "chat_turn",
            step_name: null,
            provider: null,
            model: null,
            task: null,
            success_count: 2,
            failed_count: 1,
            cancelled_count: 0,
            skipped_count: 0,
            average_duration_ms: 1200,
            min_duration_ms: 900,
            max_duration_ms: 1500,
            latest_duration_ms: 999,
            latest_completed_at: "2026-06-01T12:00:00Z"
          }
        ],
        step_averages: [],
        model_averages: []
      },
      web_events: [
        {
          timestamp: "2026-05-25T12:00:00Z",
          level: "error",
          event: "client.api.failed",
          status_code: 500,
          duration_ms: 12,
          error_class: "ApiError",
          route: "/api/chat"
        }
      ]
    }));

    expect(await screen.findByRole("region", { name: "Diagnostics ops cockpit" })).toBeInTheDocument();
    expect(screen.getByText("Attention")).toBeInTheDocument();
    expect(screen.getByText("Signals")).toBeInTheDocument();
    expect(screen.getByText("Failed Jobs")).toBeInTheDocument();
    expect(screen.getByText("Error Events")).toBeInTheDocument();
    expect(screen.getByText("Perf Rows")).toBeInTheDocument();
    expect(screen.queryByText("Debug Logging Enabled")).not.toBeInTheDocument();
    expect(screen.getByText(/authentication failed/)).toBeInTheDocument();
    expect(screen.getByText(/Retries exhausted/)).toBeInTheDocument();
    expect(screen.getByText("client.api.failed")).toBeInTheDocument();
    expect(screen.getByText(/avg 1.2s/)).toBeInTheDocument();
  });

  it("renders a calm empty diagnostics cockpit", async () => {
    await renderDiagnosticsSettings(modelSettingsPayload({
      debug_logging: { setting_key: "debug_logging_enabled", enabled: false },
      visible_sections: ["diagnostics"]
    }), diagnosticsPayload());

    expect(await screen.findByRole("region", { name: "Diagnostics ops cockpit" })).toBeInTheDocument();
    expect(screen.getByText("Healthy")).toBeInTheDocument();
    expect(screen.getByText("No active diagnostic signals")).toBeInTheDocument();
    expect(screen.getByText("No runtime performance data")).toBeInTheDocument();
    expect(screen.getByText("No recent web events")).toBeInTheDocument();
  });

  it("renders active save and scheduler health with support bundle copying", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    vi.stubGlobal("navigator", {
      ...window.navigator,
      clipboard: { writeText }
    });
    const fetchMock = await renderDiagnosticsSettings(
      modelSettingsPayload({ visible_sections: ["diagnostics"] }),
      diagnosticsPayload({
        filters: { save_id: "save-1", categories: [], limit: 50, since: null },
        signals: [
          { kind: "provider", provider: "fake", error: "authentication failed" }
        ],
        scheduler_health: {
          summary: { total: 1, healthy: 0, overdue: 0, leased: 0, failed: 1, disabled: 0 },
          tasks: [
            {
              task_id: "task-1",
              task_type: "world_suggestion_review",
              save_id: "save-1",
              status: "failed",
              enabled: true,
              interval_seconds: 60,
              next_run_at: "2026-07-08T12:05:00Z",
              lease_until: null,
              last_started_at: "2026-07-08T12:00:00Z",
              last_completed_at: "2026-07-08T12:01:00Z",
              last_job_id: "job-1",
              failure_count: 2,
              error: "review failed",
              skip_reason: null
            }
          ]
        },
        active_save_health: {
          save_id: "save-1",
          active_message_count: 42,
          recent_player_message_window: 8,
          recent_narrator_message_window: 8,
          narrator_planner_recent_player_message_window: 4,
          narrator_planner_recent_narrator_message_window: 4,
          pending_suggestion_count: 3,
          stale_pending_suggestion_count: 1,
          summary_count: 2,
          recent_failed_continuity_job_count: 1,
          recent_failed_continuity_jobs_by_type: { context_update: 1 },
          latest_context_search: null,
          latest_chat_prompt: null,
          warnings: [
            {
              code: "stale_pending_suggestions",
              severity: "warning",
              message: "1 stale suggestion needs review.",
              count: 1
            }
          ]
        }
      }),
      "save-1"
    );

    await screen.findByRole("region", { name: "Diagnostics ops cockpit" });

    expect(fetchMock).toHaveBeenCalledWith("/api/diagnostics?save_id=save-1", expect.anything());
    expect(screen.getByText("Active Save Health")).toBeInTheDocument();
    expect(await screen.findByText("Stale Pending Suggestions")).toBeInTheDocument();
    expect(screen.getByText("Scheduler Health")).toBeInTheDocument();
    expect(screen.getByText("World Suggestion Review")).toBeInTheDocument();
    expect(screen.getByText(/review failed/)).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Copy support bundle" }));

    expect(writeText).toHaveBeenCalledTimes(1);
    expect(writeText.mock.calls[0][0]).toContain("\"active_save_health\"");
    expect(writeText.mock.calls[0][0]).toContain("\"scheduler_health\"");
    expect(writeText.mock.calls[0][0]).not.toContain("secret");
  });

  it("keeps numeric settings inputs synced with refetched server values", async () => {
    const localSettingsPayload = (imageFrequency: number, maxOutputTokens: number) => modelSettingsPayload({
      automatic_image_generation: { setting_key: "automatic_image_generation_enabled", enabled: true },
      automatic_media_mode: { setting_key: "automatic_media_mode", selected: "image", options: ["image"] },
      image_frequency: { setting_key: "image_generation_frequency", value: imageFrequency, minimum: 0, maximum: 999, step: 1 },
      chat_max_output_tokens: {
        setting_key: "chat_max_output_tokens",
        enabled_setting_key: "chat_max_output_tokens_enabled",
        enabled: true,
        supported: true,
        value: maxOutputTokens,
        minimum: 64,
        maximum: 4096,
        step: 64
      }
    });
    const fetchMock = settingsFetchSequence([
      localSettingsPayload(3, 1200),
      localSettingsPayload(4, 1200),
      localSettingsPayload(4, 2048)
    ]);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={vi.fn()}
          activeSaveId="save-1"
          currentUser={{ id: "user-1", username: "Mira", role: "user", status: "active" }}
        />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "Save" }));
    const imageFrequency = await screen.findByLabelText("Image Generation Frequency");
    expect(imageFrequency).toHaveValue(3);

    await userEvent.clear(imageFrequency);
    await userEvent.type(imageFrequency, "9");
    fireEvent.blur(imageFrequency);

    await waitFor(() => expect(imageFrequency).toHaveValue(4));

    const maxTokens = screen.getByLabelText("Chat Max Output Tokens");
    expect(maxTokens).toHaveValue(1200);

    await userEvent.clear(maxTokens);
    await userEvent.type(maxTokens, "3000");
    fireEvent.blur(maxTokens);

    await waitFor(() => expect(maxTokens).toHaveValue(2048));
    expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .map(([, init]) => JSON.parse(String(init.body)))).toEqual([
        { key: "image_generation_frequency", value: 9, save_id: "save-1" },
        { key: "chat_max_output_tokens", value: 3000, save_id: "save-1" }
      ]);
  });

  it("renders provider generation settings with metadata support gates", async () => {
    const settingsPayload = {
      provider_cards: [],
      task_model_selectors: [],
      roleplay_shared_models: { setting_key: "roleplay_shared_models", enabled: true },
      roleplay_model_groups: [],
      automatic_summarization: { setting_key: "automatic_summarization_enabled", enabled: true },
      automatic_image_generation: { setting_key: "automatic_image_generation_enabled", enabled: true },
      automatic_media_mode: { setting_key: "automatic_media_mode", selected: "image", options: ["image"] },
      image_frequency: { setting_key: "image_generation_frequency", value: 3, minimum: 0, maximum: 999, step: 1 },
      chat_temperature: {
        setting_key: "chat_temperature",
        enabled_setting_key: "chat_temperature_enabled",
        enabled: false,
        supported: false,
        value: 0.7,
        minimum: 0,
        maximum: 2,
        step: 0.05
      },
      chat_max_output_tokens: {
        setting_key: "chat_max_output_tokens",
        enabled_setting_key: "chat_max_output_tokens_enabled",
        enabled: false,
        supported: true,
        value: 1200,
        minimum: 64,
        maximum: 4096,
        step: 64
      },
      image_dimension_preset: {
        setting_key: "image_dimension_preset",
        selected: "provider_default",
        options: ["provider_default", "wide_1024x576"],
        supported: true
      },
      manual_confirmation: {
        memories: { setting_key: "manual_confirmation_memories", enabled: false },
        character_registry: { setting_key: "manual_confirmation_character_registry", enabled: false },
        state_changes: { setting_key: "manual_confirmation_state_changes", enabled: false }
      }
    };
    const fetchMock = vi.fn().mockImplementation((path: string) => Promise.resolve({
      ok: true,
      json: async () => isAnySettingsReadPath(path) ? settingsPayloadForPath(path, settingsPayload) : {}
    }));
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} activeSaveId="save-1" />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "Save" }));

    expect(screen.getByText("Provider Generation")).toBeInTheDocument();
    expect(screen.getByText("Chat Temperature Enabled").closest("label")?.querySelector("input")).toBeDisabled();
    const maxTokensToggle = screen.getByText("Chat Max Output Tokens Enabled").closest("label")?.querySelector("input");
    expect(maxTokensToggle).not.toBeNull();
    await userEvent.click(maxTokensToggle as HTMLInputElement);
    await userEvent.selectOptions(screen.getByLabelText("Image Dimension Preset"), "wide_1024x576");

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/scoped")).toHaveLength(2));
    expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/scoped")
      .map(([, init]) => JSON.parse(String(init.body)))).toEqual([
        { key: "chat_max_output_tokens_enabled", value: true, save_id: "save-1" },
        { key: "image_dimension_preset", value: "wide_1024x576", save_id: "save-1" }
      ]);
  });

  it("keeps advanced OpenRouter controls collapsed until opened", async () => {
    const settingsPayload = modelSettingsPayload({
      openrouter_routing: openRouterRoutingSettings()
    });
    vi.stubGlobal("fetch", settingsFetch(settingsPayload));
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "OpenRouter" }));

    expect(screen.getByLabelText("Sort")).toBeInTheDocument();
    const advancedToggle = screen.getByRole("button", { name: /advanced openrouter routing/i });
    expect(advancedToggle).toHaveAttribute("aria-expanded", "false");
    expect(screen.queryByText("Provider Object")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Only provider search")).not.toBeInTheDocument();

    await userEvent.click(advancedToggle);

    expect(advancedToggle).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("region", { name: /advanced openrouter routing/i })).toBeInTheDocument();
    expect(screen.getByText("Provider Object")).toBeInTheDocument();
    expect(screen.getByLabelText("Only provider search")).toBeInTheDocument();
  });

  it("surfaces typed OpenRouter routing settings and saves profile changes", async () => {
    const settingsPayload = modelSettingsPayload({
      openrouter_routing: openRouterRoutingSettings()
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "OpenRouter" }));

    await expandOpenRouterAdvanced();
    expect(screen.getByText("Provider Object")).toBeInTheDocument();
    expect(screen.getByText(/"sort": "price"/)).toBeInTheDocument();

    await userEvent.selectOptions(screen.getByLabelText("Sort"), "throughput");

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/scoped")).toBe(true));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/scoped");
    const body = JSON.parse(String(saveCall?.[1]?.body));
    expect(body.key).toBe("openrouter_routing_profiles");
    expect(body.value.global.sort).toBe("throughput");
    expect(body.value.global.sort_partition).toBe("model");
    expect(body.value.task_overrides.narrator.enabled).toBe(false);
  });

  it("adds OpenRouter provider catalog suggestions to provider filters", async () => {
    const settingsPayload = modelSettingsPayload({
      openrouter_routing: openRouterRoutingSettings({
        provider_catalog: [
          {
            slug: "openai",
            name: "OpenAI",
            privacy_policy_url: "https://openai.com/privacy",
            terms_of_service_url: "https://openai.com/terms",
            status_page_url: "https://status.openai.com",
            headquarters: "US",
            datacenters: ["US", "IE"]
          },
          {
            slug: "deepinfra",
            name: "DeepInfra",
            privacy_policy_url: null,
            terms_of_service_url: null,
            status_page_url: null,
            headquarters: "US",
            datacenters: ["US"]
          }
        ],
        provider_catalog_refreshed_at: "2026-06-01T12:00:00+00:00"
      })
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "OpenRouter" }));
    await expandOpenRouterAdvanced();
    await userEvent.type(screen.getByLabelText("Only provider search"), "open");
    await userEvent.click(screen.getByRole("button", { name: "Add OpenAI to Only" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/scoped")).toBe(true));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/scoped");
    const body = JSON.parse(String(saveCall?.[1]?.body));
    expect(body.value.global.only).toEqual(["openai"]);
  });

  it("accepts custom OpenRouter provider endpoint slugs", async () => {
    const settingsPayload = modelSettingsPayload({
      openrouter_routing: openRouterRoutingSettings()
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "OpenRouter" }));
    await expandOpenRouterAdvanced();
    await userEvent.type(screen.getByLabelText("Ignore provider search"), "deepinfra/turbo");
    await userEvent.click(screen.getByRole("button", { name: "Add custom provider to Ignore" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/scoped")).toBe(true));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/scoped");
    const body = JSON.parse(String(saveCall?.[1]?.body));
    expect(body.value.global.ignore).toEqual(["deepinfra/turbo"]);
  });

  it("renders unknown saved OpenRouter provider slugs as custom chips", async () => {
    const settingsPayload = modelSettingsPayload({
      openrouter_routing: openRouterRoutingSettings({
        global_profile: openRouterRoutingProfile({
          only: ["deepinfra/turbo"]
        }),
        provider_catalog: [
          {
            slug: "openai",
            name: "OpenAI",
            privacy_policy_url: null,
            terms_of_service_url: null,
            status_page_url: null,
            headquarters: null,
            datacenters: []
          }
        ]
      })
    });
    vi.stubGlobal("fetch", settingsFetch(settingsPayload));
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "OpenRouter" }));
    await expandOpenRouterAdvanced();

    expect(screen.getByText("deepinfra/turbo")).toBeInTheDocument();
    const customChip = screen.getByRole("button", { name: "Remove deepinfra/turbo from Only" }).closest(".openrouter-provider-chip");
    expect(customChip).not.toBeNull();
    expect(within(customChip as HTMLElement).getByText("Custom slug")).toBeInTheDocument();
  });

  it("reorders OpenRouter ordered providers", async () => {
    const settingsPayload = modelSettingsPayload({
      openrouter_routing: openRouterRoutingSettings({
        global_profile: openRouterRoutingProfile({
          order: ["openai", "deepinfra"]
        }),
        provider_catalog: [
          {
            slug: "openai",
            name: "OpenAI",
            privacy_policy_url: null,
            terms_of_service_url: null,
            status_page_url: null,
            headquarters: null,
            datacenters: []
          },
          {
            slug: "deepinfra",
            name: "DeepInfra",
            privacy_policy_url: null,
            terms_of_service_url: null,
            status_page_url: null,
            headquarters: null,
            datacenters: []
          }
        ]
      })
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "OpenRouter" }));
    await expandOpenRouterAdvanced();
    await userEvent.click(screen.getByRole("button", { name: "Move DeepInfra up in Order" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/scoped")).toBe(true));
    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/scoped");
    const body = JSON.parse(String(saveCall?.[1]?.body));
    expect(body.value.global.order).toEqual(["deepinfra", "openai"]);
  });

  it("disables OpenRouter provider pickers for inactive task overrides", async () => {
    const settingsPayload = modelSettingsPayload({
      openrouter_routing: openRouterRoutingSettings()
    });
    vi.stubGlobal("fetch", settingsFetch(settingsPayload));
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "OpenRouter" }));
    await userEvent.selectOptions(screen.getByLabelText("Profile"), "narrator");
    await expandOpenRouterAdvanced();

    expect(screen.getByLabelText("Only provider search")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Add custom provider to Only" })).toBeDisabled();
  });

  it("explains OpenRouter routing controls with tooltips", async () => {
    const settingsPayload = modelSettingsPayload({
      openrouter_routing: openRouterRoutingSettings()
    });
    const fetchMock = settingsFetch(settingsPayload);
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel runJob={vi.fn()} />
      </QueryClientProvider>
    );

    await userEvent.click(await screen.findByRole("tab", { name: "OpenRouter" }));
    await expandOpenRouterAdvanced();

    expect(screen.getByLabelText("Profile")).toHaveAttribute("title", expect.stringContaining("apply globally"));
    expect(screen.getByLabelText("Sort")).toHaveAttribute("title", expect.stringContaining("prioritizes providers"));
    expect(screen.getByLabelText("Fallbacks")).toHaveAttribute("title", expect.stringContaining("backup providers"));
    expect(screen.getByText("Require Parameters").closest("label")).toHaveAttribute("title", expect.stringContaining("support every parameter"));
    expect(screen.getByLabelText("Order provider search")).toHaveAttribute("title", expect.stringContaining("Provider slugs"));
    expect(screen.getByLabelText("Data Collection")).toHaveAttribute("title", expect.stringContaining("store request data"));
    expect(screen.getByText("Zero Data Retention").closest("label")).toHaveAttribute("title", expect.stringContaining("zero data retention"));
    expect(screen.getByText("int4").closest("label")).toHaveAttribute("title", expect.stringContaining("Integer 4-bit"));
    expect(screen.getByLabelText("Min Throughput P50")).toHaveAttribute("title", expect.stringContaining("throughput"));
    expect(screen.getByLabelText("Max Price Prompt")).toHaveAttribute("title", expect.stringContaining("prompt-token price"));
    expect(screen.getByText("Provider Object")).toHaveAttribute("title", expect.stringContaining("provider object"));
  });

  it("shows retry summaries in diagnostics", async () => {
    const { DiagnosticsList } = await import("./main");

    render(
      <DiagnosticsList
        diagnostics={[
          {
            kind: "job",
            error: "provider failed",
            job_type: "post_turn_jobs",
            save_id: "save-1",
            retry_summary: "Provider retry attempts were exhausted after 3 attempts"
          }
        ]}
      />
    );

    expect(screen.getByText("post_turn_jobs")).toBeInTheDocument();
    expect(screen.getByText(/Provider retry attempts were exhausted/)).toBeInTheDocument();
  });

  it("renders role-based model routing lanes instead of broad task buckets", async () => {
    const chatOptions = [modelOption("chat-a", "Chat A", ["chat"])];
    const structuredOptions = [modelOption("struct-a", "Struct A", ["structured_output"])];
    const toolOptions = [modelOption("tool-a", "Tool A", ["tool_calling"])];
    const imageOptions = [modelOption("image-a", "Image A", ["image_generation"])];
    const editOptions = [modelOption("edit-a", "Edit A", ["image_to_image"])];
    const videoOptions = [modelOption("video-a", "Video A", ["text_to_video"])];
    const animationOptions = [modelOption("animation-a", "Animation A", ["image_plus_text_to_video"])];
    const visionOptions = [modelOption("vision-a", "Vision A", ["vision"])];

    await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("chat", chatOptions),
        modelSelector("response_planning", structuredOptions),
        modelSelector("response_verification", structuredOptions),
        modelSelector("npc_knowledge_audit", structuredOptions),
        modelSelector("action_choice_generation", structuredOptions),
        modelSelector("character_presence_assessment", structuredOptions),
        modelSelector("character_intent_planning", structuredOptions),
        modelSelector("character_action_planning", structuredOptions),
        modelSelector("director_pressure", structuredOptions),
        modelSelector("context_search", toolOptions),
        modelSelector("fact_observation", structuredOptions),
        modelSelector("memory_curation", structuredOptions),
        modelSelector("state_memory", structuredOptions),
        modelSelector("context_update", toolOptions),
        modelSelector("character_enhancement", toolOptions),
        modelSelector("character_registry_maintenance", structuredOptions),
        modelSelector("context_cleanup_scan", structuredOptions),
        modelSelector("context_cleanup_actions", structuredOptions),
        modelSelector("guided_context_cleanup", structuredOptions),
        modelSelector("state_pruning", structuredOptions),
        modelSelector("scenario_generation", chatOptions),
        modelSelector("context_cleanup", structuredOptions),
        modelSelector("scenario_evolution", structuredOptions),
        modelSelector("summarization", chatOptions),
        modelSelector("image_prompt", chatOptions),
        modelSelector("character_image_description", visionOptions),
        modelSelector("image_generation", imageOptions),
        modelSelector("image_to_image_generation", editOptions),
        modelSelector("scene_image_edit_generation", editOptions),
        modelSelector("character_image_edit_generation", editOptions),
        modelSelector("text_message_image_edit_generation", editOptions),
        modelSelector("video_generation", videoOptions),
        modelSelector("image_animation", animationOptions)
      ],
      scenario_section_model_selectors: [
        modelSelector("scenario_generation_section_opening_message", chatOptions, "chat-a", { label: "Opening Message", section_id: "opening_message" })
      ]
    }));

    expect(screen.getByText("Routing Lanes")).toBeInTheDocument();
    for (const group of ["Narration", "Cast & Direction", "Context & World", "Authoring & Media"]) {
      expect(screen.getByRole("heading", { name: group })).toBeInTheDocument();
    }
    for (const label of [
      "Narrator",
      "Narrator Planner",
      "Narrator Verifier",
      "Character Agents",
      "Action Choices",
      "Director Pressure",
      "Context Selector",
      "Observation & Memory",
      "World Updates",
      "Character Registry Maintenance",
      "Context Cleanup",
      "State Pruning",
      "Scenario Evolution",
      "Scenario Writer",
      "Summarization",
      "Image Prompt",
      "Image Details",
      "Image Generation",
      "Image Edit",
      "Video Generation",
      "Image Animation"
    ]) {
      expect(screen.getByText(label)).toBeInTheDocument();
    }
    expect(screen.queryByText("Background Text")).not.toBeInTheDocument();
    expect(screen.queryByText("Context Work")).not.toBeInTheDocument();
    expect(screen.queryByText("Maintenance")).not.toBeInTheDocument();
    expect(screen.queryByText("Summary & Image Prompts")).not.toBeInTheDocument();
    expect(screen.queryByText("Pure Text Models")).not.toBeInTheDocument();
    expect(screen.queryByText("Structured Output Models")).not.toBeInTheDocument();
    expect(screen.queryByText("Vision Models")).not.toBeInTheDocument();
  });

  it("saves loads and deletes model routing profiles", async () => {
    const fetchMock = await renderModelSettings(modelSettingsPayload({
      model_routing_profiles: modelRoutingProfilesSettings({
        last_loaded_profile_id: "fast",
        profiles: [
          {
            id: "fast",
            name: "Fast",
            roleplay_shared_models_enabled: true,
            preference_count: 2,
            preferences: [
              { task: "chat", provider: "fake", model_id: "chat-a" },
              { task: "context_update", provider: "fake", model_id: "struct-a" }
            ]
          }
        ]
      })
    }));

    expect(screen.getByText("Model Profiles")).toBeInTheDocument();
    await userEvent.clear(screen.getByLabelText("Profile name"));
    await userEvent.type(screen.getByLabelText("Profile name"), "Balanced");
    await userEvent.click(screen.getByRole("button", { name: "Save new profile" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/model-routing-profiles")).toBe(true));
    const loadButton = screen.getByRole("button", { name: "Load profile" });
    await waitFor(() => expect(loadButton).not.toBeDisabled());
    await userEvent.click(loadButton);
    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/model-routing-profiles/fast/apply")).toBe(true));
    const deleteButton = screen.getByRole("button", { name: "Delete profile" });
    await waitFor(() => expect(deleteButton).not.toBeDisabled());
    await userEvent.click(deleteButton);

    const saveCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/model-routing-profiles");
    expect(JSON.parse(String(saveCall?.[1]?.body))).toEqual({ name: "Balanced" });
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/model-routing-profiles/fast/apply")).toBe(true);
    expect(fetchMock.mock.calls.some(([path, init]) => (
      path === "/api/settings/model-routing-profiles/fast" &&
      init?.method === "DELETE"
    ))).toBe(true);
  });

  it("applies the narrator lane to shared and roleplay narrator selectors", async () => {
    const chatOptions = [
      modelOption("chat-a", "Chat A", ["chat"]),
      modelOption("chat-b", "Chat B", ["chat"])
    ];
    const fetchMock = await renderModelSettings(modelSettingsPayload({
      roleplay_shared_models: { setting_key: "roleplay_shared_models", enabled: false },
      roleplay_model_groups: [
        {
          roleplay_type: "shared_roleplay",
          label: "Shared Roleplay",
          selectors: [
            modelSelector("chat", chatOptions),
            modelSelector("chat_full_roleplay", chatOptions),
            modelSelector("chat_fantasy_roleplay", chatOptions),
            modelSelector("chat_science_fiction_roleplay", chatOptions),
            modelSelector("chat_first_contact_exploration", chatOptions),
            modelSelector("chat_survival_expedition", chatOptions),
            modelSelector("chat_time_loop", chatOptions),
            modelSelector("chat_investigation_mystery", chatOptions),
            modelSelector("chat_heist_infiltration", chatOptions),
            modelSelector("chat_political_intrigue", chatOptions),
            modelSelector("chat_character_interaction", chatOptions),
            modelSelector("chat_dating_sim", chatOptions)
          ]
        }
      ]
    }));

    const select = screen.getByLabelText("Narrator model");
    const row = select.closest(".model-routing-row");
    expect(row).not.toBeNull();
    await userEvent.selectOptions(select, "fake\u0000chat-b");
    await userEvent.click(within(row as HTMLElement).getByRole("button", { name: /apply/i }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/model-preference")).toHaveLength(11));
    expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/model-preference")
      .map(([, init]) => JSON.parse(String(init.body)))).toEqual([
        { task: "chat", provider: "fake", model_id: "chat-b" },
        { task: "chat_full_roleplay", provider: "fake", model_id: "chat-b" },
        { task: "chat_fantasy_roleplay", provider: "fake", model_id: "chat-b" },
        { task: "chat_science_fiction_roleplay", provider: "fake", model_id: "chat-b" },
        { task: "chat_first_contact_exploration", provider: "fake", model_id: "chat-b" },
        { task: "chat_survival_expedition", provider: "fake", model_id: "chat-b" },
        { task: "chat_time_loop", provider: "fake", model_id: "chat-b" },
        { task: "chat_investigation_mystery", provider: "fake", model_id: "chat-b" },
        { task: "chat_heist_infiltration", provider: "fake", model_id: "chat-b" },
        { task: "chat_political_intrigue", provider: "fake", model_id: "chat-b" },
        { task: "chat_dating_sim", provider: "fake", model_id: "chat-b" }
      ]);
  });

  it("applies the narrator lane thinking level to lane selectors", async () => {
    const chatOptions = [
      modelOption("chat-a", "Chat A", ["chat"], "fake", null, modelThinkingSupport()),
      modelOption("chat-b", "Chat B", ["chat"], "fake", null, modelThinkingSupport())
    ];
    const fetchMock = await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("chat", chatOptions, "chat-a", {
          thinking: thinkingControl("chat", "provider_default", true, "chat-a")
        }),
        modelSelector("chat_full_roleplay", chatOptions, "chat-a", {
          thinking: thinkingControl("chat_full_roleplay", "provider_default", true, "chat-a")
        })
      ]
    }));

    const select = screen.getByLabelText("Narrator model");
    const row = select.closest(".model-routing-row");
    expect(row).not.toBeNull();
    await userEvent.selectOptions(select, "fake\u0000chat-b");
    await userEvent.selectOptions(screen.getByLabelText("Narrator thinking level"), "high");
    await userEvent.click(within(row as HTMLElement).getByRole("button", { name: /apply/i }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/model-thinking")).toHaveLength(2));
    expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/model-thinking")
      .map(([, init]) => JSON.parse(String(init.body)))).toEqual([
        { task: "chat", provider: "fake", model_id: "chat-b", level: "high" },
        { task: "chat_full_roleplay", provider: "fake", model_id: "chat-b", level: "high" }
      ]);
  });

  it("applies the scenario writer lane to scenario and section selectors", async () => {
    const textOptions = [
      modelOption("text-a", "Text A", ["chat"]),
      modelOption("text-b", "Text B", ["chat"])
    ];
    const fetchMock = await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("scenario_generation", textOptions)
      ],
      scenario_section_model_selectors: [
        modelSelector("scenario_generation_section_opening_message", textOptions, "text-a", { label: "Opening Message", section_id: "opening_message" }),
        modelSelector("scenario_generation_section_premise", textOptions, "text-a", { label: "Premise", section_id: "premise" })
      ]
    }));

    const select = screen.getByLabelText("Scenario Writer model");
    const row = select.closest(".model-routing-row");
    expect(row).not.toBeNull();
    await userEvent.selectOptions(select, "fake\u0000text-b");
    await userEvent.click(within(row as HTMLElement).getByRole("button", { name: /apply/i }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/model-preference")).toHaveLength(3));
    expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/model-preference")
      .map(([, init]) => JSON.parse(String(init.body)))).toEqual([
        { task: "scenario_generation", provider: "fake", model_id: "text-b" },
        { task: "scenario_generation_section_opening_message", provider: "fake", model_id: "text-b" },
        { task: "scenario_generation_section_premise", provider: "fake", model_id: "text-b" }
      ]);
  });

  it("applies summarization and image prompt lanes independently", async () => {
    const textOptions = [
      modelOption("text-a", "Text A", ["chat"]),
      modelOption("text-b", "Text B", ["chat"])
    ];
    const fetchMock = await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("summarization", textOptions),
        modelSelector("image_prompt", textOptions)
      ]
    }));

    const summarySelect = screen.getByLabelText("Summarization model");
    const summaryRow = summarySelect.closest(".model-routing-row");
    expect(summaryRow).not.toBeNull();
    await userEvent.selectOptions(summarySelect, "fake\u0000text-b");
    await userEvent.click(within(summaryRow as HTMLElement).getByRole("button", { name: /apply/i }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/model-preference")).toHaveLength(1));
    expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/model-preference")
      .map(([, init]) => JSON.parse(String(init.body)))).toEqual([
        { task: "summarization", provider: "fake", model_id: "text-b" }
      ]);

    const imagePromptSelect = screen.getByLabelText("Image Prompt model");
    const imagePromptRow = imagePromptSelect.closest(".model-routing-row");
    expect(imagePromptRow).not.toBeNull();
    await userEvent.selectOptions(imagePromptSelect, "fake\u0000text-b");
    await userEvent.click(within(imagePromptRow as HTMLElement).getByRole("button", { name: /apply/i }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/model-preference")).toHaveLength(2));
    expect(JSON.parse(String(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/model-preference")[1][1].body))).toEqual(
      { task: "image_prompt", provider: "fake", model_id: "text-b" }
    );
  });

  it("lets the world updates lane use structured-output or tool-call models", async () => {
    const options = [
      modelOption("structured-only", "Structured Only", ["structured_output"]),
      modelOption("tool-model", "Tool Model", ["tool_calling"]),
      modelOption("function-model", "Function Model", ["function_calling"])
    ];
    await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("state_memory", options, "tool-model"),
        modelSelector("context_update", options, "tool-model"),
        modelSelector("character_enhancement", options, "tool-model")
      ]
    }));

    const select = screen.getByLabelText("World Updates model");
    expect(within(select).getByRole("option", { name: "Structured Only - fake/structured-only" })).toBeInTheDocument();
    expect(within(select).getByRole("option", { name: "Tool Model - fake/tool-model" })).toBeInTheDocument();
    expect(within(select).getByRole("option", { name: "Function Model - fake/function-model" })).toBeInTheDocument();
  });

  it("applies planner, character agent, director, and context lanes separately", async () => {
    const options = [
      modelOption("structured-a", "Structured A", ["structured_output"]),
      modelOption("structured-b", "Structured B", ["structured_output"])
    ];
    const sharedTasks = [
      "context_search",
      "state_memory",
      "context_update",
      "character_enhancement",
      "fact_observation",
      "memory_curation",
      "response_planning",
      "response_verification",
      "director_pressure",
      "action_choice_generation",
      "character_presence_assessment",
      "character_intent_planning",
      "character_action_planning",
      "npc_knowledge_audit"
    ];
    const fetchMock = await renderModelSettings(modelSettingsPayload({
      task_model_selectors: sharedTasks.map((task) => modelSelector(task, options, "structured-a"))
    }));

    const plannerSelect = screen.getByLabelText("Narrator Planner model");
    const plannerRow = plannerSelect.closest(".model-routing-row");
    expect(plannerRow).not.toBeNull();
    expect(within(plannerRow as HTMLElement).getByText("1 task")).toBeInTheDocument();
    await userEvent.selectOptions(plannerSelect, "fake\u0000structured-b");
    await userEvent.click(within(plannerRow as HTMLElement).getByRole("button", { name: /apply/i }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/model-preference")).toHaveLength(1));
    expect(JSON.parse(String(fetchMock.mock.calls.find(([path]) => path === "/api/settings/model-preference")?.[1].body))).toEqual({
      task: "response_planning",
      provider: "fake",
      model_id: "structured-b"
    });

    const characterSelect = screen.getByLabelText("Character Agents model");
    const characterRow = characterSelect.closest(".model-routing-row");
    expect(characterRow).not.toBeNull();
    await userEvent.selectOptions(characterSelect, "fake\u0000structured-b");
    await userEvent.click(within(characterRow as HTMLElement).getByRole("button", { name: /apply/i }));

    const actionChoicesSelect = screen.getByLabelText("Action Choices model");
    const actionChoicesRow = actionChoicesSelect.closest(".model-routing-row");
    expect(actionChoicesRow).not.toBeNull();
    await userEvent.selectOptions(actionChoicesSelect, "fake\u0000structured-b");
    await userEvent.click(within(actionChoicesRow as HTMLElement).getByRole("button", { name: /apply/i }));

    const directorSelect = screen.getByLabelText("Director Pressure model");
    const directorRow = directorSelect.closest(".model-routing-row");
    expect(directorRow).not.toBeNull();
    await userEvent.selectOptions(directorSelect, "fake\u0000structured-b");
    await userEvent.click(within(directorRow as HTMLElement).getByRole("button", { name: /apply/i }));

    const contextSelect = screen.getByLabelText("Context Selector model");
    const contextRow = contextSelect.closest(".model-routing-row");
    expect(contextRow).not.toBeNull();
    await userEvent.selectOptions(contextSelect, "fake\u0000structured-b");
    await userEvent.click(within(contextRow as HTMLElement).getByRole("button", { name: /apply/i }));

    await waitFor(() => expect(fetchMock.mock.calls.filter(([path]) => path === "/api/settings/model-preference")).toHaveLength(7));
    expect(fetchMock.mock.calls
      .filter(([path]) => path === "/api/settings/model-preference")
      .map(([, init]) => JSON.parse(String(init.body)))).toEqual([
        { task: "response_planning", provider: "fake", model_id: "structured-b" },
        { task: "character_presence_assessment", provider: "fake", model_id: "structured-b" },
        { task: "character_intent_planning", provider: "fake", model_id: "structured-b" },
        { task: "character_action_planning", provider: "fake", model_id: "structured-b" },
        { task: "action_choice_generation", provider: "fake", model_id: "structured-b" },
        { task: "director_pressure", provider: "fake", model_id: "structured-b" },
        { task: "context_search", provider: "fake", model_id: "structured-b" }
      ]);
  });

  it("filters planner and director lanes to structured-output models", async () => {
    const options = [
      modelOption("structured-model", "Structured Model", ["structured_output"]),
      modelOption("tool-only", "Tool Only", ["tool_calling"])
    ];
    await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("response_planning", options, "structured-model"),
        modelSelector("director_pressure", options, "structured-model")
      ]
    }));

    for (const label of ["Narrator Planner model", "Director Pressure model"]) {
      const select = screen.getByLabelText(label);
      expect(within(select).getByRole("option", { name: "Structured Model - fake/structured-model" })).toBeInTheDocument();
      expect(within(select).queryByRole("option", { name: "Tool Only - fake/tool-only" })).not.toBeInTheDocument();
    }
  });

  it("lets the context cleanup lane use structured-output or tool-call models", async () => {
    const options = [
      modelOption("structured-model", "Structured Model", ["json_schema"]),
      modelOption("tool-only", "Tool Only", ["tools"])
    ];
    await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("context_cleanup_scan", options, "structured-model"),
        modelSelector("context_cleanup_actions", options, "structured-model")
      ]
    }));

    const select = screen.getByLabelText("Context Cleanup model");
    expect(within(select).getByRole("option", { name: "Structured Model - fake/structured-model" })).toBeInTheDocument();
    expect(within(select).getByRole("option", { name: "Tool Only - fake/tool-only" })).toBeInTheDocument();
  });

  it("shows only supported fallback model selectors in the fallback area", async () => {
    await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("narrator_fallback", [modelOption("narrator-fallback", "Narrator Fallback", ["chat"])]),
        modelSelector("chat_fallback", [modelOption("chat-fallback", "Chat Fallback", ["chat"])]),
        modelSelector("structured_output_fallback", [modelOption("structured-fallback", "Structured Fallback", ["structured_output"])]),
        modelSelector("tool_call_fallback", [modelOption("tool-fallback", "Tool Fallback", ["function_calling"])]),
        modelSelector("image_fallback", [modelOption("image-fallback", "Image Fallback", ["image_generation"])]),
        modelSelector("image_edit_fallback", [modelOption("image-edit-fallback", "Image Edit Fallback", ["image_to_image"])]),
        modelSelector("video_fallback", [modelOption("video-fallback", "Video Fallback", ["text_to_video"])])
      ]
    }));

    const fallbackSection = screen.getByText("Fallback Models").closest("section");
    expect(fallbackSection).not.toBeNull();
    for (const label of ["Narrator Fallback", "Background Text Fallback", "Structured Fallback", "Tool Fallback", "Image Fallback", "Image Edit Fallback", "Video Fallback"]) {
      expect(within(fallbackSection as HTMLElement).getByText(label)).toBeInTheDocument();
    }
  });

  it("keeps routing lane task controls collapsed until opened", async () => {
    const structuredOptions = [modelOption("fake-structured", "Fake Structured", ["structured_output"])];
    const editOptions = [modelOption("fake-edit", "Fake Edit", ["image_to_image"])];
    const visionOptions = [modelOption("fake-vision", "Fake Vision", ["vision"])];
    const textOptions = [modelOption("default-chat", "Default Chat", ["chat"])];

    await renderModelSettings(modelSettingsPayload({
      roleplay_shared_models: { setting_key: "roleplay_shared_models", enabled: false },
      task_model_selectors: [
        modelSelector("chat", textOptions),
        modelSelector("context_update", structuredOptions),
        modelSelector("character_image_description", visionOptions)
      ],
      roleplay_model_groups: [
        {
          roleplay_type: "full_roleplay",
          label: "Generic Roleplay",
          selectors: [
            modelSelector("chat_full_roleplay", textOptions),
            modelSelector("context_update", structuredOptions)
          ]
        },
        {
          roleplay_type: "character_interaction",
          label: "Single-character Dating Sim",
          selectors: [
            modelSelector("chat_character_interaction", textOptions),
            modelSelector("character_interaction_character_image_edit_generation", editOptions)
          ]
        },
        {
          roleplay_type: "dating_sim",
          label: "Dating Sim",
          selectors: [
            modelSelector("chat_dating_sim", textOptions),
            modelSelector("dating_sim_context_update", structuredOptions)
          ]
        }
      ],
      scenario_section_model_selectors: [
        modelSelector("scenario_generation_section_opening_message", textOptions, "default-chat", { label: "Opening Message", section_id: "opening_message" })
      ]
    }));

    expect(screen.getByText("Shared roleplay models")).toBeInTheDocument();
    expect(screen.queryByText("Roleplay Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("Per-task Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("Scenario Section Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("Opening Message")).not.toBeInTheDocument();
    expect(screen.queryByText("Character Interaction Character Image Edit")).not.toBeInTheDocument();

    const narratorLane = await expandRoutingLane("Narrator");
    expect(within(narratorLane).getByText("Chat")).toBeInTheDocument();
    expect(within(narratorLane).getByText("Full Roleplay Chat")).toBeInTheDocument();
    expect(within(narratorLane).queryByText("Character Interaction Chat")).not.toBeInTheDocument();
    expect(within(narratorLane).getByText("Dating Sim Chat")).toBeInTheDocument();
    expect(screen.queryByText("Character Interaction Character Image Edit")).not.toBeInTheDocument();

    const scenarioLane = await expandRoutingLane("Scenario Writer");
    expect(within(scenarioLane).getByText("Opening Message")).toBeInTheDocument();

    const characterImageDescriptionLane = await expandRoutingLane("Image Details");
    expect(within(characterImageDescriptionLane).getAllByText("Image Details").length).toBeGreaterThanOrEqual(1);
    expect(screen.queryByText("Other Model Tasks")).not.toBeInTheDocument();
  });

  it("saves and clears scenario section model overrides", async () => {
    const modelOptions = [
      {
        provider: "fake",
        model_id: "default-chat",
        display_name: "Default Chat",
        available: true,
        capabilities: ["chat"]
      },
      {
        provider: "fake",
        model_id: "section-chat",
        display_name: "Section Chat",
        available: true,
        capabilities: ["chat"]
      }
    ];
    const settingsPayload = modelSettingsPayload({
      scenario_section_model_selectors: [
        {
          task: "scenario_generation_section_opening_message",
          label: "Opening Message",
          section_id: "opening_message",
          selected_provider: "fake",
          selected_model_id: "section-chat",
          selected_available: true,
          inherited_provider: "fake",
          inherited_model_id: "default-chat",
          clearable: true,
          warning: null,
          options: modelOptions
        },
        {
          task: "scenario_generation_section_premise",
          label: "Premise",
          section_id: "premise",
          selected_provider: null,
          selected_model_id: null,
          selected_available: false,
          inherited_provider: "fake",
          inherited_model_id: "default-chat",
          clearable: false,
          warning: null,
          options: modelOptions
        }
      ]
    });
    const fetchMock = await renderModelSettings(settingsPayload);

    const lane = await expandRoutingLane("Scenario Writer");
    expect(within(lane).getByText("Default: fake / default-chat")).toBeInTheDocument();
    await userEvent.selectOptions(screen.getByLabelText("Opening Message model"), "fake\u0000default-chat");
    await userEvent.click(screen.getByRole("button", { name: /use default/i }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/model-preference")).toBe(true));
    const preferenceCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/model-preference");
    expect(JSON.parse(String(preferenceCall?.[1].body))).toEqual({
      task: "scenario_generation_section_opening_message",
      provider: "fake",
      model_id: "default-chat"
    });
    expect(fetchMock.mock.calls.some(([path, init]) => (
      path === "/api/settings/model-preference/scenario_generation_section_opening_message" &&
      init?.method === "DELETE"
    ))).toBe(true);
    const inheritedSelector = screen.getByText("Premise").closest(".model-selector");
    expect(inheritedSelector).not.toBeNull();
    expect(
      within(inheritedSelector as HTMLElement).queryByRole("button", {
        name: /use default/i
      })
    ).not.toBeInTheDocument();
  });

  it("clears per-task model selectors", async () => {
    const modelOptions = [
      modelOption("alpha", "Alpha Chat", ["chat"]),
      modelOption("beta", "Beta Chat", ["chat"])
    ];
    const settingsPayload = modelSettingsPayload({
      task_model_selectors: [
        modelSelector("chat", modelOptions, "alpha", { clearable: true })
      ]
    });
    const fetchMock = await renderModelSettings(settingsPayload);

    await expandRoutingLane("Narrator");
    const selector = screen.getByText("Chat").closest(".model-selector");
    expect(selector).not.toBeNull();
    await userEvent.click(
      within(selector as HTMLElement).getByRole("button", { name: /use default/i })
    );

    await waitFor(() => expect(fetchMock.mock.calls.some(([path, init]) => (
      path === "/api/settings/model-preference/chat" &&
      init?.method === "DELETE"
    ))).toBe(true));
  });

  it("saves per-task model thinking levels", async () => {
    const modelOptions = [
      modelOption("alpha", "Alpha Chat", ["chat"], "fake", null, modelThinkingSupport())
    ];
    const settingsPayload = modelSettingsPayload({
      task_model_selectors: [
        modelSelector("chat", modelOptions, "alpha", {
          thinking: thinkingControl("chat", "provider_default", true, "alpha")
        })
      ]
    });
    const fetchMock = await renderModelSettings(settingsPayload);

    await expandRoutingLane("Narrator");
    const selector = screen.getAllByText("chat")
      .map((item) => item.closest(".model-selector"))
      .find(Boolean);
    expect(selector).not.toBeNull();
    await userEvent.selectOptions(
      within(selector as HTMLElement).getByLabelText(/thinking level/i),
      "high"
    );

    await waitFor(() => expect(fetchMock.mock.calls.some(([path]) => path === "/api/settings/model-thinking")).toBe(true));
    const thinkingCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/model-thinking");
    expect(JSON.parse(String(thinkingCall?.[1].body))).toEqual({
      task: "chat",
      provider: "fake",
      model_id: "alpha",
      level: "high"
    });
  });

  it("disables model thinking levels for unsupported models", async () => {
    const modelOptions = [
      modelOption("alpha", "Alpha Chat", ["chat"])
    ];
    const settingsPayload = modelSettingsPayload({
      task_model_selectors: [
        modelSelector("chat", modelOptions, "alpha", {
          thinking: thinkingControl("chat", "provider_default", false, "alpha")
        })
      ]
    });
    await renderModelSettings(settingsPayload);

    await expandRoutingLane("Narrator");
    const selector = screen.getAllByText("chat")
      .map((item) => item.closest(".model-selector"))
      .find(Boolean);
    expect(selector).not.toBeNull();
    expect(within(selector as HTMLElement).getByLabelText(/thinking level/i)).toBeDisabled();
  });

  it("clears roleplay override model selectors", async () => {
    const modelOptions = [
      modelOption("alpha", "Alpha Chat", ["chat"]),
      modelOption("beta", "Beta Chat", ["chat"])
    ];
    const settingsPayload = modelSettingsPayload({
      roleplay_shared_models: {
        setting_key: "roleplay_shared_models",
        enabled: false
      },
      roleplay_model_groups: [
        {
          roleplay_type: "full_roleplay",
          label: "Full Roleplay",
          selectors: [
            modelSelector("full_roleplay_chat", modelOptions, "alpha", {
              clearable: true,
              label: "Full Roleplay Chat"
            })
          ]
        }
      ]
    });
    const fetchMock = await renderModelSettings(settingsPayload);

    await expandRoutingLane("Narrator");
    const selector = screen.getByText("Full Roleplay Chat").closest(".model-selector");
    expect(selector).not.toBeNull();
    await userEvent.click(
      within(selector as HTMLElement).getByRole("button", { name: /use default/i })
    );

    await waitFor(() => expect(fetchMock.mock.calls.some(([path, init]) => (
      path === "/api/settings/model-preference/full_roleplay_chat" &&
      init?.method === "DELETE"
    ))).toBe(true));
  });

  it("filters per-task model selectors before saving a preference", async () => {
    const settingsPayload = modelSettingsPayload({
      task_model_selectors: [
        {
          task: "context_update",
          selected_provider: "fake",
          selected_model_id: "alpha",
          selected_available: true,
          warning: null,
          options: [
            {
              provider: "fake",
              model_id: "alpha",
              display_name: "Alpha Chat",
              available: true,
              capabilities: ["chat"]
            },
            {
              provider: "openrouter",
              model_id: "qwen/qwen3-32b",
              display_name: "Qwen 3 32B",
              available: true,
              capabilities: ["structured_output"]
            },
            {
              provider: "anthropic",
              model_id: "claude-sonnet-4",
              display_name: "Claude Sonnet 4",
              available: true,
              capabilities: ["chat"]
            }
          ]
        }
      ]
    });
    const fetchMock = await renderModelSettings(settingsPayload);

    await expandRoutingLane("World Updates");
    const search = screen.getByLabelText("Context Update model search");
    const select = screen.getByLabelText("Context Update model");

    await userEvent.type(search, "qwen");

    expect(within(select).getByRole("option", { name: "Qwen 3 32B - openrouter/qwen/qwen3-32b" })).toBeInTheDocument();
    expect(within(select).queryByRole("option", { name: "Claude Sonnet 4 - anthropic/claude-sonnet-4" })).not.toBeInTheDocument();

    await userEvent.selectOptions(select, "openrouter\u0000qwen/qwen3-32b");

    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith("/api/settings/model-preference", expect.anything()));
    const preferenceCall = fetchMock.mock.calls.find(([path]) => path === "/api/settings/model-preference");
    expect(JSON.parse(String(preferenceCall?.[1].body))).toEqual({
      task: "context_update",
      provider: "openrouter",
      model_id: "qwen/qwen3-32b"
    });
  });

  it("matches model search against unordered provider and model tokens", async () => {
    const settingsPayload = modelSettingsPayload({
      task_model_selectors: [
        {
          task: "context_update",
          selected_provider: "fake",
          selected_model_id: "alpha",
          selected_available: true,
          warning: null,
          options: [
            modelOption("alpha", "Alpha Chat", ["chat"]),
            modelOption("qwen/qwen3-32b", "Qwen 3 32B", ["structured_output"], "openrouter"),
            modelOption("claude-sonnet-4", "Claude Sonnet 4", ["chat"], "anthropic")
          ]
        }
      ]
    });
    await renderModelSettings(settingsPayload);

    await expandRoutingLane("World Updates");
    const search = screen.getByLabelText("Context Update model search");
    const select = screen.getByLabelText("Context Update model");

    await userEvent.type(search, "openrouter 32b");

    expect(screen.getByText("1 match")).toBeInTheDocument();
    expect(within(select).getByRole("option", { name: "Qwen 3 32B - openrouter/qwen/qwen3-32b" })).toBeInTheDocument();
    expect(within(select).queryByRole("option", { name: "Alpha Chat - fake/alpha" })).not.toBeInTheDocument();
    expect(within(select).queryByRole("option", { name: "Claude Sonnet 4 - anthropic/claude-sonnet-4" })).not.toBeInTheDocument();
  });

  it("shows model pricing in routing lanes and per-task selectors", async () => {
    const pricedOptions = [
      modelOption("chat-a", "Chat A", ["chat"], "fake", {
        input_per_million_tokens_usd: "0.15",
        output_per_million_tokens_usd: "0.6",
        cache_read_per_million_tokens_usd: "0.01",
        cache_write_per_million_tokens_usd: "0.02"
      })
    ];
    await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("chat", pricedOptions)
      ]
    }));

    const routingSelect = screen.getByLabelText("Narrator model");
    expect(within(routingSelect).getByRole("option", {
      name: "Chat A - fake/chat-a · $0.15 in / $0.60 out per 1M · cache $0.01 read / $0.02 write per 1M"
    })).toBeInTheDocument();
    expect(screen.getByText("Input $0.15 / output $0.60 per 1M tokens · Cache read $0.01 / write $0.02 per 1M tokens")).toBeInTheDocument();

    await expandRoutingLane("Narrator");
    const selector = screen.getByText("Chat").closest(".model-selector");
    expect(selector).not.toBeNull();
    expect(within(selector as HTMLElement).getByText("Input $0.15 / output $0.60 per 1M tokens · Cache read $0.01 / write $0.02 per 1M tokens")).toBeInTheDocument();
  });

  it("does not show inherited pricing for an unavailable explicit model override", async () => {
    const inheritedOption = modelOption("default-chat", "Default Chat", ["chat"], "fake", {
      input_per_million_tokens_usd: "0.15",
      output_per_million_tokens_usd: "0.6"
    });
    await renderModelSettings(modelSettingsPayload({
      scenario_section_model_selectors: [
        {
          task: "scenario_generation_section_opening_message",
          label: "Opening Message",
          section_id: "opening_message",
          selected_provider: "fake",
          selected_model_id: "retired-chat",
          selected_available: false,
          inherited_provider: "fake",
          inherited_model_id: "default-chat",
          clearable: true,
          warning: "Selected model is unavailable",
          options: [inheritedOption]
        }
      ]
    }));

    await expandRoutingLane("Scenario Writer");
    const selector = screen.getByText("Opening Message").closest(".model-selector");
    expect(selector).not.toBeNull();
    expect(within(selector as HTMLElement).getByRole("option", {
      name: "retired-chat - fake/retired-chat unavailable"
    })).toBeInTheDocument();
    expect(within(selector as HTMLElement).getByText("Pricing unavailable")).toBeInTheDocument();
    expect(within(selector as HTMLElement).queryByText("Input $0.15 / output $0.60 per 1M tokens")).not.toBeInTheDocument();
  });

  it("keeps only unmapped selectors in other model tasks", async () => {
    const chatOptions = [modelOption("chat-a", "Chat A", ["chat"])];
    const structuredOptions = [modelOption("structured-a", "Structured A", ["structured_output"])];
    const visionOptions = [modelOption("vision-a", "Vision A", ["vision"])];

    await renderModelSettings(modelSettingsPayload({
      roleplay_shared_models: { setting_key: "roleplay_shared_models", enabled: false },
      task_model_selectors: [
        modelSelector("response_verification", structuredOptions),
        modelSelector("memory_curation", structuredOptions),
        modelSelector("character_action_planning", structuredOptions),
        modelSelector("character_image_description", visionOptions),
        modelSelector("experimental_model_task", structuredOptions)
      ],
      roleplay_model_groups: [
        {
          roleplay_type: "full_roleplay",
          label: "Full Roleplay",
          selectors: [
            modelSelector("chat_full_roleplay", chatOptions),
            modelSelector("full_roleplay_response_planning", structuredOptions),
            modelSelector("full_roleplay_director_pressure", structuredOptions)
          ]
        }
      ]
    }));

    expect(screen.queryByText("Roleplay Overrides")).not.toBeInTheDocument();
    expect(screen.queryByText("Per-task Overrides")).not.toBeInTheDocument();

    const plannerLane = await expandRoutingLane("Narrator Planner");
    expect(within(plannerLane).getAllByText("Narrator Planner").length).toBeGreaterThanOrEqual(1);
    expect(within(plannerLane).getByText("Full Roleplay Narrator Planner")).toBeInTheDocument();

    const directorLane = await expandRoutingLane("Director Pressure");
    expect(within(directorLane).getAllByText("Director Pressure").length).toBeGreaterThanOrEqual(1);
    expect(within(directorLane).getByText("Full Roleplay Director Pressure")).toBeInTheDocument();

    const characterImageDescriptionLane = await expandRoutingLane("Image Details");
    expect(within(characterImageDescriptionLane).getAllByText("Image Details").length).toBeGreaterThanOrEqual(1);

    const otherSection = screen.getByText("Other Model Tasks").closest("section");
    expect(otherSection).not.toBeNull();
    expect(within(otherSection as HTMLElement).getByText("Other")).toBeInTheDocument();
    expect(within(otherSection as HTMLElement).getByText("Experimental Model Task")).toBeInTheDocument();
    expect(within(otherSection as HTMLElement).queryByText("Image Details")).not.toBeInTheDocument();
    expect(within(otherSection as HTMLElement).queryByText("Narrator Verifier")).not.toBeInTheDocument();
    expect(within(otherSection as HTMLElement).queryByText("Memory Curation")).not.toBeInTheDocument();
    expect(within(otherSection as HTMLElement).queryByText("Character Action Planning")).not.toBeInTheDocument();
  });

  it("shows admin user management only for admins", async () => {
    const fetchMock = settingsFetch(modelSettingsPayload());
    vi.stubGlobal("fetch", fetchMock);
    const { SettingsPanel } = await import("./main");

    render(
      <QueryClientProvider client={new QueryClient()}>
        <SettingsPanel
          runJob={vi.fn()}
          currentUser={{ id: "user-1", username: "Mira", role: "user", status: "active" }}
        />
      </QueryClientProvider>
    );

    expect(await screen.findByRole("tab", { name: "Save" })).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Providers" })).not.toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "Users" })).not.toBeInTheDocument();
  });

  it("lets admins list, create, edit, and reset users", async () => {
    const fetchMock = vi.fn().mockImplementation((path: string, init?: RequestInit) => {
      const ok = (payload: unknown) => Promise.resolve({
        ok: true,
        status: 200,
        json: async () => payload
      });
      if (path === "/api/settings") return ok(modelSettingsPayload());
      if (path === "/api/admin/users" && (!init?.method || init.method === "GET")) {
        return ok({
          users: [
            {
              id: "admin-1",
              username: "Mira",
              role: "admin",
              status: "active",
              content_rating: "pg-13",
              created_at: "2026-01-01T00:00:00+00:00",
              updated_at: null
            },
            {
              id: "user-2",
              username: "Rook",
              role: "user",
              status: "active",
              content_rating: "r",
              created_at: "2026-01-01T00:00:00+00:00",
              updated_at: null
            }
          ]
        });
      }
      if (path === "/api/admin/users" && init?.method === "POST") {
        return ok({
          user: {
            id: "user-3",
            username: "Ilyra",
            role: "child",
            status: "active",
            content_rating: "pg",
            created_at: "2026-01-01T00:00:00+00:00",
            updated_at: null
          }
        });
      }
      if (path === "/api/admin/users/user-2" && init?.method === "PATCH") {
        return ok({
          user: {
            id: "user-2",
            username: "Rook",
            role: "child",
            status: "disabled",
            content_rating: "pg-13",
            created_at: "2026-01-01T00:00:00+00:00",
            updated_at: "2026-01-02T00:00:00+00:00"
          }
        });
      }
      if (path === "/api/admin/users/user-2/password" && init?.method === "POST") {
        return ok({
          user: {
            id: "user-2",
            username: "Rook",
            role: "child",
            status: "disabled",
            content_rating: "pg-13",
            created_at: "2026-01-01T00:00:00+00:00",
            updated_at: "2026-01-02T00:00:00+00:00"
          }
        });
      }
      return ok({});
    });

    await renderAdminUserSettings(fetchMock);

    expect(await screen.findByText("Mira")).toBeInTheDocument();
    expect(screen.getByText("Rook")).toBeInTheDocument();

    await userEvent.type(screen.getByLabelText("New username"), "Ilyra");
    await userEvent.type(screen.getByLabelText("New password"), "temporary pass");
    await userEvent.selectOptions(screen.getByLabelText("New role"), "child");
    await userEvent.click(screen.getByRole("button", { name: "Create user" }));

    await userEvent.selectOptions(screen.getByLabelText("Rook role"), "child");
    await userEvent.selectOptions(screen.getByLabelText("Rook content rating"), "pg-13");
    await userEvent.selectOptions(screen.getByLabelText("Rook status"), "disabled");
    await userEvent.type(screen.getByLabelText("Rook new password"), "new password");
    await userEvent.click(screen.getByRole("button", { name: "Reset Rook password" }));

    await waitFor(() => {
      expect(fetchMock.mock.calls.some(([path, init]) => {
        if (path !== "/api/admin/users" || init?.method !== "POST") return false;
        const body = JSON.parse(String(init.body));
        return body.username === "Ilyra" && body.role === "child";
      })).toBe(true);
      expect(fetchMock.mock.calls.some(([path, init]) => {
        if (path !== "/api/admin/users/user-2" || init?.method !== "PATCH") return false;
        return JSON.parse(String(init.body)).role === "child";
      })).toBe(true);
      expect(fetchMock.mock.calls.some(([path, init]) => {
        if (path !== "/api/admin/users/user-2" || init?.method !== "PATCH") return false;
        return JSON.parse(String(init.body)).status === "disabled";
      })).toBe(true);
      expect(fetchMock.mock.calls.some(([path, init]) => {
        if (path !== "/api/admin/users/user-2" || init?.method !== "PATCH") return false;
        return JSON.parse(String(init.body)).content_rating === "pg-13";
      })).toBe(true);
      expect(fetchMock.mock.calls.some(([path, init]) => {
        if (path !== "/api/admin/users/user-2/password" || init?.method !== "POST") return false;
        return JSON.parse(String(init.body)).password === "new password";
      })).toBe(true);
    });
  });

  it("shows the mixed placeholder when lane selectors disagree", async () => {
    const toolOptions = [
      modelOption("tool-a", "Tool A", ["tool_calling"]),
      modelOption("tool-b", "Tool B", ["tool_calling"])
    ];
    await renderModelSettings(modelSettingsPayload({
      task_model_selectors: [
        modelSelector("context_update", toolOptions, "tool-a"),
        modelSelector("state_memory", toolOptions, "tool-b")
      ]
    }));

    const select = screen.getByLabelText("World Updates model") as HTMLSelectElement;
    const mixedOption = within(select).getByRole("option", { name: "Mixed selections" }) as HTMLOptionElement;
    expect(select.value).toBe("");
    expect(mixedOption.selected).toBe(true);
  });
});

describe("session shell", () => {
  it("loads the authenticated workbench bundle only after login", async () => {
    const fetchMock = sessionFetch({ bootstrapRequired: false, meStatus: 401 });
    vi.stubGlobal("fetch", fetchMock);
    const loadWorkbench = vi.fn(async () => ({
      default: ({ currentUser }: { currentUser: { username: string } }) => (
        <main>
          <h1>Lazy Workbench</h1>
          <span>{currentUser.username}</span>
        </main>
      )
    }));
    const { App } = await import("./appShell");

    render(<App loadAuthenticatedWorkbench={loadWorkbench} />);

    expect(await screen.findByRole("heading", { name: "Log in to Bragi" })).toBeInTheDocument();
    expect(loadWorkbench).not.toHaveBeenCalled();
    expect(fetchMock.mock.calls.some(([path]) => String(path).startsWith("/api/runtime"))).toBe(false);

    await userEvent.type(screen.getByLabelText("Username"), "Mira");
    await userEvent.type(screen.getByLabelText("Password"), "correct horse");
    await userEvent.click(screen.getByRole("button", { name: "Log in" }));

    expect(await screen.findByRole("heading", { name: "Lazy Workbench" })).toBeInTheDocument();
    expect(screen.getByText("Mira")).toBeInTheDocument();
    expect(loadWorkbench).toHaveBeenCalledTimes(1);
  });

  it("keeps first-admin bootstrap on the lightweight shell until setup completes", async () => {
    const fetchMock = sessionFetch({ bootstrapRequired: true });
    vi.stubGlobal("fetch", fetchMock);
    const loadWorkbench = vi.fn(async () => ({
      default: ({ currentUser }: { currentUser: { username: string } }) => (
        <main>
          <h1>Lazy Workbench</h1>
          <span>{currentUser.username}</span>
        </main>
      )
    }));
    const { App } = await import("./appShell");

    render(<App loadAuthenticatedWorkbench={loadWorkbench} />);

    expect(await screen.findByRole("heading", { name: "Create first admin" })).toBeInTheDocument();
    expect(loadWorkbench).not.toHaveBeenCalled();
    expect(fetchMock.mock.calls.some(([path]) => String(path).startsWith("/api/runtime"))).toBe(false);

    await userEvent.type(screen.getByLabelText("Username"), "Mira");
    await userEvent.type(screen.getByLabelText("Password"), "correct horse");
    await userEvent.click(screen.getByRole("button", { name: "Create admin" }));

    expect(await screen.findByRole("heading", { name: "Lazy Workbench" })).toBeInTheDocument();
    expect(loadWorkbench).toHaveBeenCalledTimes(1);
  });

  it("loads bootstrap and current user state from one session endpoint", async () => {
    const fetchMock = sessionFetch({ bootstrapRequired: false, meStatus: 200 });
    vi.stubGlobal("fetch", fetchMock);
    const { App } = await import("./main");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Lantern Keep" })).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/auth/session")).toBe(true);
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/bootstrap/status")).toBe(false);
    expect(fetchMock.mock.calls.some(([path]) => path === "/api/auth/me")).toBe(false);
  });

  it("creates the first admin before loading protected workbench data", async () => {
    const fetchMock = sessionFetch({ bootstrapRequired: true });
    vi.stubGlobal("fetch", fetchMock);
    const { App } = await import("./main");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Create first admin" })).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => String(path).startsWith("/api/runtime"))).toBe(false);

    await userEvent.type(screen.getByLabelText("Username"), "Mira");
    await userEvent.type(screen.getByLabelText("Password"), "correct horse");
    await userEvent.click(screen.getByRole("button", { name: "Create admin" }));

    expect(await screen.findByRole("heading", { name: "Lantern Keep" })).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => String(path).startsWith("/api/runtime"))).toBe(true);
  });

  it("submits the remote bootstrap setup token when required", async () => {
    const fetchMock = sessionFetch({ bootstrapRequired: true, setupTokenRequired: true });
    vi.stubGlobal("fetch", fetchMock);
    const { App } = await import("./main");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Create first admin" })).toBeInTheDocument();
    await userEvent.type(screen.getByLabelText("Username"), "Mira");
    await userEvent.type(screen.getByLabelText("Password"), "correct horse");
    await userEvent.type(screen.getByLabelText("Setup token"), "setup-secret");
    await userEvent.click(screen.getByRole("button", { name: "Create admin" }));

    expect(fetchMock.mock.calls.some(([path, init]) => {
      if (path !== "/api/bootstrap/admin" || init?.method !== "POST") return false;
      const body = JSON.parse(String(init.body));
      return body.setup_token === "setup-secret";
    })).toBe(true);
  });

  it("logs in before mounting the workbench", async () => {
    const fetchMock = sessionFetch({ bootstrapRequired: false, meStatus: 401 });
    vi.stubGlobal("fetch", fetchMock);
    const { App } = await import("./main");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Log in to Bragi" })).toBeInTheDocument();
    expect(fetchMock.mock.calls.some(([path]) => String(path).startsWith("/api/runtime"))).toBe(false);

    await userEvent.type(screen.getByLabelText("Username"), "Mira");
    await userEvent.type(screen.getByLabelText("Password"), "correct horse");
    await userEvent.click(screen.getByRole("button", { name: "Log in" }));

    expect(await screen.findByRole("heading", { name: "Lantern Keep" })).toBeInTheDocument();
  });

  it("returns to login when a protected request reports an expired session", async () => {
    const fetchMock = sessionFetch({
      bootstrapRequired: false,
      meStatus: 200,
      runtimeStatus: 401
    });
    vi.stubGlobal("fetch", fetchMock);
    const { App } = await import("./main");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Log in to Bragi" })).toBeInTheDocument();
    expect(screen.getByText("Session expired. Log in again to continue.")).toBeInTheDocument();
  });

  it("shows forbidden protected request failures without logging out", async () => {
    const fetchMock = sessionFetch({
      bootstrapRequired: false,
      meStatus: 200,
      runtimeStatus: 403,
      runtimeDetail: "Admin access required"
    });
    vi.stubGlobal("fetch", fetchMock);
    const { App } = await import("./main");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Bragi Workbench" })).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent("Admin access required");
    expect(screen.queryByRole("heading", { name: "Log in to Bragi" })).not.toBeInTheDocument();
  });

  it("logs out and clears the workbench", async () => {
    const fetchMock = sessionFetch({ bootstrapRequired: false, meStatus: 200 });
    vi.stubGlobal("fetch", fetchMock);
    const { App } = await import("./main");

    render(<App />);

    expect(await screen.findByRole("heading", { name: "Lantern Keep" })).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Log out" }));

    expect(await screen.findByRole("heading", { name: "Log in to Bragi" })).toBeInTheDocument();
    expect(screen.queryByRole("heading", { name: "Lantern Keep" })).not.toBeInTheDocument();
  });
});
