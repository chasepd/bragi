import React, { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, Image, Loader2, Plus, RefreshCw, Upload, Wand2, X } from "lucide-react";
import { api, postJson } from "./api";
import type { Job, RuntimeModel, ScenarioDraft, ScenarioWizardFlow } from "./api";
import {
  defaultFlows,
  defaultSecondaryScenarioType,
  canUseChildRestrictedControls,
  DialogForm,
  EmptyState,
  initialMediaDraftLabel,
  InlineNotice,
  labelize,
  ModalBackdrop,
  MANUAL_BASE_SECTION_IDS,
  MANUAL_SCENARIO_TEXTAREA_FIELDS,
  normalizedScenarioTypes,
  PanelHeader,
  progressLabel,
  runtimeQueryKey,
  runtimeResultError,
  scenarioCreationFlow,
  scenarioSectionEditorGroups,
  scenarioSectionResultText,
  SegmentedTabs,
  isRuntimeModel,
  useDialogJobWatcher
} from "./workbenchCore";
import type { CurrentUser, RunJob, ScenarioDraftPrefill, ScenarioForm, ScenarioFormTextField, SegmentOption } from "./workbenchCore";

export function ScenarioDialog({
  model,
  initialMode = "manual",
  initialDraftPrefill,
  initialSeed,
  currentUser = null,
  onClose,
  onRuntimeChanged,
  onScenarioListChanged,
  runJob
}: {
  model?: RuntimeModel;
  initialMode?: "manual" | "draft";
  initialDraftPrefill?: ScenarioDraftPrefill;
  initialSeed?: string;
  currentUser?: CurrentUser | null;
  onClose: () => void;
  onRuntimeChanged: (model: RuntimeModel) => void;
  onScenarioListChanged?: () => void;
  runJob: RunJob;
}) {
  const client = useQueryClient();
  const watchDialogJob = useDialogJobWatcher();
  const titleId = React.useId();
  const [mode, setMode] = useState<"manual" | "draft">(initialMode);
  const [form, setForm] = useState<ScenarioForm>({
    scenario_type: initialDraftPrefill?.scenario_type ?? model?.scenario_draft?.scenario_type ?? "full_roleplay",
    scenario_types: initialDraftPrefill?.scenario_types?.length
      ? initialDraftPrefill.scenario_types
      : model?.scenario_draft?.scenario_types?.length
        ? model.scenario_draft.scenario_types
        : [model?.scenario_draft?.scenario_type ?? "full_roleplay"],
    action_choices_enabled: initialDraftPrefill?.action_choices_enabled ?? model?.scenario_draft?.action_choices_enabled ?? false,
    title: "",
    premise: "",
    player_role: "",
    player_character_name: "",
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
    opening_message: ""
  });
  const [seed, setSeed] = useState(initialDraftPrefill?.seed ?? initialSeed ?? "");
  const [draft, setDraft] = useState<ScenarioDraft | null>(
    initialDraftPrefill ? null : model?.scenario_draft ?? null
  );
  const [draftVersion, setDraftVersion] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [draftProgress, setDraftProgress] = useState("");
  const flows = model?.scenario_wizard?.flows ?? defaultFlows();
  const selectedScenarioTypes = normalizedScenarioTypes(form.scenario_type, form.scenario_types);
  const hybridEnabled = selectedScenarioTypes.length > 1;
  const secondaryScenarioType = selectedScenarioTypes[1] ?? defaultSecondaryScenarioType(flows, form.scenario_type);
  const flow = scenarioCreationFlow(model, selectedScenarioTypes);
  const modeOptions: SegmentOption<"manual" | "draft">[] = [
    { value: "manual", label: "Manual" },
    { value: "draft", label: "AI draft" }
  ];
  const setScenarioDraft = (nextDraft: ScenarioDraft | null, replaceEditor = false) => {
    setDraft(nextDraft);
    if (replaceEditor) setDraftVersion((current) => current + 1);
    if (nextDraft?.scenario_type) {
      setForm((current) => ({
        ...current,
        scenario_type: nextDraft.scenario_type,
        scenario_types: normalizedScenarioTypes(nextDraft.scenario_type, nextDraft.scenario_types),
        action_choices_enabled: Boolean(nextDraft.action_choices_enabled)
      }));
    }
  };
  const setPrimaryScenarioType = (scenarioType: string) => {
    const nextSecondary = selectedScenarioTypes[1] && selectedScenarioTypes[1] !== scenarioType
      ? selectedScenarioTypes[1]
      : defaultSecondaryScenarioType(flows, scenarioType);
    setForm({
      ...form,
      scenario_type: scenarioType,
      scenario_types: hybridEnabled ? [scenarioType, nextSecondary] : [scenarioType]
    });
  };
  const setHybridEnabled = (enabled: boolean) => {
    setForm({
      ...form,
      scenario_types: enabled ? [form.scenario_type, secondaryScenarioType] : [form.scenario_type]
    });
  };
  const setSecondaryScenarioType = (scenarioType: string) => {
    setForm({ ...form, scenario_types: [form.scenario_type, scenarioType] });
  };

  const generateDraft = async () => {
    setBusy(true);
    setError("");
    setDraftProgress("");
    try {
      const created = await postJson<Job>("/api/scenarios/draft", {
        scenario_type: form.scenario_type,
        scenario_types: selectedScenarioTypes,
        seed,
        action_choices_enabled: form.action_choices_enabled
      });
      watchDialogJob(
        created.id,
        (done) => {
          setBusy(false);
          setDraftProgress("");
          if (done.status === "succeeded" && isRuntimeModel(done.result)) {
            setScenarioDraft(done.result.scenario_draft ?? null, true);
            client.setQueryData(runtimeQueryKey(done.result.active_save_id ?? null), done.result);
          } else if (done.status === "failed") {
            setError(done.error || "Scenario draft failed.");
          } else if (done.status === "cancelled") {
            setError(done.error || "Scenario draft was cancelled.");
          }
        },
        (name, data) => {
          if (name === "progress") setDraftProgress(progressLabel(data));
        }
      );
    } catch (failure) {
      setBusy(false);
      setDraftProgress("");
      setError(failure instanceof Error ? failure.message : "Could not start scenario draft");
    }
  };

  return (
    <ModalBackdrop>
      <DialogForm
        className="scenario-dialog"
        titleId={titleId}
        onClose={onClose}
        onSubmit={async (event) => {
          event.preventDefault();
          if (mode !== "manual" || busy) return;
          setBusy(true);
          setError("");
          try {
            const result = await postJson<RuntimeModel>("/api/scenarios/manual", form);
            onRuntimeChanged(result);
            client.invalidateQueries({ queryKey: ["scenarios"] });
            onScenarioListChanged?.();
            onClose();
          } catch (failure) {
            setError(failure instanceof Error ? failure.message : "Could not create scenario");
          } finally {
            setBusy(false);
          }
        }}
      >
        <header>
          <h2 id={titleId}>New scenario</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <SegmentedTabs
          className="segmented"
          label="Scenario creation modes"
          value={mode}
          onChange={setMode}
          options={modeOptions}
        />
        <SegmentedTabs
          label="Scenario draft flows"
          value={form.scenario_type}
          onChange={setPrimaryScenarioType}
          options={flows.map((item) => ({ value: item.flow_id, label: item.label }))}
        />
        <>
            <label className="toggle-row compact-toggle">
              <input
                type="checkbox"
                checked={hybridEnabled}
                disabled={mode === "draft" && draft !== null}
                onChange={(event) => setHybridEnabled(event.target.checked)}
              />
              <span>Hybrid</span>
            </label>
            {hybridEnabled ? (
              <label className="field-label">
                <span>Second genre</span>
                <select
                  aria-label="Second genre"
                  value={secondaryScenarioType}
                  disabled={mode === "draft" && draft !== null}
                  onChange={(event) => setSecondaryScenarioType(event.target.value)}
                >
                  {flows
                    .filter((item) => item.flow_id !== form.scenario_type)
                    .map((item) => <option key={item.flow_id} value={item.flow_id}>{item.label}</option>)}
                </select>
              </label>
            ) : null}
        </>
        <label className="toggle-row compact-toggle">
            <input
              type="checkbox"
              checked={form.action_choices_enabled}
              disabled={mode === "draft" && draft !== null}
              onChange={(event) => setForm({ ...form, action_choices_enabled: event.target.checked })}
            />
            <span>Action choices</span>
        </label>
        {mode === "manual" ? <ManualScenarioForm form={form} setForm={setForm} flow={flow} /> : null}
        {mode === "draft" ? (
          <>
            <textarea className="tall-field" value={seed} onChange={(event) => setSeed(event.target.value)} placeholder={flow?.seed_prompt ?? "Describe the scenario"} aria-label="Scenario seed" />
            <div className="command-row end">
              <button type="button" className="secondary-command" disabled={busy || !seed.trim()} onClick={generateDraft}>
                {busy ? <Loader2 className="spin" size={15} /> : <Wand2 size={15} />} Generate
              </button>
            </div>
            {draft ? <ScenarioDraftEditor key={draftVersion} draft={draft} flow={flow} currentUser={currentUser} runJob={runJob} onSaved={onClose} onRuntimeChanged={onRuntimeChanged} onScenarioListChanged={onScenarioListChanged} setDraft={setScenarioDraft} /> : null}
          </>
        ) : null}
        {draftProgress ? <p className="scenario-draft-status" role="status" aria-live="polite">{draftProgress}</p> : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        {mode === "manual" ? (
          <div className="command-row end">
            <button className="primary-command compact" disabled={busy}>
              {busy ? <Loader2 className="spin" size={15} /> : <Plus size={15} />} Create
            </button>
          </div>
        ) : null}
      </DialogForm>
    </ModalBackdrop>
  );
}

function ManualScenarioForm({
  form,
  setForm,
  flow
}: {
  form: ScenarioForm;
  setForm: (form: ScenarioForm) => void;
  flow?: ScenarioWizardFlow;
}) {
  const update = (field: ScenarioFormTextField, value: string) => setForm({ ...form, [field]: value });
  const flowSectionIds = flow?.editable_section_ids ?? defaultFlows()[0].editable_section_ids;
  const actionChoiceSectionIds = form.action_choices_enabled ? ["choice_style"] : [];
  const extraSectionIds = [...new Set([...flowSectionIds, ...actionChoiceSectionIds])]
    .filter((sectionId) => !MANUAL_BASE_SECTION_IDS.has(sectionId))
    .filter(isScenarioFormTextField);
  return (
    <>
      <label className="field-label">
        <span>Title</span>
        <input required value={form.title} onChange={(e) => update("title", e.target.value)} />
      </label>
      <label className="field-label">
        <span>Premise</span>
        <textarea required value={form.premise} onChange={(e) => update("premise", e.target.value)} />
      </label>
      <label className="field-label">
        <span>Player Role</span>
        <input required value={form.player_role} onChange={(e) => update("player_role", e.target.value)} />
      </label>
      <label className="field-label">
        <span>Player Character</span>
        <input value={form.player_character_name} onChange={(e) => update("player_character_name", e.target.value)} />
      </label>
      {extraSectionIds.map((sectionId) => (
        <label className="field-label" key={sectionId}>
          <span>{labelize(sectionId)}</span>
          {MANUAL_SCENARIO_TEXTAREA_FIELDS.has(sectionId) ? (
            <textarea value={form[sectionId]} onChange={(e) => update(sectionId, e.target.value)} />
          ) : (
            <input value={form[sectionId]} onChange={(e) => update(sectionId, e.target.value)} />
          )}
        </label>
      ))}
      <label className="field-label">
        <span>Opening Message</span>
        <textarea value={form.opening_message} onChange={(e) => update("opening_message", e.target.value)} />
      </label>
    </>
  );
}

function isScenarioFormTextField(sectionId: string): sectionId is ScenarioFormTextField {
  return sectionId !== "action_choices_enabled"
    && sectionId !== "scenario_types"
    && sectionId in emptyScenarioFormFields();
}

function emptyScenarioFormFields(): ScenarioForm {
  return {
    scenario_type: "",
    scenario_types: [],
    action_choices_enabled: false,
    title: "",
    premise: "",
    player_role: "",
    player_character_name: "",
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
    opening_message: ""
  };
}

function ScenarioDraftEditor({
  draft,
  flow,
  currentUser,
  runJob,
  onSaved,
  onRuntimeChanged,
  onScenarioListChanged,
  setDraft
}: {
  draft: ScenarioDraft;
  flow?: ScenarioWizardFlow;
  currentUser?: CurrentUser | null;
  runJob: RunJob;
  onSaved: () => void;
  onRuntimeChanged: (model: RuntimeModel) => void;
  onScenarioListChanged?: () => void;
  setDraft: (draft: ScenarioDraft) => void;
}) {
  const client = useQueryClient();
  const watchDialogJob = useDialogJobWatcher();
  const [sections, setSections] = useState<Record<string, string>>(() => Object.fromEntries(draft.sections));
  const [openingImage, setOpeningImage] = useState(false);
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const baseGroups = draftReviewGroups(
    flow?.review_groups ?? [{ label: "Draft", section_ids: Object.keys(sections) }],
    draft.action_choices_enabled,
    sections
  );
  const groupedSectionIds = new Set(baseGroups.flatMap((group) => group.section_ids));
  const extraSectionIds = Object.keys(sections).filter((sectionId) => !groupedSectionIds.has(sectionId));
  const groups = extraSectionIds.length
    ? [...baseGroups, { label: "Continuity", section_ids: extraSectionIds }]
    : baseGroups;
  const initialMediaLabel = initialMediaDraftLabel(draft.scenario_type);
  const initialMediaAllowed = canUseChildRestrictedControls(currentUser);
  const updateSection = (sectionId: string, value: string) => {
    const next = { ...sections, [sectionId]: value };
    setSections(next);
    setDraft({ ...draft, sections: Object.entries(next) });
  };
  return (
    <div className="draft-editor">
      {groups.map((group) => (
        <details className="model-group" key={group.label} open>
          <summary>
            <div>
              <strong>{group.label}</strong>
              <span>{group.section_ids.length} sections</span>
            </div>
          </summary>
          <div className="model-selector-list">
            {group.section_ids.map((sectionId) => (
              <label className="field-label" key={sectionId}>
                <span>{labelize(sectionId)}</span>
                <textarea value={sections[sectionId] ?? ""} onChange={(event) => updateSection(sectionId, event.target.value)} />
                <button
                  type="button"
                  className="secondary-command"
                  onClick={async () => {
                    setError("");
                    try {
                      const created = await postJson<Job>("/api/scenarios/draft/section", {
                        scenario_type: draft.scenario_type,
                        scenario_types: normalizedScenarioTypes(draft.scenario_type, draft.scenario_types),
                        seed: draft.regeneration_seed,
                        section_id: sectionId,
                        sections,
                        action_choices_enabled: Boolean(draft.action_choices_enabled)
                      });
                      watchDialogJob(created.id, (done) => {
                        if (done.status === "succeeded") {
                          const regenerated = scenarioSectionResultText(done.result, sectionId);
                          const errorMessage = runtimeResultError(done.result);
                          if (regenerated !== null) {
                            updateSection(sectionId, regenerated);
                            setError("");
                          } else if (errorMessage) {
                            setError(errorMessage);
                          } else {
                            setError("Regenerated section was not returned");
                          }
                        }
                        if (done.error) setError(done.error);
                      });
                    } catch (failure) {
                      setError(failure instanceof Error ? failure.message : "Could not regenerate section");
                    }
                  }}
                >
                  <RefreshCw size={15} /> Regenerate
                </button>
              </label>
            ))}
          </div>
        </details>
      ))}
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      {initialMediaAllowed ? (
        <label className="toggle-row compact-toggle">
          <input type="checkbox" checked={openingImage} onChange={(event) => setOpeningImage(event.target.checked)} />
          <span>{initialMediaLabel}</span>
        </label>
      ) : null}
      <div className="command-row end">
        <button
          type="button"
          className="primary-command compact"
          disabled={saving}
          onClick={async () => {
            setSaving(true);
            setError("");
            try {
              const result = await postJson<RuntimeModel>("/api/scenarios/draft/save", {
                scenario_type: draft.scenario_type,
                scenario_types: normalizedScenarioTypes(draft.scenario_type, draft.scenario_types),
                sections,
                save_title: sections.title ?? "",
                source_metadata: Object.fromEntries(draft.source_metadata ?? []),
                action_choices_enabled: Boolean(draft.action_choices_enabled)
              });
              onRuntimeChanged(result);
              client.invalidateQueries({ queryKey: ["scenarios"] });
              onScenarioListChanged?.();
              const firstNarrator = result.chronicle.messages.find((message) => message.role === "narrator");
              if (initialMediaAllowed && openingImage && firstNarrator) {
                runJob(await postJson<Job>("/api/media/initial", { message_id: firstNarrator.message_id, save_id: result.active_save_id }));
              }
              onSaved();
            } catch (failure) {
              setError(failure instanceof Error ? failure.message : "Could not save draft");
              setSaving(false);
            }
          }}
        >
          {saving ? <Loader2 className="spin" size={15} /> : <Check size={15} />} Save draft
        </button>
      </div>
    </div>
  );
}

function draftReviewGroups(
  groups: { label: string; section_ids: string[] }[],
  actionChoicesEnabled: boolean | undefined,
  sections: Record<string, string>
): { label: string; section_ids: string[] }[] {
  if (!actionChoicesEnabled || !Object.prototype.hasOwnProperty.call(sections, "choice_style")) {
    return groups;
  }
  if (groups.some((group) => group.section_ids.includes("choice_style"))) {
    return groups;
  }
  const choicesGroup = { label: "Choices", section_ids: ["choice_style"] };
  const openingIndex = groups.findIndex((group) => group.label === "Opening");
  if (openingIndex < 0) return [...groups, choicesGroup];
  return [
    ...groups.slice(0, openingIndex),
    choicesGroup,
    ...groups.slice(openingIndex)
  ];
}
