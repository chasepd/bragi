import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Archive,
  Check,
  ChevronDown,
  Download,
  Edit3,
  Eye,
  FileWarning,
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
  WorldDataModel
} from "./api";
import {
  actionIcon,
  apiRead,
  canUseAdminControls,
  canUseChildRestrictedControls,
  CHARACTER_AUTO_ENHANCE_FIELD_SET,
  CHARACTER_AUTO_ENHANCE_LABELS,
  CHARACTER_LOCK_FIELD_ALIASES,
  CHARACTER_LOCK_FIELD_IDS,
  CHARACTER_LOCK_FIELDS,
  charactersPath,
  compactInlineTitle,
  ConfirmModal,
  DataViewer,
  DialogForm,
  DialogPanel,
  EditorDirtyStatus,
  EmptyState,
  formatUsd,
  imageDimensionPresetLabel,
  imageStylePresetLabel,
  InlineNotice,
  invalidateScenePresenceQueries,
  isCharacterRegistryModel,
  labelize,
  MarkdownView,
  mediaAssetPath,
  mediaAssetThumbnailPath,
  mergeReferenceUploadLocks,
  ModalBackdrop,
  ModelPricingLine,
  modelOptionLabel,
  modelOptionSelectLabel,
  modelPricingCompactLabel,
  openDownloadInNewTab,
  PanelHeader,
  PreviewModal,
  runtimeQueryKey,
  SAVE_SCOPED_SETTING_KEYS,
  SegmentedTabs,
  selectedOption,
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
import type { CharacterEditorTab, CurrentUser, RunJob } from "./workbenchCore";


export function CharactersPanel({
  activeSaveId,
  runJob,
  currentUser = null,
  characterTextsEnabled = false
}: {
  activeSaveId: string | null;
  runJob: RunJob;
  currentUser?: CurrentUser | null;
  characterTextsEnabled?: boolean;
}) {
  const [filter, setFilter] = useState<"present" | "all">("present");
  const [search, setSearch] = useState("");
  const [exportingCharacter, setExportingCharacter] = useState<{ id: string; name: string } | null>(null);
  const [draftCharacters, setDraftCharacters] = useState<CharacterRow[]>([]);
  const nextDraftCharacterId = useRef(1);
  const characters = useQuery({
    queryKey: ["characters", activeSaveId],
    queryFn: () => api<CharacterRegistryModel>(charactersPath(activeSaveId)),
    enabled: Boolean(activeSaveId)
  });
  const client = useQueryClient();
  const charactersMatchActiveSave = Boolean(activeSaveId && characters.data?.active_save_id === activeSaveId);
  const charactersStale = Boolean(activeSaveId && characters.data && characters.data.active_save_id !== activeSaveId);
  const canMutateCharacters = canUseChildRestrictedControls(currentUser);
  useEffect(() => {
    setDraftCharacters([]);
    nextDraftCharacterId.current = 1;
  }, [activeSaveId]);
  const applyRows = async (
    rows: unknown[],
    options: { autoEnhanceCreatedAgency?: boolean } = {}
  ) => {
    if (!charactersMatchActiveSave || !activeSaveId) {
      throw new Error("Characters are still loading for the active save");
    }
    const result = await postJson<CharacterRegistryApplyResult>("/api/characters/apply", {
      active_save_id: activeSaveId,
      auto_enhance_created_agency: Boolean(options.autoEnhanceCreatedAgency),
      edits: { characters: rows }
    });
    client.setQueryData(["characters", activeSaveId], result.model);
    client.invalidateQueries({ queryKey: ["world"] });
    client.invalidateQueries({ queryKey: ["runtime"] });
  };
  const rows = [...draftCharacters, ...(characters.data?.characters ?? [])].filter((row) => {
    const matchesFilter = filter === "all" || row.present;
    const haystack = `${row.name ?? ""} ${row.aliases_text ?? ""} ${row.role ?? ""} ${row.age ?? ""} ${row.status ?? ""}`.toLowerCase();
    return matchesFilter && haystack.includes(search.toLowerCase());
  });
  const addDraftCharacter = () => {
    const draftId = `draft-character-${nextDraftCharacterId.current++}`;
    setDraftCharacters((current) => [
      {
        __draft_character: true,
        __draft_id: draftId,
        character_id: "",
        name: "New character",
        relationships_json: "{}",
        present: true
      },
      ...current
    ]);
  };
  return (
    <aside className="right-panel">
      <PanelHeader icon={<Users size={18} />} title="Characters" />
      {canMutateCharacters ? (
        <div className="command-row save-bundle-actions">
          <CharacterBundleUpload
            activeSaveId={activeSaveId}
            disabled={!charactersMatchActiveSave}
            onImported={(model) => {
              if (activeSaveId) client.setQueryData(["characters", activeSaveId], model);
              client.invalidateQueries({ queryKey: ["characters"] });
              client.invalidateQueries({ queryKey: ["world"] });
              client.invalidateQueries({ queryKey: ["runtime"] });
            }}
          />
        </div>
      ) : null}
      <SegmentedTabs
        className="segmented settings-tabs"
        label="Character filters"
        value={filter}
        onChange={setFilter}
        options={[
          { value: "present", label: "Present" },
          { value: "all", label: "All" }
        ]}
      />
      <div className="inline-tool-form">
        <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search characters" aria-label="Search characters" />
        {canMutateCharacters ? (
          <button
            type="button"
            title="Add character"
            disabled={!charactersMatchActiveSave}
            onClick={addDraftCharacter}
          >
            <Plus size={15} />
          </button>
        ) : null}
      </div>
      {characters.data?.error ? <InlineNotice>{characters.data.error}</InlineNotice> : null}
      {charactersStale ? <InlineNotice polite>Refreshing characters for active save...</InlineNotice> : null}
      <div className="stack-list roomy">
        {rows.map((character, index) => (
          <details className="entity-detail" key={`${activeSaveId}:${String(character.__draft_id ?? character.character_id ?? character.id ?? index)}`}>
            <summary>
              <span className="entity-summary-title">
                <strong>{character.name ?? "Unnamed"}</strong>
                <span>{[
                  character.is_player_character ? "Player" : character.present ? "Present" : String(character.status || "Known"),
                  character.age ? `Age ${String(character.age)}` : ""
                ].filter(Boolean).join(" · ")}</span>
              </span>
              {canMutateCharacters ? (
                <span className="entity-summary-actions">
                  <button
                    type="button"
                    className={touchActionClassName()}
                    title={`Export ${String(character.name ?? "character")}`}
                    aria-label={`Export ${String(character.name ?? "character")}`}
                    disabled={!charactersMatchActiveSave || !character.character_id}
                    onClick={(event) => {
                      event.preventDefault();
                      event.stopPropagation();
                      if (!character.character_id) return;
                      setExportingCharacter({
                        id: String(character.character_id),
                        name: String(character.name ?? "character")
                      });
                    }}
                  >
                    <TouchActionContents icon={<Download size={14} />} label="Export" />
                  </button>
                </span>
              ) : null}
            </summary>
            <CharacterEditor
              character={character}
              linkTargets={characters.data?.link_targets ?? []}
              locationChoices={characters.data?.location_choices ?? []}
              allCharacters={characters.data?.characters ?? []}
              activeSaveId={activeSaveId}
              runJob={runJob}
              disabled={!charactersMatchActiveSave || !canMutateCharacters}
              mediaGenerationDisabled={!charactersMatchActiveSave}
              showContactNameField={characterTextsEnabled}
              onSave={async (row) => {
                const isDraft = row.__draft_character === true;
                const draftId = typeof row.__draft_id === "string" ? row.__draft_id : "";
                const payload = { ...row };
                delete payload.__draft_character;
                delete payload.__draft_id;
                await applyRows([payload], { autoEnhanceCreatedAgency: isDraft });
                if (isDraft && draftId) {
                  setDraftCharacters((current) => current.filter((draft) => draft.__draft_id !== draftId));
                }
              }}
            />
          </details>
        ))}
        {!rows.length ? <p className="empty">No characters found</p> : null}
      </div>
      <PanelHeader icon={<PanelRight size={18} />} title="Links" />
      <DataViewer value={{ link_targets: characters.data?.link_targets ?? [], location_choices: characters.data?.location_choices ?? [] }} emptyLabel="No links" />
      {exportingCharacter ? (
        <CharacterExportDialog
          character={exportingCharacter}
          canIncludePrivateNotes={canUseAdminControls(currentUser)}
          onClose={() => setExportingCharacter(null)}
        />
      ) : null}
    </aside>
  );
}

function CharacterExportDialog({
  character,
  canIncludePrivateNotes,
  onClose
}: {
  character: { id: string; name: string };
  canIncludePrivateNotes: boolean;
  onClose: () => void;
}) {
  const [includePrivateNotes, setIncludePrivateNotes] = useState(false);
  const titleId = React.useId();
  const exportPath = `/api/character-bundles/export/${encodeURIComponent(character.id)}${canIncludePrivateNotes && includePrivateNotes ? "?include_private_notes=1" : ""}`;
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog" titleId={titleId} onClose={onClose}>
        <header>
          <h2 id={titleId}>Export character bundle?</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <div className="preview-title">
          <strong>{character.name}</strong>
          <span>Character bundle</span>
        </div>
        {canIncludePrivateNotes ? (
          <label className="compact-toggle">
            <input
              type="checkbox"
              checked={includePrivateNotes}
              onChange={(event) => setIncludePrivateNotes(event.target.checked)}
            />
            Include private notes
          </label>
        ) : null}
        <div className="command-row end">
          <button type="button" onClick={onClose}>Cancel</button>
          <button
            type="button"
            className="primary-command compact"
            onClick={() => {
              openDownloadInNewTab(exportPath);
              onClose();
            }}
          >
            <Download size={14} /> Export
          </button>
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function CharacterBundleUpload({
  activeSaveId,
  disabled,
  onImported
}: {
  activeSaveId: string | null;
  disabled: boolean;
  onImported: (model: CharacterRegistryModel) => void;
}) {
  const [pending, setPending] = useState<{
    preview_id: string;
    preview: CharacterBundlePreview;
  } | null>(null);
  const [importName, setImportName] = useState("");
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement | null>(null);
  return (
    <>
      <button
        type="button"
        className="upload-button"
        aria-label="Import character bundle"
        disabled={disabled || !activeSaveId}
        onClick={() => inputRef.current?.click()}
      >
        <Upload size={15} /> Import
      </button>
      <input
        ref={inputRef}
        className="upload-input"
        aria-label="Character bundle file"
        type="file"
        accept=".bragi-character"
        onChange={async (event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (!file || !activeSaveId) return;
          const form = new FormData();
          form.append("file", file);
          form.append("active_save_id", activeSaveId);
          try {
            const result = await api<{
              preview_id: string;
              preview: CharacterBundlePreview;
            }>("/api/character-bundles/preview", {
              method: "POST",
              body: form
            });
            setPending(result);
            setImportName(result.preview.suggested_name);
            setError("");
          } catch (failure) {
            setError(
              failure instanceof Error
                ? failure.message
                : "Character import preview failed"
            );
          }
        }}
      />
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      {pending ? (
        <CharacterBundlePreviewModal
          pending={pending}
          importName={importName}
          setImportName={setImportName}
          activeSaveId={activeSaveId}
          onCancel={() => setPending(null)}
          onImported={(model) => {
            setPending(null);
            onImported(model);
          }}
        />
      ) : null}
    </>
  );
}

function CharacterBundlePreviewModal({
  pending,
  importName,
  setImportName,
  activeSaveId,
  onCancel,
  onImported
}: {
  pending: { preview_id: string; preview: CharacterBundlePreview };
  importName: string;
  setImportName: (value: string) => void;
  activeSaveId: string | null;
  onCancel: () => void;
  onImported: (model: CharacterRegistryModel) => void;
}) {
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const titleId = React.useId();
  const trimmedName = importName.trim();
  const previewRows = [
    ["Aliases", pending.preview.aliases.length ? pending.preview.aliases.join(", ") : "None"],
    ["Role", pending.preview.role || "None"],
    ["Age", pending.preview.age || "None"],
    ["Status", pending.preview.status || "None"],
    ["Voice", pending.preview.voice || "None"],
    ["Texting", pending.preview.texting_style || "None"],
    ["Personality", pending.preview.personality || "None"],
    ["History", pending.preview.history || pending.preview.known_state || "None"],
    ["Appearance", pending.preview.appearance || "None"]
  ];
  const submit = async () => {
    if (!activeSaveId || !trimmedName) return;
    try {
      setSubmitting(true);
      const model = await postJson<CharacterRegistryModel>(
        `/api/character-bundles/import/${pending.preview_id}`,
        {
          active_save_id: activeSaveId,
          name: trimmedName
        }
      );
      onImported(model);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Character import failed");
      setSubmitting(false);
    }
  };
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog"
        titleId={titleId}
        onClose={onCancel}
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <header>
          <h2 id={titleId}>Import character?</h2>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <div className="preview-title">
          <strong>{pending.preview.name}</strong>
          <span>{pending.preview.role || pending.preview.status || "Character"}</span>
        </div>
        <div className="preview-grid">
          <div><span>Name</span><strong>{pending.preview.name}</strong></div>
          <div><span>Media</span><strong>{pending.preview.media_count}</strong></div>
          <div><span>Bundle</span><strong>v{pending.preview.bundle_version}</strong></div>
        </div>
        <div className="kv-list">
          {previewRows.map(([label, value]) => (
            <div className="kv-row" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
            </div>
          ))}
        </div>
        <label className="field-label">
          <span>Import name</span>
          <input
            value={importName}
            aria-label="Import name"
            onChange={(event) => setImportName(event.target.value)}
          />
        </label>
        {pending.preview.name_conflict ? (
          <InlineNotice>A character with this name already exists.</InlineNotice>
        ) : null}
        {pending.preview.warnings.map((warning) => (
          <p className="muted" key={warning}>{warning}</p>
        ))}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onCancel}>Cancel</button>
          <button
            className="primary-command compact"
            disabled={!trimmedName || submitting || !activeSaveId}
          >
            {submitting ? <Loader2 className="spin" size={15} /> : <Upload size={15} />}
            Import
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function CharacterReferenceField({
  row,
  activeSaveId,
  disabled,
  generationDisabled,
  runJob,
  onUpload,
  busy,
  setBusy,
  error,
  setError,
  enhancementBusy
}: {
  row: Record<string, unknown>;
  activeSaveId: string | null;
  disabled: boolean;
  generationDisabled: boolean;
  runJob: RunJob;
  onUpload: (model: CharacterRegistryModel, lockBaseline: string[]) => void;
  busy: "generate" | "upload" | "remove" | null;
  setBusy: (busy: "generate" | "upload" | "remove" | null) => void;
  error: string;
  setError: (error: string) => void;
  enhancementBusy: boolean;
}) {
  const client = useQueryClient();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const characterId = typeof row.character_id === "string" && row.character_id ? row.character_id : null;
  const reference = characterReferenceImage(row.reference_image);
  const hasReference = Boolean(reference);
  const refreshCharacters = (model?: CharacterRegistryModel) => {
    if (model && activeSaveId) client.setQueryData(["characters", activeSaveId], model);
    client.invalidateQueries({ queryKey: ["characters"] });
    client.invalidateQueries({ queryKey: ["runtime"] });
    client.invalidateQueries({ queryKey: ["world"] });
    invalidateScenePresenceQueries(client, activeSaveId);
  };
  const uploadFile = async (file: File) => {
    if (!activeSaveId || !characterId || enhancementBusy) return;
    setBusy("upload");
    setError("");
    const form = new FormData();
    form.append("file", file);
    form.append("save_id", activeSaveId);
    form.append("replace_existing", hasReference ? "true" : "false");
    const lockBaseline = characterLockedFields(row.locked_fields);
    try {
      const job = await api<Job>(
        `/api/characters/${encodeURIComponent(characterId)}/reference-image/upload`,
        {
          method: "POST",
          body: form
        }
      );
      runJob(job, {
        onSucceeded: (result) => {
          const model = result as CharacterRegistryModel;
          onUpload(model, lockBaseline);
          refreshCharacters(model);
        },
        onFailed: setError,
        onFinished: () => setBusy(null)
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not upload reference image");
      setBusy(null);
    }
  };
  return (
    <div className="character-reference-field">
      <div className="character-reference-preview">
        {reference ? (
          <img
            src={mediaAssetThumbnailPath(reference.media_asset_id, activeSaveId)}
            alt={reference.prompt_preview || "Character reference"}
            loading="lazy"
            decoding="async"
          />
        ) : (
          <span>
            <Image size={20} />
          </span>
        )}
      </div>
      <div className="character-reference-controls">
        <div className="command-row">
          <button
            type="button"
            className={hasReference ? undefined : "primary-command compact"}
            disabled={
              generationDisabled
              || (hasReference && disabled)
              || !activeSaveId
              || !characterId
              || busy !== null
              || enhancementBusy
            }
            onClick={async () => {
              if (!activeSaveId || !characterId || enhancementBusy) return;
              setBusy("generate");
              setError("");
              try {
                runJob(
                  await postJson<Job>(
                    `/api/characters/${encodeURIComponent(characterId)}/reference-image/generate`,
                    { save_id: activeSaveId, replace_existing: hasReference }
                  ),
                  { onSucceeded: () => refreshCharacters() }
                );
              } catch (failure) {
                setError(failure instanceof Error ? failure.message : "Could not start reference generation");
              } finally {
                setBusy(null);
              }
            }}
          >
            {busy === "generate" ? <Loader2 className="spin" size={15} /> : <Wand2 size={15} />}
            {hasReference ? "Regenerate" : "Generate"}
          </button>
          <button
            type="button"
            disabled={disabled || !activeSaveId || !characterId || busy !== null || enhancementBusy}
            onClick={() => inputRef.current?.click()}
          >
            {busy === "upload" ? <Loader2 className="spin" size={15} /> : <Upload size={15} />}
            {hasReference ? "Replace" : "Upload"}
          </button>
          {hasReference ? (
            <button
              type="button"
              className="danger-command"
              disabled={disabled || !activeSaveId || !characterId || busy !== null || enhancementBusy}
              onClick={async () => {
                if (!activeSaveId || !characterId || enhancementBusy) return;
                setBusy("remove");
                setError("");
                try {
                  refreshCharacters(await postJson<CharacterRegistryModel>(
                    `/api/characters/${encodeURIComponent(characterId)}/reference-image/remove`,
                    { save_id: activeSaveId }
                  ));
                } catch (failure) {
                  setError(failure instanceof Error ? failure.message : "Could not remove reference image");
                } finally {
                  setBusy(null);
                }
              }}
            >
              {busy === "remove" ? <Loader2 className="spin" size={15} /> : <X size={15} />}
              Remove
            </button>
          ) : null}
        </div>
        {reference ? <p className="muted">{reference.source === "uploaded" ? "Uploaded" : reference.provider}</p> : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
      </div>
      <input
        ref={inputRef}
        className="upload-input"
        aria-label={hasReference ? "Replace character reference image" : "Upload character reference image"}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        disabled={enhancementBusy}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file) void uploadFile(file);
        }}
      />
    </div>
  );
}

function characterReferenceImage(value: unknown): CharacterReferenceImage | null {
  if (!value || typeof value !== "object") return null;
  const row = value as Record<string, unknown>;
  if (typeof row.media_asset_id !== "string" || !row.media_asset_id) return null;
  return {
    media_asset_id: row.media_asset_id,
    mime_type: typeof row.mime_type === "string" ? row.mime_type : "image/png",
    prompt_preview: typeof row.prompt_preview === "string" ? row.prompt_preview : "",
    provider: typeof row.provider === "string" ? row.provider : "",
    model: typeof row.model === "string" ? row.model : "",
    created_at: typeof row.created_at === "string" || row.created_at === null ? row.created_at : null,
    source: typeof row.source === "string" || row.source === null ? row.source : null
  };
}

function characterGeneratedImages(value: unknown): CharacterReferenceImage[] {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => characterReferenceImage(item))
    .filter((item): item is CharacterReferenceImage => item !== null);
}

function CharacterPicturesSection({
  row,
  activeSaveId,
  disabled,
  generationDisabled,
  runJob,
  uploading,
  uploadError,
  onUpload,
  onMediaChanged
}: {
  row: Record<string, unknown>;
  activeSaveId: string | null;
  disabled: boolean;
  generationDisabled: boolean;
  runJob: RunJob;
  uploading: boolean;
  uploadError: string;
  onUpload: (file: File) => Promise<void>;
  onMediaChanged: (model: CharacterRegistryModel) => boolean;
}) {
  const client = useQueryClient();
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [settingReferenceId, setSettingReferenceId] = useState<string | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (disabled) setDialogOpen(false);
  }, [disabled]);
  const characterId = typeof row.character_id === "string" && row.character_id ? row.character_id : null;
  const reference = characterReferenceImage(row.reference_image);
  const generated = characterGeneratedImages(row.generated_images);
  const uploadDisabled = disabled || !activeSaveId || !characterId || uploading;
  const refreshCharacters = (model?: CharacterRegistryModel) => {
    if (model && !onMediaChanged(model)) return;
    if (model && activeSaveId) client.setQueryData(["characters", activeSaveId], model);
    client.invalidateQueries({ queryKey: ["characters", activeSaveId] });
    client.invalidateQueries({ queryKey: runtimeQueryKey(activeSaveId) });
    client.invalidateQueries({ queryKey: ["world", activeSaveId] });
    invalidateScenePresenceQueries(client, activeSaveId);
  };
  const setAsReference = async (image: CharacterReferenceImage) => {
    if (!activeSaveId || !characterId) return;
    setSettingReferenceId(image.media_asset_id);
    setError("");
    try {
      const job = await postJson<Job>(
        `/api/characters/${encodeURIComponent(characterId)}/reference-image/set`,
        {
          save_id: activeSaveId,
          media_asset_id: image.media_asset_id
        }
      );
      runJob(job, {
        applyResult: false,
        onSucceeded: (result) => {
          refreshCharacters(isCharacterRegistryModel(result) ? result : undefined);
        }
      });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not set reference image");
    } finally {
      setSettingReferenceId(null);
    }
  };
  return (
    <section className="character-pictures-section">
      <div className="section-minihead">
        <strong>Pictures</strong>
        <div className="command-row">
          <button
            type="button"
            disabled={uploadDisabled}
            onClick={() => inputRef.current?.click()}
          >
            {uploading ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <Upload size={15} aria-hidden="true" />}
            {uploading ? "Uploading image…" : "Upload image"}
          </button>
          {reference ? (
            <button
              type="button"
              className="primary-command compact"
              disabled={generationDisabled || !activeSaveId || uploading}
              onClick={() => setDialogOpen(true)}
            >
              <Image size={15} /> Generate image of this character
            </button>
          ) : null}
        </div>
      </div>
      <input
        ref={inputRef}
        className="upload-input"
        aria-label="Upload character image"
        type="file"
        accept="image/png,image/jpeg,image/webp"
        disabled={uploadDisabled}
        onChange={(event) => {
          const file = event.target.files?.[0];
          event.target.value = "";
          if (file && !uploadDisabled) {
            setError("");
            void onUpload(file);
          }
        }}
      />
      <div className="character-picture-grid">
        {generated.map((image) => {
          const title = image.prompt_preview || "Character image";
          const isCurrentReference = reference?.media_asset_id === image.media_asset_id;
          const isSetting = settingReferenceId === image.media_asset_id;
          return (
            <div
              className="character-picture-thumb"
              key={image.media_asset_id}
              title={title}
            >
              <img
                src={mediaAssetThumbnailPath(image.media_asset_id, activeSaveId)}
                alt={title}
                loading="lazy"
                decoding="async"
              />
              {!isCurrentReference ? (
                <button
                  type="button"
                  className="character-picture-reference-button"
                  title="Make reference"
                  aria-label={`Make reference: ${title}`}
                  disabled={disabled || !activeSaveId || !characterId || settingReferenceId !== null || uploading}
                  onClick={() => void setAsReference(image)}
                >
                  {isSetting ? <Loader2 className="spin" size={14} aria-hidden="true" /> : <Check size={14} aria-hidden="true" />}
                </button>
              ) : null}
            </div>
          );
        })}
      </div>
      {error ? <InlineNotice>{error}</InlineNotice> : null}
      {uploadError ? <InlineNotice>{uploadError}</InlineNotice> : null}
      {!generated.length ? <p className="muted">No pictures yet.</p> : null}
      {dialogOpen && reference && !disabled ? (
        <CharacterRegistryImageDialog
          row={row}
          reference={reference}
          activeSaveId={activeSaveId}
          disabled={disabled}
          runJob={runJob}
          onClose={() => setDialogOpen(false)}
        />
      ) : null}
    </section>
  );
}

function CharacterRegistryImageDialog({
  row,
  reference,
  activeSaveId,
  disabled,
  runJob,
  onClose
}: {
  row: Record<string, unknown>;
  reference: CharacterReferenceImage;
  activeSaveId: string | null;
  disabled: boolean;
  runJob: RunJob;
  onClose: () => void;
}) {
  const client = useQueryClient();
  const [instructions, setInstructions] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const titleId = React.useId();
  const characterId = typeof row.character_id === "string" ? row.character_id : "";
  const refresh = () => {
    client.invalidateQueries({ queryKey: ["characters"] });
    if (activeSaveId) client.invalidateQueries({ queryKey: ["characters", activeSaveId] });
    client.invalidateQueries({ queryKey: ["runtime"] });
    client.invalidateQueries({ queryKey: ["media"] });
    invalidateScenePresenceQueries(client, activeSaveId);
  };
  const submit = async () => {
    if (disabled || !activeSaveId || !characterId) return;
    setSubmitting(true);
    setError("");
    try {
      const job = await postJson<Job>(
        `/api/characters/${encodeURIComponent(characterId)}/image/generate`,
        {
          save_id: activeSaveId,
          instructions
        }
      );
      refresh();
      onClose();
      runJob(job, { onSucceeded: refresh });
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not start character image");
    } finally {
      setSubmitting(false);
    }
  };
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog character-image-dialog" titleId={titleId} onClose={onClose}>
        <header>
          <h2 id={titleId}>Generate image of {String(row.name ?? "character")}</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <div className="registry-image-reference">
          <img
            src={mediaAssetThumbnailPath(reference.media_asset_id, activeSaveId)}
            alt={reference.prompt_preview || "Character reference"}
            loading="lazy"
            decoding="async"
          />
          <span>{reference.source === "uploaded" ? "Uploaded reference" : reference.provider}</span>
        </div>
        <label className="field-label">
          <span>Instructions</span>
          <textarea
            disabled={disabled}
            value={instructions}
            onChange={(event) => setInstructions(event.target.value)}
            placeholder="Pose, outfit, mood, setting"
          />
        </label>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onClose}>Cancel</button>
          <button type="button" className="primary-command compact" disabled={disabled || submitting} onClick={submit}>
            {submitting ? <Loader2 className="spin" size={15} /> : <Image size={15} />}
            Generate
          </button>
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function CharacterEnhanceableField({
  label,
  fieldName,
  value,
  multiline = true,
  placeholder,
  disabled,
  enhanceDisabled,
  busy,
  onChange,
  onEnhance
}: {
  label: string;
  fieldName: CharacterEnhanceField;
  value: string;
  multiline?: boolean;
  placeholder?: string;
  disabled: boolean;
  enhanceDisabled: boolean;
  busy: boolean;
  onChange: (value: string) => void;
  onEnhance: () => void;
}) {
  const controlId = React.useId();
  return (
    <div className="field-label enhanced-field">
      <div className="field-label-action-row">
        <label htmlFor={controlId}>{label}</label>
        <button
          type="button"
          className={touchActionClassName("character-auto-enhance-button")}
          title={`Auto-enhance ${label}`}
          aria-label={`Auto-enhance ${label}`}
          disabled={enhanceDisabled}
          onClick={onEnhance}
        >
          <TouchActionContents
            icon={busy ? <Loader2 className="spin" size={14} aria-hidden="true" /> : <Wand2 size={14} aria-hidden="true" />}
            label="Enhance"
          />
        </button>
      </div>
      {multiline ? (
        <textarea
          id={controlId}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder ?? label}
          data-character-enhance-field={fieldName}
        />
      ) : (
        <input
          id={controlId}
          value={value}
          disabled={disabled}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder ?? label}
          data-character-enhance-field={fieldName}
        />
      )}
    </div>
  );
}

function CharacterEditor({
  character,
  linkTargets,
  locationChoices,
  allCharacters,
  activeSaveId,
  runJob,
  disabled,
  mediaGenerationDisabled,
  showContactNameField = false,
  onSave
}: {
  character: Record<string, unknown>;
  linkTargets: CharacterKnowledgeTarget[];
  locationChoices: unknown[];
  allCharacters: Record<string, unknown>[];
  activeSaveId: string | null;
  runJob: RunJob;
  disabled: boolean;
  mediaGenerationDisabled: boolean;
  showContactNameField?: boolean;
  onSave: (row: Record<string, unknown>) => Promise<void>;
}) {
  const client = useQueryClient();
  const [row, setRow] = useState<Record<string, unknown>>(character);
  const [savedRow, setSavedRow] = useState<Record<string, unknown>>(character);
  const [activeTab, setActiveTab] = useState<CharacterEditorTab>("profile");
  const [deleteText, setDeleteText] = useState("");
  const [error, setError] = useState("");
  const [referenceError, setReferenceError] = useState("");
  const [referenceBusy, setReferenceBusy] = useState<"generate" | "upload" | "remove" | null>(null);
  const [galleryUploading, setGalleryUploading] = useState(false);
  const [galleryUploadError, setGalleryUploadError] = useState("");
  const [notice, setNotice] = useState("");
  const [enhancingField, setEnhancingField] = useState<CharacterEnhanceField | null>(null);
  const [discardConfirmOpen, setDiscardConfirmOpen] = useState(false);
  const selectedCharacterId = typeof character.character_id === "string" ? character.character_id : "";
  const currentSnapshot = useMemo(() => stableEditorSnapshot(row), [row]);
  const savedSnapshot = useMemo(() => stableEditorSnapshot(savedRow), [savedRow]);
  const hasDraftChanges = currentSnapshot !== savedSnapshot;
  useEffect(() => {
    setRow((current) => current.character_id === selectedCharacterId
      && hasDraftChanges ? current : character);
    setSavedRow(character);
    setDiscardConfirmOpen(false);
  }, [character, selectedCharacterId]);
  useEffect(() => setNotice(""), [selectedCharacterId]);
  const isDraftCharacter = row.__draft_character === true;
  const hasUnsavedChanges = isDraftCharacter || hasDraftChanges;
  const update = (key: string, value: unknown) => {
    setNotice("");
    setRow((current) => ({ ...current, [key]: value }));
  };
  const targetOptions = linkTargets.filter(isCharacterKnowledgeTarget);
  const locationOptions = locationChoices as [string, string][];
  const characterId = typeof row.character_id === "string" ? row.character_id : "";
  const nameBlank = !String(row.name ?? "").trim();
  const nameError = nameBlank ? "Character name must not be blank" : "";
  const saveDisabled = disabled || nameBlank || referenceBusy === "upload" || (!isDraftCharacter && !hasDraftChanges);
  const enhanceDisabled = disabled || nameBlank || referenceBusy !== null || !activeSaveId || !characterId || enhancingField !== null;
  const lockedFields = new Set(characterLockedFields(row.locked_fields));
  const mergeCharacterMedia = (model: CharacterRegistryModel) => {
    const updated = model.characters?.find((candidate) => candidate.character_id === characterId);
    if (model.active_save_id !== activeSaveId || !updated) return false;
    const mediaPatch = { generated_images: updated.generated_images, reference_image: updated.reference_image };
    setRow((current) => ({ ...current, ...mediaPatch }));
    setSavedRow((current) => ({ ...current, ...mediaPatch }));
    return true;
  };
  const uploadGalleryImage = async (file: File) => {
    if (disabled || !activeSaveId || !characterId || galleryUploading) return;
    const saveId = activeSaveId;
    setGalleryUploading(true);
    setGalleryUploadError("");
    const form = new FormData();
    form.append("file", file);
    form.append("save_id", saveId);
    try {
      const model = await api<CharacterRegistryModel>(
        `/api/characters/${encodeURIComponent(characterId)}/image/upload`,
        { method: "POST", body: form }
      );
      if (!mergeCharacterMedia(model)) {
        throw new Error("Could not refresh character pictures");
      }
      await client.cancelQueries({ queryKey: ["characters", saveId], exact: true });
      client.setQueryData(["characters", saveId], model);
      client.invalidateQueries({ queryKey: runtimeQueryKey(saveId) });
      client.invalidateQueries({ queryKey: ["media", saveId] });
    } catch (failure) {
      setGalleryUploadError(failure instanceof Error ? failure.message : "Could not upload character image");
    } finally {
      setGalleryUploading(false);
    }
  };
  const updateLockedField = (field: string, locked: boolean) => {
    const next = new Set(characterLockedFields(row.locked_fields));
    if (locked) {
      next.add(field);
    } else {
      next.delete(field);
    }
    update("locked_fields", CHARACTER_LOCK_FIELDS.map(([id]) => id).filter((id) => next.has(id)));
  };
  const save = async (next: Record<string, unknown>) => {
    if (referenceBusy === "upload") {
      setError("Wait for reference image analysis to finish before saving");
      return;
    }
    if (!String(next.name ?? "").trim() && !next.archived) {
      return;
    }
    const saved = sanitizeCharacterLinkRow(next, targetOptions);
    try {
      await onSave(saved);
      setRow(saved);
      setSavedRow(saved);
      setError("");
      setNotice("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not save character");
      setNotice("");
    }
  };
  const enhanceField = async (fieldName: CharacterEnhanceField) => {
    if (!activeSaveId || !characterId) {
      setError("Save the character before auto-enhancing");
      setNotice("");
      return;
    }
    if (!CHARACTER_AUTO_ENHANCE_FIELD_SET.has(fieldName)) {
      setError("This character field cannot be auto-enhanced");
      setNotice("");
      return;
    }
    try {
      setEnhancingField(fieldName);
      setNotice("");
      const result = await postJson<CharacterFieldEnhanceResult>(
        `/api/characters/${encodeURIComponent(characterId)}/enhance-field`,
        {
          active_save_id: activeSaveId,
          field_name: fieldName,
          character: sanitizeCharacterLinkRow(row, targetOptions)
        }
      );
      client.setQueryData(["characters", activeSaveId], result.model);
      client.invalidateQueries({ queryKey: ["characters"] });
      client.invalidateQueries({ queryKey: ["world"] });
      client.invalidateQueries({ queryKey: ["runtime"] });
      const nextRow = result.model.characters?.find((candidate) => candidate.character_id === characterId);
      if (nextRow) {
        setRow(nextRow);
        setSavedRow(nextRow);
      }
      setError("");
      setNotice(result.notice ?? "");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not auto-enhance character field");
      setNotice("");
    } finally {
      setEnhancingField(null);
    }
  };
  const enhanceButton = (fieldName: CharacterEnhanceField) => {
    const label = CHARACTER_AUTO_ENHANCE_LABELS[fieldName];
    return (
      <button
        type="button"
        className={touchActionClassName("character-auto-enhance-button")}
        title={`Auto-enhance ${label}`}
        aria-label={`Auto-enhance ${label}`}
        disabled={enhanceDisabled}
        onClick={() => enhanceField(fieldName)}
      >
        <TouchActionContents
          icon={enhancingField === fieldName ? <Loader2 className="spin" size={14} aria-hidden="true" /> : <Wand2 size={14} aria-hidden="true" />}
          label="Enhance"
        />
      </button>
    );
  };
  const applyKnowledgeActions = async (actions: Record<string, unknown>[]) => {
    if (!activeSaveId || !characterId) {
      throw new Error("Save the character before editing knowledge");
    }
    const result = await postJson<CharacterKnowledgeApplyResult>(
      `/api/characters/${encodeURIComponent(characterId)}/knowledge/apply`,
      { active_save_id: activeSaveId, actions }
    );
    client.setQueryData(["characters", activeSaveId], result.model);
    client.invalidateQueries({ queryKey: ["characters"] });
    client.invalidateQueries({ queryKey: ["world"] });
    client.invalidateQueries({ queryKey: ["runtime"] });
    const nextRow = result.model.characters?.find((candidate) => candidate.character_id === characterId);
    if (nextRow) {
      setRow(nextRow);
      setSavedRow(nextRow);
    }
  };
  return (
    <div className="character-form">
      <SegmentedTabs
        className="character-dossier-tabs segmented character-editor-tabs"
        label="Character editor sections"
        value={activeTab}
        onChange={setActiveTab}
        options={[
          { value: "profile", label: "Profile" },
          { value: "agency", label: "Agency" },
          { value: "knowledge", label: "Knowledge" },
          { value: "pictures", label: "Pictures" },
          { value: "locks", label: "Locks" }
        ]}
      />
      <EditorDirtyStatus
        dirty={hasUnsavedChanges}
        canDiscard={hasDraftChanges}
        onDiscard={() => setDiscardConfirmOpen(true)}
      />
      {activeTab === "profile" ? (
        <>
          <label className="field-label">
            <span>Name</span>
            <input
              value={String(row.name ?? "")}
              disabled={disabled}
              onChange={(event) => update("name", event.target.value)}
              placeholder="Name"
              aria-invalid={nameBlank ? "true" : undefined}
            />
          </label>
          <CharacterReferenceField
            row={row}
            activeSaveId={activeSaveId}
            disabled={disabled || galleryUploading}
            generationDisabled={mediaGenerationDisabled || galleryUploading}
            runJob={runJob}
            busy={referenceBusy}
            setBusy={setReferenceBusy}
            error={referenceError}
            setError={setReferenceError}
            enhancementBusy={enhancingField !== null}
            onUpload={(model, lockBaseline) => {
              const updated = model.characters!.find(
                (candidate) => candidate.character_id === characterId
              )!;
              const { appearance, visual_notes, locked_fields, reference_image } = updated;
              const savedPatch = {
                appearance, visual_notes, locked_fields, reference_image
              };
              setRow((current) => ({
                ...current,
                ...savedPatch,
                locked_fields: mergeReferenceUploadLocks(
                  characterLockedFields(locked_fields),
                  lockBaseline,
                  characterLockedFields(current.locked_fields)
                )
              }));
              setSavedRow((current) => ({ ...current, ...savedPatch }));
            }}
          />
          <label className="field-label">
            <span>Aliases</span>
            <input value={String(row.aliases_text ?? "")} disabled={disabled} onChange={(event) => update("aliases_text", event.target.value)} />
          </label>
          {showContactNameField ? (
            <>
              <label className="field-label">
                <span>Contact Name</span>
                <input
                  value={String(row.contact_name ?? "")}
                  disabled={disabled}
                  onChange={(event) => update("contact_name", event.target.value)}
                />
              </label>
              <CharacterEnhanceableField
                label="Texting Style"
                fieldName="texting_style"
                value={String(row.texting_style ?? "")}
                disabled={disabled}
                enhanceDisabled={enhanceDisabled}
                busy={enhancingField === "texting_style"}
                onChange={(value) => update("texting_style", value)}
                onEnhance={() => enhanceField("texting_style")}
              />
            </>
          ) : null}
          <label className="field-label">
            <span>Role</span>
            <input value={String(row.role ?? "")} disabled={disabled} onChange={(event) => update("role", event.target.value)} />
          </label>
          <label className="field-label">
            <span>Age</span>
            <input value={String(row.age ?? "")} disabled={disabled} onChange={(event) => update("age", event.target.value)} />
          </label>
          <CharacterEnhanceableField
            label="History"
            fieldName="known_state"
            value={String(row.known_state ?? "")}
            disabled={disabled}
            enhanceDisabled={enhanceDisabled}
            busy={enhancingField === "known_state"}
            onChange={(value) => update("known_state", value)}
            onEnhance={() => enhanceField("known_state")}
          />
          <CharacterEnhanceableField
            label="Appearance"
            fieldName="appearance"
            value={String(row.appearance ?? "")}
            disabled={disabled}
            enhanceDisabled={enhanceDisabled}
            busy={enhancingField === "appearance"}
            onChange={(value) => update("appearance", value)}
            onEnhance={() => enhanceField("appearance")}
          />
          <CharacterEnhanceableField
            label="Visual notes"
            fieldName="visual_notes"
            value={String(row.visual_notes ?? "")}
            disabled={disabled}
            enhanceDisabled={enhanceDisabled}
            busy={enhancingField === "visual_notes"}
            onChange={(value) => update("visual_notes", value)}
            onEnhance={() => enhanceField("visual_notes")}
          />
          <label className="field-label">
            <span>Current clothing</span>
            <input value={String(row.current_clothing ?? "")} disabled={disabled} onChange={(event) => update("current_clothing", event.target.value)} />
          </label>
          <CharacterEnhanceableField
            label="Personality"
            fieldName="personality"
            value={String(row.personality ?? "")}
            disabled={disabled}
            enhanceDisabled={enhanceDisabled}
            busy={enhancingField === "personality"}
            onChange={(value) => update("personality", value)}
            onEnhance={() => enhanceField("personality")}
          />
          <CharacterEnhanceableField
            label="Voice"
            fieldName="voice"
            value={String(row.voice ?? "")}
            disabled={disabled}
            enhanceDisabled={enhanceDisabled}
            busy={enhancingField === "voice"}
            onChange={(value) => update("voice", value)}
            onEnhance={() => enhanceField("voice")}
          />
          <CharacterEnhanceableField
            label="Status"
            fieldName="status"
            value={String(row.status ?? "")}
            multiline={false}
            disabled={disabled}
            enhanceDisabled={enhanceDisabled}
            busy={enhancingField === "status"}
            onChange={(value) => update("status", value)}
            onEnhance={() => enhanceField("status")}
          />
          <label className="field-label">
            <span>Location</span>
            <select value={String(row.location_id ?? "")} disabled={disabled} onChange={(event) => update("location_id", event.target.value || null)}>
              <option value="">No location</option>
              {locationOptions.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
            </select>
          </label>
          <div className="checkbox-grid">
            {[
              ["met", "Met"],
              ["present", "Present"],
              ["is_player_character", "Player Character"],
              ["protected_from_maintenance", "Protected From Maintenance"]
            ].map(([key, label]) => {
              const checked = key === "present" && Boolean(row.is_player_character) ? true : Boolean(row[key]);
              const inputDisabled = disabled || (key === "present" && Boolean(row.is_player_character));
              return (
                <label key={key}>
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={inputDisabled}
                    onChange={(event) => {
                      if (key === "is_player_character" && event.target.checked) {
                        setRow((current) => ({
                          ...current,
                          is_player_character: true,
                          present: true,
                          protected_from_maintenance: true
                        }));
                        return;
                      }
                      update(key, event.target.checked);
                    }}
                  />
                  {label}
                </label>
              );
            })}
          </div>
          <RelationshipEditor
            value={String(row.relationships_json ?? "{}")}
            disabled={disabled}
            onChange={(value) => update("relationships_json", value)}
            enhanceAction={enhanceButton("relationships")}
          />
          <label className="field-label">
            <span>Merge target</span>
            <select value={String(row.merge_into_character_id ?? "")} disabled={disabled} onChange={(event) => update("merge_into_character_id", event.target.value || null)}>
              <option value="">No merge</option>
              {allCharacters.filter((candidate) => candidate.character_id !== row.character_id).map((candidate) => (
                <option key={String(candidate.character_id)} value={String(candidate.character_id)}>{String(candidate.name ?? "Unnamed")}</option>
              ))}
            </select>
          </label>
          <div className="inline-tool-form">
            <input value={deleteText} disabled={disabled} onChange={(event) => setDeleteText(event.target.value)} placeholder="Type DELETE to archive" aria-label="Archive confirmation" />
            <button type="button" title="Archive character" disabled={disabled || deleteText !== "DELETE"} onClick={() => save({ ...row, archived: true })}><Trash2 size={15} /></button>
          </div>
          {nameError || error ? <InlineNotice>{nameError || error}</InlineNotice> : null}
          {!nameError && !error && notice ? <InlineNotice polite>{notice}</InlineNotice> : null}
          <div className="command-row end">
            <button type="button" className="primary-command compact" disabled={saveDisabled} onClick={() => save(row)}><Save size={15} /> Save character</button>
          </div>
        </>
      ) : null}
      {activeTab === "agency" ? (
        <>
          <div className="character-agency-grid">
            {([
              ["goals", "Goals"],
              ["motivations", "Motivations"],
              ["current_intent", "Current Intent"],
              ["boundaries", "Boundaries"],
              ["attitude_toward_player", "Attitude Toward Player"],
              ["cooperation_conditions", "Cooperation Conditions"]
            ] as const).map(([key, label]) => (
              <CharacterEnhanceableField
                key={key}
                label={label}
                fieldName={key}
                value={String(row[key] ?? "")}
                disabled={disabled}
                enhanceDisabled={enhanceDisabled}
                busy={enhancingField === key}
                onChange={(value) => update(key, value)}
                onEnhance={() => enhanceField(key)}
              />
            ))}
          </div>
          {error ? <InlineNotice>{error}</InlineNotice> : null}
          {!error && notice ? <InlineNotice polite>{notice}</InlineNotice> : null}
          <div className="command-row end">
            <button type="button" className="primary-command compact" disabled={saveDisabled} onClick={() => save(row)}><Save size={15} /> Save character</button>
          </div>
        </>
      ) : null}
      {activeTab === "knowledge" ? (
        <CharacterKnowledgeDossier
          row={row}
          targets={targetOptions}
          disabled={disabled}
          onApply={applyKnowledgeActions}
        />
      ) : null}
      {activeTab === "pictures" ? (
        <CharacterPicturesSection
          row={row}
          activeSaveId={activeSaveId}
          disabled={disabled || referenceBusy !== null}
          generationDisabled={mediaGenerationDisabled}
          runJob={runJob}
          uploading={galleryUploading}
          uploadError={galleryUploadError}
          onUpload={uploadGalleryImage}
          onMediaChanged={mergeCharacterMedia}
        />
      ) : null}
      {activeTab === "locks" ? (
        <>
          <div className="field-label character-locks">
            <span>Locked Fields</span>
            <div className="checkbox-grid character-lock-grid">
              {CHARACTER_LOCK_FIELDS.map(([field, label]) => (
                <label key={field}>
                  <input
                    type="checkbox"
                    checked={lockedFields.has(field)}
                    disabled={disabled}
                    onChange={(event) => updateLockedField(field, event.target.checked)}
                  />
                  {`Lock ${label}`}
                </label>
              ))}
            </div>
          </div>
          {error ? <InlineNotice>{error}</InlineNotice> : null}
          <div className="command-row end">
            <button type="button" className="primary-command compact" disabled={saveDisabled} onClick={() => save(row)}><Save size={15} /> Save character</button>
          </div>
        </>
      ) : null}
      {discardConfirmOpen ? (
        <ConfirmModal
          title="Discard changes?"
          body="Unsaved character edits will be lost."
          confirmLabel="Discard"
          destructive
          onCancel={() => setDiscardConfirmOpen(false)}
          onConfirm={async () => {
            setRow(savedRow);
            setDeleteText("");
            setError("");
            setNotice("");
            setDiscardConfirmOpen(false);
          }}
        />
      ) : null}
    </div>
  );
}

type RelationshipDraft = {
  id: string;
  name: string;
  note: string;
  complexValue?: unknown;
};

function RelationshipEditor({
  value,
  disabled,
  onChange,
  enhanceAction = null
}: {
  value: string;
  disabled: boolean;
  onChange: (value: string) => void;
  enhanceAction?: React.ReactNode;
}) {
  const [drafts, setDrafts] = useState<RelationshipDraft[]>(() => relationshipDrafts(value));
  useEffect(() => setDrafts(relationshipDrafts(value)), [value]);
  const updateDrafts = (next: RelationshipDraft[]) => {
    setDrafts(next);
    onChange(relationshipsJson(next));
  };
  return (
    <section className="relationship-editor">
      <div className="section-minihead">
        <strong>Relationships</strong>
        <div className="minihead-actions">
          {enhanceAction}
          <button
            type="button"
            disabled={disabled}
            onClick={() => updateDrafts([...drafts, { id: `relationship-${Date.now()}`, name: "", note: "" }])}
          >
            <Plus size={14} /> Add
          </button>
        </div>
      </div>
      <div className="relationship-list">
        {drafts.map((draft) => (
          <div className="relationship-row" key={draft.id}>
            <label className="field-label">
              <span>Relationship name{draft.name ? ` ${draft.name}` : ""}</span>
              <input
                aria-label={`Relationship name ${draft.name || "new"}`}
                value={draft.name}
                disabled={disabled || draft.complexValue !== undefined}
                onChange={(event) => updateDrafts(drafts.map((item) => item.id === draft.id ? { ...item, name: event.target.value } : item))}
              />
            </label>
            <label className="field-label relationship-note">
              <span>Relationship note{draft.name ? ` ${draft.name}` : ""}</span>
              {draft.complexValue === undefined ? (
                <textarea
                  aria-label={`Relationship note ${draft.name || "new"}`}
                  value={draft.note}
                  disabled={disabled}
                  onChange={(event) => updateDrafts(drafts.map((item) => item.id === draft.id ? { ...item, note: event.target.value } : item))}
                />
              ) : (
                <p className="muted complex-relationship">Complex relationship data preserved</p>
              )}
            </label>
            <button
              type="button"
              className={touchActionClassName("destructive-action")}
              title="Remove relationship"
              aria-label={`Remove relationship ${draft.name || "new"}`}
              disabled={disabled}
              onClick={() => updateDrafts(drafts.filter((item) => item.id !== draft.id))}
            >
              <TouchActionContents icon={<Trash2 size={14} />} label="Remove" />
            </button>
          </div>
        ))}
        {!drafts.length ? <p className="empty">No relationships yet</p> : null}
      </div>
    </section>
  );
}

function relationshipDrafts(value: string): RelationshipDraft[] {
  const parsed = safeJsonObject(value);
  return Object.entries(parsed).map(([name, note], index) => ({
    id: `relationship-${index}-${name}`,
    name,
    note: typeof note === "string" ? note : "",
    complexValue: typeof note === "string" ? undefined : note
  }));
}

function relationshipsJson(drafts: RelationshipDraft[]): string {
  const relationships: Record<string, unknown> = {};
  drafts.forEach((draft) => {
    const name = draft.name.trim();
    if (!name) return;
    relationships[name] = draft.complexValue === undefined ? draft.note.trim() : draft.complexValue;
  });
  return JSON.stringify(relationships);
}

function safeJsonObject(value: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(value || "{}");
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed as Record<string, unknown> : {};
  } catch {
    return {};
  }
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

function CharacterKnowledgeDossier({
  row,
  targets,
  disabled,
  onApply
}: {
  row: Record<string, unknown>;
  targets: CharacterKnowledgeTarget[];
  disabled: boolean;
  onApply: (actions: Record<string, unknown>[]) => Promise<void>;
}) {
  const [search, setSearch] = useState("");
  const [addMode, setAddMode] = useState<"memory" | "fact" | null>(null);
  const [editingTargetKey, setEditingTargetKey] = useState("");
  const [busyKey, setBusyKey] = useState("");
  const [error, setError] = useState("");
  const linkedIds = characterKnowledgeIds(row);
  const query = search.trim().toLowerCase();
  const filteredTargets = targets.filter((target) => !query || knowledgeSearchText(target).includes(query));
  const linkedTargets = filteredTargets.filter((target) => characterHasKnowledgeTarget(linkedIds, target));
  const availableTargets = filteredTargets.filter((target) => !characterHasKnowledgeTarget(linkedIds, target));
  const runActions = async (actions: Record<string, unknown>[], key: string) => {
    try {
      setBusyKey(key);
      setError("");
      await onApply(actions);
      setAddMode(null);
      setEditingTargetKey("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not update knowledge");
    } finally {
      setBusyKey("");
    }
  };
  return (
    <section className="knowledge-dossier">
      <div className="knowledge-toolbar">
        <div className="world-search knowledge-search">
          <Search size={15} />
          <input
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Find memories, facts, summaries..."
            aria-label="Search character knowledge"
          />
          <span>{linkedTargets.length}/{filteredTargets.length}</span>
        </div>
        <div className="command-row">
          <button type="button" disabled={disabled} onClick={() => setAddMode(addMode === "memory" ? null : "memory")}>
            <Plus size={15} /> Add memory
          </button>
          <button type="button" disabled={disabled} onClick={() => setAddMode(addMode === "fact" ? null : "fact")}>
            <Plus size={15} /> Add fact
          </button>
        </div>
      </div>
      {addMode === "memory" ? (
        <MemoryKnowledgeForm
          disabled={disabled || Boolean(busyKey)}
          onCancel={() => setAddMode(null)}
          onSave={(payload) => runActions([{ action: "create_memory", ...payload }], "create_memory")}
        />
      ) : null}
      {addMode === "fact" ? (
        <WorldFactKnowledgeForm
          disabled={disabled || Boolean(busyKey)}
          onCancel={() => setAddMode(null)}
          onSave={(payload) => runActions([{ action: "create_world_state", ...payload }], "create_world_state")}
        />
      ) : null}
      <KnowledgeSection title="Linked knowledge" count={linkedTargets.length}>
        {linkedTargets.map((target) => (
          <KnowledgeTargetCard
            key={knowledgeTargetKey(target)}
            target={target}
            linked
            disabled={disabled || Boolean(busyKey)}
            editing={editingTargetKey === knowledgeTargetKey(target)}
            onToggle={() => runActions([
              {
                action: "unlink",
                target_type: normalizedKnowledgeTargetType(target.target_type),
                target_id: target.target_id
              }
            ], `unlink:${knowledgeTargetKey(target)}`)}
            onEdit={() => setEditingTargetKey(editingTargetKey === knowledgeTargetKey(target) ? "" : knowledgeTargetKey(target))}
            onCancelEdit={() => setEditingTargetKey("")}
            onSaveEdit={(payload) => runActions([
              {
                action: target.target_type === "memory" ? "update_memory" : "update_world_state",
                ...payload
              }
            ], `edit:${knowledgeTargetKey(target)}`)}
          />
        ))}
        {!linkedTargets.length ? <p className="empty">No linked knowledge matches</p> : null}
      </KnowledgeSection>
      <KnowledgeSection title="Available evidence" count={availableTargets.length}>
        {availableTargets.map((target) => (
          <KnowledgeTargetCard
            key={knowledgeTargetKey(target)}
            target={target}
            linked={false}
            disabled={disabled || Boolean(busyKey)}
            editing={false}
            onToggle={() => runActions([
              {
                action: "link",
                target_type: normalizedKnowledgeTargetType(target.target_type),
                target_id: target.target_id
              }
            ], `link:${knowledgeTargetKey(target)}`)}
            onEdit={() => undefined}
            onCancelEdit={() => undefined}
            onSaveEdit={() => Promise.resolve()}
          />
        ))}
        {!availableTargets.length ? <p className="empty">No available evidence matches</p> : null}
      </KnowledgeSection>
      {error ? <InlineNotice>{error}</InlineNotice> : null}
    </section>
  );
}

function KnowledgeSection({ title, count, children }: { title: string; count: number; children: React.ReactNode }) {
  return (
    <section className="knowledge-section">
      <div className="section-minihead">
        <strong>{title}</strong>
        <span>{count}</span>
      </div>
      <div className="knowledge-card-list">
        {children}
      </div>
    </section>
  );
}

function KnowledgeTargetCard({
  target,
  linked,
  disabled,
  editing,
  onToggle,
  onEdit,
  onCancelEdit,
  onSaveEdit
}: {
  target: CharacterKnowledgeTarget;
  linked: boolean;
  disabled: boolean;
  editing: boolean;
  onToggle: () => void;
  onEdit: () => void;
  onCancelEdit: () => void;
  onSaveEdit: (payload: Record<string, unknown>) => Promise<void>;
}) {
  const noun = knowledgeTargetNoun(target);
  const title = knowledgeTargetTitle(target);
  const editable = editableKnowledgeTarget(target);
  return (
    <article className={`knowledge-card ${linked ? "linked" : ""}`}>
      <header>
        <div>
          <span className="knowledge-kind">{knowledgeTargetLabel(target)}</span>
          <strong>{title}</strong>
        </div>
        <div className="knowledge-card-actions">
          {linked && editable ? (
            <button type="button" className={touchActionClassName()} title={`Edit ${title}`} aria-label={`Edit ${noun} ${title}`} disabled={disabled} onClick={onEdit}>
              <TouchActionContents icon={<Edit3 size={14} />} label="Edit" />
            </button>
          ) : null}
          <button
            type="button"
            className={touchActionClassName()}
            title={linked ? `Unlink ${title}` : `Link ${title}`}
            aria-label={`${linked ? "Unlink" : "Link"} ${noun} ${title}`}
            disabled={disabled}
            onClick={onToggle}
          >
            <TouchActionContents icon={linked ? <X size={14} /> : <Check size={14} />} label={linked ? "Unlink" : "Link"} />
          </button>
        </div>
      </header>
      <p>{knowledgeTargetBody(target)}</p>
      <div className="knowledge-pills">
        {knowledgeTargetPills(target).map((pill) => <span key={pill}>{pill}</span>)}
      </div>
      {editing && target.target_type === "memory" ? (
        <MemoryKnowledgeForm
          target={target}
          disabled={disabled}
          onCancel={onCancelEdit}
          onSave={(payload) => onSaveEdit({ memory_id: target.target_id, ...payload })}
        />
      ) : null}
      {editing && normalizedKnowledgeTargetType(target.target_type) === "world_state" ? (
        <WorldFactKnowledgeForm
          target={target}
          disabled={disabled}
          onCancel={onCancelEdit}
          onSave={(payload) => onSaveEdit({ state_id: target.target_id, ...payload })}
        />
      ) : null}
    </article>
  );
}

function MemoryKnowledgeForm({
  target,
  disabled,
  onCancel,
  onSave
}: {
  target?: CharacterKnowledgeTarget;
  disabled: boolean;
  onCancel: () => void;
  onSave: (payload: { body: string; tags: string[]; importance: number }) => void;
}) {
  const [body, setBody] = useState(target?.body ?? "");
  const [tags, setTags] = useState((target?.tags ?? []).join(", "));
  const [importance, setImportance] = useState(String(typeof target?.importance === "number" ? target.importance : 0.7));
  const trimmedBody = body.trim();
  return (
    <div className="knowledge-form">
      <label className="field-label">
        <span>Memory body</span>
        <textarea value={body} disabled={disabled} onChange={(event) => setBody(event.target.value)} />
      </label>
      <label className="field-label">
        <span>Tags</span>
        <input value={tags} disabled={disabled} onChange={(event) => setTags(event.target.value)} placeholder="comma-separated" />
      </label>
      <label className="field-label">
        <span>Importance</span>
        <input type="number" min={0} max={1} step={0.01} value={importance} disabled={disabled} onChange={(event) => setImportance(event.target.value)} />
      </label>
      <div className="command-row end">
        <button type="button" disabled={disabled} onClick={onCancel}>Cancel</button>
        <button
          type="button"
          className="primary-command compact"
          disabled={disabled || !trimmedBody}
          onClick={() => onSave({ body: trimmedBody, tags: csvValues(tags), importance: boundedNumber(importance, 0, 1) })}
        >
          <Save size={15} /> Save memory
        </button>
      </div>
    </div>
  );
}

type FactFieldDraft = {
  id: string;
  name: string;
  type: "text" | "number" | "boolean";
  value: string;
};

function WorldFactKnowledgeForm({
  target,
  disabled,
  onCancel,
  onSave
}: {
  target?: CharacterKnowledgeTarget;
  disabled: boolean;
  onCancel: () => void;
  onSave: (payload: { key: string; category: string; confidence: number; value: Record<string, unknown> }) => void;
}) {
  const [key, setKey] = useState(target?.title ?? "");
  const [category, setCategory] = useState(target?.category ?? "");
  const [confidence, setConfidence] = useState(String(typeof target?.confidence === "number" ? target.confidence : 0.8));
  const [fields, setFields] = useState<FactFieldDraft[]>(() => factFieldDrafts(target?.value));
  const updateField = (id: string, patch: Partial<FactFieldDraft>) => setFields((current) => current.map((field) => field.id === id ? { ...field, ...patch } : field));
  const addField = () => setFields((current) => [...current, { id: `fact-field-${Date.now()}`, name: "", type: "text", value: "" }]);
  const factValue = factFieldsValue(fields);
  return (
    <div className="knowledge-form">
      <label className="field-label">
        <span>Fact key</span>
        <input value={key} disabled={disabled} onChange={(event) => setKey(event.target.value)} />
      </label>
      <label className="field-label">
        <span>Fact category</span>
        <input value={category} disabled={disabled} onChange={(event) => setCategory(event.target.value)} />
      </label>
      <label className="field-label">
        <span>Fact confidence</span>
        <input type="number" min={0} max={1} step={0.01} value={confidence} disabled={disabled} onChange={(event) => setConfidence(event.target.value)} />
      </label>
      <div className="fact-field-list">
        {fields.map((field) => (
          <div className="fact-field-row" key={field.id}>
            <label className="field-label">
              <span>Fact field name</span>
              <input value={field.name} disabled={disabled} onChange={(event) => updateField(field.id, { name: event.target.value })} />
            </label>
            <label className="field-label">
              <span>Fact value type</span>
              <select value={field.type} disabled={disabled} onChange={(event) => updateField(field.id, { type: event.target.value as FactFieldDraft["type"] })}>
                <option value="text">Text</option>
                <option value="number">Number</option>
                <option value="boolean">Boolean</option>
              </select>
            </label>
            <label className="field-label">
              <span>Fact field value</span>
              <input value={field.value} disabled={disabled} onChange={(event) => updateField(field.id, { value: event.target.value })} />
            </label>
            <button type="button" className={touchActionClassName("destructive-action")} title="Remove field" aria-label="Remove fact field" disabled={disabled || fields.length <= 1} onClick={() => setFields((current) => current.filter((item) => item.id !== field.id))}>
              <TouchActionContents icon={<Trash2 size={14} />} label="Remove" />
            </button>
          </div>
        ))}
        <button type="button" disabled={disabled} onClick={addField}>
          <Plus size={14} /> Add field
        </button>
      </div>
      <div className="command-row end">
        <button type="button" disabled={disabled} onClick={onCancel}>Cancel</button>
        <button
          type="button"
          className="primary-command compact"
          disabled={disabled || !key.trim() || Object.keys(factValue).length === 0}
          onClick={() => onSave({ key: key.trim(), category: category.trim(), confidence: boundedNumber(confidence, 0, 1), value: factValue })}
        >
          <Save size={15} /> Save fact
        </button>
      </div>
    </div>
  );
}

function isCharacterKnowledgeTarget(value: unknown): value is CharacterKnowledgeTarget {
  if (!value || typeof value !== "object") return false;
  const target = value as Record<string, unknown>;
  return typeof target.target_type === "string" && typeof target.target_id === "string";
}

function characterKnowledgeIds(row: Record<string, unknown>) {
  return {
    memory: stringSet(row.linked_memory_ids),
    world_state: stringSet(row.linked_state_ids),
    summary: stringSet(row.linked_summary_ids)
  };
}

function stringSet(value: unknown): Set<string> {
  return new Set(Array.isArray(value) ? value.filter((item): item is string => typeof item === "string") : []);
}

function characterHasKnowledgeTarget(ids: ReturnType<typeof characterKnowledgeIds>, target: CharacterKnowledgeTarget): boolean {
  const type = normalizedKnowledgeTargetType(target.target_type);
  if (type === "memory") return ids.memory.has(target.target_id);
  if (type === "world_state") return ids.world_state.has(target.target_id);
  if (type === "summary") return ids.summary.has(target.target_id);
  return false;
}

function normalizedKnowledgeTargetType(value: string): string {
  return value === "state" ? "world_state" : value;
}

function knowledgeTargetKey(target: CharacterKnowledgeTarget): string {
  return `${normalizedKnowledgeTargetType(target.target_type)}:${target.target_id}`;
}

function knowledgeTargetNoun(target: CharacterKnowledgeTarget): string {
  const type = normalizedKnowledgeTargetType(target.target_type);
  if (type === "world_state") return "fact";
  return type;
}

function knowledgeTargetLabel(target: CharacterKnowledgeTarget): string {
  const type = normalizedKnowledgeTargetType(target.target_type);
  if (type === "world_state") return "World fact";
  return labelize(type);
}

function editableKnowledgeTarget(target: CharacterKnowledgeTarget): boolean {
  const type = normalizedKnowledgeTargetType(target.target_type);
  return type === "memory" || type === "world_state";
}

function knowledgeTargetTitle(target: CharacterKnowledgeTarget): string {
  if (normalizedKnowledgeTargetType(target.target_type) === "memory") return compactInlineTitle(target.body, target.title || "Memory");
  return target.title || target.target_id;
}

function knowledgeTargetBody(target: CharacterKnowledgeTarget): string {
  if (normalizedKnowledgeTargetType(target.target_type) === "world_state" && target.value && typeof target.value === "object") {
    return Object.entries(target.value)
      .map(([key, value]) => `${labelize(key)}: ${knowledgeValueText(value)}`)
      .join(" · ");
  }
  return compactInlineTitle(target.body, "No detail");
}

function knowledgeTargetPills(target: CharacterKnowledgeTarget): string[] {
  const pills: string[] = [];
  if (target.category) pills.push(target.category);
  if (typeof target.importance === "number") pills.push(`${Math.round(target.importance * 100)}% important`);
  if (typeof target.confidence === "number") pills.push(`${Math.round(target.confidence * 100)}% confident`);
  if (Array.isArray(target.tags)) pills.push(...target.tags.slice(0, 2));
  if (Array.isArray(target.linked_character_ids) && target.linked_character_ids.length) pills.push(`${target.linked_character_ids.length} linked`);
  return pills.slice(0, 4);
}

function knowledgeSearchText(target: CharacterKnowledgeTarget): string {
  return [
    target.target_type,
    target.title,
    target.body,
    target.category,
    Array.isArray(target.tags) ? target.tags.join(" ") : "",
    target.value && typeof target.value === "object" ? Object.entries(target.value).map(([key, value]) => `${key} ${knowledgeValueText(value)}`).join(" ") : ""
  ].join(" ").toLowerCase();
}

function knowledgeValueText(value: unknown): string {
  if (value === null || value === undefined || value === "") return "None";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return String(value);
  if (Array.isArray(value)) return `${value.length} items`;
  if (typeof value === "object") return "Complex value";
  return String(value);
}

function csvValues(value: string): string[] {
  return value.split(",").map((item) => item.trim()).filter(Boolean);
}

function boundedNumber(value: string, min: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return min;
  return Math.min(max, Math.max(min, parsed));
}

function factFieldDrafts(value: unknown): FactFieldDraft[] {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    const entries = Object.entries(value as Record<string, unknown>);
    if (entries.length) {
      return entries.map(([name, item], index) => ({
        id: `fact-field-${index}-${name}`,
        name,
        type: typeof item === "number" ? "number" : typeof item === "boolean" ? "boolean" : "text",
        value: typeof item === "object" && item !== null ? knowledgeValueText(item) : String(item ?? "")
      }));
    }
  }
  return [{ id: "fact-field-0", name: "text", type: "text", value: "" }];
}

function factFieldsValue(fields: FactFieldDraft[]): Record<string, unknown> {
  const value: Record<string, unknown> = {};
  fields.forEach((field) => {
    const name = field.name.trim();
    if (!name) return;
    if (field.type === "number") {
      const parsed = Number(field.value);
      value[name] = Number.isFinite(parsed) ? parsed : 0;
      return;
    }
    if (field.type === "boolean") {
      value[name] = ["true", "yes", "1", "on"].includes(field.value.trim().toLowerCase());
      return;
    }
    value[name] = field.value;
  });
  return value;
}

function sanitizeCharacterLinkRow(row: Record<string, unknown>, targetOptions: Record<string, unknown>[]): Record<string, unknown> {
  const validTargets = new Set(
    targetOptions
      .map((target) => {
        const type = target.target_type;
        const id = target.target_id;
        return typeof type === "string" && typeof id === "string" ? `${type}:${id}` : null;
      })
      .filter((target): target is string => target !== null)
  );
  return {
    ...row,
    linked_memory_ids: validCharacterLinkIds(row.linked_memory_ids, "memory", validTargets),
    linked_state_ids: validCharacterLinkIds(row.linked_state_ids, "world_state", validTargets),
    linked_summary_ids: validCharacterLinkIds(row.linked_summary_ids, "summary", validTargets),
    locked_fields: characterLockedFields(row.locked_fields)
  };
}

function validCharacterLinkIds(value: unknown, targetType: string, validTargets: Set<string>): string[] {
  if (!Array.isArray(value)) return [];
  return value.filter((id): id is string => typeof id === "string" && validTargets.has(`${targetType}:${id}`));
}

function characterLockedFields(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  const locked = new Set<string>();
  value.forEach((field) => {
    if (typeof field !== "string") return;
    const normalized = CHARACTER_LOCK_FIELD_ALIASES[field] ?? field;
    if (CHARACTER_LOCK_FIELD_IDS.has(normalized)) locked.add(normalized);
  });
  return CHARACTER_LOCK_FIELDS.map(([id]) => id).filter((id) => locked.has(id));
}
