import React, { useMemo, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  Download,
  Edit3,
  Globe2,
  Loader2,
  Plus,
  Search,
  Sparkles,
  Trash2,
  Upload,
  X
} from "lucide-react";
import { api, deleteJson, postJson } from "./api";
import type { PersistentWorld } from "./api";

const WORLD_SECTIONS = [
  ["overview", "Setting overview"],
  ["cultures", "Cultures and peoples"],
  ["geography", "Geography and places"],
  ["factions", "Factions and powers"],
  ["history_and_myths", "History and myths"],
  ["magic_or_technology", "Magic or technology"],
  ["tone", "Tone and themes"]
] as const;
const WORLD_SECTION_IDS: ReadonlySet<string> = new Set(WORLD_SECTIONS.map(([id]) => id));

type WorldDraft = {
  title: string;
  description: string;
  sections: Record<string, string>;
  source_metadata: Record<string, unknown> | null;
  content_rating: string;
};

export function PersistentWorldLibrary({
  adminControlsAllowed,
  allowImportExport,
  onChanged
}: {
  adminControlsAllowed: boolean;
  allowImportExport: boolean;
  onChanged: () => void;
}) {
  const client = useQueryClient();
  const fileInput = useRef<HTMLInputElement | null>(null);
  const [query, setQuery] = useState("");
  const [editingWorld, setEditingWorld] = useState<PersistentWorld | null | "new">(null);
  const [error, setError] = useState("");
  const [importing, setImporting] = useState(false);
  const worldsQuery = useQuery({
    queryKey: ["persistent-worlds"],
    queryFn: () => api<{ worlds: PersistentWorld[] }>("/api/worlds")
  });
  const worlds = worldsQuery.data?.worlds ?? [];
  const visibleWorlds = useMemo(() => {
    const terms = query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return worlds;
    return worlds.filter((world) => {
      const haystack = `${world.title} ${world.description} ${Object.values(world.sections).join(" ")}`.toLocaleLowerCase();
      return terms.every((term) => haystack.includes(term));
    });
  }, [query, worlds]);
  const refresh = () => {
    void client.invalidateQueries({ queryKey: ["persistent-worlds"] });
    onChanged();
  };
  const importBundle = async (file: File) => {
    setImporting(true);
    setError("");
    try {
      const body = new FormData();
      body.append("file", file);
      const preview = await api<{
        preview_id: string;
        preview: { title: string; description?: string };
      }>("/api/persistent-world-bundles/preview", { method: "POST", body });
      if (!window.confirm(`Import “${preview.preview.title}” as a new persistent world?`)) return;
      await postJson(`/api/persistent-world-bundles/import/${preview.preview_id}`, {});
      refresh();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not import world bundle");
    } finally {
      setImporting(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };
  return (
    <section className="library-panel world-library" id="persistent-worlds-panel" role="tabpanel">
      <div className="world-library-heading">
        <div>
          <span className="eyebrow"><Globe2 size={13} /> Shared settings</span>
          <h2>Persistent worlds</h2>
          <p>Build a setting once, then reuse its atmosphere and history across many scenarios.</p>
        </div>
        <button type="button" className="primary-command compact" onClick={() => setEditingWorld("new")}>
          <Plus size={15} /> New world
        </button>
      </div>
      {error ? <p className="settings-warning" role="alert">{error}</p> : null}
      <div className="world-library-toolbar">
        <label className="library-search">
          <Search size={15} aria-hidden="true" />
          <input
            aria-label="Search persistent worlds"
            placeholder="Search worlds"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <button type="button" aria-label="Clear world search" disabled={!query} onClick={() => setQuery("")}>
            <X size={14} />
          </button>
        </label>
        {allowImportExport ? (
          <>
            <input
              ref={fileInput}
              className="sr-only"
              type="file"
              accept=".bragi-world,application/zip"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void importBundle(file);
              }}
            />
            <button type="button" className="secondary-command compact" disabled={importing} onClick={() => fileInput.current?.click()}>
              {importing ? <Loader2 className="spin" size={15} /> : <Upload size={15} />} Import
            </button>
          </>
        ) : null}
        <span className="library-result-count">{visibleWorlds.length} of {worlds.length} worlds</span>
      </div>
      {worldsQuery.isLoading ? <p className="muted">Loading worlds...</p> : null}
      {worldsQuery.error ? <p className="settings-warning" role="alert">Could not load persistent worlds.</p> : null}
      <div className="world-card-grid">
        {visibleWorlds.map((world) => (
          <article className="world-card" key={world.world_id}>
            <div className="world-card-topline">
              <span className="world-card-icon"><Globe2 size={17} /></span>
              <span className="library-pill">{world.content_rating}</span>
            </div>
            <h3>{world.title}</h3>
            <p>{world.description || "No description yet. Open the world to define its shared setting prose."}</p>
            <div className="world-card-meta">
              <span>{Object.keys(world.sections).length} sections</span>
              <span>{world.scenario_count} {world.scenario_count === 1 ? "scenario" : "scenarios"}</span>
            </div>
            <div className="world-card-actions">
              {adminControlsAllowed ? (
                <button type="button" className="secondary-command compact" onClick={() => setEditingWorld(world)}>
                  <Edit3 size={14} /> Edit
                </button>
              ) : null}
              {allowImportExport ? (
                <button type="button" className="icon-button" title="Export world bundle" aria-label={`Export ${world.title}`} onClick={() => window.open(`/api/persistent-world-bundles/export/${encodeURIComponent(world.world_id)}`, "_blank", "noopener,noreferrer")}>
                  <Download size={15} />
                </button>
              ) : null}
              {adminControlsAllowed ? (
                <button
                  type="button"
                  className="icon-button destructive-action"
                  title={world.scenario_count ? "Unlink scenarios before deleting" : "Delete world"}
                  aria-label={`Delete ${world.title}`}
                  disabled={world.scenario_count > 0}
                  onClick={async () => {
                    if (!window.confirm(`Delete “${world.title}”?`)) return;
                    try {
                      await deleteJson(`/api/worlds/${encodeURIComponent(world.world_id)}`);
                      refresh();
                    } catch (failure) {
                      setError(failure instanceof Error ? failure.message : "Could not delete world");
                    }
                  }}
                >
                  <Trash2 size={15} />
                </button>
              ) : null}
            </div>
          </article>
        ))}
      </div>
      {!worldsQuery.isLoading && !visibleWorlds.length ? (
        <div className="world-library-empty">
          <Globe2 size={25} />
          <strong>{worlds.length ? "No worlds match" : "No persistent worlds yet"}</strong>
          <span>Start with a cultural setting, a fantasy realm, or any world you want to revisit.</span>
        </div>
      ) : null}
      {editingWorld ? (
        <PersistentWorldEditor
          world={editingWorld === "new" ? null : editingWorld}
          onClose={() => setEditingWorld(null)}
          onSaved={() => {
            setEditingWorld(null);
            refresh();
          }}
        />
      ) : null}
    </section>
  );
}

function PersistentWorldEditor({
  world,
  onClose,
  onSaved
}: {
  world: PersistentWorld | null;
  onClose: () => void;
  onSaved: () => void;
}) {
  const initialSections = {
    ...Object.fromEntries(Object.entries(world?.sections ?? {})),
    ...Object.fromEntries(WORLD_SECTIONS.map(([id]) => [id, world?.sections[id] ?? ""]))
  };
  const initial: WorldDraft = {
    title: world?.title ?? "",
    description: world?.description ?? "",
    sections: initialSections,
    source_metadata: world?.source_metadata ?? null,
    content_rating: world?.content_rating ?? "pg-13"
  };
  const [draft, setDraft] = useState(initial);
  const [seed, setSeed] = useState("");
  const [newSectionKey, setNewSectionKey] = useState("");
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const updateSection = (id: string, value: string) => setDraft((current) => ({
    ...current,
    sections: { ...current.sections, [id]: value }
  }));
  const customSectionIds = Object.keys(draft.sections).filter((id) => !WORLD_SECTION_IDS.has(id));
  const addCustomSection = () => {
    const key = newSectionKey.trim().toLocaleLowerCase().replace(/[^a-z0-9_-]+/g, "_");
    if (!key || WORLD_SECTION_IDS.has(key) || key in draft.sections) return;
    setDraft((current) => ({
      ...current,
      sections: { ...current.sections, [key]: "" }
    }));
    setNewSectionKey("");
  };
  const generate = async () => {
    if (!seed.trim()) {
      setError("Add a seed before generating a world draft.");
      return;
    }
    setGenerating(true);
    setError("");
    try {
      const result = await postJson<{ draft: WorldDraft }>("/api/worlds/draft", {
        seed,
        title: draft.title,
        description: draft.description
      });
      setDraft((current) => ({
        ...current,
        title: result.draft.title || current.title,
        description: result.draft.description || current.description,
        sections: { ...current.sections, ...result.draft.sections },
        source_metadata: result.draft.source_metadata,
        content_rating: result.draft.content_rating || current.content_rating
      }));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not generate world draft");
    } finally {
      setGenerating(false);
    }
  };
  const save = async (event: React.FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError("");
    try {
      await postJson(
        world ? `/api/worlds/${encodeURIComponent(world.world_id)}/definition` : "/api/worlds/manual",
        draft
      );
      onSaved();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not save world");
      setSaving(false);
    }
  };
  return (
    <div className="modal-backdrop">
      <form className="world-editor-dialog preview-dialog" role="dialog" aria-modal="true" onSubmit={save}>
        <header className="dialog-heading">
          <div>
            <span className="eyebrow"><Globe2 size={13} /> {world ? "Edit setting" : "New setting"}</span>
            <h2>{world ? "Refine persistent world" : "Create a persistent world"}</h2>
          </div>
          <button type="button" className="icon-button" title="Close" aria-label="Close" onClick={onClose}><X size={16} /></button>
        </header>
        <p className="world-editor-intro">These setting sections stay reusable across scenarios. Starting a save records the version that was current at that moment.</p>
        {!world ? (
          <div className="world-ai-draft">
            <label className="field-label">
              <span>AI world seed</span>
              <textarea value={seed} onChange={(event) => setSeed(event.target.value)} placeholder="A coastal confederacy where tides reveal ancient roads..." />
            </label>
            <button type="button" className="secondary-command" disabled={generating} onClick={() => void generate()}>
              {generating ? <Loader2 className="spin" size={15} /> : <Sparkles size={15} />} Draft setting sections
            </button>
          </div>
        ) : null}
        <label className="field-label">
          <span>World title</span>
          <input required value={draft.title} onChange={(event) => setDraft({ ...draft, title: event.target.value })} autoFocus />
        </label>
        <label className="field-label">
          <span>Short description</span>
          <textarea value={draft.description} onChange={(event) => setDraft({ ...draft, description: event.target.value })} />
        </label>
        <label className="field-label">
          <span>Content rating</span>
          <select value={draft.content_rating} onChange={(event) => setDraft({ ...draft, content_rating: event.target.value })}>
            <option value="g">G</option>
            <option value="pg">PG</option>
            <option value="pg-13">PG-13</option>
            <option value="r">R</option>
            <option value="unrated">Unrated</option>
          </select>
        </label>
        <div className="world-section-editor">
          {WORLD_SECTIONS.map(([id, label]) => (
            <label className="field-label" key={id}>
              <span>{label}</span>
              <textarea value={draft.sections[id] ?? ""} onChange={(event) => updateSection(id, event.target.value)} />
            </label>
          ))}
          {customSectionIds.map((id) => (
            <label className="field-label custom-world-section" key={id}>
              <span>
                {id.replace(/[_-]+/g, " ")}
                <button
                  type="button"
                  className="icon-button"
                  title={`Remove ${id} section`}
                  aria-label={`Remove ${id} section`}
                  onClick={() => setDraft((current) => {
                    const sections = { ...current.sections };
                    delete sections[id];
                    return { ...current, sections };
                  })}
                >
                  <X size={13} />
                </button>
              </span>
              <textarea value={draft.sections[id] ?? ""} onChange={(event) => updateSection(id, event.target.value)} />
            </label>
          ))}
        </div>
        <div className="world-custom-section-row">
          <label className="field-label">
            <span>Additional section</span>
            <input
              value={newSectionKey}
              onChange={(event) => setNewSectionKey(event.target.value)}
              placeholder="Religion, language, customs..."
            />
          </label>
          <button type="button" className="secondary-command compact" disabled={!newSectionKey.trim()} onClick={addCustomSection}>
            <Plus size={14} /> Add section
          </button>
        </div>
        {error ? <p className="settings-warning" role="alert">{error}</p> : null}
        <div className="command-row end">
          <button type="button" className="secondary-command" onClick={onClose}>Cancel</button>
          <button type="submit" className="primary-command compact" disabled={saving || generating}>
            {saving ? <Loader2 className="spin" size={15} /> : <Plus size={15} />} {world ? "Save changes" : "Save world"}
          </button>
        </div>
      </form>
    </div>
  );
}
