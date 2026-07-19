import React, { useEffect, useState } from "react";
import { Loader2, RefreshCw, X } from "lucide-react";
import { postJson } from "./api";
import type { Job, MediaAsset, MediaAssetPrompt } from "./api";
import {
  apiRead,
  DialogForm,
  InlineNotice,
  mediaAssetPromptPath,
  ModalBackdrop
} from "./workbenchCore";

export function canRegenerateImageAsset(asset: Pick<MediaAsset, "type" | "mime_type" | "provider" | "status" | "file_available">) {
  return asset.type === "image"
    && !asset.mime_type.startsWith("video/")
    && Boolean(asset.provider)
    && asset.provider !== "local"
    && asset.status === "succeeded"
    && asset.file_available !== false;
}

export function RegeneratePromptDialog({
  assetId,
  activeSaveId,
  onCancel,
  onStarted
}: {
  assetId: string;
  activeSaveId: string | null;
  onCancel: () => void;
  onStarted: (job: Job) => void;
}) {
  const titleId = React.useId();
  const [promptDraft, setPromptDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    void apiRead<MediaAssetPrompt>(
      mediaAssetPromptPath(assetId, activeSaveId),
      controller.signal,
    ).then((response) => {
      setPromptDraft(response.prompt);
    }).catch((failure) => {
      if (!controller.signal.aborted) {
        setError(failure instanceof Error ? failure.message : "Could not load image prompt");
      }
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false);
    });
    return () => controller.abort();
  }, [activeSaveId, assetId]);
  const promptRequired = !promptDraft.trim();
  const submit = async () => {
    if (promptRequired || busy) return;
    setBusy(true);
    setError("");
    try {
      onStarted(await postJson<Job>(`/api/media/${assetId}/regenerate`, {
        save_id: activeSaveId,
        prompt: promptDraft
      }));
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not start image regeneration");
      setBusy(false);
    }
  };
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog regenerate-prompt-dialog"
        titleId={titleId}
        onClose={() => {
          if (!busy) onCancel();
        }}
        onSubmit={(event) => {
          event.preventDefault();
          void submit();
        }}
      >
        <header>
          <h2 id={titleId}>Regenerate with edits</h2>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close" disabled={busy}>
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Image prompt</span>
          <textarea
            className="regenerate-prompt-field"
            value={promptDraft}
            disabled={loading || busy}
            onChange={(event) => setPromptDraft(event.currentTarget.value)}
          />
        </label>
        {loading ? (
          <p className="muted regenerate-prompt-loading">
            <Loader2 className="spin" size={15} />
            Loading prompt...
          </p>
        ) : null}
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onCancel} disabled={busy}>Cancel</button>
          <button
            type="submit"
            className="primary-command compact"
            disabled={loading || busy || promptRequired}
          >
            {busy ? <Loader2 className="spin" size={15} /> : <RefreshCw size={15} />}
            Regenerate
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}
