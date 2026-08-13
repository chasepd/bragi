import React, { useCallback, useEffect, useRef, useState } from "react";
import { keepPreviousData, useInfiniteQuery, useMutation, useQuery } from "@tanstack/react-query";
import { ArrowUp, Check, FileWarning, Film, History, Image, Loader2, Play, RefreshCw, Trash2, X } from "lucide-react";
import { deleteJson, postJson } from "./api";
import type { ChatHistoryMessage, ChatHistoryModel, ChronicleMessage, Job, MediaAsset, MediaModel, RuntimeModel } from "./api";
import {
  apiRead,
  canUseChildRestrictedControls,
  chatHistoryPath,
  ConfirmModal,
  DataViewer,
  DialogPanel,
  EmptyState,
  InlineNotice,
  labelize,
  MarkdownView,
  mediaAssetPath,
  mediaAssetThumbnailPath,
  mediaPath,
  ModalBackdrop,
  PanelHeader,
  SegmentedTabs
} from "./workbenchCore";
import type { CurrentUser, RunJob } from "./workbenchCore";
import { canRegenerateImageAsset, RegeneratePromptDialog } from "./mediaRegeneration";

export function HistoryPanel({ activeSaveId }: { activeSaveId: string | null }) {
  const [filter, setFilter] = useState("all");
  const scrollRef = useRef<HTMLElement | null>(null);
  const history = useInfiniteQuery({
    queryKey: ["chat-history", activeSaveId, filter],
    queryFn: ({ pageParam }) => apiRead<ChatHistoryModel>(chatHistoryPath(activeSaveId, filter, pageParam)),
    initialPageParam: null as string | null,
    getNextPageParam: (lastPage) => (
      lastPage.has_more_before && lastPage.oldest_message_id
        ? lastPage.oldest_message_id
        : undefined
    ),
    enabled: Boolean(activeSaveId)
  });
  const pages = history.data?.pages ?? [];
  const model = pages[0];
  const messages = [...pages].reverse().flatMap((page) => page.messages);
  const matchingMessageCount = model?.matching_message_count ?? model?.messages.length ?? 0;
  const totalMessageCount = model?.total_message_count ?? 0;
  const loadedMessageCount = messages.length;
  const loadOlderHistory = useCallback(async () => {
    if (!history.hasNextPage || history.isFetchingNextPage) return;
    const node = scrollRef.current;
    const previousScrollHeight = node?.scrollHeight ?? 0;
    await history.fetchNextPage();
    window.requestAnimationFrame(() => {
      if (!node) return;
      node.scrollTop += node.scrollHeight - previousScrollHeight;
    });
  }, [history]);
  return (
    <aside className="right-panel" ref={scrollRef}>
      <PanelHeader icon={<History size={18} />} title="History" />
      {model?.filter_options?.length ? (
        <SegmentedTabs
          className="segmented history-filters"
          label="History filters"
          value={filter}
          onChange={setFilter}
          options={model.filter_options.map((option) => ({ value: option.filter_id, label: option.label }))}
        />
      ) : null}
      {history.isLoading ? <p className="muted">Loading history...</p> : null}
      {history.error instanceof Error ? <InlineNotice>{history.error.message}</InlineNotice> : null}
      {model ? (
        <p className="history-counts">
          Showing {loadedMessageCount} of {matchingMessageCount} matching · {totalMessageCount} total
        </p>
      ) : null}
      {history.hasNextPage ? (
        <button
          type="button"
          className="history-load-earlier"
          onClick={loadOlderHistory}
          disabled={history.isFetchingNextPage}
        >
          {history.isFetchingNextPage ? <Loader2 className="spin" size={15} /> : <ArrowUp size={15} />}
          {history.isFetchingNextPage ? "Loading..." : "Load earlier"}
        </button>
      ) : null}
      {messages.length ? (
        <div className="history-list">
          {messages.map((message) => (
            <HistoryMessageRow key={message.message_id} message={message} />
          ))}
        </div>
      ) : (
        <EmptyState
          icon={<History size={30} />}
          title={model?.empty_title ?? "No save loaded"}
          body={model?.empty_detail ?? "Start or load a save to inspect its chronicle history."}
        />
      )}
    </aside>
  );
}

function HistoryMessageRow({ message }: { message: ChatHistoryMessage }) {
  const providerLabel = message.provider_model_label || [message.provider, message.model].filter(Boolean).join(" / ");
  return (
    <article className={`history-message ${message.role}`}>
      <header>
        <div>
          <strong>{message.speaker_name || message.role_label || labelize(message.role)}</strong>
          <span>{message.created_at ?? "No timestamp"}</span>
        </div>
        <div className="history-message-meta">
          {providerLabel ? <small>{providerLabel}</small> : null}
          {typeof message.token_estimate === "number" ? <small>{message.token_estimate} tokens</small> : null}
          {message.image_count ? <small>{message.image_count} images</small> : null}
        </div>
      </header>
      <MarkdownView
        message={{
          message_id: message.message_id,
          role: message.role,
          speaker_name: message.speaker_name,
          body: message.body,
          actions: [],
          markdown_blocks: message.markdown_blocks
        }}
      />
    </article>
  );
}

export function MediaPanel({ model, runJob, currentUser = null }: { model?: RuntimeModel; runJob: RunJob; currentUser?: CurrentUser | null }) {
  const [selectedAssetId, setSelectedAssetId] = useState<string | null>(null);
  const [modalAsset, setModalAsset] = useState<MediaAsset | null>(null);
  const [initialMediaError, setInitialMediaError] = useState("");
  const [initialMediaPending, setInitialMediaPending] = useState(false);
  const activeSaveId = model?.active_save_id ?? null;
  const media = useQuery({
    queryKey: ["media", activeSaveId],
    queryFn: ({ signal }) => apiRead<MediaModel>(mediaPath(activeSaveId ?? ""), signal),
    enabled: Boolean(activeSaveId),
    placeholderData: keepPreviousData
  });
  const mediaModel = media.data ?? model?.media ?? null;
  const latest = mediaModel?.latest_scene_media ?? mediaModel?.latest_scene_image;
  const assets = mediaModel?.media_history?.length ? mediaModel.media_history : mediaModel?.image_history ?? [];
  const availableAssets = uniqueMediaAssets([latest, ...assets]);
  const assetIds = availableAssets.map((asset) => asset.id).join(":");
  const selectedAsset = availableAssets.find((asset) => asset.id === selectedAssetId) ?? null;
  const primaryAsset = selectedAsset ?? latest ?? assets[0] ?? null;
  const firstNarrator = firstNarratorMessage(model);
  const initialLabel = initialMediaActionLabel(model);
  const canManageMedia = canUseChildRestrictedControls(currentUser);
  useEffect(() => {
    setModalAsset(null);
  }, [activeSaveId, canManageMedia]);
  useEffect(() => {
    if (selectedAssetId && !availableAssets.some((asset) => asset.id === selectedAssetId)) {
      setSelectedAssetId(null);
    }
  }, [assetIds, selectedAssetId]);
  return (
    <aside className="right-panel">
      <PanelHeader icon={<Image size={18} />} title="Scene Media" />
      {primaryAsset ? (
        <button
          className="image-button"
          onClick={() => setModalAsset(primaryAsset)}
          aria-label={`Open full media viewer for ${mediaSelectionLabel(primaryAsset)}`}
        >
          <MediaPrimaryPreview asset={primaryAsset} activeSaveId={activeSaveId} />
          <MediaReferenceBadge asset={primaryAsset} />
        </button>
      ) : (
        <EmptyState
          icon={<Image size={30} />}
          title="No scene image yet"
          body="Generate an image from a chronicle message to build the gallery."
          action={firstNarrator ? (
            <>
              <button
                type="button"
                className="primary-command compact"
                disabled={initialMediaPending}
                onClick={async () => {
                  setInitialMediaPending(true);
                  setInitialMediaError("");
                  try {
                    runJob(await postJson<Job>("/api/media/initial", { message_id: firstNarrator.message_id, save_id: activeSaveId }));
                  } catch (failure) {
                    setInitialMediaError(failure instanceof Error ? failure.message : "Could not start media generation");
                  } finally {
                    setInitialMediaPending(false);
                  }
                }}
              >
                {initialMediaPending ? <Loader2 className="spin" size={15} /> : <Image size={15} />}
                {" "}
                {initialLabel}
              </button>
              {initialMediaError ? <InlineNotice>{initialMediaError}</InlineNotice> : null}
            </>
          ) : null}
        />
      )}
      <div className="history-grid">
        {assets.map((asset) => (
          <button
            className={`thumb-button${asset.id === primaryAsset?.id ? " selected" : ""}`}
            key={asset.id}
            onClick={() => setSelectedAssetId(asset.id)}
            title={asset.prompt_preview || asset.status}
            aria-label={`Select ${mediaSelectionLabel(asset)}`}
            aria-current={asset.id === primaryAsset?.id ? "true" : undefined}
          >
            <MediaHistoryTile asset={asset} activeSaveId={activeSaveId} />
            <MediaReferenceBadge asset={asset} compact />
          </button>
        ))}
      </div>
      {modalAsset ? <ImagePreview asset={modalAsset} activeSaveId={activeSaveId} onClose={() => setModalAsset(null)} runJob={runJob} canGenerate canManage={canManageMedia} /> : null}
    </aside>
  );
}

function MediaPrimaryPreview({ asset, activeSaveId }: { asset: MediaAsset; activeSaveId: string | null }) {
  if (asset.file_available === false) {
    return (
      <span className="scene-media-placeholder missing">
        <FileWarning size={24} />
        <span>{mediaTypeLabel(asset)} unavailable</span>
      </span>
    );
  }
  if (isVideoAsset(asset)) {
    return (
      <span className="scene-media-placeholder video">
        <Film size={30} />
        <span>{asset.prompt_preview || "Generated video"}</span>
      </span>
    );
  }
  return (
    <img
      className="scene-image"
      src={mediaAssetThumbnailPath(asset.id, activeSaveId)}
      alt={asset.prompt_preview}
      decoding="async"
    />
  );
}

function MediaHistoryTile({ asset, activeSaveId }: { asset: MediaAsset; activeSaveId: string | null }) {
  if (asset.file_available === false) {
    return (
      <span className="media-thumb-placeholder missing">
        <FileWarning size={16} />
      </span>
    );
  }
  if (isVideoAsset(asset)) {
    return (
      <span className="video-thumb">
        <Play size={18} />
      </span>
    );
  }
  return (
    <img
      src={mediaAssetThumbnailPath(asset.id, activeSaveId)}
      alt={asset.prompt_preview}
      loading="lazy"
      decoding="async"
    />
  );
}

function MediaReferenceBadge({ asset, compact = false }: { asset: MediaAsset; compact?: boolean }) {
  if (!asset.is_character_reference) return null;
  return (
    <span className={compact ? "media-reference-badge compact" : "media-reference-badge"}>
      {asset.metadata?.source === "uploaded" ? "Uploaded" : "Reference"}
    </span>
  );
}

function uniqueMediaAssets(assets: (MediaAsset | null | undefined)[]) {
  const seen = new Set<string>();
  const unique: MediaAsset[] = [];
  for (const asset of assets) {
    if (!asset || seen.has(asset.id)) continue;
    seen.add(asset.id);
    unique.push(asset);
  }
  return unique;
}

function mediaSelectionLabel(asset: MediaAsset) {
  return asset.prompt_preview || asset.status || mediaTypeLabel(asset);
}

function initialMediaActionLabel(_model?: RuntimeModel) {
  return "Generate opening image";
}


function firstNarratorMessage(model?: RuntimeModel) {
  return model?.chronicle?.messages?.find((message) => message.role === "narrator") ?? null;
}

export function ImagePreview({
  asset,
  activeSaveId,
  onClose,
  runJob,
  canMutate,
  canGenerate = canMutate ?? true,
  canManage = canMutate ?? true
}: {
  asset: MediaAsset;
  activeSaveId: string | null;
  onClose: () => void;
  runJob: RunJob;
  canMutate?: boolean;
  canGenerate?: boolean;
  canManage?: boolean;
}) {
  const [animationPromptOpen, setAnimationPromptOpen] = useState(false);
  const [regeneratePromptOpen, setRegeneratePromptOpen] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const mediaLabel = mediaTypeLabel(asset);
  const canRegenerate = canGenerate && (!asset.is_character_reference || canManage);
  useEffect(() => {
    setAnimationPromptOpen(false);
    setRegeneratePromptOpen(false);
    setDeleteConfirmOpen(false);
  }, [activeSaveId, canGenerate, canManage]);
  const setReference = useMutation({
    mutationFn: () => postJson<Job>(`/api/media/${asset.id}/set-character-reference`, { save_id: activeSaveId }),
    onSuccess: (job) => {
      runJob(job);
      onClose();
    }
  });
  const deleteAsset = useMutation({
    mutationFn: () => deleteJson<Job>(`/api/media/${asset.id}${activeSaveId ? `?save_id=${encodeURIComponent(activeSaveId)}` : ""}`),
    onSuccess: (job) => {
      runJob(job);
      onClose();
    }
  });
  return (
    <ModalBackdrop>
      <DialogPanel
        className="image-preview"
        titleId={`media-preview-title-${asset.id}`}
        onClose={onClose}
      >
        <header>
          <div>
            <h2 id={`media-preview-title-${asset.id}`}>
              {mediaLabel} · {asset.status}
              {asset.is_character_reference ? <span className="inline-reference-badge">Reference</span> : null}
            </h2>
            <p className="muted">{mediaMetadataLine(asset)}</p>
          </div>
          <button type="button" onClick={onClose} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <MediaPreviewFrame asset={asset} activeSaveId={activeSaveId} />
        <p className="muted">{asset.prompt_preview || "No prompt preview"}</p>
        <DataViewer
          value={mediaDetailRows(asset)}
          emptyLabel="No media detail"
        />
        {canGenerate || canManage ? (
          <div className="command-row end">
            {setReference.error ? <InlineNotice>{setReference.error instanceof Error ? setReference.error.message : "Could not update reference"}</InlineNotice> : null}
            {canManage ? <button
              type="button"
              className="danger-command"
              disabled={deleteAsset.isPending}
              onClick={() => setDeleteConfirmOpen(true)}
            >
              <Trash2 size={15} /> Delete
            </button> : null}
            {canManage && asset.can_set_character_reference ? (
              <button
                type="button"
                disabled={!activeSaveId || setReference.isPending}
                onClick={() => setReference.mutate()}
              >
                {setReference.isPending ? <Loader2 className="spin" size={15} /> : <Check size={15} />}
                Set as reference
              </button>
            ) : null}
            {canGenerate && asset.can_animate ? (
              <button
                type="button"
                disabled={!activeSaveId}
                onClick={() => setAnimationPromptOpen(true)}
              >
                <Play size={15} /> Animate this
              </button>
            ) : null}
            {canRegenerate ? <button
              type="button"
              disabled={!activeSaveId || !canRegenerateImageAsset(asset)}
              onClick={() => setRegeneratePromptOpen(true)}
            >
              <RefreshCw size={15} /> Regenerate with edits
            </button> : null}
          </div>
        ) : null}
      </DialogPanel>
      {regeneratePromptOpen && canRegenerate ? (
        <RegeneratePromptDialog
          assetId={asset.id}
          activeSaveId={activeSaveId}
          onCancel={() => setRegeneratePromptOpen(false)}
          onStarted={(job) => {
            runJob(job);
            setRegeneratePromptOpen(false);
            onClose();
          }}
        />
      ) : null}
      {animationPromptOpen && canGenerate ? (
        <AnimationPromptDialog
          assetId={asset.id}
          activeSaveId={activeSaveId}
          onCancel={() => setAnimationPromptOpen(false)}
          onStarted={(job) => {
            runJob(job);
            setAnimationPromptOpen(false);
            onClose();
          }}
        />
      ) : null}
      {deleteConfirmOpen && canManage ? (
        <ConfirmModal
          title={`Delete ${mediaLabel.toLowerCase()}?`}
          body={asset.prompt_preview ? `Prompt: ${asset.prompt_preview}` : `Status: ${asset.status}`}
          confirmLabel="Delete"
          destructive
          onCancel={() => setDeleteConfirmOpen(false)}
          onConfirm={async () => {
            await deleteAsset.mutateAsync();
          }}
        />
      ) : null}
    </ModalBackdrop>
  );
}

function MediaPreviewFrame({ asset, activeSaveId }: { asset: MediaAsset; activeSaveId: string | null }) {
  const source = mediaAssetPath(asset.id, activeSaveId);
  if (asset.file_available === false) {
    return (
      <div className="media-missing-preview" role="status">
        <FileWarning size={28} />
        <strong>{mediaTypeLabel(asset)} file unavailable</strong>
        <span>{asset.mime_type}</span>
      </div>
    );
  }
  if (isVideoAsset(asset)) {
    return (
      <video
        className="media-preview-video"
        src={source}
        controls
        aria-label="Video preview"
      />
    );
  }
  return <img src={source} alt={asset.prompt_preview} />;
}

function AnimationPromptDialog({
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
  const [motionPrompt, setMotionPrompt] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog animation-dialog" titleId={titleId} onClose={onCancel}>
        <header>
          <h2 id={titleId}>Animate image</h2>
          <button type="button" onClick={onCancel} title="Close" aria-label="Close"><X size={16} /></button>
        </header>
        <label className="field-label">
          <span>Motion guidance</span>
          <textarea
            value={motionPrompt}
            placeholder="Lantern flame gutters, fog coils through the arch..."
            onChange={(event) => setMotionPrompt(event.target.value)}
          />
        </label>
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
                onStarted(await postJson<Job>(`/api/media/${assetId}/animate`, {
                  save_id: activeSaveId,
                  motion_prompt: motionPrompt
                }));
              } catch (failure) {
                setError(failure instanceof Error ? failure.message : "Could not start animation");
              } finally {
                setBusy(false);
              }
            }}
          >
            {busy ? <Loader2 className="spin" size={15} /> : <Play size={15} />}
            Start animation
          </button>
        </div>
      </DialogPanel>
    </ModalBackdrop>
  );
}

function mediaTypeLabel(asset: MediaAsset) {
  return isVideoAsset(asset) ? "Video" : "Image";
}

function mediaMetadataLine(asset: MediaAsset) {
  const parts = [
    asset.created_at ?? "No timestamp",
    asset.mime_type,
    [asset.provider, asset.model].filter(Boolean).join(" / ")
  ].filter(Boolean);
  return parts.join(" · ");
}

function mediaDetailRows(asset: MediaAsset): Record<string, unknown> {
  const metadata = asset.metadata && Object.keys(asset.metadata).length ? JSON.stringify(asset.metadata) : null;
  return {
    ...(asset.character_name ? { character_name: asset.character_name } : {}),
    type: asset.type,
    mime_type: asset.mime_type,
    provider: asset.provider,
    model: asset.model,
    created_at: asset.created_at,
    file_available: asset.file_available !== false,
    source_media_asset_id: asset.source_media_asset_id,
    source_media: sourceMediaLabel(asset.source_media),
    source_message: asset.source_message,
    prompt: asset.prompt,
    metadata
  };
}

function sourceMediaLabel(sourceMedia: MediaAsset["source_media"]) {
  if (!sourceMedia) return null;
  return [
    sourceMedia.type,
    sourceMedia.prompt_preview || sourceMedia.id,
    sourceMedia.mime_type
  ].filter(Boolean).join(" · ");
}

function isVideoAsset(asset: MediaAsset) {
  return asset.type === "video" || asset.mime_type.startsWith("video/");
}
