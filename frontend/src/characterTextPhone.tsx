import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useVirtualizer } from "@tanstack/react-virtual";
import { Check, ChevronLeft, Edit3, FileWarning, Info, Loader2, MessageSquareText, Plus, RefreshCw, Save, Search, Send, Trash2, Users, X } from "lucide-react";
import { api, deleteJson, postJson } from "./api";
import type { CharacterRegistryModel, CharacterReferenceImage, CharacterTextAttachment, CharacterTextContact, CharacterTextMessage, CharacterTextsModel, CharacterTextThread, ChronicleMessage, Job, RuntimeModel } from "./api";
import {
  apiRead,
  characterTextContactPath,
  characterTextsPath,
  characterTextThreadPath,
  characterTextThreadReadPath,
  characterTextSeenStorageKey,
  CHARACTER_TEXT_ESTIMATED_ROW_HEIGHT,
  CHARACTER_TEXT_ROW_GAP,
  CHARACTER_TEXT_ROW_OVERSCAN,
  charactersPath,
  canUseChildRestrictedControls,
  ConfirmModal,
  DialogForm,
  DialogPanel,
  EmptyState,
  actionIcon,
  applyCharacterTextJobResult,
  focusableElements,
  incomingCharacterTextContacts,
  initialVirtualBottomOffset,
  InlineNotice,
  MarkdownView,
  mediaAssetPath,
  mediaAssetThumbnailPath,
  ModalBackdrop,
  observeVirtualElementOffset,
  observeVirtualElementRect,
  runtimeQueryKey,
  setScrollTopAndNotify,
  touchActionClassName,
  TouchActionContents,
  useDialogFocus,
  useMediaQuery,
  VIRTUAL_LIST_INITIAL_RECT,
  virtualElementRect,
  WORKBENCH_MOBILE_QUERY
} from "./workbenchCore";
import type { CharacterTextSendVariables, CharacterTextSpontaneousVariables, CurrentUser, LocalCharacterTextMessage, RunJob } from "./workbenchCore";
import { canRegenerateImageAsset, RegeneratePromptDialog } from "./mediaRegeneration";

function playerNumberPermissionReason(contact: CharacterTextContact): string {
  return contact.player_number_permission?.reason
    ?? (contact.player_has_character_number
      ? "You can text them."
      : "You do not have this character's number.");
}

function characterNumberPermissionReason(contact: CharacterTextContact): string {
  return contact.character_number_permission?.reason
    ?? (contact.character_has_player_number
      ? "They can text you."
      : "They cannot text you yet.");
}

function characterTextContactPermissionSummary(contact: CharacterTextContact): string {
  if (contact.player_has_character_number && contact.character_has_player_number) {
    return "You can text them. They can text you.";
  }
  if (contact.player_has_character_number) return "You can text them.";
  if (contact.character_has_player_number) return "They can text you. You do not have their number.";
  return "No phone permission saved.";
}

function displayCharacterTextContactName(contact: CharacterTextContact | null | undefined): string {
  return contact?.contact_name?.trim() || contact?.name || "Select a contact";
}

type CharacterTextMessageRowProps = {
  message: CharacterTextMessage;
  previousMessage: CharacterTextMessage | null;
  contact: CharacterTextContact | null;
  activeSaveId: string | null;
  runJob: RunJob;
  canGenerateMedia: boolean;
  mutationsDisabled: boolean;
  onAction: (actionId: string, message: CharacterTextMessage) => void;
};

const CharacterTextMessageRow = React.memo(function CharacterTextMessageRow({
  message,
  previousMessage,
  contact,
  activeSaveId,
  runJob,
  canGenerateMedia,
  mutationsDisabled,
  onAction
}: CharacterTextMessageRowProps) {
  const grouped = characterTextMessageGrouped(message, previousMessage);
  const showDaySeparator = characterTextMessageStartsNewDay(message, previousMessage);
  return (
    <div className="character-text-message-row">
      {showDaySeparator ? (
        <div className="character-text-day-separator">
          <span>{characterTextDayLabel(message.created_at)}</span>
        </div>
      ) : null}
      <CharacterTextBubble
        message={message}
        contact={contact}
        activeSaveId={activeSaveId}
        runJob={runJob}
        canGenerateMedia={canGenerateMedia}
        mutationsDisabled={mutationsDisabled}
        grouped={grouped}
        onAction={onAction}
      />
    </div>
  );
});

export function CharacterTextPhone({
  activeSaveId,
  disabled,
  busyThreadIds,
  runJob,
  seenTextMessageIdsByThread,
  onThreadSeen,
  onClose,
  currentUser = null
}: {
  activeSaveId: string | null;
  disabled: boolean;
  busyThreadIds: ReadonlySet<string>;
  runJob: RunJob;
  seenTextMessageIdsByThread: Record<string, string>;
  onThreadSeen: (threadId: string | null | undefined, messageId: string | null | undefined) => void;
  onClose: () => void;
  currentUser?: CurrentUser | null;
}) {
  const titleId = "character-text-phone-title";
  const client = useQueryClient();
  const [selectedCharacterId, setSelectedCharacterId] = useState<string | null>(null);
  const [selectedGroupThreadId, setSelectedGroupThreadId] = useState<string | null>(null);
  const [mobileView, setMobileView] = useState<"contacts" | "thread">("contacts");
  const [draft, setDraft] = useState("");
  const [contactRepairOpen, setContactRepairOpen] = useState(false);
  const [groupDialogOpen, setGroupDialogOpen] = useState(false);
  const [editingText, setEditingText] = useState<CharacterTextMessage | null>(null);
  const [deletingText, setDeletingText] = useState<CharacterTextMessage | null>(null);
  const [textActionError, setTextActionError] = useState("");
  const [localTextMessagesByThread, setLocalTextMessagesByThread] = useState<Record<string, LocalCharacterTextMessage[]>>({});
  const localTextMessageCounter = useRef(0);
  const persistedReadKeysRef = useRef<Set<string>>(new Set());
  const photoInputRef = useRef<HTMLInputElement | null>(null);
  const textMessagesRef = useRef<HTMLDivElement | null>(null);
  const textScrollMetricsRef = useRef<{ activeSaveId: string | null; threadKey: string | null; scrollHeight: number } | null>(null);
  const isMobilePhone = useMediaQuery(WORKBENCH_MOBILE_QUERY);
  const refreshTextQueries = useCallback((saveId: string | null = activeSaveId) => {
    client.invalidateQueries({ queryKey: ["character-texts", saveId] });
    client.invalidateQueries({ queryKey: ["character-text-thread", saveId] });
  }, [activeSaveId, client]);
  const canGenerateMedia = !disabled;
  const canManageMedia = canUseChildRestrictedControls(currentUser) && !disabled;
  const modelQuery = useQuery({
    queryKey: ["character-texts", activeSaveId],
    queryFn: ({ signal }) => apiRead<CharacterTextsModel>(characterTextsPath(activeSaveId), signal),
    enabled: Boolean(activeSaveId)
  });
  const textModel = modelQuery.data;
  const groupThreads = textModel?.threads.filter((thread) => thread.kind === "group") ?? [];
  const selectedGroupThread = groupThreads.find((thread) => thread.id === selectedGroupThreadId) ?? null;
  const selectedContact = selectedGroupThread ? null : textModel?.contacts.find((contact) => contact.id === selectedCharacterId)
    ?? textModel?.contacts[0]
    ?? null;
  const selectedConversationTitle = selectedGroupThread?.title?.trim()
    || (selectedContact ? displayCharacterTextContactName(selectedContact) : "Select a conversation");
  const repairContacts = textModel?.repair_contacts ?? textModel?.contacts ?? [];
  useEffect(() => {
    if (selectedGroupThread) return;
    if (selectedContact && selectedCharacterId !== selectedContact.id) setSelectedCharacterId(selectedContact.id);
    if (!selectedContact && selectedCharacterId) setSelectedCharacterId(null);
  }, [selectedCharacterId, selectedContact, selectedGroupThread]);
  useEffect(() => {
    if (!selectedGroupThreadId) return;
    if (!groupThreads.some((thread) => thread.id === selectedGroupThreadId)) {
      setSelectedGroupThreadId(null);
    }
  }, [groupThreads, selectedGroupThreadId]);
  useEffect(() => {
    if (selectedContact || selectedGroupThread || selectedCharacterId || selectedGroupThreadId) return;
    if (groupThreads[0]) setSelectedGroupThreadId(groupThreads[0].id);
  }, [groupThreads, selectedCharacterId, selectedContact, selectedGroupThread, selectedGroupThreadId]);
  useEffect(() => {
    setLocalTextMessagesByThread({});
    setTextActionError("");
    persistedReadKeysRef.current.clear();
  }, [activeSaveId]);
  const selectedThreadId = selectedGroupThread?.id ?? selectedContact?.thread_id ?? null;
  const selectedThreadKey = selectedThreadId ?? (selectedContact ? localCharacterTextThreadKey(selectedContact.id) : null);
  const unreadContacts = incomingCharacterTextContacts(
    textModel,
    seenTextMessageIdsByThread,
  );
  const unreadContactIds = new Set(unreadContacts.map((contact) => contact.id));
  const threadQuery = useQuery({
    queryKey: ["character-text-thread", activeSaveId, selectedThreadId],
    queryFn: ({ signal }) => apiRead<CharacterTextThread>(characterTextThreadPath(activeSaveId, selectedThreadId!), signal),
    enabled: Boolean(activeSaveId && selectedThreadId)
  });
  const serverThreadMessages = selectedThreadId ? threadQuery.data?.messages ?? [] : [];
  const localThreadMessages = selectedThreadKey ? localTextMessagesByThread[selectedThreadKey] ?? [] : [];
  const threadMessages = characterTextMessagesWithLocalEchoes(serverThreadMessages, localThreadMessages);
  const threadHasActiveDelivery = threadMessages.some((message) => isActiveCharacterTextDelivery(message));
  const selectedThreadHasActiveJob = Boolean(
    selectedThreadId && busyThreadIds.has(selectedThreadId),
  );
  const visibleThreadMessages = threadMessages.filter((message) => !isPlaceholderCharacterTextDelivery(message));
  const firstVisibleThreadMessage = visibleThreadMessages[0] ?? null;
  const latestVisibleThreadMessage = visibleThreadMessages[visibleThreadMessages.length - 1] ?? null;
  const visibleThreadMessageSignal = [
    activeSaveId ?? "",
    selectedThreadKey ?? "",
    visibleThreadMessages.length,
    firstVisibleThreadMessage?.id ?? "",
    latestVisibleThreadMessage?.id ?? "",
    latestVisibleThreadMessage?.delivery_status ?? "",
    latestVisibleThreadMessage?.revision_count ?? 0,
    latestVisibleThreadMessage?.body.length ?? 0
  ].join(":");
  const characterTextVirtualizer = useVirtualizer<HTMLDivElement, HTMLDivElement>({
    count: visibleThreadMessages.length,
    getScrollElement: () => textMessagesRef.current,
    estimateSize: () => CHARACTER_TEXT_ESTIMATED_ROW_HEIGHT,
    getItemKey: (index) => `${selectedThreadKey ?? ""}:${visibleThreadMessages[index]?.id ?? index}`,
    gap: CHARACTER_TEXT_ROW_GAP,
    initialOffset: () => initialVirtualBottomOffset(visibleThreadMessages.length, CHARACTER_TEXT_ESTIMATED_ROW_HEIGHT, CHARACTER_TEXT_ROW_GAP),
    initialRect: VIRTUAL_LIST_INITIAL_RECT,
    observeElementOffset: (_instance, callback) => observeVirtualElementOffset(textMessagesRef.current, callback),
    observeElementRect: (_instance, callback) => observeVirtualElementRect(textMessagesRef.current, callback),
    overscan: CHARACTER_TEXT_ROW_OVERSCAN,
    useFlushSync: false,
    measureElement: (element) => {
      const height = element.getBoundingClientRect().height;
      return height > 0 ? height : CHARACTER_TEXT_ESTIMATED_ROW_HEIGHT;
    }
  });
  const virtualTextMessageRows = characterTextVirtualizer.getVirtualItems();
  const showContacts = !isMobilePhone || mobileView === "contacts";
  const showThread = !isMobilePhone || mobileView === "thread";
  const textMessagesNearBottom = useCallback((node: HTMLElement, scrollHeight = node.scrollHeight) => {
    return scrollHeight - node.scrollTop - node.clientHeight <= 96;
  }, []);
  const scrollToLatestCharacterText = useCallback(() => {
    const node = textMessagesRef.current;
    if (!node) return;
    if (visibleThreadMessages.length) {
      characterTextVirtualizer.scrollToIndex(visibleThreadMessages.length - 1, { align: "end" });
    }
    const scrollHeight = Math.max(node.scrollHeight, characterTextVirtualizer.getTotalSize());
    setScrollTopAndNotify(node, scrollHeight);
    textScrollMetricsRef.current = {
      activeSaveId,
      threadKey: selectedThreadKey,
      scrollHeight
    };
  }, [activeSaveId, characterTextVirtualizer, selectedThreadKey, visibleThreadMessages.length]);
  const persistThreadRead = useCallback((threadId: string | null | undefined, messageId: string | null | undefined) => {
    if (!activeSaveId || !threadId || !messageId) return;
    const readKey = `${activeSaveId}:${threadId}:${messageId}`;
    if (persistedReadKeysRef.current.has(readKey)) return;
    persistedReadKeysRef.current.add(readKey);
    void postJson<unknown>(
      characterTextThreadReadPath(threadId),
      { save_id: activeSaveId, through_message_id: messageId }
    ).then((result) => {
      applyCharacterTextJobResult(client, result, activeSaveId);
    }).catch(() => {
      persistedReadKeysRef.current.delete(readKey);
    });
  }, [activeSaveId, client]);
  const markContactThreadSeen = useCallback((contact: CharacterTextContact) => {
    onThreadSeen(contact.thread_id, contact.latest_message_id);
    persistThreadRead(contact.thread_id, contact.latest_message_id);
  }, [onThreadSeen, persistThreadRead]);
  const markGroupThreadSeen = useCallback((thread: CharacterTextThread, messages: CharacterTextMessage[] = []) => {
    const latestMessageId = lastCharacterTextMessageId(messages);
    onThreadSeen(thread.id, latestMessageId);
    persistThreadRead(thread.id, latestMessageId);
  }, [onThreadSeen, persistThreadRead]);
  const selectContact = (contact: CharacterTextContact) => {
    setSelectedGroupThreadId(null);
    setSelectedCharacterId(contact.id);
    markContactThreadSeen(contact);
    if (isMobilePhone) setMobileView("thread");
  };
  const selectGroupThread = (thread: CharacterTextThread) => {
    setSelectedGroupThreadId(thread.id);
    setSelectedCharacterId(null);
    markGroupThreadSeen(thread, thread.id === selectedThreadId ? serverThreadMessages : thread.messages ?? []);
    if (isMobilePhone) setMobileView("thread");
  };
  useEffect(() => {
    if (selectedGroupThread || !selectedContact || !showThread) return;
    markContactThreadSeen(selectedContact);
  }, [markContactThreadSeen, selectedContact, selectedGroupThread, showThread]);
  useEffect(() => {
    if (!selectedGroupThread || !showThread) return;
    markGroupThreadSeen(selectedGroupThread, serverThreadMessages);
  }, [markGroupThreadSeen, selectedGroupThread, serverThreadMessages, showThread]);
  useEffect(() => {
    if (!selectedThreadKey || !threadQuery.data) return;
    setLocalTextMessagesByThread((current) => pruneAcceptedLocalCharacterTexts(
      current,
      selectedThreadKey,
      threadQuery.data.messages,
    ));
  }, [selectedThreadKey, threadQuery.data]);
  useLayoutEffect(() => {
    if (!showThread) {
      textScrollMetricsRef.current = null;
      return undefined;
    }
    const node = textMessagesRef.current;
    if (!node) return undefined;
    const previousMetrics = textScrollMetricsRef.current;
    const threadChanged = previousMetrics?.activeSaveId !== activeSaveId
      || previousMetrics?.threadKey !== selectedThreadKey;
    const wasNearBottom = previousMetrics
      ? textMessagesNearBottom(node, previousMetrics.scrollHeight)
      : true;
    if (threadChanged || wasNearBottom) {
      scrollToLatestCharacterText();
      const frame = window.requestAnimationFrame(scrollToLatestCharacterText);
      return () => {
        window.cancelAnimationFrame(frame);
      };
    }
    textScrollMetricsRef.current = {
      activeSaveId,
      threadKey: selectedThreadKey,
      scrollHeight: node.scrollHeight
    };
    return undefined;
  }, [
    activeSaveId,
    scrollToLatestCharacterText,
    selectedThreadKey,
    showThread,
    textMessagesNearBottom,
    visibleThreadMessageSignal
  ]);
  const sendText = useMutation({
    mutationFn: async (variables: CharacterTextSendVariables) => {
      const form = new FormData();
      form.append("save_id", variables.saveId);
      form.append("body", variables.body);
      if (variables.photo) form.append("file", variables.photo);
      if (variables.isGroupThread) {
        return api<Job>(
          `/api/character-texts/threads/${encodeURIComponent(variables.threadId)}/send-image`,
          { method: "POST", body: form }
        );
      }
      if (!variables.characterId) throw new Error("No contact selected");
      form.append("character_id", variables.characterId);
      return api<Job>("/api/character-texts/send-image", {
        method: "POST",
        body: form
      });
    },
    onMutate: (variables) => {
      setTextActionError("");
      const localMessage: LocalCharacterTextMessage = {
        id: variables.localId,
        thread_id: variables.threadId,
        character_id: variables.characterId,
        sender: "player",
        sender_character_id: null,
        sender_display_name: "You",
        body: variables.body,
        delivery_status: "pending",
        markdown_blocks: [{ kind: "paragraph", spans: [{ kind: "text", text: variables.body }] }],
        local_after_message_id: variables.afterMessageId,
      };
      setLocalTextMessagesByThread((current) => ({
        ...current,
        [variables.threadKey]: [
          ...(current[variables.threadKey] ?? []),
          localMessage,
        ],
      }));
      setDraft("");
      if (photoInputRef.current) photoInputRef.current.value = "";
    },
    onSuccess: (job, variables) => {
      refreshTextQueries(variables.saveId);
      runJob(job, { applyResult: false, clearPendingMessages: false });
    },
    onError: (error, variables) => {
      setLocalTextMessagesByThread((current) => markLocalCharacterTextFailed(
        current,
        variables.threadKey,
        variables.localId,
        error instanceof Error ? error.message : "Text delivery failed",
      ));
      refreshTextQueries(variables.saveId);
    }
  });
  const requestSpontaneousText = useMutation({
    mutationFn: async (variables: CharacterTextSpontaneousVariables) => {
      return postJson<Job>("/api/character-texts/spontaneous", {
        save_id: variables.saveId,
        character_id: variables.characterId
      });
    },
    onMutate: () => {
      setTextActionError("");
    },
    onSuccess: (job, variables) => {
      refreshTextQueries(variables.saveId);
      runJob(job, {
        applyResult: false,
        clearPendingMessages: false,
        onSucceeded: () => refreshTextQueries(variables.saveId)
      });
    },
    onError: (_error, variables) => {
      refreshTextQueries(variables.saveId);
    }
  });
  const updateContactState = useMutation({
    mutationFn: async (variables: {
      contact: CharacterTextContact;
      playerHasCharacterNumber: boolean;
      characterHasPlayerNumber: boolean;
    }) => {
      if (!activeSaveId) throw new Error("No save loaded");
      const model = await postJson<CharacterTextsModel>(
        characterTextContactPath(variables.contact.id),
        {
          save_id: activeSaveId,
          player_has_character_number: variables.playerHasCharacterNumber,
          character_has_player_number: variables.characterHasPlayerNumber
        }
      );
      return { model, saveId: activeSaveId };
    },
    onSuccess: ({ model, saveId }, variables) => {
      client.setQueryData(["character-texts", saveId], model);
      if (variables.playerHasCharacterNumber) {
        setSelectedCharacterId(variables.contact.id);
        if (isMobilePhone) setMobileView("thread");
        return;
      }
      if (selectedCharacterId === variables.contact.id) {
        setSelectedCharacterId(model.contacts[0]?.id ?? null);
        if (isMobilePhone) setMobileView("contacts");
      }
    }
  });
  const updateRepairContactState = (contact: CharacterTextContact, next: {
    playerHasCharacterNumber?: boolean;
    characterHasPlayerNumber?: boolean;
  }) => {
    updateContactState.mutate({
      contact,
      playerHasCharacterNumber: next.playerHasCharacterNumber
        ?? contact.player_has_character_number,
      characterHasPlayerNumber: next.characterHasPlayerNumber
        ?? contact.character_has_player_number
    });
  };
  const displayContactName = displayCharacterTextContactName;
  const selectedContactTextable = Boolean(selectedGroupThread || selectedContact?.player_has_character_number);
  const selectedContactCanTextPlayer = Boolean(selectedContact?.character_has_player_number);
  const composerUnavailable = Boolean(
    !(selectedContact || selectedGroupThread)
    || !selectedContactTextable
    || disabled
  );
  const threadMutationDisabled = Boolean(
    composerUnavailable
    || sendText.isPending
    || threadHasActiveDelivery
    || selectedThreadHasActiveJob
  );
  const canSend = Boolean(
    activeSaveId
    && draft.trim()
    && !threadMutationDisabled
  );
  const canRequestSpontaneousText = Boolean(
    activeSaveId
    && selectedContact
    && !selectedGroupThread
    && selectedContactCanTextPlayer
    && !threadMutationDisabled
    && !requestSpontaneousText.isPending
  );
  const spontaneousTextLabel = selectedContact
    ? `Ask ${displayContactName(selectedContact)} to text you`
    : "Ask character to text you";
  let composerPlaceholder = "Select a contact";
  if (selectedGroupThread) {
    composerPlaceholder = `Message ${selectedConversationTitle}`;
  } else if (selectedContact) {
    composerPlaceholder = selectedContactTextable
      ? `Message ${displayContactName(selectedContact)}`
      : "You do not have their number";
  }
  const submitDraft = () => {
    if (!canSend || !activeSaveId || !selectedThreadKey) return;
    localTextMessageCounter.current += 1;
      sendText.mutate({
        saveId: activeSaveId,
        characterId: selectedContact?.id ?? null,
        threadKey: selectedThreadKey,
        threadId: selectedThreadId ?? selectedThreadKey,
        isGroupThread: Boolean(selectedGroupThread),
        body: draft.trim(),
        photo: photoInputRef.current?.files?.[0],
        localId: `local-character-text-${localTextMessageCounter.current}`,
        afterMessageId: lastCharacterTextMessageId(serverThreadMessages)
      });
  };
  const handleCharacterTextAction = useCallback((actionId: string, selectedMessage: CharacterTextMessage) => {
    if (threadMutationDisabled) return;
    if (actionId === "delete-text-messages-from-here") {
      setDeletingText(selectedMessage);
      return;
    }
    setEditingText(selectedMessage);
  }, [threadMutationDisabled]);
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog character-text-phone" titleId={titleId} onClose={onClose}>
        <div className="dialog-title-row character-text-phone-title">
          <div>
            <p className="eyebrow">Messages</p>
            <h2 id={titleId}>Phone</h2>
          </div>
          <button type="button" className="icon-button" aria-label="Close phone" title="Close" onClick={onClose}>
            <X size={16} aria-hidden="true" />
          </button>
        </div>
        {modelQuery.isLoading ? (
          <div className="empty-panel"><Loader2 className="spin" size={18} aria-hidden="true" /></div>
        ) : modelQuery.error instanceof Error ? (
          <InlineNotice>{modelQuery.error.message}</InlineNotice>
        ) : !textModel?.enabled ? (
          <InlineNotice>Character texts are not enabled for this save.</InlineNotice>
        ) : (
          <div className={showContacts && showThread ? "character-text-layout" : "character-text-layout single-pane"}>
            {showContacts ? (
              <section className="character-text-inbox" aria-label="Text contacts">
                <div className="character-text-inbox-header">
                  <div className="character-text-inbox-heading">
                    <strong>Contacts</strong>
                    <span>{textModel.contacts.length + groupThreads.length}</span>
                  </div>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="New group"
                    title="New group"
                    onClick={() => setGroupDialogOpen(true)}
                  >
                    <Users size={16} aria-hidden="true" />
                  </button>
                  <button
                    type="button"
                    className="icon-button"
                    aria-label="Add contact"
                    title="Add contact"
                    onClick={() => setContactRepairOpen(true)}
                  >
                    <Plus size={16} aria-hidden="true" />
                  </button>
                </div>
                <div className="character-text-contact-list">
                  {groupThreads.map((thread) => {
                    const displayName = thread.title?.trim() || "Group";
                    const selected = thread.id === selectedGroupThread?.id;
                    return (
                      <button
                        key={thread.id}
                        type="button"
                        aria-label={`Open text thread for ${displayName}`}
                        aria-pressed={selected}
                        className={selected ? "selected" : ""}
                        onClick={() => selectGroupThread(thread)}
                      >
                        <CharacterTextAvatar
                          name={displayName}
                          referenceImage={null}
                          activeSaveId={activeSaveId}
                          size="default"
                        />
                        <div className="character-text-contact-copy">
                          <strong>
                            <span>{displayName}</span>
                          </strong>
                          <span className="character-text-contact-permission">
                            {groupThreadParticipantNames(thread)}
                          </span>
                        </div>
                      </button>
                    );
                  })}
                  {textModel.contacts.length ? textModel.contacts.map((contact) => {
                    const displayName = displayContactName(contact);
                    return (
                      <button
                        key={contact.id}
                        type="button"
                        aria-label={`Open text thread for ${displayName}`}
                        aria-pressed={contact.id === selectedContact?.id}
                        className={[
                          contact.id === selectedContact?.id ? "selected" : "",
                          unreadContactIds.has(contact.id) ? "unread" : ""
                        ].filter(Boolean).join(" ")}
                        onClick={() => selectContact(contact)}
                      >
                        <CharacterTextAvatar
                          name={displayName}
                          referenceImage={contact.reference_image}
                          activeSaveId={activeSaveId}
                          size="default"
                        />
                        <div className="character-text-contact-copy">
                          <strong>
                            <span>{displayName}</span>
                            {unreadContactIds.has(contact.id) ? <small>New</small> : null}
                          </strong>
                          {contact.latest_message_body ? (
                            <MarkdownView
                              body={contact.latest_message_body}
                              markdownBlocks={contact.latest_message_markdown_blocks}
                              className="character-text-contact-preview"
                            />
                          ) : null}
                          <span className="character-text-contact-permission">
                            {characterTextContactPermissionSummary(contact)}
                          </span>
                        </div>
                      </button>
                    );
                  }) : null}
                  {!textModel.contacts.length && !groupThreads.length ? <p className="muted">No contacts.</p> : null}
                </div>
              </section>
            ) : null}
            {showThread ? (
              <section className="character-text-thread" aria-label={selectedConversationTitle ? `Conversation with ${selectedConversationTitle}` : "Text conversation"}>
                <div className="character-text-thread-header">
                  {isMobilePhone ? (
                    <button type="button" className="icon-button" aria-label="Back to contacts" title="Contacts" onClick={() => setMobileView("contacts")}>
                      <ChevronLeft size={17} aria-hidden="true" />
                    </button>
                  ) : null}
                  <div className="character-text-thread-identity">
                    <CharacterTextAvatar
                      name={selectedConversationTitle}
                      referenceImage={selectedContact?.reference_image}
                      activeSaveId={activeSaveId}
                      size="large"
                    />
                    <div>
                      <strong>{selectedConversationTitle}</strong>
                      {selectedGroupThread ? (
                        <small>{groupThreadParticipantNames(selectedGroupThread)}</small>
                      ) : selectedContact ? (
                        <small>{characterTextContactPermissionSummary(selectedContact)}</small>
                      ) : null}
                    </div>
                  </div>
                  <div className="character-text-thread-header-actions">
                    {threadHasActiveDelivery ? (
                      <span className="character-text-pending">Sending</span>
                    ) : selectedThreadHasActiveJob ? (
                      <span className="character-text-pending">Finishing…</span>
                    ) : null}
                    {!selectedGroupThread ? (
                      <button
                        type="button"
                        className="icon-button ct-spontaneous"
                        aria-label={spontaneousTextLabel}
                        title={spontaneousTextLabel}
                        disabled={!canRequestSpontaneousText}
                        onClick={() => {
                          if (!canRequestSpontaneousText || !activeSaveId || !selectedContact) return;
                          requestSpontaneousText.mutate({
                            saveId: activeSaveId,
                            characterId: selectedContact.id
                          });
                        }}
                      >
                        {requestSpontaneousText.isPending ? (
                          <Loader2 className="spin" size={16} aria-hidden="true" />
                        ) : (
                          <MessageSquareText size={16} aria-hidden="true" />
                        )}
                      </button>
                    ) : null}
                  </div>
                </div>
                <div className="character-text-messages" ref={textMessagesRef}>
                  {threadQuery.isFetching && selectedThreadId && !visibleThreadMessages.length ? (
                    <span className="muted">Loading...</span>
                  ) : visibleThreadMessages.length ? (
                    <div
                      className="character-text-virtual-list"
                      style={{ height: `${characterTextVirtualizer.getTotalSize()}px` }}
                    >
                      {virtualTextMessageRows.map((virtualRow) => {
                        const message = visibleThreadMessages[virtualRow.index];
                        if (!message) return null;
                        const previousMessage = virtualRow.index > 0 ? visibleThreadMessages[virtualRow.index - 1] : null;
                        return (
                          <div
                            key={virtualRow.key}
                            ref={characterTextVirtualizer.measureElement}
                            className="character-text-virtual-row"
                            data-index={virtualRow.index}
                            style={{ transform: `translateY(${virtualRow.start}px)` }}
                          >
                            <CharacterTextMessageRow
                              message={message}
                              previousMessage={previousMessage}
                              contact={selectedContact}
                              activeSaveId={activeSaveId}
                              runJob={runJob}
                              canGenerateMedia={canGenerateMedia}
                              mutationsDisabled={threadMutationDisabled}
                              onAction={handleCharacterTextAction}
                            />
                          </div>
                        );
                      })}
                    </div>
                  ) : (
                    <span className="muted">No messages yet.</span>
                  )}
                  {threadHasActiveDelivery && (selectedContact || selectedGroupThread) ? (
                    <div className="character-text-typing" role="status">
                      <span>{selectedGroupThread ? "Waiting for replies..." : `${displayContactName(selectedContact)} is typing...`}</span>
                    </div>
                  ) : null}
                </div>
                <form
                  className="character-text-compose"
                  onSubmit={(event) => {
                    event.preventDefault();
                    submitDraft();
                  }}
                >
                  {canManageMedia && (
                    <input
                      ref={photoInputRef}
                      type="file"
                      aria-label="Photo"
                      accept="image/*"
                      className="ct-photo"
                      disabled={composerUnavailable}
                    />
                  )}
                  <textarea
                    aria-label={selectedConversationTitle ? `Message ${selectedConversationTitle}` : "Message"}
                    value={draft}
                    disabled={composerUnavailable}
                    onChange={(event) => setDraft(event.currentTarget.value)}
                    onKeyDown={(event) => {
                      if (event.key !== "Enter" || event.shiftKey) return;
                      event.preventDefault();
                      submitDraft();
                    }}
                    placeholder={composerPlaceholder}
                    rows={1}
                  />
                  <button type="submit" className="primary-command compact ct-send" aria-label="Send text" disabled={!canSend}>
                    {sendText.isPending ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <Send size={15} aria-hidden="true" />}
                    <span>Send</span>
                  </button>
                </form>
                {sendText.error instanceof Error ? <InlineNotice>{sendText.error.message}</InlineNotice> : null}
                {requestSpontaneousText.error instanceof Error ? <InlineNotice>{requestSpontaneousText.error.message}</InlineNotice> : null}
                {textActionError ? <InlineNotice>{textActionError}</InlineNotice> : null}
              </section>
            ) : null}
          </div>
        )}
        {contactRepairOpen && textModel?.enabled ? (
          <CharacterTextAddContactDialog
            activeSaveId={activeSaveId}
            contacts={repairContacts}
            updateError={updateContactState.error instanceof Error ? updateContactState.error : null}
            updatePending={updateContactState.isPending}
            onUpdateContact={updateRepairContactState}
            onClose={() => setContactRepairOpen(false)}
          />
        ) : null}
        {groupDialogOpen && textModel?.enabled ? (
          <CharacterTextGroupDialog
            activeSaveId={activeSaveId}
            contacts={textModel.contacts}
            onCreated={(thread) => {
              client.setQueryData<CharacterTextsModel>(["character-texts", activeSaveId], (current) => (
                current ? {
                  ...current,
                  threads: current.threads.some((existing) => existing.id === thread.id)
                    ? current.threads.map((existing) => existing.id === thread.id ? thread : existing)
                    : [thread, ...current.threads]
                } : current
              ));
              setSelectedGroupThreadId(thread.id);
              setSelectedCharacterId(null);
              setGroupDialogOpen(false);
              if (isMobilePhone) setMobileView("thread");
            }}
            onClose={() => setGroupDialogOpen(false)}
          />
        ) : null}
        {editingText ? (
          <EditCharacterTextModal
            message={editingText}
            activeSaveId={activeSaveId}
            runJob={runJob}
            disabled={threadMutationDisabled}
            onClose={() => setEditingText(null)}
            onStarted={() => {
              setTextActionError("");
              refreshTextQueries();
            }}
            onFailed={(error) => {
              setTextActionError(error);
              refreshTextQueries();
            }}
          />
        ) : null}
        {deletingText ? (
          <ConfirmModal
            title="Delete from here?"
            body="This hides the selected text and every later text in this conversation from future phone context."
            confirmLabel="Delete from here"
            destructive
            disabled={threadMutationDisabled}
            onCancel={() => setDeletingText(null)}
            onConfirm={async () => {
              if (threadMutationDisabled) return;
              const job = await postJson<Job>("/api/character-texts/delete-from-here", {
                save_id: activeSaveId,
                text_message_id: deletingText.id
              });
              setDeletingText(null);
              refreshTextQueries();
              runJob(job, {
                applyResult: false,
                clearPendingMessages: false,
                onSucceeded: () => refreshTextQueries()
              });
            }}
          />
        ) : null}
      </DialogPanel>
    </ModalBackdrop>
  );
}

function EditCharacterTextModal({
  message,
  activeSaveId,
  runJob,
  disabled,
  onClose,
  onStarted,
  onFailed
}: {
  message: CharacterTextMessage;
  activeSaveId: string | null;
  runJob: RunJob;
  disabled: boolean;
  onClose: () => void;
  onStarted: () => void;
  onFailed: (error: string) => void;
}) {
  const [body, setBody] = useState(message.body);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const titleId = React.useId();
  const isCharacterCorrection = message.sender === "character";
  const unchanged = body.trim() === message.body.trim();
  const submitEdit = async (mode: "save" | "resubmit") => {
    if (disabled) return;
    setBusy(true);
    setError("");
    try {
      const endpoint = mode === "resubmit"
        ? "/api/character-texts/edit"
        : "/api/character-texts/message-edit";
      const job = await postJson<Job>(endpoint, {
        save_id: activeSaveId,
        text_message_id: message.id,
        body
      });
      runJob(job, {
        applyResult: false,
        clearPendingMessages: false,
        onSucceeded: onStarted,
        onFailed,
      });
      onStarted();
      onClose();
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not submit text edit");
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
          await submitEdit(isCharacterCorrection ? "save" : "resubmit");
        }}
      >
        <header>
          <h2 id={titleId}>Edit text</h2>
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
          {isCharacterCorrection ? (
            <button type="submit" className="primary-command compact" disabled={disabled || busy || !body.trim() || unchanged}>
              <Save size={15} /> Save
            </button>
          ) : (
            <>
              <button type="button" className="primary-command compact" disabled={disabled || busy || !body.trim() || unchanged} onClick={() => submitEdit("save")}>
                <Save size={15} /> Edit without Resubmit
              </button>
              <button type="submit" className="primary-command compact" disabled={disabled || busy || !body.trim() || unchanged}>
                <RefreshCw size={15} /> Resubmit
              </button>
            </>
          )}
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function CharacterTextGroupDialog({
  activeSaveId,
  contacts,
  onCreated,
  onClose
}: {
  activeSaveId: string | null;
  contacts: CharacterTextContact[];
  onCreated: (thread: CharacterTextThread) => void;
  onClose: () => void;
}) {
  const titleId = React.useId();
  const textableContacts = contacts.filter((contact) => contact.player_has_character_number);
  const [title, setTitle] = useState("");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(() => new Set(textableContacts.slice(0, 2).map((contact) => contact.id)));
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const selectedCount = selectedIds.size;
  const submit = async () => {
    if (!activeSaveId || selectedCount < 2) return;
    setBusy(true);
    setError("");
    try {
      const result = await postJson<{ thread: CharacterTextThread }>("/api/character-texts/groups", {
        save_id: activeSaveId,
        title,
        character_ids: Array.from(selectedIds)
      });
      onCreated(result.thread);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Could not create group");
      setBusy(false);
    }
  };
  return (
    <ModalBackdrop>
      <DialogForm
        className="preview-dialog character-text-add-contact-dialog"
        titleId={titleId}
        onClose={onClose}
        onSubmit={async (event) => {
          event.preventDefault();
          await submit();
        }}
      >
        <header>
          <h2 id={titleId}>New group</h2>
          <button type="button" onClick={onClose} title="Close" aria-label="Close">
            <X size={16} />
          </button>
        </header>
        <label className="field-label">
          <span>Name</span>
          <input value={title} onChange={(event) => setTitle(event.currentTarget.value)} />
        </label>
        <div className="character-text-repair-list">
          {textableContacts.map((contact) => {
            const displayName = displayCharacterTextContactName(contact);
            const checked = selectedIds.has(contact.id);
            return (
              <label key={contact.id} className="character-text-repair-row character-text-group-row">
                <CharacterTextAvatar
                  name={displayName}
                  referenceImage={contact.reference_image}
                  activeSaveId={activeSaveId}
                  size="default"
                />
                <span>{displayName}</span>
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={(event) => {
                    setSelectedIds((current) => {
                      const next = new Set(current);
                      if (event.currentTarget.checked) next.add(contact.id);
                      else next.delete(contact.id);
                      return next;
                    });
                  }}
                />
              </label>
            );
          })}
          {!textableContacts.length ? <p className="muted">No textable contacts.</p> : null}
        </div>
        {error ? <InlineNotice>{error}</InlineNotice> : null}
        <div className="command-row end">
          <button type="button" onClick={onClose}>Cancel</button>
          <button type="submit" className="primary-command compact" disabled={busy || selectedCount < 2}>
            {busy ? <Loader2 className="spin" size={15} aria-hidden="true" /> : <Users size={15} aria-hidden="true" />}
            <span>Create</span>
          </button>
        </div>
      </DialogForm>
    </ModalBackdrop>
  );
}

function CharacterTextAddContactDialog({
  activeSaveId,
  contacts,
  updateError,
  updatePending,
  onUpdateContact,
  onClose
}: {
  activeSaveId: string | null;
  contacts: CharacterTextContact[];
  updateError: Error | null;
  updatePending: boolean;
  onUpdateContact: (contact: CharacterTextContact, next: {
    playerHasCharacterNumber?: boolean;
    characterHasPlayerNumber?: boolean;
  }) => void;
  onClose: () => void;
}) {
  const titleId = "character-text-add-contact-title";
  const [search, setSearch] = useState("");
  const query = search.trim().toLowerCase();
  const visibleContacts = contacts.filter((contact) => {
    if (!query) return true;
    const haystack = `${contact.contact_name ?? ""} ${contact.name} ${contact.role} ${contact.status}`.toLowerCase();
    return haystack.includes(query);
  });
  const controlsDisabled = !activeSaveId || updatePending;
  const searchTargetName = (contact: CharacterTextContact): string =>
    contact.contact_name?.trim() || contact.name;
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog character-text-add-contact-dialog" titleId={titleId} onClose={onClose}>
        <div className="dialog-title-row">
          <div>
            <p className="eyebrow">Phone</p>
            <h2 id={titleId}>Add contact</h2>
          </div>
          <button type="button" className="icon-button" aria-label="Close Add contact" title="Close" onClick={onClose}>
            <X size={16} aria-hidden="true" />
          </button>
        </div>
        <div className="inline-tool-form character-text-contact-search">
          <Search size={15} aria-hidden="true" />
          <input
            value={search}
            onChange={(event) => setSearch(event.currentTarget.value)}
            placeholder="Search contacts"
            aria-label="Search contacts"
          />
        </div>
        <div className="character-text-repair-list">
          {visibleContacts.length ? visibleContacts.map((contact) => {
            const displayName = searchTargetName(contact);
            return (
              <div className="character-text-repair-row" key={contact.id}>
                <CharacterTextAvatar
                  name={displayName}
                  referenceImage={contact.reference_image}
                  activeSaveId={activeSaveId}
                  size="default"
                />
                <span className="character-text-contact-copy">
                  <strong>{displayName}</strong>
                  <span>{characterTextContactPermissionSummary(contact)}</span>
                </span>
                <div className="character-text-repair-controls">
                  <label>
                    <input
                      type="checkbox"
                      aria-label={`You can text ${displayName}`}
                      checked={contact.player_has_character_number}
                      disabled={controlsDisabled}
                      onChange={(event) => onUpdateContact(contact, {
                        playerHasCharacterNumber: event.currentTarget.checked
                      })}
                    />
                    <span>You can text them</span>
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      aria-label={`${displayName} can text you`}
                      checked={contact.character_has_player_number}
                      disabled={controlsDisabled}
                      onChange={(event) => onUpdateContact(contact, {
                        characterHasPlayerNumber: event.currentTarget.checked
                      })}
                    />
                    <span>They can text you</span>
                  </label>
                </div>
                <div className="character-text-permission-details">
                  <span>{playerNumberPermissionReason(contact)}</span>
                  <span>{characterNumberPermissionReason(contact)}</span>
                </div>
              </div>
            );
          }) : <p className="muted">No matching contacts.</p>}
        </div>
        {updateError ? <InlineNotice>{updateError.message}</InlineNotice> : null}
      </DialogPanel>
    </ModalBackdrop>
  );
}


function CharacterTextAvatar({
  name,
  referenceImage,
  activeSaveId,
  size
}: {
  name: string;
  referenceImage: CharacterReferenceImage | null | undefined;
  activeSaveId: string | null;
  size: "default" | "large";
}) {
  const [imageFailed, setImageFailed] = useState(false);
  const showImage = Boolean(referenceImage?.media_asset_id) && !imageFailed;
  const initials = name.trim().slice(0, 1).toUpperCase() || "?";
  const className = size === "large" ? "character-text-avatar large" : "character-text-avatar";
  if (showImage) {
    return (
      <span className={className} aria-hidden="true">
        <img
          src={mediaAssetThumbnailPath(referenceImage!.media_asset_id, activeSaveId)}
          alt=""
          className="character-text-avatar-image"
          loading="lazy"
          decoding="async"
          onError={() => setImageFailed(true)}
        />
      </span>
    );
  }
  return (
    <span className={className} aria-hidden="true">
      <span className="character-text-avatar-initials">{initials}</span>
    </span>
  );
}

const CharacterTextBubble = React.memo(function CharacterTextBubble({
  message,
  contact,
  activeSaveId,
  runJob,
  canGenerateMedia,
  mutationsDisabled,
  grouped,
  onAction
}: {
  message: CharacterTextMessage;
  contact: CharacterTextContact | null;
  activeSaveId: string | null;
  runJob: RunJob;
  canGenerateMedia: boolean;
  mutationsDisabled: boolean;
  grouped?: boolean;
  onAction?: (actionId: string, message: CharacterTextMessage) => void;
}) {
  const fromPlayer = message.sender === "player";
  const delivery = fromPlayer || message.delivery_status === "failed"
    ? characterTextDeliveryView(message)
    : null;
  const actions = message.actions ?? [];
  const attachments = message.attachments ?? [];
  const timeLabel = characterTextMessageTimeLabel(message);
  const proactiveReason = message.proactive_reason?.trim() ?? "";
  const speakerName = message.sender_display_name?.trim()
    || (fromPlayer ? "You" : contact?.name ?? "Character");
  const [showProactiveReason, setShowProactiveReason] = useState(false);
  const [previewAttachment, setPreviewAttachment] = useState<CharacterTextAttachment | null>(null);
  const showHeader = !grouped || Boolean(actions.length) || Boolean(message.revision_count) || Boolean(proactiveReason);
  return (
    <div className={[
      "character-text-bubble",
      fromPlayer ? "from-player" : "from-character",
      grouped ? "grouped" : ""
    ].filter(Boolean).join(" ")}>
      {showHeader ? (
        <div className="character-text-bubble-header">
          {!grouped ? <span>{speakerName}</span> : null}
          {message.revision_count ? <small className="character-text-edited">Edited</small> : null}
          <div className="character-text-actions">
            {proactiveReason ? (
              <button
                type="button"
                className={touchActionClassName("character-text-reason-action")}
                title={showProactiveReason ? "Hide why this text arrived" : "Show why this text arrived"}
                aria-label={showProactiveReason ? "Hide why this text arrived" : "Show why this text arrived"}
                onClick={() => setShowProactiveReason((current) => !current)}
              >
                <TouchActionContents icon={<Info size={14} aria-hidden="true" />} label="Why" />
              </button>
            ) : null}
            {actions.length && onAction ? actions.map((action) => (
              <button
                key={action.action_id}
                type="button"
                className={touchActionClassName(action.action_id === "delete-text-messages-from-here" && "destructive-action")}
                title={action.label}
                aria-label={action.label}
                disabled={mutationsDisabled}
                onClick={() => onAction(action.action_id, message)}
              >
                <TouchActionContents icon={actionIcon(action.action_id)} label={action.label} />
              </button>
            )) : null}
          </div>
        </div>
      ) : null}
      <MarkdownView body={message.body} markdownBlocks={message.markdown_blocks} className="character-text-bubble-body" />
      {showProactiveReason ? (
        <small className="character-text-proactive-reason">{proactiveReason}</small>
      ) : null}
      {attachments.length ? (
        <div className="character-text-attachments">
          {attachments.map((attachment) => (
            <CharacterTextAttachmentView
              key={attachment.id}
              attachment={attachment}
              activeSaveId={activeSaveId}
              onPreview={setPreviewAttachment}
            />
          ))}
        </div>
      ) : null}
      {timeLabel || delivery ? (
        <small className="character-text-message-meta">
          {timeLabel ? <span>{timeLabel}</span> : null}
          {delivery ? (
            <span className={`character-text-delivery ${delivery.kind}`}>{delivery.label}</span>
          ) : null}
        </small>
      ) : null}
      {message.delivery_status === "failed" && message.delivery_error ? (
        <small className="character-text-delivery-error">{message.delivery_error}</small>
      ) : null}
      {previewAttachment ? (
        <CharacterTextAttachmentPreview
          attachment={previewAttachment}
          activeSaveId={activeSaveId}
          runJob={runJob}
          canGenerateMedia={canGenerateMedia}
          onClose={() => setPreviewAttachment(null)}
        />
      ) : null}
    </div>
  );
});

function CharacterTextAttachmentView({
  attachment,
  activeSaveId,
  onPreview
}: {
  attachment: CharacterTextAttachment;
  activeSaveId: string | null;
  onPreview: (attachment: CharacterTextAttachment) => void;
}) {
  const label = characterTextAttachmentLabel(attachment);
  if (attachment.status !== "succeeded" || !attachment.media_asset_id) {
    return (
      <div className="character-text-attachment failed" role="status">
        <FileWarning size={15} aria-hidden="true" />
        <span>{attachment.error || "Image unavailable"}</span>
      </div>
    );
  }
  return (
    <button
      type="button"
      className="character-text-attachment image"
      aria-label={`Open ${label}`}
      title={label}
      onClick={() => onPreview(attachment)}
    >
      <img
        src={mediaAssetThumbnailPath(attachment.media_asset_id, activeSaveId)}
        alt={attachment.prompt_preview || label}
        loading="lazy"
        decoding="async"
      />
    </button>
  );
}

function CharacterTextAttachmentPreview({
  attachment,
  activeSaveId,
  runJob,
  canGenerateMedia,
  onClose
}: {
  attachment: CharacterTextAttachment;
  activeSaveId: string | null;
  runJob: RunJob;
  canGenerateMedia: boolean;
  onClose: () => void;
}) {
  const titleId = React.useId();
  const label = characterTextAttachmentLabel(attachment);
  const [regeneratePromptOpen, setRegeneratePromptOpen] = useState(false);
  if (!attachment.media_asset_id) return null;
  const attachmentType = attachment.mime_type?.startsWith("image/") ? "image" : "other";
  const canRegenerate = canGenerateMedia && canRegenerateImageAsset({
    type: attachmentType,
    mime_type: attachment.mime_type ?? "",
    provider: attachment.provider ?? undefined,
    status: attachment.status,
    file_available: true
  });
  return (
    <ModalBackdrop>
      <DialogPanel className="preview-dialog character-text-attachment-preview" titleId={titleId} onClose={onClose}>
        <header>
          <h2 id={titleId}>{label}</h2>
          <button type="button" className="icon-button" aria-label="Close image preview" title="Close" onClick={onClose}>
            <X size={16} aria-hidden="true" />
          </button>
        </header>
        <img src={mediaAssetPath(attachment.media_asset_id, activeSaveId)} alt={attachment.prompt_preview || label} />
        {canRegenerate ? (
          <div className="command-row end">
            <button
              type="button"
              disabled={!activeSaveId}
              onClick={() => setRegeneratePromptOpen(true)}
            >
              <RefreshCw size={15} aria-hidden="true" /> Regenerate with edits
            </button>
          </div>
        ) : null}
      </DialogPanel>
      {regeneratePromptOpen && canRegenerate ? (
        <RegeneratePromptDialog
          assetId={attachment.media_asset_id}
          activeSaveId={activeSaveId}
          onCancel={() => setRegeneratePromptOpen(false)}
          onStarted={(job) => {
            runJob(job);
            setRegeneratePromptOpen(false);
            onClose();
          }}
        />
      ) : null}
    </ModalBackdrop>
  );
}

function characterTextAttachmentLabel(attachment: CharacterTextAttachment) {
  return attachment.kind === "character_image" ? "text image" : attachment.kind === "uploaded_photo" ? "photo" : "text attachment";
}

function isActiveCharacterTextDelivery(message: CharacterTextMessage) {
  return message.delivery_status === "pending" || message.delivery_status === "retrying";
}

export function isPlaceholderCharacterTextDelivery(message: CharacterTextMessage) {
  return message.sender === "character"
    && isActiveCharacterTextDelivery(message);
}

function characterTextMessageGrouped(
  message: CharacterTextMessage,
  previousMessage: CharacterTextMessage | null,
) {
  if (!previousMessage || previousMessage.sender !== message.sender) return false;
  if ((previousMessage.sender_character_id ?? "") !== (message.sender_character_id ?? "")) return false;
  const currentTime = characterTextTimestampMs(message.created_at);
  const previousTime = characterTextTimestampMs(previousMessage.created_at);
  if (currentTime === null || previousTime === null) return false;
  return currentTime - previousTime >= 0 && currentTime - previousTime <= 5 * 60 * 1000;
}

function characterTextMessageStartsNewDay(
  message: CharacterTextMessage,
  previousMessage: CharacterTextMessage | null,
) {
  const dayKey = characterTextDayKey(message.created_at);
  if (!dayKey) return false;
  return dayKey !== characterTextDayKey(previousMessage?.created_at);
}

function characterTextDayKey(value: string | null | undefined): string {
  const timestamp = characterTextTimestampMs(value);
  if (timestamp === null) return "";
  const date = new Date(timestamp);
  return `${date.getFullYear()}-${date.getMonth() + 1}-${date.getDate()}`;
}

const CHARACTER_TEXT_DAY_FORMATTER = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric"
});
const CHARACTER_TEXT_TIME_FORMATTER = new Intl.DateTimeFormat("en-US", {
  hour: "numeric",
  minute: "2-digit"
});

function characterTextDayLabel(value: string | null | undefined): string {
  const timestamp = characterTextTimestampMs(value);
  if (timestamp === null) return "Earlier";
  return CHARACTER_TEXT_DAY_FORMATTER.format(new Date(timestamp));
}

function characterTextMessageTimeLabel(message: CharacterTextMessage): string {
  const inWorldSentAt = message.in_world_sent_at?.trim();
  if (inWorldSentAt) return inWorldSentAt;
  const timestamp = characterTextTimestampMs(message.created_at);
  if (timestamp === null) return "";
  return CHARACTER_TEXT_TIME_FORMATTER.format(new Date(timestamp));
}

function characterTextTimestampMs(value: string | null | undefined): number | null {
  if (!value) return null;
  const normalized = value.includes("T") ? value : value.replace(" ", "T");
  const timestamp = Date.parse(normalized);
  return Number.isNaN(timestamp) ? null : timestamp;
}

function localCharacterTextThreadKey(characterId: string) {
  return `local-character-text-thread:${characterId}`;
}

function groupThreadParticipantNames(thread: CharacterTextThread): string {
  const names = (thread.participants ?? [])
    .map((participant) => participant.name.trim())
    .filter(Boolean);
  return names.length ? names.join(", ") : "Group";
}

function lastCharacterTextMessageId(messages: CharacterTextMessage[]): string | null {
  if (!messages.length) return null;
  return messages[messages.length - 1].id;
}

function characterTextMessagesWithLocalEchoes(
  serverMessages: CharacterTextMessage[],
  localMessages: LocalCharacterTextMessage[],
): CharacterTextMessage[] {
  return [
    ...serverMessages,
    ...unacceptedLocalCharacterTextMessages(serverMessages, localMessages),
  ];
}

function pruneAcceptedLocalCharacterTexts(
  messagesByThread: Record<string, LocalCharacterTextMessage[]>,
  threadKey: string,
  serverMessages: CharacterTextMessage[],
): Record<string, LocalCharacterTextMessage[]> {
  const current = messagesByThread[threadKey] ?? [];
  if (!current.length) return messagesByThread;
  const remaining = unacceptedLocalCharacterTextMessages(serverMessages, current);
  if (remaining.length === current.length) return messagesByThread;
  if (!remaining.length) {
    const { [threadKey]: _removed, ...rest } = messagesByThread;
    return rest;
  }
  return { ...messagesByThread, [threadKey]: remaining };
}

function markLocalCharacterTextFailed(
  messagesByThread: Record<string, LocalCharacterTextMessage[]>,
  threadKey: string,
  messageId: string,
  error: string,
): Record<string, LocalCharacterTextMessage[]> {
  const messages = messagesByThread[threadKey] ?? [];
  if (!messages.length) return messagesByThread;
  return {
    ...messagesByThread,
    [threadKey]: messages.map((message) => (
      message.id === messageId
        ? {
          ...message,
          delivery_status: "failed",
          delivery_error: error,
        }
        : message
    )),
  };
}

function unacceptedLocalCharacterTextMessages(
  serverMessages: CharacterTextMessage[],
  localMessages: LocalCharacterTextMessage[],
): LocalCharacterTextMessage[] {
  const matchedServerIndexes = new Set<number>();
  return localMessages.filter((localMessage) => {
    const matchingIndex = acceptedLocalCharacterTextServerIndex(
      serverMessages,
      localMessage,
      matchedServerIndexes,
    );
    if (matchingIndex === null) return true;
    matchedServerIndexes.add(matchingIndex);
    return false;
  });
}

function acceptedLocalCharacterTextServerIndex(
  serverMessages: CharacterTextMessage[],
  localMessage: LocalCharacterTextMessage,
  matchedServerIndexes: Set<number>,
): number | null {
  const startIndex = localCharacterTextMatchStartIndex(serverMessages, localMessage.local_after_message_id ?? null);
  if (startIndex === null) return null;
  for (let index = startIndex; index < serverMessages.length; index += 1) {
    const serverMessage = serverMessages[index];
    if (matchedServerIndexes.has(index)) continue;
    if (serverMessage.sender !== "player" || serverMessage.body !== localMessage.body) continue;
    if (localMessage.delivery_status === "failed" && serverMessage.delivery_status !== "failed") continue;
    return index;
  }
  return null;
}

function localCharacterTextMatchStartIndex(
  serverMessages: CharacterTextMessage[],
  afterMessageId: string | null,
): number | null {
  if (!afterMessageId) return 0;
  const index = serverMessages.findIndex((message) => message.id === afterMessageId);
  return index === -1 ? null : index + 1;
}

function characterTextDeliveryView(message: CharacterTextMessage): { kind: string; label: string } | null {
  if (message.delivery_status === "pending") return { kind: "pending", label: "Pending" };
  if (message.delivery_status === "retrying") return { kind: "retrying", label: "Retrying" };
  if (message.delivery_status === "failed") return { kind: "failed", label: "Failed" };
  if (message.read_at) return { kind: "read", label: "Read" };
  if (message.delivered_at || message.delivery_status === "sent") {
    return { kind: "delivered", label: "Delivered" };
  }
  return null;
}
