export type InteractionMode = "roleplay" | "storyteller";

export type RuntimeModel = {
  saves: SaveListItem[];
  active_save_id: string | null;
  active_save_title: string | null;
  active_scenario_type?: string | null;
  action_choices_enabled?: boolean;
  character_texts_enabled?: boolean;
  custom_instructions?: string;
  scenario_title: string | null;
  scene_title: string;
  chronicle: ChronicleModel;
  world_time?: RuntimeWorldTime | null;
  media: MediaModel | null;
  action_choices: RuntimeActionChoices | null;
  model_indicator: string;
  failed_save: boolean;
  composer_enabled: boolean;
  failure_text: string | null;
  status: string | null;
  status_text?: string | null;
  error: string | null;
  scenario_draft?: ScenarioDraft | null;
  scenario_wizard?: ScenarioWizard | null;
  interaction_mode?: InteractionMode;
};

export type RuntimeWorldTime = {
  snapshot_id: string;
  day_index: number | null;
  day_label: string;
  phase: string;
  clock_minutes: number | null;
  period_label: string;
  source_message_id: string | null;
  confidence: number | null;
  display: string;
};

export type RuntimeActionChoice = {
  choice_id: string;
  ordinal: number;
  body: string;
};

export type RuntimeActionChoices = {
  narrator_message_id: string;
  choices: RuntimeActionChoice[];
  generation_job?: Job | null;
  generation_error?: string | null;
};

export type CharacterTextContact = {
  id: string;
  name: string;
  contact_name?: string | null;
  role: string;
  status: string;
  is_player_character: boolean;
  player_has_character_number: boolean;
  character_has_player_number: boolean;
  player_number_permission?: CharacterTextContactPermission;
  character_number_permission?: CharacterTextContactPermission;
  thread_id?: string | null;
  latest_message_id?: string | null;
  latest_message_body?: string;
  latest_message_markdown_blocks?: MarkdownBlock[];
  latest_message_sender?: string | null;
  latest_message_at?: string | null;
  latest_message_read_at?: string | null;
  reference_image?: CharacterReferenceImage | null;
};

export type CharacterTextContactPermission = {
  allowed: boolean;
  source: "none" | "chronicle" | "text_message" | "manual_or_legacy" | string;
  reason: string;
  source_message_id?: string | null;
  source_text_message_id?: string | null;
};

export type CharacterTextAttachment = {
  id: string;
  kind: "character_image" | "object_context_image" | "uploaded_photo" | string;
  status: "succeeded" | "failed" | string;
  media_asset_id?: string | null;
  mime_type?: string | null;
  provider?: string | null;
  model?: string | null;
  prompt_preview?: string;
  error?: string | null;
  created_at?: string | null;
};

export type CharacterTextMessage = {
  id: string;
  thread_id: string;
  character_id: string | null;
  sender: "player" | "character" | string;
  sender_character_id?: string | null;
  sender_display_name?: string;
  body: string;
  markdown_blocks?: MarkdownBlock[];
  attachments?: CharacterTextAttachment[];
  actions?: { action_id: string; label: string; detail_text?: string | null }[];
  revision_count?: number;
  edited_at?: string | null;
  provider?: string | null;
  model?: string | null;
  token_estimate?: number | null;
  delivery_status?: "pending" | "retrying" | "sent" | "failed" | string;
  delivery_error?: string | null;
  delivery_job_id?: string | null;
  delivery_attempt?: number | null;
  created_at?: string | null;
  in_world_sent_at?: string | null;
  delivered_at?: string | null;
  read_at?: string | null;
  reply_to_message_id?: string | null;
  proactive_reason?: string;
  proactive_trigger_type?: string;
};

export type CharacterTextThreadParticipant = {
  character_id: string;
  name: string;
  ordinal: number;
};

export type CharacterTextThread = {
  id: string;
  character_id: string | null;
  title: string;
  status: string;
  kind?: "direct" | "group" | string;
  participants?: CharacterTextThreadParticipant[];
  created_at?: string | null;
  updated_at?: string | null;
  messages: CharacterTextMessage[];
};

export type CharacterTextsModel = {
  save_id: string;
  enabled: boolean;
  contacts: CharacterTextContact[];
  repair_contacts: CharacterTextContact[];
  threads: CharacterTextThread[];
};

export type SaveListItem = {
  save_id: string;
  title: string;
  active: boolean;
  scenario_id?: string | null;
  scenario_title?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  last_opened_at?: string | null;
  supported?: boolean;
  unsupported_reason?: string | null;
  interaction_mode?: InteractionMode;
};
export type SaveEvent = {
  event_id: number;
  save_id: string | null;
  type: string;
  payload?: unknown;
};
export type Scenario = {
  scenario_id: string;
  scenario_type: string;
  scenario_types?: string[];
  title: string;
  premise: string;
  player_role: string;
  opening_message: string | null;
  save_count: number;
  has_generation_prompt?: boolean;
  action_choices_enabled?: boolean;
  created_at?: string | null;
  updated_at?: string | null;
  supported?: boolean;
  unsupported_reason?: string | null;
  interaction_mode?: InteractionMode;
};
export type ScenarioContentSection = [string, string];
export type ScenarioCharacterStarter = {
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
  relationships: Record<string, unknown>;
  goals: string;
  motivations: string;
  boundaries: string;
  status: string;
  met: boolean;
  locked_fields: string[];
  reference_image?: ScenarioStarterReferenceImage | null;
};
export type ScenarioStarterReferenceImage = {
  id: string;
  path: string;
  thumbnail_path?: string | null;
  mime_type: string;
  prompt_preview: string;
  source?: string | null;
  created_at?: string | null;
  bundle_path?: string | null;
};
export type WorldDataScenario = {
  scenario_id: string;
  scenario_type: string;
  title: string;
  premise: string;
  player_character_name: string;
  player_role: string;
  content_sections: ScenarioContentSection[];
  generation_prompt?: string | null;
  character_starters?: ScenarioCharacterStarter[];
  interaction_mode?: InteractionMode;
};
export type ChronicleMessage = {
  message_id: string;
  role: string;
  speaker_name: string | null;
  body: string;
  revision_count?: number;
  edited_at?: string | null;
  actions: { action_id: string; label: string; detail_text?: string | null }[];
  markdown_blocks?: MarkdownBlock[];
  debug_prompt?: unknown;
  debug_provider_payload?: unknown;
};
export type ChronicleModel = {
  messages: ChronicleMessage[];
  has_more_before?: boolean;
  oldest_message_id?: string | null;
};
export type ChatTurnDelta = {
  kind: "chat_turn_delta";
  version: 1;
  save_id: string;
  status: string | null;
  error: string | null;
  player_message_id: string | null;
  narrator_message_id: string | null;
  messages: ChronicleMessage[];
  action_choices: RuntimeActionChoices | null;
  save: SaveListItem | null;
  fallback_used: boolean;
  context_trimmed: boolean;
  requires_full_refresh?: boolean;
};
export type MarkdownSpan = { kind: string; text: string; target?: string | null };
export type MarkdownBlock = {
  kind?: string;
  block_type?: string;
  spans?: MarkdownSpan[];
  text?: string;
  marker?: string | null;
  ordinal?: number | null;
  language?: string | null;
};
export type MediaModel = {
  latest_scene_media?: MediaAsset | null;
  latest_scene_image: MediaImage | null;
  character_reference_image?: MediaImage | null;
  image_history: MediaImage[];
  media_history: MediaAsset[];
  image_animation_available?: boolean;
};
export type MediaSource = {
  id: string;
  type: string;
  mime_type: string;
  prompt_preview: string;
  source_message_id: string | null;
  created_at: string | null;
};
export type MediaAsset = {
  id: string;
  source_message_id: string | null;
  source_media_asset_id?: string | null;
  type: string;
  mime_type: string;
  thumbnail_path?: string | null;
  provider?: string;
  model?: string;
  source_message?: string | null;
  prompt_preview: string;
  character_name?: string | null;
  prompt?: string;
  status: string;
  created_at: string | null;
  metadata?: Record<string, unknown>;
  source_media?: MediaSource | null;
  file_available?: boolean;
  can_animate?: boolean;
  is_character_reference?: boolean;
  can_set_character_reference?: boolean;
};
export type MediaImage = MediaAsset;
export type MediaAssetPrompt = {
  media_asset_id: string;
  prompt: string;
};
export type ScenarioDraft = {
  scenario_type: string;
  scenario_types?: string[];
  sections: [string, string][];
  regeneration_seed: string;
  source_metadata: [string, unknown][];
  action_choices_enabled?: boolean;
  character_starters?: ScenarioCharacterStarter[];
  interaction_mode?: InteractionMode;
};
export type ScenarioWizard = {
  flows: ScenarioWizardFlow[];
};
export type ScenarioWizardFlow = {
  flow_id: string;
  label: string;
  seed_prompt: string;
  editable_section_ids: string[];
  review_groups: { label: string; section_ids: string[] }[];
};
export type BundlePreview = {
  save_id: string;
  title: string;
  scenario_title: string;
  message_count: number;
  media_count: number;
  bundle_version: number;
  created_at: string | null;
  updated_at: string | null;
  exported_at: string | null;
};
export type ScenarioBundlePreview = {
  scenario_id: string;
  title: string;
  scenario_type: string;
  bundle_version: number;
  created_at: string | null;
  updated_at: string | null;
  exported_at: string | null;
};
export type CharacterBundlePreview = {
  character_id: string;
  name: string;
  suggested_name: string;
  name_conflict: boolean;
  media_count: number;
  bundle_version: number;
  aliases: string[];
  role: string;
  age: string;
  known_state: string;
  history: string;
  appearance: string;
  current_clothing?: string;
  personality: string;
  voice: string;
  texting_style: string;
  status: string;
  created_at: string | null;
  updated_at: string | null;
  exported_at: string | null;
  skipped_media_count: number;
  warnings: string[];
};
export type AdminUser = {
  id: string;
  username: string;
  role: "admin" | "user" | "child" | string;
  status: "active" | "disabled" | string;
  content_rating?: string;
  created_at: string | null;
  updated_at: string | null;
};
export type ChatHistoryModel = {
  active_save_id: string | null;
  active_save_title: string | null;
  selected_filter: string;
  filter_options: ChatHistoryFilterOption[];
  messages: ChatHistoryMessage[];
  total_message_count: number;
  matching_message_count?: number;
  has_more_before?: boolean;
  oldest_message_id?: string | null;
  empty_title: string;
  empty_detail: string;
};
export type ChatHistoryFilterOption = {
  filter_id: string;
  label: string;
  active: boolean;
};
export type ChatHistoryMessage = {
  message_id: string;
  role: string;
  role_label: string;
  speaker_name: string | null;
  body: string;
  markdown_blocks?: MarkdownBlock[];
  style_class: string;
  provider: string | null;
  model: string | null;
  token_estimate: number | null;
  created_at: string | null;
  image_count: number;
  has_images?: boolean;
  provider_model_label?: string;
};
export type SettingsModel = {
  provider_cards: ProviderCard[];
  task_model_selectors: TaskModelSelector[];
  save_model_override_selectors?: TaskModelSelector[];
  roleplay_shared_models?: ToggleControl;
  roleplay_model_groups: RoleplayModelGroup[];
  scenario_section_model_selectors?: TaskModelSelector[];
  model_routing_profiles?: ModelRoutingProfilesSettings;
  retry_count?: NumberControl;
  automatic_summarization?: ToggleControl;
  summarization_context_pressure_threshold?: NumberControl;
  summarization_visibility?: ToggleControl;
  agentic_context_pipeline?: ToggleControl;
  plan_first_narrator?: ToggleControl;
  director_pressure?: ToggleControl;
  character_action_planning?: ToggleControl;
  character_action_planning_max_concurrency?: NumberControl;
  character_texts?: ToggleControl;
  character_text_proactive_random_chance?: NumberControl;
  character_text_proactive_random_cooldown?: NumberControl;
  post_turn_inference_mode?: ChoiceControl;
  npc_knowledge_audit_mode?: ChoiceControl;
  generated_text_script_guard_mode?: ChoiceControl;
  generated_phrase_denylist?: TextControl;
  save_generated_phrase_denylist?: TextControl;
  chat_fallback?: ToggleControl;
  structured_output_fallback?: ToggleControl;
  tool_call_fallback?: ToggleControl;
  image_fallback?: ToggleControl;
  video_fallback?: ToggleControl;
  venice_image_safe_mode?: ToggleControl;
  debug_logging?: ToggleControl;
  pending_jobs_display_mode?: ChoiceControl;
  user_narration_guidance?: TextControl;
  content_rating?: ContentRatingControl;
  fade_to_black?: ToggleControl;
  automatic_image_generation?: ToggleControl;
  image_style_preset?: ChoiceControl;
  chat_temperature?: OptionalNumberControl;
  chat_max_output_tokens?: OptionalNumberControl;
  image_dimension_preset?: SupportedChoiceControl;
  openrouter_routing?: OpenRouterRoutingSettings;
  automatic_media_mode?: ChoiceControl;
  image_frequency?: NumberControl;
  manual_confirmation?: ManualConfirmationControls;
  chat_history?: ChatHistoryControls;
  context_budget?: ContextBudgetControls;
  secret_storage_warning?: string | null;
  visible_sections?: string[];
};
export type ProviderSettingsModel = Pick<SettingsModel, "provider_cards" | "secret_storage_warning">;
export type LocalSettingsModel = Pick<SettingsModel, "pending_jobs_display_mode" | "user_narration_guidance" | "content_rating" | "fade_to_black" | "debug_logging">;
export type ProviderCard = {
  provider: string;
  enabled: boolean;
  has_api_key: boolean;
  model_count: number;
  last_model_refresh_at: string | null;
  refresh_status: string;
  last_error: string | null;
};
export type ToggleControl = {
  setting_key: string;
  enabled: boolean;
};
export type ContentRatingControl = {
  setting_key: string;
  selected: string;
  options: string[];
  admin_granted: boolean;
};
export type NumberControl = {
  setting_key: string;
  value: number;
  minimum: number;
  maximum?: number | null;
  step: number;
};
export type OptionalNumberControl = {
  setting_key: string;
  enabled_setting_key: string;
  enabled: boolean;
  supported: boolean;
  value: number;
  minimum: number;
  maximum: number;
  step: number;
};
export type FractionControl = {
  setting_key: string;
  value: number;
  minimum: number;
  maximum?: number | null;
};
export type ChoiceControl = {
  setting_key: string;
  selected: string;
  options: string[];
};
export type TextControl = {
  setting_key: string;
  value: string;
};
export type SupportedChoiceControl = ChoiceControl & {
  supported: boolean;
};
export type OpenRouterRoutingProfile = {
  order: string[];
  allow_fallbacks: boolean | null;
  require_parameters: boolean;
  data_collection: string;
  zdr: boolean;
  enforce_distillable_text: boolean;
  only: string[];
  ignore: string[];
  quantizations: string[];
  sort: string;
  sort_partition: string;
  preferred_min_throughput: Record<string, number>;
  preferred_max_latency: Record<string, number>;
  max_price: Record<string, number>;
};
export type OpenRouterRoutingTaskOverride = {
  task_family: string;
  label: string;
  enabled: boolean;
  profile: OpenRouterRoutingProfile;
  provider_payload: Record<string, unknown>;
  effective_provider_payload: Record<string, unknown>;
};
export type OpenRouterProviderCatalogEntry = {
  slug: string;
  name: string;
  privacy_policy_url?: string | null;
  terms_of_service_url?: string | null;
  status_page_url?: string | null;
  headquarters?: string | null;
  datacenters: string[];
};
export type OpenRouterRoutingSettings = {
  setting_key: string;
  global_profile: OpenRouterRoutingProfile;
  global_provider_payload: Record<string, unknown>;
  task_overrides: OpenRouterRoutingTaskOverride[];
  provider_catalog: OpenRouterProviderCatalogEntry[];
  provider_catalog_refreshed_at: string | null;
  sort_options: string[];
  partition_options: string[];
  data_collection_options: string[];
  quantization_options: string[];
  percentile_options: string[];
  max_price_fields: string[];
};
export type ModelRoutingProfilePreference = {
  task: string;
  provider: string;
  model_id: string;
};
export type ModelRoutingProfile = {
  id: string;
  name: string;
  roleplay_shared_models_enabled: boolean;
  preference_count: number;
  preferences: ModelRoutingProfilePreference[];
};
export type ModelRoutingProfilesSettings = {
  setting_key: string;
  last_loaded_profile_id: string | null;
  profiles: ModelRoutingProfile[];
};
export type ManualConfirmationControls = {
  memories: ToggleControl;
  character_registry: ToggleControl;
  state_changes: ToggleControl;
};
export type ChatHistoryControls = {
  planner_player_messages: NumberControl;
  planner_narrator_messages: NumberControl;
  player_messages: NumberControl;
  narrator_messages: NumberControl;
};
export type ContextBudgetControls = {
  mode: ChoiceControl;
  fixed_total_chars: NumberControl;
  adaptive_fraction: FractionControl;
};
export type DiagnosticEntry = {
  kind: string;
  job_id?: string | null;
  error: string | null;
  provider?: string | null;
  model?: string | null;
  job_type?: string | null;
  save_id?: string | null;
  path?: string | null;
  retry_summary?: string | null;
  origin?: { kind: string; label: string; route?: string; request_id?: string } | null;
  detail_available?: boolean;
};
export type MaintenanceJobDiagnostic = {
  job_id: string;
  job_type: string;
  status: string;
  save_id: string | null;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  summary: string;
  metrics: Record<string, unknown>;
};
export type RuntimePerformanceReport = {
  job_averages: RuntimePerformanceRow[];
  step_averages: RuntimePerformanceRow[];
  model_averages: RuntimePerformanceRow[];
  slowest_recent?: RuntimeSlowOperation[];
  window_started_at?: string | null;
  limit?: number | null;
};
export type RuntimePerformanceRow = {
  job_type?: string | null;
  step_name?: string | null;
  provider?: string | null;
  model?: string | null;
  task?: string | null;
  sample_count?: number;
  success_count: number;
  failed_count: number;
  cancelled_count: number;
  skipped_count: number;
  average_duration_ms: number | null;
  p50_duration_ms?: number | null;
  p95_duration_ms?: number | null;
  min_duration_ms: number | null;
  max_duration_ms: number | null;
  latest_duration_ms: number | null;
  average_queue_wait_ms?: number | null;
  p95_queue_wait_ms?: number | null;
  failure_rate?: number | null;
  latest_completed_at: string | null;
};
export type RuntimeSlowOperation = {
  job_id: string;
  save_id: string | null;
  job_type: string;
  status: string;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  queue_wait_ms?: number | null;
  slowest_step_name?: string | null;
  slowest_step_duration_ms?: number | null;
  provider?: string | null;
  model?: string | null;
  task?: string | null;
};
export type TerminalJobSummary = {
  id: string;
  type: string;
  save_id: string | null;
  status: Job["status"];
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  queue_wait_ms: number | null;
  step_count: number;
  error: string | null;
  origin?: { kind: string; label: string; route?: string; request_id?: string } | null;
  provider?: string | null;
  model?: string | null;
  detail_available?: boolean;
};
export type JobStepSummary = {
  id: string;
  name: string;
  status: string;
  provider: string | null;
  model: string | null;
  task: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
  metadata: Record<string, unknown>;
};
export type JobStepsModel = {
  job_id: string;
  steps: JobStepSummary[];
};
export type JobDiagnosticSnapshot = {
  version?: number;
  request?: {
    origin?: { kind: string; label: string; route?: string; request_id?: string };
    job_type?: string;
    task?: string;
    provider?: string | null;
    model?: string | null;
    save_id?: string | null;
    source_message_id?: string;
    source_text_message_id?: string;
    character_id?: string;
    thread_id?: string;
    source_media_asset_ids?: string[];
    prompt?: string;
    parameters?: Record<string, unknown>;
  };
  provider?: Record<string, unknown>;
  bragi?: { status?: string; error?: string };
  timing?: Record<string, unknown>;
  related?: Record<string, string>;
};
export type JobDiagnosticsModel = {
  job_id: string;
  job_type: string;
  save_id: string | null;
  status: Job["status"] | string;
  detail_level: "admin" | "metadata" | string;
  detail_available: boolean;
  diagnostics: JobDiagnosticSnapshot;
};
export type WebEventEntry = {
  timestamp: string;
  level: string;
  event: string;
  request_id?: string | null;
  job_id?: string | null;
  save_id?: string | null;
  task_type?: string | null;
  status_code?: number | null;
  status?: string | null;
  duration_ms?: number | null;
  error?: string | null;
  error_class?: string | null;
  method?: string | null;
  route?: string | null;
  job_type?: string | null;
  job_status?: string | null;
  component?: string | null;
  action?: string | null;
};
export type EngineHealthWarning = {
  code: string;
  severity: string;
  message: string;
  count?: number | null;
};
export type EngineHealthModel = {
  save_id: string;
  active_message_count: number;
  recent_player_message_window: number;
  recent_narrator_message_window: number;
  narrator_planner_recent_player_message_window: number;
  narrator_planner_recent_narrator_message_window: number;
  pending_suggestion_count: number;
  stale_pending_suggestion_count: number;
  summary_count: number;
  recent_failed_continuity_job_count: number;
  recent_failed_continuity_jobs_by_type: Record<string, number>;
  observation_curation: {
    pending_count: number;
    eligible_count: number;
    leased_count: number;
    oldest_pending_at: string | null;
    oldest_pending_age_seconds: number | null;
    total_attempt_count: number;
    max_attempt_count: number;
    terminal_failure_count: number;
  };
  latest_context_search: Record<string, unknown> | null;
  latest_chat_prompt: Record<string, unknown> | null;
  warnings: EngineHealthWarning[];
};
export type SchedulerHealthTask = {
  task_id: string;
  task_type: string;
  save_id: string | null;
  status: string;
  enabled: boolean;
  interval_seconds: number;
  next_run_at: string | null;
  lease_until: string | null;
  last_started_at: string | null;
  last_completed_at: string | null;
  last_job_id: string | null;
  failure_count: number;
  error: string | null;
  skip_reason: string | null;
};
export type SchedulerHealthReport = {
  summary: {
    total: number;
    healthy: number;
    overdue: number;
    leased: number;
    failed: number;
    disabled: number;
  };
  tasks: SchedulerHealthTask[];
};
export type DiagnosticsModel = {
  generated_at: string;
  filters: {
    save_id: string | null;
    categories: string[];
    limit: number;
    since: string | null;
    request_id?: string | null;
    job_id?: string | null;
    route?: string | null;
    component?: string | null;
  };
  signals: DiagnosticEntry[];
  maintenance_jobs: MaintenanceJobDiagnostic[];
  runtime_performance: RuntimePerformanceReport | null;
  scheduler_health: SchedulerHealthReport;
  web_events: WebEventEntry[];
  active_save_health: EngineHealthModel | null;
};
export type RoleplayModelGroup = {
  roleplay_type: string;
  label: string;
  selectors: TaskModelSelector[];
};
export type TaskModelSelector = {
  task: string;
  selected_provider: string | null;
  selected_model_id: string | null;
  selected_available: boolean;
  warning: string | null;
  options: ModelOption[];
  thinking?: ThinkingLevelControl | null;
  label?: string | null;
  section_id?: string | null;
  inherited_provider?: string | null;
  inherited_model_id?: string | null;
  clearable?: boolean;
};
export type ModelOption = {
  provider: string;
  model_id: string;
  display_name: string;
  available: boolean;
  capabilities: string[];
  pricing?: ModelPricing | null;
  thinking?: ModelThinkingSupport | null;
};
export type ModelThinkingSupport = {
  levels: string[];
  default_level?: string | null;
  default_enabled?: boolean | null;
  mandatory: boolean;
  supports_max_tokens: boolean;
};
export type ThinkingLevelControl = {
  setting_key: string;
  task: string;
  selected: string;
  supported: boolean;
  options: string[];
  provider?: string | null;
  model_id?: string | null;
  default_level?: string | null;
  default_enabled?: boolean | null;
  mandatory?: boolean;
  disabled_reason?: string | null;
};
export type ModelPricing = {
  input_per_million_tokens_usd?: string | null;
  output_per_million_tokens_usd?: string | null;
  cache_read_per_million_tokens_usd?: string | null;
  cache_write_per_million_tokens_usd?: string | null;
  request_usd?: string | null;
  image_usd?: string | null;
  note?: string | null;
};
export type Job = {
  id: string;
  type: string;
  save_id?: string | null;
  status: "queued" | "running" | "succeeded" | "failed" | "cancelled";
  completion_level?: "response_committed" | "continuity_ready" | "optional_enrichments_complete" | null;
  result: unknown;
  error: string | null;
  created_at?: number;
  updated_at?: number;
  latest_progress?: unknown | null;
};
export type ChatSubmissionStatus = {
  save_id: string | null;
  can_submit: boolean;
  reason: "no_save" | "chat_turn_active" | null;
  blocking_job_id: string | null;
  blocking_job_status: Job["status"] | null;
};
export type WorldDataModel = {
  active_save_id: string | null;
  save?: Record<string, unknown> | null;
  scenario?: WorldDataScenario | null;
  scene?: Record<string, unknown> | null;
  world_state?: unknown[];
  memories?: unknown[];
  context_inputs?: unknown[];
  summaries?: unknown[];
  locations?: unknown[];
  characters?: unknown[];
  threads?: unknown[];
  links?: unknown[];
  suggestions?: WorldDataSuggestionRow[];
  suggestion_groups?: WorldDataSuggestionGroupRow[];
  audit?: unknown[];
  error?: string | null;
  [key: string]: unknown;
};
export type WorldDataSuggestionRow = {
  suggestion_id: string;
  update_type: string;
  entity_type: string;
  entity_id: string | null;
  field_path: string;
  proposed_value_json: string;
  status: string;
  reason: string;
  confidence: number;
  source_message_ids_text: string;
  action: string;
};
export type WorldDataSuggestionGroupRow = {
  group_id: string;
  suggestion_ids: string[];
  update_type: string;
  entity_type: string;
  entity_id: string | null;
  field_path: string;
  proposed_value_json: string;
  status: string;
  reason: string;
  confidence: number;
  source_message_ids_text: string;
  suggestion_count: number;
  action: string;
};
export type WorldDataApplyResult = {
  model: WorldDataModel;
  state_archive_count: number;
  memory_archive_count: number;
  summary_archive_count: number;
  error?: string | null;
};
export type CharacterRegistryModel = {
  active_save_id: string | null;
  save?: Record<string, unknown> | null;
  characters?: CharacterRow[];
  link_targets?: CharacterKnowledgeTarget[];
  location_choices?: unknown[];
  error?: string | null;
};
export type CharacterRegistryApplyResult = {
  model: CharacterRegistryModel;
  created_count: number;
  updated_count: number;
  archived_count: number;
  created_character_ids?: string[];
  auto_enhanced_count?: number;
  error?: string | null;
};
export type CharacterKnowledgeApplyResult = CharacterRegistryApplyResult;
export type CharacterEnhanceField =
  | "known_state"
  | "appearance"
  | "visual_notes"
  | "personality"
  | "voice"
  | "texting_style"
  | "goals"
  | "motivations"
  | "current_intent"
  | "boundaries"
  | "attitude_toward_player"
  | "cooperation_conditions"
  | "status"
  | "relationships";
export type CharacterFieldEnhanceResult = {
  model: CharacterRegistryModel;
  character_id: string;
  field_name: CharacterEnhanceField;
  created_count: number;
  updated_count: number;
  archived_count: number;
  field_changed?: boolean;
  notice?: string | null;
  error?: string | null;
};
export type CharacterRow = {
  character_id?: string;
  id?: string;
  name?: string;
  contact_name?: string | null;
  texting_style?: string | null;
  current_clothing?: string | null;
  age?: string;
  goals?: string;
  motivations?: string;
  current_intent?: string;
  boundaries?: string;
  attitude_toward_player?: string;
  cooperation_conditions?: string;
  status?: string;
  present?: boolean;
  is_player_character?: boolean;
  reference_image?: CharacterReferenceImage | null;
  generated_images?: CharacterReferenceImage[];
  locked_fields?: string[];
  [key: string]: unknown;
};
export type CharacterKnowledgeTarget = {
  target_type: "memory" | "world_state" | "summary" | string;
  target_id: string;
  title: string;
  body: string;
  linked_character_ids?: string[];
  value?: Record<string, unknown> | null;
  category?: string;
  confidence?: number | null;
  tags?: string[];
  importance?: number | null;
  source_message_id?: string | null;
  [key: string]: unknown;
};
export type CharacterReferenceImage = {
  media_asset_id: string;
  mime_type: string;
  prompt_preview: string;
  provider: string;
  model: string;
  created_at?: string | null;
  source?: string | null;
};
export type ScenePresenceCharacter = {
  character_id: string;
  name: string;
  present: boolean;
  has_reference_image: boolean;
  reference_image?: CharacterReferenceImage | null;
  is_player_character: boolean;
  status: string;
};
export type ScenePresenceModel = {
  save_id: string;
  message_id: string;
  latest_message: boolean;
  characters: ScenePresenceCharacter[];
};

const BODY_LIKE_KEYS = new Set([
  "body",
  "prompt",
  "message",
  "messages",
  "content",
  "payload",
  "request",
  "response",
  "api_key",
  "apikey",
  "authorization",
  "cookie",
  "token"
]);
const UNSAFE_API_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"]);
const BRAGI_API_REQUEST_HEADER = "X-Bragi-Api-Request";
let unauthorizedHandler: (() => void) | null = null;

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function setUnauthorizedHandler(handler: (() => void) | null) {
  unauthorizedHandler = handler;
}

export async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: requestHeaders(path, init)
  });
  if (!response.ok) {
    const payload = await response.json().catch(() => ({ detail: response.statusText }));
    const message = typeof payload.detail === "string" ? payload.detail : response.statusText;
    if (response.status === 401 && shouldNotifyUnauthorized(path)) {
      unauthorizedHandler?.();
    }
    if (shouldLogApiFailure(path, response.status)) {
      logClientEvent("error", "client.api.failed", {
        method: init?.method ?? "GET",
        path: stripUrlDetails(path),
        status_code: response.status,
        error_name: "ApiError",
        error_message: message
      });
    }
    throw new ApiError(message, response.status);
  }
  return response.json() as Promise<T>;
}

export function postJson<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, { method: "POST", body: JSON.stringify(body) });
}

export function deleteJson<T>(path: string, body?: unknown): Promise<T> {
  return api<T>(path, {
    method: "DELETE",
    body: body === undefined ? undefined : JSON.stringify(body)
  });
}

function requestHeaders(path: string, init?: RequestInit): Record<string, string> | undefined {
  const headers = headersToRecord(init?.headers);
  if (
    init?.body != null &&
    !(init.body instanceof FormData) &&
    !hasHeader(headers, "Content-Type")
  ) {
    headers["Content-Type"] = "application/json";
  }
  if (isUnsafeApiRequest(path, init?.method)) {
    setHeader(headers, BRAGI_API_REQUEST_HEADER, "1");
  }
  return Object.keys(headers).length ? headers : undefined;
}

function isUnsafeApiRequest(path: string, method = "GET"): boolean {
  return path.startsWith("/api/") && UNSAFE_API_METHODS.has(method.toUpperCase());
}

function shouldNotifyUnauthorized(path: string): boolean {
  return (
    path.startsWith("/api/") &&
    path !== "/api/auth/me" &&
    !path.startsWith("/api/auth/") &&
    !path.startsWith("/api/bootstrap/") &&
    path !== "/api/log/client"
  );
}

function shouldLogApiFailure(path: string, status: number): boolean {
  if (path === "/api/log/client") return false;
  if (status === 401 && !shouldNotifyUnauthorized(path)) return false;
  return true;
}

function headersToRecord(headers?: HeadersInit): Record<string, string> {
  if (!headers) return {};
  if (headers instanceof Headers) {
    const result: Record<string, string> = {};
    headers.forEach((value, key) => {
      result[key] = value;
    });
    return result;
  }
  if (Array.isArray(headers)) {
    return Object.fromEntries(headers.map(([key, value]) => [key, String(value)]));
  }
  return Object.fromEntries(
    Object.entries(headers).map(([key, value]) => [key, String(value)])
  );
}

function hasHeader(headers: Record<string, string>, name: string): boolean {
  const lowerName = name.toLowerCase();
  return Object.keys(headers).some((key) => key.toLowerCase() === lowerName);
}

function setHeader(headers: Record<string, string>, name: string, value: string) {
  const lowerName = name.toLowerCase();
  for (const key of Object.keys(headers)) {
    if (key.toLowerCase() === lowerName && key !== name) delete headers[key];
  }
  headers[name] = value;
}

function jobApiPath(jobId: string, saveId: string | null | undefined, suffix = "") {
  const query = saveId ? `?save_id=${encodeURIComponent(saveId)}` : "";
  return `/api/jobs/${encodeURIComponent(jobId)}${suffix}${query}`;
}

type SseParseContext = {
  stream: "job" | "save";
  eventName: string;
  jobId?: string;
  saveId?: string | null;
};

type SseParseResult<T> = { ok: true; value: T } | { ok: false };

function parseSseJson<T>(event: MessageEvent, context: SseParseContext): SseParseResult<T> {
  try {
    return { ok: true, value: JSON.parse(event.data) as T };
  } catch (failure) {
    logClientEvent("error", "client.sse.parse_failed", {
      stream: context.stream,
      event_name: context.eventName,
      job_id: context.jobId,
      save_id: context.saveId,
      payload_type: typeof event.data,
      payload_length: typeof event.data === "string" ? event.data.length : null,
      error_name: failure instanceof Error ? failure.name : "Error",
      error_message: "Invalid SSE JSON payload"
    });
    return { ok: false };
  }
}

export function watchJob(
  jobId: string,
  onUpdate: (job: Job) => void,
  onEvent?: (name: string, data: unknown) => void,
  saveId?: string | null
) {
  const events = new EventSource(jobApiPath(jobId, saveId, "/events"));
  let closed = false;
  let fallbackStarted = false;
  let fallback: number | undefined;
  const closeWatcher = () => {
    closed = true;
    events.close();
    if (fallback) window.clearTimeout(fallback);
  };
  const poll = async () => {
    if (closed) return;
    try {
      const job = await api<Job>(jobApiPath(jobId, saveId));
      if (["succeeded", "failed", "cancelled"].includes(job.status)) {
        onUpdate(job);
        if (job.status === "failed") {
          logClientEvent("error", "client.job.failed", {
            job_id: job.id,
            job_type: job.type,
            status: job.status,
            error_present: Boolean(job.error)
          });
        }
        closeWatcher();
        return;
      }
    } catch (failure) {
      if (failure instanceof ApiError && failure.status === 404) {
        logClientEvent("error", "client.job.stale", {
          job_id: jobId,
          save_id: saveId,
          error_message: failure.message
        });
        onUpdate({
          id: jobId,
          type: "unknown",
          status: "cancelled",
          result: null,
          error: failure.message
        });
        closeWatcher();
        return;
      }
      logClientEvent("error", "client.job.poll_failed", {
        job_id: jobId,
        save_id: saveId,
        error_name: failure instanceof Error ? failure.name : "Error",
        error_message: failure instanceof Error ? failure.message : "Could not read job"
      });
    }
    fallback = window.setTimeout(poll, 1000);
  };
  const startFallback = () => {
    if (closed) return;
    events.close();
    if (fallbackStarted) return;
    fallbackStarted = true;
    void poll();
  };
  events.addEventListener("done", (event) => {
    const parsed = parseSseJson<Job>(event as MessageEvent, { stream: "job", eventName: "done", jobId, saveId });
    if (!parsed.ok) {
      startFallback();
      return;
    }
    const job = parsed.value;
    onUpdate(job);
    if (job.status === "failed") {
      logClientEvent("error", "client.job.failed", {
        job_id: job.id,
        job_type: job.type,
        status: job.status,
        error_present: Boolean(job.error)
      });
    }
    closeWatcher();
  });
  for (const eventName of ["progress", "completion_level", "runtime", "chat_turn_delta"]) {
    events.addEventListener(eventName, (event) => {
      const parsed = parseSseJson<unknown>(event as MessageEvent, { stream: "job", eventName, jobId, saveId });
      if (!parsed.ok) {
        startFallback();
        return;
      }
      onEvent?.(eventName, parsed.value);
    });
  }
  events.addEventListener("error", (event) => {
    const data = (event as MessageEvent).data;
    if (typeof data === "string" && data.length > 0) {
      const parsed = parseSseJson<unknown>(event as MessageEvent, { stream: "job", eventName: "error", jobId, saveId });
      if (!parsed.ok) {
        startFallback();
        return;
      }
      onEvent?.("error", parsed.value);
    }
  });
  events.onerror = () => {
    logClientEvent("error", "client.job.sse_fallback", { job_id: jobId, save_id: saveId });
    startFallback();
  };
  return () => {
    closeWatcher();
  };
}

export function watchSave(saveId: string, onEvent: (event: SaveEvent) => void, onRecover?: () => void) {
  if (typeof EventSource === "undefined") return () => undefined;
  const events = new EventSource(`/api/saves/${encodeURIComponent(saveId)}/events`);
  const eventNames = [
    "runtime_changed",
    "job_changed",
    "saves_changed",
    "save_deleted",
    "scenarios_changed",
    "character_texts_changed",
    "world_data_changed"
  ];
  let latestEventId = 0;
  const handleEvent = (eventName: string) => (event: Event) => {
    const parsed = parseSseJson<SaveEvent>(event as MessageEvent, { stream: "save", eventName, saveId });
    if (!parsed.ok) {
      onRecover?.();
      return;
    }
    const payload = parsed.value;
    if (Number.isFinite(payload.event_id)) {
      if (payload.event_id <= latestEventId) return;
      latestEventId = payload.event_id;
    }
    onEvent(payload);
  };
  for (const name of eventNames) {
    events.addEventListener(name, handleEvent(name));
  }
  events.onerror = () => {
    logClientEvent("error", "client.save.sse_error", { save_id: saveId });
  };
  return () => events.close();
}

export function logClientEvent(level: "debug" | "info" | "error", event: string, fields: Record<string, unknown> = {}) {
  const safeFields = sanitizeClientFields(fields);
  void fetch("/api/log/client", {
    method: "POST",
    headers: { "Content-Type": "application/json", [BRAGI_API_REQUEST_HEADER]: "1" },
    body: JSON.stringify({ level, event: truncate(event, 80), fields: safeFields })
  }).catch(() => undefined);
}

export function sanitizeClientFields(fields: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(
    Object.entries(fields)
      .filter(([key]) => !isBodyLikeKey(key))
      .slice(0, 24)
      .map(([key, value]) => [key.slice(0, 60), sanitizeClientValue(value)])
  );
}

function sanitizeClientValue(value: unknown): unknown {
  if (typeof value === "string") return truncate(value.replace(/\r|\n/g, " "), 240);
  if (typeof value === "number" || typeof value === "boolean" || value === null) return value;
  if (Array.isArray(value)) return value.slice(0, 10).map(sanitizeClientValue);
  if (value && typeof value === "object") return sanitizeClientFields(value as Record<string, unknown>);
  if (value === undefined) return undefined;
  return truncate(String(value), 240);
}

function stripUrlDetails(value: string): string {
  return value.split("?")[0].split("#")[0];
}

function isBodyLikeKey(key: string): boolean {
  const normalized = key.toLowerCase().replace(/-/g, "_");
  return BODY_LIKE_KEYS.has(normalized) || normalized.endsWith("_body");
}

function truncate(value: string, maxLength: number): string {
  return value.length <= maxLength ? value : `${value.slice(0, maxLength - 3)}...`;
}
