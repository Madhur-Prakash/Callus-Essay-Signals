/**
 * TypeScript mirror of the FastAPI response schemas (`app/schemas/analysis.py`).
 * Kept hand-written and narrow rather than generated: the UI only consumes a
 * subset, and an explicit type is where a backend contract change should fail.
 */

export type Classification =
  | 'human'
  | 'ai_generated'
  | 'ai_polished'
  | 'insufficient_evidence';

export type SentenceClassification =
  | 'likely_human'
  | 'uncertain'
  | 'possibly_ai_assisted'
  | 'likely_ai_assisted'
  | 'unavailable';

export type ParagraphClassification =
  | 'likely_human'
  | 'uncertain'
  | 'contains_flagged_sentence'
  | 'likely_ai_assisted'
  | 'unavailable';

export interface Meter {
  key: string;
  label: string;
  /** Machine-leaningness in [0,1]. For an inverted feature (e.g. sentence-length
   *  variation) a LOW value produces a HIGH strength. */
  strength: number;
  /** Signal strength in words: low / typical / elevated / high. */
  level: string;
  value: number;
  display: string;
  unit: string;
  reference: string;
  detail: string;
  available: boolean;
  /** Where the measured value sits vs the human distribution - distinct from
   *  `level`, which describes the signal. */
  value_level: string;
  percentile_vs_human: number | null;
}

export interface Measurement {
  name: string;
  key: string;
  value: number;
  unit: string;
  reference: string;
  percentile_vs_human: number | null;
  strength: number | null;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  standardised: number;
  contribution: number;
  direction: string;
  method: string;
}

export interface EvidenceBlock {
  meters: Meter[];
  statements: string[];
  measurements: Measurement[];
  model_contributions: FeatureContribution[];
  engine_version: string;
}

export interface SentenceResult {
  sentence_id: number;
  paragraph_id: number;
  start: number;
  end: number;
  text: string;
  score: number | null;
  classification: SentenceClassification;
  confidence: string;
  n_words: number;
  features: Record<string, number>;
  evidence?: EvidenceBlock | null;
}

export interface ParagraphResult {
  paragraph_id: number;
  start: number;
  end: number;
  n_sentences: number;
  n_words: number;
  score: number | null;
  max_sentence_score: number | null;
  human_likeness: number | null;
  classification: ParagraphClassification;
  flagged_sentence_ids: number[];
  uncertain_sentence_ids: number[];
  sentence_ids: number[];
}

export interface RhythmPoint {
  index: number;
  paragraph_index: number;
  words: number;
  deviation_from_mean: number;
  abs_diff_prev: number;
  clauses: number;
  mean_logprob: number;
  perplexity: number;
}

export interface SummaryStatistics {
  mean_words_per_sentence: number;
  sentence_length_std: number;
  sentence_length_cv: number;
  burstiness_index: number;
  perplexity: number;
  median_sentence_perplexity: number;
  fraction_top1_tokens: number;
  mean_token_logprob: number;
  mean_token_entropy: number;
  type_token_ratio: number;
  root_type_token_ratio: number;
  trigram_repeat_ratio: number;
  pos_template_repeat_ratio: number;
  max_style_shift: number;
  style_changepoints: number;
  flesch_reading_ease: number;
  transition_word_rate: number;
  contraction_rate: number;
}

export interface AnalysisSummary {
  n_words: number;
  n_characters: number;
  n_sentences: number;
  n_paragraphs: number;
  sentences_scored: number;
  flagged_sentences: number;
  uncertain_sentences: number;
  human_like_sentences: number;
  flagged_paragraphs: number;
  uncertain_paragraphs: number;
  flagged_share: number;
  statistics: SummaryStatistics;
  lm_tokens_scored: number;
  lm_windows: number;
  segmentation_backend: string;
}

export interface ModelBlock {
  detector_version: string;
  model_version: string;
  dataset_version?: string | null;
  features_version: string;
  explanation_engine_version: string;
  data_regime?: string | null;
  language_model: string;
  language_model_role: string;
  classifier?: string | null;
  trained_at?: string | null;
}

export interface RepeatedPhrase {
  phrase: string;
  length: number;
  count: number;
  sentence_indices: number[];
}

export interface RepeatedTemplate {
  template: string;
  sentence_count: number;
  sentence_indices: number[];
}

export interface AnalysisResponse {
  analysis_id: string;
  status: 'completed' | 'queued' | 'processing' | 'failed';
  classification: Classification;
  label: string;
  description: string;
  confidence: string;
  confidence_score: number;
  probabilities: Record<string, number>;
  margin: number;
  abstained: boolean;
  abstain_reason: string | null;
  summary: AnalysisSummary;
  paragraphs: ParagraphResult[];
  sentences: SentenceResult[];
  evidence: EvidenceBlock;
  rhythm: RhythmPoint[];
  repetition: {
    repeated_phrases: RepeatedPhrase[];
    repeated_syntactic_templates: RepeatedTemplate[];
  };
  model: ModelBlock;
  timings: Record<string, number>;
  content_hash: string;
  created_at: string;
  persisted: boolean;
  cached: boolean;
  warnings: string[];
  disclaimer: string;
}

export interface QueuedAnalysisResponse {
  analysis_id: string;
  status: 'queued';
  poll_url: string;
  message: string;
  content_hash: string;
  created_at: string;
}

export interface ComponentHealth {
  name: string;
  enabled: boolean;
  available: boolean;
  detail: string | null;
}

export interface HealthResponse {
  status: 'ok' | 'degraded' | 'unavailable';
  version: string;
  environment: string;
  components: ComponentHealth[];
  detector: Record<string, unknown>;
  uptime_seconds: number;
  checked_at: string;
}

export interface ModelInfoResponse {
  ready: boolean;
  error: string | null;
  detector_version: string;
  model_version: string | null;
  dataset_version: string | null;
  features_version: string | null;
  trained_at: string | null;
  data_regime: string | null;
  document_model: {
    name: string | null;
    n_features: number;
    feature_groups: string[];
    calibration: string | null;
    classes: string[];
  };
  sentence_model: Record<string, unknown>;
  language_model: Record<string, unknown>;
  metrics: Record<string, unknown>;
  training: Record<string, unknown>;
  feature_importance: Array<{
    feature: string;
    group: string | null;
    importance: number;
    std: number;
  }>;
  model_comparison: Array<Record<string, unknown>>;
  methodology: {
    summary: string;
    pipeline: string[];
    what_the_language_model_does: string;
    what_makes_the_decision: string;
    signals_measured: string[];
    limitations: string[];
  };
  /** Hard bounds and the soft floors below which the detector abstains. Read
   *  from the server so the UI cannot claim a limit it does not enforce. */
  analysis_thresholds?: {
    min_chars: number;
    max_chars: number;
    min_sentences_for_verdict: number;
    min_words_for_verdict: number;
    note?: string;
  };
}

export interface PrivacyInfo {
  save_essays_default: boolean;
  per_request_override_supported: boolean;
  retention_days: number;
  what_is_stored: string[];
  what_is_never_stored: string[];
  what_is_never_logged: string[];
  deletion_endpoint: string;
}

export interface EvaluationBundle {
  available: boolean;
  report: EvaluationReport | null;
  failures: FailureReport | null;
  dataset: DatasetCard | null;
  message: string | null;
}

export interface PerClassMetrics {
  precision: number;
  recall: number;
  f1: number;
  support: number;
  true_positive: number;
  false_positive: number;
  false_negative: number;
  true_negative: number;
  false_positive_rate: number;
  false_negative_rate: number;
}

export interface EvaluationReport {
  run_id: string;
  created_at: string;
  split: string;
  data_regime: string | null;
  data_regime_note: string | null;
  model: Record<string, unknown>;
  dataset: Record<string, unknown>;
  overall: {
    n_samples: number;
    accuracy: number;
    balanced_accuracy: number;
    macro_f1: number;
    weighted_f1: number;
    cohen_kappa: number;
    matthews_corrcoef: number;
    per_class: Record<string, PerClassMetrics>;
    confusion_matrix: {
      labels: string[];
      matrix: number[][];
      row_normalised: number[][];
    };
    roc_auc_ovr_macro?: number;
    roc_auc_per_class?: Record<string, number | null>;
    log_loss?: number | null;
    brier_score_per_class?: Record<string, number>;
    expected_calibration_error?: number;
    reliability_curve?: Array<{
      bin_lower: number;
      bin_upper: number;
      mean_confidence: number;
      observed_accuracy: number;
      count: number;
    }>;
  };
  curves: {
    roc: Record<string, Array<{ fpr: number; tpr: number }>>;
    precision_recall: Record<
      string,
      { average_precision: number; points: Array<{ recall: number; precision: number }> }
    >;
  };
  generalisation: Record<string, unknown>;
  bias: {
    question: string;
    metric: string;
    groups: Record<
      string,
      {
        n_human_documents: number;
        measurable: boolean;
        false_positive_rate?: { point: number; lower: number; upper: number; n: number };
        flagged_as_ai_generated?: number;
        flagged_as_ai_polished?: number;
        mean_human_probability?: number;
      }
    >;
    disparity: Record<string, unknown>;
    severe_limitation: string;
    other_subgroups: Record<string, unknown>;
  };
  abstention: Record<string, unknown>;
  model_comparison: {
    protocol: string;
    estimator: string;
    models: Array<{
      feature_set: string;
      feature_groups: string[];
      n_features: number;
      accuracy: number;
      macro_f1: number;
      balanced_accuracy: number;
      roc_auc_ovr_macro?: number | null;
      human_recall: number;
      per_class_f1: Record<string, number>;
    }>;
    deltas: Record<string, number | null>;
  };
  feature_importance: Array<{
    feature: string;
    group: string | null;
    importance: number;
    std: number;
  }>;
  interpretation: string[];
  training_time_comparison?: Record<string, unknown>;
}

export interface ConfidentlyWrongCase {
  rank: number;
  record_id: string;
  actual: string;
  predicted: string;
  confidence: number;
  probabilities: Record<string, number>;
  metadata: Record<string, unknown>;
  why_the_model_likely_failed: string[];
  dominant_feature_groups: Array<{
    group: string;
    share_of_contribution: number;
    n_features: number;
  }>;
  relevant_features: Array<{
    feature: string;
    group: string | null;
    value: number;
    true_class_iqr: number[];
    true_class_median: number;
    predicted_class_median: number | null;
    iqr_widths_outside: number;
    direction: string;
  }>;
  model_contributions: FeatureContribution[];
  possible_improvement: string[];
  excerpt: string | null;
  excerpt_withheld_reason: string | null;
}

export interface FailureReport {
  created_at: string;
  split: string;
  model: Record<string, unknown>;
  data_regime: string | null;
  summary: Record<string, unknown>;
  confidently_wrong: ConfidentlyWrongCase[];
  false_positives_on_human_writing: {
    count: number;
    note: string;
    cases: Array<Record<string, unknown>>;
  };
  missed_machine_writing: { count: number; note: string; cases: Array<Record<string, unknown>> };
  ai_polished_confusions: { count: number; note: string; cases: Array<Record<string, unknown>> };
}

export interface DatasetCard {
  dataset_version: string;
  created_at: string;
  data_regime: string;
  regime_note: string;
  totals: Record<string, number>;
  labels: Record<string, number>;
  splits: {
    counts: Record<string, number>;
    labels_per_split: Record<string, Record<string, number>>;
    report: Record<string, unknown>;
  };
  sources: Record<string, number>;
  models: Record<string, number>;
  strategies: Record<string, number>;
  topics: Record<string, number>;
  length_bands: Record<string, number>;
  l2_english: Record<string, unknown>;
  licenses: Record<string, number>;
  preprocessing: string[];
  known_limitations: string[];
  leakage_controls: Record<string, unknown>;
}

export interface ApiErrorBody {
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
}
