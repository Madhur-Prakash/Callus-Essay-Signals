/**
 * API client.
 *
 * Every failure mode the brief calls out is turned into a typed `ApiError` with a
 * message written for a person: backend unreachable, model not trained, essay too
 * short or long, rate limited, timeout, malformed response. The UI renders
 * `error.message` directly, so no component has to invent copy for an error case.
 */

import type {
  AnalysisResponse,
  ApiErrorBody,
  EvaluationBundle,
  HealthResponse,
  ModelInfoResponse,
  PrivacyInfo,
  QueuedAnalysisResponse,
} from '@/types/api';

const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? '/api/v1';

export class ApiError extends Error {
  readonly code: string;
  readonly status: number;
  readonly details?: Record<string, unknown>;
  readonly retryable: boolean;

  constructor(
    message: string,
    code: string,
    status: number,
    details?: Record<string, unknown>,
    retryable = false,
  ) {
    super(message);
    this.name = 'ApiError';
    this.code = code;
    this.status = status;
    this.details = details;
    this.retryable = retryable;
  }
}

const FRIENDLY_MESSAGES: Record<string, string> = {
  essay_empty: 'Please paste an essay before running the analysis.',
  essay_too_short:
    'This text is too short to analyse reliably. The detector needs enough sentences to measure variation — around 200 characters at minimum.',
  essay_too_long: 'This essay exceeds the maximum supported length. Please shorten it and try again.',
  model_not_trained:
    'The detector has not been trained yet. Run the training pipeline on the backend, then reload.',
  model_unavailable:
    'The language model could not be loaded on the server, so analysis is unavailable.',
  rate_limit_exceeded: 'Too many requests. Please wait a moment before analysing again.',
  analysis_timeout:
    'The analysis took too long and was cancelled. Try a shorter essay.',
  persistence_unavailable:
    'The database is unavailable, so previously saved analyses cannot be loaded.',
  analysis_not_found: 'That analysis no longer exists.',
  validation_error: 'The request was rejected by the server as invalid.',
  request_too_large: 'That essay is too large to send. Please shorten it.',
  network_error:
    'Could not reach the backend. Check that the API is running on port 8000, then try again.',
  malformed_response: 'The server returned a response this app could not read.',
};

const RETRYABLE_CODES = new Set([
  'network_error',
  'internal_error',
  'analysis_timeout',
  'rate_limit_exceeded',
]);

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: {
        'Content-Type': 'application/json',
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new ApiError(FRIENDLY_MESSAGES.network_error!, 'network_error', 0, undefined, true);
  }

  const raw = await response.text();
  let body: unknown = null;
  if (raw) {
    try {
      body = JSON.parse(raw);
    } catch {
      if (!response.ok) {
        throw new ApiError(
          `The server returned an error (HTTP ${response.status}).`,
          'internal_error',
          response.status,
          undefined,
          true,
        );
      }
      throw new ApiError(FRIENDLY_MESSAGES.malformed_response!, 'malformed_response', 200);
    }
  }

  if (!response.ok) {
    const errorBody = body as ApiErrorBody | null;
    const code = errorBody?.error?.code ?? `http_${response.status}`;
    const message =
      FRIENDLY_MESSAGES[code] ??
      errorBody?.error?.message ??
      `The request failed (HTTP ${response.status}).`;
    throw new ApiError(
      message,
      code,
      response.status,
      errorBody?.error?.details,
      RETRYABLE_CODES.has(code),
    );
  }

  if (body === null) {
    throw new ApiError(FRIENDLY_MESSAGES.malformed_response!, 'malformed_response', 200);
  }
  return body as T;
}

export interface AnalyseOptions {
  save?: boolean;
  asyncMode?: boolean;
}

export type AnalyseResult =
  | { kind: 'completed'; data: AnalysisResponse }
  | { kind: 'queued'; data: QueuedAnalysisResponse };

export async function analyseEssay(
  text: string,
  options: AnalyseOptions = {},
): Promise<AnalyseResult> {
  const payload = await request<AnalysisResponse | QueuedAnalysisResponse>('/analysis', {
    method: 'POST',
    body: JSON.stringify({
      text,
      save: options.save ?? null,
      async_mode: options.asyncMode ?? false,
    }),
  });

  if (payload.status === 'queued') {
    return { kind: 'queued', data: payload as QueuedAnalysisResponse };
  }
  const completed = payload as AnalysisResponse;
  if (!completed.summary || !Array.isArray(completed.sentences)) {
    throw new ApiError(FRIENDLY_MESSAGES.malformed_response!, 'malformed_response', 200);
  }
  return { kind: 'completed', data: completed };
}

export function fetchAnalysis(analysisId: string): Promise<AnalysisResponse> {
  return request<AnalysisResponse>(`/analysis/${encodeURIComponent(analysisId)}`);
}

export function fetchHealth(): Promise<HealthResponse> {
  return request<HealthResponse>('/health');
}

export function fetchModelInfo(): Promise<ModelInfoResponse> {
  return request<ModelInfoResponse>('/model/info');
}

export function fetchEvaluation(): Promise<EvaluationBundle> {
  return request<EvaluationBundle>('/evaluation');
}

export function fetchPrivacy(): Promise<PrivacyInfo> {
  return request<PrivacyInfo>('/essays/privacy');
}

export function deleteAnalysis(analysisId: string): Promise<{ deleted: boolean }> {
  return request<{ deleted: boolean }>(`/analysis/${encodeURIComponent(analysisId)}`, {
    method: 'DELETE',
  });
}

/**
 * Poll a queued analysis until it completes. Only used when the backend has the
 * Kafka path enabled and decides to queue a large essay.
 */
export async function pollAnalysis(
  analysisId: string,
  {
    intervalMs = 1500,
    timeoutMs = 180_000,
    onTick,
  }: { intervalMs?: number; timeoutMs?: number; onTick?: (status: string) => void } = {},
): Promise<AnalysisResponse> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const payload = await fetchAnalysis(analysisId);
    onTick?.(payload.status);
    if (payload.status === 'completed') return payload;
    if (payload.status === 'failed') {
      throw new ApiError('The background analysis failed on the server.', 'internal_error', 500);
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
  }
  throw new ApiError(
    'The background analysis did not finish in time.',
    'analysis_timeout',
    504,
    undefined,
    true,
  );
}
