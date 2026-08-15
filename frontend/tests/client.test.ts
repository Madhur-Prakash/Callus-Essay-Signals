import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ApiError, analyseEssay, fetchEvaluation } from '@/api/client';

import { SAMPLE_ANALYSIS } from './fixtures';

function mockFetch(status: number, body: unknown, ok = status < 400) {
  const fetchMock = vi.fn().mockResolvedValue({
    ok,
    status,
    text: async () => (body === undefined ? '' : JSON.stringify(body)),
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

describe('analyseEssay', () => {
  beforeEach(() => vi.unstubAllGlobals());

  it('returns a completed analysis', async () => {
    mockFetch(200, SAMPLE_ANALYSIS);
    const result = await analyseEssay('a fairly long essay body');
    expect(result.kind).toBe('completed');
    if (result.kind === 'completed') {
      expect(result.data.classification).toBe('ai_polished');
      expect(result.data.sentences).toHaveLength(4);
    }
  });

  it('recognises a queued response', async () => {
    mockFetch(202, {
      analysis_id: 'q1',
      status: 'queued',
      poll_url: '/api/v1/analysis/q1',
      message: 'queued',
      content_hash: 'x',
      created_at: 'now',
    });
    const result = await analyseEssay('essay');
    expect(result.kind).toBe('queued');
  });

  it('sends the opt-out flag when asked not to save', async () => {
    const fetchMock = mockFetch(200, SAMPLE_ANALYSIS);
    await analyseEssay('essay', { save: false });
    const body = JSON.parse(String(fetchMock.mock.calls[0]![1]!.body));
    expect(body.save).toBe(false);
  });

  it('maps a too-short essay to a readable message', async () => {
    mockFetch(422, { error: { code: 'essay_too_short', message: 'raw backend message' } });
    await expect(analyseEssay('hi')).rejects.toMatchObject({
      code: 'essay_too_short',
      // The friendly copy is used, not the backend's terse message.
      message: expect.stringContaining('too short to analyse reliably'),
    });
  });

  it('maps an untrained model to a readable, non-retryable error', async () => {
    mockFetch(503, { error: { code: 'model_not_trained', message: 'nope' } });
    const error = await analyseEssay('essay').catch((e) => e as ApiError);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toContain('has not been trained');
    expect((error as ApiError).retryable).toBe(false);
  });

  it('marks a rate limit as retryable', async () => {
    mockFetch(429, { error: { code: 'rate_limit_exceeded', message: 'slow down' } });
    const error = await analyseEssay('essay').catch((e) => e as ApiError);
    expect((error as ApiError).retryable).toBe(true);
  });

  it('reports an unreachable backend rather than throwing a raw fetch error', async () => {
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(new TypeError('failed to fetch')));
    const error = await analyseEssay('essay').catch((e) => e as ApiError);
    expect((error as ApiError).code).toBe('network_error');
    expect((error as ApiError).message).toContain('Could not reach the backend');
  });

  it('rejects a malformed success response', async () => {
    mockFetch(200, { analysis_id: 'x', status: 'completed' });
    const error = await analyseEssay('essay').catch((e) => e as ApiError);
    expect((error as ApiError).code).toBe('malformed_response');
  });

  it('rejects a non-JSON body', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: true, status: 200, text: async () => '<html>' }),
    );
    const error = await analyseEssay('essay').catch((e) => e as ApiError);
    expect((error as ApiError).code).toBe('malformed_response');
  });
});

describe('fetchEvaluation', () => {
  it('passes through the unavailable state instead of erroring', async () => {
    mockFetch(200, {
      available: false,
      report: null,
      failures: null,
      dataset: null,
      message: 'run the pipeline',
    });
    const bundle = await fetchEvaluation();
    expect(bundle.available).toBe(false);
    expect(bundle.message).toBe('run the pipeline');
  });
});
