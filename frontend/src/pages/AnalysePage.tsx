import { useCallback, useEffect, useState } from 'react';

import type { Route } from '@/App';
import { ApiError, analyseEssay, fetchPrivacy, pollAnalysis } from '@/api/client';
import { Banner } from '@/components/Banner';
import { EssayEditor } from '@/components/EssayEditor';
import { SignalStrip } from '@/components/SignalStrip';
import { Button } from '@/components/ui/Button';
import type { AnalysisResponse, HealthResponse, ModelInfoResponse } from '@/types/api';

interface Props {
  health: HealthResponse | null;
  modelInfo: ModelInfoResponse | null;
  backendReachable: boolean | null;
  onNavigate: (route: Route) => void;
  onAnalysed: (result: AnalysisResponse, submittedText: string) => void;
}

/** Fallbacks only. The live values come from `GET /model/info` so the UI can
 *  never advertise a limit the server does not enforce. */
const FALLBACK_THRESHOLDS = {
  min_chars: 200,
  max_chars: 60_000,
  min_sentences_for_verdict: 5,
  min_words_for_verdict: 120,
};

const STAGES = [
  'Segmenting sentences',
  'Measuring stylometry',
  'Scoring tokens with the language model',
  'Comparing against the essay baseline',
  'Classifying and generating evidence',
];

export function AnalysePage({
  health,
  modelInfo,
  backendReachable,
  onNavigate,
  onAnalysed,
}: Props) {
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState<ApiError | null>(null);
  const [saveOptOut, setSaveOptOut] = useState(false);
  const [savesEssays, setSavesEssays] = useState(false);

  // The persistence notice must reflect the server's live setting, not a guess.
  useEffect(() => {
    let cancelled = false;
    fetchPrivacy()
      .then((info) => {
        if (!cancelled) setSavesEssays(info.save_essays_default);
      })
      .catch(() => {
        /* The notice falls back to "not stored", which is the safe claim to make
           only because the request path also defaults to not storing. */
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const modelReady = modelInfo?.ready ?? health?.status !== 'unavailable';
  const thresholds = { ...FALLBACK_THRESHOLDS, ...(modelInfo?.analysis_thresholds ?? {}) };
  const bootstrapRegime = modelInfo?.data_regime === 'bootstrap';

  const runAnalysis = useCallback(async () => {
    setBusy(true);
    setError(null);
    setStage(0);

    // The stage ticker is a progress *indication*, not a report of backend state:
    // the synchronous endpoint returns one response. It advances on a timer and
    // stops at the last stage so it never claims completion the API has not sent.
    const ticker = window.setInterval(
      () => setStage((current) => Math.min(current + 1, STAGES.length - 1)),
      420,
    );

    try {
      const submitted = text;
      const response = await analyseEssay(submitted, {
        save: saveOptOut ? false : undefined,
      });
      const completed =
        response.kind === 'queued'
          ? await pollAnalysis(response.data.analysis_id)
          : response.data;
      onAnalysed(completed, submitted);
    } catch (caught) {
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError('Something went wrong while analysing the essay.', 'internal_error', 500),
      );
    } finally {
      window.clearInterval(ticker);
      setBusy(false);
    }
  }, [text, saveOptOut, onAnalysed]);

  return (
    <div className="stack stack--lg">
      {backendReachable === false && (
        <Banner tone="danger" title="Backend unreachable">
          The API is not responding. Start it with{' '}
          <code>uv run uvicorn app.main:app --reload --port 8000</code> in the{' '}
          <code>backend</code> directory, then reload this page.
        </Banner>
      )}

      {backendReachable && !modelReady && (
        <Banner tone="danger" title="Detector not trained">
          The model artifacts are missing, so analysis is unavailable. Run the training
          pipeline:{' '}
          <code>
            uv run python -m ml.training.prepare_dataset && uv run python -m
            ml.training.extract_features && uv run python -m ml.training.train
          </code>
        </Banner>
      )}

      {bootstrapRegime && (
        <Banner tone="warning" title="This detector is trained on bootstrap data">
          The corpus behind this model is synthetic: hand-authored seed essays for the human
          class, and a template generator plus a rule-based editor for the machine classes. Any
          verdict below is a demonstration of the pipeline, not a reliable measurement of
          authorship.{' '}
          <Button variant="ghost" size="sm" onClick={() => onNavigate('research')}>
            See the evaluation
          </Button>
        </Banner>
      )}

      <EssayEditor
        value={text}
        onChange={setText}
        onAnalyse={runAnalysis}
        onClear={() => setText('')}
        busy={busy}
        disabled={backendReachable === false || !modelReady}
        minChars={thresholds.min_chars}
        maxChars={thresholds.max_chars}
        minSentences={thresholds.min_sentences_for_verdict}
        minWords={thresholds.min_words_for_verdict}
        savesEssays={savesEssays}
        saveOptOut={saveOptOut}
        onSaveOptOutChange={setSaveOptOut}
      />

      {busy && (
        <div className="progress" role="status" aria-live="polite">
          <div className="progress__track">
            <div className="progress__bar" />
          </div>
          <div className="progress__steps">
            {STAGES.map((label, index) => (
              <span key={label} className={index <= stage ? 'progress__step--active' : undefined}>
                {index <= stage ? '•' : '○'} {label}
              </span>
            ))}
          </div>
        </div>
      )}

      {error && (
        <Banner
          tone={error.code === 'rate_limit_exceeded' ? 'warning' : 'danger'}
          title="Analysis failed"
          action={
            error.retryable ? (
              <Button size="sm" onClick={runAnalysis}>
                Try again
              </Button>
            ) : undefined
          }
        >
          {error.message}
        </Banner>
      )}

      {!busy && <SignalStrip />}
    </div>
  );
}
