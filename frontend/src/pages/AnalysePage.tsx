import { motion } from 'framer-motion';
import { useCallback, useEffect, useMemo, useState } from 'react';

import type { Route } from '@/App';
import { fadeUp, stagger } from '@/hooks/useMotion';
import { ApiError, analyseEssay, fetchPrivacy, pollAnalysis } from '@/api/client';
import { Banner } from '@/components/Banner';
import { EssayEditor } from '@/components/EssayEditor';
import { EvidencePanel } from '@/components/EvidencePanel';
import { HighlightedEssay, ParagraphBreakdown } from '@/components/HighlightedEssay';
import { RepetitionPanel } from '@/components/RepetitionPanel';
import { RhythmChart } from '@/components/RhythmChart';
import { SignalStrip } from '@/components/SignalStrip';
import { SummaryStats } from '@/components/SummaryStats';
import { VerdictCard } from '@/components/VerdictCard';
import type { AnalysisResponse, HealthResponse, ModelInfoResponse } from '@/types/api';

interface Props {
  health: HealthResponse | null;
  modelInfo: ModelInfoResponse | null;
  backendReachable: boolean | null;
  onNavigate: (route: Route) => void;
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

export function AnalysePage({ health, modelInfo, backendReachable, onNavigate }: Props) {
  const [text, setText] = useState('');
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [analysedText, setAnalysedText] = useState('');
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState<ApiError | null>(null);
  const [selectedSentenceId, setSelectedSentenceId] = useState<number | null>(null);
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

  const selectedSentence = useMemo(() => {
    if (!result || selectedSentenceId === null) return null;
    return result.sentences.find((s) => s.sentence_id === selectedSentenceId) ?? null;
  }, [result, selectedSentenceId]);

  const runAnalysis = useCallback(async () => {
    setBusy(true);
    setError(null);
    setSelectedSentenceId(null);
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
      let completed: AnalysisResponse;
      if (response.kind === 'queued') {
        completed = await pollAnalysis(response.data.analysis_id);
      } else {
        completed = response.data;
      }
      setResult(completed);
      setAnalysedText(submitted);
    } catch (caught) {
      setResult(null);
      setError(
        caught instanceof ApiError
          ? caught
          : new ApiError('Something went wrong while analysing the essay.', 'internal_error', 500),
      );
    } finally {
      window.clearInterval(ticker);
      setBusy(false);
    }
  }, [text, saveOptOut]);

  const clear = useCallback(() => {
    setText('');
    setResult(null);
    setAnalysedText('');
    setError(null);
    setSelectedSentenceId(null);
  }, []);

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
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={() => onNavigate('research')}
          >
            See the evaluation
          </button>
        </Banner>
      )}

      {!result && (
        <>
          <EssayEditor
            value={text}
            onChange={setText}
            onAnalyse={runAnalysis}
            onClear={clear}
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
          {!busy && <SignalStrip />}
        </>
      )}

      {busy && (
        <div className="progress" role="status" aria-live="polite">
          <div className="progress__track">
            <div className="progress__bar" />
          </div>
          <div className="progress__steps">
            {STAGES.map((label, index) => (
              <span
                key={label}
                className={index <= stage ? 'progress__step--active' : undefined}
              >
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
              <button type="button" className="btn btn--sm" onClick={runAnalysis}>
                Try again
              </button>
            ) : undefined
          }
        >
          {error.message}
        </Banner>
      )}

      {result && (
        <>
          <div className="row row--wrap">
            <button type="button" className="btn btn--sm" onClick={() => setResult(null)}>
              ← Edit the essay
            </button>
            <button type="button" className="btn btn--sm" onClick={clear}>
              Start over
            </button>
            <div className="spacer" />
            <span className="tag">
              {result.cached ? 'served from cache' : 'freshly analysed'}
            </span>
            <span className="tag">
              {result.persisted ? 'saved' : 'not saved'}
            </span>
            {result.timings.total_ms !== undefined && (
              <span className="tag mono">{Math.round(result.timings.total_ms)} ms</span>
            )}
          </div>

          {result.warnings.length > 0 && (
            <Banner tone="warning" title="Caveats for this analysis">
              <ul style={{ margin: '0.2rem 0 0', paddingLeft: '1.1rem' }}>
                {result.warnings.map((warning, index) => (
                  <li key={index} style={{ marginBottom: '0.2rem' }}>
                    {warning}
                  </li>
                ))}
              </ul>
            </Banner>
          )}

          <motion.div
            className="results"
            variants={stagger}
            initial="hidden"
            animate="visible"
          >
            <motion.div variants={fadeUp}>
              <VerdictCard result={result} />
            </motion.div>

            <motion.section className="card" variants={fadeUp}>
              <div className="card__head">
                <div>
                  <p className="card__title">The essay, marked up</p>
                  <p className="section-note">
                    Click or hover a sentence to see the measurements behind it.
                  </p>
                </div>
              </div>
              <div className="card__body">
                <div className="reader-grid">
                  <HighlightedEssay
                    result={result}
                    essayText={analysedText}
                    selectedSentenceId={selectedSentenceId}
                    onSelectSentence={setSelectedSentenceId}
                  />
                  <EvidencePanel
                    sentence={selectedSentence}
                    documentEvidence={result.evidence}
                    hasSentenceModel={result.summary.sentences_scored > 0}
                  />
                </div>
              </div>
            </motion.section>

            <motion.section className="card" variants={fadeUp}>
              <div className="card__head">
                <div>
                  <p className="card__title">Sentence rhythm</p>
                  <p className="section-note">
                    Sentence lengths across the essay, against its own mean. Variation here is
                    one feature among many — uniformity alone is not evidence of anything.
                  </p>
                </div>
              </div>
              <div className="card__body">
                <RhythmChart
                  rhythm={result.rhythm}
                  sentences={result.sentences}
                  onSelectSentence={setSelectedSentenceId}
                  selectedSentenceId={selectedSentenceId}
                />
              </div>
            </motion.section>

            <motion.div
              variants={fadeUp}
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(20rem, 1fr))',
                gap: '1.5rem',
              }}
            >
              <section className="card">
                <div className="card__head">
                  <div>
                    <p className="card__title">Paragraph breakdown</p>
                    <p className="section-note">Weighted by sentence length.</p>
                  </div>
                </div>
                <div className="card__body">
                  <ParagraphBreakdown
                    paragraphs={result.paragraphs}
                    sentences={result.sentences}
                    onSelectSentence={setSelectedSentenceId}
                  />
                </div>
              </section>

              <section className="card">
                <div className="card__head">
                  <div>
                    <p className="card__title">Repetition</p>
                    <p className="section-note">Concrete repeated spans and templates.</p>
                  </div>
                </div>
                <div className="card__body">
                  <RepetitionPanel
                    repetition={result.repetition}
                    onSelectSentence={setSelectedSentenceId}
                  />
                </div>
              </section>
            </motion.div>

            <motion.section className="card" variants={fadeUp}>
              <div className="card__head">
                <div>
                  <p className="card__title">All measured statistics</p>
                  <p className="section-note">
                    Every number below came from this analysis. Nothing is estimated in the
                    browser.
                  </p>
                </div>
              </div>
              <div className="card__body">
                <SummaryStats summary={result.summary} />
              </div>
            </motion.section>

            <Banner tone="info" title="What this result is">
              {result.disclaimer}{' '}
              <button
                type="button"
                className="btn btn--ghost btn--sm"
                onClick={() => onNavigate('how')}
              >
                How this works
              </button>
            </Banner>

            <p className="tiny muted" style={{ textAlign: 'center' }}>
              analysis {result.analysis_id.slice(0, 12)} · detector v
              {result.model.detector_version} · model v{result.model.model_version} · features v
              {result.model.features_version} · instrument {result.model.language_model}
            </p>
          </motion.div>
        </>
      )}
    </div>
  );
}
