import { MeterBar } from '@/components/MeterBar';
import { SENTENCE_LABELS, classColour, humaniseFeature, percent } from '@/lib/format';
import type { EvidenceBlock, FeatureContribution, SentenceResult } from '@/types/api';

interface Props {
  sentence: SentenceResult | null;
  documentEvidence: EvidenceBlock;
  hasSentenceModel: boolean;
}

export function EvidencePanel({ sentence, documentEvidence, hasSentenceModel }: Props) {
  if (!sentence) {
    return (
      <aside className="panel">
        <div className="panel__head">
          <div>
            <p className="card__title">Evidence</p>
            <p className="section-note">Whole essay</p>
          </div>
        </div>
        <div className="panel__body">
          <EvidenceBody evidence={documentEvidence} />
          {!hasSentenceModel && (
            <p className="tiny muted">
              The sentence-level model is unavailable, so per-sentence evidence cannot be
              shown.
            </p>
          )}
          <p className="tiny muted mt-auto">
            Hover or click any sentence in the essay to see the measurements behind it.
          </p>
        </div>
      </aside>
    );
  }

  const evidence = sentence.evidence;

  return (
    <aside className="panel">
      <div className="panel__head">
        <div className="min-w-0 flex-1">
          <p className="card__title">Why this was flagged</p>
          <p className="section-note">
            Sentence {sentence.sentence_id + 1} · paragraph {sentence.paragraph_id + 1} ·{' '}
            {sentence.n_words} words
          </p>
        </div>
        <span
          className="tag"
          style={{
            color: classColour(sentence.classification),
            borderColor: classColour(sentence.classification),
          }}
        >
          {sentence.score !== null ? percent(sentence.score, 0) : '—'}
        </span>
      </div>

      <div className="panel__body">
        <blockquote className="panel__quote">{sentence.text || '[text not stored]'}</blockquote>

        <div>
          <p className="subhead">Classification</p>
          <p className="m-0 text-xs">
            <strong style={{ color: classColour(sentence.classification) }}>
              {SENTENCE_LABELS[sentence.classification]}
            </strong>
            <span className="muted"> · confidence {sentence.confidence}</span>
          </p>
          {sentence.classification === 'uncertain' && sentence.n_words < 5 && (
            <p className="tiny muted mb-0 mt-1">
              Sentences this short give the language model too few predictions to support a
              claim either way, so they are always reported as uncertain.
            </p>
          )}
        </div>

        {evidence ? (
          <EvidenceBody evidence={evidence} />
        ) : (
          <p className="small muted">
            This sentence measured within ordinary ranges, so no evidence was generated for
            it. Evidence is attached to flagged and uncertain sentences only.
          </p>
        )}
      </div>
    </aside>
  );
}

function EvidenceBody({ evidence }: { evidence: EvidenceBlock }) {
  const meters = evidence.meters.filter((m) => m.available);
  const unavailable = evidence.meters.length - meters.length;

  return (
    <>
      {meters.length > 0 && (
        <div>
          <p className="subhead">Signals</p>
          {meters.map((meter) => (
            <MeterBar key={meter.key} meter={meter} />
          ))}
          {unavailable > 0 && (
            <p className="tiny muted">
              {unavailable} measurement{unavailable > 1 ? 's' : ''} had no training reference
              to compare against.
            </p>
          )}
        </div>
      )}

      {evidence.statements.length > 0 && (
        <div>
          <p className="subhead">What was measured</p>
          <ul className="statements">
            {evidence.statements.map((statement, index) => (
              <li key={index}>{statement}</li>
            ))}
          </ul>
        </div>
      )}

      {evidence.measurements.length > 0 && (
        <div>
          <p className="subhead">Numbers</p>
          <dl className="measure-list">
            {evidence.measurements.map((measurement) => (
              <div className="measure" key={`${measurement.key}-${measurement.name}`}>
                <dt className="measure__name">{measurement.name}</dt>
                <dd className="measure__value">{measurement.unit}</dd>
                <dd className="measure__ref">{measurement.reference}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {evidence.model_contributions.length > 0 && (
        <div>
          <p className="subhead">Model contributions</p>
          <p className="tiny muted -mt-1">
            The classifier's own arithmetic: each feature's weight times its standardised
            value. Red pushes toward machine, green toward human.
          </p>
          {evidence.model_contributions.slice(0, 8).map((contribution) => (
            <ContributionRow key={contribution.feature} contribution={contribution} />
          ))}
        </div>
      )}

      <p className="tiny muted mb-0">
        Generated deterministically from measured values by explanation engine v
        {evidence.engine_version || '—'}. No language model was asked to explain anything.
      </p>
    </>
  );
}

function ContributionRow({ contribution }: { contribution: FeatureContribution }) {
  const magnitude = Math.min(1, Math.abs(contribution.contribution) / 0.5);
  const isMachine = contribution.contribution > 0;
  return (
    <div className="contrib">
      <span className="contrib__name" title={contribution.feature}>
        {humaniseFeature(contribution.feature)}
      </span>
      <span className="contrib__value">{contribution.contribution.toFixed(3)}</span>
      <span className="contrib__bar-wrap">
        <span className="contrib__neg">
          {!isMachine && <span style={{ width: `${magnitude * 100}%` }} />}
        </span>
        <span className="contrib__pos">
          {isMachine && <span style={{ width: `${magnitude * 100}%` }} />}
        </span>
      </span>
    </div>
  );
}
