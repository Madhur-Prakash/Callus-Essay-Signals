import { motion } from 'framer-motion';
import { useCallback } from 'react';

import { EASE, useCountUp } from '@/hooks/useMotion';
import { CLASS_SHORT, classColour, percent } from '@/lib/format';
import type { AnalysisResponse } from '@/types/api';

interface Props {
  result: AnalysisResponse;
}

const PROB_ORDER = ['human', 'ai_polished', 'ai_generated'];

export function VerdictCard({ result }: Props) {
  const { summary } = result;
  const colour = classColour(result.classification);

  return (
    <section
      className="verdict"
      aria-labelledby="verdict-heading"
      style={{ ['--verdict-colour' as string]: colour }}
    >
      <div className="verdict__top">
        <div className="verdict__main">
          <p className="verdict__eyebrow">Overall assessment</p>
          <h2 className="verdict__label" id="verdict-heading">
            <span className="verdict__dot" aria-hidden="true" />
            {result.label}
          </h2>
          <p className="verdict__description">{result.description}</p>

          <dl className="verdict__confidence">
            <dt>Confidence</dt>
            <dd style={{ color: colour }}>{result.confidence}</dd>
            <dd className="muted verdict__confidence-note">
              (leading class {percent(result.confidence_score)}, margin over the next{' '}
              {percent(result.margin)})
            </dd>
          </dl>

          {result.abstained && result.abstain_reason && (
            <p className="verdict__abstain">
              <strong>Why no class was named: </strong>
              {result.abstain_reason}
            </p>
          )}

          {summary.flagged_paragraphs > 0 && (
            <p className="small muted mb-0 mt-3">
              Evidence detected in <strong>{summary.flagged_paragraphs}</strong> of{' '}
              <strong>{summary.n_paragraphs}</strong> paragraphs.
            </p>
          )}
        </div>

        <div className="verdict__side">
          <p className="subhead">Calibrated probabilities</p>
          <dl className="probs">
            {PROB_ORDER.filter((key) => key in result.probabilities).map((key) => {
              const value = result.probabilities[key] ?? 0;
              return (
                <div className="prob" key={key}>
                  <dt className="prob__name">{CLASS_SHORT[key] ?? key}</dt>
                  <dd className="prob__value">{percent(value, 1)}</dd>
                  <div className="prob__track">
                    <motion.div
                      className="prob__fill"
                      style={{ background: classColour(key) }}
                      initial={{ width: 0 }}
                      animate={{ width: `${Math.max(1, value * 100)}%` }}
                      transition={{ duration: 0.8, ease: EASE, delay: 0.1 }}
                    />
                  </div>
                </div>
              );
            })}
          </dl>
          <p className="tiny muted mb-0 mt-3">
            Calibrated with Platt scaling on a held-out split. These are estimates from a
            model trained on a specific corpus, not measurements of truth.
          </p>
        </div>
      </div>

      <div className="verdict__strip">
        <Stat value={summary.n_sentences} label="sentences analysed" />
        <Stat value={summary.n_paragraphs} label="paragraphs analysed" />
        <Stat
          value={summary.flagged_sentences}
          label="flagged sentences"
          colour={summary.flagged_sentences > 0 ? 'var(--likely)' : undefined}
        />
        <Stat
          value={summary.uncertain_sentences}
          label="uncertain sentences"
          colour={summary.uncertain_sentences > 0 ? 'var(--uncertain)' : undefined}
        />
        <Stat
          value={summary.human_like_sentences}
          label="human-like sentences"
          colour="var(--human)"
        />
      </div>
    </section>
  );
}

function Stat({
  value,
  label,
  colour,
}: {
  value: number;
  label: string;
  colour?: string;
}) {
  // GSAP drives the number; the formatter is shared with the final value so a
  // mid-flight frame can never render differently from the settled one.
  const format = useCallback((n: number) => Math.round(n).toLocaleString(), []);
  const ref = useCountUp(value, format);

  return (
    <div className="stat">
      <div
        className="stat__value"
        style={colour ? { color: colour } : undefined}
        ref={ref as React.RefObject<HTMLDivElement>}
      >
        {value.toLocaleString()}
      </div>
      <div className="stat__label">{label}</div>
    </div>
  );
}
