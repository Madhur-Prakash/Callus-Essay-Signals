import { motion } from 'framer-motion';

import { Gauge, Stat } from '@/components/ui';
import { EASE } from '@/hooks/useMotion';
import { CLASS_SHORT, classColour, percent } from '@/lib/format';
import type { AnalysisResponse } from '@/types/api';

interface Props {
  result: AnalysisResponse;
}

const PROB_ORDER = ['human', 'ai_polished', 'ai_generated'];

/**
 * The answer to the question the user asked, and the first thing on the page.
 *
 * The gauge shows the *leading class probability*, not "how AI this is" - those
 * are different claims, and the caption says which one it is. A gauge that filled
 * toward "AI" would turn a three-class calibrated distribution into a single
 * accusatory number, which is exactly the reading this product exists to avoid.
 */
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
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, ease: EASE }}
            className="flex-none"
          >
            <Gauge
              value={result.confidence_score}
              colour={colour}
              caption="leading class"
              size={168}
              ariaLabel={`Leading class probability ${percent(result.confidence_score)}`}
            />
          </motion.div>

          <div className="min-w-0">
            <p className="verdict__eyebrow">Overall assessment</p>
            <h2 className="verdict__label" id="verdict-heading">
              {result.label}
            </h2>
            <p className="verdict__description">{result.description}</p>

            <dl className="verdict__confidence">
              <dt>Confidence</dt>
              <dd style={{ color: colour }}>{result.confidence}</dd>
              <dd className="muted verdict__confidence-note">
                (margin over the next {percent(result.margin)})
              </dd>
            </dl>

            {result.abstained && result.abstain_reason && (
              <p className="verdict__abstain">
                <strong>Why no class was named: </strong>
                {result.abstain_reason}
              </p>
            )}

            {summary.flagged_paragraphs > 0 && (
              <p className="small muted mb-0 mt-4">
                Evidence detected in <strong>{summary.flagged_paragraphs}</strong> of{' '}
                <strong>{summary.n_paragraphs}</strong> paragraphs.
              </p>
            )}
          </div>
        </div>

        <div className="verdict__side">
          <p className="subhead">Calibrated probabilities</p>
          <dl className="probs">
            {PROB_ORDER.filter((key) => key in result.probabilities).map((key, index) => {
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
                      transition={{ duration: 0.9, ease: EASE, delay: 0.2 + index * 0.08 }}
                    />
                  </div>
                </div>
              );
            })}
          </dl>
          <p className="tiny muted mb-0 mt-5">
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
