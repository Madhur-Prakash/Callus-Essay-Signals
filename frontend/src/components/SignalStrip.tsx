/**
 * "What gets measured" - shown below the editor before an analysis runs.
 *
 * It exists for two reasons: the landing view was mostly empty space, and a user
 * about to paste a personal essay deserves to know what the machine is going to
 * look at before they hand it over. The icons are drawn from the measurements
 * themselves (a rhythm trace, a probability curve, a repeated template) rather
 * than a generic icon set.
 */

import { Surface } from '@/components/ui';
import { useScrollReveal } from '@/hooks/useMotion';

const SIGNALS: Array<{ name: string; desc: string; icon: JSX.Element }> = [
  {
    name: 'Token predictability',
    desc: 'How surprising each word is to a local language model - log probability, entropy and rank.',
    icon: (
      <svg className="signal__icon" viewBox="0 0 32 16" aria-hidden="true">
        <polyline points="1,14 6,4 11,9 16,2 21,10 26,6 31,12" />
      </svg>
    ),
  },
  {
    name: 'Sentence rhythm',
    desc: 'Variation in sentence length across the essay - the burstiness humans tend to have and edits tend to remove.',
    icon: (
      <svg className="signal__icon" viewBox="0 0 32 16" aria-hidden="true">
        <rect x="1" y="8" width="3" height="8" rx="1" />
        <rect x="7" y="2" width="3" height="14" rx="1" />
        <rect x="13" y="11" width="3" height="5" rx="1" />
        <rect x="19" y="5" width="3" height="11" rx="1" />
        <rect x="25" y="9" width="3" height="7" rx="1" />
      </svg>
    ),
  },
  {
    name: 'Vocabulary & syntax',
    desc: 'Lexical diversity, punctuation habits, part-of-speech and dependency structure.',
    icon: (
      <svg className="signal__icon" viewBox="0 0 32 16" aria-hidden="true">
        <path d="M16 2 v5 M16 7 H7 v4 M16 7 h9 v4" />
        <rect x="4" y="11" width="6" height="4" rx="1" />
        <rect x="22" y="11" width="6" height="4" rx="1" />
        <rect x="13" y="0" width="6" height="3" rx="1" />
      </svg>
    ),
  },
  {
    name: 'Repetition',
    desc: 'Repeated phrases and repeated grammatical templates - the latter survives paraphrasing.',
    icon: (
      <svg className="signal__icon" viewBox="0 0 32 16" aria-hidden="true">
        <rect x="1" y="2" width="13" height="4" rx="1" />
        <rect x="1" y="10" width="13" height="4" rx="1" />
        <path d="M18 4 h12 M18 12 h8" />
      </svg>
    ),
  },
  {
    name: 'Style shift',
    desc: "Where a passage departs from the author's own baseline - the localised-edit case.",
    icon: (
      <svg className="signal__icon" viewBox="0 0 32 16" aria-hidden="true">
        <polyline points="1,11 5,10 9,11 13,3 17,4 21,3 25,11 29,10" />
      </svg>
    ),
  },
  {
    name: 'Corpus similarity',
    desc: 'Distance to human and machine reference corpora, using topic-free representations.',
    icon: (
      <svg className="signal__icon" viewBox="0 0 32 16" aria-hidden="true">
        <path d="M9 8 a5 5 0 1 0 0.01 0" />
        <path d="M23 8 a5 5 0 1 0 0.01 0" />
        <path d="M14 8 h4" />
      </svg>
    ),
  },
];

export function SignalStrip() {
  // GSAP ScrollTrigger staggers the cards in as the strip enters the viewport.
  const ref = useScrollReveal('.signal');

  return (
    <section
      className="signals"
      aria-labelledby="signals-heading"
      ref={ref as React.RefObject<HTMLElement>}
    >
      <div className="signals__head">
        <span className="eyebrow">
          <span className="eyebrow__dot" aria-hidden="true" />
          Six families of measurement
        </span>
        <h2 id="signals-heading">What gets measured</h2>
        <p className="section-note text-center">
          No chat model is asked whether your essay is AI-written. These properties are
          measured, then a classifier we trained weighs them.
        </p>
      </div>
      <div className="signals__grid">
        {SIGNALS.map((signal, index) => (
          // Spotlight is affordable here and nowhere else: this is the one grid
          // on the landing view, and a light that follows the pointer over six
          // cards reads as exploration. Over a table of numbers it reads as noise.
          <Surface
            as="article"
            key={signal.name}
            className="signal"
            spotlight
            interactive
          >
            <span className="signal__index" aria-hidden="true">
              {String(index + 1).padStart(2, '0')}
            </span>
            {signal.icon}
            <h3 className="signal__name">{signal.name}</h3>
            <p className="signal__desc">{signal.desc}</p>
          </Surface>
        ))}
      </div>
    </section>
  );
}
