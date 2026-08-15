import { AnimatePresence, motion } from 'framer-motion';
import { useMemo, useState } from 'react';

import { Button } from '@/components/ui/Button';
import { EASE, useDrawPath } from '@/hooks/useMotion';
import { EXAMPLE_ESSAYS } from '@/lib/exampleEssays';
import { countParagraphs, countSentences, countWords } from '@/lib/format';

/**
 * The hero motif: a sentence-length trace over its own bars.
 *
 * It is the one decorative element in the interface, and it is drawn from the
 * thing the product actually measures — uneven sentence lengths around a mean —
 * rather than an abstract flourish. Values are fixed, not fabricated data about
 * anyone's essay.
 */
const WAVE = [14, 31, 8, 22, 11, 38, 17, 6, 26, 13, 29, 9, 20, 34, 12];

function HeroWave() {
  // GSAP measures the real path length, so the draw stays correct if the data or
  // viewBox changes — a hard-coded dasharray would not.
  const lineRef = useDrawPath<SVGPolylineElement>();
  const width = 460;
  const height = 46;
  const pad = 8;
  const step = (width - pad * 2) / (WAVE.length - 1);
  const max = Math.max(...WAVE);
  const baseline = height - 4;
  const y = (n: number) => baseline - (n / max) * (height - 14);
  const points = WAVE.map((n, i) => `${pad + i * step},${y(n)}`).join(' ');
  const mean = WAVE.reduce((sum, n) => sum + n, 0) / WAVE.length;

  return (
    <svg
      className="hero__wave"
      viewBox={`0 0 ${width} ${height}`}
      aria-hidden="true"
      focusable="false"
    >
      {/* the essay's mean, the thing the variation is measured against */}
      <line
        className="hero__wave-mean"
        x1={pad}
        x2={width - pad}
        y1={y(mean)}
        y2={y(mean)}
      />
      {WAVE.map((n, i) => (
        <motion.rect
          key={i}
          className="hero__wave-bar"
          x={pad + i * step - 1.5}
          width={3}
          rx={1.5}
          initial={{ y: baseline, height: 0, opacity: 0 }}
          animate={{ y: y(n), height: baseline - y(n), opacity: 0.32 }}
          transition={{ duration: 0.5, delay: 0.35 + i * 0.035, ease: EASE }}
        />
      ))}
      <polyline ref={lineRef} className="hero__wave-line" points={points} />
    </svg>
  );
}

interface Props {
  value: string;
  onChange: (text: string) => void;
  onAnalyse: () => void;
  onClear: () => void;
  busy: boolean;
  disabled?: boolean;
  minChars: number;
  maxChars: number;
  /** Soft floors from the server: below these the analysis runs but the detector
   *  reports "insufficient evidence". */
  minSentences: number;
  minWords: number;
  savesEssays: boolean;
  saveOptOut: boolean;
  onSaveOptOutChange: (value: boolean) => void;
}

export function EssayEditor({
  value,
  onChange,
  onAnalyse,
  onClear,
  busy,
  disabled = false,
  minChars,
  maxChars,
  minSentences,
  minWords,
  savesEssays,
  saveOptOut,
  onSaveOptOutChange,
}: Props) {
  const [showExamples, setShowExamples] = useState(false);

  const counts = useMemo(
    () => ({
      words: countWords(value),
      characters: value.length,
      paragraphs: countParagraphs(value),
      sentences: countSentences(value),
    }),
    [value],
  );

  const tooShort = counts.characters > 0 && counts.characters < minChars;
  const tooLong = counts.characters > maxChars;
  const canAnalyse = !busy && !disabled && counts.characters >= minChars && !tooLong;

  // Clears the hard limit but not the soft floors: the request will succeed and
  // come back "insufficient evidence". Saying so now is better than letting the
  // user wait for a non-answer.
  const belowVerdictFloor =
    !tooShort &&
    !tooLong &&
    counts.characters > 0 &&
    (counts.sentences < minSentences || counts.words < minWords);

  return (
    <section className="composer">
      <div className="hero">
        <span className="hero__eyebrow">
          <span className="hero__eyebrow-dot" aria-hidden="true" />
          Evidence, not a percentage
        </span>
        <h1>What does this essay measure like?</h1>
        <p className="hero__lede">
          Paste an admissions essay. Every passage that gets flagged comes with the
          measurements behind it.
        </p>
        <HeroWave />
      </div>

      <div className="editor">
        <label className="editor__label" htmlFor="essay-input">
          Paste your admissions essay
        </label>
        <textarea
          id="essay-input"
          className="editor__textarea"
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={
            'Paste the full essay here.\n\nKeep the paragraph breaks — paragraph structure is one of the things the detector measures.'
          }
          spellCheck={false}
          aria-describedby="essay-counts"
          disabled={busy}
        />
        <div className="editor__bar">
          <div
            className={`counts ${tooLong ? 'counts--warn' : ''}`}
            id="essay-counts"
            aria-live="polite"
          >
            <span>
              <strong>{counts.words.toLocaleString()}</strong> words
            </span>
            <span>
              <strong>{counts.characters.toLocaleString()}</strong> characters
            </span>
            <span>
              <strong>{counts.paragraphs}</strong> paragraphs
            </span>
            <span>
              <strong>{counts.sentences}</strong> sentences
            </span>
          </div>
          <div className="spacer" />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowExamples((open) => !open)}
            aria-expanded={showExamples}
          >
            {showExamples ? 'Hide examples' : 'Load an example'}
          </Button>
          <Button variant="ghost" size="sm" onClick={onClear} disabled={busy || !value}>
            Clear
          </Button>
        </div>

        <AnimatePresence initial={false}>
        {showExamples && (
          <motion.div
            className="examples"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.26, ease: EASE }}
            // Framer animates height to 0; without clipping the content spills
            // out of the collapsing box for the length of the transition.
            style={{ overflow: 'hidden' }}
          >
            <p className="tiny muted mb-1 mt-0">
              Each example says how it was produced, not what the detector will conclude —
              that is whatever it measures.
            </p>
            {EXAMPLE_ESSAYS.map((example) => (
              <button
                key={example.id}
                type="button"
                className="example-btn"
                onClick={() => {
                  onChange(example.text);
                  setShowExamples(false);
                }}
              >
                <strong>{example.name}</strong>
                <span className="tiny muted block font-normal">
                  {example.provenance}
                </span>
              </button>
            ))}
          </motion.div>
        )}
        </AnimatePresence>
      </div>

      <AnimatePresence initial={false} mode="wait">
      {tooShort && (
        <motion.p
          key="short"
          className="editor__notice editor__notice--warn"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.2, ease: EASE }}
        >
          <strong>{(minChars - counts.characters).toLocaleString()} more characters needed.</strong>{' '}
          The server accepts essays from {minChars.toLocaleString()} characters.
        </motion.p>
      )}
      {tooLong && (
        <motion.p
          key="long"
          className="editor__notice editor__notice--error"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.2, ease: EASE }}
        >
          <strong>
            {(counts.characters - maxChars).toLocaleString()} characters over the limit.
          </strong>{' '}
          The maximum is {maxChars.toLocaleString()} characters.
        </motion.p>
      )}
      {belowVerdictFloor && (
        <motion.p
          key="floor"
          className="editor__notice editor__notice--info"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.2, ease: EASE }}
        >
          <strong>This will come back “insufficient evidence”.</strong> The detector needs at
          least {minSentences} sentences and {minWords} words before it will name a class —
          this has {counts.sentences}{' '}
          {counts.sentences === 1 ? 'sentence' : 'sentences'} and {counts.words}{' '}
          {counts.words === 1 ? 'word' : 'words'}. You can still run it; the measurements
          will be shown without a verdict.
        </motion.p>
      )}
      </AnimatePresence>

      <div className="composer__actions">
        <Button variant="primary" onClick={onAnalyse} disabled={!canAnalyse}>
          {busy ? 'Analysing…' : 'Analyse essay'}
        </Button>
      </div>

      <p className="privacy-line">
        {savesEssays ? (
          <>
            This server is configured to <strong>store submitted essays</strong>.{' '}
            <label className="cursor-pointer">
              <input
                type="checkbox"
                checked={saveOptOut}
                onChange={(event) => onSaveOptOutChange(event.target.checked)}
                className="mr-1.5 accent-accent align-middle"
              />
              Do not save my essay
            </label>
          </>
        ) : (
          <>
            Your essay is <strong>not stored</strong>. Only derived measurements, character
            offsets and scores are kept — never the text, and never in logs.
          </>
        )}
      </p>
    </section>
  );
}
