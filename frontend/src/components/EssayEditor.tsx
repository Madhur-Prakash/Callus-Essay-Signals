import { AnimatePresence, motion } from 'framer-motion';
import { useMemo, useState } from 'react';

import { Button, Sparkline, SplitHeading } from '@/components/ui';
import { EASE } from '@/hooks/useMotion';
import { EXAMPLE_ESSAYS } from '@/lib/exampleEssays';
import { countParagraphs, countSentences, countWords } from '@/lib/format';

/**
 * The hero motif: a sentence-length trace over its own bars.
 *
 * The one decorative element in the interface, and it is drawn from the thing the
 * product actually measures — uneven sentence lengths around a mean — rather than
 * an abstract flourish. The numbers are fixed and illustrative; they are never
 * presented as data about anyone's essay.
 */
const WAVE = [14, 31, 8, 22, 11, 38, 17, 6, 26, 13, 29, 9, 20, 34, 12];

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
        <motion.span
          className="eyebrow"
          initial={{ opacity: 0, y: -6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, ease: EASE }}
        >
          <span className="eyebrow__dot" aria-hidden="true" />
          Evidence, not a percentage
        </motion.span>

        {/* The one heading in the app that gets the line reveal. */}
        <SplitHeading>What does this essay measure like?</SplitHeading>

        <motion.p
          className="hero__lede"
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.35, ease: EASE }}
        >
          Paste an admissions essay. Every passage that gets flagged comes with the
          measurements behind it.
        </motion.p>

        <div className="hero__wave">
          <Sparkline values={WAVE} />
        </div>
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
        {/* The page's one magnetic control — the effect means "this is the
            target", which stops being true the moment everything has it. */}
        <Button
          variant="primary"
          magnetic
          loading={busy}
          disabled={!canAnalyse}
          onClick={onAnalyse}
        >
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
