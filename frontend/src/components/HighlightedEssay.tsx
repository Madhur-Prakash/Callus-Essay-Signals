import { useMemo } from 'react';

import { PARAGRAPH_LABELS, percent, scoreColour } from '@/lib/format';
import type { AnalysisResponse, ParagraphResult, SentenceResult } from '@/types/api';

interface Props {
  result: AnalysisResponse;
  essayText: string;
  selectedSentenceId: number | null;
  onSelectSentence: (sentenceId: number | null) => void;
}

const PARA_MODIFIER: Record<string, string> = {
  likely_ai_assisted: 'flagged',
  contains_flagged_sentence: 'possible',
  uncertain: 'uncertain',
  likely_human: 'human',
  unavailable: 'uncertain',
};

/**
 * Renders the essay with sentence-level marks.
 *
 * Sentence text comes from the local essay whenever possible, sliced with the
 * `start`/`end` offsets the backend returned. That matters for the privacy mode:
 * when the server does not store essay text it still returns offsets, so a
 * reloaded analysis can be re-rendered against the copy the user still has in the
 * editor without the server ever having kept it.
 */
export function HighlightedEssay({
  result,
  essayText,
  selectedSentenceId,
  onSelectSentence,
}: Props) {
  const sentencesByParagraph = useMemo(() => {
    const map = new Map<number, SentenceResult[]>();
    for (const sentence of result.sentences) {
      const list = map.get(sentence.paragraph_id) ?? [];
      list.push(sentence);
      map.set(sentence.paragraph_id, list);
    }
    for (const list of map.values()) {
      list.sort((a, b) => a.start - b.start);
    }
    return map;
  }, [result.sentences]);

  const sliceText = (sentence: SentenceResult): string => {
    if (sentence.text) return sentence.text;
    const slice = essayText.slice(sentence.start, sentence.end);
    return slice || '[text not stored]';
  };

  return (
    <div>
      <div className="legend" style={{ marginBottom: '1rem' }}>
        <LegendItem colour="var(--likely)" soft="var(--likely-soft)" label="Likely AI-assisted" />
        <LegendItem
          colour="var(--possible)"
          soft="var(--possible-soft)"
          label="Possibly AI-assisted"
        />
        <LegendItem colour="var(--uncertain)" soft="transparent" label="Uncertain" />
        <span className="legend__item muted">Unmarked — measured as ordinary</span>
      </div>

      <div className="essay">
        {result.paragraphs.map((paragraph) => {
          const sentences = sentencesByParagraph.get(paragraph.paragraph_id) ?? [];
          const modifier = PARA_MODIFIER[paragraph.classification] ?? 'uncertain';
          return (
            <p
              key={paragraph.paragraph_id}
              className={`essay__paragraph essay__paragraph--${modifier}`}
            >
              <span className="essay__meta">
                <span>Paragraph {paragraph.paragraph_id + 1}</span>
                <span aria-hidden="true">·</span>
                <span>{PARAGRAPH_LABELS[paragraph.classification]}</span>
                {paragraph.human_likeness !== null && (
                  <>
                    <span aria-hidden="true">·</span>
                    <span className="mono">
                      human-likeness {paragraph.human_likeness.toFixed(2)}
                    </span>
                  </>
                )}
              </span>
              {sentences.map((sentence) => {
                const isSelected = selectedSentenceId === sentence.sentence_id;
                const classes = [
                  'sentence',
                  `sentence--${sentence.classification}`,
                  isSelected ? 'sentence--selected' : '',
                ]
                  .filter(Boolean)
                  .join(' ');
                const toggle = () =>
                  onSelectSentence(isSelected ? null : sentence.sentence_id);
                // A <span role="button"> rather than a <button>: Chrome renders
                // form controls as atomic inline-level boxes, so real <button>
                // sentences stack as blocks instead of flowing as prose. A span
                // participates in inline layout and still exposes the button role.
                return (
                  <span
                    key={sentence.sentence_id}
                    role="button"
                    tabIndex={0}
                    className={classes}
                    onClick={toggle}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault();
                        toggle();
                      }
                    }}
                    onMouseEnter={() => {
                      if (selectedSentenceId === null) onSelectSentence(sentence.sentence_id);
                    }}
                    aria-pressed={isSelected}
                    title={
                      sentence.score === null
                        ? 'Not scored'
                        : `Machine-likeness ${percent(sentence.score, 1)} — click for evidence`
                    }
                  >
                    {sliceText(sentence)}
                  </span>
                );
              })}
            </p>
          );
        })}
      </div>
    </div>
  );
}

function LegendItem({
  colour,
  soft,
  label,
}: {
  colour: string;
  soft: string;
  label: string;
}) {
  return (
    <span className="legend__item">
      <span
        className="legend__swatch"
        style={{ background: soft, boxShadow: `inset 0 -2px 0 ${colour}` }}
        aria-hidden="true"
      />
      {label}
    </span>
  );
}

export function ParagraphBreakdown({
  paragraphs,
  sentences,
  onSelectSentence,
}: {
  paragraphs: ParagraphResult[];
  sentences: SentenceResult[];
  onSelectSentence: (id: number) => void;
}) {
  const scoreById = new Map(sentences.map((s) => [s.sentence_id, s.score]));

  return (
    <div className="para-list">
      {paragraphs.map((paragraph) => (
        <button
          key={paragraph.paragraph_id}
          type="button"
          className="para-row"
          onClick={() => {
            const first =
              paragraph.flagged_sentence_ids[0] ??
              paragraph.uncertain_sentence_ids[0] ??
              paragraph.sentence_ids[0];
            if (first !== undefined) onSelectSentence(first);
          }}
        >
          <span className="para-row__id">¶{paragraph.paragraph_id + 1}</span>
          <span className="para-row__bar">
            <span
              className="para-row__fill"
              style={{
                width: `${Math.max(2, (paragraph.score ?? 0) * 100)}%`,
                background: scoreColour(paragraph.score),
              }}
            />
          </span>
          <span className="para-row__meta">
            {paragraph.human_likeness !== null
              ? `human ${paragraph.human_likeness.toFixed(2)}`
              : '—'}
            {paragraph.flagged_sentence_ids.length > 0 &&
              ` · ${paragraph.flagged_sentence_ids.length} flagged`}
          </span>
          <span className="para-row__sentences" aria-hidden="true">
            {paragraph.sentence_ids.map((id) => (
              <span
                key={id}
                className="para-row__tick"
                style={{ background: scoreColour(scoreById.get(id) ?? null) }}
              />
            ))}
          </span>
        </button>
      ))}
    </div>
  );
}
