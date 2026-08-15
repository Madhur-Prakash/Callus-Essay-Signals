import type { AnalysisResponse } from '@/types/api';

interface Props {
  repetition: AnalysisResponse['repetition'];
  onSelectSentence: (sentenceId: number) => void;
}

export function RepetitionPanel({ repetition, onSelectSentence }: Props) {
  const { repeated_phrases: phrases, repeated_syntactic_templates: templates } = repetition;

  if (phrases.length === 0 && templates.length === 0) {
    return (
      <p className="small muted" style={{ margin: 0 }}>
        No repeated phrases or syntactic templates were found above the reporting threshold.
      </p>
    );
  }

  return (
    <div style={{ display: 'grid', gap: '1.2rem' }}>
      {phrases.length > 0 && (
        <div>
          <p className="subhead">Repeated phrases</p>
          <div className="phrase-list">
            {phrases.map((phrase) => (
              <div className="phrase" key={phrase.phrase}>
                <span className="phrase__text">{phrase.phrase}</span>
                <span className="phrase__count">
                  ×{phrase.count} · {phrase.length}-gram
                </span>
                <span className="spacer" />
                {phrase.sentence_indices.slice(0, 4).map((id) => (
                  <button
                    key={id}
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => onSelectSentence(id)}
                    title={`Jump to sentence ${id + 1}`}
                  >
                    s{id + 1}
                  </button>
                ))}
              </div>
            ))}
          </div>
          <p className="tiny muted" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
            Repeated wording is ambiguous on its own — it appears in machine text drawing on a
            phrase bank and in human drafts written quickly.
          </p>
        </div>
      )}

      {templates.length > 0 && (
        <div>
          <p className="subhead">Repeated syntactic templates</p>
          <div className="phrase-list">
            {templates.map((template) => (
              <div className="phrase" key={template.template}>
                <span className="phrase__text">{template.template}</span>
                <span className="phrase__count">
                  in {template.sentence_count} sentences
                </span>
                <span className="spacer" />
                {template.sentence_indices.slice(0, 4).map((id) => (
                  <button
                    key={id}
                    type="button"
                    className="btn btn--ghost btn--sm"
                    onClick={() => onSelectSentence(id)}
                  >
                    s{id + 1}
                  </button>
                ))}
              </div>
            ))}
          </div>
          <p className="tiny muted" style={{ marginTop: '0.5rem', marginBottom: 0 }}>
            The same grammatical skeleton recurring across sentences survives paraphrasing,
            which makes it a more specific signal than repeated wording.
          </p>
        </div>
      )}
    </div>
  );
}
