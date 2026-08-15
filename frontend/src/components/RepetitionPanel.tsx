import { Button } from '@/components/ui';
import type { AnalysisResponse } from '@/types/api';

interface Props {
  repetition: AnalysisResponse['repetition'];
  onSelectSentence: (sentenceId: number) => void;
}

export function RepetitionPanel({ repetition, onSelectSentence }: Props) {
  const { repeated_phrases: phrases, repeated_syntactic_templates: templates } = repetition;

  if (phrases.length === 0 && templates.length === 0) {
    return (
      <p className="small muted m-0">
        No repeated phrases or syntactic templates were found above the reporting threshold.
      </p>
    );
  }

  return (
    <div className="grid gap-5">
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
                  <Button
                    key={id}
                    variant="ghost"
                    size="sm"
                    onClick={() => onSelectSentence(id)}
                    title={`Jump to sentence ${id + 1}`}
                  >
                    s{id + 1}
                  </Button>
                ))}
              </div>
            ))}
          </div>
          <p className="tiny muted mb-0 mt-2">
            Repeated wording is ambiguous on its own - it appears in machine text drawing on a
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
                  <Button
                    key={id}
                    variant="ghost"
                    size="sm"
                    onClick={() => onSelectSentence(id)}
                    title={`Jump to sentence ${id + 1}`}
                  >
                    s{id + 1}
                  </Button>
                ))}
              </div>
            ))}
          </div>
          <p className="tiny muted mb-0 mt-2">
            The same grammatical skeleton recurring across sentences survives paraphrasing,
            which makes it a more specific signal than repeated wording.
          </p>
        </div>
      )}
    </div>
  );
}
