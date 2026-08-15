import { fixed, percent } from '@/lib/format';
import type { AnalysisSummary } from '@/types/api';

/**
 * The raw statistics table. Every value here comes straight from the analysis
 * response — nothing is recomputed in the browser, so the table can never
 * disagree with the evidence panel.
 */
export function SummaryStats({ summary }: { summary: AnalysisSummary }) {
  const s = summary.statistics;

  const groups: Array<{ title: string; rows: Array<[string, string, string]> }> = [
    {
      title: 'Language-model predictability',
      rows: [
        ['Perplexity (whole essay)', fixed(s.perplexity, 1), 'exp(mean negative log-likelihood) under the reference model'],
        ['Mean token log probability', fixed(s.mean_token_logprob, 3), 'higher is more predictable'],
        ['Tokens that were the top choice', percent(s.fraction_top1_tokens, 1), 'model ranked the actual word first'],
        ['Mean predictive entropy', fixed(s.mean_token_entropy, 2), 'bits of uncertainty per position'],
        ['Tokens scored', summary.lm_tokens_scored.toLocaleString(), `in ${summary.lm_windows} sliding window(s)`],
      ],
    },
    {
      title: 'Sentence rhythm',
      rows: [
        ['Mean words per sentence', fixed(s.mean_words_per_sentence, 1), ''],
        ['Sentence-length std dev', fixed(s.sentence_length_std, 1), 'spread around the mean'],
        ['Coefficient of variation', fixed(s.sentence_length_cv, 3), 'scale-free spread; lower is more uniform'],
        ['Burstiness index', fixed(s.burstiness_index, 3), '−1 perfectly regular, 0 Poisson-like, positive bursty'],
      ],
    },
    {
      title: 'Vocabulary and structure',
      rows: [
        ['Type-token ratio', fixed(s.type_token_ratio, 3), 'unique words / total words'],
        ['Root type-token ratio', fixed(s.root_type_token_ratio, 2), 'length-corrected diversity'],
        ['Flesch reading ease', fixed(s.flesch_reading_ease, 1), 'higher is easier'],
        ['Transition-word rate', fixed(s.transition_word_rate, 2), 'per 100 words'],
        ['Contraction rate', fixed(s.contraction_rate, 2), 'per 100 words'],
      ],
    },
    {
      title: 'Repetition and style shift',
      rows: [
        ['Trigram repeat ratio', percent(s.trigram_repeat_ratio, 2), 'repeated 3-word sequences'],
        ['Syntactic template repeat ratio', percent(s.pos_template_repeat_ratio, 2), 'repeated part-of-speech 4-grams'],
        ['Largest style shift', fixed(s.max_style_shift, 3), 'furthest passage from the essay baseline'],
        ['Style change points', String(s.style_changepoints), 'positions where the running mean shifts'],
      ],
    },
  ];

  return (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(17rem, 1fr))',
        gap: '1.4rem',
      }}
    >
      {groups.map((group) => (
        <div key={group.title}>
          <p className="subhead">{group.title}</p>
          <dl className="measure-list">
            {group.rows.map(([name, value, note]) => (
              <div className="measure" key={name}>
                <dt className="measure__name">{name}</dt>
                <dd className="measure__value">{value}</dd>
                {note && <dd className="measure__ref">{note}</dd>}
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  );
}
