import { Reveal, Section } from '@/components/ui';
import { cn } from '@/lib/cn';
import { GROUP_LABELS } from '@/lib/format';
import type { ModelInfoResponse } from '@/types/api';

interface Props {
  modelInfo: ModelInfoResponse | null;
}

const FALLBACK_PIPELINE = [
  'Normalise the text and split it into paragraphs and sentences',
  'Measure stylometry: sentence and word statistics, punctuation, vocabulary',
  'Parse part-of-speech and dependency structure',
  'Score every token with a small local language model',
  'Measure sentence rhythm and repetition',
  'Compare each sentence against the author’s own baseline',
  'Compare the text against the human and machine reference corpora',
  'Feed the document feature vector into our trained classifier',
  'Calibrate the probability and map it to a confidence band',
  'Generate evidence deterministically from the measured values',
];

export function HowItWorksPage({ modelInfo }: Props) {
  const methodology = modelInfo?.methodology;
  const pipeline = methodology?.pipeline ?? FALLBACK_PIPELINE;

  return (
    <Reveal className="stack stack--lg mx-auto max-w-[52rem]">
      <header>
        <h1>How this works</h1>
        <p className="muted text-base">
          {methodology?.summary ??
            'This detector does not ask another AI whether an essay is AI-written. It measures properties of the writing and feeds those measurements to a classifier we trained ourselves.'}
        </p>
      </header>

      <Section title="The thing it is not">
        <div className="grid gap-5 grid-cols-[repeat(auto-fit,minmax(15rem,1fr))]">
          <div>
            <p className="subhead text-likely">Not this</p>
            <Flow tone="likely">
              {`Essay
  ↓
Chat model
  ↓
"Is this AI?"
  ↓
Verdict`}
            </Flow>
          </div>
          <div>
            <p className="subhead text-human">This</p>
            <Flow tone="human">
              {`Essay
  ↓
Tokeniser
  ↓
Local language model
  ↓
Token probabilities
  ↓
Feature extraction
  ↓
Our classifier
  ↓
Calibration
  ↓
Evidence engine
  ↓
Result`}
            </Flow>
          </div>
        </div>
        <hr className="divider" />
        <p className="mb-0">
          {methodology?.what_the_language_model_does ??
            'The language model provides one number per token: how probable that token was given the preceding text. It is a measuring instrument, like a thermometer. It is never asked to judge authorship.'}
        </p>
      </Section>

      <Section title="The pipeline">
        <ol className="m-0 grid list-decimal gap-1.5 pl-5 text-xs">
          {pipeline.map((step, index) => (
            <li key={index}>{step}</li>
          ))}
        </ol>
      </Section>

      <Section
        title="What gets measured"
        note="Grouped the way the ablation study groups them, so each block's contribution can be tested independently."
      >
        <div className="grid gap-4 grid-cols-[repeat(auto-fit,minmax(14rem,1fr))]">
          {Object.entries(GROUP_LABELS).map(([key, label]) => (
            <div key={key}>
              <p className="mb-0.5 mt-0 text-xs font-semibold">{label}</p>
              <p className="tiny muted m-0">{GROUP_DESCRIPTIONS[key]}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section title="What makes the decision">
        <p>
          {methodology?.what_makes_the_decision ??
            'A scikit-learn classifier trained on our labelled corpus, with document-level grouped splits and probability calibration.'}
        </p>
        {modelInfo?.document_model && (
          <table className="data mt-2.5">
              <tbody>
                <tr>
                  <td>Classifier</td>
                  <td className="num">{modelInfo.document_model.name ?? '—'}</td>
                </tr>
                <tr>
                  <td>Document features used</td>
                  <td className="num">{modelInfo.document_model.n_features}</td>
                </tr>
                <tr>
                  <td>Calibration</td>
                  <td className="num">{modelInfo.document_model.calibration ?? 'none'}</td>
                </tr>
                <tr>
                  <td>Instrument model</td>
                  <td className="num">
                    {String(modelInfo.language_model?.name ?? '—')}
                  </td>
                </tr>
                <tr>
                  <td>Model version</td>
                  <td className="num">{modelInfo.model_version ?? '—'}</td>
                </tr>
                <tr>
                  <td>Trained at</td>
                  <td className="num">{modelInfo.trained_at ?? '—'}</td>
                </tr>
                <tr>
                  <td>Data regime</td>
                  <td className="num">{modelInfo.data_regime ?? '—'}</td>
                </tr>
              </tbody>
          </table>
        )}
      </Section>

      <Section title="Limitations">
        <ul className="statements">
          {(methodology?.limitations ?? DEFAULT_LIMITATIONS).map((limitation, index) => (
            <li key={index}>{limitation}</li>
          ))}
        </ul>
      </Section>

      <section className="card border-warning-line bg-warning-bg">
        <div className="card__body">
          <h3 className="mb-1.5">Detection is not proof of authorship</h3>
          <p className="mb-0">
            These measurements describe text. Text does not carry a signature. A flag means
            "this passage has statistical properties in common with the machine-written examples
            in our evaluation data" — which is a reason to read more carefully and, if it
            matters, to talk to the writer. It is never a reason to accuse anyone, and it must
            never be the sole basis for a decision about a person.
          </p>
        </div>
      </section>
    </Reveal>
  );
}

/**
 * The two ASCII pipeline sketches. `whitespace-pre` rather than `whitespace-pre-wrap`
 * on purpose — the arrows only line up if the diagram is allowed to overflow and
 * scroll rather than reflow, which is why it also gets its own scroll container.
 */
function Flow({ tone, children }: { tone: 'likely' | 'human'; children: string }) {
  return (
    <pre
      className={cn(
        'mono m-0 overflow-x-auto whitespace-pre rounded-control p-3 text-[0.76rem]',
        'leading-relaxed text-ink-soft',
        tone === 'likely' ? 'bg-likely-soft' : 'bg-human-soft',
      )}
    >
      {children}
    </pre>
  );
}

const GROUP_DESCRIPTIONS: Record<string, string> = {
  lm: 'Per-token log probability, perplexity, predictive entropy, rank, and the gap to the model’s own top choice.',
  stylometric:
    'Sentence and word length, punctuation habits, contractions, vocabulary richness, hedges, connectives, nominalisations.',
  syntactic:
    'Part-of-speech and dependency distributions, clause counts, parse depth, passive constructions, sentence openers.',
  burstiness:
    'Variation in sentence length: standard deviation, coefficient of variation, adjacent differences, entropy, direction changes.',
  repetition:
    'Repeated word n-grams, repeated part-of-speech templates, repeated sentence openers, inter-sentence similarity.',
  structural:
    'Paragraph shape and balance, words per sentence, readability scores, document length.',
  style_shift:
    'How far each sentence sits from the author’s own baseline, plus change-point detection across the essay.',
  corpus:
    'Similarity to the human and machine reference corpora using character n-grams, POS n-grams and function-word profiles — deliberately topic-free.',
};

const DEFAULT_LIMITATIONS = [
  'Detection is probabilistic. A flag is evidence for review, never proof of authorship.',
  'Low perplexity alone is not evidence of machine authorship — clear, conventional human prose also scores low.',
  'AI detectors are known to over-flag writing by people who learned English as an additional language.',
  'A lightly edited human essay is mostly human text, so the AI-polished class is inherently the hardest to identify.',
];
