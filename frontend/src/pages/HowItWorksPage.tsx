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
    <div className="stack stack--lg" style={{ maxWidth: '52rem', margin: '0 auto' }}>
      <header>
        <h1>How this works</h1>
        <p className="muted" style={{ fontSize: '1rem' }}>
          {methodology?.summary ??
            'This detector does not ask another AI whether an essay is AI-written. It measures properties of the writing and feeds those measurements to a classifier we trained ourselves.'}
        </p>
      </header>

      <section className="card">
        <div className="card__head">
          <p className="card__title">The thing it is not</p>
        </div>
        <div className="card__body">
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(15rem, 1fr))',
              gap: '1.2rem',
            }}
          >
            <div>
              <p className="subhead" style={{ color: 'var(--likely)' }}>
                Not this
              </p>
              <pre
                className="mono"
                style={{
                  margin: 0,
                  fontSize: '0.76rem',
                  lineHeight: 1.7,
                  color: 'var(--ink-soft)',
                  background: 'var(--likely-soft)',
                  padding: '0.8rem',
                  borderRadius: 'var(--radius-sm)',
                  whiteSpace: 'pre',
                  overflowX: 'auto',
                }}
              >
{`Essay
  ↓
Chat model
  ↓
"Is this AI?"
  ↓
Verdict`}
              </pre>
            </div>
            <div>
              <p className="subhead" style={{ color: 'var(--human)' }}>
                This
              </p>
              <pre
                className="mono"
                style={{
                  margin: 0,
                  fontSize: '0.76rem',
                  lineHeight: 1.7,
                  color: 'var(--ink-soft)',
                  background: 'var(--human-soft)',
                  padding: '0.8rem',
                  borderRadius: 'var(--radius-sm)',
                  whiteSpace: 'pre',
                  overflowX: 'auto',
                }}
              >
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
              </pre>
            </div>
          </div>
          <hr className="divider" />
          <p style={{ marginBottom: 0 }}>
            {methodology?.what_the_language_model_does ??
              'The language model provides one number per token: how probable that token was given the preceding text. It is a measuring instrument, like a thermometer. It is never asked to judge authorship.'}
          </p>
        </div>
      </section>

      <section className="card">
        <div className="card__head">
          <p className="card__title">The pipeline</p>
        </div>
        <div className="card__body">
          <ol style={{ margin: 0, paddingLeft: '1.3rem', display: 'grid', gap: '0.4rem' }}>
            {pipeline.map((step, index) => (
              <li key={index} style={{ fontSize: '0.9rem' }}>
                {step}
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="card">
        <div className="card__head">
          <div>
            <p className="card__title">What gets measured</p>
            <p className="section-note">
              Grouped the way the ablation study groups them, so each block's contribution can
              be tested independently.
            </p>
          </div>
        </div>
        <div className="card__body">
          <div
            style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(14rem, 1fr))',
              gap: '1rem',
            }}
          >
            {Object.entries(GROUP_LABELS).map(([key, label]) => (
              <div key={key}>
                <p style={{ margin: '0 0 0.15rem', fontWeight: 600, fontSize: '0.88rem' }}>
                  {label}
                </p>
                <p className="tiny muted" style={{ margin: 0 }}>
                  {GROUP_DESCRIPTIONS[key]}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card__head">
          <p className="card__title">What makes the decision</p>
        </div>
        <div className="card__body">
          <p>
            {methodology?.what_makes_the_decision ??
              'A scikit-learn classifier trained on our labelled corpus, with document-level grouped splits and probability calibration.'}
          </p>
          {modelInfo?.document_model && (
            <table className="data" style={{ marginTop: '0.6rem' }}>
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
        </div>
      </section>

      <section className="card">
        <div className="card__head">
          <p className="card__title">Limitations</p>
        </div>
        <div className="card__body">
          <ul className="statements">
            {(methodology?.limitations ?? DEFAULT_LIMITATIONS).map((limitation, index) => (
              <li key={index}>{limitation}</li>
            ))}
          </ul>
        </div>
      </section>

      <section
        className="card"
        style={{ borderColor: 'var(--warning-border)', background: 'var(--warning-bg)' }}
      >
        <div className="card__body">
          <h3 style={{ marginBottom: '0.4rem' }}>Detection is not proof of authorship</h3>
          <p style={{ marginBottom: 0 }}>
            These measurements describe text. Text does not carry a signature. A flag means
            "this passage has statistical properties in common with the machine-written examples
            in our evaluation data" — which is a reason to read more carefully and, if it
            matters, to talk to the writer. It is never a reason to accuse anyone, and it must
            never be the sole basis for a decision about a person.
          </p>
        </div>
      </section>
    </div>
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
