import type { Route } from '@/App';
import { Banner } from '@/components/Banner';
import { Button, Reveal, Section } from '@/components/ui';
import type { ModelInfoResponse } from '@/types/api';

interface Props {
  modelInfo: ModelInfoResponse | null;
  onNavigate: (route: Route) => void;
}

/**
 * Limitations, given its own page rather than a footnote on "How it works".
 *
 * This is the page most likely to change someone's mind about acting on a
 * verdict, so it should be linkable and reachable in one click — not buried at
 * the bottom of a methodology write-up.
 */

const RANKED = [
  {
    severity: 'severe',
    title: 'The training corpus is synthetic by default',
    body: 'The human class is 36 hand-authored seed essays; the machine classes come from an offline template generator and a rule-based editor. Every metric measures how separable those three generators are — not real-world accuracy. Supply a GROQ_API_KEY and add real essays to data/raw/human/ before quoting any number.',
  },
  {
    severity: 'severe',
    title: 'It over-flags human writing',
    body: 'On the held-out split, human recall sits well below the machine classes: roughly a quarter of genuine human essays are classified as machine-polished. Overall accuracy hides this because it is carried by the machine classes. This alone disqualifies the tool from any high-stakes use.',
  },
  {
    severity: 'severe',
    title: 'Fairness for second-language writers is not established',
    body: 'Published work consistently finds AI detectors over-flag writing by people who learned English as an additional language. The L2 subset here is a simulated register across four seed groups — enough to detect a large disparity, nowhere near enough to rule one out. Overlapping confidence intervals mean "cannot tell", never "fair".',
  },
  {
    severity: 'high',
    title: 'The AI-polished class is inherently hard',
    body: 'A grammar-only pass leaves nearly every measurement intact, so those essays are usually called human — correctly, in that most of the text is human, and unhelpfully, in that the edit did happen. Where an edit is heavy enough to detect, it often looks identical to fully generated text.',
  },
  {
    severity: 'high',
    title: 'Sentence-level scores look better than they are',
    body: 'Validation ROC-AUC for the sentence model is about 1.0 on the bootstrap corpus. That is a property of a finite template bank, not a capability. Treat sentence highlighting as "where to look", never as a per-sentence verdict.',
  },
  {
    severity: 'moderate',
    title: 'Small effective sample size',
    body: 'Grouped splitting means the effective number of independent human documents is 36. Confidence intervals are wide, and differences of a few points between feature sets sit inside fold-to-fold noise.',
  },
  {
    severity: 'moderate',
    title: 'The instrument model is small and old',
    body: 'distilgpt2 is an 82M-parameter model from 2019. Text whose vocabulary or topic falls outside its training distribution looks "surprising" regardless of who wrote it, which penalises unusual-but-human writing.',
  },
  {
    severity: 'moderate',
    title: 'Trivially evadable by anyone who reads this page',
    body: 'Every feature is public and imitable. Vary your sentence lengths, use contractions, avoid "moreover", add a typo. There is no adversarial robustness here and none is claimed.',
  },
  {
    severity: 'moderate',
    title: 'A style shift has legitimate causes',
    body: 'Quoting a source, moving from narrative to reflection, or writing a stronger conclusion all produce genuine style shifts in human writing. A shift means the register changed, not that someone else wrote it.',
  },
  {
    severity: 'moderate',
    title: 'English only',
    body: 'The lexicons, the spaCy model and the instrument model are all English. Behaviour on other languages is undefined, not merely degraded.',
  },
];

const SEVERITY_COLOUR: Record<string, string> = {
  severe: 'var(--likely)',
  high: 'var(--possible)',
  moderate: 'var(--uncertain)',
};

export function LimitationsPage({ modelInfo, onNavigate }: Props) {
  return (
    <Reveal className="stack stack--lg mx-auto max-w-[54rem]">
      <header>
        <h1>Limitations</h1>
        <p className="muted text-lg">
          Read this before acting on anything this system outputs.
        </p>
      </header>

      <Banner tone="danger" title="Detection is not proof of authorship">
        This system measures properties of text. Text does not carry a signature. The
        strongest honest claim available is that a passage “has statistical properties in
        common with the machine-written examples in our evaluation data”. It cannot
        distinguish a machine from a human who writes formally, evenly and without
        contractions, because in the text alone there is no difference between those two
        things.
      </Banner>

      <Section title="Ranked limitations" note="Most severe first.">
        <ol className="limitation-list">
          {RANKED.map((item, index) => (
            <li className="limitation" key={item.title}>
              <span className="limitation__rank">{index + 1}</span>
              <div>
                <p className="limitation__title">
                  {item.title}
                  <span
                    className="limitation__severity"
                    style={{ color: SEVERITY_COLOUR[item.severity] }}
                  >
                    {item.severity}
                  </span>
                </p>
                <p className="limitation__body">{item.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </Section>

      <div className="two-col">
        <Section title="Do not">
          <ul className="statements">
            <li>Use this as evidence in an academic-integrity process.</li>
            <li>Reject an applicant based on it, in whole or in part.</li>
            <li>Present its output to a student as a finding about their honesty.</li>
            <li>
              Quote its accuracy as a real-world figure while the data regime is
              <code> bootstrap</code>.
            </li>
          </ul>
        </Section>

        <Section title="Reasonable uses">
          <ul className="statements">
            <li>Deciding which essays in a large pile to read more attentively.</li>
            <li>Prompting a conversation with a writer about their process.</li>
            <li>Research into which textual features carry signal, and which do not.</li>
            <li>Demonstrating to a non-technical audience why this problem is hard.</li>
          </ul>
        </Section>
      </div>

      {modelInfo?.methodology?.limitations?.length ? (
        <Section
          title="Reported by the running model"
          note={
            <>
              Served from <code>GET /api/v1/model/info</code>, so it reflects this deployment
              rather than this page.
            </>
          }
        >
          <ul className="statements">
            {modelInfo.methodology.limitations.map((limitation, index) => (
              <li key={index}>{limitation}</li>
            ))}
          </ul>
        </Section>
      ) : null}

      <div className="row row--wrap justify-center">
        <Button onClick={() => onNavigate('research')}>See the measured evaluation</Button>
        <Button variant="ghost" onClick={() => onNavigate('how')}>
          How the detector works
        </Button>
      </div>
    </Reveal>
  );
}
