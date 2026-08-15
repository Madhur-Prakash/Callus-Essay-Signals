import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useState } from 'react';
import { describe, expect, it, vi } from 'vitest';

import { EssayEditor } from '@/components/EssayEditor';
import { EvidencePanel } from '@/components/EvidencePanel';
import { HighlightedEssay } from '@/components/HighlightedEssay';
import { VerdictCard } from '@/components/VerdictCard';
import { AnalysePage } from '@/pages/AnalysePage';

import { SAMPLE_ANALYSIS, SAMPLE_HEALTH, SAMPLE_PRIVACY } from './fixtures';

function stubApi(analysisBody: unknown = SAMPLE_ANALYSIS, status = 200) {
  const fetchMock = vi.fn(async (url: string) => {
    const path = String(url);
    if (path.includes('/essays/privacy')) {
      return { ok: true, status: 200, text: async () => JSON.stringify(SAMPLE_PRIVACY) };
    }
    if (path.includes('/health')) {
      return { ok: true, status: 200, text: async () => JSON.stringify(SAMPLE_HEALTH) };
    }
    if (path.includes('/analysis')) {
      return {
        ok: status < 400,
        status,
        text: async () => JSON.stringify(analysisBody),
      };
    }
    return { ok: true, status: 200, text: async () => '{}' };
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/** A controlled wrapper, so typing actually updates the rendered value. */
function ControlledEditor(props: Partial<React.ComponentProps<typeof EssayEditor>> = {}) {
  const [value, setValue] = useState(props.value ?? '');
  return (
    <EssayEditor
      value={value}
      onChange={setValue}
      onAnalyse={vi.fn()}
      onClear={() => setValue('')}
      busy={false}
      minChars={200}
      maxChars={60000}
      minSentences={5}
      minWords={120}
      savesEssays={false}
      saveOptOut={false}
      onSaveOptOutChange={vi.fn()}
      {...props}
      {...(props.value !== undefined ? {} : {})}
    />
  );
}

describe('EssayEditor', () => {
  it('shows live word, character, paragraph and sentence counts', async () => {
    const user = userEvent.setup();
    render(<ControlledEditor />);

    await user.type(
      screen.getByLabelText(/paste your admissions essay/i),
      'One two three. Four five.',
    );

    const counts = document.querySelector('.counts') as HTMLElement;
    expect(within(counts).getByText('5')).toBeInTheDocument(); // words
    expect(within(counts).getByText('25')).toBeInTheDocument(); // characters
    expect(within(counts).getByText('2')).toBeInTheDocument(); // sentences
  });

  it('keeps Analyse disabled until the minimum length is met', () => {
    render(
      <EssayEditor
        value={'short'}
        onChange={vi.fn()}
        onAnalyse={vi.fn()}
        onClear={vi.fn()}
        busy={false}
        minChars={200}
        maxChars={60000}
        minSentences={5}
        minWords={120}
        savesEssays={false}
        saveOptOut={false}
        onSaveOptOutChange={vi.fn()}
      />,
    );
    expect(screen.getByRole('button', { name: /analyse essay/i })).toBeDisabled();
    expect(screen.getByText(/more characters needed/i)).toBeInTheDocument();
  });

  it('warns when the essay exceeds the maximum length', () => {
    render(
      <EssayEditor
        value={'x'.repeat(120)}
        onChange={vi.fn()}
        onAnalyse={vi.fn()}
        onClear={vi.fn()}
        busy={false}
        minChars={10}
        maxChars={100}
        minSentences={5}
        minWords={120}
        savesEssays={false}
        saveOptOut={false}
        onSaveOptOutChange={vi.fn()}
      />,
    );
    expect(screen.getByText(/characters over the/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /analyse essay/i })).toBeDisabled();
  });

  it('warns that a long-but-thin essay will come back inconclusive', () => {
    // Clears the 200-character floor but has one sentence, so the backend would
    // analyse it and then abstain. The UI should say so before the user waits.
    render(
      <EssayEditor
        value={'wordy '.repeat(60)}
        onChange={vi.fn()}
        onAnalyse={vi.fn()}
        onClear={vi.fn()}
        busy={false}
        minChars={200}
        maxChars={60000}
        minSentences={5}
        minWords={120}
        savesEssays={false}
        saveOptOut={false}
        onSaveOptOutChange={vi.fn()}
      />,
    );
    expect(screen.getByText(/insufficient evidence/i)).toBeInTheDocument();
    expect(screen.getByText(/at least 5 sentences and 120 words/i)).toBeInTheDocument();
    // It is a warning, not a block — the measurements are still worth seeing.
    expect(screen.getByRole('button', { name: /analyse essay/i })).toBeEnabled();
  });

  it('does not warn when the essay clears both floors', () => {
    render(
      <EssayEditor
        value={'This is a sentence. '.repeat(40)}
        onChange={vi.fn()}
        onAnalyse={vi.fn()}
        onClear={vi.fn()}
        busy={false}
        minChars={200}
        maxChars={60000}
        minSentences={5}
        minWords={120}
        savesEssays={false}
        saveOptOut={false}
        onSaveOptOutChange={vi.fn()}
      />,
    );
    expect(screen.queryByText(/insufficient evidence/i)).not.toBeInTheDocument();
  });

  it('states that the essay is not stored when persistence is off', () => {
    render(
      <EssayEditor
        value={''}
        onChange={vi.fn()}
        onAnalyse={vi.fn()}
        onClear={vi.fn()}
        busy={false}
        minChars={200}
        maxChars={60000}
        minSentences={5}
        minWords={120}
        savesEssays={false}
        saveOptOut={false}
        onSaveOptOutChange={vi.fn()}
      />,
    );
    expect(screen.getByText(/not stored/i)).toBeInTheDocument();
  });

  it('offers an opt-out when the server does store essays', () => {
    render(
      <EssayEditor
        value={''}
        onChange={vi.fn()}
        onAnalyse={vi.fn()}
        onClear={vi.fn()}
        busy={false}
        minChars={200}
        maxChars={60000}
        minSentences={5}
        minWords={120}
        savesEssays
        saveOptOut={false}
        onSaveOptOutChange={vi.fn()}
      />,
    );
    expect(screen.getByLabelText(/do not save my essay/i)).toBeInTheDocument();
  });

  it('loads an example essay', async () => {
    const user = userEvent.setup();
    render(<ControlledEditor />);
    await user.click(screen.getByRole('button', { name: /load an example/i }));
    // Anchored: a second example's provenance text also mentions "the
    // hand-written draft", so an unanchored match hits two buttons.
    await user.click(screen.getByRole('button', { name: /^Hand-written draft/ }));
    const textarea = screen.getByLabelText(
      /paste your admissions essay/i,
    ) as HTMLTextAreaElement;
    expect(textarea.value).toContain('The robot never worked');
  });
});

describe('VerdictCard', () => {
  it('renders the verdict, confidence band and per-class probabilities', () => {
    render(<VerdictCard result={SAMPLE_ANALYSIS} />);
    expect(screen.getByRole('heading', { name: /potentially ai-polished/i })).toBeInTheDocument();
    expect(screen.getByText('moderate')).toBeInTheDocument();
    expect(screen.getByText('71.0%')).toBeInTheDocument();
    expect(screen.getByText(/evidence detected in/i)).toBeInTheDocument();
  });

  it('explains an abstention instead of naming a class', () => {
    render(
      <VerdictCard
        result={{
          ...SAMPLE_ANALYSIS,
          classification: 'insufficient_evidence',
          label: 'Insufficient evidence',
          abstained: true,
          abstain_reason: 'The two leading possibilities are within 4% of each other.',
        }}
      />,
    );
    expect(screen.getByText(/why no class was named/i)).toBeInTheDocument();
    expect(screen.getByText(/within 4% of each other/i)).toBeInTheDocument();
  });
});

describe('HighlightedEssay', () => {
  it('marks flagged sentences and reports the selection', async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    render(
      <HighlightedEssay
        result={SAMPLE_ANALYSIS}
        essayText=""
        selectedSentenceId={null}
        onSelectSentence={onSelect}
      />,
    );

    const flagged = screen.getByRole('button', {
      name: /through this transformative journey/i,
    });
    expect(flagged).toHaveClass('sentence--likely_ai_assisted');
    await user.click(flagged);
    expect(onSelect).toHaveBeenCalledWith(2);
  });

  it('falls back to offsets when sentence text was not stored', () => {
    const withoutText = {
      ...SAMPLE_ANALYSIS,
      sentences: SAMPLE_ANALYSIS.sentences.map((s) => ({ ...s, text: '' })),
    };
    render(
      <HighlightedEssay
        result={withoutText}
        essayText={'The robot never worked here. It burned out twice that year.'}
        selectedSentenceId={null}
        onSelectSentence={vi.fn()}
      />,
    );
    expect(
      screen.getByRole('button', { name: /the robot never worked here/i }),
    ).toBeInTheDocument();
  });
});

describe('EvidencePanel', () => {
  it('shows meters, statements, numbers and model contributions for a sentence', () => {
    render(
      <EvidencePanel
        sentence={SAMPLE_ANALYSIS.sentences[2]!}
        documentEvidence={SAMPLE_ANALYSIS.evidence}
        hasSentenceModel
      />,
    );
    expect(screen.getByText(/why this was flagged/i)).toBeInTheDocument();
    // The label appears both as a meter and inside a generated statement, which is
    // expected: the meter is the bar, the statement is the sentence about it.
    expect(screen.getAllByText(/language-model predictability/i).length).toBeGreaterThan(1);
    expect(document.querySelector('.meter__blocks')).toBeInTheDocument();
    expect(screen.getByText(/essay median 31.8/i)).toBeInTheDocument();
    expect(screen.getByText(/no language model was asked to explain anything/i)).toBeInTheDocument();
  });

  it('falls back to whole-essay evidence when nothing is selected', () => {
    render(
      <EvidencePanel
        sentence={null}
        documentEvidence={SAMPLE_ANALYSIS.evidence}
        hasSentenceModel
      />,
    );
    expect(screen.getByText(/whole essay/i)).toBeInTheDocument();
    expect(screen.getByText(/hover or click any sentence/i)).toBeInTheDocument();
  });

  it('says so when a sentence has no attached evidence', () => {
    render(
      <EvidencePanel
        sentence={SAMPLE_ANALYSIS.sentences[0]!}
        documentEvidence={SAMPLE_ANALYSIS.evidence}
        hasSentenceModel
      />,
    );
    expect(screen.getByText(/measured within ordinary ranges/i)).toBeInTheDocument();
  });
});

describe('AnalysePage', () => {
  const modelInfo = {
    ready: true,
    error: null,
    detector_version: '1.0.0',
    model_version: '1.0.0',
    dataset_version: '1.0.0',
    features_version: '1.0.0',
    trained_at: null,
    data_regime: 'bootstrap',
    document_model: {
      name: 'hybrid::random_forest',
      n_features: 411,
      feature_groups: [],
      calibration: 'sigmoid',
      classes: [],
    },
    sentence_model: {},
    language_model: {},
    metrics: {},
    training: {},
    feature_importance: [],
    model_comparison: [],
    methodology: {
      summary: '',
      pipeline: [],
      what_the_language_model_does: '',
      what_makes_the_decision: '',
      signals_measured: [],
      limitations: [],
    },
  };

  it('runs an analysis and hands the result upward with the submitted text', async () => {
    stubApi();
    const user = userEvent.setup();
    const onAnalysed = vi.fn();
    render(
      <AnalysePage
        health={SAMPLE_HEALTH}
        modelInfo={modelInfo}
        backendReachable
        onNavigate={vi.fn()}
        onAnalysed={onAnalysed}
      />,
    );

    await user.click(screen.getByRole('button', { name: /load an example/i }));
    await user.click(screen.getByText(/machine-register essay/i));
    await user.click(screen.getByRole('button', { name: /analyse essay/i }));

    // Results now live on their own route, so this page reports rather than renders.
    // The submitted text is passed along because the results view re-slices it
    // using the returned offsets when the server did not store the essay.
    await waitFor(() => expect(onAnalysed).toHaveBeenCalledTimes(1));
    const [analysis, submittedText] = onAnalysed.mock.calls[0]!;
    expect(analysis.classification).toBe('ai_polished');
    expect(submittedText).toContain('From an early age');
  });

  it('surfaces a backend error without crashing', async () => {
    stubApi({ error: { code: 'model_not_trained', message: 'x' } }, 503);
    const user = userEvent.setup();
    render(
      <AnalysePage
        health={SAMPLE_HEALTH}
        modelInfo={modelInfo}
        backendReachable
        onNavigate={vi.fn()}
        onAnalysed={vi.fn()}
      />,
    );

    await user.click(screen.getByRole('button', { name: /load an example/i }));
    await user.click(screen.getByText(/machine-register essay/i));
    await user.click(screen.getByRole('button', { name: /analyse essay/i }));

    await waitFor(() => expect(screen.getByText(/analysis failed/i)).toBeInTheDocument());
    expect(screen.getByText(/has not been trained/i)).toBeInTheDocument();
  });

  it('warns when the backend is unreachable', () => {
    stubApi();
    render(
      <AnalysePage
        health={null}
        modelInfo={null}
        backendReachable={false}
        onNavigate={vi.fn()}
        onAnalysed={vi.fn()}
      />,
    );
    expect(screen.getByText(/backend unreachable/i)).toBeInTheDocument();
  });

  it('warns that the model is trained on bootstrap data', () => {
    stubApi();
    render(
      <AnalysePage
        health={SAMPLE_HEALTH}
        modelInfo={modelInfo}
        backendReachable
        onNavigate={vi.fn()}
        onAnalysed={vi.fn()}
      />,
    );
    expect(screen.getByText(/trained on bootstrap data/i)).toBeInTheDocument();
  });
});
