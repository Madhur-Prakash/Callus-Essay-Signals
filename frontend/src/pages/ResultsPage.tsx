import { motion } from 'framer-motion';
import { useMemo, useState } from 'react';

import type { Route } from '@/App';
import { Banner } from '@/components/Banner';
import { EvidencePanel } from '@/components/EvidencePanel';
import { HighlightedEssay, ParagraphBreakdown } from '@/components/HighlightedEssay';
import { RepetitionPanel } from '@/components/RepetitionPanel';
import { RhythmChart } from '@/components/RhythmChart';
import { SummaryStats } from '@/components/SummaryStats';
import { VerdictCard } from '@/components/VerdictCard';
import { TabPanel, Tabs, type TabDefinition } from '@/components/ui/Tabs';
import { fadeUp, stagger } from '@/hooks/useMotion';
import type { AnalysisResponse } from '@/types/api';

interface Props {
  result: AnalysisResponse;
  essayText: string;
  onNavigate: (route: Route) => void;
  onClear: () => void;
}

/**
 * The analysis results, split across tabs.
 *
 * Previously this was one page carrying the verdict, the marked-up essay, the
 * rhythm chart, the paragraph rollup, repetition findings and a full statistics
 * table — several screens of dense material with no way to get back to a specific
 * part. Tabs give each concern its own surface and make the second visit fast.
 *
 * The verdict stays *outside* the tabs on purpose: it is the answer to the
 * question the user asked, and it should never be one click away.
 */
export function ResultsPage({ result, essayText, onNavigate, onClear }: Props) {
  const [tab, setTab] = useState('essay');
  const [selectedSentenceId, setSelectedSentenceId] = useState<number | null>(null);

  const selectedSentence = useMemo(() => {
    if (selectedSentenceId === null) return null;
    return result.sentences.find((s) => s.sentence_id === selectedSentenceId) ?? null;
  }, [result.sentences, selectedSentenceId]);

  const repetitionCount =
    result.repetition.repeated_phrases.length +
    result.repetition.repeated_syntactic_templates.length;

  const tabs: TabDefinition[] = [
    { value: 'essay', label: 'Marked-up essay', badge: result.summary.flagged_sentences },
    { value: 'rhythm', label: 'Rhythm & structure' },
    { value: 'repetition', label: 'Repetition', badge: repetitionCount },
    { value: 'stats', label: 'All statistics' },
  ];

  /** Jump to a sentence from another tab (e.g. a repeated phrase). */
  const focusSentence = (sentenceId: number) => {
    setSelectedSentenceId(sentenceId);
    setTab('essay');
  };

  return (
    <motion.div
      className="results"
      variants={stagger}
      initial="hidden"
      animate="visible"
    >
      <motion.div className="row row--wrap" variants={fadeUp}>
        <button type="button" className="btn btn--sm" onClick={() => onNavigate('analyse')}>
          ← Analyse another essay
        </button>
        <button
          type="button"
          className="btn btn--sm"
          onClick={() => {
            onClear();
            onNavigate('analyse');
          }}
        >
          Start over
        </button>
        <div className="spacer" />
        <span className="tag">{result.cached ? 'served from cache' : 'freshly analysed'}</span>
        <span className="tag">{result.persisted ? 'saved' : 'not saved'}</span>
        {result.timings.total_ms !== undefined && (
          <span className="tag mono">{Math.round(result.timings.total_ms)} ms</span>
        )}
      </motion.div>

      {result.warnings.length > 0 && (
        <motion.div variants={fadeUp}>
          <Banner tone="warning" title="Caveats for this analysis">
            <ul style={{ margin: '0.2rem 0 0', paddingLeft: '1.1rem' }}>
              {result.warnings.map((warning, index) => (
                <li key={index} style={{ marginBottom: '0.2rem' }}>
                  {warning}
                </li>
              ))}
            </ul>
          </Banner>
        </motion.div>
      )}

      <motion.div variants={fadeUp}>
        <VerdictCard result={result} />
      </motion.div>

      <motion.div variants={fadeUp}>
        <Tabs tabs={tabs} value={tab} onValueChange={setTab} label="Analysis detail" sticky>
          <TabPanel value="essay">
            <section className="card">
              <div className="card__head">
                <div>
                  <p className="card__title">The essay, marked up</p>
                  <p className="section-note">
                    Click or hover a sentence to see the measurements behind it.
                  </p>
                </div>
              </div>
              <div className="card__body">
                <div className="reader-grid">
                  <HighlightedEssay
                    result={result}
                    essayText={essayText}
                    selectedSentenceId={selectedSentenceId}
                    onSelectSentence={setSelectedSentenceId}
                  />
                  <EvidencePanel
                    sentence={selectedSentence}
                    documentEvidence={result.evidence}
                    hasSentenceModel={result.summary.sentences_scored > 0}
                  />
                </div>
              </div>
            </section>
          </TabPanel>

          <TabPanel value="rhythm">
            <div className="stack stack--lg">
              <section className="card">
                <div className="card__head">
                  <div>
                    <p className="card__title">Sentence rhythm</p>
                    <p className="section-note">
                      Sentence lengths across the essay, against its own mean. Variation here is
                      one feature among many — uniformity alone is not evidence of anything.
                    </p>
                  </div>
                </div>
                <div className="card__body">
                  <RhythmChart
                    rhythm={result.rhythm}
                    sentences={result.sentences}
                    onSelectSentence={focusSentence}
                    selectedSentenceId={selectedSentenceId}
                  />
                </div>
              </section>

              <section className="card">
                <div className="card__head">
                  <div>
                    <p className="card__title">Paragraph breakdown</p>
                    <p className="section-note">
                      Weighted by sentence length. Select a paragraph to jump to its first
                      flagged sentence.
                    </p>
                  </div>
                </div>
                <div className="card__body">
                  <ParagraphBreakdown
                    paragraphs={result.paragraphs}
                    sentences={result.sentences}
                    onSelectSentence={focusSentence}
                  />
                </div>
              </section>
            </div>
          </TabPanel>

          <TabPanel value="repetition">
            <section className="card">
              <div className="card__head">
                <div>
                  <p className="card__title">Repetition</p>
                  <p className="section-note">
                    Concrete repeated spans and grammatical templates found in this essay.
                  </p>
                </div>
              </div>
              <div className="card__body">
                <RepetitionPanel
                  repetition={result.repetition}
                  onSelectSentence={focusSentence}
                />
              </div>
            </section>
          </TabPanel>

          <TabPanel value="stats">
            <section className="card">
              <div className="card__head">
                <div>
                  <p className="card__title">All measured statistics</p>
                  <p className="section-note">
                    Every number below came from this analysis. Nothing is estimated in the
                    browser.
                  </p>
                </div>
              </div>
              <div className="card__body">
                <SummaryStats summary={result.summary} />
              </div>
            </section>
          </TabPanel>
        </Tabs>
      </motion.div>

      <motion.div variants={fadeUp}>
        <Banner tone="info" title="What this result is">
          {result.disclaimer}{' '}
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={() => onNavigate('how')}
          >
            How this works
          </button>
          <button
            type="button"
            className="btn btn--ghost btn--sm"
            onClick={() => onNavigate('limitations')}
          >
            Known limitations
          </button>
        </Banner>
      </motion.div>

      <motion.p className="tiny muted" style={{ textAlign: 'center' }} variants={fadeUp}>
        analysis {result.analysis_id.slice(0, 12)} · detector v{result.model.detector_version} ·
        model v{result.model.model_version} · features v{result.model.features_version} ·
        instrument {result.model.language_model}
      </motion.p>
    </motion.div>
  );
}
