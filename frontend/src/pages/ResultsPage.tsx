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
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { Section } from '@/components/ui/Card';
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
        <Button size="sm" onClick={() => onNavigate('analyse')}>
          ← Analyse another essay
        </Button>
        <Button
          size="sm"
          onClick={() => {
            onClear();
            onNavigate('analyse');
          }}
        >
          Start over
        </Button>
        <div className="spacer" />
        <Badge>{result.cached ? 'served from cache' : 'freshly analysed'}</Badge>
        <Badge>{result.persisted ? 'saved' : 'not saved'}</Badge>
        {result.timings.total_ms !== undefined && (
          <Badge mono>{Math.round(result.timings.total_ms)} ms</Badge>
        )}
      </motion.div>

      {result.warnings.length > 0 && (
        <motion.div variants={fadeUp}>
          <Banner tone="warning" title="Caveats for this analysis">
            <ul className="mb-0 mt-1 grid list-disc gap-1 pl-[1.1rem]">
              {result.warnings.map((warning, index) => (
                <li key={index}>{warning}</li>
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
            <Section
              title="The essay, marked up"
              note="Click or hover a sentence to see the measurements behind it."
            >
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
            </Section>
          </TabPanel>

          <TabPanel value="rhythm">
            <div className="stack stack--lg">
              <Section
                title="Sentence rhythm"
                note="Sentence lengths across the essay, against its own mean. Variation here is one feature among many — uniformity alone is not evidence of anything."
              >
                <RhythmChart
                  rhythm={result.rhythm}
                  sentences={result.sentences}
                  onSelectSentence={focusSentence}
                  selectedSentenceId={selectedSentenceId}
                />
              </Section>

              <Section
                title="Paragraph breakdown"
                note="Weighted by sentence length. Select a paragraph to jump to its first flagged sentence."
              >
                <ParagraphBreakdown
                  paragraphs={result.paragraphs}
                  sentences={result.sentences}
                  onSelectSentence={focusSentence}
                />
              </Section>
            </div>
          </TabPanel>

          <TabPanel value="repetition">
            <Section
              title="Repetition"
              note="Concrete repeated spans and grammatical templates found in this essay."
            >
              <RepetitionPanel repetition={result.repetition} onSelectSentence={focusSentence} />
            </Section>
          </TabPanel>

          <TabPanel value="stats">
            <Section
              title="All measured statistics"
              note="Every number below came from this analysis. Nothing is estimated in the browser."
            >
              <SummaryStats summary={result.summary} />
            </Section>
          </TabPanel>
        </Tabs>
      </motion.div>

      <motion.div variants={fadeUp}>
        <Banner tone="info" title="What this result is">
          {result.disclaimer}{' '}
          <Button variant="ghost" size="sm" onClick={() => onNavigate('how')}>
            How this works
          </Button>
          <Button variant="ghost" size="sm" onClick={() => onNavigate('limitations')}>
            Known limitations
          </Button>
        </Banner>
      </motion.div>

      <motion.p className="tiny muted text-center" variants={fadeUp}>
        analysis {result.analysis_id.slice(0, 12)} · detector v{result.model.detector_version} ·
        model v{result.model.model_version} · features v{result.model.features_version} ·
        instrument {result.model.language_model}
      </motion.p>
    </motion.div>
  );
}
