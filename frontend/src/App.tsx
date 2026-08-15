import { AnimatePresence, MotionConfig, motion } from 'framer-motion';
import { useCallback, useEffect, useState } from 'react';

import { fetchHealth, fetchModelInfo } from '@/api/client';
import { Masthead } from '@/components/Masthead';
import { EASE, useSmoothScroll } from '@/hooks/useMotion';
import { useTheme } from '@/hooks/useTheme';
import { AnalysePage } from '@/pages/AnalysePage';
import { HowItWorksPage } from '@/pages/HowItWorksPage';
import { LimitationsPage } from '@/pages/LimitationsPage';
import { ResearchPage } from '@/pages/ResearchPage';
import { ResultsPage } from '@/pages/ResultsPage';
import type { AnalysisResponse, HealthResponse, ModelInfoResponse } from '@/types/api';

export type Route = 'analyse' | 'results' | 'research' | 'how' | 'limitations';

const ROUTE_FROM_HASH: Record<string, Route> = {
  '': 'analyse',
  '#/': 'analyse',
  '#/analyse': 'analyse',
  '#/results': 'results',
  '#/research': 'research',
  '#/how-it-works': 'how',
  '#/limitations': 'limitations',
};

const HASH_FROM_ROUTE: Record<Route, string> = {
  analyse: '#/analyse',
  results: '#/results',
  research: '#/research',
  how: '#/how-it-works',
  limitations: '#/limitations',
};

function currentRoute(): Route {
  // Ignore any query/sub-hash so `#/research?tab=bias` still resolves.
  const base = window.location.hash.split('?')[0] ?? '';
  return ROUTE_FROM_HASH[base] ?? 'analyse';
}

export function App() {
  const [route, setRoute] = useState<Route>(currentRoute);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [backendReachable, setBackendReachable] = useState<boolean | null>(null);
  const theme = useTheme();

  // The analysis lives here rather than in AnalysePage so it survives navigating
  // to Research and back. It is deliberately NOT persisted to storage: the essay
  // is sensitive, and "not stored" should mean not stored anywhere.
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [analysedText, setAnalysedText] = useState('');

  useSmoothScroll();

  useEffect(() => {
    const onHashChange = () => setRoute(currentRoute());
    window.addEventListener('hashchange', onHashChange);
    return () => window.removeEventListener('hashchange', onHashChange);
  }, []);

  const navigate = useCallback((next: Route) => {
    window.location.hash = HASH_FROM_ROUTE[next];
    setRoute(next);
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, []);

  // A deep link to /results with nothing analysed has nothing to show.
  useEffect(() => {
    if (route === 'results' && !result) navigate('analyse');
  }, [route, result, navigate]);

  const onAnalysed = useCallback(
    (analysis: AnalysisResponse, submittedText: string) => {
      setResult(analysis);
      setAnalysedText(submittedText);
      navigate('results');
    },
    [navigate],
  );

  const onClearResult = useCallback(() => {
    setResult(null);
    setAnalysedText('');
  }, []);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [healthResult, infoResult] = await Promise.all([
          fetchHealth(),
          fetchModelInfo().catch(() => null),
        ]);
        if (cancelled) return;
        setHealth(healthResult);
        setModelInfo(infoResult);
        setBackendReachable(true);
      } catch {
        if (!cancelled) setBackendReachable(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    // `reducedMotion="user"` makes every Framer animation collapse to an instant
    // state change when the OS asks for reduced motion. GSAP and Lenis are gated
    // separately in useMotion, so all three libraries obey the same preference.
    <MotionConfig reducedMotion="user">
      <div className="app">
        <Masthead
          route={route}
          onNavigate={navigate}
          health={health}
          theme={theme}
          hasResult={Boolean(result)}
        />
        <main className="main">
          {/* `mode="wait"` so the outgoing view finishes before the next arrives —
              cross-fading two full pages of dense text is unreadable. */}
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={route}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.28, ease: EASE }}
            >
              {route === 'analyse' && (
                <AnalysePage
                  health={health}
                  modelInfo={modelInfo}
                  backendReachable={backendReachable}
                  onNavigate={navigate}
                  onAnalysed={onAnalysed}
                />
              )}
              {route === 'results' && result && (
                <ResultsPage
                  result={result}
                  essayText={analysedText}
                  onNavigate={navigate}
                  onClear={onClearResult}
                />
              )}
              {route === 'research' && <ResearchPage />}
              {route === 'how' && <HowItWorksPage modelInfo={modelInfo} />}
              {route === 'limitations' && (
                <LimitationsPage modelInfo={modelInfo} onNavigate={navigate} />
              )}
            </motion.div>
          </AnimatePresence>
        </main>
        <footer className="footer">
          {/* No inline margin here: `.footer p` centres the measure with
              `margin: 0 auto`, and an inline `margin: 0` would win and pin it left. */}
          <p>
            Detection is probabilistic. A flag is evidence for human review, never proof of
            authorship. This tool must not be the sole basis for any decision about a person.
          </p>
        </footer>
      </div>
    </MotionConfig>
  );
}
