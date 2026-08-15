import { motion } from 'framer-motion';

import type { Route } from '@/App';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useScrollProgress } from '@/hooks/useScrollNav';
import type { useTheme } from '@/hooks/useTheme';
import type { HealthResponse } from '@/types/api';

interface Props {
  route: Route;
  onNavigate: (route: Route) => void;
  health: HealthResponse | null;
  theme: ReturnType<typeof useTheme>;
  /** Results only exist after an analysis, so the tab is conditional. */
  hasResult: boolean;
}

const BASE_NAV: Array<{ id: Route; label: string }> = [
  { id: 'analyse', label: 'Analyse' },
  { id: 'research', label: 'Research' },
  { id: 'how', label: 'How it works' },
  { id: 'limitations', label: 'Limits' },
];

export function Masthead({ route, onNavigate, health, theme, hasResult }: Props) {
  const progress = useScrollProgress();

  const nav = hasResult
    ? [BASE_NAV[0]!, { id: 'results' as Route, label: 'Results' }, ...BASE_NAV.slice(1)]
    : BASE_NAV;

  const statusColour =
    health === null
      ? 'var(--ink-faint)'
      : health.status === 'ok'
        ? 'var(--human)'
        : health.status === 'degraded'
          ? 'var(--possible)'
          : 'var(--likely)';

  const statusText =
    health === null
      ? 'checking backend'
      : health.status === 'ok'
        ? 'all components healthy'
        : health.status === 'degraded'
          ? 'running with reduced capability'
          : 'detector unavailable';

  return (
    <header className="masthead">
      <div className="masthead__inner">
        <div className="wordmark">
          {/* Routing is hash-based and owned by App.tsx — there is no Router
              provider, so a react-router <Link> would crash here. The wordmark
              uses the same onNavigate callback as the nav buttons below. */}
          <button
            type="button"
            className="wordmark__home"
            onClick={() => onNavigate('analyse')}
            aria-label="Essay Signals — go to the analyse page"
          >
            <span className="wordmark__mark" aria-hidden="true">
              ∿
            </span>
            <span>Essay Signals</span>
          </button>
          <span className="wordmark__sub">evidence-based detection</span>
        </div>

        <nav className="nav" aria-label="Main">
          {nav.map((item) => (
            <button
              key={item.id}
              type="button"
              className="nav__link"
              aria-current={route === item.id ? 'page' : undefined}
              onClick={() => onNavigate(item.id)}
            >
              {/* A shared layoutId lets the active pill slide between tabs
                  instead of blinking out and in. */}
              {route === item.id && (
                <motion.span
                  layoutId="nav-active"
                  className="nav__indicator"
                  transition={{ type: 'spring', stiffness: 420, damping: 34 }}
                />
              )}
              <span className="nav__label">{item.label}</span>
            </button>
          ))}
        </nav>

        <ThemeToggle choice={theme.choice} resolved={theme.resolved} onCycle={theme.cycle} />

        <span className="tag" title={statusText} data-testid="health-chip">
          <span
            aria-hidden="true"
            className="inline-block size-2 rounded-full"
            style={{ background: statusColour }}
          />
          {health?.status ?? '…'}
        </span>
      </div>

      {/* Reading position. Informational rather than decorative, so it is kept
          even under reduced motion — it just stops being smoothed. */}
      <div className="masthead__progress" aria-hidden="true">
        <div className="masthead__progress-bar" style={{ transform: `scaleX(${progress})` }} />
      </div>
    </header>
  );
}
