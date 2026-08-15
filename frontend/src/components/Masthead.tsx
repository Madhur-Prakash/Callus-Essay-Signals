import { motion } from 'framer-motion';

import type { Route } from '@/App';
import { ThemeToggle } from '@/components/ThemeToggle';
import type { useTheme } from '@/hooks/useTheme';
import type { HealthResponse } from '@/types/api';

interface Props {
  route: Route;
  onNavigate: (route: Route) => void;
  health: HealthResponse | null;
  theme: ReturnType<typeof useTheme>;
}

const NAV: Array<{ id: Route; label: string }> = [
  { id: 'analyse', label: 'Analyse' },
  { id: 'research', label: 'Research' },
  { id: 'how', label: 'How it works' },
];

export function Masthead({ route, onNavigate, health, theme }: Props) {
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
          {NAV.map((item) => (
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

        <ThemeToggle
          choice={theme.choice}
          resolved={theme.resolved}
          onCycle={theme.cycle}
        />

        <span
          className="tag"
          title={statusText}
          style={{ whiteSpace: 'nowrap' }}
          data-testid="health-chip"
        >
          <span
            aria-hidden="true"
            style={{
              width: '0.5rem',
              height: '0.5rem',
              borderRadius: '50%',
              background: statusColour,
              display: 'inline-block',
            }}
          />
          {health?.status ?? '…'}
        </span>
      </div>
    </header>
  );
}
