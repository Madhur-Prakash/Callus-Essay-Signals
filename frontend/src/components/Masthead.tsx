import { motion } from 'framer-motion';

import type { Route } from '@/App';
import { ThemeToggle } from '@/components/ThemeToggle';
import { useScrolled } from '@/hooks/useScrollNav';
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

/**
 * Only failures appear in the bar.
 *
 * A healthy backend is the expected case, and spending a permanent indicator on
 * saying so is how a status light becomes furniture - by the time it turns red,
 * nobody is looking at it any more. Nothing renders for `ok`, and nothing while
 * the check is still in flight, so the bar never flickers a chip on first load.
 *
 * An unreachable backend is not handled here either: `health` stays null in that
 * case and AnalysePage already shows a full banner with the command to start it.
 */
const HEALTH_ALERT: Record<string, { colour: string; label: string; title: string }> = {
  degraded: {
    colour: 'var(--possible)',
    label: 'Degraded',
    title: 'Backend running with reduced capability',
  },
  unavailable: {
    colour: 'var(--likely)',
    label: 'Detector offline',
    title: 'The detector is unavailable - analysis will fail',
  },
};

/**
 * The app bar.
 *
 * Three zones on a `1fr auto 1fr` grid - wordmark, navigation, utilities - so the
 * nav is optically centred no matter how wide the other two get. A flex row with
 * `margin-left: auto` pushed the nav against the utilities and left the bar with
 * no rhythm.
 *
 * The bar is one pill and only one. Its earlier form nested a pill inside a pill
 * inside a pill (bar → nav group → toggle → status chip), and four competing
 * rounded rectangles crammed together is what made it read as cluttered. Now the
 * only other pill is the sliding active indicator, which earns its shape by
 * carrying meaning.
 */
export function Masthead({ route, onNavigate, health, theme, hasResult }: Props) {
  const scrolled = useScrolled();

  const nav = hasResult
    ? [BASE_NAV[0]!, { id: 'results' as Route, label: 'Results' }, ...BASE_NAV.slice(1)]
    : BASE_NAV;

  const alert = health ? HEALTH_ALERT[health.status] : undefined;

  return (
    <header className="masthead" data-scrolled={scrolled || undefined}>
      <div className="masthead__inner">
        <div className="masthead__zone masthead__zone--start">
          {/* Routing is hash-based and owned by App.tsx - there is no Router
              provider, so a react-router <Link> would crash here. */}
          <button
            type="button"
            className="wordmark"
            onClick={() => onNavigate('analyse')}
            aria-label="Essay Signals - go to the analyse page"
          >
            <span className="wordmark__mark" aria-hidden="true">
              ∿
            </span>
            <span className="wordmark__name">Essay Signals</span>
          </button>
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
              {/* A shared layoutId lets the active pill slide between items
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

        <div className="masthead__zone masthead__zone--end">
          {alert && (
            // `role="status"` rather than "alert": this arrives from a poll, not
            // in response to something the user just did, so it should be
            // announced politely instead of interrupting.
            <span
              className="health"
              role="status"
              title={alert.title}
              data-testid="health-chip"
              // One property drives both the dot and its halo, so the two can
              // never drift apart - the halo is mixed from it in CSS.
              style={{ ['--dot' as string]: alert.colour }}
            >
              <span className="health__dot" aria-hidden="true" />
              <span className="health__label">{alert.label}</span>
            </span>
          )}

          <ThemeToggle choice={theme.choice} resolved={theme.resolved} onCycle={theme.cycle} />
        </div>
      </div>
    </header>
  );
}
