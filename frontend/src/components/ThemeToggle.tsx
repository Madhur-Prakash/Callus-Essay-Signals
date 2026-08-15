import type { ThemeChoice } from '@/hooks/useTheme';

interface Props {
  choice: ThemeChoice;
  resolved: 'light' | 'dark';
  onCycle: () => void;
}

const NEXT_LABEL: Record<ThemeChoice, string> = {
  system: 'light',
  light: 'dark',
  dark: 'system',
};

const ICON: Record<ThemeChoice, JSX.Element> = {
  light: (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="theme-toggle__icon">
      <circle cx="8" cy="8" r="3.2" />
      <path d="M8 1v1.8M8 13.2V15M15 8h-1.8M2.8 8H1M12.9 3.1l-1.3 1.3M4.4 11.6l-1.3 1.3M12.9 12.9l-1.3-1.3M4.4 4.4L3.1 3.1" />
    </svg>
  ),
  dark: (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="theme-toggle__icon">
      <path d="M13.5 9.6A5.8 5.8 0 0 1 6.4 2.5a5.8 5.8 0 1 0 7.1 7.1z" />
    </svg>
  ),
  system: (
    <svg viewBox="0 0 16 16" aria-hidden="true" className="theme-toggle__icon">
      <rect x="1.6" y="2.6" width="12.8" height="8.4" rx="1.4" />
      <path d="M5.6 13.6h4.8" />
    </svg>
  ),
};

/**
 * Three-state theme control. "System" is a genuine state — it keeps following the
 * OS rather than freezing whatever the OS happened to say at first paint.
 */
export function ThemeToggle({ choice, resolved, onCycle }: Props) {
  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={onCycle}
      aria-label={`Theme: ${choice}${choice === 'system' ? ` (currently ${resolved})` : ''}. Switch to ${NEXT_LABEL[choice]}.`}
      title={`Theme: ${choice} — click for ${NEXT_LABEL[choice]}`}
      data-choice={choice}
    >
      {ICON[choice]}
      <span className="theme-toggle__text">{choice}</span>
    </button>
  );
}
