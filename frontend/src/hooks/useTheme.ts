/**
 * Theme control: system / light / dark.
 *
 * "System" is a real third state, not a synonym for one of the other two - it
 * means "keep following the OS", so a user who switches their machine to dark at
 * sunset gets dark here too. The choice is written to `data-theme` on <html>
 * (absent when following the system) and persisted to localStorage.
 *
 * The corresponding CSS defines the palette three times: on bare `:root` (light),
 * under `prefers-color-scheme: dark` guarded by `:not([data-theme="light"])`, and
 * under `[data-theme="dark"]` - so an explicit choice wins in both directions.
 */

import { useCallback, useEffect, useState } from 'react';

export type ThemeChoice = 'system' | 'light' | 'dark';

const STORAGE_KEY = 'essay-signals:theme';

function readStored(): ThemeChoice {
  if (typeof localStorage === 'undefined') return 'system';
  const raw = localStorage.getItem(STORAGE_KEY);
  return raw === 'light' || raw === 'dark' || raw === 'system' ? raw : 'system';
}

function apply(choice: ThemeChoice): void {
  const root = document.documentElement;
  if (choice === 'system') {
    root.removeAttribute('data-theme');
  } else {
    root.setAttribute('data-theme', choice);
  }
}

/** Which palette is actually showing right now, after resolving "system". */
export function resolveTheme(choice: ThemeChoice): 'light' | 'dark' {
  if (choice !== 'system') return choice;
  if (typeof matchMedia === 'undefined') return 'dark';
  return matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function useTheme() {
  const [choice, setChoice] = useState<ThemeChoice>(readStored);
  const [resolved, setResolved] = useState<'light' | 'dark'>(() => resolveTheme(readStored()));

  useEffect(() => {
    apply(choice);
    setResolved(resolveTheme(choice));
    try {
      localStorage.setItem(STORAGE_KEY, choice);
    } catch {
      /* private browsing: the theme simply will not persist */
    }
  }, [choice]);

  // Keep following the OS while the choice is "system".
  useEffect(() => {
    if (choice !== 'system' || typeof matchMedia === 'undefined') return;
    const query = matchMedia('(prefers-color-scheme: dark)');
    const onChange = () => setResolved(query.matches ? 'dark' : 'light');
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, [choice]);

  const cycle = useCallback(() => {
    setChoice((current) =>
      current === 'system' ? 'light' : current === 'light' ? 'dark' : 'system',
    );
  }, []);

  return { choice, resolved, setChoice, cycle };
}

/** True when the user has asked the OS to minimise motion. */
export function prefersReducedMotion(): boolean {
  if (typeof matchMedia === 'undefined') return false;
  return matchMedia('(prefers-reduced-motion: reduce)').matches;
}
