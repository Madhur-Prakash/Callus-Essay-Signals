import '@testing-library/jest-dom/vitest';

import { afterEach, vi } from 'vitest';

/*
 * jsdom implements neither of these, and both are called on nearly every render:
 * `scrollTo` by the router, `matchMedia` by the theme and motion hooks. Left
 * unstubbed they print a stack trace per call, which buries the one line that
 * matters when a test actually fails.
 */
// Defined rather than spied: `vi.restoreAllMocks()` in afterEach would put
// jsdom's throwing implementation back, and the noise would return mid-suite.
Object.defineProperty(window, 'scrollTo', { value: () => {}, writable: true });

if (!window.matchMedia) {
  Object.defineProperty(window, 'matchMedia', {
    writable: true,
    value: (query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addEventListener: () => {},
      removeEventListener: () => {},
      addListener: () => {},
      removeListener: () => {},
      dispatchEvent: () => false,
    }),
  });
}

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});
