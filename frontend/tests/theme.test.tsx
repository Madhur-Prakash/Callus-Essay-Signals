import { act, render, renderHook, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { ThemeToggle } from '@/components/ThemeToggle';
import { resolveTheme, useTheme } from '@/hooks/useTheme';

function stubMatchMedia(prefersDark: boolean) {
  const listeners = new Set<() => void>();
  const mql = {
    matches: prefersDark,
    media: '(prefers-color-scheme: dark)',
    addEventListener: (_: string, fn: () => void) => listeners.add(fn),
    removeEventListener: (_: string, fn: () => void) => listeners.delete(fn),
    addListener: vi.fn(),
    removeListener: vi.fn(),
    dispatchEvent: vi.fn(),
    onchange: null,
  };
  vi.stubGlobal(
    'matchMedia',
    vi.fn((query: string) => ({ ...mql, media: query, matches: prefersDark })),
  );
  return { mql, listeners };
}

describe('useTheme', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.removeAttribute('data-theme');
    vi.unstubAllGlobals();
  });

  it('defaults to following the system and sets no attribute', () => {
    stubMatchMedia(true);
    const { result } = renderHook(() => useTheme());
    expect(result.current.choice).toBe('system');
    // No data-theme means the CSS media query is in charge.
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false);
  });

  it('cycles system → light → dark → system', () => {
    stubMatchMedia(true);
    const { result } = renderHook(() => useTheme());

    act(() => result.current.cycle());
    expect(result.current.choice).toBe('light');
    expect(document.documentElement.getAttribute('data-theme')).toBe('light');

    act(() => result.current.cycle());
    expect(result.current.choice).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');

    act(() => result.current.cycle());
    expect(result.current.choice).toBe('system');
    expect(document.documentElement.hasAttribute('data-theme')).toBe(false);
  });

  it('persists the choice across mounts', () => {
    stubMatchMedia(false);
    const first = renderHook(() => useTheme());
    act(() => first.result.current.setChoice('dark'));
    first.unmount();

    const second = renderHook(() => useTheme());
    expect(second.result.current.choice).toBe('dark');
    expect(document.documentElement.getAttribute('data-theme')).toBe('dark');
  });

  it('resolves "system" against the OS preference', () => {
    stubMatchMedia(true);
    expect(resolveTheme('system')).toBe('dark');
    stubMatchMedia(false);
    expect(resolveTheme('system')).toBe('light');
    // An explicit choice ignores the OS entirely.
    expect(resolveTheme('dark')).toBe('dark');
    expect(resolveTheme('light')).toBe('light');
  });

  it('survives localStorage being unavailable', () => {
    stubMatchMedia(true);
    const setItem = vi
      .spyOn(Storage.prototype, 'setItem')
      .mockImplementation(() => {
        throw new Error('private browsing');
      });
    const { result } = renderHook(() => useTheme());
    expect(() => act(() => result.current.cycle())).not.toThrow();
    expect(result.current.choice).toBe('light');
    setItem.mockRestore();
  });
});

describe('ThemeToggle', () => {
  it('announces the current state and the next one', async () => {
    const user = userEvent.setup();
    const onCycle = vi.fn();
    render(<ThemeToggle choice="system" resolved="dark" onCycle={onCycle} />);

    const button = screen.getByRole('button', { name: /theme: system/i });
    expect(button).toHaveAccessibleName(/currently dark/i);
    expect(button).toHaveAccessibleName(/switch to light/i);

    await user.click(button);
    expect(onCycle).toHaveBeenCalledOnce();
  });

  it('does not claim a resolved theme when the choice is explicit', () => {
    render(<ThemeToggle choice="dark" resolved="dark" onCycle={vi.fn()} />);
    const button = screen.getByRole('button', { name: /theme: dark/i });
    expect(button).not.toHaveAccessibleName(/currently/i);
  });
});
