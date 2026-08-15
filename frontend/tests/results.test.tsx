import { render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { ResultsPage } from '@/pages/ResultsPage';

import { SAMPLE_ANALYSIS } from './fixtures';

function renderResults(overrides: Partial<React.ComponentProps<typeof ResultsPage>> = {}) {
  const props = {
    result: SAMPLE_ANALYSIS,
    essayText: '',
    onNavigate: vi.fn(),
    onClear: vi.fn(),
    ...overrides,
  };
  return { ...render(<ResultsPage {...props} />), props };
}

describe('ResultsPage', () => {
  it('keeps the verdict outside the tabs so it is never a click away', () => {
    renderResults();
    expect(
      screen.getByRole('heading', { name: /potentially ai-polished/i }),
    ).toBeInTheDocument();
    // The verdict is not inside any tab panel.
    const verdict = screen.getByRole('heading', { name: /potentially ai-polished/i });
    expect(verdict.closest('[role="tabpanel"]')).toBeNull();
  });

  it('opens on the marked-up essay tab', () => {
    renderResults();
    const tab = screen.getByRole('tab', { name: /marked-up essay/i });
    expect(tab).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByText(/the essay, marked up/i)).toBeInTheDocument();
  });

  it('exposes every section as a tab with counts where relevant', () => {
    renderResults();
    const tablist = screen.getByRole('tablist', { name: /analysis detail/i });
    const tabs = within(tablist).getAllByRole('tab');
    expect(tabs.map((t) => t.textContent)).toEqual([
      expect.stringContaining('Marked-up essay'),
      expect.stringContaining('Rhythm & structure'),
      expect.stringContaining('Repetition'),
      expect.stringContaining('All statistics'),
    ]);
    // The flagged-sentence count is surfaced on the tab itself.
    expect(within(tabs[0]!).getByText('2')).toBeInTheDocument();
  });

  it('switches to the rhythm tab and shows its sections', async () => {
    const user = userEvent.setup();
    renderResults();

    await user.click(screen.getByRole('tab', { name: /rhythm & structure/i }));
    expect(
      screen.getByText('Sentence rhythm', { selector: '.card__title' }),
    ).toBeInTheDocument();
    expect(screen.getByText(/paragraph breakdown/i)).toBeInTheDocument();
    // The essay panel is no longer mounted.
    expect(screen.queryByText(/the essay, marked up/i)).not.toBeInTheDocument();
  });

  it('shows the statistics table on its own tab', async () => {
    const user = userEvent.setup();
    renderResults();
    await user.click(screen.getByRole('tab', { name: /all statistics/i }));
    expect(screen.getByText(/all measured statistics/i)).toBeInTheDocument();
  });

  it('jumps back to the essay tab when a repeated phrase is selected', async () => {
    const user = userEvent.setup();
    renderResults();

    await user.click(screen.getByRole('tab', { name: /repetition/i }));
    expect(screen.getByText('transformative journey')).toBeInTheDocument();

    // Selecting a sentence from another tab should take you to where it lives.
    await user.click(screen.getByRole('button', { name: 's3' }));
    expect(screen.getByRole('tab', { name: /marked-up essay/i })).toHaveAttribute(
      'aria-selected',
      'true',
    );
  });

  it('supports keyboard navigation across tabs', async () => {
    const user = userEvent.setup();
    renderResults();

    const first = screen.getByRole('tab', { name: /marked-up essay/i });
    first.focus();
    await user.keyboard('{ArrowRight}');
    // Radix moves focus with the arrow keys (roving tabindex).
    expect(screen.getByRole('tab', { name: /rhythm & structure/i })).toHaveFocus();
  });

  it('renders caveat warnings when the analysis carries them', () => {
    renderResults({
      result: { ...SAMPLE_ANALYSIS, warnings: ['Trained on the offline bootstrap corpus.'] },
    });
    expect(screen.getByText(/caveats for this analysis/i)).toBeInTheDocument();
    expect(screen.getByText(/offline bootstrap corpus/i)).toBeInTheDocument();
  });

  it('offers a route back and a way to start over', async () => {
    const user = userEvent.setup();
    const { props } = renderResults();

    await user.click(screen.getByRole('button', { name: /analyse another essay/i }));
    expect(props.onNavigate).toHaveBeenCalledWith('analyse');

    await user.click(screen.getByRole('button', { name: /start over/i }));
    expect(props.onClear).toHaveBeenCalled();
  });
});
