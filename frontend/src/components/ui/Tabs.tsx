/**
 * Tabs — a thin wrapper over Radix's tabs primitive.
 *
 * On shadcn: shadcn/ui is a Tailwind-based collection, and this project has a
 * hand-built token system rather than Tailwind. Bolting Tailwind on would mean
 * rewriting the whole design system for components that are mostly generic —
 * and the parts of this UI that matter (signal meters, sentence marks, the
 * confusion matrix) have no shadcn equivalent anyway.
 *
 * So we take what shadcn is actually built on: Radix primitives. That is where
 * the real value is — roving tabindex, arrow-key navigation, correct
 * `aria-selected`/`aria-controls` wiring, and focus management we would
 * otherwise have to hand-roll and get subtly wrong. The styling stays on our own
 * tokens.
 */

import * as RadixTabs from '@radix-ui/react-tabs';
import type { ReactNode } from 'react';

export interface TabDefinition {
  value: string;
  label: string;
  /** Optional count/badge shown after the label (e.g. "3" failures). */
  badge?: string | number;
}

interface Props {
  tabs: TabDefinition[];
  value: string;
  onValueChange: (value: string) => void;
  children: ReactNode;
  /** Accessible name for the tablist. */
  label: string;
  /** Sticks the tab bar below the masthead while scrolling a long panel. */
  sticky?: boolean;
}

export function Tabs({
  tabs,
  value,
  onValueChange,
  children,
  label,
  sticky = false,
}: Props) {
  return (
    <RadixTabs.Root value={value} onValueChange={onValueChange} activationMode="manual">
      <RadixTabs.List
        className={`tabs__list${sticky ? ' tabs__list--sticky' : ''}`}
        aria-label={label}
      >
        {tabs.map((tab) => (
          <RadixTabs.Trigger key={tab.value} value={tab.value} className="tabs__trigger">
            <span>{tab.label}</span>
            {tab.badge !== undefined && <span className="tabs__badge">{tab.badge}</span>}
          </RadixTabs.Trigger>
        ))}
      </RadixTabs.List>
      {children}
    </RadixTabs.Root>
  );
}

export function TabPanel({ value, children }: { value: string; children: ReactNode }) {
  return (
    <RadixTabs.Content value={value} className="tabs__panel" tabIndex={-1}>
      {children}
    </RadixTabs.Content>
  );
}
