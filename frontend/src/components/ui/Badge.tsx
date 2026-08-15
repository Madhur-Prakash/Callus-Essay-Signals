import { cva, type VariantProps } from 'class-variance-authority';
import type { HTMLAttributes } from 'react';

import { cn } from '@/lib/cn';

/**
 * Tones are named for the verdict they carry, not the colour they paint. A badge
 * reading "likely" is the same amber as the sentence underline and the paragraph
 * rule, because in this product colour *is* the claim - using it decoratively
 * would teach the reader to ignore it in the one place it matters.
 */
const badge = cva('tag', {
  variants: {
    tone: {
      neutral: '',
      human: 'border-human/40 bg-human-soft text-human',
      uncertain: 'border-uncertain/40 bg-uncertain-soft text-uncertain',
      possible: 'border-possible/40 bg-possible-soft text-possible',
      likely: 'border-likely/40 bg-likely-soft text-likely',
      accent: 'border-accent/40 bg-accent-soft text-accent',
    },
    mono: { true: 'font-mono tabular-nums', false: '' },
  },
  defaultVariants: { tone: 'neutral', mono: false },
});

export interface BadgeProps
  extends HTMLAttributes<HTMLSpanElement>,
    VariantProps<typeof badge> {}

export function Badge({ className, tone, mono, ...props }: BadgeProps) {
  return <span className={cn(badge({ tone, mono }), className)} {...props} />;
}
