/** Formatting and label helpers shared across components. */

import type {
  Classification,
  ParagraphClassification,
  SentenceClassification,
} from '@/types/api';

export const CLASS_LABELS: Record<string, string> = {
  human: 'Likely human-written',
  ai_generated: 'Likely AI-generated',
  ai_polished: 'Potentially AI-polished',
  insufficient_evidence: 'Insufficient evidence',
};

export const CLASS_SHORT: Record<string, string> = {
  human: 'Human',
  ai_generated: 'AI-generated',
  ai_polished: 'AI-polished',
  insufficient_evidence: 'Inconclusive',
};

export const SENTENCE_LABELS: Record<SentenceClassification, string> = {
  likely_human: 'Reads as human-written',
  uncertain: 'Uncertain',
  possibly_ai_assisted: 'Possibly AI-assisted',
  likely_ai_assisted: 'Likely AI-assisted',
  unavailable: 'Not scored',
};

export const PARAGRAPH_LABELS: Record<ParagraphClassification, string> = {
  likely_human: 'Reads as human-written',
  uncertain: 'Uncertain',
  contains_flagged_sentence: 'Contains a flagged sentence',
  likely_ai_assisted: 'Likely AI-assisted',
  unavailable: 'Not scored',
};

/** CSS custom-property colour token for a classification. */
export function classColour(value: Classification | string): string {
  switch (value) {
    case 'human':
    case 'likely_human':
      return 'var(--human)';
    case 'ai_generated':
    case 'likely_ai_assisted':
      return 'var(--likely)';
    case 'ai_polished':
    case 'possibly_ai_assisted':
    case 'contains_flagged_sentence':
      return 'var(--possible)';
    default:
      return 'var(--uncertain)';
  }
}

export function classSoftColour(value: Classification | string): string {
  switch (value) {
    case 'human':
    case 'likely_human':
      return 'var(--human-soft)';
    case 'ai_generated':
    case 'likely_ai_assisted':
      return 'var(--likely-soft)';
    case 'ai_polished':
    case 'possibly_ai_assisted':
    case 'contains_flagged_sentence':
      return 'var(--possible-soft)';
    default:
      return 'var(--uncertain-soft)';
  }
}

/**
 * Score → colour for the sentence and paragraph bars.
 * Deliberately banded rather than a continuous gradient: a smooth ramp implies
 * the score is precise to the pixel, which it is not.
 */
export function scoreColour(score: number | null): string {
  if (score === null) return 'var(--uncertain)';
  if (score >= 0.75) return 'var(--likely)';
  if (score >= 0.6) return 'var(--possible)';
  if (score >= 0.4) return 'var(--uncertain)';
  return 'var(--human)';
}

export function percent(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return `${(value * 100).toFixed(digits)}%`;
}

export function fixed(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return value.toFixed(digits);
}

export function countWords(text: string): number {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

export function countParagraphs(text: string): number {
  return text.split(/\n\s*\n/).filter((block) => block.trim().length > 0).length;
}

export function countSentences(text: string): number {
  const matches = text.match(/[^.!?]+[.!?]+/g);
  const trailing = /[^.!?\s][^.!?]*$/.test(text.trim()) ? 1 : 0;
  return (matches?.length ?? 0) + trailing;
}

/** Turn a feature name like `agg_mean_lm_frac_top1` into readable words. */
export function humaniseFeature(name: string): string {
  const expansions: Array<[RegExp, string]> = [
    [/^agg_mean_/, 'mean '],
    [/^agg_std_/, 'variation in '],
    [/^agg_min_/, 'minimum '],
    [/^agg_max_/, 'maximum '],
    [/^agg_p25_/, '25th pct '],
    [/^agg_p75_/, '75th pct '],
    [/^whole_/, 'whole-essay '],
  ];
  let label = name;
  let prefix = '';
  for (const [pattern, replacement] of expansions) {
    if (pattern.test(label)) {
      prefix = replacement;
      label = label.replace(pattern, '');
      break;
    }
  }
  const glossary: Record<string, string> = {
    lm: 'token',
    sty: '',
    syn: 'syntax',
    bur: 'rhythm',
    rep: 'repetition',
    ctx: 'vs essay',
    cor: 'corpus',
    doc: 'document',
    shift: 'style shift',
  };
  const parts = label.split('_');
  const group = parts[0] ?? '';
  const rest = parts.slice(1).join(' ');
  const groupWord = glossary[group] ?? group;
  return `${prefix}${groupWord ? `${groupWord} ` : ''}${rest}`.replace(/\s+/g, ' ').trim();
}

export const GROUP_LABELS: Record<string, string> = {
  lm: 'Language-model probability',
  stylometric: 'Stylometry',
  syntactic: 'Syntax',
  burstiness: 'Sentence rhythm',
  repetition: 'Repetition',
  structural: 'Document structure',
  style_shift: 'Within-essay style shift',
  corpus: 'Reference-corpus similarity',
};

export function relativeTime(iso: string | null | undefined): string {
  if (!iso) return '—';
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return '—';
  const seconds = Math.round((Date.now() - then) / 1000);
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}
