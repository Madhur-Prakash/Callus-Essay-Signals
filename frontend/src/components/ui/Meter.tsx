import type { ReactNode } from 'react';

import { cn } from '@/lib/cn';

interface MeterProps {
  label: string;
  /** 0–1. Drives how many blocks light up. */
  value: number;
  /** Word shown at the right of the label row — "strong", "weak", "typical". */
  level?: string;
  colour?: string;
  blocks?: number;
  /** Small print under the bar: the measurement behind the level. */
  detail?: ReactNode;
  /** The two ends of the foot row: the reading, and what it is measured against. */
  scale?: [ReactNode, ReactNode];
  /** Overrides the generated screen-reader label when the caller has better words. */
  ariaLabel?: string;
  className?: string;
}

/**
 * A segmented level meter.
 *
 * Segments rather than a continuous bar, deliberately. A smooth fill invites the
 * reader to compare two bars at a glance and conclude "this one is 12% stronger",
 * which the underlying signal does not support. Discrete blocks read as a band —
 * which is the honest resolution of these measurements.
 */
export function Meter({
  label,
  value,
  level,
  colour = 'var(--accent)',
  blocks = 5,
  detail,
  scale,
  ariaLabel,
  className,
}: MeterProps) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  const lit = Math.round(clamped * blocks);

  return (
    <div className={cn('meter', className)}>
      <div className="meter__row">
        <span className="meter__label">{label}</span>
        {level && (
          <span className="meter__level" style={{ color: colour }}>
            {level}
          </span>
        )}
      </div>

      <div
        className="meter__blocks"
        role="meter"
        aria-valuenow={lit}
        aria-valuemin={0}
        aria-valuemax={blocks}
        aria-label={ariaLabel ?? `${label}${level ? `: ${level}` : ''}`}
      >
        {Array.from({ length: blocks }, (_, index) => (
          <span
            key={index}
            className={cn('meter__block', index < lit && 'meter__block--on')}
            style={
              index < lit
                ? ({ '--level-color': colour, transitionDelay: `${index * 45}ms` } as React.CSSProperties)
                : undefined
            }
          />
        ))}
      </div>

      {scale && (
        <div className="meter__foot">
          <span>{scale[0]}</span>
          <span>{scale[1]}</span>
        </div>
      )}
      {detail && <p className="meter__detail">{detail}</p>}
    </div>
  );
}
