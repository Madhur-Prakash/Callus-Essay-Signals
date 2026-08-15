import { useCallback } from 'react';

import { useCountUp } from '@/hooks/useMotion';
import { cn } from '@/lib/cn';

interface StatProps {
  value: number;
  label: string;
  hint?: string;
  /** CSS colour. Semantic tokens only - a stat's colour is a claim about it. */
  colour?: string;
  /** Decimal places. Integers count up as integers, ratios keep their precision. */
  precision?: number;
  suffix?: string;
  className?: string;
}

/**
 * A counted-up number with its label.
 *
 * The formatter is shared between the animation and the static fallback, so a
 * mid-flight frame can never render a differently-formatted number than the one
 * that settles - the failure mode where "1,284" briefly reads "1284".
 */
export function Stat({
  value,
  label,
  hint,
  colour,
  precision = 0,
  suffix = '',
  className,
}: StatProps) {
  const format = useCallback(
    (n: number) =>
      `${precision > 0 ? n.toFixed(precision) : Math.round(n).toLocaleString()}${suffix}`,
    [precision, suffix],
  );
  const ref = useCountUp(value, format);

  return (
    <div className={cn('stat', className)}>
      <div
        className="stat__value"
        style={colour ? { color: colour } : undefined}
        ref={ref as React.RefObject<HTMLDivElement>}
      >
        {format(value)}
      </div>
      <div className="stat__label">{label}</div>
      {hint && <div className="stat__hint">{hint}</div>}
    </div>
  );
}
