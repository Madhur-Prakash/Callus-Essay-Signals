import { useDrawPath } from '@/hooks/useMotion';
import { cn } from '@/lib/cn';

interface SparklineProps {
  values: number[];
  /** Draw a bar under each point. */
  bars?: boolean;
  /** Draw the connecting trace. */
  line?: boolean;
  /** Draw a dashed rule at the mean - the reference the trace varies around. */
  mean?: boolean;
  /** Animate the trace on mount, and re-run when these change. */
  draw?: boolean;
  deps?: unknown[];
  width?: number;
  height?: number;
  className?: string;
  /** Screen-reader text. Without it the chart is `aria-hidden` decoration. */
  ariaLabel?: string;
}

/**
 * A micro line-and-bar chart.
 *
 * The product's visual motif: uneven sentence lengths varying around their own
 * mean, which is one of the things the detector actually measures. It is drawn
 * from real numbers passed in rather than a decorative squiggle, so it never
 * implies data that does not exist.
 */
export function Sparkline({
  values,
  bars = true,
  line = true,
  mean = true,
  draw = true,
  deps = [],
  width = 460,
  height = 46,
  className,
  ariaLabel,
}: SparklineProps) {
  const lineRef = useDrawPath<SVGPolylineElement>(draw ? [values.length, ...deps] : []);

  const pad = 8;
  const max = Math.max(...values, 1);
  const step = values.length > 1 ? (width - pad * 2) / (values.length - 1) : 0;
  const baseline = height - 4;
  const y = (n: number) => baseline - (n / max) * (height - 14);
  const points = values.map((n, i) => `${pad + i * step},${y(n)}`).join(' ');
  const average = values.reduce((sum, n) => sum + n, 0) / (values.length || 1);

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      className={cn('sparkline', className)}
      role={ariaLabel ? 'img' : undefined}
      aria-label={ariaLabel}
      aria-hidden={ariaLabel ? undefined : true}
    >
      {bars &&
        values.map((n, i) => (
          <rect
            key={i}
            className="sparkline__bar"
            x={pad + i * step - 1.5}
            y={y(n)}
            width={3}
            height={Math.max(0, baseline - y(n))}
            rx={1.5}
          />
        ))}

      {mean && (
        <line
          className="sparkline__mean"
          x1={pad}
          x2={width - pad}
          y1={y(average)}
          y2={y(average)}
        />
      )}

      {line && (
        <polyline ref={lineRef} className="sparkline__line" points={points} />
      )}
    </svg>
  );
}
