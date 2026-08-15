import { useId } from 'react';

import { useArcSweep, useCountUp } from '@/hooks/useMotion';
import { cn } from '@/lib/cn';

interface GaugeProps {
  /** 0–1. Values outside the range are clamped rather than drawn off the arc. */
  value: number;
  /** CSS colour for the filled arc. Pass a semantic token, not a raw hue. */
  colour?: string;
  /** Large text in the middle. Omit to show the value as a percentage. */
  label?: string;
  caption?: string;
  size?: number;
  className?: string;
  /** Announced to screen readers in place of the SVG. */
  ariaLabel?: string;
}

const TRACK_GAP = 0.22; // fraction of the circle left open at the bottom

/**
 * A radial gauge that sweeps to its value.
 *
 * Built on a `<circle>` rather than an arc path so the sweep is a plain
 * dashoffset animation with no trigonometry, and so `getTotalLength()` reports
 * the real circumference at any size. The open bottom is a rotated dash gap,
 * which keeps the geometry to two numbers.
 */
export function Gauge({
  value,
  colour = 'var(--accent)',
  label,
  caption,
  size = 176,
  className,
  ariaLabel,
}: GaugeProps) {
  const clamped = Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
  const gradientId = useId();

  const stroke = size * 0.075;
  const radius = (size - stroke) / 2;
  const centre = size / 2;
  const circumference = 2 * Math.PI * radius;
  const visible = 1 - TRACK_GAP;

  // The arc sweeps across the visible portion only, so a value of 1 fills the
  // whole track rather than running into the gap.
  const arcRef = useArcSweep(clamped * visible);

  const percent = `${Math.round(clamped * 100)}%`;
  const numberRef = useCountUp(
    clamped * 100,
    (n) => `${Math.round(n)}%`,
    1.4,
  );

  return (
    <div
      className={cn('gauge', className)}
      style={{ width: size, height: size }}
      role="img"
      aria-label={ariaLabel ?? `${label ?? percent}${caption ? `, ${caption}` : ''}`}
    >
      <svg viewBox={`0 0 ${size} ${size}`} className="gauge__svg" aria-hidden="true">
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={colour} stopOpacity="0.55" />
            <stop offset="100%" stopColor={colour} />
          </linearGradient>
        </defs>

        {/* Rotated so the dash gap sits centred at the bottom. */}
        <g transform={`rotate(${90 + (TRACK_GAP * 360) / 2} ${centre} ${centre})`}>
          <circle
            className="gauge__track"
            cx={centre}
            cy={centre}
            r={radius}
            strokeWidth={stroke}
            strokeDasharray={`${circumference * visible} ${circumference}`}
          />
          <circle
            ref={arcRef}
            className="gauge__arc"
            cx={centre}
            cy={centre}
            r={radius}
            strokeWidth={stroke}
            stroke={`url(#${gradientId})`}
            style={{ filter: `drop-shadow(0 0 10px ${colour})` }}
          />
        </g>
      </svg>

      <div className="gauge__centre">
        {label ? (
          <span className="gauge__label" style={{ color: colour }}>
            {label}
          </span>
        ) : (
          <span
            className="gauge__value"
            style={{ color: colour }}
            ref={numberRef as React.RefObject<HTMLSpanElement>}
          >
            {percent}
          </span>
        )}
        {caption && <span className="gauge__caption">{caption}</span>}
      </div>
    </div>
  );
}
