import type { Meter } from '@/types/api';

const BLOCKS = 10;

function levelColour(level: string, strength: number): string {
  if (level === 'not comparable') return 'var(--uncertain)';
  if (strength >= 0.85) return 'var(--likely)';
  if (strength >= 0.7) return 'var(--possible)';
  if (strength >= 0.35) return 'var(--uncertain)';
  return 'var(--human)';
}

/**
 * A discrete ten-block bar rather than a continuous fill. The blocks are a
 * reminder that the underlying quantity is a percentile band, not a precise
 * reading — a smooth gradient would imply more resolution than exists.
 */
export function MeterBar({ meter }: { meter: Meter }) {
  const filled = Math.round(Math.max(0, Math.min(1, meter.strength)) * BLOCKS);
  const colour = levelColour(meter.level, meter.strength);

  return (
    <div className="meter">
      <div className="meter__row">
        <span className="meter__label">{meter.label}</span>
        {/* `level` is the strength of the machine-leaning signal, not the size of
            the value. Labelling it makes that unambiguous for inverted features
            where a low value produces a strong signal. */}
        <span className="meter__level" style={{ color: colour }} title="signal strength">
          {meter.level}
        </span>
      </div>
      <div
        className="meter__blocks"
        role="img"
        aria-label={`${meter.label}: ${meter.level}, ${filled} of ${BLOCKS}`}
        style={{ ['--level-color' as string]: colour }}
      >
        {Array.from({ length: BLOCKS }, (_, i) => (
          <span
            key={i}
            className={`meter__block ${i < filled ? 'meter__block--on' : ''}`}
          />
        ))}
      </div>
      <div className="meter__foot">
        <span>
          {meter.display || meter.value.toFixed(2)}
          {meter.available && meter.value_level ? (
            <span className="muted"> · {meter.value_level} median</span>
          ) : null}
        </span>
        <span>
          {meter.percentile_vs_human !== null
            ? `${Math.round(meter.percentile_vs_human)}th pct of human corpus`
            : meter.reference}
        </span>
      </div>
      {meter.detail && <p className="meter__detail">{meter.detail}</p>}
    </div>
  );
}
