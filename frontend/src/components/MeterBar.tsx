import { Meter as MeterPrimitive } from '@/components/ui';
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
 * One row of evidence, from the API's `Meter` shape.
 *
 * A thin domain adapter over the generic `Meter` primitive: it knows what a
 * percentile against the human corpus means and which colour a strength maps to,
 * and the primitive knows how to draw a level. Keeping the split there is what
 * lets the meter be reused for anything else without dragging essay concepts
 * into the interface kit.
 */
export function MeterBar({ meter }: { meter: Meter }) {
  const strength = Math.max(0, Math.min(1, meter.strength));
  const colour = levelColour(meter.level, meter.strength);
  const filled = Math.round(strength * BLOCKS);

  return (
    <MeterPrimitive
      label={meter.label}
      value={strength}
      // `level` is the strength of the machine-leaning signal, not the size of
      // the value. Saying so makes it unambiguous for inverted features, where a
      // low reading produces a strong signal.
      level={meter.level}
      colour={colour}
      blocks={BLOCKS}
      ariaLabel={`${meter.label}: ${meter.level}, ${filled} of ${BLOCKS}`}
      scale={[
        <>
          {meter.display || meter.value.toFixed(2)}
          {meter.available && meter.value_level ? (
            <span className="muted"> · {meter.value_level} median</span>
          ) : null}
        </>,
        meter.percentile_vs_human !== null
          ? `${Math.round(meter.percentile_vs_human)}th pct of human corpus`
          : meter.reference,
      ]}
      detail={meter.detail || undefined}
    />
  );
}
