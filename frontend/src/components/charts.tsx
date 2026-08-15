/**
 * Hand-rolled SVG charts.
 *
 * No charting library: these are simple enough that a dependency would cost more
 * in bundle size and styling fights than it saves, and the axes need to say
 * exactly what these particular measurements mean.
 */

import { CLASS_SHORT, classColour } from '@/lib/format';

const W = 300;
const H = 250;
const P = { top: 12, right: 12, bottom: 34, left: 40 };

const plotW = W - P.left - P.right;
const plotH = H - P.top - P.bottom;

function frame(xLabel: string, yLabel: string) {
  return (
    <>
      {[0, 0.25, 0.5, 0.75, 1].map((t) => (
        <g key={`g${t}`}>
          <line
            className="chart__grid"
            x1={P.left}
            x2={W - P.right}
            y1={P.top + plotH - t * plotH}
            y2={P.top + plotH - t * plotH}
          />
          <text
            className="chart__label"
            x={P.left - 5}
            y={P.top + plotH - t * plotH + 3}
            textAnchor="end"
          >
            {t.toFixed(2)}
          </text>
        </g>
      ))}
      <line
        className="chart__axis"
        x1={P.left}
        x2={W - P.right}
        y1={P.top + plotH}
        y2={P.top + plotH}
      />
      <line className="chart__axis" x1={P.left} x2={P.left} y1={P.top} y2={P.top + plotH} />
      {[0, 0.5, 1].map((t) => (
        <text
          key={`x${t}`}
          className="chart__label"
          x={P.left + t * plotW}
          y={P.top + plotH + 13}
          textAnchor="middle"
        >
          {t.toFixed(1)}
        </text>
      ))}
      <text
        className="chart__label"
        x={P.left + plotW / 2}
        y={H - 4}
        textAnchor="middle"
      >
        {xLabel}
      </text>
      <text
        className="chart__label"
        x={12}
        y={P.top + plotH / 2}
        transform={`rotate(-90 12 ${P.top + plotH / 2})`}
        textAnchor="middle"
      >
        {yLabel}
      </text>
    </>
  );
}

const sx = (x: number) => P.left + Math.max(0, Math.min(1, x)) * plotW;
const sy = (y: number) => P.top + plotH - Math.max(0, Math.min(1, y)) * plotH;

export function RocChart({
  curves,
  aucByClass,
}: {
  curves: Record<string, Array<{ fpr: number; tpr: number }>>;
  aucByClass?: Record<string, number | null>;
}) {
  const entries = Object.entries(curves);
  if (entries.length === 0) {
    return <p className="small muted">No ROC curve available for this split.</p>;
  }
  return (
    <div>
      <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img" aria-label="ROC curves">
        {frame('false positive rate', 'true positive rate')}
        <line
          x1={sx(0)}
          y1={sy(0)}
          x2={sx(1)}
          y2={sy(1)}
          stroke="var(--ink-faint)"
          strokeDasharray="3 3"
          strokeWidth={1}
        />
        {entries.map(([label, points]) => (
          <polyline
            key={label}
            points={points.map((p) => `${sx(p.fpr)},${sy(p.tpr)}`).join(' ')}
            fill="none"
            stroke={classColour(label)}
            strokeWidth={1.8}
            strokeLinejoin="round"
          />
        ))}
      </svg>
      <div className="chart-legend">
        {entries.map(([label]) => (
          <span key={label} className="legend__item">
            <span
              className="group-legend__dot"
              style={{ background: classColour(label) }}
              aria-hidden="true"
            />
            {CLASS_SHORT[label] ?? label}
            {aucByClass?.[label] != null && (
              <span className="mono"> AUC {aucByClass[label]!.toFixed(3)}</span>
            )}
          </span>
        ))}
      </div>
    </div>
  );
}

export function PrChart({
  curves,
}: {
  curves: Record<
    string,
    { average_precision: number; points: Array<{ recall: number; precision: number }> }
  >;
}) {
  const entries = Object.entries(curves);
  if (entries.length === 0) {
    return <p className="small muted">No precision-recall curve available.</p>;
  }
  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Precision-recall curves"
      >
        {frame('recall', 'precision')}
        {entries.map(([label, curve]) => (
          <polyline
            key={label}
            points={curve.points.map((p) => `${sx(p.recall)},${sy(p.precision)}`).join(' ')}
            fill="none"
            stroke={classColour(label)}
            strokeWidth={1.8}
            strokeLinejoin="round"
          />
        ))}
      </svg>
      <div className="chart-legend">
        {entries.map(([label, curve]) => (
          <span key={label} className="legend__item">
            <span
              className="group-legend__dot"
              style={{ background: classColour(label) }}
              aria-hidden="true"
            />
            {CLASS_SHORT[label] ?? label}
            <span className="mono"> AP {curve.average_precision.toFixed(3)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}

export function CalibrationChart({
  points,
  ece,
}: {
  points: Array<{
    mean_confidence: number;
    observed_accuracy: number;
    count: number;
  }>;
  ece?: number;
}) {
  if (!points || points.length === 0) {
    return <p className="small muted">No calibration data available.</p>;
  }
  const maxCount = Math.max(...points.map((p) => p.count), 1);
  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Calibration reliability diagram"
      >
        {frame('mean predicted confidence', 'observed accuracy')}
        <line
          x1={sx(0)}
          y1={sy(0)}
          x2={sx(1)}
          y2={sy(1)}
          stroke="var(--ink-faint)"
          strokeDasharray="3 3"
          strokeWidth={1}
        />
        <polyline
          points={points
            .map((p) => `${sx(p.mean_confidence)},${sy(p.observed_accuracy)}`)
            .join(' ')}
          fill="none"
          stroke="var(--accent)"
          strokeWidth={1.8}
        />
        {points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.mean_confidence)}
            cy={sy(p.observed_accuracy)}
            r={2 + (p.count / maxCount) * 4}
            fill="var(--accent)"
          >
            <title>{`${p.count} documents, confidence ${p.mean_confidence.toFixed(2)}, accuracy ${p.observed_accuracy.toFixed(2)}`}</title>
          </circle>
        ))}
      </svg>
      <div className="chart-legend">
        <span>dashed line = perfect calibration</span>
        <span>marker size = documents in bin</span>
        {ece !== undefined && (
          <span>
            ECE <span className="mono">{ece.toFixed(3)}</span>
          </span>
        )}
      </div>
    </div>
  );
}
