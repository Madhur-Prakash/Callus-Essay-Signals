import { useMemo } from 'react';

import { scoreColour } from '@/lib/format';
import type { RhythmPoint, SentenceResult } from '@/types/api';

interface Props {
  rhythm: RhythmPoint[];
  sentences: SentenceResult[];
  onSelectSentence: (sentenceId: number) => void;
  selectedSentenceId: number | null;
}

const WIDTH = 720;
const HEIGHT = 190;
const PAD = { top: 14, right: 12, bottom: 26, left: 34 };

/**
 * Sentence-length bars with the essay mean drawn across them.
 *
 * This is the burstiness picture: a human draft usually shows tall/short
 * alternation around the mean, while regularised prose sits flat against it. The
 * bars are coloured by each sentence's own machine-likeness score so the shape and
 * the verdict can be read together — and disagreements between them are visible
 * rather than smoothed away.
 */
export function RhythmChart({
  rhythm,
  sentences,
  onSelectSentence,
  selectedSentenceId,
}: Props) {
  const scoreById = useMemo(
    () => new Map(sentences.map((s) => [s.sentence_id, s.score])),
    [sentences],
  );

  const stats = useMemo(() => {
    const lengths = rhythm.map((point) => point.words);
    const max = Math.max(...lengths, 1);
    const mean = lengths.reduce((sum, n) => sum + n, 0) / (lengths.length || 1);
    const sd = Math.sqrt(
      lengths.reduce((sum, n) => sum + (n - mean) ** 2, 0) / (lengths.length || 1),
    );
    return { max, mean, sd };
  }, [rhythm]);

  if (rhythm.length === 0) {
    return <p className="small muted">No sentence rhythm data available.</p>;
  }

  const plotWidth = WIDTH - PAD.left - PAD.right;
  const plotHeight = HEIGHT - PAD.top - PAD.bottom;
  const barSpace = plotWidth / rhythm.length;
  const barWidth = Math.max(2, barSpace * 0.66);
  const yFor = (words: number) => PAD.top + plotHeight - (words / stats.max) * plotHeight;

  const ticks = [0, Math.round(stats.max / 2), Math.round(stats.max)];

  return (
    <div>
      <svg
        className="chart"
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        role="img"
        aria-label={`Sentence length across ${rhythm.length} sentences. Mean ${stats.mean.toFixed(1)} words, standard deviation ${stats.sd.toFixed(1)}.`}
        preserveAspectRatio="xMidYMid meet"
      >
        {ticks.map((tick) => (
          <g key={tick}>
            <line
              className="chart__grid"
              x1={PAD.left}
              x2={WIDTH - PAD.right}
              y1={yFor(tick)}
              y2={yFor(tick)}
            />
            <text className="chart__label" x={PAD.left - 6} y={yFor(tick) + 3} textAnchor="end">
              {tick}
            </text>
          </g>
        ))}

        <line
          className="chart__mean"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={yFor(stats.mean)}
          y2={yFor(stats.mean)}
        />
        <text
          className="chart__label"
          x={WIDTH - PAD.right}
          y={yFor(stats.mean) - 4}
          textAnchor="end"
          style={{ fill: 'var(--accent)' }}
        >
          mean {stats.mean.toFixed(1)}
        </text>

        {rhythm.map((point, index) => {
          const score = scoreById.get(point.index) ?? null;
          const x = PAD.left + index * barSpace + (barSpace - barWidth) / 2;
          const y = yFor(point.words);
          const isSelected = selectedSentenceId === point.index;
          return (
            <rect
              key={point.index}
              className="chart__bar"
              x={x}
              y={y}
              width={barWidth}
              height={Math.max(1, PAD.top + plotHeight - y)}
              rx={1.5}
              fill={scoreColour(score)}
              opacity={isSelected ? 1 : 0.82}
              stroke={isSelected ? 'var(--ink)' : 'none'}
              strokeWidth={isSelected ? 1.2 : 0}
              onClick={() => onSelectSentence(point.index)}
            >
              <title>
                {`Sentence ${point.index + 1}: ${point.words} words, perplexity ${point.perplexity.toFixed(1)}`}
                {score !== null ? `, machine-likeness ${(score * 100).toFixed(0)}%` : ''}
              </title>
            </rect>
          );
        })}

        <line
          className="chart__axis"
          x1={PAD.left}
          x2={WIDTH - PAD.right}
          y1={PAD.top + plotHeight}
          y2={PAD.top + plotHeight}
        />
        <text
          className="chart__label"
          x={PAD.left + plotWidth / 2}
          y={HEIGHT - 6}
          textAnchor="middle"
        >
          sentence order →
        </text>
        <text
          className="chart__label"
          x={PAD.left - 26}
          y={PAD.top + plotHeight / 2}
          transform={`rotate(-90 ${PAD.left - 26} ${PAD.top + plotHeight / 2})`}
          textAnchor="middle"
        >
          words
        </text>
      </svg>

      <div className="chart-legend">
        <span>
          mean <span className="mono">{stats.mean.toFixed(1)}</span> words
        </span>
        <span>
          std dev <span className="mono">{stats.sd.toFixed(1)}</span>
        </span>
        <span>
          coefficient of variation{' '}
          <span className="mono">{(stats.sd / (stats.mean || 1)).toFixed(2)}</span>
        </span>
        <span className="muted">bar colour = that sentence's machine-likeness</span>
      </div>
    </div>
  );
}
