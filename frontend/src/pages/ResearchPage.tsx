import { useEffect, useState } from 'react';

import { ApiError, fetchEvaluation } from '@/api/client';
import { Banner } from '@/components/Banner';
import { CalibrationChart, PrChart, RocChart } from '@/components/charts';
import { Badge, Section, TabPanel, Tabs } from '@/components/ui';
import { useScrollReveal } from '@/hooks/useMotion';
import { cn } from '@/lib/cn';
import {
  CLASS_SHORT,
  GROUP_LABELS,
  classColour,
  fixed,
  humaniseFeature,
  percent,
} from '@/lib/format';
import type {
  ConfidentlyWrongCase,
  DatasetCard,
  EvaluationBundle,
  EvaluationReport,
  FailureReport,
} from '@/types/api';

const GROUP_COLOURS: Record<string, string> = {
  lm: '#1f4e79',
  stylometric: '#9a6614',
  syntactic: '#2f6f4f',
  burstiness: '#7a4a8f',
  repetition: '#a33a24',
  structural: '#4a463d',
  style_shift: '#0f766e',
  corpus: '#b45309',
};

/** Tab ids are part of the URL, so a specific finding can be linked to. */
const TAB_IDS = [
  'overview',
  'curves',
  'models',
  'generalisation',
  'bias',
  'failures',
  'dataset',
] as const;

function tabFromHash(): string {
  const query = window.location.hash.split('?')[1] ?? '';
  const requested = new URLSearchParams(query).get('tab') ?? '';
  return (TAB_IDS as readonly string[]).includes(requested) ? requested : 'overview';
}

export function ResearchPage() {
  const [bundle, setBundle] = useState<EvaluationBundle | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [tab, setTab] = useState<string>(tabFromHash);
  // Cards reveal as they scroll in. Re-run when the bundle lands or the tab
  // changes, since each panel's cards do not exist in the DOM until then.
  const revealRef = useScrollReveal('.card, .failure', [bundle, tab]);

  // Reflect the tab in the URL so a colleague can be sent straight to the bias
  // analysis rather than "scroll down about two thirds".
  const selectTab = (next: string) => {
    setTab(next);
    const base = window.location.hash.split('?')[0] || '#/research';
    window.history.replaceState(null, '', next === 'overview' ? base : `${base}?tab=${next}`);
  };

  useEffect(() => {
    fetchEvaluation()
      .then(setBundle)
      .catch((caught) =>
        setError(
          caught instanceof ApiError
            ? caught
            : new ApiError('Could not load the evaluation report.', 'internal_error', 500),
        ),
      );
  }, []);

  if (error) {
    return (
      <Banner tone="danger" title="Could not load the evaluation">
        {error.message}
      </Banner>
    );
  }

  if (!bundle) {
    return <p className="muted">Loading the evaluation report…</p>;
  }

  if (!bundle.available || !bundle.report) {
    return (
      <Banner tone="warning" title="No evaluation report yet">
        {bundle.message ?? 'Run the evaluation pipeline to populate this page.'}
        <pre className="mono mb-0 mt-3 whitespace-pre-wrap text-[0.78rem]">
          uv run python -m ml.evaluation.evaluate{'\n'}
          uv run python -m ml.evaluation.find_failures
        </pre>
      </Banner>
    );
  }

  const report = bundle.report;

  return (
    <div className="research-grid" ref={revealRef as React.RefObject<HTMLDivElement>}>
      <header className="research-head">
        <h1>Evaluation</h1>
        <p className="muted">
          Held-out metrics for the shipped detector, plus the generalisation, bias and failure
          analyses. Everything on this page is read from the artifacts written by{' '}
          <code>ml.evaluation.evaluate</code> and <code>ml.evaluation.find_failures</code> — no
          number is computed in the browser.
        </p>
        <div className="chips mt-3">
          <Badge>split: {report.split}</Badge>
          <Badge>{report.overall.n_samples} documents</Badge>
          <Badge tone={report.data_regime === 'bootstrap' ? 'possible' : 'neutral'}>
            regime: {report.data_regime ?? 'unknown'}
          </Badge>
          <Badge mono>
            model v{String((report.model as Record<string, unknown>).model_version ?? '—')}
          </Badge>
          <Badge>
            calibration: {String((report.model as Record<string, unknown>).calibration ?? 'none')}
          </Badge>
        </div>
      </header>

      {report.data_regime === 'bootstrap' && report.data_regime_note && (
        <Banner tone="warning" title="Read this before reading the numbers">
          {report.data_regime_note}
        </Banner>
      )}

      <Tabs
        tabs={[
          { value: 'overview', label: 'Overview' },
          { value: 'curves', label: 'Curves' },
          { value: 'models', label: 'Model comparison' },
          { value: 'generalisation', label: 'Generalisation' },
          { value: 'bias', label: 'Bias' },
          {
            value: 'failures',
            label: 'Failures',
            badge: bundle.failures?.confidently_wrong.length ?? 0,
          },
          { value: 'dataset', label: 'Dataset' },
        ]}
        value={tab}
        onValueChange={selectTab}
        label="Evaluation sections"
        sticky
      >
        <TabPanel value="overview">
          <div className="research-grid">
            <Interpretation lines={report.interpretation} />
            <Overall report={report} />
            <div className="two-col">
              <Section title="Confusion matrix" bodyClassName="scroll-x">
                <ConfusionMatrix report={report} />
              </Section>

              <Section
                title="Per-class metrics"
                note="False-positive rate on the human class is the number that matters most."
                bodyClassName="scroll-x"
              >
                <PerClassTable report={report} />
              </Section>
            </div>
          </div>
        </TabPanel>

        <TabPanel value="curves">
          <div className="two-col">
            <Section title="ROC curves (one-vs-rest)">
              <RocChart curves={report.curves.roc} aucByClass={report.overall.roc_auc_per_class} />
            </Section>
            <Section title="Precision-recall curves">
              <PrChart curves={report.curves.precision_recall} />
            </Section>
            <Section
              title="Calibration"
              note="Are the confidence values honest? Points below the diagonal mean over-confidence."
            >
              <CalibrationChart
                points={report.overall.reliability_curve ?? []}
                ece={report.overall.expected_calibration_error}
              />
            </Section>
          </div>
        </TabPanel>

        <TabPanel value="models">
          <div className="research-grid">
            <ModelComparison report={report} />
            <FeatureImportance report={report} />
          </div>
        </TabPanel>

        <TabPanel value="generalisation">
          <Generalisation report={report} />
        </TabPanel>

        <TabPanel value="bias">
          <BiasSection report={report} />
        </TabPanel>

        <TabPanel value="failures">
          {bundle.failures ? (
            <Failures failures={bundle.failures} />
          ) : (
            <Banner tone="warning" title="No failure analysis yet">
              Run <code>uv run python -m ml.evaluation.find_failures</code>.
            </Banner>
          )}
        </TabPanel>

        <TabPanel value="dataset">
          {bundle.dataset ? (
            <Dataset dataset={bundle.dataset} />
          ) : (
            <Banner tone="warning" title="No dataset card yet">
              Run <code>uv run python -m ml.training.prepare_dataset</code>.
            </Banner>
          )}
        </TabPanel>
      </Tabs>
    </div>
  );
}

function Interpretation({ lines }: { lines: string[] }) {
  return (
    <Section
      title="What these numbers mean"
      note="Generated by the evaluation script from the measured results, including the unflattering ones."
    >
      <ul className="interpretation">
        {lines.map((line, index) => {
          const isCritical = line.startsWith('FALSE POSITIVES');
          const isRegime = line.startsWith('REGIME WARNING');
          return (
            <li key={index} className={isCritical ? 'critical' : isRegime ? 'regime' : undefined}>
              {line}
            </li>
          );
        })}
      </ul>
    </Section>
  );
}

function Overall({ report }: { report: EvaluationReport }) {
  const o = report.overall;
  const metrics: Array<[string, string, string]> = [
    ['Accuracy', percent(o.accuracy, 1), 'all classes'],
    ['Balanced accuracy', percent(o.balanced_accuracy, 1), 'mean per-class recall'],
    ['Macro F1', fixed(o.macro_f1, 3), 'unweighted over 3 classes'],
    ['Weighted F1', fixed(o.weighted_f1, 3), 'by support'],
    ["Cohen's κ", fixed(o.cohen_kappa, 3), 'vs chance agreement'],
    ['MCC', fixed(o.matthews_corrcoef, 3), 'balanced correlation'],
    ['ROC-AUC (macro)', fixed(o.roc_auc_ovr_macro ?? null, 3), 'one-vs-rest'],
    ['Log loss', fixed(o.log_loss ?? null, 3), 'lower is better'],
    ['ECE', fixed(o.expected_calibration_error ?? null, 3), 'calibration error'],
  ];
  return (
    <Section
      title="Overall metrics"
      note={`Held-out ${report.split} split, ${o.n_samples} documents.`}
    >
      <div className="metric-row">
        {metrics.map(([label, value, hint]) => (
          <div className="metric" key={label}>
            <div className="metric__value">{value}</div>
            <div className="metric__label">{label}</div>
            <div className="metric__hint">{hint}</div>
          </div>
        ))}
      </div>
    </Section>
  );
}

function ConfusionMatrix({ report }: { report: EvaluationReport }) {
  const { labels, matrix, row_normalised: normalised } = report.overall.confusion_matrix;
  return (
    <>
      <table className="confusion">
        <thead>
          <tr>
            <th />
            <th colSpan={labels.length} className="text-center">
              predicted
            </th>
          </tr>
          <tr>
            <th className="row-head">actual</th>
            {labels.map((label) => (
              <th key={label}>{CLASS_SHORT[label] ?? label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={labels[i]}>
              <th className="row-head">{CLASS_SHORT[labels[i] ?? ''] ?? labels[i]}</th>
              {row.map((count, j) => {
                const share = normalised[i]?.[j] ?? 0;
                const isDiag = i === j;
                return (
                  <td
                    key={j}
                    className={isDiag ? 'diag' : undefined}
                    style={{
                      background: isDiag
                        ? `color-mix(in srgb, var(--human) ${Math.round(share * 55)}%, var(--surface))`
                        : `color-mix(in srgb, var(--likely) ${Math.round(share * 55)}%, var(--surface))`,
                    }}
                  >
                    <span className="confusion__count">{count}</span>
                    <span className="confusion__pct">{percent(share, 0)}</span>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="tiny muted mb-0 mt-2.5">
        Rows are true labels, columns are predictions. Percentages are row-normalised, so the
        top-left cell reads "of the human documents, this share were correctly called human".
      </p>
    </>
  );
}

function PerClassTable({ report }: { report: EvaluationReport }) {
  const entries = Object.entries(report.overall.per_class);
  return (
    <table className="data">
      <thead>
        <tr>
          <th>Class</th>
          <th className="num">n</th>
          <th className="num">Precision</th>
          <th className="num">Recall</th>
          <th className="num">F1</th>
          <th className="num">FP rate</th>
          <th className="num">FN rate</th>
          <th className="num">AUC</th>
        </tr>
      </thead>
      <tbody>
        {entries.map(([label, m]) => (
          <tr key={label}>
            <td>
              <span
                className="group-legend__dot"
                style={{ background: classColour(label) }}
                aria-hidden="true"
              />
              {CLASS_SHORT[label] ?? label}
            </td>
            <td className="num">{m.support}</td>
            <td className="num">{fixed(m.precision, 3)}</td>
            {/* Recall below 0.6 is the failure this page exists to surface, so it
                is marked in the table rather than left for the reader to spot. */}
            <td className={cn('num', m.recall < 0.6 && 'font-bold text-likely')}>
              {fixed(m.recall, 3)}
            </td>
            <td className="num">{fixed(m.f1, 3)}</td>
            <td className="num">{fixed(m.false_positive_rate, 3)}</td>
            <td className="num">{fixed(m.false_negative_rate, 3)}</td>
            <td className="num">
              {fixed(report.overall.roc_auc_per_class?.[label] ?? null, 3)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ModelComparison({ report }: { report: EvaluationReport }) {
  const comparison = report.model_comparison;
  const best = Math.max(...comparison.models.map((m) => m.macro_f1), 0.0001);
  return (
    <Section
      title="Does the hybrid approach actually help?"
      note={comparison.protocol}
      bodyClassName="scroll-x"
    >
      <table className="data">
          <thead>
            <tr>
              <th>Feature set</th>
              <th className="num">Features</th>
              <th className="num">Macro F1</th>
              <th />
              <th className="num">Accuracy</th>
              <th className="num">Human recall</th>
              <th className="num">AUC</th>
            </tr>
          </thead>
          <tbody>
            {comparison.models.map((model) => (
              <tr key={model.feature_set}>
                <td>
                  <strong>{model.feature_set}</strong>
                  <div className="tiny muted">
                    {model.feature_groups.map((g) => GROUP_LABELS[g] ?? g).join(' · ')}
                  </div>
                </td>
                <td className="num">{model.n_features}</td>
                <td className="num">{fixed(model.macro_f1, 3)}</td>
                <td className="w-36">
                  <span className="importance__track block">
                    <span
                      className="importance__fill block h-full"
                      style={{
                        width: `${(model.macro_f1 / best) * 100}%`,
                        background:
                          model.feature_set === 'hybrid' ? 'var(--accent)' : 'var(--ink-faint)',
                      }}
                    />
                  </span>
                </td>
                <td className="num">{fixed(model.accuracy, 3)}</td>
                <td className="num">{fixed(model.human_recall, 3)}</td>
                <td className="num">{fixed(model.roc_auc_ovr_macro ?? null, 3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="chips mt-3">
          {Object.entries(comparison.deltas).map(([key, value]) => (
            <Badge mono key={key} tone={value !== null && value > 0 ? 'human' : 'neutral'}>
              {key.replace(/_/g, ' ')}: {value === null ? '—' : value > 0 ? `+${value}` : value}
            </Badge>
          ))}
      </div>
    </Section>
  );
}

function FeatureImportance({ report }: { report: EvaluationReport }) {
  const items = report.feature_importance.slice(0, 15);
  const max = Math.max(...items.map((i) => i.importance), 0.0001);
  const groups = Array.from(new Set(items.map((i) => i.group).filter(Boolean))) as string[];

  return (
    <Section
      title="Most influential features"
      note="Permutation importance on the validation split — how much accuracy depends on each measurement. Not hardcoded; read from the trained model."
    >
      {items.length === 0 ? (
        <p className="small muted">No importance data in the report.</p>
      ) : (
        <>
          <div className="importance">
            {items.map((item, index) => (
              <div className="importance__row" key={item.feature}>
                <span className="importance__name" title={item.feature}>
                  {index + 1}. {humaniseFeature(item.feature)}
                </span>
                <span className="importance__track">
                  <span
                    className="importance__fill"
                    style={{
                      width: `${(item.importance / max) * 100}%`,
                      background: GROUP_COLOURS[item.group ?? ''] ?? 'var(--ink-faint)',
                    }}
                  />
                </span>
                <span className="importance__value">{item.importance.toFixed(4)}</span>
              </div>
            ))}
          </div>
          <div className="group-legend">
            {groups.map((group) => (
              <span key={group}>
                <span
                  className="group-legend__dot"
                  style={{ background: GROUP_COLOURS[group] ?? 'var(--ink-faint)' }}
                />
                {GROUP_LABELS[group] ?? group}
              </span>
            ))}
          </div>
        </>
      )}
    </Section>
  );
}

function Generalisation({ report }: { report: EvaluationReport }) {
  const g = report.generalisation as Record<string, Record<string, Record<string, unknown>>>;
  const sections: Array<[string, string]> = [
    ['by_length_band', 'By length band'],
    ['by_polish_transform', 'By polish transform'],
    ['by_source', 'By source'],
  ];

  return (
    <Section
      title="Generalisation slices"
      note="Slices with fewer than 8 documents are reported as too small rather than given a number."
      bodyClassName="stack stack--md"
    >
      {sections.map(([key, title]) => {
        const slice = g[key];
        if (!slice) return null;
        return (
          <div key={key}>
            <p className="subhead">{title}</p>
            <table className="data">
              <thead>
                <tr>
                  <th>Slice</th>
                  <th className="num">n</th>
                  <th className="num">Headline</th>
                  <th>Metric</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(slice).map(([name, value]) => {
                  const row = value as Record<string, unknown>;
                  const tooSmall = Boolean(row.too_small);
                  return (
                    <tr key={name}>
                      <td>{name}</td>
                      <td className="num">{String(row.n_samples ?? '—')}</td>
                      <td className="num">
                        {tooSmall
                          ? '—'
                          : fixed(Number(row.headline_value ?? row.macro_f1 ?? 0), 3)}
                      </td>
                      <td className="tiny muted">
                        {tooSmall
                          ? 'too small to report'
                          : String(row.headline_metric ?? 'macro_f1')}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        );
      })}
    </Section>
  );
}

function BiasSection({ report }: { report: EvaluationReport }) {
  const bias = report.bias;
  const groups = Object.entries(bias.groups);

  return (
    <Section title="Bias and fairness" note={bias.question} bodyClassName="stack stack--md">
      <p className="small muted m-0">Metric: {bias.metric}</p>

      <div>
          {groups.map(([name, group]) => {
            const rate = group.false_positive_rate;
            if (!rate) {
              return (
                <div className="bias-bar" key={name}>
                  <span>{name.replace(/_/g, ' ')}</span>
                  <span className="small muted">
                    {group.n_human_documents} human documents — not measurable
                  </span>
                  <span />
                </div>
              );
            }
            return (
              <div className="bias-bar" key={name}>
                <span>{name.replace(/_/g, ' ')}</span>
                <span className="bias-bar__track">
                  <span
                    className="bias-bar__interval"
                    style={{
                      left: `${rate.lower * 100}%`,
                      width: `${Math.max(1, (rate.upper - rate.lower) * 100)}%`,
                    }}
                  />
                  <span className="bias-bar__point" style={{ left: `${rate.point * 100}%` }} />
                </span>
                <span className="mono tiny">
                  {percent(rate.point, 0)} [{percent(rate.lower, 0)}–{percent(rate.upper, 0)}] n=
                  {rate.n}
                </span>
              </div>
            );
          })}
        <p className="tiny muted mt-1.5">
          Bars show the Wilson 95% interval; the tick is the point estimate.
        </p>
      </div>

      {Boolean(bias.disparity.measurable) && (
        <Banner
          tone={bias.disparity.confidence_intervals_overlap ? 'info' : 'danger'}
          title="Disparity test"
        >
          {String(bias.disparity.conclusion)}
        </Banner>
      )}
      {!bias.disparity.measurable && (
        <Banner tone="warning" title="Disparity not measurable">
          The held-out split does not contain enough human documents in each group to run the
          test. This is a gap in the evaluation, not evidence of fairness.
        </Banner>
      )}

      <Banner tone="danger" title="Severe limitation — read this">
        {bias.severe_limitation}
      </Banner>
    </Section>
  );
}

function Failures({ failures }: { failures: FailureReport }) {
  const summary = failures.summary as Record<string, unknown>;
  return (
    <Section
      title="Where the detector confidently fails"
      note={`${String(summary.n_errors ?? 0)} errors in ${String(summary.n_documents ?? 0)} documents; ${String(summary.n_confidently_wrong ?? 0)} of them above the ${String(summary.confidence_threshold ?? 0.55)} confidence threshold.`}
      bodyClassName="stack stack--md"
    >
      <div className="metric-row">
        <div className="metric">
          {/* Coloured because this is the number that should worry a reader most:
              a human being told their own writing looks machine-made. */}
          <div className="metric__value text-likely">
            {failures.false_positives_on_human_writing.count}
          </div>
          <div className="metric__label">false positives on human writing</div>
        </div>
        <div className="metric">
          <div className="metric__value">{failures.missed_machine_writing.count}</div>
          <div className="metric__label">missed machine writing</div>
        </div>
        <div className="metric">
          <div className="metric__value">{failures.ai_polished_confusions.count}</div>
          <div className="metric__label">AI-polished confusions</div>
        </div>
        <div className="metric">
          <div className="metric__value mono">
            {fixed(Number(summary.mean_confidence_when_wrong ?? 0), 2)}
          </div>
          <div className="metric__label">mean confidence when wrong</div>
          <div className="metric__hint">
            vs {fixed(Number(summary.mean_confidence_when_right ?? 0), 2)} when right
          </div>
        </div>
      </div>

      {Boolean(summary.fallback_note) && (
        <p className="tiny muted">{String(summary.fallback_note)}</p>
      )}

      {failures.confidently_wrong.map((failure) => (
        <FailureCase key={failure.record_id} failure={failure} />
      ))}
    </Section>
  );
}

function FailureCase({ failure }: { failure: ConfidentlyWrongCase }) {
  return (
    <article className="failure">
      <div className="failure__head">
        <Badge mono>#{failure.rank}</Badge>
        <span className="failure__arrow">
          <span style={{ color: classColour(failure.actual) }}>
            {CLASS_SHORT[failure.actual] ?? failure.actual}
          </span>
          {' → '}
          <span style={{ color: classColour(failure.predicted) }}>
            {CLASS_SHORT[failure.predicted] ?? failure.predicted}
          </span>
        </span>
        <Badge mono tone="likely">
          confidence {percent(failure.confidence, 0)}
        </Badge>
        <span className="spacer" />
        <span className="tiny muted mono">{failure.record_id}</span>
      </div>
      <div className="failure__body">
        {failure.excerpt && <blockquote className="failure__excerpt">{failure.excerpt}…</blockquote>}
        {failure.excerpt_withheld_reason && (
          <p className="tiny muted">{failure.excerpt_withheld_reason}</p>
        )}

        <div>
          <p className="subhead">Why the model likely failed</p>
          <ul className="statements">
            {failure.why_the_model_likely_failed.map((line, index) => (
              <li key={index}>{line}</li>
            ))}
          </ul>
        </div>

        <div>
          <p className="subhead">Dominant feature groups</p>
          <div className="chips">
            {failure.dominant_feature_groups.slice(0, 4).map((group) => (
              <Badge key={group.group}>
                <span
                  className="group-legend__dot"
                  style={{ background: GROUP_COLOURS[group.group] ?? 'var(--ink-faint)' }}
                />
                {GROUP_LABELS[group.group] ?? group.group} {percent(group.share_of_contribution, 0)}
              </Badge>
            ))}
          </div>
        </div>

        {failure.relevant_features.length > 0 && (
          <div className="scroll-x">
            <p className="subhead">Relevant features</p>
            <table className="data">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th className="num">Measured</th>
                  <th className="num">True-class range</th>
                  <th className="num">IQRs outside</th>
                </tr>
              </thead>
              <tbody>
                {failure.relevant_features.slice(0, 5).map((feature) => (
                  <tr key={feature.feature}>
                    <td className="mono text-[0.72rem]">{feature.feature}</td>
                    <td className="num">{feature.value.toPrecision(4)}</td>
                    <td className="num">
                      {feature.true_class_iqr.map((v) => Number(v).toPrecision(3)).join(' – ')}
                    </td>
                    <td className="num">
                      {feature.direction === 'above' ? '↑' : '↓'} {feature.iqr_widths_outside}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div>
          <p className="subhead">Possible improvement</p>
          {failure.possible_improvement.map((line, index) => (
            <p className="small text-ink-soft" key={index}>
              {line}
            </p>
          ))}
        </div>
      </div>
    </article>
  );
}

function Dataset({ dataset }: { dataset: DatasetCard }) {
  const labelColours: Record<string, string> = {
    human: 'var(--human)',
    ai_generated: 'var(--likely)',
    ai_polished: 'var(--possible)',
  };
  const total = Object.values(dataset.labels).reduce((sum, n) => sum + n, 0) || 1;

  return (
    <Section
      title="Dataset card"
      note={`v${dataset.dataset_version} · ${dataset.totals.documents} documents in ${dataset.totals.groups} leakage groups · regime ${dataset.data_regime}`}
      bodyClassName="stack stack--md"
    >
      <Banner tone="warning">{dataset.regime_note}</Banner>

      <div>
        <p className="subhead">Class balance</p>
        <div className="dist-bar">
          {Object.entries(dataset.labels).map(([label, count]) => (
            <div
              className="dist-bar__seg"
              key={label}
              style={{
                width: `${(count / total) * 100}%`,
                background: labelColours[label] ?? 'var(--ink-faint)',
              }}
              title={`${label}: ${count}`}
            >
              {count}
            </div>
          ))}
        </div>
        <div className="chips mt-1.5">
          {Object.entries(dataset.labels).map(([label, count]) => (
            <Badge key={label}>
              <span
                className="group-legend__dot"
                style={{ background: labelColours[label] ?? 'var(--ink-faint)' }}
              />
              {CLASS_SHORT[label] ?? label} {count}
            </Badge>
          ))}
        </div>
      </div>

      <div className="two-col">
        <div>
          <p className="subhead">Splits (by group, never by sample)</p>
          <table className="data">
            <thead>
              <tr>
                <th>Split</th>
                <th className="num">Docs</th>
                {Object.keys(dataset.labels).map((label) => (
                  <th className="num" key={label}>
                    {CLASS_SHORT[label] ?? label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(dataset.splits.counts).map(([split, count]) => (
                <tr key={split}>
                  <td>{split}</td>
                  <td className="num">{count}</td>
                  {Object.keys(dataset.labels).map((label) => (
                    <td className="num" key={label}>
                      {dataset.splits.labels_per_split[split]?.[label] ?? 0}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div>
          <p className="subhead">Leakage controls</p>
          <table className="data">
            <tbody>
              {Object.entries(dataset.leakage_controls).map(([key, value]) => (
                <tr key={key}>
                  <td>{key.replace(/_/g, ' ')}</td>
                  <td className="num">
                    {typeof value === 'boolean'
                      ? value
                        ? 'yes'
                        : 'no'
                      : String(value).slice(0, 60)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div>
        <p className="subhead">Known limitations</p>
        <ul className="statements">
          {dataset.known_limitations.map((limitation, index) => (
            <li key={index}>{limitation}</li>
          ))}
        </ul>
      </div>

      <div>
        <p className="subhead">Preprocessing</p>
        <ul className="statements">
          {dataset.preprocessing.map((step, index) => (
            <li key={index}>{step}</li>
          ))}
        </ul>
      </div>
    </Section>
  );
}
