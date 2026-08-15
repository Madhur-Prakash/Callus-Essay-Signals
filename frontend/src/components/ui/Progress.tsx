import { cn } from '@/lib/cn';

interface ProgressBarProps {
  /** 0–1. Omit for an indeterminate bar. */
  value?: number;
  className?: string;
  ariaLabel?: string;
}

/**
 * A progress bar, determinate or not.
 *
 * The indeterminate form is the honest one for this app's analyse step: the
 * synchronous endpoint returns a single response, so any percentage would be
 * invented. A sweeping bar says "working" without claiming to know how far.
 */
export function ProgressBar({ value, className, ariaLabel }: ProgressBarProps) {
  const determinate = typeof value === 'number' && Number.isFinite(value);
  const clamped = determinate ? Math.max(0, Math.min(1, value)) : 0;

  return (
    <div
      className={cn('progress__track', className)}
      role="progressbar"
      aria-label={ariaLabel}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={determinate ? Math.round(clamped * 100) : undefined}
    >
      <div
        className={cn('progress__bar', !determinate && 'progress__bar--indeterminate')}
        style={determinate ? { width: `${clamped * 100}%` } : undefined}
      />
    </div>
  );
}

interface ProgressStepsProps {
  steps: string[];
  /** Index of the furthest step reached. Everything up to it reads as done. */
  current: number;
  className?: string;
}

/** The checklist under a progress bar. Reusable wherever work has named phases. */
export function ProgressSteps({ steps, current, className }: ProgressStepsProps) {
  return (
    <ol className={cn('progress__steps', className)}>
      {steps.map((label, index) => {
        const done = index < current;
        const active = index === current;
        return (
          <li
            key={label}
            className={cn(
              'progress__step',
              (done || active) && 'progress__step--active',
              active && 'progress__step--current',
            )}
          >
            <span className="progress__dot" aria-hidden="true" />
            {label}
          </li>
        );
      })}
    </ol>
  );
}
