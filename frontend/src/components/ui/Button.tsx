import { cva, type VariantProps } from 'class-variance-authority';
import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';

import { useMagnetic } from '@/hooks/useMotion';
import { cn } from '@/lib/cn';

/**
 * The variants map onto the `.btn` classes in components.css rather than
 * inlining a wall of utilities here, so the paint lives in one place and a
 * change to the hover state is one edit rather than a hunt through TSX.
 *
 * What `cva` adds over a plain `className` string is that the variant names are
 * typed — `<Button variant="primry">` fails at compile time.
 */
const button = cva('btn', {
  variants: {
    variant: {
      default: '',
      primary: 'btn--primary',
      ghost: 'btn--ghost',
      outline: 'btn--outline',
    },
    size: {
      default: '',
      sm: 'btn--sm',
      lg: 'btn--lg',
    },
    block: { true: 'w-full', false: '' },
  },
  defaultVariants: { variant: 'default', size: 'default', block: false },
});

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof button> {
  /** Leans toward the cursor on hover. Reserve it for a page's main action. */
  magnetic?: boolean;
  /** Shows a spinner and blocks input without changing the button's width. */
  loading?: boolean;
  icon?: ReactNode;
  iconRight?: ReactNode;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    className,
    variant,
    size,
    block,
    magnetic = false,
    loading = false,
    icon,
    iconRight,
    children,
    type = 'button',
    disabled,
    ...props
  },
  forwardedRef,
) {
  const magneticRef = useMagnetic<HTMLButtonElement>();
  const ref = magnetic ? magneticRef : forwardedRef;

  return (
    // Defaulting `type` matters: a bare <button> inside a form submits it, and
    // the essay editor is a form. Every button in this app is an action.
    <button
      ref={ref}
      type={type}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={cn(button({ variant, size, block }), loading && 'btn--loading', className)}
      {...props}
    >
      {icon && (
        <span className="btn__icon" aria-hidden="true">
          {icon}
        </span>
      )}
      {/* The label keeps its box while loading so the button does not resize
          mid-click and move the pointer off its own target. */}
      <span className="btn__label">{children}</span>
      {loading && <span className="btn__spinner" aria-hidden="true" />}
      {iconRight && !loading && (
        <span className="btn__icon" aria-hidden="true">
          {iconRight}
        </span>
      )}
    </button>
  );
});
