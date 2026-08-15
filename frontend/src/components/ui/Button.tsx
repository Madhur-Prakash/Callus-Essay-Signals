import { cva, type VariantProps } from 'class-variance-authority';
import { forwardRef, type ButtonHTMLAttributes } from 'react';

import { cn } from '@/lib/cn';

/**
 * The variants map onto the `.btn` classes in components.css rather than
 * inlining a wall of utilities here. Two reasons: the same visual button also
 * appears in places that are not React (nothing today, but the CSS is the
 * contract), and keeping the paint in one place means a designer changing the
 * hover state edits one rule instead of hunting through TSX.
 *
 * What `cva` adds over a `className` string is that the variant names are typed,
 * so `<Button variant="primry">` fails at compile time.
 */
const button = cva('btn', {
  variants: {
    variant: {
      default: '',
      primary: 'btn--primary',
      ghost: 'btn--ghost',
    },
    size: {
      default: '',
      sm: 'btn--sm',
    },
  },
  defaultVariants: { variant: 'default', size: 'default' },
});

export interface ButtonProps
  extends ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof button> {}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant, size, type = 'button', ...props },
  ref,
) {
  // Defaulting `type` matters: a bare <button> inside a form submits it, and the
  // essay editor is a form. Every button in this app is an action, not a submit.
  return (
    <button
      ref={ref}
      type={type}
      className={cn(button({ variant, size }), className)}
      {...props}
    />
  );
});
