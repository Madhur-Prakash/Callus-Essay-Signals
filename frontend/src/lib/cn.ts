import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Join class names, letting later Tailwind utilities win over earlier ones.
 *
 * Plain string concatenation does not do this: `"p-4 p-8"` leaves both in the
 * class list and the winner is whichever the stylesheet happens to emit last,
 * which is not something a caller can reason about. `twMerge` resolves the
 * conflict by keyword, so a component can ship sensible defaults and still be
 * overridden from the outside with a `className` prop.
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
