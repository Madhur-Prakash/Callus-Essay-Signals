import { cva, type VariantProps } from 'class-variance-authority';
import { forwardRef, type ElementType, type HTMLAttributes } from 'react';

import { useSpotlight } from '@/hooks/useMotion';
import { cn } from '@/lib/cn';

/**
 * The panel primitive every other container is built from.
 *
 * `tone` is the material, not the colour: `glass` is translucent and blurred so
 * the atmosphere behind it shows through, `solid` is opaque for anything that
 * sits on top of dense text, `sunken` recedes. Keeping this on one component is
 * what stops nine slightly-different card treatments accumulating.
 */
const surface = cva('surface', {
  variants: {
    tone: {
      glass: 'surface--glass',
      solid: 'surface--solid',
      sunken: 'surface--sunken',
      bare: 'surface--bare',
    },
    radius: {
      control: 'rounded-control',
      card: 'rounded-card',
      panel: 'rounded-panel',
      none: 'rounded-none',
    },
    /** A soft light that follows the cursor. Costs one style write per frame. */
    spotlight: { true: 'surface--spotlight', false: '' },
    /** Lifts and brightens on hover - only for panels that are clickable. */
    interactive: { true: 'surface--interactive', false: '' },
  },
  defaultVariants: { tone: 'glass', radius: 'card', spotlight: false, interactive: false },
});

export interface SurfaceProps
  extends HTMLAttributes<HTMLElement>,
    VariantProps<typeof surface> {
  as?: ElementType;
}

export const Surface = forwardRef<HTMLElement, SurfaceProps>(function Surface(
  { as: Tag = 'div', className, tone, radius, spotlight, interactive, ...props },
  forwardedRef,
) {
  const spotRef = useSpotlight<HTMLElement>();

  // The hook owns its own ref, so a caller passing one would be dropped silently.
  // Spotlight and an external ref are never both needed, and this makes the
  // conflict a compile-time-visible choice rather than a runtime mystery.
  const ref = spotlight ? spotRef : forwardedRef;

  return (
    <Tag
      ref={ref}
      className={cn(surface({ tone, radius, spotlight, interactive }), className)}
      {...props}
    />
  );
});
