import { type ElementType, type HTMLAttributes, type ReactNode } from 'react';

import { useLineReveal, useScrollReveal } from '@/hooks/useMotion';
import { cn } from '@/lib/cn';

interface RevealProps extends HTMLAttributes<HTMLDivElement> {
  /**
   * What to stagger inside this container. Defaults to direct children, which is
   * the common case; pass a selector to reach a specific set instead.
   */
  selector?: string;
  /** Re-run when these change — needed when the children arrive asynchronously. */
  deps?: unknown[];
  y?: number;
  stagger?: number;
  children: ReactNode;
}

/**
 * Scroll-reveal wrapper.
 *
 * Declarative counterpart to `useScrollReveal`, so a page can say what reveals
 * without every page component having to hold a ref and a hook call. The
 * animation runs once and never reverses — content that re-hides on scroll-up
 * makes a page feel unreliable when you scroll back to re-read something.
 */
export function Reveal({
  selector = ':scope > *',
  deps = [],
  y,
  stagger,
  className,
  children,
  ...props
}: RevealProps) {
  const ref = useScrollReveal(selector, deps, {
    ...(y === undefined ? {} : { y }),
    ...(stagger === undefined ? {} : { stagger }),
  });

  return (
    <div ref={ref as React.RefObject<HTMLDivElement>} className={cn(className)} {...props}>
      {children}
    </div>
  );
}

interface SplitHeadingProps extends Omit<HTMLAttributes<HTMLElement>, 'children'> {
  as?: ElementType;
  /**
   * Plain text only. The reveal rebuilds the element's DOM into one masked
   * wrapper per rendered line, so nested markup would be flattened — passing a
   * string makes that constraint visible at the call site.
   */
  children: string;
  delay?: number;
}

/**
 * A heading whose lines rise out from behind a mask on mount.
 *
 * Reserved for the one heading that opens a page. Applied to every heading it
 * would turn reading into waiting.
 */
export function SplitHeading({
  as: Tag = 'h1',
  children,
  delay,
  className,
  ...props
}: SplitHeadingProps) {
  const ref = useLineReveal<HTMLElement>(
    [children],
    delay === undefined ? {} : { delay },
  );

  return (
    <Tag ref={ref} className={cn(className)} {...props}>
      {children}
    </Tag>
  );
}
