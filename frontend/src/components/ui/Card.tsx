import type { HTMLAttributes, ReactNode } from 'react';

import { Surface, type SurfaceProps } from '@/components/ui/Surface';
import { cn } from '@/lib/cn';

export function Card({ className, ...props }: SurfaceProps) {
  return <Surface className={cn('card', className)} {...props} />;
}

export function CardHead({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('card__head', className)} {...props} />;
}

export function CardBody({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('card__body', className)} {...props} />;
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLParagraphElement>) {
  return <p className={cn('card__title', className)} {...props} />;
}

interface SectionProps extends Omit<SurfaceProps, 'title' | 'children'> {
  title: ReactNode;
  /** One line under the title explaining what the reader is looking at. */
  note?: ReactNode;
  /** Rendered at the far right of the head - a filter, a count, a legend. */
  action?: ReactNode;
  /** A short mono string in the corner: a count, a version, a slice size. */
  meta?: ReactNode;
  children: ReactNode;
  bodyClassName?: string;
}

/**
 * A titled panel. This head/title/note/body arrangement appeared thirty-odd times
 * spelled out in full; collapsing it to one component is the difference between
 * reading a page's structure and scanning past its boilerplate.
 *
 * `<section>` rather than `<div>` so each is a landmark, and the title is a `<p>`
 * rather than a heading because these are peers of each other, not a document
 * outline - promoting them to `<h2>` would imply a hierarchy the tabs already own.
 */
export function Section({
  title,
  note,
  action,
  meta,
  children,
  className,
  bodyClassName,
  ...props
}: SectionProps) {
  return (
    <Surface as="section" className={cn('card', className)} {...props}>
      <div className="card__head">
        <div className="min-w-0">
          <CardTitle>{title}</CardTitle>
          {note ? <p className="section-note">{note}</p> : null}
        </div>
        {meta ? <span className="card__meta">{meta}</span> : null}
        {action ? <div className="ml-auto flex-none">{action}</div> : null}
      </div>
      <div className={cn('card__body', bodyClassName)}>{children}</div>
    </Surface>
  );
}
