import type { HTMLAttributes, ReactNode } from 'react';

import { cn } from '@/lib/cn';

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('card', className)} {...props} />;
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

interface SectionProps extends Omit<HTMLAttributes<HTMLElement>, 'title'> {
  title: ReactNode;
  /** One line under the title explaining what the reader is looking at. */
  note?: ReactNode;
  /** Rendered at the far right of the head — a filter, a count, a legend. */
  action?: ReactNode;
  children: ReactNode;
  bodyClassName?: string;
}

/**
 * A titled card. This exact head/title/note/body arrangement appeared thirty-odd
 * times across the pages, always spelled out in full; collapsing it to one
 * component is the difference between reading a page's structure and scanning
 * past its boilerplate.
 *
 * `<section>` rather than `<div>` so each one is a landmark, and the title is a
 * `<p>` rather than a heading because these are peers of each other, not a
 * document outline — promoting them to `<h2>` would imply a hierarchy the tabs
 * already own.
 */
export function Section({
  title,
  note,
  action,
  children,
  className,
  bodyClassName,
  ...props
}: SectionProps) {
  return (
    <section className={cn('card', className)} {...props}>
      <div className="card__head">
        <div className="min-w-0">
          <CardTitle>{title}</CardTitle>
          {note ? <p className="section-note">{note}</p> : null}
        </div>
        {action ? <div className="ml-auto flex-none">{action}</div> : null}
      </div>
      <div className={cn('card__body', bodyClassName)}>{children}</div>
    </section>
  );
}
