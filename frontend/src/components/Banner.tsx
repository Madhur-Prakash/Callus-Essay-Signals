import type { ReactNode } from 'react';

interface Props {
  tone?: 'info' | 'warning' | 'danger' | 'neutral';
  title?: string;
  children: ReactNode;
  action?: ReactNode;
}

export function Banner({ tone = 'neutral', title, children, action }: Props) {
  const className = tone === 'neutral' ? 'banner' : `banner banner--${tone}`;
  return (
    <div className={className} role={tone === 'danger' ? 'alert' : 'status'}>
      <div className="row items-start">
        <div className="min-w-0 flex-1">
          {title && <p className="banner__title">{title}</p>}
          <div>{children}</div>
        </div>
        {action}
      </div>
    </div>
  );
}
