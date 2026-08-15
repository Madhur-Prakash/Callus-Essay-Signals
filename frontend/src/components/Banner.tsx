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
      <div className="row" style={{ alignItems: 'flex-start' }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          {title && <p className="banner__title">{title}</p>}
          <div>{children}</div>
        </div>
        {action}
      </div>
    </div>
  );
}
