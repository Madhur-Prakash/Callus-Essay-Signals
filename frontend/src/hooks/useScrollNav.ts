/**
 * GSAP-driven scroll navigation.
 *
 * Three pieces, all ScrollTrigger-based:
 *   useScrollProgress  a 0→1 progress value for the bar under the masthead
 *   useScrollSpy       which section is currently in view, for the section rail
 *   scrollToSection    smooth scroll to a section, routed through Lenis when it
 *                      is running so the two do not fight over the scroll position
 *
 * All of it is inert under `prefers-reduced-motion` and under test, matching the
 * gate in useMotion.
 */

import { useEffect, useState } from 'react';

import { motionEnabled } from '@/hooks/useMotion';

/** Fraction of the page scrolled, 0-1. Drives the masthead progress bar. */
export function useScrollProgress(): number {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    // The progress bar is informational, not decorative, so it is kept even when
    // motion is reduced — it just updates on plain scroll events instead of via
    // ScrollTrigger.
    const read = () => {
      const doc = document.documentElement;
      const max = doc.scrollHeight - doc.clientHeight;
      setProgress(max > 0 ? Math.min(1, Math.max(0, doc.scrollTop / max)) : 0);
    };
    read();
    window.addEventListener('scroll', read, { passive: true });
    window.addEventListener('resize', read);
    return () => {
      window.removeEventListener('scroll', read);
      window.removeEventListener('resize', read);
    };
  }, []);

  return progress;
}

/**
 * Which of `ids` is currently the active section.
 *
 * Uses ScrollTrigger when motion is on (it handles resize, pinning and refresh
 * correctly) and falls back to IntersectionObserver otherwise, so the rail still
 * highlights for reduced-motion users.
 */
export function useScrollSpy(ids: string[], deps: unknown[] = []): string | null {
  const [active, setActive] = useState<string | null>(ids[0] ?? null);

  useEffect(() => {
    if (!ids.length) return;

    if (!motionEnabled()) {
      const observer = new IntersectionObserver(
        (entries) => {
          const visible = entries
            .filter((e) => e.isIntersecting)
            .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
          if (visible) setActive(visible.target.id);
        },
        { rootMargin: '-20% 0px -70% 0px' },
      );
      ids.forEach((id) => {
        const node = document.getElementById(id);
        if (node) observer.observe(node);
      });
      return () => observer.disconnect();
    }

    let ctx: { revert: () => void } | null = null;
    let cancelled = false;

    void (async () => {
      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([
        import('gsap'),
        import('gsap/ScrollTrigger'),
      ]);
      if (cancelled) return;
      gsap.registerPlugin(ScrollTrigger);

      ctx = gsap.context(() => {
        ids.forEach((id) => {
          const node = document.getElementById(id);
          if (!node) return;
          ScrollTrigger.create({
            trigger: node,
            start: 'top 30%',
            end: 'bottom 30%',
            onToggle: (self) => {
              if (self.isActive) setActive(id);
            },
          });
        });
      });
      ScrollTrigger.refresh();
    })();

    return () => {
      cancelled = true;
      ctx?.revert();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return active;
}

/**
 * Scroll to a section by id.
 *
 * Lenis owns the scroll position while it is running, so a plain
 * `scrollIntoView` would be immediately overridden by its RAF loop. The Lenis
 * instance is published on `window.__lenis` by useSmoothScroll for exactly this.
 */
export function scrollToSection(id: string, offset = 96): void {
  const node = document.getElementById(id);
  if (!node) return;

  const lenis = (window as unknown as { __lenis?: { scrollTo: (t: number, o?: object) => void } })
    .__lenis;

  const top = node.getBoundingClientRect().top + window.scrollY - offset;

  if (lenis) {
    lenis.scrollTo(top, { duration: 0.9 });
  } else {
    window.scrollTo({
      top,
      behavior: motionEnabled() ? 'smooth' : 'auto',
    });
  }
  // Move focus so keyboard users land in the section they asked for.
  node.setAttribute('tabindex', '-1');
  node.focus({ preventScroll: true });
}
