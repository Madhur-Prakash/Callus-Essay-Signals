/**
 * Scroll navigation.
 *
 *   useScrolled    has the page moved off the top? drives the masthead's state
 *   useScrollSpy   which section is currently in view, for the section rail
 *   scrollToSection  smooth scroll to a section, routed through Lenis when it is
 *                    running so the two do not fight over the scroll position
 *
 * Reading position lives in `components/ui/ScrollProgress` rather than here,
 * because a per-frame value has no business in React state — see the note there.
 *
 * The ScrollTrigger-backed pieces are inert under `prefers-reduced-motion` and
 * under test, matching the gate in useMotion.
 */

import { useEffect, useState } from 'react';

import { motionEnabled } from '@/hooks/useMotion';

/**
 * Whether the page has scrolled off the top.
 *
 * Two thresholds, not one. With a single boundary, scrolling gently around it
 * flips the state on and off repeatedly and the bar strobes. Turning on at `on`
 * and off only back below `off` gives the state somewhere to rest.
 *
 * A boolean rather than a continuous value for the same reason: a bar whose
 * geometry tracks scroll position jitters on every wheel tick, whereas one
 * transition between two states settles once and stays put.
 */
export function useScrolled(on = 32, off = 6): boolean {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    let frame = 0;
    const read = () => {
      frame = 0;
      const y = document.documentElement.scrollTop;
      // React bails out when the next value equals the current one, so this
      // re-renders on the two crossings and never in between.
      setScrolled((current) => (current ? y > off : y > on));
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(read);
    };

    read();
    window.addEventListener('scroll', schedule, { passive: true });
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('scroll', schedule);
    };
  }, [on, off]);

  return scrolled;
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
