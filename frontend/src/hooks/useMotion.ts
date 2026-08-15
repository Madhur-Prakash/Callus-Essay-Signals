/**
 * Motion primitives.
 *
 * Three libraries with three distinct jobs, deliberately non-overlapping:
 *
 *   Lenis          page-level smooth scrolling
 *   Framer Motion  declarative React enter/exit/layout transitions
 *   GSAP           imperative timelines: SVG path drawing, number count-ups,
 *                  scroll-triggered reveals
 *
 * Every one of them yields to `prefers-reduced-motion`, and all of them no-op
 * under test so the suite stays deterministic and fast.
 */

import { useEffect, useRef } from 'react';

import { prefersReducedMotion } from '@/hooks/useTheme';

/** jsdom has no layout, so animation would measure zeros and assert nothing. */
const IS_TEST = import.meta.env.MODE === 'test';

export function motionEnabled(): boolean {
  return !IS_TEST && !prefersReducedMotion();
}

/**
 * Smooth scrolling for the whole document.
 *
 * Kept gentle on purpose (short duration, near-linear easing): this interface is
 * for reading dense evidence tables, and heavy scroll smoothing makes precise
 * reading harder rather than more pleasant. It is disabled outright when the user
 * has asked for reduced motion, which also restores exact native scrolling.
 */
export function useSmoothScroll(): void {
  useEffect(() => {
    if (!motionEnabled()) return;

    let lenis: { raf: (t: number) => void; destroy: () => void } | null = null;
    let frame = 0;
    let cancelled = false;

    void (async () => {
      const { default: Lenis } = await import('lenis');
      if (cancelled) return;

      lenis = new Lenis({
        duration: 0.85,
        easing: (t: number) => 1 - Math.pow(1 - t, 3),
        smoothWheel: true,
        // Never smooth touch: on a phone it fights the platform's own physics.
        syncTouch: false,
      });

      const raf = (time: number) => {
        lenis?.raf(time);
        frame = requestAnimationFrame(raf);
      };
      frame = requestAnimationFrame(raf);
    })();

    return () => {
      cancelled = true;
      cancelAnimationFrame(frame);
      lenis?.destroy();
    };
  }, []);
}

/**
 * Animate a number from 0 to its value with GSAP.
 *
 * Used for the verdict counts and the research metrics. `format` keeps the
 * rendering identical to the static version, so a mid-flight frame can never
 * show a differently-formatted number than the final one.
 */
export function useCountUp(
  value: number,
  format: (n: number) => string = (n) => String(Math.round(n)),
  duration = 0.9,
) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!motionEnabled()) {
      node.textContent = format(value);
      return;
    }

    let tween: { kill: () => void } | null = null;
    let cancelled = false;

    void (async () => {
      const { default: gsap } = await import('gsap');
      if (cancelled || !ref.current) return;
      const counter = { n: 0 };
      tween = gsap.to(counter, {
        n: value,
        duration,
        ease: 'power2.out',
        onUpdate: () => {
          if (ref.current) ref.current.textContent = format(counter.n);
        },
        onComplete: () => {
          if (ref.current) ref.current.textContent = format(value);
        },
      });
    })();

    return () => {
      cancelled = true;
      tween?.kill();
    };
  }, [value, format, duration]);

  return ref;
}

/**
 * Draw an SVG path on mount (stroke-dashoffset from its own length to zero).
 * Measuring `getTotalLength()` rather than hard-coding a dasharray is what keeps
 * the effect correct when the viewBox or the data changes.
 */
export function useDrawPath<T extends SVGGeometryElement = SVGPathElement>(
  deps: unknown[] = [],
) {
  // SVGGeometryElement is the shared base that actually declares getTotalLength,
  // so paths, polylines and circles all satisfy it.
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!motionEnabled() || typeof node.getTotalLength !== 'function') {
      node.style.strokeDasharray = 'none';
      node.style.strokeDashoffset = '0';
      return;
    }

    let tween: { kill: () => void } | null = null;
    let cancelled = false;

    void (async () => {
      const { default: gsap } = await import('gsap');
      if (cancelled || !ref.current) return;
      const length = ref.current.getTotalLength();
      gsap.set(ref.current, { strokeDasharray: length, strokeDashoffset: length });
      tween = gsap.to(ref.current, {
        strokeDashoffset: 0,
        duration: 1.6,
        ease: 'power2.inOut',
      });
    })();

    return () => {
      cancelled = true;
      tween?.kill();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}

/**
 * Reveal children as they scroll into view, via GSAP ScrollTrigger.
 * Returns a container ref; every element matching `selector` inside it is staggered.
 */
export function useScrollReveal(selector: string, deps: unknown[] = []) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    if (!motionEnabled()) return;

    let ctx: { revert: () => void } | null = null;
    let cancelled = false;

    void (async () => {
      const [{ default: gsap }, { ScrollTrigger }] = await Promise.all([
        import('gsap'),
        import('gsap/ScrollTrigger'),
      ]);
      if (cancelled || !ref.current) return;
      gsap.registerPlugin(ScrollTrigger);

      ctx = gsap.context(() => {
        const targets = gsap.utils.toArray<HTMLElement>(selector);
        if (!targets.length) return;
        gsap.from(targets, {
          opacity: 0,
          y: 18,
          duration: 0.6,
          ease: 'power2.out',
          stagger: 0.07,
          scrollTrigger: {
            trigger: ref.current,
            start: 'top 88%',
            once: true,
          },
        });
      }, ref.current);
    })();

    return () => {
      cancelled = true;
      ctx?.revert();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}

/** Shared Framer Motion variants so timing is consistent across the app. */
export const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0 },
};

export const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.04 } },
};

export const EASE = [0.22, 1, 0.36, 1] as const;
