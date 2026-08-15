/**
 * Motion primitives.
 *
 * Three libraries with three distinct jobs, deliberately non-overlapping:
 *
 *   Lenis          page-level smooth scrolling
 *   Framer Motion  declarative React enter/exit/layout transitions
 *   GSAP           imperative timelines: text reveals, SVG drawing, count-ups,
 *                  scroll-triggered choreography, pointer-driven effects
 *
 * GSAP is loaded dynamically in every hook below so it stays out of the initial
 * bundle and off the critical path — nothing here runs before first paint.
 *
 * Every one of them yields to `prefers-reduced-motion`, and all of them no-op
 * under test so the suite stays deterministic and fast. The rule for what earns
 * an animation: it either shows a relationship (where this came from, what it
 * became) or it reports state (working, arriving, changing). Decoration that
 * does neither is not worth the frame budget or the vestibular risk.
 */

import { useEffect, useRef } from 'react';

import { prefersReducedMotion } from '@/hooks/useTheme';

/** jsdom has no layout, so animation would measure zeros and assert nothing. */
const IS_TEST = import.meta.env.MODE === 'test';

export function motionEnabled(): boolean {
  return !IS_TEST && !prefersReducedMotion();
}

/** The house easing curve — a fast start that settles, never a bounce. */
export const EASE = [0.22, 1, 0.36, 1] as const;
export const GSAP_EASE = 'power3.out';

type Killable = { kill: () => void };
type Ctx = { revert: () => void };

/** Load GSAP once and share the module across every hook that needs it. */
async function loadGsap() {
  const { default: gsap } = await import('gsap');
  return gsap;
}

async function loadScrollTrigger() {
  const [gsap, { ScrollTrigger }] = await Promise.all([
    loadGsap(),
    import('gsap/ScrollTrigger'),
  ]);
  gsap.registerPlugin(ScrollTrigger);
  return { gsap, ScrollTrigger };
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

      // Published so scrollToSection can route through Lenis. Without this, a
      // programmatic scroll is immediately overridden by Lenis's own RAF loop.
      (window as unknown as { __lenis?: unknown }).__lenis = lenis;

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
      delete (window as unknown as { __lenis?: unknown }).__lenis;
    };
  }, []);
}

/**
 * Animate a number from 0 to its value with GSAP.
 *
 * `format` keeps the rendering identical to the static version, so a mid-flight
 * frame can never show a differently-formatted number than the final one.
 */
export function useCountUp(
  value: number,
  format: (n: number) => string = (n) => String(Math.round(n)),
  duration = 1.1,
) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!motionEnabled()) {
      node.textContent = format(value);
      return;
    }

    let tween: Killable | null = null;
    let cancelled = false;

    void (async () => {
      const gsap = await loadGsap();
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
 * Draw an SVG path (stroke-dashoffset from its own length to zero).
 * Measuring `getTotalLength()` rather than hard-coding a dasharray is what keeps
 * the effect correct when the viewBox or the data changes.
 */
export function useDrawPath<T extends SVGGeometryElement = SVGPathElement>(
  deps: unknown[] = [],
  { duration = 1.6, delay = 0 }: { duration?: number; delay?: number } = {},
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

    let tween: Killable | null = null;
    let cancelled = false;

    void (async () => {
      const gsap = await loadGsap();
      if (cancelled || !ref.current) return;
      const length = ref.current.getTotalLength();
      gsap.set(ref.current, { strokeDasharray: length, strokeDashoffset: length });
      tween = gsap.to(ref.current, {
        strokeDashoffset: 0,
        duration,
        delay,
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
export function useScrollReveal(
  selector: string,
  deps: unknown[] = [],
  { y = 24, stagger = 0.07, start = 'top 88%' } = {},
) {
  const ref = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const container = ref.current;
    if (!container) return;
    if (!motionEnabled()) return;

    let ctx: Ctx | null = null;
    let cancelled = false;

    void (async () => {
      const { gsap } = await loadScrollTrigger();
      if (cancelled || !ref.current) return;

      ctx = gsap.context(() => {
        const targets = gsap.utils.toArray<HTMLElement>(selector);
        if (!targets.length) return;
        gsap.from(targets, {
          opacity: 0,
          y,
          duration: 0.7,
          ease: GSAP_EASE,
          stagger,
          scrollTrigger: { trigger: ref.current, start, once: true },
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

/**
 * Reveal a heading by line, each line rising out from behind a mask.
 *
 * The DOM is rewritten rather than animated in place, because a mask reveal needs
 * a clipping wrapper per line and there is no way to get one from CSS alone. The
 * original markup is captured first and restored on cleanup, so React re-renders
 * and Strict Mode's double-effect both stay safe.
 *
 * Splitting happens on *rendered* lines, measured after layout — not on `<br>` or
 * word count — so it stays correct at every viewport width.
 */
export function useLineReveal<T extends HTMLElement = HTMLHeadingElement>(
  deps: unknown[] = [],
  { delay = 0.05, stagger = 0.09, duration = 1 } = {},
) {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!motionEnabled()) return;

    const original = node.innerHTML;
    let ctx: Ctx | null = null;
    let cancelled = false;

    void (async () => {
      const gsap = await loadGsap();
      if (cancelled || !ref.current) return;
      const el = ref.current;

      // Wrap every word so their bounding boxes reveal where the line breaks fall.
      const words = (el.textContent ?? '').split(/(\s+)/).filter(Boolean);
      el.textContent = '';
      const spans: HTMLSpanElement[] = [];
      for (const word of words) {
        if (/^\s+$/.test(word)) {
          el.appendChild(document.createTextNode(word));
          continue;
        }
        const span = document.createElement('span');
        span.textContent = word;
        span.style.display = 'inline-block';
        el.appendChild(span);
        spans.push(span);
      }

      // Group by vertical offset: same `offsetTop` means same rendered line.
      const lines = new Map<number, HTMLSpanElement[]>();
      for (const span of spans) {
        const top = Math.round(span.offsetTop);
        const bucket = lines.get(top);
        if (bucket) bucket.push(span);
        else lines.set(top, [span]);
      }

      // Rebuild as one masked wrapper per line.
      el.textContent = '';
      const inners: HTMLSpanElement[] = [];
      for (const bucket of lines.values()) {
        const mask = document.createElement('span');
        mask.style.display = 'block';
        mask.style.overflow = 'hidden';
        // Descenders sit below the baseline and would be clipped by a tight mask.
        mask.style.paddingBottom = '0.12em';
        mask.style.marginBottom = '-0.12em';

        const inner = document.createElement('span');
        inner.style.display = 'block';
        inner.style.willChange = 'transform';
        inner.textContent = bucket.map((s) => s.textContent).join(' ');

        mask.appendChild(inner);
        el.appendChild(mask);
        inners.push(inner);
      }

      ctx = gsap.context(() => {
        gsap.from(inners, {
          yPercent: 118,
          duration,
          delay,
          stagger,
          ease: 'expo.out',
        });
      }, el);
    })();

    return () => {
      cancelled = true;
      ctx?.revert();
      // Restore the node this effect actually rewrote, not whatever the ref
      // points at now. If React swapped the element between run and cleanup,
      // `ref.current` is a fresh node we never touched — writing the captured
      // markup into it would duplicate content rather than undo anything.
      node.innerHTML = original;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}

/**
 * Magnetic hover: the element leans toward the cursor and springs back on exit.
 *
 * Deliberately small (a few pixels) and pointer-only. It is applied to the two
 * primary calls to action, not to every button — the effect says "this one is
 * the target", which stops being true the moment everything does it.
 */
export function useMagnetic<T extends HTMLElement = HTMLButtonElement>(strength = 0.32) {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!motionEnabled()) return;
    // Touch and mouse-less pointers have no hover to be magnetic about.
    if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;

    let cancelled = false;
    let quickX: ((v: number) => void) | null = null;
    let quickY: ((v: number) => void) | null = null;

    const onMove = (event: PointerEvent) => {
      if (!quickX || !quickY) return;
      const rect = node.getBoundingClientRect();
      quickX((event.clientX - (rect.left + rect.width / 2)) * strength);
      quickY((event.clientY - (rect.top + rect.height / 2)) * strength);
    };

    const onLeave = () => {
      quickX?.(0);
      quickY?.(0);
    };

    void (async () => {
      const gsap = await loadGsap();
      if (cancelled || !ref.current) return;
      // `quickTo` reuses one tween instead of allocating per pointermove, which
      // is the difference between smooth and jittery at 120Hz.
      quickX = gsap.quickTo(ref.current, 'x', { duration: 0.5, ease: 'power3.out' });
      quickY = gsap.quickTo(ref.current, 'y', { duration: 0.5, ease: 'power3.out' });
      node.addEventListener('pointermove', onMove);
      node.addEventListener('pointerleave', onLeave);
    })();

    return () => {
      cancelled = true;
      node.removeEventListener('pointermove', onMove);
      node.removeEventListener('pointerleave', onLeave);
    };
  }, [strength]);

  return ref;
}

/**
 * Publish the pointer position on a panel as `--mx` / `--my` percentages, so CSS
 * can put a soft light where the cursor is.
 *
 * Writing custom properties rather than animating a real element keeps this to a
 * single style mutation per frame and leaves the highlight entirely to CSS — the
 * panel decides how to use the light, this only says where it is.
 */
export function useSpotlight<T extends HTMLElement = HTMLDivElement>() {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!motionEnabled()) return;
    if (!window.matchMedia('(hover: hover) and (pointer: fine)').matches) return;

    let frame = 0;
    const onMove = (event: PointerEvent) => {
      cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const rect = node.getBoundingClientRect();
        node.style.setProperty('--mx', `${((event.clientX - rect.left) / rect.width) * 100}%`);
        node.style.setProperty('--my', `${((event.clientY - rect.top) / rect.height) * 100}%`);
      });
    };
    const onEnter = () => node.style.setProperty('--spot', '1');
    const onLeave = () => {
      cancelAnimationFrame(frame);
      node.style.setProperty('--spot', '0');
    };

    node.addEventListener('pointermove', onMove);
    node.addEventListener('pointerenter', onEnter);
    node.addEventListener('pointerleave', onLeave);
    return () => {
      cancelAnimationFrame(frame);
      node.removeEventListener('pointermove', onMove);
      node.removeEventListener('pointerenter', onEnter);
      node.removeEventListener('pointerleave', onLeave);
    };
  }, []);

  return ref;
}

/**
 * Drift an element as the page scrolls, scrubbed rather than eased.
 *
 * Only used on the atmosphere and the hero motif. Parallax on content makes text
 * harder to track with the eye, so nothing readable gets this.
 */
export function useParallax<T extends HTMLElement = HTMLDivElement>(distance = 80) {
  const ref = useRef<T | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (!motionEnabled()) return;

    let ctx: Ctx | null = null;
    let cancelled = false;

    void (async () => {
      const { gsap } = await loadScrollTrigger();
      if (cancelled || !ref.current) return;
      ctx = gsap.context(() => {
        gsap.to(ref.current, {
          y: distance,
          ease: 'none',
          scrollTrigger: {
            trigger: ref.current,
            start: 'top top',
            end: 'bottom top',
            scrub: 0.6,
          },
        });
      }, ref.current);
    })();

    return () => {
      cancelled = true;
      ctx?.revert();
    };
  }, [distance]);

  return ref;
}

/**
 * Sweep an SVG arc from empty to `value` (0–1).
 *
 * Separate from useDrawPath because a gauge animates a *fraction* of its own
 * circumference rather than drawing the whole stroke, and it has to re-run when
 * the value changes rather than only on mount.
 */
export function useArcSweep(value: number, { duration = 1.4, delay = 0.15 } = {}) {
  const ref = useRef<SVGCircleElement | null>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const length = typeof node.getTotalLength === 'function' ? node.getTotalLength() : 0;
    if (!length) return;

    node.style.strokeDasharray = String(length);

    if (!motionEnabled()) {
      node.style.strokeDashoffset = String(length * (1 - value));
      return;
    }

    let tween: Killable | null = null;
    let cancelled = false;

    void (async () => {
      const gsap = await loadGsap();
      if (cancelled || !ref.current) return;
      gsap.set(ref.current, { strokeDashoffset: length });
      tween = gsap.to(ref.current, {
        strokeDashoffset: length * (1 - value),
        duration,
        delay,
        ease: 'power3.inOut',
      });
    })();

    return () => {
      cancelled = true;
      tween?.kill();
    };
  }, [value, duration, delay]);

  return ref;
}

/** Shared Framer Motion variants so timing is consistent across the app. */
export const fadeUp = {
  hidden: { opacity: 0, y: 16 },
  visible: { opacity: 1, y: 0 },
};

export const stagger = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.07, delayChildren: 0.04 } },
};
