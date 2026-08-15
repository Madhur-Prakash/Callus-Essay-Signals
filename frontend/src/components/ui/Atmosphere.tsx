import { useEffect, useRef } from 'react';

import { motionEnabled } from '@/hooks/useMotion';

/**
 * The lit ground the whole app sits on: two slow-drifting colour fields and a
 * fine grain overlay.
 *
 * Three decisions worth stating:
 *
 * 1. Its colours come from `--aurora-*` tokens, so the theme toggle recolours it
 *    without this component knowing that themes exist.
 * 2. The drift is a GSAP timeline on `xPercent`/`yPercent`, not a CSS keyframe
 *    animation on `background-position`. Transforms stay on the compositor;
 *    animating a gradient's position repaints a full-screen layer every frame.
 * 3. The grain is an inline SVG `feTurbulence` data URI rather than an image
 *    file - no network request, no asset to lose, and it scales to any DPI.
 *    Grain is what stops large flat gradients from banding on 8-bit displays.
 */
export function Atmosphere() {
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const root = rootRef.current;
    if (!root) return;
    if (!motionEnabled()) return;

    let ctx: { revert: () => void } | null = null;
    let cancelled = false;

    void (async () => {
      const { default: gsap } = await import('gsap');
      if (cancelled || !rootRef.current) return;

      ctx = gsap.context(() => {
        // Long, offset, non-repeating-feeling loops. Anything faster than this
        // reads as a screensaver rather than as light in a room.
        gsap.to('.atmosphere__glow--1', {
          xPercent: 14,
          yPercent: 10,
          scale: 1.14,
          duration: 26,
          ease: 'sine.inOut',
          repeat: -1,
          yoyo: true,
        });
        gsap.to('.atmosphere__glow--2', {
          xPercent: -12,
          yPercent: 14,
          scale: 1.2,
          duration: 34,
          ease: 'sine.inOut',
          repeat: -1,
          yoyo: true,
          delay: -8,
        });
        gsap.to('.atmosphere__glow--3', {
          xPercent: 9,
          yPercent: -11,
          duration: 42,
          ease: 'sine.inOut',
          repeat: -1,
          yoyo: true,
          delay: -16,
        });
      }, rootRef.current);
    })();

    return () => {
      cancelled = true;
      ctx?.revert();
    };
  }, []);

  return (
    <div className="atmosphere" ref={rootRef} aria-hidden="true">
      <div className="atmosphere__glow atmosphere__glow--1" />
      <div className="atmosphere__glow atmosphere__glow--2" />
      <div className="atmosphere__glow atmosphere__glow--3" />
      <div className="atmosphere__grid" />
      <div className="atmosphere__grain" />
    </div>
  );
}
