/**
 * The interface kit.
 *
 * One import path for every reusable primitive, so a page reads as a composition
 * of named pieces rather than a pile of relative imports. Anything domain-specific
 * (the essay reader, the rhythm chart, the evidence panel) lives one level up in
 * `components/` - the rule is that nothing in here may know what an essay is.
 */

export { Atmosphere } from './Atmosphere';
export { Badge, type BadgeProps } from './Badge';
export { Button, type ButtonProps } from './Button';
export { Card, CardBody, CardHead, CardTitle, Section } from './Card';
export { Gauge } from './Gauge';
export { Meter } from './Meter';
export { ProgressBar, ProgressSteps } from './Progress';
export { Reveal, SplitHeading } from './Reveal';
export { Sparkline } from './Sparkline';
export { Stat } from './Stat';
export { Surface, type SurfaceProps } from './Surface';
export { TabPanel, Tabs, type TabDefinition } from './Tabs';
