export function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

export function playbackRate(reducedMotion = prefersReducedMotion()): number {
  return reducedMotion ? 0.5 : 1;
}
