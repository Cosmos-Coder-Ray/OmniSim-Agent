import { afterEach, describe, expect, it, vi } from "vitest";
import { playbackRate, prefersReducedMotion } from "../../src/replay/motion";

describe("reduced motion preference", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("follows the browser reduced-motion setting", () => {
    const matchMedia = vi.fn().mockReturnValue({ matches: true });
    vi.stubGlobal("window", { matchMedia });
    expect(prefersReducedMotion()).toBe(true);
    expect(playbackRate()).toBe(0.5);
    expect(matchMedia).toHaveBeenCalledWith("(prefers-reduced-motion: reduce)");
  });

  it("uses full speed when reduced motion is not requested", () => {
    expect(playbackRate(false)).toBe(1);
  });
});
