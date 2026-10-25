import { afterEach, describe, expect, it, vi } from "vitest";
import { prefersReducedMotion } from "../../src/replay/motion";

describe("reduced motion preference", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("follows the browser reduced-motion setting", () => {
    const matchMedia = vi.fn().mockReturnValue({ matches: true });
    vi.stubGlobal("window", { matchMedia });
    expect(prefersReducedMotion()).toBe(true);
    expect(matchMedia).toHaveBeenCalledWith("(prefers-reduced-motion: reduce)");
  });
});
