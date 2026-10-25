import { describe, expect, it } from "vitest";
import { DEFAULT_CAMPAIGN, parseUrlState, tabs } from "../../src/state/urlStore";

describe("URL state", () => {
  it("uses the mock replay defaults for an empty URL", () => {
    expect(parseUrlState(new URL("http://localhost/"))).toEqual({
      tab: "replay",
      campaign: DEFAULT_CAMPAIGN,
      episode: "",
      t: 0,
    });
  });

  it("parses only known tabs and finite non-negative times", () => {
    const url = new URL("http://localhost/?tab=atlas&campaign=run-1&episode=ep-2&t=3.2");
    expect(parseUrlState(url)).toEqual({
      tab: "atlas",
      campaign: "run-1",
      episode: "ep-2",
      t: 3.2,
    });
    url.searchParams.set("tab", "unknown");
    url.searchParams.set("t", "Infinity");
    expect(parseUrlState(url)).toMatchObject({ tab: "replay", t: 0 });
  });

  it("supports an empty campaign default in API mode", () => {
    expect(parseUrlState(new URL("http://localhost/"), "").campaign).toBe("");
  });

  it("has six workspace tabs", () => {
    expect(tabs).toHaveLength(6);
  });
});
