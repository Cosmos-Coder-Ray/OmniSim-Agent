import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DEFAULT_CAMPAIGN, parseUrlState, tabs, useUrlStore, URL_WRITE_DEBOUNCE_MS } from "../../src/state/urlStore";

let replaceState: ReturnType<typeof vi.fn>;
let pushState: ReturnType<typeof vi.fn>;

beforeEach(() => {
  vi.useFakeTimers();
  replaceState = vi.fn();
  pushState = vi.fn();
  vi.stubGlobal("window", {
    location: { href: "http://localhost/" },
    history: { state: null, replaceState, pushState },
    setTimeout: globalThis.setTimeout.bind(globalThis),
    clearTimeout: globalThis.clearTimeout.bind(globalThis),
  });
  useUrlStore.setState({ t: 0, isPlaying: false });
});

afterEach(() => {
  vi.clearAllTimers();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

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

  it("does not write browser history while playback time advances", () => {
    const { setPlaying, setT } = useUrlStore.getState();
    setPlaying(true);
    for (const time of [0.1, 0.2, 0.3, 1, 2, 3.2]) setT(time);
    vi.advanceTimersByTime(URL_WRITE_DEBOUNCE_MS * 4);
    expect(replaceState).not.toHaveBeenCalled();
    expect(pushState).not.toHaveBeenCalled();
  });

  it("writes playback time only after pause debounce", () => {
    const { setPlaying, setT } = useUrlStore.getState();
    setPlaying(true);
    setT(3.2);
    useUrlStore.getState().setPlaying(false);
    vi.advanceTimersByTime(URL_WRITE_DEBOUNCE_MS - 1);
    expect(replaceState).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(replaceState).toHaveBeenCalledTimes(1);
    const written = new URL(replaceState.mock.calls[0][2] as string, "http://localhost/");
    expect(written.searchParams.get("t")).toBe("3.20");
    expect(pushState).not.toHaveBeenCalled();
  });

  it("pauses when seeking and debounces the URL update", () => {
    useUrlStore.getState().setPlaying(true);
    useUrlStore.getState().seekTo(1.25);
    expect(useUrlStore.getState().isPlaying).toBe(false);
    vi.advanceTimersByTime(URL_WRITE_DEBOUNCE_MS);
    const written = new URL(replaceState.mock.calls[0][2] as string, "http://localhost/");
    expect(written.searchParams.get("t")).toBe("1.25");
  });
});
