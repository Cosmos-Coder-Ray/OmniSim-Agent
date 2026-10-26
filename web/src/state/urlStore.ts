import { create } from "zustand";

export const URL_WRITE_DEBOUNCE_MS = 300;
export const DEFAULT_CAMPAIGN = "mock_campaign_01";

export const tabs = [
  { id: "replay", label: "Replay" },
  { id: "campaign", label: "Live Campaign" },
  { id: "atlas", label: "Failure Atlas" },
  { id: "evidence", label: "Evidence" },
  { id: "envelope", label: "Safe Envelope" },
  { id: "benchmarks", label: "Benchmarks" },
] as const;

export type TabId = (typeof tabs)[number]["id"];

export interface UrlState {
  tab: TabId;
  campaign: string;
  episode: string;
  t: number;
}

interface AppState extends UrlState {
  isPlaying: boolean;
  setTab: (tab: TabId) => void;
  setCampaign: (campaign: string) => void;
  setEpisode: (episode: string) => void;
  setT: (t: number) => void;
  setPlaying: (isPlaying: boolean) => void;
  seekTo: (t: number) => void;
  syncFromUrl: () => void;
}

let timeWriteTimer: number | undefined;

export function parseUrlState(url: URL, emptyCampaign = DEFAULT_CAMPAIGN): UrlState {
  const tab = tabs.some((item) => item.id === url.searchParams.get("tab"))
    ? (url.searchParams.get("tab") as TabId)
    : "replay";
  const parsedTime = Number(url.searchParams.get("t") ?? 0);
  const campaign = url.searchParams.get("campaign");
  return {
    tab,
    campaign: campaign === null ? emptyCampaign : campaign,
    episode: url.searchParams.get("episode") || "",
    t: Number.isFinite(parsedTime) && parsedTime >= 0 ? parsedTime : 0,
  };
}

function currentUrlState(): UrlState {
  const emptyCampaign = import.meta.env.VITE_DATA_SOURCE === "api" ? "" : DEFAULT_CAMPAIGN;
  return typeof window === "undefined"
    ? { tab: "replay", campaign: emptyCampaign, episode: "", t: 0 }
    : parseUrlState(new URL(window.location.href), emptyCampaign);
}

function clearTimeWrite(): void {
  if (timeWriteTimer !== undefined && typeof window !== "undefined") {
    window.clearTimeout(timeWriteTimer);
    timeWriteTimer = undefined;
  }
}

function writeUrl(update: Partial<UrlState>): void {
  if (typeof window === "undefined") return;
  const url = new URL(window.location.href);
  const emptyCampaign = import.meta.env.VITE_DATA_SOURCE === "api" ? "" : DEFAULT_CAMPAIGN;
  const next = { ...parseUrlState(url, emptyCampaign), ...update };
  url.searchParams.set("tab", next.tab);
  if (next.campaign) url.searchParams.set("campaign", next.campaign);
  else url.searchParams.delete("campaign");
  if (next.episode) url.searchParams.set("episode", next.episode);
  else url.searchParams.delete("episode");
  if (next.t > 0) url.searchParams.set("t", next.t.toFixed(2));
  else url.searchParams.delete("t");
  window.history.replaceState(window.history.state, "", url);
}

function scheduleTimeWrite(t: number): void {
  if (typeof window === "undefined") return;
  clearTimeWrite();
  timeWriteTimer = window.setTimeout(() => writeUrl({ t }), URL_WRITE_DEBOUNCE_MS);
}

export const useUrlStore = create<AppState>((set, get) => ({
  ...currentUrlState(),
  isPlaying: false,
  setTab: (tab) => {
    set({ tab });
    writeUrl({ tab });
  },
  setCampaign: (campaign) => {
    set({ campaign, episode: "", t: 0 });
    writeUrl({ campaign, episode: "", t: 0 });
  },
  setEpisode: (episode) => {
    set({ episode, t: 0 });
    writeUrl({ episode, t: 0 });
  },
  setT: (t) => set({ t }),
  setPlaying: (isPlaying) => {
    if (isPlaying) {
      clearTimeWrite();
      set({ isPlaying: true });
      return;
    }
    set({ isPlaying: false });
    scheduleTimeWrite(get().t);
  },
  seekTo: (t) => {
    clearTimeWrite();
    set({ isPlaying: false, t });
    scheduleTimeWrite(t);
  },
  syncFromUrl: () => {
    clearTimeWrite();
    set({ ...currentUrlState(), isPlaying: false });
  },
}));
