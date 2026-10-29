import { expect, test, type Page } from "@playwright/test";
import { PNG } from "pngjs";

declare global {
  interface Window {
    __historyCalls: string[];
  }
}

async function recordHistory(page: Page) {
  await page.addInitScript(() => {
    window.__historyCalls = [];
    window.history.replaceState = new Proxy(window.history.replaceState, {
      apply(target, thisArg, args) {
        window.__historyCalls.push("replaceState");
        return Reflect.apply(target, thisArg, args);
      },
    });
    window.history.pushState = new Proxy(window.history.pushState, {
      apply(target, thisArg, args) {
        window.__historyCalls.push("pushState");
        return Reflect.apply(target, thisArg, args);
      },
    });
  });
}

test("loads the mock replay and renders its status, heat legend, joint readout, and canvas", async ({ page }) => {
  const requestUrls: string[] = [];
  page.on("request", (request) => requestUrls.push(request.url()));
  await page.goto("/");

  await expect(page.locator(".status-failing")).toContainText("FAILING:");
  await expect(page.getByLabel("Torque ratio heat legend")).toBeVisible();
  await expect(page.getByTestId("joint-readout")).toHaveText(/J[1-7] · TORQUE \d+%/);
  await expect(page.locator("canvas")).toBeVisible();
  await expect(page.locator("[data-source='mock']")).toContainText("SOURCE · MOCK");

  const canvas = page.locator("canvas");
  const bounds = await canvas.boundingBox();
  expect(bounds?.width).toBeGreaterThan(500);
  expect(bounds?.height).toBeGreaterThan(250);
  const image = PNG.sync.read(await canvas.screenshot());
  const colors = new Set<string>();
  for (let y = 0; y < image.height; y += 8) {
    for (let x = 0; x < image.width; x += 8) {
      const offset = (y * image.width + x) * 4;
      colors.add(`${image.data[offset]}-${image.data[offset + 1]}-${image.data[offset + 2]}`);
    }
  }
  expect(colors.size).toBeGreaterThan(12);

  const origin = new URL(page.url()).origin;
  expect(requestUrls.length).toBeGreaterThan(0);
  expect(requestUrls.every((url) => new URL(url).origin === origin)).toBe(true);
});

test("plays without history writes, then saves time after pause debounce", async ({ page }) => {
  await recordHistory(page);
  await page.goto("/");
  const time = page.getByTestId("playback-time");
  await expect(time).toHaveText("0.00 s");
  await page.evaluate(() => { window.__historyCalls.length = 0; });

  await page.getByRole("button", { name: "Play" }).click();
  await expect(page.getByRole("button", { name: "Pause" })).toBeVisible();
  await expect.poll(async () => Number.parseFloat((await time.textContent()) ?? "0"))
    .toBeGreaterThan(0.25);
  expect(await page.evaluate(() => window.__historyCalls)).toEqual([]);

  await page.getByRole("button", { name: "Pause" }).click();
  await expect.poll(async () => page.evaluate(() => window.__historyCalls.length)).toBe(1);
  const urlTime = Number(new URL(page.url()).searchParams.get("t"));
  const displayTime = Number.parseFloat((await time.textContent()) ?? "0");
  expect(urlTime).toBeCloseTo(displayTime, 2);
  expect(await page.evaluate(() => window.__historyCalls)).toEqual(["replaceState"]);
});

test("supports keyboard and event-marker seeking, and restores time from the URL", async ({ page }) => {
  await page.goto("/");
  const slider = page.getByRole("slider", { name: "Seek replay time" });
  await slider.focus();
  await page.keyboard.press("ArrowRight");
  await expect.poll(() => page.evaluate(() => new URL(location.href).searchParams.get("t")))
    .toBe("0.01");

  await page.getByRole("button", { name: /Seek to torque saturation at 2\.00 seconds/ }).click();
  await expect(page.getByTestId("playback-time")).toHaveText("2.00 s");
  await expect.poll(() => page.evaluate(() => new URL(location.href).searchParams.get("t")))
    .toBe("2.00");

  await page.goto("/?campaign=mock_campaign_01&episode=ep_017&t=3.2");
  await expect(page.getByTestId("playback-time")).toHaveText("3.20 s");
});

test("switches tabs by keyboard and keeps the replay usable at 360 pixels", async ({ page }) => {
  await page.goto("/");
  const replayTab = page.getByRole("tab", { name: "Replay" });
  await replayTab.focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.getByRole("tab", { name: "Live Campaign" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("heading", { name: "Live Campaign" })).toBeVisible();
  await expect(page.getByText("ep_016")).toBeVisible();

  await page.setViewportSize({ width: 360, height: 800 });
  await page.getByRole("tab", { name: "Replay" }).click();
  await expect(page.locator("canvas")).toBeVisible();
  const pageWidth = await page.evaluate(() => document.documentElement.scrollWidth);
  expect(pageWidth).toBeLessThanOrEqual(360);
  const canvasFrame = await page.locator(".canvas-frame").boundingBox();
  expect(canvasFrame?.width).toBeGreaterThan(300);
  expect(canvasFrame?.height).toBeGreaterThan(220);
});

test("renders mock summaries across workspace tabs and opens an episode in replay", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("tab", { name: "Live Campaign" }).click();
  await expect(page.getByRole("heading", { name: "Live Campaign" })).toBeVisible();
  await page.getByRole("tab", { name: "Failure Atlas" }).click();
  await expect(page.getByText("tracking_error:global")).toBeVisible();
  await page.getByRole("tab", { name: "Evidence" }).click();
  await expect(page.getByText("joint torque limit", { exact: true })).toBeVisible();
  await expect(page.getByText("joint_reflex", { exact: false })).toBeVisible();
  await page.getByRole("tab", { name: "Safe Envelope" }).click();
  await expect(page.getByText("2 kg", { exact: true })).toBeVisible();
  await expect(page.getByRole("link", { name: "safe_envelope.json" })).toBeVisible();
  await page.getByRole("tab", { name: "Benchmarks" }).click();
  await expect(page.getByRole("row", { name: /nemotron/ })).toBeVisible();
  await page.getByRole("tab", { name: "Live Campaign" }).click();
  await page.getByRole("button", { name: "ep_017" }).click();
  await expect(page.getByRole("tab", { name: "Replay" })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByLabel("Select episode")).toHaveValue("ep_017");
});
