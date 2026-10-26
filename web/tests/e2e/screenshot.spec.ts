import { expect, test } from "@playwright/test";
import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";

test("captures the paused replay at 3.2 seconds", async ({ page }) => {
  await page.goto("/?campaign=mock_campaign_01&episode=ep_017&t=3.2");
  await expect(page.getByTestId("playback-time")).toHaveText("3.20 s");
  await expect(page.locator(".status-failing")).toContainText("FAILING:");
  await expect(page.getByTestId("joint-readout")).toHaveText(/J[1-7] · TORQUE \d+%/);
  await expect(page.locator("canvas")).toBeVisible();

  const screenshotPath = fileURLToPath(new URL("../../../docs/img/replay.png", import.meta.url));
  mkdirSync(dirname(screenshotPath), { recursive: true });
  const screenshot = await page.screenshot({ path: screenshotPath, fullPage: false });
  const image = PNG.sync.read(screenshot);
  expect(image.width).toBe(1440);
  expect(image.height).toBe(900);
});
