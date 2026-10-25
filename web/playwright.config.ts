import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests/e2e",
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: 1,
  workers: 1,
  timeout: 60_000,
  expect: { timeout: 15_000 },
  reporter: "list",
  use: {
    baseURL: "http://127.0.0.1:4173",
    browserName: "chromium",
    channel: "chrome",
    launchOptions: {
      args: ["--enable-webgl", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
    },
    trace: "retain-on-failure",
  },
  projects: [
    {
      name: "acceptance",
      testMatch: /acceptance\.spec\.ts/,
      use: { viewport: { width: 1440, height: 900 } },
    },
    {
      name: "screenshot",
      testMatch: /screenshot\.spec\.ts/,
      use: { viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: {
    command: "pnpm preview --host 127.0.0.1 --port 4173 --strictPort",
    url: "http://127.0.0.1:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
