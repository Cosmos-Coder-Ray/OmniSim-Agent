import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";

function color(name: string, css: string): [number, number, number] {
  const value = css.match(new RegExp(`${name}:\\s*(#[0-9a-fA-F]{6})`))?.[1];
  if (!value) throw new Error(`Missing color token ${name}.`);
  return [1, 3, 5].map((index) => Number.parseInt(value.slice(index, index + 2), 16)) as [number, number, number];
}

function luminance(channels: [number, number, number]): number {
  const linear = channels.map((channel) => {
    const value = channel / 255;
    return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
  });
  return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2];
}

function contrast(first: [number, number, number], second: [number, number, number]): number {
  const values = [luminance(first), luminance(second)].sort((a, b) => b - a);
  return (values[0] + 0.05) / (values[1] + 0.05);
}

describe("theme contrast", () => {
  it("keeps body and muted text above WCAG AA contrast on page and panel colors", () => {
    const css = readFileSync("src/theme.css", "utf8");
    const page = color("--page", css);
    const panel = color("--panel", css);
    expect(contrast(color("--text", css), page)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(color("--muted", css), page)).toBeGreaterThanOrEqual(4.5);
    expect(contrast(color("--muted", css), panel)).toBeGreaterThanOrEqual(4.5);
  });
});
