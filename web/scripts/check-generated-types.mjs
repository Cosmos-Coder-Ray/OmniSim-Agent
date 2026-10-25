import { readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import openapiTS, { astToString } from "openapi-typescript";

const outputPath = fileURLToPath(new URL("../src/api/openapi.generated.ts", import.meta.url));
const generated = astToString(await openapiTS(new URL("../src/api/openapi.json", import.meta.url)));
if (process.argv.includes("--write")) {
  await writeFile(outputPath, generated, "utf8");
  console.log("Generated API types.");
} else {
  const checkedIn = await readFile(outputPath, "utf8");
  if (generated !== checkedIn) {
    console.error("Generated API types are stale. Run pnpm generate:api.");
    process.exitCode = 1;
  } else {
    console.log("Generated API types are current.");
  }
}
