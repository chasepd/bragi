import { mkdtemp, rm, mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { describe, expect, it } from "vitest";

import { checkSizeBudget } from "./check-size-budget.mjs";

function pngHeader(width, height, extraBytes = 0) {
  const header = Buffer.alloc(24 + extraBytes);
  Buffer.from("\x89PNG\r\n\x1a\n", "binary").copy(header, 0);
  header.writeUInt32BE(13, 8);
  header.write("IHDR", 12, "ascii");
  header.writeUInt32BE(width, 16);
  header.writeUInt32BE(height, 20);
  return header;
}

async function withFixture(callback) {
  const root = await mkdtemp(join(tmpdir(), "bragi-size-budget-"));
  try {
    const frontendRoot = join(root, "frontend");
    const staticRoot = join(root, "static");
    await mkdir(join(frontendRoot, "public"), { recursive: true });
    await mkdir(join(staticRoot, "assets"), { recursive: true });
    await callback({ frontendRoot, staticRoot });
  } finally {
    await rm(root, { force: true, recursive: true });
  }
}

async function writeBudget(frontendRoot, budget) {
  await writeFile(
    join(frontendRoot, "size-budget.json"),
    `${JSON.stringify(budget, null, 2)}\n`,
    "utf8"
  );
}

const BASE_BUDGET = {
  generatedAssets: {
    javascript: {
      extensions: [".js"],
      totalRawBytes: 1000,
      totalGzipBytes: 1000,
      totalBrotliBytes: 1000,
      maxFileRawBytes: 1000
    },
    css: {
      extensions: [".css"],
      totalRawBytes: 1000,
      totalGzipBytes: 1000,
      totalBrotliBytes: 1000,
      maxFileRawBytes: 1000
    }
  },
  publicAssets: [
    {
      path: "app-icon-512.png",
      width: 512,
      height: 512,
      maxRawBytes: 100
    },
    {
      path: "app-icon-192.png",
      width: 192,
      height: 192,
      maxRawBytes: 100
    },
    {
      path: "apple-touch-icon.png",
      width: 180,
      height: 180,
      maxRawBytes: 100
    }
  ]
};

async function writePassingAssets({ frontendRoot, staticRoot }) {
  await writeFile(join(staticRoot, "assets", "index.js"), "console.log('ok');");
  await writeFile(join(staticRoot, "assets", "index.css"), "body{color:#111}");
  await writeFile(
    join(frontendRoot, "public", "app-icon-512.png"),
    pngHeader(512, 512)
  );
  await writeFile(
    join(frontendRoot, "public", "app-icon-192.png"),
    pngHeader(192, 192)
  );
  await writeFile(
    join(frontendRoot, "public", "apple-touch-icon.png"),
    pngHeader(180, 180)
  );
}

describe("checkSizeBudget", () => {
  it("passes when generated assets and public icons are under budget", async () => {
    await withFixture(async ({ frontendRoot, staticRoot }) => {
      await writeBudget(frontendRoot, BASE_BUDGET);
      await writePassingAssets({ frontendRoot, staticRoot });
      const messages = [];

      const result = await checkSizeBudget({
        frontendRoot,
        staticRoot,
        log: (message) => messages.push(message)
      });

      expect(result.passed).toBe(true);
      expect(result.failures).toEqual([]);
      expect(messages.join("\n")).toContain("javascript total");
      expect(messages.join("\n")).toContain("app-icon-512.png");
    });
  });

  it("fails with actionable generated asset measurements", async () => {
    await withFixture(async ({ frontendRoot, staticRoot }) => {
      await writeBudget(frontendRoot, {
        ...BASE_BUDGET,
        generatedAssets: {
          ...BASE_BUDGET.generatedAssets,
          javascript: {
            ...BASE_BUDGET.generatedAssets.javascript,
            totalRawBytes: 10,
            maxFileRawBytes: 10
          }
        }
      });
      await writePassingAssets({ frontendRoot, staticRoot });

      const result = await checkSizeBudget({ frontendRoot, staticRoot, log: () => {} });

      expect(result.passed).toBe(false);
      expect(result.failures.join("\n")).toContain("javascript total raw");
      expect(result.failures.join("\n")).toContain("index.js raw");
      expect(result.failures.join("\n")).toContain("limit 10 B");
    });
  });

  it("fails when expected split chunks are missing", async () => {
    await withFixture(async ({ frontendRoot, staticRoot }) => {
      await writeBudget(frontendRoot, {
        ...BASE_BUDGET,
        generatedAssets: {
          ...BASE_BUDGET.generatedAssets,
          javascript: {
            ...BASE_BUDGET.generatedAssets.javascript,
            minFileCount: 2,
            requiredFilePatterns: ["^index-.*\\.js$", "^panel-media-.*\\.js$"]
          }
        }
      });
      await writePassingAssets({ frontendRoot, staticRoot });

      const result = await checkSizeBudget({ frontendRoot, staticRoot, log: () => {} });

      expect(result.passed).toBe(false);
      expect(result.failures).toContain("javascript generated 1 files, expected at least 2");
      expect(result.failures).toContain("javascript missing generated asset matching ^panel-media-.*\\.js$");
    });
  });

  it("fails when an expected icon is missing", async () => {
    await withFixture(async ({ frontendRoot, staticRoot }) => {
      await writeBudget(frontendRoot, BASE_BUDGET);
      await writePassingAssets({ frontendRoot, staticRoot });
      await rm(join(frontendRoot, "public", "apple-touch-icon.png"));

      const result = await checkSizeBudget({ frontendRoot, staticRoot, log: () => {} });

      expect(result.passed).toBe(false);
      expect(result.failures).toContain(
        "missing public asset public/apple-touch-icon.png"
      );
    });
  });

  it("fails when a PNG has the wrong dimensions", async () => {
    await withFixture(async ({ frontendRoot, staticRoot }) => {
      await writeBudget(frontendRoot, BASE_BUDGET);
      await writePassingAssets({ frontendRoot, staticRoot });
      await writeFile(
        join(frontendRoot, "public", "app-icon-192.png"),
        pngHeader(512, 512)
      );

      const result = await checkSizeBudget({ frontendRoot, staticRoot, log: () => {} });

      expect(result.passed).toBe(false);
      expect(result.failures).toContain(
        "public/app-icon-192.png dimensions 512x512 exceed expected 192x192"
      );
    });
  });
});
