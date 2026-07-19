import { readdir, readFile, stat } from "node:fs/promises";
import { extname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { brotliCompressSync, constants, gzipSync } from "node:zlib";

export async function checkSizeBudget({
  frontendRoot = defaultFrontendRoot(),
  staticRoot = defaultStaticRoot(),
  budgetPath = join(frontendRoot, "size-budget.json"),
  log = console.log
} = {}) {
  const budget = JSON.parse(await readFile(budgetPath, "utf8"));
  const failures = [];
  const measurements = {
    generatedAssets: {},
    generatedAssetChecks: {},
    publicAssets: {}
  };

  await checkGeneratedAssets({
    staticRoot,
    budget,
    failures,
    measurements,
    log
  });
  await checkPublicAssets({
    frontendRoot,
    budget,
    failures,
    measurements,
    log
  });

  return {
    passed: failures.length === 0,
    failures,
    measurements
  };
}

function defaultFrontendRoot() {
  if (import.meta.url.startsWith("file:")) {
    return fileURLToPath(new URL("../", import.meta.url));
  }
  return process.cwd();
}

function defaultStaticRoot() {
  if (import.meta.url.startsWith("file:")) {
    return fileURLToPath(new URL("../../bragi_web/static/", import.meta.url));
  }
  return join(process.cwd(), "../bragi_web/static");
}

function assetMatchesPattern(name, pattern) {
  if (pattern instanceof RegExp) return pattern.test(name);
  return new RegExp(pattern).test(name);
}

async function checkGeneratedAssets({
  staticRoot,
  budget,
  failures,
  measurements,
  log
}) {
  const assetsRoot = join(staticRoot, "assets");
  let entries;
  try {
    entries = await readdir(assetsRoot, { withFileTypes: true });
  } catch (error) {
    if (error && error.code === "ENOENT") {
      failures.push(`missing generated assets directory ${assetsRoot}`);
      return;
    }
    throw error;
  }
  for (const [label, limits] of Object.entries(budget.generatedAssets ?? {})) {
    const extensions = new Set(limits.extensions ?? []);
    const assets = [];
    for (const entry of entries) {
      if (!entry.isFile() || !extensions.has(extname(entry.name))) {
        continue;
      }
      assets.push(await measureGeneratedAsset(join(assetsRoot, entry.name), entry.name));
    }

    if (assets.length === 0) {
      failures.push(
        `missing generated assets for ${label} (${[...extensions].join(", ")})`
      );
      continue;
    }

    const totals = assets.reduce(
      (current, asset) => ({
        rawBytes: current.rawBytes + asset.rawBytes,
        gzipBytes: current.gzipBytes + asset.gzipBytes,
        brotliBytes: current.brotliBytes + asset.brotliBytes
      }),
      { rawBytes: 0, gzipBytes: 0, brotliBytes: 0 }
    );
    measurements.generatedAssets[label] = { totals, assets };

    log(
      `${label} total: raw ${formatBytes(totals.rawBytes)} / ${formatBytes(
        limits.totalRawBytes
      )}, gzip ${formatBytes(totals.gzipBytes)} / ${formatBytes(
        limits.totalGzipBytes
      )}, brotli ${formatBytes(totals.brotliBytes)} / ${formatBytes(
        limits.totalBrotliBytes
      )}`
    );

    assertBudget({
      failures,
      label: `${label} total raw`,
      value: totals.rawBytes,
      limit: limits.totalRawBytes
    });
    assertBudget({
      failures,
      label: `${label} total gzip`,
      value: totals.gzipBytes,
      limit: limits.totalGzipBytes
    });
    assertBudget({
      failures,
      label: `${label} total brotli`,
      value: totals.brotliBytes,
      limit: limits.totalBrotliBytes
    });

    for (const asset of assets) {
      log(
        `${asset.name}: raw ${formatBytes(asset.rawBytes)}, gzip ${formatBytes(
          asset.gzipBytes
        )}, brotli ${formatBytes(asset.brotliBytes)}`
      );
      assertBudget({
        failures,
        label: `${asset.name} raw`,
        value: asset.rawBytes,
        limit: limits.maxFileRawBytes
      });
    }

    if (typeof limits.minFileCount === "number" && assets.length < limits.minFileCount) {
      failures.push(
        `${label} generated ${assets.length} files, expected at least ${limits.minFileCount}`
      );
    }

    for (const pattern of limits.requiredFilePatterns ?? []) {
      if (!assets.some((asset) => assetMatchesPattern(asset.name, pattern))) {
        failures.push(`${label} missing generated asset matching ${pattern}`);
      }
    }
    measurements.generatedAssetChecks[label] = {
      fileCount: assets.length,
      requiredFilePatterns: limits.requiredFilePatterns ?? []
    };
  }
}

async function measureGeneratedAsset(path, name) {
  const source = await readFile(path);
  const gzip = gzipSync(source, { level: 9 });
  const brotli = brotliCompressSync(source, {
    params: { [constants.BROTLI_PARAM_QUALITY]: 11 }
  });
  return {
    name,
    rawBytes: source.length,
    gzipBytes: gzip.length,
    brotliBytes: brotli.length
  };
}

async function checkPublicAssets({
  frontendRoot,
  budget,
  failures,
  measurements,
  log
}) {
  for (const assetBudget of budget.publicAssets ?? []) {
    const displayPath = `public/${assetBudget.path}`;
    const path = join(frontendRoot, displayPath);
    let fileStat;
    try {
      fileStat = await stat(path);
    } catch (error) {
      if (error && error.code === "ENOENT") {
        failures.push(`missing public asset ${displayPath}`);
        continue;
      }
      throw error;
    }
    const source = await readFile(path);
    let dimensions;
    try {
      dimensions = pngDimensions(source);
    } catch (error) {
      failures.push(`${displayPath} is not a valid PNG: ${error.message}`);
      continue;
    }
    measurements.publicAssets[assetBudget.path] = {
      rawBytes: fileStat.size,
      ...dimensions
    };

    log(
      `${assetBudget.path}: raw ${formatBytes(fileStat.size)} / ${formatBytes(
        assetBudget.maxRawBytes
      )}, dimensions ${dimensions.width}x${dimensions.height}`
    );

    assertBudget({
      failures,
      label: displayPath,
      value: fileStat.size,
      limit: assetBudget.maxRawBytes
    });
    if (
      dimensions.width !== assetBudget.width ||
      dimensions.height !== assetBudget.height
    ) {
      failures.push(
        `${displayPath} dimensions ${dimensions.width}x${dimensions.height} ` +
          `exceed expected ${assetBudget.width}x${assetBudget.height}`
      );
    }
  }
}

function pngDimensions(source) {
  const signature = Buffer.from("\x89PNG\r\n\x1a\n", "binary");
  if (source.length < 24 || !source.subarray(0, 8).equals(signature)) {
    throw new Error("PNG asset is missing a valid PNG signature");
  }
  const chunkType = source.subarray(12, 16).toString("ascii");
  if (chunkType !== "IHDR") {
    throw new Error("PNG asset is missing an IHDR chunk");
  }
  return {
    width: source.readUInt32BE(16),
    height: source.readUInt32BE(20)
  };
}

function assertBudget({ failures, label, value, limit }) {
  if (typeof limit !== "number") {
    return;
  }
  if (value > limit) {
    failures.push(
      `${label} ${formatBytes(value)} exceeds limit ${formatBytes(limit)}`
    );
  }
}

function formatBytes(value) {
  return `${value} B`;
}

async function main() {
  const result = await checkSizeBudget();
  if (result.passed) {
    console.log("frontend size budget: passed");
    return;
  }
  console.error("frontend size budget: failed");
  for (const failure of result.failures) {
    console.error(`  ${failure}`);
  }
  process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
