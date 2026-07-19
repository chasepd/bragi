import { readdir, readFile, writeFile } from "node:fs/promises";
import { extname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { gzip } from "node:zlib";
import { promisify } from "node:util";

const gzipAsync = promisify(gzip);
const assetsDir = fileURLToPath(
  new URL("../../bragi_web/static/assets/", import.meta.url)
);
const compressibleExtensions = new Set([".css", ".js"]);

const entries = await readdir(assetsDir, { withFileTypes: true });
const compressedAssets = [];

for (const entry of entries) {
  if (!entry.isFile() || !compressibleExtensions.has(extname(entry.name))) {
    continue;
  }
  const assetPath = join(assetsDir, entry.name);
  const source = await readFile(assetPath);
  const compressed = await gzipAsync(source, { level: 9 });
  await writeFile(`${assetPath}.gz`, compressed);
  compressedAssets.push({
    name: entry.name,
    rawBytes: source.length,
    gzipBytes: compressed.length
  });
}

if (compressedAssets.length === 0) {
  throw new Error(`No JS or CSS assets found to compress in ${assetsDir}`);
}

for (const asset of compressedAssets) {
  console.log(
    `compressed ${asset.name}: ${asset.rawBytes} B -> ${asset.gzipBytes} B`
  );
}
