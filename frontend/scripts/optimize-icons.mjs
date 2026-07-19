import { mkdir, stat } from "node:fs/promises";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

import sharp from "sharp";

const FRONTEND_ROOT = fileURLToPath(new URL("../", import.meta.url));
const SOURCE_ICON = "assets/app-icon-source.png";
const ICONS = [
  { path: "public/app-icon-512.png", width: 512, height: 512 },
  { path: "public/app-icon-192.png", width: 192, height: 192 },
  { path: "public/apple-touch-icon.png", width: 180, height: 180 }
];

export async function optimizeIcons({ frontendRoot = FRONTEND_ROOT } = {}) {
  const sourcePath = join(frontendRoot, SOURCE_ICON);
  const sourceStat = await stat(sourcePath);
  if (!sourceStat.isFile()) {
    throw new Error(`${SOURCE_ICON} is not a file`);
  }

  const results = [];
  for (const icon of ICONS) {
    const targetPath = join(frontendRoot, icon.path);
    await mkdir(dirname(targetPath), { recursive: true });
    await sharp(sourcePath)
      .resize(icon.width, icon.height, {
        fit: "cover",
        position: "center"
      })
      .png({
        palette: true,
        colors: 128,
        compressionLevel: 9,
        effort: 10,
        dither: 0
      })
      .toFile(targetPath);
    const targetStat = await stat(targetPath);
    results.push({ ...icon, rawBytes: targetStat.size });
  }
  return results;
}

async function main() {
  const results = await optimizeIcons();
  for (const result of results) {
    console.log(
      `optimized ${result.path}: ${result.width}x${result.height}, ` +
        `${result.rawBytes} B`
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
