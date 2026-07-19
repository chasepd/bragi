import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { apiProxy } from "./src/devProxy";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "../bragi_web/static",
    emptyOutDir: true,
    rollupOptions: {
      output: {
        manualChunks(id: string) {
          if (id.includes("node_modules")) return "vendor";
          return undefined;
        }
      }
    }
  },
  server: {
    proxy: {
      "/api": apiProxy
    }
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts"
  }
});
