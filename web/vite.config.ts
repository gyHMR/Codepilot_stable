import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(() => {
  const backendUrl =
    process.env.CODEPILOT_WEB_BACKEND_URL ?? "http://127.0.0.1:8000";

  return {
    plugins: [react()],
    server: { proxy: { "/api": backendUrl } },
    build: {
      outDir: "../src/codepilot/interfaces/web/static",
      emptyOutDir: true,
      rollupOptions: {
        output: {
          manualChunks: {
            react: ["react", "react-dom", "react-router-dom", "@tanstack/react-query"],
            radix: ["radix-ui"],
            markdown: ["react-markdown", "remark-gfm", "rehype-sanitize"],
          },
        },
      },
    },
    test: { environment: "jsdom", setupFiles: "./src/test/setup.ts" },
  };
});
