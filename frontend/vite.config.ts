import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const backend = process.env.LOCALHYDRA_BACKEND_URL ?? "http://127.0.0.1:8765";
const backendHost = new URL(backend).hostname;
if (!["127.0.0.1", "localhost", "[::1]"].includes(backendHost)) {
  throw new Error(`LOCALHYDRA_BACKEND_URL must be loopback, got ${backendHost}`);
}

const proxy = { "/api": { target: backend, changeOrigin: false } };

export default defineConfig({
  plugins: [
    react(),
    {
      // Production bundle may only talk to its own origin; dev keeps HMR unrestricted.
      name: "localhydra-csp",
      apply: "build",
      transformIndexHtml: (html) =>
        html.replace(
          "<head>",
          `<head>\n    <meta http-equiv="Content-Security-Policy" content="default-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'self' data:" />`,
        ),
    },
  ],
  server: { host: "127.0.0.1", port: 5173, strictPort: true, proxy },
  preview: { host: "127.0.0.1", port: 4173, strictPort: true, proxy },
  test: { include: ["tests/**/*.test.ts"], environment: "node" },
});
