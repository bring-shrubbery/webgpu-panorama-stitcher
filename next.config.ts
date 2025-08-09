import type { NextConfig } from "next";

const dev = process.env.NODE_ENV !== "production";

// Minimal CSP that allows wasm execution + workers + blob/data images.
// Note: 'wasm-unsafe-eval' is required by modern Chromium for WebAssembly.
// We also allow 'unsafe-eval' in dev to keep Next.js HMR happy.
const csp = [
  "default-src 'self'",
  `script-src 'self' 'wasm-unsafe-eval'${
    dev ? " 'unsafe-eval' 'unsafe-inline'" : ""
  }`,
  "worker-src 'self' blob:",
  "img-src 'self' blob: data:",
  "style-src 'self' 'unsafe-inline'",
  "connect-src 'self'",
  "font-src 'self' data:",
].join("; ");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  eslint: {
    ignoreDuringBuilds: true, // Ignore ESLint errors during build
  },
  async headers() {
    return [
      {
        // Global CSP so opencv.js can initialize its WASM
        source: "/:path*",
        headers: [{ key: "Content-Security-Policy", value: csp }],
      },
      {
        // Cache the local OpenCV artifacts aggressively
        source: "/opencv/:path*",
        headers: [
          {
            key: "Cache-Control",
            value: "public, max-age=31536000, immutable",
          },
        ],
      },
    ];
  },
};

export default nextConfig;
