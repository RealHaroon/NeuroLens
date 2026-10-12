import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';

const root = fileURLToPath(new URL('.', import.meta.url));

export default defineConfig({
  // Relative base so the build works on GitHub Pages project sites and Netlify alike.
  base: './',
  server: {
    fs: {
      // Vite's allow-list check rejects every file when the project path contains a ':'
      // (e.g. "/run/media/…/Local Disk (D:)/…"), so the dev server would answer 403 for
      // everything. Relax it only in that case; the dev server listens on localhost only.
      strict: !root.includes(':'),
    },
  },
  build: {
    // The two big chunks are single vendor libraries (TF.js in the entry chunk,
    // three.js in the lazily loaded scene chunk), ~140-150 kB gzipped each.
    chunkSizeWarningLimit: 650,
  },
});
