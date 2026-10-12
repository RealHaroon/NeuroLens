import { defineConfig } from 'vite';

export default defineConfig({
  // Relative base so the build works on GitHub Pages project sites and Netlify alike.
  base: './',
  build: {
    // The two big chunks are single vendor libraries (TF.js in the entry chunk,
    // three.js in the lazily loaded scene chunk), ~140-150 kB gzipped each.
    chunkSizeWarningLimit: 650,
  },
});
