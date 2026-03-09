import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// NOTE: update `base` if your repo name is different.
// For a repo `https://github.com/<user>/ToxiTIGS`, GitHub Pages serves from `/ToxiTIGS/`.
export default defineConfig({
  base: "/ToxiTIGS/",
  build: {
    // Emit the production build directly into the repo-level `docs` folder
    // so GitHub Pages can serve it from there.
    outDir: "../docs",
    emptyOutDir: true,
  },
  plugins: [svelte()],
});
