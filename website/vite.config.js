import { svelte } from "@sveltejs/vite-plugin-svelte";
import { defineConfig } from "vite";

// NOTE: update `base` if your repo name is different.
// For a repo `https://github.com/<user>/ToxiTIGS`, GitHub Pages serves from `/ToxiTIGS/`.
export default defineConfig({
  base: "/ToxiTIGS/",
  plugins: [svelte()],
});
