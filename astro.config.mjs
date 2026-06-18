// astro.config.mjs
import { defineConfig } from 'astro/config';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://josephmclean118.github.io',
  base: '/blogpage/',

  vite: {
    plugins: [tailwindcss()]
  }
});