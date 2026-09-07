import { defineConfig } from 'vite'; // <-- Make sure this line is EXACTLY like this!
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';
import path from 'path';

export default defineConfig({
  base: '/futuristic-ai-reader/', // Make sure this matches your exact repo name
  plugins: [
    react(),
    tailwindcss()
  ],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
