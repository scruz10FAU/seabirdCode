/*
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
})
*/

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

export default defineConfig({
  plugins: [react()],
  server: {
		allowedHosts: ['seabird.targadev.com']
},
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
});
