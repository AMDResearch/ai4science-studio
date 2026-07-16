import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// base: './' so the built dist works when served by FastAPI StaticFiles from
// any mount point. The dev server proxies /api to the FastAPI backend so the
// React app can use same-origin relative URLs in both dev and prod.
export default defineConfig({
  plugins: [react()],
  base: './',
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8600',
        changeOrigin: true,
      },
    },
  },
})
