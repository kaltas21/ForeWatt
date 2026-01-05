import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    // Load env file based on mode (development, production)
    const env = loadEnv(mode, process.cwd(), '');

    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        // Proxy API requests to FastAPI backend during development
        proxy: {
          '/api': {
            target: env.VITE_API_URL || 'http://localhost:8080',
            changeOrigin: true,
            rewrite: (path) => path.replace(/^\/api/, ''),
          },
          '/forecast': {
            target: env.VITE_API_URL || 'http://localhost:8080',
            changeOrigin: true,
          },
          '/history': {
            target: env.VITE_API_URL || 'http://localhost:8080',
            changeOrigin: true,
          },
          '/health': {
            target: env.VITE_API_URL || 'http://localhost:8080',
            changeOrigin: true,
          },
        },
      },
      plugins: [react()],
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
