# Frontend

The CropFusion UI is a **React 19 + TypeScript** single-page application built
with Vite. It is a progressive web app (PWA) with offline caching.

## Stack

- React 19, React Router 6, React Query (TanStack), Zustand
- Tailwind CSS, Framer Motion, Recharts / Chart.js / react-chartjs-2
- Leaflet (react-leaflet) for maps
- React Hook Form + zod validation
- Vitest + Testing Library for tests
- ESLint + Prettier for linting/formatting

## Layout

```
frontend/
├── src/
│   ├── app/           # entry, routes, providers
│   ├── components/    # reusable UI components
│   ├── features/      # feature modules
│   └── ...
├── public/
│   ├── manifest.json  # PWA manifest
│   └── sw.js          # service worker
├── vite.config.ts
└── package.json
```

## Commands

```bash
cd frontend
npm ci            # install
npm run dev       # dev server (Vite)
npm run build     # production build -> dist/
npm test          # unit tests (vitest)
npm run test:coverage
npm run lint      # eslint
npm run format    # prettier
```

## API configuration

The backend URL is configured with the `VITE_API_BASE_URL` environment
variable at build time:

```bash
VITE_API_BASE_URL=/api npm run build
```

In the Docker image the nginx container proxies `/api` to the backend service
(see `nginx/nginx.conf`), so the UI and API share the same origin.

## PWA

- `public/manifest.json` declares the app name, icons, theme and display mode.
- `public/sw.js` provides offline caching for static assets and API fallbacks.

## Production build

```bash
# Docker (recommended)
docker build -f Dockerfile.frontend -t cropfusion/frontend .

# Local
npm run build && npx vite preview
```
