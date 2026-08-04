export const APP_NAME = 'CropFusion';
export const APP_VERSION = '1.0.0';
export const APP_DESCRIPTION =
  'AI-Powered Agricultural Decision Support System for Location-Based Multi-Crop Recommendation and Yield Prediction';

// Unset (local `npm run dev`) -> direct to the API origin.
// Empty string (baked by Docker) -> same-origin, nginx proxies `/api` -> backend.
const _apiBase = import.meta.env.VITE_API_BASE_URL;
export const API_BASE_URL =
  _apiBase === undefined ? 'http://localhost:8000' : _apiBase.replace(/\/+$/, '');
export const API_PREFIX = '/api/v1';

export const DEFAULT_MAP_CENTER: [number, number] = [12.9141, 74.856]; // Mangalore, Dakshina Kannada
export const DEFAULT_MAP_ZOOM = 10;
export const MAX_MAP_ZOOM = 18;
export const MIN_MAP_ZOOM = 8;

export const DAKSHINA_KANNADA_BOUNDS = {
  north: 13.2,
  south: 12.5,
  east: 75.5,
  west: 74.5,
};

export const SUPPORTED_SEASONS = ['Kharif', 'Rabi', 'Zaid'] as const;
export const SUPPORTED_YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025] as const;

export const CROP_COLORS: Record<string, string> = {
  Rice: '#22c55e',
  Wheat: '#eab308',
  Maize: '#f97316',
  Coconut: '#84cc16',
  Arecanut: '#a3e635',
  Banana: '#facc15',
  Pepper: '#ef4444',
  Coffee: '#78350f',
  Cashew: '#d97706',
  default: '#6b7280',
};

export const TOKEN_STORAGE_KEY = 'cropfusion_tokens';
export const THEME_STORAGE_KEY = 'cropfusion_theme';
export const USER_PREFERENCES_KEY = 'cropfusion_preferences';

export const RATE_LIMITS = {
  predictions: 30,
  history: 60,
  gis: 120,
};

export const CACHE_TTL = {
  predictions: 3600,
  locations: 86400,
  features: 86400,
};
