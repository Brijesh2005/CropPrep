export * from './api';

export type Theme = 'light' | 'dark' | 'system';

export type Season = 'Kharif' | 'Rabi' | 'Zaid';

export type MapLocation = {
  id?: string;
  lat: number;
  lon: number;
  name?: string;
  village?: string;
  district?: string;
  state?: string;
  distance_km?: number | null;
  distance_m?: number;
};

export type ChartDataPoint = {
  name: string;
  value: number;
  color?: string;
};

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export type Toast = {
  id: string;
  type: ToastType;
  message: string;
  duration?: number;
};

export type NavItem = {
  label: string;
  path: string;
  icon?: string;
  adminOnly?: boolean;
  requireAuth?: boolean;
};
