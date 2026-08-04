import { create } from 'zustand';
import { DEFAULT_MAP_CENTER, DEFAULT_MAP_ZOOM } from '@/config';
import type { LocationResponse } from '@/types';

type MapState = {
  center: [number, number];
  zoom: number;
  selected: { lat: number; lon: number } | null;
  locations: LocationResponse[];
  showBoundaries: boolean;
  setCenter: (center: [number, number]) => void;
  setZoom: (zoom: number) => void;
  setSelected: (lat: number, lon: number) => void;
  clearSelected: () => void;
  setLocations: (locations: LocationResponse[]) => void;
  setShowBoundaries: (show: boolean) => void;
};

export const useMapStore = create<MapState>((set) => ({
  center: DEFAULT_MAP_CENTER,
  zoom: DEFAULT_MAP_ZOOM,
  selected: null,
  locations: [],
  showBoundaries: true,

  setCenter: (center) => set({ center }),
  setZoom: (zoom) => set({ zoom }),
  setSelected: (lat, lon) => set({ selected: { lat, lon }, center: [lat, lon] }),
  clearSelected: () => set({ selected: null }),
  setLocations: (locations) => set({ locations }),
  setShowBoundaries: (showBoundaries) => set({ showBoundaries }),
}));
