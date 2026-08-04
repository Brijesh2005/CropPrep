import { create } from 'zustand';
import type { PredictionResponse, ExplanationResponse } from '@/types';

type PredictionState = {
  current: PredictionResponse | null;
  explanation: ExplanationResponse | null;
  selectedLat: number | null;
  selectedLon: number | null;
  year: number;
  season: string;
  setCurrent: (p: PredictionResponse | null) => void;
  setExplanation: (e: ExplanationResponse | null) => void;
  setSelectedLocation: (lat: number, lon: number) => void;
  setYear: (year: number) => void;
  setSeason: (season: string) => void;
  clear: () => void;
};

export const usePredictionStore = create<PredictionState>((set) => ({
  current: null,
  explanation: null,
  selectedLat: null,
  selectedLon: null,
  year: new Date().getFullYear(),
  season: 'Kharif',

  setCurrent: (current) => set({ current }),
  setExplanation: (explanation) => set({ explanation }),
  setSelectedLocation: (lat, lon) => set({ selectedLat: lat, selectedLon: lon }),
  setYear: (year) => set({ year }),
  setSeason: (season) => set({ season }),
  clear: () =>
    set({
      current: null,
      explanation: null,
      selectedLat: null,
      selectedLon: null,
    }),
}));
