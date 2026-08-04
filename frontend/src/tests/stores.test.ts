import { describe, it, expect, beforeEach } from 'vitest';
import { useMapStore } from '@/store/mapStore';
import { useUiStore } from '@/store/uiStore';
import { usePredictionStore } from '@/store/predictionStore';
import { useThemeStore } from '@/store/themeStore';
import { THEME_STORAGE_KEY } from '@/config';

describe('mapStore', () => {
  beforeEach(() => useMapStore.setState({ selected: null, locations: [] }));

  it('selects a point and re-centers', () => {
    useMapStore.getState().setSelected(12.9, 74.8);
    const state = useMapStore.getState();
    expect(state.selected).toEqual({ lat: 12.9, lon: 74.8 });
    expect(state.center).toEqual([12.9, 74.8]);
  });

  it('clears the selection', () => {
    useMapStore.getState().setSelected(1, 2);
    useMapStore.getState().clearSelected();
    expect(useMapStore.getState().selected).toBeNull();
  });

  it('stores locations', () => {
    const loc = [{ id: 'v1', lat: 1, lon: 2, name: 'Village', admin: {} }];
    useMapStore.getState().setLocations(loc);
    expect(useMapStore.getState().locations).toEqual(loc);
  });
});

describe('uiStore', () => {
  it('adds and removes toasts', () => {
    useUiStore.getState().addToast('success', 'hello', 0);
    expect(useUiStore.getState().toasts).toHaveLength(1);
    const id = useUiStore.getState().toasts[0].id;
    useUiStore.getState().removeToast(id);
    expect(useUiStore.getState().toasts).toHaveLength(0);
  });

  it('toggles the sidebar', () => {
    expect(useUiStore.getState().sidebarOpen).toBe(false);
    useUiStore.getState().toggleSidebar();
    expect(useUiStore.getState().sidebarOpen).toBe(true);
  });
});

describe('predictionStore', () => {
  it('stores selected location and prediction', () => {
    usePredictionStore.getState().setSelectedLocation(10, 20);
    expect(usePredictionStore.getState().selectedLat).toBe(10);
    expect(usePredictionStore.getState().selectedLon).toBe(20);

    const pred = {
      prediction_id: 1,
      location: {},
      coordinates: {},
      recommended_crop: 'Rice',
      expected_yield: 5000,
      confidence: 0.9,
      crop_probs: { Rice: 0.9 },
      model_version: 'v1',
      inference_time_ms: 10,
    };
    usePredictionStore.getState().setCurrent(pred);
    expect(usePredictionStore.getState().current?.recommended_crop).toBe('Rice');
  });

  it('clears state', () => {
    usePredictionStore.getState().setSelectedLocation(1, 2);
    usePredictionStore.getState().setCurrent({
      prediction_id: 1,
      location: {},
      coordinates: {},
      recommended_crop: 'Rice',
      expected_yield: null,
      confidence: 0.9,
      crop_probs: {},
      model_version: '',
      inference_time_ms: 0,
    });
    usePredictionStore.getState().clear();
    const s = usePredictionStore.getState();
    expect(s.current).toBeNull();
    expect(s.selectedLat).toBeNull();
  });
});

describe('themeStore', () => {
  beforeEach(() => localStorage.removeItem(THEME_STORAGE_KEY));

  it('defaults to system and resolves to light in tests', () => {
    useThemeStore.getState().init();
    expect(useThemeStore.getState().theme).toBe('system');
    expect(useThemeStore.getState().resolved).toBe('light');
  });

  it('persists and applies theme changes', () => {
    useThemeStore.getState().setTheme('dark');
    expect(useThemeStore.getState().resolved).toBe('dark');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(localStorage.getItem(THEME_STORAGE_KEY)).toBe('dark');
  });
});
