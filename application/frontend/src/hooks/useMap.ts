import { useCallback } from 'react';
import { useQuery } from '@tanstack/react-query';
import { gisService } from '@/services/gis';
import { useMapStore } from '@/store/mapStore';
import { usePredictionStore } from '@/store/predictionStore';

export function useMap() {
  const center = useMapStore((s) => s.center);
  const zoom = useMapStore((s) => s.zoom);
  const selected = useMapStore((s) => s.selected);
  const locations = useMapStore((s) => s.locations);
  const showBoundaries = useMapStore((s) => s.showBoundaries);
  const setCenter = useMapStore((s) => s.setCenter);
  const setZoom = useMapStore((s) => s.setZoom);
  const setSelected = useMapStore((s) => s.setSelected);
  const clearSelected = useMapStore((s) => s.clearSelected);
  const setLocations = useMapStore((s) => s.setLocations);
  const setShowBoundaries = useMapStore((s) => s.setShowBoundaries);
  const setSelectedLocation = usePredictionStore((s) => s.setSelectedLocation);

  const locationsQuery = useQuery({
    queryKey: ['gis', 'locations'],
    queryFn: () => gisService.getLocations(300),
    staleTime: 1000 * 60 * 30,
  });

  const boundariesQuery = useQuery({
    queryKey: ['gis', 'boundaries'],
    queryFn: () => gisService.getBoundaries(),
    staleTime: 1000 * 60 * 60,
  });

  // Sync fetched locations into the map store
  if (locationsQuery.data && locationsQuery.data !== locations) {
    setLocations(locationsQuery.data);
  }

  const selectPoint = useCallback(
    (lat: number, lon: number) => {
      setSelected(lat, lon);
      setSelectedLocation(lat, lon);
      setCenter([lat, lon]);
    },
    [setSelected, setSelectedLocation, setCenter]
  );

  const goToMyLocation = useCallback(() => {
    if (!navigator.geolocation) return Promise.reject(new Error('Geolocation not supported'));
    return new Promise<{ lat: number; lon: number }>((resolve, reject) => {
      navigator.geolocation.getCurrentPosition(
        (pos) => {
          const lat = pos.coords.latitude;
          const lon = pos.coords.longitude;
          selectPoint(lat, lon);
          resolve({ lat, lon });
        },
        (err) => reject(err),
        { enableHighAccuracy: true, timeout: 10000 }
      );
    });
  }, [selectPoint]);

  const search = useCallback(async (query: string) => {
    if (!query.trim()) return [];
    return gisService.searchLocations(query.trim(), 10);
  }, []);

  return {
    center,
    zoom,
    selected,
    locations: locationsQuery.data ?? locations,
    boundaries: boundariesQuery.data ?? [],
    showBoundaries,
    isLoadingLocations: locationsQuery.isLoading,
    locationsError: locationsQuery.error,
    setCenter,
    setZoom,
    selectPoint,
    clearSelected,
    setShowBoundaries,
    goToMyLocation,
    search,
  };
}

export default useMap;
