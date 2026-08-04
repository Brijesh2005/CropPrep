import api from './api';
import type { BoundaryResponse, LocationResponse } from '@/types';

export const gisService = {
  async getLocations(limit = 200): Promise<LocationResponse[]> {
    const response = await api.get<LocationResponse[]>('/gis/locations', {
      params: { limit },
    });
    return response.data;
  },

  async getLocation(locationId: string): Promise<LocationResponse> {
    const response = await api.get<LocationResponse>(`/gis/location/${locationId}`);
    return response.data;
  },

  async searchLocations(query: string, limit = 10): Promise<LocationResponse[]> {
    const response = await api.post<LocationResponse[]>('/gis/search', {
      query,
      limit,
    });
    return response.data;
  },

  async getBoundaries(): Promise<BoundaryResponse[]> {
    const response = await api.get<BoundaryResponse[]>('/gis/boundaries');
    return response.data;
  },
};
