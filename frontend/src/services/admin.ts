import api from './api';
import type { AdminDashboard, AdminStatistics, MonitoringMetrics } from '@/types';

export const adminService = {
  async getDashboard(): Promise<AdminDashboard> {
    const response = await api.get<AdminDashboard>('/admin/dashboard');
    return response.data;
  },

  async getStatistics(): Promise<AdminStatistics> {
    const response = await api.get<AdminStatistics>('/admin/statistics');
    return response.data;
  },

  async retrain(): Promise<{ message: string; started: boolean }> {
    const response = await api.post<{ message: string; started: boolean }>('/admin/retrain');
    return response.data;
  },

  async getMetrics(): Promise<MonitoringMetrics> {
    const response = await api.get<MonitoringMetrics>('/monitoring/metrics');
    return response.data;
  },
};
