import api from './api';
import type {
  PredictionRequest,
  PredictionResponse,
  MapPredictionRequest,
  HistoryPage,
  ExplanationResponse,
  ExplainRequest,
} from '@/types';

export const predictionService = {
  async predict(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await api.post<PredictionResponse>('/predict', request);
    return response.data;
  },

  async predictLocation(request: PredictionRequest): Promise<PredictionResponse> {
    const response = await api.post<PredictionResponse>('/predict/location', request);
    return response.data;
  },

  async predictMap(request: MapPredictionRequest): Promise<PredictionResponse[]> {
    const response = await api.post<PredictionResponse[]>('/predict/map', request);
    return response.data;
  },

  async getHistory(params?: { limit?: number; offset?: number }): Promise<HistoryPage> {
    const response = await api.get<HistoryPage>('/predictions/history', { params });
    return response.data;
  },

  async getExplanation(predictionId: number | string): Promise<ExplanationResponse> {
    const response = await api.get<ExplanationResponse>(`/explain/${predictionId}`);
    return response.data;
  },

  async explain(request: ExplainRequest): Promise<ExplanationResponse> {
    const response = await api.post<ExplanationResponse>('/explain', request);
    return response.data;
  },
};
