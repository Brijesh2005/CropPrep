import api from './api';
import type { ExplainRequest, ExplanationResponse } from '@/types';

export const explainabilityService = {
  async explain(request: ExplainRequest): Promise<ExplanationResponse> {
    const response = await api.post<ExplanationResponse>('/explain', request);
    return response.data;
  },

  async getByPrediction(predictionId: number | string): Promise<ExplanationResponse> {
    const response = await api.get<ExplanationResponse>(`/explain/${predictionId}`);
    return response.data;
  },
};
