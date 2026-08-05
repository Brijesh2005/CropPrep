import { useCallback, useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { predictionService } from '@/services/prediction';
import { getErrorMessage } from '@/services/api';
import { usePredictionStore } from '@/store/predictionStore';
import { useUiStore } from '@/store/uiStore';
import type { PredictionRequest, PredictionResponse, ExplanationResponse } from '@/types';

export function usePrediction() {
  const queryClient = useQueryClient();
  const setCurrent = usePredictionStore((s) => s.setCurrent);
  const setExplanation = usePredictionStore((s) => s.setExplanation);
  const current = usePredictionStore((s) => s.current);
  const explanation = usePredictionStore((s) => s.explanation);
  const addToast = useUiStore((s) => s.addToast);

  const [error, setError] = useState<string | null>(null);

  const predictMutation = useMutation({
    mutationFn: (request: PredictionRequest) => predictionService.predict(request),
    onSuccess: (result) => {
      setCurrent(result);
      setError(null);
      addToast('success', `Recommended crop: ${result.recommended_crop}`);
      queryClient.invalidateQueries({ queryKey: ['history'] });
    },
    onError: (err) => {
      const msg = getErrorMessage(err, 'Prediction failed');
      setError(msg);
      addToast('error', msg);
    },
  });

  const mapMutation = useMutation({
    mutationFn: (points: PredictionRequest[]) => predictionService.predictMap({ points }),
    onError: (err) => {
      const msg = getErrorMessage(err, 'Map prediction failed');
      setError(msg);
      addToast('error', msg);
    },
  });

  const explainMutation = useMutation({
    mutationFn: (predictionId: number | string) => predictionService.getExplanation(predictionId),
    onSuccess: (result) => {
      setExplanation(result);
      setError(null);
    },
    onError: (err) => {
      const msg = getErrorMessage(err, 'Explanation failed');
      setError(msg);
      addToast('error', msg);
    },
  });

  const predict = useCallback(
    async (request: PredictionRequest): Promise<PredictionResponse> => {
      return predictMutation.mutateAsync(request);
    },
    [predictMutation]
  );

  const predictMap = useCallback(
    async (requests: PredictionRequest[]): Promise<PredictionResponse[]> => {
      return mapMutation.mutateAsync(requests);
    },
    [mapMutation]
  );

  const getExplanation = useCallback(
    async (predictionId: number | string): Promise<ExplanationResponse> => {
      return explainMutation.mutateAsync(predictionId);
    },
    [explainMutation]
  );

  return {
    isLoading: predictMutation.isPending || mapMutation.isPending || explainMutation.isPending,
    prediction: current,
    explanation,
    error,
    predict,
    predictMap,
    getExplanation,
  };
}

export default usePrediction;
