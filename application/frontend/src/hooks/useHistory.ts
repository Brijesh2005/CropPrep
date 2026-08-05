import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { predictionService } from '@/services/prediction';
import { useAuth } from '@/hooks/useAuth';

export function useHistory(params?: { limit?: number; offset?: number }) {
  const { isAuthenticated } = useAuth();
  const limit = params?.limit ?? 50;
  const offset = params?.offset ?? 0;

  return useQuery({
    queryKey: ['history', limit, offset],
    queryFn: () => predictionService.getHistory({ limit, offset }),
    enabled: isAuthenticated,
    staleTime: 1000 * 60,
  });
}

export function useExplainMutation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (predictionId: number) => predictionService.getExplanation(predictionId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['explanation'] });
    },
  });
}

export default useHistory;
