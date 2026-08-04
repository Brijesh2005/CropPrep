import React from 'react';
import { Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { predictionService } from '@/services/prediction';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { FeatureChart } from '@/components/explainability/FeatureChart';
import { TemporalTimeline } from '@/components/explainability/TemporalTimeline';
import { ReasoningPanel } from '@/components/explainability/ReasoningPanel';
import { getErrorMessage } from '@/services/api';
import { confidenceValue, formatPercent, formatYield } from '@/utils/format';

export function ExplainPage() {
  const { predictionId } = useParams<{ predictionId: string }>();

  const {
    data: explanation,
    isLoading,
    error,
  } = useQuery({
    queryKey: ['explanation', predictionId],
    queryFn: () => predictionService.getExplanation(Number(predictionId)),
    enabled: !!predictionId,
    retry: 1,
  });

  const handleExport = () => {
    if (!explanation) return;
    const blob = new Blob([JSON.stringify(explanation, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cropfusion-explanation-${predictionId}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Prediction explanation
          </h1>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
            Why the model made this recommendation for prediction #{predictionId}.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleExport} disabled={!explanation}>
            Export JSON
          </Button>
          <Link to="/history">
            <Button variant="ghost">Back to history</Button>
          </Link>
        </div>
      </div>

      {isLoading && <LoadingSpinner label="Generating explanation…" className="py-16" />}

      {error && (
        <Card>
          <CardBody>
            <p className="text-sm text-red-600" role="alert">
              {getErrorMessage(error, 'Could not load the explanation.')}
            </p>
          </CardBody>
        </Card>
      )}

      {explanation && (
        <>
          {/* Summary */}
          <Card>
            <CardHeader className="flex flex-row items-center justify-between gap-4">
              <CardTitle>Summary</CardTitle>
              {explanation.observation_id && (
                <Badge variant="neutral">Obs: {explanation.observation_id}</Badge>
              )}
            </CardHeader>
            <CardBody className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="rounded-xl border border-gray-100 dark:border-gray-700 p-4">
                <p className="text-xs uppercase tracking-wide text-gray-500">Recommended crop</p>
                <p className="mt-1 text-2xl font-bold text-gray-900 dark:text-white">
                  🌾 {explanation.crop || '—'}
                </p>
              </div>
              <div className="rounded-xl border border-gray-100 dark:border-gray-700 p-4">
                <p className="text-xs uppercase tracking-wide text-gray-500">Expected yield</p>
                <p className="mt-1 text-2xl font-bold text-gray-900 dark:text-white">
                  {formatYield(explanation.yield_prediction)}
                </p>
              </div>
              <div className="rounded-xl border border-gray-100 dark:border-gray-700 p-4">
                <p className="text-xs uppercase tracking-wide text-gray-500">Confidence</p>
                <p className="mt-1 text-2xl font-bold text-gray-900 dark:text-white">
                  {formatPercent(confidenceValue(explanation.confidence))}
                </p>
              </div>
            </CardBody>
          </Card>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Feature contributions (SHAP)</CardTitle>
              </CardHeader>
              <CardBody>
                <FeatureChart topFeatures={explanation.top_features} />
              </CardBody>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Temporal importance</CardTitle>
              </CardHeader>
              <CardBody>
                <TemporalTimeline importantDates={explanation.important_dates} />
              </CardBody>
            </Card>
          </div>

          <ReasoningPanel explanation={explanation} />
        </>
      )}
    </div>
  );
}

export default ExplainPage;
