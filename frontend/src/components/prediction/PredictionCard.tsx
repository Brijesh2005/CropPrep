import React from 'react';
import { Link } from 'react-router-dom';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { CropComparison } from './CropComparison';
import { ConfidenceGauge } from './ConfidenceGauge';
import type { PredictionResponse } from '@/types';
import { formatConfidence, formatYield, locationLabel, topCropProbs } from '@/utils/format';
import { CROP_COLORS } from '@/config';

type Props = {
  prediction: PredictionResponse;
  showExplainLink?: boolean;
};

export function PredictionCard({ prediction, showExplainLink = true }: Props) {
  const top = topCropProbs(prediction.crop_probs, 5);
  const color = CROP_COLORS[prediction.recommended_crop] || CROP_COLORS.default;

  return (
    <Card className="animate-fade-in">
      <CardHeader className="flex flex-row items-center justify-between gap-4">
        <CardTitle>Prediction Result</CardTitle>
        {prediction.fallback && <Badge variant="warning">Fallback model</Badge>}
        {prediction.model_version && <Badge variant="info">v{prediction.model_version}</Badge>}
      </CardHeader>
      <CardBody className="space-y-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div
            className="rounded-xl p-4 border border-gray-100 dark:border-gray-700"
            style={{ borderLeftWidth: 4, borderLeftColor: color }}
          >
            <p className="text-xs uppercase tracking-wide text-gray-500">Recommended crop</p>
            <p className="mt-1 text-2xl font-bold text-gray-900 dark:text-white">
              🌾 {prediction.recommended_crop}
            </p>
          </div>
          <div className="rounded-xl p-4 border border-gray-100 dark:border-gray-700">
            <p className="text-xs uppercase tracking-wide text-gray-500">Expected yield</p>
            <p className="mt-1 text-2xl font-bold text-gray-900 dark:text-white">
              {formatYield(prediction.expected_yield)}
            </p>
          </div>
          <div className="rounded-xl p-4 border border-gray-100 dark:border-gray-700 flex items-center gap-4">
            <ConfidenceGauge value={prediction.confidence} size={72} />
            <div>
              <p className="text-xs uppercase tracking-wide text-gray-500">Confidence</p>
              <p className="text-xl font-bold">{formatConfidence(prediction.confidence)}</p>
            </div>
          </div>
        </div>

        {top.length > 0 && (
          <div>
            <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">
              Top crop probabilities
            </h4>
            <CropComparison data={top} />
          </div>
        )}

        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 rounded-lg bg-gray-50 dark:bg-gray-900/50 p-4 text-sm">
          <div>
            <p className="text-gray-500">Location</p>
            <p className="font-medium text-gray-900 dark:text-white">
              {locationLabel(prediction.location)}
            </p>
            {prediction.coordinates?.lat != null && prediction.coordinates?.lon != null && (
              <p className="text-xs text-gray-400">
                {Number(prediction.coordinates.lat).toFixed(4)},{' '}
                {Number(prediction.coordinates.lon).toFixed(4)}
              </p>
            )}
          </div>
          <div className="text-right text-xs text-gray-500">
            <p>Inference: {prediction.inference_time_ms?.toFixed?.(0) ?? '—'} ms</p>
            {prediction.prediction_id != null && <p>ID: {prediction.prediction_id}</p>}
          </div>
        </div>

        {showExplainLink && prediction.prediction_id != null && (
          <div className="flex justify-end">
            <Link to={`/explain/${prediction.prediction_id}`}>
              <Button variant="outline">View explanation →</Button>
            </Link>
          </div>
        )}
      </CardBody>
    </Card>
  );
}

export default PredictionCard;
