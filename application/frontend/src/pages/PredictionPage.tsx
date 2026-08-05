import React from 'react';
import { PredictionForm } from '@/components/prediction/PredictionForm';
import { PredictionCard } from '@/components/prediction/PredictionCard';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { usePrediction } from '@/hooks/usePrediction';
import type { PredictionRequest } from '@/types';

export const PredictionPage = () => {
  const { isLoading, predict, prediction, error } = usePrediction();

  const handleSubmit = async (request: PredictionRequest) => {
    await predict(request);
  };

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Location prediction</h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
          Enter coordinates, or better — pick a point on the{' '}
          <a href="/map" className="text-agriculture-600 hover:underline">
            interactive map
          </a>
          .
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Prediction inputs</CardTitle>
        </CardHeader>
        <CardBody className="max-w-xl">
          <PredictionForm onSubmit={handleSubmit} loading={isLoading} />
          {error && (
            <p className="mt-4 text-sm text-red-600" role="alert">
              {error}
            </p>
          )}
        </CardBody>
      </Card>

      {prediction && <PredictionCard prediction={prediction} />}
    </div>
  );
};

export default PredictionPage;
