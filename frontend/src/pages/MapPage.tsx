import React, { useState } from 'react';
import { MapView } from '@/components/Map/MapView';
import { PredictionCard } from '@/components/prediction/PredictionCard';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { usePrediction } from '@/hooks/usePrediction';
import { useMapStore } from '@/store/mapStore';
import { SUPPORTED_SEASONS, SUPPORTED_YEARS } from '@/config';
import type { PredictionResponse } from '@/types';

export function MapPage() {
  const { predict, isLoading } = usePrediction();
  const selected = useMapStore((s) => s.selected);

  const [year, setYear] = useState(new Date().getFullYear());
  const [season, setSeason] = useState('Kharif');
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleLocationSelect = () => {
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!selected) {
      setError('Select a location on the map first.');
      return;
    }
    setError(null);
    try {
      const res = await predict({
        lat: selected.lat,
        lon: selected.lon,
        year,
        season,
      });
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">GIS Map Prediction</h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
          Click a dataset location on the map (or use “My location”), pick a season and get the
          recommendation.
        </p>
      </div>

      <MapView height="440px" onLocationSelect={handleLocationSelect} />

      <Card>
        <CardHeader>
          <CardTitle>Prediction settings</CardTitle>
        </CardHeader>
        <CardBody>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-end">
            <div className="sm:col-span-1">
              <Select
                label="Year"
                value={year}
                onChange={(e) => setYear(parseInt(e.target.value, 10))}
                options={SUPPORTED_YEARS.map((y) => ({ value: y, label: String(y) }))}
              />
            </div>
            <div className="sm:col-span-1">
              <Select
                label="Season"
                value={season}
                onChange={(e) => setSeason(e.target.value)}
                options={SUPPORTED_SEASONS.map((s) => ({ value: s, label: s }))}
              />
            </div>
            <div className="sm:col-span-1">
              <Button onClick={handlePredict} loading={isLoading} fullWidth>
                Get prediction
              </Button>
            </div>
          </div>

          {selected && (
            <p className="mt-4 text-sm text-gray-600 dark:text-gray-300">
              Selected point:{' '}
              <span className="font-mono font-medium">
                {selected.lat.toFixed(5)}, {selected.lon.toFixed(5)}
              </span>
            </p>
          )}
          {!selected && (
            <p className="mt-4 text-sm text-amber-700 dark:text-amber-300">
              Tip: click anywhere on the map or search for a village to select a location.
            </p>
          )}
          {error && (
            <p className="mt-4 text-sm text-red-600" role="alert">
              {error}
            </p>
          )}
        </CardBody>
      </Card>

      {result && <PredictionCard prediction={result} />}
    </div>
  );
}

export default MapPage;
