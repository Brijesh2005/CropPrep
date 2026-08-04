import React, { useState } from 'react';
import { Input } from '@/components/ui/Input';
import { Select } from '@/components/ui/Select';
import { Button } from '@/components/ui/Button';
import { SUPPORTED_SEASONS, SUPPORTED_YEARS } from '@/config';
import type { PredictionRequest } from '@/types';

type Props = {
  initial?: Partial<PredictionRequest>;
  loading?: boolean;
  onSubmit: (request: PredictionRequest) => void | Promise<void>;
  submitLabel?: string;
};

export function PredictionForm({
  initial,
  loading,
  onSubmit,
  submitLabel = 'Get prediction',
}: Props) {
  const [lat, setLat] = useState(String(initial?.lat ?? ''));
  const [lon, setLon] = useState(String(initial?.lon ?? ''));
  const [year, setYear] = useState(String(initial?.year ?? new Date().getFullYear()));
  const [season, setSeason] = useState(initial?.season ?? 'Kharif');
  const [error, setError] = useState<string | null>(null);

  // Keep in sync when parent updates selected map point
  React.useEffect(() => {
    if (initial?.lat != null) setLat(String(initial.lat));
    if (initial?.lon != null) setLon(String(initial.lon));
    if (initial?.year != null) setYear(String(initial.year));
    if (initial?.season) setSeason(initial.season);
  }, [initial?.lat, initial?.lon, initial?.year, initial?.season]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const latN = parseFloat(lat);
    const lonN = parseFloat(lon);
    if (Number.isNaN(latN) || latN < -90 || latN > 90) {
      setError('Latitude must be between -90 and 90');
      return;
    }
    if (Number.isNaN(lonN) || lonN < -180 || lonN > 180) {
      setError('Longitude must be between -180 and 180');
      return;
    }
    setError(null);
    await onSubmit({
      lat: latN,
      lon: lonN,
      year: parseInt(year, 10) || undefined,
      season,
    });
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <Input
          label="Latitude"
          type="number"
          step="any"
          value={lat}
          onChange={(e) => setLat(e.target.value)}
          placeholder="12.9141"
          required
        />
        <Input
          label="Longitude"
          type="number"
          step="any"
          value={lon}
          onChange={(e) => setLon(e.target.value)}
          placeholder="74.8560"
          required
        />
        <Select
          label="Year"
          value={year}
          onChange={(e) => setYear(e.target.value)}
          options={SUPPORTED_YEARS.map((y) => ({ value: y, label: String(y) }))}
        />
        <Select
          label="Season"
          value={season}
          onChange={(e) => setSeason(e.target.value)}
          options={SUPPORTED_SEASONS.map((s) => ({ value: s, label: s }))}
        />
      </div>
      {error && <p className="text-sm text-red-600">{error}</p>}
      <Button type="submit" loading={loading} fullWidth>
        {submitLabel}
      </Button>
    </form>
  );
}

export default PredictionForm;
