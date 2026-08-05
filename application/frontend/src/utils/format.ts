/** Formatting helpers for prediction / dashboard UI. */

export function formatYield(kgPerHa: number | null | undefined): string {
  if (kgPerHa == null || Number.isNaN(kgPerHa)) return '—';
  return `${Math.round(kgPerHa).toLocaleString()} kg/ha`;
}

export function formatConfidence(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return '—';
  const pct = value <= 1 ? value * 100 : value;
  return `${pct.toFixed(1)}%`;
}

export function formatPercent(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) return '—';
  const pct = value <= 1 ? value * 100 : value;
  return `${pct.toFixed(digits)}%`;
}

export function formatCoords(lat: number, lon: number, digits = 4): string {
  return `${lat.toFixed(digits)}, ${lon.toFixed(digits)}`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  } catch {
    return iso;
  }
}

export function locationLabel(
  location: Record<string, unknown> | null | undefined,
  fallback = 'Unknown location'
): string {
  if (!location) return fallback;
  const village = (location.village ?? location.name) as string | undefined;
  const district = location.district as string | undefined;
  const state = location.state as string | undefined;
  const parts = [village, district, state].filter(Boolean);
  return parts.length ? parts.join(', ') : fallback;
}

export function topCropProbs(
  cropProbs: Record<string, number>,
  n = 5
): Array<{ crop: string; probability: number }> {
  return Object.entries(cropProbs)
    .map(([crop, probability]) => ({ crop, probability }))
    .sort((a, b) => b.probability - a.probability)
    .slice(0, n);
}

export function normalizeFeatureContributions(
  topFeatures: Array<
    [string, number] | { name: string; contribution: number; value?: string | number }
  >
): Array<{ name: string; contribution: number; value?: string | number }> {
  return topFeatures.map((f) => {
    if (Array.isArray(f)) {
      return { name: String(f[0]), contribution: Number(f[1]) };
    }
    return f;
  });
}

export function confidenceValue(
  confidence: number | Record<string, number> | null | undefined
): number {
  if (confidence == null) return 0;
  if (typeof confidence === 'number') return confidence;
  return confidence.overall ?? confidence.score ?? Object.values(confidence)[0] ?? 0;
}
