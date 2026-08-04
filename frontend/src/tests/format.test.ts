import { describe, it, expect } from 'vitest';
import {
  formatYield,
  formatConfidence,
  formatPercent,
  formatCoords,
  formatDate,
  locationLabel,
  topCropProbs,
  normalizeFeatureContributions,
  confidenceValue,
} from '@/utils/format';

describe('format helpers', () => {
  it('formatYield formats kg/ha with thousands separator', () => {
    expect(formatYield(1234.5)).toBe('1,235 kg/ha');
    expect(formatYield(null)).toBe('—');
    expect(formatYield(undefined)).toBe('—');
  });

  it('formatConfidence handles 0-1 and 0-100 ranges', () => {
    expect(formatConfidence(0.912)).toBe('91.2%');
    expect(formatConfidence(91.2)).toBe('91.2%');
    expect(formatConfidence(null)).toBe('—');
  });

  it('formatPercent handles ranges', () => {
    expect(formatPercent(0.5)).toBe('50.0%');
    expect(formatPercent(50)).toBe('50.0%');
    expect(formatPercent(null)).toBe('—');
  });

  it('formatCoords produces a comma-separated pair', () => {
    expect(formatCoords(12.91417, 74.8556)).toBe('12.9142, 74.8556');
  });

  it('formatDate renders or falls back', () => {
    expect(formatDate('2026-08-02T12:00:00Z')).toContain('2026');
    expect(formatDate(null)).toBe('—');
    expect(formatDate(undefined)).toBe('—');
  });

  it('locationLabel joins village, district, state', () => {
    expect(
      locationLabel({ village: 'Moodabidri', district: 'Dakshina Kannada', state: 'Karnataka' })
    ).toBe('Moodabidri, Dakshina Kannada, Karnataka');
    expect(locationLabel(null)).toBe('Unknown location');
    expect(locationLabel({})).toBe('Unknown location');
  });

  it('topCropProbs returns sorted top-n entries', () => {
    const probs = { Rice: 0.1, Maize: 0.5, Wheat: 0.3, Coconut: 0.9 };
    const top = topCropProbs(probs, 2);
    expect(top).toEqual([
      { crop: 'Coconut', probability: 0.9 },
      { crop: 'Maize', probability: 0.5 },
    ]);
  });

  it('normalizeFeatureContributions handles tuple and object forms', () => {
    const mixed: Array<[string, number] | { name: string; contribution: number }> = [
      ['rainfall', 0.2],
      { name: 'pH', contribution: -0.1 },
    ];
    expect(normalizeFeatureContributions(mixed)).toEqual([
      { name: 'rainfall', contribution: 0.2 },
      { name: 'pH', contribution: -0.1 },
    ]);
  });

  it('confidenceValue resolves numbers, overall, and score keys', () => {
    expect(confidenceValue(0.85)).toBe(0.85);
    expect(confidenceValue({ overall: 0.9 })).toBe(0.9);
    expect(confidenceValue({ score: 0.7 })).toBe(0.7);
    expect(confidenceValue(null)).toBe(0);
  });
});
