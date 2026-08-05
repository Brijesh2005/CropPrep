import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from 'recharts';
import { normalizeFeatureContributions } from '@/utils/format';

type Props = {
  topFeatures: Array<
    [string, number] | { name: string; contribution: number; value?: string | number }
  >;
  height?: number;
};

export function FeatureChart({ topFeatures, height = 280 }: Props) {
  const features = normalizeFeatureContributions(topFeatures)
    .slice(0, 12)
    .map((f) => ({
      name: f.name.length > 18 ? `${f.name.slice(0, 16)}…` : f.name,
      fullName: f.name,
      contribution: f.contribution,
    }))
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));

  if (!features.length) {
    return (
      <p className="text-sm text-gray-500 py-8 text-center">No feature contributions available.</p>
    );
  }

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <BarChart data={features} layout="vertical" margin={{ left: 8, right: 16 }}>
          <XAxis type="number" fontSize={11} />
          <YAxis type="category" dataKey="name" width={100} fontSize={11} />
          <Tooltip
            formatter={(v: number) => [v.toFixed(4), 'Contribution']}
            labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName ?? ''}
          />
          <ReferenceLine x={0} stroke="#9ca3af" />
          <Bar dataKey="contribution" radius={[0, 4, 4, 0]} barSize={14}>
            {features.map((f) => (
              <Cell key={f.fullName} fill={f.contribution >= 0 ? '#16a34a' : '#ef4444'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default FeatureChart;
