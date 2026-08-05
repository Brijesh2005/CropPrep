import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts';

type Props = {
  importantDates: string[];
  /** Optional map of date → importance weight */
  weights?: Record<string, number>;
  height?: number;
};

export function TemporalTimeline({ importantDates, weights, height = 220 }: Props) {
  if (!importantDates.length && !weights) {
    return <p className="text-sm text-gray-500 py-8 text-center">No temporal importance data.</p>;
  }

  const data =
    weights && Object.keys(weights).length
      ? Object.entries(weights)
          .map(([date, value]) => ({ date, value }))
          .sort((a, b) => a.date.localeCompare(b.date))
      : importantDates.map((date, i) => ({
          date,
          value: importantDates.length - i,
        }));

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <AreaChart data={data} margin={{ left: 0, right: 8, top: 8, bottom: 0 }}>
          <defs>
            <linearGradient id="tempFill" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#16a34a" stopOpacity={0.4} />
              <stop offset="100%" stopColor="#16a34a" stopOpacity={0.05} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
          <XAxis dataKey="date" fontSize={10} tick={{ fill: '#6b7280' }} />
          <YAxis fontSize={11} tick={{ fill: '#6b7280' }} />
          <Tooltip />
          <Area
            type="monotone"
            dataKey="value"
            stroke="#16a34a"
            fill="url(#tempFill)"
            name="Importance"
          />
        </AreaChart>
      </ResponsiveContainer>
      {importantDates.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {importantDates.slice(0, 8).map((d) => (
            <span key={d} className="badge-success text-xs">
              {d}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

export default TemporalTimeline;
