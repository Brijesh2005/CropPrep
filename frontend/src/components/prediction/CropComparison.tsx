import React from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { CROP_COLORS } from '@/config';
import { formatPercent } from '@/utils/format';

type Item = { crop: string; probability: number };

type Props = {
  data: Item[];
  height?: number;
};

export function CropComparison({ data, height = 200 }: Props) {
  const chartData = data.map((d) => ({
    name: d.crop,
    value: d.probability <= 1 ? d.probability * 100 : d.probability,
    raw: d.probability,
  }));

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer>
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ left: 8, right: 16, top: 4, bottom: 4 }}
        >
          <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} fontSize={11} />
          <YAxis type="category" dataKey="name" width={80} fontSize={12} />
          <Tooltip formatter={(value: number) => [`${Number(value).toFixed(1)}%`, 'Probability']} />
          <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={18}>
            {chartData.map((entry) => (
              <Cell key={entry.name} fill={CROP_COLORS[entry.name] || CROP_COLORS.default} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <ul className="mt-2 space-y-1 sm:hidden">
        {data.map((d) => (
          <li key={d.crop} className="flex justify-between text-sm">
            <span>{d.crop}</span>
            <span className="font-medium">{formatPercent(d.probability)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default CropComparison;
