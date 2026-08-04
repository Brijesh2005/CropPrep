import React from 'react';

type Props = {
  value: number;
  size?: number;
  stroke?: number;
};

/** Circular confidence gauge (0–1 or 0–100). */
export function ConfidenceGauge({ value, size = 80, stroke = 8 }: Props) {
  const pct = Math.max(0, Math.min(100, value <= 1 ? value * 100 : value));
  const r = (size - stroke) / 2;
  const c = 2 * Math.PI * r;
  const offset = c - (pct / 100) * c;
  const color = pct >= 75 ? '#16a34a' : pct >= 50 ? '#eab308' : pct >= 25 ? '#f97316' : '#ef4444';

  return (
    <svg
      width={size}
      height={size}
      className="shrink-0"
      aria-label={`Confidence ${pct.toFixed(0)}%`}
    >
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke="currentColor"
        strokeWidth={stroke}
        className="text-gray-200 dark:text-gray-700"
      />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={c}
        strokeDashoffset={offset}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
      <text
        x="50%"
        y="50%"
        dominantBaseline="central"
        textAnchor="middle"
        className="fill-gray-900 dark:fill-white text-[11px] font-semibold"
        style={{ fontSize: size * 0.22 }}
      >
        {pct.toFixed(0)}%
      </text>
    </svg>
  );
}

export default ConfidenceGauge;
