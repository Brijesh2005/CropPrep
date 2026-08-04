import React, { useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from 'recharts';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { predictionService } from '@/services/prediction';
import { adminService } from '@/services/admin';
import { useAuth } from '@/hooks/useAuth';
import { CROP_COLORS } from '@/config';
import { formatConfidence, formatYield } from '@/utils/format';

function useDashboardData() {
  const history = useQuery({
    queryKey: ['dashboard', 'history'],
    queryFn: () => predictionService.getHistory({ limit: 200, offset: 0 }),
    staleTime: 1000 * 60,
  });

  const { isAdmin } = useAuth();
  const admin = useQuery({
    queryKey: ['dashboard', 'admin'],
    queryFn: () => adminService.getStatistics(),
    enabled: isAdmin,
    retry: false,
  });

  return { history, admin };
}

export function DashboardPage() {
  const { history, admin } = useDashboardData();

  const stats = useMemo(() => {
    const items = history.data?.items ?? [];
    const total = items.length;
    const crops: Record<string, number> = {};
    const dates: Record<string, number> = {};
    const buckets: Record<string, number> = {};
    let confidenceSum = 0;
    let yieldSum = 0;
    let yieldCount = 0;
    let fallback = 0;

    for (const it of items) {
      crops[it.recommended_crop] = (crops[it.recommended_crop] ?? 0) + 1;
      confidenceSum += it.confidence ?? 0;
      if (it.expected_yield != null) {
        yieldSum += it.expected_yield;
        yieldCount += 1;
      }
      const day = it.created_at ? it.created_at.slice(0, 10) : '—';
      dates[day] = (dates[day] ?? 0) + 1;
      if (it.confidence < 0.5) fallback += 1;

      const pct = it.confidence <= 1 ? it.confidence * 100 : it.confidence;
      const key = pct < 25 ? '0–25' : pct < 50 ? '25–50' : pct < 75 ? '50–75' : '75–100';
      buckets[key] = (buckets[key] ?? 0) + 1;
    }

    return {
      total,
      avgConfidence: total ? confidenceSum / total : 0,
      avgYield: yieldCount ? yieldSum / yieldCount : 0,
      fallback,
      cropData: Object.entries(crops)
        .map(([name, value]) => ({ name, value }))
        .sort((a, b) => b.value - a.value),
      trendData: Object.entries(dates)
        .sort(([a], [b]) => a.localeCompare(b))
        .map(([date, count]) => ({ date, count })),
      bucketData: ['0–25', '25–50', '50–75', '75–100']
        .map((name) => ({ name, count: buckets[name] ?? 0 }))
        .filter((b) => b.count > 0),
    };
  }, [history.data]);

  if (history.isLoading) {
    return <LoadingSpinner label="Loading dashboard…" className="py-24" />;
  }

  const statCards = [
    { label: 'Total predictions', value: String(stats.total), icon: '🔢' },
    { label: 'Average confidence', value: formatConfidence(stats.avgConfidence), icon: '🎯' },
    { label: 'Average yield', value: formatYield(stats.avgYield), icon: '📈' },
    { label: 'Low-confidence runs', value: String(stats.fallback), icon: '⚠️' },
  ];

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Dashboard</h1>
        <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
          Your prediction activity and model performance at a glance.
        </p>
      </div>

      {admin.data && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {statCards.map((s) => (
            <Card key={s.label}>
              <CardBody>
                <span className="text-2xl" aria-hidden>
                  {s.icon}
                </span>
                <p className="mt-2 text-2xl font-bold text-gray-900 dark:text-white">{s.value}</p>
                <p className="text-xs text-gray-500">{s.label}</p>
              </CardBody>
            </Card>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Crop distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Crop distribution</CardTitle>
          </CardHeader>
          <CardBody>
            {stats.cropData.length ? (
              <div style={{ width: '100%', height: 280 }}>
                <ResponsiveContainer>
                  <PieChart>
                    <Pie
                      data={stats.cropData}
                      dataKey="value"
                      nameKey="name"
                      innerRadius={55}
                      outerRadius={90}
                      paddingAngle={2}
                    >
                      {stats.cropData.map((entry) => (
                        <Cell
                          key={entry.name}
                          fill={CROP_COLORS[entry.name] || CROP_COLORS.default}
                        />
                      ))}
                    </Pie>
                    <Tooltip />
                    <Legend />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-sm text-gray-500 py-10 text-center">No prediction data yet.</p>
            )}
          </CardBody>
        </Card>

        {/* Confidence histogram */}
        <Card>
          <CardHeader>
            <CardTitle>Confidence distribution</CardTitle>
          </CardHeader>
          <CardBody>
            {stats.bucketData.length ? (
              <div style={{ width: '100%', height: 280 }}>
                <ResponsiveContainer>
                  <BarChart data={stats.bucketData} margin={{ top: 8, right: 8, left: -16 }}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" fontSize={11} />
                    <YAxis allowDecimals={false} fontSize={11} />
                    <Tooltip />
                    <Bar dataKey="count" name="Predictions" fill="#16a34a" radius={[6, 6, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : (
              <p className="text-sm text-gray-500 py-10 text-center">No prediction data yet.</p>
            )}
          </CardBody>
        </Card>
      </div>

      {/* Trend */}
      <Card>
        <CardHeader>
          <CardTitle>Prediction activity over time</CardTitle>
        </CardHeader>
        <CardBody>
          {stats.trendData.length ? (
            <div style={{ width: '100%', height: 220 }}>
              <ResponsiveContainer>
                <BarChart data={stats.trendData} margin={{ top: 8, right: 8, left: -16 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" fontSize={10} />
                  <YAxis allowDecimals={false} fontSize={11} />
                  <Tooltip />
                  <Bar dataKey="count" name="Predictions" fill="#22c55e" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-sm text-gray-500 py-10 text-center">No prediction activity yet.</p>
          )}
        </CardBody>
      </Card>
    </div>
  );
}

export default DashboardPage;
