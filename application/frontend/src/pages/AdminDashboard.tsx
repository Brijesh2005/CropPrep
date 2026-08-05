import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { adminService } from '@/services/admin';
import { getErrorMessage } from '@/services/api';
import { useUiStore } from '@/store/uiStore';
import { CROP_COLORS } from '@/config';
import { formatConfidence } from '@/utils/format';

export function AdminDashboard() {
  const addToast = useUiStore((s) => s.addToast);
  const [retraining, setRetraining] = useState(false);

  const dashboard = useQuery({
    queryKey: ['admin', 'dashboard'],
    queryFn: () => adminService.getDashboard(),
  });

  const stats = useQuery({
    queryKey: ['admin', 'stats'],
    queryFn: () => adminService.getStatistics(),
  });

  const metrics = useQuery({
    queryKey: ['admin', 'metrics'],
    queryFn: () => adminService.getMetrics(),
    refetchInterval: 30_000,
  });

  const handleRetrain = async () => {
    setRetraining(true);
    try {
      const res = await adminService.retrain();
      addToast(res.started ? 'success' : 'warning', res.message);
    } catch (err) {
      addToast('error', getErrorMessage(err, 'Failed to start retraining'));
    } finally {
      setRetraining(false);
    }
  };

  if (dashboard.isLoading || stats.isLoading) {
    return <LoadingSpinner label="Loading admin dashboard…" className="py-24" />;
  }

  const d = dashboard.data;
  const s = stats.data;

  const systemCards = [
    { label: 'Model ready', value: d?.model_ready ? 'Yes' : 'No', ok: !!d?.model_ready },
    { label: 'Dataset ready', value: d?.dataset_ready ? 'Yes' : 'No', ok: !!d?.dataset_ready },
    { label: 'Total predictions', value: String(d?.prediction_count ?? 0), ok: true },
    { label: 'Registered users', value: String(d?.users_count ?? 0), ok: true },
  ];

  const cropData = Object.entries(s?.crop_distribution ?? {})
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value);
  const maxCrop = cropData.length ? cropData[0].value : 1;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Admin dashboard</h1>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
            System status, model health and operational metrics.
          </p>
        </div>
        <Button variant="outline" onClick={handleRetrain} loading={retraining}>
          ↻ Retrain model
        </Button>
      </div>

      {dashboard.error && (
        <p className="text-sm text-red-600" role="alert">
          {getErrorMessage(dashboard.error, 'Could not load admin data')}
        </p>
      )}

      {/* System status */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {systemCards.map((c) => (
          <Card key={c.label}>
            <CardBody>
              <div className="flex items-center justify-between">
                <p className="text-xs text-gray-500">{c.label}</p>
                {typeof c.ok === 'boolean' && (
                  <Badge variant={c.ok ? 'success' : 'danger'}>{c.ok ? 'OK' : 'OFF'}</Badge>
                )}
              </div>
              <p className="mt-2 text-2xl font-bold text-gray-900 dark:text-white">{c.value}</p>
            </CardBody>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model */}
        <Card>
          <CardHeader>
            <CardTitle>Model & inference</CardTitle>
          </CardHeader>
          <CardBody className="space-y-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-500">Model version</span>
              <span className="font-medium">{d?.model_version || '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Device</span>
              <span className="font-medium">{d?.device || '—'}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Queue size</span>
              <span className="font-medium">{d?.queue_size ?? 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Avg confidence</span>
              <span className="font-medium">{formatConfidence(s?.avg_confidence)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Avg inference time</span>
              <span className="font-medium">
                {s?.avg_inference_time_ms ? `${s.avg_inference_time_ms.toFixed(1)} ms` : '—'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-500">Fallback predictions</span>
              <span className="font-medium">{s?.fallback_count ?? 0}</span>
            </div>
          </CardBody>
        </Card>

        {/* Crop distribution */}
        <Card>
          <CardHeader>
            <CardTitle>Crop distribution</CardTitle>
          </CardHeader>
          <CardBody>
            {cropData.length ? (
              <ul className="space-y-3">
                {cropData.map((c) => (
                  <li key={c.name}>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-700 dark:text-gray-200">🌾 {c.name}</span>
                      <span className="font-medium">{c.value}</span>
                    </div>
                    <div className="h-2 rounded-full bg-gray-100 dark:bg-gray-700 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${(c.value / maxCrop) * 100}%`,
                          backgroundColor: CROP_COLORS[c.name] || CROP_COLORS.default,
                        }}
                      />
                    </div>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-gray-500 py-8 text-center">No prediction data yet.</p>
            )}
          </CardBody>
        </Card>
      </div>

      {/* Runtime metrics */}
      <Card>
        <CardHeader>
          <CardTitle>Runtime metrics</CardTitle>
        </CardHeader>
        <CardBody>
          {metrics.data ? (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="rounded-lg border border-gray-100 dark:border-gray-700 p-4">
                <p className="text-xs text-gray-500">Total requests</p>
                <p className="text-xl font-bold">{metrics.data.requests}</p>
              </div>
              <div className="rounded-lg border border-gray-100 dark:border-gray-700 p-4">
                <p className="text-xs text-gray-500">Errors</p>
                <p className="text-xl font-bold">{metrics.data.errors}</p>
              </div>
              <div className="rounded-lg border border-gray-100 dark:border-gray-700 p-4">
                <p className="text-xs text-gray-500">Avg latency</p>
                <p className="text-xl font-bold">{metrics.data.avg_latency_ms.toFixed(1)} ms</p>
              </div>
              <div className="rounded-lg border border-gray-100 dark:border-gray-700 p-4">
                <p className="text-xs text-gray-500">Uptime</p>
                <p className="text-xl font-bold">
                  {metrics.data.uptime_seconds > 3600
                    ? `${(metrics.data.uptime_seconds / 3600).toFixed(1)} h`
                    : `${Math.round(metrics.data.uptime_seconds / 60)} m`}
                </p>
              </div>
            </div>
          ) : (
            <p className="text-sm text-gray-500">No runtime metrics available.</p>
          )}
        </CardBody>
      </Card>
    </div>
  );
}

export default AdminDashboard;
