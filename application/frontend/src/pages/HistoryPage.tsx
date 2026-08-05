import React, { useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { useHistory } from '@/hooks/useHistory';
import { Table } from '@/components/ui/Table';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import { formatConfidence, formatDate, formatYield, locationLabel } from '@/utils/format';
import type { HistoryItem } from '@/types';

const PAGE_SIZE = 20;

export function HistoryPage() {
  const [offset, setOffset] = useState(0);
  const [cropFilter, setCropFilter] = useState('');
  const { data, isLoading, isError, refetch } = useHistory({
    limit: PAGE_SIZE,
    offset,
  });

  const items = data?.items ?? [];
  const total = data?.total ?? 0;

  const filtered = useMemo(() => {
    if (!cropFilter.trim()) return data?.items ?? [];
    const q = cropFilter.trim().toLowerCase();
    return (data?.items ?? []).filter(
      (it) =>
        it.recommended_crop.toLowerCase().includes(q) ||
        locationLabel(it.location).toLowerCase().includes(q)
    );
  }, [data?.items, cropFilter]);

  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const handleExport = () => {
    const rows = items.map((it) => ({
      id: it.prediction_id,
      crop: it.recommended_crop,
      yield: it.expected_yield,
      confidence: it.confidence,
      location: locationLabel(it.location),
      created_at: it.created_at,
    }));
    const csv = [
      ['id', 'crop', 'expected_yield_kg_ha', 'confidence', 'location', 'created_at'].join(','),
      ...rows.map((r) =>
        [
          r.id,
          r.crop,
          r.yield,
          r.confidence,
          `"${r.location.replace(/"/g, '""')}"`,
          r.created_at,
        ].join(',')
      ),
    ].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'cropfusion-history.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Prediction history</h1>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-300">
            {total} saved predictions. Filter by crop or location, then open the explanation.
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={handleExport} disabled={!items.length}>
            Export CSV
          </Button>
          <Button variant="ghost" onClick={() => refetch()}>
            Refresh
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <CardTitle>Saved predictions</CardTitle>
            <input
              type="search"
              className="input max-w-xs"
              placeholder="Filter by crop or location…"
              value={cropFilter}
              onChange={(e) => setCropFilter(e.target.value)}
              aria-label="Filter predictions"
            />
          </div>
        </CardHeader>
        <CardBody>
          {isLoading ? (
            <LoadingSpinner label="Loading history…" className="py-12" />
          ) : isError ? (
            <div className="flex flex-col items-center gap-3 py-12 text-center">
              <p className="text-sm text-red-600">Could not load your history.</p>
              <Button variant="outline" onClick={() => refetch()}>
                Retry
              </Button>
            </div>
          ) : (
            <Table<HistoryItem>
              data={filtered}
              rowKey={(r) => r.prediction_id}
              emptyMessage="No predictions yet. Run your first prediction from the map."
              columns={[
                {
                  key: 'recommended_crop',
                  header: 'Crop',
                  render: (r) => <span className="font-medium">🌾 {r.recommended_crop}</span>,
                },
                {
                  key: 'expected_yield',
                  header: 'Expected yield',
                  render: (r) => formatYield(r.expected_yield),
                },
                {
                  key: 'confidence',
                  header: 'Confidence',
                  render: (r) => <Badge variant="success">{formatConfidence(r.confidence)}</Badge>,
                },
                {
                  key: 'location',
                  header: 'Location',
                  render: (r) => (
                    <span className="text-gray-600 dark:text-gray-300">
                      {locationLabel(r.location)}
                    </span>
                  ),
                },
                {
                  key: 'created_at',
                  header: 'Created',
                  render: (r) => <span className="text-gray-500">{formatDate(r.created_at)}</span>,
                },
                {
                  key: 'actions',
                  header: '',
                  className: 'text-right',
                  render: (r) => (
                    <Link
                      to={`/explain/${r.prediction_id}`}
                      className="text-xs font-medium text-agriculture-600 hover:underline"
                    >
                      Explain →
                    </Link>
                  ),
                },
              ]}
            />
          )}

          {totalPages > 1 && (
            <div className="mt-4 flex items-center justify-between">
              <Button
                variant="outline"
                size="sm"
                disabled={offset <= 0}
                onClick={() => setOffset(Math.max(0, offset - PAGE_SIZE))}
              >
                ← Previous
              </Button>
              <span className="text-sm text-gray-500">
                Page {page} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={offset + PAGE_SIZE >= total}
                onClick={() => setOffset(offset + PAGE_SIZE)}
              >
                Next →
              </Button>
            </div>
          )}
        </CardBody>
      </Card>
    </div>
  );
}

export default HistoryPage;
