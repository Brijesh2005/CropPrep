import React from 'react';
import { Card, CardBody, CardHeader, CardTitle } from '@/components/ui/Card';
import type { ExplanationResponse } from '@/types';
import { formatPercent } from '@/utils/format';

type Props = {
  explanation: ExplanationResponse;
};

export function ReasoningPanel({ explanation }: Props) {
  const gates = Object.entries(explanation.modality_gates || {});

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle>AI reasoning</CardTitle>
        </CardHeader>
        <CardBody>
          {explanation.reasoning?.length ? (
            <ul className="space-y-2">
              {explanation.reasoning.map((line, i) => (
                <li key={i} className="flex gap-2 text-sm text-gray-700 dark:text-gray-200">
                  <span className="text-agriculture-600 shrink-0">•</span>
                  <span>{line}</span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-gray-500">
              {explanation.crop
                ? `The model recommends ${explanation.crop} based on multimodal evidence.`
                : 'No reasoning text available for this explanation.'}
            </p>
          )}
        </CardBody>
      </Card>

      {gates.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Modality contribution</CardTitle>
          </CardHeader>
          <CardBody className="space-y-3">
            {gates.map(([name, weight]) => {
              const pct = weight <= 1 ? weight * 100 : weight;
              return (
                <div key={name}>
                  <div className="flex justify-between text-sm mb-1">
                    <span className="capitalize text-gray-700 dark:text-gray-200">{name}</span>
                    <span className="font-medium">{formatPercent(weight)}</span>
                  </div>
                  <div className="h-2 rounded-full bg-gray-100 dark:bg-gray-700 overflow-hidden">
                    <div
                      className="h-full rounded-full bg-agriculture-500 transition-all"
                      style={{ width: `${Math.min(100, pct)}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </CardBody>
        </Card>
      )}

      {explanation.limitations?.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Limitations</CardTitle>
          </CardHeader>
          <CardBody>
            <ul className="space-y-1 text-sm text-amber-800 dark:text-amber-200">
              {explanation.limitations.map((l, i) => (
                <li key={i}>⚠ {l}</li>
              ))}
            </ul>
          </CardBody>
        </Card>
      )}
    </div>
  );
}

export default ReasoningPanel;
