import React from 'react';
import { Button } from '@/components/ui/Button';
import { cn } from '@/utils/cn';

type Props = {
  onLocate: () => void;
  onToggleBoundaries: () => void;
  showBoundaries: boolean;
  locating?: boolean;
  className?: string;
};

export function MapControls({
  onLocate,
  onToggleBoundaries,
  showBoundaries,
  locating,
  className,
}: Props) {
  return (
    <div className={cn('absolute top-3 right-3 z-[1000] flex flex-col gap-2', className)}>
      <Button
        variant="secondary"
        size="sm"
        onClick={onLocate}
        loading={locating}
        title="Use my location"
        className="shadow-md bg-white dark:bg-gray-800"
      >
        📍 My location
      </Button>
      <Button
        variant={showBoundaries ? 'primary' : 'secondary'}
        size="sm"
        onClick={onToggleBoundaries}
        title="Toggle administrative boundary info"
        className="shadow-md"
      >
        🗺️ Boundaries {showBoundaries ? 'on' : 'off'}
      </Button>
    </div>
  );
}

export default MapControls;
