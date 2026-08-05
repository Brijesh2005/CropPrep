import React, { useCallback, useEffect, useState } from 'react';
import { MapContainer, TileLayer, useMapEvents, Rectangle, Tooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { MAX_MAP_ZOOM, MIN_MAP_ZOOM } from '@/config';
import { useMap } from '@/hooks/useMap';
import { LocationMarker, ClickMarker } from './LocationMarker';
import { MapControls } from './MapControls';
import { SearchControl } from './SearchControl';
import type { LocationResponse } from '@/types';
import { LoadingSpinner } from '@/components/ui/LoadingSpinner';

function ClickHandler({ onClick }: { onClick: (lat: number, lon: number) => void }) {
  useMapEvents({
    click(e) {
      onClick(e.latlng.lat, e.latlng.lng);
    },
  });
  return null;
}

function Recenter({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMapEvents({});
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

type MapViewProps = {
  height?: string;
  onLocationSelect?: (lat: number, lon: number) => void;
  interactive?: boolean;
  className?: string;
};

export function MapView({
  height = '500px',
  onLocationSelect,
  interactive = true,
  className,
}: MapViewProps) {
  const {
    center,
    zoom,
    selected,
    locations,
    boundaries,
    showBoundaries,
    isLoadingLocations,
    selectPoint,
    setShowBoundaries,
    goToMyLocation,
    search,
  } = useMap();

  const [locating, setLocating] = useState(false);

  const handleClick = useCallback(
    (lat: number, lon: number) => {
      if (!interactive) return;
      selectPoint(lat, lon);
      onLocationSelect?.(lat, lon);
    },
    [interactive, selectPoint, onLocationSelect]
  );

  const handleLocationPick = useCallback(
    (loc: LocationResponse) => {
      selectPoint(loc.lat, loc.lon);
      onLocationSelect?.(loc.lat, loc.lon);
    },
    [selectPoint, onLocationSelect]
  );

  const handleLocate = async () => {
    setLocating(true);
    try {
      const pos = await goToMyLocation();
      onLocationSelect?.(pos.lat, pos.lon);
    } catch {
      // user denied or unavailable
    } finally {
      setLocating(false);
    }
  };

  return (
    <div
      className={`relative rounded-xl overflow-hidden border border-gray-200 dark:border-gray-700 ${className ?? ''}`}
      style={{ height }}
    >
      {isLoadingLocations && (
        <div className="absolute inset-0 z-[1100] flex items-center justify-center bg-white/50 dark:bg-gray-900/50">
          <LoadingSpinner label="Loading coverage…" />
        </div>
      )}

      <MapContainer
        center={center}
        zoom={zoom}
        minZoom={MIN_MAP_ZOOM}
        maxZoom={MAX_MAP_ZOOM}
        style={{ height: '100%', width: '100%' }}
        scrollWheelZoom={interactive}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <ClickHandler onClick={handleClick} />
        <Recenter center={center} zoom={zoom} />

        {locations.map((loc) => (
          <LocationMarker
            key={loc.id}
            location={loc}
            selected={
              !!selected &&
              Math.abs(selected.lat - loc.lat) < 1e-5 &&
              Math.abs(selected.lon - loc.lon) < 1e-5
            }
            onSelect={handleLocationPick}
          />
        ))}

        {selected && <ClickMarker lat={selected.lat} lon={selected.lon} />}

        {showBoundaries &&
          boundaries.map((b) => {
            if (!b.bbox || b.bbox.length < 4) return null;
            // bbox assumed [minLon, minLat, maxLon, maxLat]
            const [minLon, minLat, maxLon, maxLat] = b.bbox;
            return (
              <Rectangle
                key={b.name}
                bounds={[
                  [minLat, minLon],
                  [maxLat, maxLon],
                ]}
                pathOptions={{
                  color: '#16a34a',
                  weight: 1,
                  fillOpacity: 0.05,
                  dashArray: '4 4',
                }}
              >
                <Tooltip sticky>
                  {b.name} ({b.features} features)
                </Tooltip>
              </Rectangle>
            );
          })}
      </MapContainer>

      {interactive && (
        <>
          <SearchControl onSearch={search} onSelect={handleLocationPick} />
          <MapControls
            onLocate={handleLocate}
            onToggleBoundaries={() => setShowBoundaries(!showBoundaries)}
            showBoundaries={showBoundaries}
            locating={locating}
          />
        </>
      )}

      <div className="absolute bottom-3 left-3 z-[1000] rounded-lg bg-white/90 dark:bg-gray-800/90 px-3 py-2 text-xs shadow text-gray-600 dark:text-gray-300">
        <div className="flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-agriculture-600" /> Dataset
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-blue-600" /> Selected
          </span>
          <span>{locations.length} locations</span>
        </div>
      </div>
    </div>
  );
}

export default MapView;
