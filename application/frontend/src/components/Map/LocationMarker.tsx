import React from 'react';
import { Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import type { LocationResponse } from '@/types';

// Fix default Leaflet marker icons under Vite bundling
import markerIcon2x from 'leaflet/dist/images/marker-icon-2x.png';
import markerIcon from 'leaflet/dist/images/marker-icon.png';
import markerShadow from 'leaflet/dist/images/marker-shadow.png';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: markerIcon2x,
  iconUrl: markerIcon,
  shadowUrl: markerShadow,
});

const datasetIcon = new L.DivIcon({
  className: '',
  html: `<div style="width:14px;height:14px;border-radius:50%;background:#16a34a;border:2px solid white;box-shadow:0 1px 4px rgba(0,0,0,.4)"></div>`,
  iconSize: [14, 14],
  iconAnchor: [7, 7],
});

const selectedIcon = new L.DivIcon({
  className: '',
  html: `<div style="width:18px;height:18px;border-radius:50%;background:#2563eb;border:3px solid white;box-shadow:0 2px 6px rgba(0,0,0,.45)"></div>`,
  iconSize: [18, 18],
  iconAnchor: [9, 9],
});

type Props = {
  location: LocationResponse;
  selected?: boolean;
  onSelect?: (loc: LocationResponse) => void;
};

export function LocationMarker({ location, selected, onSelect }: Props) {
  const village = location.admin?.village || location.name || 'Location';
  const district = location.admin?.district || '';

  return (
    <Marker
      position={[location.lat, location.lon]}
      icon={selected ? selectedIcon : datasetIcon}
      eventHandlers={{
        click: () => onSelect?.(location),
      }}
    >
      <Popup>
        <div className="text-sm min-w-[140px]">
          <p className="font-semibold">{village}</p>
          {district && <p className="text-gray-600">{district}</p>}
          <p className="text-xs text-gray-500 mt-1">
            {location.lat.toFixed(4)}, {location.lon.toFixed(4)}
          </p>
          {onSelect && (
            <button
              type="button"
              className="mt-2 text-xs font-medium text-agriculture-700 hover:underline"
              onClick={() => onSelect(location)}
            >
              Select for prediction
            </button>
          )}
        </div>
      </Popup>
    </Marker>
  );
}

export function ClickMarker({ lat, lon }: { lat: number; lon: number }) {
  return (
    <Marker position={[lat, lon]} icon={selectedIcon}>
      <Popup>
        Selected point
        <br />
        {lat.toFixed(4)}, {lon.toFixed(4)}
      </Popup>
    </Marker>
  );
}

export default LocationMarker;
