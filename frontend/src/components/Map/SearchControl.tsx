import React, { useEffect, useState } from 'react';
import { Input } from '@/components/ui/Input';
import type { LocationResponse } from '@/types';
import { cn } from '@/utils/cn';

type Props = {
  onSearch: (query: string) => Promise<LocationResponse[]>;
  onSelect: (loc: LocationResponse) => void;
  className?: string;
};

export function SearchControl({ onSearch, onSelect, className }: Props) {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<LocationResponse[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!query.trim()) {
      setResults([]);
      return;
    }
    const handle = setTimeout(async () => {
      setLoading(true);
      try {
        const items = await onSearch(query);
        setResults(items);
        setOpen(true);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 300);
    return () => clearTimeout(handle);
  }, [query, onSearch]);

  return (
    <div className={cn('absolute top-3 left-3 z-[1000] w-72 max-w-[calc(100%-1.5rem)]', className)}>
      <Input
        placeholder="Search village or district…"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => results.length && setOpen(true)}
        className="shadow-md bg-white dark:bg-gray-800"
        aria-label="Search locations"
      />
      {loading && (
        <p className="mt-1 text-xs text-gray-500 bg-white/90 dark:bg-gray-800/90 rounded px-2 py-1">
          Searching…
        </p>
      )}
      {open && results.length > 0 && (
        <ul className="mt-1 max-h-56 overflow-y-auto rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg scrollbar-thin">
          {results.map((loc) => (
            <li key={loc.id}>
              <button
                type="button"
                className="w-full text-left px-3 py-2 text-sm hover:bg-agriculture-50 dark:hover:bg-agriculture-900/30"
                onClick={() => {
                  onSelect(loc);
                  setQuery(loc.name || loc.admin?.village || '');
                  setOpen(false);
                }}
              >
                <span className="font-medium text-gray-900 dark:text-white">
                  {loc.name || loc.admin?.village || loc.id}
                </span>
                {loc.admin?.district && (
                  <span className="block text-xs text-gray-500">{loc.admin.district}</span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default SearchControl;
