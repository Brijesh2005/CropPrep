import React from 'react';
import { useUiStore } from '@/store/uiStore';
import { cn } from '@/utils/cn';
import type { ToastType } from '@/types';

const typeStyles: Record<ToastType, string> = {
  success: 'bg-agriculture-600 text-white',
  error: 'bg-red-600 text-white',
  warning: 'bg-amber-500 text-white',
  info: 'bg-blue-600 text-white',
};

const typeIcon: Record<ToastType, string> = {
  success: '✓',
  error: '✕',
  warning: '!',
  info: 'i',
};

export function Toaster() {
  const toasts = useUiStore((s) => s.toasts);
  const removeToast = useUiStore((s) => s.removeToast);

  if (!toasts.length) return null;

  return (
    <div
      className="fixed bottom-4 right-4 z-[100] flex flex-col gap-2 max-w-sm w-full pointer-events-none"
      aria-live="polite"
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          className={cn(
            'pointer-events-auto flex items-start gap-3 rounded-lg px-4 py-3 shadow-lg animate-slide-up',
            typeStyles[t.type]
          )}
          role="alert"
        >
          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-white/20 text-xs font-bold">
            {typeIcon[t.type]}
          </span>
          <p className="flex-1 text-sm font-medium">{t.message}</p>
          <button
            type="button"
            className="shrink-0 opacity-80 hover:opacity-100"
            onClick={() => removeToast(t.id)}
            aria-label="Dismiss"
          >
            ×
          </button>
        </div>
      ))}
    </div>
  );
}

export default Toaster;
