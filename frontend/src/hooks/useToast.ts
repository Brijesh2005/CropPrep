import { useUiStore } from '@/store/uiStore';
import type { ToastType } from '@/types';

export function useToast() {
  const addToast = useUiStore((s) => s.addToast);
  const removeToast = useUiStore((s) => s.removeToast);
  const toasts = useUiStore((s) => s.toasts);

  return {
    toasts,
    toast: (type: ToastType, message: string, duration?: number) =>
      addToast(type, message, duration),
    success: (message: string) => addToast('success', message),
    error: (message: string) => addToast('error', message),
    warning: (message: string) => addToast('warning', message),
    info: (message: string) => addToast('info', message),
    dismiss: removeToast,
  };
}

export default useToast;
