/** Optional service worker registration (no-op when no SW is present). */

export function registerSW(): void {
  if (!('serviceWorker' in navigator)) return;

  // Register only if a SW file is shipped (e.g. production PWA build).
  // Avoid noisy errors in dev when public/sw.js is absent.
  const swUrl = '/sw.js';
  fetch(swUrl, { method: 'HEAD' })
    .then((res) => {
      if (res.ok) {
        return navigator.serviceWorker.register(swUrl);
      }
      return null;
    })
    .catch(() => {
      // Dev / no SW — ignore
    });
}
