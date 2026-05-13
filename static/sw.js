const CACHE_NAME = 'vibesync-static-v1';
const STATIC_ASSETS = [
  '/static/css/vibesync-theme.css',
  '/static/css/app.css',
  '/static/css/auth.css',
  '/static/css/admin.css',
  '/static/js/shared-player.js',
  '/static/js/pwa.js',
  '/static/assets/icons/icon-192.png',
  '/static/assets/icons/icon-192-maskable.png',
  '/static/assets/icons/icon-512.png',
  '/static/assets/icons/icon-512-maskable.png',
  '/static/assets/screenshots/screen-home.png',
  '/static/assets/screenshots/screen-dashboard.png'
];

self.addEventListener('install', (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS))
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(
      keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
    )).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  if (request.method !== 'GET') return;

  const url = new URL(request.url);
  const isStatic = url.pathname.startsWith('/static/');
  if (!isStatic) {
    return;
  }

  event.respondWith(
    caches.match(request).then((cached) =>
      cached || fetch(request).then((response) => {
        const copy = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, copy));
        return response;
      }).catch(() => cached)
    )
  );
});
