let deferredPrompt;

function toggleInstallUI(show) {
  const installLink = document.getElementById('installAppBtn');
  const installBanner = document.getElementById('installBanner');
  if (installLink) {
    installLink.classList.toggle('is-hidden', !show);
  }
  if (installBanner) {
    installBanner.classList.toggle('is-hidden', !show);
  }
}

async function installPWA() {
  if (!deferredPrompt) {
    if (window.showNotification) {
      window.showNotification('App is already installed or not supported in this browser.');
    }
    return;
  }
  deferredPrompt.prompt();
  const { outcome } = await deferredPrompt.userChoice;
  deferredPrompt = null;
  toggleInstallUI(false);
  if (window.showNotification) {
    window.showNotification(outcome === 'accepted' ? 'Installing VibeSync...' : 'Install dismissed');
  }
}

window.installPWA = installPWA;

window.addEventListener('beforeinstallprompt', (event) => {
  event.preventDefault();
  deferredPrompt = event;
  toggleInstallUI(true);
});

window.addEventListener('appinstalled', () => {
  deferredPrompt = null;
  toggleInstallUI(false);
  if (window.showNotification) {
    window.showNotification('VibeSync is now installed!');
  }
});

window.addEventListener('DOMContentLoaded', () => {
  const installBtn = document.getElementById('installBannerButton');
  const dismissBtn = document.getElementById('installBannerDismiss');
  const isStandalone = window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone;

  if (isStandalone) {
    toggleInstallUI(false);
  }

  if (installBtn) {
    installBtn.addEventListener('click', () => installPWA());
  }

  if (dismissBtn) {
    dismissBtn.addEventListener('click', () => toggleInstallUI(false));
  }

  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch((err) => {
      console.warn('Service worker registration failed', err);
    });
  }
});
