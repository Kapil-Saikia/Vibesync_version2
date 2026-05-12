# VibeSync Trusted Web Activity (TWA)

This folder includes the Bubblewrap configuration used to generate the Android TWA project for Play Store distribution.

## Prerequisites
- Android Studio / Android SDK (API 34 or newer)
- JDK 17
- `bubblewrap` CLI (`npx @bubblewrap/cli`)

## Generate the Android project
1. Update `android/twa-manifest.json` with your real domain and package name.
2. Update `static/.well-known/assetlinks.json` with your **release** SHA-256 fingerprint.
3. Run Bubblewrap on your local machine:

```bash
npx @bubblewrap/cli init --manifest https://your-domain.com/static/manifest.json --directory android-app
```

4. When prompted, use the values from `android/twa-manifest.json`.
5. Open the generated `android-app` folder in Android Studio and build a signed release.

## Play Store readiness checklist
- ✅ Use a **verified HTTPS domain** that matches `assetlinks.json`.
- ✅ Update the `packageId` and keep it consistent across Bubblewrap + Play Console.
- ✅ Generate a release keystore and keep its SHA-256 fingerprint in `assetlinks.json`.
- ✅ Provide high-res screenshots (already in `static/assets/screenshots`).
- ✅ Build a signed `AAB` and upload to the Play Console.

> Note: The Bubblewrap SDK download step may fail in restricted environments. If so, run the command above on your local machine with full network access.
