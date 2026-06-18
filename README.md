# Sonic NPZ Pages

Static GitHub Pages build for browser-side G1/Sonic sim2sim checks.

The page runs MuJoCo WASM in the browser, loads the bundled Sonic/WBC ONNX
policies, and accepts local `.npz` / `.json` reference motion files through the
`Upload reference motion` button. Uploaded files stay in the browser; they do
not need to be committed to this repository.

## GitHub Pages

This repo deploys through `.github/workflows/pages.yml`. After pushing to
`main`, the published page is:

```text
https://ziangzheng.github.io/sonicNPZ/
```

If the first deployment does not start, enable GitHub Pages for this repository
with source `GitHub Actions` in the repository settings.

## Local Run

```bash
npm install --include=dev
npm run dev -- --host 0.0.0.0 --port 5188
```

Open:

```text
http://<host>:5188/
```

## Build

```bash
npm run build
```
