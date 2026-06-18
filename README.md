# Sonic NPZ Pages

Static GitHub Pages build for browser-side G1/Sonic sim2sim checks.

The page runs MuJoCo WASM in the browser, loads the bundled Sonic/WBC ONNX
policies, and accepts local `.npz` / `.json` reference motion files through the
`Upload reference motion` button. Uploaded files stay in the browser; they do
not need to be committed to this repository.

## GitHub Pages

This repo deploys through `.github/workflows/pages.yml`, which builds `main`
and publishes `dist/` to the `gh-pages` branch. After pushing to `main`, the
published page is:

```text
https://ziangzheng.github.io/sonicNPZ/
```

For a new repository, enable GitHub Pages once in the repository settings:
`Settings -> Pages -> Build and deployment -> Deploy from a branch`, then select
`gh-pages` and `/ (root)`.

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
