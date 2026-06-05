# Constellation Visualizations

LEOPath includes a static Cesium viewer for the constellation designs used in the routing evaluation.

The viewer runs fully in the browser. It loads the simulator-generated TLE files, propagates satellite positions with `satellite.js`, and renders the constellation in CesiumJS.

## Open Viewer

Open the standalone viewer:

<p><a class="leopath-viewer-button" href="../cesium/index.html" target="_blank" rel="noopener">Open Cesium constellation viewer</a></p>

## Included Constellations

- `Starlink-550`: 22 orbital planes, 72 satellites per plane.
- `Kuiper-synthetic`: 34 orbital planes, 34 satellites per plane.
- `OneWeb-synthetic`: 18 orbital planes, 36 satellites per plane.
- `Telesat-synthetic`: 27 orbital planes, 13 satellites per plane.
- `Dense-LEO-synthetic`: 72 orbital planes, 72 satellites per plane.

## Controls

- Select a constellation from the drop-down menu.
- Select the ISL topology: `Ring` draws intra-plane links, while `+Grid` draws both intra-plane and inter-plane links.
- Toggle ground-station markers, nearest-visible GSL attachments, and all-satellite rendering for sampled dense views.
- Use the clock-speed slider or Cesium timeline to inspect orbital motion.

## Local Build

Build the docs plus Cesium assets:

```bash
bash scripts/build-docs-site.sh
```

Serve the generated site locally:

```bash
python -m http.server -d site
```

Then open `http://localhost:8000/cesium/`.

<style>
.leopath-viewer-button {
  display: inline-block;
  padding: 0.7rem 0.9rem;
  border: 1px solid #4b5563;
  border-radius: 0.25rem;
  background: #1f2937;
  color: #fff !important;
  font-weight: 700;
  text-decoration: none;
}
</style>
