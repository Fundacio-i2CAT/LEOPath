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

## Route Overlay

Pick a source and destination ground station to draw a route between them:

- **Topological forwarding** (yellow) forwards greedily on the satellites' orbital coordinates, the same policy evaluated in the paper.
- **Shortest-path route (link-state)** (green, optional checkbox) is Dijkstra over the inter-satellite-link graph weighted by physical distance.

A ground station is reachable through *any* satellite currently overhead, so delivery happens at the first such satellite (multi-homing). When the two lines coincide, topological forwarding is taking a shortest path; the green line is drawn slightly wider so it shows as a rim around the yellow one.

Reachability depends on the ISL topology. With `Ring` (intra-plane links only), two ground stations connect only when an orbital plane currently passes over both; otherwise the panel reports the route as *unreachable — try +Grid*. With `+Grid` (inter-plane links added) the network is connected, so routes are essentially always available and topological forwarding ties the shortest path.

> Path **quality** (how close topological is to shortest-path) and **reachability** (whether a path exists at all) are separate. The reported path stretch is measured only over reachable pairs; `Ring` gaps are a coverage limitation, not a stretch result.

Route lines are clipped at the Earth's limb, so the half on the far side is hidden behind the globe rather than drawn through it.

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
