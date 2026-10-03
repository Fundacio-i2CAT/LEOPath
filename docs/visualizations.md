# Constellation visualizations

LEOPath opens in the **live Cesium constellation explorer**. Satellites and links move continuously with the simulation clock. **Play/Pause** controls orbital motion; **Restart** returns to the beginning without changing whether playback is paused. Drag the Cesium timeline to seek, or use the speed slider. Space toggles orbital playback when a form control is not focused. Layer changes preserve the clock, camera and pause state.

Live topological routes and their shortest-path references are enabled by default for New York → Perth. Select **From** and **To**, use **Swap**, or toggle the shortest-path overlay. Yellow shows the topological route; green shows the reference. Their paths move with the satellites, and attachment/path selection refreshes every 30 simulated seconds. The live route card updates ISL hops and propagation delay (both GSLs included). Ring correctly limits forwarding to one orbital plane; OneWeb +Grid respects its open seam.

### Live failures and brick topologies

The **ISL topology** selector includes **Brick wall A** (two intra-plane links and one staggered cross-plane link) and **Brick wall B** (one intra-plane and two cross-plane links). Their parity wiring matches `generate_brick_isls`. Starlink, Kuiper and Dense support both; OneWeb supports A with an open seam. Telesat's 27 × 13 closed shell supports neither split, and invalid options are disabled with an explanation. Changing shell or topology restores previous failures.

Use **Introduce a failure** directly in the live globe:

- **Fail route ISL** cuts an ISL on the current selected route.
- **Route satellite offline** disables a route satellite, all its incident ISLs and its GSL attachments. Ground stations can attach to another available satellite.
- **Isolate ingress** cuts the current ingress satellite's live ISLs. The view reports a partition if its current attachment has no path to the destination. Orbital motion may later provide a new attachment.
- Click a satellite and open **Choose a specific satellite or ISL**, or enter satellite IDs to inject an exact failure. Only links present in the selected wiring can fail.
- **Restore** beside an individual failure repairs it; **Restore all** clears all failures. **Focus latest failure** centers the globe on affected equipment.

Failed ISLs are dashed red; offline satellites are large red points. Failures persist during orbital motion, timeline seeking, pauses and layer changes. Injection and restoration preserve time, camera and playback state, and both displayed routing policies avoid failed equipment immediately.

**Repair a blocked route** enables a browser preview repair: when the guarded pivot rule has no usable progress neighbor, a shortest-path suffix over the live graph is appended without revisiting the preceding path. Its orange segment identifies the repair. Disable repair to observe where the rule blocks even when a shortest path remains. This does not install or measure the simulator's grow exception policy; its complete results remain in failure replay.

Live fault changes are immediate and do not model failure-detection or convergence latency, packet workloads or on-board processing.

The header switches between **Live orbit** and **Simulator results**. Simulator results combine an orbital globe, a logical plane × slot grid, and an inspector of the simulator's forwarding decisions.

[Open the live constellation](../cesium/index.html){ .leopath-viewer-button }

## Simulator results

[Open simulator results](../cesium/replay.html). Its Play button advances discrete fault phases at the selected pace; orbital positions stay fixed. Use **Live orbit** for continuous orbital motion.

Select a shell and a pair of ground stations, then step through normal forwarding, a failed ISL, recovery, satellite loss, and network partitioning. The bundled replays cover the revised Starlink (72 × 22), Kuiper (34 × 34), OneWeb (12 × 49), and Telesat (27 × 13) single-shell designs. Additional examples show Telesat Ring and open-seam topologies and Kuiper's three-laser brick-wall topology.

The globe and grid show the same exported network. Background links are subdued; topological routes are lime, shortest-path references cyan, unavailable links dashed red, and exception satellites orange. A coincident cyan rim and lime route indicates matching paths. Fault detours may differ from shortest paths; their propagation-delay difference is reported directly.

- Click a satellite in either view to inspect its locator, destination, measured first-hop lengths, remaining-distance estimates, progress condition, and selected rule or exception next hop.
- Use **Step along the route** to inspect successive decisions in the selected delivered walk.
- **Expand** enlarges the logical grid; **Focus on the failure** moves the globe camera.
- Layer controls show or hide background links, other stations, the shortest-path reference, and the preceding route.
- Play/pause, phase buttons, and the pace selector control replay. Keyboard: Space plays or pauses; Left/Right step phases; P toggles presentation mode. When a grid satellite has focus, arrow keys select adjacent logical coordinates.
- **Presentation mode** hides configuration controls for recording.
- Download a replay, or load another compatible replay JSON. Downloaded files contain their configurations and provenance.

### What the numbers mean

These are **controlled demonstrations of converged routing snapshots**, not production failure traces or packet-level simulations. All phases of a replay share one orbital instant, so failures are isolated from orbital motion and attachment handovers. Recovery phases restore the baseline topology. Satellite-outage phases recompute available attachments before routing.

The exporter calls LEOPath's guarded pivot and grow exception policy directly. The browser renders exported decisions and performs no route selection. Derived Walker geometry supplies the remaining-distance estimator; live first hops use the simulator's measured lengths. Both families use one fixed attachment address per station (`K=1`, `nearest_ascending`, `sticky_nearest`). The reference is a physical-distance shortest path between the same attachment satellites. The exporter also retains source config, simulator commit, code hashes, and dependency versions.

- **Delay** is propagation delay, including ISLs and both GSLs. It excludes queueing, serialization, detection, and processing time.
- **Exception state** distinguishes aggregated region entries across the entire shell from raw destination-specific entries. The inspector reports the raw entries installed at the selected satellite.
- **Delivered / attempted** counts ordered ground-station flow walks. It is not a packet delivery ratio. Reachable pairs are also reported under the same attachment policy.
- **Partitioned**, **No visibility**, and **Blocked** distinguish disconnection, absent attachments, and forwarding failure.

No animation duration represents protocol convergence time. There is no simulated traffic workload or assertion about throughput or spacecraft energy.

## Live orbital explorer

[Open the live orbital explorer](../cesium/index.html) for continuous SGP4 propagation, constellation selection, Ring/+Grid geometry, ground-station visibility, density controls, and an orbital timeline. It uses the corrected shell dimensions and treats OneWeb as a 180-degree Walker star with an open seam. Live routes are an illustrative browser preview over the current measured geometry, using nearest visible attachments shared by both policies. The pivot walk enforces progress and rejects repeated satellites; optional repair is marked separately. This preview does not install the simulator’s grow exception state or claim equivalence to the revised paper’s derived-geometry experiments. Use the Python-exported failure replay for those forwarding and exception results.

## Build and serve locally

From the repository root, with LEOPath's Python dependencies installed:

```bash
# Export the four principal shells from the current simulator.
python scripts/export_viewer_replay.py --shells telesat starlink kuiper oneweb

# Optional topology examples.
python scripts/export_viewer_replay.py --shells telesat --topology ring
python scripts/export_viewer_replay.py --shells telesat --topology grid_seam
python scripts/export_viewer_replay.py --shells kuiper --topology brick_a

# Install locked browser dependencies and copy local Cesium/satellite.js assets.
bash scripts/viewer/build-assets.sh

# Serve the standalone viewer.
python -m http.server 8765 --bind 127.0.0.1 --directory docs/cesium
```

Open `http://127.0.0.1:8765/`. Cesium imagery, library code, and replay data are local; playback needs no API token or external CDN. To build the complete documentation site, use `bash scripts/build-docs-site.sh`.

## Verify and record

With the localhost server running and a Chromium executable available:

```bash
python scripts/verify_viewer_replays.py
python scripts/verify_viewer_network.py
LEOPATH_CHROMIUM=/usr/bin/chromium node scripts/viewer/check-viewer.cjs
LEOPATH_CHROMIUM=/usr/bin/chromium node scripts/viewer/check-live-failures.cjs
LEOPATH_CHROMIUM=/usr/bin/chromium node scripts/viewer/record-demo.cjs
# Record live failures and both brick splits directly on the moving globe.
LEOPATH_CHROMIUM=/usr/bin/chromium node scripts/viewer/record-live-failures.cjs
```

`LEOPATH_VIEWER_URL` selects another server URL; `LEOPATH_VIEWER_OUTPUT` selects an output directory. The recorder requires `ffmpeg`, captures the live routes and failure replay at 1920 × 1080, converts to H.264 MP4, and writes the captioned demo, an SRT transcript, and a JSON chapter list into `viewer-recordings/`. The video has on-screen captions and no voice narration. The browser checks cover replay phases, layers, forwarding inspection, route stepping, keyboard controls, invalid data, mobile layout, and live playback. Live checks verify clock advancement, actual satellite movement, moving routes and recomputation, endpoint selection and swapping, shortest-path overlays, Ring and open-seam behavior, pause/resume, speed, restart, preservation of time when layers change, and successful loading of the original LEOPath logos. The data verifier checks delivered paths against live edges, loop freedom, exported next hops, delay ordering, and exact restoration on recovery.

The wiring verifier compares the browser graph and brick distance estimator with the Python simulator. Live failure checks validate every displayed route against the active graph, partitions against reachability, individual restoration, failed ingress handovers, fault persistence during playback, invalid IDs and mobile controls.

## Versioned dataset releases

Published viewer data comes from the companion
[NTN evaluation dataset](https://github.com/PhD-Sergio/data-ntn-topological-evaluation).
Each stable release includes a viewer JSON/TLE bundle with checksums, shell
metadata and simulator provenance. Its release workflow dispatches this site's
Documentation workflow with the release tag and full commit SHA. The Pages build
validates those references and replay paths before deployment. The viewer shows
the dataset version and commit, with links to its release and earlier datasets.

These are controlled Python demonstrations alongside the evaluation results;
aggregate campaign CSVs are not interpreted as recorded route walks. Live route
previews continue to run in the browser. No Python backend is required.

The first dataset release with `viewer/` must be published before enabling the
release-based Pages workflow. Older dataset releases without a viewer asset are
not silently replaced with local examples. A failed build preserves the existing
public site. Cross-repository dispatch requires the dataset repository's
`LEOPATH_PAGES_DISPATCH_TOKEN` secret, restricted to LEOPath with Contents: write.
See the dataset's [viewer release instructions](https://github.com/PhD-Sergio/data-ntn-topological-evaluation/blob/main/viewer/README.md).

For a local preview of an unreleased dataset checkout:

```bash
LEOPATH_DATASET_DIR=/path/to/data-ntn-topological-evaluation bash scripts/build-docs-site.sh
```

For a manual deployment or rollback, run **Documentation → Run workflow** with a
published `dataset_tag`. Blank uses the latest stable release. Local examples are
labeled unversioned; dataset checkout previews are labeled unreleased. Released
TLEs must load from the installed bundle, without a synthetic or moving-branch
fallback.
