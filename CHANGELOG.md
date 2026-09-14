# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Fixed
- Link-state routing now treats a ground station as reachable through any
  satellite above its horizon, choosing whichever minimises path length plus
  GSL length. The Hypatia-derived code path collapsed visibility to the single
  nearest satellite before routing, which made the destination a fixed
  satellite rather than the ground station. Under sparse connectivity that
  satellite is often in an unreachable component while another visible one is
  reachable, so link-state reported failure on pairs that were deliverable:
  in a Ring topology it delivered 80 of 408 deliverable pairs instead of all
  of them.
- Path stretch is now also reported against an algorithm-independent baseline.
  The previous baseline was a shortest path to whichever egress satellite the
  algorithm being measured happened to reach, so each algorithm was graded
  against a different target and over a different subset of pairs, and neither
  the subset nor its size was reported.
- Path-stretch computation hoists shortest-path searches into one cached
  weighted and unweighted single-source run per distinct source satellite,
  instead of a fresh query per destination candidate. The metric previously
  cost roughly ten times the routing algorithm itself on Starlink-scale
  snapshots.
- Installed forwarding state is now counted from what the simulator actually
  built instead of taken on trust from the analytical constellation-size
  proxy. Reported per snapshot as `fstate_installed_*` (reachable table slots
  occupied), `fstate_markers_*` (unreachable-destination markers), and
  `fstate_neighbors_*` (resident neighbour table), side by side with the
  proxy so the two can be compared.
### Added
- Delivery accounting per snapshot (`delivery_*`): deliverable pairs, delivered
  pairs, delivery rate, forwarding failures, and the separate causes of
  non-delivery (no source visibility, no destination visibility, graph
  disconnection).
- `stretch_hop_shared` and `stretch_dist_shared`, graded against the best
  end-to-end route to any satellite the destination can see.
- `delivery_non_optimal_egress_rate`, the share of delivered pairs that exited
  through a satellite other than the optimal one.
- `scripts/run-matrix-parallel.sh` for running an evaluation matrix as parallel
  Docker jobs with per-job wall-clock recorded.
- `aux_*` metrics describing the topological-routing algorithm's per-snapshot
  auxiliary state, reported separately from installed forwarding entries:
  torus weight-model build time and resident size (row/plane edge-cost and
  path-cost tables), pivot memo-cache entries, and per-satellite work
  accounting (forwarding decisions taken, distance-function evaluations
  performed, and the distinct (neighbour, destination) pairs each node would
  memoise).
- `aux_*` metrics for link-state describing the topology database it keeps
  beside its forwarding table: database node and link entries, the
  per-satellite shortest-path tree a deployed router would hold, and the
  all-pairs matrix size and build time the simulator uses to derive every
  satellite's state at once, labelled as simulator-side.
- Failure injection for robustness evaluation, selected with `--failure-type`:
  `isl` and `satellite` outages as seeded per-element on/off processes with a
  stationary rate and mean duration, `void` for a contiguous block of failed
  satellites, `cut` for inter-plane links removed across two opposite plane
  boundaries, and `polar` for inter-plane links switched off above a latitude.
  Failures are applied to each snapshot before routing and depend only on the
  seed and scenario, so every algorithm faces the same pattern. Reported per
  snapshot as `failure_isls_removed` and `failure_satellites_down`.
- `--geometry-source nominal` for topological routing, building the pivot
  geometry from the failure-free graph so the estimator does not gain global
  failure knowledge, while next hops still consider only live neighbours.
- `delivery_failure_*` metrics splitting forwarding failures into loops, dead
  ends, links down, hop-limit exhaustion and lost egress.
- `--explicit-backup-adjacencies` to enable explicit-path single-hop local
  protection from the command line.
- `scripts/run-failure-sweep.sh` for running the failure-injection sweep as
  parallel Docker jobs.
### Removed
- `predictive_link_state` and `traditional_segment_routing`, neither of which
  was used by any published result. The former was link-state evaluated on a
  future snapshot with no update suppression, so it could only lose on every
  axis measured and did not implement the scheme its name suggested.
- The `--prediction-horizon-minutes` and `--segment-mode` flags, which no
  remaining algorithm reads.
### Documentation
- Added a topological-forwarding demo animation and its caption track to the
  Cesium viewer assets.

## [0.1.4] - 2026-06-18
### Fixed
- Bump package version to match the v0.1.4 re-release (v0.1.3 was already
  published to PyPI and rejects re-upload of an identical version string).
### Notes
- Re-release of v0.1.3 to trigger Zenodo archiving (org GitHub-app access
  was pending at v0.1.3 time). No functional content changes.

## [0.1.3] - 2026-06-17
### Added
- DRA-style hop-only logical-coordinate routing baseline (`dra_routing`).
- Pivot-weighted topological routing capturing inter-plane ISL asymmetry.
- Explicit-path routing family with strict and dynamic final-egress modes,
  refresh-interval control, and SRv6-style header-byte accounting.
- Seam (cylinder) +Grid topology for geometrically honest evaluation.
- CesiumJS constellation viewer with selectable route overlay and GSL controls.
- MkDocs documentation site and evaluation run matrix tooling.
### Notes
- Release archived for citation in the Computer Networks manuscript on
  topological forwarding for LEO mega-constellations.

## [0.1.1] - 2025-11-25
### Documentation
- Added PyPI installation instructions and badges.
- Updated project URLs in metadata.

## [0.1.0] - 2025-11-25
### Added
- Initial release of LEOPath simulator.
- Basic routing algorithms (Shortest Path, Topological).
- Visualization tools.
- Docker support.
