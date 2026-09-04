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
### Removed
- `predictive_link_state` and `traditional_segment_routing`, neither of which
  was used by any published result. The former was link-state evaluated on a
  future snapshot with no update suppression, so it could only lose on every
  axis measured and did not implement the scheme its name suggested.
- The `--prediction-horizon-minutes` and `--segment-mode` flags, which no
  remaining algorithm reads.

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
