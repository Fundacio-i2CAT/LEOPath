# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- `--gs-addressing attachment` makes a ground station's 6G-RUPA address name the
  satellite it is attached to, which is what the addressing scheme describes: a
  forwarding satellite reads the address and forwards toward that slot. It then
  needs nothing about where the ground station sits on the surface, and stops
  evaluating one distance per visible egress on every decision. The attachment is
  the nearest live visible satellite, the rule `_detect_gsl_changes` already used,
  and failed satellites leave the visibility list before routing runs, so a dead
  attachment is replaced at the next snapshot rather than stranding the station.
  The default, `visibility`, keeps the previous behaviour, where the address is
  stable and every satellite minimises over every visible egress instead.
- Stretch is reported as two factors that multiply to it. `stretch_*_egress`
  prices the choice of egress, the best route to the egress an algorithm reached
  over the best route to any egress the destination can see. `stretch_*` prices
  the forwarding, what the algorithm did once that egress was settled. Their
  product is `stretch_*_shared`, which stays the single headline figure against
  an unmoving baseline. The split matters under attachment addressing, where the
  egress follows from the destination address rather than being minimised at
  every satellite, so a suboptimal egress would otherwise be read as a
  forwarding penalty. `stretch_*` is the basis earlier runs reported, now named
  for what it measures rather than kept for continuity alone.
- `aux_gs_renumberings` reports attachment changes per snapshot. Under attachment
  addressing each one costs a directory update and a flow update to the far end
  of every active flow, so it is the price paid for dropping the ground station
  table, and it belongs in the accounting rather than in an assumption.

### Fixed
- The address assigned to a ground station at t=0 came from the first satellite
  in its visibility list, while every later snapshot used the nearest one, so a
  station could renumber immediately after starting. Both now use the nearest.
- Three of the four evaluation constellations did not match their sources.
  Starlink was built as 22 planes of 72 satellites; FCC 21-48 authorises the
  550 km shell as 72 planes of 22, which is also what Hypatia uses. Mean
  motions did not fly the altitude that `altitude_m` states, and `altitude_m`
  sets the ground-link range and the ISL length limit: Starlink flew about
  508 km (now 15.05 rev/day for 550 km), Telesat about 916 km (now 13.66 for
  1015 km, Hypatia's value) and OneWeb about 1264 km (now 13.16 for 1200 km).
  OneWeb was 18 planes of 36 spread over 360 degrees; FCC DA 23-362 gives
  12 planes of 49, and the flying constellation spreads them over about 180
  degrees. Kuiper already matched FCC 20-102. Results from earlier configs are
  preserved under the `pre-config-fix` tag.
- `generate_plus_grid_isls` claimed the last and first planes always
  counter-rotate. That holds only for a Walker star; in a Walker delta the wrap
  is an ordinary co-rotating link and `grid_seam` is a stress test.
- `aux_forwarding_exceptions` counted every decision at a satellite with no live
  link, so failed satellites dominated it: one exception per ground station per
  dead satellite. It now counts only satellites with at least one live link, and
  decisions at isolated satellites are reported as
  `aux_forwarding_exceptions_isolated`.
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
- `aggregate_eval` weights shared-basis stretch by the pairs it was measured
  over, as it already did for the legacy stretch columns. It was averaging
  shared stretch per snapshot, which over-weights sparse Ring snapshots.
### Added
- `raan_spread_degree` in the constellation config, default 360 (Walker
  delta). Set it to 180 for a Walker star. On a star shell the `grid` scenario
  builds the cylinder, because the wrap across its counter-rotating seam would
  join satellites up to half an orbit apart; run metadata records this as
  `isl_seam_wrap`, alongside `raan_spread_degree`.
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
- `leopath.experiments.state_accounting_tables` and
  `leopath.experiments.summarize_failure_sweep`, which pool each run over its
  snapshots and write per-run CSVs plus Markdown tables. The sweep summary
  combines seeds into means with 95% confidence intervals and pairs every
  variant with link-state by seed.
- The failure-sweep summary now reads the guard, local-repair and exception variants.
  It reports looping pairs, local minima at live satellites (subtracting dead
  satellites' decisions for runs recorded before the counter fix), detour and
  exception entries with the one-pass bound and their share of link-state's table,
  hops from each entry to the nearest failure, and failure events per snapshot.
- `--forwarding-guard progress` for topological routing: a satellite forwards
  only to neighbours strictly lower in (Phi, satellite id), where Phi is the
  minimum over visible egresses of distance plus GSL length. Every walk then
  terminates without loops, at an egress or at a local minimum counted as a
  forwarding exception (`aux_forwarding_exceptions`). Accepted only with the
  evaluator-independent distance modes (`torus_unit`, `torus_weighted_pivot`).
- `code_version` in run metadata, set by the runner scripts to the Docker image
  tag so outputs from different builds in one output tree can be told apart.
- `--local-repair square` for topological routing: a satellite whose ISL to a
  nominal neighbour has failed keeps that neighbour as its next hop and reaches
  it over the shortest live three-hop detour, the other sides of a grid square.
  The detour sits below the routing decision, as RINA's two-step routing puts
  the path to the next hop in a lower layer, so the progress guard still holds.
  Path following expands detour entries into their three links, a broken leg is
  classified as `link_down`, and `aux_local_detour_entries` counts them.
- `--exception-policy grow` for topological routing: where the rules cannot deliver
  to a ground station, satellites get exception entries along the shortest live
  path, as in rule-and-exception forwarding. Only the satellites where walks
  break receive entries, repeated until every reachable live satellite delivers.
  Reported as `aux_exception_entries`, with the one-pass placement (every satellite
  whose rule walk fails) as `aux_exception_entries_one_pass`, plus distinct
  satellites, aggregated groups, hops to the nearest failure and unresolved walks.
- `failure_events` per snapshot: failures appearing or clearing since the previous
  snapshot, the dissemination cost of a failure-only flooding scheme.
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
