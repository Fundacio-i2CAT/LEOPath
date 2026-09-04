# Evaluation

LEOPath evaluation focuses on path optimality, stability, and state size under dynamic constellations.

## Metrics

- **Delivery**: how many ground-station pairs were physically deliverable in a snapshot, and how many the algorithm actually delivered.
- **Stretch**: hop and distance ratio against a shortest-path baseline, reported on two bases (see below).
- **Churn**: next-hop changes between consecutive snapshots.
- **Memory footprint**: forwarding state size per satellite.
- **Compute time**: wall-clock time to compute routing state per time step.

Why these metrics:

- Delivery gives stretch a denominator. Without it, an algorithm that fails on the hard pairs is rewarded, because only its successes reach the average.
- Stretch captures the path-optimality cost of reduced state.
- Churn indicates update frequency and control-plane overhead.
- Memory footprint reflects routing-table scalability.
- Compute time is a rough proxy for algorithmic complexity on the host, not a measure of on-board forwarding cost.

### Reachability and the stretch baseline

Reachability is decided from the topology before any algorithm runs, so every algorithm is scored over the same pairs. Each ordered ground-station pair in a snapshot falls into one of five buckets, reported as `delivery_*` columns:

| Bucket | Meaning |
| --- | --- |
| `no_src_visibility` | the source ground station sees no satellite |
| `no_dst_visibility` | the destination ground station sees no satellite |
| `disconnected` | no ISL path reaches any satellite the destination can see |
| `deliverable` | a path exists, so the pair counts toward `delivery_rate` |
| `delivered` | the algorithm got a packet there; the shortfall is `forwarding_failure` |

A ground station is reachable through **any** satellite above its horizon, not only its nearest one. The baseline for stretch is therefore the best end-to-end route to any of them, which makes it identical for every algorithm. Two stretch families are written:

- `stretch_hop` / `stretch_dist` grade an algorithm against a shortest path to whichever egress satellite it happened to reach. An algorithm that delivers through a poor egress still scores near 1.0, because the baseline follows it there. Kept for continuity with earlier runs.
- `stretch_hop_shared` / `stretch_dist_shared` grade every algorithm against the same lower bound. Use these for comparisons between algorithms.

`delivery_non_optimal_egress_rate` reports how often an algorithm delivered through an egress other than the optimal one, which is what separates the two families. A shortest-path algorithm scores 1.000000 on the shared basis by construction, so link-state doubles as a correctness check on the metric itself.

Optional metrics to add later:

- **Stability window**: time between next-hop changes.
- **Outage sensitivity**: connectivity loss under ISL failures.

## Constellations

- Starlink (synthetic)
- Kuiper (synthetic)
- OneWeb (synthetic)
- Telesat (synthetic)
- Dense LEO (synthetic, stress case)

## Algorithms

- Topological routing
- Link-state baseline
- Predictive link-state
- Explicit-path routing

## ISL scenarios

- `ring`: intra-plane only
- `grid`: intra-plane + inter-plane (+grid)

## Evaluation checklist

- Fix ground-station set and simulation horizon for all runs.
- Run `ring` and `grid` for every constellation.
- Use identical time steps for churn comparisons.
- Record `algorithm_params` alongside metrics.
- Report stretch (hop + distance), churn, and forwarding state size.
