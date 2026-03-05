# Evaluation

LEOPath evaluation focuses on path optimality, stability, and state size under dynamic constellations.

## Metrics

- **Stretch**: hop and distance ratio vs shortest-path baseline.
- **Churn**: next-hop changes between consecutive snapshots.
- **Memory footprint**: forwarding state size per satellite.
- **Compute time**: wall-clock time to compute routing state per time step.

Why these metrics:

- Stretch captures path optimality cost of reduced state.
- Churn indicates update frequency and control-plane overhead.
- Memory footprint reflects routing table scalability.
- Compute time serves as a practical proxy for algorithmic complexity.

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
- Segment routing

## ISL scenarios

- `ring`: intra-plane only
- `grid`: intra-plane + inter-plane (+grid)

## Evaluation checklist

- Fix ground-station set and simulation horizon for all runs.
- Run `ring` and `grid` for every constellation.
- Use identical time steps for churn comparisons.
- Record `algorithm_params` alongside metrics.
- Report stretch (hop + distance), churn, and forwarding state size.
