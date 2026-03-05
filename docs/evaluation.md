# Evaluation

LEOPath evaluation focuses on path optimality, stability, and state size under dynamic constellations.

## Metrics

- **Stretch**: hop and distance ratio vs shortest-path baseline.
- **Churn**: next-hop changes between consecutive snapshots.
- **Memory footprint**: forwarding state size per satellite.

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
