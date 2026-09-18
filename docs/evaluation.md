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

## Failure injection

Pass `--failure-type` to the harness and it removes links or satellites from every snapshot before any routing algorithm runs. The pattern depends only on `--failure-seed` and the failure parameters, never on the algorithm, so two algorithms run with the same seed route over exactly the same broken network.

Random failures don't flicker from one minute to the next. Each link or satellite follows its own on/off process and, once down, stays down for a while: 10 minutes on average for a link and 60 for a satellite, which `--failure-mean-duration-minutes` overrides. `--failure-rate` is the long-run fraction of time an element spends down, and a run starts from that steady state instead of from a healthy network.

| `--failure-type` | What breaks |
| --- | --- |
| `isl` | single ISLs, independently of each other, at `--failure-rate` |
| `satellite` | whole satellites, with every ISL and ground link they had; a dead satellite also drops out of ground-station visibility |
| `void` | a square block of neighbouring satellites, `--failure-void-size` on each side, down for the whole run at a position drawn from the seed |
| `cut` | every inter-plane link across two opposite plane boundaries, which splits the grid in two |
| `polar` | inter-plane links with either end above `--failure-polar-latitude-deg`, recomputed each snapshot from satellite positions |

### Voids

Picture the +Grid as a sheet that wraps around in both directions, with orbital planes as columns and positions within a plane as rows. A void punches a hole in it:

```
    plane:  0   1   2   3   4   5   6
    row 0   o---o---o---o---o---o---o
            |   |   |   |   |   |   |
    row 1   o---o---X   X   X   X---o
            |   |               |   |
    row 2   o---o---X   X   X   X---o      X = failed satellite (4x4 void)
            |   |               |   |
    row 3   o---o---X   X   X   X---o
            |   |               |   |
    row 4   o---o---X   X   X   X---o
            |   |   |   |   |   |   |
    row 5   o---o---o---o---o---o---o
```

It models a regional loss, such as a batch of neighbouring satellites taken out together. Topological forwarding finds voids awkward because its distance estimate still points straight through the hole, so a packet approaching from the far side reaches the edge and finds no neighbour any closer to the destination.

### Cuts

```
    plane:  0   1   2 | 3   4   5 | (wraps to 0)
    row 0   o---o---o | o---o---o |
            |   |   | | |   |   | |
    row 1   o---o---o | o---o---o |      | = every inter-plane link removed
            |   |   | | |   |   | |
    row 2   o---o---o | o---o---o |
```

Removing the links across one boundary would only turn the torus into a cylinder, so the cut takes out two boundaries half the constellation apart. That leaves two halves with no link between them, while every satellite keeps working. A pair stays deliverable only when the destination ground station sees some satellite on the source's side, and the delivery metric counts only those pairs. It's the hardest condition for topological forwarding, since the geometry still believes the halves are joined.

Polar deactivation only matters where satellites climb above the threshold. Starlink and Kuiper stay below 60°, so at 60° and 75° it removes nothing from them and only Telesat and OneWeb are affected.

### Who knows about a failure

Every algorithm routes over the post-failure graph of the snapshot it's in. For link-state that's close to reality at one-minute snapshots, because flooding converges within seconds. Topological routing is the exception: its pivot geometry would pick up every failure in the constellation at once, which a satellite deriving that geometry from ephemerides couldn't do. `--geometry-source nominal` builds the geometry from the failure-free graph while next hops still use only live neighbours, and `observed`, the default, keeps the global view as an upper bound.

### Failure metrics

`failure_isls_removed` and `failure_satellites_down` describe each snapshot's damage, and `failure_events` counts failures that appeared or cleared since the previous snapshot. The first snapshot counts every failure present, since none has been announced yet.

When a deliverable pair isn't delivered, `delivery_failure_*` records why, and the causes add up to `delivery_forwarding_failure`:

| Cause | What happened |
| --- | --- |
| `loop` | forwarding came back to a satellite it had already visited |
| `dead_end` | a satellite held no route to the destination |
| `link_down` | an entry or a planned adjacency pointed over a link missing from the snapshot, which is how stale state meets a failure |
| `hop_limit` | the walk ran out of hops |
| `egress_lost` | forwarding finished at a satellite that can't see the destination ground station |

## Evaluation checklist

- Fix ground-station set and simulation horizon for all runs.
- Run `ring` and `grid` for every constellation.
- Use identical time steps for churn comparisons.
- Record `algorithm_params` alongside metrics.
- Report stretch (hop + distance), churn, and forwarding state size.
