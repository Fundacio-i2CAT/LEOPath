# Routing Algorithms

LEOPath exposes routing algorithms through a pluggable interface. Each one computes forwarding state for ground-station-to-ground-station traffic across the satellite network, and every algorithm sees the same topology snapshots, so runs are directly comparable.

## Implemented algorithms

- `shortest_path_link_state`: Dijkstra over the dynamic ISL graph, recomputed from scratch at every snapshot.
- `topological_routing`: 6G-RUPA addressing with neighbor-based forwarding, where the forwarding decision comes from the distance between the current node's address and the destination's.
- `dra_routing`: the DRA family of Ekici, Akyildiz and Bender, restricted to hop-count distance on logical (plane, slot) coordinates.
- `explicit_path_routing`: a protocol-agnostic proxy for centrally planned explicit paths, with the path pinned across one or more snapshots.

`dra_routing` and `topological_routing` share the same forwarding machinery and differ only in the distance function, which is deliberate: a run of one against the other isolates the contribution of the distance metric with everything else held fixed. `dra_routing` overrides any `distance_mode` you pass so a run cannot quietly turn into the weighted variant.

## Assumptions and limitations

- Every algorithm attaches ground stations to the nearest visible satellite.
- `explicit_path_routing` plans on the current snapshot. Planning against a predicted future snapshot is deliberately left out.
- Topological addressing assumes plane and satellite indices stay stable, which is what makes the address meaningful as a location.
- ISL failures are not modelled unless you inject them.
- Inter-satellite links are built as a `+Grid`, four terminals per satellite. Starlink flies three lasers,
  and [ISL Topology](isl-topology.md) works out what that layout would do to the pivot estimator. It isn't built yet.

## Design considerations

### Shortest-path link-state

Full topology knowledge at every snapshot, so it gives the lower bound on stretch and path length that everything else is measured against. It also sets the ceiling on forwarding state, since each satellite ends up holding an entry per destination.

### Topological routing

Structured addressing replaces full topology state. A satellite decides the next hop from the topological distance between its neighbors' addresses and the destination address, which means state scales with node degree rather than constellation size. The trade the design makes is low, stable state against strict path optimality; whether it actually costs any optimality depends on the distance metric.

The `distance_mode` parameter selects that metric:

- `torus_unit`: hop count on the logical torus. Every edge costs 1, so the estimator is blind to how much physically longer an inter-plane ISL is near the equator than near the poles. This is what `dra_routing` pins.
- `torus_weighted_lookahead` (the default): weighted progress with a one-hop lookahead.
- `torus_weighted_pivot`: builds a per-snapshot weight model from the measured edge lengths, then estimates distance through row and column pivots. This is the mode the Computer Networks paper evaluates.

Parameter notes:

- `plane_weight`, `sat_weight`, `shell_weight`: relative costs used by the weighted modes.

#### Forwarding under failures

Greedy forwarding on a damaged grid can loop. With `geometry_source: nominal`, a satellite whose best link has failed picks another neighbour, and that neighbour's estimate, which knows nothing about the failure, sends the packet straight back. Three options deal with this, and they stack:

- `forwarding_guard: progress` forwards only to a neighbour strictly lower in a potential Φ: the estimated distance to the closest satellite that sees the destination, plus that satellite's ground-link length, with ties broken by satellite id. Every hop goes downhill, so a packet can't revisit a satellite, and one with no lower neighbour is at a local minimum, counted in `aux_forwarding_exceptions`. Φ has to be the same whichever satellite computes it, so only `torus_unit` and `torus_weighted_pivot` accept the guard.
- `local_repair: square` keeps a next hop whose link has failed and reaches it over the shortest live three-hop detour, which on +Grid runs around one grid square. It follows RINA's two-step routing, where the path to the next hop is the lower layer's business, and because the decision still names the same next hop the guard's argument holds. A satellite needs the state of links within two hops, nothing more.
- `exception_policy: grow` adds explicit entries wherever the first two still can't deliver, along the shortest live path, in the style of rule-and-exception forwarding. Entries go only to the satellite where a walk breaks, repeated until every reachable live satellite delivers; `aux_exception_entries_one_pass` reports the larger placement that gives every failing satellite an entry. It assumes satellites learn failures by flooding only the failures over a topology they already know.

With all three on, the failure sweep delivered every deliverable pair on all four constellations, and exception state stayed mostly under 1% of link-state's table.

### Explicit-path routing

Strict satellite paths are computed per source-satellite / destination-GS pair, either centrally or at the ingress. The packet carries the remaining hop list as a strict SRv6-like adjacency header, so transit satellites only need a local neighbor and interface map. For state accounting, every satellite counts its local neighbor/interface entries, while destination-to-segment ingress bindings are counted only on satellites that currently host a ground-station attachment.

Failover follows SRv6-style local protection rather than a transit shortest-path fallback. When the active adjacency goes away, the intended behaviour is to use a precomputed local backup for that hop; with no backup the packet drops and later packets wait for the ingress or controller to replan. When the planned egress satellite can no longer see the destination ground station, the strict mode drops the packet instead of falling back to a full topology lookup, while the dynamic final-egress mode repairs delivery toward whichever egress is currently visible.

Route plans are exposed for evaluation, adjacency SID lists and strict-header byte counts included. Treat it as a family-level example, not a full SRv6 control plane.

Parameter notes:

- `segment_count`: affects sampled waypoint metadata only. Strict forwarding follows the adjacency SID list regardless.
- `segment_refresh_interval_steps`: how many timesteps a strict route plan is reused before replanning. Defaults to `1`, which replans every timestep.
- `plane_weight`, `sat_weight` and `shell_weight` are ignored here and are not tuning knobs for this algorithm.

## Algorithm parameters

Parameters go under `simulation.algorithm_params`.

### Link-state baseline

```yaml
simulation:
  dynamic_state_algorithm: shortest_path_link_state
```

### Topological routing

```yaml
simulation:
  dynamic_state_algorithm: topological_routing
  algorithm_params:
    distance_mode: torus_weighted_pivot
    plane_weight: 100.0
    sat_weight: 1.0
    shell_weight: 1000.0
```

### DRA baseline

```yaml
simulation:
  dynamic_state_algorithm: dra_routing
```

### Explicit-path routing

```yaml
simulation:
  dynamic_state_algorithm: explicit_path_routing
  algorithm_params:
    segment_count: 2
    segment_refresh_interval_steps: 1
```

The paper matrix runs all four against the same snapshots, with `torus_weighted_pivot` as the topological distance mode.

Topological routing with every failure option on:

```yaml
simulation:
  dynamic_state_algorithm: topological_routing
  algorithm_params:
    distance_mode: torus_weighted_pivot
    geometry_source: nominal
    forwarding_guard: progress
    local_repair: square
    exception_policy: grow
```
