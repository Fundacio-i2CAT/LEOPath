# Routing Algorithms

LEOPath exposes routing algorithms via a pluggable interface. Each algorithm computes forwarding state for GS-to-GS traffic by routing through the satellite network.

## Implemented algorithms

- `shortest_path_link_state`: Baseline Dijkstra over the dynamic ISL graph.
- `topological_routing`: 6G-RUPA-inspired addressing with neighbor-based forwarding.
- `predictive_link_state`: Link-state computed on a predicted future topology snapshot.
- `explicit_path_routing`: Protocol-agnostic centrally planned explicit-path proxy with pinned satellite paths.

## Assumptions and limitations

- All algorithms currently assume GS attachments use the nearest satellite.
- Predictive link-state uses deterministic orbital motion but does not model ISL failures unless injected.
- `explicit_path_routing` plans on the current snapshot only in the current implementation; predictive planning is intentionally deferred.
- Topological routing assumes stable plane/satellite indexing for address construction.

## Design considerations

### Shortest-path link-state

- Uses full topology knowledge at each snapshot.
- Provides a lower bound for stretch and path length.
- Serves as the baseline for churn and memory comparisons.

### Topological routing

- Relies on structured addressing (plane and satellite indices) rather than full topology state.
- Local decisions are based on topological distance to destination address.
- Prioritizes low state and stability over strict path optimality.

### Predictive link-state

- Uses the same Dijkstra baseline but evaluates paths on a topology snapshot at `t + horizon`.
- Works best with deterministic movement and regular update cadence (fixed time step).
- Useful as a proxy for operator-controlled precomputation without requiring proprietary details.
- Current paper plots should treat this as an exploratory variant unless the stretch baseline is validated against the same snapshot semantics.

Parameter notes:

- `prediction_horizon_minutes`: larger values may reduce churn but increase stretch if topology changes quickly.

### Explicit-path routing

- Uses centrally computed pinned satellite paths per source-satellite / destination-GS pair.
- Exposes full route plans for evaluation while keeping a first-hop forwarding proxy for compatibility.
- Serves as a paper-facing family-level explicit-path example implementation, not a protocol-faithful SRv6/MPLS model.

Parameter notes:

- `segment_count`: controls sampled waypoint metadata only; forwarding follows the full pinned path.
- `segment_refresh_interval_steps`: controls how many evaluation timesteps a pinned route plan is reused before replanning. If omitted, the current implementation defaults to `1` and replans every timestep.
- `segment_mode`, `plane_weight`, `sat_weight`, and `shell_weight` are not used by the current explicit-path implementation and should not be treated as effective tuning knobs.

## Algorithm parameters

Parameters are passed via `simulation.algorithm_params`.

### Link-state baseline

```yaml
simulation:
  dynamic_state_algorithm: shortest_path_link_state
```

### Topological routing

```yaml
simulation:
  dynamic_state_algorithm: topological_routing
```

### Predictive link-state

```yaml
simulation:
  dynamic_state_algorithm: predictive_link_state
  algorithm_params:
    prediction_horizon_minutes: 5
```

### Explicit-path routing

```yaml
simulation:
  dynamic_state_algorithm: explicit_path_routing
  algorithm_params:
    segment_count: 2
```

For paper-aligned runs, prefer:

```yaml
simulation:
  dynamic_state_algorithm: topological_routing
```

or

```yaml
simulation:
  dynamic_state_algorithm: predictive_link_state
  algorithm_params:
    prediction_horizon_minutes: 5
```

or

```yaml
simulation:
  dynamic_state_algorithm: explicit_path_routing
  algorithm_params:
    segment_count: 2
```
