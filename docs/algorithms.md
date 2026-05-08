# Routing Algorithms

LEOPath exposes routing algorithms via a pluggable interface. Each algorithm computes forwarding state for GS-to-GS traffic by routing through the satellite network.

## Implemented algorithms

- `shortest_path_link_state`: Baseline Dijkstra over the dynamic ISL graph.
- `topological_routing`: 6G-RUPA-inspired addressing with neighbor-based forwarding.
- `predictive_link_state`: Link-state computed on a predicted future topology snapshot.
- `traditional_segment_routing`: Experimental explicit-path routing with end-to-end segment lists derived from shortest paths.

## Assumptions and limitations

- All algorithms currently assume GS attachments use the nearest satellite.
- Predictive link-state uses deterministic orbital motion but does not model ISL failures unless injected.
- `traditional_segment_routing` is kept for exploratory comparisons; current paper-facing evaluation centers on link-state, predictive link-state, and topological routing.
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

### Traditional segment routing (experimental)

- Uses an explicit segment list derived from shortest paths.
- Serves as an exploratory explicit-path baseline rather than a paper-ready protocol model.
- Segment count is intentionally small to keep packet-carried guidance compact.

Parameter notes:

- `segment_count`: higher values can reduce stretch but increase state.
- `prediction_horizon_minutes`: optionally plans segments on a future snapshot.
- `segment_refresh_interval_steps`: optionally refreshes segment plans less often than every simulation step.

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

### Segment routing

```yaml
simulation:
  dynamic_state_algorithm: traditional_segment_routing
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
