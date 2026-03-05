# Routing Algorithms

LEOPath exposes routing algorithms via a pluggable interface. Each algorithm computes forwarding state for GS-to-GS traffic by routing through the satellite network.

## Implemented algorithms

- `shortest_path_link_state`: Baseline Dijkstra over the dynamic ISL graph.
- `topological_routing`: 6G-RUPA-inspired addressing with neighbor-based forwarding.
- `predictive_link_state`: Link-state computed on a predicted future topology snapshot.
- `segment_routing`: Limited-segment routing using plane alignment then intra-plane moves.

## Assumptions and limitations

- All algorithms currently assume GS attachments use the nearest satellite.
- Predictive link-state uses deterministic orbital motion but does not model ISL failures unless injected.
- Segment routing uses a small fixed segment list; it does not perform full traffic engineering.
- Topological routing assumes stable plane/satellite indexing for address construction.

## Design considerations

### Predictive link-state

- Uses the same Dijkstra baseline but evaluates paths on a topology snapshot at `t + horizon`.
- Works best with deterministic movement and regular update cadence (fixed time step).
- Useful as a proxy for operator-controlled precomputation without requiring proprietary details.

### Segment routing

- Uses a limited segment list to trade path optimality for state and churn reductions.
- Default mode prioritizes plane alignment, then intra-plane movement (`plane_then_inplane`).
- Segment count is intentionally small to keep forwarding state compact and stable.

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
  dynamic_state_algorithm: segment_routing
  algorithm_params:
    segment_mode: plane_then_inplane
    segment_count: 2
    plane_weight: 100.0
    sat_weight: 1.0
    shell_weight: 1000.0
```
