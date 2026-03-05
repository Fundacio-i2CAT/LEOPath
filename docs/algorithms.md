# Routing Algorithms

LEOPath exposes routing algorithms via a pluggable interface. Each algorithm computes forwarding state for GS-to-GS traffic by routing through the satellite network.

## Implemented algorithms

- `shortest_path_link_state`: Baseline Dijkstra over the dynamic ISL graph.
- `topological_routing`: 6G-RUPA-inspired addressing with neighbor-based forwarding.
- `predictive_link_state`: Link-state computed on a predicted future topology snapshot.
- `segment_routing`: Limited-segment routing using plane alignment then intra-plane moves.

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
