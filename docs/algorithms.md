# Routing Algorithms

LEOPath supports multiple routing algorithms via a pluggable interface.

## Implemented algorithms

- `shortest_path_link_state`
- `topological_routing`
- `predictive_link_state`
- `segment_routing`

## Algorithm parameters

Parameters are passed via `simulation.algorithm_params`.

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
