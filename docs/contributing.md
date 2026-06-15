# Contributing

## Adding a routing algorithm

1. Implement a new class under `leopath/network_state/routing_algorithms/`.
2. Register the algorithm in `leopath/network_state/routing_algorithms/routing_algorithm_factory.py`.
3. Add a configuration example and tests.

## Code style

- Format with `black` (line length 100).
- Run `flake8` and `pytest` before opening a PR.
