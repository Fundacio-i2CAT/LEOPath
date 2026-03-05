# Architecture

LEOPath generates dynamic network states from orbital configurations and applies routing algorithms to compute forwarding state.

Core components:

- `leopath/topology`: constellation and link modeling
- `leopath/network_state`: dynamic state generation and routing algorithms
- `leopath/experiments`: evaluation harness and metrics
