"""Per-decision cost of the pivot estimator, tabled against table-free, per real shell.

For each shell config under ``leopath/config/shells`` this reports what a
satellite holds and what one distance query costs:

- tabled: build the P*S^2 + S*P^2 path-cost tables from derived ISL lengths,
  then answer queries with lookups (what ``fstate_calculation`` does);
- table-free: ``DerivedPivotEstimator``, which holds the Walker constants and
  evaluates each query in O(S) from them.

A forwarding decision ranks 3-4 neighbours, so it costs 3-4 queries. Timings
are single-threaded CPython and only comparable with each other; what carries
over to flight hardware is the operation count, O(S) per query, and the state.

Run from the repository root:  python scripts/benchmark_pivot_estimators.py [shell ...]
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leopath.network_state.routing_algorithms.topological_routing.derived_pivot import (  # noqa: E402
    DerivedPivotEstimator,
)
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (  # noqa: E402
    WALKER_CONSTANT_COUNT,
    _build_torus_weight_model,
    _torus_weighted_pivot_distance,
)
from leopath.topology.walker_geometry import walker_shell_from_config  # noqa: E402

QUERIES = 2000
TIME_S = 1234.0


def _address(plane: int, slot: int) -> SimpleNamespace:
    satellite = SimpleNamespace(shell_id=0, plane_id=plane, sat_index=slot)
    return SimpleNamespace(get_satellite_address=lambda: satellite)


def _derived_graph(shell, wrap: bool):
    planes, slots = shell.planes, shell.sats_per_plane
    graph, addresses = nx.Graph(), {}
    for plane in range(planes):
        for slot in range(slots):
            node = plane * slots + slot
            addresses[node] = _address(plane, slot)
            graph.add_edge(node, plane * slots + (slot + 1) % slots, weight=shell.rail_length_m())
            if plane + 1 < planes or wrap:
                graph.add_edge(
                    node,
                    ((plane + 1) % planes) * slots + slot,
                    weight=shell.rung_length_m(plane, slot, TIME_S),
                )
    return graph, addresses


def benchmark(config_path: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    shell = walker_shell_from_config(config["constellation"])
    planes, slots = shell.planes, shell.sats_per_plane
    wrap = shell.raan_spread_deg >= 360.0
    rng = random.Random(1)
    pairs = [
        ((rng.randrange(planes), rng.randrange(slots)), (rng.randrange(planes), rng.randrange(slots)))
        for _ in range(QUERIES)
    ]

    graph, addresses = _derived_graph(shell, wrap)
    start = time.perf_counter()
    model = _build_torus_weight_model(
        graph, addresses, SimpleNamespace(n_orbits=planes, n_sats_per_orbit=slots)
    )
    build_s = time.perf_counter() - start
    start = time.perf_counter()
    for source, destination in pairs:
        model["pivot_distance_cache"].clear()
        _torus_weighted_pivot_distance(
            addresses[source[0] * slots + source[1]],
            addresses[destination[0] * slots + destination[1]],
            model,
        )
    tabled_query_s = (time.perf_counter() - start) / QUERIES

    estimator = DerivedPivotEstimator(shell, TIME_S)
    start = time.perf_counter()
    for source, destination in pairs:
        estimator.distance(source, destination)
    derived_query_s = (time.perf_counter() - start) / QUERIES

    mismatches = sum(
        1
        for source, destination in pairs[:200]
        if abs(
            estimator.distance(source, destination)
            - _torus_weighted_pivot_distance(
                addresses[source[0] * slots + source[1]],
                addresses[destination[0] * slots + destination[1]],
                model,
            )
        )
        > 1e-6
    )
    return {
        "shell": config_path.stem,
        "planes": planes,
        "slots": slots,
        "satellites": planes * slots,
        "table_entries": planes * slots * slots + slots * planes * planes,
        "table_build_ms": build_s * 1000.0,
        "tabled_query_us": tabled_query_s * 1e6,
        "derived_constants": WALKER_CONSTANT_COUNT,
        "derived_query_us": derived_query_s * 1e6,
        "derived_ops_per_query": slots,  # pivot rows, each O(1) arithmetic
        "mismatches_of_200": mismatches,
    }


def main() -> None:
    names = sys.argv[1:]
    paths = sorted((ROOT / "leopath/config/shells").glob("*.yaml"))
    if names:
        paths = [p for p in paths if p.stem in names]
    columns = [
        "shell",
        "satellites",
        "table_entries",
        "table_build_ms",
        "tabled_query_us",
        "derived_constants",
        "derived_query_us",
        "derived_ops_per_query",
        "mismatches_of_200",
    ]
    print(",".join(columns))
    for path in paths:
        row = benchmark(path)
        print(",".join(f"{row[c]:.1f}" if isinstance(row[c], float) else str(row[c]) for c in columns))


if __name__ == "__main__":
    main()
