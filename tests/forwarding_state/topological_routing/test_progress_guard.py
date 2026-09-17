import random

import networkx as nx
import pytest

from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    _admissible_neighbours,
    _build_torus_weight_model,
    _EgressPotential,
    _install_next_hop,
    _resolve_forwarding_guard,
    _routing_topological_distance,
    _scaled_gsl_distance,
)
from leopath.topology.constellation import ConstellationData
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress

PLANES = 6
SATS = 8
GS_ID = 1000
MODE = "torus_weighted_pivot"
CONSTELLATION = ConstellationData(PLANES, SATS, "20001.0", 1e6, 5e6, [])


def _address(sat_id: int) -> TopologicalNetworkAddress:
    return TopologicalNetworkAddress(0, sat_id // SATS, sat_id % SATS, 0)


def _grid() -> nx.Graph:
    # Integer lengths keep every sum exact. Intra-plane links cost the same in
    # every plane and inter-plane links depend only on the row, as in a Walker
    # grid, which makes the pivot estimate exact on the failure-free grid.
    graph = nx.Graph()
    for plane in range(PLANES):
        for index in range(SATS):
            sat = plane * SATS + index
            graph.add_edge(sat, plane * SATS + (index + 1) % SATS, weight=600.0)
            row_cost = 1200.0 + 100.0 * min(index, SATS - index)
            graph.add_edge(sat, ((plane + 1) % PLANES) * SATS + index, weight=row_cost)
    return graph


def _forwarding_state(live: nx.Graph, nominal: nx.Graph, egress: list, guard: str) -> dict:
    """Every satellite's decision toward GS_ID, with geometry from the nominal graph."""
    addresses = {sat: _address(sat) for sat in nominal.nodes()}
    model = _build_torus_weight_model(nominal, addresses, CONSTELLATION)
    candidates = [[(gsl, sat, addresses[sat]) for gsl, sat in egress]]
    potential = _EgressPotential(addresses, candidates, CONSTELLATION, MODE, model)
    fstate: dict = {}
    for sat in sorted(nominal.nodes()):
        destination = min(
            candidates[0],
            key=lambda c: _routing_topological_distance(
                addresses[sat], c[2], CONSTELLATION, distance_mode=MODE, weight_model=model
            )
            + _scaled_gsl_distance(c[0], MODE),
        )[2]
        neighbours = [
            (n, n, addresses[n], float(live.edges[sat, n]["weight"])) for n in live.neighbors(sat)
        ]
        _install_next_hop(
            fstate, sat, addresses[sat], destination, 0, GS_ID, neighbours,
            CONSTELLATION, MODE, model, potential, guard, None,
        )  # fmt: skip
    return fstate


def _walk(fstate: dict, start: int) -> str:
    current, visited = start, set()
    while True:
        decision = fstate.get((current, GS_ID))
        if decision is None:
            return "exception"
        if decision == ("GSL", GS_ID):
            return "delivered"
        if current in visited:
            return "loop"
        visited.add(current)
        current = decision


def test_progress_guard_never_loops_under_random_link_failures() -> None:
    rng = random.Random(11)
    nominal = _grid()
    loops_without_guard = 0
    for _ in range(30):
        live = nominal.copy()
        live.remove_edges_from(rng.sample(sorted(live.edges()), k=len(live.edges()) // 6))
        egress = [
            (float(rng.randint(400, 900)), sat) for sat in rng.sample(range(PLANES * SATS), 3)
        ]

        guarded = _forwarding_state(live, nominal, egress, "progress")
        unguarded = _forwarding_state(live, nominal, egress, "none")

        assert all(_walk(guarded, sat) != "loop" for sat in nominal.nodes())
        loops_without_guard += sum(_walk(unguarded, sat) == "loop" for sat in nominal.nodes())

    # The failure patterns do make the unguarded rule loop, so the guard is doing work.
    assert loops_without_guard > 0


def test_progress_guard_changes_nothing_on_the_failure_free_grid() -> None:
    rng = random.Random(5)
    grid = _grid()
    for _ in range(10):
        egress = [
            (float(rng.randint(400, 900)), sat) for sat in rng.sample(range(PLANES * SATS), 3)
        ]
        assert _forwarding_state(grid, grid, egress, "progress") == _forwarding_state(
            grid, grid, egress, "none"
        )


def test_admissible_neighbours_descend_potential_then_satellite_id() -> None:
    potentials = {10: 5.0, 1: 4.0, 2: 5.0, 30: 5.0, 4: 7.0, 5: float("inf")}
    neighbours = [(n, n, None, 1.0) for n in (1, 2, 30, 4, 5)]

    def potential(sat_id: int, _gs_idx: int) -> float:
        return potentials[sat_id]

    kept = _admissible_neighbours(10, neighbours, 0, potential, "progress")
    assert [n[0] for n in kept] == [1, 2]  # lower potential, or equal potential and lower id
    assert _admissible_neighbours(10, neighbours, 0, potential, "none") == neighbours


def test_local_minimum_installs_nothing_and_counts_an_exception() -> None:
    constellation = ConstellationData(PLANES, SATS, "20001.0", 1e6, 5e6, [])
    potentials = {0: 3.0, 1: 4.0, 8: 6.0}
    work: dict = {}
    fstate: dict = {}
    _install_next_hop(
        fstate, 0, _address(0), _address(2 * SATS + 3), 0, GS_ID,
        [(1, 0, _address(1), 1.0), (8, 1, _address(8), 1.0)],
        constellation, "torus_unit", None,
        lambda sat_id, _gs: potentials[sat_id], "progress", work,
    )  # fmt: skip
    assert fstate == {}
    assert work[0]["exceptions"] == 1


def test_guard_needs_an_evaluator_independent_distance() -> None:
    assert _resolve_forwarding_guard({}, "torus_weighted_lookahead") == "none"
    assert _resolve_forwarding_guard({"forwarding_guard": "progress"}, MODE) == "progress"
    assert _resolve_forwarding_guard({"forwarding_guard": "progress"}, "torus_unit") == "progress"
    with pytest.raises(ValueError):
        _resolve_forwarding_guard({"forwarding_guard": "progress"}, "torus_weighted_lookahead")
    with pytest.raises(ValueError):
        _resolve_forwarding_guard({"forwarding_guard": "loop-free"}, MODE)
