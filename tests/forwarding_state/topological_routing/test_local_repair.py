import random
from types import SimpleNamespace

import networkx as nx
import pytest

from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    LOCAL_DETOUR,
    _build_torus_weight_model,
    _EgressPotential,
    _install_next_hop,
    _routing_topological_distance,
    _scaled_gsl_distance,
    _shortest_three_hop_detour,
    _with_local_detours,
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
    # Integer lengths keep sums exact; inter-plane cost depends only on the row.
    graph = nx.Graph()
    for plane in range(PLANES):
        for index in range(SATS):
            sat = plane * SATS + index
            graph.add_edge(sat, plane * SATS + (index + 1) % SATS, weight=600.0)
            row_cost = 1200.0 + 100.0 * min(index, SATS - index)
            graph.add_edge(sat, ((plane + 1) % PLANES) * SATS + index, weight=row_cost)
    return graph


def _live_candidates(live: nx.Graph) -> dict:
    return {
        sat: [(n, n, _address(n), float(live.edges[sat, n]["weight"])) for n in live.neighbors(sat)]
        for sat in live.nodes()
    }


def _augmented(nominal: nx.Graph, live: nx.Graph, repair: str) -> dict:
    topology = SimpleNamespace(nominal_graph=nominal, graph=live)
    addresses = {sat: _address(sat) for sat in nominal.nodes()}
    return _with_local_detours(
        _live_candidates(live), topology, live, addresses, {"local_repair": repair}, MODE
    )


def _forwarding_state(live: nx.Graph, nominal: nx.Graph, egress: list, repair: str) -> dict:
    addresses = {sat: _address(sat) for sat in nominal.nodes()}
    model = _build_torus_weight_model(nominal, addresses, CONSTELLATION)
    candidates = [[(gsl, sat, addresses[sat]) for gsl, sat in egress]]
    potential = _EgressPotential(addresses, candidates, CONSTELLATION, MODE, model)
    neighbours = _augmented(nominal, live, repair)
    fstate: dict = {}
    for sat in sorted(nominal.nodes()):
        destination = min(
            candidates[0],
            key=lambda c: _routing_topological_distance(
                addresses[sat], c[2], CONSTELLATION, distance_mode=MODE, weight_model=model
            )
            + _scaled_gsl_distance(c[0], MODE),
        )[2]
        _install_next_hop(
            fstate, sat, addresses[sat], destination, 0, GS_ID, neighbours[sat],
            CONSTELLATION, MODE, model, potential, "progress", None,
        )  # fmt: skip
    return fstate


def _walk(fstate: dict, live: nx.Graph, start: int) -> str:
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
        if isinstance(decision, tuple) and decision[0] == LOCAL_DETOUR:
            _, first, second, target = decision
            assert live.has_edge(current, first)
            assert live.has_edge(first, second)
            assert live.has_edge(second, target)
            current = target
        else:
            current = decision


def test_detour_goes_around_the_grid_square() -> None:
    live = _grid()
    live.remove_edge(0, 1)  # plane 0, indices 0 and 1
    # Sat 8 and 9 sit next to 0 and 1 in plane 1: 1200 + 600 + 1300 around the square.
    assert _shortest_three_hop_detour(live, 0, 1) == (8, 9, 3100.0)


def test_no_detour_to_a_dead_satellite() -> None:
    live = _grid()
    live.remove_edges_from(list(live.edges(1)))
    assert _shortest_three_hop_detour(live, 0, 1) is None


def test_detours_are_added_only_for_failed_nominal_links() -> None:
    nominal = _grid()
    live = nominal.copy()
    live.remove_edge(0, 1)
    augmented = _augmented(nominal, live, "square")
    plain = _live_candidates(live)

    assert [c[1] for c in augmented[0] if c not in plain[0]] == [(LOCAL_DETOUR, 8, 9, 1)]
    assert [c[1] for c in augmented[1] if c not in plain[1]] == [(LOCAL_DETOUR, 9, 8, 0)]
    assert all(augmented[sat] == plain[sat] for sat in nominal.nodes() if sat not in (0, 1))
    assert _augmented(nominal, live, "none") == plain


def test_local_repair_validates_its_setting() -> None:
    nominal = _grid()
    topology = SimpleNamespace(nominal_graph=nominal, graph=nominal)
    with pytest.raises(ValueError):
        _with_local_detours({}, topology, nominal, {}, {"local_repair": "segment"}, MODE)
    with pytest.raises(ValueError):
        _with_local_detours(
            {}, topology, nominal, {}, {"local_repair": "square"}, "torus_weighted_lookahead"
        )


def test_guarded_detours_stay_loop_free_and_deliver_more() -> None:
    rng = random.Random(23)
    nominal = _grid()
    delivered = {"none": 0, "square": 0}
    for _ in range(30):
        live = nominal.copy()
        live.remove_edges_from(rng.sample(sorted(live.edges()), k=len(live.edges()) // 6))
        egress = [
            (float(rng.randint(400, 900)), sat) for sat in rng.sample(range(PLANES * SATS), 3)
        ]
        for repair in delivered:
            fstate = _forwarding_state(live, nominal, egress, repair)
            outcomes = [_walk(fstate, live, sat) for sat in nominal.nodes()]
            assert "loop" not in outcomes
            delivered[repair] += outcomes.count("delivered")

    assert delivered["square"] > delivered["none"]
