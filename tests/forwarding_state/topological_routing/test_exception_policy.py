import random
from types import SimpleNamespace

import networkx as nx
import pytest

from leopath.network_state.routing_algorithms.topological_routing.exception_policy import (
    apply_exception_policy,
)
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    LOCAL_DETOUR,
    _build_torus_weight_model,
    _EgressPotential,
    _install_next_hop,
    _routing_topological_distance,
    _scaled_gsl_distance,
    _with_local_detours,
)
from leopath.topology.constellation import ConstellationData
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress

PLANES = 6
SATS = 8
GS = SimpleNamespace(id=1000)
MODE = "torus_weighted_pivot"
CONSTELLATION = ConstellationData(PLANES, SATS, "20001.0", 1e6, 5e6, [])


def _address(sat_id: int) -> TopologicalNetworkAddress:
    return TopologicalNetworkAddress(0, sat_id // SATS, sat_id % SATS, 0)


def _grid() -> nx.Graph:
    graph = nx.Graph()
    for plane in range(PLANES):
        for index in range(SATS):
            sat = plane * SATS + index
            graph.add_edge(sat, plane * SATS + (index + 1) % SATS, weight=600.0)
            row_cost = 1200.0 + 100.0 * min(index, SATS - index)
            graph.add_edge(sat, ((plane + 1) % PLANES) * SATS + index, weight=row_cost)
    return graph


def _rules(live: nx.Graph, nominal: nx.Graph, egress: list, repair: str) -> dict:
    """Guarded topological forwarding state toward GS, interfaces named by neighbour id."""
    addresses = {sat: _address(sat) for sat in nominal.nodes()}
    model = _build_torus_weight_model(nominal, addresses, CONSTELLATION)
    candidates = [[(gsl, sat, addresses[sat]) for gsl, sat in egress]]
    potential = _EgressPotential(addresses, candidates, CONSTELLATION, MODE, model)
    plain = {
        sat: [
            (n, n, addresses[n], float(live.edges[sat, n]["weight"])) for n in live.neighbors(sat)
        ]
        for sat in live.nodes()
    }
    neighbours = _with_local_detours(
        plain,
        SimpleNamespace(nominal_graph=nominal, graph=live),
        live,
        addresses,
        {"local_repair": repair},
        MODE,
    )
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
            fstate, sat, addresses[sat], destination, 0, GS.id, neighbours[sat],
            CONSTELLATION, MODE, model, potential, "progress", None,
        )  # fmt: skip
    return fstate


def _interfaces(live: nx.Graph) -> dict:
    return {(sat, n): n for sat in live.nodes() for n in live.neighbors(sat)}


def _delivers(fstate: dict, live: nx.Graph, visible: set, start: int) -> bool:
    current, seen = start, set()
    while current not in seen:
        seen.add(current)
        entry = fstate.get((current, GS.id))
        if entry == ("GSL", GS.id):
            return current in visible
        if isinstance(entry, tuple) and entry and entry[0] == LOCAL_DETOUR:
            current = entry[3]
        elif isinstance(entry, int) and live.has_edge(current, entry):
            current = entry
        else:
            return False
    return False


def _failure_scenario(rng: random.Random, nominal: nx.Graph) -> tuple[nx.Graph, list]:
    live = nominal.copy()
    live.remove_edges_from(rng.sample(sorted(live.edges()), k=len(live.edges()) // 8))
    dead = rng.sample(range(PLANES * SATS), 3)
    live.remove_edges_from([edge for sat in dead for edge in list(live.edges(sat))])
    egress = [
        (float(rng.randint(400, 900)), sat)
        for sat in rng.sample([s for s in range(PLANES * SATS) if s not in dead], 3)
    ]
    return live, egress


@pytest.mark.parametrize("repair", ["none", "square"])
def test_exceptions_make_every_reachable_live_satellite_deliver(repair: str) -> None:
    rng = random.Random(41)
    nominal = _grid()
    for _ in range(20):
        live, egress = _failure_scenario(rng, nominal)
        fstate = _rules(live, nominal, egress, repair)
        report: dict = {}
        apply_exception_policy(
            fstate, live, _interfaces(live), nominal, [GS], [egress], "grow", LOCAL_DETOUR, report
        )

        visible = {sat for _gsl, sat in egress}
        reachable = {
            sat
            for sat in live.nodes()
            if live.degree(sat) > 0 and any(nx.has_path(live, sat, e) for e in visible)
        }
        assert all(_delivers(fstate, live, visible, sat) for sat in reachable)
        assert report["exception_unresolved"] == 0.0
        assert report["exception_entries"] <= report["exception_entries_one_pass"]


def test_no_failures_need_no_exceptions() -> None:
    grid = _grid()
    egress = [(500.0, 3), (700.0, 29)]
    fstate = _rules(grid, grid, egress, "none")
    report: dict = {}
    apply_exception_policy(
        fstate, grid, _interfaces(grid), grid, [GS], [egress], "grow", LOCAL_DETOUR, report
    )
    assert report["exception_entries"] == 0.0
    assert report["exception_entries_one_pass"] == 0.0


def test_policy_none_leaves_state_untouched_and_unknown_policies_fail() -> None:
    fstate = {(0, GS.id): 1}
    apply_exception_policy(fstate, nx.Graph(), {}, None, [GS], [[]], "none", LOCAL_DETOUR, None)
    assert fstate == {(0, GS.id): 1}
    with pytest.raises(ValueError):
        apply_exception_policy(fstate, nx.Graph(), {}, None, [GS], [[]], "all", LOCAL_DETOUR, None)
