"""Exception entries keyed on the destination address, for fixed-address walks.

Under attachment addressing every flow walks toward one fixed satellite
address. Where the guarded rule breaks on a damaged grid, the walk grows
exception entries keyed on (satellite, destination satellite), the station-
independent counterpart of ``exception_policy: grow``.
"""

import random

import networkx as nx

from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    _build_torus_weight_model,
    _walk_fixed_topological_address,
    _walk_with_exception_entries,
)
from leopath.topology.constellation import ConstellationData
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress

PLANES, SATS = 6, 8
MODE = "torus_weighted_pivot"
CONSTELLATION = ConstellationData(PLANES, SATS, "20001.0", 1e6, 5e6, [])


def _grid() -> nx.Graph:
    graph = nx.Graph()
    for plane in range(PLANES):
        for index in range(SATS):
            sat = plane * SATS + index
            graph.add_edge(sat, plane * SATS + (index + 1) % SATS, weight=600.0)
            graph.add_edge(
                sat,
                ((plane + 1) % PLANES) * SATS + index,
                weight=1200.0 + 100.0 * min(index, SATS - index),
            )
    return graph


def _setup(seed: int, failed_share: float = 0.2):
    nominal = _grid()
    live = nominal.copy()
    rng = random.Random(seed)
    live.remove_edges_from(
        rng.sample(sorted(nominal.edges()), int(failed_share * nominal.number_of_edges()))
    )
    addresses = {sat: TopologicalNetworkAddress(0, sat // SATS, sat % SATS, 0) for sat in nominal}
    model = _build_torus_weight_model(nominal, addresses, CONSTELLATION)
    neighbours = {
        sat: [
            (n, n, addresses[n], float(live.edges[sat, n]["weight"])) for n in live.neighbors(sat)
        ]
        for sat in live.nodes()
    }
    return live, addresses, model, neighbours


def _args(src, dst, addresses, model, neighbours):
    return (src, dst, addresses[dst], addresses, neighbours, CONSTELLATION, MODE, model, "progress")


def test_grown_entries_deliver_every_reachable_pair_the_rule_misses() -> None:
    plain_failures = 0
    for seed in range(4):
        live, addresses, model, neighbours = _setup(seed)
        exceptions: dict = {}
        toward: dict = {}
        for src in live.nodes():
            for dst in live.nodes():
                if src == dst:
                    continue
                args = _args(src, dst, addresses, model, neighbours)
                _path, plain = _walk_fixed_topological_address(*args)
                plain_failures += plain is not None
                path, failure = _walk_with_exception_entries(args, dst, live, exceptions, toward)
                if nx.has_path(live, src, dst):
                    assert failure is None, (seed, src, dst)
                    assert path[0] == src and path[-1] == dst
                    assert all(live.has_edge(a, b) for a, b in zip(path, path[1:]))
    # The test only means something if the guarded rule alone fails somewhere.
    assert plain_failures > 0


def test_entries_are_keyed_on_the_destination_and_shared_between_flows() -> None:
    live, addresses, model, neighbours = _setup(1)
    exceptions: dict = {}
    toward: dict = {}
    for src in live.nodes():
        for dst in live.nodes():
            if src != dst and nx.has_path(live, src, dst):
                _walk_with_exception_entries(
                    _args(src, dst, addresses, model, neighbours), dst, live, exceptions, toward
                )
    count = len(exceptions)
    # A second pass over the same flows finds every entry already in place.
    for src in live.nodes():
        for dst in live.nodes():
            if src != dst and nx.has_path(live, src, dst):
                _walk_with_exception_entries(
                    _args(src, dst, addresses, model, neighbours), dst, live, exceptions, toward
                )
    assert len(exceptions) == count
    assert all(live.has_edge(sat, hop) for (sat, _dst), hop in exceptions.items())


def test_a_partitioned_destination_fails_without_growing_forever() -> None:
    live, addresses, model, neighbours = _setup(0, failed_share=0.0)
    island = 5
    live.remove_edges_from(list(live.edges(island)))
    neighbours = {
        sat: [
            (n, n, addresses[n], float(live.edges[sat, n]["weight"])) for n in live.neighbors(sat)
        ]
        for sat in live.nodes()
    }
    exceptions: dict = {}
    path, failure = _walk_with_exception_entries(
        _args(0, island, addresses, model, neighbours), island, live, exceptions, {}
    )
    assert failure is not None
    assert len(exceptions) <= live.number_of_nodes()


def test_address_pair_selection_skips_pairs_split_by_a_partition() -> None:
    from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
        _live_components,
        _select_topological_address_pair,
    )

    live, addresses, model, neighbours = _setup(0, failed_share=0.0)
    # Cut every rung between planes 2-3 and 5-0: planes 0-2 and 3-5 split apart.
    live.remove_edges_from(
        [(p * SATS + s, ((p + 1) % PLANES) * SATS + s) for p in (2, 5) for s in range(SATS)]
    )
    component = _live_components(live)
    source = [(1.0, 0, addresses[0])]  # plane 0
    # Destination synonyms: plane 5 is one rung away across the wrap, which is cut;
    # plane 2, slot 4 is farther by the estimate but still reachable.
    near_but_cut, far_but_live = 5 * SATS, 2 * SATS + 4
    destinations = [
        (1.0, near_but_cut, addresses[near_but_cut]),
        (1.0, far_but_live, addresses[far_but_live]),
    ]
    args = (addresses, neighbours, CONSTELLATION, MODE, model)

    blind = _select_topological_address_pair(source, destinations, *args)
    aware = _select_topological_address_pair(source, destinations, *args, component)

    assert component[0] != component[near_but_cut]
    assert blind[3] == near_but_cut  # the estimate alone walks into the partition
    assert aware[3] == far_but_live
    assert _select_topological_address_pair(source, destinations[:1], *args, component) is None
