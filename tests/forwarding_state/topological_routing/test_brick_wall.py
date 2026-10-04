"""The brick-wall pivot must be exact on a brick wall with unit hop costs.

On a three-terminal shell no single row (split a) or plane (split b) carries a
crossing, so the single-row pivot reports most of the shell unreachable. The
brick pivot prices the closed-form staircase of docs/isl-topology.md from each
pivot row; with unit hops it must reproduce the true hop distance everywhere.
"""

from collections import Counter
from types import SimpleNamespace

import networkx as nx
import pytest

from leopath.main import generate_brick_isls
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    _build_torus_weight_model,
    _torus_weighted_pivot_distance,
)


def _address(plane: int, slot: int) -> SimpleNamespace:
    satellite = SimpleNamespace(shell_id=0, plane_id=plane, sat_index=slot)
    return SimpleNamespace(get_satellite_address=lambda: satellite)


def _brick(planes: int, slots: int, split: str, seam: bool = False):
    graph = nx.Graph()
    graph.add_nodes_from(range(planes * slots))
    graph.add_edges_from(generate_brick_isls(planes, slots, split=split, seam=seam))
    nx.set_edge_attributes(graph, 1.0, "weight")
    addresses = {n: _address(n // slots, n % slots) for n in graph.nodes}
    constellation = SimpleNamespace(n_orbits=planes, n_sats_per_orbit=slots)
    return graph, addresses, constellation


CASES = [
    ("split a, torus", 8, 6, "a", False),
    ("split a, torus, long rows", 6, 10, "a", False),
    ("split b, torus", 8, 6, "b", False),
    ("split b, torus, long planes", 6, 10, "b", False),
    ("split a, cylinder", 7, 6, "a", True),
    ("split b, cylinder", 6, 8, "b", True),
]


@pytest.mark.parametrize("label,planes,slots,split,seam", CASES, ids=[c[0] for c in CASES])
def test_brick_pivot_equals_hop_distance(label, planes, slots, split, seam) -> None:
    graph, addresses, constellation = _brick(planes, slots, split, seam)
    model = _build_torus_weight_model(graph, addresses, constellation, wiring=f"brick_{split}")
    hops = dict(nx.all_pairs_shortest_path_length(graph))
    for source in graph.nodes:
        for destination in graph.nodes:
            estimate = _torus_weighted_pivot_distance(
                addresses[source], addresses[destination], model
            )
            assert estimate == pytest.approx(hops[source][destination]), (source, destination)


def test_single_row_pivot_cannot_see_across_a_brick_wall() -> None:
    graph, addresses, constellation = _brick(8, 6, "a")
    model = _build_torus_weight_model(graph, addresses, constellation, wiring="plus_grid")
    # Three planes over on the same row: every pivot row has a gap in its rungs.
    assert _torus_weighted_pivot_distance(addresses[0], addresses[3 * 6], model) == float("inf")


@pytest.mark.parametrize("label,planes,slots,split,seam", CASES, ids=[c[0] for c in CASES])
def test_greedy_forwarding_on_the_brick_pivot_never_gets_stuck(
    label, planes, slots, split, seam
) -> None:
    """With an exact metric some neighbour is always strictly closer."""
    graph, addresses, constellation = _brick(planes, slots, split, seam)
    model = _build_torus_weight_model(graph, addresses, constellation, wiring=f"brick_{split}")

    def distance(a: int, b: int) -> float:
        return _torus_weighted_pivot_distance(addresses[a], addresses[b], model)

    stuck = Counter()
    for source in graph.nodes:
        for destination in graph.nodes:
            if source == destination:
                continue
            here = distance(source, destination)
            if min(distance(n, destination) for n in graph.neighbors(source)) >= here:
                stuck[(source, destination)] += 1
    assert not stuck


@pytest.mark.parametrize(
    "planes,slots,split,seam",
    [(7, 6, "a", False), (6, 7, "b", False)],
)
def test_brick_rejects_a_parity_that_cannot_close(planes, slots, split, seam) -> None:
    with pytest.raises(ValueError, match="even number"):
        generate_brick_isls(planes, slots, split=split, seam=seam)


@pytest.mark.parametrize("split", ["a", "b"])
def test_greedy_walks_on_real_lengths_do_not_loop(split) -> None:
    """Priced with real link lengths, the estimate must stay loop-free.

    The rule picks the neighbour minimising link length plus its estimate. An
    earlier version priced rungs at the pivot row's own length; the best pivot
    then shifted from hop to hop and walks cycled.
    """
    from leopath.topology.walker_geometry import WalkerShell

    shell = WalkerShell(12, 10, 53.0, 15.05)
    graph, addresses, constellation = _brick(shell.planes, shell.sats_per_plane, split)
    for a, b in graph.edges:
        graph.edges[a, b]["weight"] = shell.distance_m(
            (a // shell.sats_per_plane, a % shell.sats_per_plane),
            (b // shell.sats_per_plane, b % shell.sats_per_plane),
            900.0,
        )
    model = _build_torus_weight_model(graph, addresses, constellation, wiring=f"brick_{split}")

    def estimate(a: int, b: int) -> float:
        return _torus_weighted_pivot_distance(addresses[a], addresses[b], model)

    looped = 0
    for source in graph.nodes:
        for destination in graph.nodes:
            current, seen = source, {source}
            while current != destination:
                current = min(
                    graph.neighbors(current),
                    key=lambda n: (graph.edges[current, n]["weight"] + estimate(n, destination), n),
                )
                if current in seen:
                    looped += 1
                    break
                seen.add(current)
    assert looped == 0
