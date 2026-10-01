"""Stretch splits into an egress-choice factor and a forwarding factor.

Under attachment addressing the egress is fixed by the destination address
rather than minimised at every satellite, so a suboptimal egress shows up in the
same number as a suboptimal path. The two are separated here: the shared basis,
which is the one headline figure, is the product of the egress factor and the
forwarding factor, and only the second is about forwarding.
"""

import networkx as nx
import pytest

from leopath.experiments.metrics import (
    compute_fixed_address_path_stretch,
    compute_path_stretch,
)

GS_SRC = 100
GS_DST = 101


def _stretch(fstate: dict, graph: nx.Graph, satellite_ids: list[int], visible: list) -> dict:
    return compute_path_stretch(
        fstate=fstate,
        topology_graph=graph,
        satellite_ids=satellite_ids,
        ground_station_ids=[GS_SRC, GS_DST],
        attachments=[(0, 10.0), (visible[1][0][1], visible[1][0][0])],
        interface_neighbor_map={},
        max_hops=len(satellite_ids) + 2,
        ground_station_satellites_in_range=visible,
    )


def _line(*weighted_edges) -> nx.Graph:
    graph = nx.Graph()
    graph.add_weighted_edges_from(weighted_edges)
    return graph


def test_the_two_factors_multiply_to_the_shared_basis() -> None:
    # Satellite 1 is the near egress, satellite 3 the far one. Forwarding takes
    # the long way round to 3, so both factors are above 1.
    graph = _line((0, 1, 10.0), (0, 2, 4.0), (2, 3, 4.0))
    visible = [[(10.0, 0)], [(5.0, 1), (5.0, 3)]]
    fstate = {
        (0, GS_DST): (2, 0, 0),
        (2, GS_DST): (3, 0, 0),
        (3, GS_DST): (GS_DST, 0, 0),
    }

    stats = _stretch(fstate, graph, [0, 1, 2, 3], visible)

    assert stats["delivery"]["delivered"] == 1.0
    shared = stats["distance_shared"]["mean"]
    egress = stats["distance_egress"]["mean"]
    forwarding = stats["distance"]["mean"]
    assert shared == egress * forwarding


def test_reaching_the_best_egress_leaves_no_egress_penalty() -> None:
    graph = _line((0, 1, 10.0))
    visible = [[(10.0, 0)], [(5.0, 1)]]
    fstate = {(0, GS_DST): (1, 0, 0), (1, GS_DST): (GS_DST, 0, 0)}

    stats = _stretch(fstate, graph, [0, 1], visible)

    assert stats["distance_egress"]["mean"] == 1.0
    assert stats["distance_shared"]["mean"] == stats["distance"]["mean"]


def test_a_worse_egress_is_charged_to_the_egress_factor_not_to_forwarding() -> None:
    # Both egresses are reachable. Forwarding is optimal toward whichever one it
    # aims at, but it aims at the more expensive satellite 3.
    graph = _line((0, 1, 2.0), (0, 3, 9.0))
    visible = [[(10.0, 0)], [(1.0, 1), (1.0, 3)]]
    fstate = {(0, GS_DST): (3, 0, 0), (3, GS_DST): (GS_DST, 0, 0)}

    stats = _stretch(fstate, graph, [0, 1, 3], visible)

    assert stats["delivery"]["delivered"] == 1.0
    assert stats["delivery"]["non_optimal_egress"] == 1.0
    # Forwarding did the best it could toward the egress it was given.
    assert stats["distance"]["mean"] == 1.0
    # The whole penalty sits in the egress factor.
    assert stats["distance_egress"]["mean"] > 1.0
    assert stats["distance_shared"]["mean"] == stats["distance_egress"]["mean"]


def test_hop_counts_decompose_the_same_way() -> None:
    graph = _line((0, 1, 10.0), (0, 2, 4.0), (2, 3, 4.0))
    visible = [[(10.0, 0)], [(5.0, 1), (5.0, 3)]]
    fstate = {
        (0, GS_DST): (2, 0, 0),
        (2, GS_DST): (3, 0, 0),
        (3, GS_DST): (GS_DST, 0, 0),
    }

    stats = _stretch(fstate, graph, [0, 1, 2, 3], visible)

    assert stats["hop_shared"]["mean"] == stats["hop_egress"]["mean"] * stats["hop"]["mean"]


def test_fixed_address_walk_keeps_the_flow_allocator_selection() -> None:
    graph = _line((0, 1, 4.0), (1, 2, 4.0))
    routes = {
        (GS_SRC, GS_DST): {
            "source_satellite": 0,
            "destination_satellite": 2,
            "source_gsl_distance": 1.0,
            "destination_gsl_distance": 1.0,
            "satellite_path": [0, 1, 2],
            "failure": None,
        },
        (GS_DST, GS_SRC): {
            "source_satellite": 2,
            "destination_satellite": 0,
            "source_gsl_distance": 1.0,
            "destination_gsl_distance": 1.0,
            "satellite_path": [2, 1, 0],
            "failure": None,
        },
    }

    stats = compute_fixed_address_path_stretch(
        routes,
        graph,
        [0, 1, 2],
        [GS_SRC, GS_DST],
        [[(1.0, 0)], [(1.0, 2)]],
        [(0, 1.0), (2, 1.0)],
    )

    assert stats["delivery"]["delivered"] == 2.0
    assert stats["delivery"]["source_selected_egress"] == 2.0
    assert stats["delivery"]["switched_egress"] == 0.0
    assert stats["distance_shared"]["mean"] == (
        stats["distance_egress"]["mean"] * stats["distance"]["mean"]
    )


def test_fixed_address_walk_cannot_finish_at_an_unselected_synonym() -> None:
    graph = _line((0, 1, 1.0), (1, 2, 1.0))
    routes = {
        (GS_SRC, GS_DST): {
            "source_satellite": 0,
            "destination_satellite": 2,
            "source_gsl_distance": 1.0,
            "destination_gsl_distance": 1.0,
            "satellite_path": [0, 1],
            "failure": None,
        },
        (GS_DST, GS_SRC): {
            "source_satellite": 1,
            "destination_satellite": 0,
            "source_gsl_distance": 1.0,
            "destination_gsl_distance": 1.0,
            "satellite_path": [1, 0],
            "failure": None,
        },
    }

    stats = compute_fixed_address_path_stretch(
        routes,
        graph,
        [0, 1, 2],
        [GS_SRC, GS_DST],
        [[(1.0, 0)], [(1.0, 1), (1.0, 2)]],
        [(0, 1.0), (1, 1.0)],
    )

    assert stats["delivery"]["delivered"] == 1.0
    assert stats["delivery"]["failure_dead_end"] == 1.0
    assert stats["delivery"]["switched_egress"] == 0.0


def test_delay_is_path_length_over_the_speed_of_light() -> None:
    from leopath.experiments.metrics import _record_delays

    delays, best, extra = [], [], []
    # 299 792.458 km takes one second; a 3 000 km path against a 2 400 km optimum.
    _record_delays(299_792_458.0, 299_792_458.0, delays, best, extra)
    _record_delays(3_000_000.0, 2_400_000.0, delays, best, extra)
    assert delays[0] == pytest.approx(1000.0)
    assert extra[0] == pytest.approx(0.0)
    assert delays[1] == pytest.approx(10.007, abs=1e-3)
    assert extra[1] == pytest.approx(2.001, abs=1e-3)


def test_isl_delay_factors_only_add_delay_and_are_the_same_from_both_ends() -> None:
    from leopath.experiments.eval_harness import apply_isl_delay_factors, isl_delay_factor

    assert isl_delay_factor(3, 7, 0.3, 1) == isl_delay_factor(7, 3, 0.3, 1)
    factors = [isl_delay_factor(a, a + 1, 0.3, 1) for a in range(200)]
    assert all(1.0 <= f <= 1.3 for f in factors) and len(set(factors)) > 150

    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1000.0)  # satellites 0 and 1
    graph.add_edge(1, 5, weight=500.0)  # 5 is a ground station
    apply_isl_delay_factors(graph, satellite_count=2, spread=0.3, seed=1)
    assert graph.edges[0, 1]["weight"] == pytest.approx(1000.0 * isl_delay_factor(0, 1, 0.3, 1))
    assert graph.edges[1, 5]["weight"] == 500.0
