"""Link-state under attachment addressing.

Topological routing under ``gs_addressing: attachment`` can deliver to a ground
station only through the one satellite it is attached to. Link-state gets the same
restriction here, so the two are compared on equal terms, while plain link-state
keeps every visible egress and stays the optimum both are scored against.
"""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from leopath.experiments.eval_harness import prepare_algorithm_params
from leopath.network_state.routing_algorithms.flow_allocation import (
    new_flow_allocation_counters,
    update_current_addresses,
)
from leopath.network_state.routing_algorithms.shortest_path_link_state_routing.fstate_calculation import (  # noqa: E501
    _build_fixed_address_routes,
    _select_link_state_address_pair,
)
from leopath.network_state.routing_algorithms.shortest_path_link_state_routing.shortest_path_link_state_routing import (  # noqa: E501
    egresses_for_addressing,
)
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    _select_gs_attachments,
)
from leopath.topology.satellite.topological_network_address import TopologicalNetworkAddress

VISIBILITY = [
    [(900_000.0, 12), (400_000.0, 7), (650_000.0, 20)],
    [],
    [(300_000.0, 31)],
]


def test_visibility_keeps_every_egress() -> None:
    assert egresses_for_addressing(VISIBILITY, "visibility") is VISIBILITY


def test_attachment_keeps_only_the_nearest_egress() -> None:
    assert egresses_for_addressing(VISIBILITY, "attachment") == [
        [(400_000.0, 7)],
        [],
        [(300_000.0, 31)],
    ]


def test_attachment_count_keeps_k_nearest_egresses() -> None:
    assert egresses_for_addressing(VISIBILITY, "attachment", attachment_count=2) == [
        [(400_000.0, 7), (650_000.0, 20)],
        [],
        [(300_000.0, 31)],
    ]


def test_link_state_attaches_where_topological_routing_does() -> None:
    # The comparison is only fair if both algorithms see the same single egress.
    candidates = [
        [(distance, sat, TopologicalNetworkAddress(0, sat // 8, sat % 8, 0)) for distance, sat in v]
        for v in VISIBILITY
    ]
    topological = [[(c[0], c[1]) for c in cands] for cands in _select_gs_attachments(candidates)]

    assert egresses_for_addressing(VISIBILITY, "attachment") == topological


def test_link_state_selects_one_best_address_pair_at_flow_allocation() -> None:
    distances = np.asarray(
        [
            [0.0, 10.0, 2.0],
            [10.0, 0.0, 3.0],
            [2.0, 3.0, 0.0],
        ]
    )

    selected = _select_link_state_address_pair(
        [(1.0, 0), (1.0, 1)],
        [(1.0, 1), (1.0, 2)],
        {0: 0, 1: 1, 2: 2},
        distances,
    )

    assert selected == (1.0, 1, 1.0, 1)


def test_an_unknown_addressing_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown gs_addressing"):
        egresses_for_addressing(VISIBILITY, "nearest-thing")


def test_harness_passes_multihoming_parameters_to_both_algorithms() -> None:
    for algorithm in ("shortest_path_link_state", "topological_routing"):
        params = prepare_algorithm_params(
            simulation_config={"time_step_minutes": 1},
            algorithm_name=algorithm,
            segment_count=None,
            segment_refresh_interval_steps=None,
            plane_weight=None,
            sat_weight=None,
            shell_weight=None,
            distance_mode=None,
            explicit_final_egress_mode=None,
            time_step_minutes=None,
            gs_addressing="attachment",
            gs_attachment_count=4,
            gs_attachment_policy="exclusive",
        )

        assert params["gs_attachment_count"] == 4
        assert params["gs_attachment_policy"] == "exclusive"


@pytest.mark.parametrize(
    "algorithm, expected",
    [
        ("shortest_path_link_state", "attachment"),
        ("topological_routing", "attachment"),
        ("dra_routing", None),
    ],
)
def test_the_harness_passes_the_policy_to_link_state_and_topological(
    algorithm: str, expected: str | None
) -> None:
    params = prepare_algorithm_params(
        simulation_config={"time_step_minutes": 1},
        algorithm_name=algorithm,
        segment_count=None,
        segment_refresh_interval_steps=None,
        plane_weight=None,
        sat_weight=None,
        shell_weight=None,
        distance_mode=None,
        explicit_final_egress_mode=None,
        time_step_minutes=None,
        gs_addressing="attachment",
    )

    assert params.get("gs_addressing") == expected


def test_link_state_routes_between_the_same_current_addresses_as_topological() -> None:
    graph = nx.Graph()
    graph.add_edge(0, 1, weight=1.0)
    graph.add_edge(1, 2, weight=1.0)
    distances = np.asarray([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    stations = [
        SimpleNamespace(id=100, current_address_satellite_id=None, allocated_address_pairs={}),
        SimpleNamespace(id=101, current_address_satellite_id=None, allocated_address_pairs={}),
    ]
    # Sat 1 would give a zero-hop best pair; current addresses are the nearest.
    candidates = [[(100.0, 0), (200.0, 1)], [(100.0, 2), (200.0, 1)]]
    counters = new_flow_allocation_counters()
    update_current_addresses(stations, candidates, "sticky_nearest", counters)

    routes = _build_fixed_address_routes(
        "sticky_nearest", stations, candidates, graph, {0: 0, 1: 1, 2: 2}, distances, counters
    )

    assert routes[(100, 101)]["satellite_path"] == [0, 1, 2]
