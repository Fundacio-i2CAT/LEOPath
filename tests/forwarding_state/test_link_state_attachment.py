"""Link-state under attachment addressing.

Topological routing under ``gs_addressing: attachment`` can deliver to a ground
station only through the one satellite it is attached to. Link-state gets the same
restriction here, so the two are compared on equal terms, while plain link-state
keeps every visible egress and stays the optimum both are scored against.
"""

import pytest

from leopath.experiments.eval_harness import prepare_algorithm_params
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


def test_link_state_attaches_where_topological_routing_does() -> None:
    # The comparison is only fair if both algorithms see the same single egress.
    candidates = [
        [(distance, sat, TopologicalNetworkAddress(0, sat // 8, sat % 8, 0)) for distance, sat in v]
        for v in VISIBILITY
    ]
    topological = [[(c[0], c[1]) for c in cands] for cands in _select_gs_attachments(candidates)]

    assert egresses_for_addressing(VISIBILITY, "attachment") == topological


def test_an_unknown_addressing_policy_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown gs_addressing"):
        egresses_for_addressing(VISIBILITY, "nearest-thing")


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
