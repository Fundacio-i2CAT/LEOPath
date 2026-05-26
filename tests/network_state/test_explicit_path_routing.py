import networkx as nx
from astropy.time import Time
from unittest.mock import patch

import leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing_algorithm as explicit_path_module
from leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing import (
    ExplicitPathRoutingAlgorithm,
    _strip_dynamic_snapshot_fields,
)
from leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing_algorithm import (
    _build_route_plans,
    _build_waypoint_plans,
    algorithm_explicit_path_routing,
)


class _StubGslStrategy:
    def __init__(self, attachments):
        self._attachments = attachments

    def select_attachments(self, topology_with_isls, ground_stations, current_time):
        del topology_with_isls, ground_stations, current_time
        return self._attachments


def _make_topology():
    satellites = [type("Sat", (), {"id": sat_id, "number_isls": 2})() for sat_id in range(4)]
    topology = type("Topology", (), {})()
    topology.graph = nx.Graph()
    topology.graph.add_weighted_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
            (0, 3, 10.0),
        ]
    )
    topology.get_satellites = lambda: satellites
    topology.get_satellite = lambda sat_id: satellites[sat_id]
    topology.sat_neighbor_to_if = {
        (0, 1): 0,
        (1, 0): 0,
        (1, 2): 1,
        (2, 1): 1,
        (2, 3): 0,
        (3, 2): 0,
        (0, 3): 1,
        (3, 0): 1,
    }
    return topology


def test_build_route_plans_pins_shortest_satellite_path() -> None:
    topology = _make_topology()
    ground_stations = [type("GroundStation", (), {"id": 100})()]

    plans = _build_route_plans(topology, ground_stations, [[(1.0, 3)]])

    assert plans[(0, 100)]["satellite_path"] == [0, 1, 2, 3]
    assert plans[(0, 100)]["adjacency_sid_list"] == [1, 2, 3]
    assert plans[(0, 100)]["srv6_srh_bytes"] == 56
    assert plans[(0, 100)]["backup_adjacency_sid_list"] == []
    assert plans[(0, 100)]["local_protection_mode"] == "none"
    assert plans[(0, 100)]["forwarding_mode"] == "strict_adjacency_header"
    assert plans[(0, 100)]["planned_dst_sat_id"] == 3
    assert plans[(3, 100)]["satellite_path"] == [3]
    assert plans[(3, 100)]["adjacency_sid_list"] == []
    assert plans[(3, 100)]["strict_header_bytes"] == 0
    assert plans[(3, 100)]["srv6_srh_bytes"] == 0


def test_build_route_plans_can_include_backup_adjacencies() -> None:
    topology = _make_topology()
    ground_stations = [type("GroundStation", (), {"id": 100})()]

    plans = _build_route_plans(
        topology,
        ground_stations,
        [[(1.0, 3)]],
        include_backup_adjacencies=True,
    )

    assert plans[(0, 100)]["backup_adjacency_sid_list"] == [None, None, None]
    assert plans[(0, 100)]["local_protection_mode"] == "single_hop_rejoin"


def test_build_route_plans_selects_best_visible_egress_per_source() -> None:
    topology = _make_topology()
    ground_stations = [type("GroundStation", (), {"id": 100})()]

    plans = _build_route_plans(
        topology,
        ground_stations,
        [[(2.0, 3), (1.0, 1)]],
    )

    assert plans[(0, 100)]["planned_dst_sat_id"] == 1
    assert plans[(0, 100)]["satellite_path"] == [0, 1]
    assert plans[(3, 100)]["planned_dst_sat_id"] == 3
    assert plans[(3, 100)]["satellite_path"] == [3]


def test_dynamic_egress_route_plans_keep_core_path_and_repair_final_handoff() -> None:
    topology = _make_topology()
    ground_stations = [type("GroundStation", (), {"id": 100})()]

    core_plans = _build_route_plans(
        topology,
        ground_stations,
        [[(1.0, 3)]],
        final_egress_mode="dynamic",
    )
    output = algorithm_explicit_path_routing(
        constellation_data=type("ConstellationData", (), {"number_of_satellites": 4})(),
        ground_stations=ground_stations,
        topology_with_isls=topology,
        gsl_attachment_strategy=_StubGslStrategy([(1.0, 3)]),
        current_time=Time("2000-01-01T00:00:00"),
        list_gsl_interfaces_info=[],
        algorithm_params={"final_egress_mode": "dynamic"},
        current_ground_station_satellites_in_range=[[(1.0, 2)]],
        cached_route_plans=core_plans,
    )

    repaired_plan = output["route_plans"][(0, 100)]
    assert repaired_plan["satellite_path"] == [0, 1, 2, 3]
    assert repaired_plan["planned_dst_sat_id"] == 3
    assert repaired_plan["current_dst_sat_id"] == 2
    assert repaired_plan["egress_repair_required"] is True
    assert repaired_plan["egress_repair_satellite_path"] == [3, 2]
    assert repaired_plan["final_egress_mode"] == "dynamic"


def test_dynamic_egress_exposes_repair_first_hop_when_anchor_is_ingress() -> None:
    topology = _make_topology()
    ground_stations = [type("GroundStation", (), {"id": 100})()]
    core_plans = _build_route_plans(
        topology,
        ground_stations,
        [[(1.0, 3)]],
        final_egress_mode="dynamic",
    )

    output = algorithm_explicit_path_routing(
        constellation_data=type("ConstellationData", (), {"number_of_satellites": 4})(),
        ground_stations=ground_stations,
        topology_with_isls=topology,
        gsl_attachment_strategy=_StubGslStrategy([(1.0, 3)]),
        current_time=Time("2000-01-01T00:00:00"),
        list_gsl_interfaces_info=[],
        algorithm_params={"final_egress_mode": "dynamic"},
        current_ground_station_satellites_in_range=[[(1.0, 2)]],
        cached_route_plans=core_plans,
    )

    assert output["route_plans"][(3, 100)]["egress_repair_satellite_path"] == [3, 2]
    assert output["fstate"][(3, 100)] == (2, 0, 0)


def test_strip_dynamic_snapshot_fields_preserves_cached_core_route() -> None:
    route_plans = {
        (0, 100): {
            "satellite_path": [0, 1, 2, 3],
            "planned_dst_sat_id": 3,
            "current_dst_sat_id": 2,
            "egress_repair_satellite_path": [3, 2],
        }
    }

    stripped = _strip_dynamic_snapshot_fields(route_plans)

    assert stripped[(0, 100)] == {
        "satellite_path": [0, 1, 2, 3],
        "planned_dst_sat_id": 3,
    }


def test_waypoint_sampling_uses_segment_count() -> None:
    waypoints = _build_waypoint_plans(
        {(0, 100): {"satellite_path": [0, 1, 2, 3], "planned_dst_sat_id": 3}},
        segment_count=2,
    )

    assert waypoints[(0, 100)] == [2, 3]


def test_explicit_path_algorithm_returns_route_plans_and_first_hop_proxy() -> None:
    topology = _make_topology()
    ground_stations = [type("GroundStation", (), {"id": 100})()]
    constellation_data = type("ConstellationData", (), {"number_of_satellites": 4})()

    output = algorithm_explicit_path_routing(
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        topology_with_isls=topology,
        gsl_attachment_strategy=_StubGslStrategy([(1.0, 3)]),
        current_time=Time("2000-01-01T00:00:00"),
        list_gsl_interfaces_info=[
            {"id": 0, "aggregate_max_bandwidth": 1.0},
            {"id": 1, "aggregate_max_bandwidth": 1.0},
            {"id": 2, "aggregate_max_bandwidth": 1.0},
            {"id": 3, "aggregate_max_bandwidth": 1.0},
        ],
        algorithm_params={"segment_count": 2},
    )

    assert output["route_plans"][(0, 100)]["satellite_path"] == [0, 1, 2, 3]
    assert output["route_plans"][(0, 100)]["adjacency_sid_list"] == [1, 2, 3]
    assert output["route_plans"][(0, 100)]["backup_adjacency_sid_list"] == []
    assert output["route_plans"][(0, 100)]["planned_dst_sat_id"] == 3
    assert output["fstate"][(0, 100)] == (1, 0, 0)
    assert output["control_plane"]["sample_route_plans"][0]["satellite_path"] == [0, 1, 2, 3]
    assert output["control_plane"]["sample_route_plans"][0]["adjacency_sid_list"] == [1, 2, 3]
    assert output["control_plane"]["sample_route_plans"][0]["backup_adjacency_sid_list"] == []
    assert output["control_plane"]["sample_route_plans"][0]["waypoint_satellites"] == [2, 3]


def test_build_route_plans_uses_one_augmented_tree_per_ground_station() -> None:
    topology = _make_topology()
    ground_stations = [
        type("GroundStation", (), {"id": 100})(),
        type("GroundStation", (), {"id": 101})(),
    ]

    with patch.object(
        explicit_path_module,
        "_single_destination_gs_shortest_paths",
        wraps=explicit_path_module._single_destination_gs_shortest_paths,
    ) as mock_single_destination_gs_shortest_paths:
        plans = _build_route_plans(
            topology,
            ground_stations,
            [[(1.0, 3)], [(2.0, 3)]],
        )

    assert mock_single_destination_gs_shortest_paths.call_count == 2
    assert plans[(0, 100)]["satellite_path"] == [0, 1, 2, 3]
    assert plans[(0, 101)]["satellite_path"] == [0, 1, 2, 3]


@patch(
    "leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing.algorithm_explicit_path_routing"
)
@patch(
    "leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing.GSLAttachmentFactory.get_strategy"
)
def test_explicit_path_refresh_reuses_cached_route_plans(
    mock_get_strategy,
    mock_algorithm_explicit_path_routing,
) -> None:
    algorithm = ExplicitPathRoutingAlgorithm()
    mock_get_strategy.return_value = object()
    mock_algorithm_explicit_path_routing.side_effect = [
        {
            "fstate": {},
            "bandwidth": {},
            "route_plans": {
                (0, 100): {
                    "satellite_path": [0, 1, 2, 3],
                    "adjacency_sid_list": [1, 2, 3],
                    "planned_dst_sat_id": 3,
                }
            },
            "control_plane": {},
        },
        {
            "fstate": {},
            "bandwidth": {},
            "route_plans": {
                (0, 100): {
                    "satellite_path": [0, 1, 2, 3],
                    "adjacency_sid_list": [1, 2, 3],
                    "planned_dst_sat_id": 3,
                }
            },
            "control_plane": {},
        },
    ]

    params = {"segment_count": 2, "segment_refresh_interval_steps": 2, "time_step_minutes": 5}
    constellation_data = type("ConstellationData", (), {"number_of_satellites": 4})()
    ground_stations = [type("GroundStation", (), {"id": 100})()]
    topology = _make_topology()

    algorithm.compute_state(
        time_since_epoch_ns=0,
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        topology_with_isls=topology,
        ground_station_satellites_in_range=[[(1.0, 3)]],
        list_gsl_interfaces_info=[],
        algorithm_params=params,
    )
    algorithm.compute_state(
        time_since_epoch_ns=int(5 * 60 * 1e9),
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        topology_with_isls=topology,
        ground_station_satellites_in_range=[[(2.0, 2)]],
        list_gsl_interfaces_info=[],
        algorithm_params=params,
    )

    first_call = mock_algorithm_explicit_path_routing.call_args_list[0].kwargs
    second_call = mock_algorithm_explicit_path_routing.call_args_list[1].kwargs
    assert first_call["cached_route_plans"] is None
    assert second_call["cached_route_plans"] == {
        (0, 100): {
            "satellite_path": [0, 1, 2, 3],
            "adjacency_sid_list": [1, 2, 3],
            "planned_dst_sat_id": 3,
        }
    }
    assert second_call["current_ground_station_satellites_in_range"] == [[(2.0, 2)]]


@patch(
    "leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing.algorithm_explicit_path_routing"
)
@patch(
    "leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing.GSLAttachmentFactory.get_strategy"
)
def test_explicit_path_control_plane_reports_refresh_behavior(
    mock_get_strategy,
    mock_algorithm_explicit_path_routing,
) -> None:
    algorithm = ExplicitPathRoutingAlgorithm()
    mock_get_strategy.return_value = object()
    mock_algorithm_explicit_path_routing.side_effect = [
        {
            "fstate": {},
            "bandwidth": {},
            "route_plans": {
                (0, 100): {
                    "satellite_path": [0, 1, 2, 3],
                    "adjacency_sid_list": [1, 2, 3],
                    "planned_dst_sat_id": 3,
                }
            },
            "control_plane": {},
        },
        {
            "fstate": {},
            "bandwidth": {},
            "route_plans": {
                (0, 100): {
                    "satellite_path": [0, 1, 2, 3],
                    "adjacency_sid_list": [1, 2, 3],
                    "planned_dst_sat_id": 3,
                }
            },
            "control_plane": {},
        },
    ]

    params = {"segment_count": 2, "segment_refresh_interval_steps": 3, "time_step_minutes": 5}
    constellation_data = type("ConstellationData", (), {"number_of_satellites": 4})()
    ground_stations = [type("GroundStation", (), {"id": 100})()]
    topology = _make_topology()

    first_output = algorithm.compute_state(
        time_since_epoch_ns=0,
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        topology_with_isls=topology,
        ground_station_satellites_in_range=[[(1.0, 3)]],
        list_gsl_interfaces_info=[],
        algorithm_params=params,
    )
    second_output = algorithm.compute_state(
        time_since_epoch_ns=int(5 * 60 * 1e9),
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        topology_with_isls=topology,
        ground_station_satellites_in_range=[[(2.0, 2)]],
        list_gsl_interfaces_info=[],
        algorithm_params=params,
    )

    assert first_output["control_plane"]["effective_refresh_interval_steps"] == 3
    assert first_output["control_plane"]["used_cached_route_plans"] is False
    assert first_output["control_plane"]["planning_step_index"] == 0
    assert second_output["control_plane"]["effective_refresh_interval_steps"] == 3
    assert second_output["control_plane"]["used_cached_route_plans"] is True
    assert second_output["control_plane"]["planning_step_index"] == 0


@patch(
    "leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing.algorithm_explicit_path_routing"
)
@patch(
    "leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing.GSLAttachmentFactory.get_strategy"
)
def test_explicit_path_refresh_boundary_replans(
    mock_get_strategy,
    mock_algorithm_explicit_path_routing,
) -> None:
    algorithm = ExplicitPathRoutingAlgorithm()
    mock_get_strategy.return_value = object()
    mock_algorithm_explicit_path_routing.return_value = {
        "fstate": {},
        "bandwidth": {},
        "route_plans": {
            (0, 100): {
                "satellite_path": [0, 1, 2, 3],
                "adjacency_sid_list": [1, 2, 3],
                "planned_dst_sat_id": 3,
            }
        },
        "control_plane": {},
    }

    params = {"segment_count": 2, "segment_refresh_interval_steps": 2, "time_step_minutes": 5}
    constellation_data = type("ConstellationData", (), {"number_of_satellites": 4})()
    ground_stations = [type("GroundStation", (), {"id": 100})()]
    topology = _make_topology()

    algorithm.compute_state(
        time_since_epoch_ns=0,
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        topology_with_isls=topology,
        ground_station_satellites_in_range=[[(1.0, 3)]],
        list_gsl_interfaces_info=[],
        algorithm_params=params,
    )
    algorithm.compute_state(
        time_since_epoch_ns=int(10 * 60 * 1e9),
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        topology_with_isls=topology,
        ground_station_satellites_in_range=[[(2.0, 2)]],
        list_gsl_interfaces_info=[],
        algorithm_params=params,
    )

    second_call = mock_algorithm_explicit_path_routing.call_args_list[1].kwargs
    assert second_call["cached_route_plans"] is None
