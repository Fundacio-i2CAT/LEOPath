import networkx as nx
from astropy.time import Time
from unittest.mock import patch

from leopath.network_state.routing_algorithms.explicit_path_routing.explicit_path_routing import (
    ExplicitPathRoutingAlgorithm,
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
    assert plans[(0, 100)]["planned_dst_sat_id"] == 3
    assert plans[(3, 100)]["satellite_path"] == [3]


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
    assert output["route_plans"][(0, 100)]["planned_dst_sat_id"] == 3
    assert output["fstate"][(0, 100)] == (1, 0, 0)
    assert output["control_plane"]["sample_route_plans"][0]["waypoint_satellites"] == [2, 3]


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
            "route_plans": {(0, 100): {"satellite_path": [0, 1, 2, 3], "planned_dst_sat_id": 3}},
            "control_plane": {},
        },
        {
            "fstate": {},
            "bandwidth": {},
            "route_plans": {(0, 100): {"satellite_path": [0, 1, 2, 3], "planned_dst_sat_id": 3}},
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
        (0, 100): {"satellite_path": [0, 1, 2, 3], "planned_dst_sat_id": 3}
    }
    assert second_call["current_ground_station_satellites_in_range"] == [[(2.0, 2)]]


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
        "route_plans": {(0, 100): {"satellite_path": [0, 1, 2, 3], "planned_dst_sat_id": 3}},
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
