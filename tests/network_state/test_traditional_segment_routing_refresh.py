import unittest
from unittest.mock import patch

import networkx as nx
from astropy.time import Time

from leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing import (
    TraditionalSegmentRoutingAlgorithm,
)
from leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing_algorithm import (
    _build_segment_plans,
    satellite_id_from_srv6_sid,
    satellite_srv6_sid,
)


class TestTraditionalSegmentRoutingRefresh(unittest.TestCase):
    def setUp(self):
        self.algorithm = TraditionalSegmentRoutingAlgorithm()
        self.constellation_data = object()
        self.ground_stations = [object()]
        self.topology = object()
        self.visibility = [[(10.0, 7)]]
        self.gsl_info = []

    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing.algorithm_traditional_segment_routing"
    )
    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing._build_segment_plans"
    )
    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing.GSLAttachmentFactory.get_strategy"
    )
    def test_refresh_interval_reuses_cached_segment_plans(
        self,
        mock_get_strategy,
        mock_build_segment_plans,
        mock_algorithm_traditional_segment_routing,
    ):
        mock_get_strategy.return_value = object()
        mock_build_segment_plans.return_value = {("cached", 0): ["fd00:10:0:1::1"]}
        mock_algorithm_traditional_segment_routing.return_value = {"fstate": {}, "bandwidth": {}}

        params = {
            "segment_count": 2,
            "segment_refresh_interval_steps": 2,
            "time_step_minutes": 5,
        }

        self.algorithm.compute_state(
            time_since_epoch_ns=0,
            constellation_data=self.constellation_data,
            ground_stations=self.ground_stations,
            topology_with_isls=self.topology,
            ground_station_satellites_in_range=self.visibility,
            list_gsl_interfaces_info=self.gsl_info,
            algorithm_params=params,
        )
        self.algorithm.compute_state(
            time_since_epoch_ns=int(5 * 60 * 1e9),
            constellation_data=self.constellation_data,
            ground_stations=self.ground_stations,
            topology_with_isls=self.topology,
            ground_station_satellites_in_range=self.visibility,
            list_gsl_interfaces_info=self.gsl_info,
            algorithm_params=params,
        )

        self.assertEqual(mock_build_segment_plans.call_count, 1)
        self.assertEqual(mock_algorithm_traditional_segment_routing.call_count, 2)

        first_call = mock_algorithm_traditional_segment_routing.call_args_list[0].kwargs
        second_call = mock_algorithm_traditional_segment_routing.call_args_list[1].kwargs
        self.assertEqual(first_call["segment_plans"], {("cached", 0): ["fd00:10:0:1::1"]})
        self.assertEqual(second_call["segment_plans"], {("cached", 0): ["fd00:10:0:1::1"]})

    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing.algorithm_traditional_segment_routing"
    )
    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing._compute_ground_station_satellites_in_range"
    )
    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing._compute_isls"
    )
    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing._build_topologies"
    )
    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing._build_segment_plans"
    )
    @patch(
        "leopath.network_state.routing_algorithms.traditional_segment_routing.traditional_segment_routing.GSLAttachmentFactory.get_strategy"
    )
    def test_predictive_refresh_uses_future_visibility(
        self,
        mock_get_strategy,
        mock_build_segment_plans,
        mock_build_topologies,
        mock_compute_isls,
        mock_compute_visibility,
        mock_algorithm_traditional_segment_routing,
    ):
        mock_get_strategy.return_value = object()
        planning_topology = type("PlanningTopology", (), {"gsl_interfaces_info": None})()
        mock_build_topologies.return_value = (planning_topology, None)
        future_visibility = [[(20.0, 9)]]
        mock_compute_visibility.return_value = future_visibility
        mock_build_segment_plans.return_value = {("future", 0): ["fd00:10:0:1::a"]}
        mock_algorithm_traditional_segment_routing.return_value = {"fstate": {}, "bandwidth": {}}

        params = {
            "segment_count": 2,
            "segment_refresh_interval_steps": 1,
            "prediction_horizon_minutes": 5,
            "time_step_minutes": 5,
            "undirected_isls": [(1, 2)],
        }

        self.algorithm.compute_state(
            time_since_epoch_ns=0,
            constellation_data=self.constellation_data,
            ground_stations=self.ground_stations,
            topology_with_isls=self.topology,
            ground_station_satellites_in_range=self.visibility,
            list_gsl_interfaces_info=self.gsl_info,
            algorithm_params=params,
        )

        self.assertEqual(mock_build_segment_plans.call_count, 1)
        planning_args = mock_build_segment_plans.call_args.args
        self.assertIs(planning_args[0], planning_topology)
        self.assertEqual(planning_args[2], future_visibility)
        forwarded = mock_algorithm_traditional_segment_routing.call_args.kwargs
        self.assertEqual(forwarded["planning_ground_station_satellites_in_range"], future_visibility)
        self.assertEqual(
            forwarded["current_ground_station_satellites_in_range"],
            self.visibility,
        )
        self.assertEqual(forwarded["segment_plans"], {("future", 0): ["fd00:10:0:1::a"]})


class TestTraditionalSegmentRoutingSrv6(unittest.TestCase):
    def test_satellite_srv6_sid_round_trips(self):
        sid = satellite_srv6_sid(42)

        self.assertEqual(sid, "fd00:10:0:1::2b")
        self.assertEqual(satellite_id_from_srv6_sid(sid), 42)

    def test_segment_plans_use_ipv6_sids(self):
        satellites = [type("Sat", (), {"id": sat_id})() for sat_id in range(4)]
        ground_stations = [type("GroundStation", (), {"id": 100})()]
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

        plans = _build_segment_plans(
            topology,
            ground_stations,
            [[(1.0, 3)]],
            segment_count=2,
        )

        self.assertEqual(plans[(0, 100)], ["fd00:10:0:1::4"])
