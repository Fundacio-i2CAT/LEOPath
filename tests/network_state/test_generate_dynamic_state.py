import unittest
from unittest.mock import MagicMock, patch, call

import ephem
import networkx as nx
from astropy import units as astro_units

# Use a fixed time for reproducibility instead of relying on current time
from astropy.time import Time

from src import logger
from src.network_state import generate_network_state
from src.network_state.helpers import (
    _compute_isls,
)
from src.topology.satellite.satellite import Satellite
from src.topology.topology import (
    ConstellationData,
    GroundStation,
    LEOTopology,
)

log = logger.get_logger(__name__)


# --- Import logger if tests need to configure it ---
# from src import logger # Not strictly needed unless tests modify logging

# --- Configure logging level for test output (Optional) ---
# You can configure the root logger or the specific logger used by the module
# to control how much output you see during tests.
# Example: Show DEBUG messages from the tested module during tests
# logging.getLogger('src.network_state.generate_network_state').setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG) # Or configure globally


# --- Mock Helper Classes (Refined) ---
# (MockLEOTopologyRefined remains the same as before)
class MockLEOTopologyRefined(LEOTopology):
    """Mock LEOTopology reflecting actual class structure more closely."""

    def __init__(
        self,
        constellation_data: ConstellationData,
        ground_stations: list[GroundStation],
    ):
        self.constellation_data = constellation_data
        self.ground_stations = ground_stations
        self.number_of_ground_stations = len(ground_stations)
        self.graph = nx.Graph()  # Use a real graph
        self.sat_neighbor_to_if = {}
        self.number_of_isls = 0
        self.gsl_interfaces_info = [
            {"id": sat.id, "number_of_interfaces": 4} for sat in constellation_data.satellites
        ] + [{"id": gs.id, "number_of_interfaces": 2} for gs in ground_stations]
        for sat in constellation_data.satellites:
            self.graph.add_node(sat.id)
        for gs in ground_stations:
            self.graph.add_node(gs.id)

    def get_satellites(self) -> list[Satellite]:
        return self.constellation_data.satellites

    def get_satellite(self, sat_id: int) -> Satellite:
        for sat in self.constellation_data.satellites:
            if sat.id == sat_id:
                return sat
        raise KeyError(f"Mock Satellite with ID {sat_id} not found")

    def get_ground_stations(self) -> list[GroundStation]:
        return self.ground_stations

    def get_ground_station(self, gid: int) -> GroundStation:
        for gs in self.ground_stations:
            if gs.id == gid:
                return gs
        raise KeyError(f"Mock Ground Station with ID {gid} not found")


class TestDynamicStateGeneratorUpdated(unittest.TestCase):

    def setUp(self):
        """Set up common test data and mocks using actual class structures."""
        # --- (Time, Constellation, Satellite, GroundStation setup remains the same) ---
        self.mock_astropy_epoch = Time("2023-01-01T00:00:00", scale="tdb")
        self.string_epoch = "23001.00000000"
        self.num_orbits = 2
        self.sats_per_orbit = 2
        self.num_sats = self.num_orbits * self.sats_per_orbit
        self.max_isl_m = 2000 * 1000
        self.max_gsl_m = 2500 * 1000
        self.mock_ephem_bodies = [
            MagicMock(spec=ephem.Body, name=f"MockSat{i}") for i in range(self.num_sats)
        ]
        self.satellites = [
            Satellite(
                id=i,
                ephem_obj_manual=self.mock_ephem_bodies[i],
                ephem_obj_direct=self.mock_ephem_bodies[i],
            )
            for i in range(self.num_sats)
        ]
        self.constellation_data = ConstellationData(
            orbits=self.num_orbits,
            sats_per_orbit=self.sats_per_orbit,
            epoch=self.string_epoch,
            max_gsl_length_m=self.max_gsl_m,
            max_isl_length_m=self.max_isl_m,
            satellites=self.satellites,
        )
        self.gs1_id = self.num_sats
        self.gs2_id = self.num_sats + 1
        self.gs1 = GroundStation(
            gid=self.gs1_id,
            name="GS1",
            latitude_degrees_str="0.0",
            longitude_degrees_str="0.0",
            elevation_m_float=0.0,
            cartesian_x=6378137.0,
            cartesian_y=0.0,
            cartesian_z=0.0,
        )
        self.gs2 = GroundStation(
            gid=self.gs2_id,
            name="GS2",
            latitude_degrees_str="10.0",
            longitude_degrees_str="10.0",
            elevation_m_float=100.0,
            cartesian_x=6278011.9,
            cartesian_y=1105942.1,
            cartesian_z=1099176.3,
        )
        self.ground_stations = [self.gs1, self.gs2]
        self.undirected_isls = [(0, 1), (1, 2)]
        self.time_since_epoch_ns = 1 * 1e9
        from astropy import units as u

        self.current_time_absolute = self.mock_astropy_epoch + self.time_since_epoch_ns * u.ns
        self.list_gsl_interfaces_info = [
            {"id": sat.id, "number_of_interfaces": 4} for sat in self.satellites
        ] + [{"id": gs.id, "number_of_interfaces": 2} for gs in self.ground_stations]
        # --- End of common data setup ---

        # --- Patching ---
        # Patch distance tools module - provides self.mock_distance_tools
        patcher_dist_tools = patch("src.network_state.helpers.distance_tools", MagicMock())
        self.addCleanup(patcher_dist_tools.stop)
        self.mock_distance_tools = patcher_dist_tools.start()

        # Patch LEOTopology class
        patcher_topology = patch(
            "src.network_state.helpers.LEOTopology",
            side_effect=MockLEOTopologyRefined,
        )
        self.addCleanup(patcher_topology.stop)
        self.MockLEOTopologyClass_patched = patcher_topology.start()

        # Patch the routing algorithm factory to return a MagicMock algorithm instance
        patcher_algorithm_factory = patch(
            "src.network_state.routing_algorithms.routing_algorithm_factory.get_routing_algorithm",
            return_value=MagicMock(),
        )
        self.addCleanup(patcher_algorithm_factory.stop)
        self.mock_algorithm_factory = patcher_algorithm_factory.start()

        # --- Configure DEFAULT return values for mocked distance functions ---
        # Configure the methods directly on the mocked module object

        # Configure default for ISL distance
        # Ensure the attribute exists on the mock if spec/autospec wasn't perfect
        # self.mock_distance_tools.distance_m_between_satellites = MagicMock()
        self.mock_distance_tools.distance_m_between_satellites.return_value = self.max_isl_m / 2

        # *** ADD THIS: Configure default for GSL distance ***
        # Ensure the attribute exists on the mock if spec/autospec wasn't perfect
        # self.mock_distance_tools.distance_m_ground_station_to_satellite = MagicMock()
        self.mock_distance_tools.distance_m_ground_station_to_satellite.return_value = (
            self.max_gsl_m / 2
        )

    def test_compute_isls_success(self):
        """Test _compute_isls adds edges for valid distances using actual Satellite objects."""
        self.mock_distance_tools.distance_m_between_satellites.return_value = self.max_isl_m - 1000
        topology = MockLEOTopologyRefined(self.constellation_data, [])
        _compute_isls(
            topology,
            self.undirected_isls,
            self.current_time_absolute,
        )

        self.assertEqual(
            self.mock_distance_tools.distance_m_between_satellites.call_count,
            len(self.undirected_isls),
        )
        # Check calls with actual Satellite objects and the absolute time string
        self.mock_distance_tools.distance_m_between_satellites.assert_any_call(
            self.satellites[0],
            self.satellites[1],
            self.string_epoch,  # Pass the original string epoch from ConstellationData
            str(self.current_time_absolute),
        )
        self.mock_distance_tools.distance_m_between_satellites.assert_any_call(
            self.satellites[1],
            self.satellites[2],
            self.string_epoch,
            str(self.current_time_absolute),
        )
        self.assertTrue(topology.graph.has_edge(0, 1))
        self.assertTrue(topology.graph.has_edge(1, 2))
        self.assertEqual(topology.graph.number_of_edges(), 2)
        self.assertIn((0, 1), topology.sat_neighbor_to_if)
        self.assertEqual(topology.get_satellite(0).number_isls, 1)
        self.assertEqual(topology.get_satellite(1).number_isls, 2)
        self.assertEqual(topology.get_satellite(2).number_isls, 1)
        self.assertEqual(topology.number_of_isls, 2)

    def test_compute_isls_fail_too_long(self):
        """Test _compute_isls raises ValueError for distance > max_isl_length_m."""
        self.mock_distance_tools.distance_m_between_satellites.return_value = self.max_isl_m + 1000
        topology = MockLEOTopologyRefined(self.constellation_data, [])

        with self.assertRaises(ValueError) as cm:
            generate_network_state._compute_isls(
                topology, self.undirected_isls, self.current_time_absolute
            )
        self.assertIn("exceeded the maximum ISL length", str(cm.exception))

    def test_build_topologies(self):
        """Test _build_topologies creates graphs via the Patched LEOTopology."""
        topo_isl, topo_gsl = generate_network_state._build_topologies(
            self.constellation_data, self.ground_stations
        )

        self.assertEqual(self.MockLEOTopologyClass_patched.call_count, 2)
        self.MockLEOTopologyClass_patched.assert_has_calls(
            [
                call(self.constellation_data, self.ground_stations),
                call(self.constellation_data, self.ground_stations),
            ]
        )
        self.assertIsInstance(topo_isl, MockLEOTopologyRefined)
        expected_nodes = self.num_sats + len(self.ground_stations)
        self.assertEqual(len(topo_isl.graph.nodes), expected_nodes)
        self.assertIsInstance(topo_gsl, MockLEOTopologyRefined)
        self.assertEqual(len(topo_gsl.graph.nodes), expected_nodes)

    def test_compute_ground_station_satellites_in_range(self):
        """Test _compute_ground_station_satellites_in_range adds edges correctly using MOCKED distances."""  # Clarified docstring
        topology = MockLEOTopologyRefined(self.constellation_data, self.ground_stations)

        # Define mock distances based on actual GS object and Satellite object ID
        in_range_dist = self.max_gsl_m - 1000
        out_of_range_dist = self.max_gsl_m + 1000

        # Side effect function needs to handle receiving the full Satellite object now
        def side_effect_func(
            gs_obj, sat_obj, epoch_str, time_str
        ):  # Changed second arg name for clarity
            # Identify satellite based on the passed Satellite object's ID
            sat_id = sat_obj.id

            # Define which pairs should be in range for this test
            if gs_obj.id == self.gs1_id and sat_id == 0:
                return in_range_dist
            if gs_obj.id == self.gs1_id and sat_id == 1:
                return out_of_range_dist
            if gs_obj.id == self.gs2_id and sat_id == 2:
                return in_range_dist  # Assumes self.num_sats >= 3
            return out_of_range_dist

        self.mock_distance_tools.distance_m_ground_station_to_satellite.side_effect = (
            side_effect_func
        )

        # Call the function under test (this part is likely correct now)
        generate_network_state._compute_ground_station_satellites_in_range(
            topology, self.current_time_absolute
        )

        # --- Assertions ---
        # Check call count
        self.assertEqual(
            self.mock_distance_tools.distance_m_ground_station_to_satellite.call_count,
            len(self.ground_stations) * self.num_sats,
        )

        # Create the expected formatted time string
        expected_time_str = self.current_time_absolute.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]

        self.mock_distance_tools.distance_m_ground_station_to_satellite.assert_any_call(
            self.gs1,  # GroundStation object
            self.satellites[0],
            self.string_epoch,  # Epoch string from ConstellationData
            expected_time_str,
        )
        # Update other assert_any_call if you have them
        # Example assuming num_sats >= 3:
        if self.num_sats >= 3:
            self.mock_distance_tools.distance_m_ground_station_to_satellite.assert_any_call(
                self.gs2,  # GroundStation object
                self.satellites[2],  # Satellite object
                self.string_epoch,  # Epoch string from ConstellationData
                expected_time_str,  # Formatted time string
            )

        # Check graph edges based on side_effect logic (should be okay if side_effect is correct)
        if self.num_sats >= 3:  # Check depends on test setup
            expected_edges_in_test = [(self.gs1_id, 0), (self.gs2_id, 2)]
            not_expected_edges_in_test = []
            for gs in self.ground_stations:
                for sat in self.satellites:
                    if (gs.id, sat.id) not in expected_edges_in_test:
                        not_expected_edges_in_test.append((gs.id, sat.id))

            for u, v in expected_edges_in_test:
                self.assertTrue(
                    topology.graph.has_edge(u, v), f"Expected edge ({u}, {v}) not found"
                )
                edge_data = topology.graph.get_edge_data(u, v)
                weight = edge_data.get("weight") if edge_data else 0
                self.assertIsNotNone(weight, f"Edge ({u},{v}) missing weight")
                self.assertAlmostEqual(
                    float(weight),
                    float(in_range_dist),
                    delta=0.01,
                    msg=f"Edge ({u},{v}) weight incorrect",
                )

            for u, v in not_expected_edges_in_test:
                self.assertFalse(topology.graph.has_edge(u, v), f"Unexpected edge ({u}, {v}) found")

            self.assertEqual(topology.graph.number_of_edges(), len(expected_edges_in_test))

    def test_generate_dynamic_state_at_unknown_algorithm(self):
        """Test ValueError is raised for an unknown algorithm name."""
        with self.assertRaises(ValueError) as cm:
            # This call might produce log output before raising the error
            generate_network_state._generate_state_for_step(
                self.mock_astropy_epoch,
                self.time_since_epoch_ns,
                self.constellation_data,
                self.ground_stations,
                self.undirected_isls,
                self.list_gsl_interfaces_info,
                "bad_algorithm",
                None,
                None,
            )
        self.assertIn("Unknown routing algorithm", str(cm.exception))

    @patch("src.network_state.generate_network_state._generate_state_for_step")
    def test_generate_dynamic_state_loop(self, mock_generate_at):
        """Test the main loop calls _generate_state_for_step correctly."""
        simulation_end_time_ns = 3 * 1_000_000_000
        time_step_ns = 1 * 1_000_000_000
        offset_ns = 1 * 1_000_000_000
        algo_name = "algorithm_free_one_only_over_isls"
        mock_state_t1 = {"fstate": "dummy_fstate_t1", "bandwidth": "dummy_bw_t1"}
        mock_state_t2 = {"fstate": "dummy_fstate_t2", "bandwidth": "dummy_bw_t2"}
        # Dummy topology objects remain the same
        mock_topo_t1 = MagicMock(name="TopoT1")
        mock_topo_t2 = MagicMock(name="TopoT2")
        mock_generate_at.side_effect = [
            (mock_state_t1, mock_topo_t1),
            (mock_state_t2, mock_topo_t2),
        ]

        generate_network_state.generate_dynamic_state(
            self.mock_astropy_epoch,
            simulation_end_time_ns,
            time_step_ns,
            offset_ns,
            self.constellation_data,
            self.ground_stations,
            self.undirected_isls,
            self.list_gsl_interfaces_info,
            algo_name,
        )

        calls = [
            # Call for t=1e9 (offset)
            call(
                epoch=self.mock_astropy_epoch,
                time_since_epoch_ns=offset_ns,  # t=1e9
                constellation_data=self.constellation_data,
                ground_stations=self.ground_stations,
                undirected_isls=self.undirected_isls,
                list_gsl_interfaces_info=self.list_gsl_interfaces_info,
                dynamic_state_algorithm=algo_name,
                prev_output=None,
                prev_topology=None,
            ),
            # Call for t=2e9
            call(
                epoch=self.mock_astropy_epoch,
                time_since_epoch_ns=offset_ns + time_step_ns,  # t=2e9
                constellation_data=self.constellation_data,
                ground_stations=self.ground_stations,
                undirected_isls=self.undirected_isls,
                list_gsl_interfaces_info=self.list_gsl_interfaces_info,
                dynamic_state_algorithm=algo_name,
                prev_output=mock_state_t1,  # Previous state was the DICT now
                prev_topology=mock_topo_t1,
            ),
        ]
        mock_generate_at.assert_has_calls(calls, any_order=False)
        self.assertEqual(mock_generate_at.call_count, 2)

    def test_generate_dynamic_state_invalid_offset(self):
        """Test ValueError if offset is not a multiple of time_step_ns."""
        with self.assertRaises(ValueError) as cm:
            generate_network_state.generate_dynamic_state(
                self.mock_astropy_epoch,
                1000,  # end time (int)
                100,  # step (int)
                50,  # offset (int, invalid)
                self.constellation_data,
                self.ground_stations,
                self.undirected_isls,
                self.list_gsl_interfaces_info,
                # self.max_gsl_m,         # REMOVE THIS ARGUMENT
                # self.max_isl_m,         # REMOVE THIS ARGUMENT
                "algorithm_free_one_only_over_isls",  # algo name
                # False,                  # REMOVE THIS ARGUMENT (enable_verbose_logs)
            )
        self.assertIn("Offset must be a multiple of time_step_ns", str(cm.exception))

    def test_generate_dynamic_state_skips_calc_on_equal_topo(
        self,
    ):
        """
        Test that generate_dynamic_state loop skips calling the algorithm
        when _topologies_are_equal returns True (using context managers).
        """
        # Simulate 3 time steps: t=0, t=1, t=2
        simulation_end_time_ns = 3 * 1_000_000_000
        time_step_ns = 1 * 1_000_000_000
        offset_ns = 0
        algo_name = "shortest_path_link_state"  # Use the factory-based name
        state_t0 = {"fstate": {"id": "state_t0"}, "bandwidth": {"node": 1.0}}
        state_t2 = {"fstate": {"id": "state_t2"}, "bandwidth": {"node": 1.0}}
        mock_topo_t0 = MagicMock(name="TopoT0")
        mock_topo_t1 = MagicMock(name="TopoT1")
        mock_topo_t2 = MagicMock(name="TopoT2")

        with patch(
            "src.network_state.utils.graph._topologies_are_equal"
        ) as mock_topologies_equal, patch(
            "src.network_state.routing_algorithms.routing_algorithm_factory.get_routing_algorithm"
        ) as mock_factory, patch(
            "src.network_state.generate_network_state._build_topologies"
        ) as mock_build, patch(
            "src.network_state.generate_network_state._compute_isls"
        ) as mock_isls, patch(
            "src.network_state.generate_network_state._compute_ground_station_satellites_in_range"
        ) as mock_gsl:
            # Setup topology comparison
            mock_topologies_equal.side_effect = [
                False,  # t=0: not equal, compute
                True,  # t=1: equal, reuse
                False,  # t=2: not equal, compute
            ]
            # Setup factory and algorithm instance
            mock_algorithm_instance = MagicMock()
            mock_algorithm_instance.compute_state.side_effect = [
                state_t0,  # t=0
                state_t2,  # t=2
            ]
            mock_factory.return_value = mock_algorithm_instance
            # Setup topology builds
            mock_build.side_effect = [
                (mock_topo_t0, MagicMock()),  # t=0
                (mock_topo_t1, MagicMock()),  # t=1
                (mock_topo_t2, MagicMock()),  # t=2
            ]
            mock_isls.return_value = None
            mock_gsl.return_value = []

            final_states = generate_network_state.generate_dynamic_state(
                self.mock_astropy_epoch,
                simulation_end_time_ns,
                time_step_ns,
                offset_ns,
                self.constellation_data,
                self.ground_stations,
                self.undirected_isls,
                self.list_gsl_interfaces_info,
                algo_name,
            )

        # Assert topology comparison calls
        self.assertEqual(mock_topologies_equal.call_count, 3)
        mock_topologies_equal.assert_has_calls(
            [
                call(None, mock_topo_t0),
                call(mock_topo_t0, mock_topo_t1),
                call(mock_topo_t1, mock_topo_t2),
            ],
            any_order=False,
        )
        # Assert compute_state calls
        self.assertEqual(mock_algorithm_instance.compute_state.call_count, 2)
        first_call_args = mock_algorithm_instance.compute_state.call_args_list[0]
        second_call_args = mock_algorithm_instance.compute_state.call_args_list[1]
        self.assertEqual(first_call_args.kwargs["time_since_epoch_ns"], 0)
        self.assertEqual(second_call_args.kwargs["time_since_epoch_ns"], 2_000_000_000)
        self.assertEqual(len(final_states), 3)
        self.assertEqual(final_states[0]["fstate"], state_t0["fstate"])
        self.assertEqual(final_states[0]["time_since_epoch_ns"], 0)
        self.assertEqual(final_states[1]["fstate"], state_t0["fstate"])  # Reused
        self.assertEqual(final_states[1]["time_since_epoch_ns"], 1_000_000_000)
        self.assertEqual(final_states[2]["fstate"], state_t2["fstate"])  # Recalculated
        self.assertEqual(final_states[2]["time_since_epoch_ns"], 2_000_000_000)
