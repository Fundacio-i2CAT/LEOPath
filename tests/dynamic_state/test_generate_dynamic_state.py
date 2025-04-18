import math
import unittest
from unittest.mock import ANY, MagicMock, call, patch

import ephem  # Import ephem for specing mocks
import networkx as nx
import numpy as np
from astropy import units as astro_units

# Use a fixed time for reproducibility instead of relying on current time
from astropy.time import Time

from src import logger

log = logger.get_logger(__name__)


# --- Import the module(s) under test ---
from src.dynamic_state import generate_dynamic_state

# --- Import actual classes from topology ---
from src.dynamic_state.topology import (
    ConstellationData,
    GroundStation,
    LEOTopology,
    Satellite,
    SatelliteEphemeris,
)

# --- Import logger if tests need to configure it ---
# from src import logger # Not strictly needed unless tests modify logging

# --- Configure logging level for test output (Optional) ---
# You can configure the root logger or the specific logger used by the module
# to control how much output you see during tests.
# Example: Show DEBUG messages from the tested module during tests
# logging.getLogger('src.dynamic_state.generate_dynamic_state').setLevel(logging.DEBUG)
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
        self.current_time_absolute = (
            self.mock_astropy_epoch + self.time_since_epoch_ns * astro_units.ns
        )
        self.list_gsl_interfaces_info = [
            {"id": sat.id, "number_of_interfaces": 4} for sat in self.satellites
        ] + [{"id": gs.id, "number_of_interfaces": 2} for gs in self.ground_stations]
        # --- End of common data setup ---

        # --- Patching ---
        # Patch distance tools module - provides self.mock_distance_tools
        patcher_dist_tools = patch(
            "src.dynamic_state.generate_dynamic_state.distance_tools", MagicMock()
        )
        self.addCleanup(patcher_dist_tools.stop)
        self.mock_distance_tools = patcher_dist_tools.start()

        # Patch LEOTopology class
        patcher_topology = patch(
            "src.dynamic_state.generate_dynamic_state.LEOTopology",
            side_effect=MockLEOTopologyRefined,
        )
        self.addCleanup(patcher_topology.stop)
        self.MockLEOTopologyClass_patched = patcher_topology.start()

        # Patch the specific algorithm function
        patcher_algorithm = patch(
            "src.dynamic_state.generate_dynamic_state.algorithm_free_one_only_over_isls",
            MagicMock(),
        )
        self.addCleanup(patcher_algorithm.stop)
        self.mock_algorithm_func = patcher_algorithm.start()

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

    # --- Test Methods ---
    # (All test methods remain exactly the same as the previous version)

    def test_compute_isls_success(self):
        """Test _compute_isls adds edges for valid distances using actual Satellite objects."""
        self.mock_distance_tools.distance_m_between_satellites.return_value = self.max_isl_m - 1000
        topology = MockLEOTopologyRefined(self.constellation_data, [])
        generate_dynamic_state._compute_isls(
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
            generate_dynamic_state._compute_isls(
                topology, self.undirected_isls, self.current_time_absolute
            )
        self.assertIn("exceeded the maximum ISL length", str(cm.exception))

    def test_build_topologies(self):
        """Test _build_topologies creates graphs via the Patched LEOTopology."""
        topo_isl, topo_gsl = generate_dynamic_state._build_topologies(
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
        """Test _compute_ground_station_satellites_in_range adds edges correctly using actual objects."""
        topology = MockLEOTopologyRefined(self.constellation_data, self.ground_stations)

        # Define mock distances based on actual GS object and Satellite.position object
        def side_effect_func(gs_obj, sat_ephemeris_obj, epoch_str, time_str):
            # Identify satellite based on the ephemeris object passed
            sat_id = -1
            for sat in self.satellites:
                if sat.position == sat_ephemeris_obj:
                    sat_id = sat.id
                    break

            if gs_obj.id == self.gs1_id and sat_id == 0:
                return self.max_gsl_m - 1000  # In range
            if gs_obj.id == self.gs1_id and sat_id == 1:
                return self.max_gsl_m + 1000  # Out of range
            if gs_obj.id == self.gs2_id and sat_id == 2:
                return self.max_gsl_m - 500  # In range
            return self.max_gsl_m + 2000  # Default out

        self.mock_distance_tools.distance_m_ground_station_to_satellite.side_effect = (
            side_effect_func
        )
        generate_dynamic_state._compute_ground_station_satellites_in_range(
            topology, self.current_time_absolute
        )
        self.assertEqual(
            self.mock_distance_tools.distance_m_ground_station_to_satellite.call_count,
            len(self.ground_stations) * self.num_sats,
        )
        self.mock_distance_tools.distance_m_ground_station_to_satellite.assert_any_call(
            self.gs1,  # Actual GroundStation object
            self.satellites[0].position,  # SatelliteEphemeris object from Satellite 0
            self.string_epoch,
            str(self.current_time_absolute),  # Absolute time string
        )
        self.mock_distance_tools.distance_m_ground_station_to_satellite.assert_any_call(
            self.gs2,  # Actual GroundStation object
            self.satellites[2].position,  # SatelliteEphemeris object from Satellite 2
            # CORRECTED: Use the string epoch stored in ConstellationData
            self.string_epoch,
            str(self.current_time_absolute),
        )

        # Check graph edges using IDs
        self.assertTrue(topology.graph.has_edge(self.gs1_id, 0))
        self.assertFalse(topology.graph.has_edge(self.gs1_id, 1))
        self.assertTrue(topology.graph.has_edge(self.gs2_id, 2))
        self.assertEqual(
            topology.graph.get_edge_data(self.gs1_id, 0)["weight"],
            self.max_gsl_m - 1000,
        )

    def test_generate_dynamic_state_at_unknown_algorithm(self):
        """Test ValueError is raised for an unknown algorithm name."""
        with self.assertRaises(ValueError) as cm:
            # This call might produce log output before raising the error
            generate_dynamic_state.generate_dynamic_state_at(
                "/fake/dir",
                self.mock_astropy_epoch,
                self.time_since_epoch_ns,
                self.constellation_data,
                self.ground_stations,
                self.undirected_isls,
                self.list_gsl_interfaces_info,
                "bad_algorithm",
                None,
            )
        self.assertIn("Unknown dynamic state algorithm: bad_algorithm", str(cm.exception))

    @patch("src.dynamic_state.generate_dynamic_state.generate_dynamic_state_at")
    def test_generate_dynamic_state_loop(self, mock_generate_at):
        """Test the main loop calls generate_dynamic_state_at correctly."""
        output_dir = "/fake/loop/dir"

        # Define time parameters as INTEGERS
        simulation_end_time_ns = 3 * 1_000_000_000  # Use underscores or just 3000000000
        time_step_ns = 1 * 1_000_000_000  # Use underscores or just 1000000000
        offset_ns = 1 * 1_000_000_000  # Use underscores or just 1000000000
        # Alternatively:
        # simulation_end_time_ns = int(3e9)
        # time_step_ns = int(1e9)
        # offset_ns = int(1e9)

        algo_name = "algorithm_free_one_only_over_isls"
        enable_verbose_logs = False
        mock_generate_at.side_effect = ["output_t1", "output_t2"]

        # This call might produce log output (e.g., the progress print)
        generate_dynamic_state.generate_dynamic_state(
            output_dir,
            self.mock_astropy_epoch,
            simulation_end_time_ns,  # Pass int
            time_step_ns,  # Pass int
            offset_ns,  # Pass int
            self.constellation_data,
            self.ground_stations,
            self.undirected_isls,
            self.list_gsl_interfaces_info,
            self.max_gsl_m,
            self.max_isl_m,
            algo_name,
            enable_verbose_logs,
        )

        # Assertions use the integer times now
        calls = [
            call(
                output_dir,
                self.mock_astropy_epoch,
                offset_ns,  # Expect int
                self.constellation_data,
                self.ground_stations,
                self.undirected_isls,
                self.list_gsl_interfaces_info,
                algo_name,
                None,
            ),
            call(
                output_dir,
                self.mock_astropy_epoch,
                offset_ns + time_step_ns,  # Expect int (result of int + int)
                self.constellation_data,
                self.ground_stations,
                self.undirected_isls,
                self.list_gsl_interfaces_info,
                algo_name,
                "output_t1",
            ),
        ]
        mock_generate_at.assert_has_calls(calls)
        self.assertEqual(mock_generate_at.call_count, 2)

    def test_generate_dynamic_state_invalid_offset(self):
        """Test ValueError if offset is not a multiple of time_step_ns."""
        with self.assertRaises(ValueError) as cm:
            generate_dynamic_state.generate_dynamic_state(
                "/fake",
                self.mock_astropy_epoch,
                1000,
                100,
                50,
                self.constellation_data,
                self.ground_stations,
                self.undirected_isls,
                self.list_gsl_interfaces_info,
                self.max_gsl_m,
                self.max_isl_m,
                "algorithm_free_one_only_over_isls",
                False,
            )
        self.assertIn("Offset must be a multiple of time_step_ns", str(cm.exception))
