# tests/dynamic_state/test_algorithm_free_one_only_over_isls.py

import unittest
from unittest.mock import MagicMock, patch

import ephem  # For mocking spec

# Function/Module to test
from src.network_state.routing_algorithms import algorithm_free_one_only_over_isls
from src.topology.satellite.satellite import Satellite
from src.topology.topology import (
    ConstellationData,
    GroundStation,
    LEOTopology,
)


# --- Mock Helper Classes (Can reuse from previous tests) ---
class MockLEOTopologyRefined(LEOTopology):
    """Minimal mock for LEOTopology needed by the algorithm test."""

    def __init__(self, constellation_data: ConstellationData, ground_stations: list[GroundStation]):
        # Store args, methods like get_satellite are needed if accessed
        self.constellation_data = constellation_data
        self.ground_stations = ground_stations
        self.number_of_ground_stations = len(ground_stations)
        # Add attributes accessed by the algorithm if any (graph, sat_neighbor_to_if not directly used?)
        self.graph = None  # Not directly used by algorithm if helper is mocked
        self.sat_neighbor_to_if = {}  # Not directly used by algorithm if helper is mocked

    def get_satellites(self) -> list[Satellite]:
        return self.constellation_data.satellites

    def get_satellite(self, sat_id: int) -> Satellite:
        # Simple lookup assuming list holds Satellite objects
        for sat in self.constellation_data.satellites:
            if sat.id == sat_id:
                return sat
        raise KeyError(f"Mock Satellite with ID {sat_id} not found")

    # Add get_ground_stations if needed


# --- Test Class ---
class TestAlgorithmFreeOneOnlyOverIsls(unittest.TestCase):

    def setUp(self):
        """Set up common mock objects and data for algorithm tests."""
        self.time_ns = 1_000_000_000
        self.num_sats = 2
        self.num_gs = 2
        self.total_nodes = self.num_sats + self.num_gs

        # Create mock satellites
        self.mock_body = MagicMock(spec=ephem.Body)
        self.sat0 = Satellite(
            id=0, ephem_obj_manual=self.mock_body, ephem_obj_direct=self.mock_body
        )
        self.sat1 = Satellite(
            id=1, ephem_obj_manual=self.mock_body, ephem_obj_direct=self.mock_body
        )
        self.satellites = [self.sat0, self.sat1]

        # Create mock ground stations
        self.gs2 = GroundStation(
            gid=2,
            name="GS2",
            latitude_degrees_str="0",
            longitude_degrees_str="0",
            elevation_m_float=0,
            cartesian_x=0,
            cartesian_y=0,
            cartesian_z=0,
        )
        self.gs3 = GroundStation(
            gid=3,
            name="GS3",
            latitude_degrees_str="0",
            longitude_degrees_str="0",
            elevation_m_float=0,
            cartesian_x=0,
            cartesian_y=0,
            cartesian_z=0,
        )
        self.ground_stations = [self.gs2, self.gs3]  # Order matters for visibility index

        # Create ConstellationData
        self.constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=self.num_sats,
            epoch="25001.0",
            max_gsl_length_m=5000000,
            max_isl_length_m=5000000,
            satellites=self.satellites,
        )

        # Create mock Topology object
        self.mock_topology = MockLEOTopologyRefined(self.constellation_data, self.ground_stations)

        # Example Visibility Data ([list per GS index])
        self.visibility = [
            [(1000, 0)],  # GS 2 (idx 0) sees Sat 0
            [(1500, 1)],  # GS 3 (idx 1) sees Sat 1
        ]

        # Example GSL Interface Info (Bandwidth)
        # List order assumed to match nodes 0, 1, 2, 3
        self.gsl_info = [
            {"id": 0, "aggregate_max_bandwidth": 100.0},
            {"id": 1, "aggregate_max_bandwidth": 110.0},
            {"id": 2, "aggregate_max_bandwidth": 50.0},
            {"id": 3, "aggregate_max_bandwidth": 60.0},
        ]

        # Example Previous Output
        self.prev_output_data = {"fstate": {"prev": "state"}, "bandwidth": {0: 99.0}}

        self.enable_logs = False

        # --- Patch the helper function ---
        # Patch where it's looked up: in the algorithm module itself
        patcher = patch(
            "src.network_state.routing_algorithms.algorithm_free_one_only_over_isls.calculate_fstate_shortest_path_object_no_gs_relay"
        )
        self.addCleanup(patcher.stop)
        self.mock_fstate_calculator = patcher.start()

        # Configure a default return value for the mocked helper
        self.expected_fstate_result = {(0, 3): (1, 0, 0), (2, 1): (0, 0, 1)}  # Example dummy fstate
        self.mock_fstate_calculator.return_value = self.expected_fstate_result

        # Patch logger (optional, useful for checking log calls)
        patcher_log = patch(
            "src.network_state.routing_algorithms.algorithm_free_one_only_over_isls.log",
            MagicMock(),
        )
        self.addCleanup(patcher_log.stop)
        self.mock_log = patcher_log.start()

    def test_state_calculation_basic(self):
        """Test basic calculation of bandwidth and call to fstate helper."""

        result = algorithm_free_one_only_over_isls.algorithm_free_one_only_over_isls(
            time_since_epoch_ns=self.time_ns,
            constellation_data=self.constellation_data,
            ground_stations=self.ground_stations,
            topology_with_isls=self.mock_topology,
            ground_station_satellites_in_range=self.visibility,
            list_gsl_interfaces_info=self.gsl_info,
            prev_output=None,  # Test with no previous output first
        )

        # 1. Assert Bandwidth Calculation
        expected_bandwidth = {
            0: 100.0,  # Sat 0
            1: 110.0,  # Sat 1
            2: 50.0,  # GS 2
            3: 60.0,  # GS 3
        }
        self.assertDictEqual(
            result.get("bandwidth"), expected_bandwidth, "Bandwidth state calculation mismatch"
        )

        # 2. Assert F-State Helper Call
        self.mock_fstate_calculator.assert_called_once_with(
            self.mock_topology,
            self.ground_stations,
            self.visibility,
            # Note: prev_fstate_obj is NOT currently passed by the algorithm code
        )

        # 3. Assert F-State Result
        self.assertEqual(
            result.get("fstate"), self.expected_fstate_result, "F-state result mismatch"
        )
        self.assertEqual(
            result.get("fstate"),
            self.mock_fstate_calculator.return_value,
            "Returned fstate differs from helper return",
        )

    def test_previous_output_handling(self):
        """Test that previous output is processed (though not currently passed to helper)."""

        result = algorithm_free_one_only_over_isls.algorithm_free_one_only_over_isls(
            time_since_epoch_ns=self.time_ns,
            constellation_data=self.constellation_data,
            ground_stations=self.ground_stations,
            topology_with_isls=self.mock_topology,
            ground_station_satellites_in_range=self.visibility,
            list_gsl_interfaces_info=self.gsl_info,
            prev_output=self.prev_output_data,  # Pass previous output
        )

        # Assert helper was still called correctly (prev_output['fstate'] is extracted but not used in call)
        self.mock_fstate_calculator.assert_called_once_with(
            self.mock_topology,
            self.ground_stations,
            self.visibility,
        )

        # Assert results are still based on current calculation
        self.assertEqual(result.get("fstate"), self.expected_fstate_result)
        expected_bandwidth = {0: 100.0, 1: 110.0, 2: 50.0, 3: 60.0}
        self.assertDictEqual(result.get("bandwidth"), expected_bandwidth)

    def test_gsl_interface_info_length_mismatch(self):
        """Test handling when list_gsl_interfaces_info is shorter than expected."""
        # Provide a shorter list (missing info for node 3)
        short_gsl_info = self.gsl_info[:-1]  # Only nodes 0, 1, 2

        result = algorithm_free_one_only_over_isls.algorithm_free_one_only_over_isls(
            time_since_epoch_ns=self.time_ns,
            constellation_data=self.constellation_data,
            ground_stations=self.ground_stations,
            topology_with_isls=self.mock_topology,
            ground_station_satellites_in_range=self.visibility,
            list_gsl_interfaces_info=short_gsl_info,  # Use short list
            prev_output=None,
        )

        # Assert bandwidth: Node 3 should have default BW=0 due to fallback
        expected_bandwidth = {0: 100.0, 1: 110.0, 2: 50.0, 3: 0.0}  # Node 3 gets default BW
        self.assertDictEqual(
            result.get("bandwidth"), expected_bandwidth, "Bandwidth incorrect with short info list"
        )

        # Assert warning was logged
        self.mock_log.warning.assert_called()
        self.mock_log.error.assert_called_with(
            "Index 3 out of bounds for list_gsl_interfaces_info, setting BW=0 for node 3"
        )

        # Assert fstate calculation still happened and result is included
        self.mock_fstate_calculator.assert_called_once()
        self.assertEqual(result.get("fstate"), self.expected_fstate_result)

    def test_fstate_calculator_exception(self):
        """Test behavior when the fstate helper raises an exception."""
        # Configure mock helper to raise an error
        test_exception = ValueError("F-state calc failed")
        self.mock_fstate_calculator.side_effect = test_exception

        result = algorithm_free_one_only_over_isls.algorithm_free_one_only_over_isls(
            time_since_epoch_ns=self.time_ns,
            constellation_data=self.constellation_data,
            ground_stations=self.ground_stations,
            topology_with_isls=self.mock_topology,
            ground_station_satellites_in_range=self.visibility,
            list_gsl_interfaces_info=self.gsl_info,
            prev_output=None,
        )

        # Assert fstate is empty dictionary on error
        self.assertDictEqual(
            result.get("fstate"), {}, "F-state should be empty on helper exception"
        )

        # Assert bandwidth was still calculated
        expected_bandwidth = {0: 100.0, 1: 110.0, 2: 50.0, 3: 60.0}
        self.assertDictEqual(result.get("bandwidth"), expected_bandwidth)

        # Assert exception was logged
        self.mock_log.exception.assert_called_once_with(
            f"Error during forwarding state calculation: {test_exception}"
        )
