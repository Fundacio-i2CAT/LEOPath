# Filename: leo-routing-sim/tests/post_analysis/test_graph_tools.py

import unittest
from unittest.mock import patch, MagicMock
import os
import sys

# Ensure modules from 'src' can be imported.
# This assumes tests are run from the project root (e.g., 'leo-routing-sim/')
# using a command like 'python -m unittest discover -s tests'
# or that PYTHONPATH is configured to include the 'src' directory.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from src.post_analysis.graph_tools import get_path, compute_path_length_without_graph
from skyfield.api import EarthSatellite, load  # For creating dummy satellite objects
from astropy import units as u  # For time units if needed directly in test setup


class TestGetPath(unittest.TestCase):
    """Tests for the get_path function."""

    def test_simple_direct_path(self):
        # Test case: Destination is the next hop from source
        # forward_state: {(current, ultimate_dest): next_hop}
        forward_state = {(0, 1): 1}
        path = get_path(0, 1, forward_state)
        self.assertEqual(path, [0, 1])

    def test_simple_multi_hop_path(self):
        # Path: 0 -> 1 -> 2
        forward_state = {
            (0, 2): 1,  # From 0, to reach 2, next hop is 1
            (1, 2): 2,  # From 1, to reach 2, next hop is 2 (destination)
        }
        path = get_path(0, 2, forward_state)
        self.assertEqual(path, [0, 1, 2])

    def test_no_path_missing_initial_key(self):
        forward_state = {(0, 2): 1}  # No entry for (0,1)
        path = get_path(0, 1, forward_state)
        self.assertIsNone(path)

    def test_no_path_broken_midway(self):
        # Path: 0 -> 1, but no rule from 1 to 2
        forward_state = {(0, 2): 1}  # From 0 to 2, next is 1. No rule from 1 to 2.
        path = get_path(0, 2, forward_state)
        self.assertIsNone(path)

    def test_path_to_self(self):
        forward_state = {}  # Not relevant for src == dst
        path = get_path(0, 0, forward_state)
        self.assertEqual(path, [0])

    def test_loop_detection(self):
        # Path: 0 -> 1 -> 0 (loop back to 0 while trying to reach 2)
        forward_state = {(0, 2): 1, (1, 2): 0}  # This creates a loop 0 -> 1 -> 0
        path = get_path(0, 2, forward_state)
        self.assertIsNone(path)

    def test_loop_detection_longer(self):
        # Path: 0 -> 1 -> 2 -> 1 (loop 1 -> 2 -> 1 while trying to reach 3)
        forward_state = {(0, 3): 1, (1, 3): 2, (2, 3): 1}
        path = get_path(0, 3, forward_state)
        self.assertIsNone(path)

    def test_exceed_max_hops(self):
        # get_path has max_hops = 200
        # Create a forward_state for a path of 201 hops to a dummy destination 999
        forward_state = {}
        for i in range(201):  # Path 0 -> 1 -> ... -> 200
            forward_state[(i, 999)] = i + 1
        # Last hop to destination (this won't be reached if max_hops is 200 for a 201-node path)
        # The path length is number of nodes - 1. A path with 201 nodes has 200 hops.
        # If path list can have 201 nodes (max_hops=200 allows 201 items in list 'path')
        # Let's test a path that would result in len(path) == 202

        # To make path have len(path) > max_hops (e.g. > 200)
        # We need current_hop to take 200 steps, adding 200 nodes after src.
        # path = [src, n1, n2, ..., n200]. len(path) = 201. This should be allowed.
        # If it tries to add n201, path becomes [src, ..., n201], len(path) = 202 > max_hops (if max_hops is 200 for path length)
        # The get_path sets max_hops = 200 for the *length of the path list*.

        f_state_long = {}
        num_nodes_in_path = 202  # This will make len(path) = 202, which is > max_hops=200
        for i in range(num_nodes_in_path - 1):  # 0 to 200
            f_state_long[(i, num_nodes_in_path - 1)] = i + 1  # (node_i, dest_201) -> node_i+1

        path = get_path(0, num_nodes_in_path - 1, f_state_long)
        self.assertIsNone(path, "Path should be None if it exceeds max_hops")

    def test_unreachable_destination(self):
        # Destination exists, but no rules lead to it from a certain point
        forward_state = {
            (0, 3): 1,
            (1, 3): 2,
            # No rule from (2,3)
        }
        path = get_path(0, 3, forward_state)
        self.assertIsNone(path)


class TestComputePathLengthWithoutGraph(unittest.TestCase):
    """Tests for the compute_path_length_without_graph function."""

    @classmethod
    def setUpClass(cls):
        cls.ts = load.timescale()
        cls.epoch = cls.ts.utc(2025, 5, 8, 0, 0, 0)  # Fixed epoch for all tests
        cls.time_since_epoch_ns = 0  # Calculations at the epoch time

        # Mock satellite objects (actual Skyfield objects, but their positions won't matter due to patching distances)
        # We only need them to be distinguishable for the mocked distance functions.
        cls.sat0 = EarthSatellite(
            "1 25544U 98067A   25128.00000000  .00000000  00000-0  00000-0 0  9990",
            "2 25544  51.6400 200.0000 0006000 100.0000 200.0000 15.50000000000000",
            "SAT0",
            cls.ts,
        )
        cls.sat1 = EarthSatellite(
            "1 25545U 98067B   25128.00000000  .00000000  00000-0  00000-0 0  9991",
            "2 25545  51.6400 201.0000 0006000 100.0000 200.0000 15.50000000000001",
            "SAT1",
            cls.ts,
        )
        cls.sat2 = EarthSatellite(
            "1 25546U 98067C   25128.00000000  .00000000  00000-0  00000-0 0  9992",
            "2 25546  51.6400 202.0000 0006000 100.0000 200.0000 15.50000000000002",
            "SAT2",
            cls.ts,
        )
        cls.satellites = [cls.sat0, cls.sat1, cls.sat2]
        cls.num_sats = len(cls.satellites)

        # Mock ground station data
        cls.gs0_id_local = 0
        cls.gs1_id_local = 1
        cls.gs0_id_global = cls.num_sats + cls.gs0_id_local  # e.g., 3
        cls.gs1_id_global = cls.num_sats + cls.gs1_id_local  # e.g., 4

        cls.ground_stations = [
            {
                "gid": cls.gs0_id_local,
                "name": "GS0",
                "latitude_degrees_str": "0.0",
                "longitude_degrees_str": "0.0",
                "elevation_m_str": "0",
            },
            {
                "gid": cls.gs1_id_local,
                "name": "GS1",
                "latitude_degrees_str": "10.0",
                "longitude_degrees_str": "10.0",
                "elevation_m_str": "0",
            },
        ]

        cls.max_gsl_length_m = 2000 * 1000  # 2000 km
        cls.max_isl_length_m = 3000 * 1000  # 3000 km

        # Default list_isls (can be overridden in tests)
        cls.list_isls = [(0, 1), (1, 2)]  # ISLs between 0-1 and 1-2

    # Patch the distance calculation functions for predictable results
    # Note: the target for patch should be where the function is *looked up*,
    # which is in 'post_analysis.graph_tools' where it's imported.
    @patch("src.post_analysis.graph_tools.distance_m_ground_station_to_satellite")
    @patch("src.post_analysis.graph_tools.distance_m_between_satellites")
    def test_valid_path_gsl_isl_gsl(self, mock_dist_sat_sat, mock_dist_gs_sat):
        # Path: GS0 -> Sat0 -> Sat1 -> Sat2 -> GS1
        # Global IDs: gs0_id_global, 0, 1, 2, gs1_id_global
        path = [self.gs0_id_global, 0, 1, 2, self.gs1_id_global]

        # Define mocked distances
        # GS0 <-> Sat0 = 1000km
        # Sat0 <-> Sat1 = 1500km
        # Sat1 <-> Sat2 = 1600km
        # Sat2 <-> GS1 = 1100km
        def dist_gs_sat_side_effect(gs_dict, sat_obj, epoch_str, time_str):
            if gs_dict["gid"] == self.gs0_id_local and sat_obj == self.sat0:
                return 1000 * 1000
            if gs_dict["gid"] == self.gs1_id_local and sat_obj == self.sat2:
                return 1100 * 1000
            return self.max_gsl_length_m + 1000  # Default to out of range

        def dist_sat_sat_side_effect(s1_obj, s2_obj, epoch_str, time_str):
            if (s1_obj == self.sat0 and s2_obj == self.sat1) or (
                s1_obj == self.sat1 and s2_obj == self.sat0
            ):
                return 1500 * 1000
            if (s1_obj == self.sat1 and s2_obj == self.sat2) or (
                s1_obj == self.sat2 and s2_obj == self.sat1
            ):
                return 1600 * 1000
            return self.max_isl_length_m + 1000  # Default to out of range

        mock_dist_gs_sat.side_effect = dist_gs_sat_side_effect
        mock_dist_sat_sat.side_effect = dist_sat_sat_side_effect

        expected_length = (1000 + 1500 + 1600 + 1100) * 1000  # meters

        calculated_length = compute_path_length_without_graph(
            path,
            self.epoch,
            self.time_since_epoch_ns,
            self.satellites,
            self.ground_stations,
            self.list_isls,
            self.max_gsl_length_m,
            self.max_isl_length_m,
        )
        self.assertIsNotNone(calculated_length)
        self.assertAlmostEqual(calculated_length, expected_length, delta=1)

    @patch("src.post_analysis.graph_tools.distance_m_between_satellites")
    def test_invalid_isl_too_long(self, mock_dist_sat_sat):
        path = [0, 1]  # Sat0 -> Sat1
        mock_dist_sat_sat.return_value = self.max_isl_length_m + 1000  # Exceeds max length

        calculated_length = compute_path_length_without_graph(
            path,
            self.epoch,
            self.time_since_epoch_ns,
            self.satellites,
            self.ground_stations,
            self.list_isls,
            self.max_gsl_length_m,
            self.max_isl_length_m,
        )
        self.assertIsNone(calculated_length)

    @patch("src.post_analysis.graph_tools.distance_m_between_satellites")
    def test_invalid_isl_not_in_list(self, mock_dist_sat_sat):
        path = [0, 2]  # Sat0 -> Sat2 (ISL (0,2) is not in self.list_isls by default)
        mock_dist_sat_sat.return_value = 1000 * 1000  # Within range

        # self.list_isls is [(0,1), (1,2)]
        calculated_length = compute_path_length_without_graph(
            path,
            self.epoch,
            self.time_since_epoch_ns,
            self.satellites,
            self.ground_stations,
            self.list_isls,
            self.max_gsl_length_m,
            self.max_isl_length_m,
        )
        self.assertIsNone(calculated_length)

    @patch("src.post_analysis.graph_tools.distance_m_ground_station_to_satellite")
    def test_invalid_gsl_too_long(self, mock_dist_gs_sat):
        path = [self.gs0_id_global, 0]  # GS0 -> Sat0
        mock_dist_gs_sat.return_value = self.max_gsl_length_m + 1000  # Exceeds max length

        calculated_length = compute_path_length_without_graph(
            path,
            self.epoch,
            self.time_since_epoch_ns,
            self.satellites,
            self.ground_stations,
            self.list_isls,
            self.max_gsl_length_m,
            self.max_isl_length_m,
        )
        self.assertIsNone(calculated_length)

    def test_path_none(self):
        calculated_length = compute_path_length_without_graph(
            None,
            self.epoch,
            self.time_since_epoch_ns,
            self.satellites,
            self.ground_stations,
            self.list_isls,
            self.max_gsl_length_m,
            self.max_isl_length_m,
        )
        self.assertEqual(
            calculated_length, 0.0
        )  # Or None, depending on desired strictness from graph_tools

    def test_path_too_short(self):
        calculated_length = compute_path_length_without_graph(
            [0],
            self.epoch,
            self.time_since_epoch_ns,
            self.satellites,
            self.ground_stations,
            self.list_isls,
            self.max_gsl_length_m,
            self.max_isl_length_m,
        )
        self.assertEqual(calculated_length, 0.0)  # Or None

    def test_path_with_invalid_gs_id(self):
        # Global GS ID that's out of bounds for the ground_stations list
        invalid_gs_id = self.num_sats + len(
            self.ground_stations
        )  # e.g., if 2 GS, this is GS2 which doesn't exist
        path = [invalid_gs_id, 0]

        calculated_length = compute_path_length_without_graph(
            path,
            self.epoch,
            self.time_since_epoch_ns,
            self.satellites,
            self.ground_stations,
            self.list_isls,
            self.max_gsl_length_m,
            self.max_isl_length_m,
        )
        self.assertIsNone(calculated_length, "Path with out-of-bounds GS ID should be None")

    def test_path_gs_to_gs_hop(self):
        # Path contains GS -> GS, which is disallowed by the function's internal logic
        path = [self.gs0_id_global, self.gs1_id_global]
        calculated_length = compute_path_length_without_graph(
            path,
            self.epoch,
            self.time_since_epoch_ns,
            self.satellites,
            self.ground_stations,
            self.list_isls,
            self.max_gsl_length_m,
            self.max_isl_length_m,
        )
        self.assertIsNone(calculated_length, "Direct GS-GS hop should result in None path length")
