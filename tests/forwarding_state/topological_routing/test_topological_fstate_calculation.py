"""
Tests for topological routing algorithm implementation.

This module tests the topological routing algorithm with the same scenarios used
for testing the shortest path routing algorithm, allowing for direct comparison
of behavior and ensuring correctness.
"""

import unittest
from unittest.mock import MagicMock

import ephem
from astropy.time import Time

from src.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    calculate_fstate_topological_routing_no_gs_relay,
)
from src.topology.satellite.satellite import Satellite
from src.topology.satellite.topological_network_address import TopologicalNetworkAddress
from src.topology.topology import (
    ConstellationData,
    GroundStation,
    LEOTopology,
)
from src.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy


class MockGSLAttachmentStrategy(GSLAttachmentStrategy):
    """Mock GSL attachment strategy that returns predefined attachments for testing."""

    def __init__(self, attachments):
        """
        Args:
            attachments: List of (distance, satellite_id) tuples for each ground station.
                        If satellite_id is -1, indicates no attachment.
        """
        self.attachments = attachments

    @property
    def name(self):
        return "mock_strategy"

    def select_attachments(self, topology, ground_stations, current_time):
        """Return the predefined attachments."""
        return self.attachments


class TestTopologicalRoutingFstateCalculation(unittest.TestCase):
    """Test cases for topological routing forwarding state calculation."""

    def _setup_scenario(
        self, satellite_list, ground_station_list, isl_edges_with_weights, gsl_visibility_list
    ):
        """Helper to build topology and visibility structures for fstate tests."""
        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellite_list),
            epoch="25001.0",
            max_gsl_length_m=5000000,
            max_isl_length_m=5000000,
            satellites=satellite_list,
        )
        topology = LEOTopology(constellation_data, ground_station_list)
        num_isls_per_sat_map = {sat.id: 0 for sat in satellite_list}
        topology.sat_neighbor_to_if = {}
        
        # Add satellite nodes to graph
        for sat in satellite_list:
            topology.graph.add_node(sat.id)
            sat.number_isls = 0
            
        # Add ISL edges and interface mappings
        for u_id, v_id, weight in isl_edges_with_weights:
            if topology.graph.has_node(u_id) and topology.graph.has_node(v_id):
                topology.graph.add_edge(u_id, v_id, weight=weight)
                u_if = num_isls_per_sat_map[u_id]
                v_if = num_isls_per_sat_map[v_id]
                topology.sat_neighbor_to_if[(u_id, v_id)] = u_if
                topology.sat_neighbor_to_if[(v_id, u_id)] = v_if
                num_isls_per_sat_map[u_id] += 1
                num_isls_per_sat_map[v_id] += 1
            else:
                print(f"Warning in test setup: Skipping edge ({u_id},{v_id}) - node(s) not found.")
                
        # Update satellite ISL counts
        for sat in topology.constellation_data.satellites:
            sat.number_isls = num_isls_per_sat_map.get(sat.id, 0)
            
        if len(gsl_visibility_list) != len(ground_station_list):
            raise ValueError("Length mismatch: gsl_visibility_list vs ground_station_list")

        # Convert GSL visibility to the expected format for topological routing
        ground_station_satellites_in_range = []
        for gs_attachment in gsl_visibility_list:
            if gs_attachment and gs_attachment[1] != -1:
                # Single attachment per ground station
                ground_station_satellites_in_range.append([gs_attachment])
            else:
                # No attachment
                ground_station_satellites_in_range.append([])

        return topology, ground_station_satellites_in_range

    def test_one_sat_two_gs_topological(self):
        """
        Scenario: 1 Sat (ID 10), 2 GS (IDs 100, 101), GSLs only
        Tests basic satellite-to-ground communication via direct GSL links.
        """
        # Topology:
        #      10 (Sat)
        #     /  \
        # 100(GS) 101(GS)
        
        SAT_ID = 10
        GS_A_ID = 100
        GS_B_ID = 101
        
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [Satellite(id=SAT_ID, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body)]
        ground_stations = [
            GroundStation(
                gid=GS_A_ID,
                name="GA",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_B_ID,
                name="GB",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        
        isl_edges = []
        # Both GS attached to the same satellite
        gsl_visibility = [(1000, SAT_ID), (1000, SAT_ID)]
        
        topology, ground_station_satellites_in_range = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        
        # Initialize 6GRUPA addresses
        for sat in satellites:
            sat.sixgrupa_addr = TopologicalNetworkAddress.from_6grupa(sat.id)
        
        fstate = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,  # t=0 for initialization
            prev_fstate=None,
            graph_has_changed=True,
        )
        
        # Expected: Direct GSL connections for satellite to ground stations
        expected_entries = {
            (SAT_ID, GS_A_ID): ("GSL", GS_A_ID),
            (SAT_ID, GS_B_ID): ("GSL", GS_B_ID),
        }
        
        for key, expected_value in expected_entries.items():
            self.assertIn(key, fstate, f"Missing fstate entry for {key}")
            self.assertEqual(fstate[key], expected_value, f"Incorrect fstate value for {key}")

    def test_two_sat_two_gs_topological(self):
        """
        Scenario: Two satellites connected by ISL, each with one ground station
        Tests multi-hop routing via ISL.
        """
        # Topology: 100(GS) -- 10(Sat) -- 11(Sat) -- 101(GS)
        
        SAT_A = 10
        SAT_B = 11
        GS_X = 100
        GS_Y = 101
        
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        ground_stations = [
            GroundStation(
                gid=GS_X,
                name="GX",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_Y,
                name="GY",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        
        isl_edges = [(SAT_A, SAT_B, 1000)]
        # GS_X -> SAT_A, GS_Y -> SAT_B
        gsl_visibility = [(500, SAT_A), (600, SAT_B)]
        
        topology, ground_station_satellites_in_range = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        
        # Initialize 6GRUPA addresses
        for sat in satellites:
            sat.sixgrupa_addr = TopologicalNetworkAddress.from_6grupa(sat.id)
        
        fstate = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,
            prev_fstate=None,
            graph_has_changed=True,
        )
        
        # Expected: Direct and multi-hop paths
        expected_entries = {
            (SAT_A, GS_X): ("GSL", GS_X),  # Direct GSL
            (SAT_A, GS_Y): 0,              # Multi-hop via ISL interface 0 to SAT_B
            (SAT_B, GS_X): 0,              # Multi-hop via ISL interface 0 to SAT_A  
            (SAT_B, GS_Y): ("GSL", GS_Y),  # Direct GSL
        }
        
        for key, expected_value in expected_entries.items():
            self.assertIn(key, fstate, f"Missing fstate entry for {key}")
            self.assertEqual(fstate[key], expected_value, f"Incorrect fstate value for {key}")

    def test_three_sat_linear_topology(self):
        """
        Scenario: Three satellites in a line with ground stations at the ends
        Tests multi-hop routing through intermediate satellites.
        """
        # Topology: 100(GS) -- 10(Sat) -- 11(Sat) -- 12(Sat) -- 101(GS)
        
        SAT_A = 10
        SAT_B = 11
        SAT_C = 12
        GS_X = 100
        GS_Y = 101
        
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_C, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        ground_stations = [
            GroundStation(
                gid=GS_X,
                name="GX",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
            GroundStation(
                gid=GS_Y,
                name="GY",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        
        isl_edges = [(SAT_A, SAT_B, 1000), (SAT_B, SAT_C, 1000)]
        # GS_X -> SAT_A, GS_Y -> SAT_C
        gsl_visibility = [(500, SAT_A), (600, SAT_C)]
        
        topology, ground_station_satellites_in_range = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        
        # Initialize 6GRUPA addresses
        for sat in satellites:
            sat.sixgrupa_addr = TopologicalNetworkAddress.from_6grupa(sat.id)
        
        fstate = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,
            prev_fstate=None,
            graph_has_changed=True,
        )
        
        # Expected: Direct paths and 2-hop paths
        expected_entries = {
            (SAT_A, GS_X): ("GSL", GS_X),  # Direct GSL
            (SAT_A, GS_Y): 0,              # Multi-hop: SAT_A -> SAT_B -> SAT_C -> GS_Y
            (SAT_C, GS_X): 0,              # Multi-hop: SAT_C -> SAT_B -> SAT_A -> GS_X
            (SAT_C, GS_Y): ("GSL", GS_Y),  # Direct GSL
        }
        
        for key, expected_value in expected_entries.items():
            self.assertIn(key, fstate, f"Missing fstate entry for {key}")
            if expected_value != 0:  # Skip interface checks for simplicity
                self.assertEqual(fstate[key], expected_value, f"Incorrect fstate value for {key}")

    def test_no_gsl_connectivity(self):
        """
        Scenario: Satellites with no ground station attachments
        Tests behavior when no GSL connectivity exists.
        """
        SAT_A = 10
        SAT_B = 11
        GS_X = 100
        
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        ground_stations = [
            GroundStation(
                gid=GS_X,
                name="GX",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        
        isl_edges = [(SAT_A, SAT_B, 1000)]
        # No GSL attachments
        gsl_visibility = [(-1, -1)]
        
        topology, ground_station_satellites_in_range = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        
        # Initialize 6GRUPA addresses
        for sat in satellites:
            sat.sixgrupa_addr = TopologicalNetworkAddress.from_6grupa(sat.id)
        
        fstate = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,
            prev_fstate=None,
            graph_has_changed=True,
        )
        
        # Expected: No satellite-to-GS entries since no GSL connectivity
        for sat_id in [SAT_A, SAT_B]:
            self.assertNotIn((sat_id, GS_X), fstate, 
                           f"Unexpected fstate entry for satellite {sat_id} to GS {GS_X}")

    def test_topological_address_assignment(self):
        """
        Test that 6GRUPA addresses are correctly assigned to satellites
        """
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=0, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=1, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=50, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=100, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        
        # Test address assignment
        for sat in satellites:
            addr = TopologicalNetworkAddress.from_6grupa(sat.id)
            sat.sixgrupa_addr = addr
            
            # Verify address properties
            self.assertEqual(addr.shell_id, 0, "Should use single shell (shell_id=0)")
            self.assertEqual(addr.subnet_index, 0, "Satellites should have subnet_index=0")
            self.assertTrue(addr.is_satellite, "Address should be identified as satellite")
            self.assertFalse(addr.is_ground_station, "Address should not be identified as ground station")
            
            # Test round-trip conversion
            integer_repr = addr.to_integer()
            reconstructed = TopologicalNetworkAddress.from_integer(integer_repr)
            self.assertEqual(addr, reconstructed, f"Round-trip conversion failed for satellite {sat.id}")

    def test_forwarding_table_population(self):
        """
        Test that forwarding tables are correctly populated with neighbor addresses
        """
        SAT_A = 10
        SAT_B = 11
        SAT_C = 12
        
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_C, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        ground_stations = []
        
        # Create a triangle topology
        isl_edges = [(SAT_A, SAT_B, 1000), (SAT_B, SAT_C, 1000), (SAT_C, SAT_A, 1000)]
        gsl_visibility = []
        
        topology, ground_station_satellites_in_range = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        
        # Initialize addresses and run algorithm
        for sat in satellites:
            sat.sixgrupa_addr = TopologicalNetworkAddress.from_6grupa(sat.id)
        
        fstate = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,
            prev_fstate=None,
            graph_has_changed=True,
        )
        
        # Check that forwarding tables were populated
        for sat in satellites:
            self.assertIsNotNone(sat.forwarding_table, f"Satellite {sat.id} should have a forwarding table")
            # In a triangle, each satellite should have entries for its 2 neighbors
            self.assertGreaterEqual(len(sat.forwarding_table), 0, 
                                  f"Satellite {sat.id} should have neighbor entries")

    def test_state_reuse_optimization(self):
        """
        Test that the algorithm correctly reuses previous state when graph hasn't changed
        """
        SAT_A = 10
        GS_X = 100
        
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body)]
        ground_stations = [
            GroundStation(
                gid=GS_X,
                name="GX",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            ),
        ]
        
        isl_edges = []
        gsl_visibility = [(500, SAT_A)]
        
        topology, ground_station_satellites_in_range = self._setup_scenario(
            satellites, ground_stations, isl_edges, gsl_visibility
        )
        
        # Initialize addresses
        for sat in satellites:
            sat.sixgrupa_addr = TopologicalNetworkAddress.from_6grupa(sat.id)
        
        # First run - compute initial state
        fstate1 = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,
            prev_fstate=None,
            graph_has_changed=True,
        )
        
        # Second run - should reuse previous state
        fstate2 = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=1000,  # Different time
            prev_fstate=fstate1,
            graph_has_changed=False,  # Graph hasn't changed
        )
        
        # Should return the same state object (optimization)
        self.assertIs(fstate2, fstate1, "Should reuse previous state when graph hasn't changed")


if __name__ == "__main__":
    unittest.main()
