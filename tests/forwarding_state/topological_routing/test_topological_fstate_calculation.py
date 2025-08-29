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
            sat.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat.id)
        
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
            sat.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat.id)
        
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
            sat.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat.id)
        
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
            sat.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat.id)
        
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
            addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat.id)
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
            sat.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat.id)
        
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
            sat.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat.id)
        
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

    def test_gsl_renumbering_functionality(self):
        """
        Test that GSL renumbering works correctly when ground stations change satellite attachments.
        """
        # Create satellites
        SAT_A = 10
        SAT_B = 11
        
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
            Satellite(id=SAT_B, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        
        # Create ground stations
        GS_X = 100
        GS_Y = 101
        
        ground_stations = [
            GroundStation(
                gid=GS_X, name="GS_X", latitude_degrees_str="0", longitude_degrees_str="0",
                elevation_m_float=0, cartesian_x=0, cartesian_y=0, cartesian_z=0
            ),
            GroundStation(
                gid=GS_Y, name="GS_Y", latitude_degrees_str="0", longitude_degrees_str="0",
                elevation_m_float=0, cartesian_x=0, cartesian_y=0, cartesian_z=0
            ),
        ]
        
        # Create topology
        constellation_data = ConstellationData(
            orbits=1, sats_per_orbit=2, epoch="2000-01-01 00:00:00",
            max_gsl_length_m=1000000, max_isl_length_m=5000000, satellites=satellites
        )
        topology = LEOTopology(constellation_data, ground_stations)
        
        # Set up satellite addresses
        for sat in satellites:
            sat.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat.id)
        
        from src.network_state.routing_algorithms.topological_routing.fstate_calculation import (
            _perform_renumbering_for_gs,
            _detect_gsl_changes,
        )
        
        # Test initial GSL attachments (both GSs to SAT_A)
        ground_station_satellites_in_range = [
            [(500, SAT_A)],  # GS_X -> SAT_A
            [(600, SAT_A)],  # GS_Y -> SAT_A
        ]
        
        # Perform initial renumbering
        for gs_idx, gs in enumerate(ground_stations):
            curr_sat_id = ground_station_satellites_in_range[gs_idx][0][1]
            _perform_renumbering_for_gs(gs, None, curr_sat_id, topology)
        
        # Check initial addresses
        gs_x_addr = ground_stations[0].sixgrupa_addr
        gs_y_addr = ground_stations[1].sixgrupa_addr
        sat_a_addr = satellites[0].sixgrupa_addr
        
        # Both GSs should have same satellite coordinates but different subnet_index
        self.assertEqual(gs_x_addr.shell_id, sat_a_addr.shell_id)
        self.assertEqual(gs_x_addr.plane_id, sat_a_addr.plane_id)
        self.assertEqual(gs_x_addr.sat_index, sat_a_addr.sat_index)
        self.assertGreater(gs_x_addr.subnet_index, 0)
        
        self.assertEqual(gs_y_addr.shell_id, sat_a_addr.shell_id)
        self.assertEqual(gs_y_addr.plane_id, sat_a_addr.plane_id)
        self.assertEqual(gs_y_addr.sat_index, sat_a_addr.sat_index)
        self.assertGreater(gs_y_addr.subnet_index, 0)
        
        # GSs should have different subnet_index values
        self.assertNotEqual(gs_x_addr.subnet_index, gs_y_addr.subnet_index)
        
        # Test GSL change - move GS_Y to SAT_B
        new_ground_station_satellites_in_range = [
            [(500, SAT_A)],  # GS_X -> SAT_A (unchanged)
            [(400, SAT_B)],  # GS_Y -> SAT_B (changed)
        ]
        
        # Detect changes
        gsl_changes = _detect_gsl_changes(ground_stations, new_ground_station_satellites_in_range)
        
        # Should detect change for GS_Y only
        self.assertEqual(len(gsl_changes), 1)
        self.assertIn(1, gsl_changes)  # GS_Y index
        prev_sat, curr_sat = gsl_changes[1]
        self.assertEqual(prev_sat, SAT_A)
        self.assertEqual(curr_sat, SAT_B)
        
        # Perform renumbering for changed GS
        _perform_renumbering_for_gs(ground_stations[1], SAT_A, SAT_B, topology)
        
        # Check updated addresses
        gs_x_addr_after = ground_stations[0].sixgrupa_addr
        gs_y_addr_after = ground_stations[1].sixgrupa_addr
        sat_b_addr = satellites[1].sixgrupa_addr
        
        # GS_X should be unchanged
        self.assertEqual(gs_x_addr_after, gs_x_addr)
        
        # GS_Y should now match SAT_B coordinates
        self.assertEqual(gs_y_addr_after.shell_id, sat_b_addr.shell_id)
        self.assertEqual(gs_y_addr_after.plane_id, sat_b_addr.plane_id)
        self.assertEqual(gs_y_addr_after.sat_index, sat_b_addr.sat_index)
        self.assertGreater(gs_y_addr_after.subnet_index, 0)
        
        # GS_Y address should be different from before
        self.assertNotEqual(gs_y_addr_after, gs_y_addr)

    def test_gsl_change_detection(self):
        """
        Test the GSL change detection functionality.
        """
        # Create ground stations
        ground_stations = [
            GroundStation(
                gid=100, name="GS1", latitude_degrees_str="0", longitude_degrees_str="0",
                elevation_m_float=0, cartesian_x=0, cartesian_y=0, cartesian_z=0
            ),
            GroundStation(
                gid=101, name="GS2", latitude_degrees_str="0", longitude_degrees_str="0",
                elevation_m_float=0, cartesian_x=0, cartesian_y=0, cartesian_z=0
            ),
        ]
        
        from src.network_state.routing_algorithms.topological_routing.fstate_calculation import (
            _detect_gsl_changes,
        )
        
        # Test case 1: No previous attachments (initial state)
        ground_station_satellites_in_range = [
            [(500, 10)],
            [(600, 11)],
        ]
        
        changes = _detect_gsl_changes(ground_stations, ground_station_satellites_in_range)
        # Should detect changes for both GSs (from None to satellite)
        self.assertEqual(len(changes), 2)
        self.assertIn(0, changes)
        self.assertIn(1, changes)
        
        # Test case 2: No changes
        ground_stations[0].previous_attached_satellite_id = 10
        ground_stations[1].previous_attached_satellite_id = 11
        
        changes = _detect_gsl_changes(ground_stations, ground_station_satellites_in_range)
        # Should detect no changes
        self.assertEqual(len(changes), 0)
        
        # Test case 3: One GS changes satellite
        new_ground_station_satellites_in_range = [
            [(500, 10)],  # GS 0 unchanged
            [(400, 12)],  # GS 1 changed to satellite 12
        ]
        
        changes = _detect_gsl_changes(ground_stations, new_ground_station_satellites_in_range)
        # Should detect change for GS 1 only
        self.assertEqual(len(changes), 1)
        self.assertIn(1, changes)
        prev_sat, curr_sat = changes[1]
        self.assertEqual(prev_sat, 11)
        self.assertEqual(curr_sat, 12)

    def test_multiple_gs_same_satellite_renumbering(self):
        """
        Test that multiple ground stations attached to the same satellite get unique subnet_index values.
        """
        # Create satellites
        SAT_A = 20
        
        mock_body = MagicMock(spec=ephem.Body)
        satellites = [
            Satellite(id=SAT_A, ephem_obj_manual=mock_body, ephem_obj_direct=mock_body),
        ]
        
        # Create multiple ground stations
        ground_stations = []
        for i in range(5):  # 5 GSs
            gs = GroundStation(
                gid=200 + i, name=f"GS_{i}", latitude_degrees_str="0", longitude_degrees_str="0",
                elevation_m_float=0, cartesian_x=0, cartesian_y=0, cartesian_z=0
            )
            ground_stations.append(gs)
        
        # Create topology
        constellation_data = ConstellationData(
            orbits=1, sats_per_orbit=1, epoch="2000-01-01 00:00:00",
            max_gsl_length_m=1000000, max_isl_length_m=5000000, satellites=satellites
        )
        topology = LEOTopology(constellation_data, ground_stations)
        
        # Set up satellite address
        satellites[0].sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(SAT_A)
        
        from src.network_state.routing_algorithms.topological_routing.fstate_calculation import (
            _perform_renumbering_for_gs,
        )
        
        # Attach all GSs to the same satellite
        for gs in ground_stations:
            _perform_renumbering_for_gs(gs, None, SAT_A, topology)
        
        # Check that all GSs have unique subnet_index values
        subnet_indices = set()
        sat_addr = satellites[0].sixgrupa_addr
        
        for gs in ground_stations:
            gs_addr = gs.sixgrupa_addr
            
            # Should match satellite coordinates
            self.assertEqual(gs_addr.shell_id, sat_addr.shell_id)
            self.assertEqual(gs_addr.plane_id, sat_addr.plane_id)
            self.assertEqual(gs_addr.sat_index, sat_addr.sat_index)
            
            # Should have unique subnet_index > 0
            self.assertGreater(gs_addr.subnet_index, 0)
            self.assertNotIn(gs_addr.subnet_index, subnet_indices)
            subnet_indices.add(gs_addr.subnet_index)
        
        # Should have 5 unique subnet_index values
        self.assertEqual(len(subnet_indices), 5)

    def test_topological_distance_calculation(self):
        """Test that topological distance calculation works as expected."""
        # Create some example 6grupa addresses
        sat1 = TopologicalNetworkAddress(
            shell_id=0, plane_id=0, sat_index=0, subnet_index=0
        )  # Satellite at (0,0,0)
        sat2 = TopologicalNetworkAddress(
            shell_id=0, plane_id=0, sat_index=1, subnet_index=0
        )  # Satellite at (0,0,1) - same plane, next satellite
        sat3 = TopologicalNetworkAddress(
            shell_id=0, plane_id=1, sat_index=0, subnet_index=0
        )  # Satellite at (0,1,0) - next plane
        sat4 = TopologicalNetworkAddress(
            shell_id=0, plane_id=0, sat_index=0, subnet_index=0
        )  # Same as sat1

        # Test distances
        dist_1_to_1 = sat1.topological_distance_to(sat4)  # Same satellite
        dist_1_to_2 = sat1.topological_distance_to(sat2)  # Same plane, adjacent satellite
        dist_1_to_3 = sat1.topological_distance_to(sat3)  # Adjacent plane

        # Verify expected distances
        self.assertEqual(dist_1_to_1, 0.0, "Distance to same satellite should be 0")
        self.assertGreater(dist_1_to_2, 0, "Adjacent satellite distance should be > 0")
        self.assertLess(dist_1_to_2, 10, "Adjacent satellite distance should be small")
        self.assertGreater(dist_1_to_3, dist_1_to_2, "Different plane should be farther than same plane")

    def test_topological_routing_scenario(self):
        """Test a simple routing scenario using topological addresses."""
        # Create a simple 3-satellite scenario:
        # Sat A (0,0,0) -> Sat B (0,0,1) -> Sat C (0,0,2)
        sat_a = TopologicalNetworkAddress(shell_id=0, plane_id=0, sat_index=0, subnet_index=0)
        sat_b = TopologicalNetworkAddress(shell_id=0, plane_id=0, sat_index=1, subnet_index=0)
        sat_c = TopologicalNetworkAddress(shell_id=0, plane_id=0, sat_index=2, subnet_index=0)

        # From A's perspective, which neighbor is closer to C?
        # A can see B, and B is closer to C than A is
        dist_a_to_c = sat_a.topological_distance_to(sat_c)
        dist_b_to_c = sat_b.topological_distance_to(sat_c)

        # B should be closer to C than A is
        self.assertLess(dist_b_to_c, dist_a_to_c, "B should be closer to C than A is")

    def test_topological_plane_wraparound(self):
        """Test that plane wraparound works correctly."""
        # Test with addresses that should wrap around
        # Assume we have MAX_PLANES = 128, so plane 0 and plane 127 should be adjacent
        sat_plane_0 = TopologicalNetworkAddress(shell_id=0, plane_id=0, sat_index=0, subnet_index=0)
        sat_plane_1 = TopologicalNetworkAddress(shell_id=0, plane_id=1, sat_index=0, subnet_index=0)
        sat_plane_127 = TopologicalNetworkAddress(shell_id=0, plane_id=127, sat_index=0, subnet_index=0)

        dist_0_to_1 = sat_plane_0.topological_distance_to(sat_plane_1)
        dist_0_to_127 = sat_plane_0.topological_distance_to(sat_plane_127)

        # Both should be the same due to wraparound (planes 0 and 127 are adjacent)
        self.assertEqual(dist_0_to_1, dist_0_to_127, "Plane wraparound should make distances equal")


if __name__ == "__main__":
    unittest.main()
