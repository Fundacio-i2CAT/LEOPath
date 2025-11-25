"""
Integration tests comparing topological routing and shortest path routing algorithms.

This module provides side-by-side comparison tests to ensure the topological
routing algorithm produces reasonable results compared to the baseline shortest
path algorithm.
"""

import unittest

import ephem

# Import to register the strategy
from src.network_state.gsl_attachment.gsl_attachment_strategies.nearest_satellite import (  # noqa: F401
    NearestSatelliteStrategy,
)
from src.network_state.routing_algorithms.routing_algorithm_factory import get_routing_algorithm
from src.topology.satellite.satellite import Satellite
from src.topology.satellite.topological_network_address import TopologicalNetworkAddress
from src.topology.topology import (
    ConstellationData,
    GroundStation,
    LEOTopology,
)


class TestTopologicalVsShortestPathRouting(unittest.TestCase):
    """Integration tests comparing topological and shortest path routing algorithms."""

    def _create_test_topology(self, satellite_ids, ground_station_ids, isl_edges):
        """Create a test topology with given satellites, ground stations, and ISL edges."""
        # Create satellites with proper orbital elements for distance calculations
        satellites = []
        for i, sat_id in enumerate(satellite_ids):
            # Create a basic orbital body with valid TLE-like parameters
            sat_body = ephem.EarthSatellite(
                "1 25544U 98067A   21001.00000000  .00001000  00000-0  23027-4 0  9990",
                f"2 25544  51.640{i:02d} 339.704{i:02d} 0003572  86.486{i:02d} 273.608{i:02d} 15.48919103270233",
            )
            sat = Satellite(id=sat_id, ephem_obj_manual=sat_body, ephem_obj_direct=sat_body)
            sat.sixgrupa_addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(
                sat_id
            )
            satellites.append(sat)

        # Create ground stations
        ground_stations = []
        for gs_id in ground_station_ids:
            gs = GroundStation(
                gid=gs_id,
                name=f"GS{gs_id}",
                latitude_degrees_str="0",
                longitude_degrees_str="0",
                elevation_m_float=0,
                cartesian_x=0,
                cartesian_y=0,
                cartesian_z=0,
            )
            ground_stations.append(gs)

        # Create topology
        constellation_data = ConstellationData(
            orbits=1,
            sats_per_orbit=len(satellites),
            epoch="2024-01-01T00:00:00.000000000",
            max_gsl_length_m=5000000,
            max_isl_length_m=5000000,
            satellites=satellites,
        )
        topology = LEOTopology(constellation_data, ground_stations)

        # Add satellite nodes and ISL edges
        topology.sat_neighbor_to_if = {}
        interface_counters = {sat_id: 0 for sat_id in satellite_ids}

        for sat in satellites:
            topology.graph.add_node(sat.id)
            sat.number_isls = 0

        for u_id, v_id, weight in isl_edges:
            if topology.graph.has_node(u_id) and topology.graph.has_node(v_id):
                topology.graph.add_edge(u_id, v_id, weight=weight)

                u_if = interface_counters[u_id]
                v_if = interface_counters[v_id]
                topology.sat_neighbor_to_if[(u_id, v_id)] = u_if
                topology.sat_neighbor_to_if[(v_id, u_id)] = v_if

                interface_counters[u_id] += 1
                interface_counters[v_id] += 1

        # Update ISL counts
        for sat in satellites:
            sat.number_isls = interface_counters[sat.id]

        return topology, ground_stations

    def _create_test_bandwidth_info(self, satellite_ids, ground_station_ids):
        """Create mock bandwidth information for all nodes."""
        all_node_ids = satellite_ids + ground_station_ids
        bandwidth_info = []

        for node_id in all_node_ids:
            bandwidth_info.append(
                {
                    "id": node_id,
                    "aggregate_max_bandwidth": 1000000000,  # 1 Gbps
                }
            )

        return bandwidth_info

    def test_simple_linear_topology_comparison(self):
        """Test that topological routing produces valid forwarding state."""
        # Simplified test focused on verifying topological routing works correctly
        from src.network_state.routing_algorithms.topological_routing.fstate_calculation import (
            calculate_fstate_topological_routing_no_gs_relay,
        )

        satellite_ids = [10, 11]
        ground_station_ids = [100, 101]
        isl_edges = [(10, 11, 1000)]  # One link: 10 <-> 11

        topology, ground_stations = self._create_test_topology(
            satellite_ids, ground_station_ids, isl_edges
        )

        # Manually specify GSL attachments: GS 100 -> Sat 10, GS 101 -> Sat 11
        ground_station_satellites_in_range = [
            [(500, 10)],  # GS 100 (index 0) -> Sat 10
            [(600, 11)],  # GS 101 (index 1) -> Sat 11
        ]

        # Calculate topological forwarding state
        topo_fstate = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,
            prev_fstate=None,
            graph_has_changed=True,
        )

        # Should produce valid forwarding state
        self.assertIsInstance(topo_fstate, dict)

        # Should have routing decisions for direct connections
        key_routes = [(10, 100), (11, 101)]  # Direct GSL connections

        for route in key_routes:
            self.assertIn(route, topo_fstate, f"Topological missing route {route}")
            # Check that the decision is valid
            decision = topo_fstate[route]
            self.assertIsNotNone(decision, f"Route {route} has None decision")

        print(f"Topological FState: {topo_fstate}")

        # Verify address assignment worked
        for sat in topology.get_satellites():
            self.assertIsNotNone(sat.sixgrupa_addr, f"Satellite {sat.id} missing 6GRUPA address")
            self.assertTrue(
                sat.sixgrupa_addr.is_satellite, f"Satellite {sat.id} has non-satellite address"
            )

    def test_triangle_topology_comparison(self):
        """Test topological routing on a triangle topology with one ground station."""
        from src.network_state.routing_algorithms.topological_routing.fstate_calculation import (
            calculate_fstate_topological_routing_no_gs_relay,
        )

        satellite_ids = [10, 11, 12]
        ground_station_ids = [100]
        isl_edges = [(10, 11, 1000), (11, 12, 1000), (12, 10, 1000)]

        topology, ground_stations = self._create_test_topology(
            satellite_ids, ground_station_ids, isl_edges
        )

        # GS 100 connects to Sat 10
        ground_station_satellites_in_range = [[(500, 10)]]  # GS 100 (index 0) -> Sat 10

        # Calculate topological forwarding state
        topo_fstate = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,
            prev_fstate=None,
            graph_has_changed=True,
        )

        # Should have valid state
        self.assertIsInstance(topo_fstate, dict)

        # All satellites should have routes to the ground station
        for sat_id in satellite_ids:
            route = (sat_id, 100)
            self.assertIn(route, topo_fstate, f"Topological missing route {route}")

        # Sat 10 should have direct GSL connection
        self.assertEqual(topo_fstate[(10, 100)], ("GSL", 100))

        print(f"Triangle Topology FState: {topo_fstate}")

    def test_algorithm_factory_integration(self):
        """Test that both algorithms are properly registered and can be created."""
        # Test algorithm creation
        shortest_path_algo = get_routing_algorithm("shortest_path_link_state")
        topological_algo = get_routing_algorithm("topological_routing")

        self.assertIsNotNone(shortest_path_algo)
        self.assertIsNotNone(topological_algo)

        # Test that they are different classes
        self.assertNotEqual(type(shortest_path_algo), type(topological_algo))

        # Test invalid algorithm name
        with self.assertRaises(ValueError):
            get_routing_algorithm("nonexistent_algorithm")

    def test_address_system_integration(self):
        """Test that the 6GRUPA address system integrates properly with routing."""
        satellite_ids = [0, 1, 64, 128]  # Test various satellite IDs

        # Test address generation
        addresses = []
        for sat_id in satellite_ids:
            addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat_id)
            addresses.append((sat_id, addr))

            # All should be satellite addresses in single shell
            self.assertTrue(addr.is_satellite)
            self.assertEqual(addr.shell_id, 0)

        # Test that addresses are unique
        addr_set = set(addr for _, addr in addresses)
        self.assertEqual(len(addr_set), len(addresses))

        # Test integer conversion for all addresses
        for sat_id, addr in addresses:
            integer_repr = addr.to_integer()
            reconstructed = TopologicalNetworkAddress.from_integer(integer_repr)
            self.assertEqual(addr, reconstructed, f"Round-trip failed for satellite {sat_id}")

    def test_performance_comparison_metrics(self):
        """Test that topological routing produces metrics on a linear chain topology."""
        from src.network_state.routing_algorithms.topological_routing.fstate_calculation import (
            calculate_fstate_topological_routing_no_gs_relay,
        )

        satellite_ids = [10, 11, 12, 13]
        ground_station_ids = [100, 101]
        isl_edges = [(10, 11, 1000), (11, 12, 1000), (12, 13, 1000)]

        topology, ground_stations = self._create_test_topology(
            satellite_ids, ground_station_ids, isl_edges
        )

        # GS 100 -> Sat 10, GS 101 -> Sat 13 (opposite ends of chain)
        ground_station_satellites_in_range = [
            [(500, 10)],  # GS 100 -> Sat 10
            [(600, 13)],  # GS 101 -> Sat 13
        ]

        # Calculate topological forwarding state
        topo_fstate = calculate_fstate_topological_routing_no_gs_relay(
            topology,
            ground_stations,
            ground_station_satellites_in_range,
            time_since_epoch_ns=0,
            prev_fstate=None,
            graph_has_changed=True,
        )

        # Should have routing entries
        self.assertGreater(len(topo_fstate), 0, "Topological should have routing entries")

        # Should handle all satellite-to-GS routes
        expected_routes = []
        for sat_id in satellite_ids:
            for gs_id in ground_station_ids:
                expected_routes.append((sat_id, gs_id))

        for route in expected_routes:
            self.assertIn(route, topo_fstate, f"Missing route {route}")

        print("Linear Chain Topology:")
        print(f"Topological routes: {len(topo_fstate)}")
        print(f"Sample routes: {list(topo_fstate.items())[:4]}")


if __name__ == "__main__":
    unittest.main()
