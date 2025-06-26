"""
Tests for TopologicalNetworkAddress class.

This module tests the 6GRUPA address system used in topological routing,
including address generation, conversion, and validation.
"""

import unittest

from src.topology.satellite.topological_network_address import TopologicalNetworkAddress


class TestTopologicalNetworkAddress(unittest.TestCase):
    """Test cases for TopologicalNetworkAddress functionality."""

    def test_from_6grupa_single_shell(self):
        """Test address generation from satellite IDs with single shell assumption."""
        test_cases = [
            (0, 0, 0, 0),    # First satellite
            (1, 0, 0, 1),    # Second satellite in same plane
            (64, 0, 1, 0),   # First satellite in second plane (assuming 64 sats per plane)
            (65, 0, 1, 1),   # Second satellite in second plane
        ]
        
        for sat_id, expected_shell, expected_plane, expected_sat_idx in test_cases:
            with self.subTest(sat_id=sat_id):
                addr = TopologicalNetworkAddress.from_6grupa(sat_id)
                
                self.assertEqual(addr.shell_id, expected_shell, 
                               f"Incorrect shell_id for satellite {sat_id}")
                self.assertEqual(addr.plane_id, expected_plane, 
                               f"Incorrect plane_id for satellite {sat_id}")
                self.assertEqual(addr.sat_index, expected_sat_idx, 
                               f"Incorrect sat_index for satellite {sat_id}")
                self.assertEqual(addr.subnet_index, 0, 
                               f"Satellites should have subnet_index=0 for satellite {sat_id}")

    def test_satellite_vs_ground_station_properties(self):
        """Test the is_satellite and is_ground_station properties."""
        # Test satellite address (subnet_index = 0)
        sat_addr = TopologicalNetworkAddress(
            shell_id=0, plane_id=1, sat_index=10, subnet_index=0
        )
        self.assertTrue(sat_addr.is_satellite)
        self.assertFalse(sat_addr.is_ground_station)
        
        # Test ground station address (subnet_index > 0)
        gs_addr = TopologicalNetworkAddress(
            shell_id=0, plane_id=1, sat_index=10, subnet_index=5
        )
        self.assertFalse(gs_addr.is_satellite)
        self.assertTrue(gs_addr.is_ground_station)

    def test_get_satellite_address(self):
        """Test getting the satellite address from any address."""
        # Test with satellite address (should return itself)
        sat_addr = TopologicalNetworkAddress(
            shell_id=0, plane_id=2, sat_index=15, subnet_index=0
        )
        self.assertEqual(sat_addr.get_satellite_address(), sat_addr)
        
        # Test with ground station address (should return satellite part)
        gs_addr = TopologicalNetworkAddress(
            shell_id=0, plane_id=2, sat_index=15, subnet_index=7
        )
        expected_sat_addr = TopologicalNetworkAddress(
            shell_id=0, plane_id=2, sat_index=15, subnet_index=0
        )
        self.assertEqual(gs_addr.get_satellite_address(), expected_sat_addr)

    def test_to_integer_conversion(self):
        """Test conversion to integer representation."""
        test_addresses = [
            TopologicalNetworkAddress(0, 0, 0, 0),
            TopologicalNetworkAddress(0, 0, 1, 0),
            TopologicalNetworkAddress(0, 1, 0, 0),
            TopologicalNetworkAddress(1, 0, 0, 0),
            TopologicalNetworkAddress(0, 0, 0, 5),
            TopologicalNetworkAddress(1, 2, 3, 4),
        ]
        
        for addr in test_addresses:
            with self.subTest(addr=addr):
                integer_repr = addr.to_integer()
                self.assertIsInstance(integer_repr, int)
                self.assertGreaterEqual(integer_repr, 0)

    def test_from_integer_conversion(self):
        """Test conversion from integer representation."""
        test_integers = [0, 1, 32, 64, 1000, 2048, 4096]
        
        for integer_val in test_integers:
            with self.subTest(integer_val=integer_val):
                try:
                    addr = TopologicalNetworkAddress.from_integer(integer_val)
                    self.assertIsInstance(addr, TopologicalNetworkAddress)
                    
                    # Verify components are within valid ranges
                    self.assertGreaterEqual(addr.shell_id, 0)
                    self.assertGreaterEqual(addr.plane_id, 0)
                    self.assertGreaterEqual(addr.sat_index, 0)
                    self.assertGreaterEqual(addr.subnet_index, 0)
                except ValueError:
                    # Some integers may be invalid due to component limits
                    pass

    def test_round_trip_conversion(self):
        """Test round-trip conversion: address -> integer -> address."""
        test_addresses = [
            TopologicalNetworkAddress(0, 0, 0, 0),
            TopologicalNetworkAddress(0, 0, 1, 0),
            TopologicalNetworkAddress(0, 1, 0, 0),
            TopologicalNetworkAddress(0, 0, 0, 1),
            TopologicalNetworkAddress(0, 5, 10, 0),
            TopologicalNetworkAddress(0, 10, 20, 5),
        ]
        
        for original_addr in test_addresses:
            with self.subTest(addr=original_addr):
                # Convert to integer and back
                integer_repr = original_addr.to_integer()
                reconstructed_addr = TopologicalNetworkAddress.from_integer(integer_repr)
                
                # Should be identical
                self.assertEqual(original_addr, reconstructed_addr)

    def test_satellite_id_mapping_consistency(self):
        """Test that satellite ID mapping is consistent and deterministic."""
        # Test a range of satellite IDs
        sat_ids = [0, 1, 10, 50, 100, 200, 500]
        
        for sat_id in sat_ids:
            with self.subTest(sat_id=sat_id):
                # Generate address twice - should be identical
                addr1 = TopologicalNetworkAddress.from_6grupa(sat_id)
                addr2 = TopologicalNetworkAddress.from_6grupa(sat_id)
                
                self.assertEqual(addr1, addr2, 
                               f"Address generation not deterministic for satellite {sat_id}")
                
                # Test round-trip through integer conversion
                integer_repr = addr1.to_integer()
                reconstructed = TopologicalNetworkAddress.from_integer(integer_repr)
                
                self.assertEqual(addr1, reconstructed, 
                               f"Round-trip conversion failed for satellite {sat_id}")

    def test_address_ordering_by_satellite_id(self):
        """Test that satellites with sequential IDs get reasonable address distribution."""
        # Generate addresses for sequential satellite IDs
        addresses = []
        for sat_id in range(20):
            addr = TopologicalNetworkAddress.from_6grupa(sat_id)
            addresses.append((sat_id, addr))
        
        # Verify all are satellite addresses
        for sat_id, addr in addresses:
            self.assertTrue(addr.is_satellite, f"Satellite {sat_id} should have satellite address")
            self.assertEqual(addr.shell_id, 0, f"Should use single shell for satellite {sat_id}")
        
        # Verify addresses are distinct
        addr_set = set(addr for _, addr in addresses)
        self.assertEqual(len(addr_set), len(addresses), "All satellite addresses should be unique")

    def test_string_representation(self):
        """Test the string representation of addresses."""
        # Test satellite address
        sat_addr = TopologicalNetworkAddress(0, 1, 5, 0)
        str_repr = str(sat_addr)
        self.assertIn("Sat", str_repr)
        self.assertIn("sh:0", str_repr)
        self.assertIn("o:1", str_repr)
        self.assertIn("s:5", str_repr)
        
        # Test ground station address
        gs_addr = TopologicalNetworkAddress(0, 1, 5, 3)
        str_repr = str(gs_addr)
        self.assertIn("GS[3]", str_repr)
        self.assertIn("sh:0", str_repr)
        self.assertIn("o:1", str_repr)
        self.assertIn("s:5", str_repr)

    def test_address_validation(self):
        """Test address component validation."""
        # Valid address should work
        valid_addr = TopologicalNetworkAddress(0, 1, 2, 0)
        self.assertIsInstance(valid_addr, TopologicalNetworkAddress)
        
        # Test invalid shell_id
        with self.assertRaises(ValueError):
            TopologicalNetworkAddress(-1, 0, 0, 0)
        
        # Test invalid plane_id
        with self.assertRaises(ValueError):
            TopologicalNetworkAddress(0, -1, 0, 0)
        
        # Test invalid subnet_index
        with self.assertRaises(ValueError):
            TopologicalNetworkAddress(0, 0, 0, -1)

    def test_large_satellite_ids(self):
        """Test handling of large satellite IDs that might require multiple shells."""
        # Test some larger satellite IDs
        large_sat_ids = [1000, 5000, 10000]
        
        for sat_id in large_sat_ids:
            with self.subTest(sat_id=sat_id):
                try:
                    addr = TopologicalNetworkAddress.from_6grupa(sat_id)
                    
                    # Should be a valid satellite address
                    self.assertTrue(addr.is_satellite)
                    self.assertGreaterEqual(addr.shell_id, 0)
                    self.assertGreaterEqual(addr.plane_id, 0)
                    self.assertGreaterEqual(addr.sat_index, 0)
                    
                    # Test round-trip conversion
                    integer_repr = addr.to_integer()
                    reconstructed = TopologicalNetworkAddress.from_integer(integer_repr)
                    self.assertEqual(addr, reconstructed)
                    
                except ValueError as e:
                    # May exceed maximum address space - this is acceptable
                    self.assertIn("exceeds", str(e).lower(), 
                                f"Unexpected error for satellite {sat_id}: {e}")


if __name__ == "__main__":
    unittest.main()
