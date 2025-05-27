# tests/topology/test_address.py
import unittest
import math  # For potentially calculating expected int values

# Import the class to test (adjust path if needed)
from src.topology.satellite.topological_network_address import (
    TopologicalNetworkAddress,
    MAX_SHELLS,
    MAX_PLANES,
    MAX_SATS_PER_PLANE,
    MAX_ENDPOINTS_PER_SAT,
)

class TestTopologicalNetworkAddress(unittest.TestCase):

    def test_valid_creation_and_attributes(self):
        """Test successful creation and attribute access."""
        addr = TopologicalNetworkAddress(shell_id=1, plane_id=10, sat_index=5, subnet_index=0)
        self.assertEqual(addr.shell_id, 1)
        self.assertEqual(addr.plane_id, 10)
        self.assertEqual(addr.sat_index, 5)
        self.assertEqual(addr.subnet_index, 0)

    def test_validation_out_of_range(self):
        """Test ValueError is raised for out-of-range components."""
        with self.assertRaises(ValueError, msg="Shell ID out of range"):
            TopologicalNetworkAddress(shell_id=MAX_SHELLS, plane_id=0, sat_index=0, subnet_index=0)
        with self.assertRaises(ValueError, msg="Plane ID out of range"):
            TopologicalNetworkAddress(shell_id=0, plane_id=MAX_PLANES, sat_index=0, subnet_index=0)
        with self.assertRaises(ValueError, msg="Sat Index out of range for Sat"):
            TopologicalNetworkAddress(
                shell_id=0, plane_id=0, sat_index=MAX_SATS_PER_PLANE, subnet_index=0
            )
        with self.assertRaises(ValueError, msg="Subnet Index out of range"):
            TopologicalNetworkAddress(
                shell_id=0, plane_id=0, sat_index=0, subnet_index=MAX_ENDPOINTS_PER_SAT
            )
        # Add tests for negative values too
        with self.assertRaises(ValueError, msg="Negative Shell ID"):
            TopologicalNetworkAddress(shell_id=-1, plane_id=0, sat_index=0, subnet_index=0)

    def test_is_satellite_property(self):
        """Test the is_satellite property."""
        sat_addr = TopologicalNetworkAddress(shell_id=1, plane_id=1, sat_index=1, subnet_index=0)
        gs_addr = TopologicalNetworkAddress(shell_id=1, plane_id=1, sat_index=1, subnet_index=1)
        self.assertTrue(sat_addr.is_satellite)
        self.assertFalse(gs_addr.is_satellite)

    def test_is_ground_station_property(self):
        """Test the is_ground_station property."""
        sat_addr = TopologicalNetworkAddress(shell_id=1, plane_id=1, sat_index=1, subnet_index=0)
        gs_addr = TopologicalNetworkAddress(shell_id=1, plane_id=1, sat_index=1, subnet_index=1)
        self.assertFalse(sat_addr.is_ground_station)
        self.assertTrue(gs_addr.is_ground_station)

    def test_get_satellite_address(self):
        """Test getting the parent satellite address."""
        sat_addr = TopologicalNetworkAddress(shell_id=2, plane_id=5, sat_index=15, subnet_index=0)
        gs_addr = TopologicalNetworkAddress(shell_id=2, plane_id=5, sat_index=15, subnet_index=3)
        # Calling on a satellite address should return itself
        self.assertIs(sat_addr.get_satellite_address(), sat_addr)
        # Calling on a GS address should return the corresponding satellite address
        parent_sat_addr = gs_addr.get_satellite_address()
        self.assertEqual(parent_sat_addr.shell_id, 2)
        self.assertEqual(parent_sat_addr.plane_id, 5)
        self.assertEqual(parent_sat_addr.sat_index, 15)
        self.assertEqual(parent_sat_addr.subnet_index, 0)
        self.assertTrue(parent_sat_addr.is_satellite)

    def test_to_integer_serialization(self):
        """Test the integer serialization with known values."""
        # Use the actual constants from your address module for calculation
        from src.topology.satellite.topological_network_address import (
            SHELL_SHIFT,
            PLANE_SHIFT,
            SAT_IDX_SHIFT,
            SUBNET_IDX_MASK,
            SAT_IDX_MASK,
            PLANE_MASK,
            SHELL_MASK,
        )

        # Example 1: A satellite
        addr1 = TopologicalNetworkAddress(shell_id=1, plane_id=10, sat_index=5, subnet_index=0)
        expected1 = (
            (1 & SHELL_MASK) << SHELL_SHIFT
            | (10 & PLANE_MASK) << PLANE_SHIFT
            | (5 & SAT_IDX_MASK) << SAT_IDX_SHIFT
            | (0 & SUBNET_IDX_MASK)
        )
        self.assertEqual(addr1.to_integer(), expected1)

        # Example 2: A ground station associated with a different satellite
        addr2 = TopologicalNetworkAddress(shell_id=7, plane_id=71, sat_index=31, subnet_index=15)
        expected2 = (
            (7 & SHELL_MASK) << SHELL_SHIFT
            | (71 & PLANE_MASK) << PLANE_SHIFT
            | (31 & SAT_IDX_MASK) << SAT_IDX_SHIFT
            | (15 & SUBNET_IDX_MASK)
        )
        self.assertEqual(addr2.to_integer(), expected2)

        # Example 3: Edge case with zeros
        addr3 = TopologicalNetworkAddress(shell_id=0, plane_id=0, sat_index=0, subnet_index=0)
        expected3 = 0
        self.assertEqual(addr3.to_integer(), expected3)

        # Example 4: Edge case with max values (use actual MAX constants - 1)
        addr4 = TopologicalNetworkAddress(
            shell_id=MAX_SHELLS - 1,
            plane_id=MAX_PLANES - 1,
            sat_index=MAX_SATS_PER_PLANE - 1,
            subnet_index=MAX_ENDPOINTS_PER_SAT - 1,
        )
        expected4 = (
            ((MAX_SHELLS - 1) & SHELL_MASK) << SHELL_SHIFT
            | ((MAX_PLANES - 1) & PLANE_MASK) << PLANE_SHIFT
            | ((MAX_SATS_PER_PLANE - 1) & SAT_IDX_MASK) << SAT_IDX_SHIFT
            | ((MAX_ENDPOINTS_PER_SAT - 1) & SUBNET_IDX_MASK)
        )
        self.assertEqual(addr4.to_integer(), expected4)

    def test_from_integer_deserialization(self):
        """Test deserialization and round trip."""
        test_cases = [
            TopologicalNetworkAddress(shell_id=1, plane_id=10, sat_index=5, subnet_index=0),
            TopologicalNetworkAddress(shell_id=7, plane_id=71, sat_index=31, subnet_index=15),
            TopologicalNetworkAddress(shell_id=0, plane_id=0, sat_index=0, subnet_index=0),
            TopologicalNetworkAddress(
                shell_id=MAX_SHELLS - 1,
                plane_id=MAX_PLANES - 1,
                sat_index=MAX_SATS_PER_PLANE - 1,
                subnet_index=MAX_ENDPOINTS_PER_SAT - 1,
            ),
        ]
        for original_addr in test_cases:
            packed_int = original_addr.to_integer()
            reconstructed_addr = TopologicalNetworkAddress.from_integer(packed_int)
            # Check if the reconstructed object is equal to the original
            self.assertEqual(reconstructed_addr, original_addr)
            # Optionally check individual attributes too
            self.assertEqual(reconstructed_addr.shell_id, original_addr.shell_id)
            self.assertEqual(reconstructed_addr.plane_id, original_addr.plane_id)
            self.assertEqual(reconstructed_addr.sat_index, original_addr.sat_index)
            self.assertEqual(reconstructed_addr.subnet_index, original_addr.subnet_index)

    def test_from_integer_invalid(self):
        """Test ValueError for invalid integer inputs."""
        with self.assertRaises(ValueError):
            TopologicalNetworkAddress.from_integer(-1)  # Negative
        # Add test for integer too large if TOTAL_BITS < 64? Maybe less critical.
        # integer_too_large = 1 << TOTAL_BITS
        # Add test if from_integer validates decoded values? (Depends on implementation)

    def test_equality_and_hashing(self):
        """Test equality and hashability (for use in dicts/sets)."""
        addr1a = TopologicalNetworkAddress(shell_id=1, plane_id=10, sat_index=5, subnet_index=0)
        addr1b = TopologicalNetworkAddress(shell_id=1, plane_id=10, sat_index=5, subnet_index=0)
        addr2 = TopologicalNetworkAddress(
            shell_id=1, plane_id=10, sat_index=5, subnet_index=1
        )  # Diff subnet
        addr3 = TopologicalNetworkAddress(
            shell_id=1, plane_id=10, sat_index=6, subnet_index=0
        )  # Diff sat_idx

        self.assertEqual(addr1a, addr1b)
        self.assertNotEqual(addr1a, addr2)
        self.assertNotEqual(addr1a, addr3)

        # Test hashing and use in a set/dict
        addr_set = {addr1a, addr1b, addr2, addr3}
        self.assertEqual(len(addr_set), 3)  # Should contain addr1a (or 1b), addr2, addr3
        self.assertIn(addr1a, addr_set)
        self.assertIn(addr2, addr_set)
        self.assertIn(addr3, addr_set)

        addr_dict = {addr1a: "data1", addr2: "data2"}
        self.assertEqual(addr_dict[addr1b], "data1")  # Should find using equal object

    def test_string_representation(self):
        """Test the __str__ method."""
        sat_addr = TopologicalNetworkAddress(shell_id=1, plane_id=10, sat_index=5, subnet_index=0)
        gs_addr = TopologicalNetworkAddress(shell_id=2, plane_id=20, sat_index=15, subnet_index=7)
        self.assertEqual(str(sat_addr), "TopoAddr(sh:1, o:10, s:5, x:Sat)")
        self.assertEqual(str(gs_addr), "TopoAddr(sh:2, o:20, s:15, x:GS[7])")
