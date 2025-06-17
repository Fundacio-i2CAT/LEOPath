import unittest
from unittest.mock import patch, MagicMock

from src.topology.isl_builder import setup_isls_in_the_same_orbit, generate_plus_grid_isls


class TestISLBuilder(unittest.TestCase):
    def test_setup_isls_in_the_same_orbit_simple(self):
        # Test with 1 orbit, 3 satellites
        result = setup_isls_in_the_same_orbit(1, 3)
        expected = [(0, 1), (1, 2), (2, 0)]  # Connections in a ring
        self.assertEqual(result, expected)

    def test_setup_isls_in_the_same_orbit_multiple_orbits(self):
        # Test with 2 orbits, 4 satellites per orbit
        result = setup_isls_in_the_same_orbit(2, 4)
        expected = [
            # First orbit
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            # Second orbit
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
        ]
        self.assertEqual(result, expected)

    def test_setup_isls_length(self):
        # Each orbit has n connections, with n satellites
        orbits = 5
        sats_per_orbit = 6
        result = setup_isls_in_the_same_orbit(orbits, sats_per_orbit)
        # Should have exactly the same number of connections as satellites
        self.assertEqual(len(result), orbits * sats_per_orbit)

    @patch("src.topology.isl_builder.log")
    def test_generate_plus_grid_isls_min_size(self, mock_log):
        # Test with minimum size (3x3)
        result = generate_plus_grid_isls(3, 3)
        # Should have:
        # - 9 intra-orbit connections (3 orbits × 3 satellites)
        # - 9 inter-orbit connections (3 orbits × 3 satellites)
        self.assertEqual(len(result), 18)

        # Check that intra-orbit connections are correct
        intra_orbit_expected = [
            (0, 1),
            (1, 2),
            (2, 0),  # Orbit 0
            (3, 4),
            (4, 5),
            (5, 3),  # Orbit 1
            (6, 7),
            (7, 8),
            (8, 6),  # Orbit 2
        ]
        for conn in intra_orbit_expected:
            self.assertIn(conn, result)

        # Check inter-orbit connections (assuming isl_shift=0)
        inter_orbit_expected = [
            # Orbit 0 to Orbit 1
            (0, 3),
            (1, 4),
            (2, 5),
            # Orbit 1 to Orbit 2
            (3, 6),
            (4, 7),
            (5, 8),
            # Orbit 2 to Orbit 0 (wrap around)
            (6, 0),
            (7, 1),
            (8, 2),
        ]
        for conn in inter_orbit_expected:
            self.assertIn(conn, result)

    def test_generate_plus_grid_isls_with_shift(self):
        # Test with isl_shift=1
        result = generate_plus_grid_isls(3, 3, isl_shift=1)

        # Inter-orbit connections with shift=1
        # Each satellite connects to the one shifted by 1 in the next orbit
        inter_orbit_expected = [
            # Orbit 0 to Orbit 1
            (0, 4),
            (1, 5),
            (2, 3),
            # Orbit 1 to Orbit 2
            (3, 7),
            (4, 8),
            (5, 6),
            # Orbit 2 to Orbit 0 (wrap around)
            (6, 1),
            (7, 2),
            (8, 0),
        ]
        for conn in inter_orbit_expected:
            self.assertIn(conn, result)

    def test_generate_plus_grid_isls_with_offset(self):
        # Test with idx_offset=10
        result = generate_plus_grid_isls(3, 3, idx_offset=10)

        # Check that all satellite IDs are offset by 10
        for src, dst in result:
            self.assertGreaterEqual(src, 10)
            self.assertGreaterEqual(dst, 10)
            self.assertLess(src, 19)  # 10 + 3*3 = 19
            self.assertLess(dst, 19)

    def test_generate_plus_grid_isls_invalid_size(self):
        # Test with too small configuration
        result1 = generate_plus_grid_isls(2, 3)  # Too few orbits
        self.assertEqual(result1, [])

        result2 = generate_plus_grid_isls(3, 2)  # Too few satellites per orbit
        self.assertEqual(result2, [])
