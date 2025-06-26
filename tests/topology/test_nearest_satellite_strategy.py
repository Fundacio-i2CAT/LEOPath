import unittest
from unittest.mock import MagicMock, patch

from astropy.time import Time

from src.network_state.gsl_attachment.gsl_attachment_strategies.nearest_satellite import (
    NearestSatelliteStrategy,
)
from src.topology.topology import LEOTopology, GroundStation, Satellite, ConstellationData


class TestNearestSatelliteStrategy(unittest.TestCase):
    """Test the NearestSatelliteStrategy implementation."""

    def setUp(self):
        """Set up test fixtures."""
        # Create test time
        self.test_time = Time("2024-05-19 12:00:00")

        # Create mock constellation data
        self.mock_constellation_data = MagicMock(spec=ConstellationData)
        self.mock_constellation_data.epoch = "2024/01/01 00:00:00.000"
        self.mock_constellation_data.max_gsl_length_m = 1000000.0  # 1000 km

        # Create mock ground stations
        self.mock_ground_stations = []
        for i in range(3):
            mock_gs = MagicMock(spec=GroundStation)
            mock_gs.id = i
            self.mock_ground_stations.append(mock_gs)

        # Create mock satellites
        self.mock_satellites = []
        for i in range(5):
            mock_sat = MagicMock(spec=Satellite)
            mock_sat.id = i
            self.mock_satellites.append(mock_sat)

        # Create mock topology
        self.mock_topology = MagicMock(spec=LEOTopology)
        self.mock_topology.constellation_data = self.mock_constellation_data
        self.mock_topology.get_satellites.return_value = self.mock_satellites

        # Create strategy instance
        self.strategy = NearestSatelliteStrategy()

    def test_strategy_name(self):
        """Test that the strategy returns the correct name."""
        self.assertEqual(self.strategy.name(), "nearest_satellite")

    @patch("src.topology.distance_tools.distance_m_ground_station_to_satellite")
    def test_select_attachments_all_visible(self, mock_distance_func):
        """Test selecting attachments when all satellites are visible."""
        # Mock distance function to return distances in order: 500km, 800km, 300km, 1200km, 600km
        distance_values = [500000.0, 800000.0, 300000.0, 1200000.0, 600000.0]

        def distance_side_effect(gs, sat, epoch, time_str):
            # Return different distances based on satellite ID
            return distance_values[sat.id]

        mock_distance_func.side_effect = distance_side_effect

        # Test with one ground station
        result = self.strategy.select_attachments(
            self.mock_topology, [self.mock_ground_stations[0]], self.test_time
        )

        # Should return the nearest satellite (ID 2 with 300km distance)
        self.assertEqual(len(result), 1)
        distance, sat_id = result[0]
        self.assertEqual(distance, 300000.0)
        self.assertEqual(sat_id, 2)

        # Verify distance function was called for all satellites
        self.assertEqual(mock_distance_func.call_count, 5)

    @patch("src.topology.distance_tools.distance_m_ground_station_to_satellite")
    def test_select_attachments_some_out_of_range(self, mock_distance_func):
        """Test selecting attachments when some satellites are out of range."""
        # Mock distances: some within range, some beyond max_gsl_length_m (1000km)
        distance_values = [500000.0, 1500000.0, 300000.0, 2000000.0, 800000.0]

        def distance_side_effect(gs, sat, epoch, time_str):
            return distance_values[sat.id]

        mock_distance_func.side_effect = distance_side_effect

        result = self.strategy.select_attachments(
            self.mock_topology, [self.mock_ground_stations[0]], self.test_time
        )

        # Should return the nearest in-range satellite (ID 2 with 300km)
        # Satellites 1 and 3 are out of range (>1000km)
        self.assertEqual(len(result), 1)
        distance, sat_id = result[0]
        self.assertEqual(distance, 300000.0)
        self.assertEqual(sat_id, 2)

    @patch("src.topology.distance_tools.distance_m_ground_station_to_satellite")
    def test_select_attachments_no_satellites_in_range(self, mock_distance_func):
        """Test behavior when no satellites are in range."""
        # All satellites are beyond max range
        mock_distance_func.return_value = 2000000.0  # 2000km > 1000km max

        result = self.strategy.select_attachments(
            self.mock_topology, [self.mock_ground_stations[0]], self.test_time
        )

        # Should return (-1.0, -1) indicating no attachment
        self.assertEqual(len(result), 1)
        distance, sat_id = result[0]
        self.assertEqual(distance, -1.0)
        self.assertEqual(sat_id, -1)

    @patch("src.topology.distance_tools.distance_m_ground_station_to_satellite")
    def test_select_attachments_multiple_ground_stations(self, mock_distance_func):
        """Test selecting attachments for multiple ground stations."""

        # Different distances for each GS-satellite pair
        def distance_side_effect(gs, sat, epoch, time_str):
            # GS 0: nearest is sat 1 (400km)
            # GS 1: nearest is sat 3 (350km)
            # GS 2: nearest is sat 0 (450km)
            distances = {
                (0, 0): 600000.0,
                (0, 1): 400000.0,  # nearest for GS 0
                (0, 2): 800000.0,
                (0, 3): 700000.0,
                (0, 4): 900000.0,
                (1, 0): 750000.0,
                (1, 1): 850000.0,
                (1, 2): 650000.0,
                (1, 3): 350000.0,  # nearest for GS 1
                (1, 4): 950000.0,
                (2, 0): 450000.0,  # nearest for GS 2
                (2, 1): 850000.0,
                (2, 2): 750000.0,
                (2, 3): 950000.0,
                (2, 4): 650000.0,
            }
            return distances.get((gs.id, sat.id), 1000000.0)

        mock_distance_func.side_effect = distance_side_effect

        result = self.strategy.select_attachments(
            self.mock_topology, self.mock_ground_stations, self.test_time  # All 3 ground stations
        )

        # Verify correct attachments for each GS
        self.assertEqual(len(result), 3)

        # GS 0 -> Sat 1 (400km)
        distance, sat_id = result[0]
        self.assertEqual(distance, 400000.0)
        self.assertEqual(sat_id, 1)

        # GS 1 -> Sat 3 (350km)
        distance, sat_id = result[1]
        self.assertEqual(distance, 350000.0)
        self.assertEqual(sat_id, 3)

        # GS 2 -> Sat 0 (450km)
        distance, sat_id = result[2]
        self.assertEqual(distance, 450000.0)
        self.assertEqual(sat_id, 0)

    @patch("src.topology.distance_tools.distance_m_ground_station_to_satellite")
    def test_select_attachments_distance_calculation_error(self, mock_distance_func):
        """Test handling of distance calculation errors."""

        # Mock an exception for some satellites
        def distance_side_effect(gs, sat, epoch, time_str):
            if sat.id == 1:
                raise Exception("Distance calculation failed")
            return 500000.0 if sat.id == 0 else 800000.0

        mock_distance_func.side_effect = distance_side_effect

        result = self.strategy.select_attachments(
            self.mock_topology, [self.mock_ground_stations[0]], self.test_time
        )

        # Should still return the nearest valid satellite (ID 0)
        self.assertEqual(len(result), 1)
        distance, sat_id = result[0]
        self.assertEqual(distance, 500000.0)
        self.assertEqual(sat_id, 0)

    @patch("src.topology.distance_tools.distance_m_ground_station_to_satellite")
    def test_select_attachments_all_distance_errors(self, mock_distance_func):
        """Test behavior when all distance calculations fail."""
        # All distance calculations raise exceptions
        mock_distance_func.side_effect = Exception("Distance calculation failed")

        result = self.strategy.select_attachments(
            self.mock_topology, [self.mock_ground_stations[0]], self.test_time
        )

        # Should return (-1.0, -1) since no valid distances could be calculated
        self.assertEqual(len(result), 1)
        distance, sat_id = result[0]
        self.assertEqual(distance, -1.0)
        self.assertEqual(sat_id, -1)

    def test_select_attachments_empty_ground_stations(self):
        """Test behavior with empty ground stations list."""
        result = self.strategy.select_attachments(self.mock_topology, [], self.test_time)

        # Should return empty list
        self.assertEqual(len(result), 0)

    def test_select_attachments_no_satellites(self):
        """Test behavior when topology has no satellites."""
        self.mock_topology.get_satellites.return_value = []

        result = self.strategy.select_attachments(
            self.mock_topology, [self.mock_ground_stations[0]], self.test_time
        )

        # Should return (-1.0, -1) since no satellites are available
        self.assertEqual(len(result), 1)
        distance, sat_id = result[0]
        self.assertEqual(distance, -1.0)
        self.assertEqual(sat_id, -1)


if __name__ == "__main__":
    unittest.main()
