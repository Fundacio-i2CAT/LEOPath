import unittest
from unittest.mock import MagicMock, patch

from astropy.time import Time

from src.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from src.network_state.gsl_attachment.gsl_attachment_strategies.nearest_satellite import NearestSatelliteStrategy
from src.topology.topology import LEOTopology, GroundStation, Satellite, ConstellationData


class TestGSLAttachmentIntegration(unittest.TestCase):
    """Integration tests for the complete GSL attachment system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Clear factory registry
        GSLAttachmentFactory._strategies.clear()
        
        # Register the real strategy
        GSLAttachmentFactory.register_strategy(NearestSatelliteStrategy)
        
        # Create test time
        self.test_time = Time("2024-05-19 12:00:00")
        
        # Create mock constellation data
        self.mock_constellation_data = MagicMock(spec=ConstellationData)
        self.mock_constellation_data.epoch = "2024/01/01 00:00:00.000"
        self.mock_constellation_data.max_gsl_length_m = 1000000.0  # 1000 km
        
        # Create mock ground stations
        self.mock_ground_stations = []
        for i in range(2):
            mock_gs = MagicMock(spec=GroundStation)
            mock_gs.id = i
            self.mock_ground_stations.append(mock_gs)
        
        # Create mock satellites
        self.mock_satellites = []
        for i in range(3):
            mock_sat = MagicMock(spec=Satellite)
            mock_sat.id = i
            self.mock_satellites.append(mock_sat)
        
        # Create mock topology
        self.mock_topology = MagicMock(spec=LEOTopology)
        self.mock_topology.constellation_data = self.mock_constellation_data
        self.mock_topology.get_satellites.return_value = self.mock_satellites
    
    def tearDown(self):
        """Clean up after each test."""
        GSLAttachmentFactory._strategies.clear()
    
    def test_factory_and_strategy_integration(self):
        """Test that the factory correctly creates and uses the nearest satellite strategy."""
        # Get strategy from factory
        strategy = GSLAttachmentFactory.get_strategy("nearest_satellite")
        
        # Verify it's the right type
        self.assertIsInstance(strategy, NearestSatelliteStrategy)
        self.assertEqual(strategy.name(), "nearest_satellite")
    
    @patch('src.topology.distance_tools.distance_m_ground_station_to_satellite')
    def test_end_to_end_gsl_attachment(self, mock_distance_func):
        """Test the complete end-to-end GSL attachment process."""
        # Mock distance function
        def distance_side_effect(gs, sat, epoch, time_str):
            # GS 0: nearest is sat 1 (300km)
            # GS 1: nearest is sat 2 (400km)
            distances = {
                (0, 0): 800000.0,
                (0, 1): 300000.0,  # nearest for GS 0
                (0, 2): 600000.0,
                (1, 0): 700000.0,
                (1, 1): 500000.0,
                (1, 2): 400000.0,  # nearest for GS 1
            }
            return distances.get((gs.id, sat.id), 1000000.0)
        
        mock_distance_func.side_effect = distance_side_effect
        
        # Get strategy from factory
        strategy = GSLAttachmentFactory.get_strategy("nearest_satellite")
        
        # Use strategy to select attachments
        result = strategy.select_attachments(
            self.mock_topology,
            self.mock_ground_stations,
            self.test_time
        )
        
        # Verify correct attachments
        self.assertEqual(len(result), 2)
        
        # GS 0 -> Sat 1 (300km)
        distance, sat_id = result[0]
        self.assertEqual(distance, 300000.0)
        self.assertEqual(sat_id, 1)
        
        # GS 1 -> Sat 2 (400km)
        distance, sat_id = result[1]
        self.assertEqual(distance, 400000.0)
        self.assertEqual(sat_id, 2)
        
        # Verify distance function was called for all GS-satellite pairs
        expected_calls = 6  # 2 ground stations * 3 satellites
        self.assertEqual(mock_distance_func.call_count, expected_calls)
    
    def test_factory_strategy_isolation(self):
        """Test that multiple strategy instances are isolated."""
        # Get two strategy instances
        strategy1 = GSLAttachmentFactory.get_strategy("nearest_satellite")
        strategy2 = GSLAttachmentFactory.get_strategy("nearest_satellite")
        
        # They should be different instances (factory creates new instances)
        self.assertIsNot(strategy1, strategy2)
        
        # But same type and name
        self.assertIsInstance(strategy1, NearestSatelliteStrategy)
        self.assertIsInstance(strategy2, NearestSatelliteStrategy)
        self.assertEqual(strategy1.name(), strategy2.name())
    
    def test_available_strategies_list(self):
        """Test that registered strategies are correctly listed."""
        strategies = GSLAttachmentFactory.list_strategies()
        
        # Should contain our registered strategy
        self.assertEqual(len(strategies), 1)
        self.assertIn("nearest_satellite", strategies)
    
    @patch('src.topology.distance_tools.distance_m_ground_station_to_satellite')
    def test_robust_error_handling_integration(self, mock_distance_func):
        """Test that the system handles errors gracefully in integration."""
        # Mock mixed success/failure scenarios
        def distance_side_effect(gs, sat, epoch, time_str):
            if gs.id == 0 and sat.id == 1:
                return 300000.0  # Valid attachment for GS 0
            elif gs.id == 1:
                raise Exception("Satellite tracking lost")  # All satellites fail for GS 1
            else:
                return 2000000.0  # Out of range
        
        mock_distance_func.side_effect = distance_side_effect
        
        # Get strategy and use it
        strategy = GSLAttachmentFactory.get_strategy("nearest_satellite")
        result = strategy.select_attachments(
            self.mock_topology,
            self.mock_ground_stations,
            self.test_time
        )
        
        # Verify mixed results
        self.assertEqual(len(result), 2)
        
        # GS 0 should have a valid attachment
        distance, sat_id = result[0]
        self.assertEqual(distance, 300000.0)
        self.assertEqual(sat_id, 1)
        
        # GS 1 should have no attachment due to errors
        distance, sat_id = result[1]
        self.assertEqual(distance, -1.0)
        self.assertEqual(sat_id, -1)


if __name__ == "__main__":
    unittest.main()
