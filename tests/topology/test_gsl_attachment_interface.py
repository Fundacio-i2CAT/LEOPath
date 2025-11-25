import unittest
from typing import List, Tuple, Dict, Any
from unittest.mock import MagicMock

from astropy.time import Time

from src.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from src.network_state.gsl_attachment.gsl_attachment_interface import GSLAttachmentStrategy
from src.topology.topology import GroundStation, LEOTopology


class MockGSLAttachmentStrategy(GSLAttachmentStrategy):
    """Mock strategy for testing the interface."""

    def __init__(self, strategy_name: str = "mock_strategy"):
        self._strategy_name = strategy_name
        self._call_count = 0
        self._last_params: Dict[str, Any] = {}

    def name(self) -> str:
        return self._strategy_name

    def select_attachments(
        self, topology: LEOTopology, ground_stations: List[GroundStation], current_time: Time
    ) -> List[Tuple[float, int]]:
        self._call_count += 1
        self._last_params = {
            "topology": topology,
            "ground_stations": ground_stations,
            "current_time": current_time,
        }
        # Return mock attachments: alternating between valid and invalid
        result = []
        for i, gs in enumerate(ground_stations):
            if i % 2 == 0:
                result.append((100.0 + i * 10, i))  # Valid attachment
            else:
                result.append((-1.0, -1))  # No attachment
        return result


class TestGSLAttachmentInterface(unittest.TestCase):
    """Test the GSL attachment strategy interface and factory."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear factory registry before each test
        GSLAttachmentFactory._strategies.clear()

        # Create test time
        self.test_time = Time("2024-05-19 12:00:00")

    def tearDown(self):
        """Clean up after each test."""
        GSLAttachmentFactory._strategies.clear()

    def test_strategy_interface_abstract_methods(self):
        """Test that the GSLAttachmentStrategy interface defines abstract methods."""
        # Attempting to instantiate the abstract class should raise TypeError
        with self.assertRaises(TypeError):
            GSLAttachmentStrategy()  # type: ignore

    def test_mock_strategy_implementation(self):
        """Test that our mock strategy correctly implements the interface."""
        strategy = MockGSLAttachmentStrategy("test_strategy")

        # Test name method
        self.assertEqual(strategy.name(), "test_strategy")

        # Create simple mock objects
        mock_topology = MagicMock()
        mock_ground_stations = [MagicMock(), MagicMock(), MagicMock()]

        # Test select_attachments method
        result = strategy.select_attachments(
            mock_topology, mock_ground_stations, self.test_time  # type: ignore
        )

        # Verify return type and structure
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), len(mock_ground_stations))

        # Verify each element is a tuple with (distance, satellite_id)
        for i, attachment in enumerate(result):
            self.assertIsInstance(attachment, tuple)
            self.assertEqual(len(attachment), 2)
            distance, sat_id = attachment
            self.assertIsInstance(distance, float)
            self.assertIsInstance(sat_id, int)

            # Verify mock strategy behavior
            if i % 2 == 0:
                self.assertEqual(distance, 100.0 + i * 10)
                self.assertEqual(sat_id, i)
            else:
                self.assertEqual(distance, -1.0)
                self.assertEqual(sat_id, -1)

        # Verify strategy was called correctly
        self.assertEqual(strategy._call_count, 1)
        self.assertEqual(strategy._last_params["topology"], mock_topology)
        self.assertEqual(strategy._last_params["ground_stations"], mock_ground_stations)
        self.assertEqual(strategy._last_params["current_time"], self.test_time)


class TestGSLAttachmentFactory(unittest.TestCase):
    """Test the GSL attachment factory."""

    def setUp(self):
        """Set up test fixtures."""
        # Clear factory registry before each test
        GSLAttachmentFactory._strategies.clear()

    def tearDown(self):
        """Clean up after each test."""
        GSLAttachmentFactory._strategies.clear()

    def test_factory_register_strategy(self):
        """Test registering a strategy with the factory."""
        # Register a mock strategy
        GSLAttachmentFactory.register_strategy(MockGSLAttachmentStrategy)

        # Verify it's registered
        strategies = GSLAttachmentFactory.list_strategies()
        self.assertIn("mock_strategy", strategies)
        self.assertEqual(len(strategies), 1)

    def test_factory_get_strategy(self):
        """Test retrieving a strategy from the factory."""
        # Register a strategy first
        GSLAttachmentFactory.register_strategy(MockGSLAttachmentStrategy)

        # Get the strategy
        strategy = GSLAttachmentFactory.get_strategy("mock_strategy")

        # Verify it's the correct type and works
        self.assertIsInstance(strategy, MockGSLAttachmentStrategy)
        self.assertEqual(strategy.name(), "mock_strategy")

    def test_factory_get_unknown_strategy(self):
        """Test that getting an unknown strategy raises ValueError."""
        with self.assertRaises(ValueError) as context:
            GSLAttachmentFactory.get_strategy("unknown_strategy")

        self.assertIn("Unknown GSL attachment strategy", str(context.exception))
        self.assertIn("unknown_strategy", str(context.exception))
        self.assertIn("Available strategies:", str(context.exception))

    def test_factory_list_strategies_empty(self):
        """Test listing strategies when none are registered."""
        strategies = GSLAttachmentFactory.list_strategies()
        self.assertEqual(strategies, [])

    def test_factory_list_strategies_multiple(self):
        """Test listing multiple registered strategies."""

        # Create multiple mock strategies
        class Strategy1(MockGSLAttachmentStrategy):
            def name(self):
                return "strategy_1"

        class Strategy2(MockGSLAttachmentStrategy):
            def name(self):
                return "strategy_2"

        # Register them
        GSLAttachmentFactory.register_strategy(Strategy1)
        GSLAttachmentFactory.register_strategy(Strategy2)

        # Verify they're all listed
        strategies = GSLAttachmentFactory.list_strategies()
        self.assertEqual(len(strategies), 2)
        self.assertIn("strategy_1", strategies)
        self.assertIn("strategy_2", strategies)

    def test_factory_register_duplicate_strategy(self):
        """Test that registering a strategy with the same name overwrites the previous one."""
        # Register original strategy
        GSLAttachmentFactory.register_strategy(MockGSLAttachmentStrategy)

        # Create a new strategy with the same name
        class NewMockStrategy(MockGSLAttachmentStrategy):
            def select_attachments(self, topology, ground_stations, current_time):
                return [(999.0, 999) for _ in ground_stations]

        # Register the new strategy (should overwrite)
        GSLAttachmentFactory.register_strategy(NewMockStrategy)

        # Verify only one strategy is registered
        strategies = GSLAttachmentFactory.list_strategies()
        self.assertEqual(len(strategies), 1)

        # Verify it's the new strategy
        strategy = GSLAttachmentFactory.get_strategy("mock_strategy")
        mock_topology = MagicMock()
        mock_gs = MagicMock()
        mock_time = Time("2024-05-19 12:00:00")
        result = strategy.select_attachments(mock_topology, [mock_gs], mock_time)  # type: ignore
        self.assertEqual(result[0], (999.0, 999))


if __name__ == "__main__":
    unittest.main()
