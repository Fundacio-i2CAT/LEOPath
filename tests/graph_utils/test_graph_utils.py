# In tests/dynamic_state/test_generate_dynamic_state.py
import unittest
from unittest.mock import MagicMock  # Or use a simple class with a .graph attribute

import networkx as nx

from src.network_state.utils import graph as graph_utils


class TestGraphUtils(unittest.TestCase):

    def test_topologies_are_equal_first_step(self):
        """Test returns False when prev_topo is None."""
        mock_curr_topo = MagicMock()
        mock_curr_topo.graph = nx.Graph()  # Doesn't matter what graph is
        self.assertFalse(graph_utils._topologies_are_equal(None, mock_curr_topo))

    def test_topologies_are_equal_identical_graphs(self):
        """Test returns True for identical nodes, edges, and weights."""
        graph1 = nx.Graph()
        graph1.add_edge(1, 2, weight=100.5)
        graph1.add_edge(2, 3, weight=200.0)
        graph1.add_node(4)  # Isolated node

        graph2 = nx.Graph()
        graph2.add_edge(1, 2, weight=100.5)
        graph2.add_edge(2, 3, weight=200.0)
        graph2.add_node(4)

        mock_prev_topo = MagicMock(graph=graph1)
        mock_curr_topo = MagicMock(graph=graph2)

        self.assertTrue(graph_utils._topologies_are_equal(mock_prev_topo, mock_curr_topo))

    def test_topologies_are_equal_nodes_differ(self):
        """Test returns False if node sets differ."""
        graph1 = nx.Graph()
        graph1.add_nodes_from([1, 2, 3])
        graph2 = nx.Graph()
        graph2.add_nodes_from([1, 2, 4])  # Node 3 replaced with 4

        mock_prev_topo = MagicMock(graph=graph1)
        mock_curr_topo = MagicMock(graph=graph2)

        self.assertFalse(graph_utils._topologies_are_equal(mock_prev_topo, mock_curr_topo))

    def test_topologies_are_equal_edges_differ(self):
        """Test returns False if edge sets differ."""
        graph1 = nx.Graph()
        graph1.add_edge(1, 2, weight=100.0)
        graph1.add_edge(2, 3, weight=200.0)  # Edge 2-3 present

        graph2 = nx.Graph()
        graph2.add_edge(1, 2, weight=100.0)
        graph2.add_edge(1, 3, weight=200.0)  # Edge 1-3 present instead of 2-3

        mock_prev_topo = MagicMock(graph=graph1)
        mock_curr_topo = MagicMock(graph=graph2)

        self.assertFalse(graph_utils._topologies_are_equal(mock_prev_topo, mock_curr_topo))

    def test_topologies_are_equal_weights_differ_significantly(self):
        """Test returns False if weights differ beyond tolerance."""
        graph1 = nx.Graph()
        graph1.add_edge(1, 2, weight=100.0)
        graph2 = nx.Graph()
        graph2.add_edge(1, 2, weight=101.0)  # Weight differs by 1.0

        mock_prev_topo = MagicMock(graph=graph1)
        mock_curr_topo = MagicMock(graph=graph2)

        # Use a small tolerance
        self.assertFalse(
            graph_utils._topologies_are_equal(mock_prev_topo, mock_curr_topo, weight_tolerance=0.1)
        )

    def test_topologies_are_equal_weights_within_tolerance(self):
        """Test returns True if weights differ only within tolerance."""
        graph1 = nx.Graph()
        graph1.add_edge(1, 2, weight=100.00001)
        graph2 = nx.Graph()
        graph2.add_edge(1, 2, weight=100.00002)  # Very small difference

        mock_prev_topo = MagicMock(graph=graph1)
        mock_curr_topo = MagicMock(graph=graph2)

        # Use a tolerance larger than the difference
        self.assertTrue(
            graph_utils._topologies_are_equal(mock_prev_topo, mock_curr_topo, weight_tolerance=1e-4)
        )
        # Use a tolerance smaller than the difference
        self.assertFalse(
            graph_utils._topologies_are_equal(mock_prev_topo, mock_curr_topo, weight_tolerance=1e-6)
        )
