#!/usr/bin/env python3
"""
Simple test script to verify topological routing algorithm works correctly.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.network_state.routing_algorithms.routing_algorithm_factory import get_routing_algorithm


def test_topological_routing_factory():
    """Test that we can get the topological routing algorithm from the factory."""
    print("Testing topological routing algorithm factory...")

    try:
        # Test shortest path algorithm (baseline)
        shortest_path_algo = get_routing_algorithm("shortest_path_link_state")
        print(
            f"✓ Successfully created shortest path algorithm: {type(shortest_path_algo).__name__}"
        )

        # Test topological routing algorithm
        topological_algo = get_routing_algorithm("topological_routing")
        print(
            f"✓ Successfully created topological routing algorithm: {type(topological_algo).__name__}"
        )

        print("✓ Factory test passed!")
        return True

    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        return False


def test_topological_address():
    """Test the TopologicalNetworkAddress functionality."""
    print("\nTesting TopologicalNetworkAddress...")

    try:
        from src.topology.satellite.topological_network_address import TopologicalNetworkAddress

        # Test creating address from satellite ID (single shell assumption)
        for sat_id in [0, 1, 50, 100]:
            addr = TopologicalNetworkAddress.set_address_from_orbital_parameters(sat_id)
            print(f"✓ Satellite {sat_id} -> Address: {addr}")

            # Test round-trip conversion
            integer_repr = addr.to_integer()
            addr_reconstructed = TopologicalNetworkAddress.from_integer(integer_repr)

            if addr == addr_reconstructed:
                print(f"  ✓ Round-trip conversion successful: {integer_repr}")
            else:
                print(f"  ✗ Round-trip conversion failed: {addr} != {addr_reconstructed}")
                return False

        print("✓ TopologicalNetworkAddress test passed!")
        return True

    except Exception as e:
        print(f"✗ TopologicalNetworkAddress test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running topological routing algorithm tests...\n")

    success = True
    success &= test_topological_routing_factory()
    success &= test_topological_address()

    if success:
        print("\n🎉 All tests passed! Topological routing algorithm is ready.")
        print("\nYou can now use it in simulations with:")
        print("  routing_algorithm: topological_routing")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()
