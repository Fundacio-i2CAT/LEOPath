from .shortest_path_link_state_routing.shortest_path_link_state_routing import (
    ShortestPathLinkStateRoutingAlgorithm,
)


def get_routing_algorithm(name: str):
    """
    Factory for routing algorithms.
    """
    if name == "shortest_path_link_state":
        return ShortestPathLinkStateRoutingAlgorithm()
    else:
        raise ValueError(f"Unknown routing algorithm: {name}")
