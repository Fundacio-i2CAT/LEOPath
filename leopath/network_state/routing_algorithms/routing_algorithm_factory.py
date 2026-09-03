from .dra_routing.dra_routing import DRARoutingAlgorithm
from .explicit_path_routing.explicit_path_routing import ExplicitPathRoutingAlgorithm
from .shortest_path_link_state_routing.shortest_path_link_state_routing import (
    ShortestPathLinkStateRoutingAlgorithm,
)
from .topological_routing.topological_routing import TopologicalRoutingAlgorithm


def get_routing_algorithm(name: str):
    """
    Factory for routing algorithms.
    """
    if name == "dra_routing":
        return DRARoutingAlgorithm()
    elif name == "shortest_path_link_state":
        return ShortestPathLinkStateRoutingAlgorithm()
    elif name == "explicit_path_routing":
        return ExplicitPathRoutingAlgorithm()
    elif name == "topological_routing":
        return TopologicalRoutingAlgorithm()
    else:
        raise ValueError(f"Unknown routing algorithm: {name}")
