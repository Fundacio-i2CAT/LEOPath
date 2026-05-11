from .explicit_path_routing.explicit_path_routing import ExplicitPathRoutingAlgorithm
from .shortest_path_link_state_routing.shortest_path_link_state_routing import (
    ShortestPathLinkStateRoutingAlgorithm,
)
from .predictive_link_state.predictive_link_state import (
    PredictiveLinkStateRoutingAlgorithm,
)
from .traditional_segment_routing.traditional_segment_routing import (
    TraditionalSegmentRoutingAlgorithm,
)
from .topological_routing.topological_routing import TopologicalRoutingAlgorithm


def get_routing_algorithm(name: str):
    """
    Factory for routing algorithms.
    """
    if name == "shortest_path_link_state":
        return ShortestPathLinkStateRoutingAlgorithm()
    elif name == "predictive_link_state":
        return PredictiveLinkStateRoutingAlgorithm()
    elif name == "explicit_path_routing":
        return ExplicitPathRoutingAlgorithm()
    elif name == "traditional_segment_routing":
        return TraditionalSegmentRoutingAlgorithm()
    elif name == "topological_routing":
        return TopologicalRoutingAlgorithm()
    else:
        raise ValueError(f"Unknown routing algorithm: {name}")
