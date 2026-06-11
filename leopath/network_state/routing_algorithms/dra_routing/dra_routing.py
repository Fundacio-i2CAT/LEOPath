"""DRA-style logical-coordinate routing baseline.

Family-level implementation of the Datagram Routing Algorithm (DRA) of
Ekici, Akyildiz, and Bender (IEEE/ACM ToN 2001): each satellite is
identified by virtual (plane, slot) coordinates and forwards packets to
the neighbor that minimizes the remaining minimum-hop distance on the
logical grid, using purely local decisions and no physical link weights.

This module reuses the topological-routing machinery with the distance
mode pinned to the unit-torus (hop-count) metric. The resulting behavior
matches the classical DRA family at the abstraction level of this
simulator: hop-optimal direction estimation over logical coordinates,
blind to the latitude-dependent physical length of inter-plane ISLs.
It is the closest prior-art baseline to the proposed pivot-weighted
topological forwarding, which differs only in the distance function.
"""

from leopath.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from leopath.network_state.routing_algorithms.topological_routing.algorithm_topological_routing import (
    _calculate_bandwidth_state,
    algorithm_topological_routing,
)
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

#: Distance mode implementing pure hop-count forwarding on the logical torus.
DRA_DISTANCE_MODE = "torus_unit"


class DRARoutingAlgorithm(RoutingAlgorithm):
    """
    DRA-style baseline: greedy minimum-hop forwarding on logical
    (plane, slot) coordinates, ignoring physical ISL weights.
    """

    def __init__(self) -> None:
        self._cached_bandwidth_signature: tuple | None = None
        self._cached_bandwidth_state: dict | None = None

    def compute_state(
        self,
        time_since_epoch_ns: int,
        constellation_data: ConstellationData,
        ground_stations: list[GroundStation],
        topology_with_isls: LEOTopology,
        ground_station_satellites_in_range: list,
        list_gsl_interfaces_info: list,
        algorithm_params: dict | None = None,
    ) -> dict:
        bandwidth_signature = tuple(
            (
                node_info.get("id", index),
                node_info.get("aggregate_max_bandwidth", 0.0),
            )
            for index, node_info in enumerate(list_gsl_interfaces_info)
        )
        if bandwidth_signature != self._cached_bandwidth_signature:
            self._cached_bandwidth_state = _calculate_bandwidth_state(
                constellation_data,
                ground_stations,
                list_gsl_interfaces_info,
            )
            self._cached_bandwidth_signature = bandwidth_signature

        # The hop-only metric defines the DRA family; any externally
        # supplied distance_mode is overridden so runs cannot silently
        # degenerate into the proposed weighted variant.
        params = dict(algorithm_params or {})
        params["distance_mode"] = DRA_DISTANCE_MODE

        return algorithm_topological_routing(
            time_since_epoch_ns,
            constellation_data,
            ground_stations,
            topology_with_isls,
            ground_station_satellites_in_range,
            self._cached_bandwidth_state or {},
            prev_fstate=None,
            graph_has_changed=True,
            algorithm_params=params,
        )
