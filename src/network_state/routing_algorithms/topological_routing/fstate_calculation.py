import networkx as nx
import numpy as np

from src import logger
from src.topology.topology import GroundStation, LEOTopology

log = logger.get_logger(__name__)


def calculate_fstate_shortest_path_object_no_gs_relay(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
) -> dict:
    """
    Calculates forwarding state using shortest paths over ISLs only (no GS relays).
    """
    log.debug("Calculating shortest path fstate object (no GS relay)")

    full_graph = topology_with_isls.graph
    sat_neighbor_to_if = topology_with_isls.sat_neighbor_to_if
