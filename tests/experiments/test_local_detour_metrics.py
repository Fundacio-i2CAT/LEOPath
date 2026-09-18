import networkx as nx

from leopath.experiments.metrics import compute_path_stretch, normalize_next_hop
from leopath.network_state.routing_algorithms.topological_routing.fstate_calculation import (
    LOCAL_DETOUR,
)

GS_SRC = 100
GS_DST = 101


def _stretch(fstate: dict, graph: nx.Graph, satellite_ids: list[int]) -> dict:
    return compute_path_stretch(
        fstate=fstate,
        topology_graph=graph,
        satellite_ids=satellite_ids,
        ground_station_ids=[GS_SRC, GS_DST],
        attachments=[(0, 10.0), (1, 20.0)],
        interface_neighbor_map={},
        max_hops=len(satellite_ids) + 2,
        ground_station_satellites_in_range=[[(10.0, 0)], [(20.0, 1)]],
    )


def test_detour_entry_is_followed_over_its_three_links() -> None:
    # Link 0-1 has failed; satellite 0 reaches 1 around the square 0-2-3-1.
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0)])
    fstate = {(0, GS_DST): (LOCAL_DETOUR, 2, 3, 1), (1, GS_DST): (GS_DST, 0, 0)}
    stats = _stretch(fstate, graph, [0, 1, 2, 3])

    assert stats["delivery"]["delivered"] == 1.0
    assert stats["hop_shared"]["mean"] == 1.0
    assert stats["distance_shared"]["mean"] == 1.0


def test_detour_with_a_failed_leg_is_link_down() -> None:
    # The detour's middle link 2-3 is also down; 0-4-1 keeps the pair deliverable.
    graph = nx.Graph()
    graph.add_weighted_edges_from([(0, 2, 1.0), (3, 1, 1.0), (0, 4, 1.0), (4, 1, 1.0)])
    fstate = {(0, GS_DST): (LOCAL_DETOUR, 2, 3, 1), (1, GS_DST): (GS_DST, 0, 0)}
    delivery = _stretch(fstate, graph, [0, 1, 2, 3, 4])["delivery"]

    assert delivery["failure_link_down"] == 1.0


def test_detour_names_its_target_as_next_hop() -> None:
    assert normalize_next_hop((LOCAL_DETOUR, 2, 3, 1), 0, {}) == 1
