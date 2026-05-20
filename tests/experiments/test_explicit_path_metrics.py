import math

import networkx as nx

from leopath.experiments.metrics import (
    compute_forwarding_state_stats,
    compute_gs_renumbering_stats,
    compute_path_stretch,
    compute_sat_to_gs_churn,
)


def test_explicit_path_route_plan_gives_unit_stretch() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
        ]
    )

    stats = compute_path_stretch(
        fstate={},
        topology_graph=graph,
        satellite_ids=[0, 1, 2],
        ground_station_ids=[100, 101],
        attachments=[(0, 10.0), (2, 20.0)],
        interface_neighbor_map={},
        max_hops=10,
        route_plans={(0, 101): {"satellite_path": [0, 1, 2], "adjacency_sid_list": [1, 2], "planned_dst_sat_id": 2}},
        ground_station_satellites_in_range=[[(10.0, 0)], [(20.0, 2)]],
    )

    assert stats["hop"]["count"] == 1
    assert math.isclose(stats["hop"]["mean"], 1.0)
    assert math.isclose(stats["distance"]["mean"], 1.0)


def test_explicit_path_churn_uses_first_hop_from_route_plan() -> None:
    churn = compute_sat_to_gs_churn(
        prev_fstate={},
        curr_fstate={},
        satellite_ids=[0],
        ground_station_ids=[101],
        interface_neighbor_map={},
        prev_route_plans={(0, 101): {"satellite_path": [0, 1, 2], "adjacency_sid_list": [1, 2], "planned_dst_sat_id": 2}},
        curr_route_plans={(0, 101): {"satellite_path": [0, 3, 2], "adjacency_sid_list": [3, 2], "planned_dst_sat_id": 2}},
    )

    assert math.isclose(churn["churn"], 1.0)
    assert math.isclose(churn["break_rate"], 0.0)


def test_gs_renumbering_stats_match_attachment_changes() -> None:
    stats = compute_gs_renumbering_stats(
        prev_attachments=[(1, 10.0), (2, 20.0), (None, float("inf"))],
        curr_attachments=[(1, 11.0), (3, 18.0), (4, 15.0)],
    )

    assert math.isclose(stats["count"], 2.0)
    assert math.isclose(stats["rate"], 2.0 / 3.0)


def test_explicit_path_state_counts_local_neighbors_only() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 2), (0, 4), (1, 3)])

    stats = compute_forwarding_state_stats(
        fstate={},
        topology_graph=graph,
        algorithm_name="explicit_path_routing",
        satellite_ids=[0, 1],
        ground_station_ids=[100, 101],
        route_plans={
            (0, 100): {"satellite_path": [0, 2, 4]},
            (0, 101): {"satellite_path": [0]},
            (1, 100): {"satellite_path": [1, 3]},
            (1, 101): {},
        },
    )

    assert math.isclose(stats["min"], 1.0)
    assert math.isclose(stats["max"], 2.0)
    assert math.isclose(stats["mean"], 1.5)
