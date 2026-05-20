import math

import networkx as nx

from leopath.experiments.metrics import (
    compute_explicit_failover_stats,
    compute_explicit_header_stats,
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


def test_explicit_path_route_plan_uses_local_backup_repair() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            (0, 3, 1.0),
            (3, 1, 1.0),
            (1, 2, 1.0),
        ]
    )

    stats = compute_path_stretch(
        fstate={},
        topology_graph=graph,
        satellite_ids=[0, 1, 2, 3],
        ground_station_ids=[100, 101],
        attachments=[(0, 10.0), (2, 20.0)],
        interface_neighbor_map={},
        max_hops=10,
        route_plans={
            (0, 101): {
                "satellite_path": [0, 1, 2],
                "adjacency_sid_list": [1, 2],
                "backup_adjacency_sid_list": [3, None],
                "planned_dst_sat_id": 2,
            }
        },
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


def test_explicit_path_header_stats_track_strict_header_bytes() -> None:
    stats = compute_explicit_header_stats(
        {
            (0, 100): {"strict_header_bytes": 32},
            (0, 101): {"strict_header_bytes": 24},
            (1, 100): {},
        }
    )

    assert math.isclose(stats["min"], 24.0)
    assert math.isclose(stats["max"], 32.0)
    assert math.isclose(stats["mean"], 28.0)
    assert stats["count"] == 2


def test_explicit_path_failover_stats_track_repair_and_drop_causes() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            (0, 3, 1.0),
            (3, 1, 1.0),
            (1, 2, 1.0),
            (4, 5, 1.0),
        ]
    )

    stats = compute_explicit_failover_stats(
        topology_graph=graph,
        route_plans={
            (0, 100): {
                "satellite_path": [0, 1, 2],
                "adjacency_sid_list": [1, 2],
                "backup_adjacency_sid_list": [3, None],
                "planned_dst_sat_id": 2,
            },
            (4, 101): {
                "satellite_path": [4, 6],
                "adjacency_sid_list": [6],
                "backup_adjacency_sid_list": [None],
                "planned_dst_sat_id": 6,
            },
            (7, 102): {
                "satellite_path": [7],
                "adjacency_sid_list": [],
                "backup_adjacency_sid_list": [],
                "planned_dst_sat_id": 7,
            },
        },
        ground_station_ids=[100, 101, 102],
        ground_station_satellites_in_range=[[(20.0, 2)], [(15.0, 8)], []],
    )

    assert math.isclose(stats["count"], 3.0)
    assert math.isclose(stats["protected_repair_count"], 1.0)
    assert math.isclose(stats["broken_adj_count"], 1.0)
    assert math.isclose(stats["egress_not_visible_count"], 1.0)
    assert math.isclose(stats["delivered_count"], 1.0)
