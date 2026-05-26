import math

import networkx as nx

from leopath.experiments.metrics import (
    compute_explicit_failover_stats,
    compute_explicit_header_stats,
    compute_forwarding_state_stats,
    compute_gs_renumbering_stats,
    compute_path_stretch,
    compute_satellite_forwarding_state_updates,
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
        route_plans={
            (0, 101): {
                "satellite_path": [0, 1, 2],
                "adjacency_sid_list": [1, 2],
                "planned_dst_sat_id": 2,
            }
        },
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


def test_dynamic_egress_route_plan_repairs_stale_final_handoff() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
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
                "satellite_path": [0, 1, 2, 3],
                "adjacency_sid_list": [1, 2, 3],
                "planned_dst_sat_id": 3,
                "final_egress_mode": "dynamic",
                "current_dst_sat_id": 2,
                "egress_repair_satellite_path": [3, 2],
            }
        },
        ground_station_satellites_in_range=[[(10.0, 0)], [(20.0, 2)]],
    )

    assert stats["hop"]["count"] == 1
    assert math.isclose(stats["hop"]["mean"], 6.0 / 4.0)
    assert math.isclose(stats["distance"]["mean"], 34.0 / 32.0)


def test_explicit_path_churn_uses_first_hop_from_route_plan() -> None:
    churn = compute_sat_to_gs_churn(
        prev_fstate={},
        curr_fstate={},
        satellite_ids=[0],
        ground_station_ids=[101],
        interface_neighbor_map={},
        prev_route_plans={
            (0, 101): {
                "satellite_path": [0, 1, 2],
                "adjacency_sid_list": [1, 2],
                "planned_dst_sat_id": 2,
            }
        },
        curr_route_plans={
            (0, 101): {
                "satellite_path": [0, 3, 2],
                "adjacency_sid_list": [3, 2],
                "planned_dst_sat_id": 2,
            }
        },
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


def test_explicit_path_state_counts_attached_ingress_bindings_plus_local_neighbors() -> None:
    graph = nx.Graph()
    graph.add_edges_from([(0, 2), (0, 4), (1, 3)])

    stats = compute_forwarding_state_stats(
        fstate={
            (0, 100): (2, 0, 0),
            (0, 101): (0, 0, 0),
            (1, 100): (3, 0, 0),
            (1, 101): (-1, -1, -1),
        },
        topology_graph=graph,
        algorithm_name="explicit_path_routing",
        satellite_ids=[0, 1],
        ground_station_ids=[100, 101, 102],
        attachments=[(0, 10.0), (0, 12.0)],
        route_plans={
            (0, 100): {"satellite_path": [0, 2, 4]},
            (0, 101): {"satellite_path": [0]},
            (0, 102): {"satellite_path": [0, 5]},
            (1, 100): {"satellite_path": [1, 3]},
            (1, 101): {},
        },
    )

    assert math.isclose(stats["min"], 1.0)
    assert math.isclose(stats["max"], 5.0)
    assert math.isclose(stats["mean"], 3.0)


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


def test_explicit_path_header_stats_can_scope_to_active_gs_traffic() -> None:
    stats = compute_explicit_header_stats(
        {
            (0, 100): {"strict_header_bytes": 1000},
            (0, 101): {"strict_header_bytes": 24},
            (1, 100): {"strict_header_bytes": 0},
            (2, 101): {"strict_header_bytes": 500},
        },
        source_attachments=[(0, 10.0), (1, 20.0)],
        ground_station_ids=[100, 101],
    )

    assert stats["count"] == 2
    assert math.isclose(stats["min"], 0.0)
    assert math.isclose(stats["max"], 24.0)
    assert math.isclose(stats["mean"], 12.0)


def test_explicit_path_header_stats_can_use_srv6_srh_bytes() -> None:
    stats = compute_explicit_header_stats(
        {
            (0, 101): {"strict_header_bytes": 24, "srv6_srh_bytes": 24},
            (1, 100): {"strict_header_bytes": 0, "srv6_srh_bytes": 0},
        },
        source_attachments=[(0, 10.0), (1, 20.0)],
        ground_station_ids=[100, 101],
        bytes_key="srv6_srh_bytes",
    )

    assert stats["count"] == 2
    assert math.isclose(stats["mean"], 12.0)


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


def test_explicit_path_failover_counts_dynamic_egress_repair() -> None:
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        [
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 3, 1.0),
        ]
    )

    stats = compute_explicit_failover_stats(
        topology_graph=graph,
        route_plans={
            (0, 100): {
                "satellite_path": [0, 1, 2, 3],
                "adjacency_sid_list": [1, 2, 3],
                "backup_adjacency_sid_list": [],
                "planned_dst_sat_id": 3,
                "final_egress_mode": "dynamic",
                "current_dst_sat_id": 2,
                "egress_repair_satellite_path": [3, 2],
            }
        },
        ground_station_ids=[100],
        ground_station_satellites_in_range=[[(20.0, 2)]],
    )

    assert math.isclose(stats["delivered_count"], 1.0)
    assert math.isclose(stats["delivered_rate"], 1.0)
    assert math.isclose(stats["dynamic_egress_repair_count"], 1.0)
    assert math.isclose(stats["egress_not_visible_count"], 0.0)


def test_topological_satellite_updates_only_track_local_gsl_binding_changes() -> None:
    updates = compute_satellite_forwarding_state_updates(
        prev_fstate={(0, 100): 7, (1, 100): 9},
        curr_fstate={(0, 100): 8, (1, 100): 5},
        algorithm_name="topological_routing",
        satellite_ids=[0, 1],
        ground_station_ids=[100],
        prev_attachments=[(0, 10.0)],
        curr_attachments=[(1, 11.0)],
        prev_interface_neighbor_map={0: {7: 2}, 1: {9: 3}},
        curr_interface_neighbor_map={0: {8: 4}, 1: {5: 6}},
    )

    assert math.isclose(updates["add"]["mean"], 0.5)
    assert math.isclose(updates["delete"]["mean"], 0.5)
    assert math.isclose(updates["modify"]["mean"], 0.0)
    assert math.isclose(updates["total"]["mean"], 1.0)
    assert math.isclose(updates["touched_satellite_rate"], 1.0)


def test_explicit_path_satellite_updates_track_ingress_binding_changes() -> None:
    updates = compute_satellite_forwarding_state_updates(
        prev_fstate={},
        curr_fstate={},
        algorithm_name="explicit_path_routing",
        satellite_ids=[0, 1, 2],
        ground_station_ids=[100],
        prev_attachments=[(0, 10.0)],
        curr_attachments=[(0, 10.0)],
        prev_route_plans={
            (0, 100): {
                "planned_dst_sat_id": 2,
                "adjacency_sid_list": [1, 2],
                "backup_adjacency_sid_list": [None, None],
            }
        },
        curr_route_plans={
            (0, 100): {
                "planned_dst_sat_id": 2,
                "adjacency_sid_list": [3, 2],
                "backup_adjacency_sid_list": [None, None],
            }
        },
    )

    assert math.isclose(updates["modify"]["mean"], 1.0 / 3.0)
    assert math.isclose(updates["total"]["mean"], 1.0 / 3.0)
    assert math.isclose(updates["touched_satellite_count"], 1.0)


def test_link_state_satellite_updates_track_destination_entry_rewrites() -> None:
    updates = compute_satellite_forwarding_state_updates(
        prev_fstate={(0, 100): (1, 7, 8), (1, 100): (2, 9, 10)},
        curr_fstate={(0, 100): (3, 11, 12), (1, 100): (2, 9, 10)},
        algorithm_name="shortest_path_link_state",
        satellite_ids=[0, 1],
        ground_station_ids=[100],
        prev_attachments=[(4, 10.0)],
        curr_attachments=[(4, 10.0)],
        prev_interface_neighbor_map={0: {7: 1}, 1: {9: 2}},
        curr_interface_neighbor_map={0: {11: 3}, 1: {9: 2}},
    )

    assert math.isclose(updates["modify"]["mean"], 0.5)
    assert math.isclose(updates["total"]["mean"], 0.5)
    assert math.isclose(updates["touched_satellite_count"], 1.0)
