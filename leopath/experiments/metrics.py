import csv
import json
import math
from typing import Iterable

import networkx as nx
import numpy as np


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_distribution(values: Iterable[float]) -> dict:
    values_list = list(values)
    if not values_list:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "count": 0,
        }
    values_array = np.asarray(values_list, dtype=float)
    return {
        "min": float(np.min(values_array)),
        "max": float(np.max(values_array)),
        "mean": float(np.mean(values_array)),
        "median": float(np.median(values_array)),
        "p95": float(np.percentile(values_array, 95)),
        "count": int(len(values_array)),
    }


def is_unreachable(entry: object) -> bool:
    if entry is None:
        return True
    if entry == (-1, -1, -1):
        return True
    return False


def build_interface_neighbor_map(sat_neighbor_to_if: dict[tuple[int, int], int]) -> dict[int, dict[int, int]]:
    interface_map: dict[int, dict[int, int]] = {}
    for (src, dst), iface in sat_neighbor_to_if.items():
        interface_map.setdefault(src, {})[iface] = dst
    return interface_map


def normalize_next_hop(
    entry: object,
    src_node_id: int,
    interface_neighbor_map: dict[int, dict[int, int]],
) -> int | None:
    if is_unreachable(entry):
        return None
    if isinstance(entry, tuple):
        if len(entry) == 3:
            next_hop_id = entry[0]
            return next_hop_id if isinstance(next_hop_id, int) else None
        if len(entry) == 2 and entry[0] == "GSL":
            return entry[1] if isinstance(entry[1], int) else None
    if isinstance(entry, int):
        return interface_neighbor_map.get(src_node_id, {}).get(entry)
    return None


def get_gs_attachments(
    ground_station_satellites_in_range: list[list[tuple[float, int]]],
) -> list[tuple[int | None, float]]:
    attachments: list[tuple[int | None, float]] = []
    for satellites in ground_station_satellites_in_range:
        if not satellites:
            attachments.append((None, float("inf")))
            continue
        best_distance, best_sat_id = min(satellites, key=lambda item: item[0])
        attachments.append((best_sat_id, best_distance))
    return attachments


def compute_forwarding_state_stats(
    fstate: dict,
    topology_graph: nx.Graph,
    algorithm_name: str,
    satellite_ids: list[int],
    ground_station_ids: list[int],
) -> dict:
    counts = []
    if algorithm_name == "shortest_path_link_state":
        total_nodes = topology_graph.number_of_nodes()
        counts = [total_nodes for _ in satellite_ids]
    elif algorithm_name == "topological_routing":
        counts = [float(topology_graph.degree(sat_id)) for sat_id in satellite_ids]
    else:
        for sat_id in satellite_ids:
            reachable = 0
            for gs_id in ground_station_ids:
                entry = fstate.get((sat_id, gs_id))
                if not is_unreachable(entry):
                    reachable += 1
            counts.append(reachable)
    stats = summarize_distribution(counts)
    stats["total_sats"] = len(satellite_ids)
    return stats


def compute_gs_handover_rate(
    prev_attachments: list[tuple[int | None, float]],
    curr_attachments: list[tuple[int | None, float]],
) -> float:
    if not prev_attachments or not curr_attachments:
        return 0.0
    changes = 0
    total = min(len(prev_attachments), len(curr_attachments))
    for (prev_sat, _), (curr_sat, _) in zip(prev_attachments, curr_attachments):
        if prev_sat != curr_sat:
            changes += 1
    return changes / total if total else 0.0


def compute_sat_to_gs_churn(
    prev_fstate: dict,
    curr_fstate: dict,
    satellite_ids: list[int],
    ground_station_ids: list[int],
    interface_neighbor_map: dict[int, dict[int, int]],
) -> dict:
    changes = 0
    breaks = 0
    total = 0
    for sat_id in satellite_ids:
        for gs_id in ground_station_ids:
            prev_entry = normalize_next_hop(prev_fstate.get((sat_id, gs_id)), sat_id, interface_neighbor_map)
            curr_entry = normalize_next_hop(curr_fstate.get((sat_id, gs_id)), sat_id, interface_neighbor_map)
            total += 1
            if prev_entry is None and curr_entry is None:
                continue
            if (prev_entry is None) != (curr_entry is None):
                breaks += 1
                continue
            if prev_entry != curr_entry:
                changes += 1
    churn = changes / total if total else 0.0
    break_rate = breaks / total if total else 0.0
    return {
        "churn": churn,
        "break_rate": break_rate,
    }


def compute_gs_to_gs_churn(
    prev_fstate: dict,
    curr_fstate: dict,
    ground_station_ids: list[int],
    prev_attachments: list[tuple[int | None, float]],
    curr_attachments: list[tuple[int | None, float]],
    interface_neighbor_map: dict[int, dict[int, int]],
) -> dict:
    changes = 0
    breaks = 0
    total = 0
    for src_index, src_gs_id in enumerate(ground_station_ids):
        for dst_index, dst_gs_id in enumerate(ground_station_ids):
            if src_gs_id == dst_gs_id:
                continue
            total += 1
            prev_entry = _derive_gs_to_gs_next_hop(
                src_index,
                dst_gs_id,
                prev_fstate,
                prev_attachments,
                interface_neighbor_map,
            )
            curr_entry = _derive_gs_to_gs_next_hop(
                src_index,
                dst_gs_id,
                curr_fstate,
                curr_attachments,
                interface_neighbor_map,
            )
            if prev_entry is None and curr_entry is None:
                continue
            if (prev_entry is None) != (curr_entry is None):
                breaks += 1
                continue
            if prev_entry != curr_entry:
                changes += 1
    churn = changes / total if total else 0.0
    break_rate = breaks / total if total else 0.0
    return {
        "churn": churn,
        "break_rate": break_rate,
    }


def _derive_gs_to_gs_next_hop(
    src_gs_index: int,
    dst_gs_id: int,
    fstate: dict,
    attachments: list[tuple[int | None, float]],
    interface_neighbor_map: dict[int, dict[int, int]],
) -> int | None:
    if (src_gs_index >= len(attachments)) or not attachments:
        return None
    src_attachment, _ = attachments[src_gs_index]
    if src_attachment is None:
        return None
    direct_entry = fstate.get((src_attachment, dst_gs_id))
    return normalize_next_hop(direct_entry, src_attachment, interface_neighbor_map)


def compute_path_stretch(
    fstate: dict,
    topology_graph: nx.Graph,
    satellite_ids: list[int],
    ground_station_ids: list[int],
    attachments: list[tuple[int | None, float]],
    interface_neighbor_map: dict[int, dict[int, int]],
    max_hops: int,
) -> dict:
    sat_set = set(satellite_ids)
    sat_graph = topology_graph.subgraph(satellite_ids)
    hop_stretches = []
    dist_stretches = []
    for src_index, src_gs_id in enumerate(ground_station_ids):
        for dst_index, dst_gs_id in enumerate(ground_station_ids):
            if src_gs_id == dst_gs_id:
                continue
            src_sat, src_gsl_dist = attachments[src_index]
            dst_sat, dst_gsl_dist = attachments[dst_index]
            if src_sat is None or dst_sat is None:
                continue
            if src_sat not in sat_set or dst_sat not in sat_set:
                continue
            opt_hops, opt_dist = _shortest_path_lengths(sat_graph, src_sat, dst_sat)
            if opt_hops is None or opt_dist is None:
                continue
            algo_hops, algo_dist = _follow_routing_path(
                fstate,
                topology_graph,
                src_sat,
                dst_sat,
                dst_gs_id,
                src_gsl_dist,
                dst_gsl_dist,
                interface_neighbor_map,
                max_hops,
            )
            if algo_hops is None or algo_dist is None:
                continue
            if opt_hops > 0:
                hop_stretches.append(algo_hops / opt_hops)
            if opt_dist > 0.0:
                dist_stretches.append(algo_dist / opt_dist)
    return {
        "hop": summarize_distribution(hop_stretches),
        "distance": summarize_distribution(dist_stretches),
    }


def _shortest_path_lengths(
    sat_graph: nx.Graph,
    src_sat: int,
    dst_sat: int,
) -> tuple[int | None, float | None]:
    try:
        hop_len = nx.shortest_path_length(sat_graph, source=src_sat, target=dst_sat)
        dist_len = nx.shortest_path_length(
            sat_graph, source=src_sat, target=dst_sat, weight="weight"
        )
        return int(hop_len), float(dist_len)
    except nx.NetworkXNoPath:
        return None, None
    except nx.NodeNotFound:
        return None, None


def _follow_routing_path(
    fstate: dict,
    topology_graph: nx.Graph,
    src_sat: int,
    dst_sat: int,
    dst_gs_id: int,
    src_gsl_dist: float,
    dst_gsl_dist: float,
    interface_neighbor_map: dict[int, dict[int, int]],
    max_hops: int,
) -> tuple[int | None, float | None]:
    current = src_sat
    visited = set()
    total_hops = 1
    total_dist = float(src_gsl_dist)
    steps = 0
    while current != dst_sat:
        if current in visited:
            return None, None
        if steps >= max_hops:
            return None, None
        visited.add(current)
        entry = fstate.get((current, dst_gs_id))
        next_hop = normalize_next_hop(entry, current, interface_neighbor_map)
        if next_hop is None:
            return None, None
        if next_hop == dst_gs_id:
            if current != dst_sat:
                return None, None
            break
        if not topology_graph.has_edge(current, next_hop):
            return None, None
        weight = topology_graph.edges[current, next_hop].get("weight")
        if weight is None or math.isinf(weight):
            return None, None
        total_hops += 1
        total_dist += float(weight)
        current = next_hop
        steps += 1
    total_hops += 1
    total_dist += float(dst_gsl_dist)
    return total_hops, total_dist
