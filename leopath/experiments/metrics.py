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


def build_interface_neighbor_map(
    sat_neighbor_to_if: dict[tuple[int, int], int],
) -> dict[int, dict[int, int]]:
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
    attachments: list[tuple[int | None, float]] | None = None,
    algorithm_params: dict | None = None,
    route_plans: dict | None = None,
) -> dict:
    algorithm_params = algorithm_params or {}
    route_plans = route_plans or {}
    attachments = attachments or []
    counts = []
    if algorithm_name == "shortest_path_link_state":
        counts = [float(len(satellite_ids)) for _ in satellite_ids]
    elif algorithm_name == "predictive_link_state":
        counts = [float(len(satellite_ids)) for _ in satellite_ids]
    elif algorithm_name == "traditional_segment_routing":
        counts = [float(len(satellite_ids)) for _ in satellite_ids]
    elif algorithm_name == "explicit_path_routing":
        attached_satellites = {sat_id for sat_id, _ in attachments if sat_id is not None}
        for sat_id in satellite_ids:
            # All satellites keep local adjacency/interface entries. Only satellites
            # that currently act as GS-attached ingresses need destination-to-
            # segment bindings for arbitrary outbound traffic.
            reachable = 0
            if sat_id in attached_satellites:
                for gs_id in ground_station_ids:
                    if route_plans.get((sat_id, gs_id)):
                        reachable += 1
                        continue
                    entry = fstate.get((sat_id, gs_id))
                    if not is_unreachable(entry):
                        reachable += 1
            counts.append(float(reachable + len(list(topology_graph.neighbors(sat_id)))))
    elif algorithm_name == "topological_routing":
        counts = [float(len(list(topology_graph.neighbors(sat_id)))) for sat_id in satellite_ids]
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


def compute_satellite_forwarding_state_updates(
    prev_fstate: dict,
    curr_fstate: dict,
    algorithm_name: str,
    satellite_ids: list[int],
    ground_station_ids: list[int],
    prev_attachments: list[tuple[int | None, float]] | None = None,
    curr_attachments: list[tuple[int | None, float]] | None = None,
    prev_interface_neighbor_map: dict[int, dict[int, int]] | None = None,
    curr_interface_neighbor_map: dict[int, dict[int, int]] | None = None,
    prev_route_plans: dict | None = None,
    curr_route_plans: dict | None = None,
) -> dict:
    prev_state = project_satellite_forwarding_state(
        fstate=prev_fstate,
        algorithm_name=algorithm_name,
        satellite_ids=satellite_ids,
        ground_station_ids=ground_station_ids,
        attachments=prev_attachments,
        interface_neighbor_map=prev_interface_neighbor_map,
        route_plans=prev_route_plans,
    )
    curr_state = project_satellite_forwarding_state(
        fstate=curr_fstate,
        algorithm_name=algorithm_name,
        satellite_ids=satellite_ids,
        ground_station_ids=ground_station_ids,
        attachments=curr_attachments,
        interface_neighbor_map=curr_interface_neighbor_map,
        route_plans=curr_route_plans,
    )

    add_counts = []
    delete_counts = []
    modify_counts = []
    total_counts = []
    touched_satellites = 0

    for sat_id in satellite_ids:
        prev_entries = prev_state.get(sat_id, {})
        curr_entries = curr_state.get(sat_id, {})
        prev_keys = set(prev_entries)
        curr_keys = set(curr_entries)
        add_count = len(curr_keys - prev_keys)
        delete_count = len(prev_keys - curr_keys)
        modify_count = sum(
            1 for key in (prev_keys & curr_keys) if prev_entries[key] != curr_entries[key]
        )
        total_count = add_count + delete_count + modify_count
        if total_count > 0:
            touched_satellites += 1
        add_counts.append(float(add_count))
        delete_counts.append(float(delete_count))
        modify_counts.append(float(modify_count))
        total_counts.append(float(total_count))

    satellite_count = len(satellite_ids)
    return {
        "add": summarize_distribution(add_counts),
        "delete": summarize_distribution(delete_counts),
        "modify": summarize_distribution(modify_counts),
        "total": summarize_distribution(total_counts),
        "touched_satellite_count": float(touched_satellites),
        "touched_satellite_rate": (
            touched_satellites / satellite_count if satellite_count else 0.0
        ),
    }


def project_satellite_forwarding_state(
    fstate: dict,
    algorithm_name: str,
    satellite_ids: list[int],
    ground_station_ids: list[int],
    attachments: list[tuple[int | None, float]] | None = None,
    interface_neighbor_map: dict[int, dict[int, int]] | None = None,
    route_plans: dict | None = None,
) -> dict[int, dict[tuple, object]]:
    attachments = attachments or []
    interface_neighbor_map = interface_neighbor_map or {}
    route_plans = route_plans or {}
    projected_state = {sat_id: {} for sat_id in satellite_ids}
    attachment_by_gs = {
        gs_id: sat_id for gs_id, (sat_id, _) in zip(ground_station_ids, attachments)
    }

    _add_local_delivery_entries(projected_state, attachment_by_gs)

    if algorithm_name in {
        "shortest_path_link_state",
        "predictive_link_state",
        "traditional_segment_routing",
    }:
        _add_destination_forwarding_entries(
            projected_state,
            fstate,
            satellite_ids,
            ground_station_ids,
            attachment_by_gs,
            interface_neighbor_map,
            route_plans,
        )
    elif algorithm_name == "explicit_path_routing":
        _add_explicit_ingress_bindings(
            projected_state,
            fstate,
            ground_station_ids,
            attachment_by_gs,
            interface_neighbor_map,
            route_plans,
        )

    return projected_state


def _add_local_delivery_entries(
    projected_state: dict[int, dict[tuple, object]],
    attachment_by_gs: dict[int, int | None],
) -> None:
    for gs_id, sat_id in attachment_by_gs.items():
        if sat_id is None or sat_id not in projected_state:
            continue
        projected_state[sat_id][("gsl", gs_id)] = "local_delivery"


def _add_destination_forwarding_entries(
    projected_state: dict[int, dict[tuple, object]],
    fstate: dict,
    satellite_ids: list[int],
    ground_station_ids: list[int],
    attachment_by_gs: dict[int, int | None],
    interface_neighbor_map: dict[int, dict[int, int]],
    route_plans: dict,
) -> None:
    for src_sat_id in satellite_ids:
        local_state = projected_state[src_sat_id]
        for dst_gs_id in ground_station_ids:
            dst_sat_id = attachment_by_gs.get(dst_gs_id)
            if dst_sat_id is None or dst_sat_id == src_sat_id:
                continue
            next_hop = _extract_next_hop(
                src_sat_id,
                dst_gs_id,
                fstate,
                interface_neighbor_map,
                route_plans,
            )
            if next_hop is None or next_hop == dst_gs_id:
                continue
            entry_key = ("dst_sat", dst_sat_id)
            local_state.setdefault(entry_key, ("next_sat", next_hop))


def _add_explicit_ingress_bindings(
    projected_state: dict[int, dict[tuple, object]],
    fstate: dict,
    ground_station_ids: list[int],
    attachment_by_gs: dict[int, int | None],
    interface_neighbor_map: dict[int, dict[int, int]],
    route_plans: dict,
) -> None:
    attached_satellites = {
        sat_id for sat_id in attachment_by_gs.values() if sat_id in projected_state
    }
    for ingress_sat_id in attached_satellites:
        local_state = projected_state[ingress_sat_id]
        for dst_gs_id in ground_station_ids:
            route_plan = route_plans.get((ingress_sat_id, dst_gs_id)) or {}
            if route_plan:
                local_state[("binding", dst_gs_id)] = _normalize_explicit_binding(route_plan)
                continue
            entry = fstate.get((ingress_sat_id, dst_gs_id))
            if is_unreachable(entry):
                continue
            next_hop = normalize_next_hop(entry, ingress_sat_id, interface_neighbor_map)
            if next_hop is None:
                continue
            local_state[("binding", dst_gs_id)] = ("first_hop", next_hop)


def _normalize_explicit_binding(route_plan: dict) -> tuple:
    return (
        route_plan.get("planned_dst_sat_id"),
        tuple(route_plan.get("adjacency_sid_list") or []),
        tuple(route_plan.get("backup_adjacency_sid_list") or []),
        route_plan.get("local_protection_mode"),
        route_plan.get("final_egress_mode", "strict"),
    )


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


def compute_gs_renumbering_stats(
    prev_attachments: list[tuple[int | None, float]],
    curr_attachments: list[tuple[int | None, float]],
) -> dict:
    if not prev_attachments or not curr_attachments:
        return {
            "count": 0.0,
            "rate": 0.0,
        }
    changes = 0
    total = min(len(prev_attachments), len(curr_attachments))
    for (prev_sat, _), (curr_sat, _) in zip(prev_attachments, curr_attachments):
        if prev_sat != curr_sat:
            changes += 1
    return {
        "count": float(changes),
        "rate": (changes / total) if total else 0.0,
    }


def compute_explicit_header_stats(
    route_plans: dict | None,
    source_attachments: list[tuple[int | None, float]] | None = None,
    ground_station_ids: list[int] | None = None,
    bytes_key: str = "strict_header_bytes",
) -> dict:
    route_plans = route_plans or {}
    header_sizes = []
    for _, route_plan in _iter_explicit_route_plans(
        route_plans,
        source_attachments,
        ground_station_ids,
    ):
        if not route_plan:
            continue
        strict_header_bytes = route_plan.get(bytes_key)
        if strict_header_bytes is None:
            continue
        header_sizes.append(float(strict_header_bytes))
    return summarize_distribution(header_sizes)


def compute_explicit_failover_stats(
    topology_graph: nx.Graph,
    route_plans: dict | None,
    ground_station_ids: list[int],
    ground_station_satellites_in_range: list | None,
    source_attachments: list[tuple[int | None, float]] | None = None,
) -> dict:
    route_plans = route_plans or {}
    visibility_by_gs = {}
    if ground_station_satellites_in_range is not None:
        for ground_station_id, visibility in zip(
            ground_station_ids, ground_station_satellites_in_range
        ):
            visibility_by_gs[ground_station_id] = visibility

    delivered = 0
    protected_repair = 0
    broken_adj = 0
    egress_not_visible = 0
    dynamic_egress_repair = 0
    dynamic_egress_unavailable = 0
    total = 0

    for (_, dst_gs_id), route_plan in _iter_explicit_route_plans(
        route_plans,
        source_attachments,
        ground_station_ids,
    ):
        if not route_plan:
            continue
        sat_path = route_plan.get("satellite_path", [])
        adjacency_sid_list = route_plan.get("adjacency_sid_list", [])
        backup_adjacency_sid_list = route_plan.get("backup_adjacency_sid_list", [])
        planned_dst_sat_id = route_plan.get("planned_dst_sat_id")
        if not sat_path or sat_path[1:] != adjacency_sid_list:
            continue
        total += 1
        plan_failed = False
        for hop_index, (current_sat_id, primary_next_sat_id) in enumerate(
            zip(sat_path, sat_path[1:])
        ):
            if topology_graph.has_edge(current_sat_id, primary_next_sat_id):
                continue
            backup_next_sat_id = (
                backup_adjacency_sid_list[hop_index]
                if hop_index < len(backup_adjacency_sid_list)
                else None
            )
            if (
                backup_next_sat_id is not None
                and topology_graph.has_edge(current_sat_id, backup_next_sat_id)
                and topology_graph.has_edge(backup_next_sat_id, primary_next_sat_id)
            ):
                protected_repair += 1
                continue
            broken_adj += 1
            plan_failed = True
            break
        if plan_failed:
            continue
        visibility = visibility_by_gs.get(dst_gs_id, [])
        if _lookup_visible_satellite_distance(visibility, planned_dst_sat_id) is None:
            if route_plan.get("final_egress_mode") == "dynamic":
                current_dst_sat_id = route_plan.get("current_dst_sat_id")
                repair_path = route_plan.get("egress_repair_satellite_path", [])
                if (
                    current_dst_sat_id is not None
                    and repair_path
                    and repair_path[0] == planned_dst_sat_id
                    and repair_path[-1] == current_dst_sat_id
                    and _lookup_visible_satellite_distance(
                        visibility,
                        current_dst_sat_id,
                    )
                    is not None
                    and _satellite_path_is_available(topology_graph, repair_path)
                ):
                    dynamic_egress_repair += 1
                    delivered += 1
                    continue
                dynamic_egress_unavailable += 1
            egress_not_visible += 1
            continue
        delivered += 1

    return {
        "count": float(total),
        "delivered_count": float(delivered),
        "delivered_rate": (delivered / total) if total else 0.0,
        "protected_repair_count": float(protected_repair),
        "protected_repair_rate": (protected_repair / total) if total else 0.0,
        "broken_adj_count": float(broken_adj),
        "broken_adj_rate": (broken_adj / total) if total else 0.0,
        "egress_not_visible_count": float(egress_not_visible),
        "egress_not_visible_rate": (egress_not_visible / total) if total else 0.0,
        "dynamic_egress_repair_count": float(dynamic_egress_repair),
        "dynamic_egress_repair_rate": (dynamic_egress_repair / total) if total else 0.0,
        "dynamic_egress_unavailable_count": float(dynamic_egress_unavailable),
        "dynamic_egress_unavailable_rate": (dynamic_egress_unavailable / total if total else 0.0),
    }


def _satellite_path_is_available(topology_graph: nx.Graph, sat_path: list[int]) -> bool:
    for current_sat_id, next_sat_id in zip(sat_path, sat_path[1:]):
        if not topology_graph.has_edge(current_sat_id, next_sat_id):
            return False
        weight = topology_graph.edges[current_sat_id, next_sat_id].get("weight")
        if weight is None or math.isinf(weight):
            return False
    return True


def _iter_explicit_route_plans(
    route_plans: dict,
    source_attachments: list[tuple[int | None, float]] | None = None,
    ground_station_ids: list[int] | None = None,
):
    if source_attachments is None or ground_station_ids is None:
        yield from route_plans.items()
        return

    for src_gs_index, src_gs_id in enumerate(ground_station_ids):
        if src_gs_index >= len(source_attachments):
            continue
        src_sat_id, _ = source_attachments[src_gs_index]
        if src_sat_id is None:
            continue
        for dst_gs_id in ground_station_ids:
            if dst_gs_id == src_gs_id:
                continue
            route_plan = route_plans.get((src_sat_id, dst_gs_id))
            if route_plan is None:
                continue
            yield (src_sat_id, dst_gs_id), route_plan


def compute_sat_to_gs_churn(
    prev_fstate: dict,
    curr_fstate: dict,
    satellite_ids: list[int],
    ground_station_ids: list[int],
    interface_neighbor_map: dict[int, dict[int, int]],
    prev_route_plans: dict | None = None,
    curr_route_plans: dict | None = None,
) -> dict:
    prev_route_plans = prev_route_plans or {}
    curr_route_plans = curr_route_plans or {}
    changes = 0
    breaks = 0
    total = 0
    for sat_id in satellite_ids:
        for gs_id in ground_station_ids:
            prev_entry = _extract_next_hop(
                sat_id,
                gs_id,
                prev_fstate,
                interface_neighbor_map,
                prev_route_plans,
            )
            curr_entry = _extract_next_hop(
                sat_id,
                gs_id,
                curr_fstate,
                interface_neighbor_map,
                curr_route_plans,
            )
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
    prev_route_plans: dict | None = None,
    curr_route_plans: dict | None = None,
) -> dict:
    prev_route_plans = prev_route_plans or {}
    curr_route_plans = curr_route_plans or {}
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
                prev_route_plans,
            )
            curr_entry = _derive_gs_to_gs_next_hop(
                src_index,
                dst_gs_id,
                curr_fstate,
                curr_attachments,
                interface_neighbor_map,
                curr_route_plans,
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
    route_plans: dict | None = None,
) -> int | None:
    route_plans = route_plans or {}
    if (src_gs_index >= len(attachments)) or not attachments:
        return None
    src_attachment, _ = attachments[src_gs_index]
    if src_attachment is None:
        return None
    return _extract_next_hop(
        src_attachment,
        dst_gs_id,
        fstate,
        interface_neighbor_map,
        route_plans,
    )


def compute_path_stretch(
    fstate: dict,
    topology_graph: nx.Graph,
    satellite_ids: list[int],
    ground_station_ids: list[int],
    attachments: list[tuple[int | None, float]],
    interface_neighbor_map: dict[int, dict[int, int]],
    max_hops: int,
    route_plans: dict | None = None,
    ground_station_satellites_in_range: list | None = None,
) -> dict:
    route_plans = route_plans or {}
    sat_set = set(satellite_ids)
    sat_graph = topology_graph.subgraph(satellite_ids)
    hop_stretches = []
    dist_stretches = []
    for src_index, src_gs_id in enumerate(ground_station_ids):
        for dst_index, dst_gs_id in enumerate(ground_station_ids):
            if src_gs_id == dst_gs_id:
                continue
            src_sat, src_gsl_dist = attachments[src_index]
            if src_sat is None:
                continue
            if src_sat not in sat_set:
                continue
            destination_visibility = (
                None
                if ground_station_satellites_in_range is None
                else ground_station_satellites_in_range[dst_index]
            )
            dst_sat = _resolve_routed_destination_satellite(
                fstate,
                topology_graph,
                src_sat,
                dst_gs_id,
                interface_neighbor_map,
                max_hops,
                route_plans,
                destination_visibility,
            )
            if dst_sat is None:
                continue
            dst_gsl_dist = _lookup_visible_satellite_distance(destination_visibility, dst_sat)
            if dst_gsl_dist is None:
                _, nearest_dst_gsl_dist = attachments[dst_index]
                dst_gsl_dist = nearest_dst_gsl_dist
            opt_hops, opt_dist = _shortest_path_lengths(sat_graph, src_sat, dst_sat)
            if opt_hops is None or opt_dist is None:
                continue

            # Make the baseline comparable to the algorithm-induced end-to-end path:
            # include both GSL legs (GS->sat and sat->GS) in hop count and distance.
            opt_hops_total = opt_hops + 2
            opt_dist_total = float(opt_dist) + float(src_gsl_dist) + float(dst_gsl_dist)
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
                route_plans,
                destination_visibility,
            )
            if algo_hops is None or algo_dist is None:
                continue

            if opt_hops_total > 0:
                hop_stretches.append(algo_hops / opt_hops_total)
            if opt_dist_total > 0.0:
                dist_stretches.append(algo_dist / opt_dist_total)
    return {
        "hop": summarize_distribution(hop_stretches),
        "distance": summarize_distribution(dist_stretches),
    }


def _resolve_routed_destination_satellite(
    fstate: dict,
    topology_graph: nx.Graph,
    src_sat: int,
    dst_gs_id: int,
    interface_neighbor_map: dict[int, dict[int, int]],
    max_hops: int,
    route_plans: dict | None = None,
    current_destination_visibility: list[tuple[float, int]] | None = None,
) -> int | None:
    route_plans = route_plans or {}
    route_plan = route_plans.get((src_sat, dst_gs_id))
    if route_plan:
        if route_plan.get("final_egress_mode") == "dynamic":
            current_dst_sat_id = route_plan.get("current_dst_sat_id")
            if (
                current_dst_sat_id is not None
                and _lookup_visible_satellite_distance(
                    current_destination_visibility,
                    current_dst_sat_id,
                )
                is not None
            ):
                return current_dst_sat_id
            return None
        planned_dst_sat_id = route_plan.get("planned_dst_sat_id")
        if (
            _lookup_visible_satellite_distance(
                current_destination_visibility,
                planned_dst_sat_id,
            )
            is not None
        ):
            return planned_dst_sat_id
        return None

    current = src_sat
    visited = set()
    steps = 0
    while steps <= max_hops:
        entry = fstate.get((current, dst_gs_id))
        next_hop = normalize_next_hop(entry, current, interface_neighbor_map)
        if next_hop == dst_gs_id:
            return current
        if next_hop is None:
            return None
        if current in visited:
            return None
        if not topology_graph.has_edge(current, next_hop):
            return None
        visited.add(current)
        current = next_hop
        steps += 1
    return None


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
    route_plans: dict | None = None,
    current_destination_visibility: list[tuple[float, int]] | None = None,
) -> tuple[int | None, float | None]:
    route_plans = route_plans or {}
    route_plan = route_plans.get((src_sat, dst_gs_id))
    if route_plan:
        return _follow_explicit_route_plan(
            topology_graph,
            route_plan,
            src_sat,
            src_gsl_dist,
            max_hops,
            current_destination_visibility,
        )

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


def _extract_next_hop(
    sat_id: int,
    gs_id: int,
    fstate: dict,
    interface_neighbor_map: dict[int, dict[int, int]],
    route_plans: dict,
) -> int | None:
    route_plan = route_plans.get((sat_id, gs_id), {})
    adjacency_sid_list = route_plan.get("adjacency_sid_list", [])
    planned_dst_sat_id = route_plan.get("planned_dst_sat_id")
    if route_plan.get("final_egress_mode") == "dynamic" and planned_dst_sat_id == sat_id:
        current_dst_sat_id = route_plan.get("current_dst_sat_id")
        if current_dst_sat_id == sat_id:
            return gs_id
        repair_path = route_plan.get("egress_repair_satellite_path", [])
        if len(repair_path) > 1 and repair_path[0] == sat_id:
            return repair_path[1]
        return None
    if planned_dst_sat_id == sat_id:
        return gs_id
    if adjacency_sid_list:
        if adjacency_sid_list[0] == sat_id:
            remaining_sid_list = adjacency_sid_list[1:]
            return gs_id if not remaining_sid_list else remaining_sid_list[0]
        if len(adjacency_sid_list) == 1 and planned_dst_sat_id == adjacency_sid_list[0]:
            return adjacency_sid_list[0]
        return adjacency_sid_list[0]

    sat_path = route_plan.get("satellite_path", [])
    if sat_path:
        if len(sat_path) == 1 and planned_dst_sat_id == sat_id:
            return gs_id
        return sat_path[1]
    entry = fstate.get((sat_id, gs_id))
    return normalize_next_hop(entry, sat_id, interface_neighbor_map)


def _follow_explicit_route_plan(
    topology_graph: nx.Graph,
    route_plan: dict,
    src_sat: int,
    src_gsl_dist: float,
    max_hops: int,
    current_destination_visibility: list[tuple[float, int]] | None,
) -> tuple[int | None, float | None]:
    sat_path = route_plan.get("satellite_path", [])
    adjacency_sid_list = route_plan.get("adjacency_sid_list", [])
    backup_adjacency_sid_list = route_plan.get("backup_adjacency_sid_list", [])
    planned_dst_sat_id = route_plan.get("planned_dst_sat_id")
    if not sat_path or sat_path[0] != src_sat or sat_path[-1] != planned_dst_sat_id:
        return None, None
    if adjacency_sid_list != sat_path[1:]:
        return None, None
    if backup_adjacency_sid_list and len(backup_adjacency_sid_list) != len(adjacency_sid_list):
        return None, None
    if len(sat_path) > max_hops:
        return None, None

    total_hops = 1
    total_dist = float(src_gsl_dist)
    for hop_index, (current, next_hop) in enumerate(zip(sat_path, sat_path[1:])):
        effective_next_hop = next_hop
        if not topology_graph.has_edge(current, effective_next_hop):
            backup_next_hop = (
                backup_adjacency_sid_list[hop_index]
                if hop_index < len(backup_adjacency_sid_list)
                else None
            )
            if backup_next_hop is None:
                return None, None
            if not topology_graph.has_edge(current, backup_next_hop):
                return None, None
            if not topology_graph.has_edge(backup_next_hop, next_hop):
                return None, None
            first_leg_weight = topology_graph.edges[current, backup_next_hop].get("weight")
            second_leg_weight = topology_graph.edges[backup_next_hop, next_hop].get("weight")
            if (
                first_leg_weight is None
                or second_leg_weight is None
                or math.isinf(first_leg_weight)
                or math.isinf(second_leg_weight)
            ):
                return None, None
            total_hops += 2
            total_dist += float(first_leg_weight) + float(second_leg_weight)
            continue
        weight = topology_graph.edges[current, effective_next_hop].get("weight")
        if weight is None or math.isinf(weight):
            return None, None
        total_hops += 1
        total_dist += float(weight)

    if route_plan.get("final_egress_mode") == "dynamic":
        repair_path = route_plan.get("egress_repair_satellite_path", [])
        current_dst_sat_id = route_plan.get("current_dst_sat_id")
        if (
            current_dst_sat_id is None
            or not repair_path
            or repair_path[0] != planned_dst_sat_id
            or repair_path[-1] != current_dst_sat_id
        ):
            return None, None
        for repair_current, repair_next in zip(repair_path, repair_path[1:]):
            if not topology_graph.has_edge(repair_current, repair_next):
                return None, None
            repair_weight = topology_graph.edges[repair_current, repair_next].get("weight")
            if repair_weight is None or math.isinf(repair_weight):
                return None, None
            total_hops += 1
            total_dist += float(repair_weight)
        dst_gsl_dist = _lookup_visible_satellite_distance(
            current_destination_visibility,
            current_dst_sat_id,
        )
    else:
        dst_gsl_dist = _lookup_visible_satellite_distance(
            current_destination_visibility,
            planned_dst_sat_id,
        )
    if dst_gsl_dist is None:
        return None, None
    total_hops += 1
    total_dist += float(dst_gsl_dist)
    return total_hops, total_dist


def _lookup_visible_satellite_distance(
    destination_visibility: list[tuple[float, int]] | None,
    satellite_id: int | None,
) -> float | None:
    if destination_visibility is None or satellite_id is None:
        return None
    for distance, visible_satellite_id in destination_visibility:
        if visible_satellite_id == satellite_id:
            return float(distance)
    return None
