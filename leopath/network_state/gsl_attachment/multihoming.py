"""Ground-station multihoming assignments for evaluation and routing."""

from collections import Counter

import networkx as nx

ATTACHMENT_POLICIES = ("independent", "exclusive")


def select_multihoming_attachments(
    visibility: list,
    attachment_count: int,
    policy: str = "independent",
) -> tuple[list, dict[str, float]]:
    """Select up to K visible satellites per ground station.

    ``independent`` is the unconstrained top-K upper bound. ``exclusive`` solves
    a minimum-cost maximum-cardinality bipartite assignment in which a satellite
    may serve at most one ground station. Lower attachment ranks are preferred
    before distance, so scarce radios are spread across stations before a station
    receives its second or later link.
    """
    if attachment_count < 1:
        raise ValueError("gs_attachment_count must be at least 1")
    if policy not in ATTACHMENT_POLICIES:
        raise ValueError(
            f"Unknown gs_attachment_policy {policy!r}, expected one of {ATTACHMENT_POLICIES}"
        )

    ordered = [sorted(candidates, key=lambda item: (item[0], item[1])) for candidates in visibility]
    independent = [candidates[:attachment_count] for candidates in ordered]
    claims = Counter(item[1] for candidates in independent for item in candidates)
    conflicts = sum(count - 1 for count in claims.values() if count > 1)

    selected = independent
    if policy == "exclusive":
        selected = _exclusive_assignment(ordered, attachment_count)

    requested = len(visibility) * attachment_count
    assigned = sum(len(candidates) for candidates in selected)
    fully_attached = sum(len(candidates) == attachment_count for candidates in selected)
    return selected, {
        "gs_attachment_requested": float(requested),
        "gs_attachment_assigned": float(assigned),
        "gs_attachment_shortfall": float(requested - assigned),
        "gs_fully_attached": float(fully_attached),
        "gs_attachment_conflicts_unconstrained": float(conflicts),
    }


def _exclusive_assignment(visibility: list, attachment_count: int) -> list:
    source = ("source",)
    sink = ("sink",)
    graph = nx.DiGraph()
    graph.add_node(source)
    graph.add_node(sink)

    max_distance = max(
        (int(round(item[0])) for candidates in visibility for item in candidates),
        default=0,
    )
    rank_penalty = max_distance + 1

    item_by_edge: dict[tuple[int, int], object] = {}
    satellite_ids = sorted({item[1] for candidates in visibility for item in candidates})
    for satellite_id in satellite_ids:
        sat_node = ("sat", satellite_id)
        graph.add_edge(sat_node, sink, capacity=1, weight=0)

    for gs_index, candidates in enumerate(visibility):
        for rank in range(attachment_count):
            slot_node = ("slot", gs_index, rank)
            graph.add_edge(source, slot_node, capacity=1, weight=0)
            for item in candidates:
                distance, satellite_id = item[0], item[1]
                sat_node = ("sat", satellite_id)
                graph.add_edge(
                    slot_node,
                    sat_node,
                    capacity=1,
                    weight=rank * rank_penalty + int(round(distance)),
                )
                item_by_edge[(gs_index, satellite_id)] = item

    flow = nx.max_flow_min_cost(graph, source, sink, capacity="capacity", weight="weight")
    selected: list[list] = [[] for _ in visibility]
    for gs_index in range(len(visibility)):
        for rank in range(attachment_count):
            slot_node = ("slot", gs_index, rank)
            for node, value in flow.get(slot_node, {}).items():
                if value and node[0] == "sat":
                    selected[gs_index].append(item_by_edge[(gs_index, node[1])])
    for candidates in selected:
        candidates.sort(key=lambda item: (item[0], item[1]))
    return selected
