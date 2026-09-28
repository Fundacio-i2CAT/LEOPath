"""Ground-station address policies for flows over attachment-dependent synonyms.

Under attachment addressing a ground station attached to K satellites holds K
topological addresses, one per attachment, and its RMT accepts PDUs sent to any
of them. Which of those addresses flows use is a layer policy. Three are
provided:

``sticky_nearest`` (default)
    Follows the RINA flow allocator and IRATI. Each station has one current
    address. It keeps it while that attachment stays among its K; when the
    attachment is lost, the station itself moves to its nearest remaining
    attachment and every one of its flows follows (flow-allocator-spec §7.2,
    "Write Current Address"; IRATI ``FlowAllocator::address_changed``). The
    directory returns the current address. Extra attachments act as standby
    synonyms for make-before-break renumbering.
``nearest``
    Same, without stickiness: the current address is always the nearest
    attachment, so the station renumbers whenever its nearest satellite changes.
``per_flow_pair``
    Extension beyond RINA. Each flow pins its own source/destination pair,
    chosen by the routing family's metric, and the source reselects the whole
    pair when either synonym is withdrawn. Kept as a labelled alternative.

The satellite id sits at index 1 of every candidate tuple.
"""

from collections.abc import Callable

from leopath.topology.topology import GroundStation

GS_ADDRESS_POLICIES = ("sticky_nearest", "nearest", "per_flow_pair")
DEFAULT_GS_ADDRESS_POLICY = "sticky_nearest"

FLOW_ALLOCATION_COUNTERS = (
    "gs_current_address_changes",
    "flow_allocations",
    "flow_updates",
    "flow_update_messages",
)


def new_flow_allocation_counters() -> dict[str, int]:
    return dict.fromkeys(FLOW_ALLOCATION_COUNTERS, 0)


def update_current_addresses(
    ground_stations: list[GroundStation],
    gs_candidates: list,
    policy: str,
    counters: dict[str, int],
) -> None:
    """Move each station's current address as its policy dictates.

    The evaluation offers every station a flow to every other station, so a
    station that renumbers sends one flow update to each of its len - 1 peers.
    A station with no attachment this snapshot keeps its last address, which no
    satellite can deliver to until it attaches again; the first address a
    station gets is an allocation, not a renumbering.
    """
    if policy not in GS_ADDRESS_POLICIES:
        raise ValueError(
            f"Unknown gs_address_policy {policy!r}, expected one of {GS_ADDRESS_POLICIES}"
        )
    if policy == "per_flow_pair":
        return
    active_flows = max(len(ground_stations) - 1, 0)
    for gs_idx, gs in enumerate(ground_stations):
        candidates = gs_candidates[gs_idx] if gs_idx < len(gs_candidates) else []
        if not candidates:
            continue
        attached = [item[1] for item in sorted(candidates, key=lambda item: (item[0], item[1]))]
        previous = gs.current_address_satellite_id
        if policy == "sticky_nearest" and previous in attached:
            continue
        current = attached[0]
        if previous is not None and current != previous:
            counters["gs_current_address_changes"] += 1
            counters["flow_updates"] += active_flows
            counters["flow_update_messages"] += active_flows
        gs.current_address_satellite_id = current


def resolve_flow_address_pair(
    policy: str,
    routing_family: str,
    source_gs: GroundStation,
    destination_gs: GroundStation,
    source_candidates: list,
    destination_candidates: list,
    select: Callable[[list, list], tuple | None],
    counters: dict[str, int],
) -> tuple | None:
    """The address pair a flow's PDUs carry this snapshot, in ``select``'s tuple shape."""
    if policy == "per_flow_pair":
        return allocate_address_pair(
            routing_family,
            source_gs,
            destination_gs.id,
            source_candidates,
            destination_candidates,
            select,
            counters,
        )
    return select(
        [c for c in source_candidates if c[1] == source_gs.current_address_satellite_id],
        [c for c in destination_candidates if c[1] == destination_gs.current_address_satellite_id],
    )


def allocate_address_pair(
    routing_family: str,
    source_gs: GroundStation,
    destination_gs_id: int,
    source_candidates: list,
    destination_candidates: list,
    select: Callable[[list, list], tuple | None],
    counters: dict[str, int],
) -> tuple | None:
    """``per_flow_pair``: keep the flow's pinned pair while both synonyms are valid.

    ``select`` is the routing family's own pair selection. It returns a tuple
    whose items 1 and 3 are the source and destination satellites, or ``None``.
    A pinned pair is re-evaluated through ``select`` restricted to that pair, so
    the caller gets the same tuple shape whether the flow was kept or moved.
    """
    key = (routing_family, destination_gs_id)
    pinned = source_gs.allocated_address_pairs.get(key)
    if pinned is not None:
        pinned_src, pinned_dst = pinned
        kept = select(
            [c for c in source_candidates if c[1] == pinned_src],
            [c for c in destination_candidates if c[1] == pinned_dst],
        )
        if kept is not None:
            return kept

    selected = select(source_candidates, destination_candidates)
    if selected is None:
        # No pair is reachable now. The flow keeps its old pin, and the move is
        # counted when a pair becomes available again.
        return None
    new_pair = (selected[1], selected[3])
    if pinned is None:
        counters["flow_allocations"] += 1
    else:
        counters["flow_updates"] += 1
        counters["flow_update_messages"] += int(new_pair[0] != pinned[0]) + int(
            new_pair[1] != pinned[1]
        )
    source_gs.allocated_address_pairs[key] = new_pair
    return selected
