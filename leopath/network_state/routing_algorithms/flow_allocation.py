"""Ground-station address policies for flows over attachment-dependent synonyms.

Under attachment addressing a ground station attached to K satellites holds K
topological addresses, one per attachment, and its RMT accepts PDUs sent to any
of them. Which of those addresses flows use is a layer policy. Four are
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
``requester_aware``
    Inside RINA, and route-aware where RINA allows it. The station keeps one
    sticky current address as under ``sticky_nearest``. When A allocates a
    flow to B, the directory resolves B to whichever of B's synonyms is best
    for A, a location-aware directory policy; the flow keeps that synonym while
    B advertises it and otherwise moves to B's current address. The source end
    is a forwarding decision, not an address: A transmits through whichever of
    its attachments is best for the destination, chosen afresh each snapshot.
    Meant for K >= 2 with ``one_per_half`` attachment, so every station has an
    address on each half of a Walker delta shell.
``per_flow_pair``
    Extension beyond RINA. Each flow, one per station pair and shared by both
    directions, pins its own address pair, chosen by the routing family's
    metric; the whole pair is reselected when either synonym is withdrawn.
    Kept as a labelled alternative.

The satellite id sits at index 1 of every candidate tuple.
"""

from collections.abc import Callable

from leopath.topology.topology import GroundStation

GS_ADDRESS_POLICIES = ("sticky_nearest", "nearest", "requester_aware", "per_flow_pair")
STICKY_POLICIES = ("sticky_nearest", "requester_aware")
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
        gs.previous_current_address_satellite_id = previous
        if policy in STICKY_POLICIES and previous in attached:
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
            destination_gs,
            source_candidates,
            destination_candidates,
            select,
            counters,
        )
    if policy == "requester_aware":
        return _requester_aware_pair(
            routing_family,
            source_gs,
            destination_gs,
            source_candidates,
            destination_candidates,
            select,
            counters,
        )
    return select(
        [c for c in source_candidates if c[1] == source_gs.current_address_satellite_id],
        [c for c in destination_candidates if c[1] == destination_gs.current_address_satellite_id],
    )


def _without_ground_link(candidates: list) -> list:
    """The same attachments with the ground-link length (item 0) set to zero."""
    return [(0.0, *candidate[1:]) for candidate in candidates]


def _requester_aware_pair(
    routing_family: str,
    source_gs: GroundStation,
    destination_gs: GroundStation,
    source_candidates: list,
    destination_candidates: list,
    select: Callable[[list, list], tuple | None],
    counters: dict[str, int],
) -> tuple | None:
    """Destination synonym from a location-aware directory, source uplink by forwarding.

    The synonym is pinned per requesting station and destination. When B
    withdraws it, the flow moves to B's current address and B sends A one
    flow update, unless the withdrawn synonym was B's current address and B
    renumbered this snapshot, in which case B's renumbering already sent it.
    """
    key = (routing_family, destination_gs.id)
    pinned = source_gs.requested_synonyms.get(key)
    advertised = {c[1] for c in destination_candidates}
    if pinned is None:
        # B's IPC process chooses with what it can know: A's synonyms from the
        # directory and its own ground links. A's ground-link lengths stay with A,
        # so they don't weigh in here; A uses them when it picks its uplink below.
        chosen = select(_without_ground_link(source_candidates), destination_candidates)
        if chosen is None:
            return None
        counters["flow_allocations"] += 1
        destination = chosen[3]
    elif pinned in advertised:
        destination = pinned
    else:
        destination = destination_gs.current_address_satellite_id
        if destination not in advertised:
            return None
        renumbered_from_it = (
            pinned == destination_gs.previous_current_address_satellite_id and destination != pinned
        )
        if not renumbered_from_it:
            counters["flow_updates"] += 1
            counters["flow_update_messages"] += 1
    source_gs.requested_synonyms[key] = destination
    # The uplink is a forwarding choice among A's own attachments: no address changes.
    return select(source_candidates, [c for c in destination_candidates if c[1] == destination])


def allocate_address_pair(
    routing_family: str,
    source_gs: GroundStation,
    destination_gs: GroundStation,
    source_candidates: list,
    destination_candidates: list,
    select: Callable[[list, list], tuple | None],
    counters: dict[str, int],
) -> tuple | None:
    """``per_flow_pair``: keep the flow's pinned pair while both synonyms are valid.

    A flow is bidirectional, so both directions between two stations share one
    pin, stored on the station with the lower id as (its satellite, the other
    end's satellite). Whichever direction first finds a synonym withdrawn moves
    the flow and pays for it; the other direction then finds the new pin valid.

    ``select`` is the routing family's own pair selection. It returns a tuple
    whose items 1 and 3 are the source and destination satellites, or ``None``.
    A pinned pair is re-evaluated through ``select`` restricted to that pair, so
    the caller gets the same tuple shape whether the flow was kept or moved.
    """
    source_is_owner = source_gs.id < destination_gs.id
    owner, peer = (source_gs, destination_gs) if source_is_owner else (destination_gs, source_gs)
    key = (routing_family, peer.id)
    stored = owner.allocated_address_pairs.get(key)
    pinned = None if stored is None else (stored if source_is_owner else stored[::-1])
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
    owner.allocated_address_pairs[key] = new_pair if source_is_owner else new_pair[::-1]
    return selected
