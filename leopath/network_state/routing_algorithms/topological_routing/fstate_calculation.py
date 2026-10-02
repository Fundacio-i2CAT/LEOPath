import math
import time
from collections.abc import Callable
from typing import Optional

import networkx as nx

from leopath import logger
from leopath.network_state.gsl_attachment.multihoming import (
    select_multihoming_attachments,
)
from leopath.network_state.routing_algorithms.flow_allocation import (
    DEFAULT_GS_ADDRESS_POLICY,
    new_flow_allocation_counters,
    resolve_flow_address_pair,
    update_current_addresses,
)
from leopath.network_state.routing_algorithms.topological_routing.exception_policy import (
    apply_exception_policy,
)
from leopath.topology.satellite.topological_network_address import (
    TopologicalNetworkAddress,
    torus_topological_distance,
    weighted_torus_progress_distance,
)
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

log = logger.get_logger(__name__)


def algorithm_free_one_only_over_isls_topological(
    time_since_epoch_ns: int,
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    topology_with_isls: LEOTopology,
    ground_station_satellites_in_range: list,
    list_gsl_interfaces_info: list,
    prev_output: dict | None = None,
    enable_verbose_logs: bool = False,
) -> dict:
    """
    Calculates bandwidth and forwarding state using topological routing (ISLs only, no GS relaying).

    This is the main entry point for the topological routing algorithm that integrates
    with the simulation framework.

    Args:
        time_since_epoch_ns: Current time step relative to epoch (integer ns)
        constellation_data: Holds satellite list, counts, max lengths, epoch string
        ground_stations: List of GroundStation objects
        topology_with_isls: LEOTopology object containing the graph with ISL links
        ground_station_satellites_in_range: List where index=gs_idx, value=list of (distance, sat_id) tuples
        list_gsl_interfaces_info: List of dicts, one per sat/GS, with bandwidth info
        prev_output: Dictionary containing 'fstate' and 'bandwidth' objects from the previous step
        enable_verbose_logs: Boolean to enable detailed logging

    Returns:
        Dictionary containing the new 'fstate' and 'bandwidth' state objects
    """
    log.debug(
        f"Running algorithm_free_one_only_over_isls_topological for t={time_since_epoch_ns} ns"
    )

    # Calculate bandwidth state (same as shortest path algorithm)
    bandwidth_state = _calculate_bandwidth_state(
        constellation_data, ground_stations, list_gsl_interfaces_info
    )

    # Check if graph has changed by comparing with previous state
    graph_has_changed = True
    prev_fstate = None
    if prev_output is not None:
        prev_fstate = prev_output.get("fstate")
        # For now, assume graph has changed unless we implement proper change detection
        # In a full implementation, you'd compare topology graphs here
        graph_has_changed = True

    # Calculate forwarding state using topological routing
    fstate = calculate_fstate_topological_routing_no_gs_relay(
        topology_with_isls,
        ground_stations,
        ground_station_satellites_in_range,
        constellation_data,
        time_since_epoch_ns,
        prev_fstate,
        graph_has_changed,
    )

    return {
        "fstate": fstate,
        "bandwidth": bandwidth_state,
    }


def _calculate_bandwidth_state(
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    list_gsl_interfaces_info: list,
) -> dict:
    """
    Returns a dict mapping node_id to its aggregate_max_bandwidth.
    """
    num_satellites = constellation_data.number_of_satellites
    num_total_nodes = num_satellites + len(ground_stations)
    bandwidth_state = {}

    if len(list_gsl_interfaces_info) != num_total_nodes:
        log.warning(
            f"Length mismatch: list_gsl_interfaces_info ({len(list_gsl_interfaces_info)}) "
            f"vs total nodes ({num_total_nodes}). Bandwidth state might be incomplete."
        )

    for i in range(num_total_nodes):
        if i < len(list_gsl_interfaces_info):
            node_info = list_gsl_interfaces_info[i]
            node_id = node_info.get("id", i)
            bandwidth = node_info.get("aggregate_max_bandwidth", 0.0)
        else:
            node_id = i
            bandwidth = 0.0
            log.error(
                f"Index {i} out of bounds for list_gsl_interfaces_info, setting BW=0 for node {node_id}"
            )
        bandwidth_state[node_id] = bandwidth
        log.debug(f"  Bandwidth state: Node {node_id}, IF 0, BW = {bandwidth}")

    log.debug(f"  Calculated bandwidth state for {len(bandwidth_state)} nodes.")
    return bandwidth_state


def calculate_fstate_topological_routing_no_gs_relay(
    topology_with_isls: LEOTopology,
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
    constellation_data: ConstellationData | None = None,
    time_since_epoch_ns: int = 0,
    prev_fstate: dict | None = None,
    graph_has_changed: bool = True,
    algorithm_params: dict | None = None,
    state_report: dict | None = None,
    selected_egresses: dict[tuple[int, int], int] | None = None,
    fixed_address_routes: dict[tuple[int, int], dict] | None = None,
) -> dict:
    """
    Calculates forwarding state using topological routing over ISLs only (no GS relays).

    Implements the topological routing algorithm with the following steps:
    1. At t=0: Set 6GRUPA addresses to all nodes and initialize forwarding tables
    2. If graph hasn't changed, reuse previous state
    3. Handle ground station link changes with renumbering
    4. Calculate satellite-to-GS routing decisions

    Args:
        topology_with_isls: Network topology with ISL links
        ground_stations: List of ground stations
        ground_station_satellites_in_range: List where index=gs_idx, value=list of (distance, sat_id) tuples
        time_since_epoch_ns: Time since epoch in nanoseconds
        prev_fstate: Previous forwarding state (for state comparison)
        graph_has_changed: Whether the topology graph has changed since last computation

    Returns:
        Dictionary containing forwarding state
    """
    log.debug("Calculating topological routing fstate object (no GS relay)")
    algorithm_params = algorithm_params or {}
    distance_mode = str(algorithm_params.get("distance_mode", "torus_weighted_lookahead"))
    gs_addressing = str(algorithm_params.get("gs_addressing", "visibility"))
    gs_attachment_count = int(algorithm_params.get("gs_attachment_count", 1))
    gs_attachment_policy = str(algorithm_params.get("gs_attachment_policy", "independent"))
    if gs_attachment_count < 1:
        raise ValueError("gs_attachment_count must be at least 1")

    if constellation_data is None:
        constellation_data = topology_with_isls.constellation_data

    full_graph = topology_with_isls.graph

    try:
        all_satellite_ids = {sat.id for sat in topology_with_isls.get_satellites()}
    except Exception as e:
        log.exception(f"Error getting satellite IDs from topology: {e}")
        return {}
    satellite_node_ids = sorted(
        [node_id for node_id in full_graph.nodes() if node_id in all_satellite_ids]
    )

    if not satellite_node_ids:
        log.warning("No valid satellite nodes found in the graph for path calculation.")
        return {}

    satellite_only_subgraph = full_graph.subgraph(satellite_node_ids)

    if satellite_only_subgraph.number_of_nodes() == 0:
        log.warning("Satellite-only subgraph is empty. No ISL paths possible.")
        return {}

    # Step 1: Initialize at t=0 - set 6GRUPA addresses and neighbor forwarding tables
    if time_since_epoch_ns == 0:
        log.debug("t=0: Setting 6GRUPA addresses to all nodes and initializing forwarding tables")
        _set_sixgrupa_addresses_to_all_nodes(topology_with_isls, constellation_data)
        _fill_forwarding_tables_in_every_satellite(
            satellite_node_ids, satellite_only_subgraph, topology_with_isls, constellation_data
        )
        # Also assign GS addresses for initial GSL attachments. The attachment is
        # the nearest visible satellite, the same rule _detect_gsl_changes applies
        # from then on, so the first address does not renumber immediately.
        for gs_idx, gs in enumerate(ground_stations):
            curr_sat_id = None
            if gs_idx < len(ground_station_satellites_in_range):
                satellites = ground_station_satellites_in_range[gs_idx]
                if satellites:
                    _, curr_sat_id = min(satellites, key=lambda visible: visible[0])
            if curr_sat_id is not None:
                _perform_renumbering_for_gs(
                    gs,
                    None,
                    curr_sat_id,
                    topology_with_isls,
                    constellation_data,
                )
        graph_has_changed = True  # Force recalculation on first run

    # Step 2: Check if we can reuse previous state
    if not graph_has_changed and prev_fstate is not None:
        log.debug("Graph unchanged, reusing previous fstate")
        return prev_fstate

    # Step 3: Handle ground station link changes (renumbering if needed)
    gsl_changes = _detect_gsl_changes(ground_stations, ground_station_satellites_in_range)
    for gs_idx, (prev_sat_id, curr_sat_id) in gsl_changes.items():
        gs = ground_stations[gs_idx]
        log.debug(f"GSL changed for GS {gs.id}: {prev_sat_id} -> {curr_sat_id}")
        _perform_renumbering_for_gs(
            gs,
            prev_sat_id,
            curr_sat_id,
            topology_with_isls,
            constellation_data,
        )

    # Step 4: Calculate satellite-to-GS forwarding state
    fstate: dict[tuple, tuple] = {}
    satellite_addresses = {
        satellite_id: _get_satellite_address(
            topology_with_isls,
            satellite_id,
            constellation_data,
        )
        for satellite_id in satellite_node_ids
    }
    neighbor_candidates = {
        satellite_id: [
            (
                neighbor_id,
                interface,
                satellite_addresses[neighbor_id],
                float(satellite_only_subgraph.edges[satellite_id, neighbor_id].get("weight", 1.0)),
            )
            for neighbor_id in satellite_only_subgraph.neighbors(satellite_id)
            if (interface := topology_with_isls.sat_neighbor_to_if.get((satellite_id, neighbor_id)))
            is not None
            and neighbor_id in satellite_addresses
        ]
        for satellite_id in satellite_node_ids
    }
    neighbor_candidates = _with_local_detours(
        neighbor_candidates,
        topology_with_isls,
        satellite_only_subgraph,
        satellite_addresses,
        algorithm_params,
        distance_mode,
    )
    per_satellite_work: dict | None = {} if state_report is not None else None
    weight_model = None
    if distance_mode == "torus_weighted_pivot":
        geometry_source = str(algorithm_params.get("geometry_source", "observed"))
        weight_model = _build_reported_torus_weight_model(
            _geometry_subgraph(
                topology_with_isls,
                satellite_node_ids,
                satellite_only_subgraph,
                geometry_source,
                satellite_addresses=satellite_addresses,
                constellation_data=constellation_data,
                time_s=time_since_epoch_ns / 1e9,
            ),
            satellite_addresses,
            constellation_data,
            state_report,
            wiring=str(algorithm_params.get("isl_wiring", "plus_grid")),
        )
        if state_report is not None:
            state_report.update(_describe_geometry_source(geometry_source, constellation_data))
    gs_destination_candidates = _build_gs_destination_candidates(
        ground_station_satellites_in_range,
        satellite_addresses,
        gs_addressing,
        gs_attachment_count,
        gs_attachment_policy,
    )
    if state_report is not None:
        # Attachment changes are what a ground station has to renumber for, and
        # under attachment addressing each one costs a directory update and a
        # flow update to the far end of every active flow. The harness adds the
        # aux_ prefix, so the column reaches the summaries as aux_gs_renumberings.
        selected = gs_destination_candidates if gs_addressing == "attachment" else []
        address_set_changes = _update_gs_attachment_sets(
            ground_stations,
            selected,
            gs_attachment_count,
        )
        state_report["gs_renumberings"] = float(address_set_changes["stations"])
        state_report["gs_address_additions"] = float(address_set_changes["additions"])
        state_report["gs_address_removals"] = float(address_set_changes["removals"])

    _calculate_sat_to_gs_fstate(
        topology_with_isls,
        ground_stations,
        satellite_node_ids,
        satellite_addresses,
        neighbor_candidates,
        gs_destination_candidates,
        fstate,
        constellation_data,
        distance_mode,
        weight_model,
        per_satellite_work=per_satellite_work,
        potential=_EgressPotential(
            satellite_addresses,
            gs_destination_candidates,
            constellation_data,
            distance_mode,
            weight_model,
        ),
        forwarding_guard=_resolve_forwarding_guard(algorithm_params, distance_mode),
        selected_egresses=selected_egresses,
    )

    fixed_exception_report: dict = {}
    if fixed_address_routes is not None:
        flow_counters = new_flow_allocation_counters()
        address_policy = str(
            (algorithm_params or {}).get("gs_address_policy", DEFAULT_GS_ADDRESS_POLICY)
        )
        update_current_addresses(
            ground_stations, gs_destination_candidates, address_policy, flow_counters
        )
        fixed_address_routes.update(
            _build_fixed_address_routes(
                address_policy,
                ground_stations,
                gs_destination_candidates,
                satellite_addresses,
                neighbor_candidates,
                constellation_data,
                distance_mode,
                weight_model,
                _resolve_forwarding_guard(algorithm_params, distance_mode),
                flow_counters,
                exception_policy=str(algorithm_params.get("exception_policy", "none")),
                live_graph=satellite_only_subgraph,
                exception_report=fixed_exception_report,
            )
        )
        if state_report is not None:
            state_report.update({key: float(value) for key, value in flow_counters.items()})

    apply_exception_policy(
        fstate,
        satellite_only_subgraph,
        topology_with_isls.sat_neighbor_to_if,
        getattr(topology_with_isls, "nominal_graph", None),
        ground_stations,
        _exception_egresses(
            ground_station_satellites_in_range,
            gs_destination_candidates,
            gs_addressing,
        ),
        str(algorithm_params.get("exception_policy", "none")),
        LOCAL_DETOUR,
        state_report,
    )
    if state_report is not None and fixed_exception_report:
        # Under fixed addresses the entries that matter are the ones keyed on the
        # destination address; the station-keyed figures above describe a walk
        # these flows don't take.
        state_report.update(fixed_exception_report)
        for stale in (
            "exception_entries_one_pass",
            "exception_groups",
            "exception_hops_to_failure_mean",
            "exception_hops_to_failure_max",
        ):
            state_report.pop(stale, None)
    _report_forwarding_work(state_report, per_satellite_work, weight_model, fstate)
    log.debug(f"Calculated fstate with {len(fstate)} entries")
    return fstate


def _build_fixed_address_routes(
    address_policy: str,
    ground_stations: list[GroundStation],
    gs_candidates: list,
    satellite_addresses: dict[int, TopologicalNetworkAddress],
    neighbor_candidates: dict[int, list],
    constellation_data: ConstellationData,
    distance_mode: str,
    weight_model: dict | None,
    forwarding_guard: str,
    flow_counters: dict[str, int],
    exception_policy: str = "none",
    live_graph: nx.Graph | None = None,
    exception_report: dict | None = None,
) -> dict[tuple[int, int], dict]:
    """Select one address pair per flow and keep its destination fixed.

    A directory may return K attachment-dependent address synonyms.  The flow
    allocator applies this routing family's cost once, at ingress.  Transit
    satellites then see the selected destination address, never the K-element
    set.  The returned paths are realised forwarding walks used by the
    evaluation harness; they are not packet-carried source routes.
    """
    routes: dict[tuple[int, int], dict] = {}
    exceptions: dict[tuple[int, int], int] = {}
    toward: dict[int, dict[int, float]] = {}
    unresolved = 0
    # Satellites that learn failures by flooding know which part of the shell
    # each satellite sits in, so allocation and uplink choice skip address pairs
    # with no live path, as link-state does. Without an exception policy the
    # scheme assumes no failure knowledge and stays blind, as before.
    component = (
        _live_components(live_graph)
        if exception_policy != "none" and live_graph is not None
        else None
    )
    for src_idx, src_gs in enumerate(ground_stations):
        if src_idx >= len(gs_candidates):
            continue
        for dst_idx, dst_gs in enumerate(ground_stations):
            if src_gs.id == dst_gs.id or dst_idx >= len(gs_candidates):
                continue
            selected = resolve_flow_address_pair(
                address_policy,
                "topological",
                src_gs,
                dst_gs,
                gs_candidates[src_idx],
                gs_candidates[dst_idx],
                lambda sources, destinations: _select_topological_address_pair(
                    sources,
                    destinations,
                    satellite_addresses,
                    neighbor_candidates,
                    constellation_data,
                    distance_mode,
                    weight_model,
                    component,
                ),
                flow_counters,
            )
            if selected is None:
                continue
            src_dist, src_sat, dst_dist, dst_sat, dst_address = selected
            walk_args = (
                src_sat,
                dst_sat,
                dst_address,
                satellite_addresses,
                neighbor_candidates,
                constellation_data,
                distance_mode,
                weight_model,
                forwarding_guard,
            )
            if exception_policy == "grow" and live_graph is not None:
                path, failure = _walk_with_exception_entries(
                    walk_args, dst_sat, live_graph, exceptions, toward
                )
            else:
                path, failure = _walk_fixed_topological_address(*walk_args)
            routes[(src_gs.id, dst_gs.id)] = {
                "source_satellite": src_sat,
                "destination_satellite": dst_sat,
                "source_gsl_distance": float(src_dist),
                "destination_gsl_distance": float(dst_dist),
                "satellite_path": path,
                "failure": failure,
            }
            # A destination cut off by a partition fails correctly; only a reachable
            # one that the walk still misses counts against the exception rule.
            if failure is not None and src_sat in toward.get(dst_sat, {}):
                unresolved += 1
    if exception_report is not None and exception_policy == "grow":
        exception_report.update(
            {
                "exception_entries": float(len(exceptions)),
                "exception_satellites": float(len({sat for sat, _dst in exceptions})),
                "exception_unresolved": float(unresolved),
            }
        )
    return routes


def _live_components(live_graph: nx.Graph) -> dict[int, int]:
    """Index of the connected part of the live satellite graph each satellite is in."""
    return {
        satellite: index
        for index, part in enumerate(nx.connected_components(live_graph))
        for satellite in part
    }


def _select_topological_address_pair(
    source_candidates: list,
    destination_candidates: list,
    satellite_addresses: dict[int, TopologicalNetworkAddress],
    neighbor_candidates: dict[int, list],
    constellation_data: ConstellationData,
    distance_mode: str,
    weight_model: dict | None,
    component: dict[int, int] | None = None,
) -> tuple[float, int, float, int, TopologicalNetworkAddress] | None:
    """Cheapest (uplink, destination synonym) pair by the scheme's own estimate.

    ``component`` maps each live satellite to its connected part of the live
    graph; when given, pairs in different parts are skipped, since no walk can
    join them.
    """
    best_key: tuple[float, int, int] | None = None
    best: tuple[float, int, float, int, TopologicalNetworkAddress] | None = None
    for src_dist, src_sat, src_address in source_candidates:
        if src_sat not in satellite_addresses:
            continue
        plane_cost, sat_cost = _estimate_axis_step_costs(
            src_address, neighbor_candidates.get(src_sat, [])
        )
        for dst_dist, dst_sat, dst_address in destination_candidates:
            if dst_sat not in satellite_addresses:
                continue
            if component is not None and (
                component.get(src_sat) is None or component.get(src_sat) != component.get(dst_sat)
            ):
                continue
            route_cost = _routing_topological_distance(
                src_address,
                dst_address,
                constellation_data,
                distance_mode=distance_mode,
                plane_step_cost=plane_cost,
                sat_step_cost=sat_cost,
                weight_model=weight_model,
            )
            total = (
                _scaled_gsl_distance(src_dist, distance_mode)
                + route_cost
                + _scaled_gsl_distance(dst_dist, distance_mode)
            )
            key = (total, src_sat, dst_sat)
            if best_key is None or key < best_key:
                best_key = key
                best = (src_dist, src_sat, dst_dist, dst_sat, dst_address)
    return best


def _walk_fixed_topological_address(
    source_satellite: int,
    destination_satellite: int,
    destination_address: TopologicalNetworkAddress,
    satellite_addresses: dict[int, TopologicalNetworkAddress],
    neighbor_candidates: dict[int, list],
    constellation_data: ConstellationData,
    distance_mode: str,
    weight_model: dict | None,
    forwarding_guard: str,
    exceptions: dict[tuple[int, int], int] | None = None,
) -> tuple[list[int], str | None]:
    """Realise rule forwarding for one already-selected destination address.

    ``exceptions`` maps (satellite, destination satellite) to the next hop an
    exception entry installed there; a satellite holding one follows it
    instead of the rule.
    """
    current = source_satellite
    path = [current]
    seen = {current}
    hop_budget = len(satellite_addresses) + 1

    def fixed_potential(satellite_id: int, _unused_gs_idx: int) -> float:
        address = satellite_addresses.get(satellite_id)
        if address is None:
            return float("inf")
        return _routing_topological_distance(
            address,
            destination_address,
            constellation_data,
            distance_mode=distance_mode,
            weight_model=weight_model,
        )

    while current != destination_satellite and len(path) <= hop_budget:
        entry = exceptions.get((current, destination_satellite)) if exceptions else None
        if entry is not None:
            if entry in seen:
                path.append(entry)
                return path, "loop"
            path.append(entry)
            seen.add(entry)
            current = entry
            continue
        current_address = satellite_addresses.get(current)
        if current_address is None:
            return path, "dead_end"
        neighbours = _admissible_neighbours(
            current,
            neighbor_candidates.get(current, []),
            0,
            fixed_potential,
            forwarding_guard,
        )
        decision, _distance = _get_next_hop_decision_topological(
            current,
            current_address,
            destination_address,
            neighbours,
            -1,
            constellation_data,
            distance_mode,
            weight_model,
        )
        if decision is None or (isinstance(decision, tuple) and decision[:1] == ("GSL",)):
            return path, "dead_end"
        if _is_local_detour_entry(decision):
            physical_hops = [decision[1], decision[2], decision[3]]
        else:
            physical_hops = [
                neighbor_id
                for neighbor_id, interface, _address, _weight in neighbours
                if interface == decision
            ][:1]
        if not physical_hops:
            return path, "dead_end"
        for next_satellite in physical_hops:
            if next_satellite in seen:
                path.append(next_satellite)
                return path, "loop"
            path.append(next_satellite)
            seen.add(next_satellite)
        current = physical_hops[-1]

    if current == destination_satellite:
        return path, None
    return path, "hop_limit"


def _walk_with_exception_entries(
    walk_args: tuple,
    destination_satellite: int,
    live_graph: nx.Graph,
    exceptions: dict[tuple[int, int], int],
    toward: dict[int, dict[int, float]],
) -> tuple[list[int], str | None]:
    """Walk toward a fixed address, growing exception entries where it breaks.

    The same rule as ``exception_policy: grow`` under station addressing, keyed
    on the destination address instead of the station: an entry goes only to
    the satellite where a walk dead-ends, or to the first satellite of a loop
    that has none yet, pointing along the shortest live path to the
    destination satellite; then the walk restarts. Entries are shared by every
    flow toward that address in the snapshot. Each restart adds an entry, so
    the walk ends within as many restarts as there are satellites.
    """
    if destination_satellite not in toward:
        toward[destination_satellite] = nx.single_source_dijkstra_path_length(
            live_graph, destination_satellite, weight="weight"
        )
    distance = toward[destination_satellite]

    def next_on_shortest_path(satellite: int) -> int | None:
        if satellite not in distance:
            return None
        options = [
            (live_graph.edges[satellite, n].get("weight", 1.0) + distance[n], n)
            for n in live_graph.neighbors(satellite)
            if n in distance and distance[n] < distance[satellite]
        ]
        return min(options)[1] if options else None

    for _attempt in range(live_graph.number_of_nodes() + 1):
        path, failure = _walk_fixed_topological_address(*walk_args, exceptions=exceptions)
        if failure is None:
            return path, None
        if failure == "loop":
            first = path.index(path[-1])
            cycle = path[first:-1]
            breaking = next(
                (s for s in cycle if (s, destination_satellite) not in exceptions), None
            )
        else:
            breaking = path[-1]
            if (breaking, destination_satellite) in exceptions:
                breaking = None
        hop = next_on_shortest_path(breaking) if breaking is not None else None
        if hop is None:
            return path, failure
        exceptions[(breaking, destination_satellite)] = hop
    return path, "hop_limit"


def _set_sixgrupa_addresses_to_all_nodes(
    topology: LEOTopology, constellation_data: ConstellationData
):
    """
    Set 6GRUPA addresses to all satellite nodes in the topology.
    """
    log.debug("Setting 6GRUPA addresses to all satellite nodes")
    for satellite in topology.get_satellites():
        try:
            address = _get_satellite_address(topology, satellite.id, constellation_data)
            satellite.sixgrupa_addr = address
            log.debug(f"Assigned 6G-RUPA address {address} to satellite {satellite.id}")
        except Exception as e:
            log.error(f"Failed to assign 6G-RUPA address to satellite {satellite.id}: {e}")


def _get_satellite_address(
    topology: LEOTopology,
    satellite_id: int,
    constellation_data: ConstellationData,
) -> TopologicalNetworkAddress:
    satellite = topology.get_satellite(satellite_id)
    if hasattr(satellite, "sixgrupa_addr") and satellite.sixgrupa_addr is not None:
        cached_address = satellite.sixgrupa_addr
        if (
            cached_address.shell_id == 0
            and cached_address.plane_id < constellation_data.n_orbits
            and cached_address.sat_index < constellation_data.n_sats_per_orbit
        ):
            return cached_address

    try:
        return TopologicalNetworkAddress.set_address_from_constellation(
            satellite_id,
            constellation_data.n_orbits,
            constellation_data.n_sats_per_orbit,
        )
    except ValueError:
        try:
            sorted_satellite_ids = sorted(sat.id for sat in topology.get_satellites())
            logical_satellite_index = sorted_satellite_ids.index(satellite_id)
            return TopologicalNetworkAddress.set_address_from_constellation(
                logical_satellite_index,
                constellation_data.n_orbits,
                constellation_data.n_sats_per_orbit,
            )
        except (ValueError, IndexError):
            return TopologicalNetworkAddress.set_address_from_orbital_parameters(satellite_id)


def _detect_gsl_changes(
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
) -> dict[int, tuple[Optional[int], Optional[int]]]:
    """
    Detect GSL changes by comparing current attachments with previous ones.

    Args:
        ground_stations: List of ground stations
        ground_station_satellites_in_range: Current satellite attachments per GS

    Returns:
        dict: Maps GS index to (previous_sat_id, new_sat_id) for changed GSLs
    """
    gsl_changes = {}

    for gs_idx, gs in enumerate(ground_stations):
        if gs_idx >= len(ground_station_satellites_in_range):
            continue

        gsl_satellites = ground_station_satellites_in_range[gs_idx]
        current_sat_id = None

        # Get the best (closest) satellite currently attached to this GS
        if gsl_satellites:
            # Sort by distance and take the closest one
            sorted_sats = sorted(gsl_satellites, key=lambda x: x[0])
            if sorted_sats:
                current_sat_id = sorted_sats[0][1]  # (distance, sat_id)

        # Check if attachment has changed
        previous_sat_id = gs.previous_attached_satellite_id

        if previous_sat_id != current_sat_id:
            log.debug(f"GSL change detected for GS {gs.id}: {previous_sat_id} -> {current_sat_id}")
            gsl_changes[gs_idx] = (previous_sat_id, current_sat_id)

            # Update the stored previous satellite ID
            gs.previous_attached_satellite_id = current_sat_id

    return gsl_changes


def _update_gs_attachment_sets(
    ground_stations: list[GroundStation],
    ground_station_satellites_in_range: list,
    attachment_count: int,
) -> dict[str, int]:
    """Track changes to the K satellite addresses advertised by each station.

    A station renumbers when its advertised set changes. The first snapshot
    establishes the set and costs no update, matching the K=1 accounting used
    before multihoming was introduced.
    """
    changed_stations = 0
    additions = 0
    removals = 0
    for gs_idx, gs in enumerate(ground_stations):
        visible = (
            ground_station_satellites_in_range[gs_idx]
            if gs_idx < len(ground_station_satellites_in_range)
            else []
        )
        current = tuple(
            item[1]
            for item in sorted(visible, key=lambda item: (item[0], item[1]))[:attachment_count]
        )
        previous = gs.previous_advertised_satellite_ids
        if previous is not None:
            previous_set = set(previous)
            current_set = set(current)
            if previous_set != current_set:
                changed_stations += 1
                additions += len(current_set - previous_set)
                removals += len(previous_set - current_set)
        gs.previous_advertised_satellite_ids = current
    return {
        "stations": changed_stations,
        "additions": additions,
        "removals": removals,
    }


def _assign_gs_address_from_satellite(
    gs: GroundStation,
    satellite_id: int,
    gs_subnet_index: int,
    topology: LEOTopology,
    constellation_data: ConstellationData,
) -> Optional[TopologicalNetworkAddress]:
    """
    Assign a 6grupa address to a ground station based on its attached satellite.

    Args:
        gs: The ground station
        satellite_id: ID of the satellite this GS is attached to
        gs_subnet_index: Subnet index for this GS under the satellite
        topology: Topology containing satellites

    Returns:
        TopologicalNetworkAddress for the ground station
    """
    try:
        # Get the satellite's 6grupa address
        satellite = topology.get_satellite(satellite_id)
        if not hasattr(satellite, "sixgrupa_addr") or not satellite.sixgrupa_addr:
            sat_address = _get_satellite_address(topology, satellite_id, constellation_data)
            satellite.sixgrupa_addr = sat_address
        else:
            sat_address = satellite.sixgrupa_addr

        # Create GS address with same shell, plane, sat_index but different subnet_index
        gs_address = TopologicalNetworkAddress(
            shell_id=sat_address.shell_id,
            plane_id=sat_address.plane_id,
            sat_index=sat_address.sat_index,
            subnet_index=gs_subnet_index,
        )

        log.debug(
            f"Assigned 6grupa address {gs_address} to GS {gs.id} "
            f"(attached to satellite {satellite_id})"
        )

        return gs_address

    except Exception as e:
        log.error(f"Failed to assign 6grupa address to GS {gs.id}: {e}")
        return None


def _perform_renumbering_for_gs(
    gs: GroundStation,
    prev_sat_id: Optional[int],
    curr_sat_id: Optional[int],
    topology: LEOTopology,
    constellation_data: ConstellationData | None = None,
):
    """
    Perform renumbering when a ground station's satellite links change.

    Updates the GS's 6grupa address to match the new satellite attachment.
    """
    log.debug(f"Renumbering for GS {gs.id} from satellite {prev_sat_id} to {curr_sat_id}")

    if constellation_data is None:
        constellation_data = topology.constellation_data

    if curr_sat_id is not None:
        # Get the satellite's 6grupa address to match coordinates
        try:
            satellite = topology.get_satellite(curr_sat_id)
            if satellite.sixgrupa_addr is None:
                satellite.sixgrupa_addr = _get_satellite_address(
                    topology,
                    curr_sat_id,
                    constellation_data,
                )
            sat_addr = satellite.sixgrupa_addr
        except Exception:
            log.error(f"Failed to get satellite {curr_sat_id} address for GS {gs.id} renumbering")
            return

        # Count how many GSs are already attached to this satellite to assign unique subnet_index
        # Look for GSs that have 6grupa addresses matching this satellite's coordinates
        used_subnet_indices = set()
        for other_gs in topology.get_ground_stations():
            if (
                other_gs != gs
                and hasattr(other_gs, "sixgrupa_addr")
                and other_gs.sixgrupa_addr is not None
            ):
                other_addr = other_gs.sixgrupa_addr
                # Check if this GS is attached to the same satellite (same coordinates)
                if (
                    other_addr.shell_id == sat_addr.shell_id
                    and other_addr.plane_id == sat_addr.plane_id
                    and other_addr.sat_index == sat_addr.sat_index
                ):
                    used_subnet_indices.add(other_addr.subnet_index)

        # Find the next available subnet_index > 0 (0 is reserved for satellite)
        subnet_index = 1
        while subnet_index in used_subnet_indices:
            subnet_index += 1

        # Assign new address based on the current satellite
        gs_address = _assign_gs_address_from_satellite(
            gs,
            curr_sat_id,
            subnet_index,
            topology,
            constellation_data,
        )
        if gs_address:
            gs.sixgrupa_addr = gs_address
            gs.previous_attached_satellite_id = curr_sat_id  # Update the previous attachment
            log.info(f"Renumbered GS {gs.id} to new address {gs_address}")
        else:
            log.warning(f"Renumbering GS {gs.id} failed, address assignment returned None")
    else:
        # No current satellite - clear the address
        gs.sixgrupa_addr = None
        gs.previous_attached_satellite_id = None  # Clear previous attachment
        log.debug(f"GS {gs.id} detached, cleared 6grupa address")


def _fill_forwarding_tables_in_every_satellite(
    satellite_node_ids: list[int],
    satellite_only_subgraph: nx.Graph,
    topology_with_isls: LEOTopology,
    constellation_data: ConstellationData,
):
    """
    Fill forwarding tables in every satellite based on neighbor 6grupa addresses.
    """
    for satellite_id in satellite_node_ids:
        try:
            satellite = topology_with_isls.get_satellite(satellite_id)
            for neighbor_id in satellite_only_subgraph.neighbors(satellite_id):
                interface = topology_with_isls.sat_neighbor_to_if.get((satellite_id, neighbor_id))
                if interface is not None:
                    try:
                        neighbor_address = _get_satellite_address(
                            topology_with_isls,
                            neighbor_id,
                            constellation_data,
                        )
                        satellite.forwarding_table[neighbor_address.to_integer()] = interface
                        log.debug(
                            f"Forwarding entry added for satellite {satellite_id} to neighbor {neighbor_id}: "
                            f"address {neighbor_address}, interface {interface}"
                        )
                    except Exception as e:
                        log.warning(
                            f"Failed to add forwarding entry for satellite {satellite_id} -> {neighbor_id}: {e}"
                        )
        except Exception as e:
            log.error(f"Failed to process satellite {satellite_id} for forwarding table: {e}")


def _calculate_sat_to_gs_fstate(
    topology_with_isls,
    ground_stations,
    nodelist,
    satellite_addresses,
    neighbor_candidates,
    gs_destination_candidates,
    fstate,
    constellation_data,
    distance_mode: str,
    weight_model: dict | None = None,
    per_satellite_work: dict | None = None,
    potential: "_EgressPotential | None" = None,
    forwarding_guard: str = "none",
    selected_egresses: dict[tuple[int, int], int] | None = None,
):
    """
    Calculate satellite-to-ground-station forwarding state using topological routing.

    This implements the core topological routing logic:
    - For each satellite, determine next hop to reach each ground station
    - Use topological distance calculations based on 6grupa addresses
    """
    log.debug("Calculating satellite-to-GS forwarding state using topological routing")

    for curr_sat_id in nodelist:
        try:
            topology_with_isls.get_satellite(curr_sat_id)
        except KeyError:
            log.error(f"Could not find satellite object {curr_sat_id}")
            continue

        curr_satellite_address = satellite_addresses.get(curr_sat_id)
        if curr_satellite_address is None:
            log.warning(f"Satellite {curr_sat_id} has no 6grupa address assigned")
            continue

        for gs_idx, dst_gs in enumerate(ground_stations):
            dst_gs_node_id = dst_gs.id
            if gs_idx >= len(gs_destination_candidates):
                continue

            possible_dst_sats = gs_destination_candidates[gs_idx]
            if not possible_dst_sats:
                continue

            log.debug(
                f"FSTATE: Sat {curr_sat_id} -> GS {dst_gs.id}. Visible sats: {[sat_id for _, sat_id, _ in possible_dst_sats]}"
            )

            # Find the best destination satellite using topological distance
            best_destination_address = None
            best_destination_sat_id = None
            best_total_distance = float("inf")
            heuristic_costs = _estimate_axis_step_costs(
                curr_satellite_address,
                neighbor_candidates.get(curr_sat_id, []),
            )

            for dist_gs_to_sat_m, visible_sat_id, dest_sat_address in possible_dst_sats:
                try:
                    topo_distance = _routing_topological_distance(
                        curr_satellite_address,
                        dest_sat_address,
                        constellation_data,
                        distance_mode=distance_mode,
                        plane_step_cost=heuristic_costs[0],
                        sat_step_cost=heuristic_costs[1],
                        weight_model=weight_model,
                    )
                    total_distance = topo_distance + _scaled_gsl_distance(
                        dist_gs_to_sat_m,
                        distance_mode,
                    )

                    if total_distance < best_total_distance:
                        best_total_distance = total_distance
                        best_destination_address = dest_sat_address
                        best_destination_sat_id = visible_sat_id
                except Exception as e:
                    log.warning(f"Failed to process destination satellite {visible_sat_id}: {e}")
                    continue

            if best_destination_address is None:
                continue
            if selected_egresses is not None and best_destination_sat_id is not None:
                selected_egresses[(curr_sat_id, dst_gs_node_id)] = best_destination_sat_id

            if per_satellite_work is not None:
                # What this satellite alone evaluates for this destination: one
                # distance per visible egress to pick the destination satellite,
                # one for its own position, and one per neighbour. The simulator
                # computes every satellite's state in a single process, so the
                # shared memo table spans the whole constellation; on board, a
                # satellite only ever evaluates and caches its own pairs.
                neighbours = len(neighbor_candidates.get(curr_sat_id, []))
                work = per_satellite_work.setdefault(
                    curr_sat_id, {"decisions": 0, "evaluations": 0, "pairs": set()}
                )
                work["decisions"] += 1
                # Under the progress guard the satellite also evaluates each
                # neighbour's potential, one distance per visible egress.
                guard_evaluations = (
                    len(possible_dst_sats) * neighbours * (forwarding_guard != "none")
                )
                work["evaluations"] += len(possible_dst_sats) + 1 + neighbours + guard_evaluations
                dest_key = (
                    best_destination_address.get_satellite_address().plane_id,
                    best_destination_address.get_satellite_address().sat_index,
                )
                for neighbor_id, _iface, neighbor_address, _w in neighbor_candidates.get(
                    curr_sat_id, []
                ):
                    neighbour_sat = neighbor_address.get_satellite_address()
                    work["pairs"].add((neighbour_sat.plane_id, neighbour_sat.sat_index) + dest_key)

            _install_next_hop(
                fstate,
                curr_sat_id,
                curr_satellite_address,
                best_destination_address,
                gs_idx,
                dst_gs_node_id,
                neighbor_candidates.get(curr_sat_id, []),
                constellation_data,
                distance_mode,
                weight_model,
                potential,
                forwarding_guard,
                per_satellite_work,
            )


LOCAL_REPAIRS = ("none", "square")

# Forwarding entry for a detour to the routing-level next hop:
# (LOCAL_DETOUR, first relay, second relay, target).
LOCAL_DETOUR = "DETOUR"


def _is_local_detour_entry(entry: object) -> bool:
    return isinstance(entry, tuple) and len(entry) == 4 and entry[0] == LOCAL_DETOUR


def _with_local_detours(
    neighbor_candidates: dict,
    topology_with_isls: LEOTopology,
    satellite_only_subgraph: nx.Graph,
    satellite_addresses: dict,
    algorithm_params: dict,
    distance_mode: str,
) -> dict:
    """Neighbour candidates, plus a detour to each nominal neighbour cut off by a failed link.

    Routing in RINA is two steps: choose the next hop, then choose a path to it
    (Reference Model Part 3-1, section 3.2). Under ``local_repair="square"`` a
    satellite X whose ISL to a nominal neighbour Y has failed keeps Y as a
    candidate while Y is still three live hops away, and reaches it over the
    shortest such path X -> B -> C -> Y, on a +Grid the other sides of a grid
    square. The path sits below the routing decision, as a local-scope lower layer
    would provide it: the decision still names Y, so the progress guard's argument
    is unaffected. It needs the state of links within two hops, a fixed
    neighbourhood that does not grow with the constellation.
    """
    repair = str(algorithm_params.get("local_repair", "none"))
    if repair not in LOCAL_REPAIRS:
        raise ValueError(f"Unknown local repair: {repair}")
    if repair == "square" and distance_mode not in EVALUATOR_INDEPENDENT_MODES:
        raise ValueError(
            f"Local repair needs an evaluator-independent distance, not {distance_mode}"
        )
    nominal_graph = getattr(topology_with_isls, "nominal_graph", None)
    if repair == "none" or nominal_graph is None or nominal_graph is topology_with_isls.graph:
        return neighbor_candidates

    augmented = {sat_id: list(candidates) for sat_id, candidates in neighbor_candidates.items()}
    for sat_id, candidates in augmented.items():
        for target_id in _failed_nominal_neighbours(nominal_graph, satellite_only_subgraph, sat_id):
            detour = _shortest_three_hop_detour(satellite_only_subgraph, sat_id, target_id)
            if detour is None or target_id not in satellite_addresses:
                continue
            first_relay, second_relay, length = detour
            candidates.append(
                (
                    target_id,
                    (LOCAL_DETOUR, first_relay, second_relay, target_id),
                    satellite_addresses[target_id],
                    length,
                )
            )
    return augmented


def _failed_nominal_neighbours(
    nominal_graph: nx.Graph, live_subgraph: nx.Graph, satellite_id: int
) -> list[int]:
    """Satellites adjacent in the failure-free graph whose link to this one is down."""
    if not nominal_graph.has_node(satellite_id):
        return []
    return sorted(
        neighbour
        for neighbour in nominal_graph.neighbors(satellite_id)
        if live_subgraph.has_node(neighbour) and not live_subgraph.has_edge(satellite_id, neighbour)
    )


def _shortest_three_hop_detour(
    live_subgraph: nx.Graph, source_id: int, target_id: int
) -> tuple[int, int, float] | None:
    """Shortest live path source -> B -> C -> target through two distinct relays."""
    if not live_subgraph.has_node(source_id) or not live_subgraph.has_node(target_id):
        return None
    best: tuple[int, int, float] | None = None
    for first_relay in sorted(live_subgraph.neighbors(source_id)):
        if first_relay == target_id:
            continue
        for second_relay in sorted(live_subgraph.neighbors(target_id)):
            if second_relay in (source_id, first_relay):
                continue
            if not live_subgraph.has_edge(first_relay, second_relay):
                continue
            length = (
                _edge_length(live_subgraph, source_id, first_relay)
                + _edge_length(live_subgraph, first_relay, second_relay)
                + _edge_length(live_subgraph, second_relay, target_id)
            )
            if best is None or length < best[2]:
                best = (first_relay, second_relay, length)
    return best


def _edge_length(graph: nx.Graph, node_a: int, node_b: int) -> float:
    return float(graph.edges[node_a, node_b].get("weight", 1.0))


FORWARDING_GUARDS = ("none", "progress")

# Modes whose distance depends only on the two addresses. The lookahead modes
# estimate step costs from the evaluating satellite's own links, so two
# satellites can disagree on a third satellite's distance.
EVALUATOR_INDEPENDENT_MODES = ("torus_unit", "torus_weighted_pivot")


def _resolve_forwarding_guard(algorithm_params: dict, distance_mode: str) -> str:
    """Validated forwarding guard for this run.

    The progress guard's loop-freedom argument needs every satellite to agree on
    each other's potential, which holds only when the distance depends on the two
    addresses alone, so the guard refuses the lookahead modes.
    """
    guard = str(algorithm_params.get("forwarding_guard", "none"))
    if guard not in FORWARDING_GUARDS:
        raise ValueError(f"Unknown forwarding guard: {guard}")
    if guard == "progress" and distance_mode not in EVALUATOR_INDEPENDENT_MODES:
        raise ValueError(
            f"The progress guard needs an evaluator-independent distance, not {distance_mode}"
        )
    return guard


class _EgressPotential:
    """Potential of a satellite toward one ground station, memoised per snapshot.

    Phi(S) = min over the satellites e that can see the destination of
    d(S, e) + l_e, the same quantity a satellite minimises when it selects its
    egress. With an evaluator-independent distance every satellite computes the
    same Phi for a given satellite, so the potential defines one order per
    destination. A walk that strictly descends (Phi, satellite id) at every hop
    cannot revisit a satellite: it reaches an egress, or stops at a local minimum,
    within as many hops as there are satellites.
    """

    def __init__(
        self,
        satellite_addresses: dict,
        gs_destination_candidates: list,
        constellation_data: ConstellationData,
        distance_mode: str,
        weight_model: dict | None,
    ) -> None:
        self._addresses = satellite_addresses
        self._candidates = gs_destination_candidates
        self._constellation_data = constellation_data
        self._distance_mode = distance_mode
        self._weight_model = weight_model
        self._cache: dict[tuple[int, int], float] = {}

    def __call__(self, satellite_id: int, gs_idx: int) -> float:
        key = (satellite_id, gs_idx)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = float("inf")
        address = self._addresses.get(satellite_id)
        if address is not None and gs_idx < len(self._candidates):
            for dist_gs_to_sat_m, _visible_sat_id, egress_address in self._candidates[gs_idx]:
                distance = _routing_topological_distance(
                    address,
                    egress_address,
                    self._constellation_data,
                    distance_mode=self._distance_mode,
                    weight_model=self._weight_model,
                )
                value = min(
                    value, distance + _scaled_gsl_distance(dist_gs_to_sat_m, self._distance_mode)
                )
        self._cache[key] = value
        return value


def _admissible_neighbours(
    curr_sat_id: int,
    neighbours: list,
    gs_idx: int,
    potential: Callable[[int, int], float] | None,
    forwarding_guard: str,
) -> list:
    """Neighbours a hop may use toward one ground station.

    Without a guard, every live neighbour. Under the progress guard, only those
    strictly below the current satellite in (potential, satellite id), with a
    finite potential. Ties on the potential are broken by satellite id so the
    order stays strict when distances are integers, as in the unit mode.
    """
    if forwarding_guard == "none":
        return neighbours
    if potential is None:
        raise ValueError("The progress guard needs a potential")
    own = (potential(curr_sat_id, gs_idx), curr_sat_id)
    return [
        candidate
        for candidate in neighbours
        if math.isfinite(potential(candidate[0], gs_idx))
        and (potential(candidate[0], gs_idx), candidate[0]) < own
    ]


def _install_next_hop(
    fstate: dict,
    curr_sat_id: int,
    curr_satellite_address: TopologicalNetworkAddress,
    destination_address: TopologicalNetworkAddress,
    gs_idx: int,
    dst_gs_node_id: int,
    neighbours: list,
    constellation_data: ConstellationData,
    distance_mode: str,
    weight_model: dict | None,
    potential: Callable[[int, int], float] | None,
    forwarding_guard: str,
    per_satellite_work: dict | None,
) -> None:
    """Choose and install one satellite's next hop toward one ground station.

    Under the progress guard, a satellite none of whose neighbours lowers the
    potential is at a local minimum. It installs nothing and counts a forwarding
    exception: the point at which an exception entry would take over.
    """
    candidates = _admissible_neighbours(
        curr_sat_id, neighbours, gs_idx, potential, forwarding_guard
    )
    try:
        next_hop_decision, _distance = _get_next_hop_decision_topological(
            curr_sat_id,
            curr_satellite_address,
            destination_address,
            candidates,
            dst_gs_node_id,
            constellation_data,
            distance_mode,
            weight_model,
        )
    except Exception as e:
        log.warning(
            f"Failed to create routing decision for satellite {curr_sat_id} to GS {dst_gs_node_id}: {e}"
        )
        return
    if next_hop_decision is not None:
        fstate[(curr_sat_id, dst_gs_node_id)] = next_hop_decision
        return
    if forwarding_guard != "none" and per_satellite_work is not None:
        work = per_satellite_work.setdefault(
            curr_sat_id, {"decisions": 0, "evaluations": 0, "pairs": set()}
        )
        # A satellite with no live link is down rather than at a local minimum,
        # so its decisions are counted apart and exception state reflects live
        # satellites only.
        outcome = "exceptions" if neighbours else "isolated"
        work[outcome] = work.get(outcome, 0) + 1


def _get_next_hop_decision_topological(
    curr_sat_id: int,
    curr_satellite_address: TopologicalNetworkAddress,
    destination_address: TopologicalNetworkAddress,
    neighbor_candidates: list[tuple[int, int, TopologicalNetworkAddress, float]],
    dst_gs_node_id: int,
    constellation_data: ConstellationData,
    distance_mode: str,
    weight_model: dict | None = None,
) -> tuple:
    """
    Determine the next hop decision using topological routing.

    This implements topological routing where each satellite computes the next hop
    by looking at neighbor's 6grupa addresses and performing a distance function.

    Args:
        curr_sat_id: Current satellite ID
        curr_satellite: Current satellite object
        destination_address: 6grupa address of the destination satellite
        sat_subgraph: Satellite-only subgraph
        sat_neighbor_to_if: Interface mapping
        topology_with_isls: Topology object
        dst_gs_node_id: Destination ground station ID

    Returns:
        tuple: (next_hop_interface_or_decision, distance) or (None, inf) if no path
    """
    plane_step_cost, sat_step_cost = _estimate_axis_step_costs(
        curr_satellite_address,
        neighbor_candidates,
    )
    my_distance_to_dest = _routing_topological_distance(
        curr_satellite_address,
        destination_address,
        constellation_data,
        distance_mode=distance_mode,
        plane_step_cost=plane_step_cost,
        sat_step_cost=sat_step_cost,
        weight_model=weight_model,
    )

    # Check if we are already at the destination satellite
    if my_distance_to_dest == 0.0:
        # Direct GSL connection - use GSL interface
        log.debug(f"Direct GSL path: Sat {curr_sat_id} -> GS {dst_gs_node_id}")
        return ("GSL", dst_gs_node_id), 0.0

    # Find the best neighbor using topological distance
    best_neighbor_id = None
    best_distance = float("inf")
    best_interface = None
    best_tie_break = None

    # Check all neighbors
    for neighbor_id, interface, neighbor_address, edge_weight in neighbor_candidates:
        try:
            neighbor_distance_to_dest = _routing_topological_distance(
                neighbor_address,
                destination_address,
                constellation_data,
                distance_mode=distance_mode,
                plane_step_cost=plane_step_cost,
                sat_step_cost=sat_step_cost,
                weight_model=weight_model,
            )
            candidate_score = _neighbor_candidate_score(
                edge_weight=edge_weight,
                neighbor_distance_to_dest=neighbor_distance_to_dest,
                distance_mode=distance_mode,
            )
            tie_break = _routing_tie_break_tuple(
                neighbor_address,
                destination_address,
                constellation_data,
            )
            strict_progress = _routing_strict_progress_tuple(
                neighbor_address,
                destination_address,
            )

            # If this neighbor is closer to destination, consider it
            if candidate_score < best_distance:
                best_distance = candidate_score
                best_neighbor_id = neighbor_id
                best_interface = interface
                best_tie_break = tie_break
            elif candidate_score == best_distance:
                if best_tie_break is None or tie_break < best_tie_break:
                    best_neighbor_id = neighbor_id
                    best_interface = interface
                    best_tie_break = tie_break
            elif neighbor_distance_to_dest == my_distance_to_dest:
                my_strict_progress = _routing_strict_progress_tuple(
                    curr_satellite_address,
                    destination_address,
                )
                if strict_progress < my_strict_progress and (
                    best_neighbor_id is None or best_tie_break is None or tie_break < best_tie_break
                ):
                    best_neighbor_id = neighbor_id
                    best_interface = interface
                    best_tie_break = tie_break

        except Exception:
            # Skip this neighbor if we can't get its address
            continue

    if best_neighbor_id is not None and best_distance < float("inf"):
        log.debug(
            f"Topological routing: Sat {curr_sat_id} -> {best_neighbor_id} (if {best_interface}) "
            f"towards destination with distance {best_distance}"
        )
        return best_interface, best_distance

    # No better neighbor found - this shouldn't happen in a connected graph
    log.warning(f"No better neighbor found for satellite {curr_sat_id} to reach destination")
    return None, float("inf")


def _routing_topological_distance(
    source_address: TopologicalNetworkAddress,
    destination_address: TopologicalNetworkAddress,
    constellation_data: ConstellationData,
    distance_mode: str = "torus_weighted_lookahead",
    plane_step_cost: float = 1.0,
    sat_step_cost: float = 1.0,
    weight_model: dict | None = None,
) -> float:
    if distance_mode == "torus_unit":
        return torus_topological_distance(
            source_address,
            destination_address,
            plane_modulus=constellation_data.n_orbits,
            sat_modulus=constellation_data.n_sats_per_orbit,
            plane_weight=1.0,
            sat_weight=1.0,
            shell_penalty=1000.0,
        )
    if distance_mode == "torus_weighted_pivot" and weight_model is not None:
        return _torus_weighted_pivot_distance(
            source_address,
            destination_address,
            weight_model,
            shell_penalty=1000.0,
        )
    return weighted_torus_progress_distance(
        source_address,
        destination_address,
        plane_modulus=constellation_data.n_orbits,
        sat_modulus=constellation_data.n_sats_per_orbit,
        plane_step_cost=plane_step_cost,
        sat_step_cost=sat_step_cost,
        shell_penalty=1000.0,
    )


GS_ADDRESSING = ("visibility", "attachment")


def _select_gs_attachments(
    gs_destination_candidates: list,
    attachment_count: int = 1,
    attachment_policy: str = "independent",
) -> list:
    """Reduce each ground station's egress candidates to its K attachments.

    Under ``attachment`` addressing a ground station's 6G-RUPA address names the
    satellite it is attached to, so the destination a packet carries is that one
    satellite rather than the ground station itself. A forwarding satellite then
    has nothing to choose and nothing to know about where the ground station sits
    on the surface: it forwards toward the address. The attachment is the live
    visible satellite with the shortest ground link, the same rule
    ``_detect_gsl_changes`` uses to decide when the address has to change, and
    failed satellites have already left the visibility list, so a dead attachment
    is replaced at the next snapshot rather than stranding the ground station.

    Under ``visibility`` addressing the address is stable and every satellite
    minimises over all visible egresses instead, which needs the ground station's
    position on board.
    """
    selected, _stats = select_multihoming_attachments(
        gs_destination_candidates,
        attachment_count,
        attachment_policy,
    )
    return selected


def _exception_egresses(
    ground_station_satellites_in_range: list,
    gs_destination_candidates: list,
    gs_addressing: str,
) -> list:
    """Satellites an exception entry may deliver through, per ground station.

    Exceptions must end where the rule does. Under ``attachment`` addressing that
    is the attached satellite alone: the station holds no ground link to any other,
    so an entry delivering through another visible satellite would use a link that
    does not exist. Under ``visibility`` every visible satellite is an egress.
    """
    if gs_addressing != "attachment":
        return ground_station_satellites_in_range
    return [
        [(dist_gs_to_sat_m, sat_id) for dist_gs_to_sat_m, sat_id, _address in candidates]
        for candidates in gs_destination_candidates
    ]


def _build_gs_destination_candidates(
    ground_station_satellites_in_range: list,
    satellite_addresses: dict,
    gs_addressing: str,
    gs_attachment_count: int = 1,
    gs_attachment_policy: str = "independent",
) -> list:
    """Egress candidates per ground station, under the chosen addressing policy."""
    if gs_addressing not in GS_ADDRESSING:
        raise ValueError(
            f"Unknown gs_addressing {gs_addressing!r}, expected one of {GS_ADDRESSING}"
        )
    candidates_per_gs = [
        [
            (dist_gs_to_sat_m, visible_sat_id, satellite_addresses[visible_sat_id])
            for dist_gs_to_sat_m, visible_sat_id in visible
            if visible_sat_id in satellite_addresses
        ]
        for visible in ground_station_satellites_in_range
    ]
    if gs_addressing == "attachment":
        return _select_gs_attachments(
            candidates_per_gs,
            gs_attachment_count,
            gs_attachment_policy,
        )
    return candidates_per_gs


GEOMETRY_SOURCES = ("observed", "nominal", "derived")


def _geometry_subgraph(
    topology_with_isls: LEOTopology,
    satellite_node_ids: list[int],
    satellite_only_subgraph: nx.Graph,
    geometry_source: str,
    satellite_addresses: dict[int, TopologicalNetworkAddress] | None = None,
    constellation_data: ConstellationData | None = None,
    time_s: float = 0.0,
) -> nx.Graph:
    """Graph the pivot geometry is built from.

    ``observed`` is the snapshot graph as routed, so an injected link failure
    reaches every satellite's distance estimates at once: global failure
    knowledge the design never distributes. ``nominal`` is the failure-free
    graph with its SGP4-measured lengths, as if every satellite knew every
    ISL's length. ``derived`` keeps the failure-free wiring but computes each
    length from the shell's Walker constants and the clock, which is all a
    satellite needs to hold. Next hops always consider only live neighbours,
    over their measured first-hop lengths.
    """
    if geometry_source not in GEOMETRY_SOURCES:
        raise ValueError(f"Unknown geometry source: {geometry_source}")
    nominal_graph = getattr(topology_with_isls, "nominal_graph", None)
    if geometry_source == "observed":
        return satellite_only_subgraph
    wiring = (
        satellite_only_subgraph
        if nominal_graph is None
        else nominal_graph.subgraph(satellite_node_ids)
    )
    if geometry_source == "nominal":
        return wiring
    walker = getattr(constellation_data, "walker", None)
    if walker is None or satellite_addresses is None:
        raise ValueError("geometry_source=derived needs the shell's Walker constants")
    return _derived_geometry_graph(wiring, satellite_addresses, walker, time_s)


def _derived_geometry_graph(
    wiring: nx.Graph,
    satellite_addresses: dict[int, TopologicalNetworkAddress],
    walker,
    time_s: float,
) -> nx.Graph:
    """The designed ISLs, each weighted by its Walker-derived length."""
    derived = nx.Graph()
    derived.add_nodes_from(wiring.nodes())
    for sat_a_id, sat_b_id in wiring.edges():
        addr_a = satellite_addresses.get(sat_a_id)
        addr_b = satellite_addresses.get(sat_b_id)
        if addr_a is None or addr_b is None:
            continue
        a = addr_a.get_satellite_address()
        b = addr_b.get_satellite_address()
        length = walker.distance_m((a.plane_id, a.sat_index), (b.plane_id, b.sat_index), time_s)
        derived.add_edge(sat_a_id, sat_b_id, weight=length)
    return derived


# Planes, satellites per plane, inclination, mean motion, node spread, phasing
# and epoch: what a satellite holds to derive every ISL length in its shell.
WALKER_CONSTANT_COUNT = 7


def _describe_geometry_source(
    geometry_source: str, constellation_data: ConstellationData | None
) -> dict:
    """What a satellite must hold to know the geometry, apart from any cache.

    Under ``observed`` and ``nominal`` the estimator uses every ISL length in
    the shell, which would have to be measured and flooded: one entry per
    rail and per rung. Under ``derived`` it holds the Walker constants only.
    The path-cost tables built from either are a cache on top.
    """
    if geometry_source == "derived":
        required = float(WALKER_CONSTANT_COUNT)
    elif constellation_data is not None:
        required = float(2 * constellation_data.n_orbits * constellation_data.n_sats_per_orbit)
    else:
        required = float("nan")
    return {
        "geometry_derived": 1.0 if geometry_source == "derived" else 0.0,
        "geometry_required_entries": required,
    }


def _build_reported_torus_weight_model(
    satellite_only_subgraph: nx.Graph,
    satellite_addresses: dict[int, TopologicalNetworkAddress],
    constellation_data: ConstellationData,
    state_report: dict | None,
    wiring: str = "plus_grid",
) -> dict:
    """Build the pivot weight model, recording its cost when a report is requested.

    The pivot estimator rebuilds its geometry every snapshot, so both the build
    time and the resident size of each structure are recorded rather than left
    implicit.
    """
    build_start = time.perf_counter()
    weight_model = _build_torus_weight_model(
        satellite_only_subgraph,
        satellite_addresses,
        constellation_data,
        wiring=wiring,
    )
    build_ms = (time.perf_counter() - build_start) * 1000.0
    if state_report is not None:
        state_report.update(_describe_weight_model(weight_model, build_ms))
    return weight_model


def _report_forwarding_work(
    state_report: dict | None,
    per_satellite_work: dict | None,
    weight_model: dict | None,
    fstate: dict | None = None,
) -> None:
    """Record per-satellite work and pivot cache size once forwarding has run."""
    if state_report is None:
        return
    state_report.update(_summarize_per_satellite_work(per_satellite_work))
    state_report["local_detour_entries"] = float(
        sum(1 for entry in (fstate or {}).values() if _is_local_detour_entry(entry))
    )
    if weight_model is not None:
        # Filled after forwarding, since the cache only grows as pairs are queried.
        state_report["pivot_cache_entries"] = float(len(weight_model["pivot_distance_cache"]))


def _summarize_per_satellite_work(per_satellite_work: dict | None) -> dict:
    """What a single satellite computes and caches, as opposed to the simulator.

    LEOPath derives the whole constellation's forwarding state in one process,
    so its memo table holds every pair any satellite asked about. That figure
    describes the simulator, not the design. These are the per-node quantities:
    forwarding decisions taken, distance-function evaluations performed, and
    the distinct (neighbour, destination) pairs a node would memoise, which
    bounds its own cache.
    """
    if not per_satellite_work:
        return {}
    decisions = [w["decisions"] for w in per_satellite_work.values()]
    evaluations = [w["evaluations"] for w in per_satellite_work.values()]
    pairs = [len(w["pairs"]) for w in per_satellite_work.values()]

    def mean(values: list[int]) -> float:
        return float(sum(values)) / len(values) if values else 0.0

    return {
        "decisions_per_sat_mean": mean(decisions),
        "decisions_per_sat_max": float(max(decisions)) if decisions else 0.0,
        "distance_evals_per_sat_mean": mean(evaluations),
        "distance_evals_per_sat_max": float(max(evaluations)) if evaluations else 0.0,
        "cache_pairs_per_sat_mean": mean(pairs),
        "cache_pairs_per_sat_max": float(max(pairs)) if pairs else 0.0,
        "forwarding_exceptions_isolated": float(
            sum(w.get("isolated", 0) for w in per_satellite_work.values())
        ),
        "forwarding_exceptions": float(
            sum(w.get("exceptions", 0) for w in per_satellite_work.values())
        ),
    }


def _describe_weight_model(weight_model: dict, build_ms: float) -> dict:
    """Resident size of each structure the pivot estimator keeps, in entries.

    Reported per category rather than as a single figure, because the tiers
    differ in kind. The edge costs are the irreducible geometry: measured ISL
    lengths, a deterministic function of orbital elements and so derivable
    on board rather than distributed. The path-cost tables are precomputed
    from those edge costs and are a time-for-space trade, and the pivot cache
    is memoisation that grows only with the pairs actually queried. None of
    them is installed forwarding state.
    """
    planes = int(weight_model["plane_modulus"])
    sats = int(weight_model["sat_modulus"])
    return {
        "geometry_build_ms": float(build_ms),
        "geometry_row_edge_entries": float(planes * sats),
        "geometry_plane_edge_entries": float(sats * planes),
        "path_cost_row_entries": float(planes * sats * sats),
        "path_cost_plane_entries": float(sats * planes * planes),
    }


ISL_WIRINGS = ("plus_grid", "brick_a", "brick_b")


def _build_torus_weight_model(
    satellite_only_subgraph: nx.Graph,
    satellite_addresses: dict[int, TopologicalNetworkAddress],
    constellation_data: ConstellationData,
    wiring: str = "plus_grid",
) -> dict:
    if wiring not in ISL_WIRINGS:
        raise ValueError(f"Unknown ISL wiring {wiring!r}, expected one of {ISL_WIRINGS}")
    plane_modulus = constellation_data.n_orbits
    sat_modulus = constellation_data.n_sats_per_orbit
    row_edge_costs = [[float("inf")] * sat_modulus for _ in range(plane_modulus)]
    plane_edge_costs = [[float("inf")] * plane_modulus for _ in range(sat_modulus)]

    for sat_a_id, sat_b_id, edge_data in satellite_only_subgraph.edges(data=True):
        addr_a = satellite_addresses.get(sat_a_id)
        addr_b = satellite_addresses.get(sat_b_id)
        if addr_a is None or addr_b is None:
            continue
        sat_a = addr_a.get_satellite_address()
        sat_b = addr_b.get_satellite_address()
        if sat_a.shell_id != sat_b.shell_id:
            continue
        edge_weight = float(edge_data.get("weight", 1.0))

        if sat_a.plane_id == sat_b.plane_id:
            _record_forward_torus_edge(
                row_edge_costs[sat_a.plane_id],
                sat_a.sat_index,
                sat_b.sat_index,
                edge_weight,
            )
        elif sat_a.sat_index == sat_b.sat_index:
            _record_forward_torus_edge(
                plane_edge_costs[sat_a.sat_index],
                sat_a.plane_id,
                sat_b.plane_id,
                edge_weight,
            )

    row_path_costs = [
        [
            [
                _torus_path_cost(row_edge_costs[plane_index], source_row, destination_row)
                for destination_row in range(sat_modulus)
            ]
            for source_row in range(sat_modulus)
        ]
        for plane_index in range(plane_modulus)
    ]
    plane_path_costs = [
        [
            [
                _torus_path_cost(plane_edge_costs[row_index], source_plane, destination_plane)
                for destination_plane in range(plane_modulus)
            ]
            for source_plane in range(plane_modulus)
        ]
        for row_index in range(sat_modulus)
    ]

    brick = None
    if wiring in ("brick_a", "brick_b"):
        brick = _brick_lengths(row_edge_costs, plane_edge_costs, wiring)

    return {
        "plane_modulus": plane_modulus,
        "sat_modulus": sat_modulus,
        "wiring": wiring,
        "row_edge_costs": row_edge_costs,
        "plane_edge_costs": plane_edge_costs,
        "row_path_costs": row_path_costs,
        "plane_path_costs": plane_path_costs,
        "brick": brick,
        "pivot_distance_cache": {},
    }


def _record_forward_torus_edge(
    edge_costs: list[float],
    index_a: int,
    index_b: int,
    edge_weight: float,
) -> None:
    modulus = len(edge_costs)
    if modulus <= 0:
        return
    if (index_b - index_a) % modulus == 1:
        edge_index = index_a
    elif (index_a - index_b) % modulus == 1:
        edge_index = index_b
    else:
        return
    edge_costs[edge_index] = min(edge_costs[edge_index], edge_weight)


def _torus_weighted_pivot_distance(
    source_address: TopologicalNetworkAddress,
    destination_address: TopologicalNetworkAddress,
    weight_model: dict,
    shell_penalty: float = 1000.0,
) -> float:
    source_sat = source_address.get_satellite_address()
    destination_sat = destination_address.get_satellite_address()
    if source_sat == destination_sat:
        return 0.0
    if source_sat.shell_id != destination_sat.shell_id:
        shell_diff = abs(source_sat.shell_id - destination_sat.shell_id)
        return shell_penalty + shell_diff * shell_penalty

    cache_key = (
        source_sat.plane_id,
        source_sat.sat_index,
        destination_sat.plane_id,
        destination_sat.sat_index,
    )
    distance_cache = weight_model["pivot_distance_cache"]
    cached_distance = distance_cache.get(cache_key)
    if cached_distance is not None:
        return cached_distance

    if weight_model.get("brick") is not None:
        best_distance = _brick_pivot_distance(source_sat, destination_sat, weight_model)
        distance_cache[cache_key] = best_distance
        return best_distance

    row_path_costs = weight_model["row_path_costs"]
    plane_path_costs = weight_model["plane_path_costs"]
    sat_modulus = int(weight_model["sat_modulus"])
    best_distance = float("inf")

    for pivot_row in range(sat_modulus):
        source_row_cost = row_path_costs[source_sat.plane_id][source_sat.sat_index][pivot_row]
        plane_cost = plane_path_costs[pivot_row][source_sat.plane_id][destination_sat.plane_id]
        destination_row_cost = row_path_costs[destination_sat.plane_id][pivot_row][
            destination_sat.sat_index
        ]
        best_distance = min(
            best_distance,
            source_row_cost + plane_cost + destination_row_cost,
        )

    distance_cache[cache_key] = best_distance
    return best_distance


def _brick_lengths(
    row_edge_costs: list[list[float]],
    plane_edge_costs: list[list[float]],
    wiring: str,
) -> dict:
    """Link lengths a brick-wall estimate needs, per row.

    Every rail of a circular shell has the same length, so one mean serves. A
    rung's length depends on its row, so each row keeps the mean of its rungs;
    under split a a row only carries rungs of one plane parity anyway.
    """

    def mean(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return sum(finite) / len(finite) if finite else math.inf

    rail = mean([cost for plane in row_edge_costs for cost in plane])
    return {
        "rail": rail,
        "rung_by_row": [mean(row) for row in plane_edge_costs],
        "plane_wrap": (
            all(math.isfinite(row[-1]) for row in plane_edge_costs)
            if wiring == "brick_b"
            else any(math.isfinite(row[-1]) for row in plane_edge_costs)
        ),
    }


def _brick_hops(
    start: int,
    start_row: int,
    end: int,
    end_row: int,
    crossings_modulus: int,
    rows_modulus: int,
    crossing_wraps: bool,
    rows_wrap: bool,
) -> tuple[int, int] | None:
    """Fewest (staggered crossings, complete-axis moves) between two satellites.

    The closed form of docs/isl-topology.md, verified against BFS: crossing a
    staggered link needs start + row to be even (going up) or odd (going down),
    and each crossing flips that parity, so the path zigzags one row per
    crossing. Zigzag rows it wanted anyway are free; the rest are paid twice.
    It is exact when both ring sizes are even. With an odd ring, stepping
    across its wrap keeps the parity instead of flipping it, and the result is
    an estimate; no shell evaluated with a brick wall has an odd ring.
    Returns (crossings, moves) for the cheaper direction, or None if neither
    direction is allowed (a cylinder never crosses its seam).
    """
    if rows_wrap:
        row_steps = (end_row - start_row) % rows_modulus
        row_distance = min(row_steps, rows_modulus - row_steps)
    else:
        row_distance = abs(end_row - start_row)
    best = None
    for crossings, needs_shift, wraps in (
        ((end - start) % crossings_modulus, (start + start_row) % 2 == 1, end < start),
        ((start - end) % crossings_modulus, (start + start_row) % 2 == 0, end > start),
    ):
        if wraps and crossings and not crossing_wraps:
            continue
        if crossings == 0:
            candidate = (0, row_distance)
        else:
            span = max(row_distance, (crossings - 1) + (1 if needs_shift else 0))
            if (span - row_distance) % 2:
                span += 1
            candidate = (crossings, span)
        if best is None or sum(candidate) < sum(best):
            best = candidate
    return best


def _brick_pivot_distance(source_sat, destination_sat, weight_model: dict) -> float:
    """Pivot distance on a three-terminal brick wall.

    Split a (rungs staggered): walk the source plane to a pivot row, then take
    the closed-form staircase to the destination, its rungs priced at the mean
    rung length and its rails at the rail length. Split b swaps the roles of
    planes and rows. With unit link costs this is exactly the hop distance.

    The staircase is priced at one mean rung length rather than the pivot
    row's own. Pricing it per row would let the estimate prefer short
    high-latitude rungs, but a satellite's best pivot row then shifts as the
    packet moves, the estimate stops being consistent between neighbours, and
    greedy forwarding loops: on Starlink that lost 71% of pairs.
    """
    planes = int(weight_model["plane_modulus"])
    sats = int(weight_model["sat_modulus"])
    brick = weight_model["brick"]
    rail, rungs = brick["rail"], brick["rung_by_row"]
    finite_rungs = [r for r in rungs if math.isfinite(r)]
    mean_rung = sum(finite_rungs) / len(finite_rungs) if finite_rungs else math.inf
    best = math.inf
    if weight_model["wiring"] == "brick_a":
        leg = weight_model["row_path_costs"][source_sat.plane_id][source_sat.sat_index]
        for pivot_row in range(sats):
            hops = _brick_hops(
                source_sat.plane_id,
                pivot_row,
                destination_sat.plane_id,
                destination_sat.sat_index,
                planes,
                sats,
                crossing_wraps=brick["plane_wrap"],
                rows_wrap=True,
            )
            if hops is None:
                continue
            crossings, moves = hops
            best = min(best, leg[pivot_row] + crossings * mean_rung + moves * rail)
    else:
        leg = weight_model["plane_path_costs"][source_sat.sat_index][source_sat.plane_id]
        for pivot_plane in range(planes):
            if not math.isfinite(leg[pivot_plane]):
                continue
            # Transposed: the staggered links are rails, crossed along the slot
            # ring; the zigzag moves between planes, over rungs.
            hops = _brick_hops(
                source_sat.sat_index,
                pivot_plane,
                destination_sat.sat_index,
                destination_sat.plane_id,
                sats,
                planes,
                crossing_wraps=True,
                rows_wrap=brick["plane_wrap"],
            )
            if hops is None:
                continue
            crossings, moves = hops
            best = min(best, leg[pivot_plane] + crossings * rail + moves * mean_rung)
    return best


def _torus_path_cost(edge_costs: list[float], start_index: int, end_index: int) -> float:
    modulus = len(edge_costs)
    if modulus <= 0:
        return float("inf")
    if start_index == end_index:
        return 0.0

    forward_steps = (end_index - start_index) % modulus
    backward_steps = (start_index - end_index) % modulus
    forward_cost = _sum_torus_edges(edge_costs, start_index, 1, forward_steps)
    backward_cost = _sum_torus_edges(edge_costs, start_index - 1, -1, backward_steps)
    return min(forward_cost, backward_cost)


def _sum_torus_edges(
    edge_costs: list[float],
    start_edge_index: int,
    direction: int,
    steps: int,
) -> float:
    total = 0.0
    modulus = len(edge_costs)
    for step in range(steps):
        edge_cost = edge_costs[(start_edge_index + direction * step) % modulus]
        if edge_cost == float("inf"):
            return float("inf")
        total += edge_cost
    return total


def _scaled_gsl_distance(distance_m: float, distance_mode: str) -> float:
    if distance_mode in {"torus_weighted", "torus_weighted_lookahead", "torus_weighted_pivot"}:
        return float(distance_m)
    return float(distance_m) / 1000000.0


def _estimate_axis_step_costs(
    current_address: TopologicalNetworkAddress,
    neighbor_candidates: list[tuple[int, int, TopologicalNetworkAddress, float]],
) -> tuple[float, float]:
    plane_costs = []
    sat_costs = []
    for _neighbor_id, _interface, neighbor_address, edge_weight in neighbor_candidates:
        if edge_weight <= 0.0:
            continue
        same_plane = neighbor_address.plane_id == current_address.plane_id
        same_sat = neighbor_address.sat_index == current_address.sat_index
        if not same_plane and same_sat:
            plane_costs.append(float(edge_weight))
        elif same_plane and not same_sat:
            sat_costs.append(float(edge_weight))
    plane_step_cost = min(plane_costs) if plane_costs else 1.0
    sat_step_cost = min(sat_costs) if sat_costs else 1.0
    return (plane_step_cost, sat_step_cost)


def _neighbor_candidate_score(
    edge_weight: float,
    neighbor_distance_to_dest: float,
    distance_mode: str,
) -> float:
    if distance_mode in {"torus_unit", "torus_weighted"}:
        return neighbor_distance_to_dest
    return edge_weight + neighbor_distance_to_dest


def _routing_tie_break_tuple(
    source_address: TopologicalNetworkAddress,
    destination_address: TopologicalNetworkAddress,
    constellation_data: ConstellationData,
) -> tuple[int, int, int]:
    source_sat = source_address.get_satellite_address()
    destination_sat = destination_address.get_satellite_address()
    plane_forward = (destination_sat.plane_id - source_sat.plane_id) % constellation_data.n_orbits
    sat_forward = (
        destination_sat.sat_index - source_sat.sat_index
    ) % constellation_data.n_sats_per_orbit
    same_plane_priority = 0 if source_sat.plane_id == destination_sat.plane_id else 1
    return (same_plane_priority, sat_forward, plane_forward)


def _routing_strict_progress_tuple(
    source_address: TopologicalNetworkAddress,
    destination_address: TopologicalNetworkAddress,
) -> tuple[int, int, int]:
    source_sat = source_address.get_satellite_address()
    destination_sat = destination_address.get_satellite_address()
    return (
        abs(source_sat.plane_id - destination_sat.plane_id),
        abs(source_sat.sat_index - destination_sat.sat_index),
        abs(source_sat.shell_id - destination_sat.shell_id),
    )
