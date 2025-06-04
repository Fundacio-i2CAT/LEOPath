import numpy as np
from astropy.time import Time

from src import logger
from src.distance_tools import distance_tools
from src.topology.topology import ConstellationData, GroundStation, LEOTopology

log = logger.get_logger(__name__)


def _compute_isls(
    topology_with_isls: LEOTopology,
    undirected_isls: list,
    current_time_absolute: Time,
):
    """
    Computes ISLs, adds them as edges to topology_with_isls.graph,
    updates sat_neighbor_to_if map and satellite ISL counts.
    Assumes topology_with_isls.get_satellite(id) works correctly.
    """
    constellation_data = topology_with_isls.constellation_data
    # Track number of ISLs per sat *during this function* to assign IF indices
    # Initialize based on satellites actually present in constellation data
    num_isls_per_sat_map = {sat.id: 0 for sat in topology_with_isls.get_satellites()}
    # Reset ISL count on topology object
    topology_with_isls.number_of_isls = 0
    # Clear previous interface mapping
    topology_with_isls.sat_neighbor_to_if = {}

    log.debug(f"Processing {len(undirected_isls)} potential ISLs...")
    for satellite_id_a, satellite_id_b in undirected_isls:
        # Get satellite objects using the topology's getter
        try:
            sat_a = topology_with_isls.get_satellite(satellite_id_a)
            sat_b = topology_with_isls.get_satellite(satellite_id_b)
        except KeyError as e:
            log.warning(
                f"Skipping ISL ({satellite_id_a}, {satellite_id_b}): Satellite object not found ({e})."
            )
            continue

        # Calculate distance
        try:
            sat_distance_m = distance_tools.distance_m_between_satellites(
                sat_a,
                sat_b,
                str(constellation_data.epoch),
                str(current_time_absolute),
            )
        except Exception as e:
            log.error(
                f"ISL distance calculation failed for ({satellite_id_a}, {satellite_id_b}): {e}"
            )
            continue  # Skip this ISL if distance fails

        # Check distance constraint
        if sat_distance_m > constellation_data.max_isl_length_m:
            # Log warning instead of raising error? Or keep error? Let's keep error for now.
            raise ValueError(
                f"The distance between satellites ({satellite_id_a} and {satellite_id_b}) "
                f"with an ISL exceeded the maximum ISL length "
                f"({sat_distance_m:.2f}m > {constellation_data.max_isl_length_m:.2f}m "
                f"at t={str(current_time_absolute)})"
            )

        # Add edge to networkx graph
        # Ensure nodes exist in graph (should be added by _build_topologies)
        if not topology_with_isls.graph.has_node(
            satellite_id_a
        ) or not topology_with_isls.graph.has_node(satellite_id_b):
            log.error(
                f"Cannot add ISL edge ({satellite_id_a}, {satellite_id_b}): Node(s) missing from graph."
            )
            continue
        topology_with_isls.graph.add_edge(satellite_id_a, satellite_id_b, weight=sat_distance_m)

        # Interface mapping of ISLs (0-based index per satellite)
        if_a = num_isls_per_sat_map[satellite_id_a]
        if_b = num_isls_per_sat_map[satellite_id_b]
        topology_with_isls.sat_neighbor_to_if[(satellite_id_a, satellite_id_b)] = if_a
        topology_with_isls.sat_neighbor_to_if[(satellite_id_b, satellite_id_a)] = if_b
        num_isls_per_sat_map[satellite_id_a] += 1
        num_isls_per_sat_map[satellite_id_b] += 1

        topology_with_isls.number_of_isls += 1  # Count pairs

    # Final update of number_isls on satellite objects stored within topology
    total_isl_endpoints = 0
    for sat in topology_with_isls.get_satellites():
        sat.number_isls = num_isls_per_sat_map.get(sat.id, 0)
        total_isl_endpoints += sat.number_isls

    # Log summary info
    if topology_with_isls.number_of_isls > 0:
        num_isls_counts = list(num_isls_per_sat_map.values())
        log.debug(f"  > Computed {topology_with_isls.number_of_isls} ISLs.")
        try:
            log.debug(f"  > Min. ISLs/satellite.... {np.min(num_isls_counts)}")
            log.debug(f"  > Max. ISLs/satellite.... {np.max(num_isls_counts)}")
        except ValueError:  # Handles case where num_isls_counts might be empty
            log.warning("Could not calculate min/max ISLs per satellite.")
    else:
        log.debug("  > No ISLs computed or defined.")


def _build_topologies(orbital_data: ConstellationData, ground_stations: list[GroundStation]):
    """
    Builds LEOTopology instance(s). Adds nodes based on actual sat/gs IDs.

    :return: Tuple[LEOTopology, LEOTopology] -> (topology_with_isls, topology_only_gs)
             Note: topology_only_gs might be redundant.
    """
    topology_with_isls = LEOTopology(orbital_data, ground_stations)
    topology_only_gs = LEOTopology(orbital_data, ground_stations)  # May not be needed later

    # Add satellite nodes using their IDs
    for (
        sat
    ) in (
        orbital_data.satellites
    ):  # Assuming orbital_data.satellites has Satellite objects or ephem.Body
        # We need the ID. If it holds ephem.Body, we might need to adjust.
        # Assuming for now constellation_data holds Satellite objects for consistency
        # with get_satellite. If it holds ephem.Body, we need a way to get the intended ID.
        # Let's proceed assuming sat.id exists.
        if hasattr(sat, "id"):
            topology_with_isls.graph.add_node(sat.id)
            topology_only_gs.graph.add_node(sat.id)
        else:
            # This case occurs if constellation_data.satellites holds ephem.Body directly
            # We need a way to map ephem.Body back to the intended sat ID (0..N-1 or unique IDs)
            # For now, log a warning. This indicates an inconsistency to be resolved.
            log.warning(
                "Satellite object in constellation_data lacks 'id' attribute. Node addition may be incorrect."
            )
            # Fallback? Maybe try adding based on index? Requires care.

    # Add ground station nodes using their IDs
    for gs in ground_stations:
        topology_with_isls.graph.add_node(gs.id)  # Add GS to main graph too for GSLs
        topology_only_gs.graph.add_node(gs.id)

    log.debug(
        f"  > Built topologies with {len(topology_with_isls.graph.nodes())} initial nodes."
    )  # More accurate node count
    log.debug(f"  > Max. range GSL......... {orbital_data.max_gsl_length_m} m")
    log.debug(f"  > Max. range ISL......... {orbital_data.max_isl_length_m} m")
    return topology_with_isls, topology_only_gs


def _compute_gsl_interface_information(topology: LEOTopology):
    """
    Logs summary information about GSL interfaces based on data stored in the topology object.
    Requires `topology.gsl_interfaces_info` to be populated correctly.
    """
    if (
        not hasattr(topology, "gsl_interfaces_info")
        or not topology.gsl_interfaces_info
        or not isinstance(topology.gsl_interfaces_info, list)
    ):
        log.warning(
            "Cannot log GSL interface info; topology.gsl_interfaces_info not set or not a list."
        )
        return

    constellation_data = topology.constellation_data
    # Access satellite count via constellation_data, GS count via topology
    num_sats = constellation_data.number_of_satellites
    num_gs = topology.number_of_ground_stations
    expected_len = num_sats + num_gs

    if len(topology.gsl_interfaces_info) != expected_len:
        log.warning(
            f"Length of topology.gsl_interfaces_info ({len(topology.gsl_interfaces_info)}) "
            f"does not match expected node count ({expected_len}). Logging may be incomplete."
        )

    log.debug("GSL INTERFACE INFORMATION (from topology.gsl_interfaces_info):")

    try:
        # Extract number_of_interfaces, handle potential missing keys or non-dict items
        # Adapt slicing based on actual length vs expected num_sats
        actual_len = len(topology.gsl_interfaces_info)
        sat_if_counts = [
            info.get("number_of_interfaces", 0)
            for info in topology.gsl_interfaces_info[: min(actual_len, num_sats)]
            if isinstance(info, dict)
        ]
        gs_if_counts = [
            info.get("number_of_interfaces", 0)
            for info in topology.gsl_interfaces_info[num_sats:actual_len]
            if isinstance(info, dict)
        ]

        if sat_if_counts:
            log.debug(f"  > Min. GSL IFs/satellite........ {np.min(sat_if_counts)}")
            log.debug(f"  > Max. GSL IFs/satellite........ {np.max(sat_if_counts)}")
        else:
            log.debug("  > No valid satellite GSL interface data found/processed.")

        if gs_if_counts:
            log.debug(f"  > Min. GSL IFs/ground station... {np.min(gs_if_counts)}")
            log.debug(f"  > Max. GSL IFs/ground_station... {np.max(gs_if_counts)}")
        else:
            log.debug("  > No valid ground station GSL interface data found/processed.")

    except Exception as e:
        log.exception(f"Error processing GSL interface information: {e}")


def _compute_ground_station_satellites_in_range(
    topology: LEOTopology, current_time: Time  # Expects Time object
) -> list:  # Returns visibility list
    """
    Computes GS<->Sat visibility based on distance at current_time.
    Adds GSL edges with weights to the topology.graph.
    Returns the visibility list: list[ list[(distance_m, sat_id)] ] indexed by GS index.
    Assumes topology.get_satellites() and topology.get_ground_stations() work.
    """
    log.debug("Calculating GSL in-range information...")
    ground_station_satellites_in_range = []  # List to be returned, index matches gs_list order
    try:
        satellites = topology.get_satellites()
        gs_list = topology.get_ground_stations()
    except Exception as e:
        log.exception(f"Error retrieving satellites or ground stations from topology: {e}")
        return [[] for _ in range(topology.number_of_ground_stations)]  # Return empty structure

    # Iterate by index to build the return list correctly
    for gs_idx, ground_station in enumerate(gs_list):
        satellites_in_range_for_this_gs = []
        for satellite in satellites:
            if not hasattr(satellite, "position") or not hasattr(satellite, "id"):
                log.warning(f"Skipping visibility check for invalid satellite object: {satellite}")
                continue

            try:
                time_str_for_ephem = str(current_time.strftime("%Y/%m/%d %H:%M:%S.%f")[:-3])
                epoch_str_for_ephem = topology.constellation_data.epoch
                distance_m = distance_tools.distance_m_ground_station_to_satellite(
                    ground_station,  # Pass GroundStation object
                    satellite,  # Pass Satellite object (NOT satellite.position)
                    epoch_str_for_ephem,  # Pass epoch string
                    time_str_for_ephem,  # Pass formatted time string
                )
            except Exception as e:
                # Log specific error, include IDs for easier debugging
                log.error(
                    f"GSL distance calculation failed for GS {getattr(ground_station, 'id', 'N/A')} "
                    f"<-> Sat {getattr(satellite, 'id', 'N/A')}: {e}"
                )
                distance_m = float("inf")  # Treat as out of range on error

            # Check against max length from constellation_data
            if distance_m <= topology.constellation_data.max_gsl_length_m:
                satellites_in_range_for_this_gs.append((distance_m, satellite.id))
                # Add edge to the graph IN THE PASSED TOPOLOGY OBJECT
                if topology.graph.has_node(satellite.id) and topology.graph.has_node(
                    ground_station.id
                ):
                    topology.graph.add_edge(satellite.id, ground_station.id, weight=distance_m)
                else:
                    log.warning(
                        f"Cannot add GSL edge ({satellite.id}, {ground_station.id}): Node(s) missing from graph."
                    )

        ground_station_satellites_in_range.append(satellites_in_range_for_this_gs)

    # Log summary info
    if ground_station_satellites_in_range:
        try:
            ground_station_num_in_range = [
                len(visible_sats) for visible_sats in ground_station_satellites_in_range
            ]
            log.debug(
                f"  > Min. satellites in range per GS... {np.min(ground_station_num_in_range)}"
            )
            log.debug(
                f"  > Max. satellites in range per GS... {np.max(ground_station_num_in_range)}"
            )
        except ValueError:  # Handles case where list might be empty or contain non-iterables
            log.debug("  > Could not compute min/max satellites in range (list empty or invalid?).")
    else:
        log.debug("  > No ground stations processed for visibility.")
    return ground_station_satellites_in_range
