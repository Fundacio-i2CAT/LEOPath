# src/dynamic_state/generate_dynamic_state.py
# Updated based on provided LEOTopology class definition

import math

from astropy import units as astro_units
from astropy.time import Time

from src import logger
from src.topology.topology import ConstellationData, GroundStation, LEOTopology

from .helpers import _build_topologies, _compute_ground_station_satellites_in_range, _compute_isls
from .routing_algorithms.shortest_path_link_state_routing.algorithm_free_one_only_over_isls import algorithm_free_one_only_over_isls
from .utils import graph as graph_utils

log = logger.get_logger(__name__)


def generate_dynamic_state(
    output_dynamic_state_dir: str | None,
    epoch: Time,
    simulation_end_time_ns: int,
    time_step_ns: int,
    offset_ns: int,
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    undirected_isls: list,
    list_gsl_interfaces_info: list,
    dynamic_state_algorithm: str,
) -> list[dict]:
    """
    Generates dynamic state over a simulation period.
    Returns a list containing the state calculated at each time step.

    :param output_dynamic_state_dir: Directory for any non-state output files. Can be None.
    :param epoch: Astropy Time object representing the simulation epoch.
    :param simulation_end_time_ns: End time in nanoseconds since epoch (integer).
    :param time_step_ns: Simulation time step in nanoseconds (integer).
    :param offset_ns: Start time offset in nanoseconds since epoch (integer).
    :param constellation_data: ConstellationData object.
    :param ground_stations: List of GroundStation objects.
    :param undirected_isls: List of predefined ISL pairs [(sat_id_a, sat_id_b), ...].
    :param list_gsl_interfaces_info: List of dictionaries defining GSL interface properties.
    :param dynamic_state_algorithm: String identifier of the algorithm to use.
    :return: List of state dictionaries (e.g., [{'fstate':..., 'bandwidth':...}, ...]),
             one for each time step. Contains None for steps with errors.
    """
    if not isinstance(epoch, Time):
        raise TypeError("Epoch must be an astropy Time object.")
    if not all(isinstance(t, int) for t in [simulation_end_time_ns, time_step_ns, offset_ns]):
        raise TypeError("Time parameters must be integers.")
    if time_step_ns <= 0:
        log.error("time_step_ns must be positive.")
        return []
    if offset_ns % time_step_ns != 0:
        raise ValueError("Offset must be a multiple of time_step_ns")

    all_states = []
    prev_output = None
    prev_topology = None
    i = 0
    # --- Calculate iterations and progress interval ---
    if time_step_ns <= 0:
        total_iterations = 0
    else:
        total_iterations = math.floor(
            (simulation_end_time_ns - offset_ns) / time_step_ns
        )  # Use floor for accurate count
    if total_iterations > 0:
        progress_interval = max(1, int(math.floor(total_iterations / 10.0)))
    else:
        progress_interval = 1
    log.info(f"Starting dynamic state generation for {max(0, total_iterations):.0f} iterations.")
    for time_since_epoch_ns in range(offset_ns, simulation_end_time_ns, time_step_ns):
        if i % progress_interval == 0:
            log.info(
                "Progress: calculating for T=%d ns (step %d / %d)"
                % (time_since_epoch_ns, i + 1, max(1, int(total_iterations)))
            )
        i += 1

        try:
            # Call the function that calculates state at a single time step
            # Pass prev_topology and receive current_topology back
            current_output, current_topology = generate_dynamic_state_at(
                output_dynamic_state_dir=output_dynamic_state_dir,
                epoch=epoch,
                time_since_epoch_ns=time_since_epoch_ns,
                constellation_data=constellation_data,
                ground_stations=ground_stations,
                undirected_isls=undirected_isls,
                list_gsl_interfaces_info=list_gsl_interfaces_info,
                dynamic_state_algorithm=dynamic_state_algorithm,
                prev_output=prev_output,
                prev_topology=prev_topology,
            )

            # Check the state returned
            if current_output is not None:
                # Add the timestamp to the state dictionary
                current_output["time_since_epoch_ns"] = time_since_epoch_ns
                all_states.append(current_output)  # <--- Append valid state to list
                # Update previous state and topology for the next iteration
                prev_output = current_output
                # Only update prev_topology if the calculation was fully successful
                # and returned a valid topology object for this step
                if current_topology is not None:
                    prev_topology = current_topology
                else:
                    # This case should ideally not happen if current_output is not None,
                    # but as a safeguard, maybe log a warning and don't update prev_topology
                    log.warning(
                        f"generate_dynamic_state_at returned valid state but None topology at t={time_since_epoch_ns} ns."
                    )
                    # Decide whether to break or continue without topology update
                    # For safety, let's break if the topology object isn't returned correctly.
                    # Or perhaps just don't update prev_topology:
                    # prev_topology = None # Force recalc next time? Risky.
                    # Best might be to rely on generate_dynamic_state_at returning (None, None)
                    # if topology calculation failed.
                    # Let's assume if current_output is not None, current_topology is also not None.
            else:
                # Handle case where generate_dynamic_state_at returned None for the state
                log.error(
                    f"generate_dynamic_state_at returned None state at t={time_since_epoch_ns} ns. Appending None and stopping."
                )
                all_states.append({"error": "State calculation failed."})
                break  # Stop processing further time steps

        except Exception:  # Catch exceptions raised directly by generate_dynamic_state_at
            log.exception(
                f"Unhandled error during dynamic state processing at t={time_since_epoch_ns} ns. Stopping."
            )
            all_states.append({"error": "Unhandled exception occurred."})
            break

    log.info(f"Dynamic state generation finished. Generated {len(all_states)} states.")
    return all_states


def generate_dynamic_state_at(
    output_dynamic_state_dir: str | None,
    epoch: Time,
    time_since_epoch_ns: int,
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    undirected_isls: list,
    list_gsl_interfaces_info: list,
    dynamic_state_algorithm: str,
    prev_output: dict | None,  # State from previous step
    prev_topology: LEOTopology | None,  # Topology from previous step
) -> tuple[dict | None, LEOTopology | None]:  # Return state AND current topology
    """Calculates state for a single time step, potentially reusing fstate."""
    log.info(f"Generating dynamic state at t={time_since_epoch_ns} ns...")
    current_topology = None  # Initialize
    gs_sat_visibility_list = None  # Initialize
    try:
        time_absolute = epoch + time_since_epoch_ns * astro_units.ns
        log.debug(f"  > Absolute time.......... {time_absolute}")

        # Build current topology (and ignore the second one returned)
        current_topology, _ = _build_topologies(constellation_data, ground_stations)
        # Assign GSL info if not already present (might be redundant if always done in _build_topologies)
        if (
            not hasattr(current_topology, "gsl_interfaces_info")
            or not current_topology.gsl_interfaces_info
        ):
            current_topology.gsl_interfaces_info = list_gsl_interfaces_info

        log.debug("Computing ISLs...")
        _compute_isls(current_topology, undirected_isls, time_absolute)

        log.debug("Computing GSL visibility...")
        gs_sat_visibility_list = _compute_ground_station_satellites_in_range(
            current_topology, time_absolute
        )
        num_visible_gsls = sum(len(vis_list) for vis_list in gs_sat_visibility_list)
        log.info(f"  > Time {time_since_epoch_ns} ns: Found {num_visible_gsls} visible GSLs.")
        log.info(
            f"  > Time {time_since_epoch_ns} ns: Graph has {current_topology.graph.number_of_nodes()} nodes and {current_topology.graph.number_of_edges()} edges before comparison."
        )
        log.debug(
            f"  > Topology at t={time_since_epoch_ns} ns: "
            f"{current_topology.graph.number_of_nodes()} nodes, "
            f"{current_topology.graph.number_of_edges()} edges."
        )

    except Exception as e:
        log.exception(
            f"Failed during topology/visibility calculation at t={time_since_epoch_ns} ns: {e}"
        )
        # Return None for state, and None or current_topology depending on where error occurred
        # Returning None, None is safer if topology might be incomplete.
        return None, None

    # --- Optimization: Check if topology changed ---
    graphs_changed = not graph_utils._topologies_are_equal(prev_topology, current_topology)
    log.info(
        f"  > Time {time_since_epoch_ns} ns: _topologies_are_equal returned: {not graphs_changed}. Graphs changed? {graphs_changed}"
    )
    calculated_state = None
    if not graphs_changed and prev_output is not None:
        # Reuse previous state if topology hasn't changed and we have a previous state
        log.debug(f"Topology unchanged at t={time_since_epoch_ns} ns. Reusing previous state.")
        # Make sure prev_output actually contains 'fstate' and 'bandwidth'
        if "fstate" in prev_output and "bandwidth" in prev_output:
            calculated_state = prev_output.copy()  # Shallow copy is usually sufficient
        else:
            log.warning(
                f"Topology unchanged but prev_output is missing keys at t={time_since_epoch_ns} ns. Forcing recalculation."
            )
            graphs_changed = True  # Force recalculation if prev state is malformed

    if (
        calculated_state is None
    ):  # Either graphs changed, it's the first step, or prev_output was bad
        if graphs_changed:
            log.debug(f"Topology changed at t={time_since_epoch_ns} ns. Recalculating state.")
        else:  # Must be the first step (prev_output was None) or prev_output was bad
            log.debug(
                f"First step or missing keys in prev_output (t={time_since_epoch_ns} ns). Calculating state."
            )

        log.info(f"Calling dynamic state algorithm: {dynamic_state_algorithm}")

        if dynamic_state_algorithm == "algorithm_free_one_only_over_isls":
            try:
                # Pass the computed current_topology and visibility list
                calculated_state = algorithm_free_one_only_over_isls(
                    time_since_epoch_ns=time_since_epoch_ns,  # Pass time just in case
                    constellation_data=constellation_data,
                    ground_stations=ground_stations,
                    topology_with_isls=current_topology,  # Use the current topology
                    ground_station_satellites_in_range=gs_sat_visibility_list,  # Use computed visibility
                    list_gsl_interfaces_info=list_gsl_interfaces_info,
                )
            except Exception as e:
                log.exception(
                    f"Algorithm '{dynamic_state_algorithm}' execution failed at t={time_since_epoch_ns} ns: {e}"
                )
                # Return None for state, but the current topology was successfully calculated
                return None, current_topology
        else:
            log.error(f"Unknown dynamic state algorithm: {dynamic_state_algorithm}")
            raise ValueError(f"Unknown dynamic state algorithm: {dynamic_state_algorithm}")

    log.info(f"State processing complete for t={time_since_epoch_ns} ns.")
    return calculated_state, current_topology
