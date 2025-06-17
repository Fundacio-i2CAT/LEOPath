# src/dynamic_state/generate_dynamic_state.py
# Updated based on provided LEOTopology class definition

import math

from astropy import units as astro_units
from astropy.time import Time
from tqdm import tqdm  # Add this import

from src import logger
from src.topology.topology import ConstellationData, GroundStation

from .helpers import _build_topologies, _compute_ground_station_satellites_in_range, _compute_isls
from .routing_algorithms.routing_algorithm_factory import get_routing_algorithm
from .utils import graph as graph_utils

log = logger.get_logger(__name__)


def generate_dynamic_state(
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
    _validate_inputs(epoch, simulation_end_time_ns, time_step_ns, offset_ns)

    total_iterations, progress_interval = _compute_iterations_and_progress(
        simulation_end_time_ns, time_step_ns, offset_ns
    )
    log.info(f"Starting dynamic state generation for {max(0, total_iterations):.0f} iterations.")

    all_states = []
    prev_output = None
    prev_topology = None

    time_steps = range(offset_ns, simulation_end_time_ns, time_step_ns)
    pbar = tqdm(total=total_iterations, desc="Overall Simulation Progress")

    for i, time_since_epoch_ns in enumerate(time_steps):
        _log_progress(i, progress_interval, time_since_epoch_ns, total_iterations, pbar)
        try:
            current_output, current_topology = _generate_state_for_step(
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
            if current_output is not None:
                current_output["time_since_epoch_ns"] = time_since_epoch_ns
                all_states.append(current_output)
                prev_output = current_output
                prev_topology = current_topology
            else:
                log.error(
                    f"generate_dynamic_state_at returned None state at t={time_since_epoch_ns} ns. Appending None and stopping."
                )
                all_states.append({"error": "State calculation failed."})
                break
        except Exception:
            log.exception(
                f"Unhandled error during dynamic state processing at t={time_since_epoch_ns} ns. Stopping."
            )
            all_states.append({"error": "Unhandled exception occurred."})
            break

    pbar.close()  # Close the progress bar
    log.info(f"Dynamic state generation finished. Generated {len(all_states)} states.")
    return all_states


def _validate_inputs(epoch, simulation_end_time_ns, time_step_ns, offset_ns):
    if not isinstance(epoch, Time):
        raise TypeError("Epoch must be an astropy Time object.")
    if not all(isinstance(t, int) for t in [simulation_end_time_ns, time_step_ns, offset_ns]):
        raise TypeError("Time parameters must be integers.")
    if time_step_ns <= 0:
        log.error("time_step_ns must be positive.")
        raise ValueError("time_step_ns must be positive.")
    if offset_ns % time_step_ns != 0:
        raise ValueError("Offset must be a multiple of time_step_ns")


def _compute_iterations_and_progress(simulation_end_time_ns, time_step_ns, offset_ns):
    total_iterations = math.floor((simulation_end_time_ns - offset_ns) / time_step_ns)
    progress_interval = (
        max(1, int(math.floor(total_iterations / 10.0))) if total_iterations > 0 else 1
    )
    return total_iterations, progress_interval


def _log_progress(i, progress_interval, time_since_epoch_ns, total_iterations, pbar=None):
    if i % progress_interval == 0:
        if pbar is not None:
            pbar.update(progress_interval)
        log.debug(
            "Progress: calculating for T=%d ns (step %d / %d)"
            % (time_since_epoch_ns, i + 1, max(1, int(total_iterations)))
        )


def _generate_state_for_step(
    epoch,
    time_since_epoch_ns,
    constellation_data,
    ground_stations,
    undirected_isls,
    list_gsl_interfaces_info,
    dynamic_state_algorithm,
    prev_output,
    prev_topology,
):
    """
    Handles state generation for a single time step.
    Returns (state_dict, topology) or (None, None) on error.
    """
    log.info(f"Generating dynamic state at t={time_since_epoch_ns} ns...")
    try:
        time_absolute = epoch + time_since_epoch_ns * astro_units.ns
        current_topology = _build_and_prepare_topology(
            constellation_data, ground_stations, list_gsl_interfaces_info
        )
        _compute_isls(current_topology, undirected_isls, time_absolute)
        gs_sat_visibility_list = _compute_ground_station_satellites_in_range(
            current_topology, time_absolute
        )
        _log_topology_stats(current_topology, gs_sat_visibility_list, time_since_epoch_ns)
    except Exception as e:
        log.exception(
            f"Failed during topology/visibility calculation at t={time_since_epoch_ns} ns: {e}"
        )
        return None, None

    graphs_changed = not graph_utils._topologies_are_equal(prev_topology, current_topology)
    log.debug(
        f"  > Time {time_since_epoch_ns} ns: _topologies_are_equal returned: {not graphs_changed}. Graphs changed? {graphs_changed}"
    )

    calculated_state = _reuse_or_calculate_state(
        graphs_changed,
        prev_output,
        dynamic_state_algorithm,
        time_since_epoch_ns,
        constellation_data,
        ground_stations,
        current_topology,
        gs_sat_visibility_list,
        list_gsl_interfaces_info,
    )

    log.info(f"State processing complete for t={time_since_epoch_ns} ns.")
    return calculated_state, current_topology


def _build_and_prepare_topology(constellation_data, ground_stations, list_gsl_interfaces_info):
    current_topology, _ = _build_topologies(constellation_data, ground_stations)
    if (
        not hasattr(current_topology, "gsl_interfaces_info")
        or not current_topology.gsl_interfaces_info
    ):
        current_topology.gsl_interfaces_info = list_gsl_interfaces_info
    return current_topology


def _log_topology_stats(current_topology, gs_sat_visibility_list, time_since_epoch_ns):
    # Convert nanoseconds to a readable format (HH:MM:SS)
    seconds_since_epoch = time_since_epoch_ns / 1e9
    hours = int(seconds_since_epoch // 3600)
    minutes = int((seconds_since_epoch % 3600) // 60)
    seconds = int(seconds_since_epoch % 60)
    time_formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    num_visible_gsls = sum(len(vis_list) for vis_list in gs_sat_visibility_list)
    log.info(
        f"  > Time {time_formatted} ({time_since_epoch_ns} ns): Found {num_visible_gsls} visible GSLs."
    )
    log.info(
        f"  > Time {time_formatted} ({time_since_epoch_ns} ns): Graph has {current_topology.graph.number_of_nodes()} nodes and {current_topology.graph.number_of_edges()} edges before comparison."
    )
    log.info(
        f"  > Topology at t={time_formatted} ({time_since_epoch_ns} ns): "
        f"{current_topology.graph.number_of_nodes()} nodes, "
        f"{current_topology.graph.number_of_edges()} edges."
    )


def _reuse_or_calculate_state(
    graphs_changed,
    prev_output,
    dynamic_state_algorithm,
    time_since_epoch_ns,
    constellation_data,
    ground_stations,
    current_topology,
    gs_sat_visibility_list,
    list_gsl_interfaces_info,
):
    if not graphs_changed and prev_output is not None:
        log.debug(f"Topology unchanged at t={time_since_epoch_ns} ns. Reusing previous state.")
        if "fstate" in prev_output and "bandwidth" in prev_output:
            return prev_output.copy()
        else:
            log.warning(
                f"Topology unchanged but prev_output is missing keys at t={time_since_epoch_ns} ns. Forcing recalculation."
            )
            graphs_changed = True
    try:
        algorithm = get_routing_algorithm(dynamic_state_algorithm)
        return algorithm.compute_state(
            time_since_epoch_ns=time_since_epoch_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            topology_with_isls=current_topology,
            ground_station_satellites_in_range=gs_sat_visibility_list,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
        )
    except ValueError:
        raise
    except Exception as e:
        log.exception(
            f"Algorithm '{dynamic_state_algorithm}' execution failed at t={time_since_epoch_ns} ns: {e}"
        )
        return None
