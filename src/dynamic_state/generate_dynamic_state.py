# The MIT License (MIT)
#
# Copyright (c) 2020 ETH Zurich
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import networkx as nx
import numpy as np
from astropy import units as astro_units
from astropy.time import Time

from src import distance_tools, logger
from src.dynamic_state.topology import ConstellationData, GroundStation, LEOTopology

from .algorithm_free_one_only_over_isls import algorithm_free_one_only_over_isls

log = logger.get_logger(__name__)


def generate_dynamic_state(
    output_dynamic_state_dir,
    epoch,
    simulation_end_time_ns,
    time_step_ns,
    offset_ns,
    satellites,
    ground_stations,
    undirected_isls,
    list_gsl_interfaces_info,
    max_gsl_length_m,
    max_isl_length_m,
    dynamic_state_algorithm,  # Options:
    # "algorithm_free_one_only_gs_relays"
    # "algorithm_free_one_only_over_isls"
    # "algorithm_paired_many_only_over_isls"
    enable_verbose_logs,
):
    if offset_ns % time_step_ns != 0:
        raise ValueError("Offset must be a multiple of time_step_ns")
    prev_output = None
    i = 0
    # Ensure time_step_ns is not zero to avoid division by zero here too
    if time_step_ns == 0:
        total_iterations = 0  # Or handle as error
    else:
        total_iterations = (simulation_end_time_ns - offset_ns) / time_step_ns

    if total_iterations > 0:
        # Ensure divisor is at least 1
        progress_interval = max(1, int(math.floor(total_iterations) / 10.0))
    else:
        progress_interval = 1  # Avoid issues if no iterations

    for time_since_epoch_ns in range(offset_ns, simulation_end_time_ns, time_step_ns):
        if not enable_verbose_logs:
            # Use the safe progress_interval
            if i % progress_interval == 0:
                print(
                    "Progress: calculating for T=%d (time step granularity is still %d ms)"
                    % (time_since_epoch_ns, time_step_ns / 1000000)
                )
        i += 1
        prev_output = generate_dynamic_state_at(
            output_dynamic_state_dir,
            epoch,
            time_since_epoch_ns,
            satellites,  # Pass satellites (ConstellationData obj)
            ground_stations,
            undirected_isls,
            list_gsl_interfaces_info,
            dynamic_state_algorithm,
            prev_output,
        )


def _compute_isls(
    topology_with_isls: LEOTopology,
    undirected_isls: list,
    current_time_absolute: Time,
):
    """
    Computes the inter-satellite links (ISLs) based on the provided constellation data.
    The function creates a network graph representing the ISLs and verifies that the
    distances between satellites do not exceed the maximum ISL length.
    It also generates a mapping of satellite neighbors to interface numbers.
    """
    constellation_data = topology_with_isls.constellation_data
    num_isls_per_sat = [0] * constellation_data.number_of_satellites
    for satellite_id_a, satellite_id_b in undirected_isls:
        # ISLs are not permitted to exceed their maximum distance
        # TODO: Technically, they can (could just be ignored by forwarding state calculation),
        # TODO: but practically, defining a permanent ISL between two satellites which
        # TODO: can go out of distance is generally unwanted
        sat_distance_m = distance_tools.distance_m_between_satellites(
            constellation_data.satellites[satellite_id_a],
            constellation_data.satellites[satellite_id_b],
            str(constellation_data.epoch),
            str(current_time_absolute),
        )
        if sat_distance_m > constellation_data.max_isl_length_m:
            raise ValueError(
                "The distance between two satellites (%d and %d) "
                "with an ISL exceeded the maximum ISL length (%.2fm > %.2fm at t=%s)"
                % (
                    satellite_id_a,
                    satellite_id_b,
                    sat_distance_m,
                    constellation_data.max_isl_length_m,
                    str(current_time_absolute),
                )
            )
        # Add to networkx graph
        topology_with_isls.graph.add_edge(satellite_id_a, satellite_id_b, weight=sat_distance_m)
        # Interface mapping of ISLs
        topology_with_isls.sat_neighbor_to_if[(satellite_id_a, satellite_id_b)] = num_isls_per_sat[
            satellite_id_a
        ]
        topology_with_isls.sat_neighbor_to_if[(satellite_id_b, satellite_id_a)] = num_isls_per_sat[
            satellite_id_b
        ]
        satellite_a = topology_with_isls.get_satellite(satellite_id_a)
        satellite_b = topology_with_isls.get_satellite(satellite_id_b)
        satellite_a.number_isls += 1
        satellite_b.number_isls += 1
        topology_with_isls.number_of_isls += 1
        log.debug("  > Total ISLs............. " + str(len(undirected_isls)))
        log.debug("  > Min. ISLs/satellite.... " + str(np.min(num_isls_per_sat)))
        log.debug("  > Max. ISLs/satellite.... " + str(np.max(num_isls_per_sat)))


def _build_topologies(orbital_data: ConstellationData, ground_stations: list[GroundStation]):
    """
    Builds two network graphs representing the satellite network topology.

    This function creates two instances of `LEOTopology`:
    1. A graph containing only satellite nodes with inter-satellite links (ISLs).
    2. A graph containing both satellite and ground station nodes with ground-satellite links (GSLs).

    Nodes are added to the respective graphs based on the number of satellites and ground stations.
    Debug logs provide information about the number of satellites, ground stations, and the maximum
    ranges for ISLs and GSLs.

    :param orbital_data: Data describing the satellite constellation, including
        the number of satellites, maximum ISL range, and maximum GSL range.
    :type orbital_data: ConstellationData
    :param ground_stations: A list of ground station objects.
    :type ground_stations: list[GroundStation]

    :return: A tuple containing:
        - `sat_net_graph_only_satellites_with_isls` (LEOTopology): A graph with only satellite nodes and ISLs.
        - `sat_net_graph_all_with_only_gsls` (LEOTopology): A graph with both satellite and ground station nodes, including GSLs.
    :rtype: tuple

    Logs:
        - Number of satellites (including ISL nodes).
        - Number of ground stations (including GSL nodes).
        - Maximum range for GSLs.
        - Maximum range for ISLs.
    """
    sat_net_graph_only_satellites_with_isls = LEOTopology(orbital_data, ground_stations)
    sat_net_graph_all_with_only_gsls = LEOTopology(orbital_data, ground_stations)
    for i in range(orbital_data.number_of_satellites):
        sat_net_graph_only_satellites_with_isls.graph.add_node(i)
        sat_net_graph_all_with_only_gsls.graph.add_node(i)
    for i in range(orbital_data.number_of_satellites + len(ground_stations)):
        sat_net_graph_all_with_only_gsls.graph.add_node(i)
    log.debug(
        "  > Number of satellites.... "
        + str(orbital_data.number_of_satellites)
        + " (including ISL nodes)"
    )
    log.debug(
        "  > Number of ground stations " + str(len(ground_stations)) + " (including GSL nodes)"
    )
    log.debug("  > Max. range GSL......... " + str(orbital_data.max_gsl_length_m) + " m")
    log.debug("  > Max. range ISL......... " + str(orbital_data.max_isl_length_m) + " m")
    return (
        sat_net_graph_only_satellites_with_isls,
        sat_net_graph_all_with_only_gsls,
    )


def _compute_gsl_interface_information(topology: LEOTopology):
    """
    Logs information about GSL interfaces for satellites and ground stations.

    :param list_gsl_interfaces_info: List of GSL interface information.
    :param satellites: List of satellites.
    :param ground_stations: List of ground stations.
    :param enable_verbose_logs: Boolean to enable verbose logging.
    """
    constellation_data = topology.constellation_data
    log.debug("GSL INTERFACE INFORMATION")
    satellite_gsl_if_count_list = list(
        map(
            lambda x: x["number_of_interfaces"],
            topology.gsl_interfaces_info[0 : constellation_data.number_of_satellites],
        )
    )
    ground_station_gsl_if_count_list = list(
        map(
            lambda x: x["number_of_interfaces"],
            topology.gsl_interfaces_info[
                constellation_data.number_of_satellites : (
                    constellation_data.number_of_satellites + topology.number_of_ground_stations
                )
            ],
        )
    )
    log.debug("  > Min. GSL IFs/satellite........ " + str(np.min(satellite_gsl_if_count_list)))
    log.debug("  > Max. GSL IFs/satellite........ " + str(np.max(satellite_gsl_if_count_list)))
    log.debug("  > Min. GSL IFs/ground station... " + str(np.min(ground_station_gsl_if_count_list)))
    log.debug("  > Max. GSL IFs/ground_station... " + str(np.max(ground_station_gsl_if_count_list)))


def _compute_ground_station_satellites_in_range(
    topology: LEOTopology, current_time: astro_units.Quantity
):
    log.debug("\nGSL IN-RANGE INFORMATION")
    # What satellites can a ground station see
    ground_station_satellites_in_range = []
    satellites = topology.get_satellites()
    for ground_station in topology.ground_stations:
        satellites_in_range = []
        for satellite in satellites:
            distance_m = distance_tools.distance_m_ground_station_to_satellite(
                ground_station,
                satellite.position,
                topology.constellation_data.epoch,
                str(current_time),
            )
            if distance_m <= topology.constellation_data.max_gsl_length_m:
                satellites_in_range.append((distance_m, satellite.id))
                topology.graph.add_edge(satellite.id, ground_station.id, weight=distance_m)

        ground_station_satellites_in_range.append(satellites_in_range)

    ground_station_num_in_range = list(map(lambda x: len(x), ground_station_satellites_in_range))
    log.debug("  > Min. satellites in range... " + str(np.min(ground_station_num_in_range)))
    log.debug("  > Max. satellites in range... " + str(np.max(ground_station_num_in_range)))


def generate_dynamic_state_at(
    output_dynamic_state_dir,
    epoch,
    time_since_epoch_ns,
    constellation_data: ConstellationData,
    ground_stations: list[GroundStation],
    undirected_isls,
    list_gsl_interfaces_info,
    dynamic_state_algorithm,
    prev_output,
):
    log.debug(
        "FORWARDING STATE AT T = "
        + (str(time_since_epoch_ns))
        + "ns (= "
        + str(time_since_epoch_ns / 1e9)
        + " seconds)"
    )
    log.debug("BASIC INFORMATION")
    time = epoch + time_since_epoch_ns * astro_units.ns
    log.debug("  > Epoch.................. " + str(epoch))
    log.debug("  > Time since epoch....... " + str(time_since_epoch_ns) + " ns")
    log.debug("  > Absolute time.......... " + str(time))

    topology_with_isls, topology_only_gs = _build_topologies(constellation_data, ground_stations)
    log.debug("ISL INFORMATION")
    time_absolute = epoch + time_since_epoch_ns * astro_units.ns
    _compute_isls(topology_with_isls, undirected_isls, time_absolute)
    log.debug("\nGSL INTERFACE INFORMATION")
    # TODO Think about used data structures, because we are computing this twice and we shouldn't
    _compute_gsl_interface_information(topology_only_gs)
    _compute_gsl_interface_information(topology_with_isls)
    _compute_ground_station_satellites_in_range(topology_only_gs, time_since_epoch_ns)
    _compute_ground_station_satellites_in_range(topology_with_isls, time_since_epoch_ns)
    #
    # Call the dynamic state algorithm which:
    #
    # (a) Output the gsl_if_bandwidth_<t>.txt files
    # (b) Output the fstate_<t>.txt files
    #
    if dynamic_state_algorithm == "algorithm_free_one_only_over_isls":

        return algorithm_free_one_only_over_isls(
            output_dynamic_state_dir,
            time_since_epoch_ns,
            satellites,
            ground_stations,
            sat_net_graph_only_satellites_with_isls,
            ground_station_satellites_in_range,
            num_isls_per_sat,
            sat_neighbor_to_if,
            list_gsl_interfaces_info,
            prev_output,
            enable_verbose_logs,
        )

    else:
        raise ValueError("Unknown dynamic state algorithm: " + str(dynamic_state_algorithm))
