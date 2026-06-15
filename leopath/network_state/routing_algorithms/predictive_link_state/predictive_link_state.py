from astropy import units as astro_units
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from leopath.network_state.helpers import _build_topologies, _compute_isls
from leopath.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from leopath.network_state.routing_algorithms.shortest_path_link_state_routing.fstate_calculation import (
    calculate_fstate_shortest_path_object_no_gs_relay,
)
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology


class PredictiveLinkStateRoutingAlgorithm(RoutingAlgorithm):
    """
    Link-state routing with predictive topology lookup.

    It computes shortest paths over a topology snapshot at time t + prediction_horizon.
    """

    def compute_state(
        self,
        time_since_epoch_ns: int,
        constellation_data: ConstellationData,
        ground_stations: list[GroundStation],
        topology_with_isls: LEOTopology,
        ground_station_satellites_in_range: list,
        list_gsl_interfaces_info: list,
        algorithm_params: dict | None = None,
    ) -> dict:
        gsl_strategy = GSLAttachmentFactory.get_strategy("nearest_satellite")

        params = algorithm_params or {}
        horizon_minutes = float(params.get("prediction_horizon_minutes", 0.0))
        horizon_ns = int(horizon_minutes * 60 * 1e9)

        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        current_time = epoch + (time_since_epoch_ns + horizon_ns) * astro_units.ns

        topology_for_calc = topology_with_isls
        undirected_isls = params.get("undirected_isls")
        if undirected_isls:
            topology_for_calc, _ = _build_topologies(constellation_data, ground_stations)
            topology_for_calc.gsl_interfaces_info = list_gsl_interfaces_info
            _compute_isls(topology_for_calc, undirected_isls, current_time)

        fstate = calculate_fstate_shortest_path_object_no_gs_relay(
            topology_for_calc,
            ground_stations,
            gsl_strategy,
            current_time,
        )

        bandwidth_state = _calculate_bandwidth_state(
            constellation_data, ground_stations, list_gsl_interfaces_info
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
    num_satellites = constellation_data.number_of_satellites
    num_total_nodes = num_satellites + len(ground_stations)
    bandwidth_state = {}
    for i in range(num_total_nodes):
        if i < len(list_gsl_interfaces_info):
            node_info = list_gsl_interfaces_info[i]
            node_id = node_info.get("id", i)
            bandwidth = node_info.get("aggregate_max_bandwidth", 0.0)
        else:
            node_id = i
            bandwidth = 0.0
        bandwidth_state[node_id] = bandwidth
    return bandwidth_state
