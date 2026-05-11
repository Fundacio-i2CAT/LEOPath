from astropy import units as astro_units
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from leopath.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

from .explicit_path_routing_algorithm import algorithm_explicit_path_routing


class ExplicitPathRoutingAlgorithm(RoutingAlgorithm):
    """Protocol-agnostic centrally planned explicit-path proxy."""

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
        params = algorithm_params or {}
        gsl_strategy = GSLAttachmentFactory.get_strategy("nearest_satellite")
        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        current_time = epoch + (time_since_epoch_ns / 1e9) * astro_units.second
        refresh_interval = int(params.get("segment_refresh_interval_steps", 1))
        if refresh_interval < 1:
            refresh_interval = 1

        current_step_index = 0
        time_step_minutes = params.get("time_step_minutes")
        if time_step_minutes:
            step_ns = int(float(time_step_minutes) * 60 * 1e9)
            if step_ns > 0:
                current_step_index = time_since_epoch_ns // step_ns

        should_refresh = (
            not hasattr(self, "_cached_route_plans")
            or getattr(self, "_cached_route_plans", None) is None
            or current_step_index % refresh_interval == 0
        )
        cached_route_plans = None if should_refresh else getattr(self, "_cached_route_plans", None)

        output = algorithm_explicit_path_routing(
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            topology_with_isls=topology_with_isls,
            gsl_attachment_strategy=gsl_strategy,
            current_time=current_time,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            algorithm_params=params,
            current_ground_station_satellites_in_range=ground_station_satellites_in_range,
            cached_route_plans=cached_route_plans,
            control_plane_metadata={
                "step_index": current_step_index,
                "planning_step_index": getattr(self, "_cached_planning_step_index", None),
            },
        )
        if should_refresh:
            self._cached_route_plans = output.get("route_plans", {})
            self._cached_planning_step_index = current_step_index
            output.setdefault("control_plane", {})["planning_step_index"] = current_step_index
        return output
