from astropy import units as astro_units
from astropy.time import Time

from leopath.network_state.gsl_attachment.gsl_attachment_factory import GSLAttachmentFactory
from leopath.network_state.helpers import _build_topologies, _compute_ground_station_satellites_in_range, _compute_isls
from leopath.network_state.routing_algorithms.routing_algorithm import RoutingAlgorithm
from leopath.topology.topology import ConstellationData, GroundStation, LEOTopology

from .traditional_segment_routing_algorithm import (
    DEFAULT_SRV6_LOCATOR_PREFIX,
    algorithm_traditional_segment_routing,
    _build_segment_plans,
)


class TraditionalSegmentRoutingAlgorithm(RoutingAlgorithm):
    """
    Traditional segment routing baseline with explicit segment lists.

    The control plane computes a per-(source sat, destination GS) SRv6 segment
    list [sid1, sid2, ..., dst_sid]. The simulator keeps numeric next hops in
    the materialized forwarding state for compatibility with existing metrics.
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

        epoch = Time("2000-01-01 00:00:00", scale="tdb")
        current_time = epoch + (time_since_epoch_ns / 1e9) * astro_units.second

        planning_time = current_time
        refresh_interval = int(params.get("segment_refresh_interval_steps", 1))
        if refresh_interval < 1:
            refresh_interval = 1

        horizon_minutes = float(params.get("prediction_horizon_minutes", 0.0))
        if horizon_minutes != 0.0:
            planning_time = current_time + horizon_minutes * astro_units.minute

        current_step_index = 0
        time_step_minutes = params.get("time_step_minutes")
        if time_step_minutes:
            step_ns = int(float(time_step_minutes) * 60 * 1e9)
            if step_ns > 0:
                current_step_index = time_since_epoch_ns // step_ns

        should_refresh = (
            not hasattr(self, "_cached_segment_plans")
            or getattr(self, "_cached_segment_plans", None) is None
            or getattr(self, "_cached_planning_visibility", None) is None
            or current_step_index % refresh_interval == 0
        )

        planning_visibility = getattr(
            self,
            "_cached_planning_visibility",
            ground_station_satellites_in_range,
        )

        if should_refresh:
            planning_topology = topology_with_isls
            undirected_isls = params.get("undirected_isls")
            srv6_locator_prefix = params.get(
                "srv6_locator_prefix",
                DEFAULT_SRV6_LOCATOR_PREFIX,
            )
            if horizon_minutes != 0.0 and undirected_isls:
                planning_topology, _ = _build_topologies(constellation_data, ground_stations)
                planning_topology.gsl_interfaces_info = list_gsl_interfaces_info
                _compute_isls(planning_topology, undirected_isls, planning_time)
                planning_visibility = _compute_ground_station_satellites_in_range(
                    planning_topology, planning_time
                )
            else:
                planning_visibility = ground_station_satellites_in_range

            self._cached_segment_plans = _build_segment_plans(
                planning_topology,
                ground_stations,
                planning_visibility,
                int(params.get("segment_count", 2)),
                srv6_locator_prefix,
            )
            self._cached_planning_visibility = planning_visibility
            self._cached_plan_metadata = {
                "step_index": current_step_index,
                "planning_time": str(planning_time),
                "srv6_locator_prefix": srv6_locator_prefix,
                "sample_segment_plans": [
                    {
                        "source_satellite_id": src_sat_id,
                        "destination_ground_station_id": dst_gs_id,
                        "segment_sids": segment_sids,
                    }
                    for (src_sat_id, dst_gs_id), segment_sids in list(
                        self._cached_segment_plans.items()
                    )[:5]
                ],
            }

        output = algorithm_traditional_segment_routing(
            constellation_data,
            ground_stations,
            topology_with_isls,
            gsl_strategy,
            current_time,
            list_gsl_interfaces_info,
            params,
            segment_plans=getattr(self, "_cached_segment_plans", None),
            planning_ground_station_satellites_in_range=planning_visibility,
            current_ground_station_satellites_in_range=ground_station_satellites_in_range,
        )
        output["control_plane"] = getattr(self, "_cached_plan_metadata", {})
        return output
