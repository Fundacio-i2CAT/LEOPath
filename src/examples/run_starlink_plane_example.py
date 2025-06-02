# run_starlink_plane_example.py
# Corrected ISL definition further

import logging
import math
import os
import pickle
import pprint

import ephem
from astropy import units as astro_units
from astropy.time import Time

from src import logger
from src.distance_tools import geodetic2cartesian
from src.dynamic_state.generate_dynamic_state import generate_dynamic_state
from src.topology.topology import ConstellationData, GroundStation, Satellite

log = logger.get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s (%(filename)s:%(lineno)d)",
)

# Constants
EARTH_RADIUS_M = 6378135.0


def run_simulation():
    """
    Sets up and runs the dynamic state simulation for a single orbital plane
    of Starlink satellites and two ground stations over one orbital period.
    """
    log.info("--- Setting up Single Plane Simulation ---")
    tle_epoch_str = "25112.58592294"
    epoch = Time(2025, format="jyear") + (112.58592294 - 1) * astro_units.day
    log.info(f"Using TLE Epoch: {tle_epoch_str}")
    log.info(f"Converted Astropy Epoch (UTC): {epoch.utc.iso}")

    duration_s = 5800
    time_step_s = 60
    offset_s = 0
    time_step_ns = int(time_step_s * 1e9)
    simulation_end_time_ns = int(duration_s * 1e9)
    offset_ns = int(offset_s * 1e9)
    dynamic_state_algorithm = "algorithm_free_one_only_over_isls"
    output_dir = None

    # Max link lengths
    altitude_m = 550000
    min_elevation_deg = 25.0
    satellite_cone_radius_m = altitude_m / math.tan(math.radians(min_elevation_deg))
    max_gsl_length_m = math.sqrt(satellite_cone_radius_m**2 + altitude_m**2)
    max_isl_length_m = 2 * math.sqrt(
        math.pow(EARTH_RADIUS_M + altitude_m, 2) - math.pow(EARTH_RADIUS_M + 80000, 2)
    )
    log.info(f"Max GSL Length: {max_gsl_length_m / 1000:.1f} km")
    log.info(f"Max ISL Length: {max_isl_length_m / 1000:.1f} km")

    # --- Scenario Definition ---
    sat_ids = list(range(7))
    GS_ZAR_ID = 7
    GS_NYC_ID = 8
    gs_ids = [GS_ZAR_ID, GS_NYC_ID]
    all_node_ids = sat_ids + gs_ids

    # TLE Data from your script's output
    tles_in_plane = [
        (
            "STARLINK-1008",
            "1 44714U 19074B   25112.58592294  .00005641  00000+0  39726-3 0  9991",
            "2 44714  53.0538 188.1053 0001311  93.0175 267.0964 15.06401971300352",
        ),
        (
            "STARLINK-1010",
            "1 44716U 19074D   25112.59326790 -.00012419  00000+0 -81623-3 0  9996",
            "2 44716  53.0539 188.0720 0001737  85.8677 274.2511 15.06401699300336",
        ),
        (
            "STARLINK-1017",
            "1 44723U 19074L   25112.59697524  .00007108  00000+0  49566-3 0  9997",
            "2 44723  53.0535 188.0553 0001759  67.5126 292.6049 15.06396533300356",
        ),
        (
            "STARLINK-1019",
            "1 44724U 19074M   25112.62646649 -.00005710  00000+0 -36491-3 0  9996",
            "2 44724  53.0543 187.9213 0001543 118.8653 241.2491 15.06385414300350",
        ),
        (
            "STARLINK-1021",
            "1 44726U 19074P   25112.55643853  .00003310  00000+0  24094-3 0  9992",
            "2 44726  53.0534 188.2388 0001160  84.3463 275.7658 15.06411559300346",
        ),
        (
            "STARLINK-2461",
            "1 48428U 21040A   25112.60435494  .00001170  00000+0  97435-4 0  9999",
            "2 48428  53.0549 188.0134 0001489 108.0728 252.0423 15.06399276218161",
        ),
        (
            "STARLINK-2579",
            "1 48459U 21040AH  25112.58228272  .00004403  00000+0  31429-3 0  9996",
            "2 48459  53.0531 188.4798 0001450  70.3257 289.7888 15.06400187218168",
        ),
    ]
    satellites = []
    for i, tle_tuple in enumerate(tles_in_plane):
        sat_id = sat_ids[i]
        ephem_obj = ephem.readtle(*tle_tuple)
        satellites.append(
            Satellite(id=sat_id, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
        )

    # Ground Stations
    gs_defs = {
        GS_ZAR_ID: {"name": "Zaragoza", "lat": 41.65, "lon": -0.88, "elv": 200.0},
        GS_NYC_ID: {"name": "New_York", "lat": 40.7, "lon": -74.0, "elv": 10.0},
    }
    ground_stations = []
    for gid, data in gs_defs.items():
        x, y, z = geodetic2cartesian(data["lat"], data["lon"], data["elv"])
        ground_stations.append(
            GroundStation(
                gid=gid,
                name=data["name"],
                latitude_degrees_str=str(data["lat"]),
                longitude_degrees_str=str(data["lon"]),
                elevation_m_float=data["elv"],
                cartesian_x=x,
                cartesian_y=y,
                cartesian_z=z,
            )
        )

    constellation_data = ConstellationData(
        orbits=1,
        sats_per_orbit=len(satellites),
        epoch=tle_epoch_str,
        max_gsl_length_m=max_gsl_length_m,
        max_isl_length_m=max_isl_length_m,
        satellites=satellites,
    )

    # **** CORRECTED ISL LIST AGAIN ****
    # Define ISLs explicitly, removing pairs known to be too far apart at t=0
    undirected_isls = [
        (0, 1),
        (1, 2),
        # (2, 3), # Removed (too far)
        (3, 4),
        # (4, 5), # Removed (too far)
        # (5, 6),
    ]
    # Optional: add wrap-around link if appropriate (e.g., (6, 0))
    # Check distance for (6, 0) if adding wrap-around before enabling.
    log.info(
        f"Defined {len(undirected_isls)} potential in-plane ISLs (removed links > max distance at t=0)."
    )

    # GSL Interface Info
    list_gsl_interfaces_info = [
        {"id": node_id, "number_of_interfaces": 1, "aggregate_max_bandwidth": 10.0}
        for node_id in all_node_ids
    ]

    # --- Run Simulation ---
    log.info(f"Running simulation for {duration_s}s with {time_step_s}s step...")
    all_states = generate_dynamic_state(
        output_dynamic_state_dir=output_dir,
        epoch=epoch,
        simulation_end_time_ns=simulation_end_time_ns,
        time_step_ns=time_step_ns,
        offset_ns=offset_ns,
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        undirected_isls=undirected_isls,
        list_gsl_interfaces_info=list_gsl_interfaces_info,
        dynamic_state_algorithm=dynamic_state_algorithm,
    )
    log.info("Simulation run finished.")

    # --- Process Results ---
    if all_states and any(s is not None for s in all_states):
        # ... (rest of the result processing and saving code remains the same) ...
        valid_states = [s for s in all_states if s is not None]
        num_steps = len(all_states)
        log.info(
            f"Generated {len(valid_states)} valid state entries out of {num_steps} total steps."
        )

        # Example: Print the first hop for ZAR -> NYC at each step
        print("\n--- Path Evolution (First Hop ZAR -> NYC) ---")
        print("Time (s) | Next Hop ID | My IF | Next Hop IF")
        print("---------|-------------|-------|------------")
        for i, state in enumerate(all_states):
            time_s = (offset_ns + i * time_step_ns) / 1e9
            if state and isinstance(state.get("fstate"), dict):
                fstate = state["fstate"]
                hop_tuple = fstate.get((GS_ZAR_ID, GS_NYC_ID), ("N/A", -1, -1))
                next_hop_str = str(hop_tuple[0]) if hop_tuple[0] != -1 else "DROP"
                my_if_str = str(hop_tuple[1]) if hop_tuple[1] != -1 else " "
                nh_if_str = str(hop_tuple[2]) if hop_tuple[2] != -1 else " "
                print(f"{time_s:<8.1f} | {next_hop_str:<11} | {my_if_str:<5} | {nh_if_str:<10}")
            else:
                print(f"{time_s:<8.1f} | --- ERROR --- |   -   |      -")

        results_filename = "starlink_plane_orbit_results.pkl"
        try:
            with open(results_filename, "wb") as f:
                pickle.dump(all_states, f)
            log.info(f"Full simulation state list saved to: {results_filename}")
            print(f"\nResult states saved to {results_filename}")
        except Exception as e:
            log.error(f"Failed to save results to {results_filename}: {e}")
    else:
        log.warning("Simulation produced no valid state results.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s (%(filename)s:%(lineno)d)",
    )
    log.info("Starting Starlink single plane simulation example...")
    run_simulation()
    log.info("Example script finished.")
