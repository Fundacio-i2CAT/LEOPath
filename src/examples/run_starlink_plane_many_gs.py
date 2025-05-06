# run_starlink_plane_example_many_gs.py

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

# Import necessary components from your src directory
from src.dynamic_state.generate_dynamic_state import generate_dynamic_state
from src.dynamic_state.topology import ConstellationData, GroundStation, Satellite

# --- Configure Logging ---
log = logger.get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s (%(filename)s:%(lineno)d)",
)

# Constants
EARTH_RADIUS_M = 6378135.0


def run_simulation_many_gs():
    """
    Sets up and runs the dynamic state simulation for a single orbital plane
    of Starlink satellites and MANY ground stations over one orbital period.
    """
    log.info("--- Setting up Single Plane Simulation with Many Ground Stations ---")

    # --- Simulation Parameters ---
    tle_epoch_str = "25112.58592294"
    epoch = Time(2025, format="jyear") + (112.58592294 - 1) * astro_units.day
    log.info(f"Using TLE Epoch: {tle_epoch_str}")
    log.info(f"Converted Astropy Epoch (UTC): {epoch.utc.iso}")

    duration_s = 5800  # ~96.6 minutes
    time_step_s = 60  # Calculate state every 60 seconds
    offset_s = 0

    time_step_ns = int(time_step_s * 1e9)
    simulation_end_time_ns = int(duration_s * 1e9)
    offset_ns = int(offset_s * 1e9)

    dynamic_state_algorithm = "algorithm_free_one_only_over_isls"
    output_dir = None  # Not writing intermediate state files

    # Max link lengths (~550km altitude, 25 deg elevation mask)
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
    # Satellites (Keep the same 7 from the plane)
    sat_ids = list(range(7))
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
        try:
            ephem_obj = ephem.readtle(tle_tuple[0], tle_tuple[1], tle_tuple[2])
            satellites.append(
                Satellite(id=sat_id, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
            )
        except ValueError as e:
            log.error(f"Failed to read TLE for satellite {sat_id} ({tle_tuple[0]}): {e}")
            return

    # Ground Stations (Use ~15 diverse locations)
    # Start GS IDs after last satellite ID
    gs_start_id = len(sat_ids)
    gs_defs = [
        {"name": "Zaragoza", "lat": 41.65, "lon": -0.88, "elv": 200.0},
        {"name": "New_York", "lat": 40.7, "lon": -74.0, "elv": 10.0},
        {"name": "London", "lat": 51.5, "lon": -0.1, "elv": 35.0},
        {"name": "Berlin", "lat": 52.5, "lon": 13.4, "elv": 34.0},
        {"name": "Rome", "lat": 41.9, "lon": 12.5, "elv": 20.0},
        {"name": "Los_Angeles", "lat": 34.05, "lon": -118.24, "elv": 71.0},
        {"name": "Chicago", "lat": 41.88, "lon": -87.63, "elv": 181.0},
        {"name": "Sao_Paulo", "lat": -23.55, "lon": -46.63, "elv": 760.0},
        {"name": "Bogota", "lat": 4.71, "lon": -74.07, "elv": 2640.0},
        {"name": "Lagos", "lat": 6.45, "lon": 3.4, "elv": 41.0},
        {"name": "Johannesburg", "lat": -26.2, "lon": 28.04, "elv": 1753.0},
        {"name": "Tokyo", "lat": 35.68, "lon": 139.69, "elv": 40.0},
        {"name": "Singapore", "lat": 1.35, "lon": 103.8, "elv": 15.0},
        {"name": "Delhi", "lat": 28.7, "lon": 77.1, "elv": 216.0},
        {"name": "Sydney", "lat": -33.86, "lon": 151.2, "elv": 58.0},
        # Additional GSs in the USA
        {"name": "San_Francisco", "lat": 37.77, "lon": -122.42, "elv": 16.0},
        {"name": "Seattle", "lat": 47.61, "lon": -122.33, "elv": 52.0},
        {"name": "Denver", "lat": 39.74, "lon": -104.99, "elv": 1609.0},
        {"name": "Phoenix", "lat": 33.45, "lon": -112.07, "elv": 331.0},
        {"name": "Las_Vegas", "lat": 36.17, "lon": -115.14, "elv": 610.0},
        {"name": "Salt_Lake_City", "lat": 40.76, "lon": -111.89, "elv": 1288.0},
        {"name": "Portland", "lat": 45.52, "lon": -122.68, "elv": 15.0},
        {"name": "Houston", "lat": 29.76, "lon": -95.37, "elv": 13.0},
        {"name": "Miami", "lat": 25.76, "lon": -80.19, "elv": 2.0},
        {"name": "Anchorage", "lat": 61.22, "lon": -149.90, "elv": 31.0},
        # Additional GSs in Canada
        {"name": "Toronto", "lat": 43.7, "lon": -79.42, "elv": 76.0},
        {"name": "Vancouver", "lat": 49.28, "lon": -123.12, "elv": 70.0},
        {"name": "Montreal", "lat": 45.5, "lon": -73.57, "elv": 36.0},
        {"name": "Calgary", "lat": 51.05, "lon": -114.07, "elv": 1045.0},
        {"name": "Edmonton", "lat": 53.55, "lon": -113.49, "elv": 645.0},
        {"name": "Winnipeg", "lat": 49.89, "lon": -97.14, "elv": 239.0},
        {"name": "Ottawa", "lat": 45.42, "lon": -75.69, "elv": 70.0},
        {"name": "Halifax", "lat": 44.65, "lon": -63.57, "elv": 140.0},
        {"name": "Quebec_City", "lat": 46.81, "lon": -71.21, "elv": 98.0},
    ]
    ground_stations = []
    gs_ids = []
    current_gs_id = gs_start_id
    for data in gs_defs:
        try:
            x, y, z = geodetic2cartesian(data["lat"], data["lon"], data["elv"])
            ground_stations.append(
                GroundStation(
                    gid=current_gs_id,
                    name=data["name"],
                    latitude_degrees_str=str(data["lat"]),
                    longitude_degrees_str=str(data["lon"]),
                    elevation_m_float=data["elv"],
                    cartesian_x=x,
                    cartesian_y=y,
                    cartesian_z=z,
                )
            )
            gs_ids.append(current_gs_id)
            current_gs_id += 1
        except Exception as e:
            log.error(f"Could not create ground station {data['name']}: {e}")
            # Decide whether to stop or continue without this GS
            # return

    all_node_ids = sat_ids + gs_ids
    log.info(f"Created {len(satellites)} satellites and {len(ground_stations)} ground stations.")

    # ConstellationData
    constellation_data = ConstellationData(
        orbits=1,
        sats_per_orbit=len(satellites),
        epoch=tle_epoch_str,
        max_gsl_length_m=max_gsl_length_m,
        max_isl_length_m=max_isl_length_m,
        satellites=satellites,
    )

    # Define ISLs (still fragmented based on previous findings for this TLE set)
    undirected_isls = [
        (0, 1),
        (1, 2),
        (3, 4),
        #    (5, 6)
    ]
    log.info(f"Defined {len(undirected_isls)} potential in-plane ISLs.")

    # GSL Interface Info (7 Sats + N GS)
    list_gsl_interfaces_info = [
        {"id": node_id, "number_of_interfaces": 1, "aggregate_max_bandwidth": 10.0}  # Example BW
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
        valid_states = [s for s in all_states if s is not None]
        num_steps = len(all_states)
        log.info(
            f"Generated {len(valid_states)} valid state entries out of {num_steps} total steps."
        )

        # --- Example Analysis: Find connected GS pairs at specific times ---
        print("\n--- Connectivity Check (Any GS <-> Any other GS) ---")
        print("Time (s) | Connected Pairs (Showing first hop)")
        print("---------|----------------------------------------")
        for i, state in enumerate(all_states):
            time_s = (offset_ns + i * time_step_ns) / 1e9
            connected_pairs_str = ""
            if state and isinstance(state.get("fstate"), dict):
                fstate = state["fstate"]
                found_connection = False
                # Iterate through all GS pairs
                for src_gs_id in gs_ids:
                    for dst_gs_id in gs_ids:
                        if src_gs_id == dst_gs_id:
                            continue
                        hop_tuple = fstate.get((src_gs_id, dst_gs_id), (-1, -1, -1))
                        if hop_tuple[0] != -1:  # If not DROP
                            connected_pairs_str += (
                                f" ({src_gs_id}->{dst_gs_id}: Sat {hop_tuple[0]})"
                            )
                            found_connection = True
                if not found_connection:
                    connected_pairs_str = " None"

            elif state is None:
                connected_pairs_str = " ERROR in state calc"
            else:
                connected_pairs_str = " Invalid state format"

            # Print only if connections change or on first/last step?
            # For now print all, can be long
            print(f"{time_s:<8.1f} |{connected_pairs_str}")

        # Save the full results to a file
        results_filename = "starlink_many_gs_results.pkl"
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
    log.info("Starting Starlink single plane simulation with many GS...")
    run_simulation_many_gs()
    log.info("Example script finished.")
