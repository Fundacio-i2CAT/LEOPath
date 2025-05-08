# Filename: leo-routing-simu/src/examples/run_starlink_plane_many_gs.py
# Corrected version: Passes Astropy Time object to generate_dynamic_state

import logging
import math
import os
import pickle
import pprint

# --- Corrected Imports based on Project Structure ---
try:
    from src import logger
    from src.distance_tools.distance_tools import geodetic2cartesian
    from src.dynamic_state.generate_dynamic_state import generate_dynamic_state
    from src.dynamic_state.topology import ConstellationData, GroundStation, Satellite
    import ephem
    from astropy import units as astro_units
    from astropy.time import Time
    from skyfield.api import load  # Still needed for timestamp in metadata
except ImportError as e:
    print(
        f"Import Error: {e}. Please ensure the script is run from the project root or src is in PYTHONPATH."
    )
    exit()
# --- End Imports ---


# --- Configure Logging ---
log = logger.get_logger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s (%(filename)s:%(lineno)d)",
)

# Constants
EARTH_RADIUS_M = 6378135.0


def run_simulation_many_gs(output_filename="starlink_many_gs_results_with_metadata.pkl"):
    """
    Sets up and runs the dynamic state simulation for a single orbital plane
    of Starlink satellites and MANY ground stations over one orbital period.
    Saves the results along with metadata to a PKL file.

    :param output_filename: Name of the output PKL file.
    """
    log.info("--- Setting up Single Plane Simulation with Many Ground Stations ---")

    # --- Simulation Parameters ---
    tle_epoch_str = "25112.58592294"
    # Create Astropy Time object (needed by generate_dynamic_state)
    try:
        epoch_astropy = Time(2025, format="jyear") + (112.58592294 - 1) * astro_units.day
        log.info(f"Using TLE Epoch String: {tle_epoch_str}")
        log.info(f"Astropy Epoch (UTC): {epoch_astropy.utc.iso}")
    except Exception as e:
        log.exception(f"Could not create Astropy Time from TLE epoch string {tle_epoch_str}: {e}")
        return  # Cannot proceed without epoch

    duration_s = 5800
    time_step_s = 60
    offset_s = 0

    time_step_ns = int(time_step_s * 1e9)
    simulation_end_time_ns = int(duration_s * 1e9)
    offset_ns = int(offset_s * 1e9)

    dynamic_state_algorithm = "algorithm_free_one_only_over_isls"
    output_dir = None  # generate_dynamic_state returns data, doesn't write files

    # --- Link Parameters ---
    altitude_m = 550000
    min_elevation_deg = 25.0
    satellite_cone_radius_m = altitude_m / math.tan(math.radians(min_elevation_deg))
    max_gsl_length_m = math.sqrt(satellite_cone_radius_m**2 + altitude_m**2)
    max_isl_length_m = 2 * math.sqrt(
        math.pow(EARTH_RADIUS_M + altitude_m, 2) - math.pow(EARTH_RADIUS_M + 80000, 2)
    )
    log.info(f"Max GSL Length: {max_gsl_length_m / 1000:.1f} km")
    log.info(f"Max ISL Length: {max_isl_length_m / 1000:.1f} km")

    # --- Scenario Definition (Satellites - Kept exactly as original) ---
    log.info("Defining satellite TLEs...")
    sat_ids = list(range(7))
    tles_in_plane = [  # TLEs from the original script
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

    # --- Scenario Definition (Ground Stations - Kept exactly as original) ---
    log.info("Defining ground stations...")
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

    all_node_ids = sat_ids + gs_ids
    log.info(
        f"Created {len(satellites)} satellites (IDs {sat_ids}) and {len(ground_stations)} ground stations (IDs {gs_ids})."
    )

    # --- Prepare ConstellationData (Keep as in original) ---
    constellation_data = ConstellationData(
        orbits=1,
        sats_per_orbit=len(satellites),
        epoch=tle_epoch_str,  # Uses TLE string epoch
        max_gsl_length_m=max_gsl_length_m,
        max_isl_length_m=max_isl_length_m,
        satellites=satellites,
    )

    # --- Define ISLs (Keep as in original) ---
    undirected_isls = [(0, 1), (1, 2), (3, 4)]
    log.info(f"Defined {len(undirected_isls)} potential in-plane ISLs: {undirected_isls}")

    # --- GSL Interface Info (Keep as in original) ---
    list_gsl_interfaces_info = [
        {"id": node_id, "number_of_interfaces": 1, "aggregate_max_bandwidth": 10.0}
        for node_id in all_node_ids
    ]

    # --- Run Simulation ---
    log.info(f"Running dynamic state generation for {duration_s}s with {time_step_s}s step...")
    try:
        # *** CORRECTED CALL: Pass the Astropy Time object 'epoch_astropy' ***
        # *** VERIFY the actual complete signature of your generate_dynamic_state function ***
        all_states = generate_dynamic_state(
            output_dynamic_state_dir=None,  # Ensures function returns data
            epoch=epoch_astropy,  # <<<< CORRECTED: Pass Astropy Time object
            simulation_end_time_ns=simulation_end_time_ns,
            time_step_ns=time_step_ns,
            offset_ns=offset_ns,
            constellation_data=constellation_data,
            ground_stations=ground_stations,
            undirected_isls=undirected_isls,
            list_gsl_interfaces_info=list_gsl_interfaces_info,
            dynamic_state_algorithm=dynamic_state_algorithm,
        )
        log.info("Dynamic state generation finished.")
    except TypeError as e:
        # Catch specific TypeError for epoch if it persists
        if "Epoch must be an astropy Time object" in str(e):
            log.error(f"generate_dynamic_state raised TypeError: {e}")
            log.error("Passed epoch type was: {}".format(type(epoch_astropy)))
        else:
            log.exception(f"TypeError occurred during generate_dynamic_state execution: {e}")
        return
    except Exception as e:
        log.exception(f"Error occurred during generate_dynamic_state execution: {e}")
        return

    # --- Process Results and Save with Metadata ---
    if all_states and isinstance(all_states, list) and any(s is not None for s in all_states):
        valid_states = [s for s in all_states if s is not None]
        num_steps = len(all_states)
        log.info(
            f"Generated {len(valid_states)} valid state entries out of {num_steps} total steps."
        )

        # --- Construct Metadata ---
        ts = load.timescale()
        metadata = {
            "scenario_description": "Starlink Single Plane Example with Many Ground Stations",
            "tle_epoch_string_used": tle_epoch_str,
            "epoch_generation_utc_iso": epoch_astropy.utc.iso if epoch_astropy else None,
            "num_satellites": len(satellites),
            "num_ground_stations": len(ground_stations),
            "ground_station_defs": gs_defs,
            "dynamic_state_update_interval_ms": time_step_s * 1000,
            "simulation_duration_s": duration_s,
            "offset_s": offset_s,
            "fstate_calculation_algorithm": dynamic_state_algorithm,
            # Format based on user clarification: {(c,d): (nh, if1, if2)}
            "fstate_data_format": "dict_tuple_nh_if1_if2",
            "fstate_data_key": "fstate",  # Assuming 'fstate' is the key in all_states dicts
            "generation_time_utc": ts.now().utc_iso(),
            "max_gsl_length_m": max_gsl_length_m,
            "max_isl_length_m": max_isl_length_m,
        }
        log.info("Constructed metadata for output.")
        # log.debug(f"Metadata: {metadata}") # Avoid logging potentially large gs_defs

        # --- Combine Metadata and Results ---
        final_data_to_save = {"metadata": metadata, "time_step_data": all_states}

        # --- Save the Combined Data ---
        output_save_dir = "."
        full_save_path = os.path.join(output_save_dir, output_filename)
        try:
            with open(full_save_path, "wb") as f:
                pickle.dump(final_data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            log.info(f"Full simulation state list with metadata saved to: {full_save_path}")
            print(f"\nResult states with metadata saved to {full_save_path}")
        except Exception as e:
            log.error(f"Failed to save results with metadata to {full_save_path}: {e}")
    else:
        log.warning("Simulation produced no valid state results to save.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
    )

    log.info("Starting Starlink single plane simulation with many GS...")
    run_simulation_many_gs(output_filename="starlink_many_gs_results_with_metadata.pkl")
    log.info("Example script finished.")
