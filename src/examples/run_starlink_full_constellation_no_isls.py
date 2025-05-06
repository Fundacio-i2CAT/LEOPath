# run_full_starlink_example.py

import logging
import math
import os
import pickle
import pprint
import re  # For TLE parsing

import ephem
from astropy import units as astro_units
from astropy.time import Time

# Import necessary components from your src directory
# Ensure these paths are correct relative to where you run the script
try:
    from src import logger
    from src.distance_tools import geodetic2cartesian
    from src.dynamic_state.generate_dynamic_state import generate_dynamic_state
    from src.dynamic_state.topology import ConstellationData, GroundStation, Satellite
except ImportError as e:
    print(f"Error importing necessary modules: {e}")
    print("Please ensure your PYTHONPATH is set correctly or run from the project root.")
    exit()

    # --- Configure Logging ---
logger.setup_logger(is_debug=True, file_name="starlink_full_constellation_no_isls.log")


# Constants
EARTH_RADIUS_M = 6378135.0
TLE_FILE = os.path.join(os.path.dirname(__file__), "starlink_first_shell.txt")


# --- Helper TLE Parser ---
def parse_tles_from_file(filename):
    """Parses TLE data, extracts NORAD ID, and first epoch string."""
    tles = []  # List to store tuples: ((name, line1, line2), norad_id)
    first_epoch_str = None
    try:
        with open(filename, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        print(f"Read {len(lines)} non-empty lines from {filename}.")
    except FileNotFoundError:
        print(f"TLE file not found: {filename}")
        return tles, None
    except Exception as e:
        print(f"Error reading TLE file {filename}: {e}")
        return tles, None

    if not lines:
        print("No non-empty lines found in TLE file.")
        return tles, None

    line_count = len(lines)
    processed_count = 0
    malformed_count = 0

    for i in range(0, line_count - (line_count % 3), 3):
        name = lines[i]
        line1 = lines[i + 1]
        line2 = lines[i + 2]

        if (
            line1.startswith("1 ")
            and line2.startswith("2 ")
            and len(line1) >= 69
            and len(line2) >= 69
            and line1[2:7] == line2[2:7]
        ):
            try:
                norad_id = int(line1[2:7].strip())  # Use NORAD ID from TLE
                if first_epoch_str is None:
                    epoch_yyddd_fraction = line1[18:32].strip()
                    if re.match(r"^\d{2}\d{3}\.\d{8}$", epoch_yyddd_fraction):
                        first_epoch_str = epoch_yyddd_fraction
                    else:
                        print(f"Could not parse epoch: {epoch_yyddd_fraction}")

                tles.append(((name, line1, line2), norad_id))
                processed_count += 1
            except ValueError:
                print(f"Non-numeric NORAD ID? Skipping TLE near line index {i}.")
                malformed_count += 1
            except Exception as e:
                print(f"Error processing TLE near {i}: {e}")
                malformed_count += 1
        else:
            print(f"Skipping malformed TLE near line index {i}")
            malformed_count += 1

    if malformed_count > 0:
        print(f"Skipped {malformed_count} TLE entries.")
    if line_count % 3 != 0:
        print(f"TLE file line count ({line_count}) not multiple of 3.")
    print(f"Successfully parsed {processed_count} TLEs.")
    if not first_epoch_str:
        print("Could not determine epoch from TLE file.")

    return tles, first_epoch_str


def run_full_constellation_simulation():
    """
    Sets up and runs the dynamic state simulation for the full Starlink constellation
    and multiple ground stations over one orbital period.
    WARNING: Computationally intensive. Uses NO predefined ISLs.
    """
    print("--- Setting up Full Starlink Constellation Simulation (NO ISLs) ---")

    # --- Load TLEs ---
    parsed_tles, tle_epoch_str = parse_tles_from_file(TLE_FILE)
    if not parsed_tles:
        print("Halting simulation.")
        return
    if not tle_epoch_str:
        print("Cannot determine TLE epoch from file. Halting.")
        # Or set a manual epoch string here if needed, e.g.
        # tle_epoch_str = "25112.58592294" # MUST MATCH THE TLE FILE DATA
        return

    # --- Simulation Parameters ---
    # Convert TLE epoch string to Astropy Time
    try:
        year_short = int(tle_epoch_str[:2])
        day_of_year_frac = float(tle_epoch_str[2:])
        year_full = 2000 + year_short if year_short < 57 else 1900 + year_short
        epoch = Time(year_full, format="jyear") + (day_of_year_frac - 1) * astro_units.day
        print(f"Using TLE Epoch: {tle_epoch_str}")
        print(f"Converted Astropy Epoch (UTC): {epoch.utc.iso}")
    except Exception as e:
        print(f"Error converting TLE epoch string '{tle_epoch_str}' to Time object: {e}")
        return

    duration_s = 5800  # ~96.6 minutes
    time_step_s = 300  # 5 minute step to reduce computation
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
    max_isl_length_m = 20000 * 1000  # Large value, irrelevant as no ISLs defined
    print(f"Max GSL Length: {max_gsl_length_m / 1000:.1f} km")

    # --- Scenario Definition ---
    # Satellites - Create objects from parsed TLEs using NORAD ID
    satellites = []
    sat_ids = []
    for tle_tuple, sat_id in parsed_tles:
        try:
            ephem_obj = ephem.readtle(tle_tuple[0], tle_tuple[1], tle_tuple[2])
            satellites.append(
                Satellite(id=sat_id, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
            )
            sat_ids.append(sat_id)
        except ValueError as e:
            print(f"Skipping TLE ID {sat_id} ({tle_tuple[0]}) due to read error: {e}")
            continue
    print(f"Created {len(satellites)} satellite objects.")
    if not satellites:
        print("No valid satellite objects created. Halting.")
        return

    # Ground Stations (Keep diverse set, assign high IDs)
    gs_start_id = 90000  # Ensure GS IDs are unique from NORAD IDs
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
            print(f"Could not create ground station {data['name']}: {e}")
            return

    all_node_ids = sat_ids + gs_ids
    print(f"Created {len(ground_stations)} ground station objects.")

    # ConstellationData
    constellation_data = ConstellationData(
        orbits=-1,
        sats_per_orbit=-1,
        epoch=tle_epoch_str,
        max_gsl_length_m=max_gsl_length_m,
        max_isl_length_m=max_isl_length_m,
        satellites=satellites,
    )
    constellation_data.number_of_satellites = len(satellites)  # Set actual count

    # **** NO PREDEFINED ISLS ****
    undirected_isls = []
    print("Running simulation WITHOUT predefined ISLs.")

    # GSL Interface Info (All satellites + all GS)
    list_gsl_interfaces_info = [
        {"id": node_id, "number_of_interfaces": 1, "aggregate_max_bandwidth": 10.0}
        for node_id in all_node_ids
    ]

    # --- Run Simulation ---
    print(f"Running simulation for {duration_s}s with {time_step_s}s step...")
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
    print("Simulation run finished.")

    # --- Process Results ---
    if all_states and any(s is not None for s in all_states):
        valid_states = [s for s in all_states if s is not None]
        num_steps = len(all_states)
        print(f"Generated {len(valid_states)} valid state entries out of {num_steps} total steps.")

        # Example Analysis: Find connected GS pairs at specific times
        print("\n--- Connectivity Check (Any GS <-> Any other GS) ---")
        print("Time (s) | Connected Pairs (Showing first hop)")
        print("---------|----------------------------------------")
        for i, state in enumerate(all_states):
            time_s = (offset_ns + i * time_step_ns) / 1e9
            connected_pairs_str = ""
            if state and isinstance(state.get("fstate"), dict):
                fstate = state["fstate"]
                found_connection = False
                for src_gs_id in gs_ids:
                    for dst_gs_id in gs_ids:
                        if src_gs_id == dst_gs_id:
                            continue
                        hop_tuple = fstate.get((src_gs_id, dst_gs_id), (-1, -1, -1))
                        if hop_tuple[0] != -1:
                            connected_pairs_str += (
                                f" ({src_gs_id}->{dst_gs_id}: Sat {hop_tuple[0]})"
                            )
                            found_connection = True
                if not found_connection:
                    connected_pairs_str = " None"
            elif state is None:
                connected_pairs_str = " ERROR in state calc"
            else:
                connected_pairs_str = " Invalid State Format"

            # Print only first, last, and every 10th step to avoid too much output
            if i == 0 or (i + 1) % 10 == 0 or i == num_steps - 1:
                print(f"{time_s:<8.1f} |{connected_pairs_str}")

        # Save the full results
        results_filename = "starlink_full_constellation_results.pkl"
        try:
            with open(results_filename, "wb") as f:
                pickle.dump(all_states, f)
            print(f"Full simulation state list saved to: {results_filename}")
            print(f"\nResult states saved to {results_filename}")
        except Exception as e:
            print(f"Failed to save results to {results_filename}: {e}")
    else:
        print("Simulation produced no valid state results.")


if __name__ == "__main__":
    print("Starting FULL Starlink simulation example (NO ISLs)...")
    run_full_constellation_simulation()
    print("Example script finished.")
