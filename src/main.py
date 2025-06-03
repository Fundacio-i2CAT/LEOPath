import math
import yaml
import argparse
import os
from src import logger
from src.distance_tools import geodetic2cartesian
from src.network_state.generate_dynamic_state import generate_dynamic_state
from src.tles.generate_tles_from_scratch import generate_tles_from_scratch_with_sgp
from src.tles.read_tles import read_tles
from src.topology.satellite.satellite import Satellite
from src.topology.topology import ConstellationData, GroundStation


def load_config(config_path):
    """Loads the simulation configuration from a YAML file."""
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def setup_logging(config):
    """Initializes the logger based on configuration."""
    global log
    log_config = config["logging"]
    logger.setup_logger(is_debug=log_config["is_debug"], file_name=log_config["file_name"])
    log = logger.get_logger(__name__)
    log.info("Logger initialized.")


def setup_tles_and_satellites(config):
    """Generates TLEs, reads them, and creates Satellite objects."""
    tle_config = config["constellation"]
    log.info(f"Generating TLEs for {tle_config['name']}...")
    generate_tles_from_scratch_with_sgp(
        tle_config["tle_output_filename"],
        tle_config["name"],
        tle_config["num_orbits"],
        tle_config["num_sats_per_orbit"],
        tle_config["phase_diff"],
        tle_config["inclination_degree"],
        tle_config["eccentricity"],
        tle_config["arg_of_perigee_degree"],
        tle_config["mean_motion_rev_per_day"],
    )
    parsed_tles_data = read_tles(tle_config["tle_output_filename"])
    sim_satellites = [
        Satellite(id=i, ephem_obj_manual=ephem_obj, ephem_obj_direct=ephem_obj)
        for i, ephem_obj in enumerate(parsed_tles_data["satellites"])
    ]
    log.info(f"Created {len(sim_satellites)} Satellite objects.")
    return parsed_tles_data, sim_satellites


def setup_ground_stations(config):
    """Creates GroundStation objects from configuration."""
    gs_config = config["ground_stations"]
    const_config = config["constellation"]
    gs_start_id = const_config["num_orbits"] * const_config["num_sats_per_orbit"]

    ground_stations = []
    for i, gs_data in enumerate(gs_config):
        lat, lon, elv = gs_data["latitude"], gs_data["longitude"], gs_data["elevation_m"]
        x, y, z = geodetic2cartesian(lat, lon, elv)
        ground_stations.append(
            GroundStation(
                gid=gs_start_id + i,
                name=gs_data["name"],
                latitude_degrees_str=str(lat),
                longitude_degrees_str=str(lon),
                elevation_m_float=elv,
                cartesian_x=x,
                cartesian_y=y,
                cartesian_z=z,
            )
        )
    log.info(f"Created {len(ground_stations)} GroundStation objects.")
    return ground_stations


def setup_isls_in_a_whole_orbit(num_sats: int):
    undirected_isls = [(i, i + 1) for i in range(num_sats - 1)] if num_sats > 1 else []
    if num_sats > 1:
        undirected_isls.append((num_sats - 1, 0))
    return undirected_isls


def calculate_link_params(config):
    """Calculates maximum link lengths based on configuration."""
    sat_config = config["satellite"]
    earth_config = config["earth"]

    altitude_m = sat_config["altitude_m"]
    earth_radius = earth_config["radius_m"]
    cone_angle = sat_config["cone_angle_degrees"]
    min_isl_alt = earth_config["isl_min_altitude_m"]

    satellite_cone_radius_m = altitude_m / math.tan(math.radians(cone_angle))
    max_gsl = math.sqrt(math.pow(satellite_cone_radius_m, 2) + math.pow(altitude_m, 2))
    max_isl = 2 * math.sqrt(
        math.pow(earth_radius + altitude_m, 2) - math.pow(earth_radius + min_isl_alt, 2)
    )
    log.info(f"Max GSL: {max_gsl:.2f} m, Max ISL: {max_isl:.2f} m")
    return max_gsl, max_isl


def execute_simulation_run(config, parsed_tles_data, sim_satellites, ground_stations):
    """Runs the core dynamic state generation."""
    sim_config = config["simulation"]
    net_config = config["network"]

    max_gsl, max_isl = calculate_link_params(config)

    constellation_data = ConstellationData(
        orbits=parsed_tles_data["n_orbits"],
        sats_per_orbit=parsed_tles_data["n_sats_per_orbit"],
        epoch=parsed_tles_data["epoch"],
        max_gsl_length_m=max_gsl,
        max_isl_length_m=max_isl,
        satellites=sim_satellites,
    )

    num_sats = len(sim_satellites)
    undirected_isls = setup_isls_in_a_whole_orbit(num_sats)

    gsl_node_ids = list(range(num_sats)) + [gs.id for gs in ground_stations]
    list_gsl_interfaces_info = [
        {
            "id": node_id,
            "number_of_interfaces": net_config["gsl_interfaces"]["number_of_interfaces"],
            "aggregate_max_bandwidth": net_config["gsl_interfaces"]["aggregate_max_bandwidth"],
        }
        for node_id in gsl_node_ids
    ]

    simulation_end_time_ns = int(sim_config["end_time_hours"] * 60 * 60 * 1e9)
    time_step_ns = int(sim_config["time_step_minutes"] * 60 * 1e9)

    log.info("Starting dynamic state generation...")
    all_states = generate_dynamic_state(
        output_dynamic_state_dir=sim_config["output_dir"],
        epoch=parsed_tles_data["epoch"],
        simulation_end_time_ns=simulation_end_time_ns,
        time_step_ns=time_step_ns,
        offset_ns=sim_config["offset_ns"],
        constellation_data=constellation_data,
        ground_stations=ground_stations,
        undirected_isls=undirected_isls,
        list_gsl_interfaces_info=list_gsl_interfaces_info,
        dynamic_state_algorithm=sim_config["dynamic_state_algorithm"],
    )
    log.info(f"Generated {len(all_states)} dynamic states.")
    for idx, state in enumerate(all_states):
        log.info(f"Generated fstate at step {idx}: {state['fstate']}")
    log.info("Simulation finished. ✅")
    return all_states


def main():
    """Main function to parse arguments and orchestrate the simulation."""
    parser = argparse.ArgumentParser(description="Run the ETHER satellite network simulator.")
    parser.add_argument(
        "-c",
        "--config",
        default="config.yaml",
        help="Path to the simulation configuration file (default: config.yaml)",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config)
    parsed_tles_data, sim_satellites = setup_tles_and_satellites(config)
    ground_stations = setup_ground_stations(config)
    execute_simulation_run(config, parsed_tles_data, sim_satellites, ground_stations)


if __name__ == "__main__":
    main()
