import argparse
import os

import ephem

from .helpers import load_yaml_config, log
from .html_builder import write_html_file
from .js_generator import generate_ground_stations_js, generate_shells_js

SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VISUALIZATION_BASE_DIR = os.path.dirname(SCRIPT_BASE_DIR)  # Go up one level to satellite_visualisation
TOP_HTML_FILE = os.path.join(VISUALIZATION_BASE_DIR, "static_html/top.html")
BOTTOM_HTML_FILE = os.path.join(VISUALIZATION_BASE_DIR, "static_html/bottom.html")
DEFAULT_OUT_DIR_NAME = "visualisation_output"


def generate_visualization_js(config_data, config_file_path_abs):
    viz_string = ""
    global_epoch_str = config_data.get("epoch", "2000-01-01 00:00:00")
    global_ephem_epoch_for_elements = ephem.Date(global_epoch_str)
    element_shell_eccentricity = config_data.get("eccentricity", 0.0000001)
    element_shell_arg_of_perigee = config_data.get("arg_of_perigee_degree", 0.0)
    element_shell_phase_diff = config_data.get("phase_diff", True)

    if "shells" in config_data and config_data["shells"]:
        viz_string += generate_shells_js(
            config_data["shells"],
            global_epoch_str,
            global_ephem_epoch_for_elements,
            element_shell_phase_diff,
            element_shell_eccentricity,
            element_shell_arg_of_perigee,
        )
    # Add TLE and ground station JS generation here, using similar helpers
    if "ground_stations" in config_data and config_data["ground_stations"]:
        viz_string += generate_ground_stations_js(config_data["ground_stations"])
    return viz_string


def main():
    parser = argparse.ArgumentParser(
        description="Generate CesiumJS visualization for satellite constellations, TLE-defined orbits, and ground stations from a YAML configuration."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the YAML configuration file for the constellation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(SCRIPT_BASE_DIR, DEFAULT_OUT_DIR_NAME),
        help=f"Directory to save the output HTML file. Default: ./{DEFAULT_OUT_DIR_NAME} (relative to script location)",
    )
    args = parser.parse_args()
    abs_output_dir = os.path.abspath(args.output_dir)
    abs_config_file_path = os.path.abspath(args.config_file)
    config_data = load_yaml_config(abs_config_file_path)
    if not config_data:
        return
    constellation_name_from_config = config_data.get("constellation_name", "UnnamedConstellation")
    viz_string_generated = generate_visualization_js(config_data, abs_config_file_path)
    if viz_string_generated:
        write_html_file(
            viz_string_generated,
            abs_output_dir,
            constellation_name_from_config,
            TOP_HTML_FILE,
            BOTTOM_HTML_FILE,
        )
    else:
        log.warning("No visualization string generated. Check configuration and logs.")
        print("WARNING: No visualization string was generated.")


if __name__ == "__main__":
    main()
