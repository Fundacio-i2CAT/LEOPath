import math
import os
import yaml
import argparse
import time
from src import logger

log = logger.get_logger(__name__)

try:
    from . import util
except (ImportError, SystemError):
    try:
        import util
    except ImportError:
        log.critical(
            "CRITICAL: Could not import the 'util' module. "
            "Ensure it's in the correct path (e.g., src/satellite_visualisation/util.py) "
            "and the script is run appropriately (e.g., as a module from the project root)."
        )
        print(
            "CRITICAL: Could not import the 'util' module. Please check your project structure and PYTHONPATH."
        )
        exit(1)

SCRIPT_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOP_HTML_FILE = os.path.join(SCRIPT_BASE_DIR, "static_html/top.html")
BOTTOM_HTML_FILE = os.path.join(SCRIPT_BASE_DIR, "static_html/bottom.html")
DEFAULT_OUT_DIR_NAME = "visualisation_output"


def generate_satellite_trajectories_from_config(config_data):
    """
    Generates CesiumJS visualization strings for satellites, orbits, and ground stations
    based on parameters from the configuration data, using direct string concatenation.
    Satellites will not have labels by default from previous modification.
    Ground stations will also not have labels as per the new request.
    """
    viz_string = ""

    # General constellation parameters from config
    epoch = config_data.get("epoch", "2000-01-01 00:00:00")
    eccentricity = config_data.get("eccentricity", 0.0000001)
    arg_of_perigee_degree = config_data.get("arg_of_perigee_degree", 0.0)
    phase_diff = config_data.get("phase_diff", True)

    # Process Satellite Shells
    viz_string += "// --- Satellite Shell Entities ---\n"
    for shell_idx, shell_config in enumerate(config_data.get("shells", [])):
        shell_name = shell_config.get("name", f"Shell_{shell_idx+1}")
        log.info(f"Processing shell: {shell_name}")

        num_orbs = shell_config["num_orbs"]
        num_sats_per_orb = shell_config["num_sats_per_orb"]
        inclination_degree = shell_config["inclination_degree"]
        mean_motion_rev_per_day = shell_config["mean_motion_rev_per_day"]
        altitude_m = shell_config["altitude_m"]
        shell_color = shell_config.get("color", "WHITE").upper()

        sat_objs = util.generate_sat_obj_list(
            num_orbs,
            num_sats_per_orb,
            epoch,
            phase_diff,
            inclination_degree,
            eccentricity,
            arg_of_perigee_degree,
            mean_motion_rev_per_day,
            altitude_m,
        )

        for sat_idx, sat_data in enumerate(sat_objs):
            sat_data["sat_obj"].compute(epoch)

            entity_display_name = f"{shell_name}_Sat_{sat_idx+1}"
            js_var_name = f"satEntity_{shell_idx}_{sat_idx}"

            viz_string += "var " + js_var_name + " = viewer.entities.add({\n"
            viz_string += "    name: '" + entity_display_name + "',\n"
            viz_string += (
                "    position: Cesium.Cartesian3.fromDegrees("
                + str(math.degrees(sat_data["sat_obj"].sublong))
                + ", "
                + str(math.degrees(sat_data["sat_obj"].sublat))
                + ", "
                + str(sat_data["alt_km"] * 1000)
                + "),\n"
            )
            viz_string += "    ellipsoid: {\n"
            viz_string += "        radii: new Cesium.Cartesian3(30000.0, 30000.0, 30000.0),\n"
            viz_string += "        material: Cesium.Color.BLACK.withAlpha(1)\n"
            viz_string += "    }\n"
            viz_string += "});\n"

        orbit_links = util.find_orbit_links(sat_objs, num_orbs, num_sats_per_orb)
        for link_key in orbit_links:
            sat1_data = sat_objs[orbit_links[link_key]["sat1"]]
            sat2_data = sat_objs[orbit_links[link_key]["sat2"]]

            viz_string += "viewer.entities.add({\n"
            viz_string += "    name: 'orbit_link_" + str(shell_idx) + "_" + str(link_key) + "',\n"
            viz_string += "    polyline: {\n"
            viz_string += "        positions: Cesium.Cartesian3.fromDegreesArrayHeights([\n"
            viz_string += (
                "            "
                + str(math.degrees(sat1_data["sat_obj"].sublong))
                + ", "
                + str(math.degrees(sat1_data["sat_obj"].sublat))
                + ", "
                + str(sat1_data["alt_km"] * 1000)
                + ",\n"
            )
            viz_string += (
                "            "
                + str(math.degrees(sat2_data["sat_obj"].sublong))
                + ", "
                + str(math.degrees(sat2_data["sat_obj"].sublat))
                + ", "
                + str(sat2_data["alt_km"] * 1000)
                + "\n"
            )
            viz_string += "        ]),\n"
            viz_string += "        width: 0.5,\n"
            viz_string += "        arcType: Cesium.ArcType.NONE,\n"
            viz_string += "        material: new Cesium.PolylineOutlineMaterialProperty({\n"
            viz_string += "            color: Cesium.Color." + shell_color + ".withAlpha(0.4),\n"
            viz_string += "            outlineWidth: 0,\n"
            viz_string += "            outlineColor: Cesium.Color.BLACK\n"
            viz_string += "        })\n"
            viz_string += "    }\n"
            viz_string += "});\n"

    # Process Ground Stations
    if "ground_stations" in config_data and config_data["ground_stations"]:
        viz_string += "\n// --- Ground Station Entities (No Labels) ---\n"
        log.info("Processing ground stations...")
        for gs_idx, gs_data in enumerate(config_data["ground_stations"]):
            gs_name = gs_data.get("name", f"GroundStation_{gs_idx+1}")
            # Ensure latitude and longitude are present, otherwise skip or error
            if "latitude" not in gs_data or "longitude" not in gs_data:
                log.warning(
                    f"Skipping ground station '{gs_name}' due to missing latitude/longitude."
                )
                continue

            gs_lat = float(gs_data["latitude"])
            gs_lon = float(gs_data["longitude"])
            gs_alt_m = float(
                gs_data.get("altitude_m", 100.0)
            )  # Default altitude slightly above surface
            gs_color_str = gs_data.get("color", "BLUE").upper()
            gs_pixel_size = int(gs_data.get("pixel_size", 10))
            # gs_label_text = gs_data.get("label_text", gs_name) # Label text no longer used

            js_gs_var_name = f"gsEntity_{gs_idx}"

            viz_string += "var " + js_gs_var_name + " = viewer.entities.add({\n"
            viz_string += "    name: '" + gs_name + "',\n"  # Keep name for selection/debugging
            viz_string += (
                "    position: Cesium.Cartesian3.fromDegrees("
                + str(gs_lon)
                + ", "
                + str(gs_lat)
                + ", "
                + str(gs_alt_m)
                + "),\n"
            )
            viz_string += "    point: {\n"
            viz_string += "        pixelSize: " + str(gs_pixel_size) + ",\n"
            viz_string += "        color: Cesium.Color." + gs_color_str + ",\n"
            viz_string += "        outlineColor: Cesium.Color.BLACK,\n"
            viz_string += "        outlineWidth: 1\n"
            viz_string += (
                "    }\n"  # point is now the last property, no comma after its closing brace
            )
            # The 'label' property block for ground stations has been removed.
            viz_string += "});\n"
        log.info(f"Processed {len(config_data['ground_stations'])} ground station(s).")

    return viz_string


def write_html_file(viz_string_content, output_dir, html_file_name_base):
    os.makedirs(output_dir, exist_ok=True)
    output_html_file = os.path.join(output_dir, f"{html_file_name_base.replace(' ', '_')}.html")
    log.info(f"Attempting to write HTML file to: {output_html_file}")
    try:
        with open(output_html_file, "w", encoding="utf-8") as writer_html:
            if os.path.exists(TOP_HTML_FILE):
                with open(TOP_HTML_FILE, "r", encoding="utf-8") as fi:
                    writer_html.write(fi.read())
            else:
                log.warning(f"Top HTML file not found: {TOP_HTML_FILE}")
                writer_html.write("\n")

            writer_html.write(viz_string_content)
            writer_html.flush()
            if hasattr(writer_html, "fileno"):
                try:
                    os.fsync(writer_html.fileno())
                except OSError as e:
                    log.warning(f"Could not fsync file {output_html_file}: {e}")

            if os.path.exists(BOTTOM_HTML_FILE):
                with open(BOTTOM_HTML_FILE, "r", encoding="utf-8") as fb:
                    writer_html.write(fb.read())
            else:
                log.warning(f"Bottom HTML file not found: {BOTTOM_HTML_FILE}")
                writer_html.write("\n")

        log.info(f"Successfully wrote visualization to: {output_html_file}")
        print(f"ACTION: HTML file generated at: {output_html_file}")
        print(
            "Please open this file via a local web server (e.g., 'python -m http.server' in project root)."
        )

    except IOError as e_io:
        log.error(f"IOError writing HTML file {output_html_file}: {e_io}")
        print(f"IOError writing HTML file {output_html_file}: {e_io}")
    except Exception as e_gen:
        log.error(f"A general error occurred in write_html_file: {e_gen}")
        print(f"A general error occurred in write_html_file: {e_gen}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate CesiumJS visualization for satellite constellations and ground stations from a YAML configuration."
    )
    parser.add_argument(
        "config_file", type=str, help="Path to the YAML configuration file for the constellation."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(SCRIPT_BASE_DIR, DEFAULT_OUT_DIR_NAME),
        help=(
            f"Directory to save the output HTML file. "
            f"Default: ./{DEFAULT_OUT_DIR_NAME} (relative to script location)"
        ),
    )
    args = parser.parse_args()

    abs_output_dir = os.path.abspath(args.output_dir)
    log.info(f"Output directory set to: {abs_output_dir}")

    try:
        with open(args.config_file, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        log.info(f"Successfully loaded configuration from: {args.config_file}")
    except FileNotFoundError:
        log.error(f"Configuration file not found: {args.config_file}")
        print(f"ERROR: Configuration file not found: {args.config_file}")
        return
    except yaml.YAMLError as e_yaml:
        log.error(f"Error decoding YAML from configuration file: {args.config_file}\n{e_yaml}")
        print(f"ERROR: Invalid YAML in {args.config_file}: {e_yaml}")
        return
    except Exception as e_conf:
        log.error(f"An unexpected error occurred while loading configuration: {e_conf}")
        print(f"ERROR: Could not load configuration: {e_conf}")
        return

    constellation_name_from_config = config_data.get("constellation_name", "UnnamedConstellation")
    log.info(f"Generating visualization for constellation: {constellation_name_from_config}")

    viz_string_generated = generate_satellite_trajectories_from_config(config_data)

    if viz_string_generated:
        # Calculate total number of satellites for logging purposes
        num_sats_total = 0
        if "shells" in config_data and config_data["shells"]:
            for shell_conf in config_data["shells"]:
                num_sats_total += shell_conf.get("num_orbs", 0) * shell_conf.get(
                    "num_sats_per_orb", 0
                )

        log_message = f"Generated visualization string for {constellation_name_from_config}"
        if num_sats_total > 0:
            log_message += f" with {num_sats_total} satellites"
        if "ground_stations" in config_data and config_data["ground_stations"]:
            num_gs = len(config_data["ground_stations"])
            if num_gs > 0:
                log_message += f" and {num_gs} ground station(s)"
        log.info(log_message + ".")

        write_html_file(viz_string_generated, abs_output_dir, constellation_name_from_config)
    else:
        log.warning("No visualization string generated. Check configuration and logs.")
        print("WARNING: No visualization string was generated.")


if __name__ == "__main__":
    main()
