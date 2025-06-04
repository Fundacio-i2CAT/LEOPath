import argparse
import math
import os
import time

import ephem  # For TLE parsing and propagation
import yaml

from src import logger

log = logger.get_logger(__name__)

try:
    from .cesium_builder import util
except (ImportError, SystemError):
    try:
        import src.satellite_visualisation.cesium_builder.util as util
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


def generate_visualization_js(config_data, config_file_path_abs):
    """
    Generates CesiumJS visualization strings for element-defined shells, TLE-defined orbits,
    and ground stations based on parameters from the configuration data.
    For TLEs, plots current positions and links them if structure is known from TLE file header.
    """
    viz_string = ""

    global_epoch_str = config_data.get("epoch", "2000-01-01 00:00:00")
    global_ephem_epoch_for_elements = ephem.Date(global_epoch_str)
    log.info(f"Using global epoch for element-defined shells: {global_epoch_str}")

    element_shell_eccentricity = config_data.get("eccentricity", 0.0000001)
    element_shell_arg_of_perigee = config_data.get("arg_of_perigee_degree", 0.0)
    element_shell_phase_diff = config_data.get("phase_diff", True)

    # --- 1. Process Satellite Shells defined by orbital elements ---
    if "shells" in config_data and config_data["shells"]:
        viz_string += "// --- Satellite Shell Entities (from Orbital Elements) ---\n"
        for shell_idx, shell_config in enumerate(config_data.get("shells", [])):
            shell_name = shell_config.get("name", f"ElementsShell_{shell_idx+1}")
            log.info(f"Processing element-defined shell: {shell_name}")

            num_orbs = shell_config["num_orbs"]
            num_sats_per_orb = shell_config["num_sats_per_orb"]
            inclination_degree = shell_config["inclination_degree"]
            mean_motion_rev_per_day = shell_config["mean_motion_rev_per_day"]
            altitude_m = shell_config["altitude_m"]
            shell_links_color = shell_config.get("color", "SLATEGRAY").upper()

            sat_objs_from_elements = util.generate_sat_obj_list(
                num_orbs,
                num_sats_per_orb,
                global_epoch_str,
                element_shell_phase_diff,
                inclination_degree,
                element_shell_eccentricity,
                element_shell_arg_of_perigee,
                mean_motion_rev_per_day,
                altitude_m,
            )

            for sat_idx, sat_data in enumerate(sat_objs_from_elements):
                sat_data["sat_obj"].compute(global_ephem_epoch_for_elements)
                entity_disp_name = f"{shell_name}_Sat_{sat_idx+1}"
                js_var = f"elementSat_{shell_idx}_{sat_idx}"
                viz_string += f"var {js_var} = viewer.entities.add({{\n"
                viz_string += f"    name: '{entity_disp_name}',\n"
                viz_string += f"    position: Cesium.Cartesian3.fromDegrees({math.degrees(sat_data['sat_obj'].sublong)}, {math.degrees(sat_data['sat_obj'].sublat)}, {sat_data['alt_km'] * 1000}),\n"
                viz_string += "    ellipsoid: {\n"
                viz_string += "        radii: new Cesium.Cartesian3(20000.0, 20000.0, 20000.0),\n"
                viz_string += "        material: Cesium.Color.DARKSLATEGRAY.withAlpha(0.7)\n"
                viz_string += "    }\n});\n"

            orbit_links = util.find_orbit_links(sat_objs_from_elements, num_orbs, num_sats_per_orb)
            for link_key, link_data in orbit_links.items():
                sat1_data = sat_objs_from_elements[link_data["sat1"]]
                sat2_data = sat_objs_from_elements[link_data["sat2"]]
                viz_string += f"viewer.entities.add({{\n"
                viz_string += f"    name: 'elementLink_{shell_idx}_{link_key}',\n"
                viz_string += "    polyline: {\n"
                viz_string += (
                    "        positions: Cesium.Cartesian3.fromDegreesArrayHeights(["
                    + f"{math.degrees(sat1_data['sat_obj'].sublong)}, {math.degrees(sat1_data['sat_obj'].sublat)}, {sat1_data['alt_km'] * 1000}, "
                    + f"{math.degrees(sat2_data['sat_obj'].sublong)}, {math.degrees(sat2_data['sat_obj'].sublat)}, {sat2_data['alt_km'] * 1000}]),\n"
                )
                viz_string += "        width: 0.5,\n        arcType: Cesium.ArcType.NONE,\n"
                viz_string += (
                    "        material: new Cesium.PolylineOutlineMaterialProperty({ "
                    + f"color: Cesium.Color.{shell_links_color}.withAlpha(0.4), outlineWidth: 0, outlineColor: Cesium.Color.BLACK }})\n"
                )
                viz_string += "    }\n});\n"
        if config_data.get("shells"):
            log.info(f"Finished processing {len(config_data['shells'])} element-defined shell(s).")

    # --- 2. Process satellites from TLE files (plot current positions and link them if structure is known) ---
    if "tle_files" in config_data and config_data["tle_files"]:
        viz_string += "\n// --- Satellite Entities and Links (from TLEs) ---\n"
        config_file_dir = os.path.dirname(config_file_path_abs)

        for tle_group_idx, tle_group_config in enumerate(config_data["tle_files"]):
            relative_tle_path = tle_group_config.get("path")
            if not relative_tle_path:
                log.warning(f"Skipping TLE group {tle_group_idx} due to missing 'path'.")
                continue

            tle_file_abs_path = os.path.normpath(os.path.join(config_file_dir, relative_tle_path))
            if not os.path.exists(tle_file_abs_path):
                log.warning(
                    f"TLE file not found: {tle_file_abs_path}. Skipping TLE group '{tle_group_config.get('name_prefix', tle_group_idx)}'."
                )
                continue

            group_name_prefix = tle_group_config.get("name_prefix", f"TLEGroup{tle_group_idx}")
            links_color_tle = tle_group_config.get(
                "orbit_color", "GOLD"
            ).upper()  # Will be used for links
            sat_marker_color = tle_group_config.get("satellite_marker_color", "RED").upper()
            sat_marker_size = int(tle_group_config.get("satellite_marker_size", 6))

            log.info(f"Processing TLE file: {tle_file_abs_path} for group '{group_name_prefix}'")

            try:
                with open(tle_file_abs_path, "r", encoding="utf-8") as f_tle:
                    all_lines_in_file = [line.strip() for line in f_tle if line.strip()]
            except Exception as e:
                log.error(f"Could not read TLE file {tle_file_abs_path}: {e}")
                continue

            tle_lines_to_parse = []
            num_orbits_in_tle_file = 0
            num_sats_per_orbit_in_tle_file = 0
            header_found = False

            if all_lines_in_file:
                parts = all_lines_in_file[0].split()
                if len(parts) == 2 and all(part.isdigit() for part in parts):
                    num_orbits_in_tle_file = int(parts[0])
                    num_sats_per_orbit_in_tle_file = int(parts[1])
                    log.info(
                        f"Detected TLE header: {num_orbits_in_tle_file} orbits, {num_sats_per_orbit_in_tle_file} sats/orbit."
                    )
                    tle_lines_to_parse = all_lines_in_file[1:]
                    header_found = True
                else:
                    tle_lines_to_parse = all_lines_in_file
                    log.info(
                        "No header detected in TLE file, will process as a flat list of TLEs without structured linking."
                    )

            if not tle_lines_to_parse:
                log.warning(
                    f"No TLE data found in {tle_file_abs_path} after potentially skipping header."
                )
                continue

            # Store parsed satellite objects and their computed positions
            tle_sat_objects_for_linking = []

            parsed_tle_sats_count = 0
            current_orbit_in_tle = 0
            current_sat_in_orbit_tle = 0
            idx = 0

            while idx < len(tle_lines_to_parse):
                tle_name_line, line1, line2 = "", "", ""
                if idx + 2 < len(tle_lines_to_parse):
                    if tle_lines_to_parse[idx + 1].startswith("1") and tle_lines_to_parse[
                        idx + 2
                    ].startswith("2"):
                        tle_name_line = tle_lines_to_parse[idx]
                        line1 = tle_lines_to_parse[idx + 1]
                        line2 = tle_lines_to_parse[idx + 2]
                        idx += 3
                    else:
                        log.warning(
                            f"Unrecognized TLE block format at name line '{tle_lines_to_parse[idx]}' in {tle_file_abs_path}. Skipping."
                        )
                        idx += 1
                        continue
                else:
                    if idx < len(tle_lines_to_parse):
                        log.warning(
                            f"Not enough lines remaining for a full TLE entry at end of {tle_file_abs_path}."
                        )
                    break

                try:
                    sat_ephem_obj = ephem.readtle(tle_name_line, line1, line2)
                except Exception as e:
                    log.warning(f"Failed to parse TLE for '{tle_name_line}'. Error: {e}")
                    continue

                current_sat_epoch = sat_ephem_obj._epoch
                try:
                    sat_ephem_obj.compute(current_sat_epoch)
                except ValueError as ve:
                    log.error(
                        f"ValueError computing TLE satellite '{tle_name_line}' at its epoch {current_sat_epoch}: {ve}"
                    )
                    continue

                entity_js_var = f"tleSatMarker_{tle_group_idx}_{parsed_tle_sats_count}"
                entity_name_tle = f"{group_name_prefix}_{tle_name_line.replace(' ', '_')}"

                viz_string += f"var {entity_js_var} = viewer.entities.add({{\n"
                viz_string += f"    name: '{entity_name_tle} (at TLE epoch)',\n"
                viz_string += f"    position: Cesium.Cartesian3.fromDegrees({math.degrees(sat_ephem_obj.sublong)}, {math.degrees(sat_ephem_obj.sublat)}, {sat_ephem_obj.elevation}),\n"
                viz_string += "    point: {\n"
                viz_string += f"        pixelSize: {sat_marker_size},\n"
                viz_string += f"        color: Cesium.Color.{sat_marker_color}\n"
                viz_string += "    }\n});\n"

                # Store for linking if structure is known
                if header_found:
                    tle_sat_objects_for_linking.append(
                        {
                            "sat_obj": sat_ephem_obj,  # Already computed at its epoch
                            "alt_km": sat_ephem_obj.elevation
                            / 1000.0,  # Store for consistency if util.find_orbit_links is adapted
                            "orb_id": current_orbit_in_tle,
                            "orb_sat_id": current_sat_in_orbit_tle,
                        }
                    )
                    current_sat_in_orbit_tle += 1
                    if current_sat_in_orbit_tle >= num_sats_per_orbit_in_tle_file:
                        current_sat_in_orbit_tle = 0
                        current_orbit_in_tle += 1

                parsed_tle_sats_count += 1

            # Now add links for TLE satellites if structure was determined
            if header_found and tle_sat_objects_for_linking:
                log.info(
                    f"Attempting to generate links for TLE group '{group_name_prefix}' with {num_orbits_in_tle_file} orbits and {num_sats_per_orbit_in_tle_file} sats/orbit."
                )
                # We need to call a linking function. util.find_orbit_links might work if data structure is identical.
                # For simplicity here, let's replicate the core linking logic for intra-plane links.
                for orb_plane_idx in range(num_orbits_in_tle_file):
                    sats_in_this_plane = [
                        s for s in tle_sat_objects_for_linking if s["orb_id"] == orb_plane_idx
                    ]
                    sats_in_this_plane.sort(
                        key=lambda s: s["orb_sat_id"]
                    )  # Ensure they are sorted by their ID within the orbit

                    for i in range(len(sats_in_this_plane)):
                        sat1_data_tle = sats_in_this_plane[i]
                        # Link to the next satellite in the same plane, wrapping around
                        sat2_data_tle = sats_in_this_plane[(i + 1) % len(sats_in_this_plane)]

                        # sat_obj is already computed at its TLE epoch
                        viz_string += f"viewer.entities.add({{\n"
                        viz_string += f"    name: 'tleLink_{group_name_prefix}_orb{orb_plane_idx}_sat{i}-{(i+1)%len(sats_in_this_plane)}',\n"
                        viz_string += "    polyline: {\n"
                        viz_string += (
                            "        positions: Cesium.Cartesian3.fromDegreesArrayHeights(["
                            + f"{math.degrees(sat1_data_tle['sat_obj'].sublong)}, {math.degrees(sat1_data_tle['sat_obj'].sublat)}, {sat1_data_tle['alt_km'] * 1000}, "
                            + f"{math.degrees(sat2_data_tle['sat_obj'].sublong)}, {math.degrees(sat2_data_tle['sat_obj'].sublat)}, {sat2_data_tle['alt_km'] * 1000}]),\n"
                        )
                        viz_string += "        width: 0.8,\n        arcType: Cesium.ArcType.NONE,\n"  # Slightly thicker for TLE links
                        viz_string += (
                            "        material: new Cesium.PolylineOutlineMaterialProperty({ "
                            + f"color: Cesium.Color.{links_color_tle}.withAlpha(0.5), outlineWidth: 0, outlineColor: Cesium.Color.BLACK }})\n"
                        )
                        viz_string += "    }\n});\n"

            log.info(f"Processed {parsed_tle_sats_count} TLE entries from {tle_file_abs_path}.")
        if config_data.get("tle_files"):
            log.info(f"Finished processing {len(config_data['tle_files'])} TLE file group(s).")

    # --- 3. Process Ground Stations ---
    if "ground_stations" in config_data and config_data["ground_stations"]:
        viz_string += "\n// --- Ground Station Entities (No Labels) ---\n"
        log.info("Processing ground stations...")
        for gs_idx, gs_data in enumerate(config_data.get("ground_stations", [])):
            gs_name = gs_data.get("name", f"GroundStation_{gs_idx+1}")
            if "latitude" not in gs_data or "longitude" not in gs_data:
                log.warning(
                    f"Skipping ground station '{gs_name}' due to missing latitude/longitude."
                )
                continue

            gs_lat = float(gs_data["latitude"])
            gs_lon = float(gs_data["longitude"])
            gs_alt_m = float(gs_data.get("altitude_m", 100.0))
            gs_color_str = gs_data.get("color", "BLUE").upper()
            gs_pixel_size = int(gs_data.get("pixel_size", 10))

            js_gs_var = f"gsEntity_{gs_idx}"
            viz_string += f"var {js_gs_var} = viewer.entities.add({{\n"
            viz_string += f"    name: '{gs_name}',\n"
            viz_string += (
                f"    position: Cesium.Cartesian3.fromDegrees({gs_lon}, {gs_lat}, {gs_alt_m}),\n"
            )
            viz_string += "    point: {\n"
            viz_string += f"        pixelSize: {gs_pixel_size},\n"
            viz_string += f"        color: Cesium.Color.{gs_color_str},\n"
            viz_string += "        outlineColor: Cesium.Color.BLACK,\n"
            viz_string += "        outlineWidth: 1\n"
            viz_string += "    }\n});\n"
        if config_data.get("ground_stations"):
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
                writer_html.write(f"\n")

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
                writer_html.write(f"\n")

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
        description="Generate CesiumJS visualization for satellite constellations, TLE-defined orbits, and ground stations from a YAML configuration."
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
    abs_config_file_path = os.path.abspath(args.config_file)

    try:
        with open(abs_config_file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        log.info(f"Successfully loaded configuration from: {abs_config_file_path}")
    except FileNotFoundError:
        log.error(f"Configuration file not found: {abs_config_file_path}")
        print(f"ERROR: Configuration file not found: {abs_config_file_path}")
        return
    except yaml.YAMLError as e_yaml:
        log.error(f"Error decoding YAML from configuration file: {abs_config_file_path}\n{e_yaml}")
        print(f"ERROR: Invalid YAML in {abs_config_file_path}: {e_yaml}")
        return
    except Exception as e_conf:
        log.error(f"An unexpected error occurred while loading configuration: {e_conf}")
        print(f"ERROR: Could not load configuration: {e_conf}")
        return

    constellation_name_from_config = config_data.get("constellation_name", "UnnamedConstellation")
    log.info(f"Generating visualization for constellation: {constellation_name_from_config}")

    viz_string_generated = generate_visualization_js(config_data, abs_config_file_path)

    if viz_string_generated:
        num_sats_total = 0
        if "shells" in config_data and config_data.get("shells"):
            for shell_conf in config_data["shells"]:
                num_sats_total += shell_conf.get("num_orbs", 0) * shell_conf.get(
                    "num_sats_per_orb", 0
                )

        log_message = f"Generated visualization string for {constellation_name_from_config}"
        if num_sats_total > 0:
            log_message += f" with {num_sats_total} element-defined satellites"

        num_tle_files_processed = 0
        if "tle_files" in config_data and config_data.get("tle_files"):
            num_tle_files_processed = len(config_data["tle_files"])
            if num_tle_files_processed > 0:
                log_message += f" and processing {num_tle_files_processed} TLE file definition(s)"

        if "ground_stations" in config_data and config_data.get("ground_stations"):
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
