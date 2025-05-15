# Filename: leo-routing-simu/src/post_analysis/print_graphical_routes_and_rtt_pkl.py
# Final Version: Reads PKL with EMBEDDED metadata, uses correct time key, processes confirmed fstate format.

# Standard library and third-party imports
import pickle
import os
import traceback
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from astropy import units as u

# Need Skyfield Time potentially if epoch is stored as such and used directly
from skyfield.timelib import Time as SkyfieldTime

# --- Logger Setup ---
try:
    from .. import logger
except ImportError:
    try:
        import logger
    except ImportError:
        import logging

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger = logging

        class DummyLogger:
            def get_logger(self, name):
                return logging.getLogger(name)

        logger = DummyLogger()
        print("WARNING: src.logger not found, falling back to basic logging.")
log = logger.get_logger(__name__)
# --- End Logger Setup ---


# Imports from the leo-routing-sim project structure
# We still need the tools
from .graph_tools import get_path, compute_path_length_without_graph

# We need the function to plot satellite shadows
from ..ground_stations.extend_ground_stations import (
    create_basic_ground_station_for_satellite_shadow,
)

# We no longer need the static file readers (read_tles, read_isls, etc.)
# Keep exputil ONLY if local_shell is used, otherwise remove.
import exputil  # Keep for local_shell.make_full_dir


# --- Global Constants ---
GROUND_STATION_USED_COLOR = "#3b3b3b"
GROUND_STATION_UNUSED_COLOR = "#cccccc"
SATELLITE_USED_COLOR = "#cc0000"
SATELLITE_UNUSED_COLOR = "#ffaaaa"
ISL_COLOR = "#ff8c00"
SRC_DST_GS_COLOR_USED = "#00cc00"
SRC_DST_GS_COLOR_UNUSED = "#008000"
LABEL_COLOR = "#000000"


# --- Main Analysis Function ---
def print_graphical_routes_and_rtt_from_embedded_pkl(
    base_output_dir,
    fstate_pkl_path,  # Path to the PKL file containing embedded metadata and data
    src,
    dst,
):
    """
    Calculates RTT, logs path changes, and generates plots by loading simulation
    state and ALL necessary static data directly from a single .pkl file.

    Expects PKL structure: {'metadata': {...}, 'time_step_data': [...]}
    where 'metadata' contains keys like 'tles_data', 'ground_stations', 'list_isls',
    'max_gsl_length_m', 'max_isl_length_m', 'fstate_data_key', 'fstate_data_format'.

    Processes fstate based on format specified in metadata (expects 'dict_tuple_nh_if1_if2').
    Reads time information using the 'time_since_epoch_ns' key.

    :param base_output_dir: Base directory to save generated PDF and data files.
    :param fstate_pkl_path: Path to the .pkl file containing embedded metadata and time step data.
    :param src: Source node ID (global).
    :param dst: Destination node ID (global).
    :return: True if analysis completes successfully, False otherwise.
    """
    local_shell = exputil.LocalShell()

    log.info(f"Loading embedded data and metadata from: {fstate_pkl_path}")
    if not os.path.exists(fstate_pkl_path):
        log.error(f"Input PKL file not found at {fstate_pkl_path}")
        return False

    try:
        with open(fstate_pkl_path, "rb") as f:
            loaded_data = pickle.load(f)
    except Exception as e:
        log.exception(f"Error loading or parsing PKL file {fstate_pkl_path}: {e}")
        return False

    # --- Validate PKL structure and extract data/metadata ---
    if (
        not isinstance(loaded_data, dict)
        or "metadata" not in loaded_data
        or "time_step_data" not in loaded_data
    ):
        log.error(
            f"PKL file {fstate_pkl_path} does not have the expected structure (dict with 'metadata' and 'time_step_data' keys)."
        )
        return False

    metadata = loaded_data["metadata"]
    all_time_step_data = loaded_data["time_step_data"]
    log.info(f"Successfully loaded metadata and {len(all_time_step_data)} time steps from PKL.")

    # --- Extract Static Data Directly from Metadata ---
    try:
        log.info("Extracting static network data from PKL metadata...")

        # TLEs data (containing epoch and satellites list)
        # NOTE: Assumes 'satellites' are Skyfield EarthSatellite objects,
        # and 'epoch' is a Skyfield Time object as saved by read_tles previously.
        # If they were stored differently (e.g., using ephem), adjust accordingly.
        tles_data = metadata.get("tles_data")
        if tles_data is None:  # Check if key exists from runner script that generated PKL
            # Fallback: Maybe TLE list was stored directly?
            tles_list_used = metadata.get("tles_used")  # Check for list of TLE tuples
            if tles_list_used and isinstance(tles_list_used, list):
                log.warning(
                    "Found 'tles_used' in metadata. Re-parsing TLEs. Consider storing 'tles_data' dictionary directly in PKL generation."
                )
                # Need read_tles logic here or similar to parse these back into epoch/satellites
                # This requires re-importing read_tles temporarily
                from ..tles.read_tles import read_tles as temp_read_tles

                # Need to write tles_list_used to a temporary string/file to read? Simpler to store tles_data dict.
                # For now, raise error if tles_data is missing.
                raise ValueError(
                    "'tles_data' dictionary (with epoch, satellites) missing from metadata."
                )
            else:
                raise ValueError("'tles_data' dictionary missing from metadata.")

        epoch = tles_data.get("epoch")  # Expecting Skyfield Time object
        satellites = tles_data.get(
            "satellites"
        )  # Expecting list of Skyfield EarthSatellite objects
        if epoch is None or satellites is None:
            raise ValueError("'epoch' or 'satellites' missing within 'tles_data' in metadata.")
        # Optional: Check type of epoch (should be Skyfield Time)
        # if not isinstance(epoch, SkyfieldTime): log.warning(...)

        # Ground stations (list of dictionaries)
        ground_stations = metadata.get("ground_stations")
        if ground_stations is None or not isinstance(ground_stations, list):
            raise ValueError("'ground_stations' missing or not a list in metadata.")
        # Ensure GS dicts have needed keys (e.g., latitude/longitude strings)
        if ground_stations and not (
            "latitude_degrees_str" in ground_stations[0]
            and "longitude_degrees_str" in ground_stations[0]
        ):
            log.warning(
                "Ground station entries in metadata might be missing lat/lon string keys needed for plotting/distance."
            )
            # Attempt to use 'lat'/'lon' keys if present from gs_defs? Needs care.
            # Assuming 'latitude_degrees_str' etc are present as saved by read_ground_stations_extended format.

        # ISL list (list of tuples)
        list_isls = metadata.get("list_isls")
        if list_isls is None or not isinstance(list_isls, list):
            raise ValueError("'list_isls' missing or not a list in metadata.")

        # Max link lengths (floats or ints)
        max_gsl_length_m = metadata.get("max_gsl_length_m")
        max_isl_length_m = metadata.get("max_isl_length_m")
        if max_gsl_length_m is None or not isinstance(max_gsl_length_m, (int, float)):
            raise ValueError("'max_gsl_length_m' missing or invalid type in metadata.")
        if max_isl_length_m is None or not isinstance(max_isl_length_m, (int, float)):
            raise ValueError("'max_isl_length_m' missing or invalid type in metadata.")

        # Fstate info
        fstate_key = metadata.get("fstate_data_key", "fstate")
        expected_fstate_format = metadata.get("fstate_data_format", "dict_tuple_nh_if1_if2")

        log.info("Static data extracted successfully from metadata.")
        num_satellites = len(satellites)
        num_ground_stations = len(ground_stations)
        log.info(
            f"Using {num_satellites} satellites and {num_ground_stations} ground stations from embedded data."
        )

    except Exception as e:
        log.exception(f"Error extracting or validating static data from PKL metadata: {e}")
        return False

    # --- Setup outputs ---
    base_output_dir = base_output_dir.rstrip("/")
    pdf_dir = f"{base_output_dir}/pdfs_s{src}_d{dst}"
    data_dir = f"{base_output_dir}/data_s{src}_d{dst}"
    local_shell.make_full_dir(pdf_dir)
    local_shell.make_full_dir(data_dir)

    # --- Process Time Steps ---
    current_path_nodes_cache = None
    rtt_ms_list = []
    fstate_for_path_module = {}

    log.info(
        f"Analyzing path from {src} to {dst} using fstate key '{fstate_key}' (format: '{expected_fstate_format}')..."
    )

    # Optional: Check format from metadata more strictly
    if expected_fstate_format != "dict_tuple_nh_if1_if2":
        log.warning(
            f"Metadata indicates fstate format is '{expected_fstate_format}', but this script handles 'dict_tuple_nh_if1_if2'. Processing assumes the latter structure."
        )

    for item_idx, time_step_item in enumerate(all_time_step_data):

        # Get time using 'time_since_epoch_ns'
        time_ns = time_step_item.get("time_since_epoch_ns")
        if time_ns is None:
            log.warning(f"Key 'time_since_epoch_ns' missing in item index {item_idx}. Skipping.")
            continue
        try:
            time_ns = int(float(time_ns))
        except (ValueError, TypeError):
            log.warning(f"Invalid time value '{time_ns}' in item index {item_idx}. Skipping.")
            continue
        time_ms = time_ns / 1_000_000.0

        # --- Fstate Processing (Dictionary format: {(c,d): (nh, if1, if2)}) ---
        raw_fstate_dict = time_step_item.get(fstate_key)
        fstate_for_path_module.clear()
        if raw_fstate_dict and isinstance(raw_fstate_dict, dict):
            for (curr_id, dest_id), fstate_val_tuple in raw_fstate_dict.items():
                if isinstance(fstate_val_tuple, tuple) and len(fstate_val_tuple) >= 1:
                    next_hop_id = fstate_val_tuple[0]
                    if next_hop_id != -1:
                        try:
                            fstate_for_path_module[(int(curr_id), int(dest_id))] = int(next_hop_id)
                        except (ValueError, TypeError):
                            log.warning(
                                f"Non-integer fstate ID ({curr_id},{dest_id})->{next_hop_id} @{time_ms}ms. Skipping."
                            )
        # --- End Fstate Processing ---

        # --- Path and RTT Calculation (Uses static data from metadata) ---
        path_there_nodes = get_path(src, dst, fstate_for_path_module)
        path_back_nodes = get_path(dst, src, fstate_for_path_module)
        rtt_ns_current_step = 0.0
        length_src_to_dst_m = None
        length_dst_to_src_m = None
        if path_there_nodes is not None:
            length_src_to_dst_m = compute_path_length_without_graph(
                path_there_nodes,
                epoch,
                time_ns,
                satellites,
                ground_stations,
                list_isls,
                max_gsl_length_m,
                max_isl_length_m,
            )
            if length_src_to_dst_m is None:
                path_there_nodes = None
        if path_back_nodes is not None:
            length_dst_to_src_m = compute_path_length_without_graph(
                path_back_nodes,
                epoch,
                time_ns,
                satellites,
                ground_stations,
                list_isls,
                max_gsl_length_m,
                max_isl_length_m,
            )
            if length_dst_to_src_m is None:
                path_back_nodes = None
        if (
            path_there_nodes is not None
            and path_back_nodes is not None
            and length_src_to_dst_m is not None
            and length_dst_to_src_m is not None
        ):
            rtt_ns_current_step = (length_src_to_dst_m + length_dst_to_src_m) / 0.299792458
        else:
            if length_src_to_dst_m is None or length_dst_to_src_m is None:
                path_there_nodes = None
        rtt_ms_list.append((time_ms, rtt_ns_current_step / 1e6 if rtt_ns_current_step > 0 else 0.0))

        # --- Plotting and Logging on Path Change (Uses static data from metadata) ---
        if current_path_nodes_cache != path_there_nodes:
            current_path_nodes_cache = path_there_nodes
            item_name = time_step_item.get("name", f"Item_{item_idx}")
            log.info(
                f"Change at t={time_ns} ns ({time_ms:.2f} ms, Item: {item_name}) for Src={src}, Dst={dst}"
            )
            # ... (Log path, length, RTT info as before) ...
            if current_path_nodes_cache is not None:
                path_str = " -> ".join(map(str, current_path_nodes_cache))
                len_s_d = length_src_to_dst_m
                len_d_s = length_dst_to_src_m if length_dst_to_src_m is not None else 0.0
                total_rtt_len_m = len_s_d + len_d_s
                log.info(f"  Path: {path_str}")
                log.info(
                    f"  Length (m): Forward={len_s_d:.2f}, Backward={len_d_s:.2f if length_dst_to_src_m is not None else 'N/A'}, RTT_Path={total_rtt_len_m:.2f}"
                )
                log.info(f"  RTT: {rtt_ns_current_step / 1e6:.3f} ms")
            else:
                log.info("  Path: Unreachable")
                log.info(f"  RTT: 0.000 ms")

            # --- Generate PDF plot (Insert full plotting code here) ---
            pdf_filename = f"{pdf_dir}/path_{src}_to_{dst}_time_{int(time_ms)}ms.pdf"
            log.debug(f"Generating plot: {pdf_filename}")

            fig = plt.figure(figsize=(16, 10))
            ax = plt.axes(projection=ccrs.PlateCarree())
            try:
                ax.stock_img()
            except Exception as img_e:
                log.warning(f"Stock image error: {img_e}. Using coastlines.")
                ax.coastlines(resolution="110m")
            ax.add_feature(cartopy.feature.OCEAN, zorder=0, alpha=0.5)
            ax.add_feature(
                cartopy.feature.LAND,
                zorder=0,
                edgecolor="black",
                linewidth=0.3,
                alpha=0.7,
                facecolor="#c0c0c0",
            )
            ax.add_feature(
                cartopy.feature.BORDERS, zorder=1, edgecolor="gray", linewidth=0.4, alpha=0.6
            )
            gl = ax.gridlines(
                draw_labels=True,
                dms=True,
                x_inline=False,
                y_inline=False,
                linewidth=0.5,
                linestyle=":",
                color="black",
                alpha=0.5,
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.xlabel_style = {"size": 8}
            gl.ylabel_style = {"size": 8}

            time_moment_skyfield = epoch.ts.tt_jd(epoch.tt + (time_ns * 1e-9 / 86400.0))
            nodes_in_current_path = set(
                current_path_nodes_cache if current_path_nodes_cache else []
            )

            # Plot Satellites
            for sat_id in range(num_satellites):
                is_used = sat_id in nodes_in_current_path
                try:
                    shadow_gs = create_basic_ground_station_for_satellite_shadow(
                        satellites[sat_id], epoch, time_moment_skyfield
                    )
                    lat, lon = float(shadow_gs["latitude_degrees_str"]), float(
                        shadow_gs["longitude_degrees_str"]
                    )
                except Exception as e:
                    log.debug(f"Skip sat {sat_id} plot: {e}")
                    continue
                ax.plot(
                    lon,
                    lat,
                    marker="^",
                    transform=ccrs.Geodetic(),
                    color=SATELLITE_USED_COLOR if is_used else SATELLITE_UNUSED_COLOR,
                    markersize=7 if is_used else 5,
                    fillstyle="full" if is_used else "none",
                    markeredgewidth=0.7,
                    zorder=6 if is_used else 3,
                    label="_nolegend_",
                )
                if is_used:
                    ax.text(
                        lon + 1.2,
                        lat + 0.6,
                        str(sat_id),
                        color=LABEL_COLOR,
                        fontsize=8,
                        weight="bold",
                        transform=ccrs.Geodetic(),
                        zorder=11,
                    )

            # Plot Ground Stations (Uses ground_stations list from metadata)
            for gs_idx in range(num_ground_stations):
                gs_node_id = gs_idx + num_satellites
                is_used = gs_node_id in nodes_in_current_path
                is_src_or_dst = gs_node_id == src or gs_node_id == dst
                # Use keys saved by read_ground_stations_extended (e.g., 'latitude_degrees_str')
                lat, lon = float(ground_stations[gs_idx]["latitude_degrees_str"]), float(
                    ground_stations[gs_idx]["longitude_degrees_str"]
                )
                current_gs_color = GROUND_STATION_UNUSED_COLOR
                current_msize = 5
                current_fstyle = "none"
                current_zorder = 2
                mew = 1.0
                if is_used:
                    current_gs_color = GROUND_STATION_USED_COLOR
                    current_msize = 8
                    current_fstyle = "full"
                    current_zorder = 7
                if is_src_or_dst:
                    current_gs_color = SRC_DST_GS_COLOR_USED if is_used else SRC_DST_GS_COLOR_UNUSED
                    current_msize = 10
                    current_fstyle = "full"
                    current_zorder = 8
                    mew = 1.5
                ax.plot(
                    lon,
                    lat,
                    marker="o",
                    transform=ccrs.Geodetic(),
                    color=current_gs_color,
                    markersize=current_msize,
                    fillstyle=current_fstyle,
                    markeredgewidth=mew,
                    markeredgecolor="black",
                    zorder=current_zorder,
                    label="_nolegend_",
                )
                if is_used or is_src_or_dst:
                    ax.text(
                        lon + 1.2,
                        lat + 0.6,
                        f"GS{ground_stations[gs_idx].get('gid', gs_idx)}",
                        color=LABEL_COLOR,
                        fontsize=8,
                        weight="bold",
                        transform=ccrs.Geodetic(),
                        zorder=11,
                    )  # Use gid from dict

            # Plot Path Links (Uses satellites/ground_stations lists from metadata)
            if current_path_nodes_cache and len(current_path_nodes_cache) > 1:
                for i in range(len(current_path_nodes_cache) - 1):
                    n1_id, n2_id = current_path_nodes_cache[i], current_path_nodes_cache[i + 1]
                    coords = []
                    for node_id in [n1_id, n2_id]:
                        try:
                            if node_id < num_satellites:
                                sgs = create_basic_ground_station_for_satellite_shadow(
                                    satellites[node_id], epoch, time_moment_skyfield
                                )
                                coords.append(
                                    (
                                        float(sgs["longitude_degrees_str"]),
                                        float(sgs["latitude_degrees_str"]),
                                    )
                                )
                            else:
                                gs_idx_ = node_id - num_satellites
                                if 0 <= gs_idx_ < num_ground_stations:
                                    coords.append(
                                        (
                                            float(
                                                ground_stations[gs_idx_]["longitude_degrees_str"]
                                            ),
                                            float(ground_stations[gs_idx_]["latitude_degrees_str"]),
                                        )
                                    )
                                else:
                                    raise ValueError(f"Invalid GS index {gs_idx_}")
                        except Exception as coord_e:
                            log.debug(f"Coord error node {node_id}: {coord_e}")
                            coords = []
                            break
                    if len(coords) == 2:
                        ax.plot(
                            [coords[0][0], coords[1][0]],
                            [coords[0][1], coords[1][1]],
                            color=ISL_COLOR,
                            linewidth=2.5,
                            transform=ccrs.Geodetic(),
                            zorder=5,
                            label="_nolegend_",
                        )

            # Finalize Plot (Title, Legend, Save)
            ax.set_global()
            path_display_str = (
                " -> ".join(map(str, current_path_nodes_cache))
                if current_path_nodes_cache
                else "Unreachable"
            )
            rtt_title_val = rtt_ns_current_step / 1e6 if current_path_nodes_cache is not None else 0
            fig.suptitle(
                f"Route Analysis: {src} to {dst} at Time {time_ms:.0f} ms", fontsize=16, y=0.98
            )
            ax.set_title(
                f"Path: {path_display_str}\nRTT = {rtt_title_val:.3f} ms", fontsize=10, pad=15
            )
            # (Legend creation code remains the same)
            legend_handles = []
            handle_map = {
                "Src/Dst GS (in Path)": Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=SRC_DST_GS_COLOR_USED,
                    markeredgecolor="black",
                    markersize=10,
                ),
                "Src/Dst GS (not in Path)": Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=SRC_DST_GS_COLOR_UNUSED,
                    markeredgecolor="black",
                    markersize=10,
                ),
                "Other GS (in Path)": Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=GROUND_STATION_USED_COLOR,
                    markeredgecolor="black",
                    markersize=8,
                ),
                "Other GS (Unused)": Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor="w",
                    markeredgecolor=GROUND_STATION_UNUSED_COLOR,
                    markersize=5,
                    fillstyle="none",
                ),
                "Satellite (in Path)": Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor=SATELLITE_USED_COLOR,
                    markersize=7,
                ),
                "Satellite (Unused)": Line2D(
                    [0],
                    [0],
                    marker="^",
                    color="w",
                    markerfacecolor=SATELLITE_UNUSED_COLOR,
                    fillstyle="none",
                    markersize=5,
                ),
                "Active Path Link": Line2D([0], [0], color=ISL_COLOR, lw=2.5),
            }
            items_needed = set()
            if any(n == src or n == dst for n in nodes_in_current_path):
                items_needed.add("Src/Dst GS (in Path)")
            if any(
                g["gid"] + num_satellites == src or g["gid"] + num_satellites == dst
                for g in ground_stations
            ) and not any(n == src or n == dst for n in nodes_in_current_path):
                items_needed.add("Src/Dst GS (not in Path)")
            if any(n >= num_satellites and n != src and n != dst for n in nodes_in_current_path):
                items_needed.add("Other GS (in Path)")
            if any(
                g["gid"] + num_satellites not in nodes_in_current_path
                and g["gid"] + num_satellites != src
                and g["gid"] + num_satellites != dst
                for g in ground_stations
            ):
                items_needed.add("Other GS (Unused)")
            if any(n < num_satellites for n in nodes_in_current_path):
                items_needed.add("Satellite (in Path)")
            if any(n not in nodes_in_current_path for n in range(num_satellites)):
                items_needed.add("Satellite (Unused)")
            if current_path_nodes_cache and len(current_path_nodes_cache) > 1:
                items_needed.add("Active Path Link")
            for item_label in handle_map:
                if item_label in items_needed:
                    handle = handle_map[item_label]
                    handle.set_label(item_label)
                    legend_handles.append(handle)
            if legend_handles:
                ax.legend(
                    handles=legend_handles,
                    loc="lower left",
                    fontsize="small",
                    frameon=True,
                    framealpha=0.9,
                )

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            try:
                plt.savefig(pdf_filename, bbox_inches="tight", dpi=200)
                log.info(f"Saved plot: {pdf_filename}")
            except Exception as plot_save_e:
                log.error(f"Error saving plot {pdf_filename}: {plot_save_e}")
            plt.close(fig)
            # --- End Plotting Logic ---

    # --- End of Loop ---

    # --- Save all RTT data ---
    output_rtt_csv_file = f"{data_dir}/rtt_s{src}_d{dst}_all_timesteps.csv"
    try:
        with open(output_rtt_csv_file, "w") as f_out:
            f_out.write("time_ms,rtt_ms\n")
            for t_ms, rtt_val_ms in rtt_ms_list:
                f_out.write(f"{float(t_ms):.3f},{rtt_val_ms:.6f}\n")
        log.info(f"All RTT data for {src}-{dst} saved to {output_rtt_csv_file}")
    except Exception as csv_e:
        log.error(f"Error writing RTT CSV file {output_rtt_csv_file}: {csv_e}")

    return True  # Indicate success


# --- Example Usage Block ---
if __name__ == "__main__":
    if "get_logger" not in dir(logger):
        import logging

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        log = logging.getLogger(__name__)

    log.info("Running example usage: Analysis script reading PKL with EMBEDDED metadata...")

    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_base = os.path.abspath(os.path.join(script_dir, "..", ".."))
    except NameError:
        project_base = os.path.abspath(".")
        log.warning(f"__file__ not defined.")

    # --- Configuration - Point to the correct PKL file ---
    # This PKL file MUST contain the embedded metadata generated by the updated runner script
    default_pkl_relative = "starlink_many_gs_results_with_metadata.pkl"  # INPUT PKL FILE NAME
    pkl_file = os.path.join(project_base, default_pkl_relative)

    default_output_relative = "results_graphical_analysis_from_embedded"  # OUTPUT directory
    output_dir = os.path.join(project_base, default_output_relative)

    source_node_id = 720  # Example Source GS ID (Adjust!)
    dest_node_id = 770  # Example Destination GS ID (Adjust!)
    # --- End Configuration ---

    log.info(f"--- Running analysis with configuration: ---")
    log.info(f"  Project Base: {project_base}")
    log.info(f"  Output Dir: {output_dir}")
    log.info(f"  Input PKL File (must contain embedded metadata): {pkl_file}")
    log.info(f"  Source Node: {source_node_id}, Dest Node: {dest_node_id}")

    if not os.path.exists(pkl_file):
        log.error(f"PKL file not found: {pkl_file}")
        log.error("Please ensure this file exists (and contains metadata) or modify 'pkl_file'.")
    else:
        try:
            # Call the function that reads embedded metadata
            success = print_graphical_routes_and_rtt_from_embedded_pkl(
                base_output_dir=output_dir,
                fstate_pkl_path=pkl_file,
                src=source_node_id,
                dst=dest_node_id,
            )
            if success:
                log.info("Analysis finished successfully.")
            else:
                log.error("Analysis failed. Check previous errors.")
        except ImportError as e:
            log.error(f"ImportError: {e}. Check script location and PYTHONPATH.")
        except FileNotFoundError as e:
            log.error(f"FileNotFoundError: {e}. Problem accessing data.")
        except KeyError as e:
            log.error(f"KeyError: {e}. Required key missing from PKL metadata.")
        except Exception as e:
            log.exception(f"An unexpected error occurred during analysis: {e}")

# ----- END OF THE SCRIPT -----
