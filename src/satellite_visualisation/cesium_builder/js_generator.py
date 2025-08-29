import math

from . import util


def generate_shells_js(
    shells,
    global_epoch_str,
    global_ephem_epoch_for_elements,
    element_shell_phase_diff,
    element_shell_eccentricity,
    element_shell_arg_of_perigee,
):
    viz_string = ""
    for shell_idx, shell_config in enumerate(shells):
        shell_name = shell_config.get("name", f"ElementsShell_{shell_idx + 1}")
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
            entity_disp_name = f"{shell_name}_Sat_{sat_idx + 1}"
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
            viz_string += "viewer.entities.add({\n"
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
    return viz_string


def generate_ground_stations_js(ground_stations):
    viz_string = ""
    for gs_idx, gs_data in enumerate(ground_stations):
        gs_name = gs_data.get("name", f"GroundStation_{gs_idx + 1}")
        if "latitude" not in gs_data or "longitude" not in gs_data:
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
    return viz_string
