import math
import os
from src import logger

log = logger.get_logger(__name__)

try:
    from . import util
except (ImportError, SystemError):
    import util

# Generate static visualizations for entire constellation (multiple shells).

EARTH_RADIUS = 6378135.0  # WGS72 value; taken from https://geographiclib.sourceforge.io/html/NET/NETGeographicLib_8h_source.html

# CONSTELLATION GENERATION GENERAL CONSTANTS
ECCENTRICITY = (
    0.0000001  # Circular orbits are zero, but pyephem does not permit 0, so lowest possible value
)
ARG_OF_PERIGEE_DEGREE = 0.0
PHASE_DIFF = True
EPOCH = "2000-01-01 00:00:00"

# Shell wise color codes
COLOR = ["CRIMSON", "FORESTGREEN", "DODGERBLUE", "PERU", "BLUEVIOLET", "DARKMAGENTA"]

# CONSTELLATION SPECIFIC PARAMETERS
NAME = "Starlink"

SHELL_CNTR = 1

MEAN_MOTION_REV_PER_DAY = [None] * SHELL_CNTR
ALTITUDE_M = [None] * SHELL_CNTR
NUM_ORBS = [None] * SHELL_CNTR
NUM_SATS_PER_ORB = [None] * SHELL_CNTR
INCLINATION_DEGREE = [None] * SHELL_CNTR
BASE_ID = [None] * SHELL_CNTR
ORB_WISE_IDS = [None] * SHELL_CNTR

MEAN_MOTION_REV_PER_DAY[0] = 15.19  # Altitude ~550000 km
ALTITUDE_M[0] = 550000  # Altitude ~550000 km
NUM_ORBS[0] = 72
NUM_SATS_PER_ORB[0] = 22
INCLINATION_DEGREE[0] = 53
BASE_ID[0] = 0
ORB_WISE_IDS[0] = []

# MEAN_MOTION_REV_PER_DAY[1] = 13.4  # Altitude ~1110 km
# ALTITUDE_M[1] = 1110000  # Altitude ~1110 km
# NUM_ORBS[1] = 32
# NUM_SATS_PER_ORB[1] = 50
# INCLINATION_DEGREE[1] = 53.8
# BASE_ID[1] = 1584
# ORB_WISE_IDS[1] = []

# MEAN_MOTION_REV_PER_DAY[2] = 13.35  # Altitude ~1130 km
# ALTITUDE_M[2] = 1130000  # Altitude ~1130 km
# NUM_ORBS[2] = 8
# NUM_SATS_PER_ORB[2] = 50
# INCLINATION_DEGREE[2] = 74
# BASE_ID[2] = 3184
# ORB_WISE_IDS[2] = []

# MEAN_MOTION_REV_PER_DAY[3] = 12.97  # Altitude ~1275 km
# ALTITUDE_M[3] = 1275000  # Altitude ~1275 km
# NUM_ORBS[3] = 5
# NUM_SATS_PER_ORB[3] = 75
# INCLINATION_DEGREE[3] = 81
# BASE_ID[3] = 3584
# ORB_WISE_IDS[3] = []

# MEAN_MOTION_REV_PER_DAY[4] = 12.84  # Altitude ~1325 km
# ALTITUDE_M[4] = 1325000  # Altitude ~1325 km
# NUM_ORBS[4] = 6
# NUM_SATS_PER_ORB[4] = 75
# INCLINATION_DEGREE[4] = 70
# BASE_ID[4] = 3959
# ORB_WISE_IDS[4] = []

# General files needed to generate visualizations
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
topFile = os.path.join(BASE_DIR, "static_html/top.html")  # Adjusted path
bottomFile = os.path.join(BASE_DIR, "static_html/bottom.html")  # Adjusted path

# Output directory for creating visualization HTML files
OUT_DIR = os.path.join(BASE_DIR, "viz_output/")
OUT_HTML_FILE = os.path.join(OUT_DIR, NAME + ".html")


def generate_satellite_trajectories():
    """
    Generates and adds satellite orbits to visualization.
    :return: viz_string
    """
    viz_string = ""
    for i in range(0, SHELL_CNTR):
        sat_objs = util.generate_sat_obj_list(
            NUM_ORBS[i],
            NUM_SATS_PER_ORB[i],
            EPOCH,
            PHASE_DIFF,
            INCLINATION_DEGREE[i],
            ECCENTRICITY,
            ARG_OF_PERIGEE_DEGREE,
            MEAN_MOTION_REV_PER_DAY[i],
            ALTITUDE_M[i],
        )
        for j in range(len(sat_objs)):
            sat_objs[j]["sat_obj"].compute(EPOCH)
            viz_string += (
                "var redSphere = viewer.entities.add({name : '', position: Cesium.Cartesian3.fromDegrees("
                + str(math.degrees(sat_objs[j]["sat_obj"].sublong))
                + ", "
                + str(math.degrees(sat_objs[j]["sat_obj"].sublat))
                + ", "
                + str(sat_objs[j]["alt_km"] * 1000)
                + "), "
                + "ellipsoid : {radii : new Cesium.Cartesian3(30000.0, 30000.0, 30000.0), "
                + "material : Cesium.Color.BLACK.withAlpha(1),}});\n"
            )
        orbit_links = util.find_orbit_links(sat_objs, NUM_ORBS[i], NUM_SATS_PER_ORB[i])
        for key in orbit_links:
            sat1 = orbit_links[key]["sat1"]
            sat2 = orbit_links[key]["sat2"]
            viz_string += (
                "viewer.entities.add({name : '', polyline: { positions: Cesium.Cartesian3.fromDegreesArrayHeights(["
                + str(math.degrees(sat_objs[sat1]["sat_obj"].sublong))
                + ","
                + str(math.degrees(sat_objs[sat1]["sat_obj"].sublat))
                + ","
                + str(sat_objs[sat1]["alt_km"] * 1000)
                + ","
                + str(math.degrees(sat_objs[sat2]["sat_obj"].sublong))
                + ","
                + str(math.degrees(sat_objs[sat2]["sat_obj"].sublat))
                + ","
                + str(sat_objs[sat2]["alt_km"] * 1000)
                + "]), "
                + "width: 0.5, arcType: Cesium.ArcType.NONE, "
                + "material: new Cesium.PolylineOutlineMaterialProperty({ "
                + "color: Cesium.Color."
                + COLOR[i]
                + ".withAlpha(0.4), outlineWidth: 0, outlineColor: Cesium.Color.BLACK})}});"
            )
    return viz_string


def write_viz_files(viz_string):
    """
    Writes JSON and HTML files to the output folder.
    :param viz_string: The visualization string to write to the HTML file.
    :return: None
    """
    # Ensure the output directory exists
    os.makedirs(OUT_DIR, exist_ok=True)

    # Write the HTML file
    with open(OUT_HTML_FILE, "w") as writer_html:
        with open(topFile, "r") as fi:
            writer_html.write(fi.read())
        writer_html.write(viz_string)
        with open(bottomFile, "r") as fb:
            writer_html.write(fb.read())


viz_string = generate_satellite_trajectories()
log.info("Generated visualization string for {} satellites.".format(NAME))
write_viz_files(viz_string)
