"""Write one LEOPath config per real orbital shell of the multi-shell constellations.

A constellation reaches tens of thousands of satellites only by adding shells;
no operator grows a single shell that far. Each shell here becomes its own
config, and its own 6G-RUPA layer, so per-shell state and preprocessing cost
can be measured on the geometry operators actually filed.

Every shell parameter comes from an FCC order:

- Starlink Gen1, FCC 21-48 para 4
- Kuiper, FCC 20-102 fn 3 (its "42 orbital planes" at 610 km is a typo for the
  42 degree inclination: 42 x 36 is not 1296, and the order totals 98 planes)
- Starlink Gen2, FCC 22-91 para 7; 7 500 of the three authorised shells'
  10 080 slots are authorised, so these configs carry the filed geometry

Near-polar shells (inclination above 80 degrees) are laid out as a Walker star,
ascending nodes over 180 degrees, following the rule in
``generate_tles_from_scratch``: nodes 180 degrees apart would otherwise trace
the same ground track in opposite directions. The filings don't give the node
spread, so this is an assumption.

Run from the repository root:  python scripts/make_shell_configs.py
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leopath.topology.walker_geometry import WalkerShell  # noqa: E402

EARTH_MU_M3_S2 = 398600.8e9
EARTH_RADIUS_M = 6378135.0

# name, planes, sats per plane, altitude km, inclination deg, source
SHELLS = [
    ("starlink_gen1_550", 72, 22, 550, 53.0, "FCC 21-48 para 4"),
    ("starlink_gen1_540", 72, 22, 540, 53.2, "FCC 21-48 para 4"),
    ("starlink_gen1_570", 36, 20, 570, 70.0, "FCC 21-48 para 4"),
    ("starlink_gen1_560a", 6, 58, 560, 97.6, "FCC 21-48 para 4"),
    ("starlink_gen1_560b", 4, 43, 560, 97.6, "FCC 21-48 para 4"),
    ("kuiper_590", 28, 28, 590, 33.0, "FCC 20-102 fn 3"),
    ("kuiper_610", 36, 36, 610, 42.0, "FCC 20-102 fn 3, 98-plane total"),
    ("kuiper_630", 34, 34, 630, 51.9, "FCC 20-102 fn 3"),
    ("starlink_gen2_525", 28, 120, 525, 53.0, "FCC 22-91 para 7"),
    ("starlink_gen2_530", 28, 120, 530, 43.0, "FCC 22-91 para 7"),
    ("starlink_gen2_535", 28, 120, 535, 33.0, "FCC 22-91 para 7"),
    ("starlink_gen2_360", 30, 120, 360, 96.9, "FCC 22-91 para 7, largest filed shell"),
]


def mean_motion_rev_per_day(altitude_km: float) -> float:
    semi_major_axis = EARTH_RADIUS_M + altitude_km * 1000.0
    radians_per_s = math.sqrt(EARTH_MU_M3_S2 / semi_major_axis**3)
    return round(radians_per_s * 86400.0 / (2.0 * math.pi), 2)


def max_isl_length_m(template: dict, altitude_m: float) -> float:
    earth = template["earth"]
    radius, floor = earth["radius_m"], earth["isl_min_altitude_m"]
    return 2.0 * math.sqrt((radius + altitude_m) ** 2 - (radius + floor) ** 2)


def longest_rung_m(shell: WalkerShell) -> float:
    """Longest cross-plane link over one orbit, the wrap rung included when it exists."""
    period_s = 86400.0 / shell.mean_motion_rev_per_day
    wrap = shell.raan_spread_deg >= 360.0
    longest = 0.0
    for step in range(24):
        time_s = period_s * step / 24.0
        for row in range(shell.sats_per_plane):
            for plane in range(shell.planes if wrap else shell.planes - 1):
                longest = max(longest, shell.rung_length_m(plane, row, time_s))
    return longest


def main() -> None:
    template = yaml.safe_load((ROOT / "leopath/config/starlink.yaml").read_text())
    out_dir = ROOT / "leopath/config/shells"
    out_dir.mkdir(exist_ok=True)
    print(f"{'shell':22s} {'P x S':>9s} {'sats':>5s} {'spread':>6s} {'longest rung':>13s} {'LOS limit':>10s}")
    for name, planes, sats, altitude_km, inclination, source in SHELLS:
        spread = 180.0 if inclination > 80.0 else 360.0
        motion = mean_motion_rev_per_day(altitude_km)
        config = copy.deepcopy(template)
        config["constellation"] = {
            "name": name,
            "num_orbits": planes,
            "num_sats_per_orbit": sats,
            "phase_diff": True,
            "inclination_degree": inclination,
            "eccentricity": 0.0000001,
            "arg_of_perigee_degree": 0.0,
            "mean_motion_rev_per_day": motion,
            "raan_spread_degree": spread,
            "tle_output_filename": f"tles_{name}.txt",
        }
        config["satellite"]["altitude_m"] = altitude_km * 1000
        shell = WalkerShell(planes, sats, inclination, motion, spread, True)
        rung, limit = longest_rung_m(shell), max_isl_length_m(config, altitude_km * 1000)
        flag = "" if rung <= limit else "  EXCEEDS line of sight"
        print(
            f"{name:22s} {planes:>4d} x {sats:<3d} {planes * sats:>5d} {spread:>6.0f}"
            f" {rung / 1000:>10.0f} km {limit / 1000:>7.0f} km{flag}"
        )
        header = (
            f"# {name}: {planes} planes x {sats} satellites at {altitude_km} km, "
            f"{inclination} deg ({source}).\n"
            f"# Generated by scripts/make_shell_configs.py; edit the script, not this file.\n"
        )
        (out_dir / f"{name}.yaml").write_text(header + yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
