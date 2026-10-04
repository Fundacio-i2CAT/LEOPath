"""How many ground terminals does one satellite carry, and what does that cost in addressing?

A ground terminal's topological address is its attachment satellite's locator
plus an endpoint index x under that satellite. The 24 ground stations of the
routing evaluation say nothing about how wide x has to be or how often a large
terminal population renumbers, so this script measures that directly, without
routing any packet:

- place N terminals: spread evenly over the surface between +/- max-lat, or
  drawn in proportion to census population from the points that
  ``fetch_population_points.py`` builds (``--layout census --population``);
- optionally move every terminal at a constant speed on its own random
  heading (``--speed-kmh``), to check how much terminal mobility adds to the
  churn the satellites' own motion causes;
- at every snapshot, attach each terminal to one satellite above the minimum
  elevation, by one of two policies: ``nearest`` moves to the nearest satellite
  every snapshot, the most churn; ``stay_while_visible`` keeps the current
  satellite until it drops below the minimum elevation, then takes the nearest,
  the least churn a K = 1 terminal can have;
- report the per-satellite terminal load, the endpoint bits x would need to
  number the busiest satellite's terminals (plus the satellite itself), how
  many terminals change address per minute, and the directory updates per
  second that implies (one per address change).

Evenly spread terminals are the optimistic case for the busiest satellite;
census-weighted ones put terminals where people live.

Usage:
  python scripts/terminal_population.py leopath/config/starlink.yaml \\
      --terminals 1000 10000 100000 1000000 --layout uniform clustered
"""

from __future__ import annotations

import argparse
import gzip
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree
from sgp4.api import SatrecArray, jday

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leopath.tles.generate_tles_from_scratch import (  # noqa: E402
    generate_tles_from_scratch_with_sgp,
)

EARTH_RADIUS_KM = 6378.135


def satellites(config: dict, scratch: Path) -> SatrecArray:
    from sgp4.api import Satrec

    c = config["constellation"]
    path = scratch / f"tles_{c['name']}.txt"
    generate_tles_from_scratch_with_sgp(
        str(path),
        c["name"],
        c["num_orbits"],
        c["num_sats_per_orbit"],
        c["phase_diff"],
        c["inclination_degree"],
        c["eccentricity"],
        c["arg_of_perigee_degree"],
        c["mean_motion_rev_per_day"],
        c.get("raan_spread_degree", 360.0),
    )
    lines = path.read_text().splitlines()[1:]
    return SatrecArray(
        [Satrec.twoline2rv(lines[i + 1], lines[i + 2]) for i in range(0, len(lines), 3)]
    )


def gmst_rad(jd: float, fr: float) -> float:
    t = (jd - 2451545.0 + fr) / 36525.0
    seconds = (
        67310.54841 + (876600.0 * 3600.0 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t**3
    )
    return math.radians((seconds % 86400.0) / 240.0)


def unit_vectors(latitudes_deg: np.ndarray, longitudes_deg: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(latitudes_deg), np.radians(longitudes_deg)
    return np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1)


def place_terminals(count: int, layout: str, population, max_lat: float, rng):
    """Latitudes and longitudes (degrees) of ``count`` terminals."""
    if layout == "uniform":
        # Even over the sphere's surface between +/- max_lat.
        z = rng.uniform(-math.sin(math.radians(max_lat)), math.sin(math.radians(max_lat)), count)
        return np.degrees(np.arcsin(z)), rng.uniform(-180.0, 180.0, count)
    lat, lon, weight, spread = population
    pick = rng.choice(len(weight), size=count, p=weight / weight.sum())
    return (
        np.clip(lat[pick] + rng.normal(0.0, 1.0, count) * spread[pick], -89.0, 89.0),
        lon[pick] + rng.normal(0.0, 1.0, count) * spread[pick],
    )


def load_population(paths: list[str]):
    rows = []
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt") as handle:
            next(handle)
            for line in handle:
                _country, lat, lon, pop, spread = line.rstrip("\n").split(",")
                rows.append((float(lat), float(lon), float(pop), float(spread)))
    data = np.array(rows)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def move(lat, lon, heading, km: float):
    """Advance every terminal ``km`` along its heading (flat-Earth step, fine at 1 min)."""
    lat = np.clip(lat + km * np.cos(heading) / 111.2, -89.0, 89.0)
    lon = lon + km * np.sin(heading) / (111.2 * np.maximum(np.cos(np.radians(lat)), 0.01))
    return lat, lon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--terminals", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--layout", nargs="+", default=["uniform"])
    parser.add_argument("--population", nargs="*", default=[])
    parser.add_argument("--speed-kmh", type=float, nargs="+", default=[0.0])
    parser.add_argument("--policy", nargs="+", default=["nearest", "stay_while_visible"])
    parser.add_argument("--hours", type=float, default=1.0)
    parser.add_argument("--step-minutes", type=float, default=1.0)
    parser.add_argument("--min-elevation", type=float, default=25.0)
    parser.add_argument("--max-lat", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--scratch", default="/tmp")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    if "census" in args.layout and not args.population:
        parser.error("--layout census needs --population FILE")
    population = load_population(args.population) if args.population else None
    sat_array = satellites(config, Path(args.scratch))
    altitude_km = config["satellite"]["altitude_m"] / 1000.0
    # Ground range (as a central angle) out to the minimum elevation.
    elevation = math.radians(args.min_elevation)
    ratio = EARTH_RADIUS_KM / (EARTH_RADIUS_KM + altitude_km)
    max_central = math.pi / 2 - elevation - math.asin(ratio * math.cos(elevation))
    max_chord = 2.0 * math.sin(max_central / 2.0)

    jd0, fr0 = jday(2000, 1, 1, 0, 0, 0)
    steps = int(args.hours * 60 / args.step_minutes) + 1
    times = [fr0 + k * args.step_minutes / 1440.0 for k in range(steps)]
    sat_count = len(sat_array)
    print(
        "config,layout,policy,speed_kmh,terminals,unserved_share,load_mean,load_p99,load_max,"
        "x_bits_needed,address_changes_per_terminal_minute,directory_updates_per_s"
    )
    runs = [
        (lay, pol, v, n)
        for lay in args.layout
        for pol in args.policy
        for v in args.speed_kmh
        for n in args.terminals
    ]
    for layout, policy, speed, count in runs:
        rng = np.random.default_rng(args.seed)
        lat, lon = place_terminals(count, layout, population, args.max_lat, rng)
        heading = rng.uniform(0.0, 2.0 * math.pi, count)
        previous = None
        loads, unserved, changes, compared = [], 0, 0, 0
        for step, fr in enumerate(times):
            if step and speed:
                lat, lon = move(lat, lon, heading, speed * args.step_minutes / 60.0)
            terminals = unit_vectors(lat, lon)
            _err, positions, _vel = sat_array.sgp4(np.array([jd0]), np.array([fr]))
            eci = positions[:, 0, :]
            theta = gmst_rad(jd0, fr)
            c, s = math.cos(theta), math.sin(theta)
            ecef = np.stack(
                [c * eci[:, 0] + s * eci[:, 1], -s * eci[:, 0] + c * eci[:, 1], eci[:, 2]],
                axis=1,
            )
            sub_points = ecef / np.linalg.norm(ecef, axis=1, keepdims=True)
            distance, nearest = cKDTree(sub_points).query(terminals)
            served = distance <= max_chord
            attach = np.where(served, nearest, -1)
            if policy == "stay_while_visible" and previous is not None:
                keep = np.flatnonzero(previous >= 0)
                gap = np.linalg.norm(sub_points[previous[keep]] - terminals[keep], axis=1)
                still = keep[gap <= max_chord]
                attach[still] = previous[still]
                served = attach >= 0
            unserved += int((~served).sum())
            loads.append(np.bincount(attach[served], minlength=sat_count))
            if previous is not None:
                both = (attach >= 0) & (previous >= 0)
                changes += int((attach[both] != previous[both]).sum())
                compared += int(both.sum())
            previous = attach
        load = np.concatenate(loads)
        peak = int(load.max())
        minutes = (steps - 1) * args.step_minutes
        changes_per_terminal_minute = changes / max(compared, 1)
        print(
            f"{Path(args.config).stem},{layout},{policy},{speed:g},{count},"
            f"{unserved / (count * steps):.4f},{load.mean():.2f},"
            f"{np.percentile(load, 99):.1f},{peak},{max(1, math.ceil(math.log2(peak + 1)))},"
            f"{changes_per_terminal_minute / args.step_minutes:.4f},"
            f"{changes / (minutes * 60.0) if minutes else 0.0:.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
