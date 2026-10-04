"""How close do real shells fly to the ideal Walker grid the estimators assume?

Reads a CelesTrak TLE snapshot, propagates every satellite to one instant with
SGP4, keeps those within an inclination and altitude window, and infers the
grid the way a topological layer would have to: planes by clustering the right
ascension of the ascending node, slots by ordering the argument of latitude
within each plane. It then reports how regular that grid is:

- how many planes, and how full each one is;
- how evenly the planes are spread in node, against an even spacing;
- how evenly satellites are spread along each plane, against 360/S;
- how far each in-plane neighbour distance is from the ideal rail length a
  Walker model with the same plane occupancy would predict.

Usage: python scripts/celestrak_shell_geometry.py TLE_FILE INC_DEG ALT_KM [--inc-tol 0.5] [--alt-tol 15]
"""

from __future__ import annotations

import argparse
import math
import statistics
from datetime import datetime, timezone

import numpy as np
from sgp4.api import Satrec, jday

MU_KM3_S2 = 398600.8
EARTH_RADIUS_KM = 6378.135


def load(path: str) -> list[tuple[str, Satrec]]:
    lines = [line.rstrip() for line in open(path) if line.strip()]
    return [
        (lines[i].strip(), Satrec.twoline2rv(lines[i + 1], lines[i + 2]))
        for i in range(0, len(lines) - 2, 3)
    ]


def mean_altitude_km(satellite: Satrec) -> float:
    semi_major = (MU_KM3_S2 / (satellite.no_kozai / 60.0) ** 2) ** (1.0 / 3.0)
    return semi_major - EARTH_RADIUS_KM


def node_and_latitude(position: np.ndarray, velocity: np.ndarray) -> tuple[float, float]:
    """Right ascension of the ascending node and argument of latitude, in degrees."""
    h = np.cross(position, velocity)
    node = np.cross([0.0, 0.0, 1.0], h)
    raan = math.degrees(math.atan2(node[1], node[0])) % 360.0
    n_hat = node / np.linalg.norm(node)
    r_hat = position / np.linalg.norm(position)
    u = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(n_hat, r_hat))))))
    if position[2] < 0:
        u = 360.0 - u
    return raan, u


def cluster_ring(values: list[float], gap_deg: float) -> list[list[int]]:
    """Group angles on a circle, splitting wherever consecutive values differ by more than gap_deg."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    sorted_values = [values[i] for i in order]
    gaps = [
        (sorted_values[(k + 1) % len(order)] - sorted_values[k]) % 360.0 for k in range(len(order))
    ]
    start = max(range(len(gaps)), key=lambda k: gaps[k]) + 1  # begin after the widest gap
    groups, current = [], []
    for step in range(len(order)):
        k = (start + step) % len(order)
        current.append(order[k])
        if gaps[k] > gap_deg:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def ring_gaps(angles: list[float]) -> list[float]:
    ordered = sorted(angles)
    return [(ordered[(k + 1) % len(ordered)] - ordered[k]) % 360.0 for k in range(len(ordered))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tle")
    parser.add_argument("inclination", type=float)
    parser.add_argument("altitude", type=float)
    parser.add_argument("--inc-tol", type=float, default=0.5)
    parser.add_argument("--alt-tol", type=float, default=15.0)
    parser.add_argument("--plane-gap", type=float, default=1.0)
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    members = []
    for name, satellite in load(args.tle):
        if abs(math.degrees(satellite.inclo) - args.inclination) > args.inc_tol:
            continue
        altitude = mean_altitude_km(satellite)
        if abs(altitude - args.altitude) > args.alt_tol:
            continue
        error, position, velocity = satellite.sgp4(jd, fr)
        if error:
            continue
        raan, u = node_and_latitude(np.array(position), np.array(velocity))
        members.append((name, raan, u, np.array(position), altitude))

    planes = cluster_ring([m[1] for m in members], args.plane_gap)
    planes = [p for p in planes if len(p) >= 3]
    sizes = sorted(len(p) for p in planes)
    plane_nodes = [statistics.fmean(members[i][1] for i in p) for p in planes]
    node_gaps = ring_gaps(plane_nodes)
    median_node_gap = statistics.median(node_gaps)

    slot_errors, rail_errors = [], []
    for plane in planes:
        count = len(plane)
        gaps = ring_gaps([members[i][2] for i in plane])
        ideal = 360.0 / count
        slot_errors.extend(abs(g - ideal) / ideal for g in gaps)
        # In-plane neighbour chord against the ideal rail for this occupancy.
        ordered = sorted(plane, key=lambda i: members[i][2])
        radius = statistics.fmean(np.linalg.norm(members[i][3]) for i in plane)
        ideal_rail = 2.0 * radius * math.sin(math.pi / count)
        for k in range(count):
            a, b = members[ordered[k]][3], members[ordered[(k + 1) % count]][3]
            rail_errors.append(abs(float(np.linalg.norm(a - b)) - ideal_rail) / ideal_rail)

    def pct(values, q):
        return float(np.percentile(values, q)) * 100.0

    print(
        f"{args.tle}  inc {args.inclination}±{args.inc_tol}  alt {args.altitude}±{args.alt_tol} km  at {now:%Y-%m-%d %H:%M} UTC"
    )
    print(
        f"  satellites {len(members)}   planes {len(planes)}   sats/plane min {sizes[0]} median {statistics.median(sizes)} max {sizes[-1]}"
    )
    print(
        f"  plane node spacing: median {median_node_gap:.2f} deg, "
        f"relative deviation p50 {statistics.median(abs(g - median_node_gap) / median_node_gap for g in node_gaps) * 100:.1f}% "
        f"max {max(abs(g - median_node_gap) / median_node_gap for g in node_gaps) * 100:.0f}%"
    )
    print(
        f"  in-plane slot spacing vs 360/S: p50 {pct(slot_errors, 50):.1f}%  p90 {pct(slot_errors, 90):.1f}%  max {max(slot_errors) * 100:.0f}%"
    )
    print(
        f"  in-plane neighbour distance vs ideal rail: p50 {pct(rail_errors, 50):.1f}%  p90 {pct(rail_errors, 90):.1f}%  max {max(rail_errors) * 100:.0f}%"
    )


if __name__ == "__main__":
    main()
