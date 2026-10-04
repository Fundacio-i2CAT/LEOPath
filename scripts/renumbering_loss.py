"""Could renumbering lose packets? Timing of address changes at 1-second resolution.

A station renumbers when the satellite its address is under stops being its
attachment on that half (northbound or southbound). Following the address-change
procedure (Grasa et al., EuCNC 2017), the station sends a flow update to the far
end of each flow and keeps accepting the old address until a deprecation timer
fires. In a wired network the old address stays reachable for as long as the
timer says. In LEO it is tied to the old satellite S1: packets addressed to it
are delivered only while S1 still has a ground link to the station.

So a flow loses packets only if S1 sets before the peer has switched and the
packets it sent to the old address have arrived:

    slack  = time from the switch until S1 drops out of the station's range
    needed = flow update to the peer (one way) + packets in flight back (one way)
             + processing margin
    loss   iff slack < needed

The station needs no extra beam for this: the flows move to its current address,
which may be its other half's attachment, already up, and S1 is kept until
drained.

The script replays SGP4 positions at 1 s over the run and a look-ahead window,
attaches every station to its nearest satellite on each half (as one_per_half
does), and for every change reports the slack, whether the switch was forced by
S1 setting, and, for each peer station, needed against slack. Path delays come
from the +Grid ISL graph at the minute of the event.

    python scripts/renumbering_loss.py starlink --minutes 60 > out.txt
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from sgp4.api import jday

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from terminal_population import gmst_rad, satellites  # noqa: E402

C_KM_S = 299_792.458
EARTH_R_KM = 6378.135


def station_vectors(stations: list[dict]) -> np.ndarray:
    lat = np.radians([float(s["latitude"]) for s in stations])
    lon = np.radians([float(s["longitude"]) for s in stations])
    return EARTH_R_KM * np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], axis=1
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--stations", default="leopath/config/ground_stations_dense.yaml")
    parser.add_argument("--minutes", type=int, default=60)
    parser.add_argument("--lookahead-minutes", type=int, default=20)
    parser.add_argument("--processing-ms", type=float, default=10.0)
    parser.add_argument("--scratch", default="/tmp")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "leopath/config" / f"{args.config}.yaml").read_text())
    c = config["constellation"]
    planes, slots = int(c["num_orbits"]), int(c["num_sats_per_orbit"])
    wrap = float(c.get("raan_spread_degree", 360.0)) >= 359.0
    alt = config["satellite"]["altitude_m"] / 1000.0
    cone = math.radians(config["satellite"]["cone_angle_degrees"])
    max_gsl = math.hypot(alt / math.tan(cone), alt)
    payload = yaml.safe_load(
        open(ROOT / args.stations if not Path(args.stations).is_absolute() else args.stations)
    )
    stations = payload["ground_stations"] if isinstance(payload, dict) else payload
    gs = station_vectors(stations)
    n_gs = len(stations)

    sat_array = satellites(config, Path(args.scratch))
    n_sat = planes * slots
    steps = (args.minutes + args.lookahead_minutes) * 60
    jd0, fr0 = jday(2000, 1, 1, 0, 0, 0)
    frs = fr0 + np.arange(steps) / 86400.0
    _err, pos, vel = sat_array.sgp4(np.full(steps, jd0), frs)  # (sat, time, 3) in TEME km

    # Earth-fixed satellite positions, so stations stay put.
    theta = np.array([gmst_rad(jd0, f) for f in frs])
    cth, sth = np.cos(theta), np.sin(theta)
    x = cth[None, :] * pos[:, :, 0] + sth[None, :] * pos[:, :, 1]
    y = -sth[None, :] * pos[:, :, 0] + cth[None, :] * pos[:, :, 1]
    ecef = np.stack([x, y, pos[:, :, 2]], axis=2)
    ascending = vel[:, :, 2] > 0.0  # heading north

    # range[g, s, t] in km
    rng = np.empty((n_gs, n_sat, steps), dtype=np.float32)
    for g in range(n_gs):
        rng[g] = np.linalg.norm(ecef - gs[g][None, None, :], axis=2)
    visible = rng <= max_gsl

    def attach(g: int, t: int, half: bool) -> int:
        mask = visible[g, :, t] & (ascending[:, t] == half)
        if not mask.any():
            return -1
        idx = np.flatnonzero(mask)
        return int(idx[np.argmin(rng[g, idx, t])])

    # +Grid ISL graph, one per minute, shortest one-way delays between satellites.
    edges = []
    for p in range(planes):
        for s in range(slots):
            i = p * slots + s
            edges.append((i, p * slots + (s + 1) % slots))
            if p + 1 < planes or wrap:
                edges.append((i, ((p + 1) % planes) * slots + s))
    edges = np.array(edges)
    delay_cache: dict[int, np.ndarray] = {}

    def isl_delay_s(minute: int) -> np.ndarray:
        if minute not in delay_cache:
            t = minute * 60
            w = np.linalg.norm(pos[edges[:, 0], t] - pos[edges[:, 1], t], axis=1) / C_KM_S
            graph = csr_matrix(
                (
                    np.concatenate([w, w]),
                    (
                        np.concatenate([edges[:, 0], edges[:, 1]]),
                        np.concatenate([edges[:, 1], edges[:, 0]]),
                    ),
                ),
                shape=(n_sat, n_sat),
            )
            delay_cache[minute] = shortest_path(graph, method="D", directed=False)
        return delay_cache[minute]

    horizon = args.minutes * 60
    current = {(g, h): attach(g, 0, h) for g in range(n_gs) for h in (True, False)}
    events, forced, slacks, checks, losses, shortfalls = 0, 0, [], 0, 0, []
    for t in range(1, horizon):
        for g in range(n_gs):
            for half in (True, False):
                old = current[(g, half)]
                new = attach(g, t, half)
                if new == old:
                    continue
                current[(g, half)] = new
                if old < 0:
                    continue
                events += 1
                gone = np.flatnonzero(~visible[g, old, t:])
                slack = float(gone[0]) if len(gone) else float(steps - t)
                if slack == 0.0:
                    forced += 1
                slacks.append(slack)
                d = isl_delay_s(t // 60)
                for peer in range(n_gs):
                    if peer == g:
                        continue
                    peer_sat = attach(peer, t, True)
                    other = attach(peer, t, False)
                    cands = [s for s in (peer_sat, other) if s >= 0]
                    if not cands:
                        continue
                    one_way = min(
                        (rng[peer, ps, t] + rng[g, old, t]) / C_KM_S + d[ps, old] for ps in cands
                    )
                    needed = 2.0 * one_way + args.processing_ms / 1000.0
                    checks += 1
                    if slack < needed:
                        losses += 1
                        shortfalls.append(needed - slack)

    hours = args.minutes / 60.0
    s = np.array(slacks) if slacks else np.array([0.0])
    print(f"== {c['name']} ({planes}x{slots}), {n_gs} stations, {args.minutes} min at 1 s")
    print(f"renumbering events: {events} ({events / n_gs / hours:.1f} per station per hour)")
    print(
        f"forced by the old satellite setting (slack 0): {forced} ({100 * forced / max(events, 1):.1f}%)"
    )
    print(
        f"slack after the switch: min {s.min():.0f} s, p5 {np.percentile(s, 5):.0f} s, median {np.median(s):.0f} s"
    )
    print(
        f"flow checks (event x peer): {checks}; slack < needed: {losses} ({100 * losses / max(checks, 1):.2f}%)"
    )
    if shortfalls:
        print(
            f"loss window when it happens: mean {1000 * np.mean(shortfalls):.0f} ms, max {1000 * np.max(shortfalls):.0f} ms"
        )
    print(
        f"SUMMARY,{args.config},{events},{events / n_gs / hours:.2f},{forced},"
        f"{s.min():.0f},{np.percentile(s, 5):.0f},{np.median(s):.0f},{checks},{losses},"
        f"{1000 * np.mean(shortfalls) if shortfalls else 0:.0f}"
    )


if __name__ == "__main__":
    main()
