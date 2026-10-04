"""Is the attachment-addressing delay penalty caused by ascending/descending passes?

On a Walker delta shell a ground station sees satellites on northbound
(ascending) and southbound (descending) passes; in the +Grid's logical torus
those sit roughly half the shell apart. If a station's nearest satellite is on
the other pass from the one the route arrives on, the path must cross half the
torus to reach it.
"""

from __future__ import annotations

import math
import sys

import networkx as nx
import numpy as np
import yaml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from leopath.topology.walker_geometry import walker_shell_from_config  # noqa: E402

C_KM_MS = 299.792458
EARTH_R_KM = 6378.135


def gmst_rad(days_since_j2000: float) -> float:
    return math.radians((280.46061837 + 360.98564736629 * days_since_j2000) % 360.0)


def run(config_path: str, gs_path: str, snapshots: int, step_min: float) -> None:
    cfg = yaml.safe_load(open(config_path))
    shell = walker_shell_from_config(cfg["constellation"])
    P, S = shell.planes, shell.sats_per_plane
    alt = cfg["satellite"]["altitude_m"] / 1000.0
    cone = math.radians(cfg["satellite"]["cone_angle_degrees"])
    max_gsl = math.hypot(alt / math.tan(cone), alt)
    gs_payload = yaml.safe_load(open(gs_path))
    gss = gs_payload["ground_stations"] if isinstance(gs_payload, dict) else gs_payload
    wrap = shell.raan_spread_deg >= 360.0
    a = shell.semi_major_axis_m / 1000.0

    stats = {"same": [], "opposite": [], "policy_nearest": [], "policy_ascending": []}
    for k in range(snapshots):
        t = k * step_min * 60.0
        pos = np.array(
            [[c * a for c in shell.unit_position(p, s, t)] for p in range(P) for s in range(S)]
        )
        asc = np.array(
            [
                math.cos(shell.argument_of_latitude_rad(p, s, t)) > 0
                for p in range(P)
                for s in range(S)
            ]
        )
        g = nx.Graph()
        for p in range(P):
            for s in range(S):
                i = p * S + s
                j = p * S + (s + 1) % S
                g.add_edge(i, j, weight=float(np.linalg.norm(pos[i] - pos[j])))
                if p + 1 < P or wrap:
                    j = ((p + 1) % P) * S + s
                    g.add_edge(i, j, weight=float(np.linalg.norm(pos[i] - pos[j])))
        # Ground stations in the inertial frame at time t (epoch 2000-01-01 00:00 UTC).
        theta = gmst_rad(-0.5 + t / 86400.0)
        vis = []
        for gs in gss:
            lat, lon = math.radians(float(gs["latitude"])), math.radians(float(gs["longitude"]))
            x = EARTH_R_KM * np.array(
                [
                    math.cos(lat) * math.cos(lon + theta),
                    math.cos(lat) * math.sin(lon + theta),
                    math.sin(lat),
                ]
            )
            d = np.linalg.norm(pos - x, axis=1)
            idx = np.flatnonzero(d <= max_gsl)
            vis.append(sorted((float(d[i]), int(i)) for i in idx))
        sp = {}

        def dist(u, v):
            if u not in sp:
                sp[u] = nx.single_source_dijkstra_path_length(g, u, weight="weight")
            return sp[u].get(v, math.inf)

        def pick(v, policy):
            if not v:
                return None
            if policy == "ascending":
                up = [x for x in v if asc[x[1]]]
                return (up or v)[0]
            return v[0]

        for si, sv in enumerate(vis):
            for di, dv in enumerate(vis):
                if si == di or not sv or not dv:
                    continue
                best = min(gs1 + dist(s1, s2) + gs2 for gs1, s1 in sv for gs2, s2 in dv)
                if not math.isfinite(best):
                    continue
                for policy in ("nearest", "ascending"):
                    (g1, a1), (g2, a2) = pick(sv, policy), pick(dv, policy)
                    extra = (g1 + dist(a1, a2) + g2 - best) / C_KM_MS
                    stats["policy_" + policy].append(extra)
                    if policy == "nearest":
                        stats["same" if asc[a1] == asc[a2] else "opposite"].append(extra)

    def line(name, v):
        v = np.array(v)
        print(
            f"  {name:22s} pairs {len(v):6d}  extra delay mean {v.mean():6.2f} ms  p95 {np.percentile(v, 95):6.2f} ms  max {v.max():6.2f} ms"
        )

    print(
        f"{cfg['constellation']['name']}  ({P}x{S}, {'delta' if wrap else 'star'}), {snapshots} snapshots, extra over the any-satellite optimum"
    )
    print(
        " nearest attachment at both ends, split by pass direction of the two attached satellites:"
    )
    line("same direction", stats["same"])
    line("opposite directions", stats["opposite"])
    print(
        f"  share of pairs with opposite directions: {len(stats['opposite']) / (len(stats['same']) + len(stats['opposite'])) * 100:.1f}%"
    )
    print(" policy comparison, both ends:")
    line("nearest", stats["policy_nearest"])
    line("nearest ascending", stats["policy_ascending"])


if __name__ == "__main__":
    for name in sys.argv[1:]:
        run(f"leopath/config/{name}.yaml", "leopath/config/ground_stations_dense.yaml", 6, 10.0)
