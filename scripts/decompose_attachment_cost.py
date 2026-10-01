"""Where does the attachment-addressing delay come from?

For every ordered station pair and snapshot, compare the satellites a policy
attaches to with the best pair (any visible satellite at each end), and split
the extra delay by how their pass directions relate:

  match       chosen directions equal the best pair's at both ends
  src_only    only the source end is on the other half
  dst_only    only the destination end is on the other half
  both        both ends are on the other half

Within "match", the remaining cost is the plain cost of picking an endpoint
without knowing the route; the script reports how many grid steps (planes,
slots) separate the chosen satellite from the best one at each end.
"""

from __future__ import annotations

import collections
import math
import sys

import networkx as nx
import numpy as np
import yaml

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from leopath.topology.walker_geometry import walker_shell_from_config  # noqa: E402

C_KM_MS = 299.792458
EARTH_R_KM = 6378.135


def ring(a: int, b: int, n: int) -> int:
    d = abs(a - b) % n
    return min(d, n - d)


def run(name: str, snapshots: int = 6, step_min: float = 10.0) -> None:
    cfg = yaml.safe_load(open(f"leopath/config/{name}.yaml"))
    shell = walker_shell_from_config(cfg["constellation"])
    P, S = shell.planes, shell.sats_per_plane
    alt = cfg["satellite"]["altitude_m"] / 1000.0
    cone = math.radians(cfg["satellite"]["cone_angle_degrees"])
    max_gsl = math.hypot(alt / math.tan(cone), alt)
    gs_payload = yaml.safe_load(open("leopath/config/ground_stations_dense.yaml"))
    gss = gs_payload["ground_stations"] if isinstance(gs_payload, dict) else gs_payload
    wrap = shell.raan_spread_deg >= 360.0
    a = shell.semi_major_axis_m / 1000.0

    buckets = {pol: collections.defaultdict(list) for pol in ("nearest", "ascending")}
    steps = {pol: [] for pol in ("nearest", "ascending")}
    end_cost = {pol: {"src": [], "dst": []} for pol in ("nearest", "ascending")}
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
        theta = math.radians((280.46061837 + 360.98564736629 * (-0.5 + t / 86400.0)) % 360.0)
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
            vis.append(sorted((float(d[i]), int(i)) for i in np.flatnonzero(d <= max_gsl)))
        sp = {}

        def dist(u, v):
            if u not in sp:
                sp[u] = nx.single_source_dijkstra_path_length(g, u, weight="weight")
            return sp[u].get(v, math.inf)

        def pick(v, pol):
            if pol == "ascending":
                up = [x for x in v if asc[x[1]]]
                return (up or v)[0]
            return v[0]

        for si, sv in enumerate(vis):
            for di, dv in enumerate(vis):
                if si == di or not sv or not dv:
                    continue
                best, bs, bd = math.inf, None, None
                for g1, s1 in sv:
                    for g2, s2 in dv:
                        c = g1 + dist(s1, s2) + g2
                        if c < best:
                            best, bs, bd = c, (g1, s1), (g2, s2)
                if not math.isfinite(best):
                    continue
                for pol in ("nearest", "ascending"):
                    cs, cd = pick(sv, pol), pick(dv, pol)
                    extra = (cs[0] + dist(cs[1], cd[1]) + cd[0] - best) / C_KM_MS
                    flip_s = asc[cs[1]] != asc[bs[1]]
                    flip_d = asc[cd[1]] != asc[bd[1]]
                    key = {
                        (False, False): "match",
                        (True, False): "src_only",
                        (False, True): "dst_only",
                        (True, True): "both",
                    }[(flip_s, flip_d)]
                    buckets[pol][key].append(extra)
                    # Cost of each end alone: fix that end at the chosen satellite, free the other.
                    src_fixed = min(cs[0] + dist(cs[1], s2) + g2 for g2, s2 in dv)
                    dst_fixed = min(g1 + dist(s1, cd[1]) + cd[0] for g1, s1 in sv)
                    end_cost[pol]["src"].append((src_fixed - best) / C_KM_MS)
                    end_cost[pol]["dst"].append((dst_fixed - best) / C_KM_MS)
                    if key == "match":
                        for chosen, opt in ((cs[1], bs[1]), (cd[1], bd[1])):
                            steps[pol].append(
                                (ring(chosen // S, opt // S, P), ring(chosen % S, opt % S, S))
                            )

    print(f"== {cfg['constellation']['name']} ({P}x{S}, {'delta' if wrap else 'star'})")
    for pol in ("nearest", "ascending"):
        total = sum(len(v) for v in buckets[pol].values())
        allx = [x for v in buckets[pol].values() for x in v]
        print(f"  policy {pol:9s}: extra mean {np.mean(allx):5.1f} ms")
        for key in ("match", "src_only", "dst_only", "both"):
            v = buckets[pol][key]
            if v:
                share_cost = sum(v) / sum(allx) * 100
                print(
                    f"     {key:9s} pairs {len(v) / total * 100:5.1f}%   extra mean {np.mean(v):6.1f} ms  p95 {np.percentile(v, 95):6.1f}   share of total cost {share_cost:5.1f}%"
                )
        e = end_cost[pol]
        print(
            f"     cost of each end alone (other end free): source {np.mean(e['src']):5.1f} ms, destination {np.mean(e['dst']):5.1f} ms"
        )
        st = np.array(steps[pol])
        if len(st):
            print(
                f"     in 'match' pairs, chosen vs best satellite at one end: planes apart mean {st[:, 0].mean():.2f} (max {st[:, 0].max()}), slots apart mean {st[:, 1].mean():.2f} (max {st[:, 1].max()}); identical {np.mean((st[:, 0] == 0) & (st[:, 1] == 0)) * 100:.0f}%"
            )


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
