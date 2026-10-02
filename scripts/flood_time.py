"""How long a flooded failure report takes to reach every satellite of a shell.

Both routing families learn a failure the same way, by flooding it, so the
time until every satellite knows is a floor on convergence for either of them.
After that, link-state recomputes routes to every destination while the
topological scheme adds exception entries only where walks break; that part is
counted by the simulator, not here.

For each shell this builds the +Grid (the simulator's wiring, ISL shift 0)
from the seven Walker constants, weights every ISL by its propagation delay at
the speed of light, and reports, over one orbital period:

- eccentricity: how long a flood from one satellite takes to reach the last
  one, in ms (mean over satellites and snapshots), and the diameter (the worst
  case over everything);
- the same in hops, so per-hop processing time can be added on top.

Usage: python scripts/flood_time.py [--samples 12] > flood_time.csv
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leopath.topology.walker_geometry import walker_shell_from_config  # noqa: E402

SPEED_OF_LIGHT_M_S = 299_792_458.0
CONFIGS = ["telesat", "oneweb", "kuiper", "starlink"] + [
    f"shells/{p.stem}" for p in sorted((ROOT / "leopath/config/shells").glob("*.yaml"))
]


def grid_edges(planes: int, slots: int, seam: bool):
    for p in range(planes):
        for s in range(slots):
            yield (p, s), (p, (s + 1) % slots)
            if p + 1 < planes or not seam:
                yield (p, s), ((p + 1) % planes, s)


def flood_stats(shell, seam: bool, time_s: float):
    n = shell.planes * shell.sats_per_plane
    index = lambda c: c[0] * shell.sats_per_plane + c[1]  # noqa: E731
    rows, cols, delay = [], [], []
    for a, b in grid_edges(shell.planes, shell.sats_per_plane, seam):
        ms = 1e3 * shell.distance_m(a, b, time_s) / SPEED_OF_LIGHT_M_S
        rows += [index(a), index(b)]
        cols += [index(b), index(a)]
        delay += [ms, ms]
    graph = csr_matrix((delay, (rows, cols)), shape=(n, n))
    by_delay = shortest_path(graph, method="D", directed=False)
    by_hops = shortest_path(graph, method="D", directed=False, unweighted=True)
    return by_delay.max(axis=1), by_hops.max(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=12, help="snapshots over one orbit")
    args = parser.parse_args()
    print(
        "config,satellites,wiring,ecc_ms_mean,diameter_ms,ecc_hops_mean,diameter_hops,"
        "orbital_period_min,samples"
    )
    for name in CONFIGS:
        config = yaml.safe_load((ROOT / "leopath/config" / f"{name}.yaml").read_text())
        constellation = config["constellation"]
        shell = walker_shell_from_config(constellation)
        period_s = 86400.0 / shell.mean_motion_rev_per_day
        # A Walker star's first and last planes counter-rotate, so only the open
        # seam is buildable there; a delta closes the torus.
        star = shell.raan_spread_deg < 359.0
        for seam in (False, True) if not star else (True,):
            ecc_ms, ecc_hops = [], []
            for k in range(args.samples):
                ms, hops = flood_stats(shell, seam, k * period_s / args.samples)
                ecc_ms.append(ms)
                ecc_hops.append(hops)
            ecc_ms, ecc_hops = np.concatenate(ecc_ms), np.concatenate(ecc_hops)
            print(
                f"{name},{shell.planes * shell.sats_per_plane},{'grid_seam' if seam else 'grid'},"
                f"{ecc_ms.mean():.1f},{ecc_ms.max():.1f},{ecc_hops.mean():.1f},"
                f"{int(ecc_hops.max())},{period_s / 60:.1f},{args.samples}",
                flush=True,
            )


if __name__ == "__main__":
    main()
