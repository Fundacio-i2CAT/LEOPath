"""Do real satellites sit on a slot lattice with holes, or scatter?

For each plane of a CelesTrak snapshot (planes grouped as in
celestrak_shell_geometry.py), find the slot count S and phase that best explain
the satellites' arguments of latitude, by the coherence of exp(i*S*u); snap each
satellite to its nearest slot, and report the residual, the share of slots
filled, and how many satellites share a slot. A residual far below a quarter of
the slot spacing means the gaps come from empty slots, not scatter. The fitted S
is ambiguous up to multiples (S and 2S fit the same satellites); the residual is
not.

Run from ntn-paper-eval-data/revision/celestrak:  python .../celestrak_lattice_fit.py
"""

import math
import statistics
import sys
import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from celestrak_shell_geometry import load, mean_altitude_km, node_and_latitude, cluster_ring
from sgp4.api import jday
from datetime import datetime, timezone


def fit_plane(us, s_range):
    """Best (S, phi) by circular mean of phases; returns S, residuals (deg), occupancy."""
    best = None
    us = np.radians(np.array(us))
    for S in s_range:
        z = np.exp(1j * S * us).mean()  # phase coherence at S-fold symmetry
        phi = np.angle(z) / S
        slot = np.round((us - phi) * S / (2 * np.pi))
        resid = np.degrees(np.abs((us - phi) - slot * 2 * np.pi / S))
        occ = len(set((slot % S).astype(int))) / S
        score = abs(z)
        if best is None or score > best[0] + 1e-9:
            best = (score, S, resid, occ, len(us) - len(set((slot % S).astype(int))))
    return best


def run(tle, inc, alt, inc_tol, alt_tol, s_range):
    now = datetime.now(timezone.utc)
    jd, fr = jday(now.year, now.month, now.day, now.hour, now.minute, now.second)
    m = []
    for name, s in load(tle):
        if abs(math.degrees(s.inclo) - inc) > inc_tol or abs(mean_altitude_km(s) - alt) > alt_tol:
            continue
        e, p, v = s.sgp4(jd, fr)
        if e:
            continue
        raan, u = node_and_latitude(np.array(p), np.array(v))
        m.append((raan, u))
    planes = [p for p in cluster_ring([x[0] for x in m], 1.0) if len(p) >= 8]
    Ss, res, occ, doubles, coh = [], [], [], 0, []
    for p in planes:
        score, S, r, o, d = fit_plane([m[i][1] for i in p], s_range)
        Ss.append(S)
        res.extend(r)
        occ.append(o)
        doubles += d
        coh.append(score)
    radius = 6378.135 + alt
    r = np.array(res)
    print(
        f"{tle} {inc}°@{alt}km: {len(planes)} planes (>=8 sats) | fitted S: {statistics.mode(Ss)} (mode), range {min(Ss)}-{max(Ss)}"
        f" | slot occupancy median {statistics.median(occ)*100:.0f}% | sats sharing a slot {doubles}"
        f" | residual to slot p50 {np.percentile(r,50):.2f}° p90 {np.percentile(r,90):.2f}° (= {np.radians(np.percentile(r,50))*radius:.0f} / {np.radians(np.percentile(r,90))*radius:.0f} km along track)"
        f" | half-slot {180/statistics.mode(Ss):.2f}°"
    )


run("starlink_20260929.tle", 43, 480, 0.5, 10, range(20, 130))
run("starlink_20260929.tle", 53, 460, 0.5, 4, range(20, 130))
run("starlink_20260929.tle", 97.6, 470, 0.8, 10, range(20, 130))
run("oneweb_20260929.tle", 87.9, 1200, 0.5, 30, range(30, 70))
