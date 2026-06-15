#!/usr/bin/env python3
"""Aggregate the sampling-interval sensitivity sweep (#7).

For each (family, constellation, interval) it summarizes the sampling-sensitive
metrics across the 6h run: forwarding-state size, path stretch, and the
per-snapshot satellite forwarding-state update burden. The update burden is also
reported as a per-hour rate so that intervals are compared on equal footing.
"""
import csv
import statistics
from pathlib import Path

STAGING = Path("/home/sergio/phd/eval-data-staging")
SWEEP = STAGING / "sampling-sweep-6h"
PIVOT_1MIN = STAGING / "topological-pivot-runs-6h"
LS_1MIN = Path("/home/sergio/phd/ntn-paper-eval-data")

CONSTS = ["telesat", "oneweb", "kuiper", "starlink"]
INTERVAL_MIN = {"10s": 1.0 / 6.0, "1min": 1.0, "5min": 5.0}


def col(path, name):
    if not path.exists():
        return None
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    vals = [float(r[name]) for r in rows if r.get(name) not in (None, "", "nan")]
    return vals or None


def mean(vals):
    return statistics.fmean(vals) if vals else float("nan")


def summarize(run_dir, interval_label):
    ts = run_dir / "timestep_metrics.csv"
    dl = run_dir / "delta_metrics.csv"
    if not ts.exists():
        return None
    fstate = col(ts, "fstate_size_mean")
    sdist = col(ts, "stretch_dist_mean")
    sdist_max = col(ts, "stretch_dist_max")
    upd = col(dl, "sat_fstate_updates_total_mean") or [0.0]
    hand = col(dl, "gs_handover_rate") or [0.0]
    renum = col(dl, "gs_renumber_rate") or [0.0]
    snaps_per_hour = 60.0 / INTERVAL_MIN[interval_label]
    return {
        "fstate_mean": mean(fstate),
        "stretch_mean": mean(sdist),
        "stretch_max": max(sdist_max) if sdist_max else float("nan"),
        "upd_per_snap": mean(upd),
        "upd_per_hour": mean(upd) * snaps_per_hour,
        "handover_rate": mean(hand),
        "renumber_rate": mean(renum),
    }


def dirs_for(family, const):
    if family == "topological":
        return {
            "10s": SWEEP / const / "interval_10s",
            "1min": SWEEP / const / "interval_1min",
            "5min": SWEEP / const / "interval_5min",
        }
    return {
        "10s": SWEEP / const / "link_state_interval_10s",
        "1min": SWEEP / const / "link_state_interval_1min",
        "5min": SWEEP / const / "link_state_interval_5min",
    }


def main():
    for family in ["topological", "link_state"]:
        print(f"\n========== {family.upper()} ==========")
        hdr = f"{'const':9} {'intv':5} {'fstate':>8} {'str_mean':>9} {'str_max':>8} {'upd/snap':>9} {'upd/hr':>9} {'hand_rt':>8} {'renum_rt':>8}"
        print(hdr)
        for const in CONSTS:
            dd = dirs_for(family, const)
            for intv in ["10s", "1min", "5min"]:
                s = summarize(dd[intv], intv)
                if s is None:
                    print(f"{const:9} {intv:5}   (missing)")
                    continue
                print(f"{const:9} {intv:5} {s['fstate_mean']:8.2f} {s['stretch_mean']:9.4f} "
                      f"{s['stretch_max']:8.4f} {s['upd_per_snap']:9.3f} {s['upd_per_hour']:9.2f} "
                      f"{s['handover_rate']:8.4f} {s['renumber_rate']:8.4f}")


if __name__ == "__main__":
    main()
