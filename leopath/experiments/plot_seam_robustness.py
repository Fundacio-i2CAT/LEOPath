"""Seam-robustness visualizations.

Produces two figures:

1. ``seam_robustness_schematic.png`` -- a conceptual drawing of the +Grid logical
   topology unrolled along the orbital-plane axis. The cross-seam inter-plane
   wrap (last plane back to first) is physically infeasible because the planes
   counter-rotate, so it is removed, turning the torus into a cylinder with an
   open boundary. The pivot-weighted metric, built from the measured ISL
   geometry, routes around that open boundary and delivers; the hop-based DRA
   metric assumes the cyclic wrap and loops at the seam.

2. ``seam_robustness_delivery.png`` -- the measured fraction of the 552 ordered
   ground-station pairs delivered on the cylinder model, pivot vs DRA, per
   constellation, read from the ``*/grid_seam`` runs of the evaluation dataset.

Usage::

    python -m leopath.experiments.plot_seam_robustness \
        --eval-data /path/to/ntn-paper-eval-data \
        --output-dir /path/to/output
"""

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ORDERED_PAIRS = 552
CONSTELLATIONS = ["starlink", "kuiper", "telesat", "oneweb"]
CONSTELLATION_LABELS = {
    "starlink": "Starlink",
    "kuiper": "Kuiper",
    "telesat": "Telesat",
    "oneweb": "OneWeb",
}
PIVOT_COLOR = "#1b7837"  # green
DRA_COLOR = "#b2182b"  # red


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "lines.linewidth": 2.4,
        }
    )


def seam_delivery_percent(eval_data: Path, algorithm: str, constellation: str) -> float:
    """Mean fraction (%) of the 552 ordered pairs delivered on the cylinder run."""
    csv_path = eval_data / constellation / algorithm / "grid_seam" / "timestep_metrics.csv"
    if not csv_path.exists():
        return float("nan")
    with open(csv_path) as handle:
        counts = [float(row["stretch_dist_count"]) for row in csv.DictReader(handle)]
    if not counts:
        return float("nan")
    return statistics.fmean(counts) / ORDERED_PAIRS * 100.0


def plot_delivery(eval_data: Path, output_dir: Path) -> None:
    pivot = [seam_delivery_percent(eval_data, "topological_routing", c) for c in CONSTELLATIONS]
    dra = [seam_delivery_percent(eval_data, "dra_routing", c) for c in CONSTELLATIONS]

    x = range(len(CONSTELLATIONS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_pivot = ax.bar(
        [i - width / 2 for i in x], pivot, width,
        label="Topological routing with pivot (ours)", color=PIVOT_COLOR
    )
    bars_dra = ax.bar(
        [i + width / 2 for i in x], dra, width, label=r"Hop-only ($\delta_{hop}$)", color=DRA_COLOR
    )
    for bars in (bars_pivot, bars_dra):
        for rect in bars:
            h = rect.get_height()
            if h == h:  # not NaN
                ax.annotate(
                    f"{h:.1f}",
                    (rect.get_x() + rect.get_width() / 2, h),
                    ha="center",
                    va="bottom",
                    fontsize=11,
                )

    ax.set_xticks(list(x))
    ax.set_xticklabels([CONSTELLATION_LABELS[c] for c in CONSTELLATIONS])
    ax.set_ylabel("Seam delivery (% of 552 pairs)")
    ax.set_ylim(0, 109)
    ax.set_title("Delivery on the seam (cylinder) model: cross-seam ISLs removed")
    ax.axhline(100, color="grey", linestyle=":", linewidth=1)
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = output_dir / "seam_robustness_delivery.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def plot_schematic(output_dir: Path) -> None:
    """Unrolled +Grid with the cross-seam wrap removed; pivot delivers, DRA loops."""
    n_planes, n_slots = 6, 4
    fig, ax = plt.subplots(figsize=(9, 5))

    # node coordinates
    def pos(p, s):
        return (p * 1.6, s * 1.2)

    # intra-plane links (vertical within a plane) and inter-plane links (horizontal)
    for p in range(n_planes):
        for s in range(n_slots):
            x0, y0 = pos(p, s)
            # intra-plane (slot+1, cyclic within plane -- always present)
            x1, y1 = pos(p, (s + 1) % n_slots)
            if (s + 1) < n_slots:
                ax.plot([x0, x1], [y0, y1], color="#bbbbbb", lw=1.2, zorder=1)
            # inter-plane to next plane (present except across the seam)
            if p + 1 < n_planes:
                x2, y2 = pos(p + 1, s)
                ax.plot([x0, x2], [y0, y2], color="#bbbbbb", lw=1.2, zorder=1)

    # the removed cross-seam wrap: last plane -> first plane (drawn as dashed + cut)
    for s in range(n_slots):
        xL, yL = pos(n_planes - 1, s)
        # curved "would-be" wrap shown going off the right edge
        ax.annotate(
            "",
            xy=(xL + 0.9, yL),
            xytext=(xL, yL),
            arrowprops=dict(arrowstyle="-", linestyle=(0, (4, 3)), color=DRA_COLOR, lw=1.6),
            zorder=1,
        )
    ax.text(
        pos(n_planes - 1, n_slots - 1)[0] + 0.55,
        pos(0, n_slots - 1)[1] + 0.65,
        "cross-seam wrap\nremoved ✂ (counter-rotating planes)",
        color=DRA_COLOR,
        fontsize=11,
        ha="center",
        va="bottom",
    )

    # nodes
    for p in range(n_planes):
        for s in range(n_slots):
            x0, y0 = pos(p, s)
            ax.scatter([x0], [y0], s=180, color="#2166ac", zorder=3, edgecolors="white")

    # source and destination near opposite sides of the seam
    src = (n_planes - 1, 1)  # last plane
    dst = (0, 2)  # first plane
    sx, sy = pos(*src)
    dx, dy = pos(*dst)
    ax.scatter([sx], [sy], s=320, color="#000000", zorder=4)
    ax.scatter([dx], [dy], s=320, marker="*", color="#000000", zorder=4)
    ax.annotate("src", (sx, sy), textcoords="offset points", xytext=(6, 8), fontsize=12)
    ax.annotate("dst", (dx, dy), textcoords="offset points", xytext=(6, 8), fontsize=12)

    # DRA path: heads toward the (absent) seam wrap and loops there -> fail
    dra_x = [sx, sx + 0.9]
    dra_y = [sy, sy]
    ax.plot(dra_x, dra_y, color=DRA_COLOR, lw=3, zorder=2)
    ax.scatter([sx + 0.9], [sy], marker="x", s=160, color=DRA_COLOR, zorder=5, linewidths=3)
    ax.text(
        sx + 0.5,
        sy - 0.55,
        "Hop-only: assumes wrap,\nloops at seam ✗",
        color=DRA_COLOR,
        fontsize=11,
        ha="center",
        va="top",
    )

    # pivot path: routes back around the open cylinder (along planes) to dst -> success
    pivot_nodes = [src, (n_planes - 2, 1), (n_planes - 3, 2), (2, 2), (1, 2), dst]
    px = [pos(p, s)[0] for p, s in pivot_nodes]
    py = [pos(p, s)[1] for p, s in pivot_nodes]
    ax.plot(px, py, color=PIVOT_COLOR, lw=3, zorder=2)
    ax.scatter([dx], [dy], marker="o", s=120, facecolors="none", edgecolors=PIVOT_COLOR,
               zorder=5, linewidths=3)
    ax.text(
        pos(2, 2)[0],
        pos(0, 2)[1] + 0.45,
        "Pivot: routes around open boundary ✓",
        color=PIVOT_COLOR,
        fontsize=11,
        ha="center",
        va="bottom",
    )

    ax.set_xlabel("orbital-plane axis (open after seam removal)")
    ax.set_ylabel("satellite-slot axis (cyclic)")
    # Title intentionally omitted: the paper figure caption carries the description.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.8, n_planes * 1.6 + 0.6)
    ax.set_ylim(-1.2, n_slots * 1.2 + 0.4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        mpatches.Patch(color=PIVOT_COLOR, label="Pivot (delivers)"),
        mpatches.Patch(color=DRA_COLOR, label=r"Hop-only $\delta_{hop}$ (loops / removed wrap)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left")
    fig.tight_layout()
    out = output_dir / "seam_robustness_schematic.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot seam-robustness figures")
    parser.add_argument(
        "--eval-data",
        type=Path,
        required=True,
        help="Path to ntn-paper-eval-data (for the */grid_seam delivery runs)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    apply_style()
    plot_schematic(args.output_dir)
    plot_delivery(args.eval_data, args.output_dir)


if __name__ == "__main__":
    main()
