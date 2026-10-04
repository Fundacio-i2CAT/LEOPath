"""Export controlled fault demonstrations using LEOPath's actual forwarding code.

All fault states share one orbital instant, so detours are caused by faults,
not by handovers. These are converged snapshots, not a packet/convergence model.
The browser never computes a route. Configs, code hashes and faults travel with
the replay; this exporter can also produce Ring, open-seam and brick-wall views.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import logging
import math
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SIMULATOR_ROOT = Path(os.environ.get("LEOPATH_SIMULATOR_ROOT", str(ROOT))).resolve()
if not (SIMULATOR_ROOT / "leopath/topology/walker_geometry.py").is_file():
    raise SystemExit(
        "Replay generation needs the simulator revision recorded in the dataset. Set LEOPATH_SIMULATOR_ROOT to that checkout; static-site builds need no simulator."
    )
sys.path.insert(0, str(SIMULATOR_ROOT))
logging.disable(logging.CRITICAL)

import networkx as nx
import yaml
from astropy import units as u

from leopath.main import calculate_link_params, setup_ground_stations, setup_tles_and_satellites
from leopath.experiments.eval_harness import select_isls, northbound_satellites, isl_wiring
from leopath.network_state.helpers import (
    _build_topologies,
    _compute_isls,
    _compute_ground_station_satellites_in_range,
)
from leopath.network_state.gsl_attachment.multihoming import select_multihoming_attachments
from leopath.network_state.routing_algorithms.flow_allocation import (
    new_flow_allocation_counters,
    update_current_addresses,
)
from leopath.network_state.routing_algorithms.topological_routing import (
    fstate_calculation as forwarding,
)
from leopath.topology.constellation import ConstellationData
from leopath.topology.walker_geometry import walker_shell_from_config
from leopath.topology.satellite.topological_network_address import (
    TopologicalNetworkAddress as Address,
)

C = 299792458.0
MODE = "torus_weighted_pivot"


def finite(value):
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def export(shell: str, topology: str, orbit_seconds: int, output: Path):
    config_path = SIMULATOR_ROOT / "leopath/config" / f"{shell}.yaml"
    config = yaml.safe_load(config_path.read_text())
    stations = yaml.safe_load(
        (SIMULATOR_ROOT / "leopath/config/ground_stations_dense.yaml").read_text()
    )
    config["ground_stations"] = stations["ground_stations"]
    # Generated TLEs belong to the replay output, never overwrite evaluation inputs.
    output.parent.mkdir(parents=True, exist_ok=True)
    config["constellation"]["tle_output_filename"] = str(output.with_suffix(".tles.txt"))
    parsed, satellites = setup_tles_and_satellites(config)
    max_gsl, max_isl = calculate_link_params(config)
    cd = ConstellationData(
        parsed["n_orbits"],
        parsed["n_sats_per_orbit"],
        parsed["epoch"],
        max_gsl,
        max_isl,
        satellites,
        walker=walker_shell_from_config(config["constellation"]),
    )
    absolute = parsed["epoch"] + orbit_seconds * u.s
    stations_objects = setup_ground_stations(config)
    topo, _ = _build_topologies(cd, stations_objects)
    edges = select_isls(cd, topology, cd.walker.raan_spread_deg)
    _compute_isls(topo, edges, absolute)
    visibility = _compute_ground_station_satellites_in_range(topo, absolute)
    n = cd.number_of_satellites
    nominal = topo.graph.subgraph(range(n)).copy()
    addresses = {
        i: Address(0, i // cd.n_sats_per_orbit, i % cd.n_sats_per_orbit, 0) for i in range(n)
    }
    derived = forwarding._derived_geometry_graph(nominal, addresses, cd.walker, orbit_seconds)
    model = forwarding._build_torus_weight_model(
        derived, addresses, cd, wiring=isl_wiring(topology)
    )
    positions = []
    for sat in satellites:
        body = sat.position.ephem_obj_manual
        body.compute(absolute.strftime("%Y/%m/%d %H:%M:%S"))
        positions.append(
            [
                round(math.degrees(float(body.sublong)), 6),
                round(math.degrees(float(body.sublat)), 6),
                round(float(body.elevation), 2),
            ]
        )

    def frame(label, detail, dead_edges=(), dead_sats=()):
        live = nominal.copy()
        live.remove_edges_from(dead_edges)
        live.remove_nodes_from(dead_sats)
        neighbor_candidates = {
            i: [
                (j, j, addresses[j], float(live.edges[i, j]["weight"]))
                for j in sorted(live.neighbors(i))
            ]
            for i in live
        }
        available = [
            [(dist, sat) for dist, sat in candidates if sat in live] for candidates in visibility
        ]
        selected, _ = select_multihoming_attachments(
            available,
            1,
            order="nearest_ascending",
            ascending=northbound_satellites(cd, orbit_seconds * 10**9),
        )
        gs = setup_ground_stations(config)
        candidates = [[(dist, sat, addresses[sat]) for dist, sat in row] for row in selected]
        counters = new_flow_allocation_counters()
        update_current_addresses(gs, candidates, "sticky_nearest", counters)
        report, exceptions = {}, {}
        routes = forwarding._build_fixed_address_routes(
            "sticky_nearest",
            gs,
            candidates,
            addresses,
            neighbor_candidates,
            cd,
            MODE,
            model,
            "progress",
            counters,
            exception_policy="grow",
            live_graph=live,
            exception_report=report,
            nominal_graph=nominal,
            exceptions_out=exceptions,
        )
        route_rows = {}
        reachable, delivered = 0, 0
        for a, src in enumerate(gs):
            for b, dst in enumerate(gs):
                if a == b:
                    continue
                item = routes.get((src.id, dst.id))
                key = f"{a}:{b}"
                row = {
                    "source": a,
                    "target": b,
                    "topological": [],
                    "linkState": [],
                    "reason": "no_visibility",
                }
                if candidates[a] and candidates[b]:
                    src_sat, dst_sat = candidates[a][0][1], candidates[b][0][1]
                    row.update(sourceSatellite=src_sat, targetSatellite=dst_sat)
                    if nx.has_path(live, src_sat, dst_sat):
                        reachable += 1
                        ls = nx.shortest_path(live, src_sat, dst_sat, weight="weight")
                        row["linkState"] = ls
                        row["reason"] = "forwarding_failure"
                        if item is not None and item["failure"] is None:
                            row["topological"] = item["satellite_path"]
                            row["reason"] = "delivered"
                            delivered += 1
                        elif item is not None:
                            row["partial"] = item["satellite_path"]
                        gsl = candidates[a][0][0] + candidates[b][0][0]
                        for name, path in (("topological", row["topological"]), ("linkState", ls)):
                            if path:
                                length = (
                                    sum(live.edges[x, y]["weight"] for x, y in zip(path, path[1:]))
                                    + gsl
                                )
                                row[name + "DelayMs"] = round(length / C * 1000, 4)
                    else:
                        row["reason"] = "partition"
                route_rows[key] = row
        # Export simulator decisions and potentials, including the rule-only walk.
        # The inspector can show why an entry overrides a blocked rule.
        decisions = {}
        for dst in sorted({row[0][1] for row in candidates if row}):
            potentials = {}
            next_hops = []
            for sat in range(n):
                decision = forwarding._fixed_rule_step(
                    sat,
                    addresses[dst],
                    addresses,
                    neighbor_candidates,
                    cd,
                    MODE,
                    model,
                    "progress",
                    potentials,
                )
                next_hops.append(decision[0] if decision else None)
            decisions[str(dst)] = {
                "potential": [potentials.get(sat) for sat in range(n)],
                "ruleNext": next_hops,
            }
        failed = {tuple(sorted(e)) for e in dead_edges}
        failed.update(tuple(sorted(e)) for s in dead_sats for e in nominal.edges(s))
        return finite(
            {
                "label": label,
                "detail": detail,
                "timeS": orbit_seconds,
                "failedLinks": sorted(failed),
                "failedSatellites": sorted(dead_sats),
                "attachments": [[sat for _, sat in row] for row in selected],
                "routes": route_rows,
                "exceptions": [[a, d, hop] for (a, d), hop in sorted(exceptions.items())],
                "decisions": decisions,
                "stats": {
                    "attempted": len(gs) * (len(gs) - 1),
                    "reachable": reachable,
                    "delivered": delivered,
                    "regionEntries": int(report.get("exception_region_entries", 0)),
                    "rawEntries": len(exceptions),
                    "exceptionSatellites": len({s for s, _ in exceptions}),
                    "unresolved": int(report.get("exception_unresolved_walks", 0)),
                    "liveLinks": live.number_of_edges(),
                    "components": nx.number_connected_components(live),
                },
                "edgeLengths": [
                    [a, b, round(data["weight"], 4)] for a, b, data in live.edges(data=True)
                ],
            }
        )

    print(f"{shell}/{topology}: baseline", flush=True)
    normal = frame(
        "Normal forwarding", "The guarded pivot rule forwards toward one fixed attachment address."
    )
    # Find a single-link fault that actually blocks the plain rule along a real flow.
    preferred = ["New York", "London", "Madrid", "Johannesburg", "Perth"]
    names = [g["name"] for g in config["ground_stations"]]
    ranked = sorted(
        normal["routes"].items(),
        key=lambda kv: (
            names[kv[1]["source"]] not in preferred,
            names[kv[1]["target"]] not in preferred,
            -len(kv[1]["topological"]),
        ),
    )
    chosen = None
    for key, route in ranked:
        path = route["topological"]
        if len(path) < 5:
            continue
        dst = route["targetSatellite"]
        potential = normal["decisions"][str(dst)]["potential"]
        for i in sorted(range(1, len(path) - 2), key=lambda i: abs(i - len(path) / 2)):
            sat, hop = path[i : i + 2]
            admissible = [
                j
                for j in nominal.neighbors(sat)
                if potential[j] is not None and (potential[j], j) < (potential[sat], sat)
            ]
            if admissible == [hop]:
                chosen = (key, tuple(sorted((sat, hop))), sat)
                break
        if chosen:
            break
    if chosen is None:
        key, route = next((k, r) for k, r in ranked if len(r["topological"]) >= 3)
        path = route["topological"]
        chosen = (key, tuple(sorted(path[1:3])), path[1])
    pair, dead_edge, outage_sat = chosen
    print(f"{shell}/{topology}: fail link {dead_edge}, pair {pair}", flush=True)
    link = frame(
        "ISL failure",
        "One link is unavailable. Installed exception entries carry the detour.",
        [dead_edge],
    )
    outage = frame(
        "Satellite outage",
        "The failed satellite and all its incident links are unavailable.",
        dead_sats=[outage_sat],
    )
    # Two opposite cuts partition a torus; one cut suffices on an open cylinder.
    p = cd.n_orbits
    boundary = p // 2
    cut_edges = [
        e
        for e in nominal.edges()
        if {e[0] // cd.n_sats_per_orbit, e[1] // cd.n_sats_per_orbit}
        in ({boundary - 1, boundary}, {p - 1, 0})
    ]
    if topology == "ring":
        # Planes are already separate; isolate the selected ingress within its
        # ring rather than pretending that nonexistent cross-plane links fail.
        cut_edges = list(nominal.edges(normal["routes"][pair]["sourceSatellite"]))
    cut = frame(
        "Network partition",
        "Fixed attachment addresses in different components cannot communicate.",
        cut_edges,
    )
    payload = {
        "schemaVersion": 1,
        "id": f"{shell}-{topology}",
        "name": config["constellation"]["name"],
        "topology": topology,
        "planes": cd.n_orbits,
        "slots": cd.n_sats_per_orbit,
        "altitudeKm": config["satellite"]["altitude_m"] / 1000,
        "epoch": absolute.isot + "Z",
        "positions": positions,
        "groundStations": config["ground_stations"],
        "defaultPair": pair,
        "focusSatellite": outage_sat,
        "frames": [
            normal,
            link,
            {
                **normal,
                "label": "Link recovered",
                "detail": "The default rule resumes; failure exceptions are removed.",
            },
            outage,
            {
                **normal,
                "label": "Satellite recovered",
                "detail": "Connectivity and rule forwarding are restored.",
            },
            cut,
            {
                **normal,
                "label": "Partition recovered",
                "detail": "Restored links reconnect the components.",
            },
        ],
        "provenance": {
            "kind": "controlled demonstration",
            "model": "converged snapshots at a fixed orbital instant",
            "attachment": "K=1, nearest_ascending, sticky_nearest",
            "geometry": "derived Walker constants; measured first hop",
            "guard": "progress",
            "exceptionPolicy": "grow",
            "config": config,
            "codeCommit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=SIMULATOR_ROOT, text=True
            ).strip(),
            "forwardingSha256": hashlib.sha256(Path(forwarding.__file__).read_bytes()).hexdigest(),
            "exporterSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "pythonVersion": sys.version,
            "dependencies": {
                name: version(name)
                for name in ("numpy", "scipy", "networkx", "ephem", "astropy", "sgp4", "PyYAML")
            },
        },
    }
    output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))
    print(f"Wrote {output} ({output.stat().st_size / 1e6:.1f} MB); demo pair {pair}", flush=True)
    topology_label = {
        "grid": "+Grid",
        "grid_seam": "Open seam",
        "ring": "Ring",
        "brick_a": "Brick wall A",
        "brick_b": "Brick wall B",
    }[topology]
    return {
        "id": payload["id"],
        "label": f"{shell.title()} · {cd.n_orbits} × {cd.n_sats_per_orbit} · {topology_label}",
        "path": output.name,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shells",
        nargs="+",
        choices=["telesat", "starlink", "kuiper", "oneweb"],
        default=["telesat"],
    )
    parser.add_argument(
        "--topology", choices=["grid", "grid_seam", "ring", "brick_a", "brick_b"], default="grid"
    )
    parser.add_argument("--orbit-seconds", type=int, default=900)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/cesium/replays")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    for shell in args.shells:
        item = export(
            shell,
            args.topology,
            args.orbit_seconds,
            args.output_dir / f"{shell}-{args.topology}.json",
        )
        manifest = [old for old in manifest if old["id"] != item["id"]] + [item]
        manifest_path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
