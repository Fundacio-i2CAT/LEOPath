"""Publish viewer-ready data alongside evaluation CSVs, with explicit provenance.

Replay routes are controlled Python demonstrations, not reconstructed CSV walks.
Regenerate scripts/export_viewer_replay.py first when shell/policy inputs change.
"""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import yaml

from viewer_dataset import REPOSITORY, digest, git, validate

ROOT = Path(__file__).resolve().parents[1]


def export(dataset, output):
    source = ROOT / "docs/cesium"
    config = json.loads((source / "constellations.json").read_text())
    shells = [s for s in config["constellations"] if s["id"] != "dense"]
    config["constellations"] = shells
    manifest = json.loads((source / "replays/manifest.json").read_text())
    if output.exists():
        raise ValueError(
            "Output already exists; choose a fresh directory and review it before replacing viewer/"
        )
    output.mkdir(parents=True)
    (output / "data").mkdir()
    (output / "replays").mkdir()
    (output / "inputs").mkdir()
    fingerprints = {}
    for shell in shells:
        sid = shell["id"]
        rel = f"{sid}/topological_routing/grid/metadata.json"
        metadata_path = dataset / rel
        meta = json.loads(metadata_path.read_text())
        geometry = meta["constellation"]
        if (
            geometry["num_orbits"],
            geometry["num_sats_per_orbit"],
            geometry["altitude_m"],
            geometry["inclination_degree"],
            meta["raan_spread_degree"],
        ) != (
            shell["orbits"],
            shell["satsPerOrbit"],
            shell["altitudeKm"] * 1000,
            shell["inclinationDeg"],
            shell.get("raanSpreadDeg", 360),
        ):
            raise ValueError(
                f"{sid}: evaluation shell differs from viewer; regenerate its replay/config before exporting"
            )
        shell.pop("rawTleUrl", None)
        shell["requireBundledTle"] = True
        tle = source / "replays" / f"{sid}-grid.tles.txt"
        shutil.copy2(tle, output / shell["tlePath"])
        shutil.copy2(metadata_path, output / "inputs" / f"{sid}-metadata.json")
        fingerprints[rel] = digest(metadata_path)
    for entry in manifest:
        path = source / "replays" / entry["path"]
        replay = json.loads(path.read_text())
        sid = entry["id"].split("-", 1)[0]
        shell = next(s for s in shells if s["id"] == sid)
        replay_config = replay["provenance"]["config"]
        c = replay_config["constellation"]
        if (
            c["num_orbits"],
            c["num_sats_per_orbit"],
            c["inclination_degree"],
            c["mean_motion_rev_per_day"],
            replay_config["satellite"]["altitude_m"],
            c.get("raan_spread_degree", 360),
        ) != (
            shell["orbits"],
            shell["satsPerOrbit"],
            shell["inclinationDeg"],
            shell["meanMotionRevPerDay"],
            shell["altitudeKm"] * 1000,
            shell.get("raanSpreadDeg", 360),
        ):
            raise ValueError(f"{entry['id']}: replay config differs from live shell")
        shutil.copy2(path, output / "replays" / entry["path"])
        shutil.copy2(
            path.with_suffix(".tles.txt"),
            output / "replays" / path.with_suffix(".tles.txt").name,
        )
    (output / "constellations.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "replays/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    bundle = {
        "schemaVersion": 1,
        "kind": "leopath-viewer",
        "scope": "Synthetic single-shell models and controlled Python failure demonstrations. Evaluation CSVs remain separate; these are not recorded walks from the paper campaigns.",
        "simulator": {
            "repository": "Fundacio-i2CAT/LEOPath",
            "commit": git(ROOT, "rev-parse", "HEAD"),
            "exporterSha256": digest(Path(__file__)),
            "replayExporterSha256": digest(ROOT / "scripts/export_viewer_replay.py"),
            "forwardingSha256": digest(
                ROOT
                / "leopath/network_state/routing_algorithms/topological_routing/fstate_calculation.py"
            ),
        },
        "evaluationInputs": {
            "repository": REPOSITORY,
            "baseCommit": git(dataset, "rev-parse", "HEAD"),
            "files": fingerprints,
        },
        "files": {
            p.relative_to(output).as_posix(): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
    }
    (output / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")
    validate(output)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/verify_viewer_replays.py"),
            str(output / "replays"),
        ],
        check=True,
    )
    print(f"Prepared {len(shells)} shells, {len(manifest)} replays at {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    export(args.dataset.resolve(), args.output.resolve())
