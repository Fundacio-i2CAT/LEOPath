"""Validate, package and install the versioned static-viewer dataset (stdlib only)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

REPOSITORY = "PhD-Sergio/data-ntn-topological-evaluation"
ASSET = "leopath-viewer.tar.gz"
SHA = re.compile(r"[0-9a-f]{40}")


def read_json(path):
    return json.loads(
        path.read_text(), parse_constant=lambda value: fail(f"Non-finite JSON: {value}")
    )


def fail(message):
    raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_path(value):
    if not isinstance(value, str) or "\\" in value:
        fail("Invalid bundle path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or str(path) != value:
        fail(f"Unsafe bundle path: {value}")
    return path


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def validate(bundle):
    """Check complete checksums and the viewer's data references before installation."""
    bundle = bundle.resolve()
    metadata = read_json(bundle / "bundle.json")
    if metadata.get("schemaVersion") != 1 or metadata.get("kind") != "leopath-viewer":
        fail("Unsupported viewer bundle schema")
    expected = metadata.get("files")
    if not isinstance(expected, dict) or not expected:
        fail("Missing bundle checksums")
    actual = set()
    for file in bundle.rglob("*"):
        if file.is_symlink():
            fail("Symlinks are not supported in viewer bundles")
        if file.is_file() and file.relative_to(bundle).as_posix() not in (
            "bundle.json",
            "release.json",
            "README.md",
        ):
            actual.add(file.relative_to(bundle).as_posix())
    if actual != set(expected):
        fail("Bundle files do not match its manifest")
    for name, checksum in expected.items():
        relative_path(name)
        if (
            not re.fullmatch(r"[0-9a-f]{64}", str(checksum))
            or digest(bundle / name) != checksum
        ):
            fail(f"Checksum mismatch: {name}")
    if not SHA.fullmatch(str(metadata.get("simulator", {}).get("commit", ""))):
        fail("Missing simulator commit")
    config = read_json(bundle / "constellations.json")
    shells = config.get("constellations", [])
    stations = config.get("groundStations", [])
    if not shells or len(stations) < 2:
        fail("Missing shells or ground stations")
    ids = set()
    for shell in shells:
        sid = shell.get("id")
        if (
            not isinstance(sid, str)
            or not re.fullmatch(r"[a-z0-9_-]+", sid)
            or sid in ids
        ):
            fail("Invalid or duplicate shell ID")
        ids.add(sid)
        if shell.get("requireBundledTle") is not True:
            fail("Released shells must require their bundled TLEs")
        tle_name = str(relative_path(shell["tlePath"]))
        if tle_name not in expected or not tle_name.startswith("data/"):
            fail(f"Missing TLE file for {sid}")
        if shell.get("rawTleUrl"):
            fail("Released data must not fall back to an unversioned TLE URL")
        lines = (bundle / tle_name).read_text().splitlines()
        planes, slots = shell["orbits"], shell["satsPerOrbit"]
        if (
            not isinstance(planes, int)
            or not isinstance(slots, int)
            or min(planes, slots) < 1
        ):
            fail("Invalid shell dimensions")
        if list(map(int, lines[0].split())) != [planes, slots]:
            fail(f"TLE dimensions differ for {sid}")
        n = planes * slots
        if len(lines) != 1 + 3 * n or any(
            not lines[2 + 3 * i].startswith("1 ")
            or not lines[3 + 3 * i].startswith("2 ")
            for i in range(n)
        ):
            fail(f"Invalid TLE records for {sid}")
    replays = read_json(bundle / "replays/manifest.json")
    if not isinstance(replays, list) or not replays:
        fail("Missing replay manifest")
    replay_ids = set()
    for entry in replays:
        replay_id = entry["id"]
        if replay_id in replay_ids:
            fail("Duplicate replay ID")
        replay_ids.add(replay_id)
        name = str(relative_path(entry["path"]))
        if "/" in name or f"replays/{name}" not in expected:
            fail("Missing replay JSON")
        replay = read_json(bundle / "replays" / name)
        if replay.get("schemaVersion") != 1 or replay.get("id") != replay_id:
            fail("Invalid replay identity")
        shell_id = next((sid for sid in ids if replay_id.startswith(sid + "-")), None)
        if shell_id is None:
            fail("Replay does not belong to a live shell")
        shell = next(s for s in shells if s["id"] == shell_id)
        if [replay["planes"], replay["slots"]] != [
            shell["orbits"],
            shell["satsPerOrbit"],
        ]:
            fail("Replay and live shell dimensions differ")
        if len(replay.get("positions", [])) != replay["planes"] * replay[
            "slots"
        ] or not replay.get("frames"):
            fail("Incomplete replay")
        converted_stations = [
            {
                "name": s["name"],
                "latitude": s["latitude"],
                "longitude": s["longitude"],
                "elevationM": s["elevation_m"],
            }
            for s in replay["groundStations"]
        ]
        if converted_stations != stations:
            fail("Live and replay ground stations differ")
    inputs = metadata.get("evaluationInputs", {})
    if inputs.get("repository") != REPOSITORY or not SHA.fullmatch(
        str(inputs.get("baseCommit", ""))
    ):
        fail("Missing evaluation input provenance")
    fingerprints = inputs.get("files", {})
    if set(fingerprints) != {
        f"{sid}/topological_routing/grid/metadata.json" for sid in ids
    }:
        fail("Each released shell must identify its evaluation metadata")
    for name, checksum in fingerprints.items():
        relative_path(name)
        sid = name.split("/", 1)[0]
        if digest(bundle / "inputs" / f"{sid}-metadata.json") != checksum:
            fail("Evaluation input snapshot checksum mismatch")
    for entry in replays:
        replay = read_json(bundle / "replays" / entry["path"])
        provenance = replay.get("provenance", {})
        for key in ("forwardingSha256",):
            if provenance.get(key) != metadata["simulator"].get(key):
                fail("Replay and bundle simulator code differ")
    return metadata


def validate_inputs(bundle, dataset):
    metadata = validate(bundle)
    for name, checksum in metadata["evaluationInputs"]["files"].items():
        if digest(dataset / name) != checksum:
            fail(
                f"Evaluation inputs changed since viewer export: {name}; regenerate viewer data"
            )


def release_info(repository, tag, commit):
    if (
        repository != REPOSITORY
        or not tag
        or len(tag) > 128
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", tag)
    ):
        fail("Invalid dataset repository or release tag")
    if not SHA.fullmatch(commit):
        fail("Dataset commit must be a full SHA")
    return {
        "schemaVersion": 1,
        "repository": repository,
        "tag": tag,
        "commit": commit,
        "url": f"https://github.com/{repository}/releases/tag/{tag}",
        "releasesUrl": f"https://github.com/{repository}/releases",
    }


def pack(bundle, output, tag):
    validate(bundle)
    repo = bundle.parent
    validate_inputs(bundle, repo)
    commit = git(repo, "rev-parse", f"refs/tags/{tag}^{{commit}}")
    if git(repo, "rev-parse", "HEAD") != commit:
        fail("Check out the dataset release tag before packaging")
    if git(repo, "status", "--porcelain", "--", bundle.name):
        fail("Viewer bundle has uncommitted changes")
    release = release_info(REPOSITORY, tag, commit)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "viewer"
        shutil.copytree(bundle, stage)
        (stage / "release.json").write_text(json.dumps(release, indent=2) + "\n")
        with tarfile.open(output / ASSET, "w:gz") as archive:
            archive.add(stage, arcname="viewer")
    (output / (ASSET + ".sha256")).write_text(f"{digest(output / ASSET)}  {ASSET}\n")
    print(f"Packaged {tag} at {commit}")


def unpack(archive, checksum, destination):
    wanted = checksum.read_text().split()
    if len(wanted) != 2 or wanted[1] != ASSET or digest(archive) != wanted[0]:
        fail("Release archive checksum mismatch")
    with tarfile.open(archive) as package:
        members = package.getmembers()
        total = 0
        names = set()
        for item in members:
            path = relative_path(item.name)
            if (
                path.parts[0] != "viewer"
                or item.name in names
                or not (item.isfile() or item.isdir())
            ):
                fail("Unsafe release archive entry")
            names.add(item.name)
            total += item.size
            if total > 250_000_000 or len(names) > 500:
                fail("Viewer archive exceeds supported size")
        for item in members:
            target = destination / item.name
            if item.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.extractfile(item) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
    return destination / "viewer"


def install(bundle, target, expected_tag=None, expected_commit=None, preview=False):
    metadata = validate(bundle)
    if preview:
        repo = bundle.parent
        commit = git(repo, "rev-parse", "HEAD")
        tag = "unreleased"
        release = {
            "schemaVersion": 1,
            "repository": REPOSITORY,
            "tag": tag,
            "commit": commit,
            "preview": True,
            "url": f"https://github.com/{REPOSITORY}",
            "releasesUrl": f"https://github.com/{REPOSITORY}/releases",
        }
    else:
        release = read_json(bundle / "release.json")
        canonical = release_info(
            release["repository"], release["tag"], release["commit"]
        )
        if (
            release != canonical
            or (expected_tag and release["tag"] != expected_tag)
            or (expected_commit and release["commit"] != expected_commit)
        ):
            fail("Release identity does not match the requested dataset")
    # Validation completes before any installed data is changed. Replace whole directories
    # so files from an older dataset cannot survive in the new site.
    for directory in ("data", "replays"):
        stage = target / (directory + ".incoming")
        if stage.exists():
            shutil.rmtree(stage)
        shutil.copytree(bundle / directory, stage)
        if (target / directory).exists():
            shutil.rmtree(target / directory)
        stage.rename(target / directory)
    shutil.copy2(bundle / "constellations.json", target / "constellations.json")
    provenance = {
        **release,
        "bundleSha256": digest(bundle / "bundle.json"),
        "simulator": metadata["simulator"],
        "scope": metadata["scope"],
    }
    (target / "dataset.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Installed dataset {release['tag']} ({release['commit'][:12]})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("bundle", type=Path)
    check.add_argument("--dataset-root", type=Path)
    package = sub.add_parser("pack")
    package.add_argument("bundle", type=Path)
    package.add_argument("--tag", required=True)
    package.add_argument("--output", required=True, type=Path)
    load = sub.add_parser("install")
    load.add_argument("bundle", type=Path)
    load.add_argument("--target", required=True, type=Path)
    load.add_argument("--preview", action="store_true")
    extract = sub.add_parser("install-release")
    extract.add_argument("archive", type=Path)
    extract.add_argument("--checksum", required=True, type=Path)
    extract.add_argument("--target", required=True, type=Path)
    extract.add_argument("--tag", required=True)
    extract.add_argument("--commit")
    args = parser.parse_args()
    try:
        if args.command == "validate":
            validate(args.bundle)
            if args.dataset_root:
                validate_inputs(args.bundle, args.dataset_root)
            print("Viewer bundle checksums and references verified")
        elif args.command == "pack":
            pack(args.bundle, args.output, args.tag)
        elif args.command == "install":
            install(args.bundle, args.target, preview=args.preview)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                bundle = unpack(args.archive, args.checksum, Path(tmp))
                install(bundle, args.target, args.tag, args.commit)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Viewer dataset error: {error}\n")


if __name__ == "__main__":
    main()
