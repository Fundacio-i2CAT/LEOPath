"""Fetch a dataset release, checking its tag/commit against the bundled identity."""

import argparse
import json
import re
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import quote

from viewer_dataset import ASSET, REPOSITORY, SHA, install, unpack


def api(path):
    return json.loads(subprocess.check_output(["gh", "api", path], text=True))


def fetch(tag, commit, target):
    if tag and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", tag):
        raise ValueError("Invalid dataset release tag")
    if commit and not SHA.fullmatch(commit):
        raise ValueError("Invalid dataset commit")
    path = "releases/tags/" + quote(tag, safe="") if tag else "releases/latest"
    release = api(f"repos/{REPOSITORY}/{path}")
    if release["draft"] or release["prerelease"]:
        raise ValueError("Only published stable dataset releases are deployed")
    tag = release["tag_name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", tag):
        raise ValueError("Invalid published release tag")
    # Resolve the tag explicitly, including annotated tags; do not resolve a branch
    # that happens to share its name or trust the dispatch's commit unchecked.
    obj = api(f"repos/{REPOSITORY}/git/ref/tags/{quote(tag, safe='')}")["object"]
    for _ in range(8):
        if obj["type"] == "commit":
            break
        if obj["type"] != "tag":
            raise ValueError("Dataset tag does not point to a commit")
        obj = api(f"repos/{REPOSITORY}/git/tags/{obj['sha']}")["object"]
    else:
        raise ValueError("Too many nested annotated tags")
    if commit and obj["sha"] != commit:
        raise ValueError("Published dataset tag differs from dispatched commit")
    commit = obj["sha"]
    assets = {a["name"]: a for a in release["assets"]}
    for name in (ASSET, ASSET + ".sha256"):
        if name not in assets:
            raise ValueError(
                f"Dataset {tag} lacks {name}; publish its viewer bundle first"
            )
        if assets[name]["size"] > 100_000_000:
            raise ValueError("Viewer release asset exceeds supported size")
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for name in (ASSET, ASSET + ".sha256"):
            subprocess.run(
                [
                    "gh",
                    "release",
                    "download",
                    tag,
                    "--repo",
                    REPOSITORY,
                    "--pattern",
                    name,
                    "--dir",
                    str(stage),
                ],
                check=True,
            )
        bundle = unpack(stage / ASSET, stage / (ASSET + ".sha256"), stage / "unpacked")
        install(bundle, target, expected_tag=tag, expected_commit=commit)
    print(f"Using dataset release {tag} at {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--target", required=True, type=Path)
    args = parser.parse_args()
    try:
        fetch(args.tag, args.commit, args.target)
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Viewer release error: {error}\n")
