"""Exercise release integrity, installation and failure-before-mutation guarantees."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import fetch_viewer_release as fetcher

import viewer_dataset as data


class ViewerDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.bundle = self.repo / "viewer"
        (self.bundle / "data").mkdir(parents=True)
        (self.bundle / "replays").mkdir()
        (self.bundle / "inputs").mkdir()
        stations = [
            {"name": "A", "latitude": 0, "longitude": 0, "elevationM": 0},
            {"name": "B", "latitude": 1, "longitude": 1, "elevationM": 0},
        ]
        config = {
            "constellations": [
                {
                    "id": "mini",
                    "orbits": 1,
                    "satsPerOrbit": 2,
                    "tlePath": "data/mini.txt",
                    "requireBundledTle": True,
                }
            ],
            "groundStations": stations,
        }
        self.write("constellations.json", config)
        (self.bundle / "data/mini.txt").write_text(
            "1 2\nA\n1 dummy\n2 dummy\nB\n1 dummy\n2 dummy\n"
        )
        provenance = {"forwardingSha256": "f" * 64}
        replay = {
            "id": "mini-grid",
            "schemaVersion": 1,
            "planes": 1,
            "slots": 2,
            "positions": [[0, 0, 1], [1, 1, 1]],
            "frames": [{}],
            "provenance": provenance,
            "groundStations": [
                {
                    **{k: v for k, v in s.items() if k != "elevationM"},
                    "elevation_m": s["elevationM"],
                }
                for s in stations
            ],
        }
        self.write("replays/mini-grid.json", replay)
        self.write(
            "replays/manifest.json", [{"id": "mini-grid", "path": "mini-grid.json"}]
        )
        self.write("inputs/mini-metadata.json", {"constellation": {"num_orbits": 1}})
        input_path = self.repo / "mini/topological_routing/grid/metadata.json"
        input_path.parent.mkdir(parents=True)
        input_path.write_bytes((self.bundle / "inputs/mini-metadata.json").read_bytes())
        self.metadata = {
            "schemaVersion": 1,
            "kind": "leopath-viewer",
            "scope": "controlled demonstration",
            "simulator": {"commit": "a" * 40, **provenance},
            "evaluationInputs": {
                "repository": data.REPOSITORY,
                "baseCommit": "b" * 40,
                "files": {
                    "mini/topological_routing/grid/metadata.json": data.digest(
                        input_path
                    )
                },
            },
        }
        self.seal()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        self.git("config", "user.name", "Viewer tests")
        self.git("config", "user.email", "viewer@example.invalid")
        self.git("add", ".")
        self.git("commit", "-qm", "Test dataset")
        self.git("tag", "v1.2.3")

    def git(self, *args):
        return data.git(self.repo, *args)

    def write(self, name, value):
        (self.bundle / name).write_text(json.dumps(value))

    def seal(self):
        self.metadata["files"] = {
            p.relative_to(self.bundle).as_posix(): data.digest(p)
            for p in self.bundle.rglob("*")
            if p.is_file() and p.name != "bundle.json"
        }
        self.write("bundle.json", self.metadata)

    def package(self):
        out = self.root / "release"
        data.pack(self.bundle, out, "v1.2.3")
        return out

    def test_release_round_trip_and_old_files_removed(self):
        out = self.package()
        bundle = data.unpack(
            out / data.ASSET, out / (data.ASSET + ".sha256"), self.root / "extract"
        )
        target = self.root / "site"
        (target / "data").mkdir(parents=True)
        (target / "data/stale.txt").write_text("old dataset")
        data.install(bundle, target, "v1.2.3", self.git("rev-parse", "HEAD"))
        self.assertFalse((target / "data/stale.txt").exists())
        version = data.read_json(target / "dataset.json")
        self.assertEqual(version["tag"], "v1.2.3")
        self.assertEqual(version["commit"], self.git("rev-parse", "HEAD"))
        self.assertEqual(
            (target / "data/mini.txt").read_bytes(),
            (bundle / "data/mini.txt").read_bytes(),
        )

    def test_corruption_preserves_existing_installation(self):
        out = self.package()
        bundle = data.unpack(
            out / data.ASSET, out / (data.ASSET + ".sha256"), self.root / "extract"
        )
        (bundle / "data/mini.txt").write_text("corrupted")
        target = self.root / "site"
        target.mkdir()
        sentinel = target / "dataset.json"
        sentinel.write_text("existing version")
        with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
            data.install(bundle, target)
        self.assertEqual(sentinel.read_text(), "existing version")

    def test_wrong_release_commit_rejected(self):
        out = self.package()
        bundle = data.unpack(
            out / data.ASSET, out / (data.ASSET + ".sha256"), self.root / "extract"
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            data.install(bundle, self.root / "site", "v1.2.3", "c" * 40)
        self.assertFalse((self.root / "site").exists())

    def test_changed_evaluation_inputs_require_regeneration(self):
        (self.repo / "mini/topological_routing/grid/metadata.json").write_text(
            "changed"
        )
        with self.assertRaisesRegex(ValueError, "Evaluation inputs changed"):
            self.package()

    def test_uncommitted_bundle_cannot_be_released(self):
        (self.bundle / "README.md").write_text("new README")
        with self.assertRaisesRegex(ValueError, "uncommitted"):
            self.package()

    def test_unversioned_remote_fallback_rejected(self):
        config = data.read_json(self.bundle / "constellations.json")
        config["constellations"][0]["rawTleUrl"] = "https://example.invalid/main.tle"
        self.write("constellations.json", config)
        self.seal()
        with self.assertRaisesRegex(ValueError, "unversioned"):
            data.validate(self.bundle)

    def test_unsafe_archive_entries_rejected(self):
        for name, kind in [
            ("viewer/../../escape", tarfile.REGTYPE),
            ("viewer/link", tarfile.SYMTYPE),
        ]:
            archive = self.root / "unsafe.tar.gz"
            with tarfile.open(archive, "w:gz") as package:
                item = tarfile.TarInfo(name)
                item.type = kind
                item.linkname = "/etc/passwd" if kind == tarfile.SYMTYPE else ""
                package.addfile(item)
            checksum = self.root / "checksum"
            checksum.write_text(f"{data.digest(archive)}  {data.ASSET}\n")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                data.unpack(archive, checksum, self.root / "unsafe")

    def test_bad_archive_checksum_rejected(self):
        out = self.package()
        (out / (data.ASSET + ".sha256")).write_text(f"{'0'*64}  {data.ASSET}\n")
        with self.assertRaisesRegex(ValueError, "checksum"):
            data.unpack(
                out / data.ASSET, out / (data.ASSET + ".sha256"), self.root / "extract"
            )

    def test_fetch_checks_annotated_tag_and_installs_bound_release(self):
        out = self.package()
        commit = self.git("rev-parse", "HEAD")

        def api(path):
            if "/releases/" in path:
                return {
                    "draft": False,
                    "prerelease": False,
                    "tag_name": "v1.2.3",
                    "assets": [
                        {"name": n, "size": 100}
                        for n in (data.ASSET, data.ASSET + ".sha256")
                    ],
                }
            if "/git/ref/" in path:
                return {"object": {"type": "tag", "sha": "d" * 40}}
            return {"object": {"type": "commit", "sha": commit}}

        def download(command, **kwargs):
            import shutil

            name = command[command.index("--pattern") + 1]
            destination = Path(command[command.index("--dir") + 1])
            shutil.copy2(out / name, destination / name)

        with patch.object(fetcher, "api", side_effect=api), patch.object(
            fetcher.subprocess, "run", side_effect=download
        ):
            fetcher.fetch("v1.2.3", commit, self.root / "site")
        self.assertEqual(
            data.read_json(self.root / "site/dataset.json")["commit"], commit
        )

    def test_fetch_rejects_tag_move_and_missing_asset(self):
        commit = self.git("rev-parse", "HEAD")
        release = {
            "draft": False,
            "prerelease": False,
            "tag_name": "v1.2.3",
            "assets": [],
        }
        with patch.object(
            fetcher,
            "api",
            side_effect=[release, {"object": {"type": "commit", "sha": commit}}],
        ):
            with self.assertRaisesRegex(ValueError, "differs from dispatched"):
                fetcher.fetch("v1.2.3", "c" * 40, self.root / "site")
        with patch.object(
            fetcher,
            "api",
            side_effect=[release, {"object": {"type": "commit", "sha": commit}}],
        ):
            with self.assertRaisesRegex(ValueError, "lacks"):
                fetcher.fetch("v1.2.3", commit, self.root / "site")
        self.assertFalse((self.root / "site").exists())


if __name__ == "__main__":
    unittest.main()
