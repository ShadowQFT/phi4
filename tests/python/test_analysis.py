"""End-to-end analysis checks using explicitly synthetic raw fixtures only."""

import copy
import csv
import json
import os
import subprocess
import sys
import tomllib
import unittest
import uuid
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "analysis"))

from phi4_analysis.cli import analyze
from phi4_analysis.data import check_compatible, load_run, sha256
from phi4_analysis.diagnostics import blocking_curve, split_history


class Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Retain exported images for visual inspection. No Monte Carlo is invoked.
        parent = REPO / "build" / "analysis-fixtures"
        parent.mkdir(parents=True, exist_ok=True)
        cls.artifacts = parent / f"synthetic-{uuid.uuid4().hex}"
        cls.artifacts.mkdir()
        # Keep test-only font caches inside the build tree.
        os.environ.setdefault("MPLCONFIGDIR", str(parent / "matplotlib-cache"))

    def fixture(self, name, *, start="random", seed=1, count=1027):
        folder = self.artifacts / name
        folder.mkdir()
        text = (REPO / "configs/local/smoke.example.toml").read_text(encoding="utf-8")
        text = text.replace(
            'campaign = "local-smoke"', 'campaign = "synthetic-validation"'
        )
        text = text.replace(
            'run_id = "L4-T1-random-seed5409001-attempt01"', f'run_id = "{name}"'
        )
        text = text.replace('start = "random"', f'start = "{start}"')
        text = text.replace("seed = 5409001", f"seed = {seed}")
        text = text.replace("equilibration_sweeps = 100", "equilibration_sweeps = 10")
        text = text.replace("measurement_sweeps = 100", f"measurement_sweeps = {count}")
        (folder / "config.resolved.toml").write_text(text, encoding="utf-8")
        config = tomllib.loads(text)

        # Analytic histories give exact references for signs, moments, split
        # offsets, counters, and retained-tail accounting.
        x = np.arange(count)
        magnetization = 1.0 + 0.2 * np.sin(x * 0.17)
        energy = -2.0 + 0.3 * np.cos(x * 0.11)
        with (folder / "measurements.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["sample_index", "sweep", "energy_per_site", "magnetization"]
            )
            writer.writerows(
                (i, 11 + i, e, m) for i, (e, m) in enumerate(zip(energy, magnetization))
            )
        with (folder / "equilibration.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "sweep",
                    "energy_per_site",
                    "magnetization",
                    "acceptance",
                    "step_size",
                    "phase",
                ]
            )
            writer.writerows((i, -2, 1, 0.5, 1, "settle") for i in range(1, 11))
        with zipfile.ZipFile(folder / "source.zip", "w") as archive:
            archive.writestr(
                "fixture.txt", "Synthetic validation; no simulator was executed."
            )
        metadata = {
            "schema_version": 1,
            "status": "SUCCESS",
            "exit_code": 0,
            "N": 64,
            "config": config,
            "measurement_count": count,
            "final_step_size": 1,
            "dataset_kind": "synthetic_validation",
        }
        metadata["output_hashes"] = {
            name: sha256(folder / name)
            for name in (
                "measurements.csv",
                "equilibration.csv",
                "config.resolved.toml",
                "source.zip",
            )
        }
        (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        (folder / "SUCCESS").write_text("Synthetic operational fixture only.\n")
        return folder

    def rehash(self, folder, name):
        path = folder / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["output_hashes"][name] = sha256(folder / name)
        path.write_text(json.dumps(metadata))

    def test_loader_integrity_and_counters(self):
        folder = self.fixture("integrity")
        loaded = load_run(folder)
        self.assertEqual(len(loaded.energy), 1027)
        self.assertEqual(loaded.sweeps[0], 11)
        (folder / "measurements.csv").write_text(
            (folder / "measurements.csv").read_text().replace("0,11,", "0,12,", 1)
        )
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            load_run(folder)
        self.rehash(folder, "measurements.csv")
        with self.assertRaisesRegex(ValueError, "sample index or sweep"):
            load_run(folder)

    def test_status_and_equilibration_checks(self):
        folder = self.fixture("failed-status")
        (folder / "FAIL").write_text("failure")
        with self.assertRaisesRegex(ValueError, "incomplete or failed"):
            load_run(folder)
        folder = self.fixture("bad-equilibration")
        path = folder / "equilibration.csv"
        path.write_text(path.read_text().replace("0.5,1,settle", "0.5,2,settle", 1))
        self.rehash(folder, "equilibration.csv")
        with self.assertRaisesRegex(ValueError, "not frozen"):
            load_run(folder)

    def test_chain_compatibility(self):
        a = load_run(self.fixture("chain-a", seed=10))
        b = load_run(self.fixture("chain-b", seed=20, start="ordered_positive"))
        check_compatible([a, b])
        with self.assertRaisesRegex(ValueError, "same run"):
            check_compatible([a, a])
        changed = copy.deepcopy(b)
        changed.metadata["config"]["model"]["temperature"] = 2
        with self.assertRaisesRegex(ValueError, "identical model"):
            check_compatible([a, changed])
        c = copy.deepcopy(b)
        c.folder = b.folder / "another"
        with self.assertRaisesRegex(ValueError, "distinct seeds"):
            check_compatible([a, b, c])

    def test_split_history_and_block_counts(self):
        e = np.r_[np.zeros(400), np.ones(400)]
        m = np.r_[np.ones(400), np.full(400, 2)]
        result = split_history(e, m, 64, 1, 10)
        self.assertEqual(result["second_minus_first"]["e"], 1)
        self.assertEqual(result["second_minus_first"]["abs_m"], 1)
        self.assertEqual(result["second_minus_first"]["m2"], 3)
        self.assertEqual(result["second_minus_first"]["B4"], 0)
        points = blocking_curve(e, m, 64, 1)
        self.assertEqual(
            [point["block_size"] for point in points], [1, 2, 4, 8, 16, 32]
        )
        self.assertTrue(all(point["block_count"] >= 20 for point in points))

    def test_complete_analysis_outputs_and_no_raw_changes(self):
        folder = self.fixture("complete")
        ordered = self.fixture("complete-ordered", start="ordered_positive", seed=2)
        before = {p.name: sha256(p) for p in folder.iterdir()}
        output = self.artifacts / "synthetic-preview"
        analyze(
            [folder, ordered],
            output,
            block_size=8,
            bootstrap_seed=123,
            replicates=100,
            discard_samples=2,
            plots=True,
        )
        chains = json.loads((output / "summary.json").read_text())["chains"]
        self.assertEqual(len(chains), 2)
        self.assertNotEqual(
            chains[0]["block_bootstrap"]["seed"], chains[1]["block_bootstrap"]["seed"]
        )
        result = chains[0]
        self.assertEqual(result["selected_sample_count"], 1025)
        self.assertEqual(result["block_jackknife"]["dropped_tail_samples"], 1)
        self.assertEqual(result["first_selected_sweep"], 13)
        provenance = json.loads((output / "provenance.json").read_text())
        self.assertEqual(provenance["status"], "SUCCESS")
        self.assertEqual(before, {p.name: sha256(p) for p in folder.iterdir()})
        for name, checksum in provenance["output_hashes"].items():
            self.assertEqual(sha256(output / name), checksum)
        for name in ("histories.png", "histories.pdf", "blocking.png", "blocking.pdf"):
            self.assertGreater((output / name).stat().st_size, 1000)
        with self.assertRaises(FileExistsError):
            analyze(
                [folder],
                output,
                block_size=8,
                bootstrap_seed=123,
                replicates=100,
                plots=False,
            )
        with self.assertRaisesRegex(ValueError, "raw run folder"):
            analyze(
                [folder],
                folder / "analysis",
                block_size=8,
                bootstrap_seed=123,
                plots=False,
            )

    def test_failed_export_records_failure(self):
        folder = self.fixture("export-failure")
        output = self.artifacts / "failed-analysis"
        with patch(
            "phi4_analysis.cli.plot_diagnostics",
            side_effect=OSError("injected export error"),
        ):
            with self.assertRaises(OSError):
                analyze(
                    [folder], output, block_size=8, bootstrap_seed=123, replicates=100
                )
        self.assertTrue((output / "FAIL").exists())
        self.assertFalse((output / "SUCCESS").exists())
        self.assertEqual(
            json.loads((output / "provenance.json").read_text())["status"], "FAIL"
        )

    def test_cli_and_replay(self):
        folder = self.fixture("cli")
        outputs = [self.artifacts / "cli-first", self.artifacts / "cli-second"]
        for output in outputs:
            result = subprocess.run(
                [
                    sys.executable,
                    str(REPO / "scripts/analyze.py"),
                    str(folder),
                    "--output",
                    str(output),
                    "--block-size",
                    "8",
                    "--bootstrap-seed",
                    "777",
                    "--replicates",
                    "100",
                    "--no-plots",
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("summary.json", "observables.csv", "blocking.csv"):
            self.assertEqual(
                (outputs[0] / name).read_bytes(), (outputs[1] / name).read_bytes()
            )


if __name__ == "__main__":
    unittest.main()
