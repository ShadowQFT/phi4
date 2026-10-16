"""Disposable operational fixtures only; no scientific pilot is launched."""

import copy
import importlib.util
import json
import sys
import tomllib
import unittest
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("runner", REPO / "scripts/run.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
BINARY = Path(sys.argv.pop(1)).resolve()


class Workflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tomllib.loads(
            (REPO / "configs/local/smoke.example.toml").read_text()
        )

    def test_invalid_configs(self):
        cases = [
            ("model", "L", 2),
            ("model", "L", True),
            ("model", "temperature", 0),
            ("model", "u", float("nan")),
            ("model", "J", -1),
            ("model", "extra", 2),
            ("sampling", "seed", -1),
            ("sampling", "seed", 2**64),
            ("sampling", "start", "typo"),
            ("sampling", "measurement_sweeps", 0),
            ("sampling", "measure_every_sweeps", 3),
            ("output", "run_id", "../escape"),
            ("output", "root", "../escape"),
            ("output", "save_time_series", False),
        ]
        for section, key, value in cases:
            with self.subTest(section=section, key=key, value=value):
                c = copy.deepcopy(self.base)
                c[section][key] = value
                with self.assertRaises(ValueError):
                    runner.validate(c)

    def fixture(self, run_id):
        text = (REPO / "configs/local/smoke.example.toml").read_text()
        text = text.replace('root = "data/raw"', 'root = "build/workflow-fixtures"')
        text = text.replace(
            'campaign = "local-smoke"', f'campaign = "test-{uuid.uuid4().hex}"'
        )
        text = text.replace(
            'run_id = "L4-T1-disordered-seed5409001-attempt01"', f'run_id = "{run_id}"'
        )
        text = text.replace("equilibration_sweeps = 100", "equilibration_sweeps = 10")
        text = text.replace("measurement_sweeps = 100", "measurement_sweeps = 20")
        text = text.replace("adaptation_sweeps = 0", "adaptation_sweeps = 50")
        text = text.replace("measure_every_sweeps = 1", "measure_every_sweeps = 2")
        folder = REPO / "build/workflow-fixtures/inputs"
        folder.mkdir(parents=True, exist_ok=True)
        config = folder / f"{uuid.uuid4().hex}.toml"
        config.write_text(text)
        return config

    def test_replay_and_no_overwrite(self):
        config = self.fixture("first")
        a = runner.run(config, BINARY)
        c2 = config.with_name(f"{uuid.uuid4().hex}.toml")
        c2.write_text(
            config.read_text().replace('run_id = "first"', 'run_id = "second"')
        )
        b = runner.run(c2, BINARY)
        for name in ("measurements.csv", "equilibration.csv"):
            self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
        meta = json.loads((a / "metadata.json").read_text())
        self.assertEqual(meta["status"], "SUCCESS")
        self.assertEqual(meta["sample_count"], 10)
        self.assertEqual(meta["adaptation_stop_sweep"], 50)
        import csv

        with (a / "equilibration.csv").open() as f:
            settled = [r for r in csv.DictReader(f) if r["phase"] == "settle"]
        self.assertEqual(len({r["step_size"] for r in settled}), 1)
        self.assertEqual(float(settled[0]["step_size"]), meta["final_step_size"])
        with self.assertRaises(FileExistsError):
            runner.run(config, BINARY)
        # Counts and nonfinite samples must fail even after a successful process exit.
        (b / "measurements.csv").write_text(
            "sample_index,sweep,energy_per_site,magnetization\n0,62,nan,1\n"
        )
        with self.assertRaises(ValueError):
            runner.validate_outputs(b, tomllib.loads(c2.read_text()))

    def test_failed_binary_preserves_metadata(self):
        # A real executable with a failing invocation, no synthetic success markers.
        import unittest.mock

        config = self.fixture("failure")
        with unittest.mock.patch.object(
            runner.subprocess,
            "check_output",
            side_effect=RuntimeError("injected build-info failure"),
        ):
            with self.assertRaises(RuntimeError):
                runner.run(config, BINARY)
        c = tomllib.loads(config.read_text())
        folder = (
            REPO / c["output"]["root"] / c["output"]["campaign"] / c["output"]["run_id"]
        )
        self.assertTrue((folder / "FAIL").exists())
        self.assertFalse((folder / "SUCCESS").exists())
        self.assertEqual(
            json.loads((folder / "metadata.json").read_text())["status"], "FAIL"
        )

    def test_numerical_failure(self):
        config = self.fixture("overflow")
        config.write_text(config.read_text().replace("r = -1.0", "r = -1e308"))
        with self.assertRaises(RuntimeError):
            runner.run(config, BINARY)
        c = tomllib.loads(config.read_text())
        folder = (
            REPO / c["output"]["root"] / c["output"]["campaign"] / c["output"]["run_id"]
        )
        self.assertTrue((folder / "FAIL").exists())
        self.assertNotEqual(
            json.loads((folder / "metadata.json").read_text())["exit_code"], 0
        )


if __name__ == "__main__":
    unittest.main()
