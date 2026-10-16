"""Disposable scan tests; these do not produce scientific pilot results."""

import copy
import csv
import json
import sys
import tomllib
import unittest
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import run as runner
from plot_scan import plot_scans

BINARY = Path(sys.argv.pop(1)).resolve()


class TemperatureScan(unittest.TestCase):
    def fixture(self, start="random", scan=True):
        name = "scan-test-" + uuid.uuid4().hex
        folder = REPO / "build/scan-fixtures/inputs"
        folder.mkdir(parents=True, exist_ok=True)
        text = (REPO / "configs/local/smoke.example.toml").read_text()
        text = text.replace('root = "data/raw"', 'root = "build/scan-fixtures"')
        text = text.replace('campaign = "local-smoke"', f'campaign = "{name}"')
        text = text.replace('start = "random"', f'start = "{start}"')
        text = text.replace("equilibration_sweeps = 100", "equilibration_sweeps = 5")
        text = text.replace("measurement_sweeps = 100", "measurement_sweeps = 40")
        if start == "random":
            text = text.replace("temperature = 1.0", "temperature = 2.0")
            temperatures = "[2.0, 1.0]"
        else:
            temperatures = "[1.0, 2.0]"
        if scan:
            text += "\n[scan]\ntemperatures = " + temperatures + "\n"
        path = folder / (name + ".toml")
        path.write_text(text)
        return path

    def test_ladder_validation(self):
        config = tomllib.loads(self.fixture().read_text())
        for temperatures in (
            [],
            [2.0, 2.0],
            [2.0, 3.0],
            [2.0, -1.0],
            [2.0, float("nan")],
            [1.0, 0.5],
            [2.0, True],
        ):
            invalid = copy.deepcopy(config)
            invalid["scan"]["temperatures"] = temperatures
            with self.subTest(temperatures=temperatures):
                with self.assertRaises(ValueError):
                    runner.validate(invalid)

    def test_continuation_and_first_temperature_replay(self):
        scan = runner.run(self.fixture(), BINARY)
        single = runner.run(self.fixture(scan=False), BINARY)
        for filename in ("measurements.csv", "equilibration.csv"):
            self.assertEqual(
                (scan / "L4/temperature_0" / filename).read_bytes(),
                (single / filename).read_bytes(),
            )
        self.assertEqual(
            (scan / "L4/temperature_0/final_field.txt").read_bytes(),
            (scan / "L4/temperature_1/initial_field.txt").read_bytes(),
        )
        metadata = json.loads((scan / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "SUCCESS")
        self.assertEqual(len(metadata["scan_outputs"]), 2)
        self.assertEqual(metadata["scan_outputs"][1]["temperature"], 1.0)
        self.assertEqual(metadata["scan_outputs"][1]["measurement_count"], 40)

        output = scan.parent / "plots"
        plot_scans([scan], output, block_size=2)
        for filename in (
            "temperature_scan.png",
            "temperature_scan.pdf",
            "histories.png",
            "histories.pdf",
            "summary.csv",
            "plot_info.json",
        ):
            self.assertGreater((output / filename).stat().st_size, 0)
        with (output / "summary.csv").open() as input_file:
            rows = list(csv.DictReader(input_file))
        self.assertEqual(len(rows), 2)
        self.assertIn("binder", rows[0])
        self.assertIn("binder_error", rows[0])
        self.assertNotIn("B4", rows[0])
        self.assertEqual(rows[0]["block_count"], "20")
        with self.assertRaises(FileExistsError):
            plot_scans([scan], output, block_size=2)
        # Changing a history must invalidate it before plotting.
        with (scan / "L4/temperature_1/measurements.csv").open("a") as output_file:
            output_file.write("extra row\n")
        with self.assertRaises(ValueError):
            plot_scans([scan], scan.parent / "corrupted-plots", block_size=2)

    def size_fixture(self):
        path = self.fixture()
        text = path.read_text()
        path.write_text(text + "sizes = [4, 3]\n")
        return path

    def test_size_validation(self):
        config = tomllib.loads(self.size_fixture().read_text())
        for sizes in ([], [4, 4], [4, 2], [4, True], [4, 513], [3, 4], "4,3"):
            invalid = copy.deepcopy(config)
            invalid["scan"]["sizes"] = sizes
            with self.subTest(sizes=sizes):
                with self.assertRaises(ValueError):
                    runner.validate(invalid)
        config["sampling"]["seed"] = 2**64 - 1
        with self.assertRaises(ValueError):
            runner.validate(config)

    def test_size_scan_replay_and_plotting(self):
        folder = runner.run(self.size_fixture(), BINARY)
        replay = runner.run(self.size_fixture(), BINARY)
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(len(metadata["scan_outputs"]), 4)
        self.assertEqual([point["L"] for point in metadata["size_outputs"]], [4, 3])
        self.assertEqual(
            [point["seed"] for point in metadata["size_outputs"]], [5409001, 5409002]
        )
        self.assertNotIn("N", metadata)
        for size in (4, 3):
            size_folder = folder / ("L" + str(size))
            self.assertTrue((size_folder / "stdout.log").exists())
            initial = (
                (size_folder / "temperature_0/initial_field.txt")
                .read_text()
                .splitlines()
            )
            self.assertEqual(len(initial), size**3)
            for value in initial:
                self.assertTrue(-1 <= float(value) <= 1)
            self.assertEqual(
                (size_folder / "temperature_0/final_field.txt").read_bytes(),
                (size_folder / "temperature_1/initial_field.txt").read_bytes(),
            )
            for index in (0, 1):
                relative = f"L{size}/temperature_{index}/measurements.csv"
                self.assertEqual(
                    (folder / relative).read_bytes(), (replay / relative).read_bytes()
                )
        output = folder.parent / "size-plots"
        plot_scans([folder], output, block_size=2)
        with (output / "summary.csv").open() as input_file:
            rows = list(csv.DictReader(input_file))
        self.assertEqual([int(row["L"]) for row in rows], [4, 4, 3, 3])
        # For L=3 the fluctuation formula must use N=27, not the first size's N=64.
        point_path = folder / "L3/temperature_0/measurements.csv"
        with point_path.open() as input_file:
            measurements = list(csv.DictReader(input_file))
        m = [float(row["magnetization"]) for row in measurements]
        mean_m2 = sum(value**2 for value in m) / len(m)
        mean_abs_m = sum(abs(value) for value in m) / len(m)
        self.assertAlmostEqual(
            float(rows[2]["chi_abs"]), 27 / 2 * (mean_m2 - mean_abs_m**2)
        )

    def test_size_scan_at_one_temperature(self):
        config_path = self.size_fixture()
        config_path.write_text(config_path.read_text().replace("[2.0, 1.0]", "[2.0]"))
        folder = runner.run(config_path, BINARY)
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(len(metadata["scan_outputs"]), 2)
        self.assertEqual(metadata["scan_outputs"][0]["directory"], "L4/temperature_0")
        self.assertEqual(metadata["scan_outputs"][1]["directory"], "L3/temperature_0")

    def test_later_size_failure_preserves_completed_size(self):
        from unittest.mock import patch

        actual_run = runner.subprocess.run
        calls = 0

        def fail_second_size(*args, **kwargs):
            nonlocal calls
            if "--build-info" in args[0]:
                return actual_run(*args, **kwargs)
            calls += 1
            if calls == 2:
                return runner.subprocess.CompletedProcess(args[0], 1)
            return actual_run(*args, **kwargs)

        config_path = self.size_fixture()
        with patch.object(runner.subprocess, "run", side_effect=fail_second_size):
            with self.assertRaises(RuntimeError):
                runner.run(config_path, BINARY)
        config = tomllib.loads(config_path.read_text())
        output = config["output"]
        folder = REPO / output["root"] / output["campaign"] / output["run_id"]
        metadata = json.loads((folder / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "FAIL")
        self.assertEqual(metadata["size_outputs"][0]["status"], "SUCCESS")
        self.assertEqual(metadata["size_outputs"][1]["status"], "FAIL")
        self.assertTrue((folder / "L4/temperature_1/measurements.csv").exists())
        self.assertFalse((folder / "SUCCESS").exists())

    def test_both_ordered_signs_heat(self):
        for start in ("ordered_positive", "ordered_negative"):
            with self.subTest(start=start):
                folder = runner.run(self.fixture(start), BINARY)
                fields = (
                    (folder / "L4/temperature_0/initial_field.txt")
                    .read_text()
                    .splitlines()
                )
                expected = 7**0.5
                if start == "ordered_negative":
                    expected = -expected
                for value in fields:
                    self.assertAlmostEqual(float(value), expected)
                self.assertEqual(
                    (folder / "L4/temperature_0/final_field.txt").read_bytes(),
                    (folder / "L4/temperature_1/initial_field.txt").read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
