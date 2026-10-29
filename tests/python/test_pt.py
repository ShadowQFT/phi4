"""PT integration oracles; all outputs are disposable validation fixtures."""

import copy
import csv
import json
import math
import statistics
import sys
import tomllib
import unittest
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
import mill
import run as single
import run_pt

SERIAL = Path(sys.argv.pop(1)).resolve()
SINGLE = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None
MPI = Path(sys.argv.pop(1)).resolve() if len(sys.argv) > 1 else None


class ParallelTempering(unittest.TestCase):
    def fixture(
        self,
        temperatures=(1.2, 1.7, 2.5),
        adapt=200,
        settle=300,
        measure=500,
        interval=5,
        swaps=5,
        J=1,
    ):
        name = "pt-" + uuid.uuid4().hex
        folder = REPO / "build/pt-fixtures/inputs"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / (name + ".toml")
        text = mill.config_text(
            3,
            temperatures[0],
            "random",
            78142,
            name,
            0,
            adapt,
            settle,
            measure,
            interval,
        )
        text = text.replace('root = "data/raw"', 'root = "build/pt-fixtures"').replace(
            "J = 1.0", f"J = {float(J)}"
        )
        text += (
            "\n[tempering]\ntemperatures = "
            + json.dumps(list(temperatures))
            + "\nswap_every_sweeps = "
            + str(swaps)
            + "\n"
        )
        path.write_text(text, encoding="utf-8")
        return path

    def series(self, folder):
        with (folder / "measurements.csv").open(newline="") as stream:
            reader = csv.reader(stream)
            next(reader)
            return [[float(value) for value in row] for row in reader]

    def test_replay_and_no_overwrite(self):
        path = self.fixture()
        first = run_pt.launch(path, SERIAL)
        config = tomllib.loads(path.read_text())
        with self.assertRaises(FileExistsError):
            run_pt.launch(path, SERIAL)
        second_path = self.fixture()
        text = path.read_text().replace(
            config["output"]["campaign"],
            tomllib.loads(second_path.read_text())["output"]["campaign"],
        )
        second_path.write_text(text)
        second = run_pt.launch(second_path, SERIAL)
        for name in (
            "measurements.csv",
            "equilibration.csv",
            "swaps.csv",
            "walks.csv",
            "pt_summary.json",
        ):
            self.assertEqual(
                (first / name).read_bytes(), (second / name).read_bytes(), name
            )

    def test_wrong_execution_mode_is_rejected_before_mpiexec(self):
        path = self.fixture()
        with self.assertRaisesRegex(
            ValueError, "execution_mode|execution mode|serial/MPI"
        ):
            run_pt.launch(
                path, SERIAL, "mpi", mpiexec="this-executable-must-not-be-called"
            )

    def test_invalid_ladders(self):
        path = self.fixture()
        base = tomllib.loads(path.read_text())
        for values in ([1.2, 1.2], [1.7, 1.2], [1.2, float("nan")], []):
            config = copy.deepcopy(base)
            config["tempering"]["temperatures"] = values
            with self.assertRaises(ValueError):
                run_pt.validate(config)
        config = copy.deepcopy(base)
        config["sampling"]["seed"] = 2**64 - 1
        with self.assertRaises(ValueError):
            run_pt.validate(config)

    def test_hash_and_output_corruption(self):
        path = self.fixture()
        folder = run_pt.launch(path, SERIAL)
        config = tomllib.loads(path.read_text())
        file = folder / "measurements.csv"
        text = file.read_text()
        lines = text.splitlines()
        # Duplicate one replica within an otherwise complete synchronized sample.
        row = lines[2].split(",")
        row[4] = lines[1].split(",")[4]
        lines[2] = ",".join(row)
        file.write_text("\n".join(lines) + "\n")
        with self.assertRaises(ValueError):
            run_pt.validate_outputs(folder, config)

    @unittest.skipUnless(SINGLE, "single-temperature executable not provided")
    def test_one_temperature_replays_existing_sampler(self):
        path = self.fixture((1.7,), adapt=200, settle=100, measure=300, interval=1)
        pt = run_pt.launch(path, SERIAL)
        config = tomllib.loads(path.read_text())
        config.pop("tempering")
        baseline = self.fixture((1.7,), adapt=200, settle=100, measure=300, interval=1)
        text = (
            path.read_text()
            .split("[tempering]")[0]
            .replace(
                config["output"]["campaign"],
                tomllib.loads(baseline.read_text())["output"]["campaign"],
            )
        )
        baseline.write_text(text)
        serial = single.run(baseline, SINGLE)
        with (serial / "measurements.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            e_m = [
                [float(r["energy_per_site"]), float(r["magnetization"])] for r in reader
            ]
        self.assertEqual([row[5:7] for row in self.series(pt)], e_m)

    @unittest.skipUnless(MPI, "MPI executable not provided")
    def test_mpi_replays_serial_ladder(self):
        path = self.fixture()
        first = run_pt.launch(path, SERIAL)
        second_path = self.fixture()
        text = path.read_text().replace(
            tomllib.loads(path.read_text())["output"]["campaign"],
            tomllib.loads(second_path.read_text())["output"]["campaign"],
        )
        second_path.write_text(text)
        second = run_pt.launch(second_path, MPI, "mpi")
        for name in (
            "measurements.csv",
            "equilibration.csv",
            "swaps.csv",
            "walks.csv",
            "pt_summary.json",
        ):
            self.assertEqual(
                (first / name).read_bytes(), (second / name).read_bytes(), name
            )

    def test_J0_moments_against_independent_quadrature(self):
        path = self.fixture(adapt=2000, settle=5000, measure=60000, interval=1, J=0)
        folder = run_pt.launch(path, SERIAL)
        data = self.series(folder)
        for slot, T in enumerate((1.2, 1.7, 2.5)):

            def integrals(intervals):
                total = second = fourth = 0.0
                for index in range(intervals + 1):
                    x = -8 + 16 * index / intervals
                    weight = 1 if index in (0, intervals) else (4 if index % 2 else 2)
                    density = weight * math.exp(-(-x * x / 2 + x**4 / 4) / T)
                    total += density
                    second += density * x * x
                    fourth += density * x**4
                return second / total, fourth / total

            mu2, mu4 = integrals(40000)
            for a, b in zip((mu2, mu4), integrals(80000)):
                self.assertTrue(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11))
            expected = [
                -0.5 * mu2 + 0.25 * mu4,
                mu2 / 27,
                (mu4 + 3 * 26 * mu2**2) / 27**3,
            ]
            values = [(r[5], r[6] ** 2, r[6] ** 4) for r in data[slot::3]]
            blocks = [
                [
                    statistics.fmean(r[column] for r in values[i : i + 200])
                    for column in range(3)
                ]
                for i in range(0, len(values), 200)
            ]
            for column, target in enumerate(expected):
                samples = [block[column] for block in blocks]
                mean = statistics.fmean(samples)
                error = statistics.stdev(samples) / math.sqrt(len(samples))
                self.assertLess(
                    abs(mean - target),
                    6 * error + 1e-9,
                    (T, column, mean, target, error),
                )


if __name__ == "__main__":
    unittest.main()
