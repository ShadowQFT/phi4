"""Launch a serial-emulated or MPI parallel-tempering ladder, preserving raw data."""

import argparse
import csv
import json
import math
import os
import subprocess
import tomllib
from pathlib import Path

import run as single

REPO = Path(__file__).resolve().parents[1]
RAW_FILES = (
    "config.resolved.toml",
    "temperatures.txt",
    "measurements.csv",
    "equilibration.csv",
    "swaps.csv",
    "walks.csv",
    "pt_summary.json",
    "source.zip",
    "stdout.log",
    "stderr.log",
)


def validate(config):
    single.check_keys(
        config, ["schema_version", "model", "sampling", "output", "tempering"], "root"
    )
    base = {key: value for key, value in config.items() if key != "tempering"}
    single.validate(base)
    pt = config["tempering"]
    single.check_keys(pt, ["temperatures", "swap_every_sweeps"], "tempering")
    temperatures = pt["temperatures"]
    if not isinstance(temperatures, list) or not 1 <= len(temperatures) <= 256:
        raise ValueError("Require one to 256 temperatures")
    for i, T in enumerate(temperatures):
        single.check_number(T, "temperature")
        if T <= 0 or (i and T <= temperatures[i - 1]):
            raise ValueError("Temperatures must be positive and strictly increasing")
    if temperatures[0] != config["model"]["temperature"]:
        raise ValueError("model.temperature must equal first ladder temperature")
    single.check_integer(pt["swap_every_sweeps"], "swap_every_sweeps", 0)
    single.check_integer(
        config["sampling"]["seed"] + len(temperatures) - 1,
        "last replica seed",
        0,
        2**64 - 1,
    )
    return config


def validate_outputs(folder, config):
    folder = Path(folder)
    temperatures = config["tempering"]["temperatures"]
    K = len(temperatures)
    sampling = config["sampling"]
    interval = sampling["measure_every_sweeps"]
    adapt = sampling["adaptation_sweeps"]
    settle = sampling["equilibration_sweeps"]
    n = sampling["measurement_sweeps"] // interval
    summary = json.loads((folder / "pt_summary.json").read_text())
    if summary["replicas"] != K or summary["measurement_count_per_temperature"] != n:
        raise ValueError("PT sample count mismatch")
    steps = summary["step_by_temperature"]
    if len(steps) != K or any(not math.isfinite(x) or x <= 0 for x in steps):
        raise ValueError("Invalid frozen step sizes")
    rows = 0
    identities = []
    with (folder / "measurements.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [
            "sample_index",
            "sweep",
            "temperature_index",
            "temperature",
            "replica_id",
            "energy_per_site",
            "magnetization",
        ]:
            raise ValueError("Unexpected PT measurement header")
        for row in reader:
            sample, slot = divmod(rows, K)
            if (
                int(row["sample_index"]) != sample
                or int(row["temperature_index"]) != slot
                or float(row["temperature"]) != temperatures[slot]
                or int(row["sweep"]) != adapt + settle + (sample + 1) * interval
            ):
                raise ValueError("Non-synchronous or misrouted PT measurement")
            if not all(
                math.isfinite(float(row[key]))
                for key in ("energy_per_site", "magnetization")
            ):
                raise ValueError("Nonfinite PT measurement")
            identities.append(int(row["replica_id"]))
            if slot == K - 1:
                if sorted(identities) != list(range(K)):
                    raise ValueError("Duplicate/missing replica in synchronized sample")
                identities = []
            rows += 1
    if rows != n * K:
        raise ValueError("Truncated PT measurements")
    expected = []
    for phase, length, offset in (("adapt", adapt, 0), ("settle", settle, adapt)):
        ticks = list(range(interval, length + 1, interval))
        if length and (not ticks or ticks[-1] != length):
            ticks.append(length)
        for tick in ticks:
            expected.extend((offset + tick, phase, slot) for slot in range(K))
    eq_count = 0
    with (folder / "equilibration.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != [
            "sweep",
            "phase",
            "temperature_index",
            "temperature",
            "replica_id",
            "energy_per_site",
            "magnetization",
            "step_size",
        ]:
            raise ValueError("Unexpected PT equilibration header")
        for row in reader:
            if (
                eq_count >= len(expected)
                or (int(row["sweep"]), row["phase"], int(row["temperature_index"]))
                != expected[eq_count]
            ):
                raise ValueError("PT burn-in counters or phases differ")
            slot = int(row["temperature_index"])
            if float(row["temperature"]) != temperatures[slot] or not all(
                math.isfinite(float(row[k]))
                for k in ("energy_per_site", "magnetization", "step_size")
            ):
                raise ValueError("Invalid PT burn-in values")
            if row["phase"] == "settle" and float(row["step_size"]) != steps[slot]:
                raise ValueError("Step size not frozen by temperature")
            eq_count += 1
    if eq_count != len(expected):
        raise ValueError("Truncated PT burn-in")
    attempts = [0] * (K - 1)
    accepts = [0] * (K - 1)
    with (folder / "swaps.csv").open(newline="") as stream:
        for row in csv.DictReader(stream):
            a = int(row["lower_temperature_index"])
            b = int(row["upper_temperature_index"])
            if (
                not 0 <= a < b < K
                or b != a + 1
                or row["phase"] not in ("settle", "measurement")
            ):
                raise ValueError("Invalid adjacent swap record")
            log_ratio = (1 / temperatures[a] - 1 / temperatures[b]) * (
                float(row["lower_total_energy"]) - float(row["upper_total_energy"])
            )
            if not math.isclose(
                float(row["log_ratio"]), log_ratio, rel_tol=1e-12, abs_tol=1e-12
            ):
                raise ValueError("Swap exponent violates total-energy convention")
            accepted = int(row["accepted"])
            if accepted not in (0, 1) or (log_ratio >= 0 and not accepted):
                raise ValueError("Invalid swap acceptance")
            if row["phase"] == "measurement":
                attempts[a] += 1
                accepts[a] += accepted
    if attempts != summary["swap_attempts"] or accepts != summary["swap_accepts"]:
        raise ValueError("Swap summary differs from raw attempts")
    swap_every = config["tempering"]["swap_every_sweeps"]
    passes = []
    if swap_every:
        passes.extend(
            (adapt + x, "settle") for x in range(swap_every, settle + 1, swap_every)
        )
        passes.extend(
            (adapt + settle + x, "measurement")
            for x in range(swap_every, sampling["measurement_sweeps"] + 1, swap_every)
        )
    transitions = [0] * K
    last_end = [None] * K
    labels = list(range(K))
    production_started = False

    def observe(current):
        if K == 1:
            return
        for replica, slot in enumerate(current):
            if slot in (0, K - 1):
                if last_end[replica] is not None and last_end[replica] != slot:
                    transitions[replica] += 1
                last_end[replica] = slot

    walk_rows = 0
    with (folder / "walks.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["sweep", "phase", "replica_id", "temperature_index"]:
            raise ValueError("Unexpected replica walk header")
        for row in reader:
            event, replica = divmod(walk_rows, K)
            if (
                event >= len(passes)
                or (int(row["sweep"]), row["phase"]) != passes[event]
                or int(row["replica_id"]) != replica
            ):
                raise ValueError("Replica walk counters differ")
            if row["phase"] == "measurement" and not production_started:
                observe(labels)
                production_started = True
            labels[replica] = int(row["temperature_index"])
            if replica == K - 1:
                if sorted(labels) != list(range(K)):
                    raise ValueError("Walk labels are not a permutation")
                if row["phase"] == "measurement":
                    observe(labels)
            walk_rows += 1
    if (
        walk_rows != len(passes) * K
        or [n // 2 for n in transitions] != summary["round_trips"]
    ):
        raise ValueError("Round-trip summary differs from replica walks")
    return summary


def launch(config_path, binary, mode="serial", mpiexec="mpiexec"):
    config_bytes = Path(config_path).read_bytes()
    config = validate(tomllib.loads(config_bytes.decode("utf-8")))
    binary = Path(binary).resolve(strict=True)
    o = config["output"]
    s = config["sampling"]
    m = config["model"]
    pt = config["tempering"]
    folder = REPO / o["root"] / o["campaign"] / o["run_id"]
    folder.mkdir(parents=True, exist_ok=False)
    meta = {
        "schema_version": 1,
        "sampler": "parallel_tempering",
        "mode": mode,
        "status": "RUNNING",
        "started_utc": single.utc(),
        "config": config,
        "N": m["L"] ** 3,
        "ranks": len(pt["temperatures"]) if mode == "mpi" else 1,
        "replica_seeds": [s["seed"] + r for r in range(len(pt["temperatures"]))],
        "exchange_seed": s["seed"] ^ 0x9E3779B97F4A7C15,
        "adaptation": "Independent per-temperature adaptation without swaps; fixed per-temperature steps during settling/measurement",
        "data_contract": "Synchronized temperature histories; resample aligned blocks across the complete ladder",
    }
    try:
        build_info = json.loads(
            subprocess.check_output([str(binary), "--build-info"], text=True)
        )
        if build_info.get("execution_mode") != mode:
            raise ValueError("PT executable does not match requested serial/MPI mode")
        meta["build_info"] = build_info
        single.write_metadata(folder, meta)
        (folder / "config.resolved.toml").write_bytes(config_bytes)
        (folder / "temperatures.txt").write_text(
            "".join(f"{T}\n" for T in pt["temperatures"]),
            encoding="utf-8",
            newline="\n",
        )
        meta["binary_sha256"] = single.sha256(binary)
        meta["source_id"], meta["source_files"] = single.archive_sources(folder)
        arguments = [
            m["L"],
            s["seed"],
            s["start"],
            s["step_size"],
            s["adaptation_sweeps"],
            s["equilibration_sweeps"],
            s["measurement_sweeps"],
            s["measure_every_sweeps"],
            pt["swap_every_sweeps"],
            m["J"],
            m["r"],
            m["u"],
            folder / "temperatures.txt",
            folder,
        ]
        command = (
            ([mpiexec, "-n", str(len(pt["temperatures"]))] if mode == "mpi" else [])
            + [str(binary)]
            + [str(a) for a in arguments]
        )
        meta["command"] = command
        single.write_metadata(folder, meta)
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        with (
            (folder / "stdout.log").open("w") as out,
            (folder / "stderr.log").open("w") as err,
        ):
            result = subprocess.run(
                command, stdout=out, stderr=err, env=env, check=False
            )
        meta["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(
                f"PT executable exited {result.returncode}; see stderr.log"
            )
        meta["summary"] = validate_outputs(folder, config)
        meta["output_hashes"] = {
            name: single.sha256(folder / name) for name in RAW_FILES
        }
        meta["status"] = "SUCCESS"
        meta["finished_utc"] = single.utc()
        single.write_metadata(folder, meta)
        (folder / "SUCCESS").write_text(
            "Operational completion; equilibration not established.\n"
        )
    except BaseException as error:
        meta.update(
            status="FAIL",
            finished_utc=single.utc(),
            error=f"{type(error).__name__}: {error}",
        )
        single.write_metadata(folder, meta)
        (folder / "FAIL").write_text(meta["error"] + "\n")
        raise
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--mode", choices=("serial", "mpi"), default="serial")
    parser.add_argument("--mpiexec", default="mpiexec")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(tomllib.loads(args.config.read_text(encoding="utf-8")))
        print("PT configuration valid; no simulation executed")
    elif not args.binary:
        parser.error("--binary is required")
    else:
        print(launch(args.config, args.binary, args.mode, args.mpiexec))


if __name__ == "__main__":
    main()
