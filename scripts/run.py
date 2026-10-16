"""Run one phi4 simulation from a TOML configuration."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import sys
import tomllib
import zipfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    """Stream hashes so large histories need no extra full memory copy."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def exact_keys(section, keys, name):
    """Reject misspelled parameters rather than silently using defaults."""
    if not isinstance(section, dict) or set(section) != set(keys.split()):
        raise ValueError(f"{name}: expected exactly {keys}")


def number(value, name, low=None, strict=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if low is not None and (value <= low if strict else value < low):
        raise ValueError(f"{name} is below its allowed bound")


def integer(value, name, low, high=10**9):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def validate(config):
    """Validate the complete schema before creating any output."""
    exact_keys(config, "schema_version model sampling output", "root")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
    model, sampling, output_settings = (
        config["model"],
        config["sampling"],
        config["output"],
    )
    exact_keys(model, "name L temperature J r u boundary", "model")
    exact_keys(
        sampling,
        "start seed update_order step_size adaptation_sweeps equilibration_sweeps measurement_sweeps measure_every_sweeps",
        "sampling",
    )
    exact_keys(output_settings, "campaign run_id root save_time_series", "output")
    if model["name"] != "clean_phi4_3d" or model["boundary"] != "periodic":
        raise ValueError("Unsupported model or boundary")
    integer(model["L"], "L", 3, 512)
    for name in ("temperature", "u"):
        number(model[name], name, 0, True)
    number(
        model["J"], "J", 0
    )  # J=0 is allowed for the independent-distribution validation.
    number(model["r"], "r")
    if model["r"] >= 0:
        raise ValueError("r must be negative")
    if (
        sampling["start"] not in ("ordered_positive", "ordered_negative", "disordered")
        or sampling["update_order"] != "sequential"
    ):
        raise ValueError("Invalid initialization or update order")
    integer(sampling["seed"], "seed", 0, 2**64 - 1)
    number(sampling["step_size"], "step_size", 0, True)
    if not 1e-4 <= sampling["step_size"] <= 100:
        raise ValueError("step_size must be in [0.0001, 100]")
    integer(sampling["adaptation_sweeps"], "adaptation_sweeps", 0)
    for name in ("equilibration_sweeps", "measurement_sweeps", "measure_every_sweeps"):
        integer(sampling[name], name, 1)
    if sampling["measurement_sweeps"] % sampling["measure_every_sweeps"]:
        raise ValueError("measurement_sweeps must be divisible by measure_every_sweeps")
    for name in ("campaign", "run_id"):
        if not isinstance(output_settings[name], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", output_settings[name]
        ):
            raise ValueError(f"Invalid {name}")
    if output_settings["save_time_series"] is not True:
        raise ValueError("Full time series are required")
    root = (
        Path(output_settings["root"])
        if isinstance(output_settings["root"], str)
        else None
    )
    if (
        root is None
        or root.is_absolute()
        or root.drive
        or ".." in root.parts
        or not root.parts
    ):
        raise ValueError("output.root must be a relative path without parent traversal")
    if not (REPO / root).resolve().is_relative_to(REPO):
        raise ValueError("output.root escapes the repository")
    return config


def source_snapshot():
    """Capture numerical, launcher, analysis and test sources by content hash."""
    patterns = [
        "include/**/*.hpp",
        "src/*.cpp",
        "scripts/*.py",
        "analysis/**/*.py",
        "tests/**/*.py",
        "tests/**/*.cpp",
        "CMakeLists.txt",
    ]
    paths = []
    for pattern in patterns:
        paths.extend(REPO.glob(pattern))
    paths.sort()
    files = {p.relative_to(REPO).as_posix(): sha256(p) for p in paths}
    identity = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return identity, files, paths


def validate_outputs(folder, config):
    """Verify row counts and counters even after a zero simulator exit."""
    s = config["sampling"]
    count = 0
    first_sweep = s["adaptation_sweeps"] + s["equilibration_sweeps"]
    with (folder / "measurements.csv").open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != [
            "sample_index",
            "sweep",
            "energy_per_site",
            "magnetization",
        ]:
            raise ValueError("Unexpected measurement header")
        for count, row in enumerate(reader, 1):
            if (
                int(row["sample_index"]) != count - 1
                or int(row["sweep"]) != first_sweep + count * s["measure_every_sweeps"]
            ):
                raise ValueError("Unexpected sample index or sweep")
            if not all(
                math.isfinite(float(row[k]))
                for k in ("energy_per_site", "magnetization")
            ):
                raise ValueError("Nonfinite measurements")
    if count != s["measurement_sweeps"] // s["measure_every_sweeps"]:
        raise ValueError("Incorrect sample count")
    eq_count = 0
    with (folder / "equilibration.csv").open(newline="") as f:
        for eq_count, row in enumerate(csv.DictReader(f), 1):
            if int(row["sweep"]) != eq_count:
                raise ValueError("Invalid equilibration counter")
            if not all(
                math.isfinite(float(row[k]))
                for k in ("energy_per_site", "magnetization", "acceptance", "step_size")
            ):
                raise ValueError("Nonfinite equilibration")
            if not 0 <= float(row["acceptance"]) <= 1 or float(row["step_size"]) <= 0:
                raise ValueError("Invalid equilibration diagnostics")
    if eq_count != first_sweep:
        raise ValueError("Incorrect equilibration count")
    return count


def write_metadata(folder, metadata):
    """Replace the file atomically so an interruption cannot leave half a JSON file."""
    temporary = folder / "metadata.tmp"
    temporary.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    temporary.replace(folder / "metadata.json")


def archive_sources(folder):
    """Keep the source files present at launch, along with their content hashes."""
    source_id, checksums, paths = source_snapshot()
    with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path, path.relative_to(REPO).as_posix())
    return source_id, checksums


def run(config_path, binary):
    """Launch one chain and preserve successful and failed attempts."""
    raw = Path(config_path).read_bytes()
    config = validate(tomllib.loads(raw.decode("utf-8")))
    binary = Path(binary).resolve(strict=True)
    model, sampling, output_settings = (
        config["model"],
        config["sampling"],
        config["output"],
    )
    folder = (
        REPO
        / output_settings["root"]
        / output_settings["campaign"]
        / output_settings["run_id"]
    )
    # A retry needs a new run_id; completed raw data must not be replaced.
    folder.mkdir(parents=True, exist_ok=False)
    meta = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_utc": utc(),
        "config": config,
    }

    try:
        write_metadata(folder, meta)
        (folder / "config.resolved.toml").write_bytes(raw)
        meta.update(
            {
                "N": model["L"] ** 3,
                "units": "kB=1; energy units set by the recorded fixed couplings",
                "rng": "mt19937_64 with std::uniform_real_distribution",
                "platform": platform.platform(),
                "python": sys.version,
                "binary_sha256": sha256(binary),
                "adaptation": "Every 50 burn-in sweeps: acceptance <0.4 shrinks step by 10%; >0.6 grows it by 10%; clamp [1e-4,100]",
                "adaptation_stop_sweep": sampling["adaptation_sweeps"],
                "command": sys.argv,
            }
        )
        meta["source_id"], meta["source_files"] = archive_sources(folder)
        meta["build_info"] = subprocess.check_output(
            [str(binary), "--build-info"], text=True
        ).strip()
        # The source archive records files at launch. Rebuild after editing C++.
        meta["source_note"] = (
            "Source snapshot at launch; compiled-source equality is not checked"
        )
        compile_commands = binary.parent / "compile_commands.json"
        if compile_commands.exists():
            meta["compile_commands"] = json.loads(compile_commands.read_text())
        args = [
            model["L"],
            model["temperature"],
            model["J"],
            model["r"],
            model["u"],
            sampling["seed"],
            sampling["start"],
            sampling["step_size"],
            sampling["adaptation_sweeps"],
            sampling["equilibration_sweeps"],
            sampling["measurement_sweeps"],
            sampling["measure_every_sweeps"],
            folder,
        ]
        meta["simulator_arguments"] = list(map(str, args))
        write_metadata(folder, meta)
        with (
            (folder / "stdout.log").open("w") as stdout,
            (folder / "stderr.log").open("w") as stderr,
        ):
            result = subprocess.run(
                [str(binary), *map(str, args)],
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        meta["exit_code"] = result.returncode
        if result.returncode:
            raise RuntimeError(f"Simulator exited {result.returncode}; see stderr.log")
        meta.update(json.loads((folder / "stdout.log").read_text()))
        if meta["sample_count"] != validate_outputs(folder, config):
            raise ValueError("Simulator sample count differs from CSV")
        meta["output_hashes"] = {
            name: sha256(folder / name)
            for name in (
                "measurements.csv",
                "equilibration.csv",
                "config.resolved.toml",
                "source.zip",
            )
        }
        meta["status"] = "SUCCESS"
        meta["finished_utc"] = utc()
        write_metadata(folder, meta)
        (folder / "SUCCESS").write_text(
            "Operational completion; equilibrium not established.\n"
        )
    except BaseException as exc:
        # KeyboardInterrupt also leaves an explicit failed-attempt record.
        meta.update(
            status="FAIL", finished_utc=utc(), error=f"{type(exc).__name__}: {exc}"
        )
        write_metadata(folder, meta)
        (folder / "FAIL").write_text(meta["error"] + "\n")
        raise
    return folder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument(
        "--binary",
        type=Path,
        default=REPO / "build/local-debug/phi4.exe"
        if sys.platform == "win32"
        else REPO / "build/local-debug/phi4",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validate_only:
        validate(tomllib.loads(args.config.read_text(encoding="utf-8")))
        print("Configuration valid; no simulation executed")
    else:
        print(run(args.config, args.binary))
