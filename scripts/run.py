"""Run one phi4 simulation from a TOML configuration."""

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

# Resolve paths relative to the repository, even when called from another folder.
REPO = Path(__file__).resolve().parents[1]


def sha256(path):
    """Calculate a file's checksum so later changes can be detected."""
    with Path(path).open("rb") as input_file:
        checksum = hashlib.file_digest(input_file, "sha256")
    return checksum.hexdigest()


def utc():
    """Return the current time in UTC for the run record."""
    current_time = datetime.now(timezone.utc)
    return current_time.isoformat()


def check_keys(section, expected_keys, section_name):
    """Require all expected settings and reject extra or misspelled settings."""
    if not isinstance(section, dict):
        raise ValueError(f"{section_name} must be a configuration section")
    if set(section.keys()) != set(expected_keys):
        raise ValueError(f"{section_name}: expected exactly {expected_keys}")


def check_number(value, name):
    """Require an ordinary finite number; True and False are not valid numbers."""
    # Python treats bool as a kind of int, so check the exact type here.
    if type(value) not in (int, float):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def check_integer(value, name, minimum, maximum=10**9):
    """Require an integer within the allowed range, including both endpoints."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")


def validate(config):
    """Check the configuration before creating files or starting the simulator."""
    # Check the section names before accessing their contents.
    expected_sections = ["schema_version", "model", "sampling", "output"]
    if "scan" in config:
        expected_sections.append("scan")
    check_keys(config, expected_sections, "root")
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")

    model = config["model"]
    sampling = config["sampling"]
    output_settings = config["output"]

    check_keys(model, ["name", "L", "temperature", "J", "r", "u", "boundary"], "model")
    check_keys(
        sampling,
        [
            "start",
            "seed",
            "update_order",
            "step_size",
            "adaptation_sweeps",
            "equilibration_sweeps",
            "measurement_sweeps",
            "measure_every_sweeps",
        ],
        "sampling",
    )
    check_keys(
        output_settings,
        ["campaign", "run_id", "root", "save_time_series"],
        "output",
    )

    # Model parameters: positive temperature and u, nonnegative J, negative r.
    if model["name"] != "clean_phi4_3d":
        raise ValueError("Unsupported model")
    if model["boundary"] != "periodic":
        raise ValueError("Only periodic boundaries are supported")
    check_integer(model["L"], "L", 3, 512)
    for parameter_name in ("temperature", "J", "r", "u"):
        check_number(model[parameter_name], parameter_name)
    if model["temperature"] <= 0:
        raise ValueError("temperature must be positive")
    if model["u"] <= 0:
        raise ValueError("u must be positive")
    # J=0 is useful for checking the sampler against independent-site results.
    if model["J"] < 0:
        raise ValueError("J must be nonnegative")
    if model["r"] >= 0:
        raise ValueError("r must be negative")

    # Sampling settings: initialization, random seed, and sweep counts.
    allowed_starts = ["random", "ordered_positive", "ordered_negative"]
    if sampling["start"] not in allowed_starts:
        raise ValueError(f"start must be one of {allowed_starts}")
    if sampling["update_order"] != "sequential":
        raise ValueError("Only sequential updates are supported")
    check_integer(sampling["seed"], "seed", 0, 2**64 - 1)
    check_number(sampling["step_size"], "step_size")
    if sampling["step_size"] < 0.0001 or sampling["step_size"] > 100:
        raise ValueError("step_size must be in [0.0001, 100]")
    check_integer(sampling["adaptation_sweeps"], "adaptation_sweeps", 0)
    check_integer(sampling["equilibration_sweeps"], "equilibration_sweeps", 1)
    check_integer(sampling["measurement_sweeps"], "measurement_sweeps", 1)
    check_integer(sampling["measure_every_sweeps"], "measure_every_sweeps", 1)

    measurement_sweeps = sampling["measurement_sweeps"]
    measurement_interval = sampling["measure_every_sweeps"]
    if measurement_sweeps % measurement_interval != 0:
        raise ValueError("measurement_sweeps must be divisible by measure_every_sweeps")

    # Output names become folder names. Allow letters, numbers, '_' and '-'.
    for setting_name in ("campaign", "run_id"):
        folder_name = output_settings[setting_name]
        if not isinstance(folder_name, str):
            raise ValueError(f"{setting_name} must be a string")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", folder_name) is None:
            raise ValueError(f"Invalid {setting_name}")
    if output_settings["save_time_series"] is not True:
        raise ValueError("Full time series are required")

    # Keep generated files inside the repository, including through symlinks.
    if not isinstance(output_settings["root"], str):
        raise ValueError("output.root must be a relative path")
    output_root = Path(output_settings["root"])
    if output_root.is_absolute() or output_root.drive:
        raise ValueError("output.root must be a relative path")
    if ".." in output_root.parts or not output_root.parts:
        raise ValueError("output.root must be a relative path without parent traversal")
    resolved_output_root = (REPO / output_root).resolve()
    if not resolved_output_root.is_relative_to(REPO):
        raise ValueError("output.root escapes the repository")
    if "scan" in config:
        validate_scan(config)
    return config


def validate_scan(config):
    """Require a temperature ladder in the direction of the chosen start."""
    scan = config["scan"]
    expected_keys = ["temperatures"]
    if "sizes" in scan:
        expected_keys.append("sizes")
    check_keys(scan, expected_keys, "scan")
    sizes = scan.get("sizes", [config["model"]["L"]])
    if not isinstance(sizes, list) or len(sizes) == 0:
        raise ValueError("scan.sizes must be a nonempty list")
    seen_sizes = []
    for size in sizes:
        check_integer(size, "scan size", 3, 512)
        if size in seen_sizes:
            raise ValueError("scan.sizes must not contain duplicates")
        seen_sizes.append(size)
    if sizes[0] != config["model"]["L"]:
        raise ValueError("model.L must equal the first scan size")
    # Give each size a distinct recorded seed without exceeding the C++ seed type.
    check_integer(
        config["sampling"]["seed"] + len(sizes) - 1, "last size seed", 0, 2**64 - 1
    )
    temperatures = scan["temperatures"]
    if not isinstance(temperatures, list) or len(temperatures) == 0:
        raise ValueError("scan.temperatures must contain at least one temperature")
    for temperature in temperatures:
        check_number(temperature, "scan temperature")
        if temperature <= 0:
            raise ValueError("Scan temperatures must be positive")
    if temperatures[0] != config["model"]["temperature"]:
        raise ValueError("model.temperature must equal the first scan temperature")
    for index in range(1, len(temperatures)):
        previous_temperature = temperatures[index - 1]
        current_temperature = temperatures[index]
        if config["sampling"]["start"] == "random":
            if current_temperature >= previous_temperature:
                raise ValueError(
                    "A random hot start must scan from high to low temperature"
                )
        else:
            if current_temperature <= previous_temperature:
                raise ValueError(
                    "An ordered start must scan from low to high temperature"
                )


def source_snapshot():
    """List the source files and calculate a checksum for each one."""
    source_patterns = [
        "include/**/*.hpp",
        "src/*.cpp",
        "scripts/*.py",
        "analysis/**/*.py",
        "tests/**/*.py",
        "tests/**/*.cpp",
        "CMakeLists.txt",
    ]
    source_paths = []
    for pattern in source_patterns:
        source_paths.extend(REPO.glob(pattern))
    source_paths.sort()

    source_checksums = {}
    for source_path in source_paths:
        relative_path = source_path.relative_to(REPO).as_posix()
        source_checksums[relative_path] = sha256(source_path)

    # Hash the whole list too, giving this collection of sources one identifier.
    checksum_list = json.dumps(source_checksums, sort_keys=True)
    source_id = hashlib.sha256(checksum_list.encode("utf-8")).hexdigest()
    return source_id, source_checksums, source_paths


def validate_outputs(folder, config):
    """Check the saved measurements even if the simulator reports success."""
    sampling = config["sampling"]
    burn_in_sweeps = sampling["adaptation_sweeps"] + sampling["equilibration_sweeps"]
    measurement_interval = sampling["measure_every_sweeps"]
    expected_measurement_count = sampling["measurement_sweeps"] // measurement_interval

    # Measurements start after burn-in. The CSV sample index starts at zero.
    measurement_count = 0
    with (folder / "measurements.csv").open(newline="") as measurement_file:
        measurement_rows = csv.DictReader(measurement_file)
        expected_header = [
            "sample_index",
            "sweep",
            "energy_per_site",
            "magnetization",
        ]
        if measurement_rows.fieldnames != expected_header:
            raise ValueError("Unexpected measurement header")

        for row in measurement_rows:
            expected_sample_index = measurement_count
            measurement_count += 1
            expected_sweep = burn_in_sweeps + measurement_count * measurement_interval
            if int(row["sample_index"]) != expected_sample_index:
                raise ValueError("Unexpected sample index")
            if int(row["sweep"]) != expected_sweep:
                raise ValueError("Unexpected measurement sweep")
            for column_name in ("energy_per_site", "magnetization"):
                value = float(row[column_name])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite measurement: {column_name}")

    if measurement_count != expected_measurement_count:
        raise ValueError("Incorrect measurement count")

    # Burn-in has one diagnostic row per sweep, including adaptation and settling.
    equilibration_count = 0
    with (folder / "equilibration.csv").open(newline="") as equilibration_file:
        equilibration_rows = csv.DictReader(equilibration_file)
        for row in equilibration_rows:
            equilibration_count += 1
            if int(row["sweep"]) != equilibration_count:
                raise ValueError("Invalid equilibration counter")
            for column_name in (
                "energy_per_site",
                "magnetization",
                "acceptance",
                "step_size",
            ):
                value = float(row[column_name])
                if not math.isfinite(value):
                    raise ValueError(f"Nonfinite equilibration value: {column_name}")
            acceptance = float(row["acceptance"])
            step_size = float(row["step_size"])
            if acceptance < 0 or acceptance > 1:
                raise ValueError("Equilibration acceptance must be in [0, 1]")
            if step_size <= 0:
                raise ValueError("Equilibration step_size must be positive")

    if equilibration_count != burn_in_sweeps:
        raise ValueError("Incorrect equilibration count")
    return measurement_count


def write_metadata(folder, metadata):
    """Save the run record without leaving a partially written JSON file."""
    # Finish writing a temporary file, then replace the previous record.
    temporary_path = folder / "metadata.tmp"
    metadata_text = json.dumps(metadata, indent=2) + "\n"
    temporary_path.write_text(metadata_text, encoding="utf-8")
    temporary_path.replace(folder / "metadata.json")


def archive_sources(folder):
    """Save a copy of the source files alongside this run."""
    source_id, source_checksums, source_paths = source_snapshot()
    with zipfile.ZipFile(folder / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for source_path in source_paths:
            relative_path = source_path.relative_to(REPO).as_posix()
            archive.write(source_path, relative_path)
    return source_id, source_checksums


def run(config_path, binary):
    """Read the configuration, run one chain, and check and record its output."""
    # Read and validate before creating the output directory.
    config_bytes = Path(config_path).read_bytes()
    config = tomllib.loads(config_bytes.decode("utf-8"))
    validate(config)
    binary_path = Path(binary).resolve(strict=True)
    model = config["model"]
    sampling = config["sampling"]
    output_settings = config["output"]

    output_directory = (
        REPO
        / output_settings["root"]
        / output_settings["campaign"]
        / output_settings["run_id"]
    )
    # Refuse to overwrite an existing attempt. A retry needs a new run_id.
    output_directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_utc": utc(),
        "config": config,
    }

    try:
        # Save the inputs and the environment used for this attempt.
        write_metadata(output_directory, metadata)
        (output_directory / "config.resolved.toml").write_bytes(config_bytes)
        metadata["N"] = model["L"] ** 3
        metadata["units"] = "kB=1; energy units set by the recorded fixed couplings"
        metadata["rng"] = "mt19937_64 with std::uniform_real_distribution"
        metadata["platform"] = platform.platform()
        metadata["python"] = sys.version
        metadata["binary_sha256"] = sha256(binary_path)
        metadata["adaptation"] = (
            "Every 50 burn-in sweeps: acceptance <0.4 shrinks step by 10%; "
            ">0.6 grows it by 10%; clamp [1e-4,100]"
        )
        metadata["adaptation_stop_sweep"] = sampling["adaptation_sweeps"]
        metadata["command"] = sys.argv

        source_id, source_checksums = archive_sources(output_directory)
        metadata["source_id"] = source_id
        metadata["source_files"] = source_checksums
        build_info = subprocess.check_output(
            [str(binary_path), "--build-info"], text=True
        )
        metadata["build_info"] = build_info.strip()
        # The archive captures files at launch, so rebuild after changing C++.
        metadata["source_note"] = (
            "Source snapshot at launch; compiled-source equality is not checked"
        )
        compile_commands_path = binary_path.parent / "compile_commands.json"
        if compile_commands_path.exists():
            compile_commands_text = compile_commands_path.read_text()
            metadata["compile_commands"] = json.loads(compile_commands_text)

        # Each size is a separate C++ process with its own lattice and RNG.
        sizes = [model["L"]]
        if "scan" in config:
            sizes = config["scan"].get("sizes", sizes)
            temperature_path = output_directory / "temperatures.txt"
            with temperature_path.open("w") as temperature_file:
                for temperature in config["scan"]["temperatures"]:
                    temperature_file.write(str(temperature) + "\n")
            metadata.pop("N")
            metadata["scan_outputs"] = []
            metadata["size_outputs"] = []
            metadata["scan_note"] = (
                "Each size starts a fresh lattice and RNG, with seed = base seed + size index. "
                "Within each size, fields, RNG, and step size continue between temperatures. "
                "Adaptation and settling repeat at every temperature. "
                "Temperatures within a size share a history."
            )

        for size_index in range(len(sizes)):
            size = sizes[size_index]
            seed = sampling["seed"] + size_index
            simulation_directory = output_directory
            size_record = None
            if "scan" in config:
                simulation_directory = output_directory / ("L" + str(size))
                simulation_directory.mkdir(exist_ok=False)
                size_record = {
                    "L": size,
                    "N": size**3,
                    "seed": seed,
                    "directory": simulation_directory.name,
                    "status": "RUNNING",
                }
                metadata["size_outputs"].append(size_record)

            # Argument order matches argv[1] through argv[13] in C++ main().
            simulator_arguments = [
                size,
                model["temperature"],
                model["J"],
                model["r"],
                model["u"],
                seed,
                sampling["start"],
                sampling["step_size"],
                sampling["adaptation_sweeps"],
                sampling["equilibration_sweeps"],
                sampling["measurement_sweeps"],
                sampling["measure_every_sweeps"],
                simulation_directory,
            ]
            if "scan" in config:
                simulator_arguments.append(temperature_path)
            argument_strings = []
            for argument in simulator_arguments:
                argument_strings.append(str(argument))
            if size_record is None:
                metadata["simulator_arguments"] = argument_strings
            else:
                size_record["simulator_arguments"] = argument_strings
            write_metadata(output_directory, metadata)

            # Each size has its own logs so failures can be traced to that size.
            command = [str(binary_path)] + argument_strings
            stdout_path = simulation_directory / "stdout.log"
            stderr_path = simulation_directory / "stderr.log"
            with (
                stdout_path.open("w") as stdout_file,
                stderr_path.open("w") as stderr_file,
            ):
                result = subprocess.run(
                    command, stdout=stdout_file, stderr=stderr_file, check=False
                )
            metadata["exit_code"] = result.returncode
            if size_record is not None:
                size_record["exit_code"] = result.returncode
            if result.returncode != 0:
                if size_record is not None:
                    size_record["status"] = "FAIL"
                raise RuntimeError(
                    f"Simulator exited {result.returncode}; see {stderr_path}"
                )

            if "scan" in config:
                summary_lines = stdout_path.read_text().splitlines()
                temperatures = config["scan"]["temperatures"]
                if len(summary_lines) != len(temperatures):
                    raise ValueError(
                        "Simulator returned the wrong number of temperatures"
                    )
                for index in range(len(temperatures)):
                    point = json.loads(summary_lines[index])
                    point_directory = simulation_directory / (
                        "temperature_" + str(index)
                    )
                    measurement_count = validate_outputs(point_directory, config)
                    if point["measurement_count"] != measurement_count:
                        raise ValueError("Simulator measurement count differs from CSV")
                    point["L"] = size
                    point["N"] = size**3
                    point["seed"] = seed
                    point["temperature"] = temperatures[index]
                    point["directory"] = point_directory.relative_to(
                        output_directory
                    ).as_posix()
                    metadata["scan_outputs"].append(point)
                size_record["status"] = "SUCCESS"
                write_metadata(output_directory, metadata)
            else:
                simulator_summary = json.loads(stdout_path.read_text())
                metadata.update(simulator_summary)
                measurement_count = validate_outputs(output_directory, config)
                if metadata["measurement_count"] != measurement_count:
                    raise ValueError("Simulator measurement count differs from CSV")

        # Record file checksums so analysis can detect changed or damaged output.
        output_hashes = {}
        output_files = ["config.resolved.toml", "source.zip"]
        if "scan" in config:
            output_files.append("temperatures.txt")
            for size_record in metadata["size_outputs"]:
                output_files.append(size_record["directory"] + "/stdout.log")
                output_files.append(size_record["directory"] + "/stderr.log")
            for temperature_output in metadata["scan_outputs"]:
                directory_name = temperature_output["directory"]
                for filename in (
                    "measurements.csv",
                    "equilibration.csv",
                    "initial_field.txt",
                    "final_field.txt",
                ):
                    output_files.append(directory_name + "/" + filename)
        else:
            output_files.extend(["measurements.csv", "equilibration.csv"])
        for filename in output_files:
            output_hashes[filename] = sha256(output_directory / filename)
        metadata["output_hashes"] = output_hashes
        metadata["status"] = "SUCCESS"
        metadata["finished_utc"] = utc()
        write_metadata(output_directory, metadata)
        (output_directory / "SUCCESS").write_text(
            "Operational completion; equilibrium not established.\n"
        )
    except BaseException as error:
        # Include keyboard interruptions: preserve an explicit failed-attempt record.
        metadata["status"] = "FAIL"
        for size_record in metadata.get("size_outputs", []):
            if size_record["status"] == "RUNNING":
                size_record["status"] = "FAIL"
        metadata["finished_utc"] = utc()
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_metadata(output_directory, metadata)
        (output_directory / "FAIL").write_text(metadata["error"] + "\n")
        raise

    return output_directory


def main():
    """Read command-line options and either validate or launch one simulation."""
    if sys.platform == "win32":
        default_binary = REPO / "build/local-debug/phi4.exe"
    else:
        default_binary = REPO / "build/local-debug/phi4"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Path to the TOML configuration")
    parser.add_argument(
        "--binary", type=Path, default=default_binary, help="Path to the C++ executable"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check the config without running C++",
    )
    arguments = parser.parse_args()

    if arguments.validate_only:
        config_text = arguments.config.read_text(encoding="utf-8")
        config = tomllib.loads(config_text)
        validate(config)
        print("Configuration valid; no simulation executed")
    else:
        output_directory = run(arguments.config, arguments.binary)
        print(output_directory)


if __name__ == "__main__":
    main()
