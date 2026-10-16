"""Read completed raw runs without modifying their contents."""

import csv
import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    """Hash in chunks so large histories do not need another full memory copy."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class RunData:
    folder: Path
    metadata: dict
    input_hashes: dict
    sweeps: np.ndarray
    energy: np.ndarray
    magnetization: np.ndarray
    equilibration: dict

    @property
    def config(self) -> dict:
        return self.metadata["config"]

    @property
    def label(self) -> str:
        output = self.config["output"]
        return f"{output['campaign']}/{output['run_id']}"


def _read_csv(path: Path, expected_header: list[str]) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_header:
            raise ValueError(f"Unexpected CSV header: {path.name}")
        rows = list(reader)
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Empty or malformed CSV: {path.name}")
    return rows


def _column(rows: list[dict], key: str) -> np.ndarray:
    values = np.array([float(row[key]) for row in rows])
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite column: {key}")
    return values


def load_run(folder: Path) -> RunData:
    """Require successful status, exact sample counters and matching raw hashes.

    Hashes detect changes relative to the recorded metadata; they do not
    establish authenticity if somebody rewrites both data and metadata.
    """
    folder = Path(folder).resolve(strict=True)
    if not (folder / "SUCCESS").is_file() or (folder / "FAIL").exists():
        raise ValueError(f"Run is incomplete or failed: {folder}")
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("schema_version") != 1
        or metadata.get("status") != "SUCCESS"
        or metadata.get("exit_code") != 0
    ):
        raise ValueError("Unsupported or unsuccessful run metadata")

    names = (
        "measurements.csv",
        "equilibration.csv",
        "config.resolved.toml",
        "source.zip",
    )
    hashes = {name: sha256(folder / name) for name in names}
    for name, checksum in hashes.items():
        if metadata.get("output_hashes", {}).get(name) != checksum:
            raise ValueError(f"Raw-file hash mismatch: {name}")
    hashes["metadata.json"] = sha256(folder / "metadata.json")

    config = tomllib.loads(
        (folder / "config.resolved.toml").read_text(encoding="utf-8")
    )
    if config != metadata["config"] or config["schema_version"] != 1:
        raise ValueError("Resolved configuration differs from recorded metadata")
    model, sampling = config["model"], config["sampling"]
    if model["name"] != "clean_phi4_3d" or model["boundary"] != "periodic":
        raise ValueError("Unsupported physical convention")
    if type(model["L"]) is not int or not 3 <= model["L"] <= 512:
        raise ValueError("Invalid lattice length")
    if metadata["N"] != model["L"] ** 3:
        raise ValueError("Recorded site count differs from L cubed")
    if not all(
        type(model[key]) in (int, float) and np.isfinite(model[key])
        for key in ("J", "r", "u", "temperature")
    ):
        raise ValueError("Invalid model parameters")
    if (
        model["J"] < 0
        or model["r"] >= 0
        or model["u"] <= 0
        or model["temperature"] <= 0
    ):
        raise ValueError("Invalid model parameter bounds")
    for key in (
        "adaptation_sweeps",
        "equilibration_sweeps",
        "measurement_sweeps",
        "measure_every_sweeps",
    ):
        minimum = 0 if key == "adaptation_sweeps" else 1
        if type(sampling[key]) is not int or not minimum <= sampling[key] <= 10**9:
            raise ValueError(f"Invalid sampling count: {key}")
    if sampling["update_order"] != "sequential":
        raise ValueError("Unsupported update order")
    interval = sampling["measure_every_sweeps"]
    if sampling["measurement_sweeps"] % interval:
        raise ValueError("Measurement length must be divisible by interval")
    burn_in = sampling["adaptation_sweeps"] + sampling["equilibration_sweeps"]

    rows = _read_csv(
        folder / "measurements.csv",
        ["sample_index", "sweep", "energy_per_site", "magnetization"],
    )
    expected_count = sampling["measurement_sweeps"] // interval
    if len(rows) != expected_count or metadata["measurement_count"] != expected_count:
        raise ValueError("Incorrect production sample count")
    sweeps = np.array([int(row["sweep"]) for row in rows], dtype=np.int64)
    for index, row in enumerate(rows):
        if (
            int(row["sample_index"]) != index
            or sweeps[index] != burn_in + (index + 1) * interval
        ):
            raise ValueError("Unexpected production sample index or sweep")

    equilibration_rows = _read_csv(
        folder / "equilibration.csv",
        [
            "sweep",
            "energy_per_site",
            "magnetization",
            "acceptance",
            "step_size",
            "phase",
        ],
    )
    if len(equilibration_rows) != burn_in:
        raise ValueError("Incorrect burn-in row count")
    for index, row in enumerate(equilibration_rows, 1):
        phase = "adapt" if index <= sampling["adaptation_sweeps"] else "settle"
        if int(row["sweep"]) != index or row["phase"] != phase:
            raise ValueError("Unexpected burn-in sweep or phase")
    equilibration = {
        key: _column(equilibration_rows, key)
        for key in (
            "sweep",
            "energy_per_site",
            "magnetization",
            "acceptance",
            "step_size",
        )
    }
    if np.any((equilibration["acceptance"] < 0) | (equilibration["acceptance"] > 1)):
        raise ValueError("Acceptance fraction outside [0,1]")
    if np.any(equilibration["step_size"] <= 0):
        raise ValueError("Nonpositive proposal step")
    settling_steps = equilibration["step_size"][sampling["adaptation_sweeps"] :]
    if not np.all(settling_steps == metadata["final_step_size"]):
        raise ValueError("Proposal step was not frozen throughout settling")

    return RunData(
        folder,
        metadata,
        hashes,
        sweeps,
        _column(rows, "energy_per_site"),
        _column(rows, "magnetization"),
        equilibration,
    )


def check_compatible(runs: list[RunData]) -> None:
    """Allow side-by-side chains from the same ensemble, never implicit pooling."""
    if len({run.folder for run in runs}) != len(runs):
        raise ValueError("The same run was supplied more than once")
    if len({run.config["sampling"]["seed"] for run in runs}) != len(runs):
        raise ValueError("Comparison chains must use distinct seeds")
    reference = runs[0].config
    for run in runs[1:]:
        if run.config["model"] != reference["model"]:
            raise ValueError(
                "Chain comparison requires identical model, L and temperature"
            )
        for key in ("update_order", "measure_every_sweeps"):
            if run.config["sampling"][key] != reference["sampling"][key]:
                raise ValueError(f"Incompatible sampling convention: {key}")
