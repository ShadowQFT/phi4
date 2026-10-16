"""Plot preliminary observables from completed temperature scans; launch no runs."""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import run as runner

sys.path.insert(0, str(runner.REPO / "analysis"))
from phi4_analysis.statistics import block_jackknife


def load_scan(folder):
    """Require a completed, unchanged scan before using its saved measurements."""
    folder = Path(folder).resolve()
    metadata = json.loads((folder / "metadata.json").read_text())
    if metadata["status"] != "SUCCESS" or not (folder / "SUCCESS").exists():
        raise ValueError("Scan is incomplete")
    if (folder / "FAIL").exists():
        raise ValueError("Scan has a failure marker")
    runner.validate(metadata["config"])
    if "scan" not in metadata["config"]:
        raise ValueError("This run is not a temperature scan")
    for filename, expected_checksum in metadata["output_hashes"].items():
        file_path = (folder / filename).resolve()
        if not file_path.is_relative_to(folder):
            raise ValueError("Output file escapes the scan folder")
        if runner.sha256(file_path) != expected_checksum:
            raise ValueError(f"Output changed: {filename}")
    return metadata


def read_measurements(folder):
    """Read the saved energy and magnetization time series at one temperature."""
    energies = []
    magnetizations = []
    with (folder / "measurements.csv").open(newline="") as input_file:
        for row in csv.DictReader(input_file):
            energies.append(float(row["energy_per_site"]))
            magnetizations.append(float(row["magnetization"]))
    return np.array(energies), np.array(magnetizations)


def plot_scans(scan_folders, output_directory, block_size):
    """Keep each heating/cooling curve separate and estimate errors in blocks."""
    scans = []
    reference_model = None
    for folder in scan_folders:
        folder = Path(folder).resolve()
        metadata = load_scan(folder)
        model = metadata["config"]["model"]
        couplings = (model["J"], model["r"], model["u"])
        if reference_model is None:
            reference_model = couplings
        if couplings != reference_model:
            raise ValueError("Overlay scans only when J, r, and u match")
        # Older single-size scans store their points directly under the run root.
        # Read the recorded paths as they are; existing raw data is never moved.
        sizes = []
        for point in metadata["scan_outputs"]:
            size = point.get("L", model["L"])
            if size not in sizes:
                sizes.append(size)
        for size in sizes:
            points = []
            for point in metadata["scan_outputs"]:
                if point.get("L", model["L"]) == size:
                    points.append(point)
            seed = points[0].get("seed", metadata["config"]["sampling"]["seed"])
            scans.append((folder, metadata, size, seed, points))

    start_titles = {
        "random": "Hot start",
        "ordered_positive": "Ordered positive start",
        "ordered_negative": "Ordered negative start",
    }
    titles = []
    for folder, metadata, size, seed, points in scans:
        start = metadata["config"]["sampling"]["start"]
        title = start_titles[start]
        if title not in titles:
            titles.append(title)
    figure_title = " / ".join(titles)

    # Refuse to replace an earlier plot set.
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=False)
    summary_rows = []
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    observables = ["e", "abs_m", "m", "binder", "chi_abs", "heat_capacity_per_site"]
    labels = [
        r"$\langle e \rangle$",
        r"$\langle |m| \rangle$",
        r"$\langle m \rangle$",
        "binder",
        r"$\langle \chi \rangle$",
        "$c$",
    ]
    history_figure, history_axes = plt.subplots(
        len(scans), 2, figsize=(12, 3 * len(scans)), squeeze=False, layout="constrained"
    )
    provenance = {
        "block_size": block_size,
        "runs": [],
        "note": (
            "Preliminary scans; equilibration and block-length sufficiency are not established. "
            "Error bars are per-temperature block jackknife errors. "
            "Temperatures in one scan are correlated; curves are not pooled."
        ),
    }

    try:
        for scan_index, (folder, metadata, size, seed, points) in enumerate(scans):
            config = metadata["config"]
            model = config["model"]
            start = config["sampling"]["start"]
            direction = "cooling" if start == "random" else "heating"
            label = f"L = {size}"
            temperatures = []
            estimates = {}
            errors = {}
            for observable in observables:
                estimates[observable] = []
                errors[observable] = []

            for point in points:
                point_folder = folder / point["directory"]
                runner.validate_outputs(point_folder, config)
                energy, magnetization = read_measurements(point_folder)
                result = block_jackknife(
                    energy,
                    magnetization,
                    size**3,
                    point["temperature"],
                    block_size,
                )
                temperatures.append(point["temperature"])
                row = {
                    "run": str(folder),
                    "L": size,
                    "seed": seed,
                    "start": start,
                    "direction": direction,
                    "temperature": point["temperature"],
                    "measurement_count": len(energy),
                    "block_size": block_size,
                    "block_count": result["block_count"],
                    "dropped_tail_samples": result["dropped_tail_samples"],
                }
                for observable in observables:
                    estimate = result["estimate"][observable]
                    error = result["standard_error"][observable]
                    estimates[observable].append(estimate)
                    errors[observable].append(error)
                    row[observable] = estimate
                    row[observable + "_error"] = error
                summary_rows.append(row)

                # Show full production histories, including any incomplete block tail.
                sample_sweeps = np.arange(1, len(energy) + 1)
                sample_sweeps *= config["sampling"]["measure_every_sweeps"]
                history_axes[scan_index, 0].plot(
                    sample_sweeps,
                    energy,
                    linewidth=0.6,
                    label=f"T={point['temperature']:g}",
                )
                history_axes[scan_index, 1].plot(
                    sample_sweeps,
                    np.abs(magnetization),
                    linewidth=0.6,
                    label=f"T={point['temperature']:g}",
                )

            for index, observable in enumerate(observables):
                axis = axes.flat[index]
                axis.errorbar(
                    temperatures,
                    estimates[observable],
                    yerr=errors[observable],
                    marker="o",
                    markersize=4,
                    capsize=3,
                    label=label,
                )
                axis.set_xlabel("Temperature")
                axis.set_ylabel(labels[index], fontsize=13)
                axis.grid(alpha=0.25)
            history_axes[scan_index, 0].set_title(label)
            history_axes[scan_index, 0].set_ylabel("Energy per site")
            history_axes[scan_index, 1].set_ylabel("|m|")
            for axis in history_axes[scan_index]:
                axis.set_xlabel("Production sweeps at this temperature")
                axis.legend(fontsize=7)
            provenance["runs"].append(
                {
                    "path": str(folder),
                    "L": size,
                    "seed": seed,
                    "metadata_sha256": runner.sha256(folder / "metadata.json"),
                }
            )

        figure.suptitle(figure_title)
        axes.flat[0].legend(fontsize=7)
        history_figure.suptitle(figure_title + " — production histories")
        for extension in ("png", "pdf"):
            figure.savefig(
                output_directory / ("temperature_scan." + extension), dpi=160
            )
            history_figure.savefig(
                output_directory / ("histories." + extension), dpi=160
            )
        with (output_directory / "summary.csv").open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(summary_rows[0]))
            writer.writeheader()
            writer.writerows(summary_rows)
        provenance["plot_script_sha256"] = runner.sha256(Path(__file__))
        (output_directory / "plot_info.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        plt.close(figure)
        plt.close(history_figure)
    return output_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scans", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--block-size",
        type=int,
        default=100,
        help="Samples per block; at least 20 full blocks are required",
    )
    arguments = parser.parse_args()
    print(plot_scans(arguments.scans, arguments.output, arguments.block_size))


if __name__ == "__main__":
    main()
