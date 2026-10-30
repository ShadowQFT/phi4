"""Hash-verified per-chain analysis and diagnostics; never starts a simulator."""

import argparse
import csv
import hashlib
import json
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import check_compatible, load_run, sha256
from .diagnostics import blocking_curve, plot_diagnostics, split_history
from .statistics import NAMES, block_bootstrap, block_jackknife, observables


def _write_json(path, value):
    # Require finite numbers in the JSON output.
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_observable_table(output, results):
    with (output / "observables.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run",
                "observable",
                "estimate",
                "jackknife_standard_error",
                "bootstrap_lower_95",
                "bootstrap_upper_95",
                "block_size",
                "block_count",
                "dropped_tail_samples",
            ]
        )
        for result in results:
            jackknife = result["block_jackknife"]
            for name in NAMES:
                bounds = result["block_bootstrap"]["percentile_interval_95"][name]
                writer.writerow(
                    [
                        result["run"],
                        name,
                        jackknife["estimate"][name],
                        jackknife["standard_error"][name],
                        *bounds,
                        jackknife["block_size"],
                        jackknife["block_count"],
                        jackknife["dropped_tail_samples"],
                    ]
                )


def write_blocking_table(output, results):
    with (output / "blocking.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "run",
                "block_size",
                "block_count",
                "dropped_tail_samples",
                *[f"se_{name}" for name in NAMES],
            ]
        )
        for result in results:
            for point in result["blocking"]:
                writer.writerow(
                    [
                        result["run"],
                        point["block_size"],
                        point["block_count"],
                        point["dropped_tail_samples"],
                        *[point["standard_error"][name] for name in NAMES],
                    ]
                )


def analyze(
    run_paths,
    output,
    *,
    block_size,
    bootstrap_seed,
    replicates=2000,
    discard_samples=0,
    plots=True,
):
    """Analyze compatible chains separately with an explicitly selected block size."""
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    if type(discard_samples) is not int or discard_samples < 0:
        raise ValueError("discard_samples must be a nonnegative integer")
    if type(bootstrap_seed) is not int or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    if not run_paths:
        raise ValueError("Supply at least one completed run")

    runs = [load_run(path) for path in run_paths]
    check_compatible(runs)
    output = Path(output).resolve()
    for run in runs:
        if output.is_relative_to(run.folder):
            raise ValueError("Analysis output must not be inside a raw run folder")

    results = []
    # Give each chain a different, reproducible bootstrap seed.
    for index, run in enumerate(runs):
        energy = run.energy[discard_samples:]
        magnetization = run.magnetization[discard_samples:]
        N = run.metadata["N"]
        T = run.config["model"]["temperature"]
        seed = bootstrap_seed + index
        jackknife = block_jackknife(energy, magnetization, N, T, block_size)
        result = {
            "run": run.label,
            "start": run.config["sampling"]["start"],
            "seed": run.config["sampling"]["seed"],
            "raw_sample_count": len(run.energy),
            "discarded_production_samples": discard_samples,
            "selected_sample_count": len(energy),
            "first_selected_sweep": int(run.sweeps[discard_samples]),
            "measure_every_sweeps": run.config["sampling"]["measure_every_sweeps"],
            "all_selected_samples": observables(energy, magnetization, N, T),
            "block_jackknife": jackknife,
            "block_bootstrap": block_bootstrap(
                energy,
                magnetization,
                N,
                T,
                block_size,
                seed=seed,
                replicates=replicates,
            ),
            "blocking": blocking_curve(energy, magnetization, N, T),
            "split_history": split_history(energy, magnetization, N, T, block_size),
        }
        results.append(result)

    # Finish validation and statistical analysis before creating a fresh output directory.
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        "schema_version": 1,
        "status": "RUNNING",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_id": output.name,
        "model": runs[0].config["model"],
        "inputs": [
            {"path": str(run.folder), "run": run.label, "hashes": run.input_hashes}
            for run in runs
        ],
        "settings": {
            "block_size": block_size,
            "bootstrap_seed": bootstrap_seed,
            "replicates": replicates,
            "discard_samples": discard_samples,
            "plots": plots,
        },
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
        },
        "command": sys.argv,
        "limitations": [
            "Per-chain results only; no pooling, Tc or exponent fit",
            "No automatic equilibrium or block-plateau verdict",
            "Percentile intervals are conditional on stationarity and adequate block length",
            "No integrated autocorrelation time or effective sample size is claimed",
        ],
    }
    try:
        provenance["environment"]["platform"] = platform.platform()
        _write_json(output / "provenance.json", provenance)
        package = Path(__file__).resolve().parent
        sources = sorted(package.glob("*.py"))
        provenance["source_hashes"] = {path.name: sha256(path) for path in sources}
        provenance["source_id"] = hashlib.sha256(
            json.dumps(provenance["source_hashes"], sort_keys=True).encode()
        ).hexdigest()
        with zipfile.ZipFile(
            output / "analysis_source.zip", "w", zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sources:
                archive.write(path, f"phi4_analysis/{path.name}")

        _write_json(
            output / "summary.json",
            {"chains": results, "limitations": provenance["limitations"]},
        )
        write_observable_table(output, results)
        write_blocking_table(output, results)

        artifacts = [
            "summary.json",
            "observables.csv",
            "blocking.csv",
            "analysis_source.zip",
        ]
        if plots:
            import matplotlib

            provenance["environment"]["matplotlib"] = matplotlib.__version__
            artifacts.extend(plot_diagnostics(runs, results, output, output.name))
        provenance["output_hashes"] = {
            name: sha256(output / name) for name in artifacts
        }
        provenance.update(
            status="SUCCESS", finished_utc=datetime.now(timezone.utc).isoformat()
        )
        _write_json(output / "provenance.json", provenance)
        (output / "SUCCESS").write_text(
            "Analysis completed; equilibrium remains a user-reviewed diagnostic.\n"
        )
    except BaseException as error:
        provenance.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        _write_json(output / "provenance.json", provenance)
        (output / "FAIL").write_text(provenance["error"] + "\n", encoding="utf-8")
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "runs", nargs="+", type=Path, help="Completed raw run directories"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="New analysis directory"
    )
    parser.add_argument(
        "--block-size",
        required=True,
        type=int,
        help="Contiguous block length in retained samples",
    )
    parser.add_argument("--bootstrap-seed", required=True, type=int)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument(
        "--discard-samples",
        type=int,
        default=0,
        help="Explicit additional production cut; never selected automatically",
    )
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    try:
        output = analyze(
            args.runs,
            args.output,
            block_size=args.block_size,
            bootstrap_seed=args.bootstrap_seed,
            replicates=args.replicates,
            discard_samples=args.discard_samples,
            plots=not args.no_plots,
        )
    except (ValueError, OSError, KeyError, TypeError, FloatingPointError) as error:
        parser.exit(2, f"Analysis failed: {error}\n")
    print(
        f"Analysis saved to {output}; inspect diagnostics before scientific interpretation."
    )


if __name__ == "__main__":
    main()
