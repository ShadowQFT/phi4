"""Analyze synchronized MPI/serial replica-exchange ladders without losing covariance."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from phi4_analysis.tempering import analyze_ladder, load_ladder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=200)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=544000)
    parser.add_argument("--target-temperature", type=float, default=15.32)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, folder in enumerate(args.folders):
        ladder = load_ladder(folder)
        report, blocks, samples = analyze_ladder(
            ladder, args.block_size, args.replicates, args.seed + index
        )
        critical = min(
            report["temperatures"],
            key=lambda row: abs(row["temperature"] - args.target_temperature),
        )
        report["target_temperature"] = args.target_temperature
        report["nearest_temperature_benchmark"] = critical
        reports.append(report)
        np.savez_compressed(
            output / f"ladder-{index}-blocks.npz",
            joint_moment_blocks=blocks,
            temperatures=ladder["temperatures"],
        )
        np.savez_compressed(
            output / f"ladder-{index}-bootstrap.npz",
            observables=samples,
            temperatures=ladder["temperatures"],
        )
    hashes = {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in (
            "scripts/analyze_pt.py",
            "analysis/phi4_analysis/tempering.py",
            "analysis/phi4_analysis/statistics.py",
            "analysis/phi4_analysis/scaling.py",
        )
    }
    (output / "analysis.json").write_text(
        json.dumps(
            {
                "block_size": args.block_size,
                "ladders": reports,
                "source_sha256": hashes,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for report in reports:
        label = (
            f"swap every {report['swap_every_sweeps']}"
            if report["swap_every_sweeps"]
            else "No exchanges"
        )
        for ax, name in zip(axes, ("binder", "abs_m", "chi_abs")):
            ax.errorbar(
                [r["temperature"] for r in report["temperatures"]],
                [r["statistics"]["estimate"][name] for r in report["temperatures"]],
                yerr=[
                    r["statistics"]["standard_error"][name]
                    for r in report["temperatures"]
                ],
                marker="o",
                capsize=2,
                label=label,
            )
            ax.set(xlabel="Temperature", ylabel=name)
    axes[0].legend(fontsize=8)
    fig.suptitle("Replica-exchange benchmark - preliminary block errors")
    fig.savefig(output / "observables.png", dpi=180)
    fig.savefig(output / "observables.pdf")
    plt.close(fig)
    print(
        json.dumps(
            [
                {
                    "folder": r["folder"],
                    "elapsed_seconds": r["elapsed_seconds"],
                    "estimated_cpu_hours": r["estimated_allocated_cpu_hours"],
                    "swap_rates": r["swap_acceptance_by_pair"],
                    "round_trips": r["round_trips"],
                    "nearest_temperature": r["nearest_temperature_benchmark"][
                        "temperature"
                    ],
                    "effective_samples_per_cpu_hour": r[
                        "nearest_temperature_benchmark"
                    ]["limiting_effective_samples_per_cpu_hour"],
                    "warnings": r["nearest_temperature_benchmark"]["warnings"],
                }
                for r in reports
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
