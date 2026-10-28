"""Inspect independent-chain campaigns and optionally fit finite-size scaling."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from phi4_analysis.scaling import bootstrap_scaling, load_ensembles  # noqa: E402
from phi4_analysis.statistics import NAMES  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=500)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--degree", type=int, choices=(1, 2), default=1)
    parser.add_argument("--lmin", type=int, default=0)
    parser.add_argument("--crossing-sizes", nargs=2, type=int)
    parser.add_argument("--tmin", type=float)
    parser.add_argument("--tmax", type=float)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=540928)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("Use at least 100 bootstrap replicates")
    folders, incomplete = [], []
    for campaign in args.campaigns:
        for folder in sorted(campaign.iterdir()):
            if folder.is_dir():
                if (folder / "SUCCESS").is_file():
                    folders.append(folder)
                else:
                    incomplete.append(str(folder))
    ensembles, reports, runs = load_ensembles(
        folders, args.block_size, args.tmin, args.tmax
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    info = {
        "block_size": args.block_size,
        "chains": reports,
        "incomplete_or_failed": incomplete,
        "arguments": {
            k: str(v)
            if isinstance(v, Path)
            else [str(p) for p in v]
            if k == "campaigns"
            else v
            for k, v in vars(args).items()
        },
        "scientific_status": "Preliminary. Operational success does not establish equilibration.",
    }
    info["analysis_source_sha256"] = {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in (
            "scripts/analyze_campaign.py",
            "analysis/phi4_analysis/scaling.py",
            "analysis/phi4_analysis/statistics.py",
            "analysis/phi4_analysis/data.py",
        )
    }
    info["numpy_version"] = np.__version__
    info["ensemble_checks"] = [
        {"L": g["L"], "T": g["T"], "chain_comparisons": g["chain_comparisons"]}
        for g in ensembles
    ]
    if args.fit:
        try:
            info["scaling"] = bootstrap_scaling(
                ensembles,
                degree=args.degree,
                lmin=args.lmin,
                replicates=args.replicates,
                seed=args.seed,
                crossing_sizes=args.crossing_sizes,
            )
            sizes = sorted({g["L"] for g in ensembles if g["L"] >= args.lmin})
            if len(sizes) >= 4 and (
                not args.crossing_sizes or min(args.crossing_sizes) >= sizes[1]
            ):
                # Keep the Tc-defining largest pair fixed while excluding smallest L.
                info["exclude_smallest"] = bootstrap_scaling(
                    ensembles,
                    degree=args.degree,
                    lmin=sizes[1],
                    replicates=args.replicates,
                    seed=args.seed,
                    crossing_sizes=args.crossing_sizes or sizes[-2:],
                )
        except ValueError as error:
            info["scaling_withheld"] = str(error)
    for key in ("scaling", "exclude_smallest"):
        if key in info and any(
            r["warnings"] for r in reports if r["L"] in info[key]["fit"]["sizes"]
        ):
            info[key]["fit_usable"] = False
            info[key]["quality_warnings"].append(
                "Chain sampling diagnostics unresolved; inspect per-chain warnings"
            )
            info[key]["status"] = "withheld: sampling or local-fit checks unresolved"
    (output / "analysis.json").write_text(
        json.dumps(info, indent=2, allow_nan=False) + "\n"
    )
    rows = []
    for group in ensembles:
        row = {
            "L": group["L"],
            "temperature": group["T"],
            "chains": len(group["chains"]),
        }
        for i, name in enumerate(NAMES):
            row[name] = group["central"][i]
            row[name + "_error"] = group["errors"][i]
        rows.append(row)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    benchmark_rows = []
    for run, report in zip(runs, reports):
        sampling = run.config["sampling"]
        total_sweeps = sum(
            sampling[k]
            for k in ("adaptation_sweeps", "equilibration_sweeps", "measurement_sweeps")
        )
        benchmark_rows.append(
            {
                "L": report["L"],
                "temperature": report["T"],
                "start": report["start"],
                "seed": report["seed"],
                "elapsed_seconds": report["elapsed_seconds"],
                "total_sweeps": total_sweeps,
                "total_sweeps_per_second": total_sweeps / report["elapsed_seconds"],
                "slowest_tau_sweeps": max(
                    v["tau_sweeps"] for v in report["autocorrelation"].values()
                ),
                "slowest_effective_samples_per_second": report[
                    "slowest_effective_samples_per_second"
                ],
                "warning_count": len(report["warnings"]),
            }
        )
    with (output / "benchmark.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(benchmark_rows[0]))
        writer.writeheader()
        writer.writerows(benchmark_rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    for L in sorted({g["L"] for g in ensembles}):
        groups = [g for g in ensembles if g["L"] == L]
        for ax, name in zip(axes, ("B4", "abs_m", "chi_abs")):
            col = NAMES.index(name)
            ax.errorbar(
                [g["T"] for g in groups],
                [g["central"][col] for g in groups],
                yerr=[g["errors"][col] for g in groups],
                marker="o",
                label=f"L={L}",
                capsize=2,
            )
            ax.set(xlabel="Temperature", ylabel=name)
    axes[0].legend()
    fig.suptitle("Independent Metropolis chains - preliminary block errors")
    fig.savefig(output / "observables.png", dpi=180)
    fig.savefig(output / "observables.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(
        len(runs),
        2,
        figsize=(11, max(3, 1.5 * len(runs))),
        squeeze=False,
        layout="constrained",
    )
    for run, ax in zip(runs, axes):
        stride = max(1, len(run.energy) // 2000)
        ax[0].plot(run.sweeps[::stride], run.energy[::stride], linewidth=0.4)
        ax[1].plot(
            run.sweeps[::stride], np.abs(run.magnetization[::stride]), linewidth=0.4
        )
        label = f"L={run.config['model']['L']} T={run.config['model']['temperature']} {run.config['sampling']['start']}"
        ax[0].set_ylabel(label, fontsize=6)
    axes[-1, 0].set_xlabel("Sweep (production; burn-in already removed)")
    axes[0, 0].set_title("Energy per site")
    axes[0, 1].set_title("Absolute magnetization")
    fig.savefig(output / "histories.png", dpi=120)
    plt.close(fig)
    if "scaling" in info:
        result = info["scaling"]
        fit = result["fit"]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for ax, name in zip(axes, ("B4", "abs_m", "chi_abs")):
            sizes = np.array(fit["sizes"], dtype=float)
            y = np.array(fit["values_at_Tc"][name])
            model = fit["power_fits"][name]
            ax.loglog(sizes, y, "o", label="Interpolated at crossing")
            ax.loglog(
                sizes,
                model["amplitude"] * sizes ** model["slope"],
                "--",
                label="Working power fit",
            )
            ax.set(xlabel="L", ylabel="d B4 / dT" if name == "B4" else name)
        axes[0].legend(fontsize=7)
        fig.suptitle(
            "Preliminary scaling"
            if result["fit_usable"]
            else "Diagnostic candidate - local fit fails quality checks"
        )
        fig.savefig(output / "scaling.png", dpi=180)
        fig.savefig(output / "scaling.pdf")
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
        for L in fit["sizes"]:
            groups = [g for g in ensembles if g["L"] == L]
            x = [(g["T"] - fit["Tc"]) * L ** (1 / fit["nu"]) for g in groups]
            for ax, name, power in zip(
                axes,
                ("B4", "abs_m", "chi_abs"),
                (0, fit["beta_over_nu"], -fit["gamma_over_nu"]),
            ):
                col = NAMES.index(name)
                ax.errorbar(
                    x,
                    [g["central"][col] * L**power for g in groups],
                    yerr=[g["errors"][col] * L**power for g in groups],
                    marker="o",
                    capsize=2,
                    label=f"L={L}",
                )
                ax.set(xlabel="(T - Tc) L^(1/nu)", ylabel=f"{name} L^({power:.3f})")
        axes[0].legend(fontsize=7)
        fig.suptitle(
            "Preliminary data collapse"
            if result["fit_usable"]
            else "Diagnostic collapse - candidate fit fails quality checks"
        )
        fig.savefig(output / "collapse.png", dpi=180)
        fig.savefig(output / "collapse.pdf")
        plt.close(fig)
    print(
        json.dumps(
            {
                "output": str(output),
                "chains": len(reports),
                "warnings": sum(bool(r["warnings"]) for r in reports),
                "scaling_status": info.get("scaling", {}).get("status"),
                "scaling": info.get("scaling", {}).get("fit")
                if info.get("scaling", {}).get("fit_usable")
                else None,
                "scaling_withheld": info.get("scaling_withheld"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
