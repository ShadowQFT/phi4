"""Single-chain diagnostics; these do not certify equilibrium automatically."""

import numpy as np

from .statistics import block_jackknife, observables


def blocking_curve(energy, magnetization, N, T):
    """Compare powers-of-two block lengths while retaining at least 20 blocks."""
    results = []
    block_size = 1
    while len(energy) // block_size >= 20:
        results.append(block_jackknife(energy, magnetization, N, T, block_size))
        block_size *= 2
    return results


def split_history(energy, magnetization, N, T, block_size):
    """Report chronological halves without choosing a favorable burn-in cut."""
    midpoint = len(energy) // 2
    halves = []
    for name, selection in (
        ("first", slice(0, midpoint)),
        ("second", slice(midpoint, None)),
    ):
        e, m = energy[selection], magnetization[selection]
        result = {
            "half": name,
            "sample_count": len(e),
            "observables": observables(e, m, N, T),
        }
        if len(e) // block_size >= 20:
            result["block_jackknife"] = block_jackknife(e, m, N, T, block_size)
        else:
            result["uncertainty_unavailable"] = (
                "Fewer than 20 blocks in this half at the selected block length"
            )
        halves.append(result)
    # A raw difference is descriptive; no independence-based z-test is implied.
    difference = {
        key: halves[1]["observables"][key] - halves[0]["observables"][key]
        for key in ("e", "abs_m", "m2", "m4", "B4")
    }
    return {
        "halves": halves,
        "second_minus_first": difference,
        "interpretation": "Inspect histories and block stability; no automatic equilibrium verdict",
    }


def plot_diagnostics(runs, results, output, analysis_id):
    """Write labelled PNG/PDF figures with a headless Matplotlib backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 160,
        }
    )
    model = runs[0].config["model"]
    synthetic = all(
        run.metadata.get("dataset_kind") == "synthetic_validation" for run in runs
    )
    prefix = "Synthetic validation: " if synthetic else ""
    description = (
        f"L={model['L']}, T={model['temperature']:g}; "
        f"J={model['J']:g}, r={model['r']:g}, u={model['u']:g}"
    )
    artifacts = []

    def export(figure, name, caption):
        figure.text(0.02, 0.01, f"Analysis: {analysis_id}\n{caption}", fontsize=8)
        figure.tight_layout(rect=(0, 0.08, 1, 0.94))
        for extension in ("png", "pdf"):
            filename = f"{name}.{extension}"
            figure.savefig(output / filename)
            artifacts.append(filename)
        plt.close(figure)

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for index, (run, result) in enumerate(zip(runs, results)):
        color = f"C{index % 10}"
        equilibration = run.equilibration
        label = (
            f"{run.config['sampling']['start']}, seed {run.config['sampling']['seed']}"
        )
        eq_m = equilibration["magnetization"]
        for axis, eq_values, values, ylabel in zip(
            axes,
            (equilibration["energy_per_site"], np.abs(eq_m), eq_m**2),
            (run.energy, np.abs(run.magnetization), run.magnetization**2),
            (
                "Energy per site e",
                "Absolute magnetization |m|",
                "Squared magnetization m²",
            ),
        ):
            axis.plot(
                equilibration["sweep"],
                eq_values,
                color=color,
                alpha=0.35,
                linewidth=0.6,
            )
            axis.plot(
                run.sweeps, values, color=color, linewidth=0.7, alpha=0.85, label=label
            )
            axis.axvline(
                equilibration["sweep"][-1], color=color, linestyle="--", alpha=0.8
            )
            if result["discarded_production_samples"]:
                axis.axvline(result["first_selected_sweep"], color=color, linestyle=":")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.15)
    axes[0].legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Sweep (N attempted site updates)")
    figure.suptitle(f"{prefix}Chain histories — {description}")
    export(
        figure,
        "histories",
        "Dashed: end of burn-in; dotted: explicit additional cut if requested. No equilibrium claim.",
    )

    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    for run, result in zip(runs, results):
        curve = result["blocking"]
        label = (
            f"{run.config['sampling']['start']}, seed {run.config['sampling']['seed']}"
        )
        for axis, name in zip(axes.flat, ("e", "abs_m", "m2", "B4")):
            axis.plot(
                [point["block_size"] for point in curve],
                [point["standard_error"][name] for point in curve],
                marker="o",
                markersize=3,
                label=label,
            )
            axis.axvline(
                result["block_jackknife"]["block_size"],
                color="0.6",
                linestyle=":",
                linewidth=0.8,
            )
            axis.set_xscale("log", base=2)
            axis.set_xlabel("Block length (retained samples)")
            axis.set_ylabel(f"Jackknife standard error: {name}")
            axis.grid(alpha=0.15)
    axes.flat[0].legend(fontsize=8)
    figure.suptitle(f"{prefix}Block-length sensitivity — {description}")
    export(
        figure,
        "blocking",
        "Each point retains ≥20 complete blocks; incomplete tails are excluded and recorded. Inspect for a plateau.",
    )
    return artifacts
