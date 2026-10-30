"""Create report figures and CSV tables from a completed PT production review."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullLocator

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "analysis"))
from phi4_analysis.statistics import NAMES  # noqa: E402
from review_pt_production import block_estimates, reblock  # noqa: E402


def save(fig, output, name):
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{name}.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    diagnostics = json.loads(
        (args.review / "diagnostics.json").read_text(encoding="utf-8")
    )
    fits = json.loads((args.review / "fits.json").read_text(encoding="utf-8"))[
        "variants"
    ]
    central = next(x["result"] for x in fits if x["label"] == "central")
    sizes = sorted({x["L"] for x in diagnostics["ladders"]})
    colors = dict(zip(sizes, plt.get_cmap("tab10").colors))
    curves = []
    for L in sizes:
        blocks = []
        for i, r in enumerate(diagnostics["ladders"]):
            if r["L"] == L:
                with np.load(args.analysis / f"ladder-{i}-blocks.npz") as cache:
                    blocks.append(reblock(cache["joint_moment_blocks"], 5))
                    temperatures = cache["temperatures"]
        estimate, error = block_estimates(np.concatenate(blocks), L, temperatures)
        curves.append(dict(L=L, T=temperatures, estimate=estimate, error=error))
    with (args.output / "observables.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["L", "T"] + [v for name in NAMES for v in (name, name + "_SE")]
        )
        for c in curves:
            for T, estimate, error in zip(c["T"], c["estimate"], c["error"]):
                writer.writerow(
                    [c["L"], T] + [v for pair in zip(estimate, error) for v in pair]
                )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3), layout="constrained")
    panels = [
        ("binder", r"Normalized Binder $(3-B_4)/2$", False),
        ("chi_abs", r"$\chi_{|m|}$", False),
        ("binder", r"Normalized Binder $(3-B_4)/2$", True),
        ("abs_m", r"$\langle |m|\rangle$", True),
    ]
    for ax, (name, label, zoom) in zip(axes.flat, panels):
        col = NAMES.index(name)
        for c in curves:
            take = (
                (c["T"] >= 15.26) & (c["T"] <= 15.38)
                if zoom
                else np.ones(len(c["T"]), dtype=bool)
            )
            ax.errorbar(
                c["T"][take],
                c["estimate"][take, col],
                yerr=c["error"][take, col],
                marker="o",
                markersize=3,
                capsize=2,
                linewidth=1,
                color=colors[c["L"]],
                label=f"L={c['L']}",
            )
        ax.set(xlabel="Temperature", ylabel=label)
        if zoom:
            interval = central["percentile_interval_95"]["Tc"]
            ax.axvspan(*interval, color="0.5", alpha=0.13)
            ax.axvline(central["fit"]["Tc"], color="0.4", linestyle=":", linewidth=1)
        if name == "binder" and not zoom:
            ax.set_ylim(-0.05, 1.05)
            for y in (0, 1):
                ax.axhline(y, color="0.7", linewidth=0.6, linestyle=":")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(
        "Clean 3D phi4: parallel tempering, two independent starts per size\nPreliminary; error bars are block-jackknife standard errors (25,000 sweeps per block)"
    )
    save(fig, args.output, "production-observables")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout="constrained")
    fit = central["fit"]
    for ax, name, label in zip(
        axes,
        ("B4", "abs_m", "chi_abs"),
        (r"$|dB_N/dT|$", r"$\langle |m|\rangle(T_c)$", r"$\chi_{|m|}(T_c)$"),
    ):
        factor = 0.5 if name == "B4" else 1
        values = np.array(fit["values_at_Tc"][name]) * factor
        ax.loglog(fit["sizes"], values, "o", color="black", label="Interpolated values")
        parameters = fit["power_fits"][name]
        grid = np.geomspace(min(sizes), max(sizes), 100)
        ax.loglog(
            grid,
            factor * parameters["amplitude"] * grid ** parameters["slope"],
            color="tab:blue",
            label="Power fit",
        )
        ax.set(xlabel="L", ylabel=label, xticks=sizes)
        ax.set_xticklabels(sizes)
        ax.xaxis.set_minor_locator(NullLocator())
        ax.text(
            0.05, 0.08, f"slope = {parameters['slope']:.3f}", transform=ax.transAxes
        )
    fig.suptitle(
        f"Effective finite-size scaling at crossing T = {fit['Tc']:.5f}\nCentral values shown; joint parameter intervals are in the sensitivity figure"
    )
    save(fig, args.output, "finite-size-scaling")
    fig, axes = plt.subplots(1, 3, figsize=(12, 5), layout="constrained", sharey=True)
    nu = 0.63002
    eta = 0.03627
    references = dict(nu=nu, beta=nu * (1 + eta) / 2, gamma=nu * (2 - eta))
    labels = [x["label"] for x in fits]
    for ax, name in zip(axes, ("nu", "beta", "gamma")):
        for index, row in enumerate(fits):
            result = row["result"]
            if "fit" not in result:
                continue
            value = result["fit"][name]
            low, high = result["percentile_interval_95"][name]
            flagged = bool(result["quality_warnings"])
            ax.errorbar(
                value,
                index,
                xerr=[[value - low], [high - value]],
                fmt="x" if flagged else "o",
                color="tab:red" if flagged else "tab:blue",
                capsize=3,
                markersize=4,
            )
        ax.axvline(
            references[name], color="black", linestyle=":", label="Hasenbusch reference"
        )
        ax.set(xlabel=name, yticks=range(len(labels)), yticklabels=labels)
        ax.grid(axis="x", alpha=0.15)
    axes[0].invert_yaxis()
    fig.suptitle(
        "Fit sensitivity: 95% joint block-bootstrap intervals\nRed cross: failed local-fit check; finite-size/systematic uncertainty is additional"
    )
    save(fig, args.output, "exponent-sensitivity")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for r in diagnostics["ladders"]:
        j = min(
            range(len(r["temperatures"])),
            key=lambda j: abs(r["temperatures"][j] - 15.32),
        )
        for ax, name in zip(axes, ("B4", "chi_abs")):
            col = NAMES.index(name)
            baseline = r["blocks"]["1000"]["standard_errors"][j][col]
            lengths = [int(k) for k in r["blocks"]]
            ratios = [
                r["blocks"][str(k)]["standard_errors"][j][col] / baseline
                for k in lengths
            ]
            ax.plot(
                lengths,
                ratios,
                "o-" if r["start"] == "random" else "s--",
                color=colors[r["L"]],
                label=f"L={r['L']}, {r['start']}",
                markersize=3,
            )
            ax.set(
                xlabel="Block length (saved samples; multiply by 5 for sweeps)",
                ylabel=f"{name} SE / SE at 1,000 samples",
            )
            ax.axhline(1, color="0.7", linewidth=0.6)
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle(
        "Blocking check at T=15.32: residual error growth remains for some large-size runs"
    )
    save(fig, args.output, "blocking-check")
    with (args.output / "fit-sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["variant", "parameter", "estimate", "lower_95", "upper_95", "status"]
        )
        for row in fits:
            r = row["result"]
            if "fit" not in r:
                continue
            for name in ("Tc", "nu", "beta", "gamma", "hyperscaling"):
                writer.writerow(
                    [
                        row["label"],
                        name,
                        r["fit"][name],
                        *r["percentile_interval_95"][name],
                        r["status"],
                    ]
                )
    sources = [
        Path(__file__),
        REPO / "scripts/review_pt_production.py",
        args.review / "diagnostics.json",
        args.review / "fits.json",
    ]
    (args.output / "provenance.json").write_text(
        json.dumps(
            {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
