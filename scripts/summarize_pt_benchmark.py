"""Summarize matched PT benchmark efficiency using actual Slurm CPU allocation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analyses", type=Path, nargs="+")
    parser.add_argument("--accounting", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=Path("cluster/manifests"))
    args = parser.parse_args()
    accounting = {}
    with args.accounting.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="|"):
            if "." not in row["JobID"]:
                accounting[row["JobID"]] = row
    rows = []
    for path in args.analyses:
        analysis = json.loads(path.read_text(encoding="utf-8"))
        for report in analysis["ladders"]:
            raw = Path(report["folder"])
            metadata = json.loads((raw / "metadata.json").read_text())
            campaign = metadata["config"]["output"]["campaign"]
            submission = json.loads(
                (args.manifests / campaign / "submission.json").read_text()
            )
            job = submission["job_id"].split(";")[0]
            account = accounting[job]
            if account["State"] != "COMPLETED" or account["ExitCode"] != "0:0":
                raise ValueError("Non-successful benchmark accounting")
            hours = int(account["ElapsedRaw"]) * int(account["AllocCPUS"]) / 3600
            critical = report["nearest_temperature_benchmark"]
            ess = min(
                critical["blocking_effective_samples"][name]
                for name in ("e", "abs_m", "m2", "m4")
            )
            rows.append(
                {
                    "L": report["L"],
                    "seed": metadata["config"]["sampling"]["seed"],
                    "swap_every": report["swap_every_sweeps"],
                    "job_id": job,
                    "wall_seconds": int(account["ElapsedRaw"]),
                    "allocated_cpu_hours": hours,
                    "temperature": critical["temperature"],
                    "blocking_effective_samples": ess,
                    "blocking_ess_per_cpu_hour": ess / hours,
                    "acf_ess_per_cpu_hour": critical["limiting_effective_samples"]
                    / hours,
                    "minimum_round_trips": min(report["round_trips"]),
                    "total_round_trips": sum(report["round_trips"]),
                    "critical_warning_count": len(critical["warnings"]),
                }
            )
    baselines = {(r["L"], r["seed"]): r for r in rows if r["swap_every"] == 0}
    for row in rows:
        baseline = baselines[row["L"], row["seed"]]
        row["gain_vs_matched_baseline"] = (
            row["blocking_ess_per_cpu_hour"] / baseline["blocking_ess_per_cpu_hour"]
        )
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "efficiency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*args.analyses, args.accounting]
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": rows,
        "interpretation": "Same temperature grid and local sweeps within each paired seed. ALL replica CPU allocation is counted. Block-based estimates use the largest error among checked block lengths and remain uncertain. Results do not compare against concentrating all cores at a single temperature. No universal speedup or equilibration claim.",
    }
    (args.output / "benchmark.json").write_text(json.dumps(provenance, indent=2) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    sizes = sorted({r["L"] for r in rows})
    cadences = sorted({r["swap_every"] for r in rows if r["swap_every"]})
    width = 0.32
    for i, L in enumerate(sizes):
        for j, cadence in enumerate(cadences):
            gains = [
                r["gain_vs_matched_baseline"]
                for r in rows
                if r["L"] == L and r["swap_every"] == cadence
            ]
            x = j + (i - (len(sizes) - 1) / 2) * width
            ax.bar(
                x,
                np.exp(np.mean(np.log(gains))),
                width=width,
                color=f"C{i}",
                label=f"L={L}" if j == 0 else None,
                alpha=0.75,
            )
            offsets = np.linspace(-0.05, 0.05, len(gains))
            ax.scatter(x + offsets, gains, color="black", s=22, zorder=3)
    ax.axhline(1, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(cadences)), [str(x) for x in cadences])
    ax.set(
        xlabel="Local sweeps between exchange passes",
        ylabel="Effective samples per CPU-hour / matched control",
        title="Parallel tempering at T=15.32: blocking-based pilot comparison",
    )
    ax.legend()
    fig.savefig(args.output / "efficiency.png", dpi=180)
    fig.savefig(args.output / "efficiency.pdf")
    plt.close(fig)
    print(
        json.dumps(
            {
                "benchmarks": len(rows),
                "total_allocated_cpu_hours": sum(
                    r["allocated_cpu_hours"] for r in rows
                ),
                "L32_gain_range": [
                    min(
                        r["gain_vs_matched_baseline"]
                        for r in rows
                        if r["L"] == 32 and r["swap_every"]
                    ),
                    max(
                        r["gain_vs_matched_baseline"]
                        for r in rows
                        if r["L"] == 32 and r["swap_every"]
                    ),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
