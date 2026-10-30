"""Review saved PT moment blocks and repeat finite-size fits without reloading raw CSVs.

Run analyze_pt.py first. Its integrity-checked provenance and joint moment blocks
are inputs here. Reblocking averages consecutive complete blocks, preserving
all temperature/moment covariance. Raw histories remain unchanged.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from phi4_analysis.statistics import NAMES, _observables_from_moments  # noqa: E402
from phi4_analysis.tempering import bootstrap_finite_size  # noqa: E402


def reblock(blocks, factor):
    if factor < 1 or len(blocks) % factor:
        raise ValueError("Reblocking must retain all original blocks")
    return blocks.reshape(len(blocks) // factor, factor, *blocks.shape[1:]).mean(axis=1)


def block_estimates(blocks, L, temperatures):
    count = len(blocks)
    if count < 20:
        raise ValueError("Require at least 20 time blocks")
    summed = blocks.sum(axis=0)
    central, errors = [], []
    for slot, T in enumerate(temperatures):
        central.append(_observables_from_moments(summed[slot] / count, L**3, float(T)))
        leave = np.array(
            [
                _observables_from_moments(
                    (summed[slot] - b[slot]) / (count - 1), L**3, float(T)
                )
                for b in blocks
            ]
        )
        errors.append(
            np.sqrt(
                (count - 1) / count * ((leave - leave.mean(axis=0)) ** 2).sum(axis=0)
            )
        )
    return np.array(central), np.array(errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    info = json.loads((args.analysis / "analysis.json").read_text(encoding="utf-8"))
    base = info["block_size"]
    entries, reports, provenance = [], [], []
    intervals = set()
    for i, report in enumerate(info["ladders"]):
        cache = args.analysis / f"ladder-{i}-blocks.npz"
        with np.load(cache) as data:
            blocks, temperatures = data["joint_moment_blocks"], data["temperatures"]
        metadata_path = Path(report["folder"]) / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        intervals.add(metadata["config"]["sampling"]["measure_every_sweeps"])
        entry = dict(
            L=report["L"],
            temperatures=temperatures,
            blocks=blocks,
            replica_seeds=metadata["replica_seeds"],
            start=metadata["config"]["sampling"]["start"],
        )
        entries.append(entry)
        checks = {}
        for length in (1000, 2000, 5000):
            if length % base:
                raise ValueError(
                    "Requested review block length must be a multiple of the cached base length"
                )
            b = reblock(blocks, length // base)
            central, error = block_estimates(b, entry["L"], temperatures)
            checks[str(length)] = dict(
                block_count=len(b),
                estimates=central.tolist(),
                standard_errors=error.tolist(),
            )
        reports.append(
            dict(
                L=entry["L"],
                start=entry["start"],
                folder=report["folder"],
                temperatures=temperatures.tolist(),
                blocks=checks,
                round_trips=report["round_trips"],
                min_swap_acceptance=min(report["swap_acceptance_by_pair"]),
                original_warnings=[
                    dict(
                        T=r["temperature"], warnings=r["warnings"], split_z=r["split_z"]
                    )
                    for r in report["temperatures"]
                    if r["warnings"]
                ],
            )
        )
        provenance.append(
            dict(
                folder=report["folder"],
                input_hashes=report["input_hashes"],
                metadata_sha256=hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                blocks_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
            )
        )
    comparisons = []
    selected_names = ("e", "abs_m", "B4", "chi_abs")
    for L in sorted({x["L"] for x in reports}):
        a, b = [r for r in reports if r["L"] == L]
        ac, bc = (np.array(r["blocks"]["5000"]["estimates"]) for r in (a, b))
        ae, be = (np.array(r["blocks"]["5000"]["standard_errors"]) for r in (a, b))
        z = abs(ac - bc) / np.hypot(ae, be)
        for j, T in enumerate(a["temperatures"]):
            comparisons.append(
                dict(
                    L=L,
                    T=T,
                    z_scores={
                        name: float(z[j, NAMES.index(name)]) for name in selected_names
                    },
                )
            )
    if len(intervals) != 1:
        raise ValueError("Require matching measurement intervals")
    summary = dict(
        observable_order=NAMES,
        block_lengths_are_saved_samples=True,
        sweeps_per_sample=intervals.pop(),
        ladders=reports,
        independent_start_comparisons=comparisons,
        inputs=provenance,
        interpretation="Z scores are unadjusted screening diagnostics, not independent hypothesis tests. Block-length variation is retained. Round trips and stationary ACF estimates alone do not establish equilibration.",
    )
    (args.output / "diagnostics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    variants = [
        ("central", 2000, 15.30, 15.34, 2, 0, (24, 48), None),
        ("shorter-blocks", 1000, 15.30, 15.34, 2, 0, (24, 48), None),
        ("longer-blocks", 5000, 15.30, 15.34, 2, 0, (24, 48), None),
        ("wider-window", 2000, 15.28, 15.36, 2, 0, (24, 48), None),
        ("linear", 2000, 15.30, 15.34, 1, 0, (24, 48), None),
        ("exclude-L16", 2000, 15.30, 15.34, 2, 24, (24, 48), None),
        ("crossing-16-32", 2000, 15.30, 15.34, 2, 0, (16, 32), None),
        ("crossing-32-48", 2000, 15.30, 15.34, 2, 0, (32, 48), None),
        ("random-only", 2000, 15.30, 15.34, 2, 0, (24, 48), "random"),
        ("ordered-only", 2000, 15.30, 15.34, 2, 0, (24, 48), "ordered_positive"),
    ]
    fits = []
    for label, length, low, high, degree, lmin, pair, start in variants:
        subset = []
        for entry in entries:
            if start and entry["start"] != start:
                continue
            take = (entry["temperatures"] >= low) & (entry["temperatures"] <= high)
            subset.append(
                dict(
                    L=entry["L"],
                    temperatures=entry["temperatures"][take],
                    blocks=reblock(entry["blocks"][:, take, :], length // base),
                    replica_seeds=entry["replica_seeds"],
                )
            )
        controls = dict(
            block_size_samples=length,
            tmin=low,
            tmax=high,
            degree=degree,
            lmin=lmin,
            crossing_sizes=pair,
            start=start,
        )
        try:
            fit = bootstrap_finite_size(
                subset,
                degree=degree,
                lmin=lmin,
                crossing_sizes=pair,
                replicates=args.replicates,
                seed=544000,
            )
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            fit = dict(status="withheld", reason=str(error))
        result = dict(label=label, controls=controls, result=fit)
        fits.append(result)
        (args.output / f"fit-{label}.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(
            label,
            {k: fit.get("fit", {}).get(k) for k in ("Tc", "nu", "beta", "gamma")},
            fit.get("quality_warnings", []),
            fit.get("reason", ""),
            flush=True,
        )
    provenance_files = [
        Path(__file__),
        REPO / "analysis/phi4_analysis/tempering.py",
        REPO / "analysis/phi4_analysis/scaling.py",
        REPO / "analysis/phi4_analysis/statistics.py",
        args.analysis / "analysis.json",
    ]
    (args.output / "fits.json").write_text(
        json.dumps(
            dict(
                variants=fits,
                inputs=provenance,
                source_sha256={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in provenance_files
                },
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
