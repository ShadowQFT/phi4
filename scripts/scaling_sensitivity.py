"""Refit saved joint moment blocks across predeclared analysis choices."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from phi4_analysis.scaling import bootstrap_scaling  # noqa: E402
from phi4_analysis.statistics import _observables_from_moments  # noqa: E402


def reblock(group, factor):
    """Merge adjacent base blocks within each chain, preserving joint moments."""
    if type(factor) is not int or factor < 1:
        raise ValueError("Reblocking factor must be a positive integer")
    result = dict(group)
    result["blocks"] = []
    dropped = []
    for blocks in group["blocks"]:
        count = len(blocks) // factor
        if count < 20:
            raise ValueError("Require at least twenty merged blocks per chain")
        kept = count * factor
        result["blocks"].append(blocks[:kept].reshape(count, factor, 6).mean(axis=1))
        dropped.append(len(blocks) - kept)
    total = np.concatenate(result["blocks"])
    L, T = group["L"], group["T"]
    result["central"] = _observables_from_moments(total.mean(axis=0), L**3, T)
    count = len(total)
    summed = total.sum(axis=0)
    jk = np.array(
        [_observables_from_moments((summed - b) / (count - 1), L**3, T) for b in total]
    )
    result["errors"] = np.sqrt(
        (count - 1) / count * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0)
    )
    result["dropped_base_blocks"] = dropped
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=540928)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.replicates < 100:
        parser.error("Require at least 100 bootstrap replicates")
    info = json.loads(
        (args.analysis / "block_moments.json").read_text(encoding="utf-8")
    )
    groups = []
    with np.load(args.analysis / "block_moments.npz", allow_pickle=False) as saved:
        for row in info["ensembles"]:
            groups.append(
                {
                    "L": row["L"],
                    "T": row["T"],
                    "central": np.array(row["central"]),
                    "errors": np.array(row["errors"]),
                    "blocks": [saved[key] for key in row["block_arrays"]],
                }
            )
    # These cases are recorded whether or not their results agree with literature.
    cases = [
        ("full_quadratic", 15.20, 15.40, 2, 0, 1, None),
        ("central_quadratic", 15.24, 15.36, 2, 0, 1, None),
        ("narrow_quadratic", 15.28, 15.36, 2, 0, 1, None),
        ("central_linear", 15.24, 15.36, 1, 0, 1, None),
        ("exclude_L8", 15.24, 15.36, 2, 12, 1, None),
        ("exclude_L8_L12", 15.24, 15.36, 2, 16, 1, None),
        ("crossing_16_24", 15.24, 15.36, 2, 0, 1, [16, 24]),
        ("crossing_16_32", 15.24, 15.36, 2, 0, 1, [16, 32]),
        ("crossing_8_16", 15.24, 15.36, 2, 0, 1, [8, 16]),
        ("crossing_12_24", 15.24, 15.36, 2, 0, 1, [12, 24]),
        ("double_block", 15.24, 15.36, 2, 0, 2, None),
    ]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, tmin, tmax, degree, lmin, factor, pair in cases:
        selected = [
            reblock(g, factor) if factor > 1 else g
            for g in groups
            if tmin <= g["T"] <= tmax and g["L"] >= lmin
        ]
        controls = {
            "temperature_window": [tmin, tmax],
            "degree": degree,
            "lmin": lmin,
            "block_size_samples": info["block_size_samples"] * factor,
            "crossing_sizes": pair,
        }
        try:
            result = bootstrap_scaling(
                selected,
                degree=degree,
                lmin=lmin,
                replicates=args.replicates,
                seed=args.seed,
                crossing_sizes=pair,
            )
            result["controls"] = controls
            results[name] = result
            print(
                name,
                result["status"],
                {k: round(result["fit"][k], 5) for k in ("Tc", "nu", "beta", "gamma")},
                flush=True,
            )
        except ValueError as error:
            results[name] = {
                "controls": controls,
                "status": "withheld",
                "reason": str(error),
            }
            print(name, "withheld:", error, flush=True)
        (output / "sensitivity.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    provenance = {
        "input_sha256": {
            name: hashlib.sha256((args.analysis / name).read_bytes()).hexdigest()
            for name in ("analysis.json", "block_moments.json", "block_moments.npz")
        },
        "source_sha256": {
            name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
            for name in (
                "scripts/scaling_sensitivity.py",
                "analysis/phi4_analysis/scaling.py",
                "analysis/phi4_analysis/statistics.py",
            )
        },
        "seed": args.seed,
        "replicates": args.replicates,
        "sampling_status": "See original analysis.json; sensitivity fits do not override its chain warnings or establish equilibrium.",
        "note": "Each case is retained, including local-fit rejections and bootstrap failures. These are effective finite-size fits, without a correction-to-scaling extrapolation.",
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
