"""Review production-half differences by permutation of joint moment blocks.

This assumes the retained blocks are approximately exchangeable under the null.
It does not establish equilibration or override the original screening flags.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def estimates(moments, N, T):
    moments = np.asarray(moments)
    return np.stack(
        (
            moments[..., 0],
            moments[..., 2],
            moments[..., 4] / moments[..., 3] ** 2,
            N / T * (moments[..., 3] - moments[..., 2] ** 2),
        ),
        axis=-1,
    )


def holm(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    adjusted[order] = np.minimum(
        1,
        np.maximum.accumulate(
            pvalues[order] * (len(pvalues) - np.arange(len(pvalues)))
        ),
    )
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=541001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.permutations < 20000:
        parser.error("Use at least 20000 permutations for the 560-test family")
    if args.output.exists():
        parser.error("Output already exists")
    meta = json.loads(
        (args.analysis / "block_moments.json").read_text(encoding="utf-8")
    )
    rng = np.random.default_rng(args.seed)
    rows = []
    partitions = {}
    with np.load(args.analysis / "block_moments.npz", allow_pickle=False) as saved:
        for group in meta["ensembles"]:
            for label, key in zip(group["chains"], group["block_arrays"]):
                blocks = saved[key]
                n = len(blocks)
                if n % 2 or n < 40:
                    raise ValueError(
                        "Require an even count of at least forty blocks per chain"
                    )
                half = n // 2
                if n not in partitions:
                    partitions[n] = np.argsort(
                        rng.random((args.permutations, n)), axis=1
                    )[:, :half]
                first = blocks[partitions[n]].mean(axis=1)
                second = (blocks.sum(axis=0) - first * half) / half
                observed = estimates(
                    blocks[:half].mean(axis=0), group["L"] ** 3, group["T"]
                ) - estimates(blocks[half:].mean(axis=0), group["L"] ** 3, group["T"])
                permuted = estimates(first, group["L"] ** 3, group["T"]) - estimates(
                    second, group["L"] ** 3, group["T"]
                )
                p = (1 + (np.abs(permuted) >= np.abs(observed)).sum(axis=0)) / (
                    args.permutations + 1
                )
                for i, name in enumerate(("e", "abs_m", "B4", "chi_abs")):
                    rows.append(
                        {
                            "chain": label,
                            "L": group["L"],
                            "T": group["T"],
                            "observable": name,
                            "first_minus_second": float(observed[i]),
                            "permutation_p": float(p[i]),
                        }
                    )
    adjusted = holm([r["permutation_p"] for r in rows])
    for row, p in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(p)
    result = {
        "tests": len(rows),
        "permutations": args.permutations,
        "seed": args.seed,
        "block_size_samples": meta["block_size_samples"],
        "significant_after_holm_005": [r for r in rows if r["holm_adjusted_p"] < 0.05],
        "results": rows,
        "input_sha256": {
            name: hashlib.sha256((args.analysis / name).read_bytes()).hexdigest()
            for name in ("block_moments.json", "block_moments.npz")
        },
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Exploratory review of all chain/observable half comparisons, with Holm correction across the entire family. Approximate block exchangeability is required. No data excluded; original warnings retained. Failure to reject does not prove stationarity or equilibrium.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "tests": result["tests"],
                "significant_after_holm_005": result["significant_after_holm_005"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
