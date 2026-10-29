"""Finite-size fits with synchronized, whole-ladder PT resampling."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "analysis"))
from phi4_analysis.tempering import (  # noqa: E402
    bootstrap_finite_size,
    load_ladder,
    synchronized_blocks,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=1000)
    parser.add_argument("--tmin", type=float, required=True)
    parser.add_argument("--tmax", type=float, required=True)
    parser.add_argument("--degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--lmin", type=int, default=0)
    parser.add_argument("--crossing-sizes", type=int, nargs=2)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=544000)
    args = parser.parse_args()
    entries = []
    provenance = []
    couplings = set()
    intervals = set()
    for folder in args.folders:
        ladder = load_ladder(folder)
        model = ladder["config"]["model"]
        couplings.add(tuple(model[k] for k in ("name", "boundary", "J", "r", "u")))
        intervals.add(ladder["config"]["sampling"]["measure_every_sweeps"])
        selected = (ladder["temperatures"] >= args.tmin) & (
            ladder["temperatures"] <= args.tmax
        )
        blocks = synchronized_blocks(
            ladder["energy"][:, selected],
            ladder["magnetization"][:, selected],
            args.block_size,
        )
        entries.append(
            {
                "L": model["L"],
                "temperatures": ladder["temperatures"][selected],
                "blocks": blocks,
                "replica_seeds": ladder["metadata"]["replica_seeds"],
            }
        )
        provenance.append(
            {"folder": str(ladder["folder"]), "input_hashes": ladder["input_hashes"]}
        )
    if len(couplings) != 1 or len(intervals) != 1:
        raise ValueError(
            "FSS ladders need identical fixed couplings and measurement intervals"
        )
    result = bootstrap_finite_size(
        entries,
        degree=args.degree,
        lmin=args.lmin,
        crossing_sizes=args.crossing_sizes,
        replicates=args.replicates,
        seed=args.seed,
    )
    result["inputs"] = provenance
    result["controls"] = {
        "tmin": args.tmin,
        "tmax": args.tmax,
        "block_size": args.block_size,
        "degree": args.degree,
        "lmin": args.lmin,
    }
    result["source_sha256"] = {
        name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
        for name in (
            "scripts/fit_pt.py",
            "analysis/phi4_analysis/tempering.py",
            "analysis/phi4_analysis/scaling.py",
            "analysis/phi4_analysis/statistics.py",
        )
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Output already exists")
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "fit": {k: result["fit"][k] for k in ("Tc", "nu", "beta", "gamma")},
                "intervals": result["percentile_interval_95"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
