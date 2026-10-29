"""Integrity-checked synchronized PT ladders and whole-ladder block resampling."""

import hashlib
import json
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np

from .scaling import autocorrelation
from .statistics import NAMES, _block_means, _observables_from_moments, block_jackknife


def load_ladder(folder):
    folder = Path(folder).resolve(strict=True)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if (
        not (folder / "SUCCESS").is_file()
        or (folder / "FAIL").exists()
        or metadata.get("status") != "SUCCESS"
        or metadata.get("exit_code") != 0
        or metadata.get("sampler") != "parallel_tempering"
    ):
        raise ValueError("Require a completed PT ladder")
    required = {
        "config.resolved.toml",
        "temperatures.txt",
        "measurements.csv",
        "equilibration.csv",
        "swaps.csv",
        "walks.csv",
        "pt_summary.json",
        "source.zip",
        "stdout.log",
        "stderr.log",
    }
    if set(metadata.get("output_hashes", {})) != required:
        raise ValueError("Incomplete PT raw-file hash manifest")
    hashes = {}
    for name, expected in metadata["output_hashes"].items():
        path = folder / name
        if path.resolve().parent != folder:
            raise ValueError("Unexpected raw path")
        with path.open("rb") as stream:
            hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()
        if hashes[name] != expected:
            raise ValueError("PT raw hash mismatch: " + name)
    config = tomllib.loads(
        (folder / "config.resolved.toml").read_text(encoding="utf-8")
    )
    if config != metadata["config"]:
        raise ValueError("Resolved PT configuration mismatch")
    temperatures = np.array(config["tempering"]["temperatures"], dtype=float)
    sampling = config["sampling"]
    K = len(temperatures)
    n = sampling["measurement_sweeps"] // sampling["measure_every_sweeps"]
    data = np.loadtxt(folder / "measurements.csv", delimiter=",", skiprows=1, ndmin=2)
    if data.shape != (n * K, 7) or not np.all(np.isfinite(data)):
        raise ValueError("Invalid PT sample table")
    data = data.reshape(n, K, 7)
    if not (
        np.array_equal(data[:, :, 0], np.broadcast_to(np.arange(n)[:, None], (n, K)))
        and np.array_equal(data[:, :, 2], np.broadcast_to(np.arange(K), (n, K)))
        and np.array_equal(data[:, :, 3], np.broadcast_to(temperatures, (n, K)))
    ):
        raise ValueError("PT sample/temperature routing mismatch")
    expected = (
        sampling["adaptation_sweeps"]
        + sampling["equilibration_sweeps"]
        + (np.arange(n) + 1) * sampling["measure_every_sweeps"]
    )
    if not np.array_equal(data[:, :, 1], np.broadcast_to(expected[:, None], (n, K))):
        raise ValueError("PT sweep counters mismatch")
    if not np.array_equal(
        np.sort(data[:, :, 4], axis=1), np.broadcast_to(np.arange(K), (n, K))
    ):
        raise ValueError("PT replica routing is not a permutation")
    return {
        "folder": folder,
        "metadata": metadata,
        "config": config,
        "temperatures": temperatures,
        "energy": data[:, :, 5].copy(),
        "magnetization": data[:, :, 6].copy(),
        "input_hashes": hashes,
    }


def synchronized_blocks(energy, magnetization, block_size):
    energy = np.asarray(energy)
    magnetization = np.asarray(magnetization)
    if energy.ndim != 2 or energy.shape != magnetization.shape:
        raise ValueError("Require time by temperature arrays")
    # One block index addresses the same sweep interval at every temperature.
    return np.stack(
        [
            _block_means(energy[:, slot], magnetization[:, slot], block_size)[0]
            for slot in range(energy.shape[1])
        ],
        axis=1,
    )


def draw_ladder(blocks, rng):
    indices = rng.integers(0, len(blocks), size=len(blocks))
    return blocks[indices].mean(axis=0)


def ladder_bootstrap(blocks, N, temperatures, *, replicates=2000, seed=544000):
    if len(blocks) < 20 or replicates < 100:
        raise ValueError("Insufficient blocks or bootstrap replicates")
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(replicates):
        moments = draw_ladder(blocks, rng)
        samples.append(
            [
                _observables_from_moments(moments[slot], N, float(T))
                for slot, T in enumerate(temperatures)
            ]
        )
    return np.array(samples)


def analyze_ladder(ladder, block_size, replicates=2000, seed=544000):
    config = ladder["config"]
    metadata = ladder["metadata"]
    N = config["model"]["L"] ** 3
    blocks = synchronized_blocks(ladder["energy"], ladder["magnetization"], block_size)
    samples = ladder_bootstrap(
        blocks, N, ladder["temperatures"], replicates=replicates, seed=seed
    )
    elapsed = (
        datetime.fromisoformat(metadata["finished_utc"])
        - datetime.fromisoformat(metadata["started_utc"])
    ).total_seconds()
    allocated_cpu_seconds = elapsed * metadata["ranks"]
    rows = []
    for slot, T in enumerate(ladder["temperatures"]):
        e = ladder["energy"][:, slot]
        m = ladder["magnetization"][:, slot]
        stats = block_jackknife(e, m, N, float(T), block_size)
        correlations = {
            name: autocorrelation(values)
            for name, values in (
                ("e", e),
                ("abs_m", abs(m)),
                ("m2", m * m),
                ("m4", m**4),
                ("m", m),
            )
        }
        for value in correlations.values():
            value["tau_sweeps"] = (
                value["tau_samples"] * config["sampling"]["measure_every_sweeps"]
            )
        primary = [correlations[name] for name in ("e", "abs_m", "m2", "m4")]
        warnings = []
        if any(not value["tail_resolved"] for value in primary):
            warnings.append("Autocorrelation tail may be unresolved")
        if block_size < 10 * max(value["tau_samples"] for value in primary):
            warnings.append("Block shorter than ten times slowest estimated tau")
        plateau = {}
        for length in (max(1, block_size // 2), block_size, block_size * 2):
            if len(m) // length >= 20:
                plateau[str(length)] = block_jackknife(e, m, N, float(T), length)[
                    "standard_error"
                ]
        retained = len(blocks) * block_size
        blocking_ess = {}
        for name, values in (("e", e), ("abs_m", abs(m)), ("m2", m * m), ("m4", m**4)):
            largest_error = max(errors[name] for errors in plateau.values())
            blocking_ess[name] = float(
                np.var(values[:retained], ddof=1) / largest_error**2
            )
        retained_m = m[:retained]
        mean_m2 = float(np.mean(retained_m**2))
        mean_m4 = float(np.mean(retained_m**4))
        influence = (
            retained_m**4 - mean_m4
        ) / mean_m2**2 - 2 * mean_m4 / mean_m2**3 * (retained_m**2 - mean_m2)
        blocking_ess["B4_delta_method"] = float(
            np.var(influence, ddof=1)
            / max(errors["B4"] for errors in plateau.values()) ** 2
        )
        half_block = min(block_size, len(m) // 40)
        halves = [
            block_jackknife(e[indices], m[indices], N, float(T), half_block)
            for indices in np.array_split(np.arange(len(m)), 2)
        ]
        split = {}
        for name in ("e", "abs_m", "B4", "chi_abs"):
            se = np.hypot(
                halves[0]["standard_error"][name], halves[1]["standard_error"][name]
            )
            split[name] = (
                float(
                    abs(halves[0]["estimate"][name] - halves[1]["estimate"][name]) / se
                )
                if se
                else None
            )
        if any(x is not None and x > 3 for x in split.values()):
            warnings.append(
                "Production halves differ by more than three estimated standard errors"
            )
        interval = np.quantile(samples[:, slot, :], [0.025, 0.975], axis=0)
        ess = min(value["effective_samples"] for value in primary)
        rows.append(
            {
                "temperature": float(T),
                "statistics": stats,
                "autocorrelation": correlations,
                "block_errors": plateau,
                "split_z": split,
                "warnings": warnings,
                "percentile_interval_95": {
                    name: interval[:, i].tolist() for i, name in enumerate(NAMES)
                },
                "blocking_effective_samples": blocking_ess,
                "blocking_limiting_effective_samples_per_cpu_hour": min(
                    blocking_ess[name] for name in ("e", "abs_m", "m2", "m4")
                )
                / allocated_cpu_seconds
                * 3600,
                "blocking_efficiency_note": "Raw observable variance divided by the LARGEST squared jackknife standard error among the checked block sizes; approximate, with finite-block uncertainty. B4 uses its moment influence function. A short-window ACF can miss weak long tails.",
                "limiting_effective_samples": ess,
                "limiting_effective_samples_per_wall_second": ess / elapsed,
                "limiting_effective_samples_per_cpu_hour": ess
                / allocated_cpu_seconds
                * 3600,
            }
        )
    selected = [NAMES.index(name) for name in ("B4", "abs_m", "chi_abs")]
    flattened = samples[:, :, selected].reshape(replicates, -1)
    swaps = metadata["summary"]
    swap_rates = [
        a / n if n else None
        for a, n in zip(swaps["swap_accepts"], swaps["swap_attempts"])
    ]
    report = {
        "folder": str(ladder["folder"]),
        "L": config["model"]["L"],
        "mode": metadata["mode"],
        "swap_every_sweeps": config["tempering"]["swap_every_sweeps"],
        "elapsed_seconds": elapsed,
        "estimated_allocated_cpu_hours": allocated_cpu_seconds / 3600,
        "cost_note": "Rank count times full launcher elapsed time including archive/output validation; Slurm accounting is authoritative",
        "temperatures": rows,
        "swap_acceptance_by_pair": swap_rates,
        "round_trips": swaps["round_trips"],
        "input_hashes": ladder["input_hashes"],
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "covariance_order": [
            f"T={T}:{name}"
            for T in ladder["temperatures"]
            for name in ("B4", "abs_m", "chi_abs")
        ],
        "covariance": np.cov(flattened.T).tolist(),
        "method": "Joint contiguous-block bootstrap; the SAME time-block indices at every temperature and for all moments. Independent ladders must retain their identities.",
        "status": "Preliminary diagnostics; equilibration and blocking require review",
    }
    return report, blocks, samples


def bootstrap_finite_size(
    block_ladders,
    *,
    degree=2,
    lmin=0,
    crossing_sizes=None,
    replicates=2000,
    seed=544000,
):
    """Refit Tc/exponents while resampling whole independent PT ladders in time.

    Each entry contains L, temperatures, blocks[time_block,T,moment], and its
    replica_seeds. Multiple ladders at a size retain their separate identities.
    """
    from collections import Counter, defaultdict

    from .scaling import scaling_fit

    if replicates < 100:
        raise ValueError("Require at least 100 bootstrap replicates")
    groups = defaultdict(list)
    seen = set()
    for ladder in block_ladders:
        if ladder["L"] < lmin:
            continue
        seeds = set(ladder["replica_seeds"])
        if seen & seeds:
            raise ValueError(
                "FSS requires disjoint random streams between ladders; paired benchmark modes cannot be pooled"
            )
        seen.update(seeds)
        blocks = np.asarray(ladder["blocks"])
        if (
            blocks.ndim != 3
            or blocks.shape[1:] != (len(ladder["temperatures"]), 6)
            or len(blocks) < 20
        ):
            raise ValueError("Invalid whole-ladder moment blocks")
        groups[ladder["L"]].append(ladder)
    if len(groups) < 3:
        raise ValueError("Require at least three lattice sizes")
    ensembles = []
    for L, ladders in sorted(groups.items()):
        temperatures = np.asarray(ladders[0]["temperatures"])
        if any(
            not np.array_equal(ladder["temperatures"], temperatures)
            for ladder in ladders
        ):
            raise ValueError(
                "Independent ladders at the same size need matching temperatures"
            )
        total = np.concatenate([ladder["blocks"] for ladder in ladders], axis=0)
        count = len(total)
        summed = total.sum(axis=0)
        for slot, T in enumerate(temperatures):
            central = _observables_from_moments(
                total[:, slot, :].mean(axis=0), L**3, float(T)
            )
            jk = np.array(
                [
                    _observables_from_moments(
                        (summed[slot] - b[slot]) / (count - 1), L**3, float(T)
                    )
                    for b in total
                ]
            )
            errors = np.sqrt(
                (count - 1) / count * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0)
            )
            ensembles.append(
                {"L": L, "T": float(T), "central": central, "errors": errors}
            )
    central = np.array([g["central"] for g in ensembles])
    fit = scaling_fit(ensembles, central, degree, lmin, crossing_sizes)
    keys = (
        "Tc",
        "nu",
        "beta",
        "gamma",
        "beta_over_nu",
        "gamma_over_nu",
        "hyperscaling",
    )
    rng = np.random.default_rng(seed)
    samples = []
    failures = Counter()
    for _ in range(replicates):
        values = []
        for L, ladders in sorted(groups.items()):
            total_blocks = sum(len(ladder["blocks"]) for ladder in ladders)
            mean = (
                sum(
                    draw_ladder(ladder["blocks"], rng) * len(ladder["blocks"])
                    for ladder in ladders
                )
                / total_blocks
            )
            for slot, T in enumerate(ladders[0]["temperatures"]):
                values.append(_observables_from_moments(mean[slot], L**3, float(T)))
        try:
            draw = scaling_fit(
                ensembles, np.array(values), degree, lmin, crossing_sizes
            )
            samples.append([draw[key] for key in keys])
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
            failures[str(error)] += 1
    warnings = []
    for L, observables in fit["local_fits"].items():
        for name, local in observables.items():
            if local["dof"] > 0 and local["chi_square"] / local["dof"] > 5:
                warnings.append(f"L={L} {name}: local-fit chi-square/dof exceeds 5")
    if len(samples) < 0.95 * replicates:
        warnings.append("More than five percent of bootstrap fits failed")
    intervals = {}
    covariance = None
    if len(samples) >= 100:
        values = np.array(samples)
        intervals = {
            key: np.quantile(values[:, i], [0.025, 0.975]).tolist()
            for i, key in enumerate(keys)
        }
        covariance = np.cov(values.T).tolist()
    return {
        "fit": fit,
        "percentile_interval_95": intervals,
        "covariance_order": keys,
        "covariance": covariance,
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "bootstrap_successes": len(samples),
        "bootstrap_failure_reasons": dict(failures),
        "quality_warnings": warnings,
        "status": "withheld: local-fit or bootstrap checks failed"
        if warnings
        else "preliminary: equilibration, blocking and finite-size corrections still require review",
        "method": "Independent PT ladders resampled separately. Within every ladder the SAME contiguous time blocks are drawn across ALL temperatures and moments. Tc and all exponents refitted jointly. Original retained-sample weights preserved. Intervals conditional on successful fits; failures retained.",
    }
