"""Finite-size scaling of independent Metropolis chains, with joint block resampling."""

from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from .data import load_run
from .statistics import NAMES, _block_means, _observables_from_moments, block_jackknife


def autocorrelation(series, window_factor=6.0):
    """FFT autocovariance and positive-pair, self-consistent truncation.

    Tau is in saved samples. This estimator assumes stationarity; diagnostics
    do not prove equilibration. Pair truncation avoids summing a noisy tail.
    """
    x = np.asarray(series, dtype=float)
    if x.ndim != 1 or len(x) < 32 or not np.all(np.isfinite(x)):
        raise ValueError("Require at least 32 finite samples")
    x = x - x.mean()
    n = len(x)
    size = 1 << (2 * n - 1).bit_length()
    spectrum = np.fft.rfft(x, n=size)
    covariance = np.fft.irfft(spectrum * spectrum.conj(), n=size)[:n]
    covariance /= np.arange(n, 0, -1)
    if covariance[0] <= 0:
        raise ValueError("Constant series has no estimable autocorrelation")
    rho = covariance / covariance[0]
    tau, window = 0.5, 0
    for lag in range(1, n // 2 - 1, 2):
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0:
            break
        tau += pair
        window = lag + 1
        if window >= window_factor * tau:
            break
    return {
        "tau_samples": float(tau),
        "effective_samples": float(n / (2 * tau)),
        "window": window,
        "window_factor": window_factor,
        "length_over_tau": float(n / tau),
        "tail_resolved": bool(n >= 100 * tau and window < n // 4),
        "method": "FFT unbiased autocovariance; positive pairs; self-consistent window",
    }


def summarize_chain(run, block_size):
    model, sampling = run.config["model"], run.config["sampling"]
    e, m = run.energy, run.magnetization
    result = block_jackknife(e, m, model["L"] ** 3, model["temperature"], block_size)
    correlations = {}
    for name, series in (("e", e), ("abs_m", abs(m)), ("m2", m**2), ("m4", m**4)):
        correlations[name] = autocorrelation(series)
        correlations[name]["tau_sweeps"] = (
            correlations[name]["tau_samples"] * sampling["measure_every_sweeps"]
        )
    max_tau = max(v["tau_samples"] for v in correlations.values())
    plateau = {}
    for length in sorted(
        {max(1, block_size // 4), max(1, block_size // 2), block_size, block_size * 2}
    ):
        if len(m) // length >= 20:
            plateau[str(length)] = block_jackknife(
                e, m, model["L"] ** 3, model["temperature"], length
            )["standard_error"]
    first, second = np.array_split(np.arange(len(m)), 2)
    halves = []
    half_block = min(block_size, min(len(first), len(second)) // 20)
    for indices in (first, second):
        halves.append(
            block_jackknife(
                e[indices],
                m[indices],
                model["L"] ** 3,
                model["temperature"],
                half_block,
            )
        )
    split_z = {}
    for name in ("e", "abs_m", "B4", "chi_abs"):
        error = np.hypot(
            halves[0]["standard_error"][name], halves[1]["standard_error"][name]
        )
        split_z[name] = (
            float(
                abs(halves[0]["estimate"][name] - halves[1]["estimate"][name]) / error
            )
            if error
            else None
        )
    seconds = (
        datetime.fromisoformat(run.metadata["finished_utc"])
        - datetime.fromisoformat(run.metadata["started_utc"])
    ).total_seconds()
    flags = []
    if block_size < 10 * max_tau:
        flags.append("Block shorter than ten times slowest estimated tau")
    if any(not v["tail_resolved"] for v in correlations.values()):
        flags.append("Autocorrelation tail may be unresolved")
    if any(z is not None and z > 3 for z in split_z.values()):
        flags.append(
            "Production halves differ by more than three estimated standard errors"
        )
    return {
        "run": str(run.folder),
        "L": model["L"],
        "T": model["temperature"],
        "start": sampling["start"],
        "seed": sampling["seed"],
        "statistics": result,
        "autocorrelation": correlations,
        "block_errors": plateau,
        "split_z": split_z,
        "elapsed_seconds": seconds,
        "slowest_effective_samples_per_second": min(
            v["effective_samples"] for v in correlations.values()
        )
        / seconds
        if seconds > 0
        else None,
        "warnings": flags,
        "input_hashes": run.input_hashes,
    }


def load_ensembles(folders, block_size, tmin=None, tmax=None):
    groups = defaultdict(list)
    reports = []
    seen_seeds, couplings = set(), set()
    runs = []
    for folder in folders:
        run = load_run(folder)
        model = run.config["model"]
        T = model["temperature"]
        if (tmin is not None and T < tmin) or (tmax is not None and T > tmax):
            continue
        seed = run.config["sampling"]["seed"]
        if seed in seen_seeds:
            raise ValueError("Duplicate chain seed in campaign")
        seen_seeds.add(seed)
        couplings.add(tuple(model[k] for k in ("name", "boundary", "J", "r", "u")))
        groups[(model["L"], T)].append(run)
        reports.append(summarize_chain(run, block_size))
        runs.append(run)
    if len(couplings) != 1:
        raise ValueError("Require one fixed-coupling model")
    if not runs:
        raise ValueError("No completed chains in requested window")
    ensembles = []
    report_map = {r["run"]: r for r in reports}
    for (L, T), chains in sorted(groups.items()):
        blocks = [
            _block_means(r.energy, r.magnetization, block_size)[0] for r in chains
        ]
        comparisons = []
        for i, first in enumerate(chains):
            for second in chains[i + 1 :]:
                a, b = report_map[str(first.folder)], report_map[str(second.folder)]
                zscores = {}
                for name in ("e", "abs_m", "B4", "chi_abs"):
                    error = np.hypot(
                        a["statistics"]["standard_error"][name],
                        b["statistics"]["standard_error"][name],
                    )
                    zscores[name] = (
                        float(
                            abs(
                                a["statistics"]["estimate"][name]
                                - b["statistics"]["estimate"][name]
                            )
                            / error
                        )
                        if error
                        else None
                    )
                comparisons.append(
                    {"first": a["run"], "second": b["run"], "z_scores": zscores}
                )
                if any(z is not None and z > 3 for z in zscores.values()):
                    warning = "Independent chains differ by more than three estimated standard errors"
                    for report in (a, b):
                        if warning not in report["warnings"]:
                            report["warnings"].append(warning)
        total = np.concatenate(blocks)
        central = _observables_from_moments(total.mean(axis=0), L**3, T)
        # Stratify by chain: every independent start keeps its original weight.
        count = len(total)
        jk = []
        summed = total.sum(axis=0)
        for block in total:
            jk.append(
                _observables_from_moments((summed - block) / (count - 1), L**3, T)
            )
        jk = np.array(jk)
        errors = np.sqrt(
            (count - 1) / count * np.sum((jk - jk.mean(axis=0)) ** 2, axis=0)
        )
        ensembles.append(
            {
                "L": L,
                "T": T,
                "blocks": blocks,
                "central": central,
                "errors": errors,
                "chains": [str(r.folder) for r in chains],
                "chain_comparisons": comparisons,
            }
        )
    return ensembles, reports, runs


def draw_ensembles(ensembles, rng):
    result = []
    for group in ensembles:
        resampled = []
        for blocks in group["blocks"]:
            indices = rng.integers(0, len(blocks), size=len(blocks))
            resampled.append(blocks[indices])
        mean = np.concatenate(resampled).mean(axis=0)
        result.append(_observables_from_moments(mean, group["L"] ** 3, group["T"]))
    return np.array(result)


def power_fit(sizes, values, errors):
    sizes, values, errors = map(
        lambda a: np.asarray(a, dtype=float), (sizes, values, errors)
    )
    if len(sizes) < 3 or np.any(values <= 0) or np.any(errors <= 0):
        raise ValueError("Power fit requires three sizes and positive values/errors")
    x, y = np.log(sizes), np.log(values)
    weights = values / errors
    design = np.column_stack((np.ones(len(x)), x))
    parameters = np.linalg.lstsq(design * weights[:, None], y * weights, rcond=None)[0]
    residuals = y - design @ parameters
    return {
        "amplitude": float(np.exp(parameters[0])),
        "slope": float(parameters[1]),
        "chi_square": float(np.sum((residuals * weights) ** 2)),
        "dof": len(x) - 2,
        "log_residuals": residuals.tolist(),
    }


def scaling_fit(ensembles, values, degree=1, lmin=0, crossing_sizes=None):
    """Common Tc from largest-pair crossing; independent exponents at that Tc.

    No drift extrapolation is claimed. Fits use measured overlapping T coverage.
    Polynomial weights are fixed from the original block jackknife.
    """
    sizes = sorted({g["L"] for g in ensembles if g["L"] >= lmin})
    if len(sizes) < 3:
        raise ValueError("Require at least three lattice sizes for exponent fits")
    curves = {}
    center = float(np.mean([g["T"] for g in ensembles]))
    names = ("B4", "abs_m", "chi_abs")
    bounds = []
    for L in sizes:
        indices = [i for i, g in enumerate(ensembles) if g["L"] == L]
        temperatures = np.array([ensembles[i]["T"] for i in indices])
        if len(indices) < degree + 2:
            raise ValueError("Require polynomial degree + two temperatures per size")
        bounds.append((float(min(temperatures)), float(max(temperatures))))
        curves[L] = {}
        for name in names:
            col = NAMES.index(name)
            sigma = np.array([ensembles[i]["errors"][col] for i in indices])
            if np.any(sigma <= 0):
                raise ValueError("Nonpositive local-fit uncertainty")
            curve = np.polynomial.Polynomial.fit(
                temperatures - center, values[indices, col], degree, w=1 / sigma
            ).convert()
            residuals = values[indices, col] - curve(temperatures - center)
            curves[L][name] = {
                "polynomial": curve,
                "residuals": residuals.tolist(),
                "chi_square": float(np.sum((residuals / sigma) ** 2)),
                "dof": len(indices) - degree - 1,
                "typical_error": float(np.mean(sigma)),
            }
    lower = max(b[0] for b in bounds)
    upper = min(b[1] for b in bounds)
    if lower >= upper:
        raise ValueError("No overlapping measured temperature window")
    pair = tuple(crossing_sizes or sizes[-2:])
    if len(pair) != 2 or pair[0] >= pair[1] or any(L not in sizes for L in pair):
        raise ValueError(
            "Crossing sizes must be two distinct fitted sizes in increasing order"
        )
    difference = (
        curves[pair[0]]["B4"]["polynomial"] - curves[pair[1]]["B4"]["polynomial"]
    )
    roots = [
        float(r.real + center)
        for r in difference.roots()
        if abs(r.imag) < 1e-8 and lower < r.real + center < upper
    ]
    if len(roots) != 1:
        raise ValueError("Need exactly one crossing inside measured common window")
    Tc = roots[0]
    fitted, errors = {}, {}
    for name in names:
        fitted[name], errors[name] = [], []
        for L in sizes:
            curve = curves[L][name]
            value = curve["polynomial"](Tc - center)
            if name == "B4":
                derivative = curve["polynomial"].deriv()(Tc - center)
                if derivative <= 0:
                    raise ValueError(
                        "Binder moment-ratio slope must be positive near transition"
                    )
                value = derivative
                scale = max(upper - lower, 1e-10)
                error = curve["typical_error"] / scale
            else:
                error = curve["typical_error"]
            fitted[name].append(float(value))
            errors[name].append(float(error))
    fits = {name: power_fit(sizes, fitted[name], errors[name]) for name in names}
    pnu, pm, pchi = (fits[name]["slope"] for name in names)
    if pnu <= 0:
        raise ValueError("Inverse-nu fit is not positive")
    nu = 1 / pnu
    local_fits = {}
    for L in sizes:
        local_fits[str(L)] = {}
        for name in names:
            curve = curves[L][name]
            local_fits[str(L)][name] = {
                k: v for k, v in curve.items() if k != "polynomial"
            }
            local_fits[str(L)][name]["coefficients_at_temperature_center"] = curve[
                "polynomial"
            ].coef.tolist()
    return {
        "Tc": Tc,
        "nu": nu,
        "beta": -pm * nu,
        "gamma": pchi * nu,
        "beta_over_nu": -pm,
        "gamma_over_nu": pchi,
        "hyperscaling": -2 * pm + pchi,
        "sizes": sizes,
        "crossing_sizes": pair,
        "temperature_window": [lower, upper],
        "temperature_center": center,
        "polynomial_degree": degree,
        "power_fits": fits,
        "local_fits": local_fits,
        "values_at_Tc": fitted,
        "weight_note": "Power-fit weights use fixed original local moment errors as working weights; all reported parameter intervals come from joint bootstrap, including Tc and slopes",
        "Tc_note": "Finite-volume largest-pair crossing, without correction-to-scaling extrapolation",
    }


def bootstrap_scaling(
    ensembles, *, degree=1, lmin=0, replicates=2000, seed=540928, crossing_sizes=None
):
    central = np.array([g["central"] for g in ensembles])
    fit = scaling_fit(ensembles, central, degree, lmin, crossing_sizes)
    keys = [
        "Tc",
        "nu",
        "beta",
        "gamma",
        "beta_over_nu",
        "gamma_over_nu",
        "hyperscaling",
    ]
    samples, failures = [], Counter()
    rng = np.random.default_rng(seed)
    for _ in range(replicates):
        try:
            replicate = scaling_fit(
                ensembles, draw_ensembles(ensembles, rng), degree, lmin, crossing_sizes
            )
            samples.append([replicate[key] for key in keys])
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
            failures[str(error)] += 1
    count = len(samples)
    intervals = {}
    covariance = None
    if count >= 100:
        values = np.array(samples)
        intervals = {
            key: np.quantile(values[:, i], [0.025, 0.975]).tolist()
            for i, key in enumerate(keys)
        }
        covariance = np.cov(values.T).tolist()
    quality_warnings = []
    for L, observables in fit["local_fits"].items():
        for name, local in observables.items():
            if local["dof"] > 0 and local["chi_square"] / local["dof"] > 5:
                quality_warnings.append(
                    f"L={L} {name}: local polynomial chi-square/dof exceeds 5"
                )
    if count < 0.95 * replicates:
        quality_warnings.append("More than five percent of bootstrap fits failed")
    return {
        "fit": fit,
        "fit_usable": not quality_warnings,
        "quality_warnings": quality_warnings,
        "percentile_interval_95": intervals,
        "covariance_order": keys,
        "covariance": covariance,
        "bootstrap_replicates": replicates,
        "bootstrap_successes": count,
        "bootstrap_failure_reasons": dict(failures),
        "bootstrap_seed": seed,
        "intervals_reliable": count >= 0.95 * replicates and count >= 100,
        "status": "withheld: candidate fit fails quality checks"
        if quality_warnings
        else "preliminary: equilibrium, blocking and finite-size stability require review",
        "bootstrap_note": "Whole joint-moment blocks resampled within each chain; original chain weights retained; crossing and every exponent refitted in each replicate. Intervals conditional on successful fits; failure fraction reported.",
    }
