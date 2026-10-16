"""Observable formulas and block-based error estimates for one Monte Carlo chain."""

import numpy as np

NAMES = (
    "e",
    "m",
    "abs_m",
    "m2",
    "m4",
    "B4",
    "chi_abs",
    "chi_signed",
    "heat_capacity_per_site",
)


def _check_parameters(N, T):
    if type(N) is not int or N <= 0:
        raise ValueError("N must be a positive integer")
    if type(T) not in (int, float) or not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")


def _sample_moments(energy, magnetization):
    """One row per saved sample, one column per moment used in the formulas."""
    e = np.asarray(energy, dtype=float)
    m = np.asarray(magnetization, dtype=float)
    if e.ndim != 1 or e.shape != m.shape or len(e) < 2:
        raise ValueError(
            "Require matching one-dimensional series with at least 2 samples"
        )
    if not np.all(np.isfinite(e)) or not np.all(np.isfinite(m)):
        raise ValueError("Samples must be finite")
    with np.errstate(over="raise", invalid="raise"):
        return np.column_stack((e, m, np.abs(m), m**2, m**4, e**2))


def _observables_from_moments(moments, N, T):
    """Apply the physical formulas to one set of averaged moments."""
    mean_e, mean_m, mean_abs_m, mean_m2, mean_m4, mean_e2 = moments
    if mean_m2 <= 0:
        raise ValueError("The Binder ratio requires a positive mean of m squared")

    # Form the Binder ratio after averaging; preserve the distinct susceptibilities.
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        binder = mean_m4 / mean_m2**2
        chi_abs = (N / T) * (mean_m2 - mean_abs_m**2)
        chi_signed = (N / T) * (mean_m2 - mean_m**2)
        heat_capacity = (N / T / T) * (mean_e2 - mean_e**2)

    values = np.array(
        [
            mean_e,
            mean_m,
            mean_abs_m,
            mean_m2,
            mean_m4,
            binder,
            chi_abs,
            chi_signed,
            heat_capacity,
        ]
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("An observable is nonfinite")
    return values


def observables(e, m, N, T):
    """Calculate the observables using every supplied sample."""
    _check_parameters(N, T)
    mean_moments = _sample_moments(e, m).mean(axis=0)
    values = _observables_from_moments(mean_moments, N, T)
    return dict(zip(NAMES, values.tolist()))


def _block_means(e, m, block_size):
    """Average consecutive samples in equal blocks; report an incomplete tail."""
    if type(block_size) is not int or block_size < 1:
        raise ValueError("block_size must be a positive integer")
    moments = _sample_moments(e, m)
    block_count = len(moments) // block_size
    if block_count < 20:
        raise ValueError("Need at least 20 complete blocks for an error estimate")

    retained_count = block_count * block_size
    retained = moments[:retained_count]
    # Shape: (number of blocks, samples per block, six moment columns).
    blocks = retained.reshape(block_count, block_size, 6)
    means = blocks.mean(axis=1)
    dropped_count = len(moments) - retained_count
    return means, dropped_count


def block_jackknife(e, m, N, T, block_size):
    """Omit one block at a time to estimate each observable's standard error."""
    _check_parameters(N, T)
    blocks, dropped = _block_means(e, m, block_size)
    block_count = len(blocks)
    estimate = _observables_from_moments(blocks.mean(axis=0), N, T)

    # Subtracting one block sum avoids copying all remaining raw samples.
    total = blocks.sum(axis=0)
    leave_one_out = []
    for block in blocks:
        remaining_mean = (total - block) / (block_count - 1)
        values = _observables_from_moments(remaining_mean, N, T)
        leave_one_out.append(values)
    leave_one_out = np.array(leave_one_out)

    average = leave_one_out.mean(axis=0)
    squared_differences = (leave_one_out - average) ** 2
    variance = (block_count - 1) / block_count * squared_differences.sum(axis=0)
    standard_error = np.sqrt(variance)
    return {
        "estimate": dict(zip(NAMES, estimate.tolist())),
        "standard_error": dict(zip(NAMES, standard_error.tolist())),
        "block_size": block_size,
        "block_count": block_count,
        "dropped_tail_samples": dropped,
        "method": "contiguous-block delete-one jackknife; retained complete blocks; no bias correction",
    }


def block_bootstrap(e, m, N, T, block_size, *, seed, replicates=2000):
    """Draw whole blocks with replacement and recompute all observables together."""
    _check_parameters(N, T)
    if type(replicates) is not int or replicates < 100:
        raise ValueError("Require at least 100 bootstrap replicates")
    blocks, dropped = _block_means(e, m, block_size)
    block_count = len(blocks)
    rng = np.random.default_rng(seed)

    samples = []
    for _ in range(replicates):
        # Repeated indices are allowed: this is sampling with replacement.
        indices = rng.integers(0, block_count, size=block_count)
        mean_moments = blocks[indices].mean(axis=0)
        values = _observables_from_moments(mean_moments, N, T)
        samples.append(values)

    # Each column belongs to one observable; take its two percentile endpoints.
    intervals = np.quantile(samples, [0.025, 0.975], axis=0)
    bounds = {}
    for column, name in enumerate(NAMES):
        bounds[name] = intervals[:, column].tolist()
    return {
        "percentile_interval_95": bounds,
        "replicates": replicates,
        "seed": seed,
        "block_size": block_size,
        "block_count": block_count,
        "dropped_tail_samples": dropped,
        "method": "joint contiguous-block bootstrap; one chain; equilibrium and block plateau require inspection",
    }
