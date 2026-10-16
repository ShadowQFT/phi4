# Correlations and uncertainty

Keep each chain's time ordering and identity. Analyze after explicit burn-in.
Inspect histories before treating a stationary autocorrelation estimate as valid.

## Autocorrelation

For a stationary series with normalized autocorrelation rho(t):

```text
tau_int = 1/2 + sum_{t=1}^{window} rho(t)
Var(mean) approximately (2 tau_int / n) Var(sample)
effective_sample_size approximately n / (2 tau_int).
```

These tau units are retained samples; multiply by measurement interval to
express sweeps. Use a documented self-consistent/truncated window and
inspect stability against that choice. Check e, |m|, m^2, m^4 separately:
a rapidly mixing energy series does not certify Binder moments.
If the series is too short to estimate the tail reliably, report that.

[Wolff's error-analysis paper](https://arxiv.org/abs/hep-lat/0306017)
is a source for autocorrelation-based uncertainty estimation. Read the
full methods before claiming to implement its Gamma-method; the initial
project can use checked blocking and block resampling instead.

## Blocking and resampling

Increase contiguous block length until uncertainty estimates stabilize;
use blocks comfortably longer than the slowest relevant correlation time.
Keep enough blocks to estimate errors (20+ is a provisional diagnostic
minimum, not a guarantee). Compare longer block choices. A non-plateau or
too few blocks means extend the run, not claim independent samples.

Bootstrap or jackknife complete blocks within each chain, preserving the
joint sample vector (e, m, |m|, m^2, m^4). Recompute moments, B4, and chi_abs
in every replicate. Never resample each observable independently or treat
correlated sweeps as IID. Record bootstrap seed and replicate count; start
with 2000 and check interval stability if results depend on tail quantiles.

Combine compatible equilibrated chains with explicit sample counts and
weights; compare chain estimates first. Retain their identities. Avoid
forcing ordered/random chains to agree by unreported exclusions.
Independent seeds do not make unequal or unequilibrated chains exchangeable.

Recompute Tc and all subsequent fits inside each joint bootstrap replicate.
The shared Tc and nu induce correlations in beta = nu*(beta/nu) and
gamma = nu*(gamma/nu); independent error-bar multiplication is inadequate.

## Reporting

State confidence level, error method, block size, sweeps, number of chains,
and effective sample estimates. Distinguish statistical intervals from
changes under Lmin, temperature window, or fit-model choices. List excluded
runs and reasons. Preserve failed bootstrap fits and their frequency;
silently dropping them can bias intervals.
