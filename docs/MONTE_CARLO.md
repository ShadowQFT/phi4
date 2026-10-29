# Metropolis sampling contract

The baseline runs one serial chain per process and parallelizes independent
(L, T, seed, initial-state) chains with Slurm arrays. The optional MPI parallel-
tempering sampler adds exchanges between temperature replicas while retaining
Metropolis local updates. See PARALLEL_TEMPERING.md for its separate data and
resampling contract. MPI remains optional for the baseline.

## Sweep

A sweep is N attempted single-site updates. Proposed first implementation:
visit each site once in fixed index order, draw delta uniformly in
[-step_size, +step_size], and evaluate the local delta H.
Each site update preserves the target distribution; the sequential sweep
need not itself satisfy detailed balance. Record update order in metadata.

Accept delta H <= 0. Otherwise accept when U < exp(-delta_H/T), drawing U
from [0,1) with std::uniform_real_distribution. The direct probability
comparison mirrors the Metropolis formula and needs no special open-interval
random helper. Guard nonfinite arithmetic; do not silently
reject numerical failures and mark the run successful. Acceptance is a
sampler diagnostic, not an equilibration test.

Use std::mt19937_64 with an explicit 64-bit seed, separate recorded streams
for independent chains, and no time-only seeding. Library distribution
implementations can differ across toolchains: exact trajectories across
Windows and Mill are not required, but same-binary replay should work.

## Initial states and burn-in

Ordered positive/negative starts use +/-sqrt((6J-r)/u). A random start
uses independent uniform fields in [-1,1], explicitly recorded.
The start names describe the initial field configuration, not quenched disorder.
`random` draws each field independently; `ordered_positive` and
`ordered_negative` keep the two signs of the uniform ordered state distinct.

Temperature scans use these directions:

- A hot start uses `random` at the highest temperature and scans downward.
- An ordered start uses either ordered sign at the lowest temperature and scans
  upward.

Each scan carries the final lattice configuration to the next temperature,
with burn-in at each temperature before measurements. Temperatures within a scan
share a history; they are not independent chains. See [Temperature scans](SCANS.md)
for configuration and plotting instructions.

During burn-in only, tune step_size at fixed intervals using a bounded rule
targeting approximately 40-60% acceptance. Every 50 sweeps, multiply the
step by 0.9 below 40% or by 1.1 above 60%; otherwise keep it. Clamp to
[0.0001,100]. Record rule, intervals, bounds,
final value, and adaptation stop. This target is a starting heuristic.
Freeze step_size before a final fixed-kernel burn-in segment and all
measurements. Exclude all burn-in/adaptation samples from estimates.

Record burn-in diagnostics separately. Compare e, |m|, m^2, and B4 from
ordered and random starts, including split-chain stability. Do not
demand sign flips alone as proof of equilibrium or combine chains that
retain incompatible histories. Increase burn-in or length if needed.

## Production

Start by retaining one sample per sweep; measure H and m and derive moments
in analysis. Saving only run means loses blocking and resampling ability.
Benchmark measurement overhead before optimizing it. If energy is updated
incrementally, periodically compare it with full H and record drift checks.
Keep stopping criteria and output status explicit for interrupted jobs.

Production lengths come from runtime and autocorrelation pilots, not a
fixed sweep count mistaken for independent statistics. Checkpoints later
must preserve fields, RNG state, counters, config, and measurement phase;
a restarted chain is not a continuation unless that state is complete.
