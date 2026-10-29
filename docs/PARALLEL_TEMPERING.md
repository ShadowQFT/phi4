# Parallel tempering

The optional MPI sampler keeps Metropolis updates inside each replica and adds
replica exchange between temperatures. It is separate from the existing
single-temperature executable and does not change running campaigns.

## Updates and exchanges

One MPI rank owns one field and RNG. All ranks in a ladder have the same L,
J, r, u and boundary conditions. Temperatures are strictly increasing.
After a fixed number of local sweeps, rank zero gathers total energies and
attempts disjoint neighboring pairs, alternating even and odd pair sets:

```text
log R = (1/T_i - 1/T_j) (H_i - H_j)
p_accept = min(1, exp(log R)).
```

H is the total Hamiltonian, not energy per site. The coordinator uses a separate
recorded random stream. Accepted exchanges change temperature labels, keeping
fields and their local RNGs on their ranks. A broadcast updates the permutation.
This follows J1J2's GROUPED method in
sim/square_j1j2/J1J2_6tv.f90: grouped_pt_cycle and grouped_pt_attempt_swaps.
J1J2 is a read-only reference; its spin observables are not reused for phi4.

Each temperature adapts its proposal step independently without exchanges first.
Adaptation then stops. Frozen step sizes follow temperatures, not configurations,
through settling and measurements. A zero exchange interval explicitly selects
the no-exchange MPI control for matched benchmarks. One temperature degenerates
to the original Metropolis trajectory for the same seed and sampling settings.

## Build and run

Default CMake builds include phi4_pt_serial, which emulates a full ladder on
one process and supplies a deterministic oracle. MPI is optional:

```text
cmake -S . -B build/mpi -DPHI4_ENABLE_MPI=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build/mpi
python scripts/run_pt.py configs/local/pt-smoke.example.toml --binary build/mpi/phi4_pt --mode mpi
```

For serial emulation use --mode serial and phi4_pt_serial. Check inputs with
--validate-only before sampling. An attempt needs a new output identity.
MPI rank count must equal ladder length. Worker exceptions abort the communicator
so other ranks cannot hang silently; launcher failures preserve FAIL metadata.

On Mill, use the existing ignored configs/mill.local.toml and SSH setup:

```text
python scripts/mill.py deploy
python scripts/pt_mill.py build RELEASE
python scripts/pt_mill.py plan RELEASE CAMPAIGN --L 16 --temperatures 14.6 15.0 15.32 15.6 16.1 17.0 --minutes 10
python scripts/pt_mill.py submit cluster/manifests/CAMPAIGN/pt_plan.json --budget-cpu-hours 1
python scripts/mill.py pull RELEASE CAMPAIGN
```

The example six-rank ten-minute cap is one CPU-hour. Inspect validation job logs
and require the VALIDATED marker before submitting a ladder. Failed build
attempts retain their logs and scripts; --attempt NEW_NAME creates a distinct
attempt. Batch scripts initialize the complete Mill module environment before
loading openmpi/4.1.6/gcc/12.2.0. Validation uses Python's standard library;
NumPy and Matplotlib are only required for local analysis.

One ladder is one Slurm job, with K allocated CPU cores for K temperatures.
The submission guard counts all the user's queued/running jobs and expanded
array elements, including J1J2, and requires fewer than 30 after submission.
CPU-hour caps count EVERY replica: K times the worker walltime. An MPI ladder
cannot bypass the compute accounting merely by counting as one job.

## Measurements and integrity

Rank zero writes a synchronized row for every temperature at each measurement
sweep. Measurements are routed using the post-exchange temperature permutation.
The raw schema is separate from independent-chain data:

- measurements.csv: sample index, absolute sweep, temperature index/value,
  physical replica identity, energy per site and magnetization.
- equilibration.csv: sampled adaptation/settling histories and step sizes.
- swaps.csv: phase, paired slots/replicas, total energies, log ratio and outcome.
- walks.csv: every exchange pass's full rank-to-temperature permutation.
- pt_summary.json: production swap attempts/accepts and full endpoint-return
  round trips, with frozen step sizes per temperature.
- metadata.json, config.resolved.toml, source.zip, temperatures.txt and logs:
  exact inputs, seeds, source/binary identities, timestamps and raw-file hashes.

The launcher checks counts, phase counters, rank permutations, total-energy swap
exponents, fixed proposal steps and summaries against raw data. The executable's serial/MPI
mode is checked before launch. Round trips are
independently reconstructed as endpoint transitions from the walks. Raw data and
failed attempts are never overwritten. There is no checkpoint restart.

## Analysis and finite-size fits

```text
python scripts/analyze_pt.py data/raw/CAMPAIGN/chain-0000 --output results/figures/NEW_OUTPUT --block-size 1000
python scripts/fit_pt.py LADDER_L16 LADDER_L24 LADDER_L32 --output results/tables/NEW_FIT.json --tmin 15.30 --tmax 15.34 --block-size 1000 --degree 2
```

Every bootstrap draw resamples the SAME contiguous time blocks at all temperatures
and for all moments within a ladder. Independent ladders are resampled separately,
retaining their original sample weights. Refit the crossing and exponents in
every draw. Never pass PT temperatures to the independent-chain bootstrap.
At least degree+2 temperatures per fitted size and three sizes are required.
Choose production ladders with enough measured points in the narrow fit window;
the broad performance ladder is not automatically an appropriate fit window.

Paired benchmark modes intentionally share seeds and initial conditions. They
must not be pooled as independent production ladders; fit_pt rejects repeated
replica seeds. Fixed couplings, conventions and measurement intervals must match.
The fit output preserves covariance, failed fits, controls and input/source
hashes. Finite-volume crossings are not a correction-to-scaling extrapolation.
A passing local polynomial does not certify equilibration or asymptotic scaling.

Inspect e, |m|, m2 and m4 autocorrelations, block-error plateaus, production halves,
independent starts, swap rates and round trips. Signed m is an additional
mixing diagnostic. High exchange acceptance alone is insufficient. The analysis
records conventional ACF-based effective counts and a block-based cross-check:
raw-observable variance divided by the largest squared standard error among
the checked block lengths. The latter helps expose weak long tails that an
ACF window might miss; finite-block uncertainty still applies. B4's analogous
estimate uses its moment influence function. Compare both elapsed-time and
all-replica CPU-hour efficiency, with Slurm accounting authoritative for cost.

## Verification

Local Debug and Release: nine applicable CTest suites pass, including exchange
identities, PT workflow and synchronized statistics. The legacy scan suite is
incompatible with the pre-existing user edit in src/main.cpp and remains excluded.

PT-specific checks include an independent finite-state Boltzmann detailed-balance
oracle; total-energy/sign/temperature factors; label permutations and full round
trips; one-temperature replay of the original sampler; deterministic serial
ladders; invalid ladder/seed rejection; routing corruption; J=0 e, m2 and m4
against independently refined Simpson quadrature; synchronized block alignment;
cross-temperature bootstrap covariance; exact synthetic finite-size exponents;
and rejection of repeated streams between supposedly independent ladders.
The launcher checks the executable's compiled serial/MPI mode before starting
workers, preventing a serial emulation from being launched independently on every
MPI rank. This guard is tested before invoking mpiexec. Mill additionally passes
byte-for-byte MPI versus serial ladder histories,
including measurements, burn-in, swaps, walks and summary.

See ACTIVE_RESEARCH.md for the actual release, builds, benchmark jobs and results.


## References

[Earl and Deem, Parallel Tempering: Theory, Applications, and New Perspectives](https://arxiv.org/abs/physics/0508111)
provides a method overview. [Bittner, Nussbaumer and Janke](https://arxiv.org/abs/0809.0571)
discuss the effect of exchange cadence and correlation times on round-trip
performance. The implemented acceptance rule is independently tested against
the product canonical distribution; no literature exponent is a fitting target.
