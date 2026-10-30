# Clean 3D phi4: Computational Physics Lab

I am studying how a lattice of continuously varying fields develops collective
order. This is my project for PHYSICS 5409 Computational Physics Lab at Missouri
University of Science & Technology.

The model is simple: neighboring fields favor alignment, while temperature
introduces fluctuations. What makes it interesting is that its critical behavior
should match the three-dimensional Ising model. Ising spins take two values;
the fields in this model vary continuously. I am using Monte Carlo simulations
to locate the transition and estimate the critical exponents from finite lattices.

The simulation is written in C++17. Python handles the runs, statistical analysis,
and figures, and larger calculations run on Mill with MPI parallel tempering.

## Methods and workflow

- **Metropolis sampling:** real-valued fields on periodic cubic lattices,
  with random and ordered starts. Proposal sizes are adapted before production
  and then held fixed during settling and measurements.
- **Parallel tempering:** one MPI worker per temperature, with exchanges between
  neighboring temperatures to improve sampling. Local updates remain Metropolis.
- **Observables:** energy, magnetization, normalized Binder cumulant, and
  susceptibility. The plotted Binder cumulant approaches one in the ordered
  limit and zero in the Gaussian high-temperature limit.
- **Finite-size analysis:** Binder crossings and temperature slopes, magnetization
  and susceptibility scaling, and comparisons across lattice sizes.
- **Uncertainty checks:** autocorrelation, contiguous-block resampling,
  independent-start comparisons, and sensitivity to fit windows and block size.
- **Mill runs:** preserved configurations, seeds, source snapshots, raw histories,
  and checksums for each calculation.

## Model

Each lattice site holds a real-valued field. Neighboring sites favor alignment,
and the local potential has two minima:

```text
H = -J sum_<ij> phi_i phi_j + sum_i [(r/2) phi_i^2 + (u/4) phi_i^4]
```

The current study keeps `J=1`, `r=-1`, and `u=1` fixed. Lattice size, temperature,
seed, and sweep counts are configured in TOML.

![Analytical on-site potential for r=-1 and u=1](docs/images/onsite-potential.png)

*The two wells favor positive and negative fields. Neighbor interactions couple
these local choices into collective order.*

## Running locally

Requires a C++17 compiler, CMake 3.21+, Ninja, and Python 3.11+. Install the Python
analysis dependencies in your Python environment, then build from the
repository root:

```sh
python -m pip install -r analysis/requirements.txt
cmake --preset local-debug
cmake --build --preset local-debug
ctest --test-dir build/local-debug --output-on-failure
```

The default build requires no MPI installation. MPI is used for the parallel
tempering runs on Mill; see the [parallel tempering notes](docs/PARALLEL_TEMPERING.md).
The tests cover the energy calculations, sampling rules, and statistical analysis.

See [development setup](docs/DEVELOPMENT.md) for virtual environments, compiler
selection, macOS instructions, and a separate build for each platform.

### Small test run

Validate the input without running a simulation:

```sh
python scripts/run.py configs/local/smoke.example.toml --validate-only
```

To execute the tiny `L=4` check after building:

```sh
# Windows
python scripts/run.py configs/local/smoke.example.toml --binary build/local-debug/phi4.exe

# Linux or macOS
python scripts/run.py configs/local/smoke.example.toml --binary build/local-debug/phi4
```

This input performs 100 settling sweeps and saves 100 measurement rows. It writes
`measurements.csv`, `equilibration.csv`, a resolved configuration, a source
snapshot, and run metadata under `data/raw/local-smoke/`. Reusing the same output
path is refused; choose a new run ID for another attempt. This small run checks
that the build and output workflow work before starting longer calculations.

## Project layout

| Component | Start here | Purpose |
| --- | --- | --- |
| Lattice and energy | [model.hpp](include/phi4/model.hpp) | Periodic neighbors, energy, and local energy changes |
| Local sampler | [sampler.hpp](include/phi4/sampler.hpp) | Initialization, proposals, acceptance, and measurements |
| Executables | [main.cpp](src/main.cpp), [pt.cpp](src/pt.cpp) | Chain execution and serial/MPI tempering |
| Run workflow | [run.py](scripts/run.py), [run_pt.py](scripts/run_pt.py) | Validate inputs and preserve outputs and provenance |
| Analysis | [phi4_analysis](analysis/phi4_analysis/) | Load data, calculate observables, estimate errors, and plot diagnostics |
| Checks | [C++ tests](tests/cpp/), [Python tests](tests/python/) | Independent numerical and workflow checks |

For a closer look, read the [code walkthrough](docs/CODE.md) and
[validation plan](docs/VALIDATION.md). [Architecture](docs/ARCHITECTURE.md)
explains the boundaries between the components. MPI setup and replica-exchange
checks are documented in [parallel tempering](docs/PARALLEL_TEMPERING.md).

## Progress so far

The latest runs show a common Binder crossing near `T = 15.310` and a
susceptibility peak that becomes sharper as the lattice grows. Parallel tempering
has made it practical to sample several temperatures together at sizes up to
`L = 48`; separate Metropolis runs also reach `L = 64`.

The first exponent estimates are consistent with the 3D Ising comparison within
their statistical intervals. I am checking how much they change with block length,
temperature window, and the sizes included in the fit. Longer runs at the larger
sizes are the next step toward separating statistical uncertainty from finite-size
effects. The [analysis notes](docs/FINITE_SIZE_SCALING.md) describe the fitting
procedure.

Raw histories, generated figures, and local research notes stay outside Git.

## Acknowledgment

Codex assisted with implementation and debugging. I remain responsible for
reviewing the code and validating the scientific results.
