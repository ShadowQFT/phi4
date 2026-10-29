# 3D phi4 Monte Carlo simulator

A C++17 simulation project for studying ordering in a three-dimensional scalar
field, with Python tools for reproducible runs, statistical analysis, and plots.
I built this for PHYSICS 5409 Computational Physics Lab at Missouri University
of Science & Technology.

The engineering work includes implementing and testing numerical algorithms,
validating configuration files, preserving run provenance, and separating the
simulation from analysis and cluster orchestration.

## What the software does

- **C++17 simulation:** periodic cubic lattices, local Metropolis updates,
  reproducible seeds, burn-in, and measurement output.
- **Parallel tempering:** a serial reference implementation and an optional MPI
  executable that exchanges replicas between neighboring temperatures.
- **Python workflow:** TOML validation, immutable run directories, saved source
  snapshots, file checksums, and explicit success or failure records.
- **Analysis:** observables, autocorrelation diagnostics, block jackknife and
  bootstrap uncertainty estimates, and finite-size fits.
- **Independent checks:** analytical energy identities, separate full-energy
  calculations, quadrature comparisons, and synthetic statistical fixtures.

## Model

Each lattice site holds a real-valued field rather than a binary spin. Neighboring
sites interact, while the local potential favors two field values:

```text
H = -J sum_<ij> phi_i phi_j + sum_i [(r/2) phi_i^2 + (u/4) phi_i^4]
```

The example couplings are `J=1`, `r=-1`, and `u=1`. Lattice size, temperature,
seed, and sweep counts are configured in TOML.

![Analytical on-site potential for r=-1 and u=1](docs/images/onsite-potential.png)

*The curve shows the analytical on-site term only. It is a model illustration,
not a simulation result or an estimate of the transition temperature.*

## Build and check

Requires a C++17 compiler, CMake 3.21+, Ninja, and Python 3.11+. Install the Python
analysis dependencies in your development environment, then build from the
repository root:

```sh
python -m pip install -r analysis/requirements.txt
cmake --preset local-debug
cmake --build --preset local-debug
ctest --test-dir build/local-debug --output-on-failure
```

The default build requires no MPI installation. The same CTest suite is configured
in [GitHub Actions](.github/workflows/build.yml) for Windows and Linux. A green
check means those software tests passed; it does not establish equilibration or
validate a critical exponent.

See [development setup](docs/DEVELOPMENT.md) for virtual environments, compiler
selection, macOS instructions, and a separate build for each platform.

### Try a small operational example

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
path is refused; choose a new run ID for another attempt. This is an operational
example, not a production physics calculation.

## Follow the code

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

## Scientific status

The simulation, analysis, and cluster workflows are implemented. Critical
exponents remain preliminary: passing software checks and completing runs do not
by themselves establish equilibrium or asymptotic scaling. Raw histories,
independent starts, block sizes, and fit sensitivity must still support any
scientific claim. The [validation plan](docs/VALIDATION.md) explains the
checks required before interpreting results.

Generated builds, raw datasets, working plots, and internal research planning
notes stay out of Git.

## Development and AI assistance

See [CONTRIBUTING.md](CONTRIBUTING.md) for checks and the distinction between
software plumbing and physics-bearing changes. Codex assisted with implementation
and debugging. I remain responsible for reviewing the code and validating the
scientific results.
