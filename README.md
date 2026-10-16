# 3D phi4 model

My project for PHYSICS 5409 Computational Physics Lab. I'm using Metropolis
Monte Carlo to study the three-dimensional lattice phi4 model, with the goal
of estimating its critical temperature and critical exponents through
finite-size scaling.

The simulation is written in C++17, with Python scripts for analysis and plots.
The [proposal](proposal/phi4_proposal.tex) describes the project.

## Current progress

The sampler, configuration files, and basic analysis scripts are working.
The tests cover energy calculations, Metropolis acceptance, the noninteracting
field distribution, and statistical error estimates. Temperature scans and
finite-size-scaling fits are still to come; there are no measured critical
exponents yet.

The model uses periodic boundaries and the Hamiltonian

```text
H = -J sum_<ij> phi_i phi_j + sum_i [(r/2) phi_i^2 + (u/4) phi_i^4]
```

The current defaults are J=1, r=-1, u=1. The couplings, temperature, lattice
size, seed, and sweep counts are set in TOML files under configs/.

## Build and test

Requires CMake 3.21+, Ninja, a C++17 compiler, and Python 3.11+. The Python
analysis uses NumPy and Matplotlib; versions are listed in
[analysis/requirements.txt](analysis/requirements.txt).

```powershell
cmake --preset local-debug
cmake --build --preset local-debug
ctest --test-dir build/local-debug --output-on-failure
```

To check the example configuration without running a simulation:

```powershell
python scripts/run.py configs/local/smoke.example.toml --validate-only
```

See [analysis notes](docs/ANALYSIS.md) for working with saved runs.

## Code and notes

- [Code walkthrough](docs/CODE.md): where the main calculations happen.
- [Model definitions](docs/PHYSICS.md): Hamiltonian and observables.
- [Monte Carlo](docs/MONTE_CARLO.md): updates, burn-in, and measurements.
- [Statistics](docs/STATISTICS.md): blocking and uncertainty estimates.
- [Project status](docs/STATUS.md): what works and what is next.

The sampler is in include/phi4/sampler.hpp, the energy calculations are in
include/phi4/model.hpp, and src/main.cpp runs the simulation. Python analysis
code is in analysis/phi4_analysis/. Generated files and local data are kept
out of Git.

## AI assistance

I use ChatGPT/Codex for help with implementation, debugging, and code explanations.
