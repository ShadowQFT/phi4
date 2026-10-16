# Code layout

The project has three main parts:

- include/phi4/ contains the lattice, energy calculations, and sampler.
- src/main.cpp runs one chain and writes measurements.
- analysis/phi4_analysis/ loads saved runs, calculates observables and errors,
  and makes diagnostic plots.

scripts/run.py reads a TOML configuration and starts the C++ program.
scripts/analyze.py is the command-line entry point for analysis. The numerical
code does not depend on the scheduler or on a particular workstation.

Each process runs one chain. Larger scans can use separate processes for
independent temperatures, sizes, seeds, and initial states.

The [code walkthrough](CODE.md) follows these pieces in execution order.
