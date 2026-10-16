# Design notes

- **C++ for sampling, Python for analysis.** The update loop runs many times,
  while Python is convenient for statistics and plotting.
- **One chain per process.** Independent chains are a simple way to run several
  temperatures, sizes, or starts without parallel updates to the same lattice.
- **L>=3.** This avoids the special self-bond and repeated-neighbor cases of
  very small periodic lattices.
- **Standard random distributions.** The proposal and acceptance draws use
  ordinary C++ distributions so the update is easy to read.
- **Fixed proposals during measurement.** Step-size tuning ends before the
  settling and measurement phases.
- **Full energy at measurement.** Recalculating H is straightforward and avoids
  accumulating drift from repeated local changes.
- **Raw histories.** Saving e and m keeps the data needed for blocking and
  resampling. Saving only run averages would lose that information.
- **Joint block resampling.** Related moments stay together when recomputing
  nonlinear quantities such as B4.

The Hamiltonian and observable conventions are in [PHYSICS.md](PHYSICS.md).
