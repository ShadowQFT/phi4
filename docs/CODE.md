# Code walkthrough

These are the files I start with when following a run:

1. [sampler.hpp](../include/phi4/sampler.hpp) holds the current fields and
   random-number generator. initialize() sets the starting configuration;
   sweep() attempts one update at every site.
2. [model.hpp](../include/phi4/model.hpp) sets up periodic neighbors and
   calculates the on-site potential, full energy, and local energy changes.
3. [main.cpp](../src/main.cpp) runs burn-in and measurement sweeps and saves
   energy per site and magnetization.
4. [statistics.py](../analysis/phi4_analysis/statistics.py) calculates the
   observables and their block-based error estimates.

## One update

The sampler draws a displacement uniformly from [-step_size, step_size) and
adds it to one field. It calculates the energy change from that site and its
six neighbors. Downhill moves are accepted; uphill moves are accepted when

```cpp
random_number < std::exp(-delta_h / temperature)
```

After every site has had one attempted update, one sweep is complete.

## Burn-in and measurements

Step-size adjustment is optional and happens only during burn-in. Every 50
adaptation sweeps, acceptance below 40% shrinks the step by 10%; above 60%
increases it by 10%. The step is then frozen for settling and measurement.

Saved measurements contain e=H/N and m=sum(phi)/N, along with their sweep
numbers. Scans also save the fields at the beginning and end of each
temperature. Full fields are not saved for each measurement, so spatial
correlation histories cannot be reconstructed. Restart checkpoints are not
implemented yet.

## Following a parallel-tempering run

[pt.cpp](../src/pt.cpp) starts with the distinction between a physical replica
and a temperature slot. A replica keeps its fields and random-number stream;
only its temperature label moves. The serial reference owns every replica in
one process; the MPI build owns one replica per rank.

`read_run_settings()` names and checks the resolved command-line inputs.
`TemperingRun::run()` then performs adaptation, settling and measurement in
that order. To follow one sweep, start with `run_phase()`:

1. `advance_local_replicas()` sets the current temperature and proposal width
   and calls the ordinary Metropolis sweep. Only adaptation adjusts the width.
2. `gather_current_measurements()` collects total energy, magnetization and
   proposal widths. Named measurement fields keep MPI array offsets out of
   the sampling and exchange calculations.
3. On exchange sweeps, rank zero calls `attempt_adjacent_exchanges()`. It
   alternates disjoint neighboring pairs and uses the product-canonical ratio
   with total H, then broadcasts the updated temperature labels.
4. On saved sweeps, `write_samples()` routes the gathered configuration values
   into temperature order using the labels after the exchange.

Adaptation has no exchanges. Its final adjusted width is frozen separately
for each temperature, while its history logs the width actually used for the
last sweep. When production starts, swap counters and round-trip tracking
begin, while the exchange RNG and odd/even schedule continue from settling.
`write_summary()` records the resulting counters and frozen widths. The MPI
packing and collective calls are contained in `WorkerGroup`.

## Analysis

The analysis first calculates e, m, absolute m, m squared, m fourth, and e
squared for each sample. It averages these moments before calculating B4,
susceptibility, and heat capacity.

For error estimates, consecutive samples are grouped into blocks. The
jackknife removes one block at a time; the bootstrap draws whole blocks with
replacement. This keeps related moments together and accounts for short-range
correlations when blocks are long enough. Histories, split halves, and error
versus block-length plots help check whether the averages are stable.

scripts/run.py handles configuration and saved-run information.
scripts/analyze.py loads completed runs and connects the analysis functions.
Neither analysis nor a successful process exit establishes equilibration by
itself.

## Temperature scans

With a scan configuration, main() calls run_temperature_scan(). It changes the
sampler temperature and repeats run_chain() without reinitializing the lattice
or RNG. Adaptation and settling repeat at each temperature. scripts/plot_scan.py
checks the saved scan and plots each temperature using the existing statistics
functions. See [Temperature scans](SCANS.md) for the full flow and commands.

For a size list, scripts/run.py runs that C++ scan once per size. Each process
starts a new lattice, RNG, and step size. Data go to L<size>/temperature_<index>,
and plot_scan.py groups the temperature points into one curve for each L.
