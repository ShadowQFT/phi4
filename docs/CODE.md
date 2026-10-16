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
numbers. The full field configuration is not saved, so spatial correlations
cannot be reconstructed from these files. Restart checkpoints are not
implemented yet.

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
