# Per-chain analysis and diagnostics

The analysis command reads completed runs. It does not start the simulator,
choose a burn-in cut, or combine chains automatically.

## Inputs and command

From the repository root, after actual run directories exist:

```powershell
python scripts/analyze.py data/raw/CAMPAIGN/RUN `
  --output data/derived/CAMPAIGN/ANALYSIS `
  --block-size 32 --bootstrap-seed 20261015
```

CAMPAIGN/RUN/ANALYSIS are placeholders. The block length 32 is an example,
not a recommendation for an unmeasured chain. Specify a new output directory
for every analysis. NumPy is required; plots also require Matplotlib.
Use `--no-plots` for tables only. Default bootstrap replicates: 2000.

To compare independent starts, supply multiple raw run directories before
the options. They must have identical model parameters, L, T, update order
and measurement interval, and distinct seeds. Each chain keeps its own
estimates and resampling stream. No averaged estimate is created.

The loader requires SUCCESS and successful schema-1 metadata; it verifies
all recorded raw output hashes, resolved-config consistency, site/sample
counts, sweep counters, finite measurements, burn-in phases and frozen
settling steps. Failed/incomplete/changed raw runs are rejected. Input files
are read only. These hashes detect changes relative to metadata; they do
not authenticate files if both the data and metadata have been rewritten.

## Interpreting the outputs

| File | Meaning |
| --- | --- |
| summary.json | Per-chain observables, block/bootstrap results and split histories |
| observables.csv | Retained-prefix estimates, jackknife standard errors and 95% bootstrap percentile intervals |
| blocking.csv | Standard errors versus powers-of-two block lengths for all observables |
| histories.png / .pdf | e, absolute m and m squared, including excluded burn-in |
| blocking.png / .pdf | Block-length sensitivity for e, absolute m, m squared and B4 |
| provenance.json | Inputs/hashes, source identity, package versions, settings and output hashes |
| analysis_source.zip | Reconstructable analysis package source |
| SUCCESS or FAIL | Operational analysis status; not an equilibrium verdict |

The selected block size is in retained samples. Multiply by the measurement
interval to obtain sweeps. Error estimates require at least 20 complete
blocks, but this count alone does not establish reliability. Inspect a
plateau across block lengths and the behavior of e, absolute m, m squared,
m fourth and B4. The full numerical blocking table includes m fourth.

Block estimators discard only an incomplete final block and explicitly
report its sample count. In summary.json, `all_selected_samples` uses every
selected sample; `block_jackknife.estimate` and the CSV estimates use only
the complete-block prefix. Differences between those estimates must not be
mistaken for a change of estimator definition. Nonlinear observables are
recomputed inside every joint resample. Undefined replicates abort the
analysis rather than being silently removed.

Split-history summaries compare chronological halves. If a half has fewer
than 20 blocks at the selected length, its uncertainty is explicitly marked
unavailable. The raw second-minus-first differences are descriptive and are
not an independence-based significance test. Adjacent halves can correlate.

Burn-in from the simulator is already excluded from measurements.csv.
`--discard-samples K` makes an additional explicit production cut and records
K and the first selected sweep. The default is zero. Inspect histories and
justify any additional cut; the command never searches for one that improves
agreement. A new analysis directory preserves each sensitivity check.

Neither stable-looking traces nor small intervals prove equilibrium. This
tool does not estimate integrated autocorrelation times, effective sample
counts, Tc or critical exponents. There are no scientific results yet.

## Readability and development

Python code follows ruff.toml; C++ code follows .clang-format. Comments explain
physical factors and workflow decisions rather than narrating each line.

```powershell
ruff check analysis scripts tests/python
ruff format --check analysis scripts tests/python
ctest --test-dir build/local-debug --output-on-failure
```

The analysis suite builds explicitly synthetic raw fixtures, verifies failure
and corruption paths, checks split/block bookkeeping, exercises the CLI and
exports PNG/PDF previews. These fixtures are labelled synthetic and retained
under build/analysis-fixtures; they are not Monte Carlo pilot data.
