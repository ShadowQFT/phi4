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

## Independent-chain campaign analysis

For the Mill campaigns of independent single-temperature chains:

```text
python scripts/analyze_campaign.py data/raw/CAMPAIGN --output results/figures/NEW_OUTPUT --block-size 5000 --fit --degree 2 --replicates 2000
```

Each chain must have at least 20 complete blocks. Raw files and metadata are
validated by the existing hash-checking loader. Never concatenate time series
across chain boundaries. Compatible chains are pooled with retained-sample
weights, after reporting their separate estimates and pairwise disagreements.
The bootstrap resamples complete joint-moment blocks within each chain, retaining
chain identities and original weights. It recomputes moments, Binder ratio,
susceptibility, common crossing, slopes and exponents together.

The autocorrelation estimator uses an FFT autocovariance, positive lag pairs and
a self-consistent window. It is not an implementation of the full Gamma method.
Stationarity is an assumption. Energy, |m|, m2 and m4 are checked separately.
Blocking warnings, split-history discrepancies and between-chain disagreements
are explicit. They are diagnostic flags, not automatic equilibration decisions.

`--tmin`, `--tmax`, `--lmin`, `--degree` and `--crossing-sizes L1 L2` support
sensitivity checks. The default Tc is the largest-size pair crossing inside the
common measured temperature interval. This is a finite-volume quantity, without
correction-to-scaling extrapolation. No literature exponents enter the fits.
Require at least degree+2 temperatures per size and at least three sizes.
Use four or more sizes to make the Lmin comparison meaningful.

Local moment curves use inverse jackknife-error weights. Power fits use working
relative-error weights (the Binder-slope working error is scaled from the local
Binder error and window width); these are not a propagated slope covariance.
Parameter intervals and covariance come from the complete joint bootstrap,
not from the power-fit normal equations. Power-fit chi-square is therefore a
working residual statistic, not a calibrated goodness-of-fit probability.
Local polynomial chi-square uses the actual jackknife errors.

Local polynomial chi-square/dof above 5, more than 5% failed bootstrap fits, or
unresolved sampling flags withhold the candidate exponents from reporting.
These are explicit conservative screening rules, not proof that passing fits
have reached the asymptotic regime. Candidate fits are preserved in analysis.json
for inspection, clearly marked with their scientific status. Bootstrap failure
reasons and frequencies are saved; intervals are conditional on successful fits.

Outputs include summary.csv, benchmark.csv, analysis.json, observable plots,
production-history plots, and (when a candidate exists) scaling/collapse figures.
Rejected fits' figures are labeled diagnostic. Analysis JSON records raw input
hashes, analysis-code hashes, NumPy version, options, seeds and covariance order.
Elapsed benchmark time includes burn-in, source archival and output validation.
Use fresh output directories to preserve earlier analyses.


### Production fit and stationarity review

`analyze_campaign.py --save-blocks` also exports per-chain joint moment blocks
and their ensemble mapping. This is a derived cache; preserve the accompanying
analysis.json raw-file hashes. Adjacent blocks can be merged within each chain
for a longer-block check; the implementation is tested against direct raw
analysis and rejects fewer than 20 merged blocks per chain.

```text
python scripts/scaling_sensitivity.py results/figures/production-20261028 --output results/figures/NEW_SENSITIVITY --replicates 1000
python scripts/review_split_chains.py results/figures/production-20261028 --permutations 200000 --output results/figures/NEW_SPLIT_REVIEW.json
```

The sensitivity script retains all declared cases: quadratic full/central/narrow
windows, a linear alternative, smallest-size exclusions, alternative crossing
pairs and doubled blocks. It does not override sampling warnings from the main
analysis. A case passing local-fit screening is not proof of equilibrium.

The exploratory split review permutes whole joint-moment blocks within each
chain, recomputes the two halves' nonlinear observables and applies Holm's
family-wise correction across every chain/observable comparison. This requires
approximately exchangeable blocks under the null; failure to reject does not
prove stationarity. The review retains all chains and original flags. Production
used 200000 permutations; a single flagged susceptibility comparison remained.
Do not discard that chain solely to improve the exponent comparison.


### Normalized Binder plotting convention

Campaign observable and collapse plots now use B_N=(3-B4)/2, with ordered and
centered-Gaussian high-temperature limits 1 and 0. The errors are sigma_B4/2.
Values are not clipped, nor scaled using sampled minima/maxima. The slope plot
uses |d B_N/dT|=|d B4/dT|/2. Raw B4 remains stored and used internally for the
same crossing and exponent fits; an affine transformation preserves crossings
and changes slope amplitudes without changing power-law exponents.

The existing production observable figure was regenerated from its saved summary
as results/figures/production-20261028-normalized/observables.png and PDF. The
plot_info.json records the summary checksum, plotting-code checksum and formulas.
Its 35 ensemble rows were checked against the exact Binder/error transformation;
all eight existing statistics tests pass, including the limiting-value checks.
