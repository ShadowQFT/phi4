# Planned data contract

Local single-chain raw outputs are implemented, including a source.zip
archive. Per-chain diagnostic analysis is implemented; manifests, crossings
and fits remain planned. No scientific data
exists; workflow fixtures live under build/.

```text
data/raw/<campaign>/<run_id>/
  config.resolved.toml
  metadata.json
  measurements.csv
  equilibration.csv
  stdout.log
  stderr.log
  SUCCESS or FAIL
data/derived/<campaign>/<analysis_id>/
  provenance.json
  analysis_source.zip
  summary.json
  observables.csv
  blocking.csv
  histories.png / histories.pdf
  blocking.png / blocking.pdf
  SUCCESS or FAIL
```

crossings.csv and fits.json are future scaling outputs. See ANALYSIS.md for
the implemented analysis semantics, recorded cuts and incomplete-block tails.

Each production sample row: `sample_index,sweep,energy_per_site,magnetization`.
Samples are equally spaced at the recorded interval; do not mix burn-in and
measurement phases. Include a header, full double precision, finite values,
and strictly increasing counters. Analysis computes powers/absolute values
from m. `equilibration.csv` additionally records acceptance and step size.

Metadata records schema version, complete model/sampling inputs, N, units,
initialization, seed/RNG, source commit and dirty state or snapshot hash,
compiler/version/flags, binary hash, platform, timestamps, sample count,
adaptation stop/final step size, Slurm job/array IDs when present, status,
and any numerical failures. UTC timestamps avoid ambiguous local times.
Explicitly identify partially written or interrupted outputs.

SUCCESS means exit zero and validated expected outputs/counts, not
equilibration or scientifically sufficient statistics. FAIL and incomplete
folders are excluded from normal aggregation but retained for diagnosis.
Retry with a new attempt ID. Raw folders are immutable after completion and the launcher refuses existing
folders. Disposable tests deliberately corrupt their own build fixtures.

Manifest rows: `run_id,L,T,start,seed,config_path,config_sha256,source_id`.
Manifest identity and array mapping must be archived with the campaign.
Never combine differing couplings, definitions, or protocols without an
explicit scientifically justified analysis plan.

Derived provenance records input hashes, exclusions/burn-in selections,
reduction code identity, package environment, block/bootstrap settings,
Tc treatment, fits, and command. Every figure/table points to an analysis
ID and is reproducible from raw data. Do not commit large/private raw runs
into Git; keep small public synthetic fixtures under tests/fixtures instead.
