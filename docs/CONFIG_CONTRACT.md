# Input contract

scripts/run.py implements strict schema_version=1 for single-chain TOML.
The local smoke example is runnable. Campaign/Mill inputs remain drafts.
Python 3.11+ is required for tomllib.

| Section | Required meaning |
| --- | --- |
| model | clean_phi4_3d, J/r/u, periodic boundaries, L>=3, T>0 |
| sampling | start, seed, burn-in, adaptation, fixed-kernel settling, measurement sweeps, interval, update order |
| output | unique run ID, relative root, series policy |
| campaign | sizes, temperature strategy, independent seeds/starts, stage, budget |

All dimensional numeric values are finite. u>0, J>=0 (J=0 for validation); reject
unexpected model names, invalid starts, missing seeds, and unknown fields.
r stays negative in the agreed model. Sweeps and intervals are integers
with defined lower bounds. No adaptation during production.

Resolve each campaign into one explicit run per (L,T,start,seed,attempt).
Do not pass a campaign file directly to a single-chain simulator. Use the
manifest to preserve exact T values and counts; floating-point display
rounding in folder names must not determine parameter identity.
Assign unique seeds and preserve the allocation scheme. Never use
Python's process-randomized hash() for seed generation.

Production T grids and sweep counts remain unspecified until pilots locate
Tc and estimate autocorrelation/runtime. Proposed couplings r=-1,u=1 are
confirmed changeable defaults as of October 15. User has withheld runs;
parameter confirmation does not authorize a scientific pilot.

Site settings in `configs/mill.example.toml` are separate from physics.
Put private resolved settings in ignored `configs/mill.local.toml`.
Scheduler placeholders must fail preflight; they must never silently use
a fabricated account, queue, module, or resource allocation.
