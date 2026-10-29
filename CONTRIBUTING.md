# Development notes

This is a research and course project. Keep changes small enough to review
alongside the corresponding numerical definitions.

## Before changing code

Read the [project overview](README.md), [model definitions](docs/PHYSICS.md),
and the relevant page in the [documentation map](docs/README.md). Check the working tree before editing so
existing local changes remain intact.

Use C++17 and the standard library for the simulator. Keep workflow and analysis
code in Python. Do not add a dependency or portability layer without a concrete
need. Machine-specific settings belong in ignored local configuration files.

## Validation

Build a Debug configuration and run the nonempty CTest suite:

```sh
cmake --preset local-debug
cmake --build --preset local-debug
ctest --test-dir build/local-debug --output-on-failure
python scripts/run.py configs/local/smoke.example.toml --validate-only
```

Tests use small disposable fixtures. They do not authorize a research campaign
or a cluster submission. The committed source and any uncommitted local edits
must be distinguished when reporting test results.

For a change to the Hamiltonian, neighbor layout, update rule, acceptance test,
observables, or statistics, add an independent expected value or identity from
[VALIDATION.md](docs/VALIDATION.md). Comparing two implementations that share the
same potentially incorrect helper is insufficient.

Record exactly what ran, failures, and any checks not run. A successful process
exit or passing CI does not validate equilibration or a physical conclusion.

## Outputs and provenance

Keep seeds, resolved settings, source identity, raw histories, and failed-run
records. Do not overwrite an existing run or silently merge incompatible data.
Generated builds, raw datasets, credentials, and local overrides do not belong in
commits. Do not submit, cancel, or modify cluster jobs as a side effect of a code
or documentation change.

## Pull requests

Explain the behavior changed and the evidence used to check it. Include any
remaining numerical or scientific limitation. Preserve readable formulas and
use names that describe what the code calculates.
